"""
License Client - Kết nối với License Server để xác thực
Tích hợp vào Trading Bot

PRODUCTION USAGE:
    from license_client import get_license_client
    
    # Initialize with your server URL
    client = get_license_client(
        server_url="https://license.yourdomain.com/api"  # CHANGE THIS
    )
    
    # Register new user
    success, data = client.register("username", "password", "email@example.com")
    
    # Login existing user
    success, data = client.login("username", "password")
    
    # Activate license on this device
    success, data = client.activate(data['license']['license_key'])
    
    # Check if license is valid
    if client.is_valid:
        print("License OK!")
"""
import os
import json
import hashlib
import platform
import uuid
import time
import threading
import requests
import ssl
import certifi
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


# =============================================================================
# CONFIGURATION - CHANGE THIS FOR PRODUCTION
# =============================================================================

# Default License Server URL
# Development: http://localhost:8000/api
# Production:  https://license.yourdomain.com/api
# Ngrok:       https://xxx.ngrok-free.dev/api
DEFAULT_SERVER_URL = "http://localhost:8000/api"

# Timeouts (seconds)
REQUEST_TIMEOUT = 15
HEARTBEAT_INTERVAL = 60

# Offline grace period (hours)
OFFLINE_GRACE_HOURS = 72


class LicenseClient:
    """
    License Client để kết nối với License Server qua Internet
    
    Features:
    - Hardware ID generation (chống chia sẻ license)
    - License activation/validation
    - Heartbeat để duy trì kết nối
    - Offline grace period (cho phép offline tạm thời)
    - Caching để giảm requests
    - HTTPS/SSL support cho production
    - Retry logic cho unstable connections
    - JWT token management
    """
    
    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        cache_dir: Optional[str] = None,
        heartbeat_interval: int = HEARTBEAT_INTERVAL,
        offline_grace_hours: int = OFFLINE_GRACE_HOURS,
        on_license_invalid: Optional[Callable] = None,
        on_license_expiring: Optional[Callable[[int], None]] = None,
        on_license_changed: Optional[Callable[[Dict], None]] = None,
        verify_ssl: bool = True,
    ):
        """
        Initialize License Client
        
        Args:
            server_url: URL của License Server API (ví dụ: https://license.yourdomain.com/api)
            cache_dir: Thư mục cache license (default: ~/.trading_bot/license)
            heartbeat_interval: Khoảng cách giữa các heartbeat (seconds)
            offline_grace_hours: Thời gian cho phép offline (hours)
            on_license_invalid: Callback khi license không hợp lệ
            on_license_expiring: Callback khi license sắp hết hạn (nhận số ngày còn lại)
            on_license_changed: Callback khi license thay đổi (admin cập nhật)
            verify_ssl: Kiểm tra SSL certificate (True cho production)
        """
        self.server_url = server_url.rstrip('/')
        self.heartbeat_interval = heartbeat_interval
        self.offline_grace_hours = offline_grace_hours
        self.on_license_invalid = on_license_invalid
        self.on_license_expiring = on_license_expiring
        self.on_license_changed = on_license_changed
        self.verify_ssl = verify_ssl
        
        # Cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / '.trading_bot' / 'license'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self._license_key: Optional[str] = None
        self._license_data: Optional[Dict] = None
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._cached_username: Optional[str] = None
        self._hardware_id: str = self._generate_hardware_id()
        self._device_name: str = platform.node()
        self._os_info: str = f"{platform.system()} {platform.release()}"
        self._last_validation: Optional[datetime] = None
        self._is_online: bool = True
        self._connection_error_count: int = 0
        
        # Setup HTTP session with retries
        self._session = self._create_session()
        
        # Heartbeat thread
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()
        
        # Notification watcher thread
        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_watcher = threading.Event()
        self._last_notification_check: Optional[str] = None
        
        # Load cached license
        self._load_cached_license()
    
    # ============ HARDWARE ID ============
    
    def _create_session(self) -> requests.Session:
        """
        Tạo HTTP session với retry logic cho unstable connections
        """
        session = requests.Session()
        
        # Retry strategy: 3 retries with backoff
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # SSL verification
        if self.verify_ssl:
            session.verify = certifi.where()
        else:
            session.verify = False
        
        return session
    
    def _generate_hardware_id(self) -> str:
        """
        Tạo Hardware ID unique cho máy tính này
        Sử dụng: MAC address + Machine ID + Username
        """
        try:
            # Get MAC address
            mac = uuid.getnode()
            mac_str = ':'.join(('%012x' % mac)[i:i+2] for i in range(0, 12, 2))
            
            # Get machine-specific info
            machine_info = f"{platform.node()}:{platform.machine()}:{platform.processor()}"
            
            # Get username
            username = os.getlogin() if hasattr(os, 'getlogin') else os.environ.get('USERNAME', 'unknown')
            
            # Combine and hash
            combined = f"{mac_str}|{machine_info}|{username}"
            hardware_id = hashlib.sha256(combined.encode()).hexdigest()
            
            return hardware_id
            
        except Exception as e:
            print(f"⚠️ Error generating hardware ID: {e}")
            # Fallback: use random UUID stored in file
            fallback_file = self.cache_dir / '.hardware_id'
            if fallback_file.exists():
                return fallback_file.read_text().strip()
            else:
                fallback_id = str(uuid.uuid4())
                fallback_file.write_text(fallback_id)
                return fallback_id
    
    @property
    def hardware_id(self) -> str:
        return self._hardware_id
    
    # ============ CACHING ============
    
    def _cache_file(self) -> Path:
        return self.cache_dir / 'license_cache.json'
    
    def _load_cached_license(self):
        """Load license từ cache"""
        try:
            cache_file = self._cache_file()
            print(f"[License Cache] Loading from: {cache_file}")
            if cache_file.exists():
                data = json.loads(cache_file.read_text())
                self._license_key = data.get('license_key')
                self._license_data = data.get('license_data')
                self._access_token = data.get('access_token')
                self._refresh_token = data.get('refresh_token')
                self._cached_username = data.get('username')
                self._last_validation = datetime.fromisoformat(data['last_validation']) if data.get('last_validation') else None
                print(f"✅ Loaded cached license: {self._license_key[:16] if self._license_key else 'None'}...")
                print(f"✅ Loaded cached token: {self._access_token[:30] if self._access_token else 'None'}...")
            else:
                print(f"[License Cache] Cache file not found: {cache_file}")
        except Exception as e:
            print(f"⚠️ Error loading cached license: {e}")
    
    def _save_cached_license(self, username: str = None):
        """Save license vào cache"""
        try:
            cache_file = self._cache_file()
            print(f"[License Cache] Saving to: {cache_file}")
            data = {
                'license_key': self._license_key,
                'license_data': self._license_data,
                'access_token': self._access_token,
                'refresh_token': self._refresh_token,
                'last_validation': self._last_validation.isoformat() if self._last_validation else None,
                'hardware_id': self._hardware_id,
                'username': username or self._cached_username,
            }
            # Lưu username nếu có
            if username:
                self._cached_username = username
            cache_file.write_text(json.dumps(data, indent=2, default=str))
            print(f"[License Cache] Saved successfully! Token: {self._access_token[:30] if self._access_token else 'None'}...")
        except Exception as e:
            print(f"⚠️ Error saving cached license: {e}")
    
    def _clear_cache(self):
        """Clear license cache"""
        try:
            cache_file = self._cache_file()
            if cache_file.exists():
                cache_file.unlink()
            self._license_key = None
            self._license_data = None
            self._access_token = None
            self._refresh_token = None
            self._last_validation = None
            self._cached_username = None
        except Exception as e:
            print(f"⚠️ Error clearing cache: {e}")
    
    def clear_cache(self):
        """Public method to clear cache (dùng khi logout)"""
        self.stop_heartbeat()
        self.stop_notification_watcher()
        self._clear_cache()
        print("🔓 License cache cleared")
    
    def get_cached_username(self) -> Optional[str]:
        """Lấy username đã lưu trong cache"""
        try:
            cache_file = self._cache_file()
            if cache_file.exists():
                data = json.loads(cache_file.read_text())
                return data.get('username')
        except Exception as e:
            print(f"⚠️ Error getting cached username: {e}")
        return self._cached_username if hasattr(self, '_cached_username') else None

    # ============ API CALLS ============
    
    def _api_request(
        self,
        endpoint: str,
        method: str = 'POST',
        data: Optional[Dict] = None,
        timeout: int = REQUEST_TIMEOUT,
        use_auth: bool = False,
    ) -> tuple[bool, Dict]:
        """
        Make API request to license server (qua Internet)
        
        Args:
            endpoint: API endpoint (e.g., 'login/', 'license/activate/')
            method: HTTP method ('GET' or 'POST')
            data: Request body data
            timeout: Request timeout in seconds
            use_auth: Include JWT token in header
            
        Returns:
            (success, response_data)
        """
        url = f"{self.server_url}/{endpoint.lstrip('/')}"
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': f'TradingBot/{self._get_app_version()}',
            'ngrok-skip-browser-warning': 'true',  # Bypass ngrok warning page
        }
        
        if use_auth and self._access_token:
            headers['Authorization'] = f'Bearer {self._access_token}'
        
        try:
            method_upper = method.upper()
            if method_upper == 'GET':
                response = self._session.get(url, headers=headers, timeout=timeout)
            elif method_upper == 'PUT':
                response = self._session.put(url, json=data, headers=headers, timeout=timeout)
            elif method_upper == 'PATCH':
                response = self._session.patch(url, json=data, headers=headers, timeout=timeout)
            elif method_upper == 'DELETE':
                response = self._session.delete(url, headers=headers, timeout=timeout)
            else:
                response = self._session.post(url, json=data, headers=headers, timeout=timeout)
            
            self._is_online = True
            self._connection_error_count = 0
            
            if response.status_code == 200 or response.status_code == 201:
                return True, response.json()
            elif response.status_code == 401 and use_auth:
                # Token expired, try to refresh
                if self._refresh_access_token():
                    # Retry request with new token
                    headers['Authorization'] = f'Bearer {self._access_token}'
                    if method_upper == 'GET':
                        response = self._session.get(url, headers=headers, timeout=timeout)
                    elif method_upper == 'PUT':
                        response = self._session.put(url, json=data, headers=headers, timeout=timeout)
                    elif method_upper == 'PATCH':
                        response = self._session.patch(url, json=data, headers=headers, timeout=timeout)
                    elif method_upper == 'DELETE':
                        response = self._session.delete(url, headers=headers, timeout=timeout)
                    else:
                        response = self._session.post(url, json=data, headers=headers, timeout=timeout)
                    
                    if response.status_code == 200 or response.status_code == 201:
                        return True, response.json()
                
                return False, {'error': 'auth_expired', 'message': 'Please login again'}
            else:
                try:
                    return False, response.json()
                except:
                    return False, {'error': 'server_error', 'message': f'Status {response.status_code}'}
                
        except requests.exceptions.SSLError as e:
            return False, {'error': 'ssl_error', 'message': f'SSL certificate error: {str(e)}'}
        except requests.exceptions.ConnectionError:
            self._is_online = False
            self._connection_error_count += 1
            return False, {'error': 'connection_error', 'message': 'Cannot connect to license server. Check your internet connection.'}
        except requests.exceptions.Timeout:
            self._is_online = False
            self._connection_error_count += 1
            return False, {'error': 'timeout', 'message': 'License server timeout. The server may be busy or your connection is slow.'}
        except Exception as e:
            return False, {'error': 'unknown', 'message': str(e)}
    
    def _refresh_access_token(self) -> bool:
        """Refresh access token using refresh token"""
        if not self._refresh_token:
            return False
        
        try:
            url = f"{self.server_url}/auth/token/refresh/"
            headers = {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true',  # Bypass ngrok warning page
            }
            response = self._session.post(
                url,
                json={'refresh': self._refresh_token},
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                self._access_token = data.get('access')
                self._save_cached_license()
                return True
        except:
            pass
        
        return False
    
    # ============ PUBLIC METHODS ============
    
    def get_valid_token(self) -> Optional[str]:
        """
        Lấy access token hợp lệ (tự động refresh nếu đã hết hạn)
        Sử dụng method này thay vì truy cập trực tiếp _access_token
        
        Returns:
            str: Valid access token hoặc None nếu không có
        """
        # Nếu chưa có token trong memory, thử load từ cache
        if not self._access_token:
            self._load_cached_license()
        
        if not self._access_token:
            return None
        
        # Kiểm tra token còn hạn không bằng cách decode JWT
        try:
            import base64
            import json
            import time
            
            # JWT format: header.payload.signature
            parts = self._access_token.split('.')
            if len(parts) != 3:
                return self._access_token  # Không phải JWT chuẩn, trả về nguyên
            
            # Decode payload (phần 2)
            payload_b64 = parts[1]
            # Thêm padding nếu cần
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += '=' * padding
            
            payload_json = base64.b64decode(payload_b64)
            payload = json.loads(payload_json)
            
            exp = payload.get('exp', 0)
            current_time = time.time()
            
            # Nếu token sắp hết hạn trong 60s, refresh trước
            if exp - current_time < 60:
                print(f"[Token] Token sắp hết hạn ({exp - current_time:.0f}s còn lại), đang refresh...")
                if self._refresh_access_token():
                    print("[Token] Refresh thành công!")
                    return self._access_token
                else:
                    print("[Token] Refresh thất bại!")
                    return None
            
            return self._access_token
            
        except Exception as e:
            print(f"[Token] Lỗi kiểm tra token: {e}")
            # Nếu không decode được, thử refresh
            if self._refresh_access_token():
                return self._access_token
            return self._access_token  # Trả về token cũ, để server reject nếu cần
    
    def register(
        self,
        username: str,
        password: str,
        email: str,
        first_name: str = '',
        last_name: str = '',
        phone: str = '',
        language: str = 'vi'
    ) -> tuple[bool, Dict]:
        """
        Đăng ký tài khoản mới
        
        Returns:
            (success, response_data)
        """
        success, data = self._api_request('auth/register/', data={
            'username': username,
            'password': password,
            'password_confirm': password,
            'email': email,
            'first_name': first_name,
            'last_name': last_name,
            'phone': phone,
            'language': language,
        })
        
        if success and data.get('license'):
            self._license_key = data['license']['license_key']
            self._license_data = data['license']
            self._last_validation = datetime.now()
            self._save_cached_license()
        
        return success, data
    
    def activate_by_code(self, username: str, activation_code: str) -> tuple[bool, Dict]:
        """
        Kích hoạt tài khoản bằng mã kích hoạt 6 ký tự
        
        Args:
            username: Tên đăng nhập của user
            activation_code: Mã kích hoạt 6 ký tự
            
        Returns:
            (success, response_data)
        """
        success, data = self._api_request('auth/activate-code/', data={
            'username': username,
            'code': activation_code.upper().strip(),
        })
        
        if success and data.get('license'):
            self._license_key = data['license']['license_key']
            self._license_data = data['license']
            self._last_validation = datetime.now()
            self._save_cached_license()
        
        return success, data
    
    def login(self, username: str, password: str) -> tuple[bool, Dict]:
        """
        Đăng nhập và lấy license info
        
        Returns:
            (success, response_data)
        """
        success, data = self._api_request('auth/login/', data={
            'username': username,
            'password': password,
        })
        
        if success:
            # Save JWT tokens (luôn lưu nếu có, kể cả license expired)
            self._access_token = data.get('access')
            self._refresh_token = data.get('refresh')
            
            if data.get('license'):
                self._license_key = data['license'].get('license_key')
                self._license_data = data['license']
            
            self._last_validation = datetime.now()
            self._cached_username = username  # Lưu username
            self._save_cached_license(username)
            
            # Log license status
            license_status = data.get('license_status', 'active')
            print(f"[License] Login success, license_status: {license_status}")
        
        return success, data
    
    def activate(self, license_key: str) -> tuple[bool, Dict]:
        """
        Activate license với hardware ID
        
        Args:
            license_key: License key để activate
            
        Returns:
            (success, response_data)
        """
        success, data = self._api_request('license/activate/', data={
            'license_key': license_key,
            'hardware_id': self._hardware_id,
            'device_name': self._device_name,
            'os_info': self._os_info,
            'app_version': self._get_app_version(),
        })
        
        if success:
            self._license_key = license_key
            self._license_data = data.get('license')
            self._last_validation = datetime.now()
            self._save_cached_license()
            
            # Start heartbeat
            self.start_heartbeat()
        
        return success, data
    
    def validate(self, license_key: Optional[str] = None) -> tuple[bool, Dict]:
        """
        Validate license (không activate)
        
        Returns:
            (success, response_data)
        """
        key = license_key or self._license_key
        if not key:
            return False, {'error': 'no_license', 'message': 'No license key provided'}
        
        success, data = self._api_request('license/validate/', data={
            'license_key': key,
            'hardware_id': self._hardware_id,
        })
        
        if success:
            self._license_data = data.get('license')
            self._last_validation = datetime.now()
            self._save_cached_license()
            
            # Check expiring soon
            days = data.get('license', {}).get('days_remaining', 0)
            if days <= 7 and self.on_license_expiring:
                self.on_license_expiring(days)
        
        return success, data
    
    def get_profile(self) -> tuple[bool, Dict]:
        """
        Lấy thông tin profile của user hiện tại
        
        Returns:
            (success, response_data)
        """
        success, data = self._api_request('auth/profile/', method='GET', use_auth=True)
        return success, data
    
    def update_profile(self, first_name: str = None, last_name: str = None, 
                      email: str = None, phone: str = None) -> tuple[bool, Dict]:
        """
        Cập nhật thông tin profile
        
        Returns:
            (success, response_data)
        """
        update_data = {}
        if first_name is not None:
            update_data['first_name'] = first_name
        if last_name is not None:
            update_data['last_name'] = last_name
        if email is not None:
            update_data['email'] = email
        if phone is not None:
            update_data['phone'] = phone
        
        success, data = self._api_request('auth/profile/', data=update_data, method='PUT', use_auth=True)
        return success, data
    
    def change_password(self, old_password: str, new_password: str) -> tuple[bool, Dict]:
        """
        Đổi mật khẩu
        
        Returns:
            (success, response_data)
        """
        success, data = self._api_request('auth/change-password/', data={
            'old_password': old_password,
            'new_password': new_password,
            'new_password_confirm': new_password,  # Server requires confirmation
        }, use_auth=True)
        return success, data

    def heartbeat(self, trading_stats: Optional[Dict] = None) -> tuple[bool, Dict]:
        """
        Gửi heartbeat để duy trì kết nối
        
        Returns:
            (success, response_data)
        """
        if not self._license_key:
            return False, {'error': 'no_license', 'message': 'No license activated'}
        
        success, data = self._api_request('license/heartbeat/', data={
            'license_key': self._license_key,
            'hardware_id': self._hardware_id,
            'app_version': self._get_app_version(),
            'trading_stats': trading_stats or {},
        })
        
        if success:
            self._last_validation = datetime.now()
            self._license_data = data.get('license', self._license_data)
            self._save_cached_license()
        
        return success, data
    
    def deactivate(self) -> tuple[bool, Dict]:
        """
        Deactivate device này khỏi license
        
        Returns:
            (success, response_data)
        """
        if not self._license_key:
            return False, {'error': 'no_license', 'message': 'No license activated'}
        
        success, data = self._api_request('license/deactivate-device/', data={
            'license_key': self._license_key,
            'hardware_id': self._hardware_id,
        })
        
        if success:
            self.stop_heartbeat()
            self.stop_notification_watcher()
            self._clear_cache()
        
        return success, data
    
    def refresh_license_status(self) -> tuple[bool, bool]:
        """
        Refresh license status từ server để kiểm tra real-time.
        Gọi API validate để lấy thông tin license mới nhất.
        
        Returns:
            (success, is_active): 
                - success: True nếu kết nối server thành công
                - is_active: True nếu license còn hoạt động
        """
        if not self._license_key:
            return False, False
        
        try:
            # Call validate API to refresh license data
            success, data = self._api_request('license/validate/', data={
                'license_key': self._license_key,
                'hardware_id': self._hardware_id,
            })
            
            if success:
                # Update cached data
                self._license_data = data.get('license', self._license_data)
                self._last_validation = datetime.now()
                self._save_cached_license()
                
                # Return both success and active status
                return True, self.is_license_active
            else:
                # Server returned error - could be license revoked, expired, etc.
                if data.get('error') == 'connection_error':
                    # Can't reach server, use cached data
                    return False, self.is_license_active
                else:
                    # Server explicitly said invalid
                    return True, False
                    
        except Exception as e:
            print(f"⚠️ Refresh license status error: {e}")
            return False, self.is_license_active
    
    # ============ HEARTBEAT THREAD ============
    
    def start_heartbeat(self):
        """Start heartbeat thread"""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        
        self._stop_heartbeat.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        print("✅ License heartbeat started")
    
    def stop_heartbeat(self):
        """Stop heartbeat thread"""
        self._stop_heartbeat.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        print("⏹️ License heartbeat stopped")
    
    def _heartbeat_loop(self):
        """Heartbeat loop chạy trong thread riêng"""
        while not self._stop_heartbeat.is_set():
            try:
                success, data = self.heartbeat()
                
                if not success:
                    if data.get('error') == 'connection_error':
                        print("⚠️ License server offline, using cached license")
                    elif not self._check_offline_grace():
                        # Offline quá lâu
                        print("❌ Offline grace period expired")
                        if self.on_license_invalid:
                            self.on_license_invalid()
                    else:
                        # License invalid
                        print(f"❌ License invalid: {data.get('message')}")
                        if self.on_license_invalid:
                            self.on_license_invalid()
                        break
                
            except Exception as e:
                print(f"⚠️ Heartbeat error: {e}")
            
            # Wait for next heartbeat
            self._stop_heartbeat.wait(self.heartbeat_interval)
    
    # ============ NOTIFICATION WATCHER (LONG POLLING) ============
    
    def start_notification_watcher(self):
        """
        Start notification watcher thread để nhận thông báo real-time từ server.
        Sử dụng long polling - server giữ connection và trả về khi có thay đổi.
        """
        if self._watcher_thread and self._watcher_thread.is_alive():
            return
        
        self._stop_watcher.clear()
        self._watcher_thread = threading.Thread(target=self._notification_loop, daemon=True)
        self._watcher_thread.start()
        print("👁️ License notification watcher started")
    
    def stop_notification_watcher(self):
        """Stop notification watcher thread"""
        self._stop_watcher.set()
        if self._watcher_thread:
            self._watcher_thread.join(timeout=5)
        print("⏹️ License notification watcher stopped")
    
    def _notification_loop(self):
        """
        Long polling loop - chờ thông báo từ server.
        Server sẽ giữ connection tối đa 30s và trả về ngay khi có thông báo mới.
        """
        consecutive_errors = 0
        
        while not self._stop_watcher.is_set():
            try:
                # Only watch if authenticated
                if not self._access_token:
                    self._stop_watcher.wait(5)
                    continue
                
                # Call watch endpoint with long polling
                result = self._watch_for_changes(timeout=30)
                
                if result:
                    has_changes = result.get('has_changes', False)
                    notifications = result.get('notifications', [])
                    server_time = result.get('server_time')
                    
                    if has_changes and notifications:
                        print(f"📬 Received {len(notifications)} notification(s) from server")
                        
                        for notif in notifications:
                            change_type = notif.get('type', '')
                            change_data = notif.get('data', {})
                            
                            print(f"  📌 Change: {change_type}")
                            
                            # Update local license data if license changed
                            if change_type.startswith('license_'):
                                if self._license_data:
                                    # Update expire_date, days_remaining, is_valid from server
                                    if 'expire_date' in change_data:
                                        self._license_data['expire_date'] = change_data['expire_date']
                                    if 'days_remaining' in change_data:
                                        self._license_data['days_remaining'] = change_data['days_remaining']
                                    if 'is_valid' in change_data:
                                        self._license_data['is_valid'] = change_data['is_valid']
                                    if 'license_type' in change_data:
                                        self._license_data['license_type'] = change_data['license_type']
                                    
                                    # Save updated cache
                                    self._save_cached_license()
                                
                                # Trigger callback
                                if self.on_license_changed:
                                    try:
                                        self.on_license_changed({
                                            'type': change_type,
                                            'data': change_data,
                                            'timestamp': notif.get('timestamp')
                                        })
                                    except Exception as cb_err:
                                        print(f"⚠️ License changed callback error: {cb_err}")
                                
                                # If license expired, call invalid callback
                                if change_type == 'license_expired' and self.on_license_invalid:
                                    try:
                                        self.on_license_invalid()
                                    except Exception as cb_err:
                                        print(f"⚠️ License invalid callback error: {cb_err}")
                            
                            # Handle force logout
                            elif change_type == 'force_logout':
                                print("🚪 Force logout requested by server")
                                if self.on_license_invalid:
                                    self.on_license_invalid()
                    
                    # Update last check time
                    if server_time:
                        self._last_notification_check = server_time
                    
                    consecutive_errors = 0
                else:
                    # No result or error
                    consecutive_errors += 1
                    
                    # Exponential backoff on errors (max 60 seconds)
                    wait_time = min(5 * (2 ** consecutive_errors), 60)
                    self._stop_watcher.wait(wait_time)
                    
            except Exception as e:
                print(f"⚠️ Notification watcher error: {e}")
                consecutive_errors += 1
                self._stop_watcher.wait(min(5 * (2 ** consecutive_errors), 60))
    
    def _watch_for_changes(self, timeout: int = 30) -> Optional[Dict]:
        """
        Call watch API endpoint với long polling.
        Server sẽ giữ connection và trả về khi có thay đổi hoặc timeout.
        
        Args:
            timeout: Thời gian chờ tối đa (seconds)
        
        Returns:
            Response data hoặc None nếu lỗi
        """
        try:
            url = f"{self.server_url}/user/watch/"
            params = {'timeout': timeout}
            
            if self._last_notification_check:
                params['last_check'] = self._last_notification_check
            
            headers = {}
            if self._access_token:
                headers['Authorization'] = f'Bearer {self._access_token}'
            
            # Use longer timeout for long polling
            response = self._session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout + 5  # Add buffer for network latency
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                # Token expired, try refresh
                if self._refresh_token:
                    self._refresh_access_token()
                return None
            else:
                print(f"⚠️ Watch API error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            # Timeout is expected for long polling
            return {'has_changes': False, 'notifications': []}
        except requests.exceptions.ConnectionError:
            # Server offline
            return None
        except Exception as e:
            print(f"⚠️ Watch for changes error: {e}")
            return None
    
    # ============ OFFLINE GRACE ============
    
    def _check_offline_grace(self) -> bool:
        """
        Kiểm tra còn trong thời gian offline grace không
        
        Returns:
            True nếu còn trong grace period
        """
        if not self._last_validation:
            return False
        
        grace_deadline = self._last_validation + timedelta(hours=self.offline_grace_hours)
        return datetime.now() < grace_deadline
    
    # ============ STATUS ============
    
    @property
    def is_authenticated(self) -> bool:
        """
        Kiểm tra user đã đăng nhập thành công chưa (có token hợp lệ)
        Khác với is_valid - is_authenticated chỉ kiểm tra đã login, không quan tâm license hết hạn
        """
        return bool(self._access_token or self._license_key)
    
    @property
    def is_license_active(self) -> bool:
        """
        Kiểm tra license có đang hoạt động (còn hạn) không
        Dùng để kiểm tra trước khi cho phép sử dụng các dịch vụ của app
        """
        if not self._license_key:
            return False
        
        # Check cached data
        if self._license_data:
            # Check if server explicitly marked as invalid
            # Chỉ return False nếu server CÓ trả về is_valid và nó là False
            if 'is_valid' in self._license_data and not self._license_data.get('is_valid'):
                return False
            
            # Check expire date
            expire_str = self._license_data.get('expire_date')
            if expire_str:
                try:
                    expire_date = datetime.fromisoformat(expire_str.replace('Z', '+00:00'))
                    if expire_date.replace(tzinfo=None) < datetime.now():
                        return False
                except:
                    pass
            
            # Check days remaining (chỉ check nếu có field này)
            if 'days_remaining' in self._license_data:
                days = self._license_data.get('days_remaining', 0)
                if days <= 0:
                    return False
        
        return True
    
    @property
    def is_expired(self) -> bool:
        """
        Kiểm tra license đã hết hạn chưa
        True nếu có license nhưng đã hết hạn
        """
        if not self._license_key:
            return False  # Không có license thì không gọi là hết hạn
        
        return not self.is_license_active
    
    @property
    def is_valid(self) -> bool:
        """Kiểm tra license có valid không (bao gồm offline grace)"""
        if not self._license_key:
            return False
        
        # Check cached data
        if self._license_data:
            # Chỉ return False nếu server CÓ trả về is_valid và nó là False
            if 'is_valid' in self._license_data and not self._license_data.get('is_valid'):
                return False
            
            # Check expire date
            expire_str = self._license_data.get('expire_date')
            if expire_str:
                try:
                    expire_date = datetime.fromisoformat(expire_str.replace('Z', '+00:00'))
                    if expire_date.replace(tzinfo=None) < datetime.now():
                        return False
                except:
                    pass
        
        # Check offline grace
        if not self._is_online:
            return self._check_offline_grace()
        
        return True
    
    @property
    def license_key(self) -> Optional[str]:
        return self._license_key
    
    @property
    def license_data(self) -> Optional[Dict]:
        return self._license_data
    
    @property
    def days_remaining(self) -> int:
        if self._license_data:
            return self._license_data.get('days_remaining', 0)
        return 0
    
    @property
    def features(self) -> Dict:
        if self._license_data:
            return self._license_data.get('features', {})
        return {}
    
    @property
    def is_online(self) -> bool:
        return self._is_online
    
    def _get_app_version(self) -> str:
        """Get app version"""
        try:
            # Try to read from version file
            version_file = Path(__file__).parent / 'VERSION'
            if version_file.exists():
                return version_file.read_text().strip()
        except:
            pass
        return '1.0.0'
    
    def get_status(self) -> Dict:
        """Get full license status"""
        return {
            'license_key': self._license_key[:16] + '...' if self._license_key else None,
            'is_valid': self.is_valid,
            'is_online': self._is_online,
            'days_remaining': self.days_remaining,
            'features': self.features,
            'hardware_id': self._hardware_id[:16] + '...',
            'device_name': self._device_name,
            'last_validation': self._last_validation.isoformat() if self._last_validation else None,
            'server_url': self.server_url,
            'connection_errors': self._connection_error_count,
        }
    
    def test_connection(self) -> tuple[bool, str]:
        """
        Test connection to license server
        
        Returns:
            (success, message)
        """
        try:
            response = self._session.get(
                f"{self.server_url}/plans/",
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                return True, "Connected to license server successfully!"
            else:
                return False, f"Server returned status {response.status_code}"
        except requests.exceptions.SSLError as e:
            return False, f"SSL certificate error: {str(e)}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to license server. Check your internet connection."
        except requests.exceptions.Timeout:
            return False, "Connection timeout. The server may be down or unreachable."
        except Exception as e:
            return False, f"Connection error: {str(e)}"


# ============ SINGLETON INSTANCE ============

_license_client: Optional[LicenseClient] = None


def get_license_client(**kwargs) -> LicenseClient:
    """Get singleton license client instance"""
    global _license_client
    if _license_client is None:
        _license_client = LicenseClient(**kwargs)
    return _license_client


def init_license_client(**kwargs) -> LicenseClient:
    """Initialize/reinitialize license client"""
    global _license_client
    _license_client = LicenseClient(**kwargs)
    return _license_client


# ============ QUICK FUNCTIONS ============

def check_license() -> bool:
    """Quick check if license is valid"""
    client = get_license_client()
    return client.is_valid


def check_license_active() -> bool:
    """Quick check if license is active (not expired)"""
    client = get_license_client()
    return client.is_license_active


def check_authenticated() -> bool:
    """Quick check if user is authenticated (logged in)"""
    client = get_license_client()
    return client.is_authenticated


def get_license_days() -> int:
    """Get remaining days"""
    client = get_license_client()
    return client.days_remaining


def get_license_features() -> Dict:
    """Get license features"""
    client = get_license_client()
    return client.features


# ============ TEST ============

if __name__ == '__main__':
    # Test license client
    print("=" * 60)
    print("LICENSE CLIENT TEST")
    print("=" * 60)
    
    # Choose server URL
    # Development: http://localhost:8000/api
    # Production:  https://license.yourdomain.com/api
    server_url = "http://localhost:8000/api"
    
    client = LicenseClient(server_url=server_url)
    
    print(f"\n📍 Server URL: {server_url}")
    print(f"🔑 Hardware ID: {client.hardware_id[:32]}...")
    print(f"💻 Device: {client._device_name}")
    print(f"🖥️ OS: {client._os_info}")
    
    # Test connection
    print("\n--- Testing Connection ---")
    success, message = client.test_connection()
    print(f"Connection: {'✅' if success else '❌'} {message}")
    
    if success:
        # Test register (uncomment to test)
        # print("\n--- Testing Register ---")
        # success, data = client.register('newuser', 'password123', 'newuser@example.com')
        # print(f"Register: {'✅' if success else '❌'}")
        # if success:
        #     print(f"  License Key: {data['license']['license_key']}")
        
        # Test login
        print("\n--- Testing Login ---")
        success, data = client.login('testuser', 'test123456')
        print(f"Login: {'✅' if success else '❌'}")
        if success:
            print(f"  License Key: {data['license']['license_key']}")
            print(f"  Days Remaining: {data['license']['days_remaining']}")
            
            # Test activate
            print("\n--- Testing Activate ---")
            success, data = client.activate(client.license_key)
            print(f"Activate: {'✅' if success else '❌'}")
            if success:
                print(f"  Device: {data['device']['device_name']}")
                
                # Test heartbeat
                print("\n--- Testing Heartbeat ---")
                success, data = client.heartbeat({'test': True})
                print(f"Heartbeat: {'✅' if success else '❌'}")
    
    print("\n--- Status ---")
    status = client.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
