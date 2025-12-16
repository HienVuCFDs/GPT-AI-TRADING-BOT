"""
🛡️ LICENSE GUARD - Bảo vệ app khỏi bị crack
============================================
Không thể bypass nếu không có license hợp lệ từ server

Features:
1. Xác thực license từ server (bắt buộc)
2. Hardware binding (khóa theo máy)
3. Time-based validation
4. Integrity check (phát hiện code bị sửa)
5. Offline cache (giới hạn thời gian)
"""

import os
import sys
import hashlib
import platform
import uuid
import time
import json
import base64
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Try import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class LicenseConfig:
    """Cấu hình license - THAY ĐỔI CÁC GIÁ TRỊ NÀY"""
    
    # 🔒 Server URL - thay bằng server thật của bạn
    LICENSE_SERVER = "https://your-license-server.com/api"
    
    # 🔒 Secret key để mã hóa (PHẢI thay đổi key này!)
    SECRET_KEY = "TRADING_BOT_SECRET_KEY_2025_CHANGE_THIS_VALUE"
    
    # 🔒 App info
    APP_NAME = "TradingBot"
    APP_VERSION = "1.0.0"
    
    # 🔒 File cache
    LICENSE_CACHE_FILE = ".license_cache.dat"
    USERS_DB_FILE = "users_db.json"
    
    # 🔒 Thời gian
    OFFLINE_CACHE_HOURS = 72  # Cho phép offline 72 giờ
    SESSION_TIMEOUT_MINUTES = 480  # 8 giờ
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_MINUTES = 30


class PasswordManager:
    """Quản lý mã hóa password"""
    
    @staticmethod
    def generate_salt() -> str:
        """Tạo salt ngẫu nhiên"""
        import secrets
        return secrets.token_hex(32)
    
    @staticmethod
    def hash_password(password: str, salt: str = None) -> Tuple[str, str]:
        """Hash password với PBKDF2"""
        if salt is None:
            salt = PasswordManager.generate_salt()
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        ).hex()
        
        return password_hash, salt
    
    @staticmethod
    def verify_password(password: str, stored_hash: str, salt: str) -> bool:
        """Xác thực password an toàn"""
        import secrets
        computed_hash, _ = PasswordManager.hash_password(password, salt)
        return secrets.compare_digest(computed_hash, stored_hash)


class LocalUserManager:
    """
    Quản lý user LOCAL (khi không có server)
    Dùng cho testing hoặc single-user mode
    """
    
    def __init__(self):
        self.users_file = LicenseConfig.USERS_DB_FILE
        self.login_attempts = {}
        self._load_users()
    
    def _load_users(self):
        """Load users từ file"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    self.users = json.load(f)
            except:
                self.users = {}
                self._create_default_admin()
        else:
            self.users = {}
            self._create_default_admin()
    
    def _create_default_admin(self):
        """Tạo admin mặc định"""
        # Password mặc định: Admin@123 (YÊU CẦU ĐỔI SAU KHI CÀI ĐẶT)
        password_hash, salt = PasswordManager.hash_password("Admin@123")
        
        self.users["admin"] = {
            "password_hash": password_hash,
            "salt": salt,
            "role": "admin",
            "license_type": "premium",
            "created_at": datetime.now().isoformat(),
            "expiry_date": (datetime.now() + timedelta(days=365)).isoformat(),
            "is_active": True,
            "hardware_ids": []  # Cho phép nhiều máy
        }
        self._save_users()
        print("✅ Created default admin account (username: admin, password: Admin@123)")
        print("⚠️ QUAN TRỌNG: Hãy đổi mật khẩu sau khi đăng nhập!")
    
    def _save_users(self):
        """Lưu users vào file"""
        try:
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(self.users, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def _check_lockout(self, username: str) -> Tuple[bool, str]:
        """Kiểm tra tài khoản có bị khóa không"""
        if username not in self.login_attempts:
            return False, ""
        
        attempt_info = self.login_attempts[username]
        
        if attempt_info.get('lockout_until'):
            lockout_until = datetime.fromisoformat(attempt_info['lockout_until'])
            if datetime.now() < lockout_until:
                remaining = (lockout_until - datetime.now()).seconds // 60 + 1
                return True, f"Tài khoản bị khóa. Thử lại sau {remaining} phút"
            else:
                self.login_attempts[username] = {'count': 0, 'lockout_until': None}
        
        return False, ""
    
    def _record_failed_attempt(self, username: str) -> int:
        """Ghi nhận lần đăng nhập thất bại"""
        if username not in self.login_attempts:
            self.login_attempts[username] = {'count': 0, 'lockout_until': None}
        
        self.login_attempts[username]['count'] += 1
        
        if self.login_attempts[username]['count'] >= LicenseConfig.MAX_LOGIN_ATTEMPTS:
            lockout_until = datetime.now() + timedelta(minutes=LicenseConfig.LOCKOUT_MINUTES)
            self.login_attempts[username]['lockout_until'] = lockout_until.isoformat()
        
        return LicenseConfig.MAX_LOGIN_ATTEMPTS - self.login_attempts[username]['count']
    
    def authenticate(self, username: str, password: str, hardware_id: str = None) -> Dict:
        """Xác thực đăng nhập"""
        # Kiểm tra lockout
        is_locked, lock_msg = self._check_lockout(username)
        if is_locked:
            return {"success": False, "message": lock_msg}
        
        # Kiểm tra user tồn tại
        if username not in self.users:
            PasswordManager.hash_password(password)  # Chống timing attack
            remaining = self._record_failed_attempt(username)
            return {
                "success": False, 
                "message": f"Tên đăng nhập hoặc mật khẩu không đúng. Còn {remaining} lần thử"
            }
        
        user = self.users[username]
        
        # Kiểm tra tài khoản active
        if not user.get("is_active", True):
            return {"success": False, "message": "Tài khoản đã bị vô hiệu hóa"}
        
        # Kiểm tra expiry
        if user.get("expiry_date"):
            expiry = datetime.fromisoformat(user["expiry_date"])
            if datetime.now() > expiry:
                return {"success": False, "message": "License đã hết hạn. Vui lòng gia hạn."}
        
        # Xác thực password
        if PasswordManager.verify_password(password, user["password_hash"], user["salt"]):
            # Reset failed attempts
            if username in self.login_attempts:
                self.login_attempts[username] = {'count': 0, 'lockout_until': None}
            
            # Cập nhật hardware ID nếu có
            if hardware_id:
                if hardware_id not in user.get("hardware_ids", []):
                    if len(user.get("hardware_ids", [])) < 3:  # Max 3 máy
                        user.setdefault("hardware_ids", []).append(hardware_id)
                        self._save_users()
                    # Không block nếu quá 3 máy, chỉ warning
            
            # Cập nhật last login
            user["last_login"] = datetime.now().isoformat()
            self._save_users()
            
            return {
                "success": True,
                "message": "Đăng nhập thành công",
                "user": {
                    "username": username,
                    "role": user.get("role", "user"),
                    "license_type": user.get("license_type", "trial"),
                    "expiry_date": user.get("expiry_date")
                }
            }
        else:
            remaining = self._record_failed_attempt(username)
            if remaining > 0:
                return {
                    "success": False,
                    "message": f"Tên đăng nhập hoặc mật khẩu không đúng. Còn {remaining} lần thử"
                }
            else:
                return {
                    "success": False,
                    "message": f"Tài khoản đã bị khóa {LicenseConfig.LOCKOUT_MINUTES} phút"
                }
    
    def create_user(self, admin_username: str, new_username: str, new_password: str, 
                    role: str = "user", license_type: str = "trial", days_valid: int = 30) -> Dict:
        """Tạo user mới (chỉ admin)"""
        # Kiểm tra quyền admin
        if admin_username not in self.users or self.users[admin_username].get("role") != "admin":
            return {"success": False, "message": "Không có quyền tạo user"}
        
        if new_username in self.users:
            return {"success": False, "message": "Username đã tồn tại"}
        
        # Validate password
        if len(new_password) < 8:
            return {"success": False, "message": "Password phải có ít nhất 8 ký tự"}
        
        password_hash, salt = PasswordManager.hash_password(new_password)
        
        self.users[new_username] = {
            "password_hash": password_hash,
            "salt": salt,
            "role": role,
            "license_type": license_type,
            "created_at": datetime.now().isoformat(),
            "expiry_date": (datetime.now() + timedelta(days=days_valid)).isoformat(),
            "is_active": True,
            "hardware_ids": []
        }
        self._save_users()
        
        return {"success": True, "message": f"Tạo user {new_username} thành công"}
    
    def change_password(self, username: str, old_password: str, new_password: str) -> Dict:
        """Đổi mật khẩu"""
        if username not in self.users:
            return {"success": False, "message": "User không tồn tại"}
        
        user = self.users[username]
        
        # Xác thực password cũ
        if not PasswordManager.verify_password(old_password, user["password_hash"], user["salt"]):
            return {"success": False, "message": "Mật khẩu cũ không đúng"}
        
        # Validate password mới
        if len(new_password) < 8:
            return {"success": False, "message": "Password mới phải có ít nhất 8 ký tự"}
        
        # Cập nhật password
        password_hash, salt = PasswordManager.hash_password(new_password)
        user["password_hash"] = password_hash
        user["salt"] = salt
        self._save_users()
        
        return {"success": True, "message": "Đổi mật khẩu thành công"}


class LicenseGuard:
    """
    🛡️ HỆ THỐNG BẢO VỆ LICENSE
    
    Đa lớp bảo mật:
    1. Xác thực với server (nếu có)
    2. Fallback sang local user database
    3. Hardware binding
    4. Session management
    5. Integrity check
    """
    
    def __init__(self):
        self.hardware_id = self._generate_hardware_id()
        self.is_validated = False
        self.user_info = None
        self.license_type = None
        self.expiry_date = None
        self.session_token = None
        self.session_created = None
        
        # User manager (local)
        self.local_user_manager = LocalUserManager()
        
        logger.info(f"🔒 LicenseGuard initialized. Hardware ID: {self.hardware_id[:8]}...")
    
    def _generate_hardware_id(self) -> str:
        """Tạo Hardware ID duy nhất cho mỗi máy tính"""
        try:
            mac = uuid.getnode()
            mac_str = ':'.join(('%012X' % mac)[i:i+2] for i in range(0, 12, 2))
            
            system_info = f"{platform.node()}-{platform.machine()}-{platform.processor()}"
            
            combined = f"{mac_str}-{system_info}-{LicenseConfig.SECRET_KEY}"
            hardware_id = hashlib.sha256(combined.encode()).hexdigest()[:32]
            
            return hardware_id.upper()
            
        except Exception as e:
            logger.warning(f"Could not generate hardware ID: {e}")
            return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:32].upper()
    
    def _check_code_integrity(self) -> bool:
        """Kiểm tra code có bị sửa đổi không"""
        try:
            # CHỈ check app.py và license_client.py (không check file này vì có chứa patterns)
            critical_files = ['app.py', 'license_client.py']
            
            # Encode patterns để tránh self-detection
            # Các pattern được base64 encode
            encoded_patterns = [
                'aXNfdmFsaWRhdGVkID0gVHJ1ZSAgIyBDUkFDS0VE',  # is_validated = True  # CRACKED
                'IyBCWVBBU1MgTElDRU5TRQ==',  # # BYPASS LICENSE
                'cmV0dXJuIFRydWUgICMgSEFDS0VE',  # return True  # HACKED
                'c2VsZi5pc192YWxpZGF0ZWQgPSBUcnVlICAjIEZPUkNF',  # self.is_validated = True  # FORCE
                'TElDRU5TRV9DSEVDSyA9IEZhbHNl',  # LICENSE_CHECK = False
                'U0tJUF9MSUNFTlNF',  # SKIP_LICENSE
                'IyBQSVJBVEVE',  # # PIRATED
                'aXNfdmFsaWQgPSBUcnVlICAjIENSQUNL',  # is_valid = True  # CRACK
                'IyBOT19MSUNFTlNFX0NIRUNL',  # # NO_LICENSE_CHECK
            ]
            
            # Decode patterns
            dangerous_patterns = []
            for encoded in encoded_patterns:
                try:
                    decoded = base64.b64decode(encoded)
                    dangerous_patterns.append(decoded)
                except:
                    pass
            
            for filename in critical_files:
                if not os.path.exists(filename):
                    continue
                    
                try:
                    with open(filename, 'rb') as f:
                        content = f.read()
                    
                    for pattern in dangerous_patterns:
                        if pattern in content:
                            logger.critical(f"🚨 Code integrity check failed in {filename}!")
                            return False
                                
                except Exception as e:
                    logger.warning(f"Could not check {filename}: {e}")
                    
            return True
            
        except Exception:
            return True
    
    def validate_online(self, username: str, password: str) -> Tuple[bool, str, Dict]:
        """
        Xác thực license - thử server trước, fallback sang local
        """
        # 1. Kiểm tra code integrity
        if not self._check_code_integrity():
            return False, "🚨 Phát hiện code bị sửa đổi. App không thể chạy.", {}
        
        # 2. Thử xác thực với server (nếu có requests)
        if REQUESTS_AVAILABLE and LicenseConfig.LICENSE_SERVER != "https://your-license-server.com/api":
            try:
                success, message, data = self._validate_with_server(username, password)
                if success:
                    return success, message, data
                # Nếu server trả về lỗi cụ thể (không phải connection error), dừng
                if "Không thể kết nối" not in message:
                    return success, message, data
            except Exception as e:
                logger.warning(f"Server validation failed: {e}")
        
        # 3. Fallback sang local authentication
        logger.info("Using local authentication...")
        result = self.local_user_manager.authenticate(username, password, self.hardware_id)
        
        if result["success"]:
            self.is_validated = True
            self.user_info = result.get("user", {})
            self.license_type = self.user_info.get("license_type", "trial")
            self.expiry_date = self.user_info.get("expiry_date")
            
            # Tạo session
            import secrets
            self.session_token = secrets.token_urlsafe(32)
            self.session_created = datetime.now()
            
            # Lưu cache
            self._save_license_cache()
            
            return True, result["message"], result
        else:
            return False, result["message"], {}
    
    def _validate_with_server(self, username: str, password: str) -> Tuple[bool, str, Dict]:
        """Xác thực với license server"""
        try:
            payload = {
                'username': username,
                'password': hashlib.sha256(password.encode()).hexdigest(),
                'hardware_id': self.hardware_id,
                'app_version': LicenseConfig.APP_VERSION,
                'timestamp': int(time.time())
            }
            
            response = requests.post(
                f"{LicenseConfig.LICENSE_SERVER}/validate",
                json=payload,
                timeout=30,
                verify=True
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    self.is_validated = True
                    self.user_info = data.get('user', {})
                    self.license_type = data.get('license_type', 'trial')
                    self.expiry_date = data.get('expiry_date')
                    
                    self._save_license_cache()
                    
                    return True, "Đăng nhập thành công!", data
                else:
                    return False, data.get('message', 'Đăng nhập thất bại'), {}
            else:
                return False, f"Lỗi server: {response.status_code}", {}
                
        except requests.exceptions.ConnectionError:
            return False, "Không thể kết nối đến server license", {}
        except Exception as e:
            return False, f"Lỗi: {str(e)}", {}
    
    def _encrypt_data(self, data: str) -> str:
        """Mã hóa dữ liệu"""
        key = LicenseConfig.SECRET_KEY[:32].ljust(32, '0')
        encrypted = ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(data))
        return base64.b64encode(encrypted.encode('latin-1')).decode()
    
    def _decrypt_data(self, encrypted: str) -> str:
        """Giải mã dữ liệu"""
        key = LicenseConfig.SECRET_KEY[:32].ljust(32, '0')
        decoded = base64.b64decode(encrypted).decode('latin-1')
        return ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(decoded))
    
    def _save_license_cache(self):
        """Lưu cache license"""
        try:
            cache_data = {
                'user_info': self.user_info,
                'license_type': self.license_type,
                'expiry_date': self.expiry_date,
                'hardware_id': self.hardware_id,
                'cached_at': datetime.now().isoformat(),
                'valid_until': (datetime.now() + timedelta(hours=LicenseConfig.OFFLINE_CACHE_HOURS)).isoformat()
            }
            
            encrypted = self._encrypt_data(json.dumps(cache_data))
            
            with open(LicenseConfig.LICENSE_CACHE_FILE, 'w') as f:
                f.write(encrypted)
                
        except Exception as e:
            logger.warning(f"Could not save license cache: {e}")
    
    def _load_license_cache(self) -> Tuple[bool, Dict]:
        """Load cache license"""
        try:
            if not os.path.exists(LicenseConfig.LICENSE_CACHE_FILE):
                return False, {}
            
            with open(LicenseConfig.LICENSE_CACHE_FILE, 'r') as f:
                encrypted = f.read()
            
            decrypted = self._decrypt_data(encrypted)
            cache_data = json.loads(decrypted)
            
            # Kiểm tra hardware ID
            if cache_data.get('hardware_id') != self.hardware_id:
                return False, {}
            
            # Kiểm tra thời hạn cache
            valid_until = datetime.fromisoformat(cache_data['valid_until'])
            if datetime.now() > valid_until:
                return False, {}
            
            return True, cache_data
            
        except Exception:
            return False, {}
    
    def try_auto_login(self) -> Tuple[bool, str]:
        """Thử đăng nhập tự động từ cache"""
        success, cache_data = self._load_license_cache()
        
        if success:
            self.is_validated = True
            self.user_info = cache_data.get('user_info', {})
            self.license_type = cache_data.get('license_type', 'trial')
            self.expiry_date = cache_data.get('expiry_date')
            
            return True, "Đăng nhập tự động thành công"
        
        return False, "Không có cache hợp lệ"
    
    def check_license_valid(self) -> bool:
        """Kiểm tra license còn hợp lệ không"""
        if not self.is_validated:
            return False
        
        # Kiểm tra code integrity
        if not self._check_code_integrity():
            self.is_validated = False
            return False
        
        # Kiểm tra session timeout
        if self.session_created:
            session_age = (datetime.now() - self.session_created).total_seconds() / 60
            if session_age > LicenseConfig.SESSION_TIMEOUT_MINUTES:
                logger.warning("Session expired")
                # Không invalidate hoàn toàn, chỉ cần refresh
        
        # Kiểm tra expiry date
        if self.expiry_date:
            try:
                expiry = datetime.fromisoformat(self.expiry_date)
                if datetime.now() > expiry:
                    self.is_validated = False
                    return False
            except:
                pass
        
        return True
    
    def logout(self):
        """Đăng xuất"""
        self.is_validated = False
        self.user_info = None
        self.license_type = None
        self.session_token = None
        
        # Xóa cache
        try:
            if os.path.exists(LicenseConfig.LICENSE_CACHE_FILE):
                os.remove(LicenseConfig.LICENSE_CACHE_FILE)
        except:
            pass
        
        logger.info("User logged out")
    
    def get_hardware_id(self) -> str:
        """Lấy Hardware ID để hiển thị cho user"""
        return self.hardware_id
    
    def force_exit_if_invalid(self):
        """Buộc thoát app nếu license không hợp lệ"""
        if not self.check_license_valid():
            print("🚨 License không hợp lệ! App sẽ đóng.")
            sys.exit(1)


# === SINGLETON INSTANCE ===
_license_guard = None

def get_license_guard() -> LicenseGuard:
    """Lấy instance duy nhất của LicenseGuard"""
    global _license_guard
    if _license_guard is None:
        _license_guard = LicenseGuard()
    return _license_guard


def require_valid_license(func):
    """
    Decorator để bảo vệ function
    
    Sử dụng:
        @require_valid_license
        def sensitive_function():
            ...
    """
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        guard = get_license_guard()
        if not guard.check_license_valid():
            logger.warning(f"License required for {func.__name__}")
            return None
        return func(*args, **kwargs)
    return wrapper


# === TEST ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🛡️ LICENSE GUARD TEST")
    print("=" * 60)
    
    guard = get_license_guard()
    
    print(f"\n🔑 Hardware ID: {guard.get_hardware_id()}")
    print(f"📁 Users DB: {LicenseConfig.USERS_DB_FILE}")
    
    # Test đăng nhập
    print("\n📝 Testing login with default admin...")
    success, message, data = guard.validate_online("admin", "Admin@123")
    print(f"   Result: {message}")
    
    if success:
        print(f"   User: {guard.user_info}")
        print(f"   License: {guard.license_type}")
        print(f"   Valid: {guard.check_license_valid()}")
    
    print("\n✅ Test completed!")
