""" 
License Server Views - API Endpoints
"""
import logging
import sys
import threading
from datetime import timedelta

from rest_framework import status, generics
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle, UserRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.utils import timezone
from django.db import transaction
from django.shortcuts import get_object_or_404

from .models import (
    License, DeviceActivation, UsageLog, SubscriptionPlan,
    Subscription, LicenseStatus, EmailVerificationToken, ActivationCode, LicenseType,
    PasswordResetToken, get_country_flag, get_email_language_from_country
)
from .serializers import (
    RegisterSerializer, UserSerializer, ChangePasswordSerializer,
    LicenseSerializer, LicenseDetailSerializer, DeviceActivationSerializer,
    LicenseActivateSerializer, LicenseValidateSerializer, HeartbeatSerializer,
    DeactivateDeviceSerializer, SubscriptionPlanSerializer, SubscriptionSerializer
)

# Setup logger
logger = logging.getLogger(__name__)


def trigger_github_update_check():
    """Trigger GitHub update check in background thread (non-blocking)"""
    def _do_check():
        try:
            # Import here to avoid circular imports
            from run_server import GitHubUpdateThread
            GitHubUpdateThread.trigger_check()
        except Exception as e:
            logger.debug(f"GitHub update trigger failed: {e}")
    
    # Run in background thread to not block the response
    threading.Thread(target=_do_check, daemon=True).start()


# ============ THROTTLING ============

class RegisterThrottle(AnonRateThrottle):
    rate = '100/hour'  # Gioi han 100 dang ky / gio

class LoginThrottle(AnonRateThrottle):
    rate = '10/minute'  # Gioi han 10 login / phut

class HeartbeatThrottle(UserRateThrottle):
    rate = '60/minute'  # Heartbeat 1 lan / giay max


# ============ HELPER FUNCTIONS ============

def get_client_ip(request):
    """Lay IP cua client"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def log_usage(license_obj, event_type, device=None, request=None, event_data=None):
    """Helper de log usage"""
    UsageLog.objects.create(
        license=license_obj,
        device=device,
        event_type=event_type,
        event_data=event_data or {},
        ip_address=get_client_ip(request) if request else None,
        user_agent=request.META.get('HTTP_USER_AGENT', '')[:500] if request else ''
    )


def _check_and_send_expiry_warning(license_obj):
    """
    Kiem tra va gui email canh bao khi license sap het han.
    Chi gui khi con 3, 2, hoac 1 ngay VA chua gui lan nao.
    """
    import threading
    
    if license_obj.status != 'active':
        return
    
    if license_obj.expiry_warning_sent:
        return
    
    days_left = license_obj.days_remaining()
    
    if days_left > 3 or days_left < 0:
        return
    
    user = license_obj.user
    
    if not user.email or '@' not in user.email:
        return
    
    lang = 'vi'
    try:
        # Xac dinh ngon ngu tu quoc gia user
        from users.models import get_email_language_from_country
        user_country = ''
        if hasattr(user, 'profile') and user.profile.country:
            user_country = user.profile.country
        lang = get_email_language_from_country(user_country)
    except:
        pass
    
    def send_email_async():
        try:
            success = license_obj.send_expiry_warning_email(language=lang)
            if success:
                print(f"Sent expiry warning to {user.email} ({days_left} days left)")
        except Exception as e:
            print(f"Failed to send expiry warning to {user.email}: {e}")
    
    thread = threading.Thread(target=send_email_async, daemon=True)
    thread.start()


# ============ AUTH VIEWS ============

def send_activation_code_email(user, activation_code, language=None):
    """Gui email chua ma kich hoat cho user - Tu dong xac dinh ngon ngu tu quoc gia
    - VN: Tieng Viet
    - Cac quoc gia khac: Tieng Anh
    """
    from django.core.mail import send_mail
    from django.conf import settings
    from .models import get_email_language_from_country
    import threading
    
    # Tu dong xac dinh ngon ngu tu quoc gia user
    if language is None:
        if hasattr(user, 'profile') and user.profile.country:
            language = get_email_language_from_country(user.profile.country)
        else:
            language = 'en'  # Default tieng Anh neu khong co thong tin quoc gia

    if language == 'en':
        subject = 'Activation Code - Trading Bot'
        message = f"""
Hello {user.first_name or user.username},

Thank you for registering Trading Bot!

Here is your activation code:

    ACTIVATION CODE: {activation_code.code}

This code is valid for {activation_code.trial_days} days trial.

Activation instructions:
1. Open Trading Bot application
2. Enter the activation code in the input field
3. Click "Activate" button

Note: This code can only be used {activation_code.max_uses} time(s).

If you need support, please contact Admin.

Best regards,
Trading Bot Team
        """

        html_message = f"""
        <html>
        <head><meta charset="UTF-8"></head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #4CAF50;">Activation Code - Trading Bot</h2>

                <p>Hello <strong>{user.first_name or user.username}</strong>,</p>

                <p>Thank you for registering Trading Bot!</p>

                <div style="background: linear-gradient(135deg, #4CAF50, #2E7D32); padding: 25px; border-radius: 10px; text-align: center; margin: 20px 0;">
                    <p style="color: white; margin: 0 0 10px 0; font-size: 14px;">YOUR ACTIVATION CODE</p>
                    <p style="color: white; font-size: 32px; font-weight: bold; letter-spacing: 8px; margin: 0; font-family: monospace;">
                        {activation_code.code}
                    </p>
                </div>

                <div style="background: #E8F5E9; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <p style="margin: 0;"><strong>Code Information:</strong></p>
                    <ul style="margin: 10px 0;">
                        <li>Trial Period: <strong>{activation_code.trial_days} days</strong></li>
                        <li>Usage Limit: <strong>{activation_code.max_uses} time(s)</strong></li>
                    </ul>
                </div>

                <h3 style="color: #1976D2;">Activation Instructions:</h3>
                <ol>
                    <li>Open Trading Bot application</li>
                    <li>Enter the activation code in the input field</li>
                    <li>Click "Activate" button</li>
                </ol>

                <p style="color: #666; font-size: 12px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;">
                    If you need support, please contact Admin.<br>
                    Best regards,<br>
                    <strong>Trading Bot Team</strong>
                </p>
            </div>
        </body>
        </html>
        """
    else:
        # Vietnamese (default)
        subject = 'Ma Kich Hoat Tai Khoan Trading Bot'
        message = f"""
Xin chao {user.first_name or user.username},

Cam on ban da dang ky Trading Bot!

Day la ma kich hoat tai khoan cua ban:

    MA KICH HOAT: {activation_code.code}

Ma nay co hieu luc cho {activation_code.trial_days} ngay dung thu.

Huong dan kich hoat:
1. Mo ung dung Trading Bot
2. Nhap ma kich hoat vao o nhap lieu
3. Nhan nut "Kich hoat"

Luu y: Ma nay chi co the su dung {activation_code.max_uses} lan.

Neu ban can ho tro, vui long lien he Admin.

Tran trong,
Trading Bot Team
        """

        html_message = f"""
        <html>
        <head><meta charset="UTF-8"></head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #4CAF50;">Ma Kich Hoat Tai Khoan Trading Bot</h2>

                <p>Xin chao <strong>{user.first_name or user.username}</strong>,</p>

                <p>Cam on ban da dang ky Trading Bot!</p>

                <div style="background: linear-gradient(135deg, #4CAF50, #2E7D32); padding: 25px; border-radius: 10px; text-align: center; margin: 20px 0;">
                    <p style="color: white; margin: 0 0 10px 0; font-size: 14px;">MA KICH HOAT CUA BAN</p>
                    <p style="color: white; font-size: 32px; font-weight: bold; letter-spacing: 8px; margin: 0; font-family: monospace;">
                        {activation_code.code}
                    </p>
                </div>

                <div style="background: #E8F5E9; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <p style="margin: 0;"><strong>Thong tin ma:</strong></p>
                    <ul style="margin: 10px 0;">
                        <li>Thoi han Trial: <strong>{activation_code.trial_days} ngay</strong></li>
                        <li>So lan su dung: <strong>{activation_code.max_uses} lan</strong></li>
                    </ul>
                </div>

                <h3 style="color: #1976D2;">Huong dan kich hoat:</h3>
                <ol>
                    <li>Mo ung dung Trading Bot</li>
                    <li>Nhap ma kich hoat vao o nhap lieu</li>
                    <li>Nhan nut "Kich hoat"</li>
                </ol>

                <p style="color: #666; font-size: 12px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;">
                    Neu ban can ho tro, vui long lien he Admin.<br>
                    Tran trong,<br>
                    <strong>Trading Bot Team</strong>
                </p>
            </div>
        </body>
        </html>
        """

    def send_async():
        try:
            print(f"Sending email to {user.email} (lang: {language})...")
            result = send_mail(
                subject=subject,
                message=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                html_message=html_message,
                fail_silently=False
            )
            print(f"Email sent to {user.email} with code {activation_code.code} - Result: {result}")
        except Exception as e:
            print(f"Failed to send email to {user.email}: {type(e).__name__}: {e}")

    # Send email in background thread to avoid blocking
    thread = threading.Thread(target=send_async)
    thread.daemon = True
    thread.start()
    print(f"Email thread started for {user.email}")


@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([RegisterThrottle])
def register_view(request):
    """Dang ky user moi - Tu dong gui ma kich hoat qua email"""
    """Dang ky user moi - Tu dong gui ma kich hoat qua email"""
    serializer = RegisterSerializer(data=request.data)
    
    if serializer.is_valid():
        user = serializer.save()
        
        # User active ngay nhung chua co License
        user.is_active = True
        user.save()
        
        # Xac dinh ngon ngu email tu quoc gia user
        from .models import get_email_language_from_country
        user_country = ''
        if hasattr(user, 'profile') and user.profile.country:
            user_country = user.profile.country
        language = get_email_language_from_country(user_country)
        
        # Tu dong tao ma kich hoat moi cho user nay
        activation_code = ActivationCode.objects.create(
            trial_days=7,
            max_uses=1,
            is_active=True
        )
        # Ma duoc tu dong generate trong model save()
        print(f"Created activation code: {activation_code.code} for user: {user.username}")
        
        email_sent = False
        
        print(f"User email: '{user.email}' (type: {type(user.email)})")
        if user.email:
            # Gui email voi ma kich hoat (async) - truyen language
            send_activation_code_email(user, activation_code, language)
            email_sent = True
            print(f"Email function called for {user.email}")
        else:
            print(f"No email found for user {user.username}")
        
        # Thong bao theo ngon ngu
        if language == 'en':
            success_msg = 'Registration successful! Please check your email for activation code.' if email_sent else 'Registration successful! Please contact Admin to get activation code.'
        else:
            success_msg = 'Dang ky thanh cong! Vui long kiem tra email de lay ma kich hoat.' if email_sent else 'Dang ky thanh cong! Vui long lien he Admin de duoc cap ma kich hoat.'
        
        return Response({
            'status': 'success',
            'message': success_msg,
            'require_activation': True,
            'email_sent': email_sent,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
            },
            'license': None
        }, status=status.HTTP_201_CREATED)
    
    return Response({
        'status': 'error',
        'errors': serializer.errors
    }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
def activate_by_code_view(request):
    """Kich hoat tai khoan bang ma 6 ky tu - Cap Trial 7 ngay"""
    username = request.data.get('username', '').strip()
    code = request.data.get('code', '').strip().upper()
    
    if not username or not code:
        return Response({
            'status': 'error',
            'message': 'Please enter username and activation code.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Tim user
    try:
        user = User.objects.get(username=username.lower())
    except User.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Account not found.'
        }, status=status.HTTP_404_NOT_FOUND)
    
    # Kiem tra user da co license chua
    existing_license = user.licenses.filter(status=LicenseStatus.ACTIVE).first()
    if existing_license:
        return Response({
            'status': 'error',
            'message': 'Account already has an active license.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Tim ma kich hoat
    try:
        activation = ActivationCode.objects.get(code=code)
    except ActivationCode.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Invalid activation code.'
        }, status=status.HTTP_404_NOT_FOUND)
    
    # Kiem tra ma con dung duoc khong
    if not activation.is_valid():
        return Response({
            'status': 'error',
            'message': 'Activation code expired or already used.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Tao License Trial
    trial_days = activation.trial_days
    license_obj = License.objects.create(
        user=user,
        license_type=LicenseType.TRIAL,
        status=LicenseStatus.ACTIVE,
        expire_date=timezone.now() + timedelta(days=trial_days),
        max_devices=1,
        note=f'Kich hoat bang ma: {code}'
    )
    
    # Danh dau ma da dung
    activation.use()
    
    return Response({
        'status': 'success',
        'message': f'Kich hoat thanh cong! Ban duoc dung thu {trial_days} ngay.',
        'license': {
            'license_key': license_obj.license_key,
            'license_type': license_obj.license_type,
            'days_remaining': trial_days,
            'expire_date': license_obj.expire_date,
        }
    })


@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([LoginThrottle])
def login_view(request):
    """Dang nhap va nhan JWT tokens - ho tro username, email hoac so dien thoai"""
    login_id = request.data.get('username', '').strip()  # Co the la username, email hoac phone
    password = request.data.get('password', '')
    
    if not login_id or not password:
        return Response({
            'status': 'error',
            'message': 'Username/Email/Phone and password are required'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    user = None
    
    # Thu tim user theo nhieu cach
    # 1. Tim theo username (case-insensitive)
    try:
        user = User.objects.get(username__iexact=login_id)
    except User.DoesNotExist:
        pass
    
    # 2. Tim theo email (case-insensitive)
    if not user:
        try:
            user = User.objects.get(email__iexact=login_id)
        except User.DoesNotExist:
            pass
    
    # 3. Tim theo so dien thoai (trong UserProfile)
    if not user:
        from .models import UserProfile
        try:
            # Chuan hoa so dien thoai - loai bo khoang trang va dau
            phone_normalized = login_id.replace(' ', '').replace('-', '').replace('.', '')
            profile = UserProfile.objects.get(phone=phone_normalized)
            user = profile.user
        except UserProfile.DoesNotExist:
            # Thu tim voi so dien thoai goc
            try:
                profile = UserProfile.objects.get(phone=login_id)
                user = profile.user
            except UserProfile.DoesNotExist:
                pass
    
    # Xac thuc mat khau
    if user and user.check_password(password):
        pass  # User hop le
    else:
        return Response({
            'status': 'error',
            'message': 'Invalid username/email/phone or password'
        }, status=status.HTTP_401_UNAUTHORIZED)
    
    if not user.is_active:
        return Response({
            'status': 'error',
            'message': 'Account is disabled'
        }, status=status.HTTP_403_FORBIDDEN)
    
    # Lay license active
    license_obj = user.licenses.filter(status=LicenseStatus.ACTIVE).first()

    # Neu khong co license active, tim license expired moi nhat
    if not license_obj:
        license_obj = user.licenses.filter(status=LicenseStatus.EXPIRED).order_by('-expire_date').first()

    # Neu van khong co license nao
    if not license_obj:
        # Van tao token de user co the tao don thanh toan
        refresh = RefreshToken.for_user(user)
        return Response({
            'status': 'success',
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
            },
            'license': None,
            'license_status': 'no_license',
            'message': 'No license found. Please purchase a license.'
        })

    # Kiem tra expired
    if not license_obj.is_valid():
        license_obj.status = LicenseStatus.EXPIRED
        license_obj.save()
        
        # Van tao token de user co the gia han
        refresh = RefreshToken.for_user(user)
        
        # Get user profile data
        user_country = ''
        user_country_flag = ''
        if hasattr(user, 'profile'):
            user_country = user.profile.country or ''
            user_country_flag = user.profile.country_flag or get_country_flag(user.profile.country)
        
        return Response({
            'status': 'success',
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'country': user_country,
                'country_flag': user_country_flag,
            },
            'license': LicenseSerializer(license_obj).data,
            'license_status': 'expired',
            'message': 'License has expired. Please renew.'
        })
    
    # ============ CHECK IF ALREADY ONLINE - SIMPLE APPROACH ============
    force_login = request.data.get('force_login', False)
    hardware_id = request.data.get('hardware_id')
    session_id = request.data.get('session_id')  # 🔧 Get session_id from client
    
    print(f"[Login] ========== LOGIN REQUEST ==========", flush=True)
    print(f"[Login] User: {user.username}", flush=True)
    print(f"[Login] Hardware ID: {hardware_id[:16] if hardware_id else 'None'}...", flush=True)
    print(f"[Login] Session ID: {session_id[:16] if session_id else 'None'}...", flush=True)
    print(f"[Login] Force Login: {force_login}", flush=True)
    
    # 🔧 Use is_online_now for instant checking (no more 45s delay!)
    # Also check heartbeat as fallback (in case is_online_now wasn't updated properly)
    heartbeat_threshold = timezone.now() - timezone.timedelta(seconds=45)
    print(f"[Login] Heartbeat threshold: {heartbeat_threshold}", flush=True)
    print(f"[Login] Current server time: {timezone.now()}", flush=True)
    
    # Check tất cả devices của license này
    all_devices = DeviceActivation.objects.filter(
        license=license_obj,
        is_active=True
    )
    print(f"[Login] Total active devices for license: {all_devices.count()}", flush=True)
    
    for d in all_devices:
        # Check both is_online_now AND heartbeat
        heartbeat_online = d.last_heartbeat and d.last_heartbeat >= heartbeat_threshold
        print(f"[Login]   Device: {d.hardware_id[:16]}... | is_online_now: {d.is_online_now} | heartbeat_online: {heartbeat_online}", flush=True)
        
        # 🔧 AUTO-FIX: If is_online_now=True but heartbeat too old, mark offline automatically
        # This handles case when app crashed/closed without calling logout
        if d.is_online_now and not heartbeat_online:
            print(f"[Login AUTO-FIX] Device {d.hardware_id[:16]}... has stale online status, marking OFFLINE", flush=True)
            d.mark_offline()
    
    # 🔧 Re-query after auto-fix
    # Device is truly online only if BOTH is_online_now=True AND heartbeat is recent
    from django.db.models import Q
    online_devices = all_devices.filter(
        is_online_now=True,
        last_heartbeat__gte=heartbeat_threshold
    )
    # KHÔNG exclude hardware_id - vì muốn chặn cả multi-instance trên cùng 1 máy
    
    print(f"[Login] Online devices count: {online_devices.count()}", flush=True)
    
    if online_devices.exists():
        device = online_devices.first()
        print(f"[Login BLOCKED] License {license_obj.license_key[:10]}... is already online on device: {device.device_name}", flush=True)
        print(f"[Login BLOCKED] Device is_online_now: {device.is_online_now}, last_heartbeat: {device.last_heartbeat}", flush=True)
        
        if force_login:
            # 🔧 Force login - mark ALL devices offline immediately
            DeviceActivation.mark_all_offline_for_license(license_obj)
            print(f"[Force Login] ✅ Marked all devices OFFLINE for license: {license_obj.license_key[:10]}...", flush=True)
        else:
            return Response({
                'status': 'error',
                'message': 'Tài khoản đang được sử dụng ở nơi khác. Vui lòng đóng ứng dụng kia hoặc đợi 2 phút.',
                'message_en': 'Account is already in use elsewhere. Please close the other app or wait 2 minutes.',
                'error_code': 'SESSION_ACTIVE',
                'device_name': device.device_name or 'Unknown Device',
                'last_seen': device.last_heartbeat.isoformat() if device.last_heartbeat else None,
            }, status=status.HTTP_403_FORBIDDEN)
    
    print(f"[Login ALLOWED] No online devices found for license {license_obj.license_key[:10]}...", flush=True)
    # ============ END CHECK ONLINE ============
    
    # ============ MARK AS ONLINE ============
    if hardware_id:
        device, created = DeviceActivation.objects.get_or_create(
            license=license_obj,
            hardware_id=hardware_id,
            defaults={
                'device_name': request.data.get('device_name', ''),
                'os_info': request.data.get('os_info', ''),
                'ip_address': get_client_ip(request),
            }
        )
        
        # 🔧 Update device info nếu có
        if request.data.get('device_name'):
            device.device_name = request.data.get('device_name')
        if request.data.get('os_info'):
            device.os_info = request.data.get('os_info')
        device.ip_address = get_client_ip(request)
        device.is_active = True
        
        # 🔧 Register session_id để track multiple instances
        if session_id:
            # Clear old sessions first (force new session)
            device.active_sessions = {}
            device.active_sessions[session_id] = timezone.now().isoformat()
            print(f"[Login] Registered session {session_id[:16]}... for device", flush=True)
        
        device.save()
        
        # 🔧 Mark device as ONLINE immediately (for real-time dashboard)
        device.mark_online()
        print(f"[Login] ✅ Device {hardware_id[:16]}... marked ONLINE for user {user.username}", flush=True)
    # ============ END MARK ONLINE ============
    
    # Tao tokens
    refresh = RefreshToken.for_user(user)
    
    # Log login
    log_usage(license_obj, UsageLog.EventType.LOGIN, request=request)
    
    print(f"[Login SUCCESS] User {user.username} logged in with license {license_obj.license_key[:10]}...", flush=True)
    
    # Get user profile data
    user_country = ''
    user_country_flag = ''
    if hasattr(user, 'profile'):
        user_country = user.profile.country or ''
        user_country_flag = user.profile.country_flag or get_country_flag(user.profile.country)
    
    return Response({
        'status': 'success',
        'access': str(refresh.access_token),
        'refresh': str(refresh),
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'country': user_country,
            'country_flag': user_country_flag,
        },
        'license': LicenseSerializer(license_obj).data,
        'license_status': 'active'  # 🔧 Thêm license_status để client biết start heartbeat
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    """Logout - blacklist refresh token and mark device offline"""
    try:
        refresh_token = request.data.get('refresh')
        hardware_id = request.data.get('hardware_id')
        
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()
        
        # Log logout and mark device offline
        license_obj = request.user.licenses.filter(status=LicenseStatus.ACTIVE).first()
        if not license_obj:
            license_obj = request.user.licenses.filter(status=LicenseStatus.EXPIRED).first()
        
        if license_obj:
            log_usage(license_obj, UsageLog.EventType.LOGOUT, request=request)
            
            # 🔧 Mark device as OFFLINE immediately
            if hardware_id:
                device = DeviceActivation.objects.filter(
                    license=license_obj,
                    hardware_id=hardware_id
                ).first()
                if device:
                    device.mark_offline()
                    print(f"[Logout] ✅ Device {hardware_id[:16]}... marked OFFLINE", flush=True)
            else:
                # If no hardware_id, mark all devices offline
                DeviceActivation.mark_all_offline_for_license(license_obj)
                print(f"[Logout] ✅ All devices marked OFFLINE for {request.user.username}", flush=True)
        
        return Response({'status': 'success', 'message': 'Logged out successfully'})
    except Exception as e:
        print(f"[Logout] Error: {e}", flush=True)
        return Response({'status': 'success', 'message': 'Logged out'})


@api_view(['GET', 'POST', 'PUT', 'PATCH'])
@permission_classes([IsAuthenticated])
def profile_view(request):
    """Xem hoac cap nhat profile cua user hien tai"""
    user = request.user
    
    # Dam bao user co profile
    if not hasattr(user, 'profile'):
        from .models import UserProfile
        UserProfile.objects.get_or_create(user=user)
    
    if request.method == 'GET':
        # Lay thong tin profile
        license_obj = user.licenses.filter(status=LicenseStatus.ACTIVE).first()
        
        # Lay phone tu UserProfile
        phone = ''
        if hasattr(user, 'profile') and user.profile:
            phone = user.profile.phone or ''
        
        return Response({
            'status': 'success',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'phone': phone,
                'date_joined': user.date_joined,
                'last_login': user.last_login,
            },
            'license': LicenseSerializer(license_obj).data if license_obj else None
        })
    
    elif request.method in ['POST', 'PUT', 'PATCH']:
        # Cap nhat profile
        first_name = request.data.get('first_name')
        last_name = request.data.get('last_name')
        email = request.data.get('email')
        phone = request.data.get('phone')
        country = request.data.get('country')  # Country ISO code (VN, US, etc.)
        
        # Validate email if changed
        if email and email != user.email:
            if User.objects.filter(email=email).exclude(pk=user.pk).exists():
                return Response({
                    'status': 'error',
                    'message': 'Email already in use by another account'
                }, status=status.HTTP_400_BAD_REQUEST)
            user.email = email
        
        if first_name is not None:
            user.first_name = first_name
        if last_name is not None:
            user.last_name = last_name
        
        # Luu phone vao UserProfile va cap nhat country
        if hasattr(user, 'profile'):
            if phone is not None:
                user.profile.phone = phone
            
            # Cap nhat country - uu tien country gui truc tiep, neu khong co thi parse tu phone
            from .models import get_country_from_phone, get_country_flag
            if country:
                # Client gui country truc tiep
                user.profile.country = country.upper()
                user.profile.country_flag = get_country_flag(country)
            elif phone:
                # Parse country tu phone code
                detected_country = get_country_from_phone(phone)
                if detected_country:
                    user.profile.country = detected_country
                    user.profile.country_flag = get_country_flag(detected_country)
            
            user.profile.save()
        
        user.save()
        
        # Lay phone de tra ve
        saved_phone = ''
        if hasattr(user, 'profile') and user.profile:
            saved_phone = user.profile.phone or ''
        
        # Lay country info de tra ve
        saved_country = ''
        saved_country_flag = ''
        if hasattr(user, 'profile') and user.profile:
            saved_country = user.profile.country or ''
            saved_country_flag = user.profile.country_flag or ''
        
        return Response({
            'status': 'success',
            'message': 'Profile updated successfully',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'phone': saved_phone,
                'country': saved_country,
                'country_flag': saved_country_flag,
            }
        })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def change_password_view(request):
    """Doi mat khau"""
    serializer = ChangePasswordSerializer(data=request.data)
    
    if serializer.is_valid():
        user = request.user
        
        if not user.check_password(serializer.validated_data['old_password']):
            return Response({
                'status': 'error',
                'message': 'Current password is incorrect'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user.set_password(serializer.validated_data['new_password'])
        user.save()
        
        return Response({
            'status': 'success',
            'message': 'Password changed successfully'
        })
    
    return Response({
        'status': 'error',
        'errors': serializer.errors
    }, status=status.HTTP_400_BAD_REQUEST)


# ============ FORGOT PASSWORD VIEWS ============

@api_view(['POST'])
@permission_classes([AllowAny])
def forgot_password_view(request):
    """
    Step 1: Gui ma xac nhan den email de reset password
    """
    email = request.data.get('email', '').strip()
    language = request.data.get('language', 'en').lower()  # Ngon ngu tu client
    
    if not email:
        return Response({
            'status': 'error',
            'message': 'Email is required'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Tim user theo email (case-insensitive)
    try:
        user = User.objects.get(email__iexact=email)
    except User.DoesNotExist:
        # Khong tiet lo email co ton tai hay khong vi ly do bao mat
        # Nhung van tra ve success de tranh enumeration attack
        return Response({
            'status': 'success',
            'message': 'If an account exists with this email, a verification code has been sent.'
        })
    
    import random
    import string
    
    # Tao ma xac nhan 6 so
    verification_code = ''.join(random.choices(string.digits, k=6))
    
    # Xoa cac token cu
    PasswordResetToken.objects.filter(user=user).delete()
    
    # Tao token moi voi expire 15 phut
    token = PasswordResetToken.objects.create(
        user=user,
        verification_code=verification_code,
        expires_at=timezone.now() + timedelta(minutes=15)
    )
    
    # Send email (HTML format with proper encoding)
    try:
        from django.core.mail import EmailMessage
        from django.conf import settings
        
        if language == 'vi':
            subject = 'Ma xac nhan dat lai mat khau - Trading Bot'
            html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
<div style="max-width: 600px; margin: 0 auto; padding: 20px;">
<h2 style="color: #2196F3;">Dat lai mat khau</h2>
<p>Xin chao <strong>{user.username}</strong>,</p>
<p>Ban da yeu cau dat lai mat khau cho tai khoan Trading Bot.</p>
<div style="background-color: #f5f5f5; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;">
<p style="margin: 0; font-size: 14px; color: #666;">Ma xac nhan cua ban la:</p>
<p style="font-size: 32px; font-weight: bold; color: #2196F3; letter-spacing: 5px; margin: 10px 0;">{verification_code}</p>
</div>
<p style="color: #f44336;">Ma nay se het han sau <strong>15 phut</strong>.</p>
<p>Neu ban khong yeu cau dieu nay, vui long bo qua email nay.</p>
<hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
<p style="font-size: 12px; color: #999;">Tran trong,<br>Trading Bot Team</p>
</div>
</body>
</html>"""
        else:
            subject = 'Password Reset Verification Code - Trading Bot'
            html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
<div style="max-width: 600px; margin: 0 auto; padding: 20px;">
<h2 style="color: #2196F3;">Password Reset</h2>
<p>Hello <strong>{user.username}</strong>,</p>
<p>You have requested to reset your password for your Trading Bot account.</p>
<div style="background-color: #f5f5f5; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;">
<p style="margin: 0; font-size: 14px; color: #666;">Your verification code is:</p>
<p style="font-size: 32px; font-weight: bold; color: #2196F3; letter-spacing: 5px; margin: 10px 0;">{verification_code}</p>
</div>
<p style="color: #f44336;">This code will expire in <strong>15 minutes</strong>.</p>
<p>If you did not request this, please ignore this email.</p>
<hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
<p style="font-size: 12px; color: #999;">Best regards,<br>Trading Bot Team</p>
</div>
</body>
</html>"""
        
        email_msg = EmailMessage(
            subject=subject,
            body=html_content,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[user.email],
        )
        email_msg.content_subtype = 'html'
        email_msg.send(fail_silently=False)
        
    except Exception as e:
        print(f"[ForgotPassword] Error sending email: {e}", flush=True)
        return Response({
            'status': 'error',
            'message': 'Failed to send verification email. Please try again later.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return Response({
        'status': 'success',
        'message': 'Verification code has been sent to your email.'
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def verify_reset_code_view(request):
    """
    Step 2: Xac nhan ma de cho phep dat lai mat khau
    """
    email = request.data.get('email', '').strip()
    code = request.data.get('code', '').strip()
    
    if not email or not code:
        return Response({
            'status': 'error',
            'message': 'Email and verification code are required'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Tim user (case-insensitive)
    try:
        user = User.objects.get(email__iexact=email)
    except User.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Invalid email or code'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Tim token
    try:
        token = PasswordResetToken.objects.get(
            user=user,
            verification_code=code,
            is_used=False
        )
    except PasswordResetToken.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Invalid or expired verification code'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Kiem tra het han
    if token.expires_at and token.expires_at < timezone.now():
        return Response({
            'status': 'error',
            'message': 'Verification code has expired. Please request a new one.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Ma hop le - khong danh dau da dung o day, doi den khi reset password
    return Response({
        'status': 'success',
        'message': 'Verification code is valid. You can now reset your password.'
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def reset_password_view(request):
    """
    Step 3: Dat lai mat khau moi
    """
    email = request.data.get('email', '').strip()
    code = request.data.get('code', '').strip()
    new_password = request.data.get('new_password', '')
    new_password_confirm = request.data.get('new_password_confirm', '')
    
    if not email or not code or not new_password:
        return Response({
            'status': 'error',
            'message': 'Email, code, and new password are required'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if len(new_password) < 6:
        return Response({
            'status': 'error',
            'message': 'Password must be at least 6 characters'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if new_password != new_password_confirm:
        return Response({
            'status': 'error',
            'message': 'Passwords do not match'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Tim user (case-insensitive)
    try:
        user = User.objects.get(email__iexact=email)
    except User.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Invalid email or code'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Tim va xac nhan token
    try:
        token = PasswordResetToken.objects.get(
            user=user,
            verification_code=code,
            is_used=False
        )
    except PasswordResetToken.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Invalid or expired verification code'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Kiem tra het han
    if token.expires_at and token.expires_at < timezone.now():
        return Response({
            'status': 'error',
            'message': 'Verification code has expired. Please request a new one.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Doi mat khau
    user.set_password(new_password)
    user.save()
    
    # Danh dau token da su dung
    token.is_used = True
    token.save()
    
    return Response({
        'status': 'success',
        'message': 'Password has been reset successfully. You can now login with your new password.'
    })


# ============ LICENSE VIEWS ============

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_licenses_view(request):
    """Lay danh sach licenses cua user"""
    licenses = request.user.licenses.all()
    return Response({
        'licenses': LicenseDetailSerializer(licenses, many=True).data
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def activate_license_view(request):
    """Activate license voi hardware ID - khong can dang nhap"""
    serializer = LicenseActivateSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    data = serializer.validated_data
    license_key = data['license_key'].upper().replace(' ', '')
    hardware_id = data['hardware_id']
    session_id = data.get('session_id')  # Get session_id from request
    
    # Tim license
    try:
        license_obj = License.objects.get(license_key=license_key)
    except License.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Invalid license key'
        }, status=status.HTTP_404_NOT_FOUND)
    
    # Kiem tra license valid
    if not license_obj.is_valid():
        return Response({
            'status': 'error',
            'message': 'License is expired or inactive',
            'license_status': license_obj.status,
            'expire_date': license_obj.expire_date
        }, status=status.HTTP_403_FORBIDDEN)
    
    # Kiem tra xem device da activate chua
    existing_activation = DeviceActivation.objects.filter(
        license=license_obj,
        hardware_id=hardware_id
    ).first()
    
    if existing_activation:
        # Device da activate - check if we can register session
        if session_id:
            # Try to register this session
            if not existing_activation.register_session(session_id):
                # Another session is active - reject
                return Response({
                    'status': 'error',
                    'message': 'Another instance is already running with this account on this device.',
                    'error_code': 'MULTIPLE_INSTANCES'
                }, status=status.HTTP_403_FORBIDDEN)
        
        # Update last_seen
        existing_activation.last_seen = timezone.now()
        existing_activation.last_heartbeat = timezone.now()
        existing_activation.ip_address = get_client_ip(request)
        existing_activation.is_active = True
        
        if data.get('device_name'):
            existing_activation.device_name = data['device_name']
        if data.get('os_info'):
            existing_activation.os_info = data['os_info']
        
        existing_activation.save()
        
        log_usage(license_obj, UsageLog.EventType.ACTIVATE, existing_activation, request, {
            'action': 'reactivate',
            'app_version': data.get('app_version', '')
        })
        
        # Trigger GitHub update check on login (non-blocking)
        trigger_github_update_check()
        
        return Response({
            'status': 'success',
            'message': 'Device reactivated successfully',
            'license': LicenseSerializer(license_obj).data,
            'device': DeviceActivationSerializer(existing_activation).data
        })
    
    # Kiem tra co the activate them device khong
    if not license_obj.can_activate_device():
        return Response({
            'status': 'error',
            'message': f'Maximum devices ({license_obj.max_devices}) reached. Please deactivate another device first.',
            'max_devices': license_obj.max_devices,
            'active_devices': license_obj.active_device_count()
        }, status=status.HTTP_403_FORBIDDEN)
    
    # Tao activation moi
    with transaction.atomic():
        activation = DeviceActivation.objects.create(
            license=license_obj,
            hardware_id=hardware_id,
            device_name=data.get('device_name', ''),
            os_info=data.get('os_info', ''),
            ip_address=get_client_ip(request),
            last_heartbeat=timezone.now()
        )
        
        # Register session if provided
        if session_id:
            activation.register_session(session_id)
        
        # Update activated_at neu chua co
        if not license_obj.activated_at:
            license_obj.activated_at = timezone.now()
            license_obj.save()
        
        log_usage(license_obj, UsageLog.EventType.ACTIVATE, activation, request, {
            'action': 'new_activation',
            'app_version': data.get('app_version', '')
        })
    
    # Trigger GitHub update check on login (non-blocking)
    trigger_github_update_check()
    
    return Response({
        'status': 'success',
        'message': 'License activated successfully',
        'license': LicenseSerializer(license_obj).data,
        'device': DeviceActivationSerializer(activation).data
    }, status=status.HTTP_201_CREATED)


@api_view(['POST'])
@permission_classes([AllowAny])
def validate_license_view(request):
    """Validate license - kiem tra khong can activate"""
    serializer = LicenseValidateSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    license_key = serializer.validated_data['license_key'].upper().replace(' ', '')
    hardware_id = serializer.validated_data.get('hardware_id')
    
    try:
        license_obj = License.objects.get(license_key=license_key)
    except License.DoesNotExist:
        return Response({
            'status': 'error',
            'valid': False,
            'message': 'Invalid license key'
        }, status=status.HTTP_404_NOT_FOUND)
    
    is_valid = license_obj.is_valid()
    
    response_data = {
        'status': 'success' if is_valid else 'error',
        'valid': is_valid,
        'license': {
            'license_type': license_obj.license_type,
            'status': license_obj.status,
            'expire_date': license_obj.expire_date.isoformat() if license_obj.expire_date else None,
            'days_remaining': license_obj.days_remaining(),
            'features': license_obj.features,
            'is_valid': is_valid,
        }
    }
    
    # Neu co hardware_id, check xem device co duoc authorize khong
    if hardware_id:
        device = DeviceActivation.objects.filter(
            license=license_obj,
            hardware_id=hardware_id,
            is_active=True
        ).first()
        
        response_data['device_authorized'] = device is not None
        if device:
            response_data['device'] = DeviceActivationSerializer(device).data
    
    log_usage(license_obj, UsageLog.EventType.LICENSE_CHECK, request=request)
    
    # Gui email canh bao neu license sap het han (3, 2, 1 ngay)
    _check_and_send_expiry_warning(license_obj)
    
    return Response(response_data)


# ============ DEBUG: Check license online status ============
@api_view(['POST'])
@permission_classes([AllowAny])
def check_license_online_status(request):
    """Debug endpoint - check if license is currently online"""
    license_key = request.data.get('license_key', '').upper().replace(' ', '')
    
    if not license_key:
        return Response({'error': 'license_key required'}, status=400)
    
    try:
        license_obj = License.objects.get(license_key=license_key)
    except License.DoesNotExist:
        return Response({'error': 'License not found'}, status=404)
    
    online_threshold = timezone.now() - timezone.timedelta(seconds=45)
    
    devices = DeviceActivation.objects.filter(
        license=license_obj,
        is_active=True
    )
    
    device_status = []
    for d in devices:
        is_online = d.last_heartbeat and d.last_heartbeat >= online_threshold
        device_status.append({
            'hardware_id': d.hardware_id[:16] + '...',
            'device_name': d.device_name,
            'last_heartbeat': d.last_heartbeat.isoformat() if d.last_heartbeat else None,
            'is_online': is_online,
            'active_sessions': len(d.active_sessions) if d.active_sessions else 0,
        })
    
    return Response({
        'license_key': license_key[:10] + '...',
        'server_time': timezone.now().isoformat(),
        'online_threshold': online_threshold.isoformat(),
        'total_devices': devices.count(),
        'devices': device_status,
        'any_online': any(d['is_online'] for d in device_status)
    })
# ============ END DEBUG ============


@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([HeartbeatThrottle])
def heartbeat_view(request):
    """Heartbeat tu client - cap nhat last_seen va verify license"""
    serializer = HeartbeatSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    data = serializer.validated_data
    license_key = data['license_key'].upper().replace(' ', '')
    hardware_id = data['hardware_id']
    session_id = data.get('session_id')  # Get session_id from request
    
    try:
        license_obj = License.objects.get(license_key=license_key)
    except License.DoesNotExist:
        return Response({
            'status': 'error',
            'valid': False,
            'message': 'Invalid license key'
        }, status=status.HTTP_404_NOT_FOUND)
    
    # Check device first (DO NOT check license validity here!)
    device = DeviceActivation.objects.filter(
        license=license_obj,
        hardware_id=hardware_id,
        is_active=True
    ).first()
    
    if not device:
        return Response({
            'status': 'error',
            'valid': False,
            'message': 'Device not activated for this license'
        }, status=status.HTTP_403_FORBIDDEN)
    
    # SECURITY: Check if ANOTHER device is online (2 may khac nhau)
    # Neu co device khac online gan day, coi nhu device nay bi kick
    online_threshold = timezone.now() - timezone.timedelta(seconds=45)
    other_online_devices = DeviceActivation.objects.filter(
        license=license_obj,
        is_active=True,
        last_heartbeat__gte=online_threshold
    ).exclude(hardware_id=hardware_id)
    
    if other_online_devices.exists():
        other_device = other_online_devices.first()
        print(f"[Heartbeat] SECURITY: Another device is online! This: {hardware_id[:16]}, Other: {other_device.hardware_id[:16]}", flush=True)
        print(f"[Heartbeat] This device last_heartbeat: {device.last_heartbeat}, Other: {other_device.last_heartbeat}", flush=True)
        
        # CASE 1: Device nay da bi clear heartbeat (force login tu device khac)
        # -> Device nay bi kick
        if not device.last_heartbeat:
            print(f"[Heartbeat] SECURITY: This device has no heartbeat (was cleared by force login)!", flush=True)
            return Response({
                'status': 'error',
                'valid': False,
                'session_kicked': True,
                'message': 'Your session was closed because you logged in on another device.',
                'error_code': 'SESSION_KICKED',
                'other_device': other_device.device_name or 'Unknown Device'
            }, status=status.HTTP_403_FORBIDDEN)
        
        # CASE 2: Ca 2 device deu co heartbeat - device nao moi hon thi thang
        if other_device.last_heartbeat > device.last_heartbeat:
            print(f"[Heartbeat] SECURITY: This device was superseded by another device!", flush=True)
            return Response({
                'status': 'error',
                'valid': False,
                'session_kicked': True,
                'message': 'Your session was closed because you logged in on another device.',
                'error_code': 'SESSION_KICKED',
                'other_device': other_device.device_name or 'Unknown Device'
            }, status=status.HTTP_403_FORBIDDEN)
    
    # Update device heartbeat - keep device marked as online
    device.last_heartbeat = timezone.now()
    device.last_seen = timezone.now()
    device.ip_address = get_client_ip(request)
    
    # � SECURITY: Check session still valid (wasn't kicked by another login)
    session_valid = True
    if session_id:
        session_valid = device.update_session_heartbeat(session_id)
        if not session_valid:
            print(f"[Heartbeat] 🚨 SECURITY: Session {session_id[:16]}... was kicked! Returning session_kicked=true", flush=True)
            return Response({
                'status': 'error',
                'valid': False,
                'session_kicked': True,
                'message': 'Your session was closed because you logged in elsewhere.',
                'error_code': 'SESSION_KICKED'
            }, status=status.HTTP_403_FORBIDDEN)
    
    # 🔧 Ensure is_online_now is True (in case it wasn't set properly)
    if not device.is_online_now:
        device.is_online_now = True
        device.online_since = timezone.now()
    
    device.save(update_fields=['last_heartbeat', 'last_seen', 'ip_address', 'is_online_now', 'online_since'])
    print(f"[Heartbeat] ✅ Device {hardware_id[:16]}... | is_online_now: {device.is_online_now}", flush=True)
    
    # Log heartbeat (chi log moi 5 phut de giam DB load)
    last_log = UsageLog.objects.filter(
        license=license_obj,
        device=device,
        event_type=UsageLog.EventType.HEARTBEAT
    ).order_by('-timestamp').first()
    
    should_log = not last_log or (timezone.now() - last_log.timestamp).total_seconds() > 300
    
    if should_log:
        log_usage(license_obj, UsageLog.EventType.HEARTBEAT, device, request, {
            'app_version': data.get('app_version', ''),
            'trading_stats': data.get('trading_stats', {})
        })
    
    # FIX: LUON tra ve success voi license data de client detect thay doi
    # Client can biet license status de update UI, ke ca khi expired hoac renewed
    is_valid = license_obj.is_valid()
    
    return Response({
        'status': 'success',
        'valid': is_valid,
        'license': {
            'days_remaining': license_obj.days_remaining(),
            'expire_date': license_obj.expire_date.isoformat() if license_obj.expire_date else None,
            'is_valid': is_valid,
            'license_type': license_obj.license_type,
            'features': license_obj.features,
        },
        'server_time': timezone.now()
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def deactivate_device_view(request):
    """Deactivate mot device"""
    serializer = DeactivateDeviceSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    license_key = serializer.validated_data['license_key'].upper().replace(' ', '')
    hardware_id = serializer.validated_data['hardware_id']
    session_id = serializer.validated_data.get('session_id')  # Get session_id to clean up
    
    try:
        license_obj = License.objects.get(license_key=license_key)
    except License.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Invalid license key'
        }, status=status.HTTP_404_NOT_FOUND)
    
    device = DeviceActivation.objects.filter(
        license=license_obj,
        hardware_id=hardware_id
    ).first()
    
    if not device:
        return Response({
            'status': 'error',
            'message': 'Device not found'
        }, status=status.HTTP_404_NOT_FOUND)
    
    # Remove session if provided
    if session_id:
        device.remove_session(session_id)
    
    device.is_active = False
    device.save()
    
    log_usage(license_obj, UsageLog.EventType.DEACTIVATE, device, request)
    
    return Response({
        'status': 'success',
        'message': 'Device deactivated successfully',
        'active_devices': license_obj.active_device_count()
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def end_session_view(request):
    """
    End a session - mark device as offline.
    Called when app closes normally to allow login from other instances.
    """
    license_key = request.data.get('license_key', '').upper().replace(' ', '')
    hardware_id = request.data.get('hardware_id', '')
    session_id = request.data.get('session_id', '')  # 🔧 Get session_id to clear
    
    if not license_key or not hardware_id:
        return Response({
            'status': 'error',
            'message': 'Missing required fields: license_key, hardware_id'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        license_obj = License.objects.get(license_key=license_key)
    except License.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Invalid license key'
        }, status=status.HTTP_404_NOT_FOUND)
    
    device = DeviceActivation.objects.filter(
        license=license_obj,
        hardware_id=hardware_id,
        is_active=True
    ).first()
    
    if not device:
        return Response({
            'status': 'error',
            'message': 'Active device not found'
        }, status=status.HTTP_404_NOT_FOUND)
    
    # 🔧 Remove session_id from active_sessions nếu có
    if session_id:
        device.remove_session(session_id)
        print(f"[End Session] Removed session {session_id[:16]}...", flush=True)
    
    # Mark device as offline by clearing last_heartbeat
    device.last_heartbeat = None
    device.save(update_fields=['last_heartbeat'])
    print(f"[End Session] Device {hardware_id[:16]}... marked as OFFLINE", flush=True)
    
    return Response({
        'status': 'success',
        'message': 'Session ended successfully - device marked offline'
    })


# ============ SUBSCRIPTION PLAN VIEWS ============

@api_view(['GET'])
@permission_classes([AllowAny])
def subscription_plans_view(request):
    """Lay danh sach cac goi subscription"""
    plans = SubscriptionPlan.objects.filter(is_active=True)
    return Response({
        'plans': SubscriptionPlanSerializer(plans, many=True).data
    })


# ============ EMAIL VERIFICATION VIEWS ============

@api_view(['GET'])
@permission_classes([AllowAny])
def verify_email_view(request, token):
    """Xac thuc email - kich hoat tai khoan"""
    try:
        verification = EmailVerificationToken.objects.get(token=token)
        
        if not verification.is_valid():
            return Response({
                'status': 'error',
                'message': 'Link xac thuc da het han hoac da duoc su dung.',
                'expired': True
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Kich hoat user
        user = verification.user
        user.is_active = True
        user.save()
        
        # Danh dau token da su dung
        verification.is_used = True
        verification.save()
        
        return Response({
            'status': 'success',
            'message': f'Account already has an active license.',
            'username': user.username,
            'email': user.email
        })
        
    except EmailVerificationToken.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Link xac thuc khong hop le.'
        }, status=status.HTTP_404_NOT_FOUND)


@api_view(['POST'])
@permission_classes([AllowAny])
def resend_verification_view(request):
    """Gui lai email xac thuc"""
    email = request.data.get('email', '').lower()
    
    if not email:
        return Response({
            'status': 'error',
            'message': 'Please enter username and activation code.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        user = User.objects.get(email=email, is_active=False)
        
        # Xoa token cu va tao moi
        EmailVerificationToken.objects.filter(user=user).delete()
        token = EmailVerificationToken.objects.create(user=user)
        token.send_verification_email()
        
        return Response({
            'status': 'success',
            'message': 'Email xac thuc da duoc gui lai. Vui long kiem tra hop thu.'
        })
    except User.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Email khong ton tai hoac tai khoan da duoc kich hoat.'
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Khong the gui email: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ============ LEGACY VIEWS ============

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def check_subscription_view(request):
    """LEGACY: Check subscription status"""
    # Uu tien dung License system moi
    license_obj = request.user.licenses.filter(status=LicenseStatus.ACTIVE).first()
    
    if license_obj:
        return Response({
            'active': license_obj.is_valid(),
            'expire_date': license_obj.expire_date,
            'days_remaining': license_obj.days_remaining(),
            'license_type': license_obj.license_type,
            'features': license_obj.features,
        })
    
    # Fallback to legacy Subscription
    try:
        sub = request.user.subscription
        return Response({
            'active': sub.is_active(),
            'expire_date': sub.expire_date
        })
    except Subscription.DoesNotExist:
        return Response({
            'active': False,
            'message': 'No subscription found'
        }, status=status.HTTP_404_NOT_FOUND)


# ============ REAL-TIME NOTIFICATION ENDPOINTS ============

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def watch_user_changes(request):
    """
    Long polling endpoint - Client cho thong bao thay doi tu server.
    Server giu connection toi da 30 giay, tra ve ngay neu co thay doi.
    
    Client goi endpoint nay lien tuc de nhan thong bao real-time.
    """
    import time
    from .models import UserChangeNotification
    
    user = request.user
    timeout = int(request.GET.get('timeout', 30))  # Max 30 seconds
    last_check = request.GET.get('last_check')  # ISO timestamp of last check
    
    # Parse last_check timestamp
    last_check_dt = None
    if last_check:
        try:
            from datetime import datetime
            last_check_dt = datetime.fromisoformat(last_check.replace('Z', '+00:00'))
        except:
            pass
    
    # Poll for changes (check every 1 second for up to timeout seconds)
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check for undelivered notifications
        query = UserChangeNotification.objects.filter(
            user=user,
            is_delivered=False
        )
        if last_check_dt:
            query = query.filter(created_at__gt=last_check_dt)
        
        notifications = query.order_by('created_at')[:10]
        
        if notifications.exists():
            # Mark as delivered and return
            notification_list = []
            for notif in notifications:
                notification_list.append({
                    'id': notif.id,
                    'type': notif.change_type,
                    'data': notif.change_data,
                    'timestamp': notif.created_at.isoformat()
                })
                notif.is_delivered = True
                notif.save()
            
            return Response({
                'status': 'success',
                'has_changes': True,
                'notifications': notification_list,
                'server_time': timezone.now().isoformat()
            })
        
        # Sleep 1 second before next check
        time.sleep(1)
    
    # Timeout - no changes
    return Response({
        'status': 'success',
        'has_changes': False,
        'notifications': [],
        'server_time': timezone.now().isoformat()
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def mark_notifications_read(request):
    """Mark notifications as delivered/read"""
    from .models import UserChangeNotification
    
    notification_ids = request.data.get('notification_ids', [])
    
    if notification_ids:
        UserChangeNotification.objects.filter(
            user=request.user,
            id__in=notification_ids
        ).update(is_delivered=True)
    
    return Response({'status': 'success'})


# ============ AI SERVER VERIFY API ============

@api_view(['POST'])
@permission_classes([AllowAny])
def verify_user_license(request):
    """
    Verify user license for AI Server
    
    Supports verification by:
    - username (required)
    - license_key (optional)
    - hardware_id (optional)
    
    Returns user info and license status for AI Server dashboard
    """
    # Safe string handling - handle None values explicitly
    username = (request.data.get('username') or '').strip()
    license_key = (request.data.get('license_key') or '').strip().upper().replace(' ', '')
    hardware_id = (request.data.get('hardware_id') or '').strip()
    check_only = request.data.get('check_only', True)  # Don't activate, just check
    
    if not username:
        return Response({
            'status': 'error',
            'valid': False,
            'message': 'Username is required'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Find user
    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        return Response({
            'status': 'error',
            'valid': False,
            'message': f'User not found: {username}'
        }, status=status.HTTP_404_NOT_FOUND)
    
    # Find license for this user
    license_query = License.objects.filter(user=user)
    
    # If license_key provided, filter by it
    if license_key:
        license_query = license_query.filter(license_key=license_key)
    
    # Get the most recent active license
    license_obj = license_query.filter(
        status__in=[LicenseStatus.ACTIVE, LicenseStatus.EXPIRED]
    ).order_by('-created_at').first()
    
    if not license_obj:
        # Check if user has any license at all
        any_license = License.objects.filter(user=user).first()
        return Response({
            'status': 'error',
            'valid': False,
            'message': 'No active license found for this user',
            'user': {
                'username': user.username,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'email': user.email,
            },
            'license': None
        }, status=status.HTTP_404_NOT_FOUND)
    
    # Check if license is valid
    is_valid = license_obj.is_valid()
    
    # Check device authorization if hardware_id provided
    device_authorized = False
    if hardware_id:
        device = DeviceActivation.objects.filter(
            license=license_obj,
            hardware_id=hardware_id,
            is_active=True
        ).first()
        device_authorized = device is not None
    
    # Log the check
    log_usage(license_obj, UsageLog.EventType.LICENSE_CHECK, request=request, event_data={
        'source': 'ai_server',
        'check_only': check_only
    })
    
    return Response({
        'status': 'success' if is_valid else 'error',
        'valid': is_valid,
        'user': {
            'username': user.username,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'email': user.email,
        },
        'license': {
            'license_key': license_obj.license_key,
            'license_type': license_obj.license_type,
            'status': license_obj.status,
            'expire_date': license_obj.expire_date.isoformat() if license_obj.expire_date else None,
            'days_remaining': license_obj.days_remaining(),
            'features': license_obj.features,
            'is_valid': is_valid,
        },
        'device_authorized': device_authorized if hardware_id else None,
    })


# ============ HEALTH CHECK / PING ============

@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def ping_view(request):
    """
    Simple ping endpoint for keep-alive and health check.
    Used by watchdog and monitoring systems.
    """
    from datetime import datetime
    import time
    
    # Get basic server stats
    try:
        user_count = User.objects.count()
        license_count = License.objects.count()
        active_licenses = License.objects.filter(status=LicenseStatus.ACTIVE).count()
    except:
        user_count = 0
        license_count = 0
        active_licenses = 0
    
    return Response({
        'status': 'pong',
        'timestamp': datetime.now().isoformat(),
        'server': 'license_server',
        'version': '1.0.0',
        'stats': {
            'users': user_count,
            'licenses': license_count,
            'active_licenses': active_licenses,
        }
    })


# ============ GITHUB WEBHOOK ============

@api_view(['POST'])
@permission_classes([AllowAny])
def github_webhook_view(request):
    """
    GitHub Webhook endpoint - triggered when new release is published
    
    Setup on GitHub:
    1. Go to repo Settings > Webhooks > Add webhook
    2. Payload URL: https://your-server.com/api/webhook/github/
    3. Content type: application/json
    4. Secret: (set in settings.py as GITHUB_WEBHOOK_SECRET)
    5. Events: Select "Releases" only
    """
    import hmac
    import hashlib
    from django.conf import settings
    
    # Verify webhook signature (optional but recommended)
    github_secret = getattr(settings, 'GITHUB_WEBHOOK_SECRET', None)
    
    if github_secret:
        signature = request.headers.get('X-Hub-Signature-256', '')
        if signature:
            expected = 'sha256=' + hmac.new(
                github_secret.encode('utf-8'),
                request.body,
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected):
                logger.warning("[GITHUB-WEBHOOK] Invalid signature")
                return Response({'status': 'error', 'message': 'Invalid signature'}, status=403)
    
    # Check event type
    event_type = request.headers.get('X-GitHub-Event', '')
    
    if event_type != 'release':
        # Ignore non-release events (ping, push, etc.)
        return Response({'status': 'ok', 'message': f'Ignored event: {event_type}'})
    
    # Parse payload
    payload = request.data
    action = payload.get('action', '')
    
    # Only process published releases
    if action != 'published':
        return Response({'status': 'ok', 'message': f'Ignored action: {action}'})
    
    release = payload.get('release', {})
    version = release.get('tag_name', '').lstrip('v')
    release_notes = release.get('body', '')
    download_url = release.get('html_url', '')
    
    if not version:
        return Response({'status': 'error', 'message': 'No version found'}, status=400)
    
    logger.info(f"[GITHUB-WEBHOOK] New release received: v{version}")
    
    # Process in background thread (non-blocking)
    def _process_release():
        try:
            from .models import AppUpdateNotification
            from django.utils import timezone
            from django.core.mail import send_mail
            from django.conf import settings as django_settings
            from .models import License, EmailTemplate
            
            # Check if already processed
            existing = AppUpdateNotification.objects.filter(version=version).first()
            
            if existing and existing.is_sent:
                logger.info(f"[GITHUB-WEBHOOK] Version {version} already sent, skipping")
                return
            
            # Create or update notification
            if not existing:
                existing = AppUpdateNotification.objects.create(
                    version=version,
                    release_notes_vi=release_notes,
                    release_notes_en=release_notes,
                    download_url=download_url,
                )
            
            # Send emails to all active users
            active_user_ids = License.objects.filter(status='active').values_list('user_id', flat=True)
            recipients = User.objects.filter(id__in=active_user_ids, email__isnull=False).exclude(email='')
            
            logger.info(f"[GITHUB-WEBHOOK] Sending v{version} to {recipients.count()} users...")
            
            template = EmailTemplate.get_default_template('app_update')
            sent_count = 0
            
            for user in recipients:
                try:
                    context = {
                        'username': user.username,
                        'first_name': user.first_name or user.username,
                        'app_version': version,
                        'release_notes': release_notes,
                        'download_url': download_url,
                    }
                    
                    if template:
                        subject, content = template.render(context, 'vi')
                    else:
                        subject = f'🚀 Trading Bot v{version} - Bản cập nhật mới!'
                        content = f"""
                        <html><body style="font-family: Arial, sans-serif;">
                        <h2 style="color: #10b981;">🚀 Phiên bản mới: v{version}</h2>
                        <p>Xin chào <strong>{user.first_name or user.username}</strong>,</p>
                        <p>Trading Bot vừa phát hành phiên bản mới!</p>
                        <p><a href="{download_url}" style="background:#10b981;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;">⬇️ Tải ngay</a></p>
                        </body></html>
                        """
                    
                    send_mail(
                        subject=subject,
                        message='',
                        from_email=django_settings.DEFAULT_FROM_EMAIL,
                        recipient_list=[user.email],
                        html_message=content,
                        fail_silently=False
                    )
                    sent_count += 1
                    
                except Exception as e:
                    logger.error(f"[GITHUB-WEBHOOK] Failed {user.email}: {e}")
            
            existing.is_sent = True
            existing.sent_at = timezone.now()
            existing.sent_count = sent_count
            existing.save()
            
            logger.info(f"[GITHUB-WEBHOOK] ✅ Sent to {sent_count} users")
            
        except Exception as e:
            logger.error(f"[GITHUB-WEBHOOK] Error: {e}")
    
    # Run in background
    threading.Thread(target=_process_release, daemon=True).start()
    
    return Response({
        'status': 'ok',
        'message': f'Processing release v{version}',
        'version': version
    })
