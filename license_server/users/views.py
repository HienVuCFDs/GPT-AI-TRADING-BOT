"""
License Server Views - API Endpoints
"""
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
    PasswordResetToken
)
from .serializers import (
    RegisterSerializer, UserSerializer, ChangePasswordSerializer,
    LicenseSerializer, LicenseDetailSerializer, DeviceActivationSerializer,
    LicenseActivateSerializer, LicenseValidateSerializer, HeartbeatSerializer,
    DeactivateDeviceSerializer, SubscriptionPlanSerializer, SubscriptionSerializer
)


# ============ THROTTLING ============

class RegisterThrottle(AnonRateThrottle):
    rate = '100/hour'  # Giá»›i háº¡n 5 Ä‘Äƒng kÃ½ / giá»

class LoginThrottle(AnonRateThrottle):
    rate = '10/minute'  # Giá»›i háº¡n 10 login / phÃºt

class HeartbeatThrottle(UserRateThrottle):
    rate = '60/minute'  # Heartbeat 1 láº§n / giÃ¢y max


# ============ HELPER FUNCTIONS ============

def get_client_ip(request):
    """Láº¥y IP cá»§a client"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def log_usage(license_obj, event_type, device=None, request=None, event_data=None):
    """Helper Ä‘á»ƒ log usage"""
    UsageLog.objects.create(
        license=license_obj,
        device=device,
        event_type=event_type,
        event_data=event_data or {},
        ip_address=get_client_ip(request) if request else None,
        user_agent=request.META.get('HTTP_USER_AGENT', '')[:500] if request else ''
    )


# ============ AUTH VIEWS ============

def send_activation_code_email(user, activation_code, language='vi'):
    """Gửi email chứa mã kích hoạt cho user - Hỗ trợ song ngữ"""
    from django.core.mail import send_mail
    from django.conf import settings
    import threading

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
    """ÄÄƒng kÃ½ user má»›i - Tá»± Ä‘á»™ng gá»­i mÃ£ kÃ­ch hoáº¡t qua email"""
    serializer = RegisterSerializer(data=request.data)
    language = request.data.get('language', 'vi').lower()  # NgÃ´n ngá»¯ tá»« client
    
    if serializer.is_valid():
        user = serializer.save()
        
        # User active ngay nhÆ°ng chÆ°a cÃ³ License
        user.is_active = True
        user.save()
        
        # Tá»± Ä‘á»™ng táº¡o mÃ£ kÃ­ch hoáº¡t má»›i cho user nÃ y
        activation_code = ActivationCode.objects.create(
            trial_days=7,
            max_uses=1,
            is_active=True
        )
        # MÃ£ Ä‘Æ°á»£c tá»± Ä‘á»™ng generate trong model save()
        print(f"ðŸ“ Created activation code: {activation_code.code} for user: {user.username}")
        
        email_sent = False
        
        print(f"ðŸ“§ User email: '{user.email}' (type: {type(user.email)})")
        if user.email:
            # Gá»­i email vá»›i mÃ£ kÃ­ch hoáº¡t (async) - truyá»n language
            send_activation_code_email(user, activation_code, language)
            email_sent = True
            print(f"âœ… Email function called for {user.email}")
        else:
            print(f"âš ï¸ No email found for user {user.username}")
        
        # ThÃ´ng bÃ¡o theo ngÃ´n ngá»¯
        if language == 'en':
            success_msg = 'Registration successful! Please check your email for activation code.' if email_sent else 'Registration successful! Please contact Admin to get activation code.'
        else:
            success_msg = 'ÄÄƒng kÃ½ thÃ nh cÃ´ng! Vui lÃ²ng kiá»ƒm tra email Ä‘á»ƒ láº¥y mÃ£ kÃ­ch hoáº¡t.' if email_sent else 'ÄÄƒng kÃ½ thÃ nh cÃ´ng! Vui lÃ²ng liÃªn há»‡ Admin Ä‘á»ƒ Ä‘Æ°á»£c cáº¥p mÃ£ kÃ­ch hoáº¡t.'
        
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
    """KÃ­ch hoáº¡t tÃ i khoáº£n báº±ng mÃ£ 6 kÃ½ tá»± - Cáº¥p Trial 7 ngÃ y"""
    username = request.data.get('username', '').strip()
    code = request.data.get('code', '').strip().upper()
    
    if not username or not code:
        return Response({
            'status': 'error',
            'message': 'Please enter username and activation code.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # TÃ¬m user
    try:
        user = User.objects.get(username=username.lower())
    except User.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Account not found.'
        }, status=status.HTTP_404_NOT_FOUND)
    
    # Kiá»ƒm tra user Ä‘Ã£ cÃ³ license chÆ°a
    existing_license = user.licenses.filter(status=LicenseStatus.ACTIVE).first()
    if existing_license:
        return Response({
            'status': 'error',
            'message': 'Account already has an active license.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # TÃ¬m mÃ£ kÃ­ch hoáº¡t
    try:
        activation = ActivationCode.objects.get(code=code)
    except ActivationCode.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Invalid activation code.'
        }, status=status.HTTP_404_NOT_FOUND)
    
    # Kiá»ƒm tra mÃ£ cÃ²n dÃ¹ng Ä‘Æ°á»£c khÃ´ng
    if not activation.is_valid():
        return Response({
            'status': 'error',
            'message': 'Activation code expired or already used.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Táº¡o License Trial
    trial_days = activation.trial_days
    license_obj = License.objects.create(
        user=user,
        license_type=LicenseType.TRIAL,
        status=LicenseStatus.ACTIVE,
        expire_date=timezone.now() + timedelta(days=trial_days),
        max_devices=1,
        note=f'KÃ­ch hoáº¡t báº±ng mÃ£: {code}'
    )
    
    # ÄÃ¡nh dáº¥u mÃ£ Ä‘Ã£ dÃ¹ng
    activation.use()
    
    return Response({
        'status': 'success',
        'message': f'KÃ­ch hoáº¡t thÃ nh cÃ´ng! Báº¡n Ä‘Æ°á»£c dÃ¹ng thá»­ {trial_days} ngÃ y.',
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
    """ÄÄƒng nháº­p vÃ  nháº­n JWT tokens - há»— trá»£ username, email hoáº·c sá»‘ Ä‘iá»‡n thoáº¡i"""
    login_id = request.data.get('username', '').strip()  # CÃ³ thá»ƒ lÃ  username, email hoáº·c phone
    password = request.data.get('password', '')
    
    if not login_id or not password:
        return Response({
            'status': 'error',
            'message': 'Username/Email/Phone and password are required'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    user = None
    
    # Thá»­ tÃ¬m user theo nhiá»u cÃ¡ch
    # 1. TÃ¬m theo username (case-insensitive)
    try:
        user = User.objects.get(username__iexact=login_id)
    except User.DoesNotExist:
        pass
    
    # 2. TÃ¬m theo email (case-insensitive)
    if not user:
        try:
            user = User.objects.get(email__iexact=login_id)
        except User.DoesNotExist:
            pass
    
    # 3. TÃ¬m theo sá»‘ Ä‘iá»‡n thoáº¡i (trong UserProfile)
    if not user:
        from .models import UserProfile
        try:
            # Chuáº©n hÃ³a sá»‘ Ä‘iá»‡n thoáº¡i - loáº¡i bá» khoáº£ng tráº¯ng vÃ  dáº¥u
            phone_normalized = login_id.replace(' ', '').replace('-', '').replace('.', '')
            profile = UserProfile.objects.get(phone=phone_normalized)
            user = profile.user
        except UserProfile.DoesNotExist:
            # Thá»­ tÃ¬m vá»›i sá»‘ Ä‘iá»‡n thoáº¡i gá»‘c
            try:
                profile = UserProfile.objects.get(phone=login_id)
                user = profile.user
            except UserProfile.DoesNotExist:
                pass
    
    # XÃ¡c thá»±c máº­t kháº©u
    if user and user.check_password(password):
        pass  # User há»£p lá»‡
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
    
    # Láº¥y license active
    license_obj = user.licenses.filter(status=LicenseStatus.ACTIVE).first()

    # Nếu không có license active, tìm license expired mới nhất
    if not license_obj:
        license_obj = user.licenses.filter(status=LicenseStatus.EXPIRED).order_by('-expire_date').first()

    # Nếu vẫn không có license nào
    if not license_obj:
        # Vẫn tạo token để user có thể tạo đơn thanh toán
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

    # Kiá»ƒm tra expired
    if not license_obj.is_valid():
        license_obj.status = LicenseStatus.EXPIRED
        license_obj.save()
        
        # Vẫn tạo token để user có thể gia hạn
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
            'license': LicenseSerializer(license_obj).data,
            'license_status': 'expired',
            'message': 'License has expired. Please renew.'
        })    # Táº¡o tokens
    refresh = RefreshToken.for_user(user)
    
    # Log login
    log_usage(license_obj, UsageLog.EventType.LOGIN, request=request)
    
    return Response({
        'status': 'success',
        'access': str(refresh.access_token),
        'refresh': str(refresh),
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
        },
        'license': LicenseSerializer(license_obj).data
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    """Logout - blacklist refresh token"""
    try:
        refresh_token = request.data.get('refresh')
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()
        
        # Log logout
        license_obj = request.user.licenses.filter(status=LicenseStatus.ACTIVE).first()
        if license_obj:
            log_usage(license_obj, UsageLog.EventType.LOGOUT, request=request)
        
        return Response({'status': 'success', 'message': 'Logged out successfully'})
    except Exception as e:
        return Response({'status': 'success', 'message': 'Logged out'})


@api_view(['GET', 'POST', 'PUT', 'PATCH'])
@permission_classes([IsAuthenticated])
def profile_view(request):
    """Xem hoáº·c cáº­p nháº­t profile cá»§a user hiá»‡n táº¡i"""
    user = request.user
    
    # Äáº£m báº£o user cÃ³ profile
    if not hasattr(user, 'profile'):
        from .models import UserProfile
        UserProfile.objects.get_or_create(user=user)
    
    if request.method == 'GET':
        # Láº¥y thÃ´ng tin profile
        license_obj = user.licenses.filter(status=LicenseStatus.ACTIVE).first()
        
        # Láº¥y phone tá»« UserProfile
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
        # Cáº­p nháº­t profile
        first_name = request.data.get('first_name')
        last_name = request.data.get('last_name')
        email = request.data.get('email')
        phone = request.data.get('phone')
        
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
        
        # LÆ°u phone vÃ o UserProfile
        if phone is not None and hasattr(user, 'profile'):
            user.profile.phone = phone
            user.profile.save()
        
        user.save()
        
        # Láº¥y phone Ä‘á»ƒ tráº£ vá»
        saved_phone = ''
        if hasattr(user, 'profile') and user.profile:
            saved_phone = user.profile.phone or ''
        
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
            }
        })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def change_password_view(request):
    """Äá»•i máº­t kháº©u"""
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
    Step 1: Gá»­i mÃ£ xÃ¡c nháº­n Ä‘áº¿n email Ä‘á»ƒ reset password
    """
    email = request.data.get('email', '').strip()
    language = request.data.get('language', 'en').lower()  # NgÃ´n ngá»¯ tá»« client
    
    if not email:
        return Response({
            'status': 'error',
            'message': 'Email is required'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # TÃ¬m user theo email (case-insensitive)
    try:
        user = User.objects.get(email__iexact=email)
    except User.DoesNotExist:
        # KhÃ´ng tiáº¿t lá»™ email cÃ³ tá»“n táº¡i hay khÃ´ng vÃ¬ lÃ½ do báº£o máº­t
        # NhÆ°ng váº«n tráº£ vá» success Ä‘á»ƒ trÃ¡nh enumeration attack
        return Response({
            'status': 'success',
            'message': 'If an account exists with this email, a verification code has been sent.'
        })
    
    import random
    import string
    
    # Táº¡o mÃ£ xÃ¡c nháº­n 6 sá»‘
    verification_code = ''.join(random.choices(string.digits, k=6))
    
    # XÃ³a cÃ¡c token cÅ©
    PasswordResetToken.objects.filter(user=user).delete()
    
    # Táº¡o token má»›i vá»›i expire 15 phÃºt
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
            subject = 'Mã xác nhận đặt lại mật khẩu - Trading Bot'
            html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
<div style="max-width: 600px; margin: 0 auto; padding: 20px;">
<h2 style="color: #2196F3;">Đặt lại mật khẩu</h2>
<p>Xin chào <strong>{user.username}</strong>,</p>
<p>Bạn đã yêu cầu đặt lại mật khẩu cho tài khoản Trading Bot.</p>
<div style="background-color: #f5f5f5; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;">
<p style="margin: 0; font-size: 14px; color: #666;">Mã xác nhận của bạn là:</p>
<p style="font-size: 32px; font-weight: bold; color: #2196F3; letter-spacing: 5px; margin: 10px 0;">{verification_code}</p>
</div>
<p style="color: #f44336;">Mã này sẽ hết hạn sau <strong>15 phút</strong>.</p>
<p>Nếu bạn không yêu cầu điều này, vui lòng bỏ qua email này.</p>
<hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
<p style="font-size: 12px; color: #999;">Trân trọng,<br>Trading Bot Team</p>
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
        print(f"[ForgotPassword] Error sending email: {e}")
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
    Step 2: XÃ¡c nháº­n mÃ£ Ä‘á»ƒ cho phÃ©p Ä‘áº·t láº¡i máº­t kháº©u
    """
    email = request.data.get('email', '').strip()
    code = request.data.get('code', '').strip()
    
    if not email or not code:
        return Response({
            'status': 'error',
            'message': 'Email and verification code are required'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # TÃ¬m user (case-insensitive)
    try:
        user = User.objects.get(email__iexact=email)
    except User.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Invalid email or code'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # TÃ¬m token
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
    
    # Kiá»ƒm tra háº¿t háº¡n
    if token.expires_at and token.expires_at < timezone.now():
        return Response({
            'status': 'error',
            'message': 'Verification code has expired. Please request a new one.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # MÃ£ há»£p lá»‡ - khÃ´ng Ä‘Ã¡nh dáº¥u Ä‘Ã£ dÃ¹ng á»Ÿ Ä‘Ã¢y, Ä‘á»£i Ä‘áº¿n khi reset password
    return Response({
        'status': 'success',
        'message': 'Verification code is valid. You can now reset your password.'
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def reset_password_view(request):
    """
    Step 3: Äáº·t láº¡i máº­t kháº©u má»›i
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
    
    # TÃ¬m user (case-insensitive)
    try:
        user = User.objects.get(email__iexact=email)
    except User.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Invalid email or code'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # TÃ¬m vÃ  xÃ¡c nháº­n token
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
    
    # Kiá»ƒm tra háº¿t háº¡n
    if token.expires_at and token.expires_at < timezone.now():
        return Response({
            'status': 'error',
            'message': 'Verification code has expired. Please request a new one.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Äá»•i máº­t kháº©u
    user.set_password(new_password)
    user.save()
    
    # ÄÃ¡nh dáº¥u token Ä‘Ã£ sá»­ dá»¥ng
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
    """Láº¥y danh sÃ¡ch licenses cá»§a user"""
    licenses = request.user.licenses.all()
    return Response({
        'licenses': LicenseDetailSerializer(licenses, many=True).data
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def activate_license_view(request):
    """Activate license vá»›i hardware ID - khÃ´ng cáº§n Ä‘Äƒng nháº­p"""
    serializer = LicenseActivateSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    data = serializer.validated_data
    license_key = data['license_key'].upper().replace(' ', '')
    hardware_id = data['hardware_id']
    
    # TÃ¬m license
    try:
        license_obj = License.objects.get(license_key=license_key)
    except License.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Invalid license key'
        }, status=status.HTTP_404_NOT_FOUND)
    
    # Kiá»ƒm tra license valid
    if not license_obj.is_valid():
        return Response({
            'status': 'error',
            'message': 'License is expired or inactive',
            'license_status': license_obj.status,
            'expire_date': license_obj.expire_date
        }, status=status.HTTP_403_FORBIDDEN)
    
    # Kiá»ƒm tra xem device Ä‘Ã£ activate chÆ°a
    existing_activation = DeviceActivation.objects.filter(
        license=license_obj,
        hardware_id=hardware_id
    ).first()
    
    if existing_activation:
        # Device Ä‘Ã£ activate - update last_seen
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
        
        return Response({
            'status': 'success',
            'message': 'Device reactivated successfully',
            'license': LicenseSerializer(license_obj).data,
            'device': DeviceActivationSerializer(existing_activation).data
        })
    
    # Kiá»ƒm tra cÃ³ thá»ƒ activate thÃªm device khÃ´ng
    if not license_obj.can_activate_device():
        return Response({
            'status': 'error',
            'message': f'Maximum devices ({license_obj.max_devices}) reached. Please deactivate another device first.',
            'max_devices': license_obj.max_devices,
            'active_devices': license_obj.active_device_count()
        }, status=status.HTTP_403_FORBIDDEN)
    
    # Táº¡o activation má»›i
    with transaction.atomic():
        activation = DeviceActivation.objects.create(
            license=license_obj,
            hardware_id=hardware_id,
            device_name=data.get('device_name', ''),
            os_info=data.get('os_info', ''),
            ip_address=get_client_ip(request),
            last_heartbeat=timezone.now()
        )
        
        # Update activated_at náº¿u chÆ°a cÃ³
        if not license_obj.activated_at:
            license_obj.activated_at = timezone.now()
            license_obj.save()
        
        log_usage(license_obj, UsageLog.EventType.ACTIVATE, activation, request, {
            'action': 'new_activation',
            'app_version': data.get('app_version', '')
        })
    
    return Response({
        'status': 'success',
        'message': 'License activated successfully',
        'license': LicenseSerializer(license_obj).data,
        'device': DeviceActivationSerializer(activation).data
    }, status=status.HTTP_201_CREATED)


@api_view(['POST'])
@permission_classes([AllowAny])
def validate_license_view(request):
    """Validate license - kiá»ƒm tra khÃ´ng cáº§n activate"""
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
    
    # Náº¿u cÃ³ hardware_id, check xem device cÃ³ Ä‘Æ°á»£c authorize khÃ´ng
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
    
    return Response(response_data)


@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([HeartbeatThrottle])
def heartbeat_view(request):
    """Heartbeat tá»« client - cáº­p nháº­t last_seen vÃ  verify license"""
    serializer = HeartbeatSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    data = serializer.validated_data
    license_key = data['license_key'].upper().replace(' ', '')
    hardware_id = data['hardware_id']
    
    try:
        license_obj = License.objects.get(license_key=license_key)
    except License.DoesNotExist:
        return Response({
            'status': 'error',
            'valid': False,
            'message': 'Invalid license key'
        }, status=status.HTTP_404_NOT_FOUND)
    
    # Check license valid
    if not license_obj.is_valid():
        return Response({
            'status': 'error',
            'valid': False,
            'message': 'License expired or inactive',
            'license_status': license_obj.status
        }, status=status.HTTP_403_FORBIDDEN)
    
    # Check device
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
    
    # Update heartbeat
    device.last_heartbeat = timezone.now()
    device.ip_address = get_client_ip(request)
    device.save(update_fields=['last_heartbeat', 'last_seen', 'ip_address'])
    
    # Log heartbeat (chá»‰ log má»—i 5 phÃºt Ä‘á»ƒ giáº£m DB load)
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
    
    return Response({
        'status': 'success',
        'valid': True,
        'license': {
            'days_remaining': license_obj.days_remaining(),
            'expire_date': license_obj.expire_date,
            'features': license_obj.features,
        },
        'server_time': timezone.now()
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def deactivate_device_view(request):
    """Deactivate má»™t device"""
    serializer = DeactivateDeviceSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    license_key = serializer.validated_data['license_key'].upper().replace(' ', '')
    hardware_id = serializer.validated_data['hardware_id']
    
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
    
    device.is_active = False
    device.save()
    
    log_usage(license_obj, UsageLog.EventType.DEACTIVATE, device, request)
    
    return Response({
        'status': 'success',
        'message': 'Device deactivated successfully',
        'active_devices': license_obj.active_device_count()
    })


# ============ SUBSCRIPTION PLAN VIEWS ============

@api_view(['GET'])
@permission_classes([AllowAny])
def subscription_plans_view(request):
    """Láº¥y danh sÃ¡ch cÃ¡c gÃ³i subscription"""
    plans = SubscriptionPlan.objects.filter(is_active=True)
    return Response({
        'plans': SubscriptionPlanSerializer(plans, many=True).data
    })


# ============ EMAIL VERIFICATION VIEWS ============

@api_view(['GET'])
@permission_classes([AllowAny])
def verify_email_view(request, token):
    """XÃ¡c thá»±c email - kÃ­ch hoáº¡t tÃ i khoáº£n"""
    try:
        verification = EmailVerificationToken.objects.get(token=token)
        
        if not verification.is_valid():
            return Response({
                'status': 'error',
                'message': 'Link xÃ¡c thá»±c Ä‘Ã£ háº¿t háº¡n hoáº·c Ä‘Ã£ Ä‘Æ°á»£c sá»­ dá»¥ng.',
                'expired': True
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # KÃ­ch hoáº¡t user
        user = verification.user
        user.is_active = True
        user.save()
        
        # ÄÃ¡nh dáº¥u token Ä‘Ã£ sá»­ dá»¥ng
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
            'message': 'Link xÃ¡c thá»±c khÃ´ng há»£p lá»‡.'
        }, status=status.HTTP_404_NOT_FOUND)


@api_view(['POST'])
@permission_classes([AllowAny])
def resend_verification_view(request):
    """Gá»­i láº¡i email xÃ¡c thá»±c"""
    email = request.data.get('email', '').lower()
    
    if not email:
        return Response({
            'status': 'error',
            'message': 'Please enter username and activation code.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        user = User.objects.get(email=email, is_active=False)
        
        # XÃ³a token cÅ© vÃ  táº¡o má»›i
        EmailVerificationToken.objects.filter(user=user).delete()
        token = EmailVerificationToken.objects.create(user=user)
        token.send_verification_email()
        
        return Response({
            'status': 'success',
            'message': 'Email xÃ¡c thá»±c Ä‘Ã£ Ä‘Æ°á»£c gá»­i láº¡i. Vui lÃ²ng kiá»ƒm tra há»™p thÆ°.'
        })
    except User.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Email khÃ´ng tá»“n táº¡i hoáº·c tÃ i khoáº£n Ä‘Ã£ Ä‘Æ°á»£c kÃ­ch hoáº¡t.'
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'KhÃ´ng thá»ƒ gá»­i email: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ============ LEGACY VIEWS ============

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def check_subscription_view(request):
    """LEGACY: Check subscription status"""
    # Æ¯u tiÃªn dÃ¹ng License system má»›i
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
    Long polling endpoint - Client chờ thông báo thay đổi từ server.
    Server giữ connection tối đa 30 giây, trả về ngay nếu có thay đổi.
    
    Client gọi endpoint này liên tục để nhận thông báo real-time.
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


