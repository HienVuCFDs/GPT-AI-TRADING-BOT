"""
License Server Models - Quản lý License và Device Activation
"""
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.conf import settings
from datetime import timedelta
import uuid
import secrets


class UserProfile(models.Model):
    """Extended user profile với thông tin bổ sung"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile', verbose_name="Người dùng")
    phone = models.CharField("Điện thoại", max_length=20, blank=True)
    address = models.TextField("Địa chỉ", blank=True)
    country = models.CharField("Quốc gia", max_length=100, blank=True)
    avatar_url = models.URLField("Avatar URL", blank=True)
    bio = models.TextField("Giới thiệu", blank=True)
    
    # Tracking
    created_at = models.DateTimeField("Ngày tạo", auto_now_add=True)
    updated_at = models.DateTimeField("Cập nhật", auto_now=True)
    
    class Meta:
        verbose_name = "Hồ sơ người dùng"
        verbose_name_plural = "Hồ sơ người dùng"
    
    def __str__(self):
        return f"Hồ sơ của {self.user.username}"


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Tự động tạo UserProfile khi User được tạo"""
    if created:
        UserProfile.objects.get_or_create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """Tự động save UserProfile khi User được save"""
    if hasattr(instance, 'profile'):
        instance.profile.save()


class LicenseType(models.TextChoices):
    """Các loại license"""
    TRIAL = 'trial', 'Dùng thử (7 ngày)'
    MONTHLY = 'monthly', 'Tháng'
    QUARTERLY = 'quarterly', 'Quý (3 tháng)'
    YEARLY = 'yearly', 'Năm'
    LIFETIME = 'lifetime', 'Vĩnh viễn'


class LicenseStatus(models.TextChoices):
    """Trạng thái license"""
    ACTIVE = 'active', 'Hoạt động'
    EXPIRED = 'expired', 'Hết hạn'
    SUSPENDED = 'suspended', 'Tạm dừng'
    REVOKED = 'revoked', 'Thu hồi'


class License(models.Model):
    """License model với hardware binding"""
    
    # Core fields
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='licenses', verbose_name="Người dùng")
    license_key = models.CharField("Mã license", max_length=64, unique=True, db_index=True)
    
    # License type và status
    license_type = models.CharField(
        "Loại license",
        max_length=20, 
        choices=LicenseType.choices, 
        default=LicenseType.TRIAL
    )
    status = models.CharField(
        "Trạng thái",
        max_length=20,
        choices=LicenseStatus.choices,
        default=LicenseStatus.ACTIVE
    )
    
    # Thời gian
    created_at = models.DateTimeField("Ngày tạo", auto_now_add=True)
    activated_at = models.DateTimeField("Ngày kích hoạt", null=True, blank=True)
    expire_date = models.DateTimeField("Ngày hết hạn")
    
    # Hardware binding
    max_devices = models.PositiveIntegerField("Số thiết bị tối đa", default=1)
    
    # Features - JSON field để lưu các tính năng được phép
    features = models.JSONField("Tính năng", default=dict, blank=True)
    
    # Metadata
    note = models.TextField("Ghi chú", blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "License"
        verbose_name_plural = "Licenses"
        indexes = [
            models.Index(fields=['license_key', 'status']),
            models.Index(fields=['user', 'status']),
        ]
    
    def __str__(self):
        return f"{self.license_key} ({self.user.username})"
    
    @staticmethod
    def generate_license_key():
        """Tạo license key unique - Format: XXXX-XXXX-XXXX-XXXX"""
        raw = secrets.token_hex(16)
        return '-'.join([raw[i:i+4].upper() for i in range(0, 16, 4)])
    
    def is_valid(self):
        """Kiểm tra license có hợp lệ không"""
        if self.status != LicenseStatus.ACTIVE:
            return False
        if self.expire_date < timezone.now():
            return False
        return True
    
    def days_remaining(self):
        """Số ngày còn lại"""
        if self.expire_date < timezone.now():
            return 0
        delta = self.expire_date - timezone.now()
        return delta.days
    
    def update_status_from_expiry_date(self):
        """
        🔧 FIX: Tự động cập nhật status dựa trên expire_date
        - Nếu hết hạn -> EXPIRED
        - Nếu còn hạn và bị SUSPENDED/REVOKED -> Không thay đổi (yêu cầu admin xử lý)
        - Nếu còn hạn và là EXPIRED -> Chuyển thành ACTIVE
        """
        now = timezone.now()
        
        if self.expire_date < now:
            # Hết hạn
            if self.status != LicenseStatus.EXPIRED:
                self.status = LicenseStatus.EXPIRED
                UserChangeNotification.notify_license_change(
                    self.user, 
                    'license_expired',
                    {'days_remaining': 0}
                )
                print(f"🔄 [AUTO-UPDATE] License {self.license_key} EXPIRED")
        else:
            # Còn hạn
            if self.status == LicenseStatus.EXPIRED:
                # Recover từ EXPIRED thành ACTIVE
                self.status = LicenseStatus.ACTIVE
                UserChangeNotification.notify_license_change(
                    self.user,
                    'license_renewed',
                    {'days_remaining': self.days_remaining()}
                )
                print(f"🔄 [AUTO-UPDATE] License {self.license_key} RENEWED -> ACTIVE")
            # SUSPENDED/REVOKED: Giữ nguyên, không auto-recover
    
    def active_device_count(self):
        """Số device đang active"""
        return self.activations.filter(is_active=True).count()
    
    def can_activate_device(self):
        """Kiểm tra có thể activate thêm device không"""
        return self.active_device_count() < self.max_devices
    
    def save(self, *args, **kwargs):
        if not self.license_key:
            self.license_key = self.generate_license_key()
        
        # 🔧 FIX: Auto-update status dựa trên expiry date
        self.update_status_from_expiry_date()
        
        # Auto-upgrade Trial to Monthly if > 7 days
        # Trial chỉ được phép tối đa 7 ngày
        if self.license_type == LicenseType.TRIAL and self.days_remaining() > 7:
            self.license_type = LicenseType.MONTHLY
        
        # 🔧 FIX: Validate không có tổ hợp vô lý
        # Nếu là TRIAL thì chỉ được up tới 7 ngày
        if self.license_type == LicenseType.TRIAL:
            max_trial_date = self.user.licenses.filter(
                license_type=LicenseType.TRIAL
            ).first()
            if max_trial_date and self.id != max_trial_date.id:
                # User chỉ được 1 Trial
                raise ValueError("User chỉ được 1 Trial license")
        
        # Kiểm tra hạn lý hợp lệ
        if self.days_remaining() > 0 and self.status == LicenseStatus.EXPIRED:
            self.status = LicenseStatus.ACTIVE
        
        super().save(*args, **kwargs)


class DeviceActivation(models.Model):
    """Theo dõi các thiết bị đã kích hoạt license"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    license = models.ForeignKey(License, on_delete=models.CASCADE, related_name='activations', verbose_name="License")
    
    # Device info
    hardware_id = models.CharField("Hardware ID", max_length=128, db_index=True)
    device_name = models.CharField("Tên thiết bị", max_length=255, blank=True)
    os_info = models.CharField("Hệ điều hành", max_length=255, blank=True)
    ip_address = models.GenericIPAddressField("Địa chỉ IP", null=True, blank=True)
    
    # Timestamps
    first_seen = models.DateTimeField("Lần đầu kết nối", auto_now_add=True)
    last_seen = models.DateTimeField("Lần cuối kết nối", auto_now=True)
    last_heartbeat = models.DateTimeField("Heartbeat cuối", null=True, blank=True)
    
    # Status
    is_active = models.BooleanField("Đang hoạt động", default=True)
    
    class Meta:
        unique_together = ['license', 'hardware_id']
        ordering = ['-last_seen']
        verbose_name = "Thiết bị"
        verbose_name_plural = "Thiết bị đã kích hoạt"
    
    def __str__(self):
        return f"{self.device_name or self.hardware_id[:16]} - {self.license.user.username}"
    
    def is_online(self, threshold_minutes=5):
        """Kiểm tra device có online không (heartbeat trong X phút)"""
        if not self.last_heartbeat:
            return False
        threshold = timezone.now() - timezone.timedelta(minutes=threshold_minutes)
        return self.last_heartbeat >= threshold


class UsageLog(models.Model):
    """Log sử dụng để tracking và analytics"""
    
    class EventType(models.TextChoices):
        LOGIN = 'login', 'Đăng nhập'
        LOGOUT = 'logout', 'Đăng xuất'
        HEARTBEAT = 'heartbeat', 'Heartbeat'
        ACTIVATE = 'activate', 'Kích hoạt thiết bị'
        DEACTIVATE = 'deactivate', 'Hủy kích hoạt'
        LICENSE_CHECK = 'license_check', 'Kiểm tra license'
        TRADE = 'trade', 'Giao dịch'
        ERROR = 'error', 'Lỗi'
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    license = models.ForeignKey(License, on_delete=models.CASCADE, related_name='usage_logs', verbose_name="License")
    device = models.ForeignKey(DeviceActivation, on_delete=models.SET_NULL, null=True, blank=True, verbose_name="Thiết bị")
    
    # Event info
    event_type = models.CharField("Loại sự kiện", max_length=50, choices=EventType.choices)
    event_data = models.JSONField("Dữ liệu", default=dict, blank=True)
    
    # Request info
    ip_address = models.GenericIPAddressField("Địa chỉ IP", null=True, blank=True)
    user_agent = models.CharField("User Agent", max_length=500, blank=True)
    
    timestamp = models.DateTimeField("Thời gian", auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Lịch sử hoạt động"
        verbose_name_plural = "Lịch sử hoạt động"
        indexes = [
            models.Index(fields=['license', 'timestamp']),
            models.Index(fields=['event_type', 'timestamp']),
        ]
    
    def __str__(self):
        return f"{self.event_type} - {self.license.user.username} @ {self.timestamp}"


class SubscriptionPlan(models.Model):
    """Các gói subscription có sẵn"""
    
    name = models.CharField("Tên gói", max_length=100)
    name_vi = models.CharField("Tên tiếng Việt", max_length=100, blank=True)
    description = models.TextField("Mô tả", blank=True)
    description_vi = models.TextField("Mô tả tiếng Việt", blank=True)
    
    license_type = models.CharField("Loại license", max_length=20, choices=LicenseType.choices)
    duration_days = models.PositiveIntegerField("Số ngày")  # 0 = lifetime
    
    price = models.DecimalField("Giá", max_digits=10, decimal_places=2)
    currency = models.CharField("Đơn vị tiền", max_length=3, default='USD')
    
    # Features
    max_devices = models.PositiveIntegerField("Số thiết bị tối đa", default=1)
    features = models.JSONField("Tính năng", default=dict, blank=True)
    
    # Display
    is_active = models.BooleanField("Đang hoạt động", default=True)
    is_featured = models.BooleanField("Nổi bật", default=False)
    sort_order = models.PositiveIntegerField("Thứ tự", default=0)
    
    class Meta:
        ordering = ['sort_order', 'price']
        verbose_name = "Gói đăng ký"
        verbose_name_plural = "Gói đăng ký"
    
    def __str__(self):
        return f"{self.name} - ${self.price}/{self.license_type}"


# ============ LEGACY SUPPORT ============
# Giữ lại Subscription model cũ để backward compatibility
class Subscription(models.Model):
    """LEGACY: Subscription model cũ - sẽ migrate sang License"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='subscription')
    expire_date = models.DateTimeField()
    
    def is_active(self):
        return self.expire_date >= timezone.now()
    
    class Meta:
        verbose_name = "[Legacy] Subscription"


# ============ ACTIVATION CODE ============
class ActivationCode(models.Model):
    """Mã kích hoạt 6 ký tự - Admin tạo, User nhập để nhận Trial 7 ngày"""
    code = models.CharField("Mã kích hoạt", max_length=6, unique=True, db_index=True)
    created_at = models.DateTimeField("Ngày tạo", auto_now_add=True)
    expires_at = models.DateTimeField("Hết hạn", null=True, blank=True)
    max_uses = models.PositiveIntegerField("Số lần dùng tối đa", default=1)
    used_count = models.PositiveIntegerField("Đã sử dụng", default=0)
    trial_days = models.PositiveIntegerField("Số ngày Trial", default=7)
    is_active = models.BooleanField("Đang hoạt động", default=True)
    note = models.TextField("Ghi chú", blank=True)
    
    class Meta:
        verbose_name = "Mã kích hoạt"
        verbose_name_plural = "Mã kích hoạt"
        ordering = ['-created_at']
    
    @staticmethod
    def generate_code():
        """Tạo mã 6 ký tự ngẫu nhiên (chữ + số)"""
        import random
        import string
        chars = string.ascii_uppercase + string.digits
        # Loại bỏ ký tự dễ nhầm: 0, O, I, L, 1
        chars = chars.replace('0', '').replace('O', '').replace('I', '').replace('L', '').replace('1', '')
        return ''.join(random.choices(chars, k=6))
    
    def is_valid(self):
        """Kiểm tra mã còn dùng được không"""
        if not self.is_active:
            return False
        if self.expires_at and self.expires_at < timezone.now():
            return False
        if self.used_count >= self.max_uses:
            return False
        return True
    
    def use(self):
        """Tăng số lần sử dụng"""
        self.used_count += 1
        if self.used_count >= self.max_uses:
            self.is_active = False
        self.save()
    
    def save(self, *args, **kwargs):
        if not self.code:
            # Tạo mã unique
            for _ in range(10):
                code = self.generate_code()
                if not ActivationCode.objects.filter(code=code).exists():
                    self.code = code
                    break
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.code} ({self.used_count}/{self.max_uses})"


# ============ EMAIL VERIFICATION ============
class EmailVerificationToken(models.Model):
    """Token để xác thực email khi đăng ký"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='email_token')
    token = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_used = models.BooleanField(default=False)
    
    class Meta:
        verbose_name = "Token xác thực email"
        verbose_name_plural = "Token xác thực email"
    
    def save(self, *args, **kwargs):
        if not self.expires_at:
            self.expires_at = timezone.now() + timedelta(hours=24)  # Token hết hạn sau 24h
        super().save(*args, **kwargs)
    
    def is_valid(self):
        """Kiểm tra token còn hợp lệ không"""
        return not self.is_used and self.expires_at > timezone.now()
    
    def send_verification_email(self):
        """Gửi email xác thực tài khoản"""
        verification_url = f"{settings.SITE_DOMAIN}/api/auth/verify-email/{self.token}/"
        
        # Render HTML email
        html_message = render_to_string('emails/verify_email.html', {
            'user': self.user,
            'verification_url': verification_url,
            'expires_hours': 24,
        })
        plain_message = strip_tags(html_message)
        
        send_mail(
            subject='🚀 Xác thực tài khoản Trading Bot',
            message=plain_message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[self.user.email],
            html_message=html_message,
            fail_silently=False,
        )
        return True
    
    def __str__(self):
        return f"Token for {self.user.username} - {'Valid' if self.is_valid() else 'Invalid'}"


class PasswordResetToken(models.Model):
    """Token để reset mật khẩu khi quên"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='password_reset_tokens')
    verification_code = models.CharField("Mã xác nhận", max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_used = models.BooleanField(default=False)
    
    class Meta:
        verbose_name = "Token reset mật khẩu"
        verbose_name_plural = "Token reset mật khẩu"
    
    def save(self, *args, **kwargs):
        if not self.expires_at:
            self.expires_at = timezone.now() + timedelta(minutes=15)  # Token hết hạn sau 15 phút
        super().save(*args, **kwargs)
    
    def is_valid(self):
        """Kiểm tra token còn hợp lệ không"""
        return not self.is_used and self.expires_at > timezone.now()
    
    def __str__(self):
        return f"Reset token for {self.user.username} - {'Valid' if self.is_valid() else 'Invalid'}"


class PricingPlan(models.Model):
    """Gói giá subscription"""
    name = models.CharField("Tên gói", max_length=100)
    duration_months = models.PositiveIntegerField("Số tháng")
    price_usd = models.DecimalField("Giá USD", max_digits=10, decimal_places=2)
    price_vnd = models.PositiveIntegerField("Giá VND")
    description = models.TextField("Mô tả", blank=True)
    features = models.JSONField("Tính năng", default=list, blank=True)
    is_active = models.BooleanField("Hoạt động", default=True)
    created_at = models.DateTimeField("Ngày tạo", auto_now_add=True)
    
    class Meta:
        verbose_name = "Gói giá"
        verbose_name_plural = "Gói giá"
        ordering = ['duration_months']
    
    def __str__(self):
        return f"{self.name} - ${self.price_usd}"


class Payment(models.Model):
    """Lịch sử thanh toán"""
    PAYMENT_STATUS = [
        ('pending', 'Chờ xử lý'),
        ('completed', 'Thành công'),
        ('failed', 'Thất bại'),
        ('cancelled', 'Hủy'),
        ('refunded', 'Hoàn tiền'),
    ]
    
    PAYMENT_METHOD = [
        ('payos', 'PayOS (Bank)'),
        ('crypto', 'Crypto (USDT)'),
        ('manual', 'Thủ công'),
    ]
    
    license = models.ForeignKey(License, on_delete=models.CASCADE, related_name='payments', verbose_name="License")
    pricing_plan = models.ForeignKey(PricingPlan, on_delete=models.SET_NULL, null=True, verbose_name="Gói giá")
    
    amount_vnd = models.PositiveIntegerField("Số tiền VND", default=0)
    amount_usd = models.DecimalField("Số tiền USD", max_digits=10, decimal_places=2, default=0)
    
    order_code = models.CharField("Mã đơn", max_length=100, unique=True, db_index=True)
    transaction_id = models.CharField("Mã giao dịch", max_length=255, blank=True)
    
    status = models.CharField("Trạng thái", max_length=20, choices=PAYMENT_STATUS, default='pending')
    payment_method = models.CharField("Phương thức", max_length=20, choices=PAYMENT_METHOD, default='payos')
    
    payos_payment_link_id = models.CharField("PayOS Link ID", max_length=255, blank=True)
    
    note = models.TextField("Ghi chú", blank=True)
    
    created_at = models.DateTimeField("Ngày tạo", auto_now_add=True)
    updated_at = models.DateTimeField("Cập nhật", auto_now=True)
    paid_at = models.DateTimeField("Ngày thanh toán", null=True, blank=True)
    
    class Meta:
        verbose_name = "Thanh toán"
        verbose_name_plural = "Thanh toán"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Payment {self.order_code} - {self.status}"
    
    def send_success_notification(self):
        """
        🎉 Gửi email thông báo thanh toán thành công + gia hạn license
        """
        if not self.license or not self.pricing_plan:
            print(f"⚠️ Cannot send success email: missing license or pricing plan")
            return False
        
        user = self.license.user
        
        try:
            # Tính ngày hết hạn mới
            if self.pricing_plan.duration_months == 0:
                expire_date = timezone.now() + timedelta(days=36500)
                duration_text = "Vĩnh viễn (Lifetime)"
            else:
                expire_date = self.license.expire_date
                months = self.pricing_plan.duration_months
                if months == 1:
                    duration_text = "1 tháng"
                elif months == 3:
                    duration_text = "3 tháng (Quý)"
                elif months == 12:
                    duration_text = "12 tháng (Năm)"
                else:
                    duration_text = f"{months} tháng"
            
            # Tạo HTML email
            html_message = f"""
            <html>
            <head><meta charset="UTF-8"></head>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <!-- Header -->
                    <div style="background: linear-gradient(135deg, #28a745, #20c997); padding: 30px; border-radius: 10px 10px 0 0; text-align: center;">
                        <h1 style="color: white; margin: 0; font-size: 28px;">🎉 Thanh Toán Thành Công!</h1>
                        <p style="color: #e8f5e9; margin: 10px 0 0 0; font-size: 16px;">Cảm ơn bạn đã gia hạn gói cước</p>
                    </div>
                    
                    <!-- Content -->
                    <div style="background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px;">
                        <p>Xin chào <strong>{user.first_name or user.username}</strong>,</p>
                        
                        <p>Chúng tôi vui mừng thông báo rằng thanh toán của bạn đã được xử lý thành công! 🎊</p>
                        
                        <!-- Order Details -->
                        <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #28a745;">
                            <h3 style="color: #28a745; margin-top: 0;">📋 Chi tiết đơn hàng</h3>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px 0; color: #666;">Mã đơn:</td>
                                    <td style="padding: 10px 0; font-weight: bold; text-align: right;">{self.order_code}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px 0; color: #666;">Gói cước:</td>
                                    <td style="padding: 10px 0; font-weight: bold; text-align: right;">{self.pricing_plan.name}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px 0; color: #666;">Thời hạn:</td>
                                    <td style="padding: 10px 0; font-weight: bold; text-align: right;">{duration_text}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px 0; color: #666;">Số tiền:</td>
                                    <td style="padding: 10px 0; font-weight: bold; text-align: right;">
                                        {self.amount_vnd:,} ₫ / ${self.amount_usd}
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px 0; color: #666;">Ngày thanh toán:</td>
                                    <td style="padding: 10px 0; font-weight: bold; text-align: right;">{self.paid_at.strftime('%d/%m/%Y %H:%M') if self.paid_at else 'N/A'}</td>
                                </tr>
                            </table>
                        </div>
                        
                        <!-- License Info -->
                        <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #007bff;">
                            <h3 style="color: #007bff; margin-top: 0;">📱 Thông tin License</h3>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px 0; color: #666;">License Key:</td>
                                    <td style="padding: 10px 0; font-family: monospace; text-align: right; word-break: break-all;">{self.license.license_key}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px 0; color: #666;">Loại License:</td>
                                    <td style="padding: 10px 0; font-weight: bold; text-align: right;">{self.license.get_license_type_display()}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px 0; color: #666;">Hết hạn:</td>
                                    <td style="padding: 10px 0; font-weight: bold; text-align: right; color: #28a745;">
                                        {expire_date.strftime('%d/%m/%Y')}
                                    </td>
                                </tr>
                            </table>
                        </div>
                        
                        <!-- Next Steps -->
                        <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #ffc107;">
                            <h4 style="color: #856404; margin-top: 0;">⚡ Bước tiếp theo:</h4>
                            <ul style="color: #856404; margin: 10px 0; padding-left: 20px;">
                                <li>License sẽ được kích hoạt tự động trong vòng 5 phút</li>
                                <li>Khởi động lại ứng dụng Trading Bot để cập nhật</li>
                                <li>Kiểm tra trong phần "License Info" để xác nhận</li>
                            </ul>
                        </div>
                        
                        <!-- Support -->
                        <div style="background: #e7f3ff; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center;">
                            <p style="color: #004085; margin: 0;">
                                ❓ Cần hỗ trợ? Liên hệ: <strong>admin@tradingbot.com</strong>
                            </p>
                        </div>
                        
                        <p style="color: #666; text-align: center; margin-top: 30px;">
                            Cảm ơn bạn đã tin tưởng Trading Bot! 🚀
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Plain text version
            plain_message = f"""
Thanh Toán Thành Công!
=====================

Xin chào {user.first_name or user.username},

Chúng tôi vui mừng thông báo rằng thanh toán của bạn đã được xử lý thành công!

CHI TIẾT ĐƠN HÀNG
=================
Mã đơn: {self.order_code}
Gói cước: {self.pricing_plan.name}
Thời hạn: {duration_text}
Số tiền: {self.amount_vnd:,} ₫ / ${self.amount_usd}
Ngày thanh toán: {self.paid_at.strftime('%d/%m/%Y %H:%M') if self.paid_at else 'N/A'}

THÔNG TIN LICENSE
=================
License Key: {self.license.license_key}
Loại License: {self.license.get_license_type_display()}
Hết hạn: {expire_date.strftime('%d/%m/%Y')}

BƯỚC TIẾP THEO
==============
1. License sẽ được kích hoạt tự động trong vòng 5 phút
2. Khởi động lại ứng dụng Trading Bot để cập nhật
3. Kiểm tra trong phần "License Info" để xác nhận

Cảm ơn bạn đã tin tưởng Trading Bot!

---
Hỗ trợ: admin@tradingbot.com
            """
            
            # Gửi email
            send_mail(
                subject=f"🎉 Thanh Toán Thành Công - Gói {self.pricing_plan.name}",
                message=plain_message.strip(),
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                html_message=html_message,
                fail_silently=False,
            )
            
            print(f"✅ Payment success email sent to {user.email}")
            return True
            
        except Exception as e:
            print(f"❌ Error sending payment success email to {user.email}: {e}")
            return False


class UserChangeNotification(models.Model):
    """Track changes to user/license for real-time notifications to client apps"""
    
    CHANGE_TYPES = [
        ('license_updated', 'License Updated'),
        ('license_expired', 'License Expired'),
        ('license_renewed', 'License Renewed'),
        ('license_reduced', 'License Reduced'),
        ('password_changed', 'Password Changed'),
        ('profile_updated', 'Profile Updated'),
        ('force_logout', 'Force Logout'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='change_notifications')
    change_type = models.CharField("Loại thay đổi", max_length=30, choices=CHANGE_TYPES)
    change_data = models.JSONField("Dữ liệu thay đổi", default=dict, blank=True)
    created_at = models.DateTimeField("Thời điểm", auto_now_add=True)
    is_delivered = models.BooleanField("Đã gửi", default=False)
    
    class Meta:
        verbose_name = "Thông báo thay đổi"
        verbose_name_plural = "Thông báo thay đổi"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.change_type} - {self.created_at}"
    
    @classmethod
    def notify_license_change(cls, user, change_type='license_updated', extra_data=None):
        """Create notification when license changes"""
        license_obj = user.licenses.filter(status='active').first()
        if not license_obj:
            license_obj = user.licenses.order_by('-expire_date').first()
        
        data = {
            'expire_date': license_obj.expire_date.isoformat() if license_obj else None,
            'days_remaining': license_obj.days_remaining() if license_obj else 0,
            'is_valid': license_obj.is_valid() if license_obj else False,
            'license_type': license_obj.license_type if license_obj else None,
        }
        if extra_data:
            data.update(extra_data)
        
        return cls.objects.create(
            user=user,
            change_type=change_type,
            change_data=data
        )
