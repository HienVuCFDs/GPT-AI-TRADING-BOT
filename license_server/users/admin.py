"""
License Server Admin - Django Admin Configuration
"""
from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from django.utils.html import format_html
from django.utils import timezone

from .models import (
    License, DeviceActivation, UsageLog, 
    SubscriptionPlan, Subscription, UserProfile
)


# ============ CUSTOMIZE ADMIN SITE ============

admin.site.site_header = "⚙️ Admin"
admin.site.site_title = "Admin"
admin.site.index_title = "Trang quản trị hệ thống"


# ============ INLINES ============

class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name = "Thông tin bổ sung"
    verbose_name_plural = "Thông tin bổ sung"


class LicenseInline(admin.TabularInline):
    model = License
    extra = 0
    readonly_fields = ['license_key', 'license_type', 'status', 'expire_date', 'created_at']
    can_delete = False
    show_change_link = True
    verbose_name = "License"
    verbose_name_plural = "Licenses"
    
    def has_add_permission(self, request, obj=None):
        return False


class SubscriptionInline(admin.StackedInline):
    model = Subscription
    can_delete = False
    verbose_name = "Subscription (Cũ)"


class DeviceActivationInline(admin.TabularInline):
    model = DeviceActivation
    extra = 0
    readonly_fields = ['hardware_id', 'device_name', 'os_info', 'ip_address', 'first_seen', 'last_seen', 'is_active']
    can_delete = True
    show_change_link = True
    verbose_name = "Thiết bị"
    verbose_name_plural = "Thiết bị đã kích hoạt"
    
    def has_add_permission(self, request, obj=None):
        return False


# ============ USER ADMIN ============

class UserAdminWithLicenses(UserAdmin):
    inlines = (UserProfileInline, LicenseInline, SubscriptionInline,)
    list_display = ['username', 'email', 'get_fullname', 'get_phone', 'license_count', 'is_staff', 'date_joined']
    
    def get_fullname(self, obj):
        return f"{obj.first_name} {obj.last_name}".strip() or "-"
    get_fullname.short_description = 'Họ tên'
    
    def get_phone(self, obj):
        if hasattr(obj, 'profile') and obj.profile.phone:
            return obj.profile.phone
        return "-"
    get_phone.short_description = 'Điện thoại'
    
    def license_count(self, obj):
        count = obj.licenses.filter(status='active').count()
        return format_html('<span style="color: green;">{}</span>', count) if count else '0'
    license_count.short_description = 'License hoạt động'


# ============ USER PROFILE ADMIN ============

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'phone', 'country', 'created_at']
    search_fields = ['user__username', 'user__email', 'phone']
    list_filter = ['country', 'created_at']
    
    class Meta:
        verbose_name = "Hồ sơ người dùng"
        verbose_name_plural = "Hồ sơ người dùng"


# ============ LICENSE ADMIN ============

@admin.register(License)
class LicenseAdmin(admin.ModelAdmin):
    list_display = [
        'license_key_display', 'user', 'license_type', 'status_badge',
        'expire_date', 'days_remaining_display', 'active_devices_display', 'created_at'
    ]
    list_filter = ['license_type', 'status', 'created_at']
    search_fields = ['license_key', 'user__username', 'user__email']
    readonly_fields = ['id', 'license_key', 'created_at', 'activated_at']
    ordering = ['-created_at']
    inlines = [DeviceActivationInline]
    
    fieldsets = (
        ('License Info', {
            'fields': ('id', 'license_key', 'user', 'license_type', 'status')
        }),
        ('Validity', {
            'fields': ('created_at', 'activated_at', 'expire_date')
        }),
        ('Device Settings', {
            'fields': ('max_devices', 'features')
        }),
        ('Notes', {
            'fields': ('note',),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['activate_licenses', 'suspend_licenses', 'extend_30_days']
    
    def license_key_display(self, obj):
        return obj.license_key[:20] + '...'
    license_key_display.short_description = 'License Key'
    
    def status_badge(self, obj):
        colors = {
            'active': 'green',
            'expired': 'orange', 
            'suspended': 'red',
            'revoked': 'darkred',
        }
        color = colors.get(obj.status, 'gray')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color, obj.status.upper()
        )
    status_badge.short_description = 'Status'
    
    def days_remaining_display(self, obj):
        days = obj.days_remaining()
        if days <= 0:
            return format_html('<span style="color: red;">Expired</span>')
        elif days <= 7:
            return format_html('<span style="color: orange;">{} days</span>', days)
        else:
            return format_html('<span style="color: green;">{} days</span>', days)
    days_remaining_display.short_description = 'Remaining'
    
    def active_devices_display(self, obj):
        count = obj.active_device_count()
        return f"{count}/{obj.max_devices}"
    active_devices_display.short_description = 'Devices'
    
    # Admin Actions
    @admin.action(description='Activate selected licenses')
    def activate_licenses(self, request, queryset):
        queryset.update(status='active')
        # 🔧 FIX: Auto-update status based on expiry
        for license in queryset:
            license.update_status_from_expiry_date()
            license.save()
    
    @admin.action(description='Suspend selected licenses')
    def suspend_licenses(self, request, queryset):
        queryset.update(status='suspended')
    
    @admin.action(description='Extend 30 days')
    def extend_30_days(self, request, queryset):
        from datetime import timedelta
        for license in queryset:
            if license.expire_date < timezone.now():
                license.expire_date = timezone.now() + timedelta(days=30)
            else:
                license.expire_date = license.expire_date + timedelta(days=30)
            # 🔧 FIX: Auto-update status after changing expiry
            license.update_status_from_expiry_date()
            license.save()
            print(f"✅ License {license.license_key} extended to {license.expire_date} - Status: {license.status}")


# ============ DEVICE ADMIN ============

@admin.register(DeviceActivation)
class DeviceActivationAdmin(admin.ModelAdmin):
    list_display = [
        'device_name', 'hardware_id_short', 'license_info', 
        'is_active', 'is_online_display', 'last_seen', 'ip_address'
    ]
    list_filter = ['is_active', 'first_seen']
    search_fields = ['hardware_id', 'device_name', 'license__license_key', 'license__user__username']
    readonly_fields = ['id', 'first_seen', 'last_seen', 'last_heartbeat']
    actions = ['deactivate_devices', 'activate_devices']
    
    def hardware_id_short(self, obj):
        return obj.hardware_id[:16] + '...'
    hardware_id_short.short_description = 'Hardware ID'
    
    def license_info(self, obj):
        return f"{obj.license.user.username} ({obj.license.license_key[:12]}...)"
    license_info.short_description = 'License'
    
    def is_online_display(self, obj):
        if obj.is_online():
            return format_html('<span style="color: green;">● Online</span>')
        return format_html('<span style="color: gray;">○ Offline</span>')
    is_online_display.short_description = 'Status'
    
    @admin.action(description='Deactivate selected devices')
    def deactivate_devices(self, request, queryset):
        queryset.update(is_active=False)
    
    @admin.action(description='Activate selected devices')
    def activate_devices(self, request, queryset):
        queryset.update(is_active=True)


# ============ USAGE LOG ADMIN ============

@admin.register(UsageLog)
class UsageLogAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'event_type', 'license_user', 'device_name', 'ip_address']
    list_filter = ['event_type', 'timestamp']
    search_fields = ['license__user__username', 'license__license_key', 'ip_address']
    readonly_fields = ['id', 'timestamp', 'license', 'device', 'event_type', 'event_data', 'ip_address', 'user_agent']
    ordering = ['-timestamp']
    
    def license_user(self, obj):
        return obj.license.user.username
    license_user.short_description = 'User'
    
    def device_name(self, obj):
        if obj.device:
            return obj.device.device_name or obj.device.hardware_id[:12]
        return '-'
    device_name.short_description = 'Device'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


# ============ SUBSCRIPTION PLAN ADMIN ============

@admin.register(SubscriptionPlan)
class SubscriptionPlanAdmin(admin.ModelAdmin):
    list_display = ['name', 'license_type', 'duration_days', 'price', 'currency', 'max_devices', 'is_active', 'is_featured', 'sort_order']
    list_filter = ['license_type', 'is_active', 'is_featured']
    list_editable = ['is_active', 'is_featured', 'sort_order']
    ordering = ['sort_order', 'price']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('name', 'name_vi', 'description', 'description_vi')
        }),
        ('License Settings', {
            'fields': ('license_type', 'duration_days', 'max_devices', 'features')
        }),
        ('Pricing', {
            'fields': ('price', 'currency')
        }),
        ('Display', {
            'fields': ('is_active', 'is_featured', 'sort_order')
        }),
    )


# ============ LEGACY SUBSCRIPTION ============
# Subscription model đã bị ẩn khỏi admin vì không cần thiết
# Sử dụng License thay thế

# ============ RE-REGISTER USER ============

admin.site.unregister(User)
admin.site.register(User, UserAdminWithLicenses)
