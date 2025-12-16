"""
License Server Serializers - API Serializers
"""
from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.utils import timezone
from datetime import timedelta

from .models import (
    License, DeviceActivation, UsageLog, SubscriptionPlan,
    Subscription, LicenseType, LicenseStatus, UserProfile
)


# ============ USER SERIALIZERS ============

class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer cho UserProfile"""
    class Meta:
        model = UserProfile
        fields = ('phone', 'address', 'country', 'avatar_url', 'bio')


class UserSerializer(serializers.ModelSerializer):
    """Basic user info với profile"""
    phone = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'first_name', 'last_name', 'date_joined', 'phone')
        read_only_fields = ('id', 'date_joined')
    
    def get_phone(self, obj):
        if hasattr(obj, 'profile'):
            return obj.profile.phone
        return ''


class RegisterSerializer(serializers.ModelSerializer):
    """Đăng ký user mới với trial license"""
    password = serializers.CharField(write_only=True, min_length=6)
    password_confirm = serializers.CharField(write_only=True, min_length=6)
    email = serializers.EmailField(required=True)
    phone = serializers.CharField(max_length=20, required=False, allow_blank=True)
    
    class Meta:
        model = User
        fields = ('username', 'password', 'password_confirm', 'email', 'first_name', 'last_name', 'phone')
    
    def validate_username(self, value):
        if len(value) < 3:
            raise serializers.ValidationError("Username must be at least 3 characters")
        if User.objects.filter(username__iexact=value).exists():
            raise serializers.ValidationError("Username already taken")
        return value.lower()
    
    def validate_email(self, value):
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError("Email already registered")
        return value.lower()
    
    def validate(self, data):
        if data.get('password') != data.get('password_confirm'):
            raise serializers.ValidationError({"password_confirm": "Passwords do not match"})
        return data
    
    def create(self, validated_data):
        validated_data.pop('password_confirm', None)
        phone = validated_data.pop('phone', '')
        
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', '')
        )
        
        # Update profile với phone
        if phone and hasattr(user, 'profile'):
            user.profile.phone = phone
            user.profile.save()
        
        # KHÔNG tự động tạo License - Admin sẽ cấp sau
        # User chỉ có tài khoản, chưa có quyền sử dụng
        
        return user


class ChangePasswordSerializer(serializers.Serializer):
    """Đổi mật khẩu"""
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True, min_length=6)
    new_password_confirm = serializers.CharField(required=True)
    
    def validate(self, data):
        if data['new_password'] != data['new_password_confirm']:
            raise serializers.ValidationError({"new_password_confirm": "Passwords do not match"})
        return data


# ============ LICENSE SERIALIZERS ============

class LicenseSerializer(serializers.ModelSerializer):
    """Serializer cho License"""
    days_remaining = serializers.SerializerMethodField()
    is_valid = serializers.SerializerMethodField()
    active_devices = serializers.SerializerMethodField()
    user_email = serializers.SerializerMethodField()
    
    class Meta:
        model = License
        fields = [
            'id', 'license_key', 'license_type', 'status',
            'created_at', 'activated_at', 'expire_date',
            'max_devices', 'features', 'days_remaining', 
            'is_valid', 'active_devices', 'user_email'
        ]
        read_only_fields = ['id', 'license_key', 'created_at', 'user_email']
    
    def get_days_remaining(self, obj):
        return obj.days_remaining()
    
    def get_is_valid(self, obj):
        return obj.is_valid()
    
    def get_active_devices(self, obj):
        return obj.active_device_count()
    
    def get_user_email(self, obj):
        return obj.user.email


class LicenseDetailSerializer(LicenseSerializer):
    """License với danh sách devices"""
    devices = serializers.SerializerMethodField()
    
    class Meta(LicenseSerializer.Meta):
        fields = LicenseSerializer.Meta.fields + ['devices', 'note']
    
    def get_devices(self, obj):
        devices = obj.activations.filter(is_active=True)
        return DeviceActivationSerializer(devices, many=True).data


class DeviceActivationSerializer(serializers.ModelSerializer):
    """Serializer cho device activation"""
    is_online = serializers.SerializerMethodField()
    
    class Meta:
        model = DeviceActivation
        fields = [
            'id', 'hardware_id', 'device_name', 'os_info', 'ip_address',
            'first_seen', 'last_seen', 'last_heartbeat', 'is_active', 'is_online'
        ]
        read_only_fields = ['id', 'first_seen', 'last_seen', 'last_heartbeat']
    
    def get_is_online(self, obj):
        return obj.is_online()


# ============ VALIDATION SERIALIZERS ============

class LicenseActivateSerializer(serializers.Serializer):
    """Activate license với hardware_id"""
    license_key = serializers.CharField(max_length=64)
    hardware_id = serializers.CharField(max_length=128)
    device_name = serializers.CharField(max_length=255, required=False, default='')
    os_info = serializers.CharField(max_length=255, required=False, default='')
    app_version = serializers.CharField(max_length=50, required=False, default='')


class LicenseValidateSerializer(serializers.Serializer):
    """Validate license (check without activation)"""
    license_key = serializers.CharField(max_length=64)
    hardware_id = serializers.CharField(max_length=128, required=False)


class HeartbeatSerializer(serializers.Serializer):
    """Heartbeat từ client để verify license vẫn active"""
    license_key = serializers.CharField(max_length=64)
    hardware_id = serializers.CharField(max_length=128)
    app_version = serializers.CharField(max_length=50, required=False, default='')
    trading_stats = serializers.JSONField(required=False, default=dict)


class DeactivateDeviceSerializer(serializers.Serializer):
    """Deactivate một device"""
    license_key = serializers.CharField(max_length=64)
    hardware_id = serializers.CharField(max_length=128)


# ============ SUBSCRIPTION PLAN SERIALIZERS ============

class SubscriptionPlanSerializer(serializers.ModelSerializer):
    """Các gói subscription có sẵn"""
    
    class Meta:
        model = SubscriptionPlan
        fields = [
            'id', 'name', 'name_vi', 'description', 'description_vi',
            'license_type', 'duration_days', 'price', 'currency',
            'max_devices', 'features', 'is_featured'
        ]


# ============ USAGE LOG SERIALIZERS ============

class UsageLogSerializer(serializers.ModelSerializer):
    """Usage log"""
    
    class Meta:
        model = UsageLog
        fields = [
            'id', 'event_type', 'event_data', 'ip_address', 
            'user_agent', 'timestamp'
        ]
        read_only_fields = ['id', 'timestamp']


# ============ LEGACY SERIALIZERS ============

class SubscriptionSerializer(serializers.ModelSerializer):
    """LEGACY: Subscription serializer"""
    is_active = serializers.SerializerMethodField()
    
    class Meta:
        model = Subscription
        fields = ('expire_date', 'is_active')
    
    def get_is_active(self, obj):
        return obj.is_active()
