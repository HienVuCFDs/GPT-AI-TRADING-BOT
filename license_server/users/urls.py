"""
License Server URLs - API Endpoints
"""
from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from .views import (
    # Auth
    register_view,
    login_view,
    logout_view,
    change_password_view,
    profile_view,
    
    # Forgot Password
    forgot_password_view,
    verify_reset_code_view,
    reset_password_view,
    
    # Activation Code
    activate_by_code_view,
    
    # Email Verification
    verify_email_view,
    resend_verification_view,
    
    # License
    my_licenses_view,
    activate_license_view,
    validate_license_view,
    heartbeat_view,
    deactivate_device_view,
    
    # Subscription Plans
    subscription_plans_view,
    
    # Legacy
    check_subscription_view,
    
    # Real-time Notifications
    watch_user_changes,
    mark_notifications_read,
)

# Bank Payment (PayOS)
from .payments.bank import (
    get_pricing_plans,
    create_payment,
    payos_webhook,
    check_payment_status,
    payment_history,
    get_usd_rate,
)

# Crypto Payment (NOWPayments)
from .payments.crypto import (
    get_crypto_currencies,
    get_min_amount,
    create_crypto_payment,
    check_crypto_payment_status,
    nowpayments_webhook,
    get_crypto_estimate,
)

urlpatterns = [
    # ============ AUTH ============
    path('auth/register/', register_view, name='register'),
    path('auth/login/', login_view, name='login'),
    path('auth/logout/', logout_view, name='logout'),
    path('auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/change-password/', change_password_view, name='change_password'),
    path('auth/profile/', profile_view, name='profile'),
    
    # ============ FORGOT PASSWORD ============
    path('auth/forgot-password/', forgot_password_view, name='forgot_password'),
    path('auth/verify-reset-code/', verify_reset_code_view, name='verify_reset_code'),
    path('auth/reset-password/', reset_password_view, name='reset_password'),
    
    # ============ ACTIVATION CODE ============
    path('auth/activate-code/', activate_by_code_view, name='activate_by_code'),
    
    # ============ EMAIL VERIFICATION ============
    path('auth/verify-email/<uuid:token>/', verify_email_view, name='verify_email'),
    path('auth/resend-verification/', resend_verification_view, name='resend_verification'),
    
    # ============ LICENSE ============
    path('licenses/', my_licenses_view, name='my_licenses'),
    path('license/activate/', activate_license_view, name='activate_license'),
    path('license/validate/', validate_license_view, name='validate_license'),
    path('license/heartbeat/', heartbeat_view, name='heartbeat'),
    path('license/deactivate-device/', deactivate_device_view, name='deactivate_device'),
    
    # ============ PLANS ============
    path('plans/', subscription_plans_view, name='subscription_plans'),
    
    # ============ PAYMENT (PayOS - Bank) ============
    path('payment/pricing/', get_pricing_plans, name='pricing_plans'),
    path('payment/create/', create_payment, name='create_payment'),
    path('payment/status/<str:order_code>/', check_payment_status, name='payment_status'),
    path('payment/history/', payment_history, name='payment_history'),
    path('payment/usd-rate/', get_usd_rate, name='usd_rate'),
    path('webhook/payos/', payos_webhook, name='payos_webhook'),
    
    # ============ CRYPTO PAYMENT (NOWPayments) ============
    path('crypto/currencies/', get_crypto_currencies, name='crypto_currencies'),
    path('crypto/min-amount/', get_min_amount, name='crypto_min_amount'),
    path('crypto/estimate/', get_crypto_estimate, name='crypto_estimate'),
    path('crypto/create/', create_crypto_payment, name='create_crypto_payment'),
    path('crypto/status/<str:order_id>/', check_crypto_payment_status, name='crypto_payment_status'),
    path('webhook/nowpayments/', nowpayments_webhook, name='nowpayments_webhook'),
    
    # ============ REAL-TIME NOTIFICATIONS ============
    path('user/watch/', watch_user_changes, name='watch_user_changes'),
    path('user/notifications/read/', mark_notifications_read, name='mark_notifications_read'),
    
    # ============ LEGACY ============
    path('check-subscription/', check_subscription_view, name='check_subscription'),
]
