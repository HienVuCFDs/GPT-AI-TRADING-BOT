"""
License Server Settings - Django Configuration
"""
from pathlib import Path
from datetime import timedelta
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'dev-secret-key-change-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DJANGO_DEBUG', 'True').lower() == 'true'

ALLOWED_HOSTS = ['*']  # Configure properly in production

# Application definition
INSTALLED_APPS = [
    # 'django.contrib.admin',  # Removed - using custom dashboard
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third party
    'rest_framework',
    'rest_framework_simplejwt',
    'rest_framework_simplejwt.token_blacklist',  # For logout/blacklist
    'corsheaders',  # CORS support
    
    # Local apps
    'users',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # CORS - must be first
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'license_server.urls'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'templates'],
    'APP_DIRS': True,
    'OPTIONS': {
        'context_processors': [
            'django.template.context_processors.debug',
            'django.template.context_processors.request',
            'django.contrib.auth.context_processors.auth',
            'django.contrib.messages.context_processors.messages',
        ],
    },
}]

WSGI_APPLICATION = 'license_server.wsgi.application'

# Database
# For production, use PostgreSQL
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 6}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'vi'  # Vietnamese as default
TIME_ZONE = 'Asia/Ho_Chi_Minh'
USE_I18N = True
USE_L10N = True
USE_TZ = True

# Available languages
from django.utils.translation import gettext_lazy as _
LANGUAGES = [
    ('vi', _('Tiếng Việt')),
    ('en', _('English')),
]

LOCALE_PATHS = [
    BASE_DIR / 'locale',
]

# Static files
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ============ REST FRAMEWORK ============
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour',
        'user': '1000/hour',
    },
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
    ],
    'EXCEPTION_HANDLER': 'rest_framework.views.exception_handler',
}

# ============ JWT SETTINGS ============
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(hours=24),  # Changed from 1 hour to 24 hours
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'UPDATE_LAST_LOGIN': True,
    
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    
    'AUTH_HEADER_TYPES': ('Bearer',),
    'AUTH_HEADER_NAME': 'HTTP_AUTHORIZATION',
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    
    'TOKEN_TYPE_CLAIM': 'token_type',
    'JTI_CLAIM': 'jti',
}

# ============ CORS SETTINGS ============
CORS_ALLOW_ALL_ORIGINS = DEBUG  # Only in development
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

CORS_ALLOW_CREDENTIALS = True

CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

# ============ LOGGING ============
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs' / 'license_server.log',
        },
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'users': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# Create logs directory
(BASE_DIR / 'logs').mkdir(exist_ok=True)

# ============ LOGIN SETTINGS ============
LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/dashboard/'

# ============ EMAIL CONFIGURATION ============
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'vuhien2444cfds@gmail.com'
EMAIL_HOST_PASSWORD = 'ybnrempkbaxevzji'
DEFAULT_FROM_EMAIL = 'ChatGPT AI BOT <vuhien2444cfds@gmail.com>'

# Domain cho link kích hoạt
SITE_DOMAIN = os.environ.get('SITE_DOMAIN', 'http://127.0.0.1:8000')

# ============ PAYOS CONFIGURATION ============
PAYOS_CLIENT_ID = 'e2408656-4e1d-4aaf-b10f-3214dbcd5482'
PAYOS_API_KEY = '95f5ec70-37e4-44ed-86a4-7a0765b84275'
PAYOS_CHECKSUM_KEY = '9f1fc76d8a2b945211e3f5defb83162b53b30c567e3477c49be21401398a96bf'
PAYOS_RETURN_URL = SITE_DOMAIN + '/payment/success/'
PAYOS_CANCEL_URL = SITE_DOMAIN + '/payment/cancel/'

# ============ NOWPAYMENTS CONFIGURATION ============
# Dashboard: https://account.nowpayments.io
NOWPAYMENTS_API_KEY = '4KK4Y7G-9AN4RR6-JFTQZVC-5VFVSV2'
NOWPAYMENTS_PUBLIC_KEY = '8a778521-a40b-4fab-9eac-16d75db5e90e'
NOWPAYMENTS_IPN_SECRET = 'YcFWwk2NYtZrx0vbOuZ0DLXFL1+myNjS'
NOWPAYMENTS_WEBHOOK_URL = SITE_DOMAIN + '/api/webhook/nowpayments/'
