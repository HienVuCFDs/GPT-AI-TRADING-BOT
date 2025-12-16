from django.urls import path, include
from users.dashboard_views import (
    dashboard, users_list, licenses_list, devices_list, activity_logs,
    delete_user, delete_license, delete_device,
    toggle_license_status, toggle_device_status,
    add_user, edit_user, add_license, edit_license, logout_view, login_page,
    activation_codes_list, generate_activation_codes, delete_activation_code
)

urlpatterns = [
    # Root redirect
    path('', dashboard, name='home'),
    
    # Login Page
    path('login/', login_page, name='login_page'),
    
    # Custom Dashboard
    path('dashboard/', dashboard, name='dashboard'),
    path('dashboard/users/', users_list, name='dashboard_users'),
    path('dashboard/licenses/', licenses_list, name='dashboard_licenses'),
    path('dashboard/devices/', devices_list, name='dashboard_devices'),
    path('dashboard/activity/', activity_logs, name='dashboard_activity'),
    
    # Dashboard Actions - Users
    path('dashboard/users/add/', add_user, name='add_user'),
    path('dashboard/users/<int:user_id>/edit/', edit_user, name='edit_user'),
    path('dashboard/users/<int:user_id>/delete/', delete_user, name='delete_user'),
    
    # Dashboard Actions - Licenses
    path('dashboard/licenses/add/', add_license, name='add_license'),
    path('dashboard/licenses/<uuid:license_id>/edit/', edit_license, name='edit_license'),
    path('dashboard/licenses/<uuid:license_id>/delete/', delete_license, name='delete_license'),
    path('dashboard/licenses/<uuid:license_id>/<str:action>/', toggle_license_status, name='license_action'),
    
    # Dashboard Actions - Devices
    path('dashboard/devices/<uuid:device_id>/delete/', delete_device, name='delete_device'),
    path('dashboard/devices/<uuid:device_id>/<str:action>/', toggle_device_status, name='device_action'),
    
    # Dashboard Actions - Activation Codes
    path('dashboard/codes/', activation_codes_list, name='activation_codes'),
    path('dashboard/codes/generate/', generate_activation_codes, name='generate_codes'),
    path('dashboard/codes/<int:code_id>/delete/', delete_activation_code, name='delete_code'),
    
    # Auth
    path('logout/', logout_view, name='logout_view'),
    
    # API
    path('api/', include('users.urls')),
]

