"""
Custom Admin Dashboard Views
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.db.models import Count, Q
from django.db.models.functions import TruncDate
from datetime import timedelta
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from functools import wraps
from .models import License, DeviceActivation, UsageLog, SubscriptionPlan, ActivationCode


def staff_required(view_func):
    """Custom decorator để yêu cầu staff login - redirect đến /login/ thay vì admin:login"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login_page')
        if not request.user.is_staff:
            return redirect('login_page')
        return view_func(request, *args, **kwargs)
    return wrapper


def login_page(request):
    """Staff login page"""
    if request.user.is_authenticated and request.user.is_staff:
        return redirect('dashboard')
    
    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            if user.is_staff:
                login(request, user)
                return redirect('dashboard')
            else:
                error = 'Bạn không có quyền truy cập Dashboard.'
        else:
            error = 'Tên đăng nhập hoặc mật khẩu không đúng.'
    
    return render(request, 'admin/login.html', {'error': error})


@staff_required
def dashboard(request):
    """Main dashboard with statistics"""
    today = timezone.now().date()
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    
    # User stats
    total_users = User.objects.count()
    new_users_today = User.objects.filter(date_joined__date=today).count()
    new_users_week = User.objects.filter(date_joined__date__gte=week_ago).count()
    new_users_month = User.objects.filter(date_joined__date__gte=month_ago).count()
    
    # License stats
    total_licenses = License.objects.count()
    active_licenses = License.objects.filter(status='active').count()
    expired_licenses = License.objects.filter(status='expired').count()
    trial_licenses = License.objects.filter(license_type='trial', status='active').count()
    paid_licenses = License.objects.exclude(license_type='trial').filter(status='active').count()
    
    # License types breakdown
    license_types = License.objects.values('license_type').annotate(count=Count('id'))
    
    # Device stats
    total_devices = DeviceActivation.objects.filter(is_active=True).count()
    online_devices = DeviceActivation.objects.filter(
        is_active=True,
        last_heartbeat__gte=timezone.now() - timedelta(minutes=5)
    ).count()
    
    # Recent registrations (last 7 days chart data)
    registration_data = []
    for i in range(7, -1, -1):
        day = today - timedelta(days=i)
        count = User.objects.filter(date_joined__date=day).count()
        registration_data.append({
            'date': day.strftime('%d/%m'),
            'count': count
        })
    
    # Expiring soon (next 7 days)
    expiring_soon = License.objects.filter(
        status='active',
        expire_date__lte=timezone.now() + timedelta(days=7),
        expire_date__gt=timezone.now()
    ).select_related('user').order_by('expire_date')[:10]
    
    # Recent logs
    recent_logs = UsageLog.objects.select_related('license', 'license__user', 'device').order_by('-timestamp')[:20]
    
    # Recent users
    recent_users = User.objects.order_by('-date_joined')[:10]
    
    context = {
        'page': 'dashboard',
        'total_users': total_users,
        'new_users_today': new_users_today,
        'new_users_week': new_users_week,
        'new_users_month': new_users_month,
        'total_licenses': total_licenses,
        'active_licenses': active_licenses,
        'expired_licenses': expired_licenses,
        'trial_licenses': trial_licenses,
        'paid_licenses': paid_licenses,
        'license_types': list(license_types),
        'total_devices': total_devices,
        'online_devices': online_devices,
        'registration_data': registration_data,
        'expiring_soon': expiring_soon,
        'recent_logs': recent_logs,
        'recent_users': recent_users,
    }
    
    return render(request, 'admin/dashboard.html', context)


@staff_required
def users_list(request):
    """Users management page"""
    users = User.objects.select_related('profile').prefetch_related('licenses', 'licenses__activations').order_by('-date_joined')
    
    # Search
    search = request.GET.get('search', '')
    if search:
        users = users.filter(
            Q(username__icontains=search) |
            Q(email__icontains=search) |
            Q(first_name__icontains=search) |
            Q(last_name__icontains=search)
        )
    
    # Filter
    filter_type = request.GET.get('filter', 'all')
    if filter_type == 'active':
        users = users.filter(licenses__status='active').distinct()
    elif filter_type == 'expired':
        users = users.filter(licenses__status='expired').distinct()
    elif filter_type == 'trial':
        users = users.filter(licenses__license_type='trial').distinct()
    elif filter_type == 'paid':
        users = users.exclude(licenses__license_type='trial').filter(licenses__isnull=False).distinct()
    
    context = {
        'page': 'users',
        'users': users[:100],  # Limit to 100
        'search': search,
        'filter_type': filter_type,
        'total_count': users.count(),
    }
    
    return render(request, 'admin/users_list.html', context)


@staff_required  
def licenses_list(request):
    """Licenses management page"""
    licenses = License.objects.select_related('user').prefetch_related('activations').order_by('-created_at')
    
    # Search
    search = request.GET.get('search', '')
    if search:
        licenses = licenses.filter(
            Q(license_key__icontains=search) |
            Q(user__username__icontains=search) |
            Q(user__email__icontains=search)
        )
    
    # Filter by status
    status = request.GET.get('status', 'all')
    if status != 'all':
        licenses = licenses.filter(status=status)
    
    # Filter by type
    license_type = request.GET.get('type', 'all')
    if license_type != 'all':
        licenses = licenses.filter(license_type=license_type)
    
    context = {
        'page': 'licenses',
        'licenses': licenses[:100],
        'search': search,
        'status': status,
        'license_type': license_type,
        'total_count': licenses.count(),
    }
    
    return render(request, 'admin/licenses_list.html', context)


@staff_required
def devices_list(request):
    """Devices management page"""
    devices = DeviceActivation.objects.select_related('license', 'license__user').order_by('-last_seen')
    
    # Search
    search = request.GET.get('search', '')
    if search:
        devices = devices.filter(
            Q(device_name__icontains=search) |
            Q(hardware_id__icontains=search) |
            Q(license__user__username__icontains=search)
        )
    
    # Filter
    filter_type = request.GET.get('filter', 'all')
    if filter_type == 'online':
        devices = devices.filter(last_heartbeat__gte=timezone.now() - timedelta(minutes=5))
    elif filter_type == 'active':
        devices = devices.filter(is_active=True)
    elif filter_type == 'inactive':
        devices = devices.filter(is_active=False)
    
    context = {
        'page': 'devices',
        'devices': devices[:100],
        'search': search,
        'filter_type': filter_type,
        'total_count': devices.count(),
    }
    
    return render(request, 'admin/devices_list.html', context)


@staff_required
def activity_logs(request):
    """Activity logs page"""
    logs = UsageLog.objects.select_related('license', 'license__user', 'device').order_by('-timestamp')
    
    # Filter by event type
    event_type = request.GET.get('event', 'all')
    if event_type != 'all':
        logs = logs.filter(event_type=event_type)
    
    # Date filter
    date_filter = request.GET.get('date', 'all')
    today = timezone.now().date()
    if date_filter == 'today':
        logs = logs.filter(timestamp__date=today)
    elif date_filter == 'week':
        logs = logs.filter(timestamp__date__gte=today - timedelta(days=7))
    elif date_filter == 'month':
        logs = logs.filter(timestamp__date__gte=today - timedelta(days=30))
    
    context = {
        'page': 'logs',
        'logs': logs[:200],
        'event_type': event_type,
        'date_filter': date_filter,
        'total_count': logs.count(),
    }
    
    return render(request, 'admin/activity_logs.html', context)


# ============ ACTION VIEWS ============

@staff_required
def delete_user(request, user_id):
    """Delete a user and redirect back to dashboard"""
    user = get_object_or_404(User, id=user_id)
    
    if request.method == 'POST':
        username = user.username
        # Don't allow deleting superusers
        if user.is_superuser:
            messages.error(request, f'Cannot delete superuser {username}')
        else:
            user.delete()
            messages.success(request, f'User "{username}" has been deleted successfully.')
        return redirect('dashboard_users')
    
    # GET request - show confirmation page
    context = {
        'page': 'users',
        'user_to_delete': user,
        'action': 'delete',
        'action_title': 'Delete User',
        'action_message': f'Are you sure you want to delete user "{user.username}"? This will also delete all associated licenses and devices.',
        'confirm_url': f'/dashboard/users/{user_id}/delete/',
        'cancel_url': '/dashboard/users/',
    }
    return render(request, 'admin/confirm_action.html', context)


@staff_required
def delete_license(request, license_id):
    """Delete a license and redirect back to dashboard"""
    license = get_object_or_404(License, id=license_id)
    
    if request.method == 'POST':
        license_key = license.license_key[:16]
        license.delete()
        messages.success(request, f'License "{license_key}..." has been deleted successfully.')
        return redirect('dashboard_licenses')
    
    # GET request - show confirmation page
    context = {
        'page': 'licenses',
        'license_to_delete': license,
        'action': 'delete',
        'action_title': 'Delete License',
        'action_message': f'Are you sure you want to delete license "{license.license_key[:16]}..."? This will also deactivate all associated devices.',
        'confirm_url': f'/dashboard/licenses/{license_id}/delete/',
        'cancel_url': '/dashboard/licenses/',
    }
    return render(request, 'admin/confirm_action.html', context)


@staff_required
def delete_device(request, device_id):
    """Delete a device and redirect back to dashboard"""
    device = get_object_or_404(DeviceActivation, id=device_id)
    
    if request.method == 'POST':
        device_name = device.device_name or device.hardware_id[:16]
        device.delete()
        messages.success(request, f'Device "{device_name}" has been removed successfully.')
        return redirect('dashboard_devices')
    
    # GET request - show confirmation page
    context = {
        'page': 'devices',
        'device_to_delete': device,
        'action': 'delete',
        'action_title': 'Remove Device',
        'action_message': f'Are you sure you want to remove device "{device.device_name or device.hardware_id[:16]}"?',
        'confirm_url': f'/dashboard/devices/{device_id}/delete/',
        'cancel_url': '/dashboard/devices/',
    }
    return render(request, 'admin/confirm_action.html', context)


@staff_required
def toggle_license_status(request, license_id, action):
    """Activate or suspend a license"""
    license = get_object_or_404(License, id=license_id)
    
    if action == 'activate':
        license.status = 'active'
        license.save()
        messages.success(request, f'License has been activated.')
    elif action == 'suspend':
        license.status = 'suspended'
        license.save()
        messages.success(request, f'License has been suspended.')
    elif action == 'extend':
        # Extend by 30 days
        license.expire_date = license.expire_date + timedelta(days=30)
        license.save()
        messages.success(request, f'License has been extended by 30 days.')
    
    return redirect('dashboard_licenses')


@staff_required
def toggle_device_status(request, device_id, action):
    """Activate or deactivate a device"""
    device = get_object_or_404(DeviceActivation, id=device_id)
    
    if action == 'activate':
        device.is_active = True
        device.save()
        messages.success(request, f'Device has been activated.')
    elif action == 'deactivate':
        device.is_active = False
        device.save()
        messages.success(request, f'Device has been deactivated.')
    
    return redirect('dashboard_devices')


# ============ ADD/EDIT VIEWS ============

@staff_required
def add_user(request):
    """Add a new user"""
    from .models import UserProfile
    
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()
        is_staff = request.POST.get('is_staff') == 'on'
        phone = request.POST.get('phone', '').strip()
        
        # Validate
        if not username:
            messages.error(request, 'Username is required.')
            return redirect('add_user')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, f'Username "{username}" already exists.')
            return redirect('add_user')
        
        if email and User.objects.filter(email=email).exists():
            messages.error(request, f'Email "{email}" already exists.')
            return redirect('add_user')
        
        # Create user
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password or 'changeme123',
            first_name=first_name,
            last_name=last_name,
            is_staff=is_staff
        )
        
        # Update profile phone
        if phone and hasattr(user, 'profile'):
            user.profile.phone = phone
            user.profile.save()
        
        messages.success(request, f'User "{username}" has been created successfully.')
        return redirect('dashboard_users')
    
    context = {
        'page': 'users',
        'action': 'add',
    }
    return render(request, 'admin/user_form.html', context)


@staff_required
def edit_user(request, user_id):
    """Edit an existing user"""
    from .models import UserProfile
    
    user = get_object_or_404(User, id=user_id)
    
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()
        is_staff = request.POST.get('is_staff') == 'on'
        is_active = request.POST.get('is_active') == 'on'
        phone = request.POST.get('phone', '').strip()
        
        # Validate
        if not username:
            messages.error(request, 'Username is required.')
            return redirect('edit_user', user_id=user_id)
        
        if User.objects.filter(username=username).exclude(id=user_id).exists():
            messages.error(request, f'Username "{username}" already exists.')
            return redirect('edit_user', user_id=user_id)
        
        # Update user
        user.username = username
        user.email = email
        user.first_name = first_name
        user.last_name = last_name
        user.is_staff = is_staff
        user.is_active = is_active
        
        if password:
            user.set_password(password)
        
        user.save()
        
        # Update profile phone
        if hasattr(user, 'profile'):
            user.profile.phone = phone
            user.profile.save()
        
        messages.success(request, f'User "{username}" has been updated successfully.')
        return redirect('dashboard_users')
    
    context = {
        'page': 'users',
        'action': 'edit',
        'edit_user': user,
    }
    return render(request, 'admin/user_form.html', context)


@staff_required
def add_license(request):
    """Add a new license - Cấp license cho user"""
    import uuid
    
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        license_type = request.POST.get('license_type', 'trial')
        max_devices = int(request.POST.get('max_devices', 1))
        duration_days = int(request.POST.get('duration_days', 7))
        
        # Validate
        if not user_id:
            messages.error(request, 'Vui lòng chọn người dùng.')
            return redirect('add_license')
        
        user = get_object_or_404(User, id=user_id)
        
        # Kiểm tra user đã có license chưa
        if user.licenses.exists():
            messages.warning(request, f'User "{user.username}" đã có license. Hãy sửa license hiện có thay vì tạo mới.')
            return redirect('edit_license', license_id=user.licenses.first().id)
        
        # Create license
        license = License.objects.create(
            user=user,
            license_type=license_type,
            max_devices=max_devices,
            expire_date=timezone.now() + timedelta(days=duration_days),
            status='active',
            note=f"Cấp bởi Admin vào {timezone.now().strftime('%d/%m/%Y %H:%M')}"
        )
        
        messages.success(request, f'Đã cấp License {duration_days} ngày cho "{user.username}".')
        return redirect('dashboard_users')
    
    # Get all users for dropdown - ưu tiên user chưa có license
    users = User.objects.prefetch_related('licenses').order_by('username')
    
    # Lấy user_id từ query string nếu có
    preselect_user_id = request.GET.get('user_id')
    
    context = {
        'page': 'licenses',
        'action': 'add',
        'all_users': users,
        'preselect_user_id': preselect_user_id,
        'all_users': users,
    }
    return render(request, 'admin/license_form.html', context)


@staff_required
def edit_license(request, license_id):
    """Edit an existing license"""
    license = get_object_or_404(License, id=license_id)
    
    if request.method == 'POST':
        # Kiểm tra nếu là gia hạn nhanh
        extend_quick = request.POST.get('extend_quick')
        if extend_quick:
            days = int(extend_quick)
            if license.expire_date < timezone.now():
                # Nếu đã hết hạn, tính từ hôm nay
                license.expire_date = timezone.now() + timedelta(days=days)
            else:
                license.expire_date = license.expire_date + timedelta(days=days)
            license.status = 'active'  # Kích hoạt lại nếu đang suspended/expired
            # Auto-upgrade Trial to Active nếu > 7 ngày
            if license.license_type == 'trial' and license.days_remaining() > 7:
                license.license_type = 'monthly'
                messages.info(request, f'⚡ Tự động nâng cấp từ Trial lên Monthly (thời hạn > 7 ngày).')
            license.save()
            
            # Notify client app about license change
            from .models import UserChangeNotification
            UserChangeNotification.notify_license_change(license.user, 'license_renewed')
            
            messages.success(request, f'Đã gia hạn thêm {days} ngày.')
            return redirect('edit_license', license_id=license_id)
        
        # Kiểm tra gia hạn tùy chọn
        extend_days = request.POST.get('extend_days')
        if extend_days:
            days = int(extend_days)
            if license.expire_date < timezone.now():
                license.expire_date = timezone.now() + timedelta(days=days)
            else:
                license.expire_date = license.expire_date + timedelta(days=days)
            license.status = 'active'
            # Auto-upgrade Trial to Active nếu > 7 ngày
            if license.license_type == 'trial' and license.days_remaining() > 7:
                license.license_type = 'monthly'
                messages.info(request, f'⚡ Tự động nâng cấp từ Trial lên Monthly (thời hạn > 7 ngày).')
            license.save()
            
            # Notify client app about license change
            from .models import UserChangeNotification
            UserChangeNotification.notify_license_change(license.user, 'license_renewed')
            
            messages.success(request, f'Đã gia hạn thêm {days} ngày.')
            return redirect('edit_license', license_id=license_id)
        
        # Kiểm tra TRỪ thời hạn nhanh
        reduce_quick = request.POST.get('reduce_quick')
        if reduce_quick:
            days = int(reduce_quick)
            license.expire_date = license.expire_date - timedelta(days=days)
            # Nếu hết hạn, đặt status expired
            if license.expire_date <= timezone.now():
                license.status = 'expired'
                messages.warning(request, f'⚠️ License đã hết hạn sau khi trừ {days} ngày.')
            else:
                messages.success(request, f'Đã trừ {days} ngày. Còn lại: {license.days_remaining()} ngày.')
            license.save()
            
            # Notify client app about license change
            from .models import UserChangeNotification
            UserChangeNotification.notify_license_change(license.user, 'license_reduced')
            
            return redirect('edit_license', license_id=license_id)
        
        # Kiểm tra TRỪ thời hạn tùy chọn
        reduce_days = request.POST.get('reduce_days')
        if reduce_days:
            days = int(reduce_days)
            license.expire_date = license.expire_date - timedelta(days=days)
            if license.expire_date <= timezone.now():
                license.status = 'expired'
                messages.warning(request, f'⚠️ License đã hết hạn sau khi trừ {days} ngày.')
            else:
                messages.success(request, f'Đã trừ {days} ngày. Còn lại: {license.days_remaining()} ngày.')
            license.save()
            
            # Notify client app about license change
            from .models import UserChangeNotification
            UserChangeNotification.notify_license_change(license.user, 'license_reduced')
            
            return redirect('edit_license', license_id=license_id)
        
        # Cập nhật thông tin license
        license_type = request.POST.get('license_type', 'trial')
        max_devices = int(request.POST.get('max_devices', 1))
        status = request.POST.get('status', 'active')
        expire_date = request.POST.get('expire_date')
        
        license.license_type = license_type
        license.max_devices = max_devices
        license.status = status
        
        if expire_date:
            from datetime import datetime
            try:
                license.expire_date = timezone.make_aware(datetime.strptime(expire_date, '%Y-%m-%d'))
            except:
                pass
        
        license.save()
        
        # Notify client app about license change
        from .models import UserChangeNotification
        UserChangeNotification.notify_license_change(license.user, 'license_updated')
        
        messages.success(request, f'Đã cập nhật License thành công.')
        return redirect('dashboard_licenses')
    
    context = {
        'page': 'licenses',
        'action': 'edit',
        'edit_license': license,
    }
    return render(request, 'admin/license_form.html', context)


# ============ ACTIVATION CODES ============

@staff_required
def activation_codes_list(request):
    """Danh sách mã kích hoạt"""
    codes = ActivationCode.objects.all().order_by('-created_at')
    
    context = {
        'page': 'codes',
        'codes': codes,
    }
    return render(request, 'admin/activation_codes.html', context)


@staff_required
def generate_activation_codes(request):
    """Tạo mã kích hoạt mới"""
    if request.method == 'POST':
        count = int(request.POST.get('count', 1))
        trial_days = int(request.POST.get('trial_days', 7))
        max_uses = int(request.POST.get('max_uses', 1))
        note = request.POST.get('note', '')
        
        created_codes = []
        for _ in range(min(count, 100)):  # Tối đa 100 mã
            code = ActivationCode.objects.create(
                trial_days=trial_days,
                max_uses=max_uses,
                note=note
            )
            created_codes.append(code.code)
        
        messages.success(request, f'Đã tạo {len(created_codes)} mã kích hoạt: {", ".join(created_codes)}')
        return redirect('activation_codes')
    
    return redirect('activation_codes')


@staff_required
def delete_activation_code(request, code_id):
    """Xóa mã kích hoạt"""
    code = get_object_or_404(ActivationCode, id=code_id)
    code.delete()
    messages.success(request, f'Đã xóa mã kích hoạt.')
    return redirect('activation_codes')


def logout_view(request):
    """Logout and redirect to login page"""
    from django.contrib.auth import logout
    logout(request)
    return redirect('login_page')
