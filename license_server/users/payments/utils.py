"""
Payment Utilities - Các hàm dùng chung cho payment
"""
from datetime import timedelta
from django.utils import timezone


def get_realtime_usd_rate():
    """Lấy tỷ giá USD/VND realtime"""
    import requests as req
    try:
        # Lấy từ exchangerate-api (free)
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        response = req.get(url, timeout=10)
        data = response.json()
        
        if data.get('rates', {}).get('VND'):
            rate = data['rates']['VND']
            print(f"[USD Rate] From exchangerate-api: {rate:,.0f} VND")
            return rate
        
        # Fallback rate
        return 25500
    except Exception as e:
        print(f"[USD Rate] Error: {e}")
        return 25500  # Fallback rate


def extend_license(payment):
    """Gia hạn license sau khi thanh toán thành công (Bank hoặc Crypto)"""
    try:
        license_obj = payment.license
        plan = payment.pricing_plan
        
        if not plan:
            print(f"[ExtendLicense] No plan for payment {payment.id}")
            return False
        
        months = plan.duration_months
        
        # Tính ngày mới
        now = timezone.now()
        if license_obj.expire_date and license_obj.expire_date > now:
            # Còn hạn -> cộng thêm
            new_expiry = license_obj.expire_date + timedelta(days=months * 30)
        else:
            # Hết hạn -> tính từ bây giờ
            new_expiry = now + timedelta(days=months * 30)
        
        # Map duration to license type
        type_map = {
            1: 'monthly',
            3: 'quarterly', 
            6: 'semi_annual',
            12: 'yearly',
        }
        new_type = type_map.get(months, 'monthly')
        
        # Update license
        license_obj.license_type = new_type
        license_obj.expire_date = new_expiry
        license_obj.status = 'active'
        license_obj.save()
        
        print(f"[ExtendLicense] License {license_obj.id} extended to {new_expiry} ({new_type})")
        return True
        
    except Exception as e:
        print(f"[ExtendLicense] Error: {e}")
        import traceback
        traceback.print_exc()
        return False
