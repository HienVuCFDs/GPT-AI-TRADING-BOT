"""
PayOS Integration Service
Tích hợp thanh toán tự động qua PayOS
"""
import hashlib
import hmac
import json
import requests
from django.conf import settings
from django.utils import timezone
from datetime import timedelta


class PayOSService:
    """Service xử lý thanh toán PayOS"""
    
    BASE_URL = "https://api-merchant.payos.vn"
    
    def __init__(self):
        self.client_id = getattr(settings, 'PAYOS_CLIENT_ID', '')
        self.api_key = getattr(settings, 'PAYOS_API_KEY', '')
        self.checksum_key = getattr(settings, 'PAYOS_CHECKSUM_KEY', '')
    
    def _create_signature(self, data: dict) -> str:
        """Tạo chữ ký cho request"""
        # Sắp xếp theo key alphabet
        sorted_data = sorted(data.items())
        # Tạo chuỗi: key1=value1&key2=value2
        data_str = "&".join([f"{k}={v}" for k, v in sorted_data if v is not None and v != ""])
        # HMAC SHA256
        signature = hmac.new(
            self.checksum_key.encode('utf-8'),
            data_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _get_headers(self):
        """Headers cho API request"""
        return {
            "Content-Type": "application/json",
            "x-client-id": self.client_id,
            "x-api-key": self.api_key,
        }
    
    def create_payment_link(self, order_code: int, amount: int, description: str, 
                           buyer_name: str = "", buyer_email: str = "", 
                           buyer_phone: str = "", items: list = None) -> dict:
        """
        Tạo link thanh toán PayOS
        
        Args:
            order_code: Mã đơn hàng (số nguyên duy nhất)
            amount: Số tiền VND
            description: Mô tả đơn hàng (max 25 ký tự)
        
        Returns:
            dict: {checkoutUrl, qrCode, ...}
        """
        cancel_url = getattr(settings, 'PAYOS_CANCEL_URL', 'http://localhost:8000/payment/cancel/')
        return_url = getattr(settings, 'PAYOS_RETURN_URL', 'http://localhost:8000/payment/success/')
        
        # Chuẩn bị data
        data = {
            "orderCode": order_code,
            "amount": amount,
            "description": description[:25],  # Max 25 chars
            "cancelUrl": cancel_url,
            "returnUrl": return_url,
        }
        
        if buyer_name:
            data["buyerName"] = buyer_name
        if buyer_email:
            data["buyerEmail"] = buyer_email
        if buyer_phone:
            data["buyerPhone"] = buyer_phone
        if items:
            data["items"] = items
        
        # Tạo signature từ các trường bắt buộc
        signature_data = {
            "amount": amount,
            "cancelUrl": cancel_url,
            "description": description[:25],
            "orderCode": order_code,
            "returnUrl": return_url,
        }
        data["signature"] = self._create_signature(signature_data)
        
        # Gọi API
        try:
            response = requests.post(
                f"{self.BASE_URL}/v2/payment-requests",
                headers=self._get_headers(),
                json=data,
                timeout=30
            )
            result = response.json()
            
            print(f"[PayOS] Create payment response: {result}")
            
            if result.get("code") == "00":
                return {
                    "success": True,
                    "data": result.get("data", {}),
                    "checkoutUrl": result.get("data", {}).get("checkoutUrl"),
                    "qrCode": result.get("data", {}).get("qrCode"),
                    "paymentLinkId": result.get("data", {}).get("paymentLinkId"),
                }
            else:
                return {
                    "success": False,
                    "error": result.get("desc", "Unknown error"),
                    "code": result.get("code"),
                }
        except Exception as e:
            print(f"[PayOS] Error: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def get_payment_info(self, order_code: int) -> dict:
        """Lấy thông tin thanh toán theo order_code"""
        try:
            response = requests.get(
                f"{self.BASE_URL}/v2/payment-requests/{order_code}",
                headers=self._get_headers(),
                timeout=30
            )
            result = response.json()
            
            if result.get("code") == "00":
                return {
                    "success": True,
                    "data": result.get("data", {}),
                }
            else:
                return {
                    "success": False,
                    "error": result.get("desc", "Unknown error"),
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    def verify_webhook_data(self, webhook_body: dict) -> bool:
        """Xác thực dữ liệu webhook"""
        data = webhook_body.get("data", {})
        signature = webhook_body.get("signature", "")
        
        if not data or not signature:
            return False
        
        # Tạo signature từ data
        signature_data = {
            "orderCode": data.get("orderCode"),
            "amount": data.get("amount"),
            "description": data.get("description"),
            "accountNumber": data.get("accountNumber"),
            "reference": data.get("reference"),
            "transactionDateTime": data.get("transactionDateTime"),
            "currency": data.get("currency"),
            "paymentLinkId": data.get("paymentLinkId"),
            "code": data.get("code"),
            "desc": data.get("desc"),
            "counterAccountBankId": data.get("counterAccountBankId"),
            "counterAccountBankName": data.get("counterAccountBankName"),
            "counterAccountName": data.get("counterAccountName"),
            "counterAccountNumber": data.get("counterAccountNumber"),
            "virtualAccountName": data.get("virtualAccountName"),
            "virtualAccountNumber": data.get("virtualAccountNumber"),
        }
        
        # Loại bỏ None values
        signature_data = {k: v for k, v in signature_data.items() if v is not None}
        
        calculated_signature = self._create_signature(signature_data)
        return hmac.compare_digest(calculated_signature, signature)


# Singleton instance
payos_service = PayOSService()

# 🔧 FIX: Auto-update license after successful payment
def handle_payment_success(payment_obj):
    """
    Xử lý sau khi thanh toán thành công:
    1. Cập nhật hạn sử dụng license
    2. Chuyển status từ Trial/Expired -> Active
    3. Gửi email thông báo cho user
    4. Tạo notification change
    """
    try:
        license_obj = payment_obj.license
        pricing_plan = payment_obj.pricing_plan
        
        if not license_obj or not pricing_plan:
            print(f"❌ Payment {payment_obj.order_code}: Missing license or pricing plan")
            return False
        
        # Tính ngày hết hạn mới
        if pricing_plan.duration_months == 0:
            # Lifetime
            new_expire_date = timezone.now() + timedelta(days=36500)  # 100 năm
        else:
            # Tính theo tháng
            new_expire_date = timezone.now() + timedelta(days=pricing_plan.duration_months * 30)
        
        # Cập nhật license
        license_obj.expire_date = new_expire_date
        license_obj.license_type = {
            1: 'monthly',
            3: 'quarterly', 
            12: 'yearly'
        }.get(pricing_plan.duration_months, 'monthly')
        
        # Đảm bảo status là ACTIVE nếu còn hạn
        license_obj.status = 'active'
        
        # 🔧 FIX: Gọi save() để trigger update_status_from_expiry_date()
        license_obj.save()
        
        print(f"✅ License {license_obj.license_key} updated:")
        print(f"   - Type: {license_obj.license_type}")
        print(f"   - Expire: {new_expire_date}")
        print(f"   - Status: {license_obj.status}")
        
        # 🎉 GỬI EMAIL THÔNG BÁO THANH TOÁN THÀNH CÔNG
        payment_obj.send_success_notification()
        
        # 📢 TẠO NOTIFICATION CHANGE
        from .models import UserChangeNotification
        UserChangeNotification.notify_license_change(
            user=license_obj.user,
            change_type='license_renewed',
            extra_data={
                'order_code': payment_obj.order_code,
                'pricing_plan': pricing_plan.name,
                'amount': f"{payment_obj.amount_vnd:,} ₫ / ${payment_obj.amount_usd}"
            }
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling payment success: {e}")
        import traceback
        traceback.print_exc()
        return False