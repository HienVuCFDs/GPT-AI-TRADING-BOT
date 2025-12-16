"""
Bank Payment Views - PayOS Integration
Thanh toán qua ngân hàng với PayOS QR Code
"""
import json
import time
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response

from ...models import License, Payment, PricingPlan
from ...payos_service import payos_service
from ..utils import extend_license, get_realtime_usd_rate


@api_view(['GET'])
@permission_classes([AllowAny])
def get_pricing_plans(request):
    """Lấy danh sách gói giá"""
    plans = PricingPlan.objects.filter(is_active=True).order_by('duration_months')
    data = []
    for plan in plans:
        data.append({
            'id': plan.id,
            'name': plan.name,
            'duration_months': plan.duration_months,
            'price_usd': float(plan.price_usd),
            'price_vnd': plan.price_vnd,
            'description': plan.description,
            'features': plan.features,
        })
    return Response({'success': True, 'plans': data})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_payment(request):
    """Tạo thanh toán mới qua PayOS với tỷ giá USD realtime"""
    try:
        data = request.data
        plan_id = data.get('plan_id')
        
        if not plan_id:
            return Response({'success': False, 'error': 'Missing plan_id'}, status=400)
        
        # Lấy pricing plan
        try:
            plan = PricingPlan.objects.get(id=plan_id, is_active=True)
        except PricingPlan.DoesNotExist:
            return Response({'success': False, 'error': 'Invalid plan'}, status=400)
        
        # Lấy license của user
        try:
            license_obj = License.objects.get(user=request.user)
        except License.DoesNotExist:
            return Response({'success': False, 'error': 'License not found'}, status=400)
        
        # Lấy tỷ giá USD/VND realtime và tính VND
        usd_rate = get_realtime_usd_rate()
        amount_vnd = int(float(plan.price_usd) * usd_rate)
        # Làm tròn đến nghìn
        amount_vnd = round(amount_vnd / 1000) * 1000
        
        print(f"[Payment] USD: ${plan.price_usd} x Rate: {usd_rate:,.0f} = {amount_vnd:,} VND")
        
        # Tạo order code (unique)
        order_code = int(time.time() * 1000) % 9007199254740991  # JavaScript MAX_SAFE_INTEGER
        
        # Tạo payment record
        payment = Payment.objects.create(
            license=license_obj,
            pricing_plan=plan,
            amount_vnd=amount_vnd,
            amount_usd=plan.price_usd,
            order_code=str(order_code),
            status='pending',
            payment_method='bank',
        )
        
        # Mô tả ngắn (max 25 ký tự)
        description = f"License {plan.duration_months}mo"
        
        # Gọi PayOS tạo payment
        result = payos_service.create_payment_link(
            order_code=order_code,
            amount=amount_vnd,
            description=description,
            buyer_name=request.user.username,
            buyer_email=request.user.email if hasattr(request.user, 'email') else "",
        )
        
        if result.get('success'):
            # Lưu payment link info
            payment.payos_payment_link_id = result.get('paymentLinkId', '')
            payment.save()
            
            return Response({
                'success': True,
                'payment_id': payment.id,
                'order_code': order_code,
                'checkout_url': result.get('checkoutUrl'),
                'qr_code': result.get('qrCode'),
                'amount_vnd': amount_vnd,
                'amount_usd': float(plan.price_usd),
                'usd_rate': round(usd_rate, 0),
                'plan_name': plan.name,
            })
        else:
            payment.status = 'failed'
            payment.note = result.get('error', 'Unknown error')
            payment.save()
            return Response({
                'success': False,
                'error': result.get('error', 'Failed to create payment')
            }, status=400)
            
    except Exception as e:
        print(f"[Payment] Error: {e}")
        return Response({'success': False, 'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def payos_webhook(request):
    """Webhook nhận thông báo thanh toán từ PayOS"""
    try:
        body = json.loads(request.body)
        print(f"[PayOS Webhook] Received: {body}")
        
        # Verify signature
        if not payos_service.verify_webhook_data(body):
            print("[PayOS Webhook] Invalid signature")
            return JsonResponse({'success': False, 'error': 'Invalid signature'}, status=400)
        
        data = body.get('data', {})
        order_code = str(data.get('orderCode'))
        code = data.get('code')  # "00" = success
        desc = data.get('desc')
        
        print(f"[PayOS Webhook] Order: {order_code}, Code: {code}, Desc: {desc}")
        
        # Tìm payment
        try:
            payment = Payment.objects.get(order_code=order_code)
        except Payment.DoesNotExist:
            print(f"[PayOS Webhook] Payment not found: {order_code}")
            return JsonResponse({'success': False, 'error': 'Payment not found'}, status=404)
        
        # Kiểm tra nếu đã xử lý
        if payment.status == 'completed':
            print(f"[PayOS Webhook] Payment already completed: {order_code}")
            return JsonResponse({'success': True, 'message': 'Already processed'})
        
        # Cập nhật trạng thái
        if code == "00":
            payment.status = 'completed'
            payment.paid_at = timezone.now()
            payment.transaction_id = data.get('reference', '')
            payment.save()
            
            # Extend license
            extend_license(payment)
            
            print(f"[PayOS Webhook] Payment completed: {order_code}")
            return JsonResponse({'success': True, 'message': 'Payment processed'})
        else:
            payment.status = 'failed'
            payment.note = desc
            payment.save()
            print(f"[PayOS Webhook] Payment failed: {order_code} - {desc}")
            return JsonResponse({'success': True, 'message': 'Payment status updated'})
            
    except Exception as e:
        print(f"[PayOS Webhook] Error: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def check_payment_status(request, order_code):
    """Kiểm tra trạng thái thanh toán"""
    try:
        payment = Payment.objects.get(order_code=str(order_code), license__user=request.user)
        
        # Nếu pending, check với PayOS
        if payment.status == 'pending':
            result = payos_service.get_payment_info(int(order_code))
            if result.get('success'):
                data = result.get('data', {})
                status = data.get('status')
                
                if status == 'PAID':
                    payment.status = 'completed'
                    payment.paid_at = timezone.now()
                    payment.save()
                    extend_license(payment)
                elif status == 'CANCELLED':
                    payment.status = 'cancelled'
                    payment.save()
        
        return Response({
            'success': True,
            'status': payment.status,
            'order_code': payment.order_code,
            'amount_vnd': payment.amount_vnd,
        })
        
    except Payment.DoesNotExist:
        return Response({'success': False, 'error': 'Payment not found'}, status=404)
    except Exception as e:
        return Response({'success': False, 'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def payment_history(request):
    """Lịch sử thanh toán"""
    payments = Payment.objects.filter(license__user=request.user).order_by('-created_at')
    data = []
    for p in payments:
        data.append({
            'id': p.id,
            'order_code': p.order_code,
            'plan_name': p.pricing_plan.name if p.pricing_plan else 'N/A',
            'amount_vnd': p.amount_vnd,
            'amount_usd': float(p.amount_usd) if p.amount_usd else 0,
            'status': p.status,
            'payment_method': p.payment_method,
            'created_at': p.created_at.isoformat(),
            'paid_at': p.paid_at.isoformat() if p.paid_at else None,
        })
    return Response({'success': True, 'payments': data})


@api_view(['GET'])
@permission_classes([AllowAny])
def get_usd_rate(request):
    """Lấy tỷ giá USD/VND realtime từ exchangerate-api.com"""
    import requests as req
    
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        
        response = req.get(url, timeout=10)
        data = response.json()
        
        if data.get('rates') and data['rates'].get('VND'):
            usd_vnd_rate = data['rates']['VND']
            
            return Response({
                'success': True,
                'usd_vnd_rate': round(usd_vnd_rate, 0),
                'source': 'ExchangeRate-API',
                'updated_at': timezone.now().isoformat()
            })
        else:
            return Response({
                'success': True,
                'usd_vnd_rate': 25500,
                'source': 'Fallback',
                'updated_at': timezone.now().isoformat()
            })
            
    except Exception as e:
        print(f"[USD Rate] Error: {e}")
        return Response({
            'success': True,
            'usd_vnd_rate': 25500,
            'source': 'Fallback',
            'error': str(e)
        })
