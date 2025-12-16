"""
Crypto Payment Views - NOWPayments Integration
Tự động xác nhận thanh toán crypto qua webhook
"""
import hashlib
import hmac
import json
import requests
from datetime import timedelta
from django.conf import settings
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response

from ...models import Payment, License, PricingPlan
from ..utils import extend_license


# ============ NOWPAYMENTS API ============
NOWPAYMENTS_API_URL = "https://api.nowpayments.io/v1"


def get_nowpayments_headers():
    """Get headers for NOWPayments API"""
    return {
        'x-api-key': settings.NOWPAYMENTS_API_KEY,
        'Content-Type': 'application/json'
    }


@api_view(['GET'])
@permission_classes([AllowAny])
def get_crypto_currencies(request):
    """Lấy danh sách currencies được hỗ trợ"""
    try:
        response = requests.get(
            f"{NOWPAYMENTS_API_URL}/currencies",
            headers=get_nowpayments_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            # Filter phổ biến
            popular = ['usdt', 'usdttrc20', 'btc', 'eth', 'ltc', 'trx', 'bnb', 'sol', 'doge']
            currencies = [c for c in data.get('currencies', []) if c.lower() in popular]
            
            return Response({
                'success': True,
                'currencies': currencies
            })
        else:
            return Response({
                'success': False,
                'error': 'Failed to fetch currencies'
            })
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        })


@api_view(['GET'])
@permission_classes([AllowAny])
def get_min_amount(request):
    """Lấy số tiền tối thiểu cho từng loại crypto"""
    currency = request.GET.get('currency', 'usdttrc20')
    
    try:
        response = requests.get(
            f"{NOWPAYMENTS_API_URL}/min-amount",
            params={'currency_from': 'usd', 'currency_to': currency},
            headers=get_nowpayments_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return Response({
                'success': True,
                'min_amount': data.get('min_amount', 0),
                'currency': currency
            })
        else:
            return Response({
                'success': False,
                'error': 'Failed to fetch min amount'
            })
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_crypto_payment(request):
    """
    Tạo payment crypto qua NOWPayments
    Request body: {
        "plan_id": 1,
        "currency": "usdttrc20"  # optional, default USDT TRC20
    }
    """
    try:
        user = request.user
        plan_id = request.data.get('plan_id', 1)
        pay_currency = request.data.get('currency', 'usdttrc20')  # Default USDT TRC20
        
        # Get pricing plan
        try:
            plan = PricingPlan.objects.get(id=plan_id, is_active=True)
        except PricingPlan.DoesNotExist:
            return Response({'success': False, 'error': 'Invalid plan'}, status=400)
        
        # Get user's license
        try:
            license_obj = License.objects.get(user=user)
        except License.DoesNotExist:
            return Response({'success': False, 'error': 'License not found'}, status=400)
        
        # Price in USD
        amount_usd = float(plan.price_usd)
        
        # Create unique order ID
        order_id = f"CRYPTO_{user.id}_{int(timezone.now().timestamp())}"
        
        # Create NOWPayments invoice
        payload = {
            "price_amount": amount_usd,
            "price_currency": "usd",
            "pay_currency": pay_currency,
            "ipn_callback_url": settings.NOWPAYMENTS_WEBHOOK_URL,
            "order_id": order_id,
            "order_description": f"License {plan.name} - {plan.duration_months} months"
        }
        
        print(f"[NOWPayments] Creating payment: {payload}")
        
        response = requests.post(
            f"{NOWPAYMENTS_API_URL}/payment",
            json=payload,
            headers=get_nowpayments_headers(),
            timeout=30
        )
        
        print(f"[NOWPayments] Response: {response.status_code} - {response.text[:500]}")
        
        if response.status_code in [200, 201]:
            data = response.json()
            
            # Build note with crypto metadata
            crypto_note = json.dumps({
                'nowpayments_id': data.get('payment_id'),
                'pay_currency': pay_currency,
                'pay_amount': data.get('pay_amount'),
                'pay_address': data.get('pay_address'),
                'network': data.get('network'),
            })
            
            # Save to database using existing Payment model
            payment = Payment.objects.create(
                license=license_obj,
                pricing_plan=plan,
                amount_usd=amount_usd,
                amount_vnd=0,  # Crypto payment - no VND
                order_code=order_id,
                payment_method='crypto',
                status='pending',
                transaction_id=str(data.get('payment_id', '')),
                note=crypto_note,  # Store crypto metadata in note field
            )
            
            return Response({
                'success': True,
                'order_id': order_id,
                'payment_id': data.get('payment_id'),
                'pay_address': data.get('pay_address'),
                'pay_amount': data.get('pay_amount'),
                'pay_currency': pay_currency.upper(),
                'network': data.get('network', 'TRC20' if 'trc20' in pay_currency else ''),
                'amount_usd': amount_usd,
                'status': 'waiting',
                'expiration_estimate_date': data.get('expiration_estimate_date'),
                'qr_code': f"{data.get('pay_address')}"
            })
        else:
            error_msg = 'Unknown error'
            try:
                error_msg = response.json().get('message', 'Unknown error')
            except:
                error_msg = response.text[:200]
            return Response({
                'success': False,
                'error': error_msg
            }, status=400)
            
    except Exception as e:
        print(f"[NOWPayments] Error: {e}")
        import traceback
        traceback.print_exc()
        return Response({
            'success': False,
            'error': str(e)
        }, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def check_crypto_payment_status(request, order_id):
    """Kiểm tra trạng thái payment crypto"""
    try:
        payment = Payment.objects.filter(
            order_code=order_id,
            license__user=request.user,
            payment_method='crypto'
        ).first()
        
        if not payment:
            return Response({'success': False, 'error': 'Payment not found'}, status=404)
        
        # Get payment ID from note (stored as JSON)
        nowpayments_id = None
        try:
            if payment.note:
                note_data = json.loads(payment.note)
                nowpayments_id = note_data.get('nowpayments_id')
        except:
            nowpayments_id = payment.transaction_id
        
        if nowpayments_id:
            # Check status from NOWPayments
            response = requests.get(
                f"{NOWPAYMENTS_API_URL}/payment/{nowpayments_id}",
                headers=get_nowpayments_headers(),
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                np_status = data.get('payment_status', 'waiting')
                
                # Map NOWPayments status to our status
                status_map = {
                    'waiting': 'pending',
                    'confirming': 'pending',
                    'confirmed': 'pending',
                    'sending': 'pending',
                    'partially_paid': 'pending',
                    'finished': 'completed',
                    'failed': 'failed',
                    'refunded': 'cancelled',
                    'expired': 'cancelled'
                }
                
                new_status = status_map.get(np_status, 'pending')
                
                # Update local status if changed
                if payment.status != new_status:
                    payment.status = new_status
                    if new_status == 'completed':
                        payment.paid_at = timezone.now()
                        # Extend license
                        extend_license(payment)
                    payment.save()
                
                return Response({
                    'success': True,
                    'order_id': order_id,
                    'status': new_status,
                    'nowpayments_status': np_status,
                    'pay_address': data.get('pay_address'),
                    'pay_amount': data.get('pay_amount'),
                    'actually_paid': data.get('actually_paid', 0),
                    'pay_currency': data.get('pay_currency', '').upper()
                })
        
        return Response({
            'success': True,
            'order_id': order_id,
            'status': payment.status
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=500)


@api_view(['POST'])
@permission_classes([AllowAny])
def nowpayments_webhook(request):
    """
    Webhook từ NOWPayments khi có payment update
    NOWPayments sẽ gửi IPN (Instant Payment Notification)
    """
    try:
        # Verify IPN signature
        ipn_secret = settings.NOWPAYMENTS_IPN_SECRET
        
        if ipn_secret and ipn_secret != 'YOUR_IPN_SECRET_HERE':
            # Get signature from header
            received_signature = request.headers.get('x-nowpayments-sig', '')
            
            # Calculate expected signature
            sorted_data = json.dumps(request.data, sort_keys=True, separators=(',', ':'))
            expected_signature = hmac.new(
                ipn_secret.encode('utf-8'),
                sorted_data.encode('utf-8'),
                hashlib.sha512
            ).hexdigest()
            
            if received_signature.lower() != expected_signature.lower():
                print(f"[NOWPayments Webhook] Invalid signature!")
                return Response({'status': 'invalid signature'}, status=400)
        
        data = request.data
        print(f"[NOWPayments Webhook] Received: {data}")
        
        order_id = data.get('order_id')
        payment_status = data.get('payment_status')
        payment_id = data.get('payment_id')
        
        if not order_id:
            return Response({'status': 'missing order_id'}, status=400)
        
        # Find payment
        payment = Payment.objects.filter(order_code=order_id).first()
        
        if not payment:
            print(f"[NOWPayments Webhook] Payment not found: {order_id}")
            return Response({'status': 'payment not found'}, status=404)
        
        # Update status
        old_status = payment.status
        
        if payment_status == 'finished':
            payment.status = 'completed'
            payment.paid_at = timezone.now()
            
            # Update note with payment info
            try:
                note_data = json.loads(payment.note) if payment.note else {}
                note_data['actually_paid'] = data.get('actually_paid')
                note_data['outcome_amount'] = data.get('outcome_amount')
                payment.note = json.dumps(note_data)
            except:
                pass
            
            payment.save()
            
            # Extend license
            if old_status != 'completed':
                extend_license(payment)
                print(f"[NOWPayments Webhook] License extended for payment {order_id}")
                
        elif payment_status in ['failed', 'refunded', 'expired']:
            payment.status = 'cancelled' if payment_status in ['refunded', 'expired'] else 'failed'
            payment.save()
            
        elif payment_status in ['waiting', 'confirming', 'confirmed', 'sending', 'partially_paid']:
            # Still pending - update note
            try:
                note_data = json.loads(payment.note) if payment.note else {}
                note_data['last_status'] = payment_status
                payment.note = json.dumps(note_data)
                payment.save()
            except:
                pass
        
        return Response({'status': 'ok'})
        
    except Exception as e:
        print(f"[NOWPayments Webhook] Error: {e}")
        import traceback
        traceback.print_exc()
        return Response({'status': 'error', 'message': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([AllowAny])
def get_crypto_estimate(request):
    """Lấy estimate số lượng crypto cần trả cho 1 số USD"""
    amount_usd = request.GET.get('amount', 199)
    currency = request.GET.get('currency', 'usdttrc20')
    
    try:
        response = requests.get(
            f"{NOWPAYMENTS_API_URL}/estimate",
            params={
                'amount': amount_usd,
                'currency_from': 'usd',
                'currency_to': currency
            },
            headers=get_nowpayments_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return Response({
                'success': True,
                'amount_usd': float(amount_usd),
                'estimated_amount': data.get('estimated_amount'),
                'currency': currency.upper()
            })
        else:
            return Response({
                'success': False,
                'error': 'Failed to get estimate'
            })
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        })
