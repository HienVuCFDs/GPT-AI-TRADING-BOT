"""
Payment Views - Xử lý thanh toán PayOS
"""
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from django.db import transaction
import json

from .models import Payment, License, PricingPlan
from .payos_service import payos_service, handle_payment_success


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@transaction.atomic
def create_payment(request):
    """
    Tạo đơn hàng thanh toán
    
    POST /api/payment/create/
    {
        "license_id": "uuid",
        "pricing_plan_id": 1
    }
    """
    try:
        license_id = request.data.get('license_id')
        pricing_plan_id = request.data.get('pricing_plan_id')
        
        # Validate
        license_obj = License.objects.get(id=license_id, user=request.user)
        pricing_plan = PricingPlan.objects.get(id=pricing_plan_id, is_active=True)
        
        # Create payment record
        order_code = int(timezone.now().timestamp() * 1000) % 100000000  # Unique order code
        
        payment = Payment.objects.create(
            license=license_obj,
            pricing_plan=pricing_plan,
            order_code=str(order_code),
            amount_vnd=pricing_plan.price_vnd,
            amount_usd=pricing_plan.price_usd,
            status='pending',
            payment_method='payos'
        )
        
        # Create PayOS payment link
        description = f"License {pricing_plan.name} - {pricing_plan.duration_months} months"
        result = payos_service.create_payment_link(
            order_code=order_code,
            amount=pricing_plan.price_vnd,
            description=description[:25],
            buyer_name=request.user.get_full_name() or request.user.username,
            buyer_email=request.user.email,
            items=[{
                "name": pricing_plan.name,
                "quantity": 1,
                "price": pricing_plan.price_vnd
            }]
        )
        
        if result['success']:
            payment.payos_payment_link_id = result.get('paymentLinkId', '')
            payment.save()
            
            return Response({
                'status': 'success',
                'payment_id': str(payment.id),
                'order_code': payment.order_code,
                'checkoutUrl': result.get('checkoutUrl'),
                'qrCode': result.get('qrCode'),
                'amount': pricing_plan.price_vnd,
                'message': 'Payment link created successfully'
            }, status=status.HTTP_201_CREATED)
        else:
            payment.status = 'failed'
            payment.note = result.get('error', 'Unknown error')
            payment.save()
            
            return Response({
                'status': 'failed',
                'error': result.get('error'),
                'message': 'Failed to create payment link'
            }, status=status.HTTP_400_BAD_REQUEST)
            
    except License.DoesNotExist:
        return Response({
            'status': 'error',
            'error': 'License not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except PricingPlan.DoesNotExist:
        return Response({
            'status': 'error',
            'error': 'Pricing plan not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'status': 'error',
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
@transaction.atomic
def payos_webhook(request):
    """
    Webhook từ PayOS để xác nhận thanh toán
    
    POST /api/payment/webhook/payos/
    """
    try:
        webhook_body = request.data
        
        # Verify webhook signature
        if not payos_service.verify_webhook_data(webhook_body):
            return Response({
                'status': 'error',
                'message': 'Invalid webhook signature'
            }, status=status.HTTP_401_UNAUTHORIZED)
        
        # Get payment data
        data = webhook_body.get('data', {})
        order_code = str(data.get('orderCode', ''))
        transaction_id = data.get('reference', '')
        code = data.get('code', '')
        
        # Find payment record
        try:
            payment = Payment.objects.get(order_code=order_code)
        except Payment.DoesNotExist:
            print(f"❌ [WEBHOOK] Payment not found: {order_code}")
            return Response({
                'status': 'error',
                'message': f'Payment {order_code} not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # 🔧 FIX: Auto-update license when payment succeeds
        if code == '00':  # Success code from PayOS
            payment.status = 'completed'
            payment.transaction_id = transaction_id
            payment.paid_at = timezone.now()
            payment.save()
            
            # Call auto-update handler
            success = handle_payment_success(payment)
            
            if success:
                print(f"✅ [WEBHOOK] Payment {order_code} completed, license updated")
                return Response({
                    'status': 'success',
                    'message': 'Payment processed and license updated'
                }, status=status.HTTP_200_OK)
            else:
                print(f"⚠️ [WEBHOOK] Payment {order_code} processed but license update failed")
                return Response({
                    'status': 'partial',
                    'message': 'Payment received but license update failed'
                }, status=status.HTTP_200_OK)
        else:
            # Payment failed or pending
            payment.status = 'failed'
            payment.note = data.get('desc', 'Unknown error')
            payment.save()
            
            print(f"❌ [WEBHOOK] Payment {order_code} failed: {data.get('desc')}")
            return Response({
                'status': 'failed',
                'message': 'Payment failed'
            }, status=status.HTTP_200_OK)
            
    except Exception as e:
        print(f"❌ [WEBHOOK] Error: {e}")
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_payment_status(request, payment_id):
    """
    Lấy trạng thái thanh toán
    
    GET /api/payment/{payment_id}/status/
    """
    try:
        payment = Payment.objects.get(id=payment_id, license__user=request.user)
        
        return Response({
            'status': 'success',
            'payment_id': str(payment.id),
            'order_code': payment.order_code,
            'status': payment.status,
            'amount_vnd': payment.amount_vnd,
            'created_at': payment.created_at.isoformat(),
            'paid_at': payment.paid_at.isoformat() if payment.paid_at else None,
            'license': {
                'key': payment.license.license_key,
                'expire_date': payment.license.expire_date.isoformat(),
                'days_remaining': payment.license.days_remaining(),
                'license_type': payment.license.license_type,
                'status': payment.license.status
            }
        })
    except Payment.DoesNotExist:
        return Response({
            'status': 'error',
            'error': 'Payment not found'
        }, status=status.HTTP_404_NOT_FOUND)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_payments(request):
    """
    Danh sách thanh toán của user
    
    GET /api/payment/list/
    """
    payments = Payment.objects.filter(
        license__user=request.user
    ).order_by('-created_at')[:20]
    
    data = []
    for payment in payments:
        data.append({
            'id': str(payment.id),
            'order_code': payment.order_code,
            'status': payment.status,
            'amount_vnd': payment.amount_vnd,
            'created_at': payment.created_at.isoformat(),
            'license_key': payment.license.license_key,
            'pricing_plan': payment.pricing_plan.name if payment.pricing_plan else None
        })
    
    return Response({
        'status': 'success',
        'count': len(data),
        'payments': data
    })
