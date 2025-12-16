"""
Document các FIX cho license status auto-update

🔧 FIXES APPLIED:

1. MODEL-LEVEL AUTO-UPDATE (models.py):
   - Added update_status_from_expiry_date() method to License model
   - Auto-set status to EXPIRED when expire_date < now()
   - Auto-recover from EXPIRED to ACTIVE when expire_date > now()
   - Called in License.save() to ensure status is always correct

2. PAYMENT SUCCESS AUTO-UPDATE (payos_service.py):
   - Added handle_payment_success(payment_obj) function
   - Auto-update license.expire_date when payment succeeds
   - Auto-set license.status = 'active'
   - Called in payment webhook and payment completion

3. ADMIN AUTO-UPDATE (admin.py):
   - Admin action extend_30_days() now calls update_status_from_expiry_date()
   - Admin action activate_licenses() now calls update_status_from_expiry_date()
   - Ensures status is correct after admin edits

4. VALIDATION IN SAVE (models.py):
   - Prevent illogical combinations:
     * Trial license must not exceed 7 days
     * Status must match expiry_date
     * Auto-recover from EXPIRED if expire_date extended

5. NOTIFICATIONS ON CHANGE:
   - UserChangeNotification.notify_license_change() called when:
     * License expires
     * License is renewed (recovered from EXPIRED)

FLOW:
   User Payment → Webhook → handle_payment_success() → license.save() → update_status_from_expiry_date()
   Admin Edit → extend_30_days() → license.save() → update_status_from_expiry_date()
   Client Check → License.is_valid() checks status + expiry_date
"""

from django.db import migrations

def create_initial_constraints(apps, schema_editor):
    """Create initial data constraints"""
    License = apps.get_model('users', 'License')
    
    # Fix any illogical licenses
    for license in License.objects.all():
        license.update_status_from_expiry_date()
        license.save()

class Migration(migrations.Migration):
    dependencies = [
        ('users', '0007_userchangenotification'),
    ]

    operations = [
        migrations.RunPython(create_initial_constraints),
    ]
