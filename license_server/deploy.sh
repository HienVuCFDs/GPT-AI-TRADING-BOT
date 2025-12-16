#!/bin/bash
# =============================================================================
# Deployment Script for License Server
# Run this on your VPS/Cloud server
# =============================================================================

echo "========================================"
echo "License Server Deployment Script"
echo "========================================"

# Exit on error
set -e

# Configuration
APP_DIR="/var/www/license_server"
VENV_DIR="$APP_DIR/venv"
USER="www-data"
DOMAIN="license.yourdomain.com"  # CHANGE THIS

# =============================================================================
# Step 1: Install system dependencies
# =============================================================================
echo "[1/8] Installing system dependencies..."
apt-get update
apt-get install -y python3 python3-pip python3-venv postgresql postgresql-contrib nginx certbot python3-certbot-nginx

# =============================================================================
# Step 2: Create application directory
# =============================================================================
echo "[2/8] Creating application directory..."
mkdir -p $APP_DIR
mkdir -p $APP_DIR/logs

# =============================================================================
# Step 3: Setup PostgreSQL database
# =============================================================================
echo "[3/8] Setting up PostgreSQL database..."
sudo -u postgres psql << EOF
CREATE DATABASE license_db;
CREATE USER license_user WITH PASSWORD 'your-strong-password-here';
ALTER ROLE license_user SET client_encoding TO 'utf8';
ALTER ROLE license_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE license_user SET timezone TO 'Asia/Ho_Chi_Minh';
GRANT ALL PRIVILEGES ON DATABASE license_db TO license_user;
EOF

# =============================================================================
# Step 4: Setup Python virtual environment
# =============================================================================
echo "[4/8] Setting up Python virtual environment..."
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install psycopg2-binary  # PostgreSQL driver

# =============================================================================
# Step 5: Setup environment variables
# =============================================================================
echo "[5/8] Setting up environment variables..."
cat > $APP_DIR/.env << EOF
DJANGO_SECRET_KEY=$(python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())")
DJANGO_SETTINGS_MODULE=license_server.settings_production
DB_NAME=license_db
DB_USER=license_user
DB_PASSWORD=your-strong-password-here
DB_HOST=localhost
DB_PORT=5432
EOF

# =============================================================================
# Step 6: Django setup
# =============================================================================
echo "[6/8] Running Django migrations..."
export DJANGO_SETTINGS_MODULE=license_server.settings_production
python manage.py collectstatic --noinput
python manage.py migrate
python manage.py createsuperuser --noinput --username admin --email admin@yourdomain.com || true

# =============================================================================
# Step 7: Setup Gunicorn systemd service
# =============================================================================
echo "[7/8] Setting up Gunicorn service..."
cat > /etc/systemd/system/license_server.service << EOF
[Unit]
Description=License Server Gunicorn daemon
After=network.target

[Service]
User=$USER
Group=$USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$VENV_DIR/bin/gunicorn --access-logfile - --workers 3 --bind unix:$APP_DIR/license_server.sock license_server.wsgi:application

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl start license_server
systemctl enable license_server

# =============================================================================
# Step 8: Setup Nginx
# =============================================================================
echo "[8/8] Setting up Nginx..."
cat > /etc/nginx/sites-available/license_server << EOF
server {
    listen 80;
    server_name $DOMAIN;

    location = /favicon.ico { access_log off; log_not_found off; }
    
    location /static/ {
        root $APP_DIR;
    }

    location / {
        include proxy_params;
        proxy_pass http://unix:$APP_DIR/license_server.sock;
    }
}
EOF

ln -sf /etc/nginx/sites-available/license_server /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx

# =============================================================================
# Step 9: Setup SSL with Let's Encrypt
# =============================================================================
echo "[9/9] Setting up SSL certificate..."
certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@yourdomain.com

echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo ""
echo "Your License Server is now running at:"
echo "  https://$DOMAIN"
echo ""
echo "Admin panel: https://$DOMAIN/admin/"
echo ""
echo "IMPORTANT: Remember to:"
echo "  1. Change the database password in .env"
echo "  2. Set a strong Django admin password"
echo "  3. Update DOMAIN variable in this script"
echo "========================================"
