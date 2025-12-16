# 🚀 License Server Deployment Guide

## Kiến Trúc Hệ Thống

```
┌─────────────────────┐         INTERNET           ┌─────────────────────────┐
│   User's Computer   │ ◄───────────────────────► │   Your VPS/Cloud        │
│                     │                            │                         │
│  ┌───────────────┐  │       HTTPS (443)          │  ┌─────────────────┐    │
│  │ Trading Bot   │  │ ──────────────────────────►│  │ Nginx           │    │
│  │ (app.py)      │  │                            │  │ (Reverse Proxy) │    │
│  │               │  │  • Register                │  └────────┬────────┘    │
│  │ license_      │  │  • Login                   │           │             │
│  │ client.py     │  │  • Activate License        │  ┌────────▼────────┐    │
│  └───────────────┘  │  • Validate                │  │ Gunicorn        │    │
│                     │  • Heartbeat               │  │ (WSGI Server)   │    │
└─────────────────────┘                            │  └────────┬────────┘    │
                                                   │           │             │
                                                   │  ┌────────▼────────┐    │
                                                   │  │ Django App      │    │
                                                   │  │ (License Server)│    │
                                                   │  └────────┬────────┘    │
                                                   │           │             │
                                                   │  ┌────────▼────────┐    │
                                                   │  │ PostgreSQL      │    │
                                                   │  │ (Database)      │    │
                                                   │  └─────────────────┘    │
                                                   └─────────────────────────┘
```

## 📋 Yêu Cầu Server

### Minimum Requirements:
- **VPS/Cloud:** DigitalOcean, Vultr, AWS EC2, Azure, Google Cloud
- **OS:** Ubuntu 20.04+ / Debian 11+
- **RAM:** 1GB minimum, 2GB recommended
- **CPU:** 1 vCPU minimum
- **Storage:** 10GB SSD
- **Bandwidth:** 1TB/month (đủ cho ~10,000 users)

### Recommended Providers:
| Provider | Giá/tháng | RAM | Ghi chú |
|----------|-----------|-----|---------|
| DigitalOcean | $4-6 | 1GB | Dễ setup, có $200 free credit |
| Vultr | $5 | 1GB | Rẻ, nhiều locations |
| Contabo | $4.99 | 4GB | Rẻ nhất, performance tốt |
| AWS Lightsail | $5 | 1GB | Free tier 3 tháng |

## 🔧 Deployment Methods

### Method 1: Docker (Recommended) ⭐

**Bước 1: Chuẩn bị VPS**
```bash
# SSH vào VPS
ssh root@your-vps-ip

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose -y
```

**Bước 2: Upload code**
```bash
# Tạo thư mục
mkdir -p /var/www/license_server
cd /var/www/license_server

# Copy từ local (chạy trên máy local)
scp -r license_server/* root@your-vps-ip:/var/www/license_server/
```

**Bước 3: Cấu hình**
```bash
# Tạo file .env
cp .env.example .env
nano .env

# Sửa các giá trị:
# DJANGO_SECRET_KEY=<generate new key>
# DB_PASSWORD=<strong password>
# DOMAIN=license.yourdomain.com
```

**Bước 4: Deploy**
```bash
# Build và start
docker-compose up -d --build

# Tạo superuser
docker exec -it license_server python manage.py createsuperuser

# Check logs
docker logs -f license_server
```

**Bước 5: Setup SSL**
```bash
# Cài certbot
apt install certbot python3-certbot-nginx -y

# Lấy SSL certificate
certbot --nginx -d license.yourdomain.com
```

---

### Method 2: Manual Deployment

**Bước 1: Cài dependencies**
```bash
apt update && apt upgrade -y
apt install python3 python3-pip python3-venv nginx postgresql postgresql-contrib -y
```

**Bước 2: Setup PostgreSQL**
```bash
sudo -u postgres psql << EOF
CREATE DATABASE license_db;
CREATE USER license_user WITH PASSWORD 'your-strong-password';
ALTER ROLE license_user SET client_encoding TO 'utf8';
GRANT ALL PRIVILEGES ON DATABASE license_db TO license_user;
EOF
```

**Bước 3: Setup Django**
```bash
cd /var/www/license_server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn psycopg2-binary

# Migrations
python manage.py migrate --settings=license_server.settings_production
python manage.py collectstatic --noinput
python manage.py createsuperuser
```

**Bước 4: Setup Gunicorn service**
```bash
cat > /etc/systemd/system/license_server.service << EOF
[Unit]
Description=License Server
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/license_server
ExecStart=/var/www/license_server/venv/bin/gunicorn --workers 3 --bind unix:/var/www/license_server/gunicorn.sock license_server.wsgi:application

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl start license_server
systemctl enable license_server
```

**Bước 5: Setup Nginx**
```bash
cat > /etc/nginx/sites-available/license_server << EOF
server {
    listen 80;
    server_name license.yourdomain.com;

    location /static/ {
        alias /var/www/license_server/staticfiles/;
    }

    location / {
        proxy_pass http://unix:/var/www/license_server/gunicorn.sock;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

ln -s /etc/nginx/sites-available/license_server /etc/nginx/sites-enabled/
nginx -t && systemctl restart nginx
```

**Bước 6: SSL**
```bash
certbot --nginx -d license.yourdomain.com
```

---

## 🔐 Security Checklist

- [ ] Đổi DJANGO_SECRET_KEY
- [ ] Đổi DB_PASSWORD
- [ ] Set DEBUG=False
- [ ] Enable HTTPS
- [ ] Configure firewall (UFW)
- [ ] Setup fail2ban
- [ ] Regular backups

**Firewall setup:**
```bash
ufw allow ssh
ufw allow 80
ufw allow 443
ufw enable
```

---

## 🔄 Cập Nhật Trading Bot

Sau khi deploy xong, cập nhật `license_config.json` trong trading bot:

```json
{
    "server_url": "https://license.yourdomain.com/api",
    "heartbeat_interval": 60,
    "offline_grace_hours": 72,
    "verify_ssl": true
}
```

---

## 📊 Monitoring

**Check server status:**
```bash
# Docker
docker ps
docker logs license_server

# Manual
systemctl status license_server
journalctl -u license_server -f
```

**Check Nginx:**
```bash
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

---

## 🆘 Troubleshooting

**Connection refused:**
- Check firewall: `ufw status`
- Check service: `systemctl status license_server`
- Check ports: `netstat -tlnp`

**SSL errors:**
- Renew cert: `certbot renew`
- Check cert: `certbot certificates`

**Database errors:**
- Check PostgreSQL: `systemctl status postgresql`
- Check connection: `psql -U license_user -d license_db`

---

## 💰 Cost Estimation

| Component | Cost/month |
|-----------|------------|
| VPS (Contabo) | $5 |
| Domain | ~$1 (yearly) |
| SSL | Free (Let's Encrypt) |
| **Total** | **~$6/month** |

---

## 📞 Support

Nếu gặp vấn đề, kiểm tra:
1. Server logs: `docker logs license_server`
2. Nginx logs: `/var/log/nginx/error.log`
3. Database connectivity
4. Firewall rules
