# 🔄 Auto-Update System - Setup Guide

## 📋 Tổng Quan

Hệ thống auto-update cho phép user:
1. ✅ Click nút "🔄 Check for Updates" trong Menu
2. ✅ Tự động phát hiện phiên bản mới trên GitHub
3. ✅ Tải về và cài đặt update
4. ✅ Tự động restart app

## 🚀 Cách Cài Đặt

### **Bước 1: Chuẩn Bị GitHub Repository**

```bash
# 1. Tạo repo mới trên GitHub
https://github.com/new
Repository name: my_trading_bot

# 2. Clone repo
git clone https://github.com/your-username/my_trading_bot.git
cd my_trading_bot

# 3. Push code của bạn
git add .
git commit -m "Initial commit"
git push origin main
```

### **Bước 2: Cấu Hình update_manager.py**

Mở file `update_manager.py` và thay đổi:

```python
# Line 19-20
GITHUB_REPO = "your-username/my_trading_bot"  # ← Thay đổi tên repo!

# Line 24
CURRENT_VERSION = "4.3.2"  # ← Cập nhật phiên bản hiện tại
```

**Ví dụ:**
```python
GITHUB_REPO = "john-doe/my_trading_bot"
CURRENT_VERSION = "4.3.2"
```

### **Bước 3: Tạo Release trên GitHub**

#### **3.1 Tạo ZIP file**

```bash
# Windows (PowerShell)
Compress-Archive -Path . -DestinationPath my_trading_bot_v4.3.3.zip `
  -Exclude @("*.git*", "__pycache__", ".venv", "venv", "logs", "updates", "app_backup")

# Linux/Mac (Bash)
zip -r my_trading_bot_v4.3.3.zip . \
  --exclude "*.git*" "__pycache__/*" ".venv/*" "venv/*" "logs/*" "updates/*" "app_backup/*"
```

#### **3.2 Tạo Release trên GitHub Web**

1. **Đi tới Releases:**
   - Truy cập: `https://github.com/your-username/my_trading_bot/releases`

2. **Click "Create a new release"**

3. **Điền thông tin:**
   ```
   Tag: v4.3.3
   Title: Release v4.3.3
   Description: 
   - Fixed signal bugs
   - Added DCA strategy
   - Improved performance
   ```

4. **Upload ZIP file:**
   - Kéo file `my_trading_bot_v4.3.3.zip` vào "Attach binaries"

5. **Click "Publish release"**

### **Bước 4: Test Update**

1. Người dùng mở Trading Bot (v4.3.2)
2. Click Menu → "🔄 Check for Updates"
3. Nên thấy: "New version 4.3.3 available"
4. Click "Download & Install"
5. Chờ download hoàn tất
6. App tự động restart với v4.3.3

## 📝 Workflow Hàng Tuần

Mỗi lần bạn có phiên bản mới:

```bash
# 1. Cập nhật CURRENT_VERSION trong update_manager.py
CURRENT_VERSION = "4.3.3"

# 2. Push code
git add .
git commit -m "v4.3.3: Fixed signal bugs"
git push origin main

# 3. Tạo ZIP file
Compress-Archive -Path . -DestinationPath my_trading_bot_v4.3.3.zip ...

# 4. Tạo Release trên GitHub
# (Làm qua GitHub Web UI)
```

## 🔧 Cấu Hình Chi Tiết

### **Các folder/file được bỏ qua khi backup:**

Khi cài update, những file này sẽ được backup:
- ✅ `app.py` (keep settings)
- ✅ `config.json`
- ✅ `notification_config.json`
- ✅ `risk_management/` folder
- ✅ `logs/` folder

Những file/folder bị thay thế:
- ❌ Tất cả Python modules (để cập nhật code)
- ❌ UI files

### **Rollback (quay lại phiên bản cũ):**

Nếu update gặp vấn đề:

```bash
# Backup sẽ được lưu tự động trong folder: app_backup/
# Restore bằng tay:
Copy content từ app_backup/ → workspace
```

## ⚙️ Troubleshooting

### **Problem: GitHub API rate limit**
```
Error: GitHub API returned 403
```

**Solution:** Tạo Personal Access Token:
1. Đi tới: https://github.com/settings/tokens
2. Click "Generate new token"
3. Scope: `public_repo`
4. Thêm token vào request header (advanced)

### **Problem: ZIP file quá lớn**

**Solution:** 
- Loại bỏ `__pycache__`, `.git`, venv folder
- Nén images/videos nếu có
- Giới hạn tối đa ~200MB

### **Problem: Update không tìm thấy**

**Kiểm tra:**
```
1. GITHUB_REPO = "username/repo" đúng không?
2. Version tag có format v4.3.3 không?
3. ZIP file attachment có trong release không?
4. Internet connection có bình thường không?
```

## 📊 Version Comparison Logic

```python
# Cách so sánh version:
4.3.3 > 4.3.2  ✅ Update available
4.3.2 = 4.3.2  ❌ Already latest
4.3.1 < 4.3.2  ❌ Downgrade (không cho phép)
```

## 🎯 Features

✅ **Auto-Detection** - Kiểm tra version tự động
✅ **Progress Bar** - Hiển thị tiến trình tải
✅ **Auto Backup** - Backup file cũ trước cài đặt
✅ **Auto Rollback** - Rollback nếu cài đặt thất bại
✅ **Auto Restart** - Restart app sau cài đặt
✅ **Offline Support** - Kiểm tra update offline (nếu cần)

## 📦 File Structure

```
my_trading_bot/
├── app.py
├── update_manager.py  ← Auto-update manager
├── current_version.txt ← Auto-generated
├── updates/           ← Downloaded ZIP files
│   ├── my_trading_bot_v4.3.3.zip
│   └── ...
├── app_backup/        ← Backup trước cài đặt
│   ├── app.py
│   ├── config.json
│   └── ...
└── ...
```

## 🔐 Security Best Practices

1. **HTTPS only** - Luôn dùng HTTPS khi download
2. **File verification** - Update manager verify file integrity
3. **Backup before install** - Auto backup các file cấu hình
4. **Test releases** - Test update trước release production

## 📞 Support

- **GitHub Issues**: `https://github.com/your-username/my_trading_bot/issues`
- **Release Notes**: `https://github.com/your-username/my_trading_bot/releases`
- **Discussions**: `https://github.com/your-username/my_trading_bot/discussions`

---

**Status:** ✅ Setup Complete
**Last Updated:** 2025-12-17
