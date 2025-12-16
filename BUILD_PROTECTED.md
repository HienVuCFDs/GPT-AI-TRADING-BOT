# 🛡️ HƯỚNG DẪN ĐÓNG GÓI APP BẢO MẬT

## 📋 Tổng quan
File này hướng dẫn cách đóng gói Trading Bot thành file .exe được mã hóa, không thể đọc source code.

---

## 🔒 PHƯƠNG PHÁP 1: PYARMOR (Khuyến nghị)

PyArmor là công cụ mạnh nhất để bảo vệ Python code.

### Cài đặt
```powershell
pip install pyarmor
```

### Đóng gói cơ bản
```powershell
# Di chuyển đến thư mục project
cd "c:\Users\ADMIN\OneDrive\Desktop\my_trading_bot"

# Đóng gói với PyArmor + PyInstaller
pyarmor gen --pack onefile app.py
```

### Đóng gói nâng cao (bảo mật cao hơn)
```powershell
# Sử dụng license riêng
pyarmor gen --with-license outer --pack onefile app.py

# Thêm obfuscation mạnh
pyarmor gen --enable-jit --enable-themida --pack onefile app.py

# Giới hạn hardware (chỉ chạy trên máy được đăng ký)
pyarmor gen --bind-device --pack onefile app.py
```

### Cấu hình PyArmor nâng cao
Tạo file `pyarmor.cli.yaml`:
```yaml
obf:
  - mix_str: true
  - call_threshold: 1
  - restrict_module: 1
  - bcc: true
```

Sau đó chạy:
```powershell
pyarmor gen -r --pack onefile app.py
```

---

## 🔒 PHƯƠNG PHÁP 2: NUITKA

Nuitka compile Python sang C++ rồi compile thành native executable.

### Cài đặt
```powershell
pip install nuitka
pip install ordered-set  # Tùy chọn nhưng nên cài
```

### Đóng gói cơ bản
```powershell
python -m nuitka --standalone --onefile app.py
```

### Đóng gói với tối ưu (lâu hơn nhưng an toàn hơn)
```powershell
python -m nuitka --standalone --onefile --enable-plugin=pyqt5 --windows-disable-console app.py
```

### Đóng gói với obfuscation
```powershell
python -m nuitka --standalone --onefile ^
    --enable-plugin=pyqt5 ^
    --windows-icon-from-ico=robot_icon.ico ^
    --windows-company-name="Your Company" ^
    --windows-product-name="Trading Bot" ^
    --windows-file-version=4.3.2.0 ^
    --windows-product-version=4.3.2.0 ^
    --windows-file-description="AI Trading Bot" ^
    --remove-output ^
    app.py
```

---

## 🔒 PHƯƠNG PHÁP 3: PYINSTALLER + PYARMOR (Kết hợp)

### Bước 1: Obfuscate với PyArmor
```powershell
pyarmor gen -O dist/obf app.py license_guard.py license_client.py
```

### Bước 2: Đóng gói với PyInstaller
```powershell
cd dist/obf
pyinstaller --onefile --windowed --icon=../../robot_icon.ico app.py
```

---

## 🔐 CẤU HÌNH LICENSE SERVER

### Thay đổi URL server trong `license_guard.py`:
```python
class LicenseConfig:
    # Thay đổi URL này sang server thật của bạn
    LICENSE_SERVER = "https://your-domain.com/api"
    
    # Thay đổi secret key
    SECRET_KEY = "YOUR_UNIQUE_SECRET_KEY_HERE"
```

### Thay đổi trong `license_client.py`:
```python
DEFAULT_SERVER_URL = "https://your-domain.com/api"
```

---

## 📝 CHECKLIST TRƯỚC KHI ĐÓNG GÓI

1. [ ] Thay đổi `LICENSE_SERVER` URL trong `license_guard.py`
2. [ ] Thay đổi `SECRET_KEY` trong `license_guard.py`  
3. [ ] Thay đổi `DEFAULT_SERVER_URL` trong `license_client.py`
4. [ ] Xóa file `users_db.json` (chứa user test)
5. [ ] Xóa file `.license_cache.dat` (nếu có)
6. [ ] Xóa tất cả file `.pyc` và thư mục `__pycache__`
7. [ ] Test app sau khi đóng gói

---

## 🚀 SCRIPT TỰ ĐỘNG ĐÓNG GÓI

Tạo file `build_protected.bat`:
```batch
@echo off
echo === BUILDING PROTECTED TRADING BOT ===

REM Clean up
del /q /s __pycache__ 2>nul
del /q *.pyc 2>nul

REM Build với PyArmor
echo Building with PyArmor...
pyarmor gen --pack onefile app.py

echo === BUILD COMPLETE ===
echo Output: dist\app.exe
pause
```

---

## ⚠️ LƯU Ý QUAN TRỌNG

1. **KHÔNG bao giờ** commit `SECRET_KEY` thật lên Git
2. Sử dụng biến môi trường cho các key nhạy cảm trong production
3. Test kỹ trên máy khác trước khi phát hành
4. Giữ bản backup source code ở nơi an toàn
5. License Server phải có SSL/HTTPS cho production

---

## 📞 HỖ TRỢ

Nếu gặp lỗi khi đóng gói, kiểm tra:
1. Python version (khuyến nghị 3.10-3.11)
2. Cài đủ dependencies: `pip install -r requirements.txt`
3. Có Visual C++ Build Tools (cho Nuitka)
4. Quyền admin khi chạy lệnh build
