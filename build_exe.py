"""
🛡️ BUILD PROTECTED TRADING BOT EXECUTABLE
==========================================
Script này sẽ đóng gói Trading Bot thành file .exe được bảo vệ.

Yêu cầu:
1. PyInstaller: pip install pyinstaller
2. (Tùy chọn) Visual Studio Build Tools cho Nuitka

Chạy script:
    python build_exe.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Cấu hình
APP_NAME = "TradingBot"
MAIN_FILE = "app.py"
ICON_FILE = "robot_icon.ico"  # Đổi nếu có icon khác
VERSION = "4.3.2"

# Thư mục làm việc
BASE_DIR = Path(__file__).parent
BUILD_DIR = BASE_DIR / "build"
DIST_DIR = BASE_DIR / "dist"

def clean_build():
    """Xóa thư mục build cũ"""
    print("🧹 Cleaning old build files...")
    
    dirs_to_clean = [
        BUILD_DIR,
        DIST_DIR,
        BASE_DIR / f"{MAIN_FILE.replace('.py', '')}.build",
        BASE_DIR / f"{MAIN_FILE.replace('.py', '')}.dist",
        BASE_DIR / f"{MAIN_FILE.replace('.py', '')}.onefile-build",
    ]
    
    for d in dirs_to_clean:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"   Deleted: {d}")
    
    # Xóa __pycache__
    for pycache in BASE_DIR.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)
    
    # Xóa file .pyc
    for pyc in BASE_DIR.rglob("*.pyc"):
        pyc.unlink()
    
    print("✅ Cleanup complete!")


def collect_data_files():
    """Thu thập các file dữ liệu cần thiết"""
    data_files = []
    
    # Thêm các thư mục dữ liệu
    data_dirs = [
        "data",
        "ai_models/saved",
        "ai_server/saved_models",
        "indicator_output",
        "news_output",
        "pattern_price",
        "pattern_signals",
        "trendline_sr",
        "templates",
    ]
    
    for dir_name in data_dirs:
        dir_path = BASE_DIR / dir_name
        if dir_path.exists():
            data_files.append(f"--add-data={dir_path};{dir_name}")
    
    # Thêm các file config
    config_files = [
        "ai_server_config.json",
        "ai_trading_config.json",
        "license_config.json",
        "notification_config.json",
        "smart_entry_config.json",
        "trailing_stop_config.json",
        "mt5_data_config.ini",
    ]
    
    for cf in config_files:
        cf_path = BASE_DIR / cf
        if cf_path.exists():
            data_files.append(f"--add-data={cf_path};.")
    
    return data_files


def build_with_pyinstaller():
    """Build với PyInstaller (đơn giản, nhanh)"""
    print("\n" + "=" * 60)
    print("🔨 BUILDING WITH PYINSTALLER")
    print("=" * 60)
    
    # Tạo spec file tùy chỉnh
    hidden_imports = [
        "PyQt5",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
        "MetaTrader5",
        "pandas",
        "numpy",
        "requests",
        "sklearn",
        "torch",
        "xgboost",
        "ta",
        "python-dotenv",
        "sqlite3",
        "json",
        "threading",
        "multiprocessing",
    ]
    
    # Build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",                    # Single executable
        "--windowed",                   # No console window (GUI app)
        f"--name={APP_NAME}",           # Output name
        "--clean",                      # Clean cache
        "--noconfirm",                  # Don't ask confirmation
    ]
    
    # Add icon if exists
    icon_path = BASE_DIR / ICON_FILE
    if icon_path.exists():
        cmd.append(f"--icon={icon_path}")
    else:
        # Try png version
        png_icon = BASE_DIR / "robot_icon.png"
        if png_icon.exists():
            print(f"⚠️ .ico not found, using .png (may not work on Windows)")
    
    # Add hidden imports
    for hi in hidden_imports:
        cmd.append(f"--hidden-import={hi}")
    
    # Add data files
    data_files = collect_data_files()
    cmd.extend(data_files)
    
    # Add main file
    cmd.append(str(BASE_DIR / MAIN_FILE))
    
    print(f"\n📋 Command: {' '.join(cmd[:10])}...")
    print(f"📂 Working directory: {BASE_DIR}")
    
    # Run PyInstaller
    try:
        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            check=True,
            capture_output=False
        )
        
        exe_path = DIST_DIR / f"{APP_NAME}.exe"
        if exe_path.exists():
            print(f"\n✅ BUILD SUCCESSFUL!")
            print(f"📦 Output: {exe_path}")
            print(f"📏 Size: {exe_path.stat().st_size / 1024 / 1024:.2f} MB")
            return True
        else:
            print(f"\n❌ Build failed - exe not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed: {e}")
        return False


def build_with_nuitka():
    """Build với Nuitka (chậm hơn nhưng bảo mật tốt hơn)"""
    print("\n" + "=" * 60)
    print("🔨 BUILDING WITH NUITKA (Native Compilation)")
    print("=" * 60)
    print("⚠️ This requires Visual Studio Build Tools!")
    print("   Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",                  # Include all dependencies
        "--onefile",                     # Single executable
        "--enable-plugin=pyqt5",         # PyQt5 support
        "--windows-disable-console",     # No console window
        f"--output-filename={APP_NAME}.exe",
        "--remove-output",               # Clean temp files
        "--assume-yes-for-downloads",    # Auto download dependencies
    ]
    
    # Add icon if exists
    icon_path = BASE_DIR / ICON_FILE
    if icon_path.exists():
        cmd.append(f"--windows-icon-from-ico={icon_path}")
    
    # Add main file
    cmd.append(str(BASE_DIR / MAIN_FILE))
    
    print(f"\n📋 Command: {' '.join(cmd[:10])}...")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            check=True,
            capture_output=False
        )
        print("\n✅ Nuitka build completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Nuitka build failed: {e}")
        print("💡 Try installing Visual Studio Build Tools first")
        return False
    except FileNotFoundError:
        print("\n❌ Nuitka not found. Install with: pip install nuitka")
        return False


def create_installer_script():
    """Tạo script cài đặt cho người dùng cuối"""
    installer_content = '''@echo off
title Trading Bot Installer
echo ========================================
echo    TRADING BOT INSTALLER
echo ========================================
echo.

REM Check if running as admin
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Please run as Administrator!
    pause
    exit /b 1
)

REM Create installation directory
set INSTALL_DIR=C:\\TradingBot
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

REM Copy files
echo Copying files...
copy /Y "TradingBot.exe" "%INSTALL_DIR%\\"
copy /Y "*.json" "%INSTALL_DIR%\\" 2>nul
copy /Y "*.ini" "%INSTALL_DIR%\\" 2>nul
xcopy /E /I /Y "data" "%INSTALL_DIR%\\data\\" 2>nul

REM Create desktop shortcut
echo Creating desktop shortcut...
set SHORTCUT="%USERPROFILE%\\Desktop\\Trading Bot.lnk"
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%USERPROFILE%\\Desktop\\Trading Bot.lnk'); $s.TargetPath = '%INSTALL_DIR%\\TradingBot.exe'; $s.Save()"

echo.
echo ========================================
echo    INSTALLATION COMPLETE!
echo ========================================
echo.
echo Location: %INSTALL_DIR%
echo Desktop shortcut created.
echo.
pause
'''
    
    installer_path = DIST_DIR / "install.bat"
    installer_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(installer_path, 'w', encoding='utf-8') as f:
        f.write(installer_content)
    
    print(f"✅ Installer script created: {installer_path}")


def main():
    print("=" * 60)
    print("🛡️ TRADING BOT BUILD SYSTEM")
    print("=" * 60)
    print(f"📁 Project: {BASE_DIR}")
    print(f"📄 Main file: {MAIN_FILE}")
    print(f"🏷️ Version: {VERSION}")
    print()
    
    # Menu
    print("Select build method:")
    print("1. PyInstaller (Recommended - Fast, Reliable)")
    print("2. Nuitka (Slower but more secure, requires VS Build Tools)")
    print("3. Clean build files only")
    print("0. Exit")
    print()
    
    choice = input("Enter choice [1]: ").strip() or "1"
    
    if choice == "0":
        print("Bye!")
        return
    
    # Clean first
    clean_build()
    
    if choice == "3":
        print("✅ Cleanup complete!")
        return
    
    if choice == "2":
        success = build_with_nuitka()
    else:
        success = build_with_pyinstaller()
    
    if success:
        create_installer_script()
        print("\n" + "=" * 60)
        print("🎉 BUILD COMPLETE!")
        print("=" * 60)
        print(f"\n📦 Your executable is at: {DIST_DIR / f'{APP_NAME}.exe'}")
        print("\n⚠️ Important:")
        print("   - Copy all .json config files to the same folder as .exe")
        print("   - Copy 'data' folder if needed")
        print("   - First run may take longer (extracting files)")
    else:
        print("\n❌ Build failed. Check errors above.")


if __name__ == "__main__":
    main()
"""