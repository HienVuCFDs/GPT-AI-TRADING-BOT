"""
Auto-Update Manager v3.0 - Update cho EXE Build (Onedir)
Cập nhật folder _internal/ từ ZIP trên GitHub Releases

Quy trình:
1. Kiểm tra version mới trên GitHub
2. Tải ZIP chứa _internal/ folder
3. Backup _internal/ cũ
4. Extract và thay thế _internal/ mới
5. Restart app
"""
import requests
import json
import os
import sys
import shutil
import zipfile
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from PyQt5.QtCore import QThread, QObject, pyqtSignal, Qt
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
        QPushButton, QMessageBox, QTextEdit
    )
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


class UpdateManager:
    """GitHub Releases Auto-Update Manager - EXE Onedir Update
    
    Supports 2 types of updates:
    1. FULL UPDATE: update_vX.X.X.zip - Full _internal/ folder (2-4GB)
    2. PATCH UPDATE: patch_vX.X.X.zip - Only changed files (10-500KB)
    
    Patch files are preferred for small hotfixes.
    """
    
    # ============ CONFIGURATION ============
    GITHUB_REPO = "HienVuCFDs/GPT-AI-TRADING-BOT"
    GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
    
    # Current version - Đọc từ file VERSION (dynamic) hoặc fallback hardcode
    CURRENT_VERSION = "4.4.0"  # Fallback version
    
    # Folders
    UPDATES_DIR = Path("updates")
    BACKUP_DIR = Path("_internal_backup")
    INTERNAL_DIR = "_internal"
    
    # Patch settings
    PATCH_PREFIX = "patch_"  # patch_v4.3.4.zip
    UPDATE_PREFIX = "update_"  # update_v4.3.4.zip
    
    def __init__(self):
        self.update_available = False
        self.latest_version = None
        self.latest_info = None
        self.patch_info = None  # Thông tin patch (nếu có)
        self.errors = []
        self.is_exe_mode = getattr(sys, 'frozen', False)
        self.app_dir = self._get_app_directory()
        self._pending_update_dir = None  # Folder chứa files cần copy sau restart
        
        # Đọc version từ file VERSION hoặc current_version.txt nếu có
        self.CURRENT_VERSION = self._read_current_version()
    
    def _read_current_version(self) -> str:
        """Đọc phiên bản hiện tại từ file VERSION hoặc current_version.txt"""
        # Thử đọc từ _internal/VERSION (ưu tiên)
        version_file = self.app_dir / self.INTERNAL_DIR / "VERSION"
        print(f"[DEBUG] Checking VERSION file: {version_file}")
        print(f"[DEBUG] File exists: {version_file.exists()}")
        if version_file.exists():
            try:
                ver = version_file.read_text(encoding='utf-8').strip()
                print(f"[DEBUG] VERSION file content: '{ver}'")
                if ver:
                    return ver
            except Exception as e:
                print(f"[DEBUG] Error reading VERSION file: {e}")
                pass
        
        # Fallback: đọc từ current_version.txt ở root
        version_file = self.app_dir / "current_version.txt"
        print(f"[DEBUG] Checking current_version.txt: {version_file}")
        print(f"[DEBUG] File exists: {version_file.exists()}")
        if version_file.exists():
            try:
                ver = version_file.read_text(encoding='utf-8').strip()
                print(f"[DEBUG] current_version.txt content: '{ver}'")
                if ver:
                    return ver
            except Exception as e:
                print(f"[DEBUG] Error reading current_version.txt: {e}")
                pass
        
        # Fallback cuối: dùng hằng số class
        fallback_ver = self.__class__.CURRENT_VERSION
        print(f"[DEBUG] Using fallback version: {fallback_ver}")
        return fallback_ver
    
    def _get_app_directory(self) -> Path:
        """Lấy thư mục chứa app (parent của _internal/)"""
        if self.is_exe_mode:
            return Path(sys.executable).parent
        else:
            # Running as script - need to get parent of _internal/
            # __file__ is in _internal/, so parent is the app dir
            current = Path(__file__).parent
            if current.name == '_internal':
                return current.parent
            return current
    
    def check_updates(self) -> bool:
        """Kiểm tra phiên bản mới trên GitHub"""
        try:
            print("🔄 Checking for updates from GitHub...")
            print(f"📦 Mode: {'EXE' if self.is_exe_mode else 'Python Script'}")
            print(f"📁 App dir: {self.app_dir}")
            print(f"📌 Current version: v{self.CURRENT_VERSION}")
            print(f"[DEBUG] sys.frozen: {getattr(sys, 'frozen', False)}")
            print(f"[DEBUG] sys.executable: {sys.executable if hasattr(sys, 'executable') else 'N/A'}")
            
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'GoldKiller-Updater/3.0'
            }
            
            response = requests.get(self.GITHUB_API_URL, headers=headers, timeout=15)
            
            if response.status_code != 200:
                self.errors.append(f"GitHub API returned {response.status_code}")
                print(f"❌ GitHub API error: {response.status_code}")
                # Fallback to local update packages
                return self._check_local_updates()
            
            releases = response.json()
            
            if not releases:
                print("❌ No releases found")
                # Fallback to local update packages
                return self._check_local_updates()
            
            # Lấy release mới nhất
            data = releases[0]
            tag_name = data.get('tag_name', '').lstrip('v')
            release_name = data.get('name', '')
            body = data.get('body', '')
            is_prerelease = data.get('prerelease', False)
            
            print(f"📌 Latest release: v{tag_name} {'(Pre-release)' if is_prerelease else ''}")
            print(f"[DEBUG] Version comparison: '{tag_name}' vs '{self.CURRENT_VERSION}'")
            print(f"[DEBUG] Compare result: {self._compare_versions(tag_name, self.CURRENT_VERSION)}")
            
            # Tìm file ZIP
            # Tìm file ZIP - ưu tiên PATCH trước (nhỏ hơn)
            patch_asset = None
            full_asset = None
            
            for asset in data.get('assets', []):
                name_lower = asset['name'].lower()
                if name_lower.endswith('.zip'):
                    if name_lower.startswith('patch_'):
                        patch_asset = asset
                    elif name_lower.startswith('update_'):
                        full_asset = asset
                    elif not full_asset:  # Fallback for any .zip
                        full_asset = asset
            
            # Ưu tiên patch nếu có (nhỏ hơn nhiều)
            target_asset = patch_asset or full_asset
            is_patch = patch_asset is not None
            
            if not target_asset:
                self.errors.append("No ZIP file in release")
                print("❌ No ZIP file found in release")
                # Fallback to local update packages
                return self._check_local_updates()
            
            # Log loại update
            if is_patch:
                size_kb = target_asset['size'] / 1024
                print(f"📦 Found PATCH update: {target_asset['name']} ({size_kb:.1f} KB)")
            else:
                size_mb = target_asset['size'] / (1024*1024)
                print(f"📦 Found FULL update: {target_asset['name']} ({size_mb:.1f} MB)")
            
            # So sánh version
            if self._compare_versions(tag_name, self.CURRENT_VERSION) > 0:
                print(f"✅ Update available: v{tag_name}")
                
                self.update_available = True
                self.latest_version = tag_name
                self.latest_info = {
                    'version': tag_name,
                    'release_name': release_name,
                    'notes': body or 'No release notes',
                    'download_url': target_asset['browser_download_url'],
                    'filename': target_asset['name'],
                    'size': target_asset['size'],
                    'size_mb': round(target_asset['size'] / (1024*1024), 2),
                    'size_kb': round(target_asset['size'] / 1024, 1),
                    'is_patch': is_patch,  # True nếu là patch update
                }
                
                # Lưu thông tin full update nếu có cả 2
                if is_patch and full_asset:
                    self.patch_info = self.latest_info.copy()
                    self.latest_info['full_update'] = {
                        'download_url': full_asset['browser_download_url'],
                        'filename': full_asset['name'],
                        'size_mb': round(full_asset['size'] / (1024*1024), 2),
                    }
                
                return True
            else:
                print(f"✅ Already on latest version (v{self.CURRENT_VERSION})")
                # Also check if there is a local package available (useful for offline/manual patches)
                local_ok = self._check_local_updates()
                if local_ok:
                    print("📦 Local update package found")
                    return True
                return False
                
        except requests.exceptions.Timeout:
            self.errors.append("Connection timeout")
            print("❌ Connection timeout")
            # Fallback to local update packages
            return self._check_local_updates()
        except requests.exceptions.ConnectionError:
            self.errors.append("Connection error")
            print("❌ Connection error")
            # Fallback to local update packages
            return self._check_local_updates()
        except Exception as e:
            self.errors.append(str(e))
            print(f"❌ Error: {e}")
            # Fallback to local update packages
            return self._check_local_updates()

    def _check_local_updates(self) -> bool:
        """Kiểm tra gói cập nhật cục bộ trong thư mục updates/ (offline fallback)

        Ưu tiên PATCH nếu có. Chỉ đề xuất cài đặt khi tìm thấy version > CURRENT_VERSION.
        Cấu trúc file: patch_vX.Y.Z.zip hoặc update_vX.Y.Z.zip, chứa thư mục _internal/ ở root.
        """
        try:
            self.UPDATES_DIR.mkdir(exist_ok=True)
            candidates = []

            for item in self.UPDATES_DIR.glob('*.zip'):
                name = item.name
                m = re.match(r'(?i)^(patch|update)_v(\d+\.\d+\.\d+)\.zip$', name)
                if not m:
                    continue
                kind = m.group(1).lower()
                ver = m.group(2)
                # Only consider versions greater than current
                if self._compare_versions(ver, self.CURRENT_VERSION) > 0:
                    candidates.append((ver, kind, item))

            if not candidates:
                print("ℹ️ No suitable local update packages found in updates/")
                return False

            # Pick highest version; if multiple with same version, prefer patch
            candidates.sort(key=lambda x: tuple(int(p) for p in x[0].split('.')))  # sort ascending by version
            best_ver = candidates[-1][0]
            best = max([c for c in candidates if c[0] == best_ver], key=lambda x: 1 if x[1] == 'patch' else 0)

            ver, kind, path_obj = best
            size = path_obj.stat().st_size

            self.update_available = True
            self.latest_version = ver
            self.latest_info = {
                'version': ver,
                'release_name': f"Local {kind.upper()} {ver}",
                'notes': 'Local package from updates/',
                'download_url': str(path_obj.resolve()),
                'filename': path_obj.name,
                'size': size,
                'size_mb': round(size / (1024*1024), 2),
                'size_kb': round(size / 1024, 1),
                'is_patch': (kind == 'patch'),
                'is_local': True,
            }

            print(f"📦 Found LOCAL {kind.upper()} update: {path_obj.name} ({size/1024:.1f} KB)")
            return True

        except Exception as e:
            print(f"❌ Local update check failed: {e}")
            return False
    
    def download_update(self, progress_callback=None) -> Optional[str]:
        """Tải file update từ GitHub"""
        if not self.update_available or not self.latest_info:
            return None
        
        try:
            # If this is a local package, skip download
            if self.latest_info.get('is_local'):
                local_path = Path(self.latest_info['download_url'])
                if progress_callback:
                    progress_callback(100)
                print(f"📦 Using local package: {local_path}")
                return str(local_path)

            print(f"📥 Downloading {self.latest_info['filename']}...")
            
            self.UPDATES_DIR.mkdir(exist_ok=True)
            
            url = self.latest_info['download_url']
            file_path = self.UPDATES_DIR / self.latest_info['filename']
            
            response = requests.get(url, stream=True, timeout=300)
            total_size = int(response.headers.get('content-length', 0))
            
            downloaded = 0
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size:
                            progress_callback(int(downloaded / total_size * 100))
            
            print(f"✅ Downloaded: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.errors.append(f"Download failed: {e}")
            print(f"❌ Download failed: {e}")
            return None
    
    def install_update(self, zip_path: str, progress_callback=None) -> bool:
        """Cài đặt update - hỗ trợ cả FULL và PATCH update
        
        FULL update: Thay thế toàn bộ _internal/
        PATCH update: Chỉ copy đè các file thay đổi vào _internal/
        """
        try:
            zip_path = Path(zip_path)
            if not zip_path.exists():
                self.errors.append("ZIP file not found")
                return False
            
            # Xác định loại update từ tên file
            is_patch = zip_path.name.lower().startswith('patch_')
            update_type = "PATCH" if is_patch else "FULL"
            
            print(f"📦 Installing {update_type} update from: {zip_path.name}")
            
            internal_path = self.app_dir / self.INTERNAL_DIR
            backup_path = self.app_dir / self.BACKUP_DIR
            
            # 1. Backup _internal/ cũ (chỉ full update hoặc lần đầu)
            if self.is_exe_mode and internal_path.exists():
                if is_patch:
                    # Patch: chỉ backup các file sẽ bị thay thế
                    print("💾 Creating partial backup...")
                else:
                    # Full: backup toàn bộ
                    print("💾 Backing up _internal/...")
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.copytree(internal_path, backup_path)
                    print(f"  ✅ Backup created: {backup_path}")
                if progress_callback:
                    progress_callback(20)
            
            # 2. Extract ZIP
            print("📂 Extracting update...")
            extract_dir = self.UPDATES_DIR / "extracted"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            if progress_callback:
                progress_callback(50)
            
            # 3. Tìm folder _internal trong ZIP
            new_internal = None
            for item in extract_dir.iterdir():
                if item.name == self.INTERNAL_DIR and item.is_dir():
                    new_internal = item
                    break
                # Nếu ZIP có folder con
                if item.is_dir():
                    sub_internal = item / self.INTERNAL_DIR
                    if sub_internal.exists():
                        new_internal = sub_internal
                        break
            
            if not new_internal:
                self.errors.append("_internal folder not found in ZIP")
                print("❌ _internal folder not found in ZIP")
                return False
            
            # 4. Áp dụng update
            if self.is_exe_mode:
                # 🔧 FIX v4.3.9: Dùng pending_update folder
                # Vì khi app đang chạy, các file .pyc/.pyd bị lock không thể copy
                # Batch script sẽ copy SAU khi app đóng
                
                pending_update = self.app_dir / "pending_update"
                if pending_update.exists():
                    shutil.rmtree(pending_update)
                pending_update.mkdir(parents=True)
                
                print("📦 Preparing update files...")
                
                # Copy _internal vào pending
                pending_internal = pending_update / self.INTERNAL_DIR
                shutil.copytree(new_internal, pending_internal)
                print(f"  ✅ Prepared {len(list(pending_internal.rglob('*')))} files")
                
                # Copy VERSION file nếu có
                version_in_zip = extract_dir / "VERSION"
                if version_in_zip.exists():
                    shutil.copy2(version_in_zip, pending_update / "VERSION")
                    print("  ✅ Prepared VERSION file")
                
                # Copy icon files nếu có
                for ico_file in extract_dir.glob("*.ico"):
                    shutil.copy2(ico_file, pending_update / ico_file.name)
                    print(f"  ✅ Prepared {ico_file.name}")
                
                # Lưu thông tin để restart_app biết cần copy
                self._pending_update_dir = pending_update
            else:
                # Python mode - copy các file .py
                print("🔄 Copying updated files...")
                for item in new_internal.iterdir():
                    dst = self.app_dir / item.name
                    if item.is_file():
                        shutil.copy2(item, dst)
                        print(f"  ✅ {item.name}")
            
            if progress_callback:
                progress_callback(80)
            
            # 5. Cleanup
            print("🧹 Cleaning up...")
            zip_path.unlink()
            shutil.rmtree(extract_dir)
            
            # 6. Update version file
            version_file = self.app_dir / "current_version.txt"
            version_file.write_text(self.latest_version, encoding='utf-8')
            
            if progress_callback:
                progress_callback(100)
            
            print(f"✅ Update installed: v{self.latest_version}")
            return True
            
        except Exception as e:
            self.errors.append(f"Install failed: {e}")
            print(f"❌ Install failed: {e}")
            
            # Restore backup
            self._restore_backup()
            return False
    
    def _apply_patch(self, src_dir: Path, dst_dir: Path) -> int:
        """Áp dụng patch - copy đè các file từ src vào dst (recursive)
        
        Giữ nguyên các file không có trong patch.
        Chỉ thay thế/thêm các file có trong patch.
        
        Returns: số file đã patch
        """
        patched_count = 0
        
        for item in src_dir.iterdir():
            dst_item = dst_dir / item.name
            
            if item.is_file():
                # Copy file (đè nếu tồn tại)
                dst_item.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst_item)
                print(f"    📄 {item.name}")
                patched_count += 1
                
            elif item.is_dir():
                # Đệ quy vào subfolder
                if not dst_item.exists():
                    dst_item.mkdir(parents=True, exist_ok=True)
                patched_count += self._apply_patch(item, dst_item)
        
        return patched_count
    
    def _restore_backup(self):
        """Khôi phục từ backup nếu update thất bại"""
        try:
            backup_path = self.app_dir / self.BACKUP_DIR
            internal_path = self.app_dir / self.INTERNAL_DIR
            
            if backup_path.exists():
                print("↩️ Restoring backup...")
                if internal_path.exists():
                    shutil.rmtree(internal_path)
                shutil.copytree(backup_path, internal_path)
                print("✅ Backup restored")
        except Exception as e:
            print(f"❌ Restore failed: {e}")
    
    def restart_app(self, pending_update_dir: Path = None):
        """Restart app sau khi update
        
        Args:
            pending_update_dir: Folder chứa files cần copy sau khi app đóng
        """
        try:
            print("🔄 Restarting...")
            
            if self.is_exe_mode:
                exe_path = sys.executable
                app_dir = str(self.app_dir)
                
                # Tạo script restart với khả năng copy pending updates
                restart_bat = self.app_dir / "restart_update.bat"
                
                if pending_update_dir and pending_update_dir.exists():
                    # Có pending update - script sẽ copy files trước khi restart
                    pending_dir = str(pending_update_dir)
                    restart_bat.write_text(f'''@echo off
chcp 65001 >nul
echo ===============================================
echo   Gold Killer AI Trading Bot - Update
echo ===============================================
echo.
echo Waiting for app to close...
timeout /t 3 /nobreak >nul

echo Applying update...
xcopy /E /Y /I "{pending_dir}\\*" "{app_dir}\\" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to copy some files
) else (
    echo [OK] Update applied successfully
)

echo Cleaning up...
rmdir /S /Q "{pending_dir}" 2>nul

echo Starting app...
timeout /t 1 /nobreak >nul
start "" "{exe_path}"

del /f /q "%~f0" 2>nul
exit
''', encoding='utf-8')
                else:
                    # Không có pending - restart bình thường
                    restart_bat.write_text(f'''@echo off
chcp 65001 >nul
echo Restarting Gold Killer AI Trading Bot...
timeout /t 2 /nobreak >nul
start "" "{exe_path}"
del /f /q "%~f0" 2>nul
exit
''', encoding='utf-8')
                
                subprocess.Popen(['cmd', '/c', str(restart_bat)], 
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
            else:
                subprocess.Popen([sys.executable, 'app.py'])
            
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ Restart failed: {e}")
    
    @staticmethod
    def _compare_versions(v1: str, v2: str) -> int:
        """So sánh version: 1 nếu v1 > v2, -1 nếu v1 < v2, 0 nếu bằng"""
        def parse(v):
            return tuple(int(x) for x in v.split('-')[0].split('.') if x.isdigit())
        try:
            p1, p2 = parse(v1), parse(v2)
            return (p1 > p2) - (p1 < p2)
        except:
            return 0


# ============ PyQt5 GUI ============

if GUI_AVAILABLE:
    
    class UpdateWorker(QObject):
        progress = pyqtSignal(int)
        status = pyqtSignal(str)
        finished = pyqtSignal(bool, str)  # success, message
        
        def __init__(self, update_manager: UpdateManager):
            super().__init__()
            self.manager = update_manager
        
        def run(self):
            try:
                # Download
                self.status.emit("Downloading update...")
                zip_path = self.manager.download_update(
                    progress_callback=lambda p: self.progress.emit(p // 2)
                )
                
                if not zip_path:
                    self.finished.emit(False, "Download failed")
                    return
                
                # Install
                self.status.emit("Installing update...")
                success = self.manager.install_update(
                    zip_path,
                    progress_callback=lambda p: self.progress.emit(50 + p // 2)
                )
                
                if success:
                    self.finished.emit(True, f"Updated to v{self.manager.latest_version}")
                else:
                    self.finished.emit(False, "Installation failed")
                    
            except Exception as e:
                self.finished.emit(False, str(e))
    
    
    class UpdateProgressDialog(QDialog):
        """Dialog hiển thị progress update"""
        
        def __init__(self, update_manager: UpdateManager, version_info: Dict, parent=None):
            super().__init__(parent)
            self.manager = update_manager
            self.version_info = version_info
            self.thread = None
            
            self.init_ui()
            self.start_update()
        
        def init_ui(self):
            self.setWindowTitle("Updating Gold Killer AI Trading Bot")
            self.setFixedSize(480, 250)
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            
            layout = QVBoxLayout()
            
            # Title
            title = QLabel(f"🔄 Updating to v{self.version_info['version']}")
            title.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFD700;")
            layout.addWidget(title)
            
            # File info
            info = QLabel(f"📦 {self.version_info.get('filename', 'update.zip')} ({self.version_info.get('size_mb', '?')} MB)")
            info.setStyleSheet("color: #888;")
            layout.addWidget(info)
            
            # Progress bar
            self.progress = QProgressBar()
            self.progress.setRange(0, 100)
            self.progress.setStyleSheet("""
                QProgressBar { border: 2px solid #444; border-radius: 5px; height: 25px; background: #2a2a2a; color: white; text-align: center; }
                QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FFD700, stop:1 #FFA500); }
            """)
            layout.addWidget(self.progress)
            
            # Status
            self.status_label = QLabel("Starting...")
            self.status_label.setStyleSheet("color: #aaa;")
            layout.addWidget(self.status_label)
            
            # Log
            self.log = QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(80)
            self.log.setStyleSheet("background: #1a1a1a; border: 1px solid #333; color: #0f0; font-family: Consolas;")
            layout.addWidget(self.log)
            
            # Button
            self.btn = QPushButton("Cancel")
            self.btn.clicked.connect(self.reject)
            self.btn.setStyleSheet("background: #444; color: white; padding: 8px 20px; border-radius: 4px;")
            layout.addWidget(self.btn)
            
            self.setLayout(layout)
            self.setStyleSheet("QDialog { background: #2a2a2a; } QLabel { color: white; }")
        
        def add_log(self, msg):
            self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        
        def start_update(self):
            self.thread = QThread()
            self.worker = UpdateWorker(self.manager)
            self.worker.moveToThread(self.thread)
            
            self.worker.progress.connect(self.progress.setValue)
            self.worker.status.connect(lambda s: (self.status_label.setText(s), self.add_log(s)))
            self.worker.finished.connect(self.on_finished)
            
            self.thread.started.connect(self.worker.run)
            self.thread.start()
        
        def on_finished(self, success: bool, message: str):
            self.thread.quit()
            self.thread.wait()
            
            if success:
                self.progress.setValue(100)
                self.add_log(f"✅ {message}")
                self.btn.setText("Restart Now")
                self.btn.setStyleSheet("background: #FFD700; color: black; font-weight: bold; padding: 8px 20px; border-radius: 4px;")
                self.btn.clicked.disconnect()
                self.btn.clicked.connect(self.do_restart)
                
                QMessageBox.information(self, "Update Complete",
                    f"✅ {message}\n\nClick 'Restart Now' to apply.")
            else:
                self.add_log(f"❌ {message}")
                QMessageBox.critical(self, "Update Failed", f"❌ {message}")
                self.reject()
        
        def do_restart(self):
            self.accept()
            # Truyền pending_update_dir nếu có
            pending_dir = getattr(self.manager, '_pending_update_dir', None)
            self.manager.restart_app(pending_update_dir=pending_dir)


if __name__ == "__main__":
    manager = UpdateManager()
    
    print("=" * 50)
    print("Gold Killer AI - Update Manager v3.0")
    print("=" * 50)
    print(f"Version: v{manager.CURRENT_VERSION}")
    print(f"Mode: {'EXE' if manager.is_exe_mode else 'Python'}")
    print(f"Path: {manager.app_dir}")
    print()
    
    if manager.check_updates():
        print()
        print(f"✅ Update: v{manager.latest_version}")
        print(f"   File: {manager.latest_info['filename']}")
        print(f"   Size: {manager.latest_info['size_mb']} MB")
    else:
        print()
        print("✅ Already on latest version")
