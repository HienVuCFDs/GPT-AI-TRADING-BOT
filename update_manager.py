"""
Auto-Update Manager - Quản lý cập nhật phiên bản từ GitHub Releases
"""
import requests
import json
import os
import sys
import hashlib
import shutil
import zipfile
import subprocess
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
    """GitHub Releases Auto-Update Manager"""
    
    # ============ CONFIGURATION ============
    # GitHub repository configuration
    GITHUB_REPO = "HienVuCFDs/GPT-AI-TRADING-BOT"  # ✅ Configured
    GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    
    # Current version (update this when you release a new version)
    CURRENT_VERSION = "4.3.2"
    
    # Auto-update folder
    UPDATES_DIR = Path("updates")
    BACKUP_DIR = Path("app_backup")
    
    def __init__(self):
        self.update_available = False
        self.latest_version = None
        self.latest_info = None
        self.errors = []
    
    def check_updates(self) -> bool:
        """
        Kiểm tra phiên bản mới trên GitHub
        
        Returns:
            bool: True nếu có update mới, False nếu đã là latest
        """
        try:
            print("🔄 Checking for updates from GitHub...")
            
            # Fetch latest release info từ GitHub API
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'Trading-Bot-Updater/1.0'
            }
            
            response = requests.get(
                self.GITHUB_API_URL,
                headers=headers,
                timeout=10
            )
            
            if response.status_code != 200:
                error = f"GitHub API returned {response.status_code}"
                self.errors.append(error)
                print(f"❌ {error}")
                return False
            
            data = response.json()
            
            # Extract version from tag_name (e.g., v4.3.2)
            tag_name = data.get('tag_name', '').lstrip('v')
            release_name = data.get('name', '')
            body = data.get('body', '')
            
            # Find the ZIP asset
            zip_asset = None
            for asset in data.get('assets', []):
                if asset['name'].endswith('.zip'):
                    zip_asset = asset
                    break
            
            if not zip_asset:
                error = "No ZIP file found in release assets"
                self.errors.append(error)
                print(f"❌ {error}")
                return False
            
            # Compare versions
            if self._compare_versions(tag_name, self.CURRENT_VERSION) > 0:
                print(f"✅ Update found: {tag_name}")
                
                self.update_available = True
                self.latest_version = tag_name
                self.latest_info = {
                    'version': tag_name,
                    'release_name': release_name,
                    'notes': body or 'No release notes',
                    'download_url': zip_asset['browser_download_url'],
                    'filename': zip_asset['name'],
                    'size': zip_asset['size'],
                    'size_mb': round(zip_asset['size'] / (1024*1024), 2),
                    'published_at': data.get('published_at', '')
                }
                
                return True
            else:
                print(f"✅ Already on latest version (v{self.CURRENT_VERSION})")
                self.update_available = False
                return False
                
        except requests.exceptions.Timeout:
            error = "Request timeout - check your internet connection"
            self.errors.append(error)
            print(f"❌ {error}")
            return False
            
        except requests.exceptions.ConnectionError:
            error = "Connection error - check your internet connection"
            self.errors.append(error)
            print(f"❌ {error}")
            return False
            
        except Exception as e:
            error = f"Update check failed: {str(e)}"
            self.errors.append(error)
            print(f"❌ {error}")
            return False
    
    def download_update(self, progress_callback=None) -> Optional[str]:
        """
        Tải file update từ GitHub
        
        Args:
            progress_callback: Callback function để report progress (0-100)
        
        Returns:
            str: Path đến file ZIP đã tải, hoặc None nếu thất bại
        """
        if not self.update_available or not self.latest_info:
            print("❌ No update available to download")
            return None
        
        try:
            print(f"📥 Downloading {self.latest_info['filename']}...")
            
            # Create updates directory
            self.UPDATES_DIR.mkdir(exist_ok=True)
            
            url = self.latest_info['download_url']
            filename = self.latest_info['filename']
            
            zip_path = self.UPDATES_DIR / filename
            
            # Download with progress
            response = requests.get(url, stream=True, timeout=30)
            total_size = int(response.headers.get('content-length', 0))
            
            downloaded = 0
            chunk_size = 8192
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size:
                            progress = (downloaded / total_size) * 100
                            progress_callback(int(progress))
            
            print(f"✅ Downloaded: {zip_path}")
            return str(zip_path)
            
        except Exception as e:
            error = f"Download failed: {str(e)}"
            self.errors.append(error)
            print(f"❌ {error}")
            return None
    
    def install_update(self, zip_path: str) -> bool:
        """
        Cài đặt update (extract + backup + replace)
        
        Args:
            zip_path: Path đến file ZIP
        
        Returns:
            bool: True nếu cài đặt thành công
        """
        try:
            print("📦 Installing update...")
            
            zip_path = Path(zip_path)
            if not zip_path.exists():
                error = f"ZIP file not found: {zip_path}"
                self.errors.append(error)
                print(f"❌ {error}")
                return False
            
            # 1. Backup current app
            print("💾 Creating backup...")
            if self.BACKUP_DIR.exists():
                shutil.rmtree(self.BACKUP_DIR)
            
            # Backup important files/folders
            files_to_backup = [
                'app.py',
                'config.json',
                'notification_config.json',
                'ai_trading_config.json',
                'ai_server_config.json',
                'risk_management/',
                'logs/'
            ]
            
            self.BACKUP_DIR.mkdir(exist_ok=True)
            
            for item in files_to_backup:
                src = Path(item)
                if src.exists():
                    dst = self.BACKUP_DIR / src.name
                    if src.is_dir():
                        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__'))
                    else:
                        shutil.copy2(src, dst)
                    print(f"  ✅ Backed up: {item}")
            
            print(f"✅ Backup created: {self.BACKUP_DIR}")
            
            # 2. Extract new files
            print("📂 Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(Path.cwd())
            
            print("✅ Files extracted")
            
            # 3. Clean up temp files
            zip_path.unlink()
            print("🧹 Temp files cleaned")
            
            # 4. Update version file
            version_file = Path('current_version.txt')
            version_file.write_text(self.latest_version)
            
            print(f"✅ Version updated: {self.latest_version}")
            return True
            
        except Exception as e:
            error = f"Installation failed: {str(e)}"
            self.errors.append(error)
            print(f"❌ {error}")
            
            # Restore backup
            if self.BACKUP_DIR.exists():
                print("↩️ Restoring backup...")
                try:
                    for item in self.BACKUP_DIR.iterdir():
                        dst = Path.cwd() / item.name
                        if dst.exists():
                            if dst.is_dir():
                                shutil.rmtree(dst)
                            else:
                                dst.unlink()
                        
                        if item.is_dir():
                            shutil.copytree(item, dst)
                        else:
                            shutil.copy2(item, dst)
                    
                    print("✅ Backup restored")
                except Exception as restore_error:
                    print(f"❌ Restore failed: {restore_error}")
            
            return False
    
    def restart_app(self):
        """Restart application"""
        try:
            print("🔄 Restarting application...")
            
            if sys.platform == 'win32':
                # Windows
                subprocess.Popen([sys.executable, 'app.py'])
            else:
                # Linux/Mac
                subprocess.Popen([sys.executable, 'app.py'])
            
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ Failed to restart: {e}")
    
    @staticmethod
    def _compare_versions(v1: str, v2: str) -> int:
        """
        Compare two version strings
        
        Args:
            v1: Version string (e.g., "4.3.2")
            v2: Version string (e.g., "4.3.1")
        
        Returns:
            int: 1 if v1 > v2, -1 if v1 < v2, 0 if equal
        """
        def parse_version(v):
            return tuple(map(int, v.split('.')))
        
        try:
            v1_parts = parse_version(v1)
            v2_parts = parse_version(v2)
            
            if v1_parts > v2_parts:
                return 1
            elif v1_parts < v2_parts:
                return -1
            else:
                return 0
        except:
            return 0


# ============ PyQt5 GUI Components ============

if GUI_AVAILABLE:
    
    class UpdateDownloadWorker(QObject):
        """Worker thread for downloading update"""
        progress = pyqtSignal(int)
        status = pyqtSignal(str)
        finished = pyqtSignal(str)  # Returns zip_path
        
        def __init__(self, update_manager: UpdateManager):
            super().__init__()
            self.update_manager = update_manager
        
        def run(self):
            """Download update in background"""
            try:
                self.status.emit("Starting download...")
                
                zip_path = self.update_manager.download_update(
                    progress_callback=lambda p: self.progress.emit(p)
                )
                
                self.finished.emit(zip_path or "")
                
            except Exception as e:
                print(f"❌ Download worker error: {e}")
                self.finished.emit("")
    
    
    class UpdateInstallWorker(QObject):
        """Worker thread for installing update"""
        progress = pyqtSignal(int)
        status = pyqtSignal(str)
        finished = pyqtSignal(bool)  # Success/failure
        
        def __init__(self, update_manager: UpdateManager, zip_path: str):
            super().__init__()
            self.update_manager = update_manager
            self.zip_path = zip_path
        
        def run(self):
            """Install update in background"""
            try:
                self.status.emit("Installing update...")
                self.progress.emit(30)
                
                success = self.update_manager.install_update(self.zip_path)
                
                if success:
                    self.progress.emit(100)
                    self.status.emit("Installation complete!")
                
                self.finished.emit(success)
                
            except Exception as e:
                print(f"❌ Install worker error: {e}")
                self.finished.emit(False)
    
    
    class UpdateProgressDialog(QDialog):
        """Progress dialog for update download and install"""
        
        def __init__(self, update_manager: UpdateManager, version_info: Dict, parent=None):
            super().__init__(parent)
            self.update_manager = update_manager
            self.version_info = version_info
            self.download_thread = None
            self.install_thread = None
            
            self.init_ui()
            self.start_download()
        
        def init_ui(self):
            """Initialize UI"""
            self.setWindowTitle("Updating Trading Bot")
            self.setGeometry(400, 400, 500, 250)
            
            layout = QVBoxLayout()
            
            # Title
            title = QLabel(f"🔄 Downloading v{self.version_info['version']}")
            title.setStyleSheet("font-size: 14px; font-weight: bold;")
            layout.addWidget(title)
            
            # Progress bar
            self.progress = QProgressBar()
            self.progress.setRange(0, 100)
            self.progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    height: 25px;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                }
            """)
            layout.addWidget(self.progress)
            
            # Status label
            self.status_label = QLabel("Starting download...")
            self.status_label.setStyleSheet("color: #666;")
            layout.addWidget(self.status_label)
            
            # Details text
            self.details_text = QTextEdit()
            self.details_text.setReadOnly(True)
            self.details_text.setMaximumHeight(80)
            self.details_text.setStyleSheet("""
                QTextEdit {
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    font-family: monospace;
                    font-size: 11px;
                }
            """)
            layout.addWidget(self.details_text)
            
            # Cancel button
            self.cancel_btn = QPushButton("Cancel")
            self.cancel_btn.clicked.connect(self.reject)
            layout.addWidget(self.cancel_btn)
            
            self.setLayout(layout)
        
        def start_download(self):
            """Start download in thread"""
            self.download_thread = QThread()
            self.download_worker = UpdateDownloadWorker(self.update_manager)
            self.download_worker.moveToThread(self.download_thread)
            
            self.download_worker.progress.connect(self.on_download_progress)
            self.download_worker.status.connect(self.on_download_status)
            self.download_worker.finished.connect(self.on_download_finished)
            
            self.download_thread.started.connect(self.download_worker.run)
            self.download_thread.start()
        
        def on_download_progress(self, progress: int):
            """Update progress bar"""
            self.progress.setValue(progress)
        
        def on_download_status(self, status: str):
            """Update status label"""
            self.status_label.setText(status)
            self.details_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {status}")
        
        def on_download_finished(self, zip_path: str):
            """Handle download completion"""
            self.download_thread.quit()
            self.download_thread.wait()
            
            if zip_path:
                # Start installation
                self.start_install(zip_path)
            else:
                QMessageBox.critical(self,
                    "Download Failed",
                    f"Failed to download update:\n\n{chr(10).join(self.update_manager.errors)}")
                self.reject()
        
        def start_install(self, zip_path: str):
            """Start installation in thread"""
            self.status_label.setText("Installing update...")
            self.progress.setValue(50)
            
            self.install_thread = QThread()
            self.install_worker = UpdateInstallWorker(self.update_manager, zip_path)
            self.install_worker.moveToThread(self.install_thread)
            
            self.install_worker.progress.connect(self.on_install_progress)
            self.install_worker.status.connect(self.on_install_status)
            self.install_worker.finished.connect(self.on_install_finished)
            
            self.install_thread.started.connect(self.install_worker.run)
            self.install_thread.start()
        
        def on_install_progress(self, progress: int):
            """Update install progress"""
            self.progress.setValue(progress)
        
        def on_install_status(self, status: str):
            """Update install status"""
            self.status_label.setText(status)
            self.details_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {status}")
        
        def on_install_finished(self, success: bool):
            """Handle installation completion"""
            self.install_thread.quit()
            self.install_thread.wait()
            
            if success:
                reply = QMessageBox.information(self,
                    "Update Complete",
                    f"Update to v{self.version_info['version']} installed successfully!\n\n"
                    "The application will restart now.",
                    QMessageBox.Ok)
                
                self.accept()
                
                # Restart app
                self.update_manager.restart_app()
            else:
                QMessageBox.critical(self,
                    "Installation Failed",
                    f"Failed to install update:\n\n{chr(10).join(self.update_manager.errors)}")
                self.reject()


# ============ Configuration Guide ============
"""
SETUP INSTRUCTIONS:

1. Create GitHub Repository:
   - Create repository: https://github.com/new
   - Clone locally
   - Push your code

2. Configure update_manager.py:
   - Update GITHUB_REPO = "your-username/my_trading_bot"
   - Set CURRENT_VERSION = current version number

3. Create Release on GitHub:
   - Go to: https://github.com/your-username/my_trading_bot/releases
   - Click "Create a new release"
   - Tag: v4.3.2 (format: vX.Y.Z)
   - Title: Release v4.3.2
   - Description: Your release notes
   - Attach: my_trading_bot_v4.3.2.zip (create this file)
   - Publish release

4. Create ZIP file:
   - Package all files in directory
   - Create: my_trading_bot_v4.3.2.zip
   - Upload to GitHub release

5. Test:
   - User clicks "Check for Updates"
   - Should detect new version
   - Download and install

6. Update CURRENT_VERSION in this file for next check
"""

if __name__ == "__main__":
    # Test update manager
    manager = UpdateManager()
    
    print("Checking for updates...")
    if manager.check_updates():
        print(f"Update available: {manager.latest_version}")
        print(f"Release notes: {manager.latest_info['notes']}")
    else:
        print("Already on latest version")
