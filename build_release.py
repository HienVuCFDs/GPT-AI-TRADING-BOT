#!/usr/bin/env python3
"""
Build ZIP for Release - Tự động tạo ZIP file cho GitHub Release

Usage:
    python build_release.py  (automatic version increment)
    python build_release.py --version 4.3.3  (specific version)
"""

import os
import sys
import zipfile
import argparse
import json
from pathlib import Path
from datetime import datetime


def get_current_version():
    """Đọc version từ update_manager.py"""
    try:
        with open('update_manager.py', 'r') as f:
            for line in f:
                if 'CURRENT_VERSION' in line:
                    version = line.split('"')[1]
                    return version
    except:
        return "4.3.2"


def increment_version(version):
    """Tăng minor version: 4.3.2 → 4.3.3"""
    parts = version.split('.')
    if len(parts) == 3:
        parts[2] = str(int(parts[2]) + 1)
    return '.'.join(parts)


def create_zip(version, output_dir='releases'):
    """Tạo ZIP file cho release"""
    
    # Tạo output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    zip_filename = f"my_trading_bot_v{version}.zip"
    zip_path = Path(output_dir) / zip_filename
    
    # Files/folders to exclude
    exclude_patterns = {
        '.git',
        '.github',
        '__pycache__',
        '.venv',
        'venv',
        '.env',
        '.env.local',
        'logs',
        'updates',
        'app_backup',
        '.backup',
        '*.pyc',
        '*.pyo',
        '.DS_Store',
        '.vscode',
        '.idea',
        'releases',
        'build',
        'dist',
        '*.egg-info',
        '.pytest_cache',
        '.coverage',
        'htmlcov',
        'node_modules',
        '.git',
        '.gitignore',
    }
    
    print(f"📦 Creating ZIP: {zip_filename}")
    print(f"💾 Output: {zip_path}")
    
    def is_excluded(path):
        """Check if path should be excluded"""
        relative = Path(path).relative_to(Path.cwd())
        for part in relative.parts:
            if part in exclude_patterns or part.startswith('.'):
                return True
        
        name = path.name if isinstance(path, Path) else Path(path).name
        if name in exclude_patterns:
            return True
        
        for pattern in exclude_patterns:
            if pattern.startswith('*'):
                ext = pattern.replace('*', '')
                if name.endswith(ext):
                    return True
        
        return False
    
    # Create ZIP
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        total_files = 0
        total_size = 0
        
        for root, dirs, files in os.walk('.'):
            # Filter directories
            dirs[:] = [d for d in dirs if not is_excluded(os.path.join(root, d))]
            
            # Add files
            for file in files:
                file_path = os.path.join(root, file)
                
                if is_excluded(file_path):
                    continue
                
                # Get relative path for archive
                arcname = os.path.relpath(file_path)
                
                # Add to ZIP
                zipf.write(file_path, arcname)
                
                # Stats
                file_size = os.path.getsize(file_path)
                total_files += 1
                total_size += file_size
                
                # Progress
                if total_files % 50 == 0:
                    print(f"  ... Added {total_files} files")
    
    # Print summary
    zip_size = zip_path.stat().st_size
    size_mb = zip_size / (1024 * 1024)
    
    print(f"\n✅ ZIP created successfully!")
    print(f"   Files: {total_files}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Path: {zip_path}")
    print(f"\n📤 Next steps:")
    print(f"   1. Go to: https://github.com/your-username/my_trading_bot/releases")
    print(f"   2. Click 'Create a new release'")
    print(f"   3. Tag: v{version}")
    print(f"   4. Upload: {zip_filename}")
    print(f"   5. Publish release")
    
    return str(zip_path)


def update_version_in_file(new_version):
    """Update CURRENT_VERSION in update_manager.py"""
    try:
        with open('update_manager.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace version
        old_version = get_current_version()
        content = content.replace(
            f'CURRENT_VERSION = "{old_version}"',
            f'CURRENT_VERSION = "{new_version}"'
        )
        
        with open('update_manager.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Updated CURRENT_VERSION: {old_version} → {new_version}")
        return True
    except Exception as e:
        print(f"❌ Failed to update version: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Build ZIP file for GitHub Release'
    )
    parser.add_argument(
        '--version',
        type=str,
        help='Specific version (e.g., 4.3.3)',
        default=None
    )
    parser.add_argument(
        '--auto-increment',
        action='store_true',
        help='Automatically increment version',
        default=True
    )
    parser.add_argument(
        '--update-file',
        action='store_true',
        help='Update CURRENT_VERSION in update_manager.py',
        default=True
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for ZIP',
        default='releases'
    )
    
    args = parser.parse_args()
    
    # Determine version
    current_version = get_current_version()
    
    if args.version:
        new_version = args.version
    else:
        new_version = increment_version(current_version)
    
    print(f"🔄 Building release")
    print(f"   Current: v{current_version}")
    print(f"   New: v{new_version}\n")
    
    # Update version in file
    if args.update_file:
        update_version_in_file(new_version)
    
    # Create ZIP
    zip_path = create_zip(new_version, args.output_dir)
    
    print(f"\n✨ Release build complete!")
    print(f"   File: {zip_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
