#!/usr/bin/env python3
"""
Safe startup script for Trading Bot application
Handles potential startup errors and provides better error reporting
"""

import sys
import os
import subprocess
import traceback
from datetime import datetime

def safe_start():
    """Safely start the trading bot application"""
    try:
        print(f"🚀 Starting Trading Bot at {datetime.now()}")
        print("=" * 50)
        
        # Check if app.py exists
        if not os.path.exists("app.py"):
            print("❌ ERROR: app.py not found in current directory")
            return False
            
        # Set environment variables for better error handling
        os.environ['PYTHONUNBUFFERED'] = '1'
        os.environ['QT_LOGGING_RULES'] = '*.debug=false'
        
        # Import and run the main app
        print("📦 Importing application modules...")
        
        # Add current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # Import main app
        import app
        
        print("✅ Application modules loaded successfully")
        print("🖥️  Starting GUI...")
        
        # Run the application
        app.main()
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 This might be due to missing dependencies")
        traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        print("🔍 Full error details:")
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    required_modules = [
        'PyQt5',
        'pandas', 
        'numpy',
        'requests'
    ]
    
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} - OK")
        except ImportError:
            missing.append(module)
            print(f"❌ {module} - MISSING")
    
    return missing

if __name__ == "__main__":
    print("🔍 Checking dependencies...")
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("📦 Please install missing packages first")
        sys.exit(1)
    
    print("\n✅ All dependencies available")
    
    success = safe_start()
    
    if not success:
        print("\n❌ Application failed to start")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print("\n✅ Application started successfully")