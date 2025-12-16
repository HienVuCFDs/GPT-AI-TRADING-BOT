"""
License Server - Production Runner
Sử dụng Waitress WSGI server cho Windows

Chạy: python run_server.py
Hoặc: pythonw run_server.py (chạy nền, không cần terminal)
"""
import os
import sys

# Thêm project vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'license_server.settings')

def main():
    from waitress import serve
    from license_server.wsgi import application
    
    HOST = '0.0.0.0'  # Cho phép truy cập từ mạng LAN
    PORT = 8000
    
    print("=" * 50)
    print("🚀 LICENSE SERVER - PRODUCTION MODE")
    print("=" * 50)
    print(f"✅ Server running on http://{HOST}:{PORT}")
    print(f"✅ Local access: http://127.0.0.1:{PORT}")
    print(f"✅ Dashboard: http://127.0.0.1:{PORT}/dashboard/")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print()
    
    # threads: Số worker xử lý đồng thời
    # - 4 threads: ~500 users (VPS 1GB RAM)
    # - 8 threads: ~1000 users (VPS 2GB RAM)
    # - 16 threads: ~2000 users (VPS 4GB RAM)
    THREADS = 8
    
    serve(application, host=HOST, port=PORT, threads=THREADS)

if __name__ == '__main__':
    main()
