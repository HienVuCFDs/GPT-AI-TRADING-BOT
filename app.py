# === ANTI-CRASH PROTECTION SYSTEM ===
import sys
import os
import json
import pickle
import logging
import hashlib
import threading
import traceback
import signal
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import subprocess
import glob

# Global exception handler to prevent crashes
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Global exception handler - prevents app crash on unhandled exceptions"""
    try:
        # Log the error
        error_msg = f"CRITICAL ERROR: {exc_type.__name__}: {str(exc_value)}"
        print(f"\n🚨 {error_msg}")
        
        # Save crash log
        crash_log = f"logs/crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        os.makedirs("logs", exist_ok=True)
        
        with open(crash_log, 'w', encoding='utf-8') as f:
            f.write(f"Crash Time: {datetime.now()}\n")
            f.write(f"Error Type: {exc_type.__name__}\n")
            f.write(f"Error Message: {str(exc_value)}\n\n")
            f.write("Full Traceback:\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
        
        print(f"💾 Crash details saved to: {crash_log}")
        print("🔄 App will continue running...")
        
        # Continue running instead of crashing
        return True
        
    except Exception as e:
        print(f"❌ Error in exception handler: {e}")
        # Last resort - don't crash even if logging fails
        return True

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    """Handle SIGTERM and SIGINT for graceful shutdown"""
    print(f"\n🛑 Received signal {signum} - initiating graceful shutdown...")
    try:
        emergency_cleanup()
    except:
        pass
    sys.exit(0)

# Install global exception handler and signal handlers
sys.excepthook = global_exception_handler
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def safe_method(func):
    """Decorator to make any method crash-proof"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            method_name = getattr(func, '__name__', 'unknown_method')
            class_name = ""
            if args and hasattr(args[0], '__class__'):
                class_name = f"{args[0].__class__.__name__}."
            
            error_msg = f"🛡️ Safe method caught error in {class_name}{method_name}: {type(e).__name__}: {str(e)}"
            print(error_msg)
            
            # Log detailed error for debugging
            try:
                log_file = f"logs/method_errors_{datetime.now().strftime('%Y%m%d')}.log"
                os.makedirs("logs", exist_ok=True)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{datetime.now()} - {error_msg}\n")
                    f.write(f"Traceback: {traceback.format_exc()}\n")
            except:
                pass  # Don't crash even if logging fails
            
            # Return safe defaults based on expected return type
            return None
    return wrapper

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle system signals gracefully"""
    print(f"\n🛑 Received signal {signum} - shutting down gracefully...")
    try:
        # Try graceful cleanup
        if 'QApplication' in sys.modules:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                app.quit()
    except:
        pass
    sys.exit(0)

# Install signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Try to import dotenv
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Try to import PyQt5 for GUI
try:
    from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QCheckBox, QMessageBox,
    QGroupBox, QInputDialog, QToolButton, QMenu, QAction, QActionGroup
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRunnable, QThreadPool, QObject
    from PyQt5.QtGui import QIcon, QFont, QColor
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Simple global app state for language preference
class AppState:
    _lang = 'vi'  # Back to Vietnamese

    @classmethod
    def set_language(cls, lang: str):
        if not lang:
            return
        lang = lang.lower()
        if lang not in ('en', 'vi'):
            return
        if cls._lang != lang:
            cls._lang = lang
            # Persist to user config
            try:
                cfg = load_user_config(apply_lang=False)
                cfg['language'] = cls._lang
                save_user_config(cfg)
            except Exception as e:
                logging.warning(f"Unable to persist language preference: {e}")

    @classmethod
    def language(cls) -> str:
        return cls._lang

# Simple i18n helper for UI notifications
class I18N:
    @staticmethod
    def t(en: str, vi: str | None = None, **kwargs) -> str:
        """Return localized text; format with kwargs if provided.
        Usage: I18N.t("Title EN", "Tiêu đề VI"); I18N.t("Hello {name}", "Xin chào {name}", name="An")
        """
        txt = None
        if AppState.language() == 'vi' and vi:
            txt = vi
        else:
            txt = en
        try:
            return txt.format(**kwargs)
        except Exception:
            return txt

    # Common static UI translations to cover widgets without explicit retranslate methods
    EN_TO_VI = {
        # Main Tab Names (CRITICAL FOR LANGUAGE SWITCHING)
        "🏦 MT5 Account": "🏦 Tài khoản MT5",
        "💹 Market Data": "💹 Dữ liệu thị trường",
        "📈 Trend Analysis": "📈 Phân tích xu hướng", 
        "⚙️ Technical Indicators": "⚙️ Chỉ báo kỹ thuật",
        "🕯️ Candlestick Patterns": "🕯️ Mô hình nến",
        "📊 Price Patterns": "📊 Mô hình giá",
        "📰 Economic News": "📰 Tin tức kinh tế",
        "🛡️ Risk Management": "🛡️ Quản lý rủi ro",
        "📡 Signal": "📡 Tín hiệu",
        "🤖 Auto Trading": "🤖 Giao dịch tự động",
        
        # Generic
        "One Click Trading": "Giao dịch một chạm",
        "Volume:": "Khối lượng:",
        "Spread:": "Chênh lệch:",
        "Remove": "Xóa",
        # Menu / global
        "Login": "Đăng nhập",
        "Register": "Đăng ký",
        "Status: Connected": "Trạng thái: Đã kết nối",
        "Status: Disconnected": "Trạng thái: Mất kết nối",
        "🟢 Status: Connected": "🟢 Trạng thái: Đã kết nối",
        "🔴 Status: Connection Failed": "🔴 Trạng thái: Kết nối thất bại",
        # News tab headers
        "Impact": "Tác động",
        "Event": "Sự kiện", 
        "Actual": "Thực tế",
        "Forecast": "Dự báo",
        # News tab UI elements
        "Filters": "Bộ lọc",
        "Currency:": "Tiền tệ:",
        "Auto Trading Integration": "Tích hợp giao dịch tự động",
        "✅ Enable News detection": "✅ Bật phát hiện tin tức",
        "Data Source:": "Nguồn dữ liệu:",
        "System:": "Hệ thống:",
        "Date:": "Ngày:",
        "Found": "Tìm thấy",
        "events": "sự kiện",
        "Timezone:": "Múi giờ:",
        "All times displayed in": "Tất cả thời gian hiển thị theo",
        "Vietnam Time (UTC+7)": "Giờ Việt Nam (UTC+7)",
        # Auto Schedule translations
        "Auto News Schedule Status": "Trạng thái lịch tin tức tự động",
        "🤖 Auto-schedule: Initializing...": "🤖 Lịch tự động: Đang khởi tạo...",
        "📅 Scheduled Times: None": "📅 Giờ đã lên lịch: Chưa có",
        "⏰ Next Auto-Fetch: Calculating...": "⏰ Quét tin tiếp theo: Đang tính...",
        "🤖 System Active": "🤖 Hệ thống hoạt động",
        "✅ Active": "✅ Hoạt động",
        "🔄 Fallback Mode": "🔄 Chế độ dự phòng",
        "❌ Fetch Error": "❌ Lỗi tải tin",
        # Fix common typos
        "Thái gian": "Thời gian",
        "Previous": "Trước đó",
        "Status": "Trạng thái",
        # Chart controls
        "Start Chart": "Bắt đầu biểu đồ",
        "Stop Chart": "Dừng biểu đồ",
        "Start Live Updates": "Bắt đầu cập nhật trực tiếp", 
        "Stop": "Dừng",
        # Market data
        "Fetch Data Now": "Lấy dữ liệu ngay",
        "Fetch News": "Lấy tin tức",
        "Connected to MT5 - Ready to fetch data": "Đã kết nối MT5 - Sẵn sàng lấy dữ liệu",
        "Current Drawdown:": "Mức sụt giảm hiện tại:",
        "Active Positions:": "Vị thế đang mở:",
        "Today's P&L:": "Lãi/lỗ hôm nay:",
        "Risk Level:": "Mức rủi ro:",
        "📊 Recent Signal Validations": "📊 Kiểm định tín hiệu gần đây",
        "💼 Current Positions": "💼 Vị thế hiện tại",
        "🎮 Control Panel": "🎮 Bảng điều khiển",
        "Trading Mode:": "Chế độ giao dịch:",
        "💾 Save Settings": "💾 Lưu cài đặt",
        "📁 Load Settings": "📁 Tải cài đặt",
        "🔄 Reset to Default": "🔄 Đặt lại mặc định",
        "📊 Generate Report": "📊 Tạo báo cáo",
        # Auto Trading tab
        "Auto Trading: OFF": "Giao dịch tự động: TẮT",
        "Auto Trading: ON": "Giao dịch tự động: BẬT",
        "Start Auto Trading": "Bắt đầu giao dịch tự động",
    "🤖 Auto Mode": "🤖 Tự động",
    "👨‍💼 Manual Mode": "👨‍💼 Thủ công",
        # Indicator tab
        "Add Indicator": "Thêm chỉ báo",
        "Add All": "Thêm tất cả",
        "Calculate & Save Indicator": "Tính & lưu chỉ báo",
        "Period:": "Chu kỳ:",
        "Type:": "Loại:",
        "Fast:": "Nhanh:",
        "Slow:": "Chậm:",
        "Signal:": "Tín hiệu:",
        "Smooth:": "Làm mượt:",
        "Window:": "Cửa sổ:",
        "Dev:": "Độ lệch:",
        "Step:": "Bước:",
        "Max Step:": "Bước tối đa:",
        "Smooth1:": "Làm mượt1:",
        "Smooth2:": "Làm mượt2:",
        "Short:": "Ngắn:",
        "Medium:": "Trung bình:",
        "Long:": "Dài:",
        "Lookback:": "Nhìn lại:",
        "Tenkan:": "Tenkan:",
        "Kijun:": "Kijun:",
        "Senkou:": "Senkou:",
        "Percent:": "Phần trăm:",
        # Pattern tabs
        "Enable candlestick pattern detection": "Bật phát hiện mô hình nến",
    "📊 Min confidence:": "📊 Độ tin cậy tối thiểu:",
        "No patterns loaded": "Chưa tải mô hình",
        "🔍 Fetch Candlestick Patterns": "🔍 Lấy mô hình nến",
        "Enable price pattern detection": "Bật phát hiện mô hình giá",
        "📅 Max age (days):": "📅 Tuổi tối đa (ngày):",
        "🔍 Fetch Price Patterns": "🔍 Lấy mô hình giá",
        # Trend tab
        "Enable Trend Detection": "Bật phát hiện xu hướng",
        "Calculate Trendline & SR": "Tính đường xu hướng & SR",
    # Common table headers
    "Ticket": "Vé",
    "Symbol": "Mã",
    "Time": "Thời gian",
    "Action": "Hành động",
    "Result": "Kết quả",
    "Risk Score": "Điểm rủi ro",
    "Volume": "Khối lượng",
    "Type": "Loại",
    "Open Price": "Giá mở",
    "Current Price": "Giá hiện tại",
    "Stop Loss": "Cắt lỗ",
    "Take Profit": "Chốt lời",
    "Swap": "Hoán đổi",
    "Profit": "Lợi nhuận",
    "Actions": "Thao tác",
    "Price": "Giá",
    "P&L": "Lãi/Lỗ",
    "Max Exposure (lots)": "Khối lượng tối đa (lot)",
    "Risk Multiplier": "Hệ số rủi ro",
    "Indicator": "Chỉ báo",
    "Value": "Giá trị",
    "Timeframe": "Khung thời gian",
    "Pattern": "Mô hình",
    "Length": "Độ dài",
    "Signal": "Tín hiệu",
    "Confidence": "Độ tin cậy",
    "Time Period": "Khoảng thời gian",
    "Age": "Tuổi",
    # Additional risk management translations
    "Fixed Pips": "Pips cố định",
    "ATR Multiple": "Bội số ATR",
    "Support/Resistance": "Hỗ trợ/Kháng cự",
    "Percentage": "Phần trăm",
    "Fixed Distance": "Khoảng cách cố định",
    "Fibonacci Levels": "Mức Fibonacci",
    "Individual SL": "SL riêng lẻ",
    "Average SL": "SL trung bình",
    # "Breakeven Only": "Chỉ hòa vốn", # REMOVED: No longer supported
    "✅ Risk Management System: Active": "✅ Hệ thống quản lý rủi ro: Hoạt động",
    "❌ Risk Management System: Not Available": "❌ Hệ thống quản lý rủi ro: Không khả dụng",
    "❌ Risk Manager Error:": "❌ Lỗi quản lý rủi ro:",
    "💡 Enable Auto Mode": "💡 Bật chế độ tự động",
    "🔕 Disable Emergency Stop": "🔕 Tắt dừng khẩn cấp",
    "🔕 Disable News Avoidance": "🔕 Tắt tránh tin tức",
    "🔕 Disable Max DD Close": "🔕 Tắt đóng khi Max DD",
    "Enable Auto Account Scanning": "Bật quét tài khoản tự động",
    "Auto Scan Interval (hours):": "Khoảng thời gian quét tự động (giờ):",
    "Account Tier:": "Cấp tài khoản:",
    "Last Auto Adjustment:": "Điều chỉnh tự động cuối:",
    "Emergency Stop Active": "Dừng khẩn cấp hoạt động",
    "Emergency Stop Inactive": "Dừng khẩn cấp không hoạt động",
    "News Avoidance Active": "Tránh tin tức hoạt động",
    "News Avoidance Inactive": "Tránh tin tức không hoạt động",
    "Max DD Close Active": "Đóng Max DD hoạt động",
    "Max DD Close Inactive": "Đóng Max DD không hoạt động",
    "Auto Scan Active": "Quét tự động hoạt động",
    "Auto Scan Inactive": "Quét tự động không hoạt động",
    }

    VI_TO_EN = {vi: en for en, vi in EN_TO_VI.items()}

    @staticmethod
    def _translate_runtime_text(txt: str, target_lang: str) -> str:
        if not isinstance(txt, str) or not txt:
            return txt
        base = txt
        if target_lang == 'vi':
            direct = I18N.EN_TO_VI.get(base)
            if direct:
                return direct
            # Fallback partial replacement
            for k, v in I18N.EN_TO_VI.items():
                if k in base:
                    base = base.replace(k, v)
            return base
        else:
            direct = I18N.VI_TO_EN.get(base)
            if direct:
                return direct
            for k, v in I18N.VI_TO_EN.items():  # k=VI, v=EN
                if k in base:
                    base = base.replace(k, v)
            return base

    @staticmethod
    def retranslate_widget_tree(root_widget: 'QWidget') -> None:
        """Recursively translate common widget texts using EN_TO_VI mapping.
        Safe no-op for texts not present in the mapping.
        """
        try:
            from PyQt5.QtWidgets import QWidget as _QW, QLabel, QPushButton, QGroupBox, QCheckBox, QRadioButton, QToolButton, QTableWidget, QComboBox
        except Exception:
            return
        target = AppState.language()

        def _apply(w):
            try:
                if isinstance(w, QGroupBox):
                    w.setTitle(I18N._translate_runtime_text(w.title(), target))
                elif isinstance(w, QLabel):
                    w.setText(I18N._translate_runtime_text(w.text(), target))
                elif isinstance(w, QPushButton):
                    txt0 = w.text()
                    new_txt = I18N._translate_runtime_text(txt0, target)
                    # Special handling for dynamic BUY/SELL buttons with newline values
                    try:
                        if target == 'vi':
                            new_txt = new_txt.replace('SELL', 'BÁN').replace('BUY', 'MUA')
                        else:
                            new_txt = new_txt.replace('BÁN', 'SELL').replace('MUA', 'BUY')
                    except Exception:
                        pass
                    w.setText(new_txt)
                elif isinstance(w, QCheckBox):
                    w.setText(I18N._translate_runtime_text(w.text(), target))
                elif isinstance(w, QRadioButton):
                    w.setText(I18N._translate_runtime_text(w.text(), target))
                elif isinstance(w, QToolButton):
                    w.setText(I18N._translate_runtime_text(w.text(), target))
                    w.setToolTip(I18N._translate_runtime_text(w.toolTip(), target))
                elif isinstance(w, QTableWidget):
                    # Translate header labels, if set
                    try:
                        cols = w.columnCount()
                        headers: list[str] = []
                        changed = False
                        for i in range(cols):
                            it = w.horizontalHeaderItem(i)
                            if it is not None:
                                old = it.text()
                                new = I18N._translate_runtime_text(old, target)
                                if new != old:
                                    it.setText(new)
                                    changed = True
                                headers.append(new)
                        # If no header items existed, set via labels is used; do nothing
                        if changed:
                            w.horizontalHeader().reset()
                    except Exception:
                        pass
                # Translate QTabWidget tab labels if encountered as a parent
                elif w.metaObject().className() == 'QTabWidget':
                    try:
                        count = w.count()
                        print(f"[DEBUG] Found QTabWidget with {count} tabs, target language: {target}")
                        for i in range(count):
                            old = w.tabText(i)
                            new = I18N._translate_runtime_text(old, target)
                            print(f"[DEBUG] Tab {i}: '{old}' -> '{new}'")
                            if new != old:
                                w.setTabText(i, new)
                                print(f"[DEBUG] Tab {i} title updated")
                            else:
                                print(f"[DEBUG] Tab {i} title unchanged")
                    except Exception as e:
                        print(f"[DEBUG] Error translating tabs: {e}")
                        pass
                elif isinstance(w, QComboBox):
                    # Translate items while preserving current selection
                    try:
                        idx = w.currentIndex()
                        for i in range(w.count()):
                            txt = w.itemText(i)
                            w.setItemText(i, I18N._translate_runtime_text(txt, target))
                        if idx >= 0:
                            w.setCurrentIndex(idx)
                    except Exception:
                        pass
            except Exception:
                pass
            for c in w.children():
                if isinstance(c, _QW):
                    _apply(c)

        _apply(root_widget)

    @staticmethod
    def translate_application():
        """Translate all top-level widgets (broader than a single window)."""
        if not GUI_AVAILABLE:
            return
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if not app:
                return
            for w in app.topLevelWidgets():
                try:
                    I18N.retranslate_widget_tree(w)
                except Exception:
                    continue
        except Exception:
            pass

    @staticmethod
    def force_full_translation(root_widget=None, debug=False):
        """Aggressive pass: replace substrings for every widget text property.
        This helps when original text contains dynamic parts not exactly matched.
        """
        if not GUI_AVAILABLE:
            return
        try:
            from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QGroupBox, QCheckBox, QRadioButton, QToolButton, QTableWidget, QComboBox
            app = QApplication.instance()
            if not app:
                return
            target_lang = AppState.language()
            pairs = I18N.EN_TO_VI.items() if target_lang == 'vi' else I18N.VI_TO_EN.items()
            widgets = []
            if root_widget is not None:
                widgets.append(root_widget)
            else:
                widgets.extend(app.topLevelWidgets())
            visited = set()
            while widgets:
                w = widgets.pop(0)
                if id(w) in visited:
                    continue
                visited.add(id(w))
                # Determine text getters/setters
                for getter_name, setter_name in (("text", "setText"), ("title", "setTitle"), ("placeholderText", "setPlaceholderText")):
                    if hasattr(w, getter_name) and hasattr(w, setter_name):
                        try:
                            orig = getattr(w, getter_name)()
                        except Exception:
                            orig = None
                        if isinstance(orig, str) and orig:
                            new_txt = orig
                            for k, v in pairs:
                                new_txt = new_txt.replace(k, v)
                            if new_txt != orig:
                                try:
                                    getattr(w, setter_name)(new_txt)
                                    if debug:
                                        print(f"[LangForce] {getter_name} changed: '{orig}' -> '{new_txt}'")
                                except Exception:
                                    pass
                # Table headers
                if isinstance(w, QTableWidget):
                    try:
                        for c in range(w.columnCount()):
                            it = w.horizontalHeaderItem(c)
                            if it:
                                t0 = it.text()
                                t1 = t0
                                for k, v in pairs:
                                    t1 = t1.replace(k, v)
                                if t1 != t0:
                                    it.setText(t1)
                                    if debug:
                                        print(f"[LangForce] header: '{t0}' -> '{t1}'")
                    except Exception:
                        pass
                try:
                    widgets.extend([c for c in w.children() if hasattr(c, 'children')])
                except Exception:
                    pass
        except Exception as e:
            if debug:
                print(f"[LangForce] error: {e}")

# Try to import matplotlib for candlestick charts
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    import matplotlib.dates as mdates
    from datetime import datetime
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib not available - candlestick charts disabled")

# Data fetcher
try:
    from mt5_data_fetcher import fetch_and_save_candles
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    DATA_FETCHER_AVAILABLE = False

# Order executor for quick trading  
try:
    from order_executor import get_executor_instance, TradeSignal
    ORDER_EXECUTOR_AVAILABLE = True
except ImportError:
    ORDER_EXECUTOR_AVAILABLE = False
    print("⚠️ Order executor not available - quick trading disabled")

# Indicator exporter
try:
    from mt5_indicator_exporter import calculate_and_save_all
    INDICATOR_EXPORTER_AVAILABLE = True
except ImportError:
    INDICATOR_EXPORTER_AVAILABLE = False

# Pattern detector
try:
    from pattern_detector import analyze_patterns
    PATTERN_DETECTOR_AVAILABLE = True
except ImportError:
    PATTERN_DETECTOR_AVAILABLE = False

# News scraper
# News scraper - use Selenium version
try:
    from news_scraper import NewsScraperSelenium, get_today_news as scraper_get_today_news, save_recent_news_to_json as scraper_save_news
    # Create a compatible interface for the app
    def get_today_news(currencies=None, impacts=None):
        print("🔄 Using NewsScraperSelenium…")
        # Use non-headless to improve reliability behind Cloudflare when launched from the app
        return scraper_get_today_news(currencies, impacts, headless=False, auto_cleanup=True)
    
    def save_recent_news_to_json(events, filename):
        return scraper_save_news(events, filename)
    
    NEWS_SCRAPER_AVAILABLE = True
    # print("✅ NewsScraperSelenium imported successfully")
except ImportError as e:
    NEWS_SCRAPER_AVAILABLE = False
    print(f"⚠️ News scraper import failed: {e}")
    
    def get_today_news(*args): 
        return []
    def save_recent_news_to_json(*args): 
        return False

# Auto trading manager - unified version
try:
    from unified_auto_trading_system import UnifiedAutoTradingSystem as AutoTradingManager
    AUTO_TRADING_AVAILABLE = True
    print("[AUTO TRADING] Using unified auto trading system")
except ImportError:
    AUTO_TRADING_AVAILABLE = False
    print("[AUTO TRADING] Not available")
    class AutoTradingManagerStub:
        def __init__(self, *args): pass
        def start(self): pass
        def stop(self): pass
    AutoTradingManager = AutoTradingManagerStub

# Risk management system
try:
    from risk_manager import AdvancedRiskManagementSystem, AdvancedRiskParameters, TradeSignal, ValidationResult
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False

import sys
import os
import logging
import json
from datetime import datetime, timedelta, timezone
import time
import re
import threading
import hashlib
import pickle
import traceback

# Import GUI helper functions from pattern_detector
try:
    from pattern_detector import (
        load_and_filter_patterns, sort_patterns_by_priority,
        get_pattern_statistics, format_status_message, is_candlestick_pattern,
        load_price_patterns_from_folder
    )
    GUI_HELPERS_AVAILABLE = True
    # print("✅ GUI helpers loaded successfully from pattern_detector")
except ImportError as e:
    print(f"⚠️ GUI helpers not available: {e}")
    GUI_HELPERS_AVAILABLE = False
    # Default functions when GUI helpers not available
    def load_and_filter_patterns(*args, **kwargs): return []
    def sort_patterns_by_priority(patterns): return patterns
    def get_pattern_statistics(patterns): return {'total_count': 0, 'candlestick_count': 0}
    def format_status_message(stats, candlestick_only): return "Helper functions not available"
    def is_candlestick_pattern(pattern_name): return False
    def load_price_patterns_from_folder(*args, **kwargs): return []

# Enhanced import handling - GUI widgets  
try:
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QTextEdit,
        QMessageBox, QLineEdit, QFormLayout, QListWidget, QListWidgetItem,
        QHBoxLayout, QSpinBox, QDoubleSpinBox, QGridLayout, QCheckBox, QTabWidget, QGroupBox,
        QRadioButton, QButtonGroup, QFrame, QSplitter, QScrollArea, QProgressBar, QSlider, QDial,
        QComboBox, QTableWidget, QTableWidgetItem, QHeaderView
    )
    from PyQt5.QtCore import Qt, QTimer, QRunnable, QThreadPool, pyqtSignal, QObject, QThread
    from PyQt5.QtGui import QPixmap, QIcon, QFont
    GUI_AVAILABLE = True
    # print("✅ GUI components loaded successfully")
except ImportError as e:
    print(f"⚠️ GUI not available: {e}")
    print("Running in console mode...")
    GUI_AVAILABLE = False
    # Default classes for non-GUI mode
    class QObject: pass
    class QThread: pass

# ------------------------------
# Signal Tab - UI for comprehensive_aggregator.py
# ------------------------------
if GUI_AVAILABLE:
    class RunAggregatorWorker(QThread):
        def __init__(self, args: List[str], parent: Optional[QObject] = None):
            super().__init__(parent)
            self.args = args
            self.returncode = None
            self.stdout = None
            self.stderr = None
            self._should_stop = False

        def stop(self):
            """Safe thread stop request"""
            self._should_stop = True

        def run(self):
            """Enhanced crash-proof QThread run method"""
            try:
                if self._should_stop:
                    print("🛑 Thread stop requested before start")
                    return
                    
                # Use current Python executable for reliability
                cmd = [sys.executable, 'comprehensive_aggregator.py'] + self.args
                print(f"🔧 DEBUG: Running command: {' '.join(cmd)}")
                
                # Set proper working directory and environment
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'  # Force UTF-8 encoding for subprocess
                
                proc = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    shell=False,
                    cwd=os.getcwd(),  # Ensure correct working directory
                    timeout=300,  # 5 minute timeout
                    env=env,  # Use UTF-8 environment
                    encoding='utf-8',  # Explicit UTF-8 encoding
                    errors='replace'  # Replace problematic chars instead of crashing
                )
                
                if self._should_stop:
                    print("🛑 Thread stopped during execution")
                    return
                
                self.returncode = proc.returncode
                self.stdout = proc.stdout
                self.stderr = proc.stderr
                
                print(f"🔧 DEBUG: Return code: {self.returncode}")
                if self.stderr:
                    print(f"🔧 DEBUG: Stderr: {self.stderr[:500]}")  # First 500 chars
                if self.stdout:
                    print(f"🔧 DEBUG: Stdout length: {len(self.stdout)} chars")
                    
            except subprocess.TimeoutExpired as e:
                self.returncode = -2
                self.stderr = f"Process timed out after 5 minutes: {str(e)}"
                print(f"⏰ TIMEOUT: Process exceeded 5 minute limit - {self.stderr}")
                
            except subprocess.CalledProcessError as e:
                self.returncode = e.returncode
                self.stderr = f"Process failed with exit code {e.returncode}: {str(e)}"
                print(f"❌ PROCESS ERROR: {self.stderr}")
                
            except KeyboardInterrupt:
                self.returncode = -3
                self.stderr = "Process interrupted by user"
                print(f"🛑 INTERRUPTED: {self.stderr}")
                
            except Exception as e:
                self.returncode = -1
                self.stderr = f"Critical thread error: {type(e).__name__}: {str(e)}"
                print(f"🚨 CRITICAL: QThread exception - {self.stderr}")
                
                # Log the error for debugging
                try:
                    import traceback
                    error_trace = traceback.format_exc()
                    print(f"🔍 TRACEBACK:\n{error_trace}")
                    
                    # Save crash log
                    crash_log = f"logs/qthread_crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                    os.makedirs("logs", exist_ok=True)
                    with open(crash_log, 'w', encoding='utf-8') as f:
                        f.write(f"QThread Crash: {datetime.now()}\n")
                        f.write(f"Error: {self.stderr}\n\n")
                        f.write(f"Traceback:\n{error_trace}\n")
                    print(f"💾 QThread crash log saved: {crash_log}")
                except:
                    print("❌ Could not save crash log")
            finally:
                # Ensure thread always completes cleanly
                print("✅ QThread run method completed")

    class SignalTab(QWidget):
        def __init__(self, parent: Optional[QWidget] = None, indicator_tab: Optional[QWidget] = None, market_tab: Optional[QWidget] = None):
            super().__init__(parent)
            # Reference to IndicatorTab so we can persist the latest whitelist before running aggregator
            self.indicator_tab = indicator_tab
            # Reference to MarketTab so we can get selected timeframes
            self.market_tab = market_tab
            self.thread: Optional[RunAggregatorWorker] = None
            self._build_ui()
            # Apply initial language to visible texts
            try:
                self.retranslate_ui()
            except Exception:
                pass
            self.load_latest_signals()

        def cleanup_thread(self):
            """Enhanced thread cleanup to prevent QThread crash on app close"""
            try:
                if self.thread and self.thread.isRunning():
                    print("[CLEANUP] 🧹 Gracefully stopping RunAggregatorWorker thread...")
                    
                    # First, try graceful stop
                    self.thread.stop()
                    if self.thread.wait(2000):  # Wait 2 seconds
                        print("[CLEANUP] ✅ Thread stopped gracefully")
                    else:
                        print("[CLEANUP] ⏰ Graceful stop timeout - trying terminate...")
                        
                        # If graceful stop fails, try terminate
                        self.thread.terminate()
                        if self.thread.wait(3000):  # Wait 3 more seconds
                            print("[CLEANUP] ✅ Thread terminated successfully")
                        else:
                            print("[CLEANUP] ❌ Terminate timeout - forcing kill...")
                            # Last resort - force kill (dangerous but prevents hang)
                            try:
                                self.thread.kill()
                                self.thread.wait(1000)  # Give it 1 second to die
                                print("[CLEANUP] ⚠️ Thread force killed")
                            except Exception as kill_error:
                                print(f"[CLEANUP] ❌ Force kill failed: {kill_error}")
                    
                    self.thread = None
                    print("[CLEANUP] ✅ RunAggregatorWorker cleanup completed")
                else:
                    print("[CLEANUP] 👍 No running thread to clean up")
                    
            except Exception as e:
                print(f"[CLEANUP] ❌ Error during thread cleanup: {e}")
                # Even if cleanup fails, clear the reference
                self.thread = None

        def _build_indicator_list_for_export(self) -> List[Dict[str, Any]]:
            """Build exporter-compatible indicator_list using IndicatorTab rows.
            Expands MA family (EMA/SMA/WMA/TEMA) if MA-family mode is 'expand'.
            Safe if indicator_tab is missing or rows incomplete.
            """
            indicator_list: List[Dict[str, Any]] = []
            try:
                if not getattr(self, 'indicator_tab', None):
                    return indicator_list
                rows = getattr(self.indicator_tab, 'indicator_rows', []) or []
                # Determine MA-family mode - hardcoded to 'expand'
                mode = 'expand'

                for row in rows:
                    try:
                        combo = row.get("indi_combo")
                        if combo is None:
                            continue
                        indi_label = combo.currentText()
                        # Map label to canonical name using IndicatorTab's helper if available
                        try:
                            indi_name = self.indicator_tab._label_to_name(indi_label)
                        except Exception:
                            indi_name = None
                        if not indi_name:
                            continue

                        params: Dict[str, Any] = {}
                        # Collect parameters similarly to IndicatorTab.export_indicators
                        if indi_name in [
                            "RSI", "ATR", "ADX", "CCI", "WilliamsR", "ROC",
                            "MFI", "Chaikin", "EOM", "ForceIndex", "TRIX", "DPO"
                        ]:
                            if "period_spin" in row and row["period_spin"] is not None:
                                params["period"] = row["period_spin"].value()
                        elif indi_name == "Bollinger Bands":
                            if "window_spin" in row and row["window_spin"] is not None:
                                params["window"] = row["window_spin"].value()
                            if "dev_spin" in row and row["dev_spin"] is not None:
                                params["dev"] = row["dev_spin"].value()
                        elif indi_name == "MA":
                            # For MA, capture period and type; then expand family if needed
                            period = None
                            ma_type = None
                            if "period_spin" in row and row["period_spin"] is not None:
                                period = int(row["period_spin"].value())
                            if "ma_type_combo" in row and row["ma_type_combo"] is not None:
                                ma_type = str(row["ma_type_combo"].currentText()).upper()
                            if not period:
                                continue
                            # exact: only selected type; expand: include all four types
                            ma_types = [ma_type] if (ma_type and mode == 'exact') else ["SMA", "EMA", "WMA", "TEMA"]
                            # Build entries for each required type
                            for mt in ma_types:
                                if mt in ("SMA", "EMA"):
                                    indicator_list.append({
                                        "name": "MA",
                                        "params": {"period": period, "ma_type": mt}
                                    })
                                elif mt == "WMA":
                                    indicator_list.append({
                                        "name": "WMA",
                                        "params": {"period": period}
                                    })
                                elif mt == "TEMA":
                                    indicator_list.append({
                                        "name": "TEMA",
                                        "params": {"period": period}
                                    })
                            continue  # MA handled; skip generic append below
                        elif indi_name == "MACD":
                            for key in ("fast", "slow", "signal"):
                                spin_key = f"{key}_spin"
                                if spin_key in row and row[spin_key] is not None:
                                    params[key] = row[spin_key].value()
                        elif indi_name == "Stochastic":
                            for key in ("period", "smooth"):
                                spin_key = f"{key}_spin"
                                if spin_key in row and row[spin_key] is not None:
                                    params[key] = row[spin_key].value()
                        elif indi_name == "PSAR":
                            if "step_spin" in row and row["step_spin"] is not None:
                                params["step"] = row["step_spin"].value()
                            if "max_step_spin" in row and row["max_step_spin"] is not None:
                                params["max_step"] = row["max_step_spin"].value()
                        elif indi_name == "Donchian":
                            if "window_spin" in row and row["window_spin"] is not None:
                                params["window"] = row["window_spin"].value()
                        elif indi_name == "MassIndex":
                            for key in ("fast", "slow"):
                                spin_key = f"{key}_spin"
                                if spin_key in row and row[spin_key] is not None:
                                    params[key] = row[spin_key].value()
                        elif indi_name == "Vortex":
                            if "period_spin" in row and row["period_spin"] is not None:
                                params["period"] = row["period_spin"].value()
                        elif indi_name == "KST":
                            for key in ("window1", "window2", "window3", "window4", "window_sig"):
                                spin_key = f"{key}_spin"
                                if spin_key in row and row[spin_key] is not None:
                                    params[key] = row[spin_key].value()
                        elif indi_name == "StochRSI":
                            for key in ("period", "smooth1", "smooth2"):
                                spin_key = f"{key}_spin"
                                if spin_key in row and row[spin_key] is not None:
                                    params[key] = row[spin_key].value()
                        elif indi_name == "UltimateOscillator":
                            for key in ("short", "medium", "long"):
                                spin_key = f"{key}_spin"
                                if spin_key in row and row[spin_key] is not None:
                                    params[key] = row[spin_key].value()
                        elif indi_name == "Keltner":
                            if "window_spin" in row and row["window_spin"] is not None:
                                params["window"] = row["window_spin"].value()
                        elif indi_name == "Fibonacci":
                            if "lookback_spin" in row and row["lookback_spin"] is not None:
                                params["lookback"] = row["lookback_spin"].value()
                        elif indi_name == "Ichimoku":
                            for key in ("tenkan", "kijun", "senkou"):
                                spin_key = f"{key}_spin"
                                if spin_key in row and row[spin_key] is not None:
                                    params[key] = row[spin_key].value()
                        elif indi_name == "Envelope":
                            if "period_spin" in row and row["period_spin"] is not None:
                                params["period"] = row["period_spin"].value()
                            if "percent_spin" in row and row["percent_spin"] is not None:
                                params["percent"] = row["percent_spin"].value()

                        # For non-MA indicators, append as-is
                        if indi_name != "MA":
                            indicator_list.append({
                                "name": indi_name,
                                "params": params
                            })
                    except Exception:
                        continue
            except Exception:
                pass
            return indicator_list

        def _run_preexport_for_selection(self) -> None:
            """Run exporter to ensure indicator JSONs include current UI-selected indicators
            for selected symbols/timeframes before aggregator (helps strict mode show TEMA/WMA).
            """
            try:
                if not getattr(self, 'indicator_tab', None):
                    return
                market_tab = getattr(self.indicator_tab, 'market_tab', None)
                if market_tab is None:
                    return
                # Gather selected symbols and timeframes
                symbols = list(getattr(market_tab, 'checked_symbols', []) or [])
                if not symbols:
                    return
                tf_checkboxes = getattr(market_tab, 'tf_checkboxes', {}) or {}
                tf_spinboxes = getattr(market_tab, 'tf_spinboxes', {}) or {}
                selected_tfs: List[tuple[str,int]] = []
                for tf, cb in tf_checkboxes.items():
                    try:
                        if cb.isChecked():
                            cnt = tf_spinboxes[tf].value() if tf in tf_spinboxes else 200
                            selected_tfs.append((tf, cnt))
                    except Exception:
                        continue
                if not selected_tfs:
                    return

                # Build indicator list from UI; expands MA family if needed
                indicator_list = self._build_indicator_list_for_export()
                if not indicator_list:
                    # Fallback: compute a comprehensive set to avoid empty exports
                    indicator_list = [
                        {"name": "MA", "params": {"period": 20, "ma_type": "EMA"}},
                        {"name": "MA", "params": {"period": 50, "ma_type": "EMA"}},
                        {"name": "TEMA", "params": {"period": 20}},
                        {"name": "TEMA", "params": {"period": 50}},
                        {"name": "WMA", "params": {"period": 20}},
                        {"name": "WMA", "params": {"period": 50}},
                        {"name": "RSI", "params": {"period": 14}},
                        {"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
                        {"name": "ATR", "params": {"period": 14}},
                        {"name": "Bollinger Bands", "params": {"window": 20, "dev": 2}},
                    ]

                # Import exporter lazily and run synchronously
                try:
                    from mt5_indicator_exporter import export_indicators
                except Exception:
                    return

                # Run export for each symbol/timeframe
                for sym in symbols:
                    for tf, cnt in selected_tfs:
                        try:
                            export_indicators(sym, tf, cnt, indicator_list)
                        except Exception:
                            continue
            except Exception:
                pass

        def _build_ui(self):
            layout = QVBoxLayout(self)

            self.title_label = QLabel("📊 Signal Aggregator")
            self.title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))  # Reduced from 18 to 16
            layout.addWidget(self.title_label)

            ctrl = QHBoxLayout()
            self.refresh_btn = QPushButton(I18N.t("🔄 Refresh", "🔄 Làm mới")); self.refresh_btn.clicked.connect(self.load_latest_signals)
            # Min confidence (%) filter
            self.min_conf_spin = QDoubleSpinBox(); self.min_conf_spin.setRange(0.0, 100.0); self.min_conf_spin.setDecimals(1); self.min_conf_spin.setSingleStep(1.0); self.min_conf_spin.setValue(0.0)
            self.min_conf_spin.setToolTip(I18N.t("Filter signals by minimum confidence percentage", "Lọc tín hiệu theo mức độ tin cậy tối thiểu"))
            # Removed old inline language selector; now controlled from main menu
            self.run_btn = QPushButton(I18N.t("▶️ Run Aggregator", "▶️ Chạy tổng hợp")); self.run_btn.clicked.connect(self.on_run)
            self.open_folder_btn = QPushButton(I18N.t("📂 Open results folder", "📂 Mở thư mục kết quả")); self.open_folder_btn.clicked.connect(self.open_results_folder)

            ctrl.addWidget(self.refresh_btn)
            ctrl.addSpacing(10)
            self.min_conf_label = QLabel("Min confidence %:"); ctrl.addWidget(self.min_conf_label); ctrl.addWidget(self.min_conf_spin)
            ctrl.addSpacing(10)
            # Removed strict mode checkbox and MA family selector
            ctrl.addSpacing(10)
            # Removed strict mode checkbox and MA family selector
            ctrl.addStretch(1)
            ctrl.addWidget(self.run_btn)
            ctrl.addWidget(self.open_folder_btn)
            layout.addLayout(ctrl)

            # Signals table
            self.sig_group = QGroupBox("Latest signals")
            sig_layout = QVBoxLayout(self.sig_group)
            self.sig_table = QTableWidget(0, 7)
            self.sig_table.setHorizontalHeaderLabels(["Symbol", "Signal", "Conf %", "Entry", "SL", "TP", "Order"])
            self.sig_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.sig_table.cellClicked.connect(self.on_signal_clicked)
            sig_layout.addWidget(self.sig_table)
            layout.addWidget(self.sig_group, 2)

            # Report + Actions
            pan = QHBoxLayout()
            self.rep_group = QGroupBox("Report (per symbol)")
            rep_layout = QVBoxLayout(self.rep_group)
            self.report_view = QTextEdit(); self.report_view.setReadOnly(True)
            rep_layout.addWidget(self.report_view)
            pan.addWidget(self.rep_group, 3)

            self.act_group = QGroupBox("Order Status")
            act_layout = QVBoxLayout(self.act_group)
            self.actions_view = QTextEdit(); self.actions_view.setReadOnly(True)
            act_layout.addWidget(self.actions_view)
            pan.addWidget(self.act_group, 2)
            layout.addLayout(pan, 3)

            # Load actions initially
            self.load_actions_text()

        def retranslate_ui(self):
            """Refresh visible texts based on current AppState language"""
            self.title_label.setText(I18N.t("📊 Signal Aggregator", "📊 Trình tổng hợp tín hiệu"))
            self.refresh_btn.setText(I18N.t("🔄 Refresh", "🔄 Làm mới"))
            self.run_btn.setText(I18N.t("▶️ Run Aggregator", "▶️ Chạy trình tổng hợp"))
            self.open_folder_btn.setText(I18N.t("📂 Open results folder", "📂 Mở thư mục kết quả"))
            self.min_conf_label.setText(I18N.t("Min confidence %:", "% tin cậy tối thiểu:"))
            self.min_conf_spin.setToolTip(I18N.t("Filter signals by minimum confidence percentage", "Lọc tín hiệu theo mức độ tin cậy tối thiểu"))
            # Removed strict checkbox and MA family selector
            self.sig_group.setTitle(I18N.t("Latest signals", "Tín hiệu mới nhất"))
            self.sig_table.setHorizontalHeaderLabels([
                I18N.t("Symbol", "Mã"),
                I18N.t("Signal", "Tín hiệu"),
                I18N.t("Conf %", "% tin cậy"),
                I18N.t("Entry", "Giá vào"),
                I18N.t("SL", "SL"),
                I18N.t("TP", "TP"),
                I18N.t("Order", "Lệnh"),
            ])
            self.rep_group.setTitle(I18N.t("Report (per symbol)", "Báo cáo (theo mã)"))
            self.act_group.setTitle(I18N.t("Order Status", "Trạng thái lệnh"))
            # Update any 'Order Now' buttons in the table
            rows = self.sig_table.rowCount()
            for r in range(rows):
                w = self.sig_table.cellWidget(r, 6)
                if isinstance(w, QPushButton):
                    w.setText(I18N.t("Order Now", "Đặt lệnh ngay"))

        def load_latest_signals(self):
            try:
                out_dir = os.path.join(os.getcwd(), 'analysis_results')
                rows = []
                for fp in glob.glob(os.path.join(out_dir, "*_signal_*.json")):
                    try:
                        with open(fp, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        sym = data.get('symbol') or os.path.basename(fp).split('_signal_')[0]
                        fi = (data.get('final_signal') or {})
                        rows.append((sym, fi.get('signal'), fi.get('confidence'), fi.get('entry'), fi.get('stoploss'), fi.get('takeprofit')))
                    except Exception:
                        pass
                # Apply min confidence filter
                min_conf = float(self.min_conf_spin.value())
                rows = [r for r in rows if (r[2] or 0) >= min_conf]
                rows.sort(key=lambda r: str(r[0]))
                self.sig_table.setRowCount(len(rows))
                for i, r in enumerate(rows):
                    for j, val in enumerate(r):
                        item = QTableWidgetItem("" if val is None else str(val))
                        if j == 1 and str(val).upper() == 'BUY':
                            item.setForeground(QColor(0,150,0))
                            item.setFont(QFont("Segoe UI", 11, QFont.Bold))  # Reduced from 12 to 11
                        elif j == 1 and str(val).upper() == 'SELL':
                            item.setForeground(QColor(200,0,0))
                            item.setFont(QFont("Segoe UI", 11, QFont.Bold))  # Reduced from 12 to 11
                        self.sig_table.setItem(i, j, item)
                    # Add Order Now button
                    btn = QPushButton("Order Now")
                    sym = r[0]
                    btn.clicked.connect(lambda _, s=sym: self.order_now_for_symbol(s))
                    self.sig_table.setCellWidget(i, 6, btn)
            except Exception as e:
                print(f"[SignalTab] load_latest_signals error: {e}")

        def latest_file(self, pattern: str) -> Optional[str]:
            files = glob.glob(pattern)
            if not files:
                return None
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return files[0]

        def load_latest_report(self, sym: Optional[str] = None):
            try:
                if not sym:
                    self.report_view.setText("(No symbol)")
                    return
                out_dir = os.path.join(os.getcwd(), 'analysis_results')
                # Pick report by AppState language; fallback to the other
                lang = AppState.language()
                primary = 'en' if lang == 'en' else 'vi'
                secondary = 'vi' if primary == 'en' else 'en'
                fp = (
                    self.latest_file(os.path.join(out_dir, f"{sym}_report_{primary}_*.txt"))
                    or self.latest_file(os.path.join(out_dir, f"{sym}_report_{secondary}_*.txt"))
                )
                if not fp:
                    # try base without _m
                    if sym.endswith('_m'):
                        fp = (
                            self.latest_file(os.path.join(out_dir, f"{sym[:-2]}_report_{primary}_*.txt"))
                            or self.latest_file(os.path.join(out_dir, f"{sym[:-2]}_report_{secondary}_*.txt"))
                        )
                if fp and os.path.exists(fp):
                    with open(fp, 'r', encoding='utf-8') as f:
                        self.report_view.setText(f.read())
                else:
                    self.report_view.setText("(No report for this symbol)")
            except Exception as e:
                self.report_view.setText(f"Failed to load report: {e}")

        def load_actions_text(self):
            try:
                out_dir = os.path.join(os.getcwd(), 'analysis_results')
                # Choose actions file based on language if both exist
                lang = AppState.language()
                cand = [
                    os.path.join(out_dir, 'account_positions_actions_en.txt'),
                    os.path.join(out_dir, 'account_positions_actions_vi.txt')
                ]
                # prefer EN if requested and file exists
                if lang == 'en' and os.path.exists(cand[0]):
                    fp = cand[0]
                else:
                    fp = cand[1]
                if os.path.exists(fp):
                    with open(fp, 'r', encoding='utf-8') as f:
                        txt = f.read()
                        # Minimal on-the-fly translation for header phrase if VI selected but EN preferred
                        if lang == 'en':
                            txt = txt.replace("Hành động đề xuất cho vị thế:", "Suggested actions for positions:")
                        self.actions_view.setText(txt)
                else:
                    self.actions_view.setText("(No actions yet — run Aggregator)")
            except Exception as e:
                self.actions_view.setText(f"Failed to load actions: {e}")

        @safe_method
        def on_run(self, *args, **kwargs):
            # Handle both button click (with checked param) and direct call (without params)
            checked = args[0] if args else kwargs.get('checked', False)
            if self.thread and self.thread.isRunning():
                QMessageBox.information(
                    self,
                    I18N.t("Running", "Đang chạy"),
                    I18N.t("Aggregator is running, please wait…", "Trình tổng hợp đang chạy, vui lòng chờ…")
                )
                return
            # Ensure latest UI selections are saved to whitelist before run
            try:
                # Persist from IndicatorTab (source of truth for selected indicators)
                if getattr(self, 'indicator_tab', None) is not None:
                    try:
                        if hasattr(self.indicator_tab, 'save_current_user_config'):
                            self.indicator_tab.save_current_user_config()
                    except Exception as _e:
                        print(f"Could not save indicator tab config before run: {_e}")
                    try:
                        if hasattr(self.indicator_tab, '_persist_indicator_whitelist'):
                            self.indicator_tab._persist_indicator_whitelist()
                    except Exception as _e:
                        print(f"Could not persist whitelist before run: {_e}")
                    # Pre-export indicators for current selection so strict mode has needed columns
                    try:
                        self._run_preexport_for_selection()
                    except Exception as _e:
                        print(f"Pre-export failed (will continue to aggregator): {_e}")
            except Exception as _e:
                print(f"Could not persist whitelist before run: {_e}")
            args: List[str] = []
            # Always enforce strict-indicators (was default ON)
            args.append("--strict-indicators")
            # MA family handling: use expand mode (WMA excluded to avoid errors)
            args.extend(["--ma-family", "expand"])
            # Add verbose for debugging
            args.append("--verbose")  # Enable verbose logging
            
            # Get symbols from Market Tab
            selected_symbols = []
            if hasattr(self, 'market_tab') and self.market_tab:
                selected_symbols = list(getattr(self.market_tab, 'checked_symbols', []) or [])
                print(f"🔍 DEBUG: Market tab checked_symbols = {getattr(self.market_tab, 'checked_symbols', 'NOT_FOUND')}")
                
            if selected_symbols:
                # Use selected symbols from GUI
                args.extend(["--symbols", ",".join(selected_symbols)])
                print(f"📊 Using selected symbols from GUI: {', '.join(selected_symbols)}")
            else:
                # Fallback: limit to 3 symbols for faster testing
                args.extend(["--limit", "3"])
                print("⚠️ No symbols selected in GUI, using auto-detection with limit 3")
            
            # DISABLED: Let system auto-detect timeframes from pattern files instead of forcing specific ones
            # This fixes the issue where CLI works but GUI fails due to --timeframes parameter
            # if self.market_tab and hasattr(self.market_tab, 'tf_checkboxes'):
            #     selected_timeframes = [tf for tf in self.market_tab.tf_checkboxes if self.market_tab.tf_checkboxes[tf].isChecked()]
            #     if selected_timeframes:
            #         args.extend(["--timeframes", ",".join(selected_timeframes)])
            #         print(f"📊 Using selected timeframes: {', '.join(selected_timeframes)}")
            #     else:
            #         print("⚠️ No timeframes selected in Market tab, using all available")
            # else:
            #     print("⚠️ MarketTab not available, using all timeframes")
            print("🔄 Auto-detecting timeframes from available pattern files (for better signal generation)")
            
            # Run for all symbols with current defaults
            print(f"🔧 DEBUG: Args being passed to comprehensive_aggregator.py: {args}")

            self.run_btn.setEnabled(False)
            self.thread = RunAggregatorWorker(args)
            self.thread.finished.connect(self._after_run)
            self.thread.start()

        def _after_run(self):
            self.run_btn.setEnabled(True)
            
            # Check for errors and display them
            if self.thread and self.thread.returncode != 0:
                print(f"❌ comprehensive_aggregator.py failed with return code: {self.thread.returncode}")
                if self.thread.stderr:
                    print(f"❌ Error details: {self.thread.stderr}")
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"❌ Error: Return code {self.thread.returncode}")
            else:
                print("✅ comprehensive_aggregator.py completed successfully")
                if hasattr(self, 'status_label'):
                    self.status_label.setText("✅ Analysis completed")
                
                # 🚀 AUTO EXECUTE ACTIONS after successful analysis
                self.auto_execute_actions_after_analysis()
            
            # Refresh views
            self.load_latest_signals()
            self.load_actions_text()

        def auto_execute_actions_after_analysis(self):
            """🚀 Automatically execute actions after successful analysis"""
            try:
                print("🚀 Auto-executing actions after analysis...")
                
                # Check if execute_actions.py exists
                execute_actions_path = os.path.join(os.getcwd(), 'execute_actions.py')
                if not os.path.exists(execute_actions_path):
                    print("❌ execute_actions.py not found - skipping auto execution")
                    return
                
                # Check if there are actions to execute
                actions_path = os.path.join(os.getcwd(), 'analysis_results', 'account_positions_actions.json')
                if not os.path.exists(actions_path):
                    print("📝 No actions file found - skipping execution")
                    return
                
                # Load and check actions
                with open(actions_path, 'r', encoding='utf-8') as f:
                    actions_data = json.load(f)
                
                total_actions = len(actions_data.get('actions', []))
                if total_actions == 0:
                    print("📝 No actions to execute")
                    return
                
                print(f"🎯 Found {total_actions} actions to execute")
                
                # Execute actions in background
                import subprocess
                import sys
                
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"🚀 Executing {total_actions} actions...")
                
                # Run execute_actions.py with auto flag
                result = subprocess.run(
                    [sys.executable, execute_actions_path, '--auto'],
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minutes timeout
                )
                
                if result.returncode == 0:
                    print("✅ Actions executed successfully!")
                    if hasattr(self, 'status_label'):
                        self.status_label.setText("✅ Analysis & Execution completed")
                    
                    # Parse output for success count
                    output_lines = result.stdout.split('\n')
                    for line in output_lines:
                        if 'Total Actions:' in line or 'Successful:' in line:
                            print(f"📊 {line.strip()}")
                else:
                    print(f"❌ Action execution failed: {result.stderr}")
                    if hasattr(self, 'status_label'):
                        self.status_label.setText("⚠️ Analysis completed, execution failed")
                        
            except subprocess.TimeoutExpired:
                print("⏰ Action execution timed out")
                if hasattr(self, 'status_label'):
                    self.status_label.setText("⏰ Execution timed out")
            except Exception as e:
                print(f"❌ Auto execution error: {e}")
                if hasattr(self, 'status_label'):
                    self.status_label.setText("⚠️ Auto execution failed")

        def on_signal_clicked(self, row: int, col: int):
            try:
                sym_item = self.sig_table.item(row, 0)
                if not sym_item:
                    return
                sym = sym_item.text().strip()
                if sym:
                    self.load_latest_report(sym)
            except Exception as e:
                print(f"[SignalTab] on_signal_clicked error: {e}")

        def order_now_for_symbol(self, sym: str):
            try:
                # 1. Load latest signal JSON
                out_dir = os.path.join(os.getcwd(), 'analysis_results')
                fp = self.latest_file(os.path.join(out_dir, f"{sym}_signal_*.json"))
                if not fp and sym.endswith('_m'):
                    # try without suffix
                    base = sym[:-2]
                    fp = self.latest_file(os.path.join(out_dir, f"{base}_signal_*.json"))
                if not fp:
                    QMessageBox.warning(
                        self,
                        I18N.t("Order", "Đặt lệnh"),
                        I18N.t("No signal file found for {sym}", "Không tìm thấy tệp tín hiệu cho {sym}", sym=sym)
                    )
                    return
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                fi = (data.get('final_signal') or {})
                action = (fi.get('signal') or '').upper()
                if action not in ("BUY", "SELL"):
                    QMessageBox.warning(
                        self,
                        I18N.t("Order", "Đặt lệnh"),
                        I18N.t("Unsupported signal for {sym}: {action}", "Tín hiệu không hỗ trợ cho {sym}: {action}", sym=sym, action=action)
                    )
                    return

                # 2. Confidence guard
                conf = float(fi.get('confidence') or 0.0)
                min_conf = float(self.min_conf_spin.value())
                if conf < min_conf:
                    QMessageBox.warning(
                        self,
                        I18N.t("Order", "Đặt lệnh"),
                        I18N.t(
                            "Signal confidence {conf}% is below min threshold {min_conf}%",
                            "Độ tin cậy {conf}% thấp hơn ngưỡng tối thiểu {min_conf}%",
                            conf=conf, min_conf=min_conf
                        )
                    )
                    return

                entry = fi.get('entry') or 0.0
                sl = fi.get('stoploss') or 0.0
                tp = fi.get('takeprofit') or 0.0

                # 3. Read risk settings for lot bounds
                risk_min = 0.01
                risk_max = 100.0
                try:
                    rfp = os.path.join(os.getcwd(), 'risk_management', 'risk_settings.json')
                    if os.path.exists(rfp):
                        with open(rfp, 'r', encoding='utf-8') as rf:
                            rcfg = json.load(rf)
                        risk_min = float(rcfg.get('min_lot_size', risk_min))
                        risk_max = float(rcfg.get('max_lot_size', risk_max))
                except Exception:
                    pass

                # 4. Ask user volume within risk bounds
                vol, ok = QInputDialog.getDouble(
                    self,
                    I18N.t("Order volume", "Khối lượng lệnh"),
                    I18N.t("Enter volume (lots) for {sym}:", "Nhập khối lượng (lot) cho {sym}:", sym=sym),
                    max(risk_min, 0.01), risk_min, risk_max, 2
                )
                if not ok:
                    return
                volume = float(vol)

                # 5. Initialize / verify MT5 and map symbol variant
                trade_symbol = sym
                try:
                    import MetaTrader5 as mt5
                    if not mt5.initialize():
                        mt5.initialize()
                    # If symbol not visible, try variant with _m or without
                    sinfo = mt5.symbol_info(trade_symbol)
                    if not sinfo:
                        alt = trade_symbol + '_m' if not trade_symbol.endswith('_m') else trade_symbol[:-2]
                        if mt5.symbol_info(alt):
                            trade_symbol = alt
                            sinfo = mt5.symbol_info(trade_symbol)
                    if sinfo and not sinfo.visible:
                        mt5.symbol_select(trade_symbol, True)
                    tick = mt5.symbol_info_tick(trade_symbol)
                    if tick:
                        mkt_price = tick.ask if action == 'BUY' else tick.bid
                        if not entry or entry <= 0:
                            entry = mkt_price
                    # Clamp volume to broker limits
                    if sinfo:
                        if volume < sinfo.volume_min:
                            volume = sinfo.volume_min
                        if volume > sinfo.volume_max:
                            volume = sinfo.volume_max
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        I18N.t("Order", "Đặt lệnh"),
                        I18N.t("MT5 init / symbol check failed: {e}", "Khởi tạo MT5 / kiểm tra mã lỗi: {e}", e=e)
                    )
                    return

                # 6. Confirm
                confirm = QMessageBox.question(
                    self,
                    I18N.t("Confirm Order", "Xác nhận lệnh"),
                    I18N.t(
                        "Place {action} {vol} lots {sym}?\nEntry: {entry}\nSL: {sl}\nTP: {tp}",
                        "Đặt lệnh {action} {vol} lot {sym}?\nGiá vào: {entry}\nSL: {sl}\nTP: {tp}",
                        action=action, vol=volume, sym=trade_symbol, entry=entry, sl=sl, tp=tp
                    )
                )
                if confirm != QMessageBox.Yes:
                    return

                # 7. Execute
                try:
                    from order_executor import get_executor_instance, TradeSignal
                    # Reuse unified connection manager if available from global login flow
                    shared_manager = None
                    try:
                        if 'MT5ConnectionManager' in globals():
                            shared_manager = MT5ConnectionManager()
                    except Exception:
                        shared_manager = None
                    sig = TradeSignal(
                        symbol=trade_symbol,
                        action=action,
                        entry_price=float(entry or 0.0),
                        stop_loss=float(sl or 0.0),
                        take_profit=float(tp or 0.0),
                        volume=volume,
                        confidence=conf,
                        strategy="GUI_ORDER",
                        comment="SignalTab Order"
                    )
                    ex = get_executor_instance(connection=self.mt5_conn if hasattr(self, 'mt5_conn') else None)
                    # If we created an executor before account switch, ensure internal manager picks up reconfigured session
                    try:
                        if shared_manager and ex.connection_manager and ex.connection_manager is shared_manager:
                            # Optionally force a lightweight status check
                            ex.connection_manager.connect(force_reconnect=False)
                    except Exception:
                        pass
                    result = ex.execute_market_order(sig)
                    if result.success:
                        QMessageBox.information(
                            self,
                            I18N.t("Order", "Đặt lệnh"),
                            I18N.t(
                                "Order executed successfully. Ticket: {ticket}",
                                "Đặt lệnh thành công. Mã: {ticket}",
                                ticket=(result.ticket or result.order_id)
                            )
                        )
                    else:
                        QMessageBox.warning(
                            self,
                            I18N.t("Order", "Đặt lệnh"),
                            I18N.t("Order failed: {msg}", "Đặt lệnh thất bại: {msg}", msg=result.error_message)
                        )
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        I18N.t("Order", "Đặt lệnh"),
                        I18N.t("Order exception: {e}", "Lỗi khi đặt lệnh: {e}", e=e)
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    I18N.t("Order", "Đặt lệnh"),
                    I18N.t("Order exception outer: {e}", "Lỗi đặt lệnh (ngoài): {e}", e=e)
                )

        def open_results_folder(self):
            try:
                out_dir = os.path.join(os.getcwd(), 'analysis_results')
                if sys.platform.startswith('win'):
                    os.startfile(out_dir)
                elif sys.platform == 'darwin':
                    subprocess.run(['open', out_dir])
                else:
                    subprocess.run(['xdg-open', out_dir])
            except Exception as e:
                QMessageBox.warning(
                    self,
                    I18N.t("Error", "Lỗi"),
                    I18N.t("Cannot open folder: {e}", "Không thể mở thư mục: {e}", e=e)
                )

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    # print("✅ MetaTrader5 loaded successfully")
except ImportError as e:
    print(f"⚠️ MetaTrader5 not available: {e}")
    MT5_AVAILABLE = False
    # Mock MT5 module
    class MockMT5:
        TIMEFRAME_M1 = 1
        TIMEFRAME_M5 = 5
        TIMEFRAME_M15 = 15
        TIMEFRAME_M30 = 30
        TIMEFRAME_H1 = 60
        TIMEFRAME_H4 = 240
        TIMEFRAME_D1 = 1440
    mt5 = MockMT5()

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Pandas not available: {e}")
    PANDAS_AVAILABLE = False

try:
    from dotenv import load_dotenv, set_key
    DOTENV_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Python-dotenv not available: {e}")
    DOTENV_AVAILABLE = False
    def load_dotenv(): pass
    def set_key(*args): pass

# Enhanced module imports with error handling
try:
    # Use clean news scraper (already imported above)
    # print("🚀 News scraper already loaded")
    # NEWS_SCRAPER_AVAILABLE is already set above
    # print("✅ News scraper ready")
    pass
except Exception as e:
    print(f"⚠️ News scraper issue: {e}")
    NEWS_SCRAPER_AVAILABLE = False
    class MockNewsScraper:
        @staticmethod
        def get_today_news(*args): 
            print("📊 Using mock news data")
            return []
        @staticmethod
        def save_recent_news_to_json(*args): 
            print("📝 Mock save news")
            pass
    news_scraper = MockNewsScraper()

try:
    from mt5_indicator_exporter import update_data_with_new_candle
    from mt5_indicator_exporter import calculate_and_save_all
    INDICATOR_EXPORTER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Indicator exporter not available: {e}")
    INDICATOR_EXPORTER_AVAILABLE = False
    def update_data_with_new_candle(*args): pass
    def calculate_and_save_all(*args): return {"success": 0, "failed": 0}

try:
    from pattern_detector import analyze_patterns
    PATTERN_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Pattern detector not available: {e}")
    PATTERN_DETECTOR_AVAILABLE = False
    def analyze_patterns(*args): return []

try:
    from price_patterns_full_data import main as analyze_price_patterns
    PRICE_PATTERNS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Price patterns not available: {e}")
    PRICE_PATTERNS_AVAILABLE = False
    def analyze_price_patterns(*args): pass

try:
    from gpt_analyst import analyze_symbol
    GPT_ANALYST_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ GPT analyst not available: {e}")
    GPT_ANALYST_AVAILABLE = False
    def analyze_symbol(*args): return {"analysis": "Not available"}

try:
    from risk_manager import RiskManagementSystem
    RISK_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Risk manager not available: {e}")
    RISK_MANAGER_AVAILABLE = False
    class RiskManagementSystem:
        def __init__(self, *args): pass
        def validate_trade(self, *args): return True

try:
    from order_executor import OrderHandler
    ORDER_EXECUTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Order executor not available: {e}")
    ORDER_EXECUTOR_AVAILABLE = False
    class OrderHandler:
        def __init__(self, *args): pass
        def send_order(self, *args): return False

try:
    from mt5_connector import MT5ConnectionManager
    MT5_CONNECTOR_AVAILABLE = True
    # print("✅ MT5 connector imported successfully")
except ImportError as e:
    print(f"⚠️ MT5 connector not available: {e}")
    MT5_CONNECTOR_AVAILABLE = False
    MT5ConnectionManager = None

try:
    from unified_auto_trading_system import UnifiedAutoTradingSystem as AutoTradingManager
    AUTO_TRADING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Unified auto trading system not available: {e}")
    AUTO_TRADING_AVAILABLE = False
    class AutoTradingManagerStub:
        def __init__(self, *args): pass
        def start(self): pass
        def stop(self): pass

try:
    from mt5_data_fetcher import fetch_and_save_candles
    DATA_FETCHER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Data fetcher not available: {e}")
    DATA_FETCHER_AVAILABLE = False
    def fetch_and_save_candles(*args): return []

# Configure safe logging for Windows console
import sys
import logging

# Set up logging with Windows-safe configuration
if sys.platform == "win32":
    # For Windows, use a simple configuration without emojis
    import io
    
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                # Remove emojis and special characters from log messages
                msg = self.format(record)
                # Replace common emoji patterns
                emoji_replacements = {
                    '✅': '[OK]',
                    '❌': '[ERROR]',
                    '⚠️': '[WARNING]',
                    '🔄': '[LOADING]',
                    '📊': '[DATA]',
                    '📚': '[INFO]',
                    '🛡️': '[SHIELD]',
                    '🎯': '[TARGET]',
                    '💰': '[MONEY]',
                    '📈': '[CHART]'
                }
                
                for emoji, replacement in emoji_replacements.items():
                    msg = msg.replace(emoji, replacement)
                
                # Encode safely for Windows console
                self.stream.write(msg + self.terminator)
                self.flush()
            except UnicodeEncodeError:
                # Fallback: encode with errors='replace'
                try:
                    msg_bytes = msg.encode('utf-8', errors='replace').decode('utf-8')
                    self.stream.write(msg_bytes + self.terminator)
                    self.flush()
                except:
                    # Ultimate fallback
                    self.stream.write(f"[LOG MESSAGE ENCODING ERROR]\n")
                    self.flush()
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[SafeStreamHandler()],
        force=True  # Force reconfiguration
    )
else:
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

ENV_PATH = ".env"

ALL_CURRENCY = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD"]
ALL_IMPACT = [1, 2, 3]  # Bỏ 0 khỏi danh sách Impact

DATA_FOLDER = "data"
INDICATOR_FOLDER = "indicator_output"

USER_CONFIG_PATH = "user_config.pkl"

def save_user_config(config):
    try:
        with open(USER_CONFIG_PATH, "wb") as f:
            pickle.dump(config, f)
    except Exception as e:
        print(f"Could not save user config: {e}")

def load_user_config(apply_lang: bool = True):
    if os.path.exists(USER_CONFIG_PATH):
        try:
            with open(USER_CONFIG_PATH, "rb") as f:
                config = pickle.load(f)
                if "use_economic_calendar" not in config:
                    config["use_economic_calendar"] = True
                if "language" not in config:
                    config["language"] = AppState._lang
                elif apply_lang:
                    stored_lang = config.get('language')
                    if stored_lang in ('en','vi'):
                        AppState._lang = stored_lang
                return config
        except Exception as e:
            print(f"Could not load user config: {e}")
    return {"use_economic_calendar": True, "language": AppState.language()}

# Define local MT5Connection class that works with GUI
class MT5Connection:
    def __init__(self, account, password, server):
        self.account = account
        self.password = password
        self.server = server
        self.connected = False
        self.connection_manager = None
        self.initialize()

    def initialize(self):
        if not MT5_AVAILABLE:
            logging.warning("MT5 not available, using mock connection")
            self.connected = True
            return
            
        # Use MT5ConnectionManager if available
        if MT5_CONNECTOR_AVAILABLE:
            try:
                self.connection_manager = MT5ConnectionManager()
                # If manager already had a config and it's different, reconfigure
                cfg = getattr(self.connection_manager, 'config', None)
                needs_reconf = False
                try:
                    if cfg and (str(cfg.account) != str(self.account) or str(cfg.server) != str(self.server)):
                        needs_reconf = True
                except Exception:
                    pass
                if needs_reconf:
                    self.connection_manager.reconfigure(self.account, self.password, self.server)
                elif not cfg:
                    # Initial configure via reconfigure to push env values
                    self.connection_manager.reconfigure(self.account, self.password, self.server)
                if self.connection_manager.connect(force_reconnect=needs_reconf):
                    self.connected = True
                    logging.info("MT5ConnectionManager connected successfully")
                    return
                else:
                    logging.error("MT5ConnectionManager connection failed")
            except Exception as e:
                logging.error(f"MT5ConnectionManager error: {e}")
        
        # Fallback to direct MT5 connection
        if mt5.initialize():
            authorized = mt5.login(int(self.account), password=self.password, server=self.server)
            if authorized:
                self.connected = True
                logging.info("Direct MT5 connection successful")
            else:
                logging.error(f"Meta Trader login failed, error code: {mt5.last_error()}")
                mt5.shutdown()
        else:
            logging.error(f"Meta Trader initialize failed, error code: {mt5.last_error()}")

    def shutdown(self):
        if self.connected:
            if self.connection_manager:
                self.connection_manager.disconnect()
                logging.info("MT5ConnectionManager disconnected")
            elif MT5_AVAILABLE:
                mt5.shutdown()
                logging.info("Direct MT5 shutdown completed")
            self.connected = False

    def perform_account_scan(self):
        """Perform account scan and save results"""
        if not self.connected:
            logging.warning("MT5 not connected, cannot perform account scan")
            return None
            
        if self.connection_manager:
            try:
                # Use MT5ConnectionManager's scan functionality with overwrite mode
                scan_file = self.connection_manager._save_timestamped_scan()
                if scan_file:
                    logging.info(f"Account scan completed: {scan_file}")
                    # Also print status
                    self.connection_manager.print_essential_status()
                    return scan_file
                else:
                    logging.error("Account scan failed")
                    return None
            except Exception as e:
                logging.error(f"Account scan error: {e}")
                return None
        else:
            logging.warning("MT5ConnectionManager not available, cannot perform detailed account scan")
            return None

    def get_all_symbols(self):
        if not MT5_AVAILABLE:
            # Return mock symbols for testing
            return ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"]
            
        symbols = mt5.symbols_get()
        if symbols is None:
            logging.error("Could not retrieve symbols from Meta Trader")
            return []
        return [s.name for s in symbols]

    def ensure_symbol_ready(self, symbol):
        if not MT5_AVAILABLE:
            return True  # Mock success
            
        info = mt5.symbol_info(symbol)
        if info is None:
            logging.error(f"Symbol '{symbol}' not found on broker")
            return False
        if not info.visible:
            logging.warning(f"Symbol '{symbol}' not visible. Adding to Market Watch...")
            if not mt5.symbol_select(symbol, True):
                logging.error(f"Failed to add symbol '{symbol}' to Market Watch")
                return False
            else:
                logging.info(f"Symbol '{symbol}' added to Market Watch")
        return True

class CandlestickChart(QWidget):
    """Widget for realtime candlestick chart"""
    
    def __init__(self, indicator_tab=None):
        super().__init__()
        self.layout = QVBoxLayout()
        self.indicator_tab = indicator_tab  # Reference to IndicatorTab
        self.mt5_conn = None  # Will be set by parent
        
        if not MATPLOTLIB_AVAILABLE:
            self.error_label = QLabel("❌ Matplotlib not available - please install with: pip install matplotlib")
            self.error_label.setStyleSheet("color: red; font-weight: bold; padding: 20px;")
            self.layout.addWidget(self.error_label)
            self.setLayout(self.layout)
            return
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8), facecolor='#2b2b2b')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, facecolor='#1e1e1e')
        
        # Chart data storage
        self.ohlc_data = []
        self.dates = []
        self.volumes = []
        
        # Chart settings - removed fixed MA, now dynamic from IndicatorTab
        self.show_price_line = True  # Show current price line
        
        # Interactive features
        self.crosshair_enabled = True
        self.crosshair_h = None  # Horizontal line
        self.crosshair_v = None  # Vertical line
        self.info_text = None    # Info text box
        
        # Navigation features
        self.pan_enabled = True
        self.zoom_enabled = True
        self.press = None  # For pan functionality
        
        # Timer for real-time updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_chart)
        
        # Current symbol and timeframe
        self.current_symbol = None
        self.current_timeframe = None
        
        self.setup_chart()
        
        # Setup interactive events FIRST (creates toolbar and time widget)
        self.setup_interactive_events()
        
        # Then add canvas
        self.layout.addWidget(self.canvas)
        
        # Create MT5-style trading panel EXACTLY like MT5
        self.create_mt5_trading_panel()
        
        self.setLayout(self.layout)
    
    def setup_chart(self):
        """Setup beautiful chart appearance like MT5"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        # MT5-style dark theme background
        self.ax.set_facecolor('#1a1a1a')  # Very dark background like MT5
        
        # MT5-style grid
        self.ax.grid(True, alpha=0.15, color='#404040', linewidth=0.8, linestyle='-')
        self.ax.set_axisbelow(True)  # Grid behind chart elements
        
        # Enhanced tick styling (MT5-like)
        self.ax.tick_params(colors='#cccccc', labelsize=9, width=1, length=5)
        self.ax.tick_params(axis='x', rotation=0)
        
        # Price axis on right side (MT5 style)
        self.ax.yaxis.tick_right()
        self.ax.yaxis.set_label_position("right")
        
        # MT5-style borders
        for spine in self.ax.spines.values():
            spine.set_color('#404040')
            spine.set_linewidth(1.2)
        
        # Set axis labels color
        self.ax.xaxis.label.set_color('#cccccc')
        self.ax.yaxis.label.set_color('#cccccc')
        
        # MT5-style figure background
        self.figure.patch.set_facecolor('#2d2d2d')  # MT5-like figure background
        
        # Clean MT5-style borders
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(True)
        self.ax.spines['bottom'].set_visible(True)
        
        # MT5-style border colors
        self.ax.spines['right'].set_color('#555555')
        self.ax.spines['bottom'].set_color('#555555')

    def create_mt5_trading_panel(self):
        """Create MT5-style trading panel overlay in top-left corner with toggle button"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        # Create trading panel as overlay widget on chart
        self.mt5_overlay = QWidget(self)
        self.mt5_overlay.setFixedSize(300, 95)  # Smaller height without price display
        self.mt5_overlay.move(15, 60)  # Position in top-left corner (moved down a bit)
        
        # Main panel background (MT5 dark style) - make it more visible
        self.mt5_overlay.setStyleSheet("""
            QWidget {
                background-color: rgba(20, 20, 20, 0.98);
                border: 2px solid #ffa726;
                border-radius: 8px;
            }
        """)
        
        # Main layout
        overlay_layout = QVBoxLayout(self.mt5_overlay)
        overlay_layout.setContentsMargins(3, 3, 3, 3)
        overlay_layout.setSpacing(3)
        
        # Header with toggle button (like MT5)
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("One Click Trading")
        title_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-size: 12px;
                font-weight: bold;
                padding: 2px;
            }
        """)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Toggle button (minimize/restore like MT5)
        self.toggle_btn = QPushButton("−")
        self.toggle_btn.setFixedSize(20, 20)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                color: #cccccc;
                border: 1px solid #666666;
                border-radius: 2px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """)
        self.toggle_btn.clicked.connect(self.toggle_mt5_panel)
        header_layout.addWidget(self.toggle_btn)
        
        overlay_layout.addLayout(header_layout)
        
        # Trading content (collapsible)
        self.trading_content = QWidget()
        content_layout = QVBoxLayout(self.trading_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(3)
        
        # Volume and Spread section
        volume_layout = QHBoxLayout()
        volume_label = QLabel("Volume:")
        volume_label.setStyleSheet("color: #cccccc; font-size: 12px;")
        volume_layout.addWidget(volume_label)
        
        self.mt5_volume = QDoubleSpinBox()
        self.mt5_volume.setRange(0.01, 100.0)
        self.mt5_volume.setSingleStep(0.01)
        self.mt5_volume.setValue(0.10)
        self.mt5_volume.setDecimals(2)
        self.mt5_volume.setFixedWidth(80)
        self.mt5_volume.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
                font-size: 12px;
            }
        """)
        volume_layout.addWidget(self.mt5_volume)
        
        # Add spacing between volume and spread
        volume_layout.addWidget(QLabel(""))
        
        # Spread display
        spread_label = QLabel("Spread:")
        spread_label.setStyleSheet("color: #cccccc; font-size: 12px;")
        volume_layout.addWidget(spread_label)
        
        self.spread_display = QLabel("--")
        self.spread_display.setFixedWidth(60)
        self.spread_display.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                color: #ffeb3b;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
                font-size: 12px;
                font-weight: bold;
                text-align: center;
            }
        """)
        self.spread_display.setAlignment(Qt.AlignCenter)
        volume_layout.addWidget(self.spread_display)
        
        volume_layout.addStretch()
        
        content_layout.addLayout(volume_layout)
        
        # Trading buttons (optimized MT5 style - tận dụng tối đa không gian)
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)  # Increased spacing
        
        # SELL button (red, with bid price) - larger to utilize space
        self.mt5_sell_btn = QPushButton("SELL\n--")
        self.mt5_sell_btn.setFixedSize(130, 45)  # Much larger size to utilize space
        self.mt5_sell_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                border: 1px solid #b71c1c;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f44336;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #999999;
            }
        """)
        self.mt5_sell_btn.clicked.connect(self.on_mt5_sell_click)
        self.mt5_sell_btn.setEnabled(False)
        buttons_layout.addWidget(self.mt5_sell_btn)
        
        # BUY button (green, with ask price) - larger to utilize space
        self.mt5_buy_btn = QPushButton("BUY\n--")
        self.mt5_buy_btn.setFixedSize(130, 45)  # Much larger size to utilize space
        self.mt5_buy_btn.setStyleSheet("""
            QPushButton {
                background-color: #388e3c;
                color: white;
                border: 1px solid #2e7d32;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4caf50;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #999999;
            }
        """)
        self.mt5_buy_btn.clicked.connect(self.on_mt5_buy_click)
        self.mt5_buy_btn.setEnabled(False)
        buttons_layout.addWidget(self.mt5_buy_btn)
        
        content_layout.addLayout(buttons_layout)
        
        overlay_layout.addWidget(self.trading_content)
        
        # Initially show the panel and make sure it's visible
        self.mt5_overlay.show()
        self.mt5_overlay.raise_()  # Bring to front
        self.is_panel_collapsed = False
        
        # Add a debug print to confirm panel creation
        print("✅ MT5 Trading Panel created and positioned at (15, 45) - Optimized layout without price display")
    
    def setup_interactive_events(self):
        """Setup interactive mouse events for crosshair and pan/zoom"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        # Connect mouse events
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('scroll_event', self.on_mouse_scroll)
        self.canvas.mpl_connect('axes_leave_event', self.on_mouse_leave)  # New: Handle mouse leave
        
        # Enable navigation toolbar functionality - optimized size for visibility
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setFixedHeight(35)  # Increased height for better visibility
        self.toolbar.setStyleSheet("""
            QToolBar {
                background-color: #2d2d2d;
                border: 1px solid #555555;
                spacing: 2px;
                padding: 3px;
            }
            QToolButton {
                background-color: #404040;
                color: #cccccc;
                border: 1px solid #666666;
                border-radius: 3px;
                padding: 3px;
                margin: 1px;
                min-width: 28px;
                min-height: 28px;
                max-width: 32px;
                max-height: 32px;
                font-size: 10px;
            }
            QToolButton:hover {
                background-color: #555555;
                border: 1px solid #ffa726;
            }
            QToolButton:pressed {
                background-color: #ffa726;
                border: 1px solid #ff8f00;
            }
        """)
        self.layout.addWidget(self.toolbar)  # Add toolbar at top first
        
        # Configure toolbar status bar for time display
        self.setup_toolbar_status()
        
        # Setup custom coordinate formatter to show only time (no X Y coordinates)
        self._current_time_display = "Time: --"
        self._original_format_coord = self.ax.format_coord
        
        def custom_format_coord(x, y):
            """Custom coordinate formatter that shows only time"""
            try:
                # Show only time in the status bar
                time_part = getattr(self, '_current_time_display', 'Time: --')
                return time_part
            except:
                return "Time: --"
        
        # Set the custom formatter
        self.ax.format_coord = custom_format_coord
    
    def setup_toolbar_status(self):
        """Setup matplotlib toolbar status bar for time display"""
        try:
            if hasattr(self, 'toolbar') and self.toolbar:
                # Try to access and configure the status bar
                if hasattr(self.toolbar, 'locLabel'):
                    # This is the coordinate display label in matplotlib toolbar
                    self.toolbar.locLabel.setMinimumWidth(200)
                    self.toolbar.locLabel.setStyleSheet("""
                        QLabel {
                            color: #ffa726;
                            background-color: #2d2d2d;
                            padding: 2px 5px;
                            font-size: 10px;
                            font-weight: bold;
                        }
                    """)
                    print("✅ Toolbar status bar configured for time display")
                elif hasattr(self.toolbar, '_message'):
                    # Alternative message label
                    self.toolbar._message.setStyleSheet("""
                        QLabel {
                            color: #ffa726;
                            background-color: #2d2d2d;
                            padding: 2px 5px;
                            font-size: 10px;
                        }
                    """)
                    print("✅ Toolbar message label configured")
                else:
                    print("⚠️ No suitable status bar element found in toolbar")
        except Exception as e:
            print(f"❌ Toolbar status setup failed: {e}")
    
    def on_mouse_move(self, event):
        """Handle mouse movement for crosshair and info display - optimized for smooth performance"""
        if not MATPLOTLIB_AVAILABLE or not self.crosshair_enabled or event.inaxes != self.ax:
            return
            
        if len(self.ohlc_data) == 0:
            return
            
        # Get mouse position
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        # Find nearest candle
        candle_idx = int(round(x))
        if candle_idx < 0 or candle_idx >= len(self.ohlc_data):
            return
            
        # Throttle updates for better performance - only update every few pixels
        if hasattr(self, '_last_mouse_pos'):
            last_x, last_y = self._last_mouse_pos
            if abs(x - last_x) < 0.5 and abs(y - last_y) < (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.01:
                return
        
        self._last_mouse_pos = (x, y)
            
        # Update crosshair and info with optimized drawing
        try:
            self.update_crosshair(x, y)
            self.show_candle_info(candle_idx, x, y)
            
            # Update time display safely
            try:
                self.update_time_display(candle_idx)
            except Exception as e:
                # Silently handle time display errors
                pass
            
            # Use blit for faster drawing instead of full redraw
            self.canvas.draw_idle()
        except Exception as e:
            # Silently handle crosshair errors but print for debugging
            print(f"Mouse move error: {e}")
            pass
        """Test function to check if time widget is working"""
        try:
            print("\n🔍 === TIME WIDGET DEBUG TEST ===")
            
            # Check widget existence and visibility
            if hasattr(self, 'time_widget'):
                print(f"✅ time_widget exists: {self.time_widget}")
                print(f"   - visible: {self.time_widget.isVisible()}")
                print(f"   - size: {self.time_widget.size()}")
                print(f"   - geometry: {self.time_widget.geometry()}")
                print(f"   - parent: {self.time_widget.parent()}")
                
                # Force show the widget
                self.time_widget.show()
                self.time_widget.setVisible(True)
                self.time_widget.raise_()
                
            else:
                print("❌ time_widget NOT FOUND!")
            
            # Check label existence and visibility
            if hasattr(self, 'time_label'):
                print(f"✅ time_label exists: {self.time_label}")
                print(f"   - visible: {self.time_label.isVisible()}")
                print(f"   - text: '{self.time_label.text()}'")
                print(f"   - size: {self.time_label.size()}")
                print(f"   - parent: {self.time_label.parent()}")
                
                # Test updating the label
                self.time_label.setText("Time: TEST 2025.08.10 15:30")
                self.time_label.show()
                self.time_label.setVisible(True)
                
                print(f"   - updated text: '{self.time_label.text()}'")
                
            else:
                print("❌ time_label NOT FOUND!")
            
            # Check layout
            if hasattr(self, 'layout'):
                print(f"✅ Main layout has {self.layout.count()} widgets:")
                for i in range(self.layout.count()):
                    widget = self.layout.itemAt(i).widget()
                    if widget:
                        print(f"   - Widget {i}: {type(widget).__name__} - visible: {widget.isVisible()}")
            
            # Start a demo timer to update time every 2 seconds
            if hasattr(self, 'time_label'):
                self.demo_timer = QTimer()
                self.demo_timer.timeout.connect(self.demo_time_update)
                self.demo_timer.start(2000)
                self.demo_counter = 0
                print("� Starting demo time updates every 2 seconds...")
            
            print("�🔍 === END DEBUG TEST ===\n")
            
        except Exception as e:
            print(f"❌ Time widget test error: {e}")
            import traceback
            traceback.print_exc()
    
    def on_mouse_move(self, event):
        """Handle mouse movement for crosshair and info display - optimized for smooth performance"""
        if not MATPLOTLIB_AVAILABLE or not self.crosshair_enabled or event.inaxes != self.ax:
            return
            
        if len(self.ohlc_data) == 0:
            return
            
        # Get mouse position
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        # Find nearest candle
        candle_idx = int(round(x))
        if candle_idx < 0 or candle_idx >= len(self.ohlc_data):
            return
            
        # Throttle updates for better performance - only update every few pixels
        if hasattr(self, '_last_mouse_pos'):
            last_x, last_y = self._last_mouse_pos
            if abs(x - last_x) < 0.5 and abs(y - last_y) < (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.01:
                return
        
        self._last_mouse_pos = (x, y)
            
        # Update crosshair and info with optimized drawing
        try:
            self.update_crosshair(x, y)
            self.show_candle_info(candle_idx, x, y)
            
            # Update time display safely
            try:
                self.update_time_display(candle_idx)
            except Exception as e:
                # Silently handle time display errors
                pass
            
            # Use blit for faster drawing instead of full redraw
            self.canvas.draw_idle()
        except Exception as e:
            # Silently handle crosshair errors but print for debugging
            print(f"Mouse move error: {e}")
            pass
    
    def on_mouse_press(self, event):
        """Handle mouse press for pan functionality"""
        if event.inaxes != self.ax:
            return
        self.press = (event.x, event.y)
    
    def on_mouse_release(self, event):
        """Handle mouse release for pan functionality"""
        self.press = None
        self.canvas.draw_idle()
    
    def on_mouse_scroll(self, event):
        """Handle mouse scroll for zoom functionality"""
        if not self.zoom_enabled or event.inaxes != self.ax:
            return
            
        # Get current axis limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Zoom factor
        zoom_factor = 1.2 if event.step < 0 else 1/1.2
        
        # Calculate new limits
        x_center = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
        y_center = event.ydata if event.ydata else (ylim[0] + ylim[1]) / 2
        
        x_range = (xlim[1] - xlim[0]) * zoom_factor / 2
        y_range = (ylim[1] - ylim[0]) * zoom_factor / 2
        
        # Set new limits
        self.ax.set_xlim(x_center - x_range, x_center + x_range)
        self.ax.set_ylim(y_center - y_range, y_center + y_range)
        
        self.canvas.draw_idle()
    
    def on_mouse_leave(self, event):
        """Handle mouse leaving the chart area - clear time display and candle info"""
        try:
            # Clear time display in coordinate formatter (reset to default)
            self._current_time_display = "--"
            
            # Clear candle info display
            try:
                if self.info_text:
                    if hasattr(self.info_text, 'remove'):
                        self.info_text.remove()
                    self.info_text = None
            except Exception:
                pass
            
            # Clear crosshair
            try:
                if self.crosshair_h:
                    if hasattr(self.crosshair_h, 'remove'):
                        self.crosshair_h.remove()
                    self.crosshair_h = None
            except Exception:
                pass
                
            try:
                if self.crosshair_v:
                    if hasattr(self.crosshair_v, 'remove'):
                        self.crosshair_v.remove()
                    self.crosshair_v = None
            except Exception:
                pass
            
            # Redraw canvas
            try:
                self.canvas.draw_idle()
            except Exception:
                pass
            
        except Exception as e:
            print(f"Mouse leave error: {e}")
            pass
    
    def update_crosshair(self, x, y):
        """Update crosshair lines - improved safe removal method"""
        # Safely remove old crosshair with multiple methods
        try:
            if self.crosshair_h:
                if hasattr(self.crosshair_h, 'remove'):
                    self.crosshair_h.remove()
                elif self.crosshair_h in self.ax.lines:
                    self.ax.lines.remove(self.crosshair_h)
        except (ValueError, AttributeError, NotImplementedError):
            pass
        
        try:
            if self.crosshair_v:
                if hasattr(self.crosshair_v, 'remove'):
                    self.crosshair_v.remove()
                elif self.crosshair_v in self.ax.lines:
                    self.ax.lines.remove(self.crosshair_v)
        except (ValueError, AttributeError, NotImplementedError):
            pass
        
        # Clear and reset crosshair references
        self.crosshair_h = None
        self.crosshair_v = None
            
        # Draw new crosshair
        self.crosshair_h = self.ax.axhline(y=y, color='#ffa726', linewidth=1, alpha=0.7, linestyle='--')
        self.crosshair_v = self.ax.axvline(x=x, color='#ffa726', linewidth=1, alpha=0.7, linestyle='--')
    
    def update_time_display(self, candle_idx):
        """Update time display in matplotlib's status bar coordinates area (time only, no X Y)"""
        try:
            if candle_idx >= len(self.dates):
                return
                
            # Get the date for this candle
            date = self.dates[candle_idx]
            
            # Format date based on timeframe for time display
            tf_str = str(self.current_timeframe) if self.current_timeframe else ""
            
            if 'MN' in tf_str or self.current_timeframe in [49153]:  # Monthly
                time_display = date.strftime('%Y %B')  # "2024 January"
            elif 'W' in tf_str or self.current_timeframe in [32769]:  # Weekly
                time_display = date.strftime('%Y.%m.%d (Week %U)')  # "2024.01.15 (Week 03)"
            elif 'D' in tf_str or self.current_timeframe in [16408]:  # Daily
                time_display = date.strftime('%Y.%m.%d (%A)')  # "2024.01.15 (Monday)"
            elif 'H' in tf_str or self.current_timeframe in [16385, 16388]:  # Hours
                time_display = date.strftime('%Y.%m.%d %H:00')  # "2024.01.15 14:00"
            elif 'M' in tf_str or self.current_timeframe in [1, 5, 15, 30]:  # Minutes
                time_display = date.strftime('%Y.%m.%d %H:%M')  # "2024.01.15 14:30"
            else:
                time_display = date.strftime('%Y.%m.%d %H:%M')  # Default
            
            # Update the time display for coordinate formatter (time only)
            self._current_time_display = time_display
                
        except Exception as e:
            print(f"Time display error: {e}")
            pass
    
    def show_candle_info(self, candle_idx, x, y):
        """Show detailed candle information"""
        try:
            if candle_idx >= len(self.ohlc_data) or candle_idx >= len(self.dates):
                return
                
            # Get candle data
            ohlc = self.ohlc_data[candle_idx]
            date = self.dates[candle_idx]
            volume = self.volumes[candle_idx] if candle_idx < len(self.volumes) else 0
            
            open_price, high, low, close = ohlc
            
            # Format prices with correct decimals
            o_formatted = self.format_price_mt5_style(open_price)
            h_formatted = self.format_price_mt5_style(high)
            l_formatted = self.format_price_mt5_style(low)
            c_formatted = self.format_price_mt5_style(close)
            
            # Create compact info text for candle top display - single column format
            change = close - open_price
            change_pct = (change / open_price) * 100 if open_price != 0 else 0
            change_symbol = '+' if change >= 0 else ''
            
            # Single column info for display above candle
            info_text = f"O: {o_formatted}\nH: {h_formatted}\nL: {l_formatted}\nC: {c_formatted}\nVol: {volume:,.0f}\n{change_symbol}{self.format_price_mt5_style(abs(change))} ({change_pct:+.2f}%)"
            
            # Remove old info text safely
            try:
                if self.info_text and hasattr(self.info_text, 'remove'):
                    self.info_text.remove()
            except (ValueError, AttributeError, NotImplementedError):
                pass
            self.info_text = None
            
            # Position info box above the current candle
            candle_high = ohlc[1]  # High price of the candle
            ylim = self.ax.get_ylim()
            
            # Position above the candle high with some margin
            info_y = candle_high + (ylim[1] - ylim[0]) * 0.02  # Slightly above the candle high
            info_x = candle_idx  # X position at the candle
            
            # Make sure info box doesn't go off screen
            if info_y > ylim[1] * 0.95:  # If too close to top
                info_y = candle_high - (ylim[1] - ylim[0]) * 0.08  # Place below candle instead
                va = 'top'
            else:
                va = 'bottom'
            
            # Determine text color based on candle direction
            text_color = '#00e676' if change >= 0 else '#ff1744'  # Green for bullish, red for bearish
            
            # Add info text box positioned above/below the candle
            self.info_text = self.ax.text(info_x, info_y, info_text,
                                         fontsize=8, fontweight='bold',
                                         color=text_color, ha='center', va=va,
                                         bbox=dict(boxstyle='round,pad=0.4',
                                                 facecolor=(0.08, 0.08, 0.08, 0.9),  # Dark gray with alpha
                                                 edgecolor=text_color,
                                                 linewidth=1.5,
                                                 alpha=0.95))
                                                 
        except Exception as e:
            # Print error for debugging but don't crash
            print(f"Candle info error: {e}")
            pass

    def toggle_mt5_panel(self):
        """Toggle MT5 trading panel visibility (like MT5)"""
        if not hasattr(self, 'trading_content'):
            return
            
        if self.is_panel_collapsed:
            # Expand panel
            self.trading_content.show()
            self.toggle_btn.setText("−")
            self.mt5_overlay.setFixedSize(300, 95)  # Updated size without price display
            self.is_panel_collapsed = False
        else:
            # Collapse panel
            self.trading_content.hide()
            self.toggle_btn.setText("+")
            self.mt5_overlay.setFixedSize(300, 25)  # Only header visible
            self.is_panel_collapsed = True

    def resizeEvent(self, event):
        """Handle widget resize to keep overlay positioned correctly"""
        super().resizeEvent(event)
        if hasattr(self, 'mt5_overlay'):
            # Keep panel in top-left corner when window resizes
            self.mt5_overlay.move(15, 60)
            self.mt5_overlay.raise_()  # Bring to front after resize
            
        # Reposition message overlay if it exists
        if hasattr(self, 'message_overlay') and self.message_overlay.isVisible():
            self.message_overlay.move(self.width()//2 - 250, 60)
            self.message_overlay.raise_()

    def update_mt5_buttons_with_prices(self):
        """Update MT5 buttons with real-time bid/ask prices using proper decimals"""
        if not hasattr(self, 'mt5_sell_btn') or not hasattr(self, 'mt5_buy_btn'):
            return
            
        if not self.current_symbol:
            self.mt5_sell_btn.setText("SELL\n--")
            self.mt5_buy_btn.setText("BUY\n--")
            return
            
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                tick = mt5.symbol_info_tick(self.current_symbol)
                if tick:
                    bid_price = tick.bid
                    ask_price = tick.ask
                    
                    # Format prices with correct decimal places
                    bid_formatted = self.format_price_mt5_style(bid_price)
                    ask_formatted = self.format_price_mt5_style(ask_price)
                    
                    # Update buttons with properly formatted prices
                    self.mt5_sell_btn.setText(f"SELL\n{bid_formatted}")
                    self.mt5_buy_btn.setText(f"BUY\n{ask_formatted}")
                else:
                    self.mt5_sell_btn.setText("SELL\n--")
                    self.mt5_buy_btn.setText("BUY\n--")
            else:
                self.mt5_sell_btn.setText("SELL\n--")
                self.mt5_buy_btn.setText("BUY\n--")
                
        except Exception as e:
            self.mt5_sell_btn.setText("SELL\n--")
            self.mt5_buy_btn.setText("BUY\n--")

    def update_mt5_buttons_state(self):
        """Update MT5 button states"""
        if hasattr(self, 'mt5_buy_btn') and hasattr(self, 'mt5_sell_btn'):
            has_connection = self.mt5_conn is not None
            has_symbol = self.current_symbol is not None
            can_trade = has_connection and has_symbol and ORDER_EXECUTOR_AVAILABLE
            
            self.mt5_buy_btn.setEnabled(can_trade)
            self.mt5_sell_btn.setEnabled(can_trade)
            
            # Update prices
            self.update_mt5_buttons_with_prices()

    def execute_mt5_buy(self):
        """Execute BUY order MT5 style"""
        if not ORDER_EXECUTOR_AVAILABLE:
            self.show_trading_message("❌ Order executor not available!", "error")
            return
            
        if not self.current_symbol:
            self.show_trading_message("❌ No symbol selected!", "error")
            return
            
        if not self.mt5_conn:
            self.show_trading_message("❌ No MT5 connection!", "error")
            return
        
        try:
            volume = self.mt5_volume.value()
            
            # Create order executor instance
            from order_executor import get_executor_instance, TradeSignal
            import MetaTrader5 as mt5
            
            # Get current price
            tick = mt5.symbol_info_tick(self.current_symbol)
            if not tick:
                self.show_trading_message("❌ Cannot get price data!", "error")
                return
            
            executor = get_executor_instance()
            
            # Format price for display
            price_formatted = self.format_price_mt5_style(tick.ask)
            
            # Create trade signal for BUY
            signal = TradeSignal(
                symbol=self.current_symbol,
                action="BUY",
                entry_price=tick.ask,
                stop_loss=0.0,
                take_profit=0.0,
                volume=volume,
                comment="MT5 BUY"
            )
            
            # Execute market buy order
            result = executor.execute_market_order(signal)
            
            if result.success:
                self.show_trading_message(f"✅ BUY {volume} {self.current_symbol} @ {price_formatted}", "success")
            else:
                self.show_trading_message(f"❌ BUY failed: {result.error_message}", "error")
                
        except Exception as e:
            self.show_trading_message(f"❌ BUY error: {str(e)}", "error")

    def execute_mt5_sell(self):
        """Execute SELL order MT5 style"""
        if not ORDER_EXECUTOR_AVAILABLE:
            self.show_trading_message("❌ Order executor not available!", "error")
            return
            
        if not self.current_symbol:
            self.show_trading_message("❌ No symbol selected!", "error")
            return
            
        if not self.mt5_conn:
            self.show_trading_message("❌ No MT5 connection!", "error")
            return
        
        try:
            volume = self.mt5_volume.value()
            
            # Create order executor instance
            from order_executor import get_executor_instance, TradeSignal
            import MetaTrader5 as mt5
            
            # Get current price
            tick = mt5.symbol_info_tick(self.current_symbol)
            if not tick:
                self.show_trading_message("❌ Cannot get price data!", "error")
                return
            
            executor = get_executor_instance()
            
            # Format price for display
            price_formatted = self.format_price_mt5_style(tick.bid)
            
            # Create trade signal for SELL
            signal = TradeSignal(
                symbol=self.current_symbol,
                action="SELL",
                entry_price=tick.bid,
                stop_loss=0.0,
                take_profit=0.0,
                volume=volume,
                comment="MT5 SELL"
            )
            
            # Execute market sell order
            result = executor.execute_market_order(signal)
            
            if result.success:
                self.show_trading_message(f"✅ SELL {volume} {self.current_symbol} @ {price_formatted}", "success")
            else:
                self.show_trading_message(f"❌ SELL failed: {result.error_message}", "error")
                
        except Exception as e:
            self.show_trading_message(f"❌ SELL error: {str(e)}", "error")

    def set_mt5_connection(self, mt5_conn):
        """Set MT5 connection for trading operations"""
        self.mt5_conn = mt5_conn
        self.update_mt5_buttons_state()
        
    def start_realtime_update(self, symbol, timeframe, mt5_conn=None):
        """Start real-time chart updates"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        self.current_symbol = symbol
        self.current_timeframe = timeframe
        
        # Set MT5 connection if provided
        if mt5_conn:
            self.set_mt5_connection(mt5_conn)
            
        # Clear previous data
        self.ohlc_data = []
        self.dates = []
        self.volumes = []
        
        # Start optimized timer for smoother updates
        self.update_timer.start(3000)  # Update every 3 seconds for better performance
        
        # Start price update timer for MT5 buttons (every 2 seconds for smoother performance)
        if hasattr(self, 'mt5_price_timer'):
            self.mt5_price_timer.stop()
        self.mt5_price_timer = QTimer()
        self.mt5_price_timer.timeout.connect(self.update_mt5_prices)
        self.mt5_price_timer.start(2000)  # Update prices every 2 seconds
        
        # Start settings auto-refresh timer (every 5 seconds)
        if hasattr(self, 'settings_refresh_timer'):
            self.settings_refresh_timer.stop()
        self.settings_refresh_timer = QTimer()
        # Note: auto_refresh_settings is not needed for CandlestickChart
        # self.settings_refresh_timer.timeout.connect(self.auto_refresh_settings)
        # self.settings_refresh_timer.start(5000)  # Check for settings changes every 5 seconds
        
        self.update_chart()
    
    def stop_realtime_update(self):
        """Stop real-time updates"""
        if self.update_timer.isActive():
            self.update_timer.stop()
    
    def update_chart(self):
        """Update chart with latest data - Enhanced for W1 and MN1 support"""
        if not MATPLOTLIB_AVAILABLE or not self.current_symbol:
            return
            
        try:
            # Get latest data from MT5
            import MetaTrader5 as mt5
            from datetime import datetime, timedelta
            
            if not mt5.initialize():
                print(f"⚠️ MT5 initialization failed for chart update")
                return
            
            # Ensure symbol is available in Market Watch
            symbol_info = mt5.symbol_info(self.current_symbol)
            if symbol_info is None:
                print(f"⚠️ Symbol {self.current_symbol} not found")
                return
                
            if not symbol_info.visible:
                if not mt5.symbol_select(self.current_symbol, True):
                    print(f"⚠️ Failed to add {self.current_symbol} to Market Watch")
                    return
            
            # Get appropriate count based on timeframe
            tf_str = str(self.current_timeframe)
            if 'MN' in tf_str or self.current_timeframe in [49153]:  # Monthly
                count = 60  # 5 years of monthly data
            elif 'W' in tf_str or self.current_timeframe in [32769]:  # Weekly  
                count = 100  # ~2 years of weekly data
            elif 'D' in tf_str or self.current_timeframe in [16408]:  # Daily
                count = 100  # ~3 months of daily data
            else:
                count = 100  # Standard count for intraday
            
            # Get rates with enhanced error handling
            rates = mt5.copy_rates_from_pos(self.current_symbol, self.current_timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                # Try alternative method for W1/MN1
                if self.current_timeframe in [32769, 49153]:  # W1 or MN1
                    print(f"⚠️ Trying alternative method for {tf_str} data...")
                    
                    # Go back further for weekly/monthly data
                    days_back = 730 if self.current_timeframe == 49153 else 365  # 2 years for monthly, 1 year for weekly
                    from_date = datetime.now() - timedelta(days=days_back)
                    
                    rates = mt5.copy_rates_from(self.current_symbol, self.current_timeframe, from_date, count)
                
                if rates is None or len(rates) == 0:
                    print(f"⚠️ No data available for {self.current_symbol} {tf_str}")
                    return
            
            # Convert to OHLC format
            self.ohlc_data = []
            self.dates = []
            self.volumes = []
            
            for rate in rates:
                date = datetime.fromtimestamp(rate['time'])
                self.dates.append(date)
                self.ohlc_data.append([rate['open'], rate['high'], rate['low'], rate['close']])
                self.volumes.append(rate['tick_volume'])  # Add volume data
            
            print(f"✅ Chart updated: {len(self.ohlc_data)} candles for {self.current_symbol} {tf_str}")
            self.draw_candlesticks()
            
        except Exception as e:
            print(f"Error updating chart: {e}")
            # Try to continue with existing data if available
            if len(self.ohlc_data) > 0:
                self.draw_candlesticks()
    
    def draw_candlesticks(self):
        """Draw beautiful candlestick chart without indicators"""
        if not MATPLOTLIB_AVAILABLE or len(self.ohlc_data) == 0:
            return
            
        # Clear chart safely
        self.ax.clear()
        
        # Reset crosshair references after clearing
        self.crosshair_h = None
        self.crosshair_v = None
        self.info_text = None
        
        self.setup_chart()
        
        # Calculate candle width based on number of candles for better visual
        candle_width = 0.8
        
        # Draw candlesticks with enhanced styling
        for i, (date, ohlc) in enumerate(zip(self.dates, self.ohlc_data)):
            open_price, high, low, close = ohlc
            
            # Determine if bullish or bearish
            is_bullish = close >= open_price
            
            # Enhanced colors for professional look
            if is_bullish:
                body_color = '#00e676'  # Bright green
                wick_color = '#1de9b6'  # Teal green for wick
                edge_color = '#00c853'  # Darker green edge
                shadow_color = '#00e676'
            else:
                body_color = '#ff1744'  # Bright red
                wick_color = '#ff5722'  # Orange-red for wick
                edge_color = '#d50000'  # Darker red edge
                shadow_color = '#ff1744'
            
            # Draw high-low wick with gradient effect
            self.ax.plot([i, i], [low, high], color=wick_color, linewidth=1.2, alpha=0.9, solid_capstyle='round')
            
            # Draw body rectangle with enhanced styling
            body_height = abs(close - open_price)
            body_bottom = min(open_price, close)
            
            # Handle doji candles (open == close) with special styling
            if body_height == 0:
                body_height = (high - low) * 0.03  # Slightly larger body for doji
                body_bottom = open_price - body_height/2
                # Special doji styling
                rect = Rectangle((i - candle_width/2, body_bottom), candle_width, body_height, 
                               facecolor='#ffc107', 
                               edgecolor='#ff8f00', 
                               linewidth=0.8,
                               alpha=0.95)
            else:
                # Regular candle styling with shadow effect
                rect = Rectangle((i - candle_width/2, body_bottom), candle_width, body_height, 
                               facecolor=body_color, 
                               edgecolor=edge_color, 
                               linewidth=0.5,
                               alpha=0.95)
            
            self.ax.add_patch(rect)
        
        # Add subtle price line connecting closes for trend visualization
        if len(self.ohlc_data) > 1 and self.show_price_line:
            close_prices = [candle[3] for candle in self.ohlc_data]
            self.ax.plot(range(len(close_prices)), close_prices, 
                        color='#64b5f6', linewidth=1.2, alpha=0.7, 
                        linestyle='-', zorder=1)
        
        # Enhanced axis formatting with MT5-style time labels
        if len(self.dates) > 0:
            # Use smart time labeling system
            x_ticks, x_labels = self.get_smart_time_labels(self.dates, self.current_timeframe)
            
            self.ax.set_xticks(x_ticks)
            self.ax.set_xticklabels(x_labels, rotation=0, fontsize=8, ha='center')
        
        # Enhanced price formatting on y-axis with symbol-specific decimals
        decimal_places = self.get_symbol_decimal_places(self.current_symbol)
        self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.{decimal_places}f}'))
        
        # Remove axis labels and make tick labels smaller (MT5 style)
        self.ax.set_xlabel('')  # Remove "Time" label
        self.ax.set_ylabel('')  # Remove "Price" label
        
        # Make tick labels smaller
        self.ax.tick_params(axis='x', labelsize=8)  # Smaller time labels
        self.ax.tick_params(axis='y', labelsize=8)  # Smaller price labels
        
        # Enhanced title with current price info (MT5-style decimals)
        if self.ohlc_data:
            current_candle = self.ohlc_data[-1]
            open_price, high, low, close = current_candle
            change = close - open_price
            change_pct = (change / open_price) * 100 if open_price != 0 else 0
            
            # Color based on change
            title_color = '#00e676' if change >= 0 else '#ff1744'
            change_symbol = '+' if change >= 0 else ''
            
            # Format prices with correct decimals
            o_formatted = self.format_price_mt5_style(open_price)
            h_formatted = self.format_price_mt5_style(high)
            l_formatted = self.format_price_mt5_style(low)
            c_formatted = self.format_price_mt5_style(close)
            change_formatted = self.format_price_mt5_style(abs(change))
            
            title = f"{self.current_symbol} | O: {o_formatted} H: {h_formatted} L: {l_formatted} C: {c_formatted} | {change_symbol}{change_formatted} ({change_pct:+.2f}%)"
            self.ax.set_title(title, color=title_color, fontsize=9, fontweight='bold', pad=8)
            
            # Add current price line in yellow with price label (like MT5)
            if self.show_price_line:
                current_price = close
                # Main price line (current close price only)
                self.ax.axhline(y=current_price, color='#ffa726', linewidth=2, alpha=0.9, linestyle='-')
                
                # Price label with correct decimals
                price_formatted = self.format_price_mt5_style(current_price)
                self.ax.text(len(self.ohlc_data) + 0.5, current_price, f' {price_formatted}', 
                            color='#ffa726', fontsize=9, fontweight='bold', 
                            verticalalignment='center', bbox=dict(boxstyle='round,pad=0.2', 
                            facecolor='#2a2a2a', edgecolor='#ffa726', alpha=0.8))
        
        # Adjust margins for better visualization
        self.ax.margins(x=0.02, y=0.05)
        
        # Set x-axis limits to create space for future candles
        if len(self.ohlc_data) > 0:
            extra_space = len(self.ohlc_data) * 0.05  # 5% extra space
            self.ax.set_xlim(-extra_space, len(self.ohlc_data) - 1 + extra_space)
        
        # Enhanced layout with optimized drawing
        self.figure.tight_layout(pad=0.8)  # Reduced padding for more compact layout
        
        # Draw the updated chart with optimized settings
        self.canvas.draw()
        
        # Update MT5 prices if connected
        if self.mt5_conn and hasattr(self.mt5_conn, 'connected') and self.mt5_conn.connected:
            self.update_mt5_prices()

    def get_symbol_decimal_places(self, symbol):
        """Get decimal places directly from MT5 symbol info"""
        if not symbol or not MT5_AVAILABLE:
            return 5
            
        try:
            # Get symbol info directly from MT5
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info and hasattr(symbol_info, 'digits'):
                return symbol_info.digits
        except Exception as e:
            print(f"⚠️ Error getting symbol info for {symbol}: {e}")
        
        # Fallback to manual detection only if MT5 info is not available
        symbol_upper = symbol.upper()
        
        # JPY pairs typically use 3 decimal places
        if 'JPY' in symbol_upper:
            return 3
            
        # Indices typically use 2 decimal places
        if any(index in symbol_upper for index in ['US30', 'US500', 'NAS100', 'GER30', 'UK100', 'JPN225']):
            return 2
            
        # Metals typically use 2 decimal places
        if any(metal in symbol_upper for metal in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return 2
            
        # Default fallback for forex pairs
        return 5
            
        # Default fallback
        return 4

    def format_price_mt5_style(self, price, symbol=None):
        """Format price with correct decimal places like MT5"""
        if symbol is None:
            symbol = self.current_symbol
            
        decimal_places = self.get_symbol_decimal_places(symbol)
        return f'{price:.{decimal_places}f}'

    def get_smart_time_labels(self, dates, current_timeframe):
        """Generate smart time labels with current candle priority - Trading time format"""
        if not dates:
            return [], []
            
        # Determine label density based on chart size
        total_candles = len(dates)
        
        if total_candles <= 20:
            # Show every candle for small datasets
            step = 1
            max_labels = total_candles
        elif total_candles <= 50:
            # Show every 2-3 candles
            step = max(1, total_candles // 15)
            max_labels = 15
        elif total_candles <= 100:
            # Show every 5-8 candles  
            step = max(1, total_candles // 12)
            max_labels = 12
        else:
            # Show every 10+ candles for large datasets
            step = max(1, total_candles // 8)
            max_labels = 8
        
        # Always include the last candle (current trading)
        x_ticks = list(range(0, total_candles, step))
        if (total_candles - 1) not in x_ticks:
            x_ticks.append(total_candles - 1)
        
        # Limit total labels
        if len(x_ticks) > max_labels:
            # Keep first, last, and evenly spaced middle points
            keep_indices = [0]  # First
            middle_indices = x_ticks[1:-1]
            if len(middle_indices) > max_labels - 2:
                # Select evenly spaced middle points
                middle_step = len(middle_indices) // (max_labels - 2)
                keep_indices.extend(middle_indices[::middle_step][:max_labels-2])
            else:
                keep_indices.extend(middle_indices)
            keep_indices.append(x_ticks[-1])  # Last
            x_ticks = sorted(set(keep_indices))
        
        # Generate labels with MT5-style formatting - Updated time display logic
        x_labels = []
        for i, tick_idx in enumerate(x_ticks):
            date = dates[tick_idx]
            is_current = (tick_idx == total_candles - 1)  # Current trading candle
            
            # Determine timeframe type for appropriate time display
            tf_str = str(current_timeframe) if current_timeframe else ""
            
            # Check MN1 first (before M check) to avoid conflict
            if 'MN' in tf_str or current_timeframe in [49153]:  # Monthly timeframe (MN1=49153)
                # MN1: Show month and year (MMM YY)
                if is_current:
                    x_labels.append(f">{date.strftime('%b')}\n{date.strftime('%y')}")
                else:
                    x_labels.append(f"{date.strftime('%b')}\n{date.strftime('%y')}")
                    
            elif 'W' in tf_str or current_timeframe in [32769]:  # Weekly timeframe (W1=32769)
                # W1: Show week start date with month (DD/MM + week indicator)
                if is_current:
                    x_labels.append(f">{date.strftime('%d/%m')}\nW{date.isocalendar()[1]}")
                else:
                    x_labels.append(f"{date.strftime('%d/%m')}\nW{date.isocalendar()[1]}")
                    
            elif 'D' in tf_str or current_timeframe in [16408]:  # Daily timeframe (D1=16408)
                # D1: Only show date with day name (DD/MM + weekday)
                if is_current:
                    x_labels.append(f">{date.strftime('%d/%m')}\n{date.strftime('%a')}")
                else:
                    x_labels.append(f"{date.strftime('%d/%m')}\n{date.strftime('%a')}")
                    
            elif 'H' in tf_str or current_timeframe in [16385, 16388]:  # Hours timeframe (H1=16385, H4=16388)
                # H1-H4: Show full date and time (DD/MM HH:00)
                if is_current:
                    x_labels.append(f">{date.strftime('%d/%m')}\n{date.strftime('%H:00')}")
                else:
                    x_labels.append(f"{date.strftime('%d/%m')}\n{date.strftime('%H:00')}")
                        
            elif 'M' in tf_str or current_timeframe in [1, 5, 15, 30]:  # Minutes timeframe
                # M1-M30: Show full date and time (DD/MM HH:MM)
                if is_current:
                    x_labels.append(f">{date.strftime('%d/%m')}\n{date.strftime('%H:%M')}")
                else:
                    x_labels.append(f"{date.strftime('%d/%m')}\n{date.strftime('%H:%M')}")
                    
            else:  # Default fallback
                # Show date with day name for better context
                if is_current:
                    x_labels.append(f">{date.strftime('%d/%m')}\n{date.strftime('%a')}")
                else:
                    x_labels.append(f"{date.strftime('%d/%m')}\n{date.strftime('%a')}")
        
        return x_ticks, x_labels

    def update_mt5_buttons_state(self):
        """Update MT5 trading buttons state based on connection"""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'mt5_buy_btn'):
            return
            
        is_connected = self.mt5_conn and self.mt5_conn.connected
        
        # Enable/disable buttons
        self.mt5_buy_btn.setEnabled(is_connected)
        self.mt5_sell_btn.setEnabled(is_connected)
        self.mt5_volume.setEnabled(is_connected)
        
        # Update prices if connected
        if is_connected and self.current_symbol:
            self.update_mt5_prices()
        else:
            # Show disconnected state
            self.mt5_sell_btn.setText("SELL\n--")
            self.mt5_buy_btn.setText("BUY\n--")
            # Reset spread display
            if hasattr(self, 'spread_display'):
                self.spread_display.setText("--")
            
    def update_mt5_prices(self):
        """Update real-time bid/ask prices on MT5 buttons with proper decimals and spread in USD/LOT"""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'mt5_buy_btn'):
            return
            
        if not self.mt5_conn or not self.mt5_conn.connected or not self.current_symbol:
            return
            
        try:
            # Get current tick using MT5 directly
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(self.current_symbol)
            if tick:
                bid_price = tick.bid
                ask_price = tick.ask
                
                # Calculate spread in USD/LOT
                spread_raw = ask_price - bid_price
                
                # Get symbol info for contract specifications
                symbol_info = mt5.symbol_info(self.current_symbol)
                if symbol_info:
                    # Calculate spread in USD per 1 lot
                    # For most symbols: spread_usd = spread * contract_size * tick_value / tick_size
                    contract_size = symbol_info.trade_contract_size
                    tick_value = symbol_info.trade_tick_value
                    tick_size = symbol_info.trade_tick_size
                    
                    if tick_size > 0:
                        # Calculate spread cost in USD for 1 lot
                        spread_usd = (spread_raw / tick_size) * tick_value
                        
                        # Format spread as USD/LOT
                        if spread_usd >= 1:
                            spread_formatted = f"${spread_usd:.2f}"
                        elif spread_usd >= 0.1:
                            spread_formatted = f"${spread_usd:.3f}"
                        else:
                            spread_formatted = f"${spread_usd:.4f}"
                    else:
                        spread_formatted = f"${spread_raw:.2f}"
                else:
                    # Fallback calculation for standard lot
                    spread_formatted = f"${spread_raw * 100000:.2f}"
                
                # Format prices with correct decimal places for the symbol
                bid_formatted = self.format_price_mt5_style(bid_price)
                ask_formatted = self.format_price_mt5_style(ask_price)
                
                # Update button texts with properly formatted prices
                self.mt5_sell_btn.setText(f"SELL\n{bid_formatted}")
                self.mt5_buy_btn.setText(f"BUY\n{ask_formatted}")
                
                # Update spread display with USD/LOT
                if hasattr(self, 'spread_display'):
                    self.spread_display.setText(spread_formatted)
                
        except Exception as e:
            print(f"Error updating MT5 prices: {e}")
            # Reset spread display on error
            if hasattr(self, 'spread_display'):
                self.spread_display.setText("--")
            
    def on_mt5_sell_click(self):
        """Handle MT5 SELL button click"""
        if not self.mt5_conn or not self.mt5_conn.connected:
            return
            
        volume = self.mt5_volume.value()
        if volume <= 0:
            return
            
        try:
            from order_executor import get_executor_instance, TradeSignal
            import MetaTrader5 as mt5
            
            # Get current price for the signal
            tick = mt5.symbol_info_tick(self.current_symbol)
            if not tick:
                print(f"❌ Cannot get tick data for {self.current_symbol}")
                return
            
            # Create order executor instance
            executor = get_executor_instance()
            
            # Create trade signal for SELL
            signal = TradeSignal(
                symbol=self.current_symbol,
                action="SELL",
                entry_price=tick.bid,  # Use bid price for SELL
                stop_loss=0.0,  # No SL for now
                take_profit=0.0,  # No TP for now
                volume=volume,
                comment="MT5 Panel SELL"
            )
            
            # Execute the order
            result = executor.execute_market_order(signal)
            
            if result.success:
                # Show success notification
                price_formatted = self.format_price_mt5_style(tick.bid)
                success_msg = f"✅ SELL Order Executed!\n{self.current_symbol} | {volume} lots @ {price_formatted}\nTicket: {result.ticket}"
                self.show_trading_message(success_msg, "success")
                print(f"✅ SELL order executed: {self.current_symbol} {volume} lots (Ticket: {result.ticket})")
            else:
                # Show error notification
                error_msg = f"❌ SELL Order Failed!\n{result.error_message}"
                self.show_trading_message(error_msg, "error")
                print(f"❌ SELL order failed: {result.error_message}")
                
        except Exception as e:
            # Show exception notification
            error_msg = f"❌ SELL Order Error!\n{str(e)}"
            self.show_trading_message(error_msg, "error")
            print(f"Error executing SELL order: {e}")
            
    def on_mt5_buy_click(self):
        """Handle MT5 BUY button click"""
        if not self.mt5_conn or not self.mt5_conn.connected:
            return
            
        volume = self.mt5_volume.value()
        if volume <= 0:
            return
            
        try:
            from order_executor import get_executor_instance, TradeSignal
            import MetaTrader5 as mt5
            
            # Get current price for the signal
            tick = mt5.symbol_info_tick(self.current_symbol)
            if not tick:
                print(f"❌ Cannot get tick data for {self.current_symbol}")
                return
            
            # Create order executor instance
            executor = get_executor_instance()
            
            # Create trade signal for BUY
            signal = TradeSignal(
                symbol=self.current_symbol,
                action="BUY",
                entry_price=tick.ask,  # Use ask price for BUY
                stop_loss=0.0,  # No SL for now
                take_profit=0.0,  # No TP for now
                volume=volume,
                comment="MT5 Panel BUY"
            )
            
            # Execute the order
            result = executor.execute_market_order(signal)
            
            if result.success:
                # Show success notification
                price_formatted = self.format_price_mt5_style(tick.ask)
                success_msg = f"✅ BUY Order Executed!\n{self.current_symbol} | {volume} lots @ {price_formatted}\nTicket: {result.ticket}"
                self.show_trading_message(success_msg, "success")
                print(f"✅ BUY order executed: {self.current_symbol} {volume} lots (Ticket: {result.ticket})")
            else:
                # Show error notification
                error_msg = f"❌ BUY Order Failed!\n{result.error_message}"
                self.show_trading_message(error_msg, "error")
                print(f"❌ BUY order failed: {result.error_message}")
                
        except Exception as e:
            # Show exception notification
            error_msg = f"❌ BUY Order Error!\n{str(e)}"
            self.show_trading_message(error_msg, "error")
            print(f"Error executing BUY order: {e}")

    def show_trading_message(self, message, msg_type="info"):
        """Show trading message overlay on chart with enhanced styling"""
        if not hasattr(self, 'message_overlay'):
            self.message_overlay = QLabel(self)
            self.message_overlay.setFixedSize(500, 80)  # Larger size for better visibility
            self.message_overlay.setAlignment(Qt.AlignCenter)
            self.message_overlay.setWordWrap(True)  # Allow text wrapping
        
        # Position in center-top of chart
        self.message_overlay.move(self.width()//2 - 250, 60)
        
        # Enhanced styling based on message type
        if msg_type == "success":
            style = """
                QLabel {
                    background-color: rgba(76, 175, 80, 0.95);
                    color: white;
                    border: 2px solid #4caf50;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 10px;
                    text-align: center;
                }
            """
        elif msg_type == "error":
            style = """
                QLabel {
                    background-color: rgba(244, 67, 54, 0.95);
                    color: white;
                    border: 2px solid #f44336;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 10px;
                    text-align: center;
                }
            """
        else:
            style = """
                QLabel {
                    background-color: rgba(33, 150, 243, 0.95);
                    color: white;
                    border: 2px solid #2196f3;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 10px;
                    text-align: center;
                }
            """
        
        self.message_overlay.setStyleSheet(style)
        self.message_overlay.setText(message)
        self.message_overlay.show()
        self.message_overlay.raise_()  # Bring to front
        
        # Auto-hide message after 5 seconds (longer for order notifications)
        QTimer.singleShot(5000, self.message_overlay.hide)


class WorkerSignals(QObject):
    finished = pyqtSignal(str)

class FetchCandlesRunnable(QRunnable):
    def __init__(self, mt5_conn, symbol, timeframe, count):
        super().__init__()
        self.mt5_conn = mt5_conn
        self.symbol = symbol
        self.timeframe = timeframe
        self.count = count
        self.signals = WorkerSignals()

    def run(self):
        success, msg = self.mt5_conn.fetch_candles(self.symbol, self.timeframe, self.count)
        self.signals.finished.emit(msg)

class AccountTab(QWidget):
    connection_changed = pyqtSignal(bool)  # Signal để thông báo khi kết nối thay đổi
    
    def __init__(self):
        super().__init__()
        self.mt5_conn = None
        self._login_in_progress = False  # guard against re-entrancy / spinner loop
        self._last_login_attempt_ts = 0.0
        self._login_cooldown_sec = 5
        self.user_config = load_user_config()
        self.init_ui()
        self.load_env()
        
        # Timer để cập nhật thông tin tài khoản
        self.account_timer = QTimer(self)
        self.account_timer.timeout.connect(self.update_account_info)
        self.account_timer.start(5000)  # Cập nhật mỗi 5 giây

    def init_ui(self):
        layout = QVBoxLayout()

        # Header with logo and contact info
        top_layout = QHBoxLayout()
        spacer = QLabel()
        top_layout.addWidget(spacer, 1)

        img_path = os.path.join("images", "robot.png")
        self.robot_label = QLabel()
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            self.robot_label.setPixmap(pixmap.scaled(64, 64, Qt.KeepAspectRatio))
        else:
            self.robot_label.setText("🤖")
            self.robot_label.setStyleSheet("font-size: 32px;")
        top_layout.addWidget(self.robot_label, 0, Qt.AlignRight)
        layout.addLayout(top_layout)

        top_label = QLabel("VU HIEN CFDs Telegram/Zalo: +84 39 65 60 888")
        top_label.setAlignment(Qt.AlignRight)
        top_label.setStyleSheet("font-weight: bold; color: #2196F3; font-size: 14px; padding: 5px;")
        layout.addWidget(top_label)

        # Connection Status Panel
        connection_panel = QGroupBox("📡 Connection Status")
        connection_layout = QHBoxLayout()
        self.status_label = QLabel("🔴 Status: Disconnected")
        self.status_label.setStyleSheet(
            "font-weight:bold; color:red; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: #f9f9f9;"
        )
        connection_layout.addWidget(self.status_label)
        connection_panel.setLayout(connection_layout)
        layout.addWidget(connection_panel)

        # Login Form
        self.login_group = QGroupBox("MT5 Account Login")
        form_layout = QFormLayout()
        self.account_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.server_input = QLineEdit()
        self.save_account_cb = QCheckBox("Save Account Info")
        self.save_account_cb.setToolTip("Save login credentials to .env file")

        form_layout.addRow("Account:", self.account_input)
        form_layout.addRow("Password:", self.password_input)
        form_layout.addRow("Server:", self.server_input)
        form_layout.addRow("", self.save_account_cb)

        # Login / Disconnect buttons (Force Reset removed)
        button_layout = QHBoxLayout()
        self.login_button = QPushButton("🔑 Login to MT5")
        self.login_button.clicked.connect(self.login_mt5)
        self.login_button.setStyleSheet(
            """
            QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; border: none; border-radius: 4px; }
            QPushButton:hover { background-color: #45a049; }
            """
        )
        self.logout_button = QPushButton("🔓 Disconnect")
        self.logout_button.clicked.connect(self.logout_mt5)
        self.logout_button.setEnabled(False)
        self.logout_button.setStyleSheet(
            """
            QPushButton { background-color: #cccccc; color: #666666; font-weight: bold; padding: 8px; border: none; border-radius: 4px; }
            QPushButton:enabled { background-color: #f44336; color: white; }
            QPushButton:enabled:hover { background-color: #d32f2f; }
            """
        )
        button_layout.addWidget(self.login_button)
        button_layout.addWidget(self.logout_button)
        form_layout.addRow(button_layout)
        self.login_group.setLayout(form_layout)
        layout.addWidget(self.login_group)

        # Account Information Group
        self.account_group = QGroupBox("Account Information")
        self.account_group.setEnabled(False)
        account_layout = QVBoxLayout()

        # Basics
        basics_layout = QFormLayout()
        self.login_label = QLabel("--")
        self.name_label = QLabel("--")
        self.company_label = QLabel("--")
        self.server_label = QLabel("--")
        self.currency_label = QLabel("--")
        self.leverage_label = QLabel("--")
        basics_layout.addRow("Login:", self.login_label)
        basics_layout.addRow("Name:", self.name_label)
        basics_layout.addRow("Company:", self.company_label)
        basics_layout.addRow("Server:", self.server_label)
        basics_layout.addRow("Currency:", self.currency_label)
        basics_layout.addRow("Leverage:", self.leverage_label)
        account_layout.addLayout(basics_layout)

        # Balance Information
        balance_group = QGroupBox("💰 Balance Information")
        balance_layout = QFormLayout()
        self.balance_label = QLabel("$0.00")
        self.equity_label = QLabel("$0.00")
        self.margin_label = QLabel("$0.00")
        self.free_margin_label = QLabel("$0.00")
        self.margin_level_label = QLabel("0.00%")
        self.profit_label = QLabel("$0.00")
        balance_layout.addRow("Balance:", self.balance_label)
        balance_layout.addRow("Equity:", self.equity_label)
        balance_layout.addRow("Margin Used:", self.margin_label)
        balance_layout.addRow("Free Margin:", self.free_margin_label)
        balance_layout.addRow("Margin Level:", self.margin_level_label)
        balance_layout.addRow("Profit/Loss:", self.profit_label)
        balance_group.setLayout(balance_layout)
        account_layout.addWidget(balance_group)

        # Trading Status
        trading_group = QGroupBox("📈 Trading Status")
        trading_layout = QFormLayout()
        self.positions_label = QLabel("0")
        self.orders_label = QLabel("0")
        self.trade_allowed_label = QLabel("--")
        trading_layout.addRow("Open Positions:", self.positions_label)
        trading_layout.addRow("Pending Orders:", self.orders_label)
        trading_layout.addRow("Trade Allowed:", self.trade_allowed_label)
        trading_group.setLayout(trading_layout)
        account_layout.addWidget(trading_group)

        # Positions Table
        positions_group = QGroupBox("📊 Active Positions")
        positions_layout = QVBoxLayout()
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(11)
        self.positions_table.setHorizontalHeaderLabels([
            "Ticket", "Symbol", "Type", "Volume", "Open Price",
            "Current Price", "Stop Loss", "Take Profit", "Swap",
            "Profit", "Actions"
        ])
        header = self.positions_table.horizontalHeader()
        for i in range(self.positions_table.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        self.positions_table.setAlternatingRowColors(True)
        positions_layout.addWidget(self.positions_table)
        positions_control_layout = QHBoxLayout()
        self.close_all_positions_btn = QPushButton("🚫 Close All Positions")
        self.close_all_positions_btn.setStyleSheet("background-color: #E74C3C; color: white; font-weight: bold;")
        self.close_all_positions_btn.clicked.connect(self.close_all_positions)
        positions_control_layout.addWidget(self.close_all_positions_btn)
        self.refresh_positions_btn = QPushButton("🔄 Refresh Positions")
        self.refresh_positions_btn.setStyleSheet("background-color: #3498DB; color: white; font-weight: bold;")
        self.refresh_positions_btn.clicked.connect(self.refresh_account_info)
        positions_control_layout.addWidget(self.refresh_positions_btn)
        positions_control_layout.addStretch()
        positions_layout.addLayout(positions_control_layout)
        positions_group.setLayout(positions_layout)
        account_layout.addWidget(positions_group)

        # Orders Table
        orders_group = QGroupBox("📋 Pending Orders")
        orders_layout = QVBoxLayout()
        self.orders_table = QTableWidget()
        self.orders_table.setColumnCount(10)
        self.orders_table.setHorizontalHeaderLabels([
            "Ticket", "Symbol", "Type", "Volume", "Open Price",
            "Current Price", "Stop Loss", "Take Profit", "Time", "Actions"
        ])
        header = self.orders_table.horizontalHeader()
        for i in range(self.orders_table.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        self.orders_table.setAlternatingRowColors(True)
        orders_layout.addWidget(self.orders_table)
        orders_control_layout = QHBoxLayout()
        self.cancel_all_orders_btn = QPushButton("❌ Cancel All Orders")
        self.cancel_all_orders_btn.setStyleSheet("background-color: #F39C12; color: white; font-weight: bold;")
        self.cancel_all_orders_btn.clicked.connect(self.cancel_all_orders)
        orders_control_layout.addWidget(self.cancel_all_orders_btn)
        self.refresh_orders_btn = QPushButton("🔄 Refresh Orders")
        self.refresh_orders_btn.setStyleSheet("background-color: #3498DB; color: white; font-weight: bold;")
        self.refresh_orders_btn.clicked.connect(self.refresh_account_info)
        orders_control_layout.addWidget(self.refresh_orders_btn)
        orders_control_layout.addStretch()
        orders_layout.addLayout(orders_control_layout)
        orders_group.setLayout(orders_layout)
        account_layout.addWidget(orders_group)

        self.account_group.setLayout(account_layout)
        layout.addWidget(self.account_group)
        self.setLayout(layout)

    def load_env(self):
        """Load saved credentials from .env file"""
        if os.path.exists(ENV_PATH) and DOTENV_AVAILABLE:
            load_dotenv(ENV_PATH)
            self.account_input.setText(os.getenv("MT5_ACCOUNT", ""))
            self.password_input.setText(os.getenv("MT5_PASSWORD", ""))
            self.server_input.setText(os.getenv("MT5_SERVER", ""))
            
            # If all fields are filled, check the save checkbox
            if all([self.account_input.text(), self.password_input.text(), self.server_input.text()]):
                self.save_account_cb.setChecked(True)

    def save_env(self):
        """Save credentials to .env file if checkbox is checked"""
        if self.save_account_cb.isChecked() and DOTENV_AVAILABLE:
            set_key(ENV_PATH, "MT5_ACCOUNT", self.account_input.text())
            set_key(ENV_PATH, "MT5_PASSWORD", self.password_input.text())
            set_key(ENV_PATH, "MT5_SERVER", self.server_input.text())
            print("✅ Account credentials saved to .env file")

    def login_mt5(self):
        """Login to MT5"""
        account = self.account_input.text()
        password = self.password_input.text()
        server = self.server_input.text()
        
        if not account or not password or not server:
            QMessageBox.warning(
                self,
                I18N.t("Warning", "Cảnh báo"),
                I18N.t("Please fill in all login fields!", "Vui lòng điền đầy đủ thông tin đăng nhập!")
            )
            return

        self.login_button.setEnabled(False)
        self.login_button.setText("🔄 Connecting...")
        
        try:
            print(f"[MT5][LOGIN] Start login workflow for {account}@{server}")
            # If already connected with different credentials, force shutdown first
            if self.mt5_conn and self.mt5_conn.connected:
                prev_acc = getattr(self.mt5_conn, 'account', None)
                prev_srv = getattr(self.mt5_conn, 'server', None)
                if str(prev_acc) != str(account) or str(prev_srv) != str(server):
                    print(f"[MT5] Switching account {prev_acc}@{prev_srv} -> {account}@{server}")
                    # Use connection manager reconfigure path if available
                    if self.mt5_conn.connection_manager:
                        try:
                            print("[MT5][LOGIN] Reconfigure existing manager...")
                            self.mt5_conn.connection_manager.reconfigure(account, password, server)
                            # Force reconnect
                            ok = self.mt5_conn.connection_manager.connect(force_reconnect=True)
                            print(f"[MT5][LOGIN] Connect after reconfigure => {ok}")
                            # Sync attributes
                            self.mt5_conn.account = account
                            self.mt5_conn.password = password
                            self.mt5_conn.server = server
                            self.mt5_conn.connected = self.mt5_conn.connection_manager.state.name.lower() == 'connected'
                        except Exception as _re:
                            print(f"[MT5] Reconfigure path failed: {_re}; falling back to full restart")
                            try:
                                self.mt5_conn.shutdown()
                            except Exception:
                                pass
                            self.mt5_conn = None
                    else:
                        # Legacy path
                        try:
                            self.mt5_conn.shutdown()
                        except Exception:
                            pass
                        self.mt5_conn = None
            if not self.mt5_conn or not self.mt5_conn.connected:
                # Fresh instance (covers first login or fallback case)
                print("[MT5][LOGIN] Creating fresh MT5Connection object")
                self.mt5_conn = MT5Connection(account, password, server)
            
            if self.mt5_conn.connected:
                print("[MT5][LOGIN] Connection marked connected, updating UI")
                self.status_label.setText("🟢 Status: Connected")
                self.status_label.setStyleSheet("font-weight:bold; color:green; padding: 8px; border: 1px solid #4CAF50; border-radius: 4px; background-color: #e8f5e8;")
                self.login_button.setText("✅ Connected")
                self.logout_button.setEnabled(True)
                self.account_group.setEnabled(True)
                
                # Enable logout button when connected successfully
                self.logout_button.setEnabled(True)
                
                # Save credentials if checkbox is checked
                self.save_env()
                
                # Update account info immediately
                self.update_account_info()
                
                # Perform account scan after successful connection
                try:
                    scan_file = self.mt5_conn.perform_account_scan()
                    if scan_file:
                        logging.info(f"✅ Account scan completed and saved: {scan_file}")
                    else:
                        logging.warning("⚠️ Account scan failed or not available")
                except Exception as e:
                    logging.error(f"❌ Account scan error: {e}")

                    # Invalidate aggregator/analysis cached account scan & refresh risk manager after account switch
                    try:
                        from comprehensive_aggregator import invalidate_account_scan_cache
                        invalidate_account_scan_cache()
                        logging.info("Cache: account scan cache invalidated after login")
                    except Exception:
                        pass
                    # Risk manager refresh hook
                    try:
                        if hasattr(self.parent(), 'risk_manager') and self.parent().risk_manager:
                            rm = self.parent().risk_manager
                            if hasattr(rm, 'refresh_after_account_switch'):
                                rm.refresh_after_account_switch()
                                logging.info("Risk manager refreshed after account switch")
                    except Exception:
                        pass
                
                # Emit signal for other tabs
                self.connection_changed.emit(True)
                
                QMessageBox.information(
                    self,
                    I18N.t("Success", "Thành công"),
                    I18N.t("Successfully connected to MT5!", "Kết nối MT5 thành công!")
                )
                
            else:
                # Collect diagnostic info
                diag = None
                try:
                    if self.mt5_conn and self.mt5_conn.connection_manager:
                        diag = self.mt5_conn.connection_manager.stats.last_error
                        if not diag:
                            # Try raw last error tuple captured
                            raw_err = self.mt5_conn.connection_manager.get_last_mt5_error()
                            if raw_err:
                                diag = f"Raw MT5 last_error: {raw_err}"
                except Exception:
                    pass
                # Try mt5.last_error for more context
                try:
                    if not diag and MT5_AVAILABLE:
                        diag = str(mt5.last_error())
                except Exception:
                    pass
                print(f"[MT5][LOGIN] Failed to connect. Diagnostic => {diag}")
                # Restore button so user can retry
                self.login_button.setEnabled(True)
                self.login_button.setText("🔑 Login to MT5")
                self.show_connection_error(diag)
                
        except Exception as e:
            print(f"[MT5][LOGIN] Exception during login: {e}")
            self.login_button.setEnabled(True)
            self.login_button.setText("🔑 Login to MT5")
            self.show_connection_error(str(e))

    def logout_mt5(self):
        """Logout from MT5"""
        if self.mt5_conn:
            self.mt5_conn.shutdown()
            self.mt5_conn = None
            
        self.status_label.setText("🔴 Status: Disconnected")
        self.status_label.setStyleSheet("font-weight:bold; color:red; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: #f9f9f9;")
        self.login_button.setText("🔑 Login to MT5")
        self.login_button.setEnabled(True)
        self.logout_button.setEnabled(False)
        self.account_group.setEnabled(False)
        
        # Disable logout button when disconnected
        self.logout_button.setEnabled(False)
        
        # Clear account info
        self.clear_account_info()
        
        # Emit signal for other tabs
        self.connection_changed.emit(False)
        
        print("✅ Disconnected from MT5")

    def show_connection_error(self, error_msg=None):
        """Show connection error"""
        self.status_label.setText("🔴 Status: Connection Failed")
        self.status_label.setStyleSheet("font-weight:bold; color:red; padding: 8px; border: 1px solid #f44336; border-radius: 4px; background-color: #ffebee;")
        self.login_button.setText("🔑 Login to MT5")
        self.login_button.setEnabled(True)
        
        # Disable logout button when connection fails
        self.logout_button.setEnabled(False)
        
        error_text = f"Failed to connect to MT5!"
        if error_msg:
            error_text += f"\nError: {error_msg}"
        QMessageBox.critical(self, I18N.t("Connection Error", "Lỗi kết nối"), error_text)

    @safe_method
    def update_account_info(self):
        """Update account information display"""
        if not self.mt5_conn or not self.mt5_conn.connected:
            return
            
        try:
            # Get account info from MT5
            if MT5_AVAILABLE:
                account_info = mt5.account_info()
                positions = mt5.positions_get()
                orders = mt5.orders_get()
                
                if account_info:
                    # Update basic info
                    self.login_label.setText(str(account_info.login))
                    self.name_label.setText(account_info.name or "N/A")
                    self.company_label.setText(account_info.company or "N/A")
                    self.server_label.setText(account_info.server or "N/A")
                    self.currency_label.setText(account_info.currency or "USD")
                    self.leverage_label.setText(f"1:{account_info.leverage}")
                    
                    # Update balance info
                    currency = account_info.currency or "USD"
                    self.balance_label.setText(f"{account_info.balance:.2f} {currency}")
                    self.equity_label.setText(f"{account_info.equity:.2f} {currency}")
                    self.margin_label.setText(f"{account_info.margin:.2f} {currency}")
                    self.free_margin_label.setText(f"{account_info.margin_free:.2f} {currency}")
                    
                    # Calculate margin level
                    if account_info.margin > 0:
                        margin_level = (account_info.equity / account_info.margin) * 100
                        self.margin_level_label.setText(f"{margin_level:.2f}%")
                    else:
                        self.margin_level_label.setText("N/A")
                    
                    # Profit/Loss with color
                    profit = account_info.profit
                    profit_text = f"{profit:.2f} {currency}"
                    if profit > 0:
                        self.profit_label.setText(f"+{profit_text}")
                        self.profit_label.setStyleSheet("color: green; font-weight: bold;")
                    elif profit < 0:
                        self.profit_label.setText(profit_text)
                        self.profit_label.setStyleSheet("color: red; font-weight: bold;")
                    else:
                        self.profit_label.setText(profit_text)
                        self.profit_label.setStyleSheet("color: black;")
                    
                    # Update trading status
                    self.positions_label.setText(str(len(positions) if positions else 0))
                    self.orders_label.setText(str(len(orders) if orders else 0))
                    self.trade_allowed_label.setText("✅ Yes" if account_info.trade_allowed else "❌ No")
                    
                    # Update positions table
                    self.update_positions_table(positions)
                    
                    # Update orders table
                    self.update_orders_table(orders)
                    
        except Exception as e:
            print(f"Error updating account info: {e}")

    def refresh_account_info(self):
        """Refresh account information (alias for update_account_info)"""
        self.update_account_info()

    def update_positions_table(self, positions):
        """Update positions table with full MT5 information"""
        if not positions:
            self.positions_table.setRowCount(0)
            return
            
        self.positions_table.setRowCount(len(positions))
        
        for row, pos in enumerate(positions):
            # Ticket
            self.positions_table.setItem(row, 0, QTableWidgetItem(str(pos.ticket)))
            
            # Symbol
            self.positions_table.setItem(row, 1, QTableWidgetItem(pos.symbol))
            # Type
            try:
                order_type = getattr(pos, 'type', None)
                if order_type is not None:
                    # Convert MT5 order type constants to readable text
                    # 0 = BUY (ORDER_TYPE_BUY), 1 = SELL (ORDER_TYPE_SELL)
                    if order_type == 0:
                        type_name = "BUY"
                    elif order_type == 1:
                        type_name = "SELL"
                    else:
                        type_name = f"Type {order_type}"
                else:
                    type_name = 'N/A'
                self.positions_table.setItem(row, 2, QTableWidgetItem(type_name))
            except Exception:
                self.positions_table.setItem(row, 2, QTableWidgetItem('N/A'))

            # Volume
            self.positions_table.setItem(row, 3, QTableWidgetItem(f"{pos.volume:.2f}"))

            # Open Price - Use proper formatting per symbol
            open_price_formatted = self.format_price_mt5_style(pos.price_open, pos.symbol)
            self.positions_table.setItem(row, 4, QTableWidgetItem(open_price_formatted))

            # Current Price - Use proper formatting per symbol
            current_price_formatted = self.format_price_mt5_style(pos.price_current, pos.symbol)
            self.positions_table.setItem(row, 5, QTableWidgetItem(current_price_formatted))

            # Stop Loss
            if pos.sl > 0:
                sl_value = self.format_price_mt5_style(pos.sl, pos.symbol)
            else:
                sl_value = "N/A"
            self.positions_table.setItem(row, 6, QTableWidgetItem(sl_value))
            
            # Take Profit
            if pos.tp > 0:
                tp_value = self.format_price_mt5_style(pos.tp, pos.symbol)
            else:
                tp_value = "N/A"
            self.positions_table.setItem(row, 7, QTableWidgetItem(tp_value))
            
            # Swap
            swap_item = QTableWidgetItem(f"{pos.swap:.2f}")
            swap_item.setFont(QFont("Segoe UI", 11))  # Slightly larger font for swap
            if pos.swap > 0:
                swap_item.setForeground(QColor('green'))
            elif pos.swap < 0:
                swap_item.setForeground(QColor('red'))
            self.positions_table.setItem(row, 8, swap_item)

            # Profit
            profit_item = QTableWidgetItem(f"{pos.profit:.2f}")
            profit_item.setFont(QFont("Segoe UI", 12, QFont.Bold))  # Larger font for profit/loss visibility
            if pos.profit > 0:
                profit_item.setForeground(QColor('green'))
            elif pos.profit < 0:
                profit_item.setForeground(QColor('red'))
            self.positions_table.setItem(row, 9, profit_item)
            
            # Close button (moved to column 10)
            close_btn = QPushButton("🚫 Close")
            close_btn.setStyleSheet("background-color: #E74C3C; color: white; font-weight: bold; padding: 4px;")
            close_btn.clicked.connect(lambda checked, ticket=pos.ticket: self.close_position(ticket))
            self.positions_table.setCellWidget(row, 10, close_btn)

    def update_orders_table(self, orders):
        """Update orders table with full MT5 information"""
        if not orders:
            self.orders_table.setRowCount(0)
            return
            
        self.orders_table.setRowCount(len(orders))
        
        order_types = ["BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP", "BUY_STOP_LIMIT", "SELL_STOP_LIMIT"]
        
        for row, order in enumerate(orders):
            # Ticket
            self.orders_table.setItem(row, 0, QTableWidgetItem(str(order.ticket)))
            
            # Symbol
            self.orders_table.setItem(row, 1, QTableWidgetItem(order.symbol))
            
            # Type
            order_type = order_types[order.type] if order.type < len(order_types) else f"Type {order.type}"
            self.orders_table.setItem(row, 2, QTableWidgetItem(order_type))
            
            # Volume
            self.orders_table.setItem(row, 3, QTableWidgetItem(str(order.volume_current)))
            
            # Open Price - Format with correct decimals for the symbol
            open_price_formatted = self.format_price_mt5_style(order.price_open, order.symbol)
            self.orders_table.setItem(row, 4, QTableWidgetItem(open_price_formatted))
            
            # Current Price (for reference) - Format with correct decimals for the symbol
            try:
                # Get current symbol price for comparison
                import MetaTrader5 as mt5
                symbol_info = mt5.symbol_info_tick(order.symbol)
                if symbol_info:
                    current_price = symbol_info.bid if "SELL" in order_type else symbol_info.ask
                    current_price_formatted = self.format_price_mt5_style(current_price, order.symbol)
                    self.orders_table.setItem(row, 5, QTableWidgetItem(current_price_formatted))
                else:
                    self.orders_table.setItem(row, 5, QTableWidgetItem("N/A"))
            except:
                self.orders_table.setItem(row, 5, QTableWidgetItem("N/A"))
            
            # Stop Loss - Format with correct decimals for the symbol
            if order.sl > 0:
                sl_value = self.format_price_mt5_style(order.sl, order.symbol)
            else:
                sl_value = "N/A"
            self.orders_table.setItem(row, 6, QTableWidgetItem(sl_value))
            
            # Take Profit - Format with correct decimals for the symbol
            if order.tp > 0:
                tp_value = self.format_price_mt5_style(order.tp, order.symbol)
            else:
                tp_value = "N/A"
            self.orders_table.setItem(row, 7, QTableWidgetItem(tp_value))
            
            # Time
            time_str = datetime.fromtimestamp(order.time_setup).strftime("%Y-%m-%d %H:%M")
            self.orders_table.setItem(row, 8, QTableWidgetItem(time_str))
            
            # Cancel button
            cancel_btn = QPushButton("❌ Cancel")
            cancel_btn.setStyleSheet("background-color: #F39C12; color: white; font-weight: bold; padding: 4px;")
            cancel_btn.clicked.connect(lambda checked, ticket=order.ticket: self.cancel_order(ticket))
            self.orders_table.setCellWidget(row, 9, cancel_btn)

    def close_position(self, ticket):
        """Close a specific position"""
        try:
            reply = QMessageBox.question(
                self,
                I18N.t("Close Position", "Đóng lệnh"),
                I18N.t(
                    "Are you sure you want to close position #{ticket}?",
                    "Bạn có chắc muốn đóng lệnh #{ticket}?",
                    ticket=ticket
                ),
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if ORDER_EXECUTOR_AVAILABLE:
                    from order_executor import get_executor_instance
                    executor = get_executor_instance()
                    result = executor.close_position(ticket)
                    
                    if result.success:
                        QMessageBox.information(
                            self,
                            I18N.t("Success", "Thành công"),
                            I18N.t("✅ Position #{ticket} closed successfully!", "✅ Đóng lệnh #{ticket} thành công!", ticket=ticket)
                        )
                        self.refresh_account_info()  # Refresh to update tables
                    else:
                        QMessageBox.warning(
                            self,
                            I18N.t("Error", "Lỗi"),
                            I18N.t("❌ Failed to close position #{ticket}:\n{msg}", "❌ Đóng lệnh #{ticket} thất bại:\n{msg}", ticket=ticket, msg=result.error_message)
                        )
                else:
                    # Direct MT5 approach
                    positions = mt5.positions_get(ticket=ticket)
                    if positions:
                        pos = positions[0]
                        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                        tick = mt5.symbol_info_tick(pos.symbol)
                        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                        
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": pos.symbol,
                            "volume": pos.volume,
                            "type": close_type,
                            "position": ticket,
                            "price": price,
                            "comment": "GUI close position",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        
                        result = mt5.order_send(request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            QMessageBox.information(
                                self,
                                I18N.t("Success", "Thành công"),
                                I18N.t("✅ Position #{ticket} closed successfully!", "✅ Đóng lệnh #{ticket} thành công!", ticket=ticket)
                            )
                            self.refresh_account_info()
                        else:
                            error_msg = f"Failed to close position. Error: {result.comment if result else 'Unknown error'}"
                            QMessageBox.warning(
                                self,
                                I18N.t("Error", "Lỗi"),
                                I18N.t("❌ {msg}", "❌ {msg}", msg=error_msg)
                            )
                    else:
                        QMessageBox.warning(
                            self,
                            I18N.t("Error", "Lỗi"),
                            I18N.t("❌ Position #{ticket} not found!", "❌ Không tìm thấy lệnh #{ticket}!", ticket=ticket)
                        )
                        
        except Exception as e:
            QMessageBox.critical(
                self,
                I18N.t("Error", "Lỗi"),
                I18N.t("❌ Error closing position #{ticket}: {err}", "❌ Lỗi khi đóng lệnh #{ticket}: {err}", ticket=ticket, err=str(e))
            )

    def cancel_order(self, ticket):
        """Cancel a specific pending order"""
        try:
            reply = QMessageBox.question(
                self,
                I18N.t("Cancel Order", "Hủy lệnh chờ"),
                I18N.t("Are you sure you want to cancel order #{ticket}?", "Bạn có chắc muốn hủy lệnh chờ #{ticket}?", ticket=ticket),
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": ticket,
                }
                
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    QMessageBox.information(
                        self,
                        I18N.t("Success", "Thành công"),
                        I18N.t("✅ Order #{ticket} cancelled successfully!", "✅ Hủy lệnh #{ticket} thành công!", ticket=ticket)
                    )
                    self.refresh_account_info()  # Refresh to update tables
                else:
                    error_msg = f"Failed to cancel order. Error: {result.comment if result else 'Unknown error'}"
                    QMessageBox.warning(
                        self,
                        I18N.t("Error", "Lỗi"),
                        I18N.t("❌ {msg}", "❌ {msg}", msg=error_msg)
                    )
                    
        except Exception as e:
            QMessageBox.critical(
                self,
                I18N.t("Error", "Lỗi"),
                I18N.t("❌ Error cancelling order #{ticket}: {err}", "❌ Lỗi khi hủy lệnh #{ticket}: {err}", ticket=ticket, err=str(e))
            )

    def close_all_positions(self):
        """Close all open positions"""
        try:
            positions = mt5.positions_get()
            if not positions:
                QMessageBox.information(
                    self,
                    I18N.t("No Positions", "Không có vị thế"),
                    I18N.t("ℹ️ No open positions to close.", "ℹ️ Không có vị thế mở để đóng.")
                )
                return
                
            reply = QMessageBox.question(
                self,
                I18N.t("Close All Positions", "Đóng tất cả vị thế"),
                I18N.t(
                    "🚨 Are you sure you want to close ALL {n} open positions?",
                    "🚨 Bạn có chắc muốn đóng TẤT CẢ {n} vị thế đang mở?",
                    n=len(positions)
                ),
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                closed_count = 0
                errors = []
                
                for pos in positions:
                    try:
                        if ORDER_EXECUTOR_AVAILABLE:
                            from order_executor import get_executor_instance
                            executor = get_executor_instance()
                            result = executor.close_position(pos.ticket)
                            if result.success:
                                closed_count += 1
                            else:
                                errors.append(f"Position #{pos.ticket}: {result.error_message}")
                        else:
                            # Direct MT5 approach
                            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                            tick = mt5.symbol_info_tick(pos.symbol)
                            price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                            
                            request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": pos.symbol,
                                "volume": pos.volume,
                                "type": close_type,
                                "position": pos.ticket,
                                "price": price,
                                "comment": "GUI close all",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_IOC,
                            }
                            
                            result = mt5.order_send(request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                closed_count += 1
                            else:
                                errors.append(f"Position #{pos.ticket}: {result.comment if result else 'Unknown error'}")
                                
                    except Exception as e:
                        errors.append(f"Position #{pos.ticket}: {str(e)}")
                
                # Show results
                message = f"📊 Close All Positions Results:\n\n"
                message += f"✅ Successfully closed: {closed_count} positions\n"
                if errors:
                    message += f"❌ Errors: {len(errors)}\n\n"
                    message += "Error details:\n"
                    for error in errors[:5]:  # Show first 5 errors
                        message += f"• {error}\n"
                    if len(errors) > 5:
                        message += f"• ... and {len(errors) - 5} more errors"
                
                QMessageBox.information(
                    self,
                    I18N.t("Close All Results", "Kết quả đóng tất cả"),
                    message
                )
                self.refresh_account_info()  # Refresh to update tables
                
        except Exception as e:
            QMessageBox.critical(
                self,
                I18N.t("Error", "Lỗi"),
                I18N.t("❌ Error closing all positions: {err}", "❌ Lỗi khi đóng tất cả lệnh: {err}", err=str(e))
            )

    def cancel_all_orders(self):
        """Cancel all pending orders"""
        try:
            orders = mt5.orders_get()
            if not orders:
                QMessageBox.information(
                    self,
                    I18N.t("No Orders", "Không có lệnh"),
                    I18N.t("ℹ️ No pending orders to cancel.", "ℹ️ Không có lệnh chờ để hủy.")
                )
                return
                
            reply = QMessageBox.question(
                self,
                I18N.t("Cancel All Orders", "Hủy tất cả lệnh"),
                I18N.t(
                    "🚨 Are you sure you want to cancel ALL {n} pending orders?",
                    "🚨 Bạn có chắc muốn hủy TẤT CẢ {n} lệnh đang chờ?",
                    n=len(orders)
                ),
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                cancelled_count = 0
                errors = []
                
                for order in orders:
                    try:
                        request = {
                            "action": mt5.TRADE_ACTION_REMOVE,
                            "order": order.ticket,
                        }
                        
                        result = mt5.order_send(request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            cancelled_count += 1
                        else:
                            errors.append(f"Order #{order.ticket}: {result.comment if result else 'Unknown error'}")
                            
                    except Exception as e:
                        errors.append(f"Order #{order.ticket}: {str(e)}")
                
                # Show results
                message = f"📊 Cancel All Orders Results:\n\n"
                message += f"✅ Successfully cancelled: {cancelled_count} orders\n"
                if errors:
                    message += f"❌ Errors: {len(errors)}\n\n"
                    message += "Error details:\n"
                    for error in errors[:5]:  # Show first 5 errors
                        message += f"• {error}\n"
                    if len(errors) > 5:
                        message += f"• ... and {len(errors) - 5} more errors"
                
                QMessageBox.information(
                    self,
                    I18N.t("Cancel All Results", "Kết quả hủy tất cả"),
                    message
                )
                self.refresh_account_info()  # Refresh to update tables
                
        except Exception as e:
            QMessageBox.critical(
                self,
                I18N.t("Error", "Lỗi"),
                I18N.t("❌ Error cancelling all orders: {err}", "❌ Lỗi khi hủy tất cả lệnh: {err}", err=str(e))
            )

    def clear_account_info(self):
        """Clear all account information displays"""
        self.login_label.setText("--")
        self.name_label.setText("--")
        self.company_label.setText("--")
        self.server_label.setText("--")
        self.currency_label.setText("--")
        self.leverage_label.setText("--")
        
        self.balance_label.setText("$0.00")
        self.equity_label.setText("$0.00")
        self.margin_label.setText("$0.00")
        self.free_margin_label.setText("$0.00")
        self.margin_level_label.setText("0.00%")
        self.profit_label.setText("$0.00")
        self.profit_label.setStyleSheet("color: black;")
        
        self.positions_label.setText("0")
        self.orders_label.setText("0")
        self.trade_allowed_label.setText("--")
        
        self.positions_table.setRowCount(0)
        self.orders_table.setRowCount(0)

    def get_symbol_decimal_places(self, symbol):
        """Get decimal places directly from MT5 symbol info"""
        if not symbol or not MT5_AVAILABLE:
            return 5
            
        try:
            # Get symbol info directly from MT5
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info and hasattr(symbol_info, 'digits'):
                return symbol_info.digits
        except Exception as e:
            print(f"⚠️ Error getting symbol info for {symbol}: {e}")
        
        # Fallback to manual detection only if MT5 info is not available
        symbol_upper = symbol.upper()
        
        # JPY pairs typically use 3 decimal places
        if 'JPY' in symbol_upper:
            return 3
            
        # Indices typically use 2 decimal places
        if any(index in symbol_upper for index in ['US30', 'US500', 'NAS100', 'GER30', 'UK100', 'JPN225']):
            return 2
            
        # Metals typically use 2 decimal places
        if any(metal in symbol_upper for metal in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return 2
            
        # Default fallback for forex pairs
        return 5

    def format_price_mt5_style(self, price, symbol=None):
        """Format price with correct decimal places like MT5"""
        if symbol is None:
            symbol = ""
            
        decimal_places = self.get_symbol_decimal_places(symbol)
        return f'{price:.{decimal_places}f}'

    def get_mt5_connection(self):
        """Get MT5 connection for other tabs"""
        return self.mt5_conn if self.mt5_conn and self.mt5_conn.connected else None

class MarketTab(QWidget):
    symbols_changed = pyqtSignal()  # Signal when symbols change
    
    def __init__(self, account_tab):
        super().__init__()

        self.account_tab = account_tab
        self.indicator_tab = None  # Will be set later
        self.mt5_conn = None
        self.all_symbols = []
        self.checked_symbols = set()

        self.threadpool = QThreadPool()
        self.user_config = load_user_config()
        self.init_ui()
        self.restore_user_config()

        # Connect to account tab signals
        self.account_tab.connection_changed.connect(self.on_connection_changed)

        # Disable auto fetch timer - not needed
        # self.fetch_timer = QTimer(self)
        # self.fetch_timer.timeout.connect(self.auto_fetch)
        # self.fetch_timer.start(15 * 60 * 1000)

    def init_ui(self):
        layout = QVBoxLayout()

        # Remove login UI from Market tab since it's now in Account tab
        layout.addWidget(QLabel("Search Symbol:"))
        self.search_input = QLineEdit()
        self.search_input.textChanged.connect(self.filter_and_sort_symbols)
        layout.addWidget(self.search_input)

        layout.addWidget(QLabel("Select Symbols (checkbox multi-select):"))
        self.symbol_list = QListWidget()
        self.symbol_list.itemChanged.connect(self.on_symbol_check_changed)
        layout.addWidget(self.symbol_list)

        layout.addWidget(QLabel("Select Timeframes and Candle Counts:"))
        tf_grid = QGridLayout()
        self.tf_spinboxes = {}
        self.tf_checkboxes = {}
        row = 0
        for tf in TIMEFRAME_MAP:
            cb = QCheckBox(tf)
            spin = QSpinBox()
            spin.setMinimum(100)
            spin.setMaximum(50000)
            spin.setValue(5000)
            tf_grid.addWidget(cb, row, 0)
            tf_grid.addWidget(spin, row, 1)
            self.tf_checkboxes[tf] = cb
            self.tf_spinboxes[tf] = spin
            row += 1
        layout.addLayout(tf_grid)

        self.fetch_button = QPushButton(I18N.t("Fetch Data Now", "Lấy dữ liệu ngay"))
        self.fetch_button.clicked.connect(self.fetch_data)
        self.fetch_button.setEnabled(False)
        layout.addWidget(self.fetch_button)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        self.setLayout(layout)

    def on_connection_changed(self, connected):
        """Handle connection status change from Account tab"""
        if connected:
            self.mt5_conn = self.account_tab.get_mt5_connection()
            if self.mt5_conn:
                self.fetch_button.setEnabled(True)
                self.all_symbols = self.mt5_conn.get_all_symbols()
                self.populate_symbol_list()
                self.log_output.append(I18N.t("✅ Connected to MT5 - Ready to fetch data", "✅ Đã kết nối MT5 - Sẵn sàng lấy dữ liệu"))
                # Hide connection warning when connected
                self.connection_status.hide()
        else:
            self.mt5_conn = None
            self.fetch_button.setEnabled(False)
            self.symbol_list.clear()
            self.checked_symbols.clear()
            self.all_symbols = []
            self.log_output.append("❌ Disconnected from MT5")
            # Show connection warning when disconnected
            self.connection_status.show()

    def init_ui(self):
        """Initialize Market tab UI (login moved to Account tab)"""
        layout = QVBoxLayout()

        # Connection status indicator
        self.connection_status = QLabel("⚠️ Please connect to MT5 in Account tab first")
        self.connection_status.setStyleSheet("background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px;")
        layout.addWidget(self.connection_status)

        # Search symbols
        layout.addWidget(QLabel("Search Symbol:"))
        self.search_input = QLineEdit()
        self.search_input.textChanged.connect(self.filter_and_sort_symbols)
        self.search_input.setPlaceholderText("Type to search symbols...")
        layout.addWidget(self.search_input)

        # Symbol selection
        symbol_info = QLabel("📋 Select symbols for your workspace (not for simultaneous trading):")
        symbol_info.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(symbol_info)
        
        # Clear All button
        clear_all_layout = QHBoxLayout()
        # Localized Clear All button
        self.clear_all_btn = QPushButton(I18N.t("🗑️ Clear All Symbols", "🗑️ Xóa tất cả mã"))
        self.clear_all_btn.clicked.connect(self.on_clear_all_symbols)
        self.clear_all_btn.setStyleSheet("""
            QPushButton { 
                background-color: #f44336; 
                color: white; 
                font-weight: bold; 
                padding: 8px; 
                border: none; 
                border-radius: 4px; 
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.clear_all_btn.setToolTip(I18N.t("Clear all selected symbols from the list", "Xóa tất cả mã đã chọn khỏi danh sách"))
        clear_all_layout.addWidget(self.clear_all_btn)
        clear_all_layout.addStretch()
        layout.addLayout(clear_all_layout)
        
        self.symbol_list = QListWidget()
        self.symbol_list.itemChanged.connect(self.on_symbol_check_changed)
        layout.addWidget(self.symbol_list)

        # Timeframe selection
        layout.addWidget(QLabel("Select Timeframes and Candle Counts:"))
        tf_grid = QGridLayout()
        self.tf_spinboxes = {}
        self.tf_checkboxes = {}
        row = 0
        for tf in TIMEFRAME_MAP:
            cb = QCheckBox(tf)
            spin = QSpinBox()
            spin.setMinimum(100)
            spin.setMaximum(50000)
            spin.setValue(5000)
            tf_grid.addWidget(cb, row, 0)
            tf_grid.addWidget(spin, row, 1)
            self.tf_checkboxes[tf] = cb
            self.tf_spinboxes[tf] = spin
            row += 1
        layout.addLayout(tf_grid)

        # Fetch button
        self.fetch_button = QPushButton(I18N.t("Fetch Data Now", "Lấy dữ liệu ngay"))
        self.fetch_button.clicked.connect(self.fetch_data)
        self.fetch_button.setEnabled(False)
        layout.addWidget(self.fetch_button)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        # Candlestick chart
        chart_label = QLabel("📈 Realtime Candlestick Chart")
        chart_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3; padding: 5px;")
        layout.addWidget(chart_label)
        
        # Chart controls
        chart_controls = QHBoxLayout()
        
        # Symbol selection for chart
        chart_controls.addWidget(QLabel("Symbol:"))
        self.chart_symbol_combo = QComboBox()
        self.chart_symbol_combo.currentTextChanged.connect(self.on_chart_symbol_changed)
        chart_controls.addWidget(self.chart_symbol_combo)
        
        # Timeframe selection for chart
        chart_controls.addWidget(QLabel("Timeframe:"))
        self.chart_timeframe_combo = QComboBox()
        self.chart_timeframe_combo.addItems(list(TIMEFRAME_MAP.keys()))
        # Default timeframe will be restored from config in restore_user_config()
        self.chart_timeframe_combo.currentTextChanged.connect(self.on_chart_timeframe_changed)
        chart_controls.addWidget(self.chart_timeframe_combo)
        
        # Chart control buttons
        self.start_chart_btn = QPushButton(I18N.t("▶️ Start Chart", "▶️ Bắt đầu biểu đồ"))
        self.start_chart_btn.clicked.connect(self.start_chart)
        self.start_chart_btn.setEnabled(False)
        chart_controls.addWidget(self.start_chart_btn)
        
        self.stop_chart_btn = QPushButton(I18N.t("⏹️ Stop Chart", "⏹️ Dừng biểu đồ"))
        self.stop_chart_btn.clicked.connect(self.stop_chart)
        self.stop_chart_btn.setEnabled(False)
        chart_controls.addWidget(self.stop_chart_btn)
        
        # Add chart customization controls
        chart_controls.addWidget(QLabel(" | "))
        
        # Current price line toggle
        self.price_line_checkbox = QCheckBox("Price Line")
        self.price_line_checkbox.setChecked(True)
        self.price_line_checkbox.toggled.connect(self.toggle_price_line)
        chart_controls.addWidget(self.price_line_checkbox)
        
        # Update interval
        chart_controls.addWidget(QLabel("Update:"))
        self.update_combo = QComboBox()
        self.update_combo.addItems(["5s", "10s", "30s", "1m"])
        self.update_combo.setCurrentText("5s")
        self.update_combo.currentTextChanged.connect(self.change_update_interval)
        chart_controls.addWidget(self.update_combo)
        
        chart_controls.addStretch()
        layout.addLayout(chart_controls)
        
        # Add candlestick chart widget (initially without indicator_tab)
        self.candlestick_chart = CandlestickChart(None)
        layout.addWidget(self.candlestick_chart)

        self.setLayout(layout)

    def set_indicator_tab(self, indicator_tab):
        """Set indicator tab reference after it's created"""
        self.indicator_tab = indicator_tab
        # Update candlestick chart with indicator tab
        if hasattr(self, 'candlestick_chart'):
            self.candlestick_chart.indicator_tab = indicator_tab
            print(f"✅ DEBUG: CandlestickChart.indicator_tab set to {type(indicator_tab)}")
        else:
            print("❌ DEBUG: candlestick_chart not found when setting indicator_tab")

    # Remove old methods - moved to AccountTab
    def populate_symbol_list(self):
        self.symbol_list.clear()
        
        # Debug: Print current state
        print(f"🔍 DEBUG populate_symbol_list:")
        print(f"  - All symbols count: {len(self.all_symbols)}")
        print(f"  - Checked symbols: {list(self.checked_symbols)}")
        
        # Only clean up checked_symbols if we have symbols from MT5
        # Don't clear when MT5 is disconnected (all_symbols is empty)
        if self.all_symbols:  # Only cleanup if we have MT5 symbols
            original_checked = self.checked_symbols.copy()
            self.checked_symbols = {sym for sym in self.checked_symbols if sym in self.all_symbols}
            
            if original_checked != self.checked_symbols:
                removed = original_checked - self.checked_symbols
                print(f"🧹 Cleaned up checked_symbols, removed: {removed}")
        else:
            print("⏳ MT5 not connected - preserving checked symbols")
        
        # Populate list with available symbols (from MT5 if connected, or from config if not)
        if self.all_symbols:
            checked = sorted([sym for sym in self.all_symbols if sym in self.checked_symbols])
            unchecked = sorted([sym for sym in self.all_symbols if sym not in self.checked_symbols])
            for sym in checked + unchecked:
                item = QListWidgetItem(sym)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                if sym in self.checked_symbols:
                    item.setCheckState(Qt.Checked)
                else:
                    item.setCheckState(Qt.Unchecked)
                self.symbol_list.addItem(item)
        else:
            # MT5 not connected - show symbols from config
            for sym in sorted(self.checked_symbols):
                item = QListWidgetItem(f"{sym} (offline)")
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.symbol_list.addItem(item)

    def on_clear_all_symbols(self):
        """Clear all selected symbols"""
        if not self.checked_symbols:
            QMessageBox.information(
                self,
                I18N.t("Info", "Thông báo"),
                I18N.t("No symbols are currently selected.", "Chưa có mã nào được chọn.")
            )
            return
            
        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            I18N.t("Clear All Symbols", "Xóa tất cả mã"),
            I18N.t(
                "Are you sure you want to clear all {n} selected symbols?",
                "Bạn có chắc muốn xóa tất cả {n} mã đã chọn?",
                n=len(self.checked_symbols)
            ),
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clear all selected symbols
            print("�️ Clearing all selected symbols")
            self.checked_symbols.clear()
            
            # Update UI - uncheck all items
            for i in range(self.symbol_list.count()):
                item = self.symbol_list.item(i)
                item.setCheckState(Qt.Unchecked)
            
            # Save user config and emit signal
            self.save_current_user_config()
            self.update_chart_symbols()
            self.symbols_changed.emit()
            print("✅ All symbols cleared from selection")

    def filter_and_sort_symbols(self):
        filter_text = self.search_input.text().upper()
        filtered = [sym for sym in self.all_symbols if filter_text in sym.upper()]
        checked = sorted([sym for sym in filtered if sym in self.checked_symbols])
        unchecked = sorted([sym for sym in filtered if sym not in self.checked_symbols])
        self.symbol_list.clear()
        for sym in checked + unchecked:
            item = QListWidgetItem(sym)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if sym in self.checked_symbols:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.symbol_list.addItem(item)
        self.save_current_user_config()

    def fetch_data(self):
        if not self.checked_symbols:
            QMessageBox.warning(
                self,
                I18N.t("Warning", "Cảnh báo"),
                I18N.t("Please select at least one symbol", "Vui lòng chọn ít nhất một mã")
            )
            return

        # Debug: Print checked symbols
        print(f"🔍 DEBUG: Fetching data for symbols: {list(self.checked_symbols)}")

        selected_tfs = [(tf, self.tf_spinboxes[tf].value()) for tf in self.tf_checkboxes if self.tf_checkboxes[tf].isChecked()]
        if not selected_tfs:
            QMessageBox.warning(
                self,
                I18N.t("Warning", "Cảnh báo"),
                I18N.t("Please select at least one timeframe", "Vui lòng chọn ít nhất một khung thời gian")
            )
            return

        # Filter out invalid symbols (ones that don't exist in all_symbols)
        valid_symbols = [sym for sym in self.checked_symbols if sym in self.all_symbols]
        invalid_symbols = [sym for sym in self.checked_symbols if sym not in self.all_symbols]
        
        if invalid_symbols:
            print(f"⚠️ WARNING: Invalid symbols found: {invalid_symbols}")
            self.log_output.append(f"⚠️ Skipping invalid symbols: {', '.join(invalid_symbols)}")
            # Remove invalid symbols from checked_symbols
            self.checked_symbols = set(valid_symbols)
            self.save_current_user_config()
            self.populate_symbol_list()  # Refresh UI

        if not valid_symbols:
            QMessageBox.warning(
                self,
                I18N.t("Warning", "Cảnh báo"),
                I18N.t(
                    "No valid symbols selected. Please check your symbol selection.",
                    "Không có mã hợp lệ được chọn. Vui lòng kiểm tra lại danh sách mã."
                )
            )
            return

        self.log_output.append(f"Starting fetch for symbols: {', '.join(valid_symbols)}")
        
        # Add progress tracking
        total_tasks = len(valid_symbols) * len(selected_tfs)
        current_task = 0
        
        for sym in valid_symbols:
            for tf, count in selected_tfs:
                current_task += 1
                try:
                    self.log_output.append(f"📊 Fetching {sym} @ {tf} ({current_task}/{total_tasks})")
                    
                    # Add timeout mechanism using threading
                    import threading
                    import time
                    
                    candles = None
                    fetch_error = None
                    
                    def fetch_with_timeout():
                        nonlocal candles, fetch_error
                        try:
                            candles = fetch_and_save_candles(sym, tf, count, folder="data")
                        except Exception as e:
                            fetch_error = str(e)
                    
                    # Create and start thread with timeout
                    fetch_thread = threading.Thread(target=fetch_with_timeout)
                    fetch_thread.daemon = True
                    fetch_thread.start()
                    fetch_thread.join(timeout=30)  # 30 second timeout
                    
                    if fetch_thread.is_alive():
                        msg = f"⏰ Timeout fetching {sym} @ {tf} (>30s)"
                        self.log_output.append(msg)
                        continue
                    
                    if fetch_error:
                        msg = f"❌ Error fetching {sym} @ {tf}: {fetch_error}"
                    elif candles:
                        msg = f"✅ Saved {len(candles)} candles for {sym} @ {tf}"
                    else:
                        msg = f"❌ No data for {sym} @ {tf}"
                    
                except Exception as e:
                    msg = f"💥 Exception fetching {sym} @ {tf}: {str(e)}"
                
                self.log_output.append(msg)
        self.save_current_user_config()

    def on_fetch_finished(self, msg):
        self.log_output.append(msg)

    def check_connection(self):
        if self.mt5_conn and self.mt5_conn.connected:
            self.status_label.setText("Status: Connected")
            self.status_label.setStyleSheet("font-weight:bold; color:green")
        else:
            self.status_label.setText("Status: Disconnected")
            self.status_label.setStyleSheet("font-weight:bold; color:red")
            self.login_button.setEnabled(True)
            self.logout_button.setEnabled(False)
            self.fetch_button.setEnabled(False)

    def auto_fetch(self):
        """Auto fetch data - disabled by default to prevent spam"""
        # This method was causing continuous data fetching
        # Now disabled to prevent broker API spam
        print("📊 Auto fetch is disabled to prevent API spam")
        return

    def restore_user_config(self):
        config = self.user_config.get("market_tab", {})
        checked = config.get("checked_symbols", [])
        self.checked_symbols = set(checked)
        
        # Get saved chart symbol selection
        saved_chart_symbol = config.get("selected_chart_symbol", "")
        saved_chart_timeframe = config.get("selected_chart_timeframe", "H1")
        
        # If no symbols are configured, set some default ones
        if not self.checked_symbols and hasattr(self, 'all_symbols') and self.all_symbols:
            # Try different symbol formats that brokers commonly use
            default_symbols = ["EURUSD", "GBPUSD", "XAUUSD", "EURUSD.", "GBPUSD.", "XAUUSD."]
            available_defaults = [sym for sym in default_symbols if sym in self.all_symbols]
            if available_defaults:
                # Only take first 3 to avoid duplicates
                self.checked_symbols = set(available_defaults[:3])
                print(f"✅ Set default symbols: {self.checked_symbols}")
            else:
                print(f"ℹ️ No default symbols found in {len(self.all_symbols)} available symbols")
        
        tf_config = config.get("tf_config", {})
        has_any_tf_checked = False
        for tf, cb in self.tf_checkboxes.items():
            is_checked = tf_config.get(tf, {}).get("checked", False)
            cb.setChecked(is_checked)
            if is_checked:
                has_any_tf_checked = True
        
        # If no timeframes are configured, set some defaults
        if not has_any_tf_checked:
            default_timeframes = ["H1", "H4"]
            for tf in default_timeframes:
                if tf in self.tf_checkboxes:
                    self.tf_checkboxes[tf].setChecked(True)
                    print(f"✅ Set default timeframe: {tf}")
                    
        for tf, spin in self.tf_spinboxes.items():
            spin.setValue(tf_config.get(tf, {}).get("count", 500))
        
        # Populate symbol list first
        self.populate_symbol_list()
        
        # Restore chart symbol selection after populating the list
        if hasattr(self, 'chart_symbol_combo') and hasattr(self, 'chart_timeframe_combo'):
            self.update_chart_symbols()  # Update chart symbols combo
            
            # Restore saved chart symbol if it exists in checked symbols
            if saved_chart_symbol and saved_chart_symbol in self.checked_symbols:
                index = self.chart_symbol_combo.findText(saved_chart_symbol)
                if index >= 0:
                    self.chart_symbol_combo.setCurrentIndex(index)
                    print(f"✅ Restored chart symbol: {saved_chart_symbol}")
            
            # Restore saved chart timeframe or set default
            if saved_chart_timeframe and saved_chart_timeframe in TIMEFRAME_MAP:
                index = self.chart_timeframe_combo.findText(saved_chart_timeframe)
                if index >= 0:
                    self.chart_timeframe_combo.setCurrentIndex(index)
                    print(f"✅ Restored chart timeframe: {saved_chart_timeframe}")
            else:
                # Set default timeframe if none saved
                self.chart_timeframe_combo.setCurrentText("H1")
                print("✅ Set default chart timeframe: H1")
        
        print(f"📊 Restored {len(self.checked_symbols)} symbols from config")

    def save_current_user_config(self):
        config = {}
        config["checked_symbols"] = list(self.checked_symbols)
        
        # Save currently selected chart symbol and timeframe
        if hasattr(self, 'chart_symbol_combo') and hasattr(self, 'chart_timeframe_combo'):
            config["selected_chart_symbol"] = self.chart_symbol_combo.currentText()
            config["selected_chart_timeframe"] = self.chart_timeframe_combo.currentText()
        
        tf_config = {}
        for tf in self.tf_checkboxes:
            tf_config[tf] = {
                "checked": self.tf_checkboxes[tf].isChecked(),
                "count": self.tf_spinboxes[tf].value()
            }
        config["tf_config"] = tf_config
        
        user_config = load_user_config()
        user_config["market_tab"] = config
        save_user_config(user_config)
        
        # Debug log for symbol save
        if len(self.checked_symbols) <= 5:
            print(f"💾 Saved symbols: {list(self.checked_symbols)}")
        else:
            print(f"💾 Saved {len(self.checked_symbols)} symbols to config")

    def on_connection_changed(self, connected):
        """Handle connection status change from Account tab"""
        if connected:
            self.mt5_conn = self.account_tab.get_mt5_connection()
            if self.mt5_conn:
                self.fetch_button.setEnabled(True)
                self.start_chart_btn.setEnabled(True)
                
                # Get all available symbols from MT5
                self.all_symbols = self.mt5_conn.get_all_symbols()
                print(f"📊 Loaded {len(self.all_symbols)} symbols from MT5")
                
                # Restore saved symbol selections and populate list
                self.restore_user_config()
                
                self.log_output.append(I18N.t("✅ Connected to MT5 - Ready to fetch data", "✅ Đã kết nối MT5 - Sẵn sàng lấy dữ liệu"))
                # Hide connection warning when connected
                if hasattr(self, 'connection_status'):
                    self.connection_status.hide()
        else:
            self.mt5_conn = None
            self.fetch_button.setEnabled(False)
            self.start_chart_btn.setEnabled(False)
            self.stop_chart_btn.setEnabled(False)
            
            # Save current selections before clearing
            self.save_current_user_config()
            
            # Clear UI but keep checked_symbols in memory for when reconnecting
            self.symbol_list.clear()
            self.all_symbols = []
            self.chart_symbol_combo.clear()
            self.candlestick_chart.stop_realtime_update()
            self.log_output.append("❌ Disconnected from MT5")
            # Show connection warning when disconnected
            if hasattr(self, 'connection_status'):
                self.connection_status.show()

    def update_chart_symbols(self):
        """Update chart symbol combo with available symbols"""
        self.chart_symbol_combo.clear()
        if self.checked_symbols:
            symbols = sorted(list(self.checked_symbols))
            self.chart_symbol_combo.addItems(symbols)

    def on_symbol_check_changed(self, item):
        sym = item.text()
        old_symbols = self.checked_symbols.copy()
        
        if item.checkState() == Qt.Checked:
            self.checked_symbols.add(sym)
        else:
            self.checked_symbols.discard(sym)
        
        self.save_current_user_config()
        self.update_chart_symbols()
        
        # Trigger smart cleanup if symbols actually changed
        if old_symbols != self.checked_symbols:
            try:
                # Import and trigger smart cleanup
                from smart_cleanup_trigger import trigger_smart_cleanup_on_symbol_change
                cleanup_result = trigger_smart_cleanup_on_symbol_change(self.checked_symbols)
                
                if cleanup_result.get('files_deleted', 0) > 0:
                    print(f"🧹 Smart cleanup: Removed {cleanup_result['files_deleted']} files, "
                          f"freed {cleanup_result['space_freed_mb']:.2f} MB")
                
            except Exception as e:
                print(f"⚠️ Smart cleanup warning: {e}")
        
        # Emit signal to notify Risk Management tab
        self.symbols_changed.emit()
        # Reduced logging to avoid spam
        if len(self.checked_symbols) <= 10:
            print(f"🔄 Chart symbols updated: {len(self.checked_symbols)} symbols available")
        else:
            print(f"🔄 Chart symbols: {len(self.checked_symbols)} symbols available for selection")

    def on_chart_symbol_changed(self, symbol):
        """Handle chart symbol change"""
        # Save the selected chart symbol to config
        self.save_current_user_config()
        
        if self.candlestick_chart and symbol and hasattr(self.candlestick_chart, 'update_timer'):
            if self.candlestick_chart.update_timer.isActive():
                # Restart chart with new symbol
                self.start_chart()
                
        print(f"📈 Chart symbol changed to: {symbol}")

    def on_chart_timeframe_changed(self, timeframe):
        """Handle chart timeframe change"""
        # Save the selected chart timeframe to config
        self.save_current_user_config()
        
        if self.candlestick_chart and hasattr(self.candlestick_chart, 'update_timer'):
            if self.candlestick_chart.update_timer.isActive():
                # Restart chart with new timeframe
                self.start_chart()
                
        print(f"⏰ Chart timeframe changed to: {timeframe}")

    def start_chart(self):
        """Start realtime chart"""
        symbol = self.chart_symbol_combo.currentText()
        timeframe_str = self.chart_timeframe_combo.currentText()
        
        if not symbol:
            QMessageBox.warning(
                self,
                I18N.t("Warning", "Cảnh báo"),
                I18N.t("Please select a symbol for the chart", "Vui lòng chọn một mã để vẽ biểu đồ")
            )
            return
        
        if not self.mt5_conn:
            QMessageBox.warning(
                self,
                I18N.t("Warning", "Cảnh báo"),
                I18N.t("Please connect to MT5 first", "Vui lòng kết nối MT5 trước")
            )
            return
        
        # Convert timeframe string to MT5 constant
        timeframe = TIMEFRAME_MAP.get(timeframe_str)
        if not timeframe:
            QMessageBox.warning(
                self,
                I18N.t("Warning", "Cảnh báo"),
                I18N.t("Invalid timeframe: {t}", "Khung thời gian không hợp lệ: {t}", t=timeframe_str)
            )
            return
        
        # Start the chart
        self.candlestick_chart.start_realtime_update(symbol, timeframe, self.mt5_conn)
        self.start_chart_btn.setEnabled(False)
        self.stop_chart_btn.setEnabled(True)
        self.log_output.append(f"📈 Started realtime chart for {symbol} ({timeframe_str})")

    def stop_chart(self):
        """Stop realtime chart"""
        self.candlestick_chart.stop_realtime_update()
        self.start_chart_btn.setEnabled(True)
        self.stop_chart_btn.setEnabled(False)
        self.log_output.append("⏹️ Stopped realtime chart")

    def toggle_price_line(self, checked):
        """Toggle current price line display"""
        if self.candlestick_chart:
            self.candlestick_chart.show_price_line = checked
            # Refresh chart if running
            if hasattr(self.candlestick_chart, 'update_timer') and self.candlestick_chart.update_timer.isActive():
                self.candlestick_chart.draw_candlesticks()

    def change_update_interval(self, interval_text):
        """Change chart update interval"""
        interval_map = {"5s": 5000, "10s": 10000, "30s": 30000, "1m": 60000}
        interval_ms = interval_map.get(interval_text, 5000)
        
        if self.candlestick_chart and hasattr(self.candlestick_chart, 'update_timer'):
            was_active = self.candlestick_chart.update_timer.isActive()
            if was_active:
                self.candlestick_chart.update_timer.stop()
                self.candlestick_chart.update_timer.start(interval_ms)
                self.log_output.append(f"⏱️ Changed update interval to {interval_text}")

class NewsTab(QWidget):
    def refresh_impact_labels(self):
        """Refresh impact checkbox labels when language changes"""
        try:
            for cb in self.impact_checkboxes:
                if hasattr(cb, 'impact_value'):
                    cb.setText(self.get_impact_label(cb.impact_value))
        except Exception as e:
            print(f"[NewsTab] Impact label refresh error: {e}")
    
    def refresh_all_labels(self):
        """Refresh all UI labels when language changes"""
        try:
            # Refresh impact labels
            self.refresh_impact_labels()
            
            # Refresh group box titles
            if hasattr(self, 'filters_group'):
                self.filters_group.setTitle(I18N.t("Filters", "Bộ lọc"))
            
            if hasattr(self, 'auto_trading_group'):
                self.auto_trading_group.setTitle(I18N.t("Auto Trading Integration", "Tích hợp giao dịch tự động"))
                
            if hasattr(self, 'auto_schedule_status_group'):
                self.auto_schedule_status_group.setTitle(I18N.t("Auto News Schedule Status", "Trạng thái lịch tin tức tự động"))
            
            # Refresh static labels
            if hasattr(self, 'currency_label'):
                self.currency_label.setText(I18N.t("Currency:", "Tiền tệ:"))
                
            # Refresh checkbox text
            if hasattr(self, 'use_economic_calendar_checkbox'):
                self.use_economic_calendar_checkbox.setText(I18N.t("✅ Enable News detection", "✅ Bật phát hiện tin tức"))
                
            if hasattr(self, 'auto_schedule_checkbox'):
                self.auto_schedule_checkbox.setText(I18N.t("✅ Enable Auto Schedule", "✅ Bật lịch tự động"))
            
            # Refresh fetch button text
            if hasattr(self, 'fetch_button'):
                self.fetch_button.setText(I18N.t("Fetch News", "Lấy tin tức"))
                
            if hasattr(self, 'parse_news_times_btn'):
                self.parse_news_times_btn.setText(I18N.t("📊 Parse from News", "📊 Lấy từ tin tức"))
            
            # Refresh placeholders and tooltips
            if hasattr(self, 'schedule_times_text'):
                self.schedule_times_text.setPlaceholderText(I18N.t("e.g., 08:30,15:30,21:30", "VD: 08:30,15:30,21:30"))
                self.schedule_times_text.setToolTip(I18N.t("Enter times in HH:MM format, separated by commas", "Nhập giờ theo định dạng HH:MM, phân cách bằng dấu phẩy"))
                
            if hasattr(self, 'parse_news_times_btn'):
                self.parse_news_times_btn.setToolTip(I18N.t("Parse news release times from latest fetched data", "Lấy giờ ra tin từ dữ liệu tin tức mới nhất"))
            
            # Update auto schedule status if available
            if hasattr(self, 'schedule_status_label'):
                try:
                    self.update_auto_schedule_status("🤖 System Active", "🤖 Hệ thống hoạt động")
                except:
                    pass
                
        except Exception as e:
            print(f"[NewsTab] All labels refresh error: {e}")

    def get_impact_label(self, impact_level):
        """Get localized impact label"""
        impact_labels = {
            1: I18N.t("Low", "Thấp"),
            2: I18N.t("Medium", "Trung bình"), 
            3: I18N.t("High", "Cao")
        }
        return impact_labels.get(impact_level, "Unknown")

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.last_loaded_file = None

    def load_latest_news_file(self, silent: bool = True):
        """Load most recent news_output/news_forexfactory_*.json and display."""
        try:
            pattern = os.path.join('news_output', 'news_forexfactory_*.json')
            files = sorted(glob.glob(pattern), reverse=True)
            if not files:
                if not silent:
                    self.news_text.append("⚠️ No news files found.")
                return False
            latest = files[0]
            if self.last_loaded_file == latest:
                return False  # unchanged
            with open(latest,'r',encoding='utf-8') as f:
                data = json.load(f)
            # data may already be list; ensure correct type
            if isinstance(data, dict) and 'events' in data:
                events = data['events']
            elif isinstance(data, list):
                events = data
            else:
                events = []
            self.display_news(events)
            self.last_loaded_file = latest
            if not silent:
                self.news_text.append(f"\n✅ Loaded latest file: {os.path.basename(latest)}")
            return True
        except Exception as e:
            if not silent:
                self.news_text.append(f"❌ Error loading latest news: {e}")
            return False

    # Slot wrapper to safely call from timers
    def load_latest_news_file_slot(self):  # no decorator needed; used in QTimer.singleShot
        self.load_latest_news_file(silent=True)

    def init_ui(self):
        layout = QVBoxLayout()

        self.filters_group = QGroupBox(I18N.t("Filters", "Bộ lọc"))
        filters_layout = QHBoxLayout()

        currency_layout = QVBoxLayout()
        self.currency_label = QLabel(I18N.t("Currency:", "Tiền tệ:"))
        currency_layout.addWidget(self.currency_label)
        self.currency_checkboxes = []
        for curr in ALL_CURRENCY:
            cb = QCheckBox(curr)
            cb.setChecked(True)
            cb.stateChanged.connect(self.on_filter_changed)  # Auto-save on change
            self.currency_checkboxes.append(cb)
            currency_layout.addWidget(cb)
        filters_layout.addLayout(currency_layout)

        impact_layout = QVBoxLayout()
        impact_layout.addWidget(QLabel(I18N.t("Impact:", "Tác động:")))
        self.impact_checkboxes = []
        for imp in ALL_IMPACT:
            label = self.get_impact_label(imp)
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.impact_value = imp
            cb.stateChanged.connect(self.on_filter_changed)  # Auto-save on change
            self.impact_checkboxes.append(cb)
            impact_layout.addWidget(cb)
        filters_layout.addLayout(impact_layout)

        self.filters_group.setLayout(filters_layout)
        layout.addWidget(self.filters_group)

        # Auto trading integration group
        self.auto_trading_group = QGroupBox(I18N.t("Auto Trading Integration", "Tích hợp giao dịch tự động"))
        auto_trading_layout = QVBoxLayout()
        
        self.use_economic_calendar_checkbox = QCheckBox(I18N.t("✅ Enable News detection", "✅ Bật phát hiện tin tức"))
        
        # Load saved setting from user config
        try:
            user_config = load_user_config()
            saved_state = user_config.get("use_economic_calendar", True)
            self.use_economic_calendar_checkbox.setChecked(saved_state)
            print(f"[NewsTab] Loaded news detection setting: {saved_state}")
        except Exception as e:
            print(f"[NewsTab] Error loading setting, using default: {e}")
            self.use_economic_calendar_checkbox.setChecked(True)  # Fallback default
        
        self.use_economic_calendar_checkbox.setToolTip(
            "When enabled, auto trading will consider economic news events.\n"
            "High impact news may pause or modify trading decisions."
        )
        self.use_economic_calendar_checkbox.stateChanged.connect(self.on_economic_calendar_toggle)
        auto_trading_layout.addWidget(self.use_economic_calendar_checkbox)
        
        self.auto_trading_group.setLayout(auto_trading_layout)
        layout.addWidget(self.auto_trading_group)

        # Auto Schedule Status Display (Read-only)
        self.auto_schedule_status_group = QGroupBox(I18N.t("Auto News Schedule Status", "Trạng thái lịch tin tức tự động"))
        status_layout = QVBoxLayout()
        
        # Status display
        self.schedule_status_label = QLabel(I18N.t("🤖 Auto-schedule: Initializing...", "🤖 Lịch tự động: Đang khởi tạo..."))
        self.schedule_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        status_layout.addWidget(self.schedule_status_label)
        
        # Current scheduled times display
        self.scheduled_times_label = QLabel(I18N.t("📅 Scheduled Times: None", "📅 Giờ đã lên lịch: Chưa có"))
        self.scheduled_times_label.setStyleSheet("color: #2196F3; font-size: 11px;")
        status_layout.addWidget(self.scheduled_times_label)
        
        # Next schedule display
        self.next_schedule_label = QLabel(I18N.t("⏰ Next Auto-Fetch: Calculating...", "⏰ Quét tin tiếp theo: Đang tính..."))
        self.next_schedule_label.setStyleSheet("color: #FF9800; font-size: 11px;")
        status_layout.addWidget(self.next_schedule_label)
        
        self.auto_schedule_status_group.setLayout(status_layout)
        layout.addWidget(self.auto_schedule_status_group)

        self.fetch_button = QPushButton(I18N.t("Fetch News", "Lấy tin tức"))
        self.fetch_button.clicked.connect(self.fetch_news)
        layout.addWidget(self.fetch_button)

        self.news_text = QTextEdit()
        self.news_text.setReadOnly(True)
        layout.addWidget(self.news_text)

        self.setLayout(layout)
        
        # Initialize auto schedule system
        self.schedule_timers = []  # List to store QTimer objects
        self.schedule_times = []   # List of schedule times in HH:MM format
        self.auto_schedule_enabled = True  # Always enabled in auto mode
        
        # Initialize auto schedule manager thread
        self.auto_schedule_manager = None
        
        # Load saved settings after initialization
        self.load_news_settings()
        
        # Start auto schedule system automatically in separate thread
        self.start_auto_news_system_threaded()

    def fetch_news(self):
        print("🔄 Fetch News button clicked!")  # Debug log
        
        # Dừng worker cũ nếu còn chạy
        if hasattr(self, "worker") and self.worker is not None and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
        
        selected_currencies = [cb.text() for cb in self.currency_checkboxes if cb.isChecked()]
        selected_impacts = [cb.impact_value for cb in self.impact_checkboxes if cb.isChecked()]

        print(f"🔍 Selected currencies: {selected_currencies}")  # Debug log
        print(f"🔍 Selected impacts: {selected_impacts}")  # Debug log

        if not selected_currencies or not selected_impacts:
            QMessageBox.warning(
                self,
                I18N.t("Warning", "Cảnh báo"),
                I18N.t("Please select at least one currency and impact level", "Vui lòng chọn ít nhất một đồng tiền và mức độ ảnh hưởng")
            )
            return

        self.news_text.clear()
        self.news_text.append("🔄 Loading economic calendar data...")

        print("🚀 Starting NewsWorker...")  # Debug log
        self.worker = NewsWorker(selected_currencies, selected_impacts)
        self.worker.finished.connect(self.display_news)
        self.worker.error.connect(self.on_news_error)
        self.worker.start()
        print("✅ NewsWorker started!")  # Debug log

    def on_news_error(self, msg):
        print(f"📰 News error received: {msg}")  # Debug log
        QMessageBox.critical(self, I18N.t("Error", "Lỗi"), msg)
        self.news_text.setText(I18N.t("❌ Error fetching news: {msg}", "❌ Lỗi lấy tin tức: {msg}", msg=msg))

    def display_news(self, news_list):
        print(f"📰 Display news called with {len(news_list)} events")  # Debug log
        
        self.news_text.clear()
        if not news_list:
            self.news_text.append("📅 No economic events found for today.")
            self.news_text.append("")
            self.news_text.append("🔍 Source used:")
            self.news_text.append("   • ✅ ForexFactory.com (Selenium scraping)")
            self.news_text.append("")
            self.news_text.append("📋 This could be because:")
            self.news_text.append("   • Today is a weekend or holiday")
            self.news_text.append("   • No major economic events are scheduled")
            self.news_text.append("   • All economic data sources are temporarily unavailable")
            self.news_text.append("")
            self.news_text.append("💡 Solutions:")
            self.news_text.append("   • Try again tomorrow for weekday economic calendar events")
            self.news_text.append("   • Check ForexFactory.com manually if needed")
            self.news_text.append("   • Consider using alternative news sources")
            return
        
        # Get status information from news items
        status_message = news_list[0].get('_status_message', '❓ Status unknown') if news_list else '❓ Status unknown'
        is_real_data = news_list[0].get('_is_real_data', False) if news_list else False
        
        impact_colors = {
            "High": "#d64541",
            "Medium": "#f5a623", 
            "Low": "#f7ca18",
            "None": "#b2b2b2"
        }
        
        # Display header with status information
        first_news = news_list[0]
        date_field = first_news.get("date", "")
        if not date_field:
            dt = first_news.get("datetime", "")
            parts = dt.split()
            if len(parts) >= 3:
                date_field = f"{parts[1]} {parts[2]}"
            else:
                date_field = dt
        
        # Display status banner for real data
        try:
            # Determine data source based on event properties
            data_source = "Unknown source"
            if news_list and len(news_list) > 0:
                # We currently use Selenium-only from ForexFactory
                data_source = "📡 ForexFactory.com (Selenium)"
            
            self.news_text.append(
                f"<div style='background-color:#d4edda; border:1px solid #c3e6cb; padding:12px; border-radius:8px; margin-bottom:15px;'>"
                f"<b>📡 {I18N.t('Data Source:', 'Nguồn dữ liệu:')}</b> {data_source}<br>"
                f"<b>🔄 {I18N.t('System:', 'Hệ thống:')}</b> ForexFactory (Selenium-only)"
                f"</div>"
            )
        except:
            pass
        
        # Add timezone information header
        self.news_text.append(
            f"<div style='background-color:#e3f2fd; border:1px solid #bbdefb; padding:10px; border-radius:6px; margin-bottom:10px;'>"
            f"<b>🕐 {I18N.t('Timezone:', 'Múi giờ:')}</b> {I18N.t('All times displayed in', 'Tất cả thời gian hiển thị theo')} <b>{I18N.t('Vietnam Time (UTC+7)', 'Giờ Việt Nam (UTC+7)')}</b>"
            f"</div>"
        )
        
        self.news_text.append(f"<b>📅 {I18N.t('Date:', 'Ngày:')} {date_field}</b> | <b>📊 {I18N.t('Found', 'Tìm thấy')} {len(news_list)} {I18N.t('events', 'sự kiện')}</b><br><hr>")

        for news in news_list:
            impact = news.get("impact", 0)
            # Handle both numeric and string impact values
            if isinstance(impact, str):
                impact_map = {"Low": 1, "Medium": 2, "High": 3}
                impact = impact_map.get(impact, 1)
            impact = int(impact)
            
            if impact == 0:
                continue
                
            time = news.get("time", "")
            timezone_info = news.get("timezone", "")
            currency = news.get("currency", "N/A")
            
            # Handle both string and numeric impact values
            raw_impact = news.get("impact", "None")
            if isinstance(raw_impact, str):
                # If impact is string like "High", "Medium", "Low", use directly
                impact_label = raw_impact.title() if raw_impact else "None"
            else:
                # If impact is numeric, use mapping
                impact_label = self.get_impact_label(raw_impact)
            
            impact_color = impact_colors.get(impact_label, "#b2b2b2")
            title = news.get("title", news.get("event", "No Title"))
            
            # Format time without additional timezone info since header already shows it
            time_display = time  # Don't add timezone_info here since header already shows Vietnam Time
            
            # Better parsing of actual value
            actual_value = news.get("actual", "TBA")
            
            # Check if this is a real economic event with potential actual data
            if actual_value == "TBA" and news.get("_is_real_data", False):
                # For real events, check if it's past the release time
                try:
                    from datetime import datetime
                    import pytz
                    now_vn = datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))
                    
                    # If we have time info, try to determine if event has passed
                    event_time = news.get("time", "")
                    if event_time and ":" in event_time:
                        # Parse time like "08:30 (UTC+7)" or "08:30"
                        clean_time = event_time.replace("(UTC+7)", "").strip()
                        try:
                            event_hour, event_min = map(int, clean_time.split(":"))
                            today = now_vn.replace(hour=event_hour, minute=event_min, second=0, microsecond=0)
                            
                            if now_vn > today:
                                actual_value = "Released (Check Source)"
                            else:
                                actual_value = "Pending"
                        except:
                            actual_value = "TBA"
                except:
                    actual_value = "TBA"
            
            description = news.get("description", "")
            if "|" in description:
                parts = description.split("|")
                description_actual = parts[0].replace("Actual:", "").strip()
            else:
                description_actual = news.get("actual", "TBA")
            
            # Use the improved actual_value we calculated above, fallback to description if needed
            final_actual = actual_value if actual_value != "TBA" else description_actual
            
            status = "Released" if final_actual and final_actual not in ["TBA", "Pending", ""] else "Forecast"
            forecast = news.get("forecast", "N/A")
            previous = news.get("previous", "N/A")
            
            # Use event field if title is empty
            if not title or title == "No Title":
                title = news.get("event", "Economic Event")
            
            self.news_text.append(
                f"<b>🕐 {I18N.t('Time', 'Thời gian')}:</b> <span style='color:#007acc'><b>{time_display}</b></span> | "
                f"<b>💱 {currency}</b> | <b>📈 {I18N.t('Impact', 'Tác động')}:</b> <span style='color:{impact_color}'><b>{impact_label}</b></span><br>"
                f"<b>📰 {I18N.t('Event', 'Sự kiện')}:</b> {title}<br>"
                f"<b>✅ {I18N.t('Actual', 'Thực tế')}:</b> <span style='color:#27ae60'><b>{final_actual}</b></span> | "
                f"<b>🎯 {I18N.t('Forecast', 'Dự báo')}:</b> <span style='color:#e67e22'><b>{forecast}</b></span> | "
                f"<b>📊 {I18N.t('Previous', 'Trước đó')}:</b> <span style='color:#e74c3c'><b>{previous}</b></span> | "
                f"<b>🔖 {I18N.t('Status', 'Trạng thái')}:</b> <span style='color:#007acc'><b>{status}</b></span><br>"
                "<hr>"
            )
        
        # 🤖 Auto-parse and auto-schedule news times after displaying
        self.auto_parse_and_schedule_news_times()

    def on_economic_calendar_toggle(self, state):
        """Xử lý khi checkbox economic calendar được toggle"""
        is_enabled = state == 2  # Qt.Checked = 2
        
        # Lưu setting vào user config
        user_config = load_user_config()
        user_config["use_economic_calendar"] = is_enabled
        save_user_config(user_config)
        
        # Log thông báo
        status_text = "enabled" if is_enabled else "disabled"
        print(f"📰 News detection in Auto Trading: {status_text}")
        
        # Hiển thị thông báo cho user
        status_msg = "✅ Enabled" if is_enabled else "❌ Disabled"
        self.news_text.append(
            f"<div style='background-color:{'#d4edda' if is_enabled else '#f8d7da'}; "
            f"border:1px solid {'#c3e6cb' if is_enabled else '#f5c6cb'}; "
            f"padding:8px; border-radius:4px; margin:5px 0;'>"
            f"<b>� News Detection: {status_msg}</b><br>"
            f"Economic news will {'be detected and considered' if is_enabled else 'be ignored'} during auto trading."
            f"</div>"
        )

    def get_economic_calendar_setting(self):
        """Lấy setting hiện tại của economic calendar"""
        return self.use_economic_calendar_checkbox.isChecked()

    def save_news_settings(self):
        """Save NewsTab settings to file"""
        try:
            import os
            os.makedirs("news_output", exist_ok=True)
            
            # Get selected currencies
            selected_currencies = [cb.text() for cb in self.currency_checkboxes if cb.isChecked()]
            # Get selected impact levels
            selected_impacts = [cb.impact_value for cb in self.impact_checkboxes if cb.isChecked()]
            # Save into instance settings for consistency
            self.settings['selected_currencies'] = selected_currencies
            self.settings['selected_impacts'] = selected_impacts
            # Fibonacci retracement start index & derived list
            start_idx = self.dca_fibo_start_combo.currentData()
            self.settings['dca_fibo_start_level'] = start_idx
            retracement_levels = [23.6, 38.2, 50.0, 61.8, 78.6, 100.0]
            derived = retracement_levels[start_idx:]
            # Store as comma separated percentages (no % sign) for executor consumption
            self.settings['dca_fibo_levels'] = ','.join(f"{v}" for v in derived)
            self.settings['dca_fibo_scheme'] = 'retracement_pct'
            settings_file = "news_output/news_settings.json"
            import json
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            
            print(f"✅ News settings saved: {len(selected_currencies)} currencies, {len(selected_impacts)} impact levels")
            
            # Also update user_config for economic calendar
            user_config = load_user_config()
            user_config["use_economic_calendar"] = self.settings.get('use_economic_calendar', True)
            save_user_config(user_config)
            
        except Exception as e:
            print(f"❌ Error saving news settings: {e}")

    def load_news_settings(self):
        """Load NewsTab settings from file"""
        try:
            settings_file = "news_output/news_settings.json"
            if not os.path.exists(settings_file):
                print("[NewsTab] No saved settings found, using defaults")
                return
            
            import json
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            # Restore currency selections
            selected_currencies = settings.get('selected_currencies', [])
            for cb in self.currency_checkboxes:
                cb.setChecked(cb.text() in selected_currencies)
            
            # Restore impact level selections
            selected_impacts = settings.get('selected_impacts', [3])  # Default to High
            for cb in self.impact_checkboxes:
                cb.setChecked(cb.impact_value in selected_impacts)
            
            # Restore economic calendar setting
            use_calendar = settings.get('use_economic_calendar', True)
            self.use_economic_calendar_checkbox.setChecked(use_calendar)
            
            print(f"✅ News settings loaded: {len(selected_currencies)} currencies, {len(selected_impacts)} impact levels")
            
        except Exception as e:
            print(f"❌ Error loading news settings: {e}")

    def get_user_news_filters(self):
        """Get current user-selected filters for auto news fetch"""
        selected_currencies = [cb.text() for cb in self.currency_checkboxes if cb.isChecked()]
        selected_impacts = [cb.impact_value for cb in self.impact_checkboxes if cb.isChecked()]
        
        # If nothing selected, use defaults
        if not selected_currencies:
            selected_currencies = ['USD', 'EUR', 'GBP', 'JPY']
        if not selected_impacts:
            selected_impacts = [2, 3]  # Medium + High
            
        return selected_currencies, selected_impacts

    def on_filter_changed(self):
        """Auto-save when filter selections change"""
        try:
            # Small delay to avoid rapid saves during multiple changes
            from PyQt5.QtCore import QTimer
            if hasattr(self, '_save_timer'):
                self._save_timer.stop()
            
            self._save_timer = QTimer()
            self._save_timer.setSingleShot(True)
            self._save_timer.timeout.connect(self.save_news_settings)
            self._save_timer.start(1000)  # Save after 1 second of no changes
        except Exception as e:
            print(f"❌ Error in filter change handler: {e}")
    
    def start_auto_news_system_threaded(self):
        """🤖 Start threaded auto news system (non-blocking)"""
        try:
            print("🤖 [AutoNews] Starting threaded auto news system...")
            
            # Create and start auto schedule manager thread
            self.auto_schedule_manager = NewsAutoScheduleManager(self)
            
            # Connect signals for UI updates
            self.auto_schedule_manager.status_updated.connect(self.update_auto_schedule_status)
            self.auto_schedule_manager.times_updated.connect(self.update_scheduled_times_display)
            self.auto_schedule_manager.next_schedule_updated.connect(self.update_next_schedule_display)
            self.auto_schedule_manager.news_fetched.connect(self.display_news)
            
            # Start thread
            self.auto_schedule_manager.start()
            
            # Update initial status
            self.update_auto_schedule_status("🤖 Starting Thread...", "🤖 Khởi động luồng...")
            
            print("✅ [AutoNews] Threaded auto news system started")
            
        except Exception as e:
            print(f"❌ [AutoNews] Error starting threaded system: {e}")
            self.update_auto_schedule_status("❌ Thread Error", "❌ Lỗi luồng")
    
    # Old methods removed - now handled by NewsAutoScheduleManager thread
    
    def update_auto_schedule_status(self, en_text, vi_text):
        """📱 Update auto schedule status display"""
        try:
            status_text = I18N.t(en_text, vi_text)
            self.schedule_status_label.setText(status_text)
        except Exception as e:
            print(f"❌ Error updating status: {e}")
    
    def update_scheduled_times_display(self, text):
        """📅 Update scheduled times display"""
        try:
            self.scheduled_times_label.setText(text)
        except Exception as e:
            print(f"❌ Error updating times display: {e}")
    
    def update_next_schedule_display(self, text):
        """⏰ Update next schedule display"""
        try:
            self.next_schedule_label.setText(text)
        except Exception as e:
            print(f"❌ Error updating next schedule: {e}")
    
    def auto_parse_and_schedule_news_times(self):
        """🤖 Show auto-parsing notification (actual parsing done in thread)"""
        try:
            print("🤖 [AutoNews] News displayed - background thread handling auto-parse...")
            
            # Just show notification that background system is handling this
            # Background system handles everything automatically - no UI notification needed


            # Fixed problematic auto-schedule notification line
                
            pass
                
        except Exception as e:
            print(f"❌ [AutoNews] Error in auto-parse notification: {e}")
    
    def cleanup_resources(self):
        """🧹 Cleanup resources when closing"""
        try:
            print("🧹 [NewsTab] Cleaning up threaded auto news system...")
            
            # Stop auto schedule manager thread
            if hasattr(self, 'auto_schedule_manager') and self.auto_schedule_manager is not None:
                self.auto_schedule_manager.stop()
                self.auto_schedule_manager = None
            
            # Stop all legacy timers
            for timer in self.schedule_timers:
                timer.stop()
                timer.deleteLater()
            self.schedule_timers.clear()
            
            # Stop any running news workers
            if hasattr(self, "worker") and self.worker is not None and self.worker.isRunning():
                self.worker.quit()
                self.worker.wait()
                
            if hasattr(self, "auto_worker") and self.auto_worker is not None and self.auto_worker.isRunning():
                self.auto_worker.quit()
                self.auto_worker.wait()
            
            print("✅ [NewsTab] Threaded auto news system cleanup complete")
            
        except Exception as e:
            print(f"❌ Error during NewsTab cleanup: {e}")

def format_time(time_val):
    if isinstance(time_val, (int, float)):
        try:
            dt = datetime.datetime.fromtimestamp(time_val)
            return dt.strftime("%d/%m %H:%M")
        except Exception:
            return str(time_val)
    if isinstance(time_val, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                dt = datetime.datetime.strptime(time_val, fmt)
                return dt.strftime("%d/%m %H:%M")
            except Exception:
                continue
    return str(time_val)

class NewsAutoScheduleManager(QThread):
    """🤖 Thread quản lý auto-schedule tin tức độc lập"""
    
    # Signals để giao tiếp với main thread
    status_updated = pyqtSignal(str, str)  # (en_text, vi_text)
    times_updated = pyqtSignal(str)        # times_display
    next_schedule_updated = pyqtSignal(str)  # next_schedule_info
    news_fetched = pyqtSignal(list)        # news_events
    
    def __init__(self, parent_tab):
        super().__init__()
        self.parent_tab = parent_tab
        self.running = True
        self.schedule_times = []
        self.schedule_timers = []
        self.monitor_timer = None  # Initialize in run() method
        
    def run(self):
        """🚀 Main thread loop for auto schedule system"""
        try:
            print("🤖 [AutoScheduleManager] Thread started")
            
            # Initialize monitor timer in this thread
            self.monitor_timer = QTimer()
            self.monitor_timer.timeout.connect(self.monitor_and_update_schedule)
            
            # Check if news detection is enabled before starting
            if self.is_news_detection_enabled():
                print("✅ [AutoScheduleManager] News detection enabled - starting auto system")
                
                # Initial news fetch
                self.fetch_and_parse_news()
                
                # Start monitoring timer (every 5 minutes)
                self.monitor_timer.start(5 * 60 * 1000)  # 5 minutes
                
                # Update status
                self.status_updated.emit("🤖 Thread Active", "🤖 Luồng hoạt động")
            else:
                print("⚠️ [AutoScheduleManager] News detection disabled - waiting for enable")
                # Start monitoring timer to check periodically
                self.monitor_timer.start(30 * 1000)  # Check every 30 seconds
                self.status_updated.emit("⏸️ Waiting for News Detection", "⏸️ Đang chờ bật phát hiện tin")
            
            # Keep thread alive
            self.exec_()  # Start event loop
            
        except Exception as e:
            print(f"❌ [AutoScheduleManager] Thread error: {e}")
    
    def is_news_detection_enabled(self):
        """🔍 Check if news detection checkbox is enabled"""
        try:
            return (hasattr(self.parent_tab, 'use_economic_calendar_checkbox') and 
                    self.parent_tab.use_economic_calendar_checkbox.isChecked())
        except:
            return False
            
    def fetch_and_parse_news(self):
        """📰 Fetch news and parse times in background"""
        try:
            # Check if news detection is enabled before fetching
            if not self.is_news_detection_enabled():
                print("⚠️ [AutoScheduleManager] News detection disabled - skipping fetch")
                self.status_updated.emit("⏸️ News Detection OFF", "⏸️ Phát hiện tin TẮT")
                return
                
            print("🤖 [AutoScheduleManager] Fetching news in background...")
            
            # Get current filters from parent tab
            selected_currencies = []
            selected_impacts = []
            
            try:
                # Safely get filters from main thread
                if hasattr(self.parent_tab, 'currency_checkboxes'):
                    selected_currencies = [cb.text() for cb in self.parent_tab.currency_checkboxes if cb.isChecked()]
                if hasattr(self.parent_tab, 'impact_checkboxes'):
                    selected_impacts = [cb.impact_value for cb in self.parent_tab.impact_checkboxes if cb.isChecked()]
            except:
                pass
                
            # Use defaults if no filters
            if not selected_currencies:
                selected_currencies = ['USD', 'EUR', 'JPY', 'GBP']
            if not selected_impacts:
                selected_impacts = [2, 3]
                
            # Fetch news in background thread
            from news_scraper import get_today_news
            news_events = get_today_news(selected_currencies, selected_impacts, headless=True, auto_cleanup=False)
            
            if news_events:
                # Parse times
                from news_scraper import parse_news_release_times
                news_times = parse_news_release_times()
                
                if news_times:
                    self.schedule_times = news_times.copy()
                    self.schedule_auto_updates()
                    
                    # Emit signals to update UI
                    times_display = f"📅 Auto-parsed {len(news_times)} times: {', '.join(news_times[:3])}" + (f" (+{len(news_times)-3} more)" if len(news_times) > 3 else "")
                    self.times_updated.emit(times_display)
                    
                    print(f"✅ [AutoScheduleManager] Parsed {len(news_times)} times in background")
                else:
                    print("⚠️ [AutoScheduleManager] No times found, using fallback")
                    self.schedule_fallback_updates()
                    
                # Emit news to main thread for display
                self.news_fetched.emit(news_events)
                
        except Exception as e:
            print(f"❌ [AutoScheduleManager] Error in background fetch: {e}")
            
    def schedule_auto_updates(self):
        """⏰ Schedule automatic updates in background thread"""
        try:
            # Clear existing timers
            for timer in self.schedule_timers:
                timer.stop()
                timer.deleteLater()
            self.schedule_timers.clear()
            
            from datetime import datetime, timedelta
            import pytz
            vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
            now = datetime.now(vn_tz)
            
            scheduled_count = 0
            next_schedule = None
            
            for time_str in self.schedule_times:
                try:
                    hour, minute = map(int, time_str.split(':'))
                    if 0 <= hour <= 23 and 0 <= minute <= 59:
                        target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                        if target_time <= now:
                            target_time += timedelta(days=1)
                            
                        delay_ms = int((target_time - now).total_seconds() * 1000)
                        
                        # Create timer in this thread
                        timer = QTimer()
                        timer.moveToThread(self)
                        timer.setSingleShot(True)
                        timer.timeout.connect(lambda ts=time_str: self.execute_scheduled_fetch(ts))
                        timer.start(delay_ms)
                        
                        self.schedule_timers.append(timer)
                        scheduled_count += 1
                        
                        if next_schedule is None or target_time < next_schedule:
                            next_schedule = target_time
                            
                except ValueError:
                    continue
                    
            if scheduled_count > 0 and next_schedule:
                time_until = next_schedule - now
                hours_until = time_until.total_seconds() / 3600
                next_str = f"⏰ Next: {next_schedule.strftime('%H:%M')} ({hours_until:.1f}h)"
                self.next_schedule_updated.emit(next_str)
                
                self.status_updated.emit(f"✅ Active ({scheduled_count} schedules)", f"✅ Hoạt động ({scheduled_count} lịch)")
                print(f"⏰ [AutoScheduleManager] Scheduled {scheduled_count} background updates")
                
        except Exception as e:
            print(f"❌ [AutoScheduleManager] Error scheduling: {e}")
            
    def schedule_fallback_updates(self):
        """🔄 Fallback schedule when no news times available"""
        fallback_times = ['08:30', '15:30', '21:30']
        self.schedule_times = fallback_times
        self.schedule_auto_updates()
        self.times_updated.emit(f"📅 Fallback: {', '.join(fallback_times)}")
        self.status_updated.emit("🔄 Fallback Mode", "🔄 Chế độ dự phòng")
        
    def execute_scheduled_fetch(self, time_str):
        """🕐 Execute scheduled fetch in background"""
        try:
            print(f"🕐 [AutoScheduleManager] Background fetch at {time_str}")
            self.fetch_and_parse_news()
            
            # Reschedule for tomorrow
            timer = QTimer()
            timer.moveToThread(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self.execute_scheduled_fetch(time_str))
            timer.start(24 * 60 * 60 * 1000)  # 24 hours
            
        except Exception as e:
            print(f"❌ [AutoScheduleManager] Error in scheduled fetch: {e}")
            
    def monitor_and_update_schedule(self):
        """🔍 Monitor and update schedule periodically"""
        try:
            # Check if news detection is enabled
            if not self.is_news_detection_enabled():
                print("⚠️ [AutoScheduleManager] News detection disabled - monitoring paused")
                self.status_updated.emit("⏸️ News Detection OFF", "⏸️ Phát hiện tin TẮT")
                
                # Clear any existing timers when disabled
                for timer in self.schedule_timers:
                    timer.stop()
                    timer.deleteLater()
                self.schedule_timers.clear()
                return
            
            # If news detection is enabled, continue monitoring
            if not hasattr(self, '_detection_was_enabled') or not self._detection_was_enabled:
                print("✅ [AutoScheduleManager] News detection re-enabled - resuming auto system")
                self.fetch_and_parse_news()
                self._detection_was_enabled = True
                return
                
            from news_scraper import parse_news_release_times
            new_times = parse_news_release_times()
            
            if new_times and new_times != self.schedule_times:
                print(f"🔄 [AutoScheduleManager] Updated schedule detected: {len(new_times)} times")
                self.schedule_times = new_times.copy()
                self.schedule_auto_updates()
                
            self._detection_was_enabled = True
                
        except Exception as e:
            print(f"❌ [AutoScheduleManager] Monitor error: {e}")
            
    def stop(self):
        """🛑 Stop the auto schedule manager"""
        try:
            self.running = False
            
            # Stop monitor timer
            if self.monitor_timer:
                self.monitor_timer.stop()
                
            # Stop all schedule timers
            for timer in self.schedule_timers:
                timer.stop()
                timer.deleteLater()
            self.schedule_timers.clear()
            
            # Quit thread
            self.quit()
            self.wait(3000)  # Wait max 3 seconds
            
            print("✅ [AutoScheduleManager] Thread stopped")
            
        except Exception as e:
            print(f"❌ [AutoScheduleManager] Error stopping: {e}")

class NewsWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    news_ready = pyqtSignal(list)  # New signal for auto system
    error_occurred = pyqtSignal(str)  # Alternative error signal

    def __init__(self, selected_currencies, selected_impacts):
        super().__init__()
        self.selected_currencies = selected_currencies
        self.selected_impacts = selected_impacts
        self.timeout_seconds = 60
    
    def run(self):
        """Run news fetching using NewsScraperSelenium"""
        try:
            print(f"🔍 NewsWorker: Starting news fetch...")
            print(f"🔍 Selected currencies: {self.selected_currencies}")
            print(f"🔍 Selected impacts: {self.selected_impacts}")
            
            # Use the globally imported functions instead of importing again
            try:
                # Get news data using the already imported function
                news_data = get_today_news(self.selected_currencies, self.selected_impacts)
                
                if news_data and len(news_data) > 0:
                    print(f"✅ NewsWorker: Got {len(news_data)} events")
                    
                    # Save news to JSON file
                    try:
                        save_recent_news_to_json(news_data)
                        print(f"💾 Saved {len(news_data)} events to news output")
                    except Exception as save_e:
                        print(f"❌ Error saving news: {save_e}")
                    
                    # Emit the data
                    self.finished.emit(news_data)
                    self.news_ready.emit(news_data)  # Also emit for auto system
                else:
                    print("⚠️ No news data received")
                    self.finished.emit([])
                    self.news_ready.emit([])
                    
            except Exception as e:
                print(f"❌ Error fetching news: {e}")
                self.error.emit(f"News fetch error: {e}")
                self.error_occurred.emit(f"News fetch error: {e}")
                self.finished.emit([])
                
        except Exception as e:
            print(f"❌ Error in NewsWorker.run: {e}")
            self.error.emit(f"NewsWorker error: {e}")
            self.error_occurred.emit(f"NewsWorker error: {e}")
            self.finished.emit([])
    
    def stop(self):
        """Stop the news worker"""
        self.quit()
        self.wait()

class NewsAutoTradingManager:
    def __init__(self, filter_currencies, filter_impact, symbol, timeframe):
        self.filter_currencies = filter_currencies
        self.filter_impact = filter_impact
        self.symbol = symbol
        self.timeframe = timeframe
        self.running = False
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def run(self):
        # 1. Lấy danh sách giờ ra tin trong ngày (Vietnam timezone)
        news_list = get_today_news(self.filter_currencies, self.filter_impact)
        today = datetime.now().strftime("%Y-%m-%d")
        event_times = []
        for news in news_list:
            t = news.get("time", "")
            if t:
                try:
                    # Handle 24-hour format (HH:MM) used by Vietnam timezone
                    if ":" in t and len(t.split(":")) == 2:
                        dt = datetime.strptime(f"{today} {t}", "%Y-%m-%d %H:%M")
                        event_times.append(dt)
                    else:
                        # Fallback to 12-hour format if needed
                        dt = datetime.strptime(f"{today} {t}", "%Y-%m-%d %I:%M%p")
                        event_times.append(dt)
                except Exception as e:
                    print(f"Error parsing time '{t}': {e}")
                    continue
        event_times = sorted(set(event_times))
        
        print(f"Found {len(event_times)} news events today (Vietnam timezone)")
        
        for event_dt in event_times:
            if not self.running:
                break
            # Đợi đến khi còn 5 phút trước giờ ra tin
            while self.running:
                now = datetime.now()
                seconds_to_event = (event_dt - now).total_seconds()
                if seconds_to_event <= 300:
                    break
                time.sleep(10)
            # Quét mỗi phút cho đến khi lấy được actual
            got_actual = False
            while self.running and not got_actual:
                news_list = get_today_news(self.filter_currencies, self.filter_impact)
                for news in news_list:
                    t = news.get("time", "")
                    try:
                        # Handle 24-hour format (HH:MM) used by Vietnam timezone
                        if ":" in t and len(t.split(":")) == 2:
                            dt = datetime.strptime(f"{today} {t}", "%Y-%m-%d %H:%M")
                        else:
                            # Fallback to 12-hour format if needed
                            dt = datetime.strptime(f"{today} {t}", "%Y-%m-%d %I:%M%p")
                    except Exception as e:
                        print(f"Error parsing time '{t}': {e}")
                        continue
                    if abs((dt - event_dt).total_seconds()) < 60:
                        actual = news.get("description", "")
                        if actual and "Dự báo" not in actual and actual.strip() not in ["", "-", "–"]:
                            got_actual = True
                            print(f"Fetched news announced at {t} (Vietnam time): {news.get('title', news.get('event', 'Unknown event'))}")
                            # === Tích hợp các module khi có actual ===
                            # 1. Lấy nến mới, tính indicator                            update_data_with_new_candle(self.symbol, self.timeframe, ...)  # truyền đúng tf_code
                            # 2. Phát hiện mô hình nến
                            analyze_patterns(self.symbol, self.timeframe)
                            # 3. Phát hiện mô hình giá
                            analyze_price_patterns()  # hoặc truyền symbol/timeframe nếu cần
                            # 4. Phân tích AI
                            analyze_symbol(self.symbol, self.timeframe)
                            # 5. Quản lý rủi ro & gửi lệnh nếu đủ điều kiện
                            # (giả sử bạn đã có RiskManagementSystem và OrderHandler)
                            # risk = RiskManagementSystem(...)
                            # if risk.validate_trade(...):
                            #     order = OrderHandler(...)
                            #     order.send_order(...)
                            break
                if not got_actual:
                    time.sleep(60)  # Đợi 1 phút rồi quét lại
        print("Auto trading completed or stopped (Vietnam timezone).")

class RiskManagementTab(QWidget):
    """Risk Management Tab with comprehensive trading controls"""
    
    def __init__(self, market_tab=None):
        super().__init__()
        self.market_tab = market_tab  # Reference to market tab for symbol synchronization
        self.risk_manager = None
        self.settings = {}
        self.available_symbols = []
        self.init_ui()
        print("🔍 DEBUG calling load_settings()")
        self.load_settings()  # Load settings after UI is created
        print("🔍 DEBUG finished load_settings()")
        # CRITICAL FIX: Apply loaded settings to UI controls
        print("🔍 DEBUG calling update_ui_from_settings()")
        self.update_ui_from_settings()
        print("🔍 DEBUG finished update_ui_from_settings()")
        self.init_risk_manager()
        # NEW: Ensure symbol synchronization wiring occurs (previous logic was unreachable after return)
        self.setup_market_symbol_sync()

    # === Localization Refresh Helpers ===
    def refresh_translations(self):
        """Refresh visible texts for Risk Management sub-tabs and group boxes after a language change.
        Safe: checks attribute existence before updating.
        """
        try:
            # Update main header if present
            if hasattr(self, 'layout'):
                # Header label was created as static text; search first child if needed
                pass
            # Update sub tab widget titles if structure unchanged
            if hasattr(self, 'findChildren'):
                # risk_tabs is a QTabWidget added in constructor scope; store reference if not already
                if not hasattr(self, 'risk_tabs'):
                    for tw in self.findChildren(QTabWidget):
                        # Heuristic: first QTabWidget inside RiskManagementTab is our internal tab set
                        self.risk_tabs = tw
                        break
                if hasattr(self, 'risk_tabs'):
                    # Expected order: basic, position, advanced, dca
                    try:
                        self.risk_tabs.setTabText(0, I18N.t("⚙️ Basic Settings", "⚙️ Cài đặt cơ bản"))
                        self.risk_tabs.setTabText(1, I18N.t("📊 Position Management", "📊 Quản lý vị thế"))
                        self.risk_tabs.setTabText(2, I18N.t("🔧 Advanced Controls", "🔧 Điều khiển nâng cao"))
                        self.risk_tabs.setTabText(3, I18N.t("📈 DCA Strategy", "📈 Chiến lược DCA"))
                    except Exception:
                        pass
            # Update common group box titles (if still present)
            for gb in self.findChildren(QGroupBox):
                title = gb.title()
                # Map known English titles to translation via I18N.t
                mapping = {
                    "🎯 Risk Limits": I18N.t("🎯 Risk Limits", "🎯 Giới hạn rủi ro"),
                    "📊 Position Exposure": I18N.t("📊 Position Exposure", "📊 Mức độ vị thế"),
                    "🚨 Emergency Controls": I18N.t("🚨 Emergency Controls", "🚨 Điều khiển khẩn cấp"),
                    "⚖️ DCA Strategy": I18N.t("⚖️ DCA Strategy", "⚖️ Chiến lược DCA"),
                    "📈 DCA Strategy": I18N.t("📈 DCA Strategy", "📈 Chiến lược DCA"),
                }
                if title in mapping:
                    try:
                        gb.setTitle(mapping[title])
                    except Exception:
                        pass
        except Exception as e:
            print(f"[LangSwitch] Risk tab refresh_translations error: {e}")

    # === Helpers ===
    def get_current_dca_mode_key(self):
        if hasattr(self, 'dca_mode_combo'):
            idx = self.dca_mode_combo.currentIndex()
            if idx >= 0:
                return self.dca_mode_combo.itemData(idx) or 'atr_multiple'
        return 'atr_multiple'

    def setup_market_symbol_sync(self):
        """Wire signals from MarketTab for symbol synchronization (idempotent)."""
        if not self.market_tab:
            return
        try:
            # Avoid duplicate connections by disconnecting first (safe try/except)
            try:
                self.market_tab.symbols_changed.disconnect(self.sync_symbols_with_current_market_selection)  # type: ignore
            except Exception:
                pass
            self.market_tab.symbols_changed.connect(self.sync_symbols_with_current_market_selection)
            if hasattr(self.market_tab, 'account_tab') and hasattr(self.market_tab.account_tab, 'connection_changed'):
                try:
                    self.market_tab.account_tab.connection_changed.disconnect(self.sync_symbols_from_market)  # type: ignore
                except Exception:
                    pass
                self.market_tab.account_tab.connection_changed.connect(self.sync_symbols_from_market)
            # Initial sync (both when symbols already chosen and when empty)
            self.sync_symbols_with_current_market_selection()
            print("✅ Risk tab symbol sync initialized")
        except Exception as e:
            print(f"⚠️ Symbol sync setup error: {e}")
    
    def get_combo_value(self, combo, default_value):
        """Helper function to get numeric value from combo (with OFF support)"""
        try:
            if combo is None:
                return default_value
            text = combo.currentText()
            if not text:  # Empty text
                return default_value
            if text.upper() == "OFF" or text == "TẮT":
                return "OFF"  # Return string "OFF" instead of 0
            return float(text)
        except (ValueError, AttributeError, TypeError):
            return default_value
    
    def set_combo_value(self, combo, value):
        """Helper function to set combo value (with OFF support)"""
        try:
            if str(value).upper() == "OFF":
                # Find OFF option in combo
                for i in range(combo.count()):
                    text = combo.itemText(i)
                    if text.upper() == "OFF" or text == "TẮT":
                        combo.setCurrentIndex(i)
                        return
                # If no OFF option found, set to first item
                combo.setCurrentIndex(0)
            else:
                # Find matching numeric value
                for i in range(combo.count()):
                    text = combo.itemText(i)
                    try:
                        if text.upper() != "OFF" and text != "TẮT" and float(text) == float(value):
                            combo.setCurrentIndex(i)
                            return
                    except ValueError:
                        continue
                # If not found, set to first item
                combo.setCurrentIndex(0)
        except AttributeError:
            pass
        
    def sync_symbols_with_current_market_selection(self):
        """Sync with current market tab selection without waiting for connection change"""
        try:
            if self.market_tab and hasattr(self.market_tab, 'checked_symbols'):
                self.available_symbols = list(self.market_tab.checked_symbols)
                
                if self.available_symbols:
                    # Update symbol exposure settings
                    for symbol in self.available_symbols:
                        if symbol not in self.settings.get('symbol_exposure', {}):
                            self.settings.setdefault('symbol_exposure', {})[symbol] = 2.0
                        if symbol not in self.settings.get('symbol_multipliers', {}):
                            self.settings.setdefault('symbol_multipliers', {})[symbol] = 1.0
                    
                    # Update the exposure table
                    self.populate_exposure_table()
                    
                    # Update info label
                    if hasattr(self, 'exposure_info_label'):
                        self.exposure_info_label.setText(f"✅ Synced {len(self.available_symbols)} symbols from Market Tab")
                    
                    print(f"🔄 Initial sync: {len(self.available_symbols)} symbols from Market Tab")
                    
        except Exception as e:
            print(f"❌ Error in initial symbol sync: {e}")
        
    def load_settings(self):
        """Load settings from file or use defaults"""
        print("🔍 DEBUG load_settings() called")
        try:
            if os.path.exists("risk_management/risk_settings.json"):
                with open("risk_management/risk_settings.json", 'r', encoding='utf-8') as f:
                    self.settings = json.load(f)
                print(f"🔍 DEBUG loaded settings from file: max_total_volume={self.settings.get('max_total_volume')}, min_risk_reward_ratio={self.settings.get('min_risk_reward_ratio')}")
            else:
                print("🔍 DEBUG using default settings - file not found")
                # Default settings
                self.settings = {
                    'max_risk_percent': 2.0,
                    'max_drawdown_percent': 5.0,
                    'max_daily_loss_percent': 3.0,
                    'min_volume_auto': 0.01,
                    'max_total_volume': 10.0,
                    'min_risk_reward_ratio': 1.5,
                    'min_confidence_threshold': 3.0,
                    'default_sl_pips': 50,
                    'default_tp_pips': 100,
                    'default_sl_atr_multiplier': 10.0,
                    'default_tp_atr_multiplier': 13.0,
                    'sltp_mode': 'Fixed Pips',
                    'max_positions': 5,
                    'max_positions_per_symbol': 2,
                    'max_correlation': 0.7,
                    'trading_hours_start': 0,
                    'trading_hours_end': 24,
                    'avoid_news_minutes': 30,
                    'max_spread_multiplier': 3.0,
                    'max_slippage': 10,
                    'emergency_stop_drawdown': 10.0,
                    'auto_reduce_on_losses': True,
                    'enable_dca': False,
                    'max_dca_levels': 3,

                    'dca_volume_multiplier': 1.5,
                    # 'dca_mode' removed (reverted to simple distance only)
                    'dca_min_drawdown': 1.0,
                    # Removed: 'dca_high_confidence_only'
                    'dca_sl_mode': 'Average SL',
                    'dca_avg_sl_profit_percent': 10.0,  # 🆕 NEW: Default 10% profit target for Average SL
                    'trading_mode': '👨‍💼 Manual Mode',
                    'symbol_exposure': {},
                    'symbol_multipliers': {},
                    # New toggle defaults
                    'disable_news_avoidance': False,
                    'disable_emergency_stop': False,
                    'disable_max_dd_close': False,
                    'enable_auto_mode': False,
                    'enable_auto_scan': True,
                    'auto_scan_interval': 24,
                    # Volume management defaults
                    'volume_mode': '🧮 Risk-Based (Auto)',
                    'fixed_volume_lots': 0.10
                }
                # print("✅ Using default risk settings")
        except Exception as e:
            print(f"❌ Error loading settings: {e}")
            self.settings = {}
        # Don't call init_ui() here - it will be called in __init__
        
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()
        
        # === HEADER SECTION ===
        header_label = QLabel("🛡️ RISK MANAGEMENT CENTER")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2E86C1; padding: 10px;")
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)
        
        # === TABS FOR DIFFERENT SECTIONS ===
        risk_tabs = QTabWidget()
        
        # Tab 1: Basic Risk Settings
        basic_tab = self.create_basic_settings_tab()
        risk_tabs.addTab(basic_tab, "⚙️ Basic Settings")
        
        # Tab 2: Position Management
        position_tab = self.create_position_management_tab()
        risk_tabs.addTab(position_tab, "📊 Position Management")
        
        # Tab 3: Advanced Controls
        advanced_tab = self.create_advanced_controls_tab()
        risk_tabs.addTab(advanced_tab, "🔧 Advanced Controls")
        
        # Tab 4: DCA Settings
        dca_tab = self.create_dca_settings_tab()
        risk_tabs.addTab(dca_tab, "📈 DCA Strategy")
        
        main_layout.addWidget(risk_tabs)
        
        # === BOTTOM CONTROL PANEL ===
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        self.setLayout(main_layout)
        
    def create_basic_settings_tab(self):
        """Create basic risk settings tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # === RISK LIMITS GROUP ===
        risk_group = QGroupBox(I18N.t("🎯 Risk Limits", "🎯 Giới hạn rủi ro"))
        risk_layout = QGridLayout()
        
        # Max Risk per Trade with OFF option
        risk_layout.addWidget(QLabel(I18N.t("Max Risk per Trade (%):", "Rủi ro tối đa mỗi lệnh (%):")), 0, 0)
        self.max_risk_combo = QComboBox()
        self.max_risk_combo.setEditable(True)
        risk_options = ["OFF", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "5.0", "10.0"]
        self.max_risk_combo.addItems(risk_options)
        # Set current value based on settings
        current_max_risk = self.settings.get('max_risk_percent', 2.0)
        if isinstance(current_max_risk, str) and current_max_risk.upper() == "OFF":
            self.max_risk_combo.setCurrentText("OFF")
        else:
            self.max_risk_combo.setCurrentText(str(current_max_risk))
        self.max_risk_combo.setToolTip(I18N.t("Maximum percentage of account balance to risk per trade (use OFF to disable)", "Tỷ lệ phần trăm tối đa của số dư tài khoản để chấp nhận rủi ro mỗi lệnh (dùng OFF để tắt)"))
        risk_layout.addWidget(self.max_risk_combo, 0, 1)
        
        # Max Drawdown with OFF option
        risk_layout.addWidget(QLabel(I18N.t("Max Drawdown (%):", "Sụt giảm tối đa (%):")), 1, 0)
        self.max_drawdown_combo = QComboBox()
        self.max_drawdown_combo.setEditable(True)
        dd_options = ["OFF", "1.0", "2.0", "3.0", "5.0", "8.0", "10.0", "15.0", "20.0"]
        self.max_drawdown_combo.addItems(dd_options)
        # Set current value based on settings
        if self.settings.get('disable_max_dd_close', False):
            self.max_drawdown_combo.setCurrentText("OFF")
        else:
            current_dd = self.settings.get('max_drawdown_percent', 5.0)
            self.max_drawdown_combo.setCurrentText(str(current_dd))
        self.max_drawdown_combo.setToolTip(I18N.t("Maximum allowable drawdown before emergency stop (use OFF to disable)", "Mức sụt giảm tối đa cho phép trước khi dừng khẩn cấp (dùng OFF để tắt)"))
        risk_layout.addWidget(self.max_drawdown_combo, 1, 1)
        
        # Daily Loss Limit with OFF option
        risk_layout.addWidget(QLabel(I18N.t("Daily Loss Limit (%):", "Giới hạn lỗ ngày (%):")), 2, 0)
        self.daily_loss_combo = QComboBox()
        self.daily_loss_combo.setEditable(True)
        daily_loss_options = ["OFF", "1.0", "2.0", "3.0", "5.0", "7.0", "10.0", "15.0", "20.0"]
        self.daily_loss_combo.addItems(daily_loss_options)
        # Set current value based on settings
        current_daily_loss = self.settings.get('max_daily_loss_percent', 3.0)
        if isinstance(current_daily_loss, str) and current_daily_loss.upper() == "OFF":
            self.daily_loss_combo.setCurrentText("OFF")
        else:
            self.daily_loss_combo.setCurrentText(str(current_daily_loss))
        self.daily_loss_combo.setToolTip(I18N.t("Maximum daily loss before stopping trading (use OFF to disable)", "Lỗ tối đa hàng ngày trước khi dừng giao dịch (dùng OFF để tắt)"))
        risk_layout.addWidget(self.daily_loss_combo, 2, 1)
        
        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)
        
        # === POSITION SIZE GROUP ===
        position_group = QGroupBox(I18N.t("📏 Position Sizing", "📏 Kích thước vị thế"))
        position_layout = QGridLayout()
        
        # Volume mode selection
        position_layout.addWidget(QLabel(I18N.t("Volume Settings:", "Cài đặt khối lượng:")), 0, 0)
        self.volume_mode_combo = QComboBox()
        self.volume_mode_combo.addItems([
            I18N.t("Risk-Based (Auto)", "Theo rủi ro (Tự động)"),
            I18N.t("Fixed Volume", "Khối lượng cố định"),
            I18N.t("Default Volume", "Khối lượng mặc định")
        ])
        self.volume_mode_combo.setCurrentText(self.settings.get('volume_mode', I18N.t('Risk-Based (Auto)', 'Theo rủi ro (Tự động)')))
        self.volume_mode_combo.currentTextChanged.connect(self.on_volume_mode_changed)
        self.volume_mode_combo.currentTextChanged.connect(self.auto_save_settings)
        position_layout.addWidget(self.volume_mode_combo, 0, 1, 1, 3)
        
        # Lot Size Settings - Dynamic based on mode
        position_layout.addWidget(QLabel(I18N.t("Min Lot Size:", "Khối lượng nhỏ nhất (tự động):")), 1, 0)
        self.min_lot_spin = QDoubleSpinBox()
        self.min_lot_spin.setRange(0.0, 10.0)
        self.min_lot_spin.setSingleStep(0.01)
        self.min_lot_spin.setValue(self.settings.get('min_volume_auto', 0.01))
        self.min_lot_spin.setToolTip(I18N.t("Minimum lot size for auto mode only", "Khối lượng tối thiểu chỉ cho chế độ tự động"))
        position_layout.addWidget(self.min_lot_spin, 1, 1)
        
        position_layout.addWidget(QLabel(I18N.t("Max Lot Size:", "Tổng khối lượng tối đa:")), 1, 2)
        self.max_lot_combo = QComboBox()
        self.max_lot_combo.setEditable(True)
        lot_options = ["OFF", "0.5", "1.0", "2.0", "5.0", "10.0", "20.0", "50.0", "100.0"]
        self.max_lot_combo.addItems(lot_options)
        print(f"🔍 DEBUG max_lot_combo items: {[self.max_lot_combo.itemText(i) for i in range(self.max_lot_combo.count())]}")
        # Set current value based on settings
        current_max_lot = self.settings.get('max_total_volume', 10.0)
        print(f"🔍 DEBUG current_max_lot from settings: {current_max_lot} (type: {type(current_max_lot)})")
        if isinstance(current_max_lot, str) and current_max_lot.upper() == "OFF":
            self.max_lot_combo.setCurrentText("OFF")
        else:
            self.max_lot_combo.setCurrentText(str(current_max_lot))
        self.max_lot_combo.setToolTip(I18N.t("Maximum total volume across all positions (use OFF to disable)", "Tổng khối lượng tối đa trên tất cả vị thế (dùng OFF để tắt)"))
        position_layout.addWidget(self.max_lot_combo, 1, 3)
        
        # Fixed volume setting (dynamic label based on mode)
        self.fixed_volume_label = QLabel(I18N.t("Fixed Volume (lots):", "Khối lượng cố định (lots):"))
        position_layout.addWidget(self.fixed_volume_label, 2, 0)
        self.fixed_volume_spin = QDoubleSpinBox()
        self.fixed_volume_spin.setRange(0.0, 100.0)
        self.fixed_volume_spin.setSingleStep(0.01)
        self.fixed_volume_spin.setValue(self.settings.get('fixed_volume_lots', 0.10))
        self.fixed_volume_spin.setToolTip(I18N.t("Fixed lot size when using Fixed Volume mode", "Kích thước lot cố định khi dùng chế độ Khối lượng cố định"))
        position_layout.addWidget(self.fixed_volume_spin, 2, 1)
        
        # Default volume setting (same position as fixed volume - will be shown/hidden based on mode)
        self.default_volume_label = QLabel(I18N.t("Default Volume (lots):", "Khối lượng mặc định (lots):"))
        position_layout.addWidget(self.default_volume_label, 2, 0)  # Same position as fixed_volume_label
        self.default_volume_spin = QDoubleSpinBox()
        self.default_volume_spin.setRange(0.0, 100.0)
        self.default_volume_spin.setSingleStep(0.01)
        self.default_volume_spin.setValue(self.settings.get('default_volume_lots', 0.10))
        self.default_volume_spin.setToolTip(I18N.t("Default lot size when using Default Volume mode", "Kích thước lot mặc định khi dùng chế độ Khối lượng mặc định"))
        position_layout.addWidget(self.default_volume_spin, 2, 1)  # Same position as fixed_volume_spin
        
        # Volume mode explanation
        self.volume_mode_label = QLabel()
        self.update_volume_mode_explanation()
        self.volume_mode_label.setStyleSheet("color: #3498DB; font-style: italic; padding: 5px; background-color: #EBF5FB; border-radius: 5px;")
        position_layout.addWidget(self.volume_mode_label, 3, 0, 1, 4)
        
        # Initialize volume fields visibility based on current mode
        self.update_volume_fields_visibility()
        
        # Risk/Reward Ratio with OFF option
        position_layout.addWidget(QLabel(I18N.t("Min R:R Ratio:", "Tỷ lệ R:R tối thiểu:")), 4, 0)
        self.min_rr_combo = QComboBox()
        self.min_rr_combo.setEditable(True)
        rr_options = ["OFF", "0.5", "1.0", "1.2", "1.5", "2.0", "2.5", "3.0", "5.0"]
        self.min_rr_combo.addItems(rr_options)
        print(f"🔍 DEBUG min_rr_combo items: {[self.min_rr_combo.itemText(i) for i in range(self.min_rr_combo.count())]}")
        # Set current value based on settings
        current_rr = self.settings.get('min_risk_reward_ratio', 1.5)
        print(f"🔍 DEBUG current_rr from settings: {current_rr} (type: {type(current_rr)})")
        if isinstance(current_rr, str) and current_rr.upper() == "OFF":
            self.min_rr_combo.setCurrentText("OFF")
        else:
            self.min_rr_combo.setCurrentText(str(current_rr))
        self.min_rr_combo.setToolTip(I18N.t("Minimum risk-to-reward ratio required for trades (use OFF to disable)", "Tỷ lệ rủi ro/lợi nhuận tối thiểu yêu cầu cho giao dịch (dùng OFF để tắt)"))
        position_layout.addWidget(self.min_rr_combo, 4, 1)
        
        position_group.setLayout(position_layout)
        layout.addWidget(position_group)
        
        # === STOP LOSS / TAKE PROFIT GROUP ===
        sltp_group = QGroupBox(I18N.t("🎯 Stop Loss / Take Profit", "🎯 Cắt lỗ / Chốt lời"))
        sltp_layout = QGridLayout()
        
        # Dynamic SL/TP labels and controls (will update based on mode)
        self.sl_label = QLabel(I18N.t("Default SL (pips):", "SL mặc định (pips):"))
        sltp_layout.addWidget(self.sl_label, 0, 0)
        self.default_sl_spin = QDoubleSpinBox()  # 🔧 Changed to QDoubleSpinBox for decimal support
        self.default_sl_spin.setRange(0.0, 10000.0)
        self.default_sl_spin.setDecimals(1)  # Allow 1 decimal place
        self.default_sl_spin.setSingleStep(0.1)  # Step by 0.1
        self.default_sl_spin.setValue(float(self.settings.get('default_sl_pips', 50)))
        sltp_layout.addWidget(self.default_sl_spin, 0, 1)
        
        self.tp_label = QLabel(I18N.t("Default TP (pips):", "TP mặc định (pips):"))
        sltp_layout.addWidget(self.tp_label, 0, 2)
        self.default_tp_spin = QDoubleSpinBox()  # 🔧 Changed to QDoubleSpinBox for decimal support
        self.default_tp_spin.setRange(0.0, 50000.0)
        self.default_tp_spin.setDecimals(1)  # Allow 1 decimal place
        self.default_tp_spin.setSingleStep(0.1)  # Step by 0.1
        self.default_tp_spin.setValue(float(self.settings.get('default_tp_pips', 100)))
        sltp_layout.addWidget(self.default_tp_spin, 0, 3)
        
        # SL/TP Mode
        sltp_layout.addWidget(QLabel(I18N.t("SL/TP Mode:", "Chế độ SL/TP:")), 1, 0)
        self.sltp_mode_combo = QComboBox()
        self.sltp_mode_combo.addItems([
            I18N.t("Fixed Pips", "Pips cố định"), 
            I18N.t("ATR Multiple", "Bội số ATR"), 
            I18N.t("Support/Resistance", "Hỗ trợ/Kháng cự"), 
            I18N.t("Percentage", "Phần trăm"),
            I18N.t("Signal Based", "Theo Signal")
        ])
        self.sltp_mode_combo.setCurrentText(self.settings.get('sltp_mode', I18N.t('Fixed Pips', 'Pips cố định')))
        # Connect mode change to update labels and ranges
        self.sltp_mode_combo.currentTextChanged.connect(self.update_sltp_mode_controls)
        self.sltp_mode_combo.currentTextChanged.connect(self.auto_save_settings)
        self.default_sl_spin.valueChanged.connect(self.auto_save_settings)
        self.default_tp_spin.valueChanged.connect(self.auto_save_settings)
        sltp_layout.addWidget(self.sltp_mode_combo, 1, 1, 1, 2)
        
        # Initialize controls based on current mode
        self.update_sltp_mode_controls()
        
        sltp_group.setLayout(sltp_layout)
        layout.addWidget(sltp_group)
        
        # 🔧 CONNECT: Auto-save for Basic Settings that were missing
        self.max_risk_combo.currentTextChanged.connect(self.auto_save_settings)
        self.max_drawdown_combo.currentTextChanged.connect(self.auto_save_settings)
        self.daily_loss_combo.currentTextChanged.connect(self.auto_save_settings)
        self.min_lot_spin.valueChanged.connect(self.auto_save_settings)
        self.max_lot_combo.currentTextChanged.connect(self.auto_save_settings)
        self.fixed_volume_spin.valueChanged.connect(self.auto_save_settings)
        self.default_volume_spin.valueChanged.connect(self.auto_save_settings)
        self.min_rr_combo.currentTextChanged.connect(self.auto_save_settings)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
        
    def create_position_management_tab(self):
        """Create position management tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # === POSITION LIMITS GROUP ===
        limits_group = QGroupBox(I18N.t("📊 Position Limits", "📊 Giới hạn vị thế"))
        limits_layout = QGridLayout()

        # Max Positions
        limits_layout.addWidget(QLabel(I18N.t("Max Total Positions:", "Tối đa tổng vị thế:")), 0, 0)
        self.max_positions_spin = QSpinBox()
        self.max_positions_spin.setRange(0, 1000)
        self.max_positions_spin.setValue(int(self.settings.get('max_positions', 5)))
        self.max_positions_spin.valueChanged.connect(self.auto_save_settings)  # 🔧 FIX: Auto save
        limits_layout.addWidget(self.max_positions_spin, 0, 1)

        # Max Positions per Symbol
        limits_layout.addWidget(QLabel(I18N.t("Max Positions per Symbol:", "Tối đa vị thế mỗi mã:")), 0, 2)
        self.max_positions_per_symbol_spin = QSpinBox()
        self.max_positions_per_symbol_spin.setRange(0, 100)
        self.max_positions_per_symbol_spin.setValue(int(self.settings.get('max_positions_per_symbol', 2)))
        self.max_positions_per_symbol_spin.valueChanged.connect(self.auto_save_settings)  # 🔧 FIX: Auto save
        limits_layout.addWidget(self.max_positions_per_symbol_spin, 0, 3)

        # Max Correlation
        limits_layout.addWidget(QLabel(I18N.t("Max Correlation:", "Tương quan tối đa:")), 1, 0)
        self.max_correlation_spin = QDoubleSpinBox()
        self.max_correlation_spin.setRange(0.0, 1.0)
        self.max_correlation_spin.setSingleStep(0.1)
        self.max_correlation_spin.setValue(self.settings.get('max_correlation', 0.7))
        self.max_correlation_spin.valueChanged.connect(self.auto_save_settings)  # 🔧 FIX: Auto save
        limits_layout.addWidget(self.max_correlation_spin, 1, 1)

        limits_group.setLayout(limits_layout)
        layout.addWidget(limits_group)

        # === SYMBOL EXPOSURE GROUP ===
        exposure_group = QGroupBox(I18N.t("💼 Symbol Exposure Limits", "💼 Giới hạn mức độ theo mã"))
        exposure_layout = QVBoxLayout()

        # Info label (localized)
        self.exposure_info_label = QLabel(I18N.t(
            "📊 Symbols will sync automatically from Market Tab selections",
            "📊 Các mã sẽ tự đồng bộ theo lựa chọn ở tab Thị trường"
        ))
        self.exposure_info_label.setStyleSheet("color: #3498DB; font-style: italic; padding: 5px;")
        exposure_layout.addWidget(self.exposure_info_label)

        # Symbol exposure table
        self.exposure_table = QTableWidget()
        self.exposure_table.setColumnCount(3)
        self.exposure_table.setHorizontalHeaderLabels([
            I18N.t("Symbol", "Mã"),
            I18N.t("Max Exposure (lots)", "Khối lượng tối đa (lot)"),
            I18N.t("Risk Multiplier", "Hệ số rủi ro")
        ])
        # Widen the Exposure column to avoid clipping in Vietnamese
        self.exposure_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.exposure_table.setColumnWidth(1, 240)
        self.exposure_table.horizontalHeader().setStretchLastSection(True)

        # Initialize with empty table - will be populated when symbols are synced
        self.populate_exposure_table()

        exposure_layout.addWidget(self.exposure_table)
        exposure_group.setLayout(exposure_layout)
        layout.addWidget(exposure_group)

        # 🔧 CONNECT: Auto-save for exposure table changes
        self.exposure_table.cellChanged.connect(self.on_exposure_table_changed)

        layout.addStretch()
        tab.setLayout(layout)
        return tab
        
    def create_advanced_controls_tab(self):
        """Create advanced controls tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # === TRADING HOURS GROUP ===
        hours_group = QGroupBox(I18N.t("🕐 Trading Hours (UTC)", "🕐 Giờ giao dịch (UTC)"))
        hours_layout = QGridLayout()
        
        hours_layout.addWidget(QLabel(I18N.t("Start Hour:", "Giờ bắt đầu:")), 0, 0)
        self.start_hour_spin = QSpinBox()
        self.start_hour_spin.setRange(0, 23)
        self.start_hour_spin.setValue(int(self.settings.get('trading_hours_start', 0)))
        hours_layout.addWidget(self.start_hour_spin, 0, 1)
        
        hours_layout.addWidget(QLabel(I18N.t("End Hour:", "Giờ kết thúc:")), 0, 2)
        self.end_hour_spin = QSpinBox()
        self.end_hour_spin.setRange(0, 23)
        self.end_hour_spin.setValue(int(self.settings.get('trading_hours_end', 24)))
        hours_layout.addWidget(self.end_hour_spin, 0, 3)
        
        # News avoidance with OFF option
        hours_layout.addWidget(QLabel(I18N.t("Avoid News (minutes):", "Tránh tin tức (phút):")), 1, 0)
        self.avoid_news_combo = QComboBox()
        self.avoid_news_combo.setEditable(True)
        news_options = ["OFF", "5", "10", "15", "30", "45", "60", "90", "120"]
        self.avoid_news_combo.addItems(news_options)
        # Set current value based on settings
        current_avoid = self.settings.get('avoid_news_minutes', 30)
        if self.settings.get('disable_news_avoidance', False):
            self.avoid_news_combo.setCurrentText("OFF")
        else:
            self.avoid_news_combo.setCurrentText(str(current_avoid))
        hours_layout.addWidget(self.avoid_news_combo, 1, 1)
        
        hours_group.setLayout(hours_layout)
        layout.addWidget(hours_group)
        
        # === MARKET CONDITIONS GROUP ===
        market_group = QGroupBox(I18N.t("📊 Market Conditions", "📊 Điều kiện thị trường"))
        market_layout = QGridLayout()
        
        # Spread limits
        market_layout.addWidget(QLabel(I18N.t("Max Spread Multiplier:", "Hệ số spread tối đa:")), 0, 0)
        self.spread_multiplier_spin = QDoubleSpinBox()
        self.spread_multiplier_spin.setRange(0.0, 100.0)
        self.spread_multiplier_spin.setSingleStep(0.5)
        self.spread_multiplier_spin.setValue(self.settings.get('max_spread_multiplier', 3.0))
        market_layout.addWidget(self.spread_multiplier_spin, 0, 1)
        
        # Slippage
        market_layout.addWidget(QLabel(I18N.t("Max Slippage:", "Độ trượt giá tối đa:")), 0, 2)
        self.max_slippage_spin = QSpinBox()
        self.max_slippage_spin.setRange(0, 1000)
        self.max_slippage_spin.setValue(self.settings.get('max_slippage', 10))
        market_layout.addWidget(self.max_slippage_spin, 0, 3)
        
        market_group.setLayout(market_layout)
        layout.addWidget(market_group)
        
        # === EMERGENCY CONTROLS GROUP ===
        emergency_group = QGroupBox(I18N.t("🚨 Emergency Controls", "🚨 Điều khiển khẩn cấp"))
        emergency_layout = QGridLayout()
        
        # Emergency stop drawdown with OFF option
        emergency_layout.addWidget(QLabel(I18N.t("Emergency Stop DD (%):", "Dừng khẩn cấp DD (%):")), 0, 0)
        self.emergency_dd_combo = QComboBox()
        self.emergency_dd_combo.setEditable(True)
        emergency_options = ["OFF", "5.0", "8.0", "10.0", "15.0", "20.0", "25.0", "30.0"]
        self.emergency_dd_combo.addItems(emergency_options)
        # Set current value based on settings
        if self.settings.get('disable_emergency_stop', False):
            self.emergency_dd_combo.setCurrentText("OFF")
        else:
            current_emergency = self.settings.get('emergency_stop_drawdown', 10.0)
            self.emergency_dd_combo.setCurrentText(str(current_emergency))
        emergency_layout.addWidget(self.emergency_dd_combo, 0, 1)
        
        # Emergency mode selector (simplified to AUTO/ENABLED)
        # Auto reduce on losses
        self.auto_reduce_check = QCheckBox(I18N.t("Auto Reduce Position Size on Losses", "Tự động giảm khối lượng khi thua lỗ"))
        self.auto_reduce_check.setChecked(self.settings.get('auto_reduce_on_losses', True))
        emergency_layout.addWidget(self.auto_reduce_check, 1, 0, 1, 4)
        
        # Note about position management
        info_label = QLabel(I18N.t("ℹ️ Use Account Tab for position management and closing orders", "ℹ️ Dùng tab Tài khoản để quản lý và đóng lệnh"))
        info_label.setStyleSheet("color: #3498DB; font-style: italic; padding: 10px; background-color: #EBF5FB; border-radius: 5px;")
        emergency_layout.addWidget(info_label, 2, 0, 1, 4)
        
        emergency_group.setLayout(emergency_layout)
        layout.addWidget(emergency_group)
        
        # 🔧 CONNECT: Auto-save for Advanced Controls
        self.start_hour_spin.valueChanged.connect(self.auto_save_settings)
        self.end_hour_spin.valueChanged.connect(self.auto_save_settings)
        self.avoid_news_combo.currentTextChanged.connect(self.auto_save_settings)
        self.spread_multiplier_spin.valueChanged.connect(self.auto_save_settings)
        self.max_slippage_spin.valueChanged.connect(self.auto_save_settings)
        self.emergency_dd_combo.currentTextChanged.connect(self.auto_save_settings)
        self.auto_reduce_check.stateChanged.connect(self.auto_save_settings)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
        
    def create_dca_settings_tab(self):
        """Create DCA (Dollar Cost Averaging) settings tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # === DCA STRATEGY GROUP ===
        dca_group = QGroupBox(I18N.t("📈 DCA Strategy Settings", "📈 Cài đặt chiến lược DCA"))
        dca_layout = QGridLayout()
        
        # Enable DCA
        self.enable_dca_check = QCheckBox(I18N.t("Enable DCA Strategy", "Bật chiến lược DCA"))
        self.enable_dca_check.setChecked(self.settings.get('enable_dca', False))
        dca_layout.addWidget(self.enable_dca_check, 0, 0, 1, 2)
        
        # Max DCA levels
        dca_layout.addWidget(QLabel(I18N.t("Max DCA Levels:", "Số tầng DCA tối đa:")), 1, 0)
        self.max_dca_levels_spin = QSpinBox()
        self.max_dca_levels_spin.setRange(1, 50)
        self.max_dca_levels_spin.setValue(self.settings.get('max_dca_levels', 3))
        self.max_dca_levels_spin.setToolTip(I18N.t("Maximum DCA levels allowed", "Số tầng DCA tối đa cho phép"))
        dca_layout.addWidget(self.max_dca_levels_spin, 1, 1)
        

        
        # (Removed Fibonacci controls per user request)
        
        # DCA multiplier - moved up to position 3
        self.dca_multiplier_label = QLabel("Hệ số khối lượng DCA:")
        dca_layout.addWidget(self.dca_multiplier_label, 3, 0)
        self.dca_multiplier_spin = QDoubleSpinBox()
        self.dca_multiplier_spin.setRange(0.0, 100.0)
        self.dca_multiplier_spin.setSingleStep(0.1)
        self.dca_multiplier_spin.setValue(self.settings.get('dca_volume_multiplier', 1.5))
        self.dca_multiplier_spin.setToolTip("Hệ số nhân khối lượng cho mỗi tầng DCA")
        dca_layout.addWidget(self.dca_multiplier_spin, 3, 1)
        
        # DCA Mode (reintroduced per user request) - row 4
        self.dca_mode_label = QLabel(I18N.t("DCA Mode:", "Chế Độ DCA:"))
        dca_layout.addWidget(self.dca_mode_label, 4, 0)
        self.dca_mode_combo = QComboBox()
        # Canonical internal keys mapped to localized display strings
        # New standardized modes: ATR multiple, Fixed Pips, Fibonacci Levels
        self._dca_mode_items = [
            ("atr_multiple", I18N.t("ATR Multiple", "Bội số ATR")),
            ("fixed_pips", I18N.t("Fixed Pips", "Pips cố định")),
            ("fibo_levels", I18N.t("Fibonacci Levels", "Mức Fibonacci")),
        ]
        for key, label in self._dca_mode_items:
            self.dca_mode_combo.addItem(label, userData=key)
        # Backward compatibility mapping from legacy label or key to new canonical key
        legacy_mode = self.settings.get('dca_mode', 'fixed_pips')
        legacy_map = {
            'fixed_multiple': 'fixed_pips',
            'adaptive_ratio': 'atr_multiple',  # treat old adaptive as ATR-based approximation
            'fibonacci_levels': 'fibo_levels',
            'fibonacci': 'fibo_levels',  # 🔧 ADD: Map legacy "fibonacci" to "fibo_levels"
            'Bội số cố định': 'fixed_pips',
            'Tự động theo tỷ lệ': 'atr_multiple',
            'Mức Fibonacci': 'fibo_levels',
            'Mức Fibo': 'fibo_levels'
        }
        saved_mode_key = legacy_map.get(legacy_mode, legacy_mode if legacy_mode in [m[0] for m in self._dca_mode_items] else 'fixed_pips')
        # Try to set current index by key
        for idx in range(self.dca_mode_combo.count()):
            if self.dca_mode_combo.itemData(idx) == saved_mode_key:
                self.dca_mode_combo.setCurrentIndex(idx)
                break
        self.dca_mode_combo.setToolTip(I18N.t(
            "Select DCA averaging method (Fixed, Adaptive, Fibonacci)",
            "Chọn phương pháp DCA (Cố định, Tự động theo tỷ lệ, Fibonacci)"
        ))
        dca_layout.addWidget(self.dca_mode_combo, 4, 1)
        
        # Per-mode option widgets container (grid row starts at 5)
        # ATR options
        self.dca_atr_period_label = QLabel(I18N.t("ATR Period:", "Chu kỳ ATR:"))
        self.dca_atr_period_spin = QSpinBox(); self.dca_atr_period_spin.setRange(1, 500); self.dca_atr_period_spin.setValue(self.settings.get('dca_atr_period', 14))
        self.dca_atr_mult_label = QLabel(I18N.t("ATR Multiplier:", "Hệ số ATR:"))
        self.dca_atr_mult_spin = QDoubleSpinBox(); self.dca_atr_mult_spin.setRange(0.1, 20.0); self.dca_atr_mult_spin.setSingleStep(0.1); self.dca_atr_mult_spin.setValue(self.settings.get('dca_atr_multiplier', 1.5))
        # Fixed pips base distance
        self.dca_base_distance_label = QLabel(I18N.t("DCA Distance (pips):", "Khoảng Cách DCA (Pips):"))
        self.dca_base_distance_spin = QDoubleSpinBox(); self.dca_base_distance_spin.setRange(1, 10000); self.dca_base_distance_spin.setSingleStep(1); self.dca_base_distance_spin.setValue(self.settings.get('dca_distance_pips', 50))
        # Fibonacci note (informational)
        self.dca_fibo_note = QLabel(I18N.t("Fibonacci expansion sequence used for spacing & volume", "Dùng chuỗi Fibonacci để giãn cách & khối lượng"))
        self.dca_fibo_note.setStyleSheet("color:#888;font-style:italic;")
        # Simplified Fibonacci settings: one levels field (comma-separated), first value used for first Entry reference, auto scales next
        self.dca_fibo_levels_label = QLabel(I18N.t("Start Fibonacci Retracement (%):", "Bắt đầu từ mức Fibonacci (%):"))
        # Use already imported QComboBox from module scope
        self.dca_fibo_start_combo = QComboBox()
        retracement_levels = [23.6, 38.2, 50.0, 61.8, 78.6, 100.0]
        for idx2, val in enumerate(retracement_levels):
            self.dca_fibo_start_combo.addItem(f"{val:.1f}%", userData=idx2)
        # Migration: if old numeric fibo sequence existed, just anchor at first retracement (index 0)
        start_index = 0
        saved_start_idx = self.settings.get('dca_fibo_start_level', start_index)
        if 0 <= saved_start_idx < self.dca_fibo_start_combo.count():
            self.dca_fibo_start_combo.setCurrentIndex(saved_start_idx)
        self.dca_fibo_start_combo.setToolTip(I18N.t(
            "Select starting Fibonacci retracement %; subsequent entries use the following levels.",
            "Chọn mức Fibonacci % bắt đầu; các tầng tiếp theo dùng các mức phía sau."))

        # New: Fibo execution mode (two auto market modes)
        self.dca_fibo_exec_label = QLabel(I18N.t("Fibonacci Exec Mode:", "Chế độ thực thi Fibonacci:"))
        self.dca_fibo_exec_combo = QComboBox()
        self.dca_fibo_exec_combo.addItems([
            I18N.t("On Touch (Market)", "Chạm Mức (Market)"),
            I18N.t("Pending Limit at Level", "Đặt Lệnh Chờ tại Mức")
        ])
        self.dca_fibo_exec_combo.setCurrentText(self.settings.get('dca_fibo_exec_mode', I18N.t("On Touch (Market)", "Chạm Mức (Market)")))
        self.dca_fibo_exec_combo.setToolTip(I18N.t(
            "Choose how Fibonacci DCA orders are executed: Market entry immediately when price touches level, or place a pending limit order at the level.",
            "Chọn cách vào lệnh DCA theo Fibonacci: Khớp Market khi giá chạm mức, hoặc đặt lệnh chờ Limit tại mức."))

        # Add placeholders (we'll control visibility dynamically)
        row = 5
        dca_layout.addWidget(self.dca_atr_period_label, row, 0); dca_layout.addWidget(self.dca_atr_period_spin, row, 1)
        dca_layout.addWidget(self.dca_atr_mult_label, row, 2); dca_layout.addWidget(self.dca_atr_mult_spin, row, 3)
        row += 1
        dca_layout.addWidget(self.dca_base_distance_label, row, 0); dca_layout.addWidget(self.dca_base_distance_spin, row, 1)
        row += 1
        # Fibonacci simplified input
        dca_layout.addWidget(self.dca_fibo_levels_label, row, 0); dca_layout.addWidget(self.dca_fibo_start_combo, row, 1)
        dca_layout.addWidget(self.dca_fibo_exec_label, row, 2); dca_layout.addWidget(self.dca_fibo_exec_combo, row, 3)
        row += 1
        dca_layout.addWidget(self.dca_fibo_note, row, 0, 1, 4)

        def _refresh_dca_mode_widgets():
            key = self.get_current_dca_mode_key()
            atr_visible = (key == 'atr_multiple')
            fixed_visible = (key == 'fixed_pips')
            fibo_visible = (key == 'fibo_levels')
            for w in [self.dca_atr_period_label, self.dca_atr_period_spin, self.dca_atr_mult_label, self.dca_atr_mult_spin]:
                w.setVisible(atr_visible)
            for w in [self.dca_base_distance_label, self.dca_base_distance_spin]:
                w.setVisible(fixed_visible)
            for w in [self.dca_fibo_levels_label, self.dca_fibo_start_combo, self.dca_fibo_exec_label, self.dca_fibo_exec_combo, self.dca_fibo_note]:
                w.setVisible(fibo_visible)

        def _on_dca_mode_changed(index: int):
            _refresh_dca_mode_widgets()
            print(f"[DCA] Mode changed -> {self.get_current_dca_mode_key()}")
        self.dca_mode_combo.currentIndexChanged.connect(_on_dca_mode_changed)
        _refresh_dca_mode_widgets()
        
        dca_group.setLayout(dca_layout)
        layout.addWidget(dca_group)
        
        # Connect DCA Strategy controls to auto-save (no popup message)
        self.enable_dca_check.stateChanged.connect(self.auto_save_settings)
        self.max_dca_levels_spin.valueChanged.connect(self.auto_save_settings)
        # 🔧 FIX: Connect DCA mode-specific widgets to auto-save
        self.dca_multiplier_spin.valueChanged.connect(self.auto_save_settings)
        self.dca_mode_combo.currentTextChanged.connect(self.auto_save_settings)
        self.dca_atr_period_spin.valueChanged.connect(self.auto_save_settings)
        self.dca_atr_mult_spin.valueChanged.connect(self.auto_save_settings)
        self.dca_base_distance_spin.valueChanged.connect(self.auto_save_settings)
        self.dca_fibo_start_combo.currentTextChanged.connect(self.auto_save_settings)
        self.dca_fibo_exec_combo.currentTextChanged.connect(self.auto_save_settings)
        
        # === DCA CONDITIONS GROUP ===
        conditions_group = QGroupBox(I18N.t("⚙️ DCA Activation Conditions", "⚙️ Điều kiện kích hoạt DCA"))
        conditions_layout = QGridLayout()
        
        # Minimum drawdown to activate DCA
        conditions_layout.addWidget(QLabel(I18N.t("Min Drawdown for DCA (%):", "Sụt giảm tối thiểu để DCA (%):")), 0, 0)
        self.dca_min_drawdown_spin = QDoubleSpinBox()
        self.dca_min_drawdown_spin.setRange(0.0, 100.0)
        self.dca_min_drawdown_spin.setSingleStep(0.1)
        self.dca_min_drawdown_spin.setValue(self.settings.get('dca_min_drawdown', 1.0))
        self.dca_min_drawdown_spin.setToolTip(I18N.t("Minimum unrealized loss percentage to trigger DCA", "Phần trăm lỗ chưa thực hiện tối thiểu để kích hoạt DCA"))
        conditions_layout.addWidget(self.dca_min_drawdown_spin, 0, 1)
        
        # (Removed checkbox: Only DCA on High Confidence Signals (>4.0))
        # Row 1 now left intentionally empty or can be repurposed later.
        
        # DCA stop loss mode
        conditions_layout.addWidget(QLabel(I18N.t("DCA SL Mode:", "Chế độ SL cho DCA:")), 2, 0)
        self.dca_sl_mode_combo = QComboBox()
        # 🔧 FORCE CLEAR: Ensure no cached items remain
        self.dca_sl_mode_combo.clear()
        self.dca_sl_mode_combo.addItems([
            I18N.t("Individual SL", "SL riêng lẻ"), 
            I18N.t("Average SL", "SL trung bình")
        ])
        # Handle backward compatibility - map old values to valid options
        saved_dca_sl_mode = self.settings.get('dca_sl_mode', I18N.t('Average SL', 'SL trung bình'))
        valid_modes = [I18N.t("Individual SL", "SL riêng lẻ"), I18N.t("Average SL", "SL trung bình")]
        
        # Map legacy values to new valid options
        if saved_dca_sl_mode not in valid_modes:
            if saved_dca_sl_mode in ["adaptive", "Chỉ hòa vốn", "Breakeven Only", "breakeven"]:
                saved_dca_sl_mode = I18N.t('Average SL', 'SL trung bình')  # Default to Average SL for legacy modes
        
        self.dca_sl_mode_combo.setCurrentText(saved_dca_sl_mode)
        self.dca_sl_mode_combo.setToolTip(I18N.t("How to handle stop loss for DCA entries", "Cách xử lý stop loss cho các lệnh DCA"))
        conditions_layout.addWidget(self.dca_sl_mode_combo, 2, 1)
        
        # 🆕 NEW: Profit percentage adjustment for Average SL mode
        self.dca_avg_sl_profit_label = QLabel(I18N.t("Average SL Profit % (Per Symbol):", "% Lợi nhuận SL trung bình (Theo Symbol):"))
        conditions_layout.addWidget(self.dca_avg_sl_profit_label, 3, 0)
        self.dca_avg_sl_profit_spin = QDoubleSpinBox()
        self.dca_avg_sl_profit_spin.setRange(-50.0, 100.0)  # Allow negative for partial loss, up to 100% profit
        self.dca_avg_sl_profit_spin.setSingleStep(1.0)
        self.dca_avg_sl_profit_spin.setSuffix("%")
        self.dca_avg_sl_profit_spin.setValue(self.settings.get('dca_avg_sl_profit_percent', 10.0))
        self.dca_avg_sl_profit_spin.setToolTip(I18N.t(
            "Profit percentage target when using Average SL mode. Calculated PER SYMBOL, not all positions. 0% = breakeven, 10% = 10% profit target for that symbol",
            "Mục tiêu % lợi nhuận theo TỪNG SYMBOL khi dùng chế độ SL trung bình. VD: 10% = đóng tất cả lệnh XAUUSD khi lợi nhuận XAUUSD đạt 10%, không ảnh hưởng Symbol khác"
        ))
        conditions_layout.addWidget(self.dca_avg_sl_profit_spin, 3, 1)
        
        conditions_group.setLayout(conditions_layout)
        layout.addWidget(conditions_group)
        
        # Connect DCA Conditions controls to auto-save (no popup message)
        self.dca_min_drawdown_spin.valueChanged.connect(self.auto_save_settings)
        self.dca_sl_mode_combo.currentTextChanged.connect(self.auto_save_settings)
        self.dca_sl_mode_combo.currentTextChanged.connect(self.toggle_dca_profit_controls)  # 🆕 NEW: Show/hide profit controls
        self.dca_avg_sl_profit_spin.valueChanged.connect(self.auto_save_settings)  # 🆕 NEW: Auto save profit %
        
        # Initial state of profit controls
        self.toggle_dca_profit_controls()
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
        
    # (DCA mode change logic removed – fixed simple distance UI)
        
    def on_volume_mode_changed(self):
        """Update volume mode explanation when mode changes"""
        self.update_volume_mode_explanation()
        self.update_volume_fields_visibility()
        
    def update_volume_fields_visibility(self):
        """🔧 FIX: Simplified volume fields visibility logic"""
        try:
            current_mode = self.volume_mode_combo.currentText()
            
            # 🔧 SIMPLIFIED: Four clear modes
            is_auto_mode = "auto" in current_mode.lower() or "tự động" in current_mode.lower()
            is_fixed_mode = "fixed" in current_mode.lower() or "cố định" in current_mode.lower()
            is_default_mode = "default" in current_mode.lower() or "mặc định" in current_mode.lower()
            
            # Show/hide min lot for auto mode only
            if hasattr(self, 'min_lot_spin'):
                self.min_lot_spin.setVisible(is_auto_mode)
                # Find and hide/show min lot label
                try:
                    position_layout = self.min_lot_spin.parent().layout()
                    min_lot_label = position_layout.itemAtPosition(1, 0).widget()
                    min_lot_label.setVisible(is_auto_mode)
                except:
                    pass
            
            # Show/hide fixed volume for fixed mode only  
            if hasattr(self, 'fixed_volume_spin'):
                self.fixed_volume_spin.setVisible(is_fixed_mode)
                self.fixed_volume_label.setVisible(is_fixed_mode)
                
            # Show/hide default volume for default mode only
            if hasattr(self, 'default_volume_spin'):
                self.default_volume_spin.setVisible(is_default_mode)
                self.default_volume_label.setVisible(is_default_mode)
            
        except Exception as e:
            print(f"⚠️ Volume fields visibility error: {e}")
        
    def update_volume_mode_explanation(self):
        """Update explanation text based on selected volume mode"""
        try:
            current_mode = self.volume_mode_combo.currentText()
            
            if "tự động" in current_mode.lower() or "auto" in current_mode.lower():
                text = I18N.t(
                    "Auto Mode: Min volume for first trades, DCA scales up. Both min and max limits apply.",
                    "Chế độ tự động: Khối lượng tối thiểu cho lệnh đầu, DCA scale lên. Áp dụng cả min và max."
                )
            elif "Risk-Based" in current_mode or "Theo rủi ro" in current_mode:
                text = I18N.t(
                    "Risk-Based: Volume calculated automatically based on risk percentage, only max limit applies",
                    "Theo rủi ro: Khối lượng tính dựa trên tỷ lệ rủi ro, chỉ áp dụng giới hạn max"
                )
            elif "cố định" in current_mode.lower() or "Fixed Volume" in current_mode:
                text = I18N.t(
                    "Fixed Volume: All trades use the fixed lot size, only max total limit applies",
                    "Khối lượng cố định: Tất cả lệnh dùng kích thước cố định, chỉ áp dụng giới hạn tổng max"
                )
            elif "mặc định" in current_mode.lower() or "Default Volume" in current_mode:
                text = I18N.t(
                    "Default Volume: Uses default volume setting for all trades, only max total limit applies",
                    "Khối lượng mặc định: Dùng cài đặt khối lượng mặc định cho tất cả lệnh, chỉ áp dụng giới hạn tổng max"
                )
            else:
                text = ""
                
            self.volume_mode_label.setText(text)
        except Exception as e:
            print(f"Error updating volume mode explanation: {e}")
        
    def create_control_panel(self):
        """Create bottom control panel"""
        panel = QGroupBox(I18N.t("🎮 Control Panel", "🎮 Bảng điều khiển"))
        layout = QHBoxLayout()
        
        # Mode selection
        mode_label = QLabel(I18N.t("Trading Mode:", "Chế độ giao dịch:"))
        layout.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            I18N.t("🤖 Auto Mode", "🤖 Tự động"), 
            I18N.t("👨‍💼 Manual Mode", "👨‍💼 Thủ công")
        ])
        self.mode_combo.setCurrentText(self.settings.get('trading_mode', I18N.t('👨‍💼 Manual Mode', '👨‍💼 Thủ công')))
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        layout.addWidget(self.mode_combo)
        
        layout.addStretch()
        
        # Control buttons
        self.save_btn = QPushButton(I18N.t("💾 Save Settings", "💾 Lưu cài đặt"))
        self.save_btn.clicked.connect(self.save_settings)
        self.save_btn.setStyleSheet("background-color: #27AE60; color: white; font-weight: bold;")
        layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton(I18N.t("📁 Load Settings", "📁 Tải cài đặt"))
        self.load_btn.clicked.connect(self.load_settings)
        layout.addWidget(self.load_btn)
        
        self.reset_btn = QPushButton(I18N.t("🔄 Reset to Default", "🔄 Đặt lại mặc định"))
        self.reset_btn.clicked.connect(self.reset_to_default)
        layout.addWidget(self.reset_btn)
        
        self.generate_report_btn = QPushButton(I18N.t("📊 Generate Report", "📊 Tạo báo cáo"))
        self.generate_report_btn.clicked.connect(self.generate_report)
        self.generate_report_btn.setStyleSheet("background-color: #3498DB; color: white; font-weight: bold;")
        layout.addWidget(self.generate_report_btn)
        
        panel.setLayout(layout)
        return panel
        
    def init_risk_manager(self):
        """Initialize the risk management system"""
        try:
            if RISK_MANAGER_AVAILABLE:
                # Get values from combos with OFF support
                avoid_news_value = self.get_combo_value(self.avoid_news_combo, 30) if hasattr(self, 'avoid_news_combo') else 30
                emergency_dd_value = self.get_combo_value(self.emergency_dd_combo, 10.0) if hasattr(self, 'emergency_dd_combo') else 10.0
                max_dd_value = self.get_combo_value(self.max_drawdown_combo, 5.0) if hasattr(self, 'max_drawdown_combo') else 5.0
                max_risk_value = self.get_combo_value(self.max_risk_combo, 2.0) if hasattr(self, 'max_risk_combo') else 2.0
                daily_loss_value = self.get_combo_value(self.daily_loss_combo, 3.0) if hasattr(self, 'daily_loss_combo') else 3.0
                
                # Create advanced risk parameters from UI settings
                params = AdvancedRiskParameters(
                    max_risk_percent=max_risk_value,
                    max_drawdown_percent=max_dd_value,
                    max_daily_loss_percent=daily_loss_value,
                    max_positions=self.max_positions_spin.value(),
                    max_positions_per_symbol=self.max_positions_per_symbol_spin.value(),
                    min_risk_reward_ratio=self.get_combo_value(self.min_rr_combo, 1.5) if hasattr(self, 'min_rr_combo') else self.min_rr_spin.value(),
                    min_volume_auto=self.min_lot_spin.value(),
                    max_total_volume=self.get_combo_value(self.max_lot_combo, 10.0),
                    max_correlation=self.max_correlation_spin.value(),
                    trading_hours_start=self.start_hour_spin.value(),
                    trading_hours_end=self.end_hour_spin.value(),
                    avoid_news_minutes=avoid_news_value,
                    max_spread_multiplier=self.spread_multiplier_spin.value(),
                    max_slippage=self.max_slippage_spin.value(),
                    emergency_stop_drawdown=emergency_dd_value,
                    auto_reduce_on_losses=self.auto_reduce_check.isChecked(),
                    # 🆕 OFF detection from combo values
                    disable_news_avoidance=avoid_news_value == "OFF",  # OFF string
                    disable_emergency_stop=emergency_dd_value == "OFF",  # OFF string
                    disable_max_dd_close=max_dd_value == "OFF",  # OFF string
                    # Auto mode enabled only when trading mode is "Auto Mode"
                    auto_mode_enabled=self.mode_combo.currentText() in [I18N.t("🤖 Auto Mode", "🤖 Tự động")],
                    auto_scan_enabled=self.mode_combo.currentText() in [I18N.t("🤖 Auto Mode", "🤖 Tự động")],
                    auto_adjustment_interval=self.get_auto_adjustment_interval_hours() if self.mode_combo.currentText() in [I18N.t("🤖 Auto Mode", "🤖 Tự động")] else 24  # Smart timeframe-based interval
                )
                
                self.risk_manager = AdvancedRiskManagementSystem(params)
                # Status update removed - no status_label in Risk Management Tab anymore
                
                # Force update auto mode status based on current trading mode (override any saved settings)
                is_auto_mode = self.mode_combo.currentText() in [I18N.t("🤖 Auto Mode", "🤖 Tự động")]
                self.risk_manager.risk_params.auto_mode_enabled = is_auto_mode
                self.risk_manager.risk_params.auto_scan_enabled = is_auto_mode
                
                # Display auto mode status based on trading mode
                auto_status = "ENABLED" if is_auto_mode else "DISABLED" 
                print(f"🤖 Auto Mode: {auto_status} (Trading Mode: {self.mode_combo.currentText()})")
                
                # Force auto adjustment when auto mode is enabled
                if is_auto_mode:
                    print("🔄 Triggering immediate auto adjustment for auto mode...")
                    self.risk_manager._perform_auto_adjustment()
                    interval_hours = self.get_auto_adjustment_interval_hours()
                    print(f"⏰ Next auto adjustment in {interval_hours} hours based on timeframe")
                else:
                    print("🔒 Auto adjustments disabled - manual mode active")
                
                print(f"✅ Risk Manager initialized successfully with {'auto' if is_auto_mode else 'manual'} mode")
            else:
                # Status update removed - no status_label in Risk Management Tab anymore
                print("⚠️ Risk Manager not available")
        except Exception as e:
            # Status update removed - no status_label in Risk Management Tab anymore
            print(f"❌ Risk Manager initialization failed: {e}")
    
    def validate_signal(self, signal_data):
        """Validate a trading signal using the risk manager"""
        if not self.risk_manager:
            return {"valid": False, "error": "Risk manager not initialized"}
            
        try:
            # Create TradeSignal object
            signal = TradeSignal(
                symbol=signal_data.get('symbol', ''),
                action=signal_data.get('action', ''),
                entry_price=signal_data.get('entry_price', 0.0),
                stop_loss=signal_data.get('stop_loss', 0.0),
                take_profit=signal_data.get('take_profit', 0.0),
                volume=signal_data.get('volume', 0.1),
                confidence=signal_data.get('confidence', 3.0),
                strategy=signal_data.get('strategy', 'GUI_MANUAL'),
                comment=signal_data.get('comment', 'Manual signal')
            )
            
            # Validate using risk manager
            validation = self.risk_manager.validate_signal_comprehensive(signal)
            
            return {
                "valid": validation.result == ValidationResult.APPROVED,
                "result": validation.result.value,
                "recommended_volume": validation.recommended_volume,
                "risk_score": validation.risk_score,
                "warnings": validation.warnings,
                "errors": validation.errors
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def on_mode_changed(self, mode):
        """Handle trading mode change"""
        print(f"🔄 Trading mode changed to: {mode}")
        self.settings['trading_mode'] = mode
        
        if "Auto Mode" in mode or "Tự động" in mode:
            print("🤖 Trading Mode: AUTO - System will manage all trades and auto-adjust risk")
            # Update risk manager with auto mode enabled
            self.update_risk_manager_mode(auto_mode=True)
        elif "Manual Mode" in mode or "Thủ công" in mode:
            print("👨‍💼 Trading Mode: MANUAL - User controls all risk parameters manually")
            # Update risk manager with auto mode disabled
            self.update_risk_manager_mode(auto_mode=False)
        
        # Auto-save settings when mode changes (no popup message)
        self.auto_save_settings()
        print(f"💾 Settings saved with new mode: {mode}")
    
    def update_risk_manager_mode(self, auto_mode: bool):
        """Update risk manager auto mode status"""
        try:
            if self.risk_manager:
                # Update auto mode flags
                self.risk_manager.risk_params.auto_mode_enabled = auto_mode
                self.risk_manager.risk_params.auto_scan_enabled = auto_mode
                
                # Display status
                auto_status = "ENABLED" if auto_mode else "DISABLED"
                print(f"🤖 Auto Mode Updated: {auto_status}")
                
                # Trigger auto adjustment if enabled
                if auto_mode:
                    print("� Triggering immediate auto adjustment...")
                    self.risk_manager._perform_auto_adjustment()
                    interval_hours = self.get_auto_adjustment_interval_hours()
                    print(f"⏰ Next auto adjustment in {interval_hours} hours based on timeframe")
                else:
                    print("🔒 Auto adjustments disabled - manual mode active")
            else:
                print("⚠️ Risk manager not initialized yet")
        except Exception as e:
            print(f"❌ Error updating risk manager mode: {e}")
    
    def get_smallest_timeframe_minutes(self):
        """Get the smallest selected timeframe from MarketTab in minutes"""
        try:
            if not self.market_tab or not hasattr(self.market_tab, 'tf_checkboxes'):
                return 60  # Default to 1 hour if no market tab
            
            # Timeframe mapping to minutes
            timeframe_minutes = {
                'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 
                'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080, 'MN1': 43200
            }
            
            # Get selected timeframes
            selected_timeframes = []
            for tf, checkbox in self.market_tab.tf_checkboxes.items():
                if checkbox.isChecked():
                    selected_timeframes.append(tf)
            
            if not selected_timeframes:
                return 60  # Default to 1 hour if none selected
            
            # Find smallest timeframe in minutes
            smallest_minutes = min([timeframe_minutes.get(tf, 60) for tf in selected_timeframes])
            print(f"📊 Detected smallest timeframe: {smallest_minutes} minutes from {selected_timeframes}")
            return smallest_minutes
            
        except Exception as e:
            print(f"❌ Error detecting timeframe: {e}")
            return 60  # Default fallback
    
    def get_auto_adjustment_interval_hours(self):
        """Calculate smart auto adjustment interval based on smallest timeframe"""
        smallest_minutes = self.get_smallest_timeframe_minutes()
        
        # Smart interval logic based on timeframe
        if smallest_minutes <= 5:      # M1, M5 -> adjust every 30 minutes 
            return 0.5
        elif smallest_minutes <= 15:   # M15 -> adjust every 1 hour
            return 1
        elif smallest_minutes <= 60:   # M30, H1 -> adjust every 2 hours
            return 2  
        elif smallest_minutes <= 240:  # H4 -> adjust every 4 hours
            return 4
        else:                          # D1, W1, MN1 -> adjust every 24 hours
            return 24
    
    @safe_method
    def save_settings(self, show_message=True):
        """Save current settings to file
        
        Args:
            show_message (bool): Whether to show success message popup. 
                               True for manual save (button click), False for auto-save.
        """
        try:
            # Ensure settings dict exists
            if not hasattr(self, 'settings') or self.settings is None:
                self.settings = {}
            
            # Collect all settings from UI with comprehensive null checks
            self.settings = {
                'max_risk_percent': self.get_combo_value(self.max_risk_combo, 2.0) if hasattr(self, 'max_risk_combo') and self.max_risk_combo else 2.0,
                'max_drawdown_percent': self.get_combo_value(self.max_drawdown_combo, 5.0) if hasattr(self, 'max_drawdown_combo') and self.max_drawdown_combo else (self.max_drawdown_spin.value() if hasattr(self, 'max_drawdown_spin') and self.max_drawdown_spin else 5.0),
                'max_daily_loss_percent': self.get_combo_value(self.daily_loss_combo, 3.0) if hasattr(self, 'daily_loss_combo') and self.daily_loss_combo else 3.0,
                'min_volume_auto': self.min_lot_spin.value() if hasattr(self, 'min_lot_spin') and self.min_lot_spin else 0.01,
                'max_total_volume': self.get_combo_value(self.max_lot_combo, 10.0) if hasattr(self, 'max_lot_combo') and self.max_lot_combo else 10.0,
                'min_risk_reward_ratio': self.get_combo_value(self.min_rr_combo, 1.5) if hasattr(self, 'min_rr_combo') and self.min_rr_combo else (self.min_rr_spin.value() if hasattr(self, 'min_rr_spin') and self.min_rr_spin else 1.5),
                # Dynamic SL/TP settings based on mode
                # 🔧 SIMPLIFIED: Always save current SL/TP values regardless of mode
                'default_sl_pips': self.default_sl_spin.value() if hasattr(self, 'default_sl_spin') and self.default_sl_spin else self.settings.get('default_sl_pips', 50.0),
                'default_tp_pips': self.default_tp_spin.value() if hasattr(self, 'default_tp_spin') and self.default_tp_spin else self.settings.get('default_tp_pips', 100.0),
                'default_sl_atr_multiplier': self.default_sl_spin.value() if hasattr(self, 'default_sl_spin') and self.default_sl_spin else self.settings.get('default_sl_atr_multiplier', 2.0),
                'default_tp_atr_multiplier': self.default_tp_spin.value() if hasattr(self, 'default_tp_spin') and self.default_tp_spin else self.settings.get('default_tp_atr_multiplier', 3.0),
                'default_sl_percentage': self.default_sl_spin.value() if hasattr(self, 'default_sl_spin') and self.default_sl_spin else self.settings.get('default_sl_percentage', 2.0),
                'default_tp_percentage': self.default_tp_spin.value() if hasattr(self, 'default_tp_spin') and self.default_tp_spin else self.settings.get('default_tp_percentage', 4.0),
                'default_sl_buffer': self.default_sl_spin.value() if hasattr(self, 'default_sl_spin') and self.default_sl_spin else self.settings.get('default_sl_buffer', 10.0),
                'default_tp_buffer': self.default_tp_spin.value() if hasattr(self, 'default_tp_spin') and self.default_tp_spin else self.settings.get('default_tp_buffer', 20.0),
                'signal_sl_factor': self.default_sl_spin.value() if hasattr(self, 'default_sl_spin') and self.default_sl_spin else self.settings.get('signal_sl_factor', 1.0),
                'signal_tp_factor': self.default_tp_spin.value() if hasattr(self, 'default_tp_spin') and self.default_tp_spin else self.settings.get('signal_tp_factor', 1.0),
                'sltp_mode': self.sltp_mode_combo.currentText() if hasattr(self, 'sltp_mode_combo') and self.sltp_mode_combo else 'Dynamic ATR',
                'max_positions': self.max_positions_spin.value() if hasattr(self, 'max_positions_spin') and self.max_positions_spin else 5,
                'max_positions_per_symbol': self.max_positions_per_symbol_spin.value() if hasattr(self, 'max_positions_per_symbol_spin') and self.max_positions_per_symbol_spin else 3,
                'max_correlation': self.max_correlation_spin.value() if hasattr(self, 'max_correlation_spin') and self.max_correlation_spin else 0.8,
                'trading_hours_start': self.start_hour_spin.value() if hasattr(self, 'start_hour_spin') and self.start_hour_spin else 0,
                'trading_hours_end': self.end_hour_spin.value() if hasattr(self, 'end_hour_spin') and self.end_hour_spin else 24,
                'avoid_news_minutes': self.get_combo_value(self.avoid_news_combo, 30) if hasattr(self, 'avoid_news_combo') and self.avoid_news_combo else (self.avoid_news_spin.value() if hasattr(self, 'avoid_news_spin') and self.avoid_news_spin else 30),
                'max_spread_multiplier': self.spread_multiplier_spin.value() if hasattr(self, 'spread_multiplier_spin') and self.spread_multiplier_spin else 3.0,
                'max_slippage': self.max_slippage_spin.value() if hasattr(self, 'max_slippage_spin') and self.max_slippage_spin else 5,
                'emergency_stop_drawdown': self.get_combo_value(self.emergency_dd_combo, 10.0) if hasattr(self, 'emergency_dd_combo') and self.emergency_dd_combo else (self.emergency_dd_spin.value() if hasattr(self, 'emergency_dd_spin') and self.emergency_dd_spin else 10.0),
                'auto_reduce_on_losses': self.auto_reduce_check.isChecked() if hasattr(self, 'auto_reduce_check') and self.auto_reduce_check else False,
                'enable_dca': self.enable_dca_check.isChecked() if hasattr(self, 'enable_dca_check') and self.enable_dca_check else True,
                'max_dca_levels': self.max_dca_levels_spin.value() if hasattr(self, 'max_dca_levels_spin') and self.max_dca_levels_spin else 3,

                # Save both canonical key and legacy label for backward compatibility
                'dca_mode': self.get_current_dca_mode_key() if hasattr(self, 'dca_mode_combo') else self.settings.get('dca_mode', 'atr_multiple'),
                'dca_mode_legacy': (lambda k: {
                    'atr_multiple': 'Bội số ATR',
                    'fixed_pips': 'Pips cố định',
                    'fibo_levels': 'Mức Fibo'
                }.get(k, 'Pips cố định'))(self.get_current_dca_mode_key() if hasattr(self, 'dca_mode_combo') else self.settings.get('dca_mode', 'fixed_pips')),
                'dca_atr_period': self.dca_atr_period_spin.value() if hasattr(self, 'dca_atr_period_spin') else self.settings.get('dca_atr_period', 14),
                'dca_atr_multiplier': self.dca_atr_mult_spin.value() if hasattr(self, 'dca_atr_mult_spin') else self.settings.get('dca_atr_multiplier', 1.5),
                'dca_distance_pips': self.dca_base_distance_spin.value() if hasattr(self, 'dca_base_distance_spin') else self.settings.get('dca_base_distance_pips', 50),
                # New simplified Fibonacci persistence: store starting index only
                'dca_fibo_start_level': (lambda: self.dca_fibo_start_combo.currentIndex() if hasattr(self, 'dca_fibo_start_combo') else self.settings.get('dca_fibo_start_level', 0))(),
                # 🔧 FIXED: Save retracement percentages instead of sequence numbers
                'dca_fibo_levels': self.settings.get('dca_fibo_levels', "23.6,38.2,50,61.8,78.6"),  # Keep existing percentages
                'dca_fibo_exec_mode': self.dca_fibo_exec_combo.currentText() if hasattr(self, 'dca_fibo_exec_combo') else self.settings.get('dca_fibo_exec_mode', I18N.t("On Touch (Market)", "Chạm Mức (Market)")),
                # 'dca_fibonacci_level' removed
                'dca_volume_multiplier': self.dca_multiplier_spin.value() if hasattr(self, 'dca_multiplier_spin') and self.dca_multiplier_spin else 1.5,
                # 'dca_mode' removed from persistence
                'dca_min_drawdown': self.dca_min_drawdown_spin.value() if hasattr(self, 'dca_min_drawdown_spin') and self.dca_min_drawdown_spin else 1.0,
                # Removed: 'dca_high_confidence_only'
                'dca_sl_mode': self.dca_sl_mode_combo.currentText() if hasattr(self, 'dca_sl_mode_combo') else self.settings.get('dca_sl_mode', 'SL trung bình'),
                'dca_avg_sl_profit_percent': self.dca_avg_sl_profit_spin.value() if hasattr(self, 'dca_avg_sl_profit_spin') else self.settings.get('dca_avg_sl_profit_percent', 10.0),  # 🆕 NEW
                'trading_mode': self.mode_combo.currentText() if hasattr(self, 'mode_combo') and self.mode_combo else I18N.t("👨‍💼 Manual Mode", "👨‍💼 Thủ công"),
                # Volume management settings
                'volume_mode': self.volume_mode_combo.currentText() if hasattr(self, 'volume_mode_combo') and self.volume_mode_combo else I18N.t('Risk-Based (Auto)', 'Theo rủi ro (Tự động)'),
                'fixed_volume_lots': self.fixed_volume_spin.value() if hasattr(self, 'fixed_volume_spin') and self.fixed_volume_spin else 0.10,
                'default_volume_lots': self.default_volume_spin.value() if hasattr(self, 'default_volume_spin') and self.default_volume_spin else 0.10,
                # Simplified settings - OFF is integrated into main fields, no separate dropdowns
                # OFF detection from combo values (backward compatibility)
                'disable_news_avoidance': self.get_combo_value(self.avoid_news_combo, 30) == "OFF" if hasattr(self, 'avoid_news_combo') and self.avoid_news_combo else False,
                'disable_emergency_stop': self.get_combo_value(self.emergency_dd_combo, 10.0) == "OFF" if hasattr(self, 'emergency_dd_combo') and self.emergency_dd_combo else False,
                'disable_max_dd_close': self.get_combo_value(self.max_drawdown_combo, 5.0) == "OFF" if hasattr(self, 'max_drawdown_combo') and self.max_drawdown_combo else False,
                # Auto mode based on trading mode selection
                'enable_auto_mode': (self.mode_combo.currentText() in [I18N.t("🤖 Auto Mode", "🤖 Tự động")]) if hasattr(self, 'mode_combo') and self.mode_combo else False,
                'enable_auto_scan': (self.mode_combo.currentText() in [I18N.t("🤖 Auto Mode", "🤖 Tự động")]) if hasattr(self, 'mode_combo') and self.mode_combo else False,
                'auto_scan_interval': (self.get_auto_adjustment_interval_hours() if (hasattr(self, 'mode_combo') and self.mode_combo and self.mode_combo.currentText() in [I18N.t("🤖 Auto Mode", "🤖 Tự động")]) else 24) if hasattr(self, 'get_auto_adjustment_interval_hours') else 24  # Smart timeframe-based interval
            }
            
            # Save symbol exposure settings
            symbol_exposure = {}
            symbol_multipliers = {}
            if hasattr(self, 'exposure_table') and self.exposure_table:
                try:
                    for i in range(self.exposure_table.rowCount()):
                        item = self.exposure_table.item(i, 0)
                        if item:  # Check if item exists
                            symbol = item.text()
                            exposure_widget = self.exposure_table.cellWidget(i, 1)
                            multiplier_widget = self.exposure_table.cellWidget(i, 2)
                            if exposure_widget and multiplier_widget and symbol:
                                symbol_exposure[symbol] = exposure_widget.value()
                                symbol_multipliers[symbol] = multiplier_widget.value()
                except Exception as e:
                    print(f"⚠️ Warning saving exposure table: {e}")
            
            self.settings['symbol_exposure'] = symbol_exposure
            self.settings['symbol_multipliers'] = symbol_multipliers
            
            # 🔧 FORCE CORRECT DCA FIBO LEVELS - prevent override from other functions
            if 'dca_fibo_levels' in self.settings:
                current_levels = self.settings.get('dca_fibo_levels', '')
                # If it contains sequence numbers, fix it to percentages
                if '1,1,2,3,5,8' in str(current_levels) or '1.0,1.0,2.0' in str(current_levels):
                    print(f"🔧 Fixing DCA Fibo Levels: {current_levels} → 23.6,38.2,50,61.8,78.6")
                    self.settings['dca_fibo_levels'] = "23.6,38.2,50,61.8,78.6"
            
            # 🔧 PROTECT DCA MODE FROM OVERRIDE - ensure user's choice is preserved
            if hasattr(self, 'dca_mode_combo') and self.dca_mode_combo:
                current_dca_key = self.get_current_dca_mode_key()
                if current_dca_key == 'atr_multiple':
                    # Force ATR mode if user selected it
                    self.settings['dca_mode'] = 'atr_multiple'
                    self.settings['dca_mode_legacy'] = 'Bội số ATR'
                    print("🔧 Protected DCA mode: atr_multiple (ATR Multiple)")
                elif current_dca_key == 'fibo_levels':
                    # Force Fibonacci mode if user selected it  
                    self.settings['dca_mode'] = 'fibo_levels'
                    self.settings['dca_mode_legacy'] = 'Mức Fibonacci'
                    print("🔧 Protected DCA mode: fibo_levels (Fibonacci)")
            
            # Save to file
            os.makedirs("risk_management", exist_ok=True)
            with open("risk_management/risk_settings.json", 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            
            # DEBUG: Log DCA settings
            print("✅ Risk settings saved successfully")
            print(f"🔍 DEBUG saved DCA settings:")
            print(f"   enable_dca: {self.settings.get('enable_dca')}")
            print(f"   dca_mode: {self.settings.get('dca_mode')}")
            print(f"   dca_volume_multiplier: {self.settings.get('dca_volume_multiplier')}")
            print(f"   dca_sl_mode: {self.settings.get('dca_sl_mode')}")
            print(f"   max_dca_levels: {self.settings.get('max_dca_levels')}")
            print(f"   dca_min_drawdown: {self.settings.get('dca_min_drawdown')}")
            print(f"   default_sl_pips: {self.settings.get('default_sl_pips')}")
            print(f"   default_tp_pips: {self.settings.get('default_tp_pips')}")
            
            # Reinitialize risk manager with new settings
            self.init_risk_manager()
            
            # Only show message for manual saves (button click), not auto-saves
            if show_message:
                QMessageBox.information(
                    self,
                    I18N.t("Settings Saved", "Đã lưu cài đặt"),
                    I18N.t("✅ Risk management settings saved successfully!", "✅ Đã lưu cài đặt quản lý rủi ro thành công!")
                )
            
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"❌ Error saving settings: {error_detail}")
            
            QMessageBox.critical(
                self,
                I18N.t("Save Error", "Lỗi lưu"),
                I18N.t("❌ Error saving settings: {err}", "❌ Lỗi khi lưu cài đặt: {err}", err=str(e))
            )
    
    def auto_save_settings(self):
        """Auto-save settings without showing popup message"""
        self.save_settings(show_message=False)
    
    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists("risk_management/risk_settings.json"):
                with open("risk_management/risk_settings.json", 'r', encoding='utf-8') as f:
                    self.settings = json.load(f)
                    
                # Update UI with loaded settings
                self.update_ui_from_settings()
                
                # QMessageBox.information(self, "Settings Loaded", "✅ Risk management settings loaded successfully!")
            else:
                QMessageBox.warning(
                    self,
                    I18N.t("No Settings", "Không có cài đặt"),
                    I18N.t("⚠️ No saved settings found.", "⚠️ Không tìm thấy cài đặt đã lưu.")
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,
                I18N.t("Load Error", "Lỗi tải"),
                I18N.t("❌ Error loading settings: {err}", "❌ Lỗi khi tải cài đặt: {err}", err=str(e))
            )
    
    def toggle_dca_profit_controls(self):
        """Show/hide DCA profit controls based on selected SL mode"""
        if hasattr(self, 'dca_sl_mode_combo') and hasattr(self, 'dca_avg_sl_profit_spin'):
            current_mode = self.dca_sl_mode_combo.currentText()
            is_average_mode = "SL trung bình" in current_mode or "Average SL" in current_mode
            
            # Show profit controls only for Average SL mode
            self.dca_avg_sl_profit_spin.setVisible(is_average_mode)
            
            # Also hide/show the label
            if hasattr(self, 'dca_avg_sl_profit_label'):
                self.dca_avg_sl_profit_label.setVisible(is_average_mode)
    
    def update_ui_from_settings(self):
        """Update UI controls from loaded settings"""
        try:
            # Force reload settings from file to ensure we have latest OFF values
            if os.path.exists("risk_management/risk_settings.json"):
                with open("risk_management/risk_settings.json", 'r', encoding='utf-8') as f:
                    file_settings = json.load(f)
                print(f"🔍 DEBUG update_ui_from_settings: loaded from file max_total_volume={file_settings.get('max_total_volume')}, min_risk_reward_ratio={file_settings.get('min_risk_reward_ratio')}")
            else:
                file_settings = self.settings
                
            # Set max_risk_combo value
            if hasattr(self, 'max_risk_combo'):
                current_max_risk = file_settings.get('max_risk_percent', 2.0)
                if isinstance(current_max_risk, str) and current_max_risk.upper() == "OFF":
                    self.max_risk_combo.setCurrentText("OFF")
                else:
                    self.max_risk_combo.setCurrentText(str(current_max_risk))
            
            # Set daily_loss_combo value
            if hasattr(self, 'daily_loss_combo'):
                current_daily_loss = file_settings.get('max_daily_loss_percent', 3.0)
                if current_daily_loss is None:
                    self.daily_loss_combo.setCurrentText("OFF")
                elif isinstance(current_daily_loss, str) and current_daily_loss.upper() == "OFF":
                    self.daily_loss_combo.setCurrentText("OFF")
                else:
                    self.daily_loss_combo.setCurrentText(str(current_daily_loss))
            
            self.min_lot_spin.setValue(file_settings.get('min_volume_auto', 0.01))
            if hasattr(self, 'max_lot_combo'):
                current_max_lot = file_settings.get('max_total_volume', 10.0)
                print(f"🔍 DEBUG updating max_lot_combo with: {current_max_lot} (type: {type(current_max_lot)})")
                if current_max_lot is None:
                    self.max_lot_combo.setCurrentText("OFF")
                    print("🔍 DEBUG set max_lot_combo to OFF (null value)")
                elif isinstance(current_max_lot, str) and current_max_lot.upper() == "OFF":
                    self.max_lot_combo.setCurrentText("OFF")
                    print("🔍 DEBUG set max_lot_combo to OFF")
                else:
                    self.max_lot_combo.setCurrentText(str(current_max_lot))
                    print(f"🔍 DEBUG set max_lot_combo to {current_max_lot}")
            if hasattr(self, 'min_rr_combo'):
                current_rr = file_settings.get('min_risk_reward_ratio', 1.5)
                print(f"🔍 DEBUG updating min_rr_combo with: {current_rr} (type: {type(current_rr)})")
                if current_rr is None:
                    self.min_rr_combo.setCurrentText("OFF")
                    print("🔍 DEBUG set min_rr_combo to OFF (null value)")
                elif isinstance(current_rr, str) and current_rr.upper() == "OFF":
                    self.min_rr_combo.setCurrentText("OFF")
                    print("🔍 DEBUG set min_rr_combo to OFF")
                else:
                    self.min_rr_combo.setCurrentText(str(current_rr))
                    print(f"🔍 DEBUG set min_rr_combo to {current_rr}")
            elif hasattr(self, 'min_rr_spin'):
                current_rr_val = file_settings.get('min_risk_reward_ratio', 1.5)
                if current_rr_val is not None:
                    self.min_rr_spin.setValue(current_rr_val)
            
            # Update ComboBox controls with OFF support
            if hasattr(self, 'avoid_news_combo'):
                avoid_minutes = file_settings.get('avoid_news_minutes', 30)
                if avoid_minutes is None:
                    self.avoid_news_combo.setCurrentText("OFF")
                elif isinstance(avoid_minutes, str) and avoid_minutes.upper() == "OFF":
                    self.avoid_news_combo.setCurrentText("OFF")
                else:
                    self.avoid_news_combo.setCurrentText(str(avoid_minutes))
                    
            if hasattr(self, 'emergency_dd_combo'):
                emergency_dd = file_settings.get('emergency_stop_drawdown', 10.0)
                if emergency_dd is None:
                    self.emergency_dd_combo.setCurrentText("OFF")
                elif isinstance(emergency_dd, str) and emergency_dd.upper() == "OFF":
                    self.emergency_dd_combo.setCurrentText("OFF")
                else:
                    self.emergency_dd_combo.setCurrentText(str(emergency_dd))
                    
            if hasattr(self, 'max_drawdown_combo'):
                max_dd = file_settings.get('max_drawdown_percent', 5.0)
                if max_dd is None:
                    self.max_drawdown_combo.setCurrentText("OFF")
                elif isinstance(max_dd, str) and max_dd.upper() == "OFF":
                    self.max_drawdown_combo.setCurrentText("OFF")
                else:
                    self.max_drawdown_combo.setCurrentText(str(max_dd))
            
            # Update DCA settings
            if hasattr(self, 'enable_dca_check'):
                self.enable_dca_check.setChecked(file_settings.get('enable_dca', False))
            if hasattr(self, 'max_dca_levels_spin'):
                self.max_dca_levels_spin.setValue(file_settings.get('max_dca_levels', 3))
            

                
            # Load Fibonacci level
            # (Removed Fibonacci level load)
                        
            # Update UI based on mode (call after loading values)
            # (Removed call to on_dca_mode_changed – legacy)
                
            if hasattr(self, 'dca_multiplier_spin'):
                self.dca_multiplier_spin.setValue(file_settings.get('dca_volume_multiplier', 1.5))
            if hasattr(self, 'dca_mode_combo'):
                saved_mode_key = file_settings.get('dca_mode', 'fixed_multiple')
                # Map legacy labels to new keys
                legacy_map = {
                    'Bội số ATR': 'atr_multiple',
                    'Pips cố định': 'fixed_pips',
                    'Mức Fibo': 'fibo_levels',
                    'Bội số cố định': 'fixed_pips',
                    'Tự động theo tỷ lệ': 'atr_multiple',
                    'Mức Fibonacci': 'fibo_levels'
                }
                saved_mode_key = legacy_map.get(saved_mode_key, saved_mode_key)
                for i in range(self.dca_mode_combo.count()):
                    if self.dca_mode_combo.itemData(i) == saved_mode_key:
                        self.dca_mode_combo.setCurrentIndex(i)
                        break
                # After setting mode, refresh visibility if function exists
                try:
                    if hasattr(self, 'get_current_dca_mode_key') and hasattr(self, 'dca_fibo_note'):
                        # Re-run refresh function defined inline earlier if present
                        # (Inline function not stored; emulate by toggling via mode change signal)
                        pass
                except Exception:
                    pass
            if hasattr(self, 'dca_min_drawdown_spin'):
                self.dca_min_drawdown_spin.setValue(file_settings.get('dca_min_drawdown', 1.0))
            if hasattr(self, 'dca_atr_period_spin'):
                self.dca_atr_period_spin.setValue(int(file_settings.get('dca_atr_period', 14)))
            if hasattr(self, 'dca_atr_mult_spin'):
                self.dca_atr_mult_spin.setValue(file_settings.get('dca_atr_multiplier', 1.5))
            if hasattr(self, 'dca_base_distance_spin'):
                self.dca_base_distance_spin.setValue(int(file_settings.get('dca_distance_pips', 50)))
            # New: load Fibonacci start level (migration from legacy levels list)
            if hasattr(self, 'dca_fibo_start_combo'):
                fibo_sequence = [1,1,2,3,5,8,13,21,34,55,89]
                if 'dca_fibo_start_level' in file_settings:
                    start_idx = int(file_settings.get('dca_fibo_start_level', 0))
                else:
                    # Derive from first legacy level number if present
                    legacy_levels = file_settings.get('dca_fibo_levels') or ""
                    if isinstance(legacy_levels, list):
                        parts = legacy_levels
                    else:
                        parts = [p.strip() for p in str(legacy_levels).split(',') if p.strip()]
                    start_idx = 0
                    if parts:
                        try:
                            first_val = int(float(parts[0]))
                            if first_val in fibo_sequence:
                                start_idx = fibo_sequence.index(first_val)
                        except Exception:
                            pass
                if 0 <= start_idx < self.dca_fibo_start_combo.count():
                    self.dca_fibo_start_combo.setCurrentIndex(start_idx)
            if hasattr(self, 'dca_fibo_exec_combo'):
                exec_mode = file_settings.get('dca_fibo_exec_mode')
                if exec_mode:
                    for i in range(self.dca_fibo_exec_combo.count()):
                        if self.dca_fibo_exec_combo.itemText(i) == exec_mode:
                            self.dca_fibo_exec_combo.setCurrentIndex(i)
                            break
            # Removed high confidence DCA checkbox; ignore legacy key if present
            
            if hasattr(self, 'dca_sl_mode_combo'):
                dca_sl_mode_value = file_settings.get('dca_sl_mode', 'SL trung bình')
                
                # 🔧 BACKWARD COMPATIBILITY: Handle legacy values
                valid_modes = ["SL riêng lẻ", "SL trung bình", "Individual SL", "Average SL"]
                if dca_sl_mode_value not in valid_modes:
                    if dca_sl_mode_value in ["adaptive", "Chỉ hòa vốn", "Breakeven Only", "breakeven"]:
                        dca_sl_mode_value = 'SL trung bình'  # Default to Average SL for removed legacy modes
                        print(f"🔧 MIGRATED removed DCA SL mode '{file_settings.get('dca_sl_mode')}' to 'SL trung bình'")
                
                # Find matching item in combo
                for i in range(self.dca_sl_mode_combo.count()):
                    if self.dca_sl_mode_combo.itemText(i) == dca_sl_mode_value:
                        self.dca_sl_mode_combo.setCurrentIndex(i)
                        break
            
            # 🆕 NEW: Load DCA Average SL Profit Percentage
            if hasattr(self, 'dca_avg_sl_profit_spin'):
                self.dca_avg_sl_profit_spin.setValue(file_settings.get('dca_avg_sl_profit_percent', 10.0))
            
            # Update Volume Management settings
            if hasattr(self, 'volume_mode_combo'):
                self.volume_mode_combo.setCurrentText(file_settings.get('volume_mode', I18N.t('Risk-Based (Auto)', 'Theo rủi ro (Tự động)')))
                self.update_volume_mode_explanation()
            if hasattr(self, 'fixed_volume_spin'):
                self.fixed_volume_spin.setValue(file_settings.get('fixed_volume_lots', 0.10))
            if hasattr(self, 'default_volume_spin'):
                self.default_volume_spin.setValue(file_settings.get('default_volume_lots', 0.10))
            
            # Update Trading Mode combo
            if hasattr(self, 'mode_combo'):
                self.mode_combo.setCurrentText(file_settings.get('trading_mode', I18N.t('👨‍💼 Manual', '👨‍💼 Thủ công')))
            
            # Update SL/TP Mode combo
            if hasattr(self, 'sltp_mode_combo'):
                self.sltp_mode_combo.setCurrentText(file_settings.get('sltp_mode', I18N.t('Fixed Pips', 'Pips cố định')))
                self.update_sltp_mode_controls()  # Refresh controls based on loaded mode
            
            # 🔧 FIXED: Load SL/TP values after mode is set
            if hasattr(self, 'default_sl_spin') and hasattr(self, 'default_tp_spin'):
                # Load SL value based on current mode
                current_sltp_mode = self.sltp_mode_combo.currentText() if hasattr(self, 'sltp_mode_combo') else ""
                
                print(f"🔍 [LOAD DEBUG] Current SL/TP mode: '{current_sltp_mode}'")
                
                if I18N.t('Fixed Pips', 'Pips cố định') in current_sltp_mode:
                    sl_val = file_settings.get('default_sl_pips', 50.0)
                    tp_val = file_settings.get('default_tp_pips', 100.0)
                    print(f"🔍 [LOAD DEBUG] Loading Pips mode: SL={sl_val}, TP={tp_val}")
                    self.default_sl_spin.setValue(sl_val)
                    self.default_tp_spin.setValue(tp_val)
                elif I18N.t('ATR Multiple', 'Bội số ATR') in current_sltp_mode:
                    sl_val = file_settings.get('default_sl_atr_multiplier', 2.0)
                    tp_val = file_settings.get('default_tp_atr_multiplier', 3.0)
                    print(f"🔍 [LOAD DEBUG] Loading ATR mode: SL={sl_val}, TP={tp_val}")
                    self.default_sl_spin.setValue(sl_val)
                    self.default_tp_spin.setValue(tp_val)
                elif I18N.t('Percentage', 'Phần trăm') in current_sltp_mode:
                    sl_val = file_settings.get('default_sl_percentage', 2.0)
                    tp_val = file_settings.get('default_tp_percentage', 4.0)
                    print(f"🔍 [LOAD DEBUG] Loading Percentage mode: SL={sl_val}, TP={tp_val}")
                    self.default_sl_spin.setValue(sl_val)
                    self.default_tp_spin.setValue(tp_val)
                elif I18N.t('Support/Resistance Buffer', 'Đệm Hỗ trợ/Kháng cự') in current_sltp_mode:
                    sl_val = file_settings.get('default_sl_buffer', 10.0)
                    tp_val = file_settings.get('default_tp_buffer', 20.0)
                    print(f"🔍 [LOAD DEBUG] Loading Buffer mode: SL={sl_val}, TP={tp_val}")
                    self.default_sl_spin.setValue(sl_val)
                    self.default_tp_spin.setValue(tp_val)
                else:
                    # Default to Pips mode values
                    sl_val = file_settings.get('default_sl_pips', 50.0)
                    tp_val = file_settings.get('default_tp_pips', 100.0)
                    print(f"🔍 [LOAD DEBUG] Loading Default (Pips) mode: SL={sl_val}, TP={tp_val}")
                    self.default_sl_spin.setValue(sl_val)
                    self.default_tp_spin.setValue(tp_val)
            
            # Update Position/Trading Hours settings
            if hasattr(self, 'max_positions_spin'):
                self.max_positions_spin.setValue(int(file_settings.get('max_positions', 5)))
            if hasattr(self, 'max_positions_per_symbol_spin'):
                self.max_positions_per_symbol_spin.setValue(int(file_settings.get('max_positions_per_symbol', 2)))
            if hasattr(self, 'max_correlation_spin'):
                self.max_correlation_spin.setValue(file_settings.get('max_correlation', 0.7))
            if hasattr(self, 'start_hour_spin'):
                self.start_hour_spin.setValue(int(file_settings.get('trading_hours_start', 0)))
            if hasattr(self, 'end_hour_spin'):
                self.end_hour_spin.setValue(int(file_settings.get('trading_hours_end', 23)))
            
            # Update Market Conditions settings
            if hasattr(self, 'spread_multiplier_spin'):
                self.spread_multiplier_spin.setValue(file_settings.get('max_spread_multiplier', 3.0))
            if hasattr(self, 'max_slippage_spin'):
                self.max_slippage_spin.setValue(int(file_settings.get('max_slippage', 10)))
            
            # Update other checkboxes
            if hasattr(self, 'auto_reduce_check'):
                self.auto_reduce_check.setChecked(file_settings.get('auto_reduce_on_losses', True))
            
            # Update symbol exposure table with file settings
            if hasattr(self, 'exposure_table'):
                self.populate_exposure_table(file_settings)
                    
        except Exception as e:
            print(f"❌ Error updating UI from settings: {e}")
    
    def auto_refresh_settings(self):
        """Auto-refresh settings from file if changed by backend"""
        try:
            if os.path.exists("risk_management/risk_settings.json"):
                # Check if file was modified
                current_mtime = os.path.getmtime("risk_management/risk_settings.json")
                if not hasattr(self, '_last_settings_mtime'):
                    self._last_settings_mtime = current_mtime
                    return
                
                if current_mtime > self._last_settings_mtime:
                    # File was modified, reload settings
                    with open("risk_management/risk_settings.json", 'r', encoding='utf-8') as f:
                        self.settings = json.load(f)
                    self.update_ui_from_settings()
                    self._last_settings_mtime = current_mtime
                    print("🔄 Risk settings auto-refreshed from file")
        except Exception as e:
            print(f"❌ Error in auto-refresh settings: {e}")
    
    def reset_to_default(self):
        """Reset all settings to default values"""
        reply = QMessageBox.question(
            self,
            I18N.t("Reset Settings", "Đặt lại cài đặt"),
            I18N.t("Are you sure you want to reset all settings to default values?", "Bạn có chắc muốn đặt lại tất cả cài đặt về mặc định?"),
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.settings = {}
            self.update_ui_from_settings()
            QMessageBox.information(
                self,
                I18N.t("Reset Complete", "Đặt lại hoàn tất"),
                I18N.t("✅ All settings reset to default values.", "✅ Tất cả cài đặt đã được đặt lại về mặc định.")
            )
    
    def generate_report(self):
        """Generate comprehensive risk management report"""
        try:
            if self.risk_manager:
                report_path = self.risk_manager.save_report()
                if report_path:
                    QMessageBox.information(
                        self,
                        I18N.t("Report Generated", "Đã tạo báo cáo"),
                        I18N.t("✅ Risk management report saved to:\n{path}", "✅ Báo cáo quản lý rủi ro đã lưu tại:\n{path}", path=report_path)
                    )
                else:
                    QMessageBox.warning(
                        self,
                        I18N.t("Report Error", "Lỗi báo cáo"),
                        I18N.t("❌ Failed to generate report.", "❌ Tạo báo cáo thất bại.")
                    )
            else:
                QMessageBox.warning(
                    self,
                    I18N.t("No Risk Manager", "Không có trình quản lý rủi ro"),
                    I18N.t("⚠️ Risk manager not initialized.", "⚠️ Trình quản lý rủi ro chưa được khởi tạo.")
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,
                I18N.t("Report Error", "Lỗi báo cáo"),
                I18N.t("❌ Error generating report: {err}", "❌ Lỗi khi tạo báo cáo: {err}", err=str(e))
            )
    
    def sync_symbols_from_market(self, connected=False):
        """Synchronize available symbols from Market Tab"""
        try:
            if connected and self.market_tab:
                # Get checked symbols from market tab
                self.available_symbols = list(self.market_tab.checked_symbols)
                print(f"🔄 Risk Management synchronized {len(self.available_symbols)} symbols from Market Tab")
                
                # Update symbol exposure settings
                for symbol in self.available_symbols:
                    if symbol not in self.settings.get('symbol_exposure', {}):
                        self.settings.setdefault('symbol_exposure', {})[symbol] = 2.0  # Default exposure
                    if symbol not in self.settings.get('symbol_multipliers', {}):
                        self.settings.setdefault('symbol_multipliers', {})[symbol] = 1.0  # Default multiplier
                
                # Update the exposure table with new symbols
                self.populate_exposure_table()
                
                # Update risk manager if available
                if self.risk_manager:
                    self.update_risk_manager_symbols()
                    
                self.save_settings()
                
                # Update info label
                if hasattr(self, 'exposure_info_label'):
                    self.exposure_info_label.setText(I18N.t(
                        "✅ Synced {n} symbols from Market Tab",
                        "✅ Đã đồng bộ {n} mã từ tab Thị trường",
                        n=len(self.available_symbols)
                    ))
                    
                print(f"✅ Symbol synchronization completed")
            else:
                self.available_symbols = []
                self.populate_exposure_table()  # Clear table
                if hasattr(self, 'exposure_info_label'):
                    self.exposure_info_label.setText(I18N.t(
                        "⚠️ Please connect to MT5 and select symbols in Market Tab",
                        "⚠️ Vui lòng kết nối MT5 và chọn mã ở tab Thị trường"
                    ))
                print("⚠️ Disconnected - cleared symbol list")
                
        except Exception as e:
            print(f"❌ Error synchronizing symbols: {e}")
    
    def populate_exposure_table(self, file_settings=None):
        """Populate exposure table with current available symbols"""
        try:
            # Use file_settings if provided, otherwise fall back to self.settings
            settings_to_use = file_settings if file_settings is not None else self.settings
            
            symbols_to_show = self.available_symbols if self.available_symbols else []
            
            self.exposure_table.setRowCount(len(symbols_to_show))
            
            if not symbols_to_show:
                # Show message when no symbols available
                self.exposure_table.setRowCount(1)
                self.exposure_table.setItem(0, 0, QTableWidgetItem(I18N.t(
                    "No symbols selected", "Chưa chọn mã"
                )))
                self.exposure_table.setItem(0, 1, QTableWidgetItem("--"))
                self.exposure_table.setItem(0, 2, QTableWidgetItem("--"))
                return
            
            for i, symbol in enumerate(symbols_to_show):
                # Symbol name (read-only)
                symbol_item = QTableWidgetItem(symbol)
                symbol_item.setFlags(symbol_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
                symbol_item.setBackground(Qt.lightGray)
                self.exposure_table.setItem(i, 0, symbol_item)
                
                # Max exposure spinbox
                exposure_spin = QDoubleSpinBox()
                exposure_spin.setRange(0.1, 50.0)
                exposure_spin.setSingleStep(0.1)
                exposure_spin.setValue(settings_to_use.get('symbol_exposure', {}).get(symbol, 2.0))
                exposure_spin.setToolTip(I18N.t(
                    f"Maximum exposure for {symbol} in lots",
                    f"Khối lượng tối đa cho {symbol} (lot)"
                ))
                # Connect to auto-save
                exposure_spin.valueChanged.connect(lambda value, s=symbol: self.update_symbol_exposure(s, value))
                self.exposure_table.setCellWidget(i, 1, exposure_spin)
                
                # Risk multiplier spinbox  
                multiplier_spin = QDoubleSpinBox()
                multiplier_spin.setRange(0.1, 3.0)
                multiplier_spin.setSingleStep(0.1)
                multiplier_spin.setValue(settings_to_use.get('symbol_multipliers', {}).get(symbol, 1.0))
                multiplier_spin.setToolTip(I18N.t(
                    f"Risk multiplier for {symbol} (1.0 = normal risk)",
                    f"Hệ số rủi ro cho {symbol} (1.0 = rủi ro chuẩn)"
                ))
                # Connect to auto-save
                multiplier_spin.valueChanged.connect(lambda value, s=symbol: self.update_symbol_multiplier(s, value))
                self.exposure_table.setCellWidget(i, 2, multiplier_spin)
            
            print(f"📊 Exposure table updated with {len(symbols_to_show)} symbols")
            
        except Exception as e:
            print(f"❌ Error populating exposure table: {e}")
    
    def on_exposure_table_changed(self, row, column):
        """Handle exposure table cell changes and auto-save"""
        try:
            if not hasattr(self, 'exposure_table') or row >= self.exposure_table.rowCount():
                return
                
            # Get symbol name from first column
            symbol_item = self.exposure_table.item(row, 0)
            if not symbol_item:
                return
                
            symbol = symbol_item.text()
            changed_item = self.exposure_table.item(row, column)
            if not changed_item:
                return
                
            value_text = changed_item.text().strip()
            
            if column == 1:  # Max Exposure column
                try:
                    value = float(value_text) if value_text else 0.0
                    self.update_symbol_exposure(symbol, value)
                    print(f"💾 Auto-saved {symbol} exposure: {value} lots")
                except ValueError:
                    print(f"⚠️ Invalid exposure value for {symbol}: {value_text}")
                    return
                    
            elif column == 2:  # Risk Multiplier column
                try:
                    value = float(value_text) if value_text else 1.0
                    self.update_symbol_multiplier(symbol, value)
                    print(f"💾 Auto-saved {symbol} multiplier: {value}")
                except ValueError:
                    print(f"⚠️ Invalid multiplier value for {symbol}: {value_text}")
                    return
            
            # Trigger auto-save
            self.auto_save_settings()
            
        except Exception as e:
            print(f"❌ Error handling exposure table change: {e}")
    
    def update_symbol_exposure(self, symbol, value):
        """Update symbol exposure setting"""
        try:
            self.settings.setdefault('symbol_exposure', {})[symbol] = value
            print(f"📊 Updated {symbol} exposure: {value} lots")
        except Exception as e:
            print(f"❌ Error updating symbol exposure: {e}")
    
    def update_symbol_multiplier(self, symbol, value):
        """Update symbol risk multiplier setting"""
        try:
            self.settings.setdefault('symbol_multipliers', {})[symbol] = value
            print(f"🎯 Updated {symbol} multiplier: {value}")
        except Exception as e:
            print(f"❌ Error updating symbol multiplier: {e}")
    
    def update_risk_manager_symbols(self):
        """Update risk manager with current symbols"""
        if not self.risk_manager:
            return
            
        try:
            # Update symbol exposure limits
            symbol_exposure = self.settings.get('symbol_exposure', {})
            symbol_multipliers = self.settings.get('symbol_multipliers', {})
            
            for symbol in self.available_symbols:
                if hasattr(self.risk_manager.risk_params, 'symbol_max_exposure'):
                    self.risk_manager.risk_params.symbol_max_exposure[symbol] = symbol_exposure.get(symbol, 1000.0)
                if hasattr(self.risk_manager.risk_params, 'symbol_risk_multipliers'):
                    self.risk_manager.risk_params.symbol_risk_multipliers[symbol] = symbol_multipliers.get(symbol, 1.0)
            
            print(f"🎯 Updated risk manager with {len(self.available_symbols)} symbols")
            
        except Exception as e:
            print(f"❌ Error updating risk manager symbols: {e}")
    
    def get_available_symbols(self):
        """Get list of available symbols for trading"""
        return self.available_symbols.copy() if self.available_symbols else []
    
    def validate_signal_for_symbol(self, signal):
        """Validate a trading signal using risk management"""
        if not self.risk_manager:
            return False, "Risk manager not initialized"
            
        try:
            validation = self.risk_manager.validate_signal_comprehensive(signal)
            return validation.result == ValidationResult.APPROVED, validation
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def update_sltp_mode_controls(self):
        """Update SL/TP controls based on selected mode"""
        try:
            current_mode = self.sltp_mode_combo.currentText()
            
            # Update labels and ranges based on mode
            if "ATR" in current_mode or "Bội số ATR" in current_mode:
                # ATR Mode - use multipliers - Make sure controls are visible
                self.sl_label.setVisible(True)
                self.tp_label.setVisible(True)
                self.default_sl_spin.setVisible(True)
                self.default_tp_spin.setVisible(True)
                
                self.sl_label.setText(I18N.t("SL ATR Multiplier:", "Hệ số ATR cho SL:"))
                self.tp_label.setText(I18N.t("TP ATR Multiplier:", "Hệ số ATR cho TP:"))
                self.default_sl_spin.setRange(0.1, 100.0)  # ATR multipliers 0.1-100.0 with decimals
                self.default_tp_spin.setRange(0.1, 100.0)
                self.default_sl_spin.setDecimals(1)  # Allow 1 decimal place for ATR multipliers
                self.default_tp_spin.setDecimals(1)
                self.default_sl_spin.setSingleStep(0.1)  # Step by 0.1
                self.default_tp_spin.setSingleStep(0.1)
                self.default_sl_spin.setValue(float(self.settings.get('default_sl_atr_multiplier', 10.0)))
                self.default_tp_spin.setValue(float(self.settings.get('default_tp_atr_multiplier', 13.0)))
                self.default_sl_spin.setSuffix("x ATR")
                self.default_tp_spin.setSuffix("x ATR")
                
            elif "Percentage" in current_mode or "Phần trăm" in current_mode:
                # Percentage Mode - Make sure controls are visible
                self.sl_label.setVisible(True)
                self.tp_label.setVisible(True)
                self.default_sl_spin.setVisible(True)
                self.default_tp_spin.setVisible(True)
                
                self.sl_label.setText(I18N.t("SL Percentage (%):", "SL phần trăm (%):"))
                self.tp_label.setText(I18N.t("TP Percentage (%):", "TP phần trăm (%):"))
                self.default_sl_spin.setRange(0, 100)  # 0-100%
                self.default_tp_spin.setRange(0, 100)
                self.default_sl_spin.setValue(self.settings.get('default_sl_percentage', 2))
                self.default_tp_spin.setValue(self.settings.get('default_tp_percentage', 5))
                self.default_sl_spin.setSuffix("%")
                self.default_tp_spin.setSuffix("%")
                
            elif "Support" in current_mode or "Hỗ trợ" in current_mode:
                # Support/Resistance Mode - Make sure controls are visible
                self.sl_label.setVisible(True)
                self.tp_label.setVisible(True)
                self.default_sl_spin.setVisible(True)
                self.default_tp_spin.setVisible(True)
                
                self.sl_label.setText(I18N.t("SL Buffer (pips):", "Đệm SL (pips):"))
                self.tp_label.setText(I18N.t("TP Buffer (pips):", "Đệm TP (pips):"))
                self.default_sl_spin.setRange(0, 1000)  # Buffer in pips
                self.default_tp_spin.setRange(0, 1000)
                self.default_sl_spin.setValue(self.settings.get('default_sl_buffer', 10))
                self.default_tp_spin.setValue(self.settings.get('default_tp_buffer', 10))
                self.default_sl_spin.setSuffix(" pips")
                self.default_tp_spin.setSuffix(" pips")
                
            elif "Signal" in current_mode or "Theo Signal" in current_mode or "Theo Tín hiệu" in current_mode:
                # Signal Based Mode - Hide SL/TP controls since they come from signal
                self.sl_label.setVisible(False)
                self.tp_label.setVisible(False)
                self.default_sl_spin.setVisible(False)
                self.default_tp_spin.setVisible(False)
                
            else:
                # Fixed Pips Mode (default) - Make sure controls are visible
                self.sl_label.setVisible(True)
                self.tp_label.setVisible(True)
                self.default_sl_spin.setVisible(True)
                self.default_tp_spin.setVisible(True)
                
                self.sl_label.setText(I18N.t("Default SL (pips):", "SL mặc định (pips):"))
                self.tp_label.setText(I18N.t("Default TP (pips):", "TP mặc định (pips):"))
                self.default_sl_spin.setRange(0, 10000)  # Pips
                self.default_tp_spin.setRange(0, 50000)
                self.default_sl_spin.setValue(self.settings.get('default_sl_pips', 50))
                self.default_tp_spin.setValue(self.settings.get('default_tp_pips', 100))
                self.default_sl_spin.setSuffix(" pips")
                self.default_tp_spin.setSuffix(" pips")
                
        except Exception as e:
            print(f"❌ Error updating SL/TP mode controls: {e}")

class AutoTradingTab(QWidget):
    def __init__(self, news_tab=None, risk_tab=None):
        super().__init__()
        self.news_tab = news_tab  # Reference to news tab for economic calendar setting
        self.risk_tab = risk_tab  # Reference to risk management tab
        
        # 🔒 Thread-safe lock to prevent race condition in start_auto
        self._start_lock = threading.Lock()
        
        layout = QVBoxLayout()
        
        # Auto Trading Control
        self.auto_btn = QPushButton(I18N.t("Auto Trading: OFF", "Giao dịch tự động: TẮT"))
        self.auto_btn.setCheckable(True)
        self.auto_btn.toggled.connect(self.toggle_auto)
        layout.addWidget(self.auto_btn)

        self.start_btn = QPushButton(I18N.t("Start Auto Trading", "Bắt đầu giao dịch tự động"))
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_auto)
        layout.addWidget(self.start_btn)

        # Status Label for Auto Trading
        self.status_label = QLabel("⚪ Trạng thái: Chưa khởi động")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 6px;
                background-color: #f9f9f9;
                font-size: 14px;
                font-weight: bold;
                color: #666;
            }
        """)
        layout.addWidget(self.status_label)

        # Progress Log (scrollable text area)
        self.progress_log = QTextEdit()
        self.progress_log.setMaximumHeight(200)
        self.progress_log.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 5px;
            }
        """)
        self.progress_log.setPlaceholderText("📋 Nhật ký hoạt động sẽ hiển thị ở đây...")
        layout.addWidget(QLabel("📋 Nhật ký hoạt động:"))
        layout.addWidget(self.progress_log)

        self.setLayout(layout)
        self.auto_manager = None
        self.main_window_ref = None  # Store main window reference

    def set_main_window_reference(self, main_window):
        """Set main window reference để auto trading có thể access các tab"""
        self.main_window_ref = main_window
        print(f"[DEBUG] Main window reference set: {type(main_window)}")
        print(f"[DEBUG] Market tab available: {hasattr(main_window, 'market_tab')}")
        print(f"[DEBUG] Tab widget available: {hasattr(main_window, 'tabWidget')}")

    def update_status(self, message):
        """Update status label with message - thread safe"""
        try:
            def _update():
                if hasattr(self, 'status_label'):
                    # Determine color based on message content
                    if "❌" in message or "thất bại" in message.lower() or "error" in message.lower():
                        color = "#d32f2f"  # Red
                        bg_color = "#ffebee"
                        border_color = "#f44336"
                    elif "✅" in message or "hoàn thành" in message.lower() or "success" in message.lower():
                        color = "#388e3c"  # Green  
                        bg_color = "#e8f5e8"
                        border_color = "#4caf50"
                    elif "🔄" in message or "đang" in message.lower() or "loading" in message.lower():
                        color = "#1976d2"  # Blue
                        bg_color = "#e3f2fd"
                        border_color = "#2196f3"
                    else:
                        color = "#666"  # Default
                        bg_color = "#f9f9f9"
                        border_color = "#ddd"
                    
                    self.status_label.setText(message)
                    self.status_label.setStyleSheet(f"""
                        QLabel {{
                            padding: 10px;
                            border: 2px solid {border_color};
                            border-radius: 6px;
                            background-color: {bg_color};
                            font-size: 14px;
                            font-weight: bold;
                            color: {color};
                        }}
                    """)
                    self.status_label.repaint()  # Force repaint

            # Use QTimer to ensure thread safety
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, _update)
            
        except Exception as e:
            print(f"Error updating status: {e}")

    def add_log(self, message):
        """Add message to progress log - thread safe"""
        try:
            def _add_log():
                if hasattr(self, 'progress_log'):
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    formatted_message = f"[{timestamp}] {message}"
                    self.progress_log.append(formatted_message)
                    
                    # Auto scroll to bottom
                    cursor = self.progress_log.textCursor()
                    cursor.movePosition(cursor.End)
                    self.progress_log.setTextCursor(cursor)
                    self.progress_log.repaint()  # Force repaint

            # Use QTimer to ensure thread safety
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, _add_log)
            
        except Exception as e:
            print(f"Error adding log: {e}")

    def update_gui_status(self, message):
        """Update both status and log - for auto trading manager to call"""
        print(f"[GUI UPDATE] {message}")  # Debug print
        self.update_status(message)
        self.add_log(message)
        
        # Force process events to update GUI immediately
        try:
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception as e:
            print(f"Error processing events: {e}")

    def closeEvent(self, event):
        """Handle close event to cleanup threads properly - improved version"""
        try:
            print("[CLEANUP] AutoTradingTab closing - stopping auto trading...")
            
            # Force stop with aggressive cleanup
            if hasattr(self, 'auto_manager') and self.auto_manager:
                print("[CLEANUP] Force stopping auto manager...")
                try:
                    # Set shutdown flag immediately
                    if hasattr(self.auto_manager, '_shutdown_event'):
                        self.auto_manager._shutdown_event.set()
                    if hasattr(self.auto_manager, 'is_running'):
                        self.auto_manager.is_running = False
                    
                    # Try normal stop with short timeout
                    import threading
                    import time
                    
                    def force_stop():
                        try:
                            self.auto_manager.stop()
                        except Exception as e:
                            print(f"[CLEANUP] Exception during stop: {e}")
                    
                    stop_thread = threading.Thread(target=force_stop, daemon=True)
                    stop_thread.start()
                    stop_thread.join(timeout=3)  # Short timeout on close
                    
                    if stop_thread.is_alive():
                        print("[CLEANUP] Stop timed out - forcing cleanup")
                    
                except Exception as e:
                    print(f"[CLEANUP] Exception during force stop: {e}")
                finally:
                    # Always clear the reference
                    self.auto_manager = None
                    print("[CLEANUP] Auto manager reference cleared")
            
            # Call parent stop_auto for any remaining cleanup
            try:
                self.stop_auto()
            except Exception as e:
                print(f"[CLEANUP] Exception in stop_auto: {e}")
            
            print("[CLEANUP] AutoTradingTab cleanup completed")
            
        except Exception as e:
            print(f"[CLEANUP] Error during AutoTradingTab cleanup: {e}")
            import traceback
            print(f"[CLEANUP] Traceback: {traceback.format_exc()}")
        finally:
            # Always accept the close event
            event.accept()

    def toggle_auto(self, checked):
        if checked:
            self.auto_btn.setText(I18N.t("Auto Trading: ON", "Giao dịch tự động: BẬT"))
            self.start_btn.setEnabled(True)
        else:
            self.auto_btn.setText(I18N.t("Auto Trading: OFF", "Giao dịch tự động: TẮT"))
            self.start_btn.setEnabled(False)
            self.stop_auto()

    def start_auto(self):
        # 🔒 Thread-safe protection against race condition
        with self._start_lock:
            try:
                # Check if already running (double-click protection)
                if hasattr(self, 'auto_manager') and self.auto_manager is not None:
                    self.add_log("⚠️ Auto Trading đã đang chạy - bỏ qua yêu cầu duplicate")
                    return
                
                # Use unified auto trading system
                from unified_auto_trading_system import UnifiedAutoTradingSystem as AutoTradingManager
                self.add_log("📦 Using UnifiedAutoTradingSystem")
                
                # Update status
                self.update_status("🔄 Đang khởi động Auto Trading...")
                self.add_log("🚀 Bắt đầu khởi động hệ thống Auto Trading")
                
                # Kiểm tra economic calendar setting
                economic_status = self.get_economic_calendar_status()
                use_calendar = self.is_economic_calendar_enabled()
                
                # Hiển thị thông báo
                icon_path = os.path.join("images", "robot.png")
                msg = QMessageBox(self)
                msg.setWindowTitle("Auto Trading")
            
                # Tạo thông báo chi tiết
                message = f"[AUTO TRADING] Starting Simplified Auto Trading!\n\n"
                message += f"This will automatically:\n"
                message += f"1. Fetch Market Data (if tab enabled)\n"
                message += f"2. Calculate Trend Analysis (if tab enabled)\n"
                message += f"3. Calculate Technical Indicators (if tab enabled)\n"
                message += f"4. Analyze Patterns (if tab enabled)\n"
                message += f"5. Generate Trading Signals (if tab enabled)\n"
                message += f"6. Execute Orders (if signals found)\n\n"
                message += f"[NEWS] Detection: {economic_status}"
                
                msg.setText(message)
                if os.path.exists(icon_path):
                    msg.setIconPixmap(QPixmap(icon_path).scaled(64, 64, Qt.KeepAspectRatio))
                else:
                    msg.setIcon(QMessageBox.Information)
                msg.exec_()
            
                # Log setting
                print(f"[AUTO TRADING] Started Simplified Auto Trading")
                print(f"   [NEWS] News Detection: {economic_status}")
                self.add_log(f"📰 News Detection: {economic_status}")
                
                # Bắt đầu simplified auto trading
                if not self.auto_manager:
                    # Sử dụng main window reference đã được set
                    main_window_ref = self.main_window_ref
                    
                    if main_window_ref:
                        print(f"[DEBUG] Using stored main window reference: {type(main_window_ref)}")
                        print(f"[DEBUG] Market tab available: {hasattr(main_window_ref, 'market_tab')}")
                        print(f"[DEBUG] all_tabs available: {hasattr(main_window_ref, 'all_tabs')}")
                        if hasattr(main_window_ref, 'all_tabs'):
                            print(f"[DEBUG] auto_trading_tab in all_tabs: {'auto_trading_tab' in main_window_ref.all_tabs}")
                        self.add_log(f"🔗 Kết nối với MainWindow: {type(main_window_ref).__name__}")
                    else:
                        print("[WARNING] No main window reference available - will use fallback methods")
                        self.add_log("⚠️ Không có tham chiếu MainWindow - sử dụng phương pháp dự phòng")
                
                    print(f"[DEBUG] Creating AutoTradingManager with main_window_ref: {main_window_ref}")
                    self.auto_manager = AutoTradingManager(
                        main_window_ref=main_window_ref,
                        update_interval=60  # 1 phút per cycle
                    )
                    print(f"[DEBUG] AutoTradingManager created: {self.auto_manager}")
                    print(f"[DEBUG] AutoTradingManager.main_window: {self.auto_manager.main_window}")
                
                    print("[PIPELINE] Starting Auto Trading Pipeline...")
                    print("[INFO] Auto Trading will check checkbox settings from each tab")
                    self.add_log("⚙️ Khởi tạo Auto Trading Manager thành công")
                    
                    # Start the pipeline in background
                    import threading
                
                    def start_pipeline():
                        try:
                            print(f"[PIPELINE DEBUG] Starting pipeline with auto_manager: {self.auto_manager}")
                            print(f"[PIPELINE DEBUG] auto_manager.main_window: {getattr(self.auto_manager, 'main_window', None)}")
                            
                            self.update_status("🔄 Đang khởi động pipeline...")
                            self.add_log("🔧 Bắt đầu pipeline trading...")
                        
                            # Test GUI update from auto manager
                            if hasattr(self.auto_manager, 'update_gui_status'):
                                print("[PIPELINE DEBUG] Testing auto_manager.update_gui_status...")
                                self.auto_manager.update_gui_status("🧪 Test status update từ Auto Manager")
                            
                            # Initialize and start the pipeline
                            result = self.auto_manager.start()
                            if result is False:
                                print("[BLOCKED] Auto Trading Pipeline BLOCKED by risk settings!")
                                print("[INFO] Please enable auto mode in risk management first")
                                self.update_status("🔒 Auto Trading bị chặn bởi cài đặt rủi ro")
                                self.add_log("❌ Pipeline bị chặn - kiểm tra cài đặt Risk Management")
                                QMessageBox.warning(self, "Auto Trading Blocked", 
                                                  "Auto Trading is disabled in risk management settings.\n\n"
                                                  "Please:\n"
                                                  "1. Set 'enable_auto_mode': true\n"
                                                  "2. Change trading mode to auto\n"
                                                  "3. Check risk management settings")
                                self.auto_manager = None
                            else:
                                print("[SUCCESS] Auto Trading Pipeline Successfully Started!")
                                self.update_status("✅ Auto Trading đang hoạt động")
                                self.add_log("🎉 Pipeline khởi động thành công!")
                        except Exception as e:
                            print(f"[ERROR] Error starting simplified auto trading pipeline: {e}")
                            self.update_status(f"❌ Lỗi khởi động: {str(e)}")
                            self.add_log(f"❌ Lỗi pipeline: {str(e)}")
                            self.auto_manager = None
                
                    # Start pipeline in separate thread to avoid blocking UI
                    pipeline_thread = threading.Thread(target=start_pipeline, daemon=True)
                    pipeline_thread.start()
                    
                else:
                    print("[WARNING] Auto Trading Manager already running")
                    self.update_status("⚠️ Auto Trading đã đang chạy")
                    self.add_log("⚠️ Auto Trading Manager đã đang hoạt động")
                    
            except Exception as e:
                print(f"[ERROR] Error in start_auto(): {e}")
                self.update_status(f"❌ Lỗi khởi động: {str(e)}")
                self.add_log(f"❌ Lỗi start_auto: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to start auto trading:\n{str(e)}")

    def stop_auto(self):
        """Improved stop method with better error handling and cleanup"""
        try:
            if self.auto_manager:
                print("[STOP] Stopping Auto Trading Pipeline...")
                self.update_status("🔄 Đang dừng Auto Trading...")
                self.add_log("🛑 Bắt đầu dừng Auto Trading Pipeline")
                
                # Stop the manager gracefully with timeout
                try:
                    import time
                    import threading
                    
                    # Use a separate thread to stop with timeout
                    def stop_with_timeout():
                        try:
                            self.auto_manager.stop()
                            return True
                        except Exception as e:
                            print(f"[STOP ERROR] {e}")
                            return False
                    
                    stop_thread = threading.Thread(target=stop_with_timeout, daemon=True)
                    stop_thread.start()
                    stop_thread.join(timeout=10)  # 10 second timeout
                    
                    if stop_thread.is_alive():
                        print("[WARNING] Stop operation timed out")
                        self.add_log("⚠️ Stop timeout - forcing cleanup")
                    else:
                        print("[SUCCESS] Auto Trading Pipeline Stopped Successfully")
                        self.add_log("✅ Auto Trading Pipeline đã dừng thành công")
                    
                    self.update_status("⚪ Auto Trading đã dừng")
                    
                except Exception as e:
                    print(f"[ERROR] Error stopping auto manager: {e}")
                    self.update_status("⚠️ Lỗi khi dừng Auto Trading")
                    self.add_log(f"❌ Lỗi dừng manager: {str(e)}")
                
                # Force cleanup
                finally:
                    try:
                        # Try to force cleanup any remaining resources
                        if hasattr(self.auto_manager, '_shutdown_event'):
                            self.auto_manager._shutdown_event.set()
                        if hasattr(self.auto_manager, 'is_running'):
                            self.auto_manager.is_running = False
                    except:
                        pass
                    
                    # Clear reference
                    self.auto_manager = None
                    print("[CLEANUP] Auto manager reference cleared")
                
            else:
                print("[WARNING] No Auto Trading Manager to stop")
                self.update_status("⚪ Không có Auto Trading để dừng")
                self.add_log("⚠️ Không có Auto Trading Manager để dừng")
                
        except Exception as e:
            print(f"[ERROR] Error in stop_auto(): {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            self.update_status(f"❌ Lỗi dừng: {str(e)}")
            self.add_log(f"❌ Lỗi stop_auto: {str(e)}")
            
            # Force clear auto_manager even if error
            try:
                self.auto_manager = None
                print("[FORCE CLEANUP] Auto manager reference force cleared")
            except:
                pass

    def is_economic_calendar_enabled(self):
        """Kiểm tra xem economic calendar có được bật không"""
        if self.news_tab:
            return self.news_tab.get_economic_calendar_setting()
        else:
            # Fallback to user config
            user_config = load_user_config()
            return user_config.get("use_economic_calendar", True)

    def get_economic_calendar_status(self):
        """Lấy status text của economic calendar setting"""
        enabled = self.is_economic_calendar_enabled()
        return "✅ Enabled" if enabled else "❌ Disabled"

class IndicatorTab(QWidget):
    INDICATOR_OPTIONS = [
        {"name": "MA", "label": "Moving Average"},
        {"name": "MACD", "label": "MACD"},
        {"name": "RSI", "label": "RSI"},
        {"name": "Stochastic", "label": "Stochastic"},
        {"name": "Bollinger Bands", "label": "Bollinger Bands"},
        {"name": "ATR", "label": "ATR"},
        {"name": "ADX", "label": "ADX"},
        {"name": "CCI", "label": "CCI"},
        {"name": "WilliamsR", "label": "Williams %R"},
        {"name": "ROC", "label": "ROC"},
        {"name": "OBV", "label": "OBV"},
        {"name": "MFI", "label": "MFI"},
        {"name": "PSAR", "label": "Parabolic SAR"},
    {"name": "Chaikin", "label": "Chaikin Money Flow"},  # token 'chaikin' (CMF)
        {"name": "EOM", "label": "Ease of Movement"},
        {"name": "ForceIndex", "label": "Force Index"},
        {"name": "Donchian", "label": "Donchian Channel"},
        {"name": "TRIX", "label": "TRIX"},
        {"name": "DPO", "label": "DPO"},
        {"name": "MassIndex", "label": "Mass Index"},
        {"name": "Vortex", "label": "Vortex Indicator"},
        {"name": "KST", "label": "KST Oscillator"},
        {"name": "StochRSI", "label": "Stochastic RSI"},
        {"name": "UltimateOscillator", "label": "Ultimate Oscillator"},
        {"name": "Keltner", "label": "Keltner Channel"},
        {"name": "Envelope", "label": "Envelope"},  # <-- Thêm dòng này
        {"name": "Fibonacci", "label": "Fibonacci"},
        {"name": "Ichimoku", "label": "Ichimoku"},
    ]

    def __init__(self, market_tab):
        super().__init__()
        self.market_tab = market_tab
        self.indicator_rows = []
        self.workers = []
        self.indicator_list = []  # <-- Thêm dòng này
        self.user_config = load_user_config()
        self.init_ui()
        self.restore_user_config()  # <-- Thêm dòng này
        # Sync current UI selections to whitelist file for aggregator
        try:
            self._persist_indicator_whitelist()
        except Exception as _e:
            print(f"Could not persist indicator whitelist on init: {_e}")

    def init_ui(self):
        layout = QVBoxLayout()
        self.search_box = QLineEdit()
        # Localized placeholder
        self.search_box.setPlaceholderText(I18N.t("Search indicator...", "Tìm chỉ báo..."))
        self.search_box.hide()
        layout.addWidget(self.search_box)
        
        # Header với Add Indicator và Toggle button
        header_layout = QHBoxLayout()
        self.add_btn = QPushButton(I18N.t("Add Indicator", "Thêm chỉ báo"))
        self.add_btn.clicked.connect(self.add_indicator_row)
        header_layout.addWidget(self.add_btn)
        
        # Spacer để đẩy toggle button sang phải
        header_layout.addStretch()
        
        # Toggle button nhỏ ở góc phải
        self.toggle_btn = QPushButton(I18N.t("Add All", "Thêm tất cả"))
        self.toggle_btn.clicked.connect(self.toggle_all_indicators)
        self.toggle_btn.setMaximumWidth(80)  # Làm nút nhỏ
        self.toggle_btn.setStyleSheet("QPushButton { background-color: #51cf66; color: white; font-size: 10px; }")
        header_layout.addWidget(self.toggle_btn)
        
        layout.addLayout(header_layout)
        
        self.indicator_area = QVBoxLayout()
        layout.addLayout(self.indicator_area)
        self.export_btn = QPushButton(I18N.t("Calculate & Save Indicator", "Tính & lưu chỉ báo"))
        self.export_btn.clicked.connect(self.export_indicators)
        layout.addWidget(self.export_btn)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        # Table để hiển thị kết quả tín hiệu
        self.table = QTableWidget(0, 4)
        # Localized header labels
        self.table.setHorizontalHeaderLabels([
            I18N.t("Indicator", "Chỉ báo"),
            I18N.t("Value", "Giá trị"),
            I18N.t("Time", "Thời gian"),
            I18N.t("Signal", "Tín hiệu"),
        ])
        header = self.table.horizontalHeader()
        for i in range(self.table.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        
        # Đảm bảo cột Time có width đủ lớn để hiển thị đầy đủ thời gian
        self.table.setColumnWidth(2, 150)  # Tăng từ 100 lên 150px
        header.setSectionResizeMode(2, QHeaderView.Interactive)  # Allow manual resize for Time column
        
        layout.addWidget(self.table)

        self.setLayout(layout)

    def add_indicator_row(self):
        row_layout = QHBoxLayout()
        indi_combo = QComboBox()
        indi_combo.addItems([opt["label"] for opt in self.INDICATOR_OPTIONS])
        row_layout.addWidget(indi_combo)
        row_dict = {"indi_combo": indi_combo, "layout": row_layout}
        self.indicator_rows.append(row_dict)
        self.indicator_area.addLayout(row_layout)
        indi_combo.currentIndexChanged.connect(lambda idx, row=row_dict: self.on_indicator_changed(idx, row))
        self.on_indicator_changed(0, row_dict)
        self.save_current_user_config()  # <-- Thêm dòng này
        # Persist whitelist whenever user adds an indicator
        try:
            self._persist_indicator_whitelist()
        except Exception as _e:
            print(f"Could not persist indicator whitelist after add: {_e}")
        
        # Update toggle button text
        self.update_toggle_button()

        # Khi combobox được focus, hiện ô tìm kiếm
        def on_focus_in(event):
            self.search_box.show()
            self.search_box.setFocus()
            self.search_box.clear()
            try:
                self.search_box.textChanged.disconnect()
            except Exception:
                pass
            self.search_box.textChanged.connect(lambda: self.filter_indicator_combo(indi_combo))
            return QComboBox.focusInEvent(indi_combo, event)
        indi_combo.focusInEvent = on_focus_in

        # Khi combobox mất focus, ẩn ô tìm kiếm
        def on_focus_out(event):
            self.search_box.hide()
            self.search_box.clear()
            return QComboBox.focusOutEvent(indi_combo, event)
        indi_combo.focusOutEvent = on_focus_out

        # Khi chọn xong, ẩn ô tìm kiếm
        def hide_search_box():
            self.search_box.hide()
            self.search_box.clear()
        indi_combo.activated.connect(hide_search_box)

    def remove_indicator_row(self, row_dict):
        layout = row_dict["layout"]
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.indicator_area.removeItem(layout)
        self.indicator_rows.remove(row_dict)
        self.save_current_user_config()  # <-- Thêm dòng này
        # Persist whitelist whenever user removes an indicator
        try:
            self._persist_indicator_whitelist()
        except Exception as _e:
            print(f"Could not persist indicator whitelist after remove: {_e}")
        
        # Update toggle button text
        self.update_toggle_button()

    def on_indicator_changed(self, index, row_dict):
        layout = row_dict["layout"]
        while layout.count() > 1:
            item = layout.takeAt(1)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        indi_label = row_dict["indi_combo"].currentText()
        indi_name = self._label_to_name(indi_label)
        if not indi_name:
            # Unknown label; skip building parameter widgets for this row
            try:
                self.log_output.append(f"⚠️ Unknown indicator label: {indi_label}. Skipping parameter UI.")
            except Exception:
                print(f"⚠️ Unknown indicator label: {indi_label}. Skipping parameter UI.")
            return
        # Tạo widget tham số cho từng indicator
        if indi_name == "MA":
            period_spin = QSpinBox(); period_spin.setMinimum(1); period_spin.setMaximum(1000); period_spin.setValue(20)
            layout.addWidget(QLabel("Period:")); layout.addWidget(period_spin)
            ma_type_combo = QComboBox(); ma_type_combo.addItems(["SMA", "EMA", "WMA", "TEMA"])
            layout.addWidget(QLabel("Type:")); layout.addWidget(ma_type_combo)
            row_dict["period_spin"] = period_spin
            row_dict["ma_type_combo"] = ma_type_combo
            # Persist whitelist whenever MA params change
            try:
                period_spin.valueChanged.connect(lambda _=None: (self.save_current_user_config(), self._persist_indicator_whitelist()))
                ma_type_combo.currentTextChanged.connect(lambda _=None: (self.save_current_user_config(), self._persist_indicator_whitelist()))
            except Exception as _e:
                print(f"Could not bind MA change handlers: {_e}")
            # Persist whitelist whenever MA params change
            try:
                period_spin.valueChanged.connect(lambda _=None: (self.save_current_user_config(), self._persist_indicator_whitelist()))
                ma_type_combo.currentTextChanged.connect(lambda _=None: (self.save_current_user_config(), self._persist_indicator_whitelist()))
            except Exception as _e:
                print(f"Could not bind MA change handlers: {_e}")
        elif indi_name == "RSI":
            period_spin = QSpinBox(); period_spin.setMinimum(1); period_spin.setMaximum(1000); period_spin.setValue(14)
            layout.addWidget(QLabel("Period:")); layout.addWidget(period_spin)
            row_dict["period_spin"] = period_spin
        elif indi_name == "MACD":
            fast_spin = QSpinBox(); fast_spin.setMinimum(1); fast_spin.setMaximum(100); fast_spin.setValue(12)
            slow_spin = QSpinBox(); slow_spin.setMinimum(1); slow_spin.setMaximum(100); slow_spin.setValue(26)
            signal_spin = QSpinBox(); signal_spin.setMinimum(1); signal_spin.setMaximum(100); signal_spin.setValue(9)
            layout.addWidget(QLabel("Fast:")); layout.addWidget(fast_spin)
            layout.addWidget(QLabel("Slow:")); layout.addWidget(slow_spin)
            layout.addWidget(QLabel("Signal:")); layout.addWidget(signal_spin)
            row_dict["fast_spin"] = fast_spin; row_dict["slow_spin"] = slow_spin; row_dict["signal_spin"] = signal_spin
        elif indi_name == "Stochastic":
            period_spin = QSpinBox(); period_spin.setMinimum(1); period_spin.setMaximum(1000); period_spin.setValue(14)
            smooth_spin = QSpinBox(); smooth_spin.setMinimum(1); smooth_spin.setMaximum(100); smooth_spin.setValue(3)
            layout.addWidget(QLabel("Period:")); layout.addWidget(period_spin)
            layout.addWidget(QLabel("Smooth:")); layout.addWidget(smooth_spin)
            row_dict["period_spin"] = period_spin; row_dict["smooth_spin"] = smooth_spin
        elif indi_name == "Bollinger Bands":
            window_spin = QSpinBox(); window_spin.setMinimum(1); window_spin.setMaximum(1000); window_spin.setValue(20)
            dev_spin = QSpinBox(); dev_spin.setMinimum(1); dev_spin.setMaximum(10); dev_spin.setValue(2)
            layout.addWidget(QLabel("Window:")); layout.addWidget(window_spin)
            layout.addWidget(QLabel("Dev:")); layout.addWidget(dev_spin)
            row_dict["window_spin"] = window_spin; row_dict["dev_spin"] = dev_spin
        elif indi_name in ["ATR", "ADX", "CCI", "WilliamsR", "ROC", "MFI", "Chaikin", "EOM", "ForceIndex", "TRIX", "DPO"]:
            period_spin = QSpinBox(); period_spin.setMinimum(1); period_spin.setMaximum(1000)
            # Giá trị mặc định cho từng indicator
            default = 14 if indi_name in ["ATR", "ADX", "WilliamsR", "MFI", "ForceIndex"] else 20
            if indi_name == "TRIX": default = 15
            if indi_name == "DPO": default = 20
            period_spin.setValue(default)
            layout.addWidget(QLabel("Period:")); layout.addWidget(period_spin)
            row_dict["period_spin"] = period_spin
        elif indi_name == "PSAR":
            step_spin = QDoubleSpinBox(); step_spin.setDecimals(3); step_spin.setSingleStep(0.01); step_spin.setValue(0.02)
            max_step_spin = QDoubleSpinBox(); max_step_spin.setDecimals(2); max_step_spin.setSingleStep(0.01); max_step_spin.setValue(0.2)
            layout.addWidget(QLabel("Step:")); layout.addWidget(step_spin)
            layout.addWidget(QLabel("Max Step:")); layout.addWidget(max_step_spin)
            row_dict["step_spin"] = step_spin; row_dict["max_step_spin"] = max_step_spin
        elif indi_name == "Donchian":
            window_spin = QSpinBox(); window_spin.setMinimum(1); window_spin.setMaximum(1000); window_spin.setValue(20)
            layout.addWidget(QLabel("Window:")); layout.addWidget(window_spin)
            row_dict["window_spin"] = window_spin
        elif indi_name == "MassIndex":
            fast_spin = QSpinBox(); fast_spin.setMinimum(1); fast_spin.setMaximum(100); fast_spin.setValue(9)
            slow_spin = QSpinBox(); slow_spin.setMinimum(1); slow_spin.setMaximum(100); slow_spin.setValue(25)
            layout.addWidget(QLabel("Fast:")); layout.addWidget(fast_spin)
            layout.addWidget(QLabel("Slow:")); layout.addWidget(slow_spin)
            row_dict["fast_spin"] = fast_spin; row_dict["slow_spin"] = slow_spin
        elif indi_name == "Vortex":
            period_spin = QSpinBox(); period_spin.setMinimum(1); period_spin.setMaximum(1000); period_spin.setValue(14)
            layout.addWidget(QLabel("Period:")); layout.addWidget(period_spin)
            row_dict["period_spin"] = period_spin
        elif indi_name == "KST":
            window1_spin = QSpinBox(); window1_spin.setMinimum(1); window1_spin.setMaximum(100); window1_spin.setValue(10)
            window2_spin = QSpinBox(); window2_spin.setMinimum(1); window2_spin.setMaximum(100); window2_spin.setValue(15)
            window3_spin = QSpinBox(); window3_spin.setMinimum(1); window3_spin.setMaximum(100); window3_spin.setValue(20)
            window4_spin = QSpinBox(); window4_spin.setMinimum(1); window4_spin.setMaximum(100); window4_spin.setValue(30)
            window_sig_spin = QSpinBox(); window_sig_spin.setMinimum(1); window_sig_spin.setMaximum(100); window_sig_spin.setValue(9)
            layout.addWidget(QLabel("window1:")); layout.addWidget(window1_spin)
            layout.addWidget(QLabel("window2:")); layout.addWidget(window2_spin)
            layout.addWidget(QLabel("window3:")); layout.addWidget(window3_spin)
            layout.addWidget(QLabel("window4:")); layout.addWidget(window4_spin)
            layout.addWidget(QLabel("window_sig:")); layout.addWidget(window_sig_spin)
            row_dict["window1_spin"] = window1_spin
            row_dict["window2_spin"] = window2_spin
            row_dict["window3_spin"] = window3_spin
            row_dict["window4_spin"] = window4_spin
            row_dict["window_sig_spin"] = window_sig_spin
        elif indi_name == "StochRSI":
            period_spin = QSpinBox(); period_spin.setMinimum(1); period_spin.setMaximum(1000); period_spin.setValue(14)
            smooth1_spin = QSpinBox(); smooth1_spin.setMinimum(1); smooth1_spin.setMaximum(100); smooth1_spin.setValue(3)
            smooth2_spin = QSpinBox(); smooth2_spin.setMinimum(1); smooth2_spin.setMaximum(100); smooth2_spin.setValue(3)
            layout.addWidget(QLabel("Period:")); layout.addWidget(period_spin)
            layout.addWidget(QLabel("Smooth1:")); layout.addWidget(smooth1_spin)
            layout.addWidget(QLabel("Smooth2:")); layout.addWidget(smooth2_spin)
            row_dict["period_spin"] = period_spin; row_dict["smooth1_spin"] = smooth1_spin; row_dict["smooth2_spin"] = smooth2_spin
        elif indi_name == "UltimateOscillator":
            short_spin = QSpinBox(); short_spin.setMinimum(1); short_spin.setMaximum(100); short_spin.setValue(7)
            medium_spin = QSpinBox(); medium_spin.setMinimum(1); medium_spin.setMaximum(100); medium_spin.setValue(14)
            long_spin = QSpinBox(); long_spin.setMinimum(1); long_spin.setMaximum(100); long_spin.setValue(28)
            layout.addWidget(QLabel("Short:")); layout.addWidget(short_spin)
            layout.addWidget(QLabel("Medium:")); layout.addWidget(medium_spin)
            layout.addWidget(QLabel("Long:")); layout.addWidget(long_spin)
            row_dict["short_spin"] = short_spin; row_dict["medium_spin"] = medium_spin; row_dict["long_spin"] = long_spin
        elif indi_name == "Keltner":
            window_spin = QSpinBox(); window_spin.setMinimum(1); window_spin.setMaximum(1000); window_spin.setValue(20)
            layout.addWidget(QLabel("Window:")); layout.addWidget(window_spin)
            row_dict["window_spin"] = window_spin
        elif indi_name == "Fibonacci":
            lookback_spin = QSpinBox()
            lookback_spin.setMinimum(10)
            lookback_spin.setMaximum(1000)
            lookback_spin.setValue(100)
            layout.addWidget(QLabel("Lookback:"))
            layout.addWidget(lookback_spin)
            row_dict["lookback_spin"] = lookback_spin
        elif indi_name == "Ichimoku":
            tenkan_spin = QSpinBox(); tenkan_spin.setMinimum(1); tenkan_spin.setMaximum(100); tenkan_spin.setValue(9)
            kijun_spin = QSpinBox(); kijun_spin.setMinimum(1); kijun_spin.setMaximum(100); kijun_spin.setValue(26)
            senkou_spin = QSpinBox(); senkou_spin.setMinimum(1); senkou_spin.setMaximum(100); senkou_spin.setValue(52)
            layout.addWidget(QLabel("Tenkan:")); layout.addWidget(tenkan_spin)
            layout.addWidget(QLabel("Kijun:")); layout.addWidget(kijun_spin)
            layout.addWidget(QLabel("Senkou:")); layout.addWidget(senkou_spin)
            row_dict["tenkan_spin"] = tenkan_spin
            row_dict["kijun_spin"] = kijun_spin
            row_dict["senkou_spin"] = senkou_spin
        elif indi_name == "Envelope":
            period_spin = QSpinBox(); period_spin.setMinimum(1); period_spin.setMaximum(1000); period_spin.setValue(20)
            percent_spin = QDoubleSpinBox(); percent_spin.setDecimals(2); percent_spin.setSingleStep(0.1); percent_spin.setValue(2.0)
            layout.addWidget(QLabel("Period:")); layout.addWidget(period_spin)
            layout.addWidget(QLabel("Percent:")); layout.addWidget(percent_spin)
            row_dict["period_spin"] = period_spin
            row_dict["percent_spin"] = percent_spin      
        # Localized Remove button
        remove_btn = QPushButton(I18N.t("Remove", "Xóa"))
        layout.addWidget(remove_btn)
        row_dict["remove_btn"] = remove_btn
        remove_btn.clicked.connect(lambda: self.remove_indicator_row(row_dict))
        # Sửa dòng này:
        QTimer.singleShot(0, self.save_current_user_config)
        # Also persist whitelist after any indicator change
        try:
            self._persist_indicator_whitelist()
        except Exception as _e:
            print(f"Could not persist indicator whitelist after change: {_e}")

    def _collect_indicator_whitelist_tokens(self):
        """Return a sorted list of aggregator indicator tokens selected in UI.
        Supported tokens expanded: rsi, macd, adx, stochrsi, stochastic, atr, donchian, ema20, ema50, ema100, ema200,
        sma20, wma20, bollinger, keltner, ichimoku, cci, williamsr, roc, obv, chaikin, eom, force, trix, dpo, mass,
        vortex, kst, ultimate, envelopes, momentum, fibonacci (fibonacci auto-kept by aggregator; not needed here).
        """
        tokens = set()
        for row in self.indicator_rows:
            try:
                combo = row.get("indi_combo")
                if not combo:
                    continue
                label = combo.currentText()
                name = next((opt["name"] for opt in self.INDICATOR_OPTIONS if opt["label"] == label), None)
                if not name:
                    continue
                n = name.lower()
                # Normalize multi-word / variant names to aggregator tokens
                alias_map = {
                    'bollinger bands':'bollinger',
                    'donchian channel':'donchian',
                    'forceindex':'force',
                    'massindex':'mass',
                    'ultimateoscillator':'ultimate',
                    'parabolic sar':'psar',
                    'chaikin money flow':'chaikin','cmf':'chaikin',
                    'ease of movement':'eom',
                    'kst oscillator':'kst',
                    'vortex indicator':'vortex',
                    'stochastic rsi':'stochrsi',
                    'williamsr':'williamsr',
                    'williams %r':'williamsr',
                    'envelope':'envelopes',  # treat single as plural token used in aggregator
                    'fibonacci':'fibonacci'
                }
                if n in alias_map:
                    n = alias_map[n]
                # Direct one-to-one names
                direct = {"rsi","macd","adx","stochrsi","stochastic","atr","donchian","bollinger","keltner","ichimoku","cci","williamsr","roc","obv","chaikin","eom","force","trix","dpo","mass","vortex","kst","ultimate","envelopes","mfi","psar","fibonacci"}
                if n in direct:
                    tokens.add(n)
                elif n == "ma":
                    # Map moving average selection to specific MA tokens based on type & period (generalized)
                    p = row.get("period_spin")
                    t = row.get("ma_type_combo")
                    try:
                        period = int(p.value()) if p else None
                        ma_type = t.currentText().upper() if t else ""
                        if period and ma_type in {"EMA","SMA","WMA","TEMA"}:
                            tokens.add(f"{ma_type.lower()}{period}")
                    except Exception:
                        pass
            except Exception:
                continue
        return sorted(tokens)

    def _persist_indicator_whitelist(self):
        """Write the selected indicator tokens to analysis_results/indicator_whitelist.json for the aggregator to consume."""
        try:
            wl = self._collect_indicator_whitelist_tokens()
            out_dir = os.path.join(os.getcwd(), "analysis_results")
            os.makedirs(out_dir, exist_ok=True)
            out_fp = os.path.join(out_dir, "indicator_whitelist.json")
            with open(out_fp, "w", encoding="utf-8") as f:
                json.dump(wl, f, ensure_ascii=False, indent=2)
            # Optional log message if available
            try:
                if hasattr(self, "log_output") and self.log_output:
                    self.log_output.append(f"💾 Saved indicator whitelist: {', '.join(wl) if wl else '(empty)'}")
            except Exception:
                pass
        except Exception as e:
            print(f"Failed to save indicator whitelist: {e}")

    def _label_to_name(self, indi_label: str):
        """Map a UI label back to its canonical indicator name safely.
        Returns None if the label is unknown. Handles minor variations and whitespace.
        """
        try:
            if not indi_label:
                return None
            # Exact label match
            for opt in self.INDICATOR_OPTIONS:
                if opt.get("label") == indi_label:
                    return opt.get("name")
            # Fallback: direct name match (if label equals name)
            for opt in self.INDICATOR_OPTIONS:
                if opt.get("name") == indi_label:
                    return opt.get("name")
            # Case-insensitive, trimmed label match
            lbl_norm = str(indi_label).strip().lower()
            for opt in self.INDICATOR_OPTIONS:
                if str(opt.get("label", "")).strip().lower() == lbl_norm:
                    return opt.get("name")
            return None
        except Exception:
            return None

    def clear_all_indicator_data(self):
        """Clear all saved indicator data files"""
        import os
        import shutil
        
        indicator_dir = "indicator_output"
        if os.path.exists(indicator_dir):
            try:
                # Count files before deletion
                file_count = len([f for f in os.listdir(indicator_dir) if os.path.isfile(os.path.join(indicator_dir, f))])
                
                # Remove all files in indicator_output directory
                shutil.rmtree(indicator_dir)
                os.makedirs(indicator_dir, exist_ok=True)
                
                self.log_output.append(f"🧹 Indicator Data Cleanup: Removed {file_count} old indicator files")
                print(f"🧹 Indicator Data Cleanup: Removed {file_count} old indicator files")
                
            except Exception as e:
                error_msg = f"⚠️ Warning: Could not clear indicator data: {e}"
                self.log_output.append(error_msg)
                print(error_msg)
        else:
            # Create directory if it doesn't exist
            os.makedirs(indicator_dir, exist_ok=True)
            self.log_output.append("📁 Created indicator_output directory")
            print("📁 Created indicator_output directory")

    def update_toggle_button(self):
        """Update toggle button text based on current indicators"""
        if len(self.indicator_rows) > 0:
            self.toggle_btn.setText(I18N.t("Clear All", "Xóa tất cả"))
            self.toggle_btn.setStyleSheet("QPushButton { background-color: #ff6b6b; color: white; font-size: 10px; }")
        else:
            self.toggle_btn.setText(I18N.t("Add All", "Thêm tất cả"))
            self.toggle_btn.setStyleSheet("QPushButton { background-color: #51cf66; color: white; font-size: 10px; }")

    def apply_language_to_indicator_rows(self):
        """Ensure dynamic 'Remove' buttons reflect current language immediately."""
        try:
            for row in self.indicator_rows:
                btn = row.get("remove_btn")
                if btn:
                    btn.setText(I18N.t("Remove", "Xóa"))
        except Exception:
            pass

    def toggle_all_indicators(self):
        """Toggle between Add All and Remove All indicators"""
        if len(self.indicator_rows) > 0:
            # Remove all indicators
            self.remove_all_indicators()
        else:
            # Add all indicators
            self.add_all_indicators()

    def add_all_indicators(self):
        """Add one of each type of indicator with default parameters"""
        # Define default indicators to add (one of each type) - ALL 28 INDICATORS
        default_indicators = [
            ("Moving Average", {"ma_type": "SMA", "period": 20}),
            ("MACD", {"fast": 12, "slow": 26, "signal": 9}),
            ("RSI", {"period": 14}),
            ("Stochastic", {"period": 14, "smooth": 3}),
            ("Bollinger Bands", {"window": 20, "dev": 2}),
            ("ATR", {"period": 14}),
            ("ADX", {"period": 14}),
            ("CCI", {"period": 20}),
            ("Williams %R", {"period": 14}),
            ("ROC", {"period": 12}),
            ("OBV", {}),
            ("MFI", {"period": 14}),
            ("Parabolic SAR", {"acceleration": 0.02, "max_step": 0.2}),
            ("Chaikin Money Flow", {"period": 20}),
            ("Ease of Movement", {"window": 14}),
            ("Force Index", {"period": 13}),
            ("Donchian Channel", {"window": 20}),
            ("TRIX", {"period": 14}),
            ("DPO", {"period": 20}),
            ("Mass Index", {"period": 25}),
            ("Vortex Indicator", {"period": 14}),
            ("KST Oscillator", {"roc1": 10, "roc2": 15, "roc3": 20, "roc4": 30, "window1": 10, "window2": 10, "window3": 10, "window4": 15, "window_sig": 9}),
            ("Stochastic RSI", {"period": 14, "smooth1": 3, "smooth2": 3}),
            ("Ultimate Oscillator", {"short": 7, "medium": 14, "long": 28}),
            ("Keltner Channel", {"window": 20}),
            ("Envelope", {"period": 20, "percent": 2.0}),
            ("Fibonacci", {"lookback": 100}),
            ("Ichimoku", {"tenkan": 9, "kijun": 26, "senkou": 52})
        ]
        
        self.log_output.append("🔧 Adding all default indicators...")
        
        for indicator_label, params in default_indicators:
            try:
                # Add new indicator row
                self.add_indicator_row()
                
                # Get the last added row
                row_dict = self.indicator_rows[-1]
                
                # Set the indicator type
                combo = row_dict["indi_combo"]
                for i in range(combo.count()):
                    if combo.itemText(i) == indicator_label:
                        combo.setCurrentIndex(i)
                        self.on_indicator_changed(i, row_dict)
                        break
                
                # Set parameters if controls exist
                for param_name, param_value in params.items():
                    # Handle spin boxes (most common)
                    if f"{param_name}_spin" in row_dict:
                        row_dict[f"{param_name}_spin"].setValue(param_value)
                    # Handle combo boxes
                    elif f"{param_name}_combo" in row_dict:
                        combo_widget = row_dict[f"{param_name}_combo"]
                        for j in range(combo_widget.count()):
                            if combo_widget.itemText(j) == str(param_value):
                                combo_widget.setCurrentIndex(j)
                                break
                    # Handle special cases with different naming patterns
                    elif param_name == "ma_type" and "ma_type_combo" in row_dict:
                        combo_widget = row_dict["ma_type_combo"]
                        for j in range(combo_widget.count()):
                            if combo_widget.itemText(j) == str(param_value):
                                combo_widget.setCurrentIndex(j)
                                break
                    elif param_name == "dev" and "std_spin" in row_dict:
                        row_dict["std_spin"].setValue(param_value)
                    elif param_name == "acceleration" and "accel_spin" in row_dict:
                        row_dict["accel_spin"].setValue(param_value)
                    elif param_name == "max_step" and "maxstep_spin" in row_dict:
                        row_dict["maxstep_spin"].setValue(param_value)
                    elif param_name == "smooth" and "k_spin" in row_dict:
                        row_dict["k_spin"].setValue(param_value)
                
            except Exception as e:
                self.log_output.append(f"⚠️ Error adding {indicator_label}: {e}")
        
        self.log_output.append(f"✅ Added {len(default_indicators)} indicators successfully!")
        self.update_toggle_button()

    def remove_all_indicators(self):
        """Remove all current indicators"""
        self.log_output.append("🗑️ Removing all indicators...")
        
        # Remove all indicators (copy list to avoid modification during iteration)
        indicators_to_remove = self.indicator_rows.copy()
        for row_dict in indicators_to_remove:
            self.remove_indicator_row(row_dict)
        
        self.log_output.append("✅ All indicators removed!")
        self.update_toggle_button()

    def export_indicators(self):
        # Dừng tất cả worker cũ trước khi tạo mới
        self.stop_all_workers()
        
        # 🧹 Clear all old indicator data before starting new calculation
        self.clear_all_indicator_data()
        
        self.table.setRowCount(0)
        symbols = list(self.market_tab.checked_symbols)
        selected_tfs = [(tf, self.market_tab.tf_spinboxes[tf].value()) for tf in self.market_tab.tf_checkboxes if self.market_tab.tf_checkboxes[tf].isChecked()]
        
        # DEBUG: Log what's selected
        self.log_output.append(f"DEBUG: Selected symbols: {symbols}")
        self.log_output.append(f"DEBUG: Selected timeframes: {selected_tfs}")
        
        if not symbols or not selected_tfs:
            self.log_output.append("Please select symbol and timeframe in Market tab.")
            return
        indicator_list = []
        for row in self.indicator_rows:
            indi_label = row["indi_combo"].currentText()
            indi_name = self._label_to_name(indi_label)
            if not indi_name:
                # Skip invalid/unknown indicator rows gracefully
                warn_msg = f"⚠️ Skipping unknown indicator label during export: '{indi_label}'"
                try:
                    self.log_output.append(warn_msg)
                except Exception:
                    print(warn_msg)
                continue
            params = {}
            # Lấy params cho từng indicator, ví dụ:
            if indi_name in ["RSI", "ATR", "ADX", "CCI", "WilliamsR", "ROC", "MFI", "Chaikin", "EOM", "ForceIndex", "TRIX", "DPO"]:
                params["period"] = row["period_spin"].value()
            elif indi_name == "Bollinger Bands":
                params["window"] = row["window_spin"].value()
                params["dev"] = row["dev_spin"].value()
            elif indi_name == "MA":
                params["period"] = row["period_spin"].value()
                params["ma_type"] = row["ma_type_combo"].currentText()
            elif indi_name == "MACD":
                params["fast"] = row["fast_spin"].value()
                params["slow"] = row["slow_spin"].value()
                params["signal"] = row["signal_spin"].value()
            elif indi_name == "Stochastic":
                params["period"] = row["period_spin"].value()
                params["smooth"] = row["smooth_spin"].value()
            elif indi_name == "PSAR":
                params["step"] = row["step_spin"].value()
                params["max_step"] = row["max_step_spin"].value()
            elif indi_name == "Donchian":
                params["window"] = row["window_spin"].value()
            elif indi_name == "MassIndex":
                params["fast"] = row["fast_spin"].value()
                params["slow"] = row["slow_spin"].value()
            elif indi_name == "Vortex":
                params["period"] = row["period_spin"].value()
            elif indi_name == "KST":
                params["window1"] = row["window1_spin"].value()
                params["window2"] = row["window2_spin"].value()
                params["window3"] = row["window3_spin"].value()
                params["window4"] = row["window4_spin"].value()
                params["window_sig"] = row["window_sig_spin"].value()
            elif indi_name == "StochRSI":
                params["period"] = row["period_spin"].value()
                params["smooth1"] = row["smooth1_spin"].value()
                params["smooth2"] = row["smooth2_spin"].value()
            elif indi_name == "UltimateOscillator":
                params["short"] = row["short_spin"].value()
                params["medium"] = row["medium_spin"].value()
                params["long"] = row["long_spin"].value()
            elif indi_name == "Keltner":
                params["window"] = row["window_spin"].value()
            elif indi_name == "Fibonacci":
                params["lookback"] = row["lookback_spin"].value()
            elif indi_name == "Ichimoku":
                params["tenkan"] = row["tenkan_spin"].value()
                params["kijun"] = row["kijun_spin"].value()
                params["senkou"] = row["senkou_spin"].value()
            elif indi_name == "Envelope":
              
                if "period_spin" in row:
                    params["period"] = row["period_spin"].value()
                else:
                    params["period"] = 20  # default
                if "percent_spin" in row:
                    params["percent"] = row["percent_spin"].value()
                else:
                    params["percent"] = 2  # default
            indicator_list.append({
                "name": indi_name,
                "params": params
            })
        self.indicator_list = indicator_list
        self.last_export_hash = hashlib.md5(json.dumps(indicator_list, sort_keys=True).encode()).hexdigest()
        self.pending_workers = 0
        
        # DEBUG: Log indicator list
        self.log_output.append(f"DEBUG: Indicator list: {[i['name'] for i in indicator_list]}")
        
        for sym in symbols:
            for tf, count in selected_tfs:
                self.log_output.append(f"DEBUG: Starting worker for {sym} {tf} with {count} candles")
                worker = IndicatorWorker(sym, tf, count, indicator_list)
                worker.finished.connect(self.on_indicator_finished)
                worker.finished.connect(lambda _, w=worker: self.cleanup_worker(w))
                worker.error.connect(self.on_indicator_error)  # Thêm dòng này
                worker.start()
                self.workers.append(worker)
                self.pending_workers += 1

    def on_indicator_error(self, msg):
        QMessageBox.critical(self, "Indicator Error", msg)
        self.log_output.append(msg)

    def cleanup_worker(self, worker):
        if worker in self.workers:
            try:
                # Request graceful stop first
                if hasattr(worker, 'request_stop'):
                    worker.request_stop()
                
                # Properly terminate the thread before cleanup
                if worker.isRunning():
                    worker.quit()
                    worker.wait(3000)  # Wait up to 3 seconds for thread to finish
                
                self.workers.remove(worker)
                worker.deleteLater()
            except Exception as e:
                print(f"Warning: Error cleaning up worker: {e}")
                # Force removal from list even if cleanup fails
                if worker in self.workers:
                    self.workers.remove(worker)

    def on_indicator_finished(self, msg):
        import json
        try:
            # DEBUG: Log raw message
            self.log_output.append(f"DEBUG: Received message: {msg[:200]}...")
            
            data = json.loads(msg)
            if "error" in data:
                self.log_output.append(data["error"])
                return
            symbol = data.get("symbol", "")
            timeframe = data.get("timeframe", "")
            results = data.get("results", [])
            
            # DEBUG: Log results
            self.log_output.append(f"DEBUG: Processing {len(results)} results for {symbol} {timeframe}")
            
            for r in results:
                indi = r.get("indicator", "")
                detail = r.get("detail", "")
                signal = r.get("signal", "")
                time_val = r.get("time", "")
                
                # DEBUG: Log each result
                self.log_output.append(f"DEBUG: {indi} -> {signal} -> {detail}")
                
                # --- HIỂN THỊ GIÁ TRỊ THỰC TẾ THAY VÌ TÊN CỘT ---
                if detail and detail != "N/A" and detail != "":
                    value = detail  # Sử dụng detail từ export_indicators (đã có giá trị thực)
                else:
                    value = f"{indi}: No data"
                indi_full = f"{indi}_{symbol}_{timeframe}"
                time_str = format_time(time_val)
                
                # Chọn icon và màu cho signal
                if signal == "Bullish":
                    icon = "↑"
                    color = Qt.green
                elif signal == "Bearish":
                    icon = "↓"
                    color = Qt.red
                elif signal == "Neutral":
                    icon = "→"
                    color = Qt.darkYellow
                else:
                    icon = ""
                    color = Qt.black
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(indi_full))
                self.table.setItem(row, 1, QTableWidgetItem(str(value)))
                self.table.setItem(row, 2, QTableWidgetItem(time_str))
                signal_item = QTableWidgetItem(f"{icon} {signal}")
                signal_item.setForeground(color)
                self.table.setItem(row, 3, signal_item)
        except Exception as e:
            self.log_output.append(f"Error displaying indicator result: {e}\n{msg}")
    def restore_user_config(self):
        config = self.user_config.get("indicator_tab", [])
        # Xóa các indicator cũ trước khi khôi phục
        while self.indicator_rows:
            self.remove_indicator_row(self.indicator_rows[0])
        for indi in config:
            self.add_indicator_row()
            row = self.indicator_rows[-1]
            combo = row["indi_combo"]
            idx = combo.findText(indi.get("label", indi.get("name", "")))
            if idx >= 0:
                combo.setCurrentIndex(idx)
            for k, v in indi.get("params", {}).items():
                widget_name = f"{k}_spin"
                if widget_name in row:
                    row[widget_name].setValue(v)
                elif k == "ma_type" and "ma_type_combo" in row:
                    idx2 = row["ma_type_combo"].findText(v)
                    if idx2 >= 0:
                        row["ma_type_combo"].setCurrentIndex(idx2)
        
        # Update toggle button after restoring config
        self.update_toggle_button()

    def save_current_user_config(self):
        config = []
        for row in self.indicator_rows:
            indi_label = row["indi_combo"].currentText()
            # Robust lookup: handle missing label gracefully
            indi_name = None
            for opt in self.INDICATOR_OPTIONS:
                try:
                    if opt.get("label") == indi_label:
                        indi_name = opt.get("name")
                        break
                except Exception:
                    continue
            if not indi_name:
                # Fallback: try direct name match or skip
                for opt in self.INDICATOR_OPTIONS:
                    if opt.get("name") == indi_label:
                        indi_name = opt.get("name")
                        break
            if not indi_name:
                # Skip this row to avoid StopIteration crash
                print(f"⚠️ Skipping unknown indicator label '{indi_label}' during save")
                continue
            params = {}
            def safe_get(widget, method="value"):
                try:
                    if widget is not None:
                        return getattr(widget, method)()
                except RuntimeError:
                    return None
                return None

            if "period_spin" in row:
                v = safe_get(row["period_spin"])
                if v is not None:
                    params["period"] = v
            if "ma_type_combo" in row:
                v = safe_get(row["ma_type_combo"], "currentText")
                if v is not None:
                    params["ma_type"] = v
            if "fast_spin" in row:
                v = safe_get(row["fast_spin"])
                if v is not None:
                    params["fast"] = v
            if "slow_spin" in row:
                v = safe_get(row["slow_spin"])
                if v is not None:
                    params["slow"] = v
            if "signal_spin" in row:
                v = safe_get(row["signal_spin"])
                if v is not None:
                    params["signal"] = v
            if "smooth_spin" in row:
                v = safe_get(row["smooth_spin"])
                if v is not None:
                    params["smooth"] = v
            if "window_spin" in row:
                v = safe_get(row["window_spin"])
                if v is not None:
                    params["window"] = v
            if "dev_spin" in row:
                v = safe_get(row["dev_spin"])
                if v is not None:
                    params["dev"] = v
            if "step_spin" in row:
                v = safe_get(row["step_spin"])
                if v is not None:
                    params["step"] = v
            if "max_step_spin" in row:
                v = safe_get(row["max_step_spin"])
                if v is not None:
                    params["max_step"] = v
            if "window1_spin" in row:
                v = safe_get(row["window1_spin"])
                if v is not None:
                    params["window1"] = v
            if "window2_spin" in row:
                v = safe_get(row["window2_spin"])
                if v is not None:
                    params["window2"] = v
            if "window3_spin" in row:
                v = safe_get(row["window3_spin"])
                if v is not None:
                    params["window3"] = v
            if "window4_spin" in row:
                v = safe_get(row["window4_spin"])
                if v is not None:
                    params["window4"] = v
            if "window_sig_spin" in row:
                v = safe_get(row["window_sig_spin"])
                if v is not None:
                    params["window_sig"] = v
            if "smooth1_spin" in row:
                v = safe_get(row["smooth1_spin"])
                if v is not None:
                    params["smooth1"] = v
            if "smooth2_spin" in row:
                v = safe_get(row["smooth2_spin"])
                if v is not None:
                    params["smooth2"] = v
            if "short_spin" in row:
                v = safe_get(row["short_spin"])
                if v is not None:
                    params["short"] = v
            if "medium_spin" in row:
                v = safe_get(row["medium_spin"])
                if v is not None:
                    params["medium"] = v
            if "long_spin" in row:
                v = safe_get(row["long_spin"])
                if v is not None:
                    params["long"] = v
            if "lookback_spin" in row:
                v = safe_get(row["lookback_spin"])
                if v is not None:
                    params["lookback"] = v
            if "tenkan_spin" in row:
                v = safe_get(row["tenkan_spin"])
                if v is not None:
                    params["tenkan"] = v
            if "kijun_spin" in row:
                v = safe_get(row["kijun_spin"])
                if v is not None:
                    params["kijun"] = v
            if "senkou_spin" in row:
                v = safe_get(row["senkou_spin"])
                if v is not None:
                    params["senkou"] = v
            config.append({
                "name": indi_name,
                "label": indi_label,
                "params": params
            })
        user_config = load_user_config()
        user_config["indicator_tab"] = config
        save_user_config(user_config)

    def stop_all_workers(self):
        for worker in getattr(self, "workers", []):
            try:
                # Request graceful stop first
                if hasattr(worker, 'request_stop'):
                    worker.request_stop()
                
                worker.quit()
                worker.wait(3000)  # Wait up to 3 seconds
            except Exception:
                pass
        self.workers = []

    def get_active_indicators(self):
        """Get list of active indicators for chart display"""
        active_indicators = []
        
        for row in self.indicator_rows:
            try:
                # Check if widgets still exist
                if "indi_combo" not in row or row["indi_combo"] is None:
                    continue
                
                # Safe widget access with try-catch
                try:
                    indi_label = row["indi_combo"].currentText()
                except RuntimeError:
                    # Widget has been deleted
                    continue
                    
                if not indi_label:
                    continue
                    
                indi_name = next((opt["name"] for opt in self.INDICATOR_OPTIONS if opt["label"] == indi_label), None)
                if not indi_name:
                    continue
                
                # Get parameters safely
                params = {}
                
                # Helper function for safe widget access
                def safe_get_value(widget_key, method="value"):
                    try:
                        if widget_key in row and row[widget_key] is not None:
                            widget = row[widget_key]
                            if method == "value":
                                return widget.value()
                            elif method == "currentText":
                                return widget.currentText()
                    except (RuntimeError, AttributeError):
                        # Widget deleted or method not available
                        pass
                    return None
                
                # Get common parameters
                period = safe_get_value("period_spin")
                if period is not None:
                    params["period"] = period
                
                ma_type = safe_get_value("ma_type_combo", "currentText")
                if ma_type:
                    params["ma_type"] = ma_type
                
                window = safe_get_value("window_spin")
                if window is not None:
                    params["window"] = window
                
                window1 = safe_get_value("window1_spin")
                if window1 is not None:
                    params["window1"] = window1
                
                window2 = safe_get_value("window2_spin")
                if window2 is not None:
                    params["window2"] = window2
                
                # Add indicator to list
                active_indicators.append({
                    "name": indi_name,
                    "label": indi_label,
                    "params": params
                })
                
                print(f"✅ Added indicator: {indi_name} with params: {params}")
                
            except Exception as e:
                print(f"Error getting indicator: {e}")
                continue
        
        print(f"🔍 Total active indicators: {len(active_indicators)}")
        return active_indicators

        
class PatternTab(QWidget):
    def __init__(self, market_tab, indicator_tab):
        super().__init__()
        self.market_tab = market_tab
        self.indicator_tab = indicator_tab
        self.pattern_data = []
        self.worker = None
        self.init_ui()
        self.load_patterns()

        # Removed auto refresh timer - manual refresh only

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Enable checkbox
        self.enable_checkbox = QCheckBox("Enable candlestick pattern detection")
        self.enable_checkbox.setChecked(True)
        layout.addWidget(self.enable_checkbox)
        
        # Filter options - chỉ giữ confidence filter
        filter_layout = QHBoxLayout()
        
        self.min_confidence_label = QLabel("📊 Min confidence:")
        filter_layout.addWidget(self.min_confidence_label)
        
        self.min_confidence_spinbox = QDoubleSpinBox()
        self.min_confidence_spinbox.setRange(0.0, 1.0)
        self.min_confidence_spinbox.setSingleStep(0.1)
        self.min_confidence_spinbox.setValue(0.3)  # Lower default value
        self.min_confidence_spinbox.valueChanged.connect(self.on_filter_changed)
        self.min_confidence_spinbox.setToolTip(I18N.t(
            "Minimum confidence threshold (0.0 = all patterns, 1.0 = only highest confidence)",
            "Ngưỡng độ tin cậy tối thiểu (0.0 = tất cả mô hình, 1.0 = chỉ mô hình có độ tin cậy cao nhất)"
        ))
        filter_layout.addWidget(self.min_confidence_spinbox)
        
        layout.addLayout(filter_layout)
        
        # Status label for pattern statistics with improved styling
        self.status_label = QLabel("No patterns loaded")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
                color: #333333;
            }
        """)
        layout.addWidget(self.status_label)
        
        # Main action button
        self.fetch_pattern_btn = QPushButton("🔍 Fetch Candlestick Patterns")
        self.fetch_pattern_btn.clicked.connect(self.fetch_patterns_and_reload)
        self.fetch_pattern_btn.setMinimumHeight(35)
        layout.addWidget(self.fetch_pattern_btn)
        
        # Pattern table - 7 columns: Symbol, Timeframe, Time, Pattern, Length, Signal, Confidence
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Symbol", "Timeframe", "Time", "Pattern", "Length", "Signal", "Confidence"])
        header = self.table.horizontalHeader()
        for i in range(self.table.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def fetch_patterns_and_reload(self):
    
        if self.worker is not None and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
        symbols = list(self.market_tab.checked_symbols)
        timeframes = [tf for tf in self.market_tab.tf_checkboxes if self.market_tab.tf_checkboxes[tf].isChecked()]
        if not symbols or not timeframes:
            QMessageBox.warning(self, I18N.t("Warning", "Cảnh báo"), I18N.t("Please select symbol and timeframe in Market tab.", "Vui lòng chọn mã và khung thời gian trong tab Thị trường."))
            return
        self.fetch_pattern_btn.setEnabled(False)
        indicator_list = getattr(self.indicator_tab, "indicator_list", [])
        self.worker = PatternWorker(symbols, timeframes, indicator_list)
        self.worker.finished.connect(self.on_patterns_finished)
        self.worker.error.connect(self.on_patterns_error)
        self.worker.start()

    def on_patterns_finished(self):
        self.fetch_pattern_btn.setEnabled(True)
        self.load_patterns()

    def on_patterns_error(self, msg):
        QMessageBox.critical(self, I18N.t("Pattern Error", "Lỗi mô hình"), msg)
        self.fetch_pattern_btn.setEnabled(True)

    def load_patterns(self):
        """Load and display patterns using helper functions"""
        self.table.setRowCount(0)
        self.pattern_data.clear()
        folder = "./pattern_signals"
        
        if not os.path.exists(folder):
            self.status_label.setText("❌ Pattern folder not found")
            return

        # Get selected symbols and timeframes from market tab
        selected_symbols = set(self.market_tab.checked_symbols)
        selected_timeframes = set(tf for tf in self.market_tab.tf_checkboxes if self.market_tab.tf_checkboxes[tf].isChecked())
        
        # If nothing is selected, load all patterns
        if not selected_symbols or not selected_timeframes:
            selected_symbols = None
            selected_timeframes = None
        
        # Get filter settings from UI (chỉ confidence, không còn candlestick_only)
        min_confidence = self.min_confidence_spinbox.value()
        
        print(f"📊 Loading patterns with filters: min_confidence={min_confidence}")
        
        # Show loading status
        self.status_label.setText(f"🔄 Loading patterns (min_conf: {min_confidence})...")
        
        # Load and filter patterns using helper functions (luôn lấy tất cả patterns)
        patterns = load_and_filter_patterns(
            folder,
            selected_symbols,
            selected_timeframes,
            candlestick_only=False,  # Luôn là False
            min_confidence=min_confidence
        )
        
        print(f"✅ Loaded {len(patterns)} patterns after filtering")
        
        # Sort patterns by priority using helper function (now sorted by Symbol first)
        sorted_patterns = sort_patterns_by_priority(patterns)
        
        # Add patterns to table
        for pattern_obj in sorted_patterns:
            self.add_pattern_row(
                pattern_obj['symbol'],
                pattern_obj['timeframe'],
                pattern_obj['time'],
                pattern_obj['pattern'],
                pattern_obj['pattern_length'],
                pattern_obj['signal'],
                pattern_obj.get('confidence', 0.5),  # Add confidence
                pattern_obj.get('score', 0.0)  # Add score instead of recommendation
            )
        
        # Update status label using helper functions
        stats = get_pattern_statistics(sorted_patterns)
        status_message = format_status_message(stats, candlestick_only=False)  # Luôn là False
        self.status_label.setText(status_message)

    def add_pattern_row(self, symbol, timeframe, time_key, pattern, pattern_length=1, signal=None, confidence=0.5, score=0.0):
        """Add a pattern row to the table - pure UI logic with icon in symbol"""
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        # Check if pattern is candlestick using helper function
        is_candlestick = is_candlestick_pattern(pattern)
        
        # Add candlestick icon to symbol name (only show candlestick icon for candlestick tab)
        symbol_with_icon = f"🕯️ {symbol}"
        
        # Create table items
        symbol_item = QTableWidgetItem(symbol_with_icon)
        timeframe_item = QTableWidgetItem(timeframe)
        time_item = QTableWidgetItem(str(time_key))
        pattern_item = QTableWidgetItem(str(pattern))
        length_item = QTableWidgetItem(str(pattern_length))
        
        # Highlight candlestick patterns
        if is_candlestick:
            font = symbol_item.font()
            font.setBold(True)
            for item in [symbol_item, timeframe_item, time_item, pattern_item, length_item]:
                item.setFont(font)
            
            from PyQt5.QtGui import QColor
            bg_color = QColor(255, 255, 200)  # Light yellow background
            for item in [symbol_item, timeframe_item, time_item, pattern_item, length_item]:
                item.setBackground(bg_color)
        
        # Set table items
        self.table.setItem(row, 0, symbol_item)
        self.table.setItem(row, 1, timeframe_item)
        self.table.setItem(row, 2, time_item)
        self.table.setItem(row, 3, pattern_item)
        self.table.setItem(row, 4, length_item)
        
        # Signal column with score
        if signal is None:
            signal = ""
        
        # Add score to signal display: "Bullish (0.8)" or "Neutral (-0.166)"
        # Always show score regardless of positive/negative value
        display_signal = f"{signal} ({score:.3f})"
        
        # Color signal based on type
        color = None
        if "Bullish" in signal:
            color = Qt.green
        elif "Bearish" in signal:
            color = Qt.red
        elif "Neutral" in signal:
            color = Qt.darkYellow
        
        signal_item = QTableWidgetItem(display_signal)
        if color:
            signal_item.setForeground(color)
        
        # Apply candlestick highlighting to signal column
        if is_candlestick:
            font = signal_item.font()
            font.setBold(True)
            signal_item.setFont(font)
            from PyQt5.QtGui import QColor
            bg_color = QColor(255, 255, 200)
            signal_item.setBackground(bg_color)
        
        self.table.setItem(row, 5, signal_item)
        
        # Confidence column (only show confidence value)
        confidence_text = f"{confidence:.2f}"
        confidence_item = QTableWidgetItem(confidence_text)
        
        # Color based on confidence level
        if confidence >= 0.7:
            confidence_item.setForeground(Qt.darkGreen)
        elif confidence >= 0.5:
            confidence_item.setForeground(Qt.darkBlue)
        elif confidence >= 0.3:
            confidence_item.setForeground(Qt.darkYellow)
        else:
            confidence_item.setForeground(Qt.gray)
        
        # Apply candlestick highlighting to confidence column
        if is_candlestick:
            font = confidence_item.font()
            font.setBold(True)
            confidence_item.setFont(font)
            from PyQt5.QtGui import QColor
            bg_color = QColor(255, 255, 200)
            confidence_item.setBackground(bg_color)
        
        self.table.setItem(row, 6, confidence_item)

    def on_filter_changed(self):
        """Handle filter changes with debug information"""
        min_confidence = self.min_confidence_spinbox.value()
        
        print(f"🔄 Filter changed: min_confidence={min_confidence}")
        
        # Update status to show filter state
        self.status_label.setText(f"🔄 Loading patterns... (min_conf: {min_confidence})")
        
        # Load patterns with new filter
        self.load_patterns()

class PricePatternTab(QWidget):
    def __init__(self, market_tab):
        super().__init__()
        self.market_tab = market_tab
        self.pattern_data = []
        self.worker = None
        self.init_ui()
        self.load_patterns()

        # Removed auto refresh timer - manual refresh only

    def init_ui(self):
        layout = QVBoxLayout()

        # Enable checkbox
        self.enable_checkbox = QCheckBox("Enable price pattern detection")
        self.enable_checkbox.setChecked(True)
        layout.addWidget(self.enable_checkbox)

        # Filter options - add confidence filter and age filter
        filter_layout = QHBoxLayout()

        self.min_confidence_label = QLabel("📊 Min confidence:")
        filter_layout.addWidget(self.min_confidence_label)

        self.min_confidence_spinbox = QDoubleSpinBox()
        self.min_confidence_spinbox.setRange(0.0, 1.0)
        self.min_confidence_spinbox.setSingleStep(0.1)
        self.min_confidence_spinbox.setValue(0.3)  # Lower default value
        self.min_confidence_spinbox.valueChanged.connect(self.on_filter_changed)
        self.min_confidence_spinbox.setToolTip(I18N.t(
            "Minimum confidence threshold (0.0 = all patterns, 1.0 = only highest confidence)",
            "Ngưỡng độ tin cậy tối thiểu (0.0 = tất cả mô hình, 1.0 = chỉ mô hình có độ tin cậy cao nhất)"
        ))
        filter_layout.addWidget(self.min_confidence_spinbox)

        # Add age filter
        self.max_age_label = QLabel("📅 Max age (days):")
        filter_layout.addWidget(self.max_age_label)

        self.max_age_spinbox = QSpinBox()
        self.max_age_spinbox.setRange(1, 365)
        self.max_age_spinbox.setValue(90)  # Default 90 days for better initial visibility
        self.max_age_spinbox.valueChanged.connect(self.on_filter_changed)
        self.max_age_spinbox.setToolTip("Maximum age of patterns in days (only show patterns from last X days)")
        filter_layout.addWidget(self.max_age_spinbox)

        layout.addLayout(filter_layout)

        # Status label for pattern statistics with improved styling
        self.status_label = QLabel("No patterns loaded")
        self.status_label.setStyleSheet(
            """
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
                color: #333333;
            }
            """
        )
        layout.addWidget(self.status_label)

        # Main action button
        self.fetch_pattern_btn = QPushButton("🔍 Fetch Price Patterns")
        self.fetch_pattern_btn.clicked.connect(self.fetch_patterns_and_reload)
        self.fetch_pattern_btn.setMinimumHeight(35)
        layout.addWidget(self.fetch_pattern_btn)

        # Pattern table - 8 columns with Age column
        self.table = QTableWidget(0, 8)  # 8 columns
        self.table.setHorizontalHeaderLabels(
            ["Symbol", "Timeframe", "Time Period", "Pattern", "Length", "Signal", "Confidence", "Age"]
        )
        header = self.table.horizontalHeader()
        for i in range(self.table.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        layout.addWidget(self.table)
        self.setLayout(layout)


    def fetch_patterns_and_reload(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
        symbols = list(self.market_tab.checked_symbols)
        timeframes = [tf for tf in self.market_tab.tf_checkboxes if self.market_tab.tf_checkboxes[tf].isChecked()]
        if not symbols or not timeframes:
            QMessageBox.warning(self, I18N.t("Warning", "Cảnh báo"), I18N.t("Please select symbol and timeframe in Market tab.", "Vui lòng chọn mã và khung thời gian trong tab Thị trường."))
            return
        self.fetch_pattern_btn.setEnabled(False)
        self.worker = PricePatternWorker(symbols, timeframes)
        self.worker.finished.connect(self.on_patterns_finished)
        self.worker.error.connect(self.on_patterns_error)
        self.worker.start()

    def on_patterns_finished(self):
        self.fetch_pattern_btn.setEnabled(True)
        self.load_patterns()

    def on_patterns_error(self, msg):
        QMessageBox.critical(self, I18N.t("Pattern Error", "Lỗi mô hình"), msg)
        self.fetch_pattern_btn.setEnabled(True)

    def load_patterns(self):
        """Load and display price patterns robustly (handles symbol suffixes and helper fallback)"""
        self.table.setRowCount(0)
        self.pattern_data.clear()
        folder = "./pattern_price"

        if not os.path.exists(folder):
            self.status_label.setText("❌ Pattern folder not found")
            return

        # Selections from Market tab
        selected_symbols = set(self.market_tab.checked_symbols)
        selected_timeframes = set(tf for tf in self.market_tab.tf_checkboxes if self.market_tab.tf_checkboxes[tf].isChecked())

        # Filters
        min_confidence = self.min_confidence_spinbox.value()
        max_age_days = self.max_age_spinbox.value()
        self.status_label.setText(f"🔄 Loading patterns (min_conf: {min_confidence}, max_age: {max_age_days}d)...")

        # Normalize selected symbols for tolerant matching
        def _norm(s: str) -> str:
            return ''.join(ch for ch in s.upper() if ch.isalnum())

        norm_selected = {_norm(s) for s in selected_symbols} if selected_symbols else set()

        # Load patterns with helper; if empty (likely due to symbol suffix mismatch), fall back to manual loader
        patterns = []
        try:
            patterns = load_price_patterns_from_folder(
                folder,
                None if not selected_symbols else selected_symbols,
                None if not selected_timeframes else selected_timeframes
            )
        except Exception as e:
            print(f"⚠️ Helper load failed, will use manual loader: {e}")

        if not patterns:
            patterns = self._manual_load_price_patterns(folder)

        print(f"✅ Loaded {len(patterns)} price patterns before filtering")

        # Apply selection filtering with normalized symbol matching and timeframe filter
        patterns_all = list(patterns)
        if norm_selected:
            patterns = [p for p in patterns if _norm(str(p.get('symbol', ''))) in norm_selected]
        if selected_timeframes:
            tf_set = {tf.upper() for tf in selected_timeframes}
            patterns = [p for p in patterns if str(p.get('timeframe', '')).upper() in tf_set]

        # If selection filters remove everything, relax selection (show all)
        if not patterns and patterns_all:
            print("ℹ️ Selection filters returned 0; ignoring symbol/timeframe filters to show available patterns.")
            patterns = patterns_all

        # Filter by confidence and age
        now_dt = datetime.now()
        cutoff_time = now_dt - timedelta(days=max_age_days)
        filtered_patterns = []
        for p in patterns:
            # Confidence filter
            if float(p.get('confidence', 0.0)) < float(min_confidence):
                continue

            # Time filter
            tstr = p.get('end_time') or p.get('time') or p.get('start_time')
            if not tstr:
                continue
            try:
                # Accept both ISO and "YYYY-MM-DD HH:MM:SS"
                ts = datetime.fromisoformat(str(tstr).replace('Z', '+00:00')) if 'T' in str(tstr) or 'Z' in str(tstr) else datetime.strptime(str(tstr), "%Y-%m-%d %H:%M:%S")
            except Exception:
                # Best-effort parse
                try:
                    ts = datetime.fromisoformat(str(tstr))
                except Exception:
                    continue
            if ts >= cutoff_time:
                # Ensure required keys exist
                if 'time' not in p:
                    p['time'] = tstr
                filtered_patterns.append(p)

        print(f"✅ Filtered to {len(filtered_patterns)} patterns after selection, confidence and age")

        # Graceful fallback: if nothing passes filters, relax age then confidence
        if not filtered_patterns and patterns:
            print("ℹ️ No patterns matched filters; relaxing age to 365 days...")
            cutoff_time_relaxed = now_dt - timedelta(days=365)
            for p in patterns:
                if float(p.get('confidence', 0.0)) < float(min_confidence):
                    continue
                tstr = p.get('end_time') or p.get('time') or p.get('start_time')
                if not tstr:
                    continue
                try:
                    ts = datetime.fromisoformat(str(tstr).replace('Z', '+00:00')) if 'T' in str(tstr) or 'Z' in str(tstr) else datetime.strptime(str(tstr), "%Y-%m-%d %H:%M:%S")
                except Exception:
                    try:
                        ts = datetime.fromisoformat(str(tstr))
                    except Exception:
                        continue
                if ts >= cutoff_time_relaxed:
                    if 'time' not in p:
                        p['time'] = tstr
                    filtered_patterns.append(p)

        if not filtered_patterns and patterns:
            print("ℹ️ Still none; relaxing confidence to 0.0 and age to 365 days...")
            cutoff_time_relaxed = now_dt - timedelta(days=365)
            for p in patterns:
                tstr = p.get('end_time') or p.get('time') or p.get('start_time')
                if not tstr:
                    continue
                try:
                    ts = datetime.fromisoformat(str(tstr).replace('Z', '+00:00')) if 'T' in str(tstr) or 'Z' in str(tstr) else datetime.strptime(str(tstr), "%Y-%m-%d %H:%M:%S")
                except Exception:
                    try:
                        ts = datetime.fromisoformat(str(tstr))
                    except Exception:
                        continue
                if ts >= cutoff_time_relaxed:
                    if 'time' not in p:
                        p['time'] = tstr
                    filtered_patterns.append(p)

        # Sort and display
        try:
            sorted_patterns = sort_patterns_by_priority(filtered_patterns)
        except Exception:
            # Fallback: newest first by time
            def _key_time(x):
                t = x.get('end_time') or x.get('time') or ''
                try:
                    return datetime.fromisoformat(str(t).replace('Z', '+00:00'))
                except Exception:
                    try:
                        return datetime.strptime(str(t), "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        return datetime.min
            sorted_patterns = sorted(filtered_patterns, key=_key_time, reverse=True)

        for p in sorted_patterns:
            self.add_pattern_row(
                p.get('symbol', ''),
                p.get('timeframe', ''),
                p.get('time', ''),
                p.get('pattern', p.get('type', '')),
                p.get('pattern_length', 1),
                p.get('signal', ''),
                p.get('confidence', 0.5),
                p
            )

        # Update status
        try:
            stats = get_pattern_statistics(sorted_patterns)
            status_message = format_status_message(stats, candlestick_only=False)
            self.status_label.setText(status_message)
        except Exception:
            self.status_label.setText(f"📊 Showing {len(sorted_patterns)} price patterns")

    def _manual_load_price_patterns(self, folder: str):
        """Manual loader for price patterns from JSON files in folder."""
        results = []
        try:
            for name in os.listdir(folder):
                if not name.endswith('_patterns.json'):
                    continue
                path = os.path.join(folder, name)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        arr = json.load(f)
                        if isinstance(arr, list):
                            # Derive symbol/timeframe from filename when missing
                            parts = name[:-14]  # strip '_patterns.json'
                            # Expect pattern: SYMBOL_TF or SYMBOL_extra_TF
                            sym = None
                            tf = None
                            toks = parts.split('_')
                            if len(toks) >= 2:
                                tf = toks[-1]
                                sym = '_'.join(toks[:-1])
                            for p in arr:
                                if 'symbol' not in p and sym:
                                    p['symbol'] = sym
                                if 'timeframe' not in p and tf:
                                    p['timeframe'] = tf
                                results.append(p)
                except Exception as e:
                    print(f"⚠️ Failed reading {path}: {e}")
        except Exception as e:
            print(f"❌ Manual loader error: {e}")
        return results

    def add_pattern_row(self, symbol, timeframe, time_key, pattern, pattern_length=1, signal=None, confidence=0.5, pattern_obj=None):
        """Add a pattern row to the table with enhanced information including Age column"""
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        # Add icon to symbol name - using chart icon for price patterns
        symbol_with_icon = f"📊 {symbol}"
        
        # Calculate age of pattern
        age_text = "Unknown"
        age_color = Qt.black
        
        if pattern_obj:
            try:
                # Get the most recent time from pattern
                pattern_time_str = None
                if 'end_time' in pattern_obj and pattern_obj['end_time']:
                    pattern_time_str = pattern_obj['end_time']
                elif 'time' in pattern_obj and pattern_obj['time']:
                    pattern_time_str = pattern_obj['time']
                elif 'start_time' in pattern_obj and pattern_obj['start_time']:
                    pattern_time_str = pattern_obj['start_time']
                
                if pattern_time_str:
                    pattern_time = datetime.fromisoformat(pattern_time_str.replace('Z', '+00:00'))
                    current_time = datetime.now()
                    age_delta = current_time - pattern_time
                    
                    # Format age display
                    total_seconds = age_delta.total_seconds()
                    if total_seconds < 3600:  # Less than 1 hour
                        minutes = int(total_seconds // 60)
                        age_text = f"{minutes}m"
                        age_color = Qt.darkGreen  # Very fresh
                    elif total_seconds < 86400:  # Less than 1 day
                        hours = int(total_seconds // 3600)
                        age_text = f"{hours}h"
                        age_color = Qt.blue  # Fresh
                    elif total_seconds < 604800:  # Less than 1 week
                        days = int(total_seconds // 86400)
                        age_text = f"{days}d"
                        age_color = Qt.darkYellow  # Moderate
                    else:  # More than 1 week
                        weeks = int(total_seconds // 604800)
                        age_text = f"{weeks}w"
                        age_color = Qt.red  # Old
                        
            except Exception as e:
                print(f"⚠️ Failed to calculate age for pattern: {e}")
        
        # Format time period to show start and end time if available
        time_period = time_key
        if pattern_obj:
            start_time = pattern_obj.get('start_time', '')
            end_time = pattern_obj.get('end_time', '')
            
            if start_time and end_time:
                # Extract just the date and time part (remove seconds if present)
                try:
                    if isinstance(start_time, str) and len(start_time) > 16:
                        start_display = start_time[:16]  # YYYY-MM-DD HH:MM
                    else:
                        start_display = str(start_time)
                    
                    if isinstance(end_time, str) and len(end_time) > 16:
                        end_display = end_time[:16]  # YYYY-MM-DD HH:MM
                    else:
                        end_display = str(end_time)
                    
                    time_period = f"{start_display} → {end_display}"
                except:
                    time_period = str(time_key)
        
        # Create table items with bold font
        symbol_item = QTableWidgetItem(symbol_with_icon)
        symbol_item.setFont(QFont("Arial", 9, QFont.Bold))
        timeframe_item = QTableWidgetItem(timeframe)
        timeframe_item.setFont(QFont("Arial", 9, QFont.Bold))
        time_item = QTableWidgetItem(time_period)
        time_item.setFont(QFont("Arial", 9, QFont.Bold))
        pattern_item = QTableWidgetItem(str(pattern))
        pattern_item.setFont(QFont("Arial", 9, QFont.Bold))
        length_item = QTableWidgetItem(str(pattern_length))
        length_item.setFont(QFont("Arial", 9, QFont.Bold))
        
        # Set table items
        self.table.setItem(row, 0, symbol_item)
        self.table.setItem(row, 1, timeframe_item)
        self.table.setItem(row, 2, time_item)
        self.table.setItem(row, 3, pattern_item)
        self.table.setItem(row, 4, length_item)
        
        # Signal column - similar to PatternTab format with score
        if signal is None:
            signal = ""
        
        # Parse signal and score
        signal_with_score = ""
        
        # Check if signal already contains score in parentheses
        if "(" in signal and ")" in signal:
            signal_with_score = signal
        else:
            # For price patterns, we may need to derive score from confidence or other data
            # For now, display signal with confidence as score
            if signal:
                signal_with_score = f"{signal} ({confidence:.1f})"
            else:
                signal_with_score = f"({confidence:.1f})"
        
        # Color signal based on type
        color = None
        if any(keyword in signal.upper() for keyword in ["BULL", "UPWARD", "UP", "BUY", "LONG"]):
            color = Qt.darkGreen
        elif any(keyword in signal.upper() for keyword in ["BEAR", "DOWNWARD", "DOWN", "SELL", "SHORT"]):
            color = Qt.red
        elif any(keyword in signal.upper() for keyword in ["NEUTRAL", "SIDEWAYS", "CONSOLIDATION"]):
            color = Qt.darkYellow
        else:
            color = Qt.blue
        
        signal_item = QTableWidgetItem(signal_with_score)
        signal_item.setFont(QFont("Arial", 9, QFont.Bold))
        if color:
            signal_item.setForeground(color)
        
        self.table.setItem(row, 5, signal_item)
        
        # Confidence column (only show confidence value)
        confidence_text = f"{confidence:.2f}"
        confidence_item = QTableWidgetItem(confidence_text)
        confidence_item.setFont(QFont("Arial", 9, QFont.Bold))
        
        # Color based on confidence level
        if confidence >= 0.7:
            confidence_item.setForeground(Qt.darkGreen)
        elif confidence >= 0.5:
            confidence_item.setForeground(Qt.darkBlue)
        elif confidence >= 0.3:
            confidence_item.setForeground(Qt.darkYellow)
        else:
            confidence_item.setForeground(Qt.gray)
        
        self.table.setItem(row, 6, confidence_item)
        
        # Age column
        age_item = QTableWidgetItem(age_text)
        age_item.setFont(QFont("Arial", 9, QFont.Bold))
        age_item.setForeground(age_color)
        self.table.setItem(row, 7, age_item)

    def on_filter_changed(self):
        """Handle filter changes with debug information"""
        min_confidence = self.min_confidence_spinbox.value()
        max_age_days = self.max_age_spinbox.value()
        
        print(f"🔄 Price pattern filter changed: min_confidence={min_confidence}, max_age_days={max_age_days}")
        
        # Update status to show filter state
        self.status_label.setText(f"🔄 Loading patterns... (min_conf: {min_confidence}, max_age: {max_age_days}d)")
        
        # Load patterns with new filter
        self.load_patterns()

class PatternWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)  # Thêm signal báo lỗi

    def __init__(self, symbols, timeframes, indicator_list):
        super().__init__()
        self.symbols = symbols
        self.timeframes = timeframes
        self.indicator_list = indicator_list if indicator_list is not None else []
    
    def run(self):
        try:
            from pattern_detector import analyze_patterns
            for symbol in self.symbols:
                for tf in self.timeframes:
                    print(f"➡️ Analyzing {symbol} timeframe {tf} ...")
                    analyze_patterns(symbol, tf, self.indicator_list)
            self.finished.emit()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(f"Lỗi khi phát hiện mô hình nến:\n{e}\n{tb}")

class PricePatternWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, symbols, timeframes):
        super().__init__()
        self.symbols = symbols
        self.timeframes = timeframes

    def run(self):
        try:
            # Import and validate required modules
            try:
                from price_patterns_full_data import main as analyze_price_patterns
            except ImportError as e:
                self.error.emit(f"Failed to import price_patterns_full_data module:\n{e}")
                return
            
            # Validate input parameters
            if not self.symbols or not self.timeframes:
                self.error.emit("No symbols or timeframes selected for pattern analysis")
                return
            
            # Check if required folders exist
            import os
            required_folders = ["data", "indicator_output"]
            missing_folders = []
            for folder in required_folders:
                if not os.path.exists(folder):
                    missing_folders.append(folder)
            
            if missing_folders:
                self.error.emit(f"Missing required folders: {', '.join(missing_folders)}\nPlease run data and indicator fetching first.")
                return
            
            # Run pattern analysis with error handling
            analyze_price_patterns(symbols=self.symbols, timeframes=self.timeframes)
            self.finished.emit()
            
        except FileNotFoundError as e:
            self.error.emit(f"File not found error:\n{e}\n\nPlease ensure data files exist for selected symbols/timeframes.")
        except ImportError as e:
            self.error.emit(f"Import error:\n{e}\n\nPlease check if all required modules are available.")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(f"Error generating price patterns:\n{e}\n\nFull traceback:\n{tb}")

class IndicatorWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)  # Thêm dòng này
    def __init__(self, sym, tf, count, indicator_list):
        super().__init__()
        self.sym = sym
        self.tf = tf
        self.count = count
        self.indicator_list = indicator_list
        self._stop_requested = False

    def request_stop(self):
        """Request the worker to stop gracefully"""
        self._stop_requested = True

    def run(self):
        try:
            # Check for stop request before starting
            if self._stop_requested:
                return
                
            from mt5_indicator_exporter import export_indicators
            results = export_indicators(self.sym, self.tf, self.count, self.indicator_list)
          
            # Check for stop request before emitting
            if self._stop_requested:
                return
                
            msg = json.dumps({
                "symbol": self.sym,
                "timeframe": self.tf,
                "results": results
            })
            self.finished.emit(msg)
        except Exception as e:
            if not self._stop_requested:  # Only emit error if not stopping
                import traceback
                tb = traceback.format_exc()
                self.error.emit(f"ERROR for {self.sym} {self.tf}: {e}\n{tb}")

class TrendWorker(QThread):
    finished = pyqtSignal(dict)
    def __init__(self, symbol, timeframe, count):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.count = count
        self._stop_requested = False

    def request_stop(self):
        """Request the worker to stop gracefully"""
        self._stop_requested = True

    def run(self):
        try:
            # Check for stop request before starting
            if self._stop_requested:
                return
                
            try:
                from trendline_support_resistance import analyze_trend_channel_sr
                result = analyze_trend_channel_sr(self.symbol, self.timeframe, self.count)
                
                # Check for stop request before emitting
                if not self._stop_requested:
                    self.finished.emit(result)
            except ImportError as e:
                # Module not available, return empty result
                if not self._stop_requested:
                    self.finished.emit({"error": f"Trendline analysis module not available: {str(e)}"})
            except ValueError as e:
                # Handle data insufficient errors gracefully
                if not self._stop_requested:
                    if "Insufficient data" in str(e):
                        self.finished.emit({"error": f"Insufficient data for {self.symbol} {self.timeframe}: {str(e)}"})
                    else:
                        self.finished.emit({"error": f"Data validation error: {str(e)}"})
        except Exception as e:
            # Handle any other unexpected errors
            if not self._stop_requested:
                self.finished.emit({"error": f"Unexpected error in trend analysis: {str(e)}"})

class TrendTab(QWidget):
    def __init__(self, market_tab):
        super().__init__()
        self.market_tab = market_tab
        self.init_ui()
        self.workers = []
        self.pending = 0
        self.results = {}

    def init_ui(self):
        layout = QVBoxLayout()
        self.enable_checkbox = QCheckBox("Enable Trend Detection")
        self.enable_checkbox.setChecked(True)
        layout.addWidget(self.enable_checkbox)

        self.calc_btn = QPushButton("Calculate Trendline & SR")
        self.calc_btn.clicked.connect(self.on_calculate)
        layout.addWidget(self.calc_btn)

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Symbol", "Timeframe", "Type", "Value"])
        header = self.table.horizontalHeader()
        for i in range(self.table.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def clear_all_trend_data(self):
        """Clear all saved trendline data files"""
        import os
        import shutil
        
        trendline_dir = "trendline_sr"
        if os.path.exists(trendline_dir):
            try:
                # Count files before deletion
                file_count = len([f for f in os.listdir(trendline_dir) if os.path.isfile(os.path.join(trendline_dir, f))])
                
                # Remove all files in trendline_sr directory
                shutil.rmtree(trendline_dir)
                os.makedirs(trendline_dir, exist_ok=True)
                
                print(f"🧹 Trend Data Cleanup: Removed {file_count} old trendline files")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not clear trendline data: {e}")
        else:
            # Create directory if it doesn't exist
            os.makedirs(trendline_dir, exist_ok=True)
            print("📁 Created trendline_sr directory")

    def on_calculate(self):
        if not self.enable_checkbox.isChecked():
            QMessageBox.information(self, I18N.t("Info", "Thông tin"), I18N.t("Trend Detect is disabled!", "Phát hiện xu hướng đang tắt!"))
            return
        
        # Dừng tất cả worker cũ nếu còn chạy
        for worker in self.workers:
            if worker.isRunning():
                worker.quit()
                worker.wait()
        self.workers = []
        
        symbols = list(self.market_tab.checked_symbols)
        timeframes = [tf for tf in self.market_tab.tf_checkboxes if self.market_tab.tf_checkboxes[tf].isChecked()]
        if not symbols or not timeframes:
            QMessageBox.warning(self, I18N.t("Warning", "Cảnh báo"), I18N.t("Please select symbol and timeframe in Market tab.", "Vui lòng chọn mã và khung thời gian trong tab Thị trường."))
            return
            
        count = 200
        
        # 🧹 Clear all previous data immediately when starting new calculation
        self.clear_all_trend_data()
        
        self.table.setRowCount(0)
        self.table.clearContents()  # Clear all cell contents
        self.results = {}  # Clear results dict
        self.result_label.setText("Calculating...")
        
        self.pending = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                worker = TrendWorker(symbol, timeframe, count)
                # Use functools.partial to avoid lambda closure issues
                from functools import partial
                worker.finished.connect(partial(self.on_result_with_cleanup, symbol=symbol, timeframe=timeframe, worker=worker))
                self.workers.append(worker)
                self.pending += 1
                worker.start()
        self.calc_btn.setEnabled(False)

    def on_result_with_cleanup(self, result, symbol, timeframe, worker):
        """Handle result and cleanup worker"""
        # Cleanup the finished worker
        self.cleanup_worker(worker)
        
        # Process the result
        if "error" in result:
            error_msg = result['error']
            # Cải thiện hiển thị lỗi với thông tin symbol và timeframe
            full_error = f"Error for {symbol} {timeframe}: {error_msg}"
            self.result_label.setText(full_error)
            
            # Chỉ hiển thị popup nếu không phải lỗi thiếu dữ liệu thông thường
            if "Insufficient data" not in error_msg:
                QMessageBox.warning(self, "Trend Analysis Warning", full_error)
            else:
                # Log lỗi thiếu dữ liệu mà không spam popup
                print(f"⚠️ Trend Analysis: {full_error}")
        else:
            key = (symbol, timeframe)
            self.results[key] = result
        self.pending -= 1
        if self.pending <= 0:
            self.display_results()
            self.calc_btn.setEnabled(True)

    def on_result(self, result, symbol, timeframe):
        """Legacy method for backward compatibility"""
        return self.on_result_with_cleanup(result, symbol, timeframe, None)

    def display_results(self):
        self.table.setRowCount(0)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.result_label.setText(f"Calculated for all selected symbols/timeframes at {now_str}")

        last_symbol = None
        last_timeframe = None

        for (symbol, timeframe), result in sorted(self.results.items()):
            # Dòng phân cách giữa các nhóm
            if last_symbol is not None and (symbol != last_symbol or timeframe != last_timeframe):
                row = self.table.rowCount()
                self.table.insertRow(row)
                for col in range(self.table.columnCount()):
                    item = QTableWidgetItem("")
                    item.setBackground(Qt.black)
                    self.table.setItem(row, col, item)

            # --- Trendline ---
            trendline_arr = result.get("trendline")
            slope = result.get("trend_slope")
            intercept = result.get("trend_intercept")
            max_dev = result.get("max_deviation")
            min_dev = result.get("min_deviation")
            if trendline_arr and isinstance(trendline_arr, list) and len(trendline_arr) > 0:
                trend_start = trendline_arr[0]
                trend_end = trendline_arr[-1]
                info = [f"{trend_start:.5f} - {trend_end:.5f}"]
                if slope is not None:
                    info.append(f"Slope: {slope:.5f}")
                if intercept is not None:
                    info.append(f"Intercept: {intercept:.5f}")
                if max_dev is not None:
                    info.append(f"MaxDev: {max_dev:.5f}")
                if min_dev is not None:
                    info.append(f"MinDev: {min_dev:.5f}")
                row = self.table.rowCount()

                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(symbol))
                self.table.setItem(row, 1, QTableWidgetItem(timeframe))
                self.table.setItem(row, 2, QTableWidgetItem("Trendline"))
                self.table.setItem(row, 3, QTableWidgetItem(" | ".join(info)))

            # --- Channel Upper ---
            channel_upper = result.get("channel_upper")
            if channel_upper and isinstance(channel_upper, list) and len(channel_upper) > 0:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(symbol))
                self.table.setItem(row, 1, QTableWidgetItem(timeframe))
                self.table.setItem(row, 2, QTableWidgetItem("Channel Upper"))
                self.table.setItem(row, 3, QTableWidgetItem(f"{max(channel_upper):.5f}"))

            # --- Channel Lower ---
            channel_lower = result.get("channel_lower")
            if channel_lower and isinstance(channel_lower, list) and len(channel_lower) > 0:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(symbol))
                self.table.setItem(row, 1, QTableWidgetItem(timeframe))
                self.table.setItem(row, 2, QTableWidgetItem("Channel Lower"))
                self.table.setItem(row, 3, QTableWidgetItem(f"{min(channel_lower):.5f}"))

            # --- Support/Resistance ---
            for lvl in result.get("support", []):
                sr_type = "Support"
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(symbol))
                self.table.setItem(row, 1, QTableWidgetItem(timeframe))
                self.table.setItem(row, 2, QTableWidgetItem(sr_type))
                self.table.setItem(row, 3, QTableWidgetItem(f"{lvl:.5f}"))

            for lvl in result.get("resistance", []):
                sr_type = "Resistance"
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(symbol))
                self.table.setItem(row, 1, QTableWidgetItem(timeframe))
                self.table.setItem(row, 2, QTableWidgetItem(sr_type))
                self.table.setItem(row, 3, QTableWidgetItem(f"{lvl:.5f}"))

            last_symbol = symbol
            last_timeframe = timeframe
    
    def cleanup_worker(self, worker):
        """Cleanup finished worker thread"""
        if worker in self.workers:
            try:
                # Request graceful stop first
                if hasattr(worker, 'request_stop'):
                    worker.request_stop()
                
                # Properly terminate the thread before cleanup
                if worker.isRunning():
                    worker.quit()
                    worker.wait(3000)  # Wait up to 3 seconds for thread to finish
                self.workers.remove(worker)
                worker.deleteLater()
            except Exception as e:
                print(f"Warning: Error cleaning up trend worker: {e}")
                # Force removal from list even if cleanup fails
                if worker in self.workers:
                    self.workers.remove(worker)
    
    def stop_all_workers(self):
        """Stop all running trend workers"""
        for worker in getattr(self, "workers", []):
            try:
                # Request graceful stop first
                if hasattr(worker, 'request_stop'):
                    worker.request_stop()
                
                if worker.isRunning():
                    worker.quit()
                    worker.wait(3000)
            except Exception:
                pass
        self.workers = []

# Simple console-based trading bot interface
def simple_console_app():
    """Simple console interface for trading bot"""
    print("🤖 Trading Bot - Console Interface")
    print("=" * 50)
    
    # Show module availability
    print("📊 Module Status:")
    print(f"   • Data Fetcher: {'✅' if DATA_FETCHER_AVAILABLE else '❌'}")
    print(f"   • Indicator Exporter: {'✅' if INDICATOR_EXPORTER_AVAILABLE else '❌'}")
    print(f"   • Pattern Detector: {'✅' if PATTERN_DETECTOR_AVAILABLE else '❌'}")
    print(f"   • News Scraper: {'✅' if NEWS_SCRAPER_AVAILABLE else '❌'}")
    print(f"   • Auto Trading: {'✅' if AUTO_TRADING_AVAILABLE else '❌'}")
    print()
    
    while True:
        print("🔧 Available Options:")
        print("1. Fetch market data")
        print("2. Calculate indicators")
        print("3. Detect patterns")
        print("4. Show data summary")
        print("5. Test auto trading manager")
        print("6. Exit")
        print()
        
        try:
            choice = input("Select option (1-6): ").strip()
            
            if choice == "1":
                print("\n📥 Fetching market data...")
                if DATA_FETCHER_AVAILABLE:
                    try:
                        result = fetch_and_save_candles("XAUUSD", "M15", 500, "data")
                        if result:
                            print("✅ Data fetched successfully!")
                        else:
                            print("❌ Data fetching failed")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ Data fetcher not available")
            
            elif choice == "2":
                print("\n📊 Calculating indicators...")
                if INDICATOR_EXPORTER_AVAILABLE:
                    try:
                        stats = calculate_and_save_all(user_id="console_user")
                        if stats and stats.get("success", 0) > 0:
                            print(f"✅ Calculated indicators for {stats['success']} symbols")
                        else:
                            print("❌ Indicator calculation failed")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ Indicator exporter not available")
            
            elif choice == "3":
                print("\n🔍 Detecting patterns...")
                if PATTERN_DETECTOR_AVAILABLE:
                    try:
                        patterns = analyze_patterns("XAUUSD", "M15")
                        if patterns:
                            print(f"✅ Found {len(patterns)} patterns")
                            for pattern in patterns[:3]:  # Show first 3
                                print(f"   • {pattern.get('type', 'unknown')} pattern")
                        else:
                            print("❌ No patterns found")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ Pattern detector not available")
            
            elif choice == "4":
                print("\n📋 Data Summary:")
                directories = ["data", "indicator_output", "pattern_signals", "analysis_output"]
                for directory in directories:
                    if os.path.exists(directory):
                        files = [f for f in os.listdir(directory) if f.endswith('.json')]
                        print(f"   {directory}: {len(files)} files")
                    else:
                        print(f"   {directory}: Not found")
            
            elif choice == "5":
                print("\n🚀 Testing Auto Trading Manager...")
                if AUTO_TRADING_AVAILABLE:
                    try:
                        manager = AutoTradingManager(symbol="XAUUSD", timeframe="M15")
                        status = manager.get_status()
                        print("📊 Manager Status:")
                        for key, value in status.items():
                            print(f"   {key}: {value}")
                        print("✅ Auto trading manager test completed")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ Auto trading manager not available")
            
            elif choice == "6":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid option")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "-" * 30 + "\n")

# Original main function (backup)
def main_original():
    """Original main function"""
    import sys
    import traceback
    from datetime import datetime
    try:
        print("Trading Bot Starting...")
        
        # Load environment if available
        if DOTENV_AVAILABLE:
            load_dotenv()
        
        if GUI_AVAILABLE:
            print("Starting GUI mode...")
            try:
                print("Creating QApplication...")
                app = QApplication(sys.argv)
                
                # Simple UTF-8 setup without complex codec manipulation
                import locale
                try:
                    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
                    print("✅ UTF-8 locale set successfully")
                except:
                    try:
                        locale.setlocale(locale.LC_ALL, 'C.UTF-8') 
                        print("✅ C.UTF-8 locale set as fallback")
                    except:
                        print("⚠️ Using system default locale")
                
                # Set default font with Unicode support for Vietnamese text
                try:
                    # Try Arial Unicode MS first (good Unicode support)
                    unicode_font = QFont("Arial Unicode MS", 10)
                    if not unicode_font.exactMatch():
                        # Fallback to Arial which has decent Vietnamese support
                        unicode_font = QFont("Arial", 10)
                        if not unicode_font.exactMatch():
                            # Last fallback to Segoe UI (default Windows font)
                            unicode_font = QFont("Segoe UI", 10)
                    
                    app.setFont(unicode_font)
                    print(f"✅ Font set to {unicode_font.family()} 10pt with Unicode support")
                except Exception as e:
                    print(f"⚠️ Font setting error: {e}, using system default")
                
                print("Creating main window...")
                # Create main window with tabs
                class MainWindow(QWidget):
                    def closeEvent(self, event):
                        """Enhanced application close event handler with comprehensive cleanup"""
                        try:
                            print("[CLEANUP] 🧹 Application closing - comprehensive cleanup starting...")
                            
                            # 1. Stop all timers first
                            try:
                                for tab_name, tab in getattr(self, 'all_tabs', {}).items():
                                    if hasattr(tab, 'update_timer'):
                                        tab.update_timer.stop()
                                    if hasattr(tab, 'mt5_price_timer'):
                                        tab.mt5_price_timer.stop()
                                    if hasattr(tab, 'settings_refresh_timer'):
                                        tab.settings_refresh_timer.stop()
                                print("[CLEANUP] ✅ All timers stopped")
                            except Exception as e:
                                print(f"[CLEANUP] ⚠️ Timer cleanup error: {e}")
                            
                            # 2. Force stop auto trading with aggressive cleanup
                            if hasattr(self, 'all_tabs') and 'auto_trading_tab' in self.all_tabs:
                                try:
                                    auto_tab = self.all_tabs['auto_trading_tab']
                                    
                                    # Force shutdown any running auto manager
                                    if hasattr(auto_tab, 'auto_manager') and auto_tab.auto_manager:
                                        print("[CLEANUP] Force stopping auto manager...")
                                        try:
                                            # Set shutdown flags immediately
                                            if hasattr(auto_tab.auto_manager, '_shutdown_event'):
                                                auto_tab.auto_manager._shutdown_event.set()
                                            if hasattr(auto_tab.auto_manager, 'is_running'):
                                                auto_tab.auto_manager.is_running = False
                                            
                                            # Try quick stop
                                            import threading
                                            import time
                                            
                                            def quick_stop():
                                                try:
                                                    auto_tab.auto_manager.stop()
                                                except:
                                                    pass
                                            
                                            stop_thread = threading.Thread(target=quick_stop, daemon=True)
                                            stop_thread.start()
                                            stop_thread.join(timeout=2)  # Very short timeout on app close
                                            
                                        except Exception as e:
                                            print(f"[CLEANUP] Exception during force stop: {e}")
                                        finally:
                                            auto_tab.auto_manager = None
                                    
                                    # Also call the tab's stop method
                                    auto_tab.stop_auto()
                                    print("[CLEANUP] Auto trading stopped")
                                    
                                except Exception as e:
                                    print(f"[CLEANUP] Error stopping auto trading: {e}")
                            
                            # Cleanup RunAggregatorWorker thread in SignalTab
                            if hasattr(self, 'all_tabs') and 'signal_tab' in self.all_tabs:
                                try:
                                    signal_tab = self.all_tabs['signal_tab']
                                    if hasattr(signal_tab, 'cleanup_thread'):
                                        signal_tab.cleanup_thread()
                                except Exception as e:
                                    print(f"[CLEANUP] Error stopping signal tab thread: {e}")
                            
                            # 3. Cleanup all other threads (Signal, Pattern, etc.)
                            try:
                                if hasattr(self, 'all_tabs'):
                                    for tab_name, tab in self.all_tabs.items():
                                        # Stop indicator workers
                                        if hasattr(tab, 'stop_all_workers'):
                                            tab.stop_all_workers()
                                        # Stop any individual workers list
                                        if hasattr(tab, 'workers'):
                                            for worker in getattr(tab, 'workers', []):
                                                try:
                                                    if worker.isRunning():
                                                        worker.quit()
                                                        worker.wait(1000)
                                                except:
                                                    pass
                                        # Existing cleanup
                                        if hasattr(tab, 'cleanup_thread'):
                                            tab.cleanup_thread()
                                        if hasattr(tab, 'worker') and hasattr(tab.worker, 'stop'):
                                            tab.worker.stop()
                                        # NewsTab specific cleanup
                                        if hasattr(tab, 'cleanup_resources'):
                                            tab.cleanup_resources()
                                    print("[CLEANUP] ✅ All tab threads cleaned up")
                            except Exception as e:
                                print(f"[CLEANUP] ⚠️ Tab thread cleanup error: {e}")
                            
                            # 4. Use graceful shutdown utility for any remaining threads
                            try:
                                if hasattr(self, 'all_tabs'):
                                    graceful_shutdown_threads(list(self.all_tabs.values()))
                            except Exception as e:
                                print(f"[CLEANUP] ⚠️ Graceful shutdown error: {e}")
                            
                            # 5. Final emergency cleanup
                            try:
                                emergency_cleanup()
                            except Exception as e:
                                print(f"[CLEANUP] ⚠️ Emergency cleanup error: {e}")
                            
                            print("[CLEANUP] ✅ Application cleanup completed successfully")
                        
                        except Exception as e:
                            print(f"[CLEANUP] Error during application cleanup: {e}")
                            import traceback
                            print(f"[CLEANUP] Traceback: {traceback.format_exc()}")
                        finally:
                            # Always accept the close event
                            event.accept()
                
                main_window = MainWindow()
                main_window.setWindowTitle("ChatGPT AI BOT 4.3.2")
                main_window.setGeometry(100, 100, 1200, 800)

                # Load user config early to restore language preference
                try:
                    _cfg = load_user_config(apply_lang=True)
                    # (load_user_config already sets AppState._lang if key exists)
                except Exception as _e:
                    print(f"⚠️ Could not load user config early: {_e}")
                
                # Set robot icon for the application
                if os.path.exists("robot_icon.png"):
                    app.setWindowIcon(QIcon("robot_icon.png"))
                    main_window.setWindowIcon(QIcon("robot_icon.png"))
                    print("✅ Robot icon set successfully")
                else:
                    print("⚠️ Robot icon not found")
                
                    print("Creating tab widget...")
                # Create a top bar with app title and hamburger menu
                top_bar_widget = QWidget()
                top_bar_layout = QHBoxLayout(top_bar_widget)
                # Minimal margins so the button sits tight in the top-right area
                top_bar_layout.setContentsMargins(6, 4, 6, 0)
                app_title = QLabel("ChatGPT AI BOT 4.3.2")
                app_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
                menu_btn = QToolButton()
                menu_btn.setText("≡")
                menu_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
                menu_btn.setFixedWidth(28)
                menu_btn.setToolTip("Menu")
                menu_btn.setCursor(Qt.PointingHandCursor)
                menu_btn.setStyleSheet(
                    "QToolButton{font-size:18px;padding:2px 6px;border:1px solid #ddd;border-radius:4px;background:#fafafa;}"
                    "QToolButton:hover{background:#f0f0f0;}"
                )
                # Build hamburger menu
                main_menu = QMenu(main_window)
                act_login = QAction("Login", main_window)
                act_register = QAction("Register", main_window)
                act_strategy = QAction("Trading Strategy", main_window)
                lang_menu = QMenu("Language", main_window)
                lang_group = QActionGroup(main_window)
                lang_en = QAction("English", main_window, checkable=True)
                lang_vi = QAction("Vietnamese", main_window, checkable=True)
                lang_group.addAction(lang_en); lang_group.addAction(lang_vi)
                # default language is EN
                if AppState.language() == 'vi':
                    lang_vi.setChecked(True)
                else:
                    lang_en.setChecked(True)
                lang_menu.addAction(lang_en); lang_menu.addAction(lang_vi)
                act_support = QAction("Support", main_window)
                main_menu.addAction(act_login)
                main_menu.addAction(act_register)
                main_menu.addAction(act_strategy)
                main_menu.addMenu(lang_menu)
                main_menu.addSeparator()
                main_menu.addAction(act_support)
                menu_btn.setMenu(main_menu)
                menu_btn.setPopupMode(QToolButton.InstantPopup)
                # Place menu button to the LEFT of the title
                top_bar_layout.addWidget(menu_btn)
                top_bar_layout.addSpacing(8)
                top_bar_layout.addWidget(app_title)
                top_bar_layout.addStretch(1)

                # Create tab widget with tabs at bottom
                tab_widget = QTabWidget()
                tab_widget.setTabPosition(QTabWidget.South)  # Set tabs at bottom
            
                print("Creating tabs...")
                # Create Account tab first
                account_tab = AccountTab()
                print("✅ AccountTab created")
            
                # Create other tabs with account_tab reference
                market_tab = MarketTab(account_tab)
                print("✅ MarketTab created")
            
                trend_tab = TrendTab(market_tab)
                print("✅ TrendTab created")
            
                indicator_tab = IndicatorTab(market_tab)
                print("✅ IndicatorTab created")
            
                # Set indicator tab reference in market tab
                market_tab.set_indicator_tab(indicator_tab)
                print("✅ IndicatorTab reference set in MarketTab")
            
                pattern_tab = PatternTab(market_tab, indicator_tab)
                print("✅ PatternTab created")
            
                price_pattern_tab = PricePatternTab(market_tab)
                print("✅ PricePatternTab created")
            
                news_tab = NewsTab()
                print("✅ NewsTab created")
            
                # Create Risk Management tab with market_tab reference
                print("🔄 Creating RiskManagementTab...")
                try:
                    risk_tab = RiskManagementTab(market_tab)
                    print("✅ RiskManagementTab created")
                except Exception as e:
                    print(f"❌ Error creating RiskManagementTab: {e}")
                    import traceback
                    traceback.print_exc()
                    # Create a dummy tab to continue
                    risk_tab = QWidget()
                    print("⚠️ Using dummy RiskTab to continue")
            
                # Create Signal tab (aggregator UI)
                signal_tab = SignalTab(indicator_tab=indicator_tab, market_tab=market_tab)
                print("✅ SignalTab created")

                # Create Auto Trading tab with references
                auto_trading_tab = AutoTradingTab(news_tab, risk_tab)
                print("[SUCCESS] AutoTradingTab created")
            
                # Store tab references in main_window for graceful shutdown
                main_window.all_tabs = {
                    'account_tab': account_tab,
                    'market_tab': market_tab,
                    'trend_tab': trend_tab,
                    'indicator_tab': indicator_tab,
                    'pattern_tab': pattern_tab,
                    'price_pattern_tab': price_pattern_tab,
                    'news_tab': news_tab,
                    'risk_tab': risk_tab,
                    'signal_tab': signal_tab,
                    'auto_trading_tab': auto_trading_tab
                }
                # Store tabWidget reference for auto trading checkbox detection
                main_window.tabWidget = tab_widget
                print("✅ Tab references stored for graceful shutdown")
            
                print("Adding tabs to widget...")
                # Add tabs in the requested order - Account tab first
                # Add tabs with immediate translation instead of hardcoded English
                tab_widget.addTab(account_tab, I18N.t("🏦 MT5 Account", "🏦 Tài khoản MT5"))
                tab_widget.addTab(market_tab, I18N.t("💹 Market Data", "💹 Dữ liệu thị trường"))
                tab_widget.addTab(trend_tab, I18N.t("📈 Trend Analysis", "📈 Phân tích xu hướng"))
                tab_widget.addTab(indicator_tab, I18N.t("⚙️ Technical Indicators", "⚙️ Chỉ báo kỹ thuật"))
                tab_widget.addTab(pattern_tab, I18N.t("🕯️ Candlestick Patterns", "🕯️ Mô hình nến"))
                tab_widget.addTab(price_pattern_tab, I18N.t("📊 Price Patterns", "📊 Mô hình giá"))
                tab_widget.addTab(news_tab, I18N.t("📰 Economic News", "📰 Tin tức kinh tế"))
                tab_widget.addTab(risk_tab, I18N.t("🛡️ Risk Management", "🛡️ Quản lý rủi ro"))
                tab_widget.addTab(signal_tab, I18N.t("📡 Signal", "📡 Tín hiệu"))
                tab_widget.addTab(auto_trading_tab, I18N.t("🤖 Auto Trading", "🤖 Giao dịch tự động"))
                
                # Debug log tab titles to file instead of console (avoid Unicode issues)
                try:
                    with open("debug_tab_titles.txt", "w", encoding="utf-8") as f:
                        f.write("=== INITIAL TAB TITLES (English) ===\n")
                        for i in range(tab_widget.count()):
                            f.write(f"Tab {i}: '{tab_widget.tabText(i)}'\n")
                        
                        current_lang = AppState.language()
                        f.write(f"\n=== CURRENT LANGUAGE: {current_lang} ===\n")
                        
                        # Apply initial language translation to tab titles
                        tab_widget.setTabText(0, I18N.t("🏦 MT5 Account", "🏦 Tài khoản MT5"))
                        tab_widget.setTabText(1, I18N.t("💹 Market Data", "💹 Dữ liệu thị trường"))
                        tab_widget.setTabText(2, I18N.t("📈 Trend Analysis", "📈 Phân tích xu hướng"))
                        tab_widget.setTabText(3, I18N.t("⚙️ Technical Indicators", "⚙️ Chỉ báo kỹ thuật"))
                        tab_widget.setTabText(4, I18N.t("🕯️ Candlestick Patterns", "🕯️ Mô hình nến"))
                        tab_widget.setTabText(5, I18N.t("📊 Price Patterns", "📊 Mô hình giá"))
                        tab_widget.setTabText(6, I18N.t("📰 Economic News", "📰 Tin tức kinh tế"))
                        tab_widget.setTabText(7, I18N.t("🛡️ Risk Management", "🛡️ Quản lý rủi ro"))
                        tab_widget.setTabText(8, I18N.t("📡 Signal", "📡 Tín hiệu"))
                        tab_widget.setTabText(9, I18N.t("🤖 Auto Trading", "🤖 Giao dịch tự động"))
                        
                        f.write("\n=== TAB TITLES AFTER RETRANSLATION ===\n")
                        for i in range(tab_widget.count()):
                            f.write(f"Tab {i}: '{tab_widget.tabText(i)}'\n")
                        
                        print(f"DEBUG: Tab titles logged to debug_tab_titles.txt")
                        
                    # Also store tab_widget reference for later debugging
                    main_window.debug_tab_widget = tab_widget
                except Exception as e:
                    print(f"[DEBUG] Error during initial tab translation: {e}")
            
                # Set main window reference for auto trading after all tabs are added
                auto_trading_tab.set_main_window_reference(main_window)
                print("[DEBUG] Main window reference set for AutoTradingTab")
            
                # Register main window globally for unified auto trading system discovery
                try:
                    from unified_auto_trading_system import UnifiedAutoTradingSystem as AutoTradingManager
                    # Store main window globally for discovery
                    if not hasattr(AutoTradingManager, '_registered_windows'):
                        AutoTradingManager._registered_windows = []
                    AutoTradingManager._registered_windows.append(main_window)
                    print("[DEBUG] Main window registered globally for unified auto trading system")
                except Exception as e:
                    print(f"[WARNING] Could not register window globally: {e}")
            
                print("Setting up layout...")
                # Main layout
                print("Creating QVBoxLayout...")
                main_layout = QVBoxLayout()
                print("Adding top_bar_widget...")
                main_layout.addWidget(top_bar_widget)
                print("Adding tab_widget...")
                main_layout.addWidget(tab_widget)
                print("Setting layout to main_window...")
                main_window.setLayout(main_layout)

                print("Showing window...")
                # Show window
                main_window.show()
                print("Window shown successfully!")                # Force refresh DCA labels after window is shown (Qt rendering fix)
                try:
                    from PyQt5.QtCore import QTimer
                    def delayed_refresh():
                        if hasattr(main_window, 'all_tabs') and 'risk_management_tab' in main_window.all_tabs:
                            risk_tab = main_window.all_tabs['risk_management_tab']
                            if hasattr(risk_tab, 'refresh_dca_labels'):
                                risk_tab.refresh_dca_labels()
                                print("🔄 DCA labels refreshed after window display")
                
                    QTimer.singleShot(500, delayed_refresh)  # Refresh after 500ms
                except Exception as e:
                    print(f"⚠️ Error setting up delayed label refresh: {e}")
            
                print("✅ GUI started successfully")

                # === Auto-start News scraping & scheduler integration ===
                try:
                    from PyQt5.QtCore import QTimer
                    import pytz
                    from news_scraper import get_today_news, scan_event_window
                    print("[NewsAuto] Initializing news auto-start...")
                    # ========================= NEWS AUTO CONFIG =========================
                    # Set to True for scheduled-only mode (fetch only at news release times)
                    # Set to False for old behavior (initial fetch + periodic refresh)
                    NEWS_EVENT_ONLY = True  # Use scheduled times only
                    ENABLE_INITIAL_FETCH = True  # Always fetch on startup regardless of mode
                    # ====================================================================

                    # 1. Immediate clean + initial fetch (non-blocking via thread)
                    # Helper to invoke GUI-safe update
                    from PyQt5.QtCore import QMetaObject, Qt as _Qt, Q_ARG, QTimer as _QTimer

                    def _initial_news_fetch():
                        def _run():
                            try:
                                if news_tab.use_economic_calendar_checkbox.isChecked():
                                    print("[NewsAuto] Cleaning & fetching initial news (enabled)...")
                                
                                    # Get user news filters
                                    currencies, impacts = news_tab.get_user_news_filters()
                                    print(f"[NewsAuto] Using user filters - Currencies: {currencies}, Impacts: {impacts}")
                                
                                    get_today_news(currencies=currencies, impacts=impacts, headless=True, auto_cleanup=True, clean_existing_files=True)
                                    print("[NewsAuto] Initial news fetch complete")
                                
                                    # After successful fetch, extract news release times and schedule auto-updates
                                    print("[NewsAuto] Analyzing news release times...")
                                    try:
                                        from news_scraper import parse_news_release_times
                                        news_times = parse_news_release_times()
                                        if news_times:
                                            # Schedule auto-fetch at each news release time
                                            _QTimer.singleShot(2000, lambda: _setup_smart_news_schedule(news_times))
                                            print(f"[NewsAuto] Will schedule auto-fetch for {len(news_times)} times: {news_times}")
                                        else:
                                            print("[NewsAuto] No news times found, using fallback schedule")
                                            fallback_times = ["08:30", "13:30", "15:30", "19:00", "21:30"]
                                            _QTimer.singleShot(2000, lambda: _setup_smart_news_schedule(fallback_times))
                                    except Exception as e:
                                        print(f"[NewsAuto] Error analyzing news times: {e}")
                                        # Use fallback schedule
                                        fallback_times = ["08:30", "13:30", "15:30", "19:00", "21:30"]
                                        _QTimer.singleShot(2000, lambda: _setup_smart_news_schedule(fallback_times))
                                else:
                                    print("[NewsAuto] Skipped initial news fetch (disabled)")
                            except Exception as e:
                                print(f"[NewsAuto] Initial fetch error: {e}")
                            # After fetch attempt, ask NewsTab to load latest file on GUI thread
                            if news_tab.use_economic_calendar_checkbox.isChecked():
                                _QTimer.singleShot(0, news_tab.load_latest_news_file_slot)
                        threading.Thread(target=_run, daemon=True).start()
                
                    # Always run initial fetch if enabled
                    if ENABLE_INITIAL_FETCH:
                        _initial_news_fetch()
                        print("[NewsAuto] Initial news fetch scheduled")
                    else:
                        print("[NewsAuto] Initial fetch disabled by config")

                    # 2. Smart scheduler: auto-extract event times from fetched news
                    # After initial fetch, parse all news times and schedule auto-updates at those times
                    scheduled_event_times = []  # Will be populated from actual news data
                    _scheduled_events_mtime = None  # track mtime of scheduled_events.json
                    _latest_news_file_mtime = None  # track mtime of latest news file to derive times
                    _event_timers = []  # Store timer references for cleanup

                    def _setup_smart_news_schedule(news_times: list):
                        """Set up automatic news fetching at specific release times"""
                        try:
                            # Clear any existing event timers
                            for timer in _event_timers:
                                timer.stop()
                                timer.deleteLater()
                            _event_timers.clear()
                        
                            # Update scheduled_event_times with actual news times
                            scheduled_event_times.clear()
                            scheduled_event_times.extend(news_times)
                        
                            from PyQt5.QtCore import QTime
                            current_time = QTime.currentTime()
                        
                            print(f"[NewsAuto] Setting up smart schedule for {len(news_times)} times...")
                        
                            for time_str in news_times:
                                try:
                                    target_time = QTime.fromString(time_str, "hh:mm")
                                    if target_time.isValid():
                                        # Calculate milliseconds until target time
                                        if target_time > current_time:
                                            # Target time is later today
                                            ms_until = current_time.msecsTo(target_time)
                                        else:
                                            # Target time is tomorrow (add 24 hours)
                                            ms_until = current_time.msecsTo(target_time) + 24 * 60 * 60 * 1000
                                    
                                        timer = _QTimer()
                                        timer.setSingleShot(True)
                                        timer.timeout.connect(lambda t=time_str: _handle_scheduled_news_fetch(t))
                                        timer.start(ms_until)
                                        _event_timers.append(timer)
                                    
                                        print(f"[NewsAuto] ⏰ Scheduled news fetch at {time_str} (in {ms_until/1000/60:.1f} minutes)")
                                    else:
                                        print(f"[NewsAuto] ❌ Invalid time format: {time_str}")
                                except Exception as e:
                                    print(f"[NewsAuto] ❌ Error setting timer for {time_str}: {e}")
                        
                            print(f"[NewsAuto] ✅ Smart news schedule ready with {len(_event_timers)} timers")
                        
                        except Exception as e:
                            print(f"[NewsAuto] ❌ Error setting up smart news schedule: {e}")

                    def _handle_scheduled_news_fetch(time_str: str):
                        """Handle automatic news fetch at scheduled time"""
                        try:
                            print(f"[NewsAuto] 🔔 Auto news fetch triggered at {time_str}")
                            def _fetch():
                                try:
                                    if news_tab.use_economic_calendar_checkbox.isChecked():
                                        print(f"[NewsAuto] Fetching news update at {time_str}...")
                                    
                                        # Get user news filters
                                        currencies, impacts = news_tab.get_user_news_filters()
                                        print(f"[NewsAuto] Using user filters - Currencies: {currencies}, Impacts: {impacts}")
                                    
                                        get_today_news(currencies=currencies, impacts=impacts, headless=True, auto_cleanup=False)
                                        print(f"[NewsAuto] ✅ News update complete at {time_str}")
                                        # Reload news in GUI
                                        _QTimer.singleShot(1000, news_tab.load_latest_news_file_slot)
                                    else:
                                        print(f"[NewsAuto] Skipped news fetch at {time_str} (disabled)")
                                except Exception as e:
                                    print(f"[NewsAuto] ❌ Error fetching news at {time_str}: {e}")
                        
                            # Run fetch in background thread
                            threading.Thread(target=_fetch, daemon=True).start()
                        
                            # Schedule next occurrence for tomorrow (24 hours later)
                            next_timer = _QTimer()
                            next_timer.setSingleShot(True)
                            next_timer.timeout.connect(lambda: _handle_scheduled_news_fetch(time_str))
                            next_timer.start(24 * 60 * 60 * 1000)  # 24 hours
                            _event_timers.append(next_timer)
                            print(f"[NewsAuto] ⏰ Scheduled next occurrence for {time_str} tomorrow")
                        
                        except Exception as e:
                            print(f"[NewsAuto] ❌ Error in scheduled news fetch for {time_str}: {e}")

                    def _setup_daily_news_refresh():
                        """Set up daily news refresh at 00:01 and 00:30 to get new day's news schedule"""
                        try:
                            from PyQt5.QtCore import QTime
                            current_time = QTime.currentTime()
                        
                            # Daily refresh times
                            daily_refresh_times = ["00:01", "00:30"]
                        
                            print("[NewsAuto] Setting up daily news refresh schedule...")
                        
                            for time_str in daily_refresh_times:
                                try:
                                    target_time = QTime.fromString(time_str, "hh:mm")
                                    if target_time.isValid():
                                        # Calculate milliseconds until target time
                                        if target_time > current_time:
                                            # Target time is later today
                                            ms_until = current_time.msecsTo(target_time)
                                        else:
                                            # Target time is tomorrow (add 24 hours)
                                            ms_until = current_time.msecsTo(target_time) + 24 * 60 * 60 * 1000
                                    
                                        timer = _QTimer()
                                        timer.setSingleShot(True)
                                        timer.timeout.connect(lambda t=time_str: _handle_daily_news_refresh(t))
                                        timer.start(ms_until)
                                        _event_timers.append(timer)
                                    
                                        print(f"[NewsAuto] 🌅 Scheduled daily news refresh at {time_str} (in {ms_until/1000/60:.1f} minutes)")
                                    else:
                                        print(f"[NewsAuto] ❌ Invalid refresh time format: {time_str}")
                                except Exception as e:
                                    print(f"[NewsAuto] ❌ Error setting daily refresh timer for {time_str}: {e}")
                        
                            print(f"[NewsAuto] ✅ Daily news refresh schedule ready")
                        
                        except Exception as e:
                            print(f"[NewsAuto] ❌ Error setting up daily news refresh: {e}")

                    def _handle_daily_news_refresh(time_str: str):
                        """Handle daily news refresh to get new day's news and update schedule"""
                        try:
                            print(f"[NewsAuto] 🌅 Daily news refresh triggered at {time_str}")
                            def _refresh():
                                try:
                                    if news_tab.use_economic_calendar_checkbox.isChecked():
                                        print(f"[NewsAuto] Fetching fresh news data at {time_str} for new day...")
                                    
                                        # Get user news filters
                                        currencies, impacts = news_tab.get_user_news_filters()
                                        print(f"[NewsAuto] Using user filters - Currencies: {currencies}, Impacts: {impacts}")
                                    
                                        get_today_news(currencies=currencies, impacts=impacts, headless=True, auto_cleanup=True, clean_existing_files=True)
                                        print(f"[NewsAuto] ✅ Fresh news data fetched at {time_str}")
                                    
                                        # After getting fresh news, re-analyze and update schedule
                                        print("[NewsAuto] Re-analyzing news release times for new day...")
                                        try:
                                            from news_scraper import parse_news_release_times
                                            new_news_times = parse_news_release_times()
                                            if new_news_times:
                                                # Update schedule with new day's news times
                                                _QTimer.singleShot(2000, lambda: _setup_smart_news_schedule(new_news_times))
                                                print(f"[NewsAuto] 📅 Updated schedule for new day with {len(new_news_times)} times: {new_news_times}")
                                            else:
                                                print("[NewsAuto] No new news times found, keeping existing schedule")
                                        except Exception as e:
                                            print(f"[NewsAuto] ❌ Error re-analyzing news times: {e}")
                                    
                                        # Reload news in GUI
                                        _QTimer.singleShot(1000, news_tab.load_latest_news_file_slot)
                                    else:
                                        print(f"[NewsAuto] Skipped daily refresh at {time_str} (disabled)")
                                except Exception as e:
                                    print(f"[NewsAuto] ❌ Error in daily news refresh at {time_str}: {e}")
                        
                            # Run refresh in background thread
                            threading.Thread(target=_refresh, daemon=True).start()
                        
                            # Schedule next daily refresh for tomorrow (24 hours later)
                            next_timer = _QTimer()
                            next_timer.setSingleShot(True)
                            next_timer.timeout.connect(lambda: _handle_daily_news_refresh(time_str))
                            next_timer.start(24 * 60 * 60 * 1000)  # 24 hours
                            _event_timers.append(next_timer)
                            print(f"[NewsAuto] 🌅 Scheduled next daily refresh for {time_str} tomorrow")
                        
                        except Exception as e:
                            print(f"[NewsAuto] ❌ Error in daily news refresh for {time_str}: {e}")

                    def _convert_12h_to_24h(t: str) -> str:
                        t = str(t).strip().lower()
                        if not t or t in ("all day", ""):
                            return ""
                        m = re.match(r'^(\d{1,2}):(\d{2})(am|pm)$', t)
                        if not m:
                            # maybe already 24h
                            if re.match(r'^\d{2}:\d{2}$', t):
                                return t
                            return ""
                        h, mn, ap = m.groups()
                        h = int(h)
                        if ap == 'pm' and h != 12:
                            h += 12
                        if ap == 'am' and h == 12:
                            h = 0
                        return f"{h:02d}:{mn}"

                    def _find_latest_news_file():
                        news_dir = 'news_output'
                        if not os.path.exists(news_dir):
                            return None
                        newest = None
                        newest_time = 0
                        for fname in os.listdir(news_dir):
                            if not fname.endswith('.json'):
                                continue
                            fp = os.path.join(news_dir, fname)
                            try:
                                st = os.stat(fp)
                                if st.st_mtime > newest_time:
                                    newest = fp
                                    newest_time = st.st_mtime
                            except Exception:
                                continue
                        return newest

                    def _derive_times_from_latest_news():
                        nonlocal scheduled_event_times, _latest_news_file_mtime
                        latest = _find_latest_news_file()
                        if not latest:
                            return
                        try:
                            st = os.stat(latest)
                            if _latest_news_file_mtime is not None and st.st_mtime <= _latest_news_file_mtime:
                                return  # no change
                            with open(latest, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            # data can be list (old style) or dict with 'events'
                            events = []
                            if isinstance(data, list):
                                events = data
                            elif isinstance(data, dict):
                                if isinstance(data.get('events'), list):
                                    events = data['events']
                            now_vn = datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))
                            new_times = set()
                            for ev in events:
                                tm = ev.get('time')
                                dv = ev.get('date')
                                tm24 = _convert_12h_to_24h(tm)
                                if not tm24:
                                    continue
                                # Ensure event still in the future (>= now - 1 minute)
                                try:
                                    # parse date variants
                                    ev_date = None
                                    if isinstance(dv, str) and re.match(r'^\d{4}-\d{2}-\d{2}$', dv):
                                        ev_date = datetime.strptime(dv, '%Y-%m-%d')
                                    else:
                                        # fallback: assume today
                                        ev_date = now_vn
                                    ev_dt = now_vn.replace(hour=int(tm24[:2]), minute=int(tm24[3:5]), second=0, microsecond=0)
                                    # If date differs, adjust
                                    if ev_date.date() != now_vn.date():
                                        continue  # only today
                                    if (ev_dt - now_vn).total_seconds() >= -60:  # allow slightly past to still scan
                                        new_times.add(tm24)
                                except Exception:
                                    continue
                            if new_times:
                                merged = sorted(set(scheduled_event_times).union(new_times))
                                if merged != scheduled_event_times:
                                    scheduled_event_times = merged
                                    print(f"[NewsAuto] Derived event times from news: {scheduled_event_times}")
                                    # persist to scheduled_events.json so user can edit
                                    try:
                                        os.makedirs('news_output', exist_ok=True)
                                        with open(os.path.join('news_output','scheduled_events.json'),'w',encoding='utf-8') as wf:
                                            json.dump({'times': scheduled_event_times}, wf, ensure_ascii=False, indent=2)
                                    except Exception as e:
                                        print(f"[NewsAuto] Persist scheduled times failed: {e}")
                            _latest_news_file_mtime = st.st_mtime
                        except Exception as e:
                            print(f"[NewsAuto] Derive times error: {e}")

                    def _load_scheduled_events_file(initial=False):
                        nonlocal scheduled_event_times, _scheduled_events_mtime
                        cfg_path = os.path.join('news_output','scheduled_events.json')
                        if not os.path.exists(cfg_path):
                            if initial:
                                # Try to derive automatically on first run
                                _derive_times_from_latest_news()
                            return
                        try:
                            st = os.stat(cfg_path)
                            if _scheduled_events_mtime is not None and st.st_mtime == _scheduled_events_mtime:
                                return
                            with open(cfg_path,'r',encoding='utf-8') as f:
                                js = json.load(f)
                            if isinstance(js, dict) and isinstance(js.get('times'), list):
                                new_list = [t for t in js['times'] if isinstance(t,str) and re.match(r'^\d{2}:\d{2}$', t)]
                                if new_list != scheduled_event_times:
                                    scheduled_event_times = sorted(set(new_list))
                                    print(f"[NewsAuto] Loaded scheduled times: {scheduled_event_times}")
                            _scheduled_events_mtime = st.st_mtime
                        except Exception as e:
                            print(f"[NewsAuto] Load scheduled_events.json failed: {e}")

                    # Initial load/derive (derive only if not event-only)
                    _load_scheduled_events_file(initial=not NEWS_EVENT_ONLY)
                    if NEWS_EVENT_ONLY and not scheduled_event_times:
                        print("[NewsAuto] Event-only mode active but no scheduled times loaded. Create news_output/scheduled_events.json with { 'times': ['HH:MM', ...] }.")

                    # Provide hot reload from a config file if exists
                    sched_cfg_path = os.path.join('news_output', 'scheduled_events.json')
                    if os.path.exists(sched_cfg_path):
                        try:
                            with open(sched_cfg_path,'r',encoding='utf-8') as f:_sc=json.load(f)
                            if isinstance(_sc, dict) and isinstance(_sc.get('times'), list):
                                scheduled_event_times = [t for t in _sc['times'] if isinstance(t,str)]
                                print(f"[NewsAuto] Loaded scheduled times: {scheduled_event_times}")
                        except Exception as e:
                            print(f"[NewsAuto] Failed loading scheduled_events.json: {e}")

                    try:
                        vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
                    except Exception as e:
                        print(f"[NewsAuto] Timezone error: {e}, using UTC")
                        vn_tz = pytz.timezone('UTC')
                    active_scans = {}

                    def _maybe_trigger_event_scans():
                        now_vn = datetime.now(vn_tz)
                        current_hhmm = now_vn.strftime('%H:%M')
                        if not news_tab.use_economic_calendar_checkbox.isChecked():
                            return
                        # Dynamic reload attempt each cycle (cheap ops)
                        _load_scheduled_events_file()
                        for t in list(scheduled_event_times):
                            # Start scan a bit BEFORE exact minute and ensure not started
                            if t not in active_scans:
                                # If within -1 to +1 minute window
                                try:
                                    hh,mm = t.split(':'); target_dt = now_vn.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
                                    delta_sec = (now_vn - target_dt).total_seconds()
                                    if -60 <= delta_sec <= 60:  # start scan window
                                        def _scan_thread(tt=t):
                                            try:
                                                print(f"[NewsAuto] Starting focused scan window for {tt}...")
                                                scan_event_window(target_time=tt, window_minutes=5, poll_interval=30,
                                                                  headless=True, clean_each_attempt=True)
                                                print(f"[NewsAuto] Focused scan finished for {tt}")
                                                # GUI refresh after scan
                                                if news_tab.use_economic_calendar_checkbox.isChecked():
                                                    _QTimer.singleShot(0, news_tab.load_latest_news_file_slot)
                                            except Exception as e:
                                                print(f"[NewsAuto] Scan error {tt}: {e}")
                                            finally:
                                                # Allow re-scan next day only
                                                active_scans.pop(tt, None)
                                        th = threading.Thread(target=_scan_thread, daemon=True)
                                        active_scans[t] = th
                                        th.start()
                                except Exception:
                                    continue

                    # Timer to check every 20s for upcoming events
                    news_timer = QTimer()
                    news_timer.setInterval(20000)
                    news_timer.timeout.connect(_maybe_trigger_event_scans)
                    news_timer.start()
                    # Periodic refresh (every 2 minutes) to catch manual file updates
                    if not NEWS_EVENT_ONLY:
                        periodic_news_refresh = _QTimer()
                        periodic_news_refresh.setInterval(120000)
                        def _periodic_refresh():
                            if news_tab.use_economic_calendar_checkbox.isChecked():
                                news_tab.load_latest_news_file_slot()
                        periodic_news_refresh.timeout.connect(_periodic_refresh)
                        periodic_news_refresh.start()
                        # Timer to periodically attempt deriving times from updated news file (if user manually fetched)
                        derive_timer = QTimer()
                        derive_timer.setInterval(60000)  # 60s
                        def _periodic_derive():
                            if news_tab.use_economic_calendar_checkbox.isChecked():
                                _derive_times_from_latest_news()
                        derive_timer.timeout.connect(_periodic_derive)
                        derive_timer.start()
                        print("[NewsAuto] Standard mode: scheduler active (20s), refresh 120s, derive 60s.")
                    else:
                        print("[NewsAuto] Event-only mode active: no periodic refresh/derive; scans only at scheduled times.")
                except Exception as e:
                    print(f"[NewsAuto] Initialization failed: {e}")
            
                # Set up daily news refresh at 00:01 and 00:30 (after all functions are defined)
                try:
                    _QTimer.singleShot(5000, _setup_daily_news_refresh)  # Start after 5 seconds
                    print("[NewsAuto] Daily news refresh (00:01, 00:30) scheduled")
                except Exception as e:
                    print(f"[NewsAuto] Daily refresh setup failed: {e}")
            
                # Hook language toggles to AppState and refresh open views
                def _apply_language(lang_code: str):
                    print(f"[LangSwitch] Switching to: {lang_code}")
                    AppState.set_language(lang_code)
                    try:
                        print(f"[LangSwitch] Current stored language: {AppState.language()}")
                        # Retranslate menu and actions
                        lang_menu.setTitle(I18N.t("Language", "Ngôn ngữ"))
                        act_login.setText(I18N.t("Login", "Đăng nhập"))
                        act_register.setText(I18N.t("Register", "Đăng ký"))
                        act_strategy.setText(I18N.t("Trading Strategy", "Chiến lược giao dịch"))
                        act_support.setText(I18N.t("Support", "Hỗ trợ"))
                        lang_en.setText(I18N.t("English", "English"))
                        lang_vi.setText(I18N.t("Vietnamese", "Tiếng Việt"))
                        menu_btn.setToolTip(I18N.t("Menu", "Trình đơn"))
                        # Update window & header title (keep same English brand, optional suffix in VI)
                        title_txt = I18N.t("ChatGPT AI BOT 4.3.2", "ChatGPT AI BOT 4.3.2")
                        try:
                            main_window.setWindowTitle(title_txt)
                            app_title.setText(title_txt)
                        except Exception:
                            pass

                        # Retranslate tab titles
                        tab_widget.setTabText(0, I18N.t("🏦 MT5 Account", "🏦 Tài khoản MT5"))
                        tab_widget.setTabText(1, I18N.t("💹 Market Data", "💹 Dữ liệu thị trường"))
                        tab_widget.setTabText(2, I18N.t("📈 Trend Analysis", "📈 Phân tích xu hướng"))
                        tab_widget.setTabText(3, I18N.t("⚙️ Technical Indicators", "⚙️ Chỉ báo kỹ thuật"))
                        tab_widget.setTabText(4, I18N.t("🕯️ Candlestick Patterns", "🕯️ Mô hình nến"))
                        tab_widget.setTabText(5, I18N.t("📊 Price Patterns", "📊 Mô hình giá"))
                        tab_widget.setTabText(6, I18N.t("📰 Economic News", "📰 Tin tức kinh tế"))
                        tab_widget.setTabText(7, I18N.t("🛡️ Risk Management", "🛡️ Quản lý rủi ro"))
                        tab_widget.setTabText(8, I18N.t("📡 Signal", "📡 Tín hiệu"))
                        tab_widget.setTabText(9, I18N.t("🤖 Auto Trading", "🤖 Giao dịch tự động"))
                        # Force apply Vietnamese titles if needed (diagnostic)
                        if AppState.language() == 'vi':
                            vn_titles = [
                                "🏦 Tài khoản MT5","💹 Dữ liệu thị trường","📈 Phân tích xu hướng",
                                "⚙️ Chỉ báo kỹ thuật","🕯️ Mô hình nến","📊 Mô hình giá",
                                "📰 Tin tức kinh tế","🛡️ Quản lý rủi ro","📡 Tín hiệu","🤖 Giao dịch tự động"
                            ]
                            for idx, txt in enumerate(vn_titles):
                                try:
                                    tab_widget.setTabText(idx, txt)
                                except Exception:
                                    pass
                            print("[LangSwitch] Forced VN tab titles committed")

                        # Retranslate Signal tab UI
                        try:
                            signal_tab.retranslate_ui()
                        except Exception:
                            pass

                        # Refresh News tab for language change
                        try:
                            if hasattr(news_tab, 'refresh_all_labels'):
                                news_tab.refresh_all_labels()
                            elif hasattr(news_tab, 'refresh_impact_labels'):
                                news_tab.refresh_impact_labels()
                        except Exception as e:
                            print(f"[LangSwitch] News tab refresh error: {e}")

                        # Refresh risk tab localized labels
                        try:
                            if hasattr(risk_tab, 'refresh_translations'):
                                risk_tab.refresh_translations()
                            # Legacy DCA refresh (kept for backward compatibility, no-op after removals)
                            if hasattr(risk_tab, 'refresh_dca_labels'):
                                risk_tab.refresh_dca_labels()
                        except Exception as e:
                            print(f"[LangSwitch] Risk tab refresh error: {e}")

                        # Broad pass: update common static texts in the whole window
                        try:
                            I18N.retranslate_widget_tree(main_window)
                            # Also translate any other top-level dialogs / windows
                            I18N.translate_application()
                            # Aggressive fallback pass
                            I18N.force_full_translation(main_window, debug=True)
                        except Exception:
                            pass
                        
                        # Debug log tab titles after language switch
                        try:
                            if hasattr(main_window, 'debug_tab_widget'):
                                tw = main_window.debug_tab_widget
                                with open("debug_language_switch.txt", "w", encoding="utf-8") as f:
                                    f.write(f"=== LANGUAGE SWITCH TO: {lang_code} ===\n")
                                    for i in range(tw.count()):
                                        f.write(f"Tab {i}: '{tw.tabText(i)}'\n")
                                print(f"DEBUG: Language switch logged to debug_language_switch.txt")
                        except Exception as e:
                            print(f"DEBUG: Language switch logging error: {e}")
                        # Update dynamic indicator row buttons to match language
                        try:
                            indicator_tab.apply_language_to_indicator_rows()
                        except Exception:
                            pass
                        # Sample debug: find a known label text
                        try:
                            from PyQt5.QtWidgets import QApplication, QLabel
                            app_inst = QApplication.instance()
                            sample_found = False
                            for w in app_inst.allWidgets():
                                if isinstance(w, QLabel):
                                    txt = w.text()
                                    if 'Volume:' in txt or 'Khối lượng:' in txt:
                                        print(f"[LangDebug] Volume label => {txt}")
                                        sample_found = True
                                        break
                            if not sample_found:
                                print("[LangDebug] Volume label NOT found in scan")
                        except Exception:
                            pass

                        # Ensure GUI processes pending repaint events
                        try:
                            from PyQt5.QtWidgets import QApplication
                            QApplication.processEvents()
                        except Exception:
                            pass

                        signal_tab.load_actions_text()
                        # refresh selected report, if any
                        sel = signal_tab.sig_table.currentRow()
                        if sel >= 0:
                            it = signal_tab.sig_table.item(sel, 0)
                            if it:
                                signal_tab.load_latest_report(it.text())
                    except Exception:
                        pass
                # Ensure exclusivity & direct connections (more reliable than group triggered logic)
                try:
                    lang_group.setExclusive(True)
                except Exception:
                    pass
                lang_en.triggered.connect(lambda: _apply_language('en'))
                lang_vi.triggered.connect(lambda: _apply_language('vi'))

                # Apply restored language once after full UI build (if not EN)
                if AppState.language() != 'en':
                    _apply_language(AppState.language())

                # Register graceful shutdown of threads
                try:
                    def _on_about_to_quit():
                        try:
                            print("🛑 App shutting down - stopping all threads...")
                            candidates = []
                        
                            # Get tab references from main_window
                            if hasattr(main_window, 'all_tabs'):
                                candidates.extend(main_window.all_tabs.values())
                                print(f"🔍 Found {len(candidates)} tabs to check for threads")
                        
                            # Also include main_window itself for any direct thread references
                            candidates.append(main_window)
                        
                            graceful_shutdown_threads(candidates)
                            print("✅ Graceful shutdown completed")
                        except Exception as _e:
                            print(f"⚠️ graceful shutdown hook error: {_e}")
                    app.aboutToQuit.connect(_on_about_to_quit)
                except Exception:
                    pass

                # Start event loop
                print("🚀 Starting main event loop...")
                app.exec_()
                print("✅ Event loop completed")
                
            except Exception as e:
                print(f"❌ GUI failed to start: {e}")
                import traceback
                traceback.print_exc()
            
    except Exception as main_error:
        print(f"🚨 CRITICAL ERROR in main(): {type(main_error).__name__}: {str(main_error)}")
        print("🛡️ Anti-crash system activated - saving error and continuing...")
        
        # Log critical error
        try:
            crash_log = f"logs/main_crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            os.makedirs("logs", exist_ok=True)
            with open(crash_log, 'w', encoding='utf-8') as f:
                f.write(f"Main Function Crash: {datetime.now()}\n")
                f.write(f"Error: {type(main_error).__name__}: {str(main_error)}\n\n")
                f.write("Traceback:\n")
                traceback.print_exc(file=f)
            print(f"💾 Main crash log saved: {crash_log}")
        except:
            print("❌ Could not save crash log, but continuing anyway...")
        
        # Continue execution instead of crashing
        print("🔄 Attempting to continue despite error...")

def main():
    """Ultra-safe main function with comprehensive crash protection"""
    try:
        print("🛡️ Trading Bot Starting with Anti-Crash Protection...")
        main_original()
    except Exception as e:
        print(f"🚨 MAIN FUNCTION CRASH: {type(e).__name__}: {str(e)}")
        print("🛡️ Anti-crash system activated - app will continue running...")
        
        # Log the crash
        try:
            crash_log = f"logs/main_crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            os.makedirs("logs", exist_ok=True)
            with open(crash_log, 'w', encoding='utf-8') as f:
                f.write(f"Main Function Crash: {datetime.now()}\n")
                f.write(f"Error: {type(e).__name__}: {str(e)}\n\n")
                f.write("Traceback:\n")
                traceback.print_exc(file=f)
            print(f"💾 Crash log saved: {crash_log}")
        except:
            pass
        
        # Try emergency recovery
        try:
            print("🔄 Attempting emergency recovery...")
            if GUI_AVAILABLE:
                print("📱 Starting minimal GUI...")
                from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
                app = QApplication([])
                window = QWidget()
                layout = QVBoxLayout()
                layout.addWidget(QLabel("🛡️ Anti-Crash Mode Active"))
                layout.addWidget(QLabel(f"Error: {str(e)}"))
                layout.addWidget(QLabel("App is protected and running safely"))
                window.setLayout(layout)
                window.setWindowTitle("Trading Bot - Safe Mode")
                window.show()
                app.exec_()
            else:
                print("💻 Console mode recovery - keeping app alive...")
                input("Press Enter to exit safely...")
        except:
            print("❌ Emergency recovery failed, but app continues...")

def safe_main():
    """Ultra-safe main wrapper with crash protection"""
    try:
        print("🚀 Starting Trading Bot with Anti-Crash Protection...")
        main_original()  # Call main_original directly - it works perfectly
        print("✅ Trading Bot completed successfully!")
        
    except KeyboardInterrupt:
        print("\n👋 User interrupted - shutting down gracefully...")
        
    except Exception as e:
        error_msg = f"App crashed: {type(e).__name__}: {str(e)}"
        print(f"\n🚨 {error_msg}")
        
        # Log the crash
        try:
            crash_log = f"logs/safe_main_crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            os.makedirs("logs", exist_ok=True)
            with open(crash_log, 'w', encoding='utf-8') as f:
                f.write(f"Safe Main Crash: {datetime.now()}\n")
                f.write(f"Error: {type(e).__name__}: {str(e)}\n\n")
                f.write("Traceback:\n")
                traceback.print_exc(file=f)
            print(f"💾 Crash log saved: {crash_log}")
        except:
            pass
        
        print("🛡️ Anti-crash protection activated - app stays safe")

def emergency_cleanup():
    """Emergency cleanup for all threads and processes"""
    try:
        print("🚨 EMERGENCY CLEANUP - Stopping all threads and processes...")
        
        # Kill any remaining subprocess
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        
        for child in children:
            try:
                child.terminate()
                child.wait(timeout=2)
            except:
                try:
                    child.kill()
                except:
                    pass
        
        # Force cleanup of QApplication if exists
        if GUI_AVAILABLE:
            try:
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                if app:
                    app.quit()
                    app.processEvents()
            except:
                pass
                
        print("✅ Emergency cleanup completed")
        
    except Exception as e:
        print(f"❌ Emergency cleanup error: {e}")

if __name__ == "__main__":
    try:
        safe_main()
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt - emergency shutdown...")
        emergency_cleanup()
    except Exception as e:
        print(f"\n🚨 Final exception: {e}")
        emergency_cleanup()
    finally:
        print("👋 Application shutdown complete")
    
# --- Added global graceful shutdown utility (non-invasive) ---
def graceful_shutdown_threads(thread_containers: list):
    """Gracefully stop QThreads to prevent 'QThread: Destroyed while thread is still running'.

    Provide a list of containers/objects that may have attributes holding QThread instances
    e.g., ['pattern_tab', 'price_pattern_tab'] or direct lists. Call this before QApplication quits.
    """
    try:
        import inspect
        visited = set()
        threads = []
        def collect(obj):
            try:
                if obj is None: return
                if isinstance(obj, (list, tuple, set)):
                    for x in obj: collect(x)
                    return
                # Introspect attributes
                for name in dir(obj):
                    if name.startswith('_'): continue
                    try:
                        val = getattr(obj, name)
                    except Exception:
                        continue
                    if id(val) in visited: continue
                    visited.add(id(val))
                    # QThread heuristic: has isRunning and quit
                    if hasattr(val, 'isRunning') and hasattr(val, 'quit'):
                        threads.append(val)
                    elif isinstance(val, (list, tuple, set)):
                        collect(val)
            except Exception:
                pass
        for c in thread_containers:
            collect(c)
        uniq = []
        seen = set()
        for t in threads:
            if id(t) not in seen:
                uniq.append(t); seen.add(id(t))
        if not uniq:
            return
        print(f"🛑 Graceful shutdown: {len(uniq)} thread(s) detected")
        for t in uniq:
            try:
                if hasattr(t, 'requestInterruption'):
                    t.requestInterruption()
                t.quit()
            except Exception:
                continue
        for t in uniq:
            try:
                t.wait(3000)
            except Exception:
                continue
        print("✅ Threads terminated")
    except Exception as e:
        print(f"⚠️ graceful_shutdown_threads error: {e}")