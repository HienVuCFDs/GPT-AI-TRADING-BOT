#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ UNIFIED AUTO TRADING SYSTEM
Combines Auto Trading Controller + Manager into one comprehensive system
Date: October 10, 2025
"""

import json
import os
import sys
import time
import logging
import threading
import subprocess
import importlib
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QPushButton, QCheckBox


class TradingPhase(Enum):
    """Trading pipeline phases"""
    IDLE = "idle"
    MARKET_DATA = "market_data"
    TREND_ANALYSIS = "trend_analysis"
    INDICATORS = "indicators"
    CANDLESTICK_PATTERNS = "candlestick_patterns"
    PRICE_PATTERNS = "price_patterns"
    SIGNAL_GENERATION = "signal_generation"
    ORDER_EXECUTION = "order_execution"
    COMPLETED = "completed"
    ERROR = "error"


class AutoTradingSignals(QObject):
    """Qt Signals for thread-safe GUI communication"""
    status_updated = pyqtSignal(str)
    log_added = pyqtSignal(str)
    phase_changed = pyqtSignal(str)
    cycle_completed = pyqtSignal(str, bool, float)


class TradingConfig:
    """Configuration management for auto trading"""
    
    def __init__(self):
        self.update_interval = 60  # seconds
        self.large_cycle_minutes = 1  # minutes (faster cycles for testing)
        self.small_cycle_interval = 30  # seconds
        self.script_timeout = 120  # seconds
        self.max_retries = 3
        self.debug_mode = True
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'update_interval': self.update_interval,
            'large_cycle_minutes': self.large_cycle_minutes,
            'small_cycle_interval': self.small_cycle_interval,
            'script_timeout': self.script_timeout,
            'max_retries': self.max_retries,
            'debug_mode': self.debug_mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingConfig':
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class SystemController:
    """Controls system state and safety features"""
    
    @staticmethod
    def get_current_status() -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'emergency_stop': False,
            'auto_mode': False,
            'positions_count': 0,
            'account_profit': 0.0,
            'pending_actions': 0,
            'risk_controls': {}
        }
        
        # Check emergency stop
        if os.path.exists("emergency_stop.flag"):
            status['emergency_stop'] = True
        
        # Check auto mode
        try:
            with open("risk_management/risk_settings.json", 'r') as f:
                risk_settings = json.load(f)
            status['auto_mode'] = risk_settings.get('enable_auto_mode', False)
            status['risk_controls'] = {
                'max_risk_percent': risk_settings.get('max_risk_percent', 'OFF'),
                'max_positions': risk_settings.get('max_positions', 'OFF'),
                'emergency_stop_drawdown': risk_settings.get('emergency_stop_drawdown', 'OFF'),
                'disable_emergency_stop': risk_settings.get('disable_emergency_stop', True)
            }
        except:
            pass
        
        # Check account status
        try:
            with open("account_scans/mt5_essential_scan.json", 'r') as f:
                scan_data = json.load(f)
            
            account = scan_data.get('account', {})
            positions = scan_data.get('active_positions', [])
            
            status['positions_count'] = len(positions)
            status['account_profit'] = account.get('profit', 0.0)
            status['account_balance'] = account.get('balance', 0.0)
            status['account_equity'] = account.get('equity', 0.0)
        except:
            pass
        
        # Check pending actions
        try:
            with open("analysis_results/account_positions_actions.json", 'r') as f:
                actions_data = json.load(f)
            status['pending_actions'] = len(actions_data.get('actions', []))
        except:
            pass
        
        return status
    
    @staticmethod
    def safe_enable_auto_trading():
        """Enable auto trading safely with checks"""
        print("🔧 ENABLING AUTO TRADING SAFELY")
        print("=" * 50)
        
        # 1. Check current status
        status = SystemController.get_current_status()
        
        if status['emergency_stop']:
            print("❌ Cannot enable: Emergency stop is active!")
            print("   Run: python emergency_stop_auto_trading.py remove")
            return False
        
        if status['positions_count'] > 15:
            print(f"⚠️ Warning: {status['positions_count']} positions active (recommended <10)")
            confirm = input("Continue anyway? (y/N): ").strip().lower()
            if confirm != 'y':
                print("❌ Auto trading not enabled")
                return False
        
        # 2. Set safe risk parameters
        print("\n🛡️ Setting safe risk parameters...")
        
        risk_settings_path = "risk_management/risk_settings.json"
        try:
            with open(risk_settings_path, 'r') as f:
                risk_settings = json.load(f)
            
            # Safe defaults
            safe_settings = {
                'enable_auto_mode': True,
                'max_risk_percent': 1.5,
                'max_drawdown_percent': 4.0,
                'max_daily_loss_percent': 2.0,
                'max_positions': 8,
                'max_positions_per_symbol': 2,
                'emergency_stop_drawdown': 6.0,
                'disable_emergency_stop': False,
                'disable_news_avoidance': False,
                'avoid_news_minutes': 30,
                'trading_mode': '🤖 Tự động',
                'fixed_volume_lots': 0.1,  # Smaller volume
                'max_dca_levels': 3,       # Fewer DCA levels
                # 🚨 ENHANCED: Add duplicate prevention settings
                'duplicate_entry_distance_pips': 5.0,
                'enable_signal_based_adjustment': True,
                'opposite_signal_min_confidence': 85.0,  # Higher threshold
            }
            
            # Update settings
            risk_settings.update(safe_settings)
            
            with open(risk_settings_path, 'w') as f:
                json.dump(risk_settings, f, indent=2, ensure_ascii=False)
            
            print("   ✅ Safe risk parameters applied")
            
        except Exception as e:
            print(f"   ❌ Failed to update settings: {e}")
            return False
        
        # 3. Clear old actions
        print("\n🗑️ Clearing old actions...")
        try:
            actions_path = "analysis_results/account_positions_actions.json"
            empty_actions = {
                "summary": {
                    "total_symbols_analyzed": 0,
                    "signals_generated": 0,
                    "actions_by_type": {},
                    "high_priority_actions": 0,
                    "total_actions": 0
                },
                "actions": [],
                "timestamp": datetime.now().isoformat(),
                "status": "READY"
            }
            
            with open(actions_path, 'w') as f:
                json.dump(empty_actions, f, indent=2, ensure_ascii=False)
            
            print("   ✅ Actions cleared")
        except Exception as e:
            print(f"   ❌ Failed to clear actions: {e}")
        
        print(f"\n✅ AUTO TRADING ENABLED SAFELY")
        print("=" * 50)
        print("🛡️ SAFETY MEASURES ACTIVE:")
        print("   - Max Risk: 1.5% per trade")
        print("   - Max Positions: 8 total, 2 per symbol")
        print("   - Emergency Stop: 6% drawdown")
        print("   - Volume: 0.1 lots (reduced)")
        print("   - DCA Levels: 3 maximum")
        print("   - News Avoidance: 30 minutes")
        print("   - Duplicate Prevention: Enhanced")
        print("   - Opposite Signal Threshold: 85%")
        
        print(f"\n📊 CURRENT STATUS:")
        print(f"   - Positions: {status['positions_count']}")
        print(f"   - Account P/L: ${status['account_profit']:.2f}")
        
        return True
    
    @staticmethod
    def disable_auto_trading():
        """Disable auto trading"""
        print("🛑 DISABLING AUTO TRADING")
        print("=" * 30)
        
        try:
            risk_settings_path = "risk_management/risk_settings.json"
            with open(risk_settings_path, 'r') as f:
                risk_settings = json.load(f)
            
            risk_settings['enable_auto_mode'] = False
            risk_settings['trading_mode'] = '👨‍💼 Thủ công'
            
            with open(risk_settings_path, 'w') as f:
                json.dump(risk_settings, f, indent=2, ensure_ascii=False)
            
            print("✅ Auto trading disabled")
            return True
            
        except Exception as e:
            print(f"❌ Failed to disable: {e}")
            return False
    
    @staticmethod
    def show_status():
        """Show system status"""
        print("📊 AUTO TRADING STATUS")
        print("=" * 40)
        
        status = SystemController.get_current_status()
        
        # Emergency status
        if status['emergency_stop']:
            print("🚨 EMERGENCY STOP: ACTIVE")
        else:
            print("✅ Emergency Stop: Inactive")
        
        # Auto mode status
        if status['auto_mode']:
            print("🤖 Auto Mode: ENABLED")
        else:
            print("👨‍💼 Auto Mode: DISABLED (Manual)")
        
        # Account status
        print(f"\n💰 Account Status:")
        print(f"   Balance: ${status.get('account_balance', 0):,.2f}")
        print(f"   Equity: ${status.get('account_equity', 0):,.2f}")
        print(f"   Profit/Loss: ${status['account_profit']:,.2f}")
        print(f"   Active Positions: {status['positions_count']}")
        print(f"   Pending Actions: {status['pending_actions']}")
        
        # Risk controls
        print(f"\n🛡️ Risk Controls:")
        controls = status['risk_controls']
        for key, value in controls.items():
            if key.startswith('disable_') and value:
                print(f"   ❌ {key}: DISABLED")
            elif key.startswith('max_'):
                print(f"   📊 {key}: {value}")
        
        # Risk level assessment
        risk_level = "LOW"
        risk_factors = []
        
        if status['positions_count'] > 15:
            risk_level = "HIGH"
            risk_factors.append(f"Too many positions ({status['positions_count']})")
        
        if status['account_profit'] < -200:
            risk_level = "HIGH" if risk_level == "LOW" else "CRITICAL"
            risk_factors.append(f"Large loss (${status['account_profit']:.2f})")
        
        if controls.get('disable_emergency_stop', True):
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
            risk_factors.append("Emergency stop disabled")
        
        if controls.get('max_risk_percent') == 'OFF':
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
            risk_factors.append("Risk limits disabled")
        
        print(f"\n⚠️ Risk Assessment: {risk_level}")
        if risk_factors:
            for factor in risk_factors:
                print(f"   - {factor}")
    
    @staticmethod
    def monitor_mode():
        """Monitor system continuously"""
        print("👁️ MONITORING MODE STARTED")
        print("Press Ctrl+C to stop")
        print("=" * 40)
        
        try:
            while True:
                status = SystemController.get_current_status()
                
                timestamp = datetime.now().strftime('%H:%M:%S')
                mode = "🤖 AUTO" if status['auto_mode'] else "👨‍💼 MANUAL"
                emergency = " 🚨 EMERGENCY" if status['emergency_stop'] else ""
                
                print(f"\r[{timestamp}] {mode}{emergency} | Pos: {status['positions_count']:2d} | P/L: ${status['account_profit']:7.2f} | Actions: {status['pending_actions']}", end='', flush=True)
                
                # Alert conditions
                if status['account_profit'] < -300 and not status['emergency_stop']:
                    print(f"\n🚨 ALERT: Large loss ${status['account_profit']:.2f} - Consider emergency stop!")
                
                if status['positions_count'] > 20:
                    print(f"\n⚠️ WARNING: {status['positions_count']} positions (too many!)")
                
                time.sleep(5)  # Update every 5 seconds
                
        except KeyboardInterrupt:
            print(f"\n\n👁️ Monitoring stopped")


class ScriptExecutor:
    """Handles script execution with proper error handling"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def run_script(self, script_path: str, args: List[str] = None, timeout: int = 120) -> bool:
        """Execute a Python script with arguments"""
        try:
            # 🔧 FIX: Get absolute path and ensure it exists
            full_script_path = os.path.abspath(script_path)
            if not os.path.exists(full_script_path):
                self.logger.error(f"❌ Script not found: {full_script_path}")
                return False
                
            # 🔧 FIX: Use proper Python executable path (handle spaces)
            python_exe = sys.executable
            cmd = [python_exe, full_script_path]
            if args:
                cmd.extend(args)
                
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # 🔧 FIX: Use proper working directory
            work_dir = os.path.dirname(full_script_path) if os.path.dirname(full_script_path) else os.getcwd()
            
            self.logger.info(f"🔧 Executing: {' '.join(cmd)}")
            self.logger.info(f"🔧 Working dir: {work_dir}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Script completed successfully: {script_path}")
                if result.stdout:
                    self.logger.info(f"STDOUT: {result.stdout[-200:]}")  # Last 200 chars
                return True
            else:
                self.logger.error(f"❌ Script failed (exit {result.returncode}): {script_path}")
                if result.stderr:
                    self.logger.error(f"STDERR: {result.stderr[:500]}...")
                if result.stdout:
                    self.logger.error(f"STDOUT: {result.stdout[:500]}...")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"⏰ Script timeout ({timeout}s): {script_path}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Script execution error: {e}")
            return False


class GUIController:
    """Handles GUI interactions and tab management"""
    
    def __init__(self, main_window, logger: logging.Logger):
        self.main_window = main_window
        self.logger = logger
    
    def get_selected_symbols_count(self) -> int:
        """Get number of selected symbols from Market Tab"""
        try:
            # Try to find market tab and get checked symbols
            if hasattr(self.main_window, 'market_tab'):
                symbols = getattr(self.main_window.market_tab, 'checked_symbols', [])
                count = len(symbols) if symbols else 5  # Default to 5 if none selected
                self.logger.info(f"📊 Found {count} selected symbols: {symbols}")
                return count
            else:
                self.logger.warning("⚠️ No market_tab found, using default symbol count: 5")
                return 5  # Default fallback
        except Exception as e:
            self.logger.error(f"❌ Error getting symbol count: {e}")
            return 5  # Safe fallback

    def get_selected_timeframes_count(self) -> int:
        """Get number of selected timeframes from Market Tab"""
        try:
            # Try to find market tab and get checked timeframes
            if hasattr(self.main_window, 'market_tab'):
                tf_checkboxes = getattr(self.main_window.market_tab, 'tf_checkboxes', {})
                selected_tfs = [tf for tf, checkbox in tf_checkboxes.items() if checkbox.isChecked()]
                count = len(selected_tfs) if selected_tfs else 4  # Default to 4 if none selected
                self.logger.info(f"📊 Found {count} selected timeframes: {selected_tfs}")
                return count
            else:
                self.logger.warning("⚠️ No market_tab found, using default timeframe count: 4")
                return 4  # Default fallback
        except Exception as e:
            self.logger.error(f"❌ Error getting timeframe count: {e}")
            return 4  # Safe fallback
        
    def calculate_adaptive_delay(self) -> Dict[str, float]:
        """Calculate adaptive delay based on selected symbols and timeframes"""
        try:
            if not self.main_window:
                return {'tab_switch': 1.0, 'button_click': 1.0, 'post_action': 5.0}
            
            # Count symbols and timeframes
            symbol_count = 0
            timeframe_count = 0
            
            # Try to get symbol count from GUI
            try:
                symbol_count = len(getattr(self.main_window, 'selected_symbols', []))
            except:
                symbol_count = 5  # Default estimate
            
            # Try to get timeframe count from GUI
            try:
                timeframe_count = len(getattr(self.main_window, 'selected_timeframes', []))
            except:
                timeframe_count = 3  # Default estimate
            
            # Calculate delays (more symbols/timeframes = longer delays)
            base_tab_delay = 0.5
            base_click_delay = 0.3
            base_post_delay = 3.0
            
            symbol_factor = max(1.0, symbol_count / 5.0)  # Scale based on 5 symbols baseline
            timeframe_factor = max(1.0, timeframe_count / 3.0)  # Scale based on 3 timeframes baseline
            
            combined_factor = (symbol_factor + timeframe_factor) / 2.0
            
            delays = {
                'tab_switch': base_tab_delay * combined_factor,
                'button_click': base_click_delay * combined_factor,
                'post_action': base_post_delay * combined_factor,
                'symbol_count': symbol_count,
                'timeframe_count': timeframe_count,
                'scaling_factor': combined_factor
            }
            
            self.logger.debug(f"📊 Adaptive delays: {symbol_count} symbols × {timeframe_count} timeframes = {combined_factor:.1f}x scaling")
            return delays
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating adaptive delay: {e}")
            return {'tab_switch': 1.0, 'button_click': 1.0, 'post_action': 5.0}
        
    def switch_to_tab(self, tab_names: List[str]) -> bool:
        """Switch to tab by trying multiple names with adaptive delay"""
        try:
            # 🔧 FIX: Use GUI main window if available, otherwise use main_window
            target_window = getattr(self.main_window, '_gui_main_window', self.main_window)
            
            if not target_window or not hasattr(target_window, 'tabWidget'):
                self.logger.error(f"❌ No valid tabWidget found. target_window: {target_window}")
                return False
            
            delays = self.calculate_adaptive_delay()
            
            tab_widget = target_window.tabWidget
            
            # 🔧 DEBUG: List all available tabs
            available_tabs = []
            for i in range(tab_widget.count()):
                tab_text = tab_widget.tabText(i).strip()
                available_tabs.append(tab_text)
            
            self.logger.info(f"🔍 Available tabs: {available_tabs}")
            self.logger.info(f"🔍 Looking for tabs: {tab_names}")
            
            # Try each tab name
            for tab_name in tab_names:
                for i in range(tab_widget.count()):
                    current_tab_text = tab_widget.tabText(i).strip()
                    
                    # Exact match or contains check
                    if (tab_name.lower() == current_tab_text.lower() or 
                        tab_name.lower() in current_tab_text.lower()):
                        
                        tab_widget.setCurrentIndex(i)
                        time.sleep(delays['tab_switch'])
                        
                        self.logger.info(f"📑 Switched to tab: {current_tab_text}")
                        return True
            
            self.logger.warning(f"⚠️ Tab not found: {tab_names}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error switching tab: {e}")
            return False
    
    def find_and_click_button(self, tab_widget, keywords: List[str], excluded: List[str] = None) -> bool:
        """Find and click button with keywords, excluding certain patterns with adaptive delay"""
        try:
            if not tab_widget:
                return False
            
            delays = self.calculate_adaptive_delay()
            excluded = excluded or []
            
            # Find buttons
            buttons = tab_widget.findChildren(QPushButton)
            
            # 🔧 DEBUG: List all available buttons
            all_button_texts = []
            visible_button_texts = []
            
            for button in buttons:
                button_text = button.text().strip()
                all_button_texts.append(button_text)
                
                if not button.isVisible() or not button.isEnabled():
                    continue
                    
                visible_button_texts.append(button_text)
                
                # Skip excluded buttons
                if any(exc.lower() in button_text.lower() for exc in excluded):
                    continue
            
            self.logger.info(f"🔍 All buttons in tab: {all_button_texts}")
            self.logger.info(f"🔍 Visible/enabled buttons: {visible_button_texts}")
            self.logger.info(f"🔍 Looking for keywords: {keywords}")
            
            for button in buttons:
                if not button.isVisible() or not button.isEnabled():
                    continue
                
                button_text = button.text().strip()
                
                # Skip excluded buttons
                if any(exc.lower() in button_text.lower() for exc in excluded):
                    continue
                
                # Check if button matches any keyword
                for keyword in keywords:
                    if keyword.lower() in button_text.lower():
                        self.logger.info(f"🔘 Clicking button: {button_text}")
                        
                        # Click with adaptive delay
                        button.click()
                        time.sleep(delays['button_click'])
                        
                        # Post-action delay for processing
                        time.sleep(delays['post_action'])
                        
                        return True
            
            self.logger.warning(f"⚠️ Button not found with keywords: {keywords}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error clicking button: {e}")
            return False
    
    def is_checkbox_enabled(self, tab_names: List[str], keywords: List[str]) -> bool:
        """Check if checkbox is enabled in a tab"""
        try:
            if not self.switch_to_tab(tab_names):
                return None
            
            current_tab = self.main_window.tabWidget.currentWidget()
            checkboxes = current_tab.findChildren(QCheckBox)
            
            for checkbox in checkboxes:
                checkbox_text = checkbox.text().strip()
                if any(keyword.lower() in checkbox_text.lower() for keyword in keywords):
                    return checkbox.isChecked()
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error checking checkbox: {e}")
            return None


class TradingStep:
    """Base class for trading pipeline steps"""
    
    def __init__(self, name: str, phase: TradingPhase, required: bool = True):
        self.name = name
        self.phase = phase
        self.required = required
        self.enabled = True
        
    def is_enabled(self, gui_controller: GUIController) -> bool:
        """Override in subclasses to check if step should run"""
        if self.required:
            return True
        return self.enabled
        
    def execute_gui(self, gui_controller: GUIController) -> bool:
        """Override in subclasses for GUI-based execution"""
        return False
        
    def execute_fallback(self, script_executor: ScriptExecutor) -> bool:
        """Override in subclasses for fallback execution"""
        return False
        
    def execute(self, gui_controller: GUIController, script_executor: ScriptExecutor) -> bool:
        """Execute step with GUI first, fallback if needed"""
        gui_controller.logger.info(f"[STEP] 🚀 Starting execution of: {self.name}")
        
        if gui_controller.main_window:
            try:
                if self.execute_gui(gui_controller):
                    gui_controller.logger.info(f"[GUI] ✅ GUI execution successful for: {self.name}")
                    return True
                else:
                    gui_controller.logger.warning(f"[GUI] ⚠️ GUI execution failed for: {self.name}")
            except Exception as e:
                gui_controller.logger.error(f"[GUI] ❌ GUI execution error for {self.name}: {e}")
        else:
            gui_controller.logger.info(f"[GUI] ℹ️ No main window available for: {self.name}")
                
        gui_controller.logger.info(f"[FALLBACK] Attempting script execution for: {self.name}")
        try:
            result = self.execute_fallback(script_executor)
            gui_controller.logger.info(f"[FALLBACK] {'✅ Success' if result else '❌ Failed'} for: {self.name}")
            return result
        except Exception as e:
            gui_controller.logger.error(f"[FALLBACK] ❌ Script execution error for {self.name}: {e}")
            return False


class MarketDataStep(TradingStep):
    """Market data collection step"""
    
    def __init__(self):
        super().__init__("Market Data Collection", TradingPhase.MARKET_DATA, required=True)
        
    def execute_gui(self, gui_controller: GUIController) -> bool:
        """Try to click market data fetch button"""
        tab_names = [
            "💹 Market Data",
            "💹 Dữ liệu thị trường",
            "market data", 
            "market"
        ]
        if not gui_controller.switch_to_tab(tab_names):
            return False
            
        current_tab = gui_controller.main_window.tabWidget.currentWidget()
        keywords = [
            "Lấy dữ liệu ngay",
            "Fetch Data Now",
            "fetch data",
            "lấy dữ liệu"
        ]
        excluded = ["chart", "biểu đồ", "start", "stop"]
        
        success = gui_controller.find_and_click_button(current_tab, keywords, excluded)
        if success:
            # Calculate delay: 3s + 0.3s per symbol
            symbol_count = gui_controller.get_selected_symbols_count()
            delay_time = 3 + (0.3 * symbol_count)
            gui_controller.logger.info(f"⏳ Waiting {delay_time}s for market data fetching ({symbol_count} symbols)...")
            time.sleep(delay_time)
            gui_controller.logger.info("✅ Market data fetching completed")
        return success
        
    def execute_fallback(self, script_executor: ScriptExecutor) -> bool:
        """Fallback to script execution"""
        # 🔧 FIX: Use comprehensive_aggregator first with proper args parsing
        scripts_to_try = [
            ('comprehensive_aggregator.py', ['--limit', '1', '--verbose']),
            ('mt5_data_fetcher.py', [])
        ]
        
        for script_path, args in scripts_to_try:
            if script_executor.run_script(script_path, args):
                return True
                    
        return False


class TrendAnalysisStep(TradingStep):
    """Trend Analysis Step"""
    
    def __init__(self):
        super().__init__("Trend Analysis", TradingPhase.TREND_ANALYSIS, required=True)
        
    def execute_gui(self, gui_controller: GUIController) -> bool:
        """Try to click trend analysis button"""
        tab_names = [
            "📈 Trend Analysis", "📈 Phân tích xu hướng", 
            "trend analysis", "phân tích xu hướng", "trend"
        ]
        if not gui_controller.switch_to_tab(tab_names):
            return False
            
        current_tab = gui_controller.main_window.tabWidget.currentWidget()
        keywords = [
            "Calculate Trendline & SR", "Tính đường xu hướng & SR",
            "Calculate", "Tính toán", "Analyze", "Phân tích"
        ]
        excluded = ["stop", "dừng"]
        
        success = gui_controller.find_and_click_button(current_tab, keywords, excluded)
        if success:
            # Calculate delay: 3s + 0.3s per symbol (same as Market Data)
            symbol_count = gui_controller.get_selected_symbols_count()
            delay_time = 3 + (0.3 * symbol_count)
            gui_controller.logger.info(f"⏳ Waiting {delay_time}s for trend analysis ({symbol_count} symbols)...")
            time.sleep(delay_time)
            gui_controller.logger.info("✅ Trend analysis completed")
            return True
        else:
            # If no button found, still wait but shorter time
            symbol_count = gui_controller.get_selected_symbols_count()
            delay_time = 2 + (0.2 * symbol_count)
            gui_controller.logger.info(f"⏳ No button found, waiting {delay_time}s for auto-processing...")
            time.sleep(delay_time)
            return True
        
    def execute_fallback(self, script_executor: ScriptExecutor) -> bool:
        """Fallback: Run trend analysis directly"""
        scripts_to_try = [
            ('trendline_support_resistance.py', []),
        ]
        
        for script_path, args in scripts_to_try:
            if script_executor.run_script(script_path, args):
                return True
        return False


class IndicatorCalculationStep(TradingStep):
    """Indicator Calculation Step"""
    
    def __init__(self):
        super().__init__("Indicator Calculation", TradingPhase.INDICATORS, required=True)
        
    def execute_gui(self, gui_controller: GUIController) -> bool:
        """Try to click indicator calculation button"""
        tab_names = [
            "📊 Indicators", "📊 Chỉ báo kỹ thuật", 
            "indicators", "chỉ báo", "technical"
        ]
        if not gui_controller.switch_to_tab(tab_names):
            return False
            
        current_tab = gui_controller.main_window.tabWidget.currentWidget()
        keywords = [
            "Calculate & Save Indicator", "Tính & lưu chỉ báo",
            "Calculate All", "Tính tất cả", "Export", "Xuất"
        ]
        excluded = ["stop", "dừng"]
        
        success = gui_controller.find_and_click_button(current_tab, keywords, excluded)
        if success:
            # Calculate delay: 0.7s per symbol + 0.3s per timeframe
            symbol_count = gui_controller.get_selected_symbols_count()
            timeframe_count = gui_controller.get_selected_timeframes_count()
            delay_time = (0.7 * symbol_count) + (0.3 * timeframe_count)
            gui_controller.logger.info(f"⏳ Waiting {delay_time}s for indicator calculation ({symbol_count} symbols × {timeframe_count} timeframes)...")
            time.sleep(delay_time)
            gui_controller.logger.info("✅ Indicator calculation completed")
            return True
        else:
            # Shorter wait for auto-processing
            symbol_count = gui_controller.get_selected_symbols_count()
            timeframe_count = gui_controller.get_selected_timeframes_count()
            delay_time = (0.5 * symbol_count) + (0.2 * timeframe_count)
            gui_controller.logger.info(f"⏳ No button found, waiting {delay_time}s for auto-processing...")
            time.sleep(delay_time)
            return True
        
    def execute_fallback(self, script_executor: ScriptExecutor) -> bool:
        """Fallback: Run indicator calculation directly"""
        scripts_to_try = [
            ('mt5_indicator_exporter.py', []),
        ]
        
        for script_path, args in scripts_to_try:
            if script_executor.run_script(script_path, args):
                return True
        return False


class CandlestickPatternsStep(TradingStep):
    """Candlestick Patterns Detection Step"""
    
    def __init__(self):
        super().__init__("Candlestick Patterns", TradingPhase.CANDLESTICK_PATTERNS, required=True)
        
    def execute_gui(self, gui_controller: GUIController) -> bool:
        """Try to click candlestick pattern detection button"""
        tab_names = [
            "🕯️ Candlestick Patterns", "🕯️ Mô hình nến", 
            "🕯️ Mô hình nến", "candle patterns", "mô hình nến", "candlestick", "Candlestick"
        ]
        if not gui_controller.switch_to_tab(tab_names):
            return False
            
        current_tab = gui_controller.main_window.tabWidget.currentWidget()
        keywords = [
            "🔍 Fetch Candlestick Patterns", "🔍 Lấy mô hình nến",
            "Fetch Candlestick", "Lấy mô hình", "Detect Patterns", "Phát hiện mô hình", 
            "Analyze", "Phân tích", "Calculate", "Tính toán", "Start", "Bắt đầu", "Fetch"
        ]
        excluded = ["stop", "dừng", "price", "giá"]
        
        success = gui_controller.find_and_click_button(current_tab, keywords, excluded)
        if success:
            # Calculate delay: 0.7s per symbol + 0.3s per timeframe
            symbol_count = gui_controller.get_selected_symbols_count()
            timeframe_count = gui_controller.get_selected_timeframes_count()
            delay_time = (0.7 * symbol_count) + (0.3 * timeframe_count)
            gui_controller.logger.info(f"⏳ Waiting {delay_time}s for candle pattern detection ({symbol_count} symbols × {timeframe_count} timeframes)...")
            time.sleep(delay_time)
            gui_controller.logger.info("✅ Candle pattern detection completed")
            return True
        else:
            # Fallback wait for auto-processing
            symbol_count = gui_controller.get_selected_symbols_count()
            timeframe_count = gui_controller.get_selected_timeframes_count()
            delay_time = (0.5 * symbol_count) + (0.2 * timeframe_count)
            gui_controller.logger.info(f"⏳ No button found, waiting {delay_time}s for auto-processing...")
            time.sleep(delay_time)
            return True
        
    def execute_fallback(self, script_executor: ScriptExecutor) -> bool:
        """Fallback: Run pattern detection directly"""
        scripts_to_try = [
            ('pattern_detector.py', []),
        ]
        
        for script_path, args in scripts_to_try:
            if script_executor.run_script(script_path, args):
                return True
        return False


class PricePatternsStep(TradingStep):
    """Price Patterns Detection Step"""
    
    def __init__(self):
        super().__init__("Price Patterns", TradingPhase.PRICE_PATTERNS, required=True)
        
    def execute_gui(self, gui_controller: GUIController) -> bool:
        """Try to click price pattern detection button"""
        tab_names = [
            "� Price Patterns", "📊 Mô hình giá", "�📉 Price Patterns", "📉 Mô hình giá",
            "📊 Mô hình giá", "price patterns", "mô hình giá", "patterns", "Price", "Price Patterns"
        ]
        if not gui_controller.switch_to_tab(tab_names):
            return False
            
        current_tab = gui_controller.main_window.tabWidget.currentWidget()
        keywords = [
            "🔍 Fetch Price Patterns", "🔍 Lấy mô hình giá",
            "Fetch Price", "Lấy mô hình", "Analyze Price Patterns", "Phân tích mô hình giá", 
            "Detect", "Phát hiện", "Calculate", "Tính toán", "Fetch"
        ]
        excluded = ["stop", "dừng", "candlestick", "nến"]
        
        success = gui_controller.find_and_click_button(current_tab, keywords, excluded)
        if success:
            # Calculate delay: 0.7s per symbol + 0.3s per timeframe
            symbol_count = gui_controller.get_selected_symbols_count()
            timeframe_count = gui_controller.get_selected_timeframes_count()
            delay_time = (0.7 * symbol_count) + (0.3 * timeframe_count)
            gui_controller.logger.info(f"⏳ Waiting {delay_time}s for price pattern detection ({symbol_count} symbols × {timeframe_count} timeframes)...")
            time.sleep(delay_time)
            gui_controller.logger.info("✅ Price pattern detection completed")
            return True
        else:
            # Fallback wait for auto-processing
            symbol_count = gui_controller.get_selected_symbols_count()
            timeframe_count = gui_controller.get_selected_timeframes_count()
            delay_time = (0.5 * symbol_count) + (0.2 * timeframe_count)
            gui_controller.logger.info(f"⏳ No button found, waiting {delay_time}s for auto-processing...")
            time.sleep(delay_time)
            return True
        
    def execute_fallback(self, script_executor: ScriptExecutor) -> bool:
        """Fallback: Run price pattern detection directly"""
        scripts_to_try = [
            ('price_patterns_full_data.py', []),
        ]
        
        for script_path, args in scripts_to_try:
            if script_executor.run_script(script_path, args):
                return True
        return False


class SignalGenerationStep(TradingStep):
    """Signal generation step"""
    
    def __init__(self):
        super().__init__("Signal Generation", TradingPhase.SIGNAL_GENERATION, required=True)
        
    def execute_gui(self, gui_controller: GUIController) -> bool:
        """Try to click signal generation button"""
        tab_names = [
            "📡 Signal",
            "📡 Tín hiệu",
            "signal"
        ]
        if not gui_controller.switch_to_tab(tab_names):
            return False
            
        current_tab = gui_controller.main_window.tabWidget.currentWidget()
        keywords = [
            "▶️ Chạy tổng hợp",
            "▶️ Run Aggregator",
            "Chạy tổng hợp",
            "Run Aggregator",
            "run",
            "chạy"
        ]
        excluded = ["refresh", "làm mới"]
        
        success = gui_controller.find_and_click_button(current_tab, keywords, excluded)
        if success:
            # Calculate delay: 7s + 0.8s per symbol (signal generation takes longer)
            symbol_count = gui_controller.get_selected_symbols_count()
            delay_time = 7 + (0.8 * symbol_count)
            gui_controller.logger.info(f"⏳ Waiting {delay_time}s for signal generation ({symbol_count} symbols)...")
            time.sleep(delay_time)
            gui_controller.logger.info("✅ Signal generation completed")
            return True
        else:
            # Fallback wait for auto-processing
            symbol_count = gui_controller.get_selected_symbols_count()
            delay_time = 5 + (0.5 * symbol_count)
            gui_controller.logger.info(f"⏳ No button found, waiting {delay_time}s for auto-processing...")
            time.sleep(delay_time)
            return True
        
    def execute_fallback(self, script_executor: ScriptExecutor) -> bool:
        """Fallback to script execution"""
        aggregator_path = os.path.join(os.getcwd(), 'comprehensive_aggregator.py')
        if os.path.exists(aggregator_path):
            # Use enhanced arguments for auto trading
            args = ['--limit', '1', '--verbose', '--strict-indicators']
            return script_executor.run_script(aggregator_path, args)
        
        return False


class OrderExecutionStep(TradingStep):
    """Order execution step"""
    
    def __init__(self):
        super().__init__("Order Execution", TradingPhase.ORDER_EXECUTION, required=True)
        
    def execute_gui(self, gui_controller: GUIController) -> bool:
        """GUI execution delegates to script for reliability"""
        gui_controller.logger.info(f"[GUI] Order execution uses script mode for reliability")
        return False
        
    def execute_fallback(self, script_executor: ScriptExecutor) -> bool:
        """Execute order management tasks"""
        execute_actions_path = os.path.join(os.getcwd(), 'execute_actions.py')
        if os.path.exists(execute_actions_path):
            return script_executor.run_script(execute_actions_path, ['--auto'])
        else:
            script_executor.logger.error("❌ execute_actions.py not found")
            return False


class UnifiedAutoTradingSystem:
    """Unified Auto Trading System combining Controller + Manager"""
    
    def __init__(self, main_window_ref=None, update_interval=60, config: TradingConfig = None):
        # Core components
        self.main_window = main_window_ref
        self.config = config or TradingConfig()
        self.config.update_interval = update_interval
        self.logger = self._setup_logger()
        
        # Controllers
        # 🔧 FIX: Ensure ScriptExecutor uses correct working directory
        self.script_executor = ScriptExecutor(self.logger)
        self.gui_controller = GUIController(self.main_window, self.logger)
        self.system_controller = SystemController()
        
        # 🔧 FIX: Set working directory to trading bot directory
        if not os.getcwd().endswith('my_trading_bot'):
            trading_bot_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(trading_bot_dir)
            self.logger.info(f"🔧 Changed working directory to: {os.getcwd()}")
        
        # Signals for thread-safe communication
        self.signals = AutoTradingSignals()
        self._setup_signals()
        
        # State management
        self.is_running = False
        self.current_phase = TradingPhase.IDLE
        self.pipeline_thread = None
        self.cycle_count = 0
        
        # Timing
        self.next_large_cycle = None
        self.last_small_cycle = 0
        
        # Trading steps - COMPLETE 7-step pipeline
        self.steps = [
            MarketDataStep(),           # 1. Dữ liệu nến
            TrendAnalysisStep(),        # 2. Phân tích xu hướng  
            IndicatorCalculationStep(), # 3. Tính toán chỉ báo
            CandlestickPatternsStep(),  # 4. Mô hình nến
            PricePatternsStep(),        # 5. Mô hình giá
            SignalGenerationStep(),     # 6. Signal generation
            OrderExecutionStep()        # 7. Order execution
        ]
        
        self.logger.info("🚀 Unified Auto Trading System v5.0 initialized")
        self.logger.info("📋 Complete 7-step pipeline: Data→Trend→Indicators→Candle→Price→Signal→Orders")
        self.logger.info("🛡️ Enhanced duplicate prevention and safety controls active")
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with proper formatting"""
        logger = logging.getLogger(f"UnifiedAutoTrading_{id(self)}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def _setup_signals(self):
        """Connect signals for thread-safe GUI updates"""
        self.signals.status_updated.connect(self._update_gui_status)
        self.signals.log_added.connect(self._add_gui_log)
        self.signals.phase_changed.connect(self._update_gui_phase)
        
    def _update_gui_status(self, message: str):
        """Thread-safe GUI status update"""
        try:
            if self.main_window and hasattr(self.main_window, 'update_auto_trading_status'):
                self.main_window.update_auto_trading_status(message)
        except Exception as e:
            self.logger.error(f"❌ GUI status update error: {e}")
            
    def _add_gui_log(self, message: str):
        """Thread-safe GUI log update"""
        try:
            if self.main_window and hasattr(self.main_window, 'add_auto_trading_log'):
                self.main_window.add_auto_trading_log(message)
        except Exception as e:
            self.logger.error(f"❌ GUI log update error: {e}")
            
    def _update_gui_phase(self, phase: str):
        """Update current phase in GUI"""
        self.signals.log_added.emit(f"📍 Phase: {phase}")
        
    def update_status(self, message: str):
        """Update status with thread-safe signal"""
        self.signals.status_updated.emit(message)
        self.logger.info(f"📱 {message}")
        
    def add_log(self, message: str):
        """Add log with thread-safe signal"""
        self.signals.log_added.emit(message)
        
    def start(self) -> bool:
        """Start the unified auto trading system"""
        if self.is_running:
            self.logger.warning("🔄 Auto trading already running")
            return False
            
        try:
            # Safety check
            status = self.system_controller.get_current_status()
            if status['emergency_stop']:
                self.logger.error("❌ Cannot start: Emergency stop active")
                return False
            
            self.is_running = True
            self.cycle_count = 0
            self.current_phase = TradingPhase.IDLE
            
            # Start pipeline thread
            self.pipeline_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.pipeline_thread.start()
            
            self.update_status("🟢 Unified Auto Trading started")
            self.add_log("🟢 Auto Trading system activated with enhanced safety")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start auto trading: {e}")
            self.is_running = False
            return False
            
    def stop(self):
        """Stop the unified auto trading system"""
        if not self.is_running:
            return
            
        self.logger.info("🛑 Stopping unified auto trading...")
        self.is_running = False
        self.current_phase = TradingPhase.IDLE
        
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            self.pipeline_thread.join(timeout=5)
            
        self.update_status("⚪ Auto Trading stopped")
        self.add_log("🔴 Auto Trading system deactivated")
        
    def _main_loop(self):
        """Main trading loop with enhanced safety"""
        self.logger.info("🔄 Unified auto trading loop started")
        self.update_status("🚀 Auto Trading System STARTED - Running continuously...")
        self.add_log(f"💫 Auto Trading initialized with {self.config.large_cycle_minutes}-minute cycle intervals")
        self.add_log("🔄 System will run continuously until manually stopped")
        
        consecutive_errors = 0
        max_consecutive_errors = 3
        error_cooldown = 30  # seconds
        
        while self.is_running:
            try:
                # Safety check before each cycle
                status = self.system_controller.get_current_status()
                if status['emergency_stop']:
                    self.logger.warning("🚨 Emergency stop detected - pausing auto trading")
                    break
                
                # Run trading cycle
                if self._should_run_cycle():
                    success = self._run_trading_cycle()
                    
                    if success:
                        consecutive_errors = 0
                        # Show countdown to next cycle
                        next_cycle_in = int(self.next_large_cycle - time.time())
                        self.update_status(f"✅ Cycle #{self.cycle_count} completed successfully! Next cycle in {next_cycle_in}s")
                        self.add_log(f"🔄 Auto Trading continues... Next cycle #{self.cycle_count + 1} in {next_cycle_in} seconds")
                        self.add_log(f"📊 System Status: Running continuously with {self.config.large_cycle_minutes}-minute intervals")
                    else:
                        consecutive_errors += 1
                        
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error(f"❌ {consecutive_errors} consecutive errors - cooling down for {error_cooldown}s")
                        time.sleep(error_cooldown)
                        consecutive_errors = 0
                else:
                    # Show countdown while waiting for next cycle
                    if self.next_large_cycle:
                        remaining = int(self.next_large_cycle - time.time())
                        if remaining > 0:
                            # Update countdown every 30 seconds to show activity
                            if remaining % 30 == 0 or remaining <= 10:
                                self.update_status(f"⏳ Auto Trading running... Next cycle in {remaining}s")
                                self.add_log(f"🕐 Cycle #{self.cycle_count + 1} starts in {remaining} seconds")
                
                time.sleep(5)  # Basic loop delay
                
            except Exception as e:
                self.logger.error(f"❌ Main loop error: {e}")
                consecutive_errors += 1
                time.sleep(10)
                
        self.logger.info("🔄 Unified auto trading loop ended")
        
    def _should_run_cycle(self) -> bool:
        """Check if it's time to run a cycle"""
        now = time.time()
        
        # First run
        if self.next_large_cycle is None:
            self.next_large_cycle = now + (self.config.large_cycle_minutes * 60)
            return True
        
        # Check if it's time for next cycle
        return now >= self.next_large_cycle
        
    def _run_trading_cycle(self):
        """Run complete trading cycle"""
        start_time = time.time()
        self.cycle_count += 1
        
        self.update_status(f"🔄 Cycle #{self.cycle_count} starting...")
        self.add_log(f"� Starting Cycle #{self.cycle_count} - Running complete pipeline (3 steps: Data → Signals → Orders)")
        self.add_log(f"🕐 Cycle interval: {self.config.large_cycle_minutes} minutes | Auto Trading: ACTIVE")
        
        success = self._execute_pipeline()
        
        # Schedule next cycle
        self.next_large_cycle = time.time() + (self.config.large_cycle_minutes * 60)
        
        duration = time.time() - start_time
        
        if success:
            self.add_log(f"✅ Cycle #{self.cycle_count} completed in {duration:.1f}s")
        else:
            self.add_log(f"❌ Cycle #{self.cycle_count} failed after {duration:.1f}s")
            
        return success
        
    def _execute_pipeline(self) -> bool:
        """Execute simplified trading pipeline"""
        try:
            total_steps = len(self.steps)
            success_count = 0
            
            for i, step in enumerate(self.steps, 1):
                if not self.is_running:
                    break
                
                self.current_phase = step.phase
                self.signals.phase_changed.emit(step.phase.value)
                
                step_name_vi = self._get_step_name_vi(step.name)
                self.add_log(f"🔧 [{i}/{total_steps}] {step_name_vi}...")
                
                step_success = step.execute(self.gui_controller, self.script_executor)
                
                if step_success:
                    success_count += 1
                    self.add_log(f"✅ [{i}/{total_steps}] {step_name_vi} hoàn thành")
                else:
                    if step.required:
                        self.add_log(f"❌ [{i}/{total_steps}] {step_name_vi} thất bại (bắt buộc)")
                        self.logger.error(f"❌ Required step failed: {step.name}")
                        return False
                    else:
                        self.add_log(f"⚠️ [{i}/{total_steps}] {step_name_vi} thất bại (tùy chọn)")
            
            success_rate = success_count / total_steps
            self.logger.info(f"📊 Pipeline completed: {success_count}/{total_steps} steps successful ({success_rate*100:.1f}%)")
            self.add_log(f"📈 Cycle #{self.cycle_count} Results: {success_count}/{total_steps} steps successful ({success_rate*100:.1f}%)")
            self.add_log(f"🔄 Auto Trading continues running... Waiting for next cycle")
            
            return success_rate >= 0.6  # At least 60% success rate
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution error: {e}")
            self.current_phase = TradingPhase.ERROR
            return False
            
    def _get_step_name_vi(self, step_name: str) -> str:
        """Get Vietnamese step name"""
        translations = {
            "Market Data Collection": "Thu thập dữ liệu thị trường",
            "Signal Generation": "Tạo tín hiệu giao dịch",
            "Order Execution": "Thực hiện lệnh giao dịch"
        }
        return translations.get(step_name, step_name)
        
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        base_status = self.system_controller.get_current_status()
        base_status.update({
            'is_running': self.is_running,
            'current_phase': self.current_phase.value,
            'cycle_count': self.cycle_count,
            'next_large_cycle': self.next_large_cycle,
            'config': self.config.to_dict()
        })
        return base_status


# CLI Interface
def main():
    import sys
    
    if len(sys.argv) < 2:
        print("🛡️ UNIFIED AUTO TRADING SYSTEM")
        print("Usage:")
        print("  python unified_auto_trading_system.py status    - Show current status")
        print("  python unified_auto_trading_system.py enable    - Enable auto trading safely")
        print("  python unified_auto_trading_system.py disable   - Disable auto trading")
        print("  python unified_auto_trading_system.py monitor   - Monitor continuously")
        print("  python unified_auto_trading_system.py start     - Start trading pipeline")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'status':
        SystemController.show_status()
    elif command == 'enable':
        SystemController.safe_enable_auto_trading()
    elif command == 'disable':
        SystemController.disable_auto_trading()
    elif command == 'monitor':
        SystemController.monitor_mode()
    elif command == 'start':
        # Start the trading system
        system = UnifiedAutoTradingSystem()
        if system.start():
            print("✅ Trading system started")
            try:
                while system.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping system...")
                system.stop()
        else:
            print("❌ Failed to start trading system")
    else:
        print(f"❌ Unknown command: {command}")


# Backward compatibility aliases
AutoTradingManager = UnifiedAutoTradingSystem
SimpleAutoTradingManager = UnifiedAutoTradingSystem
AutoTradingController = SystemController

if __name__ == "__main__":
    main()