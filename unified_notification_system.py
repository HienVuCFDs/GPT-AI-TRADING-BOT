"""
Unified Notification System for Trading Bot
Handles notifications to multiple platforms: Telegram, Discord, Email, etc.
"""

import json
import logging
import requests
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from pathlib import Path
import traceback


class TelegramAdapter:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def send_message(self, message: str) -> bool:
        try:
            if not self.config.get('enabled', False):
                self.logger.debug("Telegram adapter disabled")
                return False
                
            bot_token = self.config.get('bot_token')
            chat_id = self.config.get('chat_id')
            
            if not bot_token or not chat_id:
                self.logger.error("Telegram bot_token or chat_id missing")
                return False
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            # Clean message for Telegram
            clean_message = self._sanitize_telegram_message(message)
            
            payload = {
                'chat_id': chat_id,
                'text': clean_message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, data=payload, timeout=10)
            
            if response.status_code == 200:
                self.logger.info("✅ Telegram message sent successfully")
                return True
            else:
                self.logger.error(f"❌ Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Telegram send error: {e}")
            return False
    
    def _format_message_for_platform(self, message: str) -> str:
        """Format message for Telegram (HTML formatting)"""
        # Replace markdown-style formatting with HTML
        formatted = message.replace('**', '<b>').replace('**', '</b>')
        formatted = formatted.replace('*', '<i>').replace('*', '</i>')
        
        # Limit message length
        if len(formatted) > 4096:
            formatted = formatted[:4090] + "..."
            
        return formatted
    
    def _sanitize_telegram_message(self, message: str) -> str:
        """Sanitize message to avoid Telegram HTML parsing errors"""
        import re
        
        # Replace < and > with safe alternatives to avoid HTML tag parsing issues
        sanitized = message.replace('<', '‹').replace('>', '›')
        
        # Replace other problematic characters that might cause parsing issues
        # Keep emojis but escape HTML-sensitive characters
        sanitized = sanitized.replace('&', '&amp;')
        
        # Fix double arrows that might be interpreted as HTML
        sanitized = re.sub(r'→\s*([A-Z0-9]+)\s*<\s*([A-Z0-9]+)\s*<\s*([A-Z0-9]+)', 
                          r'→ \1 ‹ \2 ‹ \3', sanitized)
        
        # Fix comparison operators in EMA descriptions
        sanitized = re.sub(r'([A-Z0-9]+)\s*<\s*([A-Z0-9]+)\s*<\s*([A-Z0-9]+)', 
                          r'\1 ‹ \2 ‹ \3', sanitized)
        
        # Fix <= operators
        sanitized = sanitized.replace('<=', '≤').replace('>=', '≥')
        
        # Limit message length for Telegram
        if len(sanitized) > 4096:
            sanitized = sanitized[:4090] + "..."
        
        self.logger.info(f"🧹 Sanitized message length: {len(message)} -> {len(sanitized)}")
        return sanitized
    
    def validate_config(self) -> bool:
        return bool(self.config.get('bot_token') and self.config.get('chat_id'))


class DiscordAdapter:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def send_message(self, message: str) -> bool:
        try:
            if not self.config.get('enabled', False):
                self.logger.debug("Discord adapter disabled")
                return False
                
            webhook_url = self.config.get('webhook_url')
            
            if not webhook_url:
                self.logger.error("Discord webhook_url missing")
                return False
            
            # Format for Discord
            clean_message = message[:2000]  # Discord limit
            
            payload = {
                'content': clean_message
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            
            if response.status_code == 204:
                self.logger.info("✅ Discord message sent successfully")
                return True
            else:
                self.logger.error(f"❌ Discord webhook error: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Discord send error: {e}")
            return False
    
    def validate_config(self) -> bool:
        return bool(self.config.get('webhook_url'))


class EmailAdapter:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def send_message(self, message: str) -> bool:
        try:
            if not self.config.get('enabled', False):
                self.logger.debug("Email adapter disabled")
                return False
            
            smtp_server = self.config.get('smtp_server')
            port = self.config.get('port', 587)
            email = self.config.get('email')
            password = self.config.get('password') 
            to_email = self.config.get('to_email')
            
            if not all([smtp_server, email, password, to_email]):
                self.logger.error("Email configuration incomplete")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email
            msg['To'] = to_email
            msg['Subject'] = "Trading Signal Notification"
            
            msg.attach(MIMEText(message, 'plain', 'utf-8'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, port)
            server.starttls()
            server.login(email, password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info("✅ Email sent successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Email send error: {e}")
            return False
    
    def validate_config(self) -> bool:
        required = ['smtp_server', 'email', 'password', 'to_email']
        return all(self.config.get(key) for key in required)


class SmartNotificationSystem:
    """Smart routing system for multiple notification platforms"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        """Initialize all notification platform adapters"""
        self.adapters = {}
        
        try:
            # Initialize Telegram
            if 'telegram' in self.config:
                self.adapters['telegram'] = TelegramAdapter(self.config['telegram'])
                
            # Initialize Discord  
            if 'discord' in self.config:
                self.adapters['discord'] = DiscordAdapter(self.config['discord'])
                
            # Initialize Email
            if 'email' in self.config:
                self.adapters['email'] = EmailAdapter(self.config['email'])
            
            enabled_count = sum(1 for adapter in self.adapters.values() 
                              if hasattr(adapter, 'config') and adapter.config.get('enabled', False))
            self.logger.info(f"🔧 Initialized {len(self.adapters)} adapters, {enabled_count} enabled")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing adapters: {e}")
    
    def send_to_all_platforms(self, message: str) -> Dict[str, bool]:
        """Send message to all enabled platforms"""
        results = {}
        
        for platform, adapter in self.adapters.items():
            try:
                if adapter.config.get('enabled', False):
                    results[platform] = adapter.send_message(message)
                else:
                    results[platform] = False
                    self.logger.debug(f"Platform {platform} disabled, skipping")
            except Exception as e:
                self.logger.error(f"❌ Error sending to {platform}: {e}")
                results[platform] = False
        
        return results
    
    def send_to_platform(self, platform: str, message: str) -> bool:
        """Send message to specific platform"""
        if platform not in self.adapters:
            self.logger.error(f"❌ Platform {platform} not available")
            return False
            
        return self.adapters[platform].send_message(message)
    
    def get_available_platforms(self) -> List[str]:
        return list(self.adapters.keys())
    
    def validate_all_configurations(self) -> Dict[str, bool]:
        """Validate all platform configurations"""
        validation_results = {}
        
        for platform, adapter in self.adapters.items():
            try:
                validation_results[platform] = adapter.validate_config()
            except Exception as e:
                self.logger.error(f"❌ Validation error for {platform}: {e}")
                validation_results[platform] = False
        
        return validation_results


class UnifiedNotificationSystem:
    """Main notification system with signal and execution message handling"""
    
    def __init__(self, config_file: str = "notification_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.logger = self._setup_logger()
        
        # Initialize smart notification system
        self.smart_notification = SmartNotificationSystem(self.config)
        
        # Initialize the restored NotificationService
        try:
            from notification_service import NotificationService
            self.notification_service = NotificationService(config_file)
            self.logger.info("✅ Restored NotificationService successfully loaded")
        except Exception as e:
            self.logger.error(f"❌ Failed to load NotificationService: {e}")
            self.notification_service = None
        
        # Initialize order tracking daemon (will be started when monitoring is enabled)
        self.tracking_daemon = None
        self.monitoring_enabled = False
        
        self.logger.info("🚀 UnifiedNotificationSystem initialized successfully")
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config
            else:
                # Create default config
                default_config = {
                    "telegram": {"enabled": False, "bot_token": "", "chat_id": ""},
                    "discord": {"enabled": False, "webhook_url": ""},
                    "email": {"enabled": False, "smtp_server": "", "port": 587, "email": "", "password": "", "to_email": ""},
                    "settings": {
                        "send_signals": False,
                        "send_reports": False,
                        "send_execution_results": True,
                        "daily_summary": True,
                        "format_vietnamese": True,
                        "max_message_length": 4000,
                        "track_order_updates": True,
                        "notify_sl_tp_changes": True,
                        "notify_order_close": True,
                        "update_interval_seconds": 30,
                        "custom_message": "",
                        "notification_format": "full",
                        "include_technical": True,
                        "include_indicators": True,
                        "include_summary": True,
                        "include_candlestick": True,
                        "include_price_patterns": True,
                        "send_custom_message": False,
                        "include_technical_analysis": True
                    }
                }
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                    
                return default_config
                
        except Exception as e:
            self.logger.error(f"❌ Error loading config: {e}")
            return {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for notification system"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def send_real_trade_notification(self, message: str) -> bool:
        """Send notification for real executed trades"""
        try:
            # Check if execution notifications are enabled
            if not self.config.get("settings", {}).get("send_execution_results", True):
                self.logger.info("Execution notifications disabled in settings")
                return False
            
            # Send to all platforms
            results = self.smart_notification.send_to_all_platforms(message)
            
            success = any(results.values())
            successful_count = sum(results.values())
            total_platforms = len([p for p, adapter in self.smart_notification.adapters.items() 
                                 if adapter.config.get('enabled', False)])
            
            if success:
                self.logger.info(f"✅ Trade notification sent to {successful_count}/{total_platforms} platform(s)")
            else:
                self.logger.warning(f"❌ Trade notification failed - no platforms available or all failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error sending trade notification: {e}")
            return False
    
    def send_execution_notification(self, message: str) -> bool:
        """Send execution notification - alias for send_real_trade_notification"""
        return self.send_real_trade_notification(message)
    
    def send_notification(self, message: str) -> bool:
        """Send general notification to all enabled platforms"""
        try:
            # Send to all platforms using smart notification system
            results = self.smart_notification.send_to_all_platforms(message)
            
            success = any(results.values())
            successful_count = sum(results.values())
            total_platforms = len([p for p, adapter in self.smart_notification.adapters.items() 
                                 if adapter.config.get('enabled', False)])
            
            if success:
                self.logger.info(f"✅ Notification sent to {successful_count}/{total_platforms} platform(s)")
            else:
                self.logger.warning(f"❌ Notification failed - no platforms available or all failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error sending notification: {e}")
            return False
    
    def send_telegram_message(self, message: str) -> bool:
        """Send message to Telegram only - alias for send_notification for backward compatibility"""
        return self.send_notification(message)
    
    def _save_config(self) -> bool:
        """Save current config to file"""
        try:
            config_path = Path(self.config_file)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.logger.info("✅ Config saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error saving config: {e}")
            return False
    
    @property
    def monitoring_enabled(self) -> bool:
        """Check if monitoring is enabled"""
        return self.config.get("settings", {}).get("track_order_updates", True)
    
    @monitoring_enabled.setter
    def monitoring_enabled(self, value: bool):
        """Set monitoring enabled state"""
        if "settings" not in self.config:
            self.config["settings"] = {}
        self.config["settings"]["track_order_updates"] = value

    def update_tracking_settings(self, track_orders=None, sl_tp_changes=None, order_close=None):
        """Update tracking settings"""
        try:
            if "settings" not in self.config:
                self.config["settings"] = {}
            
            if track_orders is not None:
                self.config["settings"]["track_order_updates"] = track_orders
            if sl_tp_changes is not None:
                self.config["settings"]["notify_sl_tp_changes"] = sl_tp_changes
            if order_close is not None:
                self.config["settings"]["notify_order_close"] = order_close
                
            # Save config
            return self._save_config()
        except Exception as e:
            self.logger.error(f"Error updating tracking settings: {e}")
            return False

    def start_monitoring(self):
        """Start order monitoring using SimpleOrderTracker (direct MT5 connection)"""
        try:
            from simple_order_tracker import get_tracker
            
            # Get tracker instance
            tracker = get_tracker()
            
            # Check if already running
            if tracker.is_running():
                self.logger.info("✅ Order tracker already running")
                self.monitoring_enabled = True
                self._save_config()
                return True
            
            # Start tracker (will connect to MT5 automatically)
            if tracker.start():
                self.logger.info("🚀 Simple Order Tracker started - Direct MT5 connection")
                self.monitoring_enabled = True
                self.tracking_daemon = tracker
                self._save_config()
                return True
            else:
                self.logger.error("❌ Failed to start Simple Order Tracker")
                self.logger.error("   Make sure MT5 is running and logged in")
                return False
            
        except ImportError as e:
            self.logger.error(f"❌ SimpleOrderTracker not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error starting monitoring: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def stop_monitoring(self):
        """Stop order monitoring"""
        try:
            # Stop Simple Order Tracker if running
            if hasattr(self, 'tracking_daemon') and self.tracking_daemon:
                if hasattr(self.tracking_daemon, 'stop'):
                    self.tracking_daemon.stop()
                    self.logger.info("🛑 Simple Order Tracker stopped")
            
            self.monitoring_enabled = False
            self._save_config()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping monitoring: {e}")
            return False

    def send_order_update_notification(self, order_data):
        """Send notification for order updates with DCA/Entry DISTINCTION"""
        try:
            if not self.config.get("settings", {}).get("track_order_updates", False):
                return False
            
            # Use restored NotificationService if available
            if self.notification_service:
                # Convert single order to list for grouping format
                orders = [order_data] if isinstance(order_data, dict) else order_data
                return self.notification_service.send_order_update_notification(orders)
            
            # Fallback to simple logging
            symbol = order_data.get('symbol', 'N/A') if isinstance(order_data, dict) else 'Multiple'
            self.logger.info(f"📈 Order update tracked: {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error tracking order update: {e}")
            return False
    
    def send_dca_notification(self, dca_orders):
        """Send dedicated DCA group notification"""
        try:
            if not self.config.get("settings", {}).get("track_order_updates", False):
                return False
            
            # Use restored NotificationService for DCA notifications
            if self.notification_service:
                orders = [dca_orders] if isinstance(dca_orders, dict) else dca_orders
                return self.notification_service.send_dca_group_notification(orders)
            
            # Fallback logging
            count = len(dca_orders) if isinstance(dca_orders, list) else 1
            self.logger.info(f"🔄 DCA group tracked: {count} orders")
            return True
            
        except Exception as e:
            self.logger.error(f"Error tracking DCA group: {e}")
            return False
    
    def send_entry_notification(self, entry_orders):
        """Send dedicated Entry orders notification"""
        try:
            if not self.config.get("settings", {}).get("track_order_updates", False):
                return False
            
            # Use restored NotificationService for Entry notifications
            if self.notification_service:
                orders = [entry_orders] if isinstance(entry_orders, dict) else entry_orders
                return self.notification_service.send_entry_orders_notification(orders)
            
            # Fallback logging
            count = len(entry_orders) if isinstance(entry_orders, list) else 1
            self.logger.info(f"🎯 Entry group tracked: {count} orders")
            return True
            
        except Exception as e:
            self.logger.error(f"Error tracking Entry group: {e}")
            return False

    def send_sl_tp_change_notification(self, order_data):
        """Send notification for SL/TP changes with detailed tracking"""
        try:
            if not self.config.get("settings", {}).get("notify_sl_tp_changes", False):
                return False
            
            # Use restored NotificationService for SL/TP notifications
            if self.notification_service:
                # Create SL/TP specific message
                orders = [order_data] if isinstance(order_data, dict) else order_data
                return self.notification_service.send_sl_tp_notification(orders)
                
            # Fallback to simple logging
            symbol = order_data.get('symbol', 'N/A')
            self.logger.info(f"🛡️ SL/TP change tracked: {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error tracking SL/TP change: {e}")
            return False

    def send_order_close_notification(self, order_data):
        """Send notification for order close with detailed final P&L"""
        try:
            if not self.config.get("settings", {}).get("notify_order_close", False):
                return False
            
            # Use restored NotificationService for order close notifications
            if self.notification_service:
                orders = [order_data] if isinstance(order_data, dict) else order_data
                return self.notification_service.send_order_close_notification(orders)
                
            # Fallback to simple logging
            symbol = order_data.get('symbol', 'N/A')
            self.logger.info(f"🔒 Order close tracked: {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error tracking order close: {e}")
            return False




# Global instance for backward compatibility
_notification_system = None

def get_unified_notification_system():
    """Get singleton notification system instance"""
    global _notification_system
    if _notification_system is None:
        _notification_system = UnifiedNotificationSystem()
    return _notification_system


# Deprecated functions removed - only execution notifications are supported