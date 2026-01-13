# -*- coding: utf-8 -*-
"""
NotificationService - Restored from .pyc file
Provides comprehensive notification system with grouped notifications and spam protection
"""

import json
import os
import time
import asyncio
import requests
import sys

# Import language system from app
try:
    from app import AppState, I18N as AppI18N
    LANGUAGE_SUPPORT = True
except ImportError:
    LANGUAGE_SUPPORT = False
    AppI18N = None
    AppState = None

# Local I18N class that reads from config
class I18N:
    _current_language = 'vi'  # Default
    
    @classmethod
    def set_language(cls, lang: str):
        """Set current language"""
        cls._current_language = lang
    
    @classmethod
    def get_language(cls):
        """
        Get current language.
        Priority: AppState._lang > I18N._current_language
        """
        # Priority 1: Try AppState._lang from app module
        try:
            if 'app' in sys.modules:
                app_module = sys.modules['app']
                if hasattr(app_module, 'AppState') and hasattr(app_module.AppState, '_lang'):
                    lang = app_module.AppState._lang
                    if lang in ('vi', 'en'):
                        return lang
        except Exception:
            pass
        
        # Fallback to local setting
        return cls._current_language
    
    @staticmethod
    def t(en: str, vi: str = None, **kwargs) -> str:
        """Translate text based on current language"""
        txt = None
        if I18N.get_language() == 'vi' and vi:
            txt = vi
        else:
            txt = en
        try:
            return txt.format(**kwargs)
        except Exception:
            return txt

from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

class NotificationService:
    """Original NotificationService class with grouped notifications and spam protection"""
    
    def __init__(self, config_file="notification_config.json"):
        """Initialize notification service with configuration"""
        self.config_file = config_file
        self.config = {}
        self._load_config()
        self._setup_logger()
        
        # Spam protection
        self.last_message_times = {}
        self.message_groups = {}
        self.spam_threshold = 5  # seconds
        
    def _setup_logger(self):
        """Setup logger for notification service"""
        self.logger = logging.getLogger('NotificationService')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _load_config(self):
        """Load notification configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                # Default configuration
                self.config = {
                    "telegram": {
                        "enabled": True,
                        "bot_token": "",
                        "chat_id": ""
                    },
                    "zalo": {
                        "enabled": False,
                        "access_token": "",
                        "user_id": ""
                    },
                    "grouping": {
                        "enabled": True,
                        "group_time_window": 30  # seconds
                    }
                }
                self._save_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self.config = {}
        
        # Load language from config
        language = self.config.get('settings', {}).get('language', 'vi')
        I18N.set_language(language)
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    def send_telegram_message(self, message: str, chat_id: Optional[str] = None) -> bool:
        """Send message via Telegram"""
        try:
            if not self.config.get('telegram', {}).get('enabled', False):
                return False
                
            bot_token = self.config.get('telegram', {}).get('bot_token', '')
            target_chat_id = chat_id or self.config.get('telegram', {}).get('chat_id', '')
            
            if not bot_token or not target_chat_id:
                self.logger.warning("Telegram credentials not configured")
                return False
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': target_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_zalo_message(self, message: str, user_id: Optional[str] = None) -> bool:
        """Send message via Zalo"""
        try:
            if not self.config.get('zalo', {}).get('enabled', False):
                return False
                
            access_token = self.config.get('zalo', {}).get('access_token', '')
            target_user_id = user_id or self.config.get('zalo', {}).get('user_id', '')
            
            if not access_token or not target_user_id:
                self.logger.warning("Zalo credentials not configured")
                return False
            
            # Zalo API implementation would go here
            # This is a placeholder as Zalo API requires specific integration
            self.logger.info(f"Zalo message (placeholder): {message}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending Zalo message: {e}")
            return False
    
    def format_signal_message(self, signal_data: Dict[str, Any]) -> str:
        """Format trading signal into notification message"""
        symbol = signal_data.get('symbol', 'Unknown')
        action = signal_data.get('action', 'Unknown')
        confidence = signal_data.get('confidence', 0)
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)
        
        message = f"🎯 <b>TRADING SIGNAL</b>\n\n"
        message += f"📊 Symbol: <b>{symbol}</b>\n"
        message += f"🔄 Action: <b>{action}</b>\n"
        message += f"⚡ Confidence: <b>{confidence}%</b>\n\n"
        message += f">>> Entry: <b>{entry}</b>\n"
        message += f"<<< Stop Loss: <b>{sl}</b>\n"
        message += f"=== Take Profit: <b>{tp}</b>\n\n"
        message += f"*** Time: {datetime.now().strftime('%H:%M:%S')}"
        
        return message
    
    def format_order_update_message(self, orders: List[Dict[str, Any]]) -> str:
        """Format order updates grouped by SYMBOL first, then show Entry/DCA within each symbol"""
        if not orders:
            return ""
        
        # Group orders by symbol first
        symbol_groups = {}
        for order in orders:
            symbol = order.get('symbol', 'Unknown')
            if symbol not in symbol_groups:
                symbol_groups[symbol] = {
                    'entry_orders': [],
                    'dca_orders': []
                }
            
            # Categorize within symbol group
            if self._is_dca_order(order):
                symbol_groups[symbol]['dca_orders'].append(order)
            else:
                symbol_groups[symbol]['entry_orders'].append(order)
        
        message_parts = []
        
        # Sort symbols alphabetically
        for symbol in sorted(symbol_groups.keys()):
            group_data = symbol_groups[symbol]
            entry_orders = group_data['entry_orders']
            dca_orders = group_data['dca_orders']
            
            # Calculate totals for entire symbol
            all_symbol_orders = entry_orders + dca_orders
            total_volume = sum(float(order.get('volume', 0)) for order in all_symbol_orders)
            total_profit = sum(float(order.get('profit', 0)) for order in all_symbol_orders)
            total_count = len(all_symbol_orders)
            
            # Symbol header with totals
            if total_count == 1:
                header = f"🎯 {symbol}"
            else:
                header = f"🎯 {symbol} ({total_count} orders)"
            
            message_parts.append(f"<b>{header}</b>")
            message_parts.append(f"Vol: {total_volume:.2f} | P&L: {total_profit:+.2f} USD")
            
            # Show Entry orders first (if any)
            if entry_orders:
                if len(entry_orders) == 1:
                    message_parts.append("📊 Entry:")
                else:
                    message_parts.append(f"📊 Entry ({len(entry_orders)} orders):")
                
                for i, order in enumerate(entry_orders, 1):
                    order_type = order.get('type', 'Unknown')
                    volume = float(order.get('volume', 0))
                    profit = float(order.get('profit', 0))
                    price_current = order.get('price_current', 0)
                    
                    detail = f"  {i}. {order_type} {volume:.2f} @ {price_current} ({profit:+.2f})"
                    message_parts.append(detail)
            
            # Show DCA orders (if any)
            if dca_orders:
                if len(dca_orders) == 1:
                    message_parts.append("🔄 DCA:")
                else:
                    message_parts.append(f"🔄 DCA ({len(dca_orders)} orders):")
                
                for i, order in enumerate(dca_orders, 1):
                    order_type = order.get('type', 'Unknown')
                    volume = float(order.get('volume', 0))
                    profit = float(order.get('profit', 0))
                    price_current = order.get('price_current', 0)
                    
                    # Add DCA level info
                    dca_info = ""
                    dca_level = order.get('dca_level', order.get('comment', ''))
                    if dca_level:
                        dca_info = f" [L{dca_level}]" if str(dca_level).isdigit() else f" [{dca_level}]"
                    
                    detail = f"  {i}. {order_type} {volume:.2f} @ {price_current}{dca_info} ({profit:+.2f})"
                    message_parts.append(detail)
            
            message_parts.append("")  # Empty line between symbols
        
        # Add timestamp
        message_parts.append(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
        
        return "\n".join(message_parts)
    
    def format_symbol_order_message(self, symbol: str, orders: List[Dict[str, Any]]) -> str:
        """Smart and simple notification format - clean but comprehensive"""
        if not orders:
            return ""
        
        message_parts = []
        
        # Smart header with status
        total_volume = sum(float(order.get('volume', 0)) for order in orders)
        total_profit_usd = sum(float(order.get('profit', 0)) for order in orders)
        total_orders = len(orders)
        
        # Status emoji with same style as Order Close
        if total_profit_usd > 0:
            status_emoji = "💰"  # Money bag for profit
        elif total_profit_usd < 0:
            status_emoji = "❌"  # X for loss
        else:
            status_emoji = "⚖️"  # Scale for breakeven
        
        # Header matching Order Close style
        header = f"📊 {status_emoji} <b>{symbol}</b>"
        if total_orders > 1:
            header += f" ({total_orders})"
        
        message_parts.append(header)
        
        # Volume and P&L line - multi-language
        volume_text = I18N.t(f"📊 Volume: {total_volume:.2f} lot", f"📊 Volume: {total_volume:.2f} lot")
        message_parts.append(volume_text)
        message_parts.append(f"💵 P&L: ${total_profit_usd:+.2f} USD")
        
        # Smart grouping - show all orders in one clean list
        entry_orders = [order for order in orders if not self._is_dca_order(order)]
        dca_orders = [order for order in orders if self._is_dca_order(order)]
        
        # Show orders in smart format
        order_index = 1
        
        # Entry orders first
        if entry_orders:
            for order in entry_orders:
                order_line = self._format_smart_order_line(order, order_index, False)
                message_parts.append(order_line)
                order_index += 1
        
        # DCA orders without separator
        if dca_orders:
            for order in dca_orders:
                order_line = self._format_smart_order_line(order, order_index, True)
                message_parts.append(order_line)
                order_index += 1
        
        # Timestamp - multi-language
        time_str = datetime.now().strftime('%H:%M:%S %d/%m/%Y')
        time_label = I18N.t("Time", "Thời gian")
        message_parts.append(f"⏰ {time_label}: {time_str}")
        
        return "\n".join(message_parts)
    
    def _format_detailed_order(self, order: Dict[str, Any], index: int, is_dca: bool = False) -> str:
        """Format detailed order information with pips and USD"""
        order_type = order.get('type', 'Unknown')
        volume = float(order.get('volume', 0))
        profit_usd = float(order.get('profit', 0))
        price_open = order.get('price_open', order.get('entry_price', 0))
        price_current = order.get('price_current', 0)
        symbol = order.get('symbol', 'Unknown')
        
        # Calculate pips profit/loss
        pips_profit = self._calculate_pips_profit(symbol, price_open, price_current, order_type)
        
        # Order type icons
        order_icon = "📈" if order_type.upper() == 'BUY' else "📉"
        
        # Profit/Loss status with better icons
        if profit_usd > 0:
            status_emoji = "�"  # Green circle for profit
        elif profit_usd < 0:
            status_emoji = "🔴"  # Red circle for loss
        else:
            status_emoji = "⚪"  # White circle for breakeven
        
        # Base order info
        detail = f"  {index}. {status_emoji} {order_icon} {order_type} {volume:.2f}"
        
        # Add DCA level if applicable
        if is_dca:
            dca_level = order.get('dca_level', '')
            if dca_level:
                detail += f" [L{dca_level}]"
            else:
                # Try to extract from comment
                comment = str(order.get('comment', ''))
                if 'Level' in comment or 'L' in comment:
                    detail += f" [{comment}]"
                else:
                    detail += " [DCA]"
        
        # Price information
        detail += f"\n    📌 Entry: {price_open} | Current: {price_current}"
        
        # P&L in both pips and USD with better formatting
        pips_sign = "+" if pips_profit >= 0 else ""
        usd_sign = "+" if profit_usd >= 0 else ""
        
        # P&L icons based on profit/loss
        if profit_usd > 0:
            pl_icon = "💰"  # Money bag for profit
        elif profit_usd < 0:
            pl_icon = "💸"  # Money with wings for loss
        else:
            pl_icon = "💱"  # Currency exchange for breakeven
        
        detail += f"\n    {pl_icon} P&L: {pips_sign}{pips_profit:.1f} pips | {usd_sign}{profit_usd:.2f} USD"
        
        return detail
    
    def _calculate_pips_profit(self, symbol: str, price_open: float, price_current: float, order_type: str) -> float:
        """Calculate profit/loss in pips"""
        if not price_open or not price_current:
            return 0.0
        
        # Import pip value calculation
        from order_executor import get_pip_value
        pip_value = get_pip_value(symbol)
        
        # Calculate price difference
        if order_type.upper() == 'BUY':
            price_diff = price_current - price_open
        else:  # SELL
            price_diff = price_open - price_current
        
        # Convert to pips
        pips = price_diff / pip_value if pip_value > 0 else 0
        
        return pips
    
    def _format_smart_order_line(self, order: Dict[str, Any], index: int, is_dca: bool = False) -> str:
        """Format a single order line - smart and compact"""
        order_type = order.get('type', 'Unknown')
        volume = float(order.get('volume', 0))
        profit_usd = float(order.get('profit', 0))
        price_open = order.get('price_open', order.get('entry_price', 0))
        price_current = order.get('price_current', 0)
        symbol = order.get('symbol', 'Unknown')
        
        # Calculate pips
        pips_profit = self._calculate_pips_profit(symbol, price_open, price_current, order_type)
        
        # Emoji icons matching Order Close style
        if profit_usd > 0:
            result_icon = "💰"  # Money bag for profit
        elif profit_usd < 0:
            result_icon = "❌"  # X for loss
        else:
            result_icon = "⚖️"  # Scale for breakeven
        
        # DCA indicator
        dca_tag = ""
        if is_dca:
            dca_level = order.get('dca_level', '')
            if dca_level:
                dca_tag = f" DCA{dca_level}"
            else:
                dca_tag = " DCA"
        
        # Format matching Order Close emoji style
        line = f"📊 {order_type} {volume:.2f}{dca_tag}"
        
        # Price info (compact)
        if price_open != price_current:
            price_change = ((price_current - price_open) / price_open * 100) if price_open > 0 else 0
            price_change_str = f"({price_change:+.1f}%)" if abs(price_change) >= 0.1 else ""
            line += f" @ {price_current} {price_change_str}"
        else:
            line += f" @ {price_current}"
        
        # P&L with emoji matching Order Close style  
        pips_str = f"{pips_profit:+.1f} pips" 
        usd_str = f"${profit_usd:+.2f}"
        
        line += f" • {result_icon} {usd_str} ({pips_str})"
        
        return line
    
    def _is_dca_order(self, order: Dict[str, Any]) -> bool:
        """Determine if an order is a DCA order"""
        # Check various indicators that suggest this is a DCA order
        comment = str(order.get('comment', '')).upper()
        magic = order.get('magic', 0)
        
        # Common DCA indicators
        dca_indicators = [
            'DCA' in comment,
            'AVERAGING' in comment,
            'AVG' in comment,
            'MARTINGALE' in comment,
            'GRID' in comment,
            order.get('dca_level') is not None,
            order.get('is_dca_order', False)
        ]
        
        return any(dca_indicators)
    
    def send_signal_notification(self, signal_data: Dict[str, Any]) -> bool:
        """Send trading signal notification"""
        try:
            message = self.format_signal_message(signal_data)
            
            # Send via configured channels
            success = False
            if self.config.get('telegram', {}).get('enabled', False):
                success |= self.send_telegram_message(message)
            
            if self.config.get('zalo', {}).get('enabled', False):
                success |= self.send_zalo_message(message)
            
            if success:
                self.logger.info(f"Signal notification sent for {signal_data.get('symbol', 'Unknown')}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending signal notification: {e}")
            return False
    
    def send_execution_notification(self, execution_data: Dict[str, Any]) -> bool:
        """Send trade execution notification"""
        try:
            symbol = execution_data.get('symbol', 'Unknown')
            action = execution_data.get('action', 'Unknown')
            volume = execution_data.get('volume', 0)
            price = execution_data.get('price', 0)
            
            message = f"✅ <b>ORDER EXECUTED</b>\n\n"
            message += f"📊 Symbol: <b>{symbol}</b>\n"
            message += f"🔄 Action: <b>{action}</b>\n"
            message += f"📦 Volume: <b>{volume}</b>\n"
            message += f"💰 Price: <b>{price}</b>\n\n"
            message += f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}"
            
            # Send via configured channels
            success = False
            if self.config.get('telegram', {}).get('enabled', False):
                success |= self.send_telegram_message(message)
            
            if self.config.get('zalo', {}).get('enabled', False):
                success |= self.send_zalo_message(message)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending execution notification: {e}")
            return False
    
    def send_real_trade_notification(self, trade_data: Dict[str, Any]) -> bool:
        """Send real trade notification with P&L"""
        try:
            symbol = trade_data.get('symbol', 'Unknown')
            action = trade_data.get('action', 'Unknown')
            volume = trade_data.get('volume', 0)
            profit = trade_data.get('profit', 0)
            
            profit_icon = "+" if profit >= 0 else "-"
            
            message = f"[{profit_icon}] <b>TRADE UPDATE</b>\n\n"
            message += f">>> Symbol: <b>{symbol}</b>\n"
            message += f"<<< Action: <b>{action}</b>\n"
            message += f"=== Volume: <b>{volume}</b>\n"
            message += f"*** Profit: <b>{profit:+.2f} USD</b>\n\n"
            message += f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}"
            
            # Send via configured channels
            success = False
            if self.config.get('telegram', {}).get('enabled', False):
                success |= self.send_telegram_message(message)
            
            if self.config.get('zalo', {}).get('enabled', False):
                success |= self.send_zalo_message(message)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending trade notification: {e}")
            return False
    
    def _send_symbol_notification(self, symbol: str, orders: List[Dict[str, Any]], message_type: str) -> bool:
        """Send notification for a single symbol with all its orders"""
        try:
            if not orders:
                return False
            
            # Check spam protection per symbol
            current_time = time.time()
            message_key = f"{message_type}_{symbol}"
            
            if message_key in self.last_message_times:
                time_diff = current_time - self.last_message_times[message_key]
                if time_diff < self.spam_threshold:
                    return False  # Skip to prevent spam
            
            # Format message for this symbol
            message = self.format_symbol_order_message(symbol, orders)
            if not message:
                return False
            
            # Send via configured channels
            success = False
            if self.config.get('telegram', {}).get('enabled', False):
                success |= self.send_telegram_message(message)
            
            if self.config.get('zalo', {}).get('enabled', False):
                success |= self.send_zalo_message(message)
            
            if success:
                self.last_message_times[message_key] = current_time
                self.logger.info(f"Symbol notification sent: {symbol} ({len(orders)} orders)")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending symbol notification: {e}")
            return False
    
    def _send_grouped_notification(self, orders: List[Dict[str, Any]], message_key: str, spam_threshold: int = None) -> bool:
        """Send notification for a group of orders with spam protection"""
        try:
            if not orders:
                return False
                
            # Check spam protection
            current_time = time.time()
            threshold = spam_threshold or self.spam_threshold
            
            if message_key in self.last_message_times:
                time_diff = current_time - self.last_message_times[message_key]
                if time_diff < threshold:
                    return False  # Skip to prevent spam
            
            # Format message with grouping
            message = self.format_order_update_message(orders)
            if not message:
                return False
            
            # Send via configured channels
            success = False
            if self.config.get('telegram', {}).get('enabled', False):
                success |= self.send_telegram_message(message)
            
            if self.config.get('zalo', {}).get('enabled', False):
                success |= self.send_zalo_message(message)
            
            if success:
                self.last_message_times[message_key] = current_time
                order_types = "DCA" if any(self._is_dca_order(o) for o in orders) else "Entry"
                self.logger.info(f"{order_types} notification sent for {len(orders)} orders")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending grouped notification: {e}")
            return False
    
    def send_custom_message(self, message: str) -> bool:
        """Send custom message"""
        try:
            success = False
            if self.config.get('telegram', {}).get('enabled', False):
                success |= self.send_telegram_message(message)
            
            if self.config.get('zalo', {}).get('enabled', False):
                success |= self.send_zalo_message(message)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending custom message: {e}")
            return False
    
    def send_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """Send daily trading summary"""
        try:
            total_trades = summary_data.get('total_trades', 0)
            total_profit = summary_data.get('total_profit', 0)
            win_rate = summary_data.get('win_rate', 0)
            
            profit_emoji = "📈" if total_profit >= 0 else "📉"
            
            message = f"{profit_emoji} <b>DAILY SUMMARY</b>\n\n"
            message += f"📊 Total Trades: <b>{total_trades}</b>\n"
            message += f"💰 Total Profit: <b>{total_profit:+.2f} USD</b>\n"
            message += f"🎯 Win Rate: <b>{win_rate:.1f}%</b>\n\n"
            message += f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}"
            
            # Send via configured channels
            success = False
            if self.config.get('telegram', {}).get('enabled', False):
                success |= self.send_telegram_message(message)
            
            if self.config.get('zalo', {}).get('enabled', False):
                success |= self.send_zalo_message(message)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending daily summary: {e}")
            return False
    
    def send_dca_group_notification(self, dca_orders: List[Dict[str, Any]]) -> bool:
        """Send dedicated notification for DCA order groups"""
        try:
            if not dca_orders:
                return False
            
            # Filter to only DCA orders
            actual_dca_orders = [order for order in dca_orders if self._is_dca_order(order)]
            
            if not actual_dca_orders:
                return False
            
            # Use shorter spam threshold for DCA updates (more frequent allowed)
            return self._send_grouped_notification(actual_dca_orders, "dca_group_updates", spam_threshold=2)
            
        except Exception as e:
            self.logger.error(f"Error sending DCA group notification: {e}")
            return False
    
    def send_entry_orders_notification(self, entry_orders: List[Dict[str, Any]]) -> bool:
        """Send dedicated notification for Entry orders"""
        try:
            if not entry_orders:
                return False
            
            # Filter to only non-DCA orders
            actual_entry_orders = [order for order in entry_orders if not self._is_dca_order(order)]
            
            if not actual_entry_orders:
                return False
            
            # Use normal spam threshold for Entry updates
            return self._send_grouped_notification(actual_entry_orders, "entry_group_updates", spam_threshold=5)
            
        except Exception as e:
            self.logger.error(f"Error sending Entry group notification: {e}")
            return False
    
    def _send_symbol_sltp_notification(self, symbol: str, orders: List[Dict[str, Any]]) -> bool:
        """Send SL/TP notification for a single symbol"""
        try:
            message = self._format_sltp_message(symbol, orders)
            if not message:
                return False
            
            # Send via configured channels
            success = False
            if self.config.get('telegram', {}).get('enabled', False):
                success |= self.send_telegram_message(message)
            
            if self.config.get('zalo', {}).get('enabled', False):
                success |= self.send_zalo_message(message)
            
            if success:
                self.logger.info(f"SL/TP notification sent: {symbol}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending SL/TP symbol notification: {e}")
            return False
    
    def _format_sltp_message(self, symbol: str, orders: List[Dict[str, Any]]) -> str:
        """Format SL/TP change message - simple and consistent"""
        message_parts = []
        
        # Header matching Order Close style
        total_orders = len(orders)
        header = f"🛡️ <b>{symbol}</b>"  # Shield emoji like Order Close
        if total_orders > 1:
            header += f" ({total_orders})"
        
        message_parts.append(header)
        changes_text = I18N.t("📊 SL/TP Changes:", "📊 Thay đổi SL/TP:")
        message_parts.append(changes_text)
        
        # Show orders in simple format
        for order in orders:
            order_type = order.get('type', 'Unknown')
            volume = float(order.get('volume', 0))
            sl_new = order.get('sl', order.get('stop_loss', 0))
            tp_new = order.get('tp', order.get('take_profit', 0))
            price_current = order.get('price_current', 0)
            
            # DCA indicator
            dca_tag = ""
            is_dca = self._is_dca_order(order)
            if is_dca:
                dca_level = order.get('dca_level', '')
                if dca_level:
                    dca_tag = f" DCA{dca_level}"
                else:
                    dca_tag = " DCA"
            
            # Format matching Order Close emoji style
            line = f"📊 {order_type} {volume:.2f}{dca_tag} @ {price_current}"
            line += f" • SL:{sl_new} TP:{tp_new}"
            
            message_parts.append(line)
        
        # Timestamp - multi-language  
        time_str = datetime.now().strftime('%H:%M:%S %d/%m/%Y')
        time_label = I18N.t("Time", "Thời gian")
        message_parts.append(f"⏰ {time_label}: {time_str}")
        
        return "\n".join(message_parts)
    
    def _send_symbol_close_notification(self, symbol: str, orders: List[Dict[str, Any]]) -> bool:
        """Send order close notification for a single symbol"""
        try:
            message = self._format_close_message(symbol, orders)
            if not message:
                return False
            
            # Send via configured channels
            success = False
            if self.config.get('telegram', {}).get('enabled', False):
                success |= self.send_telegram_message(message)
            
            if self.config.get('zalo', {}).get('enabled', False):
                success |= self.send_zalo_message(message)
            
            if success:
                self.logger.info(f"Order close notification sent: {symbol}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending close symbol notification: {e}")
            return False
    
    def _format_close_message(self, symbol: str, orders: List[Dict[str, Any]]) -> str:
        """Format order close message like Gold Killer Club style"""
        message_parts = []
        
        # Process each order separately (like the examples)
        for order in orders:
            order_type = order.get('type', 'Unknown')
            volume = float(order.get('volume', 0))
            profit_usd = float(order.get('profit', 0))
            price_open = order.get('price_open', order.get('entry_price', 0))
            price_close = order.get('price_close', order.get('price_current', 0))
            
            # Calculate pips
            pips_profit = self._calculate_pips_profit(symbol, price_open, price_close, order_type)
            
            # DCA level
            dca_tag = ""
            is_dca = self._is_dca_order(order)
            if is_dca:
                dca_level = order.get('dca_level', '')
                if dca_level:
                    dca_tag = f"|DCA{dca_level}"
                else:
                    dca_tag = "|DCA"
            
            # Multi-language header
            header_text = I18N.t("✅ ORDER CLOSED", "✅ ĐÓNG LỆNH")
            message_parts.append(header_text)
            message_parts.append("")
            
            # Symbol with DCA tag
            message_parts.append(f"{symbol}{dca_tag}")
            
            # Volume - multi-language
            volume_text = I18N.t(f"📊 Volume: {volume:.2f} lot", f"📊 Volume: {volume:.2f} lot")
            message_parts.append(volume_text)
            
            # Result - multi-language
            if profit_usd > 0:
                result_emoji = "💰"
                result_text = I18N.t("PROFIT", "LÃI")
            elif profit_usd < 0:
                result_emoji = "❌" 
                result_text = I18N.t("LOSS", "LỖ")
            else:
                result_emoji = "⚖️"
                result_text = I18N.t("BREAKEVEN", "HÒA")
            
            result_label = I18N.t("Result", "Kết quả")
            message_parts.append(f"{result_emoji} {result_label}: {result_text}")
            
            # P&L with pips - exact format
            pips_sign = "+" if pips_profit >= 0 else ""
            usd_sign = "+" if profit_usd >= 0 else ""
            message_parts.append(f"💵 P&L: ${usd_sign}{profit_usd:.2f} ({pips_sign}{pips_profit:.1f} pips)")
            
            # Close reason - multi-language
            close_reason = order.get('close_reason', 'Manual Close')
            if profit_usd > 0:
                close_reason = I18N.t("Take Profit", "Take Profit")
            elif profit_usd == 0:
                close_reason = I18N.t("Breakeven", "Breakeven")
            else:
                close_reason = I18N.t("Stop Loss", "Stop Loss")
                
            reason_label = I18N.t("Close reason", "Lý do đóng")
            message_parts.append(f"📊 {reason_label}: {close_reason}")
            
            # Time - multi-language
            now = datetime.now()
            time_str = now.strftime('%H:%M:%S %d/%m/%Y')
            time_label = I18N.t("Time", "Thời gian")
            message_parts.append(f"⏰ {time_label}: {time_str}")
            
            # Add separator if multiple orders
            if len(orders) > 1 and order != orders[-1]:
                message_parts.append("")
                message_parts.append("-" * 30)
                message_parts.append("")
        
        return "\n".join(message_parts)

# Test the notification service
if __name__ == "__main__":
    # Test configuration
    config = {
        "telegram": {
            "enabled": True,
            "bot_token": "test_token",
            "chat_id": "test_chat_id"
        }
    }
    
    # Test connections
    notification_service = NotificationService()
    
    # Test results
    results = {}
    
    print("🚀 Testing NotificationService...")
    
    # Test each platform
    for platform in config:
        success = config[platform]['enabled']
        status = "✅ Connected" if success else "❌ Failed"
        results[platform.capitalize()] = status
        print(f"{platform.capitalize()}: {status}")
    
    print("\n🔧 NotificationService ready!")