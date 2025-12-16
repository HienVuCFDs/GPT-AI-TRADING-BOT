#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔔 SIMPLE ORDER TRACKER
========================
Theo dõi thay đổi SL/TP và đóng lệnh trực tiếp từ MT5
Gửi thông báo qua Telegram khi có thay đổi

Tính năng:
- Đánh dấu lệnh Entry / DCA
- Nhóm các lệnh cùng symbol trong 1 tin nhắn
- Thông báo song ngữ (EN/VI)

Author: Trading Bot System
Created: 2025-12-01
"""

import threading
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict
import json

# Setup logger
logger = logging.getLogger(__name__)

# Global instance
_tracker_instance = None


class SimpleOrderTracker:
    """
    Theo dõi các thay đổi SL/TP và đóng lệnh từ MT5
    Hỗ trợ nhóm lệnh theo symbol và đánh dấu Entry/DCA
    """
    
    def __init__(self):
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        
        # Lưu trữ trạng thái các lệnh
        self._tracked_orders: Dict[int, Dict] = {}
        
        # Pending changes để nhóm lại trước khi gửi
        self._pending_sl_tp_changes: Dict[str, List[Dict]] = defaultdict(list)
        self._pending_closes: Dict[str, List[Dict]] = defaultdict(list)
        self._last_notification_time = 0
        self._notification_delay = 2  # Seconds to wait for grouping
        
        # 🆕 Track last notified pips for each order (for pip threshold smart notification)
        # Logic: Lần 1 thông báo khi lệch [threshold] pips so với giá mở
        #        Lần 2 thông báo khi lệch [threshold] pips so với lần thông báo trước đó
        self._last_notified_pips: Dict[int, float] = {}  # ticket -> last notified pips change
        
        # Thống kê
        self._stats = {
            'sl_tp_changes': 0,
            'order_closes': 0,
            'status_updates': 0,
            'last_check': None,
            'errors': 0
        }
        
        # Config
        self._config = self._load_config()
        
        # Check interval (seconds)
        self._check_interval = 5
        
        # Last P/L status notification time
        self._last_status_notification = 0
        
    def _load_config(self) -> Dict:
        """Load notification config"""
        try:
            config_path = Path(__file__).parent / "notification_config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        return {}
    
    def _reload_config(self):
        """Reload config từ file"""
        self._config = self._load_config()
    
    def _get_language(self) -> str:
        """Get current language from config"""
        return 'vi' if self._config.get('settings', {}).get('format_vietnamese', True) else 'en'
    
    def _is_sl_tp_tracking_enabled(self) -> bool:
        """Check if SL/TP tracking is enabled"""
        return self._config.get('settings', {}).get('notify_sl_tp_changes', False)
    
    def _is_close_tracking_enabled(self) -> bool:
        """Check if order close tracking is enabled"""
        return self._config.get('settings', {}).get('notify_order_close', False)
    
    def _is_order_status_tracking_enabled(self) -> bool:
        """Check if order status (P/L) tracking is enabled - 'Theo dõi thay đổi lệnh'"""
        return self._config.get('settings', {}).get('track_order_updates', False)
    
    def _get_status_update_interval(self) -> int:
        """Get interval for P/L status notifications (seconds) - MINIMUM 60 seconds"""
        interval = self._config.get('settings', {}).get('update_interval_seconds', 60)
        # Enforce minimum 60 seconds to prevent spam
        return max(interval, 60)
    
    def _get_track_orders_pip_threshold(self) -> int:
        """
        Get pip threshold for order status tracking from UI.
        Reads from AppState (UI spinbox) if available, falls back to config.
        
        UI: track_orders_pips_spin (default 50 pips)
        Config: track_orders_pip_threshold (fallback)
        """
        try:
            # Try to read from AppState (UI settings)
            import sys
            for module_name, module in sys.modules.items():
                if module and hasattr(module, 'AppState'):
                    app_state = getattr(module, 'AppState')
                    if hasattr(app_state, 'app') and app_state.app:
                        # Access UI widget value
                        main_window = app_state.app
                        if hasattr(main_window, 'track_orders_pips_spin'):
                            value = main_window.track_orders_pips_spin.value()
                            logger.debug(f"Read track_orders_pip_threshold from UI: {value}")
                            return value
        except Exception as e:
            logger.debug(f"Could not read from AppState: {e}")
        
        # Fallback to config file
        return self._config.get('settings', {}).get('track_orders_pip_threshold', 50)
    
    def _get_sltp_pip_threshold(self) -> int:
        """
        Get pip threshold for SL/TP change notifications from UI.
        Reads from AppState (UI spinbox) if available, falls back to config.
        
        UI: sltp_pips_spin (default 10 pips)
        Config: sltp_pip_threshold (fallback)
        """
        try:
            # Try to read from AppState (UI settings)
            import sys
            for module_name, module in sys.modules.items():
                if module and hasattr(module, 'AppState'):
                    app_state = getattr(module, 'AppState')
                    if hasattr(app_state, 'app') and app_state.app:
                        # Access UI widget value
                        main_window = app_state.app
                        if hasattr(main_window, 'sltp_pips_spin'):
                            value = main_window.sltp_pips_spin.value()
                            logger.debug(f"Read sltp_pip_threshold from UI: {value}")
                            return value
        except Exception as e:
            logger.debug(f"Could not read from AppState: {e}")
        
        # Fallback to config file
        return self._config.get('settings', {}).get('sltp_pip_threshold', 10)
    
    def _is_dca_order(self, order_data: Dict) -> bool:
        """Check if order is DCA based on comment or magic"""
        comment = str(order_data.get('comment', '')).upper()
        
        dca_indicators = [
            'DCA' in comment,
            'AVG' in comment,
            'AVERAGING' in comment,
            'MARTINGALE' in comment,
            'GRID' in comment,
        ]
        
        return any(dca_indicators)
    
    def _get_order_type_tag(self, order_data: Dict) -> str:
        """Get Entry or DCA tag for order"""
        if self._is_dca_order(order_data):
            # Try to extract DCA level from comment
            comment = order_data.get('comment', '')
            import re
            match = re.search(r'DCA(\d+)', comment, re.IGNORECASE)
            if match:
                return f"DCA{match.group(1)}"
            return "DCA"
        return "Entry"
    
    def start(self) -> bool:
        """Start tracking thread"""
        with self._lock:
            if self._running:
                logger.info("⚠️ Tracker already running")
                return True
            
            try:
                import MetaTrader5 as mt5
                if not mt5.initialize():
                    logger.error("❌ Cannot initialize MT5")
                    return False
                
                # Get initial positions
                positions = mt5.positions_get()
                if positions is None:
                    positions = []
                
                # Store current state
                for pos in positions:
                    pos_dict = pos._asdict()
                    ticket = pos_dict.get('ticket', 0)
                    self._tracked_orders[ticket] = {
                        'ticket': ticket,
                        'symbol': pos_dict.get('symbol', ''),
                        'type': 'BUY' if pos_dict.get('type', 0) == 0 else 'SELL',
                        'volume': pos_dict.get('volume', 0),
                        'price_open': pos_dict.get('price_open', 0),
                        'sl': pos_dict.get('sl', 0),
                        'tp': pos_dict.get('tp', 0),
                        'profit': pos_dict.get('profit', 0),
                        'price_current': pos_dict.get('price_current', 0),
                        'comment': pos_dict.get('comment', ''),
                        'magic': pos_dict.get('magic', 0),
                        'time': pos_dict.get('time', 0),
                    }
                
                mt5.shutdown()
                
                logger.info(f"📊 Loaded {len(self._tracked_orders)} existing positions")
                
                # Start thread
                self._running = True
                self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
                self._thread.start()
                
                logger.info("🚀 Order Tracker started")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error starting tracker: {e}")
                return False
    
    def stop(self):
        """Stop tracking thread"""
        with self._lock:
            self._running = False
            
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        logger.info("🛑 Order Tracker stopped")
    
    def is_running(self) -> bool:
        """Check if tracker is running"""
        return self._running
    
    def _tracking_loop(self):
        """Main tracking loop"""
        logger.info("🔄 Tracking loop started")
        
        while self._running:
            try:
                # Reload config to get latest settings
                self._reload_config()
                
                # Check if any tracking is enabled
                sl_tp_enabled = self._is_sl_tp_tracking_enabled()
                close_enabled = self._is_close_tracking_enabled()
                status_enabled = self._is_order_status_tracking_enabled()
                
                if not sl_tp_enabled and not close_enabled and not status_enabled:
                    time.sleep(self._check_interval)
                    continue
                
                # Check for changes
                self._check_positions(sl_tp_enabled, close_enabled)
                self._stats['last_check'] = datetime.now()
                
                # Send grouped notifications if pending
                self._send_pending_notifications()
                
                # Send periodic P/L status update ("Theo dõi thay đổi lệnh")
                if status_enabled:
                    self._check_and_send_status_update()
                
            except Exception as e:
                logger.error(f"❌ Tracking loop error: {e}")
                self._stats['errors'] += 1
            
            time.sleep(self._check_interval)
        
        logger.info("🔄 Tracking loop ended")
    
    def _check_positions(self, check_sl_tp: bool, check_close: bool):
        """Check for position changes"""
        try:
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                return
            
            # Get current positions
            current_positions = mt5.positions_get()
            if current_positions is None:
                current_positions = []
            
            current_tickets = set()
            
            # Check each current position
            for pos in current_positions:
                pos_dict = pos._asdict()
                ticket = pos_dict.get('ticket', 0)
                current_tickets.add(ticket)
                
                if ticket in self._tracked_orders:
                    # Existing position - check for SL/TP changes
                    if check_sl_tp:
                        self._check_sl_tp_change(ticket, pos_dict)
                    # Update current profit
                    self._tracked_orders[ticket]['profit'] = pos_dict.get('profit', 0)
                    self._tracked_orders[ticket]['price_current'] = pos_dict.get('price_current', 0)
                else:
                    # New position - add to tracking
                    self._tracked_orders[ticket] = {
                        'ticket': ticket,
                        'symbol': pos_dict.get('symbol', ''),
                        'type': 'BUY' if pos_dict.get('type', 0) == 0 else 'SELL',
                        'volume': pos_dict.get('volume', 0),
                        'price_open': pos_dict.get('price_open', 0),
                        'sl': pos_dict.get('sl', 0),
                        'tp': pos_dict.get('tp', 0),
                        'profit': pos_dict.get('profit', 0),
                        'price_current': pos_dict.get('price_current', 0),
                        'comment': pos_dict.get('comment', ''),
                        'magic': pos_dict.get('magic', 0),
                        'time': pos_dict.get('time', 0),
                    }
                    logger.debug(f"📥 New position tracked: #{ticket}")
            
            # Check for closed positions
            if check_close:
                for ticket in list(self._tracked_orders.keys()):
                    if ticket not in current_tickets:
                        # Position closed - add to pending
                        self._handle_position_close(ticket, mt5)
            
            mt5.shutdown()
            
        except Exception as e:
            logger.error(f"❌ Error checking positions: {e}")
    
    def _check_sl_tp_change(self, ticket: int, current_pos: Dict):
        """Check and queue SL/TP changes for grouped notification (with pip threshold)"""
        try:
            old_data = self._tracked_orders.get(ticket, {})
            
            old_sl = old_data.get('sl', 0)
            old_tp = old_data.get('tp', 0)
            new_sl = current_pos.get('sl', 0)
            new_tp = current_pos.get('tp', 0)
            
            symbol = old_data.get('symbol', 'Unknown')
            pip_threshold = self._get_sltp_pip_threshold()
            
            changes = []
            
            # Check SL change with pip threshold
            if abs(new_sl - old_sl) > 0.00001:
                # Calculate pip difference
                sl_pips = self._calculate_pips(symbol, old_sl, new_sl, 'BUY')
                sl_pips = abs(sl_pips)
                
                # Only add change if exceeds threshold (or threshold is 0)
                if pip_threshold == 0 or sl_pips >= pip_threshold:
                    changes.append(('SL', old_sl, new_sl))
                    logger.debug(f"SL change detected: {sl_pips:.1f} pips >= threshold {pip_threshold}")
                else:
                    logger.debug(f"SL change ignored: {sl_pips:.1f} pips < threshold {pip_threshold}")
                
            # Check TP change with pip threshold
            if abs(new_tp - old_tp) > 0.00001:
                # Calculate pip difference
                tp_pips = self._calculate_pips(symbol, old_tp, new_tp, 'BUY')
                tp_pips = abs(tp_pips)
                
                # Only add change if exceeds threshold (or threshold is 0)
                if pip_threshold == 0 or tp_pips >= pip_threshold:
                    changes.append(('TP', old_tp, new_tp))
                    logger.debug(f"TP change detected: {tp_pips:.1f} pips >= threshold {pip_threshold}")
                else:
                    logger.debug(f"TP change ignored: {tp_pips:.1f} pips < threshold {pip_threshold}")
            
            if changes:
                # Update stored data
                self._tracked_orders[ticket]['sl'] = new_sl
                self._tracked_orders[ticket]['tp'] = new_tp
                self._tracked_orders[ticket]['profit'] = current_pos.get('profit', 0)
                self._tracked_orders[ticket]['price_current'] = current_pos.get('price_current', 0)
                
                # Add to pending for grouped notification
                change_data = {
                    'ticket': ticket,
                    'symbol': symbol,
                    'type': old_data.get('type', 'N/A'),
                    'volume': old_data.get('volume', 0),
                    'profit': current_pos.get('profit', 0),
                    'comment': old_data.get('comment', ''),
                    'changes': changes,
                    'new_sl': new_sl,
                    'new_tp': new_tp,
                }
                
                self._pending_sl_tp_changes[symbol].append(change_data)
                self._last_notification_time = time.time()
                self._stats['sl_tp_changes'] += 1
            else:
                # Still update stored SL/TP even if not notifying (below threshold)
                self._tracked_orders[ticket]['sl'] = new_sl
                self._tracked_orders[ticket]['tp'] = new_tp
                
        except Exception as e:
            logger.error(f"❌ Error checking SL/TP change: {e}")
    
    def _handle_position_close(self, ticket: int, mt5):
        """Queue position close for grouped notification"""
        try:
            old_data = self._tracked_orders.get(ticket, {})
            if not old_data:
                return
            
            symbol = old_data.get('symbol', 'Unknown')
            
            # Get close info from history
            close_info = self._get_close_info(ticket, mt5)
            
            # Add to pending
            close_data = {
                'ticket': ticket,
                'symbol': symbol,
                'type': old_data.get('type', 'N/A'),
                'volume': old_data.get('volume', 0),
                'price_open': old_data.get('price_open', 0),
                'comment': old_data.get('comment', ''),
                'close_info': close_info,
            }
            
            self._pending_closes[symbol].append(close_data)
            self._last_notification_time = time.time()
            self._stats['order_closes'] += 1
            
            # Remove from tracking
            del self._tracked_orders[ticket]
            logger.info(f"📤 Position #{ticket} closed and removed from tracking")
            
        except Exception as e:
            logger.error(f"❌ Error handling position close: {e}")
    
    def _get_close_info(self, ticket: int, mt5) -> Dict:
        """Get close information from MT5 history"""
        try:
            from datetime import timedelta
            
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            deals = mt5.history_deals_get(start_time, end_time, position=ticket)
            if deals:
                for deal in deals:
                    deal_dict = deal._asdict()
                    if deal_dict.get('entry') == 1:  # DEAL_ENTRY_OUT
                        return {
                            'close_price': deal_dict.get('price', 0),
                            'profit': deal_dict.get('profit', 0),
                            'swap': deal_dict.get('swap', 0),
                            'commission': deal_dict.get('commission', 0),
                            'close_time': datetime.fromtimestamp(deal_dict.get('time', 0)),
                        }
            
            return {}
        except Exception as e:
            logger.error(f"Error getting close info: {e}")
            return {}
    
    def _send_pending_notifications(self):
        """Send grouped notifications after delay"""
        current_time = time.time()
        
        # Wait for grouping delay
        if current_time - self._last_notification_time < self._notification_delay:
            return
        
        # Send SL/TP grouped notifications
        if self._pending_sl_tp_changes:
            for symbol, changes_list in list(self._pending_sl_tp_changes.items()):
                if changes_list:
                    self._send_grouped_sl_tp_notification(symbol, changes_list)
            self._pending_sl_tp_changes.clear()
        
        # Send close grouped notifications
        if self._pending_closes:
            for symbol, closes_list in list(self._pending_closes.items()):
                if closes_list:
                    self._send_grouped_close_notification(symbol, closes_list)
            self._pending_closes.clear()
    
    def _check_and_send_status_update(self):
        """
        Check if it's time to send periodic P/L status update with proper pip threshold logic.
        
        🛡️ ANTI-SPAM MEASURES:
        - Minimum interval: 60 seconds (enforced)
        - Pip threshold: 50 pips default (prevents small movement spam)
        - Per-order tracking: Each order's pips tracked separately
        
        Logic:
        - First notification: when pips changes >= threshold from entry price
        - Subsequent: when pips changes >= threshold from LAST notified pips
        """
        current_time = time.time()
        interval = self._get_status_update_interval()  # Min 60 seconds
        pip_threshold = self._get_track_orders_pip_threshold()
        
        # 🛑 ANTI-SPAM CHECK 1: Enforce minimum interval (60 seconds)
        if current_time - self._last_status_notification < interval:
            return
        
        # 🛑 ANTI-SPAM CHECK 2: If pip threshold is 0, disable status notifications entirely
        if pip_threshold == 0:
            logger.debug(f"Status notifications disabled (pip_threshold=0). Set to 50+ to enable.")
            self._last_status_notification = current_time
            return
        
        # Only process if we have positions to track
        if not self._tracked_orders:
            return
        
        # Check if any order has changed enough pips to warrant notification
        orders_to_notify = []
        
        for ticket, order_data in self._tracked_orders.items():
            symbol = order_data.get('symbol', 'Unknown')
            order_type = order_data.get('type', 'N/A')
            price_open = order_data.get('price_open', 0)
            price_current = order_data.get('price_current', 0)
            
            # Calculate current pips from entry price
            current_pips = self._calculate_pips(symbol, price_open, price_current, order_type)
            
            # Get last notified pips (0 if never notified)
            last_notified_pips = self._last_notified_pips.get(ticket, 0)
            
            # Calculate pip change from last notification (or from 0 if first time)
            pip_change = abs(current_pips - last_notified_pips)
            
            # Only notify if pip change >= threshold
            if pip_change >= pip_threshold:
                orders_to_notify.append(ticket)
                # Update last notified pips for next check
                self._last_notified_pips[ticket] = current_pips
                logger.debug(f"Order #{ticket}: notify (pips {current_pips}, change {pip_change}>={pip_threshold})")
            else:
                logger.debug(f"Order #{ticket}: skip (pips {current_pips}, change {pip_change}<{pip_threshold})")
        
        # 🛑 ANTI-SPAM CHECK 3: Only send if at least one order meets threshold
        if orders_to_notify:
            logger.info(f"Status update: {len(orders_to_notify)}/{len(self._tracked_orders)} orders met threshold ({pip_threshold}pips)")
            self._send_status_notification(orders_to_notify)  # 🔧 Pass only orders that meet threshold
            self._stats['status_updates'] += 1
        else:
            logger.debug(f"No orders met pip threshold ({pip_threshold}pips) - suppressing notification")
        
        # Always update last notification time to enforce interval
        self._last_status_notification = current_time
    
    def _send_status_notification(self, orders_to_notify=None):
        """Send current P/L status - separate message per symbol (or single order)
        
        Args:
            orders_to_notify: List of ticket numbers to include. If None, include all.
        """
        try:
            lang = self._get_language()
            
            # 🔧 Filter orders: only include those that meet pip threshold
            if orders_to_notify is None:
                orders_to_notify = list(self._tracked_orders.keys())
            
            # Group orders by symbol - ONLY include orders in orders_to_notify
            orders_by_symbol: Dict[str, List[Dict]] = defaultdict(list)
            for ticket in orders_to_notify:
                if ticket in self._tracked_orders:
                    order_data = self._tracked_orders[ticket]
                    symbol = order_data.get('symbol', 'Unknown')
                    orders_by_symbol[symbol].append(order_data)
            
            if not orders_by_symbol:
                logger.debug(f"No orders to notify (all filtered by pip threshold)")
                return
            
            # Send separate message for each symbol
            for symbol, orders in orders_by_symbol.items():
                self._send_symbol_status_notification(symbol, orders, lang)
            
            total_orders = len(orders_to_notify)
            total_profit = sum(self._tracked_orders[t].get('profit', 0) for t in orders_to_notify)
            logger.info(f"📤 Status notification sent: {len(orders_by_symbol)} symbols, {total_orders} orders, P&L: ${total_profit:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error sending status notification: {e}")
    
    def _send_symbol_status_notification(self, symbol: str, orders: List[Dict], lang: str):
        """Send status notification for a single symbol (can be 1 or multiple orders)"""
        try:
            num_orders = len(orders)
            symbol_profit = sum(o.get('profit', 0) for o in orders)
            
            # Determine overall emoji
            if symbol_profit > 0:
                result_emoji = "💰"
                result_text = "LÃI" if lang == 'vi' else "PROFIT"
            elif symbol_profit < 0:
                result_emoji = "❌"
                result_text = "LỖ" if lang == 'vi' else "LOSS"
            else:
                result_emoji = "⚖️"
                result_text = "HÒA" if lang == 'vi' else "BREAKEVEN"
            
            profit_sign = '+' if symbol_profit >= 0 else ''
            
            # Build message header
            if lang == 'vi':
                message = f"📊 <b>TRẠNG THÁI LỆNH - {symbol}</b>"
                if num_orders > 1:
                    message += f" ({num_orders} lệnh)"
                message += "\n"
            else:
                message = f"📊 <b>ORDER STATUS - {symbol}</b>"
                if num_orders > 1:
                    message += f" ({num_orders} orders)"
                message += "\n"
            
            message += "─" * 28 + "\n"
            
            # Show each order
            for order in orders:
                order_tag = self._get_order_type_tag(order)
                order_type = order.get('type', 'N/A')
                volume = order.get('volume', 0)
                ticket = order.get('ticket', 0)
                profit = order.get('profit', 0)
                price_open = order.get('price_open', 0)
                price_current = order.get('price_current', 0)
                sl = order.get('sl', 0)
                tp = order.get('tp', 0)
                
                order_profit_sign = '+' if profit >= 0 else ''
                
                if profit > 0:
                    order_emoji = "🟢"
                elif profit < 0:
                    order_emoji = "🔴"
                else:
                    order_emoji = "⚪"
                
                # Calculate pips
                pips = self._calculate_pips(symbol, price_open, price_current, order_type)
                pips_sign = '+' if pips >= 0 else ''
                
                message += f"\n🏷️ <b>{order_tag}</b> | {order_type} {volume:.2f} lot\n"
                message += f"🎟️ #{ticket}\n"
                message += f"💰 {price_open:.5f} → {price_current:.5f}\n"
                message += f"{order_emoji} P&L: <b>{order_profit_sign}${profit:.2f}</b> ({pips_sign}{pips:.1f} pips)\n"
                
                # Show SL/TP if set
                if sl > 0 or tp > 0:
                    sl_str = f"SL: {sl:.5f}" if sl > 0 else "SL: --"
                    tp_str = f"TP: {tp:.5f}" if tp > 0 else "TP: --"
                    message += f"🎯 {sl_str} | {tp_str}\n"
            
            # Summary for multiple orders
            if num_orders > 1:
                message += "\n" + "─" * 28 + "\n"
                message += f"{result_emoji} <b>Tổng: {profit_sign}${symbol_profit:.2f}</b>\n" if lang == 'vi' else f"{result_emoji} <b>Total: {profit_sign}${symbol_profit:.2f}</b>\n"
            
            message += f"\n⏰ {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
            
            # Add footer
            footer = self._get_footer()
            if footer:
                message += f"\n\n{footer}"
            
            self._send_telegram(message)
            
        except Exception as e:
            logger.error(f"❌ Error sending symbol status notification: {e}")
    
    def _send_grouped_sl_tp_notification(self, symbol: str, changes_list: List[Dict]):
        """Send grouped SL/TP notification for symbol"""
        try:
            lang = self._get_language()
            num_orders = len(changes_list)
            
            # Calculate totals
            total_profit = sum(c.get('profit', 0) for c in changes_list)
            
            if lang == 'vi':
                message = f"🛡️ <b>THAY ĐỔI SL/TP</b>\n\n"
                message += f"📊 <b>{symbol}</b>"
                if num_orders > 1:
                    message += f" ({num_orders} lệnh)"
                message += "\n"
                message += "─" * 25 + "\n"
                
                for change in changes_list:
                    order_tag = self._get_order_type_tag(change)
                    order_type = change.get('type', 'N/A')
                    volume = change.get('volume', 0)
                    ticket = change.get('ticket', 0)
                    profit = change.get('profit', 0)
                    
                    profit_sign = '+' if profit >= 0 else ''
                    
                    message += f"\n🏷️ <b>{order_tag}</b> | {order_type} {volume:.2f} lot\n"
                    message += f"🎟️ #{ticket} | P&L: {profit_sign}${profit:.2f}\n"
                    
                    for change_type, old_val, new_val in change.get('changes', []):
                        if change_type == 'SL':
                            message += f"  🛡️ SL: {old_val:.5f} → <b>{new_val:.5f}</b>\n"
                        elif change_type == 'TP':
                            message += f"  🎯 TP: {old_val:.5f} → <b>{new_val:.5f}</b>\n"
                
                if num_orders > 1:
                    message += "\n" + "─" * 25 + "\n"
                    profit_sign = '+' if total_profit >= 0 else ''
                    message += f"💵 Tổng P&L: <b>{profit_sign}${total_profit:.2f}</b>\n"
                
                message += f"\n⏰ {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
            else:
                message = f"🛡️ <b>SL/TP CHANGED</b>\n\n"
                message += f"📊 <b>{symbol}</b>"
                if num_orders > 1:
                    message += f" ({num_orders} orders)"
                message += "\n"
                message += "─" * 25 + "\n"
                
                for change in changes_list:
                    order_tag = self._get_order_type_tag(change)
                    order_type = change.get('type', 'N/A')
                    volume = change.get('volume', 0)
                    ticket = change.get('ticket', 0)
                    profit = change.get('profit', 0)
                    
                    profit_sign = '+' if profit >= 0 else ''
                    
                    message += f"\n🏷️ <b>{order_tag}</b> | {order_type} {volume:.2f} lot\n"
                    message += f"🎟️ #{ticket} | P&L: {profit_sign}${profit:.2f}\n"
                    
                    for change_type, old_val, new_val in change.get('changes', []):
                        if change_type == 'SL':
                            message += f"  🛡️ SL: {old_val:.5f} → <b>{new_val:.5f}</b>\n"
                        elif change_type == 'TP':
                            message += f"  🎯 TP: {old_val:.5f} → <b>{new_val:.5f}</b>\n"
                
                if num_orders > 1:
                    message += "\n" + "─" * 25 + "\n"
                    profit_sign = '+' if total_profit >= 0 else ''
                    message += f"💵 Total P&L: <b>{profit_sign}${total_profit:.2f}</b>\n"
                
                message += f"\n⏰ {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
            
            # Add footer
            footer = self._get_footer()
            if footer:
                message += f"\n\n{footer}"
            
            self._send_telegram(message)
            logger.info(f"📤 SL/TP notification sent: {symbol} ({num_orders} orders)")
            
        except Exception as e:
            logger.error(f"❌ Error sending grouped SL/TP notification: {e}")
    
    def _send_grouped_close_notification(self, symbol: str, closes_list: List[Dict]):
        """Send grouped close notification for symbol"""
        try:
            lang = self._get_language()
            num_orders = len(closes_list)
            
            # Calculate totals
            total_profit = 0
            total_pips = 0
            
            order_details = []
            for close in closes_list:
                close_info = close.get('close_info', {})
                profit = close_info.get('profit', 0)
                price_open = close.get('price_open', 0)
                close_price = close_info.get('close_price', 0)
                order_type = close.get('type', 'N/A')
                
                pips = self._calculate_pips(symbol, price_open, close_price, order_type)
                total_profit += profit
                total_pips += pips
                
                order_details.append({
                    **close,
                    'profit': profit,
                    'pips': pips,
                    'close_price': close_price,
                })
            
            # Determine overall result
            if total_profit > 0:
                result_emoji = "💰"
                result_text = "LÃI" if lang == 'vi' else "PROFIT"
            elif total_profit < 0:
                result_emoji = "❌"
                result_text = "LỖ" if lang == 'vi' else "LOSS"
            else:
                result_emoji = "⚖️"
                result_text = "HÒA" if lang == 'vi' else "BREAKEVEN"
            
            if lang == 'vi':
                message = f"🏁 <b>ĐÓNG LỆNH</b>\n\n"
                message += f"📊 <b>{symbol}</b>"
                if num_orders > 1:
                    message += f" ({num_orders} lệnh)"
                message += "\n"
                message += "─" * 25 + "\n"
                
                for detail in order_details:
                    order_tag = self._get_order_type_tag(detail)
                    order_type = detail.get('type', 'N/A')
                    volume = detail.get('volume', 0)
                    ticket = detail.get('ticket', 0)
                    profit = detail.get('profit', 0)
                    pips = detail.get('pips', 0)
                    price_open = detail.get('price_open', 0)
                    close_price = detail.get('close_price', 0)
                    
                    profit_sign = '+' if profit >= 0 else ''
                    pips_sign = '+' if pips >= 0 else ''
                    
                    if profit > 0:
                        order_emoji = "💰"
                    elif profit < 0:
                        order_emoji = "❌"
                    else:
                        order_emoji = "⚖️"
                    
                    message += f"\n🏷️ <b>{order_tag}</b> | {order_type} {volume:.2f} lot\n"
                    message += f"🎟️ #{ticket}\n"
                    message += f"💰 {price_open:.5f} → {close_price:.5f}\n"
                    message += f"{order_emoji} {profit_sign}${profit:.2f} ({pips_sign}{pips:.1f} pips)\n"
                
                message += "\n" + "─" * 25 + "\n"
                profit_sign = '+' if total_profit >= 0 else ''
                pips_sign = '+' if total_pips >= 0 else ''
                message += f"{result_emoji} <b>{result_text}</b>\n"
                message += f"💵 Tổng: <b>{profit_sign}${total_profit:.2f}</b> ({pips_sign}{total_pips:.1f} pips)\n"
                message += f"\n⏰ {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
            else:
                message = f"🏁 <b>ORDER CLOSED</b>\n\n"
                message += f"📊 <b>{symbol}</b>"
                if num_orders > 1:
                    message += f" ({num_orders} orders)"
                message += "\n"
                message += "─" * 25 + "\n"
                
                for detail in order_details:
                    order_tag = self._get_order_type_tag(detail)
                    order_type = detail.get('type', 'N/A')
                    volume = detail.get('volume', 0)
                    ticket = detail.get('ticket', 0)
                    profit = detail.get('profit', 0)
                    pips = detail.get('pips', 0)
                    price_open = detail.get('price_open', 0)
                    close_price = detail.get('close_price', 0)
                    
                    profit_sign = '+' if profit >= 0 else ''
                    pips_sign = '+' if pips >= 0 else ''
                    
                    if profit > 0:
                        order_emoji = "💰"
                    elif profit < 0:
                        order_emoji = "❌"
                    else:
                        order_emoji = "⚖️"
                    
                    message += f"\n🏷️ <b>{order_tag}</b> | {order_type} {volume:.2f} lot\n"
                    message += f"🎟️ #{ticket}\n"
                    message += f"💰 {price_open:.5f} → {close_price:.5f}\n"
                    message += f"{order_emoji} {profit_sign}${profit:.2f} ({pips_sign}{pips:.1f} pips)\n"
                
                message += "\n" + "─" * 25 + "\n"
                profit_sign = '+' if total_profit >= 0 else ''
                pips_sign = '+' if total_pips >= 0 else ''
                message += f"{result_emoji} <b>{result_text}</b>\n"
                message += f"💵 Total: <b>{profit_sign}${total_profit:.2f}</b> ({pips_sign}{total_pips:.1f} pips)\n"
                message += f"\n⏰ {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
            
            # Add footer
            footer = self._get_footer()
            if footer:
                message += f"\n\n{footer}"
            
            self._send_telegram(message)
            logger.info(f"📤 Close notification sent: {symbol} ({num_orders} orders) - {result_text}")
            
        except Exception as e:
            logger.error(f"❌ Error sending grouped close notification: {e}")
    
    def _calculate_pips(self, symbol: str, price_open: float, price_close: float, order_type: str) -> float:
        """
        Calculate pips profit/loss using UNIFIED pip_value standard from utils.py
        
        Pip value standards:
        - Metals (XAUUSD, XAGUSD): 0.1
        - JPY pairs: 0.01
        - BTC/ETH: 1.0
        - Mid-crypto (SOL, LTC, BNB, etc): 0.1
        - Major FX: 0.0001
        """
        try:
            from utils import get_pip_value  # Use unified standard
            
            symbol_upper = symbol.upper()
            pip_value = get_pip_value(symbol)
            
            price_diff = price_close - price_open
            
            if order_type == 'SELL':
                price_diff = -price_diff
            
            pips = price_diff / pip_value
            return round(pips, 1)
            
        except Exception as e:
            logger.debug(f"Error calculating pips for {symbol}: {e}")
            return 0.0
    
    def _get_footer(self) -> str:
        """Get branded footer from config"""
        try:
            branding = self._config.get('branding', {})
            if not branding.get('enable_custom_footer', True):
                return ""
            
            lang = self._get_language()
            if lang == 'vi':
                system_name = branding.get('system_name_vi', 'Hệ thống AI Trading')
            else:
                system_name = branding.get('system_name_en', 'AI Trading System')
            
            phone = branding.get('phone', '')
            
            footer_parts = [f"🤖 {system_name}"]
            if phone:
                footer_parts.append(f"📱 {phone}")
            
            return '\n'.join(footer_parts)
            
        except Exception:
            return ""
    
    def _send_telegram(self, message: str) -> bool:
        """Send message via Telegram"""
        try:
            import requests
            
            telegram_config = self._config.get('telegram', {})
            if not telegram_config.get('enabled', False):
                logger.debug("Telegram not enabled")
                return False
            
            bot_token = telegram_config.get('bot_token', '').strip()
            chat_id = telegram_config.get('chat_id', '').strip()
            
            if not bot_token or not chat_id:
                logger.warning("Telegram not configured (missing token or chat_id)")
                return False
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.debug("✅ Telegram message sent")
                return True
            else:
                logger.error(f"❌ Telegram error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending Telegram: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get tracker statistics"""
        return {
            **self._stats,
            'tracked_orders': len(self._tracked_orders),
            'is_running': self._running,
        }


def get_tracker() -> SimpleOrderTracker:
    """Get singleton tracker instance"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = SimpleOrderTracker()
    return _tracker_instance


# ========================================
# 🧪 TEST
# ========================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 60)
    print("🔔 SIMPLE ORDER TRACKER - TEST (VI + EN)")
    print("=" * 60 + "\n")
    
    tracker = get_tracker()
    tracker._reload_config()
    
    # Test data
    changes_list = [
        {
            'ticket': 12345678,
            'symbol': 'XAUUSD',
            'type': 'BUY',
            'volume': 0.1,
            'profit': 50.25,
            'comment': '',
            'changes': [('SL', 2640.50, 2645.00), ('TP', 2670.00, 2680.00)],
            'new_sl': 2645.00,
            'new_tp': 2680.00,
        },
        {
            'ticket': 12345679,
            'symbol': 'XAUUSD',
            'type': 'BUY',
            'volume': 0.15,
            'profit': 75.50,
            'comment': 'DCA1',
            'changes': [('SL', 2640.50, 2645.00)],
            'new_sl': 2645.00,
            'new_tp': 2680.00,
        },
    ]
    
    tracker._send_grouped_sl_tp_notification('XAUUSD', changes_list)
    print("✅ Grouped SL/TP notification sent!\n")
    
    # Test grouped close notification
    print("📤 Testing grouped close notification...")
    
    closes_list = [
        {
            'ticket': 87654321,
            'symbol': 'BTCUSD_m',
            'type': 'BUY',
            'volume': 0.10,
            'price_open': 95000.00,
            'comment': '',
            'close_info': {
                'close_price': 96500.00,
                'profit': 150.00,
            },
        },
        {
            'ticket': 87654322,
            'symbol': 'BTCUSD_m',
            'type': 'BUY',
            'volume': 0.15,
            'price_open': 95500.00,
            'comment': 'DCA1',
            'close_info': {
                'close_price': 96500.00,
                'profit': 150.00,
            },
        },
        {
            'ticket': 87654323,
            'symbol': 'BTCUSD_m',
            'type': 'BUY',
            'volume': 0.20,
            'price_open': 96000.00,
            'comment': 'DCA2',
            'close_info': {
                'close_price': 96500.00,
                'profit': 100.00,
            },
        },
    ]
    
    tracker._send_grouped_close_notification('BTCUSD_m', closes_list)
    print("✅ Grouped close notification sent!\n")
    
    # Test status notification (P/L tracking)
    print("📤 Testing status notification (P/L tracking)...")
    
    # Simulate tracked orders for status notification
    tracker._tracked_orders = {
        11111: {
            'ticket': 11111,
            'symbol': 'XAUUSD',
            'type': 'BUY',
            'volume': 0.10,
            'price_open': 2650.00,
            'price_current': 2665.50,
            'sl': 2640.00,
            'tp': 2680.00,
            'profit': 155.00,
            'comment': '',
            'magic': 0,
        },
        11112: {
            'ticket': 11112,
            'symbol': 'XAUUSD',
            'type': 'BUY',
            'volume': 0.15,
            'price_open': 2655.00,
            'price_current': 2665.50,
            'sl': 2640.00,
            'tp': 2680.00,
            'profit': 157.50,
            'comment': 'DCA1',
            'magic': 0,
        },
        22221: {
            'ticket': 22221,
            'symbol': 'BTCUSD_m',
            'type': 'SELL',
            'volume': 0.05,
            'price_open': 97500.00,
            'price_current': 97000.00,
            'sl': 98000.00,
            'tp': 95000.00,
            'profit': 25.00,
            'comment': '',
            'magic': 0,
        },
        33331: {
            'ticket': 33331,
            'symbol': 'EURUSD',
            'type': 'SELL',
            'volume': 0.20,
            'price_open': 1.05800,
            'price_current': 1.06000,
            'sl': 1.06500,
            'tp': 1.04500,
            'profit': -40.00,
            'comment': '',
            'magic': 0,
        },
    }
    
    tracker._send_status_notification()
    print("✅ Status notifications sent (Vietnamese)!\n")
    
    # ========================================
    # 🇬🇧 TEST ENGLISH VERSION
    # ========================================
    print("=" * 60)
    print("🇬🇧 TESTING ENGLISH VERSION")
    print("=" * 60 + "\n")
    
    # Switch to English
    tracker._config['settings']['format_vietnamese'] = False
    
    # Test SL/TP English
    print("📤 Testing SL/TP notification (English)...")
    tracker._send_grouped_sl_tp_notification('XAUUSD', changes_list)
    print("✅ SL/TP notification sent (English)!\n")
    
    # Test Close English
    print("📤 Testing close notification (English)...")
    tracker._send_grouped_close_notification('BTCUSD_m', closes_list)
    print("✅ Close notification sent (English)!\n")
    
    # Test Status English
    print("📤 Testing status notification (English)...")
    tracker._send_status_notification()
    print("✅ Status notifications sent (English)!\n")
    
    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\n📱 Check Telegram for 10 messages:")
    print("\n🇻🇳 VIETNAMESE:")
    print("  1. SL/TP: XAUUSD (2 orders)")
    print("  2. Close: BTCUSD_m (3 orders)")
    print("  3. Status: XAUUSD (2 orders)")
    print("  4. Status: BTCUSD_m (1 order)")
    print("  5. Status: EURUSD (1 order)")
    print("\n🇬🇧 ENGLISH:")
    print("  6. SL/TP: XAUUSD (2 orders)")
    print("  7. Close: BTCUSD_m (3 orders)")
    print("  8. Status: XAUUSD (2 orders)")
    print("  9. Status: BTCUSD_m (1 order)")
    print(" 10. Status: EURUSD (1 order)")
    print("\n")
