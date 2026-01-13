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
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict
import json

# Import notification module
from notification import send_notification
from notification.formatters import format_sltp_changed, format_order_closed, format_order_status
from notification.helpers import get_app_language, calculate_pips, normalize_order_type

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
        
        # Cache digits cho mỗi symbol (lấy từ MT5)
        self._symbol_digits: Dict[str, int] = {}
        
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
            import sys
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "notification_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        return {}
    
    def _reload_config(self):
        """Reload config từ file"""
        self._config = self._load_config()
    
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
    
    def start(self) -> bool:
        """Start tracking thread"""
        with self._lock:
            if self._running:
                logger.info("⚠️ Tracker already running")
                print("⚠️ [OrderTracker] Already running")
                return True
            
            try:
                import MetaTrader5 as mt5
                if not mt5.initialize():
                    logger.error("❌ Cannot initialize MT5")
                    print("❌ [OrderTracker] Cannot initialize MT5")
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
                print(f"📊 [OrderTracker] Loaded {len(self._tracked_orders)} positions: {[o['symbol'] for o in self._tracked_orders.values()]}")
                
                # Start thread
                self._running = True
                self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
                self._thread.start()
                
                logger.info("🚀 Order Tracker started")
                print("🚀 [OrderTracker] Started successfully!")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error starting tracker: {e}")
                print(f"❌ [OrderTracker] Error: {e}")
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
        print("🔄 [OrderTracker] Tracking loop started")
        
        loop_count = 0
        while self._running:
            try:
                loop_count += 1
                
                # Reload config to get latest settings
                self._reload_config()
                
                # Check if any tracking is enabled
                sl_tp_enabled = self._is_sl_tp_tracking_enabled()
                close_enabled = self._is_close_tracking_enabled()
                status_enabled = self._is_order_status_tracking_enabled()
                
                # Log tracking status every 60 seconds (12 loops * 5 seconds)
                if loop_count % 12 == 1:
                    logger.info(f"📊 Tracker status: SL/TP={sl_tp_enabled}, Close={close_enabled}, Status={status_enabled}, Orders={len(self._tracked_orders)}")
                    print(f"📊 [OrderTracker] Loop #{loop_count}: SL/TP={sl_tp_enabled}, Close={close_enabled}, Status={status_enabled}, Orders={len(self._tracked_orders)}")
                
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
            sl_diff = abs(new_sl - old_sl)
            if sl_diff > 0.00001:
                # Calculate pip difference using centralized function
                sl_pips = calculate_pips(symbol, old_sl, new_sl, 'BUY')
                sl_pips = abs(sl_pips)
                
                logger.info(f"🔍 [{symbol}] SL changed: {old_sl:.5f} → {new_sl:.5f} ({sl_pips:.1f} pips, threshold={pip_threshold})")
                print(f"🔍 [OrderTracker] [{symbol}] SL: {old_sl:.5f} → {new_sl:.5f} ({sl_pips:.1f} pips)")
                
                # Only add change if exceeds threshold (or threshold is 0)
                if pip_threshold == 0 or sl_pips >= pip_threshold:
                    changes.append(('SL', old_sl, new_sl))
                    logger.info(f"✅ [{symbol}] SL change QUEUED: {sl_pips:.1f} pips >= {pip_threshold}")
                    print(f"✅ [OrderTracker] [{symbol}] SL QUEUED ({sl_pips:.1f} >= {pip_threshold})")
                else:
                    logger.debug(f"⏭️ [{symbol}] SL change SKIPPED: {sl_pips:.1f} pips < {pip_threshold}")
                    print(f"⏭️ [OrderTracker] [{symbol}] SL SKIPPED ({sl_pips:.1f} < {pip_threshold})")
                
            # Check TP change with pip threshold
            tp_diff = abs(new_tp - old_tp)
            if tp_diff > 0.00001:
                # Calculate pip difference using centralized function
                tp_pips = calculate_pips(symbol, old_tp, new_tp, 'BUY')
                tp_pips = abs(tp_pips)
                
                logger.info(f"🔍 [{symbol}] TP changed: {old_tp:.5f} → {new_tp:.5f} ({tp_pips:.1f} pips, threshold={pip_threshold})")
                print(f"🔍 [OrderTracker] [{symbol}] TP: {old_tp:.5f} → {new_tp:.5f} ({tp_pips:.1f} pips)")
                
                # Only add change if exceeds threshold (or threshold is 0)
                if pip_threshold == 0 or tp_pips >= pip_threshold:
                    changes.append(('TP', old_tp, new_tp))
                    logger.info(f"✅ [{symbol}] TP change QUEUED: {tp_pips:.1f} pips >= {pip_threshold}")
                    print(f"✅ [OrderTracker] [{symbol}] TP QUEUED ({tp_pips:.1f} >= {pip_threshold})")
                else:
                    logger.debug(f"⏭️ [{symbol}] TP change SKIPPED: {tp_pips:.1f} pips < {pip_threshold}")
                    print(f"⏭️ [OrderTracker] [{symbol}] TP SKIPPED ({tp_pips:.1f} < {pip_threshold})")
            
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
            
            # Get close info from MT5 history using standardized function
            close_info = self._get_close_info(ticket, mt5, symbol)
            
            # Fallback: If no data from history, use last tracked data
            if not close_info.get('close_price'):
                last_price = old_data.get('price_current', old_data.get('price_open', 0))
                last_profit = old_data.get('profit', 0)
                close_info['close_price'] = last_price
                close_info['profit'] = last_profit
                close_info['pips'] = 0  # Will be calculated in _send_grouped_close_notification
                logger.info(f"📌 Using last tracked data for #{ticket}: price={last_price}, profit={last_profit}")
            
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
            logger.info(f"📤 Position #{ticket} ({symbol}) closed - profit: {close_info.get('profit', 0):.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error handling position close: {e}")
    
    def _get_close_info(self, ticket: int, mt5, symbol: str = None) -> Dict:
        """Get close information from MT5 history using standardized function
        
        Args:
            ticket: Position ticket number
            mt5: MetaTrader5 module (not used - using trading_history_manager instead)
            symbol: Expected symbol to filter results
        
        Returns:
            Dict with close_price, profit, pips, swap, commission, close_time
        """
        try:
            from trading_history_manager import get_mt5_closed_positions
            
            # Get recent closed positions (last 1 minute to find just-closed order)
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=5)
            
            closed_positions = get_mt5_closed_positions(
                symbol=symbol,
                from_date=start_time,
                to_date=end_time,
                auto_trading_safe=False,
                quick_mode=True,
                max_deals=50
            )
            
            # Find the matching ticket
            for pos in closed_positions:
                if pos.get('ticket') == ticket:
                    logger.info(f"✅ Found closed position #{ticket}: profit={pos.get('profit'):.2f}, pips={pos.get('pips'):.1f}")
                    return {
                        'close_price': pos.get('close_price', 0),
                        'profit': pos.get('profit', 0),
                        'pips': pos.get('pips', 0),
                        'swap': pos.get('swap', 0),
                        'commission': pos.get('commission', 0),
                        'close_time': pos.get('close_time', ''),
                    }
            
            # Not found in recent history
            logger.warning(f"⚠️ No matching closed position found for ticket #{ticket}, symbol={symbol}")
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
        
        # 🛑 ANTI-SPAM CHECK 1: Enforce minimum interval (30 seconds instead of 60)
        time_since_last = current_time - self._last_status_notification
        if time_since_last < 30:  # Reduced from interval to 30 seconds
            return
        
        # 🛑 ANTI-SPAM CHECK 2: If pip threshold is 0, disable status notifications entirely
        if pip_threshold == 0:
            logger.debug(f"Status notifications disabled (pip_threshold=0). Set to 10+ to enable.")
            self._last_status_notification = current_time
            return
        
        # Only process if we have positions to track
        if not self._tracked_orders:
            return
        
        # Log check status
        logger.info(f"📊 Checking status for {len(self._tracked_orders)} orders (threshold={pip_threshold}pips, interval={time_since_last:.0f}s)")
        print(f"📊 [OrderTracker] Status check: {len(self._tracked_orders)} orders, threshold={pip_threshold}pips")
        
        # Check if any order has changed enough pips to warrant notification
        orders_to_notify = []
        
        for ticket, order_data in self._tracked_orders.items():
            symbol = order_data.get('symbol', 'Unknown')
            order_type = order_data.get('type', 'N/A')
            price_open = order_data.get('price_open', 0)
            price_current = order_data.get('price_current', 0)
            
            # Calculate current pips from entry price using centralized function
            current_pips = calculate_pips(symbol, price_open, price_current, normalize_order_type(order_type))
            
            # Get last notified pips (None if never notified - first time)
            last_notified_pips = self._last_notified_pips.get(ticket, None)
            
            # FIRST TIME: Only notify if |current_pips| >= threshold
            if last_notified_pips is None:
                if abs(current_pips) >= pip_threshold:
                    orders_to_notify.append(ticket)
                    self._last_notified_pips[ticket] = current_pips
                    logger.debug(f"Order #{ticket}: FIRST notify (pips {current_pips:.1f} >= {pip_threshold})")
                else:
                    # Still record pips but don't notify yet
                    self._last_notified_pips[ticket] = current_pips
                    logger.debug(f"Order #{ticket}: FIRST skip (pips {current_pips:.1f} < {pip_threshold})")
            else:
                # SUBSEQUENT: Only notify if pip change >= threshold from last notification
                pip_change = abs(current_pips - last_notified_pips)
                
                if pip_change >= pip_threshold:
                    orders_to_notify.append(ticket)
                    # Update last notified pips for next check
                    self._last_notified_pips[ticket] = current_pips
                    logger.debug(f"Order #{ticket}: notify (pips {current_pips:.1f}, change {pip_change:.1f}>={pip_threshold})")
                else:
                    logger.debug(f"Order #{ticket}: skip (pips {current_pips:.1f}, change {pip_change:.1f}<{pip_threshold})")
        
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
            orders_to_notify: List of ticket numbers that triggered notification.
                              Will include ALL orders of symbols that have at least 1 qualifying order.
        """
        try:
            lang = get_app_language()
            
            # 🔧 Get list of triggering tickets
            if orders_to_notify is None:
                orders_to_notify = list(self._tracked_orders.keys())
            
            # 🔧 Find symbols that have at least 1 qualifying order
            symbols_to_notify = set()
            for ticket in orders_to_notify:
                if ticket in self._tracked_orders:
                    order_data = self._tracked_orders[ticket]
                    symbol = order_data.get('symbol', 'Unknown')
                    symbols_to_notify.add(symbol)
            
            if not symbols_to_notify:
                logger.debug(f"No symbols to notify")
                return
            
            # 🔧 Group ALL orders by symbol - include ALL orders of qualifying symbols
            orders_by_symbol: Dict[str, List[Dict]] = defaultdict(list)
            for ticket, order_data in self._tracked_orders.items():
                symbol = order_data.get('symbol', 'Unknown')
                if symbol in symbols_to_notify:
                    orders_by_symbol[symbol].append(order_data)
            
            # Send separate message for each symbol
            for symbol, orders in orders_by_symbol.items():
                self._send_symbol_status_notification(symbol, orders, lang)
            
            total_orders = sum(len(orders) for orders in orders_by_symbol.values())
            total_profit = sum(o.get('profit', 0) for orders in orders_by_symbol.values() for o in orders)
            logger.info(f"📤 Status notification sent: {len(orders_by_symbol)} symbols, {total_orders} orders, P&L: ${total_profit:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error sending status notification: {e}")
    
    def _send_symbol_status_notification(self, symbol: str, orders: List[Dict], lang: str):
        """Send status notification for a single symbol using centralized formatter"""
        try:
            # Prepare orders data with calculated pips for formatter
            orders_with_pips = []
            for order in orders:
                order_type = normalize_order_type(order.get('type', 'N/A'))
                price_open = order.get('price_open', 0)
                price_current = order.get('price_current', 0)
                
                # Calculate pips using centralized function
                pips = calculate_pips(symbol, price_open, price_current, order_type)
                
                orders_with_pips.append({
                    **order,
                    'type': order_type,
                    'pips': pips,
                })
            
            # Use centralized formatter from notification module
            message = format_order_status(symbol, orders_with_pips)
            
            # Add footer
            footer = self._get_footer()
            if footer:
                message += f"\n\n{footer}"
            
            self._send_telegram(message, symbol)
            
        except Exception as e:
            logger.error(f"❌ Error sending symbol status notification: {e}")
    
    def _send_grouped_sl_tp_notification(self, symbol: str, changes_list: List[Dict]):
        """Send grouped SL/TP notification for symbol using centralized formatter"""
        try:
            # Use centralized formatter from notification module
            message = format_sltp_changed(symbol, changes_list)
            
            # Add footer
            footer = self._get_footer()
            if footer:
                message += f"\n\n{footer}"
            
            self._send_telegram(message, symbol)
            logger.info(f"📤 SL/TP notification sent: {symbol} ({len(changes_list)} orders)")
            
        except Exception as e:
            logger.error(f"❌ Error sending grouped SL/TP notification: {e}")
    
    def _send_grouped_close_notification(self, symbol: str, closes_list: List[Dict]):
        """Send grouped close notification for symbol using centralized formatter"""
        try:
            # Prepare order_details with data from MT5 history
            order_details = []
            for close in closes_list:
                close_info = close.get('close_info', {})
                
                # Get profit and pips directly from MT5 history (already calculated correctly)
                profit = close_info.get('profit', 0)
                pips = close_info.get('pips', 0)  # Use pips from trading_history_manager
                price_open = close.get('price_open', 0)
                close_price = close_info.get('close_price', 0)
                
                # Fallback: If pips not available from history, calculate manually
                if pips == 0 and close_price != 0 and price_open != 0:
                    order_type = normalize_order_type(close.get('type', 'N/A'))
                    pips = calculate_pips(symbol, price_open, close_price, order_type)
                
                # Fallback: If close_price still 0, use price_open
                if close_price == 0 or close_price is None:
                    close_price = price_open
                
                # Use centralized normalize_order_type from helpers
                order_type = normalize_order_type(close.get('type', 'N/A'))
                
                order_details.append({
                    **close,
                    'profit': profit,
                    'pips': pips,
                    'close_price': close_price,
                    'type': order_type,
                })
            
            # Use centralized formatter from notification module
            message = format_order_closed(symbol, order_details)
            
            # Add footer
            footer = self._get_footer()
            if footer:
                message += f"\n\n{footer}"
            
            self._send_telegram(message, symbol)
            
            # Determine result text for logging
            total_profit = sum(d.get('profit', 0) for d in order_details)
            result_text = "PROFIT" if total_profit > 0 else ("LOSS" if total_profit < 0 else "BREAKEVEN")
            logger.info(f"📤 Close notification sent: {symbol} ({len(closes_list)} orders) - {result_text}")
            
        except Exception as e:
            logger.error(f"❌ Error sending grouped close notification: {e}")
    
    def _get_footer(self) -> str:
        """Get branded footer from config"""
        try:
            branding = self._config.get('branding', {})
            if not branding.get('enable_custom_footer', True):
                return ""
            
            lang = get_app_language()
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
    
    def _send_telegram(self, message: str, symbol: str = None) -> bool:
        """Send message via notification module
        
        Args:
            message: Message content
            symbol: Symbol for unique message_type (avoid spam protection blocking)
        """
        try:
            # Use symbol-specific message_type to avoid spam protection blocking
            # when sending multiple symbol notifications in same batch
            if symbol:
                message_type = f"order_tracker_{symbol}"
            else:
                message_type = "order_tracker"
            return send_notification(message, message_type=message_type)
        except Exception as e:
            logger.error(f"❌ Error sending notification: {e}")
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
