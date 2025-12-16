"""Minimal, stable MT5 connector.

Purpose: ONLY provide a tiny, dependable surface used by the rest of the app:
  - reconfigure(account, password, server)
  - connect(force_reconnect=False)
  - disconnect()
  - is_connected()
  - get_last_mt5_error()
  - get_essential_account_info()
  - save_essential_account_scan()
  - print_essential_status()

Everything else that previously caused corruption / undefined references has been removed.
Keep this file lean until stability is confirmed.
"""
from __future__ import annotations

import os, time, json, logging, hashlib, threading, glob
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import glob

try:  # MetaTrader5 is optional at import time (unit tests, offline mode)
    import MetaTrader5 as mt5  # type: ignore
    MT5_AVAILABLE = True
except Exception:  # pragma: no cover
    MT5_AVAILABLE = False

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ConnectionState(Enum):
    DISCONNECTED = 'disconnected'
    CONNECTING = 'connecting'
    CONNECTED = 'connected'
    ERROR = 'error'


@dataclass
class MT5Config:
    account: int
    password_h: str
    server: str
    timeout: int = 30000
    max_retries: int = 5
    retry_delay: int = 5  # base seconds, multiplied by attempt for simple backoff


@dataclass 
class MonitoringConfig:
    enabled: bool = False
    interval_seconds: int = 5  # Default: update every 5 seconds for faster monitoring
    save_timestamped: bool = False  # Save with timestamp in filename (False = overwrite single file)
    keep_latest_count: int = 5  # Keep fewer files to avoid spam
    scan_directory: str = 'account_scans'
    base_filename: str = 'mt5_essential_scan'
    auto_start_on_connect: bool = True  # Auto-start monitoring when connected
    clear_old_data: bool = True  # Clear old timestamped files to prevent spam


@dataclass
class ConnectionStats:
    total_connections: int = 0
    failed_connections: int = 0
    last_connection_time: Optional[datetime] = None
    last_error: Optional[str] = None
    uptime_start: Optional[datetime] = None
    monitoring_updates: int = 0
    last_monitoring_update: Optional[datetime] = None


class MT5ConnectionManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self.config: Optional[MT5Config] = None
        self._original_password: Optional[str] = None
        self.state = ConnectionState.DISCONNECTED
        self.stats = ConnectionStats()
        self._last_mt5_error: Any = None
        self._conn_lock = threading.Lock()
        
        # Continuous monitoring attributes
        self.monitoring_config = MonitoringConfig()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_stop_event = threading.Event()
        self._monitoring_lock = threading.Lock()
        
        self._load_env_config()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _hash(self, pw: str) -> str:
        return hashlib.sha256(pw.encode()).hexdigest()

    def _load_env_config(self):
        try:
            a = os.getenv('MT5_ACCOUNT')
            p = os.getenv('MT5_PASSWORD')
            s = os.getenv('MT5_SERVER')
            if not all([a, p, s]):
                return
            acct = int(''.join(filter(str.isdigit, a)) or a)
            if acct <= 0:
                return
            self.config = MT5Config(
                account=acct,
                password_h=self._hash(p),
                server=s,
                timeout=int(os.getenv('MT5_TIMEOUT', '30000')),
                max_retries=int(os.getenv('MT5_MAX_RETRIES', '5')),
                retry_delay=int(os.getenv('MT5_RETRY_DELAY', '5')),
            )
            self._original_password = p
        except Exception:
            pass  # silent – env might not be set yet

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reconfigure(self, account: int | str, password: str, server: str) -> bool:
        """Update credentials at runtime. Does NOT auto-connect."""
        try:
            acct = int(''.join(filter(str.isdigit, str(account))) or str(account))
            if acct <= 0 or not password or not server:
                raise ValueError('Invalid credentials')
            with self._conn_lock:
                self.config = MT5Config(
                    account=acct,
                    password_h=self._hash(password),
                    server=server,
                )
                self._original_password = password
                os.environ['MT5_ACCOUNT'] = str(acct)
                os.environ['MT5_PASSWORD'] = password
                os.environ['MT5_SERVER'] = server
                self.state = ConnectionState.DISCONNECTED
            return True
        except Exception as e:  # pragma: no cover - simple logging path
            logger.error(f'Reconfigure failed: {e}')
            return False

    def connect(self, force_reconnect: bool = False) -> bool:
        if not self.config:
            logger.error('No config set; call reconfigure or set env vars.')
            return False
        if not MT5_AVAILABLE:
            # Offline/testing mode: pretend success
            self.state = ConnectionState.CONNECTED
            self.stats.last_connection_time = datetime.now()
            self.stats.uptime_start = datetime.now()
            
            # Save essential account scan immediately (mock mode)
            try:
                self.save_essential_account_scan()
                logger.info('Saved essential account scan in mock mode')
            except Exception as e:
                logger.error(f'Error saving account scan in mock mode: {e}')
            
            # Auto-start monitoring if enabled (even in mock mode)
            if self.monitoring_config.auto_start_on_connect:
                try:
                    # Auto-configure for fast monitoring (5 seconds, overwrite mode)
                    auto_start_success = self.enable_overwrite_monitoring()
                    if auto_start_success:
                        logger.info('🔄 Auto-started 5-second overwrite monitoring on mock connection')
                    else:
                        logger.warning('Failed to auto-start monitoring on mock connection')
                except Exception as e:
                    logger.error(f'Error auto-starting monitoring in mock mode: {e}')
            
            return True
        with self._conn_lock:
            if self.state == ConnectionState.CONNECTED and not force_reconnect:
                return True
            self.state = ConnectionState.CONNECTING
            self.stats.total_connections += 1
            for attempt in range(1, (self.config.max_retries or 1) + 1):
                try:
                    try:
                        mt5.shutdown()
                    except Exception:
                        pass
                    ok = mt5.initialize(
                        login=self.config.account,
                        password=self._original_password,
                        server=self.config.server,
                        timeout=self.config.timeout,
                    )
                    if ok and self._verify():
                        self.state = ConnectionState.CONNECTED
                        self.stats.last_connection_time = datetime.now()
                        self.stats.uptime_start = datetime.now()
                        
                        # Immediately save account scan on successful connection
                        try:
                            scan_path = self.save_essential_account_scan()
                            if scan_path:
                                logger.info(f'Account scan completed: {scan_path}')
                            else:
                                logger.warning('Failed to save initial account scan')
                        except Exception as e:
                            logger.error(f'Error saving initial account scan: {e}')
                        
                        # Auto-start monitoring if enabled (configure for 5-second overwrite mode)
                        if self.monitoring_config.auto_start_on_connect:
                            try:
                                # Auto-configure for fast monitoring (5 seconds, overwrite mode)
                                auto_start_success = self.enable_overwrite_monitoring()
                                if auto_start_success:
                                    logger.info('🔄 Auto-started 5-second overwrite monitoring on connection')
                                else:
                                    logger.warning('Failed to auto-start monitoring on connection')
                            except Exception as e:
                                logger.error(f'Error auto-starting monitoring: {e}')
                        
                        return True
                    else:
                        self._last_mt5_error = mt5.last_error()
                        mt5.shutdown()
                except Exception:
                    self._last_mt5_error = getattr(mt5, 'last_error', lambda: None)()
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * attempt)
            self.state = ConnectionState.ERROR
            self.stats.failed_connections += 1
            self.stats.last_error = f'Failed after {self.config.max_retries} attempts: {self._last_mt5_error}'
            logger.error(self.stats.last_error)
            return False

    def _verify(self) -> bool:
        if not MT5_AVAILABLE:
            return True
        try:
            acct = mt5.account_info()
            term = mt5.terminal_info()
            return bool(acct and term and self.config and acct.login == self.config.account)
        except Exception:
            return False

    def is_connected(self) -> bool:
        if self.state != ConnectionState.CONNECTED:
            # Try to detect existing MT5 connection even if not connected through this instance
            if MT5_AVAILABLE:
                try:
                    # Check if MT5 is already connected
                    account_info = mt5.account_info()
                    if account_info is not None:
                        # Found existing connection, update our state
                        self.state = ConnectionState.CONNECTED
                        self.stats.last_connection_time = datetime.now()
                        self.stats.uptime_start = datetime.now()
                        logger.info(f'Detected existing MT5 connection to account {account_info.login}')
                        return True
                except Exception:
                    pass
            return False
        if not MT5_AVAILABLE:
            return True
        try:
            return mt5.account_info() is not None
        except Exception:
            return False

    def disconnect(self) -> bool:
        # Stop monitoring first
        self.stop_continuous_monitoring()
        
        if not MT5_AVAILABLE:
            self.state = ConnectionState.DISCONNECTED
            return True
        with self._conn_lock:
            try:
                if self.state == ConnectionState.CONNECTED:
                    mt5.shutdown()
                self.state = ConnectionState.DISCONNECTED
                self.stats.uptime_start = None
                return True
            except Exception as e:
                logger.error(f'Disconnect error: {e}')
                return False

    def get_last_mt5_error(self):
        return self._last_mt5_error

    def get_essential_account_info(self) -> Dict[str, Any]:
        # Check connection first (might detect existing MT5 connection)
        if not self.is_connected():
            return {'error': 'not_connected'}
        if not MT5_AVAILABLE:
            return {'mock': True, 'timestamp': datetime.now().isoformat()}
        try:
            acct = mt5.account_info()
            if not acct:
                return {'error': 'account_info_failed'}
            positions = mt5.positions_get() or []
            orders = mt5.orders_get() or []
            margin_level = (acct.equity / acct.margin * 100) if acct.margin > 0 else 0
            return {
                'login': acct.login,
                'server': acct.server,
                'balance': round(acct.balance, 2),
                'equity': round(acct.equity, 2),
                'profit': round(acct.profit, 2),
                'margin': round(acct.margin, 2),
                'free_margin': round(acct.margin_free, 2),
                'margin_level': round(margin_level, 2),
                'positions': len(positions),
                'orders': len(orders),
                'timestamp': datetime.now().isoformat(),
            }
        except Exception as e:  # pragma: no cover
            return {'error': str(e)}

    def save_essential_account_scan(self, filepath: str | None = None) -> str | None:
        if not self.is_connected():
            return None
        if filepath is None:
            os.makedirs('account_scans', exist_ok=True)
            filepath = 'account_scans/mt5_essential_scan.json'
        acct_summary = self.get_essential_account_info()
        # Enrich with raw positions/orders for downstream analysis (actions, conflict detection)
        active_positions = []
        active_orders = []
        if MT5_AVAILABLE:
            try:
                for p in (mt5.positions_get() or []):
                    try:
                        active_positions.append({
                            'ticket': getattr(p, 'ticket', None),
                            'symbol': getattr(p, 'symbol', None),
                            'volume': getattr(p, 'volume', None),
                            'price_open': getattr(p, 'price_open', None),
                            'price_current': getattr(p, 'price_current', None),
                            'profit': getattr(p, 'profit', None),
                            'swap': getattr(p, 'swap', None),
                            'sl': getattr(p, 'sl', None),
                            'tp': getattr(p, 'tp', None),
                            'type': getattr(p, 'type', None),
                            'time': getattr(p, 'time', None),
                            'comment': getattr(p, 'comment', ''),  # 🎯 CRITICAL for DCA level detection
                        })
                    except Exception:
                        pass
                for o in (mt5.orders_get() or []):
                    try:
                        active_orders.append({
                            'ticket': getattr(o, 'ticket', None),
                            'symbol': getattr(o, 'symbol', None),
                            'volume_current': getattr(o, 'volume_current', None),
                            'volume_initial': getattr(o, 'volume_initial', None),
                            'price_open': getattr(o, 'price_open', None),
                            'sl': getattr(o, 'sl', None),
                            'tp': getattr(o, 'tp', None),
                            'type': getattr(o, 'type', None),
                            'time_setup': getattr(o, 'time_setup', None),
                        })
                    except Exception:
                        pass
            except Exception:
                pass
        payload = {
            'scan_info': {
                'timestamp': datetime.now().isoformat(),
                'account_id': self.config.account if self.config else None,
                'server': self.config.server if self.config else None,
            },
            'account': acct_summary,
            'active_positions': active_positions,
            'active_orders': active_orders,
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return filepath
        except Exception as e:  # pragma: no cover
            logger.error(f'Scan save error: {e}')
            return None

    def print_essential_status(self):  # convenience for CLI/manual debug
        print(json.dumps(self.get_essential_account_info(), indent=2, default=str))

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current bid price for a symbol from MT5.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSD')
            
        Returns:
            Current bid price, or None if error/not connected
        """
        if not self.is_connected():
            logger.warning(f"MT5 not connected, cannot get price for {symbol}")
            return None
            
        try:
            if not MT5_AVAILABLE:
                logger.warning("MT5 module not available")
                return None
                
            # Get symbol info tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"Failed to get tick for symbol {symbol}")
                return None
                
            # Return bid price for most accurate entry
            return float(tick.bid)
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def get_multiple_current_prices(self, symbols: list[str]) -> Dict[str, Optional[float]]:
        """Get current prices for multiple symbols efficiently.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary mapping symbol to current price (or None if error)
        """
        prices = {}
        if not self.is_connected():
            logger.warning("MT5 not connected, cannot get prices")
            return {symbol: None for symbol in symbols}
            
        for symbol in symbols:
            prices[symbol] = self.get_current_price(symbol)
            
        return prices

    # ------------------------------------------------------------------
    # Continuous monitoring functionality
    # ------------------------------------------------------------------
    def enable_fast_monitoring(self, overwrite_mode: bool = True) -> bool:
        """Enable fast 5-second monitoring with optimized settings.
        
        Args:
            overwrite_mode: If True, overwrites single file. If False, creates timestamped files.
        """
        try:
            success = self.configure_monitoring(
                enabled=True,
                interval_seconds=5,
                save_timestamped=not overwrite_mode,  # Invert for overwrite mode
                keep_latest_count=5 if overwrite_mode else 20,
                auto_start_on_connect=True
            )
            if success and self.is_connected():
                start_success = self.start_continuous_monitoring()
                if start_success:
                    mode_text = "overwrite mode" if overwrite_mode else "timestamped mode"
                    logger.info(f'🚀 Fast 5-second monitoring enabled ({mode_text})!')
                    return True
                else:
                    logger.warning('Fast monitoring configured but failed to start')
                    return False
            elif success:
                logger.info('🔧 Fast monitoring configured (will start on next connection)')
                return True
            else:
                logger.error('Failed to configure fast monitoring')
                return False
        except Exception as e:
            logger.error(f'Error enabling fast monitoring: {e}')
            return False

    def enable_overwrite_monitoring(self) -> bool:
        """Enable monitoring that overwrites a single file (no spam)."""
        return self.enable_fast_monitoring(overwrite_mode=True)

    def enable_history_monitoring(self) -> bool:
        """Enable monitoring that keeps timestamped history files."""
        return self.enable_fast_monitoring(overwrite_mode=False)

    def configure_monitoring(self, 
                           enabled: bool = True,
                           interval_seconds: int = 5,  # Default 5 seconds for fast monitoring
                           save_timestamped: bool = True,
                           keep_latest_count: int = 50,
                           auto_start_on_connect: bool = True) -> bool:
        """Configure continuous account monitoring settings."""
        try:
            with self._monitoring_lock:
                self.monitoring_config.enabled = enabled
                self.monitoring_config.interval_seconds = max(1, interval_seconds)  # Min 1 second for ultra-fast
                self.monitoring_config.save_timestamped = save_timestamped
                self.monitoring_config.keep_latest_count = max(1, keep_latest_count)
                self.monitoring_config.auto_start_on_connect = auto_start_on_connect
                self.monitoring_config.clear_old_data = False  # Keep historical data
            logger.info(f'🔄 Monitoring configured: enabled={enabled}, interval={interval_seconds}s, auto_start={auto_start_on_connect}')
            return True
        except Exception as e:
            logger.error(f'Failed to configure monitoring: {e}')
            return False

    def start_continuous_monitoring(self) -> bool:
        """Start continuous account monitoring in background thread."""
        # Check connection first - will work with existing connection
        if not self.is_connected():
            logger.warning('Cannot start monitoring: not connected to MT5. Please connect first.')
            return False
            
        with self._monitoring_lock:
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                logger.warning('Monitoring already running')
                return True
                
            self.monitoring_config.enabled = True
            self._monitoring_stop_event.clear()
            
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_worker,
                name='MT5-Monitor',
                daemon=True
            )
            self._monitoring_thread.start()
            
        logger.info(f'Started continuous monitoring (interval: {self.monitoring_config.interval_seconds}s)')
        return True

    def stop_continuous_monitoring(self) -> bool:
        """Stop continuous account monitoring."""
        with self._monitoring_lock:
            if not self._monitoring_thread or not self._monitoring_thread.is_alive():
                logger.info('Monitoring not running')
                return True
                
            self.monitoring_config.enabled = False
            self._monitoring_stop_event.set()
            
        # Wait for thread to finish (max 10 seconds)
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
            if self._monitoring_thread.is_alive():
                logger.warning('Monitoring thread did not stop cleanly')
                return False
                
        logger.info('Stopped continuous monitoring')
        return True

    def is_monitoring_active(self) -> bool:
        """Check if continuous monitoring is currently active."""
        with self._monitoring_lock:
            return (self.monitoring_config.enabled and 
                   self._monitoring_thread and 
                   self._monitoring_thread.is_alive() and
                   not self._monitoring_stop_event.is_set())

    def _monitoring_worker(self):
        """Background worker for continuous monitoring."""
        logger.info('Monitoring worker started')
        
        # Store previous positions for change detection
        previous_positions = {}
        
        while not self._monitoring_stop_event.is_set():
            try:
                if not self.is_connected():
                    logger.warning('Lost MT5 connection, stopping monitoring')
                    break
                    
                # Check for position changes BEFORE saving scan
                try:
                    self._check_position_changes(previous_positions)
                except Exception as e:
                    logger.error(f'Error checking position changes: {e}')
                    
                # Save account scan with timestamp
                filepath = self._save_timestamped_scan()
                if filepath:
                    self.stats.monitoring_updates += 1
                    self.stats.last_monitoring_update = datetime.now()
                    logger.debug(f'Monitoring update #{self.stats.monitoring_updates}: {filepath}')
                    
                    # Cleanup old files
                    self._cleanup_old_scans()
                else:
                    logger.warning('Failed to save monitoring scan')
                    
            except Exception as e:
                logger.error(f'Monitoring worker error: {e}')
                
            # Wait for next interval or stop event
            if self._monitoring_stop_event.wait(timeout=self.monitoring_config.interval_seconds):
                break  # Stop event was set
                
        logger.info('Monitoring worker stopped')

    def _check_position_changes(self, previous_positions):
        """Check for position changes and send notifications"""
        try:
            if not getattr(self, 'mock_mode', False):
                import MetaTrader5 as mt5
                current_positions = mt5.positions_get()
                
                if current_positions is None:
                    return
                    
                # Convert to dict for easier comparison
                current_pos_dict = {}
                for pos in current_positions:
                    ticket = pos.ticket
                    current_pos_dict[ticket] = {
                        'symbol': pos.symbol,
                        'type': pos.type,
                        'volume': pos.volume,
                        'sl': pos.sl,
                        'tp': pos.tp,
                        'profit': pos.profit,
                        'price_open': pos.price_open
                    }
                
                # Check for changes
                self._detect_and_notify_changes(previous_positions, current_pos_dict)
                
                # Update previous positions for next check
                previous_positions.clear()
                previous_positions.update(current_pos_dict)
                
        except Exception as e:
            logger.error(f'Error checking position changes: {e}')
    
    def _detect_and_notify_changes(self, previous_positions, current_positions):
        """Detect changes and send appropriate notifications"""
        try:
            # Import notification system
            from unified_notification_system import get_unified_notification_system
            notification_system = get_unified_notification_system()
            
            # Check settings
            settings = notification_system.config.get('settings', {})
            
            # Check for new positions (order tracking)
            if settings.get('track_order_updates', False):
                for ticket, pos in current_positions.items():
                    if ticket not in previous_positions:
                        logger.info(f"📈 New position detected: {pos['symbol']}")
                        notification_system.send_order_update_notification(pos)
            
            # Check for closed positions
            if settings.get('notify_order_close', False):
                for ticket, pos in previous_positions.items():
                    if ticket not in current_positions:
                        logger.info(f"🔒 Position closed: {pos['symbol']}")
                        notification_system.send_order_close_notification(pos)
            
            # Check for SL/TP changes
            if settings.get('notify_sl_tp_changes', False):
                for ticket, current_pos in current_positions.items():
                    if ticket in previous_positions:
                        prev_pos = previous_positions[ticket]
                        
                        # Check if SL or TP changed
                        sl_changed = prev_pos['sl'] != current_pos['sl']
                        tp_changed = prev_pos['tp'] != current_pos['tp']
                        
                        if sl_changed or tp_changed:
                            logger.info(f"🛡️ SL/TP changed for {current_pos['symbol']}")
                            notification_system.send_sl_tp_change_notification(current_pos)
                            
        except Exception as e:
            logger.error(f'Error detecting position changes: {e}')

    def _save_timestamped_scan(self) -> str | None:
        """Save account scan with timestamp in filename and optional data cleanup."""
        if not self.is_connected():
            return None
            
        try:
            os.makedirs(self.monitoring_config.scan_directory, exist_ok=True)
            
            # Clear old data if enabled
            if self.monitoring_config.clear_old_data:
                self._clear_old_scan_files()
            
            if self.monitoring_config.save_timestamped:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'{self.monitoring_config.base_filename}_{timestamp}.json'
            else:
                filename = f'{self.monitoring_config.base_filename}.json'
                
            filepath = os.path.join(self.monitoring_config.scan_directory, filename)
            return self.save_essential_account_scan(filepath)
            
        except Exception as e:
            logger.error(f'Failed to save timestamped scan: {e}')
            return None

    def _clear_old_scan_files(self):
        """Clear all existing scan files before creating new one."""
        try:
            if not os.path.exists(self.monitoring_config.scan_directory):
                return
                
            pattern = f"{self.monitoring_config.base_filename}*.json"
            scan_files = glob.glob(os.path.join(self.monitoring_config.scan_directory, pattern))
            
            for file_path in scan_files:
                try:
                    os.remove(file_path)
                    logger.debug(f"🗑️ Cleared old scan file: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Error clearing old scan files: {e}")

    def _cleanup_old_scans(self):
        """Clean up old scan files, keeping only the latest N files."""
        try:
            pattern = os.path.join(
                self.monitoring_config.scan_directory,
                f'{self.monitoring_config.base_filename}_*.json'
            )
            scan_files = glob.glob(pattern)
            
            if len(scan_files) <= self.monitoring_config.keep_latest_count:
                return  # No cleanup needed
                
            # Sort by modification time (newest first)
            scan_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            
            # Keep the latest N files, delete the rest
            files_to_delete = scan_files[self.monitoring_config.keep_latest_count:]
            
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    logger.debug(f'Cleaned up old scan: {os.path.basename(file_path)}')
                except Exception as e:
                    logger.warning(f'Failed to delete {file_path}: {e}')
                    
        except Exception as e:
            logger.error(f'Cleanup error: {e}')

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status and statistics."""
        return {
            'monitoring_enabled': self.monitoring_config.enabled,
            'monitoring_active': self.is_monitoring_active(),
            'interval_seconds': self.monitoring_config.interval_seconds,
            'save_timestamped': self.monitoring_config.save_timestamped,
            'keep_latest_count': self.monitoring_config.keep_latest_count,
            'scan_directory': self.monitoring_config.scan_directory,
            'auto_start_on_connect': self.monitoring_config.auto_start_on_connect,
            'total_updates': self.stats.monitoring_updates,
            'last_update': self.stats.last_monitoring_update.isoformat() if self.stats.last_monitoring_update else None,
        }

    def auto_start_monitoring_if_connected(self) -> bool:
        """Check if connected and auto-start monitoring if enabled and not already running."""
        if not self.monitoring_config.auto_start_on_connect:
            return False
            
        if not self.is_connected():
            logger.info('Auto-start monitoring: Not connected to MT5')
            return False
            
        if self.is_monitoring_active():
            logger.info('Auto-start monitoring: Already running')
            return True
            
        # Start monitoring for existing connection
        success = self.start_continuous_monitoring()
        if success:
            logger.info('Auto-started monitoring for existing MT5 connection')
        else:
            logger.warning('Failed to auto-start monitoring for existing connection')
        return success

    def force_start_monitoring(self) -> bool:
        """Force start monitoring without auto-start check (for manual trigger)."""
        if not self.is_connected():
            logger.warning('Cannot start monitoring: not connected to MT5')
            return False
            
        return self.start_continuous_monitoring()

    def cleanup_scan_files(self, keep_latest: int = 1) -> int:
        """Clean up old scan files, keeping only the specified number of latest files.
        
        Args:
            keep_latest: Number of latest files to keep (default: 1)
            
        Returns:
            Number of files deleted
        """
        try:
            if not os.path.exists(self.monitoring_config.scan_directory):
                logger.info('No scan directory to clean up')
                return 0
                
            # Get all scan files
            pattern = os.path.join(
                self.monitoring_config.scan_directory,
                f'{self.monitoring_config.base_filename}*.json'
            )
            scan_files = glob.glob(pattern)
            
            if len(scan_files) <= keep_latest:
                logger.info(f'Only {len(scan_files)} scan files found, no cleanup needed')
                return 0
                
            # Sort by modification time (newest first)
            scan_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            
            # Keep the latest N files, delete the rest
            files_to_delete = scan_files[keep_latest:]
            deleted_count = 0
            
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.debug(f'🗑️ Deleted: {os.path.basename(file_path)}')
                except Exception as e:
                    logger.warning(f'Failed to delete {file_path}: {e}')
                    
            logger.info(f'🧹 Cleanup completed: {deleted_count} files deleted, {keep_latest} files kept')
            return deleted_count
            
        except Exception as e:
            logger.error(f'Error during cleanup: {e}')
            return 0

    def cleanup_all_scans_except_latest(self) -> int:
        """Clean up all scan files except the latest one."""
        return self.cleanup_scan_files(keep_latest=1)

    def force_overwrite_mode(self) -> bool:
        """Force switch to overwrite mode and cleanup existing files."""
        try:
            # Stop current monitoring
            self.stop_continuous_monitoring()
            
            # Cleanup old files (keep only latest 1)
            deleted = self.cleanup_all_scans_except_latest()
            
            # Enable overwrite monitoring
            success = self.enable_overwrite_monitoring()
            
            if success:
                logger.info(f'🔄 Switched to overwrite mode, cleaned {deleted} files')
            else:
                logger.error('Failed to switch to overwrite mode')
                
            return success
            
        except Exception as e:
            logger.error(f'Error forcing overwrite mode: {e}')
            return False

    def get_connection_and_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive status of connection and monitoring."""
        status = {
            'connected': self.is_connected(),
            'monitoring_active': self.is_monitoring_active(),
            'auto_start_enabled': self.monitoring_config.auto_start_on_connect,
            'clear_old_data': self.monitoring_config.clear_old_data,
        }
        
        if self.config:
            status['account'] = self.config.account
            status['server'] = self.config.server
            
        if self.is_connected():
            try:
                account_info = self.get_essential_account_info()
                if 'error' not in account_info:
                    status['balance'] = account_info.get('balance')
                    status['equity'] = account_info.get('equity')
                    status['open_positions'] = len(account_info.get('positions', []))
                    status['pending_orders'] = len(account_info.get('orders', []))
            except Exception:
                pass
                
        return status

    def force_account_switch(self, account: str, password: str, server: str) -> bool:
        """
        🔄 Force switch to new MT5 account - Complete workflow
        
        Args:
            account: New MT5 account number
            password: New MT5 password
            server: New MT5 server
        
        Returns:
            bool: True if switch successful, False otherwise
        """
        
        logger.info("🔄 Starting force account switch...")
        logger.info(f"   Target Account: {account}")
        logger.info(f"   Target Server: {server}")
        
        try:
            # Step 1: Stop monitoring
            logger.info("1️⃣ Stopping monitoring...")
            stop_success = self.stop_continuous_monitoring()
            logger.info(f"   Monitoring stop: {'✅' if stop_success else '❌'}")
            
            # Step 2: Disconnect current connection
            logger.info("2️⃣ Disconnecting...")
            disconnect_success = self.disconnect()
            logger.info(f"   Disconnect: {'✅' if disconnect_success else '❌'}")
            
            # Step 3: Clear old scan files
            logger.info("3️⃣ Clearing old scan files...")
            cleared_count = self._clear_old_scan_files()
            logger.info(f"   Cleared {cleared_count} scan files")
            
            # Step 4: Update environment variables
            logger.info("4️⃣ Updating environment variables...")
            os.environ['MT5_ACCOUNT'] = str(account)
            os.environ['MT5_PASSWORD'] = password
            os.environ['MT5_SERVER'] = server
            logger.info("   Environment variables updated")
            
            # Step 5: Reconfigure with new credentials
            logger.info("5️⃣ Reconfiguring connection...")
            reconfig_success = self.reconfigure(account, password, server)
            if not reconfig_success:
                logger.error("   ❌ Reconfigure failed")
                return False
            logger.info("   ✅ Reconfigure successful")
            
            # Step 6: Force reconnect
            logger.info("6️⃣ Force reconnecting...")
            connect_success = self.connect(force_reconnect=True)
            if not connect_success:
                logger.error("   ❌ Connection failed")
                return False
            logger.info("   ✅ Connection successful")
            
            # Step 7: Verify new connection
            logger.info("7️⃣ Verifying connection...")
            account_info = self.get_essential_account_info()
            if 'error' in account_info:
                logger.error(f"   ❌ Verification failed: {account_info.get('error')}")
                return False
            
            logger.info("   ✅ Connection verified!")
            logger.info(f"      Account: {account_info.get('login')}")
            logger.info(f"      Server: {account_info.get('server')}")
            logger.info(f"      Balance: {account_info.get('balance')}")
            
            # Step 8: Save new account scan
            logger.info("8️⃣ Saving account scan...")
            scan_path = self.save_essential_account_scan()
            if scan_path:
                logger.info(f"   ✅ Scan saved: {scan_path}")
            else:
                logger.warning("   ⚠️ Failed to save scan")
            
            # Step 9: Restart monitoring
            logger.info("9️⃣ Restarting monitoring...")
            monitoring_success = self.enable_overwrite_monitoring()
            logger.info(f"   Monitoring: {'✅' if monitoring_success else '❌'}")
            
            logger.info("🎉 Account switch completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Force account switch failed: {e}")
            return False
    
    def _clear_old_scan_files(self) -> int:
        """Clear old account scan files"""
        import glob
        cleared_count = 0
        
        try:
            scan_dir = self.monitoring_config.scan_directory
            if os.path.exists(scan_dir):
                scan_files = glob.glob(os.path.join(scan_dir, "*.json"))
                for file_path in scan_files:
                    try:
                        os.remove(file_path)
                        cleared_count += 1
                        logger.debug(f"Deleted scan file: {os.path.basename(file_path)}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error clearing scan files: {e}")
        
        return cleared_count
    
    def quick_account_setup(self, account: str = None, password: str = None, server: str = None) -> bool:
        """
        🚀 Quick account setup từ environment hoặc parameters
        
        Args:
            account: MT5 account (optional, uses env if not provided)
            password: MT5 password (optional, uses env if not provided)  
            server: MT5 server (optional, uses env if not provided)
        
        Returns:
            bool: True if setup successful
        """
        
        # Get from environment if not provided
        account = account or os.getenv('MT5_ACCOUNT')
        password = password or os.getenv('MT5_PASSWORD')
        server = server or os.getenv('MT5_SERVER')
        
        if not all([account, password, server]):
            logger.error("❌ Missing account credentials")
            logger.error(f"   Account: {'✅' if account else '❌ NOT SET'}")
            logger.error(f"   Password: {'✅' if password else '❌ NOT SET'}")
            logger.error(f"   Server: {'✅' if server else '❌ NOT SET'}")
            return False
        
        logger.info("🚀 Quick account setup starting...")
        return self.force_account_switch(account, password, server)
    
    def clear_cache_and_refresh(self) -> bool:
        """
        🧹 Clear cache và refresh connection
        
        Returns:
            bool: True if refresh successful
        """
        
        logger.info("🧹 Clearing cache and refreshing...")
        
        try:
            # Stop monitoring và disconnect
            self.stop_continuous_monitoring()
            self.disconnect()
            
            # Clear scan files
            cleared_count = self._clear_old_scan_files()
            logger.info(f"Cleared {cleared_count} scan files")
            
            # Check for environment variables
            account = os.getenv('MT5_ACCOUNT')
            password = os.getenv('MT5_PASSWORD')
            server = os.getenv('MT5_SERVER')
            
            if all([account, password, server]):
                logger.info("Found environment credentials - reconnecting...")
                return self.force_account_switch(account, password, server)
            else:
                logger.warning("No environment credentials found")
                return False
                
        except Exception as e:
            logger.error(f"Cache clear and refresh failed: {e}")
            return False
    
    def print_account_status(self) -> None:
        """
        📊 Print comprehensive account status
        """
        
        print("📊 MT5 ACCOUNT STATUS")
        print("=" * 50)
        
        # Connection status
        status = self.get_connection_and_monitoring_status()
        print(f"🔗 Connected: {status.get('connected', False)}")
        print(f"📱 Monitoring: {status.get('monitoring_active', False)}")
        
        if status.get('connected'):
            print(f"🏦 Account: {status.get('account', 'N/A')}")
            print(f"🖥️ Server: {status.get('server', 'N/A')}")
            print(f"💰 Balance: {status.get('balance', 'N/A')}")
            print(f"📊 Equity: {status.get('equity', 'N/A')}")
            print(f"📍 Positions: {status.get('open_positions', 'N/A')}")
        else:
            print("❌ Not connected")
        
        # Environment variables
        print(f"\n🌍 ENVIRONMENT:")
        env_account = os.getenv('MT5_ACCOUNT')
        env_server = os.getenv('MT5_SERVER')
        env_password = os.getenv('MT5_PASSWORD')
        
        print(f"   Account: {env_account or 'NOT SET'}")
        print(f"   Server: {env_server or 'NOT SET'}")
        print(f"   Password: {'SET' if env_password else 'NOT SET'}")
        
        # Check for mismatches
        if status.get('connected') and env_account:
            current_account = status.get('account')
            if str(current_account) != str(env_account):
                print(f"\n⚠️ ACCOUNT MISMATCH:")
                print(f"   Connected: {current_account}")
                print(f"   Environment: {env_account}")
                print(f"   💡 Run: connection.force_account_switch()")
        
        # Scan files
        print(f"\n📂 SCAN FILES:")
        scan_dir = self.monitoring_config.scan_directory
        if os.path.exists(scan_dir):
            import glob
            scan_files = glob.glob(os.path.join(scan_dir, "*.json"))
            if scan_files:
                latest_file = max(scan_files, key=os.path.getmtime)
                mod_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
                print(f"   📄 Latest: {os.path.basename(latest_file)} ({mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
            else:
                print("   ℹ️ No scan files")
        else:
            print("   ℹ️ No scan directory")


def get_mt5_connection() -> MT5ConnectionManager:
    return MT5ConnectionManager()


if __name__ == '__main__':  # Enhanced CLI interface
    import sys
    
    mgr = MT5ConnectionManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'status':
            # Show current status
            mgr.print_account_status()
            
        elif command == 'switch':
            # Force account switch
            if len(sys.argv) >= 5:
                account = sys.argv[2]
                password = sys.argv[3]
                server = sys.argv[4]
                print(f"🔄 Switching to account: {account} on {server}")
                success = mgr.force_account_switch(account, password, server)
                if success:
                    print("✅ Account switch completed!")
                else:
                    print("❌ Account switch failed!")
            else:
                # Try from environment
                print("🔄 Attempting switch from environment variables...")
                success = mgr.quick_account_setup()
                if success:
                    print("✅ Account switch completed!")
                else:
                    print("❌ Account switch failed - check environment variables")
                    print("Usage: python mt5_connector.py switch <account> <password> <server>")
        
        elif command == 'clear':
            # Clear cache and refresh
            print("🧹 Clearing cache and refreshing...")
            success = mgr.clear_cache_and_refresh()
            if success:
                print("✅ Cache cleared and refreshed!")
            else:
                print("❌ Cache clear failed")
        
        elif command == 'connect':
            # Simple connect test
            if mgr.connect():
                print('✅ Connected OK')
                mgr.print_essential_status()
            else:
                print(f'❌ Connect failed: {mgr.get_last_mt5_error()}')
        
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands:")
            print("  status  - Show account status")
            print("  switch  - Switch account (from env or parameters)")
            print("  clear   - Clear cache and refresh")
            print("  connect - Test connection")
    
    else:
        print("🔧 MT5 CONNECTION MANAGER")
        print("=" * 60)
        print("Available commands:")
        print("  python mt5_connector.py status")
        print("  python mt5_connector.py switch [account] [password] [server]")
        print("  python mt5_connector.py clear")
        print("  python mt5_connector.py connect")
        
        print(f"\n🤖 Auto-checking status...")
        mgr.print_account_status()
        
        # Auto-detect issues và suggest solutions
        status = mgr.get_connection_and_monitoring_status()
        env_account = os.getenv('MT5_ACCOUNT')
        
        if not status.get('connected') and env_account:
            print(f"\n💡 SUGGESTION: Environment variables found but not connected")
            print(f"   Run: python mt5_connector.py switch")
        elif not env_account:
            print(f"\n💡 SUGGESTION: Set environment variables first")
            print(f"   $env:MT5_ACCOUNT = 'your_account'")
            print(f"   $env:MT5_PASSWORD = 'your_password'") 
            print(f"   $env:MT5_SERVER = 'your_server'")
        elif status.get('connected'):
            current_account = status.get('account')
            if str(current_account) != str(env_account):
                print(f"\n💡 SUGGESTION: Account mismatch detected")
                print(f"   Run: python mt5_connector.py switch")

# End of minimal file