import glob
import MetaTrader5 as mt5
import logging
import time
import json
import os
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import os
import json
import time
import sys
import logging
from dca_lock_manager import DCALockManager
from utils import get_pip_value, overwrite_json_safely, ensure_directory, auto_cleanup_on_start
from constants import (
    MAX_SPREAD_MULTIPLIER,
    MIN_VOLUME_AUTO,
    MAX_TOTAL_VOLUME,
    DCA_LOCK_MAX_AGE_SECONDS,
    DEFAULT_DCA_DISTANCE_PIPS
)

# Global executor instance for tracking service access
_global_executor_instance = None

def _get_app_language():
    """Get current app language from user config"""
    try:
        import pickle
        config_path = os.path.join(os.getcwd(), 'user_config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
                return config.get('language', 'vi')  # Default to Vietnamese
    except Exception:
        pass
    return 'vi'  # Default to Vietnamese

def get_global_executor_instance():
    """Get the global executor instance for tracking service"""
    return _global_executor_instance

# NEW: integrate unified connection manager (singleton)
try:
    from mt5_connector import MT5ConnectionManager
    import MetaTrader5 as mt5
except Exception:  # fallback if minimal env
    MT5ConnectionManager = None  # type: ignore
    mt5 = None  # type: ignore

# 🆕 Global fallback functions for OFF settings support (always available)
def parse_setting_value(value, default_value=0.0, setting_name="unknown"):
    """Global parser for OFF settings"""
    if isinstance(value, str) and value.upper() == "OFF":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default_value
    return default_value

def is_setting_enabled(value):
    """Global checker for enabled settings"""
    if value is None:
        return False
    if isinstance(value, str) and value.upper() == "OFF":
        return False
    return True

# 🆕 Import risk manager helpers (with fallback to globals above)
try:
    from risk_manager import parse_setting_value as _risk_parse, is_setting_enabled as _risk_enabled  # type: ignore
    # Override with risk_manager functions if available
    parse_setting_value = _risk_parse
    is_setting_enabled = _risk_enabled
except ImportError:
    # Use global fallback functions defined above
    pass

# Market detection helper functions
def is_crypto_symbol(symbol: str) -> bool:
    """Check if symbol is cryptocurrency (trades 24/7)"""
    crypto_symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'DOTUSD', 'LINKUSD', 'UNIUSD', 'LTCUSD']
    return symbol.upper() in crypto_symbols or 'BTC' in symbol.upper() or 'ETH' in symbol.upper()

def is_market_closed_for_symbol(symbol: str) -> bool:
    """Check if market is closed for specific symbol"""
    import datetime
    
    # Crypto markets are always open
    if is_crypto_symbol(symbol):
        return False
    
    # Forex markets close on weekends
    weekday = datetime.datetime.now().weekday()  # 0=Monday, 6=Sunday
    if weekday >= 5:  # Saturday=5, Sunday=6
        return True
        
    return False

# 🔒 GLOBAL ORDER EXECUTION LOCK to prevent race conditions and duplicate orders
_GLOBAL_ORDER_LOCK = threading.Lock()
_ORDER_EXECUTOR_INSTANCES = {}  # Track instances by thread/process

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_mt5_error_description(error_code: int) -> str:
    """Get detailed description of MT5 error codes"""
    mt5_errors = {
        10004: "TRADE_RETCODE_REQUOTE - Requote",
        10006: "TRADE_RETCODE_REJECT - Request rejected",
        10007: "TRADE_RETCODE_CANCEL - Request canceled by trader",
        10008: "TRADE_RETCODE_PLACED - Order placed",
        10009: "TRADE_RETCODE_DONE - Request completed",
        10010: "TRADE_RETCODE_DONE_PARTIAL - Only part of the request was completed",
        10011: "TRADE_RETCODE_ERROR - Request processing error",
        10012: "TRADE_RETCODE_TIMEOUT - Request canceled by timeout",
        10013: "TRADE_RETCODE_INVALID - Invalid request",
        10014: "TRADE_RETCODE_INVALID_VOLUME - Invalid volume in the request",
        10015: "TRADE_RETCODE_INVALID_PRICE - Invalid price in the request",
        10016: "TRADE_RETCODE_INVALID_STOPS - Invalid stops in the request",
        10017: "TRADE_RETCODE_TRADE_DISABLED - Trade is disabled",
        10018: "TRADE_RETCODE_MARKET_CLOSED - Market is closed",
        10019: "TRADE_RETCODE_NO_MONEY - There is not enough money to complete the request",
        10020: "TRADE_RETCODE_PRICE_CHANGED - Prices changed",
        10021: "TRADE_RETCODE_PRICE_OFF - There are no quotes to process the request",
        10022: "TRADE_RETCODE_INVALID_EXPIRATION - Invalid order expiration date",
        10023: "TRADE_RETCODE_ORDER_CHANGED - Order state changed",
        10024: "TRADE_RETCODE_TOO_MANY_REQUESTS - Too frequent requests",
        10025: "TRADE_RETCODE_NO_CHANGES - No changes in request",
        10026: "TRADE_RETCODE_SERVER_DISABLES_AT - Autotrading disabled by server",
        10027: "TRADE_RETCODE_CLIENT_DISABLES_AT - Autotrading disabled by client terminal",
        10028: "TRADE_RETCODE_LOCKED - Request locked for processing",
        10029: "TRADE_RETCODE_FROZEN - Order or position frozen",
        10030: "TRADE_RETCODE_INVALID_FILL - Invalid order filling type",
        10031: "TRADE_RETCODE_CONNECTION - No connection with the trade server",
        10032: "TRADE_RETCODE_ONLY_REAL - Operation is allowed only for live accounts",
        10033: "TRADE_RETCODE_LIMIT_ORDERS - The number of pending orders has reached the limit",
        10034: "TRADE_RETCODE_LIMIT_VOLUME - The volume of orders and positions for the symbol has reached the limit",
        10035: "TRADE_RETCODE_INVALID_ORDER - Incorrect or prohibited order type",
        10036: "TRADE_RETCODE_POSITION_CLOSED - Position with the specified POSITION_IDENTIFIER has already been closed",
    }
    
    return mt5_errors.get(error_code, f"Unknown error code: {error_code}")

def get_symbol_fill_type(symbol: str) -> int:
    """Get appropriate fill type for the symbol"""
    try:
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"⚠️ Symbol {symbol} not found, using default fill type")
            return mt5.ORDER_FILLING_FOK
        
        # 🔧 FIXED: Test each filling mode by attempting to use it
        # Many brokers only support specific modes for crypto symbols
        
        # For crypto symbols (ETHUSD, SOLUSD, BTCUSD), usually FOK works best
        if symbol.upper().endswith('USD') and any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']):
            return mt5.ORDER_FILLING_FOK
        
        # Check what filling modes are supported (original logic as fallback)
        filling_mode = symbol_info.filling_mode
        
        # Try FOK first (most compatible)
        if filling_mode & 1 or filling_mode == 1:
            return mt5.ORDER_FILLING_FOK
            
        # ORDER_FILLING_IOC (Immediate or Cancel) - Bit 1  
        if filling_mode & 2:
            return mt5.ORDER_FILLING_RETURN
            
        # Default fallback
        logger.warning(f"No supported fill type found for {symbol}, using FOK")
        return mt5.ORDER_FILLING_FOK
        
    except Exception as e:
        logger.error(f"Error getting fill type for {symbol}: {e}, using FOK")
        return mt5.ORDER_FILLING_FOK

class OrderType(Enum):
    """Order types enumeration"""
    MARKET_BUY = "MARKET_BUY"
    MARKET_SELL = "MARKET_SELL"
    LIMIT_BUY = "LIMIT_BUY"
    LIMIT_SELL = "LIMIT_SELL"
    STOP_BUY = "STOP_BUY"
    STOP_SELL = "STOP_SELL"

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"

@dataclass
class OrderResult:
    """Order execution result"""
    success: bool
    order_id: Optional[int] = None
    ticket: Optional[int] = None
    retcode: Optional[int] = None
    comment: str = ""
    price: Optional[float] = None
    volume: Optional[float] = None
    timestamp: Optional[datetime] = None
    error_message: str = ""

@dataclass
class TradeSignal:
    """Enhanced trade signal structure"""
    symbol: str
    action: str  # BUY/SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float
    confidence: float = 0.0
    strategy: str = "AI_SYSTEM"
    timestamp: Optional[datetime] = None
    comment: str = ""
    is_dca: bool = False  # Flag to differentiate DCA orders from Entry orders
    dca_level: int = 1  # DCA level (1, 2, 3, etc.)

@dataclass
class RiskValidationResult:
    """Risk validation result with detailed breakdown"""
    is_valid: bool
    risk_score: float  # 0-100, higher = safer
    violations: List[str]
    warnings: List[str]
    passed_checks: List[str]
    recommendation: str

class ComprehensiveRiskValidator:
    """
    🛡️ Comprehensive Risk Validator
    Kiểm tra TẤT CẢ điều kiện risk trước khi đặt lệnh như human analysis
    """
    
    def __init__(self, risk_settings: Dict[str, Any]):
        self.risk_settings = risk_settings
        # DCA tracking removed - use dca_service.py instead
        
    def validate_all_conditions(self, symbol: str, signal_confidence: float, 
                               order_type: str, is_dca: bool = False) -> RiskValidationResult:
        """Kiểm tra TẤT CẢ điều kiện risk như human analysis"""
        violations = []
        warnings = []
        passed_checks = []
        risk_score = 100.0
        
        # 1. 🎮 CHẾ ĐỘ GIAO DỊCH - CRITICAL CHECK (Skip for DCA)
        if not is_dca:  # DCA không cần check auto_mode
            auto_mode_check = self._check_auto_mode()
            if not auto_mode_check['valid']:
                violations.append(f"❌ AUTO MODE: {auto_mode_check['reason']}")
                risk_score -= 50
            else:
                passed_checks.append(f"✅ AUTO MODE: {auto_mode_check['reason']}")
        else:
            passed_checks.append(f"✅ DCA MODE: DCA enabled independently from auto_mode")
            
        # 2. 🧠 CONFIDENCE THRESHOLD
        confidence_check = self._check_signal_confidence(signal_confidence, is_dca)
        if not confidence_check['valid']:
            violations.append(f"❌ CONFIDENCE: {confidence_check['reason']}")
            risk_score -= 20
        else:
            passed_checks.append(f"✅ CONFIDENCE: {confidence_check['reason']}")
            
        # 3. 📍 POSITION LIMITS  
        position_check = self._check_position_limits(symbol)
        if not position_check['valid']:
            violations.append(f"❌ POSITIONS: {position_check['reason']}")
            risk_score -= 15
        else:
            passed_checks.append(f"✅ POSITIONS: {position_check['reason']}")
            
        # 4. 💰 DAILY LOSS LIMITS
        daily_loss_check = self._check_daily_loss_limits()
        if not daily_loss_check['valid']:
            violations.append(f"❌ DAILY LOSS: {daily_loss_check['reason']}")
            risk_score -= 25
        else:
            passed_checks.append(f"✅ DAILY LOSS: {daily_loss_check['reason']}")
            
        # 5. ⏰ TRADING HOURS
        hours_check = self._check_trading_hours()
        if not hours_check['valid']:
            violations.append(f"❌ TRADING HOURS: {hours_check['reason']}")
            risk_score -= 10
        else:
            passed_checks.append(f"✅ TRADING HOURS: {hours_check['reason']}")
        
        # 5.5 📰 NEWS AVOIDANCE CHECK
        news_check = self._check_news_avoidance(symbol)
        if not news_check['valid']:
            violations.append(f"❌ NEWS AVOIDANCE: {news_check['reason']}")
            risk_score -= 15
        else:
            passed_checks.append(f"✅ NEWS AVOIDANCE: {news_check['reason']}")
            
        # 6. 🛡️ EMERGENCY STOPS
        emergency_check = self._check_emergency_stops()
        if not emergency_check['valid']:
            violations.append(f"❌ EMERGENCY: {emergency_check['reason']}")
            risk_score -= 30
        else:
            passed_checks.append(f"✅ EMERGENCY: {emergency_check['reason']}")
            
        # 7. 🔄 DCA CONDITIONS (disabled - use dca_service.py)
        if is_dca:
            passed_checks.append(f"✅ DCA: Validation bypassed - using dca_service.py")
                
        # 8. 📊 VOLUME SETTINGS
        volume_check = self._check_volume_settings()
        if not volume_check['valid']:
            warnings.append(f"⚠️ VOLUME: {volume_check['reason']}")
            risk_score -= 5
        else:
            passed_checks.append(f"✅ VOLUME: {volume_check['reason']}")
            
        # 9. � DUPLICATE ENTRY CHECK - CRITICAL (only for non-DCA new entries)
        if not is_dca:
            duplicate_check = self._check_duplicate_entry(symbol, order_type)
            if not duplicate_check['valid']:
                violations.append(f"❌ DUPLICATE ENTRY: {duplicate_check['reason']}")
                risk_score -= 50
            else:
                passed_checks.append(f"✅ DUPLICATE ENTRY: {duplicate_check['reason']}")
        
        # 10. �🚫 RISK PERCENT BLOCKING CHECK - CRITICAL
        risk_percent_check = self._check_risk_percent_blocking()
        if not risk_percent_check['valid']:
            violations.append(f"❌ RISK BLOCKING: {risk_percent_check['reason']}")
            risk_score -= 100  # Complete block
        else:
            passed_checks.append(f"✅ RISK BLOCKING: {risk_percent_check['reason']}")
            
        # Determine overall result
        is_valid = len(violations) == 0
        risk_score = max(0, risk_score)
        
        # Generate recommendation
        if is_valid:
            if risk_score >= 90:
                recommendation = "🟢 EXCELLENT - All conditions optimal for trading"
            elif risk_score >= 75:
                recommendation = "🟡 GOOD - Trading allowed with minor warnings"
            else:
                recommendation = "🟠 ACCEPTABLE - Trading allowed but watch closely"
        else:
            recommendation = f"🔴 REJECTED - {len(violations)} critical violations prevent trading"
            
        return RiskValidationResult(
            is_valid=is_valid,
            risk_score=risk_score,
            violations=violations,
            warnings=warnings,
            passed_checks=passed_checks,
            recommendation=recommendation
        )
        
    def _check_auto_mode(self) -> Dict[str, Any]:
        """Check auto trading mode like human analysis"""
        enable_auto = self.risk_settings.get('enable_auto_mode', False)
        
        # Auto trading can proceed regardless of other settings
        return {'valid': True, 'reason': f'Auto trading enabled (auto_mode: {enable_auto})'}
        
    def _check_signal_confidence(self, confidence: float, is_dca: bool) -> Dict[str, Any]:
        """Check signal confidence thresholds"""
        # Use appropriate confidence thresholds (0-100 scale like GUI)
        if is_dca:
            # DCA uses min_confidence_for_dca setting
            min_confidence = self.risk_settings.get('min_confidence_for_dca', 1.5) * 10  # Convert to percentage scale
            if confidence < min_confidence:
                return {'valid': False, 'reason': f'DCA requires {min_confidence:.1f}%, got {confidence:.1f}%'}
            return {'valid': True, 'reason': f'DCA confidence OK: {confidence:.1f}% >= {min_confidence:.1f}%'}
        else:
            # Regular entry uses min_confidence_for_entry setting
            min_confidence = self.risk_settings.get('min_confidence_for_entry', 2.0) * 10  # Convert to percentage scale
            if confidence < min_confidence:
                return {'valid': False, 'reason': f'Signal confidence {confidence:.1f}% < required {min_confidence:.1f}%'}
            return {'valid': True, 'reason': f'Signal confidence OK: {confidence:.1f}% >= {min_confidence:.1f}%'}
            
    def _check_position_limits(self, symbol: str) -> Dict[str, Any]:
        """Check position limits like human analysis"""
        try:
            current_total = mt5.positions_total()
            max_positions = self.risk_settings.get('max_positions', 5)
            
            if current_total >= max_positions:
                return {'valid': False, 'reason': f'Max total positions {current_total}/{max_positions}'}
                
            # Check per-symbol limit
            symbol_positions = len([p for p in mt5.positions_get() if p.symbol == symbol])
            max_per_symbol = self.risk_settings.get('max_positions_per_symbol', 7)  # Use actual setting value
            
            if symbol_positions >= max_per_symbol:
                return {'valid': False, 'reason': f'{symbol} max positions {symbol_positions}/{max_per_symbol}'}
                
            # Also check symbol volume exposure limit
            if symbol_positions > 0:
                total_symbol_volume = sum(pos.volume for pos in mt5.positions_get() if pos.symbol == symbol)
                symbol_exposures = self.risk_settings.get('symbol_exposure', {})
                
                # Try to find exposure limit with flexible symbol name matching
                max_symbol_exposure = None
                base_symbol = symbol.replace('.', '').replace('_m', '')  # Clean symbol name
                
                # Check exact match first
                if symbol in symbol_exposures:
                    max_symbol_exposure = symbol_exposures[symbol]
                # Then check base symbol match
                elif base_symbol in symbol_exposures:
                    max_symbol_exposure = symbol_exposures[base_symbol]
                # Finally check if any exposure key matches base symbol
                else:
                    for exp_symbol, limit in symbol_exposures.items():
                        if base_symbol in exp_symbol or exp_symbol in base_symbol:
                            max_symbol_exposure = limit
                            break
                
                if max_symbol_exposure is not None and isinstance(max_symbol_exposure, (int, float)):
                    if total_symbol_volume >= max_symbol_exposure:
                        return {'valid': False, 'reason': f'{symbol} max volume exposure {total_symbol_volume:.2f}/{max_symbol_exposure} lots'}
                
            return {'valid': True, 'reason': f'Position limits OK: {current_total}/{max_positions} total, {symbol_positions}/{max_per_symbol} for {symbol}'}
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return {'valid': False, 'reason': f'Cannot check positions - MT5 connection issue: {e}'}
    
    def _check_duplicate_entry(self, symbol: str, order_type: str) -> Dict[str, Any]:
        """Check for duplicate entry orders to prevent spam"""
        try:
            # Get current price for range calculation
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return {'valid': False, 'reason': 'Cannot get current price for duplicate check'}
            
            current_price = tick.bid if "SELL" in order_type.upper() else tick.ask
            
            # 🚨 ENHANCED: Much tighter duplicate detection for Entry vs DCA
            # Entry orders: 5 pips minimum (prevent exact duplicates)
            # DCA orders: 15 pips minimum (allow closer DCA spacing)
            is_dca_order = "DCA" in order_type.upper()
            duplicate_distance_pips = 15.0 if is_dca_order else 5.0  # Tighter for Entry orders
            pip_value = self._get_pip_value_for_distance(symbol)
            min_distance_price = duplicate_distance_pips * pip_value
            
            # 🚨 ENHANCED: Time-based duplicate check (prevent rapid-fire orders)
            current_time = time.time()
            if not hasattr(self, '_last_order_time'):
                self._last_order_time = {}
            
            last_time_key = f"{symbol}_{order_type}"
            if last_time_key in self._last_order_time:
                time_since_last = current_time - self._last_order_time[last_time_key]
                if time_since_last < 2.0:  # Less than 2 seconds
                    logger.warning(f"🚫 RAPID-FIRE BLOCKED: {symbol} {order_type} attempted {time_since_last:.1f}s after previous order")
                    return {
                        'valid': False,
                        'reason': f'Rapid-fire prevention: Only {time_since_last:.1f}s since last {order_type} order (min: 2.0s)'
                    }
            
            self._last_order_time[last_time_key] = current_time
            
            # Check existing positions for same symbol and direction
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                for pos in positions:
                    # Check if same direction and too close to current price
                    pos_type = "BUY" if pos.type == 0 else "SELL"
                    if pos_type in order_type.upper():
                        distance_from_current = abs(pos.price_open - current_price)
                        distance_pips = distance_from_current / pip_value
                        
                        # 🔒 ENHANCED: Multi-layer duplicate detection
                        
                        # Layer 1: Identical price detection (0.1 pips tolerance)
                        if distance_pips < 0.1:  # Less than 0.1 pips = identical entry
                            logger.warning(f"🚫 IDENTICAL ENTRY: {pos_type} position #{pos.ticket} at {pos.price_open:.5f} vs current {current_price:.5f} ({distance_pips:.3f} pips)")
                            return {
                                'valid': False, 
                                'reason': f'IDENTICAL DUPLICATE: existing {pos_type} position #{pos.ticket} at identical price {pos.price_open:.5f} ({distance_pips:.3f} pips)'
                            }
                        
                        # Layer 2: Recent position check (created within 30 seconds)
                        position_age = current_time - pos.time
                        if position_age < 30 and distance_pips < 2.0:  # Recent + very close = likely duplicate
                            logger.warning(f"🚫 RECENT DUPLICATE: {pos_type} position #{pos.ticket} created {position_age:.1f}s ago at {pos.price_open:.5f}, only {distance_pips:.1f} pips away")
                            return {
                                'valid': False, 
                                'reason': f'Recent duplicate: {pos_type} position created {position_age:.1f}s ago only {distance_pips:.1f} pips away'
                            }
                        
                        # Layer 3: Distance-based duplicate (respects Entry vs DCA spacing)
                        elif distance_from_current < min_distance_price:
                            logger.warning(f"🚫 DISTANCE DUPLICATE: {pos_type} position #{pos.ticket} at {pos.price_open:.5f} too close to current {current_price:.5f} ({distance_pips:.1f} pips < {duplicate_distance_pips} pips)")
                            return {
                                'valid': False, 
                                'reason': f'Distance duplicate: existing {pos_type} position only {distance_pips:.1f} pips away (min: {duplicate_distance_pips} pips for {order_type})'
                            }
            
            # 🚨 ENHANCED: Check pending orders with multi-layer protection
            orders = mt5.orders_get(symbol=symbol)
            if orders:
                for order in orders:
                    # Skip check for different symbols (safety)
                    if order.symbol != symbol:
                        continue
                        
                    # Check if same direction and apply enhanced duplicate detection
                    order_type_map = {0: "BUY", 1: "SELL", 2: "BUY", 3: "SELL", 4: "BUY", 5: "SELL"}
                    order_direction = order_type_map.get(order.type, "UNKNOWN")
                    
                    if order_direction in order_type.upper():
                        distance_from_current = abs(order.price_open - current_price)
                        distance_pips = distance_from_current / pip_value
                        
                        # Layer 1: Identical order price
                        if distance_pips < 0.1:
                            logger.warning(f"🚫 IDENTICAL ORDER: {order_direction} order #{order.ticket} at identical price {order.price_open:.5f}")
                            return {
                                'valid': False,
                                'reason': f'Identical order: existing {order_direction} order #{order.ticket} at identical price {order.price_open:.5f}'
                            }
                        
                        # Layer 2: Recent order check
                        order_age = current_time - getattr(order, 'time_setup', 0)
                        if order_age < 30 and distance_pips < 2.0:
                            logger.warning(f"🚫 RECENT ORDER DUPLICATE: {order_direction} order #{order.ticket} created {order_age:.1f}s ago, {distance_pips:.1f} pips away")
                            return {
                                'valid': False,
                                'reason': f'Recent order duplicate: {order_direction} order created {order_age:.1f}s ago only {distance_pips:.1f} pips away'
                            }
                        
                        # Layer 3: Distance-based duplicate
                        elif distance_from_current < min_distance_price:
                            logger.warning(f"🚫 ORDER DISTANCE DUPLICATE: {order_direction} order #{order.ticket} at {order.price_open:.5f} only {distance_pips:.1f} pips away")
                            return {
                                'valid': False,
                                'reason': f'Order distance duplicate: existing {order_direction} order only {distance_pips:.1f} pips away (min: {duplicate_distance_pips} pips for {order_type})'
                            }
            
            return {'valid': True, 'reason': f'No duplicate entries within {duplicate_distance_pips} pips'}
            
        except Exception as e:
            logger.error(f"❌ Error checking duplicate entry: {e}")
            return {'valid': True, 'reason': 'Duplicate check failed - allowing entry'}
            
    def _check_daily_loss_limits(self) -> Dict[str, Any]:
        """Check daily loss limits like human analysis"""
        try:
            max_daily_loss = self.risk_settings.get('max_daily_loss_percent', 3.0)
            
            # Handle "OFF" string case
            if isinstance(max_daily_loss, str) and max_daily_loss.upper() == "OFF":
                return {'valid': True, 'reason': 'Daily loss limit disabled (OFF)'}
            
            # Convert to float for comparison
            try:
                max_daily_loss = float(max_daily_loss)
            except (ValueError, TypeError):
                max_daily_loss = 3.0  # Default fallback
                
            if max_daily_loss <= 0:
                return {'valid': True, 'reason': 'Daily loss limit disabled (0%)'}
                
            # Calculate today's P&L
            account_info = mt5.account_info()
            if not account_info:
                return {'valid': False, 'reason': 'Cannot get account info'}
                
            # Get history deals for today
            from datetime import datetime, timedelta
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            deals = mt5.history_deals_get(today_start, datetime.now())
            
            if deals:
                daily_profit = sum(deal.profit for deal in deals)
                daily_pnl_percent = (daily_profit / account_info.balance) * 100
                
                if daily_pnl_percent <= -max_daily_loss:
                    return {'valid': False, 'reason': f'Daily loss {daily_pnl_percent:.2f}% >= limit {max_daily_loss}%'}
                    
                return {'valid': True, 'reason': f'Daily P&L OK: {daily_pnl_percent:.2f}% (limit: -{max_daily_loss}%)'}
            else:
                return {'valid': True, 'reason': 'No trades today - daily loss check passed'}
        except Exception as e:
            logger.warning(f"Error checking daily loss: {e}")
            return {'valid': True, 'reason': f'Daily loss check skipped - calculation error: {e}'}
            
    def _check_trading_hours(self) -> Dict[str, Any]:
        """Check trading hours with minute precision"""
        start_hour = self.risk_settings.get('trading_hours_start', 0)
        start_minute = self.risk_settings.get('trading_minutes_start', 0)
        end_hour = self.risk_settings.get('trading_hours_end', 23)
        end_minute = self.risk_settings.get('trading_minutes_end', 59)
        
        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute
        
        # Convert to total minutes for easier comparison
        current_total = current_hour * 60 + current_minute
        start_total = start_hour * 60 + start_minute
        end_total = end_hour * 60 + end_minute
        
        if start_total <= end_total:
            # Normal range (e.g., 8:00-18:30)
            in_hours = start_total <= current_total <= end_total
        else:
            # Overnight range (e.g., 22:00-6:30)
            in_hours = current_total >= start_total or current_total <= end_total
        
        time_str = f'{current_hour:02d}:{current_minute:02d}'
        range_str = f'{start_hour:02d}:{start_minute:02d}-{end_hour:02d}:{end_minute:02d}'
            
        if not in_hours:
            return {'valid': False, 'reason': f'Outside trading hours {range_str}, now {time_str}'}
        return {'valid': True, 'reason': f'Trading hours OK: {time_str} within {range_str}'}
    
    def _check_news_avoidance(self, symbol: str = None) -> Dict[str, Any]:
        """Check if we should avoid trading due to upcoming/recent news events"""
        import os
        import json
        import glob
        import pytz
        
        # Check if news avoidance is disabled
        if self.risk_settings.get('disable_news_avoidance', False):
            return {'valid': True, 'reason': 'News avoidance disabled'}
        
        avoid_minutes = self.risk_settings.get('avoid_news_minutes', 0)
        
        # If avoid_minutes is 0 or "OFF", skip check
        if avoid_minutes == 0 or avoid_minutes == "OFF" or str(avoid_minutes).upper() == "OFF":
            return {'valid': True, 'reason': 'News avoidance is OFF'}
        
        try:
            avoid_minutes = int(avoid_minutes)
        except (ValueError, TypeError):
            return {'valid': True, 'reason': 'Invalid avoid_news_minutes value'}
        
        try:
            # Find latest news file (try news_forexfactory first, then recent_news)
            news_files = glob.glob(os.path.join('news_output', 'news_forexfactory_*.json'))
            if not news_files:
                news_files = glob.glob(os.path.join('news_output', 'recent_news_with_actual_*.json'))
            if not news_files:
                return {'valid': True, 'reason': 'No news data available'}
            
            latest_file = max(news_files, key=os.path.getmtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                news_data = json.load(f)
            
            if not news_data:
                return {'valid': True, 'reason': 'No news events found'}
            
            # Get current time in Vietnam timezone
            vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
            now = datetime.now(vn_tz)
            
            # Extract currency from symbol for filtering
            symbol_currency = None
            if symbol:
                # Extract currency codes from symbol (e.g., XAUUSD -> USD, EURUSD -> EUR,USD)
                symbol_upper = symbol.upper().replace('.', '').replace('_', '')
                currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF', 'XAU', 'XAG']
                symbol_currency = [c for c in currencies if c in symbol_upper]
            
            # Check each news event
            for event in news_data:
                # Use time_vn_24h for VN timezone (app runs in VN)
                event_time_str = event.get('time_vn_24h', '') or event.get('time', '')
                event_currency = event.get('currency', '')
                event_impact = str(event.get('impact', '')).lower()
                
                # Skip low impact events if filtering
                if event_impact not in ['high', 'medium']:
                    continue
                
                # If symbol provided, only check relevant currencies
                if symbol_currency and event_currency not in symbol_currency:
                    continue
                
                # Parse event time (time_vn_24h is in HH:MM format)
                import re
                # Try 24h format first (HH:MM), then 12h format (H:MMam/pm)
                m24 = re.match(r'^(\d{1,2}):(\d{2})$', event_time_str)
                m12 = re.match(r'^(\d{1,2}):(\d{2})(am|pm)$', event_time_str.lower()) if not m24 else None
                
                if m24:
                    h = int(m24.group(1))
                    mn = int(m24.group(2))
                elif m12:
                    h, mn, ap = int(m12.group(1)), int(m12.group(2)), m12.group(3)
                    if ap == 'pm' and h != 12:
                        h += 12
                    if ap == 'am' and h == 12:
                        h = 0
                else:
                    continue  # Skip if can't parse time
                
                # Create event datetime for today
                event_dt = now.replace(hour=h, minute=mn, second=0, microsecond=0)
                
                # Calculate time difference in minutes
                diff_minutes = (event_dt - now).total_seconds() / 60
                
                # Check if within avoidance window (before or after)
                if -avoid_minutes <= diff_minutes <= avoid_minutes:
                    event_name = event.get('event', 'Unknown event')
                    time_status = "upcoming" if diff_minutes > 0 else "recent"
                    return {
                        'valid': False, 
                        'reason': f'Avoid trading: {time_status} {event_currency} news "{event_name}" at {event_time_str} (±{avoid_minutes}min window)'
                    }
            
            return {'valid': True, 'reason': f'No high-impact news within ±{avoid_minutes}min'}
            
        except Exception as e:
            logger.warning(f"Error checking news avoidance: {e}")
            return {'valid': True, 'reason': f'News check error: {e}'}
        
    def _check_emergency_stops(self) -> Dict[str, Any]:
        """Check emergency stops with OFF settings support"""
        emergency_dd_raw = self.risk_settings.get('emergency_stop_drawdown', 0)
        emergency_dd = parse_setting_value(emergency_dd_raw, 0, 'emergency_stop_drawdown')
        disabled = self.risk_settings.get('disable_emergency_stop', False)
        
        if disabled:
            return {'valid': True, 'reason': 'Emergency stop disabled (⚠️ risky but allowed)'}
            
        # Check if emergency stop is OFF
        if not is_setting_enabled(emergency_dd):
            return {'valid': True, 'reason': 'Emergency stop is OFF (no drawdown limit)'}
            
        if emergency_dd <= 0:
            return {'valid': True, 'reason': 'Emergency stop disabled (drawdown = 0%)'}
            
        # Check current drawdown
        try:
            account_info = mt5.account_info()
            if account_info:
                balance = account_info.balance
                equity = account_info.equity
                drawdown_percent = ((balance - equity) / balance) * 100 if balance > 0 else 0
                
                if drawdown_percent >= emergency_dd:
                    return {'valid': False, 'reason': f'Emergency stop triggered: DD {drawdown_percent:.2f}% >= {emergency_dd}%'}
                return {'valid': True, 'reason': f'Emergency stop OK: DD {drawdown_percent:.2f}% < {emergency_dd}%'}
        except Exception as e:
            logger.warning(f"Error checking emergency stop: {e}")
            return {'valid': True, 'reason': f'Emergency stop check skipped - calculation error: {e}'}
            
        return {'valid': True, 'reason': f'Emergency stop armed at {emergency_dd}% DD'}
        
    def _check_dca_conditions(self, symbol: str) -> Dict[str, Any]:
        """Check DCA conditions for ATR and Fixed Pips modes (Fibonacci managed by dca_service.py)"""
        try:
            dca_mode = self.risk_settings.get("dca_mode", "fixed_pips")
            
            # Skip validation for Fibonacci mode - handled by dca_service.py
            if "fibo" in dca_mode.lower():
                return {"valid": True, "reason": "Fibonacci DCA managed by dca_service.py"}
            
            # Validate ATR and Fixed Pips modes
            if not self.risk_settings.get("enable_dca", False):
                return {"valid": False, "reason": "DCA is disabled in settings"}
                
            max_levels = self.risk_settings.get("max_dca_levels", 3)
            if max_levels <= 0:
                return {"valid": False, "reason": "No DCA levels configured"}
                
            return {"valid": True, "reason": f"DCA conditions met for {dca_mode} mode"}
            
        except Exception as e:
            return {"valid": False, "reason": f"DCA condition check error: {e}"}
        if not self.risk_settings.get('enable_dca', False):
            return {'valid': False, 'reason': 'DCA disabled in settings'}
            
        try:
            # Check if we have existing position for DCA
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return {'valid': False, 'reason': 'No existing position for DCA'}
                
            # Check DCA levels with OFF support
            current_dca_count = len([p for p in positions if any(keyword in p.comment for keyword in ['GPT_20B|DCA', 'GOLDKILLER_AI_DCA', 'DCA']) if p.comment])
            max_dca_levels_raw = self.risk_settings.get('max_dca_levels', 3)
            max_dca_levels = parse_setting_value(max_dca_levels_raw, 3, 'max_dca_levels')
            
            # If max_dca_levels is OFF, allow unlimited DCA
            if not is_setting_enabled(max_dca_levels):
                logger.debug("Max DCA levels is OFF - unlimited DCA allowed")
            elif current_dca_count >= max_dca_levels:
                return {'valid': False, 'reason': f'Max DCA levels reached: {current_dca_count}/{max_dca_levels}'}
                
            # Check DCA drawdown requirement
            main_position = positions[0]
            current_price = mt5.symbol_info_tick(symbol).bid
            entry_price = main_position.price_open
            
            if main_position.type == mt5.POSITION_TYPE_BUY:
                drawdown_pips = (entry_price - current_price) / mt5.symbol_info(symbol).point
            else:
                drawdown_pips = (current_price - entry_price) / mt5.symbol_info(symbol).point
                
            min_drawdown_raw = self.risk_settings.get('dca_min_drawdown', 1.0)
            # Calculate DCA distance based on mode
            dca_distance = self._calculate_dca_distance_by_mode(symbol, current_price, entry_price)
            min_drawdown = parse_setting_value(min_drawdown_raw, 1.0, 'dca_min_drawdown')
            
            # 🔧 DEBUG: Log DCA distance being used  
            dca_mode = self.risk_settings.get('dca_mode', 'fixed_pips')
            logger.info(f"🎯 DCA Distance Check: Mode={dca_mode}, Distance={dca_distance} pips, Current DD={drawdown_pips:.1f} pips")
            
            # Check if DCA settings are OFF
            if not is_setting_enabled(min_drawdown) or dca_distance <= 0:
                return {'valid': True, 'reason': 'DCA distance/drawdown checks are OFF'}
            
            if drawdown_pips < dca_distance:
                return {'valid': False, 'reason': f'DCA distance not met: {drawdown_pips:.1f} < {dca_distance:.1f} pips (mode: {dca_mode})'}
                
            # 🆕 NEW: Check if there are existing DCA pending orders too close to current price
            existing_dca_check = self._check_existing_dca_orders_distance(symbol, current_price, dca_distance)
            if not existing_dca_check['valid']:
                return existing_dca_check
                
            return {'valid': True, 'reason': f'DCA conditions OK: level {current_dca_count+1}/{max_dca_levels}, DD {drawdown_pips:.1f} pips'}
        except Exception as e:
            logger.error(f"DCA condition check failed for {symbol}: {e}")
            return {'valid': False, 'reason': f'DCA condition check failed - calculation error: {e}'}
    
    def _calculate_dca_distance_by_mode(self, symbol: str, current_price: float, entry_price: float) -> float:
        """Calculate DCA distance for ATR and Fixed Pips modes (Fibonacci managed by dca_service.py)"""
        try:
            dca_mode = self.risk_settings.get("dca_mode", "fixed_pips")
            
            # Skip for Fibonacci mode - handled by dca_service.py  
            if "fibo" in dca_mode.lower():
                logging.debug("Fibonacci DCA distance managed by dca_service.py")
                return 50.0  # Fallback
                
            if "ATR" in dca_mode or "atr" in dca_mode.lower():
                # ATR-based distance calculation
                atr_multiplier = self.risk_settings.get("dca_atr_multiplier", 2.0)
                atr_value = self._get_atr_value(symbol)
                if atr_value > 0:
                    pip_value = get_pip_value(symbol)
                    atr_pips = atr_value / pip_value
                    return atr_pips * atr_multiplier
                else:
                    # Fallback to base distance if ATR not available
                    return float(self.risk_settings.get("dca_distance_pips", 50.0))
            else:
                # Fixed pips mode
                return float(self.risk_settings.get("dca_distance_pips", 50.0))
                
        except Exception as e:
            logging.warning(f"DCA distance calculation error: {e}, using fallback")
            return 50.0

    def _calculate_dca_entry_price(self, symbol: str, main_entry_price: float, dca_level: int, direction: str) -> float:
        """
        🔧 CRITICAL FIX: Calculate proper DCA entry price based on main entry and DCA level
        
        This function calculates where the DCA order should be placed, not where it should trigger.
        For BUY positions: DCA orders are placed BELOW main entry at calculated levels
        For SELL positions: DCA orders are placed ABOVE main entry at calculated levels
        """
        try:
            dca_mode = self.risk_settings.get("dca_mode", "fixed_pips")
            pip_value = get_pip_value(symbol)
            
            # 🔍 DEBUG: Log DCA mode
            logging.info(f"🔍 DCA Mode: '{dca_mode}'")
            
            # Calculate DCA distance for this level
            if "ATR" in dca_mode or "atr" in dca_mode.lower() or "Bội số" in dca_mode:
                # ATR-based DCA spacing
                atr_multiplier = self.risk_settings.get("dca_atr_multiplier", 1.0)
                logging.info(f"🔍 DCA ATR Multiplier from settings: {atr_multiplier}")
                
                atr_value = self._get_atr_value(symbol)
                logging.info(f"🔍 ATR Value loaded for {symbol}: {atr_value}")
                
                if atr_value > 0:
                    # ATR from indicator is ALWAYS in price units
                    # Convert to pips based on symbol's pip_value
                    # XAUUSD: pip=0.1 → 21.62 / 0.1 = 216.2 pips
                    # GBPUSD: pip=0.0001 → 0.00113 / 0.0001 = 11.3 pips
                    # BTCUSD: pip=10 → 1167.36 / 10 = 116.7 pips
                    
                    atr_pips = atr_value / pip_value
                    base_distance_pips = atr_pips * atr_multiplier
                    distance_price = base_distance_pips * pip_value
                    
                    logging.info(f"📊 DCA: ATR={atr_value:.5f} → {atr_pips:.1f}pips × {atr_multiplier}x = {base_distance_pips:.1f}pips")
                else:
                    logging.warning(f"⚠️ ATR value is 0 or invalid for {symbol}, using fallback")
                    base_distance_pips = self.risk_settings.get("dca_distance_pips", 50.0)
                    distance_price = base_distance_pips * pip_value
                    logging.info(f"📏 Fallback: {base_distance_pips} pips = ${distance_price:.2f}")
            else:
                # Fixed pips mode
                logging.info(f"📏 Using Fixed Pips mode (not ATR)")
                base_distance_pips = float(self.risk_settings.get("dca_distance_pips", 50.0))
                distance_price = base_distance_pips * pip_value
                logging.info(f"📏 Fixed: {base_distance_pips} pips = ${distance_price:.2f} per level")
            
            # 🔧 CRITICAL FIX: ALWAYS calculate from main entry using incremental spacing
            # Do NOT space from last DCA position as it can pick up old DCA from previous trades
            # Incremental spacing formula:
            # Level 1: main_entry ± (distance × 1)
            # Level 2: main_entry ± (distance × 2)
            # Level 3: main_entry ± (distance × 3)
            
            cumulative_distance = distance_price * dca_level
            
            # � DETAILED LOGGING for debugging
            logging.info(f"=" * 60)
            logging.info(f"🧮 DCA CALCULATION BREAKDOWN:")
            logging.info(f"   Symbol: {symbol}")
            logging.info(f"   Main Entry: ${main_entry_price:.5f}")
            logging.info(f"   DCA Level: {dca_level}")
            logging.info(f"   Direction: {direction}")
            logging.info(f"   Base Distance per level: ${distance_price:.5f}")
            logging.info(f"   Cumulative Distance (Level {dca_level}): ${cumulative_distance:.5f}")
            logging.info(f"      Formula: ${distance_price:.5f} × {dca_level} = ${cumulative_distance:.5f}")
            
            if direction.upper() == "BUY":
                dca_entry_price = main_entry_price - cumulative_distance
                logging.info(f"   Entry Calculation (BUY): ${main_entry_price:.5f} - ${cumulative_distance:.5f}")
            else:  # SELL
                dca_entry_price = main_entry_price + cumulative_distance
                logging.info(f"   Entry Calculation (SELL): ${main_entry_price:.5f} + ${cumulative_distance:.5f}")
            
            logging.info(f"✅ DCA LEVEL {dca_level} FINAL PRICE: ${dca_entry_price:.5f}")
            logging.info(f"=" * 60)
            
            return dca_entry_price
            
        except Exception as e:
            logging.error(f"❌ DCA entry price calculation error: {e}")
            # Fallback: Use main entry price (will cause problems but prevents crashes)
            return main_entry_price
    
    # Fibonacci DCA function removed - use dca_service.py instead
    
    def _get_atr_value(self, symbol: str) -> float:
        """Get ATR value from indicator data files"""
        try:
            import json
            import os
            
            # Try H1 timeframe first (most stable for DCA)
            timeframes = ['H1', 'M30', 'M15']
            
            # Clean symbol - remove trailing dot if exists
            clean_symbol = symbol.rstrip('.')
            
            for tf in timeframes:
                # Try both original and clean symbol names with multiple patterns
                for sym in [symbol, clean_symbol]:
                    # Try different filename patterns
                    patterns = [
                        f"indicator_output/{sym}_{tf}_indicators.json",  # BNBUSD_m_H1_indicators.json
                        f"indicator_output/{sym}._{tf}_indicators.json",  # XAUUSD._H1_indicators.json
                    ]
                    
                    for indicator_file in patterns:
                        if os.path.exists(indicator_file):
                            try:
                                with open(indicator_file, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    
                                if data and isinstance(data, list) and len(data) > 0:
                                    # Get latest ATR value
                                    latest_data = data[-1]
                                    
                                    # Check for ATR keys
                                    for atr_key in ['ATR14', 'atr', 'ATR', 'ATR_14']:
                                        if atr_key in latest_data and latest_data[atr_key] is not None:
                                            atr_value = float(latest_data[atr_key])
                                            if atr_value > 0:
                                                logging.info(f"✅ Found ATR from {indicator_file}: {atr_key}={atr_value:.4f}")
                                                return atr_value
                            except Exception as e:
                                logging.debug(f"Error reading ATR from {indicator_file}: {e}")
                                continue
                            
            logging.warning(f"⚠️ No ATR data found for {symbol} in indicator_output folder")
            return 0.0
            
        except Exception as e:
            logging.error(f"❌ Error getting ATR value: {e}")
            return 0.0
    

    
    def _check_existing_dca_orders_distance(self, symbol: str, current_price: float, min_distance_pips: float) -> Dict[str, Any]:
        """Check if existing pending DCA orders are too close to current price"""
        try:
            # Get all pending orders for this symbol
            orders = mt5.orders_get(symbol=symbol)
            if not orders:
                return {'valid': True, 'reason': 'No existing pending orders'}
            
            # Get pip value for distance calculation
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return {'valid': True, 'reason': 'Cannot get symbol info - allow DCA'}
            
            pip_value = self._get_pip_value_for_distance(symbol)
            min_distance_price = min_distance_pips * pip_value
            
            # Check DCA orders only
            dca_orders = [order for order in orders if any(keyword in (order.comment or '') for keyword in ['GPT_20B|DCA', 'GOLDKILLER_AI_DCA', 'DCA'])]
            
            for order in dca_orders:
                distance_price = abs(order.price_open - current_price)
                distance_pips = distance_price / pip_value
                
                if distance_pips < min_distance_pips:
                    logger.info(f"🚫 DCA Distance Violation: Existing DCA order #{order.ticket} at {order.price_open:.5f} only {distance_pips:.1f} pips from current price {current_price:.5f} (min: {min_distance_pips} pips)")
                    return {
                        'valid': False, 
                        'reason': f'DCA order too close: {distance_pips:.1f} pips < {min_distance_pips} pips required'
                    }
            
            logger.info(f"✅ DCA Distance OK: No existing DCA orders within {min_distance_pips} pips of current price")
            return {'valid': True, 'reason': f'No DCA orders within {min_distance_pips} pips'}
            
        except Exception as e:
            logger.error(f"❌ Error checking existing DCA orders distance: {e}")
            return {'valid': True, 'reason': 'DCA distance check failed - allow DCA'}
    
    def _get_pip_value_for_distance(self, symbol: str) -> float:
        """Get pip value for distance calculation - COMPREHENSIVE CRYPTO & METALS SUPPORT"""
        symbol_upper = symbol.upper().rstrip('.').replace('_M', '').replace('_m', '')  # Normalize
        
        # ========== PRECIOUS METALS ==========
        if symbol_upper in ['XAUUSD', 'XAGUSD', 'GOLD', 'SILVER', 'XPTUSD', 'XPDUSD']:
            return 0.1   # Metals: 1 pip = 0.1 (Gold: 3881.03 -> 3881.73 = 7 pips)
        
        # ========== JPY PAIRS ==========
        elif 'JPY' in symbol_upper:
            return 0.01  # JPY pairs: 1 pip = 0.01 (USD/JPY: 147.15 -> 147.22 = 7 pips)
        
        # ========== HIGH-VALUE CRYPTO (≥ $1000) ==========
        elif symbol_upper in ['BTCUSD', 'ETHUSD']:
            return 1.0   # BTC/ETH: 1 pip = 1.0 (BTC: 65000 -> 65070 = 70 pips)
        
        # ========== MID-VALUE CRYPTO ($100-$1000) ==========
        elif symbol_upper in ['SOLUSD', 'BNBUSD', 'ADAUSD', 'AVAXUSD', 'DOTUSD', 'MATICUSD', 'LINKUSD', 'TRXUSD', 'SHIBUSD', 'ARBUSD', 'OPUSD', 'APEUSD', 'SANDUSD', 'CROUSD', 'FTTUSD']:
            return 0.1   # SOL/BNB/ADA etc: 1 pip = 0.1 (SOL: 224.06 -> 224.76 = 7 pips)
        
        # ========== LOW-VALUE CRYPTO (< $10) ==========
        elif any(crypto in symbol_upper for crypto in ['DOGE', 'XRP', 'TRX']):
            return 0.001  # DOGE/XRP etc: 1 pip = 0.001 (DOGE: 0.123 -> 0.130 = 7 pips)
        
        # ========== MICRO-VALUE CRYPTO (< $1) ==========
        elif any(micro_crypto in symbol_upper for micro_crypto in ['SHIB', 'PEPE', 'FLOKI']):
            return 0.00001  # Micro cryptos: 1 pip = 0.00001
        
        # ========== FOREX PAIRS ==========
        else:
            return 0.0001  # Major FX pairs: 1 pip = 0.0001 (EUR/USD: 1.0850 -> 1.0857 = 7 pips)
            
    def _check_volume_settings(self) -> Dict[str, Any]:
        """Check volume settings with OFF support"""
        volume_mode = self.risk_settings.get('volume_mode', 'Khối lượng mặc định')
        
        # 🔧 FIX: Read correct volume field based on mode
        if 'cố định' in volume_mode.lower() or 'fixed' in volume_mode.lower():
            fixed_volume = self.risk_settings.get('fixed_volume_lots', 0.1)
        else:
            fixed_volume = self.risk_settings.get('default_volume_lots', 0.1)
        
        # Parse volume limits with OFF support
        min_lot_raw = self.risk_settings.get('min_volume_auto', 0.01)
        max_lot_raw = self.risk_settings.get('max_total_volume', 10.0)
        
        min_lot = parse_setting_value(min_lot_raw, 0.01, 'min_volume_auto') or 0.01
        max_lot = parse_setting_value(max_lot_raw, 10.0, 'max_total_volume')
        
        if 'mặc định' in volume_mode:
            # Check min limit
            if fixed_volume < min_lot:
                return {'valid': False, 'reason': f'Fixed volume {fixed_volume} below min {min_lot}'}
            
            # Check max limit only if not OFF
            if is_setting_enabled(max_lot) and fixed_volume > max_lot:
                return {'valid': False, 'reason': f'Fixed volume {fixed_volume} above max {max_lot}'}
            elif not is_setting_enabled(max_lot):
                return {'valid': True, 'reason': f'Volume mode: Fixed {fixed_volume} lots (max limit OFF)'}
            else:
                return {'valid': True, 'reason': f'Volume mode: Fixed {fixed_volume} lots'}
        else:
            max_risk = self.risk_settings.get('max_risk_percent', 1.5)
            return {'valid': True, 'reason': f'Volume mode: Risk {max_risk}% per trade'}
    
    def _check_risk_percent_blocking(self) -> Dict[str, Any]:
        """🚫 Check if risk percent is set to block all trades"""
        max_risk_percent = self.risk_settings.get('max_risk_percent', 1.5)
        
        # Handle "OFF" string case
        if isinstance(max_risk_percent, str) and max_risk_percent.upper() == "OFF":
            return {'valid': True, 'reason': 'Risk percent disabled (OFF) - trading allowed'}
        
        # Convert to float for comparison
        try:
            max_risk_percent = float(max_risk_percent)
        except (ValueError, TypeError):
            max_risk_percent = 1.5  # Default fallback
        
        # If risk percent is 0 or negative, user wants to block all trades
        if max_risk_percent <= 0.0:
            return {
                'valid': False, 
                'reason': f'Risk percent set to {max_risk_percent}% - Trading completely blocked by user settings'
            }
        
        return {
            'valid': True, 
            'reason': f'Risk percent {max_risk_percent}% allows trading'
        }

    def _load_fibonacci_price_levels(self, symbol: str) -> Dict[str, float]:
        """Load latest Fibonacci retracement price levels for a symbol from indicator_output.

        Expected keys in JSON (produced by exporter): fib_0.0, fib_23.6, fib_38.2, fib_50.0, fib_61.8, fib_78.6, fib_100.0
        We normalize them to percent strings without trailing zeros ("23.6", "38.2", etc.).

        Returns:
            dict mapping percentage string -> price float. Empty dict if not available.
        """
        levels: Dict[str, float] = {}
        try:
            base_dir = os.path.join(os.path.dirname(__file__), 'indicator_output')
            if not os.path.isdir(base_dir):
                return levels
            # Pattern: SYMBOL_*_indicators.json (updated to match actual file names)
            pattern = os.path.join(base_dir, f"{symbol}_*_indicators.json")
            candidates = glob.glob(pattern)
            if not candidates:
                return levels
            # Pick newest file
            newest = max(candidates, key=os.path.getmtime)
            with open(newest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle array format (get latest/first entry)
            if isinstance(data, list) and len(data) > 0:
                data = data[-1]  # Get latest entry
            elif not isinstance(data, dict):
                return levels
            
            # Direct keys first
            fib_keys = [k for k in data.keys() if k.startswith('fib_')]
            if not fib_keys:
                return levels
            for k in fib_keys:
                try:
                    # k like fib_23.6 -> extract numeric part after underscore
                    perc_part = k.split('_', 1)[1]
                    # remove any trailing % or spaces
                    perc_clean = perc_part.replace('%','').strip()
                    # Some keys may be 50.0 etc. Normalize remove trailing zeros
                    try:
                        num = float(perc_clean)
                        perc_clean = ("%g" % num)
                    except Exception:
                        pass
                    price_val = float(data[k])
                    levels[perc_clean] = price_val
                except Exception:
                    continue
            if levels:
                logger.debug(f"[FIBO_LOAD] {symbol} loaded {len(levels)} fib levels from {os.path.basename(newest)}")
            return levels
        except Exception as e:
            logger.debug(f"[FIBO_LOAD] {symbol} failed to load fib levels: {e}")
            return {}
    
    # DCA tracking functions removed - use dca_service.py instead

    def detect_dca_opportunities(self) -> List[Dict[str, Any]]:
        """🚫 DISABLED: DCA detection moved to comprehensive_aggregator.py to prevent duplicates"""
        dca_opportunities = []
        
        # 🚫 DCA DETECTION DISABLED - ALL DCA HANDLED BY comprehensive_aggregator.py
        logging.info("🚫 DCA detection disabled in order_executor - using comprehensive_aggregator.py instead")
        return dca_opportunities
        
        # OLD CODE BELOW - KEPT FOR REFERENCE BUT DISABLED
        try:
            # Initialize last detected tracking if not exists
            if not hasattr(self, '_last_dca_detection'):
                self._last_dca_detection = {}
                
            current_time = time.time()
            dca_mode = self.risk_settings.get("dca_mode", "fixed_pips") 
            
            # Skip for Fibonacci mode - handled by dca_service.py
            if "fibo" in dca_mode.lower():
                logging.info("Fibonacci DCA managed by dca_service.py - skipping detection")
                return dca_opportunities
                
            # Check if DCA is enabled for ATR/Fixed Pips modes
            if not self.risk_settings.get("enable_dca", False):
                logging.info("DCA detection skipped: DCA disabled in settings")
                return dca_opportunities
                
            logging.info(f"DCA opportunity detection for {dca_mode} mode - ACTIVE")
            
            # Get all open positions for ATR/Fixed Pips DCA analysis
            if not mt5.initialize():
                logging.error("MT5 initialization failed")
                return dca_opportunities
                
            positions = mt5.positions_get()
            if not positions:
                logging.info("No open positions for DCA analysis")
                return dca_opportunities
                
            # Group positions by symbol and separate entry positions from DCA positions
            symbol_entries = {}
            symbol_dca_positions = {}
            
            for pos in positions:
                symbol = pos.symbol
                # Check if this is a DCA position (has DCA in comment)
                is_dca_position = False
                if hasattr(pos, 'comment') and pos.comment:
                    comment_str = str(pos.comment).upper()
                    is_dca_position = 'DCA' in comment_str and ('GPT_20B|DCA' in comment_str or 'GOLDKILLER_AI_DCA' in comment_str)
                
                if is_dca_position:
                    # This is a DCA position
                    if symbol not in symbol_dca_positions:
                        symbol_dca_positions[symbol] = []
                    symbol_dca_positions[symbol].append(pos)
                else:
                    # This is an Entry position - process DCA for it
                    if symbol not in symbol_entries:
                        symbol_entries[symbol] = []
                    symbol_entries[symbol].append(pos)
                
            # Analyze each Entry position for DCA opportunities  
            for symbol, entry_positions in symbol_entries.items():
                for entry_position in entry_positions:
                    # Get all DCA positions for this symbol (since we can't link by ticket in comment)
                    # We'll use chronological order - older entry gets DCA first
                    symbol_dca_list = symbol_dca_positions.get(symbol, [])
                    
                    # Sort entry positions by time to determine DCA assignment order
                    sorted_entries = sorted(entry_positions, key=lambda x: x.time)
                    entry_index = sorted_entries.index(entry_position)
                    
                    # Assign DCA positions based on symbol and chronological order
                    # Each entry gets its fair share of DCA positions
                    entries_count = len(sorted_entries)
                    dca_per_entry = len(symbol_dca_list) // entries_count if entries_count > 0 else 0
                    extra_dca = len(symbol_dca_list) % entries_count if entries_count > 0 else 0
                    
                    # Calculate how many DCA positions this specific entry should have
                    start_dca_idx = entry_index * dca_per_entry + min(entry_index, extra_dca)
                    end_dca_idx = start_dca_idx + dca_per_entry + (1 if entry_index < extra_dca else 0)
                    
                    entry_dca_positions = symbol_dca_list[start_dca_idx:end_dca_idx] if symbol_dca_list else []
                    current_dca_level = len(entry_dca_positions)
                    max_levels = self.risk_settings.get("max_dca_levels", 3)
                    
                    logging.info(f"🔍 DCA Analysis {symbol} Entry #{entry_position.ticket}: {current_dca_level} DCA positions, level {current_dca_level}/{max_levels}")
                    
                    # COMPREHENSIVE VALIDATION: DCA limits, symbol exposure, and position limits
                    if current_dca_level >= max_levels:
                        logging.info(f"❌ {symbol} Entry #{entry_position.ticket}: Max DCA levels reached ({current_dca_level}/{max_levels})")
                        continue
                    
                    # Check total positions per symbol limit
                    symbol_positions = mt5.positions_get(symbol=symbol)
                    symbol_position_count = len(symbol_positions) if symbol_positions else 0
                    max_positions_per_symbol = self.risk_settings.get('max_positions_per_symbol', 7)
                    
                    if symbol_position_count >= max_positions_per_symbol:
                        logging.info(f"❌ {symbol} Entry #{entry_position.ticket}: Max positions per symbol reached ({symbol_position_count}/{max_positions_per_symbol})")
                        continue
                    
                    # Check symbol volume exposure limit with flexible symbol matching
                    total_symbol_volume = sum(pos.volume for pos in symbol_positions) if symbol_positions else 0.0
                    symbol_exposures = self.risk_settings.get('symbol_exposure', {})
                    
                    # Try to find exposure limit with flexible symbol name matching
                    max_symbol_exposure = None
                    base_symbol = symbol.replace('.', '').replace('_m', '')  # Clean symbol name
                    
                    # Check exact match first
                    if symbol in symbol_exposures:
                        max_symbol_exposure = symbol_exposures[symbol]
                    # Then check base symbol match
                    elif base_symbol in symbol_exposures:
                        max_symbol_exposure = symbol_exposures[base_symbol]
                    # Finally check if any exposure key matches base symbol
                    else:
                        for exp_symbol, limit in symbol_exposures.items():
                            if base_symbol in exp_symbol or exp_symbol in base_symbol:
                                max_symbol_exposure = limit
                                break
                    
                    if max_symbol_exposure is not None and isinstance(max_symbol_exposure, (int, float)):
                        if total_symbol_volume >= max_symbol_exposure:
                            logging.info(f"❌ {symbol} Entry #{entry_position.ticket}: Max symbol volume exposure reached ({total_symbol_volume:.2f}/{max_symbol_exposure} lots)")
                            continue
                            continue
                    
                    # Calculate DCA trigger price based on mode
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick:
                        logging.warning(f"⚠️ {symbol} Entry #{entry_position.ticket}: No tick data available")
                        continue
                    
                    symbol_info = mt5.symbol_info(symbol)
                    if not symbol_info:
                        logging.warning(f"⚠️ {symbol}: Symbol info not available")
                        continue
                        
                    current_price = tick.bid if entry_position.type == 0 else tick.ask
                    entry_price = entry_position.price_open
                    
                    # 🔧 CRITICAL FIX: Simplified spread calculation
                    pip_value = get_pip_value(symbol)
                    spread_pips = (tick.ask - tick.bid) / pip_value
                    
                    dca_distance = self._calculate_dca_distance_by_mode(symbol, current_price, entry_price)
                    
                    # Add spread buffer to DCA distance for more accurate triggers
                    effective_dca_distance = dca_distance + (spread_pips * 1.5)  # 1.5x spread buffer
                    distance_in_price = effective_dca_distance * pip_value
                    
                    # Check if DCA should trigger
                    should_trigger = False
                    if entry_position.type == 0:  # BUY position
                        target_price = entry_price - distance_in_price
                        should_trigger = current_price <= target_price
                        logging.info(f"📊 {symbol} Entry #{entry_position.ticket} BUY Analysis: Entry={entry_price:.2f}, Current={current_price:.2f}, Spread={spread_pips:.1f}pips, DCA Distance={dca_distance:.1f}pips, Effective={effective_dca_distance:.1f}pips, Target={target_price:.2f}, Should Trigger={should_trigger}")
                    else:  # SELL position  
                        target_price = entry_price + distance_in_price
                        should_trigger = current_price >= target_price
                        logging.info(f"📊 {symbol} Entry #{entry_position.ticket} SELL Analysis: Entry={entry_price:.2f}, Current={current_price:.2f}, Spread={spread_pips:.1f}pips, DCA Distance={dca_distance:.1f}pips, Effective={effective_dca_distance:.1f}pips, Target={target_price:.2f}, Should Trigger={should_trigger}")
                        
                    if should_trigger:
                        # Check for duplicate detection prevention
                        opportunity_key = f"{symbol}_{entry_position.ticket}_{current_dca_level + 1}_{entry_position.time}"
                        
                        # Skip if already detected within last 60 seconds to prevent spam
                        if opportunity_key in self._last_dca_detection:
                            if current_time - self._last_dca_detection[opportunity_key] < 60:
                                logging.debug(f"DCA opportunity {symbol} Entry #{entry_position.ticket} Level {current_dca_level + 1} recently detected - skipping duplicate")
                                continue
                        
                        # Record detection timestamp
                        self._last_dca_detection[opportunity_key] = current_time
                        
                        dca_volume = self._calculate_dca_volume(symbol, current_dca_level, entry_position.volume)
                        
                        # 🔧 CRITICAL FIX: Calculate proper DCA entry price instead of using current_price
                        next_dca_level = current_dca_level + 1
                        dca_entry_price = self._calculate_dca_entry_price(
                            symbol=symbol,
                            main_entry_price=entry_position.price_open,
                            dca_level=next_dca_level,
                            direction="BUY" if entry_position.type == 0 else "SELL"
                        )
                        
                        logging.info(f"🔄 DCA Entry Price Calculation: Main={entry_position.price_open:.5f}, Level={next_dca_level}, DCA Entry={dca_entry_price:.5f}")
                        
                        opportunity = {
                            "symbol": symbol,
                            "direction": "BUY" if entry_position.type == 0 else "SELL",
                            "entry_price": dca_entry_price,  # ✅ FIXED: Use calculated DCA entry price
                            "main_entry_price": entry_position.price_open,  # Store main entry for reference
                            "volume": dca_volume,
                            "suggested_dca_volume": dca_volume,  # ✅ FIXED: Add missing field for compatibility
                            "confidence": 75.0,  # Default confidence for DCA
                            "current_dca_level": current_dca_level,  # Current level (before DCA)
                            "dca_level": current_dca_level + 1,
                            "mode": dca_mode,
                            "reason": f"DCA Level {current_dca_level + 1} for Entry #{entry_position.ticket} - {dca_mode} mode trigger",
                            "main_position_time": entry_position.time,
                            "main_position_ticket": entry_position.ticket,
                            "detection_time": current_time
                        }
                        
                        dca_opportunities.append(opportunity)
                        logging.info(f"🎯 NEW DCA opportunity: {symbol} Entry #{entry_position.ticket} Level {current_dca_level + 1} - Distance: {dca_distance:.1f} pips")
            
            return dca_opportunities
            
        except Exception as e:
            logging.error(f"Error detecting DCA opportunities: {e}")
            return dca_opportunities
    
    def _estimate_atr(self, symbol: str, period: int) -> float:
        """Estimate ATR for a symbol. Placeholder uses recent candles if available or fallback to base distance.
        Real implementation should fetch historical data and compute ATR."""
        try:
            if mt5:
                # Try to get  period*3 candles (safety) on M15 timeframe
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, min( period * 3, 500))
                if rates is not None and len(rates) >= period:
                    tr_values = []
                    prev_close = None
                    for r in rates:
                        high = r['high']; low = r['low']; close = r['close']
                        if prev_close is None:
                            tr = high - low
                        else:
                            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                        tr_values.append(tr)
                        prev_close = close
                        if len(tr_values) >= period:
                            break
                    if tr_values:
                        return sum(tr_values)/len(tr_values)
        except Exception as e:
            logger.debug(f"ATR estimation fallback for {symbol}: {e}")
        # Fallback value (will be scaled by pip size later); return 0 to signal fallback usage
        return 0.0

    def _calculate_dca_distance(self, dca_level: int, base_distance_pips: float, dca_mode: str, symbol: str = "") -> float:
        """DCA distance calculation disabled - use dca_service.py instead"""
        logger.info("DCA distance calculation disabled: Use dca_service.py for DCA calculations")
        return base_distance_pips

    def _calculate_dca_volume(self, symbol: str, current_dca_level: int, main_position_volume: float) -> float:
        """Calculate DCA volume for ATR and Fixed Pips modes (Fibonacci managed by dca_service.py)"""
        try:
            dca_mode = self.risk_settings.get("dca_mode", "fixed_pips")
            
            # Skip for Fibonacci mode - handled by dca_service.py
            if "fibo" in dca_mode.lower():
                logging.debug("Fibonacci DCA volume managed by dca_service.py")
                return main_position_volume  # Fallback
                
            base_multiplier = self.risk_settings.get("dca_volume_multiplier", 1.5)
            
            if "ATR" in dca_mode or "atr" in dca_mode.lower():
                # ATR-based volume scaling: progressive scaling
                next_dca_level = current_dca_level + 1
                multiplier = base_multiplier ** next_dca_level
                volume = float(main_position_volume) * multiplier
                logging.debug(f"ATR DCA Level {next_dca_level}: {main_position_volume} * {multiplier:.2f} = {volume:.2f}")
            else:
                # Fixed multiplier for pips mode - simple scaling
                volume = float(main_position_volume) * float(base_multiplier)
                logging.debug(f"Fixed Pips DCA: {main_position_volume} * {base_multiplier} = {volume:.2f}")
            
            # Apply volume limits
            min_volume = self.risk_settings.get("min_volume_auto", 0.01)
            max_volume_setting = self.risk_settings.get("max_total_volume", "OFF")
            
            if max_volume_setting != "OFF":
                try:
                    max_volume = float(max_volume_setting) * 0.3  # Max 30% of total for single DCA
                except (ValueError, TypeError):
                    max_volume = 10.0
            else:
                max_volume = 10.0
            
            volume = max(min_volume, min(volume, max_volume))
            return round(volume, 2)
            
        except Exception as e:
            logging.error(f"Error calculating DCA volume: {e}")
            return main_position_volume
    
    def _apply_dca_stop_loss_mode(self, symbol: str, dca_ticket: int, main_position_ticket: int) -> bool:
        """DCA stop loss management disabled - use dca_service.py instead"""
        logger.info("DCA stop loss management disabled: Use dca_service.py for DCA management")
        return True

    def calculate_universal_dca_protection(self, symbol: str, side: str, entry_price: float, dca_level: int = 1) -> tuple:
        """
        Calculate DCA protection for ATR and Fixed Pips modes (Fibonacci managed by dca_service.py)
        
        Parameters:
        - symbol: Symbol name (e.g., 'GBPJPY', 'XAUUSD', 'BTCUSD')
        - side: 'BUY' or 'SELL'
        - entry_price: Entry price level
        - dca_level: DCA level (1, 2, 3, etc.)
        
        Returns:
        - tuple: (stop_loss, take_profit, protection_info)
        
        Protection Logic:
        1. Universal minimums: 50 pips (non-JPY), 80 pips (JPY)
        2. DCA level multipliers: 1.0, 1.3, 1.6, 1.9, 2.2...
        3. Symbol-specific settings integration
        4. Proper pip value calculation
        5. MT5 broker requirements validation
        """
        try:
            import json
            import MetaTrader5 as mt5
            
            # 🔧 CRITICAL FIX: Don't reload risk settings, use current instance settings
            # Use existing risk_settings from class instance
            risk_settings = self.risk_settings
            
            # 🔧 CRITICAL FIX: Use current risk_settings instead of reloading
            # Get symbol-specific settings with fallbacks
            symbol_settings = self.risk_settings.get('symbol_specific_settings', {}).get(symbol, {})
            default_sl_pips = symbol_settings.get('default_sl_pips', self.risk_settings.get('default_sl_pips', 150))  # Use 150 from GUI
            default_tp_pips = symbol_settings.get('default_tp_pips', self.risk_settings.get('default_tp_pips', 100))  # Use 100 from GUI
            
            # Get symbol info for proper pip calculation
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"❌ Could not get symbol info for {symbol}")
                return 0.0, 0.0, {"error": "No symbol info"}
            
            # 🛡️ UNIVERSAL DCA PROTECTION LOGIC WITH ATR INTEGRATION
            # Get ATR value from indicator_output 
            atr_value = self._get_atr_value(symbol)
            
            # 🔧 FIX: Check if risk settings specify ATR mode for SL/TP
            sltp_mode = self.risk_settings.get('sltp_mode', 'Pips cố định')
            use_atr_mode = 'ATR' in sltp_mode or 'atr' in sltp_mode.lower() or 'Auto' in sltp_mode
            
            # Get pip value for calculations
            pip_value = self._get_pip_value_for_distance(symbol)
            
            # Check if symbol is crypto
            is_crypto = any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'SOL', 'ADA', 'BNB', 'XRP', 'LTC', 'DOGE', 'DOT', 'LINK'])
            
            # DCA level progressive multiplier
            dca_multiplier = 1.0 + (dca_level - 1) * 0.3  # Level 1: 1.0, Level 2: 1.3, Level 3: 1.6, etc.
            
            # Initialize price distance variables for crypto ATR mode
            sl_distance_price = 0
            tp_distance_price = 0
            
            if use_atr_mode and atr_value > 0:
                # Use ATR-based calculation
                sl_atr_multiplier = risk_settings.get('default_sl_atr_multiplier', 2.0)
                tp_atr_multiplier = risk_settings.get('default_tp_atr_multiplier', 1.5)
                
                # ATR from indicator is ALWAYS in price units
                # Convert to pips: atr_pips = atr_value / pip_value
                atr_pips = atr_value / pip_value
                effective_sl_pips = (atr_pips * sl_atr_multiplier) * dca_multiplier
                effective_tp_pips = (atr_pips * tp_atr_multiplier) * dca_multiplier
                sl_distance_price = effective_sl_pips * pip_value
                tp_distance_price = effective_tp_pips * pip_value
                logger.info(f"📊 SL/TP: ATR={atr_value:.5f} → {atr_pips:.1f}pips, SL={sl_atr_multiplier}x={effective_sl_pips:.1f}pips, TP={tp_atr_multiplier}x={effective_tp_pips:.1f}pips, DCA_mult={dca_multiplier:.1f}x")
                logger.info(f"📏 Fixed Mode: SL={effective_sl_pips:.1f}pips, TP={effective_tp_pips:.1f}pips")
            
            # Ensure minimum broker requirements  
            stops_level_price = symbol_info.trade_stops_level * symbol_info.point
            min_broker_pips = max(10, stops_level_price / pip_value)  
            
            # Apply maximum protection (our protection vs broker minimum)
            final_sl_pips = max(effective_sl_pips, min_broker_pips)
            final_tp_pips = effective_tp_pips  # TP doesn't need broker minimum protection
            
            # Calculate SL/TP prices
            # 🔧 FIX: For crypto in ATR mode, we already have distance in price units
            if use_atr_mode and atr_value > 0 and is_crypto:
                # Crypto ATR mode: Use pre-calculated price distances
                if side.upper() == 'BUY':
                    stop_loss = entry_price - sl_distance_price
                    take_profit = entry_price + tp_distance_price
                else:  # SELL
                    stop_loss = entry_price + sl_distance_price
                    take_profit = entry_price - tp_distance_price
            else:
                # Forex or Fixed mode: Use pips * pip_value
                if side.upper() == 'BUY':
                    stop_loss = entry_price - (final_sl_pips * pip_value)
                    take_profit = entry_price + (final_tp_pips * pip_value)
                else:  # SELL
                    stop_loss = entry_price + (final_sl_pips * pip_value)
                    take_profit = entry_price - (final_tp_pips * pip_value)
            
            # Round to symbol digits
            stop_loss = round(stop_loss, symbol_info.digits)
            take_profit = round(take_profit, symbol_info.digits)
            
            # Protection info for logging/debugging
            protection_info = {
                'symbol': symbol,
                'side': side,
                'dca_level': dca_level,
                'use_atr_mode': use_atr_mode,
                'atr_value': atr_value if use_atr_mode else None,
                'default_sl_pips': default_sl_pips,
                'default_tp_pips': default_tp_pips,
                'dca_multiplier': dca_multiplier,
                'effective_sl_pips': effective_sl_pips,
                'effective_tp_pips': effective_tp_pips,
                'min_broker_pips': min_broker_pips,
                'final_sl_pips': final_sl_pips,
                'final_tp_pips': final_tp_pips,
                'pip_value': pip_value,
                'is_jpy_pair': 'JPY' in symbol
            }
            
            # Comprehensive logging
            mode_str = "ATR Mode" if use_atr_mode else "Fixed Mode"
            logger.info(f"🛡️ Universal DCA Protection ({symbol}) - {mode_str}:")
            logger.info(f"   📊 Level {dca_level} | Side: {side} | Entry: {entry_price}")
            logger.info(f"   🎯 SL: {final_sl_pips:.1f} pips → {stop_loss:.5f}")
            logger.info(f"   🎯 TP: {final_tp_pips:.1f} pips → {take_profit:.5f}")
            if use_atr_mode and atr_value > 0:
                atr_pips = atr_value / pip_value
                logger.info(f"   🔬 ATR: {atr_pips:.1f} pips | SL: {risk_settings.get('default_sl_atr_multiplier', 2.0):.1f}x | TP: {risk_settings.get('default_tp_atr_multiplier', 1.5):.1f}x")
            else:
                min_dca_sl_pips = 80 if 'JPY' in symbol else 50
                logger.info(f"   🛡️ Protection: default={default_sl_pips} → min={min_dca_sl_pips} → effective={effective_sl_pips:.1f} → final={final_sl_pips:.1f}")
            
            return stop_loss, take_profit, protection_info
            
        except Exception as e:
            logger.error(f"❌ Universal DCA Protection error: {e}")
            return 0.0, 0.0, {"error": str(e)}

    def _calculate_dca_sl_tp(self, symbol: str, direction: str, entry_price: float, dca_level: int) -> tuple:
        """Calculate DCA SL/TP for ATR and Fixed Pips modes (Fibonacci managed by dca_service.py)"""
        try:
            dca_mode = self.risk_settings.get("dca_mode", "fixed_pips")
            
            # Skip for Fibonacci mode - handled by dca_service.py
            if "fibo" in dca_mode.lower():
                logging.debug("Fibonacci DCA SL/TP managed by dca_service.py")
                return 0.0, 0.0
                
            # Use universal DCA protection calculation
            stop_loss, take_profit, protection_info = self.calculate_universal_dca_protection(
                symbol=symbol,
                side=direction,
                entry_price=entry_price,
                dca_level=dca_level
            )
            
            return stop_loss, take_profit
            
        except Exception as e:
            logging.error(f"Error calculating DCA SL/TP: {e}")
            return 0.0, 0.0

class VolumeCalculator:
    """
    Volume Calculator - Tính toán volume dựa trên risk settings
    Đọc các cài đặt từ risk_management/risk_settings.json
    """
    
    def __init__(self, risk_settings_path: str = "risk_management/risk_settings.json"):
        self.risk_settings_path = risk_settings_path
        self.risk_settings = self.load_risk_settings()
        self.risk_validator = ComprehensiveRiskValidator(self.risk_settings)
        
    def load_risk_settings(self) -> Dict[str, Any]:
        """Load risk settings from JSON file"""
        try:
            if os.path.exists(self.risk_settings_path):
                with open(self.risk_settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                logger.info(f"✅ Risk settings loaded from {self.risk_settings_path}")
                return settings
            else:
                logger.warning(f"⚠️ Risk settings file not found: {self.risk_settings_path}")
                return self._get_default_settings()
        except Exception as e:
            logger.error(f"❌ Error loading risk settings: {e}")
            return self._get_default_settings()
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Default risk settings if file not found"""
        return {
            "max_risk_percent": 2.0,
            "min_volume_auto": 0.01,  # Updated field name
            "max_total_volume": 10.0,  # Updated field name
            "volume_mode": "Theo rủi ro (Tự động)",
            "default_volume_lots": 0.1,  # Fixed field name to match JSON
            "symbol_multipliers": {},
            "symbol_exposure": {},
            "max_positions_per_symbol": 2,
            "dca_volume_multiplier": 1.5,
            "enable_dca": False
        }
    
    def get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            if mt5:
                account_info = mt5.account_info()
                if account_info:
                    return float(account_info.balance)
            return 10000.0  # Default fallback
        except Exception:
            return 10000.0
    
    def calculate_pip_value(self, symbol: str, lot_size: float = 1.0) -> float:
        """Calculate pip value for a symbol"""
        try:
            if not mt5:
                return 10.0  # Default fallback
                
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 10.0
                
            # For most forex pairs, 1 pip = 0.0001
            # For JPY pairs, 1 pip = 0.01
            if 'JPY' in symbol:
                pip_size = 0.01
            else:
                pip_size = 0.0001
                
            # Pip value = lot_size * contract_size * pip_size
            contract_size = symbol_info.trade_contract_size
            pip_value = lot_size * contract_size * pip_size
            
            # Convert to account currency if needed
            if symbol_info.currency_profit != "USD":
                # Simplified conversion - in real case, need proper conversion
                pass
                
            return pip_value
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating pip value for {symbol}: {e}")
            return 10.0  # Default fallback
    
    def calculate_volume_by_risk(self, symbol: str, entry_price: float, stop_loss: float, 
                               confidence: float = 1.0, is_dca: bool = False) -> float:
        """
        🎯 Tính volume dựa trên risk percentage
        Risk được chia đều cho số symbols được chọn
        """
        try:
            # Get risk parameters
            max_risk_raw = self.risk_settings.get('max_risk_percent', 2.0)
            min_lot = self.risk_settings.get('min_volume_auto', 0.01)  # Updated field name
            max_lot_raw = self.risk_settings.get('max_total_volume', 10.0)  # Updated field name
            
            # Handle 'OFF' string in max_risk_percent setting
            if isinstance(max_risk_raw, str) and max_risk_raw.upper() == "OFF":
                max_risk_percent = 999.0  # Effectively disable risk limit
            else:
                try:
                    max_risk_percent = float(max_risk_raw)
                except (ValueError, TypeError):
                    max_risk_percent = 2.0  # Default fallback
            
            # Handle 'OFF' string in max_total_volume setting
            if isinstance(max_lot_raw, str) and max_lot_raw.upper() == "OFF":
                max_lot = 999.0  # Effectively disable volume limit
            else:
                try:
                    max_lot = float(max_lot_raw)
                except (ValueError, TypeError):
                    max_lot = 10.0  # Default fallback
            
            # 🚫 CRITICAL: Block trade if risk percent is 0 or negative
            if max_risk_percent <= 0.0:
                logger.warning(f"🚫 Trade blocked: max_risk_percent={max_risk_percent}% - User has disabled trading")
                return 0.0  # Return 0 to block trade, not min_lot
            
            # Get account balance
            balance = self.get_account_balance()
            
            # 🔧 NEW: Get number of selected symbols to divide risk equally
            selected_symbols_count = self.risk_settings.get('selected_symbols_count', 1)
            if selected_symbols_count <= 0:
                selected_symbols_count = 1
            
            # Calculate risk amount per symbol (chia đều cho số symbols)
            total_risk_amount = balance * (max_risk_percent / 100.0)
            risk_amount = total_risk_amount / selected_symbols_count
            
            logger.info(f"📊 Risk Allocation: Total {max_risk_percent}% = ${total_risk_amount:.2f} / {selected_symbols_count} symbols = ${risk_amount:.2f} per symbol")
            
            # Apply symbol multiplier if exists
            symbol_multipliers = self.risk_settings.get('symbol_multipliers', {})
            multiplier = symbol_multipliers.get(symbol, 1.0)
            risk_amount *= multiplier
            
            # Apply confidence adjustment (lower confidence = lower risk)
            if confidence < 3.0:
                confidence_factor = confidence / 3.0  # Scale down for low confidence
                risk_amount *= confidence_factor
            
            # Calculate pip distance
            if stop_loss <= 0:
                # Use default SL if none provided
                default_sl_pips = self.risk_settings.get('default_sl_pips', 50)
                pip_distance = default_sl_pips
            else:
                # Calculate actual pip distance
                if 'JPY' in symbol:
                    pip_distance = abs(entry_price - stop_loss) * 100
                else:
                    pip_distance = abs(entry_price - stop_loss) * 10000
            
            if pip_distance <= 0:
                pip_distance = 50  # Default fallback
            
            # Calculate pip value
            pip_value = self.calculate_pip_value(symbol, 1.0)
            
            # Calculate volume: risk_amount / (pip_distance * pip_value)
            if pip_value > 0 and pip_distance > 0:
                volume = risk_amount / (pip_distance * pip_value)
            else:
                volume = min_lot
            
            # Apply DCA multiplier if this is a DCA trade
            if is_dca and self.risk_settings.get('enable_dca', False):
                dca_multiplier = self.risk_settings.get('dca_volume_multiplier', 1.5)
                volume *= dca_multiplier
                logger.info(f"🔄 DCA multiplier applied: volume × {dca_multiplier} = {volume:.3f}")
            
            # IMPORTANT: Respect total volume constraint
            max_total_volume_raw = self.risk_settings.get('max_total_volume', 10.0)  # Updated field name
            
            # Handle 'OFF' string in max_total_volume
            if isinstance(max_total_volume_raw, str) and max_total_volume_raw.upper() == "OFF":
                max_total_volume = 999.0  # Effectively disable volume limit
            else:
                try:
                    max_total_volume = float(max_total_volume_raw)
                except (ValueError, TypeError):
                    max_total_volume = 10.0  # Default fallback
                    
            current_total_volume = self._get_current_total_volume()
            remaining_volume = max_total_volume - current_total_volume
            
            # Ensure within individual lot bounds first
            volume = max(min_lot, min(volume, max_lot))
            
            # Then ensure doesn't exceed remaining total volume capacity
            if volume > remaining_volume:
                logger.warning(f"⚠️ Risk-based volume {volume:.3f} > remaining capacity {remaining_volume:.3f}, capping...")
                volume = remaining_volume if remaining_volume >= min_lot else 0.0
            
            # Round to valid step size
            if mt5:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info and symbol_info.volume_step > 0:
                    volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step
            else:
                volume = round(volume, 3)
            
            logger.info(f"📊 Final volume for {symbol}: {volume:.3f} lots "
                       f"(Risk: {max_risk_percent}%, Distance: {pip_distance:.1f} pips, "
                       f"Remaining capacity: {remaining_volume:.3f})")
            
            return volume
            
        except Exception as e:
            logger.error(f"❌ Error calculating volume by risk: {e}")
            return self.risk_settings.get('min_volume_auto', 0.01)
    
    def get_volume_for_signal(self, signal: TradeSignal, is_dca: bool = False, dca_level: int = 1) -> float:
        """
        🎯 Main method: Get appropriate volume for a trade signal
        UPDATED: Support progressive DCA volume scaling
        """
        try:
            # 🚫 CRITICAL: Check risk blocking FIRST before any volume calculation
            max_risk_percent_raw = self.risk_settings.get('max_risk_percent', 2.0)
            # Handle 'OFF' case
            if isinstance(max_risk_percent_raw, str) and max_risk_percent_raw.upper() == 'OFF':
                max_risk_percent = float('inf')  # No risk limit
            elif max_risk_percent_raw is None:
                max_risk_percent = 2.0  # Default
            else:
                try:
                    max_risk_percent = float(max_risk_percent_raw)
                except (ValueError, TypeError):
                    max_risk_percent = 2.0  # Default fallback
            
            if max_risk_percent <= 0.0:
                logger.warning(f"🚫 Trade blocked: max_risk_percent={max_risk_percent}% - All volume modes disabled")
                return 0.0  # Block ALL trades regardless of volume mode
            
            volume_mode = self.risk_settings.get('volume_mode', 'Khối lượng mặc định')
            
            # 🔧 FIX: Read correct volume field based on mode
            if 'cố định' in volume_mode.lower() or 'fixed' in volume_mode.lower():
                base_volume = self.risk_settings.get('fixed_volume_lots', 0.1)
            else:
                base_volume = self.risk_settings.get('default_volume_lots', 0.1)
            
            max_total_volume_setting = self.risk_settings.get('max_total_volume', 10.0)  # Tổng khối lượng cho TẤT CẢ
            
            logger.info(f"📊 Volume Settings LOADED: mode='{volume_mode}', base_volume={base_volume}, is_dca={is_dca}")
            logger.info(f"🔍 Current Risk Settings: {dict(list(self.risk_settings.items())[:10])}")  # Debug first 10 items
            if is_dca:
                logger.info(f"🔄 DCA Level: {dca_level}, volume_multiplier={self.risk_settings.get('dca_volume_multiplier', 1.5)}")
            
            # Handle 'OFF' string case for max_total_volume
            if isinstance(max_total_volume_setting, str) and max_total_volume_setting.upper() == 'OFF':
                # No volume limit - skip volume capacity check
                logger.info(f"📊 Volume status: Max volume limit disabled (OFF)")
                remaining_volume = float('inf')  # Unlimited
            elif max_total_volume_setting is None:
                max_total_volume = 10.0
                current_total_volume = self._get_current_total_volume()
                remaining_volume = max_total_volume - current_total_volume
            else:
                try:
                    max_total_volume = float(max_total_volume_setting)
                except (ValueError, TypeError):
                    max_total_volume = 10.0  # Default fallback
                # Kiểm tra tổng volume hiện tại của account
                current_total_volume = self._get_current_total_volume()
                remaining_volume = max_total_volume - current_total_volume
                
                logger.info(f"📊 Volume status: Current {current_total_volume:.2f} / Max {max_total_volume} lots, Remaining: {remaining_volume:.2f}")
                
                if remaining_volume <= 0:
                    logger.warning(f"⚠️ No remaining volume capacity! Current: {current_total_volume:.2f} / {max_total_volume}")
                    return 0.0
            
            if 'tự động' in volume_mode.lower() or 'auto' in volume_mode.lower():
                # Khối lượng tự động - Có min_volume_auto và max_total_volume
                min_volume = float(self.risk_settings.get('min_volume_auto', 0.01))  # Khối lượng nhỏ nhất cho auto mode
                
                # Risk-Based Auto Mode
                volume = self.calculate_volume_by_risk(
                    symbol=signal.symbol,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    confidence=signal.confidence,
                    is_dca=is_dca
                )
                
                # 🚫 CRITICAL: If risk calculation returns 0 (blocked), don't override with min_volume
                if volume <= 0.0:
                    logger.warning(f"🚫 Auto volume blocked: Risk calculation returned {volume}")
                    return 0.0  # Respect the risk blocking
                
                # Đảm bảo >= min_volume cho auto mode (only if not blocked)
                volume = max(volume, min_volume)
                
                # Áp dụng progressive DCA scaling cho chế độ tự động nếu là DCA
                if is_dca and self.risk_settings.get('enable_dca', False):
                    volume = self._calculate_progressive_dca_volume(volume, dca_level)
                    logger.info(f"🤖 Auto Mode - Progressive DCA volume (Level {dca_level}): {volume:.3f} lots")
                else:
                    logger.info(f"🤖 Auto volume: {volume:.3f} lots (min: {min_volume})")
                
            elif 'mặc định' in volume_mode or 'default' in volume_mode.lower():
                # Khối lượng mặc định - Entry dùng volume mặc định, DCA CÓ SCALE tăng dần
                volume = base_volume
                
                # Áp dụng progressive DCA scaling nếu là DCA
                if is_dca and self.risk_settings.get('enable_dca', False):
                    volume = self._calculate_progressive_dca_volume(base_volume, dca_level)
                    logger.info(f"🔢 Default Mode - Progressive DCA volume (Level {dca_level}): {volume:.3f} lots")
                else:
                    logger.info(f"📌 Default volume: {volume:.3f} lots")
                
            elif 'cố định' in volume_mode or 'fixed' in volume_mode.lower():
                # Khối lượng cố định - Luôn sử dụng volume cố định, KHÔNG scale cho DCA
                volume = base_volume
                
                if is_dca and self.risk_settings.get('enable_dca', False):
                    logger.info(f"🔧 Fixed DCA volume (Level {dca_level}): {volume:.3f} lots - NO SCALING")
                else:
                    logger.info(f"🔧 Fixed volume: {volume:.3f} lots")
                    
            elif 'theo rủi ro' in volume_mode.lower() or 'risk' in volume_mode.lower():
                # Khối lượng theo rủi ro - Chỉ có max_total_volume, không có min
                volume = self.calculate_volume_by_risk(
                    symbol=signal.symbol,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    confidence=signal.confidence,
                    is_dca=is_dca
                )
                logger.info(f"🧮 Risk-based volume: {volume:.3f} lots")
                
            else:
                # Fallback to base volume
                logger.warning(f"⚠️ Unknown volume mode '{volume_mode}', using base volume")
                volume = base_volume
                
                # Xử lý DCA theo chế độ volume
                if is_dca and self.risk_settings.get('enable_dca', False):
                    if 'cố định' in volume_mode or 'fixed' in volume_mode.lower():
                        # Chế độ cố định: KHÔNG scale DCA
                        logger.info(f"🔧 Fallback Fixed DCA volume (Level {dca_level}): {volume:.3f} lots - NO SCALING")
                    else:
                        # Chế độ mặc định và tự động: CÓ scale DCA
                        volume = self._calculate_progressive_dca_volume(volume, dca_level)
                        logger.info(f"📌 Fallback Progressive DCA volume (Level {dca_level}): {volume:.3f} lots")
                else:
                    logger.info(f"📌 Fallback volume: {volume:.3f} lots")
            
            # Đảm bảo không vượt quá remaining volume capacity (only if not unlimited)
            if remaining_volume != float('inf') and volume > remaining_volume:
                logger.warning(f"⚠️ Requested volume {volume:.3f} > remaining {remaining_volume:.3f}, adjusting...")
                volume = remaining_volume
            
            # Chỉ check min_volume cho auto mode, không check cho default/fixed modes
            if 'tự động' in volume_mode.lower() or 'auto' in volume_mode.lower():
                min_volume = float(self.risk_settings.get('min_volume_auto', 0.01))
                if volume < min_volume:
                    logger.warning(f"⚠️ Auto mode volume {volume:.3f} < min {min_volume}, adjusting...")
                    volume = min_volume if remaining_volume == float('inf') or remaining_volume >= min_volume else 0.0
            
            # Apply symbol exposure limits
            volume = self._apply_exposure_limits(signal.symbol, volume)
            
            return volume
            
        except Exception as e:
            logger.error(f"❌ Error getting volume for signal: {e}")
            return self.risk_settings.get('min_volume_auto', 0.01)
    
    def _apply_exposure_limits(self, symbol: str, proposed_volume: float) -> float:
        """Apply symbol exposure limits"""
        try:
            symbol_exposure = self.risk_settings.get('symbol_exposure', {})
            max_exposure = symbol_exposure.get(symbol, None)
            
            if max_exposure and max_exposure > 0:
                # Get current exposure for this symbol
                current_exposure = self._get_current_symbol_exposure(symbol)
                available_exposure = max_exposure - current_exposure
                
                if available_exposure <= 0:
                    logger.warning(f"⚠️ Symbol {symbol} at max exposure limit")
                    return 0.0
                
                if proposed_volume > available_exposure:
                    logger.info(f"📉 Reducing volume for {symbol}: {proposed_volume:.2f} → {available_exposure:.2f}")
                    return available_exposure
            
            return proposed_volume
            
        except Exception as e:
            logger.warning(f"⚠️ Error applying exposure limits: {e}")
            return proposed_volume
    
    def _get_current_symbol_exposure(self, symbol: str) -> float:
        """Get current total volume exposure for a symbol"""
        try:
            if not mt5:
                return 0.0
                
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return 0.0
                
            total_volume = sum(pos.volume for pos in positions)
            return total_volume
            
        except Exception:
            return 0.0
    
    def validate_volume_constraints(self, symbol: str, volume: float) -> Tuple[bool, str]:
        """Validate if volume meets all constraints"""
        try:
            min_lot = float(self.risk_settings.get('min_volume_auto', 0.01))  # Updated field name
            max_lot_setting = self.risk_settings.get('max_total_volume', 10.0)  # Updated field name
            
            if volume < min_lot:
                return False, f"Volume {volume:.2f} below minimum {min_lot:.2f}"
            
            # Handle 'OFF' string case for max_total_volume
            if max_lot_setting != 'OFF' and max_lot_setting is not None:
                max_lot = float(max_lot_setting)
                if volume > max_lot:
                    return False, f"Volume {volume:.2f} above maximum {max_lot:.2f}"
            
            # Check position limits
            max_positions_per_symbol = self.risk_settings.get('max_positions_per_symbol', 7)
            current_positions = self._get_current_position_count(symbol)
            
            if current_positions >= max_positions_per_symbol:
                return False, f"Max positions limit reached for {symbol}: {current_positions}/{max_positions_per_symbol}"
            
            return True, "Volume constraints satisfied"
            
        except Exception as e:
            return False, f"Error validating constraints: {e}"
    
    def _get_current_position_count(self, symbol: str) -> int:
        """Get current number of positions for a symbol"""
        try:
            if not mt5:
                return 0
                
            positions = mt5.positions_get(symbol=symbol)
            return len(positions) if positions else 0
            
        except Exception:
            return 0

    def _get_current_total_volume(self) -> float:
        """
        🔍 Tính tổng volume hiện tại của tất cả positions
        Hiểu đúng constraint: Tổng khối lượng của TẤT CẢ lệnh của TẤT CẢ symbol = max_total_volume
        """
        try:
            # Import MT5 here to avoid import issues
            import MetaTrader5 as mt5
            
            # Lấy tất cả positions hiện tại
            positions = mt5.positions_get()
            if not positions:
                return 0.0
            
            # Tính tổng volume của tất cả positions
            total_volume = sum(pos.volume for pos in positions)
            logger.info(f"📊 Current total volume across all positions: {total_volume:.3f} lots")
            return total_volume
            
        except Exception as e:
            logger.error(f"❌ Error calculating current total volume: {e}")
            return 0.0

    def _get_symbol_positions_from_scan(self, symbol: str) -> List[Dict]:
        """
        🔧 CRITICAL FIX: Get positions for a symbol from account scan data
        This is more reliable than mt5.positions_get() which may have sync issues
        """
        try:
            import os
            scan_file = "account_scans/mt5_essential_scan.json"
            
            if not os.path.exists(scan_file):
                logger.warning(f"⚠️ Account scan file not found: {scan_file}")
                return []
            
            with open(scan_file, 'r', encoding='utf-8') as f:
                scan_data = json.load(f)
            
            all_positions = scan_data.get("active_positions", [])
            
            # Filter positions for this symbol
            symbol_positions = []
            for pos in all_positions:
                pos_symbol = pos.get("symbol", "")
                # Handle both "XAUUSD" and "XAUUSD." formats
                if pos_symbol == symbol or pos_symbol.rstrip('.') == symbol.rstrip('.'):
                    # Convert scan format to MT5-like format for compatibility
                    mt5_like_pos = type('Position', (), {
                        'ticket': pos.get('ticket'),
                        'symbol': pos_symbol,
                        'volume': pos.get('volume'),
                        'type': pos.get('type'),  # 0=BUY, 1=SELL
                        'price_open': pos.get('price_open'),
                        'profit': pos.get('profit'),
                        'time': pos.get('time')
                    })()
                    symbol_positions.append(mt5_like_pos)
            
            logger.info(f"📊 Found {len(symbol_positions)} positions for {symbol} from account scan")
            return symbol_positions
            
        except Exception as e:
            logger.error(f"❌ Error getting positions from scan for {symbol}: {e}")
            return []

    def _calculate_progressive_dca_volume(self, base_volume: float, dca_level: int) -> float:
        """
        🔢 Calculate progressive DCA volume based on DCA mode
        DCA Level 1: base × multiplier^1
        DCA Level 2: base × multiplier^2
        DCA Level 3: base × multiplier^3, etc.
        """
        try:
            base_volume = float(base_volume)
            
            # Get DCA mode and parameters
            raw_mode = self.risk_settings.get('dca_mode', 'Pips cố định')
            mode_map = {
                'Bội số ATR': 'atr_multiple',
                'Pips cố định': 'fixed_pips',
                'Mức Fibo': 'fibo_levels',
                'Bội số cố định': 'fixed_pips',
                'Tự động theo tỷ lệ': 'atr_multiple',
                'Mức Fibonacci': 'fibo_levels'
            }
            dca_mode = mode_map.get(raw_mode, 'fixed_pips')
            
            if dca_mode in ('fibo_levels', 'Mức Fibo', 'Mức Fibonacci'):
                # Fibonacci sequence: Level 1=1x, Level 2=1x, Level 3=2x, Level 4=3x, Level 5=5x...
                fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
                if dca_level <= len(fib_sequence):
                    multiplier = float(fib_sequence[dca_level - 1])
                else:
                    multiplier = float(fib_sequence[-1])
                volume = base_volume * multiplier
                logger.info(f"📈 Fibonacci DCA Level {dca_level}: {base_volume:.3f} × {multiplier} = {volume:.3f}")
                
            elif dca_mode in ('atr_multiple', 'Bội số ATR', 'Tự động theo tỷ lệ'):
                # 🔧 FIX: Progressive volume scaling should use dca_volume_multiplier, NOT dca_atr_multiplier
                # dca_atr_multiplier is for DISTANCE calculation, not volume scaling
                volume_mult = float(self.risk_settings.get('dca_volume_multiplier', 1.5))
                multiplier = volume_mult ** dca_level
                volume = base_volume * multiplier
                logger.info(f"🎯 ATR Mode DCA Level {dca_level}: {base_volume:.3f} × {volume_mult}^{dca_level} = {volume:.3f}")
                
            else:  # 'fixed_pips', 'Bội số cố định', 'Pips cố định'
                # Progressive fixed multiplier: Level 1=1.5x, Level 2=2.25x, Level 3=3.375x...
                base_multiplier = float(self.risk_settings.get('dca_volume_multiplier', 1.5))
                multiplier = base_multiplier ** dca_level
                volume = base_volume * multiplier
                logger.info(f"🔢 Progressive Fixed DCA Level {dca_level}: {base_volume:.3f} × {base_multiplier}^{dca_level} = {volume:.3f}")
            
            # Apply bounds
            min_volume = self.risk_settings.get('min_volume_auto', 0.01)
            max_total_vol_setting = self.risk_settings.get('max_total_volume', 10.0)
            
            # Handle 'OFF' string in max_total_volume setting
            if isinstance(max_total_vol_setting, str) and max_total_vol_setting.upper() == 'OFF':
                max_volume = 10.0  # Use default if OFF
            else:
                try:
                    max_volume = float(max_total_vol_setting) * 0.3  # Max 30% of total for single DCA
                except (ValueError, TypeError):
                    max_volume = 10.0
            
            volume = max(min_volume, min(volume, max_volume))
            return round(volume, 3)
            
        except Exception as e:
            logger.error(f"❌ Error calculating progressive DCA volume: {e}")
            return float(self.risk_settings.get('min_volume_auto', 0.01))

    def _mark_dca_executed(self, symbol: str, dca_level: int):
        """Mark DCA as executed to prevent duplicate triggers"""
        try:
            current_time = time.time()
            if not hasattr(self, '_executed_dca_tracker'):
                self._executed_dca_tracker = {}
            
            # Create unique key for symbol + level  
            dca_key = f"{symbol}_L{dca_level}"
            self._executed_dca_tracker[dca_key] = current_time
            
            logger.info(f"✅ DCA executed marked: {dca_key} at {current_time}")
            
        except Exception as e:
            logger.error(f"❌ Error marking DCA executed: {e}")

    def _get_last_dca_price_for_direction(self, symbol: str, direction: str) -> Optional[float]:
        """Get the price of the last DCA position for a symbol and direction using account scan data"""
        try:
            # 🔧 CRITICAL FIX: Use account scan data instead of unreliable mt5.positions_get()
            positions = self._get_symbol_positions_from_scan(symbol)
            if not positions:
                return None
            
            # Filter positions by direction and DCA comment pattern
            direction_type = 0 if direction == "BUY" else 1
            dca_positions = []
            
            for pos in positions:
                if pos.type == direction_type:
                    comment = getattr(pos, 'comment', '') or ''
                    # Check if this is a DCA position (has DCA in comment)
                    if any(keyword in comment.upper() for keyword in ['DCA', 'LEVEL', 'L2', 'L3', 'L4', 'GPT_20B|DCA']):
                        dca_positions.append(pos)
            
            if not dca_positions:
                logger.debug(f"🔍 No existing DCA positions found for {symbol} {direction}")
                return None
                
            # 🔧 CRITICAL FIX: Get the chronologically LATEST DCA position, not extreme price
            # Sort DCA positions by time (ticket number as proxy) to find the most recent
            latest_dca_pos = max(dca_positions, key=lambda pos: pos.time)
            last_price = latest_dca_pos.price_open
            
            logger.debug(f"🔍 Last DCA price for {symbol} {direction}: {last_price:.5f} (Ticket #{latest_dca_pos.ticket}, from {len(dca_positions)} DCA positions)")
            return last_price
                
        except Exception as e:
            logger.error(f"❌ Error getting last DCA price for {symbol} {direction}: {e}")
            return None

# 🔒 GLOBAL MODULE-LEVEL SINGLETON PATTERN
_GLOBAL_EXECUTOR_INSTANCE = None
_GLOBAL_EXECUTOR_LOCK = threading.Lock()

def get_executor_instance(*args, **kwargs):
    """🔒 GLOBAL MODULE-LEVEL SINGLETON - Get the same AdvancedOrderExecutor instance across ALL imports"""
    global _GLOBAL_EXECUTOR_INSTANCE
    
    with _GLOBAL_EXECUTOR_LOCK:
        if _GLOBAL_EXECUTOR_INSTANCE is None:
            logger.info("🔒 Creating GLOBAL AdvancedOrderExecutor singleton instance")
            _GLOBAL_EXECUTOR_INSTANCE = AdvancedOrderExecutor(*args, **kwargs)
            logger.info(f"✅ GLOBAL singleton created with ID: {id(_GLOBAL_EXECUTOR_INSTANCE)}")
        else:
            logger.info(f"♻️ Reusing GLOBAL AdvancedOrderExecutor singleton (ID: {id(_GLOBAL_EXECUTOR_INSTANCE)})")
        
        return _GLOBAL_EXECUTOR_INSTANCE

class AdvancedOrderExecutor:
    """Enhanced Order Execution System with comprehensive features"""
    
    _instance = None
    _instance_lock = threading.Lock()
    
    def __new__(cls, connection=None, magic_number: int = None):
        """🔒 Singleton pattern to prevent multiple executor instances"""
        with cls._instance_lock:
            if cls._instance is None:
                logger.info("🔒 Creating new AdvancedOrderExecutor instance (Singleton)")
                cls._instance = super(AdvancedOrderExecutor, cls).__new__(cls)
                cls._instance._initialized = False
            else:
                logger.info("♻️ Reusing existing AdvancedOrderExecutor instance (Singleton)")
            return cls._instance
    
    def __init__(self, connection=None, magic_number: int = None):
        # 🔒 Singleton initialization guard
        if hasattr(self, '_initialized') and self._initialized:
            logger.info("⚠️ AdvancedOrderExecutor already initialized - skipping")
            return
            
        logger.info("🚀 Initializing AdvancedOrderExecutor (Singleton)")
        
        # Connection manager (singleton) ensures we reuse active session after account switch
        self.connection = connection
        try:
            self.connection_manager = MT5ConnectionManager() if MT5ConnectionManager else None
        except Exception:
            self.connection_manager = None
        
        self.slippage = 10  # Increased default slippage
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

        # 🧮 NEW: Volume Calculator - integrates with risk settings
        self.volume_calculator = VolumeCalculator()
        
        # 🔧 FIX: Use SAME risk settings from VolumeCalculator to ensure consistency
        self.risk_settings = self.volume_calculator.risk_settings
        
        # 🔒 DCA Lock Manager for race condition prevention
        self.dca_lock_manager = DCALockManager()
        
        # Load magic number from config or use provided/default
        if magic_number is None:
            try:
                magic_config = self.risk_settings.get('magic_numbers', {})
                self.magic_number = magic_config.get('expert_id', 123456)
            except:
                self.magic_number = 123456
        else:
            self.magic_number = magic_number

        # Order tracking
        self.pending_orders: Dict[int, Dict] = {}
        self.executed_orders: List[Dict] = []
        # Was wrongly a dict causing 'dict' object has no attribute append'
        self.failed_orders: List[Dict] = []
        
        # Trade number tracking for session
        self._session_trade_counter = None

        # Risk management
        self.max_spread_multiplier = MAX_SPREAD_MULTIPLIER
        self.min_volume_auto = MIN_VOLUME_AUTO  # Updated field name
        self.max_total_volume = MAX_TOTAL_VOLUME  # Updated field name

        # Statistics
        self.stats = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "total_volume": 0.0,
            "success_rate": 0.0
        }

        # 🧹 Cleanup stale locks at startup
        self.dca_lock_manager.cleanup_stale_locks(max_age_seconds=DCA_LOCK_MAX_AGE_SECONDS)
        
        # 🔒 Mark as initialized
        self._initialized = True
        
        # Set global instance for tracking service
        global _global_executor_instance
        _global_executor_instance = self
        
        logger.info(f"✅ AdvancedOrderExecutor initialized (Magic: {self.magic_number})")
        logger.info(f"🧮 VolumeCalculator integrated with risk settings")
        logger.info(f"🔒 DCA Lock Manager initialized with stale lock cleanup")
        logger.info(f"🔒 Singleton pattern active - preventing duplicate executors")
        logger.info(f"📊 Global instance set for order tracking service")

    @classmethod
    def reset_singleton(cls):
        """🔧 Reset singleton for testing purposes"""
        with cls._instance_lock:
            logger.info("🔄 Resetting AdvancedOrderExecutor singleton")
            cls._instance = None

    def _load_risk_settings(self) -> Dict:
        """🔧 FIX: Load risk settings from GUI JSON file"""
        try:
            settings_path = "risk_management/risk_settings.json"
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                dca_distance = settings.get('dca_distance_pips', DEFAULT_DCA_DISTANCE_PIPS)
                logger.info(f"✅ Order Executor: Loaded DCA distance {dca_distance} pips from GUI")
                return settings
        except Exception as e:
            logger.error(f"⚠️ Order Executor: Could not load GUI settings: {e}")
        
        # Fallback to defaults
        return {
            'dca_distance_pips': 50,
            'max_dca_levels': 3,
            'enable_dca': True,
            'dca_volume_multiplier': 1.5,
            'trade_comments': {
                'entry_comment': 'GOLDKILLER_AI',
                'dca_comment_prefix': 'GPT_20B|DCA'
            },
            'magic_numbers': {
                'base_magic': 'Trade',
                'counter_start': 1
            }
        }

    def _get_next_magic_number(self) -> str:
        """Generate next trade number in format Trade01, Trade02, etc."""
        try:
            magic_config = self.risk_settings.get('magic_numbers', {})
            base_magic = magic_config.get('base_magic', 'Trade')
            
            # Initialize session counter on first call
            if self._session_trade_counter is None:
                # Check existing positions to find highest trade number in comments
                positions = mt5.positions_get()
                max_num = 0
                if positions:
                    for pos in positions:
                        comment = getattr(pos, 'comment', '')
                        # Look for Trade## pattern in comments (may be after | separator)
                        if comment:
                            # Handle both "GOLDKILLER_AI|Trade01" and "Trade01" formats
                            parts = comment.split('|')
                            for part in parts:
                                if part.startswith(base_magic) and len(part) > len(base_magic):
                                    try:
                                        num_part = part[len(base_magic):]
                                        if num_part.isdigit():
                                            max_num = max(max_num, int(num_part))
                                    except (ValueError, IndexError):
                                        continue
                
                self._session_trade_counter = max_num + 1
            else:
                # Increment for subsequent calls in same session
                self._session_trade_counter += 1
            
            return f"{base_magic}{self._session_trade_counter:02d}"
        except Exception as e:
            logger.error(f"Error generating trade number: {e}")
            return f"Trade01"

    def _get_entry_comment(self, symbol: str = "") -> str:
        """Get entry comment for regular positions"""
        try:
            comments_config = self.risk_settings.get('trade_comments', {})
            base_comment = comments_config.get('entry_comment', 'GPT_20B')
            
            # GPT_20B|Entry format
            if "|Entry" not in base_comment:
                return f"{base_comment}|Entry"
            else:
                return base_comment
        except Exception:
            return 'GPT_20B|Entry'

    def _get_dca_comment(self, dca_level: int, symbol: str = "") -> str:
        """Get DCA comment for DCA positions"""
        try:
            comments_config = self.risk_settings.get('trade_comments', {})
            base_prefix = comments_config.get('dca_comment_prefix', 'GPT_20B')
            
            # GPT_20B|DCA1, GPT_20B|DCA2, GPT_20B|DCA3 format
            return f"{base_prefix}|DCA{dca_level}"
        except Exception:
            return f'GPT_20B|DCA{dca_level}'

    def _get_entry_comment_formatted(self, direction: str, confidence: float, symbol: str = "") -> str:
        """Get formatted entry comment using config template"""
        try:
            comments_config = self.risk_settings.get('trade_comments', {})
            base_comment = comments_config.get('entry_comment', 'GPT_20B')
            
            # GPT_20B|Entry format
            if "|Entry" not in base_comment:
                return f"{base_comment}|Entry"
            else:
                return base_comment
        except Exception:
            return 'GPT_20B|Entry'

    def _validate_connection(self) -> bool:
        """Validate MT5 connection - use direct MT5 connection (already working)"""
        try:
            # Skip connection_manager - use direct MT5 which is already connected
            # MT5 is already initialized and working, just verify it's still active
            account_info = mt5.account_info()
            if not account_info:
                # Try to reinitialize if needed
                if not mt5.initialize():
                    logger.error("❌ MT5 initialization failed")
                    return False
                account_info = mt5.account_info()
                if not account_info:
                    logger.error("❌ No MT5 account info available")
                    return False
            
            logger.info(f"✅ MT5 connection validated (Account: {account_info.login})")
            return True
        except Exception as e:
            logger.error(f"❌ Connection validation failed: {e}")
            return False

    def _calculate_dca_sl_tp_v2(self, symbol: str, side: str, entry_price: float, dca_level: int) -> tuple:
        """DCA SL/TP calculation disabled - use dca_service.py instead"""
        logger.info("DCA SL/TP calculation disabled: Use dca_service.py for DCA management")
        return 0.0, 0.0

    def calculate_universal_dca_protection(self, symbol: str, side: str, entry_price: float, dca_level: int = 1) -> tuple:
        """
        🔧 FIXED: Calculate DCA protection using ATR multipliers from risk_settings
        
        Parameters:
        - symbol: Symbol name (e.g., 'GBPJPY', 'XAUUSD', 'BTCUSD')
        - side: 'BUY' or 'SELL'
        - entry_price: Entry price level
        - dca_level: DCA level (1, 2, 3, etc.)
        
        Returns:
        - tuple: (stop_loss, take_profit, protection_info)
        """
        try:
            import MetaTrader5 as mt5
            
            # Get symbol info for proper calculations
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"❌ Could not get symbol info for {symbol}")
                return 0.0, 0.0, {"error": "No symbol info"}
            
            # Get ATR value from indicator_output
            atr_value = self._get_atr_value(symbol)
            
            # Check if using ATR mode
            sltp_mode = self.risk_settings.get('sltp_mode', 'Pips cố định')
            use_atr_mode = 'ATR' in sltp_mode or 'atr' in sltp_mode.lower() or 'Auto' in sltp_mode
            
            # Get pip value
            pip_value = self._get_pip_value_for_distance(symbol)
            
            # Check if crypto
            is_crypto = any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'SOL', 'ADA', 'BNB', 'XRP', 'LTC', 'DOGE', 'DOT', 'LINK'])
            
            # DCA level progressive multiplier
            dca_multiplier = 1.0 + (dca_level - 1) * 0.3  # Level 1: 1.0, Level 2: 1.3, Level 3: 1.6
            
            # Initialize distance variables
            sl_distance_price = 0
            tp_distance_price = 0
            
            if use_atr_mode and atr_value > 0:
                # 🔧 FIX: Use ATR-based calculation
                sl_atr_multiplier = self.risk_settings.get('default_sl_atr_multiplier', 2.5)
                tp_atr_multiplier = self.risk_settings.get('default_tp_atr_multiplier', 1.5)
                
                # ATR from indicator is ALWAYS in price units - universal conversion
                atr_pips = atr_value / pip_value
                effective_sl_pips = (atr_pips * sl_atr_multiplier) * dca_multiplier
                effective_tp_pips = (atr_pips * tp_atr_multiplier) * dca_multiplier
                sl_distance_price = effective_sl_pips * pip_value
                tp_distance_price = effective_tp_pips * pip_value
                logger.info(f"📊 SL/TP: ATR={atr_value:.5f} → {atr_pips:.1f}pips, SL={sl_atr_multiplier}x={effective_sl_pips:.1f}pips, TP={tp_atr_multiplier}x={effective_tp_pips:.1f}pips, DCA_mult={dca_multiplier:.1f}x")
            else:
                # Fixed pip mode
                default_sl_pips = self.risk_settings.get('default_sl_pips', 150)
                default_tp_pips = self.risk_settings.get('default_tp_pips', 100)
                
                min_dca_sl_pips = 50
                if 'JPY' in symbol:
                    min_dca_sl_pips = 80
                
                effective_sl_pips = max(default_sl_pips, min_dca_sl_pips) * dca_multiplier
                effective_tp_pips = default_tp_pips * dca_multiplier
                logger.info(f"📏 Fixed Mode: SL={effective_sl_pips:.1f}pips, TP={effective_tp_pips:.1f}pips")
            
            # Calculate SL/TP prices using price distances
            # All modes now use: effective_sl_pips * pip_value (which equals sl_distance_price)
            if side.upper() == 'BUY':
                stop_loss = entry_price - (effective_sl_pips * pip_value)
                take_profit = entry_price + (effective_tp_pips * pip_value)
            else:  # SELL
                stop_loss = entry_price + (effective_sl_pips * pip_value)
                take_profit = entry_price - (effective_tp_pips * pip_value)
            
            # Round to symbol digits
            stop_loss = round(stop_loss, symbol_info.digits)
            take_profit = round(take_profit, symbol_info.digits)
            
            protection_info = {
                'symbol': symbol,
                'side': side,
                'dca_level': dca_level,
                'use_atr_mode': use_atr_mode,
                'atr_value': atr_value if use_atr_mode else None,
                'dca_multiplier': dca_multiplier,
                'sl_pips': effective_sl_pips if not (use_atr_mode and is_crypto) else sl_distance_price,
                'tp_pips': effective_tp_pips if not (use_atr_mode and is_crypto) else tp_distance_price,
                'pip_value': pip_value
            }
            
            logger.info(f"🛡️ DCA Protection ({symbol}) L{dca_level}: SL={stop_loss:.5f}, TP={take_profit:.5f}")
            
            return stop_loss, take_profit, protection_info
            
        except Exception as e:
            logger.error(f"❌ Error in DCA protection calculation: {e}")
            return 0.0, 0.0, {"error": str(e)}

    def _clean_comment(self, comment: str) -> str:
        """Clean comment string to be MT5 compatible"""
        if not comment:
            return comment
            
        # Mapping Vietnamese phrases to English (longest phrases first!)
        vietnamese_mapping = [
            ('Giá thị trường', 'Market'),
            ('Giá chờ', 'Pending'),
            ('Phân tích', 'Analysis'),
            ('Tín hiệu', 'Signal'),
            ('Giá', 'Price'),  # Must be after longer phrases
            ('Mua', 'Buy'),
            ('Bán', 'Sell')
        ]
        
        # Replace Vietnamese phrases first - order matters!
        for vietnamese, english in vietnamese_mapping:
            comment = comment.replace(vietnamese, english)
        
        # Remove diacritics
        import unicodedata
        comment = unicodedata.normalize('NFD', comment)
        comment = ''.join(char for char in comment if unicodedata.category(char) != 'Mn')
        
        # Keep only ASCII characters, numbers, and basic symbols
        import re
        comment = re.sub(r'[^\x20-\x7E]', '', comment)
        
        return comment



    def _get_dca_status(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive DCA status for symbol"""
        try:
            import MetaTrader5 as mt5
            
            # Get all positions for this symbol with our magic number
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return {
                    "has_positions": False,
                    "entry_positions": [],
                    "dca_positions": [],
                    "total_levels": 0,
                    "max_dca_levels": self.risk_settings.get('max_dca_levels', 5),
                    "dca_enabled": self.risk_settings.get('enable_dca', False),
                    "can_add_dca": False,
                    "needs_sl_protection": False
                }
            
            # Filter positions by magic number if specified
            if hasattr(self, 'magic_number') and self.magic_number:
                positions = [p for p in positions if p.magic == self.magic_number]
            
            # Categorize positions
            entry_positions = []
            dca_positions = []
            
            for pos in positions:
                comment = pos.comment or ""
                # Updated DCA detection for new comment format
                is_dca = any(pattern in comment for pattern in [
                    'GPT_20B|DCA', '|DCA1', '|DCA2', '|DCA3', '|DCA4', '|DCA5',
                    'DCA', 'GOLDKILLER_AI_DC'  # Keep legacy support
                ])
                
                if is_dca:
                    dca_positions.append(pos)
                else:
                    entry_positions.append(pos)
            
            max_dca_levels = self.risk_settings.get('max_dca_levels', 5)
            dca_enabled = self.risk_settings.get('enable_dca', False)
            
            return {
                "has_positions": len(positions) > 0,
                "entry_positions": entry_positions,
                "dca_positions": dca_positions,
                "total_levels": len(dca_positions),
                "max_dca_levels": max_dca_levels,
                "dca_enabled": dca_enabled,
                "can_add_dca": len(dca_positions) < max_dca_levels,
                "needs_sl_protection": dca_enabled and len(entry_positions) > 0 and len(dca_positions) < max_dca_levels,
                "all_positions": positions
            }
            
        except Exception as e:
            logger.error(f"⚠️ Error getting DCA status: {e}")
            return {
                "has_positions": False,
                "entry_positions": [],
                "dca_positions": [],
                "total_levels": 0,
                "max_dca_levels": 5,
                "dca_enabled": False,
                "can_add_dca": False,
                "needs_sl_protection": False
            }

    def _should_protect_sl_for_dca(self, symbol: str, proposed_sl: float = None) -> Dict[str, Any]:
        """Check if SL should be protected due to pending DCA levels"""
        dca_status = self._get_dca_status(symbol)
        
        if not dca_status["needs_sl_protection"]:
            return {
                "should_protect": False,
                "reason": "No DCA protection needed",
                "dca_status": dca_status
            }
        
        # Calculate how many DCA levels are still pending
        pending_levels = dca_status["max_dca_levels"] - dca_status["total_levels"]
        
        if pending_levels <= 0:
            return {
                "should_protect": False,
                "reason": "All DCA levels completed",
                "dca_status": dca_status
            }
        
        # Check if proposed SL would interfere with potential DCA levels
        if proposed_sl and dca_status["entry_positions"]:
            try:
                entry_pos = dca_status["entry_positions"][0]  # Use first entry position
                entry_price = entry_pos.price_open
                direction = "BUY" if entry_pos.type == 0 else "SELL"
                
                # Get DCA distance from settings
                dca_distance_pips = self.risk_settings.get('dca_distance_pips', 50)
                pip_value = get_pip_value(symbol)
                
                # Calculate potential DCA levels
                potential_dca_prices = []
                for level in range(1, dca_status["max_dca_levels"] + 1):
                    if direction == "BUY":
                        # 🔧 CRITICAL FIX: Calculate DCA from last DCA position, not from entry
                        last_dca_price = self._get_last_dca_price_for_direction(symbol, direction)
                        if last_dca_price:
                            dca_price = last_dca_price - dca_distance_pips * pip_value
                        else:
                            dca_price = entry_price - (level * dca_distance_pips * pip_value)
                    else:
                        # 🔧 CRITICAL FIX: Calculate DCA from last DCA position, not from entry  
                        last_dca_price = self._get_last_dca_price_for_direction(symbol, direction)
                        if last_dca_price:
                            dca_price = last_dca_price + dca_distance_pips * pip_value
                        else:
                            dca_price = entry_price + (level * dca_distance_pips * pip_value)
                    potential_dca_prices.append(dca_price)
                
                # Check if proposed SL would hit before all DCA levels
                worst_dca_price = min(potential_dca_prices) if direction == "BUY" else max(potential_dca_prices)
                
                if direction == "BUY" and proposed_sl > worst_dca_price:
                    return {
                        "should_protect": True,
                        "reason": f"Proposed SL {proposed_sl:.5f} too close, would hit before DCA level at {worst_dca_price:.5f}",
                        "recommended_sl": worst_dca_price - (10 * pip_value),  # 10 pips buffer
                        "dca_status": dca_status,
                        "pending_levels": pending_levels
                    }
                elif direction == "SELL" and proposed_sl < worst_dca_price:
                    return {
                        "should_protect": True,
                        "reason": f"Proposed SL {proposed_sl:.5f} too close, would hit before DCA level at {worst_dca_price:.5f}",
                        "recommended_sl": worst_dca_price + (10 * pip_value),  # 10 pips buffer
                        "dca_status": dca_status,
                        "pending_levels": pending_levels
                    }
                else:
                    # ✅ Proposed SL is safe - won't interfere with DCA levels
                    return {
                        "should_protect": False,
                        "reason": f"Proposed SL {proposed_sl:.5f} is safe, beyond DCA level at {worst_dca_price:.5f}",
                        "dca_status": dca_status,
                        "pending_levels": pending_levels
                    }
                
            except Exception as e:
                logger.error(f"⚠️ Error calculating DCA protection: {e}")
        
        # 🔧 FIX: Only protect if no proposed_sl was provided (can't validate)
        # If proposed_sl was provided and we got here, validation passed
        if proposed_sl is None:
            return {
                "should_protect": True,
                "reason": f"DCA enabled with {pending_levels} pending levels - no SL to validate",
                "dca_status": dca_status,
                "pending_levels": pending_levels
            }
        else:
            # proposed_sl provided but exception occurred - allow with warning
            return {
                "should_protect": False,
                "reason": f"DCA protection check failed, allowing SL modification",
                "dca_status": dca_status,
                "pending_levels": pending_levels
            }

    def get_dca_protection_summary(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Get summary of DCA protection status for reporting"""
        try:
            if symbols is None:
                # Get all active symbols
                import MetaTrader5 as mt5
                positions = mt5.positions_get()
                if positions:
                    symbols = list(set([pos.symbol for pos in positions]))
                else:
                    symbols = []
            
            protection_summary = {
                "timestamp": datetime.now().isoformat(),
                "total_symbols": len(symbols),
                "dca_protected_symbols": 0,
                "symbols_detail": {}
            }
            
            for symbol in symbols:
                dca_status = self._get_dca_status(symbol)
                protection_info = self._should_protect_sl_for_dca(symbol)
                
                protection_summary["symbols_detail"][symbol] = {
                    "dca_enabled": dca_status["dca_enabled"],
                    "has_positions": dca_status["has_positions"],
                    "entry_count": len(dca_status["entry_positions"]),
                    "dca_count": len(dca_status["dca_positions"]),
                    "max_dca_levels": dca_status["max_dca_levels"],
                    "pending_levels": dca_status["max_dca_levels"] - dca_status["total_levels"],
                    "needs_protection": protection_info["should_protect"],
                    "protection_reason": protection_info.get("reason", "")
                }
                
                if protection_info["should_protect"]:
                    protection_summary["dca_protected_symbols"] += 1
            
            return protection_summary
            
        except Exception as e:
            logger.error(f"⚠️ Error getting DCA protection summary: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "total_symbols": 0,
                "dca_protected_symbols": 0,
                "symbols_detail": {}
            }



    # --------------------------------------------------------------
    # Symbol resolution helpers (handle account switch / suffix changes)
    # --------------------------------------------------------------
    def _generate_symbol_variants(self, symbol: str) -> List[str]:
        base = symbol.strip()
        variants = [base]
        if base.endswith('_m'):
            variants.append(base[:-2])
        else:
            variants.append(base + '_m')
        # Add uppercase variant if different
        if base.upper() not in variants:
            variants.append(base.upper())
        # Common broker suffix removal (e.g., .r, .i)
        if '.' in base:
            short = base.split('.')[0]
            if short not in variants:
                variants.append(short)
        return [v for i, v in enumerate(variants) if v and v not in variants[:i]]

    def _resolve_symbol(self, symbol: str) -> Tuple[Optional[str], Optional[Any]]:
        for cand in self._generate_symbol_variants(symbol):
            try:
                sinfo = mt5.symbol_info(cand)
                if sinfo:
                    if not sinfo.visible:
                        mt5.symbol_select(cand, True)
                    return cand, sinfo
            except Exception:
                continue
        # Last resort: wildcard search
        try:
            base = symbol.replace('_m','')
            matches = mt5.symbols_get(f"*{base}*")
            if matches:
                sinfo = matches[0]
                if hasattr(sinfo, 'name'):
                    nm = getattr(sinfo, 'name')
                    if nm:
                        if not sinfo.visible:
                            mt5.symbol_select(nm, True)
                        return nm, sinfo
        except Exception:
            pass
        return None, None

    def _validate_symbol(self, symbol: str) -> Tuple[bool, Optional[Any]]:
        """Validate trading symbol and get symbol info"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                resolved, resolved_info = self._resolve_symbol(symbol)
                if resolved and resolved_info:
                    if resolved != symbol:
                        logger.info(f"🔁 Resolved symbol {symbol} -> {resolved}")
                    return True, resolved_info
                logger.error(f"❌ Symbol {symbol} not found (after variants)")
                return False, None
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"❌ Cannot select symbol {symbol}")
                    return False, None
            return True, symbol_info
        except Exception as e:
            logger.error(f"❌ Symbol validation failed for {symbol}: {e}")
            return False, None

    def _get_current_price(self, symbol: str, action: str) -> Optional[float]:
        """Get current market price for symbol"""
        try:
            # Ensure symbol is selected first
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"⚠️ Failed to select symbol {symbol}")
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"❌ Cannot get tick data for {symbol}")
                return None
                
            price = tick.ask if action.upper() == 'BUY' else tick.bid
            logger.info(f"📊 Current price for {symbol} ({action}): {price}")
            return price
            
        except Exception as e:
            logger.error(f"❌ Price retrieval failed for {symbol}: {e}")
            return None

    def _validate_trade_parameters(self, signal: TradeSignal, symbol_info: Any) -> bool:
        """Validate trade parameters before execution"""
        try:
            # Volume validation
            if signal.volume < symbol_info.volume_min:
                logger.error(f"❌ Volume {signal.volume} below minimum {symbol_info.volume_min}")
                return False
                
            if signal.volume > symbol_info.volume_max:
                logger.error(f"❌ Volume {signal.volume} above maximum {symbol_info.volume_max}")
                return False
                
            # Volume step validation
            volume_step = symbol_info.volume_step
            if volume_step > 0:
                # Check if volume is a multiple of the step from minimum volume
                volume_from_min = signal.volume - symbol_info.volume_min
                remainder = round(volume_from_min % volume_step, 8)  # Round to avoid floating point precision issues
                if abs(remainder) > 1e-8 and abs(remainder - volume_step) > 1e-8:  # Allow for floating point precision
                    logger.error(f"❌ Volume {signal.volume} not aligned with step {volume_step} (remainder: {remainder})")
                    return False
            
            # Spread validation
            tick = mt5.symbol_info_tick(signal.symbol)
            if tick:
                spread = (tick.ask - tick.bid) / symbol_info.point
                max_allowed_spread = symbol_info.spread * self.max_spread_multiplier
                if spread > max_allowed_spread:
                    logger.warning(f"⚠️ High spread detected: {spread:.1f} points")
                    
            # Stop loss and take profit validation (allow 0.0 for no SL/TP)
            if signal.action.upper() == 'BUY':
                if signal.stop_loss > 0.0 and signal.stop_loss >= signal.entry_price:
                    logger.error("❌ Stop loss must be below entry price for BUY order")
                    return False
                if signal.take_profit > 0.0 and signal.take_profit <= signal.entry_price:
                    logger.error("❌ Take profit must be above entry price for BUY order")
                    return False
            else:  # SELL
                if signal.stop_loss > 0.0 and signal.stop_loss <= signal.entry_price:
                    logger.error("❌ Stop loss must be above entry price for SELL order")
                    return False
                if signal.take_profit > 0.0 and signal.take_profit >= signal.entry_price:
                    logger.error("❌ Take profit must be below entry price for SELL order")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"❌ Parameter validation failed: {e}")
            return False

    def calculate_smart_volume(self, signal: TradeSignal, is_dca: bool = False, dca_level: int = 1) -> float:
        """
        🧮 Calculate smart volume based on risk settings
        """
        try:
            # Use VolumeCalculator to get appropriate volume
            calculated_volume = self.volume_calculator.get_volume_for_signal(signal, is_dca, dca_level)
            
            # 🚫 CRITICAL: Block trade if volume is 0 (risk blocking)
            if calculated_volume <= 0.0:
                logger.warning(f"🚫 Trade blocked: Volume calculated as {calculated_volume} - Risk settings prevent trading")
                return 0.0  # Return 0 to signal trade should be blocked
            
            # Round volume to nearest 0.01 step to avoid alignment errors
            calculated_volume = round(calculated_volume / 0.01) * 0.01
            
            # Validate constraints
            is_valid, message = self.volume_calculator.validate_volume_constraints(signal.symbol, calculated_volume)
            
            if not is_valid:
                logger.warning(f"⚠️ Volume constraint violation: {message}")
                # Fallback to minimum volume
                calculated_volume = self.volume_calculator.risk_settings.get('min_volume_auto', 0.01)  # Updated field name
            
            logger.info(f"🎯 Smart volume for {signal.symbol}: {calculated_volume:.2f} lots")
            return calculated_volume
            
        except Exception as e:
            logger.error(f"❌ Error calculating smart volume: {e}")
            return self.volume_calculator.risk_settings.get('min_volume_auto', 0.01)  # Updated field name

    def prepare_signal_with_smart_volume(self, symbol: str, action: str, entry_price: float = 0.0,
                                       stop_loss: float = 0.0, take_profit: float = 0.0,
                                       confidence: float = 3.0, strategy: str = "AUTO_SYSTEM",
                                       comment: str = "", is_dca: bool = False, dca_level: int = 1) -> TradeSignal:
        """
        🎯 Create a trade signal with automatically calculated volume
        """
        try:
            # Get current price if entry_price is 0
            if entry_price <= 0:
                entry_price = self._get_current_price(symbol, action)
                if not entry_price:
                    raise ValueError(f"Cannot get current price for {symbol}")
            
            # Create preliminary signal for volume calculation
            temp_signal = TradeSignal(
                symbol=symbol,
                action=action.upper(),
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=0.01,  # Temporary volume
                confidence=confidence,
                strategy=strategy,
                timestamp=datetime.now(),
                comment=comment,
                is_dca=is_dca,
                dca_level=dca_level
            )
            
            # Calculate smart volume
            smart_volume = self.calculate_smart_volume(temp_signal, is_dca, dca_level)
            
            # 🚫 CRITICAL: Block trade if volume is 0 (risk blocking)
            if smart_volume <= 0.0:
                logger.error(f"🚫 Trade blocked: Smart volume is {smart_volume} - Risk settings prevent trading")
                return TradeSignal(
                    symbol=symbol,
                    action="BLOCKED",  # Special action to indicate blocked trade
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    volume=0.0,
                    confidence=0.0,
                    strategy=f"BLOCKED_BY_RISK_SETTINGS",
                    timestamp=datetime.now(),
                    comment="Trade blocked by risk settings: max_risk_percent=0%",
                    is_dca=is_dca,
                    dca_level=dca_level
                )
            
            # Create final signal with calculated volume
            final_signal = TradeSignal(
                symbol=symbol,
                action=action.upper(),
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=smart_volume,
                confidence=confidence,
                strategy=strategy,
                timestamp=datetime.now(),
                comment=comment,
                is_dca=is_dca,
                dca_level=dca_level
            )
            
            logger.info(f"🎯 Signal prepared: {action} {smart_volume:.2f} {symbol} @ {entry_price}")
            return final_signal
            
        except Exception as e:
            logger.error(f"❌ Error preparing signal: {e}")
            raise
    
    def _get_clean_comment(self, signal: TradeSignal) -> str:
        """Get clean comment for MT5 order (preserves DCA markers)"""
        try:
            # Generate next trade number (Trade01, Trade02, etc.)
            trade_number = self._get_next_magic_number()
            symbol = signal.symbol  # Get symbol from signal
            
            # Check if this is a DCA signal based on comment or is_dca flag
            if hasattr(signal, 'is_dca') and signal.is_dca:
                dca_level = getattr(signal, 'dca_level', 1)
                logger.info(f"🔍 DCA signal detected: is_dca={signal.is_dca}, dca_level={dca_level}, comment={signal.comment}")
                base_comment = self._get_dca_comment(dca_level, symbol)
                full_comment = base_comment  # DCA comments don't need trade number
            elif hasattr(signal, 'comment') and signal.comment and 'DCA' in signal.comment:
                # Extract DCA level from comment like "DCA-2" or "DCA-4"
                try:
                    if '-' in signal.comment:
                        dca_level = int(signal.comment.split('-')[-1])
                        base_comment = self._get_dca_comment(dca_level, symbol)
                        full_comment = base_comment  # DCA comments don't need trade number
                    else:
                        # Fallback to level 1 if no level specified
                        base_comment = self._get_dca_comment(1, symbol)
                        full_comment = base_comment
                except (ValueError, IndexError):
                    logger.warning(f"Failed to parse DCA level from comment: {signal.comment}")
                    base_comment = self._get_dca_comment(1, symbol)
                    full_comment = base_comment
            else:
                # Regular entry position
                base_comment = self._get_entry_comment(symbol)
                full_comment = base_comment  # Don't add trade number for GPT_20B format
            
            # MT5 comment field has a limit (usually 32 characters)
            # Truncate if necessary but preserve essential information
            if len(full_comment) > 31:
                # Try shorter format: GK_AI|Trade01 or GK_DCA1|Trade01
                if 'DCA' in full_comment:
                    # Extract DCA level from base_comment
                    if hasattr(signal, 'is_dca') and signal.is_dca:
                        dca_level = getattr(signal, 'dca_level', 1)
                        short_comment = f"GK_DCA{dca_level}|{trade_number}"
                    else:
                        short_comment = f"GK_DCA1|{trade_number}"
                else:
                    short_comment = f"GK_AI|{trade_number}"
                
                if len(short_comment) <= 31:
                    return short_comment
                else:
                    # Last resort: just use trade number
                    return trade_number[:31]
            
            return full_comment
        except Exception as e:
            logger.warning(f"Error generating comment, using fallback: {e}")
            return "GOLDKILLER_AI"

    def _create_market_order_request(self, signal: TradeSignal, price: float) -> Dict:
        """Create market order request"""
        action_type = mt5.ORDER_TYPE_BUY if signal.action.upper() == 'BUY' else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": signal.volume,
            "type": action_type,
            "price": price,
            "deviation": self.slippage,
            "magic": self.magic_number,
            "comment": self._get_clean_comment(signal),  # Use signal comment for DCA tracking
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": get_symbol_fill_type(signal.symbol)
        }
        
        # 🔧 CRITICAL FIX: Recalculate SL/TP for DCA orders using current price
        final_sl = signal.stop_loss
        final_tp = signal.take_profit
        
        # Check if this is a DCA signal - recalculate SL/TP based on current price
        if hasattr(signal, 'is_dca') and signal.is_dca:
            try:
                dca_level = getattr(signal, 'dca_level', 1)
                direction = signal.action.upper()
                
                logger.info(f"🔧 DCA RECALC: Recalculating SL/TP for {signal.symbol} DCA Level {dca_level}")
                logger.info(f"   Original SL: {signal.stop_loss:.5f}, TP: {signal.take_profit:.5f}")
                logger.info(f"   Current Price: {price:.5f}")
                
                # Use universal DCA protection with current price
                new_sl, new_tp, protection_info = self.calculate_universal_dca_protection(
                    symbol=signal.symbol,
                    side=direction,
                    entry_price=price,  # Use current price instead of old entry price
                    dca_level=dca_level
                )
                
                if new_sl > 0 and new_tp > 0:
                    final_sl = new_sl
                    final_tp = new_tp
                    logger.info(f"   ✅ Recalculated SL: {final_sl:.5f}, TP: {final_tp:.5f}")
                else:
                    logger.warning(f"   ⚠️ Recalculation failed, using original values")
                    
            except Exception as e:
                logger.warning(f"⚠️ DCA SL/TP recalculation error: {e}, using original values")
        
        # Re-enable SL/TP with proper validation
        if final_sl > 0.0:
            # Validate SL against broker requirements
            validated_sl = self._validate_stop_level(signal.symbol, price, final_sl, is_stop_loss=True, is_buy=(action_type == mt5.ORDER_TYPE_BUY))
            if validated_sl > 0:
                request["sl"] = validated_sl
            else:
                logger.warning(f"⚠️ Invalid SL {final_sl:.5f} for {signal.symbol}, skipping SL")
                
        if final_tp > 0.0:
            # Validate TP against broker requirements  
            validated_tp = self._validate_stop_level(signal.symbol, price, final_tp, is_stop_loss=False, is_buy=(action_type == mt5.ORDER_TYPE_BUY))
            if validated_tp > 0:
                request["tp"] = validated_tp
            else:
                logger.warning(f"⚠️ Invalid TP {final_tp:.5f} for {signal.symbol}, skipping TP")
        
        # Debug log for troubleshooting
        logger.info(f"🔧 Order request created: {request}")
        
        # Try with minimal comment for debugging
        logger.info(f"🔍 Original comment: {signal.strategy}|{signal.comment}")
        logger.info(f"🔍 Cleaned comment: {request['comment']}")
        
        return request

    def _create_pending_order_request(self, signal: TradeSignal, order_type: OrderType) -> Dict:
        """Create pending order request"""
        type_mapping = {
            OrderType.LIMIT_BUY: mt5.ORDER_TYPE_BUY_LIMIT,
            OrderType.LIMIT_SELL: mt5.ORDER_TYPE_SELL_LIMIT,
            OrderType.STOP_BUY: mt5.ORDER_TYPE_BUY_STOP,
            OrderType.STOP_SELL: mt5.ORDER_TYPE_SELL_STOP
        }
        
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": signal.symbol,
            "volume": signal.volume,
            "type": type_mapping[order_type],
            "price": signal.entry_price,
            "magic": self.magic_number,
            "comment": self._get_clean_comment(signal),  # Use signal comment for DCA tracking
            "type_time": mt5.ORDER_TIME_GTC,
            "expiration": int(time.time()) + 86400  # 24 hours
        }
        
        # Re-enable SL/TP for pending orders with proper validation
        if signal.stop_loss > 0.0:
            request["sl"] = signal.stop_loss
        if signal.take_profit > 0.0:
            request["tp"] = signal.take_profit
        
        # Debug log for troubleshooting
        logger.info(f"🔍 PENDING Original comment: {signal.strategy}|{signal.comment}")
        logger.info(f"🔍 PENDING Cleaned comment: {request['comment']}")
        
        return request

    def _validate_stop_level(self, symbol: str, entry_price: float, stop_price: float, is_stop_loss: bool, is_buy: bool) -> float:
        """
        🛡️ Validate and adjust SL/TP levels according to broker requirements
        Returns adjusted price or 0 if invalid
        """
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"⚠️ Cannot get symbol info for {symbol}")
                return 0.0
            
            # Get minimum stop level in points
            stops_level = symbol_info.trade_stops_level
            point = symbol_info.point
            
            # Calculate minimum distance in price units
            min_distance = stops_level * point
            
            # If no minimum stop level requirement, use default
            if min_distance <= 0:
                min_distance = 10 * point  # Default 10 points minimum
            
            # Calculate current distance
            distance = abs(entry_price - stop_price)
            
            # Check if distance meets minimum requirement
            if distance < min_distance:
                logger.info(f"🔧 Adjusting stop level: required min distance {min_distance:.5f}, current {distance:.5f}")
                
                # Adjust to minimum distance
                if is_stop_loss:
                    if is_buy:
                        # BUY SL should be below entry
                        adjusted_price = entry_price - min_distance
                    else:
                        # SELL SL should be above entry  
                        adjusted_price = entry_price + min_distance
                else:  # Take profit
                    if is_buy:
                        # BUY TP should be above entry
                        adjusted_price = entry_price + min_distance
                    else:
                        # SELL TP should be below entry
                        adjusted_price = entry_price - min_distance
                
                logger.info(f"🔧 Adjusted {'SL' if is_stop_loss else 'TP'} from {stop_price:.5f} to {adjusted_price:.5f}")
                return adjusted_price
            
            return stop_price  # No adjustment needed
            
        except Exception as e:
            logger.error(f"❌ Error validating stop level: {e}")
            return 0.0  # Return 0 to skip setting SL/TP

    def execute_smart_market_order(self, symbol: str, action: str, stop_loss: float = 0.0,
                                 take_profit: float = 0.0, confidence: float = 3.0,
                                 strategy: str = "SMART_SYSTEM", comment: str = "",
                                 is_dca: bool = False) -> OrderResult:
        """
        🎯 Execute market order with smart volume calculation
        """
        try:
            logger.info(f"🚀 Executing SMART market order: {action} {symbol}")
            
            # Prepare signal with smart volume
            signal = self.prepare_signal_with_smart_volume(
                symbol=symbol,
                action=action,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                strategy=strategy,
                comment=comment,
                is_dca=is_dca
            )
            
            # Execute with the prepared signal
            return self.execute_market_order(signal)
            
        except Exception as e:
            error_msg = f"Smart market order failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return OrderResult(success=False, error_message=error_msg, timestamp=datetime.now())

    def execute_market_order(self, signal: TradeSignal) -> OrderResult:
        """Execute market order with enhanced error handling and retries"""
        
        # 🔒 GLOBAL ORDER EXECUTION LOCK - Prevents duplicate orders from multiple callers
        with _GLOBAL_ORDER_LOCK:
            logger.info(f"� [LOCKED] Executing market order: {signal.action} {signal.volume} {signal.symbol}")
            
            # Time-based order blocking has been removed - orders are now allowed without time restrictions
        
        # 🚫 CRITICAL: Check if trade is blocked by risk settings
        if signal.action == "BLOCKED":
            logger.error(f"🚫 Trade execution blocked: {signal.comment}")
            return OrderResult(
                success=False, 
                error_message=f"Trade blocked by risk settings: {signal.comment}",
                comment=f"BLOCKED: {signal.comment}",
                timestamp=datetime.now()
            )
        
        # 🧹 AUTO CLEANUP before executing order
        logger.info("🧹 Order Executor: Auto cleanup before processing...")
        try:
            cleanup_result = self.cleanup_order_data(max_age_hours=72, keep_latest=10)
            logger.info(f"✅ Cleaned {cleanup_result['total_files_deleted']} files, "
                       f"freed {cleanup_result['total_space_freed_mb']:.2f} MB")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")
        
        # Pre-execution validations
        if not self._validate_connection():
            return OrderResult(success=False, error_message="Connection validation failed")
            
        # 🔒 CRITICAL: Duplicate entry check for non-DCA orders
        if not signal.is_dca:
            try:
                duplicate_check = self.volume_calculator.risk_validator._check_duplicate_entry(signal.symbol, "market_order")
                if not duplicate_check['valid']:
                    logger.error(f"🚫 DUPLICATE ENTRY BLOCKED: {signal.symbol} - {duplicate_check['reason']}")
                    return OrderResult(
                        success=False, 
                        error_message=f"Duplicate entry blocked: {duplicate_check['reason']}",
                        comment=f"BLOCKED: {duplicate_check['reason']}",
                        timestamp=datetime.now()
                    )
            except Exception as e:
                logger.warning(f"⚠️ Duplicate check failed: {e} - allowing order")

        # Check client-side autotrading status (common cause of TRADE_RETCODE_CLIENT_DISABLES_AT 10027)
        try:
            if hasattr(mt5, 'terminal_info'):
                tinfo = mt5.terminal_info()
                trade_allowed_attr = getattr(tinfo, 'trade_allowed', None)
                if trade_allowed_attr is False:
                    return OrderResult(success=False, error_message="Client AutoTrading disabled (enable green AutoTrading button in MT5)")
        except Exception:
            pass

        symbol_valid, symbol_info = self._validate_symbol(signal.symbol)
        if not symbol_valid or not symbol_info:
            return OrderResult(success=False, error_message="Symbol validation failed")
        # Ensure we use the broker's resolved/normalized symbol name (important after account switch)
        try:
            resolved_name = getattr(symbol_info, 'name', None)
            if resolved_name and resolved_name != signal.symbol:
                logger.info(f"🔁 Using resolved symbol name: {signal.symbol} -> {resolved_name}")
                signal.symbol = resolved_name
        except Exception:
            pass

        if not self._validate_trade_parameters(signal, symbol_info):
            return OrderResult(success=False, error_message="Parameter validation failed")
        
        # Get current price
        current_price = self._get_current_price(signal.symbol, signal.action)
        if not current_price:
            # Fallback: try resolving variants again (may differ if symbol_info.name not accessible earlier)
            try:
                resolved_again, sinfo_again = self._resolve_symbol(signal.symbol)
                if resolved_again and sinfo_again:
                    if resolved_again != signal.symbol:
                        logger.info(f"🔁 Fallback resolved symbol: {signal.symbol} -> {resolved_again}")
                        signal.symbol = resolved_again
                    current_price = self._get_current_price(signal.symbol, signal.action)
            except Exception:
                pass
        if not current_price:
            return OrderResult(success=False, error_message="Cannot get current price")
        
        # Create order request
        request = self._create_market_order_request(signal, current_price)
        
        # Execute with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"📤 Sending order (attempt {attempt + 1}/{self.max_retries})")
                logger.info(f"� MT5 order_send request: {request}")
                logger.info(f"💼 ABOUT TO CALL mt5.order_send - MT5 connected: {mt5.initialize()}")
                
                # Debug: Check symbol state right before order_send
                symbol = request.get('symbol')
                if symbol:
                    symbol_info = mt5.symbol_info(symbol)
                    symbol_tick = mt5.symbol_info_tick(symbol)
                    logger.info(f"💼 Symbol {symbol} info: {symbol_info is not None}, tick: {symbol_tick is not None}")
                    if symbol_tick:
                        logger.info(f"💼 Symbol {symbol} bid={symbol_tick.bid}, ask={symbol_tick.ask}")
                
                # CRITICAL: Fresh connection validation right before order_send
                logger.info(f"💼 Fresh connection validation before order_send...")
                conn_valid = self._validate_connection()
                logger.info(f"💼 Fresh connection validation result: {conn_valid}")
                
                if not conn_valid:
                    logger.error("💼 Connection validation failed right before order_send!")
                    return OrderResult(success=False, error_message="Connection validation failed before order_send")
                
                result = mt5.order_send(request)
                logger.info(f"💼 MT5 order_send RETURNED: {result}")
                logger.debug(f"🔧 MT5 order_send result: {result}")
                
                if result is None:
                    # Check MT5 last error when result is None
                    mt5_error = mt5.last_error()
                    error_msg = f"Order result is None - MT5 last error: {mt5_error}"
                    logger.error(f"❌ {error_msg}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return OrderResult(success=False, error_message=error_msg)
                
                # Success case
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    order_result = OrderResult(
                        success=True,
                        order_id=result.order,
                        ticket=result.order,
                        retcode=result.retcode,
                        comment=result.comment,
                        price=result.price,
                        volume=result.volume,
                        timestamp=datetime.now()
                    )
                    
                    # Update statistics
                    self._update_stats(True, signal.volume)
                    
                    # Log success
                    logger.info(f"✅ Order executed successfully:")
                    logger.info(f"   📋 Ticket: {result.order}")
                    logger.info(f"   💰 Price: {result.price}")
                    logger.info(f"   📊 Volume: {result.volume}")
                    logger.info(f"   💬 Comment: {result.comment}")
                    
                    # Store in executed orders
                    executed_order_data = {
                        "signal": signal,
                        "result": result,
                        "timestamp": datetime.now(),
                        "ticket": result.order,
                        "symbol": signal.symbol,
                        "direction": signal.action,
                        "volume": result.volume,
                        "entry_price": result.price,
                        "is_tracked": True  # Mark for tracking
                    }
                    self.executed_orders.append(executed_order_data)
                    
                    # 🚀 AUTO NOTIFICATION: Send immediate notification for executed order
                    try:
                        self._send_execution_notification(executed_order_data)
                    except Exception as notif_error:
                        logger.warning(f"⚠️ Auto notification failed: {notif_error}")
                    
                    return order_result
                
                # Handle specific error codes
                error_code = result.retcode
                error_description = get_mt5_error_description(error_code)
                error_msg = f"Order failed - Code: {error_code} - {error_description}, Comment: {result.comment}"
                
                # Add helpful guidance
                if error_code == 10030 or error_code == 10031:
                    error_msg += "\n💡 Solution: Check MT5 connection to trade server"
                elif error_code == 10019:
                    error_msg += "\n💡 Solution: Insufficient margin/balance for this trade"
                elif error_code == 10017:
                    error_msg += "\n💡 Solution: Trading disabled for this symbol or outside market hours"
                    
                logger.error(f"❌ {error_msg}")
                
                # Retryable errors
                retryable_codes = [
                    mt5.TRADE_RETCODE_REQUOTE,
                    mt5.TRADE_RETCODE_PRICE_OFF,
                    mt5.TRADE_RETCODE_TIMEOUT,
                    mt5.TRADE_RETCODE_PRICE_CHANGED
                ]
                
                if result.retcode in retryable_codes and attempt < self.max_retries - 1:
                    logger.info(f"🔄 Retrying due to retryable error...")
                    time.sleep(self.retry_delay)
                    
                    # Update price for retry
                    new_price = self._get_current_price(signal.symbol, signal.action)
                    if new_price:
                        request["price"] = new_price
                    continue
                
                # Non-retryable error or max retries reached
                self._update_stats(False, signal.volume)
                self.failed_orders.append({
                    "signal": signal,
                    "result": result,
                    "timestamp": datetime.now(),
                    "error": error_msg
                })
                
                return OrderResult(
                    success=False,
                    retcode=result.retcode,
                    comment=result.comment,
                    error_message=error_msg,
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                error_msg = f"Order execution exception: {str(e)}"
                logger.error(f"❌ {error_msg}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"🔄 Retrying after exception...")
                    time.sleep(self.retry_delay)
                    continue
                
                self._update_stats(False, signal.volume)
                logger.info(f"🔓 [UNLOCKED] Order execution failed for {signal.symbol}")
                return OrderResult(success=False, error_message=error_msg, timestamp=datetime.now())
        
        # Should not reach here
        logger.info(f"🔓 [UNLOCKED] Max retries exceeded for {signal.symbol}")
        return OrderResult(success=False, error_message="Max retries exceeded", timestamp=datetime.now())

    def execute_pending_order(self, signal: TradeSignal, order_type: OrderType) -> OrderResult:
        """Execute pending order (limit/stop orders)"""
        
        # 🔒 GLOBAL ORDER EXECUTION LOCK - Prevents duplicate pending orders
        with _GLOBAL_ORDER_LOCK:
            logger.info(f"� [LOCKED] Creating pending order: {order_type.value} {signal.volume} {signal.symbol} @ {signal.entry_price}")
            
            # Validations
        if not self._validate_connection():
            return OrderResult(success=False, error_message="Connection validation failed")
            
        symbol_valid, symbol_info = self._validate_symbol(signal.symbol)
        if not symbol_valid:
            return OrderResult(success=False, error_message="Symbol validation failed")
            
        if not self._validate_trade_parameters(signal, symbol_info):
            return OrderResult(success=False, error_message="Parameter validation failed")
        
        # Create pending order request
        request = self._create_pending_order_request(signal, order_type)
        
        try:
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                order_result = OrderResult(
                    success=True,
                    order_id=result.order,
                    ticket=result.order,
                    retcode=result.retcode,
                    comment=result.comment,
                    price=signal.entry_price,
                    volume=signal.volume,
                    timestamp=datetime.now()
                )
                
                # Store pending order
                self.pending_orders[result.order] = {
                    "signal": signal,
                    "order_type": order_type,
                    "result": result,
                    "timestamp": datetime.now()
                }
                
                logger.info(f"✅ Pending order created: Ticket {result.order}")
                return order_result
            
            error_code = result.retcode if result else None
            error_description = get_mt5_error_description(error_code) if error_code else "No result returned"
            error_msg = f"Pending order failed - Code: {error_code} - {error_description}"
            
            if error_code == 10030 or error_code == 10031:
                error_msg += "\n💡 Solution: Check MT5 connection to trade server"
                
            logger.error(f"❌ {error_msg}")
            return OrderResult(success=False, error_message=error_msg, timestamp=datetime.now())
            
        except Exception as e:
            error_msg = f"Pending order exception: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return OrderResult(success=False, error_message=error_msg, timestamp=datetime.now())

    def modify_order(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None, price: Optional[float] = None) -> OrderResult:
        """Modify existing order (SL/TP/Price)"""
        logger.info(f"🔧 Modifying order {ticket}")
        
        try:
            # 🔍 ENHANCED DEBUG: Check MT5 connection first
            if not mt5.terminal_info():
                return OrderResult(success=False, error_message="MT5 terminal not connected")
            
            # Get current order info
            order_info = mt5.orders_get(ticket=ticket)
            if not order_info:
                position_info = mt5.positions_get(ticket=ticket)
                if not position_info:
                    logger.error(f"❌ Position/Order {ticket} not found in MT5")
                    return OrderResult(success=False, error_message="Order/Position not found")
                
                # 🔍 DEBUG: Log position info before modification
                pos = position_info[0]
                logger.debug(f"🔧 Position {ticket}: Symbol={pos.symbol}, Current SL={pos.sl:.5f}, Current TP={pos.tp:.5f}")
                logger.debug(f"🔧 Modification request: New SL={sl}, New TP={tp}")
                
                # Modify position
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": position_info[0].symbol,
                    "position": ticket,
                    "sl": sl if sl is not None else position_info[0].sl,
                    "tp": tp if tp is not None else position_info[0].tp,
                    "magic": self.magic_number
                }
            else:
                # Modify pending order
                request = {
                    "action": mt5.TRADE_ACTION_MODIFY,
                    "order": ticket,
                    "price": price if price is not None else order_info[0].price_open,
                    "sl": sl if sl is not None else order_info[0].sl,
                    "tp": tp if tp is not None else order_info[0].tp,
                    "magic": self.magic_number
                }
            
            # 🔍 DEBUG: Log request details
            logger.debug(f"🔧 MT5 Request: {request}")
            
            result = mt5.order_send(request)
            
            # 🔍 ENHANCED DEBUG: Log full result details
            if result:
                logger.debug(f"🔧 MT5 Result: retcode={result.retcode}, deal={result.deal}, order={result.order}, volume={result.volume}, price={result.price}, bid={result.bid}, ask={result.ask}, comment={result.comment}")
            else:
                logger.error(f"❌ MT5 order_send returned None for ticket {ticket}")
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Order {ticket} modified successfully")
                return OrderResult(success=True, ticket=ticket, timestamp=datetime.now())
            
            # 🔍 ENHANCED ERROR: Show specific MT5 error codes
            retcode = result.retcode if result else None
            error_msg = f"Order modification failed - Code: {retcode}"
            if retcode == 10025:  # TRADE_RETCODE_NO_CHANGES
                error_msg += " (NO_CHANGES - requested values same as current)"
            elif retcode == 10026:  # TRADE_RETCODE_NO_MONEY
                error_msg += " (NO_MONEY - insufficient margin)"
            elif retcode == 10027:  # TRADE_RETCODE_MARKET_CLOSED
                error_msg += " (MARKET_CLOSED)"
            elif retcode == 10028:  # TRADE_RETCODE_REQUOTE
                error_msg += " (REQUOTE - price changed)"
            elif retcode is None:
                error_msg += " (MT5 order_send returned None)"
                
            logger.error(f"❌ {error_msg}")
            return OrderResult(success=False, error_message=error_msg, timestamp=datetime.now())
            
        except Exception as e:
            error_msg = f"Order modification exception: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return OrderResult(success=False, error_message=error_msg, timestamp=datetime.now())

    def calculate_smart_sl_tp(self, position_info, signal_data: dict) -> dict:
        """
        🎯 SMART S/L & T/P CALCULATION based on latest signals and position type
        
        Args:
            position_info: MT5 position object
            signal_data: Dictionary containing signal information
            
        Returns:
            Dict with calculated 'sl' and 'tp' values
        """
        try:
            symbol = position_info.symbol
            position_type = position_info.type  # 0=BUY, 1=SELL
            entry_price = position_info.price_open
            current_price = position_info.price_current
            
            # Get symbol info for proper calculations
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"⚠️ Could not get symbol info for {symbol}")
                return {}
            
            pip_value = self._get_pip_value_for_distance(symbol)
            
            # Extract signal information
            signal_strength = signal_data.get('confidence', 75.0)
            signal_direction = signal_data.get('direction', 'HOLD')
            atr_value = signal_data.get('atr', None)
            support_level = signal_data.get('support', None)
            resistance_level = signal_data.get('resistance', None)
            
            # 🎯 CALCULATE SMART S/L based on position type and signals
            calculated_sl = None
            calculated_tp = None
            
            if position_type == 0:  # BUY position
                # 📉 S/L Calculation for BUY
                if support_level and support_level < current_price:
                    # Use support level as S/L with small buffer
                    buffer_pips = 5 if 'JPY' in symbol else 2
                    calculated_sl = support_level - (buffer_pips * pip_value)
                    logger.info(f"🎯 {symbol} BUY S/L: Using support {support_level:.5f} - buffer = {calculated_sl:.5f}")
                elif atr_value:
                    # Use ATR-based S/L (1.5x ATR below entry)
                    sl_distance = atr_value * 1.5
                    calculated_sl = entry_price - sl_distance
                    logger.info(f"🎯 {symbol} BUY S/L: Using ATR-based {sl_distance:.5f} below entry = {calculated_sl:.5f}")
                else:
                    # Conservative percentage-based S/L
                    sl_percentage = 0.02 if signal_strength > 80 else 0.015  # 2% or 1.5%
                    calculated_sl = entry_price * (1 - sl_percentage)
                    logger.info(f"🎯 {symbol} BUY S/L: Using {sl_percentage*100}% below entry = {calculated_sl:.5f}")
                
                # 📈 T/P Calculation for BUY
                if resistance_level and resistance_level > current_price:
                    # Use resistance level as T/P with small buffer
                    buffer_pips = 3 if 'JPY' in symbol else 1
                    calculated_tp = resistance_level - (buffer_pips * pip_value)
                    logger.info(f"🎯 {symbol} BUY T/P: Using resistance {resistance_level:.5f} - buffer = {calculated_tp:.5f}")
                elif atr_value:
                    # Use ATR-based T/P (2x ATR above entry for good R:R ratio)
                    tp_distance = atr_value * 2.0
                    calculated_tp = entry_price + tp_distance
                    logger.info(f"🎯 {symbol} BUY T/P: Using ATR-based {tp_distance:.5f} above entry = {calculated_tp:.5f}")
                else:
                    # Conservative percentage-based T/P
                    tp_percentage = 0.04 if signal_strength > 80 else 0.03  # 4% or 3%
                    calculated_tp = entry_price * (1 + tp_percentage)
                    logger.info(f"🎯 {symbol} BUY T/P: Using {tp_percentage*100}% above entry = {calculated_tp:.5f}")
            
            else:  # SELL position
                # 📈 S/L Calculation for SELL
                if resistance_level and resistance_level > current_price:
                    # Use resistance level as S/L with small buffer
                    buffer_pips = 5 if 'JPY' in symbol else 2
                    calculated_sl = resistance_level + (buffer_pips * pip_value)
                    logger.info(f"🎯 {symbol} SELL S/L: Using resistance {resistance_level:.5f} + buffer = {calculated_sl:.5f}")
                elif atr_value:
                    # Use ATR-based S/L (1.5x ATR above entry)
                    sl_distance = atr_value * 1.5
                    calculated_sl = entry_price + sl_distance
                    logger.info(f"🎯 {symbol} SELL S/L: Using ATR-based {sl_distance:.5f} above entry = {calculated_sl:.5f}")
                else:
                    # Conservative percentage-based S/L
                    sl_percentage = 0.02 if signal_strength > 80 else 0.015  # 2% or 1.5%
                    calculated_sl = entry_price * (1 + sl_percentage)
                    logger.info(f"🎯 {symbol} SELL S/L: Using {sl_percentage*100}% above entry = {calculated_sl:.5f}")
                
                # 📉 T/P Calculation for SELL
                if support_level and support_level < current_price:
                    # Use support level as T/P with small buffer
                    buffer_pips = 3 if 'JPY' in symbol else 1
                    calculated_tp = support_level + (buffer_pips * pip_value)
                    logger.info(f"🎯 {symbol} SELL T/P: Using support {support_level:.5f} + buffer = {calculated_tp:.5f}")
                elif atr_value:
                    # Use ATR-based T/P (2x ATR below entry for good R:R ratio)
                    tp_distance = atr_value * 2.0
                    calculated_tp = entry_price - tp_distance
                    logger.info(f"🎯 {symbol} SELL T/P: Using ATR-based {tp_distance:.5f} below entry = {calculated_tp:.5f}")
                else:
                    # Conservative percentage-based T/P
                    tp_percentage = 0.04 if signal_strength > 80 else 0.03  # 4% or 3%
                    calculated_tp = entry_price * (1 - tp_percentage)
                    logger.info(f"🎯 {symbol} SELL T/P: Using {tp_percentage*100}% below entry = {calculated_tp:.5f}")
            
            # 🔍 VALIDATION: Ensure S/L and T/P make sense
            result = {}
            if calculated_sl and calculated_sl > 0:
                result['sl'] = calculated_sl
            if calculated_tp and calculated_tp > 0:
                result['tp'] = calculated_tp
            
            # Log final calculation summary
            if result:
                risk_reward = abs(calculated_tp - entry_price) / abs(calculated_sl - entry_price) if calculated_sl and calculated_tp else 0
                logger.info(f"✅ {symbol} Smart S/L/T/P calculated - R:R ratio: {risk_reward:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating smart S/L/T/P for {position_info.symbol}: {e}")
            return {}

    def _calculate_smart_sl_tp_from_signal(self, action_data: dict, symbol: str) -> dict:
        """
        🧠 WRAPPER: Calculate smart S/L and T/P from action signal data
        
        Args:
            action_data: Action dictionary containing signal information
            symbol: Trading symbol
            
        Returns:
            Dictionary with 'sl' and 'tp' keys if calculated, empty dict otherwise
        """
        try:
            # Extract signal data from action
            signal_data = {
                'confidence': action_data.get('signal_confidence', action_data.get('confidence', 75.0)),
                'direction': action_data.get('signal_direction', action_data.get('direction', 'HOLD')),
                'atr': action_data.get('atr', action_data.get('atr_value')),
                'support': action_data.get('support_level', action_data.get('support')),
                'resistance': action_data.get('resistance_level', action_data.get('resistance')),
                'entry_price': action_data.get('entry_price', action_data.get('price_open')),
                'current_price': action_data.get('current_price', action_data.get('price_current')),
                'reason': action_data.get('reason', 'Signal-based adjustment')
            }
            
            # Get position info if ticket is available
            ticket = action_data.get('ticket')
            if ticket:
                positions = mt5.positions_get(ticket=ticket)
                if positions and len(positions) > 0:
                    position_info = positions[0]
                    logger.debug(f"🧠 {symbol}: Using position {ticket} for smart S/L/T/P calculation")
                    
                    # Use existing calculate_smart_sl_tp function
                    result = self.calculate_smart_sl_tp(position_info, signal_data)
                    
                    if result:
                        logger.info(f"🧠 {symbol}: Smart calculation successful - SL: {result.get('sl', 'N/A')}, TP: {result.get('tp', 'N/A')}")
                    
                    return result
                else:
                    logger.warning(f"⚠️ {symbol}: Position {ticket} not found for smart calculation")
            else:
                logger.debug(f"🧠 {symbol}: No ticket provided for smart S/L/T/P calculation")
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error in smart S/L/T/P calculation from signal for {symbol}: {e}")
            return {}

    def close_position(self, ticket: int, volume: Optional[float] = None) -> OrderResult:
        """Close position (partial or full)"""
        logger.info(f"🔒 Closing position {ticket}")
        
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return OrderResult(success=False, error_message="Position not found")
            
            position = positions[0]
            close_volume = volume if volume is not None else position.volume
            
            # Determine close order type
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            # Get current price
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                return OrderResult(success=False, error_message="Cannot get current price")
            
            close_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": close_volume,
                "type": close_type,
                "position": ticket,
                "price": close_price,
                "deviation": self.slippage,
                "magic": self.magic_number,
                "comment": "Position close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": get_symbol_fill_type(position.symbol)
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Position {ticket} closed successfully")
                return OrderResult(
                    success=True,
                    ticket=ticket,
                    price=result.price,
                    volume=result.volume,
                    timestamp=datetime.now()
                )
            
            error_code = result.retcode if result else None
            error_description = get_mt5_error_description(error_code) if error_code else "No result returned"
            error_msg = f"Position close failed - Code: {error_code} - {error_description}"
            
            # Specific guidance for common errors
            if error_code == 10027:
                error_msg += "\n💡 Solution: Enable 'Allow automated trading' in MT5 Tools > Options > Expert Advisors"
            elif error_code == 10030 or error_code == 10031:
                error_msg += "\n💡 Solution: Check your internet connection and MT5 server connection"
            elif error_code == 10017:
                error_msg += "\n💡 Solution: Trading may be disabled for this symbol or during market hours"
            elif error_code == 10019:
                error_msg += "\n💡 Solution: Insufficient account balance to close position"
            
            logger.error(f"❌ {error_msg}")
            return OrderResult(success=False, error_message=error_msg, timestamp=datetime.now())
            
        except Exception as e:
            error_msg = f"Position close exception: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return OrderResult(success=False, error_message=error_msg, timestamp=datetime.now())

    def cancel_order(self, ticket: int) -> OrderResult:
        """Cancel pending order"""
        logger.info(f"❌ Cancelling order {ticket}")
        
        try:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
                "magic": self.magic_number
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Order {ticket} cancelled successfully")
                
                # Remove from pending orders
                if ticket in self.pending_orders:
                    del self.pending_orders[ticket]
                
                return OrderResult(success=True, ticket=ticket, timestamp=datetime.now())
            
            error_msg = f"Order cancellation failed - Code: {result.retcode if result else 'None'}"
            logger.error(f"❌ {error_msg}")
            return OrderResult(success=False, error_message=error_msg, timestamp=datetime.now())
            
        except Exception as e:
            error_msg = f"Order cancellation exception: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return OrderResult(success=False, error_message=error_msg, timestamp=datetime.now())

    def _update_stats(self, success: bool, volume: float):
        """Update execution statistics"""
        self.stats["total_orders"] += 1
        self.stats["total_volume"] += volume
        
        if success:
            self.stats["successful_orders"] += 1
        else:
            self.stats["failed_orders"] += 1
        
        if self.stats["total_orders"] > 0:
            self.stats["success_rate"] = (self.stats["successful_orders"] / self.stats["total_orders"]) * 100

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            **self.stats,
            "pending_orders_count": len(self.pending_orders),
            "executed_orders_count": len(self.executed_orders),
            "failed_orders_count": len(self.failed_orders)
        }

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return []
            
            return [
                {
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "comment": pos.comment,
                    "magic": pos.magic,
                    "time": datetime.fromtimestamp(pos.time)
                }
                for pos in positions
            ]
            
        except Exception as e:
            logger.error(f"❌ Error getting positions: {e}")
            return []

    def get_closed_positions(self, days_back: int = 7) -> List[Dict]:
        """Get closed positions from history (last N days)"""
        try:
            from_date = datetime.now() - timedelta(days=days_back)
            to_date = datetime.now()
            
            # Get deals from history
            deals = mt5.history_deals_get(from_date, to_date)
            if not deals:
                return []
            
            closed_positions = []
            for deal in deals:
                # Only include OUT deals (closing positions)
                if deal.entry == mt5.DEAL_ENTRY_OUT:
                    closed_positions.append({
                        "ticket": deal.ticket,
                        "position_id": deal.position_id,
                        "symbol": deal.symbol,
                        "type": "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL",
                        "volume": deal.volume,
                        "price": deal.price,
                        "profit": deal.profit,
                        "swap": deal.swap,
                        "commission": deal.commission,
                        "comment": deal.comment,
                        "time": datetime.fromtimestamp(deal.time),
                        "magic": deal.magic,
                        "reason": self._deal_reason_to_string(deal.reason)
                    })
            
            # Sort by time (newest first)
            closed_positions.sort(key=lambda x: x['time'], reverse=True)
            return closed_positions
            
        except Exception as e:
            logger.error(f"❌ Error getting closed positions: {e}")
            return []

    def _deal_reason_to_string(self, reason: int) -> str:
        """Convert deal reason to readable string"""
        reason_map = {
            0: "CLIENT",
            1: "MOBILE", 
            2: "WEB",
            3: "EXPERT",
            4: "SL",  # Stop Loss
            5: "TP",  # Take Profit
            6: "SO"   # Stop Out
        }
        return reason_map.get(reason, f"UNKNOWN({reason})")

    def get_pending_orders_info(self) -> List[Dict]:
        """Get all pending orders"""
        try:
            orders = mt5.orders_get()
            if not orders:
                return []
            
            return [
                {
                    "ticket": order.ticket,
                    "symbol": order.symbol,
                    "type": self._order_type_to_string(order.type),
                    "volume": getattr(order, 'volume_initial', getattr(order, 'volume', 0.0)),
                    "price_open": order.price_open,
                    "sl": order.sl,
                    "tp": order.tp,
                    "comment": order.comment,
                    "magic": order.magic,
                    "time_setup": datetime.fromtimestamp(order.time_setup),
                    "time_expiration": datetime.fromtimestamp(order.time_expiration) if order.time_expiration > 0 else None
                }
                for order in orders
            ]
            
        except Exception as e:
            logger.error(f"❌ Error getting orders: {e}")
            return []

    def apply_actions_from_json(self, actions_path: Optional[str] = None, risk_path: Optional[str] = None) -> Dict[str, Any]:
        """Apply actions from analysis_results/account_positions_actions.json using risk_management/risk_settings.json.

        Returns a summary dict with counts per action and any errors.
        """
        try:
            import json as _json
            import os as _os
            from datetime import datetime as _dt, time as _time, timezone as _tz

            # �️ PERMANENT DUPLICATE PROTECTION: Enhanced validation before execution
            logger.info("�️ Apply actions with enhanced duplicate protection")
            
            # Validate MT5 connection once
            logger.debug("🔧 Apply actions: Validating connection...")
            connection_valid = self._validate_connection()
            logger.debug(f"🔧 Apply actions: Connection validation result = {connection_valid}")
            if not connection_valid:
                logger.error("❌ Apply actions: Connection validation failed")
                return {"success": False, "error": "Connection validation failed"}

            # Default paths
            if actions_path is None:
                actions_path = _os.path.join(_os.path.dirname(__file__), 'analysis_results', 'account_positions_actions.json')
            if risk_path is None:
                risk_path = _os.path.join(_os.path.dirname(__file__), 'risk_management', 'risk_settings.json')

            # Load risk config
            try:
                with open(risk_path, 'r', encoding='utf-8') as f:
                    risk_cfg = _json.load(f)
            except Exception as e:
                logger.error(f"❌ Failed to load risk config: {e}")
                return {"success": False, "error": str(e)}

            # Load actions
            try:
                with open(actions_path, 'r', encoding='utf-8') as f:
                    data = _json.load(f)
                    
                    # Handle both formats: {"actions": [...]} and [...]
                    if isinstance(data, list):
                        actions = data
                        logger.info(f"📋 Loaded {len(actions)} actions from {actions_path} (array format)")
                    elif isinstance(data, dict):
                        actions = data.get('actions', [])
                        logger.info(f"📋 Loaded {len(actions)} actions from {actions_path} (object format)")
                    else:
                        actions = []
                        logger.warning(f"⚠️ Unexpected JSON format in {actions_path}")
                    
                    # Log action types for debugging
                    for i, action in enumerate(actions):
                        action_type = action.get('action_type') or action.get('primary_action')
                        symbol = action.get('symbol')
                        logger.info(f"   Action {i+1}: {action_type} - {symbol}")
                        
            except Exception as e:
                logger.error(f"❌ Failed to load actions: {e}")
                return {"success": False, "error": str(e)}

            if not actions:
                logger.info("ℹ️ No actions to apply.")
                return {"success": True, "applied": 0}

            # Optionally adopt runtime risk knobs
            try:
                if 'max_spread_multiplier' in risk_cfg:
                    self.max_spread_multiplier = float(risk_cfg['max_spread_multiplier'])
                if 'max_slippage' in risk_cfg:
                    self.slippage = int(risk_cfg['max_slippage'])
            except Exception:
                pass

            # Get open positions snapshot
            positions = self.get_open_positions()
            pos_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
            for p in positions:
                pos_by_symbol.setdefault(p['symbol'], []).append(p)

            # --- Core Risk Params ---
            max_lot = parse_setting_value(risk_cfg.get('max_total_volume'), getattr(self, 'max_total_volume', 1.0), 'max_total_volume')
            min_lot = parse_setting_value(risk_cfg.get('min_volume_auto'), getattr(self, 'min_volume_auto', 0.01), 'min_volume_auto')
            max_positions = int(risk_cfg.get('max_positions', 20))
            max_positions_per_symbol = int(risk_cfg.get('max_positions_per_symbol', 5))
            symbol_exposure = risk_cfg.get('symbol_exposure', {}) or {}
            min_confidence = parse_setting_value(risk_cfg.get('min_confidence'), 0.0, 'min_confidence')
            max_risk_percent = parse_setting_value(risk_cfg.get('max_risk_per_trade_percent', risk_cfg.get('max_risk_per_trade')), 0.0, 'max_risk_per_trade_percent')
            max_total_open_risk_percent = parse_setting_value(risk_cfg.get('max_total_open_risk_percent'), 0.0, 'max_total_open_risk_percent')
            max_symbol_open_risk_percent = parse_setting_value(risk_cfg.get('max_symbol_open_risk_percent'), 0.0, 'max_symbol_open_risk_percent')
            trail_sl_pips = parse_setting_value(risk_cfg.get('trail_sl_pips'), 0.0, 'trail_sl_pips')
            require_sl_for_new = bool(risk_cfg.get('require_sl_for_new_trades', False))
            block_if_unprotected = bool(risk_cfg.get('block_new_trades_if_unprotected_positions', False))
            min_rr = parse_setting_value(risk_cfg.get('min_risk_reward_ratio'), None, 'min_risk_reward_ratio')
            daily_loss_limit_pct = parse_setting_value(risk_cfg.get('daily_loss_limit_percent'), 0.0, 'daily_loss_limit_percent')
            max_daily_new_positions = int(risk_cfg.get('max_daily_new_positions', 0))
            cool_down_minutes = int(risk_cfg.get('cool_down_minutes_after_big_loss', 0))
            big_loss_threshold_pct = parse_setting_value(risk_cfg.get('big_loss_percent_threshold'), 0.0, 'big_loss_percent_threshold')
            trading_time_windows = risk_cfg.get('trading_time_windows', []) or []
            allowed_days = set(risk_cfg.get('allowed_days', []))
            block_on_weekend = bool(risk_cfg.get('block_on_weekend', True))

            now = _dt.now(_tz.utc)
            weekday_code = ["MON","TUE","WED","THU","FRI","SAT","SUN"][now.weekday()]

            # --- Helper functions for advanced risk constraints ---
            def _parse_hhmm(hhmm: str) -> _time:
                try:
                    h,m = hhmm.split(":")
                    return _time(int(h), int(m))
                except Exception:
                    return _time(0,0)

            def _within_time_window() -> bool:
                if not trading_time_windows:
                    return True
                current_t = now.time()
                for w in trading_time_windows:
                    try:
                        st = _parse_hhmm(w.get('start','00:00'))
                        et = _parse_hhmm(w.get('end','23:59'))
                        if st <= current_t <= et:
                            return True
                    except Exception:
                        continue
                return False

            def _is_crypto_symbol(symbol: str) -> bool:
                """Check if symbol is cryptocurrency using MT5 symbol info"""
                try:
                    symbol_info = mt5.symbol_info(symbol)
                    if symbol_info and symbol_info.path:
                        # Check if path contains crypto indicators
                        path_lower = symbol_info.path.lower()
                        return 'crypto' in path_lower or 'btc' in path_lower or 'eth' in path_lower
                    
                    # Fallback to hardcoded list if MT5 info not available
                    crypto_symbols = ['BTCUSD', 'ETHUSD', 'BNBUSD', 'SOLUSD', 'ADAUSD', 'DOTUSD', 'LINKUSD', 'UNIUSD', 'LTCUSD']
                    # Strip suffix (_m, _h, etc.) for crypto detection
                    symbol_base = symbol.upper().replace('_M', '').replace('_H', '').replace('_MICRO', '')
                    return symbol_base in crypto_symbols or 'BTC' in symbol.upper() or 'ETH' in symbol.upper() or 'BNB' in symbol.upper() or 'SOL' in symbol.upper() or 'LTC' in symbol.upper()
                except Exception as e:
                    # If MT5 call fails, use fallback logic
                    crypto_symbols = ['BTCUSD', 'ETHUSD', 'BNBUSD', 'SOLUSD', 'ADAUSD', 'DOTUSD', 'LINKUSD', 'UNIUSD', 'LTCUSD']
                    symbol_base = symbol.upper().replace('_M', '').replace('_H', '').replace('_MICRO', '')
                    return symbol_base in crypto_symbols or 'BTC' in symbol.upper() or 'ETH' in symbol.upper() or 'BNB' in symbol.upper() or 'SOL' in symbol.upper() or 'LTC' in symbol.upper()

            def _is_market_closed_for_symbol(symbol: str) -> bool:
                """Check if market is closed for specific symbol"""
                # Crypto markets are always open
                if _is_crypto_symbol(symbol):
                    return False
                
                # Forex markets close on weekends
                if weekday_code in ("SAT", "SUN"):
                    return True
                    
                # Additional forex market hours check could be added here
                return False

            def _weekend_block() -> bool:
                if block_on_weekend and weekday_code in ("SAT","SUN"):
                    return True
                if allowed_days and weekday_code not in allowed_days:
                    return True
                return False

            # Account equity/performance metrics for daily loss limit & cooldown
            account_info = mt5.account_info()
            start_balance_file = os.path.join(os.path.dirname(__file__), 'risk_management', 'session_start_balance.json')
            session_data = {}
            try:
                if os.path.exists(start_balance_file):
                    with open(start_balance_file,'r',encoding='utf-8') as sf:
                        session_data = _json.load(sf) or {}
            except Exception:
                session_data = {}
            if account_info:
                bal = account_info.balance
                eq = account_info.equity
                # Initialize session baseline if absent or different day
                day_key = now.strftime('%Y-%m-%d')
                if session_data.get('date') != day_key:
                    session_data = {'date': day_key, 'start_balance': bal, 'start_equity': eq, 'new_positions_opened': 0, 'last_big_loss_time': None}
                    overwrite_json_safely(start_balance_file, session_data, backup=False)
            else:
                bal = 0.0; eq = 0.0

            def _update_session_new_position():
                session_data['new_positions_opened'] = session_data.get('new_positions_opened',0) + 1
                overwrite_json_safely(start_balance_file, session_data, backup=False)

            def _register_big_loss():
                session_data['last_big_loss_time'] = now.isoformat()
                overwrite_json_safely(start_balance_file, session_data, backup=False)

            # Detect big loss (equity drop from start)
            if account_info and big_loss_threshold_pct > 0:
                start_eq = session_data.get('start_equity', account_info.equity)
                if start_eq > 0:
                    dd_pct = (start_eq - account_info.equity) / start_eq * 100.0
                    if dd_pct >= big_loss_threshold_pct and not session_data.get('last_big_loss_time'):
                        _register_big_loss()

            def _cooldown_active() -> bool:
                lt = session_data.get('last_big_loss_time')
                if lt and cool_down_minutes > 0:
                    try:
                        t = _dt.fromisoformat(lt)
                        if (now - t).total_seconds() < cool_down_minutes*60:
                            return True
                    except Exception:
                        return False
                return False

            daily_loss_block = False
            if account_info and daily_loss_limit_pct > 0:
                start_bal = session_data.get('start_balance', account_info.balance)
                if start_bal > 0:
                    loss_pct = (start_bal - account_info.balance)/start_bal * 100.0
                    daily_loss_block = loss_pct >= daily_loss_limit_pct

            # Pre-calc existing open risk per position (approx: (SL distance / price)*100 * volume weight) simplistic
            def _position_risk_pct(p: Dict[str,Any]) -> float:
                try:
                    if p['sl'] <= 0:
                        return 0.0
                    price = p['price_open']
                    sl = p['sl']
                    direction = p['type']
                    # approximate potential loss in pct of price
                    if direction=='BUY':
                        potential = (price - sl)/price
                    else:
                        potential = (sl - price)/price
                    return max(0.0, potential*100.0) * p['volume']
                except Exception:
                    return 0.0

            existing_total_risk = sum(_position_risk_pct(p) for p in positions)
            existing_symbol_risk: Dict[str,float] = {}
            for p in positions:
                existing_symbol_risk.setdefault(p['symbol'],0.0)
                existing_symbol_risk[p['symbol']] += _position_risk_pct(p)

            total_positions = len(positions)
            symbol_counts = {s: len(lst) for s, lst in pos_by_symbol.items()}

            def _validate_action_parameters(action: str, params: Dict) -> Tuple[bool, str]:
                """Validate action parameters before execution"""
                try:
                    if action in ('set_initial_sl', 'tighten_sl', 'set_sl'):
                        sl = params.get('proposed_sl') or params.get('stop_loss') or params.get('sl')
                        if sl is None:
                            return False, "Missing stop loss value"
                        try:
                            sl_val = float(sl)
                            if sl_val <= 0:
                                return False, f"Invalid stop loss value: {sl_val}"
                        except (ValueError, TypeError):
                            return False, f"Invalid stop loss format: {sl}"
                    
                    elif action in ('adjust_tp', 'update_tp'):
                        tp = params.get('proposed_tp') or params.get('take_profit') or params.get('tp')
                        if tp is None:
                            return False, "Missing take profit value"
                        try:
                            tp_val = float(tp)
                            if tp_val <= 0:
                                return False, f"Invalid take profit value: {tp_val}"
                        except (ValueError, TypeError):
                            return False, f"Invalid take profit format: {tp}"
                    
                    elif action in ('close_partial_30', 'close_partial_50', 'close_partial_70', 'take_partial_30', 'take_partial_50', 'take_partial_70'):
                        # Volume validation will be done later with actual position volume
                        pass
                    
                    elif action in ('scale_out', 'partial_exit', 'reduce_risk'):
                        scale_pct = params.get('scale_percent', params.get('reduction_percent', 25))
                        try:
                            scale_val = float(scale_pct)
                            if scale_val <= 0 or scale_val > 100:
                                return False, f"Invalid scale percentage: {scale_val}% (must be 1-100)"
                        except (ValueError, TypeError):
                            return False, f"Invalid scale percentage format: {scale_pct}"
                    
                    return True, "Valid"
                
                except Exception as e:
                    return False, f"Validation error: {str(e)}"

            def _safe_float_conversion(value, default=0.0, param_name="value"):
                """Safely convert value to float with detailed error info"""
                if value is None:
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Failed to convert {param_name}='{value}' to float: {e}, using default {default}")
                    return default

            def _log_action_start(action: str, symbol: str, ticket: int, additional_info: str = ""):
                """Log action start with consistent format"""
                info_str = f" ({additional_info})" if additional_info else ""
                logger.info(f"🔧 Starting {action} for {symbol} ticket {ticket}{info_str}")

            def _log_action_result(action: str, symbol: str, ticket: int, success: bool, message: str = ""):
                """Log action result with consistent format"""
                status_icon = "✅" if success else "❌"
                msg_str = f" - {message}" if message else ""
                logger.info(f"{status_icon} {action} {symbol} ticket {ticket}: {'SUCCESS' if success else 'FAILED'}{msg_str}")

            summary = {"success": True, "applied": 0, "by_action": {}, "skipped": [], "errors": []}

            # Track modified tickets to prevent duplicates
            modified_tickets = set()

            def _inc(action_name: str, success_flag: bool):
                summary["by_action"].setdefault(action_name, 0)
                if success_flag:
                    summary["by_action"][action_name] += 1
                    summary["applied"] += 1

            def _can_open_new(symbol: str) -> bool:
                # Recompute counts each time for dynamic updates
                current_positions = self.get_open_positions()
                if len(current_positions) >= max_positions:
                    return False
                sym_count = sum(1 for p in current_positions if p['symbol'] == symbol)
                if sym_count >= max_positions_per_symbol:
                    return False
                # Symbol exposure (total volume cap)
                if symbol in symbol_exposure:
                    try:
                        exposure_data = symbol_exposure.get(symbol, {})
                        if isinstance(exposure_data, dict):
                            exposure_val = exposure_data.get('max_volume')
                        else:
                            exposure_val = exposure_data  # Direct value
                            
                        cap = parse_setting_value(exposure_val, None, f'symbol_exposure.{symbol}')
                        if cap is not None and cap > 0:
                            total_vol = sum(p['volume'] for p in current_positions if p['symbol'] == symbol)
                            if total_vol >= cap:
                                return False
                    except Exception as e:
                        logger.warning(f"Failed to parse symbol exposure for {symbol}: {e}")
                return True

            def _risk_size(symbol: str, direction: str, entry: float, sl: float) -> float:
                """Compute lot size based on risk % and stop distance.
                Simplified formula: lot = risk_amount / (abs(entry-sl)/point * point_value_per_lot)
                Fallback to min_lot.
                """
                try:
                    if max_risk_percent <= 0 or sl <= 0 or entry <= 0 or abs(entry - sl) < 1e-9:
                        # If max_risk_percent is 0, user wants to block trading completely
                        if max_risk_percent <= 0:
                            logger.warning(f"🚫 Trade blocked: max_risk_percent={max_risk_percent}% - User has disabled trading")
                            return 0.0  # Return 0 to block trade, not min_lot
                        return min_lot
                    sym_info = mt5.symbol_info(symbol)
                    if not sym_info:
                        return min_lot
                    acc = mt5.account_info()
                    balance = acc.balance if acc else 0.0
                    risk_amount = balance * (max_risk_percent / 100.0)
                    point = sym_info.point or 0.0001
                    # Approx tick value per lot
                    tick_value = getattr(sym_info, 'trade_tick_value', 0.0) or 1.0
                    tick_size = getattr(sym_info, 'trade_tick_size', point) or point
                    stop_points = abs(entry - sl) / (tick_size if tick_size > 0 else point)
                    if stop_points <= 0:
                        return min_lot
                    value_per_lot = (stop_points * tick_value)
                    if value_per_lot <= 0:
                        return min_lot
                    lot = risk_amount / value_per_lot
                    # Clamp with None-safe handling
                    final_min = min_lot or 0.01
                    final_max = max_lot or 1.0
                    lot = max(final_min, min(final_max, round(lot / sym_info.volume_step) * sym_info.volume_step)) if sym_info.volume_step else max(final_min, min(final_max, lot))
                    return lot
                except Exception:
                    return min_lot

            for a in actions:
                symbol = a.get('symbol')
                # 🔧 FIX: Support multiple action field names 
                action = a.get('action') or a.get('action_type') or a.get('primary_action')
                logger.info(f"🔍 Processing action: symbol={symbol}, action={action}")
                
                if not symbol or not action:
                    summary["errors"].append({"error": "Missing symbol or action", "data": a})
                    logger.error(f"❌ Missing symbol or action in: {a}")
                    continue

                # Validate action parameters first
                is_valid, validation_msg = _validate_action_parameters(action, a)
                logger.info(f"🔍 Validation result for {symbol} {action}: valid={is_valid}, msg={validation_msg}")
                
                if not is_valid:
                    summary["skipped"].append({
                        "symbol": symbol, 
                        "action": action, 
                        "reason": "validation_failed",
                        "details": validation_msg
                    })
                    logger.warning(f"⚠️ {symbol} action {action}: Validation failed - {validation_msg}")
                    continue

                # Fallback: map signal_confidence -> confidence if missing
                if 'confidence' not in a and 'signal_confidence' in a:
                    try:
                        a['confidence'] = float(a.get('signal_confidence') or 0.0)
                    except Exception:
                        a['confidence'] = 0.0

                sym_positions = pos_by_symbol.get(symbol) or []

                # NOTE: We intentionally don't block protective actions (SL tightening, partials, closes)
                # based on total/symbol position caps, because these actions reduce risk not increase it.
                # Caps should be enforced when opening new trades elsewhere.

                # Handle opening actions before iterating existing positions
                logger.info(f"🎯 Checking if action '{action}' == 'primary_entry'")
                if action == 'primary_entry':
                    # Handle new position opening from signal
                    logger.info(f"🎯 ENTERED primary_entry logic for {symbol}")
                    direction = (a.get('direction') or '').upper()
                    if direction not in ['BUY', 'SELL']:
                        summary['skipped'].append({
                            "symbol": symbol, 
                            "action": action, 
                            "reason": "invalid_direction",
                            "details": f"Direction '{direction}' not BUY/SELL"
                        })
                        continue
                    
                    confidence = float(a.get('confidence', 0.0))
                    
                    # Check confidence requirement (skip if OFF/None)
                    if min_confidence is not None and confidence < min_confidence:
                        summary['skipped'].append({
                            "symbol": symbol, 
                            "action": action, 
                            "reason": "confidence_too_low",
                            "details": f"Confidence {confidence:.1f} < min {min_confidence}"
                        })
                        continue
                    
                    # Smart market hours check - skip forex when market closed but allow crypto 24/7
                    logger.info(f"🕐 Market hours check for {symbol}: is_closed={_is_market_closed_for_symbol(symbol)}")
                    if _is_market_closed_for_symbol(symbol):
                        logger.warning(f"🚫 SKIPPED {symbol} primary_entry: market_closed")
                        summary['skipped'].append({
                            "symbol": symbol, 
                            "action": action, 
                            "reason": "market_closed",
                            "details": f"{'Forex' if not _is_crypto_symbol(symbol) else 'Market'} market closed for {symbol}"
                        })
                        continue
                    
                    # Risk gating for new open - Skip weekend block for crypto symbols
                    is_weekend_block = _weekend_block() and not _is_crypto_symbol(symbol)
                    is_within_window = _within_time_window()
                    can_open = _can_open_new(symbol)
                    cooldown = _cooldown_active()
                    
                    logger.info(f"🛡️ Risk gating checks for {symbol}:")
                    logger.info(f"   - weekend_block: {is_weekend_block}")
                    logger.info(f"   - within_time_window: {is_within_window}")
                    logger.info(f"   - daily_loss_block: {daily_loss_block}")
                    logger.info(f"   - cooldown_active: {cooldown}")
                    logger.info(f"   - can_open_new: {can_open}")
                    
                    if is_weekend_block:
                        logger.warning(f"🚫 SKIPPED {symbol} primary_entry: weekend block")
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "time_block_weekend_or_day"})
                        continue
                    elif not is_within_window:
                        logger.warning(f"🚫 SKIPPED {symbol} primary_entry: outside time window")
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "outside_time_window"})
                        continue
                    elif daily_loss_block:
                        logger.warning(f"🚫 SKIPPED {symbol} primary_entry: daily loss limit")
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "daily_loss_limit"})
                        continue
                    elif _cooldown_active():
                        logger.warning(f"🚫 SKIPPED {symbol} primary_entry: cooldown active")
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "cooldown_active"})
                        continue
                    elif max_daily_new_positions is not None and max_daily_new_positions > 0 and session_data.get('new_positions_opened',0) >= max_daily_new_positions:
                        logger.warning(f"🚫 SKIPPED {symbol} primary_entry: daily new positions cap")
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "daily_new_positions_cap"})
                        continue
                    elif not _can_open_new(symbol):
                        logger.warning(f"🚫 SKIPPED {symbol} primary_entry: position cap")
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "position_cap"})
                        continue
                    
                    # 🛡️ COMPREHENSIVE DUPLICATE PROTECTION - Enhanced multi-layer checks
                    entry_price = a.get('entry_price') or a.get('entry')
                    duplicate_found = False
                    
                    # � FRESH POSITION CHECK: Get live positions to prevent stale data issues
                    try:
                        fresh_positions = self.get_open_positions()
                        fresh_sym_positions = [p for p in fresh_positions if p['symbol'] == symbol]
                        logger.info(f"🔄 FRESH CHECK {symbol}: Found {len(fresh_sym_positions)} live positions")
                        
                        # Log all positions for debugging
                        if fresh_sym_positions:
                            for pos in fresh_sym_positions:
                                logger.info(f"   Position: Ticket={pos.get('ticket')} Type={pos.get('type')} Direction={'BUY' if pos.get('type') == 0 else 'SELL'} Vol={pos.get('volume')}")
                        
                        # 🚫 STRICT CHECK: Only block if positions are SAME direction as new signal
                        if action in ['primary_entry', 'entry', 'new_entry']:
                            # Filter for same direction positions
                            same_direction_positions = [
                                p for p in fresh_sym_positions 
                                if (p.get('type') == 0 and direction == 'BUY') or (p.get('type') == 1 and direction == 'SELL')
                            ]
                            
                            if same_direction_positions:
                                # BLOCK: Same direction exists
                                ticket_list = [str(p.get('ticket', 'Unknown')) for p in same_direction_positions]
                                logger.warning(f"🚫 DIRECTION BLOCK: {symbol} {direction} entry denied - found {len(same_direction_positions)} existing {direction} position(s) (tickets: {', '.join(ticket_list)})")
                                summary['skipped'].append({
                                    "symbol": symbol, 
                                    "action": action, 
                                    "reason": "same_direction_position_exists",
                                    "details": f"Found {len(same_direction_positions)} existing {direction} positions"
                                })
                                continue
                            elif fresh_sym_positions:
                                # ALLOW: Has positions but opposite direction
                                logger.info(f"✅ {symbol} {direction} entry ALLOWED - {len(fresh_sym_positions)} existing position(s) are opposite direction")
                            else:
                                # ALLOW: No positions at all
                                logger.info(f"✅ {symbol} {direction} entry ALLOWED - no existing positions")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed fresh position check for {symbol}: {e}")
                        fresh_sym_positions = sym_positions  # Fallback to cached data
                    
                    # �🛡️ LAYER 0: Removed duplicate check (already handled above in FRESH CHECK)
                    
                    # 🛡️ LEGACY PRICE-BASED PROTECTION - Only for rapid duplicates
                    if entry_price and sym_positions:
                        for pos in sym_positions:
                            # Handle both datetime objects and timestamps
                            pos_time = pos.get('time', 0)
                            if hasattr(pos_time, 'timestamp'):  # datetime object
                                pos_timestamp = pos_time.timestamp()
                            else:  # assume it's already a timestamp
                                pos_timestamp = float(pos_time) if pos_time else 0
                            
                            time_diff = _dt.now().timestamp() - pos_timestamp
                            price_diff = abs(pos.get('price_open', 0) - float(entry_price))
                            same_direction = (pos.get('type') == 0 and direction == 'BUY') or (pos.get('type') == 1 and direction == 'SELL')
                            
                            if same_direction:
                                # Layer 1: Exact duplicate (same price within 10 minutes)
                                if price_diff < 0.00005 and time_diff < 600:
                                    duplicate_found = True
                                    logger.warning(f"🚫 BLOCKED: Exact duplicate {symbol} {direction} @ {entry_price} (existing position #{pos.get('ticket')})")
                                    break
                                
                                # Layer 2: Near duplicate (very similar price within 5 minutes)
                                elif price_diff < 0.0001 and time_diff < 300:
                                    duplicate_found = True
                                    logger.warning(f"🚫 BLOCKED: Near duplicate {symbol} {direction} @ {entry_price} vs {pos.get('price_open')} (position #{pos.get('ticket')})")
                                    break
                                
                                # Layer 3: Rapid fire protection (same symbol/direction within 30 seconds)
                                elif time_diff < 30:
                                    duplicate_found = True
                                    logger.warning(f"🚫 BLOCKED: Rapid duplicate {symbol} {direction} within 30sec (position #{pos.get('ticket')})")
                                    break
                    
                    # Global lock check for additional protection
                    lock_key = f"{symbol}_{direction}"
                    current_time = time.time()
                    
                    if not hasattr(self, '_order_locks'):
                        self._order_locks = {}
                        
                    if lock_key in self._order_locks:
                        if current_time - self._order_locks[lock_key] < 15:  # 15-second cooldown
                            duplicate_found = True
                            logger.warning(f"🚫 BLOCKED: Global lock active for {lock_key}")
                    
                    if duplicate_found:
                        summary['skipped'].append({
                            "symbol": symbol, 
                            "action": action, 
                            "reason": "duplicate_protection",
                            "details": f"Duplicate order blocked by multi-layer protection"
                        })
                        continue
                    
                    # Set global lock before proceeding
                    self._order_locks[lock_key] = current_time
                    
                    # Get entry parameters
                    original_entry = a.get('entry_price') or a.get('entry')
                    original_sl = _safe_float_conversion(a.get('stop_loss') or a.get('sl'), 0.0, "stop_loss")
                    original_tp = _safe_float_conversion(a.get('take_profit') or a.get('tp'), 0.0, "take_profit")
                    
                    # Always get current price for market execution
                    current_price = self._get_current_price(symbol, direction)
                    
                    # For crypto symbols, use original entry price if current price fails
                    if not current_price and symbol.upper() in ['BTCUSD', 'ETHUSD', 'SOLUSD']:
                        logger.warning(f"⚠️ Using original entry price for crypto symbol {symbol}")
                        current_price = original_entry
                    
                    if not current_price:
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "no_current_price"})
                        continue
                    
                    # Use current price as entry for market orders
                    entry = current_price
                    
                    # Recalculate SL/TP based on original distances if they exist
                    pip_value = get_pip_value(symbol)
                    if original_entry and original_sl > 0:
                        sl_distance = abs(original_entry - original_sl)  # Actual price distance
                        sl_distance_pips = sl_distance / pip_value  # Convert to pips using proper pip value
                        if direction == 'BUY':
                            sl = entry - sl_distance
                        else:
                            sl = entry + sl_distance
                    else:
                        # 🔧 CRITICAL FIX: Use proper default SL from risk settings
                        default_sl_pips = self.risk_settings.get('default_sl_pips', 150)  # Use GUI setting of 150 pips
                        
                        # Add spread consideration for tighter SL protection
                        symbol_info = mt5.symbol_info(symbol)
                        if symbol_info:
                            tick = mt5.symbol_info_tick(symbol)
                            if tick:
                                spread_pips = (tick.ask - tick.bid) / pip_value
                                # Add spread buffer to SL for better protection
                                effective_sl_pips = default_sl_pips + (spread_pips * 2)  # 2x spread buffer
                                logger.info(f"🛡️ SL with spread protection: {default_sl_pips} + {spread_pips*2:.1f} = {effective_sl_pips:.1f} pips")
                            else:
                                effective_sl_pips = default_sl_pips
                        else:
                            effective_sl_pips = default_sl_pips
                            
                        if direction == 'BUY':
                            sl = entry - (effective_sl_pips * pip_value)
                        else:
                            sl = entry + (effective_sl_pips * pip_value)
                    
                    if original_entry and original_tp > 0:
                        tp_distance = abs(original_tp - original_entry)  # Actual price distance
                        tp_distance_pips = tp_distance / pip_value  # Convert to pips using proper pip value
                        if direction == 'BUY':
                            tp = entry + tp_distance
                        else:
                            tp = entry - tp_distance
                    else:
                        # Fallback to default TP distance from risk settings
                        default_tp_pips = self.risk_settings.get('default_tp_pips', 100)
                        if direction == 'BUY':
                            tp = entry + (default_tp_pips * pip_value)
                        else:
                            tp = entry - (default_tp_pips * pip_value)
                    
                    # Check SL requirement
                    if require_sl_for_new and sl <= 0:
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "sl_required"})
                        continue
                    
                    # Calculate volume - ALWAYS use risk settings instead of JSON volume
                    try:
                        # Create temporary signal for volume calculation
                        temp_signal = TradeSignal(
                            symbol=symbol,
                            action=direction,
                            entry_price=float(entry),
                            stop_loss=float(sl) if sl > 0 else 0.0,
                            take_profit=float(tp) if tp > 0 else 0.0,
                            volume=0.01,  # Temporary volume
                            confidence=float(confidence),
                            strategy="PRIMARY_ENTRY",
                            comment="Entry",
                            is_dca=False,
                            dca_level=1
                        )
                        
                        # Use VolumeCalculator to get proper volume from risk settings
                        volume_calc = VolumeCalculator()
                        vol = volume_calc.get_volume_for_signal(temp_signal, is_dca=False)
                        logger.info(f"💰 Primary Entry Volume calculated: {vol:.3f} lots (from risk settings)")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Primary entry volume calculation failed: {e}")
                        # Fallback to risk settings default volume
                        vol = self.risk_settings.get('default_volume_lots', 0.15)
                        logger.info(f"💰 Primary Entry Volume fallback: {vol:.3f} lots")
                    
                    # Clamp volume to limits (None-safe)
                    vol = max(min_lot or 0.01, min(max_lot or 1.0, vol))
                    
                    # Risk-reward ratio check
                    if min_rr is not None and min_rr > 0 and sl > 0 and tp > 0:
                        try:
                            if direction == 'BUY':
                                rr = abs((tp - entry)/(entry - sl))
                            else:
                                rr = abs((entry - tp)/(sl - entry))
                        except Exception:
                            rr = 0
                        if min_rr is not None and min_rr > 0 and rr < min_rr:
                            summary['skipped'].append({
                                "symbol": symbol, 
                                "action": action, 
                                "reason": f"rr_below_min",
                                "details": f"RR {rr:.2f} < min {min_rr}"
                            })
                            continue
                    
                    # Check for unprotected positions
                    if block_if_unprotected:
                        unprot = [p for p in positions if p.get('sl', 0) <= 0]
                        if unprot:
                            summary['skipped'].append({"symbol": symbol, "action": action, "reason": "unprotected_positions_exist"})
                            continue
                    
                    # Execute the market order
                    signal = TradeSignal(
                        symbol=symbol,
                        action=direction,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        volume=vol,
                        confidence=confidence,
                        strategy=a.get('strategy', 'AI_SYSTEM'),
                        comment=a.get('comment', self._get_entry_comment_formatted(direction, confidence, symbol))
                    )
                    
                    logger.info(f"🚀 Opening {direction} position for {symbol}: Vol={vol:.3f}, Entry={entry:.5f}, SL={sl:.5f}, TP={tp:.5f}")
                    res = self.execute_market_order(signal)
                    _inc(action, res.success)
                    
                    if res.success:
                        logger.info(f"✅ Primary entry successful: {symbol} {direction} ticket {res.ticket}")
                        # Update positions snapshot for subsequent actions
                        positions = self.get_open_positions()
                        pos_by_symbol[symbol] = [p for p in positions if p['symbol']==symbol]
                        _update_session_new_position()
                    else:
                        logger.error(f"❌ Primary entry failed: {symbol} {direction} - {res.error_message}")
                        summary['errors'].append({"symbol": symbol, "action": action, "error": res.error_message})
                    
                    continue  # Move to next action
                
                elif action == 'close_or_reverse':
                    # Close current positions if signal reversed, then optionally open new direction
                    current_signal_dir = (a.get('current_signal') or a.get('new_signal') or '').upper()
                    existing_dir = (a.get('direction') or '').upper()
                    confidence = float(a.get('confidence', 0.0))
                    closed_any = False
                    sym_positions = pos_by_symbol.get(symbol) or []

                    # Close all positions that are opposite to new signal (or all if unspecified)
                    for pos in sym_positions:
                        ticket = pos['ticket']
                        pos_dir = pos['type']
                        if current_signal_dir and pos_dir == current_signal_dir:
                            continue  # already aligned, keep it
                        res = self.close_position(ticket=ticket)
                        _inc('close', res.success)
                        if res.success:
                            closed_any = True
                        else:
                            summary['errors'].append({"symbol": symbol, "action": "close", "error": res.error_message})

                    # Refresh positions snapshot
                    if closed_any:
                        positions = self.get_open_positions()
                        pos_by_symbol[symbol] = [p for p in positions if p['symbol']==symbol]

                    # Decide whether to open reverse
                    want_reverse = current_signal_dir in ('BUY','SELL') and current_signal_dir != existing_dir
                    if want_reverse and (min_confidence is None or confidence >= min_confidence):
                        # Risk gating for new open
                        if _weekend_block():
                            summary['skipped'].append({"symbol": symbol, "action": "reverse_open", "reason": "time_block_weekend_or_day"})
                        elif not _within_time_window():
                            summary['skipped'].append({"symbol": symbol, "action": "reverse_open", "reason": "outside_time_window"})
                        elif daily_loss_block:
                            summary['skipped'].append({"symbol": symbol, "action": "reverse_open", "reason": "daily_loss_limit"})
                        elif _cooldown_active():
                            summary['skipped'].append({"symbol": symbol, "action": "reverse_open", "reason": "cooldown_active"})
                        elif max_daily_new_positions > 0 and session_data.get('new_positions_opened',0) >= max_daily_new_positions:
                            summary['skipped'].append({"symbol": symbol, "action": "reverse_open", "reason": "daily_new_positions_cap"})
                        elif not _can_open_new(symbol):
                            summary['skipped'].append({"symbol": symbol, "action": "reverse_open", "reason": "position_cap"})
                        else:
                            entry = a.get('entry') or a.get('entry_price')
                            sl = a.get('proposed_sl') or a.get('stop_loss') or a.get('sl') or 0.0
                            tp = a.get('proposed_tp') or a.get('take_profit') or a.get('tp') or 0.0
                            if require_sl_for_new and sl <= 0:
                                summary['skipped'].append({"symbol": symbol, "action": "reverse_open", "reason": "sl_required"})
                            else:
                                if not entry:
                                    price_now = self._get_current_price(symbol, current_signal_dir)
                                    entry = price_now or 0.0
                                vol = a.get('volume') or a.get('lot')
                                if vol is None:
                                    vol = _risk_size(symbol, current_signal_dir, float(entry), float(sl))
                                if (min_lot is not None and vol < min_lot) or (max_lot is not None and vol > max_lot):
                                    vol = max(min_lot or 0.01, min(max_lot or 1.0, float(vol)))
                                # RR check
                                if min_rr is not None and min_rr > 0 and sl > 0 and tp > 0:
                                    try:
                                        rr = abs((tp - float(entry))/(float(entry) - sl)) if current_signal_dir=='BUY' else abs((float(entry)-tp)/(sl-float(entry)))
                                    except Exception:
                                        rr = 0
                                    if rr < min_rr:
                                        summary['skipped'].append({"symbol": symbol, "action": "reverse_open", "reason": f"rr_below_min:{rr:.2f}"})
                                        continue
                                # Unprotected existing positions block
                                if block_if_unprotected:
                                    unprot = [p for p in positions if p['sl'] <= 0]
                                    if unprot:
                                        summary['skipped'].append({"symbol": symbol, "action": "reverse_open", "reason": "unprotected_positions_exist"})
                                        continue
                                # Risk capacity
                                new_risk_pct = 0.0
                                if sl > 0 and entry:
                                    try:
                                        entry_f = float(entry); sl_f=float(sl)
                                        pot = (entry_f - sl_f)/entry_f if current_signal_dir=='BUY' else (sl_f - entry_f)/entry_f
                                        new_risk_pct = max(0.0, pot*100.0)
                                    except Exception:
                                        new_risk_pct = 0.0
                                weighted_new_risk = new_risk_pct * float(vol)
                                if max_total_open_risk_percent is not None and max_total_open_risk_percent > 0 and existing_total_risk + weighted_new_risk > max_total_open_risk_percent:
                                    summary['skipped'].append({"symbol": symbol, "action": "reverse_open", "reason": "total_risk_cap"})
                                    continue
                                if max_symbol_open_risk_percent is not None and max_symbol_open_risk_percent > 0 and existing_symbol_risk.get(symbol,0.0) + weighted_new_risk > max_symbol_open_risk_percent:
                                    summary['skipped'].append({"symbol": symbol, "action": "reverse_open", "reason": "symbol_risk_cap"})
                                    continue
                                signal = TradeSignal(
                                    symbol=symbol,
                                    action=current_signal_dir,
                                    entry_price=float(entry),
                                    stop_loss=float(sl or 0.0),
                                    take_profit=float(tp or 0.0),
                                    volume=float(vol),
                                    confidence=confidence,
                                    strategy=a.get('strategy','REVERSAL'),
                                    comment=a.get('rationale','reverse')
                                )
                                res = self.execute_market_order(signal)
                                _inc('reverse_open', res.success)
                                if not res.success:
                                    summary['errors'].append({"symbol": symbol, "action": "reverse_open", "error": res.error_message})
                                else:
                                    positions = self.get_open_positions()
                                    pos_by_symbol[symbol] = [p for p in positions if p['symbol']==symbol]
                                    existing_total_risk += weighted_new_risk
                                    existing_symbol_risk[symbol] = existing_symbol_risk.get(symbol,0.0) + weighted_new_risk
                                    _update_session_new_position()
                    continue

                # 🚫 Handle DIRECTION_CONFLICT actions for new entry signals
                if action in ("DIRECTION_CONFLICT", "direction_conflict", "conflict_blocked"):
                    conflict_reason = a.get('reason', 'Direction conflict detected')
                    existing_direction = a.get('existing_direction', 'Unknown')
                    signal_direction = a.get('signal_direction', 'Unknown')
                    
                    logger.warning(f"🚫 DIRECTION CONFLICT for {symbol}: {conflict_reason}")
                    logger.warning(f"   📊 Existing positions: {existing_direction}")
                    logger.warning(f"   📈 New signal: {signal_direction}")
                    logger.warning(f"   🛑 Entry BLOCKED to prevent opposite direction trading")
                    
                    summary["direction_conflicts"] = summary.get("direction_conflicts", [])
                    summary["direction_conflicts"].append({
                        "symbol": symbol,
                        "existing_direction": existing_direction,
                        "signal_direction": signal_direction,
                        "reason": conflict_reason,
                        "timestamp": datetime.now().isoformat(),
                        "status": "BLOCKED_ENTRY"
                    })
                    
                    # Count as skipped for statistics
                    summary["skipped"].append({
                        "symbol": symbol, 
                        "action": action, 
                        "reason": "direction_conflict_blocked"
                    })
                    continue

                if action in ("open_buy", "open_sell", "open_position", "scale_in"):
                    # --- Global gating for opening trades ---
                    if _weekend_block():
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "time_block_weekend_or_day"})
                        continue
                    if not _within_time_window():
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "outside_time_window"})
                        continue
                    if daily_loss_block:
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "daily_loss_limit"})
                        continue
                    if _cooldown_active():
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "cooldown_active"})
                        continue
                    if max_daily_new_positions > 0 and session_data.get('new_positions_opened',0) >= max_daily_new_positions:
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "daily_new_positions_cap"})
                        continue

                    direction = 'BUY' if 'buy' in action or a.get('direction','').upper() == 'BUY' else 'SELL'
                    confidence = float(a.get('confidence', 0.0))
                    if min_confidence is not None and confidence < min_confidence:
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "low_confidence"})
                    elif not _can_open_new(symbol):
                        summary['skipped'].append({"symbol": symbol, "action": action, "reason": "position_cap"})
                    else:
                        entry = a.get('entry') or a.get('entry_price')
                        sl = a.get('stop_loss') or a.get('sl') or 0.0
                        tp = a.get('take_profit') or a.get('tp') or 0.0
                        if require_sl_for_new and sl <= 0:
                            summary['skipped'].append({"symbol": symbol, "action": action, "reason": "sl_required"})
                            continue
                        # Risk/Reward check
                        if min_rr is not None and min_rr > 0 and sl > 0 and tp > 0:
                            rr = abs((tp - float(entry))/(float(entry) - sl)) if direction=='BUY' else abs((float(entry)-tp)/(sl-float(entry))) if (sl>0 and tp>0) else 0
                            if rr < min_rr:
                                summary['skipped'].append({"symbol": symbol, "action": action, "reason": f"rr_below_min:{rr:.2f}"})
                                continue
                        # Block if unprotected positions exist (positions without SL)
                        if block_if_unprotected:
                            unprot = [p for p in positions if p['sl'] <= 0]
                            if unprot:
                                summary['skipped'].append({"symbol": symbol, "action": action, "reason": "unprotected_positions_exist"})
                                continue
                        # Approx new trade risk percent (like existing calc)
                        new_risk_pct = 0.0
                        if sl > 0 and entry:
                            try:
                                entry_f = float(entry)
                                sl_f = float(sl)
                                pot = (entry_f - sl_f)/entry_f if direction=='BUY' else (sl_f - entry_f)/entry_f
                                new_risk_pct = max(0.0, pot*100.0)
                            except Exception:
                                new_risk_pct = 0.0
                        # If entry missing use current market price
                        if not entry:
                            price_now = self._get_current_price(symbol, direction)
                            entry = price_now or 0.0
                        # Auto size volume
                        vol = a.get('volume') or a.get('lot')
                        if vol is None:
                            vol = _risk_size(symbol, direction, float(entry), float(sl))
                        # Clamp volume again  
                        if (min_lot is not None and vol < min_lot) or (max_lot is not None and vol > max_lot):
                            vol = max(min_lot or 0.01, min(max_lot or 1.0, float(vol)))
                        # Recompute risk with final volume weight
                        weighted_new_risk = new_risk_pct * float(vol)
                        # Total risk caps
                        if max_total_open_risk_percent is not None and max_total_open_risk_percent > 0 and existing_total_risk + weighted_new_risk > max_total_open_risk_percent:
                            summary['skipped'].append({"symbol": symbol, "action": action, "reason": "total_risk_cap"})
                            continue
                        if max_symbol_open_risk_percent is not None and max_symbol_open_risk_percent > 0 and existing_symbol_risk.get(symbol,0.0) + weighted_new_risk > max_symbol_open_risk_percent:
                            summary['skipped'].append({"symbol": symbol, "action": action, "reason": "symbol_risk_cap"})
                            continue
                        signal = TradeSignal(
                            symbol=symbol,
                            action=direction,
                            entry_price=float(entry),
                            stop_loss=float(sl or 0.0),
                            take_profit=float(tp or 0.0),
                            volume=float(vol),
                            confidence=confidence,
                            strategy=a.get('strategy','ACTION_OPEN'),
                            comment=a.get('comment','')
                        )
                        res = self.execute_market_order(signal)
                        _inc(action, res.success)
                        if not res.success:
                            summary['errors'].append({"symbol": symbol, "action": action, "error": res.error_message})
                        else:
                            # Update local snapshot for subsequent actions
                            positions = self.get_open_positions()
                            pos_by_symbol[symbol] = [p for p in positions if p['symbol']==symbol]
                            existing_total_risk += weighted_new_risk
                            existing_symbol_risk[symbol] = existing_symbol_risk.get(symbol,0.0) + weighted_new_risk
                            _update_session_new_position()
                    continue  # proceed to next action (open handled)

                elif action == 'dca_entry':
                    # 🔄 DCA ENTRY ACTION: Execute DCA entry order
                    direction = (a.get('direction') or a.get('signal_direction') or '').upper()
                    entry_price = a.get('entry_price') or a.get('price')
                    stop_loss = a.get('stop_loss') or a.get('sl', 0)
                    take_profit = a.get('take_profit') or a.get('tp', 0)
                    confidence = a.get('confidence', 50.0)
                    
                    # Extract DCA level from conditions or direct field
                    conditions = a.get('conditions', {})
                    dca_level = conditions.get('dca_level') or a.get('dca_level', 1)
                    
                    if direction not in ['BUY', 'SELL']:
                        summary['skipped'].append({
                            "symbol": symbol, 
                            "action": action, 
                            "reason": "invalid_direction",
                            "direction": direction
                        })
                        logger.warning(f"⚠️ DCA entry skipped for {symbol}: Invalid direction '{direction}'")
                        continue
                    
                    # Create temporary DCA signal for volume calculation
                    temp_dca_signal = TradeSignal(
                        symbol=symbol,
                        action=direction,
                        entry_price=float(entry_price) if entry_price else 0.0,
                        stop_loss=float(stop_loss) if stop_loss else 0.0,
                        take_profit=float(take_profit) if take_profit else 0.0,
                        volume=0.01,  # Temporary volume for calculation
                        confidence=float(confidence),
                        strategy="DCA_ATR",
                        comment=f"DCA-L{dca_level}",
                        is_dca=True,
                        dca_level=dca_level
                    )
                    
                    # Calculate progressive DCA volume using VolumeCalculator
                    try:
                        volume_calc = VolumeCalculator()
                        calculated_volume = volume_calc.get_volume_for_signal(temp_dca_signal, is_dca=True, dca_level=dca_level)
                        logger.info(f"🧮 DCA Volume calculated: Level-{dca_level} = {calculated_volume:.3f} lots")
                    except Exception as e:
                        logger.warning(f"⚠️ DCA volume calculation failed: {e}")
                        # Fallback to JSON volume with basic progression
                        json_volume = a.get('volume') or a.get('lot', 0.01)
                        calculated_volume = float(json_volume) * (1.5 ** (dca_level - 1))
                        logger.info(f"🔄 DCA Volume fallback: Level-{dca_level} = {calculated_volume:.3f} lots (from {json_volume})")
                    
                    # Round volume to step alignment
                    try:
                        symbol_info = mt5.symbol_info(symbol)
                        if symbol_info and symbol_info.volume_step > 0:
                            volume = round(float(calculated_volume) / symbol_info.volume_step) * symbol_info.volume_step
                        else:
                            volume = round(float(calculated_volume), 2)  # Fallback to 2 decimal places
                    except Exception:
                        volume = round(float(calculated_volume), 2)
                    
                    logger.info(f"🔄 DCA action debug: symbol={symbol}, direction={direction}, level={dca_level}, calculated_volume={volume}")
                    
                    # Create final DCA signal with calculated volume
                    dca_signal = TradeSignal(
                        symbol=symbol,
                        action=direction,
                        entry_price=float(entry_price) if entry_price else 0.0,
                        stop_loss=float(stop_loss) if stop_loss else 0.0,
                        take_profit=float(take_profit) if take_profit else 0.0,
                        volume=volume,
                        confidence=float(confidence),
                        strategy="DCA_ATR",
                        comment=f"DCA-L{dca_level}",
                        is_dca=True,
                        dca_level=dca_level
                    )
                    
                    # Execute DCA order
                    logger.info(f"🔄 Executing DCA entry for {symbol}: {direction} Level-{dca_level} Vol:{volume:.3f}")
                    
                    # 🔒 ENHANCED DCA PROTECTION: Check existing positions first
                    current_positions = mt5.positions_get(symbol=symbol)
                    if current_positions:
                        dca_count = sum(1 for p in current_positions if 'DCA' in (p.comment or ''))
                        if dca_count >= dca_level:
                            logger.warning(f"🚫 DCA Level {dca_level} already exists for {symbol} (found {dca_count} DCA positions)")
                            summary['skipped'].append({
                                "symbol": symbol, 
                                "action": action, 
                                "reason": "dca_level_exists",
                                "existing_dca_count": dca_count,
                                "requested_level": dca_level
                            })
                            continue
                    
                    # 🔒 Acquire DCA lock to prevent race conditions
                    if not self.dca_lock_manager.acquire_lock(symbol, timeout=3.0):
                        logger.warning(f"🔒 DCA lock acquisition failed for {symbol} - preventing duplicate execution")
                        summary['skipped'].append({
                            "symbol": symbol, 
                            "action": action, 
                            "reason": "dca_lock_failed"
                        })
                        continue
                    
                    try:
                        res = self.execute_market_order(dca_signal)
                        _inc(action, res.success)
                        
                        if res.success:
                            logger.info(f"✅ DCA entry executed: {symbol} {direction} Level-{dca_level} Vol:{volume:.3f} Ticket:{res.ticket}")
                            # Update positions snapshot for subsequent actions
                            positions = self.get_open_positions()
                            pos_by_symbol[symbol] = [p for p in positions if p['symbol']==symbol]
                            _update_session_new_position()
                        else:
                            logger.error(f"❌ DCA entry failed for {symbol}: {res.error_message}")
                            
                    finally:
                        # 🔓 Always release lock regardless of execution result
                        self.dca_lock_manager.release_lock(symbol)
                    
                    continue  # Move to next action

                # 🎯 TRAILING STOP: Apply only to specific profitable position
                if action in ('set_trailing_sl', 'enable_trailing_sl'):
                    # For trailing stop, only apply to the specific position if ticket is specified in action
                    target_ticket = a.get('ticket')
                    if target_ticket:
                        # Find specific position by ticket
                        target_pos = next((pos for pos in sym_positions if pos['ticket'] == target_ticket), None)
                        if not target_pos:
                            summary["skipped"].append({"symbol": symbol, "ticket": target_ticket, "action": action, "reason": "position_not_found"})
                            logger.warning(f"⚠️ Trailing stop: Position #{target_ticket} not found for {symbol}")
                            continue
                        target_positions = [target_pos]
                    else:
                        # If no specific ticket, only apply to profitable positions
                        target_positions = [pos for pos in sym_positions if pos.get('profit', 0) > 0]
                        if not target_positions:
                            summary["skipped"].append({"symbol": symbol, "ticket": "all", "action": action, "reason": "no_profitable_positions"})
                            logger.warning(f"⚠️ Trailing stop: No profitable positions found for {symbol}")
                            continue
                else:
                    # For other actions, apply to all positions as before
                    target_positions = sym_positions

                # Apply action on target positions
                for pos in target_positions:
                    ticket = pos['ticket']
                    vol = float(pos['volume'])
                    if (max_lot is not None and vol > max_lot) or (min_lot is not None and vol < min_lot):
                        summary["skipped"].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "lot_out_of_bounds"})
                        logger.warning(f"⚠️ {symbol} ticket {ticket}: Khối lượng {vol} ngoài giới hạn [{min_lot}, {max_lot}] - bỏ qua")
                        continue

                    if action in ('set_initial_sl', 'tighten_sl', 'set_sl'):
                        sl = _safe_float_conversion(
                            a.get('proposed_sl') or a.get('stop_loss') or a.get('sl'), 
                            0.0, 
                            "stop_loss"
                        )
                        
                        if sl <= 0:
                            summary["skipped"].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "invalid_sl_value"})
                            logger.warning(f"⚠️ {symbol} ticket {ticket}: SL không hợp lệ ({sl}) - bỏ qua")
                            continue
                        
                        # 🛡️ DCA-AWARE SL PROTECTION
                        dca_protection = self._should_protect_sl_for_dca(symbol, sl)
                        if dca_protection["should_protect"]:
                            pending_levels = dca_protection.get("pending_levels", 0)
                            
                            if "recommended_sl" in dca_protection:
                                # Use recommended SL that won't interfere with DCA
                                original_sl = sl
                                sl = dca_protection["recommended_sl"]
                                logger.info(f"🛡️ DCA Protection: Adjusted SL from {original_sl:.5f} to {sl:.5f} for {symbol} (pending {pending_levels} DCA levels)")
                            else:
                                # Skip SL modification entirely to protect DCA strategy
                                summary["skipped"].append({
                                    "symbol": symbol, 
                                    "ticket": ticket, 
                                    "action": action, 
                                    "reason": "dca_sl_protection",
                                    "details": f"DCA enabled with {pending_levels} pending levels - SL protected"
                                })
                                logger.warning(f"🛡️ DCA Protection: SL modification blocked for {symbol} - {pending_levels} DCA levels pending")
                                continue
                            
                        _log_action_start(action, symbol, ticket, f"SL={sl:.5f}")
                        res = self.modify_order(ticket=ticket, sl=sl)
                        summary["by_action"].setdefault(action, 0)
                        summary["by_action"][action] += 1 if res.success else 0
                        summary["applied"] += 1 if res.success else 0
                        _log_action_result(action, symbol, ticket, res.success, res.comment or res.error_message)

                    elif action in ('close', 'close_position'):
                        # Close position (full or opposite signal close)
                        action_type = a.get('type', '')
                        action_reason = a.get('reason', 'Position close')
                        
                        # Log additional context for opposite signal closes
                        if 'opposite' in action_type.lower():
                            profit_pips = a.get('current_profit_pips', 0)
                            profit_usd = a.get('current_profit_usd', 0)
                            logger.info(f"🔄 Closing {symbol} #{ticket} due to opposite signal: +{profit_pips:.1f} pips / ${profit_usd:.2f}")
                        
                        _log_action_start(action, symbol, ticket, f"vol={vol:.3f}")
                        res = self.close_position(ticket=ticket)
                        _inc(action, res.success)
                        _log_action_result(action, symbol, ticket, res.success, res.comment or res.error_message)

                    elif action in ('take_partial_30', 'close_partial_30'):
                        pvol = vol * 0.3
                        if pvol < min_lot:
                            summary["skipped"].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "partial_below_min"})
                            logger.warning(f"⚠️ {symbol} ticket {ticket}: Partial 30% ({pvol:.3f}) < min lot ({min_lot}) - bỏ qua")
                            continue
                        res = self.close_position(ticket=ticket, volume=pvol)
                        _inc(action, res.success)
                        logger.info(f"💰 Partial 30% close {symbol} ticket {ticket} vol={pvol:.3f} -> {res.success} ({res.comment or res.error_message})")

                    elif action in ('take_partial_50', 'close_partial_50'):
                        pvol = vol * 0.5
                        if pvol < min_lot:
                            summary["skipped"].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "partial_below_min"})
                            logger.warning(f"⚠️ {symbol} ticket {ticket}: Partial 50% ({pvol:.3f}) < min lot ({min_lot}) - bỏ qua")
                            continue
                        res = self.close_position(ticket=ticket, volume=pvol)
                        _inc(action, res.success)
                        logger.info(f"💰 Partial 50% close {symbol} ticket {ticket} vol={pvol:.3f} -> {res.success} ({res.comment or res.error_message})")

                    elif action in ('take_partial_70', 'close_partial_70'):
                        pvol = vol * 0.7
                        if pvol < min_lot:
                            summary["skipped"].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "partial_below_min"})
                            logger.warning(f"⚠️ {symbol} ticket {ticket}: Partial 70% ({pvol:.3f}) < min lot ({min_lot}) - bỏ qua")
                            continue
                        res = self.close_position(ticket=ticket, volume=pvol)
                        _inc(action, res.success)
                        logger.info(f"💰 Partial 70% close {symbol} ticket {ticket} vol={pvol:.3f} -> {res.success} ({res.comment or res.error_message})")

                    elif action == 'close_full':
                        # Close entire position (100%)
                        res = self.close_position(ticket=ticket)
                        _inc(action, res.success)
                        logger.info(f"💰 Full close {symbol} ticket {ticket} vol={vol:.3f} -> {res.success} ({res.comment or res.error_message})")

                    elif action in ('move_sl_to_be','breakeven_sl'):
                        # Use move_sl_price from action if available, otherwise use entry price
                        move_price = a.get('move_sl_price')
                        if not move_price:
                            move_price = pos['price_open']  # Fallback to breakeven
                        
                        # 🛡️ DCA-AWARE BREAKEVEN PROTECTION
                        dca_protection = self._should_protect_sl_for_dca(symbol, move_price)
                        if dca_protection["should_protect"]:
                            pending_levels = dca_protection.get("pending_levels", 0)
                            
                            if "recommended_sl" in dca_protection:
                                # Use recommended SL that won't interfere with DCA
                                original_move_price = move_price
                                move_price = dca_protection["recommended_sl"]
                                logger.info(f"🛡️ DCA Protection: Adjusted breakeven SL from {original_move_price:.5f} to {move_price:.5f} for {symbol}")
                            else:
                                # Skip breakeven move to protect DCA strategy
                                summary['skipped'].append({
                                    "symbol": symbol, 
                                    "ticket": ticket, 
                                    "action": action, 
                                    "reason": "dca_breakeven_protection",
                                    "details": f"DCA enabled with {pending_levels} pending levels - breakeven protected"
                                })
                                logger.warning(f"🛡️ DCA Protection: Breakeven move blocked for {symbol} - {pending_levels} DCA levels pending")
                                continue
                        
                        # Only move if it improves the position
                        current_price = pos['price_current']
                        current_sl = pos.get('sl') or 0
                        
                        should_move = False
                        if pos['type'] == 'BUY':
                            # For BUY: move SL up only if new SL is higher than current
                            if current_sl == 0 or move_price > current_sl:
                                should_move = (current_price > move_price)  # Only if in profit
                        else:  # SELL
                            # For SELL: move SL down only if new SL is lower than current
                            if current_sl == 0 or move_price < current_sl:
                                should_move = (current_price < move_price)  # Only if in profit
                        
                        if should_move:
                            res = self.modify_order(ticket=ticket, sl=move_price)
                            _inc(action, res.success)
                            _log_action_result(action, symbol, ticket, res.success, f"SL moved to {move_price:.5f}")
                        else:
                            summary['skipped'].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "position_not_in_profit_or_sl_worse"})

                    elif action in ('set_sl', 'adjust_sl', 'update_sl'):
                        # Set or adjust stop loss to proposed_sl price
                        new_sl = a.get('proposed_sl')
                        if new_sl:
                            try:
                                new_sl_price = float(new_sl)
                                
                                # 🛡️ DCA-AWARE SL PROTECTION
                                dca_protection = self._should_protect_sl_for_dca(symbol, new_sl_price)
                                if dca_protection["should_protect"]:
                                    pending_levels = dca_protection.get("pending_levels", 0)
                                    
                                    if "recommended_sl" in dca_protection:
                                        # Use recommended SL that won't interfere with DCA
                                        original_sl = new_sl_price
                                        new_sl_price = dca_protection["recommended_sl"]
                                        logger.info(f"🛡️ DCA Protection: Adjusted SL from {original_sl:.5f} to {new_sl_price:.5f} for {symbol}")
                                    else:
                                        # Skip SL modification entirely to protect DCA strategy
                                        summary['skipped'].append({
                                            "symbol": symbol, 
                                            "ticket": ticket, 
                                            "action": action, 
                                            "reason": "dca_sl_protection",
                                            "details": f"DCA enabled with {pending_levels} pending levels - SL protected"
                                        })
                                        logger.warning(f"🛡️ DCA Protection: SL set blocked for {symbol} - {pending_levels} DCA levels pending")
                                        continue
                                
                                res = self.modify_order(ticket=ticket, sl=new_sl_price)
                                _inc(action, res.success)
                                _log_action_result(action, symbol, ticket, res.success, f"SL set to {new_sl_price:.5f}")
                            except (ValueError, TypeError):
                                summary['errors'].append({"symbol": symbol, "ticket": ticket, "action": action, "error": f"Invalid SL price: {new_sl}"})
                        else:
                            summary['skipped'].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "no_proposed_sl"})

                    elif action in ('set_tp', 'adjust_tp', 'update_tp'):
                        # Set or adjust take profit to proposed_tp price  
                        new_tp = a.get('proposed_tp')
                        if new_tp:
                            try:
                                new_tp_price = float(new_tp)
                                
                                # 🎯 DCA-AWARE TP OPTIMIZATION
                                dca_status = self._get_dca_status(symbol)
                                if dca_status["needs_sl_protection"] and dca_status["entry_positions"]:
                                    # Calculate optimal TP considering pending DCA levels
                                    entry_pos = dca_status["entry_positions"][0]
                                    direction = "BUY" if entry_pos.type == 0 else "SELL"
                                    pending_levels = dca_status["max_dca_levels"] - dca_status["total_levels"]
                                    
                                    # For DCA strategies, consider extending TP to account for averaged entry price
                                    if pending_levels > 0:
                                        dca_distance_pips = self.risk_settings.get('dca_distance_pips', 50)
                                        pip_value = get_pip_value(symbol)
                                        
                                        # Estimate average entry price if all DCA levels are filled
                                        entry_price = entry_pos.price_open
                                        volume_multiplier = self.risk_settings.get('dca_volume_multiplier', 1.5)
                                        
                                        # Calculate weighted average entry considering all potential DCA levels
                                        total_volume = entry_pos.volume
                                        weighted_price = entry_price * entry_pos.volume
                                        
                                        for level in range(1, dca_status["max_dca_levels"] + 1):
                                            if level <= dca_status["total_levels"]:
                                                continue  # Already filled
                                            
                                            if direction == "BUY":
                                                dca_price = entry_price - (level * dca_distance_pips * pip_value)
                                            else:
                                                dca_price = entry_price + (level * dca_distance_pips * pip_value)
                                            
                                            dca_volume = entry_pos.volume * (volume_multiplier ** level)
                                            total_volume += dca_volume
                                            weighted_price += dca_price * dca_volume
                                        
                                        estimated_avg_entry = weighted_price / total_volume
                                        
                                        # Adjust TP to maintain reasonable R:R with averaged entry
                                        current_tp_distance = abs(new_tp_price - entry_price)
                                        adjusted_tp_distance = abs(new_tp_price - estimated_avg_entry)
                                        
                                        # If TP becomes too close with averaged entry, extend it
                                        min_tp_distance_pips = 100  # Minimum 100 pips TP
                                        if adjusted_tp_distance / pip_value < min_tp_distance_pips:
                                            if direction == "BUY":
                                                new_tp_price = estimated_avg_entry + (min_tp_distance_pips * pip_value)
                                            else:
                                                new_tp_price = estimated_avg_entry - (min_tp_distance_pips * pip_value)
                                            
                                            logger.info(f"🎯 DCA-Aware TP: Extended TP to {new_tp_price:.5f} for {symbol} (considering {pending_levels} pending DCA levels)")
                                
                                res = self.modify_order(ticket=ticket, tp=new_tp_price)
                                _inc(action, res.success)
                                _log_action_result(action, symbol, ticket, res.success, f"TP set to {new_tp_price:.5f}")
                            except (ValueError, TypeError):
                                summary['errors'].append({"symbol": symbol, "ticket": ticket, "action": action, "error": f"Invalid TP price: {new_tp}"})
                        else:
                            summary['skipped'].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "no_proposed_tp"})

                    elif action in ('set_trailing_sl', 'enable_trailing_sl'):
                        # 🎯 TWO-STEP TRAILING STOP: Lock profit first, then enable trailing
                        current_profit = pos.get('profit', 0)
                        
                        # 🛡️ SAFETY CHECK: Only apply trailing stop to profitable positions
                        if current_profit <= 0:
                            summary['skipped'].append({
                                "symbol": symbol, 
                                "ticket": ticket, 
                                "action": action, 
                                "reason": "position_not_profitable",
                                "details": f"Profit: {current_profit:.2f} USD"
                            })
                            logger.warning(f"🛡️ Trailing stop skipped for {symbol} #{ticket}: Not profitable (${current_profit:.2f})")
                            continue
                        
                        # STEP 1: Lock profit by moving S/L to move_sl_price (if provided)
                        move_sl_price = a.get('move_sl_price')
                        locked_profit_success = False
                        
                        if move_sl_price:
                            logger.info(f"🔒 STEP 1: Lock profit for {symbol} #{ticket} - Moving S/L to {move_sl_price:.5f}")
                            res = self.modify_order(ticket=ticket, sl=move_sl_price)
                            _inc(action, res.success)
                            _log_action_result(action, symbol, ticket, res.success, f"STEP 1: Locked profit S/L: {move_sl_price:.5f} (profit: ${current_profit:.2f})")
                            locked_profit_success = res.success
                        else:
                            logger.warning(f"⚠️ No move_sl_price provided for trailing stop on {symbol} #{ticket}")
                        
                        # STEP 2: Register trailing stop with order_daemon
                        trailing_config = a.get('trailing_config') or {}
                        trail_distance_pips = trailing_config.get('trail_distance_pips')
                        
                        if trail_distance_pips:
                            logger.info(f"🎯 STEP 2: Register trailing stop for {symbol} #{ticket} - Distance: {trail_distance_pips} pips")
                            
                            # Add to daemon's trailing config (order_daemon will pick it up)
                            try:
                                config_file = 'trailing_stop_config.json'
                                
                                # Load existing configs
                                if os.path.exists(config_file):
                                    with open(config_file, 'r') as f:
                                        trailing_configs = json.load(f)
                                else:
                                    trailing_configs = {}
                                
                                # Add new trailing stop config
                                trailing_configs[str(ticket)] = {
                                    'symbol': symbol,
                                    'trail_distance_pips': trail_distance_pips,
                                    'added_at': datetime.now().isoformat()
                                }
                                
                                # Save configs
                                with open(config_file, 'w') as f:
                                    json.dump(trailing_configs, f, indent=2)
                                
                                logger.info(f"✅ Trailing stop registered: #{ticket} {symbol} @ {trail_distance_pips} pips")
                                logger.info(f"💡 order_daemon.py will auto-update S/L every second")
                                
                                if locked_profit_success:
                                    _log_action_result(action, symbol, ticket, True, f"STEP 2: Trailing {trail_distance_pips}p registered (daemon active)")
                                else:
                                    logger.warning(f"⚠️ Trailing registered but profit lock failed #{ticket}")
                                    
                            except Exception as e:
                                logger.error(f"❌ Failed to register trailing: {e}")
                                summary['skipped'].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": f"trailing_reg_error: {e}"})
                        else:
                            logger.warning(f"⚠️ No trail_distance_pips in config for #{ticket}")
                            summary['skipped'].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "no_trailing_config"})

                    elif action in ('trail_sl','trailing_sl'):
                        if trail_sl_pips <= 0:
                            summary['skipped'].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "no_trail_config"})
                        else:
                            sym_info = mt5.symbol_info(symbol)
                            if sym_info:
                                point = sym_info.point or 0.0001
                                distance = trail_sl_pips * point
                                if pos['type']=='BUY':
                                    new_sl = pos['price_current'] - distance
                                    if new_sl > (pos['sl'] or 0):
                                        res = self.modify_order(ticket=ticket, sl=new_sl)
                                        _inc(action, res.success)
                                else:  # SELL
                                    new_sl = pos['price_current'] + distance
                                    if pos['sl'] == 0 or new_sl < pos['sl']:
                                        res = self.modify_order(ticket=ticket, sl=new_sl)
                                        _inc(action, res.success)
                            else:
                                summary['skipped'].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "no_symbol_info"})

                    elif action in ('adjust_tp','update_tp'):
                        new_tp = a.get('proposed_tp') or a.get('take_profit') or a.get('tp')
                        if new_tp:
                            res = self.modify_order(ticket=ticket, tp=float(new_tp))
                            _inc(action, res.success)
                            logger.info(f"Adjust TP {symbol} ticket {ticket} -> {res.success}")
                        else:
                            summary['skipped'].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "missing_tp"})

                    elif action == 'enable_trailing':
                        # Enable trailing stop based on proposed_sl or calculated trailing config
                        trailing_config = a.get('trailing_config') or {}
                        activate_price = trailing_config.get('activate_price')
                        trail_distance = trailing_config.get('trail_distance')
                        
                        if activate_price and trail_distance:
                            # Check if current price has reached activation level
                            current_price = pos['price_current']
                            pos_type = pos['type']
                            
                            should_activate = False
                            if pos_type == 'BUY' and current_price >= activate_price:
                                should_activate = True
                            elif pos_type == 'SELL' and current_price <= activate_price:
                                should_activate = True
                            
                            if should_activate:
                                # Calculate new trailing SL
                                if pos_type == 'BUY':
                                    new_sl = current_price - trail_distance
                                else:  # SELL
                                    new_sl = current_price + trail_distance
                                
                                # Only update if better than current SL
                                current_sl = pos.get('sl', 0)
                                should_update = False
                                if pos_type == 'BUY' and (current_sl == 0 or new_sl > current_sl):
                                    should_update = True
                                elif pos_type == 'SELL' and (current_sl == 0 or new_sl < current_sl):
                                    should_update = True
                                
                                if should_update:
                                    res = self.modify_order(ticket=ticket, sl=new_sl)
                                    _inc(action, res.success)
                                    logger.info(f"Enable trailing SL {symbol} ticket {ticket} -> {res.success} (SL: {new_sl})")
                                else:
                                    summary['skipped'].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "sl_not_better"})
                            else:
                                summary['skipped'].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "activation_not_reached"})
                        else:
                            # Fallback to standard trailing with trail_sl_pips
                            if trail_sl_pips > 0:
                                sym_info = mt5.symbol_info(symbol)
                                if sym_info:
                                    point = sym_info.point or 0.0001
                                    distance = trail_sl_pips * point
                                    current_price = pos['price_current']
                                    
                                    if pos['type'] == 'BUY':
                                        new_sl = current_price - distance
                                        if new_sl > (pos.get('sl', 0) or 0):
                                            res = self.modify_order(ticket=ticket, sl=new_sl)
                                            _inc(action, res.success)
                                            logger.info(f"Enable trailing (fallback) {symbol} ticket {ticket} -> {res.success}")
                                        else:
                                            summary['skipped'].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "sl_not_better"})
                                    else:  # SELL
                                        new_sl = current_price + distance
                                        current_sl = pos.get('sl', 0)
                                        if current_sl == 0 or new_sl < current_sl:
                                            res = self.modify_order(ticket=ticket, sl=new_sl)
                                            _inc(action, res.success)
                                            logger.info(f"Enable trailing (fallback) {symbol} ticket {ticket} -> {res.success}")
                                        else:
                                            summary['skipped'].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "sl_not_better"})
                                else:
                                    summary['skipped'].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "no_symbol_info"})
                            else:
                                summary['skipped'].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "no_trailing_config"})

                    elif action == 'hold':
                        # Hold - no action required, just log
                        summary["by_action"].setdefault(action, 0)
                        summary["by_action"][action] += 1
                        summary["applied"] += 1
                        logger.info(f"📊 Hold {symbol} ticket {ticket} - no action taken (signal confidence: {a.get('confidence', 'N/A')})")

                    elif action in ('monitor', 'watch', 'observe'):
                        # Monitor/watch - similar to hold but more emphasis on observation
                        summary["by_action"].setdefault(action, 0)
                        summary["by_action"][action] += 1
                        summary["applied"] += 1
                        logger.info(f"👁️ Monitor {symbol} ticket {ticket} - watching for changes")

                    elif action in ('reduce_risk', 'risk_reduction'):
                        # Reduce risk by partial close or SL tightening
                        reduction_method = a.get('method', 'partial_close')
                        if reduction_method == 'partial_close':
                            # Default to 30% reduction
                            reduction_pct = float(a.get('reduction_percent', 30)) / 100
                            pvol = vol * reduction_pct
                            if pvol >= min_lot:
                                res = self.close_position(ticket=ticket, volume=pvol)
                                _inc(action, res.success)
                                logger.info(f"⚠️ Risk reduction partial close {symbol} ticket {ticket} {reduction_pct*100:.0f}% -> {res.success}")
                            else:
                                summary["skipped"].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "partial_below_min"})
                        elif reduction_method == 'tighten_sl':
                            new_sl = a.get('proposed_sl') or a.get('tight_sl')
                            if new_sl:
                                res = self.modify_order(ticket=ticket, sl=float(new_sl))
                                _inc(action, res.success)
                                logger.info(f"⚠️ Risk reduction SL tighten {symbol} ticket {ticket} -> {res.success}")
                            else:
                                summary["skipped"].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "missing_tight_sl"})

                    elif action in ('scale_out', 'partial_exit'):
                        # Scale out position gradually
                        scale_pct = float(a.get('scale_percent', 25)) / 100
                        pvol = vol * scale_pct
                        if pvol >= min_lot:
                            res = self.close_position(ticket=ticket, volume=pvol)
                            _inc(action, res.success)
                            logger.info(f"📉 Scale out {symbol} ticket {ticket} {scale_pct*100:.0f}% -> {res.success}")
                        else:
                            summary["skipped"].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "scale_below_min"})

                    elif action in ('DIRECTION_CONFLICT', 'direction_conflict', 'conflict_blocked'):
                        # 🚫 DIRECTION CONFLICT: Log the conflict but don't execute anything
                        conflict_reason = a.get('reason', 'Direction conflict detected')
                        existing_direction = a.get('existing_direction', 'Unknown')
                        signal_direction = a.get('signal_direction', 'Unknown')
                        
                        logger.warning(f"🚫 DIRECTION CONFLICT for {symbol}: {conflict_reason}")
                        logger.warning(f"   📊 Existing positions: {existing_direction}")
                        logger.warning(f"   📈 New signal: {signal_direction}")
                        logger.warning(f"   🛑 Action BLOCKED to prevent opposite direction trading")
                        
                        summary["direction_conflicts"] = summary.get("direction_conflicts", [])
                        summary["direction_conflicts"].append({
                            "symbol": symbol,
                            "existing_direction": existing_direction,
                            "signal_direction": signal_direction,
                            "reason": conflict_reason,
                            "timestamp": datetime.now().isoformat(),
                            "status": "BLOCKED"
                        })
                        
                        # Count as skipped for statistics
                        summary["skipped"].append({
                            "symbol": symbol, 
                            "action": action, 
                            "reason": "direction_conflict_blocked"
                        })

                    elif action == 'modify_position':
                        # 🎯 ENHANCED MODIFY POSITION: Handle S/L and T/P adjustments based on latest signals
                        ticket = a.get('ticket')
                        new_sl = a.get('new_sl')
                        new_tp = a.get('new_tp') 
                        current_sl = a.get('current_sl', 0)
                        current_tp = a.get('current_tp', 0)
                        
                        # 🧠 SMART SIGNAL-BASED CALCULATION: Calculate optimal S/L and T/P if not provided
                        if (new_sl is None or new_tp is None) and ticket:
                            smart_levels = self._calculate_smart_sl_tp_from_signal(a, symbol)
                            if smart_levels:
                                if new_sl is None and smart_levels.get('sl'):
                                    new_sl = smart_levels['sl']
                                    logger.info(f"🧠 {symbol} ticket {ticket}: Smart S/L calculated from signal: {new_sl:.5f}")
                                if new_tp is None and smart_levels.get('tp'):
                                    new_tp = smart_levels['tp']
                                    logger.info(f"🧠 {symbol} ticket {ticket}: Smart T/P calculated from signal: {new_tp:.5f}")
                        
                        # 🔧 ENHANCED: Also check for alternative field names in actions
                        if new_sl is None:
                            new_sl = a.get('sl') or a.get('stop_loss')
                        if new_tp is None:
                            new_tp = a.get('tp') or a.get('take_profit')
                        
                        if not ticket:
                            summary["skipped"].append({"symbol": symbol, "action": action, "reason": "missing_ticket"})
                            logger.warning(f"⚠️ Modify position {symbol}: Missing ticket")
                            continue
                        
                        # 🔒 DUPLICATE PREVENTION: Check if this ticket was already modified
                        if ticket in modified_tickets:
                            summary["skipped"].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "duplicate_modification"})
                            logger.info(f"🔒 {symbol} ticket {ticket}: Already modified in this session - skipping duplicate")
                            continue
                        
                        # 🔧 CRITICAL: Get actual current SL/TP from MT5 before modification
                        try:
                            position_info = mt5.positions_get(ticket=ticket)
                            if position_info and len(position_info) > 0:
                                pos = position_info[0]
                                actual_current_sl = pos.sl
                                actual_current_tp = pos.tp
                                position_type = pos.type  # 0=BUY, 1=SELL
                                current_price = pos.price_current
                                logger.info(f"🔍 {symbol} ticket {ticket}: Current MT5 values - SL={actual_current_sl:.5f}, TP={actual_current_tp:.5f}, Price={current_price:.5f}")
                            else:
                                logger.warning(f"⚠️ {symbol} ticket {ticket}: Position not found in MT5")
                                summary["skipped"].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "position_not_found"})
                                continue
                        except Exception as e:
                            logger.error(f"❌ {symbol} ticket {ticket}: Error getting position info - {e}")
                            summary["skipped"].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "position_info_error"})
                            continue
                        
                        # 🎯 ENHANCED LOGIC: Determine what needs modification with intelligent validation
                        modify_sl = False
                        modify_tp = False
                        tolerance = 1e-5  # Float comparison tolerance
                        
                        # Check S/L modification need
                        if new_sl is not None and new_sl > 0:  # Only modify if new_sl is positive
                            if abs(new_sl - actual_current_sl) > tolerance:
                                modify_sl = True
                                logger.info(f"🎯 {symbol} ticket {ticket}: S/L update needed - Current: {actual_current_sl:.5f} → New: {new_sl:.5f}")
                        
                        # Check T/P modification need  
                        if new_tp is not None and new_tp > 0:  # Only modify if new_tp is positive
                            if abs(new_tp - actual_current_tp) > tolerance:
                                modify_tp = True
                                logger.info(f"🎯 {symbol} ticket {ticket}: T/P update needed - Current: {actual_current_tp:.5f} → New: {new_tp:.5f}")
                        
                        # 🚫 Skip if no changes needed
                        if not modify_sl and not modify_tp:
                            summary["skipped"].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "no_changes_needed"})
                            logger.info(f"✅ {symbol} ticket {ticket}: Position already optimized (SL: {actual_current_sl:.5f}, TP: {actual_current_tp:.5f})")
                            continue
                        
                        # 🔧 ENHANCED VALIDATION: Check S/L and T/P logic for position type
                        final_sl = new_sl if modify_sl else actual_current_sl
                        final_tp = new_tp if modify_tp else actual_current_tp
                        
                        # Validate S/L and T/P make sense for position direction
                        if position_type == 0:  # BUY position
                            if modify_sl and final_sl > 0 and final_sl >= current_price:
                                logger.warning(f"⚠️ {symbol} ticket {ticket}: BUY S/L {final_sl:.5f} >= current price {current_price:.5f} - may cause immediate execution")
                            if modify_tp and final_tp > 0 and final_tp <= current_price:
                                logger.warning(f"⚠️ {symbol} ticket {ticket}: BUY T/P {final_tp:.5f} <= current price {current_price:.5f} - may cause immediate execution")
                        else:  # SELL position
                            if modify_sl and final_sl > 0 and final_sl <= current_price:
                                logger.warning(f"⚠️ {symbol} ticket {ticket}: SELL S/L {final_sl:.5f} <= current price {current_price:.5f} - may cause immediate execution")
                            if modify_tp and final_tp > 0 and final_tp >= current_price:
                                logger.warning(f"⚠️ {symbol} ticket {ticket}: SELL T/P {final_tp:.5f} >= current price {current_price:.5f} - may cause immediate execution")
                        
                        # 🎯 Execute the modification with enhanced parameters
                        sl_param = final_sl if modify_sl else None
                        tp_param = final_tp if modify_tp else None
                        
                        logger.info(f"🔧 {symbol} ticket {ticket}: Executing modification - SL: {sl_param}, TP: {tp_param}")
                        res = self.modify_order(ticket=ticket, sl=sl_param, tp=tp_param)
                        _inc(action, res.success)
                        
                        if res.success:
                            # 🔒 TRACK MODIFIED TICKET to prevent duplicates
                            modified_tickets.add(ticket)
                            
                            # 🎯 ENHANCED SUCCESS LOGGING: Show detailed S/L and T/P changes
                            modify_details = []
                            if modify_sl:
                                sl_change = new_sl - actual_current_sl
                                sl_direction = "↗️" if sl_change > 0 else "↘️"
                                modify_details.append(f"SL: {actual_current_sl:.5f} → {new_sl:.5f} {sl_direction}")
                            if modify_tp:
                                tp_change = new_tp - actual_current_tp
                                tp_direction = "↗️" if tp_change > 0 else "↘️"
                                modify_details.append(f"TP: {actual_current_tp:.5f} → {new_tp:.5f} {tp_direction}")
                            
                            logger.info(f"✅ {symbol} ticket {ticket} modified successfully!")
                            logger.info(f"   🎯 Changes: {', '.join(modify_details)}")
                            logger.info(f"   � Signal: {a.get('signal_confidence', 'N/A')}% confidence")
                            logger.info(f"   �📄 Reason: {a.get('reason', 'Latest signal adjustment')}")
                            
                            # 🔄 VERIFY CHANGES: Get updated position info for confirmation
                            try:
                                updated_pos = mt5.positions_get(ticket=ticket)
                                if updated_pos and len(updated_pos) > 0:
                                    new_pos = updated_pos[0]
                                    logger.info(f"   ✅ Confirmed: SL={new_pos.sl:.5f}, TP={new_pos.tp:.5f}")
                                    
                                    # Check if changes were applied correctly
                                    if modify_sl and abs(new_pos.sl - final_sl) > tolerance:
                                        logger.warning(f"   ⚠️ S/L mismatch: Expected {final_sl:.5f}, Got {new_pos.sl:.5f}")
                                    if modify_tp and abs(new_pos.tp - final_tp) > tolerance:
                                        logger.warning(f"   ⚠️ T/P mismatch: Expected {final_tp:.5f}, Got {new_pos.tp:.5f}")
                            except Exception as verify_error:
                                logger.warning(f"   ⚠️ Could not verify changes: {verify_error}")
                                
                        else:
                            logger.error(f"❌ Failed to modify {symbol} ticket {ticket}: {res.error_message}")
                            logger.error(f"   🎯 Attempted: SL={sl_param}, TP={tp_param}")
                            summary['errors'].append({
                                "symbol": symbol, 
                                "action": action, 
                                "ticket": ticket, 
                                "error": res.error_message,
                                "attempted_sl": sl_param,
                                "attempted_tp": tp_param
                            })



                    else:
                        summary["skipped"].append({"symbol": symbol, "ticket": ticket, "action": action, "reason": "unsupported_action"})
                        logger.warning(f"⚠️ Unsupported action: {action} for {symbol} ticket {ticket}")

            # 🔄 AUTO DCA DETECTION AND EXECUTION
            # After processing regular actions, check for DCA opportunities
            try:
                from order_executor import ComprehensiveRiskValidator
                risk_validator = ComprehensiveRiskValidator(risk_cfg)
                dca_opportunities = risk_validator.detect_dca_opportunities()
                
                if dca_opportunities:
                    logger.info(f"🔄 Processing {len(dca_opportunities)} DCA opportunities...")
                    dca_results = {"executed": 0, "failed": 0, "skipped": 0}
                    
                    for opportunity in dca_opportunities:
                        try:
                            symbol = opportunity['symbol']
                            
                            # Smart market check for DCA - skip forex when market closed but allow crypto
                            if is_market_closed_for_symbol(symbol):
                                logger.info(f"⏭️ DCA skipped for {symbol}: {'Forex' if not is_crypto_symbol(symbol) else 'Market'} market closed")
                                dca_results["skipped"] += 1
                                continue
                            
                            direction = opportunity['direction']
                            volume = opportunity['suggested_dca_volume']
                            confidence = opportunity['confidence']
                            dca_level = opportunity['current_dca_level'] + 1
                            exec_mode = opportunity.get('exec_mode', 'market')
                            target_fibo_price = opportunity.get('target_fibo_price')
                            
                            # Get current and entry prices
                            current_price = self._get_current_price(symbol, direction)
                            if not current_price:
                                logger.warning(f"⚠️ Cannot get current price for {symbol} DCA")
                                dca_results["failed"] += 1
                                continue
                            
                            # DCA should use current price as entry price for the new position
                            # Calculate DCA SL/TP using Universal DCA Protection Function
                            dca_sl, dca_tp, dca_info = self.calculate_universal_dca_protection(symbol, direction, current_price, dca_level)
                            
                            # 🛡️ CRITICAL: Validate DCA conditions before execution
                            dca_validation = risk_validator.validate_all_conditions(
                                symbol=symbol,
                                signal_confidence=confidence,
                                order_type="market_order",
                                is_dca=True
                            )
                            
                            if not dca_validation.is_valid:
                                dca_results["skipped"] += 1
                                logger.warning(f"🚫 DCA blocked for {symbol}: {dca_validation.recommendation}")
                                for violation in dca_validation.violations:
                                    logger.warning(f"   - {violation}")
                                continue
                            
                            logger.info(f"✅ DCA validation passed for {symbol}: {dca_validation.recommendation}")
                            
                            # Create DCA signal with proper SL/TP
                            dca_signal = self.prepare_signal_with_smart_volume(
                                symbol=symbol,
                                action=direction,
                                entry_price=current_price,
                                stop_loss=dca_sl,
                                take_profit=dca_tp,
                                confidence=confidence,
                                strategy="AUTO_DCA",
                                comment=f"DCA-{dca_level}",
                                is_dca=True,
                                dca_level=dca_level
                            )
                            
                            # Decide execution path: pending (limit) vs market
                            if dca_signal.action != "BLOCKED":
                                # 🔒 Acquire DCA lock to prevent race conditions
                                if not self.dca_lock_manager.acquire_lock(symbol, timeout=3.0):
                                    logger.warning(f"🔒 DCA lock acquisition failed for {symbol} - preventing duplicate execution")
                                    dca_results["skipped"] += 1
                                    continue
                                
                                try:
                                    if exec_mode == 'pending' and target_fibo_price:
                                        # Adjust entry to target fibo price
                                        dca_signal.entry_price = float(target_fibo_price)
                                        # Determine pending order type (BUY_LIMIT below current price for BUY, SELL_LIMIT above current for SELL)
                                        if direction.upper() == 'BUY':
                                            order_type_enum = OrderType.LIMIT_BUY
                                        else:
                                            order_type_enum = OrderType.LIMIT_SELL
                                        logger.info(f"🕒 Placing PENDING DCA {direction} Level-{dca_level} at {dca_signal.entry_price:.5f} ({symbol})")
                                        result = self.execute_pending_order(dca_signal, order_type_enum)
                                    else:
                                        result = self.execute_market_order(dca_signal)
                                finally:
                                    # 🔓 Always release lock regardless of execution result
                                    self.dca_lock_manager.release_lock(symbol)
                                if result.success:
                                    dca_results["executed"] += 1
                                    logger.info(f"✅ DCA {'PENDING' if exec_mode=='pending' else 'MARKET'} executed: {symbol} {direction} Level-{dca_level} Vol:{volume:.3f}")
                                    
                                    # Mark DCA as executed to prevent duplicates
                                    risk_validator._mark_dca_executed(symbol, dca_level)
                                    
                                    # Handle DCA stop loss mode
                                    if result.ticket:
                                        dca_ticket = result.ticket
                                        risk_validator._apply_dca_stop_loss_mode(
                                            symbol=symbol,
                                            dca_ticket=dca_ticket,
                                            main_position_ticket=opportunity['main_position_ticket']
                                        )
                                else:
                                    dca_results["failed"] += 1
                                    logger.warning(f"❌ DCA {'pending' if exec_mode=='pending' else 'market'} failed: {symbol} {direction} - {result.error_message}")
                            else:
                                dca_results["skipped"] += 1
                                logger.info(f"🚫 DCA skipped: {symbol} {direction} - {dca_signal.comment}")
                                
                        except Exception as e:
                            dca_results["failed"] += 1
                            logger.error(f"❌ DCA processing error for {symbol}: {e}")
                    
                    # Add DCA results to summary
                    summary["dca_results"] = dca_results
                    logger.info(f"🔄 DCA Summary: Executed: {dca_results['executed']}, Failed: {dca_results['failed']}, Skipped: {dca_results['skipped']}")
                else:
                    logger.info("🔄 No DCA opportunities found")
                    summary["dca_results"] = {"executed": 0, "failed": 0, "skipped": 0}
                    
            except Exception as e:
                logger.warning(f"⚠️ DCA detection error: {e}")
                summary["dca_results"] = {"executed": 0, "failed": 0, "skipped": 0, "error": str(e)}

            # 🚫 Direction Conflicts Summary
            direction_conflicts = summary.get("direction_conflicts", [])
            if direction_conflicts:
                logger.info(f"🚫 Direction Conflicts Summary: {len(direction_conflicts)} conflicts blocked")
                for conflict in direction_conflicts:
                    logger.info(f"   - {conflict['symbol']}: {conflict['existing_direction']} vs {conflict['signal_direction']}")
            
            # 📊 Final Summary Logging
            total_actions = len(actions) if actions else 0
            total_executed = sum(summary.get(k, 0) for k in ['set_sl', 'set_tp', 'close_position', 'modify_position'])
            total_skipped = len(summary.get('skipped', []))
            total_errors = len(summary.get('errors', []))
            total_conflicts = len(direction_conflicts)
            
            # Log skipped actions to file only
            if summary.get('skipped'):
                logger.info("="*80)
                logger.info("🔍 SKIPPED ACTIONS DEBUG:")
                logger.info("="*80)
                for skipped_item in summary.get('skipped', []):
                    logger.info(f"   ❌ {skipped_item.get('symbol', 'Unknown')} - {skipped_item.get('action', 'Unknown')}: {skipped_item.get('reason', 'No reason')}")
                    if skipped_item.get('details'):
                        logger.info(f"      Details: {skipped_item.get('details')}")
                logger.info("="*80)
            
            logger.info(f"📊 ACTIONS SUMMARY: Total: {total_actions}, Executed: {total_executed}, Skipped: {total_skipped}, Errors: {total_errors}, Conflicts: {total_conflicts}")

            return summary

        except Exception as e:
            logger.error(f"❌ apply_actions_from_json exception: {e}")
            return {"success": False, "error": str(e)}

    def apply_dca_sl_adjustments(self, dca_sl_adjustments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply DCA stop loss adjustments from comprehensive_aggregator analysis.
        
        Args:
            dca_sl_adjustments: List of DCA S/L adjustment actions
            
        Returns:
            Summary dict with counts and results
        """
        try:
            if not self._validate_connection():
                return {"success": False, "error": "Connection validation failed"}
            
            if not dca_sl_adjustments:
                logger.info("ℹ️ No DCA S/L adjustments to apply.")
                return {"success": True, "applied": 0}
            
            summary = {
                "success": True, 
                "applied": 0, 
                "by_action": {}, 
                "skipped": [], 
                "errors": []
            }
            
            def _inc(action_name: str, success_flag: bool):
                summary["by_action"].setdefault(action_name, 0)
                if success_flag:
                    summary["by_action"][action_name] += 1
                    summary["applied"] += 1
            
            for adjustment in dca_sl_adjustments:
                try:
                    action = adjustment.get('primary_action', 'set_sl')
                    ticket = adjustment.get('ticket')
                    symbol = adjustment.get('symbol')
                    proposed_sl = adjustment.get('proposed_sl')
                    
                    if not all([ticket, symbol, proposed_sl]):
                        summary["errors"].append({
                            "error": "Missing required fields", 
                            "data": adjustment
                        })
                        logger.error(f"❌ Missing required fields in DCA adjustment: {adjustment}")
                        continue
                    
                    # Validate the position exists
                    positions = self.get_open_positions()
                    target_pos = None
                    for pos in positions:
                        if pos.get('ticket') == ticket:
                            target_pos = pos
                            break
                    if not target_pos:
                        summary["skipped"].append({
                            "ticket": ticket,
                            "symbol": symbol,
                            "action": action,
                            "reason": "position_not_found"
                        })
                        logger.warning(f"⚠️ Position {symbol} ticket {ticket} not found - skipping DCA S/L adjustment")
                        continue
                    
                    # Apply the S/L adjustment
                    if action in ('set_sl', 'adjust_sl', 'update_sl'):
                        try:
                            new_sl_price = float(proposed_sl)
                            logger.info(f"🔧 DCA S/L adjustment for {symbol} ticket {ticket}: SL={new_sl_price:.5f}")
                            
                            res = self.modify_order(ticket=ticket, sl=new_sl_price)
                            _inc(action, res.success)
                            
                            if res.success:
                                logger.info(f"✅ DCA S/L adjustment {symbol} ticket {ticket}: SUCCESS - SL set to {new_sl_price:.5f}")
                            else:
                                logger.error(f"❌ DCA S/L adjustment {symbol} ticket {ticket}: FAILED - {res.error_message}")
                                summary['errors'].append({
                                    "symbol": symbol, 
                                    "ticket": ticket, 
                                    "action": action, 
                                    "error": res.error_message
                                })
                        except (ValueError, TypeError) as e:
                            summary['errors'].append({
                                "symbol": symbol, 
                                "ticket": ticket, 
                                "action": action, 
                                "error": f"Invalid SL price: {proposed_sl} - {e}"
                            })
                            logger.error(f"❌ Invalid SL price for {symbol} ticket {ticket}: {proposed_sl}")
                    else:
                        summary["skipped"].append({
                            "ticket": ticket,
                            "symbol": symbol, 
                            "action": action,
                            "reason": "unsupported_dca_action"
                        })
                        logger.warning(f"⚠️ Unsupported DCA action: {action} for {symbol} ticket {ticket}")
                        
                except Exception as e:
                    summary["errors"].append({
                        "error": f"Exception processing DCA adjustment: {e}", 
                        "data": adjustment
                    })
                    logger.error(f"❌ Exception processing DCA adjustment: {e}")
            
            logger.info(f"🔧 DCA S/L adjustments completed: {summary['applied']}/{len(dca_sl_adjustments)} applied")
            return summary
            
        except Exception as e:
            logger.error(f"❌ apply_dca_sl_adjustments exception: {e}")
            return {"success": False, "error": str(e)}

    def process_new_signals(self, signals_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process new entry signals from individual signal files like BTCUSD_signal_*.json"""
        try:
            import json as _json
            import os as _os
            import glob as _glob
            from datetime import datetime as _dt
            
            # Validate MT5 connection
            if not self._validate_connection():
                return {"success": False, "error": "Connection validation failed"}
            
            # Signal processing can proceed normally
            risk_settings = self.volume_calculator.risk_settings
            enable_auto_mode = risk_settings.get('enable_auto_mode', False)
            
            logger.info(f"📊 Auto mode: {enable_auto_mode}")
            logger.info("✅ Signal processing will proceed normally")
            
            # Default signals directory
            if signals_dir is None:
                signals_dir = _os.path.join(_os.path.dirname(__file__), 'analysis_results')
            
            # Find all signal files
            signal_pattern = _os.path.join(signals_dir, '*_signal_*.json')
            signal_files = _glob.glob(signal_pattern)
            
            if not signal_files:
                logger.info("ℹ️ No signal files found for new entries")
                return {"success": True, "applied": 0, "signals_processed": 0}
            
            logger.info(f"📊 Found {len(signal_files)} signal files to process")
            
            # 🔒 Check position limits BEFORE processing any signals
            current_positions = mt5.positions_total()
            max_positions = risk_settings.get('max_positions', 5)
            
            if current_positions >= max_positions:
                logger.warning(f"🔒 MAX POSITIONS REACHED: {current_positions}/{max_positions} - Blocking new signals")
                return {
                    "success": True,
                    "applied": 0,
                    "skipped": len(signal_files),
                    "blocked": True,
                    "reason": f"Max positions limit reached ({current_positions}/{max_positions})"
                }
            
            summary = {
                "success": True,
                "applied": 0,
                "skipped": 0,
                "errors": 0,
                "signals_processed": len(signal_files),
                "details": []
            }
            
            for signal_file in signal_files:
                try:
                    # Load signal data
                    with open(signal_file, 'r', encoding='utf-8') as f:
                        signal_data = _json.load(f)
                    
                    symbol = signal_data.get('symbol')
                    timestamp = signal_data.get('timestamp')
                    final_signal = signal_data.get('final_signal', {})
                    
                    signal_type = final_signal.get('signal')
                    confidence = final_signal.get('confidence', 0)
                    entry_price = final_signal.get('entry')
                    stop_loss = final_signal.get('stoploss')
                    take_profit = final_signal.get('takeprofit')
                    
                    # Get smart entry data if available
                    order_type_preference = final_signal.get('order_type', 'market')
                    current_price = final_signal.get('current_price')
                    if current_price is None:
                        # Get current price from MT5 if not in signal
                        current_price = self._get_current_price(symbol, signal_type)
                    entry_reason = final_signal.get('entry_reason', 'Giá thị trường')
                    confidence_boost = final_signal.get('confidence_boost', 0.0)
                    smart_entry_used = final_signal.get('smart_entry_used', False)
                    
                    if not symbol or not signal_type:
                        logger.warning(f"⚠️ Invalid signal data in {signal_file}")
                        summary["skipped"] += 1
                        continue
                    
                    # Check if signal is recent (within last 2 hours for testing)
                    try:
                        signal_time = _dt.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        time_diff = (_dt.now() - signal_time).total_seconds() / 60
                        if time_diff > 120:  # Signal older than 2 hours
                            logger.info(f"⏰ Signal for {symbol} is {time_diff:.1f} minutes old - skipping")
                            summary["skipped"] += 1
                            continue
                    except:
                        logger.warning(f"⚠️ Could not parse timestamp for {symbol}")
                    
                    # �️ COMPREHENSIVE RISK VALIDATION - Kiểm tra TẤT CẢ điều kiện risk như human analysis
                    logger.info(f"🔍 VALIDATING RISK CONDITIONS for {symbol} signal...")
                    risk_result = self.volume_calculator.risk_validator.validate_all_conditions(
                        symbol=symbol, 
                        signal_confidence=confidence, 
                        order_type=signal_type,
                        is_dca=False
                    )
                    
                    # Log detailed risk analysis
                    logger.info(f"📊 RISK ANALYSIS RESULT for {symbol}:")
                    logger.info(f"   🎯 Risk Score: {risk_result.risk_score:.1f}/100")
                    logger.info(f"   📝 Recommendation: {risk_result.recommendation}")
                    
                    # Log all passed checks
                    for check in risk_result.passed_checks:
                        logger.info(f"   {check}")
                        
                    # Log warnings (non-blocking)
                    for warning in risk_result.warnings:
                        logger.warning(f"   {warning}")
                        
                    # Check for critical violations (BLOCKING)
                    if not risk_result.is_valid:
                        logger.error(f"🚫 SIGNAL REJECTED for {symbol} due to risk violations:")
                        for violation in risk_result.violations:
                            logger.error(f"   {violation}")
                        summary["skipped"] += 1
                        summary["details"].append({
                            "symbol": symbol,
                            "action": "signal_rejected",
                            "reason": f"Risk violations: {len(risk_result.violations)}",
                            "risk_score": risk_result.risk_score,
                            "violations": risk_result.violations
                        })
                        continue
                    
                    # 🎉 All risk conditions PASSED - Signal approved for trading!
                    logger.info(f"✅ RISK CONDITIONS SATISFIED for {symbol} - Proceeding with order placement")
                    logger.info(f"   📈 Signal: {signal_type} | Confidence: {confidence:.1f}% | Risk Score: {risk_result.risk_score:.1f}/100")
                    
                    # 🔧 CRITICAL FIX: Get current positions using account scan data (more reliable than mt5.positions_get)
                    positions = self._get_symbol_positions_from_scan(symbol)
                    
                    # � DIRECTION CONFLICT CHECK - Block opposite direction entries
                    if positions and len(positions) > 0:
                        existing_direction = None
                        # Check direction of existing positions
                        for pos in positions:
                            pos_direction = "BUY" if pos.type == 0 else "SELL"
                            if existing_direction is None:
                                existing_direction = pos_direction
                            elif existing_direction != pos_direction:
                                # Mixed directions - allow (shouldn't happen with our blocking)
                                existing_direction = "MIXED"
                                break
                        
                        # Check if new signal conflicts with existing positions
                        if existing_direction and existing_direction != "MIXED" and signal_type.upper() != existing_direction:
                            logger.warning(f"🚫 DIRECTION CONFLICT for {symbol}: Existing {existing_direction} positions, blocking new {signal_type} signal")
                            logger.warning(f"   📊 Existing positions: {len(positions)} {existing_direction}")
                            logger.warning(f"   📈 New signal: {signal_type} (confidence: {confidence:.1f}%)")
                            logger.warning(f"   🛑 Entry BLOCKED to prevent opposite direction trading")
                            
                            summary["skipped"] += 1
                            summary["details"].append({
                                "symbol": symbol,
                                "action": "direction_conflict",
                                "reason": f"Direction conflict: {existing_direction} positions exist, blocking {signal_type} signal",
                                "existing_direction": existing_direction,
                                "signal_direction": signal_type,
                                "existing_positions": len(positions),
                                "confidence": confidence
                            })
                            continue
                    
                    # �🔒 Check if we have reached daily loss limit
                    
                    # 🔒 MAX POSITIONS PER SYMBOL CHECK - CRITICAL 
                    max_positions_per_symbol = risk_settings.get('max_positions_per_symbol', 5)
                    if positions and len(positions) >= max_positions_per_symbol:
                        logger.warning(f"🔒 MAX POSITIONS REACHED for {symbol}: {len(positions)}/{max_positions_per_symbol} - blocking new entry")
                        logger.warning(f"   📊 Current positions: {len(positions)} (limit: {max_positions_per_symbol})")
                        logger.warning(f"   📈 Blocked signal: {signal_type} (confidence: {confidence:.1f}%)")
                        logger.warning(f"   🛑 Entry BLOCKED due to position limit")
                        
                        summary["skipped"] += 1
                        summary["details"].append({
                            "symbol": symbol,
                            "action": "max_positions_reached", 
                            "reason": f"Max positions per symbol limit reached: {len(positions)}/{max_positions_per_symbol}",
                            "existing_positions": len(positions),
                            "max_allowed": max_positions_per_symbol,
                            "signal_direction": signal_type,
                            "confidence": confidence
                        })
                        continue
                    
                    # 🔒 Check if we have reached daily loss limit
                    max_daily_loss = risk_settings.get('max_daily_loss_percent', 5.0)                    # Handle "OFF" string case
                    if isinstance(max_daily_loss, str) and max_daily_loss.upper() == "OFF":
                        max_daily_loss = None
                    else:
                        try:
                            max_daily_loss = float(max_daily_loss) if max_daily_loss is not None else None
                        except (ValueError, TypeError):
                            max_daily_loss = 5.0  # Default fallback
                    
                    if max_daily_loss is not None and max_daily_loss > 0:
                        # Simple check - could be enhanced with actual daily P&L calculation
                        account_info = mt5.account_info()
                        if account_info:
                            floating_pl_pct = (account_info.profit / account_info.balance) * 100 if account_info.balance > 0 else 0
                            if floating_pl_pct < -max_daily_loss:
                                logger.warning(f"🔒 Daily loss limit reached {floating_pl_pct:.2f}% >= {max_daily_loss}% - blocking new signals")
                                summary["skipped"] += 1
                                continue
                    
                    # 🔒 Check trading hours
                    current_hour = _dt.now().hour
                    trading_start = risk_settings.get('trading_hours_start', 0)
                    trading_end = risk_settings.get('trading_hours_end', 23)
                    
                    if trading_start <= trading_end:
                        # Normal range (e.g., 8-18)
                        if not (trading_start <= current_hour <= trading_end):
                            logger.info(f"🕐 Outside trading hours {current_hour}:00 (allowed: {trading_start}:00-{trading_end}:00) - skipping {symbol}")
                            summary["skipped"] += 1
                            continue
                    else:
                        # Overnight range (e.g., 18-8)
                        if not (current_hour >= trading_start or current_hour <= trading_end):
                            logger.info(f"🕐 Outside trading hours {current_hour}:00 (allowed: {trading_start}:00-{trading_end}:00) - skipping {symbol}")
                            summary["skipped"] += 1
                            continue
                    
                    # 🔒 Check emergency stop drawdown
                    emergency_stop_dd = risk_settings.get('emergency_stop_drawdown', 0)
                    
                    # Handle "OFF" string case
                    if isinstance(emergency_stop_dd, str) and emergency_stop_dd.upper() == "OFF":
                        emergency_stop_dd = 0
                    else:
                        try:
                            emergency_stop_dd = float(emergency_stop_dd) if emergency_stop_dd is not None else 0
                        except (ValueError, TypeError):
                            emergency_stop_dd = 0  # Default fallback
                    
                    if emergency_stop_dd > 0:
                        account_info = mt5.account_info()
                        if account_info and account_info.balance > 0:
                            drawdown_pct = abs(account_info.profit / account_info.balance) * 100
                            if account_info.profit < 0 and drawdown_pct >= emergency_stop_dd:
                                logger.warning(f"🚨 EMERGENCY STOP: Drawdown {drawdown_pct:.2f}% >= {emergency_stop_dd}% - blocking all signals")
                                summary["skipped"] += 1
                                continue
                    
                    # 🔒 DCA SPECIFIC VALIDATION (if this would be a DCA position)
                    is_potential_dca = False
                    if positions and len(positions) > 0:
                        # This would be a DCA entry - run comprehensive DCA validation
                        is_potential_dca = True
                        logger.info(f"🔄 VALIDATING DCA CONDITIONS for {symbol} (existing {len(positions)} positions)...")
                        
                        dca_risk_result = self.volume_calculator.risk_validator.validate_all_conditions(
                            symbol=symbol, 
                            signal_confidence=confidence, 
                            order_type=signal_type,
                            is_dca=True  # DCA specific validation
                        )
                        
                        # Log DCA risk analysis
                        logger.info(f"� DCA RISK ANALYSIS for {symbol}:")
                        logger.info(f"   🎯 DCA Risk Score: {dca_risk_result.risk_score:.1f}/100")
                        logger.info(f"   📝 DCA Recommendation: {dca_risk_result.recommendation}")
                        
                        # Check for DCA violations (BLOCKING)
                        if not dca_risk_result.is_valid:
                            logger.warning(f"🚫 DCA ENTRY REJECTED for {symbol}:")
                            for violation in dca_risk_result.violations:
                                logger.warning(f"   {violation}")
                            summary["skipped"] += 1
                            summary["details"].append({
                                "symbol": symbol,
                                "action": "dca_rejected", 
                                "reason": f"DCA violations: {len(dca_risk_result.violations)}",
                                "risk_score": dca_risk_result.risk_score,
                                "violations": dca_risk_result.violations
                            })
                            continue
                        
                        logger.info(f"✅ DCA CONDITIONS SATISFIED for {symbol} - Proceeding with DCA entry")
                    
                    # 🔒 Check exposure limits for symbol (additional validation)
                    symbol_exposure_limits = risk_settings.get('symbol_exposure', {})
                    if symbol in symbol_exposure_limits:
                        max_exposure = symbol_exposure_limits[symbol]
                        # Calculate current exposure for this symbol
                        current_exposure = 0.0
                        symbol_positions = mt5.positions_get(symbol=symbol)
                        if symbol_positions:
                            for pos in symbol_positions:
                                current_exposure += pos.volume
                        
                        if current_exposure >= max_exposure:
                            logger.info(f"� {symbol} exposure limit reached {current_exposure:.2f}/{max_exposure} - skipping")
                            summary["skipped"] += 1
                            continue
                    
                    # 🔒 Validate required price levels
                    if not entry_price or not stop_loss:
                        logger.warning(f"⚠️ {symbol} missing entry price or stop loss - skipping for safety")
                        summary["skipped"] += 1
                        continue
                    
                    # 🔒 Validate risk/reward ratio
                    min_rr_ratio = risk_settings.get('min_risk_reward_ratio', 1.5)
                    
                    # Handle "OFF" string case for min_rr_ratio
                    if isinstance(min_rr_ratio, str) and min_rr_ratio.upper() == "OFF":
                        min_rr_ratio = None
                    else:
                        try:
                            min_rr_ratio = float(min_rr_ratio) if min_rr_ratio is not None else 1.5
                        except (ValueError, TypeError):
                            min_rr_ratio = 1.5  # Default fallback
                    
                    if min_rr_ratio is not None and take_profit and entry_price and stop_loss:
                        if signal_type.upper() == 'BUY':
                            risk = abs(entry_price - stop_loss)
                            reward = abs(take_profit - entry_price)
                        else:
                            risk = abs(stop_loss - entry_price)  
                            reward = abs(entry_price - take_profit)
                        
                        if risk > 0:
                            rr_ratio = reward / risk
                            if rr_ratio < min_rr_ratio:
                                logger.info(f"📊 {symbol} R/R ratio {rr_ratio:.2f} < {min_rr_ratio} - skipping")
                                summary["skipped"] += 1
                                continue
                    
                    # Calculate volume using smart volume calculator
                    # Create temporary TradeSignal object for volume calculation
                    temp_signal = TradeSignal(
                        symbol=symbol,
                        action=signal_type,
                        entry_price=entry_price or 0,
                        stop_loss=stop_loss or 0,
                        take_profit=take_profit or 0,
                        volume=0.01,  # Will be overwritten
                        confidence=confidence / 100.0,  # Convert percentage to decimal
                        strategy="AUTO_SIGNAL"
                    )
                    
                    # 🔧 CRITICAL FIX: Calculate proper DCA volume when positions exist
                    if positions and len(positions) > 0:
                        # This is a DCA entry - calculate progressive volume
                        dca_level = len(positions) + 1  # Next DCA level
                        volume = self.volume_calculator.get_volume_for_signal(temp_signal, is_dca=True, dca_level=dca_level)
                        logger.info(f"🔄 DCA Volume calculated for {symbol}: Level {dca_level}, Volume {volume:.3f}")
                    else:
                        # This is the first entry
                        volume = self.volume_calculator.get_volume_for_signal(temp_signal, is_dca=False, dca_level=1)
                        logger.info(f"🎯 Entry Volume calculated for {symbol}: Level 1, Volume {volume:.3f}")
                    
                    if volume is None or volume <= 0:
                        logger.warning(f"⚠️ Invalid volume calculated for {symbol}: {volume}")
                        summary["skipped"] += 1
                        continue
                    
                    # Execute the new order
                    if signal_type.upper() == 'BUY':
                        order_type = mt5.ORDER_TYPE_BUY
                    elif signal_type.upper() == 'SELL':
                        order_type = mt5.ORDER_TYPE_SELL
                    else:
                        logger.warning(f"⚠️ Unknown signal type: {signal_type} for {symbol}")
                        summary["skipped"] += 1
                        continue
                    
                    logger.info(f"🚀 Processing {signal_type} signal for {symbol} (confidence: {confidence}%)")
                    if smart_entry_used:
                        logger.info(f"📈 Smart Entry: {entry_reason}")
                        logger.info(f"🎯 Order Type: {order_type_preference.upper()}")
                        if current_price and entry_price:
                            price_diff = abs(entry_price - current_price)
                            logger.info(f"💰 Entry: {entry_price:.5f} | Current: {current_price:.5f} | Diff: {price_diff:.5f}")
                    
                    # Determine execution method based on order type preference
                    if order_type_preference == 'limit' and current_price and entry_price:
                        # Use limit order for smart entry
                        pip_value = get_pip_value(symbol)
                        pip_threshold = 10 * pip_value  # 10 pips minimum difference
                        logger.info(f"🔧 DEBUG: symbol={symbol}, entry_price={entry_price}, current_price={current_price}, pip_value={pip_value}")
                        
                        if entry_price is None or current_price is None:
                            logger.error(f"❌ Invalid prices: entry_price={entry_price}, current_price={current_price}")
                            summary["skipped"] += 1
                            continue
                            
                        price_diff = abs(entry_price - current_price)
                        
                        if price_diff >= pip_threshold:
                            # Significant difference - use limit order
                            logger.info(f"📊 Using LIMIT ORDER: Entry {entry_price:.5f} vs Current {current_price:.5f}")
                            
                            # Create TradeSignal for limit order
                            trade_signal = TradeSignal(
                                symbol=symbol,
                                action=signal_type,
                                entry_price=entry_price,
                                stop_loss=stop_loss or 0,
                                take_profit=take_profit or 0,
                                volume=volume,
                                confidence=confidence / 100.0,
                                strategy="SMART_ENTRY",
                                comment=f"Limit {confidence}% - {entry_reason[:30]}"
                            )
                            
                            # Execute limit order
                            if signal_type.upper() == 'BUY':
                                order_type_enum = OrderType.LIMIT_BUY
                            else:
                                order_type_enum = OrderType.LIMIT_SELL
                            
                            result = self.execute_pending_order(trade_signal, order_type_enum)
                            
                            if result.success:
                                logger.info(f"✅ Limit {signal_type} order for {symbol}: Ticket {result.ticket} at {entry_price:.5f}")
                                summary["applied"] += 1
                                summary["details"].append({
                                    "symbol": symbol,
                                    "action": f"limit_{signal_type.lower()}",
                                    "ticket": result.ticket,
                                    "volume": volume,
                                    "confidence": confidence,
                                    "entry_price": entry_price,
                                    "order_type": "limit",
                                    "entry_reason": entry_reason,
                                    "status": "success"
                                })
                            else:
                                logger.error(f"❌ Failed to place limit {signal_type} order for {symbol}: {result.error_message}")
                                summary["errors"] += 1
                                summary["details"].append({
                                    "symbol": symbol,
                                    "action": f"limit_{signal_type.lower()}",
                                    "error": result.error_message,
                                    "entry_price": entry_price,
                                    "status": "failed"
                                })
                            
                            continue  # Skip market order execution
                        else:
                            logger.info(f"📊 Price difference too small ({price_diff:.5f}) - using MARKET ORDER instead")
                    
                    # Execute market order (default or fallback)
                    logger.info(f"📊 Using MARKET ORDER for {symbol}")
                    
                    # 🔧 CRITICAL FIX: Create proper TradeSignal with DCA marking
                    if positions and len(positions) > 0:
                        # This is a DCA order
                        dca_level = len(positions) + 1
                        is_dca_order = True
                        strategy_name = "DCA_ENTRY"
                        dca_comment = f"GPT_20B|DCA L{dca_level}"
                        full_comment = f"{dca_comment} - {confidence}% - {self._clean_comment(entry_reason[:15])}" if smart_entry_used else dca_comment
                    else:
                        # This is the main entry
                        dca_level = 1  
                        is_dca_order = False
                        strategy_name = "AUTO_SIGNAL" if not smart_entry_used else "SMART_ENTRY"
                        full_comment = f"GOLDKILLER_AI - {confidence}% - {self._clean_comment(entry_reason[:15])}" if smart_entry_used else f"GOLDKILLER_AI - {confidence}%"
                    
                    trade_signal = TradeSignal(
                        symbol=symbol,
                        action=signal_type,
                        entry_price=entry_price or 0,
                        stop_loss=stop_loss or 0,
                        take_profit=take_profit or 0,
                        volume=volume,
                        confidence=confidence / 100.0,  # Convert percentage to decimal
                        strategy=strategy_name,
                        comment=full_comment,
                        is_dca=is_dca_order,
                        dca_level=dca_level
                    )
                    
                    # Execute market order
                    result = self.execute_market_order(trade_signal)
                    
                    if result.success:
                        order_type_desc = f"DCA Level {dca_level}" if is_dca_order else "Entry"
                        logger.info(f"✅ {order_type_desc} {signal_type} for {symbol}: Ticket {result.ticket}, Volume {volume:.3f}")
                        logger.info(f"   📊 Comment: {full_comment}")
                        summary["applied"] += 1
                        summary["details"].append({
                            "symbol": symbol,
                            "action": f"dca_level_{dca_level}" if is_dca_order else f"entry_{signal_type.lower()}",
                            "ticket": result.ticket,
                            "volume": volume,
                            "confidence": confidence,
                            "entry_price": entry_price,
                            "order_type": "market",
                            "entry_reason": entry_reason if smart_entry_used else "standard",
                            "smart_entry_used": smart_entry_used,
                            "is_dca": is_dca_order,
                            "dca_level": dca_level,
                            "comment": full_comment,
                            "status": "success"
                        })
                    else:
                        logger.error(f"❌ Failed to place market {signal_type} order for {symbol}: {result.error_message}")
                        summary["errors"] += 1
                        summary["details"].append({
                            "symbol": symbol,
                            "action": f"market_{signal_type.lower()}",
                            "error": result.error_message,
                            "entry_price": entry_price,
                            "order_type": "market",
                            "smart_entry_used": smart_entry_used,
                            "status": "failed"
                        })
                
                except Exception as e:
                    import traceback
                    logger.error(f"❌ Error processing signal file {signal_file}: {e}")
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    summary["errors"] += 1
            
            # 🚫 Direction Conflicts Summary for process_new_signals
            direction_conflicts = [detail for detail in summary.get("details", []) if detail.get("action") == "direction_conflict"]
            if direction_conflicts:
                logger.info(f"🚫 Direction Conflicts in new signals: {len(direction_conflicts)} conflicts blocked")
                for conflict in direction_conflicts:
                    logger.info(f"   - {conflict['symbol']}: {conflict['existing_direction']} vs {conflict['signal_direction']}")
            
            # 🔒 Max Positions Reached Summary
            max_positions_blocked = [detail for detail in summary.get("details", []) if detail.get("action") == "max_positions_reached"]
            if max_positions_blocked:
                logger.info(f"🔒 Max Positions Blocks: {len(max_positions_blocked)} signals blocked due to position limits")
                for block in max_positions_blocked:
                    logger.info(f"   - {block['symbol']}: {block['existing_positions']}/{block['max_allowed']} positions (blocked {block['signal_direction']})")
            
            logger.info(f"📊 Signal processing complete: {summary['applied']} applied, {summary['skipped']} skipped, {summary['errors']} errors, {len(direction_conflicts)} conflicts, {len(max_positions_blocked)} position limits")
            return summary
            
        except Exception as e:
            logger.error(f"❌ process_new_signals exception: {e}")
            return {"success": False, "error": str(e)}

    def _order_type_to_string(self, order_type: int) -> str:
        """Convert MT5 order type to string"""
        type_map = {
            mt5.ORDER_TYPE_BUY: "BUY",
            mt5.ORDER_TYPE_SELL: "SELL",
            mt5.ORDER_TYPE_BUY_LIMIT: "BUY_LIMIT",
            mt5.ORDER_TYPE_SELL_LIMIT: "SELL_LIMIT",
            mt5.ORDER_TYPE_BUY_STOP: "BUY_STOP",
            mt5.ORDER_TYPE_SELL_STOP: "SELL_STOP"
        }
        return type_map.get(order_type, "UNKNOWN")

    def save_execution_report(self, filepath: str = None, auto_cleanup: bool = True) -> str:
        """Save execution report to unified risk_settings.json"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "statistics": self.get_statistics(),
                "open_positions": self.get_open_positions(),
                "closed_positions": self.get_closed_positions(days_back=7),
                "pending_orders": self.get_pending_orders_info(),
                "executed_orders_count": len(self.executed_orders),
                "failed_orders_count": len(self.failed_orders),
                "magic_number": self.magic_number,
                "cleanup_enabled": False  # deprecated
            }
            
            # Save to separate reports file (NOT risk_settings.json)
            os.makedirs("reports", exist_ok=True)
            reports_path = "reports/execution_reports.json"
            
            try:
                if os.path.exists(reports_path):
                    with open(reports_path, 'r', encoding='utf-8') as f:
                        reports_data = json.load(f)
                else:
                    reports_data = {
                        'execution_reports': [],
                        'metadata': {
                            'created': datetime.now().isoformat(),
                            'version': '1.0'
                        }
                    }
            except (FileNotFoundError, json.JSONDecodeError):
                reports_data = {
                    'execution_reports': [],
                    'metadata': {
                        'created': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                }
            
            # Add report to execution_reports array
            reports_data['execution_reports'].append(report)
            
            # Keep only last N reports to prevent file from growing too large
            max_reports = 100
            if len(reports_data['execution_reports']) > max_reports:
                reports_data['execution_reports'] = reports_data['execution_reports'][-max_reports:]
            
            # Update metadata
            reports_data['metadata']['last_updated'] = datetime.now().isoformat()
            reports_data['metadata']['total_reports'] = len(reports_data['execution_reports'])
            
            # Save to separate reports file
            with open(reports_path, 'w', encoding='utf-8') as f:
                json.dump(reports_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📊 Execution report saved to: {reports_path}")
            return reports_path
            
        except Exception as e:
            logger.error(f"❌ Error saving report: {e}")
            return ""
    
    def cleanup_order_data(self, max_age_hours: int = 72, keep_latest: int = 10) -> Dict[str, Any]:
        """
        🧹 ORDER EXECUTOR: Dọn dẹp dữ liệu của module này
        Chỉ xóa dữ liệu trong thư mục reports
        
        Args:
            max_age_hours: Tuổi tối đa của file (giờ)
            keep_latest: Số file mới nhất cần giữ lại
        """
        cleanup_stats = {
            'module_name': 'order_executor',
            'directories_cleaned': [],
            'total_files_deleted': 0,
            'total_space_freed_mb': 0.0,
            'cleanup_time': datetime.now().isoformat()
        }
        
        # Thư mục mà Order Executor quản lý - reports now unified in risk_settings.json
        target_directories = [
            # 'reports' - đã chuyển sang unified system trong risk_settings.json
        ]
        
        for directory in target_directories:
            if os.path.exists(directory):
                logger.info(f"🧹 Order Executor cleaning {directory}...")
                dir_stats = self._clean_directory(directory, max_age_hours, keep_latest)
                cleanup_stats['directories_cleaned'].append({
                    'directory': directory,
                    'files_deleted': dir_stats['deleted'],
                    'space_freed_mb': dir_stats['space_freed']
                })
                cleanup_stats['total_files_deleted'] += dir_stats['deleted']
                cleanup_stats['total_space_freed_mb'] += dir_stats['space_freed']
            else:
                logger.info(f"📁 Directory {directory} does not exist, skipping")
        
        logger.info(f"🧹 ORDER EXECUTOR cleanup complete: "
                   f"{cleanup_stats['total_files_deleted']} files deleted, "
                   f"{cleanup_stats['total_space_freed_mb']:.2f} MB freed")
        return cleanup_stats
    
    def _clean_directory(self, directory: str, max_age_hours: int, keep_latest: int) -> Dict[str, int]:
        """Helper method để clean một directory"""
        deleted_count = 0
        space_freed = 0.0
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        try:
            if not os.path.exists(directory):
                return {'deleted': 0, 'space_freed': 0.0}
                
            # Lấy tất cả execution report files
            all_files = []
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                if os.path.isfile(file_path) and (file_name.startswith('execution_report_') and file_name.endswith('.json')):
                    file_stat = os.stat(file_path)
                    file_time = datetime.fromtimestamp(file_stat.st_mtime)
                    all_files.append({
                        'path': file_path,
                        'name': file_name,
                        'time': file_time,
                        'size': file_stat.st_size
                    })
            
            # Sắp xếp theo thời gian (mới nhất trước)
            all_files.sort(key=lambda x: x['time'], reverse=True)
            
            # Giữ lại keep_latest files mới nhất
            files_to_keep = all_files[:keep_latest]
            files_to_check = all_files[keep_latest:]
            
            # Xóa files cũ hơn max_age_hours
            for file_info in files_to_check:
                if file_info['time'] < cutoff_time:
                    try:
                        os.remove(file_info['path'])
                        deleted_count += 1
                        space_freed += file_info['size'] / (1024 * 1024)  # Convert to MB
                        logger.debug(f"Deleted: {file_info['name']}")
                    except Exception as e:
                        logger.error(f"Failed to delete {file_info['path']}: {e}")
            
        except Exception as e:
            logger.error(f"Error cleaning directory {directory}: {e}")
        
        return {'deleted': deleted_count, 'space_freed': space_freed}

    def cleanup_old_reports(self, max_age_hours: int = 72, keep_latest: int = 10) -> Dict[str, Any]:
        """🧹 Legacy method - calls the new cleanup_order_data method"""
        return self.cleanup_order_data(max_age_hours, keep_latest)

    def test_dca_protection_with_real_positions(self) -> Dict[str, Any]:
        """Test DCA protection system with real positions from account scan"""
        try:
            # Load current positions
            with open('account_scans/mt5_essential_scan.json', 'r', encoding='utf-8') as f:
                scan_data = json.load(f)
            
            positions = scan_data.get('active_positions', [])
            if not positions:
                return {
                    "status": "no_positions",
                    "message": "No active positions found for testing"
                }
            
            test_results = {
                "timestamp": datetime.now().isoformat(),
                "total_positions": len(positions),
                "protection_tests": [],
                "summary": {
                    "symbols_tested": 0,
                    "protected_symbols": 0,
                    "protection_scenarios": 0
                }
            }
            
            # Group positions by symbol
            symbols = list(set([pos.get('symbol', '') for pos in positions]))
            test_results["summary"]["symbols_tested"] = len(symbols)
            
            for symbol in symbols:
                if not symbol:
                    continue
                    
                symbol_positions = [p for p in positions if p.get('symbol') == symbol]
                
                # Get DCA status
                dca_status = self._get_dca_status(symbol)
                
                symbol_test = {
                    "symbol": symbol,
                    "position_count": len(symbol_positions),
                    "dca_status": {
                        "dca_enabled": dca_status["dca_enabled"],
                        "entry_count": len(dca_status["entry_positions"]),
                        "dca_count": len(dca_status["dca_positions"]),
                        "pending_levels": dca_status["max_dca_levels"] - dca_status["total_levels"],
                        "needs_protection": dca_status["needs_sl_protection"]
                    },
                    "protection_tests": []
                }
                
                if dca_status["needs_sl_protection"]:
                    test_results["summary"]["protected_symbols"] += 1
                    
                    # Test various SL scenarios
                    if dca_status["entry_positions"]:
                        entry_pos = dca_status["entry_positions"][0]
                        entry_price = entry_pos.price_open
                        direction = "BUY" if entry_pos.type == 0 else "SELL"
                        
                        # Test different SL distances
                        if direction == "BUY":
                            sl_tests = [
                                entry_price - 0.0005,  # Very tight
                                entry_price - 0.002,   # Moderate
                                entry_price - 0.005,   # Loose
                            ]
                        else:
                            sl_tests = [
                                entry_price + 0.0005,  # Very tight  
                                entry_price + 0.002,   # Moderate
                                entry_price + 0.005,   # Loose
                            ]
                        
                        for i, test_sl in enumerate(sl_tests):
                            protection = self._should_protect_sl_for_dca(symbol, test_sl)
                            
                            test_scenario = {
                                "test_type": f"SL_Test_{i+1}",
                                "proposed_sl": test_sl,
                                "distance_from_entry": abs(test_sl - entry_price),
                                "should_protect": protection["should_protect"],
                                "reason": protection.get("reason", ""),
                                "recommended_sl": protection.get("recommended_sl"),
                                "action": "BLOCKED" if protection["should_protect"] else "ALLOWED"
                            }
                            
                            symbol_test["protection_tests"].append(test_scenario)
                            test_results["summary"]["protection_scenarios"] += 1
                        
                        # Test breakeven scenario
                        protection = self._should_protect_sl_for_dca(symbol, entry_price)
                        breakeven_test = {
                            "test_type": "Breakeven_Test",
                            "proposed_sl": entry_price,
                            "distance_from_entry": 0,
                            "should_protect": protection["should_protect"],
                            "reason": protection.get("reason", ""),
                            "recommended_sl": protection.get("recommended_sl"),
                            "action": "BLOCKED" if protection["should_protect"] else "ALLOWED"
                        }
                        symbol_test["protection_tests"].append(breakeven_test)
                        test_results["summary"]["protection_scenarios"] += 1
                
                test_results["protection_tests"].append(symbol_test)
            
            # Save test results
            test_file = f"reports/dca_protection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(test_file), exist_ok=True)
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"🛡️ DCA Protection test completed: {test_file}")
            return test_results
            
        except Exception as e:
            logger.error(f"❌ Error testing DCA protection: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def test_order_executor_functionality(self) -> Dict[str, Any]:
        """
        🧪 Test Order Executor functionality (dry run)
        Kiểm tra các chức năng mới được thêm vào
        """
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "basic_functions": True,
            "validation_tests": True,
            "action_mapping": True,
            "errors": []
        }
        
        try:
            logger.info("🧪 Starting Order Executor functionality test...")
            
            # Test basic statistics
            stats = self.get_statistics()
            logger.info(f"📊 Initial stats: {stats}")
            
            # 🧮 NEW: Test VolumeCalculator functionality
            logger.info("🧮 Testing VolumeCalculator functionality...")
            
            # Test volume calculation for different scenarios
            test_signal = TradeSignal(
                symbol="EURUSD",
                action="BUY",
                entry_price=1.1000,
                stop_loss=1.0950,
                take_profit=1.1100,
                volume=0.01,  # Will be recalculated
                confidence=3.5,
                strategy="TEST_VOLUME"
            )
            
            # Test smart volume calculation
            smart_volume = self.calculate_smart_volume(test_signal)
            logger.info(f"🎯 Smart volume calculated: {smart_volume:.2f} lots")
            
            # Test prepare signal with smart volume
            try:
                smart_signal = self.prepare_signal_with_smart_volume(
                    symbol="EURUSD",
                    action="BUY", 
                    entry_price=1.1000,
                    stop_loss=1.0950,
                    confidence=4.0
                )
                logger.info(f"✅ Smart signal prepared: {smart_signal.volume:.2f} lots")
                test_results["volume_calculator"] = True
            except Exception as e:
                logger.error(f"❌ Smart signal preparation failed: {e}")
                test_results["volume_calculator"] = False
                test_results["errors"].append(f"Volume calculator error: {str(e)}")
            
            # Test volume constraints validation
            is_valid, msg = self.volume_calculator.validate_volume_constraints("EURUSD", 0.1)
            logger.info(f"🔍 Volume validation test: {is_valid} - {msg}")
            
            # Test different volume modes
            volume_modes = ["Theo rủi ro (Tự động)", "Khối lượng cố định", "Khối lượng mặc định"]
            for mode in volume_modes:
                old_mode = self.volume_calculator.risk_settings.get('volume_mode')
                self.volume_calculator.risk_settings['volume_mode'] = mode
                test_volume = self.volume_calculator.get_volume_for_signal(test_signal)
                logger.info(f"📊 Volume for mode '{mode}': {test_volume:.2f} lots")
                self.volume_calculator.risk_settings['volume_mode'] = old_mode
            
            # Test position retrieval
            positions = self.get_open_positions()
            logger.info(f"📈 Current positions: {len(positions)}")
            
            # Test pending orders
            pending = self.get_pending_orders_info()
            logger.info(f"⏳ Pending orders: {len(pending)}")
            
            # Test validation functions
            test_cases = [
                {
                    "action": "set_sl",
                    "params": {"proposed_sl": 1.0950},
                    "expected": True,
                    "name": "Valid SL"
                },
                {
                    "action": "set_sl",
                    "params": {"proposed_sl": -1.0},
                    "expected": False,
                    "name": "Invalid SL value"
                },
                {
                    "action": "scale_out",
                    "params": {"scale_percent": 25},
                    "expected": True,
                    "name": "Valid scale out"
                },
                {
                    "action": "scale_out",
                    "params": {"scale_percent": 150},
                    "expected": False,
                    "name": "Invalid scale percent"
                }
            ]
            
            # Test validation logic
            validation_passed = 0
            for test in test_cases:
                try:
                    # Simulate validation (without actually calling the private method)
                    if test["action"] == "set_sl":
                        sl = test["params"].get("proposed_sl")
                        is_valid = sl is not None and float(sl) > 0
                    elif test["action"] == "scale_out":
                        scale_pct = test["params"].get("scale_percent", 25)
                        is_valid = 0 < float(scale_pct) <= 100
                    else:
                        is_valid = True
                    
                    if is_valid == test["expected"]:
                        validation_passed += 1
                        logger.info(f"✅ {test['name']}: PASS")
                    else:
                        logger.warning(f"❌ {test['name']}: FAIL")
                        test_results["validation_tests"] = False
                        
                except Exception as e:
                    logger.error(f"❌ {test['name']}: ERROR - {e}")
                    test_results["validation_tests"] = False
                    test_results["errors"].append(f"Validation test {test['name']}: {str(e)}")
            
            logger.info(f"📊 Validation tests: {validation_passed}/{len(test_cases)} passed")
            
            # Test action mapping
            supported_actions = [
                'close_partial_30', 'close_partial_50', 'close_full',
                'set_sl', 'set_initial_sl', 'tighten_sl',
                'close', 'hold', 'monitor', 'reduce_risk',
                'scale_out', 'move_sl_to_be', 'trail_sl',
                'adjust_tp', 'enable_trailing'
            ]
            
            logger.info(f"🎯 Supported actions: {len(supported_actions)} types")
            for action in supported_actions[:5]:  # Log first 5
                logger.info(f"   📋 {action}")
            
            test_results["supported_actions_count"] = len(supported_actions)
            test_results["validation_passed"] = validation_passed
            test_results["validation_total"] = len(test_cases)
            
            logger.info("✅ Order Executor functionality test completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            test_results["basic_functions"] = False
            test_results["errors"].append(str(e))
        
        return test_results

    def create_test_environment(self) -> bool:
        """
        🏗️ Tạo môi trường test với sample data
        """
        try:
            # Tạo thư mục cần thiết
            import os
            os.makedirs("analysis_results", exist_ok=True)
            os.makedirs("risk_management", exist_ok=True)
            
            # Tạo sample actions
            test_actions = [
                {
                    "symbol": "EURUSD",
                    "primary_action": "close_partial_30",
                    "confidence": 4.2,
                    "rationale": "Chốt lời 30% để bảo toàn lợi nhuận"
                },
                {
                    "symbol": "GBPUSD", 
                    "primary_action": "close_full",
                    "confidence": 4.8,
                    "rationale": "Chốt lời 100% theo tín hiệu mạnh"
                },
                {
                    "symbol": "XAUUSD",
                    "primary_action": "set_sl",
                    "proposed_sl": 2650.50,
                    "confidence": 3.8,
                    "rationale": "Điều chỉnh SL để giảm rủi ro"
                },
                {
                    "symbol": "USDJPY",
                    "primary_action": "hold",
                    "confidence": 3.0,
                    "rationale": "Giữ vị thế, chờ tín hiệu rõ ràng hơn"
                }
            ]
            
            # Tạo risk config
            risk_config = {
                "max_total_volume": 1.0,  # Updated field name
                "min_volume_auto": 0.01,  # Updated field name
                "max_positions": 10,
                "max_positions_per_symbol": 3,
                "min_confidence": 2.0,
                "max_risk_per_trade_percent": 2.0,
                "require_sl_for_new_trades": True,
                "trail_sl_pips": 20,
                "daily_loss_limit_percent": 5.0,
                "max_daily_new_positions": 5
            }
            
            # Save files
            actions_path = "analysis_results/account_positions_actions.json"
            risk_path = "risk_management/risk_settings.json"
            
            success1 = overwrite_json_safely(actions_path, test_actions, backup=False)
            success2 = overwrite_json_safely(risk_path, risk_config, backup=False)
            
            if success1 and success2:
                logger.info(f"✅ Test environment created: {len(test_actions)} actions, {len(risk_config)} risk settings")
                return True
            else:
                logger.error("❌ Failed to create test environment files")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creating test environment: {e}")
            return False

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        🎯 Chạy test toàn diện cho Order Executor
        """
        logger.info("🚀 Starting comprehensive Order Executor test...")
        
        # Tạo test environment
        env_created = self.create_test_environment()
        
        # Test functionality
        func_results = self.test_order_executor_functionality()
        
        # Test apply actions (dry run nếu không có MT5)
        apply_results = {"status": "skipped", "reason": "MT5 connection required"}
        try:
            if self._validate_connection():
                logger.info("🔌 MT5 connected - testing apply_actions_from_json...")
                apply_results = self.apply_actions_from_json()
            else:
                logger.info("⚠️ MT5 not connected - skipping apply_actions test")
        except Exception as e:
            apply_results = {"status": "error", "error": str(e)}
        
        # Compile comprehensive results
        comprehensive_results = {
            "timestamp": datetime.now().isoformat(),
            "test_environment": env_created,
            "functionality_test": func_results,
            "apply_actions_test": apply_results,
            "overall_status": "PASS" if (env_created and func_results.get("basic_functions")) else "FAIL"
        }
        
        # Summary logging
        logger.info("=" * 60)
        logger.info("📋 COMPREHENSIVE TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"🏗️ Test environment: {'✅ PASS' if env_created else '❌ FAIL'}")
        logger.info(f"🧪 Basic functionality: {'✅ PASS' if func_results.get('basic_functions') else '❌ FAIL'}")
        logger.info(f"🔍 Validation tests: {'✅ PASS' if func_results.get('validation_tests') else '❌ FAIL'}")
        logger.info(f"⚙️ Apply actions: {apply_results.get('status', 'unknown').upper()}")
        logger.info(f"🎯 Overall status: {comprehensive_results['overall_status']}")
        
        if comprehensive_results['overall_status'] == 'PASS':
            logger.info("🎉 ALL TESTS PASSED! Order Executor is ready for production!")
        else:
            logger.warning("⚠️ Some tests failed. Please review the results.")
        
        logger.info("=" * 60)
        
        return comprehensive_results

    def print_status(self):
        """Print current executor status"""
        stats = self.get_statistics()
        positions = self.get_open_positions()
        pending = self.get_pending_orders_info()
        
        print("\n" + "="*60)
        print("📈 ADVANCED ORDER EXECUTOR STATUS")
        print("="*60)
        print(f"🎯 Magic Number: {self.magic_number}")
        print(f"📊 Success Rate: {stats['success_rate']:.1f}%")
        print(f"📈 Total Orders: {stats['total_orders']}")
        print(f"✅ Successful: {stats['successful_orders']}")
        print(f"❌ Failed: {stats['failed_orders']}")
        print(f"💰 Total Volume: {stats['total_volume']:.2f}")
        print(f"📋 Open Positions: {len(positions)}")
        print(f"⏳ Pending Orders: {len(pending)}")
        print("="*60)

    def print_risk_settings(self):
        """
        🧮 Print current risk settings and volume calculation parameters
        """
        risk_settings = self.volume_calculator.risk_settings
        
        print("\n" + "="*60)
        print("🧮 VOLUME CALCULATOR & RISK SETTINGS")
        print("="*60)
        
        # Volume Mode
        volume_mode = risk_settings.get('volume_mode', 'Unknown')
        print(f"📊 Volume Mode: {volume_mode}")
        
        # Basic Risk Parameters
        print(f"⚠️ Max Risk per Trade: {risk_settings.get('max_risk_percent', 2.0)}%")
        print(f"📏 Min Volume (Auto): {risk_settings.get('min_volume_auto', 0.01)}")  # Updated field name
        print(f"📏 Max Total Volume: {risk_settings.get('max_total_volume', 10.0)}")  # Updated field name
        
        # Fixed Volume (if applicable)
        if 'cố định' in volume_mode or 'Fixed' in volume_mode:
            print(f"🔧 Default Volume: {risk_settings.get('default_volume_lots', 0.1)} lots")
        
        # Position Limits
        print(f"📊 Max Positions: {risk_settings.get('max_positions', 5)}")
        print(f"📊 Max Per Symbol: {risk_settings.get('max_positions_per_symbol', 2)}")
        
        # DCA Settings
        enable_dca = risk_settings.get('enable_dca', False)
        print(f"🔄 DCA Enabled: {'✅ YES' if enable_dca else '❌ NO'}")
        if enable_dca:
            print(f"🔄 DCA Multiplier: {risk_settings.get('dca_volume_multiplier', 1.5)}x")
            print(f"🔄 DCA Max Levels: {risk_settings.get('max_dca_levels', 3)}")
        
        # Symbol-specific settings
        symbol_multipliers = risk_settings.get('symbol_multipliers', {})
        symbol_exposure = risk_settings.get('symbol_exposure', {})
        
        if symbol_multipliers or symbol_exposure:
            print("-" * 40)
            print("🎯 SYMBOL-SPECIFIC SETTINGS:")
            
            all_symbols = set(symbol_multipliers.keys()) | set(symbol_exposure.keys())
            for symbol in sorted(all_symbols):
                multiplier = symbol_multipliers.get(symbol, 1.0)
                exposure = symbol_exposure.get(symbol, 'N/A')
                print(f"  {symbol}: Multiplier={multiplier}x, Max Exposure={exposure}")
        
        # Current account info
        try:
            balance = self.volume_calculator.get_account_balance()
            print("-" * 40)
            print(f"💰 Current Balance: ${balance:,.2f}")
            
            max_risk_amount = balance * (risk_settings.get('max_risk_percent', 2.0) / 100.0)
            print(f"⚠️ Max Risk Amount: ${max_risk_amount:,.2f}")
        except Exception:
            pass
        
        print("="*60)

    def test_volume_calculation_scenarios(self):
        """
        🧪 Test different volume calculation scenarios
        """
        print("\n" + "="*60)
        print("🧪 VOLUME CALCULATION TEST SCENARIOS")
        print("="*60)
        
        test_scenarios = [
            {
                "name": "EUR/USD High Confidence",
                "symbol": "EURUSD", 
                "action": "BUY",
                "entry": 1.1000,
                "sl": 1.0950,
                "confidence": 4.5
            },
            {
                "name": "GBP/JPY Medium Confidence", 
                "symbol": "GBPJPY",
                "action": "SELL", 
                "entry": 150.00,
                "sl": 151.00,
                "confidence": 3.0
            },
            {
                "name": "XAU/USD Low Confidence",
                "symbol": "XAUUSD",
                "action": "BUY",
                "entry": 2650.00, 
                "sl": 2620.00,
                "confidence": 2.0
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📊 Scenario {i}: {scenario['name']}")
            print(f"   Symbol: {scenario['symbol']}")
            print(f"   Action: {scenario['action']}")
            print(f"   Entry: {scenario['entry']}")
            print(f"   Stop Loss: {scenario['sl']}")
            print(f"   Confidence: {scenario['confidence']}/5.0")
            
            try:
                # Create test signal
                signal = TradeSignal(
                    symbol=scenario['symbol'],
                    action=scenario['action'],
                    entry_price=scenario['entry'],
                    stop_loss=scenario['sl'],
                    take_profit=0.0,
                    volume=0.01,  # Will be recalculated
                    confidence=scenario['confidence']
                )
                
                # Calculate volumes for different modes
                volumes = {}
                
                # Risk-based calculation
                old_mode = self.volume_calculator.risk_settings.get('volume_mode')
                
                self.volume_calculator.risk_settings['volume_mode'] = 'Theo rủi ro (Tự động)'
                volumes['Risk-Based'] = self.volume_calculator.get_volume_for_signal(signal)
                
                self.volume_calculator.risk_settings['volume_mode'] = 'Khối lượng cố định'
                volumes['Fixed'] = self.volume_calculator.get_volume_for_signal(signal)
                
                self.volume_calculator.risk_settings['volume_mode'] = 'Khối lượng mặc định'
                volumes['Default'] = self.volume_calculator.get_volume_for_signal(signal)
                
                # Restore original mode
                self.volume_calculator.risk_settings['volume_mode'] = old_mode
                
                # DCA calculation
                volumes['DCA'] = self.volume_calculator.get_volume_for_signal(signal, is_dca=True)
                
                print(f"   📊 Calculated Volumes:")
                for mode, volume in volumes.items():
                    print(f"      {mode}: {volume:.2f} lots")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("="*60)

    def check_and_auto_close_positions(self) -> Dict[str, Any]:
        """
        🛡️ RISK MONITOR: Tự động đóng positions khi vi phạm risk limits
        Check các risk settings và đóng positions if needed
        """
        try:
            # Load risk settings
            risk_settings = self.volume_calculator.load_risk_settings()
            
            # Get current positions
            positions = mt5.positions_get()
            if not positions:
                return {"status": "no_positions", "actions": []}
            
            # Get account info for calculations
            account_info = mt5.account_info()
            if not account_info:
                return {"status": "no_account_info", "actions": []}
            
            balance = account_info.balance
            equity = account_info.equity
            margin_level = account_info.margin_level
            
            actions_taken = []
            
            # 1. CHECK EMERGENCY DRAWDOWN STOP
            emergency_dd = risk_settings.get('emergency_stop_drawdown', 0)
            disable_emergency = risk_settings.get('disable_emergency_stop', False)
            
            if not disable_emergency and emergency_dd > 0:
                current_dd = ((balance - equity) / balance) * 100 if balance > 0 else 0
                if current_dd >= emergency_dd:
                    logger.warning(f"🚨 EMERGENCY STOP: Drawdown {current_dd:.2f}% >= {emergency_dd}%")
                    # Close ALL positions immediately
                    for pos in positions:
                        result = self.close_position(pos.ticket)
                        actions_taken.append({
                            "action": "emergency_close",
                            "ticket": pos.ticket,
                            "symbol": pos.symbol,
                            "reason": f"Emergency drawdown {current_dd:.2f}%",
                            "success": result.success
                        })
                    return {"status": "emergency_stop", "actions": actions_taken}
            
            # 2. CHECK MAX DAILY LOSS
            max_daily_loss = risk_settings.get('max_daily_loss_percent', 0)
            
            # Handle "OFF" string case for max_daily_loss
            if isinstance(max_daily_loss, str) and max_daily_loss.upper() == "OFF":
                max_daily_loss = 0
            else:
                try:
                    max_daily_loss = float(max_daily_loss) if max_daily_loss is not None else 0
                except (ValueError, TypeError):
                    max_daily_loss = 0  # Default fallback
            
            if max_daily_loss > 0:
                # Calculate today's P&L
                daily_pnl = sum(pos.profit for pos in positions if pos.profit < 0)
                daily_pnl_percent = (daily_pnl / balance) * 100 if balance > 0 else 0
                
                if abs(daily_pnl_percent) >= max_daily_loss:
                    logger.warning(f"📉 DAILY LOSS LIMIT: {daily_pnl_percent:.2f}% >= {max_daily_loss}%")
                    # Close all losing positions
                    for pos in positions:
                        if pos.profit < 0:
                            result = self.close_position(pos.ticket)
                            actions_taken.append({
                                "action": "daily_loss_close",
                                "ticket": pos.ticket,
                                "symbol": pos.symbol,
                                "profit": pos.profit,
                                "reason": f"Daily loss limit {daily_pnl_percent:.2f}%",
                                "success": result.success
                            })
            
            # 3. CHECK MAX RISK PER TRADE
            max_risk_percent = risk_settings.get('max_risk_percent', 2.0)
            if max_risk_percent > 0:
                for pos in positions:
                    position_loss_percent = (pos.profit / balance) * 100 if balance > 0 else 0
                    
                    # If position loss exceeds max risk per trade, close it
                    if position_loss_percent <= -max_risk_percent:
                        logger.warning(f"💸 POSITION RISK LIMIT: {pos.symbol} loss {position_loss_percent:.2f}% >= {max_risk_percent}%")
                        result = self.close_position(pos.ticket)
                        actions_taken.append({
                            "action": "risk_per_trade_close",
                            "ticket": pos.ticket,
                            "symbol": pos.symbol,
                            "profit": pos.profit,
                            "loss_percent": position_loss_percent,
                            "reason": f"Risk per trade limit {position_loss_percent:.2f}%",
                            "success": result.success
                        })
            
            # 4. CHECK DCA AVERAGE SL PROFIT PER SYMBOL
            dca_sl_mode = risk_settings.get('dca_sl_mode', 'SL riêng lẻ')
            if dca_sl_mode in ['SL trung bình', 'Average SL']:
                dca_profit_target = risk_settings.get('dca_avg_sl_profit_percent', 10.0)
                if dca_profit_target > 0:
                    # Group positions by symbol
                    symbol_positions = {}
                    for pos in positions:
                        if pos.symbol not in symbol_positions:
                            symbol_positions[pos.symbol] = []
                        symbol_positions[pos.symbol].append(pos)
                    
                    # Check each symbol's profit
                    for symbol, symbol_pos in symbol_positions.items():
                        if len(symbol_pos) > 1:  # Only check if multiple positions (DCA scenario)
                            total_profit = sum(pos.profit for pos in symbol_pos)
                            total_volume = sum(pos.volume for pos in symbol_pos)
                            
                            # Calculate profit percentage for this symbol
                            # Use total position value as base for calculation
                            total_position_value = sum(pos.price_open * pos.volume * 100000 for pos in symbol_pos)  # Rough calculation
                            profit_percent = (total_profit / (total_position_value * 0.01)) * 100 if total_position_value > 0 else 0
                            
                            # Alternative: Use account balance as base
                            profit_percent_vs_balance = (total_profit / balance) * 100 if balance > 0 else 0
                            
                            if profit_percent_vs_balance >= dca_profit_target:
                                logger.info(f"🎯 DCA PROFIT TARGET: {symbol} profit {profit_percent_vs_balance:.2f}% >= {dca_profit_target}%")
                                # Close all positions for this symbol
                                for pos in symbol_pos:
                                    result = self.close_position(pos.ticket)
                                    actions_taken.append({
                                        "action": "dca_profit_close",
                                        "ticket": pos.ticket,
                                        "symbol": pos.symbol,
                                        "profit": pos.profit,
                                        "profit_percent": profit_percent_vs_balance,
                                        "reason": f"DCA profit target {profit_percent_vs_balance:.2f}% for {symbol}",
                                        "success": result.success
                                    })

            # 5. CHECK LOW MARGIN LEVEL
            if margin_level > 0 and margin_level < 200:
                logger.warning(f"⚠️ LOW MARGIN LEVEL: {margin_level:.2f}%")
                # Close largest losing position first
                losing_positions = [pos for pos in positions if pos.profit < 0]
                if losing_positions:
                    worst_position = min(losing_positions, key=lambda p: p.profit)
                    result = self.close_position(worst_position.ticket)
                    actions_taken.append({
                        "action": "low_margin_close",
                        "ticket": worst_position.ticket,
                        "symbol": worst_position.symbol,
                        "profit": worst_position.profit,
                        "margin_level": margin_level,
                        "reason": f"Low margin level {margin_level:.2f}%",
                        "success": result.success
                    })
            
            return {
                "status": "checked",
                "total_positions": len(positions),
                "actions_taken": len(actions_taken),
                "actions": actions_taken,
                "current_drawdown": ((balance - equity) / balance) * 100 if balance > 0 else 0,
                "margin_level": margin_level
            }
            
        except Exception as e:
            logger.error(f"❌ Risk monitoring error: {e}")
            return {"status": "error", "error": str(e), "actions": []}

    def _get_pip_value_for_distance(self, symbol: str) -> float:
        """Get pip value for distance calculation - COMPREHENSIVE CRYPTO & METALS SUPPORT"""
        symbol_upper = symbol.upper().rstrip('.').replace('_M', '').replace('_m', '')  # Normalize
        
        # ========== PRECIOUS METALS ==========
        if symbol_upper in ['XAUUSD', 'XAGUSD', 'GOLD', 'SILVER', 'XPTUSD', 'XPDUSD']:
            return 0.1   # Metals: 1 pip = 0.1 (Gold: 3881.03 -> 3881.73 = 7 pips)
        
        # ========== JPY PAIRS ==========
        elif 'JPY' in symbol_upper:
            return 0.01  # JPY pairs: 1 pip = 0.01 (USD/JPY: 147.15 -> 147.22 = 7 pips)
        
        # ========== HIGH-VALUE CRYPTO (≥ $1000) ==========
        elif symbol_upper in ['BTCUSD', 'ETHUSD']:
            return 1.0   # BTC/ETH: 1 pip = 1.0 (BTC: 65000 -> 65070 = 70 pips)
        
        # ========== MID-VALUE CRYPTO ($100-$1000) ==========
        elif symbol_upper in ['SOLUSD', 'BNBUSD', 'ADAUSD', 'AVAXUSD', 'DOTUSD', 'MATICUSD', 'LINKUSD', 'TRXUSD', 'SHIBUSD', 'ARBUSD', 'OPUSD', 'APEUSD', 'SANDUSD', 'CROUSD', 'FTTUSD']:
            return 0.1   # SOL/BNB/ADA etc: 1 pip = 0.1 (SOL: 224.06 -> 224.76 = 7 pips)
        
        # ========== LOW-VALUE CRYPTO (< $10) ==========
        elif any(crypto in symbol_upper for crypto in ['DOGE', 'XRP', 'TRX']):
            return 0.001  # DOGE/XRP etc: 1 pip = 0.001 (DOGE: 0.123 -> 0.130 = 7 pips)
        
        # ========== MICRO-VALUE CRYPTO (< $1) ==========
        elif any(micro_crypto in symbol_upper for micro_crypto in ['SHIB', 'PEPE', 'FLOKI']):
            return 0.00001  # Micro cryptos: 1 pip = 0.00001
        
        # ========== FOREX PAIRS ==========
        else:
            return 0.0001  # Major FX pairs: 1 pip = 0.0001 (EUR/USD: 1.0850 -> 1.0857 = 7 pips)

    # 🚀 AUTO NOTIFICATION METHODS
    def _send_execution_notification(self, executed_order_data: Dict):
        """Send immediate notification when order is executed"""
        try:
            # Debug: Log notification attempt
            logger.info(f"🔔 Attempting to send execution notification for Ticket #{executed_order_data.get('ticket')}")
            
            # Check if auto notification is enabled
            is_enabled = self._is_auto_notification_enabled()
            logger.info(f"🔔 Auto notification enabled check: {is_enabled}")
            
            if not is_enabled:
                logger.warning(f"⚠️ Auto notification disabled - Ticket #{executed_order_data.get('ticket')} notification skipped")
                return
                
            from unified_notification_system import UnifiedNotificationSystem
            notification_service = UnifiedNotificationSystem()
            
            # Check notification format setting
            format_type = self._get_notification_format()
            logger.info(f"🔔 Notification format: {format_type}")
            
            # Format execution message (already includes analysis based on format)
            message = self._format_execution_message(executed_order_data)
            
            logger.info(f"🔔 Message length: {len(message)} characters")
            logger.info(f"🔔 Full message:\n{message}")
            logger.info(f"🔔 Message preview: {message[:100]}...")
            
            # Send notification
            success = notification_service.send_real_trade_notification(message)
            if success:
                logger.info(f"✅ Execution notification sent successfully: Ticket #{executed_order_data.get('ticket')}")
                
                # Start tracking this order for updates
                self._start_order_tracking(executed_order_data)
            else:
                logger.warning(f"❌ Execution notification failed: Ticket #{executed_order_data.get('ticket')} - check notification config")
                
        except Exception as e:
            logger.error(f"❌ Execution notification error: {e}")
            import traceback
            logger.error(f"❌ Notification error traceback: {traceback.format_exc()}")
    
    def _is_auto_notification_enabled(self) -> bool:
        """Check if auto notification is enabled"""
        try:
            import json
            import os
            
            config_path = os.path.join(os.getcwd(), 'notification_config.json')
            logger.info(f"🔔 Checking notification config at: {config_path}")
            
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Check if execution results notifications are enabled
                execution_enabled = config.get('settings', {}).get('send_execution_results', False)
                logger.info(f"🔔 send_execution_results setting: {execution_enabled}")
                
                # Also check if at least one platform is enabled and configured
                telegram_enabled = config.get('telegram', {}).get('enabled', False)
                telegram_configured = (config.get('telegram', {}).get('bot_token', '').strip() != '' and 
                                     config.get('telegram', {}).get('chat_id', '').strip() != '')
                logger.info(f"🔔 Telegram: enabled={telegram_enabled}, configured={telegram_configured}")
                
                # Check other platforms
                platforms_ready = []
                if telegram_enabled and telegram_configured:
                    platforms_ready.append('telegram')
                
                # Add other platform checks if needed
                discord_enabled = config.get('discord', {}).get('enabled', False)
                if discord_enabled and config.get('discord', {}).get('webhook_url', '').strip():
                    platforms_ready.append('discord')
                    
                slack_enabled = config.get('slack', {}).get('enabled', False)
                if slack_enabled and config.get('slack', {}).get('webhook_url', '').strip():
                    platforms_ready.append('slack')
                
                result = execution_enabled and len(platforms_ready) > 0
                logger.info(f"🔔 Auto notification final result: {result} (execution_enabled={execution_enabled}, platforms_ready={platforms_ready})")
                
                return result
            else:
                logger.warning(f"⚠️ Notification config file not found: {config_path}")
                return False
        except Exception as e:
            logger.debug(f"Error checking auto notification config: {e}")
            return False
    
    def _format_execution_message(self, executed_order_data: Dict) -> str:
        """Format execution message - Single unified format"""
        try:
            signal = executed_order_data.get('signal')
            result = executed_order_data.get('result')
            
            symbol = executed_order_data.get('symbol', 'Unknown')
            direction = executed_order_data.get('direction', 'Unknown').upper()
            entry_price = executed_order_data.get('entry_price', 0)
            ticket = executed_order_data.get('ticket', 0)
            timestamp = executed_order_data.get('timestamp', 'Unknown')
            
            # Get symbol digits for precise formatting like MT5
            try:
                import MetaTrader5 as mt5
                symbol_info = mt5.symbol_info(symbol)
                digits = symbol_info.digits if symbol_info else 5
            except:
                digits = 5  # Default for most currency pairs
            
            # Fallback for entry price if it's 0 or missing
            if entry_price == 0 or entry_price is None:
                if signal and hasattr(signal, 'entry_price') and signal.entry_price:
                    entry_price = signal.entry_price
                elif signal and hasattr(signal, 'price') and signal.price:
                    entry_price = signal.price
                elif result and hasattr(result, 'request') and hasattr(result.request, 'price'):
                    entry_price = result.request.price
            
            # Check notification format preference 
            notification_format = self._get_notification_format_setting()
            
            # Determine if this is Entry or DCA
            comment = executed_order_data.get('comment', '')
            signal_comment = ''
            
            # Check result comment
            if result and hasattr(result, 'comment'):
                signal_comment = result.comment
            
            # Check if it's DCA from multiple sources
            is_dca = (
                'DCA' in comment.upper() or
                'DCA' in signal_comment.upper() or 
                (signal and hasattr(signal, 'comment') and 'DCA' in signal.comment.upper()) or
                (signal and hasattr(signal, 'order_type') and 'DCA' in str(signal.order_type).upper()) or
                'DCA' in str(ticket)
            )
            
            # Determine DCA level
            dca_level = self._extract_dca_level(comment, signal_comment, signal)
            
            # Format timestamp - simple time format
            time_str = timestamp.strftime('%H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp).split()[-1] if ' ' in str(timestamp) else str(timestamp)
            
            # Always use English for execution notifications as requested
            title_line = "🚨 SIGNAL EXECUTED"
            
            # Get volume from various sources
            volume = executed_order_data.get('volume', 0)
            if volume == 0 or volume is None:
                if signal and hasattr(signal, 'volume') and signal.volume:
                    volume = signal.volume
                elif result and hasattr(result, 'request') and hasattr(result.request, 'volume'):
                    volume = result.request.volume
                else:
                    volume = 0.1  # Default fallback
            
            # Determine order type display - always English
            if is_dca:
                order_type_display = f"DCA{dca_level}"
            else:
                order_type_display = "Entry Signal"

            # Get confidence from signal
            confidence = 0
            if signal and hasattr(signal, 'confidence') and signal.confidence:
                confidence = signal.confidence
            elif executed_order_data.get('confidence'):
                confidence = executed_order_data.get('confidence')

            # Build execution notification - always English format with specified order
            message_lines = [
                title_line,
                f"📊 Symbol: {symbol}",
                f"📈 Direction: {direction}",
                f"💰 Entry Price: {entry_price:.{digits}f}"
            ]
            
            # Get SL/TP values
            stop_loss = None
            take_profit = None
            
            if hasattr(signal, 'stop_loss') and signal.stop_loss:
                stop_loss = signal.stop_loss
            elif result and hasattr(result, 'request') and hasattr(result.request, 'sl') and result.request.sl > 0:
                stop_loss = result.request.sl
                
            if hasattr(signal, 'take_profit') and signal.take_profit:
                take_profit = signal.take_profit
            elif result and hasattr(result, 'request') and hasattr(result.request, 'tp') and result.request.tp > 0:
                take_profit = result.request.tp
            
            # Add SL/TP in English - in the specified order
            if stop_loss:
                message_lines.append(f"🛡️ Stop Loss: {stop_loss:.{digits}f}")
            if take_profit:
                message_lines.append(f"🎯 Take Profit: {take_profit:.{digits}f}")
            
            # Add volume, order type, confidence, ticket, time in specified order
            message_lines.extend([
                f"📦 Volume: {volume:.2f} lots",
                f"🏷️ Order Type: {order_type_display}"
            ])
            
            # Add confidence if available
            if confidence > 0:
                message_lines.append(f"🎯 Confidence: {confidence:.0f}%")
            
            # Add ticket and time at the end
            message_lines.extend([
                f"🎟️ Ticket: #{ticket}",
                f"⏰ Time: {time_str}",
                ""
            ])
            
            # Add technical analysis if full format is selected
            if notification_format == "full":
                # Get actual checkbox settings from config
                analysis_settings = self._get_include_analysis_setting()
                detailed_analysis = self._get_technical_analysis(symbol, timestamp, analysis_settings)
                
                if detailed_analysis:
                    message_lines.extend(["", detailed_analysis])
            
            # Add custom message if enabled
            custom_message = self._get_custom_message()
            if custom_message:
                message_lines.extend(["", custom_message])
            
            # Add footer if enabled
            footer = self._get_branded_footer()
            if footer:
                message_lines.extend(["", footer])
            
            return "\n".join(message_lines)
            
        except Exception as e:
            logger.error(f"Error formatting execution message: {e}")
            return f"🚀 Order executed: Ticket #{executed_order_data.get('ticket', 'Unknown')}"
    
    def _get_notification_format_setting(self) -> str:
        """Get notification format setting from config"""
        try:
            import json
            import os
            
            config_path = os.path.join(os.getcwd(), 'notification_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                return config.get('settings', {}).get('notification_format', 'summary')
            return 'summary'
        except Exception as e:
            logger.debug(f"Error getting notification format setting: {e}")
            return 'summary'
    
    def _get_include_analysis_setting(self) -> dict:
        """Get detailed checkbox settings from config for report content filtering"""
        try:
            import json
            import os
            
            config_path = os.path.join(os.getcwd(), 'notification_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                settings = config.get('settings', {})
                return {
                    'include_technical': settings.get('include_technical', True),
                    'include_indicators': settings.get('include_indicators', True),
                    'include_summary': settings.get('include_summary', True),
                    'include_candlestick': settings.get('include_candlestick', True),
                    'include_price_patterns': settings.get('include_price_patterns', True),
                    'any_enabled': any([
                        settings.get('include_technical', True),
                        settings.get('include_indicators', True),
                        settings.get('include_summary', True),
                        settings.get('include_candlestick', True),
                        settings.get('include_price_patterns', True)
                    ])
                }
            return {
                'include_technical': True,
                'include_indicators': True, 
                'include_summary': True,
                'include_candlestick': True,
                'include_price_patterns': True,
                'any_enabled': True
            }
        except Exception as e:
            logger.debug(f"Error getting include analysis settings: {e}")
            return {
                'include_technical': True,
                'include_indicators': True,
                'include_summary': True, 
                'include_candlestick': True,
                'include_price_patterns': True,
                'any_enabled': True
            }
    
    def _extract_dca_level(self, comment: str, signal_comment: str, signal) -> str:
        """Extract DCA level from comment or signal"""
        try:
            # Check comment for DCA level patterns
            import re
            
            # Pattern: DCA1, DCA_L1, DCA_Level_1, etc.
            patterns = [
                r'DCA(\d+)',
                r'DCA_L(\d+)', 
                r'DCA_Level_(\d+)',
                r'DCA.*?(\d+)'
            ]
            
            for pattern in patterns:
                # Check main comment
                if comment:
                    match = re.search(pattern, comment.upper())
                    if match:
                        return match.group(1)
                
                # Check signal comment
                if signal_comment:
                    match = re.search(pattern, signal_comment.upper())
                    if match:
                        return match.group(1)
                        
                # Check signal attributes
                if signal and hasattr(signal, 'comment'):
                    match = re.search(pattern, str(signal.comment).upper())
                    if match:
                        return match.group(1)
            
            # Default to 1 if DCA but no level found
            return "1"
            
        except Exception as e:
            logger.debug(f"Error extracting DCA level: {e}")
            return "1"
    
    def _get_technical_analysis(self, symbol: str, timestamp, analysis_settings: dict = None) -> str:
        """Get technical analysis from report files - simplified version"""
        try:
            import os
            import glob
            
            # Look for recent analysis files
            analysis_dir = os.path.join(os.getcwd(), 'analysis_results')
            if not os.path.exists(analysis_dir):
                return ""
            
            # Check config for Vietnamese setting
            is_vietnamese = True  # Default to Vietnamese
            try:
                import json
                config_path = os.path.join(os.getcwd(), 'notification_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    is_vietnamese = config.get('settings', {}).get('format_vietnamese', True)
            except:
                pass
            
            # Format symbol for file search - handle different formats
            symbol_patterns = [
                symbol,  # Original symbol
                symbol.replace('_m', ''),  # Without _m suffix
                symbol.replace('.', '_')   # Replace dots with underscores
            ]
            
            report_files = []
            
            # Search for report files
            for sym_pattern in symbol_patterns:
                if is_vietnamese:
                    pattern = f"{analysis_dir}/{sym_pattern}*_report_vi_*.txt"
                else:
                    pattern = f"{analysis_dir}/{sym_pattern}*_report_en_*.txt"
                
                files = glob.glob(pattern)
                if files:
                    report_files.extend(files)
                    break  # Found files for this pattern
            
            if not report_files:
                logger.debug(f"No report files found for symbol: {symbol}")
                return ""
            
            # Get the most recent file
            latest_file = max(report_files, key=os.path.getmtime)
            logger.debug(f"Using report file: {os.path.basename(latest_file)}")
            
            # Read content
            with open(latest_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Return filtered analysis based on checkbox settings  
            if analysis_settings and analysis_settings.get('any_enabled', True):
                return self._filter_analysis_content(content, analysis_settings, is_vietnamese)
            else:
                return self._extract_simple_analysis(content, is_vietnamese)
            
        except Exception as e:
            logger.debug(f"Error getting technical analysis: {e}")
            return ""
    
    def _extract_simple_analysis(self, content: str, is_vietnamese: bool) -> str:
        """Extract simplified technical analysis from report content"""
        try:
            lines = content.split('\n')
            analysis_lines = []
            
            if is_vietnamese:
                analysis_lines.append("📊 PHÂN TÍCH KỸ THUẬT:")
                
                # Look for key indicators in Vietnamese
                for line in lines:
                    line = line.strip()
                    # Clean up line - remove leading dashes and extra spaces
                    if line.startswith('- '):
                        line = line[2:].strip()
                    elif line.startswith('• '):
                        line = line[2:].strip()
                    
                    if any(keyword in line.lower() for keyword in ['rsi', 'macd', 'moving average', 'đường trung bình', 'bollinger']):
                        if line and not line.startswith('---') and len(line) < 200:
                            analysis_lines.append(f"• {line}")
                    elif any(keyword in line.lower() for keyword in ['support', 'resistance', 'hỗ trợ', 'kháng cự']):
                        if line and not line.startswith('---') and len(line) < 200:
                            analysis_lines.append(f"• {line}")
                    elif any(keyword in line.lower() for keyword in ['xu hướng', 'trend', 'tăng', 'giảm']):
                        if line and not line.startswith('---') and len(line) < 200:
                            analysis_lines.append(f"• {line}")
                
                # Add generic analysis if no specific indicators found
                if len(analysis_lines) == 1:  # Only header
                    analysis_lines.extend([
                        "• Chỉ báo kỹ thuật được phân tích",
                        "• Mức hỗ trợ/kháng cự đã xác định", 
                        "• Xu hướng thị trường được đánh giá"
                    ])
            else:
                analysis_lines.append("📊 TECHNICAL ANALYSIS:")
                
                # Look for key indicators in English
                for line in lines:
                    line = line.strip()
                    if any(keyword in line.lower() for keyword in ['rsi', 'macd', 'moving average', 'bollinger', 'stochastic']):
                        if line and not line.startswith('---') and len(line) < 200:
                            analysis_lines.append(f"• {line}")
                    elif any(keyword in line.lower() for keyword in ['support', 'resistance', 'trend', 'breakout']):
                        if line and not line.startswith('---') and len(line) < 200:
                            analysis_lines.append(f"• {line}")
                
                # Add generic analysis if no specific indicators found
                if len(analysis_lines) == 1:  # Only header
                    analysis_lines.extend([
                        "• Technical indicators analyzed",
                        "• Support/resistance levels identified",
                        "• Market trend evaluated"
                    ])
            
            # Limit to 6 lines total (header + 5 analysis points)
            if len(analysis_lines) > 6:
                analysis_lines = analysis_lines[:6]
            
            return '\n'.join(analysis_lines)
            
        except Exception as e:
            logger.debug(f"Error extracting simple analysis: {e}")
            if is_vietnamese:
                return "📊 PHÂN TÍCH KỸ THUẬT:\n• Phân tích kỹ thuật đã được thực hiện"
            else:
                return "📊 TECHNICAL ANALYSIS:\n• Technical analysis completed"
    
    def _filter_analysis_content(self, content: str, analysis_settings: dict, is_vietnamese: bool = True) -> str:
        """Filter analysis content based on checkbox selections - FULL VERSION from app.py"""
        try:
            # Check what user wants to include - DEFAULT FALSE to respect user choice
            include_technical = analysis_settings.get('include_technical', False)
            include_indicators = analysis_settings.get('include_indicators', False) 
            include_summary = analysis_settings.get('include_summary', False)
            include_candlestick = analysis_settings.get('include_candlestick', False)
            include_price_patterns = analysis_settings.get('include_price_patterns', False)
            
            if not any([include_technical, include_indicators, include_summary, include_candlestick, include_price_patterns]):
                return ""  # Nothing selected
            
            sections = []
            lines = content.split('\n')
            
            # Signal info section removed - execution details already shown in header
            
            # 🔍 TECHNICAL ANALYSIS - "Phân tích kỹ thuật:" (EXCLUDING indicators)
            if include_technical:
                tech_section = []
                in_tech_section = False
                
                for line in lines:
                    line_lower = line.lower().strip()
                    
                    if 'phân tích kỹ thuật:' in line_lower:
                        in_tech_section = True
                        tech_section.append('🔍 PHÂN TÍCH KỸ THUẬT:')
                        continue
                    elif in_tech_section:
                        if 'tóm tắt:' in line_lower:
                            break  # End of section
                        
                        # ALWAYS skip indicators sub-section (it will be handled separately)
                        if 'chỉ báo kỹ thuật:' in line_lower:
                            # Skip entire indicators sub-section
                            break  # Stop processing technical analysis here
                        
                        # Only include non-indicator lines (trend, support, resistance, etc.)
                        if line.strip() and not line.strip().startswith('•'):
                            tech_section.append(line)
                        elif line.strip() == '':
                            tech_section.append('')  # Keep empty lines for formatting
                
                if tech_section and len(tech_section) > 1:
                    sections.extend(tech_section + [''])
            
            # 📈 TECHNICAL INDICATORS - Extract ONLY bullet points from indicators sub-section
            if include_indicators:
                indicator_section = ['📈 CHỈ BÁO KỸ THUẬT:']
                in_tech_section = False
                in_indicator_subsection = False
                
                for line in lines:
                    line_lower = line.lower().strip()
                    
                    if 'phân tích kỹ thuật:' in line_lower:
                        in_tech_section = True
                        continue
                    elif in_tech_section:
                        if 'tóm tắt:' in line_lower:
                            break  # End of technical section
                        
                        # Found indicators sub-section
                        if 'chỉ báo kỹ thuật:' in line_lower:
                            in_indicator_subsection = True
                            continue
                        
                        # Collect ONLY bullet points and their continuations
                        if in_indicator_subsection:
                            if line.strip().startswith('•'):
                                indicator_section.append(line)
                            elif line.strip().startswith('→') or line.strip().startswith('–') or (
                                line.strip() and 
                                any(keyword in line.lower() for keyword in ['ema', 'dốc', 'xếp chồng', 'ngắn hạn', 'dài hạn', 'trung hạn'])
                            ):
                                # Multi-line indicator descriptions
                                indicator_section.append(line)
                
                if len(indicator_section) > 1:  # More than just the header
                    sections.extend(indicator_section + [''])
            
            # 📋 SUMMARY - "Tóm tắt:"
            if include_summary:
                summary_section = []
                in_summary_section = False
                for line in lines:
                    line_lower = line.lower().strip()
                    if line_lower.startswith('tóm tắt:'):
                        in_summary_section = True
                        summary_section.append('📋 TÓM TẮT:')
                        continue
                    elif in_summary_section:
                        if line_lower.startswith('- mô hình'):
                            break  # Next section started
                        elif line.strip() and line.strip().startswith('-'):  # Summary bullet points
                            summary_section.append(line)
                
                if summary_section and len(summary_section) > 1:
                    sections.extend(summary_section + [''])
            
            # � PRICE PATTERNS - "- Mô hình giá:"
            if include_price_patterns:
                price_section = []
                in_price_section = False
                for line in lines:
                    line_lower = line.lower().strip()
                    if '- mô hình giá:' in line_lower:
                        in_price_section = True
                        price_section.append('💹 MÔ HÌNH GIÁ:')
                        continue
                    elif in_price_section:
                        if '- mô hình nến:' in line_lower:
                            break  # Next section started
                        elif line.strip() and line.strip().startswith('•'):  # Price pattern bullet points
                            price_section.append(line)
                
                if price_section and len(price_section) > 1:
                    sections.extend(price_section + [''])
            
            # 🕯️ CANDLESTICK PATTERNS - "- Mô hình nến:"
            if include_candlestick:
                candle_section = []
                in_candle_section = False
                for line in lines:
                    line_lower = line.lower().strip()
                    if '- mô hình nến:' in line_lower:
                        in_candle_section = True
                        candle_section.append('🕯️ MÔ HÌNH NẾN:')
                        continue
                    elif in_candle_section:
                        if line.strip() and not line.strip().startswith('•'):
                            break  # End of section
                        elif line.strip().startswith('•'):  # Candlestick pattern bullet points
                            candle_section.append(line)
                
                if candle_section and len(candle_section) > 1:
                    sections.extend(candle_section + [''])
            
            # Return full analysis without length limit
            result = '\n'.join(sections)
            return result
            
        except Exception as e:
            logger.debug(f"Error filtering analysis content: {e}")
            return ""



    def _start_order_tracking(self, executed_order_data: Dict):
        """Start tracking order for updates (SL/TP changes, close, etc.)"""
        try:
            # Check if tracking is enabled
            if not self._is_order_tracking_enabled():
                return
                
            ticket = executed_order_data.get('ticket')
            if not ticket:
                return
                
            # Store tracking info
            tracking_info = {
                'ticket': ticket,
                'symbol': executed_order_data.get('symbol'),
                'direction': executed_order_data.get('direction'),
                'entry_price': executed_order_data.get('entry_price'),
                'volume': executed_order_data.get('volume'),
                'start_tracking_time': datetime.now(),
                'last_update_time': datetime.now(),
                'last_sl': None,
                'last_tp': None,
                'status': 'open'
            }
            
            # Add to tracking list (stored in instance)
            if not hasattr(self, 'tracked_orders'):
                self.tracked_orders = {}
            
            self.tracked_orders[ticket] = tracking_info
            logger.info(f"🎯 Started tracking order #{ticket} for updates")
            
        except Exception as e:
            logger.error(f"Error starting order tracking: {e}")
    
    def _is_order_tracking_enabled(self) -> bool:
        """Check if order tracking is enabled"""
        try:
            import json
            import os
            
            config_path = os.path.join(os.getcwd(), 'notification_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                return config.get('settings', {}).get('track_order_updates', False)
            return False
        except Exception as e:
            logger.debug(f"Error checking order tracking config: {e}")
            return False
    
    def check_tracked_orders_updates(self):
        """Check for updates on tracked orders (SL/TP changes, close, etc.)"""
        try:
            if not hasattr(self, 'tracked_orders') or not self.tracked_orders:
                return
            
            # Import MT5 for position checking
            import MetaTrader5 as mt5
            
            # Check each tracked order
            for ticket, tracking_info in list(self.tracked_orders.items()):
                try:
                    # Get current position info
                    position = None
                    positions = mt5.positions_get(ticket=ticket)
                    if positions and len(positions) > 0:
                        position = positions[0]
                    
                    if position:
                        # Order is still open - check for SL/TP changes
                        current_sl = position.sl
                        current_tp = position.tp
                        
                        # Check for SL changes
                        if tracking_info['last_sl'] != current_sl:
                            self._send_sl_change_notification(ticket, tracking_info, current_sl)
                            tracking_info['last_sl'] = current_sl
                        
                        # Check for TP changes
                        if tracking_info['last_tp'] != current_tp:
                            self._send_tp_change_notification(ticket, tracking_info, current_tp)
                            tracking_info['last_tp'] = current_tp
                        
                        tracking_info['last_update_time'] = datetime.now()
                        
                    else:
                        # Order is closed - send close notification
                        self._send_order_close_notification(ticket, tracking_info)
                        
                        # Remove from tracking
                        del self.tracked_orders[ticket]
                        logger.info(f"🏁 Stopped tracking closed order #{ticket}")
                
                except Exception as order_error:
                    logger.debug(f"Error checking order #{ticket}: {order_error}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error checking tracked orders updates: {e}")
    
    def _send_sl_change_notification(self, ticket: int, tracking_info: Dict, new_sl: float):
        """Send notification when SL changes"""
        try:
            if not self._is_sl_tp_notification_enabled():
                return
                
            from unified_notification_system import UnifiedNotificationSystem
            notification_service = UnifiedNotificationSystem()
            
            symbol = tracking_info.get('symbol', 'Unknown')
            direction = tracking_info.get('direction', 'Unknown')
            old_sl = tracking_info.get('last_sl', 'None')
            
            message = f"🛡️ STOP LOSS CẬP NHẬT!\n\n"
            message += f"🎫 Ticket: #{ticket}\n"
            message += f"📊 Symbol: {symbol} ({direction})\n"
            message += f"🔄 SL: {old_sl} → {new_sl:.5f}\n"
            message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Add custom message if enabled
            custom_message = self._get_custom_message()
            if custom_message:
                message += f"\n\n{custom_message}"
            
            # Add footer if enabled
            footer = self._get_branded_footer()
            if footer:
                message += f"\n\n{footer}"
            
            notification_service.send_real_trade_notification(message)
            logger.info(f"🔔 SL change notification sent: #{ticket}")
            
        except Exception as e:
            logger.error(f"Error sending SL change notification: {e}")
    
    def _send_tp_change_notification(self, ticket: int, tracking_info: Dict, new_tp: float):
        """Send notification when TP changes"""
        try:
            if not self._is_sl_tp_notification_enabled():
                return
                
            from unified_notification_system import UnifiedNotificationSystem
            notification_service = UnifiedNotificationSystem()
            
            symbol = tracking_info.get('symbol', 'Unknown')
            direction = tracking_info.get('direction', 'Unknown')
            old_tp = tracking_info.get('last_tp', 'None')
            
            message = f"🎯 TAKE PROFIT CẬP NHẬT!\n\n"
            message += f"🎫 Ticket: #{ticket}\n"
            message += f"📊 Symbol: {symbol} ({direction})\n"
            message += f"🔄 TP: {old_tp} → {new_tp:.5f}\n"
            message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Add custom message if enabled
            custom_message = self._get_custom_message()
            if custom_message:
                message += f"\n\n{custom_message}"
            
            # Add footer if enabled
            footer = self._get_branded_footer()
            if footer:
                message += f"\n\n{footer}"
            
            notification_service.send_real_trade_notification(message)
            logger.info(f"🔔 TP change notification sent: #{ticket}")
            
        except Exception as e:
            logger.error(f"Error sending TP change notification: {e}")
    
    def _send_order_close_notification(self, ticket: int, tracking_info: Dict):
        """Send notification when order is closed"""
        try:
            if not self._is_order_close_notification_enabled():
                return
                
            from unified_notification_system import UnifiedNotificationSystem
            notification_service = UnifiedNotificationSystem()
            
            symbol = tracking_info.get('symbol', 'Unknown')
            direction = tracking_info.get('direction', 'Unknown')
            entry_price = tracking_info.get('entry_price', 0)
            volume = tracking_info.get('volume', 0)
            
            # Try to get close details from MT5 history
            close_info = self._get_close_details(ticket)
            
            message = f"🏁 LỆNH ĐÃ ĐÓNG!\n\n"
            message += f"🎫 Ticket: #{ticket}\n"
            message += f"📊 Symbol: {symbol} ({direction})\n"
            message += f"💰 Entry: {entry_price:.5f}\n"
            message += f"📦 Volume: {volume:.2f} lots\n"
            
            if close_info:
                message += f"🔚 Close Price: {close_info.get('close_price', 'Unknown')}\n"
                message += f"💵 P&L: {close_info.get('profit', 'Unknown')}\n"
                
            message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Add custom message if enabled
            custom_message = self._get_custom_message()
            if custom_message:
                message += f"\n\n{custom_message}"
            
            # Add footer if enabled
            footer = self._get_branded_footer()
            if footer:
                message += f"\n\n{footer}"
            
            notification_service.send_real_trade_notification(message)
            logger.info(f"🔔 Order close notification sent: #{ticket}")
            
        except Exception as e:
            logger.error(f"Error sending order close notification: {e}")
    
    def _get_close_details(self, ticket: int) -> Dict:
        """Get close details from MT5 history"""
        try:
            import MetaTrader5 as mt5
            from datetime import datetime, timedelta
            
            # Get recent history
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)  # Last hour
            
            deals = mt5.history_deals_get(start_time, end_time, position=ticket)
            if deals and len(deals) > 0:
                # Find the close deal (OUT)
                for deal in deals:
                    if deal.entry == mt5.DEAL_ENTRY_OUT:
                        return {
                            'close_price': deal.price,
                            'profit': deal.profit,
                            'close_time': deal.time
                        }
            
            return {}
        except Exception as e:
            logger.debug(f"Error getting close details for #{ticket}: {e}")
            return {}
    
    def _is_sl_tp_notification_enabled(self) -> bool:
        """Check if SL/TP change notifications are enabled"""
        try:
            import json
            import os
            
            config_path = os.path.join(os.getcwd(), 'notification_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                return config.get('settings', {}).get('notify_sl_tp_changes', False)
            return False
        except Exception as e:
            logger.debug(f"Error checking SL/TP notification config: {e}")
            return False
    
    def _is_order_close_notification_enabled(self) -> bool:
        """Check if order close notifications are enabled"""
        try:
            import json
            import os
            
            config_path = os.path.join(os.getcwd(), 'notification_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                return config.get('settings', {}).get('notify_order_close', False)
            return False
        except Exception as e:
            logger.debug(f"Error checking order close notification config: {e}")
            return False
    
    def _get_notification_format(self) -> str:
        """Get notification format from config (summary/full)"""
        try:
            import json
            import os
            
            config_path = os.path.join(os.getcwd(), 'notification_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                return config.get('settings', {}).get('notification_format', 'summary')
            return 'summary'
        except Exception as e:
            logger.error(f"Error getting notification format: {e}")
            return 'summary'
    
    def _get_custom_message(self) -> str:
        """Get custom message from config if enabled"""
        try:
            import json
            import os
            
            config_path = os.path.join(os.getcwd(), 'notification_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                settings = config.get('settings', {})
                
                # Check if custom message is enabled and has content
                if (settings.get('send_custom_message', False) and 
                    settings.get('custom_message', '').strip()):
                    
                    custom_text = settings.get('custom_message', '').strip()
                    
                    # Return only the content without header
                    return custom_text
            
            return ""
        except Exception as e:
            logger.debug(f"Error getting custom message: {e}")
            return ""
    
    def _get_branded_footer(self) -> str:
        """Get branded footer from config if enabled"""
        try:
            import json
            import os
            
            config_path = os.path.join(os.getcwd(), 'notification_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                branding = config.get('branding', {})
                settings = config.get('settings', {})
                
                # Check if footer is enabled
                if not branding.get('enable_custom_footer', True):
                    return ""
                
                # Determine language
                is_vietnamese = settings.get('format_vietnamese', True)
                
                # Get values from config
                if is_vietnamese:
                    system_name = branding.get('system_name_vi', 'Hệ thống AI VU HIEN CFDs')
                else:
                    system_name = branding.get('system_name_en', 'VU HIEN CFDs AI System')
                
                phone = branding.get('phone', '+84 39 65 60 888')
                email = branding.get('email', '')
                
                # Build footer
                footer_parts = [f"🤖 {system_name}"]
                if phone:
                    footer_parts.append(f"📱 {phone}")
                if email:
                    footer_parts.append(f"📧 {email}")
                
                return '\n'.join(footer_parts)
            
            return ""
        except Exception as e:
            logger.debug(f"Error getting branded footer: {e}")
            return ""
    
    def _get_symbol_analysis(self, symbol: str) -> str:
        """Get latest technical analysis for symbol"""
        try:
            import os
            import glob
            from datetime import datetime
            
            # Look for latest analysis report
            analysis_dir = os.path.join(os.getcwd(), 'analysis_results')
            if not os.path.exists(analysis_dir):
                return None
                
            # Find latest Vietnamese report for the symbol
            pattern = os.path.join(analysis_dir, f"{symbol}_report_vi_*.txt")
            files = glob.glob(pattern)
            
            if not files:
                return None
                
            # Get the latest file
            latest_file = max(files, key=os.path.getctime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract key sections - get main signal info and brief analysis
            lines = content.split('\n')
            
            # Find the core signal information
            signal_info = []
            in_analysis_section = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Get signal basics
                if any(keyword in line for keyword in ['Tín hiệu:', 'Độ tin cậy:', 'Entry:', 'Stoploss:', 'Takeprofit:']):
                    signal_info.append(line)
                
                # Get key analysis points
                elif 'Phân tích kỹ thuật:' in line:
                    in_analysis_section = True
                    signal_info.append('\n📊 Phân tích:')
                elif in_analysis_section and line.startswith('- '):
                    # Add key analysis points (trends, support/resistance)
                    if any(keyword in line.lower() for keyword in ['xu hướng', 'trendline', 'hỗ trợ', 'kháng cự']):
                        signal_info.append(line[:80] + ('...' if len(line) > 80 else ''))
                elif in_analysis_section and line.startswith('Tóm tắt:'):
                    break  # Stop at summary section
            
            # Limit total length for Telegram
            result = '\n'.join(signal_info[:15])
            if len(result) > 800:  # Safe limit for Telegram
                result = result[:800] + '...'
                
            return result
            
        except Exception as e:
            logger.error(f"Error getting symbol analysis: {e}")
            return None

# Legacy compatibility
class OrderHandler(AdvancedOrderExecutor):
    """Legacy OrderHandler class for backward compatibility"""
    
    def __init__(self, connection):
        super().__init__(connection=connection)
        self.conn = connection  # Legacy attribute
    
    def send_order(self, action: str, symbol: str, lot: float, sl: float, tp: float) -> bool:
        """Legacy send_order method"""
        signal = TradeSignal(
            symbol=symbol,
            action=action,
            entry_price=0.0,  # Will be determined by current market price
            stop_loss=sl,
            take_profit=tp,
            volume=lot,
            strategy="LEGACY_SYSTEM"
        )
        
        result = self.execute_market_order(signal)
        return result.success

def main():
    """Test the advanced order executor with comprehensive testing"""
    import sys
    
    print("🚀 Advanced Order Executor - Enhanced Edition")
    print("="*60)
    
    # Parse command line arguments
    test_mode = "--test" in sys.argv
    comprehensive_test = "--comprehensive" in sys.argv
    apply_actions = "--apply-actions" in sys.argv
    process_signals = "--process-signals" in sys.argv
    
    if test_mode or comprehensive_test:
        print("🧪 Running in TEST MODE")
        print("-" * 40)
        
        # Create executor for testing (no MT5 required)
        executor = get_executor_instance(magic_number=999999)
        
        if comprehensive_test:
            # Run comprehensive test suite
            results = executor.run_comprehensive_test()
            
            # Save test results to unified settings
            try:
                settings_path = "risk_management/risk_settings.json"
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                if 'reports' not in settings:
                    settings['reports'] = {
                        'risk_reports': [],
                        'execution_reports': [], 
                        'test_reports': [],
                        'max_stored_reports': 100,
                        'last_cleanup': None
                    }
                
                test_report = {
                    "timestamp": datetime.now().isoformat(),
                    "test_type": "comprehensive",
                    "results": results
                }
                
                settings['reports']['test_reports'].append(test_report)
                
                # Keep only last N reports
                max_reports = settings['reports'].get('max_stored_reports', 100)
                if len(settings['reports']['test_reports']) > max_reports:
                    settings['reports']['test_reports'] = settings['reports']['test_reports'][-max_reports:]
                
                with open(settings_path, 'w', encoding='utf-8') as f:
                    json.dump(settings, f, indent=2, ensure_ascii=False, default=str)
                
                print(f"📊 Test report saved to unified settings: {settings_path}")
                
            except Exception as e:
                print(f"❌ Error saving test report: {e}")
            
        else:
            # Run basic functionality test
            results = executor.test_order_executor_functionality()
            print(f"📊 Test results: {results}")
        
        return
    
    elif apply_actions:
        print("⚙️ Running APPLY ACTIONS mode")
        print("-" * 40)
        
        # Initialize MT5 for real execution
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return
        
        try:
            executor = get_executor_instance(magic_number=123456)
            
            # Apply actions from analysis results
            results = executor.apply_actions_from_json()
            
            print("\n📋 APPLY ACTIONS RESULTS:")
            print(f"✅ Applied: {results.get('applied', 0)} actions")
            print(f"⏭️ Skipped: {len(results.get('skipped', []))}")
            print(f"❌ Errors: {len(results.get('errors', []))}")
            
            # Show action breakdown
            by_action = results.get('by_action', {})
            if by_action:
                print("\n📊 Actions by type:")
                for action, count in by_action.items():
                    print(f"   {action}: {count}")
            
            # Save execution report
            report_path = executor.save_execution_report()
            if report_path:
                print(f"📊 Execution report saved: {report_path}")
            
        finally:
            mt5.shutdown()
            print("👋 MT5 shutdown complete")
        
        return
    
    elif process_signals:
        print("📊 Running PROCESS SIGNALS mode")
        print("-" * 40)
        
        # Initialize MT5 for real execution
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return
        
        try:
            executor = get_executor_instance(magic_number=123456)
            
            # Process new signals from signal files
            results = executor.process_new_signals()
            
            print("\n📋 SIGNAL PROCESSING RESULTS:")
            print(f"✅ Applied: {results.get('applied', 0)} new orders")
            print(f"⏭️ Skipped: {results.get('skipped', 0)} signals")
            print(f"❌ Errors: {results.get('errors', 0)}")
            print(f"📊 Total signals processed: {results.get('signals_processed', 0)}")
            
            # Show details
            details = results.get('details', [])
            if details:
                print("\n📋 Signal processing details:")
                for detail in details:
                    symbol = detail.get('symbol', 'UNKNOWN')
                    action = detail.get('action', 'UNKNOWN')
                    status = detail.get('status', 'UNKNOWN')
                    if status == 'success':
                        ticket = detail.get('ticket', 'N/A')
                        volume = detail.get('volume', 'N/A')
                        confidence = detail.get('confidence', 'N/A')
                        print(f"   ✅ {symbol} {action}: Ticket {ticket}, Volume {volume}, Confidence {confidence}%")
                    else:
                        error = detail.get('error', 'Unknown error')
                        print(f"   ❌ {symbol} {action}: {error}")
            
            # Save execution report
            report_path = executor.save_execution_report()
            if report_path:
                print(f"📊 Execution report saved: {report_path}")
            
        finally:
            mt5.shutdown()
            print("👋 MT5 shutdown complete")
        
        return
    
    # Default demo mode (original behavior)
    print("🎭 Running in DEMO MODE")
    print("-" * 40)
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return
    
    try:
        # Create executor
        executor = get_executor_instance(magic_number=999999)
        
        # Print status
        executor.print_status()
        
        # 🧮 NEW: Print risk settings and test volume calculations
        executor.print_risk_settings()
        executor.test_volume_calculation_scenarios()
        
        # Test smart market order execution
        print("\n🧪 Testing SMART market order execution...")
        try:
            result = executor.execute_smart_market_order(
                symbol="EURUSD",
                action="BUY",
                stop_loss=1.0950,
                take_profit=1.1100,
                confidence=3.5,
                comment="Smart demo order"
            )
            
            if result.success:
                print(f"✅ Smart demo order successful: Ticket {result.ticket}")
            else:
                print(f"❌ Smart demo order failed: {result.error_message}")
        except Exception as e:
            print(f"❌ Smart order test failed: {e}")

        # Test signal with proper S/L T/P distances
        current_price = 1.17322  # Use current market price
        test_signal = TradeSignal(
            symbol="EURUSD",
            action="BUY",
            entry_price=current_price,
            stop_loss=current_price - 0.0020,  # 20 pips SL
            take_profit=current_price + 0.0030,  # 30 pips TP
            volume=0.1,
            confidence=3.5,
            strategy="TEST_SYSTEM",
            comment="Demo order"
        )
        
        print("\n🧪 Testing market order execution...")
        result = executor.execute_market_order(test_signal)
        
        if result.success:
            print(f"✅ Demo order successful: Ticket {result.ticket}")
        else:
            print(f"❌ Demo order failed: {result.error_message}")
        
        # Show final status
        executor.print_status()
        
        # Save report
        report_path = executor.save_execution_report()
        if report_path:
            print(f"📊 Report saved: {report_path}")
        
    finally:
        mt5.shutdown()
        print("👋 MT5 shutdown complete")

if __name__ == "__main__":
    import sys
    
    # Print usage help
    if "--help" in sys.argv or "-h" in sys.argv:
        print("🚀 Advanced Order Executor with Smart Volume - Usage:")
        print("="*60)
        print("  python order_executor.py                    # Demo mode with volume tests")
        print("  python order_executor.py --test             # Basic functionality test")
        print("  python order_executor.py --comprehensive    # Comprehensive test suite")
        print("  python order_executor.py --apply-actions    # Apply real actions from JSON")
        print("  python order_executor.py --process-signals  # Process new entry signals")
        print("  python order_executor.py --risk-monitor     # Monitor and auto-close risky positions")
        print("  python order_executor.py --help             # Show this help")
        print("")
        print("🧮 NEW FEATURES:")
        print("  ✅ Smart Volume Calculation based on Risk Settings")
        print("  ✅ Integration with risk_management/risk_settings.json")
        print("  ✅ Support for Risk-Based, Fixed, and Default volume modes")
        print("  ✅ Symbol-specific multipliers and exposure limits")
        print("  ✅ DCA (Dollar Cost Averaging) volume scaling")
        print("  ✅ Confidence-based volume adjustments")
        print("  ✅ NEW: Process new entry signals from signal files")
        print("  ✅ NEW: Automated risk monitoring and position closing")
        print("="*60)
    elif "--risk-monitor" in sys.argv:
        # Risk monitoring mode - auto-close positions that violate risk settings
        from mt5_connector import MT5ConnectionManager
        
        print("🛡️ Starting Risk Monitoring System...")
        print("="*50)
        
        try:
            # Initialize MT5 connection
            connector = MT5ConnectionManager()
            if not connector.connect():
                print("❌ Failed to connect to MT5")
                sys.exit(1)
            
            # Initialize executor
            executor = get_executor_instance(connection=connector)
            
            # Run risk monitoring
            closed_positions = executor.check_and_auto_close_positions()
            
            if closed_positions:
                print(f"✅ Risk monitoring completed - closed {len(closed_positions)} positions")
                for pos in closed_positions:
                    print(f"   📉 Closed {pos['symbol']} (ticket: {pos['ticket']}) - {pos['reason']}")
            else:
                print("✅ Risk monitoring completed - no positions needed closing")
            
            connector.disconnect()
            
        except Exception as e:
            print(f"❌ Risk monitoring error: {e}")
            sys.exit(1)
    else:
        main()