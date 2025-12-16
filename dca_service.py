#!/usr/bin/env python3
"""
DCA Service - Quản lý toàn bộ logic DCA theo Fibonacci
Hỗ trợ 2 chế độ thực thi:
1. "Chạm Mức (Market)" - Market order khi giá chạm mức Fibonacci  
2. "Đặt Lệnh Chờ tại Mức" - Pending limit order tại các mức Fibonacci với điều chỉnh giá theo thời gian thực
"""
import json
import os
import sys
import time
import glob
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Add project directory to path
sys.path.append(str(Path(__file__).parent))

# Import risk validator for volume limit checking
from order_executor import ComprehensiveRiskValidator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import MetaTrader 5
try:
    import MetaTrader5 as mt5
except ImportError:
    logger.error("MetaTrader5 package not found. Please install: pip install MetaTrader5")
    mt5 = None

class FibonacciDCAService:
    def __init__(self):
        self.running = False
        self.scan_interval = 5  # seconds
        self.risk_settings = {}
        self.load_risk_settings()
        # Initialize risk validator for volume limit checking
        self.risk_validator = None
        
    def load_risk_settings(self):
        """Load risk settings from file"""
        try:
            risk_settings_path = 'risk_management/risk_settings.json'
            if os.path.exists(risk_settings_path):
                with open(risk_settings_path, 'r', encoding='utf-8') as f:
                    self.risk_settings = json.load(f)
                logger.info("✅ DCA Service: Risk settings loaded")
                return True
            else:
                logger.error("❌ DCA Service: Risk settings file not found")
                return False
        except Exception as e:
            logger.error(f"❌ DCA Service: Failed to load settings: {e}")
            return False
    
    def initialize_risk_validator(self):
        """Initialize risk validator for volume limit checking"""
        try:
            if self.risk_validator is None:
                self.risk_validator = ComprehensiveRiskValidator(self.risk_settings)
                logger.info("✅ DCA Service: Risk validator initialized")
            return True
        except Exception as e:
            logger.error(f"❌ DCA Service: Failed to initialize risk validator: {e}")
            return False
    
    def is_dca_enabled(self) -> bool:
        """Check if DCA is enabled"""
        return self.risk_settings.get('enable_dca', False)
    
    def is_fibonacci_mode(self) -> bool:
        """Check if DCA mode is Fibonacci"""
        dca_mode = self.risk_settings.get('dca_mode', '')
        # Check for new key format and legacy formats
        return dca_mode in ['fibo_levels', 'fibonacci', 'Fibonacci', 'Mức Fibonacci', 'Mức Fibo'] or 'fibo' in dca_mode.lower()
    
    def get_execution_mode(self) -> str:
        """Get Fibonacci execution mode"""
        return self.risk_settings.get('dca_fibo_exec_mode', 'Chạm Mức (Market)')
    
    def process_fibonacci_trigger(self) -> bool:
        """Process Fibonacci DCA trigger file from comprehensive_aggregator.py"""
        trigger_path = "dca_locks/fibonacci_trigger.json"
        
        try:
            if not os.path.exists(trigger_path):
                return False
                
            # Read trigger file
            with open(trigger_path, 'r', encoding='utf-8') as f:
                trigger_data = json.load(f)
            
            # Validate trigger data
            if trigger_data.get('trigger_source') != 'comprehensive_aggregator':
                logger.warning(f"Unknown trigger source: {trigger_data.get('trigger_source')}")
                return False
                
            if trigger_data.get('dca_mode') != 'fibonacci':
                logger.warning(f"Non-Fibonacci trigger: {trigger_data.get('dca_mode')}")
                return False
            
            symbols = trigger_data.get('symbols', [])
            logger.info(f"Fibonacci trigger received for {len(symbols)} symbols: {symbols}")
            
            # Process DCA opportunities immediately for triggered symbols
            self.process_dca_opportunities()
            
            # Remove trigger file after processing
            os.remove(trigger_path)
            logger.info(f"Processed and removed trigger file: {trigger_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing Fibonacci trigger: {e}")
            return False
    
    def should_run_service(self) -> bool:
        """Check if service should run"""
        if not self.is_dca_enabled():
            logger.warning("⚠️ DCA Strategy is disabled")
            return False
            
        if not self.is_fibonacci_mode():
            logger.warning("⚠️ DCA mode is not Fibonacci")
            return False
            
        return True
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5:
                logger.error("❌ MetaTrader5 module not available")
                return False
                
            if not mt5.initialize():
                logger.error(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False
            
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("❌ Failed to get account info")
                return False
                
            logger.info(f"✅ MT5 connected - Account: {account_info.login}")
            return True
            
        except Exception as e:
            logger.error(f"❌ MT5 connection error: {e}")
            return False
    
    def get_current_positions(self) -> List[Dict]:
        """Get current entry positions"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                # Only get positions with GPT Entry comment (flexible pattern)
                if pos.comment and ("GPT_AI20B|Entry" in pos.comment or "GPT_20B|Entry" in pos.comment):
                    result.append({
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'type': pos.type,  # 0=BUY, 1=SELL
                        'volume': pos.volume,
                        'price_open': pos.price_open,
                        'price_current': pos.price_current,
                        'profit': pos.profit,
                        'comment': pos.comment
                    })
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting positions: {e}")
            return []
    
    def load_atr_for_symbol(self, symbol: str) -> Optional[float]:
        """Load ATR value from indicator data for DCA distance calculation"""
        try:
            # Use H1 timeframe for ATR (standard for DCA)
            timeframe = 'H1'
            file_path = f"indicator_output/{symbol}_{timeframe}_indicators.json"
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Get latest data
                if isinstance(data, list) and len(data) > 0:
                    latest_data = data[-1]
                    
                    # Try different ATR key formats
                    atr_keys = ['ATR_14', 'ATR14', 'ATR', 'atr_14', 'atr14', 'atr']
                    for key in atr_keys:
                        if key in latest_data and latest_data[key] is not None:
                            atr = float(latest_data[key])
                            logger.info(f"📊 {symbol} ATR loaded: {atr:.5f} from {timeframe}")
                            return atr
            
            logger.warning(f"⚠️ {symbol} ATR not found in indicator data")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error loading ATR for {symbol}: {e}")
            return None
    
    def load_fibonacci_data(self, symbol: str) -> Optional[Dict]:
        """Load latest Fibonacci data from smallest available timeframe"""
        try:
            # Priority order for timeframes (SMALLEST first for tighter levels)
            timeframes = ['M15', 'M30', 'H1', 'H4']
            
            for timeframe in timeframes:
                file_path = f"indicator_output/{symbol}_{timeframe}_indicators.json"
                
                if os.path.exists(file_path):
                    logger.debug(f"📐 Loading Fibonacci data from {file_path}")
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Get latest data (last element - most recent candle)
                    if isinstance(data, list) and len(data) > 0:
                        latest_data = data[-1]
                        
                        # Extract Fibonacci levels from latest candle
                        fib_data = {}
                        for key, value in latest_data.items():
                            if key.startswith('fib_') and value is not None and key != 'fib_signal':
                                # Keep original key format
                                fib_data[key] = float(value)
                        
                        if fib_data:
                            logger.info(f"📐 Loaded {len(fib_data)} Fibonacci levels from {symbol}_{timeframe} (smallest timeframe - tighter levels)")
                            return fib_data
            
            logger.warning(f"⚠️ No Fibonacci data files found for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error loading Fibonacci data for {symbol}: {e}")
            return None
    
    def get_realtime_fibonacci_prices(self, position: Dict) -> List[float]:
        """Get real-time Fibonacci DCA price levels with ATR-based minimum distance"""
        try:
            symbol = position['symbol']
            position_type = position['type']
            entry_price = position['price_open']
            
            # 🎯 LOAD ATR FROM INDICATOR DATA
            atr_value = self.load_atr_for_symbol(symbol)
            
            # Get DCA mode and settings
            dca_mode = self.risk_settings.get('dca_mode', 'fibo_levels')
            dca_multiplier = self.risk_settings.get('dca_atr_multiplier', 1.5)
            dca_distance_pips = self.risk_settings.get('dca_distance_pips', 20)
            
            # Calculate minimum distance - use ATR for Fibonacci mode too
            # Fibonacci mode should also respect ATR-based minimum spacing
            if atr_value:
                # Use ATR * multiplier as minimum distance between DCA levels
                min_distance = atr_value * dca_multiplier
                logger.info(f"📐 {symbol} Using ATR-based min distance: {atr_value:.5f} * {dca_multiplier} = {min_distance:.5f}")
            else:
                # Fallback to configured pips or 20 pips default
                symbol_upper = symbol.upper()
                if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
                    pip_value = 0.1
                elif any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'SOL', 'ADA', 'BNB', 'LTC']):
                    pip_value = 1.0
                elif 'JPY' in symbol_upper:
                    pip_value = 0.01
                else:
                    pip_value = 0.0001
                
                min_distance = dca_distance_pips * pip_value
                logger.warning(f"⚠️ {symbol} ATR not available, using {dca_distance_pips} pips = {min_distance:.5f}")
            
            # Load latest Fibonacci data from indicator
            fibo_data = self.load_fibonacci_data(symbol)
            if not fibo_data:
                logger.debug(f"📐 No real-time Fibonacci data for {symbol} - using calculated levels")
                calc_levels = self.calculate_fibonacci_dca_levels(position)
                return [level['target_price'] for level in calc_levels]
            
            # Get all available Fibonacci levels from indicator data (sorted)
            all_fib_levels = []
            for key, value in fibo_data.items():
                if key.startswith('fib_') and key != 'fib_signal' and value is not None:
                    # Extract percentage from key
                    if key == 'fib_23.6' or key == 'fib_236':
                        percent = 23.6
                    elif key == 'fib_38.2' or key == 'fib_382':
                        percent = 38.2
                    elif key == 'fib_50.0' or key == 'fib_500':
                        percent = 50.0
                    elif key == 'fib_61.8' or key == 'fib_618':
                        percent = 61.8
                    elif key == 'fib_78.6' or key == 'fib_786':
                        percent = 78.6
                    elif key == 'fib_0.0':
                        percent = 0.0
                    elif key == 'fib_100.0':
                        percent = 100.0
                    else:
                        continue
                    
                    all_fib_levels.append({
                        'percent': percent,
                        'price': float(value),
                        'distance_from_entry': abs(float(value) - entry_price)
                    })
            
            # Filter levels that are in DCA direction and meet minimum distance
            valid_levels = []
            for level in all_fib_levels:
                price = level['price']
                distance = level['distance_from_entry']
                
                # Check DCA direction based on position type
                # BUY position: DCA khi giá giảm -> mua thêm ở mức thấp hơn entry
                # SELL position: DCA khi giá tăng -> bán thêm ở mức cao hơn entry
                if position_type == 0:  # BUY position
                    if price < entry_price and distance >= min_distance:
                        valid_levels.append(level)
                else:  # SELL position
                    if price > entry_price and distance >= min_distance:
                        valid_levels.append(level)
            
            # Sort by distance from entry (closest first)
            valid_levels.sort(key=lambda x: x['distance_from_entry'])
            
            if not valid_levels:
                logger.warning(f"⚠️ {symbol} No valid Fibonacci levels found meeting ATR-based minimum distance ({min_distance:.5f})")
                return []
            
            # Get configured Fibonacci sequence
            fibo_levels_cfg = self.risk_settings.get('dca_fibo_levels', '23.6,38.2,50,61.8,78.6')
            if isinstance(fibo_levels_cfg, str):
                config_levels_str = fibo_levels_cfg.split(',')
            else:
                config_levels_str = [str(x) for x in fibo_levels_cfg]
            config_levels = [float(level.strip()) for level in config_levels_str if level.strip()]
            
            # Get start level index (ensure it's an integer)
            start_level_raw = self.risk_settings.get('dca_fibo_start_level', 0)
            try:
                start_level = int(start_level_raw)
            except (ValueError, TypeError):
                start_level = 0
            
            # Try to use configured sequence first
            target_sequence = config_levels[start_level:] if start_level < len(config_levels) else config_levels
            dca_prices = []
            
            logger.info(f"📐 {symbol} Attempting DCA sequence: {target_sequence} (start index: {start_level})")
            
            for i, target_percent in enumerate(target_sequence):
                found_level = None
                
                # Try to find exact match first
                for level in valid_levels:
                    if abs(level['percent'] - target_percent) < 0.1:  # Close match
                        found_level = level
                        break
                
                # If no exact match, find nearest valid level for DCA1 only
                if found_level is None and i == 0:
                    logger.info(f"📐 {symbol} Target {target_percent}% not available, finding nearest valid level")
                    if valid_levels:
                        found_level = valid_levels[0]  # Closest to entry
                        logger.info(f"📐 {symbol} Using nearest level: {found_level['percent']}% at {found_level['price']:.5f}")
                
                if found_level:
                    dca_prices.append(found_level['price'])
                    # Remove used level to avoid duplicates
                    if found_level in valid_levels:
                        valid_levels.remove(found_level)
                
                # Limit to max 5 DCA levels
                if len(dca_prices) >= 5:
                    break
            
            if dca_prices:
                logger.info(f"📐 {symbol} Final DCA levels: {[round(p, 5) for p in dca_prices]} (ATR-based distance: {min_distance:.5f})")
                return dca_prices
            else:
                logger.warning(f"⚠️ {symbol} No suitable DCA levels found")
                return []
            
        except Exception as e:
            logger.error(f"❌ Error getting real-time Fibonacci prices: {e}")
        
        # Fallback to calculated levels
        logger.debug(f"📐 Using calculated Fibonacci levels for {symbol}")
        calc_levels = self.calculate_fibonacci_dca_levels(position)
        return [level['target_price'] for level in calc_levels]
    
    def calculate_fibonacci_dca_levels(self, position: Dict) -> List[Dict]:
        """Calculate Fibonacci DCA levels for position using REAL swing high/low from indicator"""
        try:
            symbol = position['symbol']
            entry_price = position['price_open']
            current_price = position['price_current']
            position_type = position['type']  # 0=BUY, 1=SELL
            
            # Get DCA settings - ensure start_level is an integer
            start_level_raw = self.risk_settings.get('dca_fibo_start_level', 0)
            try:
                start_level = int(start_level_raw)
            except (ValueError, TypeError):
                start_level = 0
                
            max_levels = self.risk_settings.get('max_dca_levels', 3)
            fibo_levels_cfg = self.risk_settings.get('dca_fibo_levels', '23.6,38.2,50,61.8,78.6')
            
            # Parse configured Fibonacci levels
            if isinstance(fibo_levels_cfg, str):
                cfg_parts = [p.strip() for p in fibo_levels_cfg.split(',') if p.strip()]
            else:
                cfg_parts = [str(p) for p in fibo_levels_cfg]
            
            fib_sequence = []
            for p in cfg_parts:
                try:
                    fib_sequence.append(float(p))
                except ValueError:
                    continue
            
            if not fib_sequence:
                fib_sequence = [23.6, 38.2, 50.0, 61.8, 78.6]
            
            # Apply start level and max levels
            if start_level < len(fib_sequence):
                adjusted_sequence = fib_sequence[start_level:][:max_levels]
            else:
                adjusted_sequence = fib_sequence[:max_levels]
            
            logger.info(f"📐 {symbol} Calculating DCA levels: sequence={adjusted_sequence}, start_idx={start_level}, max={max_levels}")
            
            # 🔧 FIX: Load swing high/low from indicator file (fib_0.0 = low, fib_100.0 = high)
            fibo_data = self.load_fibonacci_data(symbol)
            if fibo_data:
                swing_low = fibo_data.get('fib_0.0', None)
                swing_high = fibo_data.get('fib_100.0', None)
                if swing_low and swing_high:
                    logger.info(f"📐 {symbol} Using indicator swing: Low={swing_low:.5f}, High={swing_high:.5f}")
                else:
                    swing_low = swing_high = None
            else:
                swing_low = swing_high = None
            
            # Fallback: estimate swing range from entry price if indicator data not available
            if not swing_low or not swing_high:
                symbol_upper = symbol.upper()
                if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
                    swing_range_pct = 0.02  # 2% range for gold
                elif any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'SOL', 'ADA', 'BNB', 'LTC']):
                    swing_range_pct = 0.05  # 5% range for crypto
                elif 'JPY' in symbol_upper:
                    swing_range_pct = 0.015  # 1.5% for JPY pairs
                else:
                    swing_range_pct = 0.01  # 1% for forex
                
                swing_high = entry_price * (1 + swing_range_pct)
                swing_low = entry_price * (1 - swing_range_pct)
                logger.warning(f"⚠️ {symbol} No indicator data, estimating swing: Low={swing_low:.5f}, High={swing_high:.5f}")
            
            swing_range = swing_high - swing_low
            
            dca_levels = []
            for i, target_percent in enumerate(adjusted_sequence):
                # 🔧 FIX: Calculate Fibonacci retracement correctly
                # Fibonacci retracement: price = high - (high - low) * fib_level
                # For DCA: we want levels between entry and opposite direction
                fib_factor = target_percent / 100.0
                
                if position_type == 0:  # BUY position - DCA below entry
                    # Mua thêm khi giá giảm về các mức Fibonacci thấp hơn
                    target_price = swing_high - (swing_range * fib_factor)
                    # Chỉ lấy các mức dưới entry price
                    if target_price >= entry_price:
                        # Skip levels at or above entry
                        continue
                else:  # SELL position - DCA above entry  
                    # Bán thêm khi giá tăng về các mức Fibonacci cao hơn
                    target_price = swing_low + (swing_range * fib_factor)
                    # Chỉ lấy các mức trên entry price
                    if target_price <= entry_price:
                        # Skip levels at or below entry
                        continue
                
                # Check if should trigger
                should_trigger = False
                if position_type == 0:  # BUY position
                    should_trigger = current_price <= target_price
                else:  # SELL position
                    should_trigger = current_price >= target_price
                
                dca_levels.append({
                    'level': len(dca_levels) + 1,  # Sequential level number
                    'fibonacci_percent': target_percent,
                    'target_price': target_price,
                    'should_trigger': should_trigger,
                    'volume': self.calculate_dca_volume(position, len(dca_levels) + 1)
                })
                
                logger.info(f"[FIBO_DCA] {symbol} {'BUY' if position_type == 0 else 'SELL'} Level {len(dca_levels)}: {target_percent}% -> target {target_price:.5f} (swing: {swing_low:.5f}-{swing_high:.5f})")
            
            return dca_levels
            
        except Exception as e:
            logger.error(f"❌ Error calculating Fibonacci DCA levels: {e}")
            return []
    


    def calculate_dca_volume(self, position: Dict, level: int) -> float:
        """Calculate DCA volume"""
        try:
            base_volume = position['volume']
            multiplier = self.risk_settings.get('dca_volume_multiplier', 1.5)
            
            # Volume increases with level
            dca_volume = base_volume * (multiplier ** (level - 1))
            
            # Round to minimum step
            return round(dca_volume, 2)
            
        except Exception as e:
            logger.error(f"❌ Error calculating DCA volume: {e}")
            return position['volume']
    
    def check_existing_dca_orders(self, symbol: str, level: int) -> bool:
        """Check if DCA order already exists for this level with enhanced duplicate prevention"""
        try:
            dca_comment_pattern = f"GPT_20B|DCA{level}"
            current_time = datetime.now()
            
            # Check positions with enhanced matching
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                for pos in positions:
                    if pos.comment and dca_comment_pattern in pos.comment:
                        logger.info(f"🔍 Found existing DCA{level} position for {symbol}: {pos.ticket}")
                        return True
            
            # Check pending orders with enhanced matching
            orders = mt5.orders_get(symbol=symbol)
            if orders:
                for order in orders:
                    if order.comment and dca_comment_pattern in order.comment:
                        logger.info(f"🔍 Found existing DCA{level} pending order for {symbol}: {order.ticket}")
                        return True
            
            # Additional check: Look for recent DCA orders (last 30 seconds) to prevent race conditions
            recent_threshold = current_time - timedelta(seconds=30)
            
            # Get all recent orders from history (last 1 minute)
            from_time = current_time - timedelta(minutes=1)
            deals = mt5.history_deals_get(
                from_date=from_time,
                to_date=current_time,
                symbol=symbol
            )
            
            if deals:
                for deal in deals:
                    if deal.comment and dca_comment_pattern in deal.comment:
                        deal_time = datetime.fromtimestamp(deal.time)
                        if deal_time >= recent_threshold:
                            logger.warning(f"⚠️ Found recent DCA{level} deal for {symbol} at {deal_time} - preventing duplicate")
                            return True
                        
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking existing DCA orders: {e}")
            return True  # Conservative: assume exists to prevent duplicates
    
    def execute_market_dca(self, position: Dict, dca_level: Dict) -> bool:
        """Execute DCA using Market Order when price touches Fibonacci level"""
        try:
            symbol = position['symbol']
            volume = dca_level['volume']
            level = dca_level['level']
            position_type = position['type']
            entry_price = position['price_open']
            target_price = dca_level['target_price']
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.error(f"❌ Cannot get tick for {symbol}")
                return False
            
            current_price = tick.ask if position_type == 0 else tick.bid
            
            # Validate DCA direction before executing
            # BUY position: DCA khi giá giảm xuống dưới entry
            # SELL position: DCA khi giá tăng lên trên entry
            if position_type == 0:  # BUY position
                if target_price >= entry_price:
                    logger.warning(f"⚠️ {symbol} BUY DCA{level} target {target_price:.5f} not below entry {entry_price:.5f}")
                    return False
                price = tick.ask
                order_type = mt5.ORDER_TYPE_BUY
            else:  # SELL position
                if target_price <= entry_price:
                    logger.warning(f"⚠️ {symbol} SELL DCA{level} target {target_price:.5f} not above entry {entry_price:.5f}")
                    return False
                price = tick.bid
                order_type = mt5.ORDER_TYPE_SELL
            
            # Calculate SL/TP
            sl, tp = self.calculate_dca_sltp(symbol, position_type, price)
            
            # Create request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234567,  # Different magic number from entry
                "comment": f"GPT_20B|DCA{level}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Initialize risk validator if needed
            if not self.initialize_risk_validator():
                logger.error(f"❌ Cannot execute DCA for {symbol}: Risk validator not available")
                return False
            
            # Check volume limits through risk validator
            validation = self.risk_validator._check_position_limits(symbol)
            if not validation['valid']:
                logger.warning(f"⚠️ DCA blocked for {symbol} Level {level}: {validation['reason']}")
                return False
            
            # Send order through MT5 (order executor doesn't have direct execution method)
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"❌ Market DCA failed for {symbol} Level {level}: {result.retcode} - {result.comment}")
                return False
            
            logger.info(f"✅ Market DCA executed: {symbol} Level {level} @ {price:.5f} | Volume: {volume} | Ticket: {result.order}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error executing market DCA: {e}")
            return False
    
    def check_and_adjust_pending_orders(self, position: Dict) -> bool:
        """Check and adjust existing pending DCA orders with real-time Fibonacci levels"""
        try:
            symbol = position['symbol']
            
            # Get real-time Fibonacci levels
            realtime_levels = self.get_realtime_fibonacci_prices(position)
            if not realtime_levels:
                logger.debug(f"📐 No Fibonacci levels to adjust for {symbol}")
                return False
            
            # Get existing pending orders for this symbol
            orders = mt5.orders_get(symbol=symbol)
            if not orders:
                return True  # No orders to adjust
            
            adjustments_made = 0
            
            for order in orders:
                # Only adjust DCA orders (check comment)
                if not order.comment or "DCA" not in order.comment:
                    continue
                
                # Extract DCA level from comment (e.g., "GPT_AI20B|DCA1")
                try:
                    level_num = int(order.comment.split("DCA")[1])
                    if level_num > len(realtime_levels):
                        continue
                    
                    # Get target price for this level (array is 0-indexed)
                    target_price = realtime_levels[level_num - 1]
                    current_order_price = order.price_open
                    
                    # Check if adjustment is needed (significant price difference)
                    price_diff = abs(target_price - current_order_price)
                    min_adjustment = 0.00010  # Minimum 1 pip adjustment
                    
                    if price_diff > min_adjustment:
                        # Modify the existing order
                        request = {
                            "action": mt5.TRADE_ACTION_MODIFY,
                            "order": order.ticket,
                            "price": target_price,
                            "sl": order.sl,
                            "tp": order.tp,
                        }
                        
                        result = mt5.order_send(request)
                        
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"🔄 Adjusted {symbol} DCA{level_num}: {current_order_price:.5f} → {target_price:.5f}")
                            adjustments_made += 1
                        else:
                            logger.warning(f"⚠️ Failed to adjust {symbol} DCA{level_num}: {result.comment}")
                    
                except (ValueError, IndexError) as e:
                    logger.debug(f"Could not parse DCA level from comment {order.comment}: {e}")
                    continue
            
            if adjustments_made > 0:
                logger.info(f"✅ Adjusted {adjustments_made} pending orders for {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adjusting pending orders: {e}")
            return False
    
    def execute_pending_dca(self, position: Dict, dca_level: Dict) -> bool:
        """Execute DCA using Pending Limit Order with calculated Fibonacci levels"""
        try:
            symbol = position['symbol']
            volume = dca_level['volume']
            level = dca_level['level']
            position_type = position['type']
            entry_price = position['price_open']
            
            # 🔧 FIX: Use target_price directly from calculated dca_level dict
            # Don't override with get_realtime_fibonacci_prices() which can cause duplicates
            target_price = dca_level.get('target_price')
            if not target_price:
                logger.warning(f"⚠️ No target price for {symbol} DCA{level}")
                return False
            
            logger.info(f"📐 Using calculated Fibonacci price for {symbol} DCA{level}: {target_price:.5f}")
            
            # Get current price for validation
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.error(f"❌ Cannot get tick for {symbol}")
                return False
            
            current_price = tick.ask if position_type == 0 else tick.bid
            
            # CRITICAL VALIDATION: Don't place DCA orders at or near entry price  
            price_diff_from_entry = abs(target_price - entry_price) / entry_price * 100
            min_distance_percent = 0.05  # Minimum 0.05% distance from entry (more lenient)
            
            if price_diff_from_entry < min_distance_percent:
                logger.warning(f"⚠️ {symbol} DCA{level} too close to entry: {target_price:.5f} vs entry {entry_price:.5f} ({price_diff_from_entry:.2f}%)")
                return False
            
            # Validate pending order logic for DCA direction
            # BUY position: DCA khi giá giảm -> đặt BUY LIMIT dưới giá hiện tại
            # SELL position: DCA khi giá tăng -> đặt SELL LIMIT trên giá hiện tại
            if position_type == 0:  # BUY position - DCA should be BELOW entry and current price
                if target_price >= entry_price:
                    logger.warning(f"⚠️ {symbol} BUY DCA{level} price {target_price:.5f} not below entry {entry_price:.5f}")
                    return False
                if target_price >= current_price:
                    logger.debug(f"⚠️ {symbol} BUY DCA{level} pending price {target_price:.5f} not below current {current_price:.5f}")
                    return False
                order_type = mt5.ORDER_TYPE_BUY_LIMIT
            else:  # SELL position - DCA should be ABOVE entry (sell at higher price)
                if target_price <= entry_price:
                    logger.warning(f"⚠️ {symbol} SELL DCA{level} price {target_price:.5f} not above entry {entry_price:.5f}")
                    return False
                if target_price <= current_price:
                    logger.debug(f"⚠️ {symbol} SELL DCA{level} pending price {target_price:.5f} not above current {current_price:.5f}")
                    return False
                order_type = mt5.ORDER_TYPE_SELL_LIMIT
            
            # Calculate SL/TP
            sl, tp = self.calculate_dca_sltp(symbol, position_type, target_price)
            
            # Create request
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": target_price,
                "sl": sl,
                "tp": tp,
                "magic": 234567,  # Different magic number from entry
                "comment": f"GPT_20B|DCA{level}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            
            # Initialize risk validator if needed
            if not self.initialize_risk_validator():
                logger.error(f"❌ Cannot execute pending DCA for {symbol}: Risk validator not available")
                return False
            
            # Check volume limits through risk validator
            validation = self.risk_validator._check_position_limits(symbol)
            if not validation['valid']:
                logger.warning(f"⚠️ Pending DCA blocked for {symbol} Level {level}: {validation['reason']}")
                return False
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"❌ Pending DCA failed for {symbol} Level {level}: {result.retcode} - {result.comment}")
                return False
            
            logger.info(f"✅ Pending DCA placed: {symbol} Level {level} @ {target_price:.5f} | Volume: {volume} | Order: {result.order}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error executing pending DCA: {e}")
            return False
    
    def calculate_dca_sltp(self, symbol: str, position_type: int, entry_price: float) -> tuple:
        """Calculate SL/TP for DCA order"""
        try:
            # Get SL/TP settings
            sl_pips = self.risk_settings.get('default_sl_pips', 50)
            tp_pips = self.risk_settings.get('default_tp_pips', 100)
            
            # Calculate pip value based on symbol type
            symbol_upper = symbol.upper()
            if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
                pip_value = 0.1
            elif any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'SOL', 'ADA']):
                pip_value = 0.1  # Fixed: Was 1.0, causing S/L too close to entry
            elif 'JPY' in symbol_upper:
                pip_value = 0.01
            else:
                pip_value = 0.0001
            
            # Calculate SL/TP based on position type
            if position_type == 0:  # BUY position
                sl = entry_price - (sl_pips * pip_value)
                tp = entry_price + (tp_pips * pip_value)
            else:  # SELL position
                sl = entry_price + (sl_pips * pip_value)
                tp = entry_price - (tp_pips * pip_value)
            
            return sl, tp
            
        except Exception as e:
            logger.error(f"❌ Error calculating SL/TP: {e}")
            return 0.0, 0.0
    
    def process_dca_opportunities(self):
        """Process DCA opportunities"""
        try:
            positions = self.get_current_positions()
            if not positions:
                logger.debug("🔍 No entry positions found")
                return
            
            execution_mode = self.get_execution_mode()
            logger.info(f"🔍 Scanning {len(positions)} positions - Mode: {execution_mode}")
            
            opportunities_found = 0
            
            for position in positions:
                symbol = position['symbol']
                
                # Calculate Fibonacci DCA levels for this position
                dca_levels = self.calculate_fibonacci_dca_levels(position)
                
                if not dca_levels:
                    continue
                
                for dca_level in dca_levels:
                    level = dca_level['level']
                    
                    # Check if DCA order already exists for this level
                    if self.check_existing_dca_orders(symbol, level):
                        continue
                    

                    
                    # Process based on execution mode
                    if "Market" in execution_mode or "Chạm Mức" in execution_mode:
                        # Market mode: execute only when price has reached level
                        if dca_level['should_trigger']:

                                
                            if self.execute_market_dca(position, dca_level):
                                opportunities_found += 1
                                logger.info(f"🔄 Market DCA Opportunity: {symbol} Level {level} executed")
                    
                    elif "Pending" in execution_mode or "Đặt Lệnh Chờ" in execution_mode:
                        # Pending mode: place limit orders at levels

                            
                        if self.execute_pending_dca(position, dca_level):
                            opportunities_found += 1
                            logger.info(f"🔄 Pending DCA Opportunity: {symbol} Level {level} placed")
            
            if opportunities_found > 0:
                logger.info(f"✅ Processed {opportunities_found} DCA opportunities")
            
        except Exception as e:
            logger.error(f"❌ Error processing DCA opportunities: {e}")
    
    def run_service(self):
        """Run main DCA service"""
        logger.info("🚀 Starting Fibonacci DCA Service...")
        
        # Check conditions
        if not self.should_run_service():
            logger.error("❌ DCA Service conditions not met")
            return
        
        # Initialize MT5
        if not self.initialize_mt5():
            logger.error("❌ Cannot initialize MT5")
            return
        
        execution_mode = self.get_execution_mode()
        logger.info(f"✅ Fibonacci DCA Service started - Mode: {execution_mode}")
        
        self.running = True
        
        try:
            while self.running:
                # Check for trigger files first
                if self.process_fibonacci_trigger():
                    logger.info("📐 Processed Fibonacci trigger from comprehensive_aggregator")
                
                # Reload settings periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    self.load_risk_settings()
                
                self.process_dca_opportunities()
                time.sleep(self.scan_interval)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Service stopped by user")
        except Exception as e:
            logger.error(f"❌ Service error: {e}")
        finally:
            self.stop_service()
    
    def stop_service(self):
        """Stop service"""
        self.running = False
        if mt5:
            mt5.shutdown()
        logger.info("✅ Fibonacci DCA Service stopped")

def main():
    """Main function"""
    try:
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Create and start service
        service = FibonacciDCAService()
        service.run_service()
    except Exception as e:
        logger.error(f"❌ Main error: {e}")

if __name__ == "__main__":
    main()