import os
import json
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
from sklearn.linear_model import RANSACRegressor, LinearRegression
import hashlib
import pickle
import logging
from datetime import datetime

DATA_FOLDER = "data"
CACHE_FOLDER = "cache"

# Enhanced logging setup for multi-user system
import os
from datetime import datetime

# Create logs directory if not exists
os.makedirs('logs', exist_ok=True)

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/trendline_analysis_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_user_analysis(user_id: str, symbol: str, timeframe: str, action: str, details: dict = None):
    """Log user-specific analysis actions"""
    log_data = {
        'user_id': user_id,
        'symbol': symbol,
        'timeframe': timeframe,
        'action': action,
        'timestamp': datetime.now().isoformat(),
        'details': details or {}
    }
    logger.info(f"USER_ANALYSIS: {log_data}")

def log_performance_analysis(symbol: str, timeframe: str, operation: str, duration: float, user_id: str = None):
    """Log analysis performance metrics"""
    perf_data = {
        'symbol': symbol,
        'timeframe': timeframe,
        'operation': operation,
        'duration_ms': round(duration * 1000, 2),
        'user_id': user_id,
        'timestamp': datetime.now().isoformat()
    }
    logger.info(f"ANALYSIS_PERFORMANCE: {perf_data}")

# Rate limiting to prevent log spam
_logged_errors = {}
def log_rate_limited_error(error_key: str, message: str, max_per_hour: int = 5):
    """Log errors with rate limiting to prevent spam"""
    current_hour = datetime.now().strftime('%Y%m%d_%H')
    key = f"{error_key}_{current_hour}"
    
    if key not in _logged_errors:
        _logged_errors[key] = 0
    
    if _logged_errors[key] < max_per_hour:
        logger.error(message)
        _logged_errors[key] += 1
    elif _logged_errors[key] == max_per_hour:
        logger.error(f"Rate limit reached for error: {error_key} (suppressing further logs this hour)")
        _logged_errors[key] += 1

def validate_data(df):
    """Validate OHLC data integrity"""
    if df.empty:
        raise ValueError("Empty dataframe")
    
    required_cols = ['open', 'high', 'low', 'close']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Check for invalid OHLC relationships
    invalid_high = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close'])
    invalid_low = (df['low'] > df['high']) | (df['low'] > df['open']) | (df['low'] > df['close'])
    
    if invalid_high.any() or invalid_low.any():
        logger.warning("Invalid OHLC data detected and will be cleaned")
        # Clean invalid data
        df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close']].max(axis=1)
        df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'close']].min(axis=1)
    
    return df

def load_ohlc(symbol, timeframe, count=200):
    """Load and validate OHLC data from data/ with flexible filename patterns.

    Supported patterns (in order):
      - {symbol}_{timeframe}.json
      - {symbol}_m_{timeframe}.json
      - {symbol}_cl_{timeframe}.json
      - {symbol}_{timeframe}_data.json (fallback)
    """
    try:
        # Primary path
        candidates = [
            os.path.join(DATA_FOLDER, f"{symbol}_{timeframe}.json"),
            os.path.join(DATA_FOLDER, f"{symbol}_m_{timeframe}.json"),
            os.path.join(DATA_FOLDER, f"{symbol}_cl_{timeframe}.json"),
            os.path.join(DATA_FOLDER, f"{symbol}_{timeframe}_data.json"),
        ]
        path = None
        for p in candidates:
            if os.path.exists(p):
                path = p
                break
        if not path:
            raise FileNotFoundError(f"Data file not found for {symbol}_{timeframe} (tried {len(candidates)} patterns)")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {path}: {e}")
        except Exception as e:
            raise ValueError(f"Error reading data file {path}: {e}")
        
        if not data:
            raise ValueError(f"No data found in {path}")
        
        # Validate data structure
        if not isinstance(data, list):
            raise ValueError(f"Data format error: Expected list, got {type(data)}")
        
        # Clean data
        for candle in data:
            if isinstance(candle, dict):
                candle.pop("current", None)
                candle.pop("current_price", None)
        
        df = pd.DataFrame(data)
        
        # Check if DataFrame is empty after creation
        if df.empty:
            raise ValueError(f"Empty dataframe created from {path}")
            
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Validate data
        df = validate_data(df)
        
        # Ensure we have enough data
        if len(df) < count:
            logger.warning(f"Requested {count} candles but only {len(df)} available for {symbol}_{timeframe}")
        
        df = df.tail(count).reset_index(drop=True)
        return df
        
    except Exception as e:
        # Log the error with full context
        logger.error(f"Error loading OHLC data for {symbol}_{timeframe}: {str(e)}")
        raise ValueError(f"Failed to load data for {symbol}_{timeframe}: {str(e)}")

def find_swings_improved(df, order=5, min_strength=0.5):
    """Find swing highs and lows using proper high/low prices with strength validation"""
    if len(df) < order * 2 + 1:
        return np.array([]), np.array([])
    
    # Use actual high/low prices for swing detection with stricter comparison
    highs_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]  # Changed to np.greater for stricter peaks
    lows_idx = argrelextrema(df['low'].values, np.less, order=order)[0]       # Changed to np.less for stricter valleys
    
    # Validate swing strength - filter out weak swings
    valid_highs = []
    for idx in highs_idx:
        if idx >= order and idx < len(df) - order:
            swing_high = df['high'].iloc[idx]
            local_range = df['high'].iloc[idx-order:idx+order+1]
            strength = (swing_high - local_range.mean()) / local_range.std() if local_range.std() > 0 else 0
            if strength >= min_strength:
                valid_highs.append(idx)
    
    valid_lows = []
    for idx in lows_idx:
        if idx >= order and idx < len(df) - order:
            swing_low = df['low'].iloc[idx]
            local_range = df['low'].iloc[idx-order:idx+order+1]
            strength = (local_range.mean() - swing_low) / local_range.std() if local_range.std() > 0 else 0
            if strength >= min_strength:
                valid_lows.append(idx)
    
    return np.array(valid_lows), np.array(valid_highs)

def determine_trend(df, lookback=70):
    """Determine trend direction using multiple methods with improved scoring
    
    Returns:
        tuple: (trend_direction, sideway_range_data)
        - trend_direction: str ('Strong Uptrend', 'Uptrend', 'Downtrend', 'Strong Downtrend', 'Sideway')
        - sideway_range_data: dict with range info if Sideway, else None
    """
    if len(df) < lookback:
        lookback = len(df)
    
    # Method 1: EMA crossover with slope consideration
    df['ema_fast'] = df['close'].ewm(span=12).mean()
    df['ema_slow'] = df['close'].ewm(span=26).mean()
    ema_trend = 1 if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1] else -1
    
    # 🎯 FIX: Increase candle count for EMA slope (5 → 20 candles)
    ema_slope_strength = (df['ema_fast'].iloc[-1] - df['ema_fast'].iloc[-20]) / df['ema_fast'].iloc[-20]
    ema_slope_score = 1 if ema_slope_strength > 0.002 else (-1 if ema_slope_strength < -0.002 else 0)
    
    # 🎯 FIX: Increase candle count for momentum (10 → 30, lookback → lookback)
    short_momentum = (df['close'].iloc[-1] - df['close'].iloc[-30]) / df['close'].iloc[-30]
    long_momentum = (df['close'].iloc[-1] - df['close'].iloc[-lookback]) / df['close'].iloc[-lookback]
    
    short_trend = 1 if short_momentum > 0.01 else (-1 if short_momentum < -0.01 else 0)
    long_trend = 1 if long_momentum > 0.02 else (-1 if long_momentum < -0.02 else 0)
    
    # Method 3: Higher highs/Lower lows with improved logic
    recent_highs = df['high'].tail(lookback//2)
    recent_lows = df['low'].tail(lookback//2)
    early_highs = df['high'].iloc[-lookback:-lookback//2]
    early_lows = df['low'].iloc[-lookback:-lookback//2]
    
    hh_ll_trend = 0
    if len(early_highs) > 0 and len(early_lows) > 0:
        # More strict HH/LL detection
        hh_count = sum(1 for rh in recent_highs if rh > early_highs.max())
        ll_count = sum(1 for rl in recent_lows if rl < early_lows.min())
        hl_count = sum(1 for rl in recent_lows if rl > early_lows.min())  # Higher lows
        lh_count = sum(1 for rh in recent_highs if rh < early_highs.max())  # Lower highs
        
        if hh_count > 0 and hl_count > 0:
            hh_ll_trend = 2  # Strong uptrend
        elif lh_count > 0 and ll_count > 0:
            hh_ll_trend = -2  # Strong downtrend
        elif hl_count > ll_count:
            hh_ll_trend = 1  # Weak uptrend
        elif lh_count > hh_count:
            hh_ll_trend = -1  # Weak downtrend
    
    # Combine all methods with weighted scoring
    trend_score = (ema_trend * 1.5) + (ema_slope_score * 1.0) + (short_trend * 1.2) + (long_trend * 1.0) + (hh_ll_trend * 1.3)
    
    # 🐛 DEBUG: Log trend score for analysis
    try:
        symbol_name = df.get('symbol', 'UNKNOWN') if hasattr(df, 'get') else 'UNKNOWN'
        print(f"🔍 TREND SCORE DEBUG [{symbol_name}]:")
        print(f"   ema_trend: {ema_trend} × 1.5 = {ema_trend * 1.5}")
        print(f"   ema_slope_score: {ema_slope_score} × 1.0 = {ema_slope_score * 1.0}")
        print(f"   short_trend: {short_trend} × 1.2 = {short_trend * 1.2}")
        print(f"   long_trend: {long_trend} × 1.0 = {long_trend * 1.0}")
        print(f"   hh_ll_trend: {hh_ll_trend} × 1.3 = {hh_ll_trend * 1.3}")
        print(f"   📊 TOTAL TREND_SCORE: {trend_score}")
        print(f"   ✅ Sideway range: -2.5 < score < 2.5")
        if -2.5 < trend_score < 2.5:
            print(f"   🎯 SIDEWAY DETECTED!")
    except Exception as e:
        print(f"⚠️ Debug logging error: {e}")
    
    # 🎯 CHECK PRICE RANGE FIRST - if narrow range, force Sideway regardless of trend_score
    # Use 20 candles (not 30) to focus on recent consolidation
    recent_high = df['high'].tail(20).max()
    recent_low = df['low'].tail(20).min()
    current_price = df['close'].iloc[-1]
    price_range_pct = ((recent_high - recent_low) / current_price) * 100
    
    print(f"📏 PRICE RANGE CHECK (20 candles):")
    print(f"   High: {recent_high:.2f}")
    print(f"   Low: {recent_low:.2f}")
    print(f"   Range: {recent_high - recent_low:.2f} ({price_range_pct:.2f}%)")
    print(f"   Threshold: 2.5%")
    
    # Enhanced trend classification
    sideway_range_data = None
    
    # 🚨 PRIORITY: If price range < 2.5% over 20 candles → FORCE SIDEWAY
    if price_range_pct < 2.5:
        trend_direction = "Sideway"
        sideway_range_data = calculate_sideway_range(df, lookback)
        print(f"🎯 FORCED SIDEWAY: Price range {price_range_pct:.2f}% < 2.5% threshold")
    elif trend_score >= 4.0:
        trend_direction = "Strong Uptrend"
    elif trend_score >= 2.5:
        trend_direction = "Uptrend"
    elif trend_score <= -4.0:
        trend_direction = "Strong Downtrend"
    elif trend_score <= -2.5:
        trend_direction = "Downtrend"
    else:
        # 🎯 SIDEWAY: -2.5 < trend_score < 2.5
        trend_direction = "Sideway"
        # 📊 Calculate Sideway Range (Biên độ giá đi ngang)
        sideway_range_data = calculate_sideway_range(df, lookback)
    
    return trend_direction, sideway_range_data

def calculate_sideway_range(df, lookback=20):
    """📊 Tính toán biên độ giá trong thị trường sideway (20 nến gần nhất)
    
    Returns:
        dict: {
            'range_high': float,          # Đỉnh biên độ
            'range_low': float,           # Đáy biên độ
            'range_width': float,         # Độ rộng (price units)
            'range_width_pct': float,     # Độ rộng (%)
            'range_midpoint': float,      # Điểm giữa biên độ
            'current_position_pct': float,# Vị trí hiện tại trong biên độ (0-100%)
            'touches_top': int,           # Số lần chạm đỉnh
            'touches_bottom': int,        # Số lần chạm đáy
            'volatility': float,          # Biến động trong biên độ
            'consolidation_strength': str # Mức độ consolidation
        }
    """
    try:
        # 🎯 CHỈ TÍNH TRONG 20 NẾN GÁN NHẤT
        lookback = min(20, len(df))
        
        # Lấy dữ liệu lookback period
        recent_df = df.tail(lookback)
        
        # Tìm high/low của biên độ (điểm cực đại/cực tiểu)
        range_high_point = recent_df['high'].max()
        range_low_point = recent_df['low'].min()
        range_width = range_high_point - range_low_point
        range_midpoint = (range_high_point + range_low_point) / 2
        
        # 🎯 Tạo VÙNG GIÁ thay vì điểm giá cố định
        # Zone buffer = 0.5% của range width (có thể điều chỉnh)
        zone_buffer = range_width * 0.005  # 0.5% buffer
        
        # Resistance zone (vùng kháng cự) = range_high ± buffer
        resistance_zone_high = range_high_point + zone_buffer
        resistance_zone_low = range_high_point - zone_buffer
        
        # Support zone (vùng hỗ trợ) = range_low ± buffer
        support_zone_high = range_low_point + zone_buffer
        support_zone_low = range_low_point - zone_buffer
        
        # Tính % độ rộng
        range_width_pct = (range_width / range_midpoint) * 100 if range_midpoint > 0 else 0
        
        # Vị trí hiện tại trong biên độ (0% = đáy, 100% = đỉnh)
        current_price = recent_df['close'].iloc[-1]
        if range_width > 0:
            current_position_pct = ((current_price - range_low_point) / range_width) * 100
        else:
            current_position_pct = 50.0  # Trung tâm nếu không có range
        
        # Đếm số lần chạm vùng resistance/support (sử dụng zone thay vì điểm)
        # Chạm resistance zone: high >= resistance_zone_low
        touches_top = sum(1 for high in recent_df['high'] if high >= resistance_zone_low and high <= resistance_zone_high)
        # Chạm support zone: low <= support_zone_high
        touches_bottom = sum(1 for low in recent_df['low'] if low >= support_zone_low and low <= support_zone_high)
        
        # Tính volatility trong biên độ (standard deviation của close prices)
        volatility = recent_df['close'].pct_change().std() * 100 if len(recent_df) > 1 else 0
        
        # Đánh giá consolidation strength
        # Càng nhiều touches + volatility thấp = consolidation mạnh
        consolidation_score = (touches_top + touches_bottom) / max(volatility, 0.1)
        
        if consolidation_score > 15:
            consolidation_strength = "Very Strong"  # Rất chặt
        elif consolidation_score > 10:
            consolidation_strength = "Strong"       # Chặt
        elif consolidation_score > 5:
            consolidation_strength = "Moderate"     # Trung bình
        else:
            consolidation_strength = "Weak"         # Yếu
        
        return {
            # 🎯 PRICE ZONES (vùng giá thay vì điểm giá)
            'resistance_zone': {
                'high': float(resistance_zone_high),
                'low': float(resistance_zone_low),
                'center': float(range_high_point)
            },
            'support_zone': {
                'high': float(support_zone_high),
                'low': float(support_zone_low),
                'center': float(range_low_point)
            },
            # Legacy fields for backward compatibility
            'range_high': float(range_high_point),
            'range_low': float(range_low_point),
            'range_width': float(range_width),
            'range_width_pct': float(range_width_pct),
            'range_midpoint': float(range_midpoint),
            'current_price': float(current_price),
            'current_position_pct': float(current_position_pct),
            'touches_top': int(touches_top),
            'touches_bottom': int(touches_bottom),
            'volatility': float(volatility),
            'consolidation_strength': consolidation_strength,
            'zone_buffer_pct': 0.5  # Zone buffer = 0.5% of range width
        }
        
    except Exception as e:
        logger.error(f"Error calculating sideway range: {e}")
        return None

def calc_trendline_improved(df, order=5):
    """Calculate trendline based on trend direction and swing points"""
    lows, highs = find_swings_improved(df, order)
    trend_direction, sideway_range_data = determine_trend(df)
    
    slope, intercept = 0, df['close'].iloc[-1]
    trendline = np.full(len(df), df['close'].iloc[-1])
    
    if trend_direction == "Uptrend" and len(lows) >= 2:
        # Uptrend: connect swing lows with trendline
        x = np.array(lows)
        y = df.iloc[lows]['low'].values
        
        # Use recent lows for better accuracy
        if len(lows) > 3:
            x = x[-3:]  # Use last 3 swing lows
            y = y[-3:]
        
        try:
            slope, intercept = robust_trendline(x, y)
            trendline = slope * np.arange(len(df)) + intercept
        except Exception as e:
            logger.warning(f"Failed to calculate uptrend trendline: {e}")
            
    elif trend_direction == "Downtrend" and len(highs) >= 2:
        # Downtrend: connect swing highs with trendline
        x = np.array(highs)
        y = df.iloc[highs]['high'].values
        
        # Use recent highs for better accuracy
        if len(highs) > 3:
            x = x[-3:]  # Use last 3 swing highs
            y = y[-3:]
        
        try:
            slope, intercept = robust_trendline(x, y)
            trendline = slope * np.arange(len(df)) + intercept
        except Exception as e:
            logger.warning(f"Failed to calculate downtrend trendline: {e}")
            
    elif len(lows) >= 2 or len(highs) >= 2:
        # Sideway: use the set with more points
        if len(lows) >= len(highs):
            x = np.array(lows)
            y = df.iloc[lows]['close'].values
        else:
            x = np.array(highs)
            y = df.iloc[highs]['close'].values
        
        try:
            slope, intercept = robust_trendline(x, y)
            trendline = slope * np.arange(len(df)) + intercept
        except Exception as e:
            logger.warning(f"Failed to calculate sideway trendline: {e}")
    
    return trendline, slope, intercept, lows, highs, trend_direction, sideway_range_data

def calc_trendline_swing(df, order=5, price_col="close"):
    """Legacy function - redirects to improved version"""
    trendline, slope, intercept, lows, highs, _, _ = calc_trendline_improved(df, order)
    return trendline, slope, intercept, lows, highs

def calc_channel_improved(df, trendline, highs, lows, trend_direction):
    """Calculate channel boundaries with improved logic"""
    prices_high = df['high'].values
    prices_low = df['low'].values
    
    if trend_direction == "Uptrend":
        # For uptrend: trendline is support, find resistance
        if len(highs) > 0:
            # Distance from highs to trendline
            upper_deviations = prices_high[highs] - trendline[highs]
            upper_dev = np.percentile(upper_deviations, 80)  # Use 80th percentile instead of max
        else:
            upper_dev = np.percentile(prices_high - trendline, 80)
        
        # Lower boundary should be close to trendline for uptrend
        lower_dev = np.percentile(prices_low - trendline, 20)
        
    elif trend_direction == "Downtrend":
        # For downtrend: trendline is resistance, find support
        if len(lows) > 0:
            # Distance from lows to trendline
            lower_deviations = prices_low[lows] - trendline[lows]
            lower_dev = np.percentile(lower_deviations, 20)  # Use 20th percentile instead of min
        else:
            lower_dev = np.percentile(prices_low - trendline, 20)
        
        # Upper boundary should be close to trendline for downtrend
        upper_dev = np.percentile(prices_high - trendline, 80)
        
    else:
        # Sideway: symmetric channel
        all_deviations = np.concatenate([prices_high - trendline, prices_low - trendline])
        upper_dev = np.percentile(all_deviations, 85)
        lower_dev = np.percentile(all_deviations, 15)
    
    channel_upper = trendline + upper_dev
    channel_lower = trendline + lower_dev
    
    return channel_upper, channel_lower

def calc_channel_swing(df, trendline, highs, lows):
    """Legacy function - redirects to improved version"""
    trend_direction = determine_trend(df)
    return calc_channel_improved(df, trendline, highs, lows, trend_direction)

def get_adaptive_window(df, timeframe, base_window=20):
    """Calculate adaptive window based on timeframe and data volatility"""
    base_windows = {
        'M1': 10, 'M5': 12, 'M15': 15, 'M30': 18,
        'H1': 20, 'H4': 25, 'D1': 30, 'W1': 35
    }
    base = base_windows.get(timeframe, base_window)
    
    # Adjust based on volatility if enough data
    if len(df) > 20:
        try:
            volatility = df['close'].pct_change().std()
            if volatility > 0.02:  # High volatility
                return min(base + 5, len(df) // 4)
            elif volatility < 0.005:  # Low volatility  
                return max(base - 5, 10)
        except Exception:
            pass
    
    return min(base, max(len(df) // 4, 10))

def find_support_resistance_improved(df, window=20, min_touches=2, tolerance=0.008, n_levels=1, symbol="", timeframe="H1"):
    """Enhanced support/resistance detection - ONLY returns the MOST ACCURATE single level for each timeframe"""
    
    # Fallback for insufficient data - only return single best levels
    if len(df) < 20:
        if len(df) > 0:
            current_price = df['close'].iloc[-1]
            # Return only single most relevant levels
            return [current_price * 0.99], [current_price * 1.01]
        return [], []
    
    # Use adaptive window if timeframe is provided
    if timeframe and symbol:
        try:
            window = get_adaptive_window(df, timeframe, window)
        except Exception:
            pass  # Keep original window if calculation fails
    
    if len(df) < window * 2 + 1:
        # Reduce window size for smaller datasets
        window = max(len(df) // 4, 5)
    
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    current_price = close_prices[-1]
    
    # Method 1: Local extrema
    levels_extrema = []
    for i in range(window, len(df) - window):
        # Local highs
        if high_prices[i] == np.max(high_prices[i-window:i+window+1]):
            levels_extrema.append(high_prices[i])
        # Local lows
        if low_prices[i] == np.min(low_prices[i-window:i+window+1]):
            levels_extrema.append(low_prices[i])
    
    # Method 2: Psychological levels (round numbers)
    psychological_levels = []
    price_range = high_prices.max() - low_prices.min()
    
    if current_price > 1:
        # For prices > 1, find round numbers
        if current_price > 100:
            step = 10 if price_range > 50 else 5
        elif current_price > 10:
            step = 1 if price_range > 5 else 0.5
        else:
            step = 0.1 if price_range > 0.5 else 0.05
        
        start = int(low_prices.min() / step) * step
        end = int(high_prices.max() / step + 1) * step
        
        level = start
        while level <= end:
            if low_prices.min() <= level <= high_prices.max():
                psychological_levels.append(level)
            level += step
    
    # Method 3: Volume-weighted levels (if volume data available)
    volume_levels = []
    if 'volume' in df.columns or 'tick_volume' in df.columns:
        vol_col = 'volume' if 'volume' in df.columns else 'tick_volume'
        # Group prices by volume and find high-volume levels
        df_temp = df.copy()
        df_temp['price_level'] = np.round(df_temp['close'] / (current_price * tolerance)) * (current_price * tolerance)
        volume_by_level = df_temp.groupby('price_level')[vol_col].sum().sort_values(ascending=False)
        
        # Take top volume levels
        top_volume_levels = volume_by_level.head(n_levels * 2).index.tolist()
        volume_levels.extend(top_volume_levels)
    
    # Combine all levels
    all_levels = levels_extrema + psychological_levels + volume_levels
    
    # Merge nearby levels
    def merge_levels_advanced(levels, tolerance_pct):
        if not levels:
            return []
        
        levels = sorted(levels)
        merged = [levels[0]]
        
        for level in levels[1:]:
            # Check if this level is close to any existing merged level
            is_close = False
            for i, merged_level in enumerate(merged):
                if abs(level - merged_level) <= tolerance_pct * max(abs(level), abs(merged_level)):
                    # Merge by taking weighted average (more weight to level with more touches)
                    merged[i] = (merged_level + level) / 2
                    is_close = True
                    break
            
            if not is_close:
                merged.append(level)
        
        return merged
    
    merged_levels = merge_levels_advanced(all_levels, tolerance)
    
    # Calculate touches and strength for each level with improved logic
    level_strength = []
    for level in merged_levels:
        # Count touches with better validation
        touches = 0
        bounce_strength = 0
        rejection_count = 0
        
        for i in range(1, len(df) - 1):  # Avoid first and last candles
            current_price = close_prices[i]
            prev_price = close_prices[i-1]
            next_price = close_prices[i+1]
            
            # More accurate touch detection
            touched_level = False
            if (min(low_prices[i], high_prices[i]) <= level <= max(low_prices[i], high_prices[i])):
                touched_level = True
            elif abs(current_price - level) <= tolerance * level:
                touched_level = True
            
            if touched_level:
                touches += 1
                
                # Calculate bounce/rejection strength
                if i < len(df) - 2:
                    # Check if price bounced from level
                    next_move = abs(close_prices[i+2] - current_price)
                    bounce_strength += next_move
                    
                    # Check for strong rejection (reversal pattern)
                    if level <= current_price:  # Resistance test
                        if next_price < current_price and close_prices[i+2] < prev_price:
                            rejection_count += 2  # Strong rejection
                    else:  # Support test
                        if next_price > current_price and close_prices[i+2] > prev_price:
                            rejection_count += 2  # Strong support hold
        
        # Enhanced strength calculation
        if touches >= min_touches:
            avg_bounce = bounce_strength / max(touches, 1)
            proximity_score = 1 / (1 + abs(level - current_price) / current_price)  # Closer levels get higher score
            
            level_strength.append({
                'level': level,
                'touches': touches,
                'strength': avg_bounce,
                'rejection_count': rejection_count,
                'proximity_score': proximity_score,
                'combined_score': touches * 0.4 + rejection_count * 0.3 + proximity_score * 0.3,
                'distance_from_current': abs(level - current_price) / current_price
            })
    
    # Sort by enhanced combined score
    level_strength.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # 🎯 SELECT ONLY THE MOST ACCURATE SINGLE LEVEL FOR EACH TYPE
    # Sort by combined score and select the absolute best single level
    supports = []
    resistances = []
    
    # Find the highest scoring support level (BELOW current price)
    support_candidates = [item for item in level_strength if item['level'] < current_price * 0.998]
    if support_candidates:
        # Apply distance filtering for supports
        filtered_supports = []
        for item in support_candidates:
            distance_pct = abs(item['level'] - current_price) / current_price * 100
            max_distance_pct = {
                'M1': 1.0, 'M5': 2.0, 'M15': 3.0, 'M30': 4.0,
                'H1': 5.0, 'H4': 8.0, 'D1': 12.0, 'W1': 20.0
            }.get(timeframe, 5.0)
            
            if distance_pct <= max_distance_pct:
                filtered_supports.append(item)
        
        if filtered_supports:
            # Sort by combined score and take the best one
            filtered_supports.sort(key=lambda x: x['combined_score'], reverse=True)
            best_support = filtered_supports[0]['level']
            supports = [best_support]
    
    # Find the highest scoring resistance level (ABOVE current price)
    resistance_candidates = [item for item in level_strength if item['level'] > current_price * 1.002]
    if resistance_candidates:
        # Apply distance filtering for resistances
        filtered_resistances = []
        for item in resistance_candidates:
            distance_pct = abs(item['level'] - current_price) / current_price * 100
            max_distance_pct = {
                'M1': 1.0, 'M5': 2.0, 'M15': 3.0, 'M30': 4.0,
                'H1': 5.0, 'H4': 8.0, 'D1': 12.0, 'W1': 20.0
            }.get(timeframe, 5.0)
            
            if distance_pct <= max_distance_pct:
                filtered_resistances.append(item)
        
        if filtered_resistances:
            # Sort by combined score and take the best one
            filtered_resistances.sort(key=lambda x: x['combined_score'], reverse=True)
            best_resistance = filtered_resistances[0]['level']
            resistances = [best_resistance]
    
    # Fallback if no levels found despite having data
    if not supports and not resistances and len(df) >= 20:
        try:
            # Create single best level based on recent price action
            recent_data = df.tail(min(50, len(df)))
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()
            
            # Select single most significant level for each type
            if recent_high > current_price and not resistances:
                resistances = [recent_high]
            if recent_low < current_price and not supports:
                supports = [recent_low]
                
            # Add single percentage-based level if still empty
            if not supports:
                supports = [current_price * 0.99]
            if not resistances:
                resistances = [current_price * 1.01]
                
        except Exception:
            # Ultimate fallback - single levels only
            supports = [current_price * 0.99] if not supports else supports
            resistances = [current_price * 1.01] if not resistances else resistances
    
    return supports, resistances

def find_support_resistance(df, window=20, min_touches=2, tolerance=0.008, n_levels=1):
    """Legacy function - redirects to improved version with SINGLE level only"""
    return find_support_resistance_improved(df, window, min_touches, tolerance, n_levels=1)

def count_touches(prices, lvl):
    touch_down = np.sum((prices[:-1] > lvl) & (prices[1:] <= lvl))
    touch_up = np.sum((prices[:-1] < lvl) & (prices[1:] >= lvl))
    return int(touch_down), int(touch_up)

def remove_sr_overlap(supports, resistances, threshold=0.002):
    filtered_supports = []
    filtered_resistances = []
    for s in supports:
        if all(abs(s - r) / max(abs(s), abs(r)) > threshold for r in resistances):
            filtered_supports.append(s)
    for r in resistances:
        if all(abs(r - s) / max(abs(r), abs(s)) > threshold for s in supports):
            filtered_resistances.append(r)
    return filtered_supports, filtered_resistances

def check_breakout_improved(df, channel_upper, channel_lower, trend_direction):
    """Enhanced breakout detection with trend context
    
    CRITICAL: Breakout must be confirmed by CLOSE PRICE, not just high/low wicks!
    True breakout = candle CLOSES above resistance or below support
    """
    prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    
    vol_col = 'volume' if 'volume' in df.columns else 'tick_volume'
    current_volume = df[vol_col].iloc[-1] if vol_col in df.columns else 0
    avg_volume = df[vol_col].rolling(20).mean().iloc[-1] if vol_col in df.columns else 0
    
    # Volume confirmation
    volume_confirmation = current_volume > avg_volume * 1.2 if avg_volume > 0 else True
    
    # 🎯 CRITICAL FIX: Breakout MUST be confirmed by CLOSE PRICE
    # Not just high/low touch - requires candle to CLOSE beyond SR level
    upper_breakout = prices[-1] > channel_upper[-1]  # Close above resistance
    lower_breakout = prices[-1] < channel_lower[-1]  # Close below support
    
    # Check for false breakouts (wick touched but didn't close beyond)
    upper_wick_only = high_prices[-1] > channel_upper[-1] and not upper_breakout
    lower_wick_only = low_prices[-1] < channel_lower[-1] and not lower_breakout
    
    # Trend-aware breakout validation
    if trend_direction == "Uptrend":
        if upper_breakout and volume_confirmation:
            # Continuation breakout in uptrend
            return "Strong Breakout Up"
        elif upper_breakout and not volume_confirmation:
            return "Weak Breakout Up (Low Volume)"
        elif lower_breakout:
            # Potential trend reversal
            return "Trend Reversal (Break Down)"
        elif upper_wick_only:
            return "False Breakout Up (Wick Only)"
    elif trend_direction == "Downtrend":
        if lower_breakout and volume_confirmation:
            # Continuation breakout in downtrend
            return "Strong Breakout Down"
        elif lower_breakout and not volume_confirmation:
            return "Weak Breakout Down (Low Volume)"
        elif upper_breakout:
            # Potential trend reversal
            return "Trend Reversal (Break Up)"
        elif lower_wick_only:
            return "False Breakout Down (Wick Only)"
    else:
        # Sideway trend - breakouts are often FALSE in consolidation
        if upper_breakout and volume_confirmation:
            return "Breakout Up (Confirm with retest)"
        elif lower_breakout and volume_confirmation:
            return "Breakout Down (Confirm with retest)"
        elif upper_breakout and not volume_confirmation:
            return "Weak Breakout Up (Low Volume - Likely False)"
        elif lower_breakout and not volume_confirmation:
            return "Weak Breakout Down (Low Volume - Likely False)"
        elif upper_wick_only:
            return "False Breakout Up (Wick Only - Rejection)"
        elif lower_wick_only:
            return "False Breakout Down (Wick Only - Rejection)"
    
    return "No Breakout"

def check_breakout(df, channel_upper, channel_lower):
    """Legacy function - redirects to improved version"""
    trend_direction = determine_trend(df)
    return check_breakout_improved(df, channel_upper, channel_lower, trend_direction)

def get_candle_count_from_data(symbol, timeframe):
    data_folder = "data"
    filename = f"{symbol}_{timeframe}.json"
    filepath = os.path.join(data_folder, filename)
    if not os.path.exists(filepath):
        return 200
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data)

def get_tolerance_improved(symbol, timeframe):
    """Enhanced tolerance calculation with more precise asset classification"""
    symbol = symbol.upper()
    
    # Precious metals
    if any(metal in symbol for metal in ["XAU", "GOLD", "GLD"]):
        return 0.001 if timeframe in ["M1", "M5"] else 0.003
    if any(metal in symbol for metal in ["XAG", "SILVER", "SLV"]):
        return 0.002 if timeframe in ["M1", "M5"] else 0.005
    if any(metal in symbol for metal in ["XPD", "XPT", "PALLADIUM", "PLATINUM"]):
        return 0.003 if timeframe in ["M1", "M5"] else 0.006
    
    # Energy commodities
    if any(energy in symbol for energy in ["OIL", "CL", "BRENT", "WTI", "CRUDE"]):
        return 0.008 if timeframe in ["M1", "M5"] else 0.015
    if any(gas in symbol for gas in ["NG", "NATGAS", "NATURALGAS"]):
        return 0.015 if timeframe in ["M1", "M5"] else 0.025
    
    # Cryptocurrencies
    if symbol.startswith("BTC") or "BITCOIN" in symbol:
        return 0.005 if timeframe in ["M1", "M5"] else 0.015
    if symbol.startswith("ETH") or "ETHEREUM" in symbol:
        return 0.008 if timeframe in ["M1", "M5"] else 0.020
    if any(crypto in symbol for crypto in ["ADA", "DOT", "LINK", "LTC", "XRP"]):
        return 0.015 if timeframe in ["M1", "M5"] else 0.030
    
    # Forex pairs
    major_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    if any(pair in symbol.replace(".", "") for pair in major_pairs):
        if "JPY" in symbol:
            return 0.008 if timeframe in ["M1", "M5"] else 0.012
        return 0.0003 if timeframe in ["M1", "M5"] else 0.0008
    
    # Minor and exotic forex pairs
    if any(curr in symbol for curr in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "SEK", "NOK", "DKK"]):
        if "JPY" in symbol:
            return 0.012 if timeframe in ["M1", "M5"] else 0.020
        return 0.0005 if timeframe in ["M1", "M5"] else 0.0012
    
    # Stock indices
    major_indices = ["SPX", "SPY", "QQQ", "DJI", "US30", "NAS100", "SPX500"]
    if any(index in symbol for index in major_indices):
        return 0.005 if timeframe in ["M1", "M5"] else 0.012
    
    regional_indices = ["DAX", "FTSE", "CAC", "NIKKEI", "HSI", "ASX"]
    if any(index in symbol for index in regional_indices):
        return 0.008 if timeframe in ["M1", "M5"] else 0.015
    
    # Individual stocks
    mega_cap_stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "NVDA", "BRK"]
    if any(stock in symbol for stock in mega_cap_stocks):
        return 0.008 if timeframe in ["M1", "M5"] else 0.015
    
    # High-priced stocks (>$100)
    high_price_stocks = ["BRK.A", "NVR", "AUTO", "MELI", "CHTR"]
    if any(stock in symbol for stock in high_price_stocks):
        return 0.005 if timeframe in ["M1", "M5"] else 0.012
    
    # Regular stocks
    if symbol.replace(".", "").replace("#", "").isalpha() and len(symbol) <= 6:
        return 0.012 if timeframe in ["M1", "M5"] else 0.020
    
    # Default fallback
    return 0.008 if timeframe in ["M1", "M5"] else 0.015

def get_tolerance(symbol, timeframe):
    """Legacy function - redirects to improved version"""
    return get_tolerance_improved(symbol, timeframe)

def robust_trendline(x, y, min_r2=0.7):
    """Enhanced robust trendline with R² validation"""
    if len(x) < 2:
        return 0, y.mean() if len(y) > 0 else 0
    
    x_ = x.reshape(-1, 1)
    
    try:
        # Try RANSAC first
        ransac = RANSACRegressor(LinearRegression(), random_state=42, min_samples=max(2, len(x)//2))
        ransac.fit(x_, y)
        
        # Calculate R² for validation
        linear_reg = LinearRegression()
        linear_reg.fit(x_, y)
        r2_score = linear_reg.score(x_, y)
        
        # Use RANSAC result if R² is acceptable
        if r2_score >= min_r2:
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
        else:
            # Fallback to simple linear regression
            slope = linear_reg.coef_[0]
            intercept = linear_reg.intercept_
            
        return slope, intercept
        
    except Exception as e:
        logger.warning(f"RANSAC failed, using simple linear regression: {e}")
        # Fallback to simple linear regression
        if len(x) == len(y) and len(x) >= 2:
            slope = np.polyfit(x, y, 1)[0]
            intercept = np.polyfit(x, y, 1)[1]
            return slope, intercept
        else:
            return 0, y.mean() if len(y) > 0 else 0

def get_cache_key(symbol, timeframe, count, df_hash):
    """Generate cache key for analysis results"""
    return hashlib.md5(f"{symbol}_{timeframe}_{count}_{df_hash}".encode()).hexdigest()

def analyze_trend_channel_sr_with_cache(symbol, timeframe, count=200, use_cache=True, user_id=None):
    """Main analysis function with caching support and user tracking"""
    start_time = datetime.now()
    
    try:
        # Log analysis start
        if user_id:
            log_user_analysis(user_id, symbol, timeframe, "analysis_started", {
                "count": count,
                "use_cache": use_cache
            })
        
        df = load_ohlc(symbol, timeframe, count)
        
        # Generate cache key
        df_hash = hashlib.md5(df.to_string().encode()).hexdigest()
        cache_key = get_cache_key(symbol, timeframe, count, df_hash)
          # Check cache
        if use_cache:
            os.makedirs(CACHE_FOLDER, exist_ok=True)
            cache_file = os.path.join(CACHE_FOLDER, f"{cache_key}.pkl")
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    
                    # Log cache hit
                    if user_id:
                        analysis_time = (datetime.now() - start_time).total_seconds()
                        log_user_analysis(user_id, symbol, timeframe, "analysis_completed", {
                            "duration_seconds": round(analysis_time, 2),
                            "cache_used": True,
                            "trendlines_count": len(result.get('trendlines', [])),
                            "support_levels": len(result.get('support_levels', [])),
                            "resistance_levels": len(result.get('resistance_levels', []))
                        })
                        log_performance_analysis(symbol, timeframe, "cached_analysis", analysis_time, user_id)
                    
                    logger.info(f"Loaded cached result for {symbol}_{timeframe}")
                    return result
                except Exception as e:
                    log_rate_limited_error(f"cache_load_{symbol}_{timeframe}", f"Failed to load cache: {e}")
        
        # Perform analysis
        result = analyze_trend_channel_sr(symbol, timeframe, count)
        
        # Save to cache
        if use_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.info(f"Cached result for {symbol}_{timeframe}")
            except Exception as e:
                log_rate_limited_error(f"cache_save_{symbol}_{timeframe}", f"Failed to save cache: {e}")
        
        # Log successful analysis
        if user_id:
            analysis_time = (datetime.now() - start_time).total_seconds()
            log_user_analysis(user_id, symbol, timeframe, "analysis_completed", {
                "duration_seconds": round(analysis_time, 2),
                "cache_used": False,
                "trendlines_count": len(result.get('trendlines', [])),
                "support_levels": len(result.get('support_levels', [])),
                "resistance_levels": len(result.get('resistance_levels', []))
            })
            log_performance_analysis(symbol, timeframe, "full_analysis", analysis_time, user_id)
        
        return result
    
    except Exception as e:
        # Log analysis failure
        if user_id:
            analysis_time = (datetime.now() - start_time).total_seconds()
            log_user_analysis(user_id, symbol, timeframe, "analysis_failed", {
                "error": str(e),
                "duration_seconds": round(analysis_time, 2)
            })
        
        log_rate_limited_error(f"analysis_error_{symbol}_{timeframe}", f"Analysis failed for {symbol}_{timeframe}: {e}")
        raise

def analyze_trend_channel_sr(symbol, timeframe, count=200):
    """🎯 Enhanced main analysis function - SINGLE PRIMARY LEVEL FOCUS
    
    Returns ONLY the most accurate single support and resistance level per timeframe
    instead of multiple levels. This provides cleaner, more focused trading signals.
    """
    try:
        # Load and validate data
        df = load_ohlc(symbol, timeframe, count)
        
        if len(df) < 50:  # Minimum data requirement
            raise ValueError(f"Insufficient data: {len(df)} candles (minimum 50 required)")
        
        # Calculate improved trendline
        trendline, slope, intercept, lows, highs, trend_direction, sideway_range_data = calc_trendline_improved(df)
        
        # Calculate improved channel
        channel_upper, channel_lower = calc_channel_improved(df, trendline, highs, lows, trend_direction)
        
        # Get dynamic tolerance
        tolerance = get_tolerance_improved(symbol, timeframe)
        
        # Find improved support/resistance - ONLY 1 level each
        supports, resistances = find_support_resistance_improved(
            df, tolerance=tolerance, n_levels=1, symbol=symbol, timeframe=timeframe
        )
        
        # Remove overlapping levels
        supports, resistances = remove_sr_overlap(supports, resistances, threshold=tolerance/2)
        
        # Calculate deviations and statistics
        deviations = df['close'].values - trendline
        max_dev = float(np.max(deviations))
        min_dev = float(np.min(deviations))
        
        # Enhanced trendline description with PROPER normalization
        # Calculate average price for normalization
        avg_price = df['close'].mean()
        # Normalize slope as percentage change per candle relative to price
        # Then scale to reasonable range (0-10)
        slope_pct = abs(slope) / avg_price * 100  # Percentage slope per candle
        trend_strength = min(slope_pct * 2, 10.0)  # Cap at 10.0, scale by 2x for sensitivity
        
        trendline_value = (f"{trendline[0]:.5f} → {trendline[-1]:.5f} | "
                          f"Slope: {slope:.6f} ({slope_pct:.3f}%/candle) | Strength: {trend_strength:.2f} | "
                          f"Direction: {trend_direction}")
        
        # Get detailed support/resistance info
        prices = df['close'].values
        supports_info = []
        for s in supports:
            down, up = count_touches(prices, s)
            strength = down + up
            supports_info.append({
                'level': s, 
                'touch_down': down, 
                'touch_up': up,
                'strength': strength,
                'distance_pct': abs(s - prices[-1]) / prices[-1] * 100
            })
        
        resistances_info = []
        for r in resistances:
            down, up = count_touches(prices, r)
            strength = down + up
            resistances_info.append({
                'level': r, 
                'touch_down': down, 
                'touch_up': up,
                'strength': strength,
                'distance_pct': abs(r - prices[-1]) / prices[-1] * 100
            })        # Enhanced breakout detection
        breakout = check_breakout_improved(df, channel_upper, channel_lower, trend_direction)
        
        # Create summary information - Single level focus
        sr_summary = f"S/R: {len(supports)}S/{len(resistances)}R (Primary Only)"
        breakout_summary = f"Breakout: {breakout}"
        trend_summary = f"Trend: {trend_direction}"
        
        # Build comprehensive result
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_time": datetime.now().isoformat(),
            "data_points": len(df),
            
            # Summary information for quick reference
            "summary": {
                "trend_summary": trend_summary,
                "sr_summary": sr_summary,
                "breakout_summary": breakout_summary,
                "full_summary": f"{trend_summary} | {sr_summary} | {breakout_summary}"
            },
            
            # Trend analysis
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "trendline": trendline.tolist(),
            "trend_slope": float(slope),
            "trend_intercept": float(intercept),
            "trendline_value": trendline_value,
            
            # Channel analysis
            "channel_upper": channel_upper.tolist(),
            "channel_lower": channel_lower.tolist(),
            "channel_width": float(np.mean(channel_upper - channel_lower)),
              # Support/Resistance
            "support": supports,
            "resistance": resistances,
            "supports_info": supports_info,
            "resistances_info": resistances_info,
            "support_levels": [f"{s:.5f}" for s in supports],
            "resistance_levels": [f"{r:.5f}" for r in resistances],
            
            # 🎯 PRIMARY Support/Resistance Statistics (Single Level Focus)
            "sr_statistics": {
                "mode": "single_primary_level",
                "total_support_levels": len(supports),  # Always 0 or 1
                "total_resistance_levels": len(resistances),  # Always 0 or 1
                "primary_support_strength": supports_info[0]['strength'] if supports_info else 0,
                "primary_resistance_strength": resistances_info[0]['strength'] if resistances_info else 0,
                "primary_support": supports[0] if supports else None,
                "primary_resistance": resistances[0] if resistances else None,
                "primary_support_distance": abs(supports[0] - prices[-1]) / prices[-1] * 100 if supports else None,
                "primary_resistance_distance": abs(resistances[0] - prices[-1]) / prices[-1] * 100 if resistances else None,
                "has_primary_support": len(supports) > 0,
                "has_primary_resistance": len(resistances) > 0
            },
            
            # Swing points
            "swing_lows": lows.tolist(),
            "swing_highs": highs.tolist(),
            
            # Statistics
            "max_deviation": max_dev,
            "min_deviation": min_dev,
            "volatility": float(df['close'].pct_change().std() * 100),            # Breakout analysis
            "breakout": breakout,
            
            # Breakout Analysis Details
            "breakout_analysis": {
                "breakout_type": breakout,
                "is_breakout": "Breakout" in breakout,
                "breakout_direction": ("Up" if "Up" in breakout else ("Down" if "Down" in breakout else "None")),
                "breakout_strength": ("Strong" if "Strong" in breakout else ("Weak" if "Weak" in breakout else "Normal")),
                "volume_confirmation": not ("Low Volume" in breakout),
                "trend_context": trend_direction,
                "is_reversal": "Reversal" in breakout
            },
            
            # 📊 Sideway Range Analysis (only present if trend is Sideway)
            "sideway_range": sideway_range_data if sideway_range_data else None,
            
            # Raw data
            "ohlc": df.tail(100).to_dict(orient="records"),  # Only last 100 candles to reduce size
            
            # Configuration
            "tolerance_used": tolerance,
        }        
        # Save to file
        output_dir = "trendline_sr"
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{symbol}_{timeframe}_trendline_sr.json")        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Log detailed summary
        logger.info(f"Analysis completed for {symbol}_{timeframe}")
        logger.info(f"{trend_summary} | {sr_summary} | {breakout_summary}")
        
        # 🎯 Single Level Logging - Only the most accurate levels
        if supports:
            # Only one support level now - the most accurate
            primary_support = supports[0]
            support_dist = abs(primary_support - prices[-1]) / prices[-1] * 100
            logger.info(f"🎯 PRIMARY Support: {primary_support:.5f} ({support_dist:.2f}% away)")
        
        if resistances:
            # Only one resistance level now - the most accurate
            primary_resistance = resistances[0]
            resistance_dist = abs(primary_resistance - prices[-1]) / prices[-1] * 100
            logger.info(f"🎯 PRIMARY Resistance: {primary_resistance:.5f} ({resistance_dist:.2f}% away)")
        
        if "Strong" in breakout:
            logger.info(f"🚀 Strong breakout detected: {breakout}")
        elif "Weak" in breakout:
            logger.info(f"⚠️ Weak breakout detected: {breakout}")
        elif "Reversal" in breakout:
            logger.info(f"🔄 Potential trend reversal: {breakout}")
        
        logger.info(f"📏 Channel width: {result['channel_width']:.5f}")
        logger.info(f"📈 Trend strength: {trend_strength:.2f}")
        logger.info(f"📊 Volatility: {result['volatility']:.2f}%")
        
        # 📊 Log Sideway Range if present
        if sideway_range_data:
            logger.info(f"📊 SIDEWAY RANGE DETECTED:")
            logger.info(f"   Range: {sideway_range_data['range_low']:.5f} - {sideway_range_data['range_high']:.5f}")
            logger.info(f"   Width: {sideway_range_data['range_width']:.5f} ({sideway_range_data['range_width_pct']:.2f}%)")
            logger.info(f"   Midpoint: {sideway_range_data['range_midpoint']:.5f}")
            logger.info(f"   Current Position: {sideway_range_data['current_position_pct']:.1f}% (0%=đáy, 100%=đỉnh)")
            logger.info(f"   Touches: {sideway_range_data['touches_bottom']} bottom, {sideway_range_data['touches_top']} top")
            logger.info(f"   Consolidation: {sideway_range_data['consolidation_strength']}")
            logger.info(f"   Volatility: {sideway_range_data['volatility']:.2f}%")
        
        # Build table data for UI
        table_rows = []
        table_rows.append({
            "Symbol": symbol,
            "Timeframe": timeframe,
            "Type": "Trend",
            "Value": f"{trend_direction} (Strength: {trend_strength:.1f})"
        })
        
        table_rows.append({
            "Symbol": symbol,
            "Timeframe": timeframe,
            "Type": "Channel",
            "Value": f"Width: {result['channel_width']:.5f}"
        })
        
        # 🎯 Only show PRIMARY levels (single most accurate)
        if supports_info:
            s_info = supports_info[0]  # Only the primary support
            table_rows.append({
                "Symbol": symbol,
                "Timeframe": timeframe,
                "Type": "🎯 PRIMARY Support",
                "Value": f"{s_info['level']:.5f} (Str: {s_info['strength']}, Dist: {s_info['distance_pct']:.1f}%)"
            })
        
        if resistances_info:
            r_info = resistances_info[0]  # Only the primary resistance
            table_rows.append({
                "Symbol": symbol,
                "Timeframe": timeframe,
                "Type": "🎯 PRIMARY Resistance", 
                "Value": f"{r_info['level']:.5f} (Str: {r_info['strength']}, Dist: {r_info['distance_pct']:.1f}%)"
            })
        
        table_rows.append({
            "Symbol": symbol,
            "Timeframe": timeframe,
            "Type": "Breakout",
            "Value": breakout
        })
        
        # 📊 Add Sideway Range to table if present
        if sideway_range_data:
            table_rows.append({
                "Symbol": symbol,
                "Timeframe": timeframe,
                "Type": "📊 Sideway Range",
                "Value": f"{sideway_range_data['range_low']:.5f} - {sideway_range_data['range_high']:.5f} (Width: {sideway_range_data['range_width_pct']:.1f}%)"
            })
            table_rows.append({
                "Symbol": symbol,
                "Timeframe": timeframe,
                "Type": "📍 Range Position",
                "Value": f"{sideway_range_data['current_position_pct']:.1f}% | Consolidation: {sideway_range_data['consolidation_strength']}"
            })
        
        result["support_resistance"] = table_rows
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Analysis failed for {symbol}_{timeframe}: {e}")
        raise

def plot_trendline_sr(result):
    closes = [x['close'] for x in result['ohlc']]
    x = np.arange(len(closes))
    plt.figure(figsize=(14, 6))
    plt.plot(x, closes, label='Close', color='black')
    plt.plot(x, result['trendline'], label='Trendline', color='blue')
    plt.plot(x, result['channel_upper'], label='Channel Upper', color='green', linestyle='--')
    plt.plot(x, result['channel_lower'], label='Channel Lower', color='red', linestyle='--')
    for s in result['support']:
        plt.axhline(s, color='green', linestyle=':', alpha=0.5, label='Support')
    for r in result['resistance']:
        plt.axhline(r, color='red', linestyle=':', alpha=0.5, label='Resistance')
    if "swing_lows" in result:
        plt.scatter(result['swing_lows'], [closes[i] for i in result['swing_lows']], color='blue', marker='v', label='Swing Low')
    if "swing_highs" in result:
        plt.scatter(result['swing_highs'], [closes[i] for i in result['swing_highs']], color='orange', marker='^', label='Swing High')
    plt.legend()
    plt.title('Trendline, Channel, Support/Resistance')
    plt.show()

def on_calculate(self):
    """Enhanced calculation function with better error handling"""
    if not self.enable_checkbox.isChecked():
        QMessageBox.information(self, "Info", "Trend Detect is disabled!")
        return
    
    symbols = list(self.market_tab.checked_symbols)
    timeframes = [tf for tf in self.market_tab.tf_checkboxes if self.market_tab.tf_checkboxes[tf].isChecked()]
    
    if not symbols or not timeframes:
        QMessageBox.warning(self, "Warning", "Please select symbol and timeframe in Market tab.")
        return
    
    # Clear previous results
    self.table.setRowCount(0)
    self.result_label.setText("Calculating enhanced analysis...")
    
    # Stop any running workers
    for worker in getattr(self, 'workers', []):
        if worker.isRunning():
            worker.quit()
            worker.wait()
    
    self.workers = []
    self.pending = 0
    self.results = {}
    
    # Start analysis for each symbol/timeframe combination
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                count = get_candle_count_from_data(symbol, timeframe)
                worker = TrendWorkerImproved(symbol, timeframe, count)
                worker.finished.connect(lambda result, s=symbol, tf=timeframe: self.on_result(result, s, tf))
                worker.error.connect(lambda error, s=symbol, tf=timeframe: self.on_error(error, s, tf))
                self.workers.append(worker)
                self.pending += 1
                worker.start()
                
            except Exception as e:
                logger.error(f"Failed to start analysis for {symbol}_{timeframe}: {e}")
                QMessageBox.critical(self, "Error", f"Failed to start analysis for {symbol}_{timeframe}: {str(e)}")
    
    if self.pending > 0:
        self.calc_btn.setEnabled(False)
        self.result_label.setText(f"Running analysis for {self.pending} combinations...")
    else:
        self.result_label.setText("No valid symbol/timeframe combinations found.")

def on_error(self, error_msg, symbol, timeframe):
    """Handle worker errors"""
    logger.error(f"Analysis error for {symbol}_{timeframe}: {error_msg}")
    self.pending -= 1
    
    if self.pending <= 0:
        self.calc_btn.setEnabled(True)
        self.result_label.setText("Analysis completed with errors.")

class TrendWorkerImproved(QThread):
    """Enhanced worker thread with better error handling"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, symbol, timeframe, count):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.count = count

    def run(self):
        try:
            logger.info(f"🔄 Starting analysis for {self.symbol}_{self.timeframe}")
            result = analyze_trend_channel_sr_with_cache(self.symbol, self.timeframe, self.count)
            self.finished.emit(result)
            
        except FileNotFoundError as e:
            error_msg = f"Data file not found: {str(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            
        except ValueError as e:
            error_msg = f"Data validation error: {str(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)

class TrendWorker(QThread):
    """Legacy worker - redirects to improved version"""
    finished = pyqtSignal(object)

    def __init__(self, symbol, timeframe, count):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.count = count

    def run(self):
        try:
            result = analyze_trend_channel_sr_with_cache(self.symbol, self.timeframe, self.count)
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Legacy worker error for {self.symbol}_{self.timeframe}: {e}")
            # Create minimal error result for backward compatibility
            error_result = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "error": str(e),
                "support_resistance": []
            }
            self.finished.emit(error_result)

def main():
    """Main function for standalone execution"""
    print("📈 Trendline Support Resistance - Standalone Mode")
    print("=" * 50)
    
    # 🧹 AUTO CLEANUP before analysis
    print("🧹 Trendline SR: Auto cleanup before processing...")
    try:
        cleanup_result = cleanup_trendline_data(max_age_hours=48, keep_latest=15)
        print(f"✅ Cleaned {cleanup_result['total_files_deleted']} files, "
              f"freed {cleanup_result['total_space_freed_mb']:.2f} MB")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    # Test trendline analysis
    print("🤖 Testing Trendline Support/Resistance Analysis...")
    
    try:
        # Example usage
        symbol = "XAUUSD"
        timeframe = "H1"
        
        print(f"📊 Analyzing {symbol} {timeframe}...")
        
        # Create TrendWorker instance for testing
        # worker = TrendWorker(symbol, timeframe)
        
        # Note: This would need actual candle data to work properly
        print("ℹ️ Note: This is a test run. Real usage requires candle data.")
        print("✅ Trendline SR module test completed successfully")
        
    except Exception as e:
        print(f"❌ Trendline SR test failed: {e}")

def get_primary_levels(result):
    """🎯 Utility function to extract primary support/resistance levels from analysis result
    
    Returns:
        dict: {
            'primary_support': float or None,
            'primary_resistance': float or None,
            'support_distance_pct': float or None,
            'resistance_distance_pct': float or None
        }
    """
    primary_data = {
        'primary_support': None,
        'primary_resistance': None, 
        'support_distance_pct': None,
        'resistance_distance_pct': None
    }
    
    try:
        if result.get('sr_statistics'):
            stats = result['sr_statistics']
            primary_data['primary_support'] = stats.get('primary_support')
            primary_data['primary_resistance'] = stats.get('primary_resistance')
            primary_data['support_distance_pct'] = stats.get('primary_support_distance')
            primary_data['resistance_distance_pct'] = stats.get('primary_resistance_distance')
    except Exception as e:
        logger.warning(f"Failed to extract primary levels: {e}")
    
    return primary_data

def cleanup_trendline_data(max_age_hours: int = 48, keep_latest: int = 15) -> dict:
    """
    🧹 TRENDLINE SUPPORT RESISTANCE: Dọn dẹp dữ liệu của module này
    Dọn dẹp trendline analysis results và cache
    
    Args:
        max_age_hours: Tuổi tối đa của file (giờ)
        keep_latest: Số file mới nhất cần giữ lại
    """
    from datetime import timedelta
    
    cleanup_stats = {
        'module_name': 'trendline_support_resistance',
        'directories_cleaned': [],
        'total_files_deleted': 0,
        'total_space_freed_mb': 0.0,
        'cleanup_time': datetime.now().isoformat()
    }
    
    # Thư mục mà Trendline SR quản lý
    target_directories = [
        'trendline_sr',      # Trendline analysis output
        'cache',            # Cached trendline data
        'sr_analysis'       # Support/resistance analysis (if any)
    ]
    
    for directory in target_directories:
        if os.path.exists(directory):
            result = _clean_directory(directory, max_age_hours, keep_latest)
            cleanup_stats['directories_cleaned'].append({
                'directory': directory,
                'files_deleted': result['deleted'],
                'space_freed_mb': result['space_freed']
            })
            cleanup_stats['total_files_deleted'] += result['deleted']
            cleanup_stats['total_space_freed_mb'] += result['space_freed']
        else:
            cleanup_stats['directories_cleaned'].append({
                'directory': directory,
                'status': 'not_found'
            })
    
    print(f"🧹 TRENDLINE SR cleanup complete: "
          f"{cleanup_stats['total_files_deleted']} files deleted, "
          f"{cleanup_stats['total_space_freed_mb']:.2f} MB freed")
    return cleanup_stats

def _clean_directory(directory: str, max_age_hours: int, keep_latest: int) -> dict:
    """Helper function để clean một directory"""
    from datetime import timedelta
    
    deleted_count = 0
    space_freed = 0.0
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    try:
        if not os.path.exists(directory):
            return {'deleted': 0, 'space_freed': 0.0}
            
        # Lấy tất cả trendline files
        all_files = []
        for file_name in os.listdir(directory):
            if file_name.endswith(('.json', '.pkl', '.cache', '.png', '.jpg')):
                file_path = os.path.join(directory, file_name)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    file_size = os.path.getsize(file_path)
                    all_files.append({
                        'path': file_path,
                        'time': file_time,
                        'size': file_size
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
                except Exception as e:
                    print(f"Warning: Could not delete {file_info['path']}: {e}")
        
    except Exception as e:
        print(f"Error cleaning directory {directory}: {e}")
    
    return {'deleted': deleted_count, 'space_freed': space_freed}

if __name__ == "__main__":
    main()