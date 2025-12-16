# Enhanced logging setup for multi-user indicator system with data cleanup
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
import glob
import logging
import os
import json
from utils import overwrite_json_safely, ensure_directory, auto_cleanup_on_start

def log_rate_limited_error(message: str):
    """Log rate limited error"""
    print(f"RATE LIMITED: {message}")

def log_user_action(user_id: str, action: str, details: dict = None):
    """Log user action"""
    print(f"USER_ACTION: {user_id} - {action} - {details or {}}")

def log_performance(operation: str, duration: float, user_id: str = None):
    """Log performance"""
    print(f"PERFORMANCE: {operation} - {duration}ms - {user_id or 'unknown'}")

def cleanup_files_by_age(directory: str, hours: int = 72, **kwargs):
    """Cleanup files by age"""
    pass

def create_mock_indicator():
    """Create mock indicator"""
    class MockIndicator:
        def __call__(self, *args, **kwargs):
            return []
    return MockIndicator()

# Safe imports setup
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    DataFrame = pd.DataFrame
    Series = pd.Series
except ImportError:
    PANDAS_AVAILABLE = False
    class MockDataFrame:
        def __init__(self, *args, **kwargs): pass
        def __getitem__(self, key): return []
        def copy(self): return self
    DataFrame = MockDataFrame
    Series = MockDataFrame
    pd = type('pd', (), {'DataFrame': DataFrame, 'Series': Series})()

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class MockNumpy:
        def array(self, x): return x
        def nan(self): return float('nan')
        inf = float('inf')
    np = MockNumpy()

try:
    import ta  # Technical Analysis library
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    ta = None

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    def njit(func): return func
    def prange(x): return range(x)
    NUMBA_AVAILABLE = False

# Get TA indicators if available
if TA_AVAILABLE and ta:
    # Import additional indicators
    try:
        from ta.trend import CCIIndicator, PSARIndicator, TRIXIndicator, DPOIndicator, MassIndex, VortexIndicator, KSTIndicator, IchimokuIndicator, WMAIndicator, SMAIndicator, EMAIndicator
        from ta.volatility import KeltnerChannel
        from ta.momentum import ROCIndicator, WilliamsRIndicator, StochRSIIndicator, UltimateOscillator, RSIIndicator
        from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, ChaikinMoneyFlowIndicator, EaseOfMovementIndicator, ForceIndexIndicator
        from ta.volatility import AverageTrueRange
        from ta.trend import ADXIndicator
    except ImportError:
        # Create mock indicators for missing ones
        MockIndicator = create_mock_indicator()
        CCIIndicator = SMAIndicator = RSIIndicator = ADXIndicator = AverageTrueRange = MockIndicator
        PSARIndicator = TRIXIndicator = DPOIndicator = MassIndex = VortexIndicator = MockIndicator
        KSTIndicator = IchimokuIndicator = WMAIndicator = KeltnerChannel = MockIndicator
        ROCIndicator = WilliamsRIndicator = StochRSIIndicator = UltimateOscillator = MockIndicator
        OnBalanceVolumeIndicator = MFIIndicator = ChaikinMoneyFlowIndicator = MockIndicator
        EaseOfMovementIndicator = ForceIndexIndicator = MockIndicator
else:
    # Create mock indicators when TA not available
    MockIndicator = create_mock_indicator()
    EMAIndicator = SMAIndicator = RSIIndicator = ADXIndicator = AverageTrueRange = MockIndicator
    CCIIndicator = PSARIndicator = TRIXIndicator = DPOIndicator = MassIndex = VortexIndicator = MockIndicator
    KSTIndicator = IchimokuIndicator = WMAIndicator = KeltnerChannel = MockIndicator
    ROCIndicator = WilliamsRIndicator = StochRSIIndicator = UltimateOscillator = MockIndicator
    OnBalanceVolumeIndicator = MFIIndicator = ChaikinMoneyFlowIndicator = MockIndicator
    EaseOfMovementIndicator = ForceIndexIndicator = MockIndicator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_indicator_request(user_id: str, symbol: str, timeframe: str, indicators: List[str], details: dict = None):
    """Log user indicator calculation requests"""
    log_user_action(
        user_id=user_id,
        action=f"indicator_request_{symbol}_{timeframe}",
        details={'indicators': indicators, 'timeframe': timeframe, 'symbol': symbol, **(details or {})},
        logger=logger
    )

def log_indicator_performance(symbol: str, timeframe: str, indicator: str, duration: float, user_id: str = None):
    """Log indicator calculation performance"""
    log_performance(
        operation=f"indicator_{indicator}_{symbol}_{timeframe}",
        duration=duration,
        user_id=user_id,
        logger=logger
    )

def log_indicator_result(user_id: str, symbol: str, timeframe: str, indicators_count: int, signals_count: int, duration: float):
    """Log indicator calculation results"""
    log_user_action(
        user_id=user_id,
        action=f"indicator_result_{symbol}_{timeframe}",
        details={
            'indicators_calculated': indicators_count,
            'signals_generated': signals_count,
            'total_duration_seconds': round(duration, 2)
        },
        logger=logger
    )

# Configuration - now loadable from config file
def load_config():
    """Load configuration from JSON file or use defaults"""
    config_file = "indicator_config.json"
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                logger.info(f"Configuration loaded from {config_file}")
                return config
    except Exception as e:
        logger.warning(f"Could not load config file: {e}")
    
    # Default configuration
    logger.info("Using default configuration")
    return {
        "symbols": ["XAUUSD", "GBPUSD", "GBPJPY"],
        "timeframes": {
            "M15": {"num_candles": 500},
            "M30": {"num_candles": 400},
            "H1": {"num_candles": 400}, 
            "H4": {"num_candles": 150}
        },
        "data_folder": "./data",
        "output_folder": "./indicator_output"
    }

# Load configuration
CONFIG = load_config()
DATA_FOLDER = CONFIG["data_folder"]
INDICATOR_OUTPUT_DIR = CONFIG["output_folder"]
ensure_directory(INDICATOR_OUTPUT_DIR)

symbols = CONFIG["symbols"]
timeframes = list(CONFIG["timeframes"].keys())
num_candles = {tf: CONFIG["timeframes"][tf]["num_candles"] for tf in timeframes}

# ------------------------------------------------------------------
# Runtime feature flags
# ------------------------------------------------------------------
# Enable whitelist filtering to respect GUI selection
DISABLE_INDICATOR_WHITELIST = False
# Use tick_volume as canonical volume
USE_TICK_VOLUME_AS_VOLUME = True

# Auto cleanup on module import
auto_cleanup_on_start([INDICATOR_OUTPUT_DIR, 'logs'], 48)

# -------------------------------------------------------------
# Indicator Whitelist (user-selected indicators only)
# -------------------------------------------------------------
WHITELIST_PATH = os.path.join('analysis_results', 'indicator_whitelist.json')

def load_indicator_whitelist():
    if 'DISABLE_INDICATOR_WHITELIST' in globals() and DISABLE_INDICATOR_WHITELIST:
        logger.info("Indicator whitelist disabled; exporting all indicators.")
        return None
    try:
        if os.path.exists(WHITELIST_PATH):
            with open(WHITELIST_PATH, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                if isinstance(raw, list):
                    wl = {normalize_indicator_key(x) for x in raw if isinstance(x, str)}
                    logger.info(f"Loaded indicator whitelist: {wl}")
                    return wl
    except Exception as e:
        logger.warning(f"Failed loading whitelist, using full set: {e}")
    return None  # None => no filtering

def normalize_indicator_key(name: str) -> str:
    return ''.join(ch for ch in name.lower() if ch.isalnum())

INDICATOR_WHITELIST = load_indicator_whitelist()

def indicator_allowed(name: str) -> bool:
    if INDICATOR_WHITELIST is None:
        return True
    # Normalize both
    key = normalize_indicator_key(name)
    # Provide a few synonym normalizations
    synonyms = {
        'bollingerbands': 'bollinger',
        'bollinger': 'bollinger',
        'stochastic': 'stochastic',
        'stochrsi': 'stochrsi',
        'movingaverage': 'ma',
        'ema': 'ema',
        'sma': 'sma',
        'mass': 'massindex'
    }
    if key in synonyms:
        key = synonyms[key]
    return key in INDICATOR_WHITELIST

def collect_selected_indicator_columns(df, selected_indicator_names):
    """Infer which columns belong to selected indicators for pruning output.
    Keep base OHLC/time/volume columns + selected indicator columns + their signals.
    """
    base_cols = {'time','open','high','low','close','volume','tick_volume'} & set(df.columns)
    keep = set(base_cols)
    # Mapping of indicator logical name -> column patterns
    patterns = {
        'rsi': ['RSI', 'rsi'],
        'macd': ['MACD_', 'MACD_signal_', 'MACD_hist_', 'macd'],
        'atr': ['ATR', 'atr'],
        'adx': ['ADX', 'adx'],
        'donchian': ['Donchian_Upper', 'Donchian_Middle', 'Donchian_Lower', 'donchian_upper','donchian_middle','donchian_lower'],
        'stochrsi': ['StochRSI', 'StochRSI_K_', 'StochRSI_D_', 'stochrsi'],
        'stochastic': ['StochK_', 'StochD_', 'stoch_k','stoch_d'],
        'ma': ['SMA', 'EMA', 'WMA', 'TEMA'],
        'ema': ['EMA'],
        'ema10': ['EMA10'],
        'ema20': ['EMA20'],
        'ema50': ['EMA50'],
        'ema100': ['EMA100'],
        'ema200': ['EMA200'],
        'sma': ['SMA'],
        'sma20': ['SMA20'],
        'sma50': ['SMA50'],
        'bollinger': ['BB_Upper', 'BB_Middle', 'BB_Lower', 'bb_upper','bb_middle','bb_lower'],
        'psar': ['PSAR', 'psar'],
        'mfi': ['MFI', 'mfi'],
        'obv': ['OBV', 'obv'],
        'cci': ['CCI', 'cci'],
        'williamsr': ['WilliamsR', 'williamsr', 'williams_r'],
        'roc': ['ROC', 'roc'],
        'ichimoku': ['ichimoku_','tenkan_','kijun_','senkou_a_','senkou_b_','chikou_'],
        'envelope': ['Envelope_Upper','Envelope_Middle','Envelope_Lower','envelope_upper','envelope_lower','envelope_middle'],
        'keltner': ['Keltner_Upper','Keltner_Middle','Keltner_Lower'],
        'dpo': ['DPO'],
        'trix': ['TRIX'],
        'massindex': ['MassIndex_'],
        'mass': ['MassIndex_'],
        'vortex': ['VI+_','VI-_'],
        'kst': ['KST','KST_sig'],
        'ultimateoscillator': ['UltimateOscillator'],
        'forceindex': ['ForceIndex'],
        'chaikin': ['Chaikin'],
        'eom': ['EOM'],
        'vwap': ['VWAP','vwap'],
        'fibonacci': ['fib_','fib0.','fib_236','fib_382','fib_500','fib_618','fib_786']
    }
    normalized_selected = {normalize_indicator_key(n) for n in selected_indicator_names}
    logger.info(f"[collect_selected_indicator_columns] normalized_selected: {normalized_selected}")
    logger.info(f"[collect_selected_indicator_columns] Available columns: {list(df.columns)}")
    
    # First pass: keep underlying indicator value columns
    for col in df.columns:
        if col.endswith('_signal') or col == 'overall_signal':
            continue  # handle in second pass
        for indicator_key in normalized_selected:
            pats = patterns.get(indicator_key, [])
            for p in pats:
                if col.startswith(p):
                    keep.add(col)
                    break
            if col in keep:
                break

    # Second pass: add signal columns only if their base indicator is kept (or overall_signal)
    kept_lower = {k.lower() for k in keep}
    for col in df.columns:
        if col == 'overall_signal':
            keep.add(col)
            continue
        if col.endswith('_signal'):
            base = col[:-7].lower()  # remove '_signal'
            # Keep if any kept column shares prefix or base shares prefix with kept
            related = any((base.startswith(k) or k.startswith(base)) for k in kept_lower)
            if related:
                keep.add(col)
    return list(keep)

def fetch_candles_from_json(symbol: str, timeframe: str, count: int):
    """
    Load candle data from JSON file with improved error handling and validation
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe (M15, H1, H4)
        count: Number of candles needed
        
    Returns:
        DataFrame with validated candle data or None if error
    """
    try:
        # Tìm file phù hợp với mọi định dạng tên
        pattern = os.path.join(DATA_FOLDER, f"{symbol}*_{timeframe}.json")
        files = glob.glob(pattern)
        
        if not files:
            logger.warning(f"No data file found for {symbol} {timeframe} (pattern: {pattern})")
            return None
            
        file_path = files[0]  # Lấy file đầu tiên khớp
        logger.debug(f"Loading data from: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            candles = json.load(f)
            
        if len(candles) < count:
            logger.warning(f"File {file_path} has only {len(candles)} candles, requested {count}")
            # Use available candles if less than requested
            if len(candles) < 50:  # Minimum threshold
                logger.error(f"Too few candles ({len(candles)}) for meaningful analysis")
                return None
                
        # Take the last 'count' candles or all available
        candles = candles[-min(count, len(candles)):]
        df = pd.DataFrame(candles)
        
        # Validate and convert data
        df = validate_candle_data(df)
        if df is None:
            return None

        # Map tick_volume to volume if requested
        if USE_TICK_VOLUME_AS_VOLUME and 'tick_volume' in df.columns:
            try:
                df['volume'] = df['tick_volume']
            except Exception as e:
                logger.warning(f"Could not map tick_volume to volume: {e}")
            
        logger.info(f"Successfully loaded {len(df)} candles for {symbol} {timeframe}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading candles for {symbol} {timeframe}: {e}")
        return None

def validate_candle_data(df):
    """Validate and clean candle data"""
    try:
        # Convert time column
        df['time'] = pd.to_datetime(df['time'])
        
        # Validate required price columns
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return None
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Handle volume data
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        elif 'tick_volume' in df.columns:
            df['volume'] = pd.to_numeric(df['tick_volume'], errors='coerce')
        else:
            # Create dummy volume if not available
            df['volume'] = 1.0
            logger.warning("No volume data available, using dummy values")
            
        # Remove rows with invalid data
        initial_len = len(df)
        df = df.dropna(subset=required_cols + ['volume'])
        
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} rows with invalid data")
            
        if len(df) == 0:
            logger.error("No valid data remaining after cleaning")
            return None
            
        return df
        
    except Exception as e:
        logger.error(f"Error validating candle data: {e}")
        return None

# --- SMA ---
@njit(parallel=True)
def fast_sma(arr, period):
    n = len(arr)
    out = np.full(n, np.nan)
    cumsum = np.zeros(n + 1)
    for i in range(n):
        cumsum[i + 1] = cumsum[i] + arr[i]
    for i in prange(period - 1, n):
        out[i] = (cumsum[i + 1] - cumsum[i + 1 - period]) / period
    return out

# --- Optimized calculation functions with numba ---
@njit(parallel=True)
def fast_rsi(close, period):
    """Fast RSI calculation using numba optimization"""
    n = len(close)
    rsi = np.full(n, np.nan)
    for i in prange(period, n):
        gains = 0.0
        losses = 0.0
        for j in range(i-period+1, i+1):
            diff = close[j] - close[j-1]
            if diff > 0:
                gains += diff
            else:
                losses -= diff
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    return rsi

def calc_dpo(df, period=20):
    """Calculate Detrended Price Oscillator"""
    if TA_AVAILABLE:
        from ta.trend import DPOIndicator
        return DPOIndicator(close=df['close'], window=period).dpo()
    else:
        # Basic fallback calculation
        sma = df['close'].rolling(window=period).mean()
        shift_value = (period // 2) + 1
        return df['close'] - sma.shift(shift_value)

# =============================================================================
# INTEGRATED UTILITY FUNCTIONS (Previously in utils_indicators.py and utils_signals.py)
# =============================================================================

def calc_sma(close, period: int = 20):
    """Calculate Simple Moving Average"""
    if TA_AVAILABLE:
        return SMAIndicator(close, window=period).sma_indicator()
    else:
        return close.rolling(window=period).mean()

def calc_ema(close, period: int = 20):
    """Calculate Exponential Moving Average"""
    if TA_AVAILABLE:
        return EMAIndicator(close, window=period).ema_indicator()
    else:
        return close.ewm(span=period).mean()

def calc_wma(close, period: int = 20):
    """Calculate Weighted Moving Average"""
    # Ensure we're working with numeric data only
    if hasattr(close, 'dtype') and 'datetime' in str(close.dtype):
        raise ValueError("WMA cannot be calculated on datetime columns")
    
    # Convert to numeric if needed
    close_numeric = pd.to_numeric(close, errors='coerce')
    
    weights = np.arange(1, period + 1)
    return close_numeric.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

def calc_tema(close, period: int = 20):
    """Calculate Triple Exponential Moving Average"""
    ema1 = calc_ema(close, period)
    ema2 = calc_ema(ema1, period)
    ema3 = calc_ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3

def calc_vwap(df):
    """Calculate Volume Weighted Average Price"""
    if 'volume' not in df.columns:
        # Use tick_volume as fallback
        volume = df.get('tick_volume', pd.Series([1] * len(df)))
    else:
        volume = df['volume']
    
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_volume = volume.cumsum()
    cumulative_typical_price_volume = (typical_price * volume).cumsum()
    
    return cumulative_typical_price_volume / cumulative_volume

def integrated_calc_rsi(close, period: int = 14):
    """Calculate Relative Strength Index"""
    if TA_AVAILABLE:
        return RSIIndicator(close, window=period).rsi()
    else:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Handle division by zero
        rs = pd.Series(index=close.index, dtype=float)
        for i in range(len(avg_gain)):
            if avg_loss.iloc[i] == 0:
                rs.iloc[i] = 100 if avg_gain.iloc[i] > 0 else 50
            else:
                rs.iloc[i] = avg_gain.iloc[i] / avg_loss.iloc[i]
        
        rsi = 100 - (100 / (1 + rs))
        return rsi

def integrated_calc_atr(df, period: int = 14):
    """Calculate Average True Range"""
    if TA_AVAILABLE:
        return AverageTrueRange(
            high=df['high'], 
            low=df['low'], 
            close=df['close'], 
            window=period
        ).average_true_range()
    else:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

def integrated_calc_bollinger_bands(close, window: int = 20, std_dev: float = 2):
    """Calculate Bollinger Bands"""
    sma = calc_sma(close, window)
    std = close.rolling(window=window).std()
    
    return {
        'bb_middle': sma,
        'bb_upper': sma + (std * std_dev),
        'bb_lower': sma - (std * std_dev),
        'bb_width': (sma + (std * std_dev)) - (sma - (std * std_dev)),
        'bb_percent': (close - (sma - (std * std_dev))) / ((sma + (std * std_dev)) - (sma - (std * std_dev)))
    }

def integrated_calc_macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD"""
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_histogram': histogram
    }

def integrated_calc_stochastic(df, period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    """Calculate Stochastic Oscillator"""
    lowest_low = df['low'].rolling(window=period).min()
    highest_high = df['high'].rolling(window=period).max()
    
    k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    k_smooth = k_percent.rolling(window=smooth_k).mean()
    d_smooth = k_smooth.rolling(window=smooth_d).mean()
    
    return {
        'stoch_k': k_smooth,
        'stoch_d': d_smooth
    }

def calc_williams_r(df, period: int = 14):
    """Calculate Williams %R"""
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    
    return -100 * ((highest_high - df['close']) / (highest_high - lowest_low))

def calc_cci(df, period: int = 20):
    """Calculate Commodity Channel Index"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    
    return (typical_price - sma_tp) / (0.015 * mad)

def calc_roc(close, period: int = 12):
    """Calculate Rate of Change"""
    return ((close - close.shift(period)) / close.shift(period)) * 100

def calc_donchian_channel(df, window: int = 20):
    """Calculate Donchian Channel and return upper, middle, lower series (legacy compat)."""
    upper = df['high'].rolling(window=window).max()
    lower = df['low'].rolling(window=window).min()
    middle = (upper + lower) / 2
    return upper, middle, lower

def calc_envelope(close, period: int = 20, percent: float = 2.0):
    """Calculate Envelope"""
    sma = calc_sma(close, period)
    multiplier = percent / 100
    
    return {
        'envelope_upper': sma * (1 + multiplier),
        'envelope_lower': sma * (1 - multiplier),
        'envelope_middle': sma
    }

def calc_ichimoku(df, tenkan: int = 9, kijun: int = 26, senkou: int = 52):
    """Calculate Ichimoku Cloud and return component series (tenkan, kijun, senkou_a, senkou_b, chikou).
    Single canonical implementation (removed duplicate later definition)."""
    tenkan_sen = (df['high'].rolling(window=tenkan).max() + df['low'].rolling(window=tenkan).min()) / 2
    kijun_sen = (df['high'].rolling(window=kijun).max() + df['low'].rolling(window=kijun).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_span_b = ((df['high'].rolling(window=senkou).max() + df['low'].rolling(window=senkou).min()) / 2).shift(kijun)
    chikou_span = df['close'].shift(-kijun)
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def cleanup_indicator_output_files(indicator_output_dir: str = "indicator_output", max_age_hours: int = 48) -> Dict[str, Any]:
    """Clean up old indicator files in the indicator_output directory.
    
    Args:
        indicator_output_dir: Directory containing indicator files
        max_age_hours: Maximum age in hours before files are deleted
        
    Returns:
        Dict with cleanup statistics
    """
    try:
        if not os.path.exists(indicator_output_dir):
            return {"deleted": 0, "space_freed": 0, "message": "Directory not found"}
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0
        space_freed_mb = 0.0
        
        # Get all JSON files in indicator_output directory
        pattern = os.path.join(indicator_output_dir, "*.json")
        indicator_files = glob.glob(pattern)
        
        print(f"🧹 Cleaning indicator files older than {max_age_hours} hours...")
        
        for file_path in indicator_files:
            try:
                # Get file modification time
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_mtime < cutoff_time:
                    # Get file size before deletion
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    
                    # Delete the file
                    os.remove(file_path)
                    deleted_count += 1
                    space_freed_mb += file_size_mb
                    
                    print(f"   Deleted: {os.path.basename(file_path)} ({file_size_mb:.2f} MB)")
                    
            except Exception as e:
                print(f"   Error deleting {file_path}: {e}")
                continue
        
        result = {
            "deleted": deleted_count,
            "space_freed": round(space_freed_mb, 2),
            "message": f"Cleaned {deleted_count} files, freed {space_freed_mb:.2f} MB"
        }
        
        if deleted_count > 0:
            print(f"✅ Cleanup complete: {deleted_count} files deleted, {space_freed_mb:.2f} MB freed")
        else:
            print("✅ No old files found to delete")
            
        return result
        
    except Exception as e:
        error_msg = f"Error during cleanup: {e}"
        print(f"❌ {error_msg}")
        return {"deleted": 0, "space_freed": 0, "error": error_msg}

# =============================================================================
# SIGNAL GENERATION FUNCTIONS (Previously in utils_signals.py)
# =============================================================================

def add_trend_signals(df):
    """Add trend-based signals"""
    # MA crossover signals
    if 'ma_20' in df.columns and 'ma_50' in df.columns:
        df['ma_crossover_signal'] = 'Hold'
        df.loc[df['ma_20'] > df['ma_50'], 'ma_crossover_signal'] = 'Buy'
        df.loc[df['ma_20'] < df['ma_50'], 'ma_crossover_signal'] = 'Sell'
    
    # MACD signals
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_signal_flag'] = 'Hold'
        df.loc[df['macd'] > df['macd_signal'], 'macd_signal_flag'] = 'Buy'
        df.loc[df['macd'] < df['macd_signal'], 'macd_signal_flag'] = 'Sell'
    
    # ADX trend strength (expanded categorical mapping)
    if 'adx' in df.columns:
        # Preserve existing column for backward compatibility
        df['adx_trend_strength'] = 'Weak'
        df.loc[df['adx'] > 25, 'adx_trend_strength'] = 'Strong'
        df.loc[df['adx'] > 50, 'adx_trend_strength'] = 'Very Strong'

        # New granular category column
        adx_vals = df['adx']
        categories = pd.Series(index=df.index, dtype='object')
        categories[:] = 'Very_Weak'
        categories[(adx_vals >= 15) & (adx_vals < 20)] = 'Weak'
        categories[(adx_vals >= 20) & (adx_vals < 25)] = 'Developing'
        categories[(adx_vals >= 25) & (adx_vals < 40)] = 'Strong'
        categories[(adx_vals >= 40) & (adx_vals < 50)] = 'Very_Strong'
        categories[adx_vals >= 50] = 'Extreme'
        df['adx_trend_category'] = categories
    
    # Parabolic SAR signals (support both 'PSAR' and 'psar' column naming)
    if ('psar' in df.columns or 'PSAR' in df.columns) and 'close' in df.columns:
        psar_col = 'psar' if 'psar' in df.columns else 'PSAR'
        # Create lowercase alias if only uppercase exists for downstream uniformity
        if psar_col == 'PSAR' and 'psar' not in df.columns:
            try:
                df['psar'] = df['PSAR']
            except Exception:
                pass
        psar_col = 'psar' if 'psar' in df.columns else psar_col
        df['psar_signal'] = 'Hold'
        try:
            df.loc[df['close'] > df[psar_col], 'psar_signal'] = 'Buy'
            df.loc[df['close'] < df[psar_col], 'psar_signal'] = 'Sell'
        except Exception as e:
            logger.error(f"Error assigning psar_signal: {e}")
        # Directional numeric + trend text fields expected by aggregator
        try:
            df['PSAR_dir'] = 0
            df.loc[df['close'] > df[psar_col], 'PSAR_dir'] = 1
            df.loc[df['close'] < df[psar_col], 'PSAR_dir'] = -1
            df['PSAR_trend'] = 'neutral'
            df.loc[df['PSAR_dir'] == 1, 'PSAR_trend'] = 'bullish'
            df.loc[df['PSAR_dir'] == -1, 'PSAR_trend'] = 'bearish'
        except Exception as e:
            logger.error(f"Error deriving PSAR_dir/PSAR_trend: {e}")
    
    return df

def add_momentum_signals(df):
    """Add momentum-based signals"""
    # RSI signals
    if 'rsi' in df.columns:
        df['rsi_signal'] = 'Hold'
        df.loc[df['rsi'] < 30, 'rsi_signal'] = 'Oversold'
        df.loc[df['rsi'] > 70, 'rsi_signal'] = 'Overbought'
        df.loc[(df['rsi'] >= 30) & (df['rsi'] <= 70), 'rsi_signal'] = 'Neutral'
    
    # Stochastic signals
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        df['stoch_signal'] = 'Hold'
        df.loc[(df['stoch_k'] < 20) & (df['stoch_d'] < 20), 'stoch_signal'] = 'Oversold'
        df.loc[(df['stoch_k'] > 80) & (df['stoch_d'] > 80), 'stoch_signal'] = 'Overbought'
        df.loc[df['stoch_k'] > df['stoch_d'], 'stoch_signal'] = 'Buy'
        df.loc[df['stoch_k'] < df['stoch_d'], 'stoch_signal'] = 'Sell'
    
    # Williams %R signals
    if 'williams_r' in df.columns:
        df['williams_r_signal'] = 'Hold'
        df.loc[df['williams_r'] < -80, 'williams_r_signal'] = 'Oversold'
        df.loc[df['williams_r'] > -20, 'williams_r_signal'] = 'Overbought'
    
    # CCI signals
    if 'cci' in df.columns:
        df['cci_signal'] = 'Hold'
        df.loc[df['cci'] < -100, 'cci_signal'] = 'Oversold'
        df.loc[df['cci'] > 100, 'cci_signal'] = 'Overbought'
    
    # ROC signals
    if 'roc' in df.columns:
        df['roc_signal'] = 'Hold'
        df.loc[df['roc'] > 0, 'roc_signal'] = 'Buy'
        df.loc[df['roc'] < 0, 'roc_signal'] = 'Sell'
    
    return df

def add_volatility_signals(df):
    """Add volatility-based signals"""
    # Bollinger Bands signals
    if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'close']):
        df['bb_signal'] = 'Hold'
        df.loc[df['close'] > df['bb_upper'], 'bb_signal'] = 'Overbought'
        df.loc[df['close'] < df['bb_lower'], 'bb_signal'] = 'Oversold'
        
        # BB squeeze detection
        if 'bb_width' in df.columns:
            bb_width_ma = df['bb_width'].rolling(window=20).mean()
            df['bb_squeeze'] = 'Normal'
            df.loc[df['bb_width'] < bb_width_ma * 0.5, 'bb_squeeze'] = 'Squeeze'
    
    # ATR volatility analysis
    if 'atr' in df.columns:
        atr_ma = df['atr'].rolling(window=14).mean()
        df['atr_volatility'] = 'Normal'
        df.loc[df['atr'] > atr_ma * 1.5, 'atr_volatility'] = 'High'
        df.loc[df['atr'] < atr_ma * 0.5, 'atr_volatility'] = 'Low'
    
    # Donchian Channel signals
    if all(col in df.columns for col in ['donchian_upper', 'donchian_lower', 'close']):
        df['donchian_signal'] = 'Hold'
        df.loc[df['close'] >= df['donchian_upper'], 'donchian_signal'] = 'Breakout_High'
        df.loc[df['close'] <= df['donchian_lower'], 'donchian_signal'] = 'Breakout_Low'
    
    return df

def add_volume_signals(df):
    """Add volume-based signals"""
    # Volume analysis
    if 'volume' in df.columns or 'tick_volume' in df.columns:
        volume_col = 'volume' if 'volume' in df.columns else 'tick_volume'
        volume_ma = df[volume_col].rolling(window=20).mean()
        
        df['volume_signal'] = 'Normal'
        df.loc[df[volume_col] > volume_ma * 2, 'volume_signal'] = 'High'
        df.loc[df[volume_col] < volume_ma * 0.5, 'volume_signal'] = 'Low'
    
    # OBV signals
    if 'obv' in df.columns:
        obv_ma = df['obv'].rolling(window=20).mean()
        df['obv_signal'] = 'Hold'
        df.loc[df['obv'] > obv_ma, 'obv_signal'] = 'Buy'
        df.loc[df['obv'] < obv_ma, 'obv_signal'] = 'Sell'
    
    # MFI signals
    if 'mfi' in df.columns:
        df['mfi_signal'] = 'Hold'
        df.loc[df['mfi'] < 20, 'mfi_signal'] = 'Oversold'
        df.loc[df['mfi'] > 80, 'mfi_signal'] = 'Overbought'
    
    # VWAP signals
    if 'vwap' in df.columns and 'close' in df.columns:
        df['vwap_signal'] = 'Hold'
        df.loc[df['close'] > df['vwap'], 'vwap_signal'] = 'Above_VWAP'
        df.loc[df['close'] < df['vwap'], 'vwap_signal'] = 'Below_VWAP'
    
    return df

def add_support_resistance_signals(df):
    """Add support/resistance level signals"""
    # Fibonacci levels
    fib_levels = ['fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786']
    if 'close' in df.columns and any(level in df.columns for level in fib_levels):
        df['fib_signal'] = 'Neutral'
        for level in fib_levels:
            if level in df.columns:
                # Price near Fibonacci level (within 0.1%)
                near_level = abs(df['close'] - df[level]) / df['close'] < 0.001
                df.loc[near_level, 'fib_signal'] = f'Near_{level}'
    
    # Envelope signals
    if all(col in df.columns for col in ['envelope_upper', 'envelope_lower', 'close']):
        df['envelope_signal'] = 'Hold'
        df.loc[df['close'] > df['envelope_upper'], 'envelope_signal'] = 'Overbought'
        df.loc[df['close'] < df['envelope_lower'], 'envelope_signal'] = 'Oversold'
    
    return df

def add_ichimoku_signals(df):
    """Add Ichimoku-specific signals"""
    ichimoku_cols = ['ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b']
    
    if all(col in df.columns for col in ichimoku_cols) and 'close' in df.columns:
        try:
            shift_period = 26
            tenkan = df['ichimoku_tenkan']
            kijun = df['ichimoku_kijun']
            span_a = df['ichimoku_senkou_a']
            span_b = df['ichimoku_senkou_b']

            # Tenkan / Kijun relationship
            bullish_now = (tenkan > kijun) & tenkan.notna() & kijun.notna()
            bearish_now = (tenkan < kijun) & tenkan.notna() & kijun.notna()
            prev_bullish = bullish_now.shift(1).fillna(False)
            prev_bearish = bearish_now.shift(1).fillna(False)
            fresh_bull_cross = bullish_now & prev_bearish
            fresh_bear_cross = bearish_now & prev_bullish

            # Adaptive distance filtering to avoid over-Flat classification
            price_ref = df['close'] if 'close' in df.columns else kijun
            dist = (tenkan - kijun).abs()
            # Percent distance vs price
            with np.errstate(divide='ignore', invalid='ignore'):
                dist_pct = (dist / price_ref).abs()
            # Base tiny threshold (10th percentile or fallback)
            tiny_thr = dist_pct.quantile(0.10) if dist_pct.notna().any() else 0
            tiny_thr = max(tiny_thr, 0.00002)  # safeguard floor (~0.2 bps)

            # ATR reference optional for alternative thresholding
            if 'atr_14' in df.columns:
                atr_ref = df['atr_14']
                # dynamic: choose smaller of ATR-based % and percentile expansion
                atr_thr = (atr_ref * 0.02)  # 2% of ATR
                dist_ok = (dist > atr_thr) | (dist_pct > tiny_thr * 2)
            else:
                # Use interquartile style threshold (25th percentile * factor)
                q25 = dist_pct.quantile(0.25) if dist_pct.notna().any() else 0
                dist_ok = dist_pct > max(q25 * 0.5, tiny_thr * 1.5)

            tk_signal = pd.Series('Flat', index=df.index)
            # Primary assignments where distance is meaningful
            tk_signal[bullish_now & dist_ok] = 'Bullish'
            tk_signal[bearish_now & dist_ok] = 'Bearish'
            tk_signal[fresh_bull_cross & dist_ok] = 'Bullish_Cross'
            tk_signal[fresh_bear_cross & dist_ok] = 'Bearish_Cross'

            # Secondary pass: sustained direction (>=2 of last 3 bars) even if distance small
            sustained_bull = (bullish_now.rolling(3).sum() >= 2) & bullish_now
            sustained_bear = (bearish_now.rolling(3).sum() >= 2) & bearish_now
            tk_signal[(tk_signal=='Flat') & sustained_bull & (dist_pct > tiny_thr)] = 'Bullish'
            tk_signal[(tk_signal=='Flat') & sustained_bear & (dist_pct > tiny_thr)] = 'Bearish'

            # Relax if still overwhelmingly Flat (>70%) by reclassifying remaining directional bars
            flat_ratio = tk_signal.value_counts(normalize=True).get('Flat', 0)
            if flat_ratio > 0.70:
                tk_signal[(tk_signal=='Flat') & bullish_now] = 'Bullish'
                tk_signal[(tk_signal=='Flat') & bearish_now] = 'Bearish'
                tk_signal[(tk_signal=='Flat') & fresh_bull_cross] = 'Bullish_Cross'
                tk_signal[(tk_signal=='Flat') & fresh_bear_cross] = 'Bearish_Cross'
            df['ichimoku_tk_signal'] = tk_signal

            # Cloud alignment (first try shifted alignment)
            if len(df) > shift_period:
                span_a_aligned = span_a.shift(-shift_period)
                span_b_aligned = span_b.shift(-shift_period)
            else:
                span_a_aligned = span_a.copy()
                span_b_aligned = span_b.copy()

            cloud_top = np.maximum(span_a_aligned, span_b_aligned)
            cloud_bottom = np.minimum(span_a_aligned, span_b_aligned)
            cloud_sig = pd.Series('Neutral', index=df.index)
            above_cloud = (df['close'] > cloud_top) & cloud_top.notna()
            below_cloud = (df['close'] < cloud_bottom) & cloud_bottom.notna()
            in_cloud = (~above_cloud) & (~below_cloud) & cloud_top.notna() & cloud_bottom.notna()
            cloud_sig[above_cloud] = 'Above_Cloud'
            cloud_sig[below_cloud] = 'Below_Cloud'
            cloud_sig[in_cloud] = 'In_Cloud'

            # Fallback: if virtually all Neutral then recompute without alignment shift
            if (cloud_sig.value_counts(normalize=True).get('Neutral', 0) > 0.95):
                cloud_top2 = np.maximum(span_a, span_b)
                cloud_bottom2 = np.minimum(span_a, span_b)
                above_cloud2 = (df['close'] > cloud_top2) & cloud_top2.notna()
                below_cloud2 = (df['close'] < cloud_bottom2) & cloud_bottom2.notna()
                in_cloud2 = (~above_cloud2) & (~below_cloud2) & cloud_top2.notna() & cloud_bottom2.notna()
                cloud_sig[:] = 'Neutral'
                cloud_sig[above_cloud2] = 'Above_Cloud'
                cloud_sig[below_cloud2] = 'Below_Cloud'
                cloud_sig[in_cloud2] = 'In_Cloud'
            df['ichimoku_cloud_signal'] = cloud_sig

            # Chikou confirmation
            if 'ichimoku_chikou' in df.columns:
                chikou_confirm = (df['close'] > df['close'].shift(shift_period)) & df['close'].notna()
            else:
                chikou_confirm = pd.Series(False, index=df.index)

            bias_list = []
            # Optional momentum boost: price relative to Kijun & cloud thickness
            price = df['close'] if 'close' in df.columns else kijun
            cloud_thickness = (span_a - span_b).abs()
            thick_quantile = cloud_thickness.quantile(0.6) if cloud_thickness.notna().any() else 0
            for i,(tk, cloud, confirm) in enumerate(zip(df['ichimoku_tk_signal'], df['ichimoku_cloud_signal'], chikou_confirm)):
                p = price.iat[i] if i < len(price) else np.nan
                kj = kijun.iat[i] if i < len(kijun) else np.nan
                thick = cloud_thickness.iat[i] if i < len(cloud_thickness) else np.nan
                momentum_up = pd.notna(p) and pd.notna(kj) and (p - kj) / kj > 0.001  # >0.1%
                momentum_down = pd.notna(p) and pd.notna(kj) and (kj - p) / kj > 0.001
                thick_cloud = pd.notna(thick) and thick > thick_quantile
                if cloud == 'Above_Cloud' and tk in ('Bullish','Bullish_Cross') and confirm and (momentum_up or thick_cloud):
                    bias_list.append('Strong_Bullish')
                elif cloud == 'Above_Cloud' and tk in ('Bullish','Bullish_Cross'):
                    bias_list.append('Bullish')
                elif cloud == 'In_Cloud':
                    bias_list.append('Neutral')
                elif cloud == 'Below_Cloud' and tk in ('Bearish','Bearish_Cross') and confirm and (momentum_down or thick_cloud):
                    bias_list.append('Strong_Bearish')
                elif cloud == 'Below_Cloud' and tk in ('Bearish','Bearish_Cross'):
                    bias_list.append('Bearish')
                else:
                    bias_list.append('Mixed')
            df['ichimoku_bias'] = bias_list

            # If Mixed dominates (>70%), reclassify some Mixed based on weaker heuristics to add directionality
            mix_ratio = pd.Series(bias_list).value_counts(normalize=True).get('Mixed',0)
            if mix_ratio > 0.7:
                bias_series = df['ichimoku_bias']
                # Promote Mixed where tk bullish & above cloud to Bullish
                promote_long = (df['ichimoku_bias']=='Mixed') & (df['ichimoku_tk_signal'].isin(['Bullish','Bullish_Cross'])) & (df['ichimoku_cloud_signal']=='Above_Cloud')
                promote_short = (df['ichimoku_bias']=='Mixed') & (df['ichimoku_tk_signal'].isin(['Bearish','Bearish_Cross'])) & (df['ichimoku_cloud_signal']=='Below_Cloud')
                bias_series.loc[promote_long] = 'Bullish'
                bias_series.loc[promote_short] = 'Bearish'
                df['ichimoku_bias'] = bias_series

            # Debug distribution logs (only once per call)
            try:
                logger.info(
                    "Ichimoku TK counts: %s | Cloud counts: %s | Bias counts: %s" % (
                        df['ichimoku_tk_signal'].value_counts().to_dict(),
                        df['ichimoku_cloud_signal'].value_counts().to_dict(),
                        df['ichimoku_bias'].value_counts().to_dict()
                    )
                )
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Ichimoku signal derivation error: {e}")
    
    return df

def add_comprehensive_signals(df):
    """Add all signal categories with per-phase error isolation and indicator alias normalization."""

    try:
        df = normalize_indicator_aliases(df)
    except Exception as e:
        logger.error(f"Alias normalization failed: {e}")

    for fn, label in [
        (add_trend_signals, 'trend'),
        (add_momentum_signals, 'momentum'),
        (add_volatility_signals, 'volatility'),
        (add_volume_signals, 'volume'),
        (add_support_resistance_signals, 'support_resistance'),
        (add_ichimoku_signals, 'ichimoku')
    ]:
        try:
            df = fn(df)
        except Exception as e:
            logger.error(f"Signal phase '{label}' failed: {e}")

    return df

def generate_overall_signal(df):
    """Generate overall trading signal based on multiple indicators"""
    # Define signal columns to analyze
    buy_signals = []
    sell_signals = []
    
    # Collect all signal columns
    signal_columns = [col for col in df.columns if '_signal' in col]
    
    for idx in df.index:
        buy_count = 0
        sell_count = 0
        total_signals = 0
        
        for col in signal_columns:
            if col in df.columns:
                signal = df.loc[idx, col]
                total_signals += 1
                
                if signal in ['Buy', 'Oversold', 'Above_VWAP', 'Above_Cloud']:
                    buy_count += 1
                elif signal in ['Sell', 'Overbought', 'Below_VWAP', 'Below_Cloud']:
                    sell_count += 1
        
        buy_signals.append(buy_count)
        sell_signals.append(sell_count)
    
    df['buy_signal_count'] = buy_signals
    df['sell_signal_count'] = sell_signals
    df['overall_signal'] = [
        'Strong_Buy' if b > s * 1.5 and b > 3 else
        'Buy' if b > s and b > 2 else
        'Strong_Sell' if s > b * 1.5 and s > 3 else
        'Sell' if s > b and s > 2 else
        'Hold'
        for b, s in zip(buy_signals, sell_signals)
    ]
    
    return df

# =============================================================================
# ADDITIONAL INDICATOR FUNCTIONS WITH TA-LIB INTEGRATION
# =============================================================================

def calc_adx(df, period=14):
    if TA_AVAILABLE:
        from ta.trend import ADXIndicator
        return ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=period).adx()
    else:
        # Basic fallback calculation
        return pd.Series([25] * len(df), index=df.index)

def calc_obv(df):
    if TA_AVAILABLE:
        from ta.volume import OnBalanceVolumeIndicator
        return OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    else:
        # Basic fallback calculation
        return df['volume'].cumsum()

def calc_mfi(df, period=14):
    if TA_AVAILABLE:
        from ta.volume import MFIIndicator
        return MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=period).money_flow_index()
    else:
        # Basic fallback calculation
        return pd.Series([50] * len(df), index=df.index)

def calc_psar(df):
    if TA_AVAILABLE:
        from ta.trend import PSARIndicator
        return PSARIndicator(high=df['high'], low=df['low'], close=df['close']).psar()
    else:
        # Basic fallback calculation
        return df['close']

def calc_chaikin(df, period=20):
    if TA_AVAILABLE:
        from ta.volume import ChaikinMoneyFlowIndicator
        return ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=period).chaikin_money_flow()
    else:
        # Manual Chaikin Money Flow calculation
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        money_flow_multiplier = money_flow_multiplier.fillna(0)  # Handle division by zero
        money_flow_volume = money_flow_multiplier * df['volume']
        cmf = money_flow_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return cmf.fillna(0)

def calc_eom(df, period=14):
    if TA_AVAILABLE:
        from ta.volume import EaseOfMovementIndicator
        return EaseOfMovementIndicator(high=df['high'], low=df['low'], volume=df['volume'], window=period).ease_of_movement()
    else:
        # Basic fallback calculation
        return pd.Series([0] * len(df), index=df.index)

def calc_force_index(df, period=13):
    if TA_AVAILABLE:
        from ta.volume import ForceIndexIndicator
        return ForceIndexIndicator(close=df['close'], volume=df['volume'], window=period).force_index()
    else:
        # Basic fallback calculation
        return pd.Series([0] * len(df), index=df.index)

def calc_trix(df, period=15):
    if TA_AVAILABLE:
        from ta.trend import TRIXIndicator
        return TRIXIndicator(close=df['close'], window=period).trix()
    else:
        # Basic fallback calculation
        return pd.Series([0] * len(df), index=df.index)

def calc_mass_index(df, fast=9, slow=25):
    if TA_AVAILABLE:
        from ta.trend import MassIndex
        return MassIndex(high=df['high'], low=df['low'], window_fast=fast, window_slow=slow).mass_index()
    else:
        # Basic fallback calculation
        return pd.Series([25] * len(df), index=df.index)

def calc_vortex(df, period=14):
    if TA_AVAILABLE:
        from ta.trend import VortexIndicator
        vi = VortexIndicator(high=df['high'], low=df['low'], close=df['close'], window=period)
        return vi.vortex_indicator_pos(), vi.vortex_indicator_neg()
    else:
        # Basic fallback calculation
        return pd.Series([1] * len(df), index=df.index), pd.Series([1] * len(df), index=df.index)

# Legacy calc_ma wrapper function for backward compatibility
def calc_ma(df, period: int = 20, ma_type: str = "SMA"):
    """Calculate moving average - wrapper for backward compatibility"""
    try:
        if ma_type == "SMA":
            return calc_sma(df['close'], period)
        elif ma_type == "EMA":
            return calc_ema(df['close'], period)
        elif ma_type == "WMA":
            return calc_wma(df['close'], period)
        elif ma_type == "TEMA":
            return calc_tema(df['close'], period)
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")
    except Exception as e:
        logger.error(f"Error calculating {ma_type}{period}: {e}")
        return pd.Series(index=df.index, dtype=float)

# Legacy RSI wrapper function for backward compatibility  
def calc_rsi(df, period: int = 14):
    """Calculate RSI - wrapper for backward compatibility"""
    try:
        return integrated_calc_rsi(df['close'], period)
    except Exception as e:
        logger.error(f"Error calculating RSI{period}: {e}")
        return pd.Series(index=df.index, dtype=float)

# Legacy ATR wrapper function for backward compatibility  
def calc_atr(df, period: int = 14):
    """Calculate ATR - wrapper for backward compatibility"""
    try:
        return integrated_calc_atr(df, period)
    except Exception as e:
        logger.error(f"Error calculating ATR{period}: {e}")
        return pd.Series(index=df.index, dtype=float)

# Legacy MACD wrapper function for backward compatibility  
def calc_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD - wrapper for backward compatibility"""
    try:
        result = integrated_calc_macd(df['close'], fast, slow, signal)
        return result['macd'], result['macd_signal'], result['macd_histogram']
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float)

# Legacy Stochastic wrapper function for backward compatibility  
def calc_stochastic(df, period=14, smooth_window=3):
    """Calculate Stochastic - wrapper for backward compatibility"""
    try:
        result = integrated_calc_stochastic(df, period, smooth_window, smooth_window)
        return result['stoch_k'], result['stoch_d']
    except Exception as e:
        logger.error(f"Error calculating Stochastic: {e}")
        return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float)

# Legacy Bollinger Bands wrapper function for backward compatibility
def calc_bollinger_bands(df, window=20, window_dev=2):
    """Calculate Bollinger Bands - wrapper for backward compatibility"""
    try:
        result = integrated_calc_bollinger_bands(df['close'], window, window_dev)
        return result['bb_upper'], result['bb_middle'], result['bb_lower']
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        middle = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        upper = middle + window_dev * std
        lower = middle - window_dev * std
        return upper, middle, lower

# =============================================================================
# FAST NUMBA-OPTIMIZED CALCULATION FUNCTIONS (Performance critical)
# =============================================================================

# Fast RSI implementation
@njit(parallel=True)
def fast_rsi(close_arr, period):
    n = len(close_arr)
    out = np.full(n, np.nan)
    for i in prange(period, n):
        gains = 0.0
        losses = 0.0
        for j in range(i-period+1, i+1):
            change = close_arr[j] - close_arr[j-1]
            if change >= 0:
                gains += change
            else:
                losses -= change
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out

# Fast SMA implementation  
@njit(parallel=True)
def fast_sma(arr, period):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in prange(period-1, n):
        out[i] = np.mean(arr[i-period+1:i+1])
    return out

# Fast Stochastic implementation
@njit(parallel=True)
def fast_stoch_k(close, high, low, period):
    n = len(close)
    out = np.full(n, np.nan)
    for i in prange(period-1, n):
        low_min = np.min(low[i-period+1:i+1])
        high_max = np.max(high[i-period+1:i+1])
        if high_max - low_min != 0:
            out[i] = 100 * (close[i] - low_min) / (high_max - low_min)
    return out
    return out

@njit(parallel=True)
def fast_donchian_high(high, window):
    n = len(high)
    out = np.full(n, np.nan)
    for i in prange(window-1, n):
        out[i] = np.max(high[i-window+1:i+1])
    return out

@njit(parallel=True)
def fast_donchian_low(low, window):
    n = len(low)
    out = np.full(n, np.nan)
    for i in prange(window-1, n):
        out[i] = np.min(low[i-window+1:i+1])
    return out

def calc_donchian_channel(df, window=20):
    high = df['high'].values
    low = df['low'].values
    upper = pd.Series(fast_donchian_high(high, window), index=df.index)
    lower = pd.Series(fast_donchian_low(low, window), index=df.index)
    middle = (upper + lower) / 2
    return upper, middle, lower

# --- Stochastic ---
@njit(parallel=True)
def fast_stoch_k(close, high, low, period):
    n = len(close)
    stoch_k = np.full(n, np.nan)
    for i in prange(period-1, n):
        low_min = np.min(low[i-period+1:i+1])
        high_max = np.max(high[i-period+1:i+1])
        if high_max - low_min != 0:
            stoch_k[i] = 100 * (close[i] - low_min) / (high_max - low_min)
    return stoch_k

def calc_stochastic(df, period=14, smooth_window=3):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    stoch_k = fast_stoch_k(close, high, low, period)
    stoch_k_pd = pd.Series(stoch_k, index=df.index)
    stoch_d = pd.Series(stoch_k_pd).rolling(window=smooth_window).mean()
    return stoch_k_pd, stoch_d

def add_indicator_signals(df):
    """
    Add trading signals for all indicators in the dataframe.
    
    This function has been replaced by the integrated signal functions
    but is kept for backward compatibility. Use add_comprehensive_signals() instead.
    """
    logger.warning("add_indicator_signals() is deprecated. Use add_comprehensive_signals() instead")
    
    try:
        # Use the new comprehensive signal generation
        add_comprehensive_signals(df)
        
        # Generate an overall signal if not already present
        if 'overall_signal' not in df.columns:
            df['overall_signal'] = generate_overall_signal(df)
            
    except Exception as e:
        logger.error(f"Error in add_indicator_signals: {e}")
        # Fallback to basic signal generation
        import numpy as np
        
        # Basic trend signals (simplified fallback)
        for col in df.columns:
            if any(col.startswith(prefix) for prefix in ["SMA", "EMA", "WMA", "TEMA"]) and "signal" not in col:
                if "close" in df.columns:
                    df[f"{col}_signal"] = np.where(df["close"] > df[col], "Bullish", "Bearish")
        
    return df
def validate_and_fix_indicators(df):
    """Validate and fix indicator values to ensure proper display on frontend"""
    logger.info(f"Validating indicators for {len(df)} records")
    
    # Get actual indicators from dataframe columns instead of hardcoded list
    actual_indicators = [col for col in df.columns if not col in ['time', 'open', 'high', 'low', 'close', 
                        'tick_volume', 'spread', 'real_volume', 'checksum', 'current', 'current_price', 'volume',
                        'rsi', 'rsi_signal', 'volume_signal', 'buy_signal_count', 'sell_signal_count', 'overall_signal']]

    # If a whitelist is active, only validate indicators whose base family is whitelisted
    if INDICATOR_WHITELIST is not None:
        def base_family(name: str) -> str:
            n = name.lower()
            if n.startswith(('sma','ema','wma','tema')): return 'ma'
            if n.startswith('rsi'): return 'rsi'
            if n.startswith('macd'): return 'macd'
            if n.startswith('stochk') or n.startswith('stochd'): return 'stochastic'
            if n.startswith('bb_'): return 'bollinger'
            if n.startswith('atr'): return 'atr'
            if n.startswith('adx'): return 'adx'
            if n.startswith('cci'): return 'cci'
            if n.startswith('williamsr'): return 'williamsr'
            if n.startswith('roc'): return 'roc'
            if n == 'obv': return 'obv'
            if n.startswith('mfi'): return 'mfi'
            if n.startswith('psar'): return 'psar'
            return n.split('_')[0]
        filtered = []
        wl = {normalize_indicator_key(x) for x in INDICATOR_WHITELIST}
        # Map certain families to canonical whitelist tokens
        family_alias = {
            'bollinger': 'bollinger',
            'stochastic': 'stochastic',  # user may choose stochrsi instead; keep separate
            'ma': 'ma',
        }
        for ind in actual_indicators:
            # Check direct mapping first (e.g., SMA20 -> sma20) 
            direct_key = normalize_indicator_key(ind)
            if direct_key in wl:
                filtered.append(ind)
                continue
            
            # Then check family mapping
            fam = base_family(ind)
            fam_key = normalize_indicator_key(family_alias.get(fam, fam))
            # Only keep if family explicitly selected in whitelist
            if fam_key in wl:
                filtered.append(ind)
        # If filtering removed all, fall back to original to avoid empty validation silently
        if filtered:
            actual_indicators = filtered
            logger.info(f"Whitelist active: validating only {len(actual_indicators)} indicators: {actual_indicators}")
        else:
            logger.info("Whitelist active but none of actual indicators selected; skipping validation stage")
            return df
    
    issues_found = 0
    
    for indicator in actual_indicators:
        if indicator in df.columns:
            # Count NaN values
            nan_count = df[indicator].isna().sum()
            total_count = len(df)
            valid_percentage = ((total_count - nan_count) / total_count) * 100
            
            logger.info(f"{indicator}: {valid_percentage:.1f}% valid data ({total_count - nan_count}/{total_count})")
            
            if valid_percentage < 50:  # Less than 50% valid data
                issues_found += 1
                logger.warning(f"ERROR {indicator} has too many NaN values ({valid_percentage:.1f}% valid)")
            
            # Try to forward fill some NaN values if there's valid data later
            if nan_count > 0 and nan_count < total_count:
                # Use bfill with limit (fillna(method='bfill') deprecated)
                df[indicator] = df[indicator].bfill(limit=10)  # Back fill limited
                
        else:
            logger.warning(f"WARN Expected indicator {indicator} not found in data")
            issues_found += 1
    
    logger.info(f"Validation complete. Issues found: {issues_found}")
    return df

def save_to_json(df, symbol: str, timeframe: str) -> bool:
    """Save indicator data to JSON file with improved error handling"""
    try:
        # Add comprehensive signals directly (avoid deprecated wrapper to reduce errors)
        try:
            if 'overall_signal' not in df.columns:
                df = add_comprehensive_signals(df)
                # Optionally generate overall signal if not created inside helpers
                if 'overall_signal' not in df.columns:
                    # generate_overall_signal returns df; ensure alignment
                    df = generate_overall_signal(df)
        except Exception as e:
            # Avoid legacy fallback causing column length mismatch noise
            logger.error(f"Failed comprehensive signal generation in save_to_json (no legacy fallback used): {e}")
        
        # Validate and fix indicators
        df = validate_and_fix_indicators(df)

        # --- Guard: drop any columns whose length != len(df) to prevent "Columns must be same length as key" ---
        bad_cols = []
        n_rows = len(df)
        for c in list(df.columns):
            try:
                if getattr(df[c], '__len__', None) and len(df[c]) != n_rows:
                    bad_cols.append(c)
            except Exception:
                continue
        if bad_cols:
            logger.warning(f"Dropping {len(bad_cols)} malformed columns with length mismatch: {bad_cols[:12]}{'...' if len(bad_cols)>12 else ''}")
            df.drop(columns=bad_cols, inplace=True, errors='ignore')
        
        # Check if current candle exists in original data
        current = False
        current_price = None
        
        try:
            candle_file_pattern = os.path.join(DATA_FOLDER, f"{symbol}*_{timeframe}.json")
            candle_files = glob.glob(candle_file_pattern)
            
            if candle_files:
                with open(candle_files[0], "r", encoding="utf-8") as f:
                    candles = json.load(f)
                    
                if candles and isinstance(candles[-1], dict) and candles[-1].get("current", False):
                    current = True
                    current_price = candles[-1].get("current_price")
                    
        except Exception as e:
            logger.warning(f"Could not check current candle status: {e}")
        
        # Clean DataFrame & prune by whitelist if configured
        df_clean = df.copy()
        if INDICATOR_WHITELIST is not None:
            try:
                selected_cols = collect_selected_indicator_columns(df_clean, INDICATOR_WHITELIST)
                # Ensure at least base cols remain
                if selected_cols:
                    df_clean = df_clean[selected_cols]
                logger.info(f"Pruned columns to whitelist set: kept={len(df_clean.columns)}")
            except Exception as e:
                logger.error(f"Failed pruning by whitelist: {e}")
        for col in ["current", "current_price"]:
            if col in df_clean.columns:
                df_clean = df_clean.drop(columns=[col])
        
        # Convert to records
        records = df_clean.to_dict(orient="records")
        
        # Handle NaN values and convert to None (null in JSON)
        for rec in records:
            for key, value in rec.items():
                if pd.isna(value) or (isinstance(value, float) and (value != value)):  # NaN check
                    rec[key] = None
                elif isinstance(value, (int, float)) and not pd.isna(value):
                    rec[key] = float(value)  # Ensure consistent number format
        
        # Convert timestamps to strings
        for rec in records:
            if "time" in rec and not isinstance(rec["time"], str):
                rec["time"] = str(rec["time"])
        
        # Mark current candle if applicable
        if current and records:
            records[-1]["current"] = True
            if current_price is not None:
                records[-1]["current_price"] = current_price
        
        # Save to file
        filename = os.path.join(INDICATOR_OUTPUT_DIR, f"{symbol}_{timeframe}_indicators.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Successfully saved indicators to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving indicators for {symbol} {timeframe}: {e}")
        return False

def calculate_and_save_all(user_id: str = None) -> Dict[str, int]:
    """
    Calculate indicators for all symbols and timeframes with improved error handling
    
    Args:
        user_id: Optional user identifier for tracking
        
    Returns:
        Dictionary with processing statistics
    """
    overall_start_time = datetime.now()
    stats = {"success": 0, "failed": 0, "total": 0}
    
    logger.info("Starting indicator calculation for all symbols and timeframes")
    
    # Log bulk processing start
    if user_id:
        log_indicator_request(user_id, "ALL", "ALL", ["BULK_PROCESSING"], {
            "total_symbols": len(symbols),
            "total_timeframes": len(timeframes),
            "total_combinations": len(symbols) * len(timeframes)
        })
    
    for symbol in symbols:
        for tf in timeframes:
            process_start_time = datetime.now()
            stats["total"] += 1
            
            try:
                logger.info(f"Processing {symbol} {tf}")
                
                count = num_candles[tf]
                df = fetch_candles_from_json(symbol, tf, count)
                
                if df is None:
                    logger.warning(f"Skipping {symbol} {tf} - no data available")
                    if user_id:
                        log_indicator_request(user_id, symbol, tf, ["SKIPPED"], {"reason": "no_data"})
                    stats["failed"] += 1
                    continue
                  # Calculate all indicators with error handling
                success = calculate_all_indicators(df, symbol, tf)
                
                if success:
                    # Save to JSON
                    if save_to_json(df, symbol, tf):
                        # Log successful processing
                        if user_id:
                            process_time = (datetime.now() - process_start_time).total_seconds()
                            log_indicator_result(user_id, symbol, tf, 20, 15, process_time)  # Approximate counts
                        
                        stats["success"] += 1
                        logger.info(f"Successfully processed {symbol} {tf}")
                    else:
                        stats["failed"] += 1
                else:
                    stats["failed"] += 1
                    
            except Exception as e:
                # Log processing error
                if user_id:
                    process_time = (datetime.now() - process_start_time).total_seconds()
                    log_indicator_request(user_id, symbol, tf, ["PROCESSING_ERROR"], {
                        "error": str(e),
                        "duration_seconds": round(process_time, 2)
                    })
                
                log_rate_limited_error(f"processing_error_{symbol}_{tf}", f"Error processing {symbol} {tf}: {e}")
                stats["failed"] += 1
    
    # Log overall completion
    total_time = (datetime.now() - overall_start_time).total_seconds()
    
    if user_id:
        log_indicator_request(user_id, "ALL", "ALL", ["BULK_COMPLETED"], {
            "success_count": stats['success'],
            "failed_count": stats['failed'],
            "total_count": stats['total'],
            "total_duration_seconds": round(total_time, 2),
            "average_time_per_symbol": round(total_time / max(stats['total'], 1), 2)
        })
    
    logger.info(f"Processing complete. Success: {stats['success']}, Failed: {stats['failed']}, Total: {stats['total']}")
    return stats

def calculate_all_indicators(df, symbol: str, timeframe: str) -> bool:
    """
    Calculate all indicators for a dataframe with comprehensive error handling.
    Now uses the integrated indicator calculation functions.
    """
    try:
        logger.info(f"Calculating indicators for {symbol} {timeframe}")
        
        # Define the standard indicator set to calculate
        standard_indicators = [
            {"name": "MA", "params": {"period": 20, "ma_type": "EMA"}},
            {"name": "MA", "params": {"period": 50, "ma_type": "EMA"}},
            {"name": "MA", "params": {"period": 100, "ma_type": "EMA"}},
            {"name": "MA", "params": {"period": 200, "ma_type": "EMA"}},
            {"name": "RSI", "params": {"period": 14}},
            {"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
            {"name": "Stochastic", "params": {"period": 14, "smooth": 3}},
            {"name": "Bollinger Bands", "params": {"window": 20, "dev": 2}},
            {"name": "ATR", "params": {"period": 14}},
            {"name": "ADX", "params": {"period": 14}},
            {"name": "CCI", "params": {"period": 20}},
            {"name": "WilliamsR", "params": {"period": 14}},
            {"name": "ROC", "params": {"period": 12}},
            {"name": "PSAR", "params": {}},
            {"name": "Donchian", "params": {"window": 20}},
            {"name": "DPO", "params": {"period": 20}},
            {"name": "TRIX", "params": {"period": 15}},
            {"name": "MassIndex", "params": {"fast": 9, "slow": 25}},
            {"name": "Vortex", "params": {"period": 14}},
            {"name": "KST", "params": {}},
            {"name": "StochRSI", "params": {"period": 14}},
            {"name": "UltimateOscillator", "params": {}},
            {"name": "Keltner", "params": {"window": 20}},
            {"name": "TEMA", "params": {"period": 20}},
            {"name": "TEMA", "params": {"period": 50}},
            {"name": "TEMA", "params": {"period": 100}},
            {"name": "TEMA", "params": {"period": 200}},
            {"name": "Envelope", "params": {"period": 20, "percent": 2}},
            {"name": "Ichimoku", "params": {"tenkan": 9, "kijun": 26, "senkou": 52}},
        ]
        
        # Add volume-based indicators only if volume data is available
        if 'volume' in df.columns and df['volume'].sum() > 0:
            volume_indicators = [
                {"name": "OBV", "params": {}},
                {"name": "MFI", "params": {"period": 14}},
                {"name": "Chaikin", "params": {"period": 20}},
                {"name": "EOM", "params": {"period": 14}},
                {"name": "ForceIndex", "params": {"period": 13}},
                {"name": "VWAP", "params": {}},
            ]
            standard_indicators.extend(volume_indicators)
            logger.info(f"Volume data available for {symbol} {timeframe}, including volume indicators")
        else:
            logger.info(f"No volume data for {symbol} {timeframe}, skipping volume indicators")
        
        # Calculate all indicators using the optimized function
        df = calculate_selected_indicators(df, standard_indicators)
        
        # Add comprehensive signals using integrated functions
        try:
            add_comprehensive_signals(df)
            logger.info(f"Generated comprehensive signals for {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {e}")
            # Removed legacy fallback to avoid misaligned column additions
        
        logger.info(f"Successfully calculated all indicators for {symbol} {timeframe}")
        return True
        
    except Exception as e:
        logger.error(f"Error calculating indicators for {symbol} {timeframe}: {e}")
def calc_fibonacci_levels(df, lookback: int = 100) -> Dict[str, float]:
    """
    Tính các mức Fibonacci dựa trên swing high và swing low trong lookback nến gần nhất.
    Trả về dict các mức giá Fibonacci.
    """
    if len(df) < lookback:
        lookback = len(df)
    recent = df[-lookback:]
    high = recent['high'].max()
    low = recent['low'].min()
    diff = high - low
    levels = {
        "fib_0.0": low,
        "fib_23.6": high - 0.236 * diff,
        "fib_38.2": high - 0.382 * diff,
        "fib_50.0": high - 0.5 * diff,
        "fib_61.8": high - 0.618 * diff,
        "fib_78.6": high - 0.786 * diff,
        "fib_100.0": high
    }
    # Gán giá trị này cho tất cả các dòng (broadcast)
    for key, value in levels.items():
        df[key] = value
    return levels

def export_indicators(symbol, timeframe, count, indicator_list, skip_cleanup=False):
    # 🧹 AUTO CLEANUP before processing indicators (unless skipped for aggregator preexport)
    if not skip_cleanup:
        logger.info("🧹 MT5 Indicator Exporter: Auto cleanup before processing...")
        try:
            cleanup_result = cleanup_indicator_data(max_age_hours=48, keep_latest=15)
            logger.info(f"✅ Cleaned {cleanup_result['total_files_deleted']} files, "
                       f"freed {cleanup_result['total_space_freed_mb']:.2f} MB")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")
    else:
        logger.info("🔄 Skipping cleanup for aggregator preexport - preserving existing files")
    
    df = fetch_candles_from_json(symbol, timeframe, count)
    if df is None:
        print(f"No candle data for {symbol} {timeframe}")
        return []
    # TÍNH CÁC INDICATOR THEO TÙY CHỌN NGƯỜI DÙNG
    df = calculate_selected_indicators(df, indicator_list)
    # Lưu file indicator với các cột đã tính
    save_to_json(df, symbol, timeframe)
    results = []
    for indi in indicator_list:
        name = indi["name"]
        params = indi.get("params", {})
        try:
            signal = "Neutral"
            detail = ""
            close = df["close"].iloc[-1]
            # RSI
            if name == "RSI":
                period = params.get("period", 14)
                rsi = df.get(f"RSI{period}", df.get("RSI", None))
                if rsi is not None:
                    # Check if last value is not null/nan
                    rsi_val = rsi.iloc[-1]
                    if rsi_val is not None and not pd.isna(rsi_val):
                        if rsi_val > 60:
                            signal = "Bullish"
                        elif rsi_val < 40:
                            signal = "Bearish"
                        detail = f"RSI={rsi_val:.2f}"
                    else:
                        detail = "RSI: No data (null value)"
                else:
                    detail = "RSI: Column not found"
            # MACD
            elif name == "MACD":
                fast = params.get("fast", 12)
                slow = params.get("slow", 26)
                signal_p = params.get("signal", 9)
                macd = df.get(f"MACD_{fast}_{slow}_{signal_p}", None)
                if macd is not None:
                    macd_val = macd.iloc[-1]
                    if macd_val > 0:
                        signal = "Bullish"
                    elif macd_val < 0:
                        signal = "Bearish"
                    detail = f"MACD={macd_val:.2f}"
            # Stochastic
            elif name == "Stochastic":
                period = params.get("period", 14)
                smooth = params.get("smooth", 3)
                k = df.get(f'StochK_{period}_{smooth}', None)
                d = df.get(f'StochD_{period}_{smooth}', None)
                if k is not None:
                    k_val = k.iloc[-1]
                    if k_val > 80:
                        signal = "Bullish"
                    elif k_val < 20:
                        signal = "Bearish"
                    d_val = d.iloc[-1] if d is not None else None
                    if d_val is not None:
                        detail = f"Stoch_K={k_val:.2f}, Stoch_D={d_val:.2f}"
                    else:
                        detail = f"Stoch_K={k_val:.2f}"
            # Bollinger Bands
            elif name == "Bollinger Bands":
                window = params.get("window", 20)
                dev = params.get("dev", 2)
                upper = df.get(f"BB_Upper_{window}_{dev}", None)
                lower = df.get(f"BB_Lower_{window}_{dev}", None)
                if upper is not None and lower is not None:
                    upper_val = upper.iloc[-1]
                    lower_val = lower.iloc[-1]
                    if close > upper_val:
                        signal = "Bullish"
                    elif close < lower_val:
                        signal = "Bearish"
                    detail = f"Close={close:.2f}, Upper={upper_val:.2f}, Lower={lower_val:.2f}"
            # ATR
            elif name == "ATR":
                period = params.get("period", 14)
                atr = df.get(f"ATR{period}", df.get("ATR", None))
                if atr is not None:
                    atr_val = atr.iloc[-1]
                    mean_atr = atr.mean()
                    if atr_val > mean_atr:
                        signal = "Bullish"
                    elif atr_val < mean_atr:
                        signal = "Bearish"
                    detail = f"ATR={atr_val:.2f}, MeanATR={mean_atr:.2f}"
            # ADX
            elif name == "ADX":
                period = params.get("period", 14)
                adx = df.get(f"ADX{period}", df.get("ADX", None))
                if adx is not None:
                    adx_val = adx.iloc[-1]
                    if adx_val > 25:
                        signal = "Bullish"
                    detail = f"ADX={adx_val:.2f}"
            # CCI
            elif name == "CCI":
                period = params.get("period", 20)
                cci = df.get(f"CCI{period}", None)
                if cci is not None:
                    cci_val = cci.iloc[-1]
                    if cci_val > 100:
                        signal = "Bullish"
                    elif cci_val < -100:
                        signal = "Bearish"
                    detail = f"CCI={cci_val:.2f}"
            # Williams %R
            elif name == "WilliamsR":
                period = params.get("period", 14)
                wr = df.get(f"WilliamsR{period}", df.get("WilliamsR", None))
                if wr is not None:
                    wr_val = wr.iloc[-1]
                    if wr_val > -20:
                        signal = "Bullish"
                    elif wr_val < -80:
                        signal = "Bearish"
                    detail = f"WilliamsR={wr_val:.2f}"
            # ROC
            elif name == "ROC":
                period = params.get("period", 12)
                roc = df.get(f"ROC{period}", df.get("ROC", None))
                if roc is not None:
                    roc_val = roc.iloc[-1]
                    if roc_val > 0:
                        signal = "Bullish"
                    elif roc_val < 0:
                        signal = "Bearish"
                    detail = f"ROC={roc_val:.2f}"
            # OBV
            elif name == "OBV":
                obv = df.get("OBV", None)
                if obv is not None:
                    obv_val = obv.iloc[-1]
                    prev_obv = obv.iloc[-2] if len(obv) > 1 else obv_val
                    if obv_val > prev_obv:
                        signal = "Bullish"
                    elif obv_val < prev_obv:
                        signal = "Bearish"
                    detail = f"OBV={obv_val:.2f}"
            # MFI
            elif name == "MFI":
                period = params.get("period", 14)
                mfi = df.get(f"MFI{period}", df.get("MFI", None))
                if mfi is not None:
                    mfi_val = mfi.iloc[-1]
                    if mfi_val > 80:
                        signal = "Bullish"
                    elif mfi_val < 20:
                        signal = "Bearish"
                    detail = f"MFI={mfi_val:.2f}"
            # PSAR
            elif name == "PSAR":
                psar = df.get("PSAR", None)
                if psar is not None:
                    psar_val = psar.iloc[-1]
                    if close > psar_val:
                        signal = "Bullish"
                    elif close < psar_val:
                        signal = "Bearish"
                    detail = f"Close={close:.2f}, PSAR={psar_val:.2f}"
            # Chaikin Money Flow
            elif name == "Chaikin":
                period = params.get("period", 20)
                chaikin = df.get(f"Chaikin{period}", None)
                if chaikin is not None:
                    chaikin_val = chaikin.iloc[-1]
                    if chaikin_val > 0:
                        signal = "Bullish"
                    elif chaikin_val < 0:
                        signal = "Bearish"
                    detail = f"Chaikin={chaikin_val:.2f}"
            # EOM
            elif name == "EOM":
                period = params.get("period", 14)
                eom = df.get(f"EOM{period}", df.get("EOM", None))
                if eom is not None:
                    eom_val = eom.iloc[-1]
                    if eom_val > 0:
                        signal = "Bullish"
                    elif eom_val < 0:
                        signal = "Bearish"
                    detail = f"EOM={eom_val:.2f}"
            # Force Index
            elif name == "ForceIndex":
                period = params.get("period", 13)
                fi = df.get(f"ForceIndex{period}", df.get("ForceIndex", None))
                if fi is not None:
                    fi_val = fi.iloc[-1]
                    if fi_val > 0:
                        signal = "Bullish"
                    elif fi_val < 0:
                        signal = "Bearish"
                    detail = f"ForceIndex={fi_val:.2f}"
            # Donchian Channel
            elif name == "Donchian":
                window = params.get("window", 20)
                dc_upper = df.get(f"Donchian_Upper_{window}", None)
                dc_lower = df.get(f"Donchian_Lower_{window}", None)
                if dc_upper is not None and dc_lower is not None:
                    if close >= dc_upper.iloc[-1]:
                        signal = "Bullish"
                    elif close <= dc_lower.iloc[-1]:
                        signal = "Bearish"
                    detail = f"Close={close:.2f}, Upper={dc_upper.iloc[-1]:.2f}, Lower={dc_lower.iloc[-1]:.2f}"
            # TEMA
            elif name == "TEMA":
                period = params.get("period", 20)
                tema = df.get(f"TEMA{period}", df.get("TEMA", None))
                if tema is not None:
                    tema_val = tema.iloc[-1]
                    if close > tema_val:
                        signal = "Bullish"
                    elif close < tema_val:
                        signal = "Bearish"
                    detail = f"Close={close:.2f}, TEMA={tema_val:.2f}"
            # TRIX
            elif name == "TRIX":
                period = params.get("period", 15)
                trix = df.get(f"TRIX{period}", df.get("TRIX", None))
                if trix is not None:
                    trix_val = trix.iloc[-1]
                    if trix_val > 0:
                        signal = "Bullish"
                    elif trix_val < 0:
                        signal = "Bearish"
                    detail = f"TRIX={trix_val:.2f}"
            # DPO
            elif name == "DPO":
                period = params.get("period", 20)
                dpo = df.get(f"DPO{period}", df.get("DPO", None))
                if dpo is not None:
                    dpo_val = dpo.iloc[-1]
                    if dpo_val is not None and not pd.isna(dpo_val):
                        if dpo_val > 0:
                            signal = "Bullish"
                        elif dpo_val < 0:
                            signal = "Bearish"
                        detail = f"DPO={dpo_val:.2f}"
                    else:
                        detail = "DPO: No data (null value)"
                else:
                    detail = "DPO: Column not found"
            # Mass Index
            elif name == "MassIndex":
                fast = params.get("fast", 9)
                slow = params.get("slow", 25)
                mi = df.get(f"MassIndex_{fast}_{slow}", df.get("MassIndex", None))
                if mi is not None:
                    mi_val = mi.iloc[-1]
                    if mi_val is not None and not pd.isna(mi_val):
                        if mi_val > 27:
                            signal = "Bullish"
                        detail = f"MassIndex={mi_val:.2f}"
                    else:
                        detail = "MassIndex: No data (null value)"
                else:
                    detail = "MassIndex: Column not found"
            # Vortex
            elif name == "Vortex":
                period = params.get("period", 14)
                vi_plus = df.get(f"VI+_{period}", None)
                vi_minus = df.get(f"VI-_{period}", None)
                if vi_plus is not None and vi_minus is not None:
                    vi_plus_val = vi_plus.iloc[-1]
                    vi_minus_val = vi_minus.iloc[-1]
                    if (vi_plus_val is not None and not pd.isna(vi_plus_val) and 
                        vi_minus_val is not None and not pd.isna(vi_minus_val)):
                        if vi_plus_val > vi_minus_val:
                            signal = "Bullish"
                        elif vi_plus_val < vi_minus_val:
                            signal = "Bearish"
                        detail = f"VI+={vi_plus_val:.2f}, VI-={vi_minus_val:.2f}"
                    else:
                        detail = "Vortex: No data (null values)"
                else:
                    detail = "Vortex: Column not found"
            # KST
            elif name == "KST":
                kst = df.get("KST", None)
                if kst is not None:
                    kst_val = kst.iloc[-1]
                    if kst_val > 0:
                        signal = "Bullish"
                    elif kst_val < 0:
                        signal = "Bearish"
                    detail = f"KST={kst_val:.2f}"
            # StochRSI
            elif name == "StochRSI":
                period = params.get("period", 14)
                stochrsi = df.get(f"StochRSI{period}", df.get("StochRSI", None))
                if stochrsi is not None:
                    stochrsi_val = stochrsi.iloc[-1]
                    if stochrsi_val is not None and not pd.isna(stochrsi_val):
                        if stochrsi_val > 0.8:
                            signal = "Bullish"
                        elif stochrsi_val < 0.2:
                            signal = "Bearish"
                        detail = f"StochRSI={stochrsi_val:.2f}"
                    else:
                        detail = "StochRSI: No data (null value)"
                else:
                    detail = "StochRSI: Column not found"
            # Ultimate Oscillator
            elif name == "UltimateOscillator":
                uo = df.get("UltimateOscillator", None)
                if uo is not None:
                    uo_val = uo.iloc[-1]
                    if uo_val > 70:
                        signal = "Bullish"
                    elif uo_val < 30:
                        signal = "Bearish"
                    detail = f"UltimateOscillator={uo_val:.2f}"
            # Keltner Channel
            elif name == "Keltner":
                window = params.get("window", 20)
                kc_upper = df.get(f"Keltner_Upper_{window}", None)
                kc_lower = df.get(f"Keltner_Lower_{window}", None)
                if kc_upper is not None and kc_lower is not None:
                    upper_val = kc_upper.iloc[-1]
                    lower_val = kc_lower.iloc[-1]
                    if upper_val is not None and lower_val is not None and not pd.isna(upper_val) and not pd.isna(lower_val):
                        if close > upper_val:
                            signal = "Bullish"
                        elif close < lower_val:
                            signal = "Bearish"
                        detail = f"Close={close:.2f}, Upper={upper_val:.2f}, Lower={lower_val:.2f}"
                    else:
                        detail = "Keltner: No data (null values)"
                else:
                    detail = "Keltner: Column not found"
            # MA (SMA/EMA)
            elif name == "MA":
                period = params.get("period", 20)
                ma_type = params.get("ma_type", "SMA")
                ma_col = f"{ma_type}{period}"
                ma = df.get(ma_col, None)
                if ma is not None:
                    ma_val = ma.iloc[-1]
                    if close > ma_val:
                        signal = "Bullish"
                    elif close < ma_val:
                        signal = "Bearish"
                    detail = f"Close={close:.2f}, {ma_type}={ma_val:.2f}"
            # WMA (Weighted Moving Average)
            elif name == "WMA":
                period = params.get("period", 20)
                wma_col = f"WMA{period}"
                wma = df.get(wma_col, None)
                if wma is not None:
                    wma_val = wma.iloc[-1]
                    if close > wma_val:
                        signal = "Bullish"
                    elif close < wma_val:
                        signal = "Bearish"
                    detail = f"Close={close:.2f}, WMA{period}={wma_val:.2f}"
                else:
                    detail = f"WMA{period}: Column not found"
            # TEMA (Triple Exponential Moving Average)
            elif name == "TEMA":
                period = params.get("period", 20)
                tema_col = f"TEMA{period}"
                tema = df.get(tema_col, None)
                if tema is not None:
                    tema_val = tema.iloc[-1]
                    if close > tema_val:
                        signal = "Bullish"
                    elif close < tema_val:
                        signal = "Bearish"
                    detail = f"Close={close:.2f}, TEMA{period}={tema_val:.2f}"
                else:
                    detail = f"TEMA{period}: Column not found"
            # Envelope
            elif name == "Envelope":
                period = params.get("period", 20)
                percent = params.get("percent", 2)
                upper = df.get(f"Envelope_Upper_{period}_{percent}", None)
                lower = df.get(f"Envelope_Lower_{period}_{percent}", None)
                if upper is not None and lower is not None:
                    upper_val = upper.iloc[-1]
                    lower_val = lower.iloc[-1]
                    if close > upper_val:
                        signal = "Bullish"
                    elif close < lower_val:
                        signal = "Bearish"
                    detail = f"Close={close:.2f}, Upper={upper_val:.2f}, Lower={lower_val:.2f}"
            # VWAP
            elif name == "VWAP":
                vwap = df.get("VWAP", None)
                if vwap is not None:
                    vwap_val = vwap.iloc[-1]
                    if close > vwap_val:
                        signal = "Bullish"
                    elif close < vwap_val:
                        signal = "Bearish"
                    detail = f"Close={close:.2f}, VWAP={vwap_val:.2f}"
            # Ichimoku
            elif name == "Ichimoku":
                tenkan = params.get("tenkan", 9)
                kijun = params.get("kijun", 26)
                senkou = params.get("senkou", 52)
                df["tenkan"], df["kijun"], df["senkou_a"], df["senkou_b"], df["chikou"] = calc_ichimoku(df, tenkan, kijun, senkou)
                tenkan_val = df["tenkan"].iloc[-1]
                kijun_val = df["kijun"].iloc[-1]
                close = df["close"].iloc[-1]
                if close > tenkan_val and close > kijun_val:
                    signal = "Bullish"
                elif close < tenkan_val and close < kijun_val:
                    signal = "Bearish"
                else:
                    signal = "Neutral"
                detail = f"Conv={tenkan_val:.2f}, Base={kijun_val:.2f}"
            # --- THÊM XỬ LÝ FIBONACCI ---
            elif name == "Fibonacci":
                lookback = params.get("lookback", 100)
                # Các mức fibo đã được gán vào df bởi calc_fibonacci_levels
                fib_levels = ["fib_0.0", "fib_23.6", "fib_38.2", "fib_50.0", "fib_61.8", "fib_78.6", "fib_100.0"]
                fib_vals = []
                for lvl in fib_levels:
                    if lvl in df.columns:
                        fib_vals.append(f"{lvl.split('_')[1]}={df[lvl].iloc[-1]:.2f}")
                detail = ", ".join(fib_vals)
                signal = "Neutral"
            # THÊM DEFAULT CASE CHO CÁC INDICATORS KHÔNG ĐƯỢC XỬ LÝ
            else:
                logger.warning(f"Indicator {name} not handled in export_indicators")
                detail = f"{name}: Not implemented"
                signal = "Neutral"
                
            # Get latest time from dataframe and convert to string
            latest_time = ""
            if "time" in df.columns and len(df) > 0:
                time_val = df["time"].iloc[-1]
                if hasattr(time_val, 'strftime'):
                    latest_time = time_val.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    latest_time = str(time_val)
                
            # KHÔNG append kết quả ở đây, chỉ append 1 lần ở cuối mỗi vòng lặp
            results.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "indicator": name,
                "signal": signal,
                "detail": detail,
                "time": latest_time
            })
        except Exception as e:
            # Get latest time for error case too
            latest_time = ""
            if "time" in df.columns and len(df) > 0:
                try:
                    time_val = df["time"].iloc[-1]
                    if hasattr(time_val, 'strftime'):
                        latest_time = time_val.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        latest_time = str(time_val)
                except:
                    latest_time = ""
            results.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "indicator": name,
                "signal": "Error",
                "detail": str(e),
                "time": latest_time
            })
    return results

def calculate_selected_indicators(df, indicator_list):
    """
    Calculate only the indicators specified in indicator_list for performance optimization.
    This function now properly filters indicators to avoid calculating unnecessary ones.
    
    Args:
        df: DataFrame with OHLCV data
        indicator_list: List of indicator dictionaries with "name" and optional "params"
        
    Returns:
        DataFrame with calculated indicators
    """
    if not indicator_list:
        logger.warning("No indicators specified in indicator_list")
        return df
    
    # Create a set of requested indicator names for fast lookup
    requested_indicators = {indi["name"] for indi in indicator_list}
    
    logger.info(f"Calculating {len(requested_indicators)} selected indicators: {list(requested_indicators)}")
    
    # Provide alias normalization so UI names map to internal calculation keys
    name_alias = {
        "Parabolic SAR": "PSAR",
        "ParabolicSAR": "PSAR",
        "SAR": "PSAR",
        "Ichimoku Cloud": "Ichimoku",
        "Bollinger": "Bollinger Bands",
    }

    for indi in indicator_list:
        raw_name = indi.get("name")
        name = name_alias.get(raw_name, raw_name)
        params = indi.get("params", {})
        
        logger.info(f"Processing indicator: {raw_name} -> {name} with params: {params}")
        
        try:
            # Use the integrated indicator functions
            if name == "MACD":
                fast = params.get("fast", 12)
                slow = params.get("slow", 26)
                signal = params.get("signal", 9)
                macd, macd_signal, macd_hist = calc_macd(df, fast, slow, signal)
                df[f"MACD_{fast}_{slow}_{signal}"] = macd
                df[f"MACD_signal_{fast}_{slow}_{signal}"] = macd_signal
                df[f"MACD_hist_{fast}_{slow}_{signal}"] = macd_hist
                
            elif name == "RSI":
                period = params.get("period", 14)
                try:
                    logger.info(f"Calculating RSI with period {period}")
                    rsi_values = calc_rsi(df, period)
                    df[f"RSI{period}"] = rsi_values
                    logger.info(f"RSI calculated successfully, last value: {rsi_values.iloc[-1] if len(rsi_values) > 0 else 'No data'}")
                except Exception as e:
                    logger.error(f"Error calculating RSI: {e}")
                    df[f"RSI{period}"] = None
                
            elif name == "MA":
                period = params.get("period", 20)
                ma_type = params.get("ma_type", "SMA")
                col_name = f"{ma_type}{period}"
                df[col_name] = calc_ma(df, period, ma_type)
                try:
                    last_val = df[col_name].iloc[-1]
                except Exception:
                    last_val = 'N/A'
                logger.info(f"Created MA column {col_name}, last={last_val}")
                
            elif name == "WMA":
                period = params.get("period", 20)
                df[f"WMA{period}"] = calc_wma(df['close'], period)
                col_name = f"WMA{period}"
                last_val = df[col_name].iloc[-1] if not df[col_name].isna().all() else "N/A"
                logger.info(f"Created WMA column {col_name}, last={last_val}")
                
            elif name == "TEMA":
                period = params.get("period", 20)
                # Use close price series for TEMA calculation
                df[f"TEMA{period}"] = calc_tema(df['close'], period)
                
            elif name == "Stochastic":
                period = params.get("period", 14)
                smooth = params.get("smooth", 3)
                stoch_k, stoch_d = calc_stochastic(df, period, smooth)
                df[f"StochK_{period}_{smooth}"] = stoch_k
                df[f"StochD_{period}_{smooth}"] = stoch_d
                
            elif name == "Bollinger Bands":
                window = params.get("window", 20)
                dev = params.get("dev", 2)
                upper, middle, lower = calc_bollinger_bands(df, window, dev)
                df[f"BB_Upper_{window}_{dev}"] = upper
                df[f"BB_Middle_{window}_{dev}"] = middle
                df[f"BB_Lower_{window}_{dev}"] = lower
                
            elif name == "ATR":
                period = params.get("period", 14)
                df[f"ATR{period}"] = calc_atr(df, period)
                
            elif name == "ADX":
                period = params.get("period", 14)
                df[f"ADX{period}"] = calc_adx(df, period)
                
            elif name == "CCI":
                period = params.get("period", 20)
                df[f"CCI{period}"] = calc_cci(df, period)
                
            elif name == "WilliamsR":
                period = params.get("period", 14)
                df[f"WilliamsR{period}"] = calc_williams_r(df, period)
                
            elif name == "ROC":
                period = params.get("period", 12)
                df[f"ROC{period}"] = calc_roc(df['close'], period)
                
            elif name == "OBV":
                df["OBV"] = calc_obv(df)
                
            elif name == "MFI":
                period = params.get("period", 14)
                df[f"MFI{period}"] = calc_mfi(df, period)
                
            elif name == "PSAR":
                df["PSAR"] = calc_psar(df)
                
            elif name == "Chaikin":
                period = params.get("period", 20)
                df[f"Chaikin{period}"] = calc_chaikin(df, period)
                
            elif name == "EOM":
                period = params.get("period", 14)
                df[f"EOM{period}"] = calc_eom(df, period)
                
            elif name == "ForceIndex":
                period = params.get("period", 13)
                df[f"ForceIndex{period}"] = calc_force_index(df, period)
                
            elif name == "Donchian":
                window = params.get("window", 20)
                try:
                    upper, middle, lower = calc_donchian_channel(df, window)
                    df[f"Donchian_Upper_{window}"] = upper
                    df[f"Donchian_Middle_{window}"] = middle
                    df[f"Donchian_Lower_{window}"] = lower
                except Exception as e:
                    logger.error(f"Error calculating Donchian channel: {e}")
                
            elif name == "DPO":
                period = params.get("period", 20)
                try:
                    logger.info(f"Calculating DPO with period {period}")
                    dpo_values = calc_dpo(df, period)
                    df[f"DPO{period}"] = dpo_values
                    # Check last value
                    last_val = dpo_values.iloc[-1] if len(dpo_values) > 0 else None
                    logger.info(f"DPO calculated, last value: {last_val}")
                except Exception as e:
                    logger.error(f"Error calculating DPO: {e}")
                    df[f"DPO{period}"] = None
                
            elif name == "TRIX":
                period = params.get("period", 15)
                df[f"TRIX{period}"] = calc_trix(df, period)
                
            elif name == "MassIndex":
                fast = params.get("fast", 9)
                slow = params.get("slow", 25)
                try:
                    logger.info(f"Calculating MassIndex with fast={fast}, slow={slow}")
                    mi_values = calc_mass_index(df, fast, slow)
                    df[f"MassIndex_{fast}_{slow}"] = mi_values
                    last_val = mi_values.iloc[-1] if len(mi_values) > 0 else None
                    logger.info(f"MassIndex calculated, last value: {last_val}")
                except Exception as e:
                    logger.error(f"Error calculating MassIndex: {e}")
                    df[f"MassIndex_{fast}_{slow}"] = None
                
            elif name == "Vortex":
                period = params.get("period", 14)
                try:
                    logger.info(f"Calculating Vortex with period {period}")
                    vi_pos, vi_neg = calc_vortex(df, period)
                    df[f"VI+_{period}"] = vi_pos
                    df[f"VI-_{period}"] = vi_neg
                    last_pos = vi_pos.iloc[-1] if len(vi_pos) > 0 else None
                    last_neg = vi_neg.iloc[-1] if len(vi_neg) > 0 else None
                    logger.info(f"Vortex calculated, last values: VI+={last_pos}, VI-={last_neg}")
                except Exception as e:
                    logger.error(f"Error calculating Vortex: {e}")
                    df[f"VI+_{period}"] = None
                    df[f"VI-_{period}"] = None
                
            elif name == "KST":
                kst, kst_sig = calc_kst(df)
                df["KST"] = kst
                df["KST_sig"] = kst_sig
                
            elif name == "StochRSI":
                period = params.get("period", 14)
                try:
                    logger.info(f"Calculating StochRSI with period {period}")
                    stochrsi, stochrsi_k, stochrsi_d = calc_stoch_rsi(df, period)
                    df[f"StochRSI{period}"] = stochrsi
                    df[f"StochRSI_K_{period}"] = stochrsi_k
                    df[f"StochRSI_D_{period}"] = stochrsi_d
                    logger.info(f"StochRSI calculated successfully")
                except Exception as e:
                    logger.error(f"Error calculating StochRSI: {e}")
                    df[f"StochRSI{period}"] = None
                
            elif name == "UltimateOscillator":
                try:
                    logger.info("Calculating UltimateOscillator")
                    uo_values = calc_ultimate_oscillator(df)
                    df["UltimateOscillator"] = uo_values
                    logger.info(f"UltimateOscillator calculated successfully")
                except Exception as e:
                    logger.error(f"Error calculating UltimateOscillator: {e}")
                    df["UltimateOscillator"] = None
                
            elif name == "Keltner":
                window = params.get("window", 20)
                try:
                    logger.info(f"Calculating Keltner Channel with window {window}")
                    kc_upper, kc_middle, kc_lower = calc_keltner_channel(df, window)
                    df[f"Keltner_Upper_{window}"] = kc_upper
                    df[f"Keltner_Middle_{window}"] = kc_middle
                    df[f"Keltner_Lower_{window}"] = kc_lower
                    logger.info(f"Keltner Channel calculated successfully")
                except Exception as e:
                    logger.error(f"Error calculating Keltner Channel: {e}")
                    df[f"Keltner_Upper_{window}"] = None
                    df[f"Keltner_Middle_{window}"] = None
                    df[f"Keltner_Lower_{window}"] = None
                
            elif name == "VWAP":
                df["VWAP"] = calc_vwap(df)
                
            elif name == "Ichimoku":
                tenkan = params.get("tenkan", 9)
                kijun = params.get("kijun", 26)
                senkou = params.get("senkou", 52)
                tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou = calc_ichimoku(df, tenkan, kijun, senkou)
                # Parameterized columns
                df[f"tenkan_{tenkan}"] = tenkan_sen
                df[f"kijun_{kijun}"] = kijun_sen
                df[f"senkou_a_{senkou}"] = senkou_a
                df[f"senkou_b_{senkou}"] = senkou_b
                df[f"chikou_{kijun}"] = chikou
                # Generic columns expected by signal layer
                df['ichimoku_tenkan'] = tenkan_sen
                df['ichimoku_kijun'] = kijun_sen
                df['ichimoku_senkou_a'] = senkou_a
                df['ichimoku_senkou_b'] = senkou_b
                df['ichimoku_chikou'] = chikou
                logger.info("Ichimoku components assigned (tenkan/kijun/senkou_a/senkou_b/chikou)")
                
            elif name == "Envelope":
                period = params.get("period", 20)
                percent = params.get("percent", 2)
                envelope_result = calc_envelope(df['close'], period, percent)
                df[f"Envelope_Upper_{period}_{percent}"] = envelope_result['envelope_upper']
                df[f"Envelope_Middle_{period}_{percent}"] = envelope_result['envelope_middle']
                df[f"Envelope_Lower_{period}_{percent}"] = envelope_result['envelope_lower']
                
            elif name == "Fibonacci":
                lookback = params.get("lookback", 100)
                try:
                    fib_levels = calc_fibonacci_levels(df, lookback)
                    # Map returned keys (with decimals) to simplified names used elsewhere
                    mapping = {
                        "fib_23.6": "fib_236",
                        "fib_38.2": "fib_382",
                        "fib_50.0": "fib_500",
                        "fib_61.8": "fib_618",
                        "fib_78.6": "fib_786"
                    }
                    for k, v in fib_levels.items():
                        target_key = mapping.get(k, k)
                        df[target_key] = v
                except Exception as e:
                    logger.error(f"Error calculating Fibonacci levels: {e}")
                
            else:
                logger.warning(f"Unknown indicator: {name}")
                
        except Exception as e:
            logger.error(f"Error calculating indicator {name}: {e}")
    
    # Extract requested MAs from indicator_list for ensure_required_mas
    requested_mas = []
    for indi in indicator_list:
        if indi.get("name") == "MA":
            params = indi.get("params", {})
            period = params.get("period", 20)
            ma_type = params.get("ma_type", "EMA")
            ma_col = f"{ma_type}{period}"
            requested_mas.append((ma_col, period, ma_type))
            
    try:
        ensure_required_mas(df, requested_mas)
    except Exception as e:
        logger.error(f"Error ensuring required MAs: {e}")

    logger.info(f"Successfully calculated {len(requested_indicators)} indicators")
    # Note: Removed hardcoded WMA/TEMA generation to respect UI indicator selection
    # Only calculate indicators explicitly requested by user interface
    
    return df

def update_data_with_new_candle(symbol, timeframe, count=100):
    """
    Update data with new candle and recalculate indicators.
    Now uses the modular calculate_all_indicators function to avoid code duplication.
    """
    logger.info(f"Updating data for {symbol} {timeframe}")
    
    df = fetch_candles_from_json(symbol, timeframe, count)
    if df is None:
        logger.warning(f"No candle data for {symbol} {timeframe}")
        return
        
    # Calculate all indicators using the unified function
    success = calculate_all_indicators(df, symbol, timeframe)
    
    if success:
        # Save the updated data
        if save_to_json(df, symbol, timeframe):
            logger.info(f"✅ Updated and saved indicators for {symbol} {timeframe}")
        else:
            logger.error(f"Failed to save indicators for {symbol} {timeframe}")
    else:
        logger.error(f"Failed to calculate indicators for {symbol} {timeframe}")

# Keep some specific calculation functions for backward compatibility and specific use cases

# Note: These legacy functions are kept for backward compatibility only
# All indicator calculations are now integrated in this file

# Legacy functions - these will be deprecated in future versions

def get_signal_for_indicator(row, indicator_name, params):
    if indicator_name == "MACD":
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal_p = params.get("signal", 9)
        field = f"MACD_signal_state_{fast}_{slow}_{signal_p}"
        return row.get(field, "N/A")
    if indicator_name == "RSI":
        period = params.get("period", 14)
        field = f"RSI{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "MA":
        period = params.get("period", 20)
        ma_type = params.get("ma_type", "SMA")
        field = f"{ma_type}{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "WMA":
        period = params.get("period", 20)
        field = f"WMA{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "Stochastic":
        period = params.get("period", 14)
        smooth = params.get("smooth", 3)
        field = f"StochK_{period}_{smooth}_signal"
        return row.get(field, "N/A")
    if indicator_name == "Bollinger Bands":
        window = params.get("window", 20)
        dev = params.get("dev", 2)
        field = f"BB_signal_{window}_{dev}"
        return row.get(field, "N/A")
    if indicator_name == "ATR":
        period = params.get("period", 14)
        field = f"ATR{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "ADX":
        period = params.get("period", 14)
        field = f"ADX{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "CCI":
        period = params.get("period", 20)
        field = f"CCI{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "WilliamsR":
        period = params.get("period", 14)
        field = f"WilliamsR{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "ROC":
        period = params.get("period", 12)
        field = f"ROC{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "OBV":
        field = "OBV_signal"
        return row.get(field, "N/A")
    if indicator_name == "MFI":
        period = params.get("period", 14)
        field = f"MFI{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "PSAR":
        field = "PSAR_signal"
        return row.get(field, "N/A")
    if indicator_name == "Chaikin":
        period = params.get("period", 20)
        field = f"Chaikin{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "EOM":
        period = params.get("period", 14)
        field = f"EOM{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "ForceIndex":
        period = params.get("period", 13)
        field = f"ForceIndex{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "Donchian":
        window = params.get("window", 20)
        field = f"Donchian_signal_{window}"
        return row.get(field, "N/A")
    if indicator_name == "TEMA":
        period = params.get("period", 20)
        field = f"TEMA{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "TRIX":
        period = params.get("period", 15)
        field = f"TRIX{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "DPO":
        period = params.get("period", 20)
        field = f"DPO{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "MassIndex":
        fast = params.get("fast", 9)
        slow = params.get("slow", 25)
        field = f"MassIndex_{fast}_{slow}_signal"
        return row.get(field, "N/A")
    if indicator_name == "Vortex":
        period = params.get("period", 14)
        field = f"Vortex_signal_{period}"
        return row.get(field, "N/A")
    if indicator_name == "KST":
        field = "KST_signal"
        return row.get(field, "N/A")
    if indicator_name == "StochRSI":
        period = params.get("period", 14)
        field = f"StochRSI{period}_signal"
        return row.get(field, "N/A")
    if indicator_name == "UltimateOscillator":
        field = "UltimateOscillator_signal"
        return row.get(field, "N/A")
    if indicator_name == "Keltner":
        window = params.get("window", 20)
        field = f"Keltner_signal_{window}"
        return row.get(field, "N/A")
    if indicator_name == "VWAP":
        field = "VWAP_signal"
        return row.get(field, "N/A")
    if indicator_name == "Ichimoku":
        tenkan = params.get("tenkan", 9)
        field = f"Ichimoku_signal_{tenkan}"
        return row.get(field, "N/A")
    if indicator_name == "Envelope":
        period = params.get("period", 20)
        percent = params.get("percent", 2)
        field = f"Envelope_signal_{period}_{percent}"
        return row.get(field, "N/A")
    return "N/A"

# All indicator calculation functions have been integrated into this file
# This removes dependency on external utility files

def calc_kst(df):
    kst = KSTIndicator(close=df['close'])
    return kst.kst(), kst.kst_sig()

def calc_stoch_rsi(df, period=14, smooth1=3, smooth2=3):
    stochrsi = StochRSIIndicator(close=df['close'], window=period, smooth1=smooth1, smooth2=smooth2)
    return stochrsi.stochrsi(), stochrsi.stochrsi_k(), stochrsi.stochrsi_d()

def calc_ultimate_oscillator(df, short=7, medium=14, long=28):
    return UltimateOscillator(high=df['high'], low=df['low'], close=df['close'], window1=short, window2=medium, window3=long).ultimate_oscillator()

def calc_keltner_channel(df, window=20):
    kc = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=window)
    return kc.keltner_channel_hband(), kc.keltner_channel_mband(), kc.keltner_channel_lband()

# --- Ichimoku ---
"""Removed duplicate calc_ichimoku & fast_ichimoku implementation to avoid conflicting return signatures."""

# ---------------------------------------------------------------------------
# Helper functions added for stabilization
# ---------------------------------------------------------------------------
def ensure_required_mas(df, requested_mas=None):
    """Ensure only requested moving averages exist (dynamic based on UI selection).
    If missing, compute directly from closing prices.
    """
    close = df.get('close')
    if close is None:
        return
    
    # Only create MAs that were specifically requested, not hardcoded defaults
    if requested_mas is None:
        return  # Don't create anything if no specific MAs requested
        
    ma_specs = requested_mas
    for col, period, ma_type in ma_specs:
        if col not in df.columns:
            try:
                series = calc_ma(df, period, ma_type)
                df[col] = series
                logger.info(f"[ensure_required_mas] Created missing {col}, last={series.iloc[-1] if len(series)>0 else 'N/A'}")
            except Exception as e:
                logger.error(f"Failed to create missing {col}: {e}")

def normalize_indicator_aliases(df):
    """Create lowercase/generic alias columns expected by legacy signal functions without duplicating heavy computations."""
    alias_map = [
        ("RSI14", "rsi"),
        ("StochK_14_3", "stoch_k"),
        ("StochD_14_3", "stoch_d"),
        ("WilliamsR14", "williams_r"),
        ("CCI20", "cci"),
        ("ROC12", "roc"),
        ("ATR14", "atr"),
        ("OBV", "obv"),
        ("MFI14", "mfi"),
        ("PSAR", "psar"),
    ]
    for src, alias in alias_map:
        if src in df.columns and alias not in df.columns:
            df[alias] = df[src]

    # Bollinger Bands (choose standard 20,2 if present)
    for prefix, generic in [("BB_Upper_20_2", "bb_upper"), ("BB_Middle_20_2", "bb_middle"), ("BB_Lower_20_2", "bb_lower")]:
        if prefix in df.columns and generic not in df.columns:
            df[generic] = df[prefix]

    # Donchian (20)
    for prefix, generic in [("Donchian_Upper_20", "donchian_upper"), ("Donchian_Middle_20", "donchian_middle"), ("Donchian_Lower_20", "donchian_lower")]:
        if prefix in df.columns and generic not in df.columns:
            df[generic] = df[prefix]

    # Envelopes
    for prefix, generic in [("Envelope_Upper_20_2", "envelope_upper"), ("Envelope_Middle_20_2", "envelope_middle"), ("Envelope_Lower_20_2", "envelope_lower")]:
        if prefix in df.columns and generic not in df.columns:
            df[generic] = df[prefix]

    return df

def generate_summary_report(stats: Dict[str, int]) -> None:
    """Generate and save a summary report of the indicator export process"""
    try:
        summary = {
            "generated_at": datetime.now().isoformat(),
            "processing_stats": stats,
            "symbols_processed": symbols,
            "timeframes_processed": timeframes,
            "configuration": {
                "data_folder": DATA_FOLDER,
                "output_folder": INDICATOR_OUTPUT_DIR,
                "num_candles_per_timeframe": num_candles
            }
        }
        
        # Calculate success rate
        if stats["total"] > 0:
            success_rate = (stats["success"] / stats["total"]) * 100
            summary["success_rate_percent"] = round(success_rate, 2)
        
        # Save summary
        summary_file = os.path.join(INDICATOR_OUTPUT_DIR, "indicator_export_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Summary report saved to {summary_file}")
        
        # Print summary to console
        print(f"\n{'='*60}")
        print("MT5 INDICATOR EXPORT SUMMARY")
        print(f"{'='*60}")
        print(f"Generated at: {summary['generated_at']}")
        print(f"Total processed: {stats['total']}")
        print(f"Successful: {stats['success']}")
        print(f"Failed: {stats['failed']}")
        if stats["total"] > 0:
            print(f"Success rate: {summary['success_rate_percent']}%")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")

def cleanup_indicator_data(max_age_hours: int = 48, keep_latest: int = 15) -> Dict[str, Any]:
    """
    🧹 SMART CLEANUP for MT5 Indicator Exporter: Only keep data for active symbols
    """
    import pickle
    
    # Get active symbols from user config
    active_symbols = set()
    try:
        with open('user_config.pkl', 'rb') as f:
            config = pickle.load(f)
            active_symbols = set(config.get('checked_symbols', []))
    except Exception:
        active_symbols = set()
    
    cleanup_stats = {
        'module_name': 'mt5_indicator_exporter_smart',
        'active_symbols': list(active_symbols),
        'total_files_deleted': 0,
        'total_space_freed_mb': 0.0,
        'cleanup_time': datetime.now().isoformat()
    }
    
    # Directory to clean
    if not os.path.exists(INDICATOR_OUTPUT_DIR):
        return cleanup_stats
    
    logger.info(f"🧹 Smart Indicator cleanup in {INDICATOR_OUTPUT_DIR}...")
    deleted_count = 0
    space_freed = 0.0
    
    for filename in os.listdir(INDICATOR_OUTPUT_DIR):
        if not filename.endswith('_indicators.json'):
            continue
            
        file_path = os.path.join(INDICATOR_OUTPUT_DIR, filename)
        
        try:
            # Extract symbol from filename (e.g., XAUUSD_m_M15_indicators.json)
            symbol_from_file = None
            for symbol in active_symbols:
                if symbol in filename:
                    symbol_from_file = symbol
                    break
            
            # If symbol not active or file too old, delete it
            should_delete = False
            if symbol_from_file is None:
                should_delete = True
            else:
                # Check file age
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < datetime.now() - timedelta(hours=max_age_hours):
                    should_delete = True
            
            if should_delete:
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                os.remove(file_path)
                deleted_count += 1
                space_freed += file_size
                    
        except Exception as e:
            logger.warning(f"Error processing {filename}: {e}")
    
    cleanup_stats['total_files_deleted'] = deleted_count
    cleanup_stats['total_space_freed_mb'] = space_freed
    
    logger.info(f"🧹 SMART INDICATOR cleanup complete: "
                f"{cleanup_stats['total_files_deleted']} files deleted, "
                f"{cleanup_stats['total_space_freed_mb']:.2f} MB freed")
    
    return cleanup_stats

def cleanup_old_indicator_files(max_age_hours: int = 48, keep_latest: int = 15) -> Dict[str, Any]:
    """
    🧹 Legacy function - calls the new cleanup_indicator_data function
    """
    return cleanup_indicator_data(max_age_hours, keep_latest)

def save_indicators_with_cleanup(df, symbol: str, timeframe: str, 
                                auto_cleanup: bool = True) -> bool:
    """
    💾 Lưu indicators với tự động dọn dẹp file cũ
    
    Args:
        df: DataFrame with indicator data
        symbol: Symbol name
        timeframe: Timeframe
        auto_cleanup: Tự động dọn dẹp file cũ không
    """
    try:
        # Create output filename
        output_filename = f"{symbol}_{timeframe}_indicators.json"
        output_path = os.path.join(INDICATOR_OUTPUT_DIR, output_filename)
        
        # Auto cleanup deprecated - now handled by module auto-cleanup
        
        # Convert DataFrame to dict for JSON serialization
        if PANDAS_AVAILABLE:
            indicators_data = df.to_dict('records')
        else:
            indicators_data = []
        
        # Add metadata
        output_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'indicators_count': len(indicators_data),
            'data': indicators_data
        }
        
        # Save file
        success = overwrite_json_safely(output_path, output_data, backup=False)
        
        if success:
            logger.info(f"💾 Indicators saved with cleanup: {output_filename}")
        else:
            logger.error(f"❌ Failed to save indicators: {output_filename}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error saving indicators for {symbol}_{timeframe}: {e}")
        return False

def get_indicator_storage_report() -> Dict[str, Any]:
    """📊 Tạo báo cáo sử dụng storage cho indicators"""
    logger.warning("Storage report is deprecated. Use system disk utilities instead.")
    return {'note': 'deprecated_function', 'use': 'system_disk_utilities'}

def main():
    """Main function to run the improved MT5 indicator exporter.
    """
    import sys
    try:
        logger.info("Starting MT5 Indicator Exporter")

        # Clean up old indicator files first
        print("🧹 Cleaning old indicator files...")
        cleanup_result = cleanup_indicator_output_files("indicator_output", max_age_hours=48)
        
        # Run the indicator calculation and export
        stats = calculate_and_save_all()
        
        # Generate summary report
        generate_summary_report(stats)
        
        logger.info("MT5 Indicator Exporter completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()