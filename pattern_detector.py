import os
import json
import logging
import argparse
import glob
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from utils import overwrite_json_safely, ensure_directory, auto_cleanup_on_start

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Safe imports
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
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    class MockPandas:
        class DataFrame:
            def __init__(self, *args, **kwargs): pass
            def __getitem__(self, key): return []
            def copy(self): return self
    pd = MockPandas()

try:
    # import talib as ta  # Optional: Technical Analysis library (not available)
    ta = None
    TA_AVAILABLE = False
except ImportError:
    ta = None
    TA_AVAILABLE = False

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Numba not available: {e}")
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def njit(func):
        return func

try:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    CONCURRENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Concurrent futures not available: {e}")
    CONCURRENT_AVAILABLE = False
    class ProcessPoolExecutor:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, func, *args): 
            class MockFuture:
                def result(self): return func(*args)
            return MockFuture()
    def as_completed(futures): return futures

# Get logger instance (already configured by setup_module_logging)
logger = logging.getLogger(__name__)

DATA_FOLDER = "data"
INDICATOR_FOLDER = "indicator_output"
OUTPUT_FOLDER = "pattern_signals"

# Định nghĩa biến này ở đầu file
INDICATORS_TO_USE = ["RSI", "MACD", "StochRSI", "UltimateOscillator", "EMA", "SMA", "MA", 
                     "BollingerBands", "ADX", "CCI", "ATR", "VWAP", "IchimokuCloud", "FibonacciRetracement", "PivotPoints"]

def flatten_ohlc_columns(df):
    """Flatten OHLC columns with error handling"""
    if not PANDAS_AVAILABLE:
        logger.warning("Pandas not available, skipping column flattening")
        return df
        
    try:
        for col in ['open', 'close', 'high', 'low']:
            if hasattr(df, 'columns') and col in df.columns:
                # Nếu đã là float hoặc int thì bỏ qua
                if hasattr(pd.api.types, 'is_numeric_dtype') and pd.api.types.is_numeric_dtype(df[col]):
                    continue
                    
                def safe_float(x):
                    if isinstance(x, dict):
                        x = list(x.values())[0]
                    elif hasattr(x, 'iloc'):  # pandas Series
                        x = x.iloc[0]
                    elif isinstance(x, (list, tuple)) and NUMPY_AVAILABLE:
                        if len(x) > 0:
                            x = np.ravel(x)[0] if hasattr(np, 'ravel') else x[0]
                        else:
                            return float('nan')
                    try:
                        return float(x)
                    except Exception:
                        return float('nan')
                        
                if hasattr(df[col], 'apply'):
                    df[col] = df[col].apply(safe_float).astype(float)
        return df
    except Exception as e:
        logger.error(f"Error flattening OHLC columns: {e}")
        return df

# ===== Enhanced Vectorized Candlestick Pattern Detection =====

def detect_doji(df):
    """Detect doji patterns with error handling"""
    try:
        if not PANDAS_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("Required libraries not available for doji detection")
            return df
            
        if not all(col in df.columns for col in ['open', 'close', 'high', 'low']):
            logger.warning("Required OHLC columns not found")
            return df
            
        open_ = df['open'].values if hasattr(df['open'], 'values') else df['open']
        close = df['close'].values if hasattr(df['close'], 'values') else df['close']
        high = df['high'].values if hasattr(df['high'], 'values') else df['high']
        low = df['low'].values if hasattr(df['low'], 'values') else df['low']

        body = np.abs(close - open_) if hasattr(np, 'abs') else [abs(c - o) for c, o in zip(close, open_)]
        candle_range = [h - l for h, l in zip(high, low)]
        
        # Calculate body ratio with division by zero protection
        body_ratio = []
        for b, r in zip(body, candle_range):
            if r == 0:
                body_ratio.append(0)
            else:
                body_ratio.append(b / r)

        doji = [ratio < 0.1 for ratio in body_ratio]
        
        # Ensure correct length
        if len(doji) > len(df):
            doji = doji[:len(df)]
        elif len(doji) < len(df):
            doji.extend([False] * (len(df) - len(doji)))
            
        df['doji'] = doji
        return df
        
    except Exception as e:
        logger.error(f"Error detecting doji patterns: {e}")
        if 'doji' not in df.columns:
            df['doji'] = [False] * len(df)
        return df

def detect_engulfing(df):
    """Detect engulfing patterns with error handling"""
    if not PANDAS_AVAILABLE:
        logger.warning("Pandas not available for engulfing detection")
        df.setdefault('bullish_engulfing', [False]*len(df)) if hasattr(df,'setdefault') else None
        df.setdefault('bearish_engulfing', [False]*len(df)) if hasattr(df,'setdefault') else None
        df.setdefault('engulfing', [False]*len(df)) if hasattr(df,'setdefault') else None
        return df

    if not all(col in df.columns for col in ['open', 'close']):
        logger.warning("Required columns not found for engulfing detection")
        for col in ['bullish_engulfing','bearish_engulfing','engulfing']:
            if col not in df.columns:
                df[col] = [False]*len(df)
        return df

    try:
        # Use vector operations with shift to avoid index errors
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        open_ = df['open']
        close = df['close']

        cond_prev_bear = prev_close < prev_open
        cond_prev_bull = prev_close > prev_open
        # Bullish engulfing
        bull = (
            cond_prev_bear &
            (close > open_) &
            (open_ < prev_close) &
            (close > prev_open) &
            ( (close - open_).abs() > (prev_close - prev_open).abs() * 1.1 )
        )
        # Bearish engulfing
        bear = (
            cond_prev_bull &
            (close < open_) &
            (open_ > prev_close) &
            (close < prev_open) &
            ( (close - open_).abs() > (prev_close - prev_open).abs() * 1.1 )
        )
        bull = bull.fillna(False)
        bear = bear.fillna(False)
        df['bullish_engulfing'] = bull.tolist()
        df['bearish_engulfing'] = bear.tolist()
        df['engulfing'] = (bull | bear).tolist()
    except Exception as e:
        logger.error(f"Error detecting engulfing patterns: {e}")
        for col in ['bullish_engulfing','bearish_engulfing','engulfing']:
            if col not in df.columns:
                df[col]=[False]*len(df)
    return df

def detect_pin_bar(df):
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    body = np.abs(close - open_)
    upper_shadow = high - np.maximum(close, open_)
    lower_shadow = np.minimum(close, open_) - low
    candle_range = high - low

    condition1 = candle_range > 0
    long_upper_shadow = (upper_shadow >= 2 * body)
    long_lower_shadow = (lower_shadow >= 2 * body)
    small_body = (body / candle_range) < 0.3

    pin_bar = condition1 & small_body & (long_upper_shadow | long_lower_shadow)
    pin_bar = np.asarray(pin_bar).flatten()[:len(df)]
    df['pin_bar'] = pin_bar
    return df

def detect_inside_bar(df):
    high = df['high'].values
    low = df['low'].values

    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)

    inside_bar = (high < prev_high) & (low > prev_low)
    inside_bar = np.asarray(inside_bar).flatten()[:len(df)]
    df['inside_bar'] = inside_bar
    return df

def detect_spinning_top(df):
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    body = np.abs(close - open_)
    candle_range = high - low
    upper_shadow = high - np.maximum(close, open_)
    lower_shadow = np.minimum(close, open_) - low

    condition1 = candle_range > 0
    small_body = (body / candle_range) < 0.3
    long_upper_shadow = upper_shadow >= 0.3 * candle_range
    long_lower_shadow = lower_shadow >= 0.3 * candle_range

    spinning_top = condition1 & small_body & long_upper_shadow & long_lower_shadow
    spinning_top = np.asarray(spinning_top).flatten()[:len(df)]
    df['spinning_top'] = spinning_top
    return df

def detect_hammer_patterns(df):
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    body = np.abs(close - open_)
    candle = high - low
    upper = high - np.maximum(open_, close)
    lower = np.minimum(open_, close) - low

    candle = np.where(candle == 0, 1e-8, candle)
    hammer = (body / candle < 0.3) & (lower / candle > 0.5) & (upper / candle < 0.2)
    inv_hammer = (body / candle < 0.3) & (upper / candle > 0.5) & (lower / candle < 0.2)
    hammer = np.asarray(hammer).flatten()[:len(df)]
    inv_hammer = np.asarray(inv_hammer).flatten()[:len(df)]
    df['hammer'] = hammer
    df['inverted_hammer'] = inv_hammer
    return df

def detect_hanging_shooting(df):
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    body = np.abs(close - open_)
    candle = high - low
    upper = high - np.maximum(open_, close)
    lower = np.minimum(open_, close) - low

    candle = np.where(candle == 0, 1e-8, candle)
    hanging_man = (body / candle < 0.3) & (lower / candle > 0.5) & (upper / candle < 0.2)
    shooting_star = (body / candle < 0.3) & (upper / candle > 0.5) & (lower / candle < 0.2)
    hanging_man = np.asarray(hanging_man).flatten()[:len(df)]
    shooting_star = np.asarray(shooting_star).flatten()[:len(df)]
    df['hanging_man'] = hanging_man
    df['shooting_star'] = shooting_star
    return df

@njit
def detect_morning_evening_star_nb(open_, close, high, low):
    n = len(close)
    ms = np.zeros(n, dtype=np.bool_)
    es = np.zeros(n, dtype=np.bool_)
    for i in range(2, n):
        # Morning Star
        c1 = close[i-2] < open_[i-2]
        c2 = abs(close[i-1] - open_[i-1]) < 0.3 * (high[i-1] - low[i-1])
        c3 = close[i] > open_[i]
        gap1 = open_[i-1] < close[i-2]
        gap2 = open_[i] > close[i-1]
        if c1 and c2 and c3 and gap1 and gap2:
            ms[i] = True
        # Evening Star
        c1e = close[i-2] > open_[i-2]
        c2e = abs(close[i-1] - open_[i-1]) < 0.3 * (high[i-1] - low[i-1])
        c3e = close[i] < open_[i]
        gap1e = open_[i-1] > close[i-2]
        gap2e = open_[i] < close[i-1]
        if c1e and c2e and c3e and gap1e and gap2e:
            es[i] = True
    return ms, es

def detect_morning_evening_star(df):
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    ms, es = detect_morning_evening_star_nb(open_, close, high, low)
    df['morning_star'] = ms
    df['evening_star'] = es
    return df

def detect_harami(df):
    open_ = df['open'].values
    close = df['close'].values
    prev_open = np.roll(open_, 1)
    prev_close = np.roll(close, 1)
    bullish = (prev_close < prev_open) & (close > open_) & (open_ > prev_close) & (close < prev_open)
    bearish = (prev_close > prev_open) & (close < open_) & (open_ < prev_close) & (close > prev_open)
    bullish = np.asarray(bullish).flatten()[:len(df)]
    bearish = np.asarray(bearish).flatten()[:len(df)]
    df['bullish_harami'] = bullish
    df['bearish_harami'] = bearish
    return df

def detect_harami_cross(df):
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    prev_open = np.roll(open_, 1)
    prev_close = np.roll(close, 1)
    body = np.abs(close - open_)
    candle = high - low
    doji = (body / candle < 0.1)
    inside = (open_ > np.minimum(prev_open, prev_close)) & (close < np.maximum(prev_open, prev_close))
    harami_cross = doji & inside
    harami_cross = np.asarray(harami_cross).flatten()[:len(df)]
    df['harami_cross'] = harami_cross
    return df

def detect_marubozu(df):
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    body = np.abs(close - open_)
    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low
    candle_range = high - low
    marubozu = (upper_shadow < 0.05 * candle_range) & (lower_shadow < 0.05 * candle_range) & (body > 0.8 * candle_range)
    marubozu = np.asarray(marubozu).flatten()[:len(df)]
    df['marubozu'] = marubozu
    return df

def detect_tweezer_top_bottom(df):
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    tweezer_top = (np.abs(high - np.roll(high, 1)) < 1e-4) & (close < open_) & (np.roll(close, 1) > np.roll(open_, 1))
    tweezer_bottom = (np.abs(low - np.roll(low, 1)) < 1e-4) & (close > open_) & (np.roll(close, 1) < np.roll(open_, 1))
    tweezer_top = np.asarray(tweezer_top).flatten()[:len(df)]
    tweezer_bottom = np.asarray(tweezer_bottom).flatten()[:len(df)]
    df['tweezer_top'] = tweezer_top
    df['tweezer_bottom'] = tweezer_bottom
    return df

def detect_piercing_darkcloud(df):
    open_ = df['open'].values
    close = df['close'].values
    prev_open = np.roll(open_, 1)
    prev_close = np.roll(close, 1)
    piercing = (prev_close < prev_open) & (open_ < prev_close) & (close > open_) & (close > (prev_open + prev_close)/2) & (close < prev_open)
    darkcloud = (prev_close > prev_open) & (open_ > prev_close) & (close < open_) & (close < (prev_open + prev_close)/2) & (close > prev_open)
    piercing = np.asarray(piercing).flatten()[:len(df)]
    darkcloud = np.asarray(darkcloud).flatten()[:len(df)]
    df['piercing_line'] = piercing
    df['dark_cloud_cover'] = darkcloud
    return df

def detect_abandoned_baby(df):
    if len(df) < 3:
        df['abandoned_baby_bull'] = [False] * len(df)
        df['abandoned_baby_bear'] = [False] * len(df)
        return df
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    body = np.abs(close - open_)
    candle = high - low
    doji = (body / candle < 0.1)
    doji = np.asarray(doji, dtype=bool).flatten()
    bullish = np.zeros(len(df), dtype=bool)
    bearish = np.zeros(len(df), dtype=bool)

    for i in range(2, len(df)):
        # Morning Star
        c1 = close[i-2] < open_[i-2]
        c2 = abs(close[i-1] - open_[i-1]) < 0.3 * (high[i-1] - low[i-1])
        c3 = close[i] > open_[i]
        gap1 = open_[i-1] < close[i-2]
        gap2 = open_[i] > close[i-1]
        if c1 and c2 and c3 and gap1 and gap2:
            bullish[i] = True
        # Evening Star
        c1e = close[i-2] > open_[i-2]
        c2e = abs(close[i-1] - open_[i-1]) < 0.3 * (high[i-1] - low[i-1])
        c3e = close[i] < open_[i]
        gap1e = open_[i-1] > close[i-2]
        gap2e = open_[i] < close[i-1]
        if c1e and c2e and c3e and gap1e and gap2e:
            bearish[i] = True
    bullish = np.asarray(bullish).flatten()[:len(df)]
    bearish = np.asarray(bearish).flatten()[:len(df)]
    df['abandoned_baby_bull'] = bullish
    df['abandoned_baby_bear'] = bearish
    return df

def detect_tristar(df):
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    body = np.abs(close - open_)
    candle = high - low
    doji = (body / candle < 0.1)
    bullish = np.zeros(len(df), dtype=bool)
    bearish = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        if doji[i-2] and doji[i-1] and doji[i]:
            if low[i-1] > high[i-2] and low[i] > high[i-1]:
                bullish[i] = True
            if high[i-1] < low[i-2] and high[i] < low[i-1]:
                bearish[i] = True
    bullish = np.asarray(bullish).flatten()[:len(df)]
    bearish = np.asarray(bearish).flatten()[:len(df)]
    df['tristar_bull'] = bullish
    df['tristar_bear'] = bearish
    return df

def detect_three_white_soldiers(df):
    close = df['close'].values
    open_ = df['open'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        if (close[i-2] < open_[i-2] and close[i-1] > open_[i-1] and close[i] > open_[i] and
            close[i-2] < close[i-1] < close[i]):
            pattern[i] = True
    df['three_white_soldiers'] = pattern
    return df

def detect_three_black_crows(df):
    close = df['close'].values
    open_ = df['open'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        if (close[i-2] > open_[i-2] and close[i-1] < open_[i-1] and close[i] < open_[i] and
            close[i-2] > close[i-1] > close[i]):
            pattern[i] = True
    df['three_black_crows'] = pattern
    return df

def detect_stick_sandwich(df):
    open_ = df['open'].values
    close = df['close'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        if (close[i-2] < open_[i-2] and close[i-1] > open_[i-1] and close[i] < open_[i] and
            abs(close[i-2] - close[i]) < 1e-4):
            pattern[i] = True
    df['stick_sandwich'] = pattern
    return df

def detect_kicker(df):
    open_ = df['open'].values
    close = df['close'].values
    bullish = np.zeros(len(df), dtype=bool)
    bearish = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if (close[i-1] < open_[i-1] and close[i] > open_[i] and open_[i] >= close[i-1] and
            abs(close[i] - open_[i]) > abs(close[i-1] - open_[i-1])):
            bullish[i] = True
        if (close[i-1] > open_[i-1] and close[i] < open_[i] and open_[i] <= close[i-1] and
            abs(close[i] - open_[i]) > abs(close[i-1] - open_[i-1])):
            bearish[i] = True
    df['kicker_bull'] = bullish
    df['kicker_bear'] = bearish
    return df

def detect_counterattack(df):
    open_ = df['open'].values
    close = df['close'].values
    counter_bull = np.zeros(len(df), dtype=bool)
    counter_bear = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if (close[i-1] < open_[i-1] and close[i] > open_[i] and abs(close[i] - close[i-1]) < 1e-4):
            counter_bull[i] = True
        if (close[i-1] > open_[i-1] and close[i] < open_[i] and abs(close[i] - close[i-1]) < 1e-4):
            counter_bear[i] = True
    return counter_bull, counter_bear

@njit
def detect_stick_sandwich_nb(open_, close_):
    n = len(close_)
    stick = np.zeros(n, dtype=np.bool_)
    for i in range(2, n):
        if (close_[i-2] < open_[i-2] and close_[i-1] > open_[i-1] and close_[i] < open_[i] and
            abs(close_[i-2] - close_[i]) < 1e-4):
            stick[i] = True
    return stick

@njit
def detect_tristar_nb(open_, close_, high, low):
    n = len(close_)
    body = np.abs(close_ - open_)
    candle = high - low
    doji = (body / candle < 0.1)
    bullish = np.zeros(n, dtype=np.bool_)
    bearish = np.zeros(n, dtype=np.bool_)
    for i in range(2, n):
        if doji[i-2] and doji[i-1] and doji[i]:
            if low[i-1] > high[i-2] and low[i] > high[i-1]:
                bullish[i] = True
            if high[i-1] < low[i-2] and high[i] < low[i-1]:
                bearish[i] = True
    return bullish, bearish

@njit
def detect_abandoned_baby_nb(open_, close_, high, low):
    n = len(close_)
    body = np.abs(close_ - open_)
    candle = high - low
    bullish = np.zeros(n, dtype=np.bool_)
    bearish = np.zeros(n, dtype=np.bool_)
    for i in range(2, n):
        # Morning Star
        c1 = close_[i-2] < open_[i-2]
        c2 = abs(close_[i-1] - open_[i-1]) < 0.3 * (high[i-1] - low[i-1])
        c3 = close_[i] > open_[i]
        gap1 = open_[i-1] < close_[i-2]
        gap2 = open_[i] > close_[i-1]
        if c1 and c2 and c3 and gap1 and gap2:
            bullish[i] = True
        # Evening Star
        c1e = close_[i-2] > open_[i-2]
        c2e = abs(close_[i-1] - open_[i-1]) < 0.3 * (high[i-1] - low[i-1])
        c3e = close_[i] < open_[i]
        gap1e = open_[i-1] > close_[i-2]
        gap2e = open_[i] < close_[i-1]
        if c1e and c2e and c3e and gap1e and gap2e:
            bearish[i] = True
    return bullish, bearish
def detect_marubozu_white_black(df):
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    body = np.abs(close - open_)
    candle_range = high - low
    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low
    marubozu_white = (close > open_) & (upper_shadow < 0.05 * candle_range) & (lower_shadow < 0.05 * candle_range) & (body > 0.8 * candle_range)
    marubozu_black = (close < open_) & (upper_shadow < 0.05 * candle_range) & (lower_shadow < 0.05 * candle_range) & (body > 0.8 * candle_range)
    df['marubozu_white'] = marubozu_white
    df['marubozu_black'] = marubozu_black
    return df

def detect_special_doji(df):
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    body = np.abs(close - open_)
    candle_range = high - low
    dragonfly = (body / candle_range < 0.1) & ((high - np.maximum(open_, close)) / candle_range < 0.1) & ((np.minimum(open_, close) - low) / candle_range > 0.6)
    gravestone = (body / candle_range < 0.1) & ((high - np.maximum(open_, close)) / candle_range > 0.6) & ((np.minimum(open_, close) - low) / candle_range < 0.1)
    long_legged = (body / candle_range < 0.1) & ((high - np.maximum(open_, close)) / candle_range > 0.3) & ((np.minimum(open_, close) - low) / candle_range > 0.3)
    df['dragonfly_doji'] = dragonfly
    df['gravestone_doji'] = gravestone
    df['long_legged_doji'] = long_legged
    return df

def detect_belt_hold(df):
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    candle_range = high - low
    belt_hold_bull = (open_ == low) & (close > open_) & ((high - close) / candle_range < 0.1)
    belt_hold_bear = (open_ == high) & (close < open_) & ((close - low) / candle_range < 0.1)
    df['belt_hold_bull'] = belt_hold_bull
    df['belt_hold_bear'] = belt_hold_bear
    return df
def detect_three_inside_down(df):
    open_ = df['open'].values
    close = df['close'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        if (close[i-2] > open_[i-2] and
            close[i-1] < open_[i-1] and
            open_[i-1] < close[i-2] and close[i-1] > open_[i-2] and
            close[i] < open_[i] and close[i] < open_[i-2]):
            pattern[i] = True
    df['three_inside_down'] = pattern
    return df

def detect_three_outside_up(df):
    open_ = df['open'].values
    close = df['close'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        if (close[i-2] < open_[i-2] and
            close[i-1] > open_[i-1] and
            open_[i-1] < close[i-2] and close[i-1] > open_[i-2] and
            close[i] > open_[i] and close[i] > close[i-1]):
            pattern[i] = True
    df['three_outside_up'] = pattern
    return df

def detect_three_outside_down(df):
    open_ = df['open'].values
    close = df['close'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        if (close[i-2] > open_[i-2] and
            close[i-1] < open_[i-1] and
            open_[i-1] > close[i-2] and close[i-1] < open_[i-2] and
            close[i] < open_[i] and close[i] < close[i-1]):
            pattern[i] = True
    df['three_outside_down'] = pattern
    return df

def detect_unique_three_river(df):
    open_ = df['open'].values
    close = df['close'].values
    low = df['low'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        if (close[i-2] < open_[i-2] and
            close[i-1] < open_[i-1] and open_[i-1] < open_[i-2] and close[i-1] > close[i-2] and low[i-1] < low[i-2] and
            close[i] > open_[i] and open_[i] > close[i-1] and close[i] > close[i-1]):
            pattern[i] = True
    df['unique_three_river'] = pattern
    return df

def detect_deliberation(df):
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        body1 = abs(close[i-2] - open_[i-2])
        body2 = abs(close[i-1] - open_[i-1])
        body3 = abs(close[i] - open_[i])
        upper_shadow3 = high[i] - max(open_[i], close[i])
        candle3_range = high[i] - low[i]
        if (close[i-2] > open_[i-2] and close[i-1] > open_[i-1] and close[i] > open_[i] and
            close[i-2] < close[i-1] < close[i] and
            body1 > 0.6 * (high[i-2] - low[i-2]) and
            body2 > 0.6 * (high[i-1] - low[i-1]) and
            body3 > 0.6 * (high[i] - low[i]) and
            upper_shadow3 > 0 and upper_shadow3 < 0.2 * candle3_range):
            pattern[i] = True
    df['deliberation'] = pattern
    return df
def detect_three_inside_up(df):
    open_ = df['open'].values
    close = df['close'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        # 1. Bearish candle
        # 2. Bullish candle inside previous body (Harami)
        # 3. Bullish close above first candle's open
        if (close[i-2] < open_[i-2] and
            close[i-1] > open_[i-1] and
            open_[i-1] > close[i-2] and close[i-1] < open_[i-2] and
            close[i] > open_[i] and close[i] > open_[i-2]):
            pattern[i] = True
    df['three_inside_up'] = pattern
    return df
def detect_concealing_baby_swallow(df):
    # Rất hiếm gặp, mô hình 4 nến giảm liên tiếp, 2 nến giữa là Marubozu, nến 3 che phủ hoàn toàn nến 2
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(3, len(df)):
        # 4 nến giảm liên tiếp
        if (close[i-3] < open_[i-3] and close[i-2] < open_[i-2] and close[i-1] < open_[i-1] and close[i] < open_[i]):
            # Nến 2 và 3 là marubozu đen
            body2 = abs(close[i-2] - open_[i-2])
            body3 = abs(close[i-1] - open_[i-1])
            range2 = high[i-2] - low[i-2]
            range3 = high[i-1] - low[i-1]
            marubozu2 = (body2 > 0.9 * range2)
            marubozu3 = (body3 > 0.9 * range3)
            # Nến 3 che phủ hoàn toàn nến 2
            covers = (high[i-1] >= high[i-2]) and (low[i-1] <= low[i-2])
            if marubozu2 and marubozu3 and covers:
                pattern[i] = True
    df['concealing_baby_swallow'] = pattern
    return df

def detect_advance_block(df):
    # 3 nến tăng liên tiếp, thân nến nhỏ dần, bóng trên dài dần
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        if (close[i-2] > open_[i-2] and close[i-1] > open_[i-1] and close[i] > open_[i]):
            body1 = abs(close[i-2] - open_[i-2])
            body2 = abs(close[i-1] - open_[i-1])
            body3 = abs(close[i] - open_[i])
            upper1 = high[i-2] - max(open_[i-2], close[i-2])
            upper2 = high[i-1] - max(open_[i-1], close[i-1])
            upper3 = high[i] - max(open_[i], close[i])
            if (body1 > body2 > body3) and (upper1 < upper2 < upper3):
                pattern[i] = True
    df['advance_block'] = pattern
    return df

def detect_breakaway(df):
    # 5 nến, 1 xu hướng mạnh, 3 nến tiếp theo tiếp diễn, nến 5 đảo chiều mạnh
    open_ = df['open'].values
    close = df['close'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(4, len(df)):
        # Breakaway giảm
        if (close[i-4] > open_[i-4] and
            close[i-3] < open_[i-3] and close[i-2] < open_[i-2] and close[i-1] < open_[i-1] and
            close[i] > open_[i] and close[i] > close[i-1]):
            pattern[i] = True
        # Breakaway tăng
        if (close[i-4] < open_[i-4] and
            close[i-3] > open_[i-3] and close[i-2] > open_[i-2] and close[i-1] > open_[i-1] and
            close[i] < open_[i] and close[i] < close[i-1]):
            pattern[i] = True
    df['breakaway'] = pattern
    return df

def detect_upside_gap_two_crows(df):
    # 3 nến: 1 tăng, 2 giảm, 2 nến giảm nằm trên gap so với nến tăng
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        if (close[i-2] > open_[i-2] and
            open_[i-1] > close[i-2] and close[i-1] < open_[i-1] and
            open_[i] > close[i-1] and close[i] < open_[i] and
            low[i-1] > high[i-2] and low[i] > high[i-2]):
            pattern[i] = True
    df['upside_gap_two_crows'] = pattern
    return df

def detect_downside_gap_three_methods(df):
    # 5 nến: 1 giảm mạnh, 3 tăng nhỏ, 1 giảm mạnh, 3 nến giữa nằm trong vùng nến 1
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(4, len(df)):
        if (close[i-4] < open_[i-4] and
            close[i-3] > open_[i-3] and close[i-2] > open_[i-2] and close[i-1] > open_[i-1] and
            close[i] < open_[i] and
            high[i-3] < high[i-4] and low[i-3] > low[i-4] and
            high[i-2] < high[i-4] and low[i-2] > low[i-4] and
            high[i-1] < high[i-4] and low[i-1] > low[i-4] and
            close[i] < low[i-1]):
            pattern[i] = True
    df['downside_gap_three_methods'] = pattern
    return df

def detect_upside_gap_three_methods(df):
    # 5 nến: 1 tăng mạnh, 3 giảm nhỏ, 1 tăng mạnh, 3 nến giữa nằm trong vùng nến 1
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(4, len(df)):
        if (close[i-4] > open_[i-4] and
            close[i-3] < open_[i-3] and close[i-2] < open_[i-2] and close[i-1] < open_[i-1] and
            close[i] > open_[i] and
            high[i-3] < high[i-4] and low[i-3] > low[i-4] and
            high[i-2] < high[i-4] and low[i-2] > low[i-4] and
            high[i-1] < high[i-4] and low[i-1] > low[i-4] and
            close[i] > high[i-1]):
            pattern[i] = True
    df['upside_gap_three_methods'] = pattern
    return df
def detect_patterns(df):
    df = detect_doji(df)
    df = detect_engulfing(df)
    df = detect_pin_bar(df)
    df = detect_inside_bar(df)
    df = detect_spinning_top(df)
    df = detect_hammer_patterns(df)
    df = detect_hanging_shooting(df)
    df = detect_morning_evening_star(df)
    df = detect_harami(df)
    df = detect_harami_cross(df)
    df = detect_marubozu(df)
    df = detect_tweezer_top_bottom(df)
    df = detect_piercing_darkcloud(df)
    df = detect_abandoned_baby(df)
    df = detect_tristar(df)
    df = detect_marubozu_white_black(df)
    df = detect_special_doji(df)
    df = detect_belt_hold(df)
    df = detect_three_inside_up(df)
    df = detect_three_inside_down(df)
    df = detect_three_outside_up(df)
    df = detect_three_outside_down(df)
    df = detect_unique_three_river(df)
    df = detect_deliberation(df)
    df = detect_concealing_baby_swallow(df)
    df = detect_advance_block(df)
    df = detect_breakaway(df)
    df = detect_upside_gap_two_crows(df)
    df = detect_downside_gap_three_methods(df)
    df = detect_upside_gap_three_methods(df)
    counter_bull, counter_bear = detect_counterattack(df)
    df['counterattack_bull'] = counter_bull
    df['counterattack_bear'] = counter_bear
    df = detect_three_white_soldiers(df)
    df = detect_three_black_crows(df)   
    df = detect_kicker(df)
    df = detect_doji_star(df)
    df = detect_morning_doji_star(df)
    df = detect_evening_doji_star(df)
    df = detect_mat_hold(df)
    df = detect_rising_three_methods(df)
    df = detect_falling_three_methods(df)
    df = detect_ladder_bottom(df)
    df = detect_ladder_top(df)
    df = detect_on_neck(df)
    df = detect_in_neck(df)
    df = detect_thrusting(df)
    df = detect_tasuki_gap(df)
    df = detect_separating_lines(df)
    df = detect_matching_low(df)
    df = detect_matching_high(df)
    df = detect_side_by_side_white_lines(df)
    return df
def detect_mat_hold(df):
    # Mat Hold: 5 nến, 1 tăng mạnh, 3 điều chỉnh nhỏ, 1 tăng mạnh vượt đỉnh nến 1
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(4, len(df)):
        if (close[i-4] > open_[i-4] and
            close[i-3] < open_[i-3] and close[i-2] < open_[i-2] and close[i-1] < open_[i-1] and
            low[i-3] > low[i-4] and low[i-2] > low[i-4] and low[i-1] > low[i-4] and
            close[i] > close[i-4] and close[i] > open_[i]):
            pattern[i] = True
    df['mat_hold'] = pattern
    return df

def detect_rising_three_methods(df):
    # Rising Three Methods: 1 tăng mạnh, 3 giảm nhỏ, 1 tăng mạnh vượt đỉnh nến 1
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(4, len(df)):
        if (close[i-4] > open_[i-4] and
            close[i-3] < open_[i-3] and close[i-2] < open_[i-2] and close[i-1] < open_[i-1] and
            high[i-3] < high[i-4] and low[i-3] > low[i-4] and
            high[i-2] < high[i-4] and low[i-2] > low[i-4] and
            high[i-1] < high[i-4] and low[i-1] > low[i-4] and
            close[i] > close[i-4] and close[i] > open_[i]):
            pattern[i] = True
    df['rising_three_methods'] = pattern
    return df

def detect_falling_three_methods(df):
    # Falling Three Methods: 1 giảm mạnh, 3 tăng nhỏ, 1 giảm mạnh vượt đáy nến 1
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(4, len(df)):
        if (close[i-4] < open_[i-4] and
            close[i-3] > open_[i-3] and close[i-2] > open_[i-2] and close[i-1] > open_[i-1] and
            high[i-3] < high[i-4] and low[i-3] > low[i-4] and
            high[i-2] < high[i-4] and low[i-2] > low[i-4] and
            high[i-1] < high[i-4] and low[i-1] > low[i-4] and
            close[i] < close[i-4] and close[i] < open_[i]):
            pattern[i] = True
    df['falling_three_methods'] = pattern
    return df

def detect_ladder_bottom(df):
    # Ladder Bottom: 5 nến giảm liên tiếp, 3 nến đầu thân dài, 2 nến sau thân nhỏ, nến cuối tăng mạnh
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(4, len(df)):
        body1 = abs(close[i-4] - open_[i-4])
        body2 = abs(close[i-3] - open_[i-3])
        body3 = abs(close[i-2] - open_[i-2])
        body4 = abs(close[i-1] - open_[i-1])
        body5 = abs(close[i] - open_[i])
        if (close[i-4] < open_[i-4] and close[i-3] < open_[i-3] and close[i-2] < open_[i-2] and
            body1 > body4 and body1 > body5 and body2 > body4 and body2 > body5 and body3 > body4 and body3 > body5 and
            close[i-1] > open_[i-1] and close[i] > open_[i] and close[i] > close[i-1]):
            pattern[i] = True
    df['ladder_bottom'] = pattern
    return df

def detect_ladder_top(df):
    # Ladder Top: 5 nến tăng liên tiếp, 3 nến đầu thân dài, 2 nến sau thân nhỏ, nến cuối giảm mạnh
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(4, len(df)):
        body1 = abs(close[i-4] - open_[i-4])
        body2 = abs(close[i-3] - open_[i-3])
        body3 = abs(close[i-2] - open_[i-2])
        body4 = abs(close[i-1] - open_[i-1])
        body5 = abs(close[i] - open_[i])
        if (close[i-4] > open_[i-4] and close[i-3] > open_[i-3] and close[i-2] > open_[i-2] and
            body1 > body4 and body1 > body5 and body2 > body4 and body2 > body5 and body3 > body4 and body3 > body5 and
            close[i-1] < open_[i-1] and close[i] < open_[i] and close[i] < close[i-1]):
            pattern[i] = True
    df['ladder_top'] = pattern
    return df

def detect_on_neck(df):
    # On Neck: 2 nến, nến 1 giảm mạnh, nến 2 tăng, giá đóng cửa nến 2 gần bằng giá thấp nến 1
    open_ = df['open'].values
    close = df['close'].values
    low = df['low'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if (close[i-1] < open_[i-1] and
            close[i] > open_[i] and
            abs(close[i] - low[i-1]) < 0.1 * (open_[i-1] - close[i-1])):
            pattern[i] = True
    df['on_neck'] = pattern
    return df

def detect_in_neck(df):
    # In Neck: 2 nến, nến 1 giảm mạnh, nến 2 tăng, giá đóng cửa nến 2 hơi trên giá thấp nến 1
    open_ = df['open'].values
    close = df['close'].values
    low = df['low'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if (close[i-1] < open_[i-1] and
            close[i] > open_[i] and
            0 < (close[i] - low[i-1]) < 0.2 * (open_[i-1] - close[i-1])):
            pattern[i] = True
    df['in_neck'] = pattern
    return df

def detect_thrusting(df):
    # Thrusting: 2 nến, nến 1 giảm mạnh, nến 2 tăng, giá đóng cửa nến 2 nằm trong thân nến 1 nhưng trên giữa thân
    open_ = df['open'].values
    close = df['close'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if (close[i-1] < open_[i-1] and
            close[i] > open_[i] and
            close[i] > close[i-1] and close[i] < open_[i-1] and
            close[i] > (open_[i-1] + close[i-1]) / 2):
            pattern[i] = True
    df['thrusting'] = pattern
    return df

def detect_tasuki_gap(df):
    # Tasuki Gap: 3 nến, gap giữa nến 1 và 2, nến 3 đảo chiều lấp 1 phần gap
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        # Tasuki Gap tăng
        if (close[i-2] > open_[i-2] and
            open_[i-1] > close[i-2] and close[i-1] > open_[i-1] and
            open_[i] < close[i-1] and close[i] < open_[i] and
            close[i] > open_[i-1]):
            pattern[i] = True
        # Tasuki Gap giảm
        if (close[i-2] < open_[i-2] and
            open_[i-1] < close[i-2] and close[i-1] < open_[i-1] and
            open_[i] > close[i-1] and close[i] > open_[i] and
            close[i] < open_[i-1]):
            pattern[i] = True
    df['tasuki_gap'] = pattern
    return df

def detect_separating_lines(df):
    # Separating Lines: 2 nến, nến 1 giảm, nến 2 tăng, giá mở cửa nến 2 bằng giá mở cửa nến 1
    open_ = df['open'].values
    close = df['close'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if (close[i-1] < open_[i-1] and
            close[i] > open_[i] and
            abs(open_[i] - open_[i-1]) < 1e-4):
            pattern[i] = True
        if (close[i-1] > open_[i-1] and
            close[i] < open_[i] and
            abs(open_[i] - open_[i-1]) < 1e-4):
            pattern[i] = True
    df['separating_lines'] = pattern
    return df

def detect_matching_low(df):
    # Matching Low: 2 nến giảm, giá đóng cửa 2 nến gần bằng nhau
    close = df['close'].values
    open_ = df['open'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if (close[i-1] < open_[i-1] and close[i] < open_[i] and abs(close[i] - close[i-1]) < 1e-4):
            pattern[i] = True
    df['matching_low'] = pattern
    return df

def detect_matching_high(df):
    # Matching High: 2 nến tăng, giá đóng cửa 2 nến gần bằng nhau
    close = df['close'].values
    open_ = df['open'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if (close[i-1] > open_[i-1] and close[i] > open_[i] and abs(close[i] - close[i-1]) < 1e-4):
            pattern[i] = True
    df['matching_high'] = pattern
    return df

def detect_side_by_side_white_lines(df):
    # Side-by-Side White Lines: 3 nến tăng, 2 nến sau mở cửa bằng nhau, đóng cửa tăng dần
    open_ = df['open'].values
    close = df['close'].values
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        if (close[i-2] > open_[i-2] and close[i-1] > open_[i-1] and close[i] > open_[i] and
            abs(open_[i-1] - open_[i]) < 1e-4 and close[i-1] < close[i]):
            pattern[i] = True
    df['side_by_side_white_lines'] = pattern
    return df
def detect_doji_star(df):
    # Doji Star: Nến đầu là tăng/giảm mạnh, nến sau là doji, có gap rõ rệt
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    body = np.abs(close - open_)
    candle = high - low
    doji = (body / candle < 0.1)
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        # Nến đầu tăng hoặc giảm mạnh
        strong_bull = close[i-1] > open_[i-1] and (close[i-1] - open_[i-1]) > 0.6 * (high[i-1] - low[i-1])
        strong_bear = close[i-1] < open_[i-1] and (open_[i-1] - close[i-1]) > 0.6 * (high[i-1] - low[i-1])
        # Nến sau là doji và có gap
        gap_up = open_[i] > close[i-1] and doji[i]
        gap_down = open_[i] < close[i-1] and doji[i]
        if (strong_bull and gap_up) or (strong_bear and gap_down):
            pattern[i] = True
    df['doji_star'] = pattern
    return df

def detect_morning_doji_star(df):
    # Morning Doji Star: Nến 1 giảm mạnh, nến 2 doji có gap xuống, nến 3 tăng mạnh đóng trên giữa nến 1
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    body = np.abs(close - open_)
    candle = high - low
    doji = (body / candle < 0.1)
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        # Nến 1 giảm mạnh
        strong_bear = close[i-2] < open_[i-2] and (open_[i-2] - close[i-2]) > 0.6 * (high[i-2] - low[i-2])
        # Nến 2 doji, gap xuống
        gap_down = open_[i-1] < close[i-2] and doji[i-1]
        # Nến 3 tăng mạnh, đóng trên giữa nến 1
        strong_bull = close[i] > open_[i] and (close[i] - open_[i]) > 0.6 * (high[i] - low[i])
        close_above_mid = close[i] > (open_[i-2] + close[i-2]) / 2
        if strong_bear and gap_down and strong_bull and close_above_mid:
            pattern[i] = True
    df['morning_doji_star'] = pattern
    return df

def detect_evening_doji_star(df):
    # Evening Doji Star: Nến 1 tăng mạnh, nến 2 doji có gap lên, nến 3 giảm mạnh đóng dưới giữa nến 1
    open_ = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    body = np.abs(close - open_)
    candle = high - low
    doji = (body / candle < 0.1)
    pattern = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        # Nến 1 tăng mạnh
        strong_bull = close[i-2] > open_[i-2] and (close[i-2] - open_[i-2]) > 0.6 * (high[i-2] - low[i-2])
        # Nến 2 doji, gap lên
        gap_up = open_[i-1] > close[i-2] and doji[i-1]
        # Nến 3 giảm mạnh, đóng dưới giữa nến 1
        strong_bear = close[i] < open_[i] and (open_[i] - close[i]) > 0.6 * (high[i] - low[i])
        close_below_mid = close[i] < (open_[i-2] + close[i-2]) / 2
        if strong_bull and gap_up and strong_bear and close_below_mid:
            pattern[i] = True
    df['evening_doji_star'] = pattern
    return df

def combine_with_indicator(ind_row, indicator_list=None, symbol=None, timeframe=None, trendline_folder="trendline_sr"):
    all_indicators = []
    indicator_list = indicator_list or []

    for indi in indicator_list:
        name = indi["name"]
        params = indi.get("params", {})
        key = None
        signal_key = None

        # Xác định key và signal_key
        if name == "MA":
            ma_type = params.get("ma_type", "SMA")
            period = params.get("period", 20)
            key = f"{ma_type.upper()}{period}"
            signal_key = f"{ma_type.upper()}{period}_signal"
        elif name in ["SMA", "EMA", "TEMA", "WMA"]:
            period = params.get("period", 20)
            key = f"{name.upper()}{period}"
            signal_key = f"{name.upper()}{period}_signal"
        elif name == "MACD":
            fast = params.get("fast", 12)
            slow = params.get("slow", 26)
            signal = params.get("signal", 9)
            key = f"MACD_{fast}_{slow}_{signal}"
            signal_key = f"MACD_{fast}_{slow}_{signal}_signal"
        elif name == "RSI":
            period = params.get("period", 14)
            key = f"RSI{period}"
            signal_key = f"RSI{period}_signal"
        elif name == "Stochastic":
            period = params.get("period", 14)
            smooth = params.get("smooth", 3)
            key = f"StochK_{period}_{smooth}"
            signal_key = f"StochK_{period}_{smooth}_signal"
        elif name == "Bollinger Bands":
            window = params.get("window", 20)
            dev = params.get("dev", 2)
            key = f"BB_Upper_{window}_{dev}"
            signal_key = f"BB_Upper_{window}_{dev}_signal"
        elif name == "ATR":
            period = params.get("period", 14)
            key = f"ATR{period}"
            signal_key = f"ATR{period}_signal"
        elif name == "ADX":
            period = params.get("period", 14)
            key = f"ADX{period}"
            signal_key = f"ADX{period}_signal"
        elif name == "CCI":
            period = params.get("period", 20)
            key = f"CCI{period}"
            signal_key = f"CCI{period}_signal"
        elif name == "WilliamsR":
            period = params.get("period", 14)
            key = f"WilliamsR{period}"
            signal_key = f"WilliamsR{period}_signal"
        elif name == "ROC":
            period = params.get("period", 20)
            key = f"ROC{period}"
            signal_key = f"ROC{period}_signal"
        elif name == "OBV":
            key = "OBV"
            signal_key = "OBV_signal"
        elif name == "MFI":
            period = params.get("period", 14)
            key = f"MFI{period}"
            signal_key = f"MFI{period}_signal"
        elif name == "PSAR":
            key = "PSAR"
            signal_key = "PSAR_signal"
        elif name == "Chaikin":
            period = params.get("period", 20)
            key = f"Chaikin{period}"
            signal_key = f"Chaikin{period}_signal"
        elif name == "EOM":
            period = params.get("period", 14)
            key = f"EOM{period}"
            signal_key = f"EOM{period}_signal"
        elif name == "ForceIndex":
            period = params.get("period", 13)
            key = f"ForceIndex{period}"
            signal_key = f"ForceIndex{period}_signal"
        elif name == "Donchian":
            window = params.get("window", 20)
            key = f"Donchian_Upper_{window}"
            signal_key = f"Donchian_Upper_{window}_signal"
        elif name == "TRIX":
            period = params.get("period", 15)
            key = f"TRIX{period}"
            signal_key = f"TRIX{period}_signal"
        elif name == "DPO":
            period = params.get("period", 20)
            key = f"DPO{period}"
            signal_key = f"DPO{period}_signal"
        elif name == "MassIndex":
            fast = params.get("fast", 9)
            slow = params.get("slow", 25)
            key = f"MassIndex_{fast}_{slow}"
            signal_key = f"MassIndex_{fast}_{slow}_signal"
        elif name == "Vortex":
            period = params.get("period", 14)
            key = f"VI+_{period}"
            signal_key = f"VI+_{period}_signal"
        elif name == "KST":
            key = "KST"
            signal_key = "KST_signal"
        elif name == "StochRSI":
            period = params.get("period", 14)
            key = f"StochRSI{period}"
            signal_key = f"StochRSI{period}_signal"
        elif name == "UltimateOscillator":
            key = "UltimateOscillator"
            signal_key = "UltimateOscillator_signal"
        elif name == "Keltner":
            window = params.get("window", 20)
            key = f"Keltner_Upper_{window}"
            signal_key = f"Keltner_Upper_{window}_signal"
        elif name == "Fibonacci":
            key = "fib_0.0"
            signal_key = "fib_0.0_signal"
        elif name == "Envelope":
            period = params.get("period", 20)
            percent = params.get("percent", 2)
            key = f"Envelope_Upper_{period}_{percent}"
            signal_key = f"Envelope_Upper_{period}_{percent}_signal"
        else:
            key = name
            signal_key = f"{name}_signal"

        value = ind_row.get(key)
        try:
            value = float(value)
        except Exception:
            value = None

        signal = ind_row.get(signal_key)
        if signal is not None and not isinstance(signal, str):
            signal = str(signal)

        all_indicators.append({
            "name": key,
            "value": value,
            "signal": signal
        })

    return None, None, None, None, all_indicators, None

def confirm_pattern_by_indicators_and_trendline(pattern_row, indicator_row, trend_data, candle_row, threshold=2, threshold_pct=0.005):
    """
    Ưu tiên xác nhận tín hiệu Bull/Bear theo Trend (support/resistance/trendline/channel).
    Nếu trend xác nhận thì trả về tín hiệu mạnh, indicator chỉ là phụ.
    Nếu trend không xác nhận thì chỉ trả về Neutral hoặc tín hiệu yếu.
    """
    count_bullish = 0
    count_bearish = 0
    confirmed = []

    indicators = pattern_row.get("confirmed_indicators", [])
    for indi in indicators:
        name = indi.get("name")
        value = indi.get("value")
        signal = indicator_value_to_signal(name, value)
        if signal == "bullish":
            count_bullish += 1
        elif signal == "bearish":
            count_bearish += 1
        confirmed.append({
            "name": name,
            "value": value,
            "signal": indi.get("signal")
        })

    close = candle_row.get("close")
    trend_bull = False
    trend_bear = False
    if close is not None and trend_data is not None:
        for s in trend_data.get("support", []):
            if abs(close - s) / close < threshold_pct:
                trend_bull = True
                break
        for lower in trend_data.get("channel_lower", []):
            if abs(close - lower) / close < threshold_pct:
                trend_bull = True
                break
        for r in trend_data.get("resistance", []):
            if abs(close - r) / close < threshold_pct:
                trend_bear = True
                break
        for upper in trend_data.get("channel_upper", []):
            if abs(close - upper) / close < threshold_pct:
                trend_bear = True
                break
        for tl in trend_data.get("trendline", []):
            if abs(close - tl) / close < threshold_pct:
                if close < tl:
                    trend_bull = True
                else:
                    trend_bear = True
                break

    # Ưu tiên trend
    if trend_bull:
        overall_signal = "Bullish"
    elif trend_bear:
        overall_signal = "Bearish"
    elif count_bullish >= threshold:
        overall_signal = "Bullish_weak"
    elif count_bearish >= threshold:
        overall_signal = "Bearish_weak"
    else:
        overall_signal = "Neutral"

    return overall_signal, confirmed

def indicator_value_to_signal(name, value, close=None):
    """
    Xác nhận tín hiệu chủ đạo: indicator xu hướng (MA, MACD, ADX, KST, TRIX, Vortex, PSAR, Trendline, Channel, ...).
    Bổ trợ: indicator động lượng (RSI, Stoch, StochRSI, CCI, WilliamsR, MFI, ROC, UltimateOscillator, Chaikin, ForceIndex, EOM, ...).
    Phụ trợ: các indicator khác.
    """
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None

    lname = name.lower()

    # === Chủ đạo: Indicator xu hướng ===
    if any(ma in lname for ma in ["ma", "sma", "ema", "wma", "tema"]):
        if close is not None:
            if close > value:
                return "bullish"
            elif close < value:
                return "bearish"
        return None
    elif "macd" in lname and "signal" not in lname:
        if value > 0:
            return "bullish"
        elif value < 0:
            return "bearish"

    elif "adx" in lname:
        if value > 25:
            return "bullish"
    elif "kst" in lname:
        if value > 0:
            return "bullish"
        elif value < 0:
            return "bearish"
    elif "trix" in lname:
        if value > 0:
                      
                      
            return "bullish"
        elif value < 0:
            return "bearish"
    elif "vortex" in lname:
        if value > 1:
            return "bullish"
        elif value < 1:
            return "bearish"
    elif "psar" in lname:
        if close is not None:
            if close > value:
                return "bullish"
            elif close < value:
                return "bearish"
        return None
    elif "trendline" in lname or "channel" in lname:
        # Nếu có close, xác nhận theo vị trí giá so với trendline/channel
        if close is not None:
            if close > value:
                return "bullish"
            elif close < value:
                return "bearish"
        return None

    # === Bổ trợ: Indicator động lượng ===
    elif "rsi" in lname and "stoch" not in lname:
        if value > 70:
            return "bearish"
        elif value < 30:
            return "bullish"
    elif "stoch" in lname and "rsi" not in lname:
        if value > 80:
            return "bearish"
        elif value < 20:
            return "bullish"
    elif "stochrsi" in lname:
        if value > 0.8:
            return "bearish"
        elif value < 0.2:
            return "bullish"
    elif "cci" in lname:
        if value > 100:
            return "bullish"
        elif value < -100:
            return "bearish"
    elif "williamsr" in lname or "williams" in lname:
        if value > -20:
            return "bearish"
        elif value < -80:
            return "bullish"
    elif "mfi" in lname:
        if value > 80:
            return "bearish"
        elif value < 20:
            return "bullish"
    elif "roc" in lname:
        if value > 0:
            return "bullish"
        elif value < 0:
            return "bearish"
    elif "ultimateoscillator" in lname or "ultimate oscillator" in lname:
        if value > 70:
            return "bearish"
        elif value < 30:
            return "bullish"
    elif "chaikin" in lname:
        if value > 0:
            return "bullish"
        elif value < 0:
            return "bearish"
    elif "forceindex" in lname or "force index" in lname:
        if value > 0:
            return "bullish"
        elif value < 0:
            return "bearish"
    elif "eom" in lname:
        if value > 0:
            return "bullish"
        elif value < 0:
            return "bearish"

    # === Phụ trợ: các indicator khác ===
    # ATR, OBV, Donchian, Mass Index, Keltner, Fibonacci, Envelope, v.v. chỉ phụ trợ, không xác nhận trực tiếp
    return None

def advanced_indicator_analysis(indicator_row, pattern_type):
    """Advanced indicator analysis với weighted scoring system"""
    if indicator_row is None or len(indicator_row) == 0:
        return {"score": 0, "signals": [], "strength": "Neutral", "confidence": 0}
    
    bullish_signals = 0
    bearish_signals = 0
    signal_details = []
    total_weight = 0
    
    # Trọng số cho từng loại indicator
    indicator_weights = {
        "RSI": 1.5, "MACD": 2.0, "StochRSI": 1.2, "UltimateOscillator": 1.0,
        "EMA": 1.8, "SMA": 1.5, "MA": 1.5, "BollingerBands": 1.3,
        "ADX": 1.7, "CCI": 1.0, "ATR": 0.8, "VWAP": 1.6,
        "IchimokuCloud": 2.2, "FibonacciRetracement": 1.4, "PivotPoints": 1.1, "Envelope": 1.2
    }
    
    for key, value in indicator_row.items():
        if "_signal" not in key:
            continue
        
        indicator_name = key.split("_signal")[0].split("_")[0]
        weight = indicator_weights.get(indicator_name, 1.0)
        total_weight += weight
        
        if value == "Bullish":
            bullish_signals += weight
            signal_details.append({"indicator": indicator_name, "signal": "Bullish", "weight": weight})
        elif value == "Bearish":
            bearish_signals += weight
            signal_details.append({"indicator": indicator_name, "signal": "Bearish", "weight": weight})
        else:
            signal_details.append({"indicator": indicator_name, "signal": "Neutral", "weight": weight})
    
    if total_weight == 0:
        return {"score": 0, "signals": signal_details, "strength": "Neutral", "confidence": 0}
    
    net_score = (bullish_signals - bearish_signals) / total_weight
    confidence = (bullish_signals + bearish_signals) / total_weight
    
    if net_score > 0.6 and confidence > 0.7:
        strength = "Bullish_Strong"
    elif net_score > 0.3:
        strength = "Bullish"
    elif net_score < -0.6 and confidence > 0.7:
        strength = "Bearish_Strong"
    elif net_score < -0.3:
        strength = "Bearish"
    else:
        strength = "Neutral"
    
    return {
        "score": net_score, "signals": signal_details, "strength": strength, "confidence": confidence,
        "bullish_weight": bullish_signals, "bearish_weight": bearish_signals, "total_weight": total_weight
    }

def advanced_trend_analysis(trend_data, candle_row, lookback_candles=None):
    """Advanced trend analysis với multiple levels và context"""
    # Fix pandas Series ambiguity - check if candle_row is empty or None properly
    if not trend_data or candle_row is None or (hasattr(candle_row, 'empty') and candle_row.empty):
        return {"strength": "Neutral", "context": [], "levels": [], "trend_direction": "Unknown", "momentum": "Neutral"}
    
    close = candle_row.get("close")
    if not close:
        return {"strength": "Neutral", "context": [], "levels": [], "trend_direction": "Unknown", "momentum": "Neutral"}
    
    analysis_result = {"strength": "Neutral", "context": [], "levels": [], "trend_direction": "Unknown", "momentum": "Neutral"}
    all_levels = []
    
    # Analyze all support/resistance levels
    for level in trend_data.get("support", []):
        distance_pct = abs(close - level) / close * 100
        all_levels.append({"type": "support", "value": level, "distance": distance_pct, "position": "above" if close > level else "below"})
    
    for level in trend_data.get("resistance", []):
        distance_pct = abs(close - level) / close * 100
        all_levels.append({"type": "resistance", "value": level, "distance": distance_pct, "position": "above" if close > level else "below"})
    
    for level in trend_data.get("channel_lower", []):
        distance_pct = abs(close - level) / close * 100
        all_levels.append({"type": "channel_lower", "value": level, "distance": distance_pct, "position": "above" if close > level else "below"})
        
    for level in trend_data.get("channel_upper", []):
        distance_pct = abs(close - level) / close * 100
        all_levels.append({"type": "channel_upper", "value": level, "distance": distance_pct, "position": "above" if close > level else "below"})
    
    for level in trend_data.get("trendline", []):
        distance_pct = abs(close - level) / close * 100
        all_levels.append({"type": "trendline", "value": level, "distance": distance_pct, "position": "above" if close > level else "below"})
    
    all_levels.sort(key=lambda x: x["distance"])
    analysis_result["levels"] = all_levels[:5]
    
    # Determine trend context
    bullish_factors = bearish_factors = 0
    
    for level in all_levels[:3]:
        if level["distance"] < 0.5:
            if level["type"] in ["support", "channel_lower", "trendline"] and level["position"] == "above":
                bullish_factors += 1
                analysis_result["context"].append(f"Above {level['type']} at {level['value']:.5f}")
            elif level["type"] in ["resistance", "channel_upper"] and level["position"] == "below": 
                bearish_factors += 1
                analysis_result["context"].append(f"Below {level['type']} at {level['value']:.5f}")
    
    if bullish_factors > bearish_factors:
        analysis_result["strength"] = "Bullish" if bullish_factors - bearish_factors == 1 else "Bullish_Strong"
        analysis_result["trend_direction"] = "Uptrend"
    elif bearish_factors > bullish_factors:
        analysis_result["strength"] = "Bearish" if bearish_factors - bullish_factors == 1 else "Bearish_Strong" 
        analysis_result["trend_direction"] = "Downtrend"
    else:
        analysis_result["strength"] = "Neutral"
        analysis_result["trend_direction"] = "Sideways"
    
    return analysis_result

def comprehensive_pattern_confirmation(pattern_row, indicator_row, trend_data, candle_row):
    """Comprehensive pattern confirmation combining indicators, trend, and price action"""
    pattern_name = pattern_row.get("pattern", "").lower()
    pattern_type = get_pattern_type(pattern_name)
    
    indicator_analysis = advanced_indicator_analysis(indicator_row, pattern_type)
    trend_analysis = advanced_trend_analysis(trend_data, candle_row)
    
    final_score = 0
    confidence_factors = []
    
    pattern_weights = {"doji": 0.8, "engulfing": 1.5, "hammer": 1.2, "shooting_star": 1.2, "pin_bar": 1.3, "morning_star": 1.8, "evening_star": 1.8, "harami": 1.0, "marubozu": 1.4}
    
    base_pattern_weight = 1.0
    for key, weight in pattern_weights.items():
        if key in pattern_name:
            base_pattern_weight = weight
            break
    
    # Weight distribution: 40% indicators, 35% trend, 25% pattern
    indicator_weight, trend_weight, pattern_weight = 0.40, 0.35, 0.25
    
    if indicator_analysis["confidence"] > 0:
        indicator_score = indicator_analysis["score"] * indicator_analysis["confidence"]
        final_score += indicator_score * indicator_weight
        confidence_factors.append(f"Indicators: {indicator_analysis['strength']} (conf: {indicator_analysis['confidence']:.2f})")
    
    trend_score_map = {"Bullish_Strong": 1.0, "Bullish": 0.6, "Neutral": 0.0, "Bearish": -0.6, "Bearish_Strong": -1.0}
    trend_score = trend_score_map.get(trend_analysis["strength"], 0)
    final_score += trend_score * trend_weight
    confidence_factors.append(f"Trend: {trend_analysis['strength']}")
    
    pattern_score_map = {"bullish": 0.8, "bearish": -0.8, "neutral": 0.0}
    pattern_base_score = pattern_score_map.get(pattern_type, 0)
    weighted_pattern_score = pattern_base_score * base_pattern_weight
    final_score += weighted_pattern_score * pattern_weight
    confidence_factors.append(f"Pattern: {pattern_name} ({pattern_type})")
    
    # Calculate proper confidence based on indicator strength and pattern reliability
    pattern_confidence = base_pattern_weight * 0.6  # Pattern reliability (60% of base weight)
    indicator_confidence = indicator_analysis.get("confidence", 0.3) * 0.3  # Indicator confidence (30%)
    trend_confidence = min(0.1, abs(trend_score) * 0.1)  # Trend alignment (10%)
    
    # Final confidence is combination of all factors (0 to 1)
    final_confidence = min(1.0, pattern_confidence + indicator_confidence + trend_confidence)
    
    # Signal classification based on score
    if final_score > 0.6:
        final_signal = "Bullish_Strong"
    elif final_score > 0.2:
        final_signal = "Bullish"
    elif final_score < -0.6:
        final_signal = "Bearish_Strong"
    elif final_score < -0.2:
        final_signal = "Bearish"
    else:
        final_signal = "Neutral"
    
    return {
        "signal": final_signal, 
        "score": final_score, 
        "confidence": final_confidence,  # Now properly calculated confidence
        "indicator_analysis": indicator_analysis, 
        "trend_analysis": trend_analysis,
        "confidence_factors": confidence_factors, 
        "pattern_weight": base_pattern_weight
    }

def analyze_patterns(symbol, timeframe, indicator_list=None, count=None):
    """Enhanced pattern analysis với comprehensive indicator and trend integration"""
    try:
        logger.info(f"🔍 Starting enhanced pattern analysis for {symbol} {timeframe}")
        
        # 🧹 CLEANUP OLD PATTERNS - Run once per session (check if already done)
        if not hasattr(analyze_patterns, '_cleanup_done'):
            logger.info("🧹 Running pattern cleanup (first run in this session)...")
            try:
                # Get selected timeframes from data folder
                selected_timeframes = set()
                selected_symbols = set()
                if os.path.exists(DATA_FOLDER):
                    for filename in os.listdir(DATA_FOLDER):
                        if filename.endswith('.json'):
                            parts = filename.replace('.json', '').split('_')
                            if len(parts) >= 2:
                                symbol_part = parts[0]
                                # Check for patterns like XAUUSD._ or XAUUSD_m_
                                if '.' in symbol_part:
                                    symbol_part = symbol_part.split('.')[0]
                                selected_symbols.add(symbol_part)
                                
                                # Extract timeframe
                                tf_part = parts[-1]  # Last part is usually timeframe
                                tf_clean = tf_part.replace('m_', '').strip()
                                if tf_clean in ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']:
                                    selected_timeframes.add(tf_clean)
                
                if selected_timeframes:
                    logger.info(f"🔍 Detected selected timeframes: {', '.join(sorted(selected_timeframes))}")
                    deleted_count, space_freed = cleanup_unselected_timeframes(list(selected_symbols), list(selected_timeframes))
                    logger.info(f"🗑️ Cleanup completed: {deleted_count} files deleted, {space_freed:.2f} MB freed")
                
                analyze_patterns._cleanup_done = True  # Mark cleanup as done
            except Exception as cleanup_err:
                logger.warning(f"⚠️ Cleanup failed (non-critical): {cleanup_err}")
        
        # Load candle data - try different naming conventions
        data_path = os.path.join(DATA_FOLDER, f"{symbol}_{timeframe}.json")
        if not os.path.exists(data_path):
            # Try alternative naming convention with m_ prefix
            data_path = os.path.join(DATA_FOLDER, f"{symbol}_m_{timeframe}.json")
        if not os.path.exists(data_path):
            # Try alternative naming convention with dot-underscore
            data_path = os.path.join(DATA_FOLDER, f"{symbol}._{timeframe}.json")
        
        if not os.path.exists(data_path):
            logger.warning(f"Data file not found: {data_path}")
            return []
            
        with open(data_path, 'r') as f:
            candles_data = json.load(f)
        
        if not candles_data:
            logger.warning(f"Empty data for {symbol} {timeframe}")
            return []
            
        candles = pd.DataFrame(candles_data)
        if len(candles) < 10:
            logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(candles)} candles")
            return []
            
        # Flatten OHLC columns if needed
        candles = flatten_ohlc_columns(candles)
        
        # Remove current candle if exists
        if not candles.empty and candles.iloc[-1].get("current", False):
            candles = candles.iloc[:-1]
            
        # Take last 20 candles for analysis (more context)
        candles = candles.tail(20).reset_index(drop=True)
        
        # Detect all patterns
        logger.info(f"🔍 Detecting candlestick patterns...")
        candles = detect_patterns(candles)
        
        # Load indicator data
        indicator_data = []
        pattern = os.path.join(INDICATOR_FOLDER, f"{symbol}*{timeframe}*indicat*.json")
        files = glob.glob(pattern)
        if files:
            try:
                with open(files[0], 'r') as f:
                    indi_data = json.load(f)
                    indi_data = pd.DataFrame(indi_data)
                    indi_data = flatten_ohlc_columns(indi_data)
                    # Take corresponding 20 records
                    indicator_data = indi_data.tail(20).reset_index(drop=True)
                logger.info(f"Loaded {len(indicator_data)} indicator records")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load indicators: {e}")
        else:
            logger.warning(f"⚠️ Indicator file not found for {symbol} {timeframe}")
            
        # Load trend data
        trend_data = load_trendline_sr(symbol, timeframe)
        if trend_data:
            logger.info(f"📈 Loaded trend data with {len(trend_data.get('support', []))} supports, {len(trend_data.get('resistance', []))} resistances")
        else:
            logger.warning(f"⚠️ No trend data found for {symbol} {timeframe}")
        
        # Analyze patterns with enhanced confirmation
        patterns_found = []
        pattern_columns = [col for col in candles.columns if candles[col].dtype == bool]
        
        # Use new single priority pattern logic (only 1 pattern per symbol/timeframe)
        priority_patterns = get_single_priority_pattern(candles, indicator_data, indicator_list, trend_data, lookback=5)
        
        for pattern_data in priority_patterns:
            patterns_found.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'type': pattern_data['type'],
                'time': pattern_data['time'],
                'confidence': pattern_data['confidence'],
                'score': pattern_data['score'],
                'signal': pattern_data['signal'],
                'pattern_length': pattern_data['pattern_length'],
                'indicator_analysis': pattern_data.get('indicator_analysis', {}),
                'trend_analysis': pattern_data.get('trend_analysis', {}),
                'confidence_factors': pattern_data.get('confidence_factors', [])
            })
        
        # Sort by confidence score (descending)
        patterns_found.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Save enhanced results
        if patterns_found:
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            output_path = os.path.join(OUTPUT_FOLDER, f"{symbol}_{timeframe}_priority_patterns.json")
            
            with open(output_path, 'w') as f:
                json.dump(patterns_found, f, indent=2, default=str)
                
            logger.info(f"Enhanced analysis complete: {len(patterns_found)} patterns found")
            
            # Log top patterns (removed emoji to fix encoding)
            for i, pattern in enumerate(patterns_found[:3]):
                logger.info(f"   #{i+1}: {pattern['type']} - {pattern['signal']} (conf: {pattern['confidence']}, score: {pattern.get('score', 0)})")
                logger.info(f"        Indicators: {pattern['indicator_analysis'].get('strength', 'Neutral')}, Trend: {pattern['trend_analysis'].get('strength', 'Neutral')}")
        else:
            logger.info(f"No patterns found for {symbol} {timeframe}")
            
        return patterns_found
        
    except Exception as e:
        logger.error(f"Error in enhanced pattern analysis: {e}")  # Remove Unicode emoji
        import traceback
        traceback.print_exc()
        return []

def load_trendline_sr(symbol, timeframe, folder="trendline_sr"):
    # Tìm file phù hợp
    pattern = f"{symbol}_*{timeframe}*_trendline_sr.json"
    files = [f for f in os.listdir(folder) if f.startswith(f"{symbol}_") and f.endswith(f"{timeframe}_trendline_sr.json")]
    if not files:
        return None
    path = os.path.join(folder, files[0])
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
def get_nearest_level(close, trend_data):
    levels = []
    for s in trend_data.get("support", []):
        levels.append(("support", s, abs(close - s)))
    for r in trend_data.get("resistance", []):
        levels.append(("resistance", r, abs(close - r)))
    for lower in trend_data.get("channel_lower", []):
        levels.append(("channel_lower", lower, abs(close - lower)))
    for upper in trend_data.get("channel_upper", []):
        levels.append(("channel_upper", upper, abs(close - upper)))
    for tl in trend_data.get("trendline", []):
        levels.append(("trendline", tl, abs(close - tl)))
    if not levels:
        return None, None
    levels.sort(key=lambda x: x[2])
    return levels[0][0], levels[0][1]
# filepath: c:\Users\ADMIN\OneDrive\Desktop\my_trading_bot\pattern_detector.py

def confirm_pattern_by_trend(pattern_row, candle_row, trend_data, threshold_pct=0.005):
    close = candle_row.get("close")
    if close is None or trend_data is None:
        return None

    level_type, level_value = get_nearest_level(close, trend_data)
    if level_type is None:
        return None

    if level_type in ["support", "channel_lower", "trendline"]:
        if close >= level_value:
            return {"type": f"above_{level_type}", "value": level_value, "signal": "Bullish"}
        else:
            return {"type": f"below_{level_type}", "value": level_value, "signal": "Bearish"}
    elif level_type in ["resistance", "channel_upper"]:
        if close <= level_value:
            return {"type": f"below_{level_type}", "value": level_value, "signal": "Bearish"}
        else:
            return {"type": f"above_{level_type}", "value": level_value, "signal": "Bullish"}
    return None
def confirm_pattern_by_trend_priority(pattern_row, indicator_row, trend_data, candle_row, threshold_pct=0.002):
    close = candle_row.get("close")
    if close is None:
        return "Neutral", []

    pattern_name = pattern_row.get("pattern", "").lower()
    pattern_type = get_pattern_type(pattern_name)  # "bullish", "bearish", "neutral"

    # Xác nhận với Trend (gần nhất)
    trend_signal = None
    level_type, level_value = None, None
    if trend_data is not None:
        level_type, level_value = get_nearest_level(close, trend_data)
        if level_type:
            if level_type in ["support", "channel_lower", "trendline"]:
                trend_signal = "bullish" if close >= level_value else "bearish"
            elif level_type in ["resistance", "channel_upper"]:
                trend_signal = "bearish" if close <= level_value else "bullish"

    # Xác nhận với indicator xu hướng mạnh
    indicators = pattern_row.get("confirmed_indicators", [])
    indicator_signal = None
    trend_indicators = []
    for indi in indicators:
        name = indi.get("name", "").lower()
        value = indi.get("value")
        if any(x in name for x in ["ema", "sma", "wma", "tema", "macd", "adx", "kst", "trix", "vortex", "psar"]):
            trend_indicators.append((name, value, indi))
    def extract_period(name):
        import re
        periods = [int(x) for x in re.findall(r'\d+', name)]
        return max(periods) if periods else 0
    trend_indicators.sort(key=lambda x: extract_period(x[0]), reverse=True)
    for name, value, indi in trend_indicators:
        signal = indicator_value_to_signal(name, value, close)
        if signal == "bullish":
            indicator_signal = "bullish"
            break
        elif signal == "bearish":
            indicator_signal = "bearish"
            break

    # Ưu tiên đặc tính Bull/Bear của mô hình
    if pattern_type in ["bullish", "bearish"]:
        if trend_signal == pattern_type:
            return f"{pattern_type.capitalize()}_Strong", []
        elif trend_signal and trend_signal != pattern_type:
            return "Neutral", []
        elif not trend_signal:
            if indicator_signal == pattern_type:
                return pattern_type.capitalize(), []
            else:
                return "Neutral", []
    else:
        # Nếu không phải mô hình Bull/Bear, giữ logic cũ
        if trend_signal and indicator_signal:
            if trend_signal == indicator_signal:
                return f"{trend_signal.capitalize()}_Strong", []
            else:
                return trend_signal.capitalize(), []
        elif trend_signal:
            return trend_signal.capitalize(), []
        elif indicator_signal:
            return f"{indicator_signal.capitalize()}_weak", []
        else:
            return "Neutral", []
def get_single_priority_pattern(candles, indi_data, indicator_list, trend_data, lookback=5):
    """
    Lấy chỉ 1 mô hình nến duy nhất theo ưu tiên:
    1. Thời gian gần nhất (5 nến cuối)
    2. Mô hình đa nến (pattern length cao hơn)
    3. Độ tin cậy cao hơn
    """
    if len(candles) < lookback:
        return []
    
    # Lấy 5 nến cuối cùng
    recent_candles = candles.tail(lookback).reset_index(drop=True)
    recent_indicators = indi_data.tail(lookback).reset_index(drop=True) if len(indi_data) > 0 else []
    
    # Duyệt từ nến gần nhất về nến cũ hơn (ưu tiên thời gian)
    for idx in reversed(range(len(recent_candles))):
        row = recent_candles.iloc[idx]
        patterns_at_time = []
        
        # Tìm tất cả patterns tại thời điểm này
        for pattern_name, value in row.items():
            if isinstance(value, (bool, np.bool_)) and value and pattern_name not in ["current"]:
                # Simple pattern length inference based on common patterns
                name = pattern_name.lower()
                if any(keyword in name for keyword in ['five', 'mat_hold', 'rising_three_methods', 'falling_three_methods', 'ladder']):
                    pattern_length = 5
                elif any(keyword in name for keyword in ['four', 'concealing_baby_swallow']):
                    pattern_length = 4
                elif any(keyword in name for keyword in ['three', 'morning', 'evening', 'inside', 'outside', 'abandoned']):
                    pattern_length = 3
                elif any(keyword in name for keyword in ['two', 'double', 'tweezer', 'harami', 'engulfing']):
                    pattern_length = 2
                else:
                    pattern_length = 1  # Default for single-candle patterns
                
                # Tính confidence cho pattern này
                ind_row = recent_indicators.iloc[idx] if idx < len(recent_indicators) else {}
                _, _, _, _, indicators, _ = combine_with_indicator(ind_row, indicator_list)
                
                pattern_row = {
                    "pattern": pattern_name,
                    "time": row["time"],
                    "confirmed_indicators": indicators,
                    "pattern_length": pattern_length
                }
                
                # Tính confidence
                confirmation = comprehensive_pattern_confirmation(pattern_row, ind_row, trend_data, row)
                confidence = confirmation.get('confidence', 0)
                
                patterns_at_time.append({
                    'pattern_name': pattern_name,
                    'pattern_length': pattern_length,
                    'confidence': confidence,
                    'pattern_row': pattern_row,
                    'candle_row': row,
                    'confirmation': confirmation,
                    'time_index': idx
                })
        
        if patterns_at_time:
            # Sắp xếp theo: pattern_length (giảm dần) → confidence (giảm dần)
            patterns_at_time.sort(key=lambda x: (-x['pattern_length'], -x['confidence']))
            
            # Lấy pattern có độ ưu tiên cao nhất tại thời điểm này
            best_pattern = patterns_at_time[0]
            
            # Tạo pattern object cuối cùng
            pattern_obj = {
                'type': best_pattern['pattern_name'],
                'time': best_pattern['pattern_row']['time'],
                'confidence': best_pattern['confidence'],
                'signal': best_pattern['confirmation']['signal'],
                'score': best_pattern['confirmation']['score'],
                'pattern_length': best_pattern['pattern_length'],
                'indicator_analysis': best_pattern['confirmation'].get('indicator_analysis', {}),
                'trend_analysis': best_pattern['confirmation'].get('trend_analysis', {}),
                'confidence_factors': best_pattern['confirmation'].get('confidence_factors', [])
            }
            
            # Trả về pattern đầu tiên tìm được (vì đã duyệt từ gần nhất)
            return [pattern_obj]
    
    return []
def get_all_patterns_signal(candles, indi_data, indicator_list, trend_data):
    results = []
    for idx in reversed(range(len(candles))):
        row = candles.iloc[idx]
        for pattern_name, value in row.items():
            if isinstance(value, bool) and value and pattern_name not in ["current"]:
                ind_row = indi_data.iloc[idx] if idx < len(indi_data) else {}
                _, _, _, _, indicators, _ = combine_with_indicator(ind_row, indicator_list)
                pattern_row = {
                    "pattern": pattern_name,
                    "time": row["time"],
                    "confirmed_indicators": indicators
                }
                trend_confirm = confirm_pattern_by_trend(pattern_row, row, trend_data)
                pattern_row["trend_confirm"] = trend_confirm
                signal, confirmed = confirm_pattern_by_indicators_and_trendline(pattern_row, ind_row, trend_data, row)
                pattern_row["signal"] = signal
                pattern_row["confirmed_indicators"] = confirmed
                results.append(pattern_row)
    return results
def get_pattern_type(pattern_name):
    bullish_patterns = [
        "bullish_engulfing", "morning_star", "hammer", "inverted_hammer", "piercing_line",
        "morning_doji_star", "three_white_soldiers", "abandoned_baby_bull", "kicker_bull",
        "counterattack_bull", "three_inside_up", "three_outside_up", "ladder_bottom",
        "dragonfly_doji", "belt_hold_bull", "mat_hold", "rising_three_methods", "stick_sandwich",
        "unique_three_river", "tristar_bull", "bullish_harami", "tweezer_bottom",
        "marubozu_white", "upside_gap_three_methods", "matching_low", "side_by_side_white_lines"
    ]
    bearish_patterns = [
        "bearish_engulfing", "evening_star", "shooting_star", "hanging_man", "dark_cloud_cover",
        "evening_doji_star", "three_black_crows", "abandoned_baby_bear", "kicker_bear",
        "counterattack_bear", "three_inside_down", "three_outside_down", "ladder_top",
        "gravestone_doji", "belt_hold_bear", "falling_three_methods", "advance_block",
        "tristar_bear", "bearish_harami", "tweezer_top", "marubozu_black",
        "downside_gap_three_methods", "matching_high", "upside_gap_two_crows"
    ]
    # Các mẫu nến trung tính hoặc không xác định rõ bull/bear sẽ trả về "neutral"
    name = pattern_name.lower()
    if name in bullish_patterns:
        return "bullish"
    if name in bearish_patterns:
        return "bearish"
    return "neutral"

def cleanup_all_pattern_data():
    """🗑️ Xóa TẤT CẢ dữ liệu pattern cũ trước khi chạy phân tích mới
    
    Returns:
        tuple: (số file đã xóa, dung lượng giải phóng MB)
    """
    directory = "pattern_signals"
    total_deleted = 0
    total_space_freed = 0.0
    
    if not os.path.exists(directory):
        return total_deleted, total_space_freed
    
    try:
        logger.info(f"🗑️ Xóa TẤT CẢ dữ liệu pattern cũ trong {directory}/")
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                try:
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    os.remove(file_path)
                    total_deleted += 1
                    total_space_freed += file_size
                    logger.debug(f"Deleted: {filename}")
                except Exception as e:
                    logger.error(f"Failed to delete {filename}: {e}")
        
        logger.info(f"✅ Đã xóa {total_deleted} file, giải phóng {total_space_freed:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error cleaning all pattern data: {e}")
    
    return total_deleted, total_space_freed

def cleanup_pattern_data(max_age_hours: int = 72, keep_latest: int = 10):
    """🧹 Smart cleanup - Delete only old files in pattern_signals, keep recent ones"""
    
    directory = "pattern_signals"
    total_deleted = 0
    total_space_freed = 0.0
    
    if not os.path.exists(directory):
        return total_deleted, total_space_freed
    
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        # Get all files with their modification times
        files_with_time = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                try:
                    mtime = os.path.getmtime(file_path)
                    files_with_time.append((file_path, mtime, filename))
                except Exception as e:
                    logger.error(f"Error getting mtime for {filename}: {e}")
        
        # Sort by modification time (newest first)
        files_with_time.sort(key=lambda x: x[1], reverse=True)
        
        # Keep latest files and delete old ones
        for i, (file_path, mtime, filename) in enumerate(files_with_time):
            file_age_seconds = current_time - mtime
            
            # Delete if older than max_age_hours OR beyond keep_latest count
            should_delete = (file_age_seconds > max_age_seconds) or (i >= keep_latest)
            
            if should_delete:
                try:
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    os.remove(file_path)
                    total_deleted += 1
                    total_space_freed += file_size
                    logger.debug(f"Deleted old pattern file: {filename} (age: {file_age_seconds/3600:.1f}h)")
                except Exception as e:
                    logger.error(f"Failed to delete {filename}: {e}")
                    
    except Exception as e:
        logger.error(f"Error cleaning pattern_signals: {e}")
    
    if total_deleted > 0:
        logger.info(f"🧹 Smart cleanup pattern_signals: {total_deleted} files deleted, {total_space_freed:.2f} MB freed (kept {len(files_with_time) - total_deleted} recent files)")
    else:
        logger.info(f"🧹 Pattern cleanup: All files are recent, kept {len(files_with_time) if 'files_with_time' in locals() else 0} files")
    
    return total_deleted, total_space_freed


def cleanup_unselected_timeframes(selected_symbols: List[str], selected_timeframes: List[str]) -> tuple:
    """🧹 Delete pattern files for unselected timeframes to avoid stale data in reports
    
    Args:
        selected_symbols: List of selected symbols (e.g., ['XAUUSD', 'EURUSD'])
        selected_timeframes: List of selected timeframes (e.g., ['M15', 'M30', 'H1'])
    
    Returns:
        tuple: (deleted_count, space_freed_mb)
    """
    directory = "pattern_signals"
    total_deleted = 0
    total_space_freed = 0.0
    
    if not os.path.exists(directory):
        logger.info(f"📁 Directory {directory} does not exist")
        return total_deleted, total_space_freed
    
    # All possible timeframes
    all_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
    unselected_timeframes = [tf for tf in all_timeframes if tf not in selected_timeframes]
    
    if not unselected_timeframes:
        logger.info("📊 All timeframes selected, no cleanup needed")
        return total_deleted, total_space_freed
    
    logger.info(f"🧹 Cleaning pattern files for unselected timeframes: {', '.join(unselected_timeframes)}")
    
    try:
        for filename in os.listdir(directory):
            if not filename.endswith('.json'):
                continue
            
            # Check if file matches unselected timeframe pattern
            # Pattern: SYMBOL_TF_*.json or SYMBOL._TF_*.json
            should_delete = False
            for tf in unselected_timeframes:
                # Match patterns like: XAUUSD_H4_priority_patterns.json, XAUUSD._H4_patterns.json
                if f"_{tf}_" in filename or f"._{tf}_" in filename or f"_{tf}." in filename:
                    should_delete = True
                    break
            
            if should_delete:
                file_path = os.path.join(directory, filename)
                try:
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    os.remove(file_path)
                    total_deleted += 1
                    total_space_freed += file_size
                    logger.info(f"   🗑️ Deleted unselected timeframe file: {filename}")
                except Exception as e:
                    logger.error(f"   ❌ Failed to delete {filename}: {e}")
    
    except Exception as e:
        logger.error(f"❌ Error cleaning unselected timeframes: {e}")
    
    if total_deleted > 0:
        logger.info(f"✅ Cleanup complete: {total_deleted} files deleted, {total_space_freed:.2f} MB freed")
    else:
        logger.info(f"✅ No files to clean for unselected timeframes")
    
    return total_deleted, total_space_freed


# Clean start for simplified version

def save_patterns_to_file(patterns: List[Dict], symbol: str, timeframe: str) -> bool:
    """Save detected patterns to JSON file"""
    try:
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{timeframe}_patterns_{timestamp}.json"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        
        output_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': timestamp,
            'patterns_count': len(patterns),
            'patterns': patterns
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved {len(patterns)} patterns to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving patterns for {symbol}_{timeframe}: {e}")
        return False

def cleanup_old_pattern_files(max_age_hours: int = 48, keep_latest: int = 10) -> Dict[str, Any]:
    """
    🧹 Legacy function - calls the new comprehensive cleanup_pattern_data function
    """
    return cleanup_pattern_data(max_age_hours, keep_latest)

def cleanup_all_pattern_directories(max_age_hours: int = 72, force_clean: bool = False) -> Dict[str, Any]:
    """
    🧹 COMPREHENSIVE PATTERN DIRECTORY CLEANUP - Cleans all pattern-related directories
    
    Args:
        max_age_hours: Delete files older than this many hours
        force_clean: If True, delete files regardless of active symbols
    
    Returns:
        Dict with cleanup statistics
    """
    import pickle
    
    # Get active symbols from user config (unless force_clean is True)
    active_symbols = set()
    if not force_clean:
        try:
            with open('user_config.pkl', 'rb') as f:
                config = pickle.load(f)
                active_symbols = set(config.get('checked_symbols', []))
        except Exception:
            active_symbols = set()
    
    cleanup_stats = {
        'module_name': 'pattern_detector_comprehensive_all',
        'active_symbols': list(active_symbols) if not force_clean else ['ALL_SYMBOLS'],
        'force_clean': force_clean,
        'total_files_deleted': 0,
        'total_space_freed_mb': 0.0,
        'cleanup_time': datetime.now().isoformat(),
        'directories_cleaned': []
    }
    
    # All possible pattern-related directories
    all_pattern_directories = [
        "pattern_signals",      # Main pattern detection output
        "pattern_price",        # Price pattern analysis
        "trendline_sr",         # Support/resistance and trendlines  
        # "smart_analysis_data", # No longer used - moved to analysis_results
        # "reports",            # Analysis reports now unified in risk_settings.json
        "analysis_results",     # Analysis results that may include patterns
        "cache"                 # General cache that may contain pattern data
    ]
    
    logger.info(f"🧹 Starting comprehensive pattern directory cleanup...")
    logger.info(f"   Mode: {'FORCE CLEAN (all files)' if force_clean else 'SMART CLEAN (active symbols only)'}")
    logger.info(f"   Max age: {max_age_hours} hours")
    logger.info(f"   Active symbols: {len(active_symbols) if not force_clean else 'ALL'}")
    
    for directory in all_pattern_directories:
        if not os.path.exists(directory):
            logger.debug(f"📁 Directory {directory} does not exist, skipping")
            continue
            
        logger.info(f"🧹 Cleaning {directory}...")
        deleted_count = 0
        space_freed = 0.0
        
        try:
            for filename in os.listdir(directory):
                # Only process relevant files
                if not (filename.endswith('.json') or filename.endswith('.txt') or filename.endswith('.log')):
                    continue
                    
                file_path = os.path.join(directory, filename)
                
                try:
                    should_delete = False
                    
                    if force_clean:
                        # Force clean mode: delete old files regardless of symbol
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_time < datetime.now() - timedelta(hours=max_age_hours):
                            should_delete = True
                            logger.debug(f"Force deleting {filename}: older than {max_age_hours} hours")
                    else:
                        # Smart clean mode: check symbol and age
                        symbol_from_file = None
                        for symbol in active_symbols:
                            # Handle different filename formats and symbol variations
                            clean_symbol = symbol.replace('.', '').replace('_m', '')
                            if (symbol in filename or 
                                clean_symbol in filename or 
                                filename.startswith(symbol) or 
                                filename.startswith(clean_symbol)):
                                symbol_from_file = symbol
                                break
                        
                        if symbol_from_file is None:
                            should_delete = True
                            logger.debug(f"Smart deleting {filename}: symbol not in active list")
                        else:
                            # Check file age for active symbols
                            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                            if file_time < datetime.now() - timedelta(hours=max_age_hours):
                                should_delete = True
                                logger.debug(f"Smart deleting {filename}: older than {max_age_hours} hours")
                    
                    if should_delete:
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        os.remove(file_path)
                        deleted_count += 1
                        space_freed += file_size
                        
                except Exception as e:
                    logger.warning(f"Error processing {filename}: {e}")
                    
        except Exception as e:
            logger.error(f"Error accessing directory {directory}: {e}")
        
        if deleted_count > 0:
            logger.info(f"   ✅ {directory}: {deleted_count} files, {space_freed:.2f} MB")
        else:
            logger.debug(f"   ➖ {directory}: no files deleted")
            
        cleanup_stats['total_files_deleted'] += deleted_count
        cleanup_stats['total_space_freed_mb'] += space_freed
        cleanup_stats['directories_cleaned'].append({
            'directory': directory,
            'files_deleted': deleted_count,
            'space_freed_mb': space_freed
        })
    
    logger.info(f"🧹 COMPREHENSIVE PATTERN CLEANUP complete:")
    logger.info(f"   📁 Directories processed: {len(all_pattern_directories)}")
    logger.info(f"   🗑️ Total files deleted: {cleanup_stats['total_files_deleted']}")
    logger.info(f"   💾 Total space freed: {cleanup_stats['total_space_freed_mb']:.2f} MB")
    
    return cleanup_stats

def save_patterns_with_cleanup(patterns: List[Dict], symbol: str, timeframe: str, 
                              auto_cleanup: bool = True) -> bool:
    """
    💾 Lưu patterns với tự động dọn dẹp file cũ
    
    Args:
        patterns: Pattern data list
        symbol: Symbol name
        timeframe: Timeframe
        auto_cleanup: Tự động dọn dẹp file cũ không
    """
    try:
        # Create output filename
        output_filename = f"{symbol}_{timeframe}_priority_patterns.json"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Auto cleanup deprecated - now handled by module auto-cleanup
        
        # Add metadata
        output_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'patterns_count': len(patterns),
            'patterns': patterns
        }
        
        # Save file
        success = overwrite_json_safely(output_path, output_data, backup=False)
        
        if success:
            logger.info(f"� Patterns saved with cleanup: {output_filename}")
        else:
            logger.error(f"❌ Failed to save patterns: {output_filename}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error saving patterns for {symbol}_{timeframe}: {e}")
        return False

def get_pattern_storage_report() -> Dict[str, Any]:
    """📊 Get comprehensive storage report for all pattern directories"""
    
    pattern_directories = [
        "pattern_signals", "pattern_price", "trendline_sr", 
        "analysis_results", "cache"
        # "smart_analysis_data" - no longer used 
        # "reports" - now unified in risk_settings.json
    ]
    
    storage_report = {
        'total_files': 0,
        'total_size_mb': 0.0,
        'directories': [],
        'timestamp': datetime.now().isoformat()
    }
    
    for directory in pattern_directories:
        if not os.path.exists(directory):
            continue
            
        dir_info = {
            'directory': directory,
            'files': 0,
            'size_mb': 0.0,
            'file_types': {}
        }
        
        try:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    file_ext = os.path.splitext(filename)[1].lower()
                    
                    dir_info['files'] += 1
                    dir_info['size_mb'] += file_size
                    
                    if file_ext in dir_info['file_types']:
                        dir_info['file_types'][file_ext] += 1
                    else:
                        dir_info['file_types'][file_ext] = 1
                        
        except Exception as e:
            logger.warning(f"Error scanning {directory}: {e}")
            continue
        
        if dir_info['files'] > 0:
            storage_report['directories'].append(dir_info)
            storage_report['total_files'] += dir_info['files']
            storage_report['total_size_mb'] += dir_info['size_mb']
    
    return storage_report

def cleanup_pattern_files_on_demand(
    directories: List[str] = None, 
    max_age_hours: int = 48,
    min_files_to_keep: int = 5,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    🧹 ON-DEMAND PATTERN CLEANUP - Flexible cleanup for specific needs
    
    Args:
        directories: List of specific directories to clean (None = default pattern dirs)
        max_age_hours: Delete files older than this many hours
        min_files_to_keep: Always keep at least this many recent files per symbol
        dry_run: If True, show what would be deleted without actually deleting
    
    Returns:
        Dict with cleanup statistics and actions taken
    """
    if directories is None:
        directories = ["pattern_signals", "pattern_price", "trendline_sr"]
    
    cleanup_stats = {
        'mode': 'ON_DEMAND_CLEANUP',
        'dry_run': dry_run,
        'directories_processed': directories,
        'max_age_hours': max_age_hours,
        'min_files_to_keep': min_files_to_keep,
        'total_files_found': 0,
        'total_files_to_delete': 0,
        'total_files_deleted': 0,
        'total_space_to_free_mb': 0.0,
        'total_space_freed_mb': 0.0,
        'directory_results': [],
        'timestamp': datetime.now().isoformat()
    }
    
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    for directory in directories:
        if not os.path.exists(directory):
            logger.info(f"📁 Directory {directory} does not exist, skipping")
            continue
            
        dir_result = {
            'directory': directory,
            'files_found': 0,
            'files_to_delete': 0,
            'files_deleted': 0,
            'space_to_free_mb': 0.0,
            'space_freed_mb': 0.0,
            'actions': []
        }
        
        logger.info(f"🔍 {'DRY RUN: ' if dry_run else ''}Analyzing {directory}...")
        
        try:
            # Group files by symbol for smart keeping
            symbol_files = {}
            all_files = []
            
            for filename in os.listdir(directory):
                if not filename.endswith('.json'):
                    continue
                    
                file_path = os.path.join(directory, filename)
                file_stat = os.stat(file_path)
                file_time = datetime.fromtimestamp(file_stat.st_mtime)
                file_size = file_stat.st_size / (1024 * 1024)  # MB
                
                # Extract symbol from filename
                symbol = 'unknown'
                for potential_symbol in ['XAUUSD', 'GBPUSD', 'EURUSD', 'USDJPY']:
                    if potential_symbol in filename:
                        symbol = potential_symbol
                        break
                
                file_info = {
                    'filename': filename,
                    'path': file_path,
                    'symbol': symbol,
                    'time': file_time,
                    'size_mb': file_size,
                    'age_hours': (datetime.now() - file_time).total_seconds() / 3600
                }
                
                all_files.append(file_info)
                
                if symbol not in symbol_files:
                    symbol_files[symbol] = []
                symbol_files[symbol].append(file_info)
            
            # Sort files within each symbol group by time (newest first)
            for symbol in symbol_files:
                symbol_files[symbol].sort(key=lambda x: x['time'], reverse=True)
            
            dir_result['files_found'] = len(all_files)
            cleanup_stats['total_files_found'] += len(all_files)
            
            # Determine which files to delete
            for symbol, files in symbol_files.items():
                for i, file_info in enumerate(files):
                    should_delete = False
                    reason = ""
                    
                    # Always keep minimum number of recent files per symbol
                    if i >= min_files_to_keep:
                        # Delete if older than max_age_hours
                        if file_info['time'] < cutoff_time:
                            should_delete = True
                            reason = f"older than {max_age_hours}h"
                    
                    if should_delete:
                        dir_result['files_to_delete'] += 1
                        dir_result['space_to_free_mb'] += file_info['size_mb']
                        
                        action = {
                            'action': 'DELETE' if not dry_run else 'WOULD_DELETE',
                            'filename': file_info['filename'],
                            'symbol': symbol,
                            'age_hours': round(file_info['age_hours'], 1),
                            'size_mb': round(file_info['size_mb'], 3),
                            'reason': reason
                        }
                        dir_result['actions'].append(action)
                        
                        # Actually delete if not dry run
                        if not dry_run:
                            try:
                                os.remove(file_info['path'])
                                dir_result['files_deleted'] += 1
                                dir_result['space_freed_mb'] += file_info['size_mb']
                                logger.debug(f"Deleted: {file_info['filename']}")
                            except Exception as e:
                                logger.error(f"Failed to delete {file_info['filename']}: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing directory {directory}: {e}")
        
        cleanup_stats['total_files_to_delete'] += dir_result['files_to_delete']
        cleanup_stats['total_files_deleted'] += dir_result['files_deleted']
        cleanup_stats['total_space_to_free_mb'] += dir_result['space_to_free_mb']
        cleanup_stats['total_space_freed_mb'] += dir_result['space_freed_mb']
        cleanup_stats['directory_results'].append(dir_result)
        
        if dir_result['files_to_delete'] > 0:
            action_verb = "Would delete" if dry_run else "Deleted"
            logger.info(f"   🗑️ {action_verb}: {dir_result['files_to_delete']} files, "
                       f"{dir_result['space_to_free_mb']:.2f} MB")
        else:
            logger.info(f"   ✅ No files to delete in {directory}")
    
    mode_desc = "DRY RUN" if dry_run else "ACTUAL CLEANUP"
    logger.info(f"🧹 ON-DEMAND PATTERN CLEANUP ({mode_desc}) complete:")
    logger.info(f"   📊 Total files processed: {cleanup_stats['total_files_found']}")
    logger.info(f"   🗑️ Files {'to delete' if dry_run else 'deleted'}: {cleanup_stats['total_files_to_delete' if dry_run else 'total_files_deleted']}")
    logger.info(f"   💾 Space {'to free' if dry_run else 'freed'}: {cleanup_stats['total_space_to_free_mb' if dry_run else 'total_space_freed_mb']:.2f} MB")
    
    return cleanup_stats

def main(clean_all=False):
    """Main function for testing pattern detector with automatic cleanup
    
    Args:
        clean_all: Nếu True, xóa TẤT CẢ dữ liệu cũ trước khi chạy
    """
    print("🔍 Pattern Detector - Auto cleanup enabled")
    print("=" * 50)
    
    # 🗑️ XÓA TẤT CẢ dữ liệu cũ nếu được yêu cầu
    if clean_all:
        print("🗑️ Xóa TẤT CẢ dữ liệu pattern cũ...")
        deleted_all, space_freed_all = cleanup_all_pattern_data()
        if deleted_all > 0:
            print(f"✅ Đã xóa: {deleted_all} files, {space_freed_all:.2f} MB")
        else:
            print("✅ Không có dữ liệu cũ để xóa")
        print()
    
    # 🧹 AUTO CLEANUP - Clean unselected timeframes FIRST (before age-based cleanup)
    # Read selected timeframes from data files
    selected_timeframes = set()
    selected_symbols = set()
    if os.path.exists(DATA_FOLDER):
        for filename in os.listdir(DATA_FOLDER):
            if filename.endswith('.json'):
                # Extract timeframe from filename (e.g., XAUUSD_M15.json, XAUUSD_m_H1.json)
                parts = filename.replace('.json', '').split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    # Check if it has 'm' prefix
                    if parts[1] == 'm' and len(parts) > 2:
                        timeframe = parts[2]
                    else:
                        timeframe = parts[1]
                    
                    # Extract just the timeframe part (remove _indicators, etc.)
                    tf_clean = timeframe.split('_')[0].upper()
                    if tf_clean in ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']:
                        selected_timeframes.add(tf_clean)
                        selected_symbols.add(symbol)
    
    if selected_timeframes:
        print(f"🔍 Detected selected timeframes: {', '.join(sorted(selected_timeframes))}")
        print(f"🧹 Cleaning pattern files for unselected timeframes...")
        deleted_tf, space_freed_tf = cleanup_unselected_timeframes(list(selected_symbols), list(selected_timeframes))
        if deleted_tf > 0:
            print(f"✅ Cleaned unselected TF files: {deleted_tf} files, {space_freed_tf:.2f} MB freed")
        else:
            print("✅ No unselected timeframe files to clean")
    
    # 🧹 AUTO CLEANUP - Smart cleanup keeping recent files
    print("🧹 Smart cleaning old pattern_signals...")
    deleted, space_freed = cleanup_pattern_data(max_age_hours=72, keep_latest=20)
    if deleted > 0:
        print(f"✅ Cleaned: {deleted} old files, {space_freed:.2f} MB freed")
    else:
        print("✅ All pattern files are recent")
    
    # Show availability status
    print("\n📊 Module Status:")
    print(f"   • NumPy: {'✅' if NUMPY_AVAILABLE else '❌'}")
    print(f"   • Pandas: {'✅' if PANDAS_AVAILABLE else '❌'}")
    print(f"   • Numba: {'✅' if NUMBA_AVAILABLE else '❌'}")
    print(f"   • Concurrent Futures: {'✅' if CONCURRENT_AVAILABLE else '❌'}")
    print()
    
    # Check data directory
    if os.path.exists(DATA_FOLDER):
        data_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]
        print(f"📁 Found {len(data_files)} data files in {DATA_FOLDER}")
        
        if data_files:
            # Process all available files instead of just the first one
            print(f"🧪 Testing with all {len(data_files)} files...")
            
            # Auto cleanup before analysis
            logger.info("🧹 Pattern Detector: Auto cleanup before analysis...")
            deleted_pre, space_freed_pre = cleanup_pattern_data(max_age_hours=48, keep_latest=15)
            logger.info(f"✅ Pattern cleanup: {deleted_pre} files deleted, {space_freed_pre:.2f} MB freed")
            
            # Process each file
            total_patterns = 0
            processed_symbols = []
            
            for data_file in data_files:
                # Extract symbol and timeframe from filename
                # Expecting format: SYMBOL_m_TF_indicators.json or SYMBOL_TF_indicators.json
                file_base = data_file.replace('.json', '')
                parts = file_base.split('_')
                
                if len(parts) >= 3:
                    symbol_part = parts[0]
                    # Check if it has 'm' prefix (SYMBOL_m_TF format)
                    if parts[1] == 'm':
                        timeframe = parts[2] if len(parts) > 2 else 'H1'
                    else:
                        timeframe = parts[1] if len(parts) > 1 else 'H1'
                else:
                    symbol_part = parts[0]
                    timeframe = 'H1'  # Default timeframe
                
                print(f"📈 Analyzing {symbol_part} {timeframe.lower()}...")
                
                # Load and analyze data
                patterns = analyze_patterns(
                    symbol=symbol_part,
                    timeframe=timeframe
                )
                
                if patterns:
                    total_patterns += len(patterns)
                    processed_symbols.append(f"{symbol_part}({len(patterns)})")
                    print(f"✅ Found {len(patterns)} patterns for {symbol_part}")
                    # Show top 2 patterns for each symbol
                    for i, pattern in enumerate(patterns[:2], 1):
                        conf = pattern.get('confidence', 0)
                        signal = pattern.get('signal', 'neutral')
                        print(f"   {i}. {pattern['type']} (confidence: {conf:.2f}, signal: {signal})")
                else:
                    print(f"❌ No patterns detected for {symbol_part}")
            
            print(f"\n🎯 Summary: {total_patterns} total patterns across {len(processed_symbols)} symbols")
            if processed_symbols:
                print(f"   Processed: {', '.join(processed_symbols)}")
                
        else:
            print("❌ No data files found for testing")
    else:
        print(f"❌ Data folder {DATA_FOLDER} not found")
    
    print("\n✅ Pattern detector test completed")

if __name__ == "__main__":
    import sys
    
    # Kiểm tra argument để xóa tất cả dữ liệu cũ
    clean_all = "--clean-all" in sys.argv or "-c" in sys.argv
    
    if clean_all:
        print("🗑️ MODE: Xóa TẤT CẢ dữ liệu pattern cũ trước khi chạy")
    
    main(clean_all=clean_all)

# =============================================================================
# GUI HELPER FUNCTIONS FOR PATTERN LOADING AND PROCESSING
# Moved from gui_helpers.py to centralize all pattern-related logic
# =============================================================================

from typing import List, Dict, Set, Optional, Any

class PatternLoader:
    """Helper class to handle pattern loading and filtering logic for GUI"""
    
    # Define candlestick patterns (consistent with pattern detection above)
    BULLISH_PATTERNS = {
        "bullish_engulfing", "morning_star", "hammer", "inverted_hammer", "piercing_line",
        "morning_doji_star", "three_white_soldiers", "abandoned_baby_bull", "kicker_bull",
        "counterattack_bull", "three_inside_up", "three_outside_up", "ladder_bottom",
        "dragonfly_doji", "belt_hold_bull", "mat_hold", "rising_three_methods", "stick_sandwich",
        "unique_three_river", "tristar_bull", "bullish_harami", "tweezer_bottom",
        "marubozu_white", "upside_gap_three_methods", "matching_low", "side_by_side_white_lines",
        "bullish_pinbar", "bullish_pin_bar"
    }
    
    BEARISH_PATTERNS = {
        "bearish_engulfing", "evening_star", "shooting_star", "hanging_man", "dark_cloud_cover",
        "evening_doji_star", "three_black_crows", "abandoned_baby_bear", "kicker_bear",
        "counterattack_bear", "three_inside_down", "three_outside_down", "ladder_top",
        "gravestone_doji", "belt_hold_bear", "falling_three_methods", "advance_block",
        "tristar_bear", "bearish_harami", "tweezer_top", "marubozu_black",
        "downside_gap_three_methods", "matching_high", "upside_gap_two_crows",
        "bearish_pinbar", "bearish_pin_bar"
    }
    
    NEUTRAL_PATTERNS = {
        "doji", "spinning_top", "high_wave", "pinbar", "pin_bar", 
        "inside_bar", "outside_bar", "long_legged_doji", "four_price_doji"
    }
    
    @classmethod
    def get_all_candlestick_patterns(cls) -> Set[str]:
        """Get all defined candlestick patterns"""
        return cls.BULLISH_PATTERNS | cls.BEARISH_PATTERNS | cls.NEUTRAL_PATTERNS
    
    @staticmethod
    def infer_pattern_length(pattern_name: str) -> int:
        """
        Infer pattern length from pattern name
        Longer patterns are considered more significant
        """
        name = pattern_name.lower()
        if "five" in name:
            return 5
        if "four" in name:
            return 4
        if "three" in name:
            return 3
        if "two" in name or "double" in name:
            return 2
        if "single" in name or "one" in name:
            return 1
        
        # Look for numbers in pattern name
        numbers = re.findall(r'\d+', name)
        if numbers:
            return int(numbers[0])
        
        return 1  # Default length
    
    @staticmethod
    def is_candlestick_pattern(pattern_name: str) -> bool:
        """Check if a pattern is a candlestick pattern"""
        return pattern_name.lower() in PatternLoader.get_all_candlestick_patterns()
    
    @staticmethod
    def parse_filename(filename: str) -> Optional[tuple]:
        """Parse pattern filename to extract symbol and timeframe"""
        if filename.endswith("_priority_patterns.json"):
            base = filename[:-len("_priority_patterns.json")]
        elif filename.endswith("_enhanced_patterns.json"):
            base = filename[:-len("_enhanced_patterns.json")]
        elif filename.endswith("_pattern_signals.json"):
            base = filename[:-len("_pattern_signals.json")]
        else:
            return None
        
        # Find last underscore to separate timeframe
        last_underscore = base.rfind("_")
        if last_underscore == -1:
            return None
            
        symbol = base[:last_underscore]
        timeframe = base[last_underscore+1:]
        return symbol, timeframe
    
    @staticmethod
    def load_patterns_from_folder(
        folder_path: str,
        selected_symbols: Optional[Set[str]] = None,
        selected_timeframes: Optional[Set[str]] = None,
        candlestick_only: bool = False,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Load and filter patterns from folder
        Returns list of pattern objects ready for display
        """
        if not os.path.exists(folder_path):
            return []
        
        all_patterns = []
        candlestick_patterns = PatternLoader.get_all_candlestick_patterns()
        
        for filename in os.listdir(folder_path):
            # Only process pattern files
            if not (filename.endswith("_pattern_signals.json") or 
                   filename.endswith("_enhanced_patterns.json") or
                   filename.endswith("_priority_patterns.json")):
                continue
            
            # Parse filename
            parsed = PatternLoader.parse_filename(filename)
            if not parsed:
                continue
            
            symbol, timeframe = parsed
            
            # Apply symbol/timeframe filters
            if selected_symbols is not None and symbol not in selected_symbols:
                continue
            if selected_timeframes is not None and timeframe not in selected_timeframes:
                continue
            
            file_path = os.path.join(folder_path, filename)
            if os.path.getsize(file_path) == 0:
                continue  # Skip empty files
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Handle different data formats
                if isinstance(data, list):
                    patterns = data
                elif isinstance(data, dict) and "patterns" in data:
                    patterns = data["patterns"]
                else:
                    patterns = [data] if data else []
                
                for pattern in patterns:
                    pattern_obj = PatternLoader._process_pattern(
                        pattern, symbol, timeframe, candlestick_patterns,
                        candlestick_only, min_confidence
                    )
                    if pattern_obj:
                        all_patterns.append(pattern_obj)
                        
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
        
        return all_patterns
    
    @staticmethod
    def _process_pattern(
        pattern: Dict,
        symbol: str,
        timeframe: str,
        candlestick_patterns: Set[str],
        candlestick_only: bool,
        min_confidence: float
    ) -> Optional[Dict[str, Any]]:
        """Process a single pattern and apply filters"""
        time_key = pattern.get("time", "")
        pattern_name = pattern.get("type", pattern.get("pattern", ""))
        signal = pattern.get("signal", "")
        confidence = pattern.get("confidence", 0)
        
        # Apply filters
        is_candlestick = pattern_name.lower() in candlestick_patterns
        meets_confidence = confidence >= min_confidence
        
        # Filter logic
        if candlestick_only and not is_candlestick:
            return None  # Skip non-candlestick patterns if filter is on
        
        if not meets_confidence:
            return None  # Skip low confidence patterns
        
        # Create pattern object with separate signal and confidence
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'time': time_key,
            'pattern': pattern_name,
            'signal': signal,  # Keep signal separate from confidence
            'confidence': confidence,
            'score': pattern.get('score', 0.0),  # Add score field
            'is_candlestick': is_candlestick,
            'pattern_length': PatternLoader.infer_pattern_length(pattern_name)
        }
    
    @staticmethod
    def sort_patterns_by_priority(patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort patterns by priority: Recent patterns first, then by confidence and pattern quality"""
        def get_sort_key(pattern):
            # Get the most recent time from pattern (prioritize end_time > time > start_time)
            pattern_time = None
            if 'end_time' in pattern and pattern['end_time']:
                pattern_time = pattern['end_time']
            elif 'time' in pattern and pattern['time']:
                pattern_time = pattern['time']
            elif 'start_time' in pattern and pattern['start_time']:
                pattern_time = pattern['start_time']
            else:
                pattern_time = datetime.min.isoformat()  # Default to very old time
            
            # Convert to comparable format if needed
            if isinstance(pattern_time, str):
                try:
                    pattern_time = datetime.fromisoformat(pattern_time.replace('Z', '+00:00'))
                except:
                    pattern_time = datetime.min
            elif not isinstance(pattern_time, datetime):
                pattern_time = datetime.min
            
            return (
                pattern['symbol'],                    # Group by Symbol first
                pattern['timeframe'],                 # Then by timeframe within each symbol
                not pattern.get('is_candlestick', False),  # Candlestick patterns first
                -pattern_time.timestamp(),            # Most recent patterns first (negative for descending)
                -pattern.get('confidence', 0),        # Higher confidence first
                -pattern.get('score', 0),             # Higher score first
                -pattern.get('pattern_length', 0)     # Longer patterns first
            )
        
        return sorted(patterns, key=get_sort_key)

class PatternUIHelper:
    """Helper class for pattern UI operations"""
    
    @staticmethod
    def get_pattern_statistics(patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for pattern display"""
        if not patterns:
            return {
                'total_count': 0,
                'candlestick_count': 0,
                'avg_length': 0.0,
                'max_length': 0,
                'avg_confidence': 0.0,
                'max_confidence': 0.0
            }
        
        candlestick_count = sum(1 for p in patterns if p['is_candlestick'])
        total_count = len(patterns)
        avg_length = sum(p['pattern_length'] for p in patterns) / total_count
        max_length = max(p['pattern_length'] for p in patterns)
        avg_confidence = sum(p['confidence'] for p in patterns) / total_count
        max_confidence = max(p['confidence'] for p in patterns)
        
        return {
            'total_count': total_count,
            'candlestick_count': candlestick_count,
            'avg_length': avg_length,
            'max_length': max_length,
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence
        }
    
    @staticmethod
    def format_status_message(stats: Dict[str, Any], candlestick_only: bool) -> str:
        """Format status message based on pattern statistics"""
        if stats['total_count'] == 0:
            return "❌ No patterns found matching criteria"
        
        if candlestick_only:
            return (
                f"🕯️ Showing {stats['total_count']} candlestick patterns only | "
                f"Avg confidence: {stats['avg_confidence']:.2f} | "
                f"Max confidence: {stats['max_confidence']:.2f}"
            )
        else:
            return (
                f"📊 Showing {stats['total_count']} patterns total "
                f"({stats['candlestick_count']} candlestick + {stats['total_count'] - stats['candlestick_count']} other) | "
                f"Avg confidence: {stats['avg_confidence']:.2f} | "
                f"Max confidence: {stats['max_confidence']:.2f}"
            )

class PricePatternLoader:
    """Helper class for price pattern loading and processing"""
    
    @staticmethod
    def parse_price_pattern_filename(filename: str) -> Optional[tuple]:
        """Parse price pattern filename to extract symbol and timeframe"""
        if filename.endswith("_patterns.json"):
            base = filename[:-len("_patterns.json")]
            parts = base.split("_")
            if len(parts) >= 2:
                symbol = "_".join(parts[:-1])
                timeframe = parts[-1]
                # Clean up any dots from symbol names
                symbol = symbol.replace(".", "")
                return symbol, timeframe
        return None
    
    @staticmethod
    def load_price_patterns_from_folder(
        folder_path: str,
        selected_symbols: Optional[Set[str]] = None,
        selected_timeframes: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """Load price patterns from folder with filtering"""
        if not os.path.exists(folder_path):
            return []
        
        all_patterns = []
        
        for filename in os.listdir(folder_path):
            if not filename.endswith("_patterns.json"):
                continue
            
            # Parse filename
            parsed = PricePatternLoader.parse_price_pattern_filename(filename)
            if not parsed:
                continue
            
            symbol, timeframe = parsed
            
            file_path = os.path.join(folder_path, filename)
            if os.path.getsize(file_path) == 0:
                continue  # Skip empty files
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                for pattern in data:
                    # Handle different time field formats
                    time_key = (
                        pattern.get("start_time") or 
                        pattern.get("head_time") or 
                        pattern.get("peak1_time") or 
                        pattern.get("valley1_time") or 
                        pattern.get("time", "")
                    )
                    
                    # Handle different pattern type field formats
                    pattern_type = (
                        pattern.get("type") or 
                        pattern.get("pattern", "")
                    )
                    
                    # Use symbol from JSON file, not from filename (to handle dots properly)
                    pattern_symbol = pattern.get("symbol", symbol)
                    pattern_timeframe = pattern.get("timeframe", timeframe)
                    
                    # Clean symbol for filtering (remove dots for comparison)
                    clean_symbol = pattern_symbol.replace(".", "")
                    
                    # Apply symbol filter using cleaned symbol
                    if selected_symbols is not None and len(selected_symbols) > 0:
                        if clean_symbol not in selected_symbols and pattern_symbol not in selected_symbols:
                            continue
                    
                    # Apply timeframe filter
                    if selected_timeframes is not None and len(selected_timeframes) > 0:
                        if pattern_timeframe not in selected_timeframes:
                            continue
                    
                    pattern_obj = {
                        'symbol': clean_symbol,  # Use cleaned symbol for display consistency
                        'timeframe': pattern_timeframe,
                        'time': time_key,
                        'pattern': pattern_type,
                        'is_candlestick': False,  # Price patterns are not candlestick patterns
                        'confidence': pattern.get('confidence', 0.5),  # Default confidence
                        'pattern_length': pattern.get('pattern_length', 1),  # Default length
                        'signal': pattern.get('signal', 'Neutral'),  # Default signal
                        'start_time': pattern.get('start_time', ''),  # Add start_time
                        'end_time': pattern.get('end_time', '')  # Add end_time
                    }
                    all_patterns.append(pattern_obj)
                    
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
        
        return all_patterns

# =============================================================================
# MAIN GUI HELPER FUNCTIONS - Easy import for app.py
# =============================================================================

def load_and_filter_patterns(
    folder_path: str,
    selected_symbols: Optional[Set[str]] = None,
    selected_timeframes: Optional[Set[str]] = None,
    candlestick_only: bool = False,
    min_confidence: float = 0.0
) -> List[Dict[str, Any]]:
    """Main function to load and filter patterns for GUI"""
    return PatternLoader.load_patterns_from_folder(
        folder_path, selected_symbols, selected_timeframes,
        candlestick_only, min_confidence
    )

def sort_patterns_by_priority(patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Main function to sort patterns by priority for GUI"""
    return PatternLoader.sort_patterns_by_priority(patterns)

def get_pattern_statistics(patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Main function to get pattern statistics for GUI"""
    return PatternUIHelper.get_pattern_statistics(patterns)

def format_status_message(stats: Dict[str, Any], candlestick_only: bool) -> str:
    """Main function to format status message for GUI"""
    return PatternUIHelper.format_status_message(stats, candlestick_only)

def is_candlestick_pattern(pattern_name: str) -> bool:
    """Main function to check if pattern is candlestick for GUI"""
    return PatternLoader.is_candlestick_pattern(pattern_name)

def load_price_patterns_from_folder(
    folder_path: str,
    selected_symbols: Optional[Set[str]] = None,
    selected_timeframes: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    """Main function to load price patterns for GUI"""
    return PricePatternLoader.load_price_patterns_from_folder(
        folder_path, selected_symbols, selected_timeframes
    )

def infer_pattern_length(pattern_name: str) -> int:
    """
    Suy luận độ dài pattern từ tên pattern
    Pattern dài hơn được ưu tiên cao hơn
    """
    name = pattern_name.lower()
    
    # 5-candle patterns (ưu tiên cao nhất)
    if any(keyword in name for keyword in ['five', 'mat_hold', 'rising_three_methods', 'falling_three_methods', 'ladder']):
        return 5
    
    # 4-candle patterns
    if any(keyword in name for keyword in ['four', 'concealing_baby_swallow']):
        return 4
    
    # 3-candle patterns
    if any(keyword in name for keyword in ['three', 'morning_star', 'evening_star', 'abandoned_baby', 'tristar', 
                                          'white_soldiers', 'black_crows', 'inside_up', 'inside_down', 
                                          'outside_up', 'outside_down', 'advance_block', 'deliberation']):
        return 3
    
    # 2-candle patterns
    if any(keyword in name for keyword in ['two', 'engulfing', 'harami', 'piercing', 'dark_cloud', 
                                          'tweezer', 'kicker', 'counterattack', 'on_neck', 'in_neck',
                                          'thrusting', 'separating_lines', 'matching']):
        return 2
    
    # 1-candle patterns (ưu tiên thấp nhất)
    return 1

# Cập nhật hàm analyze_patterns để sử dụng logic mới
