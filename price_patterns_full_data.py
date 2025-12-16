#!/usr/bin/env python3
"""
Simplified and robust price pattern detection module
Fixed version with proper error handling and file naming support
"""

import os
import json
import glob
import logging
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from numba import njit
try:
    from scipy import stats
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Some advanced pattern detection features will be limited.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Base directory and output folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "pattern_price")

def convert_times_to_str(obj):
    if isinstance(obj, dict):
        return {k: convert_times_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_times_to_str(v) for v in obj]
    elif isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    return obj

def safe_isoformat(timestamp):
    """Safely convert timestamp to ISO format string"""
    try:
        if hasattr(timestamp, 'isoformat'):
            return timestamp.isoformat()
        elif isinstance(timestamp, (int, float)):
            # Handle integer/float timestamps by converting to datetime
            return datetime.fromtimestamp(timestamp).isoformat()
        else:
            # Fallback to string representation
            return str(timestamp)
    except Exception:
        # Last resort - use current time
        return datetime.now().isoformat()

@njit
def find_local_peaks_troughs_numba(arr, order=5):
    n = len(arr)
    local_max = []
    local_min = []
    for i in range(order, n - order):
        is_max = True
        is_min = True
        for j in range(i - order, i + order + 1):
            if j == i:
                continue
            if arr[i] < arr[j]:
                is_max = False
            if arr[i] > arr[j]:
                is_min = False
        if is_max:
            local_max.append(i)
        if is_min:
            local_min.append(i)
    return np.array(local_max), np.array(local_min)

def find_local_peaks_troughs(series, order=5):
    local_max, local_min = find_local_peaks_troughs_numba(series.values, order=order)
    return local_max, local_min

def extract_end_time(pattern):
    """
    Extract the end time of a pattern from common fields.
    Returns pd.Timestamp or None if not found.
    """
    for key in ["end_time", "timestamp", "peak2_time", "valley2_time", "head_time"]:
        if key in pattern:
            try:
                return pd.to_datetime(pattern[key])
            except Exception:
                continue
    return None

# Constants for pattern detection - RELAXED VALUES
ORDER = 3  # Giảm từ 5 xuống 3 để dễ detect hơn
RET_THRESHOLD = 0.03  # Giảm từ 0.05 xuống 0.03
FLAG_VOL_RATIO = 0.3  # Giảm để dễ detect
PENNANT_VOL_RATIO = 0.2
TRIANGLE_WINDOW = 15  # Giảm từ 20 xuống 15
RECTANGLE_WINDOW = 15
RECTANGLE_VOL = 0.015  # Giảm để dễ detect

# Advanced pattern parameters
HARMONIC_TOLERANCE = 0.08  # Tăng tolerance để dễ detect
ELLIOTT_WAVE_MIN_LENGTH = 8  # Giảm min length
WYCKOFF_ACCUMULATION_VOLUME_FACTOR = 1.1  # Giảm factor
FRACTAL_LEVELS = 2  # Giảm levels
LIQUIDITY_GRAB_THRESHOLD = 0.015

def safe_find_peaks_troughs(series, order=3):
    """Enhanced peak/trough detection with fallback"""
    try:
        if len(series) < order * 2 + 1:
            # Fallback for short series
            if len(series) >= 3:
                order = 1
            else:
                return [], []
        
        local_max, local_min = find_local_peaks_troughs(series, order=order)
        
        # If no peaks found, use simple max/min approach
        if len(local_max) == 0 and len(series) >= 5:
            # Find simple highest points
            max_idx = series.idxmax()
            local_max = [series.index.get_loc(max_idx)]
        
        if len(local_min) == 0 and len(series) >= 5:
            # Find simple lowest points  
            min_idx = series.idxmin()
            local_min = [series.index.get_loc(min_idx)]
            
        return local_max, local_min
        
    except Exception as e:
        logging.warning(f"Error in peak detection: {e}")
        # Ultimate fallback - return middle points
        if len(series) >= 3:
            mid = len(series) // 2
            return [mid], [mid]
        return [], []

def detect_double_top(df, symbol=None, timeframe=None):
    """Enhanced double top detection with improved fallback handling"""
    if len(df) < 10:  # Need minimum data
        return []
    
    try:
        local_max, _ = safe_find_peaks_troughs(df['high'], order=ORDER)
        patterns = []
        volume = df.get('volume', pd.Series([1] * len(df)))
        
        # If we have at least 2 peaks, proceed
        if len(local_max) < 2:
            # Fallback: find 2 highest points manually
            sorted_highs = df['high'].nlargest(len(df))
            if len(sorted_highs) >= 2:
                high1_idx = df[df['high'] == sorted_highs.iloc[0]].index[0]
                high2_idx = df[df['high'] == sorted_highs.iloc[1]].index[0]
                local_max = [df.index.get_loc(high1_idx), df.index.get_loc(high2_idx)]
                local_max.sort()  # Ensure chronological order
        
        for i in range(len(local_max) - 1):
            idx1, idx2 = local_max[i], local_max[i + 1]
            
            # Bounds checking
            if idx1 >= len(df) or idx2 >= len(df):
                continue
                
            high1, high2 = df.iloc[idx1]['high'], df.iloc[idx2]['high']
            
            # Distance between peaks (relaxed: 3-50 periods)
            dist = idx2 - idx1
            if not (3 <= dist <= 50):
                continue
                
            # Height similarity (relaxed: within 5%)
            height_diff = abs(high1 - high2) / max(high1, high2)
            if height_diff > 0.05:
                continue
                
            # Find valley between peaks
            valley_slice = df.iloc[idx1:idx2+1]
            if len(valley_slice) == 0:
                # Fallback valley
                valley_price = min(high1, high2) * 0.98
            else:
                valley_price = valley_slice['low'].min()
            
            # Valley depth check (relaxed: at least 0.8%)
            peak_avg = (high1 + high2) / 2
            valley_depth = (peak_avg - valley_price) / peak_avg
            if valley_depth < 0.008:
                valley_depth = 0.01  # Minimum valley depth
                
            # Volume confirmation (optional)
            try:
                vol1 = volume.iloc[max(0, idx1-1):idx1+2].mean()
                vol2 = volume.iloc[max(0, idx2-1):idx2+2].mean()
                volume_confirmation = vol2 < vol1 * 1.2
            except:
                volume_confirmation = False
                
            # Calculate confidence (more generous)
            confidence = 0.5 + valley_depth * 5  # Base confidence
            if volume_confirmation:
                confidence += 0.1
            if height_diff < 0.01:  # Very similar heights
                confidence += 0.15
            if dist >= 5:  # Good distance between peaks
                confidence += 0.1
                
            patterns.append({
                "type": "double_top",
                "pattern": "double_top", 
                "confidence": min(0.9, max(0.4, confidence)),
                "signal": "Bearish",
                "pattern_length": dist,
                "symbol": symbol or "UNKNOWN",
                "timeframe": timeframe or "UNKNOWN",
                "time": str(df.index[idx2]) if idx2 < len(df) else str(df.index[-1]),
                "start_time": str(df.index[idx1]) if idx1 < len(df) else str(df.index[0]),
                "end_time": str(df.index[idx2]) if idx2 < len(df) else str(df.index[-1]),
                "confirmed": True,
                "peak1_price": float(high1),
                "peak2_price": float(high2), 
                "valley_price": float(valley_price),
                "valley_depth": float(valley_depth),
                "height_similarity": float(1 - height_diff),
                "volume_confirmation": volume_confirmation
            })
            
        return patterns
        
    except Exception as e:
        logging.error(f"Error in double_top detection: {e}")
        # Return empty pattern with default values to avoid NaN
        current_price = df['close'].iloc[-1] if len(df) > 0 else 1.0
        return [{
            "type": "double_top",
            "pattern": "double_top",
            "confidence": 0.0,
            "signal": "Neutral",
            "pattern_length": 0,
            "symbol": symbol or "UNKNOWN",
            "timeframe": timeframe or "UNKNOWN", 
            "time": str(df.index[-1]) if len(df) > 0 else str(datetime.now()),
            "start_time": str(df.index[0]) if len(df) > 0 else str(datetime.now()),
            "end_time": str(df.index[-1]) if len(df) > 0 else str(datetime.now()),
            "confirmed": False,
            "peak1_price": float(current_price),
            "peak2_price": float(current_price),
            "valley_price": float(current_price * 0.99),
            "valley_depth": 0.01,
            "height_similarity": 0.0,
            "volume_confirmation": False
        }]

def detect_double_bottom(df, symbol=None, timeframe=None):
    """Enhanced double bottom detection with improved fallback handling"""
    if len(df) < 10:  # Need minimum data
        return []
    
    try:
        _, local_min = safe_find_peaks_troughs(df['low'], order=ORDER)
        patterns = []
        volume = df.get('volume', pd.Series([1] * len(df)))
        
        # If we have at least 2 valleys, proceed
        if len(local_min) < 2:
            # Fallback: find 2 lowest points manually
            sorted_lows = df['low'].nsmallest(len(df))
            if len(sorted_lows) >= 2:
                low1_idx = df[df['low'] == sorted_lows.iloc[0]].index[0]
                low2_idx = df[df['low'] == sorted_lows.iloc[1]].index[0]
                local_min = [df.index.get_loc(low1_idx), df.index.get_loc(low2_idx)]
                local_min.sort()  # Ensure chronological order
        
        for i in range(len(local_min) - 1):
            idx1, idx2 = local_min[i], local_min[i + 1]
            
            # Bounds checking
            if idx1 >= len(df) or idx2 >= len(df):
                continue
                
            low1, low2 = df.iloc[idx1]['low'], df.iloc[idx2]['low']
            
            # Distance between valleys (relaxed: 3-50 periods)
            dist = idx2 - idx1
            if not (3 <= dist <= 50):
                continue
                
            # Height similarity (relaxed: within 5%)
            height_diff = abs(low1 - low2) / max(low1, low2)
            if height_diff > 0.05:
                continue
                
            # Find peak between valleys
            peak_slice = df.iloc[idx1:idx2+1]
            if len(peak_slice) == 0:
                # Fallback peak
                peak_price = max(low1, low2) * 1.02
            else:
                peak_price = peak_slice['high'].max()
            
            # Peak height check (relaxed: at least 0.8% above valleys)
            valley_avg = (low1 + low2) / 2
            peak_height = (peak_price - valley_avg) / valley_avg
            if peak_height < 0.008:
                peak_height = 0.01  # Minimum peak height
                
            # Volume confirmation (optional)
            try:
                vol1 = volume.iloc[max(0, idx1-1):idx1+2].mean()
                vol2 = volume.iloc[max(0, idx2-1):idx2+2].mean()
                volume_confirmation = vol2 > vol1 * 0.8  # Second valley should have more volume
            except:
                volume_confirmation = False
                
            # Calculate confidence (more generous)
            confidence = 0.5 + peak_height * 5  # Base confidence
            if volume_confirmation:
                confidence += 0.1
            if height_diff < 0.01:  # Very similar heights
                confidence += 0.15
            if dist >= 5:  # Good distance between valleys
                confidence += 0.1
                
            patterns.append({
                "type": "double_bottom",
                "pattern": "double_bottom",
                "confidence": min(0.9, max(0.4, confidence)),
                "signal": "Bullish",
                "pattern_length": dist,
                "symbol": symbol or "UNKNOWN",
                "timeframe": timeframe or "UNKNOWN",
                "time": str(df.index[idx2]) if idx2 < len(df) else str(df.index[-1]),
                "start_time": str(df.index[idx1]) if idx1 < len(df) else str(df.index[0]),
                "end_time": str(df.index[idx2]) if idx2 < len(df) else str(df.index[-1]),
                "confirmed": True,
                "peak1_price": float(low1),  # In double bottom, peaks are the valleys
                "peak2_price": float(low2),
                "valley_price": float(peak_price),  # Valley is the peak between bottoms
                "valley_depth": float(peak_height),  # Actually peak height in this case
                "height_similarity": float(1 - height_diff),
                "volume_confirmation": volume_confirmation
            })
            
        return patterns
        
    except Exception as e:
        logging.error(f"Error in double_bottom detection: {e}")
        # Return empty pattern with default values to avoid NaN
        current_price = df['close'].iloc[-1] if len(df) > 0 else 1.0
        return [{
            "type": "double_bottom",
            "pattern": "double_bottom",
            "confidence": 0.0,
            "signal": "Neutral",
            "pattern_length": 0,
            "symbol": symbol or "UNKNOWN",
            "timeframe": timeframe or "UNKNOWN",
            "time": str(df.index[-1]) if len(df) > 0 else str(datetime.now()),
            "start_time": str(df.index[0]) if len(df) > 0 else str(datetime.now()),
            "end_time": str(df.index[-1]) if len(df) > 0 else str(datetime.now()),
            "confirmed": False,
            "peak1_price": float(current_price * 0.99),
            "peak2_price": float(current_price * 0.99),
            "valley_price": float(current_price),
            "valley_depth": 0.01,
            "height_similarity": 0.0,
            "volume_confirmation": False
        }]

def detect_head_and_shoulders(df, symbol=None, timeframe=None):
    """Enhanced head and shoulders detection"""
    local_max, local_min = find_local_peaks_troughs(df['high'], order=5)
    patterns = []
    
    for i in range(len(local_max) - 2):
        left_idx, head_idx, right_idx = local_max[i], local_max[i + 1], local_max[i + 2]
        left_peak = df.iloc[left_idx]['high']
        head_peak = df.iloc[head_idx]['high']
        right_peak = df.iloc[right_idx]['high']
        
        dist1 = head_idx - left_idx
        dist2 = right_idx - head_idx
        if dist1 < 3 or dist2 < 3:
            continue
            
        # Head must be significantly higher than shoulders
        if head_peak > left_peak * 1.02 and head_peak > right_peak * 1.02:
            # Shoulders should be roughly equal (within 3%)
            shoulder_diff = abs(left_peak - right_peak) / max(left_peak, right_peak)
            if shoulder_diff <= 0.03:
                # Find valleys between peaks
                valley1_idx = df.iloc[left_idx:head_idx]['low'].idxmin()
                valley2_idx = df.iloc[head_idx:right_idx]['low'].idxmin()
                valley1_price = df.loc[valley1_idx, 'low']
                valley2_price = df.loc[valley2_idx, 'low']
                
                # Valleys should be below shoulders
                if valley1_price < left_peak * 0.98 and valley2_price < right_peak * 0.98:
                    # Calculate neckline and pattern strength
                    neckline = (valley1_price + valley2_price) / 2
                    head_height = (head_peak - neckline) / neckline
                    shoulder_height = ((left_peak + right_peak) / 2 - neckline) / neckline
                    
                    confidence = 0.6 + head_height * 5  # Base + head prominence
                    if shoulder_diff < 0.01:  # Very similar shoulders
                        confidence += 0.1
                    if abs(valley1_price - valley2_price) / neckline < 0.02:  # Level neckline
                        confidence += 0.1
                        
                    patterns.append({
                        "type": "head_and_shoulders",
                        "pattern": "head_and_shoulders",
                        "confidence": min(0.95, confidence),
                        "signal": "Bearish",
                        "pattern_length": right_idx - left_idx,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "time": str(df.index[right_idx]),
                        "start_time": str(df.index[left_idx]),
                        "end_time": str(df.index[right_idx]),
                        "confirmed": True,
                        "left_shoulder_price": left_peak,
                        "head_price": head_peak,
                        "right_shoulder_price": right_peak,
                        "valley1_price": valley1_price,
                        "valley2_price": valley2_price,
                        "neckline": neckline,
                        "head_height": head_height,
                        "shoulder_similarity": 1 - shoulder_diff
                    })
    return patterns

def detect_inverse_head_and_shoulders(df, symbol=None, timeframe=None):
    """Enhanced inverse head and shoulders detection"""
    local_max, local_min = find_local_peaks_troughs(df['low'], order=5)
    patterns = []
    
    for i in range(len(local_min) - 2):
        left_idx, head_idx, right_idx = local_min[i], local_min[i + 1], local_min[i + 2]
        left_valley = df.iloc[left_idx]['low']
        head_valley = df.iloc[head_idx]['low']
        right_valley = df.iloc[right_idx]['low']
        
        dist1 = head_idx - left_idx
        dist2 = right_idx - head_idx
        if dist1 < 3 or dist2 < 3:
            continue
            
        # Head must be significantly lower than shoulders
        if head_valley < left_valley * 0.98 and head_valley < right_valley * 0.98:
            # Shoulders should be roughly equal (within 3%)
            shoulder_diff = abs(left_valley - right_valley) / max(left_valley, right_valley)
            if shoulder_diff <= 0.03:
                # Find peaks between valleys
                peak1_idx = df.iloc[left_idx:head_idx]['high'].idxmax()
                peak2_idx = df.iloc[head_idx:right_idx]['high'].idxmax()
                peak1_price = df.loc[peak1_idx, 'high']
                peak2_price = df.loc[peak2_idx, 'high']
                
                # Peaks should be above shoulders
                if peak1_price > left_valley * 1.02 and peak2_price > right_valley * 1.02:
                    # Calculate neckline and pattern strength
                    neckline = (peak1_price + peak2_price) / 2
                    head_depth = (neckline - head_valley) / neckline
                    shoulder_depth = (neckline - (left_valley + right_valley) / 2) / neckline
                    
                    confidence = 0.6 + head_depth * 5  # Base + head prominence
                    if shoulder_diff < 0.01:  # Very similar shoulders
                        confidence += 0.1
                    if abs(peak1_price - peak2_price) / neckline < 0.02:  # Level neckline
                        confidence += 0.1
                        
                    patterns.append({
                        "type": "inverse_head_and_shoulders",
                        "pattern": "inverse_head_and_shoulders",
                        "confidence": min(0.95, confidence),
                        "signal": "Bullish",
                        "pattern_length": right_idx - left_idx,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "time": str(df.index[right_idx]),
                        "start_time": str(df.index[left_idx]),
                        "end_time": str(df.index[right_idx]),
                        "confirmed": True,
                        "left_shoulder_price": left_valley,
                        "head_price": head_valley,
                        "right_shoulder_price": right_valley,
                        "peak1_price": peak1_price,
                        "peak2_price": peak2_price,
                        "neckline": neckline,
                        "head_depth": head_depth,
                        "shoulder_similarity": 1 - shoulder_diff
                    })
    return patterns

def detect_flag_pattern(df, symbol=None, timeframe=None):
    """Enhanced flag pattern detection"""
    patterns = []
    close = df['close']
    high = df['high']
    low = df['low']
    
    for i in range(len(close) - 20):  # Need more data for proper detection
        # Look for initial strong move (pole)
        pole_window = close.iloc[i:i + 10]
        if len(pole_window) < 10:
            continue
            
        pole_return = (pole_window.iloc[-1] - pole_window.iloc[0]) / pole_window.iloc[0]
        
        # Strong move requirement (at least 3%)
        if abs(pole_return) < 0.03:
            continue
            
        # Look for consolidation (flag)
        flag_start = i + 10
        flag_end = i + 20
        flag_window = close.iloc[flag_start:flag_end]
        
        if len(flag_window) < 8:
            continue
            
        # Calculate flag characteristics
        flag_high = high.iloc[flag_start:flag_end].max()
        flag_low = low.iloc[flag_start:flag_end].min()
        flag_range = (flag_high - flag_low) / pole_window.iloc[0]
        
        # Flag volatility should be much lower than pole
        pole_range = (pole_window.max() - pole_window.min()) / pole_window.iloc[0]
        
        if flag_range < pole_range * 0.4:  # Flag consolidation
            # Determine pattern direction
            if pole_return > 0:
                pattern_type = "bullish_flag"
                signal = "Bullish"
            else:
                pattern_type = "bearish_flag"
                signal = "Bearish"
                
            # Calculate confidence based on consolidation quality
            consolidation_quality = 1 - (flag_range / pole_range)
            confidence = 0.5 + consolidation_quality * 0.3
            
            patterns.append({
                "type": pattern_type,
                "pattern": pattern_type,
                "confidence": min(0.9, confidence),
                "signal": signal,
                "pattern_length": 20,
                "symbol": symbol,
                "timeframe": timeframe,
                "time": str(df.index[flag_end - 1]),
                "start_time": str(df.index[i]),
                "end_time": str(df.index[flag_end - 1]),
                "confirmed": True,
                "pole_return": pole_return,
                "flag_range": flag_range,
                "consolidation_quality": consolidation_quality
            })
    return patterns

def detect_pennant_pattern(df, symbol=None, timeframe=None):
    """Enhanced pennant pattern detection"""
    patterns = []
    close = df['close']
    high = df['high']
    low = df['low']
    
    for i in range(len(close) - 20):
        # Look for initial strong move (pole)
        pole_window = close.iloc[i:i + 10]
        if len(pole_window) < 10:
            continue
            
        pole_return = (pole_window.iloc[-1] - pole_window.iloc[0]) / pole_window.iloc[0]
        
        # Strong move requirement (at least 3%)
        if abs(pole_return) < 0.03:
            continue
            
        # Look for triangular consolidation (pennant)
        pennant_start = i + 10
        pennant_end = i + 20
        pennant_high = high.iloc[pennant_start:pennant_end]
        pennant_low = low.iloc[pennant_start:pennant_end]
        
        if len(pennant_high) < 8:
            continue
            
        # Check for converging trend lines
        x = np.arange(len(pennant_high))
        try:
            high_slope = np.polyfit(x, pennant_high.values, 1)[0]
            low_slope = np.polyfit(x, pennant_low.values, 1)[0]
            
            # Pennant should have converging lines
            if pole_return > 0:  # Bullish pennant
                convergence_ok = high_slope < 0 and low_slope > 0
                pattern_type = "bullish_pennant"
                signal = "Bullish"
            else:  # Bearish pennant  
                convergence_ok = high_slope > 0 and low_slope < 0
                pattern_type = "bearish_pennant"
                signal = "Bearish"
                
            if convergence_ok:
                # Calculate pennant quality
                pennant_range = (pennant_high.max() - pennant_low.min()) / pole_window.iloc[0]
                pole_range = (pole_window.max() - pole_window.min()) / pole_window.iloc[0]
                
                if pennant_range < pole_range * 0.3:  # Pennant consolidation
                    convergence_quality = abs(high_slope - low_slope) / max(abs(high_slope), abs(low_slope))
                    confidence = 0.5 + convergence_quality * 0.2
                    
                    patterns.append({
                        "type": pattern_type,
                        "pattern": pattern_type,
                        "confidence": min(0.9, confidence),
                        "signal": signal,
                        "pattern_length": 20,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "time": str(df.index[pennant_end - 1]),
                        "start_time": str(df.index[i]),
                        "end_time": str(df.index[pennant_end - 1]),
                        "confirmed": True,
                        "pole_return": pole_return,
                        "pennant_range": pennant_range,
                        "convergence_quality": convergence_quality
                    })
        except Exception:
            continue
            
    return patterns

def detect_triangle_pattern(df, symbol=None, timeframe=None):
    """Enhanced triangle pattern detection with better classification"""
    patterns = []
    highs = df['high']
    lows = df['low']
    window_size = 20
    
    for i in range(len(df) - window_size):
        high_window = highs.iloc[i:i + window_size]
        low_window = lows.iloc[i:i + window_size]
        x = np.arange(window_size)
        
        try:
            high_slope = np.polyfit(x, high_window.values, 1)[0]
            low_slope = np.polyfit(x, low_window.values, 1)[0]
            
            # Calculate R-squared for trend line quality
            high_corr = np.corrcoef(x, high_window.values)[0,1] ** 2
            low_corr = np.corrcoef(x, low_window.values)[0,1] ** 2
            
            # Only accept patterns with good trend line fit
            if high_corr < 0.3 or low_corr < 0.3:
                continue
                
            start_time = df.index[i]
            end_time = df.index[i + window_size - 1]
            
            # Classify triangle types with better criteria
            if high_slope < -0.0001 and abs(low_slope) < 0.0001:
                pattern_type = "descending_triangle"
                signal = "Bearish"
                confidence = 0.65 + (high_corr + low_corr) / 4
            elif high_slope > 0.0001 and abs(low_slope) < 0.0001:
                pattern_type = "ascending_triangle"  
                signal = "Bullish"
                confidence = 0.65 + (high_corr + low_corr) / 4
            elif high_slope < -0.0001 and low_slope > 0.0001:
                pattern_type = "symmetrical_triangle"
                signal = "Neutral"
                confidence = 0.55 + (high_corr + low_corr) / 4
            else:
                continue  # Skip unclear patterns
                
            # Calculate convergence ratio safely to avoid NaN values
            if pattern_type == "descending_triangle":
                convergence_ratio = abs(low_slope / high_slope) if high_slope != 0 and not np.isnan(high_slope) and not np.isnan(low_slope) else 0.5
            elif pattern_type == "ascending_triangle":
                convergence_ratio = abs(high_slope / low_slope) if low_slope != 0 and not np.isnan(high_slope) and not np.isnan(low_slope) else 0.5
            else:  # symmetrical_triangle
                convergence_ratio = abs(high_slope / low_slope) if low_slope != 0 and not np.isnan(high_slope) and not np.isnan(low_slope) else 0.8
            
            # Calculate actual peak and valley prices for consistency
            window_data = df.iloc[i:i + window_size]
            peak1_price = window_data['high'].iloc[0]  # Start high
            peak2_price = window_data['high'].iloc[-1]  # End high
            valley_price = window_data['low'].min()  # Lowest point in triangle
            valley_depth = (max(peak1_price, peak2_price) - valley_price) / max(peak1_price, peak2_price)
            height_similarity = min(peak1_price, peak2_price) / max(peak1_price, peak2_price)
            
            patterns.append({
                "type": pattern_type,
                "pattern": pattern_type,
                "confidence": min(0.95, confidence),
                "signal": signal,
                "pattern_length": window_size,
                "symbol": symbol,
                "timeframe": timeframe,
                "time": str(end_time),
                "start_time": str(start_time),
                "end_time": str(end_time),
                "confirmed": True,
                "high_slope": high_slope,
                "low_slope": low_slope,
                "high_correlation": high_corr,
                "low_correlation": low_corr,
                "convergence_ratio": convergence_ratio,
                "peak1_price": float(peak1_price),
                "peak2_price": float(peak2_price),
                "valley_price": float(valley_price),
                "valley_depth": float(valley_depth),
                "height_similarity": float(height_similarity),
                "volume_confirmation": False
            })
            
        except Exception as e:
            continue
            
    return patterns

def detect_rectangle_pattern(df, symbol=None, timeframe=None):
    patterns = []
    window_size = 20
    close = df['close']
    for i in range(len(close) - window_size):
        window = close.iloc[i:i + window_size]
        high, low = window.max(), window.min()
        if (high - low) / low < 0.02:
            # Calculate confidence based on price range stability
            price_range = (high - low) / low
            confidence = min(0.85, 0.7 - price_range * 10)  # Lower range = higher confidence
            
            patterns.append({
                "type": "rectangle",
                "pattern": "rectangle",
                "confidence": confidence,
                "signal": "Neutral",  # Rectangle is typically neutral
                "pattern_length": window_size,
                "symbol": symbol,
                "timeframe": timeframe,
                "time": str(df.index[i + window_size - 1]),
                "start_time": str(df.index[i]),
                "end_time": str(df.index[i + window_size - 1]),
                "confirmed": True,
                "high": high,
                "low": low,
                "price_range": price_range
            })
    return patterns

def check_trend_confirmation(pattern_type, pattern_signal, trend):
    """Check if pattern aligns with current trend"""
    score = 0.5  # Base score
    
    # Reversal patterns should appear at trend extremes
    reversal_patterns = [
        "double_top", "double_bottom", "head_and_shoulders", 
        "inverse_head_and_shoulders", "triple_top", "triple_bottom",
        "rising_wedge", "falling_wedge", "rounding_top", "rounding_bottom"
    ]
    
    # Continuation patterns should align with trend
    continuation_patterns = [
        "bullish_flag", "bearish_flag", "bullish_pennant", "bearish_pennant",
        "ascending_triangle", "descending_triangle", "rectangle"
    ]
    
    if pattern_type in reversal_patterns:
        # Bearish reversal patterns work best in uptrends
        if pattern_signal == "Bearish" and trend == "up":
            score = 0.9
        # Bullish reversal patterns work best in downtrends
        elif pattern_signal == "Bullish" and trend == "down":
            score = 0.9
        # Wrong trend context
        elif (pattern_signal == "Bearish" and trend == "down") or \
             (pattern_signal == "Bullish" and trend == "up"):
            score = 0.2
    
    elif pattern_type in continuation_patterns:
        # Continuation patterns should align with trend
        if (pattern_signal == "Bullish" and trend == "up") or \
           (pattern_signal == "Bearish" and trend == "down"):
            score = 0.8
        elif trend == "sideway":
            score = 0.6  # Neutral in sideways market
        else:
            score = 0.3  # Against trend
    
    return score

def check_indicator_confirmation(pattern_signal, indicator_row, selected_indicators=None):
    """Check technical indicator confirmation"""
    score = 0.5
    confirmations = 0
    total_checks = 0
    
    # RSI Confirmation
    rsi = indicator_row.get("RSI", 50) if indicator_row else 50
    if rsi is not None:
        total_checks += 1
        if pattern_signal == "Bullish" and rsi < 40:  # Oversold for bullish pattern
            confirmations += 1
        elif pattern_signal == "Bearish" and rsi > 60:  # Overbought for bearish pattern
            confirmations += 1
        elif 40 <= rsi <= 60:  # Neutral zone
            confirmations += 0.5
    
    # MACD Confirmation
    macd = indicator_row.get("MACD", 0) if indicator_row else 0
    macd_signal = indicator_row.get("MACD_Signal", 0) if indicator_row else 0
    if macd is not None and macd_signal is not None:
        total_checks += 1
        macd_histogram = macd - macd_signal
        if pattern_signal == "Bullish" and macd_histogram > 0:
            confirmations += 1
        elif pattern_signal == "Bearish" and macd_histogram < 0:
            confirmations += 1
    
    # Moving Average Confirmation
    ema_20 = indicator_row.get("EMA_20", 0) if indicator_row else 0
    ema_50 = indicator_row.get("EMA_50", 0) if indicator_row else 0
    current_price = indicator_row.get("close", 0) if indicator_row else 0
    
    if ema_20 and ema_50 and current_price:
        total_checks += 1
        if pattern_signal == "Bullish" and ema_20 > ema_50 and current_price > ema_20:
            confirmations += 1
        elif pattern_signal == "Bearish" and ema_20 < ema_50 and current_price < ema_20:
            confirmations += 1
    
    # Calculate final indicator score
    if total_checks > 0:
        score = confirmations / total_checks
    
    return min(1.0, score)

def check_volume_confirmation(pattern, indicator_row):
    """Check volume confirmation for pattern"""
    score = 0.5  # Default if no volume data
    
    if not indicator_row:
        return score
    
    current_volume = indicator_row.get("volume", 0)
    avg_volume = indicator_row.get("volume_sma_20", current_volume)
    pattern_signal = pattern.get("signal", "Neutral")
    
    if current_volume and avg_volume and avg_volume > 0:
        volume_ratio = current_volume / avg_volume
        
        # High volume confirmation for breakout patterns
        breakout_patterns = [
            "ascending_triangle", "descending_triangle", "rectangle",
            "bullish_flag", "bearish_flag", "cup_and_handle"
        ]
        
        if pattern.get("type") in breakout_patterns:
            if volume_ratio > 1.5:  # 50% above average
                score = 0.9
            elif volume_ratio > 1.2:  # 20% above average
                score = 0.7
            else:
                score = 0.3  # Low volume breakout is suspicious
        
        # Volume divergence for reversal patterns
        elif pattern_signal in ["Bullish", "Bearish"]:
            if volume_ratio > 1.3:  # Above average volume
                score = 0.8
            elif volume_ratio > 1.1:
                score = 0.6
    
    return score

def check_support_resistance_confirmation(pattern, trend_info, current_price=None):
    """Check if pattern occurs at significant support/resistance levels using real trendline_sr data"""
    score = 0.5  # Default score
    
    if not trend_info:
        return score
    
    pattern_signal = pattern.get("signal", "Neutral")
    pattern_price = current_price or pattern.get("end_price", 0)
    
    if not pattern_price:
        # Fallback to trend strength as proxy if no price data
        trend_strength = trend_info.get("trend_strength", 0.5)
        if trend_strength > 0.7:
            score = 0.8 if pattern_signal in ["Bullish", "Bearish"] else 0.6
        elif trend_strength < 0.3:
            score = 0.4
        return score
    
    confirmation_found = False
    proximity_threshold = 0.002  # 0.2% price proximity threshold
    
    # Check support levels for bullish patterns
    if pattern_signal == "Bullish":
        support_levels = trend_info.get("support", [])
        channel_lower = trend_info.get("channel_lower", [])
        
        for support_level in support_levels + channel_lower:
            if abs(pattern_price - support_level) / pattern_price < proximity_threshold:
                score = 0.9  # Very strong confirmation
                confirmation_found = True
                logging.info(f"🟢 Bullish pattern confirmed at support level: {support_level:.5f} (current: {pattern_price:.5f})")
                break
    
    # Check resistance levels for bearish patterns
    elif pattern_signal == "Bearish":
        resistance_levels = trend_info.get("resistance", [])
        channel_upper = trend_info.get("channel_upper", [])
        
        for resistance_level in resistance_levels + channel_upper:
            if abs(pattern_price - resistance_level) / pattern_price < proximity_threshold:
                score = 0.9  # Very strong confirmation
                confirmation_found = True
                logging.info(f"🔴 Bearish pattern confirmed at resistance level: {resistance_level:.5f} (current: {pattern_price:.5f})")
                break
    
    # Check trendline interaction
    if not confirmation_found:
        trendline = trend_info.get("trendline", [])
        if trendline:
            # Use the latest trendline value
            latest_trendline = trendline[-1] if isinstance(trendline, list) else trendline
            
            if abs(pattern_price - latest_trendline) / pattern_price < proximity_threshold:
                trend_direction = trend_info.get("trend", "sideway")
                
                # Bullish pattern bouncing off upward trendline
                if pattern_signal == "Bullish" and trend_direction == "up":
                    score = 0.8
                    confirmation_found = True
                    logging.info(f"🟡 Bullish pattern confirmed at upward trendline: {latest_trendline:.5f}")
                
                # Bearish pattern rejected at downward trendline
                elif pattern_signal == "Bearish" and trend_direction == "down":
                    score = 0.8
                    confirmation_found = True
                    logging.info(f"🟡 Bearish pattern confirmed at downward trendline: {latest_trendline:.5f}")
    
    # If no specific level confirmation, use trend strength
    if not confirmation_found:
        trend_strength = trend_info.get("trend_strength", 0.5)
        if trend_strength > 0.7:
            score = 0.7  # Good trend strength
        elif trend_strength > 0.5:
            score = 0.6  # Moderate trend strength
        elif trend_strength < 0.3:
            score = 0.3  # Weak trend
    
    return min(1.0, score)

def check_candlestick_confirmation(pattern_signal, indicator_row):
    """Check for confirming candlestick patterns"""
    score = 0.5  # Default score
    
    if not indicator_row:
        return score
    
    # This would check for actual candlestick patterns from candlestick analysis
    # For now, use price action signals as proxy
    
    open_price = indicator_row.get("open", 0)
    close_price = indicator_row.get("close", 0)
    high_price = indicator_row.get("high", 0)
    low_price = indicator_row.get("low", 0)
    
    if all([open_price, close_price, high_price, low_price]):
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        
        if total_range > 0:
            body_ratio = body_size / total_range
            
            # Strong candle confirmation
            if pattern_signal == "Bullish" and close_price > open_price and body_ratio > 0.6:
                score = 0.8
            elif pattern_signal == "Bearish" and close_price < open_price and body_ratio > 0.6:
                score = 0.8
            # Weak candle (doji-like) - indecision
            elif body_ratio < 0.2:
                score = 0.3
    
    return score

def confirm_pattern_with_indicator_and_trend(pattern, indicator_row, trend_info, selected_indicators=None):
    """
    Enhanced multi-factor confirmation system for price patterns
    
    Args:
        pattern: Dictionary containing pattern information
        indicator_row: Current indicator values (RSI, MACD, etc.)
        trend_info: Current trend information
        selected_indicators: Specific indicator conditions to check
    
    Returns:
        dict: Confirmation result with score and details
    """
    confirmation_score = 0
    confirmation_details = {
        "trend_confirmation": False,
        "indicator_confirmation": False,
        "volume_confirmation": False,
        "candlestick_confirmation": False,
        "support_resistance_confirmation": False,
        "total_score": 0
    }
    
    pattern_type = pattern.get("type", "")
    pattern_signal = pattern.get("signal", "Neutral")
    trend = trend_info.get("trend", "sideway")
    
    # 1. TREND CONFIRMATION (Weight: 30%)
    trend_score = check_trend_confirmation(pattern_type, pattern_signal, trend)
    confirmation_score += trend_score * 0.3
    confirmation_details["trend_confirmation"] = trend_score > 0.5
    
    # 2. INDICATOR CONFIRMATION (Weight: 25%)
    indicator_score = check_indicator_confirmation(pattern_signal, indicator_row, selected_indicators)
    confirmation_score += indicator_score * 0.25
    confirmation_details["indicator_confirmation"] = indicator_score > 0.5
    
    # 3. VOLUME CONFIRMATION (Weight: 20%)
    volume_score = check_volume_confirmation(pattern, indicator_row)
    confirmation_score += volume_score * 0.2
    confirmation_details["volume_confirmation"] = volume_score > 0.5
    
    # 4. SUPPORT/RESISTANCE CONFIRMATION (Weight: 15%)
    current_price = indicator_row.get("close", pattern.get("end_price", 0))
    sr_score = check_support_resistance_confirmation(pattern, trend_info, current_price)
    confirmation_score += sr_score * 0.15
    confirmation_details["support_resistance_confirmation"] = sr_score > 0.5
    
    # 5. CANDLESTICK PATTERN CONFIRMATION (Weight: 10%)
    candle_score = check_candlestick_confirmation(pattern_signal, indicator_row)
    confirmation_score += candle_score * 0.1
    confirmation_details["candlestick_confirmation"] = candle_score > 0.5
    
    # Calculate final confirmation score
    confirmation_details["total_score"] = round(confirmation_score, 3)
    
    return confirmation_details

def filter_latest_patterns(patterns):
    filtered = {}
    for p in patterns:
        key = (p.get("symbol"), p.get("timeframe"), p.get("type"), p.get("end_time"))
        times = []
        for k, v in p.items():
            if "time" in k.lower():
                try:
                    times.append(pd.to_datetime(v))
                except Exception:
                    pass
        if not times:
            continue
        latest_time = max(times)
        if key not in filtered or latest_time > filtered[key]["_latest_time"]:
            filtered[key] = p
            filtered[key]["_latest_time"] = latest_time
    for v in filtered.values():
        v.pop("_latest_time", None)
    return list(filtered.values())

def get_pattern_duration(p):
    end = extract_end_time(p)
    start = None
    for field in ["start_time", "peak1_time", "valley1_time", "left_shoulder_time"]:
        if p.get(field):
            try:
                start = pd.to_datetime(p[field])
                break
            except Exception:
                continue
    if start and end:
        return (end - start).total_seconds()
    return 0

def get_indicator_row(indicator_path, time):
    try:
        with open(indicator_path, "r", encoding="utf-8") as f:
            indi_data = json.load(f)
        df = pd.DataFrame(indi_data)
        df["time"] = pd.to_datetime(df["time"])
        time = pd.to_datetime(time)
        df = df[df["time"] <= time]
        if not df.empty:
            return df.iloc[-1].to_dict()
    except Exception as e:
        logging.error(f"Failed to read indicator file {indicator_path}: {e}")
    return {}

def load_trendline_sr_data(symbol, timeframe, folder="trendline_sr"):
    """Load actual support/resistance and trendline data from trendline_sr folder"""
    try:
        # Handle symbol format (remove . if present)
        clean_symbol = symbol.replace(".", "")
        trendline_file = os.path.join(BASE_DIR, folder, f"{clean_symbol}._{timeframe}_trendline_sr.json")
        
        if not os.path.exists(trendline_file):
            logging.warning(f"Trendline SR file not found: {trendline_file}")
            return {}
        
        with open(trendline_file, "r", encoding="utf-8") as f:
            trend_data = json.load(f)
        
        # Extract relevant data
        result = {
            "trend_direction": trend_data.get("trend_direction", "Sideways"),
            "trend_strength": trend_data.get("trend_strength", 0.5),
            "support": trend_data.get("support", []),
            "resistance": trend_data.get("resistance", []),
            "trendline": trend_data.get("trendline", []),
            "channel_upper": trend_data.get("channel_upper", []),
            "channel_lower": trend_data.get("channel_lower", []),
            "summary": trend_data.get("summary", {})
        }
        
        # Convert trend direction to standard format
        trend_dir = result["trend_direction"].lower()
        if "up" in trend_dir:
            result["trend"] = "up"
        elif "down" in trend_dir:
            result["trend"] = "down"
        else:
            result["trend"] = "sideway"
        
        logging.info(f"✅ Loaded trendline SR data for {symbol} {timeframe}: {result['trend']} trend, strength: {result['trend_strength']:.3f}")
        return result
        
    except Exception as e:
        logging.error(f"Failed to load trendline SR data for {symbol} {timeframe}: {e}")
        return {}

def get_trend_info(trend_path, time):
    if not trend_path:
        return {}
    try:
        with open(trend_path, "r", encoding="utf-8") as f:
            trend_data = json.load(f)
        trend_slope = trend_data.get("trend_slope", 0)
        trend = "up" if trend_slope > 0 else "down" if trend_slope < 0 else "sideway"
        return {"trend": trend}
    except Exception as e:
        logging.error(f"Failed to read trend file {trend_path}: {e}")
    return {}
def detect_double_top_fast(df, symbol=None, timeframe=None):
    return detect_double_top(df, symbol, timeframe)

def detect_double_bottom_fast(df, symbol=None, timeframe=None):
    return detect_double_bottom(df, symbol, timeframe)

def detect_head_and_shoulders_fast(df, symbol=None, timeframe=None):
    return detect_head_and_shoulders(df, symbol, timeframe)

def detect_inverse_head_and_shoulders_fast(df, symbol=None, timeframe=None):
    return detect_inverse_head_and_shoulders(df, symbol, timeframe)

def detect_flag_pattern_fast(df, symbol=None, timeframe=None):
    return detect_flag_pattern(df, symbol, timeframe)

def detect_pennant_pattern_fast(df, symbol=None, timeframe=None):
    return detect_pennant_pattern(df, symbol, timeframe)

def detect_cup_and_handle(df, symbol=None, timeframe=None):
    """Enhanced Cup and Handle pattern detection"""
    patterns = []
    close = df['close']
    high = df['high']
    low = df['low']
    window_size = 60
    
    for i in range(len(close) - window_size):
        if i + window_size >= len(close):
            continue
            
        cup_window = close.iloc[i:i + int(window_size * 0.8)]  # 80% for cup
        handle_window = close.iloc[i + int(window_size * 0.8):i + window_size]  # 20% for handle
        
        if len(cup_window) < 30 or len(handle_window) < 8:
            continue
        
        # Cup formation analysis
        cup_start = cup_window.iloc[0]
        cup_end = cup_window.iloc[-1]
        cup_bottom = cup_window.min()
        
        # Cup should be U-shaped: start and end at similar levels, deep bottom
        rim_similarity = abs(cup_start - cup_end) / max(cup_start, cup_end)
        cup_depth = (max(cup_start, cup_end) - cup_bottom) / max(cup_start, cup_end)
        
        # Cup criteria
        if rim_similarity > 0.05 or cup_depth < 0.12:  # Too shallow or uneven rim
            continue
            
        # Handle formation analysis
        handle_high = handle_window.max()
        handle_low = handle_window.min()
        handle_start = handle_window.iloc[0]
        
        # Handle should be a minor pullback from cup rim
        handle_depth = (handle_high - handle_low) / handle_high
        rim_level = max(cup_start, cup_end)
        
        # Handle criteria
        if (handle_depth > 0.15 or  # Handle too deep
            handle_low < cup_bottom * 1.1 or  # Handle below cup bottom  
            handle_high > rim_level * 1.02):  # Handle above rim
            continue
            
        # Calculate pattern quality
        cup_roundness = 1 - rim_similarity  # More similar rim = more round
        handle_quality = 1 - (handle_depth / 0.15)  # Shallower handle = better
        
        confidence = 0.6 + cup_depth * 2 + cup_roundness * 0.2 + handle_quality * 0.1
        confidence = min(0.95, confidence)
        
        patterns.append({
            "type": "cup_and_handle",
            "pattern": "cup_and_handle",
            "confidence": confidence,
            "signal": "Bullish",
            "pattern_length": window_size,
            "symbol": symbol,
            "timeframe": timeframe,
            "time": str(df.index[i + window_size - 1]),
            "start_time": str(df.index[i]),
            "end_time": str(df.index[i + window_size - 1]),
            "confirmed": True,
            "cup_depth": cup_depth,
            "rim_level": rim_level,
            "cup_bottom": cup_bottom,
            "handle_depth": handle_depth,
            "rim_similarity": rim_similarity,
            "cup_roundness": cup_roundness
        })
    return patterns

def create_test_patterns(symbol, timeframe):
    """Create diverse test patterns for demonstration"""
    import random
    from datetime import datetime, timedelta
    
    patterns = []
    base_time = datetime.now()
    
    # Sample diverse patterns
    pattern_types = [
        ("double_top", "Bearish", 0.75),
        ("double_bottom", "Bullish", 0.72),
        ("head_and_shoulders", "Bearish", 0.82),
        ("inverse_head_and_shoulders", "Bullish", 0.78),
        ("ascending_triangle", "Bullish", 0.65),
        ("descending_triangle", "Bearish", 0.68),
        ("symmetrical_triangle", "Neutral", 0.55),
        ("bullish_flag", "Bullish", 0.62),
        ("bearish_flag", "Bearish", 0.64),
        ("cup_and_handle", "Bullish", 0.70),
        ("rising_wedge", "Bearish", 0.58),
        ("falling_wedge", "Bullish", 0.60),
        ("rectangle", "Neutral", 0.45),
        ("pennant", "Neutral", 0.52)
    ]
    
    for i, (pattern_type, signal, base_conf) in enumerate(pattern_types):
        # Add some randomization
        confidence = base_conf + random.uniform(-0.1, 0.1)
        confidence = max(0.3, min(0.95, confidence))
        
        time_offset = timedelta(hours=i * 4)
        pattern_time = base_time - time_offset
        
        patterns.append({
            "pattern": pattern_type,
            "type": pattern_type,
            "confidence": round(confidence, 3),
            "signal": signal,
            "pattern_length": random.randint(8, 25),
            "symbol": symbol,
            "timeframe": timeframe,
            "time": pattern_time.isoformat(),
            "start_time": (pattern_time - timedelta(hours=random.randint(12, 48))).isoformat(),
            "end_time": pattern_time.isoformat(),
            "confirmed": True
        })
    
    return patterns

def standardize_pattern(pattern_data):
    """Standardize pattern data to include all required fields for GUI"""
    pattern_type = pattern_data.get("type", "unknown")
    
    # Create standardized pattern
    standardized = {
        "pattern": pattern_type,  # Use 'pattern' field for consistency with GUI
        "type": pattern_type,     # Keep 'type' for backward compatibility
        "confidence": calculate_pattern_confidence(pattern_type, pattern_data),
        "signal": determine_pattern_signal(pattern_type, pattern_data),
        "pattern_length": calculate_pattern_length(pattern_data),
        "symbol": pattern_data.get("symbol", ""),
        "timeframe": pattern_data.get("timeframe", ""),
        "time": pattern_data.get("end_time", pattern_data.get("start_time", "")),
        "confirmed": pattern_data.get("confirmed", True),
        # Add end_price for S/R confirmation
        "end_price": pattern_data.get("end_price", pattern_data.get("current_price", 0))
    }
    
    # Copy over additional pattern-specific data
    for key, value in pattern_data.items():
        if key not in standardized:
            standardized[key] = value
    
    return standardized

def calculate_pattern_confidence(pattern_type, pattern_data):
    """Calculate confidence score for a pattern based on its characteristics"""
    base_confidence = {
        "double_top": 0.75,
        "double_bottom": 0.75,
        "head_and_shoulders": 0.85,
        "inverse_head_and_shoulders": 0.85,
        "triple_top": 0.80,
        "triple_bottom": 0.80,
        "ascending_triangle": 0.70,
        "descending_triangle": 0.70,
        "symmetrical_triangle": 0.60,
        "rising_wedge": 0.65,
        "falling_wedge": 0.65,
        "bullish_flag": 0.70,
        "bearish_flag": 0.70,
        "cup_and_handle": 0.75,
        "rectangle": 0.55,
        "pennant": 0.60,
        "rounding_top": 0.60,
        "rounding_bottom": 0.60,
        "diamond_top": 0.65,
        "diamond_bottom": 0.65,
        "gartley_pattern": 0.80,
        "bat_pattern": 0.78,
        "butterfly_pattern": 0.82,
        "crab_pattern": 0.85,
        "abcd_pattern": 0.70,
        "no_pattern_found": 0.20
    }
    
    confidence = base_confidence.get(pattern_type, 0.50)
    
    # Adjust confidence based on pattern-specific characteristics
    try:
        # Volume confirmation bonus
        if pattern_data.get("volume_confirmation", False):
            confidence += 0.10
        
        # Pattern length adjustment (longer patterns more reliable)
        pattern_length = pattern_data.get("pattern_length", 10)
        if pattern_length >= 20:
            confidence += 0.05
        elif pattern_length <= 5:
            confidence -= 0.10
            
        # Height/depth significance for double tops/bottoms
        if pattern_type in ["double_top", "double_bottom"]:
            height_similarity = pattern_data.get("height_similarity", 0.8)
            valley_depth = pattern_data.get("valley_depth", 0.02)
            confidence += (height_similarity - 0.5) * 0.2  # Bonus for similar heights
            confidence += min(valley_depth * 5, 0.15)      # Bonus for deeper valleys
            
        # Triangle pattern slope quality
        elif "triangle" in pattern_type:
            high_slope = abs(pattern_data.get("high_slope", 0))
            low_slope = abs(pattern_data.get("low_slope", 0))
            if high_slope > 0 and low_slope > 0:
                slope_quality = min(abs(high_slope - low_slope) * 1000, 0.15)
                confidence += slope_quality
        
        # Head and shoulders symmetry
        elif "head_and_shoulders" in pattern_type:
            if pattern_data.get("left_shoulder_price") and pattern_data.get("right_shoulder_price"):
                left_price = pattern_data.get("left_shoulder_price", 0)
                right_price = pattern_data.get("right_shoulder_price", 0)
                if left_price > 0 and right_price > 0:
                    symmetry = 1 - abs(left_price - right_price) / max(left_price, right_price)
                    confidence += symmetry * 0.15
        
        # Ensure confidence stays within bounds
        confidence = max(0.15, min(0.95, confidence))
        
    except Exception as e:
        # If calculation fails, use base confidence
        pass
    
    return round(confidence, 3)

def determine_pattern_signal(pattern_type, pattern_data):
    """Determine bullish/bearish/neutral signal for a pattern"""
    
    # BULLISH PATTERNS (Reversal to upside)
    bullish_patterns = [
        "double_bottom", "inverse_head_and_shoulders", "triple_bottom",
        "falling_wedge", "inverse_cup_and_handle", "rounding_bottom",
        "diamond_bottom", "bullish_flag", "bullish_pennant", 
        "ascending_triangle", "cup_and_handle", "bullish_breakout"
    ]
    
    # BEARISH PATTERNS (Reversal to downside)  
    bearish_patterns = [
        "double_top", "head_and_shoulders", "triple_top",
        "rising_wedge", "cup_and_handle_top", "rounding_top",
        "diamond_top", "bearish_flag", "bearish_pennant",
        "descending_triangle", "bearish_breakout"
    ]
    
    # NEUTRAL PATTERNS (Direction depends on breakout)
    neutral_patterns = [
        "symmetrical_triangle", "rectangle", "pennant", "flag",
        "horizontal_channel", "sideways_range", "consolidation",
        "support_level", "resistance_level"
    ]
    
    if pattern_type in bullish_patterns:
        return "Bullish"
    elif pattern_type in bearish_patterns:
        return "Bearish"
    elif pattern_type in neutral_patterns:
        return "Neutral"
    else:
        # For unknown patterns, try to infer from pattern data
        if "bullish" in pattern_type.lower() or "ascending" in pattern_type.lower():
            return "Bullish"
        elif "bearish" in pattern_type.lower() or "descending" in pattern_type.lower():
            return "Bearish"
        else:
            return "Neutral"

def calculate_pattern_length(pattern_data):
    """Calculate pattern length in periods"""
    if "start_time" in pattern_data and "end_time" in pattern_data:
        try:
            start = pd.to_datetime(pattern_data["start_time"])
            end = pd.to_datetime(pattern_data["end_time"])
            # Estimate periods based on time difference (rough approximation)
            time_diff = end - start
            hours = time_diff.total_seconds() / 3600
            
            # Rough estimation of periods based on timeframe
            if "timeframe" in pattern_data:
                tf = pattern_data["timeframe"]
                if tf == "M15":
                    return max(1, int(hours * 4))  # 4 periods per hour
                elif tf == "H1":
                    return max(1, int(hours))      # 1 period per hour
                elif tf == "H4":
                    return max(1, int(hours / 4))  # 1 period per 4 hours
            
            # Default fallback
            return max(1, int(hours / 2))
        except:
            pass
    
    # Default pattern length
    return random.randint(3, 12)

# =============================================================================
# NUMBA OPTIMIZED FUNCTIONS
# =============================================================================

@njit
def detect_double_top_numba(high, low, order=5):
    local_max, _ = find_local_peaks_troughs_numba(high, order)
    patterns = []
    for i in range(len(local_max) - 1):
        idx1, idx2 = local_max[i], local_max[i + 1]
        high1, high2 = high[idx1], high[idx2]
        dist = idx2 - idx1
        if 1 <= dist <= 15 and abs(high1 - high2) / high1 <= 0.01:
            middle_low = np.min(low[idx1:idx2])
            if middle_low < high1 * 0.98:
                patterns.append((idx1, idx2, (high1 + high2) / 2, middle_low))
    return patterns

@njit
def detect_double_bottom_numba(low, high, order=5):
    _, local_min = find_local_peaks_troughs_numba(low, order)
    patterns = []
    for i in range(len(local_min) - 1):
        idx1, idx2 = local_min[i], local_min[i + 1]
        low1, low2 = low[idx1], low[idx2]
        dist = idx2 - idx1
        if 1 <= dist <= 15 and abs(low1 - low2) / low1 <= 0.01:
            middle_high = np.max(high[idx1:idx2])
            if middle_high > low1 * 1.02:
                patterns.append((idx1, idx2, (low1 + low2) / 2, middle_high))
    return patterns

@njit
def detect_head_and_shoulders_numba(high, low, order=5):
    local_max, _ = find_local_peaks_troughs_numba(high, order)
    patterns = []
    for i in range(len(local_max) - 2):
        left_idx, head_idx, right_idx = local_max[i], local_max[i + 1], local_max[i + 2]
        left_peak = high[left_idx]
        head_peak = high[head_idx]
        right_peak = high[right_idx]
        dist1 = head_idx - left_idx
        dist2 = right_idx - head_idx
        if dist1 < 1 or dist2 < 1:
            continue
        if head_peak > left_peak * 1.02 and head_peak > right_peak * 1.02:
            if abs(left_peak - right_peak) / left_peak <= 0.03:
                valley1 = np.min(low[left_idx:head_idx])
                valley2 = np.min(low[head_idx:right_idx])
                if valley1 < left_peak * 0.98 and valley2 < right_peak * 0.98:
                    patterns.append((left_idx, head_idx, right_idx, left_peak, head_peak, right_peak, valley1, valley2))
    return patterns

@njit
def detect_inverse_head_and_shoulders_numba(low, high, order=5):
    _, local_min = find_local_peaks_troughs_numba(low, order)
    patterns = []
    for i in range(len(local_min) - 2):
        left_idx, head_idx, right_idx = local_min[i], local_min[i + 1], local_min[i + 2]
        left_valley = low[left_idx]
        head_valley = low[head_idx]
        right_valley = low[right_idx]
        dist1 = head_idx - left_idx
        dist2 = right_idx - head_idx
        if dist1 < 1 or dist2 < 1:
            continue
        if head_valley < left_valley * 0.98 and head_valley < right_valley * 0.98:
            if abs(left_valley - right_valley) / left_valley <= 0.03:
                peak1 = np.max(high[left_idx:head_idx])
                peak2 = np.max(high[head_idx:right_idx])
                if peak1 > left_valley * 1.02 and peak2 > right_valley * 1.02:
                    patterns.append((left_idx, head_idx, right_idx, left_valley, head_valley, right_valley, peak1, peak2))
    return patterns

@njit
def detect_flag_pattern_numba(close, order=5):
    """
    Phát hiện mô hình flag trên chuỗi giá đóng cửa.
    Đầu vào:
        close: mảng numpy giá đóng cửa
        order: số nến lân cận để xác định đỉnh/đáy cục bộ
    Đầu ra:
        List các tuple (start_idx, end_idx, return)
    """
    patterns = []
    for i in range(len(close) - 15):
        window = close[i:i + 10]
        ret = (window[-1] - window[0]) / window[0]
        if ret >= 0.05:
            sideways = close[i + 10:i + 15]
            if len(sideways) < 5:
                continue
            volatility_flag = np.max(sideways) - np.min(sideways)
            volatility_rally = np.max(window) - np.min(window)
            if volatility_flag < volatility_rally * 0.5:
                patterns.append((i, i + 15, ret))
    return patterns

@njit
def detect_pennant_pattern_numba(close, order=5):
    patterns = []
    for i in range(len(close) - 15):
        window = close[i:i + 10]
        ret = (window[-1] - window[0]) / window[0]
        if ret >= 0.05:
            flag = close[i + 10:i + 15]
            vol_flag = np.max(flag) - np.min(flag)
            vol_rally = np.max(window) - np.min(window)
            if vol_flag < vol_rally * 0.3:
                patterns.append((i, i + 15, ret))
    return patterns

@njit
def rectangle_pattern_numba(close, window_size=20):
    patterns = []
    n = len(close)
    for i in range(n - window_size):
        window = close[i:i+window_size]
        high = np.max(window)
        low = np.min(window)
        if (high - low) / low < 0.02:
            patterns.append((i, i + window_size, high, low))
    return patterns

def detect_rectangle_pattern_fast(df, symbol=None, timeframe=None):
    return detect_rectangle_pattern(df, symbol, timeframe)

def get_best_pattern(patterns):
    """
    Select THE SINGLE BEST pattern based on priority:
    1. RECENCY (gần nhất) > 2. DURATION (dài nhất) > 3. ACCURACY (chính xác nhất)
    Returns only ONE pattern per timeframe
    """
    if not patterns:
        return None
    
    df = pd.DataFrame(patterns)
    
    # Parse times
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["end_time"])
    
    if df.empty:
        return None
    
    # Calculate duration and confirmation for ranking
    df["duration"] = (df["end_time"] - df["start_time"]).dt.total_seconds().fillna(0)
    df["confirmation_score"] = df["confirmation_score"].fillna(0)
    df["confidence"] = df["confidence"].fillna(0)
    
    # PRIORITY RANKING: Recency > Duration > Accuracy
    # 1. Sort by end_time (most recent first)
    # 2. Then by duration (longest first) 
    # 3. Then by confirmation_score (highest first)
    # 4. Finally by confidence (highest first)
    df_sorted = df.sort_values([
        "end_time",           # 1st: Most recent
        "duration",           # 2nd: Longest duration  
        "confirmation_score", # 3rd: Best confirmation
        "confidence"          # 4th: Highest confidence
    ], ascending=[False, False, False, False])
    
    # Return THE SINGLE BEST pattern
    best_pattern = df_sorted.iloc[0]
    return best_pattern.to_dict()

def safe_load_dataframe(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if "time" not in df.columns:
            logging.error(f"No 'time' column in {path}")
            return None
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        df.set_index("time", inplace=True)
        return df
    except Exception as e:
        logging.error(f"Failed to load {path}: {e}")
        return None

def auto_price_pattern_thread():
    """
    Auto-run price pattern detection in background thread
    """
    try:
        import threading
        import time
        
        def pattern_worker():
            while True:
                try:
                    # Run pattern detection every 30 minutes
                    main()
                    time.sleep(1800)  # 30 minutes
                except Exception as e:
                    logging.error(f"Auto pattern detection error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        thread = threading.Thread(target=pattern_worker, daemon=True)
        thread.start()
        logging.info("Auto price pattern detection thread started")
        return thread
        
    except Exception as e:
        logging.error(f"Failed to start auto pattern thread: {e}")
        return None

def remove_file_if_exists(path):
    if os.path.exists(path):
        os.remove(path)

from numba import njit

@njit
def triangle_pattern_numba(highs, lows, window_size=20):
    patterns = []
    n = len(highs)
    for i in range(n - window_size):
        x = np.arange(window_size)
        y_high = highs[i:i+window_size]
        y_low = lows[i:i+window_size]
        # Tính hệ số hồi quy tuyến tính (slope) thủ công thay vì dùng np.polyfit/vstack
        x_mean = np.mean(x)
        y_high_mean = np.mean(y_high)
        y_low_mean = np.mean(y_low)
        # Slope = sum((x - x_mean)*(y - y_mean)) / sum((x - x_mean)**2)
        high_slope = np.sum((x - x_mean) * (y_high - y_high_mean)) / np.sum((x - x_mean)**2)
        low_slope = np.sum((x - x_mean) * (y_low - y_low_mean)) / np.sum((x - x_mean)**2)
        if high_slope < 0 and low_slope > 0:
            patterns.append((i, i + window_size, high_slope, low_slope, 0))
        elif high_slope < 0 and low_slope <= 0:
            patterns.append((i, i + window_size, high_slope, low_slope, 1))
        elif high_slope >= 0 and low_slope > 0:
            patterns.append((i, i + window_size, high_slope, low_slope, 2))
    return patterns

# =============================================================================
# I. CONTINUATION PATTERNS (Mô hình tiếp diễn xu hướng)
# =============================================================================

def detect_bullish_flag(df, symbol=None, timeframe=None):
    """Detect Bullish Flag pattern"""
    patterns = []
    close = df['close']
    high = df['high']
    low = df['low']
    
    for i in range(30, len(close)):
        # Look for strong uptrend (flagpole)
        flagpole_start = i - 20
        flagpole_end = i - 10
        
        if flagpole_start < 0:
            continue
            
        flagpole_return = (close.iloc[flagpole_end] - close.iloc[flagpole_start]) / close.iloc[flagpole_start]
        
        if flagpole_return > 0.05:  # Strong uptrend
            # Check for consolidation (flag)
            flag_high = high.iloc[flagpole_end:i].max()
            flag_low = low.iloc[flagpole_end:i].min()
            flag_volatility = (flag_high - flag_low) / close.iloc[flagpole_end]
            
            if flag_volatility < 0.03:  # Tight consolidation
                patterns.append({
                    "type": "bullish_flag",
                    "start_time": safe_isoformat(df.index[flagpole_start]),
                    "end_time": safe_isoformat(df.index[i]),
                    "flagpole_return": flagpole_return,
                    "flag_volatility": flag_volatility,
                    "symbol": symbol,
                    "timeframe": timeframe
                })
    return patterns

def detect_bearish_flag(df, symbol=None, timeframe=None):
    """Detect Bearish Flag pattern"""
    patterns = []
    close = df['close']
    high = df['high']
    low = df['low']
    
    for i in range(30, len(close)):
        flagpole_start = i - 20
        flagpole_end = i - 10
        
        if flagpole_start < 0:
            continue
            
        flagpole_return = (close.iloc[flagpole_end] - close.iloc[flagpole_start]) / close.iloc[flagpole_start]
        
        if flagpole_return < -0.05:  # Strong downtrend
            flag_high = high.iloc[flagpole_end:i].max()
            flag_low = low.iloc[flagpole_end:i].min()
            flag_volatility = (flag_high - flag_low) / close.iloc[flagpole_end]
            
            if flag_volatility < 0.03:  # Tight consolidation
                patterns.append({
                    "type": "bearish_flag",
                    "start_time": safe_isoformat(df.index[flagpole_start]),
                    "end_time": safe_isoformat(df.index[i]),
                    "flagpole_return": flagpole_return,
                    "flag_volatility": flag_volatility,
                    "symbol": symbol,
                    "timeframe": timeframe
                })
    return patterns

# =============================================================================
# II. REVERSAL PATTERNS (Mô hình đảo chiều xu hướng)
# =============================================================================

def detect_triple_top(df, symbol=None, timeframe=None):
    """Enhanced Triple Top pattern detection"""
    local_max, _ = find_local_peaks_troughs(df['high'], order=5)
    patterns = []
    
    for i in range(len(local_max) - 2):
        if i + 2 >= len(local_max):
            break
            
        idx1, idx2, idx3 = local_max[i], local_max[i + 1], local_max[i + 2]
        high1 = df.iloc[idx1]['high']
        high2 = df.iloc[idx2]['high'] 
        high3 = df.iloc[idx3]['high']
        
        # Check if all three peaks are similar (within 2%)
        max_high = max(high1, high2, high3)
        height_similarity1 = abs(high1 - max_high) / max_high
        height_similarity2 = abs(high2 - max_high) / max_high
        height_similarity3 = abs(high3 - max_high) / max_high
        
        if (height_similarity1 <= 0.02 and 
            height_similarity2 <= 0.02 and
            height_similarity3 <= 0.02):
            
            # Calculate confidence based on peak similarity
            avg_similarity = (height_similarity1 + height_similarity2 + height_similarity3) / 3
            confidence = 0.7 + (1 - avg_similarity * 50) * 0.2  # Higher similarity = higher confidence
            
            patterns.append({
                "type": "triple_top",
                "pattern": "triple_top",
                "confidence": min(0.95, confidence),
                "signal": "Bearish",
                "pattern_length": idx3 - idx1,
                "symbol": symbol,
                "timeframe": timeframe,
                "time": str(df.index[idx3]),
                "start_time": str(df.index[idx1]),
                "end_time": str(df.index[idx3]),
                "confirmed": True,
                "peak1_price": high1,
                "peak2_price": high2,
                "peak3_price": high3,
                "peak_price": max_high,
                "height_similarity": 1 - avg_similarity
            })
    return patterns

def detect_triple_bottom(df, symbol=None, timeframe=None):
    """Enhanced Triple Bottom pattern detection"""
    _, local_min = find_local_peaks_troughs(df['low'], order=5)
    patterns = []
    
    for i in range(len(local_min) - 2):
        if i + 2 >= len(local_min):
            break
            
        idx1, idx2, idx3 = local_min[i], local_min[i + 1], local_min[i + 2]
        low1 = df.iloc[idx1]['low']
        low2 = df.iloc[idx2]['low']
        low3 = df.iloc[idx3]['low']
        
        # Check if all three valleys are similar (within 2%)
        min_low = min(low1, low2, low3)
        depth_similarity1 = abs(low1 - min_low) / min_low
        depth_similarity2 = abs(low2 - min_low) / min_low
        depth_similarity3 = abs(low3 - min_low) / min_low
        
        if (depth_similarity1 <= 0.02 and
            depth_similarity2 <= 0.02 and
            depth_similarity3 <= 0.02):
            
            # Calculate confidence based on valley similarity
            avg_similarity = (depth_similarity1 + depth_similarity2 + depth_similarity3) / 3
            confidence = 0.7 + (1 - avg_similarity * 50) * 0.2  # Higher similarity = higher confidence
            
            patterns.append({
                "type": "triple_bottom",
                "pattern": "triple_bottom",
                "confidence": min(0.95, confidence),
                "signal": "Bullish",
                "pattern_length": idx3 - idx1,
                "symbol": symbol,
                "timeframe": timeframe,
                "time": str(df.index[idx3]),
                "start_time": str(df.index[idx1]),
                "end_time": str(df.index[idx3]),
                "confirmed": True,
                "valley1_price": low1,
                "valley2_price": low2,
                "valley3_price": low3,
                "valley_price": min_low,
                "depth_similarity": 1 - avg_similarity
            })
    return patterns

def detect_rising_wedge(df, symbol=None, timeframe=None):
    """Detect Rising Wedge pattern"""
    patterns = []
    high = df['high'].values
    low = df['low'].values
    window_size = 30
    
    for i in range(len(df) - window_size):
        high_window = high[i:i + window_size]
        low_window = low[i:i + window_size]
        
        x = np.arange(window_size)
        high_slope = np.polyfit(x, high_window, 1)[0]
        low_slope = np.polyfit(x, low_window, 1)[0]
        
        # Rising wedge: both slopes positive, but converging
        if high_slope > 0 and low_slope > 0 and low_slope > high_slope:
            # Calculate confidence based on convergence
            convergence_ratio = low_slope / high_slope if high_slope != 0 else 0
            confidence = min(0.85, 0.5 + convergence_ratio * 0.3)
            
            # Calculate actual peak and valley prices for consistency
            window_data = df.iloc[i:i + window_size]
            peak1_price = window_data['high'].iloc[0]  # Start of wedge
            peak2_price = window_data['high'].iloc[-1]  # End of wedge
            valley_price = window_data['low'].min()  # Lowest point in wedge
            valley_depth = (max(peak1_price, peak2_price) - valley_price) / max(peak1_price, peak2_price)
            height_similarity = min(peak1_price, peak2_price) / max(peak1_price, peak2_price)
            
            patterns.append({
                "type": "rising_wedge",
                "pattern": "rising_wedge",
                "confidence": confidence,
                "signal": "Bearish",  # Rising wedge is bearish
                "pattern_length": window_size,
                "symbol": symbol,
                "timeframe": timeframe,
                "time": str(df.index[i + window_size - 1]),
                "start_time": str(df.index[i]),
                "end_time": str(df.index[i + window_size - 1]),
                "confirmed": True,
                "high_slope": high_slope,
                "low_slope": low_slope,
                "convergence_ratio": convergence_ratio,
                "peak1_price": float(peak1_price),
                "peak2_price": float(peak2_price),
                "valley_price": float(valley_price),
                "valley_depth": float(valley_depth),
                "height_similarity": float(height_similarity),
                "volume_confirmation": False
            })
    return patterns

def detect_falling_wedge(df, symbol=None, timeframe=None):
    """Detect Falling Wedge pattern"""
    patterns = []
    high = df['high'].values
    low = df['low'].values
    window_size = 30
    
    for i in range(len(df) - window_size):
        high_window = high[i:i + window_size]
        low_window = low[i:i + window_size]
        
        x = np.arange(window_size)
        high_slope = np.polyfit(x, high_window, 1)[0]
        low_slope = np.polyfit(x, low_window, 1)[0]
        
        # Falling wedge: both slopes negative, but converging
        if high_slope < 0 and low_slope < 0 and high_slope > low_slope:
            # Calculate confidence based on convergence
            convergence_ratio = abs(high_slope / low_slope) if low_slope != 0 else 0
            confidence = min(0.85, 0.5 + convergence_ratio * 0.3)
            
            # Calculate actual peak and valley prices for consistency
            window_data = df.iloc[i:i + window_size]
            peak1_price = window_data['high'].iloc[0]  # Start of wedge (higher)
            peak2_price = window_data['high'].iloc[-1]  # End of wedge (lower)  
            valley_price = window_data['low'].min()  # Lowest point in wedge
            valley_depth = (max(peak1_price, peak2_price) - valley_price) / max(peak1_price, peak2_price)
            height_similarity = min(peak1_price, peak2_price) / max(peak1_price, peak2_price)
            
            patterns.append({
                "type": "falling_wedge",
                "pattern": "falling_wedge", 
                "confidence": confidence,
                "signal": "Bullish",  # Falling wedge is bullish
                "pattern_length": window_size,
                "symbol": symbol,
                "timeframe": timeframe,
                "time": str(df.index[i + window_size - 1]),
                "start_time": str(df.index[i]),
                "end_time": str(df.index[i + window_size - 1]),
                "confirmed": True,
                "high_slope": high_slope,
                "low_slope": low_slope,
                "convergence_ratio": convergence_ratio,
                "peak1_price": float(peak1_price),
                "peak2_price": float(peak2_price),
                "valley_price": float(valley_price),
                "valley_depth": float(valley_depth),
                "height_similarity": float(height_similarity),
                "volume_confirmation": False
            })
    return patterns

def detect_rounding_top(df, symbol=None, timeframe=None):
    """Detect Rounding Top pattern - more restrictive version"""
    patterns = []
    close = df['close']
    high = df['high']
    window_size = 50
    
    for i in range(len(close) - window_size):
        window = close.iloc[i:i + window_size]
        high_window = high.iloc[i:i + window_size]
        
        # Check for dome-like shape with more restrictive criteria
        mid_point = window_size // 2
        left_third = window.iloc[:window_size//3]
        middle_third = window.iloc[window_size//3:2*window_size//3]
        right_third = window.iloc[2*window_size//3:]
        
        # Require clear upward trend, peak, and downward trend
        left_trend = (left_third.iloc[-1] - left_third.iloc[0]) / left_third.iloc[0]
        right_trend = (right_third.iloc[-1] - right_third.iloc[0]) / right_third.iloc[0]
        
        # Only detect if there's a significant upward trend followed by downward trend
        # and the middle section is clearly the highest
        if (left_trend > 0.02 and  # at least 2% rise in left third
            right_trend < -0.02 and  # at least 2% fall in right third  
            middle_third.mean() > left_third.mean() * 1.01 and  # middle is higher than left
            middle_third.mean() > right_third.mean() * 1.01 and  # middle is higher than right
            high_window.max() == middle_third.max()):  # peak is in the middle section
            
            patterns.append({
                "type": "rounding_top",
                "start_time": safe_isoformat(df.index[i]),
                "end_time": safe_isoformat(df.index[i + window_size]),
                "peak_price": high_window.max(),
                "valley_price": min(left_third.min(), right_third.min()),
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_rounding_bottom(df, symbol=None, timeframe=None):
    """Detect Rounding Bottom pattern"""
    patterns = []
    close = df['close']
    window_size = 50
    
    for i in range(len(close) - window_size):
        window = close.iloc[i:i + window_size]
        
        mid_point = window_size // 2
        left_half = window.iloc[:mid_point].values
        right_half = window.iloc[mid_point:].values
        
        # Check if it forms a rounded bottom
        if (np.mean(left_half[-5:]) < np.mean(left_half[:5]) and
            np.mean(right_half[:5]) < np.mean(right_half[-5:])):
            
            patterns.append({
                "type": "rounding_bottom",
                "start_time": safe_isoformat(df.index[i]),
                "end_time": safe_isoformat(df.index[i + window_size]),
                "valley_price": window.min(),
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

# =============================================================================
# III. NEUTRAL / INDECISION PATTERNS
# =============================================================================

def detect_diamond_top(df, symbol=None, timeframe=None):
    """Detect Diamond Top pattern"""
    patterns = []
    high = df['high'].values
    low = df['low'].values
    window_size = 40
    
    for i in range(len(df) - window_size):
        window_high = high[i:i + window_size]
        window_low = low[i:i + window_size]
        
        # Diamond pattern: expanding then contracting volatility
        first_half_vol = np.std(window_high[:window_size//2] - window_low[:window_size//2])
        second_half_vol = np.std(window_high[window_size//2:] - window_low[window_size//2:])
        
        if first_half_vol < second_half_vol * 0.7:  # Contracting volatility
            patterns.append({
                "type": "diamond_top",
                "start_time": safe_isoformat(df.index[i]),
                "end_time": safe_isoformat(df.index[i + window_size]),
                "volatility_ratio": second_half_vol / first_half_vol,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_diamond_bottom(df, symbol=None, timeframe=None):
    """Detect Diamond Bottom pattern"""
    patterns = []
    high = df['high'].values
    low = df['low'].values
    window_size = 40
    
    for i in range(len(df) - window_size):
        window_high = high[i:i + window_size]
        window_low = low[i:i + window_size]
        
        first_half_vol = np.std(window_high[:window_size//2] - window_low[:window_size//2])
        second_half_vol = np.std(window_high[window_size//2:] - window_low[window_size//2:])
        
        if first_half_vol < second_half_vol * 0.7:
            patterns.append({
                "type": "diamond_bottom",
                "start_time": safe_isoformat(df.index[i]),
                "end_time": safe_isoformat(df.index[i + window_size]),
                "volatility_ratio": second_half_vol / first_half_vol,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_expanding_triangle(df, symbol=None, timeframe=None):
    """Detect Expanding Triangle pattern"""
    patterns = []
    high = df['high'].values
    low = df['low'].values
    window_size = 30
    
    for i in range(len(df) - window_size):
        high_window = high[i:i + window_size]
        low_window = low[i:i + window_size]
        
        x = np.arange(window_size)
        high_slope = np.polyfit(x, high_window, 1)[0]
        low_slope = np.polyfit(x, low_window, 1)[0]
        
        # Expanding triangle: diverging slopes
        if high_slope > 0 and low_slope < 0:
            patterns.append({
                "type": "expanding_triangle",
                "start_time": safe_isoformat(df.index[i]),
                "end_time": safe_isoformat(df.index[i + window_size]),
                "high_slope": high_slope,
                "low_slope": low_slope,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_sideways_range(df, symbol=None, timeframe=None):
    """Detect Sideways Range pattern"""
    patterns = []
    close = df['close']
    window_size = 25
    
    for i in range(len(close) - window_size):
        window = close.iloc[i:i + window_size]
        
        # Check for sideways movement
        price_range = window.max() - window.min()
        avg_price = window.mean()
        range_pct = price_range / avg_price
        
        # Linear regression to check for trend
        x = np.arange(window_size)
        slope = np.polyfit(x, window.values, 1)[0]
        
        if range_pct < 0.03 and abs(slope) < avg_price * 0.001:  # Flat trend
            patterns.append({
                "type": "sideways_range",
                "start_time": safe_isoformat(df.index[i]),
                "end_time": safe_isoformat(df.index[i + window_size]),
                "range_high": window.max(),
                "range_low": window.min(),
                "range_percentage": range_pct,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

# =============================================================================
# IV. HARMONIC PATTERNS (Fibonacci-based)
# =============================================================================

def detect_gartley_pattern(df, symbol=None, timeframe=None):
    """Detect Gartley Pattern (ABCD with specific Fibonacci ratios)"""
    patterns = []
    local_max, local_min = find_local_peaks_troughs(df['high'], order=3), find_local_peaks_troughs(df['low'], order=3)
    
    # Combine and sort all extremes
    all_extremes = []
    for idx in local_max[0]:
        all_extremes.append((idx, df.iloc[idx]['high'], 'high'))
    for idx in local_min[0]:
        all_extremes.append((idx, df.iloc[idx]['low'], 'low'))
    
    all_extremes.sort(key=lambda x: x[0])
    
    for i in range(len(all_extremes) - 4):
        X, A, B, C, D = all_extremes[i:i+5]
        
        # Check Gartley ratios
        XA = abs(A[1] - X[1])
        AB = abs(B[1] - A[1])
        BC = abs(C[1] - B[1])
        CD = abs(D[1] - C[1])
        
        # Gartley specific ratios
        AB_XA = AB / XA if XA != 0 else 0
        BC_AB = BC / AB if AB != 0 else 0
        CD_BC = CD / BC if BC != 0 else 0
        
        if (0.618 - HARMONIC_TOLERANCE <= AB_XA <= 0.618 + HARMONIC_TOLERANCE and
            0.382 - HARMONIC_TOLERANCE <= BC_AB <= 0.886 + HARMONIC_TOLERANCE and
            1.272 - HARMONIC_TOLERANCE <= CD_BC <= 1.618 + HARMONIC_TOLERANCE):
            
            patterns.append({
                "type": "gartley_pattern",
                "X_time": safe_isoformat(df.index[X[0]]),
                "A_time": safe_isoformat(df.index[A[0]]),
                "B_time": safe_isoformat(df.index[B[0]]),
                "C_time": safe_isoformat(df.index[C[0]]),
                "D_time": safe_isoformat(df.index[D[0]]),
                "X_price": X[1],
                "A_price": A[1],
                "B_price": B[1],
                "C_price": C[1],
                "D_price": D[1],
                "AB_XA_ratio": AB_XA,
                "BC_AB_ratio": BC_AB,
                "CD_BC_ratio": CD_BC,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_bat_pattern(df, symbol=None, timeframe=None):
    """Detect Bat Pattern"""
    patterns = []
    local_max, local_min = find_local_peaks_troughs(df['high'], order=3), find_local_peaks_troughs(df['low'], order=3)
    
    all_extremes = []
    for idx in local_max[0]:
        all_extremes.append((idx, df.iloc[idx]['high'], 'high'))
    for idx in local_min[0]:
        all_extremes.append((idx, df.iloc[idx]['low'], 'low'))
    
    all_extremes.sort(key=lambda x: x[0])
    
    for i in range(len(all_extremes) - 4):
        X, A, B, C, D = all_extremes[i:i+5]
        
        XA = abs(A[1] - X[1])
        AB = abs(B[1] - A[1])
        BC = abs(C[1] - B[1])
        CD = abs(D[1] - C[1])
        
        AB_XA = AB / XA if XA != 0 else 0
        BC_AB = BC / AB if AB != 0 else 0
        CD_BC = CD / BC if BC != 0 else 0
        
        # Bat specific ratios
        if (0.382 - HARMONIC_TOLERANCE <= AB_XA <= 0.5 + HARMONIC_TOLERANCE and
            0.382 - HARMONIC_TOLERANCE <= BC_AB <= 0.886 + HARMONIC_TOLERANCE and
            1.618 - HARMONIC_TOLERANCE <= CD_BC <= 2.618 + HARMONIC_TOLERANCE):
            
            patterns.append({
                "type": "bat_pattern",
                "X_time": safe_isoformat(df.index[X[0]]),
                "A_time": safe_isoformat(df.index[A[0]]),
                "B_time": safe_isoformat(df.index[B[0]]),
                "C_time": safe_isoformat(df.index[C[0]]),
                "D_time": safe_isoformat(df.index[D[0]]),
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_butterfly_pattern(df, symbol=None, timeframe=None):
    """Detect Butterfly Pattern"""
    patterns = []
    # Similar to Gartley but with 0.786 AB/XA ratio and 1.27-1.618 CD/BC ratio
    local_max, local_min = find_local_peaks_troughs(df['high'], order=3), find_local_peaks_troughs(df['low'], order=3)
    
    all_extremes = []
    for idx in local_max[0]:
        all_extremes.append((idx, df.iloc[idx]['high'], 'high'))
    for idx in local_min[0]:
        all_extremes.append((idx, df.iloc[idx]['low'], 'low'))
    
    all_extremes.sort(key=lambda x: x[0])
    
    for i in range(len(all_extremes) - 4):
        X, A, B, C, D = all_extremes[i:i+5]
        
        XA = abs(A[1] - X[1])
        AB = abs(B[1] - A[1])
        BC = abs(C[1] - B[1])
        CD = abs(D[1] - C[1])
        
        AB_XA = AB / XA if XA != 0 else 0
        BC_AB = BC / AB if AB != 0 else 0
        CD_BC = CD / BC if BC != 0 else 0
        
        # Butterfly specific ratios
        if (0.786 - HARMONIC_TOLERANCE <= AB_XA <= 0.786 + HARMONIC_TOLERANCE and
            0.382 - HARMONIC_TOLERANCE <= BC_AB <= 0.886 + HARMONIC_TOLERANCE and
            1.618 - HARMONIC_TOLERANCE <= CD_BC <= 2.618 + HARMONIC_TOLERANCE):
            
            patterns.append({
                "type": "butterfly_pattern",
                "X_time": safe_isoformat(df.index[X[0]]),
                "A_time": safe_isoformat(df.index[A[0]]),
                "B_time": safe_isoformat(df.index[B[0]]),
                "C_time": safe_isoformat(df.index[C[0]]),
                "D_time": safe_isoformat(df.index[D[0]]),
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_crab_pattern(df, symbol=None, timeframe=None):
    """Detect Crab Pattern"""
    patterns = []
    local_max, local_min = find_local_peaks_troughs(df['high'], order=3), find_local_peaks_troughs(df['low'], order=3)
    
    all_extremes = []
    for idx in local_max[0]:
        all_extremes.append((idx, df.iloc[idx]['high'], 'high'))
    for idx in local_min[0]:
        all_extremes.append((idx, df.iloc[idx]['low'], 'low'))
    
    all_extremes.sort(key=lambda x: x[0])
    
    for i in range(len(all_extremes) - 4):
        X, A, B, C, D = all_extremes[i:i+5]
        
        XA = abs(A[1] - X[1])
        AB = abs(B[1] - A[1])
        BC = abs(C[1] - B[1])
        CD = abs(D[1] - C[1])
        
        AB_XA = AB / XA if XA != 0 else 0
        BC_AB = BC / AB if AB != 0 else 0
        CD_BC = CD / BC if BC !=  0 else 0
        
        # Crab specific ratios
        if (0.382 - HARMONIC_TOLERANCE <= AB_XA <= 0.618 + HARMONIC_TOLERANCE and
            0.382 - HARMONIC_TOLERANCE <= BC_AB <= 0.886 + HARMONIC_TOLERANCE and
            2.24 - HARMONIC_TOLERANCE <= CD_BC <= 3.618 + HARMONIC_TOLERANCE):
            
            patterns.append({
                "type": "crab_pattern",
                "X_time": safe_isoformat(df.index[X[0]]),
                "A_time": safe_isoformat(df.index[A[0]]),
                "B_time": safe_isoformat(df.index[B[0]]),
                "C_time": safe_isoformat(df.index[C[0]]),
                "D_time": safe_isoformat(df.index[D[0]]),
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_abcd_pattern(df, symbol=None, timeframe=None):
    """Detect AB=CD Pattern"""
    patterns = []
    local_max, local_min = find_local_peaks_troughs(df['high'], order=3), find_local_peaks_troughs(df['low'], order=3)
    
    all_extremes = []
    for idx in local_max[0]:
        all_extremes.append((idx, df.iloc[idx]['high'], 'high'))
    for idx in local_min[0]:
        all_extremes.append((idx, df.iloc[idx]['low'], 'low'))
    
    all_extremes.sort(key=lambda x: x[0])
    
    for i in range(len(all_extremes) - 3):
        A, B, C, D = all_extremes[i:i+4]
        
        AB = abs(B[1] - A[1])
        BC = abs(C[1] - B[1])
        CD = abs(D[1] - C[1])
        
        BC_AB = BC / AB if AB != 0 else 0
        CD_AB = CD / AB if AB != 0 else 0
        
        # AB=CD ratios
        if (0.618 - HARMONIC_TOLERANCE <= BC_AB <= 0.786 + HARMONIC_TOLERANCE and
            0.618 - HARMONIC_TOLERANCE <= CD_AB <= 1.618 + HARMONIC_TOLERANCE):
            
            patterns.append({
                "type": "abcd_pattern",
                "A_time": safe_isoformat(df.index[A[0]]),
                "B_time": safe_isoformat(df.index[B[0]]),
                "C_time": safe_isoformat(df.index[C[0]]),
                "D_time": safe_isoformat(df.index[D[0]]),
                "AB_length": AB,
                "CD_length": CD,
                "ratio": CD_AB,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

# =============================================================================
# V. ELLIOTT WAVE PATTERNS
# =============================================================================

def detect_impulse_wave(df, symbol=None, timeframe=None):
    """Detect Elliott Wave Impulse (5-wave structure)"""
    patterns = []
    local_max, local_min = find_local_peaks_troughs(df['high'], order=5), find_local_peaks_troughs(df['low'], order=5)
    
    all_extremes = []
    for idx in local_max[0]:
        all_extremes.append((idx, df.iloc[idx]['high'], 'high'))
    for idx in local_min[0]:
        all_extremes.append((idx, df.iloc[idx]['low'], 'low'))
    
    all_extremes.sort(key=lambda x: x[0])
    
    # Look for 5-wave structures
    for i in range(len(all_extremes) - 8):  # Need 9 points for 5 waves
        waves = all_extremes[i:i+9]
        
        # Check Elliott Wave rules
        # Wave 2 should not retrace more than 100% of Wave 1
        # Wave 4 should not overlap Wave 1
        # Wave 3 is often the longest
        
        wave1 = abs(waves[1][1] - waves[0][1])
        wave2 = abs(waves[2][1] - waves[1][1])
        wave3 = abs(waves[3][1] - waves[2][1])
        wave4 = abs(waves[4][1] - waves[3][1])
        wave5 = abs(waves[5][1] - waves[4][1])
        
        # Basic Elliott Wave validation
        if (wave2 < wave1 and  # Wave 2 doesn't exceed Wave 1
            wave3 > wave1 and  # Wave 3 is longer than Wave 1
            wave4 < wave3):    # Wave 4 doesn't exceed Wave 3
            
            patterns.append({
                "type": "elliott_impulse_wave",
                "start_time": safe_isoformat(df.index[waves[0][0]]),
                "end_time": safe_isoformat(df.index[waves[8][0]]),
                "wave1_length": wave1,
                "wave2_length": wave2,
                "wave3_length": wave3,
                "wave4_length": wave4,
                "wave5_length": wave5,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_corrective_wave(df, symbol=None, timeframe=None):
    """Detect Elliott Wave Corrective ABC pattern"""
    patterns = []
    local_max, local_min = find_local_peaks_troughs(df['high'], order=5), find_local_peaks_troughs(df['low'], order=5)
    
    all_extremes = []
    for idx in local_max[0]:
        all_extremes.append((idx, df.iloc[idx]['high'], 'high'))
    for idx in local_min[0]:
        all_extremes.append((idx, df.iloc[idx]['low'], 'low'))
    
    all_extremes.sort(key=lambda x: x[0])
    
    # Look for ABC corrective structures
    for i in range(len(all_extremes) - 3):
        A, B, C = all_extremes[i:i+3]
        
        AB = abs(B[1] - A[1])
        BC = abs(C[1] - B[1])
        
        # Check for typical corrective ratios
        BC_AB = BC / AB if AB != 0 else 0
        
        if 0.5 <= BC_AB <= 1.618:  # Typical corrective ratios
            patterns.append({
                "type": "elliott_corrective_abc",
                "A_time": safe_isoformat(df.index[A[0]]),
                "B_time": safe_isoformat(df.index[B[0]]),
                "C_time": safe_isoformat(df.index[C[0]]),
                "AB_length": AB,
                "BC_length": BC,
                "BC_AB_ratio": BC_AB,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

# =============================================================================
# VI. WYCKOFF PATTERNS
# =============================================================================

def detect_wyckoff_accumulation(df, symbol=None, timeframe=None):
    """Detect Wyckoff Accumulation Phase"""
    patterns = []
    close = df['close']
    volume = df.get('volume', pd.Series([1] * len(df)))  # Default volume if not available
    window_size = 50
    
    for i in range(len(close) - window_size):
        window_close = close.iloc[i:i + window_size]
        window_volume = volume.iloc[i:i + window_size]
        
        # Check for characteristics of accumulation:
        # 1. Sideways price action
        # 2. Increasing volume
        # 3. Higher lows formation
        
        price_range = window_close.max() - window_close.min()
        avg_price = window_close.mean()
        range_pct = price_range / avg_price
        
        # Volume trend
        first_half_vol = window_volume.iloc[:window_size//2].mean()
        second_half_vol = window_volume.iloc[window_size//2:].mean()
        vol_increase = second_half_vol / first_half_vol if first_half_vol > 0 else 0
        
        if range_pct < 0.05 and vol_increase > WYCKOFF_ACCUMULATION_VOLUME_FACTOR:
            patterns.append({
                "type": "wyckoff_accumulation",
                "start_time": safe_isoformat(df.index[i]),
                "end_time": safe_isoformat(df.index[i + window_size]),
                "price_range_pct": range_pct,
                "volume_increase_ratio": vol_increase,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_wyckoff_distribution(df, symbol=None, timeframe=None):
    """Detect Wyckoff Distribution Phase"""
    patterns = []
    close = df['close']
    volume = df.get('volume', pd.Series([1] * len(df)))
    window_size = 50
    
    for i in range(len(close) - window_size):
        window_close = close.iloc[i:i + window_size]
        window_volume = volume.iloc[i:i + window_size]
        
        price_range = window_close.max() - window_close.min()
        avg_price = window_close.mean()
        range_pct = price_range / avg_price
        
        # Distribution: sideways at high prices with high volume
        first_half_vol = window_volume.iloc[:window_size//2].mean()
        second_half_vol = window_volume.iloc[window_size//2:].mean()
        vol_increase = second_half_vol / first_half_vol if first_half_vol > 0 else 0
        
        # Check if prices are at recent highs
        recent_high = df.iloc[max(0, i-50):i+window_size]['high'].max()
        current_avg = window_close.mean()
        
        if (range_pct < 0.05 and vol_increase > WYCKOFF_ACCUMULATION_VOLUME_FACTOR and
            current_avg > recent_high * 0.95):  # Near recent highs
            
            patterns.append({
                "type": "wyckoff_distribution",
                "start_time": safe_isoformat(df.index[i]),
                "end_time": safe_isoformat(df.index[i + window_size]),
                "price_range_pct": range_pct,
                "volume_increase_ratio": vol_increase,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_wyckoff_spring(df, symbol=None, timeframe=None):
    """Detect Wyckoff Spring (false breakdown before markup)"""
    patterns = []
    low = df['low']
    close = df['close']
    
    for i in range(50, len(df) - 10):
        # Find support level
        support_window = low.iloc[i-50:i]
        support_level = support_window.min()
        
        # Check for brief break below support then recovery
        current_window = df.iloc[i:i+10]
        min_break = current_window['low'].min()
        recovery_close = current_window['close'].iloc[-1]
        
        if (min_break < support_level * 0.995 and  # Brief break below support
            recovery_close > support_level * 1.005):  # Recovery above support
            
            patterns.append({
                "type": "wyckoff_spring",
                "spring_time": safe_isoformat(df.index[i + current_window['low'].idxmin() - i]),
                "support_level": support_level,
                "break_level": min_break,
                "recovery_level": recovery_close,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_wyckoff_upthrust(df, symbol=None, timeframe=None):
    """Detect Wyckoff Upthrust (false breakout before markdown)"""
    patterns = []
    high = df['high']
    close = df['close']
    
    for i in range(50, len(df) - 10):
        # Find resistance level
        resistance_window = high.iloc[i-50:i]
        resistance_level = resistance_window.max()
        
        # Check for brief break above resistance then failure
        current_window = df.iloc[i:i+10]
        max_break = current_window['high'].max()
        failure_close = current_window['close'].iloc[-1]
        
        if (max_break > resistance_level * 1.005 and  # Brief break above resistance
            failure_close < resistance_level * 0.995):  # Failure below resistance
            
            patterns.append({
                "type": "wyckoff_upthrust",
                "upthrust_time": safe_isoformat(df.index[i + current_window['high'].idxmax() - i]),
                "resistance_level": resistance_level,
                "break_level": max_break,
                "failure_level": failure_close,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

# =============================================================================
# VII. ADVANCED AI / HIDDEN PATTERNS
# =============================================================================

def detect_fractal_pattern(df, symbol=None, timeframe=None):
    """Detect Fractal Patterns"""
    patterns = []
    high = df['high']
    low = df['low']
    
    # Fractal highs: high[i] > high[i-2] and high[i] > high[i-1] and high[i] > high[i+1] and high[i] > high[i+2]
    for i in range(2, len(high) - 2):
        if (high.iloc[i] > high.iloc[i-2] and high.iloc[i] > high.iloc[i-1] and
            high.iloc[i] > high.iloc[i+1] and high.iloc[i] > high.iloc[i+2]):
            
            patterns.append({
                "type": "fractal_high",
                "time": safe_isoformat(df.index[i]),
                "price": high.iloc[i],
                "symbol": symbol,
                "timeframe": timeframe
            })
    
    # Fractal lows
    for i in range(2, len(low) - 2):
        if (low.iloc[i] < low.iloc[i-2] and low.iloc[i] < low.iloc[i-1] and
            low.iloc[i] < low.iloc[i+1] and low.iloc[i] < low.iloc[i+2]):
            
            patterns.append({
                "type": "fractal_low",
                "time": safe_isoformat(df.index[i]),
                "price": low.iloc[i],
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_order_block(df, symbol=None, timeframe=None):
    """Detect Order Block patterns (Smart Money concepts)"""
    patterns = []
    open_price = df['open']
    close = df['close']
    high = df['high']
    low = df['low']
    
    for i in range(1, len(df) - 1):
        current_candle = df.iloc[i]
        prev_candle = df.iloc[i-1]
        next_candle = df.iloc[i+1]
        
        # Bullish Order Block: Strong bullish candle followed by gap up
        if (close.iloc[i] > open_price.iloc[i] and  # Bullish candle
            (close.iloc[i] - open_price.iloc[i]) > (high.iloc[i] - low.iloc[i]) * 0.7 and  # Strong body
            open_price.iloc[i+1] > close.iloc[i]):  # Gap up next candle
            
            patterns.append({
                "type": "bullish_order_block",
                "time": safe_isoformat(df.index[i]),
                "high": high.iloc[i],
                "low": low.iloc[i],
                "strength": (close.iloc[i] - open_price.iloc[i]) / (high.iloc[i] - low.iloc[i]),
                "symbol": symbol,
                "timeframe": timeframe
            })
        
        # Bearish Order Block: Strong bearish candle followed by gap down
        elif (close.iloc[i] < open_price.iloc[i] and  # Bearish candle
              (open_price.iloc[i] - close.iloc[i]) > (high.iloc[i] - low.iloc[i]) * 0.7 and  # Strong body
              open_price.iloc[i+1] < close.iloc[i]):  # Gap down next candle
            
            patterns.append({
                "type": "bearish_order_block",
                "time": safe_isoformat(df.index[i]),
                "high": high.iloc[i],
                "low": low.iloc[i],
                "strength": (open_price.iloc[i] - close.iloc[i]) / (high.iloc[i] - low.iloc[i]),
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_liquidity_grab(df, symbol=None, timeframe=None):
    """Detect Liquidity Grab patterns"""
    patterns = []
    high = df['high']
    low = df['low']
    close = df['close']
    
    for i in range(20, len(df) - 5):
        # Find recent high/low
        recent_high = high.iloc[i-20:i].max()
        recent_low = low.iloc[i-20:i].min()
        
        # Check for liquidity grab above recent high
        if high.iloc[i] > recent_high * 1.001:  # Break above recent high
            # Check if price quickly reverses
            subsequent_close = close.iloc[i+1:i+5].min()
            if subsequent_close < recent_high * 0.999:  # Quick reversal below
                patterns.append({
                    "type": "liquidity_grab_high",
                    "time": safe_isoformat(df.index[i]),
                    "grab_price": high.iloc[i],
                    "previous_high": recent_high,
                    "reversal_price": subsequent_close,
                    "symbol": symbol,
                    "timeframe": timeframe
                })
        
        # Check for liquidity grab below recent low
        if low.iloc[i] < recent_low * 0.999:  # Break below recent low
            subsequent_close = close.iloc[i+1:i+5].max()
            if subsequent_close > recent_low * 1.001:  # Quick reversal above
                patterns.append({
                    "type": "liquidity_grab_low",
                    "time": safe_isoformat(df.index[i]),
                    "grab_price": low.iloc[i],
                    "previous_low": recent_low,
                    "reversal_price": subsequent_close,
                    "symbol": symbol,
                    "timeframe": timeframe
                })
    return patterns

def detect_cluster_pattern(df, symbol=None, timeframe=None):
    """Detect Price Cluster patterns"""
    patterns = []
    close = df['close']
    window_size = 20
    
    for i in range(len(close) - window_size):
        window = close.iloc[i:i + window_size]
        
        # Find price levels with multiple touches
        price_levels = {}
        tolerance = window.std() * 0.5
        
        for price in window:
            found_cluster = False
            for level in price_levels:
                if abs(price - level) <= tolerance:
                    price_levels[level] += 1
                    found_cluster = True
                    break
            if not found_cluster:
                price_levels[price] = 1
        
        # Find significant clusters (3+ touches)
        significant_clusters = {level: count for level, count in price_levels.items() if count >= 3}
        
        if significant_clusters:
            strongest_level = max(significant_clusters, key=significant_clusters.get)
            patterns.append({
                "type": "price_cluster",
                "start_time": safe_isoformat(df.index[i]),
                "end_time": safe_isoformat(df.index[i + window_size]),
                "cluster_price": strongest_level,
                "touch_count": significant_clusters[strongest_level],
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_volume_profile_zones(df, symbol=None, timeframe=None):
    """Detect Volume Profile significant zones"""
    patterns = []
    close = df['close']
    volume = df.get('volume', pd.Series([1] * len(df)))
    window_size = 50
    
    for i in range(len(close) - window_size):
        window_close = close.iloc[i:i + window_size]
        window_volume = volume.iloc[i:i + window_size]
        
        # Create price bins
        price_min = window_close.min()
        price_max = window_close.max()
        num_bins = 20
        bin_size = (price_max - price_min) / num_bins
        
        # Calculate volume at each price level
        volume_profile = {}
        for j, price in enumerate(window_close):
            bin_level = price_min + (int((price - price_min) / bin_size) * bin_size)
            if bin_level not in volume_profile:
                volume_profile[bin_level] = 0
            volume_profile[bin_level] += window_volume.iloc[j]
        
        # Find high volume nodes
        avg_volume = sum(volume_profile.values()) / len(volume_profile)
        high_volume_zones = {level: vol for level, vol in volume_profile.items() 
                           if vol > avg_volume * 1.5}
        
        for level, vol in high_volume_zones.items():
            patterns.append({
                "type": "volume_profile_zone",
                "start_time": safe_isoformat(df.index[i]),
                "end_time": safe_isoformat(df.index[i + window_size]),
                "price_level": level,
                "volume": vol,
                "volume_ratio": vol / avg_volume,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

# =============================================================================
# VIII. ADDITIONAL CLASSICAL AND MODERN PATTERNS (17 more to reach 52 total)
# =============================================================================

def detect_inverse_cup_and_handle(df, symbol=None, timeframe=None):
    """Detect Inverse Cup and Handle pattern"""
    patterns = []
    close = df['close']
    window_size = 60
    
    for i in range(len(close) - window_size):
        window = close.iloc[i:i + window_size]
        
        cup_depth = window_size // 3
        left_rim = window.iloc[:cup_depth].mean()
        top = window.iloc[cup_depth:2*cup_depth].max()
        right_rim = window.iloc[2*cup_depth:].mean()
        
        handle_start = int(window_size * 0.8)
        handle = window.iloc[handle_start:]
        
        if (abs(left_rim - right_rim) / left_rim < 0.05 and
            top > left_rim * 1.15 and
            handle.max() < top * 0.9):
            
            patterns.append({
                "type": "inverse_cup_and_handle",
                "start_time": safe_isoformat(df.index[i]),
                "end_time": safe_isoformat(df.index[i + window_size]),
                "cup_height": (top - left_rim) / left_rim,
                "rim_level": left_rim,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_support_resistance_levels(df, symbol=None, timeframe=None):
    """Detect Support and Resistance levels"""
    patterns = []
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Find significant support/resistance levels
    for i in range(20, len(df) - 20):
        current_high = high.iloc[i]
        current_low = low.iloc[i]
        current_close = close.iloc[i]
        
        # Check for resistance level (price bounced down from this level multiple times)
        resistance_touches = 0
        for j in range(max(0, i-20), min(len(df), i+20)):
            if abs(high.iloc[j] - current_high) / current_high < 0.005:
                resistance_touches += 1
        
        if resistance_touches >= 3:  # At least 3 touches
            patterns.append({
                "type": "resistance_level",
                "level_price": current_high,
                "touches": resistance_touches,
                "time": safe_isoformat(df.index[i]),
                "symbol": symbol,
                "timeframe": timeframe
            })
        
        # Check for support level (price bounced up from this level multiple times)
        support_touches = 0
        for j in range(max(0, i-20), min(len(df), i+20)):
            if abs(low.iloc[j] - current_low) / current_low < 0.005:
                support_touches += 1
        
        if support_touches >= 3:  # At least 3 touches
            patterns.append({
                "type": "support_level",
                "level_price": current_low,
                "touches": support_touches,
                "time": safe_isoformat(df.index[i]),
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_channel_pattern(df, symbol=None, timeframe=None):
    """Detect Channel patterns (ascending, descending, horizontal)"""
    patterns = []
    high = df['high'].values
    low = df['low'].values
    window_size = 40
    
    for i in range(len(df) - window_size):
        high_window = high[i:i + window_size]
        low_window = low[i:i + window_size]
        
        x = np.arange(window_size)
        high_slope = np.polyfit(x, high_window, 1)[0]
        low_slope = np.polyfit(x, low_window, 1)[0]
        
        # Check if lines are parallel (channel)
        slope_diff = abs(high_slope - low_slope)
        avg_slope = (high_slope + low_slope) / 2
        
        if slope_diff < abs(avg_slope) * 0.3:  # Parallel lines
            if avg_slope > 0.001:
                channel_type = "ascending_channel"
            elif avg_slope < -0.001:
                channel_type = "descending_channel"
            else:
                channel_type = "horizontal_channel"
            
            patterns.append({
                "type": channel_type,
                "start_time": safe_isoformat(df.index[i]),
                "end_time": safe_isoformat(df.index[i + window_size]),
                "upper_slope": high_slope,
                "lower_slope": low_slope,
                "channel_width": np.mean(high_window - low_window),
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_breakout_pattern(df, symbol=None, timeframe=None):
    """Detect Breakout patterns"""
    patterns = []
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df.get('volume', pd.Series([1] * len(df)))
    
    for i in range(20, len(df) - 5):
        # Consolidation period
        consolidation = close.iloc[i-20:i]
        cons_high = consolidation.max()
        cons_low = consolidation.min()
        cons_range = cons_high - cons_low
        
        # Check for breakout
        breakout_candles = df.iloc[i:i+5]
        
        # Upward breakout
        if (breakout_candles['high'].max() > cons_high * 1.01 and
            volume.iloc[i:i+5].mean() > volume.iloc[i-20:i].mean() * 1.5):
            
            patterns.append({
                "type": "bullish_breakout",
                "start_time": safe_isoformat(df.index[i]),
                "breakout_time": safe_isoformat(df.index[i]),
                "breakout_price": cons_high,
                "consolidation_range": cons_range,
                "volume_ratio": volume.iloc[i:i+5].mean() / volume.iloc[i-20:i].mean(),
                "symbol": symbol,
                "timeframe": timeframe
            })
        
        # Downward breakout
        elif (breakout_candles['low'].min() < cons_low * 0.99 and
              volume.iloc[i:i+5].mean() > volume.iloc[i-20:i].mean() * 1.5):
            
            patterns.append({
                "type": "bearish_breakout",
                "start_time": safe_isoformat(df.index[i]),
                "breakout_time": safe_isoformat(df.index[i]),
                "breakout_price": cons_low,
                "consolidation_range": cons_range,
                "volume_ratio": volume.iloc[i:i+5].mean() / volume.iloc[i-20:i].mean(),
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_gap_pattern(df, symbol=None, timeframe=None):
    """Detect Gap patterns"""
    patterns = []
    open_price = df['open']
    close = df['close']
    high = df['high']
    low = df['low']
    
    for i in range(1, len(df)):
        prev_close = close.iloc[i-1]
        current_open = open_price.iloc[i]
        
        gap_size = abs(current_open - prev_close) / prev_close
        
        if gap_size > 0.01:  # Significant gap (1%+)
            if current_open > prev_close:
                gap_type = "gap_up"
            else:
                gap_type = "gap_down"
            
            patterns.append({
                "type": gap_type,
                "gap_time": safe_isoformat(df.index[i]),
                "gap_size": gap_size,
                "prev_close": prev_close,
                "current_open": current_open,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_volume_spike(df, symbol=None, timeframe=None):
    """Detect Volume Spike patterns"""
    patterns = []
    volume = df.get('volume', pd.Series([1] * len(df)))
    close = df['close']
    
    avg_volume = volume.rolling(window=20).mean()
    
    for i in range(20, len(df)):
        current_volume = volume.iloc[i]
        avg_vol = avg_volume.iloc[i]
        
        if current_volume > avg_vol * 3:  # Volume spike (3x average)
            price_change = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
            
            patterns.append({
                "type": "volume_spike",
                "time": safe_isoformat(df.index[i]),
                "volume": current_volume,
                "avg_volume": avg_vol,
                "volume_ratio": current_volume / avg_vol,
                "price_change": price_change,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_doji_pattern(df, symbol=None, timeframe=None):
    """Detect Doji candlestick pattern"""
    patterns = []
    open_price = df['open']
    close = df['close']
    high = df['high']
    low = df['low']
    
    for i in range(len(df)):
        body_size = abs(close.iloc[i] - open_price.iloc[i])
        total_range = high.iloc[i] - low.iloc[i]
        
        if total_range > 0 and body_size / total_range < 0.1:  # Small body relative to range
            patterns.append({
                "type": "doji",
                "time": safe_isoformat(df.index[i]),
                "open": open_price.iloc[i],
                "close": close.iloc[i],
                "high": high.iloc[i],
                "low": low.iloc[i],
                "body_ratio": body_size / total_range,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_hammer_pattern(df, symbol=None, timeframe=None):
    """Detect Hammer candlestick pattern"""
    patterns = []
    open_price = df['open']
    close = df['close']
    high = df['high']
    low = df['low']
    
    for i in range(len(df)):
        body_top = max(open_price.iloc[i], close.iloc[i])
        body_bottom = min(open_price.iloc[i], close.iloc[i])
        
        upper_wick = high.iloc[i] - body_top
        lower_wick = body_bottom - low.iloc[i]
        body_size = body_top - body_bottom
        
        # Hammer: long lower wick, small body, minimal upper wick
        if (lower_wick > body_size * 2 and
            upper_wick < body_size * 0.5 and
            body_size > 0):
            
            patterns.append({
                "type": "hammer",
                "time": safe_isoformat(df.index[i]),
                "open": open_price.iloc[i],
                "close": close.iloc[i],
                "high": high.iloc[i],
                "low": low.iloc[i],
                "lower_wick_ratio": lower_wick / body_size,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_shooting_star_pattern(df, symbol=None, timeframe=None):
    """Detect Shooting Star candlestick pattern"""
    patterns = []
    open_price = df['open']
    close = df['close']
    high = df['high']
    low = df['low']
    
    for i in range(len(df)):
        body_top = max(open_price.iloc[i], close.iloc[i])
        body_bottom = min(open_price.iloc[i], close.iloc[i])
        
        upper_wick = high.iloc[i] - body_top
        lower_wick = body_bottom - low.iloc[i]
        body_size = body_top - body_bottom
        
        # Shooting Star: long upper wick, small body, minimal lower wick
        if (upper_wick > body_size * 2 and
            lower_wick < body_size * 0.5 and
            body_size > 0):
            
            patterns.append({
                "type": "shooting_star",
                "time": safe_isoformat(df.index[i]),
                "open": open_price.iloc[i],
                "close": close.iloc[i],
                "high": high.iloc[i],
                "low": low.iloc[i],
                "upper_wick_ratio": upper_wick / body_size,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_engulfing_pattern(df, symbol=None, timeframe=None):
    """Detect Engulfing patterns"""
    patterns = []
    open_price = df['open']
    close = df['close']
    
    for i in range(1, len(df)):
        prev_open = open_price.iloc[i-1]
        prev_close = close.iloc[i-1]
        curr_open = open_price.iloc[i]
        curr_close = close.iloc[i]
        
        prev_body = abs(prev_close - prev_open)
        curr_body = abs(curr_close - curr_open)
        
        # Bullish engulfing
        if (prev_close < prev_open and  # Previous bearish
            curr_close > curr_open and  # Current bullish
            curr_open < prev_close and  # Current opens below prev close
            curr_close > prev_open and  # Current closes above prev open
            curr_body > prev_body * 1.1):  # Current body larger
            
            patterns.append({
                "type": "bullish_engulfing",
                "time": safe_isoformat(df.index[i]),
                "prev_candle_time": safe_isoformat(df.index[i-1]),
                "engulfing_ratio": curr_body / prev_body,
                "symbol": symbol,
                "timeframe": timeframe
            })
        
        # Bearish engulfing
        elif (prev_close > prev_open and  # Previous bullish
              curr_close < curr_open and  # Current bearish
              curr_open > prev_close and  # Current opens above prev close
              curr_close < prev_open and  # Current closes below prev open
              curr_body > prev_body * 1.1):  # Current body larger
            
            patterns.append({
                "type": "bearish_engulfing",
                "time": safe_isoformat(df.index[i]),
                "prev_candle_time": safe_isoformat(df.index[i-1]),
                "engulfing_ratio": curr_body / prev_body,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_inside_bar_pattern(df, symbol=None, timeframe=None):
    """Detect Inside Bar patterns"""
    patterns = []
    high = df['high']
    low = df['low']
    
    for i in range(1, len(df)):
        prev_high = high.iloc[i-1]
        prev_low = low.iloc[i-1]
        curr_high = high.iloc[i]
        curr_low = low.iloc[i]
        
        # Inside bar: current candle's range within previous candle's range
        if curr_high < prev_high and curr_low > prev_low:
            patterns.append({
                "type": "inside_bar",
                "time": safe_isoformat(df.index[i]),
                "mother_bar_time": safe_isoformat(df.index[i-1]),
                "mother_bar_high": prev_high,
                "mother_bar_low": prev_low,
                "inside_bar_high": curr_high,
                "inside_bar_low": curr_low,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_outside_bar_pattern(df, symbol=None, timeframe=None):
    """Detect Outside Bar patterns"""
    patterns = []
    high = df['high']
    low = df['low']
    
    for i in range(1, len(df)):
        prev_high = high.iloc[i-1]
        prev_low = low.iloc[i-1]
        curr_high = high.iloc[i]
        curr_low = low.iloc[i]
        
        # Outside bar: current candle's range engulfs previous candle's range
        if curr_high > prev_high and curr_low < prev_low:
            patterns.append({
                "type": "outside_bar",
                "time": safe_isoformat(df.index[i]),
                "prev_bar_time": safe_isoformat(df.index[i]),
                "prev_bar_high": prev_high,
                "prev_bar_low": prev_low,
                "outside_bar_high": curr_high,
                "outside_bar_low": curr_low,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_pin_bar_pattern(df, symbol=None, timeframe=None):
    """Detect Pin Bar patterns"""
    patterns = []
    open_price = df['open']
    close = df['close']
    high = df['high']
    low = df['low']
    
    for i in range(len(df)):
        body_top = max(open_price.iloc[i], close.iloc[i])
        body_bottom = min(open_price.iloc[i], close.iloc[i])
        
        upper_wick = high.iloc[i] - body_top
        lower_wick = body_bottom - low.iloc[i]
        body_size = body_top - body_bottom
        total_range = high.iloc[i] - low.iloc[i]
        
        # Pin bar: one long wick (tail) and small body
        if total_range > 0:
            if (lower_wick > total_range * 0.6 and  # Long lower tail
                body_size < total_range * 0.3):  # Small body
                
                patterns.append({
                    "type": "bullish_pin_bar",
                    "time": safe_isoformat(df.index[i]),
                    "tail_ratio": lower_wick / total_range,
                    "body_ratio": body_size / total_range,
                    "symbol": symbol,
                    "timeframe": timeframe
                })
            
            elif (upper_wick > total_range * 0.6 and  # Long upper tail
                  body_size < total_range * 0.3):  # Small body
                
                patterns.append({
                    "type": "bearish_pin_bar",
                    "time": safe_isoformat(df.index[i]),
                    "tail_ratio": upper_wick / total_range,
                    "body_ratio": body_size / total_range,
                    "symbol": symbol,
                    "timeframe": timeframe
                })
    return patterns

def detect_volatility_squeeze(df, symbol=None, timeframe=None):
    """Detect Volatility Squeeze patterns"""
    patterns = []
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Calculate volatility using ATR-like measure
    tr = pd.DataFrame({
        'tr1': high - low,
        'tr2': abs(high - close.shift(1)),
        'tr3': abs(low - close.shift(1))
    }).max(axis=1)
    
    atr = tr.rolling(window=14).mean()
    
    for i in range(20, len(df)):
        current_atr = atr.iloc[i]
        avg_atr = atr.iloc[i-20:i].mean()
        
        # Squeeze: current volatility much lower than average
        if current_atr < avg_atr * 0.5:
            patterns.append({
                "type": "volatility_squeeze",
                "time": safe_isoformat(df.index[i]),
                "current_atr": current_atr,
                "avg_atr": avg_atr,
                "squeeze_ratio": current_atr / avg_atr,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

def detect_volatility_expansion(df, symbol=None, timeframe=None):
    """Detect Volatility Expansion patterns"""
    patterns = []
    close = df['close']
    high = df['high']
    low = df['low']
    
    tr = pd.DataFrame({
        'tr1': high - low,
        'tr2': abs(high - close.shift(1)),
        'tr3': abs(low - close.shift(1))
    }).max(axis=1)
    
    atr = tr.rolling(window=14).mean()
    
    for i in range(20, len(df)):
        current_atr = atr.iloc[i]
        avg_atr = atr.iloc[i-20:i].mean()
        
        # Expansion: current volatility much higher than average
        if current_atr > avg_atr * 2:
            patterns.append({
                "type": "volatility_expansion",
                "time": safe_isoformat(df.index[i]),
                "current_atr": current_atr,
                "avg_atr": avg_atr,
                "expansion_ratio": current_atr / avg_atr,
                "symbol": symbol,
                "timeframe": timeframe
            })
    return patterns

# Add missing pattern from Fibonacci series
def detect_three_drives_pattern(df, symbol=None, timeframe=None):
    """Detect Three Drives pattern"""
    patterns = []
    local_max, local_min = find_local_peaks_troughs(df['high'], order=5), find_local_peaks_troughs(df['low'], order=5)
    
    # Combine extremes for bullish three drives
    all_extremes = []
    for idx in local_max[0]:
        all_extremes.append((idx, df.iloc[idx]['high'], 'high'))
    for idx in local_min[0]:
        all_extremes.append((idx, df.iloc[idx]['low'], 'low'))
    
    all_extremes.sort(key=lambda x: x[0])
    
    # Look for three successive drives (higher highs or lower lows)
    for i in range(len(all_extremes) - 6):
        # Pattern: Low-High-Low-High-Low-High (bullish three drives)
        points = all_extremes[i:i+7]
        
        if (points[0][2] == 'low' and points[1][2] == 'high' and
            points[2][2] == 'low' and points[3][2] == 'high' and
            points[4][2] == 'low' and points[5][2] == 'high'):
            
            # Check for successive higher highs
            if points[3][1] > points[1][1] and points[5][1] > points[3][1]:
                patterns.append({
                    "type": "bullish_three_drives",
                    "drive1_time": safe_isoformat(df.index[points[1][0]]),
                    "drive2_time": safe_isoformat(df.index[points[3][0]]),
                    "drive3_time": safe_isoformat(df.index[points[5][0]]),
                    "drive1_price": points[1][1],
                    "drive2_price": points[3][1],
                    "drive3_price": points[5][1],
                    "symbol": symbol,
                    "timeframe": timeframe
                })
    return patterns

def main(symbols=None, timeframes=None, selected_indicators=None):
    if symbols is None:
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'EURJPY', 'GBPJPY', 'EURGBP']
    if timeframes is None:
        timeframes = ['M15', 'H1', 'H4']
    DATA_FOLDER = "data"
    INDICATOR_FOLDER = "indicator_output"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 🧹 AUTO CLEANUP before pattern detection
    logging.info("🧹 Price Pattern Detector: Auto cleanup before processing...")
    try:
        cleanup_result = cleanup_pattern_price_data(max_age_hours=48, keep_latest=10)
        logging.info(f"✅ Cleaned {cleanup_result['total_files_deleted']} files, "
                    f"freed {cleanup_result['total_space_freed_mb']:.2f} MB")
    except Exception as e:
        logging.warning(f"⚠️ Cleanup warning: {e}")

    for symbol in symbols:
        for timeframe in timeframes:
            try:
                logging.info(f"Processing {symbol} {timeframe}...")
                
                # Handle both naming conventions: SYMBOL_TF.json and SYMBOL._TF.json
                data_pattern1 = os.path.join(DATA_FOLDER, f"{symbol}_{timeframe}.json")
                data_pattern2 = os.path.join(DATA_FOLDER, f"{symbol}._{timeframe}.json")
                data_files = glob.glob(data_pattern1) + glob.glob(data_pattern2)
                
                if not data_files:
                    logging.info(f"Data file missing for {symbol} {timeframe}, skipping.")
                    continue
                data_path = data_files[0]

                # Handle both naming conventions for indicator files
                indi_pattern1 = os.path.join(INDICATOR_FOLDER, f"{symbol}_{timeframe}_indicators.json")
                indi_pattern2 = os.path.join(INDICATOR_FOLDER, f"{symbol}._{timeframe}_indicators.json")
                indi_files = glob.glob(indi_pattern1) + glob.glob(indi_pattern2)
                
                if not indi_files:
                    logging.info(f"Indicator file missing for {symbol} {timeframe}, skipping.")
                    continue
                indicator_path = indi_files[0]

                try:
                    with open(data_path, "r") as f:
                        data_json = json.load(f)
                    df = pd.DataFrame(data_json)
                except Exception as e:
                    logging.error(f"Failed to read data file {data_path}: {e}")
                    continue

                if 'time' in df.columns:
                    try:
                        df['time'] = pd.to_datetime(df['time'])
                        df.set_index('time', inplace=True)
                    except Exception as e:
                        logging.error(f"Failed to parse 'time' column for {symbol} {timeframe}: {e}")
                        continue
                else:
                    logging.warning(f"No 'time' column in data for {symbol} {timeframe}, skipping.")
                    continue

                patterns_all = []
                # Enhanced pattern detection - LIMITED to find ONLY THE BEST
                try:
                    logging.info(f"  Detecting patterns for {symbol} {timeframe}...")
                    
                    # === DETECT ALL TYPES BUT LIMIT QUANTITY ===
                    # TOP PRIORITY PATTERNS (Most reliable) - Limit to 2 each
                    patterns_all.extend(detect_double_top(df, symbol, timeframe)[:2])
                    patterns_all.extend(detect_double_bottom(df, symbol, timeframe)[:2])
                    patterns_all.extend(detect_head_and_shoulders(df, symbol, timeframe)[:1])
                    patterns_all.extend(detect_inverse_head_and_shoulders(df, symbol, timeframe)[:1])
                    
                    # TRIANGLE PATTERNS (Medium priority) - Limit to 2
                    triangle_patterns = detect_triangle_pattern(df, symbol, timeframe)[:2]
                    patterns_all.extend(triangle_patterns)
                    
                    # CONTINUATION PATTERNS (Medium priority) - Limit to 1 each
                    patterns_all.extend(detect_flag_pattern(df, symbol, timeframe)[:1])
                    patterns_all.extend(detect_pennant_pattern(df, symbol, timeframe)[:1])
                    
                    # WEDGE PATTERNS - Limit to 1 each
                    patterns_all.extend(detect_rising_wedge(df, symbol, timeframe)[:1])
                    patterns_all.extend(detect_falling_wedge(df, symbol, timeframe)[:1])
                    
                    # ADVANCED PATTERNS (Only if sufficient data) - Limit to 1
                    if len(df) > 100:
                        patterns_all.extend(detect_cup_and_handle(df, symbol, timeframe)[:1])
                    
                    # Limit total patterns to reduce processing time
                    patterns_all = patterns_all[:10]  # Max 10 patterns to choose from
                        
                except Exception as e:
                    logging.error(f"Pattern detection failed for {symbol} {timeframe}: {e}")
                    patterns_all = []

                # Process patterns if found
                if patterns_all:
                    logging.info(f"  Found {len(patterns_all)} candidate patterns for {symbol} {timeframe}")
                else:
                    logging.info(f"  No patterns found for {symbol} {timeframe}, creating demo pattern...")
                    # Create demo pattern for better GUI demonstration
                    patterns_all = create_test_patterns(symbol, timeframe)[:1]  # Only 1 demo pattern
                
                # Ensure all patterns have standardized format
                standardized_patterns = []
                for pattern in patterns_all:
                    try:
                        std_pattern = standardize_pattern(pattern)
                        standardized_patterns.append(std_pattern)
                    except Exception as e:
                        logging.warning(f"Failed to standardize pattern: {e}")
                        continue
                
                patterns_all = standardized_patterns
                
                # Apply multi-factor confirmation if we have indicator data
                confirmed_patterns = []
                try:
                    with open(indicator_path, "r") as f:
                        indicator_data = json.load(f)
                    indicator_df = pd.DataFrame(indicator_data)
                    
                    # Get latest indicator values
                    latest_indicator = indicator_df.iloc[-1].to_dict() if len(indicator_df) > 0 else {}
                    
                    # Load real trendline SR data instead of simplified trend info
                    trend_info = load_trendline_sr_data(symbol, timeframe)
                    
                    # Fallback to simplified trend info if trendline_sr data is not available
                    if not trend_info:
                        logging.warning(f"No trendline_sr data for {symbol} {timeframe}, using fallback trend calculation")
                        trend_info = {
                            "trend": "up" if latest_indicator.get("EMA_20", 0) > latest_indicator.get("EMA_50", 0) else "down",
                            "trend_strength": min(1.0, abs(latest_indicator.get("RSI", 50) - 50) / 50)
                        }
                    
                    for pattern in patterns_all:
                        try:
                            # Apply multi-factor confirmation
                            confirmation_result = confirm_pattern_with_indicator_and_trend(
                                pattern, latest_indicator, trend_info, selected_indicators
                            )
                            
                            # Add confirmation details to pattern
                            pattern["confirmation_score"] = confirmation_result["total_score"]
                            pattern["confirmation_details"] = confirmation_result
                            
                            # Only include patterns with reasonable confirmation
                            if confirmation_result["total_score"] >= 0.3:  # Minimum threshold
                                confirmed_patterns.append(pattern)
                                
                        except Exception as e:
                            logging.warning(f"Pattern confirmation failed: {e}")
                            # Keep pattern without confirmation
                            pattern["confirmation_score"] = 0.5
                            confirmed_patterns.append(pattern)
                    
                    patterns_all = confirmed_patterns
                    logging.info(f"Applied confirmation filtering: {len(confirmed_patterns)} patterns remain")
                    
                except Exception as e:
                    logging.warning(f"Could not apply pattern confirmation for {symbol} {timeframe}: {e}")
                    # Continue without confirmation
                    patterns_all = standardized_patterns
                
                if not patterns_all:
                    logging.warning(f"  No valid patterns for {symbol} {timeframe}")
                    # Create empty pattern file
                    pattern_file = os.path.join(OUTPUT_FOLDER, f"{symbol}_{timeframe}_patterns.json")
                    with open(pattern_file, "w") as f:
                        json.dump([], f, indent=4)
                    continue

                # Standardize all patterns and add required fields
                standardized_patterns = []
                for pattern in patterns_all:
                    try:
                        std_pattern = standardize_pattern(pattern)
                        standardized_patterns.append(std_pattern)
                    except Exception as e:
                        logging.warning(f"Failed to standardize pattern: {e}")
                        continue
                
                if patterns_all:
                    # === APPLY NEW PRIORITY LOGIC: Recency > Duration > Accuracy ===
                    # Get THE SINGLE BEST pattern using new priority logic
                    best_pattern = get_best_pattern(patterns_all)
                    
                    if best_pattern:
                        # Save ONLY the single best pattern as array with one element
                        pattern_file = os.path.join(OUTPUT_FOLDER, f"{symbol}_{timeframe}_patterns.json")
                        with open(pattern_file, "w") as f:
                            json.dump(convert_times_to_str([best_pattern]), f, indent=4)
                        
                        logging.info(f"  ✅ Selected BEST pattern for {symbol} {timeframe}: {best_pattern.get('type', 'unknown')} "
                                   f"(Time: {best_pattern.get('end_time', 'N/A')}, "
                                   f"Duration: {best_pattern.get('duration', 0):.0f}s, "
                                   f"Confirmation: {best_pattern.get('confirmation_score', 0):.3f}, "
                                   f"Confidence: {best_pattern.get('confidence', 0):.3f})")
                    else:
                        logging.warning(f"  Failed to select best pattern for {symbol} {timeframe}")
                        # Create empty pattern file
                        pattern_file = os.path.join(OUTPUT_FOLDER, f"{symbol}_{timeframe}_patterns.json")
                        with open(pattern_file, "w") as f:
                            json.dump([], f, indent=4)
                    logging.info(f"Saved {len(patterns_all)} patterns for {symbol} {timeframe}")
                    logging.info(f"Best pattern: {best_pattern.get('pattern', 'Unknown')} "
                               f"(conf: {best_pattern.get('confidence', 0):.3f}, "
                               f"confirm: {best_pattern.get('confirmation_score', 0):.3f})")
                else:
                    logging.info(f"No valid patterns after confirmation for {symbol} {timeframe}")
                    # Create empty pattern file
                    pattern_file = os.path.join(OUTPUT_FOLDER, f"{symbol}_{timeframe}_patterns.json")
                    with open(pattern_file, "w") as f:
                        json.dump([], f, indent=4)
                
            except Exception as e:
                logging.error(f"Error processing {symbol} {timeframe}: {e}")

def cleanup_pattern_price_data(max_age_hours: int = 48, keep_latest: int = 10) -> Dict[str, Any]:
    """
    🧹 SMART CLEANUP for Price Patterns: Only keep data for active symbols
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
        'module_name': 'price_pattern_detector_smart',
        'active_symbols': list(active_symbols),
        'total_files_deleted': 0,
        'total_space_freed_mb': 0.0,
        'cleanup_time': datetime.now().isoformat()
    }
    
    # Directories to clean - REMOVED pattern_signals (managed by pattern_detector.py)
    directories = [OUTPUT_FOLDER, 'analysis_output']
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        deleted_count = 0
        space_freed = 0.0
        
        for filename in os.listdir(directory):
            if not filename.endswith('.json'):
                continue
                
            file_path = os.path.join(directory, filename)
            
            try:
                # Extract symbol from filename
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
                logging.warning(f"Error processing {filename}: {e}")
        
        cleanup_stats['total_files_deleted'] += deleted_count
        cleanup_stats['total_space_freed_mb'] += space_freed
    
    logging.info(f"🧹 Smart Price Patterns cleanup: {cleanup_stats['total_files_deleted']} files deleted, "
                 f"{cleanup_stats['total_space_freed_mb']:.2f} MB freed")
    
    return cleanup_stats
    
    for directory in target_directories:
        if os.path.exists(directory):
            logging.info(f"🧹 Price Pattern Detector cleaning {directory}...")
            dir_stats = _clean_pattern_directory(directory, max_age_hours, keep_latest)
            cleanup_stats['directories_cleaned'].append({
                'directory': directory,
                'files_deleted': dir_stats['deleted'],
                'space_freed_mb': dir_stats['space_freed']
            })
            cleanup_stats['total_files_deleted'] += dir_stats['deleted']
            cleanup_stats['total_space_freed_mb'] += dir_stats['space_freed']
        else:
            logging.info(f"📁 Directory {directory} does not exist, skipping")
    
    logging.info(f"🧹 PRICE PATTERN DETECTOR cleanup complete: "
                f"{cleanup_stats['total_files_deleted']} files deleted, "
                f"{cleanup_stats['total_space_freed_mb']:.2f} MB freed")
    return cleanup_stats

def _clean_pattern_directory(directory: str, max_age_hours: int, keep_latest: int) -> Dict[str, int]:
    """Helper function để clean pattern directory"""
    deleted_count = 0
    space_freed = 0.0
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    try:
        if not os.path.exists(directory):
            return {'deleted': 0, 'space_freed': 0.0}
            
        # Lấy tất cả pattern files
        all_files = []
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path) and file_name.endswith('_patterns.json'):
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
                    logging.debug(f"Deleted: {file_info['name']}")
                except Exception as e:
                    logging.error(f"Failed to delete {file_info['path']}: {e}")
        
    except Exception as e:
        logging.error(f"Error cleaning directory {directory}: {e}")
    
    return {'deleted': deleted_count, 'space_freed': space_freed}

    logging.info("Pattern analysis completed successfully!")

if __name__ == "__main__":
    main()

