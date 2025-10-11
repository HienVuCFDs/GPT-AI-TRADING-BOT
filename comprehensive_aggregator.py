#!/usr/bin/env python3
"""Comprehensive multi-timeframe aggregator & report generator.

Cleaned header: removed stale placeholder & duplicated minimal implementation that caused
syntax errors and shadowed the real (full) implementation further below.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Any, List, Dict, Tuple, Set, Union
import os, sys, json, gzip, glob, re, logging, time

# Fix encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables early
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without

# Module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')

# ---- Config fallback (original project provides trading_mode_config.CFG) ----
try:  # pragma: no cover - defensive
    from trading_mode_config import CFG  # type: ignore
except Exception:  # pragma: no cover
    class _DummyCFG:
        OUT = 'analysis_results'
        IND = 'indicator_output'
        DATA = 'data'
        PPRICE = 'pattern_price'
        PSIG = 'pattern_signals'
        SR = 'trendline_sr'
        ACCT = 'account_scans'
        TF = ["D1","H4","H1","M30","M15","M5"]
        STRICT_IND_ONLY = False
    CFG = _DummyCFG()  # type: ignore

def price_decimals(symbol: str) -> int:
    """Heuristic decimal precision for pretty printing (aligned with order rounding)."""
    s = (symbol or '').upper()
    if s.startswith('XAU'): return 2
    if any(tag in s for tag in ('BTC','ETH','SOL','ADA','DOGE','BNB','XRP','TRX','LTC','DOT','AVAX')): return 2
    if s.endswith('JPY'): return 3
    fx = ('USD','EUR','GBP','AUD','NZD','CAD','CHF')
    if any(s.startswith(x) or s.endswith(x) for x in fx): return 5
    return 4

# ------------------------------
# Account scan utilities (open positions / orders)
# ------------------------------
_ACCOUNT_SCAN_CACHE: dict | None = None
_ACCOUNT_SCAN_MTIME: float | None = None

# Global variable to store current price across function calls
current_price_from_indicators: Optional[float] = None

def load_account_scan(path: Optional[str] = None, force: bool = False) -> Optional[dict]:
    """Load (and cache) the mt5_essential_scan.json.

    Improvements:
      - Detect file modification time; auto-reload if file changed (e.g. when switching account).
      - Optional force reload.
    """
    global _ACCOUNT_SCAN_CACHE, _ACCOUNT_SCAN_MTIME
    try:
        base = path or os.path.join(CFG.ACCT, 'mt5_essential_scan.json')
        logger.info(f"🔍 DEBUG: Loading account scan from path: {base}")
        logger.info(f"🔍 DEBUG: File exists: {os.path.exists(base)}")
        if not os.path.exists(base):
            _ACCOUNT_SCAN_CACHE = None
            _ACCOUNT_SCAN_MTIME = None
            logger.warning(f"🚫 Account scan file not found: {base}")
            return None
        mtime = os.path.getmtime(base)
        if force or _ACCOUNT_SCAN_CACHE is None or _ACCOUNT_SCAN_MTIME is None or mtime != _ACCOUNT_SCAN_MTIME:
            with open(base, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"🔍 DEBUG: Loaded JSON data keys: {list(data.keys()) if isinstance(data, dict) else 'not_dict'}")
            if isinstance(data, dict):
                active_pos = data.get('active_positions', [])
                logger.info(f"🔍 DEBUG: Raw active_positions length from file: {len(active_pos)}")
                _ACCOUNT_SCAN_CACHE = data
                _ACCOUNT_SCAN_MTIME = mtime
        return _ACCOUNT_SCAN_CACHE
    except Exception:
        return None

def invalidate_account_scan_cache():
    """Manually clear cached account scan (call after account change)."""
    global _ACCOUNT_SCAN_CACHE, _ACCOUNT_SCAN_MTIME
    _ACCOUNT_SCAN_CACHE = None
    _ACCOUNT_SCAN_MTIME = None

def force_reload_account_scan(path: Optional[str] = None) -> Optional[dict]:
    """Convenience wrapper to reload ignoring cache."""
    return load_account_scan(path=path, force=True)

def calculate_risk_metrics(scan: dict, positions: List[dict]) -> dict:
    """Calculate comprehensive risk metrics for better position management."""
    try:
        account = scan.get('account', {})
        equity = account.get('equity', 0.0)
        balance = account.get('balance', 0.0)
        margin = account.get('margin', 0.0)
        free_margin = account.get('free_margin', 0.0)
        margin_level = account.get('margin_level', 0.0)
        
        # Portfolio metrics
        total_exposure = sum(abs(p.get('profit', 0)) for p in positions)
        max_single_loss = min((p.get('profit', 0) for p in positions), default=0)
        avg_loss = sum(p.get('profit', 0) for p in positions if p.get('profit', 0) < 0) / max(1, len([p for p in positions if p.get('profit', 0) < 0]))
        
        # Risk per symbol
        symbol_exposure = {}
        symbol_volume = {}
        for p in positions:
            sym = p.get('symbol', '')
            if sym:
                symbol_exposure[sym] = symbol_exposure.get(sym, 0) + p.get('profit', 0)
                symbol_volume[sym] = symbol_volume.get(sym, 0) + p.get('volume', 0)
        
        # Floating P&L calculation (percentage of unrealized gains/losses)
        # This shows current floating performance, not true drawdown
        floating_pnl_pct = ((equity - balance) / balance * 100) if balance > 0 else 0
        
        # For true drawdown, we would need peak equity history
        # For now, show floating P&L instead of misleading "drawdown"
        drawdown_pct = abs(floating_pnl_pct) if floating_pnl_pct < 0 else 0
        
        # Risk levels
        risk_level = 'LOW'
        if drawdown_pct > 20 or margin_level < 200:
            risk_level = 'HIGH'
        elif drawdown_pct > 10 or margin_level < 500:
            risk_level = 'MEDIUM'
            
        # Position concentration risk
        max_symbol_exposure = max((abs(exp) for exp in symbol_exposure.values()), default=0)
        concentration_risk = (max_symbol_exposure / equity * 100) if equity > 0 else 0
        
        return {
            'overall_risk_level': risk_level,
            'drawdown_pct': round(drawdown_pct, 2),
            'floating_pnl_pct': round(floating_pnl_pct, 2),  # Add actual floating P&L percentage
            'margin_level': margin_level,
            'total_exposure': round(total_exposure, 2),
            'exposure_pct': round(total_exposure / equity * 100, 2) if equity > 0 else 0,
            'max_single_loss': round(max_single_loss, 2),
            'max_single_loss_pct': round(max_single_loss / equity * 100, 2) if equity > 0 else 0,
            'avg_loss': round(avg_loss, 2),
            'concentration_risk_pct': round(concentration_risk, 2),
            'symbol_exposure': {k: round(v, 2) for k, v in symbol_exposure.items()},
            'symbol_volume': {k: round(v, 2) for k, v in symbol_volume.items()},
            'positions_count': len(positions),
            'losing_positions': len([p for p in positions if p.get('profit', 0) < 0]),
            'winning_positions': len([p for p in positions if p.get('profit', 0) > 0])
        }
    except Exception:
        return {
            'overall_risk_level': 'UNKNOWN',
            'drawdown_pct': 0,
            'margin_level': 0,
            'total_exposure': 0,
            'exposure_pct': 0,
            'max_single_loss': 0,
            'max_single_loss_pct': 0,
            'avg_loss': 0,
            'concentration_risk_pct': 0,
            'symbol_exposure': {},
            'symbol_volume': {},
            'positions_count': 0,
            'losing_positions': 0,
            'winning_positions': 0
        }

def calculate_pips(symbol: str, entry_price: float, current_price: float, direction: str) -> float:
    """
    Calculate pips profit/loss for a position with comprehensive instrument support.
    
    Pip Standards:
    - Major FX pairs (EURUSD, GBPUSD, etc.): 1 pip = 0.0001 (4th decimal)
    - JPY pairs (USDJPY, EURJPY, etc.): 1 pip = 0.01 (2nd decimal) 
    - High-value Crypto (BTC, ETH): 1 pip = 1.0 (whole number unit)
    - SOL Crypto: 1 pip = 0.1 (1st decimal)
    - Other Crypto (ADA, DOGE, etc.): 1 pip = 0.01 (2nd decimal)
    - Metals (XAUUSD, XAGUSD): 1 pip = 0.1 (1st decimal)
    - Indices (US30, NAS100): 1 pip = 1.0 (whole number)
    - Oil (USOIL, UKOIL): 1 pip = 0.01 (2nd decimal)
    - Bonds: 1 pip = 0.001 (3rd decimal)
    """
    try:
        # Clean symbol name for proper matching
        clean_symbol = symbol.replace('_m', '').replace('.', '').upper()
        
        # Determine pip size based on comprehensive instrument classification
        pip_size = 0.0001  # Default for major FX pairs (1 pip = 0.0001)
        instrument_type = "Major FX"
        
        # JPY Currency Pairs (2 decimal places)
        if clean_symbol.endswith('JPY') or clean_symbol.startswith('JPY'):
            pip_size = 0.01  # 1 pip = 0.01
            instrument_type = "JPY Pair"
            
        # Cryptocurrency - pip values based on price level and contract specification
        elif any(crypto in clean_symbol for crypto in [
            'BTC', 'ETH', 'SOL', 'ADA', 'DOGE', 'LTC', 'XRP', 'DOT', 'LINK', 'UNI',
            'AVAX', 'MATIC', 'ATOM', 'ALGO', 'FTM', 'NEAR', 'LUNA', 'ICP', 'VET', 'MANA',
            'BNB', 'TRX', 'SHIB', 'ARB', 'OP', 'APE', 'SAND', 'CRO', 'FTT'
        ]):
            # High-value crypto (BTC, ETH): 1 pip = 1.0 (whole price unit)
            if any(high_crypto in clean_symbol for high_crypto in ['BTC', 'ETH']):
                pip_size = 1.0  # BTC: 100000.50 → 100001.50 = 1 pip, ETH: 4000 → 4001 = 1 pip
            # Mid-value crypto (BNB, SOL, LTC): 1 pip = 0.1 (1st decimal)
            elif any(mid_crypto in clean_symbol for mid_crypto in ['BNB', 'SOL', 'LTC', 'AVAX', 'DOT', 'LINK', 'UNI', 'MATIC', 'ATOM']):
                pip_size = 0.1  # BNB: 1160.0 → 1160.1 = 1 pip, SOL: 220.0 → 220.1 = 1 pip, LTC: 120.2 → 120.3 = 1 pip
            # Low-value crypto (ADA, DOGE, etc): 1 pip = 0.01 (2nd decimal)  
            else:
                pip_size = 0.01  # ADA, DOGE, etc: 120.00 → 120.01 = 1 pip
            instrument_type = "Crypto"
            
        # Precious Metals (1 decimal place)
        elif any(metal in clean_symbol for metal in [
            'XAU', 'XAG', 'GOLD', 'SILVER', 'PLATINUM', 'PALLADIUM'
        ]):
            pip_size = 0.1   # 1 pip = 0.1
            instrument_type = "Metal"
            
        # Stock Indices (whole numbers)
        elif any(index in clean_symbol for index in [
            'US30', 'US500', 'US100', 'NAS100', 'SPX500', 'DJ30', 'GER30', 'UK100', 
            'FRA40', 'ESP35', 'ITA40', 'AUS200', 'JPN225', 'HK50', 'CHINA50'
        ]):
            pip_size = 1.0    # 1 pip = 1.0
            instrument_type = "Index"
            
        # Energy/Oil (2 decimal places)
        elif any(oil in clean_symbol for oil in [
            'USOIL', 'UKOIL', 'NGAS', 'CRUDE', 'WTI', 'BRENT'
        ]):
            pip_size = 0.01  # 1 pip = 0.01
            instrument_type = "Energy"
            
        # Government Bonds (3 decimal places)
        elif any(bond in clean_symbol for bond in [
            'USTB', 'GERB', 'UKGB', 'JPNB', 'BOND'
        ]) or 'BOND' in clean_symbol or clean_symbol.endswith('B'):
            pip_size = 0.001  # 1 pip = 0.001
            instrument_type = "Bond"
            
        # Exotic FX pairs or minor currencies (4 decimal places)
        elif any(exotic in clean_symbol for exotic in [
            'ZAR', 'TRY', 'MXN', 'RUB', 'PLN', 'HUF', 'CZK', 'NOK', 'SEK', 'DKK'
        ]):
            pip_size = 0.0001  # 1 pip = 0.0001
            instrument_type = "Exotic FX"
            
        # Individual Stocks (2 decimal places)
        elif any(stock_suffix in clean_symbol for stock_suffix in [
            '.US', '.UK', '.DE', '.FR', '.JP'
        ]) or len(clean_symbol) <= 5:  # Short symbols likely stocks
            pip_size = 0.01  # 1 pip = 0.01
            instrument_type = "Stock"
            
        # Calculate price difference based on direction
        if direction.upper() == 'BUY':
            price_diff = current_price - entry_price
        else:  # SELL
            price_diff = entry_price - current_price
            
        # Convert to pips by dividing price difference by pip size
        pips = price_diff / pip_size
        
        print(f"DEBUG PIPS: {symbol} ({instrument_type}) {direction} entry={entry_price} current={current_price} diff={price_diff} pip_size={pip_size} pips={pips}")
        
        return round(pips, 1)
        
    except Exception as e:
        print(f"ERROR in calculate_pips: {e}")
        return 0.0

def get_pip_value(symbol: str) -> float:
    """Get pip value for symbol (point size for pip calculation) - UPDATED FOR NEW PIP STANDARDS"""
    try:
        s = (symbol or '').upper()
        
        # Gold/Silver metals
        if s.startswith('XAU') or s.startswith('XAG'):
            return 0.1  # 0.1 = 1 pip for metals
            
        # Cryptocurrency - distinguish between high-value and other crypto
        elif any(crypto in s for crypto in ['BTC', 'ETH', 'SOL', 'ADA', 'DOGE', 'BNB', 'TRX', 'SHIB', 'ARB', 'OP', 'APE', 'SAND', 'CRO', 'FTT']):
            # High-value crypto (BTC, ETH): 1 pip = 1.0 unit (70 pips = $70 USD)
            if any(high_crypto in s for high_crypto in ['BTC', 'ETH']):
                return 1.0  # BTC/ETH: 1 pip = 1.0 price unit (e.g., ETH 4000.00 -> 4070.00 = 70 pips)
            elif 'SOL' in s:
                return 0.1  # SOL: 1 pip = 0.1 (SOL 220.00 -> 220.70 = 7 pips)
            else:
                return 0.01  # ADA, DOGE, BNB, etc: 1 pip = 0.01
                
        # JPY pairs (2 decimal places)
        elif s.endswith('JPY'):
            return 0.01  # 0.01 = 1 pip for JPY pairs
            
        # Major FX pairs (4 decimal places) 
        else:
            return 0.0001  # 0.0001 = 1 pip for major pairs
            
    except Exception:
        return 0.0001  # Default fallback for major FX

def calculate_smart_sl_buffer(symbol: str, current_price: float, entry_price: float, direction: str, confidence: float) -> float:
    """
    🧠 SMART DYNAMIC SL BUFFER CALCULATION
    
    Tính toán SL buffer thông minh dựa trên:
    1. ATR (Average True Range) - Volatility thực tế của thị trường
    2. Support/Resistance levels - Vùng kỹ thuật quan trọng
    3. Market session volatility - Volatility theo session
    4. Price action context - Bối cảnh price action
    5. Confidence-based adjustment - Điều chỉnh theo độ tin cậy tín hiệu
    
    Returns: SL buffer in pips (float)
    """
    try:
        import MetaTrader5 as mt5
        from datetime import datetime, timedelta
        import numpy as np
        
        pip_value = get_pip_value(symbol)
        if not pip_value or not current_price or not entry_price:
            return 80.0  # Safe fallback
        
        # 1️⃣ ATR-BASED DYNAMIC CALCULATION
        # Get recent price data để tính ATR
        try:
            # Select symbol first
            if not mt5.symbol_select(symbol, True):
                logger.debug(f"⚠️ Could not select {symbol} for ATR calculation")
            
            # Get H1 bars for ATR calculation (last 14 periods)
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 14)
            
            if rates is not None and len(rates) >= 10:
                # Calculate True Range for each period
                true_ranges = []
                for i in range(1, len(rates)):
                    high = rates[i]['high']
                    low = rates[i]['low'] 
                    prev_close = rates[i-1]['close']
                    
                    tr1 = high - low
                    tr2 = abs(high - prev_close)
                    tr3 = abs(low - prev_close)
                    
                    true_ranges.append(max(tr1, tr2, tr3))
                
                if true_ranges:
                    # ATR = Average of True Ranges
                    atr_price = np.mean(true_ranges)
                    atr_pips = atr_price / pip_value
                    
                    logger.debug(f"📊 {symbol} ATR: {atr_pips:.1f} pips")
                else:
                    atr_pips = 50.0  # Fallback
            else:
                atr_pips = 50.0  # Fallback if no data
                
        except Exception as e:
            logger.debug(f"⚠️ ATR calculation error for {symbol}: {e}")
            atr_pips = 50.0  # Fallback
        
        # 2️⃣ MARKET SESSION VOLATILITY ADJUSTMENT
        current_hour = datetime.now().hour
        session_multiplier = 1.0
        
        # Asian session (lower volatility)
        if 0 <= current_hour < 8:
            session_multiplier = 0.8
        # European session (medium volatility)  
        elif 8 <= current_hour < 16:
            session_multiplier = 1.0
        # US session (higher volatility)
        elif 16 <= current_hour < 24:
            session_multiplier = 1.2
        
        # 3️⃣ INSTRUMENT-SPECIFIC ADJUSTMENTS
        instrument_multiplier = 1.0
        
        # JPY pairs - higher volatility, need more buffer
        if 'JPY' in symbol:
            instrument_multiplier = 1.3
        # Major EUR/USD pairs - standard volatility
        elif any(pair in symbol for pair in ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']):
            instrument_multiplier = 1.0
        # Crypto pairs - much higher volatility
        elif any(crypto in symbol for crypto in ['BTC', 'ETH', 'SOL', 'ADA']):
            instrument_multiplier = 1.8
        # Metals - high volatility
        elif 'XAU' in symbol or 'XAG' in symbol:
            instrument_multiplier = 1.5
        
        # 4️⃣ CONFIDENCE-BASED ADJUSTMENT
        # Higher confidence = tighter SL, Lower confidence = wider SL
        confidence_multiplier = 1.0
        if confidence >= 80:
            confidence_multiplier = 0.9  # Tighter SL for high confidence
        elif confidence >= 70:
            confidence_multiplier = 1.0  # Standard SL
        elif confidence >= 60:  
            confidence_multiplier = 1.1  # Slightly wider SL
        else:
            confidence_multiplier = 1.3  # Much wider SL for low confidence
        
        # 5️⃣ PRICE ACTION CONTEXT
        # Check if we're in trending or ranging market
        entry_to_current_pips = abs(current_price - entry_price) / pip_value
        
        price_action_multiplier = 1.0
        # If price has moved significantly from entry, we need wider buffer
        if entry_to_current_pips > 100:
            price_action_multiplier = 1.2  # Wider buffer in strong moves
        elif entry_to_current_pips < 20:
            price_action_multiplier = 0.9  # Tighter buffer if price is stable
        
        # 6️⃣ COMBINE ALL FACTORS
        # Base SL from ATR (minimum 2x ATR, maximum 4x ATR for safety)
        base_sl_pips = max(30, min(atr_pips * 2.5, 150))  # Reasonable bounds
        
        # Apply all multipliers
        smart_sl_pips = (base_sl_pips * 
                        session_multiplier * 
                        instrument_multiplier * 
                        confidence_multiplier * 
                        price_action_multiplier)
        
        # 7️⃣ SAFETY BOUNDS - Prevent extreme values
        min_sl_pips = 25  # Absolute minimum
        max_sl_pips = 300  # Absolute maximum 
        
        # Special minimums for volatile instruments
        if 'JPY' in symbol:
            min_sl_pips = 40
        elif any(crypto in symbol for crypto in ['BTC', 'ETH', 'SOL']):
            min_sl_pips = 60
        elif 'XAU' in symbol:
            min_sl_pips = 50
            
        final_sl_pips = max(min_sl_pips, min(smart_sl_pips, max_sl_pips))
        
        # 8️⃣ COMPREHENSIVE LOGGING
        logger.info(f"🧠 SMART SL for {symbol}:")
        logger.info(f"   📊 ATR: {atr_pips:.1f} pips → Base: {base_sl_pips:.1f} pips")
        logger.info(f"   🌍 Session: {session_multiplier:.1f}x | 📈 Instrument: {instrument_multiplier:.1f}x")
        logger.info(f"   🎯 Confidence: {confidence:.1f}% → {confidence_multiplier:.1f}x")  
        logger.info(f"   📊 Price Action: {entry_to_current_pips:.1f} pips → {price_action_multiplier:.1f}x")
        logger.info(f"   🛡️ Final SL Buffer: {final_sl_pips:.1f} pips")
        
        return final_sl_pips
        
    except Exception as e:
        logger.error(f"❌ Smart SL calculation error for {symbol}: {e}")
        # Intelligent fallback based on symbol type
        if 'JPY' in symbol:
            return 80.0
        elif any(crypto in symbol for crypto in ['BTC', 'ETH', 'SOL']):
            return 120.0 
        elif 'XAU' in symbol:
            return 100.0
        else:
            return 60.0

def enhanced_action_decision(position: dict, signal: dict, risk_metrics: dict, args, risk_settings: dict = None) -> dict:
    """Enhanced action decision making with risk-adjusted logic."""
    try:
        # Extract position data with proper None checking and TYPE CONVERSION
        pos_dir = str(position.get('direction', '') or '')
        profit = float(position.get('profit', 0) or 0)
        pct_eq = float(position.get('pct_equity', 0) or 0)
        pct_balance = float(position.get('pct_balance', 0) or 0)
        position_risk_pct = float(position.get('position_risk_pct', 0) or 0)
        risk_level = str(position.get('risk_level', 'LOW') or 'LOW')
        signal_alignment = str(position.get('signal_alignment', 'NEUTRAL') or 'NEUTRAL')
        symbol = str(position.get('symbol', '') or '')
        entry_price = float(position.get('price_open', 0) or 0)
        current_price = float(position.get('price_current', 0) or 0)
        
        # Calculate pips profit/loss with None safety and TYPE CONVERSION
        pips = 0.0
        if entry_price and current_price and pos_dir:
            try:
                pips = calculate_pips(symbol, float(entry_price), float(current_price), str(pos_dir))
                pips = float(pips or 0)  # Ensure pips is never None and always float
                
                # 🛡️ CRITICAL VALIDATION: Ensure pips calculation is correct for XAUUSD
                if symbol.upper().startswith('XAU'):
                    print(f"🔍 XAUUSD PIPS VALIDATION: {symbol} {pos_dir} entry={entry_price} current={current_price} pips={pips}")
                    # Double-check for XAUUSD: if profit is negative, pips MUST be negative
                    actual_profit = float(position.get('profit', 0) or 0)
                    if actual_profit < 0 and pips > 0:
                        print(f"🚨 CRITICAL BUG DETECTED: XAUUSD profit={actual_profit} but pips={pips} > 0!")
                        print(f"🔧 FORCE CORRECTION: Setting pips to negative value based on profit")
                        pips = -abs(pips)  # Force negative pips for negative profit
                    elif actual_profit > 0 and pips < 0:
                        print(f"🚨 CRITICAL BUG DETECTED: XAUUSD profit={actual_profit} but pips={pips} < 0!")
                        print(f"🔧 FORCE CORRECTION: Setting pips to positive value based on profit")
                        pips = abs(pips)  # Force positive pips for positive profit
                
            except Exception as e:
                print(f"ERROR calculating pips for {symbol}: {e}")
                pips = 0.0
        
        # Extract signal data with None safety and TYPE CONVERSION
        sig_dir = str(signal.get('signal', 'NEUTRAL') or 'NEUTRAL')
        conf = float(signal.get('confidence', 0) or 0)
        entry = signal.get('entry')
        if entry is not None:
            entry = float(entry)
        idea_sl = signal.get('stoploss')
        if idea_sl is not None:
            idea_sl = float(idea_sl)
        idea_tp = signal.get('takeprofit')
        if idea_tp is not None:
            idea_tp = float(idea_tp)
        
        # Extract risk metrics with None safety and TYPE CONVERSION
        overall_risk = str(risk_metrics.get('overall_risk_level', 'LOW') or 'LOW')
        drawdown_pct = float(risk_metrics.get('drawdown_pct', 0) or 0)
        margin_level = float(risk_metrics.get('margin_level', 0) or 0)
        concentration_risk = float(risk_metrics.get('concentration_risk_pct', 0) or 0)
        
        # 🛡️ NEW: Get risk settings from args with None safety and TYPE CONVERSION
        max_risk_per_trade_val = getattr(args, 'max_risk_per_trade', 5.0)
        max_risk_per_trade = float(max_risk_per_trade_val) if max_risk_per_trade_val is not None else 5.0
        
        max_total_risk_val = getattr(args, 'max_total_risk', 20.0)
        max_total_risk = float(max_total_risk_val) if max_total_risk_val is not None else 20.0
        
        max_drawdown_val = getattr(args, 'max_drawdown', 15.0)
        max_drawdown = float(max_drawdown_val) if max_drawdown_val is not None else 15.0
        
        min_margin_level = float(getattr(args, 'min_margin_level', 300.0) or 300.0)
        risk_mode = str(getattr(args, 'risk_mode', 'moderate') or 'moderate')
        auto_reduce_size = bool(getattr(args, 'auto_reduce_size', False) or False)
        emergency_close = bool(getattr(args, 'emergency_close', False) or False)
        risk_scaling = bool(getattr(args, 'risk_scaling', False) or False)
        risk_trail_stops = bool(getattr(args, 'risk_trail_stops', False) or False)
        
        # Decision matrix based on multiple factors
        action = 'hold'
        priority_score = 0
        rationale = []
        
        # BOT ACTION DEFINITIONS:
        # 'close_full' - Đóng toàn bộ lệnh (100%)
        # 'close_partial_30' - Đóng 30% khối lượng lệnh
        # 'close_partial_50' - Đóng 50% khối lượng lệnh  
        # 'close_partial_70' - Đóng 70% khối lượng lệnh
        # 'reduce_size' - Giảm khối lượng lệnh
        # 'set_tp' - Đặt Take Profit tại mức cụ thể
        # 'set_sl' - Đặt Stop Loss tại mức cụ thể
        # 'set_trailing_sl' - Đặt Trailing Stop Loss
        # 'move_sl_to_be' - Di chuyển SL về breakeven
        # 'hold' - Giữ nguyên lệnh
        
        # 🛡️ ADAPTIVE PIP THRESHOLDS BASED ON RISK MODE
        base_multipliers = {
            'conservative': 0.7,  # Tighter thresholds
            'moderate': 1.0,      # Default thresholds  
            'aggressive': 1.4     # Looser thresholds
        }
        multiplier = float(base_multipliers.get(str(risk_mode), 1.0))
        
        # Define pip-based thresholds (can be made configurable)
        # Different symbols may need different thresholds
        if 'XAU' in symbol or 'XAG' in symbol:  # Gold/Silver
            pip_threshold_large = int(200 * multiplier)
            pip_threshold_medium = int(100 * multiplier)
            pip_threshold_small = int(50 * multiplier)
            pip_threshold_minimal = int(20 * multiplier)
            pip_loss_large = int(-100 * multiplier)
            pip_loss_critical = int(-200 * multiplier)
        elif any(crypto in symbol for crypto in ['BTC', 'ETH', 'SOL', 'LTC', 'ADA', 'DOGE', 'XRP', 'DOT', 'LINK', 'UNI', 'AVAX', 'MATIC', 'ATOM', 'BNB']):  # Crypto
            pip_threshold_large = int(1000 * multiplier)
            pip_threshold_medium = int(500 * multiplier)
            pip_threshold_small = int(200 * multiplier)
            pip_threshold_minimal = int(100 * multiplier)
            pip_loss_large = int(-500 * multiplier)
            pip_loss_critical = int(-1000 * multiplier)
        elif symbol.endswith('JPY'):  # JPY pairs
            pip_threshold_large = int(100 * multiplier)
            pip_threshold_medium = int(50 * multiplier)
            pip_threshold_small = int(30 * multiplier)
            pip_threshold_minimal = int(10 * multiplier)
            pip_loss_large = int(-50 * multiplier)
            pip_loss_critical = int(-100 * multiplier)
        else:  # Major FX pairs
            pip_threshold_large = int(100 * multiplier)
            pip_threshold_medium = int(50 * multiplier)
            pip_threshold_small = int(30 * multiplier)
            pip_threshold_minimal = int(15 * multiplier)
            pip_loss_large = int(-50 * multiplier)
            pip_loss_critical = int(-100 * multiplier)
        
        # 🚨 1. EMERGENCY RISK MANAGEMENT (GUI-controlled)
        # Load risk settings first to check GUI disable flags
        try:
            import os
            import json
            risk_settings_path = 'risk_management/risk_settings.json'
            risk_settings = {}
            if os.path.exists(risk_settings_path):
                with open(risk_settings_path, 'r', encoding='utf-8') as f:
                    risk_settings = json.load(f)
        except Exception:
            risk_settings = {}
        
        # Check if emergency features are disabled by GUI
        disable_emergency_stop = risk_settings.get('disable_emergency_stop', False)
        disable_max_dd_close = risk_settings.get('disable_max_dd_close', False)
        disable_margin_protection = risk_settings.get('disable_margin_protection', False)
        
        # Only apply emergency logic if not disabled by GUI
        if not disable_emergency_stop and not disable_max_dd_close:
            if emergency_close and (drawdown_pct >= max_drawdown or overall_risk == 'HIGH'):
                action = 'close_full'
                priority_score = 100
                rationale.append(f'🚨 KHẨN CẤP: Đóng tất cả - drawdown {drawdown_pct:.1f}% >= {max_drawdown}%')
        
        if not disable_margin_protection:
            if margin_level is not None and margin_level < min_margin_level and margin_level > 0:
                # Chỉ đóng khi có signal ngược, tuân theo logic mới
                if signal_alignment == 'OPPOSITE':
                    action = 'close_full'
                    priority_score = max(priority_score, 95)
                    rationale.append(f'⚠️ Margin thấp ({margin_level:.0f}%) + tín hiệu ngược - đóng lệnh')
        
        # 🎯 2. SIGNAL-BASED POSITION MANAGEMENT (Simplified Logic)
        # Chỉ đóng lệnh khi có tín hiệu ngược, bất kể lãi hay lỗ
        if signal_alignment == 'OPPOSITE' and conf >= 60:
            action = 'close_full'
            priority_score = max(priority_score, 85)
            if pips > 0:
                rationale.append(f'� Lãi ({pips:.1f} pips) + tín hiệu ngược mạnh - đóng toàn bộ')
            else:
                rationale.append(f'� Lỗ ({pips:.1f} pips) + tín hiệu ngược mạnh - đóng toàn bộ')
        
        # 💰 3. PROFIT MANAGEMENT LOGIC (Enhanced with Market Volatility)
        elif pips is not None and pips > 0:
            # 🛡️ CRITICAL SAFETY CHECK: Triple validation for truly profitable position
            actual_profit = float(position.get('profit', 0) or 0)
            pct_equity = float(position.get('pct_equity', 0) or 0)
            
            # 🚨 ENHANCED BUG PREVENTION: Multiple checks to prevent move_sl_to_be on losing positions
            if actual_profit < 0:
                print(f"🚨 BUG PREVENTED [PROFIT CHECK]: {symbol} pips={pips} > 0 but profit=${actual_profit} < 0!")
                print(f"🔧 FORCING to loss logic - preventing move_sl_to_be on losing position")
                action = 'hold'
                priority_score = max(priority_score, 20)
                rationale.append(f'⏳ Lỗ ({actual_profit:.2f}$) - BUG PREVENTED: pips vs profit mismatch')
            elif pct_equity < 0:
                print(f"🚨 BUG PREVENTED [EQUITY CHECK]: {symbol} pips={pips} > 0 but pct_equity={pct_equity}% < 0!")
                print(f"🔧 FORCING to loss logic - preventing move_sl_to_be on negative equity position")
                action = 'hold'
                priority_score = max(priority_score, 20)
                rationale.append(f'⏳ Lỗ ({pct_equity:.2f}%) - BUG PREVENTED: pips vs equity mismatch')
            elif action != 'close_full':  # Nếu chưa có quyết định đóng từ tín hiệu ngược
                # 🎯 HARDCODED SL MANAGEMENT RULES - Không phụ thuộc vào risk_settings
                move_to_be_min = 20      # Lãi tối thiểu để move SL to BE
                move_to_be_max = 50      # Lãi tối đa cho move SL to BE
                trailing_activation = 70  # Lãi kích hoạt trailing stop
                
                if pips >= trailing_activation:
                    # 🛡️ VALIDATION: Check before trailing stop
                    if actual_profit <= 0 or pct_equity <= 0:
                        print(f"🚨 BUG PREVENTED: {symbol} almost set trailing_sl but profit={actual_profit}, equity={pct_equity}%!")
                        action = 'hold'
                        priority_score = max(priority_score, 20)
                        rationale.append(f'⏳ EMERGENCY HOLD - prevented trailing on losing position')
                    else:
                        # Trailing stop for high profits (>= 70 pips) - VALIDATED
                        print(f"✅ VALIDATED TRAILING: {symbol} pips={pips}, profit=${actual_profit}, equity={pct_equity}%")
                        action = 'set_trailing_sl'
                        priority_score = max(priority_score, 60)
                        rationale.append(f'🎯 Lãi cao ({pips:.1f} pips) - kích hoạt trailing stop dựa trên biến động thị trường')
                elif move_to_be_min <= pips <= move_to_be_max:
                    # 🛡️ FINAL SAFETY CHECK: Ultimate validation before move_sl_to_be
                    if actual_profit <= 0 or pct_equity <= 0:
                        print(f"🚨 CRITICAL BUG PREVENTED: {symbol} almost set move_sl_to_be but profit={actual_profit}, equity={pct_equity}%!")
                        print(f"🔧 EMERGENCY OVERRIDE: Switching to hold instead of move_sl_to_be")
                        action = 'hold'
                        priority_score = max(priority_score, 20)
                        rationale.append(f'⏳ EMERGENCY HOLD - prevented move_sl_to_be on losing position')
                    else:
                        # Move to breakeven for moderate profits (20-50 pips) - TRIPLE VALIDATED
                        print(f"✅ VALIDATED MOVE_SL_TO_BE: {symbol} pips={pips}, profit=${actual_profit}, equity={pct_equity}%")
                        action = 'move_sl_to_be'
                        priority_score = max(priority_score, 45)
                        rationale.append(f'📈 Lãi ({pips:.1f} pips) trong vùng {move_to_be_min}-{move_to_be_max} - Move S/L về breakeven')
                else:
                    action = 'hold'
                    priority_score = max(priority_score, 25)
                    rationale.append(f'📊 Giữ nguyên ({pips:.1f} pips) - chưa đủ điều kiện để điều chỉnh SL')
        
        # 📉 4. POSITION HOLDING LOGIC (Simplified)
        # Giữ lệnh trừ khi có tín hiệu ngược (đã xử lý ở trên)
        elif pips is not None and pips < 0:
            if action != 'close_full':  # Nếu chưa có quyết định đóng từ tín hiệu ngược
                action = 'hold'
                priority_score = max(priority_score, 20)
                rationale.append(f'⏳ Lỗ ({pips:.1f} pips) - chờ thị trường phục hồi')
        
        # 📊 5. NEUTRAL POSITION (no significant profit/loss)
        else:  # pips == 0 or very close to breakeven
            if action != 'close_full':  # Nếu chưa có quyết định đóng từ tín hiệu ngược
                action = 'hold'
                priority_score = max(priority_score, 15)
                rationale.append(f'� Giữ nguyên (breakeven)')
        
        # 🎯 6. RISK CONCENTRATION MANAGEMENT (Disabled - chỉ đóng khi có signal ngược)
        
        # Default rationale if none provided
        if not rationale:
            rationale.append(f'📊 Giữ nguyên ({pips:.1f} pips)')
        
        # Calculate proposed SL/TP based on action and risk settings
        proposed_sl = None
        proposed_tp = None
        trailing_config = None
        move_sl_price = None
        move_tp_price = None
        
        def apply_enhanced_sl_protection(candidate_sl: float, pos_dir: str, current_price: float, pip_value: float, 
                                       min_protection_pips: float = 15.0) -> float:
            """
            🛡️ UNIVERSAL ENHANCED SL PROTECTION - applies to ALL symbol types and ALL scenarios
            Ensures SL is never placed dangerously close to current price
            
            Args:
                candidate_sl: The proposed SL price from any calculation method
                pos_dir: 'BUY' or 'SELL'  
                current_price: Current market price
                pip_value: Pip value for the symbol (varies by symbol type)
                min_protection_pips: Minimum pips distance from current price (default: 15)
                
            Returns:
                Protected SL price that maintains minimum distance from current price
            """
            # Calculate minimum allowed distance in price units
            min_distance_price = min_protection_pips * pip_value
            
            if pos_dir == 'BUY':
                # For BUY: SL must be BELOW current price by at least min_protection_pips
                min_allowed_sl = current_price - min_distance_price
                
                # If candidate SL is too close to current price (too high), use minimum allowed
                if candidate_sl > min_allowed_sl:
                    logger.debug(f"🛡️ BUY SL Protection: {candidate_sl:.5f} too close to {current_price:.5f}, using {min_allowed_sl:.5f}")
                    return min_allowed_sl
                else:
                    # Candidate SL is safe (far enough from current price)
                    return candidate_sl
                    
            else:  # SELL
                # For SELL: SL must be ABOVE current price by at least min_protection_pips  
                min_allowed_sl = current_price + min_distance_price
                
                # If candidate SL is too close to current price (too low), use minimum allowed
                if candidate_sl < min_allowed_sl:
                    logger.debug(f"🛡️ SELL SL Protection: {candidate_sl:.5f} too close to {current_price:.5f}, using {min_allowed_sl:.5f}")
                    return min_allowed_sl
                else:
                    # Candidate SL is safe (far enough from current price)
                    return candidate_sl
        
        if action == 'set_sl':
            # 🎯 ENHANCED SL CALCULATION with multiple factors
            if idea_sl:
                # 🛡️ ENHANCED PROTECTION FOR IDEA_SL
                pip_value = get_pip_value(symbol)
                min_sl_buffer_from_entry_pips = 35  # Force minimum - prevent premature stops
                
                if entry_price and pip_value:
                    min_sl_from_entry = min_sl_buffer_from_entry_pips * pip_value
                    
                    if pos_dir == 'BUY':
                        min_allowed_sl = entry_price - min_sl_from_entry
                        candidate_sl = min(idea_sl, min_allowed_sl)
                    else:  # SELL
                        min_allowed_sl = entry_price + min_sl_from_entry
                        candidate_sl = max(idea_sl, min_allowed_sl)
                        
                    # 🛡️ APPLY UNIVERSAL ENHANCED PROTECTION
                    proposed_sl = apply_enhanced_sl_protection(candidate_sl, pos_dir, current_price, pip_value)
                    
                    if proposed_sl != idea_sl:
                        logger.debug(f"🛡️ IDEA SL PROTECTED: {idea_sl:.5f} -> {proposed_sl:.5f}")
                else:
                    # No entry price info - apply protection to raw idea_sl
                    pip_value = get_pip_value(symbol) if pip_value is None else pip_value
                    proposed_sl = apply_enhanced_sl_protection(idea_sl, pos_dir, current_price, pip_value)
            elif current_price and entry_price:
                # Load risk settings from file to get default_sl_pips
                try:
                    import os
                    risk_settings_path = 'risk_management/risk_settings.json'
                    default_sl_pips = 50  # Fallback
                    
                    if os.path.exists(risk_settings_path):
                        with open(risk_settings_path, 'r', encoding='utf-8') as f:
                            risk_settings = json.load(f)
                            default_sl_pips = risk_settings.get('default_sl_pips', 50)
                    
                    # 🎯 ENHANCED PROTECTIVE SL CALCULATION - AVOID ENTRY POINT PROXIMITY
                    pip_value = get_pip_value(symbol)
                    
                    # 🧠 SMART DYNAMIC SL CALCULATION - Market-Based Protection
                    min_sl_buffer_from_entry_pips = calculate_smart_sl_buffer(symbol, current_price, entry_price, pos_dir, conf)

                    
                    # Factor 1: Signal strength adjustment
                    signal_multiplier = 1.5 if conf >= 80 else 1.2 if conf >= 70 else 1.0 if conf >= 50 else 0.8
                    
                    # Factor 2: Risk mode adjustment
                    risk_multiplier = 0.8 if risk_mode == 'conservative' else 1.2 if risk_mode == 'aggressive' else 1.0
                    
                    # Factor 3: Volatility adjustment
                    volatility_multiplier = 1.3 if overall_risk == 'HIGH' else 0.8 if overall_risk == 'LOW' else 1.0
                    
                    # Calculate adaptive SL distance with MINIMUM SAFETY BUFFER
                    base_sl_pips = max(default_sl_pips, min_sl_buffer_from_entry_pips + 10)  # Ensure minimum distance
                    adaptive_sl_pips = base_sl_pips * signal_multiplier * risk_multiplier * volatility_multiplier
                    
                    # 🛡️ SAFETY CHECK: Ensure S/L is NOT too close to entry point
                    entry_to_current_pips = abs(current_price - entry_price) / pip_value
                    
                    # Calculate candidate SL based on case type
                    if pips < 0 and abs(pips) < pip_threshold_small:  # Small loss case - use entry-based protection
                        min_sl_from_entry = min_sl_buffer_from_entry_pips * pip_value
                        
                        if pos_dir == 'BUY':
                            candidate_sl = entry_price - min_sl_from_entry
                        else:  # SELL
                            candidate_sl = entry_price + min_sl_from_entry
                    else:
                        # Normal case - use adaptive calculation
                        if pos_dir == 'BUY':
                            candidate_sl = current_price - (adaptive_sl_pips * pip_value)
                        else:  # SELL
                            candidate_sl = current_price + (adaptive_sl_pips * pip_value)
                    
                    # �️ APPLY UNIVERSAL ENHANCED PROTECTION TO ALL CASES
                    proposed_sl = apply_enhanced_sl_protection(candidate_sl, pos_dir, current_price, pip_value)
                        
                except Exception as e:
                    # 🛡️ UNIVERSAL FALLBACK LOGIC with Enhanced Protection
                    pip_value = get_pip_value(symbol)
                    min_sl_buffer_pips = 35  # Minimum safe buffer - prevent premature stops
                    
                    # Use larger distance for protection even in fallback
                    sl_distance = max(pip_threshold_small * 0.8, min_sl_buffer_pips + 5) if signal_alignment == 'ALIGNED' else max(pip_threshold_small * 0.5, min_sl_buffer_pips)
                    
                    # Calculate candidate SL
                    if pos_dir == 'BUY':
                        # Base calculation from current price
                        candidate_sl_base = current_price - (sl_distance * pip_value)
                        
                        # Entry protection if available
                        if entry_price:
                            min_sl_from_entry = min_sl_buffer_pips * pip_value
                            candidate_sl_entry = entry_price - min_sl_from_entry
                            candidate_sl = min(candidate_sl_base, candidate_sl_entry)
                        else:
                            candidate_sl = candidate_sl_base
                            
                    else:  # SELL position
                        # Base calculation from current price
                        candidate_sl_base = current_price + (sl_distance * pip_value)
                        
                        # Entry protection if available
                        if entry_price:
                            min_sl_from_entry = min_sl_buffer_pips * pip_value
                            candidate_sl_entry = entry_price + min_sl_from_entry
                            candidate_sl = max(candidate_sl_base, candidate_sl_entry)
                        else:
                            candidate_sl = candidate_sl_base
                    
                    # �️ APPLY UNIVERSAL ENHANCED PROTECTION TO FALLBACK
                    proposed_sl = apply_enhanced_sl_protection(candidate_sl, pos_dir, current_price, pip_value)
                
        elif action == 'set_tp':
            # Calculate appropriate TP based on risk settings and signal
            if idea_tp:
                proposed_tp = idea_tp
            elif current_price and entry_price:
                # Load risk settings from file to get default_tp_pips
                try:
                    import os
                    risk_settings_path = 'risk_management/risk_settings.json'
                    default_tp_pips = 100  # Fallback
                    
                    if os.path.exists(risk_settings_path):
                        with open(risk_settings_path, 'r', encoding='utf-8') as f:
                            risk_settings = json.load(f)
                            default_tp_pips = risk_settings.get('default_tp_pips', 100)
                    
                    # Use default_tp_pips from risk settings instead of dynamic calculation
                    pip_value = get_pip_value(symbol)
                    
                    if pos_dir == 'BUY':
                        # For BUY positions, TP should be above current price
                        proposed_tp = current_price + (default_tp_pips * pip_value)
                    else:  # SELL position
                        # For SELL positions, TP should be below current price
                        proposed_tp = current_price - (default_tp_pips * pip_value)
                        
                except Exception as e:
                    # Fallback to original logic if risk settings can't be loaded
                    if pos_dir == 'BUY':
                        tp_distance = pip_threshold_medium * 1.5 if signal_alignment == 'ALIGNED' else pip_threshold_small
                        pip_value = get_pip_value(symbol)
                        proposed_tp = current_price + (tp_distance * pip_value)
                    else:
                        tp_distance = pip_threshold_medium * 1.5 if signal_alignment == 'ALIGNED' else pip_threshold_small
                        pip_value = get_pip_value(symbol)
                        proposed_tp = current_price - (tp_distance * pip_value)
                    
        elif action == 'move_sl_to_be':
            # Move SL to breakeven (entry price)
            move_sl_price = entry_price
            
        elif action == 'set_trailing_sl':
            # 🎯 ENHANCED TRAILING SL với biến động thị trường từ trendline_sr
            if current_price and entry_price:
                # 🎯 HARDCODED TRAILING SETTINGS - Không phụ thuộc vào risk_settings
                trailing_min = 20        # Khoảng cách trailing tối thiểu (pips)
                trailing_max = 50        # Khoảng cách trailing tối đa (pips)  
                use_volatility = True    # Luôn sử dụng market volatility
                
                # Get market volatility data từ trendline_sr
                volatility = 0.6  # Default moderate volatility
                trend_strength = 0.5  # Default moderate trend
                trend_direction = 'Sideways'  # Default trend
                
                # Load volatility từ trendline_sr file
                try:
                    import json
                    sr_file = f"trendline_sr/{symbol}_H1_trendline_sr.json"
                    if os.path.exists(sr_file):
                        with open(sr_file, 'r', encoding='utf-8') as f:
                            sr_data = json.load(f)
                        volatility = min(max(sr_data.get('volatility', 0.6), 0.1), 2.0)
                        trend_strength = sr_data.get('trend_strength', 0.5)
                        trend_direction = sr_data.get('trend_direction', 'Sideways')
                        logger.debug(f"📊 Market data loaded: volatility={volatility:.3f}, trend={trend_direction}")
                except Exception as e:
                    logger.debug(f"⚠️ Could not load market data: {e}")
                    pass  # Use defaults
                
                # Calculate dynamic trailing distance dựa trên biến động thị trường
                if use_volatility:
                    # Base trailing distance from settings range
                    base_distance = (trailing_min + trailing_max) / 2  # Mid-point
                    
                    # Adjust for volatility (high volatility = wider trailing)
                    if volatility > 1.2:  # High volatility
                        trail_distance_pips = trailing_max * 0.9  # Near maximum
                        logger.debug(f"🔥 High volatility ({volatility:.3f}) - wide trailing: {trail_distance_pips}")
                    elif volatility > 0.8:  # Moderate volatility  
                        trail_distance_pips = base_distance  # Medium trailing
                        logger.debug(f"⚖️ Moderate volatility ({volatility:.3f}) - standard trailing: {trail_distance_pips}")
                    else:  # Low volatility
                        trail_distance_pips = trailing_min * 1.2  # Tighter trailing
                        logger.debug(f"🔒 Low volatility ({volatility:.3f}) - tight trailing: {trail_distance_pips}")
                    
                    # Adjust for trend strength (strong trend = tighter trailing)
                    if trend_strength > 0.7:
                        trail_distance_pips *= 0.8  # Tighter for strong trends
                        logger.debug(f"💪 Strong trend ({trend_strength:.3f}) - tightened to: {trail_distance_pips}")
                    elif trend_strength < 0.3:
                        trail_distance_pips *= 1.3  # Looser for weak trends
                        logger.debug(f"😴 Weak trend ({trend_strength:.3f}) - widened to: {trail_distance_pips}")
                else:
                    # Fixed trailing distance if volatility disabled
                    trail_distance_pips = trailing_min
                
                # Ensure within bounds
                trail_distance_pips = max(trailing_min, min(trail_distance_pips, trailing_max))
                
                pip_value = get_pip_value(symbol)
                trail_distance = trail_distance_pips * pip_value
                
                if pos_dir == 'BUY':
                    # For BUY positions, trailing distance from current high
                    activate_price = current_price - (trail_distance * 0.5)
                else:  # SELL position
                    # For SELL positions, trailing distance from current low
                    activate_price = current_price + (trail_distance * 0.5)
                
                trailing_config = {
                    'trail_distance_pips': round(trail_distance_pips, 1),
                    'trail_distance': round(trail_distance, 5),
                    'activate_price': round(activate_price, 5),
                    'mode': f'volatility_adaptive_{trend_direction.lower()}',
                    'symbol': symbol,
                    'direction': pos_dir,
                    'market_conditions': {
                        'volatility': round(volatility, 3),
                        'trend_strength': round(trend_strength, 3),
                        'trend_direction': trend_direction,
                        'trailing_range': f'{trailing_min}-{trailing_max} pips'
                    }
                }
        
        return {
            'action': action,
            'priority_score': priority_score,
            'rationale': " | ".join(rationale),
            'proposed_sl': proposed_sl,
            'proposed_tp': proposed_tp,
            'move_sl_price': move_sl_price,
            'move_tp_price': move_tp_price,
            'trailing_config': trailing_config,
            'pips': pips,
            'risk_factors': {
                'signal_alignment': signal_alignment,
                'confidence': conf,
                'position_risk_pct': position_risk_pct,
                'overall_risk': overall_risk,
                'pips_profit_loss': pips,
                'risk_mode': risk_mode,
                'risk_scaling_active': risk_scaling,
                'auto_reduce_enabled': auto_reduce_size
            }
        }
    except Exception as e:
        return {
            'action': 'hold',
            'priority_score': 0,
            'rationale': f'❌ Lỗi quyết định: {str(e)}',
            'proposed_sl': None,
            'proposed_tp': None,
            'trailing_config': None,
            'pips': 0,
            'risk_factors': {}
        }

def save_action_execution_history(actions: list, symbol: str = None) -> None:
    """Save action execution history to reports/execution_reports.json"""
    try:
        import os
        import json
        from datetime import datetime
        
        if not actions:
            return
            
        # Create reports directory
        os.makedirs("reports", exist_ok=True)
        reports_path = "reports/execution_reports.json"
        
        # Load existing data
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
        
        # Create execution report
        report = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol or 'MULTIPLE',
            'actions_count': len(actions),
            'actions': actions[:10],  # Store only first 10 actions to prevent huge files
            'action_types': {},
            'analysis_type': 'position_management',
            'source': 'comprehensive_aggregator'
        }
        
        # Count action types
        for action in actions:
            action_type = action.get('type', action.get('action', 'unknown'))
            report['action_types'][action_type] = report['action_types'].get(action_type, 0) + 1
        
        # Add to reports
        reports_data['execution_reports'].append(report)
        
        # Keep only last 100 reports
        max_reports = 100
        if len(reports_data['execution_reports']) > max_reports:
            reports_data['execution_reports'] = reports_data['execution_reports'][-max_reports:]
        
        # Update metadata
        reports_data['metadata']['last_updated'] = datetime.now().isoformat()
        reports_data['metadata']['total_reports'] = len(reports_data['execution_reports'])
        
        # Save to file
        with open(reports_path, 'w', encoding='utf-8') as f:
            json.dump(reports_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📊 Action execution history saved: {len(actions)} actions for {symbol}")
        
    except Exception as e:
        logger.error(f"❌ Error saving action execution history: {e}")

def analyze_positions(scan: dict, current_symbol: str, current_signal: str, signal_results: dict = None) -> dict:
    """Enhanced position analysis with comprehensive risk assessment."""
    out = {"positions": [], "summary": {}, "risk_metrics": {}}
    try:
        # Fix: equity is in scan['account']['equity'], not scan['account_info']['balance_metrics']['equity']
        equity = (scan.get('account') or {}).get('equity') or 0.0
        balance = (scan.get('account') or {}).get('balance') or 0.0
        margin_level = (scan.get('account') or {}).get('margin_level') or 0.0
        pos_list = scan.get('active_positions') or []
        
        total_unreal = 0.0
        sym_exposure: dict[str, float] = {}
        positions_data = []
        
        for p in pos_list:
            if not isinstance(p, dict):
                continue
            sym = p.get('symbol') or p.get('Symbol')
            vol = p.get('volume') or p.get('Volume') or 0
            profit = p.get('profit') or p.get('Profit') or 0.0
            direction = p.get('type') or p.get('Type')  # buy/sell numeric maybe
            price_open = p.get('price_open') or p.get('price_open', 0)
            price_current = p.get('price_current') or p.get('price_current', 0)
            
            dir_txt = str(direction).lower()
            
            # Handle None/missing direction - try to infer from profit/price movement
            if direction is None or dir_txt in ('none', 'null'):
                # Infer direction from price movement vs profit
                if price_open and price_current and profit != 0:
                    price_movement = price_current - price_open  # positive = price went up
                    profit_sign = 1 if float(profit) > 0 else -1
                    # If price went up and profit is positive, likely BUY
                    # If price went up and profit is negative, likely SELL
                    if (price_movement > 0 and profit_sign > 0) or (price_movement < 0 and profit_sign < 0):
                        dir_txt = 'BUY'
                    else:
                        dir_txt = 'SELL'
                else:
                    # Default fallback
                    dir_txt = 'BUY'
            elif dir_txt in ('0','buy','long'): 
                dir_txt = 'BUY'
            elif dir_txt in ('1','sell','short'): 
                dir_txt = 'SELL'
            else: 
                dir_txt = dir_txt.upper()
            
            total_unreal += float(profit)
            if isinstance(sym, str):
                sym_exposure[sym] = sym_exposure.get(sym, 0.0) + float(profit)
            
            # Enhanced risk calculation per position
            pct_equity = (float(profit)/equity*100) if equity else 0.0
            pct_balance = (float(profit)/balance*100) if balance else 0.0
            
            # Position size risk
            position_value = abs(float(vol) * float(price_current)) if price_current else 0
            position_risk_pct = (position_value / equity * 100) if equity > 0 else 0
            
            # Unrealized P&L as percentage of position value
            pnl_vs_position = (abs(float(profit)) / position_value * 100) if position_value > 0 else 0
            
            # Calculate pips profit/loss
            pips = calculate_pips(sym, price_open, price_current, dir_txt) if price_open and price_current else 0
            
            suggestion = []
            risk_level = 'LOW'
            
            # Enhanced suggestion logic based on multiple factors
            if pct_equity >= 5:
                suggestion.append("Lãi lớn — nên chốt 50% để bảo toàn lợi nhuận")
                risk_level = 'PROFIT_HIGH'
            elif pct_equity >= 3:
                suggestion.append("Đang lãi — cân nhắc chốt 30% hoặc trailing")
                risk_level = 'PROFIT_MEDIUM'
            elif pct_equity >= 1:
                suggestion.append("Lãi nhẹ — đặt trailing SL để bảo vệ")
                risk_level = 'PROFIT_LOW'
            
            if pct_equity <= -10:
                suggestion.append("LỖ NẶNG — ưu tiên đóng lệnh ngay lập tức")
                risk_level = 'LOSS_CRITICAL'
            elif pct_equity <= -5:
                suggestion.append("Lỗ lớn — xem xét đóng hoặc siết SL gấp")
                risk_level = 'LOSS_HIGH'
            elif pct_equity <= -2:
                suggestion.append("Đang lỗ — ưu tiên giảm rủi ro (đặt/siết SL)")
                risk_level = 'LOSS_MEDIUM'
            
            # Position size risk warning
            if position_risk_pct > 10:
                suggestion.append("CẢNH BÁO: Khối lượng lệnh quá lớn (>10% equity)")
            elif position_risk_pct > 5:
                suggestion.append("Cảnh báo: Khối lượng lệnh cao (>5% equity)")
            
            # Normalize symbols for comparison (remove _m suffix)
            normalized_sym = sym[:-2] if isinstance(sym, str) and sym.endswith('_m') else sym
            normalized_current = current_symbol[:-2] if isinstance(current_symbol, str) and current_symbol.endswith('_m') else current_symbol
            
            # Get signal for this specific symbol
            symbol_signal = 'NEUTRAL'
            if signal_results and normalized_sym in signal_results:
                symbol_signal = signal_results[normalized_sym].get('signal', 'NEUTRAL')
            elif normalized_sym == normalized_current:
                symbol_signal = current_signal
            
            signal_alignment = 'NEUTRAL'
            if symbol_signal != 'NEUTRAL':
                if symbol_signal == 'BUY' and dir_txt == 'SELL':
                    suggestion.append(f"⚠️ XUNG ĐỘT: Tín hiệu BUY ({symbol_signal}) nhưng đang SELL — xem xét đóng/hedge")
                    signal_alignment = 'OPPOSITE'
                elif symbol_signal == 'SELL' and dir_txt == 'BUY':
                    suggestion.append(f"⚠️ XUNG ĐỘT: Tín hiệu SELL ({symbol_signal}) nhưng đang BUY — xem xét đóng/hedge")
                    signal_alignment = 'OPPOSITE'
                elif symbol_signal == dir_txt:
                    suggestion.append(f"✅ ĐỒNG PHA: Tín hiệu {symbol_signal} và lệnh {dir_txt} cùng hướng")
                    signal_alignment = 'ALIGNED'
            
            position_data = {
                "symbol": sym,
                "direction": dir_txt,
                "volume": vol,
                "profit": float(profit),
                "pct_equity": round(pct_equity, 2),
                "pct_balance": round(pct_balance, 2),
                "position_risk_pct": round(position_risk_pct, 2),
                "pnl_vs_position": round(pnl_vs_position, 2),
                "price_open": price_open,
                "price_current": price_current,
                "pips": round(pips, 1),
                "risk_level": risk_level,
                "signal_alignment": signal_alignment,
                "current_signal": symbol_signal,
                "suggest": "; ".join(suggestion) if suggestion else "Giữ nguyên"
            }
            positions_data.append(position_data)
            out["positions"].append(position_data)
        
        # Calculate comprehensive risk metrics
        risk_metrics = calculate_risk_metrics(scan, positions_data)
        
        out["summary"] = {
            "positions_count": len(out["positions"]),
            "total_unrealized": round(total_unreal, 2),
            "total_unrealized_pct": round(total_unreal / equity * 100, 2) if equity > 0 else 0,
            "equity": equity,
            "balance": balance,
            "margin_level": margin_level,
            "symbols_positive": [s for s,v in sym_exposure.items() if v>0],
            "symbols_negative": [s for s,v in sym_exposure.items() if v<0],
            "worst_performer": min(sym_exposure.items(), key=lambda x: x[1]) if sym_exposure else None,
            "best_performer": max(sym_exposure.items(), key=lambda x: x[1]) if sym_exposure else None
        }
        
        out["risk_metrics"] = risk_metrics
        
        # 📊 EXECUTION HISTORY LOGGING: Save all generated actions for this analysis
        try:
            # Collect all actions from out["positions"] 
            all_actions = []
            for pos in out.get("positions", []):
                if pos.get("suggest") and pos.get("suggest") != "Giữ nguyên":
                    action_record = {
                        "symbol": pos.get("symbol"),
                        "action": pos.get("suggest"),
                        "profit": pos.get("profit", 0),
                        "pct_equity": pos.get("pct_equity", 0),
                        "current_signal": pos.get("current_signal", ""),
                        "risk_level": pos.get("risk_level", ""),
                        "direction": pos.get("direction", ""),
                        "volume": pos.get("volume", 0),
                        "pips": pos.get("pips", 0)
                    }
                    all_actions.append(action_record)
            
            # Save execution history if we have actions
            if all_actions:
                save_action_execution_history(all_actions, current_symbol)
                logger.debug(f"💾 Saved {len(all_actions)} position actions to execution history")
        except Exception as e:
            logger.error(f"Failed to save execution history: {e}")
        
    except Exception:
        pass
    return out
    
def _post_clean_english_reports():
    """Final pass to scrub any residual Vietnamese phrases from EN reports (file-based)."""
    repl_map = {
        'thu hẹp (nén)': 'narrowing (squeeze)',
        'có thể sắp bứt phá': 'potential breakout soon',
        'độ rộng': 'width',
        'xếp chồng tăng': 'bullish stack',
        'xếp chồng up': 'bullish stack',
        'xếp chồng giảm': 'bearish stack'
    }
    try:
        for fp in glob.glob(os.path.join(CFG.OUT, '*_report_en_*.txt')):
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    txt = f.read()
                new_txt = txt
                for k,v in repl_map.items():
                    new_txt = new_txt.replace(k, v)
                if new_txt != txt:
                    with open(fp, 'w', encoding='utf-8') as f:
                        f.write(new_txt)
            except Exception:
                continue
    except Exception:
        logger.warning('English post-translation step failed')


def overwrite_json_safely(path: str, data: Any, backup: bool = False) -> None:
    """Write JSON atomically; if backup requested and file exists, create .bak once.
    This is a lightweight fallback implementation in case the original util is missing."""
    try:
        if backup and os.path.exists(path) and not os.path.exists(path + ".bak"):
            try:
                import shutil
                shutil.copy2(path, path + ".bak")
            except Exception:
                pass
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

# Global pattern family weights and fallback polarity for comprehensive price patterns
PATTERN_FAMILY_WEIGHT = {
    # Classical reversal/continuation
    "double_top": 1.00, "double_bottom": 1.00,
    "head_and_shoulders": 1.05, "inverse_head_and_shoulders": 1.05,
    "triple_top": 0.95, "triple_bottom": 0.95,
    "bullish_flag": 0.95, "bearish_flag": 0.95,
    "bullish_pennant": 0.95, "bearish_pennant": 0.95,
    "rising_wedge": 0.90, "falling_wedge": 0.90,
    "ascending_triangle": 0.90, "descending_triangle": 0.90,
    "symmetrical_triangle": 0.60, "rectangle": 0.60,

    # Cup/Handle
    "cup_and_handle": 0.95, "inverse_cup_and_handle": 0.95,

    # Channel
    "ascending_channel": 0.80, "descending_channel": 0.80, "horizontal_channel": 0.50,

    # Breakout / Gap
    "bullish_breakout": 1.05, "bearish_breakout": 1.05,
    "gap_up": 0.70, "gap_down": 0.70,

    # Candle (light weight because already handled elsewhere)
    "bullish_engulfing": 0.50, "bearish_engulfing": 0.50,
    "hammer": 0.45, "shooting_star": 0.45,
}

def ffloat(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def ensure_dir(path: str) -> None:
    try:
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    except Exception:
        pass

# Make sure output directory exists
try:
    ensure_dir(getattr(CFG, 'OUT', os.path.join(os.getcwd(), 'analysis_results')))
except Exception:
    pass

def ts_now() -> str:
    try:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    except Exception:
        return str(int(datetime.now().timestamp()))

def load_json(fp: Optional[str]) -> Any:
    try:
        if not fp or not os.path.exists(fp):
            return None
        if fp.endswith('.gz'):
            with gzip.open(fp, 'rt', encoding='utf-8') as f:
                return json.load(f)
        with open(fp, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def compute_patterns_aggregate(symbol: str, tf_list: List[str]) -> Dict[str, Any]:
    """Lightweight stub: try to summarize best pattern confidences if files exist."""
    try:
        best: Dict[str, Tuple[str, float]] = {}
        for tf in tf_list:
            # attempt to locate a best pattern file
            cand = []
            base = CFG.PPRICE
            for pat in (f"{symbol}_{tf}_best.json", f"{symbol}_{tf}_patterns.json", f"{tf}_best.json"):
                p = os.path.join(base, symbol, pat)
                if os.path.exists(p):
                    cand.append(p)
            if not cand:
                # fallback to glob search
                patt = os.path.join(base, symbol, f"*{tf}*best*.json*")
                cand = glob.glob(patt)
            if not cand:
                continue
            data = load_json(cand[-1])
            if isinstance(data, dict):
                typ = (data.get('type') or data.get('pattern') or '').strip()
                conf = data.get('confidence') or data.get('confidence_pct') or data.get('score')
                try:
                    cf = float(conf)
                    if cf <= 1.5:
                        cf *= 100.0
                except Exception:
                    cf = 0.0
                best[tf] = (typ, max(0.0, min(100.0, cf)))
        if not best:
            return {}
        # pick top by confidence
        top_tf = max(best.items(), key=lambda kv: kv[1][1])[0]
        return {"aggregate": {"top_tf": top_tf, "direction": best[top_tf][0], "confidence": best[top_tf][1]}}
    except Exception:
        return {}

def _load_best_pattern(symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    # Try common filenames and glob search
    base = CFG.PPRICE
    for pat in (f"{symbol}_{tf}_best.json", f"{tf}_best.json"):
        p = os.path.join(base, symbol, pat)
        if os.path.exists(p):
            data = load_json(p)
            if isinstance(data, dict):
                return data
    patt = os.path.join(base, symbol, f"*{tf}*best*.json*")
    files = glob.glob(patt)
    if files:
        return load_json(sorted(files)[-1])
    return None

class Extract:
    # Max bars to look back when last bar has null for an indicator (warm-up periods)
    LOOKBACK_BARS = 5

    @staticmethod
    def _last_from_list(ind: Any, key: str, default: Optional[float] = None) -> Optional[float]:
        """Return latest non-null value (prefer bar flagged current:true) within LOOKBACK_BARS from end."""
        try:
            if not isinstance(ind, list) or not ind or len(ind) < Extract.LOOKBACK_BARS:
                return default
            # Prefer explicit current bar if present at tail region
            tail = ind[-Extract.LOOKBACK_BARS:]
            current_rows = [r for r in reversed(tail) if isinstance(r, dict) and r.get('current')]
            scan_rows = current_rows if current_rows else list(reversed(tail))
            for row in scan_rows:
                if isinstance(row, dict) and key in row and row.get(key) is not None:
                    return ffloat(row.get(key), default)
            return default
        except Exception:
            return default

    @staticmethod
    def _last_two_from_list(ind: Any, key: str) -> Tuple[Optional[float], Optional[float]]:
        """Get previous and current numeric values by key from the last two bars."""
        try:
            if isinstance(ind, list) and len(ind) >= 2 and isinstance(ind[-1], dict) and isinstance(ind[-2], dict):
                a = ind[-2]
                b = ind[-1]
                va = ffloat(a.get(key), None) if key in a and a.get(key) is not None else None
                vb = ffloat(b.get(key), None) if key in b and b.get(key) is not None else None
                return va, vb
            return None, None
        except Exception:
            return None, None

    # -------- Basic value helpers from candles/indicators --------
    @staticmethod
    def last_close(candles: Any) -> Optional[float]:
        try:
            if isinstance(candles, dict):
                for key in ("rates", "candles", "data", "bars"):
                    seq = candles.get(key)
                    if isinstance(seq, list) and seq:
                        last = seq[-1]
                        if isinstance(last, dict):
                            v = last.get("close") or last.get("c")
                            return ffloat(v, None)
            elif isinstance(candles, list) and candles:
                last = candles[-1]
                if isinstance(last, dict):
                    v = last.get("close") or last.get("c")
                    return ffloat(v, None)
        except Exception:
            return None
        return None

    @staticmethod
    def _get_price_from_ind(ind: Any) -> Optional[float]:
        """Best-effort to obtain the latest close price from an indicator/candles structure.

        Accepts:
        - list of bar dicts (prefer rows marked current:true within last LOOKBACK_BARS)
        - single dict snapshot with 'close'/'c' or nested under common keys
        - fallback: use last numeric 'close'/'c' in tail section
        """
        try:
            # List-of-bars layout
            if isinstance(ind, list) and ind:
                tail = ind[-Extract.LOOKBACK_BARS:]
                # Prefer explicit current bar
                for row in reversed(tail):
                    if isinstance(row, dict) and row.get('current'):
                        v = row.get('close') or row.get('c')
                        if v is not None:
                            return ffloat(v, None)
                # Fallback: last non-null in tail
                for row in reversed(tail):
                    if isinstance(row, dict):
                        v = row.get('close') or row.get('c')
                        if v is not None:
                            return ffloat(v, None)
                return None
            # Dict snapshot layout
            if isinstance(ind, dict):
                # Direct fields
                v = ind.get('close') or ind.get('c')
                if v is not None:
                    return ffloat(v, None)
                # Nested under common containers
                for key in ("rates", "candles", "data", "bars"):
                    seq = ind.get(key)
                    if isinstance(seq, list) and seq:
                        last = seq[-1]
                        if isinstance(last, dict):
                            vv = last.get('close') or last.get('c')
                            if vv is not None:
                                return ffloat(vv, None)
            return None
        except Exception:
            return None

    @staticmethod
    def stoch_k(ind: Any) -> Optional[float]:
        """Extract stochastic %K from common exporter keys."""
        try:
            # Column-style last rows
            val = (
                Extract._last_from_list(ind, "StochK_14_3")
                or Extract._last_from_list(ind, "StochK")
                or Extract._last_from_list(ind, "%K")
                or Extract._last_from_list(ind, "stoch_k")
            )
            if val is not None:
                return val
            # Dict snapshot variants
            if isinstance(ind, dict):
                for k in ("StochK", "stoch_k", "%K"):
                    v = ind.get(k)
                    if isinstance(v, list) and v:
                        return ffloat(v[-1])
                    if isinstance(v, (int, float)):
                        return ffloat(v)
            return None
        except Exception:
            return None

    @staticmethod
    def stoch_d(ind: Any) -> Optional[float]:
        """Extract stochastic %D from common exporter keys."""
        try:
            val = (
                Extract._last_from_list(ind, "StochD_14_3")
                or Extract._last_from_list(ind, "StochD")
                or Extract._last_from_list(ind, "%D")
                or Extract._last_from_list(ind, "stoch_d")
            )
            if val is not None:
                return val
            if isinstance(ind, dict):
                for k in ("StochD", "stoch_d", "%D"):
                    v = ind.get(k)
                    if isinstance(v, list) and v:
                        return ffloat(v[-1])
                    if isinstance(v, (int, float)):
                        return ffloat(v)
            return None
        except Exception:
            return None

    @staticmethod
    def _last_key(ind: Any, keys: List[str]) -> Optional[float]:
        if not isinstance(ind, list) or not ind:
            return None
        tail = ind[-Extract.LOOKBACK_BARS:]
        # Prefer current:true rows first
        ordered = [r for r in reversed(tail) if isinstance(r, dict) and r.get('current')] + [r for r in reversed(tail) if isinstance(r, dict) and not r.get('current')]
        seen = set()
        for row in ordered:
            if id(row) in seen: continue
            seen.add(id(row))
            for k in keys:
                if k in row and row.get(k) is not None:
                    return ffloat(row.get(k), None)
        return None

    @staticmethod
    def _last_pair(ind: Any, k_up: List[str], k_mid: Optional[List[str]] = None, k_low: Optional[List[str]] = None) -> Optional[Dict[str, float]]:
        if not isinstance(ind, list) or not ind or not isinstance(ind[-1], dict):
            return None
        row = ind[-1]
        def first(keys: Optional[List[str]]) -> Optional[float]:
            if not keys:
                return None
            for k in keys:
                if k in row and row.get(k) is not None:
                    return ffloat(row.get(k), None)
            return None
        up = first(k_up); mid = first(k_mid); lo = first(k_low)
        out = {}
        if up is not None: out['upper'] = up
        if mid is not None: out['middle'] = mid
        if lo is not None: out['lower'] = lo
        return out if out else None

    # -------- Indicator readers from indicator_output --------
    @staticmethod
    def rsi(ind: Any, rsi_period: Optional[int] = None) -> Optional[float]:
        """Extract RSI value with dynamic parameter detection"""
        if rsi_period:
            # Try dynamic template first
            template_keys = [f"RSI{rsi_period}", f"RSI_{rsi_period}"]
            for key in template_keys:
                val = Extract._last_key(ind, [key])
                if val is not None:
                    return val
        # Fallback to common variants
        return Extract._last_key(ind, ["RSI14", "RSI", "RSI_14", "rsi", "rsi_14"])

    @staticmethod
    def macd_hist(ind: Any, macd_fast: Optional[int] = None, macd_slow: Optional[int] = None, macd_signal: Optional[int] = None) -> Optional[float]:
        """Extract MACD histogram with dynamic parameter detection"""
        if macd_fast and macd_slow and macd_signal:
            # Try dynamic template first
            template_keys = [f"MACD_hist_{macd_fast}_{macd_slow}_{macd_signal}", f"MACD_{macd_fast}_{macd_slow}_{macd_signal}_hist"]
            for key in template_keys:
                val = Extract._last_key(ind, [key])
                if val is not None:
                    return val
        # Fallback to common variants
        return Extract._last_key(ind, ["MACD_hist_12_26_9", "MACD_hist", "macd_hist"])

    @staticmethod
    def adx(ind: Any, adx_period: Optional[int] = None) -> Optional[float]:
        """Extract ADX value with dynamic parameter detection"""
        if adx_period:
            # Try dynamic template first
            template_keys = [f"ADX{adx_period}", f"ADX_{adx_period}"]
            for key in template_keys:
                val = Extract._last_key(ind, [key])
                if val is not None:
                    return val
        # Fallback to common variants
        return Extract._last_key(ind, ["ADX_14", "ADX", "adx"])

    @staticmethod
    def _last_key_with_params(ind: Any, template: str, params: Dict[str, Any]) -> Optional[float]:
        """Extract value using template with dynamic parameters, e.g. 'BB_Upper_{bb_win}_{bb_dev}'"""
        try:
            # Format template with params - convert floats to ints when they are whole numbers
            format_params = {}
            for k, v in params.items():
                if isinstance(v, float) and v.is_integer():
                    format_params[k] = int(v)
                else:
                    format_params[k] = v
            field_name = template.format(**format_params)
            print(f"DEBUG: Template '{template}' + params {format_params} = field '{field_name}'")
            result = Extract._last_from_list(ind, field_name)
            print(f"DEBUG: Field '{field_name}' result: {result}")
            return result
        except Exception as e:
            print(f"DEBUG: Template '{template}' failed: {e}")
            return None

    @staticmethod
    def _last_pair_with_params(ind: Any, template_up: str, template_mid: str, template_low: str, params: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract upper/middle/lower values using templates with dynamic parameters"""
        try:
            up = Extract._last_key_with_params(ind, template_up, params)
            mid = Extract._last_key_with_params(ind, template_mid, params)
            low = Extract._last_key_with_params(ind, template_low, params)
            if None not in (up, mid, low):
                return {"upper": up, "middle": mid, "lower": low}
            return None
        except Exception:
            return None

    @staticmethod
    def stochastic(ind: Any, stoch_k: Optional[int] = None, stoch_d: Optional[int] = None) -> Optional[Dict[str, float]]:
        """Extract Stochastic values with dynamic parameter detection"""
        if not isinstance(ind, list) or not ind or not isinstance(ind[-1], dict):
            return None
        row = ind[-1]
        
        # Try dynamic templates first
        k_val = None
        d_val = None
        if stoch_k and stoch_d:
            template_keys_k = [f"StochK_{stoch_k}_{stoch_d}", f"Stoch_K_{stoch_k}_{stoch_d}"]
            template_keys_d = [f"StochD_{stoch_k}_{stoch_d}", f"Stoch_D_{stoch_k}_{stoch_d}"]
            for key in template_keys_k:
                k_val = row.get(key)
                if k_val is not None:
                    break
            for key in template_keys_d:
                d_val = row.get(key)
                if d_val is not None:
                    break
        
        # Fallback to common variants
        if k_val is None:
            k_val = row.get("StochK_14_3")
        if d_val is None:
            d_val = row.get("StochD_14_3")
            
        out: Dict[str, float] = {}
        if k_val is not None:
            out['k'] = ffloat(k_val, None)
        if d_val is not None:
            out['d'] = ffloat(d_val, None)
        return out if out else None

    @staticmethod
    def stochrsi(ind: Any, stochrsi_period: Optional[int] = None) -> Optional[float]:
        """Extract StochRSI value with dynamic parameter detection"""
        if stochrsi_period:
            # Try dynamic template first
            template_keys = [f"StochRSI{stochrsi_period}", f"StochRSI_{stochrsi_period}"]
            for key in template_keys:
                val = Extract._last_key(ind, [key])
                if val is not None:
                    return val
        # Fallback to common variants
        return Extract._last_key(ind, ["StochRSI_14", "stochrsi", "StochRSI"])

    @staticmethod
    def atr_latest(ind: Any, atr_period: Optional[int] = None) -> Optional[float]:
        """Extract ATR value with dynamic parameter detection"""
        if atr_period:
            # Try dynamic template first
            template_keys = [f"ATR{atr_period}", f"ATR_{atr_period}"]
            for key in template_keys:
                val = Extract._last_key(ind, [key])
                if val is not None:
                    return val
        # Fallback to common variants
        return Extract._last_key(ind, ["ATR14", "atr14", "ATR_14"])

    @staticmethod
    def mfi(ind: Any, mfi_period: Optional[int] = None) -> Optional[float]:
        """Extract MFI value with dynamic parameter detection"""
        if mfi_period:
            # Try dynamic template first
            template_keys = [f"MFI{mfi_period}", f"MFI_{mfi_period}"]
            for key in template_keys:
                val = Extract._last_key(ind, [key])
                if val is not None:
                    return val
        # Fallback to common variants
        return Extract._last_key(ind, ["MFI_14", "MFI", "mfi"])

    @staticmethod
    def cci(ind: Any, cci_period: Optional[int] = None) -> Optional[float]:
        """Extract CCI value with dynamic parameter detection"""
        if cci_period:
            # Try dynamic template first
            template_keys = [f"CCI{cci_period}", f"CCI_{cci_period}"]
            for key in template_keys:
                val = Extract._last_key(ind, [key])
                if val is not None:
                    return val
        # Fallback to common variants (including lowercase 'cci' that mt5_indicator_exporter uses)
        return Extract._last_key(ind, ["CCI_20", "CCI", "cci"])

    @staticmethod
    def williams_r(ind: Any, willr_period: Optional[int] = None) -> Optional[float]:
        """Extract Williams %R value with dynamic parameter detection"""
        if willr_period:
            # Try dynamic template first
            template_keys = [f"WilliamsR{willr_period}", f"WILLR{willr_period}", f"WilliamsR_{willr_period}", f"WILLR_{willr_period}"]
            for key in template_keys:
                val = Extract._last_key(ind, [key])
                if val is not None:
                    return val
        # Fallback to common variants
        return Extract._last_key(ind, ["WilliamsR_14", "WILLR_14", "WilliamsR", "WILLR"])

    @staticmethod
    def roc(ind: Any, roc_period: Optional[int] = None) -> Optional[float]:
        """Extract ROC value with dynamic parameter detection"""
        if roc_period:
            # Try dynamic template first
            template_keys = [f"ROC{roc_period}", f"ROC_{roc_period}"]
            for key in template_keys:
                val = Extract._last_key(ind, [key])
                if val is not None:
                    return val
        # Fallback to common variants
        return Extract._last_key(ind, ["ROC_20", "ROC", "roc"])

    @staticmethod
    def ma_value(ind: Any, ma_type: str, period: int) -> Optional[float]:
        """Extract MA value with dynamic parameter detection for any MA type"""
        template_keys = [f"{ma_type.upper()}{period}", f"{ma_type.upper()}_{period}", f"{ma_type.lower()}{period}", f"{ma_type.lower()}_{period}"]
        for key in template_keys:
            val = Extract._last_key(ind, [key])
            if val is not None:
                return val
        return None

    @staticmethod
    def psar_dir(ind: Any) -> Optional[int]:
        # Various schemas: PSAR_dir (1/-1), PSAR_trend ('bullish'/'bearish')
        if not isinstance(ind, list) or not ind or not isinstance(ind[-1], dict):
            return None
        row = ind[-1]
        v = row.get("PSAR_dir") or row.get("psar_dir")
        if isinstance(v, (int, float)):
            try:
                vi = int(v)
                return 1 if vi > 0 else (-1 if vi < 0 else 0)
            except Exception:
                return None
        t = row.get("PSAR_trend") or row.get("psar_trend")
        if isinstance(t, str):
            t = t.lower()
            if 'bull' in t: return 1
            if 'bear' in t: return -1
        return None

    @staticmethod
    def keltner(ind: Any) -> Optional[Dict[str, float]]:
        return Extract._last_pair(ind, ["Keltner_Upper"], ["Keltner_Middle"], ["Keltner_Lower"])

    @staticmethod
    def cci(ind: Any) -> Optional[float]:
        return Extract._last_key(ind, ["CCI_20", "CCI", "cci"])

    @staticmethod
    def williams_r(ind: Any) -> Optional[float]:
        return Extract._last_key(ind, ["WilliamsR_14", "WILLR_14", "WilliamsR", "willr"])

    @staticmethod
    def roc(ind: Any) -> Optional[float]:
        return Extract._last_key(ind, ["ROC_12", "ROC", "roc"])

    @staticmethod
    def chaikin(ind: Any) -> Optional[float]:
        # Try to use dynamically detected parameter first
        if hasattr(Extract, '_current_params') and Extract._current_params.get('chaikin'):
            chaikin_period = Extract._current_params['chaikin']
            result = Extract._last_key_with_params(ind, f"Chaikin{chaikin_period}", {})
            if result is not None:
                return result
        # Support multiple exporter variants: oscillator and generic period naming
        return Extract._last_key(ind, [
            "Chaikin_Osc", "ChaikinOsc", "chaikin_osc",
            "Chaikin20", "Chaikin_20",
            "Chaikin", "chaikin"
        ])

    @staticmethod
    def eom(ind: Any) -> Optional[float]:
        # Try to use dynamically detected parameter first
        if hasattr(Extract, '_current_params') and Extract._current_params.get('eom'):
            eom_period = Extract._current_params['eom']
            result = Extract._last_key_with_params(ind, f"EOM{eom_period}", {})
            if result is not None:
                return result
        # Fallback to standard patterns
        return Extract._last_key(ind, ["EOM_14", "EOM", "eom"])

    @staticmethod
    def force_index(ind: Any) -> Optional[float]:
        # Accept multiple exporter variants (period 13 or 14) to maximize hit rate
        return Extract._last_key(ind, [
            "ForceIndex14", "ForceIndex_14", "Force_Index_14",
            "ForceIndex13", "ForceIndex_13", "Force_Index_13",
            "ForceIndex_13", "Force_Index", "force_index"
        ])

    @staticmethod
    def trix(ind: Any) -> Optional[float]:
        # Try to use dynamically detected parameter first
        if hasattr(Extract, '_current_params') and Extract._current_params.get('trix'):
            trix_period = Extract._current_params['trix']
            result = Extract._last_key_with_params(ind, f"TRIX{trix_period}", {})
            if result is not None:
                return result
        # Fallback to standard patterns
        return Extract._last_key(ind, ["TRIX_15", "TRIX", "trix"])

    @staticmethod
    def dpo(ind: Any) -> Optional[float]:
        # Try to use dynamically detected parameter first
        if hasattr(Extract, '_current_params') and Extract._current_params.get('dpo'):
            dpo_period = Extract._current_params['dpo']
            result = Extract._last_key_with_params(ind, f"DPO{dpo_period}", {})
            if result is not None:
                return result
        # Fallback to standard patterns
        return Extract._last_key(ind, ["DPO_20", "DPO", "dpo"])

    @staticmethod
    def mass_index(ind: Any) -> Optional[float]:
        return Extract._last_key(ind, ["MassIndex_9_25", "MassIndex_25", "MassIndex", "mass_index"])

    @staticmethod
    def vortex(ind: Any) -> Optional[Dict[str, float]]:
        if not isinstance(ind, list) or not ind or not isinstance(ind[-1], dict):
            return None
        row = ind[-1]
        vp = row.get("Vortex_plus") or row.get("VI+") or row.get("vi_plus")
        vm = row.get("Vortex_minus") or row.get("VI-") or row.get("vi_minus")
        out: Dict[str, float] = {}
        if vp is not None: out['vi_plus'] = ffloat(vp, None)
        if vm is not None: out['vi_minus'] = ffloat(vm, None)
        return out if out else None

    @staticmethod
    def kst(ind: Any) -> Optional[float]:
        return Extract._last_key(ind, ["KST", "kst"])

    @staticmethod
    def ultimate_osc(ind: Any) -> Optional[float]:
        return Extract._last_key(ind, ["Ultimate_Osc", "UltimateOscillator", "ultimate_osc"])

    @staticmethod
    def envelopes(ind: Any) -> Optional[Dict[str, float]]:
        """Extract simple envelopes structure if present (upper/lower/middle)."""
        try:
            if not isinstance(ind, (list, dict)) or not ind:
                return None
            # Column style: look for last row with keys
            rows = ind if isinstance(ind, list) else None
            if isinstance(rows, list) and rows and isinstance(rows[-1], dict):
                row = rows[-1]
                up = row.get("Envelopes_upper") or row.get("Env_Upper")
                lo = row.get("Envelopes_lower") or row.get("Env_Lower")
                mid = row.get("Envelopes_middle") or row.get("Env_Middle")
                out: Dict[str, float] = {}
                if up is not None: out['upper'] = ffloat(up, None)
                if lo is not None: out['lower'] = ffloat(lo, None)
                if mid is not None: out['middle'] = ffloat(mid, None)
                return out if out else None
            # Dict style
            if isinstance(ind, dict):
                node = ind.get("Envelopes") or ind.get("envelopes") or ind.get("ENV")
                if isinstance(node, dict):
                    up = node.get('upper'); lo = node.get('lower'); mid = node.get('middle') or node.get('basis')
                    out: Dict[str, float] = {}
                    if up is not None: out['upper'] = ffloat(up, None)
                    if lo is not None: out['lower'] = ffloat(lo, None)
                    if mid is not None: out['middle'] = ffloat(mid, None)
                    return out if out else None
            return None
        except Exception:
            return None

    @staticmethod
    def cci(ind: Any) -> Optional[float]:
        try:
            # column-style
            v = (
                Extract._last_from_list(ind, "CCI20")
                or Extract._last_from_list(ind, "CCI_20")
            )
            if v is not None:
                return v
            # dict-style
            node = ind.get("CCI") or ind.get("cci") if isinstance(ind, dict) else None
            if isinstance(node, list) and node:
                return ffloat(node[-1])
            if isinstance(node, (int, float)):
                return ffloat(node)
            if isinstance(node, dict):
                for sk in ("values", "data", "series"):
                    if isinstance(node.get(sk), list) and node[sk]:
                        return ffloat(node[sk][-1])
            return None
        except Exception:
            return None

    @staticmethod
    def bbands_bias(ind: Any) -> Optional[float]:
        """Return bias relative to Bollinger bands: +1 near lower band (buy), -1 near upper band (sell)."""
        try:
            # column-style first
            up = Extract._last_from_list(ind, "BB_Upper_20_2")
            mid = Extract._last_from_list(ind, "BB_Middle_20_2")
            lo = Extract._last_from_list(ind, "BB_Lower_20_2")
            price = Extract._get_price_from_ind(ind)
            if None in (up, mid, lo, price):
                # dict-style fallback
                node = ind.get("BBANDS") or ind.get("bbands") if isinstance(ind, dict) else None
                if not isinstance(node, dict):
                    return None
                up = node.get("upper") or node.get("upperband")
                mid = node.get("middle") or node.get("middleband") or node.get("basis")
                lo = node.get("lower") or node.get("lowerband")
                pclose = ind.get("close") if isinstance(ind, dict) else None
                def last(v):
                    if isinstance(v, list) and v: return ffloat(v[-1], None)
                    if isinstance(v, (int, float)): return ffloat(v, None)
                    return None
                up = last(up); mid = last(mid); lo = last(lo); price = last(pclose)
                if None in (up, mid, lo, price):
                    return None
            width = max(up - lo, 1e-6)
            pos = (price - lo) / width  # 0..1
            if pos < 0.2: return +1.0
            if pos > 0.8: return -1.0
            return 0.0
        except Exception:
            return None

    @staticmethod
    def bbands(ind: Any) -> Optional[Dict[str, Optional[float]]]:
        """Return Bollinger band snapshot dict if available."""
        try:
            up = Extract._last_from_list(ind, "BB_Upper_20_2")
            mid = Extract._last_from_list(ind, "BB_Middle_20_2")
            lo = Extract._last_from_list(ind, "BB_Lower_20_2")
            if None not in (up, mid, lo):
                return {"upper": up, "middle": mid, "lower": lo}
            node = ind.get("BBANDS") or ind.get("bbands") if isinstance(ind, dict) else None
            if isinstance(node, dict):
                def last(v):
                    if isinstance(v, list) and v: return ffloat(v[-1], None)
                    if isinstance(v, (int, float)): return ffloat(v, None)
                    return None
                up = last(node.get("upper") or node.get("upperband"))
                mid = last(node.get("middle") or node.get("middleband") or node.get("basis"))
                lo = last(node.get("lower") or node.get("lowerband"))
                if None not in (up, mid, lo):
                    return {"upper": up, "middle": mid, "lower": lo}
            return None
        except Exception:
            return None

    @staticmethod
    def obv_bias(ind: Any) -> Optional[int]:
        try:
            # column-style
            prev, curr = Extract._last_two_from_list(ind, "OBV")
            if prev is not None and curr is not None:
                return 1 if curr > prev else -1 if curr < prev else 0
            # dict-style
            node = ind.get("OBV") or ind.get("obv") if isinstance(ind, dict) else None
            if isinstance(node, list) and len(node) >= 2:
                return 1 if ffloat(node[-1]) > ffloat(node[-2]) else -1 if ffloat(node[-1]) < ffloat(node[-2]) else 0
            return None
        except Exception:
            return None

    # --- Additional extractors for more indicators present in indicator_output ---
    @staticmethod
    def mfi(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "MFI14")
            or Extract._last_from_list(ind, "MFI_14")
        )

    @staticmethod
    def williams_r(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "WilliamsR14")
            or Extract._last_from_list(ind, "WilliamsR_14")
            or Extract._last_from_list(ind, "WILLR14")
            or Extract._last_from_list(ind, "WILLR_14")
            or Extract._last_from_list(ind, "Williams %R")
        )

    @staticmethod
    def roc(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "ROC12")
            or Extract._last_from_list(ind, "ROC_12")
            or Extract._last_from_list(ind, "ROC")
        )

    @staticmethod
    def psar(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "PSAR")
            or Extract._last_from_list(ind, "SAR")
        )

    @staticmethod
    def psar_dir(ind: Any) -> Optional[int]:
        """Return PSAR directional hint: +1 if price > psar, -1 if price < psar."""
        try:
            ps = Extract.psar(ind)
            price = Extract._get_price_from_ind(ind)
            if ps is None or price is None:
                return None
            return 1 if price > ps else -1 if price < ps else 0
        except Exception:
            return None

    @staticmethod
    def donchian_bias(ind: Any) -> Optional[float]:
        # Prefer dynamic period from last row if available
        up = Extract._last_from_list(ind, "Donchian_Upper_20")
        lo = Extract._last_from_list(ind, "Donchian_Lower_20")
        if (up is None or lo is None) and isinstance(ind, list) and ind and isinstance(ind[-1], dict):
            try:
                last = ind[-1]
                uppers = [(k, v) for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith(("Donchian_Upper_", "DONCHIAN_UPPER_"))]
                lowers = [(k, v) for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith(("Donchian_Lower_", "DONCHIAN_LOWER_"))]
                # build by period
                import re as _re
                map_u: dict[int, float] = {}
                map_l: dict[int, float] = {}
                for k, v in uppers:
                    m = _re.search(r"(\\d+)$", str(k))
                    if m:
                        map_u[int(m.group(1))] = ffloat(v)
                for k, v in lowers:
                    m = _re.search(r"(\\d+)$", str(k))
                    if m:
                        map_l[int(m.group(1))] = ffloat(v)
                both = [p for p in map_u.keys() if p in map_l]
                if both:
                    use_p = max(both)
                    up = map_u.get(use_p)
                    lo = map_l.get(use_p)
                else:
                    # fallback to the largest available side
                    if not up and map_u:
                        up = map_u.get(max(map_u.keys()))
                    if not lo and map_l:
                        lo = map_l.get(max(map_l.keys()))
            except Exception:
                pass
        price = Extract._get_price_from_ind(ind)
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def keltner_bias(ind: Any) -> Optional[float]:
        up = Extract._last_from_list(ind, "Keltner_Upper_20")
        lo = Extract._last_from_list(ind, "Keltner_Lower_20")
        price = Extract._get_price_from_ind(ind)
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def keltner(ind: Any) -> Optional[Dict[str, Optional[float]]]:
        up = Extract._last_from_list(ind, "Keltner_Upper_20")
        lo = Extract._last_from_list(ind, "Keltner_Lower_20")
        if None not in (up, lo):
            return {"upper": up, "lower": lo}
        return None

    @staticmethod
    def envelope_bias(ind: Any) -> Optional[float]:
        # Try common exact keys first
        up = Extract._last_from_list(ind, "Envelope_Upper_20_2.0") or Extract._last_from_list(ind, "Envelope_Upper_20_2")
        lo = Extract._last_from_list(ind, "Envelope_Lower_20_2.0") or Extract._last_from_list(ind, "Envelope_Lower_20_2")
        price = Extract._get_price_from_ind(ind)
        # Fallback: scan last bar for any Envelope_Upper/Lower keys with numeric values
        if (up is None or lo is None) and isinstance(ind, list) and ind and isinstance(ind[-1], dict):
            try:
                last = ind[-1]
                up_keys = [k for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith("Envelope_Upper")]
                lo_keys = [k for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith("Envelope_Lower")]
                if up is None and up_keys:
                    # Pick the lexicographically last (often the most parameterized)
                    up = ffloat(last[sorted(up_keys)[-1]])
                if lo is None and lo_keys:
                    lo = ffloat(last[sorted(lo_keys)[-1]])
            except Exception:
                pass
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def envelopes(ind: Any) -> Optional[Dict[str, Optional[float]]]:
        """Return envelopes snapshot dict if available."""
        try:
            up = Extract._last_from_list(ind, "Envelope_Upper_20_2.0") or Extract._last_from_list(ind, "Envelope_Upper_20_2")
            lo = Extract._last_from_list(ind, "Envelope_Lower_20_2.0") or Extract._last_from_list(ind, "Envelope_Lower_20_2")
            if None not in (up, lo):
                return {"upper": up, "lower": lo}
            node = ind.get("BBANDS") or ind.get("bbands") if isinstance(ind, dict) else None
            if isinstance(node, dict):
                def last(v):
                    if isinstance(v, list) and v: return ffloat(v[-1], None)
                    if isinstance(v, (int, float)): return ffloat(v, None)
                    return None
                up = last(node.get("upper") or node.get("upperband"))
                mid = last(node.get("middle") or node.get("middleband") or node.get("basis"))
                lo = last(node.get("lower") or node.get("lowerband"))
                if None not in (up, mid, lo):
                    return {"upper": up, "middle": mid, "lower": lo}
            return None
        except Exception:
            return None

    @staticmethod
    def obv_bias(ind: Any) -> Optional[int]:
        try:
            # column-style
            prev, curr = Extract._last_two_from_list(ind, "OBV")
            if prev is not None and curr is not None:
                return 1 if curr > prev else -1 if curr < prev else 0
            # dict-style
            node = ind.get("OBV") or ind.get("obv") if isinstance(ind, dict) else None
            if isinstance(node, list) and len(node) >= 2:
                return 1 if ffloat(node[-1]) > ffloat(node[-2]) else -1 if ffloat(node[-1]) < ffloat(node[-2]) else 0
            return None
        except Exception:
            return None

    # --- Lightweight stubs for indicators referenced elsewhere ---
    @staticmethod
    def chaikin(ind: Any) -> Optional[float]:
        try:
            for k in ("Chaikin", "ChaikinOsc", "chaikin", "chaikin_osc"):
                v = ind.get(k) if isinstance(ind, dict) else None
                if isinstance(v, list) and v:
                    return ffloat(v[-1])
                if isinstance(v, (int, float)):
                    return ffloat(v)
            return None
        except Exception:
            return None

    @staticmethod
    def eom(ind: Any) -> Optional[float]:
        try:
            v = ind.get("EOM") if isinstance(ind, dict) else None
            if isinstance(v, list) and v:
                return ffloat(v[-1])
            if isinstance(v, (int, float)):
                return ffloat(v)
            return None
        except Exception:
            return None

    @staticmethod
    def force_index(ind: Any) -> Optional[float]:
        try:
            v = ind.get("ForceIndex") if isinstance(ind, dict) else None
            if isinstance(v, list) and v:
                return ffloat(v[-1])
            if isinstance(v, (int, float)):
                return ffloat(v)
            return None
        except Exception:
            return None

    @staticmethod
    def trix(ind: Any) -> Optional[float]:
        try:
            v = ind.get("TRIX") if isinstance(ind, dict) else None
            if isinstance(v, list) and v:
                return ffloat(v[-1])
            if isinstance(v, (int, float)):
                return ffloat(v)
            return None
        except Exception:
            return None

    @staticmethod
    def dpo(ind: Any) -> Optional[float]:
        try:
            v = ind.get("DPO") if isinstance(ind, dict) else None
            if isinstance(v, list) and v:
                return ffloat(v[-1])
            if isinstance(v, (int, float)):
                return ffloat(v)
            return None
        except Exception:
            return None

    @staticmethod
    def mass_index(ind: Any) -> Optional[float]:
        try:
            # Try different possible column names for Mass Index
            for key in ["MassIndex_9_25", "MassIndex_25", "MassIndex", "mass_index"]:
                v = ind.get(key) if isinstance(ind, dict) else None
                if v is not None:
                    if isinstance(v, list) and v:
                        return ffloat(v[-1])
                    if isinstance(v, (int, float)):
                        return ffloat(v)
            return None
        except Exception:
            return None

    @staticmethod
    def vortex(ind: Any) -> Optional[Dict[str, float]]:
        try:
            node = ind.get("Vortex") if isinstance(ind, dict) else None
            if isinstance(node, dict):
                vp = node.get("vi_plus") or node.get("VI+") or node.get("VIp")
                vm = node.get("vi_minus") or node.get("VI-") or node.get("VIm")
                out: Dict[str, float] = {}
                if isinstance(vp, (int, float)):
                    out["vi_plus"] = ffloat(vp)
                if isinstance(vm, (int, float)):
                    out["vi_minus"] = ffloat(vm)
                return out if out else None
            return None
        except Exception:
            return None

    @staticmethod
    def kst(ind: Any) -> Optional[float]:
        try:
            v = ind.get("KST") if isinstance(ind, dict) else None
            if isinstance(v, list) and v:
                return ffloat(v[-1])
            if isinstance(v, (int, float)):
                return ffloat(v)
            return None
        except Exception:
            return None

    @staticmethod
    def ultimate_osc(ind: Any) -> Optional[float]:
        try:
            for k in ("UltimateOsc", "Ultimate", "ultimate_osc"):
                v = ind.get(k) if isinstance(ind, dict) else None
                if isinstance(v, list) and v:
                    return ffloat(v[-1])
                if isinstance(v, (int, float)):
                    return ffloat(v)
            return None
        except Exception:
            return None

    # --- Additional extractors for more indicators present in indicator_output ---
    @staticmethod
    def mfi(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "MFI14")
            or Extract._last_from_list(ind, "MFI_14")
        )

    @staticmethod
    def williams_r(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "WilliamsR14")
            or Extract._last_from_list(ind, "WilliamsR_14")
            or Extract._last_from_list(ind, "WILLR14")
            or Extract._last_from_list(ind, "WILLR_14")
            or Extract._last_from_list(ind, "Williams %R")
        )

    @staticmethod
    def roc(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "ROC12")
            or Extract._last_from_list(ind, "ROC_12")
            or Extract._last_from_list(ind, "ROC")
        )

    @staticmethod
    def psar(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "PSAR")
            or Extract._last_from_list(ind, "SAR")
        )

    @staticmethod
    def psar_dir(ind: Any) -> Optional[int]:
        """Return PSAR directional hint: +1 if price > psar, -1 if price < psar."""
        try:
            ps = Extract.psar(ind)
            price = Extract._get_price_from_ind(ind)
            if ps is None or price is None:
                return None
            return 1 if price > ps else -1 if price < ps else 0
        except Exception:
            return None

    @staticmethod
    def donchian_bias(ind: Any) -> Optional[float]:
        # Prefer dynamic period from last row if available
        up = Extract._last_from_list(ind, "Donchian_Upper_20")
        lo = Extract._last_from_list(ind, "Donchian_Lower_20")
        if (up is None or lo is None) and isinstance(ind, list) and ind and isinstance(ind[-1], dict):
            try:
                last = ind[-1]
                uppers = [(k, v) for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith(("Donchian_Upper_", "DONCHIAN_UPPER_"))]
                lowers = [(k, v) for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith(("Donchian_Lower_", "DONCHIAN_LOWER_"))]
                import re as _re
                map_u: dict[int, float] = {}
                map_l: dict[int, float] = {}
                for k, v in uppers:
                    m = _re.search(r"(\\d+)$", str(k))
                    if m:
                        map_u[int(m.group(1))] = ffloat(v)
                for k, v in lowers:
                    m = _re.search(r"(\\d+)$", str(k))
                    if m:
                        map_l[int(m.group(1))] = ffloat(v)
                both = [p for p in map_u.keys() if p in map_l]
                if both:
                    use_p = max(both)
                    up = map_u.get(use_p)
                    lo = map_l.get(use_p)
                else:
                    if not up and map_u:
                        up = map_u.get(max(map_u.keys()))
                    if not lo and map_l:
                        lo = map_l.get(max(map_l.keys()))
            except Exception:
                pass
        price = Extract._get_price_from_ind(ind)
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def keltner_bias(ind: Any) -> Optional[float]:
        up = Extract._last_from_list(ind, "Keltner_Upper_20")
        lo = Extract._last_from_list(ind, "Keltner_Lower_20")
        price = Extract._get_price_from_ind(ind)
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def keltner(ind: Any) -> Optional[Dict[str, Optional[float]]]:
        up = Extract._last_from_list(ind, "Keltner_Upper_20")
        lo = Extract._last_from_list(ind, "Keltner_Lower_20")
        if None not in (up, lo):
            return {"upper": up, "lower": lo}
        return None

    @staticmethod
    def envelope_bias(ind: Any) -> Optional[float]:
        # Try common exact keys first
        up = Extract._last_from_list(ind, "Envelope_Upper_20_2.0") or Extract._last_from_list(ind, "Envelope_Upper_20_2")
        lo = Extract._last_from_list(ind, "Envelope_Lower_20_2.0") or Extract._last_from_list(ind, "Envelope_Lower_20_2")
        price = Extract._get_price_from_ind(ind)
        # Fallback: scan last bar for any Envelope_Upper/Lower keys with numeric values
        if (up is None or lo is None) and isinstance(ind, list) and ind and isinstance(ind[-1], dict):
            try:
                last = ind[-1]
                up_keys = [k for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith("Envelope_Upper")]
                lo_keys = [k for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith("Envelope_Lower")]
                if up is None and up_keys:
                    # Pick the lexicographically last (often the most parameterized)
                    up = ffloat(last[sorted(up_keys)[-1]])
                if lo is None and lo_keys:
                    lo = ffloat(last[sorted(lo_keys)[-1]])
            except Exception:
                pass
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def envelopes(ind: Any) -> Optional[Dict[str, Optional[float]]]:
        """Return envelopes snapshot dict if available."""
        try:
            up = Extract._last_from_list(ind, "Envelope_Upper_20_2.0") or Extract._last_from_list(ind, "Envelope_Upper_20_2")
            lo = Extract._last_from_list(ind, "Envelope_Lower_20_2.0") or Extract._last_from_list(ind, "Envelope_Lower_20_2")
            if None not in (up, lo):
                return {"upper": up, "lower": lo}
            node = ind.get("BBANDS") or ind.get("bbands") if isinstance(ind, dict) else None
            if isinstance(node, dict):
                def last(v):
                    if isinstance(v, list) and v: return ffloat(v[-1], None)
                    if isinstance(v, (int, float)): return ffloat(v, None)
                    return None
                up = last(node.get("upper") or node.get("upperband"))
                mid = last(node.get("middle") or node.get("middleband") or node.get("basis"))
                lo = last(node.get("lower") or node.get("lowerband"))
                if None not in (up, mid, lo):
                    return {"upper": up, "middle": mid, "lower": lo}
            return None
        except Exception:
            return None

    @staticmethod
    def obv_bias(ind: Any) -> Optional[int]:
        try:
            # column-style
            prev, curr = Extract._last_two_from_list(ind, "OBV")
            if prev is not None and curr is not None:
                return 1 if curr > prev else -1 if curr < prev else 0
            # dict-style
            node = ind.get("OBV") or ind.get("obv") if isinstance(ind, dict) else None
            if isinstance(node, list) and len(node) >= 2:
                return 1 if ffloat(node[-1]) > ffloat(node[-2]) else -1 if ffloat(node[-1]) < ffloat(node[-2]) else 0
            return None
        except Exception:
            return None

    # --- Additional extractors for more indicators present in indicator_output ---
    @staticmethod
    def mfi(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "MFI14")
            or Extract._last_from_list(ind, "MFI_14")
        )

    @staticmethod
    def williams_r(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "WilliamsR14")
            or Extract._last_from_list(ind, "WilliamsR_14")
            or Extract._last_from_list(ind, "WILLR14")
            or Extract._last_from_list(ind, "WILLR_14")
            or Extract._last_from_list(ind, "Williams %R")
        )

    @staticmethod
    def roc(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "ROC12")
            or Extract._last_from_list(ind, "ROC_12")
            or Extract._last_from_list(ind, "ROC")
        )

    @staticmethod
    def psar(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "PSAR")
            or Extract._last_from_list(ind, "SAR")
        )

    @staticmethod
    def psar_dir(ind: Any) -> Optional[int]:
        """Return PSAR directional hint: +1 if price > psar, -1 if price < psar."""
        try:
            ps = Extract.psar(ind)
            price = Extract._get_price_from_ind(ind)
            if ps is None or price is None:
                return None
            return 1 if price > ps else -1 if price < ps else 0
        except Exception:
            return None

    @staticmethod
    def donchian_bias(ind: Any) -> Optional[float]:
        # Prefer dynamic period from last row if available
        up = Extract._last_from_list(ind, "Donchian_Upper_20")
        lo = Extract._last_from_list(ind, "Donchian_Lower_20")
        if (up is None or lo is None) and isinstance(ind, list) and ind and isinstance(ind[-1], dict):
            try:
                last = ind[-1]
                uppers = [(k, v) for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith(("Donchian_Upper_", "DONCHIAN_UPPER_"))]
                lowers = [(k, v) for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith(("Donchian_Lower_", "DONCHIAN_LOWER_"))]
                import re as _re
                map_u: dict[int, float] = {}
                map_l: dict[int, float] = {}
                for k, v in uppers:
                    m = _re.search(r"(\\d+)$", str(k))
                    if m:
                        map_u[int(m.group(1))] = ffloat(v)
                for k, v in lowers:
                    m = _re.search(r"(\\d+)$", str(k))
                    if m:
                        map_l[int(m.group(1))] = ffloat(v)
                both = [p for p in map_u.keys() if p in map_l]
                if both:
                    use_p = max(both)
                    up = map_u.get(use_p)
                    lo = map_l.get(use_p)
                else:
                    if not up and map_u:
                        up = map_u.get(max(map_u.keys()))
                    if not lo and map_l:
                        lo = map_l.get(max(map_l.keys()))
            except Exception:
                pass
        price = Extract._get_price_from_ind(ind)
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def keltner_bias(ind: Any) -> Optional[float]:
        up = Extract._last_from_list(ind, "Keltner_Upper_20")
        lo = Extract._last_from_list(ind, "Keltner_Lower_20")
        price = Extract._get_price_from_ind(ind)
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def keltner(ind: Any) -> Optional[Dict[str, Optional[float]]]:
        up = Extract._last_from_list(ind, "Keltner_Upper_20")
        lo = Extract._last_from_list(ind, "Keltner_Lower_20")
        if None not in (up, lo):
            return {"upper": up, "lower": lo}
        return None

    @staticmethod
    def envelope_bias(ind: Any) -> Optional[float]:
        # Try common exact keys first
        up = Extract._last_from_list(ind, "Envelope_Upper_20_2.0") or Extract._last_from_list(ind, "Envelope_Upper_20_2")
        lo = Extract._last_from_list(ind, "Envelope_Lower_20_2.0") or Extract._last_from_list(ind, "Envelope_Lower_20_2")
        price = Extract._get_price_from_ind(ind)
        # Fallback: scan last bar for any Envelope_Upper/Lower keys with numeric values
        if (up is None or lo is None) and isinstance(ind, list) and ind and isinstance(ind[-1], dict):
            try:
                last = ind[-1]
                up_keys = [k for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith("Envelope_Upper")]
                lo_keys = [k for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith("Envelope_Lower")]
                if up is None and up_keys:
                    # Pick the lexicographically last (often the most parameterized)
                    up = ffloat(last[sorted(up_keys)[-1]])
                if lo is None and lo_keys:
                    lo = ffloat(last[sorted(lo_keys)[-1]])
            except Exception:
                pass
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def envelopes(ind: Any) -> Optional[Dict[str, Optional[float]]]:
        """Return envelopes snapshot dict if available."""
        try:
            up = Extract._last_from_list(ind, "Envelope_Upper_20_2.0") or Extract._last_from_list(ind, "Envelope_Upper_20_2")
            lo = Extract._last_from_list(ind, "Envelope_Lower_20_2.0") or Extract._last_from_list(ind, "Envelope_Lower_20_2")
            if None not in (up, lo):
                return {"upper": up, "lower": lo}
            node = ind.get("BBANDS") or ind.get("bbands") if isinstance(ind, dict) else None
            if isinstance(node, dict):
                def last(v):
                    if isinstance(v, list) and v: return ffloat(v[-1], None)
                    if isinstance(v, (int, float)): return ffloat(v, None)
                    return None
                up = last(node.get("upper") or node.get("upperband"))
                mid = last(node.get("middle") or node.get("middleband") or node.get("basis"))
                lo = last(node.get("lower") or node.get("lowerband"))
                if None not in (up, mid, lo):
                    return {"upper": up, "middle": mid, "lower": lo}
            return None
        except Exception:
            return None

    @staticmethod
    def obv_bias(ind: Any) -> Optional[int]:
        try:
            # column-style
            prev, curr = Extract._last_two_from_list(ind, "OBV")
            if prev is not None and curr is not None:
                return 1 if curr > prev else -1 if curr < prev else 0
            # dict-style
            node = ind.get("OBV") or ind.get("obv") if isinstance(ind, dict) else None
            if isinstance(node, list) and len(node) >= 2:
                return 1 if ffloat(node[-1]) > ffloat(node[-2]) else -1 if ffloat(node[-1]) < ffloat(node[-2]) else 0
            return None
        except Exception:
            return None

    # --- Additional extractors for more indicators present in indicator_output ---
    @staticmethod
    def mfi(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "MFI14")
            or Extract._last_from_list(ind, "MFI_14")
        )

    @staticmethod
    def williams_r(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "WilliamsR14")
            or Extract._last_from_list(ind, "WilliamsR_14")
            or Extract._last_from_list(ind, "WILLR14")
            or Extract._last_from_list(ind, "WILLR_14")
            or Extract._last_from_list(ind, "Williams %R")
        )

    @staticmethod
    def roc(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "ROC12")
            or Extract._last_from_list(ind, "ROC_12")
            or Extract._last_from_list(ind, "ROC")
        )

    @staticmethod
    def psar(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "PSAR")
            or Extract._last_from_list(ind, "SAR")
        )

    @staticmethod
    def psar_dir(ind: Any) -> Optional[int]:
        """Return PSAR directional hint: +1 if price > psar, -1 if price < psar."""
        try:
            ps = Extract.psar(ind)
            price = Extract._get_price_from_ind(ind)
            if ps is None or price is None:
                return None
            return 1 if price > ps else -1 if price < ps else 0
        except Exception:
            return None

    @staticmethod
    def donchian_bias(ind: Any) -> Optional[float]:
        # Prefer dynamic period from last row if available
        up = Extract._last_from_list(ind, "Donchian_Upper_20")
        lo = Extract._last_from_list(ind, "Donchian_Lower_20")
        if (up is None or lo is None) and isinstance(ind, list) and ind and isinstance(ind[-1], dict):
            try:
                last = ind[-1]
                uppers = [(k, v) for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith(("Donchian_Upper_", "DONCHIAN_UPPER_"))]
                lowers = [(k, v) for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith(("Donchian_Lower_", "DONCHIAN_LOWER_"))]
                import re as _re
                map_u: dict[int, float] = {}
                map_l: dict[int, float] = {}
                for k, v in uppers:
                    m = _re.search(r"(\\d+)$", str(k))
                    if m:
                        map_u[int(m.group(1))] = ffloat(v)
                for k, v in lowers:
                    m = _re.search(r"(\\d+)$", str(k))
                    if m:
                        map_l[int(m.group(1))] = ffloat(v)
                both = [p for p in map_u.keys() if p in map_l]
                if both:
                    use_p = max(both)
                    up = map_u.get(use_p)
                    lo = map_l.get(use_p)
                else:
                    if not up and map_u:
                        up = map_u.get(max(map_u.keys()))
                    if not lo and map_l:
                        lo = map_l.get(max(map_l.keys()))
            except Exception:
                pass
        price = Extract._get_price_from_ind(ind)
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def keltner_bias(ind: Any) -> Optional[float]:
        up = Extract._last_from_list(ind, "Keltner_Upper_20")
        lo = Extract._last_from_list(ind, "Keltner_Lower_20")
        price = Extract._get_price_from_ind(ind)
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def keltner(ind: Any) -> Optional[Dict[str, Optional[float]]]:
        up = Extract._last_from_list(ind, "Keltner_Upper_20")
        lo = Extract._last_from_list(ind, "Keltner_Lower_20")
        if None not in (up, lo):
            return {"upper": up, "lower": lo}
        return None

    @staticmethod
    def envelope_bias(ind: Any) -> Optional[float]:
        # Try common exact keys first
        up = Extract._last_from_list(ind, "Envelope_Upper_20_2.0") or Extract._last_from_list(ind, "Envelope_Upper_20_2")
        lo = Extract._last_from_list(ind, "Envelope_Lower_20_2.0") or Extract._last_from_list(ind, "Envelope_Lower_20_2")
        price = Extract._get_price_from_ind(ind)
        # Fallback: scan last bar for any Envelope_Upper/Lower keys with numeric values
        if (up is None or lo is None) and isinstance(ind, list) and ind and isinstance(ind[-1], dict):
            try:
                last = ind[-1]
                up_keys = [k for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith("Envelope_Upper")]
                lo_keys = [k for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith("Envelope_Lower")]
                if up is None and up_keys:
                    # Pick the lexicographically last (often the most parameterized)
                    up = ffloat(last[sorted(up_keys)[-1]])
                if lo is None and lo_keys:
                    lo = ffloat(last[sorted(lo_keys)[-1]])
            except Exception:
                pass
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def envelopes(ind: Any) -> Optional[Dict[str, Optional[float]]]:
        """Return envelopes snapshot dict if available."""
        try:
            up = Extract._last_from_list(ind, "Envelope_Upper_20_2.0") or Extract._last_from_list(ind, "Envelope_Upper_20_2")
            lo = Extract._last_from_list(ind, "Envelope_Lower_20_2.0") or Extract._last_from_list(ind, "Envelope_Lower_20_2")
            if None not in (up, lo):
                return {"upper": up, "lower": lo}
            node = ind.get("BBANDS") or ind.get("bbands") if isinstance(ind, dict) else None
            if isinstance(node, dict):
                def last(v):
                    if isinstance(v, list) and v: return ffloat(v[-1], None)
                    if isinstance(v, (int, float)): return ffloat(v, None)
                    return None
                up = last(node.get("upper") or node.get("upperband"))
                mid = last(node.get("middle") or node.get("middleband") or node.get("basis"))
                lo = last(node.get("lower") or node.get("lowerband"))
                if None not in (up, mid, lo):
                    return {"upper": up, "middle": mid, "lower": lo}
            return None
        except Exception:
            return None

    @staticmethod
    def obv_bias(ind: Any) -> Optional[int]:
        try:
            # column-style
            prev, curr = Extract._last_two_from_list(ind, "OBV")
            if prev is not None and curr is not None:
                return 1 if curr > prev else -1 if curr < prev else 0
            # dict-style
            node = ind.get("OBV") or ind.get("obv") if isinstance(ind, dict) else None
            if isinstance(node, list) and len(node) >= 2:
                return 1 if ffloat(node[-1]) > ffloat(node[-2]) else -1 if ffloat(node[-1]) < ffloat(node[-2]) else 0
            return None
        except Exception:
            return None

    # --- Additional extractors for more indicators present in indicator_output ---
    @staticmethod
    def mfi(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "MFI14")
            or Extract._last_from_list(ind, "MFI_14")
        )

    @staticmethod
    def williams_r(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "WilliamsR14")
            or Extract._last_from_list(ind, "WilliamsR_14")
            or Extract._last_from_list(ind, "WILLR14")
            or Extract._last_from_list(ind, "WILLR_14")
            or Extract._last_from_list(ind, "Williams %R")
        )

    @staticmethod
    def roc(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "ROC12")
            or Extract._last_from_list(ind, "ROC_12")
            or Extract._last_from_list(ind, "ROC")
        )

    @staticmethod
    def psar(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "PSAR")
            or Extract._last_from_list(ind, "SAR")
        )

    @staticmethod
    def psar_dir(ind: Any) -> Optional[int]:
        """Return PSAR directional hint: +1 if price > psar, -1 if price < psar."""
        try:
            ps = Extract.psar(ind)
            price = Extract._get_price_from_ind(ind)
            if ps is None or price is None:
                return None
            return 1 if price > ps else -1 if price < ps else 0
        except Exception:
            return None

    @staticmethod
    def donchian_bias(ind: Any) -> Optional[float]:
        up = Extract._last_from_list(ind, "Donchian_Upper_20")
        lo = Extract._last_from_list(ind, "Donchian_Lower_20")
        price = Extract._get_price_from_ind(ind)
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def keltner_bias(ind: Any) -> Optional[float]:
        up = Extract._last_from_list(ind, "Keltner_Upper_20")
        lo = Extract._last_from_list(ind, "Keltner_Lower_20")
        price = Extract._get_price_from_ind(ind)
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def keltner(ind: Any) -> Optional[Dict[str, Optional[float]]]:
        up = Extract._last_from_list(ind, "Keltner_Upper_20")
        lo = Extract._last_from_list(ind, "Keltner_Lower_20")
        if None not in (up, lo):
            return {"upper": up, "lower": lo}
        return None

    @staticmethod
    def envelope_bias(ind: Any) -> Optional[float]:
        # Try common exact keys first
        up = Extract._last_from_list(ind, "Envelope_Upper_20_2.0") or Extract._last_from_list(ind, "Envelope_Upper_20_2")
        lo = Extract._last_from_list(ind, "Envelope_Lower_20_2.0") or Extract._last_from_list(ind, "Envelope_Lower_20_2")
        price = Extract._get_price_from_ind(ind)
        # Fallback: scan last bar for any Envelope_Upper/Lower keys with numeric values
        if (up is None or lo is None) and isinstance(ind, list) and ind and isinstance(ind[-1], dict):
            try:
                last = ind[-1]
                up_keys = [k for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith("Envelope_Upper")]
                lo_keys = [k for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith("Envelope_Lower")]
                if up is None and up_keys:
                    # Pick the lexicographically last (often the most parameterized)
                    up = ffloat(last[sorted(up_keys)[-1]])
                if lo is None and lo_keys:
                    lo = ffloat(last[sorted(lo_keys)[-1]])
            except Exception:
                pass
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def envelopes(ind: Any) -> Optional[Dict[str, Optional[float]]]:
        """Return envelopes snapshot dict if available."""
        try:
            up = Extract._last_from_list(ind, "Envelope_Upper_20_2.0") or Extract._last_from_list(ind, "Envelope_Upper_20_2")
            lo = Extract._last_from_list(ind, "Envelope_Lower_20_2.0") or Extract._last_from_list(ind, "Envelope_Lower_20_2")
            if None not in (up, lo):
                return {"upper": up, "lower": lo}
            node = ind.get("BBANDS") or ind.get("bbands") if isinstance(ind, dict) else None
            if isinstance(node, dict):
                def last(v):
                    if isinstance(v, list) and v: return ffloat(v[-1], None)
                    if isinstance(v, (int, float)): return ffloat(v, None)
                    return None
                up = last(node.get("upper") or node.get("upperband"))
                mid = last(node.get("middle") or node.get("middleband") or node.get("basis"))
                lo = last(node.get("lower") or node.get("lowerband"))
                if None not in (up, mid, lo):
                    return {"upper": up, "middle": mid, "lower": lo}
            return None
        except Exception:
            return None

    @staticmethod
    def obv_bias(ind: Any) -> Optional[int]:
        try:
            # column-style
            prev, curr = Extract._last_two_from_list(ind, "OBV")
            if prev is not None and curr is not None:
                return 1 if curr > prev else -1 if curr < prev else 0
            # dict-style
            node = ind.get("OBV") or ind.get("obv") if isinstance(ind, dict) else None
            if isinstance(node, list) and len(node) >= 2:
                return 1 if ffloat(node[-1]) > ffloat(node[-2]) else -1 if ffloat(node[-1]) < ffloat(node[-2]) else 0
            return None
        except Exception:
            return None

    # --- Additional extractors for more indicators present in indicator_output ---
    @staticmethod
    def mfi(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "MFI14")
            or Extract._last_from_list(ind, "MFI_14")
        )

    @staticmethod
    def williams_r(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "WilliamsR14")
            or Extract._last_from_list(ind, "WilliamsR_14")
            or Extract._last_from_list(ind, "WILLR14")
            or Extract._last_from_list(ind, "WILLR_14")
            or Extract._last_from_list(ind, "Williams %R")
        )

    @staticmethod
    def roc(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "ROC12")
            or Extract._last_from_list(ind, "ROC_12")
            or Extract._last_from_list(ind, "ROC")
        )

    @staticmethod
    def psar(ind: Any) -> Optional[float]:
        return (
            Extract._last_from_list(ind, "PSAR")
            or Extract._last_from_list(ind, "SAR")
        )

    @staticmethod
    def psar_dir(ind: Any) -> Optional[int]:
        """Return PSAR directional hint: +1 if price > psar, -1 if price < psar."""
        try:
            ps = Extract.psar(ind)
            price = Extract._get_price_from_ind(ind)
            if ps is None or price is None:
                return None
            return 1 if price > ps else -1 if price < ps else 0
        except Exception:
            return None

    @staticmethod
    def donchian_bias(ind: Any) -> Optional[float]:
        up = Extract._last_from_list(ind, "Donchian_Upper_20")
        lo = Extract._last_from_list(ind, "Donchian_Lower_20")
        price = Extract._get_price_from_ind(ind)
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def keltner_bias(ind: Any) -> Optional[float]:
        up = Extract._last_from_list(ind, "Keltner_Upper_20")
        lo = Extract._last_from_list(ind, "Keltner_Lower_20")
        price = Extract._get_price_from_ind(ind)
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def keltner(ind: Any) -> Optional[Dict[str, Optional[float]]]:
        up = Extract._last_from_list(ind, "Keltner_Upper_20")
        lo = Extract._last_from_list(ind, "Keltner_Lower_20")
        if None not in (up, lo):
            return {"upper": up, "lower": lo}
        return None

    @staticmethod
    def envelope_bias(ind: Any) -> Optional[float]:
        # Try common exact keys first
        up = Extract._last_from_list(ind, "Envelope_Upper_20_2.0") or Extract._last_from_list(ind, "Envelope_Upper_20_2")
        lo = Extract._last_from_list(ind, "Envelope_Lower_20_2.0") or Extract._last_from_list(ind, "Envelope_Lower_20_2")
        price = Extract._get_price_from_ind(ind)
        # Fallback: scan last bar for any Envelope_Upper/Lower keys with numeric values
        if (up is None or lo is None) and isinstance(ind, list) and ind and isinstance(ind[-1], dict):
            try:
                last = ind[-1]
                up_keys = [k for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith("Envelope_Upper")]
                lo_keys = [k for k, v in last.items() if isinstance(v, (int, float)) and str(k).startswith("Envelope_Lower")]
                if up is None and up_keys:
                    # Pick the lexicographically last (often the most parameterized)
                    up = ffloat(last[sorted(up_keys)[-1]])
                if lo is None and lo_keys:
                    lo = ffloat(last[sorted(lo_keys)[-1]])
            except Exception:
                pass
        if None in (up, lo, price):
            return None
        width = max(up - lo, 1e-6)
        pos = (price - lo) / width
        if pos < 0.2: return +1.0
        if pos > 0.8: return -1.0
        return 0.0

    @staticmethod
    def envelopes(ind: Any) -> Optional[Dict[str, Optional[float]]]:
        """Return envelopes snapshot dict if available."""
        try:
            up = Extract._last_from_list(ind, "Envelope_Upper_20_2.0") or Extract._last_from_list(ind, "Envelope_Upper_20_2")
            lo = Extract._last_from_list(ind, "Envelope_Lower_20_2.0") or Extract._last_from_list(ind, "Envelope_Lower_20_2")
            if None not in (up, lo):
                return {"upper": up, "lower": lo}
            node = ind.get("BBANDS") or ind.get("bbands") if isinstance(ind, dict) else None
            if isinstance(node, dict):
                def last(v):
                    if isinstance(v, list) and v: return ffloat(v[-1], None)
                    if isinstance(v, (int, float)): return ffloat(v, None)
                    return None
                up = last(node.get("upper") or node.get("upperband"))
                mid = last(node.get("middle") or node.get("middleband") or node.get("basis"))
                lo = last(node.get("lower") or node.get("lowerband"))
                if None not in (up, mid, lo):
                    return {"upper": up, "middle": mid, "lower": lo}
            return None
        except Exception:
            return None

    @staticmethod
    def obv_bias(ind: Any) -> Optional[int]:
        try:
            # column-style
            prev, curr = Extract._last_two_from_list(ind, "OBV")
            if prev is not None and curr is not None:
                return 1 if curr > prev else -1 if curr < prev else 0
            # dict-style
            node = ind.get("OBV") or ind.get("obv") if isinstance(ind, dict) else None
            if isinstance(node, list) and len(node) >= 2:
                return 1 if ffloat(node[-1]) > ffloat(node[-2]) else -1 if ffloat(node[-1]) < ffloat(node[-2]) else 0
            return None
        except Exception:
            return None


# ------------------------------
# Reporting EN + VI
# ------------------------------
def _translate_vi_to_en(text: str) -> str:
    """Lightweight VI->EN textual mapping for report lines."""
    if not isinstance(text, str) or not text:
        return ""
    # Normalize to NFC to ensure diacritics match mapping keys
    try:
        import unicodedata
        text = unicodedata.normalize("NFC", text)
    except Exception:
        pass
    replacements = [
        # Headers and key labels
        ("Thời gian:", "Time:"),
        ("Ký hiệu:", "Symbol:"),
        ("Tín hiệu:", "Signal:"),
        ("Độ tin cậy:", "Confidence:"),
        ("Điểm vào lệnh:", "Entry:"),
        ("Điểm vào:", "Entry:"),
        ("Cắt lỗ:", "Stoploss:"),
        ("Chốt lời:", "Takeprofit:"),
        ("Tóm tắt:", "Summary:"),
        ("Phân tích kỹ thuật:", "Technical analysis:"),
        ("Xu hướng:", "Trend:"),
        ("Kênh giá:", "Price channel:"),
        ("Kháng cự gần nhất:", "Nearest resistance:"),
        ("Hỗ trợ gần nhất:", "Nearest support:"),
        ("Vùng kháng cự:", "Resistance zone:"),
        ("Vùng hỗ trợ:", "Support zone:"),
        ("Mô hình giá:", "Price patterns:"),
        ("Mô hình nến:", "Candlestick patterns:"),
        ("Giá hiện tại:", "Current price:"),
        ("Chỉ báo:", "Indicator:"),
        ("Mô hình:", "Pattern:"),
        ("Khung thời gian:", "Timeframe:"),
        ("sức mạnh", "strength"),
    ("Chỉ báo kỹ thuật:", "Technical indicators:"),
    ("CẢNH BÁO:", "WARNING:"),
    ("Trong kênh", "Inside channel"),
    (" – chờ breakout", " – wait for breakout"),
    ("chờ breakout", "wait for breakout"),
    (" – chờ phá vỡ", " – wait for breakout"),
    ("chờ phá vỡ", "wait for breakout"),
    ("breakout lên", "breakout up"),
    ("breakout xuống", "breakout down"),
    ("Trendline đi ngang", "Trendline flat"),
    ("đi ngang", "sideways"),
    ("đang dốc lên", "sloping up"),
    ("đang dốc xuống", "sloping down"),
    ("trong dải", "within bands"),
    ("độ rộng trung bình", "average width"),
    ("độ rộng", "width"),
    ("rộng", "wide"),
    ("biến động cao", "high volatility"),
    ("biến động thấp", "low volatility"),
    ("ngắn hạn", "short-term"),
    ("trung hạn", "medium-term"),
    ("dài hạn", "long-term"),
    ("trung-dài hạn", "medium-long term"),
    ("trọng số ngắn hạn", "weighted short-term"),
    ("trọng số", "weighted"),
    ("xác nhận đà", "momentum confirmation"),
    ("Giá near", "Price near"),
    ("Giá gần", "Price near"),
    ("Giá ", "Price "),
    ("mức", "level"),
    ("Ủng hộ", "Supports"),
    ("ủng hộ", "supports"),
    ("TĂNG", "UP"),
    ("GIẢM", "DOWN"),
    ("Đảo chiều", "Reversal"),
    ("Tiếp diễn", "Continuation"),
    ("Tích lũy", "Accumulation"),
    ("Phản ứng giá", "Price reaction"),
    ("Mô hình", "Pattern"),
    ("rất cao", "very high"),
    ("khá", "fair"),
    ("cao hơn", "above"),
    ("thấp hơn", "below"),
    ("cao", "high"),
    ("trung bình", "average"),
    ("Độ hội tụ chỉ báo:", "Indicator convergence:"),
    ("Độ hội tụ chỉ báo", "Indicator convergence"),
    ("Độ convergence chỉ báo:", "Indicator convergence:"),
    ("Độ convergence chỉ báo", "Indicator convergence"),
    ("Chỉ số hội tụ:", "Convergence score:"),
    ("Chỉ số hội tụ", "Convergence score"),
    ("Chỉ số convergence:", "Convergence score:"),
    ("Chỉ số convergence", "Convergence score"),
    ("Biến động", "Volatility"),
    ("Top tín hiệu nổi bật", "Top notable signals"),
    ("Phân rã xung đột", "Conflict breakdown"),
    ("Trung tính", "Neutral"),
    ("trung tính", "neutral"),
    ("dương", "positive"),
    ("âm", "negative"),
    ("nghiêng", "biased toward"),
    ("so với giá", "relative to price"),
    ("chưa phát hiện clear", "not clearly detected"),
    ("chưa phát hiện", "not detected"),
    ("chưa rõ", "unclear"),
    ("(chưa phát hiện clear)", "(not clearly detected)"),
    ("kỳ", "periods"),
    ("giá =", "price ="),
    ("giá", "price"),
    ("không near level quan trọng", "not near an important level"),
    ("quan trọng", "important"),
    ("không", "not"),
    ("fibo", "Fibonacci"),
    ("Trendline breakout lên", "Trendline breakout up"),
    ("Trendline breakout xuống", "Trendline breakout down"),

        # Common words/phrases
    ("thu hẹp (nén)", "narrowing (squeeze)"),
    ("mở rộng (biến động cao)", "expanding (high volatility)"),
    ("xếp chồng tăng", "bullish stack"),
    ("xếp chồng giảm", "bearish stack"),
    ("xếp chồng up", "bullish stack"),  # safety in case generic 'tăng' translated first
    ("EMA20 đang dốc lên", "EMA20 sloping up"),
    ("EMA20 đang dốc xuống", "EMA20 sloping down"),
    ("Ultimate Osc tăng so với nến trước", "Ultimate Osc higher than previous candle"),
    ("Ultimate Osc giảm so với nến trước", "Ultimate Osc lower than previous candle"),
    ("ROC đổi dấu (động lượng đổi chiều)", "ROC sign change (momentum shift)"),
    ("ROC tăng tốc (mạnh hơn)", "ROC accelerating (stronger)"),
    ("DPO vừa vượt 0 (pha tăng chu kỳ)", "DPO just crossed above 0 (cycle upswing)"),
    ("DPO vừa rơi dưới 0 (pha giảm chu kỳ)", "DPO just fell below 0 (cycle downswing)"),
    ("biến động vừa", "medium volatility"),
    ("biến động thấp", "low volatility"),  # ensure explicit mapping even if already implied
    ("chu kỳ phía trên nền", "cycle above baseline"),
    ("chu kỳ phía dưới nền", "cycle below baseline"),
    ("chu kỳ cân bằng", "cycle balanced"),
    ("cảnh báo đảo chiều", "reversal warning"),
    ("tiệm cận cảnh báo", "approaching warning"),
    ("bình thường", "normal"),
    ("xu hướng lên mạnh", "strong uptrend"),
    ("ưu thế lên nhẹ", "slight bullish bias"),
    ("xu hướng xuống mạnh", "strong downtrend"),
    ("ưu thế xuống nhẹ", "slight bearish bias"),
    ("cân bằng", "balanced"),
    ("gần quá mua", "near overbought"),
    ("tăng ổn định", "steady rise"),
    ("gần quá bán", "near oversold"),
    ("giảm nhẹ", "mild decline"),
    ("quá mua dòng tiền", "money flow overbought"),
    ("dòng tiền mạnh", "strong money flow"),
    ("quá bán dòng tiền", "money flow oversold"),
    ("dòng tiền yếu", "weak money flow"),
    ("tích lũy dòng tiền", "money flow accumulation"),
    ("phân phối dòng tiền", "money flow distribution"),
    ("dòng tiền trung tính", "neutral money flow"),
    ("dòng tiền vào", "money inflow"),
    ("dòng tiền ra", "money outflow"),
    ("dòng tiền cân bằng", "balanced money flow"),
    ("giá đi lên dễ dàng", "price rising with ease"),
    ("giá đi xuống dễ dàng", "price falling with ease"),
    ("chuyển động cân bằng", "balanced movement"),
    ("lực mua", "buying force"),
    ("lực bán", "selling force"),
    ("lực trung tính", "neutral force"),
    ("lực mua mạnh", "strong buying force"),
    ("lực mua vừa", "moderate buying force"),
    ("lực mua yếu", "weak buying force"),
    ("lực bán mạnh", "strong selling force"),
    ("lực bán vừa", "moderate selling force"),
    ("lực bán yếu", "weak selling force"),
    ("xung lực trung-dài hạn tăng", "medium-long term momentum up"),
    ("xung lực trung-dài hạn giảm", "medium-long term momentum down"),
    ("xung lực bằng phẳng", "flat momentum"),
    ("Tenkan > Kijun", "Tenkan > Kijun"),  # keep as-is (Japanese terms)
    ("Tenkan <= Kijun", "Tenkan <= Kijun"),
    ("trong kênh", "inside channel"),
    ("chờ phá vỡ", "awaiting breakout"),
    ("tín hiệu tăng", "bullish signal"),
    ("tín hiệu giảm", "bearish signal"),
    ("không khả dụng", "not available"),
    ("giá cao hơn", "price above"),
    ("giá thấp hơn", "price below"),
    ("giá bám sát", "price hugging"),
    ("đang lãi; chốt bớt 50%", "in profit; take partial 50%"),
    ("đang lãi; chốt bớt 30%", "in profit; take partial 30%"),
    ("đồng pha tín hiệu; khóa lợi nhuận bằng trailing", "aligned; lock gains with trailing"),
    ("đang lãi; tín hiệu chưa rõ/khác pha — chốt bớt", "in profit; ambiguous/misaligned signal — partial close"),
    ("đang lãi nhẹ; chờ tín hiệu rõ ràng", "small profit; wait for clearer signal"),
    ("Vị thế ngược tín hiệu; độ tin cậy cao", "Position against signal; high confidence"),
    ("Vị thế ngược tín hiệu; lỗ lớn", "Position against signal; large loss"),
    ("Vị thế ngược tín hiệu; ưu tiên giảm rủi ro", "Position against signal; reduce risk first"),
    ("Đồng pha tín hiệu", "Aligned with signal"),
    ("(không tính được)", "(not computable)"),
    ("tạm tính", "approx"),
    ("ngắn hạn", "short-term"),  # duplicate safe
    ("trung hạn", "medium-term"),
    ("dài hạn", "long-term"),
    ("trung-dài hạn", "medium-long term"),
    ("yếu", "weak"),
    ("vừa", "moderate"),
    ("mạnh", "strong"),
    ("có thể sắp bứt phá", "potential breakout soon"),
        ("tăng mạnh", "strong up"),
        ("giảm mạnh", "strong down"),
        ("tăng", "up"),
        ("giảm", "down"),
        ("trung tính", "neutral"),
        ("đang hình thành", "forming"),
        ("rõ ràng", "clear"),
        ("tiếp diễn", "continuation"),
        ("đảo chiều", "reversal"),
        ("xác nhận", "confirmation"),
        ("phá vỡ", "breakout"),
        ("động lượng", "momentum"),
        ("hội tụ", "convergence"),
        ("phân kỳ", "divergence"),
        ("tích cực", "positive"),
        ("tiêu cực", "negative"),
        ("không có", "none"),
        ("gần", "near"),
        ("xa", "far"),
        ("vượt", "break above"),
        ("thủng", "break below"),

        # Footer/fallback
        ("(Bản tóm tắt tối giản do lỗi kết xuất báo cáo chi tiết)", "(Minimal summary due to report rendering issue)"),
    ]
    # Apply longer phrases first; run a few passes to catch staged conversions
    try:
        sorted_repls = sorted(replacements, key=lambda p: len(p[0]), reverse=True)
    except Exception:
        sorted_repls = replacements
    out = text
    for _ in range(3):
        before = out
        for vi, en in sorted_repls:
            try:
                # Normalize keys if unicodedata is available in this scope
                vi_key = unicodedata.normalize("NFC", vi)
            except Exception:
                vi_key = vi
            out = out.replace(vi_key, en)
        if out == before:
            break

    # Fallback secondary replacements (rare leftovers)
    leftovers = {
        'thu hẹp (nén)': 'narrowing (squeeze)',
        'xếp chồng tăng': 'bullish stack',
        'có thể sắp bứt phá': 'potential breakout soon'
    }
    for k,v in leftovers.items():
        out = out.replace(k, v)
    # Accent-insensitive fallback for stubborn phrases
    try:
        import unicodedata, re
        def strip_acc(s: str) -> str:
            return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        accent_map = {
            'thu hep (nen)': 'narrowing (squeeze)',
            'xep chong tang': 'bullish stack',
            'co the sap but pha': 'potential breakout soon'
        }
        lowered = strip_acc(out).lower()
        for raw, rep in accent_map.items():
            if raw in lowered:
                # Build a regex to replace original (accented) variants heuristically
                pattern = re.compile(r'x[eé]p ch[oô]ng (?:t[aă]ng|up)', re.IGNORECASE)
                if 'xep chong' in raw:
                    out = pattern.sub('bullish stack', out)
                else:
                    # direct replacements for other phrases
                    for variant in ['thu hẹp (nén)','thu hep (nen)']:
                        out = out.replace(variant, 'narrowing (squeeze)')
                    for variant in ['có thể sắp bứt phá','co the sap but pha']:
                        out = out.replace(variant, 'potential breakout soon')
    except Exception:
        pass
    return out


class Report:
    @staticmethod
    def _tf_block(tf: str, node: Dict[str, Any], lang: str) -> str:
        price = node.get("price")
        ind = node.get("indicator", {})
        pat = node.get("patterns", {})
        sr = node.get("sr", {})
        comp = node.get("composite", {})
        if lang == "vi":
            lines = [
                f"[Khung thời gian: {tf}]",
                f"  Giá hiện tại: {price}",
                f"  Chỉ báo: {ind.get('signal')} (sức mạnh {round(ind.get('strength',0)*100)}%)",
                f"  Mô hình: {pat.get('signal')} (sức mạnh {round(pat.get('strength',0)*100)}%)",
                # lược bỏ điểm tổng hợp để báo cáo gọn hơn
            ]
            if sr:
                sup = sr.get('nearest_support')
                res = sr.get('nearest_resistance')
                if sup is not None:
                    lines.append(f"  Hỗ trợ gần nhất: {sup} (~{round(sr.get('support_distance_percent') or 0,2)}%)")
                if res is not None:
                    lines.append(f"  Kháng cự gần nhất: {res} (~{round(sr.get('resistance_distance_percent') or 0,2)}%)")
                # bỏ ATR% để tránh rườm rà
            return "\n".join(lines)
        else:
            # Trimmed English block: only show timeframe header; omit price/indicators/patterns
            lines = [
                f"[Timeframe: {tf}]",
            ]
            if sr:
                sup = sr.get('nearest_support')
                res = sr.get('nearest_resistance')
                if sup is not None:
                    lines.append(f"  Nearest support: {sup} (~{round(sr.get('support_distance_percent') or 0,2)}%)")
                if res is not None:
                    lines.append(f"  Nearest resistance: {res} (~{round(sr.get('resistance_distance_percent') or 0,2)}%)")
                # drop ATR% to reduce noise
            return "\n".join(lines)

    @staticmethod
    def build(analysis: Dict[str, Any], lang: str) -> str:
        # If Vietnamese requested, use the structured VI builder
        if lang == "vi":
            return Report._build_vietnamese_structured(analysis)

        # English report (without Additional Info footer)
        sym = analysis.get("symbol")
        ts = analysis.get("timestamp")
        final = analysis.get("final_signal", {})
        tfs = analysis.get("timeframes", {})
        idea = analysis.get("trade_idea")

        header = [
            "="*80,
            f"MULTI-TIMEFRAME ANALYSIS REPORT - {sym}",
            f"Analysis time: {ts}",
            "="*80,
            "",
            "SIGNAL SUMMARY",
            f"  Recommendation: {final.get('signal')}",
            f"  Confidence: {final.get('confidence')}%",
            "",
            "TRADE IDEA",
            (
                f"  Direction: {idea.get('direction')} | TF: {idea.get('timeframe')}\n"
                f"  Entry: {idea.get('entry')} | SL: {idea.get('sl')} | TP: {idea.get('tp')} | RR: {idea.get('rr')}\n"
            ) if idea else "  (No trade suggestion)",
            "",
        ]

        # English timeframe details removed per request; nothing appended here.
        return "\n".join(header) + "\n"

    @staticmethod
    def _build_vietnamese_structured(analysis: Dict[str, Any]) -> str:
        """Render VI report with header (time/symbol/signal) and Entry/SL/TP on top, plus enriched indicator/trend context."""
        sym = analysis.get("symbol")
        ts = analysis.get("timestamp")
        final = analysis.get("final_signal", {}) or {}
        idea = analysis.get("trade_idea") or {}
        tfs: Dict[str, Any] = analysis.get("timeframes", {}) or {}
    # Whitelist removed: we trust indicator_output contents. No indicator gating here.

        # Helpers and TF selection
        loader = Loader(sym or "")
        def tf_ordered() -> List[str]:
            # Try to get from analysis result first (user selection)
            if analysis.get("available_timeframes"):
                return analysis["available_timeframes"]
            
            # Fallback: If user specified timeframes, use those first; otherwise auto-detect from data files
            import glob
            available_files = glob.glob(f"data/{sym}_m_*.json")
            if not available_files:
                arr = list(tfs.keys())
                pref = list(CFG.TF)
                return [tf for tf in pref if tf in arr]
            else:
                available_tfs = [f.split('_m_')[1].split('.json')[0] for f in available_files]
                pref = list(CFG.TF)
                return [tf for tf in pref if tf in available_tfs]
        def pick(tf_list: List[str], target: List[str]) -> Optional[str]:
            for t in target:
                if t in tf_list:
                    return t
            return tf_list[0] if tf_list else None
        tflist = tf_ordered()
        # Prefer higher TFs for trend as requested: MN1 > W1 > D1 > H4 > H1 (but only from available timeframes)
        tf_trend = pick(tflist, ["MN1", "W1", "D1", "H4", "H1"]) or (tflist[0] if tflist else "H1")
        # Upgrade: use the largest timeframe for channel and trendline sections as requested
        tf_chan = tf_trend

        # Load TF data
        def safe_load_tf(tf: str) -> TFData:
            try:
                return loader.load_tf(tf)
            except Exception:
                return TFData(candles=None, indicators=None, price_patterns=None, priority_patterns=None, sr=None)
        data_trend = safe_load_tf(tf_trend)
        data_chan = safe_load_tf(tf_chan)
        # Preload all available TF data for deeper fallbacks
        tf_data_cache: Dict[str, TFData] = {}
        for _tf in tflist:
            if _tf not in tf_data_cache:
                tf_data_cache[_tf] = safe_load_tf(_tf)
        # Only load additional timeframes that are actually available in the analysis data
        # DO NOT load all timeframes by default - respect user's timeframe selection
        try:
            # Only preload pattern TFs that are explicitly in the analysis timeframes
            available_pattern_tfs = [tf for tf in tflist if tf in ["D1", "H4", "H1", "M30", "M15", "M5"]]
            for _tf in available_pattern_tfs:
                if _tf not in tf_data_cache:
                    tf_data_cache[_tf] = safe_load_tf(_tf)
                    try:
                        logger.info(f"[REPORT] Preloaded pattern TF {_tf} for {sym}")
                    except Exception:
                        pass
        except Exception:
            try:
                logger.warning("[REPORT] Failed pattern timeframe preload", exc_info=True)
            except Exception:
                pass

        # Price and indicators maps
        price_trend = Extract.last_close(data_trend.candles) if data_trend.candles else None
        # Fallback: if price_trend is None/NaN, try last valid close from candles list
        if (price_trend is None or (isinstance(price_trend,(int,float)) and price_trend != price_trend)) and data_trend and data_trend.candles:
            try:
                closes_list = []
                c_src = data_trend.candles.get('rates') if isinstance(data_trend.candles, dict) else data_trend.candles
                if isinstance(c_src, list):
                    for r in reversed(c_src):
                        if isinstance(r, dict) and 'close' in r:
                            v = r.get('close')
                            if isinstance(v,(int,float)) and v==v:
                                price_trend = v
                                break
            except Exception:
                pass
        # Keep indicators as-is (list or dict). Many exporters produce a list of bar dicts.
        ind_trend = data_trend.indicators if (data_trend and data_trend.indicators is not None) else {}
        ind_chan = data_chan.indicators if (data_chan and data_chan.indicators is not None) else {}
        print(f"DEBUG: data_trend.indicators type: {type(data_trend.indicators) if data_trend else 'No data_trend'}")
        print(f"DEBUG: data_chan.indicators type: {type(data_chan.indicators) if data_chan else 'No data_chan'}")
        
        # Extract current price from indicator data for use in Aggregator
        global current_price_from_indicators
        current_price_from_indicators = None
        
        if isinstance(ind_trend, list) and ind_trend and isinstance(ind_trend[-1], dict):
            print(f"DEBUG: ind_trend last row sample keys: {list(ind_trend[-1].keys())[:15]}")
            last_row = ind_trend[-1]
            
            # Try to extract current price
            for key in ['close', 'c', 'Close', 'open', 'o', 'Open', 'high', 'h', 'High', 'low', 'l', 'Low']:
                if key in last_row:
                    try:
                        current_price_from_indicators = float(last_row[key])
                        print(f"DEBUG: Extracted price from ind_trend {key}: {current_price_from_indicators}")
                        break
                    except (ValueError, TypeError):
                        continue
                        
        # Fallback to chan data if no price from trend
        if not current_price_from_indicators and isinstance(ind_chan, list) and ind_chan and isinstance(ind_chan[-1], dict):
            last_row = ind_chan[-1]
            for key in ['close', 'c', 'Close', 'open', 'o', 'Open', 'high', 'h', 'High', 'low', 'l', 'Low']:
                if key in last_row:
                    try:
                        current_price_from_indicators = float(last_row[key])
                        print(f"DEBUG: Extracted price from ind_chan {key}: {current_price_from_indicators}")
                        break
                    except (ValueError, TypeError):
                        continue
                        
        if isinstance(ind_trend, list) and ind_trend and isinstance(ind_trend[-1], dict):
            print(f"DEBUG: ind_trend last row sample keys: {list(ind_trend[-1].keys())[:15]}")
            # Look for BB fields specifically
            bb_keys = [k for k in ind_trend[-1].keys() if 'BB' in k.upper()]
            print(f"DEBUG: BB-related keys found: {bb_keys}")
            
            # Extract parameters from all indicators dynamically
            bb_win_detected = None
            bb_dev_detected = None
            kelt_win_detected = None
            env_win_detected = None
            env_dev_detected = None
            donch_win_detected = None
            eom_win_detected = None
            dpo_win_detected = None
            chaikin_win_detected = None
            trix_win_detected = None
            mass_fast_detected = None
            mass_slow_detected = None
            ichi_tenkan_detected = None
            ichi_kijun_detected = None
            ichi_senkou_detected = None
            
            # Add universal indicator parameter detection
            rsi_period_detected = None
            adx_period_detected = None
            atr_period_detected = None
            mfi_period_detected = None
            cci_period_detected = None
            willr_period_detected = None
            roc_period_detected = None
            stoch_k_detected = None
            stoch_d_detected = None
            stochrsi_period_detected = None
            macd_fast_detected = None
            macd_slow_detected = None
            macd_signal_detected = None
            
            # Collect all available MA periods dynamically
            ema_periods = set()
            sma_periods = set()
            wma_periods = set()
            tema_periods = set()
            
            for bb_key in bb_keys:
                import re
                m = re.fullmatch(r"BB_Upper_(\d+)_([0-9]+(?:\.[0-9]+)?)", bb_key)
                if not m:
                    m = re.fullmatch(r"BB_Middle_(\d+)_([0-9]+(?:\.[0-9]+)?)", bb_key)
                if not m:
                    m = re.fullmatch(r"BB_Lower_(\d+)_([0-9]+(?:\.[0-9]+)?)", bb_key)
                if m:
                    bb_win_detected = int(m.group(1))
                    bb_dev_detected = float(m.group(2))
                    print(f"DEBUG: Detected BB params from key '{bb_key}': bb_win={bb_win_detected}, bb_dev={bb_dev_detected}")
                    break
            
            # Universal indicator parameter detection
            all_keys = list(ind_trend[-1].keys())
            for key in all_keys:
                # RSI patterns: RSI14, RSI_14, rsi14, etc.
                m = re.fullmatch(r"RSI[_]?(\d+)", key, re.IGNORECASE)
                if m and rsi_period_detected is None:
                    rsi_period_detected = int(m.group(1))
                    print(f"DEBUG: Detected RSI params from key '{key}': rsi_period={rsi_period_detected}")
                
                # ADX patterns: ADX14, ADX_14, adx14, etc.
                m = re.fullmatch(r"ADX[_]?(\d+)", key, re.IGNORECASE)
                if m and adx_period_detected is None:
                    adx_period_detected = int(m.group(1))
                    print(f"DEBUG: Detected ADX params from key '{key}': adx_period={adx_period_detected}")
                
                # ATR patterns: ATR14, ATR_14, atr14, etc.
                m = re.fullmatch(r"ATR[_]?(\d+)", key, re.IGNORECASE)
                if m and atr_period_detected is None:
                    atr_period_detected = int(m.group(1))
                    print(f"DEBUG: Detected ATR params from key '{key}': atr_period={atr_period_detected}")
                
                # MFI patterns: MFI14, MFI_14, mfi14, etc.
                m = re.fullmatch(r"MFI[_]?(\d+)", key, re.IGNORECASE)
                if m and mfi_period_detected is None:
                    mfi_period_detected = int(m.group(1))
                    print(f"DEBUG: Detected MFI params from key '{key}': mfi_period={mfi_period_detected}")
                
                # CCI patterns: CCI20, CCI_20, cci20, etc.
                m = re.fullmatch(r"CCI[_]?(\d+)", key, re.IGNORECASE)
                if m and cci_period_detected is None:
                    cci_period_detected = int(m.group(1))
                    print(f"DEBUG: Detected CCI params from key '{key}': cci_period={cci_period_detected}")
                
                # Williams %R patterns: WilliamsR14, WILLR14, etc.
                m = re.fullmatch(r"(?:Williams?R|WILLR)[_]?(\d+)", key, re.IGNORECASE)
                if m and willr_period_detected is None:
                    willr_period_detected = int(m.group(1))
                    print(f"DEBUG: Detected Williams %R params from key '{key}': willr_period={willr_period_detected}")
                
                # ROC patterns: ROC20, ROC_20, roc12, etc.
                m = re.fullmatch(r"ROC[_]?(\d+)", key, re.IGNORECASE)
                if m and roc_period_detected is None:
                    roc_period_detected = int(m.group(1))
                    print(f"DEBUG: Detected ROC params from key '{key}': roc_period={roc_period_detected}")
                
                # Stochastic patterns: Stoch_K_5_3, Stoch_D_5_3, etc.
                m = re.fullmatch(r"Stoch[_]?K[_]?(\d+)[_]?(\d+)", key, re.IGNORECASE)
                if m and stoch_k_detected is None:
                    stoch_k_detected = int(m.group(1))
                    stoch_d_detected = int(m.group(2))
                    print(f"DEBUG: Detected Stochastic params from key '{key}': stoch_k={stoch_k_detected}, stoch_d={stoch_d_detected}")
                
                # StochRSI patterns: StochRSI14, StochRSI_14, etc.
                m = re.fullmatch(r"StochRSI[_]?(\d+)", key, re.IGNORECASE)
                if m and stochrsi_period_detected is None:
                    stochrsi_period_detected = int(m.group(1))
                    print(f"DEBUG: Detected StochRSI params from key '{key}': stochrsi_period={stochrsi_period_detected}")
                
                # MACD patterns: MACD_14_26_9, MACD_fast_slow_signal, etc.
                m = re.fullmatch(r"MACD[_]?(\d+)[_](\d+)[_](\d+)", key, re.IGNORECASE)
                if m and macd_fast_detected is None:
                    macd_fast_detected = int(m.group(1))
                    macd_slow_detected = int(m.group(2))
                    macd_signal_detected = int(m.group(3))
                    print(f"DEBUG: Detected MACD params from key '{key}': fast={macd_fast_detected}, slow={macd_slow_detected}, signal={macd_signal_detected}")
                
                # EMA patterns: EMA20, EMA25, EMA50, EMA100, EMA200, etc.
                m = re.fullmatch(r"EMA[_]?(\d+)", key, re.IGNORECASE)
                if m:
                    period = int(m.group(1))
                    ema_periods.add(period)
                
                # SMA patterns: SMA20, SMA50, SMA100, SMA200, etc.
                m = re.fullmatch(r"SMA[_]?(\d+)", key, re.IGNORECASE)
                if m:
                    period = int(m.group(1))
                    sma_periods.add(period)
                
                # WMA patterns: WMA20, WMA50, WMA100, WMA200, etc.
                m = re.fullmatch(r"WMA[_]?(\d+)", key, re.IGNORECASE)
                if m:
                    period = int(m.group(1))
                    wma_periods.add(period)
                
                # TEMA patterns: TEMA20, TEMA50, TEMA100, TEMA200, etc.
                m = re.fullmatch(r"TEMA[_]?(\d+)", key, re.IGNORECASE)
                if m:
                    period = int(m.group(1))
                    tema_periods.add(period)
                
                # Existing patterns (keep these)
                # Keltner Channels
                m = re.fullmatch(r"Keltner_(?:Upper|Middle|Lower)_(\d+)", key)
                if m and kelt_win_detected is None:
                    kelt_win_detected = int(m.group(1))
                    print(f"DEBUG: Detected Keltner params from key '{key}': kelt_win={kelt_win_detected}")
                
                # Envelopes  
                m = re.fullmatch(r"Envelope_(?:Upper|Middle|Lower)_(\d+)_([0-9]+(?:\.[0-9]+)?)", key)
                if m and env_win_detected is None:
                    env_win_detected = int(m.group(1))
                    env_dev_detected = float(m.group(2))
                    print(f"DEBUG: Detected Envelope params from key '{key}': env_win={env_win_detected}, env_dev={env_dev_detected}")
                
                # Donchian Channels
                m = re.fullmatch(r"Donchian_(?:Upper|Middle|Lower)_(\d+)", key)
                if m and donch_win_detected is None:
                    donch_win_detected = int(m.group(1))
                    print(f"DEBUG: Detected Donchian params from key '{key}': donch_win={donch_win_detected}")
                
                # EOM (Ease of Movement)
                m = re.fullmatch(r"EOM(\d+)", key)
                if m and eom_win_detected is None:
                    eom_win_detected = int(m.group(1))
                    print(f"DEBUG: Detected EOM params from key '{key}': eom_win={eom_win_detected}")
                
                # DPO (Detrended Price Oscillator)
                m = re.fullmatch(r"DPO(\d+)", key)
                if m and dpo_win_detected is None:
                    dpo_win_detected = int(m.group(1))
                    print(f"DEBUG: Detected DPO params from key '{key}': dpo_win={dpo_win_detected}")
                
                # Chaikin Money Flow
                m = re.fullmatch(r"Chaikin(\d+)", key)
                if m and chaikin_win_detected is None:
                    chaikin_win_detected = int(m.group(1))
                    print(f"DEBUG: Detected Chaikin params from key '{key}': chaikin_win={chaikin_win_detected}")
                
                # TRIX
                m = re.fullmatch(r"TRIX(\d+)", key)
                if m and trix_win_detected is None:
                    trix_win_detected = int(m.group(1))
                    print(f"DEBUG: Detected TRIX params from key '{key}': trix_win={trix_win_detected}")
                
                # Mass Index
                m = re.fullmatch(r"MassIndex_(\d+)_(\d+)", key)
                if m and mass_fast_detected is None:
                    mass_fast_detected = int(m.group(1))
                    mass_slow_detected = int(m.group(2))
                    print(f"DEBUG: Detected Mass Index params from key '{key}': mass_fast={mass_fast_detected}, mass_slow={mass_slow_detected}")
                
                # Ichimoku Tenkan
                m = re.fullmatch(r"tenkan_(\d+)", key, re.IGNORECASE)
                if m and ichi_tenkan_detected is None:
                    ichi_tenkan_detected = int(m.group(1))
                    print(f"DEBUG: Detected Ichimoku Tenkan params from key '{key}': ichi_tenkan={ichi_tenkan_detected}")
                
                # Ichimoku Kijun  
                m = re.fullmatch(r"kijun_(\d+)", key, re.IGNORECASE)
                if m and ichi_kijun_detected is None:
                    ichi_kijun_detected = int(m.group(1))
                    print(f"DEBUG: Detected Ichimoku Kijun params from key '{key}': ichi_kijun={ichi_kijun_detected}")
                
                # Ichimoku Senkou (use kijun as senkou period is typically 26)
                m = re.fullmatch(r"senkou_[ab]_(\d+)", key, re.IGNORECASE)
                if m and ichi_senkou_detected is None:
                    ichi_senkou_detected = int(m.group(1))
                    print(f"DEBUG: Detected Ichimoku Senkou params from key '{key}': ichi_senkou={ichi_senkou_detected}")
            
            # Debug output for detected MA periods
            if ema_periods:
                print(f"DEBUG: Detected EMA periods: {sorted(ema_periods)}")
            if sma_periods:
                print(f"DEBUG: Detected SMA periods: {sorted(sma_periods)}")
            if wma_periods:
                print(f"DEBUG: Detected WMA periods: {sorted(wma_periods)}")
            if tema_periods:
                print(f"DEBUG: Detected TEMA periods: {sorted(tema_periods)}")
        if isinstance(ind_chan, list) and ind_chan and isinstance(ind_chan[-1], dict):
            print(f"DEBUG: ind_chan last row sample keys: {list(ind_chan[-1].keys())[:15]}")
        # Extract base OHLCV sequences for fallback calculations
        closes: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        volumes: List[float] = []
        try:
            candle_src = None
            if data_trend and data_trend.candles:
                if isinstance(data_trend.candles, dict):
                    for k in ('rates','candles','data','bars'):
                        if k in data_trend.candles and isinstance(data_trend.candles[k], list):
                            candle_src = data_trend.candles[k]
                            break
                elif isinstance(data_trend.candles, list):
                    candle_src = data_trend.candles
            if candle_src:
                for r in candle_src:
                    if isinstance(r, dict):
                        c = r.get('close'); h = r.get('high'); l = r.get('low'); v = r.get('volume', r.get('tick_volume'))
                        if isinstance(c,(int,float)): closes.append(c)
                        if isinstance(h,(int,float)): highs.append(h)
                        if isinstance(l,(int,float)): lows.append(l)
                        if isinstance(v,(int,float)): volumes.append(v)
        except Exception:
            pass
        # Secondary price fallback: use last close from extracted list if still missing
        if (price_trend is None or (isinstance(price_trend,(int,float)) and price_trend != price_trend)) and closes:
            price_trend = closes[-1]

        # Trend and SR snapshot with multi-TF fallbacks
        tfnode_trend = tfs.get(tf_trend, {}) or {}
        ind_sig = (tfnode_trend.get("indicator", {}) or {}).get("signal")
        if not ind_sig:
            for tfp in tflist:
                node = tfs.get(tfp, {}) or {}
                ind_sig = (node.get("indicator", {}) or {}).get("signal")
                if ind_sig:
                    break
        ind_sig = ind_sig or "NEUTRAL"
        trend_word = "tăng" if ind_sig == "BUY" else ("giảm" if ind_sig == "SELL" else "đi ngang")
        def pick_sr_any() -> Tuple[Optional[Any], Optional[Any], Optional[str], Optional[str]]:
            # Prefer highest timeframe first
            for tfp in tflist:
                node = tfs.get(tfp, {}) or {}
                srn = node.get("sr") if isinstance(node, dict) else None
                if isinstance(srn, dict):
                    su = srn.get("nearest_support"); re = srn.get("nearest_resistance")
                    if su is not None or re is not None:
                        return su, re, tfp, (srn.get("sr_event") if isinstance(srn, dict) else None)
            # Fallback to loaded TF SR files in order
            for tfp in tflist:
                d = tf_data_cache.get(tfp)
                srn = d.sr if d and isinstance(d.sr, dict) else None
                if isinstance(srn, dict):
                    su = srn.get("nearest_support"); re = srn.get("nearest_resistance")
                    if su is not None or re is not None:
                        return su, re, tfp, (srn.get("sr_event") if isinstance(srn, dict) else None)
            return None, None, None, None
        sup, res, sr_tf, sr_event = pick_sr_any()

        # Channel text from Donchian on tf_chan (prefer any available period)
        def _donch_from_ind_list(ind_list) -> tuple[Optional[float], Optional[float], Optional[int]]:
            """Return (lower, upper, period) by scanning last row for Donchian_*_<n> keys."""
            try:
                if not (isinstance(ind_list, list) and ind_list and isinstance(ind_list[-1], dict)):
                    return None, None, None
                last = ind_list[-1]
                periods: dict[int, dict[str, float]] = {}
                import re as _re
                for k, v in last.items():
                    if not isinstance(v, (int, float)):
                        continue
                    ks = str(k)
                    m = _re.fullmatch(r"(?:Donchian|DONCHIAN)_(?:Upper|UPPER)_(\d+)", ks)
                    if m:
                        p = int(m.group(1)); periods.setdefault(p, {})['upper'] = ffloat(v)
                        continue
                    m = _re.fullmatch(r"(?:Donchian|DONCHIAN)_(?:Lower|LOWER)_(\d+)", ks)
                    if m:
                        p = int(m.group(1)); periods.setdefault(p, {})['lower'] = ffloat(v)
                        continue
                if not periods:
                    return None, None, None
                # Prefer a period where both upper and lower exist; pick the largest such period
                both = [p for p, d in periods.items() if 'upper' in d and 'lower' in d]
                if both:
                    use_p = max(both)
                    return periods[use_p].get('lower'), periods[use_p].get('upper'), use_p
                # Fallback: take the largest period even if only one side exists
                use_p = max(periods.keys())
                return periods[use_p].get('lower'), periods[use_p].get('upper'), use_p
            except Exception:
                return None, None, None

        d_lo, d_up, d_win = _donch_from_ind_list(ind_chan) if ind_chan else (None, None, None)
        price_chan = Extract.last_close(data_chan.candles) if data_chan.candles else price_trend
        # If Donchian missing, compute from candles quickly (20-period) unless strict mode
        if (d_lo is None or d_up is None) and not CFG.STRICT_IND_ONLY:
            try:
                seq = None
                src = data_chan.candles or data_trend.candles
                if isinstance(src, dict):
                    for key in ("rates", "candles", "data", "bars"):
                        if isinstance(src.get(key), list):
                            seq = src.get(key)
                            break
                elif isinstance(src, list):
                    seq = src
                highs: List[float] = []
                lows: List[float] = []
                if isinstance(seq, list):
                    for b in seq:
                        if isinstance(b, dict):
                            h = b.get("high") or b.get("h")
                            l = b.get("low") or b.get("l")
                            if h is not None and l is not None:
                                highs.append(ffloat(h)); lows.append(ffloat(l))
                if len(highs) >= 20 and len(lows) >= 20:
                    up_calc = max(highs[-20:])
                    lo_calc = min(lows[-20:])
                    if d_up is None:
                        d_up = up_calc
                    if d_lo is None:
                        d_lo = lo_calc
            except Exception:
                pass
        # Optionally fallback SR to Donchian bounds if missing
        if (sup is None and isinstance(d_lo, (int, float))):
            sup = d_lo
            if not sr_tf:
                sr_tf = tf_chan
        if (res is None and isinstance(d_up, (int, float))):
            res = d_up
            if not sr_tf:
                sr_tf = tf_chan
        # Ensure price formatting helpers are available before channel text
        def price_decimals(symbol: Optional[str]) -> int:
            """Return price decimals by symbol: XAU=2, JPY pairs=3, crypto=8, else 4."""
            try:
                s = (symbol or "").upper()
            except Exception:
                s = ""
            # Metals
            if s.startswith("XAU") or s == "GOLD" or s == "XAUUSD":
                return 2
            # JPY pairs
            if s.endswith("JPY"):
                return 3
            # Crypto (common tickers)
            crypto_tags = ("BTC", "ETH", "XRP", "SOL", "ADA", "DOGE", "BNB", "TRX", "LTC", "DOT", "AVAX", "USDT", "USDC")
            if any(tag in s for tag in crypto_tags):
                return 8
            return 4
        price_nd = price_decimals(sym)
        def pfmt(val: Optional[float]) -> str:
            if not isinstance(val, (int, float)):
                return "-"
            v = float(val)
            s_upper = (sym or "").upper()
            # Detect crypto (same tags as above price_decimals)
            crypto_tags = ("BTC", "ETH", "XRP", "SOL", "ADA", "DOGE", "BNB", "TRX", "LTC", "DOT", "AVAX", "USDT", "USDC")
            is_crypto = any(tag in s_upper for tag in crypto_tags)
            if is_crypto:
                # Adaptive precision: big numbers 2dp, medium 4dp, tiny 6dp
                if abs(v) >= 1000:
                    raw = f"{v:.2f}"
                elif abs(v) >= 1:
                    raw = f"{v:.4f}"
                else:
                    raw = f"{v:.6f}"
            else:
                raw = f"{v:.{price_nd}f}"
            # Trim trailing zeros & dot
            if "." in raw:
                raw = raw.rstrip('0').rstrip('.')
            return raw

        chan_txt = None
        chan_break_note: Optional[str] = None
        if isinstance(d_lo, (int, float)) and isinstance(d_up, (int, float)) and price_chan:
            lo = float(d_lo); up = float(d_up)
            if up > lo:
                w = (up - lo) / max(price_chan, 1e-12)
                # Channel width formatting capped at 4 decimals
                def _wfmt(x: float) -> str:
                    txt = f"{x:.4f}" if abs(x) < 1000 else f"{x:.2f}"
                    if '.' in txt:
                        txt = txt.rstrip('0').rstrip('.')
                    return txt
                if price_chan > up:
                    chan_txt = f"{tf_chan}: Phá vỡ lên khỏi kênh {pfmt(lo)}-{pfmt(up)} (W={_wfmt(w)})"
                    chan_break_note = "Breakout kênh: biên trên"
                elif price_chan < lo:
                    chan_txt = f"{tf_chan}: Phá vỡ xuống khỏi kênh {pfmt(lo)}-{pfmt(up)} (W={_wfmt(w)})"
                    chan_break_note = "Breakout kênh: biên dưới"
                else:
                    chan_txt = f"{tf_chan}: Trong kênh {pfmt(lo)}-{pfmt(up)} (W={_wfmt(w)}) – chờ phá vỡ"
        if not chan_txt:
            chan_txt = f"{tf_chan}: Trong kênh – chờ phá vỡ"

        # Trendline from SR slope (prefer channel tf; fallback others)
        def _extract_trend_slope(sr_obj: Optional[dict]) -> Optional[float]:
            """Return a numeric slope if present in common SR schemas.
            Accepts:
            - sr["trendline"] as dict with key "slope"
            - sr["trend_slope"] as a float
            - sr["slope"] as a float (alternative naming)
            """
            if not isinstance(sr_obj, dict):
                return None
            # Case 1: nested trendline dict with slope
            tl = sr_obj.get("trendline")
            if isinstance(tl, dict) and isinstance(tl.get("slope"), (int, float)):
                return float(tl.get("slope"))
            # Case 2: flat numeric fields
            for key in ("trend_slope", "slope"):
                val = sr_obj.get(key)
                if isinstance(val, (int, float)):
                    return float(val)
            # Case 3: parse from text value if available (e.g., "Slope: 0.00012")
            txt = sr_obj.get("trendline_value") or sr_obj.get("trend_value")
            if isinstance(txt, str) and "Slope:" in txt:
                try:
                    part = txt.split("Slope:", 1)[1].strip()
                    # take first token that looks like a number
                    tok = part.split()[0].strip("|,")
                    return float(tok)
                except Exception:
                    return None
            return None

        def _extract_trend_dir(sr_obj: Optional[dict]) -> Optional[str]:
            if not isinstance(sr_obj, dict):
                return None
            dir_txt = sr_obj.get("trend_direction") or sr_obj.get("trend") or sr_obj.get("direction")
            if isinstance(dir_txt, str):
                return dir_txt.strip().lower()
            return None

        sr_chan = data_chan.sr if isinstance(data_chan.sr, dict) else {}
        slope = _extract_trend_slope(sr_chan)
        dir_txt = _extract_trend_dir(sr_chan)
        # Trendline breakout sniff: use SR flags if present
        trendline_break_note: Optional[str] = None
        def _infer_break_note(obj: dict) -> Optional[str]:
            if not isinstance(obj, dict):
                return None
            # Direct textual breakout field
            br_txt = obj.get('breakout') or obj.get('breakout_summary') or obj.get('breakout_type')
            def _classify(txt: str) -> Optional[str]:
                low = txt.lower()
                # Reversal (đảo chiều) detection first so we keep that nuance
                if 'reversal' in low and 'down' in low:
                    return 'BreakDown đảo chiều xuống'
                if 'reversal' in low and ('up' in low or 'break up' in low):
                    return 'BreakOut đảo chiều lên'
                if 'break down' in low or 'breakout down' in low or 'breakdown' in low:
                    return 'BreakDown xuống'
                if 'break out' in low or 'breakout up' in low or 'break up' in low or 'breakout (break up' in low:
                    return 'BreakOut lên'
                return None
            if isinstance(br_txt, str):
                lab = _classify(br_txt)
                if lab:
                    return lab
            # Nested analysis
            ba = obj.get('breakout_analysis')
            if isinstance(ba, dict):
                btyp = ba.get('breakout_type') or ''
                bdir = (ba.get('breakout_direction') or '').lower()
                lab = _classify(str(btyp))
                if not lab and bdir:
                    if bdir == 'up':
                        lab = 'BreakOut lên'
                    elif bdir == 'down':
                        lab = 'BreakDown xuống'
                if lab:
                    return lab
            # Generic flags
            for k in ("trendline_breakout", "trend_breakout", "break_trendline", "trendline_event"):
                val = obj.get(k)
                if isinstance(val, str) and 'break' in val.lower():
                    return 'BreakOut trendline'
                if isinstance(val, bool) and val:
                    return 'BreakOut trendline'
            return None
        if isinstance(sr_chan, dict):
            trendline_break_note = _infer_break_note(sr_chan)
        if slope is None:
            # Try from higher-priority TFs
            for tfp in tflist:
                d = tf_data_cache.get(tfp)
                srp = d.sr if d and isinstance(d.sr, dict) else None
                slope = _extract_trend_slope(srp)
                dir_txt = _extract_trend_dir(srp) or dir_txt
                if trendline_break_note is None and isinstance(srp, dict):
                    trendline_break_note = _infer_break_note(srp)
                if slope is not None or (isinstance(dir_txt, str) and dir_txt in ("sideway", "flat", "neutral")):
                    tf_chan = tfp
                    break

        # --- Simplified Trendline description ---
        def _norm_dir(raw: Optional[str], slp: Optional[float]) -> str:
            raw_l = (raw or '').lower()
            mapping_pos = ("up","uptrend","increase","increasing","bull","bullish","tang","tăng")
            mapping_neg = ("down","downtrend","decrease","decreasing","bear","bearish","giam","giảm")
            mapping_flat = ("side","sideway","flat","neutral","rang","ngang")
            if raw_l:
                if any(k in raw_l for k in mapping_pos): return "tăng"
                if any(k in raw_l for k in mapping_neg): return "giảm"
                if any(k in raw_l for k in mapping_flat): return "đi ngang"
            if isinstance(slp,(int,float)):
                if abs(slp) < 1e-4: return "đi ngang"
                return "tăng" if slp > 0 else "giảm"
            return "(chưa xác định)"

        dir_vi = _norm_dir(dir_txt, slope)
        slope_note = ''
        if isinstance(slope,(int,float)) and slope is not None:
            try:
                slope_note = f" (slope {slope:.6f})" if abs(slope) >= 1e-6 else " (slope≈0)"
            except Exception:
                slope_note = ''
        if dir_vi == "(chưa xác định)":
            trendline_txt = f"{tf_chan}: Trendline (không có dữ liệu)"
        elif dir_vi == "đi ngang":
            trendline_txt = f"{tf_chan}: Trendline đi ngang{slope_note}".rstrip()
        else:
            trendline_txt = f"{tf_chan}: Trendline {dir_vi}{slope_note}".rstrip()

    # Formatting helpers
        def fmt(val, nd=4):
            return (f"{val:.{nd}f}" if isinstance(val, (int, float)) else "-")
        # Price precision by symbol (JPY pairs 3 decimals; default 4)
        def price_decimals(symbol: Optional[str]) -> int:
            """Return price decimals by symbol: XAU=2, JPY pairs=3, crypto=8, else 4."""
            try:
                s = (symbol or "").upper()
            except Exception:
                s = ""
            if s.startswith("XAU") or s == "GOLD" or s == "XAUUSD":
                return 2
            if s.endswith("JPY"):
                return 3
            crypto_tags = ("BTC", "ETH", "XRP", "SOL", "ADA", "DOGE", "BNB", "TRX", "LTC", "DOT", "AVAX", "USDT", "USDC")
            if any(tag in s for tag in crypto_tags):
                return 8
            return 4
        price_nd = price_decimals(sym)
        def pfmt(val):
            if not isinstance(val, (int, float)):
                return "-"
            v = float(val)
            s_upper = (sym or "").upper()
            crypto_tags = ("BTC", "ETH", "XRP", "SOL", "ADA", "DOGE", "BNB", "TRX", "LTC", "DOT", "AVAX", "USDT", "USDC")
            is_crypto = any(tag in s_upper for tag in crypto_tags)
            if is_crypto:
                if abs(v) >= 1000:
                    raw = f"{v:.2f}"
                elif abs(v) >= 1:
                    raw = f"{v:.4f}"
                else:
                    raw = f"{v:.6f}"
            else:
                # Non-crypto: use native precision but cap trailing zeros
                raw = f"{v:.{price_nd}f}"
            if '.' in raw:
                raw = raw.rstrip('0').rstrip('.')
            return raw
        def above_below(val):
            # Robust comparison handling None/NaN
            try:
                import math
                if val is None or price_trend is None:
                    return "-"
                if isinstance(val,(int,float)) and (val != val):  # NaN
                    return "-"
                if isinstance(price_trend,(int,float)) and (price_trend != price_trend):
                    return "-"
                if price_trend > val: return "trên"
                if price_trend < val: return "dưới"
                return "trong"
            except Exception:
                return "-"

        # Simple tag for support/oppose notation
        def support_tag(flag: Optional[bool]) -> str:
            if flag is True:
                return "– ủng hộ BUY"
            if flag is False:
                return "– ủng hộ SELL"
            return ""

        # Helper: last two closes from candles
        def last_two_closes(candles: Any) -> Tuple[Optional[float], Optional[float]]:
            try:
                if isinstance(candles, dict):
                    for key in ("rates", "candles", "data", "bars"):
                        seq = candles.get(key)
                        if isinstance(seq, list) and len(seq) >= 2:
                            a = seq[-2]; b = seq[-1]
                            pa = ffloat((a.get("close") or a.get("c"))) if isinstance(a, dict) else None
                            pb = ffloat((b.get("close") or b.get("c"))) if isinstance(b, dict) else None
                            return pa, pb
                elif isinstance(candles, list) and len(candles) >= 2:
                    a = candles[-2]; b = candles[-1]
                    pa = ffloat((a.get("close") or a.get("c"))) if isinstance(a, dict) else None
                    pb = ffloat((b.get("close") or b.get("c"))) if isinstance(b, dict) else None
                    return pa, pb
            except Exception:
                return None, None
            return None, None

        # Extract indicator values with fallback to channel TF and then all TFs
        def _fallback(call, primary, secondary):
            try:
                v = call(primary)
            except Exception:
                v = None
            if v is None or (isinstance(v, dict) and not v):
                try:
                    v2 = call(secondary)
                    if v2 is not None and (not isinstance(v2, dict) or bool(v2)):
                        return v2
                except Exception:
                    pass
                for tfp in tflist:
                    d = tf_data_cache.get(tfp)
                    indm = d.indicators if d else None
                    if indm is None or indm is primary or indm is secondary:
                        continue
                    try:
                        v3 = call(indm)
                        if v3 is not None and (not isinstance(v3, dict) or bool(v3)):
                            return v3
                    except Exception:
                        continue
            return v

        def _synonyms(key: str) -> List[str]:
            m = {
                # EMA variants (upper/lowercase + with/without underscore)
                "EMA_20": ["EMA_20", "EMA20", "ema20", "ema_20"],
                "EMA_50": ["EMA_50", "EMA50", "ema50", "ema_50"],
                "EMA_200": ["EMA_200", "EMA200", "ema200", "ema_200"],
                "EMA_100": ["EMA_100", "EMA100", "ema100", "ema_100"],
                # SMA variants
                "SMA_20": ["SMA_20", "SMA20", "sma20", "sma_20"],
                "SMA_50": ["SMA_50", "SMA50", "sma50", "sma_50"],
                # WMA variants
                "WMA_20": ["WMA_20", "WMA20", "wma20", "wma_20", "WMA20", "WMA_20"],
                # TEMA variants
                "TEMA_20": ["TEMA_20", "TEMA20", "tema20", "tema_20"],
                "TEMA_50": ["TEMA_50", "TEMA50", "tema50", "tema_50"],
                "TEMA_100": ["TEMA_100", "TEMA100", "tema100", "tema_100"],
                "TEMA_200": ["TEMA_200", "TEMA200", "tema200", "tema_200"],
                # Ichimoku exporters vary
                "Ichimoku_Tenkan": ["Ichimoku_Tenkan", "tenkan_9", "Tenkan", "tenkan", "ichimoku_tenkan"],
                "Ichimoku_Kijun": ["Ichimoku_Kijun", "kijun_26", "Kijun", "kijun", "ichimoku_kijun"],
                "Ichimoku_Senkou_A": ["Ichimoku_Senkou_A", "senkou_a", "SenkouA", "ichimoku_senkou_a"],
                "Ichimoku_Senkou_B": ["Ichimoku_Senkou_B", "senkou_b", "SenkouB", "ichimoku_senkou_b"],
                "Ichimoku_Chikou": ["Ichimoku_Chikou", "chikou", "Chikou", "ichimoku_chikou"],
                # Chaikin Money Flow variants
                "Chaikin": ["Chaikin", "Chaikin18", "CMF", "ChaikinMoneyFlow", "chaikin_money_flow"],
                # Envelope variants with decimal support
                "Envelope_Upper": ["Envelope_Upper", "Envelope_Upper_23_2", "Envelope_Upper_23_2.0"],
                "Envelope_Middle": ["Envelope_Middle", "Envelope_Middle_23_2", "Envelope_Middle_23_2.0"],
                "Envelope_Lower": ["Envelope_Lower", "Envelope_Lower_23_2", "Envelope_Lower_23_2.0"],
            }
            return m.get(key, [key])

        def _last_from(ind: Any, key: str):
            if not ind:
                return None
            for k in _synonyms(key):
                v = Extract._last_from_list(ind, k)
                if v is not None:
                    return v
            return None

        def _last_from_fallback(key: str):
            v = _last_from(ind_trend, key)
            if v is None:
                v = _last_from(ind_chan, key)
            if v is None:
                for tfp in tflist:
                    d = tf_data_cache.get(tfp)
                    indm = d.indicators if d else None
                    if indm is None:
                        continue
                    for k in _synonyms(key):
                        vv = Extract._last_from_list(indm, k)
                        if vv is not None:
                            return vv
            return v

        def _last_two_from(ind: Any, key: str) -> Tuple[Optional[float], Optional[float]]:
            if not ind:
                return (None, None)
            for k in _synonyms(key):
                p, n = Extract._last_two_from_list(ind, k)
                if p is not None and n is not None:
                    return p, n
            return (None, None)

        def _last_two_fallback(key: str):
            p1, n1 = _last_two_from(ind_trend, key)
            if (p1 is None or n1 is None):
                p2, n2 = _last_two_from(ind_chan, key)
                p1 = p1 if p1 is not None else p2
                n1 = n1 if n1 is not None else n2
            if (p1 is None or n1 is None):
                for tfp in tflist:
                    d = tf_data_cache.get(tfp)
                    indm = d.indicators if d else None
                    pp, nn = _last_two_from(indm, key)
                    if pp is not None and nn is not None:
                        return pp, nn
                    if p1 is None and pp is not None:
                        p1 = pp
                    if n1 is None and nn is not None:
                        n1 = nn
                    if p1 is not None and n1 is not None:
                        break
            return p1, n1

        # Simple helpers for on-the-fly indicator calculations
        def _sma(arr: List[float], n: int) -> Optional[float]:
            if not arr or len(arr) < n:
                return None
            return sum(arr[-n:]) / float(n)
        def _ema(arr: List[float], n: int) -> Optional[float]:
            if not arr or len(arr) < n:
                return None
            k = 2.0 / (n + 1.0)
            ema_val = sum(arr[:n]) / n
            for v in arr[n:]:
                ema_val = v * k + ema_val * (1 - k)
            return ema_val
        def _wma(arr: List[float], n: int) -> Optional[float]:
            if not arr or len(arr) < n:
                return None
            w = list(range(1, n + 1))
            seg = arr[-n:]
            return sum(a*b for a, b in zip(seg, w)) / float(sum(w))
        def _rsi(closes: List[float], n: int = 14) -> Optional[float]:
            """Return last RSI value using Wilder's smoothing."""
            if not closes or len(closes) <= n:
                return None
            gains: List[float] = []
            losses: List[float] = []
            for i in range(1, len(closes)):
                ch = closes[i] - closes[i-1]
                gains.append(max(0.0, ch))
                losses.append(max(0.0, -ch))
            # Seed averages
            avg_gain = sum(gains[:n]) / n
            avg_loss = sum(losses[:n]) / n
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / max(avg_loss, 1e-12)
            rsi_val = 100.0 - (100.0 / (1.0 + rs))
            # Smooth through remaining data
            for i in range(n, len(gains)):
                avg_gain = (avg_gain * (n - 1) + gains[i]) / n
                avg_loss = (avg_loss * (n - 1) + losses[i]) / n
                if avg_loss == 0:
                    rsi_val = 100.0
                else:
                    rs = avg_gain / max(avg_loss, 1e-12)
                    rsi_val = 100.0 - (100.0 / (1.0 + rs))
            return rsi_val
        def _rsi_series(closes: List[float], n: int = 14) -> List[float]:
            # Return RSI series for each close starting at index n
            if not closes or len(closes) <= n:
                return []
            rsis: List[float] = [None] * len(closes)
            gains = [0.0]
            losses = [0.0]
            for i in range(1, len(closes)):
                ch = closes[i] - closes[i-1]
                gains.append(max(ch, 0.0))
                losses.append(max(-ch, 0.0))
            avg_gain = sum(gains[1:n+1]) / n
            avg_loss = sum(losses[1:n+1]) / n
            rs = (avg_gain / max(avg_loss, 1e-12)) if avg_loss else float('inf')
            rsis[n] = 100.0 - (100.0 / (1.0 + rs))
            for i in range(n+1, len(gains)):
                avg_gain = (avg_gain * (n-1) + gains[i]) / n
                avg_loss = (avg_loss * (n-1) + losses[i]) / n
                rs = avg_gain / max(avg_loss, 1e-12)
                rsis[i] = 100.0 - (100.0 / (1.0 + rs))
            return rsis
        def _stochrsi(closes: List[float], rsi_period: int = 14, stoch_period: int = 14) -> Optional[float]:
            rsi_vals = _rsi_series(closes, rsi_period)
            if len(rsi_vals) < stoch_period:
                return None
            window = rsi_vals[-stoch_period:]
            lo = min(window); hi = max(window)
            if hi == lo:
                return 0.5
            # Normalize to 0..1
            return (rsi_vals[-1] - lo) / (hi - lo)
        def _stoch(highs: List[float], lows: List[float], closes: List[float], k: int = 14, d: int = 3) -> Tuple[Optional[float], Optional[float]]:
            if len(highs) < k or len(lows) < k or len(closes) < k:
                return None, None
            k_values: List[float] = []
            for i in range(k-1, len(closes)):
                lo = min(lows[i-k+1:i+1]); hi = max(highs[i-k+1:i+1])
                if hi == lo:
                    k_values.append(50.0)
                else:
                    k_values.append(100.0 * (closes[i] - lo) / (hi - lo))
            k_last = _sma(k_values, d)
            d_last = _sma(k_values[-d*2:], d) if len(k_values) >= d*2 else k_last
            return k_last, d_last
        def _atr(highs: List[float], lows: List[float], closes: List[float], n: int = 14) -> Optional[float]:
            if len(highs) < n+1 or len(lows) < n+1 or len(closes) < n+1:
                return None
            trs: List[float] = []
            for i in range(1, len(closes)):
                tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
                trs.append(tr)
            atr_val = sum(trs[:n]) / n
            for i in range(n, len(trs)):
                atr_val = (atr_val * (n - 1) + trs[i]) / n
            return atr_val
        def _bbands(closes: List[float], n: int = 20, sd: float = 2.0) -> Optional[Dict[str, float]]:
            if len(closes) < n:
                return None
            seg = closes[-n:]
            ma = sum(seg) / n
            var = sum((x - ma) ** 2 for x in seg) / n
            st = var ** 0.5
            return {"middle": ma, "upper": ma + sd * st, "lower": ma - sd * st}
        def _macd_hist(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[float]:
            if len(closes) < slow + signal:
                return None
            # Build MACD line progressively (approximation) then signal EMA
            macd_line_series: List[float] = []
            for i in range(slow, len(closes)+1):
                ema_f = _ema(closes[:i], fast)
                ema_s = _ema(closes[:i], slow)
                if ema_f is None or ema_s is None:
                    continue
                macd_line_series.append(ema_f - ema_s)
            if len(macd_line_series) < signal:
                return None
            signal_line = _ema(macd_line_series, signal)
            if signal_line is None:
                return None
            return macd_line_series[-1] - signal_line
        def _mfi(highs: List[float], lows: List[float], closes: List[float], volumes: List[float], n: int = 14) -> Optional[float]:
            if len(highs) < n+1 or len(lows) < n+1 or len(closes) < n+1 or len(volumes) < n+1:
                return None
            tp = [ (highs[i]+lows[i]+closes[i])/3.0 for i in range(len(closes)) ]
            rmf_pos = []
            rmf_neg = []
            for i in range(1, len(tp)):
                rmf = tp[i] * volumes[i]
                if tp[i] > tp[i-1]:
                    rmf_pos.append(rmf); rmf_neg.append(0.0)
                elif tp[i] < tp[i-1]:
                    rmf_pos.append(0.0); rmf_neg.append(rmf)
                else:
                    rmf_pos.append(0.0); rmf_neg.append(0.0)
            if len(rmf_pos) < n or len(rmf_neg) < n:
                return None
            pos_sum = sum(rmf_pos[-n:]); neg_sum = sum(rmf_neg[-n:])
            if neg_sum == 0:
                return 100.0
            mr = pos_sum / max(neg_sum,1e-12)
            return 100.0 - (100.0 / (1.0 + mr))
        def _obv(closes: List[float], volumes: List[float]) -> Optional[float]:
            if not closes or not volumes or len(closes) != len(volumes):
                return None
            obv = 0.0
            for i in range(1, len(closes)):
                if closes[i] > closes[i-1]:
                    obv += volumes[i]
                elif closes[i] < closes[i-1]:
                    obv -= volumes[i]
            return obv
        def _chaikin_osc(highs: List[float], lows: List[float], closes: List[float], volumes: List[float], fast: int = 3, slow: int = 10) -> Optional[float]:
            if not highs or not lows or not closes or not volumes:
                return None
            if not (len(highs) == len(lows) == len(closes) == len(volumes)):
                return None
            adl: List[float] = []
            for i in range(len(closes)):
                high, low, close, vol = highs[i], lows[i], closes[i], volumes[i]
                mfm = 0.0 if high == low else ((close - low) - (high - close)) / (high - low)
                adl.append(mfm * vol)
            def ema_list(arr: List[float], n: int) -> List[float]:
                if len(arr) < n:
                    return []
                k = 2.0/(n+1.0)
                out: List[float] = []
                ema_val = sum(arr[:n]) / n
                out.append(ema_val)
                for v in arr[n:]:
                    ema_val = v * k + ema_val * (1-k)
                    out.append(ema_val)
                return out
            efast = ema_list(adl, fast)
            eslow = ema_list(adl, slow)
            if not efast or not eslow or len(efast) != len(eslow):
                return None
            # last non-None pair
            for i in range(len(efast)-1, -1, -1):
                a = efast[i]; b = eslow[i]
                if a is not None and b is not None:
                    return a - b
            return None
        def _eom(highs: List[float], lows: List[float], closes: List[float], volumes: List[float], n: int = 14) -> Optional[float]:
            if not highs or not lows or not closes or not volumes:
                return None
            if not (len(highs) == len(lows) == len(closes) == len(volumes)):
                return None
            eom_vals: List[float] = []
            for i in range(1, len(closes)):
                box_ratio = (highs[i] - lows[i])
                vol = volumes[i] if volumes[i] != 0 else 1.0
                dist = ((highs[i] + lows[i]) / 2.0) - ((highs[i-1] + lows[i-1]) / 2.0)
                if box_ratio == 0:
                    eom_vals.append(0.0)
                else:
                    eom_vals.append((dist / box_ratio) / vol)
            if len(eom_vals) < n:
                return eom_vals[-1] if eom_vals else None
            return sum(eom_vals[-n:]) / n
        def _force_index(closes: List[float], volumes: List[float], n: int = 13) -> Optional[float]:
            if not closes or not volumes or len(closes) != len(volumes):
                return None
            fi_vals: List[float] = []
            for i in range(1, len(closes)):
                fi_vals.append((closes[i] - closes[i-1]) * volumes[i])
            if len(fi_vals) < n:
                return fi_vals[-1] if fi_vals else None
            # simple EMA smoothing
            return _ema(fi_vals, n)
        def _mass_index(highs: List[float], lows: List[float], ema_period: int = 9, sum_period: int = 25) -> Optional[float]:
            if len(highs) < ema_period*2 or len(lows) < ema_period*2:
                return None
            ranges = [h - l for h, l in zip(highs, lows)]
            ema1_list: List[float] = []
            k = 2.0/(ema_period+1.0)
            # first EMA
            ema_val = sum(ranges[:ema_period]) / ema_period
            ema1_list.extend([None]*(ema_period-1))
            ema1_list.append(ema_val)
            for v in ranges[ema_period:]:
                ema_val = v * k + ema_val * (1-k)
                ema1_list.append(ema_val)
            # second EMA of ema1
            ema2_list: List[float] = []
            seed = [x for x in ema1_list if x is not None]
            if len(seed) < ema_period:
                return None
            ema_val2 = sum(seed[:ema_period]) / ema_period
            idx = ema1_list.index(seed[0])
            ema2_list.extend([None]*(idx+ema_period-1))
            ema2_list.append(ema_val2)
            for i in range(idx+ema_period, len(ema1_list)):
                v = ema1_list[i]
                if v is None:
                    ema2_list.append(None)
                else:
                    ema_val2 = v * k + ema_val2 * (1-k)
                    ema2_list.append(ema_val2)
            ratio: List[float] = []
            for a,b in zip(ema1_list, ema2_list):
                if a is None or b is None or b == 0:
                    ratio.append(None)
                else:
                    ratio.append(a/b)
            vals = [x for x in ratio if x is not None]
            if len(vals) < sum_period:
                return vals[-1] if vals else None
            return sum(vals[-sum_period:]) / sum_period
            
        # Collect indicator values across TFs with fallbacks
        
        # First detect parameters from indicator data
        def _detect_params_from_last_row() -> Dict[str, Optional[Union[int, float]]]:
            """Extract indicator parameters from last row of available indicator data."""
            print(f"DEBUG: ind_trend type: {type(ind_trend)}, len: {len(ind_trend) if isinstance(ind_trend, list) else 'N/A'}")
            print(f"DEBUG: ind_chan type: {type(ind_chan)}, len: {len(ind_chan) if isinstance(ind_chan, list) else 'N/A'}")
            if isinstance(ind_trend, list) and ind_trend and isinstance(ind_trend[-1], dict):
                print(f"DEBUG: ind_trend last row keys: {list(ind_trend[-1].keys())[:10]}...")  # First 10 keys
            params: Dict[str, Optional[Union[int, float]]] = {
                'rsi': None, 'adx': None, 'atr': None, 'mfi': None,
                'cci': None, 'willr': None, 'roc': None, 'stoch_period': None, 'stoch_smooth': None,
                'stochrsi': None, 'bb_win': None, 'bb_dev': None,
                'donch_win': None, 'kelt_win': None, 'env_win': None, 'env_dev': None,
                'trix': None, 'dpo': None, 'chaikin': None, 'eom': None, 'force': None,
                'mass_fast': None, 'mass_slow': None, 'vi_period': None,
                'macd_fast': None, 'macd_slow': None, 'macd_sig': None,
                'ichi_tenkan': None, 'ichi_kijun': None, 'ichi_senkou': None
            }
            def _scan_keys(keys: List[str]):
                print(f"DEBUG: _scan_keys processing {len(keys)} keys")
                bb_keys = [k for k in keys if 'BB' in k]
                if bb_keys:
                    bb_positions = [keys.index(k) for k in bb_keys]
                    print(f"DEBUG: BB keys at positions: {bb_positions}")
                    print(f"DEBUG: BB keys in this scan: {bb_keys}")
                count = 0
                for key in keys:
                    count += 1
                    if count <= 25:  # Print first 25 positions
                        print(f"DEBUG: Position {count}: '{key}' (type: {type(key)})")
                    if not isinstance(key, str):
                        print(f"DEBUG: Skipping non-string key at position {count}: {type(key)}")
                        continue
                    
                    ksu = key.upper()
                    if 'BB' in key:
                        print(f"DEBUG: Processing BB key #{count}: '{key}'")
                        # Test the regex manually
                        import re
                        m = re.fullmatch(r"BB_Upper_(\d+)_([0-9]+(?:\.[0-9]+)?)", key)
                        if m:
                            print(f"DEBUG: REGEX MATCHED! groups: {m.groups()}")
                        else:
                            print(f"DEBUG: REGEX DID NOT MATCH")
                    m = re.fullmatch(r"RSI[_-]?(\d+)$", ksu)
                    if m: params['rsi'] = int(m.group(1)); continue
                    m = re.fullmatch(r"ADX[_-]?(\d+)$", ksu)
                    if m: params['adx'] = int(m.group(1)); continue
                    m = re.fullmatch(r"ATR[_-]?(\d+)$", ksu)
                    if m: params['atr'] = int(m.group(1)); continue
                    m = re.fullmatch(r"MFI[_-]?(\d+)$", ksu)
                    if m: params['mfi'] = int(m.group(1)); continue
                    m = re.fullmatch(r"CCI[_-]?(\d+)$", ksu)
                    if m: params['cci'] = int(m.group(1)); continue
                    m = re.fullmatch(r"WILLIAMSR[_-]?(\d+)$", ksu)
                    if m: params['willr'] = int(m.group(1)); continue
                    m = re.fullmatch(r"ROC[_-]?(\d+)$", ksu)
                    if m: params['roc'] = int(m.group(1)); continue
                    m = re.fullmatch(r"STOCHRSI[_-]?(\d+)$", ksu)
                    if m: params['stochrsi'] = int(m.group(1)); continue
                    m = re.fullmatch(r"STOCHK_(\d+)_(\d+)", ksu)
                    if m: params['stoch_period'] = int(m.group(1)); params['stoch_smooth'] = int(m.group(2)); continue
                    m = re.fullmatch(r"STOCHD_(\d+)_(\d+)", ksu)
                    if m and params['stoch_period'] is None:
                        params['stoch_period'] = int(m.group(1)); params['stoch_smooth'] = int(m.group(2)); continue
                    # Bollinger Bands - use original case key
                    m = re.fullmatch(r"BB_Upper_(\d+)_([0-9]+(?:\.[0-9]+)?)", key)
                    if m: 
                        print(f"DEBUG: BB_Upper matched: {key} -> bb_win={m.group(1)}, bb_dev={m.group(2)}")
                        params['bb_win'] = int(m.group(1)); params['bb_dev'] = float(m.group(2)); continue
                    m = re.fullmatch(r"BB_Middle_(\d+)_([0-9]+(?:\.[0-9]+)?)", key)
                    if m and params['bb_win'] is None:
                        print(f"DEBUG: BB_Middle matched: {key} -> bb_win={m.group(1)}, bb_dev={m.group(2)}")
                        params['bb_win'] = int(m.group(1)); params['bb_dev'] = float(m.group(2)); continue
                    # Envelopes
                    m = re.fullmatch(r"ENVELOPE_(?:UPPER|MIDDLE|LOWER)_(\d+)_([0-9]+(?:\.[0-9]+)?)", ksu)
                    if m:
                        params['env_win'] = int(m.group(1)); params['env_dev'] = float(m.group(2)); continue
                    m = re.fullmatch(r"DONCHIAN_(?:UPPER|MIDDLE|LOWER)_(\d+)", ksu)
                    if m: params['donch_win'] = int(m.group(1)); continue
                    m = re.fullmatch(r"KELTNER_(?:UPPER|MIDDLE|LOWER)_(\d+)", ksu)
                    if m: params['kelt_win'] = int(m.group(1)); continue
                    m = re.fullmatch(r"TRIX[_-]?(\d+)$", ksu)
                    if m: params['trix'] = int(m.group(1)); continue
                    m = re.fullmatch(r"DPO[_-]?(\d+)$", ksu)
                    if m: params['dpo'] = int(m.group(1)); continue
                    m = re.fullmatch(r"CHAIKIN[_-]?(\d+)$", ksu)
                    if m: params['chaikin'] = int(m.group(1)); continue
                    m = re.fullmatch(r"CMF[_-]?(\d+)$", ksu)
                    if m: params['chaikin'] = int(m.group(1)); continue
                    m = re.fullmatch(r"EOM[_-]?(\d+)$", ksu)
                    if m: params['eom'] = int(m.group(1)); continue
                    m = re.fullmatch(r"FORCEINDEX[_-]?(\d+)$", ksu)
                    if m: params['force'] = int(m.group(1)); continue
                    m = re.fullmatch(r"(?i:MASSINDEX)_(\d+)_(\d+)", ksu)
                    if m: params['mass_fast'] = int(m.group(1)); params['mass_slow'] = int(m.group(2)); continue
                    m = re.fullmatch(r"VI[\+\-]_(\d+)", ksu)
                    if m: params['vi_period'] = int(m.group(1)); continue
                    m = re.fullmatch(r"MACD_(\d+)_(\d+)_(\d+)", ksu)
                    if m: params['macd_fast'] = int(m.group(1)); params['macd_slow'] = int(m.group(2)); params['macd_sig'] = int(m.group(3)); continue
                    m = re.fullmatch(r"MACD_HIST_(\d+)_(\d+)_(\d+)", ksu)
                    if m and params['macd_fast'] is None:
                        params['macd_fast'] = int(m.group(1)); params['macd_slow'] = int(m.group(2)); params['macd_sig'] = int(m.group(3)); continue
                    # Ichimoku: try to detect custom periods
                    m = re.fullmatch(r"(?:ICHIMOKU_)?TENKAN_(\d+)$", ksu)
                    if m: params['ichi_tenkan'] = int(m.group(1)); continue
                    m = re.fullmatch(r"(?:ICHIMOKU_)?KIJUN_(\d+)$", ksu)
                    if m: params['ichi_kijun'] = int(m.group(1)); continue
                    # Prefer SENKOU_B if available; else A; else generic
                    m = re.fullmatch(r"(?:ICHIMOKU_)?SENKOU_B_(\d+)$", ksu)
                    if m: params['ichi_senkou'] = int(m.group(1)); continue
                    m = re.fullmatch(r"(?:ICHIMOKU_)?SENKOU_A_(\d+)$", ksu)
                    if m and params['ichi_senkou'] is None:
                        params['ichi_senkou'] = int(m.group(1)); continue
                    m = re.fullmatch(r"(?:ICHIMOKU_)?SENKOU_(\d+)$", ksu)
                    if m and params['ichi_senkou'] is None:
                        params['ichi_senkou'] = int(m.group(1)); continue
                    m = re.fullmatch(r"EOM[_-]?(\d+)$", ksu)
                    if m: params['eom'] = int(m.group(1)); continue
                    m = re.fullmatch(r"FORCEINDEX[_-]?(\d+)$", ksu)
                    if m: params['force'] = int(m.group(1)); continue
                    m = re.fullmatch(r"(?i:MASSINDEX)_(\d+)_(\d+)", ksu)
                    if m: params['mass_fast'] = int(m.group(1)); params['mass_slow'] = int(m.group(2)); continue
                    m = re.fullmatch(r"VI[\+\-]_(\d+)", ksu)
                    if m: params['vi_period'] = int(m.group(1)); continue
                    m = re.fullmatch(r"MACD_(\d+)_(\d+)_(\d+)", ksu)
                    if m: params['macd_fast'] = int(m.group(1)); params['macd_slow'] = int(m.group(2)); params['macd_sig'] = int(m.group(3)); continue
                    m = re.fullmatch(r"MACD_HIST_(\d+)_(\d+)_(\d+)", ksu)
                    if m and params['macd_fast'] is None:
                        params['macd_fast'] = int(m.group(1)); params['macd_slow'] = int(m.group(2)); params['macd_sig'] = int(m.group(3)); continue
                    # Ichimoku: try to detect custom periods
                    m = re.fullmatch(r"(?:ICHIMOKU_)?TENKAN_(\d+)$", ksu)
                    if m: params['ichi_tenkan'] = int(m.group(1)); continue
                    m = re.fullmatch(r"(?:ICHIMOKU_)?KIJUN_(\d+)$", ksu)
                    if m: params['ichi_kijun'] = int(m.group(1)); continue
                    # Prefer SENKOU_B if available; else A; else generic
                    m = re.fullmatch(r"(?:ICHIMOKU_)?SENKOU_B_(\d+)$", ksu)
                    if m: params['ichi_senkou'] = int(m.group(1)); continue
                    m = re.fullmatch(r"(?:ICHIMOKU_)?SENKOU_A_(\d+)$", ksu)
                    if m and params['ichi_senkou'] is None:
                        params['ichi_senkou'] = int(m.group(1)); continue
                    m = re.fullmatch(r"(?:ICHIMOKU_)?SENKOU_(\d+)$", ksu)
                    if m and params['ichi_senkou'] is None:
                        params['ichi_senkou'] = int(m.group(1)); continue
            try:
                # Pass 1: last row fast path on trend TF
                last = ind_trend[-1] if isinstance(ind_trend, list) and ind_trend else {}
                keys = list(last.keys()) if isinstance(last, dict) else []
                print(f"DEBUG: Pass 1 - processing {len(keys)} keys from ind_trend")
                if keys:
                    print(f"DEBUG: Sample keys: {keys[:10]}")
                _scan_keys(keys)
                # Pass 2: scan all rows backwards on trend TF
                if isinstance(ind_trend, list) and ind_trend:
                    for row in reversed(ind_trend):
                        if not any(params[k] is None for k in params):
                            break
                        if isinstance(row, dict):
                            _scan_keys(row.keys())
                # Pass 3: scan across all loaded timeframes for parameter hints
                try:
                    if isinstance(tf_data_cache, dict) and tf_data_cache:
                        for _tf, _d in tf_data_cache.items():
                            ind_any = getattr(_d, 'indicators', None)
                            if isinstance(ind_any, list):
                                # last row of this TF
                                if ind_any:
                                    last2 = ind_any[-1] if isinstance(ind_any[-1], dict) else None
                                    if isinstance(last2, dict):
                                        _scan_keys(last2.keys())
                                # backwards scan
                                for row in reversed(ind_any):
                                    if not any(params[k] is None for k in params):
                                        break
                                    if isinstance(row, dict):
                                        _scan_keys(row.keys())
                except Exception:
                    pass
            except Exception:
                return params
            return params

        _params = _detect_params_from_last_row()
        # Override BB parameters with detected values (with safe checks)
        try:
            if 'bb_win_detected' in globals() and bb_win_detected is not None:
                _params['bb_win'] = bb_win_detected
        except NameError:
            pass
        try:
            if 'bb_dev_detected' in globals() and bb_dev_detected is not None:
                _params['bb_dev'] = bb_dev_detected
        except NameError:
            pass
        # Override Keltner parameters with detected values
        try:
            if 'kelt_win_detected' in globals() and kelt_win_detected is not None:
                _params['kelt_win'] = kelt_win_detected
        except NameError:
            pass
        # Override Envelope parameters with detected values  
        try:
            if 'env_win_detected' in globals() and env_win_detected is not None:
                _params['env_win'] = env_win_detected
        except NameError:
            pass
        try:
            if 'env_dev_detected' in globals() and env_dev_detected is not None:
                _params['env_dev'] = env_dev_detected
        except NameError:
            pass
        # Override Donchian parameters with detected values
        try:
            if 'donch_win_detected' in globals() and donch_win_detected is not None:
                _params['donch_win'] = donch_win_detected
        except NameError:
            pass
        # Override EOM parameters with detected values
        try:
            if 'eom_win_detected' in globals() and eom_win_detected is not None:
                _params['eom'] = eom_win_detected
        except NameError:
            pass
        # Override DPO parameters with detected values
        try:
            if 'dpo_win_detected' in globals() and dpo_win_detected is not None:
                _params['dpo'] = dpo_win_detected
        except NameError:
            pass
        # Override Chaikin parameters with detected values
        try:
            if 'chaikin_win_detected' in globals() and chaikin_win_detected is not None:
                _params['chaikin'] = chaikin_win_detected
        except NameError:
            pass
        # Override TRIX parameters with detected values
        try:
            if 'trix_win_detected' in globals() and trix_win_detected is not None:
                _params['trix'] = trix_win_detected
        except NameError:
            pass
        # Override Mass Index parameters with detected values
        try:
            if 'mass_fast_detected' in globals() and mass_fast_detected is not None:
                _params['mass_fast'] = mass_fast_detected
        except NameError:
            pass
        try:
            if 'mass_slow_detected' in globals() and mass_slow_detected is not None:
                _params['mass_slow'] = mass_slow_detected
        except NameError:
            pass
        # Override Ichimoku parameters with detected values
        try:
            if 'ichi_tenkan_detected' in globals() and ichi_tenkan_detected is not None:
                _params['ichi_tenkan'] = ichi_tenkan_detected
        except NameError:
            pass
        try:
            if 'ichi_kijun_detected' in globals() and ichi_kijun_detected is not None:
                _params['ichi_kijun'] = ichi_kijun_detected
        except NameError:
            pass
        try:
            if 'ichi_senkou_detected' in globals() and ichi_senkou_detected is not None:
                _params['ichi_senkou'] = ichi_senkou_detected
        except NameError:
            pass
        print(f"DEBUG: Detected params: {_params}")
        
        rsi = _fallback(Extract.rsi, ind_trend, ind_chan)
        macd_h = _fallback(Extract.macd_hist, ind_trend, ind_chan)
        adx = _fallback(Extract.adx, ind_trend, ind_chan)
        stoch = _fallback(Extract.stochastic, ind_trend, ind_chan) or {}
        stoch_k = stoch.get("k"); stoch_d = stoch.get("d")
        stochrsi = _fallback(Extract.stochrsi, ind_trend, ind_chan)
        # Bollinger Bands with dynamic parameters
        bb = {}
        try:
            # Try to extract with detected parameters
            bb_up_template = "BB_Upper_{bb_win}_{bb_dev}"
            bb_mid_template = "BB_Middle_{bb_win}_{bb_dev}"
            bb_low_template = "BB_Lower_{bb_win}_{bb_dev}"
            print(f"DEBUG BB: Looking for bb_win={_params.get('bb_win')}, bb_dev={_params.get('bb_dev')}")
            bb_up = Extract._last_key_with_params(ind_trend, bb_up_template, _params) or Extract._last_key_with_params(ind_chan, bb_up_template, _params)
            bb_mid = Extract._last_key_with_params(ind_trend, bb_mid_template, _params) or Extract._last_key_with_params(ind_chan, bb_mid_template, _params)
            bb_low = Extract._last_key_with_params(ind_trend, bb_low_template, _params) or Extract._last_key_with_params(ind_chan, bb_low_template, _params)
            print(f"DEBUG BB: bb_up={bb_up}, bb_mid={bb_mid}, bb_low={bb_low}")
            if None not in (bb_up, bb_mid, bb_low):
                bb = {"upper": bb_up, "middle": bb_mid, "lower": bb_low}
                print(f"DEBUG BB: Success! bb={bb}")
            else:
                # Fallback to old method
                bb = _fallback(Extract.bbands, ind_trend, ind_chan) or {}
                print(f"DEBUG BB: Fallback bb={bb}")
        except Exception as e:
            bb = _fallback(Extract.bbands, ind_trend, ind_chan) or {}
            print(f"DEBUG BB: Exception {e}, fallback bb={bb}")
        atr = _fallback(Extract.atr_latest, ind_trend, ind_chan)
        mfi = _fallback(Extract.mfi, ind_trend, ind_chan)
        psar_dir = _fallback(Extract.psar_dir, ind_trend, ind_chan)
        ema20 = _last_from_fallback("EMA_20")
        ema50 = _last_from_fallback("EMA_50")
        ema200 = _last_from_fallback("EMA_200")
        tema20 = _last_from_fallback("TEMA_20")
        tema50 = _last_from_fallback("TEMA_50")
        tema100 = _last_from_fallback("TEMA_100")
        tema200 = _last_from_fallback("TEMA_200")
        sma20 = _last_from_fallback("SMA_20")
        sma50 = _last_from_fallback("SMA_50")
        wma20 = _last_from_fallback("WMA_20")
        ich_tenkan = _last_from_fallback("Ichimoku_Tenkan")
        ich_kijun = _last_from_fallback("Ichimoku_Kijun")
        ich_senkou_a = _last_from_fallback("Ichimoku_Senkou_A")
        ich_senkou_b = _last_from_fallback("Ichimoku_Senkou_B")
        ich_chikou = _last_from_fallback("Ichimoku_Chikou")
        chaikin_mf = _last_from_fallback("Chaikin")
        env_upper = _last_from_fallback("Envelope_Upper")
        env_middle = _last_from_fallback("Envelope_Middle")
        env_lower = _last_from_fallback("Envelope_Lower")
        # Keltner Channels with dynamic parameters
        kelt = {}
        try:
            # Try to extract with detected parameters
            kelt_up_template = "Keltner_Upper_{kelt_win}"
            kelt_mid_template = "Keltner_Middle_{kelt_win}"
            kelt_low_template = "Keltner_Lower_{kelt_win}"
            kelt_up = Extract._last_key_with_params(ind_trend, kelt_up_template, _params) or Extract._last_key_with_params(ind_chan, kelt_up_template, _params)
            kelt_mid = Extract._last_key_with_params(ind_trend, kelt_mid_template, _params) or Extract._last_key_with_params(ind_chan, kelt_mid_template, _params)
            kelt_low = Extract._last_key_with_params(ind_trend, kelt_low_template, _params) or Extract._last_key_with_params(ind_chan, kelt_low_template, _params)
            if None not in (kelt_up, kelt_low):  # Don't require middle since it's often null
                kelt = {"upper": kelt_up, "lower": kelt_low}
                if kelt_mid is not None:
                    kelt["middle"] = kelt_mid
            else:
                # Fallback to old method
                kelt = _fallback(Extract.keltner, ind_trend, ind_chan) or {}
        except Exception:
            kelt = _fallback(Extract.keltner, ind_trend, ind_chan) or {}
        cci = _fallback(Extract.cci, ind_trend, ind_chan)
        willr = _fallback(Extract.williams_r, ind_trend, ind_chan)
        roc = _fallback(Extract.roc, ind_trend, ind_chan)
        obv = _last_from_fallback("OBV")
        
        # Chaikin Money Flow using detected parameters and direct field lookup
        if _params.get('chaikin'):
            chaikin_template = f"Chaikin{_params['chaikin']}"
            chaikin = Extract._last_key_with_params(ind_trend, chaikin_template, _params) or Extract._last_key_with_params(ind_chan, chaikin_template, _params)
        else:
            chaikin = chaikin_mf or _fallback(Extract.chaikin, ind_trend, ind_chan)
        
        # EOM using detected parameters
        if _params.get('eom'):
            eom_template = f"EOM{_params['eom']}"
            eom = Extract._last_key_with_params(ind_trend, eom_template, _params) or Extract._last_key_with_params(ind_chan, eom_template, _params)
        else:
            eom = _fallback(Extract.eom, ind_trend, ind_chan)
        force = _fallback(Extract.force_index, ind_trend, ind_chan)
        
        # TRIX using detected parameters
        if _params.get('trix'):
            trix_template = f"TRIX{_params['trix']}"
            trix = Extract._last_key_with_params(ind_trend, trix_template, _params) or Extract._last_key_with_params(ind_chan, trix_template, _params)
        else:
            trix = _fallback(Extract.trix, ind_trend, ind_chan)
            
        # DPO using detected parameters
        if _params.get('dpo'):
            dpo_template = f"DPO{_params['dpo']}"
            dpo = Extract._last_key_with_params(ind_trend, dpo_template, _params) or Extract._last_key_with_params(ind_chan, dpo_template, _params)
        else:
            dpo = _fallback(Extract.dpo, ind_trend, ind_chan)
            
        massi = _fallback(Extract.mass_index, ind_trend, ind_chan)
        vortex = _fallback(Extract.vortex, ind_trend, ind_chan) or {}
        kst = _fallback(Extract.kst, ind_trend, ind_chan)
        ult = _fallback(Extract.ultimate_osc, ind_trend, ind_chan)
        # Envelopes with dynamic parameters  
        env = {}
        try:
            # Use direct field lookup with fallback to catch both decimal formats
            env_upper = _last_from_fallback("Envelope_Upper")
            env_middle = _last_from_fallback("Envelope_Middle") 
            env_lower = _last_from_fallback("Envelope_Lower")
            if None not in (env_upper, env_lower):  # Don't require middle
                env = {"upper": env_upper, "lower": env_lower}
                if env_middle is not None:
                    env["middle"] = env_middle
            else:
                # Final fallback to old method
                env = _fallback(Extract.envelopes, ind_trend, ind_chan) or {}
        except Exception:
            env = _fallback(Extract.envelopes, ind_trend, ind_chan) or {}

        # --------------------------------------------------
        # Fallback compute for moving averages if missing/NaN
        # --------------------------------------------------
        def _is_nan(x):
            return isinstance(x,(int,float)) and x != x
        incomplete_ma_periods: Set[str] = set()
        def _compute_sma(seq, period):
            if not seq:
                return None
            if len(seq) < period:
                incomplete_ma_periods.add(f"SMA{period}")
                return sum(seq) / len(seq)
            return sum(seq[-period:]) / period
        def _compute_ema(seq, period):
            if not seq:
                return None
            k = 2.0/(period+1.0)
            if len(seq) < period:
                # seed with simple mean of available and iterate
                incomplete_ma_periods.add(f"EMA{period}")
                ema = sum(seq) / len(seq)
                return ema  # no smoothing window yet
            ema = sum(seq[:period]) / period
            for v in seq[period:]:
                ema = v * k + ema * (1-k)
            return ema
        def _compute_wma(seq, period):
            if not seq:
                return None
            if len(seq) < period:
                incomplete_ma_periods.add(f"WMA{period}")
                # approximate with SMA of available
                return sum(seq)/len(seq)
            w = list(range(1,period+1))
            seg = seq[-period:]
            return sum(a*b for a,b in zip(seg,w)) / sum(w)

        def _compute_tema(seq, period):
            """Compute last-bar TEMA value using standard definition.
            TEMA = 3*EMA1 - 3*EMA2 + EMA3, where EMA2 = EMA(EMA1), EMA3 = EMA(EMA2).
            Best-effort for short sequences.
            """
            try:
                if not seq or period <= 0:
                    return None
                k = 2.0/(period+1.0)
                if len(seq) < period:
                    incomplete_ma_periods.add(f"TEMA{period}")
                    # Seed with EMA over available
                    ema = sum(seq)/len(seq)
                    return ema
                # Build EMA1 series
                ema1_series: List[float] = []
                ema1 = sum(seq[:period]) / period
                ema1_series.append(ema1)
                for v in seq[period:]:
                    ema1 = v * k + ema1 * (1-k)
                    ema1_series.append(ema1)
                # EMA2 series (EMA on EMA1 series)
                if len(ema1_series) < period:
                    return ema1_series[-1]
                ema2_series: List[float] = []
                ema2 = sum(ema1_series[:period]) / period
                ema2_series.append(ema2)
                for v in ema1_series[period:]:
                    ema2 = v * k + ema2 * (1-k)
                    ema2_series.append(ema2)
                # EMA3 over EMA2 series
                if len(ema2_series) < period:
                    return 3*ema1_series[-1] - 3*ema2_series[-1] + ema2_series[-1]
                ema3 = sum(ema2_series[:period]) / period
                for v in ema2_series[period:]:
                    ema3 = v * k + ema3 * (1-k)
                return 3*ema1_series[-1] - 3*ema2_series[-1] + ema3
            except Exception:
                return None

        if closes:
            # Ensure ema100 symbol exists
            try:
                ema100  # noqa: F821
            except Exception:
                ema100 = None  # type: ignore
            if ema20 is None or _is_nan(ema20):
                ema20 = _compute_ema(closes,20)
            if ema50 is None or _is_nan(ema50):
                ema50 = _compute_ema(closes,50)
            if ema100 is None or _is_nan(ema100):
                fallback_ema100 = _last_from_fallback("EMA100") or _last_from_fallback("EMA_100")
                ema100 = fallback_ema100 if fallback_ema100 is not None else _compute_ema(closes,100)
            if ema200 is None or _is_nan(ema200):
                if len(closes) >= 200:
                    ema200 = _compute_ema(closes,200)
            if sma20 is None or _is_nan(sma20):
                sma20 = _compute_sma(closes,20)
            if wma20 is None or _is_nan(wma20):
                wma20 = _compute_wma(closes,20)
        # Extra pull from raw indicators list (last bar) if still missing
        try:
            if (ema20 is None or _is_nan(ema20) or ema50 is None or _is_nan(ema50) or ema100 is None or _is_nan(ema100) or ema200 is None or _is_nan(ema200) or sma20 is None or _is_nan(sma20) or wma20 is None or _is_nan(wma20)) and isinstance(ind_trend, list) and ind_trend:
                last_ind = ind_trend[-1]
                def pick(keys):
                    for k in keys:
                        if k in last_ind and isinstance(last_ind[k], (int,float)) and last_ind[k]==last_ind[k]:
                            return last_ind[k]
                    return None
                if ema20 is None or _is_nan(ema20): ema20 = pick(["EMA20","EMA_20","ema20","ema_20"]) or ema20
                if ema50 is None or _is_nan(ema50): ema50 = pick(["EMA50","EMA_50","ema50","ema_50"]) or ema50
                if 'ema100' in locals():
                    if ema100 is None or _is_nan(ema100): ema100 = pick(["EMA100","EMA_100","ema100","ema_100"]) or ema100
                else:
                    ema100 = pick(["EMA100","EMA_100","ema100","ema_100"])  # define if exists
                if ema200 is None or _is_nan(ema200): ema200 = pick(["EMA200","EMA_200","ema200","ema_200"]) or ema200
                if sma20 is None or _is_nan(sma20): sma20 = pick(["SMA20","SMA_20","sma20","sma_20"]) or sma20
                if wma20 is None or _is_nan(wma20): wma20 = pick(["WMA20","WMA_20","wma20","wma_20"]) or wma20
                # TEMA quick picks if exporter provided
                try:
                    if 'tema20' in locals():
                        if tema20 is None or _is_nan(tema20): tema20 = pick(["TEMA20","TEMA_20","tema20","tema_20"]) or tema20
                    else:
                        tema20 = pick(["TEMA20","TEMA_20","tema20","tema_20"])  # type: ignore
                    if 'tema50' in locals():
                        if tema50 is None or _is_nan(tema50): tema50 = pick(["TEMA50","TEMA_50","tema50","tema_50"]) or tema50
                    else:
                        tema50 = pick(["TEMA50","TEMA_50","tema50","tema_50"])  # type: ignore
                    if 'tema100' in locals():
                        if tema100 is None or _is_nan(tema100): tema100 = pick(["TEMA100","TEMA_100","tema100","tema_100"]) or tema100
                    else:
                        tema100 = pick(["TEMA100","TEMA_100","tema100","tema_100"])  # type: ignore
                    if 'tema200' in locals():
                        if tema200 is None or _is_nan(tema200): tema200 = pick(["TEMA200","TEMA_200","tema200","tema_200"]) or tema200
                    else:
                        tema200 = pick(["TEMA200","TEMA_200","tema200","tema_200"])  # type: ignore
                except Exception:
                    pass
        except Exception:
            pass

        # Derive minimal indicators from candles if still missing (disabled in strict mode)
        # Pre-initialize series to avoid UnboundLocalError in later blocks when any exception happens above
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []
        volumes: List[float] = []
        vol_available = False
        try:
            # prefer trend candles, else channel candles
            src = data_trend.candles or data_chan.candles
            seq = None
            if isinstance(src, dict):
                for key in ("rates", "candles", "data", "bars"):
                    if isinstance(src.get(key), list):
                        seq = src.get(key)
                        break
            elif isinstance(src, list):
                seq = src
            if isinstance(seq, list):
                for b in seq:
                    if isinstance(b, dict):
                        h = b.get("high") or b.get("h")
                        l = b.get("low") or b.get("l")
                        c = b.get("close") or b.get("c")
                        # Volume selection: prefer any non-zero real volume or classic volume; else use tick volume
                        v = None
                        # 1) Generic volume fields if positive
                        if b.get("volume") is not None and ffloat(b.get("volume")) > 0:
                            v = b.get("volume")
                        elif b.get("v") is not None and ffloat(b.get("v")) > 0:
                            v = b.get("v")
                        # 2) Real volume only if positive
                        elif b.get("real_volume") is not None and ffloat(b.get("real_volume")) > 0:
                            v = b.get("real_volume")
                        elif b.get("realVolume") is not None and ffloat(b.get("realVolume")) > 0:
                            v = b.get("realVolume")
                        # 3) Tick volume variants (often available on FX)
                        elif b.get("tick_volume") is not None and ffloat(b.get("tick_volume")) > 0:
                            v = b.get("tick_volume")
                        elif b.get("tickVolume") is not None and ffloat(b.get("tickVolume")) > 0:
                            v = b.get("tickVolume")
                        elif b.get("tickVol") is not None and ffloat(b.get("tickVol")) > 0:
                            v = b.get("tickVol")
                        # 4) Fallback alias
                        elif b.get("vol") is not None and ffloat(b.get("vol")) > 0:
                            v = b.get("vol")
                        if h is not None and l is not None and c is not None:
                            highs.append(ffloat(h)); lows.append(ffloat(l)); closes.append(ffloat(c))
                            # Always append a volume value to keep arrays aligned (use tiny synthetic if missing)
                            if v is None:
                                v = 0.0001
                            vv = ffloat(v)
                            volumes.append(vv)
                            if vv > 0.00005:
                                vol_available = True
            # If no real positive volume detected but we have synthetic tiny values only, treat as available for calculation
            if not vol_available and volumes:
                vol_available = True
            if closes and not CFG.STRICT_IND_ONLY:
                if rsi is None:
                    rsi = _rsi(closes)
                if macd_h is None:
                    macd_h = _macd_hist(closes)
                if (not bb):
                    bb_calc = _bbands(closes)
                    bb = bb_calc or {}
                if ema20 is None:
                    ema20 = _ema(closes, 20)
                if ema50 is None:
                    ema50 = _ema(closes, 50)
                if ema200 is None:
                    ema200 = _ema(closes, 200)
                if sma20 is None:
                    sma20 = _sma(closes, 20)
                if sma50 is None:
                    sma50 = _sma(closes, 50)
                if wma20 is None:
                    wma20 = _wma(closes, 20)
                if (stoch_k is None or stoch_d is None) and highs and lows:
                    kk, dd = _stoch(highs, lows, closes)
                    if stoch_k is None:
                        stoch_k = kk
                    if stoch_d is None:
                        stoch_d = dd
                if stochrsi is None:
                    stochrsi = _stochrsi(closes)
                if atr is None and highs and lows:
                    atr = _atr(highs, lows, closes)
                # Additional on-the-fly indicators
                # Minimal PSAR direction from candles (fallback)
                def _psar_dir(highs: List[float], lows: List[float], step: float = 0.02, step_max: float = 0.2) -> Optional[int]:
                    try:
                        n = len(highs)
                        if n < 5 or len(lows) < n:
                            return None
                        # Initialize trend by first two bars movement
                        init_up = (highs[1] + lows[1]) >= (highs[0] + lows[0])
                        trend_up = init_up
                        sar = lows[0] if trend_up else highs[0]
                        ep = highs[0] if trend_up else lows[0]
                        af = step
                        for i in range(1, n):
                            sar = sar + af * (ep - sar)
                            if trend_up:
                                # SAR cannot be above prior lows
                                prev_low = lows[i-1]
                                prev2_low = lows[i-2] if i >= 2 else prev_low
                                sar = min(sar, prev_low, prev2_low)
                                # Reversal?
                                if lows[i] < sar:
                                    trend_up = False
                                    sar = ep
                                    ep = lows[i]
                                    af = step
                                else:
                                    if highs[i] > ep:
                                        ep = highs[i]
                                        af = min(af + step, step_max)
                            else:
                                # SAR cannot be below prior highs
                                prev_high = highs[i-1]
                                prev2_high = highs[i-2] if i >= 2 else prev_high
                                sar = max(sar, prev_high, prev2_high)
                                # Reversal?
                                if highs[i] > sar:
                                    trend_up = True
                                    sar = ep
                                    ep = highs[i]
                                    af = step
                                else:
                                    if lows[i] < ep:
                                        ep = lows[i]
                                        af = min(af + step, step_max)
                        return 1 if trend_up else -1
                    except Exception:
                        return None
                # Helper: RSI series for divergence detection
                def _rsi_series(closes: List[float], n: int = 14) -> List[Optional[float]]:
                    if len(closes) < n + 1:
                        return [None] * len(closes)
                    rsis: List[Optional[float]] = [None] * len(closes)
                    gains = [0.0]
                    losses = [0.0]
                    for i in range(1, len(closes)):
                        ch = closes[i] - closes[i-1]
                        gains.append(max(ch, 0.0))
                        losses.append(max(-ch, 0.0))
                    avg_gain = sum(gains[1:n+1]) / n
                    avg_loss = sum(losses[1:n+1]) / n
                    rs = (avg_gain / max(avg_loss, 1e-12)) if avg_loss else float('inf')
                    rsis[n] = 100.0 - (100.0 / (1.0 + rs))
                    for i in range(n+1, len(gains)):
                        avg_gain = (avg_gain * (n-1) + gains[i]) / n
                        avg_loss = (avg_loss * (n-1) + losses[i]) / n
                        rs = avg_gain / max(avg_loss, 1e-12)
                        rsis[i] = 100.0 - (100.0 / (1.0 + rs))
                    return rsis
                # Helper: MACD series (line, signal, histogram)
                def _macd_series(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
                    def ema_series(vals: List[float], n: int) -> List[Optional[float]]:
                        if len(vals) < n:
                            return [None] * len(vals)
                        res: List[Optional[float]] = [None] * (n-1)
                        sma = sum(vals[:n]) / n
                        res.append(sma)
                        k = 2.0 / (n + 1.0)
                        for i in range(n, len(vals)):
                            prev = res[-1] if res[-1] is not None else vals[i-1]
                            res.append(vals[i] * k + (prev or 0.0) * (1 - k))
                        return res
                    e_fast = ema_series(closes, fast)
                    e_slow = ema_series(closes, slow)
                    macd_line: List[Optional[float]] = []
                    for i in range(len(closes)):
                        ef = e_fast[i] if i < len(e_fast) else None
                        es = e_slow[i] if i < len(e_slow) else None
                        macd_line.append((ef - es) if (ef is not None and es is not None) else None)
                    # signal line on macd_line values
                    ml_vals = [x for x in macd_line if x is not None]
                    offset = len(macd_line) - len(ml_vals)
                    sig_series_rel = ema_series(ml_vals, signal) if ml_vals else []
                    signal_line: List[Optional[float]] = [None] * offset + sig_series_rel
                    hist: List[Optional[float]] = []
                    for i in range(len(macd_line)):
                        m = macd_line[i]
                        s = signal_line[i] if i < len(signal_line) else None
                        hist.append((m - s) if (m is not None and s is not None) else None)
                    return macd_line, signal_line, hist
                # Helper: find swing points (indices) for highs/lows
                def _find_swings(arr: List[float], window: int = 3, find_high: bool = True) -> List[int]:
                    idxs: List[int] = []
                    n = len(arr)
                    for i in range(window, n - window):
                        mid = arr[i]
                        ok = True
                        for j in range(i - window, i + window + 1):
                            if j == i:
                                continue
                            if find_high and not (mid >= arr[j]):
                                ok = False; break
                            if not find_high and not (mid <= arr[j]):
                                ok = False; break
                        if ok:
                            idxs.append(i)
                    return idxs
                def _adx(highs: List[float], lows: List[float], closes: List[float], n: int = 14) -> Optional[float]:
                    if len(highs) < n+1 or len(lows) < n+1 or len(closes) < n+1:
                        return None
                    dm_plus: List[float] = []
                    dm_minus: List[float] = []
                    tr_list: List[float] = []
                    for i in range(1, len(closes)):
                        up = highs[i] - highs[i-1]
                        dn = lows[i-1] - lows[i]
                        dm_plus.append(max(up, 0.0) if up > dn else 0.0)
                        dm_minus.append(max(dn, 0.0) if dn > up else 0.0)
                        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
                        tr_list.append(tr)
                    # Smooth averages (Wilder)
                    def smooth(vals: List[float], n: int) -> List[float]:
                        if len(vals) < n:
                            return []
                        s = sum(vals[:n]) / n
                        res = [s]
                        for i in range(n, len(vals)):
                            s = (s * (n - 1) + vals[i]) / n
                            res.append(s)
                        return res
                    tr_s = smooth(tr_list, n)
                    dm_p_s = smooth(dm_plus, n)
                    dm_m_s = smooth(dm_minus, n)
                    if not tr_s or not dm_p_s or not dm_m_s:
                        return None
                    di_plus = [ (dm_p_s[i] / tr_s[i]) * 100.0 if tr_s[i] else 0.0 for i in range(len(tr_s)) ]
                    di_minus = [ (dm_m_s[i] / tr_s[i]) * 100.0 if tr_s[i] else 0.0 for i in range(len(tr_s)) ]
                    dx = []
                    for i in range(len(di_plus)):
                        s = di_plus[i] + di_minus[i]
                        d = abs(di_plus[i] - di_minus[i])
                        dx.append((d/s)*100.0 if s else 0.0)
                    if len(dx) < n:
                        return None
                    adx_vals = [sum(dx[:n]) / n]
                    for i in range(n, len(dx)):
                        adx_vals.append((adx_vals[-1]*(n-1) + dx[i]) / n)
                    return adx_vals[-1] if adx_vals else None
                def _cci(highs: List[float], lows: List[float], closes: List[float], n: int = 20) -> Optional[float]:
                    if len(highs) < n or len(lows) < n or len(closes) < n:
                        return None
                    tp = [ (highs[i] + lows[i] + closes[i]) / 3.0 for i in range(len(closes)) ]
                    sma_tp = _sma(tp, n)
                    if sma_tp is None:
                        return None
                    md = sum(abs(x - sma_tp) for x in tp[-n:]) / n
                    if md == 0:
                        return 0.0
                    return (tp[-1] - sma_tp) / (0.015 * md)
                def _willr(highs: List[float], lows: List[float], closes: List[float], n: int = 14) -> Optional[float]:
                    if len(highs) < n or len(lows) < n or len(closes) < n:
                        return None
                    hh = max(highs[-n:]); ll = min(lows[-n:])
                    if hh == ll:
                        return 0.0
                    return -100.0 * (hh - closes[-1]) / (hh - ll)
                def _roc(closes: List[float], n: int = 12) -> Optional[float]:
                    if len(closes) <= n:
                        return None
                    prev = closes[-n-1]
                    if prev == 0:
                        return 0.0
                    return 100.0 * (closes[-1] - prev) / prev
                def _vortex(highs: List[float], lows: List[float], closes: List[float], n: int = 14) -> Optional[Dict[str, float]]:
                    if len(highs) < n+1 or len(lows) < n+1 or len(closes) < n+1:
                        return None
                    tr_list = []
                    vm_plus = []
                    vm_minus = []
                    for i in range(1, len(closes)):
                        tr_list.append(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])))
                        vm_plus.append(abs(highs[i] - lows[i-1]))
                        vm_minus.append(abs(lows[i] - highs[i-1]))
                    def rolling_sum(vals: List[float], n: int) -> List[float]:
                        if len(vals) < n:
                            return []
                        s = sum(vals[:n])
                        res = [s]
                        for i in range(n, len(vals)):
                            s += vals[i] - vals[i-n]
                            res.append(s)
                        return res
                    tr_n = rolling_sum(tr_list, n)
                    vm_p_n = rolling_sum(vm_plus, n)
                    vm_m_n = rolling_sum(vm_minus, n)
                    if not tr_n or not vm_p_n or not vm_m_n:
                        return None
                    vi_plus = vm_p_n[-1] / max(tr_n[-1], 1e-12)
                    vi_minus = vm_m_n[-1] / max(tr_n[-1], 1e-12)
                    return {"vi_plus": vi_plus, "vi_minus": vi_minus}
                def _dpo(closes: List[float], n: int = 20) -> Optional[float]:
                    if len(closes) < n+1:
                        return None
                    sma_n = _sma(closes, n)
                    if sma_n is None:
                        return None
                    return closes[-1] - sma_n
                def _trix(closes: List[float], n: int = 15) -> Optional[float]:
                    # Efficient triple-EMA implementation returning last TRIX value
                    if len(closes) < n*3 + 2:
                        return None
                    def ema_series_full(arr: List[float], p: int) -> List[Optional[float]]:
                        if len(arr) < p:
                            return [None] * len(arr)
                        res: List[Optional[float]] = [None] * (p - 1)
                        sma = sum(arr[:p]) / p
                        res.append(sma)
                        k = 2.0 / (p + 1.0)
                        for i in range(p, len(arr)):
                            prev = res[-1] if res[-1] is not None else arr[i-1]
                            res.append(arr[i] * k + (prev or 0.0) * (1 - k))
                        return res
                    # E1 over closes
                    e1 = ema_series_full(closes, n)
                    # E2 over numeric part of E1
                    e1_num = [x for x in e1 if x is not None]
                    e2_comp = ema_series_full(e1_num, n)
                    e2 = [None] * (len(e1) - len(e2_comp)) + e2_comp
                    # E3 over numeric part of E2
                    e2_num = [x for x in e2 if x is not None]
                    e3_comp = ema_series_full(e2_num, n)
                    e3 = [None] * (len(e2) - len(e3_comp)) + e3_comp
                    # Find last two numeric E3 values
                    e3_vals = [x for x in e3 if x is not None]
                    if len(e3_vals) < 2:
                        return None
                    prev, cur = e3_vals[-2], e3_vals[-1]
                    if prev == 0:
                        return 0.0
                    return 100.0 * (cur - prev) / abs(prev)
                def _ultimate(highs: List[float], lows: List[float], closes: List[float]) -> Optional[float]:
                    # Periods 7, 14, 28
                    def ult_for(n: int) -> Optional[float]:
                        if len(closes) < n+1:
                            return None
                        bp = 0.0; tr_sum = 0.0
                        for i in range(len(closes)-n+1, len(closes)):
                            bp += closes[i] - min(lows[i], closes[i-1])
                            tr_sum += max(highs[i], closes[i-1]) - min(lows[i], closes[i-1])
                        return (bp / max(tr_sum,1e-12))*100.0
                    u7 = ult_for(7); u14 = ult_for(14); u28 = ult_for(28)
                    if u7 is None or u14 is None or u28 is None:
                        return None
                    return (4*u7 + 2*u14 + u28) / 7.0
                def _kst(closes: List[float]) -> Optional[float]:
                    # KST uses ROCs with different windows and SMAs; we approximate last value
                    def roc(arr: List[float], n: int) -> Optional[float]:
                        if len(arr) <= n:
                            return None
                        p = arr[-n-1]
                        if p == 0:
                            return 0.0
                        return 100.0 * (arr[-1] - p) / p
                    r1 = roc(closes, 10); r2 = roc(closes, 15); r3 = roc(closes, 20); r4 = roc(closes, 30)
                    parts = [ (r1 or 0.0), 2*(r2 or 0.0), 3*(r3 or 0.0), 4*(r4 or 0.0) ]
                    return sum(parts)
                def _ichimoku(highs: List[float], lows: List[float]) -> Tuple[Optional[float], Optional[float]]:
                    # Tenkan (9), Kijun (26)
                    if len(highs) < 26 or len(lows) < 26:
                        return None, None
                    tenkan = (max(highs[-9:]) + min(lows[-9:])) / 2.0
                    kijun = (max(highs[-26:]) + min(lows[-26:])) / 2.0
                    return tenkan, kijun
                # Series helpers (minimal, last values focus)
                def _stoch_series(highs: List[float], lows: List[float], closes: List[float], n: int = 14, d: int = 3) -> Tuple[List[Optional[float]], List[Optional[float]]]:
                    k_raw: List[Optional[float]] = []
                    if not highs or not lows or not closes:
                        return k_raw, k_raw
                    L = min(len(highs), len(lows), len(closes))
                    for i in range(L):
                        if i < n - 1:
                            k_raw.append(None)
                            continue
                        hi = max(highs[i-n+1:i+1])
                        lo = min(lows[i-n+1:i+1])
                        rng = hi - lo
                        if rng == 0:
                            k_raw.append(50.0)
                        else:
                            k_raw.append(100.0 * (closes[i] - lo) / rng)
                    def sma(arr: List[Optional[float]], w: int) -> List[Optional[float]]:
                        out: List[Optional[float]] = []
                        buf: List[float] = []
                        for v in arr:
                            if v is None:
                                buf.append(0.0)
                            else:
                                buf.append(float(v))
                            if len(buf) < w:
                                out.append(None)
                            else:
                                out.append(sum(buf[-w:]) / w)
                        return out
                    k = sma(k_raw, d)
                    dline = sma(k, d) if len(k) >= d else k
                    return k, dline
                def _stochrsi_series_from_rsi(rsi_series: List[Optional[float]], n: int = 14) -> List[Optional[float]]:
                    out: List[Optional[float]] = []
                    for i in range(len(rsi_series)):
                        if i < n - 1:
                            out.append(None)
                            continue
                        window = [x for x in rsi_series[i-n+1:i+1] if x is not None]
                        if not window:
                            out.append(None)
                            continue
                        hi = max(window); lo = min(window)
                        rng = hi - lo
                        if rng == 0:
                            out.append(0.5)
                        else:
                            out.append((rsi_series[i] - lo) / rng)
                    return out
                def _cci_series(highs: List[float], lows: List[float], closes: List[float], n: int = 20) -> List[Optional[float]]:
                    out: List[Optional[float]] = []
                    if not highs or not lows or not closes:
                        return out
                    L = min(len(highs), len(lows), len(closes))
                    tp: List[float] = []
                    for i in range(L):
                        tpi = (highs[i] + lows[i] + closes[i]) / 3.0
                        tp.append(tpi)
                        if i < n - 1:
                            out.append(None)
                            continue
                        sma = sum(tp[i-n+1:i+1]) / n
                        md = sum(abs(tp[j] - sma) for j in range(i-n+1, i+1)) / n
                        if md == 0:
                            out.append(0.0)
                        else:
                            out.append((tp[i] - sma) / (0.015 * md))
                    return out
                def _willr_series(highs: List[float], lows: List[float], closes: List[float], n: int = 14) -> List[Optional[float]]:
                    out: List[Optional[float]] = []
                    if not highs or not lows or not closes:
                        return out
                    L = min(len(highs), len(lows), len(closes))
                    for i in range(L):
                        if i < n - 1:
                            out.append(None)
                            continue
                        hh = max(highs[i-n+1:i+1]); ll = min(lows[i-n+1:i+1])
                        rng = hh - ll
                        if rng == 0:
                            out.append(0.0)
                        else:
                            out.append(-100.0 * (hh - closes[i]) / rng)
                    return out
                def _di_last(highs: List[float], lows: List[float], closes: List[float], n: int = 14) -> Tuple[Optional[float], Optional[float]]:
                    try:
                        if len(highs) < n + 1 or len(lows) < n + 1 or len(closes) < n + 1:
                            return None, None
                        tr = []
                        dm_p = []
                        dm_m = []
                        for i in range(1, len(highs)):
                            up = highs[i] - highs[i-1]
                            dn = lows[i-1] - lows[i]
                            dm_p.append(max(up, 0.0) if up > dn else 0.0)
                            dm_m.append(max(dn, 0.0) if dn > up else 0.0)
                            tr.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
                        def rma(arr: List[float], period: int) -> List[float]:
                            out: List[float] = []
                            if not arr:
                                return out
                            avg = sum(arr[:period]) / period
                            out.append(avg)
                            alpha = 1.0 / period
                            for v in arr[period:]:
                                avg = (1 - alpha) * avg + alpha * v
                                out.append(avg)
                            return out
                        tr_s = rma(tr, n)
                        dm_p_s = rma(dm_p, n)
                        dm_m_s = rma(dm_m, n)
                        if not tr_s or not dm_p_s or not dm_m_s:
                            return None, None
                        di_p = (dm_p_s[-1] / tr_s[-1]) * 100.0 if tr_s[-1] else None
                        di_m = (dm_m_s[-1] / tr_s[-1]) * 100.0 if tr_s[-1] else None
                        return di_p, di_m
                    except Exception:
                        return None, None
                # Compute missing values
                if adx is None and highs and lows:
                    adx = _adx(highs, lows, closes)
                if cci is None and highs and lows:
                    cci = _cci(highs, lows, closes)
                if willr is None and highs and lows:
                    willr = _willr(highs, lows, closes)
                if roc is None:
                    roc = _roc(closes)
                if (not vortex) and highs and lows:
                    vcalc = _vortex(highs, lows, closes)
                    vortex = vcalc or {}
                if dpo is None:
                    dpo = _dpo(closes)
                if trix is None:
                    trix = _trix(closes)
                if ult is None and highs and lows:
                    ult = _ultimate(highs, lows, closes)
                if kst is None:
                    kst = _kst(closes)
                if (ich_tenkan is None or ich_kijun is None) and highs and lows:
                    ten, kij = _ichimoku(highs, lows)
                    if ich_tenkan is None:
                        ich_tenkan = ten
                    if ich_kijun is None:
                        ich_kijun = kij
                if (not kelt) and atr is not None and ema20 is not None:
                    kelt = {"upper": ema20 + 2*atr, "lower": ema20 - 2*atr, "middle": ema20}
                if (not env) and sma20 is not None:
                    env = {"upper": sma20 * 1.02, "lower": sma20 * 0.98, "middle": sma20}
                # Volume-based fallbacks
                if vol_available and not CFG.STRICT_IND_ONLY:
                    if mfi is None and highs and lows:
                        mfi = _mfi(highs, lows, closes, volumes)
                    if obv is None:
                        obv = _obv(closes, volumes)
                    if chaikin is None and highs and lows:
                        chaikin = _chaikin_osc(highs, lows, closes, volumes)
                    if eom is None and highs and lows:
                        eom = _eom(highs, lows, closes, volumes)
                    if force is None:
                        force = _force_index(closes, volumes)
                # PSAR dir fallback from candles if still None
                if psar_dir is None and highs and lows and not CFG.STRICT_IND_ONLY:
                    psar_dir = _psar_dir(highs, lows)
                # Mass Index (no volume needed)
                if massi is None and highs and lows and not CFG.STRICT_IND_ONLY:
                    massi = _mass_index(highs, lows)
        except Exception:
            pass

        # Qualify supports and classification
        # --- Force re-extract missing indicators from raw indicator_output (scan backwards for non-null) ---
        def _scan_last(ind_any: Any, key: str) -> Optional[float]:
            if not ind_any:
                return None
            if isinstance(ind_any, list):
                for row in reversed(ind_any):
                    if isinstance(row, dict) and key in row:
                        v = row.get(key)
                        if v is not None:
                            try:
                                return float(v)
                            except Exception:
                                return v
            return None
        def _scan_first_non_null(keys: List[str]) -> Optional[float]:
            for k in keys:
                v = _scan_last(ind_trend, k) or _scan_last(ind_chan, k)
                if v is not None:
                    return v
            return None
        # Dynamic key name mappings using detected parameters
        if rsi is None:
            try:
                if 'rsi_period_detected' in globals() and rsi_period_detected:
                    rsi = Extract.rsi(ind_trend, rsi_period_detected) or Extract.rsi(ind_chan, rsi_period_detected)
            except NameError:
                pass
            if rsi is None:
                rsi = _scan_first_non_null(["RSI14","rsi","RSI_14"])  # fallback to main RSI
        if macd_h is None:
            try:
                if ('macd_fast_detected' in globals() and 'macd_slow_detected' in globals() and 
                    'macd_signal_detected' in globals() and macd_fast_detected and macd_slow_detected and macd_signal_detected):
                    macd_h = Extract.macd_hist(ind_trend, macd_fast_detected, macd_slow_detected, macd_signal_detected) or Extract.macd_hist(ind_chan, macd_fast_detected, macd_slow_detected, macd_signal_detected)
            except NameError:
                pass
            if macd_h is None:
                macd_h = _scan_first_non_null(["MACD_hist_12_26_9","macd_hist","MACD_HIST"])  # fallback MACD histogram
        if adx is None:
            try:
                if 'adx_period_detected' in globals() and adx_period_detected:
                    adx = Extract.adx(ind_trend, adx_period_detected) or Extract.adx(ind_chan, adx_period_detected)
            except NameError:
                pass
            if adx is None:
                adx = _scan_first_non_null(["ADX14","adx","ADX_14"])  # fallback ADX
        if stoch_k is None or stoch_d is None:
            try:
                if ('stoch_k_detected' in globals() and 'stoch_d_detected' in globals() and 
                    stoch_k_detected and stoch_d_detected):
                    stoch_data = Extract.stochastic(ind_trend, stoch_k_detected, stoch_d_detected) or Extract.stochastic(ind_chan, stoch_k_detected, stoch_d_detected)
                    if stoch_data:
                        if stoch_k is None:
                            stoch_k = stoch_data.get('k')
                        if stoch_d is None:
                            stoch_d = stoch_data.get('d')
            except NameError:
                pass
            if stoch_k is None:
                stoch_k = _scan_first_non_null(["StochK_14_3","stoch_k"])  # fallback Stoch K
            if stoch_d is None:
                stoch_d = _scan_first_non_null(["StochD_14_3","stoch_d"])  # fallback Stoch D
        if stochrsi is None:
            try:
                if 'stochrsi_period_detected' in globals() and stochrsi_period_detected:
                    stochrsi = Extract.stochrsi(ind_trend, stochrsi_period_detected) or Extract.stochrsi(ind_chan, stochrsi_period_detected)
            except NameError:
                pass
            if stochrsi is None:
                stochrsi = _scan_first_non_null(["StochRSI14","StochRSI_K_14"])  # fallback raw stoch rsi (0-1 expected)
            if stochrsi and stochrsi > 1.5:  # some exports maybe 0-100
                stochrsi = stochrsi / 100.0
        if atr is None:
            try:
                if 'atr_period_detected' in globals() and atr_period_detected:
                    atr = Extract.atr_latest(ind_trend, atr_period_detected) or Extract.atr_latest(ind_chan, atr_period_detected)
            except NameError:
                pass
            if atr is None:
                atr = _scan_first_non_null(["ATR14","atr14","ATR_14"])  # fallback ATR
        if mfi is None:
            try:
                if 'mfi_period_detected' in globals() and mfi_period_detected:
                    # Use template keys directly to bypass signature conflicts
                    template_keys = [f"MFI{mfi_period_detected}", f"MFI_{mfi_period_detected}"]
                    for key in template_keys:
                        val = _scan_first_non_null([key])
                        if val is not None:
                            mfi = val
                            break
            except NameError:
                pass
            if mfi is None:
                mfi = _scan_first_non_null(["MFI14","mfi14","MFI_14"])  # fallback Money Flow Index
        if cci is None:
            try:
                if 'cci_period_detected' in globals() and cci_period_detected:
                    # CCI với tham số động - bypass signature conflict như MFI
                    template_keys = [f"CCI{cci_period_detected}", f"CCI_{cci_period_detected}"]
                    for key in template_keys:
                        val = _scan_first_non_null([key])
                        if val is not None:
                            cci = val
                            break
            except NameError:
                pass
            if cci is None:
                cci = _scan_first_non_null(["CCI20","cci20","CCI_20"])  # fallback CCI
        if willr is None:
            try:
                if 'willr_period_detected' in globals() and willr_period_detected:
                    # Williams %R với tham số động - bypass signature conflict
                    template_keys = [f"WilliamsR{willr_period_detected}", f"WilliamsR_{willr_period_detected}", f"WILLR{willr_period_detected}"]
                    for key in template_keys:
                        val = _scan_first_non_null([key])
                        if val is not None:
                            willr = val
                            break
            except NameError:
                pass
            if willr is None:
                willr = _scan_first_non_null(["WilliamsR14","williams_r14","WILLR14"])  # fallback Williams %R
        if roc is None:
            try:
                if 'roc_period_detected' in globals() and roc_period_detected:
                    # ROC với tham số động - bypass signature conflict
                    template_keys = [f"ROC{roc_period_detected}", f"ROC_{roc_period_detected}"]
                    for key in template_keys:
                        val = _scan_first_non_null([key])
                        if val is not None:
                            roc = val
                            break
            except NameError:
                pass
            if roc is None:
                roc = _scan_first_non_null(["ROC20","ROC_20","ROC12","roc12","ROC_12"])  # fallback ROC (prefer longer period if exported)
        if obv is None:
            obv = _scan_first_non_null(["OBV"])  # OBV
        if chaikin is None:
            # Try various Chaikin parameter alternatives
            chaikin = _scan_first_non_null(["Chaikin20","Chaikin_20","Chaikin18","Chaikin_18","Chaikin14","Chaikin_14"])  # Chaikin Osc
        if eom is None:
            eom = _scan_first_non_null(["EOM20","EOM_20"])  # EOM
        if force is None:
            force = _scan_first_non_null(["ForceIndex14","ForceIndex_14","ForceIndex13","Force_13"])  # Force Index any period
        if trix is None:
            trix = _scan_first_non_null(["TRIX14","TRIX_14"])  # TRIX
        if dpo is None:
            dpo = _scan_first_non_null(["DPO20","DPO_20"])  # DPO
        if massi is None:
            # Use dynamic Mass Index detection with extracted parameters
            if _params.get('mass_fast') and _params.get('mass_slow'):
                mass_template = f"MassIndex_{_params['mass_fast']}_{_params['mass_slow']}"
                massi = _scan_first_non_null([mass_template])
            if massi is None:
                massi = _scan_first_non_null(["MassIndex_9_25","MassIndex","MASS_INDEX"])  # Fallback
        # Vortex
        if (not vortex) or vortex.get('vi_plus') is None:
            vip = _scan_first_non_null(["VI+_14","VIplus_14"])  # may contain + in key
            vim = _scan_first_non_null(["VI-_14","VIminus_14"])  # minus line
            if vip is not None or vim is not None:
                vortex = {"vi_plus": vip, "vi_minus": vim}
        if kst is None:
            kst = _scan_first_non_null(["KST"])  # KST composite
        if ult is None:
            ult = _scan_first_non_null(["UltimateOscillator","Ultimate_Osc"])  # Ultimate Oscillator
        # Keltner & Envelopes (assemble dicts if missing)
        if (not kelt):
            ku = _scan_first_non_null(["Keltner_Upper_20"]) ; km = _scan_first_non_null(["Keltner_Middle_20"]) ; kl = _scan_first_non_null(["Keltner_Lower_20"]) 
            if any(v is not None for v in [ku,km,kl]):
                kelt = {"upper": ku, "middle": km, "lower": kl}
        if (not env):
            eu = _scan_first_non_null(["Envelope_Upper_20_2.0"]) ; em = _scan_first_non_null(["Envelope_Middle_20_2.0"]) ; el = _scan_first_non_null(["Envelope_Lower_20_2.0"]) 
            if any(v is not None for v in [eu,em,el]):
                env = {"upper": eu, "middle": em, "lower": el}
        # Bollinger manual assemble if bb missing
        if (not bb):
            bup = _scan_first_non_null(["BB_Upper_20_2"]) ; bmid = _scan_first_non_null(["BB_Middle_20_2"]) ; blo = _scan_first_non_null(["BB_Lower_20_2"]) 
            if any(v is not None for v in [bup,bmid,blo]):
                bb = {"upper": bup, "middle": bmid, "lower": blo}
        # Donchian was handled earlier for channel; ensure values if present using detected params
        try:
            if d_lo is None and 'donch_win_detected' in globals() and donch_win_detected is not None:
                d_lo = Extract._last_key_with_params(ind_chan or ind_trend or {}, f"Donchian_Lower_{donch_win_detected}", {})
        except NameError:
            pass
        if d_lo is None:
            d_lo = _scan_first_non_null(["Donchian_Lower_20"])  # fallback to default
        try:
            if d_up is None and 'donch_win_detected' in globals() and donch_win_detected is not None:
                d_up = Extract._last_key_with_params(ind_chan or ind_trend or {}, f"Donchian_Upper_{donch_win_detected}", {})
        except NameError:
            pass
        if d_up is None:
            d_up = _scan_first_non_null(["Donchian_Upper_20"])  # fallback to default
        # EMA100 addition
        ema100 = _last_from_fallback("EMA_100") or _scan_first_non_null(["EMA100","EMA_100"])  # if not exported, compute
        if ema100 is None and closes and len(closes) >= 100 and not CFG.STRICT_IND_ONLY:
            ema100 = _ema(closes,100)
        # Later we can add classification and crossovers with ema100
        rsi_sup = None if rsi is None else (rsi >= 55)
        macd_sup = None if macd_h is None else (macd_h > 0)
        psar_sup = None if psar_dir is None else (psar_dir > 0)
        ema_sup = None if ema20 is None or price_trend is None else (price_trend > ema20)
        ema50_sup = None if ema50 is None or price_trend is None else (price_trend > ema50)
        ema100_sup = None if ema100 is None or price_trend is None else (price_trend > ema100)
        ema200_sup = None if ema200 is None or price_trend is None else (price_trend > ema200)
        sma_sup = None if sma20 is None or price_trend is None else (price_trend > sma20)
        sma50_sup = None if sma50 is None or price_trend is None else (price_trend > sma50)
        wma_sup = None if wma20 is None or price_trend is None else (price_trend > wma20)
        # TEMA support flag
        try:
            tema_sup = None if tema20 is None or price_trend is None else (price_trend > tema20)
        except Exception:
            tema_sup = None
        ichi_sup = None
        if ich_tenkan is not None and ich_kijun is not None:
            ichi_sup = (ffloat(ich_tenkan, 0) > ffloat(ich_kijun, 0))
        vortex_sup = None
        if isinstance(vortex, dict) and vortex.get('vi_plus') is not None and vortex.get('vi_minus') is not None:
            try:
                vortex_sup = float(vortex.get('vi_plus')) > float(vortex.get('vi_minus'))
            except Exception:
                vortex_sup = None
        kst_sup = None if kst is None else (kst > 0)
        ult_sup = None if ult is None else (ult > 50)

        # Dynamic MA value extraction using detected periods
        def _get_ma_value_dynamic(ma_type: str, period: int) -> Optional[float]:
            """Get MA value using dynamic period detection"""
            # Try detected periods first with safe checks
            try:
                if ma_type.upper() == "EMA" and 'ema_periods' in globals() and ema_periods and period in ema_periods:
                    return Extract.ma_value(ind_trend, "EMA", period) or Extract.ma_value(ind_chan, "EMA", period)
                elif ma_type.upper() == "SMA" and 'sma_periods' in globals() and sma_periods and period in sma_periods:
                    return Extract.ma_value(ind_trend, "SMA", period) or Extract.ma_value(ind_chan, "SMA", period)
                elif ma_type.upper() == "WMA" and 'wma_periods' in globals() and wma_periods and period in wma_periods:
                    return Extract.ma_value(ind_trend, "WMA", period) or Extract.ma_value(ind_chan, "WMA", period)
                elif ma_type.upper() == "TEMA" and 'tema_periods' in globals() and tema_periods and period in tema_periods:
                    return Extract.ma_value(ind_trend, "TEMA", period) or Extract.ma_value(ind_chan, "TEMA", period)
            except (NameError, UnboundLocalError):
                pass
            
            # Fallback to hardcoded key lookup
            key = f"{ma_type.upper()}_{period}"
            return _scan_first_non_null([key, f"{ma_type.upper()}{period}", f"{ma_type.lower()}{period}"])
            
        def _get_ma_two_values_dynamic(ma_type: str, period: int) -> Tuple[Optional[float], Optional[float]]:
            """Get last two MA values using dynamic period detection"""
            # Try detected periods first with safe checks
            try:
                if ma_type.upper() == "EMA" and 'ema_periods' in globals() and ema_periods and period in ema_periods:
                    return _last_two_fallback(f"EMA{period}") if f"EMA{period}" in (ind_trend[-1] if ind_trend else {}) else _last_two_fallback(f"EMA_{period}")
                elif ma_type.upper() == "SMA" and 'sma_periods' in globals() and sma_periods and period in sma_periods:
                    return _last_two_fallback(f"SMA{period}") if f"SMA{period}" in (ind_trend[-1] if ind_trend else {}) else _last_two_fallback(f"SMA_{period}")
                elif ma_type.upper() == "WMA" and 'wma_periods' in globals() and wma_periods and period in wma_periods:
                    return _last_two_fallback(f"WMA{period}") if f"WMA{period}" in (ind_trend[-1] if ind_trend else {}) else _last_two_fallback(f"WMA_{period}")
                elif ma_type.upper() == "TEMA" and 'tema_periods' in globals() and tema_periods and period in tema_periods:
                    return _last_two_fallback(f"TEMA{period}") if f"TEMA{period}" in (ind_trend[-1] if ind_trend else {}) else _last_two_fallback(f"TEMA_{period}")
            except (NameError, UnboundLocalError):
                pass
            
            # Fallback to hardcoded approach
            key = f"{ma_type.upper()}_{period}"
            return _last_two_fallback(key)

        # Update MA values using ALL detected periods dynamically
        # Create local variables for all detected MA periods
        try:
            if 'ema_periods' in globals() and ema_periods:
                for period in ema_periods:
                    locals()[f"ema{period}"] = _get_ma_value_dynamic("EMA", period)
        except NameError:
            pass
        
        try:
            if 'sma_periods' in globals() and sma_periods:
                for period in sma_periods:
                    locals()[f"sma{period}"] = _get_ma_value_dynamic("SMA", period)
        except NameError:
            pass
            
        try:
            if 'wma_periods' in globals() and wma_periods:
                for period in wma_periods:
                    locals()[f"wma{period}"] = _get_ma_value_dynamic("WMA", period)
        except NameError:
            pass
            
        try:
            if 'tema_periods' in globals() and tema_periods:
                for period in tema_periods:
                    locals()[f"tema{period}"] = _get_ma_value_dynamic("TEMA", period)
        except NameError:
            pass

        # Detect EMA20/EMA50 cross and price vs EMA50 recency (also extended with EMA100)
        ema20_prev, ema20_now = _get_ma_two_values_dynamic("EMA", 20)
        ema50_prev, ema50_now = _get_ma_two_values_dynamic("EMA", 50)
        ema100_prev, ema100_now = _get_ma_two_values_dynamic("EMA", 100) if ema100 is not None else (None, None)
        p_prev, p_now = last_two_closes(data_trend.candles)
        ema20_50_note = None
        try:
            if ema20_prev is not None and ema50_prev is not None and ema20_now is not None and ema50_now is not None:
                prev_diff = ffloat(ema20_prev) - ffloat(ema50_prev)
                now_diff = ffloat(ema20_now) - ffloat(ema50_now)
                if prev_diff <= 0 and now_diff > 0:
                    ema20_50_note = "EMA20 đang cắt lên EMA50"
                elif prev_diff >= 0 and now_diff < 0:
                    ema20_50_note = "EMA20 đang cắt xuống EMA50"
        except Exception:
            ema20_50_note = None

        price_vs_ema50_note = None
        try:
            if p_prev is not None and p_now is not None and ema50_prev is not None and ema50_now is not None:
                if p_prev <= ema50_prev and p_now > ema50_now:
                    price_vs_ema50_note = "Giá vừa đóng cửa trên EMA50"
                elif p_prev >= ema50_prev and p_now < ema50_now:
                    price_vs_ema50_note = "Giá vừa đóng cửa dưới EMA50"
        except Exception:
            price_vs_ema50_note = None

        # Additional MA cross states (EMA50/200, EMA50/100, EMA100/200)
        ema50_200_note = None
        ema50_100_note = None
        ema100_200_note = None
        try:
            ema200_prev, ema200_now = _get_ma_two_values_dynamic("EMA", 200)
            if ema50_prev is not None and ema200_prev is not None and ema50_now is not None and ema200_now is not None:
                prev_diff = ffloat(ema50_prev) - ffloat(ema200_prev)
                now_diff = ffloat(ema50_now) - ffloat(ema200_now)
                if prev_diff <= 0 and now_diff > 0:
                    ema50_200_note = "EMA50 đang cắt lên EMA200 (Golden cross)"
                elif prev_diff >= 0 and now_diff < 0:
                    ema50_200_note = "EMA50 đang cắt xuống EMA200 (Death cross)"
            if ema50_prev is not None and ema100_prev is not None and ema50_now is not None and ema100_now is not None:
                prev_diff2 = ffloat(ema50_prev) - ffloat(ema100_prev)
                now_diff2 = ffloat(ema50_now) - ffloat(ema100_now)
                if prev_diff2 <= 0 and now_diff2 > 0:
                    ema50_100_note = "EMA50 đang cắt lên EMA100"
                elif prev_diff2 >= 0 and now_diff2 < 0:
                    ema50_100_note = "EMA50 đang cắt xuống EMA100"
            if ema100_prev is not None and ema200_prev is not None and ema100_now is not None and ema200_now is not None:
                prev_diff3 = ffloat(ema100_prev) - ffloat(ema200_prev)
                now_diff3 = ffloat(ema100_now) - ffloat(ema200_now)
                if prev_diff3 <= 0 and now_diff3 > 0:
                    ema100_200_note = "EMA100 đang cắt lên EMA200"
                elif prev_diff3 >= 0 and now_diff3 < 0:
                    ema100_200_note = "EMA100 đang cắt xuống EMA200"
        except Exception:
            ema50_200_note = None
            ema50_100_note = None
            ema100_200_note = None

        sma20_50_note = None
        try:
            sma20_prev, sma20_now = _get_ma_two_values_dynamic("SMA", 20)
            sma50_prev2, sma50_now2 = _get_ma_two_values_dynamic("SMA", 50)
            if sma20_prev is not None and sma50_prev2 is not None and sma20_now is not None and sma50_now2 is not None:
                prev_diff = ffloat(sma20_prev) - ffloat(sma50_prev2)
                now_diff = ffloat(sma20_now) - ffloat(sma50_now2)
                if prev_diff <= 0 and now_diff > 0:
                    sma20_50_note = "SMA20 đang cắt lên SMA50"
                elif prev_diff >= 0 and now_diff < 0:
                    sma20_50_note = "SMA20 đang cắt xuống SMA50"
        except Exception:
            sma20_50_note = None

        # Divergence detection (RSI & MACD histogram)
        rsi_div_note = None
        macd_div_note = None
        try:
            # Build series if we have closes/highs/lows
            if closes and highs and lows:
                rsi_ser = _rsi_series(closes) or []
                macd_line_ser, macd_sig_ser, macd_hist_ser = _macd_series(closes)
                # Find price swings
                p_high_idxs = _find_swings(highs, window=3, find_high=True)
                p_low_idxs = _find_swings(lows, window=3, find_high=False)
                # RSI swings
                rsi_vals = [v for v in rsi_ser]
                rsi_high_idxs = _find_swings([v if v is not None else 0.0 for v in rsi_vals], window=3, find_high=True)
                rsi_low_idxs = _find_swings([v if v is not None else 0.0 for v in rsi_vals], window=3, find_high=False)
                # MACD hist swings
                mh_vals = [v if v is not None else 0.0 for v in (macd_hist_ser or [])]
                mh_high_idxs = _find_swings(mh_vals, window=3, find_high=True)
                mh_low_idxs = _find_swings(mh_vals, window=3, find_high=False)
                # Evaluate RSI divergence
                if len(p_low_idxs) >= 2 and len(rsi_low_idxs) >= 2:
                    pl1, pl2 = p_low_idxs[-2], p_low_idxs[-1]
                    rl1, rl2 = rsi_low_idxs[-2], rsi_low_idxs[-1]
                    if lows[pl2] < lows[pl1] and (rsi_vals[rl2] or 0) > (rsi_vals[rl1] or 0):
                        rsi_div_note = "phân kỳ tăng"
                if not rsi_div_note and len(p_high_idxs) >= 2 and len(rsi_high_idxs) >= 2:
                    ph1, ph2 = p_high_idxs[-2], p_high_idxs[-1]
                    rh1, rh2 = rsi_high_idxs[-2], rsi_high_idxs[-1]
                    if highs[ph2] > highs[ph1] and (rsi_vals[rh2] or 0) < (rsi_vals[rh1] or 0):
                        rsi_div_note = "phân kỳ giảm"
                # Evaluate MACD histogram divergence
                if len(p_low_idxs) >= 2 and len(mh_low_idxs) >= 2:
                    pl1, pl2 = p_low_idxs[-2], p_low_idxs[-1]
                    ml1, ml2 = mh_low_idxs[-2], mh_low_idxs[-1]
                    if lows[pl2] < lows[pl1] and mh_vals[ml2] > mh_vals[ml1]:
                        macd_div_note = "phân kỳ tăng"
                if not macd_div_note and len(p_high_idxs) >= 2 and len(mh_high_idxs) >= 2:
                    ph1, ph2 = p_high_idxs[-2], p_high_idxs[-1]
                    mh1, mh2 = mh_high_idxs[-2], mh_high_idxs[-1]
                    if highs[ph2] > highs[ph1] and mh_vals[mh2] < mh_vals[mh1]:
                        macd_div_note = "phân kỳ giảm"
        except Exception:
            rsi_div_note = None
            macd_div_note = None

        # RSI classification (granular) - Enhanced with more descriptive states
        if rsi is None:
            rsi_state = None
        else:
            if rsi > 80:
                rsi_state = "quá mua mạnh (>80)"
            elif rsi > 70:
                rsi_state = "quá mua (>70)"
            elif rsi > 60:
                rsi_state = "thiên mua (>60)"
            elif 40 <= rsi <= 60:
                rsi_state = "trung tính (40-60)"
            elif rsi < 20:
                rsi_state = "quá bán mạnh (<20)"
            elif rsi < 30:
                rsi_state = "quá bán (<30)"
            elif rsi < 40:
                rsi_state = "thiên bán (<40)"
            else:
                rsi_state = "trung tính"
        rsi_extra_note = None
        try:
            if closes:
                rsi_ser = _rsi_series(closes) or []
                # 50-cross bias (recent change)
                if len(rsi_ser) >= 2 and rsi_ser[-2] is not None and rsi_ser[-1] is not None:
                    if rsi_ser[-2] <= 50 and rsi_ser[-1] > 50:
                        rsi_extra_note = "RSI vừa cắt lên 50 (thiên mua)"
                    elif rsi_ser[-2] >= 50 and rsi_ser[-1] < 50:
                        rsi_extra_note = "RSI vừa cắt xuống 50 (thiên bán)"
                # Strong extremes
                if rsi is not None:
                    if rsi >= 80:
                        rsi_extra_note = (rsi_extra_note + "; " if rsi_extra_note else "") + "RSI vùng cực mạnh (>80)"
                    elif rsi <= 20:
                        rsi_extra_note = (rsi_extra_note + "; " if rsi_extra_note else "") + "RSI vùng cực yếu (<20)"
        except Exception:
            rsi_extra_note = None

        # Fibonacci proximity: prefer fib_* from indicator_output; fallback to last 120-bar hi/lo, then Donchian bounds
        fibo_note = None
        fibo_status = None  # default status text when not near key levels
        try:
            def _eval_fibo_near(price: Optional[float], lo: Optional[float], hi: Optional[float]) -> Tuple[Optional[str], Optional[str]]:
                if price is None or lo is None or hi is None or hi <= lo:
                    return None, "không khả dụng"
                lvl382 = lo + 0.382 * (hi - lo)
                lvl500 = lo + 0.500 * (hi - lo)
                lvl618 = lo + 0.618 * (hi - lo)
                dist = [
                    (abs(price - lvl382) / max(abs(price), 1e-12), "38.2%"),
                    (abs(price - lvl500) / max(abs(price), 1e-12), "50%"),
                    (abs(price - lvl618) / max(abs(price), 1e-12), "61.8%"),
                ]
                dist.sort(key=lambda x: x[0])
                if dist[0][0] <= 0.003:  # within 0.3%
                    return f"Giá gần mức {dist[0][1]} fibo", None
                return None, "không gần mức quan trọng"

            # QUICK DIRECT PATH: if fib levels exist in any indicator row, evaluate immediately
            try:
                if fibo_note is None:
                    fib_level_keys = [
                        "fib_0.0","fib_0","fib_100.0","fib_100","fib_23.6","fib_38.2","fib_50.0","fib_61.8","fib_78.6",
                        "fib_236","fib_382","fib_500","fib_618","fib_786"
                    ]
                    last_row = None
                    # Prefer trend indicators list
                    if isinstance(ind_trend, list) and ind_trend:
                        cand = ind_trend[-1]
                        if isinstance(cand, dict):
                            last_row = cand
                    if last_row is None and isinstance(ind_chan, list) and ind_chan:
                        cand = ind_chan[-1]
                        if isinstance(cand, dict):
                            last_row = cand
                    if last_row is not None:
                        # derive price if missing
                        if price_trend is None:
                            prw = last_row.get("close") or last_row.get("c")
                            if isinstance(prw, (int, float)):
                                price_trend = float(prw)
                        lo_vals = []
                        hi_vals = []
                        for k,v in last_row.items():
                            if not isinstance(v,(int,float)):
                                continue
                            if k.startswith("fib_0"):
                                lo_vals.append(float(v))
                            elif k.startswith("fib_100"):
                                hi_vals.append(float(v))
                        if price_trend is not None and lo_vals and hi_vals:
                            lo_b = min(lo_vals); hi_b = max(hi_vals)
                            f_note, f_status = _eval_fibo_near(price_trend, lo_b, hi_b)
                            if f_note or f_status:
                                fibo_note = f_note
                                fibo_status = f_status
                            if DEBUG_MODE:
                                logger.debug(f"[FIBO_DIRECT] price={price_trend} lo={lo_b} hi={hi_b} note={fibo_note} status={fibo_status}")
            except Exception:
                pass

            # Prefer indicator-based fib levels when available
            try:
                def _get_from_ind_levels(keys: List[str]) -> Optional[float]:
                    for k in keys:
                        v = _last_from_fallback(k)
                        if isinstance(v, (int, float)):
                            return float(v)
                    return None
                # Extended key sets include simplified forms produced by exporter (fib_382, fib_500, fib_618, fib_786)
                fib_candidates: List[Tuple[str, Optional[float]]] = [
                    ("38.2%", _get_from_ind_levels(["fib_38.2", "fib_38", "fib_382"])),
                    ("50%",   _get_from_ind_levels(["fib_50.0", "fib_50", "fib_500"])),
                    ("61.8%", _get_from_ind_levels(["fib_61.8", "fib_61", "fib_618"])),
                    ("78.6%", _get_from_ind_levels(["fib_78.6", "fib_78", "fib_786"]))
                ]
                # Derive price_trend if missing using indicators or candles
                if price_trend is None:
                    price_trend = Extract._get_price_from_ind(ind_trend or ind_chan or {})
                if price_trend is None and data_trend and isinstance(data_trend.candles, dict):
                    _cseq = None
                    for _k in ("rates", "candles", "data", "bars"):
                        if isinstance(data_trend.candles.get(_k), list) and data_trend.candles.get(_k):
                            _cseq = data_trend.candles.get(_k)
                            break
                    if isinstance(_cseq, list) and _cseq and isinstance(_cseq[-1], dict):
                        _cl = _cseq[-1].get("close") or _cseq[-1].get("c")
                        if isinstance(_cl, (int, float)):
                            price_trend = float(_cl)
                if price_trend is not None:
                    avail = [(abs(price_trend - float(v)) / max(abs(price_trend), 1e-12), name) for name, v in fib_candidates if isinstance(v, (int, float))]
                    if avail:
                        avail.sort(key=lambda x: x[0])
                        if avail[0][0] <= 0.003:  # within 0.3%
                            fibo_note = f"Giá gần mức {avail[0][1]} fibo"
                            fibo_status = None
                        else:
                            fibo_status = "không gần mức quan trọng"
                # If still no conclusion but we do have fib bounds (0% / 100%), evaluate generic proximity
                if fibo_note is None:
                    lo_bound = _last_from_fallback("fib_0.0") or _last_from_fallback("fib_0")
                    hi_bound = _last_from_fallback("fib_100.0") or _last_from_fallback("fib_100")
                    if isinstance(lo_bound, (int, float)) and isinstance(hi_bound, (int, float)) and price_trend is not None:
                        # Re-use evaluation (adds 38.2/50/61.8 logic)
                        f_note, f_status = _eval_fibo_near(price_trend, float(lo_bound), float(hi_bound))
                        if f_note or f_status:
                            fibo_note = f_note
                            fibo_status = f_status
                # Secondary broad scan if still unavailable (sometimes exporter structure differs)
                if fibo_note is None and (fibo_status is None or fibo_status == "không khả dụng"):
                    def _scan_any_fib_bounds() -> Tuple[Optional[float], Optional[float]]:
                        lows_found: List[float] = []
                        highs_found: List[float] = []
                        # Candidate indicator containers to inspect
                        cand_inds = [ind_trend or {}, ind_chan or {}]
                        for tfp in tflist:
                            td = tf_data_cache.get(tfp)
                            if td and td.indicators:
                                cand_inds.append(td.indicators)
                        for indset in cand_inds:
                            if isinstance(indset, list) and indset:
                                last_row = indset[-1]
                                if isinstance(last_row, dict):
                                    for k, v in last_row.items():
                                        if isinstance(v, (int, float)) and k.startswith("fib_"):
                                            if k.startswith("fib_0"):
                                                lows_found.append(float(v))
                                            elif k.startswith("fib_100"):
                                                highs_found.append(float(v))
                            elif isinstance(indset, dict):
                                # Dict where keys are indicator names -> lists
                                for k, vlist in indset.items():
                                    if isinstance(k, str) and k.startswith("fib_") and isinstance(vlist, list) and vlist:
                                        last_v = vlist[-1]
                                        if isinstance(last_v, (int, float)):
                                            if k.startswith("fib_0"):
                                                lows_found.append(float(last_v))
                                            elif k.startswith("fib_100"):
                                                highs_found.append(float(last_v))
                        lo_b = min(lows_found) if lows_found else None
                        hi_b = max(highs_found) if highs_found else None
                        return lo_b, hi_b
                    lo_b, hi_b = _scan_any_fib_bounds()
                    if price_trend is None:
                        # derive price from any candles if still missing
                        for td in (data_trend, data_chan):
                            if td and isinstance(td.candles, dict):
                                for _k in ("rates", "candles", "data", "bars"):
                                    seq = td.candles.get(_k)
                                    if isinstance(seq, list) and seq and isinstance(seq[-1], dict):
                                        _cl = seq[-1].get("close") or seq[-1].get("c")
                                        if isinstance(_cl, (int, float)):
                                            price_trend = float(_cl)
                                            break
                                if price_trend is not None:
                                    break
                    if price_trend is not None and lo_b is not None and hi_b is not None and hi_b > lo_b:
                        f_note, f_status = _eval_fibo_near(price_trend, lo_b, hi_b)
                        if f_note or f_status:
                            fibo_note = f_note
                            fibo_status = f_status
                    if DEBUG_MODE:
                        logger.debug(f"[FIBO_SCAN] price={price_trend} lo_b={lo_b} hi_b={hi_b} note={fibo_note} status={fibo_status}")
                if DEBUG_MODE:
                    logger.debug(f"[FIBO] price={price_trend} candidates={[(n, v) for n,v in fib_candidates]} note={fibo_note} status={fibo_status}")
            except Exception:
                pass

            # If no indicator fib conclusion, fallback to candle/Donchian computation
            if fibo_note is None and fibo_status is None:
                highs_f: List[float] = []
                lows_f: List[float] = []
                # Prefer trend candles; fallback to channel candles for broader availability
                src_candles = data_trend.candles or data_chan.candles
                if isinstance(src_candles, dict):
                    for key in ("rates", "candles", "data", "bars"):
                        seq_f = src_candles.get(key)
                        if isinstance(seq_f, list) and seq_f:
                            take = seq_f[-120:] if len(seq_f) > 120 else seq_f
                            for b in take:
                                if isinstance(b, dict):
                                    h = b.get("high") or b.get("h")
                                    l = b.get("low") or b.get("l")
                                    if h is not None:
                                        highs_f.append(ffloat(h))
                                    if l is not None:
                                        lows_f.append(ffloat(l))
                            break
                # Primary: use candle hi/lo
                if highs_f and lows_f and price_trend is not None:
                    hi = max(highs_f); lo = min(lows_f)
                    fibo_note, fibo_status = _eval_fibo_near(price_trend, lo, hi)
                else:
                    # Fallback: Donchian bounds from indicators using detected params or quick compute over candles
                    if donch_win_detected is not None:
                        d_lo = Extract._last_key_with_params(ind_chan or ind_trend or {}, f"Donchian_Lower_{donch_win_detected}", {})
                        d_up = Extract._last_key_with_params(ind_chan or ind_trend or {}, f"Donchian_Upper_{donch_win_detected}", {})
                    else:
                        d_lo = Extract._last_from_list(ind_chan or ind_trend or {}, "Donchian_Lower_20")
                        d_up = Extract._last_from_list(ind_chan or ind_trend or {}, "Donchian_Upper_20")
                    price_fb = price_trend
                    # If price_trend missing, try get from indicators
                    if price_fb is None:
                        price_fb = Extract._get_price_from_ind(ind_trend or ind_chan or {})
                    # If Donchian missing but we do have candles (any timeframe), compute quick 20-period
                    if (d_lo is None or d_up is None) and isinstance(src_candles, dict):
                        seq = None
                        for key in ("rates", "candles", "data", "bars"):
                            if isinstance(src_candles.get(key), list):
                                seq = src_candles.get(key)
                                break
                        if isinstance(seq, list) and len(seq) >= 20:
                            highs_20 = [ffloat(b.get("high") or b.get("h")) for b in seq[-20:] if isinstance(b, dict) and (b.get("high") or b.get("h")) is not None]
                            lows_20 = [ffloat(b.get("low") or b.get("l")) for b in seq[-20:] if isinstance(b, dict) and (b.get("low") or b.get("l")) is not None]
                            if highs_20 and lows_20:
                                d_up = max(highs_20)
                                d_lo = min(lows_20)
                    fibo_note, fibo_status = _eval_fibo_near(price_fb, d_lo, d_up)
        except Exception:
            fibo_note = None
            fibo_status = "không khả dụng"

        # Stochastic template extras
        stoch_extra_note = None
        try:
            if highs and lows and closes:
                k_ser, d_ser = _stoch_series(highs, lows, closes)
                if len(k_ser) >= 2 and len(d_ser) >= 2 and k_ser[-2] is not None and d_ser[-2] is not None and k_ser[-1] is not None and d_ser[-1] is not None:
                    # Cross note
                    crossed_up = k_ser[-2] <= d_ser[-2] and k_ser[-1] > d_ser[-1]
                    crossed_dn = k_ser[-2] >= d_ser[-2] and k_ser[-1] < d_ser[-1]
                    if crossed_up:
                        stoch_extra_note = "Stochastic: K vừa cắt lên D"
                    elif crossed_dn:
                        stoch_extra_note = "Stochastic: K vừa cắt xuống D"
                # Embedded state: last 3 bars K and D >80 or <20
                if len(k_ser) >= 3 and len(d_ser) >= 3:
                    last3k = [x for x in k_ser[-3:] if x is not None]
                    last3d = [x for x in d_ser[-3:] if x is not None]
                    if len(last3k) == 3 and len(last3d) == 3:
                        if min(last3k) > 80 and min(last3d) > 80:
                            stoch_extra_note = (stoch_extra_note + "; " if stoch_extra_note else "") + "Stochastic: embedded quá mua (>80 nhiều nến)"
                        elif max(last3k) < 20 and max(last3d) < 20:
                            stoch_extra_note = (stoch_extra_note + "; " if stoch_extra_note else "") + "Stochastic: embedded quá bán (<20 nhiều nến)"
        except Exception:
            stoch_extra_note = None

        # StochRSI bias at 0.5 with extreme cues
        stochrsi_extra_note = None
        try:
            if closes:
                rsi_ser = _rsi_series(closes) or []
                srsi_ser = _stochrsi_series_from_rsi(rsi_ser)
                if len(srsi_ser) >= 2 and srsi_ser[-2] is not None and srsi_ser[-1] is not None:
                    if srsi_ser[-2] <= 0.5 and srsi_ser[-1] > 0.5:
                        stochrsi_extra_note = "StochRSI vừa vượt 0.5 (thiên mua)"
                    elif srsi_ser[-2] >= 0.5 and srsi_ser[-1] < 0.5:
                        stochrsi_extra_note = "StochRSI vừa rơi dưới 0.5 (thiên bán)"
                    # Cross 0.8 or 0.2: short-term reversal cues
                    if srsi_ser[-2] is not None and srsi_ser[-1] is not None:
                        if srsi_ser[-2] >= 0.8 and srsi_ser[-1] < 0.8:
                            stochrsi_extra_note = (stochrsi_extra_note + "; " if stochrsi_extra_note else "") + "Cắt ↓ 0.8: đảo chiều ngắn hạn"
                        if srsi_ser[-2] <= 0.2 and srsi_ser[-1] > 0.2:
                            stochrsi_extra_note = (stochrsi_extra_note + "; " if stochrsi_extra_note else "") + "Cắt ↑ 0.2: đảo chiều ngắn hạn"
                if stochrsi is not None:
                    if stochrsi >= 0.8:
                        stochrsi_extra_note = (stochrsi_extra_note + "; " if stochrsi_extra_note else "") + "StochRSI vùng cực mạnh (>0.8)"
                    elif stochrsi <= 0.2:
                        stochrsi_extra_note = (stochrsi_extra_note + "; " if stochrsi_extra_note else "") + "StochRSI vùng cực yếu (<0.2)"
        except Exception:
            stochrsi_extra_note = None
        # MACD cross, histogram slope & phase
        macd_cross_note = None
        macd_hist_slope_note = None
        macd_phase_note = None
        try:
            if closes and len(closes) >= 35:
                macd_line_ser, macd_sig_ser, macd_hist_ser = _macd_series(closes)
                if macd_line_ser and macd_sig_ser and len(macd_line_ser) >= 2 and len(macd_sig_ser) >= 2:
                    a1, a2 = macd_line_ser[-2], macd_line_ser[-1]
                    b1, b2 = macd_sig_ser[-2], macd_sig_ser[-1]
                    if a1 is not None and a2 is not None and b1 is not None and b2 is not None:
                        if a1 <= b1 and a2 > b2:
                            macd_cross_note = "MACD cắt ↑ signal: mua"
                        elif a1 >= b1 and a2 < b2:
                            macd_cross_note = "MACD cắt ↓ signal: bán"
                if macd_hist_ser and len(macd_hist_ser) >= 2 and macd_hist_ser[-2] is not None and macd_hist_ser[-1] is not None:
                    h_prev, h_now = macd_hist_ser[-2], macd_hist_ser[-1]
                    if h_now > h_prev:
                        macd_hist_slope_note = "Histogram ↑: đà mạnh lên"
                    elif h_now < h_prev:
                        macd_hist_slope_note = "Histogram ↓: đà suy yếu"
                    if h_now > 0 and h_now > h_prev:
                        macd_phase_note = "tăng gia tốc"
                    elif h_now > 0 and h_now < h_prev:
                        macd_phase_note = "tăng chậm lại"
                    elif h_now < 0 and h_now < h_prev:
                        macd_phase_note = "giảm gia tốc"
                    elif h_now < 0 and h_now > h_prev:
                        macd_phase_note = "giảm suy yếu"
                    elif abs(h_now) < 1e-6:
                        macd_phase_note = "chuyển pha"
        except Exception:
            macd_cross_note = None
            macd_hist_slope_note = None
            macd_phase_note = None

        # TRIX slope, signal cross & phase
        trix_slope_note = None
        trix_signal_cross_note = None
        trix_phase_note = None
        try:
            if closes and len(closes) >= 60:
                # slope by comparing last two TRIX values
                t_prev = _trix(closes[:-1])
                t_now = _trix(closes)
                if t_prev is not None and t_now is not None:
                    if t_now > t_prev:
                        trix_slope_note = "Độ dốc TRIX ↑: động lượng tăng"
                    elif t_now < t_prev:
                        trix_slope_note = "Độ dốc TRIX ↓: động lượng giảm"
                # lightweight TRIX series for last ~80 points to check signal cross
                tail = closes[-120:] if len(closes) > 120 else closes[:]
                trix_series: List[Optional[float]] = []
                for i in range(30, len(tail)+1):
                    tv = _trix(tail[:i])
                    trix_series.append(tv)
                trix_vals = [x for x in trix_series if x is not None]
                if len(trix_vals) >= 10:
                    # simple EMA9 on TRIX values
                    def ema_simple(vals: List[float], p: int) -> List[float]:
                        k = 2.0 / (p + 1.0)
                        out: List[float] = []
                        ema_val = sum(vals[:p]) / p
                        out.append(ema_val)
                        for v in vals[p:]:
                            ema_val = v * k + ema_val * (1 - k)
                            out.append(ema_val)
                        return out
                    sig = ema_simple(trix_vals, 9)
                    if len(sig) >= 2:
                        t1, t2 = trix_vals[-2], trix_vals[-1]
                        s1, s2 = sig[-2], sig[-1]
                        if t1 <= s1 and t2 > s2:
                            trix_signal_cross_note = "TRIX cắt ↑ signal: mua"
                        elif t1 >= s1 and t2 < s2:
                            trix_signal_cross_note = "TRIX cắt ↓ signal: bán"
                    # phase detection (early vs mature based on zero-line crosses in recent window)
                    last = trix_vals[-1]
                    window = trix_vals[-6:]
                    if last is not None and window and all(w is not None for w in window):
                        crossed_up = any(w < 0 for w in window[:-1]) and last > 0
                        crossed_dn = any(w > 0 for w in window[:-1]) and last < 0
                        if crossed_up and last > 0:
                            trix_phase_note = "pha tăng sớm"
                        elif crossed_dn and last < 0:
                            trix_phase_note = "pha giảm sớm"
                        elif last > 0:
                            trix_phase_note = "pha tăng trưởng thành"
                        elif last < 0:
                            trix_phase_note = "pha giảm trưởng thành"
        except Exception:
            trix_slope_note = None
            trix_signal_cross_note = None
            trix_phase_note = None

        # Ichimoku cloud context (approximate)
        ichi_cloud_note = None
        ichi_cloud_extra = None
        try:
            if highs and lows and price_trend is not None and len(highs) >= 52 and len(lows) >= 52:
                tenkan_now = (max(highs[-9:]) + min(lows[-9:])) / 2.0
                kijun_now = (max(highs[-26:]) + min(lows[-26:])) / 2.0
                senkouA = (tenkan_now + kijun_now) / 2.0
                senkouB = (max(highs[-52:]) + min(lows[-52:])) / 2.0
                top = max(senkouA, senkouB); bot = min(senkouA, senkouB)
                if price_trend > top:
                    ichi_cloud_note = "Giá > mây: xu hướng tăng"
                elif price_trend < bot:
                    ichi_cloud_note = "Giá < mây: xu hướng giảm"
                else:
                    ichi_cloud_note = "Giá trong mây: sideway"
                thickness = abs(senkouA - senkouB) / max(price_trend, 1e-12)
                ichi_cloud_extra = f"Độ dày mây: {thickness:.4f}"
        except Exception:
            ichi_cloud_note = None
            ichi_cloud_extra = None

        # Vortex cross, gap & intensity tiers
        vortex_cross_note = None
        vortex_gap_note = None
        vortex_intensity_note = None
        try:
            if highs and lows and closes and isinstance(vortex, dict) and len(highs) >= 16:
                prev_vi = _vortex(highs[:-1], lows[:-1], closes[:-1])
                vp_prev = prev_vi.get('vi_plus') if isinstance(prev_vi, dict) else None
                vm_prev = prev_vi.get('vi_minus') if isinstance(prev_vi, dict) else None
                vp = vortex.get('vi_plus'); vm = vortex.get('vi_minus')
                if all(isinstance(x, (int,float)) for x in [vp_prev, vm_prev, vp, vm]):
                    if vp_prev <= vm_prev and vp > vm:
                        vortex_cross_note = "Vortex: VI+ cắt ↑ VI- (chuyển pha tăng)"
                    elif vp_prev >= vm_prev and vp < vm:
                        vortex_cross_note = "Vortex: VI+ cắt ↓ VI- (chuyển pha giảm)"
                    gapv = abs(vp - vm)
                    if gapv >= 0.2:
                        vortex_gap_note = "Khoảng cách VI lớn: xu hướng mạnh"
                    if gapv >= 0.60:
                        vortex_intensity_note = "cực mạnh"
                    elif gapv >= 0.40:
                        vortex_intensity_note = "mạnh"
                    elif gapv >= 0.25:
                        vortex_intensity_note = "vừa"
                    elif gapv >= 0.15:
                        vortex_intensity_note = "yếu"
        except Exception:
            vortex_cross_note = None
            vortex_gap_note = None
            vortex_intensity_note = None

        # Mass Index reversal watch (classic threshold ~27); note only when elevated
        massi_note = None
        try:
            if isinstance(massi, (int,float)) and massi > 27:
                massi_note = "Mass Index cao >27: chú ý khả năng đảo chiều khi co hẹp"
        except Exception:
            massi_note = None

        # KST phase (Know Sure Thing) qualitative multi-cycle momentum interpretation
        kst_phase_note = None
        try:
            if isinstance(kst, (int,float)):
                if kst_sup is True and kst > 0:
                    kst_phase_note = "pha tăng đa chu kỳ đồng thuận"
                elif kst_sup is False and kst < 0:
                    kst_phase_note = "pha giảm đa chu kỳ đồng thuận"
                elif abs(kst) < 0.1:
                    kst_phase_note = "pha chuyển tiếp / tích lũy"
        except Exception:
            kst_phase_note = None

        # Ultimate Oscillator breadth note
        ult_extra_note = None
        try:
            if isinstance(ult,(int,float)):
                if ult >= 70:
                    ult_extra_note = "mua mạnh trên 3 khung (Ultimate >70)"
                elif ult <= 30:
                    ult_extra_note = "bán mạnh đa khung (Ultimate <30)"
        except Exception:
            ult_extra_note = None

        # Ease of Movement / Force Index qualitative volume-pressure context
        eom_extra_note = None
        try:
            if isinstance(eom,(int,float)):
                if eom > 0:
                    eom_extra_note = "dòng tiền đẩy giá dễ dàng"
                elif eom < 0:
                    eom_extra_note = "áp lực giá gặp cản dòng tiền"
        except Exception:
            eom_extra_note = None

        force_extra_note = None
        force_intensity_note = None
        try:
            if isinstance(force,(int,float)):
                if force > 0:
                    force_extra_note = "lực mua chủ động"
                elif force < 0:
                    force_extra_note = "lực bán chủ động"
                if isinstance(atr,(int,float)) and isinstance(price_trend,(int,float)) and atr and price_trend:
                    try:
                        norm_force = force / (atr * price_trend)
                        anv = abs(norm_force)
                        if anv >= 5:
                            force_intensity_note = "cực mạnh"
                        elif anv >= 3:
                            force_intensity_note = "mạnh"
                        elif anv >= 1.5:
                            force_intensity_note = "vừa"
                        elif anv >= 0.8:
                            force_intensity_note = "yếu"
                    except Exception:
                        force_intensity_note = None
        except Exception:
            force_extra_note = None
            force_intensity_note = None

        # Keltner vs Bollinger squeeze (BB-in-KC)
        bb_in_kc_note = None
        try:
            if isinstance(bb, dict) and isinstance(kelt, dict) and price_trend is not None:
                bu = bb.get('upper'); bl = bb.get('lower')
                ku = kelt.get('upper'); kl = kelt.get('lower')
                if all(isinstance(x,(int,float)) for x in [bu, bl, ku, kl]) and bu > bl and ku > kl:
                    bwidth = (bu - bl) / max(price_trend,1e-12)
                    kwidth = (ku - kl) / max(price_trend,1e-12)
                    if bwidth <= 0.010:
                        # Localized Vietnamese text
                        bb_in_kc_note = "Độ rộng Bollinger < độ rộng Keltner: nén mạnh (BB-in-KC)"
                    elif bwidth >= 0.030:
                        # Use f-string to inject bwidth value correctly
                        bb_in_kc_note = f"Bollinger: Rộng (độ rộng {bwidth:.4f}) – biến động cao"
        except Exception:
            bb_in_kc_note = None

        # Channel percentile and breakout (Donchian)
        channel_percentile_note = None
        channel_breakout_note = None
        try:
            if isinstance(d_lo, (int,float)) and isinstance(d_up, (int,float)) and price_trend is not None and d_up > d_lo:
                pct = (price_trend - d_lo) / max(d_up - d_lo, 1e-12) * 100.0
                if pct >= 70:
                    channel_percentile_note = f"Channel percentile: {pct:.1f}% (gần biên trên)"
                elif pct <= 30:
                    channel_percentile_note = f"Channel percentile: {pct:.1f}% (gần biên dưới)"
                else:
                    channel_percentile_note = f"Channel percentile: {pct:.1f}%"
                if price_trend > d_up:
                    channel_breakout_note = "Breakout biên trên (Donchian)"
                elif price_trend < d_lo:
                    channel_breakout_note = "Breakout biên dưới (Donchian)"
        except Exception:
            channel_percentile_note = None
            channel_breakout_note = None

        # MFI cross 50, Chaikin cross 0
        mfi_cross_note = None
        chaikin_cross_note = None
        try:
            if highs and lows and closes and volumes and vol_available:
                if len(highs) >= 15 and len(lows) >= 15 and len(closes) >= 15 and len(volumes) >= 15:
                    prev_mfi = _mfi(highs[:-1], lows[:-1], closes[:-1], volumes[:-1])
                    if prev_mfi is not None and mfi is not None:
                        if prev_mfi <= 50 < mfi:
                            mfi_cross_note = "MFI cắt ↑ 50: đổi bias tăng"
                        elif prev_mfi >= 50 > mfi:
                            mfi_cross_note = "MFI cắt ↓ 50: đổi bias giảm"
                prev_chaikin = _chaikin_osc(highs[:-1], lows[:-1], closes[:-1], volumes[:-1]) if len(highs) == len(lows) == len(closes) == len(volumes) and len(closes) >= 2 else None
                if prev_chaikin is not None and chaikin is not None:
                    if prev_chaikin <= 0 < chaikin:
                        chaikin_cross_note = "Chaikin cắt ↑ 0: tích luỹ"
                    elif prev_chaikin >= 0 > chaikin:
                        chaikin_cross_note = "Chaikin cắt ↓ 0: phân phối"
        except Exception:
            mfi_cross_note = None
            chaikin_cross_note = None

        # Envelopes state vs price
        envelopes_state_note = None
        try:
            if isinstance(env, dict) and price_trend is not None:
                eu = env.get('upper'); el = env.get('lower')
                if isinstance(eu, (int,float)) and isinstance(el, (int,float)):
                    if price_trend > eu:
                        envelopes_state_note = "Đóng trên dải trên: đà tăng mạnh / breakout"
                    elif price_trend < el:
                        envelopes_state_note = "Đóng dưới dải dưới: đà giảm mạnh / breakout"
        except Exception:
            envelopes_state_note = None

        # CCI extras: extremes and +/-100 phase crosses
        cci_extra_note = None
        try:
            if highs and lows and closes:
                cci_ser = _cci_series(highs, lows, closes)
                if len(cci_ser) >= 2 and cci_ser[-2] is not None and cci_ser[-1] is not None:
                    if cci_ser[-2] <= -100 and cci_ser[-1] > -100:
                        cci_extra_note = "CCI vừa vượt -100 (giảm yếu dần)"
                    elif cci_ser[-2] >= 100 and cci_ser[-1] < 100:
                        cci_extra_note = "CCI vừa rơi dưới 100 (tăng yếu dần)"
                if cci is not None:
                    if cci >= 200:
                        cci_extra_note = (cci_extra_note + "; " if cci_extra_note else "") + "CCI cực mạnh (≥200)"
                    elif cci <= -200:
                        cci_extra_note = (cci_extra_note + "; " if cci_extra_note else "") + "CCI cực yếu (≤-200)"
        except Exception:
            cci_extra_note = None

        # Williams %R extras: exit extremes
        willr_extra_note = None
        try:
            if highs and lows and closes:
                wr_ser = _willr_series(highs, lows, closes)
                if len(wr_ser) >= 2 and wr_ser[-2] is not None and wr_ser[-1] is not None:
                    if wr_ser[-2] <= -80 and wr_ser[-1] > -80:
                        willr_extra_note = "%R thoát vùng quá bán (>-80)"
                    elif wr_ser[-2] >= -20 and wr_ser[-1] < -20:
                        willr_extra_note = "%R thoát vùng quá mua (<-20)"
        except Exception:
            willr_extra_note = None

        # ADX DI bias note
        adx_extra_note = None
        try:
            if highs and lows and closes:
                di_p, di_m = _di_last(highs, lows, closes)
                if di_p is not None and di_m is not None:
                    if di_p > di_m:
                        adx_extra_note = "+DI > -DI (thiên mua)"
                    elif di_p < di_m:
                        adx_extra_note = "+DI < -DI (thiên bán)"
        except Exception:
            adx_extra_note = None

        # EMA stacking and EMA20 slope
        ema_stack_note = None
        try:
            if ema20 is not None and ema50 is not None and ema200 is not None:
                if ema20 > ema50 > ema200:
                    ema_stack_note = "EMA20 > EMA50 > EMA200 (xếp chồng tăng)"
                elif ema20 < ema50 < ema200:
                    ema_stack_note = "EMA20 < EMA50 < EMA200 (xếp chồng giảm)"
        except Exception:
            ema_stack_note = None
        ema20_slope_note = None
        try:
            if ema20_prev is not None and ema20_now is not None:
                if ffloat(ema20_now) > ffloat(ema20_prev):
                    ema20_slope_note = "EMA20 đang dốc lên"
                elif ffloat(ema20_now) < ffloat(ema20_prev):
                    ema20_slope_note = "EMA20 đang dốc xuống"
        except Exception:
            ema20_slope_note = None

        # Ultimate Osc direction note (vs previous)
        ult_extra_note = None
        try:
            if highs and lows and closes:
                def ult_for_window(end: int) -> Optional[float]:
                    return _ultimate(highs[:end], lows[:end], closes[:end]) if end > 0 else None
                u_prev = ult_for_window(len(closes)-1)
                u_now = ult_for_window(len(closes))
                if u_prev is not None and u_now is not None:
                    if u_now > u_prev:
                        ult_extra_note = "Ultimate Osc tăng so với nến trước"
                    elif u_now < u_prev:
                        ult_extra_note = "Ultimate Osc giảm so với nến trước"
        except Exception:
            ult_extra_note = None

        # ROC acceleration / sign change note
        roc_extra_note = None
        try:
            if closes and len(closes) >= 15:
                def roc_for_window(arr: List[float]) -> Optional[float]:
                    return _roc(arr)
                r_prev = roc_for_window(closes[:-1])
                r_now = roc_for_window(closes)
                if r_prev is not None and r_now is not None:
                    if (r_prev <= 0 < r_now) or (r_prev >= 0 > r_now):
                        roc_extra_note = "ROC đổi dấu (động lượng đổi chiều)"
                    elif abs(r_now) > abs(r_prev):
                        roc_extra_note = "ROC tăng tốc (mạnh hơn)"
        except Exception:
            roc_extra_note = None

        # DPO phase change at 0
        dpo_extra_note = None
        try:
            if closes and len(closes) >= 30:
                def dpo_for(arr: List[float]) -> Optional[float]:
                    return _dpo(arr)
                d_prev = dpo_for(closes[:-1])
                d_now = dpo_for(closes)
                if d_prev is not None and d_now is not None:
                    crossed_up = d_prev <= 0 and d_now > 0
                    crossed_dn = d_prev >= 0 and d_now < 0
                    if crossed_up:
                        dpo_extra_note = "DPO vừa vượt 0 (pha tăng chu kỳ)"
                    elif crossed_dn:
                        dpo_extra_note = "DPO vừa rơi dưới 0 (pha giảm chu kỳ)"
        except Exception:
            dpo_extra_note = None

        # ---- Enriched descriptive text additions ----
        # Bollinger width classification
        bwidth = None
        boll_cat = None
        boll_history_note = None
        if isinstance(bb, dict) and price_trend:
            bu = bb.get('upper'); bl = bb.get('lower')
            if isinstance(bu,(int,float)) and isinstance(bl,(int,float)) and bu>bl:
                bwidth = (bu - bl) / max(price_trend,1e-12)
                if bwidth <= 0.010:
                    boll_cat = "nén mạnh (squeeze)"
                elif bwidth <= 0.020:
                    boll_cat = "độ rộng trung bình"
                else:
                    boll_cat = "rộng – biến động cao"
                # Lịch sử độ rộng Bollinger: percentile so với 100 độ rộng gần nhất
                try:
                    if closes and len(closes) >= 60:
                        widths: List[float] = []
                        # tái tính rolling width tương đối (upper-lower)/close
                        for i in range(40, len(closes)+1):
                            seg = closes[:i]
                            bb_tmp = _bbands(seg, 20, 2.0)
                            if bb_tmp and isinstance(bb_tmp.get('upper'), (int,float)) and isinstance(bb_tmp.get('lower'), (int,float)):
                                wv = (bb_tmp['upper'] - bb_tmp['lower']) / max(seg[-1],1e-12)
                                widths.append(wv)
                        if len(widths) >= 30 and bwidth is not None:
                            recent = widths[-120:] if len(widths) > 120 else widths
                            sorted_w = sorted(recent)
                            rank = sum(1 for x in sorted_w if x <= bwidth)
                            pct = rank / max(len(sorted_w),1) * 100.0
                            if pct <= 15:
                                boll_history_note = f"Độ rộng rất thấp (≤15th pct ~{pct:.1f}%) – tiềm năng bùng nổ"
                            elif pct <= 30:
                                boll_history_note = f"Độ rộng dưới trung bình (~{pct:.1f}%)"
                            elif pct >= 85:
                                boll_history_note = f"Độ rộng rất cao (≥85th pct ~{pct:.1f}%) – biến động cực đại"
                            elif pct >= 70:
                                boll_history_note = f"Độ rộng trên trung bình (~{pct:.1f}%)"
                            else:
                                boll_history_note = f"Độ rộng trung bình (~{pct:.1f}%)"
                except Exception:
                    boll_history_note = None
        # ATR volatility classification
        atr_vol_txt = None
        if isinstance(atr,(int,float)) and price_trend:
            vr = atr / max(price_trend,1e-12)
            if vr < 0.002:
                atr_vol_txt = "biến động thấp"
            elif vr < 0.005:
                atr_vol_txt = "biến động trung bình"
            else:
                atr_vol_txt = "biến động cao"
        # Donchian percentile inline (if earlier computed)
        donch_inline = None
        if isinstance(d_lo,(int,float)) and isinstance(d_up,(int,float)) and price_trend and d_up>d_lo:
            pct = (price_trend - d_lo)/max(d_up-d_lo,1e-12)*100.0
            if pct >= 70:
                donch_inline = f"vị trí {pct:.1f}% gần biên trên"
            elif pct <= 30:
                donch_inline = f"vị trí {pct:.1f}% gần biên dưới"
            else:
                donch_inline = f"vị trí {pct:.1f}% trong kênh"
        # EMA fallback note if missing
        def ema_fallback_note(val, period):
            # Treat None or NaN as missing; detect partial calc
            import math
            label_variants = [f"EMA{period}", f"SMA{period}", f"WMA{period}"]
            if val is None or (isinstance(val,(int,float)) and math.isnan(val)):
                if closes and len(closes) < period:
                    return f"(chưa đủ <{period} nến)"
                return "(không tính được)"
            # If we computed with insufficient candles mark temporary
            if any(v in incomplete_ma_periods for v in label_variants):
                return f"(tạm tính <{period} nến)"
            return ""
        ema20_note = ema_fallback_note(ema20,20)
        sma20_note = ema_fallback_note(sma20,20)
        # WMA20 fallback compute if absent
        if (wma20 is None or (isinstance(wma20,(int,float)) and wma20 != wma20)) and closes and len(closes)>=20:
            try:
                w = list(range(1,21))
                seg = closes[-20:]
                denom = sum(w)
                wma20 = sum(a*b for a,b in zip(seg, w))/denom
            except Exception:
                pass
        # Approximate WMA20 if still missing using EMA20 & SMA20 blend
        if (wma20 is None or (isinstance(wma20,(int,float)) and wma20 != wma20)) and (ema20 is not None) and (sma20 is not None):
            try:
                wma20 = (2*ema20 + sma20)/3
            except Exception:
                pass
        wma20_note = ema_fallback_note(wma20,20)
        # Note for TEMA20 if computed via fallback
        try:
            tema20_note = ema_fallback_note(tema20,20)
        except Exception:
            tema20_note = None
        ema50_note = ema_fallback_note(ema50,50)
        ema200_note = ema_fallback_note(ema200,200)
    # (debug removed)
        # Neutral display for missing momentum / volume indicators (avoid bare '-')
        def disp(val, neutral_txt="trung tính"):
            return fmt(val,4) if isinstance(val,(int,float)) else neutral_txt

        # ---- Ensure price_trend has a numeric fallback just before MA formatting ----
        if (price_trend is None or (isinstance(price_trend,(int,float)) and price_trend!=price_trend)) and closes:
            try:
                price_trend = closes[-1]
            except Exception:
                pass
        # Additional fallback: derive from indicator records if still missing
        if (price_trend is None or (isinstance(price_trend,(int,float)) and price_trend!=price_trend)) and isinstance(ind_trend, list):
            try:
                for rec in reversed(ind_trend):
                    if not isinstance(rec, dict):
                        continue
                    c = rec.get('close') or rec.get('Close')
                    if isinstance(c,(int,float)) and not (c!=c):
                        price_trend = c
                        break
            except Exception:
                pass
        # If decimals unknown but we now have a price, infer price_nd (max 8)
        if 'price_nd' in locals():
            try:
                if (not isinstance(price_nd,int) or price_nd < 0) and isinstance(price_trend,(int,float)):
                    sp = f"{price_trend}".split('.')
                    if len(sp) == 2:
                        price_nd = min(8, len(sp[1]))
            except Exception:
                pass

        # Cache to store finalized MA values computed inside ma_line
        ma_cache: Dict[str, float] = {}

        # Helper to compute khoảng cách (pips & %) giữa giá và MA
        def rel_dist(val):
            try:
                if price_trend is None or val is None:
                    return ""
                if isinstance(val,(int,float)) and (val!=val):
                    return ""
                diff = price_trend - val
                base = abs(val) if isinstance(val,(int,float)) else None
                pct = diff/val*100 if (val not in (0,None)) else 0
                # Asset-aware pip sizing
                sym_upper = str(sym).upper() if 'sym' in locals() else ''
                # default decimals inference
                dec = price_nd if isinstance(price_nd,int) and price_nd>=0 else 4
                if 'JPY' in sym_upper and dec < 3:
                    pip_unit = 0.01
                elif price_trend and price_trend > 1000:  # crypto / indices
                    pip_unit = 1.0
                elif dec >= 4:
                    pip_unit = 0.0001
                else:
                    pip_unit = 10 ** (-min(dec,4))
                if not pip_unit:
                    return ""
                pips = diff / pip_unit
                # Clamp near zero
                if abs(pips) < 0.05:
                    pips = 0.0
                sign = "+" if pips>0 else ""
                return f" ({sign}{pips:.1f} pips | {sign}{pct:.2f}% | Δ={diff:.2f})"
            except Exception:
                return ""
        # ---- MA line helper (dynamic by period & family) ----
        def ma_line(label, val, sup_flag, note_txt):
            # support tag & numeric details removed per request (only keep descriptive phrase)
            # Determine decimals for epsilon
            eps = 10 ** (-(price_nd+2)) if isinstance(price_nd,int) else 1e-6
            pos = '-'
            local_note_extra = ''
            # On-the-fly compute if missing and we have closes
            if (val is None or (isinstance(val,(int,float)) and val!=val)) and closes and not getattr(CFG,'STRICT_IND_ONLY',False):
                digits = ''.join(ch for ch in label if ch.isdigit())
                try:
                    period = int(digits) if digits else 0
                except Exception:
                    period = 0
                if period > 0:
                    take = closes[-min(len(closes), period):]
                    if take:
                        if label.startswith('WMA') and len(take) >= 2:
                            # proper weighted moving average fallback
                            weights = list(range(1, len(take)+1))
                            denom = sum(weights)
                            try:
                                val = sum(c*w for c,w in zip(take, weights)) / denom if denom else sum(take)/len(take)
                            except Exception:
                                val = sum(take)/len(take)
                        else:
                            # simple mean as fallback (approx for EMA when data insufficient)
                            val = sum(take)/len(take)
                        if len(closes) < period:
                            local_note_extra = f"(tạm tính <{period} nến)"
            # Store computed value
            if isinstance(val,(int,float)) and not (val!=val):
                ma_cache[label] = val
            # Robust numeric cast
            pt_valid = False
            vt_valid = False
            try:
                pt = float(price_trend)
                if not (pt!=pt):
                    pt_valid = True
            except Exception:
                pt_valid = False
            try:
                vv = float(val)
                if not (vv!=vv):
                    vt_valid = True
            except Exception:
                vt_valid = False
            if pt_valid and vt_valid:
                diff = pt - vv
                if abs(diff) <= eps:
                    pos = 'trong'
                else:
                    pos = 'trên' if diff>0 else 'dưới'
            # Force fallback classification if still '-' and we have numeric values
            if pos == '-' and pt_valid and vt_valid:
                pos = 'trên' if pt>vv else 'dưới' if pt<vv else 'trong'
            # distance (pips/%) removed
            # Dynamic context based on family and period
            fam = ''.join(ch for ch in label if ch.isalpha()).upper()
            try:
                digits = ''.join(ch for ch in label if ch.isdigit())
                period = int(digits) if digits else None
            except Exception:
                period = None
            def _ctx_for(f, p):
                if p is None:
                    return ''
                # General buckets
                if p <= 21:
                    bucket = 'ngắn hạn'
                elif p <= 55:
                    bucket = 'trung hạn'
                elif p <= 120:
                    bucket = 'trung-dài hạn'
                else:
                    bucket = 'dài hạn'
                if f == 'SMA':
                    return f"xác nhận đà {p} kỳ"
                elif f == 'WMA':
                    # Keep simple, emphasize weighting on short periods
                    if p <= 34:
                        return f"trọng số {bucket}"
                    return f"trọng số {p} kỳ"
                else:
                    return bucket
            ctx = _ctx_for(fam, period)
            if pos == 'trên':
                phrase = f"giá cao hơn {label} {ctx}".rstrip()
            elif pos == 'dưới':
                phrase = f"giá thấp hơn {label} {ctx}".rstrip()
            elif pos == 'trong':
                phrase = f"giá bám sát {label} {ctx}".rstrip()
            else:
                phrase = ''
            # Merge extra note if any
            merged_note = ' '.join(part for part in [note_txt or '', local_note_extra] if part).strip()
            # Append MA value and price for transparency if we have them
            # remove MA/price raw value display
            # Sentiment tag (avoid duplicate SELL wording if already implied externally)
            sent_tag = ''
            if pos == 'trên':
                sent_tag = '(tích cực)'
            elif pos == 'dưới':
                sent_tag = '(tiêu cực)'
            elif pos == 'trong':
                sent_tag = '(cân bằng)'
            # Build new compact numeric format: "giá =<price> cao hơn EMA50 <ma> trung hạn (tích cực)"
            # Dynamic rounding rules
            def _price_decimals(p):
                if p is None: return 4
                if p >= 100: return 2
                if p >= 1: return 4
                return 6
            if pt_valid and vt_valid:
                p_dec = _price_decimals(pt)
                m_dec = max(3, p_dec+1) if p_dec <=4 else 3  # keep MA one extra precision for mid-range
                try:
                    p_str = f"{pt:.{p_dec}f}".rstrip('0').rstrip('.')
                except Exception:
                    p_str = str(pt)
                try:
                    m_str = f"{vv:.{m_dec}f}".rstrip('0').rstrip('.')
                except Exception:
                    m_str = str(vv)
            else:
                p_str = '-' if not pt_valid else str(pt)
                m_str = '-' if not vt_valid else str(vv)
            if pos == 'trên':
                core = f"giá ={p_str} cao hơn {label} {m_str} {ctx} {sent_tag}".strip()
            elif pos == 'dưới':
                core = f"giá ={p_str} thấp hơn {label} {m_str} {ctx} {sent_tag}".strip()
            elif pos == 'trong':
                core = f"giá ={p_str} gần {label} {m_str} {ctx} {sent_tag}".strip()
            else:
                # Missing numeric; prefer concise 'chưa có'
                core = f"{label} chưa có"
            if merged_note:
                core = f"{core} {merged_note}".strip()
            # Normalize double spaces
            core = ' '.join(core.split())
            line = f"        • {label}: {core}".rstrip()
            # Collapse duplicate spaces (keep indentation and bullet)
            head = '        • '
            body = ' '.join(line[len(head):].split())
            return head + body

        # ---- Indicator lines assembly (clean) ----
        # Descriptor helpers for richer Vietnamese phrases
        def _desc_atr(a, price):
            try:
                if a is None or price is None: return None
                r = a/price if price else 0
                if r < 0.003: return "biến động thấp"
                if r < 0.007: return "biến động vừa"
                return "biến động cao"
            except Exception: return None
        def _desc_roc(v):
            if v is None: return None
            mag = abs(v)
            tier = "yếu" if mag < 0.1 else ("vừa" if mag < 0.5 else "mạnh")
            if v > 0: return f"động lượng tăng {tier}"
            if v < 0: return f"động lượng giảm {tier}"
            return "động lượng phẳng"
        def _desc_obv(v):
            if v is None: return None
            if v > 0: return "tích lũy dòng tiền"
            if v < 0: return "phân phối dòng tiền"
            return "dòng tiền trung tính"
        def _desc_chaikin(v):
            if v is None: return None
            if v > 0: return "dòng tiền vào"
            if v < 0: return "dòng tiền ra"
            return "dòng tiền cân bằng"
        def _desc_eom(v):
            if v is None: return None
            if v > 0: return "giá đi lên dễ dàng"
            if v < 0: return "giá đi xuống dễ dàng"
            return "chuyển động cân bằng"
        def _desc_force(v):
            if v is None: return None
            mag = abs(v)
            tier = "mạnh" if mag > 100000 else ("vừa" if mag > 10000 else "yếu")
            if v > 0: return f"lực mua {tier}"
            if v < 0: return f"lực bán {tier}"
            return "lực trung tính"
        def _desc_trix(v):
            if v is None: return None
            if v > 0: return "xung lực trung-dài hạn tăng"
            if v < 0: return "xung lực trung-dài hạn giảm"
            return "xung lực bằng phẳng"
        def _desc_dpo(v):
            if v is None: return None
            if v > 0: return "chu kỳ phía trên nền"
            if v < 0: return "chu kỳ phía dưới nền"
            return "chu kỳ cân bằng"
        def _desc_mass(v):
            if v is None: return None
            if v >= 27: return "cảnh báo đảo chiều"
            if v >= 26: return "tiệm cận cảnh báo"
            return "bình thường"
        def _desc_vortex(vp, vn):
            try:
                if vp is None or vn is None: return None
                diff = vp - vn
                if diff > 0.20: return "xu hướng lên mạnh"
                if diff > 0.08: return "ưu thế lên nhẹ"
                if diff < -0.20: return "xu hướng xuống mạnh"
                if diff < -0.08: return "ưu thế xuống nhẹ"
                return "cân bằng"
            except Exception: return None
        def _desc_ultimate(v):
            if v is None: return None
            if v >= 70: return "gần quá mua"
            if v >= 55: return "tăng ổn định"
            if v <= 30: return "gần quá bán"
            if v <= 45: return "giảm nhẹ"
            return "trung tính"
        def _desc_mfi(v):
            if v is None: return None
            if v >= 80: return "quá mua dòng tiền"
            if v >= 60: return "dòng tiền mạnh"
            if v <= 20: return "quá bán dòng tiền"
            if v <= 40: return "dòng tiền yếu"
            return "trung tính"
        def _desc_willr(v):
            if v is None: return None
            if v >= -20: return "gần quá mua"
            if v <= -80: return "gần quá bán"
            return "trung tính"
        def _desc_boll(cat, w):
            if w is None: return cat
            if w < 0.010: return "thu hẹp (nén)"
            if w > 0.030: return "mở rộng (biến động cao)"
            return cat
        # Only show EMA-specific arrow notes when the exact referenced EMA periods are selected.
        # Default behavior (no whitelist): allow notes.
        ema_notes_allowed = True
        ema_stack_allowed = True
        ema20_slope_allowed = True
        # Pair-wise cross/cut notes gating
        ema20_50_allowed = True
        ema50_200_allowed = True
        ema50_100_allowed = True
        ema100_200_allowed = True
        price_vs_ema50_allowed = True
        try:
            wl0 = None
            if isinstance(analysis, dict) and '_indicator_whitelist' in analysis:
                wl0 = analysis.get('_indicator_whitelist')
            elif isinstance(analysis, dict):
                wl0 = analysis.get('indicator_whitelist')
            if isinstance(wl0, (set, list)) and wl0:
                wl_norm0 = {str(x).lower() for x in wl0}
                # Any EMA present? (legacy broad gate)
                ema_notes_allowed = any(t.startswith('ema') for t in wl_norm0)
                # Precise gates per note
                has_ema20 = 'ema20' in wl_norm0
                has_ema50 = 'ema50' in wl_norm0
                has_ema100 = 'ema100' in wl_norm0
                has_ema200 = 'ema200' in wl_norm0
                ema_stack_allowed = has_ema20 and has_ema50 and has_ema200
                ema20_slope_allowed = has_ema20
                ema20_50_allowed = has_ema20 and has_ema50
                ema50_200_allowed = has_ema50 and has_ema200
                ema50_100_allowed = has_ema50 and has_ema100
                ema100_200_allowed = has_ema100 and has_ema200
                price_vs_ema50_allowed = has_ema50
        except Exception:
            # On any error, fall back to permissive behavior
            ema_notes_allowed = True
            ema_stack_allowed = True
            ema20_slope_allowed = True
            ema20_50_allowed = True
            ema50_200_allowed = True
            ema50_100_allowed = True
            ema100_200_allowed = True
            price_vs_ema50_allowed = True
        # Dynamic last-value helpers for primary bullets (non-MA families)
        def _last_any_prefix_primary(prefixes: list[str]):
            try:
                if isinstance(ind_trend, list) and ind_trend:
                    # Check last row first
                    last = ind_trend[-1]
                    if isinstance(last, dict):
                        for k, vv in last.items():
                            kl = str(k).lower()
                            if any(kl.startswith(p.lower()) for p in prefixes) and isinstance(vv,(int,float)) and not (vv!=vv):
                                return float(vv)
                    # Scan backward for first available
                    for row in reversed(ind_trend):
                        if isinstance(row, dict):
                            for k, vv in row.items():
                                kl = str(k).lower()
                                if any(kl.startswith(p.lower()) for p in prefixes) and isinstance(vv,(int,float)) and not (vv!=vv):
                                    return float(vv)
            except Exception:
                pass
            return None

        # Prefer dynamic values when present; fallback to computed ones
        rsi_val_disp = _last_any_prefix_primary(['rsi']) or rsi
        adx_val_disp = _last_any_prefix_primary(['adx']) or adx
        stochrsi_val_disp = _last_any_prefix_primary(['stochrsi']) or stochrsi
        stoch_k_disp = _last_any_prefix_primary(['stochk','stoch_k','%k']) or stoch_k
        stoch_d_disp = _last_any_prefix_primary(['stochd','stoch_d','%d']) or stoch_d
        atr_val_disp = _last_any_prefix_primary(['atr']) or atr
        willr_val_disp = _last_any_prefix_primary(['williamsr','willr']) or willr
        obv_val_disp = _last_any_prefix_primary(['obv']) or obv
        mfi_val_disp = _last_any_prefix_primary(['mfi']) or mfi
        force_val_disp = _last_any_prefix_primary(['forceindex','force']) or force
        kst_val_disp = _last_any_prefix_primary(['kst']) or kst
        ult_val_disp = _last_any_prefix_primary(['ultimate']) or ult
        # PSAR directional dynamic
        psar_dir_disp = psar_dir
        try:
            _ps = _last_any_prefix_primary(['psar'])
            if isinstance(_ps,(int,float)):
                psar_dir_disp = _ps
        except Exception:
            pass

        # ---- Dynamic label detection (derive parameters from exporter column names) ----
        # Parameters already detected above

        try:
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Detected indicator params: %s", {k: v for k, v in _params.items() if v is not None})
        except Exception:
            pass
        def _wrap(name: str, *vals: Optional[int]) -> str:
            got = [v for v in vals if v is not None]
            return f"{name}({','.join(str(v) for v in got)})" if got else name

        rsi_label = _wrap("RSI", _params['rsi'])
        macd_label = _wrap("MACD", _params['macd_fast'], _params['macd_slow'], _params['macd_sig'])
        adx_label = _wrap("ADX", _params['adx'])
        stoch_label = _wrap("Stochastic", _params['stoch_period'], _params['stoch_smooth'])
        stochrsi_label = _wrap("StochRSI", _params['stochrsi'])
        bb_label = _wrap("Bollinger", _params['bb_win'], _params['bb_dev'])
        atr_label = _wrap("ATR", _params['atr'])
        mfi_label = _wrap("MFI", _params['mfi'])
        cci_label = _wrap("CCI", _params['cci'])
        willr_label = _wrap("Williams %R", _params['willr'])
        roc_label = _wrap("ROC", _params['roc'])
        chaikin_label = _wrap("Chaikin Money Flow", _params['chaikin'])
        eom_label = _wrap("EOM", _params['eom'])
        force_label = _wrap("Force Index", _params['force'])
        donch_label = _wrap("Donchian", _params['donch_win'])
        trix_label = _wrap("TRIX", _params['trix'])
        dpo_label = _wrap("DPO", _params['dpo'])
        kelt_label = _wrap("Keltner", _params['kelt_win'])
        mass_label = _wrap("Mass Index", _params['mass_fast'], _params['mass_slow'])
        vortex_label = _wrap("Vortex", _params['vi_period'])
        ichi_label = _wrap("Ichimoku", _params.get('ichi_tenkan'), _params.get('ichi_kijun'), _params.get('ichi_senkou'))

        # Generate ADX strength description
        adx_for_desc = adx_val_disp if adx_val_disp is not None else adx
        if adx_for_desc is None:
            adx_txt = None
        else:
            if adx_for_desc < 15:
                adx_txt = "không xu hướng (<15)"
            elif adx_for_desc < 20:
                adx_txt = "xu hướng yếu (15-20)"
            elif adx_for_desc < 25:
                adx_txt = "xu hướng đang hình thành (20-25)"
            elif adx_for_desc < 35:
                adx_txt = "xu hướng mạnh (25-35)"
            else:
                adx_txt = "xu hướng rất mạnh (>35)"

        # Generate all indicator descriptions using displayed values for consistency
        # Stochastic descriptions
        stoch_k_for_desc = stoch_k_disp if stoch_k_disp is not None else stoch_k
        stoch_d_for_desc = stoch_d_disp if stoch_d_disp is not None else stoch_d
        stoch_txt = None
        if stoch_k_for_desc is not None and stoch_d_for_desc is not None:
            if stoch_k_for_desc >= 80 and stoch_d_for_desc >= 80:
                stoch_txt = "quá mua (>80)"
            elif stoch_k_for_desc <= 20 and stoch_d_for_desc <= 20:
                stoch_txt = "quá bán (<20)"
            elif stoch_k_for_desc >= 70:
                stoch_txt = "gần quá mua (>70)"
            elif stoch_k_for_desc <= 30:
                stoch_txt = "gần quá bán (<30)"
            elif stoch_k_for_desc > stoch_d_for_desc:
                stoch_txt = "tăng ngắn hạn (K>D)"
            elif stoch_k_for_desc < stoch_d_for_desc:
                stoch_txt = "giảm ngắn hạn (K<D)"
            else:
                stoch_txt = "trung tính"

        # StochRSI descriptions  
        stochrsi_for_desc = stochrsi_val_disp if stochrsi_val_disp is not None else stochrsi
        stochrsi_txt = None if stochrsi_for_desc is None else (
            "quá mua mạnh (1.0)" if stochrsi_for_desc >= 1.0 else
            "quá mua (>0.8)" if stochrsi_for_desc >= 0.8 else 
            "quá bán mạnh (0.0)" if stochrsi_for_desc <= 0.0 else
            "quá bán (<0.2)" if stochrsi_for_desc <= 0.2 else 
            "trung tính (0.2-0.8)"
        )
        
        # MFI descriptions
        mfi_for_desc = mfi_val_disp if mfi_val_disp is not None else mfi
        mfi_txt = None if mfi_for_desc is None else (
            "quá mua mạnh (>90)" if mfi_for_desc >= 90 else
            "quá mua (>80)" if mfi_for_desc >= 80 else 
            "thiên mua (>60)" if mfi_for_desc >= 60 else
            "quá bán mạnh (<10)" if mfi_for_desc <= 10 else
            "quá bán (<20)" if mfi_for_desc <= 20 else 
            "thiên bán (<40)" if mfi_for_desc <= 40 else
            "trung tính (20-80)"
        )

        # Williams %R descriptions
        willr_for_desc = willr_val_disp if willr_val_disp is not None else willr
        willr_txt = None if willr_for_desc is None else (
            "quá mua mạnh (>-10)" if willr_for_desc > -10 else
            "quá mua (>-20)" if willr_for_desc > -20 else 
            "thiên mua (>-40)" if willr_for_desc > -40 else
            "quá bán mạnh (<-90)" if willr_for_desc < -90 else
            "quá bán (<-80)" if willr_for_desc < -80 else 
            "thiên bán (<-60)" if willr_for_desc < -60 else
            "trung tính (-20 to -80)"
        )
        
        # CCI descriptions  
        cci_for_desc = _last_any_prefix_primary(['cci']) or cci
        cci_txt = None if cci_for_desc is None else (
            "quá mua mạnh (>200)" if cci_for_desc > 200 else
            "quá mua (>100)" if cci_for_desc > 100 else 
            "thiên mua (>50)" if cci_for_desc > 50 else
            "quá bán mạnh (<-200)" if cci_for_desc < -200 else
            "quá bán (<-100)" if cci_for_desc < -100 else 
            "thiên bán (<-50)" if cci_for_desc < -50 else
            "trung tính (-100 to 100)"
        )
        
        # ROC descriptions
        roc_for_desc = _last_any_prefix_primary(['roc']) or roc
        roc_txt = None if roc_for_desc is None else (
            "tăng mạnh (>5%)" if roc_for_desc > 5 else
            "tăng vừa (1-5%)" if roc_for_desc > 1 else 
            "tăng nhẹ (0-1%)" if roc_for_desc > 0 else
            "giảm mạnh (<-5%)" if roc_for_desc < -5 else
            "giảm vừa (-1 to -5%)" if roc_for_desc < -1 else
            "giảm nhẹ (-1 to 0%)" if roc_for_desc < 0 else
            "không đổi"
        )
        
        # TRIX descriptions
        trix_for_desc = _last_any_prefix_primary(['trix']) or trix
        trix_txt = None if trix_for_desc is None else (
            "tăng mạnh (>0.001)" if trix_for_desc > 0.001 else
            "tăng" if trix_for_desc > 0 else 
            "giảm mạnh (<-0.001)" if trix_for_desc < -0.001 else
            "giảm" if trix_for_desc < 0 else 
            "trung tính"
        )
        
        # DPO descriptions
        dpo_for_desc = _last_any_prefix_primary(['dpo']) or dpo
        dpo_txt = None if dpo_for_desc is None else (
            "tăng chu kỳ mạnh" if dpo_for_desc > 10 else
            "tăng chu kỳ" if dpo_for_desc > 0 else 
            "giảm chu kỳ mạnh" if dpo_for_desc < -10 else
            "giảm chu kỳ" if dpo_for_desc < 0 else 
            "trung tính chu kỳ"
        )
        
        # EOM descriptions
        eom_for_desc = _last_any_prefix_primary(['eom']) or eom
        eom_txt = None if eom_for_desc is None else (
            "dòng tiền tăng mạnh" if eom_for_desc > 100000 else
            "dòng tiền tăng" if eom_for_desc > 0 else 
            "dòng tiền giảm mạnh" if eom_for_desc < -100000 else
            "dòng tiền giảm" if eom_for_desc < 0 else 
            "dòng tiền trung tính"
        )
        
        # Force Index descriptions
        force_for_desc = force_val_disp if force_val_disp is not None else force
        force_txt = None if force_for_desc is None else (
            "lực mua mạnh" if force_for_desc > 50000 else
            "lực mua" if force_for_desc > 0 else 
            "lực bán mạnh" if force_for_desc < -50000 else
            "lực bán" if force_for_desc < 0 else 
            "lực trung tính"
        )
        
        # ATR descriptions - volatility levels
        atr_for_desc = atr_val_disp if atr_val_disp is not None else atr
        atr_txt = None if atr_for_desc is None or price_trend is None else (
            "biến động rất cao" if (atr_for_desc / price_trend) > 0.02 else
            "biến động cao" if (atr_for_desc / price_trend) > 0.015 else
            "biến động vừa" if (atr_for_desc / price_trend) > 0.01 else
            "biến động thấp" if (atr_for_desc / price_trend) > 0.005 else
            "biến động rất thấp"
        )
        
        # Simple text descriptions for other indicators
        obv_txt = None if obv is None else ("tăng" if obv > 0 else ("giảm" if obv < 0 else "trung tính"))
        obv_slope_note = None  # Initialize obv_slope_note to prevent undefined variable error
        
        # Chaikin Money Flow descriptions - detailed analysis
        chaikin_for_desc = chaikin
        chaikin_txt = None if chaikin_for_desc is None else (
            "dòng tiền mạnh vào (mua mạnh)" if chaikin_for_desc > 0.1 else
            "dòng tiền tích cực (mua vừa)" if chaikin_for_desc > 0.05 else 
            "dòng tiền nhẹ vào (mua nhẹ)" if chaikin_for_desc > 0 else
            "dòng tiền mạnh ra (bán mạnh)" if chaikin_for_desc < -0.1 else
            "dòng tiền tiêu cực (bán vừa)" if chaikin_for_desc < -0.05 else
            "dòng tiền nhẹ ra (bán nhẹ)" if chaikin_for_desc < 0 else
            "dòng tiền trung tính"
        )
        
        ult_txt = None if ult is None else ("tăng" if ult > 50 else ("giảm" if ult < 50 else "trung tính"))
        kst_txt = None if kst is None else ("tăng" if kst > 0 else ("giảm" if kst < 0 else "trung tính"))

        # Mass Index descriptions - volatility and reversal analysis  
        mass_for_desc = massi
        mass_txt = None if mass_for_desc is None else (
            "cực cao - cảnh báo đảo chiều" if mass_for_desc > 27 else
            "tiệm cận cảnh báo" if mass_for_desc > 25 else
            "cao - biến động tăng" if mass_for_desc > 22 else
            "vừa - biến động bình thường" if mass_for_desc > 18 else
            "thấp - biến động giảm" if mass_for_desc <= 18 else
            "bình thường"
        )

        # Now assign _desc variables using our enhanced _txt descriptions
        atr_desc = None  # Use atr_txt instead
        roc_desc = roc_txt  # Use our enhanced ROC description
        cci_desc = cci_txt  # Use our enhanced CCI description
        obv_desc = _desc_obv(obv)
        chaikin_desc = _desc_chaikin(chaikin)
        eom_desc = eom_txt  # Use our enhanced EOM description  
        force_desc = force_txt  # Use our enhanced Force description
        trix_desc = trix_txt  # Use our enhanced TRIX description
        dpo_desc = dpo_txt  # Use our enhanced DPO description
        mass_desc = _desc_mass(massi)
        vortex_desc = _desc_vortex(vortex.get('vi_plus'), vortex.get('vi_minus')) if isinstance(vortex,dict) else None
        ult_desc = _desc_ultimate(ult)
        mfi_desc = mfi_txt  # Use our enhanced MFI description
        will_desc = willr_txt  # Use our enhanced Williams %R description
        boll_desc = _desc_boll(boll_cat, bwidth)

        ind_lines = [
            # RSI with state only (no threshold hint)
            (f"        • {rsi_label}: {fmt(rsi_val_disp,1)}" + (f" – {rsi_state}" if rsi_state else "") + f" {support_tag(rsi_sup)}") if (rsi_val_disp is not None) else f"        • {rsi_label}: - (trung tính)",
            (f"          → RSI: {rsi_div_note}" if rsi_div_note else None),
            (f"          → {rsi_extra_note}" if rsi_extra_note else None),
            # MACD with histogram
            f"        • {macd_label}: {'TĂNG' if (macd_h or 0)>0 else 'GIẢM' if (macd_h or 0)<0 else 'TRUNG TÍNH'} (hist {fmt(macd_h,4)}) {support_tag(macd_sup)}",
            (f"          → MACD: {macd_div_note}" if macd_div_note else None),
            (f"          → {macd_cross_note}" if macd_cross_note else None),
            (f"          → {macd_hist_slope_note}" if macd_hist_slope_note else None),
            (f"          → Pha MACD: {macd_phase_note}" if macd_phase_note else None),
            # ADX
            f"        • {adx_label}: {fmt(adx_val_disp,1)}" + (f" – {adx_txt}" if adx_txt else ""),
            (f"          → {adx_extra_note}" if adx_extra_note else None),
            # Stochastic without threshold hint
            f"        • {stoch_label}: K {fmt(stoch_k_disp,1)} / D {fmt(stoch_d_disp,1)}" + (f" – {stoch_txt}" if stoch_txt else ""),
            (f"          → {stoch_extra_note}" if stoch_extra_note else None),
            # StochRSI without threshold hint
            f"        • {stochrsi_label}: {fmt(stochrsi_val_disp,2)}" + (f" – {stochrsi_txt}" if stochrsi_txt else (" – chưa có tín hiệu")),
            (f"          → {stochrsi_extra_note}" if stochrsi_extra_note else None),
            # Bands and ATR - Enhanced descriptions
            f"        • {bb_label}: " + (
                'trong dải' if bb and bb.get('upper') and bb.get('lower') and price_trend and bb['lower'] <= price_trend <= bb['upper'] else 
                f"trên dải (giá {fmt(price_trend,4)} > {fmt(bb.get('upper'),4)})" if bb and bb.get('upper') and price_trend and price_trend > bb['upper'] else
                f"dưới dải (giá {fmt(price_trend,4)} < {fmt(bb.get('lower'),4)})" if bb and bb.get('lower') and price_trend and price_trend < bb['lower'] else
                f"giá trung tâm (SMA {fmt(bb.get('middle'),4)})" if bb and bb.get('middle') and price_trend else
                f"width {fmt(bwidth,4)}" if bwidth is not None and bwidth > 0 else
                "không đủ dữ liệu"
            ) + (f" – {boll_desc} (width {bwidth:.4f})" if boll_desc and bwidth is not None else (f" – (width {bwidth:.4f})" if bwidth is not None else "")),
            (f"          → {boll_history_note}" if boll_history_note else None),
            f"        • {atr_label}: {fmt(atr_val_disp,4)}" + (f" – {atr_txt}" if atr_txt else ""),
            # MFI without threshold hint (still show volume note)
            f"        • {mfi_label}: {fmt(mfi_val_disp,1)}" + (f" – {mfi_desc}" if mfi_desc else (f" – {mfi_txt}" if mfi_txt else (" – thiếu dữ liệu khối lượng" if not (mfi_val_disp is not None) and not vol_available else ""))),
            # PSAR
            f"        • PSAR: " + (
                'tín hiệu tăng' if isinstance(psar_dir_disp,(int,float)) and psar_dir_disp>0 else (
                'tín hiệu giảm' if isinstance(psar_dir_disp,(int,float)) and psar_dir_disp<0 else (
                'trung tính' if isinstance(psar_dir_disp,(int,float)) and psar_dir_disp==0 else 'chưa xác định'))
            ) + f" {support_tag(psar_sup)}",
            # EMA/SMA and cross notes
            ma_line("EMA20", ema20, ema_sup, ema20_note),
            (f"          → {ema20_50_note}" if (ema20_50_note and ema20_50_allowed) else None),
            (f"          → {ema50_200_note}" if (ema50_200_note and ema50_200_allowed) else None),
            (f"          → {ema50_100_note}" if (ema50_100_note and ema50_100_allowed) else None),
            (f"          → {ema100_200_note}" if (ema100_200_note and ema100_200_allowed) else None),
            (f"          → {ema20_slope_note}" if (ema20_slope_note and ema20_slope_allowed) else None),
            ma_line("SMA20", sma20, sma_sup, sma20_note),
            (f"          → {sma20_50_note}" if sma20_50_note else None),
            (ma_line("EMA50", ema50, ema50_sup, ema50_note) if True else None),
            (f"          → {price_vs_ema50_note}" if (price_vs_ema50_note and price_vs_ema50_allowed) else None),
            (ma_line("EMA200", ema200, ema200_sup, ema200_note) if True else None),
            (lambda _v=ema100: ma_line("EMA100", _v,
                        True if (price_trend and _v and price_trend>_v) else False if (price_trend and _v and price_trend<_v) else None,
                        "(không tính được)" if (_v is None or (isinstance(_v,(int,float)) and (_v!=_v))) else ""))(),
            (f"          → {ema_stack_note}" if (ema_stack_note and ema_stack_allowed) else None),
            ma_line("WMA20", wma20, wma_sup, wma20_note),
            # Include TEMA20 primary bullet so it can be filtered/enriched like others
            ma_line("TEMA20", tema20, tema_sup, tema20_note),
            # Ichimoku and Keltner - Enhanced descriptions
            f"        • {ichi_label}: " + (
                'Tenkan > Kijun (xu hướng tăng mạnh)' if (ffloat(ich_tenkan,0)>ffloat(ich_kijun,0)) else 
                'Tenkan <= Kijun (xu hướng giảm hoặc chuyển đổi)' if (ich_tenkan is not None and ich_kijun is not None) else 
                f"Tenkan {fmt(ich_tenkan,4)} (đường chuyển đổi)" if ich_tenkan is not None else
                f"Kijun {fmt(ich_kijun,4)} (đường cơ sở)" if ich_kijun is not None else
                f"Senkou A {fmt(ich_senkou_a,4)} (mây dẫn đầu)" if ich_senkou_a is not None else
                f"Senkou B {fmt(ich_senkou_b,4)} (mây dẫn đầu)" if ich_senkou_b is not None else
                f"Chikou {fmt(ich_chikou,4)} (đường trễ)" if ich_chikou is not None else
                'chờ dữ liệu đầy đủ'
            ) + f" {support_tag(ichi_sup)}",
            (f"          → {ichi_cloud_note}" if ichi_cloud_note else None),
            (f"          → {ichi_cloud_extra}" if ichi_cloud_extra else None),
            f"        • {kelt_label}: " + (
                'trong kênh' if kelt and kelt.get('upper') and kelt.get('lower') and price_trend and kelt['lower'] <= price_trend <= kelt['upper'] else 
                f"trên kênh (giá {fmt(price_trend,4)} > {fmt(kelt.get('upper'),4)})" if kelt and kelt.get('upper') and price_trend and price_trend > kelt['upper'] else
                f"dưới kênh (giá {fmt(price_trend,4)} < {fmt(kelt.get('lower'),4)})" if kelt and kelt.get('lower') and price_trend and price_trend < kelt['lower'] else
                f"upper={fmt(kelt.get('upper'),4)}, lower={fmt(kelt.get('lower'),4)}" if kelt and (kelt.get('upper') or kelt.get('lower')) else
                "không đủ dữ liệu"
            ),
            # CCI and Williams without threshold hints
            f"        • {cci_label}: {fmt(_last_any_prefix_primary(['cci']) or cci,1)}" + (f" – {cci_txt}" if cci_txt else ""),
            (f"          → {cci_extra_note}" if cci_extra_note else None),
            f"        • {willr_label}: {fmt(willr_val_disp,1)}" + (f" – {will_desc}" if will_desc else (f" – {willr_txt}" if willr_txt else "")),
            (f"          → {willr_extra_note}" if willr_extra_note else None),
            # Momentum and volume family
            f"        • {roc_label}: {fmt(_last_any_prefix_primary(['roc']) or roc,2)}" + (f" – {roc_desc}" if roc_desc else ""),
            f"        • OBV: {fmt(obv_val_disp,0)}" + (f" – {obv_desc}" if obv_desc else (f" – {obv_txt}" if obv_txt else "")),
            (f"          → {obv_slope_note}" if 'obv_slope_note' in locals() and obv_slope_note else None),
            f"        • {eom_label}: " + (f"{fmt(eom,3)}" if eom is not None else "chưa tính được") + (f" – {eom_txt}" if eom_txt and eom is not None else "") + (f" ({eom_extra_note})" if eom_extra_note else ""),
            f"        • {force_label}: {fmt(force_val_disp,3)}" + (f" – {force_desc}" if force_desc else (f" – {force_txt}" if force_txt else "")),
            (f"          → {force_extra_note}" if force_extra_note else None),
            (f"          → Cường độ: {force_intensity_note}" if force_intensity_note else None),
            # Channels and cycles - Enhanced descriptions
            f"        • {donch_label}: " + (
                'trong kênh' if (isinstance(d_lo,(int,float)) and isinstance(d_up,(int,float)) and d_lo<= (price_trend or d_lo) <= d_up) else 
                f"trên kênh (giá {fmt(price_trend,4)} > {fmt(d_up,4)})" if (d_up is not None and price_trend and price_trend > d_up) else
                f"dưới kênh (giá {fmt(price_trend,4)} < {fmt(d_lo,4)})" if (d_lo is not None and price_trend and price_trend < d_lo) else
                f'high={fmt(d_up,4)}, low={fmt(d_lo,4)}' if (d_lo is not None and d_up is not None) else 
                f'high={fmt(d_up,4)}' if d_up is not None else
                f'low={fmt(d_lo,4)}' if d_lo is not None else
                'không đủ dữ liệu'
            ) + (f" – {donch_inline}" if donch_inline else ""),
            f"        • {trix_label}: " + (f"{fmt(trix,4)}" if trix is not None else "chưa tính được") + (f" – {trix_txt}" if trix_txt and trix is not None else ""),
            (f"          → {trix_slope_note}" if trix_slope_note else None),
            (f"          → {trix_signal_cross_note}" if trix_signal_cross_note else None),
            (f"          → Pha TRIX: {trix_phase_note}" if trix_phase_note else None),
            f"        • {dpo_label}: " + (f"{fmt(dpo,3)}" if dpo is not None else "chưa tính được") + (f" – {dpo_txt}" if dpo_txt and dpo is not None else ""),
            (f"          → {dpo_extra_note}" if dpo_extra_note else None),
            f"        • {mass_label}: {fmt(massi,2)}" + (f" – {mass_txt}" if mass_txt else (f" – {massi_note}" if massi_note else "")),
            f"        • {vortex_label}: VI+ {fmt(vortex.get('vi_plus'),2)} / VI- {fmt(vortex.get('vi_minus'),2)} {support_tag(vortex_sup)}" + (f" – {vortex_desc}" if vortex_desc else (f" – {vortex_gap_note}" if vortex_gap_note else "")) + (f" ({vortex_intensity_note})" if vortex_intensity_note else ""),
            (f"          → {vortex_cross_note}" if vortex_cross_note else None),
            (f"          → {vortex_gap_note}" if vortex_gap_note else None),
            f"        • KST: {('TĂNG' if (kst or 0)>0 else 'GIẢM' if (kst or 0)<0 else 'TRUNG TÍNH')} (diff {fmt(kst,2)}) {support_tag(kst_sup)}" + (f" – {kst_txt}" if kst_txt else "") + (f" ({kst_phase_note})" if kst_phase_note else ""),
            f"        • Ultimate Osc: {fmt(ult,1)} {support_tag(ult_sup)}" + (f" – {ult_desc}" if ult_desc else (f" – {ult_txt}" if ult_txt else "")) + (f" ({ult_extra_note})" if ult_extra_note else ""),
            f"        • {(_wrap('Envelopes', _params.get('env_win'), _params.get('env_dev')) if 'env_win' in _params else 'Envelopes(20,2.0)')}: " + (
                'trong dải (giá dao động bình thường)' if env and env.get('upper') and env.get('lower') and price_trend and env['lower'] <= price_trend <= env['upper'] else 
                f"trên dải - đột phá tăng (giá {fmt(price_trend,4)} > {fmt(env.get('upper'),4)})" if env and env.get('upper') and price_trend and price_trend > env['upper'] else
                f"dưới dải - đột phá giảm (giá {fmt(price_trend,4)} < {fmt(env.get('lower'),4)})" if env and env.get('lower') and price_trend and price_trend < env['lower'] else
                f"upper={fmt(env.get('upper'),4)}, lower={fmt(env.get('lower'),4)}" if env and (env.get('upper') or env.get('lower')) else
                'không đủ dữ liệu'
            ),
            (f"          → {envelopes_state_note}" if envelopes_state_note else None),
            (f"          → {bb_in_kc_note}" if bb_in_kc_note else None),
            (f"          → {roc_extra_note}" if roc_extra_note else None),
            # Composite momentum line
            (f"        • Momentum/Cycle: MACD[{macd_phase_note or '-'}]; TRIX[{trix_phase_note or '-'}]; KST[{kst_phase_note or '-'}]; DPO[{dpo_txt or '-'}]"),
            # Always show Fibonacci line
            (f"        • Fibonacci: {fibo_note}" if fibo_note else f"        • Fibonacci: {fibo_status or 'không gần mức quan trọng'}"),
            (f"        • Donchian %ile: {channel_percentile_note}" if channel_percentile_note else None),
            (f"          → {channel_breakout_note}" if channel_breakout_note else None),
            (f"          → {mfi_cross_note}" if mfi_cross_note else None),
            (f"          → {chaikin_cross_note}" if chaikin_cross_note else None),
        ]

        # Always keep indicator lines (user requirement: luôn hiển thị).
        # Remove verbose missing-data Vietnamese phrases to avoid clutter.
        cleaned: List[str] = []
        for ln in ind_lines:
            if not ln:
                continue
            # Strip phrases indicating missing data; leave the line so indicator still shows.
            ln = ln.replace(" – thiếu dữ liệu khối lượng", "").replace(" – không khả dụng", "")
            cleaned.append(ln)
        ind_lines = cleaned

        # Lightweight indicator whitelist filtering (only top-level bullet lines) if whitelist provided.
        try:
            wl = None
            if isinstance(analysis, dict) and '_indicator_whitelist' in analysis:
                wl = analysis.get('_indicator_whitelist')
            elif isinstance(analysis, dict):
                wl = analysis.get('indicator_whitelist')  # fallback key if ever set elsewhere
            if isinstance(wl, (set, list)) and wl:
                wl_norm = {str(x).lower() for x in wl}
                try:
                    if logger and logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Whitelist (normalized): %s", ", ".join(sorted(wl_norm)))
                except Exception:
                    pass
                # Synonym bridging (only exact user intent; no synthetic EMA group expansion anymore)
                # Removed automatic cross-expansion between stochastic and stochrsi
                # to strictly respect user-selected tokens from GUI.
                if 'williamsr' in wl_norm:
                    wl_norm.update({'williams %r','williams % r'})
                def map_name(line: str) -> Optional[str]:
                    if '•' not in line:
                        return None
                    core = line.split('•',1)[1].strip()
                    if core.startswith('→'):  # detail line keep only if parent kept
                        return None
                    base = core.split(':',1)[0].split('(',1)[0].strip().lower()
                    # Normalization / alias map (keep in sync with GUI tokens in app._collect_indicator_whitelist_tokens)
                    alias_map = {
                        'rsi':'rsi','rsi14':'rsi','rsi(14)':'rsi',
                        'macd':'macd',
                        'adx':'adx','adx14':'adx','adx(14)':'adx',
                        'stoch rsi':'stochrsi','stochrsi':'stochrsi','stochastic rsi':'stochrsi',
                        'stochastic':'stochastic',  # if ever present separately
                        'ema alignment':'ema','emaalignment':'ema',
                        'ema20':'ema20','ema50':'ema50','ema100':'ema100','ema200':'ema200',
                        'tema20':'tema20','tema50':'tema50','tema100':'tema100','tema200':'tema200',
                        'sma20':'sma20','wma20':'wma20',
                        'ichimoku cloud':'ichimoku','ichimoku':'ichimoku',
                        'atr14':'atr','atr':'atr',
                        'bollinger':'bollinger','bollinger(20,2)':'bollinger',
                        'keltner':'keltner','keltner(20,1.5)':'keltner',
                        'cci':'cci','cci20':'cci','cci(20)':'cci',
                        'williams %r':'williamsr','williams % r':'williamsr','williams%r':'williamsr','williamsr':'williamsr',
                        'roc':'roc','roc10':'roc','roc(10)':'roc',
                        'obv':'obv',
                        # Chaikin Money Flow (a.k.a CMF). Keep token 'chaikin' for backward-compat.
                        'chaikin osc':'chaikin','chaikin':'chaikin','chaikin money flow':'chaikin','cmf':'chaikin',
                        'eom':'eom','eom20':'eom','eom(20)':'eom',
                        'force index':'force','force':'force',
                        'trix':'trix','trix15':'trix','trix(15)':'trix',
                        'dpo':'dpo','dpo20':'dpo','dpo(20)':'dpo',
                        'mass index':'mass','mass':'mass',
                        'vortex':'vortex',
                        'kst':'kst',
                        'ultimate osc':'ultimate','ultimate':'ultimate',
                        'envelopes':'envelopes','envelopes(20,2.0)':'envelopes',
                        'momentum/cycle':'momentum','momentum':'momentum',
                        'fibonacci':'fibonacci',
                        'donchian %ile':'donchian','donchian':'donchian',
                        'psar':'psar'
                    }
                    return alias_map.get(base, base)
                filtered = []
                keep_prev = False
                prev_key = None
                seen_primary = set()
                for ln in ind_lines:
                    stripped = ln.strip()
                    is_detail = stripped.startswith('→')
                    # Remove redundant placeholder MACD[-] bullet if main MACD already captured
                    if not is_detail and '•' in ln and 'macd[-]' in stripped.lower():
                        if any(sp == 'macd' for sp in seen_primary):
                            continue
                    if not is_detail:
                        key = map_name(ln)
                        prev_key = key
                        # Special handling: split Momentum/Cycle composite into sub-indicators if composite itself not whitelisted
                        if key == 'momentum':
                            # Only keep composite if explicitly selected now. Do NOT auto-expand.
                            if key not in wl_norm:
                                keep_prev = False
                                continue
                        keep_prev = (key in wl_norm) if key else False
                        if keep_prev:
                            primary = key or stripped.split(':',1)[0].lower()
                            if primary in seen_primary:
                                # Skip placeholder duplicates like 'MACD[-]' if already had MACD bullet
                                continue
                            # Drop pure placeholder MACD lines of form '• MACD[-]' (no colon)
                            if primary == 'macd' and 'macd[-]' in stripped.lower() and any(sp == 'macd' for sp in seen_primary):
                                continue
                            seen_primary.add(primary)
                            filtered.append(ln)
                    else:
                        if keep_prev and prev_key in wl_norm:
                            filtered.append(ln)
                if filtered:
                    ind_lines = filtered
                    try:
                        if logger and logger.isEnabledFor(logging.DEBUG):
                            kept = []
                            for ln in ind_lines:
                                if '•' in ln and not ln.strip().startswith('→'):
                                    base = ln.split('•',1)[1].strip().split(':',1)[0]
                                    kept.append(base)
                            logger.debug("Kept primary bullets after filtering: %s", ", ".join(kept))
                    except Exception:
                        pass
        except Exception:
            pass

    # (Revert to original stable behavior: keep previously cleaned lines without aggressive pruning)

        # Ensure 100% mapping: append placeholder lines for any selected indicators missing after filtering
        try:
            if isinstance(analysis, dict) and '_indicator_whitelist' in analysis:
                selected_tokens = {str(x).lower() for x in analysis.get('_indicator_whitelist') if isinstance(x, str)}
                if selected_tokens:
                    try:
                        if logger and logger.isEnabledFor(logging.DEBUG):
                            logger.debug("Indicator selection (selected): %s", ", ".join(sorted(selected_tokens)))
                    except Exception:
                        pass
                    # Map existing primary bullets to tokens & enrich existing '-' values
                    alias = {
                        'rsi':'rsi','macd':'macd','adx':'adx','stochrsi':'stochrsi','stochastic':'stochastic',
                        'atr14':'atr','atr':'atr','bollinger':'bollinger','donchian':'donchian','donchian %ile':'donchian',
                        'ema20':'ema20','ema50':'ema50','ema100':'ema100','ema200':'ema200',
                        'sma20':'sma20','wma20':'wma20',
                        # Explicit TEMA aliases to ensure present_tokens picks them up reliably
                        'tema20':'tema20','tema50':'tema50','tema100':'tema100','tema200':'tema200',
                        'ichimoku':'ichimoku','keltner':'keltner','cci':'cci',
                        'williams %r':'williamsr','williamsr':'williamsr','roc':'roc','obv':'obv','chaikin osc':'chaikin','chaikin':'chaikin',
                        'eom':'eom','force index':'force','force':'force','trix':'trix','dpo':'dpo','mass index':'mass','mass':'mass',
                        'vortex':'vortex','kst':'kst','ultimate osc':'ultimate','ultimate':'ultimate','envelopes':'envelopes',
                        'momentum/cycle':'momentum','momentum':'momentum','fibonacci':'fibonacci','psar':'psar','patterns':'patterns'
                    }
                    # Normalize selected tokens to canonical keys used in present_tokens
                    import re as _re
                    def _normalize_selected_token(t: str) -> str:
                        tl = (t or '').strip().lower()
                        if not tl:
                            return tl
                        # Convert MA family forms like 'ema(22)' -> 'ema22'
                        m = _re.fullmatch(r"(ema|sma|wma|tema)\s*\(?\s*(\d{1,3})\s*\)?", tl)
                        if m:
                            fam = m.group(1)
                            per = m.group(2)
                            return f"{fam}{per}"
                        # Strip parameter parentheses for non-MA (e.g., 'bollinger(20,2.0)')
                        base = tl.split('(',1)[0].strip()
                        # Alias mapping to canonical
                        return alias.get(base, base)
                    selected_norm = {_normalize_selected_token(t) for t in selected_tokens}
                    try:
                        if logger and logger.isEnabledFor(logging.DEBUG):
                            logger.debug("Indicator selection (normalized): %s", ", ".join(sorted(selected_norm)))
                    except Exception:
                        pass
                    # Note: exporter variants like ForceIndex14 / ROC20 are normalized earlier during scan.
                    present_tokens: dict[str,int] = {}
                    for idx,_ln in enumerate(ind_lines):
                        if '•' in _ln and not _ln.strip().startswith('→'):
                            try:
                                core = _ln.split('•',1)[1].strip()
                                base = core.split(':',1)[0].split('(',1)[0].strip().lower()
                                tok = alias.get(base, base)
                                if tok not in present_tokens:
                                    present_tokens[tok] = idx
                            except Exception:
                                continue
                    try:
                        if logger and logger.isEnabledFor(logging.DEBUG):
                            logger.debug("Indicator selection (present-initial): %s", ", ".join(sorted(present_tokens.keys())))
                    except Exception:
                        pass
                    
                    # Force-mark indicators as present if we have data and parameters for them
                    if chaikin is not None and _params.get('chaikin') is not None:
                        # Mark 'chaikin' as present to match normalized token
                        present_tokens['chaikin'] = -1
                    def _fmt_num(v, nd=4):
                        return f"{v:.{nd}f}" if isinstance(v,(int,float)) else "-"
                    # Enrich existing lines if they contain '-' placeholders
                    def _rel_price_line(val):
                        if not isinstance(val,(int,float)) or not isinstance(price_trend,(int,float)):
                            return ''
                        if price_trend > val: return ' (giá trên)'
                        if price_trend < val: return ' (giá dưới)'
                        return ' (giá ≈)'
                    # Helper: parse MA token like 'ema34' -> ('EMA', 34)
                    import re as _re
                    def _parse_ma_token(tok: str):
                        try:
                            tl = tok.lower().strip()
                            if not (tl.startswith('ema') or tl.startswith('sma') or tl.startswith('wma') or tl.startswith('tema')):
                                return None
                            m = _re.search(r'(ema|sma|wma|tema)(\d{1,3})$', tl)
                            if not m:
                                return None
                            fam = m.group(1).upper()
                            per = int(m.group(2))
                            if per <= 0 or per > 600:
                                return None
                            return fam, per
                        except Exception:
                            return None
                    # Pre-compute selected MA periods by family for cross detection
                    ma_selected: dict[str, list[int]] = {}
                    for _t in selected_tokens:
                        parsed = _parse_ma_token(_t)
                        if parsed:
                            fam, per = parsed
                            ma_selected.setdefault(fam, []).append(per)
                    for k in ma_selected:
                        ma_selected[k] = sorted(set(ma_selected[k]))

                    def _last2_ma(fam: str, per: int):
                        try:
                            key = f"{fam}_{per}"
                            return _last_two_fallback(key)
                        except Exception:
                            return (None, None)

                    for tok, idx in list(present_tokens.items()):
                        try:
                            if tok not in selected_norm:
                                # Remove indicators not selected
                                ind_lines[idx] = None
                                continue
                            line = ind_lines[idx]
                            if line is None:
                                continue
                            if tok == 'macd' and 'hist -' in line and 'macd_h' in locals() and isinstance(macd_h,(int,float)):
                                state = 'TĂNG' if macd_h>0 else 'GIẢM' if macd_h<0 else 'TRUNG TÍNH'
                                ind_lines[idx] = f"        • {macd_label}: {state} (hist {macd_h:.4f})"
                            if tok == 'rsi' and ('- (trung tính)' in line or 'trung tính' in line) and isinstance(rsi,(int,float)):
                                ind_lines[idx] = f"        • {rsi_label}: {rsi:.1f} – {(rsi_state or 'trung tính')}"
                            if tok.startswith('ema') and isinstance(price_trend,(int,float)):
                                # ensure we show relation for common EMA variants
                                if tok == 'ema20' and isinstance(ema20,(int,float)) and 'EMA20' in line:
                                    if 'giá =' in line:
                                        pass
                                    else:
                                        ind_lines[idx] = f"        • EMA20: {ema20:.5f}{_rel_price_line(ema20)}"
                                elif tok == 'ema50' and 'EMA50' in line:
                                    v = locals().get('ema50')
                                    if isinstance(v,(int,float)) and 'giá =' not in line:
                                        ind_lines[idx] = f"        • EMA50: {v:.5f}{_rel_price_line(v)}"
                                elif tok == 'ema100' and 'EMA100' in line:
                                    v = locals().get('ema100')
                                    if isinstance(v,(int,float)) and 'giá =' not in line:
                                        ind_lines[idx] = f"        • EMA100: {v:.5f}{_rel_price_line(v)}"
                                elif tok == 'ema200' and 'EMA200' in line:
                                    v = locals().get('ema200')
                                    if isinstance(v,(int,float)) and 'giá =' not in line:
                                        ind_lines[idx] = f"        • EMA200: {v:.5f}{_rel_price_line(v)}"
                                # Append slope / nearest-higher-period cross inline
                                try:
                                    p = int(tok.replace('ema',''))
                                    fam = 'EMA'
                                    prev, now = _last2_ma(fam, p)
                                    suffix = ''
                                    if isinstance(prev,(int,float)) and isinstance(now,(int,float)):
                                        if now > prev:
                                            suffix += " – dốc lên"
                                        elif now < prev:
                                            suffix += " – dốc xuống"
                                    # nearest higher cross within selected
                                    higher = [q for q in ma_selected.get(fam, []) if q > p]
                                    if higher:
                                        q = min(higher)
                                        p_prev, p_now = _last2_ma(fam, p)
                                        q_prev, q_now = _last2_ma(fam, q)
                                        if all(isinstance(x,(int,float)) for x in (p_prev,p_now,q_prev,q_now)):
                                            diff_prev = float(p_prev) - float(q_prev)
                                            diff_now = float(p_now) - float(q_now)
                                            if diff_prev <= 0 and diff_now > 0:
                                                suffix += f" – cắt lên {fam}{q}"
                                            elif diff_prev >= 0 and diff_now < 0:
                                                suffix += f" – cắt xuống {fam}{q}"
                                    if suffix and suffix not in ind_lines[idx]:
                                        ind_lines[idx] = ind_lines[idx] + suffix
                                except Exception:
                                    pass
                            if tok.startswith('sma') and isinstance(price_trend,(int,float)):
                                if tok == 'sma20' and isinstance(sma20,(int,float)) and 'SMA20' in line and 'giá =' not in line:
                                    ind_lines[idx] = f"        • SMA20: {sma20:.5f}{_rel_price_line(sma20)}"
                                elif tok == 'sma50' and 'SMA50' in line:
                                    v = locals().get('sma50')
                                    if isinstance(v,(int,float)) and 'giá =' not in line:
                                        ind_lines[idx] = f"        • SMA50: {v:.5f}{_rel_price_line(v)}"
                                # Slope / nearest-higher cross for SMA
                                try:
                                    p = int(tok.replace('sma',''))
                                    fam = 'SMA'
                                    prev, now = _last2_ma(fam, p)
                                    suffix = ''
                                    if isinstance(prev,(int,float)) and isinstance(now,(int,float)):
                                        if now > prev:
                                            suffix += " – dốc lên"
                                        elif now < prev:
                                            suffix += " – dốc xuống"
                                    higher = [q for q in ma_selected.get(fam, []) if q > p]
                                    if higher:
                                        q = min(higher)
                                        p_prev, p_now = _last2_ma(fam, p)
                                        q_prev, q_now = _last2_ma(fam, q)
                                        if all(isinstance(x,(int,float)) for x in (p_prev,p_now,q_prev,q_now)):
                                            diff_prev = float(p_prev) - float(q_prev)
                                            diff_now = float(p_now) - float(q_now)
                                            if diff_prev <= 0 and diff_now > 0:
                                                suffix += f" – cắt lên {fam}{q}"
                                            elif diff_prev >= 0 and diff_now < 0:
                                                suffix += f" – cắt xuống {fam}{q}"
                                    if suffix and suffix not in ind_lines[idx]:
                                        ind_lines[idx] = ind_lines[idx] + suffix
                                except Exception:
                                    pass
                            if tok.startswith('wma') and isinstance(price_trend,(int,float)):
                                if tok == 'wma20' and isinstance(wma20,(int,float)) and 'WMA20' in line and 'giá =' not in line:
                                    ind_lines[idx] = f"        • WMA20: {wma20:.5f}{_rel_price_line(wma20)}"
                                # Slope / nearest-higher cross for WMA (if higher exists)
                                try:
                                    p = int(tok.replace('wma',''))
                                    fam = 'WMA'
                                    prev, now = _last2_ma(fam, p)
                                    suffix = ''
                                    if isinstance(prev,(int,float)) and isinstance(now,(int,float)):
                                        if now > prev:
                                            suffix += " – dốc lên"
                                        elif now < prev:
                                            suffix += " – dốc xuống"
                                    higher = [q for q in ma_selected.get(fam, []) if q > p]
                                    if higher:
                                        q = min(higher)
                                        p_prev, p_now = _last2_ma(fam, p)
                                        q_prev, q_now = _last2_ma(fam, q)
                                        if all(isinstance(x,(int,float)) for x in (p_prev,p_now,q_prev,q_now)):
                                            diff_prev = float(p_prev) - float(q_prev)
                                            diff_now = float(p_now) - float(q_now)
                                            if diff_prev <= 0 and diff_now > 0:
                                                suffix += f" – cắt lên {fam}{q}"
                                            elif diff_prev >= 0 and diff_now < 0:
                                                suffix += f" – cắt xuống {fam}{q}"
                                    if suffix and suffix not in ind_lines[idx]:
                                        ind_lines[idx] = ind_lines[idx] + suffix
                                except Exception:
                                    pass
                            if tok.startswith('tema') and isinstance(price_trend,(int,float)):
                                if tok == 'tema20' and isinstance(tema20,(int,float)) and 'TEMA20' in line and 'giá =' not in line:
                                    ind_lines[idx] = f"        • TEMA20: {tema20:.5f}{_rel_price_line(tema20)}"
                                elif tok == 'tema50' and 'TEMA50' in line:
                                    v = locals().get('tema50')
                                    if isinstance(v,(int,float)) and 'giá =' not in line:
                                        ind_lines[idx] = f"        • TEMA50: {v:.5f}{_rel_price_line(v)}"
                                elif tok == 'tema100' and 'TEMA100' in line:
                                    v = locals().get('tema100')
                                    if isinstance(v,(int,float)) and 'giá =' not in line:
                                        ind_lines[idx] = f"        • TEMA100: {v:.5f}{_rel_price_line(v)}"
                                elif tok == 'tema200' and 'TEMA200' in line:
                                    v = locals().get('tema200')
                                    if isinstance(v,(int,float)) and 'giá =' not in line:
                                        ind_lines[idx] = f"        • TEMA200: {v:.5f}{_rel_price_line(v)}"
                                # Slope / nearest-higher cross for TEMA
                                try:
                                    p = int(tok.replace('tema',''))
                                    fam = 'TEMA'
                                    prev, now = _last2_ma(fam, p)
                                    suffix = ''
                                    if isinstance(prev,(int,float)) and isinstance(now,(int,float)):
                                        if now > prev:
                                            suffix += " – dốc lên"
                                        elif now < prev:
                                            suffix += " – dốc xuống"
                                    higher = [q for q in ma_selected.get(fam, []) if q > p]
                                    if higher:
                                        q = min(higher)
                                        p_prev, p_now = _last2_ma(fam, p)
                                        q_prev, q_now = _last2_ma(fam, q)
                                        if all(isinstance(x,(int,float)) for x in (p_prev,p_now,q_prev,q_now)):
                                            diff_prev = float(p_prev) - float(q_prev)
                                            diff_now = float(p_now) - float(q_now)
                                            if diff_prev <= 0 and diff_now > 0:
                                                suffix += f" – cắt lên {fam}{q}"
                                            elif diff_prev >= 0 and diff_now < 0:
                                                suffix += f" – cắt xuống {fam}{q}"
                                    if suffix and suffix not in ind_lines[idx]:
                                        ind_lines[idx] = ind_lines[idx] + suffix
                                except Exception:
                                    pass
                            # New enrichments for commonly selected tokens
                            if tok == 'adx' and isinstance(adx,(int,float)):
                                # If line lacks value or hint, inject
                                if 'ADX' in line and (' – ' not in line or any(x in line for x in ('-','không khả dụng'))):
                                    txt = adx_txt if 'adx_txt' in locals() and adx_txt else None
                                    ind_lines[idx] = f"        • {adx_label}: {adx:.1f}" + (f" – {txt}" if txt else '')
                            if tok == 'atr' and isinstance(atr,(int,float)):
                                if 'ATR' in line and (' – ' not in line or ' -' in line):
                                    ind_lines[idx] = f"        • {atr_label}: {atr:.4f}" + (f" – {atr_desc}" if 'atr_desc' in locals() and atr_desc else '')
                            if tok == 'roc' and isinstance(roc,(int,float)):
                                if 'ROC' in line and (' -' in line or ' – ' not in line):
                                    ind_lines[idx] = f"        • {roc_label}: {roc:.2f}" + (f" – {roc_desc}" if 'roc_desc' in locals() and roc_desc else '')
                            if tok == 'force' and isinstance(force,(int,float)):
                                if 'Force Index' in line and (' -' in line or ' – ' not in line):
                                    ind_lines[idx] = f"        • {force_label}: {force:.3f}" + (f" – {force_desc}" if 'force_desc' in locals() and force_desc else '')
                            if tok == 'williamsr':
                                # Prefer direct variable; otherwise scan last indicator row for any 'williamsr*' or 'willr*' key
                                wv = willr if isinstance(willr,(int,float)) else None
                                if wv is None:
                                    try:
                                        if isinstance(ind_trend, list) and ind_trend:
                                            row = ind_trend[-1]
                                            if isinstance(row, dict):
                                                for k, vv in row.items():
                                                    try:
                                                        kl = str(k).lower()
                                                        if (kl.startswith('williamsr') or kl.startswith('willr')) and isinstance(vv,(int,float)):
                                                            wv = float(vv)
                                                            break
                                                    except Exception:
                                                        continue
                                    except Exception:
                                        pass
                                if isinstance(wv,(int,float)) and 'Williams %R' in line and (' -' in line or ' – ' not in line):
                                    ind_lines[idx] = f"        • {willr_label}: {wv:.1f}" + (f" – {will_desc}" if 'will_desc' in locals() and will_desc else '')
                            if tok == 'cci':
                                # Try default CCI20, else dynamic from last row any 'cci*'
                                cv = None
                                try:
                                    cv = _last_from_fallback('CCI20')  # may be None
                                except Exception:
                                    cv = None
                                if not isinstance(cv,(int,float)):
                                    try:
                                        if isinstance(ind_trend, list) and ind_trend:
                                            row = ind_trend[-1]
                                            if isinstance(row, dict):
                                                for k, vv in row.items():
                                                    try:
                                                        kl = str(k).lower()
                                                        if kl.startswith('cci') and isinstance(vv,(int,float)):
                                                            cv = float(vv)
                                                            break
                                                    except Exception:
                                                        continue
                                    except Exception:
                                        pass
                                # Fix: Don't replace if line already has description (contains ' – ')
                                if isinstance(cv,(int,float)) and 'CCI' in line and (' -' in line or ' – ' not in line) and ' – ' not in line:
                                    ind_lines[idx] = f"        • {cci_label}: {cv:.1f}"
                            if tok == 'obv' and isinstance(obv,(int,float)):
                                if 'OBV' in line and (' -' in line or ' – ' not in line):
                                    ind_lines[idx] = f"        • OBV: {obv:.0f}" + (f" – {obv_desc}" if 'obv_desc' in locals() and obv_desc else '')
                            if tok == 'bollinger' and isinstance(bwidth,(int,float)):
                                if 'Bollinger' in line and 'width' not in line:
                                    # Append width info minimally
                                    ind_lines[idx] = line + f" (width {bwidth:.4f})"
                            if tok == 'donchian' and isinstance(d_lo,(int,float)) and isinstance(d_up,(int,float)):
                                if 'Donchian' in line and line.strip().endswith(' -'):
                                    rng = d_up - d_lo
                                    ind_lines[idx] = f"        • {donch_label}: chờ phá vỡ – range {rng:.5f}"
                            if tok == 'ultimate' and isinstance(ult,(int,float)):
                                if 'Ultimate' in line and (' -' in line or ' – ' not in line):
                                    ind_lines[idx] = f"        • Ultimate Osc: {ult:.1f}" + (f" – {ult_desc}" if 'ult_desc' in locals() and ult_desc else '')
                        except Exception as _enrich_err:
                            try:
                                if logger and logger.isEnabledFor(logging.DEBUG):
                                    logger.debug("Indicator enrichment error for %s: %s", tok, _enrich_err)
                            except Exception:
                                pass
                    # Remove None entries from pruning
                    ind_lines = [l for l in ind_lines if l]
                    # Rebuild present_tokens after pruning to ensure accuracy
                    present_tokens = {}
                    for idx,_ln in enumerate(ind_lines):
                        if '•' in _ln and not _ln.strip().startswith('→'):
                            try:
                                core = _ln.split('•',1)[1].strip()
                                base = core.split(':',1)[0].split('(',1)[0].strip().lower()
                                tok = alias.get(base, base)
                                if tok not in present_tokens:
                                    present_tokens[tok] = idx
                            except Exception:
                                continue
                    try:
                        if logger and logger.isEnabledFor(logging.DEBUG):
                            logger.debug("Indicator selection (present-final): %s", ", ".join(sorted(present_tokens.keys())))
                    except Exception:
                        pass
                    
                    # Force-mark indicators as present if we have data and parameters for them (after rebuild)
                    if chaikin is not None and _params.get('chaikin') is not None and 'chaikin' not in present_tokens:
                        # Add Chaikin line dynamically since it's missing from ind_lines
                        chaikin_line = f"        • {chaikin_label}: {chaikin:.3f} – {chaikin_txt if chaikin_txt else 'tăng' if chaikin > 0 else 'giảm'}"
                        ind_lines.append(chaikin_line)
                        present_tokens['chaikin'] = len(ind_lines) - 1
                        
                    # Add missing tokens as new bullets
                    missing = [t for t in selected_norm if t not in present_tokens]
                    # Remove indicators from missing list if we have data and parameters for them
                    if chaikin is not None and _params.get('chaikin') is not None and 'chaikin' in missing:
                        missing.remove('chaikin')
                    try:
                        if logger and logger.isEnabledFor(logging.DEBUG):
                            logger.debug("Indicator selection (missing): %s", ", ".join(sorted(missing)))
                    except Exception:
                        pass
                    placeholders: list[str] = []
                    # Helpers for dynamic MA parsing and lookup
                    # (moved _parse_ma_token above to avoid NameError during ma_selected precompute)
                    def _isnum(x):
                        return isinstance(x,(int,float)) and not (x!=x)
                    def _last_any(keys: list[str]):
                        try:
                            for k in keys:
                                v = _last_from_fallback(k)
                                if _isnum(v):
                                    return float(v)
                            # Try scan-first-non-null across indicators if available
                            try:
                                v = _scan_first_non_null(keys)  # type: ignore
                                if _isnum(v):
                                    return float(v)
                            except Exception:
                                pass
                            # As a last resort, peek last indicator record
                            try:
                                if isinstance(ind_trend, list) and ind_trend:
                                    last = ind_trend[-1]
                                    if isinstance(last, dict):
                                        for k in keys:
                                            vv = last.get(k)
                                            if _isnum(vv):
                                                return float(vv)
                            except Exception:
                                pass
                        except Exception:
                            pass
                        return None
                    def _last_any_prefix(prefixes: list[str]):
                        """Find last non-null numeric value for any key starting with given prefixes.
                        Searches the last indicator record first, then scans backward if needed.
                        """
                        try:
                            if isinstance(ind_trend, list) and ind_trend:
                                # Check last row keys first
                                for row in (ind_trend[-1],):
                                    if isinstance(row, dict):
                                        for k, vv in row.items():
                                            try:
                                                kl = str(k).lower()
                                                if any(kl.startswith(p.lower()) for p in prefixes) and _isnum(vv):
                                                    return float(vv)
                                            except Exception:
                                                continue
                                # Scan backward for first available
                                for row in reversed(ind_trend):
                                    if isinstance(row, dict):
                                        for k, vv in row.items():
                                            try:
                                                kl = str(k).lower()
                                                if any(kl.startswith(p.lower()) for p in prefixes) and _isnum(vv):
                                                    return float(vv)
                                            except Exception:
                                                continue
                        except Exception:
                            pass
                        return None
                    for tok in sorted(missing):
                        display = tok.upper()
                        if tok.startswith('ema'):
                            display = tok.replace('ema','EMA')
                        if tok.startswith('tema'):
                            display = tok.replace('tema','TEMA')
                        if tok.startswith('sma'):
                            display = tok.replace('sma','SMA')
                        if tok.startswith('wma'):
                            display = tok.replace('wma','WMA')
                        # Normalize friendly display names to match GUI labels (avoid hardcoded periods)
                        if tok == 'stochrsi':
                            display = 'StochRSI'
                        elif tok == 'stochastic':
                            display = 'Stochastic'
                        elif tok == 'chaikin':
                            display = 'Chaikin Money Flow'
                        if tok == 'stochrsi': display = 'StochRSI'
                        elif tok == 'stochastic': display = 'Stochastic'
                        elif tok == 'bollinger': display = 'Bollinger'
                        elif tok == 'donchian': display = 'Donchian'
                        elif tok == 'atr': display = 'ATR'
                        elif tok == 'williamsr': display = 'Williams %R'
                        elif tok == 'force': display = 'Force Index'
                        elif tok == 'ultimate': display = 'Ultimate Osc'
                        elif tok == 'mass': display = 'Mass Index'
                        elif tok == 'momentum': display = 'Momentum/Cycle'
                        elif tok == 'mfi': display = 'MFI'
                        # Build description
                        desc = 'chưa có'
                        try:
                            if tok == 'rsi':
                                rv = None
                                try:
                                    if isinstance(rsi,(int,float)):
                                        rv = float(rsi)
                                except Exception:
                                    rv = None
                                if not _isnum(rv):
                                    rv = _last_any_prefix(['rsi'])
                                if _isnum(rv):
                                    desc = f"{rv:.1f}"
                            elif tok == 'macd' and isinstance(macd_h,(int,float)):
                                desc = f"hist {macd_h:.4f}"
                            elif tok.startswith(('ema','sma','wma','tema')):
                                # Dynamic MA rendering: use ma_line for consistent phrasing.
                                parsed = _parse_ma_token(tok)
                                mv = None
                                label_for_line = display
                                if parsed:
                                    fam, per = parsed
                                    # Use dynamic MA value extraction
                                    mv = _get_ma_value_dynamic(fam.upper(), per)
                                else:
                                    # Backward-compat for known fixed aliases - use dynamic lookup
                                    m = re.match(r'(ema|sma|wma|tema)(\d+)', tok)
                                    if m:
                                        ma_type, period_str = m.groups()
                                        period = int(period_str)
                                        mv = _get_ma_value_dynamic(ma_type.upper(), period)
                                
                                # Skip placeholders for MA that don't have data
                                if mv is None:
                                    continue
                                    
                                # Push a full MA line using ma_line formatting
                                placeholders.append(ma_line(label_for_line, mv, None, ""))
                                continue
                            elif tok == 'cci':
                                cv = _last_any(['CCI20','cci20','CCI_20'])
                                if not _isnum(cv):
                                    cv = _last_any_prefix(['cci'])
                                if _isnum(cv):
                                    desc = f"{cv:.1f}"
                            elif tok == 'williamsr':
                                wv = None
                                try:
                                    if isinstance(willr,(int,float)):
                                        wv = float(willr)
                                except Exception:
                                    wv = None
                                if not _isnum(wv):
                                    wv = _last_any_prefix(['williamsr','willr'])
                                if _isnum(wv):
                                    desc = f"{wv:.1f}"
                            elif tok == 'adx':
                                av = None
                                if isinstance(adx,(int,float)):
                                    av = float(adx)
                                if not _isnum(av):
                                    av = _last_any_prefix(['adx'])
                                if _isnum(av):
                                    desc = f"{av:.1f}"
                            elif tok == 'stochrsi':
                                sv = None
                                if isinstance(stochrsi,(int,float)):
                                    sv = float(stochrsi)
                                if not _isnum(sv):
                                    sv = _last_any_prefix(['stochrsi'])
                                if _isnum(sv):
                                    desc = f"{sv:.2f}"
                            elif tok == 'stochastic':
                                kv = stoch_k if isinstance(stoch_k,(int,float)) else None
                                dv = stoch_d if isinstance(stoch_d,(int,float)) else None
                                if not _isnum(kv):
                                    kv = _last_any_prefix(['stochk','stoch_k','%k'])
                                if not _isnum(dv):
                                    dv = _last_any_prefix(['stochd','stoch_d','%d'])
                                if _isnum(kv) and _isnum(dv):
                                    desc = f"K {kv:.1f} / D {dv:.1f}"
                            elif tok == 'atr':
                                av = atr if isinstance(atr,(int,float)) else None
                                if not _isnum(av):
                                    av = _last_any_prefix(['atr'])
                                if _isnum(av):
                                    desc = f"{av:.5f}"
                            elif tok == 'bollinger':
                                bw = bwidth if isinstance(bwidth,(int,float)) else None
                                if not _isnum(bw):
                                    bw = _last_any_prefix(['bollwidth','bwidth','bbwidth','bandwidth'])
                                if _isnum(bw):
                                    desc = f"width {bw:.4f}"
                            elif tok == 'donchian':
                                dlo = d_lo if isinstance(d_lo,(int,float)) else None
                                dup = d_up if isinstance(d_up,(int,float)) else None
                                if not _isnum(dlo) or not _isnum(dup):
                                    # Try prefixes for lower/upper
                                    dlo = _last_any_prefix(['donchlower','donch_low','donchianlower','donchian_lo','donch_lo'])
                                    dup = _last_any_prefix(['donchupper','donch_high','donchianupper','donchian_hi','donch_hi'])
                                if _isnum(dlo) and _isnum(dup):
                                    desc = f"range {(dup-dlo):.5f}"
                            elif tok == 'roc':
                                rv = roc if isinstance(roc,(int,float)) else None
                                if not _isnum(rv):
                                    rv = _last_any_prefix(['roc'])
                                if _isnum(rv):
                                    desc = f"{rv:.2f}"
                            elif tok == 'obv':
                                ov = obv if isinstance(obv,(int,float)) else None
                                if not _isnum(ov):
                                    ov = _last_any_prefix(['obv'])
                                if _isnum(ov):
                                    desc = f"{ov:.0f}"
                            elif tok == 'mfi':
                                mv_ = mfi if isinstance(mfi,(int,float)) else None
                                if not _isnum(mv_):
                                    mv_ = _last_any_prefix(['mfi'])
                                if _isnum(mv_):
                                    desc = f"{mv_:.1f}"
                            elif tok == 'force':
                                fv = force if isinstance(force,(int,float)) else None
                                if not _isnum(fv):
                                    fv = _last_any_prefix(['forceindex','force'])
                                if _isnum(fv):
                                    desc = f"{fv:.2f}"
                            elif tok == 'chaikin':
                                cv = chaikin if isinstance(chaikin,(int,float)) else None
                                if not _isnum(cv):
                                    cv = _last_any_prefix(['chaikin','chaikin18'])
                                if _isnum(cv):
                                    desc = f"{cv:.3f}"
                            elif tok == 'vortex' and isinstance(vortex,dict):
                                vp = vortex.get('vi_plus'); vn = vortex.get('vi_minus')
                                if isinstance(vp,(int,float)) and isinstance(vn,(int,float)):
                                    desc = f"VI+ {vp:.2f} / VI- {vn:.2f}"
                            elif tok == 'kst':
                                kv = kst if isinstance(kst,(int,float)) else None
                                if not _isnum(kv):
                                    kv = _last_any_prefix(['kst'])
                                if _isnum(kv):
                                    desc = f"diff {kv:.2f}"
                            elif tok == 'ultimate':
                                uv = ult if isinstance(ult,(int,float)) else None
                                if not _isnum(uv):
                                    uv = _last_any_prefix(['ultimate'])
                                if _isnum(uv):
                                    desc = f"{uv:.1f}"
                            elif tok == 'psar':
                                pdv = psar_dir if isinstance(psar_dir,(int,float)) else None
                                if pdv is None:
                                    pdv = _last_any_prefix(['psar'])
                                if isinstance(pdv,(int,float)):
                                    desc = 'tăng' if pdv>0 else 'giảm' if pdv<0 else 'trung tính'
                            elif tok == 'fibonacci' and fibo_note:
                                desc = (fibo_note[:40] + '…') if len(fibo_note)>43 else fibo_note
                            elif tok == 'patterns':
                                desc = 'xem mục mô hình'
                        except Exception:
                            pass
                        placeholders.append(f"        • {display}: {desc}")
                    if placeholders:
                        ind_lines.extend(placeholders)
                        try:
                            if logger and logger.isEnabledFor(logging.DEBUG):
                                logger.debug("Appended placeholder indicators for selected tokens: %s", ", ".join(sorted(missing)))
                        except Exception:
                            pass
                    # Final safety: if any TEMA tokens are selected but no TEMA primary bullet exists, force-append it
                    try:
                        tema_selected = sorted([t for t in selected_tokens if t.startswith('tema')])
                        if tema_selected:
                            # Collect existing primary labels (e.g., 'EMA20','SMA20','WMA20','TEMA20')
                            prim_labels = set()
                            for _ln in ind_lines:
                                if '•' in _ln and not _ln.strip().startswith('→'):
                                    base_lbl = _ln.split('•',1)[1].strip().split(':',1)[0].split('(',1)[0].strip().upper()
                                    prim_labels.add(base_lbl)
                            # Append any missing TEMA{p}
                            forced: list[str] = []
                            for tok in tema_selected:
                                try:
                                    p = int(tok.replace('tema',''))
                                except Exception:
                                    continue
                                label = f"TEMA{p}"
                                if label not in prim_labels:
                                    # fetch last value; only compute fallback when not strict
                                    mv = _last_any([f"TEMA_{p}", f"TEMA{p}"])
                                    if mv is None and isinstance(closes, list) and closes and not CFG.STRICT_IND_ONLY:
                                        mv = _compute_tema(closes, p)
                                    # Only append if we have valid data - skip missing values to avoid "chưa có" clutter
                                    if mv is not None and isinstance(mv, (int, float)) and not (mv != mv):  # mv != mv checks for NaN
                                        ind_lines.append(ma_line(label, mv, None, ""))
                                        forced.append(tok)
                            if forced and logger and logger.isEnabledFor(logging.DEBUG):
                                logger.debug("Force-appended TEMA indicators: %s", ", ".join(forced))
                    except Exception:
                        pass
                    # (Removed coverage summary line per user request)
        except Exception:
            pass

    # (Optional) could propagate ma_cache back to variables if later logic needs them
    # for k,v in ma_cache.items():
    #     if k == 'EMA20': ema20 = v
    #     elif k == 'EMA50': ema50 = v
    #     elif k == 'EMA100': ema100 = v
    #     elif k == 'EMA200': ema200 = v
    #     elif k == 'SMA20': sma20 = v
    #     elif k == 'WMA20': wma20 = v

        # Smart summary (confluence)
        # Whitelist-aware confluence flags
        raw_flags = [
            ('rsi', rsi_sup), ('macd', macd_sup), ('psar', psar_sup),
            ('ema20', ema_sup), ('ema50', ema50_sup), ('ema200', ema200_sup), ('ema100', ema100_sup),
            ('sma20', sma_sup), ('sma50', sma50_sup), ('wma20', wma_sup),
            ('ichimoku', ichi_sup), ('vortex', vortex_sup), ('kst', kst_sup), ('ultimate', ult_sup)
        ]
        wl_tokens = None
        try:
            if isinstance(analysis, dict) and '_indicator_whitelist' in analysis:
                wl_tokens = {str(x).lower() for x in analysis.get('_indicator_whitelist')}
        except Exception:
            wl_tokens = None
        filtered_flags: List[Optional[bool]] = []
        for tok, flag in raw_flags:
            if wl_tokens and tok not in wl_tokens:
                continue
            filtered_flags.append(flag)
        if not filtered_flags:  # fallback to old behavior if nothing matched
            filtered_flags = [f for _, f in raw_flags]
        bull_count = sum(1 for x in filtered_flags if x is True)
        bear_count = sum(1 for x in filtered_flags if x is False)
        neut_count = sum(1 for x in filtered_flags if x is None)
        summary_lines = [
            "Tóm tắt:",
            f"  - Độ hội tụ chỉ báo: Ủng hộ BUY {bull_count} | Ủng hộ SELL {bear_count} | Trung tính {neut_count}",
        ]
        # Helper for whitelist gate inside summary
        def _wl(tok: str) -> bool:
            return not wl_tokens or tok in wl_tokens
        # Confluence score (simple balance metric)
        try:
            total_sig = bull_count + bear_count + neut_count
            if total_sig > 0:
                bias_ratio = (bull_count - bear_count) / max(total_sig,1)
                summary_lines.append(f"  - Chỉ số hội tụ: {bias_ratio:+.2f} (dương = nghiêng BUY)")
        except Exception:
            pass
            # Quick heatmap (momentum/volatility/trend breadth)
            try:
                heat_parts: List[str] = []
                if rsi is not None and _wl('rsi'):
                    heat_parts.append(f"RSI {rsi_state or ''}".strip())
                if macd_h is not None and _wl('macd'):
                    heat_parts.append("MACD+" if macd_h>0 else "MACD-")
                if adx is not None and _wl('adx'):
                    heat_parts.append(f"ADX {('M' if adx>=25 else 'W')}")
                if vortex_intensity_note and _wl('vortex'):
                    heat_parts.append(f"VTX {vortex_intensity_note}")
                if atr and price_trend and _wl('atr'):
                    atrr = (atr/price_trend)
                    atr_band = 'H' if atrr>0.015 else 'M' if atrr>0.007 else 'L'
                    heat_parts.append(f"ATR {atrr:.3f}({atr_band})")
                if heat_parts:
                    summary_lines.append("  - Heatmap: " + " | ".join(heat_parts))
            except Exception:
                pass
            if rsi_div_note:
                summary_lines.append(f"  - RSI divergence: {rsi_div_note}")
            if macd_div_note:
                summary_lines.append(f"  - MACD divergence: {macd_div_note}")
        # Add ATR/price summary if available
        if (atr is not None and price_trend) and _wl('atr'):
            summary_lines.append(f"  - Biến động (ATR14): {atr/price_trend:.4f} so với giá")
        # Strongest indicator hint (dominant signal) – diversify (không để chỉ MACD lặp lại)
        try:
            dom: List[Tuple[float, str]] = []
            sig = (final.get('signal') or '').upper()
            # Channel breakout is decisive
            if channel_breakout_note:
                dom.append((1.0, channel_breakout_note))
            # EMA stack trend
            if ema_stack_note:
                dom.append((0.9, ema_stack_note))
            # ADX strength
            ax = ffloat(adx, None)
            if ax is not None:
                dom.append((min(1.0, ax/50.0), f"ADX {ax:.1f}{(' – ' + adx_txt) if adx_txt else ''}"))
            # MACD histogram magnitude
            if macd_h is not None:
                mscore = min(1.0, abs(macd_h)/0.001)
                mtxt = "MACD dương" if macd_h > 0 else ("MACD âm" if macd_h < 0 else "MACD trung tính")
                dom.append((mscore, f"{mtxt} (hist {macd_h:.4f})"))
            # RSI distance from 50
            if rsi is not None:
                rscore = min(1.0, abs(rsi-50.0)/25.0)
                dom.append((rscore, f"RSI {rsi:.1f}{(' – ' + rsi_state) if rsi_state else ''}"))
            # Vortex gap
            try:
                vip = ffloat(vortex.get('vi_plus'), None) if isinstance(vortex, dict) else None
                vim = ffloat(vortex.get('vi_minus'), None) if isinstance(vortex, dict) else None
                if vip is not None and vim is not None:
                    vgap = abs(vip - vim)
                    dom.append((min(1.0, vgap/1.5), f"Vortex chênh lệch {vgap:.2f}"))
            except Exception:
                pass
            if dom:
                # Add Donchian percentile extremes & phase notes
                try:
                    if channel_percentile_note:
                        m = re.search(r'(\d+\.?\d*)%', channel_percentile_note)
                        if m:
                            v = float(m.group(1))
                            if v >= 90:
                                dom.append((0.82, f"Donchian % {v:.1f} (gần đỉnh kênh)"))
                            elif v <= 10:
                                dom.append((0.82, f"Donchian % {v:.1f} (gần đáy kênh)"))
                except Exception:
                    pass
                if macd_phase_note:
                    dom.append((0.70, f"Pha MACD: {macd_phase_note}"))
                if trix_phase_note:
                    dom.append((0.68, f"Pha TRIX: {trix_phase_note}"))
                dom.sort(key=lambda x: x[0], reverse=True)
                top = dom[:3]
                # Loại bỏ trùng loại (ví dụ nhiều MACD) – ưu tiên đa dạng
                seen_phr = set()
                uniq: List[str] = []
                for _, txt in top:
                    base_key = txt.split(':')[0].split('(')[0].strip()
                    if base_key not in seen_phr:
                        seen_phr.add(base_key)
                        uniq.append(txt)
                try:
                    wl_tokens = None
                    if isinstance(analysis, dict) and '_indicator_whitelist' in analysis:
                        wl_tokens = {str(x).lower() for x in analysis.get('_indicator_whitelist')}
                    if wl_tokens:
                        def phrase_token(ph: str) -> str:
                            low = ph.lower()
                            if 'rsi' in low and 'stoch' not in low:
                                return 'rsi'
                            if 'macd' in low:
                                return 'macd'
                            if 'adx' in low:
                                return 'adx'
                            if 'donchian' in low:
                                return 'donchian'
                            if 'ema20' in low:
                                return 'ema20'
                            if 'ema50' in low:
                                return 'ema50'
                            if 'ema100' in low:
                                return 'ema100'
                            if 'ema200' in low:
                                return 'ema200'
                            if 'bollinger' in low:
                                return 'bollinger'
                            if 'trix' in low:
                                return 'trix'
                            if 'kst' in low:
                                return 'kst'
                            if 'dpo' in low:
                                return 'dpo'
                            if 'vortex' in low:
                                return 'vortex'
                            if 'ultimate' in low:
                                return 'ultimate'
                            return low.split()[0]
                        rebuilt = []
                        for ph in uniq:
                            low = ph.lower()
                            if any(e in low for e in ['ema20','ema50','ema100','ema200']):
                                present = [e for e in ['EMA20','EMA50','EMA100','EMA200'] if e.lower() in low]
                                sel = [e for e in present if e.lower() in wl_tokens]
                                if not sel:
                                    continue
                                if len(sel) == 1:
                                    ph = sel[0]
                                else:
                                    ph = ' > '.join(sel)
                            if phrase_token(ph) in wl_tokens or (ph.startswith('EMA') and any(t.startswith('ema') for t in wl_tokens)):
                                rebuilt.append(ph)
                        uniq = rebuilt
                    summary_lines.append("  - Top tín hiệu nổi bật: " + "; ".join(uniq))
                except Exception:
                    summary_lines.append("  - Top tín hiệu nổi bật: " + "; ".join(uniq))
        except Exception:
            pass
        # News penalty transparency (only show if non-zero)
        try:
            base_c = ffloat(final.get('base_confidence'), None) if isinstance(final, dict) else None
            adj_c = ffloat(final.get('confidence'), None) if isinstance(final, dict) else None
            npen = ffloat(final.get('news_penalty'), None) if isinstance(final, dict) else None
            if base_c is not None and adj_c is not None and npen is not None and abs(npen) > 1e-6:
                summary_lines.append(f"  - Tin tức: phạt -{npen:.1f}% (từ {base_c:.1f}% còn {adj_c:.1f}%)")
        except Exception:
            pass
        # Donchian summary (add percentile if computed)
        if isinstance(d_lo, (int,float)) and isinstance(d_up, (int,float)) and price_trend is not None:
            width = (d_up - d_lo) / max(price_trend, 1e-12)
            pos = 'gần biên trên' if (price_trend - d_lo) > (d_up - price_trend) else 'gần biên dưới'
            pct_line = None
            try:
                if channel_percentile_note:
                    # Extract numeric % from existing note
                    import re
                    m = re.search(r'(\d+\.?\d*)%', channel_percentile_note)
                    if m:
                        v = float(m.group(1))
                        tier = 'cực cao' if v>=90 else 'cao' if v>=70 else 'trung bình' if v>=30 else 'thấp'
                        pct_line = f"percentile {v:.1f}% ({tier})"
            except Exception:
                pct_line = None
            try:
                donch_p = None
                # Prefer detected params earlier; else use d_win captured from channel scan
                if '_params' in locals():
                    donch_p = _params.get('donch_win')
                if not donch_p:
                    donch_p = locals().get('d_win')
            except Exception:
                donch_p = None
            label = f"Donchian({donch_p})" if donch_p else "Donchian"
            summary_lines.append(f"  - {label}: rộng {width:.4f}, {pos}" + (f", {pct_line}" if pct_line else ""))
        # Bollinger squeeze/wide context
        try:
            if isinstance(bb, dict) and price_trend:
                bu = bb.get('upper'); bl = bb.get('lower')
                if isinstance(bu, (int,float)) and isinstance(bl, (int,float)) and bu > bl:
                    bwidth = (bu - bl) / max(price_trend, 1e-12)
                    # Keep Vietnamese phrasing for VI report; EN will be translated later
                    if bwidth <= 0.010:
                        summary_lines.append(f"  - Bollinger: Squeeze (độ rộng {bwidth:.4f}) – có thể sắp bứt phá")
                    elif bwidth >= 0.030:
                        summary_lines.append(f"  - Bollinger: Rộng (độ rộng {bwidth:.4f}) – biến động cao")
        except Exception:
            pass
        # Near support/resistance warnings
        try:
            if price_trend is not None:
                if sup is not None:
                    dist_s = abs(price_trend - float(sup)) / max(price_trend, 1e-12)
                    if dist_s <= 0.002:
                        summary_lines.append("  - Cảnh báo: Giá đang gần vùng hỗ trợ")
                if res is not None:
                    dist_r = abs(price_trend - float(res)) / max(price_trend, 1e-12)
                    if dist_r <= 0.002:
                        summary_lines.append("  - Cảnh báo: Giá đang gần vùng kháng cự")
        except Exception:
            pass
        # Conflict note when indicators are split
        if bull_count >= 4 and bear_count >= 4:
            summary_lines.append("  - Lưu ý: Chỉ báo đang mâu thuẫn mạnh, ưu tiên bối cảnh khung thời gian lớn hơn")
        else:
            # Conflict breakdown: list opposing groups
            try:
                bulls = []
                bears = []
                wl_tokens = None
                try:
                    if isinstance(analysis, dict) and '_indicator_whitelist' in analysis:
                        wl_tokens = {str(x).lower() for x in analysis.get('_indicator_whitelist')}
                except Exception:
                    wl_tokens = None
                def _allowed(label: str) -> bool:
                    if not wl_tokens:
                        return True
                    m = label.lower()
                    mapping = {
                        'rsi':'rsi','macd':'macd','ema20':'ema20','psar':'psar','ichimoku':'ichimoku','kst':'kst','ultimate':'ultimate','vortex':'vortex'
                    }
                    token = mapping.get(m, m)
                    # accept any ema variant mapped to ema20 stack if one EMA selected
                    if token.startswith('ema') and any(t.startswith('ema') for t in wl_tokens):
                        return True
                    return token in wl_tokens
                def _add(flag,label):
                    if not _allowed(label):
                        return
                    if flag is True: bulls.append(label)
                    elif flag is False: bears.append(label)
                _add(rsi_sup,'RSI')
                _add(macd_sup,'MACD')
                _add(ema_sup,'EMA20')
                _add(psar_sup,'PSAR')
                _add(ichi_sup,'Ichimoku')
                _add(kst_sup,'KST')
                _add(ult_sup,'Ultimate')
                if vortex_sup is not None:
                    _add(vortex_sup,'Vortex')
                if bulls or bears:
                    summary_lines.append("  - Phân rã xung đột: BUY="+",".join(bulls if bulls else ['∅'])+" | SELL="+",".join(bears if bears else ['∅']))
            except Exception:
                pass

    # (Scenario generation removed per user request)
    # (Pattern enhancement reverted for stability; existing pattern extraction below remains)

    # (Pattern & candlestick enhanced block removed temporarily – will re-add inside function properly)
        # Assemble report
        lines: List[str] = []
        lines += [
            f"Thời gian: {ts}",
            "",
            f"Ký hiệu: {sym}",
            "",
            f"Tín hiệu: {(final.get('signal') or '').upper()}",
            f"Độ tin cậy: {int(round(ffloat(final.get('confidence'),0.0)))}%",
            # Use execution precision if provided in trade_idea
            (lambda _idea: (
                (lambda decs: (
                    f"Entry: {((lambda v: f'{v:.{decs}f}' if isinstance(v,(int,float)) else '-')(_idea.get('entry')))}",
                    f"Stoploss: {((lambda v: f'{v:.{decs}f}' if isinstance(v,(int,float)) else '-')(_idea.get('sl')))}",
                    f"Takeprofit: {((lambda v: f'{v:.{decs}f}' if isinstance(v,(int,float)) else '-')(_idea.get('tp')))}"
                ))((_idea.get('precision') if isinstance(_idea, dict) and isinstance(_idea.get('precision'), int) else price_decimals(sym)))
            ) if idea else ("Entry: -","Stoploss: -","Takeprofit: -"))(idea)[0],
            (lambda _idea: (
                (lambda decs: (
                    f"Entry: {((lambda v: f'{v:.{decs}f}' if isinstance(v,(int,float)) else '-')(_idea.get('entry')))}",
                    f"Stoploss: {((lambda v: f'{v:.{decs}f}' if isinstance(v,(int,float)) else '-')(_idea.get('sl')))}",
                    f"Takeprofit: {((lambda v: f'{v:.{decs}f}' if isinstance(v,(int,float)) else '-')(_idea.get('tp')))}"
                ))((_idea.get('precision') if isinstance(_idea, dict) and isinstance(_idea.get('precision'), int) else price_decimals(sym)))
            ) if idea else ("Entry: -","Stoploss: -","Takeprofit: -"))(idea)[1],
            (lambda _idea: (
                (lambda decs: (
                    f"Entry: {((lambda v: f'{v:.{decs}f}' if isinstance(v,(int,float)) else '-')(_idea.get('entry')))}",
                    f"Stoploss: {((lambda v: f'{v:.{decs}f}' if isinstance(v,(int,float)) else '-')(_idea.get('sl')))}",
                    f"Takeprofit: {((lambda v: f'{v:.{decs}f}' if isinstance(v,(int,float)) else '-')(_idea.get('tp')))}"
                ))((_idea.get('precision') if isinstance(_idea, dict) and isinstance(_idea.get('precision'), int) else price_decimals(sym)))
            ) if idea else ("Entry: -","Stoploss: -","Takeprofit: -"))(idea)[2],
            "",
            "Phân tích kỹ thuật:",
            f"  - Xu hướng: {trend_word} ({tf_trend})",
        ]
        # SR lines with breakout note if any
        # Support / Resistance lines + breakout detection (also parse sr_event variants)
        def _sr_break_flags(ev: Optional[str]) -> Tuple[bool,bool]:
            if not isinstance(ev,str):
                return False, False
            low = ev.lower()
            brk_down = any(k in low for k in ("breakout_down","breakdown","break-down","break down"))
            brk_up = any(k in low for k in ("breakout_up","breakoutup","break-up","break up","breakout_upward"))
            return brk_down, brk_up
        brk_down, brk_up = _sr_break_flags(sr_event)
        # Enhanced dynamic Support / Resistance resolution with quality filtering
        def _load_sr_candidates(symbol: str) -> Tuple[list[float], list[float]]:
            sup_list: list[float] = []
            res_list: list[float] = []
            try:
                tf_priority = ["H1","M30","M15","H4","D1"]  # H1 first for best accuracy
                import json as _json
                for tfc in tf_priority:
                    fp = os.path.join(CFG.SR, f"{symbol}_m_{tfc}_trendline_sr.json")
                    if not os.path.exists(fp):
                        continue
                    try:
                        with open(fp,'r',encoding='utf-8') as _f:
                            data = _json.load(_f)
                        if isinstance(data, dict):
                            # Load supports with strength info for quality filtering
                            supports_info = data.get("supports_info", [])
                            resistances_info = data.get("resistances_info", [])
                            
                            # Filter by strength (min touches >= 2, good distance)
                            for sup_info in supports_info:
                                if isinstance(sup_info, dict):
                                    level = sup_info.get("level")
                                    touches = sup_info.get("touches", 0)
                                    if isinstance(level, (int, float)) and touches >= 2:
                                        sup_list.append(float(level))
                            
                            for res_info in resistances_info:
                                if isinstance(res_info, dict):
                                    level = res_info.get("level")
                                    touches = res_info.get("touches", 0)
                                    if isinstance(level, (int, float)) and touches >= 2:
                                        res_list.append(float(level))
                            
                            # Fallback to simple arrays if no info available
                            if not sup_list and not res_list:
                                for kname, target in (("supports", sup_list),("support", sup_list),("resistances", res_list),("resistance", res_list)):
                                    vals = data.get(kname)
                                    if isinstance(vals, (list, tuple)):
                                        for v in vals:
                                            if isinstance(v,(int,float)):
                                                target.append(float(v))
                    except Exception:
                        continue
            except Exception:
                pass
            
            # Remove duplicates and sort
            sup_list = sorted(list(set(sup_list)), reverse=True)  # Highest support first
            res_list = sorted(list(set(res_list)))  # Lowest resistance first
            
            return sup_list, res_list

        # Safe current price resolution (avoid NameError if local price variable not set)
        try:
            base_price = locals().get('price') if 'price' in locals() else analysis.get('price') if isinstance(analysis, dict) else None
        except Exception:
            base_price = None
        try:
            pt_price = locals().get('price_trend') if 'price_trend' in locals() else analysis.get('price_trend') if isinstance(analysis, dict) else None
        except Exception:
            pt_price = None
        cur_price = pt_price or base_price
        extra_sup_list: list[float] = []
        extra_res_list: list[float] = []
        if (sup is None or res is None) and isinstance(sym, str):
            extra_sup_list, extra_res_list = _load_sr_candidates(sym)
        def _pick_nearest(levels: list[float], below: bool) -> Optional[float]:
            if cur_price is None or not levels:
                return None
            
            # Filter levels within reasonable distance (max 0.5% for major pairs, 1% for others)
            max_distance_pct = 0.005 if any(pair in sym.upper() for pair in ['EUR', 'GBP', 'USD', 'JPY']) else 0.01
            
            filt = []
            for lv in levels:
                if below and lv < cur_price:
                    distance_pct = abs(cur_price - lv) / cur_price
                    if distance_pct <= max_distance_pct:
                        filt.append((lv, distance_pct))
                elif not below and lv > cur_price:
                    distance_pct = abs(lv - cur_price) / cur_price
                    if distance_pct <= max_distance_pct:
                        filt.append((lv, distance_pct))
            
            if not filt:
                # If no levels within reasonable distance, get closest but warn
                all_levels = [lv for lv in levels if (lv < cur_price if below else lv > cur_price)]
                if not all_levels:
                    return None
                closest = max(all_levels) if below else min(all_levels)
                distance_pct = abs(closest - cur_price) / cur_price
                # Only return if distance is less than 2%
                return closest if distance_pct < 0.02 else None
            
            # Sort by distance and return closest
            filt.sort(key=lambda x: x[1])
            return filt[0][0]
        chosen_sup = sup
        if chosen_sup is None and extra_sup_list:
            chosen_sup = _pick_nearest(extra_sup_list, below=True)
        chosen_res = res
        if chosen_res is None and extra_res_list:
            chosen_res = _pick_nearest(extra_res_list, below=False)
        # De-duplicate if sup/res collapse
        if chosen_sup is not None and chosen_res is not None and abs(chosen_res - chosen_sup) < 1e-9:
            chosen_res = None
        if chosen_sup is not None:
            lines.append((f"  - Vùng hỗ trợ: {pfmt(chosen_sup)} ({sr_tf})" if (sr_tf and sup is not None) else f"  - Vùng hỗ trợ: {pfmt(chosen_sup)}"))
            if brk_down and sup is not None:
                lines.append("        → CẢNH BÁO: Giá phá vỡ hỗ trợ")
        else:
            lines.append("  - Vùng hỗ trợ: (chưa phát hiện rõ ràng)")
        if chosen_res is not None:
            lines.append((f"  - Vùng kháng cự: {pfmt(chosen_res)} ({sr_tf})" if (sr_tf and res is not None) else f"  - Vùng kháng cự: {pfmt(chosen_res)}"))
            if brk_up and res is not None:
                lines.append("        → CẢNH BÁO: Giá phá vỡ kháng cự")
        else:
            lines.append("  - Vùng kháng cự: (chưa phát hiện rõ ràng)")
        # Channel with breakout annotation
        lines.append(f"  - Kênh giá: {chan_txt}")
        if chan_break_note:
            lines.append(f"        → {chan_break_note}")
        # Trendline with breakout annotation if flagged by SR
        lines.append(f"  - Trendline: {trendline_txt}")
        if trendline_break_note:
            lbl = trendline_break_note
            if lbl.startswith('BreakOut'):
                if 'đảo chiều' in lbl:
                    lines.append("        → CẢNH BÁO: Trendline phá vỡ lên (đảo chiều)")
                else:
                    lines.append("        → CẢNH BÁO: Trendline phá vỡ lên")
            elif lbl.startswith('BreakDown'):
                if 'đảo chiều' in lbl:
                    lines.append("        → CẢNH BÁO: Trendline phá vỡ xuống (đảo chiều)")
                else:
                    lines.append("        → CẢNH BÁO: Trendline phá vỡ xuống")
            else:
                lines.append(f"        → {lbl}")
        # Indicators section header
        lines.append(f"  - Chỉ báo kỹ thuật:")
        
        # Sort indicators according to GUI order
        gui_whitelist = analysis.get('_indicator_whitelist')
        if gui_whitelist and isinstance(gui_whitelist, (list, set)):
            # Create mapping from token to line
            token_to_line = {}
            remaining_lines = []
            
            for line in ind_lines:
                if '•' in line and not line.strip().startswith('→'):
                    try:
                        core = line.split('•',1)[1].strip()
                        base = core.split(':',1)[0].split('(',1)[0].strip().lower()
                        # Apply alias mapping
                        alias = {
                            'rsi':'rsi','rsi14':'rsi','macd':'macd','macd12_26':'macd','macd14_26':'macd',
                            'adx':'adx','adx14':'adx','stochastic':'stochastic','stochastic rsi':'stochrsi','stochrsi':'stochrsi','stochastic_rsi':'stochrsi',
                            'atr':'atr','atr14':'atr','bollinger':'bollinger','bollinger bands':'bollinger','bb':'bollinger',
                            'donchian':'donchian','donchian channel':'donchian','keltner':'keltner','keltner channel':'keltner',
                            'ichimoku':'ichimoku','ichimoku cloud':'ichimoku','cci':'cci','cci20':'cci','williams %r':'williamsr','williamsr':'williamsr','willr':'williamsr',
                            'roc':'roc','rate of change':'roc','obv':'obv','on balance volume':'obv',
                            'chaikin osc':'chaikin','chaikin':'chaikin','chaikin money flow':'chaikin','cmf':'chaikin',
                            'eom':'eom','ease of movement':'eom','force index':'force','force':'force','trix':'trix','dpo':'dpo','detrended price oscillator':'dpo',
                            'mass index':'mass','mass':'mass','vortex':'vortex','vortex indicator':'vortex','vi':'vortex',
                            'kst':'kst','know sure thing':'kst','ultimate':'ultimate','ultimate oscillator':'ultimate',
                            'envelopes':'envelopes','envelope':'envelopes','env':'envelopes','momentum':'momentum','psar':'psar','parabolic sar':'psar',
                            'mfi':'mfi','money flow index':'mfi'
                        }
                        token = alias.get(base, base)
                        token_to_line[token] = line
                    except Exception:
                        remaining_lines.append(line)
                else:
                    remaining_lines.append(line)
            
            # Sort according to GUI whitelist order
            sorted_lines = []
            gui_order = list(gui_whitelist) if isinstance(gui_whitelist, set) else gui_whitelist
            
            for token in gui_order:
                if token in token_to_line:
                    sorted_lines.append(token_to_line[token])
                    # Add any sub-lines (starting with →) that follow this indicator
                    try:
                        original_idx = ind_lines.index(token_to_line[token])
                        for i in range(original_idx + 1, len(ind_lines)):
                            if ind_lines[i].strip().startswith('→'):
                                sorted_lines.append(ind_lines[i])
                            else:
                                break
                    except ValueError:
                        pass  # Line not found in original list
            
            # Add any remaining lines that couldn't be matched
            for line in remaining_lines:
                if line not in sorted_lines:
                    sorted_lines.append(line)
            
            lines += sorted_lines
        else:
            lines += ind_lines
        lines += ["", *summary_lines, ""]
        # Pattern & candlestick sections (scenario removed)
        # Enhanced pattern & candlestick extraction with direction arrows & confidence tiers
        patt_lines: List[str] = []
        candle_lines: List[str] = []
        def _conf_norm(v):
            try:
                x = float(v)
                if x <= 1.5: x *= 100
                return max(0.0, min(100.0, x))
            except Exception:
                return 0.0
        def _tier(c: float) -> str:
            return 'rất cao' if c>=85 else 'cao' if c>=70 else 'khá' if c>=55 else 'trung bình' if c>=40 else 'thấp'
        def _price_pattern_meta(name: str) -> Tuple[str,str]:
            n = name.lower()
            if 'falling wedge' in n: return '↑','Tích lũy/đảo chiều tăng'
            if 'rising wedge' in n: return '↓','Tích lũy/đảo chiều giảm'
            if 'ascending triangle' in n: return '↑','Tiếp diễn tăng'
            if 'descending triangle' in n: return '↓','Tiếp diễn giảm'
            if 'triangle' in n: return '→','Tích lũy'
            if 'head and shoulders' in n: return '↓','Đảo chiều giảm'
            if 'inverse head' in n: return '↑','Đảo chiều tăng'
            if 'double top' in n: return '↓','Đảo chiều giảm'
            if 'double bottom' in n: return '↑','Đảo chiều tăng'
            if 'channel' in n and 'ascending' in n: return '↑','Kênh tăng'
            if 'channel' in n and 'descending' in n: return '↓','Kênh giảm'
            if 'rectangle' in n: return '→','Đi ngang'
            if 'flag' in n and 'bull' in n: return '↑','Tiếp diễn tăng'
            if 'flag' in n and 'bear' in n: return '↓','Tiếp diễn giảm'
            if 'pennant' in n and 'bull' in n: return '↑','Tiếp diễn tăng'
            if 'pennant' in n and 'bear' in n: return '↓','Tiếp diễn giảm'
            if 'cup and handle' in n: return '↑','Tiếp diễn tăng'
            return '→','Mô hình'
        def _candle_meta(name: str, pol: str) -> Tuple[str,str]:
            n = name.lower(); p = pol.lower()
            if 'engulfing' in n and 'bull' in p: return '↑','Đảo chiều tăng'
            if 'engulfing' in n and 'bear' in p: return '↓','Đảo chiều giảm'
            if 'pin bar' in n: return ('↑' if p.startswith('bull') else '↓' if p.startswith('bear') else '→','Phản ứng giá')
            if 'hammer' in n and 'inverted' not in n: return '↑','Rũ bỏ đáy'
            if 'shooting star' in n or 'inverted hammer' in n: return '↓','Suy yếu đỉnh'
            if 'morning star' in n: return '↑','Đảo chiều tăng'
            if 'evening star' in n: return '↓','Đảo chiều giảm'
            if 'harami' in n: return ('↑' if 'bull' in p else '↓' if 'bear' in p else '→','Harami')
            if 'doji' in n: return '→','Do dự'
            if 'inside bar' in n: return '→','Tích lũy'
            if 'outside bar' in n: return '→','Biến động'
            return ('↑' if p.startswith('bull') else '↓' if p.startswith('bear') else '→','Mô hình')
        try:
            import os
            import glob
            logger.debug("[REPORT] Starting pattern aggregation for %s", sym)
            # Check what timeframes have patterns available
            available_pattern_tfs = []
            import glob
            
            # Check pattern_price folder for available timeframes
            price_pattern_files = glob.glob(f"pattern_price/{sym}_H1_*.json") + glob.glob(f"pattern_price/{sym}_*_patterns.json")
            for f in price_pattern_files:
                try:
                    if '_H1_' in f:
                        available_pattern_tfs.append('H1')
                    elif '_H4_' in f:
                        available_pattern_tfs.append('H4')
                    elif '_D1_' in f:
                        available_pattern_tfs.append('D1')
                    elif '_M30_' in f:
                        available_pattern_tfs.append('M30')
                    elif '_M15_' in f:
                        available_pattern_tfs.append('M15')
                    elif '_M5_' in f:
                        available_pattern_tfs.append('M5')
                except:
                    pass
            
            # Check pattern_signals folder too
            signal_pattern_files = glob.glob(f"pattern_signals/{sym}_H1_*.json") + glob.glob(f"pattern_signals/{sym}_*_patterns.json")  
            for f in signal_pattern_files:
                try:
                    if '_H1_' in f:
                        if 'H1' not in available_pattern_tfs:
                            available_pattern_tfs.append('H1')
                    elif '_H4_' in f:
                        if 'H4' not in available_pattern_tfs:
                            available_pattern_tfs.append('H4')
                    elif '_D1_' in f:
                        if 'D1' not in available_pattern_tfs:
                            available_pattern_tfs.append('D1')
                    elif '_M30_' in f:
                        if 'M30' not in available_pattern_tfs:
                            available_pattern_tfs.append('M30')
                    elif '_M15_' in f:
                        if 'M15' not in available_pattern_tfs:
                            available_pattern_tfs.append('M15')
                    elif '_M5_' in f:
                        if 'M5' not in available_pattern_tfs:
                            available_pattern_tfs.append('M5')
                except:
                    pass
            
            available_pattern_tfs = list(set(available_pattern_tfs))  # remove duplicates
            
            # If we found pattern files, load them directly instead of relying on tf_data_cache
            tf_pref = available_pattern_tfs if available_pattern_tfs else tf_ordered()
            # Collect price pattern candidates
            cand_price: List[Tuple[float,int,str]] = []  # (conf, tf_rank, line)
            for idx, tfp in enumerate(tf_pref):
                td = tf_data_cache.get(tfp)
                
                # If no cached data, try to load patterns directly
                if not td or not td.price_patterns:
                    # Try loading patterns directly from files
                    direct_patterns = None
                    try:
                        # Try pattern_price folder first
                        pattern_files = [
                            f"pattern_price/{sym}_{tfp}_patterns.json",
                            f"pattern_price/{sym}_{tfp}_best.json",
                            f"pattern_price/{sym}_m_{tfp}_patterns.json",
                        ]
                        for pf in pattern_files:
                            if os.path.exists(pf):
                                direct_patterns = load_json(pf)
                                break
                                
                        # If not found, try pattern_signals folder
                        if not direct_patterns:
                            signal_files = [
                                f"pattern_signals/{sym}_{tfp}_priority_patterns.json",
                                f"pattern_signals/{sym}_{tfp}_patterns.json",
                            ]
                            for sf in signal_files:
                                if os.path.exists(sf):
                                    direct_patterns = load_json(sf)
                                    print(f"DEBUG [PATTERN]: Loaded direct signals from {sf}")
                                    break
                                    
                        if direct_patterns:
                            # Create a temporary TFData object
                            td = TFData(candles=None, indicators=None, price_patterns=direct_patterns, priority_patterns=None, sr=None)
                            
                    except Exception as e:
                        pass
                        
                if not td or not td.price_patterns: continue
                try:
                    if isinstance(td.price_patterns, list):
                        logger.debug("[REPORT] %s %s price_patterns=%d", sym, tfp, len(td.price_patterns))
                    else:
                        logger.debug("[REPORT] %s %s price_patterns(type=%s)", sym, tfp, type(td.price_patterns))
                except Exception:
                    pass
                plist = td.price_patterns
                seq: List[dict] = []
                if isinstance(plist, list):
                    seq = [p for p in plist if isinstance(p, dict)]
                elif isinstance(plist, dict):
                    if isinstance(plist.get('patterns'), list):
                        seq = [p for p in plist.get('patterns') if isinstance(p, dict)]
                    elif any(k in plist for k in ('type','pattern')):
                        seq = [plist]
                for p in seq[:5]:  # limit per TF
                    ptype = (p.get('type') or p.get('pattern') or '').replace('_',' ')
                    if not ptype: continue
                    conf = _conf_norm(p.get('confidence') or p.get('confidence_pct') or p.get('score'))
                    arrow, meta = _price_pattern_meta(ptype)
                    cand_price.append((conf, idx, f"        • {ptype} {arrow} ({tfp}) – {meta} – {int(round(conf))}% ({_tier(conf)})"))
            # Rank: confidence desc then tf order
            cand_price.sort(key=lambda x: (-x[0], x[1]))
            seen_base = set()
            for conf, idx, line in cand_price:
                base = line.split('–')[0].strip()
                if base not in seen_base:
                    patt_lines.append(line)
                    seen_base.add(base)
                if len(patt_lines) >= 6: break
            # Fallback aggregate if none
            if not patt_lines:
                try:
                    tflist_agg = [tf for tf in tf_pref if tf in tf_data_cache]
                    from math import isnan
                    # attempt simple majority of pattern directions
                    dirs: List[int] = []
                    for conf, idx, line in cand_price:
                        if '↑' in line: dirs.append(1)
                        elif '↓' in line: dirs.append(-1)
                    if dirs:
                        s = sum(dirs)
                        direction = 'nghiêng tăng' if s>0 else 'nghiêng giảm' if s<0 else 'trung tính'
                        patt_lines.append(f"        • Tổng hợp: {direction} ({len(dirs)} mẫu)")
                except Exception:
                    pass
            # Candlestick patterns
            cand_candles: List[Tuple[float,int,str]] = []
            for idx, tfp in enumerate(tf_pref):
                td = tf_data_cache.get(tfp)
                
                # If no cached data, try to load candlestick patterns directly
                if not td or not td.priority_patterns:
                    # Try loading candlestick patterns directly from files
                    direct_candle_patterns = None
                    try:
                        # Try pattern_signals folder first
                        signal_files = [
                            f"pattern_signals/{sym}_{tfp}_priority_patterns.json",
                            f"pattern_signals/{sym}_{tfp}_patterns.json",
                            f"pattern_signals/{sym}_m_{tfp}_priority_patterns.json",
                        ]
                        for sf in signal_files:
                            if os.path.exists(sf):
                                direct_candle_patterns = load_json(sf)
                                break
                                
                        if direct_candle_patterns:
                            # Create a temporary TFData object or update existing one
                            if td:
                                # Update existing td with priority_patterns
                                td = TFData(candles=td.candles, indicators=td.indicators, 
                                          price_patterns=td.price_patterns, priority_patterns=direct_candle_patterns, sr=td.sr)
                            else:
                                # Create new td with just priority_patterns
                                td = TFData(candles=None, indicators=None, price_patterns=None, 
                                          priority_patterns=direct_candle_patterns, sr=None)
                            
                    except Exception as e:
                        pass
                        
                if not td or not td.priority_patterns: continue
                cp = td.priority_patterns
                try:
                    if isinstance(cp, list):
                        logger.debug("[REPORT] %s %s candle_patterns=%d", sym, tfp, len(cp))
                except Exception:
                    pass
                seq: List[dict] = []
                if isinstance(cp, list):
                    seq = [c for c in cp if isinstance(c, dict)]
                elif isinstance(cp, dict) and isinstance(cp.get('patterns'), list):
                    seq = [c for c in cp.get('patterns') if isinstance(c, dict)]
                for c in seq[:5]:
                    ctype = (c.get('type') or c.get('pattern') or '').replace('_',' ')
                    pol = (c.get('signal') or c.get('direction') or '').capitalize()
                    if not ctype: continue
                    conf = _conf_norm(c.get('confidence') or c.get('score'))
                    arrow, meta = _candle_meta(ctype, pol)
                    cand_candles.append((conf, idx, f"        • {ctype} {pol} {arrow} ({tfp}) – {meta} – {int(round(conf))}% ({_tier(conf)})"))
            cand_candles.sort(key=lambda x: (-x[0], x[1]))
            seen_cand = set()
            for conf, idx, line in cand_candles:
                base = line.split('–')[0].strip()
                if base not in seen_cand:
                    candle_lines.append(line)
                    seen_cand.add(base)
                if len(candle_lines) >= 6: break
        except Exception as e:
            try:
                logger.error("[REPORT] Pattern aggregation failed for %s: %s", sym, e, exc_info=True)
            except Exception:
                pass
        lines.append("  - Mô hình giá:")
        if not patt_lines:
            try:
                logger.debug("[REPORT] No price patterns aggregated for %s; cand_price_raw=%d", sym, len(cand_price) if 'cand_price' in locals() else -1)
            except Exception:
                pass
        lines += (patt_lines if patt_lines else ["        • -"])
        lines.append("  - Mô hình nến:")
        # Deduplicate candle lines
        if candle_lines:
            seen_c: set[str] = set()
            dedup_c: list[str] = []
            for ln in candle_lines:
                k = ln.strip()
                if k not in seen_c:
                    dedup_c.append(ln)
                    seen_c.add(k)
            candle_lines = dedup_c
        if not candle_lines:
            # Derive simple + advanced patterns from synthesized candles (indicator-derived) if available
            try:
                # Pick first TF that has candles in cache
                sample_tf = None
                candles_sample = None
                for _tf in ["H4","H1","M30","M15","D1","M5"]:
                    td = tf_data_cache.get(_tf)
                    if td and isinstance(td.candles, list) and len(td.candles) >= 5:
                        sample_tf = _tf
                        candles_sample = td.candles[-15:]
                        break
                if sample_tf and candles_sample:
                    def _num(v):
                        try: return float(v)
                        except: return None
                    derived: list[str] = []
                    seen_code: set[str] = set()
                    # Pairwise patterns
                    for i in range(1, len(candles_sample)):
                        prev = candles_sample[i-1]; cur = candles_sample[i]
                        if not (isinstance(prev, dict) and isinstance(cur, dict)): continue
                        o1=_num(prev.get('open') or prev.get('o')); c1=_num(prev.get('close') or prev.get('c'))
                        o2=_num(cur.get('open') or cur.get('o')); c2=_num(cur.get('close') or cur.get('c'))
                        h2=_num(cur.get('high') or cur.get('h')); l2=_num(cur.get('low') or cur.get('l'))
                        if None in (o1,c1,o2,c2,h2,l2): continue
                        body1=abs(c1-o1); body2=abs(c2-o2); range2=abs(h2-l2) or 1e-9
                        upper_wick = h2 - max(o2,c2)
                        lower_wick = min(o2,c2) - l2
                        # Doji
                        if body2/range2 < 0.1 and 'doji' not in seen_code:
                            derived.append(f"        • Doji (tự suy luận) ({sample_tf}) – lưỡng lự")
                            seen_code.add('doji')
                        # Engulfing
                        if c2>o2 and c1<o1 and o2< c1 and c2> o1 and 'bull_engulf' not in seen_code:
                            derived.append(f"        • Bullish engulfing (tự suy luận) ({sample_tf}) – đảo chiều tăng")
                            seen_code.add('bull_engulf')
                        if c2<o2 and c1>o1 and o2> c1 and c2< o1 and 'bear_engulf' not in seen_code:
                            derived.append(f"        • Bearish engulfing (tự suy luận) ({sample_tf}) – đảo chiều giảm")
                            seen_code.add('bear_engulf')
                        # Pin bar variants
                        if body2>0 and upper_wick> body2*2 and lower_wick < body2*0.5 and 'shooting_star' not in seen_code:
                            derived.append(f"        • Shooting star (tự suy luận) ({sample_tf}) – cảnh báo giảm")
                            seen_code.add('shooting_star')
                        if body2>0 and lower_wick> body2*2 and upper_wick < body2*0.5 and 'hammer' not in seen_code:
                            derived.append(f"        • Hammer (tự suy luận) ({sample_tf}) – tiềm năng bật lên")
                            seen_code.add('hammer')
                        if len(derived) >= 6:
                            break
                    # Advanced 3-candle patterns
                    try:
                        if len(candles_sample) >= 5 and len(derived) < 6:
                            cs = candles_sample
                            def body(b):
                                return abs(_num(b.get('close') or b.get('c')) - _num(b.get('open') or b.get('o')))
                            def bull(b):
                                return _num(b.get('close') or b.get('c')) > _num(b.get('open') or b.get('o'))
                            def bear(b):
                                return _num(b.get('close') or b.get('c')) < _num(b.get('open') or b.get('o'))
                            last3 = cs[-3:]
                            if all(isinstance(x, dict) for x in last3):
                                b1,b2,b3 = last3
                                try:
                                    if bear(b1) and bull(b3) and body(b1) and body(b3) and body(b2) < body(b1)*0.6 and body(b2) < body(b3)*0.6 and _num(b3.get('close') or b3.get('c')) > (_num(b1.get('open') or b1.get('o')) + _num(b1.get('close') or b1.get('c')))/2 and 'morning_star' not in seen_code:
                                        derived.append(f"        • Morning star (tự suy luận) ({sample_tf}) – đảo chiều tăng")
                                        seen_code.add('morning_star')
                                except Exception: pass
                                try:
                                    if bull(b1) and bear(b3) and body(b1) and body(b3) and body(b2) < body(b1)*0.6 and body(b2) < body(b3)*0.6 and _num(b3.get('close') or b3.get('c')) < (_num(b1.get('open') or b1.get('o')) + _num(b1.get('close') or b1.get('c')))/2 and 'evening_star' not in seen_code:
                                        derived.append(f"        • Evening star (tự suy luận) ({sample_tf}) – đảo chiều giảm")
                                        seen_code.add('evening_star')
                                except Exception: pass
                            last4 = cs[-4:]
                            if all(isinstance(b, dict) for b in last4):
                                bA,bB,bC,bD = last4
                                try:
                                    if bull(bB) and bull(bC) and bull(bD) and 'three_white_soldiers' not in seen_code:
                                        cseq = [_num(x.get('close') or x.get('c')) for x in (bB,bC,bD)]
                                        oseq = [_num(x.get('open') or x.get('o')) for x in (bB,bC,bD)]
                                        if None not in cseq+oseq and cseq[0]<cseq[1]<cseq[2] and oseq[0]<=oseq[1]<=oseq[2]:
                                            derived.append(f"        • Three white soldiers (tự suy luận) ({sample_tf}) – xác nhận tăng")
                                            seen_code.add('three_white_soldiers')
                                    if bear(bB) and bear(bC) and bear(bD) and 'three_black_crows' not in seen_code:
                                        cseq = [_num(x.get('close') or x.get('c')) for x in (bB,bC,bD)]
                                        oseq = [_num(x.get('open') or x.get('o')) for x in (bB,bC,bD)]
                                        if None not in cseq+oseq and cseq[0]>cseq[1]>cseq[2] and oseq[0]>=oseq[1]>=oseq[2]:
                                            derived.append(f"        • Three black crows (tự suy luận) ({sample_tf}) – xác nhận giảm")
                                            seen_code.add('three_black_crows')
                                except Exception: pass
                            # Harami last 2
                            if len(cs) >= 2:
                                b_prev, b_cur = cs[-2], cs[-1]
                                try:
                                    o_prev=_num(b_prev.get('open') or b_prev.get('o')); c_prev=_num(b_prev.get('close') or b_prev.get('c'))
                                    o_cur=_num(b_cur.get('open') or b_cur.get('o')); c_cur=_num(b_cur.get('close') or b_cur.get('c'))
                                    if None not in (o_prev,c_prev,o_cur,c_cur):
                                        high_prev=max(o_prev,c_prev); low_prev=min(o_prev,c_prev)
                                        high_cur=max(o_cur,c_cur); low_cur=min(o_cur,c_cur)
                                        if low_cur >= low_prev and high_cur <= high_prev:
                                            if bear(b_prev) and bull(b_cur) and 'bull_harami' not in seen_code:
                                                derived.append(f"        • Bullish harami (tự suy luận) ({sample_tf}) – khả năng đảo chiều lên")
                                                seen_code.add('bull_harami')
                                            if bull(b_prev) and bear(b_cur) and 'bear_harami' not in seen_code:
                                                derived.append(f"        • Bearish harami (tự suy luận) ({sample_tf}) – khả năng đảo chiều xuống")
                                                seen_code.add('bear_harami')
                                except Exception: pass
                    except Exception:
                        pass
                    if derived:
                        candle_lines = derived[:6]
            except Exception:
                pass
        if not candle_lines:
            try:
                logger.debug("[REPORT] No candlestick patterns aggregated for %s; cand_candles_raw=%d", sym, len(cand_candles) if 'cand_candles' in locals() else -1)
            except Exception:
                pass
            # Explicit placeholder with reason
            candle_lines = ["        • (Không có mô hình nến – thiếu dữ liệu hoặc không phát hiện)"]
        lines += candle_lines
        return "\n".join(lines) + "\n"

    @staticmethod
    def build_bilingual(analysis: Dict[str, Any], indicator_whitelist: Optional[Union[set[str], list[str]]] = None) -> str:
        """Return VI-only content (English removed). If indicator_whitelist provided, filter output."""
        if indicator_whitelist:
            # store copy so build() can access - preserve order if list
            try:
                if isinstance(indicator_whitelist, list):
                    analysis['_indicator_whitelist'] = indicator_whitelist
                else:
                    analysis['_indicator_whitelist'] = set(indicator_whitelist)
            except Exception:
                pass
        return Report.build(analysis, lang="vi")
    # ...existing code...


# ------------------------------
# Data structures and Loader/Aggregator
# ------------------------------
@dataclass
class TFData:
    candles: Optional[Any]
    indicators: Optional[Any]
    price_patterns: Optional[Any]
    priority_patterns: Optional[Any]
    sr: Optional[Any]

class Loader:
    def __init__(self, symbol: str):
        self.symbol = symbol

    @staticmethod
    def detect_symbols() -> List[str]:
        # Try scanning all known data folders for symbol subdirectories
        syms: List[str] = []
        dbg_sources: List[str] = []
        for base in (CFG.IND, CFG.DATA, CFG.PPRICE, CFG.PSIG, CFG.SR):
            if not os.path.exists(base):
                continue
            names = os.listdir(base)
            for name in names:
                if name.startswith('.'):
                    continue
                p = os.path.join(base, name)
                if os.path.isdir(p):
                    syms.append(name)
                    dbg_sources.append(f"dir:{base}/{name}")
            # If base contains flat files like EURUSD_m_H1_indicators.json
            if base == CFG.IND and not syms:
                for name in names:
                    if not name.lower().endswith('.json'):
                        continue
                    # Supported indicator filename variants:
                    #   SYMBOL_m_TF_indicators.json
                    #   SYMBOL_TF_indicators.json
                    #   SYMBOL._TF_indicators.json (legacy bug with stray dot)
                    #   SYMBOL._m_TF_indicators.json (very rare legacy)
                    m1 = re.match(r"([A-Z0-9_]+)_m_([A-Z0-9]+)_indicators\.json", name, re.IGNORECASE)
                    m2 = re.match(r"([A-Z0-9_]+)_([A-Z0-9]+)_indicators\.json", name, re.IGNORECASE)
                    m3 = re.match(r"([A-Z0-9_]+)\._([A-Z0-9]+)_indicators\.json", name, re.IGNORECASE)
                    m4 = re.match(r"([A-Z0-9_]+)\._m_([A-Z0-9]+)_indicators\.json", name, re.IGNORECASE)
                    mm = m1 or m2 or m3 or m4
                    if mm:
                        symbol_raw = mm.group(1)
                        # Normalize stray trailing dots / underscores
                        symbol_norm = symbol_raw.rstrip('._')
                        if symbol_norm not in syms:
                            syms.append(symbol_norm)
                            dbg_sources.append(f"flat_ind:{name}")
        # Fallback: read account_scans list if available
        if not syms:
            try:
                fp = os.path.join(CFG.ACCT, 'mt5_essential_scan.json')
                if os.path.exists(fp):
                    data = load_json(fp)
                    if isinstance(data, list):
                        for it in data:
                            if isinstance(it, str):
                                syms.append(it)
                                dbg_sources.append("acct:str")
                            elif isinstance(it, dict):
                                s = it.get('symbol') or it.get('Symbol') or it.get('pair')
                                if isinstance(s, str):
                                    syms.append(s)
                                    dbg_sources.append("acct:dict")
            except Exception:
                pass
        # Deduplicate preserving order
        out: List[str] = []
        for s in syms:
            if s not in out:
                out.append(s)
        if out:
            logger.debug("Symbol detection sources: " + "; ".join(dbg_sources[:50]))
        return out

    def _find(self, base: str, symbol: str, tf: str, suffix: str) -> Optional[str]:
        # e.g., indicator_output/<symbol>/<tf>_indicators.json
        patts = [
            os.path.join(base, symbol, f"{tf}{suffix}"),
            os.path.join(base, symbol, f"{symbol}_{tf}{suffix}"),
        ]
        for p in patts:
            if os.path.exists(p):
                return p
        # try glob
        g = os.path.join(base, symbol, f"*{tf}*{suffix}*")
        files = glob.glob(g)
        return files[-1] if files else None

    def load_tf(self, tf: str) -> 'TFData':
        # candles
        candles = None
        cfp = self._find(CFG.DATA, self.symbol, tf, "_candles.json")
        if not cfp:
            cfp = self._find(CFG.DATA, self.symbol, tf, ".json")
        candles = load_json(cfp)
        # indicators
        ind = None
        if os.path.exists(os.path.join(CFG.IND, self.symbol)):
            # prefer list-of-bars JSON.gz created by exporter
            patt = os.path.join(CFG.IND, self.symbol, f"*{tf}*ind*.json*")
            cands = glob.glob(patt)
            if cands:
                ind = load_json(sorted(cands)[-1])
        # flat files e.g. indicator_output/EURUSD_m_H1_indicators.json(.gz)
        if ind is None and os.path.exists(CFG.IND):
            try:
                names = os.listdir(CFG.IND)
                # Common patterns: SYMBOL_m_TF_indicators.json or SYMBOL_TF_indicators.json
                patt1 = re.compile(rf"^{re.escape(self.symbol)}_m_{re.escape(tf)}_indicators\.json(\.gz)?$", re.IGNORECASE)
                patt2 = re.compile(rf"^{re.escape(self.symbol)}_{re.escape(tf)}_indicators\.json(\.gz)?$", re.IGNORECASE)
                # Legacy variant with stray dot: SYMBOL._TF_indicators.json
                patt3 = re.compile(rf"^{re.escape(self.symbol)}\._{re.escape(tf)}_indicators\.json(\.gz)?$", re.IGNORECASE)
                patt4 = re.compile(rf"^{re.escape(self.symbol)}\._m_{re.escape(tf)}_indicators\.json(\.gz)?$", re.IGNORECASE)
                for name in names:
                    if patt1.match(name) or patt2.match(name) or patt3.match(name) or patt4.match(name):
                        ind = load_json(os.path.join(CFG.IND, name))
                        break
                # As a fallback, pick any file that contains both symbol and tf and 'indicators'
                if ind is None:
                    for name in names:
                        low = name.lower()
                        if self.symbol.lower() in low and tf.lower() in low and 'indicator' in low and name.lower().endswith('.json'):
                            ind = load_json(os.path.join(CFG.IND, name))
                            break
            except Exception:
                ind = None
        # price patterns
        price_patterns = None
        ppf = self._find(CFG.PPRICE, self.symbol, tf, "_best.json")
        price_patterns = load_json(ppf) if ppf else None
        # Fallback: many installs only have *_patterns.json (not *_best.json)
        if price_patterns is None:
            try:
                # Candidate flat filenames under pattern_price root
                cand_names = [
                    f"{self.symbol}_m_{tf}_patterns.json",
                    f"{self.symbol}_{tf}_patterns.json",
                    f"{self.symbol}._{tf}_patterns.json",  # legacy stray dot variant
                ]
                for name in cand_names:
                    p = os.path.join(CFG.PPRICE, name)
                    if os.path.exists(p):
                        price_patterns = load_json(p)
                        break
                # Also check symbol subfolder if exists
                if price_patterns is None and os.path.isdir(os.path.join(CFG.PPRICE, self.symbol)):
                    base_dir = os.path.join(CFG.PPRICE, self.symbol)
                    for name in cand_names:
                        p = os.path.join(base_dir, name)
                        if os.path.exists(p):
                            price_patterns = load_json(p)
                            break
                # As last resort, glob any file containing tf & 'patterns'
                if price_patterns is None:
                    gpatts = [
                        os.path.join(CFG.PPRICE, f"*{self.symbol}*{tf}*patterns.json*"),
                        os.path.join(CFG.PPRICE, self.symbol, f"*{tf}*patterns.json*"),
                        os.path.join(CFG.PPRICE, f"*{self.symbol}._{tf}*patterns.json*"),  # legacy
                    ]
                    found: list[str] = []
                    for gp in gpatts:
                        found.extend(glob.glob(gp))
                    if found:
                        found.sort(key=lambda fp: os.path.getmtime(fp))
                        price_patterns = load_json(found[-1])
            except Exception:
                price_patterns = price_patterns  # keep None silently
        # Extra fallback: some datasets embed '_m' in symbol filenames while analysis symbol omits it
        if price_patterns is None and not self.symbol.endswith('_m'):
            try:
                alt_symbol = f"{self.symbol}_m"
                cand_names_alt = [
                    f"{alt_symbol}_{tf}_best.json",
                    f"{alt_symbol}_{tf}_patterns.json",
                    f"{alt_symbol}_m_{tf}_patterns.json",  # defensive
                ]
                for name in cand_names_alt:
                    p = os.path.join(CFG.PPRICE, name)
                    if os.path.exists(p):
                        price_patterns = load_json(p)
                        break
                if price_patterns is None:
                    # glob any alt symbol + tf patterns
                    gpatts_alt = [
                        os.path.join(CFG.PPRICE, f"*{alt_symbol}*{tf}*patterns.json*"),
                    ]
                    found_alt: list[str] = []
                    for gp in gpatts_alt:
                        found_alt.extend(glob.glob(gp))
                    if found_alt:
                        found_alt.sort(key=lambda fp: os.path.getmtime(fp))
                        price_patterns = load_json(found_alt[-1])
            except Exception:
                pass
        # priority candle patterns
        prf = self._find(CFG.PSIG, self.symbol, tf, "_priority_patterns.json")
        priority_patterns = load_json(prf) if prf else None
        # Additional flat filename variants (legacy stray dot): SYMBOL._TF_priority_patterns.json
        if priority_patterns is None:
            try:
                legacy_name = f"{self.symbol}._{tf}_priority_patterns.json"
                lp = os.path.join(CFG.PSIG, legacy_name)
                if os.path.exists(lp):
                    priority_patterns = load_json(lp)
            except Exception:
                pass
        if priority_patterns is None and not self.symbol.endswith('_m'):
            try:
                alt_symbol = f"{self.symbol}_m"
                cand_names_alt = [
                    f"{alt_symbol}_{tf}_priority_patterns.json",
                    f"{alt_symbol}_m_{tf}_priority_patterns.json",  # defensive
                    f"{alt_symbol}._{tf}_priority_patterns.json",  # legacy stray dot variant
                ]
                for name in cand_names_alt:
                    p = os.path.join(CFG.PSIG, name)
                    if os.path.exists(p):
                        priority_patterns = load_json(p)
                        break
                if priority_patterns is None:
                    gpatts_alt = [
                        os.path.join(CFG.PSIG, f"*{alt_symbol}*{tf}*priority_patterns.json*"),
                        os.path.join(CFG.PSIG, f"*{alt_symbol}._{tf}*priority_patterns.json*"),
                    ]
                    found_alt: list[str] = []
                    for gp in gpatts_alt:
                        found_alt.extend(glob.glob(gp))
                    if found_alt:
                        found_alt.sort(key=lambda fp: os.path.getmtime(fp))
                        priority_patterns = load_json(found_alt[-1])
            except Exception:
                pass
        # SR/trendline
        sr = None
        srf = self._find(CFG.SR, self.symbol, tf, "_sr.json")
        sr = load_json(srf) if srf else None
        # Fallback: many installations store flat files like SYMBOL_TF_trendline_sr.json directly under trendline_sr/
        if sr is None:
            try:
                base = CFG.SR
                # Candidate exact filenames (flat structure, no symbol subfolder)
                candidates = [
                    f"{self.symbol}_{tf}_trendline_sr.json",
                    f"{self.symbol}_{tf}_trendline.json",
                    f"{self.symbol}_{tf}_sr.json",
                    f"{self.symbol}_{tf}_support_resistance.json",
                    f"{self.symbol}._{tf}_trendline_sr.json",  # legacy stray dot variant
                ]
                for name in candidates:
                    p = os.path.join(base, name)
                    if os.path.exists(p):
                        sr = load_json(p)
                        if isinstance(sr, dict):
                            break
                # Glob fallback (broader match) if still not found
                if sr is None or not isinstance(sr, dict):
                    patterns = [
                        os.path.join(base, f"{self.symbol}_{tf}_*trendline*sr*.json*"),
                        os.path.join(base, f"*{self.symbol}*_{tf}_trendline_sr.json*"),
                        os.path.join(base, f"*{self.symbol}*{tf}*trendline*sr*.json*"),
                        os.path.join(base, f"*{self.symbol}._{tf}*trendline*sr*.json*"),
                    ]
                    found: list[str] = []
                    for gpat in patterns:
                        found.extend(glob.glob(gpat))
                    if found:
                        # pick the latest modified
                        found.sort(key=lambda fp: os.path.getmtime(fp))
                        sr = load_json(found[-1])
            except Exception:
                sr = sr  # keep None silently
        try:
            pc_count = len(price_patterns) if isinstance(price_patterns, list) else (
                len(price_patterns.get('patterns')) if isinstance(price_patterns, dict) and isinstance(price_patterns.get('patterns'), list) else -1)
            pr_count = len(priority_patterns) if isinstance(priority_patterns, list) else (
                len(priority_patterns.get('patterns')) if isinstance(priority_patterns, dict) and isinstance(priority_patterns.get('patterns'), list) else -1)
            logger.debug(f"[LOAD_TF] {self.symbol} {tf}: price_patterns={pc_count} priority_patterns={pr_count}")
        except Exception:
            pass
        # Normalize flat indicator list-of-bars keys: add lowercase convenience duplicates if missing
        try:
            if isinstance(ind, list) and ind:
                last = ind[-1]
                if isinstance(last, dict):
                    # If RSI14 present but RSI absent, duplicate for downstream generic scans
                    if 'RSI14' in last and 'RSI' not in last:
                        for row in ind:
                            if isinstance(row, dict) and 'RSI14' in row and 'RSI' not in row:
                                row['RSI'] = row['RSI14']
                    # Normalize MACD histogram naming
                    if 'MACD_hist_12_26_9' in last and 'MACD_hist' not in last:
                        for row in ind:
                            if isinstance(row, dict) and 'MACD_hist_12_26_9' in row and 'MACD_hist' not in row:
                                row['MACD_hist'] = row['MACD_hist_12_26_9']
                    # Normalize ATR
                    if 'ATR14' in last and 'ATR_14' not in last:
                        for row in ind:
                            if isinstance(row, dict) and 'ATR14' in row and 'ATR_14' not in row:
                                row['ATR_14'] = row['ATR14']
                    # Normalize stochastic
                    if 'StochK_14_3' in last and 'StochK' not in last:
                        for row in ind:
                            if isinstance(row, dict) and 'StochK_14_3' in row and 'StochK' not in row:
                                row['StochK'] = row['StochK_14_3']
                    if 'StochD_14_3' in last and 'StochD' not in last:
                        for row in ind:
                            if isinstance(row, dict) and 'StochD_14_3' in row and 'StochD' not in row:
                                row['StochD'] = row['StochD_14_3']
        except Exception:
            pass
        return TFData(candles=candles, indicators=ind, price_patterns=price_patterns, priority_patterns=priority_patterns, sr=sr)

# Import smart entry calculation from separate module
from calculate_smart_entry import calculate_smart_entry

# Import risk-aware action generator
# RiskAwareActionGenerator integration moved to risk_manager.py

# Import trading analyst for duplicate checking
from trading_analyst import TradingAnalyst

class Aggregator:
    def __init__(self, symbol: str, indicators: Optional[List[str]] = None, whitelist: Optional[Set[str]] = None, 
                 user_timeframes: Optional[List[str]] = None, dca_settings: Optional[Dict] = None, current_price: Optional[float] = None):
        self.symbol = symbol
        # indicators param kept for backward compat (unused)
        # whitelist: normalized set of indicator tokens selected by user (rsi, macd, ema20, atr, bollinger, donchian, patterns, adx, stochrsi, stochastic, psar, fibonacci)
        try:
            self.whitelist = {w.strip().lower() for w in (whitelist or set()) if isinstance(w, str)} or None
        except Exception:
            self.whitelist = None
        self.user_timeframes = user_timeframes  # User-specified timeframes
        self.dca_settings = dca_settings or {}  # DCA configuration
        
        # DCA settings will be synced from risk_settings after loading
        self.current_price = current_price  # Current price extracted from indicators
        
        # Load risk settings from file
        self.risk_settings = self._load_risk_settings()
        self._sync_dca_settings()

    def _sync_dca_settings(self):
        """Sync DCA settings from risk_settings"""
        if not self.dca_settings:
            self.dca_settings = {}
        
        # Sync key DCA settings from risk_settings
        self.dca_settings.update({
            'enable_dca': self.risk_settings.get('enable_dca', False),
            'max_dca_levels': self.risk_settings.get('max_dca_levels', 3),
            'min_confidence_for_dca': 2.0,  # Very low threshold for DCA - matches current confidence levels
            'enable_smart_limits': True
        })
        
        logger.debug(f"Synced DCA settings: {self.dca_settings}")

    def _load_risk_settings(self) -> Dict:
        """🔄 Load and validate risk settings with comprehensive mode detection"""
        settings = {}
        
        try:
            # 1️⃣ Try loading from risk_management/risk_settings.json (primary source)
            with open("risk_management/risk_settings.json", 'r', encoding='utf-8') as f:
                settings = json.load(f)
            logger.info(f"✅ Risk settings loaded from risk_management/risk_settings.json")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load from risk_settings.json: {e}")
            
            # 2️⃣ Fallback: Try loading from user_config.pkl
            try:
                import pickle
                with open('user_config.pkl', 'rb') as f:
                    config = pickle.load(f)
                gui_settings = config.get('risk_management', {})
                if gui_settings:
                    settings.update(gui_settings)
                    logger.info(f"🔄 Fallback: Settings loaded from user_config.pkl")
            except Exception as e2:
                logger.warning(f"⚠️ Could not load from user_config.pkl: {e2}")
        
        # 3️⃣ Validate and enhance loaded settings
        settings = self._validate_and_enhance_settings(settings)
        
        # 4️⃣ Log comprehensive mode information
        self._log_comprehensive_mode_info(settings)
        
        return settings
    
    def _validate_and_enhance_settings(self, settings: Dict) -> Dict:
        """🔧 Validate and enhance settings with proper defaults"""
        
        # Core defaults
        defaults = {
            # Volume settings
            "volume_mode": "Khối lượng mặc định",
            "fixed_volume_lots": 0.01,
            "min_volume_auto": 0.01,
            "max_total_volume": "OFF",
            
            # S/L T/P Mode settings
            "sltp_mode": "Bội số ATR",  # Default to ATR mode
            "default_sl_pips": 50,
            "default_tp_pips": 100,
            "default_sl_atr_multiplier": 2.5,
            "default_tp_atr_multiplier": 1.5,
            "default_sl_percentage": 2.0,
            "default_tp_percentage": 4.0,
            "signal_sl_factor": 1.0,
            "signal_tp_factor": 1.5,
            
            # DCA Mode settings  
            "dca_mode": "atr_multiple",  # Default to ATR-based DCA
            "enable_dca": True,
            "max_dca_levels": 4,
            "dca_distance_pips": 50.0,
            "dca_atr_multiplier": 0.85,
            "dca_volume_multiplier": 1.5,
            "dca_percentage": 1.5,
            "dca_base_distance_pips": 50.0,
            
            # Risk management
            "max_risk_percent": "OFF",
            
            # Signal-based adjustment settings
            "enable_signal_based_adjustment": True,  # Enable signal-based S/L T/P adjustment
            "signal_improvement_min_profit_pips": 15,  # Min profit to allow signal improvement
            "min_sl_buffer_from_entry_pips": 35,  # Min buffer from entry when adjusting S/L
            "opposite_signal_min_profit_pips": 30,  # Min profit to act on opposite signals
            "opposite_signal_sl_buffer_pips": 20,  # Buffer when locking profit on opposite signal
            "signal_adjustment_sl_pips": 40,  # Default S/L adjustment pips
            "signal_adjustment_tp_pips": 120,  # Default T/P adjustment pips
        }
        
        # Apply defaults for missing settings
        for key, default_value in defaults.items():
            if key not in settings or settings[key] is None:
                settings[key] = default_value
                logger.info(f"🔧 Applied default: {key} = {default_value}")
        
        # Validate mode combinations
        sltp_mode = settings.get('sltp_mode', 'Bội số ATR')
        dca_mode = settings.get('dca_mode', 'atr_multiple')
        
        # Smart mode validation
        if sltp_mode == 'Bội số ATR' and dca_mode != 'atr_multiple':
            logger.warning(f"⚠️ Mode mismatch: S/L T/P is ATR but DCA is {dca_mode}. Consider using 'atr_multiple' for DCA.")
        
        return settings
    
    def _log_comprehensive_mode_info(self, settings: Dict):
        """📊 Log comprehensive information about current modes and settings"""
        
        logger.info(f"🎯 ===== COMPREHENSIVE MODE CONFIGURATION =====")
        
        # S/L T/P Mode
        sltp_mode = settings.get('sltp_mode', 'Unknown')
        logger.info(f"🎚️ S/L T/P Mode: {sltp_mode}")
        
        if sltp_mode == "Bội số ATR":
            sl_mult = settings.get('default_sl_atr_multiplier', 2.0)
            tp_mult = settings.get('default_tp_atr_multiplier', 1.5)
            logger.info(f"   📊 ATR Multipliers: SL={sl_mult}x, TP={tp_mult}x")
        elif sltp_mode == "Pips cố định":
            sl_pips = settings.get('default_sl_pips', 50)
            tp_pips = settings.get('default_tp_pips', 100)
            logger.info(f"   📏 Fixed Distances: SL={sl_pips}p, TP={tp_pips}p")
        elif sltp_mode == "Phần trăm":
            sl_pct = settings.get('default_sl_percentage', 2.0)
            tp_pct = settings.get('default_tp_percentage', 4.0)
            logger.info(f"   💰 Percentages: SL={sl_pct}%, TP={tp_pct}%")
        elif sltp_mode == "Theo Signal":
            sl_factor = settings.get('signal_sl_factor', 1.0)
            tp_factor = settings.get('signal_tp_factor', 1.5)
            logger.info(f"   🎯 Signal Factors: SL={sl_factor}x, TP={tp_factor}x")
        
        # DCA Mode
        dca_mode = settings.get('dca_mode', 'Unknown')
        logger.info(f"🔄 DCA Mode: {dca_mode}")
        
        if dca_mode == 'atr_multiple':
            dca_mult = settings.get('dca_atr_multiplier', 0.85)
            logger.info(f"   📊 ATR DCA Multiplier: {dca_mult}x")
        elif dca_mode in ['fixed_pips', 'Pips cố định']:
            dca_pips = settings.get('dca_distance_pips', 50)
            logger.info(f"   📏 Fixed DCA Distance: {dca_pips} pips")
        elif dca_mode == 'percentage':
            dca_pct = settings.get('dca_percentage', 1.5)
            logger.info(f"   💰 DCA Percentage: {dca_pct}%")
        elif dca_mode in ['fibonacci', 'fibo_levels', 'Fibonacci']:
            logger.info(f"   📐 Fibonacci DCA: Managed by dca_service.py")
            logger.info(f"   🎯 Execution Mode: Đặt Lệnh Chờ tại Mức")
        
        # General settings
        max_levels = settings.get('max_dca_levels', 4)
        vol_mult = settings.get('dca_volume_multiplier', 1.5)
        logger.info(f"⚙️ DCA Settings: Max {max_levels} levels, Volume multiplier {vol_mult}x")
        
        logger.info(f"🎯 ==============================================")

    def _get_market_volatility(self, timeframe: str = 'H1') -> Dict[str, float]:
        """Load market volatility and trend strength from trendline_sr data"""
        try:
            sr_file = f"trendline_sr/{self.symbol}_{timeframe}_trendline_sr.json"
            if os.path.exists(sr_file):
                with open(sr_file, 'r', encoding='utf-8') as f:
                    sr_data = json.load(f)
                
                volatility = sr_data.get('volatility', 0.5)  # Default moderate
                trend_strength = sr_data.get('trend_strength', 0.5)
                trend_direction = sr_data.get('trend_direction', 'Sideways')
                
                # Normalize volatility (0.0 to 1.0+)
                normalized_volatility = min(max(volatility, 0.1), 2.0)  # Cap between 0.1-2.0
                
                logger.debug(f"📊 Market Volatility {self.symbol} {timeframe}: vol={normalized_volatility:.3f}, trend_strength={trend_strength:.3f}, direction={trend_direction}")
                
                return {
                    'volatility': normalized_volatility,
                    'trend_strength': trend_strength, 
                    'trend_direction': trend_direction,
                    'source_file': sr_file
                }
            else:
                logger.warning(f"⚠️ Trendline SR file not found: {sr_file}")
                return {'volatility': 0.5, 'trend_strength': 0.5, 'trend_direction': 'Sideways'}
                
        except Exception as e:
            logger.error(f"Error loading market volatility: {e}")
            return {'volatility': 0.5, 'trend_strength': 0.5, 'trend_direction': 'Sideways'}

    def _calculate_volume_from_risk_settings(self, entry_price: float, sl_price: float, 
                                           entry_type: str = "NEW_ENTRY", account_balance: float = None) -> float:
        """Calculate volume based on risk settings from JSON file"""
        try:
            volume_mode = self.risk_settings.get("volume_mode", "Theo rủi ro (Tự động)")
            fixed_volume = self.risk_settings.get("fixed_volume_lots", 0.05)
            logger.info(f"🔍 VOLUME CALCULATION DEBUG:")
            logger.info(f"   volume_mode: '{volume_mode}'")  
            logger.info(f"   fixed_volume_lots: {fixed_volume}")
            logger.info(f"   entry_type: {entry_type}")
            
            # ========== PRIMARY VOLUME CALCULATION (Base Volume) ==========
            
            if "cố định" in volume_mode.lower() or "Fixed Volume" in volume_mode:
                # MODE 2: Fixed Volume - Tất cả lệnh dùng volume cố định
                base_volume = self.risk_settings.get("fixed_volume_lots", 0.05)  # Use proper fallback
                logger.info(f"💰 Fixed Volume Mode: {base_volume} lots (no scaling)")
                
            elif "mặc định" in volume_mode.lower() or "Default Volume" in volume_mode:
                # MODE 3: Default Volume - Use fixed_volume_lots setting
                base_volume = self.risk_settings.get("fixed_volume_lots", 0.05)  # Use proper default volume
                logger.info(f"💰 Default Volume Mode: {base_volume} lots (DCA scales from this)")
                
            elif "Theo rủi ro" in volume_mode and "Tự động" in volume_mode:
                # MODE 1: Auto Risk-Based - Lệnh đầu = min volume, DCA scales up
                base_volume = self.risk_settings.get("min_volume_auto", 0.01)
                logger.info(f"💰 Auto Risk-Based Mode: {base_volume} lots (min volume, DCA scales up)")
                
            else:
                # Fallback to min volume
                base_volume = self.risk_settings.get("min_volume_auto", 0.01)
                logger.warning(f"Unknown volume mode '{volume_mode}', using fallback: {base_volume} lots")
            
            # ========== DCA SCALING LOGIC (Depends on volume mode) ==========
            
            enable_dca = self.risk_settings.get("enable_dca", False)
            final_volume = base_volume
            
            # Fixed Volume Mode: NO DCA scaling - always use fixed volume
            if "cố định" in volume_mode.lower() or "Fixed Volume" in volume_mode:
                final_volume = base_volume  # No scaling regardless of entry type
                if entry_type in ["DCA_ADD", "DCA_SCALE"]:
                    logger.info(f"💰 Fixed Volume DCA: {final_volume} lots (no scaling in fixed mode)")
                
            # Auto Mode & Default Mode: Apply DCA scaling
            elif enable_dca and entry_type in ["DCA_ADD", "DCA_SCALE"]:
                dca_mode = self.risk_settings.get("dca_mode", "Cố định")
                dca_multiplier = self.risk_settings.get("dca_volume_multiplier", 1.5)
                
                if "Fibo" in dca_mode or "Fibonacci" in dca_mode:
                    # Fibonacci DCA scaling
                    dca_level = getattr(self, '_current_dca_level', 1)
                    fibo_levels = self.risk_settings.get("dca_fibo_levels", "1,1,2,3,5,8,13,21,34,55,89")
                    try:
                        fibo_sequence = [float(x.strip()) for x in fibo_levels.split(',')]
                        if dca_level < len(fibo_sequence):
                            fibo_multiplier = fibo_sequence[dca_level] / fibo_sequence[0]
                            final_volume = base_volume * fibo_multiplier
                            logger.info(f"📈 DCA Fibonacci Level {dca_level}: {final_volume} lots (base: {base_volume}, fibo: {fibo_multiplier:.2f}x)")
                        else:
                            final_volume = base_volume * dca_multiplier
                            logger.info(f"📈 DCA Standard scaling: {final_volume} lots (base: {base_volume}, multiplier: {dca_multiplier}x)")
                    except:
                        final_volume = base_volume * dca_multiplier
                        logger.warning(f"📈 DCA Fibo parsing failed, using standard: {final_volume} lots")
                else:
                    # Standard DCA multiplier scaling
                    final_volume = base_volume * dca_multiplier
                    logger.info(f"📈 DCA Standard scaling: {final_volume} lots (base: {base_volume}, multiplier: {dca_multiplier}x)")
            
            # Hedge positions (all modes)
            elif entry_type == "HEDGE":
                final_volume = base_volume * 0.5
                logger.info(f"🛡️ Hedge scaling: {final_volume} lots (base: {base_volume}, hedge: 0.5x)")
            
            # ========== VOLUME LIMITS & VALIDATION ==========
            
            # Apply min/max limits
            min_vol = self.risk_settings.get("min_volume_auto", 0.01)
            max_vol = self.risk_settings.get("max_total_volume", "OFF")
            
            if max_vol != "OFF" and max_vol is not None:
                try:
                    max_vol = float(max_vol)
                    final_volume = min(final_volume, max_vol)
                except (TypeError, ValueError):
                    logger.warning(f"Invalid max_total_volume value: {max_vol}, ignoring limit")
            
            final_volume = max(final_volume, min_vol)
            final_volume = round(final_volume, 2)
            
            logger.debug(f"Volume calculation for {self.symbol}: entry_type={entry_type}, volume={final_volume}")
            return final_volume
            
        except Exception as e:
            logger.error(f"Error calculating volume from risk settings: {e}")
            return self.risk_settings.get("min_volume_auto", 0.01)
    
    def _calculate_risk_based_volume(self, entry_price: float, sl_price: float, account_balance: float = None) -> float:
        """Calculate volume based on risk percentage and stop loss distance"""
        try:
            # Get account balance
            if account_balance is None:
                try:
                    with open("account_scans/mt5_essential_scan.json", 'r') as f:
                        account_data = json.load(f)
                        account_balance = account_data.get("account", {}).get("balance", 10000)
                except:
                    account_balance = 10000  # Default fallback
            
            # Get risk percentage
            max_risk_percent = self.risk_settings.get("max_risk_percent", "OFF")
            if max_risk_percent == "OFF":
                risk_percent = 2.0  # Default 2%
            else:
                risk_percent = float(max_risk_percent)
            
            # Calculate risk amount (maximum loss allowed)
            risk_amount = account_balance * (risk_percent / 100)
            
            # Calculate risk distance in price
            risk_distance = abs(entry_price - sl_price)
            if risk_distance == 0:
                return self.risk_settings.get("min_volume_auto", 0.01)
            
            # Calculate pip value and lot size
            pip_value = self._get_pip_value(self.symbol)
            if pip_value == 0:
                return self.risk_settings.get("min_volume_auto", 0.01)
            
            # Volume calculation for risk management
            # Formula: Volume = Risk Amount / (SL Distance in Pips × Pip Value in USD)
            
            sl_distance_pips = risk_distance / pip_value  # Convert price distance to pips
            
            if any(crypto in self.symbol.upper() for crypto in ['BTC', 'ETH', 'SOL', 'ADA', 'DOGE']):
                # Crypto: Direct volume calculation (simplified)
                calculated_volume = risk_amount / risk_distance if risk_distance > 0 else 0.01
            elif self.symbol.upper().endswith('JPY'):
                # JPY pairs: Pip value ~$9-10 per pip per standard lot
                pip_value_usd = 9.5  # Approximate pip value for JPY pairs
                calculated_volume = risk_amount / (sl_distance_pips * pip_value_usd) if sl_distance_pips > 0 else 0.01
            else:
                # Major forex pairs (EUR/USD, GBP/USD, etc.): Pip value ~$10 per pip per standard lot
                pip_value_usd = 10.0  # Standard pip value for major pairs
                calculated_volume = risk_amount / (sl_distance_pips * pip_value_usd) if sl_distance_pips > 0 else 0.01
            
            calculated_volume = round(calculated_volume, 2)
            logger.info(f"💰 Risk-based volume: {calculated_volume} lots (Risk: {risk_percent}%, Amount: ${risk_amount:.2f})")
            return calculated_volume
            
        except Exception as e:
            logger.error(f"Error in risk-based volume calculation: {e}")
            return self.risk_settings.get("min_volume_auto", 0.01)
            logger.error(f"Error calculating volume: {e}")
            return self.risk_settings.get("min_volume_auto", 0.01)

    def _calculate_sl_tp_from_risk_settings(self, entry_price: float, signal: str, atr: float = None, 
                                          support_levels: List[float] = None, resistance_levels: List[float] = None,
                                          signal_data: dict = None) -> Tuple[float, float]:
        """
        🔧 ENHANCED: Tính toán SL/TP theo chế độ user chọn trong risk_settings.json
        
        Các chế độ S/L T/P được hỗ trợ:
        1. "Pips cố định" - Sử dụng default_sl_pips/default_tp_pips
        2. "Bội số ATR" - Sử dụng ATR * multiplier
        3. "% Entry" - Sử dụng % của giá entry
        4. "Hỗ trợ/Kháng cự" - Dựa trên support/resistance levels
        5. "Auto/Tự động" - Kết hợp nhiều yếu tố
        
        Args:
            entry_price: Giá entry
            signal: 'BUY' hoặc 'SELL'
            atr: Giá trị ATR hiện tại
            support_levels: Danh sách mức hỗ trợ
            resistance_levels: Danh sách mức kháng cự
            signal_data: Dữ liệu signal bổ sung
            
        Returns:
            Tuple[sl_price, tp_price]: SL/TP theo risk settings mode
        """
        try:
            # 🔍 Đọc chế độ S/L T/P từ risk settings
            sltp_mode = self.risk_settings.get('sltp_mode', 'Pips cố định')
            logger.info(f"🎯 SL/TP Mode: {sltp_mode} for {self.symbol}")
            logger.info(f"🎯 Entry: {entry_price}, ATR: {atr}")
            
            pip_value = self._get_pip_value(self.symbol)
            
            # 📊 CHẾ ĐỘ 1: PIPS CỐ ĐỊNH
            if sltp_mode == "Pips cố định":
                sl_pips = self.risk_settings.get('default_sl_pips', 50)
                tp_pips = self.risk_settings.get('default_tp_pips', 100)
                
                logger.info(f"📏 Fixed Pips Mode: SL={sl_pips} pips, TP={tp_pips} pips")
                
                if signal == "BUY":
                    sl_price = entry_price - (sl_pips * pip_value)
                    tp_price = entry_price + (tp_pips * pip_value)
                else:  # SELL
                    sl_price = entry_price + (sl_pips * pip_value)
                    tp_price = entry_price - (tp_pips * pip_value)
                
                logger.info(f"✅ Fixed Pips SL/TP: SL={sl_price:.5f}, TP={tp_price:.5f}")
                return sl_price, tp_price
            
            # 📈 CHẾ ĐỘ 2: BỘI SỐ ATR
            elif sltp_mode == "Bội số ATR":
                sl_multiplier = self.risk_settings.get('default_sl_atr_multiplier', 2.0)
                tp_multiplier = self.risk_settings.get('default_tp_atr_multiplier', 1.5)
                
                if atr and atr > 0:
                    # Sử dụng ATR thực từ indicator - FIXED: không chia 100
                    atr_sl_distance = atr * sl_multiplier  # ATR * multiplier trực tiếp 
                    atr_tp_distance = atr * tp_multiplier
                    
                    logger.info(f"📊 ATR Mode: ATR={atr:.5f}, SL_mult={sl_multiplier}, TP_mult={tp_multiplier}")
                else:
                    # Fallback nếu không có ATR - sử dụng pips cố định
                    logger.warning(f"⚠️ No ATR available, using fallback calculation")
                    sl_pips = self.risk_settings.get('default_sl_pips', 80)
                    tp_pips = self.risk_settings.get('default_tp_pips', 160)
                    atr_sl_distance = sl_pips * pip_value
                    atr_tp_distance = tp_pips * pip_value
                    logger.info(f"🔄 ATR Fallback: SL={sl_pips} pips, TP={tp_pips} pips")
                
                if signal == "BUY":
                    sl_price = entry_price - atr_sl_distance
                    tp_price = entry_price + atr_tp_distance
                else:  # SELL
                    sl_price = entry_price + atr_sl_distance
                    tp_price = entry_price - atr_tp_distance
                
                logger.info(f"✅ ATR SL/TP: SL={sl_price:.5f}, TP={tp_price:.5f}")
                return sl_price, tp_price
            
            # 📊 CHẾ ĐỘ 3: % ENTRY PRICE / PHẦN TRĂM
            elif sltp_mode in ["% Entry", "Phần trăm"]:
                sl_percent = self.risk_settings.get('default_sl_percentage', 2.0) / 100.0  # Convert % to decimal
                tp_percent = self.risk_settings.get('default_tp_percentage', 4.0) / 100.0
                
                logger.info(f"💰 Percentage Mode: SL={sl_percent*100:.1f}%, TP={tp_percent*100:.1f}%")
                
                if signal == "BUY":
                    sl_price = entry_price * (1 - sl_percent)
                    tp_price = entry_price * (1 + tp_percent)
                else:  # SELL
                    sl_price = entry_price * (1 + sl_percent)
                    tp_price = entry_price * (1 - tp_percent)
                
                logger.info(f"✅ Percentage SL/TP: SL={sl_price:.5f}, TP={tp_price:.5f}")
                return sl_price, tp_price
                
            # 🎯 CHẾ ĐỘ 4: THEO SIGNAL / SIGNAL BASED
            elif sltp_mode in ["Signal Based", "Theo Signal"]:
                sl_factor = self.risk_settings.get('signal_sl_factor', 1.0)
                tp_factor = self.risk_settings.get('signal_tp_factor', 1.5)
                
                # Get SL/TP from signal data if available
                if signal_data:
                    signal_sl = signal_data.get('stop_loss')
                    signal_tp = signal_data.get('take_profit')
                    
                    if signal_sl and signal_tp:
                        # Apply factors to signal's SL/TP
                        if signal == "BUY":
                            sl_distance = (entry_price - signal_sl) * sl_factor
                            tp_distance = (signal_tp - entry_price) * tp_factor
                            sl_price = entry_price - sl_distance
                            tp_price = entry_price + tp_distance
                        else:  # SELL
                            sl_distance = (signal_sl - entry_price) * sl_factor
                            tp_distance = (entry_price - signal_tp) * tp_factor
                            sl_price = entry_price + sl_distance
                            tp_price = entry_price - tp_distance
                        
                        logger.info(f"🎯 Signal-based SL/TP: SL_factor={sl_factor}, TP_factor={tp_factor}")
                        logger.info(f"✅ Signal SL/TP: SL={sl_price:.5f}, TP={tp_price:.5f}")
                        return sl_price, tp_price
                
                # Fallback: Use ATR if no signal SL/TP
                logger.warning(f"⚠️ No signal SL/TP data, using ATR fallback")
                return self._calculate_atr_fallback_sl_tp(entry_price, signal, atr)
            
            # 🏆 CHẾ ĐỘ 4: HỖ TRỢ/KHÁNG CỰ
            elif sltp_mode == "Hỗ trợ/Kháng cự":
                sl_price, tp_price = self._calculate_support_resistance_sl_tp(
                    entry_price, signal, support_levels, resistance_levels, atr
                )
                if sl_price and tp_price:
                    logger.info(f"✅ Support/Resistance SL/TP: SL={sl_price:.5f}, TP={tp_price:.5f}")
                    return sl_price, tp_price
                else:
                    logger.warning(f"⚠️ Support/Resistance calculation failed, using ATR fallback")
                    # Fallback to ATR method
                    return self._calculate_atr_fallback_sl_tp(entry_price, signal, atr)
            
            # 🚀 CHẾ ĐỘ 5: AUTO/TỰ ĐỘNG - Kết hợp nhiều yếu tố
            elif sltp_mode in ["Auto", "Tự động", "Auto/Tự động"]:
                sl_price, tp_price = self._calculate_dynamic_sl_tp(entry_price, signal, atr, signal_data)
                if sl_price and tp_price:
                    logger.info(f"✅ Dynamic Auto SL/TP: SL={sl_price:.5f}, TP={tp_price:.5f}")
                    return sl_price, tp_price
                else:
                    logger.warning(f"⚠️ Dynamic calculation failed, using ATR fallback")
                    return self._calculate_atr_fallback_sl_tp(entry_price, signal, atr)
            
            # 🔄 FALLBACK: Chế độ không được hỗ trợ - dùng legacy
            else:
                logger.warning(f"⚠️ Unknown SL/TP mode '{sltp_mode}', using legacy calculation")
                return self._calculate_sl_tp_legacy_modes(entry_price, signal, atr, support_levels, resistance_levels, signal_data)
            
        except Exception as e:
            logger.error(f"❌ Error calculating SL/TP for {self.symbol}: {e}")
            # 🚨 Emergency fallback - Fixed pips
            sl_pips = 50
            tp_pips = 100
            if signal == "BUY":
                return entry_price - (sl_pips * pip_value), entry_price + (tp_pips * pip_value)
            else:
                return entry_price + (sl_pips * pip_value), entry_price - (tp_pips * pip_value)

    def _calculate_support_resistance_sl_tp(self, entry_price: float, signal: str, 
                                          support_levels: List[float] = None, 
                                          resistance_levels: List[float] = None,
                                          atr: float = None) -> Tuple[float, float]:
        """
        🏆 Calculate SL/TP based on Support/Resistance levels from trendline_sr data
        
        Args:
            entry_price: Entry price
            signal: 'BUY' or 'SELL'
            support_levels: Support levels (can be None, will load from trendline_sr)
            resistance_levels: Resistance levels (can be None, will load from trendline_sr)
            atr: ATR for buffer calculation
            
        Returns:
            Tuple[sl_price, tp_price]: SL/TP based on S/R levels with buffer
        """
        try:
            # 📊 Load S/R data from trendline_sr if not provided
            if not support_levels or not resistance_levels:
                sr_data = self._load_trendline_sr_data()
                if sr_data:
                    support_levels = sr_data.get('support_levels', [])
                    resistance_levels = sr_data.get('resistance_levels', [])
                    logger.info(f"📊 Loaded S/R from trendline_sr: {len(support_levels)} supports, {len(resistance_levels)} resistances")
                else:
                    logger.warning(f"⚠️ No trendline_sr data found for {self.symbol}")
                    return None, None
            
            # Convert string levels to float if needed
            if support_levels:
                support_levels = [float(x) if isinstance(x, str) else x for x in support_levels if x]
            if resistance_levels:
                resistance_levels = [float(x) if isinstance(x, str) else x for x in resistance_levels if x]
            
            # 🎯 Calculate buffer from risk settings
            pip_value = self._get_pip_value(self.symbol)
            
            # Get buffer settings from risk settings (default: 20 pips)
            sr_buffer_pips = self.risk_settings.get('sr_buffer_pips', 20.0)
            
            # 🔧 Enhanced buffer calculation with ATR if available
            if atr and atr > 0:
                atr_pips = atr / pip_value
                # Use 30% of ATR but minimum sr_buffer_pips
                dynamic_buffer_pips = max(sr_buffer_pips, atr_pips * 0.3)
                logger.info(f"🔧 Dynamic S/R buffer: {dynamic_buffer_pips:.1f} pips (ATR-based: {atr_pips:.1f} pips)")
            else:
                dynamic_buffer_pips = sr_buffer_pips
                logger.info(f"🔧 Fixed S/R buffer: {dynamic_buffer_pips:.1f} pips")
            
            buffer_distance = dynamic_buffer_pips * pip_value
            
            if signal == "BUY":
                # 🟢 BUY Signal: SL below nearest support, TP above nearest resistance
                
                # Find nearest support below entry for SL
                valid_supports = [s for s in (support_levels or []) if s < entry_price]
                if valid_supports:
                    nearest_support = max(valid_supports)  # Closest support below entry
                    sl_price = nearest_support - buffer_distance  # SL below support with buffer
                    logger.info(f"🟢 BUY SL: Support {nearest_support:.5f} - buffer {dynamic_buffer_pips:.1f}pips = {sl_price:.5f}")
                else:
                    # No support found, use fallback
                    fallback_sl_pips = self.risk_settings.get('default_sl_pips', 150)
                    sl_price = entry_price - (fallback_sl_pips * pip_value)
                    logger.warning(f"⚠️ No support below entry, using fallback SL: {fallback_sl_pips} pips")
                
                # Find nearest resistance above entry for TP
                valid_resistances = [r for r in (resistance_levels or []) if r > entry_price]
                if valid_resistances:
                    nearest_resistance = min(valid_resistances)  # Closest resistance above entry
                    tp_price = nearest_resistance - buffer_distance  # TP below resistance with buffer
                    logger.info(f"🟢 BUY TP: Resistance {nearest_resistance:.5f} - buffer {dynamic_buffer_pips:.1f}pips = {tp_price:.5f}")
                else:
                    # No resistance found, use fallback
                    fallback_tp_pips = self.risk_settings.get('default_tp_pips', 100)
                    tp_price = entry_price + (fallback_tp_pips * pip_value)
                    logger.warning(f"⚠️ No resistance above entry, using fallback TP: {fallback_tp_pips} pips")
                
            else:  # SELL Signal
                # 🔴 SELL Signal: SL above nearest resistance, TP below nearest support
                
                # Find nearest resistance above entry for SL  
                valid_resistances = [r for r in (resistance_levels or []) if r > entry_price]
                if valid_resistances:
                    nearest_resistance = min(valid_resistances)  # Closest resistance above entry
                    sl_price = nearest_resistance + buffer_distance  # SL above resistance with buffer
                    logger.info(f"🔴 SELL SL: Resistance {nearest_resistance:.5f} + buffer {dynamic_buffer_pips:.1f}pips = {sl_price:.5f}")
                else:
                    # No resistance found, use fallback
                    fallback_sl_pips = self.risk_settings.get('default_sl_pips', 150)
                    sl_price = entry_price + (fallback_sl_pips * pip_value)
                    logger.warning(f"⚠️ No resistance above entry, using fallback SL: {fallback_sl_pips} pips")
                
                # Find nearest support below entry for TP
                valid_supports = [s for s in (support_levels or []) if s < entry_price]
                if valid_supports:
                    nearest_support = max(valid_supports)  # Closest support below entry
                    tp_price = nearest_support + buffer_distance  # TP above support with buffer
                    logger.info(f"🔴 SELL TP: Support {nearest_support:.5f} + buffer {dynamic_buffer_pips:.1f}pips = {tp_price:.5f}")
                else:
                    # No support found, use fallback
                    fallback_tp_pips = self.risk_settings.get('default_tp_pips', 100)
                    tp_price = entry_price - (fallback_tp_pips * pip_value)
                    logger.warning(f"⚠️ No support below entry, using fallback TP: {fallback_tp_pips} pips")
            
            # 🛡️ Validate SL/TP are reasonable
            if sl_price and tp_price:
                # Calculate distances in pips for validation
                if signal == "BUY":
                    sl_pips = (entry_price - sl_price) / pip_value
                    tp_pips = (tp_price - entry_price) / pip_value
                else:
                    sl_pips = (sl_price - entry_price) / pip_value
                    tp_pips = (entry_price - tp_price) / pip_value
                
                logger.info(f"✅ S/R SL/TP calculated: SL={sl_pips:.1f}pips, TP={tp_pips:.1f}pips")
                
                # Enhanced validation - minimum and maximum distances based on symbol type
                min_sl_pips = 30.0  # Minimum 30 pips for SL
                min_tp_pips = 20.0  # Minimum 20 pips for TP
                max_sl_pips = 300.0  # Maximum 300 pips for SL to prevent too wide stops
                max_tp_pips = 500.0  # Maximum 500 pips for TP
                
                # Crypto pairs need larger distances
                if any(crypto in self.symbol for crypto in ['BTC', 'ETH', 'BNB', 'SOL', 'LTC']):
                    min_sl_pips = 100.0  # 100 pips minimum for crypto
                    min_tp_pips = 50.0   # 50 pips minimum for crypto TP
                    max_sl_pips = 500.0  # 500 pips maximum for crypto SL
                    max_tp_pips = 800.0  # 800 pips maximum for crypto TP
                
                # Metals (XAUUSD, XAGUSD) have different ranges
                elif any(metal in self.symbol for metal in ['XAU', 'XAG']):
                    min_sl_pips = 20.0   # 20 pips minimum for metals
                    min_tp_pips = 30.0   # 30 pips minimum for metals TP
                    max_sl_pips = 150.0  # 150 pips maximum for metals SL
                    max_tp_pips = 300.0  # 300 pips maximum for metals TP
                
                # Check if calculated distances are within reasonable ranges
                if (min_sl_pips <= sl_pips <= max_sl_pips and 
                    min_tp_pips <= tp_pips <= max_tp_pips):
                    logger.info(f"✅ Support/Resistance SL/TP: SL={sl_price:.5f}, TP={tp_price:.5f}")
                    return sl_price, tp_price
                else:
                    logger.warning(f"⚠️ S/R calculation out of range: SL={sl_pips:.1f} (range:{min_sl_pips}-{max_sl_pips}), TP={tp_pips:.1f} (range:{min_tp_pips}-{max_tp_pips}) pips")
                    
                    # Use fallback with minimum distances
                    fallback_sl_pips = max(self.risk_settings.get('default_sl_pips', 150), min_sl_pips)
                    fallback_tp_pips = max(self.risk_settings.get('default_tp_pips', 100), min_tp_pips)
                    
                    if signal == "BUY":
                        sl_price = entry_price - (fallback_sl_pips * pip_value)
                        tp_price = entry_price + (fallback_tp_pips * pip_value)
                    else:
                        sl_price = entry_price + (fallback_sl_pips * pip_value)
                        tp_price = entry_price - (fallback_tp_pips * pip_value)
                    
                    logger.info(f"🔧 Using fallback SL/TP: SL={fallback_sl_pips}pips, TP={fallback_tp_pips}pips")
                    return sl_price, tp_price
            
            # Final fallback if no S/R calculation worked
            logger.warning(f"⚠️ S/R calculation failed, using default pips for {self.symbol}")
            fallback_sl_pips = self.risk_settings.get('default_sl_pips', 150)
            fallback_tp_pips = self.risk_settings.get('default_tp_pips', 100)
            
            # Ensure minimum distances for crypto
            if any(crypto in self.symbol for crypto in ['BTC', 'ETH', 'BNB', 'SOL', 'LTC']):
                fallback_sl_pips = max(fallback_sl_pips, 100)
                fallback_tp_pips = max(fallback_tp_pips, 50)
            
            if signal == "BUY":
                sl_price = entry_price - (fallback_sl_pips * pip_value)
                tp_price = entry_price + (fallback_tp_pips * pip_value)
            else:
                sl_price = entry_price + (fallback_sl_pips * pip_value)
                tp_price = entry_price - (fallback_tp_pips * pip_value)
            
            logger.info(f"🔧 Fallback SL/TP applied: SL={fallback_sl_pips}pips, TP={fallback_tp_pips}pips")
            return sl_price, tp_price
            
        except Exception as e:
            logger.error(f"❌ Error in S/R calculation for {self.symbol}: {e}")
            return None, None
    
    def _load_trendline_sr_data(self) -> dict:
        """
        📊 Load Support/Resistance data from trendline_sr folder
        
        Returns:
            dict: S/R data with support_levels and resistance_levels
        """
        import json
        import os
        
        try:
            # Try H1 first, then H4, M30, M15
            timeframes = ['H1', 'H4', 'M30', 'M15']
            
            # Try both symbol formats: ETHUSD and ETHUSD_m
            symbol_variants = [self.symbol, f"{self.symbol}_m"]
            
            for tf in timeframes:
                for symbol_variant in symbol_variants:
                    sr_file = f"trendline_sr/{symbol_variant}_{tf}_trendline_sr.json"
                    if os.path.exists(sr_file):
                        with open(sr_file, 'r', encoding='utf-8') as f:
                            sr_data = json.load(f)
                            logger.info(f"📊 Loaded S/R data from {sr_file}")
                            return sr_data
            
            logger.warning(f"⚠️ No trendline_sr file found for {self.symbol} (tried variants: {symbol_variants})")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error loading trendline_sr data for {self.symbol}: {e}")
            return None
    
    def _calculate_dynamic_sl_tp(self, entry_price: float, signal: str, atr: float = None, 
                               signal_data: dict = None) -> Tuple[float, float]:
        """
        🚀 NEW: Tính SL/TP từ indicators động (ATR, volatility, momentum)
        """
        pip_value = self._get_pip_value(self.symbol)
        
        # ========== METHOD 1: ATR-based dynamic calculation ==========
        if atr and atr > 0:
            # Normalize ATR to pips
            if self.symbol.upper().rstrip('.') in ['XAUUSD', 'XAGUSD', 'GOLD', 'SILVER']:
                atr_pips = atr / pip_value  # For metals
            elif 'JPY' in self.symbol.upper():
                atr_pips = atr / pip_value  # For JPY pairs
            else:
                atr_pips = atr / pip_value  # Standard calculation
            
            # ✅ Use configured ATR multipliers from risk settings
            sl_mult = self.risk_settings.get('default_sl_atr_multiplier', 2.0)
            tp_mult = self.risk_settings.get('default_tp_atr_multiplier', 1.5)
            logger.info(f"📊 Using configured multipliers: SL={sl_mult}x ATR, TP={tp_mult}x ATR (ATR={atr_pips:.1f} pips)")
            
            # Calculate SL/TP distances
            sl_distance = sl_mult * atr
            tp_distance = tp_mult * atr
            
            # Apply to entry price
            if signal == "BUY":
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
            else:  # SELL
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance
            
            logger.info(f"💎 ATR Dynamic: {sl_mult}x ATR SL, {tp_mult}x ATR TP ({atr_pips:.1f} pips base)")
            return sl_price, tp_price
        
        # ========== METHOD 2: Momentum/Volatility indicators ==========
        # TODO: Add Bollinger Band width, Stochastic levels, RSI divergence analysis
        # For now, return None to trigger fallback
        return None, None
    
    def _calculate_atr_fallback_sl_tp(self, entry_price: float, signal: str, atr: float = None) -> Tuple[float, float]:
        """
        🔄 FALLBACK: Simple ATR or fixed calculation when dynamic fails
        """
        pip_value = self._get_pip_value(self.symbol)
        
        if atr and atr > 0:
            # Use configured ATR multipliers from risk settings
            sl_mult = self.risk_settings.get('default_sl_atr_multiplier', 2.0)
            tp_mult = self.risk_settings.get('default_tp_atr_multiplier', 1.5)
            sl_distance = sl_mult * atr
            tp_distance = tp_mult * atr
            logger.info(f"🔄 ATR Fallback: {sl_mult}x ATR SL, {tp_mult}x ATR TP")
        else:
            # Fixed pip fallback based on symbol type
            if any(crypto in self.symbol.upper() for crypto in ['BTC', 'ETH', 'SOL', 'ADA']):
                sl_pips, tp_pips = 80, 160  # Crypto
            elif 'JPY' in self.symbol.upper():
                sl_pips, tp_pips = 60, 120   # JPY pairs
            elif self.symbol.upper().rstrip('.') in ['XAUUSD', 'XAGUSD']:
                sl_pips, tp_pips = 50, 100   # Metals
            else:
                sl_pips, tp_pips = 40, 80    # Major forex
            
            sl_distance = sl_pips * pip_value
            tp_distance = tp_pips * pip_value
            logger.info(f"🔄 Fixed Fallback: {sl_pips}/{tp_pips} pips")
        
        if signal == "BUY":
            return entry_price - sl_distance, entry_price + tp_distance
        else:
            return entry_price + sl_distance, entry_price - tp_distance

    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol calculation within Aggregator class - USE CLASS METHOD"""
        # Use the comprehensive method from line 10566 instead of standalone function
        symbol_upper = symbol.upper().rstrip('.').replace('_M', '').replace('_m', '')  # 🔧 Normalize all variants
        
        # 🎯 DEBUG: Log symbol matching
        logger.debug(f"🎯 _get_pip_value: '{symbol}' -> '{symbol_upper}'")
        
        # ========== PRECIOUS METALS ==========
        if symbol_upper in ['XAUUSD', 'XAGUSD', 'GOLD', 'SILVER', 'XPTUSD', 'XPDUSD']:
            logger.debug(f"🎯 Matched METALS: {symbol_upper} -> 0.1")
            return 0.1   # Metals: 1 pip = 0.1 (Gold: 3881.03 -> 3881.73 = 7 pips)
        
        # ========== JPY PAIRS ==========
        elif 'JPY' in symbol_upper:
            logger.debug(f"🎯 Matched JPY pair: {symbol_upper} -> 0.01")
            return 0.01  # JPY pairs: 1 pip = 0.01 (USD/JPY: 147.15 -> 147.22 = 7 pips)
        
        # ========== HIGH-VALUE CRYPTO (≥ $1000) ==========
        elif symbol_upper in ['BTCUSD', 'ETHUSD']:
            logger.debug(f"🎯 Matched HIGH-VALUE CRYPTO: {symbol_upper} -> 1.0")
            return 1.0   # BTC/ETH: 1 pip = 1.0 (BTC: 65000 -> 65070 = 70 pips)
        
        # ========== MID-VALUE CRYPTO ($100-$1000) ==========
        elif symbol_upper in ['SOLUSD', 'LTCUSD', 'BNBUSD', 'AVAXUSD', 'DOTUSD', 'MATICUSD', 'LINKUSD', 'TRXUSD', 'SHIBUSD', 'ARBUSD', 'OPUSD', 'APEUSD', 'SANDUSD', 'CROUSD', 'FTTUSD']:
            logger.debug(f"🎯 Matched MID-VALUE CRYPTO: {symbol_upper} -> 0.1")
            return 0.1   # SOL/LTC/BNB etc: 1 pip = 0.1 (SOL: 224.06 -> 224.76 = 7 pips)
        
        # ========== LOW-VALUE CRYPTO ($1-$10) - Standard Forex Pip ==========
        elif symbol_upper in ['ADAUSD']:
            logger.debug(f"🎯 Matched LOW-VALUE CRYPTO: {symbol_upper} -> 0.0001")
            return 0.0001  # ADA etc: 1 pip = 0.0001 (ADA: 0.8391 -> 0.8398 = 7 pips)
        
        # ========== LOW-VALUE CRYPTO (< $10) ==========
        elif any(crypto in symbol_upper for crypto in ['DOGE', 'XRP', 'TRX']):
            logger.debug(f"🎯 Matched LOW-VALUE CRYPTO: {symbol_upper} -> 0.001")
            return 0.001  # DOGE/XRP etc: 1 pip = 0.001 (DOGE: 0.123 -> 0.130 = 7 pips)
        
        # ========== MICRO-VALUE CRYPTO (< $1) ==========
        elif any(micro_crypto in symbol_upper for micro_crypto in ['SHIB', 'PEPE', 'FLOKI']):
            logger.debug(f"🎯 Matched MICRO-VALUE CRYPTO: {symbol_upper} -> 0.00001")
            return 0.00001  # Micro cryptos: 1 pip = 0.00001
        
        # ========== FOREX PAIRS ==========
        else:
            logger.debug(f"🎯 Default FX pair: {symbol_upper} -> 0.0001")
            return 0.0001  # Major FX pairs: 1 pip = 0.0001 (EUR/USD: 1.0850 -> 1.0857 = 7 pips)

    def _calculate_sl_tp_legacy_modes(self, entry_price: float, signal: str, atr: float = None,
                                    support_levels: List[float] = None, resistance_levels: List[float] = None,
                                    signal_data: dict = None) -> Tuple[float, float]:
        """
        � Legacy calculation modes for backward compatibility
        
        Xử lý các chế độ cũ hoặc không nhận dạng được từ risk_settings
        """
        try:
            pip_value = self._get_pip_value(self.symbol)
            
            # Try to determine mode from signal_sl_factor and signal_tp_factor
            sl_factor = self.risk_settings.get('signal_sl_factor', 1.5)
            tp_factor = self.risk_settings.get('signal_tp_factor', 2.0)
            
            # Use ATR if available, otherwise default pips
            if atr and atr > 0:
                sl_distance = atr * sl_factor
                tp_distance = atr * tp_factor
                logger.info(f"� Legacy ATR Mode: ATR={atr:.5f}, SL_factor={sl_factor}, TP_factor={tp_factor}")
            else:
                # Fallback to fixed pips
                default_sl = self.risk_settings.get('default_sl_pips', 50)
                default_tp = self.risk_settings.get('default_tp_pips', 100)
                sl_distance = default_sl * pip_value
                tp_distance = default_tp * pip_value
                logger.info(f"🔄 Legacy Pips Mode: SL={default_sl} pips, TP={default_tp} pips")
            
            if signal == "BUY":
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
            else:  # SELL
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance
            
            logger.info(f"✅ Legacy SL/TP: SL={sl_price:.5f}, TP={tp_price:.5f}")
            return sl_price, tp_price
            
        except Exception as e:
            logger.error(f"❌ Error in legacy SL/TP calculation: {e}")
            # Final emergency fallback
            pip_value = self._get_pip_value(self.symbol)
            if signal == "BUY":
                return entry_price - (50 * pip_value), entry_price + (100 * pip_value)
            else:
                return entry_price + (50 * pip_value), entry_price - (100 * pip_value)
        
        """
        🚫 LEGACY CODE COMMENTED OUT - All old SL/TP modes disabled  
        Now using dynamic indicator-based calculation only
        Order Executor will adjust final values according to Risk GUI settings
        
        Legacy modes: "Bội số ATR", "Pips cố định", "Phần trăm", "Hỗ trợ/Kháng cự", "Theo tín hiệu" 
        All ~300 lines of legacy code removed - see git history for original implementation
        """
                    
                    

    def _get_symbol_specific_settings(self) -> Dict[str, Any]:
        """Get symbol-specific risk settings from risk_settings.json"""
        try:
            unified_file = "risk_management/risk_settings.json"
            if os.path.exists(unified_file):
                with open(unified_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # 🔧 FIX: Normalize symbol for lookup (handle variants like SOLUSD_m -> SOLUSD)
                normalized_symbol = self.symbol.upper().rstrip('.').replace('_M', '').replace('_m', '')
                
                # Try multiple symbol variations
                symbol_variants = [
                    normalized_symbol,  # SOLUSD_m -> SOLUSD
                    self.symbol.upper().rstrip('.'),  # Original without dots
                    self.symbol  # Original as-is
                ]
                
                symbol_settings = {}
                for variant in symbol_variants:
                    symbol_settings = settings.get('symbol_specific_settings', {}).get(variant, {})
                    if symbol_settings:
                        logger.info(f"🎯 Found symbol settings for '{variant}' (from '{self.symbol}')")
                        break
                
                if not symbol_settings:
                    available_symbols = list(settings.get('symbol_specific_settings', {}).keys())
                    logger.warning(f"⚠️ No symbol-specific settings found for {self.symbol} or variants {symbol_variants}")
                    logger.info(f"📋 Available symbols: {available_symbols}")
                
                # Return symbol settings with fallback to global defaults
                global_settings = settings.get('basic_settings', {})
                return {
                    'default_sl_pips': symbol_settings.get('default_sl_pips', 
                                                          global_settings.get('default_sl_pips', 80)),
                    'default_tp_pips': symbol_settings.get('default_tp_pips', 
                                                          global_settings.get('default_tp_pips', 150)),
                    'default_sl_buffer': symbol_settings.get('default_sl_buffer', 
                                                           global_settings.get('default_sl_buffer', 800)),
                    'default_tp_buffer': symbol_settings.get('default_tp_buffer', 
                                                           global_settings.get('default_tp_buffer', 400)),
                    'dca_distance_pips': symbol_settings.get('dca_distance_pips', 
                                                            settings.get('dca_distance_pips', global_settings.get('dca_base_distance_pips', 5.0))),
                    'min_risk_reward_ratio': symbol_settings.get('min_risk_reward_ratio', 
                                                                global_settings.get('min_risk_reward_ratio', 2.0)),
                    'max_spread': symbol_settings.get('max_spread', 20)
                }
        except Exception as e:
            logger.error(f"❌ Error loading symbol settings for {self.symbol}: {e}")
            
        # Default fallback settings for crypto vs forex - REASONABLE VALUES
        if any(crypto in self.symbol.upper() for crypto in ['BTC', 'ETH', 'SOL', 'ADA', 'DOGE']):
            return {'default_sl_pips': 80, 'default_tp_pips': 160, 'default_sl_buffer': 100, 'default_tp_buffer': 120, 'dca_distance_pips': 100, 'min_risk_reward_ratio': 2.0, 'max_spread': 50}
        else:  # Forex
            return {'default_sl_pips': 50, 'default_tp_pips': 90, 'default_sl_buffer': 60, 'default_tp_buffer': 40, 'dca_distance_pips': 30, 'min_risk_reward_ratio': 2.0, 'max_spread': 5}

    def _get_dca_distance_pips(self) -> float:
        """🧠 Comprehensive DCA distance calculation supporting multiple modes"""
        
        # � Get DCA mode from risk settings
        dca_mode = self.risk_settings.get('dca_mode', 'fixed_pips')
        logger.info(f"🎯 DCA Mode: {dca_mode} for {self.symbol}")
        
        try:
            # 🔧 MODE 1: ATR MULTIPLE - Dynamic based on ATR
            if dca_mode == 'atr_multiple':
                return self._calculate_atr_based_dca_distance()
            
            # 🔧 MODE 2: FIXED PIPS - Static distance
            elif dca_mode in ['fixed_pips', 'Pips cố định']:
                return self._calculate_fixed_pips_dca_distance()
            
            # � MODE 3: PERCENTAGE - Based on entry price percentage  
            elif dca_mode in ['percentage', 'Phần trăm']:
                return self._calculate_percentage_dca_distance()
            
            # 🔧 MODE 4: FIBONACCI - Progressive Fibonacci levels
            elif dca_mode in ['fibonacci', 'Fibonacci', 'fibo_levels']:
                return self._calculate_fibonacci_dca_distance()
            
            # 🔧 MODE 5: SUPPORT/RESISTANCE - Based on S/R levels
            elif dca_mode in ['support_resistance', 'Hỗ trợ/Kháng cự']:
                return self._calculate_sr_based_dca_distance()
            
            # 🔧 FALLBACK: Unknown mode
            else:
                logger.warning(f"⚠️ Unknown DCA mode '{dca_mode}', using fixed pips fallback")
                return self._calculate_fixed_pips_dca_distance()
                
        except Exception as e:
            logger.error(f"❌ Error calculating DCA distance: {e}")
            # Emergency fallback
            return self._get_emergency_dca_fallback()

    def _calculate_atr_based_dca_distance(self) -> float:
        """📊 Calculate DCA distance based on ATR"""
        atr_multiplier = self.risk_settings.get('dca_atr_multiplier', 0.85)
        
        # Get ATR from current analysis
        current_atr = getattr(self, 'current_atr', None)
        if current_atr and current_atr > 0:
            # Calculate ATR-based distance in pips
            pip_size = self._get_pip_value(self.symbol)
            atr_distance_pips = (current_atr * atr_multiplier) / pip_size
            
            logger.info(f"📊 ATR-based DCA: ATR={current_atr:.5f}, multiplier={atr_multiplier}")
            logger.info(f"🎯 ATR DCA distance: {atr_distance_pips:.1f} pips")
            
            # Apply smart market adjustments
            smart_distance = self._calculate_smart_dca_distance(atr_distance_pips)
            logger.info(f"✅ Smart ATR DCA: {smart_distance:.1f} pips")
            return smart_distance
        else:
            logger.warning(f"⚠️ ATR mode selected but no ATR available, using fallback")
            return self._calculate_fixed_pips_dca_distance()

    def _calculate_fixed_pips_dca_distance(self) -> float:
        """📏 Calculate fixed pips DCA distance"""
        gui_dca_distance = self.risk_settings.get('dca_distance_pips', None)
        
        if gui_dca_distance is not None and gui_dca_distance > 0:
            base_distance = float(gui_dca_distance)
            logger.info(f"📱 Using GUI DCA distance: {base_distance} pips")
        else:
            # Symbol-specific fallback
            symbol_settings = self._get_symbol_specific_settings()
            base_distance = symbol_settings.get('dca_distance_pips', 50.0)
            logger.info(f"� Using symbol-specific DCA distance: {base_distance} pips")
        
        # Apply smart adjustments
        smart_distance = self._calculate_smart_dca_distance(base_distance)
        logger.info(f"✅ Smart Fixed DCA: {smart_distance:.1f} pips")
        return smart_distance

    def _calculate_percentage_dca_distance(self) -> float:
        """💰 Calculate DCA distance based on entry price percentage"""
        dca_percentage = self.risk_settings.get('dca_percentage', 1.5) / 100.0  # 1.5% default
        
        # Get current price as reference (entry price approximation)
        current_price = getattr(self, 'current_price', 1.0)
        if current_price <= 0:
            current_price = 1.0  # Fallback
        
        # Calculate percentage-based distance
        price_distance = current_price * dca_percentage
        
        # Convert to pips
        pip_size = self._get_pip_value(self.symbol)
        distance_pips = price_distance / pip_size
        
        logger.info(f"💰 Percentage DCA: {dca_percentage*100:.1f}% of {current_price:.5f} = {distance_pips:.1f} pips")
        
        # Apply smart adjustments
        smart_distance = self._calculate_smart_dca_distance(distance_pips)
        logger.info(f"✅ Smart Percentage DCA: {smart_distance:.1f} pips")
        return smart_distance

    def _calculate_fibonacci_dca_distance(self) -> float:
        """📐 Calculate progressive Fibonacci DCA distances"""
        # Fibonacci ratios for DCA levels: 23.6%, 38.2%, 61.8%, 100%
        fib_ratios = [0.236, 0.382, 0.618, 1.0]
        base_distance = self.risk_settings.get('dca_base_distance_pips', 50.0)
        
        # Get current DCA level (assume level 1 for now, can be enhanced)
        current_level = getattr(self, 'current_dca_level', 1)
        if current_level > len(fib_ratios):
            current_level = len(fib_ratios)
        
        # Calculate Fibonacci-based distance
        fib_multiplier = fib_ratios[current_level - 1]
        fib_distance = base_distance * fib_multiplier
        
        logger.info(f"📐 Fibonacci DCA Level {current_level}: {fib_multiplier:.3f} × {base_distance} = {fib_distance:.1f} pips")
        
        # Apply smart adjustments
        smart_distance = self._calculate_smart_dca_distance(fib_distance)
        logger.info(f"✅ Smart Fibonacci DCA: {smart_distance:.1f} pips")
        return smart_distance

    def _calculate_sr_based_dca_distance(self) -> float:
        """🏆 Calculate DCA distance based on Support/Resistance levels"""
        try:
            # Load S/R data
            sr_data = self._load_trendline_sr_data()
            if not sr_data:
                logger.warning(f"⚠️ No S/R data available, using fixed pips fallback")
                return self._calculate_fixed_pips_dca_distance()
            
            support_levels = sr_data.get('support_levels', [])
            resistance_levels = sr_data.get('resistance_levels', [])
            
            if not support_levels and not resistance_levels:
                logger.warning(f"⚠️ Empty S/R levels, using fixed pips fallback")
                return self._calculate_fixed_pips_dca_distance()
            
            # Get current price
            current_price = getattr(self, 'current_price', None)
            if not current_price:
                logger.warning(f"⚠️ No current price available for S/R calculation")
                return self._calculate_fixed_pips_dca_distance()
            
            # Find nearest support/resistance levels
            all_levels = sorted(support_levels + resistance_levels)
            nearest_distance = float('inf')
            
            for level in all_levels:
                distance = abs(level - current_price)
                if distance < nearest_distance:
                    nearest_distance = distance
            
            # Convert to pips
            pip_size = self._get_pip_value(self.symbol)
            sr_distance_pips = nearest_distance / pip_size
            
            logger.info(f"🏆 S/R-based DCA: Nearest level distance = {sr_distance_pips:.1f} pips")
            
            # Apply smart adjustments and minimum distance
            smart_distance = max(20.0, self._calculate_smart_dca_distance(sr_distance_pips))
            logger.info(f"✅ Smart S/R DCA: {smart_distance:.1f} pips")
            return smart_distance
            
        except Exception as e:
            logger.error(f"❌ Error in S/R DCA calculation: {e}")
            return self._calculate_fixed_pips_dca_distance()

    def _get_emergency_dca_fallback(self) -> float:
        """🚨 Emergency DCA fallback when all else fails"""
        # Symbol-specific emergency fallback
        if any(crypto in self.symbol.upper() for crypto in ['BTC', 'ETH', 'SOL']):
            emergency_distance = 100.0  # Crypto default
        else:
            emergency_distance = 50.0   # Forex default
        
        logger.warning(f"🚨 Emergency DCA fallback: {emergency_distance} pips")
        return emergency_distance
        
        return self._calculate_smart_dca_distance(fallback_distance)
    
    def _calculate_smart_dca_distance(self, base_distance: float) -> float:
        """
        🧠 Calculate smart DCA distance with market-based adjustments
        
        Takes base distance from GUI/settings và makes it smarter based on:
        - Current market volatility (ATR)
        - Market session 
        - Instrument characteristics
        """
        try:
            # Ensure reasonable base
            if base_distance <= 0:
                base_distance = 50.0  # Default từ GUI
                
            # 1️⃣ Market session adjustment
            from datetime import datetime
            current_hour = datetime.now().hour
            
            session_multiplier = 1.0
            # Asian session - lower volatility, tighter DCA
            if 0 <= current_hour < 8:
                session_multiplier = 0.9
            # European session - standard
            elif 8 <= current_hour < 16:
                session_multiplier = 1.0  
            # US session - higher volatility, wider DCA
            elif 16 <= current_hour < 24:
                session_multiplier = 1.1
                
            # 2️⃣ Apply session adjustment
            smart_distance = base_distance * session_multiplier
            
            # 3️⃣ Ensure reasonable bounds (keep close to GUI setting)
            min_distance = base_distance * 0.8  # Không nhỏ hơn 80% GUI setting
            max_distance = base_distance * 1.3  # Không lớn hơn 130% GUI setting
            
            final_distance = max(min_distance, min(smart_distance, max_distance))
            
            logger.debug(f"🧠 Smart DCA: Base={base_distance} → Session={session_multiplier:.1f}x → Final={final_distance:.1f} pips")
            
            return final_distance
            
        except Exception as e:
            logger.error(f"❌ Smart DCA calculation error: {e}")
            return base_distance  # Return original if error

    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for different symbol types - COMPREHENSIVE CRYPTO & METALS SUPPORT"""
        symbol_upper = symbol.upper().rstrip('.').replace('_M', '').replace('_m', '')  # 🔧 Normalize all variants
        
        # 🎯 DEBUG: Log symbol matching
        logger.debug(f"🎯 _get_pip_value: '{symbol}' -> '{symbol_upper}'")
        
        # ========== PRECIOUS METALS ==========
        if symbol_upper in ['XAUUSD', 'XAGUSD', 'GOLD', 'SILVER', 'XPTUSD', 'XPDUSD']:
            logger.debug(f"🎯 Matched METALS: {symbol_upper} -> 0.1")
            return 0.1   # Metals: 1 pip = 0.1 (Gold: 3881.03 -> 3881.73 = 7 pips)
        
        # ========== JPY PAIRS ==========
        elif 'JPY' in symbol_upper:
            logger.debug(f"🎯 Matched JPY pair: {symbol_upper} -> 0.01")
            return 0.01  # JPY pairs: 1 pip = 0.01 (USD/JPY: 147.15 -> 147.22 = 7 pips)
        
        # ========== HIGH-VALUE CRYPTO (≥ $1000) ==========
        elif symbol_upper in ['BTCUSD', 'ETHUSD']:
            logger.debug(f"🎯 Matched HIGH-VALUE CRYPTO: {symbol_upper} -> 1.0")
            return 1.0   # BTC/ETH: 1 pip = 1.0 (BTC: 65000 -> 65070 = 70 pips)
        
        # ========== MID-VALUE CRYPTO ($100-$1000) ==========
        elif symbol_upper in ['SOLUSD', 'LTCUSD', 'BNBUSD', 'AVAXUSD', 'DOTUSD', 'MATICUSD', 'LINKUSD', 'TRXUSD', 'SHIBUSD', 'ARBUSD', 'OPUSD', 'APEUSD', 'SANDUSD', 'CROUSD', 'FTTUSD']:
            logger.debug(f"🎯 Matched MID-VALUE CRYPTO: {symbol_upper} -> 0.1")
            return 0.1   # SOL/LTC/BNB etc: 1 pip = 0.1 (SOL: 224.06 -> 224.76 = 7 pips)
        
        # ========== LOW-VALUE CRYPTO ($1-$10) - Standard Forex Pip ==========
        elif symbol_upper in ['ADAUSD']:
            logger.debug(f"🎯 Matched LOW-VALUE CRYPTO: {symbol_upper} -> 0.0001")
            return 0.0001  # ADA etc: 1 pip = 0.0001 (ADA: 0.8391 -> 0.8398 = 7 pips)
        
        # ========== LOW-VALUE CRYPTO (< $10) ==========
        elif any(crypto in symbol_upper for crypto in ['DOGE', 'XRP', 'TRX']):
            logger.debug(f"🎯 Matched LOW-VALUE CRYPTO: {symbol_upper} -> 0.001")
            return 0.001  # DOGE/XRP etc: 1 pip = 0.001 (DOGE: 0.123 -> 0.130 = 7 pips)
        
        # ========== MICRO-VALUE CRYPTO (< $1) ==========
        elif any(micro_crypto in symbol_upper for micro_crypto in ['SHIB', 'PEPE', 'FLOKI']):
            logger.debug(f"🎯 Matched MICRO-VALUE CRYPTO: {symbol_upper} -> 0.00001")
            return 0.00001  # Micro cryptos: 1 pip = 0.00001
        
        # ========== FOREX PAIRS ==========
        else:
            logger.debug(f"🎯 Default FX pair: {symbol_upper} -> 0.0001")
            return 0.0001  # Major FX pairs: 1 pip = 0.0001 (EUR/USD: 1.0850 -> 1.0857 = 7 pips)

    def _get_enhanced_sl_distance(self) -> float:
        """Get enhanced S/L distance for current symbol"""
        try:
            symbol_settings = self.risk_settings.get('symbol_specific_settings', {})
            if self.symbol in symbol_settings:
                return symbol_settings[self.symbol].get('default_sl_pips', 50)
            else:
                return 50  # Default fallback
        except Exception as e:
            logger.debug(f"Error getting enhanced SL distance: {e}")
            return 50

    def calculate_dca_adaptive_sl(self, existing_positions: List[Dict], current_price: float, direction: str) -> Optional[float]:
        """
        Calculate new adaptive S/L for all positions when DCA strategy is active
        
        Args:
            existing_positions: List of position data including entry prices
            current_price: Current market price
            direction: 'BUY' or 'SELL'
            
        Returns:
            New S/L price that protects all DCA levels, or None if no_sl mode
        """
        if not self.dca_settings.get('enable_dca', False):
            return None
            
        dca_sl_mode = self.dca_settings.get('dca_sl_mode', 'fixed')
        
        if dca_sl_mode == 'no_sl':
            return None
            
        if dca_sl_mode != 'adaptive':
            return None  # Don't modify for fixed mode
            
        try:
            # Get DCA settings
            max_dca_levels = self.dca_settings.get('max_dca_levels', 3)
            dca_distance_pips = self._get_dca_distance_pips()
            pip_value = get_pip_value(self.symbol)
            
            # Find the worst possible entry point (furthest from current favorable direction)
            entry_prices = [pos.get('price_open', pos.get('entry', current_price)) for pos in existing_positions]
            
            if direction == 'BUY':
                # For BUY: worst entry is the highest price (bought at peak)
                worst_entry = max(entry_prices) if entry_prices else current_price
                
                # Calculate furthest possible DCA level below worst entry
                total_dca_distance = max_dca_levels * dca_distance_pips * pip_value
                furthest_dca_level = worst_entry - total_dca_distance
                
                # Add safety buffer below furthest DCA
                safety_buffer = 30 * pip_value  # 30 pips buffer
                new_sl = furthest_dca_level - safety_buffer
                
                logger.debug(f"DCA Adaptive SL calculation for BUY:")
                logger.debug(f"  Existing entries: {entry_prices}")
                logger.debug(f"  Worst entry: {worst_entry}")
                logger.debug(f"  Furthest DCA level: {furthest_dca_level}")
                logger.debug(f"  New adaptive S/L: {new_sl}")
                
            else:  # SELL
                # For SELL: worst entry is the lowest price (sold at bottom)
                worst_entry = min(entry_prices) if entry_prices else current_price
                
                # Calculate furthest possible DCA level above worst entry
                total_dca_distance = max_dca_levels * dca_distance_pips * pip_value
                furthest_dca_level = worst_entry + total_dca_distance
                
                # Add safety buffer above furthest DCA
                safety_buffer = 30 * pip_value  # 30 pips buffer
                new_sl = furthest_dca_level + safety_buffer
                
                logger.debug(f"DCA Adaptive SL calculation for SELL:")
                logger.debug(f"  Existing entries: {entry_prices}")
                logger.debug(f"  Worst entry: {worst_entry}")
                logger.debug(f"  Furthest DCA level: {furthest_dca_level}")
                logger.debug(f"  New adaptive S/L: {new_sl}")
                
            return new_sl
            
        except Exception as e:
            logger.error(f"Error calculating DCA adaptive S/L: {e}")
            return None

    def should_update_existing_sl(self, existing_positions: List[Dict], new_sl: float, direction: str, current_price: float = None) -> bool:
        """
        🎯 SMART S/L UPDATE LOGIC - Enhanced with profit requirements and safety buffers
        
        Args:
            existing_positions: List of existing positions
            new_sl: Proposed new S/L price
            direction: 'BUY' or 'SELL'
            current_price: Current market price for profit calculation
            
        Returns:
            True if S/L should be updated (with enhanced conditions)
        """
        if not existing_positions:
            return False
            
        try:
            # Get current S/L from first position (assume all have same S/L in DCA group)
            current_sl = existing_positions[0].get('sl', existing_positions[0].get('stop_loss'))
            entry_price = existing_positions[0].get('price_open', existing_positions[0].get('entry', 0))
            
            if current_sl is None or not isinstance(current_sl, (int, float)):
                return True  # No existing S/L, should set one
                
            if new_sl is None:
                return False  # Don't remove existing S/L unless explicitly no_sl mode
            
            # 🎯 ENHANCED LOGIC: Calculate current profit in pips
            pip_value = get_pip_value(self.symbol)
            if current_price and entry_price and pip_value:
                if direction == 'BUY':
                    profit_pips = (current_price - entry_price) / pip_value
                else:  # SELL
                    profit_pips = (entry_price - current_price) / pip_value
                
                logger.info(f"📊 Position profit check: {profit_pips:.1f} pips for {direction} position")
                
                # 🚨 SAFETY RULE 1: Only move S/L closer to entry if profit >= 70 pips
                min_profit_for_sl_move = self.risk_settings.get('min_profit_for_sl_move_pips', 70)
                if direction == 'BUY':
                    # Moving S/L UP (closer to entry) requires minimum profit
                    if new_sl > current_sl and profit_pips < min_profit_for_sl_move:
                        logger.warning(f"⚠️ Rejecting S/L move UP: profit {profit_pips:.1f} < required {min_profit_for_sl_move} pips")
                        return False
                else:  # SELL
                    # Moving S/L DOWN (closer to entry) requires minimum profit
                    if new_sl < current_sl and profit_pips < min_profit_for_sl_move:
                        logger.warning(f"⚠️ Rejecting S/L move DOWN: profit {profit_pips:.1f} < required {min_profit_for_sl_move} pips")
                        return False
                
                # 🚨 SAFETY RULE 2: Never move S/L too close to entry (minimum 15 pips buffer)
                min_sl_buffer_pips = self.risk_settings.get('min_sl_buffer_from_entry_pips', 35)
                if direction == 'BUY':
                    min_allowed_sl = entry_price - (min_sl_buffer_pips * pip_value)
                    if new_sl > min_allowed_sl:
                        logger.warning(f"⚠️ Rejecting S/L too close to entry: {new_sl} > min_allowed {min_allowed_sl}")
                        return False
                else:  # SELL
                    max_allowed_sl = entry_price + (min_sl_buffer_pips * pip_value)
                    if new_sl < max_allowed_sl:
                        logger.warning(f"⚠️ Rejecting S/L too close to entry: {new_sl} < max_allowed {max_allowed_sl}")
                        return False
                
                # 🎯 PROFIT-BASED S/L MOVE: Move S/L to entry when profit >= 100 pips
                breakeven_profit_threshold = self.risk_settings.get('breakeven_profit_threshold_pips', 100)
                if profit_pips >= breakeven_profit_threshold:
                    # Allow moving S/L to breakeven (entry price)
                    if direction == 'BUY' and new_sl >= entry_price:
                        logger.info(f"✅ Moving S/L to breakeven: profit {profit_pips:.1f} >= {breakeven_profit_threshold} pips")
                        return True
                    elif direction == 'SELL' and new_sl <= entry_price:
                        logger.info(f"✅ Moving S/L to breakeven: profit {profit_pips:.1f} >= {breakeven_profit_threshold} pips")
                        return True
            
            # 🎯 ORIGINAL DCA LOGIC: Check if new S/L gives more room for DCA
            if direction == 'BUY':
                # For BUY: lower S/L is better (gives more room) - but only if not moving closer to entry
                should_update = new_sl < current_sl
                logger.debug(f"SL Update check for BUY: new_sl={new_sl} < current_sl={current_sl} = {should_update}")
            else:  # SELL
                # For SELL: higher S/L is better (gives more room) - but only if not moving closer to entry  
                should_update = new_sl > current_sl
                logger.debug(f"SL Update check for SELL: new_sl={new_sl} > current_sl={current_sl} = {should_update}")
                
            return should_update
            
        except Exception as e:
            logger.error(f"Error checking S/L update requirement: {e}")
            return False

    def check_losing_position_sl_adjustment(self, position: Dict, current_signal: str, suggested_sl: float, current_price: float, profit_pips: float) -> Dict:
        """
        🎯 CHECK S/L ADJUSTMENT FOR LOSING POSITIONS BASED ON NEW SIGNAL
        
        Args:
            position: Single position dictionary
            current_signal: Current signal ('BUY' or 'SELL')
            suggested_sl: Suggested new S/L level
            current_price: Current market price
            profit_pips: Current profit in pips (negative for loss)
            
        Returns:
            Dictionary with adjustment decision: {'should_adjust': bool, 'new_sl': float, 'reason': str}
        """
        try:
            pos_direction = 'BUY' if position.get('type', 0) == 0 else 'SELL'
            entry_price = position.get('price_open', position.get('entry', 0))
            current_sl = position.get('sl', position.get('stop_loss', 0))
            
            logger.info(f"🔍 Losing position S/L check: {pos_direction} {self.symbol}, profit: {profit_pips:.1f} pips, current S/L: {current_sl}")
            
            # Only adjust S/L for same direction signals on losing positions
            if pos_direction != current_signal:
                return {'should_adjust': False, 'reason': f'Position {pos_direction} != Signal {current_signal}'}
            
            # Only adjust losing positions (profit_pips < 0)
            if profit_pips >= 0:
                return {'should_adjust': False, 'reason': f'Position not losing: {profit_pips:.1f} pips'}
            
            # Check if position already has S/L set
            if current_sl > 0:
                # Check if suggested S/L is better than current S/L
                pip_value = get_pip_value(self.symbol)
                
                if pos_direction == 'BUY':
                    # For BUY: better S/L is higher (closer to entry, tighter)
                    if suggested_sl > current_sl:
                        sl_improvement_pips = (suggested_sl - current_sl) / pip_value if pip_value else 0
                        return {
                            'should_adjust': True,
                            'new_sl': suggested_sl,
                            'reason': f'Strengthen BUY support: {suggested_sl:.5f} > {current_sl:.5f} (+{sl_improvement_pips:.1f} pips)'
                        }
                else:  # SELL
                    # For SELL: better S/L is lower (closer to entry, tighter)
                    if suggested_sl < current_sl:
                        sl_improvement_pips = (current_sl - suggested_sl) / pip_value if pip_value else 0
                        return {
                            'should_adjust': True,
                            'new_sl': suggested_sl,
                            'reason': f'Strengthen SELL resistance: {suggested_sl:.5f} < {current_sl:.5f} (+{sl_improvement_pips:.1f} pips)'
                        }
            else:
                # No S/L set, suggest setting one for losing position
                return {
                    'should_adjust': True,
                    'new_sl': suggested_sl,
                    'reason': f'Set S/L for losing {pos_direction} position: {suggested_sl:.5f}'
                }
            
            return {'should_adjust': False, 'reason': 'No improvement in S/L level'}
            
        except Exception as e:
            logger.error(f"Error checking losing position S/L adjustment: {e}")
            return {'should_adjust': False, 'reason': f'Error: {e}'}

    def check_signal_based_adjustment(self, existing_positions: List[Dict], new_signal: str, new_sl: float, new_tp: float, current_price: float) -> Dict:
        """
        🎯 CHECK IF S/L AND T/P SHOULD BE ADJUSTED BASED ON NEW SIGNAL
        
        Args:
            existing_positions: List of existing positions
            new_signal: New signal ('BUY' or 'SELL')
            new_sl: New S/L from signal
            new_tp: New T/P from signal  
            current_price: Current market price
            
        Returns:
            Dictionary with adjustment recommendations: {'adjust_sl': bool, 'adjust_tp': bool, 'new_sl': float, 'new_tp': float, 'reason': str}
        """
        if not existing_positions:
            return {'adjust_sl': False, 'adjust_tp': False}
        
        try:
            position = existing_positions[0]  # Assume all positions have same direction
            pos_direction = 'BUY' if position.get('type', 0) == 0 else 'SELL'
            entry_price = position.get('price_open', position.get('entry', 0))
            current_sl = position.get('sl', position.get('stop_loss', 0))
            current_tp = position.get('tp', position.get('take_profit', 0))
            
            # Calculate current profit
            pip_value = get_pip_value(self.symbol)
            if pos_direction == 'BUY':
                profit_pips = (current_price - entry_price) / pip_value if pip_value else 0
            else:  # SELL
                profit_pips = (entry_price - current_price) / pip_value if pip_value else 0
            
            logger.info(f"📊 Signal-based adjustment check: {pos_direction} position, profit: {profit_pips:.1f} pips, new signal: {new_signal}")
            
            result = {'adjust_sl': False, 'adjust_tp': False, 'new_sl': current_sl, 'new_tp': current_tp, 'reason': 'No adjustment needed'}
            
            # 🎯 CASE 1: Same direction signal - potentially better S/L and T/P
            if pos_direction == new_signal:
                signal_improvement_threshold = self.risk_settings.get('signal_improvement_min_profit_pips', 15)  # Reduced from 50 to 15
                
                # Allow adjustment even for small losses if signal is strong enough (within -20 pips)
                if profit_pips >= signal_improvement_threshold or (profit_pips >= -20 and profit_pips < signal_improvement_threshold):
                    # Check if new S/L is better (tighter but not too close to entry)
                    if pos_direction == 'BUY' and new_sl > current_sl and new_sl < entry_price:
                        # For BUY: new S/L should be higher than current S/L but still below entry
                        min_buffer = self.risk_settings.get('min_sl_buffer_from_entry_pips', 35) * pip_value
                        if new_sl >= (entry_price - min_buffer):
                            result['adjust_sl'] = True
                            result['new_sl'] = new_sl
                            result['reason'] = f"Tightening S/L with profit {profit_pips:.1f} pips"
                    elif pos_direction == 'SELL' and new_sl < current_sl and new_sl > entry_price:
                        # For SELL: new S/L should be lower than current S/L but still above entry
                        min_buffer = self.risk_settings.get('min_sl_buffer_from_entry_pips', 35) * pip_value
                        if new_sl <= (entry_price + min_buffer):
                            result['adjust_sl'] = True
                            result['new_sl'] = new_sl
                            result['reason'] = f"Tightening S/L with profit {profit_pips:.1f} pips"
                    
                    # Check if new T/P is better (further target)
                    if pos_direction == 'BUY' and new_tp > current_tp:
                        result['adjust_tp'] = True
                        result['new_tp'] = new_tp
                        result['reason'] += f" + Extending T/P target"
                    elif pos_direction == 'SELL' and new_tp < current_tp:
                        result['adjust_tp'] = True
                        result['new_tp'] = new_tp
                        result['reason'] += f" + Extending T/P target"
                else:
                    logger.info(f"⏳ Signal improvement requires {signal_improvement_threshold} pips profit, current: {profit_pips:.1f}")
            
            # 🎯 CASE 2: Opposite signal - consider closing or partial close
            elif pos_direction != new_signal and profit_pips > 20:
                opposite_signal_profit_threshold = self.risk_settings.get('opposite_signal_min_profit_pips', 30)
                
                if profit_pips >= opposite_signal_profit_threshold:
                    # Suggest tightening S/L to lock in profits when opposite signal appears
                    if pos_direction == 'BUY':
                        # Move S/L closer to current price but keep some buffer
                        buffer_pips = self.risk_settings.get('opposite_signal_sl_buffer_pips', 20)
                        suggested_sl = current_price - (buffer_pips * pip_value)
                        if suggested_sl > current_sl:
                            result['adjust_sl'] = True
                            result['new_sl'] = suggested_sl
                            result['reason'] = f"Opposite signal detected - locking profit {profit_pips:.1f} pips"
                    else:  # SELL
                        buffer_pips = self.risk_settings.get('opposite_signal_sl_buffer_pips', 20)
                        suggested_sl = current_price + (buffer_pips * pip_value)
                        if suggested_sl < current_sl:
                            result['adjust_sl'] = True
                            result['new_sl'] = suggested_sl
                            result['reason'] = f"Opposite signal detected - locking profit {profit_pips:.1f} pips"
                else:
                    logger.info(f"⏳ Opposite signal action requires {opposite_signal_profit_threshold} pips profit, current: {profit_pips:.1f}")
            
            logger.info(f"🎯 Signal adjustment result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error checking signal-based adjustment: {e}")
            return {'adjust_sl': False, 'adjust_tp': False}

    def generate_sl_update_actions(self, existing_positions: List[Dict], new_sl: Optional[float]) -> List[Dict]:
        """
        Generate action recommendations to update S/L for existing positions
        
        Args:
            existing_positions: List of existing positions
            new_sl: New S/L price (or None for no_sl mode)
            
        Returns:
            List of action dictionaries for S/L updates
        """
        actions = []
        
        try:
            for position in existing_positions:
                ticket = position.get('ticket') or position.get('position_id')
                symbol = position.get('symbol', self.symbol)
                
                if not ticket:
                    continue
                    
                action = {
                    'primary_action': 'set_sl',
                    'ticket': ticket,
                    'symbol': symbol,
                    'proposed_sl': new_sl,
                    'reason': f"DCA S/L adjustment - {self.dca_settings.get('dca_sl_mode', 'adaptive')} mode",
                    'priority': 'high',
                    'dca_related': True
                }
                
                actions.append(action)
                
            logger.debug(f"Generated {len(actions)} S/L update actions for DCA strategy")
            return actions
            
        except Exception as e:
            logger.error(f"Error generating S/L update actions: {e}")
            return []

    def _check_individual_position_risk_limit(self, position: dict, account_balance: float) -> bool:
        """
        Kiểm tra giới hạn rủi ro cho từng lệnh riêng lẻ
        
        Args:
            position: Thông tin position từ MT5
            account_balance: Số dư tài khoản hiện tại
            
        Returns:
            True nếu position vi phạm giới hạn rủi ro và cần đóng
        """
        try:
            max_risk_percent = self.risk_settings.get("max_risk_percent", "OFF")
            
            # Nếu OFF thì bỏ qua kiểm tra
            if max_risk_percent == "OFF" or not isinstance(max_risk_percent, (int, float)):
                return False
                
            # Tính thua lỗ hiện tại của position
            current_profit = position.get('profit', 0)
            
            # Nếu đang lãi thì không cần đóng
            if current_profit >= 0:
                return False
                
            # Tính % thua lỗ so với tài khoản
            loss_percent = abs(current_profit) / account_balance * 100
            
            # Kiểm tra vượt giới hạn
            if loss_percent >= max_risk_percent:
                logger.warning(f"⚠️ Position {position.get('ticket')} loss {loss_percent:.2f}% exceeds limit {max_risk_percent}% - Need to close!")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking individual position risk limit: {e}")
            return False

    def _check_total_drawdown_limit(self, account_info: dict) -> bool:
        """
        Kiểm tra giới hạn sụt giảm tối đa (drawdown) của tài khoản
        
        Args:
            account_info: Thông tin tài khoản từ MT5 scan
            
        Returns:
            True nếu vượt giới hạn drawdown và cần đóng tất cả lệnh
        """
        try:
            max_drawdown_percent = self.risk_settings.get("max_drawdown_percent", "OFF")
            
            # Nếu OFF thì bỏ qua kiểm tra
            if max_drawdown_percent == "OFF" or not isinstance(max_drawdown_percent, (int, float)):
                return False
                
            # Lấy thông tin tài khoản
            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', 0)
            
            if balance <= 0 or equity <= 0:
                return False
                
            # Tính % drawdown hiện tại
            drawdown_percent = (balance - equity) / balance * 100
            
            # Kiểm tra vượt giới hạn
            if drawdown_percent >= max_drawdown_percent:
                logger.error(f"🚨 DRAWDOWN ALERT: {drawdown_percent:.2f}% exceeds limit {max_drawdown_percent}% - CLOSE ALL POSITIONS!")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking drawdown limit: {e}")
            return False

    def _check_daily_loss_limit(self, account_info: dict, daily_start_balance: float = None) -> bool:
        """
        Kiểm tra giới hạn thua lỗ trong ngày
        
        Args:
            account_info: Thông tin tài khoản từ MT5 scan
            daily_start_balance: Số dư đầu ngày (nếu có tracking)
            
        Returns:
            True nếu vượt giới hạn thua lỗ ngày và cần ngừng giao dịch
        """
        try:
            max_daily_loss_percent = self.risk_settings.get("max_daily_loss_percent", "OFF")
            
            # Nếu OFF thì bỏ qua kiểm tra
            if max_daily_loss_percent == "OFF" or not isinstance(max_daily_loss_percent, (int, float)):
                return False
                
            # Sử dụng balance hiện tại làm reference nếu không có daily_start_balance
            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', 0)
            
            if not daily_start_balance:
                daily_start_balance = balance
                
            # Tính thua lỗ trong ngày
            daily_loss = daily_start_balance - equity
            
            if daily_loss <= 0:  # Không thua lỗ
                return False
                
            # Tính % thua lỗ so với số dư đầu ngày
            daily_loss_percent = daily_loss / daily_start_balance * 100
            
            # Kiểm tra vượt giới hạn
            if daily_loss_percent >= max_daily_loss_percent:
                logger.error(f"🚨 DAILY LOSS ALERT: {daily_loss_percent:.2f}% exceeds limit {max_daily_loss_percent}% - STOP TRADING!")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking daily loss limit: {e}")
            return False

    def _apply_risk_protection(self, account_scan: dict) -> Dict[str, Any]:
        """
        Áp dụng các biện pháp bảo vệ rủi ro dựa trên cài đặt
        
        Args:
            account_scan: Dữ liệu account scan từ MT5
            
        Returns:
            Dict chứa các hành động cần thực hiện (đóng lệnh, ngừng giao dịch, etc.)
        """
        protection_actions = {
            'close_positions': [],  # Danh sách positions cần đóng
            'close_all_positions': False,  # Đóng tất cả positions
            'stop_trading': False,  # Ngừng giao dịch
            'warnings': []  # Cảnh báo
        }
        
        try:
            account_info = account_scan.get('account_info', {})
            positions = account_scan.get('positions', [])
            
            if not account_info:
                return protection_actions
                
            balance = account_info.get('balance', 0)
            
            # 1. Kiểm tra từng position riêng lẻ
            for position in positions:
                if self._check_individual_position_risk_limit(position, balance):
                    protection_actions['close_positions'].append(position.get('ticket'))
                    
            # 2. Kiểm tra tổng drawdown
            if self._check_total_drawdown_limit(account_info):
                protection_actions['close_all_positions'] = True
                protection_actions['stop_trading'] = True
                protection_actions['warnings'].append("Maximum drawdown exceeded - All positions closed!")
                
            # 3. Kiểm tra thua lỗ trong ngày
            if self._check_daily_loss_limit(account_info):
                protection_actions['stop_trading'] = True
                protection_actions['warnings'].append("Daily loss limit exceeded - Trading stopped!")
                
            # Log kết quả
            if protection_actions['close_positions']:
                logger.warning(f"🛡️ Risk Protection: Close {len(protection_actions['close_positions'])} positions")
            if protection_actions['close_all_positions']:
                logger.error(f"🛡️ Risk Protection: CLOSE ALL POSITIONS due to drawdown")
            if protection_actions['stop_trading']:
                logger.error(f"🛡️ Risk Protection: STOP TRADING activated")
                
        except Exception as e:
            logger.error(f"Error applying risk protection: {e}")
            
        return protection_actions

    def run(self) -> Dict[str, Any]:
        logger.debug(f"Starting run() for {self.symbol}")
        loader = Loader(self.symbol)
        # Timeframes ưu tiên cho quyết định (bỏ các TF quá lớn vì chưa có dữ liệu đồng bộ)
        tf_eval = [tf for tf in ["H4", "H1", "M30", "M15"] if tf in CFG.TF]
        raw_tf_data: Dict[str, TFData] = {}
        tfs_present: Dict[str, Any] = {}
        
        # Get available timeframes for the report
        if self.user_timeframes:
            # User specified - filter by available data and user selection
            import glob
            available_files = glob.glob(f"data/{self.symbol}_m_*.json")
            available_tfs = [f.split('_m_')[1].split('.json')[0] for f in available_files]
            selected_timeframes = [tf for tf in self.user_timeframes if tf in available_tfs]
        else:
            # Auto-detect from data files
            import glob
            available_files = glob.glob(f"data/{self.symbol}_m_*.json")
            if available_files:
                available_tfs = [f.split('_m_')[1].split('.json')[0] for f in available_files]
                pref = list(CFG.TF)
                selected_timeframes = [tf for tf in pref if tf in available_tfs]
            else:
                selected_timeframes = tf_eval
        
        # ---- Helper functions ----
        def _ind_list(ind_any):
            if isinstance(ind_any, list):
                return ind_any
            if isinstance(ind_any, dict):
                # nếu là 1 dict single snapshot -> bọc lại
                return [ind_any]
            return []
        def _last_num(ind_list, keys: List[str]):
            if not ind_list:
                return None
            row = ind_list[-1] if isinstance(ind_list[-1], dict) else None
            if not isinstance(row, dict):
                return None
            for k in keys:
                if k in row and row.get(k) is not None:
                    try:
                        return float(row.get(k))
                    except Exception:
                        return None
            return None
        def _ema_stack(ind_list):
            e20 = _last_num(ind_list, ["EMA20", "EMA_20"]) ; e50 = _last_num(ind_list, ["EMA50","EMA_50"]) ; e200 = _last_num(ind_list, ["EMA200","EMA_200"]) 
            if all(isinstance(x,(int,float)) for x in [e20,e50,e200]):
                if e20 > e50 > e200:
                    return 1
                if e20 < e50 < e200:
                    return -1
            return 0
        def _price(ind_list):
            # lấy close cuối
            row = ind_list[-1] if ind_list and isinstance(ind_list[-1], dict) else None
            if isinstance(row, dict):
                # Debug: print a few keys to see what's available
                print(f"DEBUG _price: Available keys: {list(row.keys())[:10]}")
                
                # Try different price fields in order of preference
                for key in ['close', 'c', 'Close', 'CLOSE']:
                    if key in row:
                        try:
                            val = float(row[key])
                            print(f"DEBUG _price: Found {key}={val}")
                            return val
                        except (ValueError, TypeError):
                            continue
                
                # Fallback to open/high/low
                for key in ['open', 'o', 'Open', 'high', 'h', 'High', 'low', 'l', 'Low']:
                    if key in row:
                        try:
                            val = float(row[key])
                            print(f"DEBUG _price: Fallback to {key}={val}")
                            return val
                        except (ValueError, TypeError):
                            continue
                            
                print(f"DEBUG _price: No price found in keys: {list(row.keys())}")
                return None
            return None
        def _pattern_bias(pattern_obj) -> Optional[int]:
            # 1 bullish, -1 bearish, 0 / None neutral
            if isinstance(pattern_obj, dict):
                txt = (pattern_obj.get('signal') or pattern_obj.get('direction') or pattern_obj.get('type') or pattern_obj.get('pattern') or '').lower()
                if 'bull' in txt or 'ascending' in txt or 'rising' in txt or 'triangle bullish' in txt:
                    return 1
                if 'bear' in txt or 'descending' in txt or 'falling' in txt or 'wedge bearish' in txt:
                    return -1
            return None
        def _top_pattern(tf: str, data: TFData):
            # patterns list flat: pattern_price/<SYMBOL>_m_<TF>_patterns.json đã load ở data.price_patterns nếu trước đó logic sửa; nếu không thì thử đọc trực tiếp.
            if isinstance(data.price_patterns, list) and data.price_patterns:
                # chọn theo confidence cao nhất
                def _score(x):
                    if not isinstance(x, dict):
                        return 0.0
                    v = x.get('confidence') or x.get('score') or x.get('confidence_pct')
                    try:
                        val = float(v)
                        if val <= 1.5:
                            val *= 100
                        return val
                    except Exception:
                        return 0.0
                return max(data.price_patterns, key=_score)
            # fallback đọc file flat nếu chưa load
            flat = os.path.join(CFG.PPRICE, f"{self.symbol}_m_{tf}_patterns.json")
            if os.path.exists(flat):
                try:
                    arr = load_json(flat)
                    if isinstance(arr, list) and arr:
                        return arr[0]
                except Exception:
                    return None
            return None

        # ---- Load data & lightweight per-TF signal snapshot ----
        # Use user-selected timeframes if provided, otherwise detect from files
        import os
        import glob
        
        if self.user_timeframes:
            # User specified timeframes - validate they have data files
            data_pattern = os.path.join(CFG.DATA, f"{self.symbol}_m_*.json")
            available_files = glob.glob(data_pattern)
            available_timeframes = []
            
            for file_path in available_files:
                basename = os.path.basename(file_path)
                if basename.startswith(f"{self.symbol}_m_") and basename.endswith(".json"):
                    tf = basename[len(f"{self.symbol}_m_"):-5]
                    if tf in CFG.TF:
                        available_timeframes.append(tf)
            
            # Filter user selection to only include timeframes with actual data
            selected_timeframes = [tf for tf in self.user_timeframes if tf in available_timeframes]
            if not selected_timeframes:
                print(f"WARNING: No data files found for user-selected timeframes {self.user_timeframes}")
                selected_timeframes = available_timeframes
        else:
            # Auto-detect available timeframes from data files (legacy behavior)
            data_pattern = os.path.join(CFG.DATA, f"{self.symbol}_m_*.json")
            available_files = glob.glob(data_pattern)
            selected_timeframes = []
            
            for file_path in available_files:
                basename = os.path.basename(file_path)
                if basename.startswith(f"{self.symbol}_m_") and basename.endswith(".json"):
                    tf = basename[len(f"{self.symbol}_m_"):-5]
                    if tf in CFG.TF:
                        selected_timeframes.append(tf)
            
            # If no data files found, try to find pattern files
            if not selected_timeframes:
                print(f"No data files found, checking for pattern files...")
                # Check pattern_price folder - support both with and without dot
                symbol_variations = [self.symbol, f"{self.symbol}.", self.symbol.rstrip('.')]
                for symbol_var in symbol_variations:
                    pattern_files = glob.glob(f"pattern_price/{symbol_var}_*_patterns.json")
                    if pattern_files:
                        print(f"Found pattern files for {symbol_var}: {len(pattern_files)} files")
                        break
                
                for pf in pattern_files:
                    basename = os.path.basename(pf)
                    for tf_candidate in ['H1', 'H4', 'D1', 'M15', 'M30', 'M5']:
                        if f'_{tf_candidate}_' in basename and tf_candidate in CFG.TF:
                            if tf_candidate not in selected_timeframes:
                                selected_timeframes.append(tf_candidate)
                            break
                
                # Check pattern_signals folder - support both with and without dot
                if not selected_timeframes:
                    for symbol_var in symbol_variations:
                        signal_files = glob.glob(f"pattern_signals/{symbol_var}_*_priority_patterns.json")
                        if signal_files:
                            print(f"Found signal files for {symbol_var}: {len(signal_files)} files")
                            break
                    
                    for sf in signal_files:
                        basename = os.path.basename(sf)
                        for tf_candidate in ['H1', 'H4', 'D1', 'M15', 'M30', 'M5']:
                            if f'_{tf_candidate}_' in basename and tf_candidate in CFG.TF:
                                if tf_candidate not in selected_timeframes:
                                    selected_timeframes.append(tf_candidate)
                                break
                
                print(f"Found pattern timeframes: {selected_timeframes}")
        
        # Sort timeframes by preference order (larger first)
        pref_order = ["D1", "H4", "H1", "M30", "M15", "M5", "M1"]
        selected_timeframes = [tf for tf in pref_order if tf in selected_timeframes]
        
        # ---- EARLY PRICE EXTRACTION ----
        # Extract current price before main analysis to ensure it's available for smart entry calculation
        # Try to load data from common timeframes directly, regardless of selected_timeframes
        for early_tf in ['H1', 'M30', 'M15', 'H4', 'D1']:
            try:
                early_data = loader.load_tf(early_tf)
                if early_data and early_data.indicators:
                    early_ind_list = _ind_list(early_data.indicators)
                    early_price = _price(early_ind_list)
                    if early_price and not self.current_price:
                        self.current_price = early_price
                        print(f"DEBUG: Early extracted indicator current_price={early_price} from {early_tf} for {self.symbol}")
                        break
            except Exception:
                continue
        
        for tf in selected_timeframes:
            try:
                d = loader.load_tf(tf)
            except Exception:
                continue
            if not any([d.candles, d.indicators, d.price_patterns, d.priority_patterns, d.sr]):
                continue
            raw_tf_data[tf] = d
            tnode: Dict[str, Any] = {}
            ind_list = _ind_list(d.indicators)
            macd_h = Extract.macd_hist(d.indicators) if d.indicators is not None else None
            rsi_v = Extract.rsi(d.indicators) if d.indicators is not None else None
            bias = None
            if isinstance(macd_h,(int,float)) and abs(macd_h) > 1e-9:
                bias = 'BUY' if macd_h > 0 else 'SELL'
            elif isinstance(rsi_v,(int,float)):
                if rsi_v > 55:
                    bias = 'BUY'
                elif rsi_v < 45:
                    bias = 'SELL'
            tnode['indicator'] = {'signal': bias or 'NEUTRAL'}
            if isinstance(d.sr, dict):
                tnode['sr'] = d.sr
            tfs_present[tf] = tnode

        # ---- ENHANCED SCORING: Pattern-Driven Signal Generation ----
        # Dynamic timeframe priority system - adapts to any timeframe combination
        
        # Create dynamic weights based on selected timeframes only
        available_timeframes = [tf for tf in selected_timeframes if tf in tfs_present]
        
        # Timeframe importance hierarchy (descending order)
        timeframe_hierarchy = ["D1", "H4", "H1", "M30", "M15", "M5", "M1"]
        
        # Auto-generate weights prioritizing larger timeframes
        def create_dynamic_weights(timeframes: list[str]) -> dict[str, float]:
            """Create exponentially decreasing weights for available timeframes"""
            if not timeframes:
                return {}
            
            # Sort timeframes by hierarchy (larger first)
            sorted_tfs = []
            for tf in timeframe_hierarchy:
                if tf in timeframes:
                    sorted_tfs.append(tf)
            
            # Generate exponential weights (base weight decreases by ~40% each step)
            weights = {}
            base_weight = 8.0  # Start with higher base for maximum differentiation
            decay_factor = 0.6  # Each step down is 60% of previous
            
            for i, tf in enumerate(sorted_tfs):
                weights[tf] = base_weight * (decay_factor ** i)
                
            return weights
            
        weights = create_dynamic_weights(available_timeframes)
        
        # Primary signal components (Pattern-based decision makers)
        pattern_bull = 0.0; pattern_bear = 0.0
        candle_bull = 0.0; candle_bear = 0.0
        trend_bull = 0.0; trend_bear = 0.0
        
        # Secondary components (Confidence modifiers)
        indicator_confidence_boost = 0.0  # +/- confidence modifier
        technical_confirmation = 0.0      # Additional confirmation score
        
        contributors = 0
        contributor_detail: list[tuple[str,str,float]] = []
        context: Dict[str, Dict[str, Any]] = {}
        
        # Get current price from H1 timeframe (fallback to first available)
        # Note: self.current_price might already contain live price from MT5 (passed in constructor)
        atr = None
        ema20_h1 = None
        support_levels = []
        resistance_levels = []
        
        for check_tf in ['H1', 'M30', 'M15'] + selected_timeframes:
            if check_tf in raw_tf_data:
                d = raw_tf_data[check_tf]
                ind_list = _ind_list(d.indicators)
                indicator_price = _price(ind_list)
                if indicator_price:
                    # Store the extracted price in the instance ONLY if no live price from constructor
                    if not self.current_price:
                        self.current_price = indicator_price
                        print(f"DEBUG: Stored indicator current_price={indicator_price} in Aggregator instance")
                    else:
                        print(f"DEBUG: Keeping live price {self.current_price}, ignoring indicator price {indicator_price}")
                    
                    # Get additional data from this timeframe
                    print(f"DEBUG: About to extract ATR from {check_tf}, ind_list len={len(ind_list) if isinstance(ind_list, list) else 'N/A'}")
                    if ind_list and isinstance(ind_list[-1], dict):
                        print(f"DEBUG: Last record keys (ATR related): {[k for k in ind_list[-1].keys() if 'atr' in k.lower() or 'ATR' in k]}")
                    
                    atr = _last_num(ind_list, ["ATR14", "ATR", "atr14", "ATR_14"])
                    ema20_h1 = _last_num(ind_list, ["EMA20", "EMA_20"])
                    
                    # Debug ATR extraction
                    if atr:
                        print(f"DEBUG: Extracted ATR={atr} from {check_tf} timeframe")
                    else:
                        print(f"DEBUG: No ATR found in {check_tf} timeframe, available keys: {list(ind_list[-1].keys())[:10] if ind_list and isinstance(ind_list[-1], dict) else 'N/A'}")
                    
                    # Get support/resistance from first available timeframe
                    if d.sr and isinstance(d.sr, dict):
                        support_levels = d.sr.get('support_levels', [])
                        resistance_levels = d.sr.get('resistance_levels', [])
                        
                        # Convert to float to avoid string comparison errors
                        try:
                            support_levels = [float(x) for x in support_levels if x is not None]
                        except (ValueError, TypeError):
                            support_levels = []
                        
                        try:
                            resistance_levels = [float(x) for x in resistance_levels if x is not None]
                        except (ValueError, TypeError):
                            resistance_levels = []
                    break

        # Whitelist gating: if self.whitelist is provided only score allowed tokens.
        def _use(token: str) -> bool:
            if not self.whitelist:
                return True
            t = token.lower()
            wl = self.whitelist
            if t in wl:
                return True
            # Composite allowances
            if t.startswith('ema') and any(x in wl for x in ('ema20','ema50','ema100','ema200','ema')):
                return True
            if t == 'patterns' and 'pattern' in wl:
                return True
            if t == 'stochrsi' and 'stochastic' in wl:
                return True
            if t == 'stochastic' and 'stochrsi' in wl:
                return True
            return False

        # Load data for selected timeframes only
        print(f"DEBUG: About to process selected_timeframes: {selected_timeframes}")
        print(f"DEBUG: Available raw_tf_data keys: {list(raw_tf_data.keys())}")
        
        for tf in selected_timeframes:
            print(f"DEBUG: Processing timeframe {tf}")
            d = raw_tf_data.get(tf)
            if not d:
                print(f"DEBUG: No data found for timeframe {tf}")
                continue
                
            print(f"DEBUG: Found data for {tf}, checking patterns...")
            print(f"DEBUG: d.priority_patterns type: {type(d.priority_patterns)}, length: {len(d.priority_patterns) if d.priority_patterns else 0}")
            print(f"DEBUG: d.price_patterns type: {type(d.price_patterns)}, length: {len(d.price_patterns) if d.price_patterns else 0}")
            
            w = weights.get(tf, 1.0)
            ind_list = _ind_list(d.indicators)
            close_px = _price(ind_list)
            
            # ========== NEW LOGIC: CANDLESTICK-DRIVEN SIGNAL GENERATION ==========
            
            # 1. CANDLESTICK PATTERNS - PRIMARY SIGNAL DECISION MAKER (Weight: 4.0 * timeframe_weight)
            if d.priority_patterns and isinstance(d.priority_patterns, list):
                print(f"DEBUG: Processing {len(d.priority_patterns)} candlestick patterns for {tf} (PRIMARY SIGNAL)")
                for candle_pattern in d.priority_patterns:
                    if isinstance(candle_pattern, dict):
                        signal = candle_pattern.get('signal', '').lower()
                        confidence = candle_pattern.get('confidence', 0)
                        pattern_type = candle_pattern.get('type', 'unknown')
                        print(f"DEBUG: 🕯️ PRIMARY Candle {pattern_type}: signal='{signal}', confidence={confidence}")
                        
                        if isinstance(confidence, (int, float)) and confidence > 0.5:  # Lower threshold for primary signals
                            candle_weight = confidence * w * 4.0  # HIGHEST base weight - CANDLESTICK DOMINATES
                            print(f"DEBUG: Primary Candle weight calculation: {confidence} * {w} * 4.0 = {candle_weight}")
                            
                            if 'bullish' in signal or 'bull' in signal:
                                candle_bull += candle_weight
                                contributors += 1
                                contributor_detail.append((f"�️PRIMARY[{pattern_type}][{tf}]", "BUY", candle_weight))
                                print(f"DEBUG: Added PRIMARY BULL candle weight {candle_weight}, total candle_bull now: {candle_bull}")
                            elif 'bearish' in signal or 'bear' in signal:
                                candle_bear += candle_weight
                                contributors += 1
                                contributor_detail.append((f"�️PRIMARY[{pattern_type}][{tf}]", "SELL", candle_weight))
                                print(f"DEBUG: Added PRIMARY BEAR candle weight {candle_weight}, total candle_bear now: {candle_bear}")
                            else:
                                print(f"DEBUG: Primary Candle signal '{signal}' not recognized as bull/bear")
                        else:
                            print(f"DEBUG: Primary Candle confidence {confidence} too low (threshold 0.5)")
            else:
                print(f"DEBUG: No candlestick patterns for {tf} or not a list")

            # 2. PRICE PATTERNS - TREND CONFIRMATION (Weight: 1.5 * timeframe_weight) 
            if d.price_patterns and isinstance(d.price_patterns, list):
                print(f"DEBUG: Processing {len(d.price_patterns)} price patterns for {tf} (TREND CONFIRMATION)")
                for pattern in d.price_patterns:
                    if isinstance(pattern, dict):
                        signal = pattern.get('signal', '').lower()
                        confidence = pattern.get('confidence', 0)
                        pattern_name = pattern.get('pattern', 'unknown')
                        print(f"DEBUG: 💎 TREND Pattern {pattern_name}: signal='{signal}', confidence={confidence}")
                        
                        if isinstance(confidence, (int, float)) and confidence > 0.6:  # Higher threshold for trend confirmation
                            pattern_weight = confidence * w * 1.5  # REDUCED - now supporting role
                            print(f"DEBUG: Trend Pattern weight calculation: {confidence} * {w} * 1.5 = {pattern_weight}")
                            
                            if 'bullish' in signal or 'bull' in signal:
                                pattern_bull += pattern_weight
                                contributors += 1
                                contributor_detail.append((f"�TREND[{pattern_name}][{tf}]", "BUY", pattern_weight))
                                print(f"DEBUG: Added TREND BULL pattern weight {pattern_weight}, total pattern_bull now: {pattern_bull}")
                            elif 'bearish' in signal or 'bear' in signal:
                                pattern_bear += pattern_weight
                                contributors += 1
                                contributor_detail.append((f"�TREND[{pattern_name}][{tf}]", "SELL", pattern_weight))
                                print(f"DEBUG: Added TREND BEAR pattern weight {pattern_weight}, total pattern_bear now: {pattern_bear}")
                            else:
                                print(f"DEBUG: Trend Pattern signal '{signal}' not recognized as bull/bear")
                        else:
                            print(f"DEBUG: Trend Pattern confidence {confidence} too low (threshold 0.6)")
            else:
                print(f"DEBUG: No price patterns for {tf} or not a list")

            # 3. TREND ANALYSIS - Supporting Factor (Weight: 0.8 * timeframe_weight) - REDUCED!
            if d.sr and isinstance(d.sr, dict):
                trend_direction = d.sr.get('trend_direction', '').lower()
                trend_strength = d.sr.get('trend_strength', 0)
                summary = d.sr.get('summary', {})
                breakout_summary = summary.get('breakout_summary', '').lower() if isinstance(summary, dict) else ''
                
                # Trend direction scoring - REDUCED weight
                if isinstance(trend_strength, (int, float)) and trend_strength > 0.4:  # Higher threshold for quality
                    # Cap trend strength to prevent overwhelming patterns
                    capped_strength = min(trend_strength, 3.0)  # Max 3.0 to prevent domination
                    trend_weight = capped_strength * w * 0.8  # REDUCED base weight
                    
                    if 'uptrend' in trend_direction or 'up' in trend_direction:
                        trend_bull += trend_weight
                        contributors += 1
                        contributor_detail.append((f"📈Trend[{tf}]", "BUY", trend_weight))
                    elif 'downtrend' in trend_direction or 'down' in trend_direction:
                        trend_bear += trend_weight
                        contributors += 1
                        contributor_detail.append((f"📉Trend[{tf}]", "SELL", trend_weight))
                
                # Breakout analysis (high importance but capped)
                if 'break' in breakout_summary:
                    breakout_weight = w * 1.0  # Reduced from 1.5
                    if 'break up' in breakout_summary or 'breakout up' in breakout_summary:
                        trend_bull += breakout_weight
                        contributors += 1
                        contributor_detail.append((f"🚀Breakout[{tf}]", "BUY", breakout_weight))
                    elif 'break down' in breakout_summary or 'breakout down' in breakout_summary or 'reversal' in breakout_summary:
                        trend_bear += breakout_weight
                        contributors += 1
                        contributor_detail.append((f"💥Breakout[{tf}]", "SELL", breakout_weight))

            # ========== SECONDARY SOURCES (CONFIDENCE MODIFIERS) ==========
            
            # Extract indicators for confidence calculation
            macd_h = Extract.macd_hist(d.indicators) if d.indicators is not None else None
            rsi_v = Extract.rsi(d.indicators) if d.indicators is not None else None
            try:
                adx_v = Extract.adx(d.indicators) if d.indicators is not None else None
            except Exception:
                adx_v = None
            
            ema_state = _ema_stack(ind_list)
            ema20 = _last_num(ind_list, ["EMA20","EMA_20"])
            
            # Confidence boosters (not primary signals, but confirmation)
            confidence_modifier = 0.0
            
            if _use('macd') and isinstance(macd_h,(int,float)) and abs(macd_h) > 1e-9:
                conf_boost = 0.3 * w * abs(macd_h) / (abs(macd_h) + 0.1)  # Normalized boost
                if macd_h > 0:
                    confidence_modifier += conf_boost
                    contributor_detail.append((f"📊MACD[{tf}]","CONF_BUY",conf_boost))
                else:
                    confidence_modifier -= conf_boost
                    contributor_detail.append((f"📊MACD[{tf}]","CONF_SELL",conf_boost))
                    
            if _use('rsi') and isinstance(rsi_v,(int,float)):
                if 50 <= rsi_v <= 70:  # Healthy bullish zone
                    conf_boost = 0.2 * w * ((rsi_v - 50) / 20)
                    confidence_modifier += conf_boost
                    contributor_detail.append((f"📊RSI[{tf}]","CONF_BUY",conf_boost))
                elif 30 <= rsi_v <= 50:  # Healthy bearish zone
                    conf_boost = 0.2 * w * ((50 - rsi_v) / 20)
                    confidence_modifier -= conf_boost
                    contributor_detail.append((f"📊RSI[{tf}]","CONF_SELL",conf_boost))
                    
            if _use('ema20'):
                if ema_state == 1:
                    conf_boost = 0.4 * w
                    confidence_modifier += conf_boost
                    contributor_detail.append((f"📊EMAStack[{tf}]","CONF_BUY",conf_boost))
                elif ema_state == -1:
                    conf_boost = 0.4 * w
                    confidence_modifier -= conf_boost
                    contributor_detail.append((f"📊EMAStack[{tf}]","CONF_SELL",conf_boost))
                    
            if _use('ema20') and isinstance(close_px,(int,float)) and isinstance(ema20,(int,float)):
                price_ema_diff = abs(close_px - ema20) / max(ema20, 1e-9)
                if price_ema_diff < 0.02:  # Close to EMA20, less significant
                    conf_boost = 0.1 * w
                else:
                    conf_boost = 0.15 * w
                    
                if close_px > ema20:
                    confidence_modifier += conf_boost
                    contributor_detail.append((f"📊Price>EMA20[{tf}]","CONF_BUY",conf_boost))
                else:
                    confidence_modifier -= conf_boost
                    contributor_detail.append((f"📊Price<EMA20[{tf}]","CONF_SELL",conf_boost))
            
            # ADX trend strength confirmation
            if _use('adx') and isinstance(adx_v,(int,float)) and adx_v >= 25:
                adx_boost = 0.25 * w * min(adx_v / 50, 1.0)  # Normalize ADX effect
                # Direction from primary signals
                primary_net = (pattern_bull + candle_bull + trend_bull) - (pattern_bear + candle_bear + trend_bear)
                if primary_net > 0:
                    confidence_modifier += adx_boost
                    contributor_detail.append((f"📊ADX[{tf}]","CONF_BUY",adx_boost))
                elif primary_net < 0:
                    confidence_modifier -= adx_boost
                    contributor_detail.append((f"📊ADX[{tf}]","CONF_SELL",adx_boost))
            
            # Accumulate confidence modifier
            indicator_confidence_boost += confidence_modifier

            # Store context for debugging
            context[tf] = {
                'pattern_bull': pattern_bull,
                'pattern_bear': pattern_bear,
                'candle_bull': candle_bull,
                'candle_bear': candle_bear,
                'trend_bull': trend_bull,
                'trend_bear': trend_bear,
                'confidence_modifier': confidence_modifier,
                'close_price': close_px,
                'timeframe_weight': w
            }

        # ========== FINAL SIGNAL DECISION (PATTERN-DRIVEN) ==========
        
        # Calculate primary signal strength (pattern-based)
        primary_bull = pattern_bull + candle_bull + trend_bull
        primary_bear = pattern_bear + candle_bear + trend_bear
        primary_net = primary_bull - primary_bear
        
        # Calculate minimum threshold for signal generation
        # Base on total potential primary signal strength
        total_potential_primary = sum(weights[tf] for tf in selected_timeframes if tf in raw_tf_data) * 3.0  # 3.0 = max combined weight per TF
        primary_thresh = total_potential_primary * 0.05  # Need at least 5% of potential primary signals (lowered from 15%)
        
        print(f"DEBUG: Pattern-driven scoring:")
        print(f"  Primary - Bull: {primary_bull:.2f}, Bear: {primary_bear:.2f}, Net: {primary_net:.2f}")
        print(f"  Primary threshold: {primary_thresh:.2f}")
        print(f"  Confidence modifier: {indicator_confidence_boost:.2f}")
        print(f"  Total contributors: {contributors}")
        
        # Determine base signal from patterns
        if abs(primary_net) < primary_thresh or primary_thresh == 0:
            # Not enough pattern conviction or no patterns at all
            base_signal = 'NEUTRAL'
            print(f"  Base signal: NEUTRAL (insufficient pattern conviction)")
        elif primary_net > 0:
            base_signal = 'BUY'
            print(f"  Base signal: BUY (pattern conviction)")
        else:
            base_signal = 'SELL'
            print(f"  Base signal: SELL (pattern conviction)")
        
        # Apply confidence modifiers to determine final signal
        if base_signal == 'NEUTRAL':
            # For neutral base, indicators can tip the scale only if they're very strong
            strong_indicator_thresh = total_potential_primary * 0.10  # Lowered from 0.25 to make indicator override easier
            if indicator_confidence_boost > strong_indicator_thresh:
                final_sig = 'BUY'
                print(f"  Final: BUY (strong indicator override)")
            elif indicator_confidence_boost < -strong_indicator_thresh:
                final_sig = 'SELL'
                print(f"  Final: SELL (strong indicator override)")
            else:
                final_sig = 'NEUTRAL'
                print(f"  Final: NEUTRAL (no strong override)")
        else:
            # For directional base signal, check if indicators strongly contradict
            contradiction_thresh = total_potential_primary * 0.3
            if base_signal == 'BUY' and indicator_confidence_boost < -contradiction_thresh:
                final_sig = 'NEUTRAL'  # Contradicted by indicators
                print(f"  Final: NEUTRAL (indicators contradict BUY pattern)")
            elif base_signal == 'SELL' and indicator_confidence_boost > contradiction_thresh:
                final_sig = 'NEUTRAL'  # Contradicted by indicators
                print(f"  Final: NEUTRAL (indicators contradict SELL pattern)")
            else:
                final_sig = base_signal  # Patterns dominate
                print(f"  Final: {final_sig} (pattern-driven, indicators align)")

        # ========== ENHANCED CONFIDENCE CALCULATION ==========
        
        # Base confidence from pattern strength
        if abs(primary_net) > 0:
            pattern_confidence = min(85.0, 40.0 + (abs(primary_net) / max(total_potential_primary, 1e-6)) * 45.0)
        else:
            pattern_confidence = 30.0
        
        # Confidence boost from indicator alignment
        indicator_alignment = abs(indicator_confidence_boost) / max(total_potential_primary * 0.2, 1e-6)
        indicator_boost = min(15.0, indicator_alignment * 15.0)
        
        # Coverage bonus (more timeframes = higher confidence)
        if len(selected_timeframes) > 0:
            timeframe_coverage = len([tf for tf in selected_timeframes if tf in raw_tf_data]) / len(selected_timeframes)
        else:
            timeframe_coverage = 0.0
        coverage_bonus = timeframe_coverage * 10.0
        
        # Final confidence calculation
        base_confidence = pattern_confidence + indicator_boost + coverage_bonus
        
        # Penalty for conflicting signals
        if final_sig == 'NEUTRAL' and base_signal != 'NEUTRAL':
            base_confidence *= 0.7  # Reduce confidence for contradicted patterns
        
        confidence = max(25.0, min(95.0, base_confidence))
        
        print(f"  Confidence breakdown:")
        print(f"    Pattern base: {pattern_confidence:.1f}%")
        print(f"    Indicator boost: {indicator_boost:.1f}%") 
        print(f"    Coverage bonus: {coverage_bonus:.1f}%")
        print(f"    Final confidence: {confidence:.1f}%")

        # Calculate smart entry using new intelligent logic
        # PRIORITY 1: Use live price from self.current_price (if set from MT5)
        current_price = self.current_price
        if current_price:
            print(f"DEBUG: Using LIVE PRICE from self.current_price: {current_price}")
        else:
            # PRIORITY 2: Try to get price from global variable set during indicator processing (FALLBACK)
            try:
                global current_price_from_indicators
                if current_price_from_indicators:
                    current_price = current_price_from_indicators
                    print(f"DEBUG: Using indicator price as fallback: {current_price}")
            except NameError:
                print(f"DEBUG: Global current_price_from_indicators not available")
                
            # PRIORITY 3: Final fallback - use Bollinger Band middle as current price
            if not current_price and hasattr(self, 'bb_fallback_price'):
                current_price = self.bb_fallback_price
                print(f"DEBUG: Using bb_fallback_price as final fallback: {current_price}")
        
        logger.debug(f"About to calculate smart entry for {self.symbol}, final_sig={final_sig}, current_price={current_price}")
        smart_entry_data = None
        order_type = 'market'
        entry_reason = 'Giá thị trường'
        confidence_boost = 0.0
        
        # Store ATR for DCA calculations
        self.current_atr = atr
        logger.debug(f"Extracted ATR={atr} from H1 timeframe")
        
        if final_sig in ('BUY', 'SELL') and isinstance(current_price, (int, float)):
            try:
                logger.debug(f"About to calculate smart entry: symbol={self.symbol}, signal={final_sig}, current_price={current_price}, atr={atr}, ema20_h1={ema20_h1}")
                logger.debug(f"Support levels: {support_levels}, Resistance levels: {resistance_levels}")
                
                smart_entry_data = calculate_smart_entry(
                    symbol=self.symbol,
                    signal=final_sig,
                    current_price=current_price,
                    atr=atr,
                    ema20=ema20_h1,
                    support_levels=support_levels,
                    resistance_levels=resistance_levels,
                    tf_data=raw_tf_data
                )
            except Exception as e:
                logger.error(f"❌ Smart entry calculation failed: {e}")
                logger.debug(f"Debug info - ema20_h1 type: {type(ema20_h1)}, value: {ema20_h1}")
                logger.debug(f"Debug info - support_levels types: {[type(x) for x in (support_levels or [])]}")
                logger.debug(f"Debug info - resistance_levels types: {[type(x) for x in (resistance_levels or [])]}")
                raise
            
            if smart_entry_data:
                entry = smart_entry_data['entry_price']
                order_type = smart_entry_data['order_type']
                entry_reason = smart_entry_data['entry_reason']
                confidence_boost = smart_entry_data['confidence_boost']
                
                # Apply confidence boost to final confidence
                confidence = min(95.0, confidence + confidence_boost)
            else:
                # 🔧 CRITICAL FIX: Adjust entry price with spread for market orders
                entry = self._calculate_spread_adjusted_entry(current_price, final_sig)
        else:
            # 🔧 CRITICAL FIX: Adjust entry price with spread for market orders
            entry = self._calculate_spread_adjusted_entry(current_price, final_sig) if isinstance(current_price, (int, float)) else None

        # Calculate SL/TP based on risk_settings and DCA settings
        sl = tp = None
        if isinstance(entry, (int, float)) and final_sig in ('BUY', 'SELL'):
            
            # 🎯 CALCULATE SL/TP USING RISK SETTINGS FROM GUI
            # Create signal data stub for potential signal-based SL/TP
            signal_data_stub = {
                'signal': final_sig,
                'confidence': confidence,
                'stoploss': None,  # Will be filled if signal provides SL/TP
                'takeprofit': None
            }
            
            sl, tp = self._calculate_sl_tp_from_risk_settings(
                entry_price=entry,
                signal=final_sig,
                atr=atr,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                signal_data=signal_data_stub
            )
            
            # Check if DCA is enabled to adjust S/L calculation
            enable_dca = self.dca_settings.get('enable_dca', False)
            max_dca_levels = self.dca_settings.get('max_dca_levels', 3)
            dca_distance_pips = self._get_dca_distance_pips()
            dca_sl_mode = self.dca_settings.get('dca_sl_mode', 'fixed')  # 'fixed', 'adaptive', 'no_sl', 'support_resistance'
            
            # 🎯 DCA OVERRIDE: Adjust SL/TP if DCA is enabled with special modes
            if enable_dca and dca_sl_mode == 'adaptive':
                # DCA-ADAPTIVE S/L: Place S/L beyond furthest DCA level
                pip_value = self._get_pip_value(self.symbol)
                furthest_dca_distance = max_dca_levels * dca_distance_pips * pip_value
                
                # 🔧 ENHANCED: Use enhanced buffer instead of fixed 20 pips
                # Get enhanced distances from symbol-specific settings
                enhanced_sl_pips = self._get_enhanced_sl_distance()
                dca_buffer = max(20, enhanced_sl_pips * 0.3) * pip_value  # At least 20 pips or 30% of enhanced SL
                
                if final_sig == 'BUY':
                    furthest_dca_level = entry - furthest_dca_distance
                    adaptive_sl = furthest_dca_level - dca_buffer
                    # 🎯 Use enhanced SL if it's more conservative than DCA adaptive
                    if sl is not None:
                        sl = min(sl, adaptive_sl)  # Keep whichever is more conservative
                    else:
                        sl = adaptive_sl
                    logger.debug(f"DCA-Adaptive SL for BUY: entry={entry}, adaptive_sl={adaptive_sl}, enhanced_buffer={dca_buffer/pip_value:.1f} pips")
                    
                elif final_sig == 'SELL':
                    furthest_dca_level = entry + furthest_dca_distance
                    adaptive_sl = furthest_dca_level + dca_buffer
                    # 🎯 Use enhanced SL if it's more conservative than DCA adaptive
                    if sl is not None:
                        sl = max(sl, adaptive_sl)  # Keep whichever is more conservative
                    else:
                        sl = adaptive_sl
                    logger.debug(f"DCA-Adaptive SL for SELL: entry={entry}, adaptive_sl={adaptive_sl}, enhanced_buffer={dca_buffer/pip_value:.1f} pips")
                    
            elif enable_dca and dca_sl_mode == 'no_sl':
                # 🚫 NO S/L: Let DCA handle all drawdowns
                sl = None
                logger.debug(f"DCA No-SL mode: S/L disabled for {final_sig} signal")

        def _order_decimals(symbol: str, ref_price: float) -> int:
            s = (symbol or '').upper()
            if s.startswith('XAU'): return 2
            if any(tag in s for tag in ('BTC','ETH','SOL','ADA','DOGE','BNB','XRP','TRX','LTC','DOT','AVAX')): return 2 if (ref_price or 0) >= 100 else 4
            if s.endswith('JPY'): return 3
            fx_maj = ('USD','EUR','GBP','AUD','NZD','CAD','CHF')
            if any(s.startswith(a) or s.endswith(a) for a in fx_maj): return 5
            return 4
        def _round_to_tick(val: float, decimals: int) -> float:
            if not isinstance(val,(int,float)): return val
            factor = 10 ** decimals
            return round(val * factor) / factor
            
        # Handle rounding and validation for different SL scenarios
        rr = None
        if final_sig in ('BUY','SELL') and isinstance(entry,(int,float)) and isinstance(tp,(int,float)):
            decs = _order_decimals(self.symbol, entry)
            entry = _round_to_tick(entry, decs)
            tp = _round_to_tick(tp, decs)
            tick = 10 ** (-decs)
            
            if isinstance(sl, (int, float)):
                # Standard case with S/L
                sl = _round_to_tick(sl, decs)
                if final_sig == 'BUY':
                    if sl >= entry: sl = max(entry - tick, tick)
                    if tp <= entry: tp = entry + tick
                else:  # SELL
                    if sl <= entry: sl = entry + tick
                    if tp >= entry: tp = entry - tick
                rr = abs((tp - entry)/(entry - sl)) if (entry - sl) != 0 else None
            else:
                # DCA No-SL case: sl = None
                if final_sig == 'BUY':
                    if tp <= entry: tp = entry + tick
                else:  # SELL
                    if tp >= entry: tp = entry - tick
                rr = None  # No risk-reward calculation without S/L

        trade_idea = None
        if final_sig in ('BUY','SELL') and isinstance(entry,(int,float)) and isinstance(tp,(int,float)):
            decs = _order_decimals(self.symbol, entry)
            trade_idea = {
                'direction': final_sig,
                'timeframe': 'H1',
                'entry': _round_to_tick(entry, decs),
                'sl': _round_to_tick(sl, decs) if isinstance(sl, (int, float)) else None,
                'tp': _round_to_tick(tp, decs),
                'rr': round(rr,2) if rr else None,
                'precision': decs,
                'order_type': order_type,
                'entry_reason': entry_reason,
                'current_price': _round_to_tick(current_price, decs) if isinstance(current_price, (int, float)) else None,
                'confidence_boost': confidence_boost,
                'smart_entry_used': smart_entry_data is not None,
                'dca_mode': 'enabled' if self.dca_settings.get('enable_dca', False) else 'disabled',
                'dca_sl_mode': self.dca_settings.get('dca_sl_mode', 'fixed')
            }

        # 🔄 DCA SERVICE INTEGRATION NOTE
        dca_sl_adjustments = []
        dca_service_note = None
        if self.dca_settings.get('enable_dca', False) and final_sig in ('BUY', 'SELL'):
            logger.info("🔄 DCA enabled - ensure dca_service.py is running for DCA management")
            dca_service_note = "DCA managed by independent dca_service.py - ensure service is running"
            try:
                # Simulate existing positions for calculation (in real implementation, this would come from MT5)
                # For now, we provide the framework for when this gets integrated with actual position data
                
                # This would be replaced with actual position retrieval in production:
                # existing_positions = mt5.positions_get(symbol=self.symbol)
                existing_positions = []  # Placeholder - would be populated from MT5
                
                if existing_positions:
                    # Calculate new adaptive S/L for existing positions
                    new_adaptive_sl = self.calculate_dca_adaptive_sl(
                        existing_positions=existing_positions,
                        current_price=current_price,
                        direction=final_sig
                    )
                    
                    # Check if S/L update is needed (with enhanced conditions)
                    if self.should_update_existing_sl(existing_positions, new_adaptive_sl, final_sig, current_price):
                        # Generate S/L update actions
                        sl_update_actions = self.generate_sl_update_actions(existing_positions, new_adaptive_sl)
                        dca_sl_adjustments = sl_update_actions
                        
                        logger.info(f"🔄 DCA S/L Adjustment: Generated {len(sl_update_actions)} update actions")
                        logger.info(f"🎯 New adaptive S/L: {new_adaptive_sl}")
                    else:
                        logger.debug(f"🔄 DCA S/L Adjustment: No update needed")
                else:
                    logger.debug(f"🔄 DCA S/L Adjustment: No existing positions found")
                    
            except Exception as e:
                logger.error(f"Error calculating DCA S/L adjustments: {e}")

        # 🚀 ENHANCED RISK-AWARE ACTIONS GENERATION
        risk_aware_actions = []
        try:
            if final_sig in ('BUY', 'SELL'):
                # Prepare signal data for action generator
                signal_data = {
                    'symbol': self.symbol,
                    'signal': final_sig,
                    'confidence': confidence,
                    'entry': entry,
                    'stoploss': sl,
                    'takeprofit': tp,
                    'order_type': order_type,
                    'entry_reason': entry_reason
                }
                
                # Prepare market context
                market_context = {
                    'support_levels': support_levels or [],
                    'resistance_levels': resistance_levels or [],
                    'atr': atr,
                    'ema20': ema20_h1,
                    'current_price': current_price,
                    'pattern_data': {
                        'bull_score': primary_bull,
                        'bear_score': primary_bear,
                        'net_score': primary_net
                    }
                }
                
                # 🎯 INTEGRATED RISK-AWARE ACTION GENERATION
                # Generate comprehensive trading actions based on signal, entry quality, and DCA settings
                risk_aware_actions = []
                
                # 📊 SCAN CURRENT ACCOUNT STATE WITH FRESH MT5 DATA
                # Get fresh account state directly from MT5 instead of cached file
                logger.info("🔄 Getting fresh account state directly from MT5...")
                try:
                    # Import MT5 module
                    import MetaTrader5 as mt5
                    
                    # Get live positions from MT5 
                    live_positions = mt5.positions_get()
                    live_orders = mt5.orders_get()
                    
                    if live_positions is not None:
                        logger.info(f"📊 LIVE MT5 SCAN: {len(live_positions)} positions, {len(live_orders) if live_orders else 0} orders")
                        
                        # Convert MT5 positions to account scan format
                        account_scan = {
                            'active_positions': [],
                            'active_orders': []
                        }
                        
                        for pos in live_positions:
                            account_scan['active_positions'].append({
                                'symbol': pos.symbol,
                                'ticket': pos.ticket,
                                'type': pos.type,
                                'volume': pos.volume,
                                'price_open': pos.price_open,
                                'price_current': pos.price_current,
                                'profit': pos.profit,
                                'sl': pos.sl,
                                'tp': pos.tp
                            })
                        
                        if live_orders:
                            for order in live_orders:
                                account_scan['active_orders'].append({
                                    'symbol': order.symbol,
                                    'ticket': order.ticket,
                                    'type': order.type,
                                    'volume': order.volume,
                                    'price_open': order.price_open
                                })
                    else:
                        logger.warning("⚠️ MT5 positions_get() returned None - falling back to cached file")
                        account_scan = force_reload_account_scan()
                except Exception as e:
                    logger.error(f"❌ Error getting live MT5 data: {e}")
                    logger.info("🔄 Falling back to cached account scan file...")
                    account_scan = force_reload_account_scan()
                existing_positions = []
                existing_orders = []
                
                if account_scan:
                    # 🛡️ APPLY RISK PROTECTION MEASURES
                    risk_protection = self._apply_risk_protection(account_scan)
                    
                    # Check if trading should be stopped
                    if risk_protection['stop_trading']:
                        logger.error("🚨 TRADING STOPPED due to risk protection limits!")
                        for warning in risk_protection['warnings']:
                            logger.error(f"⚠️ Risk Alert: {warning}")
                        
                        # Return early with risk protection info
                        return {
                            'symbol': self.symbol,
                            'signal': 'RISK_STOP',
                            'confidence': 0,
                            'status': 'TRADING_STOPPED',
                            'risk_protection': risk_protection,
                            'message': 'Trading stopped due to risk protection limits',
                            'warnings': risk_protection['warnings']
                        }
                    
                    # Filter positions for current symbol
                    all_positions = account_scan.get('active_positions', [])
                    logger.info(f"🔍 DEBUG: account_scan keys = {list(account_scan.keys())}")
                    logger.info(f"🔍 DEBUG: active_positions length = {len(all_positions)}")
                    if all_positions:
                        logger.info(f"🔍 DEBUG: first position = {all_positions[0]}")
                    logger.info(f"🔍 DEBUG: self.symbol = '{self.symbol}', all_positions symbols: {[pos.get('symbol', 'NO_SYMBOL') for pos in all_positions[:5]]}")
                    # Match symbols with or without trailing dot (MT5 adds dots to symbols)
                    existing_positions = [pos for pos in all_positions if pos.get('symbol', '').rstrip('.') == self.symbol.rstrip('.')]
                    
                    # Filter orders for current symbol (match with or without trailing dot)
                    all_orders = account_scan.get('active_orders', [])
                    existing_orders = [order for order in all_orders if order.get('symbol', '').rstrip('.') == self.symbol.rstrip('.')]
                    
                    logger.info(f"📋 Account scan for {self.symbol}: {len(existing_positions)} positions, {len(existing_orders)} orders")
                    
                    # Add risk protection actions to result if any positions need closing
                    if risk_protection['close_positions'] or risk_protection['close_all_positions']:
                        logger.warning(f"🛡️ Risk protection triggered for {self.symbol}")
                        for warning in risk_protection['warnings']:
                            logger.warning(f"⚠️ Risk Alert: {warning}")
                else:
                    logger.warning("⚠️ No account scan available - proceeding without position context")
                
                # Update market context with account state
                market_context['existing_positions'] = existing_positions
                market_context['existing_orders'] = existing_orders
                market_context['has_existing_position'] = len(existing_positions) > 0
                market_context['position_direction'] = None
                if existing_positions:
                    # MT5 type: 0=BUY, 1=SELL
                    first_pos = existing_positions[0]
                    market_context['position_direction'] = 'BUY' if first_pos.get('type') == 0 else 'SELL'
                
                try:
                    # 🎯 SMART ACTION STRATEGY: Generate only 1 primary action per symbol
                    # Decision logic based on existing positions and DCA settings
                    
                    # 🔒 CRITICAL: Always use REAL-TIME position check to prevent Dup Entry
                    realtime_positions = self._get_existing_positions_for_symbol(self.symbol)
                    has_positions = len(realtime_positions) > 0
                    
                    # Use realtime positions instead of potentially stale account_scan data
                    if has_positions:
                        logger.info(f"🚫 REAL-TIME CHECK: {self.symbol} has {len(realtime_positions)} existing position(s) - no new Entry allowed")
                        existing_positions = realtime_positions  # Update with fresh data
                    
                    dca_enabled = self.dca_settings.get('enable_dca', False)
                    dca_confidence_ok = confidence >= self.dca_settings.get('min_confidence_for_dca', 2.0)
                    
                    if has_positions:
                        # EXISTING POSITION: Check for conflicting directions and handle accordingly
                        existing_direction = market_context.get('position_direction')
                        signal_conflicts_with_positions = (
                            (existing_direction == 'BUY' and final_sig == 'SELL') or 
                            (existing_direction == 'SELL' and final_sig == 'BUY')
                        )
                        
                        if signal_conflicts_with_positions:
                            logger.warning(f"🚫 DIRECTION CONFLICT: {self.symbol} has {existing_direction} positions but signal is {final_sig}")
                            logger.warning(f"🛑 BLOCKING: No new {final_sig} entries until all {existing_direction} positions are closed")
                            logger.info(f"✅ ALLOWING: {existing_direction} DCA activities still permitted for existing positions")
                            
                            # 🎯 SMART POLICY: Block opposite entries but allow same-direction DCA
                            # - NO new entries in opposite direction  
                            # - YES DCA activities for existing position direction
                            # - System maintains existing positions while blocking conflicts
                            
                            risk_aware_actions = []  # Block new entries, but DCA will be handled below
                            
                            # 🎯 PRIORITY: Same-direction DCA actions (if enabled and conditions met)
                            if dca_enabled and dca_confidence_ok:
                                # Generate DCA for EXISTING position direction (not signal direction)
                                dca_action = self._generate_single_dca_action(
                                    existing_direction, entry, sl, tp, confidence, market_context, atr, existing_positions
                                )
                                if dca_action:
                                    risk_aware_actions.append(dca_action)
                                    logger.info(f"✅ Added same-direction DCA action for existing {existing_direction} positions")
                            
                            logger.info(f"� SOLUTION: Manually close {existing_direction} positions or wait for S/L to clear {self.symbol}")
                            logger.info(f"💡 THEN: System will allow {final_sig} entries when no conflicting positions remain")
                        else:
                            logger.info(f"📊 Existing {len(existing_positions)} positions for {self.symbol} ({existing_direction}) - signal {final_sig} compatible")
                            
                            # 🎯 PRIORITY 1: Signal-based S/L adjustment (lowered threshold for better triggering)
                            if confidence >= 0.5 and final_sig in ('BUY', 'SELL') and not signal_conflicts_with_positions:
                                sl_adjustment_actions = self._generate_signal_based_sl_adjustment_actions(
                                    existing_positions=existing_positions,
                                    new_signal=final_sig,
                                    new_entry=entry,
                                    new_sl=sl,
                                    new_tp=tp,
                                    confidence=confidence
                                )
                                if sl_adjustment_actions:
                                    risk_aware_actions.extend(sl_adjustment_actions)
                                    logger.info(f"🎯 Added {len(sl_adjustment_actions)} S/L adjustment actions for {self.symbol}")
                            
                            # 🎯 PRIORITY 2: DCA actions (if enabled and conditions met)
                            # ✅ REMOVED time-based blocking - was blocking legitimate DCA opportunities
                            if dca_enabled and dca_confidence_ok and not signal_conflicts_with_positions:
                                # Generate single DCA level action (only if same direction)
                                dca_action = self._generate_single_dca_action(
                                    final_sig, entry, sl, tp, confidence, market_context, atr, existing_positions
                                )
                                if dca_action:
                                    risk_aware_actions.append(dca_action)
                            else:
                                # Skip simple position management here - use detailed analysis in account scan section  
                                logger.info(f"🔄 Existing positions detected for {self.symbol} - will be handled by detailed account analysis")
                                pass
                    else:
                        # NO EXISTING POSITIONS: Generate single primary entry based on DCA settings
                        logger.info(f"✅ NO existing positions for {self.symbol} - safe to create new Entry")
                        primary_entry = None
                        if dca_enabled and dca_confidence_ok:
                            logger.info(f"💎 DCA enabled for {self.symbol} - generating DCA-aware primary entry")
                            # Create primary entry with DCA planning (but only 1 action for now)
                            primary_entry = self._create_dca_primary_entry(
                                final_sig, entry, sl, tp, confidence, order_type, entry_reason, market_context
                            )
                        else:
                            logger.info(f"🎯 Standard entry for {self.symbol} - generating single optimized entry")
                            # Create single optimized entry (limit if entry not optimal, market if good)
                            primary_entry = self._create_smart_primary_entry(
                                final_sig, entry, sl, tp, confidence, order_type, entry_reason, market_context
                            )
                        
                        if primary_entry:
                            risk_aware_actions.append(primary_entry)
                    
                    # 🚨 ENHANCED: Deduplicate actions before returning
                    risk_aware_actions = self._deduplicate_actions(risk_aware_actions)
                    
                    logger.info(f"🎯 Generated {len(risk_aware_actions)} integrated risk-aware actions for {self.symbol} (after deduplication)")
                    
                    # Log summary of actions
                    action_summary = {}
                    for action in risk_aware_actions:
                        action_type = action.get('action_type', 'unknown')
                        action_summary[action_type] = action_summary.get(action_type, 0) + 1
                    
                    logger.info(f"📊 Action summary: {action_summary}")
                
                except Exception as e:
                    logger.error(f"Error generating integrated risk-aware actions: {e}")
                    # 🔒 CRITICAL: Check for existing positions before fallback entry using same real-time method
                    if has_positions:  # Use same variable as main logic
                        logger.warning(f"🚫 BLOCKING FALLBACK ENTRY: {self.symbol} already has existing position(s)")
                        risk_aware_actions = []  # No fallback action if positions exist
                    else:
                        # Fallback to basic primary entry only if no existing positions
                        risk_aware_actions = [self._create_basic_primary_entry(final_sig, entry, sl, tp, confidence, order_type)]
                        logger.info(f"✅ Fallback entry created for {self.symbol} - no existing positions found")
                
        except Exception as e:
            logger.error(f"Error generating risk-aware actions: {e}")

        logger.debug(f"About to return result for {self.symbol}")
        result = {
            'symbol': self.symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframes': tfs_present,
            'available_timeframes': selected_timeframes,  # Add selected/available timeframes for Report
            'final_signal': { 'signal': final_sig, 'confidence': round(confidence,2) },
            'primary_scores': {
                'pattern_bull': round(primary_bull, 3),
                'pattern_bear': round(primary_bear, 3),
                'candle_bull': round(candle_bull, 3),
                'candle_bear': round(candle_bear, 3),
                'trend_bull': round(trend_bull, 3),
                'trend_bear': round(trend_bear, 3),
                'net_primary': round(primary_net, 3)
            },
            'confidence_modifier': round(indicator_confidence_boost, 3),
            'base_signal': base_signal,
            'contributors': contributors,
            'contributor_detail': [ {'factor': f, 'direction': d, 'weight': round(wt,3)} for (f,d,wt) in contributor_detail ],
            'trade_idea': trade_idea,
            'debug_context': context,
            'dca_sl_adjustments': dca_sl_adjustments,  # NEW: S/L adjustment actions for existing positions
            'risk_aware_actions': risk_aware_actions  # NEW: Enhanced risk-aware trading actions
        }
        
        # Add DCA service note if applicable
        if dca_service_note:
            result['dca_service_note'] = dca_service_note
        
        return result

    def _get_existing_positions_for_symbol(self, symbol: str) -> List[Dict]:
        """Get all existing positions for a symbol from MT5 with smart symbol matching"""
        try:
            import MetaTrader5 as mt5
            
            # 🎯 CRITICAL FIX: Try both with and without dot suffix for proper symbol matching
            base_symbol = symbol.rstrip('.')  # Remove trailing dot if exists
            symbol_variants = [base_symbol, f"{base_symbol}."]
            
            all_found_positions = []
            
            for variant in symbol_variants:
                positions = mt5.positions_get(symbol=variant)
                if positions:
                    logger.info(f"✅ Found {len(positions)} positions for symbol variant '{variant}'")
                    for pos in positions:
                        all_found_positions.append({
                            'ticket': pos.ticket, 
                            'type': pos.type, 
                            'price_open': pos.price_open, 
                            'time': pos.time, 
                            'comment': pos.comment or '',
                            'symbol': pos.symbol  # Include actual symbol from MT5
                        })
                else:
                    logger.debug(f"No positions found for symbol variant '{variant}'")
            
            if all_found_positions:
                logger.info(f"🎯 Total positions found for {symbol}: {len(all_found_positions)}")
            
            return all_found_positions
        except Exception as e:
            logger.error(f"❌ Error getting positions for {symbol}: {e}")
            return []

    # ❌ REMOVED: _check_recent_entry_for_symbol - Time-based blocking was too restrictive and blocked legitimate DCA opportunities

    def _create_smart_primary_entry(self, direction: str, entry: float, sl: float, tp: float, 
                                   confidence: float, order_type: str, entry_reason: str, 
                                   market_context: Dict) -> Dict:
        """Create smart primary entry with optimized order type based on entry quality and existing positions"""
        
        # Check existing positions to determine if this is DCA or new entry
        existing_positions = market_context.get('existing_positions', [])
        has_existing_position = market_context.get('has_existing_position', False)
        position_direction = market_context.get('position_direction')
        
        # Determine entry type
        if has_existing_position:
            if position_direction == direction:
                entry_type = "DCA_ADD"  # Same direction = DCA add to position
            else:
                entry_type = "HEDGE"    # Opposite direction = hedge position
        else:
            entry_type = "NEW_ENTRY"    # No existing position = new entry
        
        # Determine if we have a "good entry" based on entry_reason
        has_good_entry = any(keyword in entry_reason.lower() for keyword in [
            'support', 'resistance', 'ema', 'fibonacci', 'trendline', 'confluence'
        ])
        
        # Smart order type logic
        if has_good_entry and confidence >= 70.0:
            # Good entry + decent confidence = market order for quick execution
            final_order_type = "market"
            reason_suffix = f" (Market: good entry at {entry_reason})"
        elif confidence >= 80.0:
            # Very high confidence = market order regardless
            final_order_type = "market" 
            reason_suffix = f" (Market: high confidence {confidence:.1f}%)"
        else:
            # Default to limit order for better price
            final_order_type = "limit"
            reason_suffix = f" (Limit: optimize entry)"
        
        # Calculate position size using risk settings
        try:
            # Get account balance from recent scan
            account_balance = None
            try:
                with open("account_scans/mt5_essential_scan.json", 'r') as f:
                    account_data = json.load(f)
                    account_balance = account_data.get("account", {}).get("balance", 10000)
            except:
                account_balance = 10000  # Default fallback
            
            # Calculate volume based on risk settings
            position_size = self._calculate_volume_from_risk_settings(
                entry_price=entry,
                sl_price=sl,
                entry_type=entry_type,
                account_balance=account_balance
            )
            
            logger.debug(f"Volume calculation for {self.symbol}: entry_type={entry_type}, volume={position_size}")
            
        except Exception as e:
            logger.warning(f"Risk settings calculation failed: {e}, using fallback")
            # Fallback to minimum volume
            position_size = self.risk_settings.get("min_volume_auto", 0.01)
        
        # Store the base calculated size
        base_position_size = position_size
        
        # Adjust volume based on entry type
        if entry_type == "DCA_ADD":
            # DCA: Use proper DCA scaling from risk settings
            # Volume calculation is already done by _calculate_volume_from_risk_settings with DCA_ADD type
            # Don't override it here - the method already handles DCA scaling
            action_type = 'dca_entry'
            reason_prefix = f"DCA {direction} add"
        elif entry_type == "HEDGE":
            # Hedge: smaller size, opposite direction
            position_size = round(base_position_size * 0.3, 2)  # 30% for hedge
            action_type = 'hedge_entry'
            reason_prefix = f"HEDGE {direction} against existing position"
        else:
            # New entry: full size
            # position_size already calculated by risk manager
            action_type = 'primary_entry'
            reason_prefix = f"Primary {direction} entry"
        
        return {
            'action_type': action_type,
            'symbol': self.symbol,
            'direction': direction,
            'entry_price': entry,
            'volume': position_size,
            'stop_loss': sl,
            'take_profit': tp,
            'reason': f"{reason_prefix} with {confidence:.1f}% confidence{reason_suffix}",
            'confidence': confidence,
            'risk_level': 'low' if confidence >= 80 else 'moderate' if confidence >= 65 else 'high',
            'order_type': final_order_type,
            'priority': 1 if entry_type == "NEW_ENTRY" else 2,  # DCA/hedge lower priority
            'conditions': {
                'entry_quality': 'good' if has_good_entry else 'standard',
                'original_order_type': order_type,
                'entry_type': entry_type,
                'existing_positions': len(existing_positions)
            }
        }

    def _generate_dca_scale_actions(self, direction: str, entry: float, sl: float, tp: float, 
                                  confidence: float, market_context: Dict, atr: float = None) -> List[Dict]:
        """
        🚫 LEGACY DCA generation disabled
        DCA now handled by independent dca_service.py
        """
        logger.info("🚫 DCA generation moved to dca_service.py - returning empty list")
        return []

    def _generate_smart_limit_actions(self, direction: str, entry: float, confidence: float, 
                                    market_context: Dict) -> List[Dict]:
        """Generate additional smart limit orders based on S/R levels"""
        actions = []
        
        support_levels = market_context.get('support_levels', [])
        resistance_levels = market_context.get('resistance_levels', [])
        
        if direction == "BUY" and support_levels:
            # Create limit buys near support levels
            for support in support_levels[:3]:  # Max 3 levels
                if support < entry:  # Only below current entry
                    limit_action = {
                        'action_type': 'limit_order',
                        'symbol': self.symbol,
                        'direction': direction,
                        'entry_price': support + (get_pip_value(self.symbol) * 5),  # 5 pips above support
                        'volume': 0.01,
                        'stop_loss': support - (get_pip_value(self.symbol) * 20),  # 20 pips below support
                        'take_profit': entry + (entry - support) * 1.5,  # 1.5x risk-reward
                        'reason': f"BUY limit near support {support}",
                        'confidence': confidence * 0.8,  # Reduced confidence
                        'risk_level': 'moderate',
                        'order_type': 'limit',
                        'priority': 6,
                        'conditions': {
                            'support_level': support,
                            'limit_type': 'support_bounce'
                        }
                    }
                    actions.append(limit_action)
                    
        elif direction == "SELL" and resistance_levels:
            # Create limit sells near resistance levels
            for resistance in resistance_levels[:3]:  # Max 3 levels
                if resistance > entry:  # Only above current entry
                    limit_action = {
                        'action_type': 'limit_order',
                        'symbol': self.symbol,
                        'direction': direction,
                        'entry_price': resistance - (get_pip_value(self.symbol) * 5),  # 5 pips below resistance
                        'volume': 0.01,
                        'stop_loss': resistance + (get_pip_value(self.symbol) * 20),  # 20 pips above resistance
                        'take_profit': entry - (resistance - entry) * 1.5,  # 1.5x risk-reward
                        'reason': f"SELL limit near resistance {resistance}",
                        'confidence': confidence * 0.8,  # Reduced confidence
                        'risk_level': 'moderate',
                        'order_type': 'limit',
                        'priority': 6,
                        'conditions': {
                            'resistance_level': resistance,
                            'limit_type': 'resistance_rejection'
                        }
                    }
                    actions.append(limit_action)
        
        return actions

    def _generate_risk_management_actions(self, direction: str, entry: float, sl: float, tp: float, 
                                        confidence: float, market_context: Dict) -> List[Dict]:
        """Generate risk management actions like trailing stops"""
        actions = []
        
        # Trailing stop for high confidence trades
        if confidence >= 75.0 and sl:
            trail_distance = get_pip_value(self.symbol) * 30  # 30 pips trail distance
            activation_profit = get_pip_value(self.symbol) * 20  # Activate after 20 pips profit
            
            trailing_action = {
                'action_type': 'trailing_stop',
                'symbol': self.symbol,
                'direction': direction,
                'entry_price': entry,
                'volume': 0.0,  # No volume for stop order
                'stop_loss': sl,
                'take_profit': None,
                'reason': f"Trailing stop with {30} pips distance",
                'confidence': confidence,
                'risk_level': 'low',
                'order_type': 'stop',
                'priority': 8,
                'conditions': {
                    'trail_distance': trail_distance,
                    'activation_profit': activation_profit
                }
            }
            actions.append(trailing_action)
        
        return actions

    def _create_basic_primary_entry(self, direction: str, entry: float, sl: float, tp: float, 
                                  confidence: float, order_type: str) -> Dict:
        """Fallback basic primary entry"""
        # Calculate volume from risk settings, handle None SL
        if sl is None:
            calculated_volume = self.risk_settings.get('fixed_volume_lots', 0.03)
        else:
            calculated_volume = self._calculate_volume_from_risk_settings(entry, sl, "NEW_ENTRY")
        
        return {
            'action_type': 'primary_entry',
            'symbol': self.symbol,
            'direction': direction,
            'entry_price': entry,
            'volume': calculated_volume,
            'stop_loss': sl,
            'take_profit': tp,
            'reason': f"Basic {direction} entry with {confidence:.1f}% confidence",
            'confidence': confidence,
            'risk_level': 'moderate',
            'order_type': order_type or 'market',
            'priority': 1,
            'requires_notification': False,  # Primary entry doesn't need notification
            'conditions': {}
        }

    def _trigger_fibonacci_dca_service(self, existing_positions):
        """
        🎯 Trigger dca_service.py for Fibonacci DCA management
        
        This function notifies the running dca_service.py process that there are positions
        requiring Fibonacci DCA management. The service will:
        1. Calculate Fibonacci levels for existing positions
        2. Place pending orders at appropriate Fibonacci distances
        3. Monitor and execute DCA when price hits levels
        """
        try:
            import subprocess
            import threading
            
            # Get position info for dca_service
            position_symbols = [pos.get('symbol', 'UNKNOWN') for pos in existing_positions]
            logger.info(f"🔔 Notifying dca_service.py about positions: {position_symbols}")
            
            # Create a trigger file that dca_service.py can monitor
            trigger_file = "dca_locks/fibonacci_trigger.json"
            os.makedirs(os.path.dirname(trigger_file), exist_ok=True)
            
            trigger_data = {
                "timestamp": datetime.now().isoformat(),
                "positions_count": len(existing_positions),
                "symbols": position_symbols,
                "trigger_source": "comprehensive_aggregator",
                "dca_mode": "fibonacci"
            }
            
            with open(trigger_file, 'w', encoding='utf-8') as f:
                json.dump(trigger_data, f, indent=2)
                
            logger.info(f"✅ Fibonacci DCA trigger created: {trigger_file}")
            logger.info(f"📐 dca_service.py will process Fibonacci levels for {len(existing_positions)} positions")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error triggering Fibonacci DCA service: {e}")
            return False

    def _generate_single_dca_action(self, signal, entry, sl, tp, confidence, market_context, atr, existing_positions):
        """Generate single DCA action for existing positions"""
        try:
            # 🚨 FIXED: Check for opposite signal confidence before DCA
            if existing_positions:
                first_pos = existing_positions[0]
                position_direction = 'BUY' if first_pos.get('type') == 0 else 'SELL'
                
                # Check for opposite signal with high confidence
                is_opposite_signal = (
                    (position_direction == 'BUY' and signal == 'SELL') or 
                    (position_direction == 'SELL' and signal == 'BUY')
                )
                
                if is_opposite_signal and confidence >= 0.75:  # 75%+ opposite signal
                    logger.warning(f"🚫 DCA BLOCKED: Opposite signal {signal} with {confidence*100:.1f}% confidence vs {position_direction} positions")
                    return None
            
            # Determine next DCA level based on existing DCA positions only
            dca_positions = [p for p in existing_positions if 'DCA' in (p.get('comment', '') or '')]
            current_level = len(dca_positions)  # Count only DCA positions, not all positions
            next_level = current_level + 1
            
            # 🚨 FIXED: Check for duplicate DCA levels by comment pattern
            dca_level_comments = [p.get('comment', '') for p in dca_positions if p.get('comment')]
            next_dca_comment = f"DCA{next_level}"
            if any(next_dca_comment in comment for comment in dca_level_comments):
                logger.warning(f"🚫 DCA BLOCKED: {next_dca_comment} already exists in comments: {dca_level_comments}")
                return None
                
            logger.info(f"🔍 DCA Level calculation: {len(existing_positions)} total positions, {len(dca_positions)} DCA positions, next level: {next_level}")
            
            if next_level > self.dca_settings.get('max_dca_levels', 5):
                logger.info(f"🚫 Maximum DCA levels ({self.dca_settings.get('max_dca_levels', 5)}) reached for {self.symbol}")
                return None
            
            # Get current market price from positions or signal
            current_price = entry  # Default to signal entry
            if existing_positions:
                # Use current price from the first position
                current_price = existing_positions[0].get('price_current', entry)
            
            # Get position direction from existing positions
            position_direction = 'BUY'  # Default
            if existing_positions:
                first_pos = existing_positions[0]
                position_direction = 'BUY' if first_pos.get('type') == 0 else 'SELL'
            
            # Get the LAST DCA level entry for proper sequential spacing
            reference_entry = current_price
            if existing_positions:
                if dca_positions:
                    # Use the most recent (highest level) DCA position as reference
                    dca_entries = [pos.get('price_open', current_price) for pos in dca_positions]
                    if position_direction == 'BUY':
                        reference_entry = min(dca_entries)  # Lowest DCA entry (most recent for BUY)
                    else:
                        reference_entry = max(dca_entries)  # Highest DCA entry (most recent for SELL)
                    logger.info(f"🔍 Using DCA Level {current_level} entry as reference: {reference_entry:.5f}")
                else:
                    # No DCA positions yet, use main entry as reference
                    entries = [pos.get('price_open', current_price) for pos in existing_positions]
                    if position_direction == 'BUY':
                        reference_entry = max(entries)  # Highest entry for BUY positions
                    else:
                        reference_entry = min(entries)  # Lowest entry for SELL positions
                    logger.info(f"🔍 Using main entry as reference: {reference_entry:.5f}")
            
            # Calculate DCA distance requirement based on mode
            pip_value = self._get_pip_value(self.symbol)
            dca_mode = self.risk_settings.get('dca_mode', 'fixed_pips')
            
            # 🔥 FIBONACCI DCA MODE: Delegate to dca_service.py
            if dca_mode in ['fibonacci', 'fibo_levels', 'Fibonacci']:
                logger.info(f"📐 Fibonacci DCA Mode detected - delegating to dca_service.py")
                logger.info(f"🎯 DCA Service will calculate Fibonacci levels and place pending orders")
                
                # Trigger dca_service.py to handle Fibonacci DCA
                self._trigger_fibonacci_dca_service(existing_positions)
                
                # Return None để comprehensive_aggregator không tự tạo DCA
                # dca_service.py sẽ handle tất cả Fibonacci DCA logic
                return None
            
            # 🎯 NON-FIBONACCI MODES: Use comprehensive DCA distance calculation
            dca_distance_pips = self._get_dca_distance_pips()
            dca_distance = dca_distance_pips * pip_value
            logger.info(f"🔍 Comprehensive DCA distance: {dca_distance_pips:.1f} pips (mode: {dca_mode})")
            
            # 🔒 ENHANCED DCA PROTECTION: Check if current price has moved far enough for DCA
            price_movement = 0
            dca_triggered = False
            
            # 🛡️ DCA SPAM PROTECTION: Check if we already have a pending DCA at this level
            pending_dca_actions = []
            actions_file = os.path.join(os.path.dirname(__file__), 'analysis_results', 'account_positions_actions.json')
            if os.path.exists(actions_file):
                try:
                    with open(actions_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        actions = data.get('actions', [])
                        pending_dca_actions = [
                            a for a in actions 
                            if (a.get('action_type') == 'dca_entry' or a.get('action') == 'dca_entry') 
                            and a.get('symbol') == self.symbol
                        ]
                except Exception as e:
                    logger.warning(f"⚠️ Could not check pending DCA actions: {e}")
            
            if pending_dca_actions:
                logger.info(f"🚫 DCA SPAM PROTECTION: {len(pending_dca_actions)} pending DCA actions already exist for {self.symbol}")
                return None
            
            if position_direction == 'BUY':
                price_movement = reference_entry - current_price  # How far down from reference entry
                dca_triggered = price_movement >= dca_distance  # Price moved down enough from last level
                # 🔒 SAFER DCA ENTRY: Place DCA entry at exact distance, not current price
                dca_entry = reference_entry - dca_distance  # Full distance from reference
            else:
                price_movement = current_price - reference_entry  # How far up from reference entry  
                dca_triggered = price_movement >= dca_distance  # Price moved up enough from last level
                # 🔒 SAFER DCA ENTRY: Place DCA entry at exact distance, not current price  
                dca_entry = reference_entry + dca_distance  # Full distance from reference
            
            logger.info(f"🔍 DCA Check for {self.symbol}: Direction={position_direction}, Movement={price_movement/pip_value:.1f} pips, Required={dca_distance_pips} pips, Triggered={dca_triggered}")
            
            if not dca_triggered:
                logger.info(f"⏳ DCA not triggered for {self.symbol} - need {dca_distance_pips - (price_movement/pip_value):.1f} more pips")
                return None
                
            # Calculate DCA volume - Use proper base volume from settings
            base_volume = self.risk_settings.get('fixed_volume_lots', 0.15)  # Use same as entry volume
            dca_multiplier = self.risk_settings.get('dca_volume_multiplier', 1.5)
            dca_volume = base_volume * (dca_multiplier ** next_level)  # Level 1 = base × 1.5^1
            
            # Reduce confidence for higher DCA levels
            dca_confidence = confidence * (0.95 ** next_level)
            
            # 🎯 DCA DYNAMIC S/L T/P: Calculate from indicators like main signals
            try:
                # Use the same dynamic SL/TP calculation as main signals  
                # Function signature: _calculate_sl_tp_from_risk_settings(entry_price, signal, atr, support_levels, resistance_levels, signal_data)
                dca_sl, dca_tp = self._calculate_sl_tp_from_risk_settings(
                    dca_entry, position_direction, atr
                )
                dca_sl_pips = abs(dca_sl - dca_entry) / pip_value
                dca_tp_pips = abs(dca_tp - dca_entry) / pip_value
                sl_reason = f"Risk Settings ({self.risk_settings.get('sltp_mode', 'Unknown')})"
                tp_reason = f"Risk Settings ({self.risk_settings.get('sltp_mode', 'Unknown')})"
                logger.info(f"🎯 DCA DYNAMIC S/L T/P: SL={dca_sl:.5f} ({dca_sl_pips:.1f}pips, {sl_reason}), TP={dca_tp:.5f} ({dca_tp_pips:.1f}pips, {tp_reason})")
            except Exception as e:
                logger.warning(f"⚠️ DCA dynamic SL/TP failed: {e}, using fallback")
                # Fallback to symbol-specific settings
                symbol_settings = self._get_symbol_specific_settings()
                dca_sl_pips = symbol_settings.get("default_sl_pips", symbol_settings.get("default_sl_buffer", 50))
                dca_tp_pips = symbol_settings.get("default_tp_pips", symbol_settings.get("default_tp_buffer", 100))
                
                if position_direction == 'BUY':
                    dca_sl = dca_entry - (dca_sl_pips * pip_value)
                    dca_tp = dca_entry + (dca_tp_pips * pip_value)
                else:  # SELL
                    dca_sl = dca_entry + (dca_sl_pips * pip_value)
                    dca_tp = dca_entry - (dca_tp_pips * pip_value)
            
            dca_mode = self.risk_settings.get('dca_mode', 'Pips cố định')
            
            logger.info(f"✅ DCA Action Generated for {self.symbol}: Level {next_level}, Entry {dca_entry:.5f}, Volume {dca_volume:.3f}")
            logger.info(f"🎯 DCA S/L T/P: SL={dca_sl:.5f} ({dca_sl_pips} pips), TP={dca_tp:.5f} ({dca_tp_pips} pips)")
            
            return {
                'action_type': 'dca_entry',
                'symbol': self.symbol,
                'direction': position_direction,  # Use actual position direction
                'entry_price': dca_entry,
                'volume': round(dca_volume, 3),
                'stop_loss': dca_sl,
                'take_profit': dca_tp,
                'reason': f"DCA Level {next_level} - {position_direction} at {dca_entry:.5f} (Movement: {price_movement/pip_value:.1f} pips)",
                'confidence': round(dca_confidence, 1),
                'risk_level': 'moderate',
                'order_type': 'limit',
                'priority': next_level + 1,
                'conditions': {
                    'dca_level': next_level,
                    'existing_positions': len(existing_positions),
                    'dca_mode': dca_mode,
                    'price_movement_pips': round(price_movement/pip_value, 1),
                    'dca_distance_required': dca_distance_pips
                }
            }
        except Exception as e:
            logger.error(f"Error generating DCA action: {e}")
            return None

    def _generate_position_management_action(self, existing_positions, signal, confidence):
        """Generate position management actions (SL/TP setup or trailing stop) for ALL positions"""
        try:
            # Debug position data
            logger.debug(f"🔍 Position data: {existing_positions}")
            logger.debug(f"🔍 Position types: {[type(p) for p in existing_positions]}")
            logger.debug(f"🔍 self.symbol: {self.symbol} (type: {type(self.symbol)})")
            
            # Find existing positions for this symbol
            symbol_positions = [p for p in existing_positions if p.get('symbol') == self.symbol]
            logger.debug(f"🔍 symbol_positions: {symbol_positions}")
            if not symbol_positions:
                return []
            
            actions = []
            
            # Process each position individually
            for position in symbol_positions:
                logger.debug(f"🔍 Processing position: {position} (type: {type(position)})")
                
                profit_pips = float(position.get('profit_pips', 0))
                logger.debug(f"🔍 profit_pips: {profit_pips}")
                
                current_sl = float(position.get('sl', 0))
                logger.debug(f"🔍 current_sl: {current_sl}")
                
                current_tp = float(position.get('tp', 0))
                logger.debug(f"🔍 current_tp: {current_tp}")
                
                pos_type = position.get('type_str', 'BUY')
                logger.debug(f"🔍 pos_type: {pos_type}")
                
                logger.debug(f"🔍 signal object: {signal} (type: {type(signal)})")
                
                # Handle signal as string or dict
                if isinstance(signal, dict):
                    current_price = signal.get('close', float(position.get('price_current', 0)))
                else:
                    # Signal is string like "SELL", get current price from position
                    current_price = float(position.get('price_current', 0))
                
                logger.debug(f"🔍 current_price: {current_price}")
                
                # Priority 1: Add missing SL/TP for positions without them
                if current_sl == 0 or current_tp == 0:
                    logger.info(f"🔧 Position {self.symbol} ticket {position.get('ticket')} missing SL/TP - generating setup action")
                    
                    # Calculate SL/TP using risk management
                    entry_price = float(position.get('price_open', current_price))
                    
                    if pos_type == 'BUY':
                        if current_sl == 0:
                            # Add SL below current price
                            sl_distance = self.risk_settings.get('default_sl_pips', 50)
                            new_sl = entry_price - (sl_distance * self._get_pip_size())
                        else:
                            new_sl = current_sl
                            
                        if current_tp == 0:
                            # Add TP above current price
                            tp_distance = self.risk_settings.get('default_tp_pips', 100)
                            new_tp = entry_price + (tp_distance * self._get_pip_size())
                        else:
                            new_tp = current_tp
                    else:  # SELL
                        if current_sl == 0:
                            # Add SL above current price
                            sl_distance = self.risk_settings.get('default_sl_pips', 50)
                            new_sl = entry_price + (sl_distance * self._get_pip_size())
                        else:
                            new_sl = current_sl
                            
                        if current_tp == 0:
                            # Add TP below current price
                            tp_distance = self.risk_settings.get('default_tp_pips', 100)
                            new_tp = entry_price - (tp_distance * self._get_pip_size())
                        else:
                            new_tp = current_tp
                    
                    actions.append({
                        'action_type': 'modify_position',
                        'symbol': self.symbol,
                        'direction': pos_type,
                        'volume': 0,  # Modification, not new volume
                        'entry_price': entry_price,
                        'stop_loss': new_sl,
                        'take_profit': new_tp,
                        'reason': f"Add missing SL/TP to position #{position.get('ticket')} (SL: {current_sl}, TP: {current_tp})",
                        'confidence': confidence,
                        'risk_level': 'medium',
                        'order_type': 'modification',
                        'priority': 1,  # High priority
                        'requires_notification': True,  # Position setup needs notification
                        'conditions': {
                            'position_ticket': position.get('ticket'),
                            'missing_sl': current_sl == 0,
                            'missing_tp': current_tp == 0
                        }
                    })
                
                # Priority 1.6: 🎯 SIGNAL-BASED T/P ADJUSTMENT (CHO TẤT CẢ LỆNH)
                elif (current_tp > 0 and 
                      self.risk_settings.get('enable_signal_based_adjustment', True)):
                    # Get current signal from the analysis
                    current_signal = signal if isinstance(signal, str) else signal.get('signal', 'NEUTRAL')
                    
                    if current_signal in ('BUY', 'SELL'):
                        # Get new T/P from current analysis for all positions
                        entry_price = float(position.get('price_open', current_price))
                        
                        # Calculate suggested new T/P based on current signal
                        pip_size = self._get_pip_size()
                        if current_signal == 'BUY':
                            # For BUY: T/P above entry
                            suggested_tp_pips = self.risk_settings.get('signal_adjustment_tp_pips', 120)
                            new_tp_suggestion = entry_price + (suggested_tp_pips * pip_size)
                        else:  # SELL
                            # For SELL: T/P below entry
                            suggested_tp_pips = self.risk_settings.get('signal_adjustment_tp_pips', 120)
                            new_tp_suggestion = entry_price - (suggested_tp_pips * pip_size)
                        
                        # Check if T/P adjustment is recommended
                        adjustment = self.check_signal_based_adjustment(
                            existing_positions=[position],
                            new_signal=current_signal,
                            new_sl=current_sl,  # Keep current S/L for T/P-only adjustment
                            new_tp=new_tp_suggestion,
                            current_price=current_price
                        )
                        
                        if adjustment.get('adjust_tp'):
                            logger.info(f"🎯 Signal-based T/P adjustment for {self.symbol} (profit: {profit_pips:.1f} pips): {adjustment.get('reason')}")
                            
                            actions.append({
                                'action': 'modify_position',
                                'type': 'tp_adjustment_signal_based',  # Different type for T/P adjustment
                                'symbol': self.symbol,
                                'ticket': position.get('ticket'),
                                'direction': pos_type,
                                'position_type': pos_type,
                                'volume': 0,
                                'entry_price': entry_price,
                                'current_sl': current_sl,
                                'current_tp': current_tp,
                                'new_sl': current_sl,  # Keep current S/L unchanged
                                'new_tp': adjustment.get('new_tp', current_tp),
                                'reason': f"T/P adjustment: {adjustment.get('reason')}",
                                'confidence': confidence * 100,
                                'risk_level': 'low',
                                'order_type': 'modification',
                                'priority': 3,  # Lower priority than S/L adjustment
                                'requires_notification': True,
                                'conditions': {
                                    'position_ticket': position.get('ticket'),
                                    'adjustment_type': 'tp_all_positions',
                                    'old_sl': current_sl,
                                    'old_tp': current_tp,
                                    'signal': current_signal,
                                    'profit_pips': profit_pips
                                }
                            })
            
                # Priority 3: Trailing stop for profitable positions with SL/TP  
                elif profit_pips > 0:
                    trail_distance = self.risk_settings.get('trailing_stop_distance_pips', 20)
                    activation_profit = self.risk_settings.get('trailing_activation_profit_pips', 15)
                    
                    if profit_pips >= activation_profit:
                        logger.info(f"📈 Position {self.symbol} ticket {position.get('ticket')} profitable ({profit_pips:.1f} pips) - generating trailing stop")
                        
                        actions.append({
                            'action_type': 'trailing_stop',
                            'symbol': self.symbol,
                            'direction': pos_type,
                            'volume': 0,  # Modification, not new volume
                            'entry_price': 0,  # Not applicable
                            'stop_loss': 0,   # Will be calculated dynamically
                            'take_profit': 0, # Keep existing TP
                            'reason': f"Trailing stop for position #{position.get('ticket')} (profit: {profit_pips:.1f} pips)",
                            'confidence': confidence,
                            'risk_level': 'low',
                            'order_type': 'modification',
                            'priority': 2,
                            'requires_notification': True,  # Trailing stop needs notification
                            'conditions': {
                                'trail_distance': trail_distance,
                                'activation_profit': activation_profit,
                                'position_ticket': position.get('ticket')
                            }
                        })
                else:
                    # No management action needed for this position
                    logger.debug(f"No position management needed for {self.symbol} ticket {position.get('ticket')} (profit: {profit_pips:.1f}, SL: {current_sl}, TP: {current_tp})")
            
            # Return all actions generated
            if actions:
                logger.info(f"🎯 Generated {len(actions)} position management actions for {self.symbol}")
                return actions
            else:
                logger.debug(f"No position management actions needed for {self.symbol}")
                return []
            
        except Exception as e:
            logger.error(f"Error generating management actions: {e}")
            return []

    def _generate_signal_based_sl_adjustment_actions(self, existing_positions: List[Dict], 
                                                   new_signal: str, new_entry: float, 
                                                   new_sl: float, new_tp: float, 
                                                   confidence: float) -> List[Dict]:
        """
        🎯 Dịch chuyển S/L theo signal mới cho các lệnh hiện tại
        
        Logic:
        - Chỉ dịch chuyển S/L cho các lệnh CHƯA CÓ LÃI (chưa đến BE)
        - Bỏ qua các lệnh đã có lãi (đã dịch chuyển S/L về BE hoặc có trailing stop)
        - Áp dụng cho cả Entry và DCA orders
        - Sử dụng S/L từ signal mới làm reference
        
        Args:
            existing_positions: Danh sách positions hiện tại
            new_signal: Signal mới (BUY/SELL)
            new_entry: Entry price từ signal mới
            new_sl: S/L từ signal mới
            new_tp: T/P từ signal mới  
            confidence: Độ tin cậy signal mới
            
        Returns:
            List[Dict]: Danh sách actions để update S/L
        """
        try:
            actions = []
            
            if not existing_positions or confidence < 0.6:
                logger.debug(f"🎯 Signal S/L adjustment skipped: positions={len(existing_positions or [])}, confidence={confidence}")
                return actions
            
            # Get current price for profit calculation
            current_price = self.current_price or new_entry
            
            logger.info(f"🎯 Signal S/L Adjustment: {new_signal} signal, new_sl={new_sl:.5f}, positions={len(existing_positions)}")
            
            # Process each existing position
            for pos in existing_positions:
                try:
                    # MT5 type: 0=BUY, 1=SELL
                    pos_type_raw = pos.get('type', -1)
                    if pos_type_raw == 0:
                        pos_type = 'BUY'
                    elif pos_type_raw == 1:
                        pos_type = 'SELL'
                    else:
                        # Handle string types as fallback
                        pos_type = str(pos.get('type', '')).upper()
                        if pos_type not in ['BUY', 'SELL']:
                            logger.debug(f"Unknown position type: {pos.get('type')} for ticket {pos.get('ticket')}")
                            continue
                    
                    pos_entry = float(pos.get('price_open', 0))
                    current_sl = pos.get('sl', 0)
                    current_tp = pos.get('tp', 0)
                    ticket = pos.get('ticket', 0)
                    volume = pos.get('volume', 0)
                    
                    if not pos_entry or not ticket:
                        continue
                        
                    # Calculate current profit/loss
                    if pos_type == 'BUY':
                        profit_pips = (current_price - pos_entry) / self._get_pip_value(self.symbol)
                        is_profitable = current_price > pos_entry
                    else:  # SELL
                        profit_pips = (pos_entry - current_price) / self._get_pip_value(self.symbol)
                        is_profitable = current_price < pos_entry
                    
                    # 🎯 CONDITION CHECK: Chỉ adjust S/L cho positions CHƯA CÓ LÃI
                    logger.info(f"📊 DEBUG Position {ticket}: type={pos_type}, profit_pips={profit_pips:.1f}, is_profitable={is_profitable}")
                    if is_profitable:
                        logger.info(f"🟢 Position {ticket} is profitable (+{profit_pips:.1f} pips) - Skip S/L adjustment (already in BE/trailing zone)")
                        continue
                    
                    # 🎯 SIGNAL COMPATIBILITY CHECK: Chỉ adjust nếu signal tương thích
                    signal_compatible = False
                    
                    if pos_type == 'BUY' and new_signal == 'BUY':
                        # BUY position with BUY signal - strengthen support
                        signal_compatible = True
                    elif pos_type == 'SELL' and new_signal == 'SELL':
                        # SELL position with SELL signal - strengthen resistance
                        signal_compatible = True
                    elif pos_type == 'BUY' and new_signal == 'SELL':
                        # BUY position with SELL signal - tighten S/L for protection
                        signal_compatible = True
                    elif pos_type == 'SELL' and new_signal == 'BUY':
                        # SELL position with BUY signal - tighten S/L for protection
                        signal_compatible = True
                    
                    logger.info(f"📊 DEBUG Signal compatibility: {new_signal} signal vs {pos_type} position = {signal_compatible}")
                    if not signal_compatible:
                        logger.info(f"❌ Signal {new_signal} not compatible with {pos_type} position {ticket}")
                        continue
                    
                    # 🎯 CALCULATE NEW S/L based on signal direction and position type
                    new_position_sl = None
                    sl_reason = ""
                    
                    if pos_type == 'BUY':
                        if new_signal == 'BUY':
                            # BUY + BUY: Use new S/L if it's better (higher) than current
                            if not current_sl or new_sl > current_sl:
                                new_position_sl = new_sl
                                sl_reason = f"Strengthen BUY support: {new_sl:.5f} > {current_sl:.5f}"
                        else:  # new_signal == 'SELL'
                            # BUY + SELL: Tighten S/L for protection (use new_sl but not below current)
                            protective_sl = max(new_sl, current_sl) if current_sl else new_sl
                            new_position_sl = protective_sl
                            sl_reason = f"Protective S/L for BUY (SELL signal): {protective_sl:.5f}"
                    
                    else:  # pos_type == 'SELL'
                        if new_signal == 'SELL':
                            # SELL + SELL: Use new S/L if it's better (lower) than current
                            if not current_sl or new_sl < current_sl:
                                new_position_sl = new_sl
                                sl_reason = f"Strengthen SELL resistance: {new_sl:.5f} < {current_sl:.5f}"
                        else:  # new_signal == 'BUY'
                            # SELL + BUY: Tighten S/L for protection (use new_sl but not above current)
                            protective_sl = min(new_sl, current_sl) if current_sl else new_sl
                            new_position_sl = protective_sl
                            sl_reason = f"Protective S/L for SELL (BUY signal): {protective_sl:.5f}"
                    
                    # 🎯 VALIDATION: Only proceed if new S/L is significantly different
                    if new_position_sl and current_sl:
                        sl_diff_pips = abs(new_position_sl - current_sl) / self._get_pip_value(self.symbol)
                        if sl_diff_pips < 5.0:  # Minimum 5 pips difference
                            logger.debug(f"S/L difference too small ({sl_diff_pips:.1f} pips) for position {ticket}")
                            continue
                    
                    if new_position_sl:
                        # Create S/L adjustment action
                        action = {
                            'action': 'modify_position',
                            'type': 'sl_adjustment_signal_based',
                            'symbol': self.symbol,
                            'ticket': ticket,
                            'current_sl': current_sl,
                            'new_sl': new_position_sl,
                            'new_tp': current_tp,  # Keep existing TP
                            'reason': sl_reason,
                            'signal_trigger': new_signal,
                            'confidence': confidence,
                            'position_type': pos_type,
                            'position_entry': pos_entry,
                            'current_profit_pips': profit_pips,
                            'priority': 'medium',  # Not urgent as high priority actions
                            'timestamp': datetime.now().isoformat(),
                            'risk_level': 'low'  # S/L adjustment is risk reduction
                        }
                        
                        actions.append(action)
                        logger.info(f"🎯 Signal S/L Adjustment: {pos_type} position {ticket} - {sl_reason}")
                        
                except Exception as e:
                    logger.error(f"Error processing position {pos.get('ticket', 'unknown')}: {e}")
                    continue
            
            # Summary
            if actions:
                logger.info(f"🎯 Generated {len(actions)} signal-based S/L adjustment actions for {self.symbol}")
            else:
                logger.debug(f"🎯 No signal-based S/L adjustments needed for {self.symbol}")
                
            return actions
            
        except Exception as e:
            logger.error(f"Error generating signal-based S/L adjustments: {e}")
            return []

    def _generate_opposite_signal_close_actions(self, existing_positions: List[Dict], 
                                              opposite_signal: str, signal_confidence: float) -> List[Dict]:
        """
        🚨 Generate actions to close positions when opposite signal detected
        
        Logic:
        - Close losing positions immediately when strong opposite signal
        - Protect winning positions by moving S/L to breakeven or partial profit
        - Consider position age and current P&L
        
        Args:
            existing_positions: Current positions for symbol
            opposite_signal: The conflicting signal (BUY vs SELL positions)
            signal_confidence: Confidence of opposite signal
            
        Returns:
            List[Dict]: Actions to close or protect positions
        """
        try:
            actions = []
            current_price = self.current_price
            
            if not existing_positions or not current_price:
                return actions
            
            logger.info(f"🚨 Opposite Signal Analysis: {opposite_signal} signal vs {len(existing_positions)} positions (confidence: {signal_confidence:.1f})")
            
            for pos in existing_positions:
                try:
                    # Get position details
                    pos_type_raw = pos.get('type', -1)
                    pos_type = 'BUY' if pos_type_raw == 0 else 'SELL' if pos_type_raw == 1 else 'UNKNOWN'
                    
                    if pos_type == 'UNKNOWN':
                        continue
                        
                    pos_entry = float(pos.get('price_open', 0))
                    current_sl = pos.get('sl', 0)
                    current_tp = pos.get('tp', 0)
                    ticket = pos.get('ticket', 0)
                    
                    if not pos_entry or not ticket:
                        continue
                    
                    # Calculate current P&L
                    if pos_type == 'BUY':
                        profit_pips = (current_price - pos_entry) / self._get_pip_value(self.symbol)
                    else:  # SELL
                        profit_pips = (pos_entry - current_price) / self._get_pip_value(self.symbol)
                    
                    is_profitable = profit_pips > 0
                    is_losing_badly = profit_pips < -20  # More than 20 pips loss
                    
                    logger.info(f"📊 Position {ticket}: {pos_type}, P&L: {profit_pips:.1f} pips, profitable: {is_profitable}")
                    
                    # Decision logic based on P&L and signal strength
                    action_type = None
                    reason = ""
                    
                    if is_losing_badly and signal_confidence >= 0.7:
                        # Close losing positions when strong opposite signal
                        action_type = 'close_position'
                        reason = f"Close losing position ({profit_pips:.1f} pips) - strong opposite {opposite_signal} signal"
                        
                    elif not is_profitable and signal_confidence >= 0.8:
                        # Close breakeven/small loss positions when very strong opposite signal  
                        action_type = 'close_position'
                        reason = f"Close position at {profit_pips:.1f} pips - very strong opposite {opposite_signal} signal"
                        
                    elif is_profitable and profit_pips > 10:
                        # Protect profitable positions by moving S/L to breakeven or partial profit
                        action_type = 'modify_position'
                        protective_sl = pos_entry + (0.5 * (current_price - pos_entry)) if pos_type == 'BUY' else pos_entry - (0.5 * (pos_entry - current_price))
                        reason = f"Protect profit ({profit_pips:.1f} pips) - opposite {opposite_signal} signal detected"
                        
                    if action_type == 'close_position':
                        action = {
                            'action': 'close_position',
                            'type': 'opposite_signal_close',
                            'symbol': self.symbol,
                            'ticket': ticket,
                            'reason': reason,
                            'signal_trigger': opposite_signal,
                            'confidence': signal_confidence,
                            'position_type': pos_type,
                            'position_entry': pos_entry,
                            'current_profit_pips': profit_pips,
                            'priority': 'high',  # High priority for opposite signals
                            'timestamp': datetime.now().isoformat(),
                            'risk_level': 'medium'
                        }
                        actions.append(action)
                        
                    elif action_type == 'modify_position':
                        action = {
                            'action': 'modify_position', 
                            'type': 'opposite_signal_protection',
                            'symbol': self.symbol,
                            'ticket': ticket,
                            'current_sl': current_sl,
                            'new_sl': protective_sl,
                            'new_tp': current_tp,
                            'reason': reason,
                            'signal_trigger': opposite_signal,
                            'confidence': signal_confidence,
                            'position_type': pos_type,
                            'position_entry': pos_entry,
                            'current_profit_pips': profit_pips,
                            'priority': 'high',
                            'timestamp': datetime.now().isoformat(),
                            'risk_level': 'low'
                        }
                        actions.append(action)
                        
                    if action_type:
                        logger.info(f"🚨 Opposite Signal Action: {action_type} for {pos_type} position {ticket} - {reason}")
                        
                except Exception as e:
                    logger.error(f"Error processing position {pos.get('ticket', 'unknown')} for opposite signal: {e}")
                    continue
            
            return actions
            
        except Exception as e:
            logger.error(f"Error generating opposite signal actions: {e}")
            return []
    
    def _get_pip_size(self):
        """Get pip size for current symbol - MUST MATCH get_pip_value() function"""
        # Use the standardized get_pip_value function for consistency
        return get_pip_value(self.symbol)
    
    def _create_dca_primary_entry(self, signal, entry, sl, tp, confidence, order_type, entry_reason, market_context):
        """Create primary entry optimized for DCA strategy"""
        try:
            # For DCA-enabled strategy, use proper volume calculation
            # Calculate volume from risk settings, handle None SL
            if sl is None:
                # Use default volume if no SL available for risk calculation
                base_volume = self.risk_settings.get('fixed_volume_lots', 0.03)
            else:
                base_volume = self._calculate_volume_from_risk_settings(entry, sl, "NEW_ENTRY")
            
            # Optimize entry type for DCA start
            if confidence >= 90:
                final_order_type = 'market'  # High confidence = immediate entry
                reason_suffix = f" with DCA scaling planned"
            else:
                final_order_type = 'limit'   # Lower confidence = wait for better price
                reason_suffix = f" with DCA backup plan"
            
            # Handle signal parameter (could be string or dict)
            if isinstance(signal, str):
                direction = signal
            else:
                direction = signal.get('signal', 'BUY')
            
            return {
                'action_type': 'primary_entry',
                'symbol': self.symbol,
                'direction': direction,
                'volume': base_volume,
                'entry_price': entry,
                'stop_loss': sl,
                'take_profit': tp,
                'reason': f"{entry_reason}{reason_suffix}",
                'confidence': confidence,
                'risk_level': 'low',  # Conservative start for DCA
                'order_type': final_order_type,
                'requires_notification': False,  # Primary entry doesn't need notification
                'priority': 1,
                'conditions': {
                    'dca_enabled': True,
                    'max_dca_levels': self.risk_settings.get('max_dca_levels', 3),
                    'dca_mode': self.risk_settings.get('dca_mode', 'Mức Fibo')
                }
            }
        except Exception as e:
            logger.error(f"Error creating DCA primary entry: {e}")
            # 🔒 CRITICAL: Check for existing positions before fallback
            existing_positions = self._get_existing_positions_for_symbol(self.symbol)
            if existing_positions and len(existing_positions) > 0:
                logger.warning(f"🚫 BLOCKING DCA FALLBACK ENTRY: {self.symbol} already has {len(existing_positions)} position(s)")
                return None  # No fallback if positions exist
            else:
                logger.info(f"✅ DCA fallback entry for {self.symbol} - no existing positions found")
                return self._create_basic_primary_entry(
                    signal, entry, sl, tp, confidence, order_type
                )

    def _deduplicate_actions(self, actions: List[Dict]) -> List[Dict]:
        """
        🚨 ENHANCED: Deduplicate actions to prevent duplicate orders
        
        Removes duplicate actions based on:
        1. Same symbol + direction + action type
        2. Similar entry price (within 1 pip)
        3. Same volume and order type
        """
        try:
            if not actions:
                return actions
                
            seen_signatures = set()
            deduped_actions = []
            
            for action in actions:
                # Don't deduplicate S/L adjustments for LOSING positions only - they need individual ticket handling
                action_type_check = action.get('type', action.get('action_type', ''))
                current_profit_pips = action.get('current_profit_pips', 0)
                
                # Only bypass deduplication for S/L adjustments on losing positions (negative profit)
                if ('sl_adjustment' in action_type_check or 'tp_adjustment' in action_type_check) and current_profit_pips < 0:
                    deduped_actions.append(action)
                    continue
                
                # Extract key components for signature (only for regular actions)
                symbol = action.get('symbol', '').upper()
                direction = action.get('direction', '').upper()
                action_type = action.get('action_type', '')
                entry_price = action.get('entry_price', 0.0)
                volume = action.get('volume', 0.0)
                order_type = action.get('order_type', '')
                
                # Round price to nearest pip for comparison (prevents micro-differences)
                price_rounded = round(entry_price, 5)
                
                # Create unique signature
                signature = f"{symbol}_{direction}_{action_type}_{price_rounded:.5f}_{volume}_{order_type}"
                
                if signature in seen_signatures:
                    logger.warning(f"🚫 DUPLICATE ACTION REMOVED: {signature}")
                    continue
                
                seen_signatures.add(signature)
                deduped_actions.append(action)
                
            removed_count = len(actions) - len(deduped_actions)
            if removed_count > 0:
                logger.info(f"🚨 Removed {removed_count} duplicate actions from {len(actions)} total")
                
            return deduped_actions
            
        except Exception as e:
            logger.error(f"❌ Error deduplicating actions: {e}")
            return actions  # Return original if deduplication fails

    # (Duplicate legacy block removed)
# ------------------------------
# CLI
# ------------------------------
import argparse

def main(argv: Optional[List[str]] = None) -> int:
    import os  # Ensure os is available in main scope
    parser = argparse.ArgumentParser(description="Smart multi-timeframe aggregator")
    parser.add_argument("--symbols", nargs='*', default=None, help="Symbols to analyze (auto-detect if omitted)")
    parser.add_argument("--limit", type=int, default=0, help="Max symbols to analyze (0 = no limit)")
    parser.add_argument(
        "--all", action="store_true", help="Shortcut: analyze all detected symbols (equivalent to --limit 0)"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--indicators",
        type=str,
        default=None,
        help=(
            "Whitelist of indicators to use (comma or space separated). "
            "Supported names: rsi, macd, ema50, adx, stochrsi, atr, donchian"
        ),
    )
    parser.add_argument(
        "--strict-indicators",
        action="store_true",
        help="Use only indicators provided in indicator_output; disable all computed fallbacks from candles",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default=None,
        help="Comma-separated list of timeframes to use (e.g., H4,H1,M30,M15,M5). If not specified, auto-detect from available data files.",
    )
    # Decision-tuning parameters
    parser.add_argument("--conf-strong", type=float, default=65.0, help="Confidence threshold for strong signal")
    parser.add_argument("--conf-medium", type=float, default=55.0, help="Confidence threshold for medium signal")
    parser.add_argument("--profit-partial30", type=float, default=3.0, help="Equity %% gain to take 30%% profit")
    parser.add_argument("--profit-partial50", type=float, default=5.0, help="Equity %% gain to take 50%% profit")
    parser.add_argument("--loss-neutral-close", type=float, default=-7.0, help="Equity %% loss to close when neutral")
    parser.add_argument("--loss-aligned-close-weak", type=float, default=-6.0, help="Loss to close when aligned and signal weak")
    parser.add_argument("--loss-aligned-close-med", type=float, default=-8.0, help="Loss to close when aligned and signal medium")
    parser.add_argument("--loss-aligned-close-strong", type=float, default=-10.0, help="Loss to close when aligned and signal strong")
    parser.add_argument("--loss-opposite-close", type=float, default=-6.0, help="Loss to close when opposite and not high confidence")
    parser.add_argument("--language", type=str, choices=["en","vi"], default="vi", help="Language of generated textual reports (vi/en)")
    parser.add_argument(
        "--ma-family",
        dest="ma_family",
        choices=["expand", "exact"],
        default="expand",
        help="MA handling: 'expand' shows EMA/SMA/WMA/TEMA for any selected MA period; 'exact' shows only variants explicitly selected"
    )
    parser.add_argument("--english-only", action="store_true", help="Only generate English text reports (skip Vietnamese textual files)")
    
    # 🛡️ RISK MANAGEMENT ARGUMENTS
    parser.add_argument("--max-risk-per-trade", type=float, default=5.0, help="Maximum risk per trade percentage (default: 5.0%%)")
    parser.add_argument("--max-total-risk", type=float, default=20.0, help="Maximum total portfolio risk percentage (default: 20.0%%)")
    parser.add_argument("--max-drawdown", type=float, default=15.0, help="Maximum drawdown before emergency close all (default: 15.0%%)")
    parser.add_argument("--min-margin-level", type=float, default=300.0, help="Minimum margin level before risk actions (default: 300%%)")
    parser.add_argument("--risk-mode", choices=["conservative", "moderate", "aggressive"], default="moderate", help="Risk management mode")
    parser.add_argument("--auto-reduce-size", action="store_true", help="Automatically reduce position sizes when risk limits are approached")
    parser.add_argument("--emergency-close", action="store_true", help="Enable emergency close all positions when max drawdown hit")
    parser.add_argument("--risk-scaling", action="store_true", help="Enable risk-based position scaling (larger positions = stricter rules)")
    parser.add_argument("--correlation-risk", action="store_true", help="Consider symbol correlation for risk calculations")
    parser.add_argument("--risk-trail-stops", action="store_true", help="Use tighter trailing stops for high-risk positions")
    
    # 📊 HISTORY & LOGGING OPTIONS
    parser.add_argument("--auto-history", action="store_true", help="Enable auto-save trading history (may slow down auto-trading)")
    parser.add_argument("--history", action="store_true", help="Same as --auto-history")
    parser.add_argument("--no-history", action="store_true", help="Disable auto-save trading history completely")
    args = parser.parse_args(argv)
    
    # 🛡️ Load risk settings from JSON and override args defaults
    try:
        risk_settings_path = os.path.join(os.path.dirname(__file__), 'risk_management', 'risk_settings.json')
        if os.path.exists(risk_settings_path):
            with open(risk_settings_path, 'r', encoding='utf-8') as f:
                risk_settings = json.load(f)
            
            # Override args with risk_settings values if they are not null or "OFF"
            max_risk_val = risk_settings.get('max_risk_percent')
            if max_risk_val is not None and max_risk_val != "OFF":
                args.max_risk_per_trade = max_risk_val
            else:
                args.max_risk_per_trade = None  # Mark as disabled
                
            max_total_risk_val = risk_settings.get('max_total_risk')
            if max_total_risk_val is not None and max_total_risk_val != "OFF":
                args.max_total_risk = max_total_risk_val
            else:
                args.max_total_risk = None  # Mark as disabled
                
            max_drawdown_val = risk_settings.get('max_drawdown_percent')
            if max_drawdown_val is not None and max_drawdown_val != "OFF":
                args.max_drawdown = max_drawdown_val
            else:
                args.max_drawdown = None  # Mark as disabled
    except Exception as e:
        logger.warning(f"⚠️ Could not load risk settings: {e}")
    
    # 🛡️ Display risk management settings if verbose
    if args.verbose:
        print("\n🛡️ RISK MANAGEMENT SETTINGS:")
        print("="*50)
        print(f"📊 Risk Mode: {args.risk_mode}")
        print(f"⚖️ Max Risk Per Trade: {'❌ Disabled' if args.max_risk_per_trade is None else f'{args.max_risk_per_trade}%'}")
        print(f"📈 Max Total Risk: {'❌ Disabled' if args.max_total_risk is None else f'{args.max_total_risk}%'}")
        print(f"📉 Max Drawdown: {'❌ Disabled' if args.max_drawdown is None else f'{args.max_drawdown}%'}")
        print(f"💰 Min Margin Level: {args.min_margin_level}%")
        print(f"🔧 Auto Reduce Size: {'✅ Enabled' if args.auto_reduce_size else '❌ Disabled'}")
        print(f"🚨 Emergency Close: {'✅ Enabled' if args.emergency_close else '❌ Disabled'}")
        print(f"📏 Risk Scaling: {'✅ Enabled' if args.risk_scaling else '❌ Disabled'}")
        print(f"🔗 Correlation Risk: {'✅ Enabled' if args.correlation_risk else '❌ Disabled'}")
        print(f"🎯 Risk Trail Stops: {'✅ Enabled' if args.risk_trail_stops else '❌ Disabled'}")
        print("="*50)
    
    # Expose language choices to downstream generation blocks without refactoring large code sections
    globals()['_RUNTIME_LANGUAGE'] = getattr(args, 'language', 'vi')
    globals()['_RUNTIME_EN_ONLY'] = getattr(args, 'english_only', False)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    indicator_whitelist: Optional[set[str]] = None
    gui_original_order: Optional[list[str]] = None  # For preserving GUI order

    def _expand_ma_variants(tokens: set[str], do_expand: bool = True) -> set[str]:
        """Ensure that for any MA token like 'sma20' all 4 variants for the same period
        are included (ema/sma/wma/tema). This guarantees reports show the full MA family
        when the user selects any one variant in the GUI/CLI.
        """
        try:
            if not do_expand:
                # Return as-is for strict mapping
                return set(tokens)
            out = set(tokens)
            # Collect periods seen for any ma type
            periods: set[str] = set()
            for t in list(tokens):
                m = re.fullmatch(r"(ema|sma|wma|tema)(\d{1,3})", t)
                if m:
                    periods.add(m.group(2))
            for p in periods:
                out.update({f"ema{p}", f"sma{p}", f"wma{p}", f"tema{p}"})
            return out
        except Exception:
            return tokens
    if args.indicators:
        parts = re.split(r"[\s,;]+", args.indicators.strip())
        wl = [p.strip().lower() for p in parts if p.strip()]
        if wl:
            gui_original_order = wl.copy()  # Preserve original CLI order
            indicator_whitelist = _expand_ma_variants(set(wl), do_expand=(args.ma_family == "expand"))
            logger.info("Output indicator whitelist (raw): %s", ", ".join(sorted(set(wl))))
            if args.ma_family == "expand":
                logger.info("Output indicator whitelist (MA-expanded): %s", ", ".join(sorted(indicator_whitelist)))
            else:
                logger.info("Output indicator whitelist (MA-exact): %s", ", ".join(sorted(indicator_whitelist)))
    else:
        # No CLI whitelist provided: attempt to load GUI-persisted whitelist file
        try:
            import json
            gui_wl_path = os.path.join('analysis_results', 'indicator_whitelist.json')
            if os.path.exists(gui_wl_path):
                with open(gui_wl_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    # Normalization & support mapping
                    norm_map = {
                        'rsi':'rsi','rsi14':'rsi',
                        'macd':'macd',
                        'adx':'adx','adx14':'adx',
                        'stochrsi':'stochrsi','stochastic rsi':'stochrsi','stochastic_rsi':'stochrsi',
                        'stochastic':'stochastic',
                        'atr':'atr','atr14':'atr',
                        'donchian':'donchian','donchian20':'donchian',
                        'ema20':'ema20','ema_20':'ema20','ema50':'ema50','ema_50':'ema50','ema100':'ema100','ema200':'ema200',
                        'sma20':'sma20','wma20':'wma20',
                        'bollinger':'bollinger','bollinger bands':'bollinger',
                        'keltner':'keltner','keltner channel':'keltner',
                        'ichimoku':'ichimoku','ichimoku cloud':'ichimoku',
                        'cci':'cci','commodity channel index':'cci',
                        'williamsr':'williamsr','williams %r':'williamsr','williams % r':'williamsr','williams%r':'williamsr',
                        'roc':'roc','rate of change':'roc',
                        'obv':'obv','on balance volume':'obv',
                        'chaikin':'chaikin','chaikin osc':'chaikin',
                        'eom':'eom','ease of movement':'eom',
                        'mfi':'mfi','money flow index':'mfi','mfi14':'mfi',
                        'force':'force','force index':'force',
                        'trix':'trix','dpo':'dpo','mass':'mass','mass index':'mass',
                        'vortex':'vortex','kst':'kst','ultimate':'ultimate','ultimate osc':'ultimate',
                        'envelopes':'envelopes','momentum':'momentum','momentum/cycle':'momentum',
                        'fibonacci':'fibonacci',
                        'psar':'psar',
                        'pattern':'patterns','patterns':'patterns','price patterns':'patterns','price pattern':'patterns'
                    }
                    cleaned = []  # Use list to preserve order
                    import re as _re
                    for item in raw:
                        if not isinstance(item, str):
                            continue
                        key = item.strip().lower()
                        key = norm_map.get(key, key)
                        # Keep supported tokens. Accept generic MA patterns: ema|sma|wma|tema + digits
                        if key in {
                            'rsi','macd','adx','stochrsi','stochastic','atr','donchian',
                            'bollinger','keltner','ichimoku','cci','williamsr','roc','obv','chaikin','eom','mfi','force','trix','dpo','mass','vortex','kst','ultimate','envelopes','momentum','psar','fibonacci'
                        }:
                            if key not in cleaned:  # Avoid duplicates
                                cleaned.append(key)
                            continue
                        if _re.fullmatch(r'(ema|sma|wma|tema)\d{1,3}', key or ''):
                            if key not in cleaned:  # Avoid duplicates
                                cleaned.append(key)
                        # Allow sentinel to force exact MA mapping from GUI without changing CLI defaults
                        if key in {'ma_exact', '__ma_exact__'}:
                            if key not in cleaned:
                                cleaned.append(key)
                    if cleaned:
                        # Determine if GUI requested exact MA mapping via sentinel
                        gui_requests_exact = '__ma_exact__' in cleaned
                        effective_cleaned = [x for x in cleaned if x != '__ma_exact__']
                        # GUI whitelist should always use exact mapping to reflect user selection
                        indicator_whitelist = _expand_ma_variants(set(effective_cleaned), do_expand=False)
                        gui_original_order = effective_cleaned  # Preserve GUI order from JSON
                        logger.info("Loaded GUI indicator whitelist (raw %d): %s", len(effective_cleaned), ", ".join(effective_cleaned))
                        logger.info("Effective GUI whitelist (exact as selected): %s", ", ".join(effective_cleaned))
                    else:
                        logger.debug("GUI whitelist file present but contained no supported indicators; using auto mode")
        except Exception as e:
            logger.warning(f"Could not load GUI indicator whitelist: {e}")

    # Apply strict mode toggle
    try:
        CFG.STRICT_IND_ONLY = bool(args.strict_indicators)
        if CFG.STRICT_IND_ONLY:
            logger.info("Strict indicators mode enabled: using only indicator_output values; no candle-based fallbacks")
    except Exception:
        pass

    ensure_dir(CFG.OUT)
    # Clean output directory on each run (non-recursive file cleanup)
    try:
        for name in os.listdir(CFG.OUT):
            # Preserve whitelist files created by the GUI
            if name.lower().startswith("indicator_whitelist"):
                continue
            fp = os.path.join(CFG.OUT, name)
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
            except Exception:
                pass
    except Exception:
        pass

    symbols = args.symbols or Loader.detect_symbols()
    
    # Handle comma-separated symbols from GUI
    if symbols and len(symbols) == 1 and ',' in symbols[0]:
        # GUI passes symbols as single comma-separated string
        symbols = [s.strip() for s in symbols[0].split(',') if s.strip()]
        logger.debug(f"Parsed comma-separated symbols: {symbols}")
    
    # Apply --all override
    if args.all:
        args.limit = 0
    if not symbols:
        logger.error("No symbols detected.")
        return 2
    detected_total = len(symbols)
    # Deterministic ordering: sort alphabetically unless user supplied explicit order
    if not args.symbols:
        symbols.sort()
    if args.limit and args.limit > 0:
        symbols = symbols[: args.limit]
    logger.info(
        f"Detected {detected_total} symbols; analyzing {len(symbols)}: "
        + ", ".join(symbols[:20])
        + (" ..." if len(symbols) > 20 else "")
    )

    ok = 0
    run_ts = ts_now()
    
    # Parse user timeframes if provided
    user_timeframes = None
    if args.timeframes:
        user_timeframes = [tf.strip() for tf in args.timeframes.replace(',', ' ').split() if tf.strip()]
        logger.info(f"User selected timeframes: {', '.join(user_timeframes)}")
    
    signal_results: dict[str, dict] = {}
    combined_signals: list[dict] = []
    report_signal_lines: list[str] = []
    
    # Initialize MT5 connection for live price data
    mt5_manager = None
    try:
        from mt5_connector import get_mt5_connection
        mt5_manager = get_mt5_connection()
        mt5_connected = mt5_manager.connect()
        if mt5_connected:
            logger.info("✅ MT5 connected for live price data")
        else:
            logger.warning("⚠️ MT5 connection failed, using indicator prices")
    except Exception as e:
        logger.warning(f"⚠️ MT5 initialization failed: {e}, using indicator prices")
    
    for sym in symbols:
        try:
            logger.info(f"Analyze {sym}")
            
            # Try to get live current price from MT5 first, fallback to indicator price
            current_price_for_analysis = None
            
            # 1. Try MT5 live price (most accurate)
            if mt5_manager and mt5_manager.is_connected():
                try:
                    live_price = mt5_manager.get_current_price(sym)
                    if live_price:
                        current_price_for_analysis = live_price
                        logger.info(f"📡 Using MT5 live price for {sym}: {live_price}")
                    else:
                        logger.warning(f"⚠️ Failed to get MT5 price for {sym}")
                except Exception as e:
                    logger.warning(f"⚠️ MT5 price fetch error for {sym}: {e}")
            
            # 2. Fallback to indicator price
            if current_price_for_analysis is None:
                try:
                    global current_price_from_indicators
                    extracted_price = current_price_from_indicators
                    if extracted_price:
                        current_price_for_analysis = extracted_price
                        logger.info(f"📊 Using indicator price for {sym}: {extracted_price}")
                    print(f"DEBUG: Passing extracted_price={extracted_price} to Aggregator for {sym}")
                except NameError:
                    print(f"DEBUG: No global current_price_from_indicators available for {sym}")
            
            if current_price_for_analysis is None:
                logger.warning(f"⚠️ No current price available for {sym}, analysis may be inaccurate")
            
            ag = Aggregator(sym, indicators=None, whitelist=indicator_whitelist, user_timeframes=user_timeframes, current_price=current_price_for_analysis)
            result = ag.run()
            if result is None:
                logger.error(f"Aggregator.run() returned None for {sym}")
                continue
            # Build minimal signal summary per symbol (no per-symbol files)
            fi = dict(result.get("final_signal") or {})
            idea = result.get("trade_idea") or {}
            if idea:
                fi["entry"] = idea.get("entry")
                fi["stoploss"] = idea.get("sl")
                fi["takeprofit"] = idea.get("tp")
                # Include smart entry information
                fi["order_type"] = idea.get("order_type", "market")
                fi["entry_reason"] = idea.get("entry_reason", "Giá thị trường")
                fi["confidence_boost"] = idea.get("confidence_boost", 0.0)
                fi["smart_entry_used"] = idea.get("smart_entry_used", False)
            else:
                fi.setdefault("entry", None)
                fi.setdefault("stoploss", None)
                fi.setdefault("takeprofit", None)
            signal_json = {
                "symbol": result.get("symbol"),
                "timestamp": result.get("timestamp"),
                "final_signal": fi,
                "risk_aware_actions": [
                    {
                        'action_type': action.get('action_type', 'unknown'),
                        'symbol': action.get('symbol', ''),
                        'direction': action.get('direction', ''),
                        'entry_price': action.get('entry_price', 0.0),
                        'volume': round(action.get('volume', 0.0), 2),
                        'stop_loss': action.get('stop_loss'),
                        'take_profit': action.get('take_profit'),
                        'reason': action.get('reason', ''),
                        'confidence': round(action.get('confidence', 0.0), 1),
                        'risk_level': action.get('risk_level', 'moderate'),
                        'order_type': action.get('order_type', 'market'),
                        'priority': action.get('priority', 1),
                        'conditions': action.get('conditions', {})
                    } for action in result.get("risk_aware_actions", [])
                ]
            }
            # Write per-symbol signal JSON and VI report (restore original behavior)
            ts = ts_now()
            json_fp = os.path.join(CFG.OUT, f"{sym}_signal_{ts}.json")
            overwrite_json_safely(json_fp, signal_json, backup=False)
            txt_fp = os.path.join(CFG.OUT, f"{sym}_report_vi_{ts}.txt")
            txt_en_fp = os.path.join(CFG.OUT, f"{sym}_report_en_{ts}.txt")
            logger.debug(f"Render report for {sym}")
            content: str = ""
            try:
                content = Report.build_bilingual(result, indicator_whitelist=gui_original_order or indicator_whitelist) or ""
            except Exception as rep_err:
                logger.exception(f"Report render failed for {sym}: {rep_err}")
                content = ""
            if not isinstance(content, str) or not content.strip():
                fi_sig = (result.get("final_signal") or {}).get("signal") or "NEUTRAL"
                fi_conf = int(round(ffloat((result.get("final_signal") or {}).get("confidence"), 0.0)))
                idea = result.get("trade_idea") or {}
                entry = idea.get("entry"); sl = idea.get("sl"); tp = idea.get("tp")
                content = (
                    f"Thời gian: {result.get('timestamp')}\n\n"
                    f"Ký hiệu: {result.get('symbol')}\n\n"
                    f"Tín hiệu: {str(fi_sig).upper()}\n"
                    f"Độ tin cậy: {fi_conf}%\n"
                    f"Entry: {entry}\n"
                    f"Stoploss: {sl}\n"
                    f"Takeprofit: {tp}\n\n"
                    "(Bản tóm tắt tối giản do lỗi kết xuất báo cáo chi tiết)\n"
                )
            with open(txt_fp, "w", encoding="utf-8") as f:
                f.write(content)

            # English report = translated Vietnamese content
            try:
                content_en: str = _translate_vi_to_en(content)
                if not isinstance(content_en, str) or not content_en.strip():
                    # Minimal fallback
                    fi_sig = (result.get("final_signal") or {}).get("signal") or "NEUTRAL"
                    fi_conf = int(round(ffloat((result.get("final_signal") or {}).get("confidence"), 0.0)))
                    idea = result.get("trade_idea") or {}
                    entry = idea.get("entry"); sl = idea.get("sl"); tp = idea.get("tp")
                    content_en = (
                        f"Time: {result.get('timestamp')}\n\n"
                        f"Symbol: {result.get('symbol')}\n\n"
                        f"Signal: {str(fi_sig).upper()}\n"
                        f"Confidence: {fi_conf}%\n"
                        f"Entry: {entry}\n"
                        f"Stoploss: {sl}\n"
                        f"Takeprofit: {tp}\n\n"
                        "(Minimal summary due to translation issue)\n"
                    )
            except Exception as rep_en_err:
                logger.exception(f"English translation failed for {sym}: {rep_en_err}")
                content_en = ""
            try:
                if content_en:
                    # Immediate cleanup before writing
                    try:
                        for k,v in (
                            ('thu hẹp (nén)', 'narrowing (squeeze)'),
                            ('có thể sắp bứt phá', 'potential breakout soon'),
                            ('xếp chồng up', 'bullish stack'),
                            ('xếp chồng tăng', 'bullish stack'),
                            ('xếp chồng giảm', 'bearish stack')
                        ):
                            content_en = content_en.replace(k, v)
                    except Exception:
                        pass
                    with open(txt_en_fp, "w", encoding="utf-8") as f:
                        f.write(content_en)
            except Exception as io_en_err:
                logger.exception(f"Failed writing English report for {sym}: {io_en_err}")

            logger.info(
                f"Saved {json_fp}\n"
                f"Saved {txt_fp} ({len(content.encode('utf-8'))} bytes)\n"
                f"Saved {txt_en_fp} ({len((content_en or '').encode('utf-8'))} bytes)"
            )
            try:
                signal_results[sym] = result  # Store complete result for enhanced actions
            except Exception:
                pass
            ok += 1
        except Exception as e:
            logger.exception(f"Failed {sym}: {e}")

    # After processing symbols: generate standalone account management report & bot JSON
    try:
        scan = load_account_scan()
        if scan:
            # For cross-check, use first symbol's final signal if available; else NEUTRAL
            current_sig = "NEUTRAL"
            try:
                # Use signal from signal_results if available
                if symbols and symbols[0] in signal_results:
                    # Fix: Get signal from final_signal sub-dict, not direct 'signal' key
                    result_data = signal_results[symbols[0]]
                    current_sig = (result_data.get('final_signal') or {}).get('signal') or 'NEUTRAL'
                    logger.info(f"Using signal for {symbols[0]}: {current_sig}")
                elif 'result' in locals() and isinstance(result, dict):
                    current_sig = (result.get('final_signal') or {}).get('signal') or 'NEUTRAL'
            except Exception as e:
                logger.warning(f"Error getting current signal: {e}")
                pass
            acct = analyze_positions(scan, symbols[0] if symbols else '', current_sig, signal_results)
            # Enrich each position with action_hint for BOT
            for p in acct.get('positions', []):
                hint = 'hold'
                peq = p.get('pct_equity') or 0
                if peq >= 3:
                    hint = 'scale_out'
                elif peq <= -2:
                    hint = 'cut'
                # conflict detection already described in suggest text; simple rule:
                sugg_txt = (p.get('suggest') or '').lower()
                if 'hedge' in sugg_txt or 'trái chiều' in sugg_txt:
                    hint = 'review_conflict'
                p['action_hint'] = hint

            # Calculate overall risk metrics
            risk_metrics = calculate_risk_metrics(scan, acct.get('positions', []))
            
            # Load risk settings for enhanced decision logic
            try:
                import json
                with open('risk_management/risk_settings.json', 'r', encoding='utf-8') as f:
                    decision_risk_settings = json.load(f)
            except Exception:
                decision_risk_settings = {}
            
            # Build advanced per-position actions using enhanced decision logic
            def _norm_symbol(s: str) -> str:
                if not isinstance(s, str):
                    return ''
                if s.endswith('_m'):
                    return s[:-2]
                return s
            
            actions: list[dict] = []
            # Store for later merging into final JSON
            global position_management_actions
            position_management_actions = []
            
            for p in acct.get('positions', []):
                psym = p.get('symbol')
                base = _norm_symbol(psym)
                # Fix signal lookup - try normalized base first, then psym, then base without dots
                base_clean = base.rstrip('.')
                psym_clean = psym.rstrip('.')
                sig = (signal_results.get(base_clean) or 
                       signal_results.get(psym_clean) or 
                       signal_results.get(base) or 
                       signal_results.get(psym) or {})
                logger.debug(f"Position signal lookup: psym={psym}, base={base}, base_clean={base_clean}, sig_keys={list(signal_results.keys())}, sig_signal={sig.get('final_signal', {}).get('signal', 'NO_SIGNAL')}")
                
                # Use enhanced decision logic
                decision = enhanced_action_decision(p, sig, risk_metrics, args, decision_risk_settings)
                
                # Extract decision results
                rec_primary = decision.get('action', 'hold')
                priority_score = int(decision.get('priority_score', 0))
                rationale_text = decision.get('rationale', 'Giữ nguyên')
                proposed_sl = decision.get('proposed_sl')
                proposed_tp = decision.get('proposed_tp')
                move_sl_price = decision.get('move_sl_price')
                move_tp_price = decision.get('move_tp_price')
                trailing_config = decision.get('trailing_config')
                risk_factors = decision.get('risk_factors', {})
                
                rec_list = [rec_primary]
                rationale = [rationale_text]
                
                # Build action output using enhanced decision results
                actions.append({
                    'symbol': psym,
                    'base_symbol': base,
                    'ticket': p.get('ticket'),
                    'direction': p.get('direction'),
                    'volume': p.get('volume'),
                    'profit': p.get('profit'),
                    'pct_equity': p.get('pct_equity'),
                    'pct_balance': p.get('pct_balance'),
                    'current_signal': sig.get('final_signal', {}).get('signal', 'NEUTRAL'),
                    'signal_confidence': sig.get('final_signal', {}).get('confidence', 0),
                    'signal_alignment': risk_factors.get('signal_alignment', 'NEUTRAL'),
                    'primary_action': rec_primary,
                    'priority_score': priority_score,
                    'rationale': rationale_text,
                    'proposed_sl': proposed_sl,
                    'proposed_tp': proposed_tp,
                    'move_sl_price': move_sl_price,
                    'move_tp_price': move_tp_price,
                    'trailing_config': trailing_config,
                    'risk_level': p.get('risk_level', 'LOW'),
                    'position_risk_pct': risk_factors.get('position_risk_pct', 0),
                    'overall_risk': risk_factors.get('overall_risk', 'LOW'),
                    'current_sl': p.get('sl') or p.get('stoploss'),
                    'current_tp': p.get('tp') or p.get('takeprofit'),
                    'entry_price': p.get('price_open'),
                    'current_price': p.get('price_current'),
                    'pips': risk_factors.get('pips_profit_loss', 0)
                })
            
            # Sync action_hint with primary actions for each position
            try:
                act_map = {}
                for a in actions:
                    sym_a = a.get('symbol')
                    if sym_a and sym_a not in act_map:
                        act_map[sym_a] = a.get('primary_action')
                for p in acct.get('positions', []):
                    ps = p.get('symbol')
                    if ps in act_map:
                        p['action_hint'] = act_map[ps]
            except Exception:
                pass

            # Write enhanced order-handling outputs with risk metrics
            actions_json_path = os.path.join(CFG.OUT, 'account_positions_actions.json')
            risk_summary = {
                'overall_risk_level': risk_metrics.get('overall_risk_level', 'UNKNOWN'),
                'drawdown_pct': risk_metrics.get('drawdown_pct', 0),
                'floating_pnl_pct': risk_metrics.get('floating_pnl_pct', 0),  # Add floating P&L
                'margin_level': risk_metrics.get('margin_level', 0),
                'concentration_risk_pct': risk_metrics.get('concentration_risk_pct', 0),
                'total_exposure_pct': risk_metrics.get('exposure_pct', 0),
                'positions_count': risk_metrics.get('positions_count', 0),
                'losing_positions': risk_metrics.get('losing_positions', 0),
                'winning_positions': risk_metrics.get('winning_positions', 0)
            }
            
            # Store position management actions for final JSON merge
            position_management_actions = actions.copy()
            
            # Sort actions by priority score (highest first) - ensure int conversion
            def safe_priority(action):
                priority = action.get('priority_score', 0)
                try:
                    return int(priority) if priority is not None else 0
                except (ValueError, TypeError):
                    return 0
            actions.sort(key=safe_priority, reverse=True)
            
            output_data = {
                'risk_summary': risk_summary,
                'actions': actions,
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'enhanced_position_management'
            }
            overwrite_json_safely(actions_json_path, output_data, backup=False)
            
            # Create detailed reports
            actions_txt_path = os.path.join(CFG.OUT, 'account_positions_actions_vi.txt')
            actions_txt_path_en = os.path.join(CFG.OUT, 'account_positions_actions_en.txt')
            
            try:
                # Vietnamese report
                lines_vi = [
                    "=== BÁO CÁO QUẢN LÝ VỊ THẾ ===",
                    f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "--- TỔNG QUAN RỦI RO ---",
                    f"Mức rủi ro tổng thể: {risk_summary['overall_risk_level']}",
                    f"P&L Floating: {risk_summary['floating_pnl_pct']:+.2f}%",
                    f"Margin Level: {risk_summary['margin_level']:.0f}%",
                    f"Rủi ro tập trung: {risk_summary['concentration_risk_pct']:.1f}%",
                    f"Tổng exposure: {risk_summary['total_exposure_pct']:.1f}%",
                    f"Vị thế: {risk_summary['positions_count']} (Lãi: {risk_summary['winning_positions']}, Lỗ: {risk_summary['losing_positions']})",
                    "",
                    "--- HÀNH ĐỘNG ƯU TIÊN (theo điểm số) ---"
                ]
                
                # English report
                lines_en = [
                    "=== ENHANCED POSITION MANAGEMENT REPORT ===",
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "--- RISK OVERVIEW ---",
                    f"Overall Risk Level: {risk_summary['overall_risk_level']}",
                    f"Floating P&L: {risk_summary['floating_pnl_pct']:+.2f}%",
                    f"Margin Level: {risk_summary['margin_level']:.0f}%",
                    f"Concentration Risk: {risk_summary['concentration_risk_pct']:.1f}%",
                    f"Total Exposure: {risk_summary['total_exposure_pct']:.1f}%",
                    f"Positions: {risk_summary['positions_count']} (Winning: {risk_summary['winning_positions']}, Losing: {risk_summary['losing_positions']})",
                    "",
                    "--- PRIORITY ACTIONS (by score) ---"
                ]
                
                for a in actions:
                    # Extract position details
                    entry_price = a.get('entry_price', 0)
                    current_price = a.get('current_price', 0)
                    volume = a.get('volume', 0)
                    profit = a.get('profit', 0)
                    direction = a.get('direction', '')
                    
                    # Vietnamese action line with detailed position info
                    line_vi = f"• {a['symbol']} {direction} Vol:{volume}"
                    if entry_price and current_price:
                        line_vi += f" | Entry:{entry_price:.2f} -> Current:{current_price:.2f}"
                    line_vi += f" | P&L: {profit:.0f}$ ({a.get('pct_equity', 0):+.2f}%)"
                    line_vi += f" | [{a['priority_score']}] -> {a['primary_action']}"
                    line_vi += f" | {a['rationale']}"
                    line_vi += f" | Rủi ro: {a.get('position_risk_pct', 0):.1f}%"
                    line_vi += f" | Signal: {a.get('current_signal', 'NEUTRAL')}"
                    
                    # English action line with detailed position info
                    line_en = f"• {a['symbol']} {direction} Vol:{volume}"
                    if entry_price and current_price:
                        line_en += f" | Entry:{entry_price:.2f} -> Current:{current_price:.2f}"
                    line_en += f" | P&L: {profit:.0f}$ ({a.get('pct_equity', 0):+.2f}%)"
                    line_en += f" | [{a['priority_score']}] -> {a['primary_action']}"
                    
                    # Translate rationale to English
                    rationale_en = a['rationale']
                    translations = {
                        # 🚨 Emergency & Urgent
                        '🚨 KHẨN CẤP:': '🚨 URGENT:',
                        'KHẨN CẤP:': 'URGENT:',
                        'đóng ngay': 'close immediately',
                        'đóng ngay lập tức': 'close immediately',
                        'đóng toàn bộ': 'close all',
                        
                        # 📊 Risk & Margin
                        'Margin thấp': 'Low margin',
                        'tín hiệu ngược chiều': 'opposite signal', 
                        'tín hiệu ngược': 'opposite signal',
                        'tín hiệu đồng pha': 'aligned signal',
                        'tín hiệu đồng pha mạnh': 'strong aligned signal',
                        'tín hiệu trung tính': 'neutral signal',
                        
                        # 💰 Volume & Size Management
                        'Khối lượng quá lớn': 'Volume too large',
                        'Khối lượng lớn': 'Large volume',
                        'Khối lượng cao': 'High volume',
                        '🔧 giảm': '🔧 reduce',
                        'giảm khối lượng': 'reduce volume',
                        'giảm 30%': 'reduce 30%',
                        'giảm 50%': 'reduce 50%',
                        'giảm 70%': 'reduce 70%',
                        
                        # 💎 Profit Taking
                        '💎 Lãi rất lớn': '💎 Very large profit',
                        'Lãi rất lớn': 'Very large profit',
                        '💰 Lãi lớn': '💰 Large profit', 
                        'Lãi lớn': 'Large profit',
                        '📊 Lãi khá': '📊 Good profit',
                        'Lãi khá': 'Good profit',
                        '✅ Lãi': '✅ Profit',
                        'Lãi nhỏ': 'Small profit',
                        'Lãi ít': 'Little profit',
                        'chốt': 'take profit',
                        'chốt 100%': 'take 100% profit',
                        'chốt 30%': 'take 30% profit',
                        'chốt 50%': 'take 50% profit',
                        'chốt 70%': 'take 70% profit',
                        'bảo toàn': 'secure',
                        
                        # 📉 Loss Management
                        '💀 Lỗ cực lớn': '💀 Extreme loss',
                        'Lỗ cực lớn': 'Extreme loss',
                        '💥 Lỗ lớn': '💥 Large loss',
                        'Lỗ lớn': 'Large loss',
                        '📉 Lỗ trung bình': '📉 Medium loss',
                        'Lỗ trung bình': 'Medium loss',
                        'Lỗ TB': 'Medium loss',
                        '📈 Lỗ nhỏ': '📈 Small loss',
                        'Lỗ nhỏ': 'Small loss',
                        
                        # 🛡️ Protection & Stops
                        'trailing SL': 'trailing SL',
                        'trailing SL chặt': 'tight trailing SL',
                        'siết SL': 'tighten SL',
                        'SL về BE': 'SL to breakeven',
                        'SL về breakeven': 'SL to breakeven',
                        '🛡️ đặt SL': '🛡️ set SL',
                        'đặt SL': 'set SL',
                        'đặt TP': 'set TP',
                        'bảo vệ': 'protect',
                        
                        # 🎯 Risk Management
                        '🎯 Tập trung rủi ro cao': '🎯 High concentration risk',
                        'Tập trung rủi ro cao': 'High concentration risk',
                        'Rủi ro tập trung cao': 'High concentration risk',
                        
                        # 📊 Neutral & Hold
                        '📊 Giữ nguyên': '📊 Hold',
                        'Giữ nguyên': 'Hold',
                        '📊 Breakeven': '📊 Breakeven',
                        'Breakeven': 'Breakeven',
                        'giữ': 'hold',
                        'giữ lệnh': 'hold position',
                        'quan sát': 'observe',
                        
                        # Icons & Emojis preservation
                        'pips': 'pips',
                        'đóng/đảo chiều': 'close/reverse'
                    }
                    for vi_text, en_text in translations.items():
                        rationale_en = rationale_en.replace(vi_text, en_text)
                    
                    line_en += f" | {rationale_en}"
                    line_en += f" | Risk: {a.get('position_risk_pct', 0):.1f}%"
                    line_en += f" | Signal: {a.get('current_signal', 'NEUTRAL')}"
                    
                    # Add SL/TP info if available
                    if a.get('proposed_sl'):
                        line_vi += f" | SL đề xuất: {a['proposed_sl']}"
                        line_en += f" | Proposed SL: {a['proposed_sl']}"
                    if a.get('proposed_tp'):
                        line_vi += f" | TP đề xuất: {a['proposed_tp']}"
                        line_en += f" | Proposed TP: {a['proposed_tp']}"
                    if a.get('trailing_config'):
                        tr = a['trailing_config']
                        line_vi += f" | Trailing: {tr.get('activate_price', 'N/A')} (khoảng cách: {tr.get('trail_distance', 'N/A')})"
                        line_en += f" | Trailing: {tr.get('activate_price', 'N/A')} (distance: {tr.get('trail_distance', 'N/A')})"
                    
                    lines_vi.append(line_vi)
                    lines_en.append(line_en)
                
                # Write reports
                with open(actions_txt_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines_vi) + "\n")
                with open(actions_txt_path_en, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines_en) + "\n")
                
                logger.info(f"Enhanced reports saved:\n{actions_json_path}\n{actions_txt_path}\n{actions_txt_path_en}")
                
            except Exception:
                logger.exception('Failed writing enhanced action reports')
    except Exception:
        logger.exception("Account report generation failed")

    # 🚀 SAVE ENHANCED RISK-AWARE ACTIONS SUMMARY
    try:
        # Collect all risk-aware actions from all analyzed symbols
        all_enhanced_actions = []
        
        enhanced_summary = {
            'total_symbols_analyzed': len(signal_results),
            'signals_generated': 0,
            'actions_by_type': {},
            'high_priority_actions': 0,
            'total_actions': 0
        }
        
        for symbol, result in signal_results.items():
            if not result:
                continue
                
            final_signal = result.get('final_signal', {})
            risk_aware_actions = result.get('risk_aware_actions', [])
            
            if final_signal.get('signal') in ['BUY', 'SELL']:
                enhanced_summary['signals_generated'] += 1
            
            for action in risk_aware_actions:
                # Add symbol context to action
                action_with_context = dict(action.__dict__) if hasattr(action, '__dict__') else dict(action)
                action_with_context['analysis_timestamp'] = result.get('timestamp', '')
                all_enhanced_actions.append(action_with_context)
                
                # Update summary stats
                action_type = action_with_context.get('action_type', 'unknown')
                enhanced_summary['actions_by_type'][action_type] = enhanced_summary['actions_by_type'].get(action_type, 0) + 1
                enhanced_summary['total_actions'] += 1
                
                # Fix: Ensure priority is int for comparison
                try:
                    priority_val = int(action_with_context.get('priority', 10))
                    if priority_val <= 3:
                        enhanced_summary['high_priority_actions'] += 1
                except (ValueError, TypeError):
                    # If priority is not a valid number, treat as low priority
                    pass
        
        # Add position management actions if they exist (from account scan section)
        try:
            if 'position_management_actions' in globals():
                all_enhanced_actions.extend(position_management_actions)
                
                # Update summary to reflect merged actions
                for pm_action in position_management_actions:
                    action_type = pm_action.get('primary_action', 'position_management')
                    enhanced_summary['actions_by_type'][action_type] = enhanced_summary['actions_by_type'].get(action_type, 0) + 1
                    enhanced_summary['total_actions'] += 1
                    
                    priority_score = int(pm_action.get('priority_score', 100))
                    if priority_score >= 70:  # High priority threshold
                        enhanced_summary['high_priority_actions'] += 1
        except Exception as e:
            logger.debug(f"Error merging position management actions: {e}")
        
        # Save enhanced actions to account_positions_actions.json (consolidated)
        actions_path = os.path.join(CFG.OUT, 'account_positions_actions.json')
        actions_data = {
            'summary': enhanced_summary,
            'actions': all_enhanced_actions,
            'generation_timestamp': datetime.now().isoformat(),
            'analysis_type': 'comprehensive_trading_actions_with_account_context'
        }
        
        overwrite_json_safely(actions_path, actions_data, backup=False)
        
        # 📊 ENHANCED EXECUTION HISTORY LOGGING: Save all comprehensive actions
        try:
            if all_enhanced_actions:
                # Convert enhanced actions to execution history format
                history_actions = []
                for action in all_enhanced_actions:
                    history_record = {
                        "symbol": action.get("symbol", ""),
                        "action": action.get("action", ""),
                        "type": action.get("type", ""),
                        "ticket": action.get("ticket", ""),
                        "reason": action.get("reason", ""),
                        "signal_trigger": action.get("signal_trigger", ""),
                        "confidence": action.get("confidence", 0),
                        "priority": action.get("priority", ""),
                        "risk_level": action.get("risk_level", ""),
                        "current_profit_pips": action.get("current_profit_pips", 0),
                        "position_type": action.get("position_type", ""),
                        "new_sl": action.get("new_sl", 0),
                        "new_tp": action.get("new_tp", 0)
                    }
                    history_actions.append(history_record)
                
                # Get symbol from first action or use generic name
                symbol_for_history = history_actions[0].get("symbol", "MULTI") if history_actions else "NONE"
                save_action_execution_history(history_actions, symbol_for_history)
                logger.info(f"💾 Saved {len(history_actions)} enhanced actions to execution history")
        except Exception as e:
            logger.error(f"Failed to save enhanced execution history: {e}")
        
        # Skip consolidated text reports - use detailed position management reports instead
        if False and all_enhanced_actions:
            try:
                actions_txt_path = os.path.join(CFG.OUT, 'account_positions_actions_vi.txt')
                actions_txt_path_en = os.path.join(CFG.OUT, 'account_positions_actions_en.txt')
                
                # Sort actions by priority (highest first)
                sorted_actions = sorted(all_enhanced_actions, key=lambda x: x.get('priority', 10))
                
                # Separate actions by notification requirement
                need_notification = [a for a in sorted_actions if a.get('requires_notification', False)]
                no_notification = [a for a in sorted_actions if not a.get('requires_notification', False)]
                
                # Vietnamese consolidated report
                lines_vi = [
                    "=== BÁO CÁO HÀNH ĐỘNG TRADING TỔNG HỢP ===",
                    f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "--- TỔNG QUAN ---",
                    f"Tổng symbols phân tích: {enhanced_summary['total_symbols_analyzed']}",
                    f"Tín hiệu được tạo: {enhanced_summary['signals_generated']}",
                    f"Tổng hành động: {enhanced_summary['total_actions']} (Ưu tiên cao: {enhanced_summary['high_priority_actions']})",
                    f"Phân loại: {', '.join([f'{k}: {v}' for k, v in enhanced_summary['actions_by_type'].items()])}",
                    "",
                ]
                
                # English consolidated report
                lines_en = [
                    "=== COMPREHENSIVE TRADING ACTIONS REPORT ===",
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "--- OVERVIEW ---",
                    f"Total symbols analyzed: {enhanced_summary['total_symbols_analyzed']}",
                    f"Signals generated: {enhanced_summary['signals_generated']}",
                    f"Total actions: {enhanced_summary['total_actions']} (High priority: {enhanced_summary['high_priority_actions']})",
                    f"Categories: {', '.join([f'{k}: {v}' for k, v in enhanced_summary['actions_by_type'].items()])}",
                    f"📢 Require notification: {len(need_notification)} | 🔕 No notification: {len(no_notification)}",
                    "",
                ]
                
                # Function to create action line
                def create_action_line(action, idx_global):
                    symbol = action.get('symbol', '')
                    direction = action.get('direction', '')
                    action_type = action.get('action_type', '')
                    entry_price = action.get('entry_price', 0)
                    volume = action.get('volume', 0)
                    stop_loss = action.get('stop_loss', 0)
                    take_profit = action.get('take_profit', 0)
                    confidence = action.get('confidence', 0)
                    reason = action.get('reason', '')
                    priority = action.get('priority', 10)
                    risk_level = action.get('risk_level', 'unknown')
                    order_type = action.get('order_type', 'market')
                    entry_type = action.get('conditions', {}).get('entry_type', 'UNKNOWN')
                    requires_notification = action.get('requires_notification', False)
                    
                    # Vietnamese action line
                    action_type_vi = {
                        'primary_entry': 'Lệnh chính',
                        'hedge_entry': 'Lệnh hedge', 
                        'dca_entry': 'Lệnh DCA',
                        'limit_order': 'Lệnh limit',
                        'trailing_stop': 'Trailing stop'
                    }.get(action_type, action_type)
                    
                    risk_level_vi = {
                        'low': 'thấp',
                        'moderate': 'trung bình', 
                        'high': 'cao'
                    }.get(risk_level, risk_level)
                    
                    # Add notification indicator
                    notif_icon = "" if requires_notification else ""
                    
                    line_vi = f"{idx_global}. {notif_icon} [{action_type_vi}] {symbol} {direction}"
                    line_vi += f" | Giá vào: {entry_price:.5f}" if entry_price else ""
                    line_vi += f" | Khối lượng: {volume}" if volume else " | Khối lượng: auto"
                    line_vi += f" | SL: {stop_loss:.5f}" if stop_loss else ""
                    line_vi += f" | TP: {take_profit:.5f}" if take_profit else ""
                    line_vi += f" | Độ tin cậy: {confidence:.1f}%"
                    line_vi += f" | Ưu tiên: {priority}"
                    line_vi += f" | Rủi ro: {risk_level_vi}"
                    line_vi += f" | Loại: {order_type}"
                    line_vi += f" | Kiểu: {entry_type}"
                    line_vi += f" | Lý do: {reason}"
                    
                    # English action line
                    line_en = f"{idx_global}. {notif_icon} [{action_type.upper()}] {symbol} {direction}"
                    line_en += f" | Entry: {entry_price:.5f}" if entry_price else ""
                    line_en += f" | Volume: {volume}" if volume else " | Volume: auto"
                    line_en += f" | SL: {stop_loss:.5f}" if stop_loss else ""
                    line_en += f" | TP: {take_profit:.5f}" if take_profit else ""
                    line_en += f" | Confidence: {confidence:.1f}%"
                    line_en += f" | Priority: {priority}"
                    line_en += f" | Risk: {risk_level}"
                    line_en += f" | Type: {order_type}"
                    line_en += f" | Entry Type: {entry_type}"
                    
                    # Translate reason for English
                    reason_en = reason
                    translations = {
                        'HEDGE': 'HEDGE',
                        'DCA': 'DCA', 
                        'Lệnh chính': 'Primary entry',
                        'Lệnh limit': 'Limit order',
                        'gần resistance': 'near resistance',
                        'gần support': 'near support',
                        'tối ưu entry': 'optimize entry',
                        'entry tốt': 'good entry',
                        'Gần EMA20': 'Near EMA20',
                        'chống lại vị thế hiện có': 'against existing position',
                        'với': 'with',
                        'độ tin cậy': 'confidence'
                    }
                    for vi_text, en_text in translations.items():
                        reason_en = reason_en.replace(vi_text, en_text)
                    
                    line_en += f" | Reason: {reason_en}"
                    
                    return line_vi, line_en
                
                # Add actions requiring notification first
                if need_notification:
                    lines_vi.append("--- CẦN THÔNG BÁO: XỬ LÝ LỆNH TRẠNG THÁI HIỆN CÓ ---")
                    lines_en.append("--- NOTIFICATION REQUIRED: MANAGE EXISTING POSITIONS ---")
                    
                    idx = 1
                    for action in need_notification:
                        line_vi, line_en = create_action_line(action, idx)
                        lines_vi.append(line_vi)
                        lines_en.append(line_en)
                        idx += 1
                    
                    lines_vi.append("")
                    lines_en.append("")
                
                # Write consolidated reports
                with open(actions_txt_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines_vi) + "\n")
                with open(actions_txt_path_en, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines_en) + "\n")
                
            except Exception:
                logger.exception('Failed writing consolidated action reports')
            
            logger.info(f"💼 Risk-Aware Actions saved: {actions_path}")
            logger.info(f"📊 Total: {enhanced_summary['total_actions']} actions, High Priority: {enhanced_summary['high_priority_actions']}")
            logger.info(f"📈 Signals: {enhanced_summary['signals_generated']}/{enhanced_summary['total_symbols_analyzed']}")
        
    except Exception as e:
        logger.error(f"Failed to save enhanced risk-aware actions: {e}")

    # Final English cleanup pass (remove any lingering Vietnamese words)
    try:
        _post_clean_english_reports()
    except Exception:
        pass
    
    # � DISABLED: AUTO DCA DETECTION AND EXECUTION 
    # DCA is now handled internally by comprehensive analysis above
    # This independent DCA detection was causing duplicates and incorrect reference logic
    try:
        logger.info("🔄 DCA opportunities handled by comprehensive analysis - skipping independent detection")
        
        # Load risk settings for reporting only
        risk_settings_path = 'risk_management/risk_settings.json'
        dca_enabled = False
        
        if os.path.exists(risk_settings_path):
            with open(risk_settings_path, 'r', encoding='utf-8') as f:
                risk_settings = json.load(f)
                dca_enabled = risk_settings.get('enable_dca', False)
                
        if dca_enabled:
            logger.info("✅ DCA is enabled - opportunities managed by comprehensive analysis above")
            logger.info("🚫 Independent DCA detection disabled to prevent duplicates")
        else:
            logger.info("❌ DCA is disabled in risk settings")
        
        # 🚫 COMPLETELY SKIP independent DCA detection
        # All DCA handling is done by comprehensive analysis above with correct reference logic
        logger.info("🔄 DCA detection skipped: DCA disabled in settings")
    
    except Exception as e:
        logger.error(f"❌ DCA detection/execution failed: {e}")
        import traceback
        traceback.print_exc()

    # 📊 AUTO-SAVE TRADING HISTORY (Tối ưu cho auto trading)
    try:
        # Kiểm tra auto history settings từ args
        auto_history_enabled = False
        
        if hasattr(args, 'no_history') and args.no_history:
            auto_history_enabled = False  # Tắt hoàn toàn
        elif hasattr(args, 'auto_history') and args.auto_history:
            auto_history_enabled = True   # Bắt buộc bật
        elif hasattr(args, 'history') and args.history:
            auto_history_enabled = True   # Bắt buộc bật
        else:
            # Mặc định: Chỉ chạy nếu không phải auto trading (ít symbols + không có --limit cao)
            is_likely_auto_trading = (
                len(symbols) <= 3 and 
                args.limit > 0 and args.limit <= 5 and
                any(['--verbose' in str(arg) for arg in sys.argv if isinstance(arg, str)])
            )
            auto_history_enabled = not is_likely_auto_trading
        
        if auto_history_enabled:
            logger.info("📊 Thu thập nhanh lịch sử giao dịch (chế độ tối ưu)...")
            
            # Chỉ lưu execution reports hiện có, không query MT5 để tránh xung đột
            final_actions = []
            try:
                import os
                execution_reports_file = "reports/execution_reports.json"
                if os.path.exists(execution_reports_file):
                    import json
                    with open(execution_reports_file, 'r', encoding='utf-8') as f:
                        execution_data = json.load(f)
                    
                    # Lấy các actions gần đây
                    recent_reports = execution_data.get('execution_reports', [])[-10:]  # Giảm xuống 10
                    for report in recent_reports:
                        final_actions.extend(report.get('actions', []))
                    
                    logger.info(f"📋 Cập nhật từ {len(final_actions)} actions gần đây (không query MT5)")
            except Exception as e:
                logger.debug(f"Không thể đọc execution reports: {e}")
            
            # Chỉ lưu action summary, không query MT5 closed positions
            if final_actions:
                try:
                    # Tạo summary từ actions thay vì closed positions
                    action_summary_file = "reports/action_summary.json"
                    summary_data = {
                        'timestamp': datetime.now().isoformat(),
                        'total_actions': len(final_actions),
                        'action_types': {},
                        'recent_actions': final_actions[-5:],  # 5 actions gần nhất
                        'note': 'Auto-trading optimized summary (no MT5 query)'
                    }
                    
                    # Đếm loại actions
                    for action in final_actions:
                        action_type = action.get('type', action.get('action', 'unknown'))
                        summary_data['action_types'][action_type] = summary_data['action_types'].get(action_type, 0) + 1
                    
                    os.makedirs('reports', exist_ok=True)
                    with open(action_summary_file, 'w', encoding='utf-8') as f:
                        json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)
                    
                    logger.info(f"� Đã cập nhật action summary ({len(final_actions)} actions)")
                except Exception as e:
                    logger.debug(f"Lỗi tạo action summary: {e}")
        else:
            logger.debug("📊 Auto-history bị tắt để tối ưu cho auto trading")
    except Exception as e:
        logger.debug(f"Lỗi auto-save (bỏ qua): {e}")
        # Không log error để tránh làm nhiễu auto trading

    # Close MT5 connection if it was opened
    try:
        if mt5_manager and mt5_manager.is_connected():
            mt5_manager.disconnect()
            logger.info("🔌 MT5 connection closed")
    except Exception:
        pass
    
    logger.info(f"Completed: {ok}/{len(symbols)}")
    return 0 if ok else 1


def _calculate_dca_entry_price(risk_settings: dict, entry_price: float, level: int = 1, direction: str = "BUY", atr: float = None, swing_high: float = None, swing_low: float = None, symbol: str = "XAUUSD", timeframe: str = "M5") -> float:
    """
    Calculate DCA entry price for ATR and Fixed Pips modes (Fibonacci managed by dca_service.py)
    """
    try:
        dca_mode = risk_settings.get("dca_mode", "fixed_pips")
        
        # Skip for Fibonacci mode - handled by dca_service.py
        if "fibo" in dca_mode.lower():
            logger.debug("Fibonacci DCA entry price managed by dca_service.py")
            pip_value = _get_pip_value_static(entry_price)
            distance = (50.0 * pip_value) * level  # Simple fallback
            return entry_price - distance if direction == "BUY" else entry_price + distance
            
        pip_value = _get_pip_value_static(entry_price)
        
        # ATR-based distance calculation
        if ("ATR" in dca_mode or "atr" in dca_mode.lower()) and atr and atr > 0:
            # Use ATR for dynamic spacing - equal distance between adjacent levels
            atr_multiplier = risk_settings.get("dca_atr_multiplier", 1.0)  # Default 1x ATR
            single_step_distance = atr * atr_multiplier  # Distance between adjacent levels
            total_distance = single_step_distance * level  # Total distance from main entry to this level
            logger.debug(f"ATR DCA Level {level}: Single step={single_step_distance:.5f}, Total from entry={total_distance:.5f}")
        else:
            # Fixed pips mode - equal distance between adjacent levels  
            base_distance_pips = risk_settings.get("dca_distance_pips", 50.0)
            single_step_distance = base_distance_pips * pip_value
            total_distance = single_step_distance * level
            logger.debug(f"Fixed Pips DCA Level {level}: {base_distance_pips} pips * {level} = {total_distance:.5f}")
        
        # Calculate entry price based on direction
        if direction.upper() == "BUY":
            dca_entry = entry_price - total_distance
        else:  # SELL
            dca_entry = entry_price + total_distance
            
        logger.debug(f"DCA Entry Price L{level} ({direction}): {entry_price:.5f} → {dca_entry:.5f} (total_distance: {total_distance:.5f})")
        return dca_entry
        
    except Exception as e:
        logger.error(f"Error calculating DCA entry price: {e}")
        # Fallback calculation
        pip_value = _get_pip_value_static(entry_price)
        total_distance = (50.0 * pip_value) * level
        return entry_price - total_distance if direction == "BUY" else entry_price + total_distance


def _calculate_fibonacci_dca_distance(risk_settings: dict, entry_price: float, level: int, swing_high: float = None, swing_low: float = None, symbol: str = "XAUUSD", timeframe: str = "M5") -> float:
    """
    🚫 DISABLED: DCA Fibonacci calculation moved to dca_service.py
    Use FibonacciDCAService for all DCA operations
    """
    logger.info(f"🚫 _calculate_fibonacci_dca_distance disabled - use dca_service.py instead")
    return 50.0  # Fallback distance

def _get_fibonacci_level_from_indicator(fibo_level_name: str, symbol: str = "XAUUSD", timeframe: str = "M5") -> float:
    """
    🚫 DISABLED: Fibonacci level reading moved to dca_service.py
    Use FibonacciDCAService.load_fibonacci_data() instead
    """
    logger.info(f"🚫 _get_fibonacci_level_from_indicator disabled - use dca_service.py instead")
    return None

def _calculate_dca_volume(risk_settings: dict, base_volume: float, level: int, dca_volume_multiplier: float) -> float:
    """
    Calculate DCA volume for ATR and Fixed Pips modes (Fibonacci managed by dca_service.py)
    """
    try:
        dca_mode = risk_settings.get("dca_mode", "fixed_pips")
        
        # Skip for Fibonacci mode - handled by dca_service.py
        if "fibo" in dca_mode.lower():
            logger.debug("Fibonacci DCA volume managed by dca_service.py")
            return base_volume
            
        # ATR mode: Progressive scaling
        if "ATR" in dca_mode or "atr" in dca_mode.lower():
            multiplier = dca_volume_multiplier ** level
            volume = float(base_volume) * multiplier
            logger.debug(f"ATR DCA Level {level}: {base_volume} * {multiplier:.2f} = {volume:.2f}")
        else:
            # Fixed Pips mode: Simple scaling
            volume = float(base_volume) * float(dca_volume_multiplier)
            logger.debug(f"Fixed Pips DCA: {base_volume} * {dca_volume_multiplier} = {volume:.2f}")
        
        # Apply volume limits
        min_volume = risk_settings.get("min_volume_auto", 0.01)
        max_volume_setting = risk_settings.get("max_total_volume", "OFF")
        
        if max_volume_setting != "OFF":
            try:
                max_volume = float(max_volume_setting) * 0.3  # Max 30% for single DCA
            except (ValueError, TypeError):
                max_volume = 10.0
        else:
            max_volume = 10.0
        
        volume = max(min_volume, min(volume, max_volume))
        return round(volume, 2)
        
    except Exception as e:
        logger.error(f"Error calculating DCA volume: {e}")
        return base_volume


def _get_symbol_specific_pip_value(symbol: str, entry_price: float) -> float:
    """
    Get accurate pip value based on symbol and price - FIXED FOR DCA
    """
    symbol_upper = symbol.upper().rstrip('.').replace('_M', '').replace('_m', '')
    
    # PRECIOUS METALS
    if symbol_upper in ['XAUUSD', 'XAGUSD', 'GOLD', 'SILVER']:
        return 0.1  # Gold/Silver: 1 pip = 0.1
    
    # JPY PAIRS  
    elif 'JPY' in symbol_upper:
        return 0.01  # JPY pairs: 1 pip = 0.01
    
    # CRYPTO
    elif any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'XRP', 'DOGE', 'MATIC', 'AVAX', 'BNB']):
        if entry_price > 10000:  # BTC range
            return 1.0
        elif entry_price > 2000:  # ETH/High crypto  
            return 0.1  # ✅ Fixed: ETH should be 0.1, not 1.0
        else:  # Mid/Low crypto
            return 0.1
    
    # STANDARD FX
    else:
        return 0.0001  # Standard FX pip

def _get_pip_value_static(entry_price: float) -> float:
    """
    Static version of pip value calculation for standalone functions - UPDATED FOR NEW STANDARDS
    Note: This is a fallback when symbol is not available.
    Updated to match new BTC/ETH pip calculation: 1 pip = 1.0 price unit.
    """
    if entry_price > 10000:  # Likely BTC (very high-value crypto)
        return 1.0    # BTC: 1 pip = 1.0 price unit (100000.5 -> 100001.5 = 1 pip)  
    elif entry_price > 2000:  # Likely ETH (high-value crypto) or METALS
        return 0.1    # ✅ FIXED: ETH/METALS: 1 pip = 0.1 (4000 -> 4000.1 = 1 pip)
    elif entry_price > 100:  # Likely SOL/ADA/other mid-range crypto
        return 0.1    # ✅ FIXED: Other crypto: 1 pip = 0.1 (200.0 -> 200.1 = 1 pip)
    elif entry_price > 1:    # Likely major FX pairs
        return 0.0001  # Standard FX pip size (1.10000 -> 1.10001 = 0.1 pip)
    else:                    # Likely small crypto or JPY pairs
        return 0.01    # Use 0.01 for small crypto/JPY

    def _calculate_spread_adjusted_entry(self, current_price: float, signal_direction: str) -> float:
        """
        🔧 CRITICAL FIX: Calculate spread-adjusted entry price for market orders
        
        For BUY orders: Use ASK price (higher)
        For SELL orders: Use BID price (lower)
        If spread data unavailable, add conservative buffer
        """
        try:
            import MetaTrader5 as mt5
            
            # Try to get real-time tick data for accurate spread
            tick = mt5.symbol_info_tick(self.symbol)
            symbol_info = mt5.symbol_info(self.symbol)
            
            if tick and symbol_info:
                spread_points = tick.ask - tick.bid
                spread_pips = spread_points / self._get_pip_value(self.symbol)
                
                if signal_direction == 'BUY':
                    # For BUY: Use ASK price (we buy at higher price)
                    adjusted_entry = tick.ask
                    logger.debug(f"🔧 BUY Entry: Current={current_price:.5f} → ASK={adjusted_entry:.5f} (spread={spread_pips:.1f}pips)")
                else:  # SELL
                    # For SELL: Use BID price (we sell at lower price) 
                    adjusted_entry = tick.bid
                    logger.debug(f"🔧 SELL Entry: Current={current_price:.5f} → BID={adjusted_entry:.5f} (spread={spread_pips:.1f}pips)")
                    
                return adjusted_entry
            else:
                # Fallback: Add conservative spread buffer
                pip_value = self._get_pip_value(self.symbol)
                conservative_spread_pips = 2.0  # Conservative 2 pip spread assumption
                spread_buffer = conservative_spread_pips * pip_value
                
                if signal_direction == 'BUY':
                    adjusted_entry = current_price + spread_buffer
                    logger.warning(f"⚠️ BUY Entry fallback: {current_price:.5f} + {conservative_spread_pips}pips = {adjusted_entry:.5f}")
                else:  # SELL
                    adjusted_entry = current_price - spread_buffer  
                    logger.warning(f"⚠️ SELL Entry fallback: {current_price:.5f} - {conservative_spread_pips}pips = {adjusted_entry:.5f}")
                    
                return adjusted_entry
                
        except Exception as e:
            logger.error(f"❌ Spread adjustment error: {e}, using current price")
            return current_price


def save_trading_history(closed_positions: List[Dict], final_actions: List[Dict] = None, 
                        symbol: str = None) -> None:
    """
    Lưu lịch sử giao dịch các lệnh đã đóng với kết quả lãi/lỗ và các hành động cuối cùng
    
    Args:
        closed_positions: Danh sách các lệnh đã đóng
        final_actions: Các hành động cuối cùng được thực hiện
        symbol: Symbol cụ thể (tùy chọn)
    """
    try:
        from datetime import datetime
        import json
        import os
        
        if not closed_positions:
            logger.info("📊 Không có lệnh đã đóng để lưu lịch sử")
            return
            
        # Tạo thư mục reports nếu chưa tồn tại
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Tên file lịch sử giao dịch
        history_file = os.path.join(reports_dir, "trading_history.json")
        
        # Load dữ liệu hiện tại nếu có
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
            else:
                history_data = {
                    'metadata': {
                        'created': datetime.now().isoformat(),
                        'version': '1.0',
                        'description': 'Lịch sử giao dịch các lệnh đã đóng'
                    },
                    'closed_trades': [],
                    'summary_stats': {
                        'total_trades': 0,
                        'profitable_trades': 0,
                        'losing_trades': 0,
                        'total_profit': 0.0,
                        'total_loss': 0.0,
                        'net_profit': 0.0,
                        'win_rate': 0.0,
                        'symbols_traded': {}
                    }
                }
        except (FileNotFoundError, json.JSONDecodeError):
            history_data = {
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'version': '1.0',
                    'description': 'Lịch sử giao dịch các lệnh đã đóng'
                },
                'closed_trades': [],
                'summary_stats': {
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'losing_trades': 0,
                    'total_profit': 0.0,
                    'total_loss': 0.0,
                    'net_profit': 0.0,
                    'win_rate': 0.0,
                    'symbols_traded': {}
                }
            }
        
        # Xử lý từng lệnh đã đóng
        for position in closed_positions:
            try:
                # Lấy thông tin cơ bản
                ticket = position.get('ticket') or position.get('Ticket')
                symbol = position.get('symbol') or position.get('Symbol', 'UNKNOWN')
                position_type = position.get('type') or position.get('Type')
                volume = position.get('volume') or position.get('Volume', 0)
                open_price = position.get('price_open') or position.get('price_open', 0)
                close_price = position.get('price_close') or position.get('price_close', 0)
                open_time = position.get('time_open') or position.get('time_open')
                close_time = position.get('time_close') or position.get('time_close')
                profit = position.get('profit') or position.get('Profit', 0.0)
                swap = position.get('swap') or position.get('Swap', 0.0)
                commission = position.get('commission') or position.get('Commission', 0.0)
                sl = position.get('sl') or position.get('SL', 0)
                tp = position.get('tp') or position.get('TP', 0)
                
                # Tính toán pips
                direction = 'BUY' if str(position_type).lower() in ['0', 'buy', 'long'] else 'SELL'
                pips = calculate_pips(symbol, open_price, close_price, direction) if open_price and close_price else 0
                
                # Xác định loại kết quả
                net_profit = float(profit) + float(swap) + float(commission)
                result_type = 'PROFIT' if net_profit > 0 else ('LOSS' if net_profit < 0 else 'BREAKEVEN')
                
                # Tìm các hành động liên quan đến ticket này
                related_actions = []
                if final_actions:
                    related_actions = [
                        action for action in final_actions
                        if action.get('ticket') == ticket or 
                           action.get('position_id') == ticket or
                           action.get('symbol') == symbol
                    ]
                
                # Tạo record lịch sử giao dịch
                trade_record = {
                    'timestamp': datetime.now().isoformat(),
                    'ticket': ticket,
                    'symbol': symbol,
                    'direction': direction,
                    'volume': float(volume),
                    'open_price': float(open_price) if open_price else 0,
                    'close_price': float(close_price) if close_price else 0,
                    'open_time': open_time,
                    'close_time': close_time,
                    'sl': float(sl) if sl else 0,
                    'tp': float(tp) if tp else 0,
                    'profit': float(profit),
                    'swap': float(swap),
                    'commission': float(commission),
                    'net_profit': net_profit,
                    'pips': pips,
                    'result_type': result_type,
                    'final_actions': related_actions[:5],  # Lưu tối đa 5 hành động cuối cùng
                    'actions_count': len(related_actions)
                }
                
                # Thêm vào lịch sử
                history_data['closed_trades'].append(trade_record)
                
                # Cập nhật thống kê
                stats = history_data['summary_stats']
                stats['total_trades'] += 1
                
                if net_profit > 0:
                    stats['profitable_trades'] += 1
                    stats['total_profit'] += net_profit
                elif net_profit < 0:
                    stats['losing_trades'] += 1
                    stats['total_loss'] += abs(net_profit)
                
                stats['net_profit'] = stats['total_profit'] - stats['total_loss']
                stats['win_rate'] = (stats['profitable_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
                
                # Thống kê theo symbol
                if symbol not in stats['symbols_traded']:
                    stats['symbols_traded'][symbol] = {
                        'trades': 0, 'profit': 0.0, 'loss': 0.0, 'net': 0.0
                    }
                
                symbol_stats = stats['symbols_traded'][symbol]
                symbol_stats['trades'] += 1
                if net_profit > 0:
                    symbol_stats['profit'] += net_profit
                else:
                    symbol_stats['loss'] += abs(net_profit)
                symbol_stats['net'] = symbol_stats['profit'] - symbol_stats['loss']
                
                logger.info(f"💰 Đã lưu lịch sử: {symbol} #{ticket} {direction} {result_type} {net_profit:.2f}$")
                
            except Exception as e:
                logger.error(f"❌ Lỗi xử lý position {position}: {e}")
                continue
        
        # Giữ lại chỉ 1000 giao dịch gần nhất
        max_trades = 1000
        if len(history_data['closed_trades']) > max_trades:
            history_data['closed_trades'] = history_data['closed_trades'][-max_trades:]
        
        # Cập nhật metadata
        history_data['metadata']['last_updated'] = datetime.now().isoformat()
        history_data['metadata']['total_records'] = len(history_data['closed_trades'])
        
        # Lưu file
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Tạo file tóm tắt nhanh
        summary_file = os.path.join(reports_dir, "trading_summary.txt")
        stats = history_data['summary_stats']
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"=== TỔNG KẾT GIAO DỊCH ===\n")
            f.write(f"Cập nhật lần cuối: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"📊 Tổng số giao dịch: {stats['total_trades']}\n")
            f.write(f"✅ Giao dịch có lãi: {stats['profitable_trades']}\n")
            f.write(f"❌ Giao dịch thua lỗ: {stats['losing_trades']}\n")
            f.write(f"📈 Tỷ lệ thắng: {stats['win_rate']:.1f}%\n")
            f.write(f"💰 Tổng lãi: {stats['total_profit']:.2f}$\n")
            f.write(f"💸 Tổng lỗ: {stats['total_loss']:.2f}$\n")
            f.write(f"💵 Lãi ròng: {stats['net_profit']:.2f}$\n\n")
            
            f.write("=== THỐNG KÊ THEO SYMBOL ===\n")
            for sym, sym_stats in stats['symbols_traded'].items():
                win_rate = (sym_stats.get('profitable_trades', 0) / sym_stats['trades'] * 100) if sym_stats['trades'] > 0 else 0
                f.write(f"{sym}: {sym_stats['trades']} lệnh, Lãi ròng: {sym_stats['net']:.2f}$\n")
        
        logger.info(f"📋 Đã lưu lịch sử giao dịch: {len(closed_positions)} lệnh đóng vào {history_file}")
        logger.info(f"📈 Tổng kết: {stats['total_trades']} lệnh, Tỷ lệ thắng: {stats['win_rate']:.1f}%, Lãi ròng: {stats['net_profit']:.2f}$")
        
    except Exception as e:
        logger.error(f"❌ Lỗi lưu lịch sử giao dịch: {e}")


def get_mt5_closed_positions(symbol: str = None, days_back: int = 7, auto_trading_safe: bool = True) -> List[Dict]:
    """
    Lấy danh sách các lệnh đã đóng từ MT5 (Tối ưu cho auto trading)
    
    Args:
        symbol: Symbol cụ thể (tùy chọn)
        days_back: Số ngày quay lại để lấy lịch sử  
        auto_trading_safe: Nếu True, sẽ tránh xung đột với auto trading
    
    Returns:
        List các position đã đóng
    """
    try:
        import MetaTrader5 as mt5
        from datetime import datetime, timedelta
        
        # Auto trading safety check
        if auto_trading_safe:
            # Kiểm tra nếu có MT5 connection khác đang chạy
            try:
                if mt5.initialize():
                    account_info = mt5.account_info()
                    if account_info is None:
                        logger.debug("MT5 không sẵn sàng cho history query")
                        mt5.shutdown()
                        return []
                else:
                    logger.debug("Không thể kết nối MT5 cho history query")
                    return []
            except Exception:
                logger.debug("MT5 đang bận, bỏ qua history query")
                return []
        else:
            if not mt5.initialize():
                logger.error("❌ Không thể khởi tạo MT5")
                return []
        
        # Thời gian tìm kiếm (giới hạn để tăng tốc)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=min(days_back, 7))  # Tối đa 7 ngày cho auto trading
        
        # Lấy lịch sử giao dịch với timeout ngắn
        try:
            if symbol:
                history = mt5.history_deals_get(start_time, end_time, group=f"*{symbol}*")
            else:
                history = mt5.history_deals_get(start_time, end_time)
            
            if history is None:
                logger.debug("⚠️ Không có lịch sử giao dịch hoặc timeout")
                mt5.shutdown()
                return []
            
            # Giới hạn số lượng để tránh lag
            max_deals = 100 if auto_trading_safe else 1000
            history = history[-max_deals:] if len(history) > max_deals else history
            
        except Exception as query_error:
            logger.debug(f"Lỗi query MT5 history: {query_error}")
            mt5.shutdown()
            return []
        
        # Chuyển đổi nhanh sang dict và lọc các lệnh đóng
        closed_positions = []
        for deal in history:
            try:
                deal_dict = deal._asdict()
                # Chỉ lấy các deal out (đóng lệnh)
                if deal_dict.get('entry') == 1:  # 1 = DEAL_ENTRY_OUT (đóng lệnh)
                    closed_positions.append(deal_dict)
            except Exception:
                continue  # Bỏ qua deal lỗi
        
        mt5.shutdown()
        
        if not auto_trading_safe:  # Chỉ log khi không phải auto trading mode
            logger.info(f"📊 Tìm thấy {len(closed_positions)} lệnh đã đóng trong {min(days_back, 7)} ngày")
        
        return closed_positions
        
    except Exception as e:
        # Đảm bảo shutdown MT5 kể cả khi có lỗi
        try:
            mt5.shutdown()
        except:
            pass
        
        if not auto_trading_safe:
            logger.error(f"❌ Lỗi lấy lịch sử MT5: {e}")
        else:
            logger.debug(f"MT5 history query failed (auto-trading mode): {e}")
        return []


if __name__ == "__main__":
    sys.exit(main())
