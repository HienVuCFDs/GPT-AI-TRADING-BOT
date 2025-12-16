"""
Trading Bot Constants
=====================
Central configuration for all magic numbers and thresholds.

IMPORTANT: This file contains all hardcoded values used across the trading system.
When you need to adjust trading parameters, modify values here instead of 
searching through multiple files.

Last Updated: 2025-11-27
"""

# =============================================================================
# TECHNICAL INDICATOR THRESHOLDS
# =============================================================================

# Stochastic Oscillator
STOCHASTIC_OVERBOUGHT = 70  # Overbought level (default: 70)
STOCHASTIC_OVERSOLD = 30    # Oversold level (default: 30)
# Rationale: Standard stochastic levels. Values > 70 indicate overbought,
# values < 30 indicate oversold conditions.

# RSI (Relative Strength Index)
RSI_OVERBOUGHT = 70         # RSI overbought threshold
RSI_OVERSOLD = 30           # RSI oversold threshold
# Rationale: Classic RSI levels for momentum reversal signals

# Ultimate Oscillator
UO_OVERBOUGHT = 70          # Ultimate oscillator overbought
UO_OVERSOLD = 30            # Ultimate oscillator oversold

# Williams %R
WILLIAMS_R_OVERBOUGHT = -20  # Williams %R overbought level
WILLIAMS_R_OVERSOLD = -80    # Williams %R oversold level

# =============================================================================
# CONFIDENCE LEVELS
# =============================================================================

# Signal Confidence Thresholds
CONFIDENCE_VERY_HIGH = 80   # Very high confidence signal (>= 80%)
CONFIDENCE_HIGH = 70        # High confidence signal (>= 70%)
CONFIDENCE_MEDIUM = 60      # Medium confidence signal (>= 60%)
CONFIDENCE_LOW = 50         # Low confidence signal (>= 50%)

# Confidence Multipliers for SL/TP
CONFIDENCE_MULTIPLIER_VERY_HIGH = 0.9   # Tighter SL for very high confidence
CONFIDENCE_MULTIPLIER_HIGH = 1.0        # Standard SL for high confidence
CONFIDENCE_MULTIPLIER_MEDIUM = 1.1      # Slightly wider SL for medium confidence
CONFIDENCE_MULTIPLIER_LOW = 1.3         # Much wider SL for low confidence

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

# Spread Management
MAX_SPREAD_MULTIPLIER = 3.0         # Maximum allowed spread (3x normal)
# Rationale: Reject orders if spread exceeds 3x the normal spread
# to avoid excessive slippage in volatile conditions

# Volume Limits
MIN_VOLUME_AUTO = 0.01              # Minimum volume for auto-trading (lots)
MAX_TOTAL_VOLUME = 10.0             # Maximum total volume across all positions (lots)

# Stop Loss / Take Profit Bounds
MIN_SL_PIPS = 30                    # Minimum stop loss distance (pips)
MAX_SL_PIPS = 150                   # Maximum stop loss distance (pips)
# Rationale: Prevent too tight SL (< 30 pips) that gets hit by noise,
# and too wide SL (> 150 pips) that risks too much capital

MIN_TP_PIPS = 40                    # Minimum take profit distance (pips)
MAX_TP_PIPS = 300                   # Maximum take profit distance (pips)

# ATR Multipliers for SL/TP
ATR_SL_MULTIPLIER = 2.5             # Stop loss = ATR x 2.5
ATR_TP_MULTIPLIER = 3.0             # Take profit = ATR x 3.0

# =============================================================================
# DCA (Dollar Cost Averaging)
# =============================================================================

# DCA Distance
DEFAULT_DCA_DISTANCE_PIPS = 50      # Default distance between DCA levels (pips)
DCA_MIN_DISTANCE_PIPS = 20          # Minimum DCA distance (pips)
DCA_MAX_DISTANCE_PIPS = 200         # Maximum DCA distance (pips)

# DCA Levels
DEFAULT_DCA_LEVELS = 4              # Default number of DCA levels
MAX_DCA_LEVELS = 6                  # Maximum DCA levels allowed

# DCA Volume Multiplier
DCA_VOLUME_MULTIPLIER = 1.5         # Each DCA level = previous volume x 1.5

# =============================================================================
# PRICE ACTION & MOVEMENT
# =============================================================================

# Price Movement Thresholds (in pips)
PRICE_MOVEMENT_SIGNIFICANT = 100    # Significant price movement (> 100 pips)
PRICE_MOVEMENT_STABLE = 20          # Stable price (< 20 pips)

# Price Action Multipliers
PRICE_ACTION_STRONG_MOVE = 1.2      # Wider SL during strong moves
PRICE_ACTION_STABLE = 0.9           # Tighter SL when price is stable

# =============================================================================
# SESSION & TIME-BASED MULTIPLIERS
# =============================================================================

# Trading Session Volatility Multipliers
SESSION_LONDON_MULTIPLIER = 1.2     # London session (higher volatility)
SESSION_NEWYORK_MULTIPLIER = 1.3    # New York session (highest volatility)
SESSION_ASIAN_MULTIPLIER = 0.9      # Asian session (lower volatility)

# =============================================================================
# INSTRUMENT-SPECIFIC MULTIPLIERS
# =============================================================================

# Forex Major Pairs
FOREX_MAJOR_MULTIPLIER = 1.0        # EUR/USD, GBP/USD, USD/JPY, etc.

# Forex Minor & Exotic Pairs
FOREX_MINOR_MULTIPLIER = 1.3        # Higher spread, more volatility

# Gold & Precious Metals
GOLD_MULTIPLIER = 1.5               # XAU/USD requires wider SL

# Crypto
CRYPTO_MULTIPLIER = 2.0             # Cryptocurrencies have high volatility

# Indices
INDEX_MULTIPLIER = 1.2              # Stock indices (US30, NAS100, etc.)

# =============================================================================
# LOCK & CLEANUP SETTINGS
# =============================================================================

# DCA Lock Management
DCA_LOCK_MAX_AGE_SECONDS = 300      # Clean up locks older than 5 minutes
ORDER_LOCK_TIMEOUT_SECONDS = 30     # Order placement timeout

# =============================================================================
# CONFIDENCE BOOST CALCULATIONS
# =============================================================================

# Base confidence boost for indicator signals
CONFIDENCE_BOOST_BASE = 8.0         # Base boost value
CONFIDENCE_BOOST_MULTIPLIER = 0.2   # Additional boost per point beyond threshold

# Example: Stochastic at 75 (overbought = 70)
# Boost = 8.0 + (75 - 70) * 0.2 = 9.0

# =============================================================================
# SMART ENTRY ADJUSTMENTS
# =============================================================================

# Entry price adjustments (percentage)
ENTRY_ADJUSTMENT_MIN = 0.0001       # 0.01% minimum adjustment
ENTRY_ADJUSTMENT_MAX = 0.001        # 0.1% maximum adjustment

# =============================================================================
# TIMEFRAME WEIGHTS
# =============================================================================

# Multi-timeframe analysis weights
TIMEFRAME_H4_WEIGHT = 1.5           # H4 timeframe weight
TIMEFRAME_H1_WEIGHT = 1.2           # H1 timeframe weight
TIMEFRAME_M30_WEIGHT = 1.0          # M30 timeframe weight (baseline)
TIMEFRAME_M15_WEIGHT = 0.8          # M15 timeframe weight (less reliable)

# =============================================================================
# SIGNAL MULTIPLIERS
# =============================================================================

# Confidence-based signal strength multipliers
SIGNAL_MULTIPLIER_HIGH = 1.5        # Confidence >= 80%
SIGNAL_MULTIPLIER_MEDIUM_HIGH = 1.2 # Confidence >= 70%
SIGNAL_MULTIPLIER_MEDIUM = 1.0      # Confidence >= 50%
SIGNAL_MULTIPLIER_LOW = 0.8         # Confidence < 50%

# =============================================================================
# NOTES & BEST PRACTICES
# =============================================================================

"""
MODIFICATION GUIDELINES:
1. Always test changes in demo account first
2. Document the reason for changes in git commit
3. Keep backups of previous values in comments
4. Consider market conditions when adjusting thresholds

COMMON ADJUSTMENTS:
- Increase STOCHASTIC_OVERBOUGHT (70→75) for more conservative entries
- Decrease MAX_SPREAD_MULTIPLIER (3.0→2.5) for better execution
- Adjust ATR_SL_MULTIPLIER (2.5→3.0) for wider stop loss in volatile markets
- Modify DCA_DISTANCE_PIPS based on instrument characteristics

BACKTESTING:
When backtesting different strategies, create a copy of this file
(e.g., constants_aggressive.py, constants_conservative.py) and
swap the import in your trading modules.
"""
