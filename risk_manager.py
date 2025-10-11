import logging
import time
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import os
import logging

# Enhanced MT5 import with fallback + unified connection manager
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    print("⚠️ MetaTrader5 not available - using mock data")
    MT5_AVAILABLE = False
try:
    from mt5_connector import MT5ConnectionManager  # singleton
except Exception:  # pragma: no cover
    MT5ConnectionManager = None  # type: ignore

    class MockMT5:
        @staticmethod
        def account_info():
            class MockAccount:
                def _asdict(self):
                    return {
                        'balance': 10000.0,
                        'equity': 10000.0,
                        'margin': 0.0,
                        'free_margin': 10000.0
                    }
            return MockAccount()
        
        @staticmethod
        def symbol_info(symbol):
            class MockSymbol:
                point = 0.00001 if "XAU" not in symbol else 0.01
                spread = 20
                volume_min = 0.01
                volume_max = 100.0
                contract_size = 100000
            return MockSymbol()
            
        @staticmethod
        def positions_get():
            return []
            
        @staticmethod
        def orders_get():
            return []
    mt5 = MockMT5()

# Enhanced logging setup with UTF-8 support
def setup_logging():
    """Setup enhanced logging with UTF-8 encoding"""
    logger = logging.getLogger(__name__)
    
    if not logger.handlers:
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(
            f'logs/risk_manager_{datetime.now().strftime("%Y%m%d")}.log',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    
    return logger

logger = setup_logging()

def parse_setting_value(value, default_value=0.0, setting_name="unknown"):
    """
    🔧 Parse setting value, handle "OFF" option and convert to appropriate type
    
    Args:
        value: The setting value (can be number, "OFF", or other string)
        default_value: Default value if OFF or invalid
        setting_name: Name of setting for logging
    
    Returns:
        Parsed value or None if OFF
    """
    try:
        # Handle "OFF" case - return None to indicate disabled
        if isinstance(value, str) and value.upper() == "OFF":
            logger.info(f"📴 Setting '{setting_name}' is OFF (disabled)")
            return None
            
        # Handle numeric values
        if isinstance(value, (int, float)):
            return float(value)
            
        # Handle string numeric values
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                logger.warning(f"⚠️ Invalid value for '{setting_name}': {value}, using default: {default_value}")
                return default_value
                
        # Fallback to default
        return default_value
        
    except Exception as e:
        logger.error(f"❌ Error parsing '{setting_name}': {e}, using default: {default_value}")
        return default_value

def is_setting_enabled(value):
    """Check if a setting is enabled (not OFF)"""
    if value is None:
        return False
    if isinstance(value, str) and value.upper() == "OFF":
        return False
    return True

class RiskLevel(Enum):
    """Risk level enumeration"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW" 
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

class ValidationResult(Enum):
    """Validation result types"""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    WARNING = "WARNING"
    PENDING = "PENDING"

@dataclass
class AdvancedRiskParameters:
    """Enhanced risk management parameters with advanced controls and auto-adjustment"""
    # Basic risk limits
    max_risk_percent: float = 2.0
    max_drawdown_percent: float = 5.0
    max_daily_loss_percent: float = 3.0
    max_weekly_loss_percent: float = 8.0
    max_monthly_loss_percent: float = 15.0
    
    # SL/TP Settings (from GUI)
    sltp_mode: str = "Hỗ trợ/Kháng cự"
    default_sl_pips: float = 150.0
    default_tp_pips: float = 100.0
    sr_buffer_pips: float = 20.0
    default_sl_atr_multiplier: float = 4.0
    default_tp_atr_multiplier: float = 2.5
    
    # Volume Settings (from GUI)
    volume_mode: str = "Khối lượng mặc định"
    fixed_volume_lots: float = 0.15
    
    # DCA Settings (from GUI)
    enable_dca: bool = True
    max_dca_levels: int = 5
    dca_distance_pips: float = 20.0
    dca_mode: str = "atr_multiple"
    dca_sl_mode: str = "SL riêng lẻ"
    dca_volume_multiplier: float = 1.5
    
    # Trading Mode (from GUI)
    trading_mode: str = "👨‍💼 Thủ công"
    
    # Position limits
    max_positions: int = 5
    max_positions_per_symbol: int = 2
    max_correlation: float = 0.7
    min_risk_reward_ratio: float = 1.5
    
    # Advanced controls
    max_total_volume: float = 10.0
    min_volume_auto: float = 0.01
    max_spread_multiplier: float = 3.0
    max_slippage: int = 10
    
    # Time-based controls
    trading_hours_start: int = 0  # UTC hour
    trading_hours_end: int = 24   # UTC hour
    avoid_news_minutes: int = 30  # Minutes before/after news
    
    # Symbol-specific limits
    symbol_max_exposure: Dict[str, float] = field(default_factory=lambda: {})
    symbol_risk_multipliers: Dict[str, float] = field(default_factory=lambda: {})
    
    # Emergency controls
    emergency_stop_drawdown: float = 10.0
    emergency_stop_daily_loss: float = 5.0
    auto_reduce_on_losses: bool = True
    
    # 🆕 DROPDOWN-BASED RISK CONTROLS WITH AUTO MODE
    news_mode: str = "AVOID"                    # AUTO, AVOID, OFF
    emergency_mode: str = "ENABLED"             # AUTO, ENABLED, OFF  
    max_dd_mode: str = "ENABLED"                # AUTO, ENABLED, OFF
    
    # Legacy toggleable controls (maintained for backward compatibility)
    disable_news_avoidance: bool = False        # Derived from news_mode
    disable_emergency_stop: bool = False        # Derived from emergency_mode
    disable_max_dd_close: bool = False          # Derived from max_dd_mode
    
    # Confidence thresholds
    min_confidence_threshold: float = 3.0
    high_confidence_threshold: float = 4.5
    confidence_position_multiplier: Dict[float, float] = field(default_factory=lambda: {
        3.0: 0.5,  # Low confidence = 50% position size
        4.0: 1.0,  # Medium confidence = 100% position size
        4.5: 1.5   # High confidence = 150% position size
    })
    
    # 🆕 ENHANCED AUTO MODE SETTINGS
    auto_mode_enabled: bool = False
    auto_scan_enabled: bool = True                   # Enable automatic account scanning
    auto_adjustment_interval: int = 24              # Hours between auto adjustments
    last_auto_adjustment: Optional[datetime] = None
    last_account_scan: Optional[datetime] = None    # Track last scan time
    
    # Auto adjustment ranges (min, max)
    auto_risk_percent_range: Tuple[float, float] = (0.5, 5.0)
    auto_drawdown_percent_range: Tuple[float, float] = (3.0, 10.0)
    auto_positions_range: Tuple[int, int] = (2, 10)
    auto_confidence_range: Tuple[float, float] = (2.0, 5.0)
    
    # Account tier thresholds for auto adjustment
    account_tiers: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "micro": {
            "balance_min": 100,
            "balance_max": 1000,
            "default_risk": 1.0,
            "default_drawdown": 3.0,
            "default_positions": 2,
            "risk_profile": "conservative"
        },
        "mini": {
            "balance_min": 1000,
            "balance_max": 10000,
            "default_risk": 2.0,
            "default_drawdown": 5.0,
            "default_positions": 3,
            "risk_profile": "moderate"
        },
        "standard": {
            "balance_min": 10000,
            "balance_max": 50000,
            "default_risk": 2.5,
            "default_drawdown": 6.0,
            "default_positions": 5,
            "risk_profile": "balanced"
        },
        "professional": {
            "balance_min": 50000,
            "balance_max": float('inf'),
            "default_risk": 3.0,
            "default_drawdown": 8.0,
            "default_positions": 8,
            "risk_profile": "aggressive"
        }
    })

@dataclass
class RiskAwareAction:
    """Represent a risk-aware trading action"""
    action_type: str  # 'dca_entry', 'limit_order', 'sl_update', 'tp_update'
    symbol: str
    direction: str  # 'BUY', 'SELL'
    entry_price: float
    volume: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    confidence: float = 0.0
    risk_level: str = "moderate"  # 'low', 'moderate', 'high'
    order_type: str = "limit"  # 'market', 'limit', 'stop'
    conditions: Dict[str, Any] = None
    priority: int = 5  # 1=highest, 10=lowest

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}

@dataclass
class TradeSignal:
    """Enhanced trade signal data structure"""
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
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class RiskMetrics:
    """Risk metrics tracking"""
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    total_exposure: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    active_positions: int = 0
    correlation_risk: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    result: ValidationResult
    signal: TradeSignal
    recommended_volume: float
    risk_score: float
    checks: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class AdvancedRiskManagementSystem:
    """Enhanced Risk Management System with comprehensive controls and analytics"""
    
    def __init__(self, risk_params: Optional[AdvancedRiskParameters] = None):
        self.risk_params = risk_params or AdvancedRiskParameters()
        self.account_info = None
        # Initialize integrated action generator
        self.action_generator = None
        # Unified MT5 connection manager (handles account switch)
        try:
            self.connection_manager = MT5ConnectionManager() if MT5ConnectionManager else None
        except Exception:
            self.connection_manager = None
        # Failure tracking (to suppress log spam)
        self._consecutive_account_failures = 0
        self._warn_every = 3  # only warn every N failures after first
        
        # Performance tracking
        self.daily_trades = deque(maxlen=1000)
        self.pnl_history = deque(maxlen=30)  # 30 days
        self.risk_metrics = RiskMetrics()
        
        # Position tracking
        self.active_positions = []
        self.pending_orders = []
        self.symbol_exposure = defaultdict(float)
        
        # Emergency controls
        self.emergency_stop_active = False
        self.trading_suspended = False
        
        # Cache and optimization
        self._last_account_update = None
        self._account_cache_duration = timedelta(seconds=30)
        
        logger.info(f"✅ AdvancedRiskManagementSystem initialized")
        logger.info(f"📊 Risk Parameters: Max Risk {self.risk_params.max_risk_percent}%, Max DD {self.risk_params.max_drawdown_percent}%")
        logger.info(f"🤖 Auto Mode: {'ENABLED' if self.risk_params.auto_mode_enabled else 'DISABLED'}")
        
        # Load saved parameters first
        self.load_risk_parameters()
        
        # 🆕 Sync legacy toggle settings from mode selections
        self._sync_toggle_settings_from_modes()
        
        self.update_account_info()
        self._load_historical_data()
        
        # Check if auto adjustment is needed
        if self.risk_params.auto_mode_enabled:
            self._check_auto_adjustment()

    def _load_historical_data(self):
        """Load historical P&L and performance data"""
        try:
            # Create risk_management directory if it doesn't exist
            os.makedirs("risk_management", exist_ok=True)
            
            history_file = "risk_management/risk_history.json"
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Load recent PnL history
                for pnl_entry in data.get('pnl_history', [])[-30:]:
                    self.pnl_history.append(pnl_entry)
                    
                logger.info(f"📚 Loaded {len(self.pnl_history)} days of PnL history")
            else:
                logger.info("📚 No historical data found - starting fresh")
                
        except Exception as e:
            logger.error(f"❌ Error loading historical data: {e}")

    def _save_historical_data(self):
        """Save current performance data to disk"""
        try:
            os.makedirs("risk_management", exist_ok=True)
            
            data = {
                'pnl_history': list(self.pnl_history),
                'last_updated': datetime.now().isoformat(),
                'emergency_stop_count': getattr(self, 'emergency_stop_count', 0)
            }
            
            with open("risk_management/risk_history.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            logger.error(f"❌ Error saving historical data: {e}")

    def get_symbol_specific_settings(self, symbol: str) -> Dict[str, Any]:
        """Get symbol-specific risk settings from risk_settings.json"""
        try:
            unified_file = "risk_management/risk_settings.json"
            if os.path.exists(unified_file):
                with open(unified_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                symbol_settings = settings.get('symbol_specific_settings', {}).get(symbol, {})
                
                # Return symbol settings with fallback to defaults
                return {
                    'default_sl_pips': symbol_settings.get('default_sl_pips', 300),
                    'default_tp_pips': symbol_settings.get('default_tp_pips', 600),
                    'dca_distance_pips': symbol_settings.get('dca_distance_pips', 200),
                    'min_risk_reward_ratio': symbol_settings.get('min_risk_reward_ratio', 2.0),
                    'max_spread': symbol_settings.get('max_spread', 20)
                }
        except Exception as e:
            logger.error(f"❌ Error loading symbol settings for {symbol}: {e}")
            
        # Default fallback settings
        return {
            'default_sl_pips': 300,
            'default_tp_pips': 600,
            'dca_distance_pips': 200,
            'min_risk_reward_ratio': 2.0,
            'max_spread': 20
        }

    def calculate_symbol_sl_tp(self, symbol: str, entry_price: float, 
                              trade_type: str, signal_strength: float = 1.0) -> Tuple[float, float]:
        """Calculate symbol-specific SL and TP based on settings"""
        try:
            symbol_settings = self.get_symbol_specific_settings(symbol)
            
            # Get symbol point value
            if MT5_AVAILABLE:
                symbol_info = mt5.symbol_info(symbol)
                point = symbol_info.point if symbol_info else 0.00001
            else:
                point = 0.01 if "XAU" in symbol else 0.00001
            
            # Calculate SL and TP distances
            sl_pips = symbol_settings['default_sl_pips']
            tp_pips = symbol_settings['default_tp_pips']
            
            # Adjust for signal strength
            sl_pips = int(sl_pips * (1.0 / signal_strength))  # Stronger signal = tighter SL
            tp_pips = int(tp_pips * signal_strength)  # Stronger signal = wider TP
            
            # Calculate actual prices
            if trade_type.upper() in ['BUY', 'LONG']:
                stop_loss = entry_price - (sl_pips * point)
                take_profit = entry_price + (tp_pips * point)
            else:  # SELL/SHORT
                stop_loss = entry_price + (sl_pips * point)
                take_profit = entry_price - (tp_pips * point)
            
            # Ensure minimum risk/reward ratio
            min_rr = symbol_settings['min_risk_reward_ratio']
            risk_distance = abs(entry_price - stop_loss)
            reward_distance = abs(take_profit - entry_price)
            
            if reward_distance < (risk_distance * min_rr):
                if trade_type.upper() in ['BUY', 'LONG']:
                    take_profit = entry_price + (risk_distance * min_rr)
                else:
                    take_profit = entry_price - (risk_distance * min_rr)
                
                logger.info(f"📊 Adjusted TP for {symbol} to meet {min_rr}:1 R/R ratio")
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"❌ Error calculating SL/TP for {symbol}: {e}")
            # Fallback calculation
            default_sl_points = 300 * (0.01 if "XAU" in symbol else 0.00001)
            default_tp_points = 600 * (0.01 if "XAU" in symbol else 0.00001)
            
            if trade_type.upper() in ['BUY', 'LONG']:
                return entry_price - default_sl_points, entry_price + default_tp_points
            else:
                return entry_price + default_sl_points, entry_price - default_tp_points

    def _check_auto_adjustment(self):
        """Check if auto adjustment should be performed"""
        if not self.risk_params.auto_mode_enabled:
            return
            
        now = datetime.now()
        
        # Check if enough time has passed since last adjustment
        if (self.risk_params.last_auto_adjustment and 
            now - self.risk_params.last_auto_adjustment < timedelta(hours=self.risk_params.auto_adjustment_interval)):
            return
            
        logger.info("🤖 Performing auto risk parameter adjustment...")
        self._perform_auto_adjustment()
        self.risk_params.last_auto_adjustment = now
        self._save_risk_parameters()

    def _perform_auto_adjustment(self):
        """Automatically adjust risk parameters based on account performance and balance"""
        if not self.account_info:
            logger.warning("⚠️ Cannot perform auto adjustment - no account info")
            return
            
        try:
            # 🆕 Enhanced account scanning if enabled
            if self.risk_params.auto_scan_enabled:
                self._perform_account_scan()
                
            balance = self.account_info['balance']
            equity = self.account_info['equity']
            current_dd = self.risk_metrics.current_drawdown
            
            # Determine account tier
            account_tier = self._determine_account_tier(balance)
            tier_settings = self.risk_params.account_tiers[account_tier]
            
            logger.info(f"🎯 Account Tier: {account_tier.upper()} (Balance: ${balance:,.2f})")
            logger.info(f"📊 Current Performance: DD={current_dd:.2f}%, Equity=${equity:,.2f}")
            
            # Store original parameters for comparison
            original_params = {
                'risk_percent': self.risk_params.max_risk_percent,
                'drawdown_percent': self.risk_params.max_drawdown_percent,
                'max_positions': self.risk_params.max_positions,
                'confidence_threshold': self.risk_params.min_confidence_threshold
            }
            
            # 1. Adjust risk percentage based on account performance
            self._auto_adjust_risk_percentage(tier_settings, current_dd)
            
            # 2. Adjust drawdown limits based on tier
            self._auto_adjust_drawdown_limits(tier_settings, current_dd)
            
            # 3. Adjust position limits based on balance
            self._auto_adjust_position_limits(tier_settings, balance)
            
            # 4. Adjust confidence thresholds based on recent performance
            self._auto_adjust_confidence_settings(tier_settings)
            
            # 5. Adjust emergency stops based on tier
            self._auto_adjust_emergency_settings(tier_settings)
            
            # 6. 🆕 Auto-adjust dropdown-based controls based on account health
            health_score = self._calculate_account_health_score()
            self._auto_adjust_mode_controls(tier_settings, health_score)
            
            # Log adjustments
            adjustments = []
            if original_params['risk_percent'] != self.risk_params.max_risk_percent:
                # Safe format handling for risk percentage
                try:
                    old_risk = float(original_params['risk_percent']) if original_params['risk_percent'] else 0.0
                    new_risk = float(self.risk_params.max_risk_percent) if self.risk_params.max_risk_percent else 0.0
                    adjustments.append(f"Risk: {old_risk:.1f}% → {new_risk:.1f}%")
                except (ValueError, TypeError):
                    adjustments.append(f"Risk: {original_params['risk_percent']} → {self.risk_params.max_risk_percent}")
                    
            if original_params['drawdown_percent'] != self.risk_params.max_drawdown_percent:
                # Safe format handling for drawdown percentage  
                try:
                    old_dd = float(original_params['drawdown_percent']) if original_params['drawdown_percent'] else 0.0
                    new_dd = float(self.risk_params.max_drawdown_percent) if self.risk_params.max_drawdown_percent else 0.0
                    adjustments.append(f"DD: {old_dd:.1f}% → {new_dd:.1f}%")
                except (ValueError, TypeError):
                    adjustments.append(f"DD: {original_params['drawdown_percent']} → {self.risk_params.max_drawdown_percent}")
            if original_params['max_positions'] != self.risk_params.max_positions:
                adjustments.append(f"Positions: {original_params['max_positions']} → {self.risk_params.max_positions}")
            if original_params['confidence_threshold'] != self.risk_params.min_confidence_threshold:
                # Safe format handling for confidence values
                try:
                    old_conf = float(original_params['confidence_threshold']) if original_params['confidence_threshold'] else 0.0
                    new_conf = float(self.risk_params.min_confidence_threshold)
                    adjustments.append(f"Confidence: {old_conf:.1f} → {new_conf:.1f}")
                except (ValueError, TypeError):
                    # Fallback for non-numeric values
                    adjustments.append(f"Confidence: {original_params['confidence_threshold']} → {self.risk_params.min_confidence_threshold}")
                
            if adjustments:
                logger.info(f"🔧 Auto Adjustments Applied: {', '.join(adjustments)}")
            else:
                logger.info("✅ No adjustments needed - parameters optimal")
                
        except Exception as e:
            logger.error(f"❌ Auto adjustment error: {e}")
            
    def _perform_account_scan(self):
        """Enhanced account scanning for automatic parameter optimization"""
        try:
            logger.info("🔍 Performing enhanced account scan...")
            
            # Update last scan time
            self.risk_params.last_account_scan = datetime.now()
            
            if not self.account_info:
                logger.warning("⚠️ No account info available for scan")
                return
                
            # Gather comprehensive account metrics
            balance = self.account_info['balance']
            equity = self.account_info['equity']
            margin_level = self.account_info.get('margin_level', 0)
            free_margin = self.account_info.get('free_margin', 0)
            
            # Calculate account health metrics
            health_score = self._calculate_account_health_score()
            
            # Analyze recent trading performance
            recent_performance = self._analyze_recent_performance()
            
            # Assess market exposure and diversification
            exposure_analysis = self._analyze_portfolio_exposure()
            
            logger.info(f"📊 Account Health Score: {health_score:.2f}/10")
            logger.info(f"🎯 Recent Performance Score: {recent_performance:.2f}/10")
            logger.info(f"📈 Portfolio Exposure Score: {exposure_analysis:.2f}/10")
            
            # Overall account score influences auto adjustment aggressiveness
            overall_score = (health_score + recent_performance + exposure_analysis) / 3
            
            if overall_score < 4.0:
                logger.warning("⚠️ Low account score - conservative adjustments recommended")
                # More conservative parameters
                self.risk_params.max_risk_percent = max(0.5, self.risk_params.max_risk_percent * 0.8)
                self.risk_params.max_positions = max(1, int(self.risk_params.max_positions * 0.7))
            elif overall_score > 7.0:
                logger.info("✅ High account score - optimized adjustments possible")
                # Allow more aggressive parameters within limits
                self.risk_params.max_risk_percent = min(5.0, self.risk_params.max_risk_percent * 1.2)
                self.risk_params.max_positions = min(10, int(self.risk_params.max_positions * 1.3))
                
            logger.info(f"🎯 Overall Account Score: {overall_score:.2f}/10")
            
        except Exception as e:
            logger.error(f"❌ Account scan error: {e}")
            
    def _calculate_account_health_score(self) -> float:
        """Calculate account health score (0-10)"""
        try:
            if not self.account_info:
                return 5.0  # Neutral score
                
            balance = self.account_info['balance']
            equity = self.account_info['equity']
            margin_level = self.account_info.get('margin_level', 1000)
            
            # Health factors
            equity_ratio = equity / balance if balance > 0 else 1.0
            margin_safety = min(10.0, margin_level / 100.0) if margin_level > 0 else 10.0
            drawdown_factor = max(0, 10 - (self.risk_metrics.current_drawdown * 2))
            
            # Weighted score
            health_score = (equity_ratio * 4 + margin_safety * 3 + drawdown_factor * 3) / 10
            return min(10.0, max(0.0, health_score))
            
        except Exception as e:
            logger.error(f"❌ Health score calculation error: {e}")
            return 5.0
            
    def _analyze_recent_performance(self) -> float:
        """Analyze recent trading performance (0-10)"""
        try:
            # Simple performance analysis based on available data
            # This could be enhanced with historical trade data
            
            if len(self.pnl_history) < 5:
                return 5.0  # Neutral if not enough data
                
            recent_pnl = list(self.pnl_history)[-7:]  # Last week
            avg_pnl = sum(recent_pnl) / len(recent_pnl)
            
            # Score based on average PnL and consistency
            if avg_pnl > 0:
                performance_score = min(10.0, 5.0 + (avg_pnl * 10))
            else:
                performance_score = max(0.0, 5.0 + (avg_pnl * 10))
                
            return performance_score
            
        except Exception as e:
            logger.error(f"❌ Performance analysis error: {e}")
            return 5.0
            
    def _analyze_portfolio_exposure(self) -> float:
        """Analyze portfolio exposure and diversification (0-10)"""
        try:
            if not self.active_positions:
                return 10.0  # No exposure = perfect score
                
            # Analyze symbol diversification
            symbols = set(pos.symbol for pos in self.active_positions)
            symbol_count = len(symbols)
            
            # Analyze position sizing distribution
            total_exposure = sum(abs(pos.volume) for pos in self.active_positions)
            avg_position_size = total_exposure / len(self.active_positions) if self.active_positions else 0
            
            # Score based on diversification and reasonable position sizing
            diversification_score = min(10.0, symbol_count * 2)  # More symbols = better
            sizing_score = min(10.0, 10 - (avg_position_size * 2))  # Smaller average = better
            
            exposure_score = (diversification_score + sizing_score) / 2
            return min(10.0, max(0.0, exposure_score))
            
        except Exception as e:
            logger.error(f"❌ Exposure analysis error: {e}")
            return 5.0

    def _determine_account_tier(self, balance: float) -> str:
        """Determine account tier based on balance"""
        for tier, settings in self.risk_params.account_tiers.items():
            if settings['balance_min'] <= balance <= settings['balance_max']:
                return tier
        return 'micro'  # Fallback

    def _auto_adjust_risk_percentage(self, tier_settings: Dict, current_dd: float):
        """Auto adjust risk percentage based on performance and tier"""
        base_risk = tier_settings['default_risk']
        
        # Performance-based adjustment
        if current_dd > 3.0:  # Poor performance - reduce risk
            performance_factor = max(0.5, 1.0 - (current_dd / 10.0))
            adjusted_risk = base_risk * performance_factor
        elif current_dd < 1.0:  # Good performance - can increase slightly
            performance_factor = min(1.3, 1.0 + (1.0 - current_dd) * 0.1)
            adjusted_risk = base_risk * performance_factor
        else:  # Normal performance
            adjusted_risk = base_risk
            
        # Apply range limits
        min_risk, max_risk = self.risk_params.auto_risk_percent_range
        self.risk_params.max_risk_percent = max(min_risk, min(max_risk, adjusted_risk))

    def _auto_adjust_drawdown_limits(self, tier_settings: Dict, current_dd: float):
        """Auto adjust drawdown limits based on tier and current drawdown"""
        base_dd = tier_settings['default_drawdown']
        
        # If currently in drawdown, be more conservative
        if current_dd > 2.0:
            adjustment_factor = 0.8  # Reduce drawdown limit by 20%
        else:
            adjustment_factor = 1.0
            
        adjusted_dd = base_dd * adjustment_factor
        
        # Apply range limits
        min_dd, max_dd = self.risk_params.auto_drawdown_percent_range
        self.risk_params.max_drawdown_percent = max(min_dd, min(max_dd, adjusted_dd))

    def _auto_adjust_position_limits(self, tier_settings: Dict, balance: float):
        """Auto adjust position limits based on balance and tier"""
        base_positions = tier_settings['default_positions']
        
        # Balance-based scaling
        if balance < 1000:
            scale_factor = 0.6  # Very small accounts - fewer positions
        elif balance < 5000:
            scale_factor = 0.8
        elif balance > 50000:
            scale_factor = 1.4  # Large accounts - more positions
        else:
            scale_factor = 1.0
            
        adjusted_positions = int(base_positions * scale_factor)
        
        # Apply range limits
        min_pos, max_pos = self.risk_params.auto_positions_range
        self.risk_params.max_positions = max(min_pos, min(max_pos, adjusted_positions))

    def _auto_adjust_confidence_settings(self, tier_settings: Dict):
        """Auto adjust confidence thresholds based on tier and recent performance"""
        risk_profile = tier_settings['risk_profile']
        
        # Profile-based confidence requirements
        if risk_profile == 'conservative':
            base_confidence = 3.5
        elif risk_profile == 'moderate':
            base_confidence = 3.0
        elif risk_profile == 'balanced':
            base_confidence = 2.8
        else:  # aggressive
            base_confidence = 2.5
            
        # Recent performance adjustment
        if len(self.pnl_history) > 5:
            # Calculate recent win rate (simplified)
            recent_performance = sum(1 for pnl in list(self.pnl_history)[-5:] if pnl > 0) / 5
            if recent_performance < 0.4:  # Poor recent performance
                base_confidence += 0.5  # Require higher confidence
            elif recent_performance > 0.7:  # Good recent performance
                base_confidence -= 0.2  # Can accept lower confidence
                
        # Apply range limits
        min_conf, max_conf = self.risk_params.auto_confidence_range
        self.risk_params.min_confidence_threshold = max(min_conf, min(max_conf, base_confidence))

    def _auto_adjust_emergency_settings(self, tier_settings: Dict):
        """Auto adjust emergency stop settings based on tier"""
        risk_profile = tier_settings['risk_profile']
        
        if risk_profile == 'conservative':
            self.risk_params.emergency_stop_drawdown = self.risk_params.max_drawdown_percent * 1.5
            self.risk_params.emergency_stop_daily_loss = 2.0
        elif risk_profile == 'moderate':
            self.risk_params.emergency_stop_drawdown = self.risk_params.max_drawdown_percent * 1.8
            self.risk_params.emergency_stop_daily_loss = 3.0
        elif risk_profile == 'balanced':
            self.risk_params.emergency_stop_drawdown = self.risk_params.max_drawdown_percent * 2.0
            self.risk_params.emergency_stop_daily_loss = 4.0
        else:  # aggressive
            self.risk_params.emergency_stop_drawdown = self.risk_params.max_drawdown_percent * 2.2
            self.risk_params.emergency_stop_daily_loss = 5.0

    def _auto_adjust_mode_controls(self, tier_settings: Dict, health_score: float):
        """🆕 Auto-adjust dropdown mode controls based on account health and tier"""
        try:
            risk_profile = tier_settings['risk_profile']
            balance = self.account_info.get('balance', 0) if self.account_info else 0
            current_dd = abs(self.risk_metrics.current_drawdown) if self.risk_metrics else 0
            
            # Auto-adjust news mode based on account health and market conditions
            if self.risk_params.news_mode == "AUTO" or "TỰ ĐỘNG" in self.risk_params.news_mode:
                if health_score >= 8.0 and current_dd < 2.0:
                    # Account is very healthy - can trade through minor news
                    self.risk_params.disable_news_avoidance = True
                    logger.info("🤖 AUTO News Mode: OFF (healthy account, low DD)")
                elif health_score >= 6.0 and current_dd < 5.0:
                    # Account is decent - avoid major news only
                    self.risk_params.disable_news_avoidance = False
                    logger.info("🤖 AUTO News Mode: AVOID (moderate account health)")
                else:
                    # Account struggling - strict news avoidance
                    self.risk_params.disable_news_avoidance = False
                    logger.info("🤖 AUTO News Mode: STRICT AVOID (account needs protection)")
            
            # Auto-adjust emergency stop based on account status
            if self.risk_params.emergency_mode == "AUTO" or "TỰ ĐỘNG" in self.risk_params.emergency_mode:
                if health_score >= 7.5 and balance >= 50000 and current_dd < 3.0:
                    # Professional account with good health - can relax emergency stops
                    self.risk_params.disable_emergency_stop = True
                    logger.info("🤖 AUTO Emergency Mode: OFF (professional account, strong health)")
                elif health_score >= 6.0 and current_dd < 5.0:
                    # Good account - normal emergency protection
                    self.risk_params.disable_emergency_stop = False
                    logger.info("🤖 AUTO Emergency Mode: ENABLED (normal protection)")
                else:
                    # Struggling account - strict emergency protection
                    self.risk_params.disable_emergency_stop = False
                    logger.info("🤖 AUTO Emergency Mode: STRICT (account needs emergency protection)")
            
            # Auto-adjust max DD close based on account profile
            if self.risk_params.max_dd_mode == "AUTO" or "TỰ ĐỘNG" in self.risk_params.max_dd_mode:
                if health_score >= 8.0 and balance >= 100000 and risk_profile == 'aggressive':
                    # Large aggressive account - can handle higher DD
                    self.risk_params.disable_max_dd_close = True
                    logger.info("🤖 AUTO Max DD Mode: OFF (large aggressive account)")
                elif health_score >= 6.5 and current_dd < 3.0:
                    # Healthy account - normal DD protection
                    self.risk_params.disable_max_dd_close = False
                    logger.info("🤖 AUTO Max DD Mode: ENABLED (standard protection)")
                else:
                    # Needs protection - strict DD limits
                    self.risk_params.disable_max_dd_close = False
                    logger.info("🤖 AUTO Max DD Mode: STRICT (enhanced protection)")
                    
            logger.info(f"🎯 Auto Mode Controls Updated: News={not self.risk_params.disable_news_avoidance}, Emergency={not self.risk_params.disable_emergency_stop}, MaxDD={not self.risk_params.disable_max_dd_close}")
            
        except Exception as e:
            logger.error(f"❌ Error auto-adjusting mode controls: {e}")
            # Fallback to safe defaults
            self.risk_params.disable_news_avoidance = False
            self.risk_params.disable_emergency_stop = False  
            self.risk_params.disable_max_dd_close = False

    def _sync_toggle_settings_from_modes(self):
        """🆕 Sync legacy toggle settings from dropdown mode selections"""
        try:
            # Sync news avoidance setting
            if hasattr(self.risk_params, 'news_mode'):
                if "OFF" in self.risk_params.news_mode or "TẮT" in self.risk_params.news_mode:
                    self.risk_params.disable_news_avoidance = True
                elif "AUTO" in self.risk_params.news_mode or "TỰ ĐỘNG" in self.risk_params.news_mode:
                    # AUTO mode will be handled by auto adjustment
                    pass
                else:
                    self.risk_params.disable_news_avoidance = False
            
            # Sync emergency stop setting
            if hasattr(self.risk_params, 'emergency_mode'):
                if "OFF" in self.risk_params.emergency_mode or "TẮT" in self.risk_params.emergency_mode:
                    self.risk_params.disable_emergency_stop = True
                elif "AUTO" in self.risk_params.emergency_mode or "TỰ ĐỘNG" in self.risk_params.emergency_mode:
                    # AUTO mode will be handled by auto adjustment
                    pass
                else:
                    self.risk_params.disable_emergency_stop = False
            
            # Sync max DD close setting
            if hasattr(self.risk_params, 'max_dd_mode'):
                if "OFF" in self.risk_params.max_dd_mode or "TẮT" in self.risk_params.max_dd_mode:
                    self.risk_params.disable_max_dd_close = True
                elif "AUTO" in self.risk_params.max_dd_mode or "TỰ ĐỘNG" in self.risk_params.max_dd_mode:
                    # AUTO mode will be handled by auto adjustment
                    pass
                else:
                    self.risk_params.disable_max_dd_close = False
                    
            logger.info(f"🔄 Mode Settings Synced: News={self.risk_params.news_mode}, Emergency={self.risk_params.emergency_mode}, MaxDD={self.risk_params.max_dd_mode}")
            
        except Exception as e:
            logger.error(f"❌ Error syncing toggle settings: {e}")

    def _save_risk_parameters(self):
        """Save current risk parameters to file with comprehensive data"""
        try:
            os.makedirs("risk_management", exist_ok=True)
            
            # Load existing settings to preserve OFF values (null)
            existing_settings = self._load_existing_settings()
            
            # Convert dataclass to comprehensive dict for JSON serialization
            params_dict = {
                # Basic risk limits - preserve null if already set
                'max_risk_percent': self.risk_params.max_risk_percent,
                'max_drawdown_percent': self.risk_params.max_drawdown_percent,
                'max_daily_loss_percent': existing_settings.get('max_daily_loss_percent', self.risk_params.max_daily_loss_percent),
                'max_weekly_loss_percent': existing_settings.get('max_weekly_loss_percent', self.risk_params.max_weekly_loss_percent),
                'max_monthly_loss_percent': existing_settings.get('max_monthly_loss_percent', self.risk_params.max_monthly_loss_percent),
                
                # Position limits
                'max_positions': self.risk_params.max_positions,
                'max_positions_per_symbol': self.risk_params.max_positions_per_symbol,
                'max_correlation': self.risk_params.max_correlation,
                'min_risk_reward_ratio': self.risk_params.min_risk_reward_ratio,
                
                # Advanced controls
                'max_total_volume': self.risk_params.max_total_volume,
                'min_volume_auto': self.risk_params.min_volume_auto,
                'max_spread_multiplier': self.risk_params.max_spread_multiplier,
                'max_slippage': self.risk_params.max_slippage,
                
                # SL/TP Settings
                'default_sl_pips': self.risk_params.default_sl_pips,
                'default_tp_pips': self.risk_params.default_tp_pips,
                'default_sl_atr_multiplier': self.risk_params.default_sl_atr_multiplier,
                'default_tp_atr_multiplier': self.risk_params.default_tp_atr_multiplier,
                'sltp_mode': self.risk_params.sltp_mode,
                
                # DCA Settings
                'enable_dca': self.risk_params.enable_dca,
                'max_dca_levels': self.risk_params.max_dca_levels,
                'dca_distance_pips': self.risk_params.dca_distance_pips,
                'dca_mode': self.risk_params.dca_mode,
                'dca_mode_legacy': getattr(self.risk_params, 'dca_mode_legacy', 'Bội số ATR'),
                'dca_sl_mode': self.risk_params.dca_sl_mode,
                'dca_volume_multiplier': self.risk_params.dca_volume_multiplier,
                'dca_atr_period': getattr(self.risk_params, 'dca_atr_period', 14),
                'dca_atr_multiplier': getattr(self.risk_params, 'dca_atr_multiplier', 1.5),
                
                # Time-based controls
                'trading_hours_start': self.risk_params.trading_hours_start,
                'trading_hours_end': self.risk_params.trading_hours_end,
                'avoid_news_minutes': self.risk_params.avoid_news_minutes,
                
                # Symbol-specific limits
                'symbol_max_exposure': dict(self.risk_params.symbol_max_exposure),
                'symbol_risk_multipliers': dict(self.risk_params.symbol_risk_multipliers),
                
                # Emergency controls - preserve null if already set
                'emergency_stop_drawdown': self.risk_params.emergency_stop_drawdown,
                'emergency_stop_daily_loss': existing_settings.get('emergency_stop_daily_loss', self.risk_params.emergency_stop_daily_loss),
                'auto_reduce_on_losses': self.risk_params.auto_reduce_on_losses,
                
                # Confidence thresholds - preserve null if already set  
                'min_confidence_threshold': existing_settings.get('min_confidence_threshold', self.risk_params.min_confidence_threshold),
                'high_confidence_threshold': existing_settings.get('high_confidence_threshold', self.risk_params.high_confidence_threshold),
                'confidence_position_multiplier': dict(self.risk_params.confidence_position_multiplier),
                
                # Auto mode settings
                'auto_mode_enabled': self.risk_params.auto_mode_enabled,
                'auto_adjustment_interval': self.risk_params.auto_adjustment_interval,
                'last_auto_adjustment': self.risk_params.last_auto_adjustment.isoformat() if self.risk_params.last_auto_adjustment else None,
                
                # 🆕 Dropdown mode settings
                'news_mode': getattr(self.risk_params, 'news_mode', 'AVOID'),
                'emergency_mode': getattr(self.risk_params, 'emergency_mode', 'ENABLED'),
                'max_dd_mode': getattr(self.risk_params, 'max_dd_mode', 'ENABLED'),
                
                # Legacy toggle settings (for backward compatibility)
                'disable_news_avoidance': self.risk_params.disable_news_avoidance,
                'disable_emergency_stop': self.risk_params.disable_emergency_stop,
                'disable_max_dd_close': self.risk_params.disable_max_dd_close,
                
                # Auto adjustment ranges
                'auto_risk_percent_range': self.risk_params.auto_risk_percent_range,
                'auto_drawdown_percent_range': self.risk_params.auto_drawdown_percent_range,
                'auto_positions_range': self.risk_params.auto_positions_range,
                'auto_confidence_range': self.risk_params.auto_confidence_range,
                
                # Account tiers
                'account_tiers': dict(self.risk_params.account_tiers),
                
                # Metadata
                'saved_timestamp': datetime.now().isoformat(),
                'version': '2.0',
                'system_info': {
                    'emergency_stop_active': self.emergency_stop_active,
                    'trading_suspended': self.trading_suspended,
                    'current_risk_level': self.risk_metrics.risk_level.value if hasattr(self.risk_metrics, 'risk_level') else 'UNKNOWN'
                }
            }
            
            # Save to unified risk_settings.json file
            self._save_to_unified_file(params_dict)
                
            logger.info("💾 Comprehensive risk parameters saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving risk parameters: {e}")

    def _load_existing_settings(self) -> Dict[str, Any]:
        """Load existing settings to preserve OFF values (null)"""
        try:
            unified_file = "risk_management/risk_settings.json"
            if os.path.exists(unified_file):
                with open(unified_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                    
                # Return only the OFF settings to preserve
                off_settings = {}
                preserve_keys = [
                    'max_daily_loss_percent',
                    'max_weekly_loss_percent', 
                    'max_monthly_loss_percent',
                    'emergency_stop_daily_loss',
                    'min_confidence_threshold',
                    'high_confidence_threshold',
                    'avoid_news_minutes',
                    'max_total_volume',
                    'min_risk_reward_ratio'
                ]
                
                for key in preserve_keys:
                    if key in existing and existing[key] is None:
                        off_settings[key] = None
                        
                return off_settings
        except Exception as e:
            logger.error(f"❌ Error loading existing settings: {e}")
            
        return {}

    def update_gui_settings(self, gui_settings: Dict[str, Any]) -> bool:
        """
        🎛️ Update risk settings from GUI controls
        
        Args:
            gui_settings: Dictionary containing all GUI settings from risk tab
            
        Returns:
            bool: Success status
        """
        try:
            logger.info("🎛️ Updating risk settings from GUI...")
            
            # Update basic risk parameters
            if 'max_risk_percent' in gui_settings:
                self.risk_params.max_risk_percent = float(gui_settings['max_risk_percent']) if gui_settings['max_risk_percent'] != "OFF" else 0.0
                
            if 'max_drawdown_percent' in gui_settings:
                self.risk_params.max_drawdown_percent = float(gui_settings['max_drawdown_percent']) if gui_settings['max_drawdown_percent'] != "OFF" else 0.0
                
            if 'max_daily_loss_percent' in gui_settings:
                self.risk_params.max_daily_loss_percent = parse_setting_value(gui_settings['max_daily_loss_percent'], 3.0, "max_daily_loss_percent")
                
            # Update position limits
            if 'max_positions' in gui_settings:
                self.risk_params.max_positions = int(gui_settings['max_positions'])
                
            if 'max_positions_per_symbol' in gui_settings:
                self.risk_params.max_positions_per_symbol = int(gui_settings['max_positions_per_symbol'])
                
            # Update SL/TP settings
            if 'sltp_mode' in gui_settings:
                self.risk_params.sltp_mode = gui_settings['sltp_mode']
                
            if 'default_sl_pips' in gui_settings:
                self.risk_params.default_sl_pips = float(gui_settings['default_sl_pips'])
                
            if 'default_tp_pips' in gui_settings:
                self.risk_params.default_tp_pips = float(gui_settings['default_tp_pips'])
                
            if 'sr_buffer_pips' in gui_settings:
                self.risk_params.sr_buffer_pips = float(gui_settings['sr_buffer_pips'])
                
            # Update DCA settings
            if 'enable_dca' in gui_settings:
                self.risk_params.enable_dca = bool(gui_settings['enable_dca'])
                
            if 'max_dca_levels' in gui_settings:
                self.risk_params.max_dca_levels = int(gui_settings['max_dca_levels'])
                
            if 'dca_distance_pips' in gui_settings:
                self.risk_params.dca_distance_pips = float(gui_settings['dca_distance_pips'])
                
            if 'dca_mode' in gui_settings:
                self.risk_params.dca_mode = gui_settings['dca_mode']
                
            if 'dca_sl_mode' in gui_settings:
                self.risk_params.dca_sl_mode = gui_settings['dca_sl_mode']
                
            # Update volume settings
            if 'volume_mode' in gui_settings:
                self.risk_params.volume_mode = gui_settings['volume_mode']
                
            if 'fixed_volume_lots' in gui_settings:
                self.risk_params.fixed_volume_lots = float(gui_settings['fixed_volume_lots'])
                
            # Update dropdown mode controls
            if 'news_mode' in gui_settings:
                self.risk_params.news_mode = gui_settings['news_mode']
                
            if 'emergency_mode' in gui_settings:
                self.risk_params.emergency_mode = gui_settings['emergency_mode']
                
            if 'max_dd_mode' in gui_settings:
                self.risk_params.max_dd_mode = gui_settings['max_dd_mode']
                
            # Update auto mode settings
            if 'auto_mode_enabled' in gui_settings:
                self.risk_params.auto_mode_enabled = bool(gui_settings['auto_mode_enabled'])
                
            # Sync toggle settings from modes
            self._sync_toggle_settings_from_modes()
            
            # Save to file
            self._save_risk_parameters()
            
            logger.info("✅ GUI settings updated and saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating GUI settings: {e}")
            return False
    
    def get_gui_settings(self) -> Dict[str, Any]:
        """
        🎛️ Get current risk settings for GUI display
        
        Returns:
            Dict containing all settings formatted for GUI controls
        """
        try:
            return {
                # Basic risk parameters
                'max_risk_percent': self.risk_params.max_risk_percent,
                'max_drawdown_percent': self.risk_params.max_drawdown_percent,
                'max_daily_loss_percent': self.risk_params.max_daily_loss_percent,
                'max_weekly_loss_percent': getattr(self.risk_params, 'max_weekly_loss_percent', 8.0),
                'max_monthly_loss_percent': getattr(self.risk_params, 'max_monthly_loss_percent', 15.0),
                
                # Position limits
                'max_positions': self.risk_params.max_positions,
                'max_positions_per_symbol': self.risk_params.max_positions_per_symbol,
                'max_correlation': self.risk_params.max_correlation,
                'min_risk_reward_ratio': self.risk_params.min_risk_reward_ratio,
                
                # Volume settings
                'max_total_volume': self.risk_params.max_total_volume,
                'min_volume_auto': self.risk_params.min_volume_auto,
                'volume_mode': getattr(self.risk_params, 'volume_mode', 'Khối lượng mặc định'),
                'fixed_volume_lots': getattr(self.risk_params, 'fixed_volume_lots', 0.15),
                
                # SL/TP settings
                'sltp_mode': getattr(self.risk_params, 'sltp_mode', 'Hỗ trợ/Kháng cự'),
                'default_sl_pips': getattr(self.risk_params, 'default_sl_pips', 150),
                'default_tp_pips': getattr(self.risk_params, 'default_tp_pips', 100),
                'sr_buffer_pips': getattr(self.risk_params, 'sr_buffer_pips', 20.0),
                'default_sl_atr_multiplier': getattr(self.risk_params, 'default_sl_atr_multiplier', 4.0),
                'default_tp_atr_multiplier': getattr(self.risk_params, 'default_tp_atr_multiplier', 2.5),
                
                # DCA settings
                'enable_dca': getattr(self.risk_params, 'enable_dca', True),
                'max_dca_levels': getattr(self.risk_params, 'max_dca_levels', 5),
                'dca_distance_pips': getattr(self.risk_params, 'dca_distance_pips', 20.0),
                'dca_mode': getattr(self.risk_params, 'dca_mode', 'atr_multiple'),
                'dca_sl_mode': getattr(self.risk_params, 'dca_sl_mode', 'SL riêng lẻ'),
                'dca_volume_multiplier': getattr(self.risk_params, 'dca_volume_multiplier', 1.5),
                
                # Trading controls
                'trading_hours_start': self.risk_params.trading_hours_start,
                'trading_hours_end': self.risk_params.trading_hours_end,
                'avoid_news_minutes': self.risk_params.avoid_news_minutes,
                'max_spread_multiplier': self.risk_params.max_spread_multiplier,
                'max_slippage': self.risk_params.max_slippage,
                
                # Dropdown mode controls
                'news_mode': getattr(self.risk_params, 'news_mode', 'AVOID'),
                'emergency_mode': getattr(self.risk_params, 'emergency_mode', 'ENABLED'),
                'max_dd_mode': getattr(self.risk_params, 'max_dd_mode', 'ENABLED'),
                
                # Auto mode settings
                'auto_mode_enabled': self.risk_params.auto_mode_enabled,
                'auto_adjustment_interval': self.risk_params.auto_adjustment_interval,
                
                # Trading mode
                'trading_mode': getattr(self.risk_params, 'trading_mode', '👨‍💼 Thủ công'),
                
                # Symbol exposure
                'symbol_exposure': dict(getattr(self.risk_params, 'symbol_exposure', {})),
                'symbol_multipliers': dict(getattr(self.risk_params, 'symbol_multipliers', {})),
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting GUI settings: {e}")
            return {}
    
    def validate_gui_settings(self, gui_settings: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        🔍 Validate GUI settings before applying
        
        Args:
            gui_settings: GUI settings to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Validate risk percentages
            if 'max_risk_percent' in gui_settings:
                risk_pct = gui_settings['max_risk_percent']
                if risk_pct != "OFF" and (float(risk_pct) < 0.1 or float(risk_pct) > 10.0):
                    errors.append("Max risk percent must be between 0.1% and 10.0%")
                    
            if 'max_drawdown_percent' in gui_settings:
                dd_pct = gui_settings['max_drawdown_percent']
                if dd_pct != "OFF" and (float(dd_pct) < 1.0 or float(dd_pct) > 20.0):
                    errors.append("Max drawdown percent must be between 1.0% and 20.0%")
                    
            # Validate position limits
            if 'max_positions' in gui_settings:
                max_pos = int(gui_settings['max_positions'])
                if max_pos < 1 or max_pos > 100:
                    errors.append("Max positions must be between 1 and 100")
                    
            if 'max_positions_per_symbol' in gui_settings:
                max_pos_sym = int(gui_settings['max_positions_per_symbol'])
                if max_pos_sym < 1 or max_pos_sym > 20:
                    errors.append("Max positions per symbol must be between 1 and 20")
                    
            # Validate DCA settings
            if 'max_dca_levels' in gui_settings:
                max_dca = int(gui_settings['max_dca_levels'])
                if max_dca < 1 or max_dca > 10:
                    errors.append("Max DCA levels must be between 1 and 10")
                    
            if 'dca_distance_pips' in gui_settings:
                dca_dist = float(gui_settings['dca_distance_pips'])
                if dca_dist < 1.0 or dca_dist > 1000.0:
                    errors.append("DCA distance must be between 1.0 and 1000.0 pips")
                    
            # Validate volume settings
            if 'fixed_volume_lots' in gui_settings:
                vol = float(gui_settings['fixed_volume_lots'])
                if vol < 0.01 or vol > 100.0:
                    errors.append("Fixed volume must be between 0.01 and 100.0 lots")
                    
            # Validate SL/TP settings
            if 'default_sl_pips' in gui_settings:
                sl_pips = float(gui_settings['default_sl_pips'])
                if sl_pips < 5.0 or sl_pips > 2000.0:
                    errors.append("Default SL pips must be between 5.0 and 2000.0")
                    
            if 'default_tp_pips' in gui_settings:
                tp_pips = float(gui_settings['default_tp_pips'])
                if tp_pips < 5.0 or tp_pips > 2000.0:
                    errors.append("Default TP pips must be between 5.0 and 2000.0")
                    
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors

    def apply_gui_settings_immediately(self, gui_settings: Dict[str, Any]) -> bool:
        """
        🚀 Apply GUI settings immediately without validation (for real-time updates)
        
        Args:
            gui_settings: GUI settings to apply
            
        Returns:
            bool: Success status
        """
        try:
            logger.info("🚀 Applying GUI settings immediately...")
            
            # Add any missing attributes to risk_params if needed
            for key, value in gui_settings.items():
                if hasattr(self.risk_params, key):
                    setattr(self.risk_params, key, value)
                else:
                    # Add as new attribute for dynamic settings
                    setattr(self.risk_params, key, value)
                    logger.debug(f"Added new attribute: {key} = {value}")
            
            # Sync toggle settings from modes
            self._sync_toggle_settings_from_modes()
            
            # Save to file immediately
            self._save_risk_parameters()
            
            logger.info("✅ GUI settings applied immediately")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error applying GUI settings immediately: {e}")
            return False
    
    def get_risk_tab_status(self) -> Dict[str, Any]:
        """
        📊 Get comprehensive status for Risk Management tab display
        
        Returns:
            Dict containing formatted status for GUI display
        """
        try:
            status = {
                'connection_status': 'Connected' if self.account_info else 'Disconnected',
                'emergency_stop': self.emergency_stop_active,
                'trading_suspended': self.trading_suspended,
                'auto_mode': self.risk_params.auto_mode_enabled,
                'current_risk_level': self.risk_metrics.risk_level.value if hasattr(self.risk_metrics, 'risk_level') else 'UNKNOWN',
                
                # Account metrics
                'account_balance': self.account_info.get('balance', 0) if self.account_info else 0,
                'account_equity': self.account_info.get('equity', 0) if self.account_info else 0,
                'current_drawdown': abs(self.risk_metrics.current_drawdown) if self.risk_metrics else 0,
                'free_margin': self.account_info.get('free_margin', 0) if self.account_info else 0,
                'margin_level': self.account_info.get('margin_level', 0) if self.account_info else 0,
                
                # Position metrics  
                'active_positions': len(self.active_positions),
                'pending_orders': len(self.pending_orders),
                'total_exposure': sum(abs(getattr(pos, 'volume', 0)) for pos in self.active_positions),
                
                # Risk settings summary
                'max_risk_percent': self.risk_params.max_risk_percent,
                'max_drawdown_percent': self.risk_params.max_drawdown_percent,
                'max_positions': self.risk_params.max_positions,
                'dca_enabled': getattr(self.risk_params, 'enable_dca', True),
                'sltp_mode': getattr(self.risk_params, 'sltp_mode', 'Hỗ trợ/Kháng cự'),
                
                # Mode controls
                'news_mode': getattr(self.risk_params, 'news_mode', 'AVOID'),
                'emergency_mode': getattr(self.risk_params, 'emergency_mode', 'ENABLED'), 
                'max_dd_mode': getattr(self.risk_params, 'max_dd_mode', 'ENABLED'),
                
                'last_updated': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting risk tab status: {e}")
            return {'error': str(e)}

    def _save_to_unified_file(self, params_dict: Dict[str, Any]):
        """Save parameters to unified risk_settings.json file"""
        try:
            unified_file = "risk_management/risk_settings.json"
            
            # Load existing settings or create new
            if os.path.exists(unified_file):
                with open(unified_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
            else:
                settings = {}
            
            # Update the main settings directly (flatten structure)
            # Handle nested dicts carefully
            for key, value in params_dict.items():
                settings[key] = value
            
            # Preserve reports section if it exists
            if 'reports' not in settings:
                settings['reports'] = {
                    'risk_reports': [],
                    'execution_reports': [],
                    'test_reports': [],
                    'max_stored_reports': 100,
                    'last_cleanup': None
                }
            
            # Create directory if needed
            os.makedirs(os.path.dirname(unified_file), exist_ok=True)
            
            # Save to file
            with open(unified_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"❌ Error saving to unified file: {e}")

    def _save_emergency_event_to_unified_file(self, event: Dict[str, Any]):
        """Save emergency event to unified risk_settings.json file"""
        try:
            unified_file = "risk_management/risk_settings.json"
            
            # Load existing settings
            if os.path.exists(unified_file):
                with open(unified_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
            else:
                settings = {}
            
            # Add event to emergency_events list
            if 'emergency_events' not in settings:
                settings['emergency_events'] = []
            settings['emergency_events'].append(event)
            
            # Preserve reports section if it exists
            if 'reports' not in settings:
                settings['reports'] = {
                    'risk_reports': [],
                    'execution_reports': [],
                    'test_reports': [],
                    'max_stored_reports': 100,
                    'last_cleanup': None
                }
            
            # Update timestamp
            settings['metadata']['last_updated'] = datetime.now().isoformat()
            
            # Save to file
            with open(unified_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            logger.error(f"❌ Error saving emergency event to unified file: {e}")

    def _calculate_pip_value(self, symbol: str, contract_size: float, point: float, entry_price: float) -> float:
        """Calculate pip value correctly for different asset types"""
        try:
            # Clean symbol name
            clean_symbol = symbol.replace('_m', '').replace('.', '').upper()
            
            # *** IMPORTANT: Check metals FIRST before forex ***
            # 1. PRECIOUS METALS
            if any(metal in clean_symbol for metal in ['XAU', 'XAG', 'GOLD', 'SILVER']):
                if 'XAU' in clean_symbol:  # Gold
                    # XAUUSD: 1 pip = 0.1, contract = 100 oz
                    # pip_value = 100 oz * $0.1 = $10 per pip
                    return 100 * 0.1  # $10 per pip for 1 lot
                elif 'XAG' in clean_symbol:  # Silver
                    # XAGUSD: 1 pip = 0.001, contract = 5000 oz
                    # pip_value = 5000 oz * $0.001 = $5 per pip
                    return 5000 * 0.001  # $5 per pip for 1 lot
                else:
                    # Generic metals: use point * contract_size / 10 (assume 10 points = 1 pip)
                    return (contract_size * point) / 10
            
            # 2. CRYPTOCURRENCIES  
            elif any(crypto in clean_symbol for crypto in ['BTC', 'ETH', 'LTC', 'XRP', 'ADA', 'SOL', 'DOGE']):
                if any(high_crypto in clean_symbol for high_crypto in ['BTC', 'ETH', 'SOL']):
                    # BTC/ETH: 1 pip = $1 USD (e.g., ETH 4000.00 -> 4070.00 = 70 pips = $70)
                    # SOL: 1 pip = $0.1 USD (e.g., SOL 220.0 -> 220.7 = 7 pips = $0.7)
                    return 1.0  # $1 per pip for high-value crypto
                else:
                    # Other cryptos: 1 pip = $0.01 typically
                    return 0.01
            
            # 3. STOCK INDICES
            elif any(index in clean_symbol for index in ['US30', 'US500', 'NAS100', 'GER30', 'UK100', 'JP225']):
                if 'US30' in clean_symbol:  # Dow Jones
                    return 1.0  # $1 per point
                elif 'US500' in clean_symbol:  # S&P 500
                    return 1.0  # $1 per point
                elif 'NAS100' in clean_symbol:  # Nasdaq
                    return 1.0  # $1 per point
                else:
                    return 1.0  # Default $1 per point for indices
            
            # 4. COMMODITIES
            elif any(commodity in clean_symbol for commodity in ['WTI', 'BRENT', 'NGAS', 'WHEAT', 'CORN']):
                if 'WTI' in clean_symbol or 'BRENT' in clean_symbol:  # Oil
                    return 10.0  # $10 per pip for oil
                elif 'NGAS' in clean_symbol:  # Natural Gas
                    return 10.0  # $10 per pip
                else:
                    return 1.0  # Default for other commodities
            
            # 5. FOREX PAIRS (Check AFTER metals to avoid USD conflict)
            elif any(fx in clean_symbol for fx in ['USD', 'EUR', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']) and not any(metal in clean_symbol for metal in ['XAU', 'XAG']):
                if 'JPY' in clean_symbol:
                    # JPY pairs: 1 pip = 0.01, pip value varies with price
                    # For XXXJPY: pip_value = (contract_size * 0.01) / current_price
                    return (contract_size * 0.01) / entry_price
                else:
                    # Major pairs: 1 pip = 0.0001, fixed pip value
                    # For XXXUSD: pip_value = contract_size * 0.0001 = $10 for 1 lot
                    return contract_size * 0.0001
            
            # 6. STOCKS
            elif len(clean_symbol) <= 5 and not any(char.isdigit() for char in clean_symbol):
                # Individual stocks: pip value = point value
                return 1.0  # $1 per point typically
            
            # 7. DEFAULT FALLBACK
            else:
                # Use original calculation as fallback
                if 'JPY' in clean_symbol:
                    return (contract_size * point) / entry_price
                else:
                    return contract_size * point
                    
        except Exception as e:
            logger.error(f"❌ Error calculating pip value for {symbol}: {e}")
            # Fallback to original calculation
            if 'JPY' in symbol:
                return (contract_size * point) / entry_price
            else:
                return contract_size * point

    def _get_default_unified_settings(self) -> Dict[str, Any]:
        """Get default structure for unified settings file"""
        return {
            "basic_settings": {},
            "dca_settings": {},
            "mode_settings": {},
            "auto_risk_params": {},
            "session_data": {},
            "emergency_events": [],
            "symbol_exposure": {},
            "symbol_multipliers": {},
            "advanced_settings": {},
            "metadata": {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "unified_file": True
            }
        }

    def load_risk_parameters(self) -> bool:
        """Load saved risk parameters from unified risk_settings.json file"""
        try:
            # Load from unified file
            unified_file = "risk_management/risk_settings.json"
            
            if not os.path.exists(unified_file):
                logger.info("📚 No saved risk parameters found - using defaults")
                return False
                
            with open(unified_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            # Load SL/TP settings from main level (saved by app.py)
            if 'default_sl_pips' in settings:
                self.risk_params.default_sl_pips = float(settings['default_sl_pips'])
            if 'default_tp_pips' in settings:
                self.risk_params.default_tp_pips = float(settings['default_tp_pips'])
            if 'default_sl_atr_multiplier' in settings:
                self.risk_params.default_sl_atr_multiplier = float(settings['default_sl_atr_multiplier'])
            if 'default_tp_atr_multiplier' in settings:
                self.risk_params.default_tp_atr_multiplier = float(settings['default_tp_atr_multiplier'])
            if 'sltp_mode' in settings:
                self.risk_params.sltp_mode = settings['sltp_mode']
                
            # Extract basic_settings section and merge with root settings
            params_dict = settings.get('basic_settings', {})
            # Merge root level settings for compatibility
            root_keys = ['enable_auto_mode', 'trading_mode', 'fixed_volume_lots', 'max_positions', 'enable_dca', 'max_dca_levels', 'dca_volume_multiplier']
            for key in root_keys:
                if key in settings and key not in params_dict:
                    params_dict[key] = settings[key]
                
            # Update basic risk parameters with OFF support
            self.risk_params.max_risk_percent = parse_setting_value(
                params_dict.get('max_risk_percent', self.risk_params.max_risk_percent), 
                self.risk_params.max_risk_percent, 'max_risk_percent'
            ) or self.risk_params.max_risk_percent
            
            self.risk_params.max_drawdown_percent = parse_setting_value(
                params_dict.get('max_drawdown_percent', self.risk_params.max_drawdown_percent), 
                self.risk_params.max_drawdown_percent, 'max_drawdown_percent'
            )
            
            self.risk_params.max_daily_loss_percent = parse_setting_value(
                params_dict.get('max_daily_loss_percent', self.risk_params.max_daily_loss_percent), 
                self.risk_params.max_daily_loss_percent, 'max_daily_loss_percent'
            )
            
            self.risk_params.max_weekly_loss_percent = parse_setting_value(
                params_dict.get('max_weekly_loss_percent', self.risk_params.max_weekly_loss_percent), 
                self.risk_params.max_weekly_loss_percent, 'max_weekly_loss_percent'
            )
            
            self.risk_params.max_monthly_loss_percent = parse_setting_value(
                params_dict.get('max_monthly_loss_percent', self.risk_params.max_monthly_loss_percent), 
                self.risk_params.max_monthly_loss_percent, 'max_monthly_loss_percent'
            )
            
            # Update position limits with OFF support
            self.risk_params.max_positions = int(parse_setting_value(
                params_dict.get('max_positions', self.risk_params.max_positions), 
                self.risk_params.max_positions, 'max_positions'
            ) or self.risk_params.max_positions)
            
            self.risk_params.max_positions_per_symbol = int(parse_setting_value(
                params_dict.get('max_positions_per_symbol', self.risk_params.max_positions_per_symbol), 
                self.risk_params.max_positions_per_symbol, 'max_positions_per_symbol'
            ) or self.risk_params.max_positions_per_symbol)
            
            self.risk_params.max_correlation = parse_setting_value(
                params_dict.get('max_correlation', self.risk_params.max_correlation), 
                self.risk_params.max_correlation, 'max_correlation'
            )
            
            self.risk_params.min_risk_reward_ratio = parse_setting_value(
                params_dict.get('min_risk_reward_ratio', self.risk_params.min_risk_reward_ratio), 
                self.risk_params.min_risk_reward_ratio, 'min_risk_reward_ratio'
            )
            
            # Update advanced controls with OFF support
            self.risk_params.max_total_volume = parse_setting_value(
                params_dict.get('max_total_volume', self.risk_params.max_total_volume), 
                self.risk_params.max_total_volume, 'max_total_volume'
            )
            
            self.risk_params.min_volume_auto = parse_setting_value(
                params_dict.get('min_volume_auto', self.risk_params.min_volume_auto), 
                self.risk_params.min_volume_auto, 'min_volume_auto'
            ) or self.risk_params.min_volume_auto
            
            # Backward compatibility for old field names
            if 'max_lot_size' in params_dict and 'max_total_volume' not in params_dict:
                self.risk_params.max_total_volume = params_dict.get('max_lot_size', self.risk_params.max_total_volume)
                logger.info("🔄 Migrated max_lot_size to max_total_volume")
            if 'min_lot_size' in params_dict and 'min_volume_auto' not in params_dict:
                self.risk_params.min_volume_auto = params_dict.get('min_lot_size', self.risk_params.min_volume_auto)
                logger.info("🔄 Migrated min_lot_size to min_volume_auto")
            
            self.risk_params.max_spread_multiplier = parse_setting_value(
                params_dict.get('max_spread_multiplier', self.risk_params.max_spread_multiplier), 
                self.risk_params.max_spread_multiplier, 'max_spread_multiplier'
            )
            
            self.risk_params.max_slippage = parse_setting_value(
                params_dict.get('max_slippage', self.risk_params.max_slippage), 
                self.risk_params.max_slippage, 'max_slippage'
            )
            
            # Update time-based controls with OFF support
            self.risk_params.trading_hours_start = int(parse_setting_value(
                params_dict.get('trading_hours_start', self.risk_params.trading_hours_start), 
                self.risk_params.trading_hours_start, 'trading_hours_start'
            ) or self.risk_params.trading_hours_start)
            
            self.risk_params.trading_hours_end = int(parse_setting_value(
                params_dict.get('trading_hours_end', self.risk_params.trading_hours_end), 
                self.risk_params.trading_hours_end, 'trading_hours_end'
            ) or self.risk_params.trading_hours_end)
            
            self.risk_params.avoid_news_minutes = parse_setting_value(
                params_dict.get('avoid_news_minutes', self.risk_params.avoid_news_minutes), 
                self.risk_params.avoid_news_minutes, 'avoid_news_minutes'
            )
            
            # Update symbol-specific limits
            if 'symbol_max_exposure' in params_dict:
                self.risk_params.symbol_max_exposure.update(params_dict['symbol_max_exposure'])
            if 'symbol_risk_multipliers' in params_dict:
                self.risk_params.symbol_risk_multipliers.update(params_dict['symbol_risk_multipliers'])
            
            # Update emergency controls with OFF support
            self.risk_params.emergency_stop_drawdown = parse_setting_value(
                params_dict.get('emergency_stop_drawdown', self.risk_params.emergency_stop_drawdown), 
                self.risk_params.emergency_stop_drawdown, 'emergency_stop_drawdown'
            )
            
            self.risk_params.emergency_stop_daily_loss = parse_setting_value(
                params_dict.get('emergency_stop_daily_loss', self.risk_params.emergency_stop_daily_loss), 
                self.risk_params.emergency_stop_daily_loss, 'emergency_stop_daily_loss'
            ) or self.risk_params.emergency_stop_daily_loss
            self.risk_params.auto_reduce_on_losses = params_dict.get('auto_reduce_on_losses', self.risk_params.auto_reduce_on_losses)
            
            # Update confidence thresholds
            self.risk_params.min_confidence_threshold = params_dict.get('min_confidence_threshold', self.risk_params.min_confidence_threshold)
            self.risk_params.high_confidence_threshold = params_dict.get('high_confidence_threshold', self.risk_params.high_confidence_threshold)
            if 'confidence_position_multiplier' in params_dict:
                self.risk_params.confidence_position_multiplier.update(params_dict['confidence_position_multiplier'])
            
            # Update auto mode settings
            self.risk_params.auto_mode_enabled = params_dict.get('auto_mode_enabled', params_dict.get('enable_auto_mode', self.risk_params.auto_mode_enabled))
            self.risk_params.auto_adjustment_interval = params_dict.get('auto_adjustment_interval', self.risk_params.auto_adjustment_interval)
            
            # Parse last adjustment time
            if params_dict.get('last_auto_adjustment'):
                self.risk_params.last_auto_adjustment = datetime.fromisoformat(params_dict['last_auto_adjustment'])
            
            # Update auto ranges if available
            if 'auto_risk_percent_range' in params_dict:
                self.risk_params.auto_risk_percent_range = tuple(params_dict['auto_risk_percent_range'])
            if 'auto_drawdown_percent_range' in params_dict:
                self.risk_params.auto_drawdown_percent_range = tuple(params_dict['auto_drawdown_percent_range'])
            if 'auto_positions_range' in params_dict:
                self.risk_params.auto_positions_range = tuple(params_dict['auto_positions_range'])
            if 'auto_confidence_range' in params_dict:
                self.risk_params.auto_confidence_range = tuple(params_dict['auto_confidence_range'])
            
            # Update account tiers if available
            if 'account_tiers' in params_dict:
                self.risk_params.account_tiers.update(params_dict['account_tiers'])
            
            # 🆕 Load dropdown mode settings
            self.risk_params.news_mode = params_dict.get('news_mode', getattr(self.risk_params, 'news_mode', 'AVOID'))
            self.risk_params.emergency_mode = params_dict.get('emergency_mode', getattr(self.risk_params, 'emergency_mode', 'ENABLED'))
            self.risk_params.max_dd_mode = params_dict.get('max_dd_mode', getattr(self.risk_params, 'max_dd_mode', 'ENABLED'))
            
            # Load legacy toggle settings (for backward compatibility)
            self.risk_params.disable_news_avoidance = params_dict.get('disable_news_avoidance', False)
            self.risk_params.disable_emergency_stop = params_dict.get('disable_emergency_stop', False)
            self.risk_params.disable_max_dd_close = params_dict.get('disable_max_dd_close', False)
                
            logger.info(f"📚 Risk parameters loaded successfully from {os.path.basename(unified_file)}")
            logger.info(f"🤖 Auto Mode: {'ENABLED' if self.risk_params.auto_mode_enabled else 'DISABLED'}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading risk parameters: {e}")
            return False

    def toggle_auto_mode(self, enabled: bool = None) -> bool:
        """Toggle auto mode on/off"""
        if enabled is None:
            enabled = not self.risk_params.auto_mode_enabled
            
        old_state = self.risk_params.auto_mode_enabled
        self.risk_params.auto_mode_enabled = enabled
        
        if enabled and not old_state:
            logger.info("🤖 AUTO MODE ENABLED - Risk parameters will be automatically adjusted")
            # Perform immediate adjustment if account info is available
            if self.account_info:
                self._perform_auto_adjustment()
                self.risk_params.last_auto_adjustment = datetime.now()
        elif not enabled and old_state:
            logger.info("🤖 AUTO MODE DISABLED - Using manual risk parameters")
            
        # Save the new state
        self._save_risk_parameters()
        
        return self.risk_params.auto_mode_enabled

    def get_auto_status(self) -> Dict[str, Any]:
        """Get comprehensive auto mode status with toggleable features"""
        status = {
            'auto_mode_enabled': self.risk_params.auto_mode_enabled,
            'auto_scan_enabled': self.risk_params.auto_scan_enabled,
            'last_adjustment': self.risk_params.last_auto_adjustment.isoformat() if self.risk_params.last_auto_adjustment else None,
            'last_account_scan': self.risk_params.last_account_scan.isoformat() if self.risk_params.last_account_scan else None,
            'adjustment_interval_hours': self.risk_params.auto_adjustment_interval,
            'account_tier': self._determine_account_tier(self.account_info['balance']) if self.account_info else 'unknown',
            'toggleable_features': {
                'news_avoidance': not self.risk_params.disable_news_avoidance,
                'emergency_stop': not self.risk_params.disable_emergency_stop,
                'max_dd_close': not self.risk_params.disable_max_dd_close
            },
            'current_parameters': {
                'risk_percent': self.risk_params.max_risk_percent,
                'drawdown_percent': self.risk_params.max_drawdown_percent,
                'max_positions': self.risk_params.max_positions,
                'min_confidence': self.risk_params.min_confidence_threshold,
                'emergency_drawdown': self.risk_params.emergency_stop_drawdown
            }
        }
        
        # Calculate time until next adjustment
        if self.risk_params.auto_mode_enabled and self.risk_params.last_auto_adjustment:
            next_adjustment = self.risk_params.last_auto_adjustment + timedelta(hours=self.risk_params.auto_adjustment_interval)
            time_until_next = next_adjustment - datetime.now()
            status['next_adjustment_in_hours'] = max(0, time_until_next.total_seconds() / 3600)
        else:
            status['next_adjustment_in_hours'] = None
            
        return status

    def update_account_info(self, force: bool = False) -> bool:
        """Update account information with caching, connection manager support & spam suppression.

        force: bypass cache.
        """
        now = datetime.now()
        if not force and self._last_account_update and (now - self._last_account_update < self._account_cache_duration):
            return self.account_info is not None

        try:
            # Prefer unified manager if available
            if self.connection_manager:
                if not self.connection_manager.is_connected():
                    # Attempt silent reconnect (no force to avoid long backoff delay here)
                    self.connection_manager.connect(force_reconnect=False)
                if self.connection_manager.is_connected():
                    # Pull minimal essential info
                    essential = self.connection_manager.get_essential_account_info()
                    if isinstance(essential, dict) and 'error' not in essential:
                        # Normalize keys to match legacy account_info schema subset
                        self.account_info = {
                            'login': essential.get('login'),
                            'server': essential.get('server'),
                            'balance': essential.get('balance'),
                            'equity': essential.get('equity'),
                            'margin': essential.get('margin'),
                            'free_margin': essential.get('free_margin'),
                            'profit': essential.get('profit'),
                        }
                        self._last_account_update = now
                        self._consecutive_account_failures = 0
                        self._update_positions_and_orders()
                        self._update_risk_metrics()
                        if self.risk_params.auto_mode_enabled:
                            self._check_auto_adjustment()
                        logger.debug("✅ Account info (manager) updated")
                        return True
                    else:
                        self._consecutive_account_failures += 1
                        if self._consecutive_account_failures == 1 or (self._consecutive_account_failures % self._warn_every) == 0:
                            logger.warning("⚠️ Failed to retrieve account info (manager path)")
                        # If MT5 manager says connected but returns error repeatedly, trigger periodic forced reconnect
                        if self._consecutive_account_failures % (self._warn_every * 3) == 0:
                            try:
                                logger.warning("🔄 Forcing MT5 reconnect due to repeated logical connection without data (%d attempts)", self._consecutive_account_failures)
                                self.connection_manager.connect(force_reconnect=True)
                            except Exception:
                                pass
                        return False

            # Fallback direct MT5 path
            if MT5_AVAILABLE:
                acct = mt5.account_info()
                if acct:
                    self.account_info = acct._asdict()
                    self._last_account_update = now
                    self._consecutive_account_failures = 0
                    self._update_positions_and_orders()
                    self._update_risk_metrics()
                    if self.risk_params.auto_mode_enabled:
                        self._check_auto_adjustment()
                    logger.debug("✅ Account info updated (direct)")
                    return True
                else:
                    self._consecutive_account_failures += 1
                    if self._consecutive_account_failures == 1 or (self._consecutive_account_failures % self._warn_every) == 0:
                        # Provide richer diagnostics (connection state, last MT5 error if available)
                        try:
                            conn_state = None
                            last_err = None
                            if self.connection_manager:
                                conn_state = getattr(self.connection_manager, 'state', None)
                                last_err = getattr(self.connection_manager, 'get_last_mt5_error', lambda: None)()
                            raw_last_error = None
                            try:
                                raw_last_error = mt5.last_error() if hasattr(mt5, 'last_error') else None
                            except Exception:
                                raw_last_error = None
                            logger.warning(
                                "⚠️ Failed to retrieve account info from MT5 (attempt=%d, conn_state=%s, mgr_last_err=%s, mt5_last_err=%s)",
                                self._consecutive_account_failures,
                                getattr(conn_state, 'value', conn_state),
                                last_err,
                                raw_last_error
                            )
                            # Detect specific 'No IPC connection' error and attempt staged recovery
                            try:
                                if isinstance(raw_last_error, tuple) and len(raw_last_error) >= 2:
                                    code, msg = raw_last_error[0], str(raw_last_error[1]).lower()
                                else:
                                    code, msg = None, str(raw_last_error).lower() if raw_last_error else ''
                                if (code == -10004 or 'no ipc connection' in msg):
                                    # Backoff tiers
                                    if self._consecutive_account_failures in (5,10,20):
                                        logger.warning("🔌 MT5 appears stale (No IPC). Forcing reconnect attempt #%d", self._consecutive_account_failures)
                                        try:
                                            if self.connection_manager:
                                                self.connection_manager.connect(force_reconnect=True)
                                        except Exception:
                                            pass
                                    if self._consecutive_account_failures % 15 == 0:
                                        # Last resort: shutdown + slight sleep + reconnect
                                        logger.warning("♻️ Performing deep reconnect cycle for MT5 after %d failed attempts", self._consecutive_account_failures)
                                        try:
                                            import MetaTrader5 as _mt5
                                            try:
                                                _mt5.shutdown()
                                            except Exception:
                                                pass
                                            time.sleep(2)
                                            if self.connection_manager:
                                                self.connection_manager.connect(force_reconnect=True)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                        except Exception:
                            logger.warning("⚠️ Failed to retrieve account info from MT5 (attempt=%d)", self._consecutive_account_failures)
                    return False
            else:
                # Mock mode
                self.account_info = {
                    'balance': 10000.0,
                    'equity': 10000.0,
                    'margin': 0.0,
                    'free_margin': 10000.0
                }
                self._last_account_update = now
                self._consecutive_account_failures = 0
                logger.debug("✅ Using mock account info")
                return True
        except Exception as e:
            self._consecutive_account_failures += 1
            logger.error(f"❌ Account info update error: {e}")
            self.account_info = None
            return False

    def refresh_after_account_switch(self):
        """Call after credentials changed to clear cache and force fresh pull."""
        self._last_account_update = None
        self.account_info = None
        self._consecutive_account_failures = 0
        return self.update_account_info(force=True)

    def _update_positions_and_orders(self):
        """Update current positions and pending orders"""
        try:
            if MT5_AVAILABLE:
                # Get positions
                positions = mt5.positions_get()
                self.active_positions = list(positions) if positions else []
                
                # Get pending orders  
                orders = mt5.orders_get()
                self.pending_orders = list(orders) if orders else []
                
                # Update symbol exposure
                self.symbol_exposure.clear()
                for pos in self.active_positions:
                    self.symbol_exposure[pos.symbol] += pos.volume
                    
                logger.debug(f"📊 Updated: {len(self.active_positions)} positions, {len(self.pending_orders)} orders")
                
            else:
                # Mock data
                self.active_positions = []
                self.pending_orders = []
                
        except Exception as e:
            logger.error(f"❌ Error updating positions/orders: {e}")

    def _update_risk_metrics(self):
        """Update comprehensive risk metrics"""
        if not self.account_info:
            return
            
        try:
            balance = self.account_info['balance']
            equity = self.account_info['equity']
            
            # Update drawdown
            if balance > 0:
                self.risk_metrics.current_drawdown = ((balance - equity) / balance) * 100
            
            # Update position count
            self.risk_metrics.active_positions = len(self.active_positions)
            
            # Calculate total exposure
            total_exposure = 0.0
            for pos in self.active_positions:
                if hasattr(pos, 'volume') and hasattr(pos, 'price_current'):
                    total_exposure += pos.volume * pos.price_current
            self.risk_metrics.total_exposure = total_exposure
            
            # Determine risk level
            dd = self.risk_metrics.current_drawdown
            # Safeguard: Only evaluate thresholds that are enabled (numeric) to avoid None/"OFF" comparisons
            em_stop = getattr(self.risk_params, 'emergency_stop_drawdown', None)
            max_dd = getattr(self.risk_params, 'max_drawdown_percent', None)
            def _is_number(v):
                return isinstance(v, (int, float)) and not isinstance(v, bool)

            if _is_number(em_stop) and dd >= em_stop:
                self.risk_metrics.risk_level = RiskLevel.VERY_HIGH
            elif _is_number(max_dd) and dd is not None and _is_number(max_dd) and dd >= max_dd:
                self.risk_metrics.risk_level = RiskLevel.HIGH
            elif _is_number(max_dd) and dd is not None and _is_number(max_dd) and dd >= (max_dd * 0.7):
                self.risk_metrics.risk_level = RiskLevel.MEDIUM
            else:
                self.risk_metrics.risk_level = RiskLevel.LOW
                
            self.risk_metrics.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Error updating risk metrics: {e}")

    def calculate_optimal_position_size(self, signal: TradeSignal) -> Tuple[float, Dict[str, Any]]:
        """Calculate optimal position size with advanced risk considerations"""
        if not self.account_info:
            logger.warning("⚠️ No account info available for position size calculation")
            return 0.0, {"error": "No account info"}
        
        try:
            calculation_details = {
                "symbol": signal.symbol,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "confidence": signal.confidence
            }
            
            # Base risk amount
            base_risk_amount = self.account_info['balance'] * (self.risk_params.max_risk_percent / 100)
            calculation_details["base_risk_amount"] = base_risk_amount
            
            # Get symbol information
            if MT5_AVAILABLE:
                symbol_info = mt5.symbol_info(signal.symbol)
                if not symbol_info:
                    logger.error(f"❌ Symbol {signal.symbol} not found")
                    return 0.0, {"error": "Symbol not found"}
                    
                point = symbol_info.point
                # Fix: use trade_contract_size instead of contract_size
                contract_size = getattr(symbol_info, 'trade_contract_size', 100000)
                min_lot = symbol_info.volume_min
                max_lot = symbol_info.volume_max
            else:
                # Mock values
                point = 0.01 if "XAU" in signal.symbol else 0.00001
                contract_size = 100000
                min_lot = 0.01
                max_lot = 100.0
            
            calculation_details.update({
                "point": point,
                "contract_size": contract_size,
                "min_lot": min_lot,
                "max_lot": max_lot
            })
            
            # Calculate risk in points
            risk_points = abs(signal.entry_price - signal.stop_loss) / point
            if risk_points == 0:
                logger.warning("⚠️ Stop loss equals entry price")
                return 0.0, {"error": "Invalid stop loss"}
            
            calculation_details["risk_points"] = risk_points
            
            # Calculate pip value with proper logic for different asset types
            pip_value = self._calculate_pip_value(signal.symbol, contract_size, point, signal.entry_price)
            
            # Adjust for confidence level
            confidence_multiplier = 1.0
            for conf_threshold, multiplier in self.risk_params.confidence_position_multiplier.items():
                if signal.confidence >= conf_threshold:
                    confidence_multiplier = multiplier
            
            calculation_details["confidence_multiplier"] = confidence_multiplier
            
            # Calculate base lot size
            base_lot_size = base_risk_amount / (risk_points * pip_value)
            adjusted_lot_size = base_lot_size * confidence_multiplier
            
            calculation_details.update({
                "base_lot_size": base_lot_size,
                "adjusted_lot_size": adjusted_lot_size
            })
            
            # Apply symbol-specific limits
            symbol_multiplier = self.risk_params.symbol_risk_multipliers.get(signal.symbol, 1.0)
            adjusted_lot_size *= symbol_multiplier
            
            # Apply position limits
            current_exposure = self.symbol_exposure.get(signal.symbol, 0.0)
            max_symbol_exposure = self.risk_params.symbol_max_exposure.get(signal.symbol, float('inf'))
            
            if current_exposure + adjusted_lot_size > max_symbol_exposure:
                adjusted_lot_size = max(0, max_symbol_exposure - current_exposure)
                calculation_details["symbol_limit_applied"] = True
            
            # Apply global limits - check if max_total_volume is OFF
            max_volume_limit = float('inf')
            if is_setting_enabled(self.risk_params.max_total_volume):
                max_volume_limit = self.risk_params.max_total_volume
                logger.info(f"📊 Max volume limit: {max_volume_limit}")
            else:
                logger.info("📴 Max volume limit is OFF")
            
            final_lot_size = max(min_lot, min(adjusted_lot_size, max_lot, max_volume_limit))
            
            # Round to valid increment
            if hasattr(symbol_info, 'volume_step') and symbol_info.volume_step > 0:
                step = symbol_info.volume_step
                final_lot_size = round(final_lot_size / step) * step
            else:
                final_lot_size = round(final_lot_size, 2)
            
            calculation_details["final_lot_size"] = final_lot_size
            
            # Validate minimum size
            if final_lot_size < min_lot:
                logger.warning(f"⚠️ Calculated lot size {final_lot_size} below minimum {min_lot}")
                return 0.0, calculation_details
            
            logger.info(f"💰 Position size calculated: {final_lot_size} lots for {signal.symbol}")
            logger.debug(f"📊 Risk: ${base_risk_amount:.2f}, Points: {risk_points:.1f}, Confidence: {signal.confidence}")
            
            return final_lot_size, calculation_details
            
        except Exception as e:
            logger.error(f"❌ Position size calculation error: {e}")
            return 0.0, {"error": str(e)}

    def check_trading_conditions(self, signal: TradeSignal) -> Tuple[bool, List[str]]:
        """Comprehensive trading conditions check with OFF settings support"""
        conditions = []
        warnings = []
        
        try:
            # 1. Market hours check
            current_hour = datetime.now().hour
            if not (self.risk_params.trading_hours_start <= current_hour <= self.risk_params.trading_hours_end):
                conditions.append(f"Outside trading hours ({self.risk_params.trading_hours_start}-{self.risk_params.trading_hours_end})")
            
            # 2. Emergency stop check (toggleable)
            if not self.risk_params.disable_emergency_stop and self.emergency_stop_active:
                conditions.append("Emergency stop is active")
            elif self.risk_params.disable_emergency_stop and self.emergency_stop_active:
                warnings.append("Emergency stop is active but disabled by settings")
            
            # 3. Trading suspension check
            if self.trading_suspended:
                conditions.append("Trading is suspended")
            
            # 4. Drawdown check (toggleable max DD close) - skip if OFF / None
            if is_setting_enabled(self.risk_params.max_drawdown_percent):
                try:
                    cur_dd = float(self.risk_metrics.current_drawdown)
                    max_dd_val = float(self.risk_params.max_drawdown_percent)
                    if not self.risk_params.disable_max_dd_close and cur_dd >= max_dd_val:
                        conditions.append(f"Drawdown limit exceeded: {cur_dd:.2f}%")
                    elif self.risk_params.disable_max_dd_close and cur_dd >= max_dd_val:
                        warnings.append(f"Drawdown limit exceeded but auto-close disabled: {cur_dd:.2f}%")
                except (TypeError, ValueError):
                    logger.warning("⚠️ Skipping drawdown comparison due to invalid values")
            else:
                logger.debug("📴 Max drawdown check is OFF")
            
            # 5. Position limits check
            if len(self.active_positions) >= self.risk_params.max_positions:
                conditions.append(f"Maximum positions reached: {len(self.active_positions)}")
            
            # 6. Symbol-specific position check
            symbol_positions = sum(1 for pos in self.active_positions if pos.symbol == signal.symbol)
            if symbol_positions >= self.risk_params.max_positions_per_symbol:
                conditions.append(f"Maximum positions for {signal.symbol} reached: {symbol_positions}")
            
            # 7. Market conditions check (including news avoidance if enabled)
            market_ok, market_warnings = self._check_market_conditions(signal.symbol)
            if not market_ok:
                conditions.append("Poor market conditions")
            warnings.extend(market_warnings)
            
            # 8. Risk level check
            if self.risk_metrics.risk_level == RiskLevel.VERY_HIGH:
                conditions.append("Risk level is VERY HIGH")
            elif self.risk_metrics.risk_level == RiskLevel.HIGH:
                warnings.append("Risk level is HIGH - proceed with caution")
            
            return len(conditions) == 0, conditions + warnings
            
        except Exception as e:
            logger.error(f"❌ Trading conditions check error: {e}")
            return False, [f"Conditions check error: {str(e)}"]

    def _check_market_conditions(self, symbol: str) -> Tuple[bool, List[str]]:
        """Check symbol-specific market conditions with toggleable news avoidance"""
        warnings = []
        
        try:
            # 🆕 News avoidance check (toggleable)
            if not self.risk_params.disable_news_avoidance:
                # TODO: Implement news check logic
                # For now, just add a placeholder check
                current_time = datetime.now()
                # Simulate news avoidance - this should be replaced with actual news data
                warnings.append("News avoidance active - monitoring for major events")
            else:
                warnings.append("News avoidance disabled - trading during news events allowed")
                
            if MT5_AVAILABLE:
                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    return False, ["Symbol not found"]
                
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    return False, ["No tick data available"]
                
                # Spread check
                spread = (tick.ask - tick.bid) / symbol_info.point
                max_allowed_spread = symbol_info.spread * self.risk_params.max_spread_multiplier
                
                if spread > max_allowed_spread:
                    warnings.append(f"High spread: {spread:.1f} points (max: {max_allowed_spread:.1f})")
                
                # Volume check
                if symbol_info.volume_min > self.risk_params.max_total_volume:
                    return False, ["Minimum volume too high"]
                
                logger.debug(f"📊 Market conditions for {symbol}: Spread={spread:.1f}, Volume OK")
                return True, warnings
            else:
                # Mock conditions
                return True, warnings
                
        except Exception as e:
            logger.error(f"❌ Market conditions check error for {symbol}: {e}")
            return False, [f"Market check error: {str(e)}"]
            return False, [f"Market check error: {str(e)}"]

    def validate_signal_comprehensive(self, signal: TradeSignal) -> ValidationReport:
        """Comprehensive signal validation with detailed reporting"""
        logger.info(f"🔍 Comprehensive validation for {signal.symbol} {signal.action}")
        
        report = ValidationReport(
            result=ValidationResult.PENDING,
            signal=signal,
            recommended_volume=0.0,
            risk_score=0.0
        )
        
        try:
            # Update account info if needed
            if not self.update_account_info():
                report.result = ValidationResult.REJECTED
                report.errors.append("Cannot update account information")
                return report
            
            # Check trading conditions
            conditions_ok, condition_messages = self.check_trading_conditions(signal)
            report.checks["trading_conditions"] = conditions_ok
            
            if not conditions_ok:
                report.result = ValidationResult.REJECTED
                report.errors.extend([msg for msg in condition_messages if "limit" in msg.lower() or "stop" in msg.lower()])
                report.warnings.extend([msg for msg in condition_messages if msg not in report.errors])
                return report
            
            # Calculate position size
            position_size, calc_details = self.calculate_optimal_position_size(signal)
            report.recommended_volume = position_size
            report.metrics.update(calc_details)
            
            if position_size == 0:
                report.result = ValidationResult.REJECTED
                report.errors.append("Cannot calculate valid position size")
                return report
            
            report.checks["position_size"] = True
            
            # Risk/Reward ratio check
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            report.metrics["risk_reward_ratio"] = rr_ratio
            # Risk/Reward threshold may be disabled (OFF/None)
            if is_setting_enabled(self.risk_params.min_risk_reward_ratio):
                try:
                    min_rr = float(self.risk_params.min_risk_reward_ratio)
                    report.checks["risk_reward"] = rr_ratio >= min_rr
                    if rr_ratio < min_rr:
                        report.warnings.append(f"Poor R:R ratio: {rr_ratio:.2f} (min: {min_rr})")
                except (TypeError, ValueError):
                    report.checks["risk_reward"] = True  # Don't fail if invalid threshold
                    logger.warning("⚠️ Invalid min_risk_reward_ratio value; skipping R:R enforcement")
            else:
                report.checks["risk_reward"] = True  # Disabled
            
            # Confidence check
            if hasattr(self.risk_params, 'min_confidence_threshold') and is_setting_enabled(self.risk_params.min_confidence_threshold):
                try:
                    min_conf = float(self.risk_params.min_confidence_threshold)
                    confidence_ok = signal.confidence >= min_conf
                    report.checks["confidence"] = confidence_ok
                    report.metrics["confidence"] = signal.confidence
                    if not confidence_ok:
                        report.warnings.append(f"Low confidence: {signal.confidence} (min: {min_conf})")
                except (TypeError, ValueError):
                    report.checks["confidence"] = True
                    logger.warning("⚠️ Invalid min_confidence_threshold; skipping confidence check")
            else:
                report.checks["confidence"] = True
                report.metrics["confidence"] = signal.confidence
            
            # Calculate risk score (0-100)
            risk_score = self._calculate_risk_score(signal, position_size)
            report.risk_score = risk_score
            report.metrics["risk_score"] = risk_score
            
            # Final decision
            if len(report.errors) > 0:
                report.result = ValidationResult.REJECTED
            elif risk_score > 80 or len(report.warnings) > 3:
                report.result = ValidationResult.WARNING
            else:
                report.result = ValidationResult.APPROVED
            
            # Log result
            result_emoji = {
                ValidationResult.APPROVED: "✅",
                ValidationResult.WARNING: "⚠️", 
                ValidationResult.REJECTED: "❌",
                ValidationResult.PENDING: "⏳"
            }
            
            logger.info(f"{result_emoji[report.result]} Validation result: {report.result.value}")
            logger.info(f"📊 Risk Score: {risk_score:.1f}/100, Position: {position_size} lots")
            
            if report.warnings:
                logger.warning(f"⚠️ Warnings: {'; '.join(report.warnings)}")
            if report.errors:
                logger.error(f"❌ Errors: {'; '.join(report.errors)}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Comprehensive validation error: {e}")
            report.result = ValidationResult.REJECTED
            report.errors.append(f"Validation error: {str(e)}")
            return report

    def _calculate_risk_score(self, signal: TradeSignal, position_size: float) -> float:
        """Calculate comprehensive risk score (0-100)"""
        try:
            score = 0.0
            
            # Drawdown impact (0-25 points)
            dd_impact = min(25, (self.risk_metrics.current_drawdown / self.risk_params.max_drawdown_percent) * 25)
            score += dd_impact
            
            # Position count impact (0-15 points)
            pos_impact = min(15, (len(self.active_positions) / self.risk_params.max_positions) * 15)
            score += pos_impact
            
            # Confidence impact (0-20 points, inverted)
            conf_impact = max(0, 20 - (signal.confidence / 5.0) * 20)
            score += conf_impact
            
            # Spread/market impact (0-15 points)
            if MT5_AVAILABLE:
                symbol_info = mt5.symbol_info(signal.symbol)
                tick = mt5.symbol_info_tick(signal.symbol)
                if symbol_info and tick:
                    spread = (tick.ask - tick.bid) / symbol_info.point
                    spread_impact = min(15, (spread / (symbol_info.spread * 3)) * 15)
                    score += spread_impact
            
            # Time-based impact (0-10 points)
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 22:  # Off-hours trading
                score += 10
            elif current_hour < 8 or current_hour > 20:  # Less optimal hours
                score += 5
            
            # Volatility impact (0-15 points)
            risk_points = abs(signal.entry_price - signal.stop_loss) / (signal.entry_price * 0.001)  # As percentage
            if risk_points > 2.0:  # High volatility
                score += 15
            elif risk_points > 1.0:
                score += 8
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.error(f"❌ Risk score calculation error: {e}")
            return 50.0  # Neutral score on error

    def emergency_stop(self, reason: str = "Manual trigger"):
        """Activate emergency stop"""
        logger.critical(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")
        
        self.emergency_stop_active = True
        self.trading_suspended = True
        
        # Save emergency event
        self._save_emergency_event(reason)
        
        # Could add automatic position closing here
        logger.critical("🚨 All trading suspended - manual intervention required")

    def _save_emergency_event(self, reason: str):
        """Save emergency stop event to history"""
        try:
            os.makedirs("risk_management", exist_ok=True)
            
            event = {
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "account_balance": self.account_info.get('balance', 0) if self.account_info else 0,
                "drawdown": self.risk_metrics.current_drawdown,
                "active_positions": len(self.active_positions)
            }
            
            # Save event to unified file
            self._save_emergency_event_to_unified_file(event)
                
        except Exception as e:
            logger.error(f"❌ Error saving emergency event: {e}")

    def reset_emergency_stop(self, authorized_by: str = "System"):
        """Reset emergency stop (requires authorization)"""
        logger.warning(f"⚠️ Emergency stop reset by: {authorized_by}")
        
        self.emergency_stop_active = False
        self.trading_suspended = False
        
        # Log reset event
        logger.info("✅ Emergency stop reset - trading can resume")

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk management report"""
        try:
            # Update metrics first
            self.update_account_info()
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "system_status": {
                    "emergency_stop": self.emergency_stop_active,
                    "trading_suspended": self.trading_suspended,
                    "risk_level": self.risk_metrics.risk_level.value,
                    "last_update": self.risk_metrics.last_updated.isoformat()
                },
                "account_metrics": {
                    "balance": self.account_info.get('balance', 0) if self.account_info else 0,
                    "equity": self.account_info.get('equity', 0) if self.account_info else 0,
                    "free_margin": self.account_info.get('free_margin', 0) if self.account_info else 0,
                    "margin_usage": self.account_info.get('margin', 0) if self.account_info else 0,
                    "current_drawdown": self.risk_metrics.current_drawdown
                },
                "position_metrics": {
                    "active_positions": len(self.active_positions),
                    "pending_orders": len(self.pending_orders),
                    "total_exposure": self.risk_metrics.total_exposure,
                    "symbol_exposure": dict(self.symbol_exposure)
                },
                "risk_parameters": {
                    "max_risk_percent": self.risk_params.max_risk_percent,
                    "max_drawdown_percent": self.risk_params.max_drawdown_percent,
                    "max_positions": self.risk_params.max_positions,
                    "min_confidence": self.risk_params.min_confidence_threshold,
                    "auto_mode_enabled": self.risk_params.auto_mode_enabled,
                    "account_tier": self._determine_account_tier(self.account_info['balance']) if self.account_info else 'unknown'
                },
                "performance_history": {
                    "pnl_history_days": len(self.pnl_history),
                    "recent_trades": len(self.daily_trades)
                }
            }
            
            # Add warnings and recommendations
            warnings = []
            recommendations = []
            
            if self.risk_metrics.current_drawdown > self.risk_params.max_drawdown_percent * 0.8:
                warnings.append(f"Approaching drawdown limit: {self.risk_metrics.current_drawdown:.2f}%")
                recommendations.append("Consider reducing position sizes")
            
            if len(self.active_positions) > self.risk_params.max_positions * 0.8:
                warnings.append(f"Approaching position limit: {len(self.active_positions)}")
                recommendations.append("Avoid opening new positions")
            
            if self.risk_metrics.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
                warnings.append(f"High risk level: {self.risk_metrics.risk_level.value}")
                recommendations.append("Review and close risky positions")
            
            report["warnings"] = warnings
            report["recommendations"] = recommendations
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Report generation error: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "report_generation_failed"
            }

    def save_report(self, filepath: str = None) -> str:
        """Save comprehensive report to unified risk_settings.json"""
        try:
            report = self.get_comprehensive_report()
            
            # Load current risk_settings.json
            settings_path = "risk_management/risk_settings.json"
            
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logger.error(f"❌ Cannot load {settings_path}")
                return ""
            
            # Add report to risk_reports array
            if 'reports' not in settings:
                settings['reports'] = {
                    'risk_reports': [],
                    'execution_reports': [], 
                    'test_reports': [],
                    'max_stored_reports': 100,
                    'last_cleanup': None
                }
            
            settings['reports']['risk_reports'].append(report)
            
            # Keep only last N reports to prevent file from growing too large
            max_reports = settings['reports'].get('max_stored_reports', 100)
            if len(settings['reports']['risk_reports']) > max_reports:
                settings['reports']['risk_reports'] = settings['reports']['risk_reports'][-max_reports:]
            
            # Save back to file
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📊 Risk report saved to unified settings: {settings_path}")
            return settings_path
            
        except Exception as e:
            logger.error(f"❌ Error saving report: {e}")
            return ""

    def print_status(self):
        """Print comprehensive system status"""
        report = self.get_comprehensive_report()
        
        print("=" * 70)
        print("🛡️ ADVANCED RISK MANAGEMENT SYSTEM STATUS")
        print("=" * 70)
        
        system_status = report.get('system_status', {})
        account_metrics = report.get('account_metrics', {})
        position_metrics = report.get('position_metrics', {})
        
        emergency_active = "🔴 ACTIVE" if system_status.get('emergency_stop_active') else "🟢 INACTIVE"
        trading_suspended = "YES" if system_status.get('trading_suspended') else "NO"
        risk_level = system_status.get('risk_level', 'UNKNOWN')
        
        # Auto mode status
        auto_status = self.get_auto_status()
        auto_mode = "🤖 ENABLED" if auto_status['auto_mode_enabled'] else "📝 MANUAL"
        account_tier = auto_status['account_tier'].upper() if auto_status['account_tier'] != 'unknown' else 'UNKNOWN'
        
        print(f"🔴 Emergency Stop: {emergency_active}")
        print(f"⏸️ Trading Suspended: {trading_suspended}")
        print(f"🤖 Risk Mode: {auto_mode}")
        if auto_status['auto_mode_enabled']:
            print(f"🎯 Account Tier: {account_tier}")
            if auto_status['next_adjustment_in_hours'] is not None:
                print(f"⏰ Next Auto Adjustment: {auto_status['next_adjustment_in_hours']:.1f}h")
        print(f"⚠️ Risk Level: {risk_level}")
        print(f"💰 Balance: ${account_metrics.get('balance', 0):.2f}")
        print(f"📊 Equity: ${account_metrics.get('equity', 0):.2f}")
        print(f"📉 Drawdown: {account_metrics.get('current_drawdown', 0):.2f}%")
        print(f"💳 Free Margin: ${account_metrics.get('free_margin', 0):.2f}")
        print(f"📈 Active Positions: {position_metrics.get('active_positions', 0)}")
        print(f"⏳ Pending Orders: {position_metrics.get('pending_orders', 0)}")
        print(f"💼 Total Exposure: ${position_metrics.get('total_exposure', 0):.2f}")
        
        # Show current auto-adjusted parameters
        if auto_status['auto_mode_enabled']:
            print(f"📊 Auto Parameters: Risk={auto_status['current_parameters']['risk_percent']:.1f}% | "
                  f"DD={auto_status['current_parameters']['drawdown_percent']:.1f}% | "
                  f"Pos={auto_status['current_parameters']['max_positions']} | "
                  f"Conf={auto_status['current_parameters']['min_confidence']:.1f}")
        
        print("=" * 70)
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get list of open positions from MT5"""
        try:
            if not self.mt5_connector or not self.mt5_connector.is_connected():
                return []
            
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            return [
                {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': pos.type,
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'comment': pos.comment
                }
                for pos in positions
            ]
            
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []

    # ================================
    # 🆕 INTEGRATED RISK-AWARE ACTION GENERATION
    # ================================
    
    def initialize_action_generator(self):
        """Initialize the integrated action generator"""
        if not self.action_generator:
            # Convert risk_params to dict for action generator
            risk_settings = {
                'max_risk_per_trade': self.risk_params.max_risk_percent,
                'max_dca_levels': self.risk_params.max_dca_levels,
                'dca_distance_pips': self.risk_params.dca_distance_pips,
                'min_confidence_for_dca': self.risk_params.min_confidence_threshold * 20,  # Convert to percentage
                'max_position_correlation': self.risk_params.max_correlation,
                'enable_smart_entries': True,
                'enable_dca': self.risk_params.enable_dca,
                'enable_limit_orders': True
            }
            self.action_generator = RiskAwareActionGenerator(risk_settings)
    
    def generate_risk_aware_actions(self, signal_data: Dict, market_context: Dict = None) -> List[RiskAwareAction]:
        """
        Generate comprehensive risk-aware trading actions
        
        Args:
            signal_data: Trading signal with entry, SL, TP, confidence
            market_context: Additional market information
            
        Returns:
            List of RiskAwareAction objects
        """
        if not self.action_generator:
            self.initialize_action_generator()
            
        return self.action_generator.generate_actions(signal_data, market_context)
    
    def actions_to_json(self, actions: List[RiskAwareAction]) -> str:
        """Convert actions to JSON format"""
        if not self.action_generator:
            self.initialize_action_generator()
            
        return self.action_generator.actions_to_json(actions)

class RiskAwareActionGenerator:
    """Generate risk-aware trading actions based on signals and market conditions"""
    
    def __init__(self, risk_settings: Dict = None):
        self.risk_settings = risk_settings or self._load_risk_settings_from_file()
        
    def _load_risk_settings_from_file(self) -> Dict:
        """🔧 Load risk settings from actual GUI JSON file"""
        try:
            # Load from GUI settings file
            settings_path = "risk_management/risk_settings.json"
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    gui_settings = json.load(f)
                
                logger.info(f"✅ Risk Actions: Loaded DCA distance {gui_settings.get('dca_distance_pips', 50)} pips from GUI")
                return gui_settings
                
        except Exception as e:
            logger.warning(f"⚠️ Risk Actions: Could not load GUI settings: {e}")
        
        # Fallback to defaults only if file loading fails
        return self._get_default_risk_settings()
        
    def _get_default_risk_settings(self) -> Dict:
        """Default risk management settings (fallback only)"""
        return {
            'max_risk_per_trade': 2.0,  # % of account
            'max_dca_levels': 3,
            'dca_distance_pips': 50,  # This should NOT be used if GUI file exists
            'min_confidence_for_dca': 60.0,
            'max_position_correlation': 0.7,
            'enable_smart_entries': True,
            'enable_dca': True,
            'enable_limit_orders': True
        }
    
    def generate_actions(self, signal_data: Dict, market_context: Dict = None) -> List[RiskAwareAction]:
        """
        Generate comprehensive risk-aware trading actions
        
        Args:
            signal_data: Trading signal with entry, SL, TP, confidence
            market_context: Additional market information
            
        Returns:
            List of RiskAwareAction objects
        """
        actions = []
        market_context = market_context or {}
        
        try:
            # Extract signal information
            symbol = signal_data.get('symbol', '')
            direction = signal_data.get('signal', '')
            confidence = signal_data.get('confidence', 0.0)
            entry_price = signal_data.get('entry', 0.0)
            stop_loss = signal_data.get('stoploss', 0.0)
            take_profit = signal_data.get('takeprofit', 0.0)
            order_type = signal_data.get('order_type', 'market')
            
            if not all([symbol, direction, entry_price]):
                logger.warning("Insufficient signal data for action generation")
                return actions
            
            # 1. Primary Entry Action
            primary_action = self._create_primary_entry_action(
                symbol, direction, entry_price, stop_loss, take_profit, confidence, order_type
            )
            if primary_action:
                actions.append(primary_action)
            
            # 2. DCA Scale Strategy (if enabled and conditions met)
            if self.risk_settings.get('enable_dca', True) and confidence >= self.risk_settings.get('min_confidence_for_dca', 60.0):
                dca_actions = self._generate_dca_actions(
                    symbol, direction, entry_price, stop_loss, take_profit, confidence, market_context
                )
                actions.extend(dca_actions)
            
            # 3. Smart Limit Orders (if enabled)
            if self.risk_settings.get('enable_limit_orders', True):
                limit_actions = self._generate_smart_limit_orders(
                    symbol, direction, entry_price, confidence, market_context
                )
                actions.extend(limit_actions)
            
            # 4. Risk Management Actions
            risk_actions = self._generate_risk_management_actions(
                symbol, direction, entry_price, stop_loss, take_profit, confidence, market_context
            )
            actions.extend(risk_actions)
            
            # 5. Opposite Signal Close Actions (if enabled)
            if self.risk_settings.get('enable_opposite_signal_close', False):
                opposite_actions = self._generate_opposite_signal_close_actions(
                    symbol, direction, confidence, market_context
                )
                actions.extend(opposite_actions)
            
            # Sort by priority
            actions.sort(key=lambda x: x.priority)
            
            logger.info(f"Generated {len(actions)} risk-aware actions for {symbol} {direction}")
            
        except Exception as e:
            logger.error(f"Error generating risk-aware actions: {e}")
            
        return actions
    
    def _create_primary_entry_action(self, symbol: str, direction: str, entry_price: float, 
                                   stop_loss: float, take_profit: float, confidence: float, 
                                   order_type: str = None) -> Optional[RiskAwareAction]:
        """Create the primary entry action"""
        try:
            # Calculate position size based on risk
            volume = self._calculate_position_size(symbol, entry_price, stop_loss, confidence)
            
            # Use provided order_type or determine based on market conditions
            if order_type is None:
                order_type = "market" if confidence > 75.0 else "limit"
            
            return RiskAwareAction(
                action_type="primary_entry",
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                volume=volume,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"Primary {direction} entry with {confidence:.1f}% confidence",
                confidence=confidence,
                risk_level=self._determine_risk_level(confidence),
                order_type=order_type,
                priority=1
            )
        except Exception as e:
            logger.error(f"Error creating primary entry action: {e}")
            return None
    
    def _generate_dca_actions(self, symbol: str, direction: str, entry_price: float,
                            stop_loss: float, take_profit: float, confidence: float,
                            market_context: Dict) -> List[RiskAwareAction]:
        """Generate DCA entry actions"""
        actions = []
        
        try:
            # 🚨 FIXED: Check for opposite signal before generating DCA
            existing_positions = market_context.get('existing_positions', [])
            symbol_positions = [pos for pos in existing_positions 
                              if self._normalize_symbol_for_match(pos.get('symbol', '')) == 
                                 self._normalize_symbol_for_match(symbol)]
            
            if symbol_positions:
                first_pos = symbol_positions[0]
                position_direction = "BUY" if first_pos.get('type', 0) == 0 else "SELL"
                
                # Check for opposite signal with high confidence
                is_opposite_signal = (
                    (position_direction == 'BUY' and direction == 'SELL') or 
                    (position_direction == 'SELL' and direction == 'BUY')
                )
                
                if is_opposite_signal and confidence >= 0.75:  # 75%+ opposite signal
                    logger.warning(f"🚫 RISK_MANAGER DCA BLOCKED: Opposite signal {direction} with {confidence*100:.1f}% confidence vs {position_direction} positions")
                    return actions  # Return empty actions list
            max_levels = self.risk_settings.get('max_dca_levels', 3)
            dca_distance = self._get_pip_value(symbol) * self.risk_settings.get('dca_distance_pips', 50)
            
            for level in range(1, max_levels + 1):
                # Calculate DCA entry price
                if direction == "BUY":
                    dca_entry = entry_price - (dca_distance * level)
                else:  # SELL
                    dca_entry = entry_price + (dca_distance * level)
                
                # Reduce volume for each DCA level
                base_volume = self._calculate_position_size(symbol, entry_price, stop_loss, confidence)
                dca_volume = base_volume * (0.8 ** level)  # Reduce by 20% each level
                
                # Adjust confidence for DCA levels
                dca_confidence = confidence * (0.9 ** level)  # Slight reduction
                
                dca_action = RiskAwareAction(
                    action_type="dca_entry",
                    symbol=symbol,
                    direction=direction,
                    entry_price=dca_entry,
                    volume=dca_volume,
                    stop_loss=stop_loss,  # Same SL for all levels
                    take_profit=take_profit,
                    reason=f"DCA Level {level} - {direction} at {dca_entry:.5f}",
                    confidence=dca_confidence,
                    risk_level=self._determine_risk_level(dca_confidence),
                    order_type="limit",
                    priority=level + 1,
                    conditions={
                        'dca_level': level,
                        'trigger_distance': dca_distance * level,
                        'depends_on': 'primary_entry' if level == 1 else f'dca_level_{level-1}'
                    }
                )
                
                actions.append(dca_action)
                
        except Exception as e:
            logger.error(f"Error generating DCA actions: {e}")
            
        return actions
    
    def _generate_smart_limit_orders(self, symbol: str, direction: str, entry_price: float,
                                   confidence: float, market_context: Dict) -> List[RiskAwareAction]:
        """Generate smart limit orders based on support/resistance"""
        actions = []
        
        try:
            support_levels = market_context.get('support_levels', [])
            resistance_levels = market_context.get('resistance_levels', [])
            pip_value = self._get_pip_value(symbol)
            
            if direction == "BUY" and support_levels:
                # Look for BUY limit orders near support
                for support in support_levels:
                    distance = abs(entry_price - support)
                    if distance <= (pip_value * 100):  # Within 100 pips
                        limit_price = support + (pip_value * 5)  # 5 pips above support
                        
                        volume = self._calculate_position_size(symbol, limit_price, support - (pip_value * 20), confidence * 0.8)
                        
                        action = RiskAwareAction(
                            action_type="limit_order",
                            symbol=symbol,
                            direction=direction,
                            entry_price=limit_price,
                            volume=volume,
                            stop_loss=support - (pip_value * 20),
                            take_profit=limit_price + (pip_value * 40),  # 2:1 RR
                            reason=f"BUY limit near support {support:.5f}",
                            confidence=confidence * 0.8,
                            risk_level="low",
                            order_type="limit",
                            priority=6,
                            conditions={
                                'support_level': support,
                                'limit_type': 'support_bounce'
                            }
                        )
                        actions.append(action)
            
            elif direction == "SELL" and resistance_levels:
                # Look for SELL limit orders near resistance
                for resistance in resistance_levels:
                    distance = abs(entry_price - resistance)
                    if distance <= (pip_value * 100):  # Within 100 pips
                        limit_price = resistance - (pip_value * 5)  # 5 pips below resistance
                        
                        volume = self._calculate_position_size(symbol, limit_price, resistance + (pip_value * 20), confidence * 0.8)
                        
                        action = RiskAwareAction(
                            action_type="limit_order",
                            symbol=symbol,
                            direction=direction,
                            entry_price=limit_price,
                            volume=volume,
                            stop_loss=resistance + (pip_value * 20),
                            take_profit=limit_price - (pip_value * 40),  # 2:1 RR
                            reason=f"SELL limit near resistance {resistance:.5f}",
                            confidence=confidence * 0.8,
                            risk_level="low",
                            order_type="limit",
                            priority=6,
                            conditions={
                                'resistance_level': resistance,
                                'limit_type': 'resistance_rejection'
                            }
                        )
                        actions.append(action)
                        
        except Exception as e:
            logger.error(f"Error generating smart limit orders: {e}")
            
        return actions
    
    def _generate_risk_management_actions(self, symbol: str, direction: str, entry_price: float,
                                        stop_loss: float, take_profit: float, confidence: float,
                                        market_context: Dict) -> List[RiskAwareAction]:
        """Generate risk management actions like trailing stops"""
        actions = []
        
        try:
            # Trailing Stop Action
            if confidence > 70.0:
                pip_value = self._get_pip_value(symbol)
                trail_distance = pip_value * 30  # 30 pips trailing
                
                trailing_action = RiskAwareAction(
                    action_type="trailing_stop",
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    volume=0.0,  # No volume for management action
                    stop_loss=stop_loss,
                    reason=f"Trailing stop with {trail_distance/pip_value:.0f} pips distance",
                    confidence=confidence,
                    risk_level="low",
                    order_type="stop",
                    priority=8,
                    conditions={
                        'trail_distance': trail_distance,
                        'activation_profit': pip_value * 20  # Activate after 20 pips profit
                    }
                )
                actions.append(trailing_action)
                
        except Exception as e:
            logger.error(f"Error generating risk management actions: {e}")
            
        return actions
    
    def _generate_opposite_signal_close_actions(self, symbol: str, signal_direction: str, 
                                              confidence: float, market_context: Dict) -> List[RiskAwareAction]:
        """
        Generate actions to close positions when opposite signal detected
        
        FIXED: Raised confidence thresholds to avoid conflicts with signal-based S/L adjustments
        Now requires 85%+ confidence instead of 70% to avoid premature closes
        
        Args:
            symbol: Trading symbol
            signal_direction: Current signal direction (BUY/SELL)
            confidence: Signal confidence %
            market_context: Additional market data including existing positions
            
        Returns:
            List of close actions for opposite positions
        """
        actions = []
        
        try:
            # Check if feature is enabled and confidence meets threshold
            if not self.risk_settings.get('enable_opposite_signal_close', False):
                return actions
                
            min_confidence = self.risk_settings.get('opposite_signal_min_confidence', 85.0)  # RAISED from 70% to 85%
            if confidence < min_confidence:
                logger.debug(f"Opposite signal close: Confidence {confidence:.1f}% < {min_confidence}% threshold")
                return actions
            
            # Get existing positions for this symbol
            existing_positions = market_context.get('existing_positions', [])
            symbol_positions = [pos for pos in existing_positions 
                              if self._normalize_symbol_for_match(pos.get('symbol', '')) == 
                                 self._normalize_symbol_for_match(symbol)]
            
            if not symbol_positions:
                return actions
                
            # Check for opposite direction positions
            opposite_positions = []
            for pos in symbol_positions:
                pos_direction = "BUY" if pos.get('type', 0) == 0 else "SELL"
                if pos_direction != signal_direction:
                    opposite_positions.append(pos)
            
            if not opposite_positions:
                return actions
                
            # Generate close actions for opposite positions
            require_confirmation = self.risk_settings.get('opposite_signal_require_confirmation', True)
            close_delay_minutes = self.risk_settings.get('opposite_signal_close_delay_minutes', 15)
            
            for pos in opposite_positions:
                # Check if position should be closed based on additional criteria
                should_close, close_reason = self._should_close_opposite_position(
                    pos, signal_direction, confidence, market_context
                )
                
                if should_close:
                    # Determine close type (full or partial)
                    close_volume = pos.get('volume', 0.0)
                    close_type = "full"
                    
                    # For high-confidence signals, consider partial close first
                    if confidence < 80.0 and pos.get('profit', 0) > 0:
                        close_volume = close_volume * 0.5  # Close 50% if profitable
                        close_type = "partial"
                    
                    close_action = RiskAwareAction(
                        action_type="close_opposite_signal",
                        symbol=symbol,
                        direction="CLOSE",  # Special direction for close actions
                        entry_price=0.0,  # Not applicable for close
                        volume=close_volume,
                        stop_loss=0.0,
                        take_profit=0.0,
                        reason=f"{close_type.title()} close opposite position: {close_reason}",
                        confidence=confidence,
                        risk_level="medium",  # Medium risk as it's closing existing positions
                        order_type="market",
                        priority=3,  # High priority to execute quickly
                        conditions={
                            'ticket': pos.get('ticket'),
                            'original_direction': "BUY" if pos.get('type', 0) == 0 else "SELL",
                            'close_type': close_type,
                            'signal_direction': signal_direction,
                            'require_confirmation': require_confirmation,
                            'delay_minutes': close_delay_minutes,
                            'position_profit': pos.get('profit', 0),
                            'position_pips': self._calculate_position_pips(pos)
                        }
                    )
                    actions.append(close_action)
                    
                    logger.info(f"🚨 Generated opposite signal close action: {symbol} "
                              f"{pos.get('type', 0)} -> {signal_direction} "
                              f"(confidence: {confidence:.1f}%)")
        
        except Exception as e:
            logger.error(f"Error generating opposite signal close actions: {e}")
            
        return actions
    
    def _should_close_opposite_position(self, position: Dict, signal_direction: str, 
                                      confidence: float, market_context: Dict) -> Tuple[bool, str]:
        """
        Determine if an opposite position should be closed
        
        Returns:
            Tuple of (should_close: bool, reason: str)
        """
        try:
            # Get position details
            pos_profit = position.get('profit', 0)
            pos_pips = self._calculate_position_pips(position)
            pos_direction = "BUY" if position.get('type', 0) == 0 else "SELL"
            
            # Rule 1: Always close if losing significantly
            if pos_pips < -50:  # More than 50 pips loss
                return True, f"Large loss ({pos_pips:.1f} pips) - cut losses"
            
            # Rule 2: Close profitable positions if very high confidence opposite signal (RAISED THRESHOLD)
            if confidence >= 90.0 and pos_profit > 0:
                return True, f"Very high confidence opposite signal ({confidence:.1f}%) - secure profits"
            
            # Rule 3: Close small losing positions to free up margin
            if -20 <= pos_pips <= -5:  # Small loss range
                return True, f"Small loss ({pos_pips:.1f} pips) - free margin for new signal"
                
            # Rule 4: Close break-even positions (DISABLED - let signal-based adjustment handle)
            # FIXED: Don't auto-close breakeven positions - let comprehensive_aggregator handle signal-based S/L adjustment
            # if abs(pos_pips) <= 3:  # Within 3 pips of breakeven
            #     return True, f"Near breakeven ({pos_pips:.1f} pips) - redirect to stronger signal"
                
            # Rule 5: Don't close large profitable positions unless extremely confident
            if pos_pips > 30 and confidence < 90.0:
                return False, f"Large profit ({pos_pips:.1f} pips) - hold unless extremely confident"
            
            # Rule 6: High confidence with small profits - consider close (RAISED THRESHOLD)
            if confidence >= 85.0 and 5 <= pos_pips <= 30:
                return True, f"Medium profit ({pos_pips:.1f} pips) - high confidence opposite"
            
            # Default: Don't close
            return False, f"Hold position - insufficient criteria met"
            
        except Exception as e:
            logger.error(f"Error evaluating opposite position close: {e}")
            return False, "Error in evaluation"
    
    def _calculate_position_pips(self, position: Dict) -> float:
        """Calculate position profit/loss in pips"""
        try:
            symbol = position.get('symbol', '')
            entry_price = position.get('price_open', 0)
            current_price = position.get('price_current', 0)
            pos_type = position.get('type', 0)  # 0=BUY, 1=SELL
            
            if not all([entry_price, current_price]):
                return 0.0
                
            # Get pip value for symbol
            pip_value = self._get_pip_value(symbol)
            
            # Calculate price difference
            if pos_type == 0:  # BUY position
                price_diff = current_price - entry_price
            else:  # SELL position  
                price_diff = entry_price - current_price
                
            # Convert to pips
            pips = price_diff / pip_value
            return round(pips, 1)
            
        except Exception as e:
            logger.error(f"Error calculating position pips: {e}")
            return 0.0
    
    def _normalize_symbol_for_match(self, symbol: str) -> str:
        """Normalize symbol for matching (remove suffixes like .)"""
        return symbol.replace('.', '').replace('_m', '').upper()
    
    def _calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, confidence: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Base risk percentage
            base_risk = self.risk_settings.get('max_risk_per_trade', 2.0)
            
            # Adjust risk based on confidence
            if confidence > 80.0:
                risk_percentage = base_risk * 1.2  # Increase risk for high confidence
            elif confidence < 60.0:
                risk_percentage = base_risk * 0.6  # Reduce risk for low confidence
            else:
                risk_percentage = base_risk
            
            # Calculate stop loss distance in pips
            pip_value = self._get_pip_value(symbol)
            sl_distance_pips = abs(entry_price - stop_loss) / pip_value
            
            # Mock account balance (in production, get from MT5)
            account_balance = 10000.0  # Mock value
            
            # Calculate position size
            risk_amount = account_balance * (risk_percentage / 100.0)
            
            # Mock contract size and pip value for calculation
            if symbol.endswith('USD'):
                contract_size = 100000  # Standard lot
                pip_cost = pip_value * contract_size  # Cost per pip for 1 lot
            else:
                contract_size = 100000
                pip_cost = pip_value * contract_size
            
            # Calculate lots
            lots = risk_amount / (sl_distance_pips * pip_cost)
            
            # Apply minimum and maximum limits
            min_volume = 0.01
            max_volume = 1.0  # Conservative maximum
            
            return max(min_volume, min(lots, max_volume))
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Minimum fallback
    
    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol"""
        symbol_upper = symbol.upper()
        
        if 'JPY' in symbol_upper:
            return 0.01  # JPY pairs
        elif symbol_upper in ['XAUUSD', 'XAGUSD', 'GOLD', 'SILVER']:
            return 0.01  # Precious metals
        elif symbol_upper in ['BTCUSD', 'ETHUSD', 'LTCUSD']:
            return 0.01  # Crypto
        else:
            return 0.0001  # Major FX pairs
    
    def _determine_risk_level(self, confidence: float) -> str:
        """Determine risk level based on confidence"""
        if confidence >= 80.0:
            return "low"
        elif confidence >= 60.0:
            return "moderate"
        else:
            return "high"
    
    def actions_to_json(self, actions: List[RiskAwareAction]) -> str:
        """Convert actions to JSON format"""
        actions_data = []
        for action in actions:
            action_dict = {
                'action_type': action.action_type,
                'symbol': action.symbol,
                'direction': action.direction,
                'entry_price': action.entry_price,
                'volume': round(action.volume, 2),
                'stop_loss': action.stop_loss,
                'take_profit': action.take_profit,
                'reason': action.reason,
                'confidence': round(action.confidence, 1),
                'risk_level': action.risk_level,
                'order_type': action.order_type,
                'priority': action.priority,
                'conditions': action.conditions
            }
            actions_data.append(action_dict)
        
        return json.dumps(actions_data, indent=2, ensure_ascii=False)
        """Get current open positions"""
        try:
            if not MT5_AVAILABLE:
                return []
                
            positions = mt5.positions_get()
            if not positions:
                return []
            
            position_list = []
            for pos in positions:
                position_data = {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == 0 else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'profit': pos.profit,
                    'time': datetime.fromtimestamp(pos.time) if hasattr(pos, 'time') else datetime.now(),
                    'sl': getattr(pos, 'sl', 0),
                    'tp': getattr(pos, 'tp', 0)
                }
                position_list.append(position_data)
            
            return position_list
            
        except Exception as e:
            logger.error(f"❌ Error getting open positions: {e}")
            return []
        
        print("\n" + "="*70)
        print("🛡️ ADVANCED RISK MANAGEMENT SYSTEM STATUS")
        print("="*70)
        
        # System status
        status = report["system_status"]
        print(f"🔴 Emergency Stop: {'ACTIVE' if status['emergency_stop'] else 'INACTIVE'}")
        print(f"⏸️ Trading Suspended: {'YES' if status['trading_suspended'] else 'NO'}")
        print(f"⚠️ Risk Level: {status['risk_level']}")
        
        # Account metrics
        acc = report["account_metrics"]
        print(f"💰 Balance: ${acc['balance']:,.2f}")
        print(f"📊 Equity: ${acc['equity']:,.2f}") 
        print(f"📉 Drawdown: {acc['current_drawdown']:.2f}%")
        print(f"💳 Free Margin: ${acc['free_margin']:,.2f}")
        
        # Position metrics
        pos = report["position_metrics"]
        print(f"📈 Active Positions: {pos['active_positions']}")
        print(f"⏳ Pending Orders: {pos['pending_orders']}")
        print(f"💼 Total Exposure: ${pos['total_exposure']:,.2f}")
        
        # Warnings and recommendations
        if report.get("warnings"):
            print("\n⚠️ WARNINGS:")
            for warning in report["warnings"]:
                print(f"   • {warning}")
        
        if report.get("recommendations"):
            print("\n💡 RECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")
        
        print("="*70)

# Legacy compatibility class
class RiskManagementSystem(AdvancedRiskManagementSystem):
    """Legacy compatibility wrapper"""
    
    def __init__(self, risk_params=None):
        # Convert old parameters if provided
        if risk_params and hasattr(risk_params, 'max_risk_percent'):
            advanced_params = AdvancedRiskParameters(
                max_risk_percent=risk_params.max_risk_percent,
                max_drawdown_percent=risk_params.max_drawdown_percent,
                max_daily_loss_percent=getattr(risk_params, 'max_daily_loss_percent', 3.0),
                max_positions=risk_params.max_positions,
                max_correlation=risk_params.max_correlation,
                min_risk_reward_ratio=risk_params.min_risk_reward_ratio
            )
        else:
            advanced_params = AdvancedRiskParameters()
        
        super().__init__(advanced_params)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, symbol: str) -> float:
        """Legacy method for backward compatibility"""
        signal = TradeSignal(
            symbol=symbol,
            action="BUY",  # Default action
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=entry_price + abs(entry_price - stop_loss) * 2,  # Default 2:1 R:R
            volume=0.1,  # Will be calculated
            confidence=3.0  # Default confidence
        )
        
        position_size, _ = self.calculate_optimal_position_size(signal)
        return position_size
    
    def validate_trade_signal(self, signal: TradeSignal) -> Dict[str, Any]:
        """Legacy validation method"""
        report = self.validate_signal_comprehensive(signal)
        
        # Convert to legacy format
        return {
            "valid": report.result == ValidationResult.APPROVED,
            "symbol": signal.symbol,
            "checks": report.checks,
            "errors": report.errors,
            "warnings": report.warnings,
            "timestamp": report.timestamp.isoformat(),
            "position_size": report.recommended_volume,
            "risk_reward_ratio": report.metrics.get("risk_reward_ratio", 0)
        }

def main():
    """Test the advanced risk management system"""
    print("🛡️ Advanced Risk Management System - Test Mode")
    print("=" * 70)
    
    # 🧹 AUTO CLEANUP before testing risk manager
    print("🧹 Risk Manager: Auto cleanup before processing...")
    try:
        cleanup_result = cleanup_risk_manager_data(max_age_hours=72, keep_latest=10)
        print(f"✅ Cleaned {cleanup_result['total_files_deleted']} files, "
              f"freed {cleanup_result['total_space_freed_mb']:.2f} MB")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    # Initialize system with advanced parameters (Auto Mode OFF initially)
    advanced_params = AdvancedRiskParameters(
        max_risk_percent=2.0,
        max_drawdown_percent=5.0,
        max_daily_loss_percent=3.0,
        max_positions=3,
        min_confidence_threshold=3.5,
        confidence_position_multiplier={
            3.0: 0.5,
            4.0: 1.0,
            4.5: 1.5
        },
        auto_mode_enabled=False  # Start with manual mode
    )
    
    risk_manager = AdvancedRiskManagementSystem(advanced_params)
    
    # Print initial status
    print("\n📋 Initial System Status (Manual Mode):")
    risk_manager.print_status()
    
    # Test Auto Mode Toggle
    print(f"\n🤖 Testing Auto Mode Features:")
    print(f"   Current Auto Mode: {'ON' if risk_manager.risk_params.auto_mode_enabled else 'OFF'}")
    
    # Enable Auto Mode
    print(f"   Enabling Auto Mode...")
    risk_manager.toggle_auto_mode(True)
    print(f"   Auto Mode: {'ON' if risk_manager.risk_params.auto_mode_enabled else 'OFF'}")
    
    # Show auto status
    auto_status = risk_manager.get_auto_status()
    print(f"   Account Tier: {auto_status['account_tier'].upper()}")
    print(f"   Auto Adjustment Interval: {auto_status['adjustment_interval_hours']}h")
    
    # Print status after auto adjustment
    print(f"\n📋 System Status After Auto Adjustment:")
    risk_manager.print_status()
    
    # Test manual disable
    print(f"\n🔧 Testing Manual Disable:")
    risk_manager.toggle_auto_mode(False)
    print(f"   Auto Mode: {'ON' if risk_manager.risk_params.auto_mode_enabled else 'OFF'}")
    
    # Re-enable for final demo
    risk_manager.toggle_auto_mode(True)
    
    # Test signal validation
    test_signals = [
        TradeSignal(
            symbol="XAUUSD",
            action="BUY",
            entry_price=2000.0,
            stop_loss=1990.0,
            take_profit=2020.0,
            volume=0.1,
            confidence=4.2,
            strategy="TEST_HIGH_CONF",
            comment="High confidence gold trade"
        ),
        TradeSignal(
            symbol="EURUSD",
            action="SELL",
            entry_price=1.1000,
            stop_loss=1.1050,
            take_profit=1.0900,
            volume=0.1,
            confidence=2.8,
            strategy="TEST_LOW_CONF",
            comment="Low confidence EUR trade"
        ),
        TradeSignal(
            symbol="GBPJPY",
            action="BUY",
            entry_price=180.00,
            stop_loss=179.50,
            take_profit=181.00,
            volume=0.1,
            confidence=4.8,
            strategy="TEST_VERY_HIGH_CONF",
            comment="Very high confidence GBP trade"
        )
    ]
    
    # Test each signal
    for i, signal in enumerate(test_signals, 1):
        print(f"\n🔍 Testing Signal {i}: {signal.symbol} {signal.action}")
        print(f"   Entry: {signal.entry_price}, SL: {signal.stop_loss}, TP: {signal.take_profit}")
        print(f"   Confidence: {signal.confidence}, Strategy: {signal.strategy}")
        
        # Comprehensive validation
        validation = risk_manager.validate_signal_comprehensive(signal)
        
        # Print results
        result_emoji = {
            ValidationResult.APPROVED: "✅",
            ValidationResult.WARNING: "⚠️",
            ValidationResult.REJECTED: "❌",
            ValidationResult.PENDING: "⏳"
        }
        
        print(f"   Result: {result_emoji[validation.result]} {validation.result.value}")
        print(f"   Recommended Volume: {validation.recommended_volume:.2f} lots")
        print(f"   Risk Score: {validation.risk_score:.1f}/100")
        
        if validation.metrics.get("risk_reward_ratio"):
            print(f"   R:R Ratio: {validation.metrics['risk_reward_ratio']:.2f}")
        
        if validation.warnings:
            print(f"   ⚠️ Warnings: {'; '.join(validation.warnings[:2])}")
        
        if validation.errors:
            print(f"   ❌ Errors: {'; '.join(validation.errors[:2])}")
    
    # Test emergency controls
    print(f"\n🚨 Testing Emergency Controls:")
    print(f"   Emergency Stop: {'ACTIVE' if risk_manager.emergency_stop_active else 'INACTIVE'}")
    
    # Simulate emergency condition
    print(f"   Simulating emergency stop...")
    risk_manager.emergency_stop("Test emergency condition")
    print(f"   Emergency Stop: {'ACTIVE' if risk_manager.emergency_stop_active else 'INACTIVE'}")
    
    # Reset emergency
    risk_manager.reset_emergency_stop("Test Administrator")
    print(f"   After reset: {'ACTIVE' if risk_manager.emergency_stop_active else 'INACTIVE'}")
    
    # Generate and save report
    print(f"\n📊 Generating comprehensive report...")
    report_path = risk_manager.save_report()
    if report_path:
        print(f"   Report saved: {report_path}")
    
    # Final status
    print(f"\n📋 Final System Status:")
    risk_manager.print_status()
    
    print(f"\n✅ Advanced Risk Management System test completed!")

def cleanup_risk_manager_data(max_age_hours: int = 72, keep_latest: int = 10) -> Dict[str, Any]:
    """
    🧹 RISK MANAGER: Dọn dẹp dữ liệu của module này
    Dọn dẹp risk reports và logs
    
    Args:
        max_age_hours: Tuổi tối đa của file (giờ)
        keep_latest: Số file mới nhất cần giữ lại
    """
    cleanup_stats = {
        'module_name': 'risk_manager',
        'directories_cleaned': [],
        'total_files_deleted': 0,
        'total_space_freed_mb': 0.0,
        'cleanup_time': datetime.now().isoformat()
    }
    
    # Thư mục mà Risk Manager quản lý - reports now unified in risk_settings.json
    target_directories = [
        # 'reports' - đã chuyển sang unified system trong risk_settings.json
        'risk_logs',        # Risk manager logs
        'emergency_logs'    # Emergency stop logs
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
    
    print(f"🧹 RISK MANAGER cleanup complete: "
          f"{cleanup_stats['total_files_deleted']} files deleted, "
          f"{cleanup_stats['total_space_freed_mb']:.2f} MB freed")
    return cleanup_stats

def _clean_directory(directory: str, max_age_hours: int, keep_latest: int) -> Dict[str, int]:
    """Helper function để clean một directory"""
    import os
    from datetime import timedelta
    
    deleted_count = 0
    space_freed = 0.0
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    try:
        if not os.path.exists(directory):
            return {'deleted': 0, 'space_freed': 0.0}
            
        # Lấy tất cả risk management files
        all_files = []
        for file_name in os.listdir(directory):
            if file_name.endswith(('.json', '.txt', '.log', '.csv')):
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

def get_dca_mode_settings(dca_mode: str) -> Dict[str, Any]:
    """
    🔧 Get DCA mode specific settings with OFF support
    
    Args:
        dca_mode: DCA mode setting value
        
    Returns:
        Dict with DCA mode configuration
    """
    if dca_mode == "OFF":
        return {
            'enabled': False,
            'type': 'disabled',
            'description': 'DCA mode is OFF (disabled)',
            'reason': 'DCA mode is OFF'
        }
    
    mode_configs = {
        # Vietnamese labels (GUI)
        'Khoảng cách cố định': {
            'enabled': True,
            'type': 'fixed_distance',
            'description': 'Fixed pip distance between DCA entries'
        },
        'Bội số ATR': {
            'enabled': True,
            'type': 'atr_multiple',
            'description': 'ATR-based distance between DCA entries'
        },
        'Mức Fibonacci': {
            'enabled': True,
            'type': 'fibonacci',
            'description': 'Fibonacci retracement levels for DCA'
        },
        'Mức Fibo': {
            'enabled': True,
            'type': 'fibonacci',
            'description': 'Fibonacci retracement levels for DCA'
        },
        # English/Internal keys (backend)
        'fixed_pips': {
            'enabled': True,
            'type': 'fixed_distance',
            'description': 'Fixed pip distance between DCA entries'
        },
        'atr_multiple': {
            'enabled': True,
            'type': 'atr_multiple',
            'description': 'ATR-based distance between DCA entries'
        },
        'fibo_levels': {
            'enabled': True,
            'type': 'fibonacci',
            'description': 'Fibonacci retracement levels for DCA'
        }
    }
    
    return mode_configs.get(dca_mode, {
        'enabled': True,
        'type': 'fixed_distance',
        'description': 'Default fixed distance mode'
    })

def get_dca_sl_mode_settings(dca_sl_mode: str) -> Dict[str, Any]:
    """
    🔧 Get DCA stop-loss mode specific settings with OFF support
    
    Args:
        dca_sl_mode: DCA SL mode setting value
        
    Returns:
        Dict with DCA SL mode configuration
    """
    if dca_sl_mode == "OFF":
        return {
            'enabled': False,
            'type': 'disabled',
            'description': 'DCA SL mode is OFF (disabled)',
            'reason': 'DCA SL mode is OFF'
        }
    
    sl_mode_configs = {
        'SL riêng lẻ': {
            'enabled': True,
            'type': 'individual',
            'description': 'Individual stop loss for each DCA entry'
        },
        'SL trung bình': {
            'enabled': True,
            'type': 'average',
            'description': 'Average stop loss across all DCA entries'
        }
        # NOTE: 'Chỉ hòa vốn' mode removed from GUI - no longer supported
    }
    
    return sl_mode_configs.get(dca_sl_mode, {
        'enabled': True,
        'type': 'average',
        'description': 'Default average SL mode'
    })

def test_integrated_risk_aware_actions():
    """Test the integrated risk-aware action generator"""
    print("🧪 TESTING INTEGRATED RISK-AWARE ACTION GENERATOR")
    print("=" * 60)
    
    # Mock signal data
    signal_data = {
        'symbol': 'EURUSD',
        'signal': 'BUY',
        'confidence': 75.0,
        'entry': 1.0850,
        'stoploss': 1.0800,
        'takeprofit': 1.0950
    }
    
    # Mock market context
    market_context = {
        'support_levels': [1.0845, 1.0820],
        'resistance_levels': [1.0880, 1.0920],
        'atr': 0.0015,
        'ema20': 1.0840
    }
    
    # Initialize risk management system
    risk_manager = AdvancedRiskManagementSystem()
    
    # Generate actions using integrated generator
    actions = risk_manager.generate_risk_aware_actions(signal_data, market_context)
    
    print(f"Generated {len(actions)} actions:")
    print(risk_manager.actions_to_json(actions))
    
    return actions

if __name__ == "__main__":
    # Test integrated functionality
    print("🎯 TESTING INTEGRATED RISK MANAGER WITH ACTION GENERATION")
    print("=" * 70)
    
    # Run standard risk manager test
    main()
    
    print("\n" + "=" * 70)
    
    # Run integrated action generator test
    test_integrated_risk_aware_actions()