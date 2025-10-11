# -*- coding: utf-8 -*-
"""
AUTO RISK MANAGER - CHE DO TU DONG ONLY
Chi hoat dong khi user chon che do TU DONG tren GUI va click "Luu cai dat"
Tu dong dieu chinh cac thong so risk dua tren thong tin tai khoan tu mt5_essential_scan.json

Phan chia trach nhiem:
- risk_manager.py: CHE DO TU DONG - Auto adjustment
- app.py: CHE DO THU CONG - Save user settings
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

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

class RiskLevel(Enum):
    """Risk level enumeration"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW" 
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

@dataclass
class AutoRiskParameters:
    """Tham so risk tu dong dua tren thong tin tai khoan"""
    # Thong tin tai khoan tu MT5 scan
    account_balance: float = 0.0
    account_equity: float = 0.0
    account_profit: float = 0.0
    margin_level: float = 0.0
    active_positions: int = 0
    
    # Thong so risk tu dong dieu chinh
    max_risk_percent: float = 2.0
    max_drawdown_percent: float = 5.0
    max_positions: int = 5
    max_positions_per_symbol: int = 3
    
    # SL/TP tu dong (chi ap dung cho ATR mode)
    default_sl_atr_multiplier: float = 2.0
    default_tp_atr_multiplier: float = 1.5
    signal_sl_factor: float = 1.2
    signal_tp_factor: float = 0.8
    
    # DCA tu dong (chi ap dung cho ATR mode)
    enable_dca: bool = True
    max_dca_levels: int = 3
    dca_atr_multiplier: float = 1.5
    dca_volume_multiplier: float = 1.5
    
    # Account tier tu dong
    account_tier: str = "standard"
    risk_profile: str = "balanced"
    
    # Timestamp
    last_adjustment: Optional[datetime] = None

class AutoRiskManager:
    """
    AUTO RISK MANAGER
    Chi hoat dong khi user chon che do TU DONG tren GUI
    """
    
    def __init__(self):
        self.auto_params = AutoRiskParameters()
        self.is_auto_mode = False
        self.last_scan_data = None
        
        logger.info("AutoRiskManager initialized (AUTO MODE ONLY)")
        logger.info("Waiting for AUTO MODE activation from GUI...")
    
    def is_auto_mode_enabled(self) -> bool:
        """Kiem tra xem co dang o che do AUTO khong"""
        try:
            # Doc tu risk_settings.json de check trading_mode
            if os.path.exists("risk_management/risk_settings.json"):
                with open("risk_management/risk_settings.json", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                trading_mode = data.get('trading_mode', 'Thu cong')
                auto_mode_enabled = data.get('auto_mode_enabled', False)
                
                # Kiem tra ca trading_mode VA auto_mode_enabled
                is_auto = ('Tu dong' in trading_mode or 
                          'Auto Mode' in trading_mode or 
                          auto_mode_enabled)
                
                logger.info(f"Mode check: trading_mode='{trading_mode}', auto_enabled={auto_mode_enabled} -> AUTO: {is_auto}")
                return is_auto
        except Exception as e:
            logger.error(f"Error checking auto mode: {e}")
        
        return False
    
    def load_mt5_scan_data(self) -> Dict[str, Any]:
        """Load thong tin tai khoan tu mt5_essential_scan.json"""
        try:
            scan_file = "account_scans/mt5_essential_scan.json"
            if not os.path.exists(scan_file):
                logger.warning(f"MT5 scan file not found: {scan_file}")
                return {}
            
            with open(scan_file, 'r', encoding='utf-8') as f:
                scan_data = json.load(f)
            
            logger.info(f"Loaded MT5 scan data from {scan_file}")
            return scan_data
        
        except Exception as e:
            logger.error(f"Error loading MT5 scan data: {e}")
            return {}
    
    def analyze_account_info(self, scan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Phan tich thong tin tai khoan va xac dinh tier"""
        try:
            account = scan_data.get('account', {})
            positions = scan_data.get('active_positions', [])
            
            balance = account.get('balance', 0.0)
            equity = account.get('equity', 0.0)
            profit = account.get('profit', 0.0)
            margin_level = account.get('margin_level', 1000.0)
            position_count = len(positions)
            
            # Xac dinh account tier dua tren balance
            if balance < 1000:
                tier = "micro"
                risk_profile = "conservative"
            elif balance < 10000:
                tier = "mini" 
                risk_profile = "moderate"
            elif balance < 50000:
                tier = "standard"
                risk_profile = "balanced"
            else:
                tier = "professional"
                risk_profile = "aggressive"
            
            # Tinh toan risk level dua tren equity/balance ratio
            equity_ratio = equity / balance if balance > 0 else 1.0
            current_drawdown = ((balance - equity) / balance * 100) if balance > 0 else 0.0
            
            analysis = {
                'balance': balance,
                'equity': equity,
                'profit': profit,
                'margin_level': margin_level,
                'position_count': position_count,
                'equity_ratio': equity_ratio,
                'current_drawdown': current_drawdown,
                'account_tier': tier,
                'risk_profile': risk_profile,
                'health_score': self._calculate_health_score(equity_ratio, margin_level, current_drawdown)
            }
            
            logger.info(f"Account Analysis:")
            logger.info(f"   Balance: ${balance:,.2f} | Equity: ${equity:,.2f}")
            logger.info(f"   Tier: {tier.upper()} | Profile: {risk_profile}")
            logger.info(f"   Drawdown: {current_drawdown:.2f}% | Health: {analysis['health_score']:.1f}/10")
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing account info: {e}")
            return {}
    
    def _calculate_health_score(self, equity_ratio: float, margin_level: float, drawdown: float) -> float:
        """Tinh diem suc khoe tai khoan (0-10)"""
        try:
            # Diem equity ratio (0-4)
            equity_score = min(4.0, equity_ratio * 4)
            
            # Diem margin level (0-3) 
            margin_score = min(3.0, margin_level / 1000 * 3) if margin_level > 0 else 0
            
            # Diem drawdown (0-3)
            drawdown_score = max(0, 3.0 - (drawdown * 0.3))
            
            total_score = equity_score + margin_score + drawdown_score
            return min(10.0, max(0.0, total_score))
        
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 5.0
    
    def calculate_auto_risk_parameters(self, account_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Tinh toan thong so risk tu dong dua tren thong tin tai khoan"""
        try:
            balance = account_analysis.get('balance', 10000)
            tier = account_analysis.get('account_tier', 'standard')
            health_score = account_analysis.get('health_score', 5.0)
            current_drawdown = account_analysis.get('current_drawdown', 0.0)
            position_count = account_analysis.get('position_count', 0)
            
            # Base parameters theo tier
            tier_settings = {
                "micro": {
                    "base_risk": 1.0,
                    "base_drawdown": 3.0,
                    "base_positions": 2,
                    "sl_multiplier": 2.5,
                    "tp_multiplier": 1.8
                },
                "mini": {
                    "base_risk": 1.5,
                    "base_drawdown": 4.0,
                    "base_positions": 3,
                    "sl_multiplier": 2.2,
                    "tp_multiplier": 1.6
                },
                "standard": {
                    "base_risk": 2.0,
                    "base_drawdown": 5.0,
                    "base_positions": 5,
                    "sl_multiplier": 2.0,
                    "tp_multiplier": 1.5
                },
                "professional": {
                    "base_risk": 2.5,
                    "base_drawdown": 6.0,
                    "base_positions": 8,
                    "sl_multiplier": 1.8,
                    "tp_multiplier": 1.4
                }
            }
            
            base = tier_settings.get(tier, tier_settings["standard"])
            
            # Dieu chinh dua tren health score
            health_factor = health_score / 10.0
            risk_factor = 0.5 + (health_factor * 0.5)  # 0.5 - 1.0
            
            # Dieu chinh dua tren drawdown hien tai
            if current_drawdown > 3.0:
                drawdown_penalty = 1.0 - (current_drawdown * 0.1)  # Giam risk neu dang loss
                risk_factor *= max(0.3, drawdown_penalty)
            
            # Tinh toan cac thong so cuoi cung
            auto_risk_params = {
                # Core Risk Settings
                'max_risk_percent': round(base['base_risk'] * risk_factor, 1),
                'max_drawdown_percent': base['base_drawdown'],
                'max_positions': base['base_positions'],
                'max_positions_per_symbol': min(3, max(1, base['base_positions'] // 2)),
                'max_correlation': 0.8,
                
                # Trading hours (luon 24/7 cho auto mode)
                'trading_hours_start': 0,
                'trading_hours_end': 24,
                'max_spread_multiplier': 3.0,
                'max_slippage': 5.0,
                'auto_reduce_on_losses': True,
                
                # SL/TP ATR mode (auto luon dung ATR)
                'sltp_mode': 'ATR Multiplier',
                'default_sl_atr_multiplier': base['sl_multiplier'],
                'default_tp_atr_multiplier': base['tp_multiplier'],
                'signal_sl_factor': 1.2,
                'signal_tp_factor': 0.8,
                
                # DCA Auto Settings
                'enable_dca': health_score > 5.0,  # Chi enable DCA neu account khoe
                'max_dca_levels': 3 if health_score > 7.0 else 2,
                'dca_mode': 'atr_multiple',
                'dca_mode_legacy': 'Boi so ATR',
                'dca_atr_period': 14,
                'dca_atr_multiplier': 1.5,
                'dca_volume_multiplier': 1.5,
                'dca_min_drawdown': 1.0,
                'dca_sl_mode': 'SL trung binh',
                'dca_avg_sl_profit_percent': 10.0,
                
                # Volume Settings
                'volume_mode': 'Theo rui ro (Tu dong)',
                'fixed_volume_lots': 0.1,
                'default_volume_lots': 0.1,
                
                # Auto Mode Flags
                'trading_mode': 'Tu dong',
                'auto_mode_enabled': True,
                'auto_adjustment_interval': 24,
                'last_auto_adjustment': datetime.now().isoformat(),
                
                # System Flags
                'disable_news_avoidance': False,
                'disable_emergency_stop': False,
                'disable_max_dd_close': False,
                
                # Mode Settings
                'news_mode': 'AVOID',
                'emergency_mode': 'ENABLED', 
                'max_dd_mode': 'ENABLED',
                
                # Metadata
                'account_tier': tier,
                'risk_profile': account_analysis.get('risk_profile', 'balanced'),
                'health_score': health_score,
                'adjustment_reason': f"Auto adjustment based on {tier} tier, health score {health_score:.1f}"
            }
            
            logger.info(f"Auto Risk Parameters calculated:")
            logger.info(f"   Risk: {auto_risk_params['max_risk_percent']}% | Positions: {auto_risk_params['max_positions']}")
            logger.info(f"   SL/TP: {auto_risk_params['default_sl_atr_multiplier']}x / {auto_risk_params['default_tp_atr_multiplier']}x ATR")
            logger.info(f"   DCA: {'ON' if auto_risk_params['enable_dca'] else 'OFF'} ({auto_risk_params['max_dca_levels']} levels)")
            
            return auto_risk_params
        
        except Exception as e:
            logger.error(f"Error calculating auto risk parameters: {e}")
            return {}
    
    def save_auto_risk_settings(self, auto_params: Dict[str, Any]) -> bool:
        """Luu settings tu dong vao risk_settings.json"""
        try:
            # Tao directory neu chua co
            os.makedirs("risk_management", exist_ok=True)
            
            # Doc settings hien tai (neu co)
            existing_settings = {}
            settings_file = "risk_management/risk_settings.json"
            
            if os.path.exists(settings_file):
                try:
                    with open(settings_file, 'r', encoding='utf-8') as f:
                        existing_settings = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load existing settings: {e}")
            
            # Merge auto params voi existing settings
            merged_settings = existing_settings.copy()
            merged_settings.update(auto_params)
            
            # Them metadata
            merged_settings.update({
                'saved_timestamp': datetime.now().isoformat(),
                'version': '2.0',
                'auto_adjustment_source': 'risk_manager.py'
            })
            
            # Luu file
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(merged_settings, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Auto risk settings saved to {settings_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving auto risk settings: {e}")
            return False
    
    def process_auto_adjustment(self) -> bool:
        """Xu ly dieu chinh tu dong - chi khi user chon AUTO mode tren GUI"""
        try:
            if not self.is_auto_mode_enabled():
                logger.info("Auto mode not enabled - skipping auto adjustment")
                return False
            
            logger.info("Starting auto risk adjustment process...")
            
            # Load MT5 scan data
            scan_data = self.load_mt5_scan_data()
            if not scan_data:
                logger.error("No MT5 scan data available")
                return False
            
            # Analyze account
            account_analysis = self.analyze_account_info(scan_data)
            if not account_analysis:
                logger.error("Failed to analyze account")
                return False
            
            # Calculate auto params
            auto_params = self.calculate_auto_risk_parameters(account_analysis)
            if not auto_params:
                logger.error("Failed to calculate auto params")
                return False
            
            # Save settings
            success = self.save_auto_risk_settings(auto_params)
            if success:
                logger.info("Auto adjustment completed successfully")
                self.auto_params.last_adjustment = datetime.now()
            
            return success
        
        except Exception as e:
            logger.error(f"Error in auto adjustment: {e}")
            return False
    
    def update_gui_settings(self, gui_settings: Dict[str, Any], force_save: bool = False) -> bool:
        """Update settings tu GUI - CHI khi o AUTO MODE"""
        try:
            trading_mode = gui_settings.get('trading_mode', 'Thu cong')
            auto_mode_enabled = gui_settings.get('auto_mode_enabled', False)
            
            # Kiem tra auto mode request
            is_auto_request = ('Tu dong' in trading_mode or 
                              'Auto Mode' in trading_mode or 
                              auto_mode_enabled)
            
            logger.info(f"GUI update: mode={trading_mode}, auto={auto_mode_enabled}")
            
            if is_auto_request:
                logger.info("AUTO MODE detected - triggering auto adjustment...")
                return self.process_auto_adjustment()
            else:
                logger.info("Manual mode - handled by app.py")
                return False
        
        except Exception as e:
            logger.error(f"Error updating GUI settings: {e}")
            return False
    
    def get_auto_status(self) -> Dict[str, Any]:
        """Lay trang thai auto mode"""
        try:
            is_auto = self.is_auto_mode_enabled()
            
            return {
                'auto_mode_enabled': is_auto,
                'last_adjustment': self.auto_params.last_adjustment.isoformat() if self.auto_params.last_adjustment else None,
                'account_tier': self.auto_params.account_tier,
                'risk_profile': self.auto_params.risk_profile,
                'mt5_scan_available': os.path.exists("account_scans/mt5_essential_scan.json"),
                'settings_file_exists': os.path.exists("risk_management/risk_settings.json")
            }
        
        except Exception as e:
            logger.error(f"Error getting auto status: {e}")
            return {'error': str(e)}

# Singleton instance for compatibility
_auto_risk_manager = None

def get_auto_risk_manager() -> AutoRiskManager:
    """Get singleton auto risk manager instance"""
    global _auto_risk_manager
    if _auto_risk_manager is None:
        _auto_risk_manager = AutoRiskManager()
    return _auto_risk_manager

# Legacy compatibility class for existing code
class AdvancedRiskManagementSystem:
    """Legacy compatibility wrapper - redirects to AutoRiskManager for auto mode"""
    
    def __init__(self, risk_params=None):
        self.auto_manager = get_auto_risk_manager()
        logger.info("Legacy AdvancedRiskManagementSystem -> AutoRiskManager")
    
    def update_gui_settings(self, gui_settings: Dict[str, Any], force_save: bool = False) -> bool:
        return self.auto_manager.update_gui_settings(gui_settings, force_save)
    
    def is_auto_mode_enabled(self) -> bool:
        return self.auto_manager.is_auto_mode_enabled()
    
    def get_auto_status(self) -> Dict[str, Any]:
        return self.auto_manager.get_auto_status()
    
    def process_auto_adjustment(self) -> bool:
        return self.auto_manager.process_auto_adjustment()

# Legacy compatibility
RiskManagementSystem = AdvancedRiskManagementSystem

def main():
    """Test auto risk manager"""
    print("AUTO RISK MANAGER - Test Mode")
    print("=" * 50)
    
    manager = AutoRiskManager()
    
    # Test auto mode detection
    print(f"Auto Mode Check:")
    is_auto = manager.is_auto_mode_enabled()
    print(f"   Auto Mode Enabled: {is_auto}")
    
    if is_auto:
        print(f"\nRunning Auto Adjustment...")
        success = manager.process_auto_adjustment()
        print(f"   Auto Adjustment: {'SUCCESS' if success else 'FAILED'}")
        
        # Show status
        status = manager.get_auto_status()
        print(f"\nAuto Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
    else:
        print(f"\nManual Mode Active")
        print(f"   To enable auto mode:")
        print(f"   1. Open GUI Risk Management tab")
        print(f"   2. Select 'Tu dong' in Trading Mode")  
        print(f"   3. Click 'Luu cai dat'")
    
    print(f"\nTest completed")

if __name__ == "__main__":
    main()