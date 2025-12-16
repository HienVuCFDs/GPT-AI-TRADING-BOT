#!/usr/bin/env python3
"""
Create Advanced Trading Strategies
Tạo các chiến lược giao dịch nâng cao cho Strategy Manager
"""

import json
import os
from datetime import datetime, timedelta

def create_advanced_strategies():
    """Tạo các chiến lược giao dịch nâng cao"""
    
    strategies_file = "saved_strategies.json"
    
    # Load existing strategies if any
    existing_strategies = {}
    if os.path.exists(strategies_file):
        try:
            with open(strategies_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only parse if file has content
                    existing_strategies = json.loads(content)
                else:
                    print("⚠️ Empty strategies file found, creating new one...")
        except json.JSONDecodeError:
            print("⚠️ Invalid JSON in strategies file, creating new one...")
    
    # Advanced trading strategies
    new_strategies = {
        
        # 🏃‍♂️ SCALPING STRATEGIES
        "⚡ Scalping EURUSD M1": {
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "version": "1.0",
            "description": "Chiến lược scalping nhanh trên EURUSD M1 với breakout confirmation",
            "tabs": {
                "account": {
                    "lot_size": "0.01",
                    "risk_percentage": "0.5"  # Risk thấp cho scalping
                },
                "news": {
                    "use_economic_calendar": True,
                    "auto_refresh": True
                },
                "analysis": {
                    "timeframe": "M1",
                    "symbols": ["EURUSD"]
                }
            },
            "smart_entry": {
                "enabled": True,
                "breakout_confirmation": {
                    "enabled": True,
                    "buffer_percentage": 0.02  # Buffer nhỏ cho M1
                },
                "pullback_entry": {
                    "enabled": False,
                    "buffer_percentage": 0.01
                }
            },
            "notifications": {
                "technical_analysis": True,
                "comprehensive_format": True
            }
        },
        
        "🔥 Scalping Gold M5": {
            "timestamp": (datetime.now() - timedelta(hours=18)).isoformat(),
            "version": "1.0", 
            "description": "Scalping vàng trên M5 với volume cao và risk management ketat",
            "tabs": {
                "account": {
                    "lot_size": "0.02",
                    "risk_percentage": "1"
                },
                "news": {
                    "use_economic_calendar": True,
                    "auto_refresh": True
                },
                "analysis": {
                    "timeframe": "M5",
                    "symbols": ["XAUUSD", "XAGUSD"]  # Gold và Silver
                }
            },
            "smart_entry": {
                "enabled": True,
                "breakout_confirmation": {
                    "enabled": True,
                    "buffer_percentage": 0.05
                },
                "pullback_entry": {
                    "enabled": True,
                    "buffer_percentage": 0.03
                }
            },
            "notifications": {
                "technical_analysis": True,
                "comprehensive_format": True
            }
        },
        
        # 📈 SWING TRADING STRATEGIES  
        "📊 Swing Trading Majors H4": {
            "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
            "version": "1.0",
            "description": "Swing trading các cặp major trên H4 với pullback entry",
            "tabs": {
                "account": {
                    "lot_size": "0.05",
                    "risk_percentage": "2"
                },
                "news": {
                    "use_economic_calendar": True,
                    "auto_refresh": False  # Manual check cho swing
                },
                "analysis": {
                    "timeframe": "H4", 
                    "symbols": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
                }
            },
            "smart_entry": {
                "enabled": True,
                "breakout_confirmation": {
                    "enabled": False,
                    "buffer_percentage": 0.1
                },
                "pullback_entry": {
                    "enabled": True,
                    "buffer_percentage": 0.15  # Buffer lớn hơn cho swing
                }
            },
            "notifications": {
                "technical_analysis": True,
                "comprehensive_format": False  # Đơn giản hơn cho swing
            }
        },
        
        "🌊 Trend Following D1": {
            "timestamp": (datetime.now() - timedelta(days=3)).isoformat(),
            "version": "1.0",
            "description": "Follow trend dài hạn trên daily với risk cao",
            "tabs": {
                "account": {
                    "lot_size": "0.1", 
                    "risk_percentage": "5"  # Risk cao cho position lớn
                },
                "news": {
                    "use_economic_calendar": False,  # News ít quan trọng cho D1
                    "auto_refresh": False
                },
                "analysis": {
                    "timeframe": "D1",
                    "symbols": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]
                }
            },
            "smart_entry": {
                "enabled": True,
                "breakout_confirmation": {
                    "enabled": True,
                    "buffer_percentage": 0.2  # Buffer lớn cho D1
                },
                "pullback_entry": {
                    "enabled": True, 
                    "buffer_percentage": 0.25
                }
            },
            "notifications": {
                "technical_analysis": False,  # Đơn giản cho trend following
                "comprehensive_format": False
            }
        },
        
        # 🤖 CRYPTO STRATEGIES
        "₿ Crypto Momentum H1": {
            "timestamp": (datetime.now() - timedelta(hours=6)).isoformat(),
            "version": "1.0",
            "description": "Trade momentum crypto trên H1 với volume cao",
            "tabs": {
                "account": {
                    "lot_size": "0.03",
                    "risk_percentage": "3"  # Risk trung bình cho crypto
                },
                "news": {
                    "use_economic_calendar": False,  # Crypto ít phụ thuộc economic news
                    "auto_refresh": False
                },
                "analysis": {
                    "timeframe": "H1",
                    "symbols": ["BTCUSD", "ETHUSD", "BNBUSD", "ADAUSD"]
                }
            },
            "smart_entry": {
                "enabled": True,
                "breakout_confirmation": {
                    "enabled": True,
                    "buffer_percentage": 0.08  # Crypto volatility cao hơn
                },
                "pullback_entry": {
                    "enabled": False,  # Momentum strategy ít dùng pullback
                    "buffer_percentage": 0.05
                }
            },
            "notifications": {
                "technical_analysis": True,
                "comprehensive_format": True
            }
        },
        
        "🚀 Altcoin Breakout M30": {
            "timestamp": (datetime.now() - timedelta(hours=12)).isoformat(),
            "version": "1.0", 
            "description": "Breakout trading altcoins trên M30 với risk management chặt",
            "tabs": {
                "account": {
                    "lot_size": "0.02",
                    "risk_percentage": "2"
                },
                "news": {
                    "use_economic_calendar": False,
                    "auto_refresh": False
                },
                "analysis": {
                    "timeframe": "M30",
                    "symbols": ["SOLUSD", "LTCUSD", "BNBUSD", "LINKUSD"]
                }
            },
            "smart_entry": {
                "enabled": True,
                "breakout_confirmation": {
                    "enabled": True,
                    "buffer_percentage": 0.12  # Buffer cao cho altcoin volatility
                },
                "pullback_entry": {
                    "enabled": False,
                    "buffer_percentage": 0.08
                }
            },
            "notifications": {
                "technical_analysis": True,
                "comprehensive_format": True
            }
        },
        
        # 🛡️ CONSERVATIVE STRATEGIES
        "🏦 Conservative Banking H4": {
            "timestamp": (datetime.now() - timedelta(days=4)).isoformat(),
            "version": "1.0",
            "description": "Chiến lược bảo thủ cho account lớn với risk thấp",
            "tabs": {
                "account": {
                    "lot_size": "0.01",
                    "risk_percentage": "0.25"  # Risk rất thấp
                },
                "news": {
                    "use_economic_calendar": True,
                    "auto_refresh": True
                },
                "analysis": {
                    "timeframe": "H4",
                    "symbols": ["EURUSD", "GBPUSD"]  # Chỉ major pairs
                }
            },
            "smart_entry": {
                "enabled": True,
                "breakout_confirmation": {
                    "enabled": False,  # Tránh breakout rủi ro
                    "buffer_percentage": 0.03
                },
                "pullback_entry": {
                    "enabled": True,  # Chỉ dùng pullback an toàn
                    "buffer_percentage": 0.2
                }
            },
            "notifications": {
                "technical_analysis": True,
                "comprehensive_format": True
            }
        },
        
        # ⚖️ BALANCED STRATEGIES
        "⚖️ Balanced Portfolio M15": {
            "timestamp": (datetime.now() - timedelta(hours=3)).isoformat(),
            "version": "1.0",
            "description": "Chiến lược cân bằng cho portfolio đa dạng",
            "tabs": {
                "account": {
                    "lot_size": "0.03",
                    "risk_percentage": "1.5"
                },
                "news": {
                    "use_economic_calendar": True,
                    "auto_refresh": True
                },
                "analysis": {
                    "timeframe": "M15",
                    "symbols": ["EURUSD", "XAUUSD", "BTCUSD", "GBPUSD", "USDJPY"]
                }
            },
            "smart_entry": {
                "enabled": True,
                "breakout_confirmation": {
                    "enabled": True,
                    "buffer_percentage": 0.07
                },
                "pullback_entry": {
                    "enabled": True,
                    "buffer_percentage": 0.1
                }
            },
            "notifications": {
                "technical_analysis": True,
                "comprehensive_format": True
            }
        },
        
        # 🌃 SESSION-BASED STRATEGIES
        "🇺🇸 US Session Power H1": {
            "timestamp": (datetime.now() - timedelta(hours=8)).isoformat(),
            "version": "1.0",
            "description": "Trade mạnh trong session US với USD pairs",
            "tabs": {
                "account": {
                    "lot_size": "0.04",
                    "risk_percentage": "2.5"
                },
                "news": {
                    "use_economic_calendar": True,
                    "auto_refresh": True
                },
                "analysis": {
                    "timeframe": "H1",
                    "symbols": ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDJPY"]
                }
            },
            "smart_entry": {
                "enabled": True,
                "breakout_confirmation": {
                    "enabled": True,
                    "buffer_percentage": 0.06
                },
                "pullback_entry": {
                    "enabled": True,
                    "buffer_percentage": 0.08
                }
            },
            "notifications": {
                "technical_analysis": True,
                "comprehensive_format": False
            }
        },
        
        "🇯🇵 Asian Session Quiet M30": {
            "timestamp": (datetime.now() - timedelta(hours=15)).isoformat(),
            "version": "1.0",
            "description": "Trade nhẹ nhàng trong session Asia với JPY focus",
            "tabs": {
                "account": {
                    "lot_size": "0.02",
                    "risk_percentage": "1"
                },
                "news": {
                    "use_economic_calendar": False,
                    "auto_refresh": False
                },
                "analysis": {
                    "timeframe": "M30",
                    "symbols": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY"]
                }
            },
            "smart_entry": {
                "enabled": True,
                "breakout_confirmation": {
                    "enabled": False,  # Session Asia ít breakout
                    "buffer_percentage": 0.04
                },
                "pullback_entry": {
                    "enabled": True,
                    "buffer_percentage": 0.12
                }
            },
            "notifications": {
                "technical_analysis": False,
                "comprehensive_format": False
            }
        }
    }
    
    # Merge với existing strategies
    all_strategies = {**existing_strategies, **new_strategies}
    
    # Save to file
    with open(strategies_file, 'w', encoding='utf-8') as f:
        json.dump(all_strategies, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Created {len(new_strategies)} advanced strategies!")
    print(f"📊 Total strategies: {len(all_strategies)}")
    
    # Display new strategies
    print("\n🎯 NEW STRATEGIES CREATED:")
    print("=" * 60)
    
    categories = {
        "⚡ SCALPING": ["⚡ Scalping EURUSD M1", "🔥 Scalping Gold M5"],
        "📈 SWING TRADING": ["📊 Swing Trading Majors H4", "🌊 Trend Following D1"],
        "🤖 CRYPTO": ["₿ Crypto Momentum H1", "🚀 Altcoin Breakout M30"],
        "🛡️ CONSERVATIVE": ["🏦 Conservative Banking H4"],
        "⚖️ BALANCED": ["⚖️ Balanced Portfolio M15"],
        "🌍 SESSION-BASED": ["🇺🇸 US Session Power H1", "🇯🇵 Asian Session Quiet M30"]
    }
    
    for category, strategies in categories.items():
        print(f"\n{category}:")
        for strategy_name in strategies:
            if strategy_name in new_strategies:
                strategy = new_strategies[strategy_name]
                print(f"  📋 {strategy_name}")
                print(f"     💰 Lot: {strategy['tabs']['account']['lot_size']} | Risk: {strategy['tabs']['account']['risk_percentage']}%")
                print(f"     📈 TF: {strategy['tabs']['analysis']['timeframe']} | Symbols: {len(strategy['tabs']['analysis']['symbols'])}")
                print(f"     🔧 Smart Entry: {'✅' if strategy['smart_entry']['enabled'] else '❌'}")
                print(f"     📝 Description: {strategy.get('description', 'N/A')}")

if __name__ == "__main__":
    print("🚀 Creating Advanced Trading Strategies...")
    print("=" * 60)
    
    create_advanced_strategies()
    
    print("\n" + "=" * 60)
    print("✅ All strategies created successfully!")
    print("💡 To use these strategies:")
    print("   1. Run: python app.py")  
    print("   2. Click hamburger menu (☰)")
    print("   3. Select 'Chiến lược giao dịch'")
    print("   4. Browse and load your preferred strategy")
    print("   5. Enjoy professional trading setups! 🎯")
