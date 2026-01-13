# Changelog

## v4.4.0 (2026-01-13)

### 🔧 Bug Fixes
- **Notification System**: Fixed duplicate notification spam (2x notifications)
- **Pip Calculation**: Now uses MT5's actual symbol info instead of hardcoded values
- **Order Tracker**: Fixed missing notifications for some symbols

### ⚡ Improvements  
- Added extensive debug logging for order tracking
- Reduced status update interval from 60s to 30s for faster notifications
- Improved first-time order notification logic

### 🛠️ Technical Changes
- Disabled duplicate notification sources in `mt5_connector.py`
- Disabled duplicate notification methods in `notification_service.py`
- Updated `simple_order_tracker._calculate_pips()` to use `mt5.symbol_info()`

---

## v4.3.9 (Previous)
- Baseline version before notification fixes
