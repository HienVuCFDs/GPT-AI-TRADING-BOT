#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🗂️ TRADING HISTORY MANAGER
===========================
Quản lý lịch sử giao dịch - Lưu và truy vấn các lệnh đã đóng từ MT5

Tính năng:
1. Lưu lịch sử khi khởi động app
2. Auto-save mỗi 1 giờ
3. Thống kê theo ngày/tuần/tháng/năm/toàn bộ
4. Báo cáo song ngữ (Tiếng Anh + Tiếng Việt)
5. Thông tin tài khoản: Balance, Deposit, Profit %, Pips

Files được tạo (tối giản):
- trading_history_en.txt  : Báo cáo tiếng Anh (ghi đè)
- trading_history_vi.txt  : Báo cáo tiếng Việt (ghi đè)
- trading_history.json    : Dữ liệu JSON (ghi đè)

Author: Trading Bot System
Created: 2025-12-01
"""

import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict

# Setup logger
logger = logging.getLogger(__name__)

# Đường dẫn lưu trữ
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Global flag để kiểm soát auto-save thread
_auto_save_running = False
_auto_save_thread = None

# Pip value mapping for common symbols
PIP_VALUES = {
    # Forex pairs (standard 0.0001 pip, JPY pairs 0.01)
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'AUDUSD': 0.0001, 'NZDUSD': 0.0001,
    'USDCAD': 0.0001, 'USDCHF': 0.0001, 'EURGBP': 0.0001, 'EURJPY': 0.01,
    'GBPJPY': 0.01, 'USDJPY': 0.01, 'AUDJPY': 0.01, 'NZDJPY': 0.01,
    'GBPUSD.': 0.0001, 'EURUSD.': 0.0001,
    # Gold/Silver
    'XAUUSD': 0.01, 'XAUUSD.': 0.01, 'XAGUSD': 0.001,
    # Crypto (adjust based on your broker)
    'BTCUSD': 1.0, 'BTCUSD_m': 1.0,
    'ETHUSD': 0.01, 'ETHUSD_m': 0.01,
    'BNBUSD': 0.01, 'BNBUSD_m': 0.01,
    'SOLUSD': 0.01, 'SOLUSD_m': 0.01,
    'LTCUSD': 0.01, 'LTCUSD_m': 0.01,
    'XRPUSD': 0.0001, 'XRPUSD_m': 0.0001,
    'DOGEUSD': 0.00001, 'DOGEUSD_m': 0.00001,
}


# ========================================
# 💰 GET ACCOUNT INFO
# ========================================

def get_account_info() -> Dict[str, Any]:
    """Lấy thông tin tài khoản từ MT5"""
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            return {}
        
        account = mt5.account_info()
        if account is None:
            mt5.shutdown()
            return {}
        
        account_dict = account._asdict()
        
        # Tính deposit (tiền nạp ban đầu) từ balance - profit
        balance = account_dict.get('balance', 0)
        profit = account_dict.get('profit', 0)  # Floating P/L
        
        # Lấy total deposit từ history_deals (DEAL_TYPE_BALANCE)
        from_date = datetime(2020, 1, 1)
        to_date = datetime.now()
        deals = mt5.history_deals_get(from_date, to_date)
        
        total_deposit = 0
        total_withdrawal = 0
        
        if deals:
            for deal in deals:
                deal_dict = deal._asdict()
                deal_type = deal_dict.get('type', -1)
                deal_profit = deal_dict.get('profit', 0)
                
                # DEAL_TYPE_BALANCE = 2
                if deal_type == 2:
                    if deal_profit > 0:
                        total_deposit += deal_profit
                    else:
                        total_withdrawal += abs(deal_profit)
        
        mt5.shutdown()
        
        # Net deposit = deposits - withdrawals
        net_deposit = total_deposit - total_withdrawal
        
        # Realized profit = balance - net_deposit
        realized_profit = balance - net_deposit
        
        # Profit percentage
        profit_pct = (realized_profit / net_deposit * 100) if net_deposit > 0 else 0
        
        return {
            'balance': round(balance, 2),
            'equity': round(account_dict.get('equity', 0), 2),
            'margin': round(account_dict.get('margin', 0), 2),
            'free_margin': round(account_dict.get('margin_free', 0), 2),
            'floating_pl': round(profit, 2),
            'total_deposit': round(total_deposit, 2),
            'total_withdrawal': round(total_withdrawal, 2),
            'net_deposit': round(net_deposit, 2),
            'realized_profit': round(realized_profit, 2),
            'profit_pct': round(profit_pct, 2),
            'currency': account_dict.get('currency', 'USD'),
            'server': account_dict.get('server', ''),
            'login': account_dict.get('login', 0),
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting account info: {e}")
        return {}


def get_pip_value(symbol: str) -> float:
    """Lấy giá trị pip cho symbol"""
    # Try exact match first
    if symbol in PIP_VALUES:
        return PIP_VALUES[symbol]
    
    # Try without suffix
    base_symbol = symbol.rstrip('._m')
    if base_symbol in PIP_VALUES:
        return PIP_VALUES[base_symbol]
    
    # Default values based on symbol type
    if 'JPY' in symbol:
        return 0.01
    elif 'XAU' in symbol or 'GOLD' in symbol:
        return 0.01
    elif 'BTC' in symbol:
        return 1.0
    elif any(crypto in symbol for crypto in ['ETH', 'BNB', 'SOL', 'LTC']):
        return 0.01
    else:
        return 0.0001  # Default forex


# ========================================
# 📊 GET MT5 CLOSED POSITIONS
# ========================================

def get_mt5_closed_positions(
    symbol: str = None, 
    days_back: int = 7, 
    auto_trading_safe: bool = True, 
    quick_mode: bool = True, 
    max_deals: int = 5000,
    from_date: datetime = None,
    to_date: datetime = None
) -> List[Dict]:
    """
    ⚡ Lấy danh sách lệnh đã đóng từ MT5
    """
    try:
        start_time = time.time()
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            logger.error("❌ Không thể khởi tạo MT5")
            return []
        
        if auto_trading_safe:
            try:
                if mt5.account_info() is None:
                    mt5.shutdown()
                    return []
            except:
                mt5.shutdown()
                return []
        
        if to_date is None:
            to_date = datetime.now()
        if from_date is None:
            from_date = to_date - timedelta(days=days_back)
        
        try:
            if symbol:
                history = mt5.history_deals_get(from_date, to_date, group=f"*{symbol}*")
            else:
                history = mt5.history_deals_get(from_date, to_date)
            
            if history is None:
                mt5.shutdown()
                return []
            
            if quick_mode and auto_trading_safe:
                max_deals = min(max_deals, 100)
            history = history[-max_deals:] if len(history) > max_deals else history
            
        except Exception as e:
            mt5.shutdown()
            return []
        
        # Build position map for entry prices
        position_entry_prices = {}
        for deal in history:
            try:
                deal_dict = deal._asdict()
                pos_id = deal_dict.get('position_id', 0)
                entry = deal_dict.get('entry', 0)
                if entry == 0:  # DEAL_ENTRY_IN
                    position_entry_prices[pos_id] = deal_dict.get('price', 0)
            except:
                continue
        
        closed_positions = []
        for deal in history:
            try:
                deal_dict = deal._asdict()
                if deal_dict.get('entry') == 1:  # DEAL_ENTRY_OUT
                    deal_time = deal_dict.get('time', 0)
                    close_datetime = datetime.fromtimestamp(deal_time) if deal_time else None
                    
                    symbol_name = deal_dict.get('symbol', '')
                    pos_id = deal_dict.get('position_id', deal_dict.get('order', 0))
                    close_price = deal_dict.get('price', 0)
                    entry_price = position_entry_prices.get(pos_id, 0)
                    deal_type = deal_dict.get('type', 0)
                    profit = deal_dict.get('profit', 0)
                    
                    # Calculate pips
                    pip_value = get_pip_value(symbol_name)
                    if entry_price > 0 and close_price > 0:
                        price_diff = close_price - entry_price
                        # For SELL, profit is when price goes down
                        if deal_type == 1:  # SELL close means original was BUY
                            pips = price_diff / pip_value
                        else:  # BUY close means original was SELL
                            pips = -price_diff / pip_value
                    else:
                        pips = 0
                    
                    formatted_deal = {
                        'ticket': pos_id,
                        'deal_id': deal_dict.get('ticket', 0),
                        'symbol': symbol_name,
                        'type': 'BUY' if deal_type == 0 else 'SELL',
                        'volume': deal_dict.get('volume', 0),
                        'entry_price': entry_price,
                        'close_price': close_price,
                        'close_time': close_datetime.strftime('%Y-%m-%dT%H:%M:%S') if close_datetime else '',
                        'close_date': close_datetime.strftime('%Y-%m-%d') if close_datetime else '',
                        'profit': deal_dict.get('profit', 0),
                        'pips': round(pips, 1),
                        'swap': deal_dict.get('swap', 0),
                        'commission': deal_dict.get('commission', 0),
                        'comment': deal_dict.get('comment', ''),
                        'reason': deal_dict.get('reason', 0),
                        'magic': deal_dict.get('magic', 0),
                    }
                    closed_positions.append(formatted_deal)
            except:
                continue
        
        mt5.shutdown()
        
        elapsed = time.time() - start_time
        if not auto_trading_safe:
            logger.info(f"📊 Found {len(closed_positions)} closed positions ({elapsed:.2f}s)")
        
        return closed_positions
        
    except Exception as e:
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
        except:
            pass
        return []


# ========================================
# 📈 CALCULATE STATISTICS
# ========================================

def calculate_statistics(positions: List[Dict], account_info: Dict = None) -> Dict[str, Any]:
    """Tính toán thống kê từ danh sách positions"""
    if not positions:
        return {}
    
    total_profit = sum(p.get('profit', 0) for p in positions)
    total_swap = sum(p.get('swap', 0) for p in positions)
    total_commission = sum(p.get('commission', 0) for p in positions)
    total_pips = sum(p.get('pips', 0) for p in positions)
    
    buy_count = sum(1 for p in positions if p.get('type') == 'BUY')
    sell_count = sum(1 for p in positions if p.get('type') == 'SELL')
    
    profitable = [p for p in positions if p.get('profit', 0) > 0]
    losing = [p for p in positions if p.get('profit', 0) < 0]
    
    win_count = len(profitable)
    loss_count = len(losing)
    breakeven_count = len(positions) - win_count - loss_count
    
    total_trades = win_count + loss_count
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    gross_profit = sum(p.get('profit', 0) for p in profitable) if profitable else 0
    gross_loss = abs(sum(p.get('profit', 0) for p in losing)) if losing else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.99
    
    # Pips stats
    win_pips = sum(p.get('pips', 0) for p in profitable)
    loss_pips = abs(sum(p.get('pips', 0) for p in losing))
    
    avg_win = gross_profit / win_count if win_count > 0 else 0
    avg_loss = gross_loss / loss_count if loss_count > 0 else 0
    avg_profit = total_profit / len(positions) if positions else 0
    
    avg_win_pips = win_pips / win_count if win_count > 0 else 0
    avg_loss_pips = loss_pips / loss_count if loss_count > 0 else 0
    avg_pips = total_pips / len(positions) if positions else 0
    
    best_trade = max(positions, key=lambda x: x.get('profit', 0)) if positions else None
    worst_trade = min(positions, key=lambda x: x.get('profit', 0)) if positions else None
    
    best_pips = max(positions, key=lambda x: x.get('pips', 0)) if positions else None
    worst_pips = min(positions, key=lambda x: x.get('pips', 0)) if positions else None
    
    symbol_stats = defaultdict(lambda: {'count': 0, 'profit': 0, 'pips': 0, 'wins': 0, 'losses': 0})
    for pos in positions:
        sym = pos.get('symbol', 'UNKNOWN')
        symbol_stats[sym]['count'] += 1
        symbol_stats[sym]['profit'] += pos.get('profit', 0)
        symbol_stats[sym]['pips'] += pos.get('pips', 0)
        if pos.get('profit', 0) > 0:
            symbol_stats[sym]['wins'] += 1
        elif pos.get('profit', 0) < 0:
            symbol_stats[sym]['losses'] += 1
    
    sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
    
    # Calculate profit % if account info available
    net_deposit = account_info.get('net_deposit', 0) if account_info else 0
    profit_pct = (total_profit / net_deposit * 100) if net_deposit > 0 else 0
    
    return {
        'total_trades': len(positions),
        'total_profit': round(total_profit, 2),
        'total_swap': round(total_swap, 2),
        'total_commission': round(total_commission, 2),
        'net_profit': round(total_profit + total_swap + total_commission, 2),
        'profit_pct': round(profit_pct, 2),
        # Pips
        'total_pips': round(total_pips, 1),
        'win_pips': round(win_pips, 1),
        'loss_pips': round(loss_pips, 1),
        'avg_pips': round(avg_pips, 1),
        'avg_win_pips': round(avg_win_pips, 1),
        'avg_loss_pips': round(avg_loss_pips, 1),
        'best_pips': round(best_pips.get('pips', 0), 1) if best_pips else 0,
        'worst_pips': round(worst_pips.get('pips', 0), 1) if worst_pips else 0,
        # Win/Loss
        'buy_count': buy_count,
        'sell_count': sell_count,
        'win_count': win_count,
        'loss_count': loss_count,
        'breakeven_count': breakeven_count,
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor < 999 else 999.99,
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'avg_profit': round(avg_profit, 2),
        'best_trade': round(best_trade.get('profit', 0), 2) if best_trade else 0,
        'worst_trade': round(worst_trade.get('profit', 0), 2) if worst_trade else 0,
        'by_symbol': dict(sorted_symbols)
    }


# ========================================
# 📅 GET PERIOD POSITIONS
# ========================================

def get_today_positions() -> List[Dict]:
    """Lấy lệnh đóng hôm nay"""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return get_mt5_closed_positions(from_date=today, to_date=datetime.now(), auto_trading_safe=False, quick_mode=False)


def get_week_positions() -> List[Dict]:
    """Lấy lệnh đóng tuần này"""
    today = datetime.now()
    start_of_week = (today - timedelta(days=today.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return get_mt5_closed_positions(from_date=start_of_week, to_date=datetime.now(), auto_trading_safe=False, quick_mode=False)


def get_month_positions() -> List[Dict]:
    """Lấy lệnh đóng tháng này"""
    today = datetime.now()
    start_of_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return get_mt5_closed_positions(from_date=start_of_month, to_date=datetime.now(), auto_trading_safe=False, quick_mode=False)


def get_year_positions() -> List[Dict]:
    """Lấy lệnh đóng năm nay"""
    today = datetime.now()
    start_of_year = today.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return get_mt5_closed_positions(from_date=start_of_year, to_date=datetime.now(), auto_trading_safe=False, quick_mode=False, max_deals=50000)


def get_all_time_positions(max_days: int = 365) -> List[Dict]:
    """Lấy toàn bộ lệnh đóng"""
    return get_mt5_closed_positions(days_back=max_days, auto_trading_safe=False, quick_mode=False, max_deals=100000)


# ========================================
# 📝 GENERATE REPORT - ENGLISH
# ========================================

def generate_report_english(account_info: Dict, today_pos: List[Dict], week_pos: List[Dict], 
                           month_pos: List[Dict], year_pos: List[Dict], all_pos: List[Dict]) -> str:
    """Generate comprehensive report in English"""
    now = datetime.now()
    
    today_stats = calculate_statistics(today_pos, account_info)
    week_stats = calculate_statistics(week_pos, account_info)
    month_stats = calculate_statistics(month_pos, account_info)
    year_stats = calculate_statistics(year_pos, account_info)
    all_stats = calculate_statistics(all_pos, account_info)
    
    lines = []
    lines.append("=" * 75)
    lines.append("📊 TRADING HISTORY REPORT")
    lines.append(f"📅 Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 75)
    
    # Account Info Section
    if account_info:
        lines.append("\n💰 ACCOUNT INFORMATION")
        lines.append("-" * 55)
        lines.append(f"  Account:           #{account_info.get('login', 'N/A')}")
        lines.append(f"  Server:            {account_info.get('server', 'N/A')}")
        lines.append(f"  Balance:           ${account_info.get('balance', 0):>12,.2f}")
        lines.append(f"  Equity:            ${account_info.get('equity', 0):>12,.2f}")
        lines.append(f"  Floating P/L:      ${account_info.get('floating_pl', 0):>12,.2f}")
        lines.append("-" * 55)
        lines.append(f"  Total Deposit:     ${account_info.get('total_deposit', 0):>12,.2f}")
        lines.append(f"  Total Withdrawal:  ${account_info.get('total_withdrawal', 0):>12,.2f}")
        lines.append(f"  Net Deposit:       ${account_info.get('net_deposit', 0):>12,.2f}")
        lines.append("-" * 55)
        lines.append(f"  Realized Profit:   ${account_info.get('realized_profit', 0):>12,.2f}")
        profit_pct = account_info.get('profit_pct', 0)
        sign = '+' if profit_pct >= 0 else ''
        lines.append(f"  Profit %:          {sign}{profit_pct:>12.2f}%")
    
    # Helper function
    def add_period_section(title: str, period: str, stats: Dict, positions: List[Dict] = None, show_trades: bool = False):
        lines.append(f"\n{title}")
        lines.append(f"Period: {period}")
        lines.append("-" * 55)
        
        if stats:
            net_profit = stats.get('net_profit', 0)
            total_pips = stats.get('total_pips', 0)
            sign_p = '+' if net_profit >= 0 else ''
            sign_pips = '+' if total_pips >= 0 else ''
            
            lines.append(f"  Total Trades:      {stats.get('total_trades', 0):>10}")
            lines.append(f"  Win/Loss:          {stats.get('win_count', 0):>5} / {stats.get('loss_count', 0)}")
            lines.append(f"  Win Rate:          {stats.get('win_rate', 0):>9.1f}%")
            lines.append(f"  Net P/L:           {sign_p}${abs(net_profit):>11,.2f}")
            lines.append(f"  Total Pips:        {sign_pips}{total_pips:>11.1f} pips")
            lines.append(f"  Profit Factor:     {stats.get('profit_factor', 0):>12.2f}")
            lines.append(f"  Avg Win:           ${stats.get('avg_win', 0):>11,.2f}  ({stats.get('avg_win_pips', 0):>+.1f} pips)")
            lines.append(f"  Avg Loss:          ${stats.get('avg_loss', 0):>11,.2f}  ({stats.get('avg_loss_pips', 0):>+.1f} pips)")
            lines.append(f"  Best Trade:        ${stats.get('best_trade', 0):>11,.2f}  ({stats.get('best_pips', 0):>+.1f} pips)")
            lines.append(f"  Worst Trade:       ${stats.get('worst_trade', 0):>11,.2f}  ({stats.get('worst_pips', 0):>+.1f} pips)")
            
            by_symbol = stats.get('by_symbol', {})
            if by_symbol:
                lines.append(f"\n  📈 Top Symbols:")
                for sym, sdata in list(by_symbol.items())[:5]:
                    profit = sdata.get('profit', 0)
                    pips = sdata.get('pips', 0)
                    count = sdata.get('count', 0)
                    sign = '+' if profit >= 0 else ''
                    sign_pip = '+' if pips >= 0 else ''
                    lines.append(f"    {sym:12} | {count:4} trades | {sign}${abs(profit):>9,.2f} | {sign_pip}{pips:>7.1f} pips")
            
            # Show recent trades for TODAY
            if show_trades and positions:
                lines.append(f"\n  📋 Recent Closed Positions:")
                sorted_pos = sorted(positions, key=lambda x: x.get('close_time', ''), reverse=True)
                for pos in sorted_pos[:10]:
                    close_time = pos.get('close_time', '')[:16].replace('T', ' ')
                    symbol = pos.get('symbol', '')
                    ptype = pos.get('type', '')
                    volume = pos.get('volume', 0)
                    profit = pos.get('profit', 0)
                    pips = pos.get('pips', 0)
                    sign = '+' if profit >= 0 else ''
                    sign_pip = '+' if pips >= 0 else ''
                    lines.append(f"    {close_time} | {symbol:10} {ptype:4} {volume:<5.2f} | {sign}${abs(profit):>7.2f} | {sign_pip}{pips:>6.1f}p")
        else:
            lines.append("  No trades")
    
    # Add sections
    add_period_section("📅 TODAY", now.strftime('%Y-%m-%d'), today_stats, today_pos, show_trades=True)
    add_period_section("📆 THIS WEEK", f"{(now - timedelta(days=now.weekday())).strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}", week_stats)
    add_period_section("📆 THIS MONTH", now.strftime('%Y-%m'), month_stats)
    add_period_section("📆 THIS YEAR", now.strftime('%Y'), year_stats)
    add_period_section("📆 ALL TIME", "Last 365 days", all_stats)
    
    lines.append("\n" + "=" * 75)
    return "\n".join(lines)


# ========================================
# 📝 GENERATE REPORT - VIETNAMESE
# ========================================

def generate_report_vietnamese(account_info: Dict, today_pos: List[Dict], week_pos: List[Dict], 
                               month_pos: List[Dict], year_pos: List[Dict], all_pos: List[Dict]) -> str:
    """Generate comprehensive report in Vietnamese"""
    now = datetime.now()
    
    today_stats = calculate_statistics(today_pos, account_info)
    week_stats = calculate_statistics(week_pos, account_info)
    month_stats = calculate_statistics(month_pos, account_info)
    year_stats = calculate_statistics(year_pos, account_info)
    all_stats = calculate_statistics(all_pos, account_info)
    
    lines = []
    lines.append("=" * 75)
    lines.append("📊 BÁO CÁO LỊCH SỬ GIAO DỊCH")
    lines.append(f"📅 Thời gian tạo: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 75)
    
    # Account Info Section
    if account_info:
        lines.append("\n💰 THÔNG TIN TÀI KHOẢN")
        lines.append("-" * 55)
        lines.append(f"  Tài khoản:         #{account_info.get('login', 'N/A')}")
        lines.append(f"  Server:            {account_info.get('server', 'N/A')}")
        lines.append(f"  Số dư:             ${account_info.get('balance', 0):>12,.2f}")
        lines.append(f"  Vốn:               ${account_info.get('equity', 0):>12,.2f}")
        lines.append(f"  Lời/Lỗ thả nổi:    ${account_info.get('floating_pl', 0):>12,.2f}")
        lines.append("-" * 55)
        lines.append(f"  Tổng nạp:          ${account_info.get('total_deposit', 0):>12,.2f}")
        lines.append(f"  Tổng rút:          ${account_info.get('total_withdrawal', 0):>12,.2f}")
        lines.append(f"  Nạp ròng:          ${account_info.get('net_deposit', 0):>12,.2f}")
        lines.append("-" * 55)
        lines.append(f"  Lợi nhuận đã thực hiện:  ${account_info.get('realized_profit', 0):>12,.2f}")
        profit_pct = account_info.get('profit_pct', 0)
        sign = '+' if profit_pct >= 0 else ''
        lines.append(f"  % Lợi nhuận:             {sign}{profit_pct:>12.2f}%")
    
    def add_period_section(title: str, period: str, stats: Dict, positions: List[Dict] = None, show_trades: bool = False):
        lines.append(f"\n{title}")
        lines.append(f"Khoảng thời gian: {period}")
        lines.append("-" * 55)
        
        if stats:
            net_profit = stats.get('net_profit', 0)
            total_pips = stats.get('total_pips', 0)
            sign_p = '+' if net_profit >= 0 else ''
            sign_pips = '+' if total_pips >= 0 else ''
            
            lines.append(f"  Tổng giao dịch:    {stats.get('total_trades', 0):>10}")
            lines.append(f"  Thắng/Thua:        {stats.get('win_count', 0):>5} / {stats.get('loss_count', 0)}")
            lines.append(f"  Tỷ lệ thắng:       {stats.get('win_rate', 0):>9.1f}%")
            lines.append(f"  Tổng P/L:          {sign_p}${abs(net_profit):>11,.2f}")
            lines.append(f"  Tổng Pips:         {sign_pips}{total_pips:>11.1f} pips")
            lines.append(f"  Profit Factor:     {stats.get('profit_factor', 0):>12.2f}")
            lines.append(f"  TB Thắng:          ${stats.get('avg_win', 0):>11,.2f}  ({stats.get('avg_win_pips', 0):>+.1f} pips)")
            lines.append(f"  TB Thua:           ${stats.get('avg_loss', 0):>11,.2f}  ({stats.get('avg_loss_pips', 0):>+.1f} pips)")
            lines.append(f"  Trade tốt nhất:    ${stats.get('best_trade', 0):>11,.2f}  ({stats.get('best_pips', 0):>+.1f} pips)")
            lines.append(f"  Trade tệ nhất:     ${stats.get('worst_trade', 0):>11,.2f}  ({stats.get('worst_pips', 0):>+.1f} pips)")
            
            by_symbol = stats.get('by_symbol', {})
            if by_symbol:
                lines.append(f"\n  📈 Top Symbols:")
                for sym, sdata in list(by_symbol.items())[:5]:
                    profit = sdata.get('profit', 0)
                    pips = sdata.get('pips', 0)
                    count = sdata.get('count', 0)
                    sign = '+' if profit >= 0 else ''
                    sign_pip = '+' if pips >= 0 else ''
                    lines.append(f"    {sym:12} | {count:4} lệnh  | {sign}${abs(profit):>9,.2f} | {sign_pip}{pips:>7.1f} pips")
            
            if show_trades and positions:
                lines.append(f"\n  📋 Lệnh đóng gần đây:")
                sorted_pos = sorted(positions, key=lambda x: x.get('close_time', ''), reverse=True)
                for pos in sorted_pos[:10]:
                    close_time = pos.get('close_time', '')[:16].replace('T', ' ')
                    symbol = pos.get('symbol', '')
                    ptype = pos.get('type', '')
                    volume = pos.get('volume', 0)
                    profit = pos.get('profit', 0)
                    pips = pos.get('pips', 0)
                    sign = '+' if profit >= 0 else ''
                    sign_pip = '+' if pips >= 0 else ''
                    lines.append(f"    {close_time} | {symbol:10} {ptype:4} {volume:<5.2f} | {sign}${abs(profit):>7.2f} | {sign_pip}{pips:>6.1f}p")
        else:
            lines.append("  Không có giao dịch")
    
    add_period_section("📅 HÔM NAY", now.strftime('%Y-%m-%d'), today_stats, today_pos, show_trades=True)
    add_period_section("📆 TUẦN NÀY", f"{(now - timedelta(days=now.weekday())).strftime('%Y-%m-%d')} đến {now.strftime('%Y-%m-%d')}", week_stats)
    add_period_section("📆 THÁNG NÀY", now.strftime('%Y-%m'), month_stats)
    add_period_section("📆 NĂM NAY", now.strftime('%Y'), year_stats)
    add_period_section("📆 TOÀN BỘ", "365 ngày gần nhất", all_stats)
    
    lines.append("\n" + "=" * 75)
    return "\n".join(lines)


# ========================================
# 💾 SAVE TRADING HISTORY
# ========================================

def save_trading_history(
    closed_positions: List[Dict] = None,
    final_actions: List[Dict] = None,
    symbol: str = None,
    quick_mode: bool = False,
    days_back: int = 1,
    include_summary: bool = True,
    save_comprehensive: bool = True
) -> bool:
    """
    💾 Lưu lịch sử giao dịch - Chỉ 3 file (ghi đè)
    
    Files:
    - trading_history_en.txt : Báo cáo tiếng Anh
    - trading_history_vi.txt : Báo cáo tiếng Việt  
    - trading_history.json   : Dữ liệu JSON
    """
    try:
        now = datetime.now()
        
        # Lấy thông tin tài khoản
        logger.info("💰 Fetching account info...")
        account_info = get_account_info()
        
        # Lấy dữ liệu cho các khoảng thời gian
        logger.info("📊 Fetching trading history data...")
        today_pos = get_today_positions()
        week_pos = get_week_positions()
        month_pos = get_month_positions()
        year_pos = get_year_positions()
        all_pos = get_all_time_positions(max_days=365)
        
        # 1. Tạo báo cáo tiếng Anh
        report_en = generate_report_english(account_info, today_pos, week_pos, month_pos, year_pos, all_pos)
        file_en = REPORTS_DIR / "trading_history_en.txt"
        with open(file_en, 'w', encoding='utf-8') as f:
            f.write(report_en)
        logger.info(f"✅ Saved English report: {file_en}")
        
        # 2. Tạo báo cáo tiếng Việt
        report_vi = generate_report_vietnamese(account_info, today_pos, week_pos, month_pos, year_pos, all_pos)
        file_vi = REPORTS_DIR / "trading_history_vi.txt"
        with open(file_vi, 'w', encoding='utf-8') as f:
            f.write(report_vi)
        logger.info(f"✅ Saved Vietnamese report: {file_vi}")
        
        # 3. Lưu JSON
        today_stats = calculate_statistics(today_pos, account_info)
        week_stats = calculate_statistics(week_pos, account_info)
        month_stats = calculate_statistics(month_pos, account_info)
        year_stats = calculate_statistics(year_pos, account_info)
        all_stats = calculate_statistics(all_pos, account_info)
        
        json_data = {
            'generated_at': now.isoformat(),
            'account': account_info,
            'today': {
                'period': now.strftime('%Y-%m-%d'),
                'count': len(today_pos),
                'stats': today_stats,
                'positions': today_pos
            },
            'this_week': {
                'period': f"{(now - timedelta(days=now.weekday())).strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}",
                'count': len(week_pos),
                'stats': week_stats
            },
            'this_month': {
                'period': now.strftime('%Y-%m'),
                'count': len(month_pos),
                'stats': month_stats
            },
            'this_year': {
                'period': now.strftime('%Y'),
                'count': len(year_pos),
                'stats': year_stats
            },
            'all_time': {
                'period': 'Last 365 days',
                'count': len(all_pos),
                'stats': all_stats
            }
        }
        
        file_json = REPORTS_DIR / "trading_history.json"
        with open(file_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"✅ Saved JSON data: {file_json}")
        
        logger.info(f"✅ Trading history saved: Today={len(today_pos)}, Week={len(week_pos)}, Month={len(month_pos)}, Year={len(year_pos)}, All={len(all_pos)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving trading history: {e}")
        import traceback
        traceback.print_exc()
        return False


# ========================================
# 📖 LOAD TRADING HISTORY
# ========================================

def load_trading_history() -> Optional[Dict]:
    """Đọc lịch sử giao dịch từ file JSON"""
    try:
        file_path = REPORTS_DIR / "trading_history.json"
        if not file_path.exists():
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading history: {e}")
        return None


# ========================================
# 📊 GET TRADING STATISTICS
# ========================================

def get_trading_statistics(days_back: int = 7, symbol: str = None) -> Dict[str, Any]:
    """Lấy thống kê giao dịch"""
    try:
        positions = get_mt5_closed_positions(symbol=symbol, days_back=days_back, auto_trading_safe=False, quick_mode=False)
        if not positions:
            return {'status': 'no_data', 'message': 'No closed positions'}
        account_info = get_account_info()
        stats = calculate_statistics(positions, account_info)
        stats['status'] = 'success'
        stats['days_back'] = days_back
        return stats
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# ========================================
# 🧹 CLEANUP OLD FILES
# ========================================

def cleanup_old_history_files() -> int:
    """Xóa các file lịch sử cũ, giữ lại chỉ 3 file chính"""
    try:
        import glob
        deleted = 0
        
        # Các file cần giữ
        keep_files = {
            'trading_history_en.txt',
            'trading_history_vi.txt', 
            'trading_history.json',
            'execution_reports.json',
            'action_summary.json'
        }
        
        # Xóa các file trading_history_*.json cũ
        for pattern in ['trading_history_*.json', 'trading_history_*.txt', 
                       'comprehensive_report.*', 'trading_summary.txt',
                       'latest_trading_history.json']:
            for f in glob.glob(str(REPORTS_DIR / pattern)):
                fname = os.path.basename(f)
                if fname not in keep_files:
                    try:
                        os.remove(f)
                        deleted += 1
                        logger.debug(f"🗑️ Deleted: {fname}")
                    except:
                        pass
        
        if deleted > 0:
            logger.info(f"🧹 Cleaned up {deleted} old files")
        
        return deleted
        
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")
        return 0


# ========================================
# 🔄 AUTO SAVE SYSTEM
# ========================================

def start_auto_save(interval_minutes: int = 60, save_on_start: bool = True):
    """
    🔄 Bắt đầu auto-save trong background thread
    
    Args:
        interval_minutes: Khoảng thời gian (phút), mặc định 60
        save_on_start: Lưu ngay khi khởi động
    """
    global _auto_save_running, _auto_save_thread
    
    if _auto_save_running:
        logger.warning("⚠️ Auto-save already running")
        return
    
    def auto_save_loop():
        global _auto_save_running
        
        logger.info(f"🔄 Starting auto-save (every {interval_minutes} minutes)")
        
        # Cleanup old files first
        cleanup_old_history_files()
        
        # Save on start
        if save_on_start:
            logger.info("💾 Saving trading history on startup...")
            try:
                save_trading_history()
            except Exception as e:
                logger.error(f"❌ Startup save error: {e}")
        
        last_save = time.time()
        
        while _auto_save_running:
            try:
                time.sleep(60)  # Check every minute
                
                elapsed = time.time() - last_save
                if elapsed >= interval_minutes * 60:
                    logger.info(f"💾 Auto-saving trading history...")
                    save_trading_history()
                    
                    # 🤖 AI TRAINING: Collect new training data
                    try:
                        collect_training_data_from_new_trades()
                    except Exception as te:
                        logger.warning(f"⚠️ Training data collection skipped: {te}")
                    
                    last_save = time.time()
                    
            except Exception as e:
                logger.error(f"❌ Auto-save error: {e}")
                time.sleep(60)
        
        logger.info("⏹️ Auto-save stopped")
    
    _auto_save_running = True
    _auto_save_thread = threading.Thread(target=auto_save_loop, daemon=True)
    _auto_save_thread.start()
    
    logger.info(f"✅ Auto-save started (interval: {interval_minutes} min)")


def stop_auto_save():
    """Dừng auto-save"""
    global _auto_save_running
    _auto_save_running = False
    logger.info("⏹️ Stopping auto-save...")


def is_auto_save_running() -> bool:
    """Kiểm tra auto-save có đang chạy không"""
    return _auto_save_running


# ========================================
# 🤖 AI TRAINING DATA INTEGRATION
# ========================================

def collect_training_data_from_new_trades():
    """
    Thu thập training data từ trades mới đóng
    Gọi từ auto-save để tự động cập nhật AI training database
    """
    try:
        from ai_server.training_data_collector import get_collector
        collector = get_collector()
        
        # Get recent closed positions (last 24 hours)
        positions = get_mt5_closed_positions(days_back=1, auto_trading_safe=True)
        
        if not positions:
            return 0
        
        saved_count = 0
        for pos in positions:
            ticket = pos.get('ticket', 0)
            symbol = pos.get('symbol', '')
            trade_type = pos.get('type', '')
            profit_pips = pos.get('pips', 0)
            profit_money = pos.get('profit', 0)
            entry_price = pos.get('entry_price', 0)
            close_price = pos.get('close_price', 0)
            close_time = pos.get('close_time', '')
            swap = pos.get('swap', 0)
            commission = pos.get('commission', 0)
            
            # Load current indicators for context
            indicators = _load_indicators_for_symbol(symbol)
            
            # Determine exit reason based on profit
            if profit_pips > 50:
                exit_reason = "TP_HIT"
            elif profit_pips < -50:
                exit_reason = "SL_HIT"
            elif profit_money > 0:
                exit_reason = "PROFIT_CLOSE"
            else:
                exit_reason = "LOSS_CLOSE"
            
            # Save to training database
            result_id = collector.save_trade_result(
                ticket_id=ticket,
                symbol=symbol,
                trade_type=trade_type,
                entry_time="",  # Not available from deals
                entry_price=entry_price,
                volume=pos.get('volume', 0.01),
                exit_time=close_time,
                exit_price=close_price,
                exit_reason=exit_reason,
                profit_money=profit_money,
                profit_pips=profit_pips,
                duration_minutes=0,
                swap=swap,
                commission=commission,
                indicators_at_entry=indicators
            )
            
            if result_id > 0:
                saved_count += 1
        
        if saved_count > 0:
            logger.info(f"🤖 AI Training: Saved {saved_count} new trade results")
            
            # Generate training examples from new data
            collector.generate_training_examples()
        
        return saved_count
        
    except ImportError:
        # AI training module not available
        return 0
    except Exception as e:
        logger.warning(f"⚠️ AI training data collection error: {e}")
        return 0


def _load_indicators_for_symbol(symbol: str) -> Dict:
    """Load latest indicators for a symbol from indicator_output"""
    indicators = {}
    
    indicator_dir = Path(__file__).parent / "indicator_output"
    
    for tf in ['M15', 'M30', 'H1']:
        indicator_file = indicator_dir / f"{symbol}_{tf}.json"
        if indicator_file.exists():
            try:
                with open(indicator_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        indicators = data[-1]
                    elif isinstance(data, dict):
                        indicators = data
                break
            except:
                pass
    
    return indicators


def export_trading_history_for_ai(days_back: int = 365) -> Dict:
    """
    Export trading history in format optimized for AI training
    Includes indicators, patterns, and trade outcomes
    """
    positions = get_mt5_closed_positions(days_back=days_back, auto_trading_safe=True, max_deals=10000)
    
    if not positions:
        return {"trades": [], "statistics": {}}
    
    # Calculate statistics
    stats = calculate_statistics(positions)
    
    # Group by symbol for pattern analysis
    by_symbol = {}
    for pos in positions:
        symbol = pos.get('symbol', 'UNKNOWN')
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(pos)
    
    # Calculate per-symbol patterns
    symbol_patterns = {}
    for symbol, trades in by_symbol.items():
        wins = [t for t in trades if t.get('profit', 0) > 0]
        losses = [t for t in trades if t.get('profit', 0) < 0]
        
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win_pips = sum(t.get('pips', 0) for t in wins) / len(wins) if wins else 0
        avg_loss_pips = abs(sum(t.get('pips', 0) for t in losses) / len(losses)) if losses else 0
        
        symbol_patterns[symbol] = {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 2),
            "wins": len(wins),
            "losses": len(losses),
            "avg_win_pips": round(avg_win_pips, 1),
            "avg_loss_pips": round(avg_loss_pips, 1),
            "recommendation": "TRADE" if win_rate > 50 else "AVOID" if win_rate < 40 else "NEUTRAL"
        }
    
    return {
        "trades": positions,
        "statistics": stats,
        "symbol_patterns": symbol_patterns,
        "exported_at": datetime.now().isoformat(),
        "days_covered": days_back
    }


def get_ai_training_summary() -> Dict:
    """
    Get summary of AI training data available
    """
    try:
        from ai_server.training_data_collector import get_collector
        collector = get_collector()
        return collector.get_training_stats()
    except ImportError:
        return {"error": "AI training module not available"}
    except Exception as e:
        return {"error": str(e)}


# ========================================
# 🧪 TEST / MAIN
# ========================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 75)
    print("🗂️ TRADING HISTORY MANAGER - TEST")
    print("=" * 75 + "\n")
    
    # Cleanup old files first
    print("🧹 Cleaning up old files...")
    deleted = cleanup_old_history_files()
    print(f"   → Deleted {deleted} old files\n")
    
    # Save trading history
    print("💾 Saving trading history (bilingual with pips)...")
    success = save_trading_history()
    print(f"   → {'✅ Success' if success else '❌ Failed'}\n")
    
    # Show summary
    print("📊 Loading saved data...")
    data = load_trading_history()
    if data:
        # Account info
        acc = data.get('account', {})
        print(f"\n💰 Account Info:")
        print(f"   Balance:    ${acc.get('balance', 0):,.2f}")
        print(f"   Net Deposit: ${acc.get('net_deposit', 0):,.2f}")
        print(f"   Realized P/L: ${acc.get('realized_profit', 0):,.2f}")
        print(f"   Profit %:    {acc.get('profit_pct', 0):+.2f}%")
        
        print(f"\n📈 Trading Summary:")
        print(f"   {'Period':<15} | {'Trades':>6} | {'Profit':>12} | {'Pips':>10} | {'Win%':>6}")
        print("   " + "-" * 60)
        for period in ['today', 'this_week', 'this_month', 'this_year', 'all_time']:
            p_data = data.get(period, {})
            stats = p_data.get('stats', {})
            count = p_data.get('count', 0)
            profit = stats.get('net_profit', 0)
            pips = stats.get('total_pips', 0)
            win_rate = stats.get('win_rate', 0)
            sign = '+' if profit >= 0 else ''
            sign_pips = '+' if pips >= 0 else ''
            print(f"   {period:<15} | {count:>6} | {sign}${abs(profit):>10,.2f} | {sign_pips}{pips:>8.1f}p | {win_rate:>5.1f}%")
    
    print("\n" + "=" * 75)
    print("✅ Done! Check reports/ folder for:")
    print("   - trading_history_en.txt (English)")
    print("   - trading_history_vi.txt (Vietnamese)")
    print("   - trading_history.json   (JSON data)")
    print("=" * 75 + "\n")
