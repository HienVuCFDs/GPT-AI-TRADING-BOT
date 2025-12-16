"""
Training Data Collector for Self-Learning AI Trading System
Thu thập data từ trades thực tế để train AI

Tích hợp với:
- reports/trading_history.json - Lịch sử giao dịch
- reports/execution_reports.json - Chi tiết các lệnh đã thực hiện
- analysis_results/*_signal_*.json - Signals đã tạo
- indicator_output/*.json - Indicator data
"""

import os
import json
import sqlite3
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "ai_server", "trading_training_data.db")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis_results")
INDICATOR_DIR = os.path.join(BASE_DIR, "indicator_output")


class TrainingDataCollector:
    """Thu thập và lưu trữ data để train AI"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Khởi tạo SQLite database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bảng lưu signals đã tạo
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                entry_price REAL,
                stoploss REAL,
                takeprofit REAL,
                confidence REAL,
                
                -- Indicators tại thời điểm tạo signal
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                ema_20 REAL,
                ema_50 REAL,
                ema_200 REAL,
                adx REAL,
                atr REAL,
                stoch_k REAL,
                stoch_d REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                current_price REAL,
                
                -- Pattern data
                candle_patterns TEXT,
                price_patterns TEXT,
                support_levels TEXT,
                resistance_levels TEXT,
                trend_direction TEXT,
                
                -- Source
                source TEXT,
                raw_data TEXT,
                
                -- Trade tracking
                trade_executed INTEGER DEFAULT 0,
                ticket_id INTEGER,
                
                UNIQUE(timestamp, symbol, source)
            )
        """)
        
        # Bảng lưu kết quả trade (từ trading_history.json)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                ticket_id INTEGER UNIQUE,
                symbol TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                
                -- Entry
                entry_time TEXT,
                entry_price REAL,
                volume REAL,
                
                -- Exit
                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,
                
                -- Results
                profit_money REAL,
                profit_pips REAL,
                duration_minutes INTEGER,
                swap REAL,
                commission REAL,
                
                -- Market conditions
                rsi_at_entry REAL,
                rsi_at_exit REAL,
                trend_at_entry TEXT,
                
                -- Evaluation
                outcome TEXT,
                quality_score REAL,
                
                -- Indicators snapshot (JSON)
                indicators_at_entry TEXT,
                
                FOREIGN KEY (signal_id) REFERENCES signals(id)
            )
        """)
        
        # Bảng lưu training examples
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                
                -- Input (what AI sees)
                input_prompt TEXT NOT NULL,
                
                -- Output (correct answer based on outcome)
                expected_output TEXT NOT NULL,
                
                -- Metadata
                signal_id INTEGER,
                trade_result_id INTEGER,
                outcome TEXT,
                profit_pips REAL,
                symbol TEXT,
                
                -- Training status
                used_in_training INTEGER DEFAULT 0,
                training_batch TEXT,
                
                FOREIGN KEY (signal_id) REFERENCES signals(id),
                FOREIGN KEY (trade_result_id) REFERENCES trade_results(id)
            )
        """)
        
        # Bảng lưu model versions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                training_examples_count INTEGER,
                win_rate_before REAL,
                win_rate_after REAL,
                notes TEXT,
                model_path TEXT,
                is_active INTEGER DEFAULT 0
            )
        """)
        
        # 🆕 Bảng lưu executions (S/L modifications, closes, opens, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                ticket INTEGER,
                success INTEGER NOT NULL,
                result_message TEXT,
                
                -- Details JSON
                details TEXT,
                
                -- Extracted fields from details for easy querying
                old_sl REAL,
                new_sl REAL,
                old_tp REAL,
                new_tp REAL,
                entry_price REAL,
                exit_price REAL,
                profit_pips REAL,
                profit_usd REAL,
                reason TEXT,
                confidence REAL,
                outcome TEXT,
                
                -- For learning
                used_in_training INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_path}")
    
    # ============================================
    # IMPORT FROM EXISTING DATA
    # ============================================
    
    def import_from_trading_history(self):
        """Import trades từ reports/trading_history.json và tạo synthetic training data"""
        
        history_file = os.path.join(REPORTS_DIR, "trading_history.json")
        if not os.path.exists(history_file):
            print(f"❌ File not found: {history_file}")
            return 0
        
        with open(history_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get yearly stats (most complete data)
        year_stats = data.get('this_year', {}).get('stats', {})
        by_symbol = year_stats.get('by_symbol', {})
        
        imported = 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Import summary per symbol và tạo synthetic training examples
        for symbol, stats in by_symbol.items():
            wins = stats.get('wins', 0)
            losses = stats.get('losses', 0)
            total_profit = stats.get('profit', 0)
            total_pips = stats.get('pips', 0)
            count = stats.get('count', 0)
            
            if count == 0:
                continue
            
            win_rate = (wins / count * 100) if count > 0 else 50
            avg_pips = total_pips / count if count > 0 else 0
            
            print(f"📊 {symbol}: {wins}W/{losses}L ({win_rate:.1f}% WR), {total_profit:.2f}$, {total_pips:.1f} pips")
            
            # Tạo synthetic training examples based on win rate
            # Symbols với win rate > 55% → Learn to trade them
            # Symbols với win rate < 45% → Learn to avoid or reverse
            
            self._create_synthetic_examples(cursor, symbol, win_rate, avg_pips, wins, losses)
            imported += 1
        
        conn.commit()
        conn.close()
        print(f"✅ Imported {imported} symbol statistics from trading_history.json")
        return imported
    
    def _create_synthetic_examples(self, cursor, symbol: str, win_rate: float, 
                                   avg_pips: float, wins: int, losses: int):
        """Tạo synthetic training examples từ thống kê symbol"""
        
        # Load indicator data for this symbol if available
        indicators = self._load_latest_indicators(symbol)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create examples based on win rate patterns
        if win_rate >= 55:
            # Good symbol - create positive examples
            for i in range(min(wins, 20)):  # Max 20 examples per symbol
                # BUY worked well
                input_prompt = self._build_input_prompt(
                    symbol, 
                    indicators.get('close', 1.0),
                    indicators.get('RSI14', 35 + i),  # Vary RSI
                    indicators.get('MACD', 0.001),
                    indicators.get('EMA20', 1.0),
                    indicators.get('EMA50', 0.99),
                    indicators.get('ADX14', 25 + i % 10),
                    indicators.get('ATR14', 0.001)
                )
                
                expected_output = json.dumps({
                    "signal": "BUY",
                    "confidence": int(min(90, 60 + win_rate/3)),
                    "reason": f"Historical win rate {win_rate:.1f}% on {symbol}",
                    "learned_from": "WIN",
                    "avg_pips": avg_pips
                })
                
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO training_examples 
                        (created_at, input_prompt, expected_output, outcome, profit_pips, symbol)
                        VALUES (?, ?, ?, 'WIN', ?, ?)
                    """, (timestamp, input_prompt, expected_output, avg_pips, symbol))
                except:
                    pass
                    
        elif win_rate <= 45:
            # Bad symbol - create negative examples (learn to avoid)
            for i in range(min(losses, 15)):
                input_prompt = self._build_input_prompt(
                    symbol,
                    indicators.get('close', 1.0),
                    indicators.get('RSI14', 50 + i),
                    indicators.get('MACD', -0.001),
                    indicators.get('EMA20', 1.0),
                    indicators.get('EMA50', 1.01),
                    indicators.get('ADX14', 15 + i % 10),
                    indicators.get('ATR14', 0.001)
                )
                
                expected_output = json.dumps({
                    "signal": "HOLD",
                    "confidence": 60,
                    "reason": f"Avoid - historical win rate only {win_rate:.1f}% on {symbol}",
                    "learned_from": "LOSS",
                    "avg_pips": avg_pips
                })
                
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO training_examples 
                        (created_at, input_prompt, expected_output, outcome, profit_pips, symbol)
                        VALUES (?, ?, ?, 'LOSS', ?, ?)
                    """, (timestamp, input_prompt, expected_output, avg_pips, symbol))
                except:
                    pass
        
        # Mixed examples for learning context
        for i in range(min(5, wins)):
            input_prompt = self._build_input_prompt(
                symbol,
                indicators.get('close', 1.0),
                30 + i * 5,  # RSI from 30 to 50
                0.0005,
                indicators.get('EMA20', 1.0),
                indicators.get('EMA50', 1.0),
                20 + i * 2,
                indicators.get('ATR14', 0.001)
            )
            
            # Decide based on win rate
            if win_rate > 50:
                signal = "BUY"
                confidence = int(50 + win_rate/3)
            else:
                signal = "HOLD"
                confidence = 50
            
            expected_output = json.dumps({
                "signal": signal,
                "confidence": confidence,
                "reason": f"Based on {win_rate:.1f}% historical win rate",
                "learned_from": "MIXED"
            })
            
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO training_examples 
                    (created_at, input_prompt, expected_output, outcome, profit_pips, symbol)
                    VALUES (?, ?, ?, 'MIXED', ?, ?)
                """, (timestamp, input_prompt, expected_output, avg_pips, symbol))
            except:
                pass
    
    def _load_latest_indicators(self, symbol: str) -> Dict:
        """Load latest indicators for a symbol"""
        indicators = {}
        
        for tf in ['M15', 'M30', 'H1']:
            indicator_file = os.path.join(INDICATOR_DIR, f"{symbol}_{tf}.json")
            if os.path.exists(indicator_file):
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
    
    def import_from_execution_reports(self):
        """Import trades từ reports/execution_reports.json"""
        
        reports_file = os.path.join(REPORTS_DIR, "execution_reports.json")
        if not os.path.exists(reports_file):
            print(f"❌ File not found: {reports_file}")
            return 0
        
        with open(reports_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        reports = data.get('execution_reports', [])
        print(f"📄 Found {len(reports)} execution reports")
        
        imported = 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for report in reports:
            timestamp = report.get('timestamp', '')
            symbol = report.get('symbol', '')
            actions = report.get('actions', [])
            
            for action in actions:
                if not action.get('action'):  # Skip empty actions
                    continue
                
                action_type = action.get('action', '')
                signal_trigger = action.get('signal_trigger', '')
                confidence = action.get('confidence', 0)
                ticket = action.get('ticket')
                profit_pips = action.get('current_profit_pips', 0)
                position_type = action.get('position_type', '')
                
                if signal_trigger and ticket:
                    # Save as signal
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO signals 
                            (timestamp, symbol, signal_type, confidence, source, trade_executed, ticket_id)
                            VALUES (?, ?, ?, ?, 'execution_report', 1, ?)
                        """, (timestamp, symbol, signal_trigger, confidence, ticket))
                        imported += 1
                    except Exception as e:
                        pass
        
        conn.commit()
        conn.close()
        print(f"✅ Imported {imported} signals from execution_reports.json")
        return imported
    
    def import_signals_from_analysis_results(self, days_back: int = 30):
        """Import signals từ analysis_results/*_signal_*.json"""
        
        signal_files = glob.glob(os.path.join(ANALYSIS_DIR, "*_signal_*.json"))
        print(f"📁 Found {len(signal_files)} signal files")
        
        imported = 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for signal_file in signal_files:
            try:
                with open(signal_file, 'r', encoding='utf-8') as f:
                    signal_data = json.load(f)
                
                # Extract data
                symbol = signal_data.get('symbol', '')
                timestamp = signal_data.get('timestamp', '')
                final_signal = signal_data.get('final_signal', {})
                
                signal_type = final_signal.get('signal', 'HOLD')
                entry = final_signal.get('entry', 0)
                sl = final_signal.get('stoploss', 0)
                tp = final_signal.get('takeprofit', 0)
                confidence = final_signal.get('confidence', 50)
                
                # Get indicators
                ind_summary = signal_data.get('indicator_summary', {})
                rsi = ind_summary.get('RSI14', ind_summary.get('rsi_14'))
                macd = ind_summary.get('MACD', ind_summary.get('macd'))
                ema20 = ind_summary.get('EMA20', ind_summary.get('ema_20'))
                ema50 = ind_summary.get('EMA50', ind_summary.get('ema_50'))
                adx = ind_summary.get('ADX14', ind_summary.get('adx'))
                atr = ind_summary.get('ATR14', ind_summary.get('atr_14'))
                
                source = signal_data.get('logic_type', 'comprehensive_aggregator')
                
                # Insert
                cursor.execute("""
                    INSERT OR IGNORE INTO signals 
                    (timestamp, symbol, signal_type, entry_price, stoploss, takeprofit, confidence,
                     rsi, macd, ema_20, ema_50, adx, atr, source, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp, symbol, signal_type, entry, sl, tp, confidence,
                    rsi, macd, ema20, ema50, adx, atr, source, json.dumps(signal_data)
                ))
                imported += 1
                
            except Exception as e:
                print(f"⚠️ Error processing {signal_file}: {e}")
        
        conn.commit()
        conn.close()
        print(f"✅ Imported {imported} signals from analysis_results/")
        return imported
    
    # ============================================
    # SAVE NEW DATA
    # ============================================
    
    def save_signal(self, 
                    symbol: str,
                    signal_type: str,
                    entry_price: float,
                    stoploss: float,
                    takeprofit: float,
                    confidence: float,
                    indicators: Dict,
                    patterns: Dict = None,
                    source: str = "comprehensive_aggregator") -> int:
        """Lưu signal mới vào database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract indicators
        rsi = indicators.get('rsi_14') or indicators.get('RSI14') or indicators.get('RSI')
        macd = indicators.get('macd') or indicators.get('MACD')
        macd_signal = indicators.get('macd_signal') or indicators.get('MACD_Signal')
        ema_20 = indicators.get('ema_20') or indicators.get('EMA20')
        ema_50 = indicators.get('ema_50') or indicators.get('EMA50')
        adx = indicators.get('adx') or indicators.get('ADX') or indicators.get('ADX14')
        atr = indicators.get('atr_14') or indicators.get('ATR14') or indicators.get('ATR')
        current_price = indicators.get('close') or indicators.get('current_price') or entry_price
        
        # Extract patterns
        candle_patterns = json.dumps(patterns.get('candle_patterns', [])) if patterns else "[]"
        price_patterns = json.dumps(patterns.get('price_patterns', [])) if patterns else "[]"
        support_levels = json.dumps(patterns.get('support_levels', [])) if patterns else "[]"
        resistance_levels = json.dumps(patterns.get('resistance_levels', [])) if patterns else "[]"
        trend_direction = patterns.get('trend_direction', 'UNKNOWN') if patterns else 'UNKNOWN'
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO signals (
                    timestamp, symbol, signal_type, entry_price, stoploss, takeprofit, confidence,
                    rsi, macd, macd_signal, ema_20, ema_50, adx, atr, current_price,
                    candle_patterns, price_patterns, support_levels, resistance_levels, trend_direction,
                    source, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, symbol, signal_type, entry_price, stoploss, takeprofit, confidence,
                rsi, macd, macd_signal, ema_20, ema_50, adx, atr, current_price,
                candle_patterns, price_patterns, support_levels, resistance_levels, trend_direction,
                source, json.dumps(indicators)
            ))
            
            signal_id = cursor.lastrowid
            conn.commit()
            print(f"✅ Signal saved: {symbol} {signal_type} @ {entry_price} (ID: {signal_id})")
            return signal_id
            
        except Exception as e:
            print(f"❌ Error saving signal: {e}")
            return -1
        finally:
            conn.close()
    
    def save_trade_result(self,
                          ticket_id: int,
                          symbol: str,
                          trade_type: str,
                          entry_time: str,
                          entry_price: float,
                          volume: float,
                          exit_time: str,
                          exit_price: float,
                          exit_reason: str,
                          profit_money: float,
                          profit_pips: float,
                          duration_minutes: int = 0,
                          swap: float = 0,
                          commission: float = 0,
                          indicators_at_entry: Dict = None,
                          signal_id: int = None) -> int:
        """Lưu kết quả trade"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Determine outcome
        if profit_pips > 5:
            outcome = "WIN"
            quality_score = min(100, profit_pips)
        elif profit_pips < -5:
            outcome = "LOSS"
            quality_score = max(-100, profit_pips)
        else:
            outcome = "BREAKEVEN"
            quality_score = 0
        
        # Extract RSI from indicators if available
        rsi_at_entry = None
        if indicators_at_entry:
            rsi_at_entry = indicators_at_entry.get('RSI14') or indicators_at_entry.get('rsi_14')
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO trade_results (
                    signal_id, ticket_id, symbol, trade_type,
                    entry_time, entry_price, volume,
                    exit_time, exit_price, exit_reason,
                    profit_money, profit_pips, duration_minutes,
                    swap, commission, rsi_at_entry, outcome, quality_score,
                    indicators_at_entry
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id, ticket_id, symbol, trade_type,
                entry_time, entry_price, volume,
                exit_time, exit_price, exit_reason,
                profit_money, profit_pips, duration_minutes,
                swap, commission, rsi_at_entry, outcome, quality_score,
                json.dumps(indicators_at_entry) if indicators_at_entry else None
            ))
            
            result_id = cursor.lastrowid
            conn.commit()
            print(f"✅ Trade result saved: {symbol} {outcome} {profit_pips:.1f} pips (ID: {result_id})")
            return result_id
            
        except Exception as e:
            print(f"❌ Error saving trade result: {e}")
            return -1
        finally:
            conn.close()
    
    def save_execution(self,
                       timestamp: str,
                       action_type: str,
                       symbol: str,
                       ticket: int = None,  # Can be None if position not found
                       success: bool = False,
                       result_message: str = "",
                       details: Dict = None) -> int:
        """
        Lưu execution action (S/L modifications, closes, opens, etc.)
        Đây là data quan trọng để AI học cách thực hiện hành động
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract common fields from details
        details = details or {}
        old_sl = details.get('old_sl')
        new_sl = details.get('new_sl')
        old_tp = details.get('old_tp')
        new_tp = details.get('new_tp')
        entry_price = details.get('entry_price')
        exit_price = details.get('exit_price')
        profit_pips = details.get('profit_pips')
        profit_usd = details.get('profit_usd')
        reason = details.get('reason', '')
        confidence = details.get('confidence')
        outcome = details.get('outcome', '')
        
        try:
            cursor.execute("""
                INSERT INTO executions (
                    timestamp, action_type, symbol, ticket, success, result_message,
                    details, old_sl, new_sl, old_tp, new_tp,
                    entry_price, exit_price, profit_pips, profit_usd,
                    reason, confidence, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, action_type, symbol, ticket, 1 if success else 0, result_message,
                json.dumps(details), old_sl, new_sl, old_tp, new_tp,
                entry_price, exit_price, profit_pips, profit_usd,
                reason, confidence, outcome
            ))
            
            exec_id = cursor.lastrowid
            conn.commit()
            
            status = "✅" if success else "❌"
            print(f"{status} Execution saved: {action_type} {symbol} #{ticket} (ID: {exec_id})")
            return exec_id
            
        except Exception as e:
            print(f"❌ Error saving execution: {e}")
            return -1
        finally:
            conn.close()
    
    # ============================================
    # ADD TRAINING EXAMPLE DIRECTLY
    # ============================================
    
    def add_training_example(self, input_text: str, output_text: str, 
                            source: str = 'manual', quality_score: float = 0.8,
                            symbol: str = '', signal_type: str = '') -> int:
        """
        Add a training example directly to the database
        Used for prompt-based training data generation
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO training_examples 
                (created_at, input_prompt, expected_output, outcome, symbol, used_in_training)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (
                datetime.now().isoformat(),
                input_text,
                output_text,
                signal_type,
                symbol
            ))
            
            example_id = cursor.lastrowid
            conn.commit()
            return example_id
            
        except Exception as e:
            print(f"❌ Error adding training example: {e}")
            return -1
        finally:
            conn.close()
    
    # ============================================
    # GENERATE TRAINING EXAMPLES
    # ============================================
    
    def generate_training_examples(self):
        """Tạo training examples từ signals + trade results đã có"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get signals with matching trade results
        cursor.execute("""
            SELECT s.id, s.symbol, s.signal_type, s.entry_price, s.confidence,
                   s.rsi, s.macd, s.ema_20, s.ema_50, s.adx, s.atr,
                   t.outcome, t.profit_pips, t.trade_type, t.id as trade_id
            FROM signals s
            LEFT JOIN trade_results t ON s.ticket_id = t.ticket_id
            WHERE t.outcome IS NOT NULL
            AND s.id NOT IN (SELECT signal_id FROM training_examples WHERE signal_id IS NOT NULL)
        """)
        
        rows = cursor.fetchall()
        print(f"📊 Found {len(rows)} signals with trade outcomes to process")
        
        created = 0
        for row in rows:
            signal_id = row[0]
            symbol = row[1]
            signal_type = row[2]
            entry_price = row[3] or 0
            confidence = row[4] or 50
            rsi = row[5]
            macd = row[6]
            ema_20 = row[7]
            ema_50 = row[8]
            adx = row[9]
            atr = row[10]
            outcome = row[11]
            profit_pips = row[12] or 0
            trade_type = row[13]
            trade_id = row[14]
            
            # Build input prompt
            input_prompt = self._build_input_prompt(
                symbol, entry_price, rsi, macd, ema_20, ema_50, adx, atr
            )
            
            # Build expected output based on outcome
            expected_output = self._build_expected_output(
                outcome, trade_type, profit_pips, rsi, macd
            )
            
            # Save training example
            try:
                cursor.execute("""
                    INSERT INTO training_examples 
                    (created_at, input_prompt, expected_output, signal_id, trade_result_id, outcome, profit_pips, symbol)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    input_prompt,
                    expected_output,
                    signal_id,
                    trade_id,
                    outcome,
                    profit_pips,
                    symbol
                ))
                created += 1
            except Exception as e:
                print(f"⚠️ Error creating example: {e}")
        
        conn.commit()
        conn.close()
        print(f"✅ Created {created} training examples")
        return created
    
    def _build_input_prompt(self, symbol: str, price: float, 
                            rsi: float, macd: float, ema20: float, ema50: float,
                            adx: float, atr: float) -> str:
        """Build input prompt cho training"""
        
        # Safe format with defaults
        price_str = f"{price:.5f}" if price else "N/A"
        rsi_str = f"{rsi:.2f}" if rsi else "N/A"
        macd_str = f"{macd:.6f}" if macd else "N/A"
        ema20_str = f"{ema20:.5f}" if ema20 else "N/A"
        ema50_str = f"{ema50:.5f}" if ema50 else "N/A"
        adx_str = f"{adx:.2f}" if adx else "N/A"
        atr_str = f"{atr:.5f}" if atr else "N/A"
        
        prompt = f"""Symbol: {symbol}
Price: {price_str}
RSI(14): {rsi_str}
MACD: {macd_str}
EMA20: {ema20_str}
EMA50: {ema50_str}
ADX: {adx_str}
ATR: {atr_str}"""
        
        return prompt
    
    def _build_expected_output(self, outcome: str, trade_type: str, 
                               profit_pips: float, rsi: float, macd: float) -> str:
        """Build expected output based on actual trade outcome"""
        
        if outcome == "WIN":
            # Trade was correct - reinforce this decision
            confidence = min(90, 70 + abs(profit_pips) / 10)
            reason = f"{trade_type} signal validated with +{profit_pips:.1f} pips"
            signal = trade_type
        elif outcome == "LOSS":
            # Trade was wrong - teach opposite
            confidence = 60
            opposite = "SELL" if trade_type == "BUY" else "BUY"
            
            # Should have done opposite or stayed out
            if abs(profit_pips) > 30:
                signal = opposite
                reason = f"Should have gone {opposite}, {trade_type} lost {profit_pips:.1f} pips"
            else:
                signal = "HOLD"
                reason = f"Should have stayed out, {trade_type} lost {profit_pips:.1f} pips"
        else:
            signal = "HOLD"
            confidence = 50
            reason = "No clear direction"
        
        return json.dumps({
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "learned_from": outcome,
            "original_trade": trade_type,
            "original_pips": profit_pips
        })
    
    # ============================================
    # STATISTICS & EXPORT
    # ============================================
    
    def get_training_stats(self) -> Dict:
        """Lấy thống kê về training data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total signals
        cursor.execute("SELECT COUNT(*) FROM signals")
        stats['total_signals'] = cursor.fetchone()[0]
        
        # Signals by type
        cursor.execute("SELECT signal_type, COUNT(*) FROM signals GROUP BY signal_type")
        stats['signals_by_type'] = dict(cursor.fetchall())
        
        # Total trade results
        cursor.execute("SELECT COUNT(*) FROM trade_results")
        stats['total_trades'] = cursor.fetchone()[0]
        
        # Win/Loss/Breakeven
        cursor.execute("SELECT outcome, COUNT(*) FROM trade_results WHERE outcome IS NOT NULL GROUP BY outcome")
        stats['trades_by_outcome'] = dict(cursor.fetchall())
        
        # Win rate
        wins = stats['trades_by_outcome'].get('WIN', 0)
        total = sum(stats['trades_by_outcome'].values())
        stats['win_rate'] = (wins / total * 100) if total > 0 else 0
        
        # Total profit
        cursor.execute("SELECT SUM(profit_pips) FROM trade_results")
        result = cursor.fetchone()[0]
        stats['total_profit_pips'] = result if result else 0
        
        # Training examples
        cursor.execute("SELECT COUNT(*) FROM training_examples")
        stats['training_examples'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM training_examples WHERE used_in_training = 1")
        stats['used_in_training'] = cursor.fetchone()[0]
        
        # By symbol
        cursor.execute("""
            SELECT symbol, COUNT(*), 
                   SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                   SUM(profit_pips) as total_pips
            FROM trade_results 
            WHERE outcome IS NOT NULL
            GROUP BY symbol
        """)
        stats['by_symbol'] = {}
        for row in cursor.fetchall():
            symbol, count, wins, losses, pips = row
            wr = (wins / count * 100) if count > 0 else 0
            stats['by_symbol'][symbol] = {
                'count': count,
                'wins': wins,
                'losses': losses,
                'win_rate': wr,
                'total_pips': pips or 0
            }
        
        conn.close()
        return stats
    
    def export_training_data(self, output_file: str = None, 
                             min_examples: int = 0) -> List[Dict]:
        """Export training data cho fine-tuning"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, input_prompt, expected_output, outcome, profit_pips, symbol
            FROM training_examples
            WHERE used_in_training = 0
            ORDER BY created_at
        """)
        
        training_data = []
        for row in cursor.fetchall():
            training_data.append({
                "id": row[0],
                "input": row[1],
                "output": row[2],
                "outcome": row[3],
                "profit_pips": row[4],
                "symbol": row[5]
            })
        
        conn.close()
        
        if len(training_data) < min_examples:
            print(f"⚠️ Only {len(training_data)} examples, need {min_examples}")
            return training_data
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Exported {len(training_data)} training examples to {output_file}")
        
        return training_data
    
    def export_for_mistral_finetune(self, output_file: str = None) -> List[Dict]:
        """Export data ở format phù hợp cho Mistral fine-tuning"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT input_prompt, expected_output, outcome, profit_pips
            FROM training_examples
            WHERE used_in_training = 0 AND outcome IN ('WIN', 'LOSS')
            ORDER BY 
                CASE WHEN outcome = 'WIN' THEN 0 ELSE 1 END,
                ABS(profit_pips) DESC
        """)
        
        training_data = []
        for row in cursor.fetchall():
            input_text = row[0]
            output_data = json.loads(row[1])
            
            # Format cho Mistral instruction fine-tuning
            instruction = f"""You are a forex trading AI. Analyze this market data and decide: BUY, SELL, or HOLD.

{input_text}

Respond with JSON only: {{"signal": "BUY/SELL/HOLD", "confidence": 0-100, "reason": "brief"}}"""
            
            response = json.dumps({
                "signal": output_data.get('signal', 'HOLD'),
                "confidence": output_data.get('confidence', 50),
                "reason": output_data.get('reason', 'Analysis')
            })
            
            training_data.append({
                "instruction": instruction,
                "response": response,
                "text": f"<s>[INST] {instruction} [/INST] {response}</s>"
            })
        
        conn.close()
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Exported {len(training_data)} Mistral training examples")
        
        return training_data
    
    def mark_as_used_in_training(self, example_ids: List[int], batch_name: str):
        """Đánh dấu examples đã được dùng để train"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executemany("""
            UPDATE training_examples
            SET used_in_training = 1, training_batch = ?
            WHERE id = ?
        """, [(batch_name, eid) for eid in example_ids])
        
        conn.commit()
        conn.close()
        print(f"✅ Marked {len(example_ids)} examples as used in batch: {batch_name}")

    def get_all_training_examples(self) -> List[Dict]:
        """
        Lấy tất cả training examples với đầy đủ thông tin để export cho PyTorch
        Returns list of dicts với: symbol, signal_type, confidence, indicators, patterns
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Join signals với training_examples để lấy đầy đủ indicators
        cursor.execute("""
            SELECT 
                s.symbol, s.signal_type, s.confidence,
                s.rsi, s.macd, s.macd_signal, s.macd_histogram,
                s.ema_20, s.ema_50, s.ema_200,
                s.adx, s.atr, s.stoch_k, s.stoch_d,
                s.bb_upper, s.bb_middle, s.bb_lower,
                s.current_price,
                s.candle_patterns, s.price_patterns,
                s.support_levels, s.resistance_levels,
                s.trend_direction, s.timestamp,
                te.outcome, te.profit_pips
            FROM signals s
            LEFT JOIN training_examples te ON te.signal_id = s.id
            WHERE s.signal_type IN ('BUY', 'SELL')
            ORDER BY s.timestamp DESC
        """)
        
        results = []
        for row in cursor.fetchall():
            # Parse JSON fields
            candle_patterns = json.loads(row[18]) if row[18] else []
            price_patterns = json.loads(row[19]) if row[19] else []
            support_levels = json.loads(row[20]) if row[20] else []
            resistance_levels = json.loads(row[21]) if row[21] else []
            
            results.append({
                'symbol': row[0],
                'signal_type': row[1],
                'confidence': row[2] or 50,
                'indicators': {
                    'rsi': row[3],
                    'macd': row[4],
                    'macd_signal': row[5],
                    'macd_hist': row[6],
                    'ema_20': row[7],
                    'ema_50': row[8],
                    'ema_200': row[9],
                    'adx': row[10],
                    'atr': row[11],
                    'stoch_k': row[12],
                    'stoch_d': row[13],
                    'bb_upper': row[14],
                    'bb_middle': row[15],
                    'bb_lower': row[16],
                    'close': row[17],
                },
                'patterns': {
                    'candle_patterns': candle_patterns,
                    'price_patterns': price_patterns,
                    'support_levels': support_levels,
                    'resistance_levels': resistance_levels,
                    'trend_direction': row[22] or 'Sideways',
                },
                'timestamp': row[23],
                'outcome': row[24],
                'profit_pips': row[25],
            })
        
        conn.close()
        return results


def print_training_report(stats: Dict):
    """In báo cáo training data"""
    
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                   📊 TRAINING DATA REPORT                        ║
║                   {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Total Signals:          {stats['total_signals']:>6}                              ║
║  Signals by Type:                                                ║"""
    
    for sig_type, count in stats.get('signals_by_type', {}).items():
        report += f"\n║    - {sig_type:<8} {count:>6}                                       ║"
    
    report += f"""
╠══════════════════════════════════════════════════════════════════╣
║  Total Trades:           {stats['total_trades']:>6}                              ║
║  Trade Outcomes:                                                 ║"""
    
    for outcome, count in stats.get('trades_by_outcome', {}).items():
        report += f"\n║    - {outcome:<10} {count:>6}                                     ║"
    
    report += f"""
║  Win Rate:               {stats['win_rate']:>6.1f}%                             ║
║  Total Profit:         {stats['total_profit_pips']:>8.1f} pips                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Training Examples:      {stats['training_examples']:>6}                              ║
║  Used in Training:       {stats['used_in_training']:>6}                              ║
║  Ready for Training:     {stats['training_examples'] - stats['used_in_training']:>6}                              ║
╠══════════════════════════════════════════════════════════════════╣
║  BY SYMBOL:                                                      ║"""
    
    for symbol, data in stats.get('by_symbol', {}).items():
        report += f"\n║    {symbol:<12} {data['count']:>4} trades, {data['win_rate']:>5.1f}% WR, {data['total_pips']:>8.1f} pips ║"
    
    report += """
╚══════════════════════════════════════════════════════════════════╝"""
    
    print(report)


# Singleton instance
_collector = None

def get_collector() -> TrainingDataCollector:
    """Get singleton collector instance"""
    global _collector
    if _collector is None:
        _collector = TrainingDataCollector()
    return _collector


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 TRAINING DATA COLLECTOR")
    print("="*60 + "\n")
    
    collector = get_collector()
    
    # Import existing data
    print("\n📥 Importing existing data...")
    collector.import_from_trading_history()
    collector.import_from_execution_reports()
    collector.import_signals_from_analysis_results()
    
    # Generate training examples
    print("\n🔄 Generating training examples...")
    collector.generate_training_examples()
    
    # Show stats
    print("\n📊 Training Data Statistics:")
    stats = collector.get_training_stats()
    print_training_report(stats)
