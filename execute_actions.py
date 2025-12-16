#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIFIED ACTIONS EXECUTOR
Executes all types of actions from analysis_results/account_positions_actions.json
- Regular trading actions (open/close positions, DCA, etc.)
- Signal-based S/L adjustments (losing positions only)  
- Signal-based T/P adjustments (all positions)
"""

import json
import os
import sys
import logging
from order_executor import get_executor_instance
from mt5_connector import MT5ConnectionManager

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Local Training Data Collector (optional - won't fail if not available)
try:
    from ai_models.training.data_collector_local import get_local_collector
    LOCAL_TRAINING_AVAILABLE = True
except ImportError:
    LOCAL_TRAINING_AVAILABLE = False

# Store executed actions for training
EXECUTED_ACTIONS_LOG = []

def log_executed_action(action_type: str, symbol: str, ticket: int, details: dict, success: bool, result_message: str = ""):
    """
    Log an executed action for AI training (local only)
    """
    import datetime
    
    action_log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "action_type": action_type,
        "symbol": symbol,
        "ticket": ticket,
        "success": success,
        "result_message": result_message,
        "details": details
    }
    
    EXECUTED_ACTIONS_LOG.append(action_log)
    
    # Save to local training data
    if LOCAL_TRAINING_AVAILABLE:
        try:
            collector = get_local_collector()
            collector.log_execution(
                action_type=action_type,
                symbol=symbol,
                ticket=ticket,
                details=details,
                success=success,
                result_message=result_message
            )
        except Exception as e:
            pass  # Silent fail - don't interrupt execution


def collect_training_data_local(actions, execution_results):
    """
    Thu thập và lưu dữ liệu training local
    KHÔNG gửi đến server
    """
    global EXECUTED_ACTIONS_LOG
    
    if not LOCAL_TRAINING_AVAILABLE:
        return
    
    try:
        collector = get_local_collector()
        
        print("\n>> 📊 COLLECTING TRAINING DATA (Local)...")
        print("-" * 50)
        
        # Collect all symbols from actions
        symbols_processed = set()
        for action in actions:
            symbol = action.get('symbol', '')
            if symbol:
                symbols_processed.add(symbol)
        
        # Save training data for each symbol
        records_saved = 0
        for symbol in symbols_processed:
            try:
                # Get action direction for this symbol
                direction = "HOLD"
                confidence = 50
                entry_price = 0
                
                for action in actions:
                    if action.get('symbol') == symbol:
                        action_type = action.get('action', action.get('primary_action', '')).upper()
                        if action_type in ['BUY', 'SELL']:
                            direction = action_type
                        elif action.get('direction'):
                            direction = action.get('direction').upper()
                        confidence = action.get('confidence', action.get('signal_confidence', 50))
                        entry_price = action.get('entry_price', action.get('price', 0))
                        break
                
                # Collect and save training record
                saved = collector.collect_and_save(
                    symbol=symbol,
                    signal_type=direction,
                    confidence=int(confidence),
                    entry_price=float(entry_price) if entry_price else 0,
                    execution_details={
                        'close_result': execution_results.get('close', {}),
                        'sl_tp_result': execution_results.get('sl_tp', {}),
                        'regular_result': execution_results.get('regular', {})
                    }
                )
                
                if saved:
                    records_saved += 1
                    print(f"   ✅ Training data saved for {symbol} ({direction})")
                else:
                    print(f"   ⏭️  Duplicate skipped for {symbol}")
                    
            except Exception as e:
                print(f"   ⚠️ Failed to collect data for {symbol}: {e}")
        
        # Clear the execution log
        EXECUTED_ACTIONS_LOG = []
        
        # Summary
        stats = collector.get_stats()
        print(f"\n   📊 Training Data Summary:")
        print(f"      Records saved: {records_saved}")
        print(f"      Symbols processed: {len(symbols_processed)}")
        print(f"      Total pending files: {stats['pending_training_files']}")
        
    except Exception as e:
        print(f"   ⚠️ Training data collection failed: {e}")

def load_actions():
    """Load actions from JSON file"""
    actions_path = 'analysis_results/account_positions_actions.json'
    if not os.path.exists(actions_path):
        actions_path = os.path.join(os.getcwd(), 'analysis_results/account_positions_actions.json')
        
    if not os.path.exists(actions_path):
        print(f"[ERROR] Actions file not found at: {actions_path}")
        return []
    
    with open(actions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('actions', [])

def categorize_actions(actions):
    """Categorize actions by type"""
    sl_tp_actions = []
    regular_actions = []
    close_actions = []
    
    for action in actions:
        # Check action type from multiple possible fields
        action_type = action.get('action') or action.get('primary_action') or action.get('action_type')
        
        # 🛡️ Skip pure hold actions without ticket or SL/TP modifications
        if action_type == 'hold':
            has_sl_tp = (action.get('new_sl') or action.get('proposed_sl') or action.get('move_sl_price') or
                         action.get('new_tp') or action.get('proposed_tp') or action.get('move_tp_price'))
            if not has_sl_tp:
                # Pure hold - no action needed, skip entirely
                continue
        
        # Old format: action='modify_position' with type='sl_adjustment_signal_based'
        if (action.get('action') == 'modify_position' and 
            action.get('type') in ['sl_adjustment_signal_based', 'tp_adjustment_signal_based']):
            sl_tp_actions.append(action)
        # New format: primary_action='set_sl' or 'set_tp' or 'move_sl_to_be' or 'move_sl'
        elif action_type in ['set_sl', 'set_tp', 'move_sl_to_be', 'move_sl', 'set_trailing_sl']:
            sl_tp_actions.append(action)
        # Close actions: opposite signal or close_full
        elif (action.get('action') == 'close_position' and 
              ('opposite' in action.get('type', '').lower() or 'close_full' in action.get('type', '').lower())):
            close_actions.append(action)
        elif action.get('action') == 'close_opposite_signal' or action_type == 'close_full':
            close_actions.append(action)
        else:
            regular_actions.append(action)
    
    return sl_tp_actions, regular_actions, close_actions

def deduplicate_actions_execution(actions):
    """
    🚨 ENHANCED: Final deduplication before execution
    Removes duplicate actions that could cause double orders
    """
    if not actions:
        return actions
    
    deduped = []
    seen_signatures = set()
    
    for action in actions:
        try:
            # Create signature for duplicate detection
            symbol = action.get('symbol', 'UNKNOWN')
            action_type = action.get('action_type', action.get('action', action.get('primary_action', 'UNKNOWN')))
            direction = action.get('direction', action.get('position_type', 'UNKNOWN'))
            entry_price = action.get('entry_price', action.get('price', 0))
            volume = action.get('volume', 0)
            
            # Round price to avoid minor price differences causing duplicates
            price_rounded = round(float(entry_price), 5) if entry_price else 0
            
            signature = f"{symbol}_{action_type}_{direction}_{price_rounded}_{volume}"
            
            if signature in seen_signatures:
                print(f"🚫 EXECUTION DUPLICATE REMOVED: {signature}")
                continue
            
            seen_signatures.add(signature)
            deduped.append(action)
            
        except Exception as e:
            # Keep action if signature creation fails
            print(f"⚠️ Signature creation failed for action: {e}")
            deduped.append(action)
    
    removed = len(actions) - len(deduped)
    if removed > 0:
        print(f"🚨 EXECUTION DEDUPLICATION: Removed {removed} duplicate actions")
    
    return deduped

def show_actions_summary(sl_tp_actions, regular_actions, close_actions=None):
    """Show summary of all actions before execution"""
    close_actions = close_actions or []
    total_actions = len(sl_tp_actions) + len(regular_actions) + len(close_actions)
    
    if total_actions == 0:
        print("[ERROR] No actions found to execute")
        return False
    
    print("� ACTIONS TO EXECUTE:")
    print("=" * 50)
    
    if sl_tp_actions:
        # Count by both old and new action type formats
        sl_count = len([a for a in sl_tp_actions if (a.get('type') == 'sl_adjustment_signal_based' or 
                                                      a.get('primary_action') in ['set_sl', 'move_sl_to_be', 'move_sl', 'set_trailing_sl'])])
        tp_count = len([a for a in sl_tp_actions if (a.get('type') == 'tp_adjustment_signal_based' or 
                                                      a.get('primary_action') == 'set_tp')])
        print(f">> SIGNAL-BASED ADJUSTMENTS ({len(sl_tp_actions)} actions):")
        print(f"   - S/L Adjustments: {sl_count}")
        print(f"   - T/P Adjustments: {tp_count}")
        
        for i, action in enumerate(sl_tp_actions, 1):
            symbol = action.get('symbol')
            ticket = action.get('ticket')
            pos_type = action.get('position_type') or action.get('direction')
            
            # Get action type from multiple fields
            action_type = action.get('type') or action.get('primary_action') or action.get('action_type', 'unknown')
            
            # Get S/L T/P values
            new_sl = action.get('new_sl') or action.get('proposed_sl') or action.get('move_sl_price')
            new_tp = action.get('new_tp') or action.get('proposed_tp') or action.get('move_tp_price')
            current_sl = action.get('current_sl', 0)
            current_tp = action.get('current_tp', 0)
            
            # Get reason
            reason = action.get('reason') or action.get('rationale', '')
            
            print(f"   {i}. {symbol} {pos_type} #{ticket} ({action_type})")
            if action_type in ['sl_adjustment_signal_based', 'set_sl', 'move_sl_to_be', 'move_sl', 'set_trailing_sl']:
                if new_sl:
                    print(f"      S/L: {current_sl} → {new_sl}")
            elif action_type in ['tp_adjustment_signal_based', 'set_tp']:
                if new_tp:
                    print(f"      T/P: {current_tp} → {new_tp}")
            if reason:
                print(f"      Reason: {reason}")
        print()
    
    if close_actions:
        print(f">> OPPOSITE SIGNAL CLOSE ACTIONS ({len(close_actions)} actions):")
        for i, action in enumerate(close_actions, 1):
            symbol = action.get('symbol', 'N/A')
            ticket = action.get('ticket')
            close_type = action.get('close_type', 'full')
            reason = action.get('reason', '')
            confidence = action.get('confidence', 0)
            print(f"   {i}. {symbol} #{ticket} ({close_type} close)")
            print(f"      Confidence: {confidence:.1f}%")
            print(f"      Reason: {reason}")
        print()
    
    if regular_actions:
        print(f">> REGULAR TRADING ACTIONS ({len(regular_actions)} actions):")
        for i, action in enumerate(regular_actions, 1):
            # Get action type with fallback to primary_action, then current_signal
            action_type = action.get('action_type', action.get('action', action.get('primary_action', 'unknown')))
            symbol = action.get('symbol', 'N/A')
            current_signal = action.get('current_signal', '')
            direction = action.get('direction', '')
            
            # Show more meaningful info
            if action_type == 'hold' and current_signal:
                display_action = f"{action_type} ({current_signal} signal)"
            elif direction:
                display_action = f"{action_type} ({direction})"
            else:
                display_action = action_type
                
            print(f"   {i}. {symbol} - {display_action}")
        print()
    
    print(f">> TOTAL: {total_actions} actions ({len(sl_tp_actions)} adjustments + {len(close_actions)} closes + {len(regular_actions)} regular)")
    return True

def execute_sl_tp_adjustments(sl_tp_actions):
    """Execute S/L and T/P adjustment actions"""
    if not sl_tp_actions:
        return {'success': 0, 'failed': 0}
    
    print(">> EXECUTING SIGNAL-BASED S/L & T/P ADJUSTMENTS...")
    print("-" * 50)
    
    # Use existing executor instance and ensure MT5 connection
    try:
        executor = get_executor_instance()
        print("[OK] Using global executor instance for S/L & T/P adjustments")
        
        # 🔧 CRITICAL FIX: Ensure MT5 is connected before modifications
        # Import directly and check if MT5 terminal is available
        import MetaTrader5 as MT5
        
        # Try to initialize MT5 if not already connected
        terminal_info = MT5.terminal_info()
        if not terminal_info:
            print("[INFO] MT5 not initialized, attempting to connect...")
            if MT5.initialize():
                print("[OK] MT5 initialized successfully")
                terminal_info = MT5.terminal_info()
            else:
                error = MT5.last_error()
                print(f"[ERROR] Failed to initialize MT5: {error}")
                print("[ERROR] Make sure MT5 terminal is running and logged in")
                return {'success': 0, 'failed': len(sl_tp_actions)}
        
        # Verify we can get account info
        account_info = MT5.account_info()
        if not account_info:
            print("[ERROR] Cannot get MT5 account info - check terminal login")
            return {'success': 0, 'failed': len(sl_tp_actions)}
        
        print(f"[OK] MT5 connected - Account: {account_info.login}, Balance: {account_info.balance:.2f}")
        
        # Store MT5 connection for executor to use
        # Even though executor has connection_manager, we use MT5 directly for modifications
        if not executor.connection:
            # Create a simple connection wrapper
            from mt5_connector import MT5ConnectionManager
            try:
                mt5_mgr = MT5ConnectionManager()
                if mt5_mgr.connect():
                    executor.connection = mt5_mgr
                    print("[OK] Created MT5ConnectionManager wrapper")
            except Exception as e:
                print(f"[WARNING] Could not create connection manager wrapper: {e}")
                print("[OK] Will use direct MT5 calls instead")
        
    except Exception as e:
        print(f"[ERROR] Failed to get executor instance for adjustments: {e}")
        import traceback
        traceback.print_exc()
        return {'success': 0, 'failed': len(sl_tp_actions)}
    
    success_count = 0
    skipped_count = 0
    
    for i, action in enumerate(sl_tp_actions, 1):
        symbol = action.get('symbol')
        ticket = action.get('ticket')
        
        # Handle both old and new action type formats
        action_type = action.get('type') or action.get('primary_action') or action.get('action_type', 'unknown')
        
        # Get S/L T/P values from multiple possible fields
        new_sl = action.get('new_sl') or action.get('proposed_sl') or action.get('move_sl_price')
        new_tp = action.get('new_tp') or action.get('proposed_tp') or action.get('move_tp_price')
        current_sl = action.get('current_sl', 0)
        current_tp = action.get('current_tp', 0)
        
        # Get reason/rationale
        reason = action.get('reason') or action.get('rationale', 'Signal-based adjustment')
        confidence = action.get('confidence') or action.get('signal_confidence', 0)
        
        # 🛡️ SKIP: Actions without ticket (cannot modify without position reference)
        if not ticket:
            print(f"{i}. >> {symbol} [SKIP] No ticket - cannot modify position")
            skipped_count += 1
            continue
        
        # 🛡️ SKIP: Pure hold actions with no SL/TP changes (nothing to execute)
        if action_type == 'hold' and new_sl is None and new_tp is None:
            print(f"{i}. >> {symbol} #{ticket} [SKIP] Hold action - no modification needed")
            skipped_count += 1
            continue
        
        print(f"{i}. >> {symbol} #{ticket} ({action_type}):")
        
        try:
            # Old format: sl_adjustment_signal_based, tp_adjustment_signal_based
            # New format: set_sl, set_tp, move_sl_to_be, move_sl, set_trailing_sl
            
            if action_type in ['sl_adjustment_signal_based', 'set_sl', 'move_sl_to_be', 'move_sl', 'set_trailing_sl']:
                if new_sl is None:
                    print(f"   [ERROR] No S/L value provided")
                    log_executed_action('modify_sl', symbol, ticket, {
                        'action_type': action_type, 'reason': reason, 'error': 'No S/L value provided'
                    }, False, 'No S/L value provided')
                    continue
                
                # 🎯 Check if we also need to update T/P
                tp_to_set = new_tp if new_tp is not None else None
                    
                result = executor.modify_order(ticket=ticket, sl=new_sl, tp=tp_to_set)
                if result.success:
                    print(f"   [OK] S/L updated to {new_sl}")
                    if tp_to_set is not None:
                        print(f"   [OK] T/P also updated to {tp_to_set}")
                    print(f"   Reason: {reason}")
                    success_count += 1
                    
                    # 🆕 Log to AI training
                    log_executed_action('modify_sl', symbol, ticket, {
                        'action_type': action_type,
                        'old_sl': current_sl,
                        'new_sl': new_sl,
                        'old_tp': current_tp,
                        'new_tp': tp_to_set,
                        'reason': reason,
                        'confidence': confidence
                    }, True, f'S/L updated: {current_sl} → {new_sl}')
                else:
                    print(f"   [ERROR] S/L update failed: {result.error_message}")
                    log_executed_action('modify_sl', symbol, ticket, {
                        'action_type': action_type,
                        'old_sl': current_sl,
                        'new_sl': new_sl,
                        'reason': reason,
                        'error': result.error_message
                    }, False, result.error_message)
                    
            elif action_type in ['tp_adjustment_signal_based', 'set_tp']:
                if new_tp is None:
                    print(f"   [ERROR] No T/P value provided")
                    log_executed_action('modify_tp', symbol, ticket, {
                        'action_type': action_type, 'reason': reason, 'error': 'No T/P value provided'
                    }, False, 'No T/P value provided')
                    continue
                    
                result = executor.modify_order(ticket=ticket, sl=None, tp=new_tp)
                if result.success:
                    print(f"   [OK] T/P updated to {new_tp}")
                    print(f"   Reason: {reason}")
                    success_count += 1
                    
                    # 🆕 Log to AI training
                    log_executed_action('modify_tp', symbol, ticket, {
                        'action_type': action_type,
                        'old_tp': current_tp,
                        'new_tp': new_tp,
                        'reason': reason,
                        'confidence': confidence
                    }, True, f'T/P updated: {current_tp} → {new_tp}')
                else:
                    print(f"   [ERROR] T/P update failed: {result.error_message}")
                    log_executed_action('modify_tp', symbol, ticket, {
                        'action_type': action_type,
                        'old_tp': current_tp,
                        'new_tp': new_tp,
                        'reason': reason,
                        'error': result.error_message
                    }, False, result.error_message)
                    
            elif action_type == 'hold' and new_tp is not None:
                # 🎯 Special case: hold action but need to update T/P based on new signal
                result = executor.modify_order(ticket=ticket, sl=None, tp=new_tp)
                if result.success:
                    print(f"   [OK] T/P updated to {new_tp} (hold action with T/P adjustment)")
                    print(f"   Reason: {reason}")
                    success_count += 1
                    
                    # 🆕 Log to AI training
                    log_executed_action('modify_tp_hold', symbol, ticket, {
                        'action_type': 'hold_with_tp_adjustment',
                        'old_tp': current_tp,
                        'new_tp': new_tp,
                        'reason': reason,
                        'confidence': confidence
                    }, True, f'T/P updated on hold: {current_tp} → {new_tp}')
                else:
                    print(f"   [ERROR] T/P update failed: {result.error_message}")
                    log_executed_action('modify_tp_hold', symbol, ticket, {
                        'action_type': 'hold_with_tp_adjustment',
                        'old_tp': current_tp,
                        'new_tp': new_tp,
                        'reason': reason,
                        'error': result.error_message
                    }, False, result.error_message)
            else:
                print(f"   [WARNING] Unknown action type: {action_type}")
                    
        except Exception as e:
            print(f"   [ERROR] Execution error: {e}")
    
    failed_count = len(sl_tp_actions) - success_count - skipped_count
    if skipped_count > 0:
        print(f">> Skipped {skipped_count} actions (no ticket or pure hold)")
    return {'success': success_count, 'failed': failed_count, 'skipped': skipped_count}

def execute_regular_actions(regular_actions):
    """Execute regular trading actions using AdvancedOrderExecutor"""
    if not regular_actions:
        return {'applied': 0, 'errors': [], 'skipped': []}
    
    print(">> EXECUTING REGULAR TRADING ACTIONS...")
    print("-" * 50)
    
    try:
        executor = get_executor_instance()
        # � NOTE: Regular actions are processed by reading the full JSON file
        # Volume calculations are handled by order_executor.py based on risk settings
        result = executor.apply_actions_from_json()
        print("[OK] Regular actions processed by AdvancedOrderExecutor")
        print(f"     Volume calculation: Handled by order_executor.py risk settings")
        return result
        
    except Exception as e:
        print(f"[ERROR] Regular actions execution error: {e}")
        return {'applied': 0, 'errors': [str(e)], 'skipped': []}

def execute_close_actions(close_actions):
    """Execute opposite signal close actions using AdvancedOrderExecutor"""
    if not close_actions:
        return {'success': 0, 'failed': 0}
    
    print(">> EXECUTING OPPOSITE SIGNAL CLOSE ACTIONS...")
    print("-" * 50)
    
    try:
        # Use AdvancedOrderExecutor singleton instead of MT5ConnectionManager
        from order_executor import get_executor_instance
        executor = get_executor_instance()
        
        success_count = 0
        failed_count = 0
        
        for i, action in enumerate(close_actions, 1):
            try:
                symbol = action.get('symbol', '')
                ticket = action.get('ticket')
                close_type = action.get('close_type', 'full')
                volume = action.get('volume', 0.0)
                reason = action.get('reason', '')
                confidence = action.get('confidence', 0)
                profit_pips = action.get('current_profit_pips', 0)
                profit_usd = action.get('current_profit_usd', 0)
                entry_price = action.get('entry_price', 0)
                current_price = action.get('current_price', 0)
                position_type = action.get('position_type', action.get('type', ''))
                
                print(f"[{i}/{len(close_actions)}] Closing {symbol} #{ticket} ({close_type})...")
                print(f"    Reason: {reason}")
                print(f"    Profit: +{profit_pips:.1f} pips / ${profit_usd:.2f}")
                print(f"    Confidence: {confidence:.1f}%")
                
                # Use AdvancedOrderExecutor's close_position method
                if close_type == 'partial' and volume > 0:
                    result = executor.close_position(ticket=ticket, volume=volume)
                else:
                    result = executor.close_position(ticket=ticket)
                
                if result.success:
                    print(f"    [SUCCESS] Position closed successfully")
                    print(f"    Comment: {result.comment}")
                    success_count += 1
                    
                    # 🆕 Log to AI training - CLOSE POSITION SUCCESS
                    log_executed_action('close_position', symbol, ticket, {
                        'close_type': close_type,
                        'position_type': position_type,
                        'volume': volume,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pips': profit_pips,
                        'profit_usd': profit_usd,
                        'reason': reason,
                        'confidence': confidence,
                        'outcome': 'WIN' if profit_pips > 0 else 'LOSS' if profit_pips < 0 else 'BREAKEVEN'
                    }, True, f'Closed {position_type} @ {current_price}, P/L: {profit_pips:.1f} pips')
                else:
                    error_msg = result.error_message or result.comment or "Unknown error"
                    print(f"    [ERROR] Close failed: {error_msg}")
                    failed_count += 1
                    
                    # 🆕 Log to AI training - CLOSE POSITION FAILED
                    log_executed_action('close_position', symbol, ticket, {
                        'close_type': close_type,
                        'position_type': position_type,
                        'volume': volume,
                        'reason': reason,
                        'error': error_msg
                    }, False, error_msg)
                
            except Exception as e:
                print(f"    [ERROR] Exception closing position: {str(e)}")
                failed_count += 1
                
                # 🆕 Log exception
                log_executed_action('close_position', symbol, ticket, {
                    'close_type': close_type,
                    'reason': reason,
                    'error': str(e)
                }, False, str(e))
            
            print()
        
        print(f"[CLOSE SUMMARY] Success: {success_count}, Failed: {failed_count}")
        return {'success': success_count, 'failed': failed_count}
        
    except Exception as e:
        print(f"[ERROR] Close actions execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': 0, 'failed': len(close_actions), 'message': str(e)}

def main():
    """Main execution function"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print(">> UNIFIED ACTIONS EXECUTOR")
    print("=" * 60)
    
    # Load and categorize actions
    all_actions = load_actions()
    
    # 🚨 ENHANCED: Deduplicate actions before execution
    all_actions = deduplicate_actions_execution(all_actions)
    
    sl_tp_actions, regular_actions, close_actions = categorize_actions(all_actions)
    
    # Show summary
    if not show_actions_summary(sl_tp_actions, regular_actions, close_actions):
        return False
    
    # Check for auto-execution mode (when called from other scripts)
    auto_execute = len(sys.argv) > 1 and ('--auto' in sys.argv or '--yes' in sys.argv or '-y' in sys.argv)
    
    if auto_execute:
        print(">> AUTO-EXECUTION MODE: Proceeding without confirmation")
    else:
        # Ask for confirmation in interactive mode
        try:
            response = input("Execute all these actions? [y/N]: ").strip().lower()
            
            if response not in ['y', 'yes']:
                print("[CANCELLED] Execution cancelled by user")
                return False
        except (EOFError, KeyboardInterrupt):
            # Handle cases where input is not available (auto-execution)
            print(">> AUTO-EXECUTION: No input available, proceeding automatically")
            auto_execute = True
    
    print("\n>> STARTING EXECUTION...")
    print("=" * 60)
    
    # Execute close actions first (highest priority)
    close_result = execute_close_actions(close_actions)
    
    # Execute S/L & T/P adjustments
    sl_tp_result = execute_sl_tp_adjustments(sl_tp_actions)
    
    # Execute regular actions
    regular_result = execute_regular_actions(regular_actions)
    
    # 📊 COLLECT TRAINING DATA (Local only - no server)
    execution_results = {
        'close': close_result,
        'sl_tp': sl_tp_result,
        'regular': regular_result
    }
    collect_training_data_local(all_actions, execution_results)
    
    # Summary
    print("\n" + "=" * 60)
    print(">> EXECUTION SUMMARY:")
    print("-" * 60)
    
    # Close Actions Summary
    close_total = close_result['success'] + close_result['failed']
    if close_total > 0:
        print(f">> Opposite Signal Closes:")
        print(f"   Total: {close_total}")
        print(f"   Success: {close_result['success']}")
        print(f"   Failed: {close_result['failed']}")
        print(f"   Success rate: {close_result['success']/close_total*100:.1f}%")
    
    # S/L & T/P Summary
    sl_tp_total = sl_tp_result['success'] + sl_tp_result['failed']
    if sl_tp_total > 0:
        print(f">> Signal-based Adjustments:")
        print(f"   Total: {sl_tp_total}")
        print(f"   Successful: {sl_tp_result['success']}")
        print(f"   Failed: {sl_tp_result['failed']}")
        print(f"   Success rate: {sl_tp_result['success']/sl_tp_total*100:.1f}%")
    
    # Regular Actions Summary  
    regular_applied = regular_result.get('applied', 0)
    regular_errors = len(regular_result.get('errors', []))
    regular_skipped = len(regular_result.get('skipped', []))
    regular_total = regular_applied + regular_errors + regular_skipped
    
    if regular_total > 0:
        print(f">> Regular Actions:")
        print(f"   Total: {regular_total}")
        print(f"   Applied: {regular_applied}")
        print(f"   Failed: {regular_errors}")
        print(f"   Skipped: {regular_skipped}")
        
        # 🔍 DETAILED SKIP DEBUG
        if regular_skipped > 0:
            print("\n" + "="*80)
            print("🔍 DETAILED SKIP REASONS:")
            print("="*80)
            for skip_item in regular_result.get('skipped', []):
                print(f"\n❌ Symbol: {skip_item.get('symbol', 'Unknown')}")
                print(f"   Action: {skip_item.get('action', 'Unknown')}")
                print(f"   Reason: {skip_item.get('reason', 'No reason given')}")
                details = skip_item.get('details', '')
                if details:
                    print(f"   Details: {details}")
            print("="*80 + "\n")
    
    # Overall Summary - FIX: Include processed actions even if they are hold actions
    total_success = sl_tp_result['success'] + regular_applied + close_result['success']
    total_processed = sl_tp_result.get('success', 0) + (regular_total - regular_errors) + close_result.get('success', 0)
    total_actions = sl_tp_total + regular_total + close_total
    
    print(f">> OVERALL:")
    print(f"   Total actions: {total_actions}")
    print(f"   Total successful: {total_success}")
    print(f"   Total processed: {total_processed}")  # Include hold actions
    print(f"   Overall success rate: {total_success/total_actions*100:.1f}%" if total_actions > 0 else "   No actions processed")
    
    # FIX: Consider execution successful if at least 1 action succeeded
    # Previously required 80% success rate which was too strict for small action sets
    execution_successful = (total_actions > 0) and (total_success > 0)
    
    if execution_successful:
        print("\n>> EXECUTION COMPLETED!")
        print(f">> {total_success}/{total_actions} actions successful")
        print(">> Check MT5 platform to verify changes")
    else:
        print("\n>> All actions failed or were skipped")
        
    return execution_successful

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)