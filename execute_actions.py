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
        if (action.get('action') == 'modify_position' and 
            action.get('type') in ['sl_adjustment_signal_based', 'tp_adjustment_signal_based']):
            sl_tp_actions.append(action)
        elif action.get('action') == 'close_opposite_signal':
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
            action_type = action.get('action', action.get('primary_action', 'UNKNOWN'))
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
        sl_count = len([a for a in sl_tp_actions if a.get('type') == 'sl_adjustment_signal_based'])
        tp_count = len([a for a in sl_tp_actions if a.get('type') == 'tp_adjustment_signal_based'])
        print(f">> SIGNAL-BASED ADJUSTMENTS ({len(sl_tp_actions)} actions):")
        print(f"   - S/L Adjustments (losing positions): {sl_count}")
        print(f"   - T/P Adjustments (all positions): {tp_count}")
        
        for i, action in enumerate(sl_tp_actions, 1):
            symbol = action.get('symbol')
            ticket = action.get('ticket')
            pos_type = action.get('position_type')
            action_type = action.get('type', 'unknown')
            new_sl = action.get('new_sl')
            new_tp = action.get('new_tp')
            current_sl = action.get('current_sl', 0)
            current_tp = action.get('current_tp', 0)
            reason = action.get('reason', '')
            
            print(f"   {i}. {symbol} {pos_type} #{ticket} ({action_type})")
            if action_type == 'sl_adjustment_signal_based':
                print(f"      S/L: {current_sl} → {new_sl}")
            elif action_type == 'tp_adjustment_signal_based':
                print(f"      T/P: {current_tp} → {new_tp}")
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
            action_type = action.get('action', action.get('primary_action', 'unknown'))
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
    
    # Use existing executor instance (already has MT5 connection)
    try:
        executor = get_executor_instance()
        print("[OK] Using global executor instance for S/L & T/P adjustments")
        
    except Exception as e:
        print(f"[ERROR] Failed to get executor instance for adjustments: {e}")
        return {'success': 0, 'failed': len(sl_tp_actions)}
    
    success_count = 0
    
    for i, action in enumerate(sl_tp_actions, 1):
        symbol = action.get('symbol')
        ticket = action.get('ticket')
        action_type = action.get('type', 'unknown')
        new_sl = action.get('new_sl')
        new_tp = action.get('new_tp')
        reason = action.get('reason', 'Signal-based adjustment')
        confidence = action.get('confidence', 0)
        
        print(f"{i}. >> {symbol} #{ticket} ({action_type}):")
        
        try:
            if action_type == 'sl_adjustment_signal_based':
                result = executor.modify_order(ticket=ticket, sl=new_sl, tp=None)
                if result.success:
                    print(f"   [OK] S/L updated to {new_sl}")
                    success_count += 1
                else:
                    print(f"   [ERROR] S/L update failed: {result.error_message}")
                    
            elif action_type == 'tp_adjustment_signal_based':
                result = executor.modify_order(ticket=ticket, sl=None, tp=new_tp)
                if result.success:
                    print(f"   [OK] T/P updated to {new_tp}")
                    success_count += 1
                else:
                    print(f"   [ERROR] T/P update failed: {result.error_message}")
                    
        except Exception as e:
            print(f"   [ERROR] Execution error: {e}")
    
    failed_count = len(sl_tp_actions) - success_count
    return {'success': success_count, 'failed': failed_count}

def execute_regular_actions(regular_actions):
    """Execute regular trading actions using AdvancedOrderExecutor"""
    if not regular_actions:
        return {'applied': 0, 'errors': [], 'skipped': []}
    
    print(">> EXECUTING REGULAR TRADING ACTIONS...")
    print("-" * 50)
    
    try:
        executor = get_executor_instance()
        result = executor.apply_actions_from_json()
        print("[OK] Regular actions processed by AdvancedOrderExecutor")
        return result
        
    except Exception as e:
        print(f"[ERROR] Regular actions execution error: {e}")
        return {'applied': 0, 'errors': [str(e)], 'skipped': []}

def execute_close_actions(close_actions):
    """Execute opposite signal close actions"""
    if not close_actions:
        return {'success': 0, 'failed': 0}
    
    print(">> EXECUTING OPPOSITE SIGNAL CLOSE ACTIONS...")
    print("-" * 50)
    
    # Initialize MT5
    try:
        mt5_conn = MT5ConnectionManager()
        if not mt5_conn.connect():
            print("[ERROR] Failed to initialize MT5 for close actions")
            return {'success': 0, 'failed': len(close_actions)}
        
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
                
                print(f"[{i}/{len(close_actions)}] Closing {symbol} #{ticket} ({close_type})...")
                print(f"    Reason: {reason}")
                print(f"    Confidence: {confidence:.1f}%")
                
                # Get position info
                positions = mt5_conn.positions_get(ticket=ticket)
                if not positions:
                    print(f"    [ERROR] Position #{ticket} not found")
                    failed_count += 1
                    continue
                
                position = positions[0]
                pos_volume = position.volume
                close_volume = volume if close_type == 'partial' else pos_volume
                
                # Validate close volume
                if close_volume <= 0 or close_volume > pos_volume:
                    print(f"    [ERROR] Invalid close volume: {close_volume}")
                    failed_count += 1
                    continue
                
                # Execute close order
                close_request = {
                    "action": mt5_conn.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": close_volume,
                    "type": mt5_conn.ORDER_TYPE_SELL if position.type == 0 else mt5_conn.ORDER_TYPE_BUY,
                    "position": ticket,
                    "price": mt5_conn.symbol_info_tick(symbol).bid if position.type == 0 else mt5_conn.symbol_info_tick(symbol).ask,
                    "deviation": 20,
                    "magic": position.magic,
                    "comment": f"Opposite signal close - {confidence:.0f}%",
                    "type_time": mt5_conn.ORDER_TIME_GTC,
                    "type_filling": mt5_conn.ORDER_FILLING_IOC,
                }
                
                result = mt5_conn.order_send(close_request)
                
                if result and result.retcode == mt5_conn.TRADE_RETCODE_DONE:
                    print(f"    [SUCCESS] {close_type.title()} close executed: {close_volume} lots")
                    print(f"    Deal ID: {result.deal}")
                    success_count += 1
                else:
                    error_msg = result.comment if result else "Unknown error"
                    print(f"    [ERROR] Close failed: {error_msg}")
                    failed_count += 1
                
            except Exception as e:
                print(f"    [ERROR] Exception closing position: {str(e)}")
                failed_count += 1
            
            print()
        
        print(f"[CLOSE SUMMARY] Success: {success_count}, Failed: {failed_count}")
        return {'success': success_count, 'failed': failed_count}
        
    except Exception as e:
        print(f"[ERROR] Close actions execution failed: {str(e)}")
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
    
    # Overall Summary
    total_success = sl_tp_result['success'] + regular_applied + close_result['success']
    total_actions = sl_tp_total + regular_total + close_total
    
    print(f">> OVERALL:")
    print(f"   Total actions: {total_actions}")
    print(f"   Total successful: {total_success}")
    print(f"   Overall success rate: {total_success/total_actions*100:.1f}%" if total_actions > 0 else "   No actions processed")
    
    if total_success > 0:
        print("\n>> EXECUTION COMPLETED!")
        print(">> Check MT5 platform to verify changes")
    else:
        print("\n>> No actions were successfully executed")
        
    return total_success > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)