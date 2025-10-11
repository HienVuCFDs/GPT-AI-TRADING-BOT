def get_pip_value(symbol: str) -> float:
    """Calculate pip value for different symbol types"""
    symbol_upper = symbol.upper()
    
    if 'JPY' in symbol_upper:
        return 0.01  # JPY pairs: 1 pip = 0.01
    elif symbol_upper in ['XAUUSD', 'XAGUSD', 'GOLD', 'SILVER']:
        return 0.1   # Precious metals: 1 pip = 0.1 (metals standard)
    elif symbol_upper in ['BTCUSD', 'ETHUSD', 'LTCUSD']:
        return 1.0   # High-value crypto: 1 pip = 1 USD (e.g., ETH 4000.00 -> 4070.00 = 70 pips)
    elif symbol_upper in ['SOLUSD']:
        return 0.1   # SOL: 1 pip = 0.1 (SOL 220.00 -> 220.70 = 7 pips)
    else:
        return 0.0001  # Major FX pairs: 1 pip = 0.0001

def calculate_smart_entry(symbol: str, signal: str, current_price: float, atr: float, 
                          ema20: float, support_levels: list, resistance_levels: list, 
                          tf_data: dict) -> dict:
    """
    Calculate intelligent entry price with INDICATOR-BASED logic
    NEW LOGIC:
    - Candlestick patterns determine BUY/SELL signal
    - Price patterns determine trend strength
    - INDICATORS determine optimal ENTRY POINT (this function)
    - Support/Resistance determine SL/TP (handled elsewhere)
    
    Returns: {
        'entry_price': float,
        'order_type': 'market' | 'limit',
        'limit_price': float | None,
        'entry_reason': str,
        'confidence_boost': float
    }
    """
    try:
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            return None
            
        # Get key levels for analysis
        support_levels = [s for s in (support_levels or []) if isinstance(s, (int, float))]
        resistance_levels = [r for r in (resistance_levels or []) if isinstance(r, (int, float))]
        
        # Define pip value for the symbol
        pip_value = get_pip_value(symbol)
        pip_range = 20 * pip_value  # 20 pips range for "good entry"
        
        entry_data = {
            'entry_price': current_price,
            'order_type': 'market',
            'limit_price': None,
            'entry_reason': 'Giá thị trường',
            'confidence_boost': 0.0
        }
        
        # ========== NEW LOGIC: INDICATOR-BASED ENTRY POINTS ==========
        # Extract indicator data from tf_data for entry point calculation
        indicators = tf_data.get('indicators', []) if tf_data else []
        latest_ind = indicators[-1] if indicators else {}
        
        # Get key indicator values for entry calculation
        bb_upper = latest_ind.get('BB_Upper_20_2', None)
        bb_middle = latest_ind.get('BB_Middle_20_2', None) 
        bb_lower = latest_ind.get('BB_Lower_20_2', None)
        stoch_k = latest_ind.get('StochK_14_3', None)
        stoch_d = latest_ind.get('StochD_14_3', None)
        ema50 = latest_ind.get('EMA50', None)
        ema100 = latest_ind.get('EMA100', None)
        ema200 = latest_ind.get('EMA200', None)
        ichimoku_senkou_a = latest_ind.get('ichimoku_senkou_a', None)
        ichimoku_senkou_b = latest_ind.get('ichimoku_senkou_b', None)
        
        # Smart entry logic for BUY signals
        if signal == 'BUY':
            best_entry = current_price
            best_reason = 'Giá thị trường'
            best_boost = 0.0
            
            # INDICATOR PRIORITY 1: EMA CONFLUENCE ENTRY
            ema_levels = [ema for ema in [ema20, ema50, ema100, ema200] if ema and isinstance(ema, (int, float))]
            if ema_levels:
                closest_ema = min(ema_levels, key=lambda x: abs(current_price - x))
                ema_distance = abs(current_price - closest_ema)
                
                # Check if price is near EMA for good entry
                if ema_distance <= 10 * pip_value:  # Within 10 pips of EMA
                    confidence_boost = 12.0 - (ema_distance / pip_value) * 1.0  # Up to 12% boost
                    if confidence_boost > best_boost:
                        best_entry = closest_ema + (2 * pip_value)  # Enter slightly above EMA
                        best_reason = f'EMA Entry: {closest_ema:.5f}'
                        best_boost = confidence_boost
                        entry_data['order_type'] = 'limit' if abs(best_entry - current_price) > 3 * pip_value else 'market'
            
            # INDICATOR PRIORITY 2: BOLLINGER BAND ENTRY
            if bb_lower and bb_middle and current_price:
                # Buy near BB lower for oversold bounce
                bb_distance = abs(current_price - bb_lower)
                if bb_distance <= 15 * pip_value:  # Within 15 pips of BB lower
                    confidence_boost = 15.0 - (bb_distance / pip_value) * 0.8  # Up to 15% boost for BB lower
                    if confidence_boost > best_boost:
                        best_entry = bb_lower + (3 * pip_value)  # Enter above BB lower
                        best_reason = f'BB Lower Bounce: {bb_lower:.5f}'
                        best_boost = confidence_boost
                        entry_data['order_type'] = 'limit'
                        
                # Or buy on BB middle retest in uptrend
                elif bb_middle:
                    bb_mid_distance = abs(current_price - bb_middle)
                    if bb_mid_distance <= 8 * pip_value:  # Within 8 pips of BB middle
                        confidence_boost = 10.0 - (bb_mid_distance / pip_value) * 1.0  # Up to 10% boost
                        if confidence_boost > best_boost:
                            best_entry = bb_middle + (1 * pip_value)
                            best_reason = f'BB Middle Retest: {bb_middle:.5f}'
                            best_boost = confidence_boost
            
            # INDICATOR PRIORITY 3: STOCHASTIC OVERSOLD ENTRY
            if stoch_k and isinstance(stoch_k, (int, float)):
                if stoch_k < 30:  # Oversold condition
                    # Wait for stochastic to turn up from oversold
                    if stoch_d and stoch_k > stoch_d:  # K line crossing above D line
                        confidence_boost = 8.0 + (30 - stoch_k) * 0.2  # Higher boost for deeper oversold
                        if confidence_boost > best_boost:
                            best_entry = current_price
                            best_reason = f'Stoch Oversold Turn: K={stoch_k:.1f}'
                            best_boost = confidence_boost
            
            # INDICATOR PRIORITY 4: ICHIMOKU CLOUD ENTRY  
            if ichimoku_senkou_a and ichimoku_senkou_b:
                cloud_top = max(ichimoku_senkou_a, ichimoku_senkou_b)
                cloud_bottom = min(ichimoku_senkou_a, ichimoku_senkou_b)
                
                # Buy above cloud or on cloud support
                if current_price >= cloud_top:
                    cloud_distance = current_price - cloud_top
                    if cloud_distance <= 5 * pip_value:  # Just above cloud
                        confidence_boost = 7.0 - (cloud_distance / pip_value) * 1.0
                        if confidence_boost > best_boost:
                            best_entry = cloud_top + (1 * pip_value)
                            best_reason = f'Ichimoku Cloud Support: {cloud_top:.5f}'
                            best_boost = confidence_boost
            
            # FALLBACK: Support level consideration (lower priority now)
            for support in support_levels:
                distance = abs(current_price - support)
                if distance <= 8 * pip_value:  # Close to support
                    confidence_boost = 5.0 - (distance / pip_value) * 0.5  # Reduced importance
                    if confidence_boost > best_boost:
                        best_entry = support + (2 * pip_value)
                        best_reason = f'Support Level: {support:.5f}'
                        best_boost = confidence_boost
            
            # Update entry data with best option
            entry_data.update({
                'entry_price': best_entry,
                'entry_reason': best_reason,
                'confidence_boost': best_boost
            })
            
        # Smart entry logic for SELL signals - INDICATOR DRIVEN
        elif signal == 'SELL':
            best_entry = current_price
            best_reason = 'Giá thị trường'
            best_boost = 0.0
            
            # INDICATOR PRIORITY 1: EMA CONFLUENCE ENTRY
            ema_levels = [ema for ema in [ema20, ema50, ema100, ema200] if ema and isinstance(ema, (int, float))]
            if ema_levels:
                closest_ema = min(ema_levels, key=lambda x: abs(current_price - x))
                ema_distance = abs(current_price - closest_ema)
                
                # Check if price is near EMA for good entry (sell from EMA resistance)
                if ema_distance <= 10 * pip_value:  # Within 10 pips of EMA
                    confidence_boost = 12.0 - (ema_distance / pip_value) * 1.0  # Up to 12% boost
                    if confidence_boost > best_boost:
                        best_entry = closest_ema - (2 * pip_value)  # Enter slightly below EMA
                        best_reason = f'EMA Resistance: {closest_ema:.5f}'
                        best_boost = confidence_boost
                        entry_data['order_type'] = 'limit' if abs(best_entry - current_price) > 3 * pip_value else 'market'
            
            # INDICATOR PRIORITY 2: BOLLINGER BAND ENTRY
            if bb_upper and bb_middle and current_price:
                # Sell near BB upper for overbought rejection
                bb_distance = abs(current_price - bb_upper)
                if bb_distance <= 15 * pip_value:  # Within 15 pips of BB upper
                    confidence_boost = 15.0 - (bb_distance / pip_value) * 0.8  # Up to 15% boost for BB upper
                    if confidence_boost > best_boost:
                        best_entry = bb_upper - (3 * pip_value)  # Enter below BB upper
                        best_reason = f'BB Upper Rejection: {bb_upper:.5f}'
                        best_boost = confidence_boost
                        entry_data['order_type'] = 'limit'
                        
                # Or sell on BB middle retest in downtrend
                elif bb_middle:
                    bb_mid_distance = abs(current_price - bb_middle)
                    if bb_mid_distance <= 8 * pip_value:  # Within 8 pips of BB middle
                        confidence_boost = 10.0 - (bb_mid_distance / pip_value) * 1.0  # Up to 10% boost
                        if confidence_boost > best_boost:
                            best_entry = bb_middle - (1 * pip_value)
                            best_reason = f'BB Middle Resistance: {bb_middle:.5f}'
                            best_boost = confidence_boost
            
            # INDICATOR PRIORITY 3: STOCHASTIC OVERBOUGHT ENTRY
            if stoch_k and isinstance(stoch_k, (int, float)):
                if stoch_k > 70:  # Overbought condition
                    # Wait for stochastic to turn down from overbought
                    if stoch_d and stoch_k < stoch_d:  # K line crossing below D line
                        confidence_boost = 8.0 + (stoch_k - 70) * 0.2  # Higher boost for deeper overbought
                        if confidence_boost > best_boost:
                            best_entry = current_price
                            best_reason = f'Stoch Overbought Turn: K={stoch_k:.1f}'
                            best_boost = confidence_boost
            
            # INDICATOR PRIORITY 4: ICHIMOKU CLOUD ENTRY  
            if ichimoku_senkou_a and ichimoku_senkou_b:
                cloud_top = max(ichimoku_senkou_a, ichimoku_senkou_b)
                cloud_bottom = min(ichimoku_senkou_a, ichimoku_senkou_b)
                
                # Sell below cloud or on cloud resistance
                if current_price <= cloud_bottom:
                    cloud_distance = cloud_bottom - current_price
                    if cloud_distance <= 5 * pip_value:  # Just below cloud
                        confidence_boost = 7.0 - (cloud_distance / pip_value) * 1.0
                        if confidence_boost > best_boost:
                            best_entry = cloud_bottom - (1 * pip_value)
                            best_reason = f'Ichimoku Cloud Resistance: {cloud_bottom:.5f}'
                            best_boost = confidence_boost
            
            # FALLBACK: Resistance level consideration (lower priority now)
            for resistance in resistance_levels:
                distance = abs(current_price - resistance)
                if distance <= 8 * pip_value:  # Close to resistance
                    confidence_boost = 5.0 - (distance / pip_value) * 0.5  # Reduced importance
                    if confidence_boost > best_boost:
                        best_entry = resistance - (2 * pip_value)
                        best_reason = f'Resistance Level: {resistance:.5f}'
                        best_boost = confidence_boost
            
            # Update entry data with best option
            entry_data.update({
                'entry_price': best_entry,
                'entry_reason': best_reason,
                'confidence_boost': best_boost
            })
        
        return entry_data
        
    except Exception as e:
        # Fallback to market entry
        return {
            'entry_price': current_price,
            'order_type': 'market',
            'limit_price': None,
            'entry_reason': 'Market entry (fallback)',
            'confidence_boost': 0.0
        }