from utils import get_pip_value
from constants import (
    STOCHASTIC_OVERBOUGHT,
    STOCHASTIC_OVERSOLD,
    CONFIDENCE_BOOST_BASE,
    CONFIDENCE_BOOST_MULTIPLIER
)

def calculate_smart_entry(symbol: str, signal: str, current_price: float, atr: float, 
                          ema20: float, support_levels: list, resistance_levels: list, 
                          tf_data: dict, sideway_range: dict = None) -> dict:
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
        
        # 🎯 PRIORITY: SIDEWAY RANGE LOGIC (if sideway_range provided)
        if sideway_range:
            support_zone = sideway_range.get('support_zone', {})
            resistance_zone = sideway_range.get('resistance_zone', {})
            
            if signal == 'BUY' and support_zone:
                # BUY at support zone - entry at zone center or current price
                support_center = support_zone.get('center', support_zone.get('low', current_price))
                entry_price = current_price  # Use current price for market order
                
                # S/L below support zone, T/P at resistance zone
                sl_price = support_zone.get('low', current_price * 0.99) * 0.997  # 0.3% below support zone
                tp_price = resistance_zone.get('center', current_price * 1.015) if resistance_zone else current_price * 1.015
                
                return {
                    'entry_price': entry_price,
                    'order_type': 'market',
                    'limit_price': None,
                    'entry_reason': f'Sideway BUY at support zone',
                    'confidence_boost': 10.0,  # Boost confidence for zone-based entry
                    'sl_price': sl_price,
                    'tp_price': tp_price
                }
            
            elif signal == 'SELL' and resistance_zone:
                # SELL at resistance zone - entry at zone center or current price
                resistance_center = resistance_zone.get('center', resistance_zone.get('high', current_price))
                entry_price = current_price  # Use current price for market order
                
                # S/L above resistance zone, T/P at support zone
                sl_price = resistance_zone.get('high', current_price * 1.01) * 1.003  # 0.3% above resistance zone
                tp_price = support_zone.get('center', current_price * 0.985) if support_zone else current_price * 0.985
                
                return {
                    'entry_price': entry_price,
                    'order_type': 'market',
                    'limit_price': None,
                    'entry_reason': f'Sideway SELL at resistance zone',
                    'confidence_boost': 10.0,  # Boost confidence for zone-based entry
                    'sl_price': sl_price,
                    'tp_price': tp_price
                }
            
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
                if stoch_k < STOCHASTIC_OVERSOLD:  # Oversold condition
                    # Wait for stochastic to turn up from oversold
                    if stoch_d and stoch_k > stoch_d:  # K line crossing above D line
                        confidence_boost = CONFIDENCE_BOOST_BASE + (STOCHASTIC_OVERSOLD - stoch_k) * CONFIDENCE_BOOST_MULTIPLIER  # Higher boost for deeper oversold
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
                if stoch_k > STOCHASTIC_OVERBOUGHT:  # Overbought condition
                    # Wait for stochastic to turn down from overbought
                    if stoch_d and stoch_k < stoch_d:  # K line crossing below D line
                        confidence_boost = CONFIDENCE_BOOST_BASE + (stoch_k - STOCHASTIC_OVERBOUGHT) * CONFIDENCE_BOOST_MULTIPLIER  # Higher boost for deeper overbought
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