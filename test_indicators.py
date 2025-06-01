"""
Test script to verify technical indicators using real market data.
This script tests each indicator in TECHNICAL_INDICATORS and visualizes the results.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from signals import TECHNICAL_INDICATORS
import talib

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [15, 10]

def test_indicators():
    """Test all indicators with real market data and plot results."""
    # Download some real market data (last 1 year)
    print("Downloading data...")
    # Use Indian stocks with NSE suffix
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    data = yf.download(tickers, period="1y")
    
    # Extract data frames
    closes = data['Close']
    highs = data['High']
    lows = data['Low']
    volumes = data['Volume']
    
    print(f"Downloaded data for {len(tickers)} tickers over {len(closes)} days")
    
    # Test each indicator type with its first window value
    for indicator_name, window_values in TECHNICAL_INDICATORS.items():
        print(f"\nTesting {indicator_name}...")
        
        # Skip indicators with no window if empty
        if not window_values[0]:
            window = None
        else:
            window = int(window_values[0])
            
        # Use first ticker for detailed visualization
        ticker = tickers[0]
        close = closes[ticker].values.astype(np.float64)
        high = highs[ticker].values.astype(np.float64)
        low = lows[ticker].values.astype(np.float64)
        volume = volumes[ticker].values.astype(np.float64)  # Convert volume to float64
        
        # Calculate and plot the indicator
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot price series
        axes[0].plot(closes.index, closes[ticker], label=f'{ticker} Close')
        axes[0].set_title(f'{ticker} Price Chart')
        axes[0].legend()
        axes[0].grid(True)
        
        # Calculate and plot indicator
        if indicator_name == "SMA":
            if window:
                sma = talib.SMA(close, timeperiod=window)
                axes[1].plot(closes.index, sma, 'r-', label=f'SMA({window})')
                axes[0].plot(closes.index, sma, 'r--', alpha=0.7, label=f'SMA({window})')
        
        elif indicator_name == "EMA":
            if window:
                ema = talib.EMA(close, timeperiod=window)
                axes[1].plot(closes.index, ema, 'g-', label=f'EMA({window})')
                axes[0].plot(closes.index, ema, 'g--', alpha=0.7, label=f'EMA({window})')
        
        elif indicator_name == "WMA":
            if window:
                wma = talib.WMA(close, timeperiod=window)
                axes[1].plot(closes.index, wma, 'b-', label=f'WMA({window})')
                axes[0].plot(closes.index, wma, 'b--', alpha=0.7, label=f'WMA({window})')
        
        elif indicator_name == "RSI":
            if window:
                rsi = talib.RSI(close, timeperiod=window)
                axes[1].plot(closes.index, rsi, 'purple', label=f'RSI({window})')
                # Add overbought/oversold lines
                axes[1].axhline(y=70, color='r', linestyle='-', alpha=0.3)
                axes[1].axhline(y=30, color='g', linestyle='-', alpha=0.3)
                axes[1].set_ylim([0, 100])
        
        elif indicator_name == "WILLR":
            if window:
                willr = talib.WILLR(high, low, close, timeperiod=window)
                axes[1].plot(closes.index, willr, 'orange', label=f'WILLR({window})')
                axes[1].axhline(y=-20, color='r', linestyle='-', alpha=0.3)
                axes[1].axhline(y=-80, color='g', linestyle='-', alpha=0.3)
                axes[1].set_ylim([-100, 0])
        
        elif indicator_name == "CCI":
            if window:
                cci = talib.CCI(high, low, close, timeperiod=window)
                axes[1].plot(closes.index, cci, 'c', label=f'CCI({window})')
                axes[1].axhline(y=100, color='r', linestyle='-', alpha=0.3)
                axes[1].axhline(y=-100, color='g', linestyle='-', alpha=0.3)
                
        elif indicator_name == "ROC":
            if window:
                roc = talib.ROC(close, timeperiod=window)
                axes[1].plot(closes.index, roc, 'm', label=f'ROC({window})')
                axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
        elif indicator_name == "ATR":
            if window:
                atr = talib.ATR(high, low, close, timeperiod=window)
                axes[1].plot(closes.index, atr, 'y', label=f'ATR({window})')
                
        elif indicator_name == "SUPERTREND":
            # Simplified display of Supertrend direction
            if window:
                # Calculate ATR
                atr = talib.ATR(high, low, close, timeperiod=window)
                # Basic bands (no final band calculation for simplicity)
                multiplier = 3.0
                hl2 = (high + low) / 2
                upper = hl2 + (multiplier * atr)
                lower = hl2 - (multiplier * atr)
                
                # Plot bands
                axes[0].plot(closes.index, upper, 'r--', alpha=0.5, label=f'Upper({window})')
                axes[0].plot(closes.index, lower, 'g--', alpha=0.5, label=f'Lower({window})')
                
                # Direction (simplified): 1 if close > upper, -1 if close < lower
                direction = np.zeros_like(close)
                for i in range(1, len(close)):
                    if close[i] > upper[i-1]:
                        direction[i] = 1
                    elif close[i] < lower[i-1]:
                        direction[i] = -1
                    else:
                        direction[i] = direction[i-1]
                
                axes[1].plot(closes.index, direction, 'k', label=f'SUPERTREND({window})')
                axes[1].set_ylim([-1.5, 1.5])
                
        elif indicator_name == "BBANDS":
            if window:
                upper, middle, lower = talib.BBANDS(close, timeperiod=window)
                axes[0].plot(closes.index, upper, 'r--', alpha=0.7, label=f'Upper BB({window})')
                axes[0].plot(closes.index, middle, 'y--', alpha=0.7, label=f'Middle BB({window})')
                axes[0].plot(closes.index, lower, 'g--', alpha=0.7, label=f'Lower BB({window})')
                
                # Calculate %B
                b_percent = (close - lower) / (upper - lower)
                axes[1].plot(closes.index, b_percent, 'b', label=f'%B({window})')
                axes[1].axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
                axes[1].axhline(y=0.5, color='y', linestyle='-', alpha=0.3)
                axes[1].axhline(y=0.0, color='g', linestyle='-', alpha=0.3)
                
        elif indicator_name == "OBV":
            obv = talib.OBV(close, volume)  # Volume now properly converted to float64
            axes[1].plot(closes.index, obv, 'darkblue', label='OBV')
            
        elif indicator_name == "AD":
            ad = talib.AD(high, low, close, volume)  # Volume now properly converted to float64
            axes[1].plot(closes.index, ad, 'darkgreen', label='A/D Line')
            
        elif indicator_name == "MACD":
            # MACD requires 3 values
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            axes[1].plot(closes.index, macd, 'blue', label='MACD Line')
            axes[1].plot(closes.index, macd_signal, 'red', label='Signal Line')
            axes[1].bar(closes.index, macd_hist, color=['green' if x >= 0 else 'red' for x in macd_hist], 
                        alpha=0.5, label='Histogram')
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
        elif indicator_name == "STOCH":
            # Stochastic oscillator
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            axes[1].plot(closes.index, slowk, 'blue', label='%K Line')
            axes[1].plot(closes.index, slowd, 'red', label='%D Line')
            axes[1].axhline(y=80, color='r', linestyle='-', alpha=0.3)
            axes[1].axhline(y=20, color='g', linestyle='-', alpha=0.3)
            axes[1].set_ylim([0, 100])
            
        axes[0].legend()
        axes[1].legend()
        axes[1].set_title(f'{indicator_name} Indicator')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{indicator_name}_test.png")
        plt.close()
        
        print(f"Saved {indicator_name}_test.png")
    
    print("\nTesting complete! Check the generated image files to verify each indicator.")

if __name__ == "__main__":
    test_indicators() 