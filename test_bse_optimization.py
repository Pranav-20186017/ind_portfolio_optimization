"""
Test script to debug BSE optimization with specific tickers
"""
import sys
import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("bse_test")

# Import necessary modules from srv.py
try:
    from srv import (
        optimize_portfolio, fetch_and_align_data, sanitize_bse_prices,
        StockItem, ExchangeEnum, OptimizationMethod, TickerRequest, 
        BenchmarkName, APIError
    )
    
    print("Successfully imported modules from srv.py")
except ImportError as e:
    print(f"ERROR: Could not import from srv.py: {str(e)}")
    print("Make sure this script is in the same directory as srv.py")
    sys.exit(1)

async def run_test():
    """Run the BSE optimization test"""
    print("=" * 80)
    print("STARTING BSE OPTIMIZATION TEST")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 80)
    
    try:
        # Create test request with the specified BSE tickers
        request = TickerRequest(
            stocks=[
                StockItem(ticker="ABB", exchange=ExchangeEnum.BSE),
                StockItem(ticker="JSWSTEEL", exchange=ExchangeEnum.BSE),
                StockItem(ticker="HINDALCO", exchange=ExchangeEnum.BSE)
            ],
            methods=[OptimizationMethod.MVO],
            benchmark=BenchmarkName.sensex  # Using Sensex as benchmark
        )
        
        print(f"Created request with {len(request.stocks)} BSE tickers")
        
        # Option 1: Test just the fetch_and_align_data function first
        print("\n" + "=" * 40)
        print("STEP 1: TESTING fetch_and_align_data")
        print("=" * 40)
        
        # Convert StockItem objects to proper ticker format
        tickers = [f"{s.ticker}.BO" for s in request.stocks]
        benchmark = "^BSESN"  # Sensex benchmark
        
        # Test fetch_and_align_data with sanitize_bse=True
        try:
            print(f"Calling fetch_and_align_data with tickers={tickers}, sanitize_bse=True")
            df, benchmark_df = await fetch_and_align_data(tickers, benchmark, sanitize_bse=True)
            
            print("\nDATAFRAME SUMMARY:")
            print(f"Shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
            
            # Calculate returns to check for issues
            returns = df.pct_change().dropna()
            print(f"\nRETURNS SUMMARY:")
            print(f"Shape: {returns.shape}")
            print(f"NaN count: {returns.isna().sum().sum()}")
            try:
                inf_count = np.isinf(returns).sum().sum()
                print(f"Infinity count: {inf_count}")
                
                max_val = returns.max().max()
                min_val = returns.min().min()
                print(f"Returns range: min={min_val:.6f}, max={max_val:.6f}")
                
                if max_val > 1.0 or min_val < -0.9:
                    print("WARNING: Extreme return values detected!")
            except Exception as e:
                print(f"Error checking returns: {str(e)}")
        
        except Exception as e:
            print(f"ERROR in fetch_and_align_data: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Option 2: Run the full optimization
        print("\n" + "=" * 40)
        print("STEP 2: TESTING full optimization_portfolio")
        print("=" * 40)
        
        try:
            # Call optimize_portfolio directly
            print("Calling optimize_portfolio...")
            result = await optimize_portfolio(request)
            
            # If we get here, it worked!
            print("\nOPTIMIZATION SUCCESSFUL!")
            print(f"Result has {len(result.get('results', {}))} optimization methods")
            
            # Print portfolio weights
            for method, opt_result in result.get('results', {}).items():
                print(f"\nMethod: {method}")
                print("Portfolio weights:")
                for ticker, weight in sorted(opt_result['weights'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {ticker}: {weight:.4f}")
                
                # Print key performance metrics
                perf = opt_result['performance']
                print("Performance metrics:")
                print(f"  Expected Return: {perf['expected_return']:.4f}")
                print(f"  Volatility: {perf['volatility']:.4f}")
                print(f"  Sharpe Ratio: {perf['sharpe']:.4f}")
                print(f"  Max Drawdown: {perf['max_drawdown']:.4f}")
            
        except APIError as e:
            print(f"API ERROR: {e.code} - {e.message}")
            if hasattr(e, 'details') and e.details:
                print(f"Details: {e.details}")
        except Exception as e:
            print(f"ERROR in optimize_portfolio: {str(e)}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"GENERAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_test()) 