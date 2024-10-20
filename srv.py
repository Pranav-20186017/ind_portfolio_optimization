from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns
from pypfopt.risk_models import CovarianceShrinkage
from functools import lru_cache
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Add other origins if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Exchange Enum
class ExchangeEnum(str, Enum):
    NSE = "NSE"
    BSE = "BSE"

# Pydantic models for request and response
class PortfolioPerformance(BaseModel):
    expected_return: float
    volatility: float
    sharpe: float

class OptimizationResult(BaseModel):
    weights: Dict[str, float]
    performance: PortfolioPerformance

class PortfolioOptimizationResponse(BaseModel):
    MVO: Optional[OptimizationResult]
    MinVol: Optional[OptimizationResult]
    start_date: datetime
    end_date: datetime

# Define StockItem model
class StockItem(BaseModel):
    ticker: str
    exchange: ExchangeEnum  # Use Enum for exchange

# Update TickerRequest to use the new StockItem model
class TickerRequest(BaseModel):
    stocks: List[StockItem]

# Cache the yf.download call
@lru_cache(maxsize=32)
def cached_yf_download(ticker: str) -> pd.Series:
    return yf.download(ticker)['Adj Close']

# Helper function to format tickers based on exchange
def format_tickers(stocks: List[StockItem]) -> List[str]:
    formatted_tickers = []
    for stock in stocks:
        if stock.exchange == ExchangeEnum.BSE:
            formatted_tickers.append(stock.ticker + ".BO")
        elif stock.exchange == ExchangeEnum.NSE:
            formatted_tickers.append(stock.ticker + ".NS")
        else:
            raise ValueError(f"Invalid exchange: {stock.exchange}")
    return formatted_tickers

# Helper function to fetch and align data
def fetch_and_align_data(tickers: List[str]) -> pd.DataFrame:
    # Download data for each ticker using the cached function
    data = {}
    for ticker in tickers:
        try:
            df = cached_yf_download(ticker)
            if not df.empty:
                data[ticker] = df
            else:
                print(f"No data for ticker: {ticker}")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    if not data:
        raise ValueError("No valid data available for the provided tickers.")
    
    # Find the maximum of the minimum dates across all tickers to ensure common dates
    min_date = max(df.index.min() for df in data.values())
    
    # Filter data from the common minimum date for all tickers
    filtered_data = {ticker: df[df.index >= min_date] for ticker, df in data.items()}
    
    # Combine the filtered data into a single DataFrame
    combined_df = pd.concat(filtered_data.values(), axis=1, keys=filtered_data.keys())
    
    # Drop rows with any NaN values
    combined_df = combined_df.dropna()
    
    return combined_df

# Helper function to compute portfolio optimization
def compute_optimal_portfolio(df: pd.DataFrame) -> Dict[str, Optional[OptimizationResult]]:
    """
    Computes optimal portfolios using Mean-Variance Optimization (MVO) and Minimum Volatility.

    Args:
        df (pd.DataFrame): DataFrame of adjusted close prices.

    Returns:
        Dict[str, Optional[OptimizationResult]]: Dictionary containing optimal weights and performance metrics.
    """
    # Set the risk-free rate
    risk_free_rate = 0.05

    # Calculate expected returns and Ledoit-Wolf covariance matrix
    mu = expected_returns.mean_historical_return(df, frequency=252)
    S = CovarianceShrinkage(df).ledoit_wolf()

    # Prepare results dictionary
    results = {}

    try:
        # Mean-Variance Optimization (MVO) - Maximize Sharpe Ratio
        ef_mvo = EfficientFrontier(mu, S)
        ef_mvo.max_sharpe(risk_free_rate=risk_free_rate)  # Maximize Sharpe ratio
        cleaned_weights_mvo = ef_mvo.clean_weights()
        mvo_performance = ef_mvo.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

        # Store MVO results
        results["MVO"] = OptimizationResult(
            weights=cleaned_weights_mvo,
            performance=PortfolioPerformance(
                expected_return=mvo_performance[0],
                volatility=mvo_performance[1],
                sharpe=mvo_performance[2]
            )
        )
    except Exception as e:
        print(f"Error in MVO optimization: {e}")
        results["MVO"] = None

    try:
        # Minimum Volatility Portfolio
        ef_min_vol = EfficientFrontier(mu, S)
        ef_min_vol.min_volatility()  # Minimize volatility
        cleaned_weights_min_vol = ef_min_vol.clean_weights()
        min_vol_performance = ef_min_vol.portfolio_performance(verbose=False)

        # Store Min Vol results
        results["MinVol"] = OptimizationResult(
            weights=cleaned_weights_min_vol,
            performance=PortfolioPerformance(
                expected_return=min_vol_performance[0],
                volatility=min_vol_performance[1],
                sharpe=min_vol_performance[2]
            )
        )
    except Exception as e:
        print(f"Error in MinVol optimization: {e}")
        results["MinVol"] = None

    return results

@app.post("/optimize/", response_model=PortfolioOptimizationResponse)
def optimize_portfolio(request: TickerRequest):
    try:
        # Format tickers based on exchange
        formatted_tickers = format_tickers(request.stocks)
        
        # Fetch and align data
        df = fetch_and_align_data(formatted_tickers)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found for the given tickers.")
        
        # Get the time period
        start_date = df.index.min().date()
        end_date = df.index.max().date()
        
        # Compute optimal portfolio
        results = compute_optimal_portfolio(df)
        
        # Construct the response
        response = PortfolioOptimizationResponse(
            MVO=results.get("MVO"),
            MinVol=results.get("MinVol"),
            start_date=start_date,
            end_date=end_date
        )
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        print(f"Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
