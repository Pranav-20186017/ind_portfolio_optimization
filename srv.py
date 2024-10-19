from typing import List, Dict, Union, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns
from pypfopt.risk_models import CovarianceShrinkage
from functools import lru_cache

app = FastAPI()

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

class TickerRequest(BaseModel):
    tickers: List[str]
    exchange: ExchangeEnum  # Use Enum for exchange

# Cache the yf.download call
@lru_cache(maxsize=32)
def cached_yf_download(ticker: str) -> pd.DataFrame:
    return yf.download(ticker)['Adj Close']

# Helper function to format tickers based on exchange
def format_tickers(tickers: List[str], exchange: ExchangeEnum) -> List[str]:
    if exchange == ExchangeEnum.BSE:
        return [ticker + ".BO" for ticker in tickers]
    elif exchange == ExchangeEnum.NSE:
        return [ticker + ".NS" for ticker in tickers]
    else:
        raise ValueError("Invalid exchange. Use 'BSE' or 'NSE'.")

# Helper function to fetch and align data
def fetch_and_align_data(tickers: List[str]) -> pd.DataFrame:
    # Download data for each ticker using the cached function
    data = {ticker: cached_yf_download(ticker) for ticker in tickers}
    
    # Filter out empty DataFrames
    data = {ticker: df for ticker, df in data.items() if not df.empty}

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
def compute_optimal_portfolio(df: pd.DataFrame) -> PortfolioOptimizationResponse:
    """
    Computes optimal portfolios using Mean-Variance Optimization (MVO) and Minimum Volatility.

    Args:
        df (pd.DataFrame): DataFrame of adjusted close prices.

    Returns:
        PortfolioOptimizationResponse: Dictionary containing optimal weights and performance metrics.
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
        results["MinVol"] = None

    return PortfolioOptimizationResponse(**results)

@app.post("/optimize/", response_model=PortfolioOptimizationResponse)
def optimize_portfolio(request: TickerRequest):
    try:
        # Format tickers based on exchange
        formatted_tickers = format_tickers(request.tickers, request.exchange)
        
        # Fetch and align data
        df = fetch_and_align_data(formatted_tickers)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found for the given tickers.")
        
        # Compute optimal portfolio
        results = compute_optimal_portfolio(df)
        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
