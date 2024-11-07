# srv.py

from typing import List, Dict, Optional, Tuple
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
from fastapi.encoders import jsonable_encoder
import matplotlib.pyplot as plt
import os
import base64
import io
import quantstats as qs
# Set Matplotlib to use 'Agg' backend
plt.switch_backend('Agg')

# Ensure the output directory exists
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)

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
    returns_dist : Optional[str] = None
    quantstats_report: Optional[str] = None 
    

class PortfolioOptimizationResponse(BaseModel):
    MVO: Optional[OptimizationResult]
    MinVol: Optional[OptimizationResult]
    start_date: datetime
    end_date: datetime
    cumulative_returns: Dict[str, List[Optional[float]]]  # New field
    dates: List[datetime]  # New field
    nifty_returns: List[float]  # New field

# Define StockItem model
class StockItem(BaseModel):
    ticker: str
    exchange: ExchangeEnum  # Use Enum for exchange

# Update TickerRequest to use the new StockItem model
class TickerRequest(BaseModel):
    stocks: List[StockItem]



# Function to convert saved plot to base64 string
def file_to_base64(filepath):
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Cache the yf.download call
@lru_cache(maxsize=128)
def cached_yf_download(ticker: str, start_date: datetime) -> pd.Series:
    return yf.download(ticker, start=start_date)['Adj Close']

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
def fetch_and_align_data(tickers: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    # Download data for each ticker using the cached function
    data = {}
    for ticker in tickers:
        try:
            df = cached_yf_download(ticker, datetime(1990, 1, 1))
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

    # Fetch Nifty index data
    nifty_df = yf.download('^NSEI', start=min_date)['Adj Close']
    nifty_df = nifty_df[nifty_df.index >= min_date]
    nifty_df = nifty_df.dropna()

    # Align Nifty data with combined_df
    common_dates = combined_df.index.intersection(nifty_df.index)
    combined_df = combined_df.loc[common_dates]
    nifty_df = nifty_df.loc[common_dates]

    return combined_df, nifty_df

# Helper function to save QuantStats report
def save_quantstats_report(returns, filename):
    report_path = os.path.join(output_dir, filename)
    qs.reports.html(returns, output=report_path)
    return report_path


# Helper function to compute portfolio optimization
def compute_optimal_portfolio(df: pd.DataFrame, nifty_df: pd.Series) -> Tuple[Dict[str, Optional[OptimizationResult]], pd.DataFrame, pd.Series]:
    """
    Computes optimal portfolios using Mean-Variance Optimization (MVO) and Minimum Volatility.

    Args:
        df (pd.DataFrame): DataFrame of adjusted close prices.
        nifty_df (pd.Series): Series of Nifty index adjusted close prices.

    Returns:
        Tuple containing:
            - Dict[str, Optional[OptimizationResult]]: Dictionary containing optimal weights and performance metrics.
            - pd.DataFrame: DataFrame of cumulative returns for portfolios.
            - pd.Series: Series of cumulative returns for Nifty index.
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
        mvo_returns_dist = df.pct_change().dropna().dot(pd.Series(cleaned_weights_mvo))
        mvo_report_path = save_quantstats_report(mvo_returns_dist, "mvo_port_report.html")
        
        plt.figure(figsize=(10, 6))
        plt.hist(mvo_returns_dist, bins=50, edgecolor='black', alpha=0.7)
        plt.title("Distribution of MVO Portfolio Returns")
        plt.xlabel("Returns")
        plt.ylabel("Frequency")
        plt.grid(True)
        # Save the plot
        mvo_output_file = os.path.join(output_dir, "mvo_port_dist.png")
        plt.savefig(mvo_output_file)
        # Convert the saved plot to base64
        mvo_image_base64 = file_to_base64(mvo_output_file)
        

        # Store MVO results
        results["MVO"] = OptimizationResult(
            weights=cleaned_weights_mvo,
            performance=PortfolioPerformance(
                expected_return=mvo_performance[0],
                volatility=mvo_performance[1],
                sharpe=mvo_performance[2]
            ),
            returns_dist = mvo_image_base64,
            quantstats_report=mvo_report_path
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
        min_vol_returns_dist = df.pct_change().dropna().dot(pd.Series(cleaned_weights_mvo))
        # Generate QuantStats report for Min Vol
        min_vol_report_path = save_quantstats_report(min_vol_returns_dist, "min_vol_port_report.html")
        plt.figure(figsize=(10, 6))
        plt.hist(min_vol_returns_dist, bins=50, edgecolor='black', alpha=0.7)
        plt.title("Distribution of Min Vol Portfolio Returns")
        plt.xlabel("Returns")
        plt.ylabel("Frequency")
        plt.grid(True)
        # Save the plot
        min_vol_output_file = os.path.join(output_dir, "min_vol_port_dist.png")
        plt.savefig( min_vol_output_file)
        min_vol_image_base64 = file_to_base64(min_vol_output_file)



        # Store Min Vol results
        results["MinVol"] = OptimizationResult(
            weights=cleaned_weights_min_vol,
            performance=PortfolioPerformance(
                expected_return=min_vol_performance[0],
                volatility=min_vol_performance[1],
                sharpe=min_vol_performance[2]
            ),
            returns_dist = min_vol_image_base64,
            quantstats_report=min_vol_report_path
        )
    except Exception as e:
        print(f"Error in MinVol optimization: {e}")
        results["MinVol"] = None

    # Calculate cumulative returns
    cumulative_returns = pd.DataFrame(index=df.index)

    # Portfolio cumulative returns
    returns = df.pct_change().dropna()

    if results.get("MVO") and results["MVO"].weights:
        weights_mvo = pd.Series(results["MVO"].weights)
        portfolio_returns_mvo = returns.dot(weights_mvo)
        cumulative_returns['MVO'] = (1 + portfolio_returns_mvo).cumprod()
    else:
        cumulative_returns['MVO'] = None

    if results.get("MinVol") and results["MinVol"].weights:
        weights_minvol = pd.Series(results["MinVol"].weights)
        portfolio_returns_minvol = returns.dot(weights_minvol)
        cumulative_returns['MinVol'] = (1 + portfolio_returns_minvol).cumprod()
    else:
        cumulative_returns['MinVol'] = None

    # Nifty cumulative returns
    nifty_returns = nifty_df.pct_change().dropna()
    cumulative_nifty_returns = (1 + nifty_returns).cumprod()

    # Align cumulative returns
    cumulative_returns = cumulative_returns.loc[cumulative_nifty_returns.index]

    return results, cumulative_returns, cumulative_nifty_returns

@app.post("/optimize/", response_model=PortfolioOptimizationResponse)
def optimize_portfolio(request: TickerRequest):
    try:
        # Format tickers based on exchange
        formatted_tickers = format_tickers(request.stocks)
        
        # Fetch and align data
        df, nifty_df = fetch_and_align_data(formatted_tickers)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found for the given tickers.")
        
        # Get the time period
        start_date = df.index.min().date()
        end_date = df.index.max().date()
        
        # Compute optimal portfolio
        results, cumulative_returns_df, cumulative_nifty_returns = compute_optimal_portfolio(df, nifty_df)

        # Prepare cumulative returns data for JSON response
        dates = cumulative_returns_df.index.tolist()
        cumulative_returns = {
            'MVO': cumulative_returns_df['MVO'].tolist() if 'MVO' in cumulative_returns_df else [],
            'MinVol': cumulative_returns_df['MinVol'].tolist() if 'MinVol' in cumulative_returns_df else []
        }
        nifty_returns = cumulative_nifty_returns.tolist()

        # Construct the response
        response = PortfolioOptimizationResponse(
            MVO=results.get("MVO"),
            MinVol=results.get("MinVol"),
            start_date=start_date,
            end_date=end_date,
            cumulative_returns=cumulative_returns,
            dates=dates,
            nifty_returns=nifty_returns
        )
        return jsonable_encoder(response)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        print(f"Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")