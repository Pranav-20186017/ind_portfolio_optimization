from typing import List, Dict, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns

app = FastAPI()

# Pydantic model for request validation
class TickerRequest(BaseModel):
    tickers: List[str]
    exchange: str

# Helper function to format tickers based on exchange
def format_tickers(tickers: List[str], exchange: str) -> List[str]:
    if exchange == "BSE":
        return [ticker + ".BO" for ticker in tickers]
    elif exchange == "NSE":
        return [ticker + ".NS" for ticker in tickers]
    else:
        raise ValueError("Invalid exchange. Use 'BSE' or 'NSE'.")

# Helper function to fetch and align data
def fetch_and_align_data(tickers: List[str]) -> pd.DataFrame:
    data = yf.download(tickers, start="2020-01-01", end="2023-01-01")['Adj Close']
    data = data.dropna(axis=1, how='any')  # Drop tickers with missing data
    data = data.dropna()  # Ensure we have common dates for all tickers
    return data

# Helper function to compute portfolio optimization
def compute_optimal_portfolio(df: pd.DataFrame) -> Dict[str, Union[Dict[str, float], float]]:
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # Mean-Variance Optimization (MVO)
    ef_mvo = EfficientFrontier(mu, S)
    weights_mvo = ef_mvo.max_sharpe()
    cleaned_weights_mvo = ef_mvo.clean_weights()
    mvo_sharpe = ef_mvo.portfolio_performance()[2]  # Sharpe ratio

    # Minimum Volatility Portfolio
    ef_min_vol = EfficientFrontier(mu, S)
    weights_min_vol = ef_min_vol.min_volatility()
    cleaned_weights_min_vol = ef_min_vol.clean_weights()
    min_vol_sharpe = ef_min_vol.portfolio_performance()[2]  # Sharpe ratio

    return {
        "MVO": {"weights": cleaned_weights_mvo, "sharpe": mvo_sharpe},
        "MinVol": {"weights": cleaned_weights_min_vol, "sharpe": min_vol_sharpe}
    }

@app.post("/optimize/")
async def optimize_portfolio(request: TickerRequest):
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
