from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import yfinance as yf
import pandas as pd
from pypfopt.base_optimizer import BaseOptimizer
from pypfopt import EfficientFrontier, expected_returns
from pypfopt.risk_models import CovarianceShrinkage
from functools import lru_cache
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from fastapi.encoders import jsonable_encoder
import matplotlib.pyplot as plt
import os
import base64
import numpy as np

# Set Matplotlib to use 'Agg' backend
plt.switch_backend('Agg')
########################################
# Ensure output directory
########################################
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)

########################################
# FastAPI app + CORS
########################################
app = FastAPI()

origins = [
    "https://indportfoliooptimization.vercel.app",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

########################################
# Pydantic Models
########################################

class ExchangeEnum(str, Enum):
    NSE = "NSE"
    BSE = "BSE"

class PortfolioPerformance(BaseModel):
    # From PyPortfolioOpt
    expected_return: float
    volatility: float
    sharpe: float
    # Custom
    sortino: float
    max_drawdown: float
    romad: float
    var_95: float
    cvar_95: float
    var_90: float
    cvar_90: float
    cagr: float  # Newly added

class OptimizationResult(BaseModel):
    weights: Dict[str, float]
    performance: PortfolioPerformance
    returns_dist: Optional[str] = None
    max_drawdown_plot: Optional[str] = None

class PortfolioOptimizationResponse(BaseModel):
    MVO: Optional[OptimizationResult]
    MinVol: Optional[OptimizationResult]
    MaxQuadraticUtility: Optional[OptimizationResult]
    EquiWeighted: Optional[OptimizationResult]
    start_date: datetime
    end_date: datetime
    cumulative_returns: Dict[str, List[Optional[float]]]
    dates: List[datetime]
    nifty_returns: List[float]

class StockItem(BaseModel):
    ticker: str
    exchange: ExchangeEnum

class TickerRequest(BaseModel):
    stocks: List[StockItem]

class EquiWeightedOptimizer(BaseOptimizer):
    def __init__(self, n_assets: int, tickers: List[str] = None):
        super().__init__(n_assets, tickers)
        self.returns = None
    def optimize(self)->dict:
        equal_weight = 1 / self.n_assets
        self.weights = np.full(self.n_assets, equal_weight)
        return self.clean_weights(cutoff=1e-4, rounding=5)
    def portfolio_performance(self, verbose=False, risk_free_rate=0.05):
        """
        Required custom implementation since BaseOptimizer
        doesn't provide this method
        """
        if self.returns is None:
            raise ValueError("Historical returns not set")

        # Calculate metrics manually
        port_returns = self.returns @ self.weights
        annual_return = port_returns.mean() * 252
        annual_volatility = port_returns.std() * np.sqrt(252)
        sharpe = (annual_return - risk_free_rate) / annual_volatility

        if verbose:
            print(f"Annual Return: {annual_return:.2%}")
            print(f"Annual Volatility: {annual_volatility:.2%}")
            print(f"Sharpe Ratio: {sharpe:.2f}")

        return annual_return, annual_volatility, sharpe

########################################
# Utility Functions
########################################

def file_to_base64(filepath: str) -> str:
    """Convert a saved image file to base64 string."""
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@lru_cache(maxsize=128)
def cached_yf_download(ticker: str, start_date: datetime) -> pd.Series:
    """Cached download of Adjusted Close from yfinance."""
    return yf.download(ticker, start=start_date,multi_level_index=False)['Close']

def format_tickers(stocks: List[StockItem]) -> List[str]:
    """Convert StockItem list into yfinance-friendly tickers."""
    formatted_tickers = []
    for stock in stocks:
        if stock.exchange == ExchangeEnum.BSE:
            formatted_tickers.append(stock.ticker + ".BO")
        elif stock.exchange == ExchangeEnum.NSE:
            formatted_tickers.append(stock.ticker + ".NS")
        else:
            raise ValueError(f"Invalid exchange: {stock.exchange}")
    return formatted_tickers


########################################
# 1) fetch_and_align_data
########################################
def fetch_and_align_data(tickers: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Download & align data for each ticker, plus Nifty index."""
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
    
    # Align all tickers to the latest min_date among them
    min_date = max(df.index.min() for df in data.values())
    filtered_data = {t: df[df.index >= min_date] for t, df in data.items()}

    # Combine into multi-index DataFrame
    combined_df = pd.concat(filtered_data.values(), axis=1, keys=filtered_data.keys())
    combined_df.dropna(inplace=True)

    # Nifty
    nifty_df = yf.download('^NSEI', start=min_date,multi_level_index=False)['Close'].dropna()
    common_dates = combined_df.index.intersection(nifty_df.index)
    combined_df = combined_df.loc[common_dates]
    nifty_df = nifty_df.loc[common_dates]

    return combined_df, nifty_df


########################################
# 2) Helper: Compute Additional Metrics
########################################
def compute_custom_metrics(port_returns: pd.Series, risk_free_rate: float = 0.05) -> Dict[str, float]:
    """
    Compute custom metrics:
      - sortino
      - max_drawdown
      - romad
      - var_95, cvar_95
      - var_90, cvar_90
      - cagr
    Then we'll combine these with PyPortfolioOpt's expected_return/volatility/sharpe.
    """
    ann_factor = 252

    # Sortino
    downside_std = port_returns[port_returns < 0].std()
    sortino = 0.0
    if downside_std > 1e-9:
        mean_daily = port_returns.mean()
        annual_ret = mean_daily * ann_factor
        sortino = (annual_ret - risk_free_rate) / (downside_std * np.sqrt(ann_factor))

    # Drawdown stats
    cum = (1 + port_returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()  # negative

    # RoMaD
    final_cum = cum.iloc[-1] - 1.0
    romad = final_cum / abs(max_dd) if max_dd < 0 else 0.0

    # VaR / CVaR
    var_95 = np.percentile(port_returns, 5)
    below_95 = port_returns[port_returns <= var_95]
    cvar_95 = below_95.mean() if not below_95.empty else var_95

    var_90 = np.percentile(port_returns, 10)
    below_90 = port_returns[port_returns <= var_90]
    cvar_90 = below_90.mean() if not below_90.empty else var_90

    # CAGR
    # number of daily returns:
    n_days = len(port_returns)
    if n_days > 1:
        final_growth = cum.iloc[-1]  # e.g. 1.25 => up 25%
        # annualize via 252 trading days
        cagr = final_growth ** (ann_factor / n_days) - 1.0
    else:
        cagr = 0.0

    return {
        "sortino": sortino,
        "max_drawdown": max_dd,
        "romad": romad,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "var_90": var_90,
        "cvar_90": cvar_90,
        "cagr": cagr
    }


########################################
# 3) compute_optimal_portfolio
########################################
def compute_optimal_portfolio(
    df: pd.DataFrame,
    nifty_df: pd.Series
) -> Tuple[Dict[str, Optional[OptimizationResult]], pd.DataFrame, pd.Series]:
    """
    Runs 3 portfolio optimizations: MVO, MinVol, MaxQuadraticUtility.
    Uses PyPortfolioOpt for (expected_return, volatility, sharpe).
    Adds custom metrics (sortino, drawdown, var/cvar, cagr).
    Plots distribution & drawdown. Returns results + cumulative returns + Nifty.
    """
    risk_free_rate = 0.05
    mu = expected_returns.mean_historical_return(df, frequency=252)
    S = CovarianceShrinkage(df).ledoit_wolf()

    results = {}
    returns = df.pct_change().dropna()

    methods = ["MVO", "MinVol", "MaxQuadraticUtility","EquiWeighted"]

    for method in methods:
        try:
            ef = EfficientFrontier(mu, S)
            if method == "MVO":
                ef.max_sharpe(risk_free_rate=risk_free_rate)
                label = "MVO"
            elif method == "MinVol":
                ef.min_volatility()
                label = "MinVol"
            elif method=="MaxQuadraticUtility":
                ef.max_quadratic_utility(risk_aversion=5)
                label = "MaxQuadraticUtility"
            elif method == "EquiWeighted":
                ew = EquiWeightedOptimizer(n_assets=len(mu), tickers=mu.index)
                ew.returns = returns
                weights = ew.optimize()
                label = "EquiWeighted"
            if method != "EquiWeighted":
                # Weights & performance from PyPortfolioOpt
                weights = ef.clean_weights()
                w_series = pd.Series(weights)
                pfolio_perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            # pfolio_perf -> (expected_return, volatility, sharpe)
            else:
                w_series = pd.Series(weights)
                pfolio_perf = ew.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

            # Daily returns of this portfolio
            port_returns = returns.dot(w_series)

            # 1) Compute custom metrics
            custom = compute_custom_metrics(port_returns, risk_free_rate)

            # 2) Plot distribution + VaR/CVaR lines
            plt.figure(figsize=(10, 6))
            plt.hist(port_returns, bins=50, edgecolor='black', alpha=0.7, label='Daily Returns')
            plt.title(f"Distribution of {label} Portfolio Returns")
            plt.xlabel("Returns")
            plt.ylabel("Frequency")
            plt.grid(True)

            # Add var/cvar lines
            plt.axvline(custom["var_95"], color='r', linestyle='--',
                        label=f"VaR 95%: {custom['var_95']:.4f}")
            plt.axvline(custom["cvar_95"], color='r', linestyle='-',
                        label=f"CVaR 95%: {custom['cvar_95']:.4f}")
            plt.axvline(custom["var_90"], color='g', linestyle='--',
                        label=f"VaR 90%: {custom['var_90']:.4f}")
            plt.axvline(custom["cvar_90"], color='g', linestyle='-',
                        label=f"CVaR 90%: {custom['cvar_90']:.4f}")
            plt.legend()

            dist_file = os.path.join(output_dir, f"{label.lower()}_dist.png")
            plt.savefig(dist_file)
            plt.close()
            dist_b64 = file_to_base64(dist_file)

            # 3) Plot drawdown
            cum = (1 + port_returns).cumprod()
            peak = cum.cummax()
            drawdown = (cum - peak) / peak
            plt.figure(figsize=(10, 6))
            plt.plot(drawdown, color='red', label='Drawdown')
            plt.title(f"Drawdown of {label} Portfolio")
            plt.xlabel("Date")
            plt.ylabel("Drawdown")
            plt.grid(True)
            plt.legend()

            dd_file = os.path.join(output_dir, f"{label.lower()}_drawdown.png")
            plt.savefig(dd_file)
            plt.close()
            dd_b64 = file_to_base64(dd_file)

            # Merge everything into one Performance object
            performance = PortfolioPerformance(
                expected_return=pfolio_perf[0],
                volatility=pfolio_perf[1],
                sharpe=pfolio_perf[2],
                sortino=custom["sortino"],
                max_drawdown=custom["max_drawdown"],
                romad=custom["romad"],
                var_95=custom["var_95"],
                cvar_95=custom["cvar_95"],
                var_90=custom["var_90"],
                cvar_90=custom["cvar_90"],
                cagr=custom["cagr"]
            )

            results[label] = OptimizationResult(
                weights=weights,
                performance=performance,
                returns_dist=dist_b64,
                max_drawdown_plot=dd_b64
            )
        except Exception as e:
            print(f"Error in {method} optimization: {e}")
            results[method] = None

    # 4) Cumulative returns
    cumulative_returns = pd.DataFrame(index=df.index)
    for method in methods:
        if results.get(method) and results[method].weights:
            w = pd.Series(results[method].weights)
            port_daily = returns.dot(w)
            cumulative_returns[method] = (1 + port_daily).cumprod()
        else:
            cumulative_returns[method] = None

    # 5) Nifty
    nifty_ret = nifty_df.pct_change().dropna()
    cum_nifty = (1 + nifty_ret).cumprod()
    cumulative_returns = cumulative_returns.loc[cum_nifty.index]

    return results, cumulative_returns, cum_nifty

########################################
# 4) The /optimize Endpoint
########################################
@app.post("/optimize", response_model=PortfolioOptimizationResponse)
def optimize_portfolio(request: TickerRequest):
    try:
        # 1) Format tickers
        formatted_tickers = format_tickers(request.stocks)
        
        # 2) Fetch & align
        df, nifty_df = fetch_and_align_data(formatted_tickers)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found for the given tickers.")

        # 3) Start/end date
        start_date = df.index.min().date()
        end_date = df.index.max().date()

        # 4) Compute
        results, cum_returns_df, cum_nifty = compute_optimal_portfolio(df, nifty_df)

        # 5) Build JSON
        dates = cum_returns_df.index.tolist()
        cumulative_returns = {
            "MVO": cum_returns_df["MVO"].tolist() if "MVO" in cum_returns_df else [],
            "MinVol": cum_returns_df["MinVol"].tolist() if "MinVol" in cum_returns_df else [],
            "MaxQuadraticUtility": (
                cum_returns_df["MaxQuadraticUtility"].tolist()
                if "MaxQuadraticUtility" in cum_returns_df
                else []
            ),
            "EquiWeighted": cum_returns_df["EquiWeighted"].tolist() if "EquiWeighted" in cum_returns_df else [],
        }
        nifty_returns = cum_nifty.tolist()

        # 6) Response
        response = PortfolioOptimizationResponse(
            MVO=results.get("MVO"),
            MinVol=results.get("MinVol"),
            MaxQuadraticUtility=results.get("MaxQuadraticUtility"),
            EquiWeighted = results.get("EquiWeighted"),
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
