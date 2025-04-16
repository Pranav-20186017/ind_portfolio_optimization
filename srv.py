from typing import List, Dict, Optional, Tuple, Union
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from enum import Enum
import yfinance as yf
import pandas as pd
from scipy.stats import entropy
from pypfopt.base_optimizer import BaseOptimizer
from pypfopt import EfficientFrontier, expected_returns
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt.efficient_frontier import EfficientCDaR
import statsmodels.api as sm
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import CLA
from pypfopt.hierarchical_portfolio import HRPOpt
from functools import lru_cache
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from fastapi.encoders import jsonable_encoder
import matplotlib.pyplot as plt
import os
import base64
import numpy as np
import warnings
import requests
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from io import StringIO
warnings.filterwarnings("ignore")
import seaborn as sns
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
    "https://indportfoliooptimization.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

########################################
# Pydantic Models and Enums
########################################

class ExchangeEnum(str, Enum):
    NSE = "NSE"
    BSE = "BSE"

class OptimizationMethod(str, Enum):
    MVO = "MVO"
    MIN_VOL = "MinVol"
    MAX_QUADRATIC_UTILITY = "MaxQuadraticUtility"
    EQUI_WEIGHTED = "EquiWeighted"
    CRITICAL_LINE_ALGORITHM = "CriticalLineAlgorithm"
    HRP = "HRP"  # New HRP optimization method
    MIN_CVAR = "MinCVaR"
    MIN_CDAR = "MinCDaR"

# New enum for CLA sub-methods
class CLAOptimizationMethod(str, Enum):
    MVO = "MVO"
    MIN_VOL = "MinVol"
    BOTH = "Both"

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
    cagr: float
    portfolio_beta: float
    skewness: float
    kurtosis: float
    entropy: float

class OptimizationResult(BaseModel):
    weights: Dict[str, float]
    performance: PortfolioPerformance
    returns_dist: Optional[str] = None
    max_drawdown_plot: Optional[str] = None

class PortfolioOptimizationResponse(BaseModel):
    results: Dict[str, Optional[OptimizationResult]]
    start_date: datetime
    end_date: datetime
    cumulative_returns: Dict[str, List[Optional[float]]]
    dates: List[datetime]
    nifty_returns: List[float]
    stock_yearly_returns: Optional[Dict[str, Dict[str, float]]]
    covariance_heatmap: Optional[str] = None
    risk_free_rate : float

class StockItem(BaseModel):
    ticker: str
    exchange: ExchangeEnum

# Updated request to include an optional CLA sub-method field.
class TickerRequest(BaseModel):
    stocks: List[StockItem]
    methods: List[OptimizationMethod] = [OptimizationMethod.MVO]
    cla_method: Optional[CLAOptimizationMethod] = CLAOptimizationMethod.BOTH

########################################
# Custom EquiWeighted Optimizer
########################################
class EquiWeightedOptimizer(BaseOptimizer):
    def __init__(self, n_assets: int, tickers: List[str] = None):
        super().__init__(n_assets, tickers)
        self.returns = None

    def optimize(self) -> dict:
        equal_weight = 1 / self.n_assets
        self.weights = np.full(self.n_assets, equal_weight)
        return self.clean_weights(cutoff=1e-4, rounding=5)

    def portfolio_performance(self, verbose=False, risk_free_rate=0.05):
        """
        Custom implementation of portfolio_performance for equal weighting.
        """
        if self.returns is None:
            raise ValueError("Historical returns not set")

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
    """Cached download of 'Close' price from yfinance."""
    return yf.download(ticker, start=start_date, multi_level_index=False)['Close']

def format_tickers(stocks: List[StockItem]) -> List[str]:
    """Convert StockItem list into yfinance-friendly tickers (adding .BO or .NS)."""
    formatted_tickers = []
    for stock in stocks:
        if stock.exchange == ExchangeEnum.BSE:
            formatted_tickers.append(stock.ticker + ".BO")
        elif stock.exchange == ExchangeEnum.NSE:
            formatted_tickers.append(stock.ticker + ".NS")
        else:
            raise ValueError(f"Invalid exchange: {stock.exchange}")
    return formatted_tickers

def fetch_and_align_data(tickers: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Download & align data for each ticker, plus Nifty index.
    Returns (combined_df, nifty_close).
    """
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
    nifty_df = yf.download('^NSEI', start=min_date, multi_level_index=False)['Close'].dropna()
    common_dates = combined_df.index.intersection(nifty_df.index)
    combined_df = combined_df.loc[common_dates]
    nifty_df = nifty_df.loc[common_dates]

    return combined_df, nifty_df

def freedman_diaconis_bins(port_returns: pd.Series) -> int:
    n = len(port_returns)
    if n < 2:
        return 1  # minimal bin count for very small datasets

    # Calculate IQR (Interquartile Range)
    iqr = port_returns.quantile(0.75) - port_returns.quantile(0.25)
    
    # Compute bin width using Freedman-Diaconis rule
    bin_width = 2 * iqr / np.cbrt(n)
    
    # If bin_width is zero (e.g., when returns are nearly constant), default to a fixed number
    if bin_width == 0:
        return 50

    # Calculate the number of bins over the data range
    data_range = port_returns.max() - port_returns.min()
    bins = int(np.ceil(data_range / bin_width))
    logging.info(f"no of bins computed: {bins}")
    return bins if bins > 0 else 50



def compute_custom_metrics(port_returns: pd.Series, nifty_df: pd.Series, risk_free_rate: float = 0.05) -> Dict[str, float]:
    """
    Compute custom daily-return metrics:
      - sortino
      - max_drawdown
      - romad
      - var_95, cvar_95
      - var_90, cvar_90
      - cagr
      - portfolio_beta
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
    n_days = len(port_returns)
    if n_days > 1:
        final_growth = cum.iloc[-1]
        cagr = final_growth ** (ann_factor / n_days) - 1.0
    else:
        cagr = 0.0

    # Portfolio Beta
    nifty_ret = nifty_df.pct_change().dropna()
    merged_returns = pd.DataFrame({
        'Portfolio': port_returns,
        'Nifty': nifty_ret
    }).dropna()
    risk_free_daily = risk_free_rate / ann_factor
    # Calculate excess returns
    excess_portfolio = merged_returns['Portfolio'] - risk_free_daily
    excess_market = merged_returns['Nifty'] - risk_free_daily
    # Prepare and run regression
    X = sm.add_constant(excess_market)  # Adds intercept term
    model = sm.OLS(excess_portfolio, X).fit()
    portfolio_beta = model.params['Nifty']

    skewness = port_returns.skew()
    kurtosis = port_returns.kurt()
    bins = freedman_diaconis_bins(port_returns)
    counts, _ = np.histogram(port_returns, bins=bins)
    probs = counts / counts.sum()
    port_entropy  = entropy(probs)
    return {
        "sortino": sortino,
        "max_drawdown": max_dd,
        "romad": romad,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "var_90": var_90,
        "cvar_90": cvar_90,
        "cagr": cagr,
        "portfolio_beta": portfolio_beta,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "entropy" : port_entropy
    }

def generate_plots(port_returns: pd.Series, method: str) -> Tuple[str, str]:
    """Generate distribution and drawdown plots, return base64 encoded images"""
    # Plot distribution (histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(port_returns, bins=50, edgecolor='black', alpha=0.7, label='Daily Returns')
    plt.title(f"Distribution of {method} Portfolio Returns")
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    # VaR/CVaR lines
    var_95 = np.percentile(port_returns, 5)
    below_95 = port_returns[port_returns <= var_95]
    cvar_95 = below_95.mean() if not below_95.empty else var_95
    
    var_90 = np.percentile(port_returns, 10)
    below_90 = port_returns[port_returns <= var_90]
    cvar_90 = below_90.mean() if not below_90.empty else var_90
    
    plt.axvline(var_95, color='r', linestyle='--', label=f"VaR 95%: {var_95:.4f}")
    plt.axvline(cvar_95, color='r', linestyle='-', label=f"CVaR 95%: {cvar_95:.4f}")
    plt.axvline(var_90, color='g', linestyle='--', label=f"VaR 90%: {var_90:.4f}")
    plt.axvline(cvar_90, color='g', linestyle='-', label=f"CVaR 90%: {cvar_90:.4f}")
    plt.legend()

    dist_file = os.path.join(output_dir, f"{method.lower()}_dist.png")
    plt.savefig(dist_file)
    plt.close()
    dist_b64 = file_to_base64(dist_file)

    # Plot drawdown
    cum = (1 + port_returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    plt.figure(figsize=(10, 6))
    plt.plot(drawdown, color='red', label='Drawdown')
    plt.title(f"Drawdown of {method} Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.legend()
    
    dd_file = os.path.join(output_dir, f"{method.lower()}_drawdown.png")
    plt.savefig(dd_file)
    plt.close()
    dd_b64 = file_to_base64(dd_file)
    
    return dist_b64, dd_b64

def run_optimization(method: OptimizationMethod, mu, S, returns, nifty_df, risk_free_rate=0.05):
    """Run a specific optimization method (non-CLA, non-HRP) and return the results"""
    try:
        if method == OptimizationMethod.MVO:
            ef = EfficientFrontier(mu, S)
            ef.max_sharpe(risk_free_rate=risk_free_rate)
            weights = ef.clean_weights()
            pfolio_perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            
        elif method == OptimizationMethod.MIN_VOL:
            ef = EfficientFrontier(mu, S)
            ef.min_volatility()
            weights = ef.clean_weights()
            pfolio_perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            
        elif method == OptimizationMethod.MAX_QUADRATIC_UTILITY:
            ef = EfficientFrontier(mu, S)
            ef.max_quadratic_utility(risk_aversion=5)
            weights = ef.clean_weights()
            pfolio_perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            
        elif method == OptimizationMethod.EQUI_WEIGHTED:
            ew = EquiWeightedOptimizer(n_assets=len(mu), tickers=list(mu.index))
            ew.returns = returns
            weights = ew.optimize()
            pfolio_perf = ew.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        else:
            raise ValueError(f"Method {method} not handled in run_optimization.")
        
        # Calculate portfolio returns
        w_series = pd.Series(weights)
        port_returns = returns.dot(w_series)
        
        # Compute custom metrics
        custom = compute_custom_metrics(port_returns, nifty_df, risk_free_rate)
        
        # Generate plots
        dist_b64, dd_b64 = generate_plots(port_returns, method.value)
        
        # Create performance object
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
            cagr=custom["cagr"],
            portfolio_beta=custom["portfolio_beta"],
            skewness=custom["skewness"],
            kurtosis=custom["kurtosis"],
            entropy=custom["entropy"]

        )
        
        # Create result object
        result = OptimizationResult(
            weights=weights,
            performance=performance,
            returns_dist=dist_b64,
            max_drawdown_plot=dd_b64
        )
        
        # Calculate cumulative returns
        cum_returns = (1 + port_returns).cumprod()
        
        return result, cum_returns
        
    except Exception as e:
        print(f"Error in {method} optimization: {e}")
        return None, None
    
def run_optimization_MIN_CVAR(mu, returns, nifty_df, risk_free_rate=0.05):
    try:
        # Initialize EfficientCVaR with the provided expected returns and historical returns.
        ef_cvar = EfficientCVaR(expected_returns=mu, returns=returns, beta=0.95, weight_bounds=(0, 1))
        
        # Obtain weights by minimizing CVaR.
        weights = ef_cvar.min_cvar()
        
        # Although ef_cvar.portfolio_performance() returns (expected_return, CVaR),
        # we will compute full metrics manually from the portfolio’s daily returns.
        w_series = pd.Series(weights)
        port_returns = returns.dot(w_series)

        pfolio_perf = ef_cvar.portfolio_performance(verbose=False)
        
        # Compute annualized return, volatility, and Sharpe ratio manually.
        ann_factor = 252
        annual_return = port_returns.mean() * ann_factor
        annual_volatility = port_returns.std() * np.sqrt(ann_factor)
        sharpe = ((annual_return - risk_free_rate) / annual_volatility) if annual_volatility > 0 else 0.0
        
        # Compute additional custom metrics (sortino, drawdown, CVaRs, CAGR, beta, etc.)
        custom = compute_custom_metrics(port_returns, nifty_df, risk_free_rate)
        
        # Generate plots: return distribution and drawdown (both encoded to base64)
        dist_b64, dd_b64 = generate_plots(port_returns, OptimizationMethod.MIN_CVAR.value)
        
        # Create a performance object with all required metrics.
        performance = PortfolioPerformance(
            expected_return=pfolio_perf[0],
            volatility=annual_volatility,
            sharpe=sharpe,
            sortino=custom["sortino"],
            max_drawdown=custom["max_drawdown"],
            romad=custom["romad"],
            var_95=custom["var_95"],
            cvar_95=custom["cvar_95"],
            var_90=custom["var_90"],
            cvar_90=custom["cvar_90"],
            cagr=custom["cagr"],
            portfolio_beta=custom["portfolio_beta"],
            skewness=custom["skewness"],
            kurtosis=custom["kurtosis"],
            entropy=custom["entropy"]
        )
        
        # Bundle the results.
        result = OptimizationResult(
            weights=weights,
            performance=performance,
            returns_dist=dist_b64,
            max_drawdown_plot=dd_b64
        )
        
        # Compute cumulative portfolio returns.
        cum_returns = (1 + port_returns).cumprod()
        
        return result, cum_returns
    except Exception as e:
        print(f"Error in MIN_CVAR optimization: {e}")
        return None, None

def run_optimization_MIN_CDAR(mu, returns, nifty_df, risk_free_rate=0.05):
    try:
        # Initialize EfficientCDaR with the provided expected returns and historical returns.
        ef_cdar = EfficientCDaR(expected_returns=mu, returns=returns, beta=0.95, weight_bounds=(0, 1))
        
        # Obtain weights by minimizing CDaR.
        weights = ef_cdar.min_cdar()
        
        # Although ef_cdar.portfolio_performance() returns (expected_return, CVaR),
        # we will compute full metrics manually from the portfolio’s daily returns.
        w_series = pd.Series(weights)
        port_returns = returns.dot(w_series)

        pfolio_perf = ef_cdar.portfolio_performance(verbose=False)
        
        # Compute annualized return, volatility, and Sharpe ratio manually.
        ann_factor = 252
        annual_return = port_returns.mean() * ann_factor
        annual_volatility = port_returns.std() * np.sqrt(ann_factor)
        sharpe = ((annual_return - risk_free_rate) / annual_volatility) if annual_volatility > 0 else 0.0
        
        # Compute additional custom metrics (sortino, drawdown, CVaRs, CAGR, beta, etc.)
        custom = compute_custom_metrics(port_returns, nifty_df, risk_free_rate)
        
        # Generate plots: return distribution and drawdown (both encoded to base64)
        dist_b64, dd_b64 = generate_plots(port_returns, OptimizationMethod.MIN_CDAR.value)
        
        # Create a performance object with all required metrics.
        performance = PortfolioPerformance(
            expected_return=pfolio_perf[0],
            volatility=annual_volatility,
            sharpe=sharpe,
            sortino=custom["sortino"],
            max_drawdown=custom["max_drawdown"],
            romad=custom["romad"],
            var_95=custom["var_95"],
            cvar_95=custom["cvar_95"],
            var_90=custom["var_90"],
            cvar_90=custom["cvar_90"],
            cagr=custom["cagr"],
            portfolio_beta=custom["portfolio_beta"],
            skewness=custom["skewness"],
            kurtosis=custom["kurtosis"],
            entropy=custom["entropy"]
        )
        
        # Bundle the results.
        result = OptimizationResult(
            weights=weights,
            performance=performance,
            returns_dist=dist_b64,
            max_drawdown_plot=dd_b64
        )
        
        # Compute cumulative portfolio returns.
        cum_returns = (1 + port_returns).cumprod()
        
        return result, cum_returns
    except Exception as e:
        print(f"Error in MIN_CDAR optimization: {e}")
        return None, None


def run_optimization_CLA(sub_method: str, mu, S, returns, nifty_df, risk_free_rate=0.05):
    """Run a CLA optimization using either max_sharpe (MVO) or min_volatility (MinVol)"""
    try:
        cla = CLA(mu, S)
        if sub_method.upper() == "MVO":
            cla.max_sharpe()  # Removed risk_free_rate argument
        elif sub_method.upper() == "MINVOL" or sub_method.upper() == "MIN_VOL":
            cla.min_volatility()  # Removed risk_free_rate argument
        else:
            raise ValueError(f"Invalid CLA sub-method: {sub_method}")
        
        weights = cla.clean_weights()
        pfolio_perf = cla.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        
        # Calculate portfolio returns
        w_series = pd.Series(weights)
        port_returns = returns.dot(w_series)
        
        # Compute custom metrics
        custom = compute_custom_metrics(port_returns, nifty_df, risk_free_rate)
        
        # Generate plots with a method name including the sub-method
        method_label = f"CriticalLineAlgorithm_{sub_method.upper()}"
        dist_b64, dd_b64 = generate_plots(port_returns, method_label)
        
        # Create performance object
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
            cagr=custom["cagr"],
            portfolio_beta=custom["portfolio_beta"],
            skewness=custom["skewness"],
            kurtosis=custom["kurtosis"],
            entropy=custom["entropy"]
        )
        
        # Create result object
        result = OptimizationResult(
            weights=weights,
            performance=performance,
            returns_dist=dist_b64,
            max_drawdown_plot=dd_b64
        )
        
        # Calculate cumulative returns
        cum_returns = (1 + port_returns).cumprod()
        
        return result, cum_returns
        
    except Exception as e:
        print(f"Error in CLA ({sub_method}) optimization: {e}")
        return None, None

def run_optimization_HRP(returns: pd.DataFrame, cov_matrix: pd.DataFrame, nifty_df: pd.Series, risk_free_rate=0.05, linkage_method="single"):
    """Run HRP optimization using HRPOpt"""
    try:
        hrp = HRPOpt(returns=returns, cov_matrix=cov_matrix)
        weights = hrp.optimize(linkage_method=linkage_method)
        pfolio_perf = hrp.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate, frequency=252)
        
        # Calculate portfolio returns
        w_series = pd.Series(weights)
        port_returns = returns.dot(w_series)
        
        # Compute custom metrics
        custom = compute_custom_metrics(port_returns, nifty_df, risk_free_rate)
        
        # Generate plots using "HRP" as the method label
        method_label = "HRP"
        dist_b64, dd_b64 = generate_plots(port_returns, method_label)
        
        # Create performance object
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
            cagr=custom["cagr"],
            portfolio_beta=custom["portfolio_beta"],
            skewness=custom["skewness"],
            kurtosis=custom["kurtosis"],
            entropy=custom["entropy"]
        )
        
        # Create result object
        result = OptimizationResult(
            weights=weights,
            performance=performance,
            returns_dist=dist_b64,
            max_drawdown_plot=dd_b64
        )
        
        # Calculate cumulative returns
        cum_returns = (1 + port_returns).cumprod()
        
        return result, cum_returns
        
    except Exception as e:
        print(f"Error in HRP optimization: {e}")
        return None, None

def get_risk_free_rate(start_date,end_date)->float:
    # Convert date objects to strings in YYYYMMDD format
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")

    url = f"https://stooq.com/q/d/l/?s=10yiny.b&f={start_date_str}&t={end_date_str}&i=m"
    response = requests.get(url)
    if response.status_code != 200:
        logging.error("Error fetching data from Stooq API. Status code: %s", response.status_code)
        raise Exception("Error fetching data from Stooq API.")
    # Read the CSV content into a pandas DataFrame.
    data = pd.read_csv(StringIO(response.text))
    
    # Ensure the 'Date' column is a datetime type and sort the DataFrame by Date.
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)
    
    # Check if the 'Close' column exists.
    if 'Close' not in data.columns:
        logging.error("Expected 'Close' column not found in the retrieved data.")
        raise Exception("Expected 'Close' column not found in data.")
    
    # Compute the full average of the 'Close' column.
    avg_rate = data['Close'].mean()
    
    # Log the computed full average risk-free rate (before dividing by 100).
    logging.info("Computed full average risk-free rate: %f", avg_rate)
    print(f"Computed full average risk-free rate:{avg_rate}")

    
    # Return the risk-free rate divided by 100.
    return avg_rate / 100.0

def compute_yearly_returns_stocks(daily_returns: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute yearly compounded returns for each stock in a DataFrame of daily returns.
    
    Parameters:
        daily_returns (pd.DataFrame): Daily returns for each stock (as decimals), indexed by date.
    
    Returns:
        Dict[str, Dict[str, float]]: A dictionary mapping each stock ticker to a dictionary of yearly returns.
                                     For example:
                                     {
                                       "RELIANCE.NS": {"2020": 0.12, "2021": 0.08, ...},
                                       "TCS.NS": {"2020": 0.15, "2021": 0.10, ...}
                                     }
    """
    results = {}
    for ticker in daily_returns.columns:
        # Compute the yearly compounded return for each stock.
        yearly_returns = (1 + daily_returns[ticker]).resample('Y').prod() - 1
        # Use items() instead of iteritems()
        results[ticker] = {str(date.year): ret for date, ret in yearly_returns.items()}
    return results

def generate_covariance_heatmap(
    cov_matrix: Union[pd.DataFrame, np.ndarray],
    method: str = "covariance",
    show_tickers: bool = True
) -> str:
    """
    Generate a variance–covariance matrix heatmap using seaborn and updated matplotlib settings,
    save the plot in the output directory, and return the base64‑encoded image string.
    
    The covariance matrix shows the variances on the diagonal and the covariances off-diagonal,
    which provides insight into the absolute risk (variance) of each asset and their joint movements.
    
    Parameters:
        cov_matrix (pd.DataFrame or np.ndarray): The covariance matrix to plot.
        method (str, optional): A label for naming the file. Defaults to "covariance".
        show_tickers (bool, optional): Whether to display the numerical values in the heatmap.
                                       Defaults to True.
    
    Returns:
        str: Base64‑encoded string of the saved heatmap image.
    """
    # Ensure cov_matrix is a DataFrame.
    if not isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = pd.DataFrame(cov_matrix)
    
    # Update seaborn theme and matplotlib rcParams for a modern look.
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        'figure.figsize': (10, 8),
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'savefig.dpi': 300
    })
    
    # Create the heatmap with a more modern palette ("rocket") and additional styling.
    plt.figure()
    ax = sns.heatmap(
        cov_matrix,
        annot=show_tickers,
        fmt=".2f",
        cmap="rocket",       # using the "rocket" palette for a robust look
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.75}
    )
    
    
    # Optionally remove tick labels if not desired.
    if not show_tickers:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    # Construct the file path using your global output_dir (assumed to be defined).
    filepath = os.path.join(output_dir, f"{method.lower()}_cov_heatmap.png")
    
    # Save the figure.
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    
    # Return the base64-encoded image string using your existing utility function.
    return file_to_base64(filepath)

########################################
# Main Endpoint
########################################

@app.post("/optimize")
def optimize_portfolio(request: TickerRequest = Body(...)):
    try:
        # Format tickers
        formatted_tickers = format_tickers(request.stocks)
        
        # Fetch & align data
        df, nifty_df = fetch_and_align_data(formatted_tickers)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found for the given tickers.")
        
        # Start/end date
        start_date = df.index.min().date()
        end_date = df.index.max().date()

        risk_free_rate = get_risk_free_rate(start_date,end_date)
        
        # Prepare data for optimization
        returns = df.pct_change().dropna()
        mu = expected_returns.mean_historical_return(df, frequency=252)
        S = CovarianceShrinkage(df).ledoit_wolf()
        cov_heatmap_b64 = generate_covariance_heatmap(S)

        stock_yearly_returns = compute_yearly_returns_stocks(returns)
        
        # Initialize dictionaries to hold results and cumulative returns
        results: Dict[str, Optional[OptimizationResult]] = {}
        cum_returns_df = pd.DataFrame(index=returns.index)
        
        for method in request.methods:
            if method == OptimizationMethod.HRP:
                # For HRP, use sample covariance matrix (returns.cov())
                sample_cov = returns.cov()
                result, cum_returns = run_optimization_HRP(returns, sample_cov, nifty_df,risk_free_rate)
                if result:
                    results[method.value] = result
                    cum_returns_df[method.value] = cum_returns
            elif method == OptimizationMethod.MIN_CVAR:
                result, cum_returns = run_optimization_MIN_CVAR(mu,returns,nifty_df,risk_free_rate)
                if result:
                    results[method.value] = result
                    cum_returns_df[method.value] = cum_returns
            elif method == OptimizationMethod.MIN_CDAR:
                result, cum_returns = run_optimization_MIN_CDAR(mu,returns,nifty_df,risk_free_rate)
                if result:
                    results[method.value] = result
                    cum_returns_df[method.value] = cum_returns
            elif method != OptimizationMethod.CRITICAL_LINE_ALGORITHM:
                result, cum_returns = run_optimization(method, mu, S, returns, nifty_df,risk_free_rate)
                if result:
                    results[method.value] = result
                    cum_returns_df[method.value] = cum_returns
            else:
                # Handle CLA separately
                if request.cla_method == CLAOptimizationMethod.BOTH:
                    result_mvo, cum_returns_mvo = run_optimization_CLA("MVO", mu, S, returns, nifty_df,risk_free_rate)
                    result_min_vol, cum_returns_min_vol = run_optimization_CLA("MinVol", mu, S, returns, nifty_df,risk_free_rate)
                    if result_mvo:
                        results["CriticalLineAlgorithm_MVO"] = result_mvo
                        cum_returns_df["CriticalLineAlgorithm_MVO"] = cum_returns_mvo
                    if result_min_vol:
                        results["CriticalLineAlgorithm_MinVol"] = result_min_vol
                        cum_returns_df["CriticalLineAlgorithm_MinVol"] = cum_returns_min_vol
                else:
                    sub_method = request.cla_method.value
                    result, cum_returns = run_optimization_CLA(sub_method, mu, S, returns, nifty_df,risk_free_rate)
                    if result:
                        results[f"CriticalLineAlgorithm_{sub_method}"] = result
                        cum_returns_df[f"CriticalLineAlgorithm_{sub_method}"] = cum_returns
        
        # Calculate Nifty returns
        nifty_ret = nifty_df.pct_change().dropna()
        cum_nifty = (1 + nifty_ret).cumprod()
        # Align with Nifty
        common_dates = cum_returns_df.index.intersection(cum_nifty.index)
        cum_returns_df = cum_returns_df.loc[common_dates]
        cum_nifty = cum_nifty.loc[common_dates]
        
        # Build response
        cumulative_returns = {key: cum_returns_df[key].tolist() for key in cum_returns_df.columns}
        response = PortfolioOptimizationResponse(
            results=results,
            start_date=start_date,
            end_date=end_date,
            cumulative_returns=cumulative_returns,
            dates=cum_returns_df.index.tolist(),
            nifty_returns=cum_nifty.tolist(),
            stock_yearly_returns=stock_yearly_returns,
            covariance_heatmap = cov_heatmap_b64,
            risk_free_rate = risk_free_rate
        )
        
        return jsonable_encoder(response)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
