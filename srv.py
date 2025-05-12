from typing import List, Dict, Optional, Tuple, Union
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from enum import Enum, IntEnum
from typing import Sequence
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
import sys
from dotenv import load_dotenv
import logfire
from fastapi.responses import JSONResponse
import time
import uuid
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- MOSEK License Configuration ----
def configure_mosek_license():
    """Configure MOSEK license by setting the appropriate environment variable.
    
    This function looks for the MOSEK license in the following locations:
    1. Environment variable MOSEK_LICENSE_CONTENT (base64 encoded license)
    2. Environment variable MOSEK_LICENSE_PATH (path to license file)
    3. Default locations: './mosek/mosek.lic', './mosek.lic', '~/mosek/mosek.lic'
    
    Returns:
        bool: True if a license was found and configured, False otherwise
    """
    # Check if license content is provided as environment variable (CI/CD)
    mosek_license_content = os.environ.get('MOSEK_LICENSE_CONTENT')
    if mosek_license_content:
        try:
            # Decode the base64 content
            license_content = base64.b64decode(mosek_license_content).decode('utf-8')
            # Create a temporary license file in the current directory
            license_path = os.path.abspath(os.path.join(os.getcwd(), 'mosek', 'mosek.lic'))
            os.makedirs(os.path.dirname(license_path), exist_ok=True)
            
            with open(license_path, 'w') as f:
                f.write(license_content)
                
            # Set the license path environment variable
            os.environ['MOSEKLM_LICENSE_FILE'] = license_path
            logger.info(f"MOSEK license configured from environment variable at path: {license_path}")
            
            # Log the current value of the environment variable
            current_env = os.environ.get('MOSEKLM_LICENSE_FILE', 'Not set')
            logger.info(f"Current MOSEKLM_LICENSE_FILE value: {current_env}")
            
            return True
        except Exception as e:
            logger.warning(f"Failed to configure MOSEK license from environment variable: {e}")
    
    # Check if license path is provided
    mosek_license_path = os.environ.get('MOSEK_LICENSE_PATH')
    if mosek_license_path and os.path.exists(mosek_license_path):
        os.environ['MOSEKLM_LICENSE_FILE'] = mosek_license_path
        logger.info(f"MOSEK license configured from path: {mosek_license_path}")
        return True
    
    # Check common locations
    common_paths = [
        os.path.abspath(os.path.join(os.getcwd(), 'mosek', 'mosek.lic')),
        os.path.abspath(os.path.join(os.getcwd(), 'mosek.lic')),
        os.path.expanduser('~/mosek/mosek.lic'),
        '/app/mosek/mosek.lic',  # Docker container path
        '/root/mosek/mosek.lic'   # Alternative Docker path
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            os.environ['MOSEKLM_LICENSE_FILE'] = path
            logger.info(f"MOSEK license configured from path: {path}")
            return True
    
    logger.warning("MOSEK license not found. Optimization methods requiring MOSEK will use fallbacks.")
    return False

# ---- Custom Error Handling ----
class ErrorCode(IntEnum):
    """Enumeration of API error codes for detailed error reporting"""
    # Input validation errors (400 range)
    INSUFFICIENT_STOCKS = 40001
    NO_DATA_FOUND = 40002
    INVALID_TICKER = 40003
    INVALID_DATE_RANGE = 40004
    INVALID_OPTIMIZATION_METHOD = 40005
    
    # Processing errors (500 range)
    OPTIMIZATION_FAILED = 50001
    DATA_FETCH_ERROR = 50002
    RISK_FREE_RATE_ERROR = 50003
    COVARIANCE_CALCULATION_ERROR = 50004
    UNEXPECTED_ERROR = 50099

class APIError(Exception):
    """Custom API exception with error code, message, and HTTP status code"""
    def __init__(self, code: ErrorCode, message: str, status_code: int = 400, details: Optional[Dict] = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

# ── Logger & Handlers Setup ───────────────────────────────────────────────────
# Set up root logger first for all logs including FastAPI and Uvicorn
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Application logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid duplication
for handler in root_logger.handlers:
    root_logger.removeHandler(handler)
for handler in logger.handlers:
    logger.removeHandler(handler)

# Configure Logfire handler only
load_dotenv()  # loads your .env containing LOGFIRE_TOKEN
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
logfire.configure(token=LOGFIRE_TOKEN, environment="production")
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
lf_handler = logfire.LogfireLoggingHandler()
lf_handler.setLevel(logging.INFO)
lf_handler.setFormatter(formatter)
logger.addHandler(lf_handler)
root_logger.addHandler(lf_handler)

# Prevent double-logging up the hierarchy
logger.propagate = False
# ──────────────────────────────────────────────────────────────────────────────

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

# Configure MOSEK license
has_mosek_license = configure_mosek_license()

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

# Add request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    
    # Generate a request ID
    request_id = str(uuid.uuid4())
    
    # Extract client details
    client_host = request.client.host if request.client else "unknown"
    client_port = request.client.port if request.client else 0
    
    # Log request details in a format similar to Uvicorn's access logs
    request_line = f"{request.method} {request.url.path}"
    if request.url.query:
        request_line += f"?{request.url.query}"
    
    logger.info(
        f"{client_host}:{client_port} - \"{request_line} HTTP/{request.scope.get('http_version', '1.1')}\"",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.url.query),
            "client_host": client_host,
            "client_port": client_port,
            "headers": dict(request.headers),
            "http_version": request.scope.get("http_version", "1.1")
        }
    )
    
    # Process the request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        process_time_ms = round(process_time * 1000, 2)
        
        # Log successful response in a format similar to Uvicorn's access logs
        logger.info(
            f"{client_host}:{client_port} - \"{request_line} HTTP/{request.scope.get('http_version', '1.1')}\" {response.status_code} {process_time_ms}ms",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time_ms": process_time_ms,
                "client_host": client_host,
                "client_port": client_port,
                "path": request.url.path,
                "method": request.method
            }
        )
        
        # Add custom header with processing time
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        # Log failed response
        process_time = time.time() - start_time
        process_time_ms = round(process_time * 1000, 2)
        
        logger.error(
            f"{client_host}:{client_port} - \"{request_line} HTTP/{request.scope.get('http_version', '1.1')}\" ERROR {process_time_ms}ms: {str(e)}",
            extra={
                "request_id": request_id,
                "error": str(e),
                "process_time_ms": process_time_ms,
                "client_host": client_host,
                "client_port": client_port,
                "path": request.url.path,
                "method": request.method
            },
            exc_info=True
        )
        raise

# Register exception handlers
@app.exception_handler(APIError)
async def api_exception_handler(request, exc: APIError):
    """Handle APIError exceptions by returning a formatted JSON response"""
    logger.error("API Error: %s - %s", exc.code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": int(exc.code),
                "message": exc.message,
                "details": exc.details
            }
        }
    )

# Add a generic exception handler for unexpected errors
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception):
    """Handle any unhandled exceptions"""
    logger.exception("Unhandled exception occurred")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": int(ErrorCode.UNEXPECTED_ERROR),
                "message": "An unexpected error occurred",
                "details": {"error_type": str(type(exc).__name__)}
            }
        }
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

class BenchmarkName(str, Enum):
    nifty      = "nifty"
    sensex     = "sensex"
    bank_nifty = "bank_nifty"

# Add this after the BenchmarkName enum definition
BENCHMARK_TICKERS = {
    BenchmarkName.nifty: "^NSEI",
    BenchmarkName.sensex: "^BSESN",
    BenchmarkName.bank_nifty: "^NSEBANK"
}

class BenchmarkReturn(BaseModel):
    name: BenchmarkName
    returns: List[float]   
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
    blume_adjusted_beta : float
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
    benchmark_returns: List[BenchmarkReturn]
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
    benchmark: BenchmarkName = BenchmarkName.nifty  # Default to Nifty

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
    return yf.download(ticker, start=start_date, multi_level_index=False, progress=False, auto_adjust=True)['Close']

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

def fetch_and_align_data(tickers: List[str], benchmark_ticker: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Download & align data for each ticker, plus benchmark index.
    Returns (combined_df, benchmark_close).
    """
    data = {}
    failed_tickers = []
    
    for ticker in tickers:
        try:
            df = cached_yf_download(ticker, datetime(1990, 1, 1))
            if not df.empty:
                data[ticker] = df
            else:
                logger.warning("No data for ticker %s", ticker)
                failed_tickers.append(ticker)
        except Exception as e:
            logger.exception("Error fetching data for %s: %s", ticker, str(e))
            failed_tickers.append(ticker)

    if not data:
        logger.warning("No valid data available for the provided tickers")
        details = {"failed_tickers": failed_tickers}
        if failed_tickers:
            details["last_ticker"] = failed_tickers[-1]
            
        raise APIError(
            code=ErrorCode.NO_DATA_FOUND,
            message="No valid data available for the provided tickers",
            details=details
        )

    # Align all tickers to the latest min_date among them
    min_date = max(df.index.min() for df in data.values())
    filtered_data = {t: df[df.index >= min_date] for t, df in data.items()}

    # Combine into multi-index DataFrame
    combined_df = pd.concat(filtered_data.values(), axis=1, keys=filtered_data.keys())
    combined_df.dropna(inplace=True)

    # Fetch benchmark data
    try:
        benchmark_df = yf.download(benchmark_ticker, start=min_date, multi_level_index=False, progress=False, auto_adjust=True)['Close'].dropna()
    except Exception as e:
        logger.exception("Error fetching benchmark index: %s", str(e))
        raise APIError(
            code=ErrorCode.DATA_FETCH_ERROR,
            message="Error fetching market index data",
            status_code=500,
            details={"error": str(e)}
        )
    
    if benchmark_df.empty:
        raise APIError(
            code=ErrorCode.NO_DATA_FOUND,
            message="No data available for benchmark index",
            status_code=500
        )
    
    # Align with market index
    common_dates = combined_df.index.intersection(benchmark_df.index)
    if len(common_dates) == 0:
        raise APIError(
            code=ErrorCode.INVALID_DATE_RANGE,
            message="No overlapping dates between stock data and market index",
            status_code=400,
            details={
                "stock_date_range": [str(combined_df.index.min().date()), str(combined_df.index.max().date())],
                "index_date_range": [str(benchmark_df.index.min().date()), str(benchmark_df.index.max().date())]
            }
        )
        
    combined_df = combined_df.loc[common_dates]
    benchmark_df = benchmark_df.loc[common_dates]
    
    # Add warning if some tickers failed
    if failed_tickers:
        logger.warning("Some tickers failed to fetch data: %s", failed_tickers)
    
    return combined_df, benchmark_df

def freedman_diaconis_bins(port_returns: pd.Series) -> int:
    n = len(port_returns)
    logger.info("Computing bins for %d data points", n)
    if n < 3:
        logger.info("Not enough data points. Returning 1 bin.")
        return 1  # minimal bin count for very small datasets

    # Calculate IQR (Interquartile Range)
    iqr = port_returns.quantile(0.75) - port_returns.quantile(0.25)

    # Compute bin width using Freedman-Diaconis rule
    bin_width = 2 * iqr / np.cbrt(n)
    logger.info("Computed bin width: %f", bin_width)
    
    if bin_width == 0:
        logger.info("Bin width is zero; returning 50 bins.")
        return 50

    # Calculate the number of bins over the data range
    data_range = port_returns.max() - port_returns.min()
    bins = int(np.ceil(data_range / bin_width))
    logger.info("No. of bins computed based on Freedman-Diaconis rule: %d", bins)
    return bins if bins > 0 else 50


def compute_custom_metrics(port_returns: pd.Series, benchmark_df: pd.Series, risk_free_rate: float = 0.05) -> Dict[str, float]:
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
    benchmark_ret = benchmark_df.pct_change().dropna()
    
    # Initialize excess returns with empty series
    risk_free_daily = risk_free_rate / ann_factor
    excess_portfolio = pd.Series(dtype=float)
    excess_market = pd.Series(dtype=float)
    
    # Only proceed with beta calculation if we have benchmark data
    if not benchmark_ret.empty:
        merged_returns = pd.DataFrame({
            'Portfolio': port_returns,
            'Benchmark': benchmark_ret
        }).dropna()
        
        if not merged_returns.empty:
            excess_portfolio = merged_returns['Portfolio'] - risk_free_daily
            excess_market = merged_returns['Benchmark'] - risk_free_daily
    
    # Initialize portfolio_beta with a default value
    portfolio_beta = 0.0
    
    # Only calculate beta if we have enough data
    if len(excess_portfolio) > 1 and len(excess_market) > 1:
        # Check if we have enough variation to calculate beta
        if excess_market.var() > 1e-9:
            # Prepare and run regression
            X = sm.add_constant(excess_market)  # Adds intercept term
            
            try:
                model = sm.OLS(excess_portfolio, X).fit()
                portfolio_beta = model.params['Benchmark']
            except Exception as e:
                logger.warning(f"Error calculating beta: {e}")
                # Keep default beta of 0.0
    b = 0.67 #Bloom Adjustment Factor
    blume_adjusted_beta = 1 + (b * (portfolio_beta - 1))
    skewness = port_returns.skew()
    kurtosis = port_returns.kurt()
    
    # Calculate entropy safely
    bins = freedman_diaconis_bins(port_returns)
    counts, _ = np.histogram(port_returns, bins=bins) if len(port_returns) > 1 else ([1], [0, 1])
    
    # Handle zero counts
    counts = np.array([c if c > 0 else 1 for c in counts])
    probs = counts / counts.sum()
    port_entropy = entropy(probs)
    
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
        "blume_adjusted_beta": blume_adjusted_beta,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "entropy": port_entropy
    }

@lru_cache(maxsize=64)
def cached_covariance_matrix(tickers: Tuple[str], start_date: datetime):
    """Cache the covariance matrix computation for a given set of tickers and start date."""
    df = pd.concat([cached_yf_download(t, start_date) for t in tickers], axis=1)
    df.dropna(inplace=True)
    return CovarianceShrinkage(df).ledoit_wolf()

@lru_cache(maxsize=64)
def cached_benchmark_returns(ticker: str, start_date: datetime):
    """Cache benchmark returns for a given ticker and start date."""
    return yf.download(ticker, start=start_date, progress=False, auto_adjust=True)['Close'].pct_change().dropna()

@lru_cache(maxsize=32)
def cached_risk_free_rate(start_date: datetime, end_date: datetime):
    """Cache risk-free rate for a given date range."""
    return get_risk_free_rate(start_date, end_date)

def generate_plots(port_returns: pd.Series, method: str) -> Tuple[str, str]:
    """Generate distribution and drawdown plots in memory, return base64 encoded images"""
    bins = freedman_diaconis_bins(port_returns)
    
    # Plot distribution (histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(port_returns, bins=bins, edgecolor='black', alpha=0.7, label='Daily Returns')
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

    # Save to BytesIO instead of file
    buf_hist = BytesIO()
    plt.savefig(buf_hist, format='png', bbox_inches='tight')
    plt.close()
    dist_b64 = base64.b64encode(buf_hist.getvalue()).decode('utf-8')

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
    
    # Save to BytesIO instead of file
    buf_dd = BytesIO()
    plt.savefig(buf_dd, format='png', bbox_inches='tight')
    plt.close()
    dd_b64 = base64.b64encode(buf_dd.getvalue()).decode('utf-8')
    
    return dist_b64, dd_b64

def run_optimization(method: OptimizationMethod, mu, S, returns, benchmark_df, risk_free_rate=0.05):
    """Run a specific optimization method (non-CLA, non-HRP) and return the results"""
    try:
        if method == OptimizationMethod.MVO:
            ef = EfficientFrontier(mu, S)
            try:
                # Check if any assets have returns above risk-free rate
                if (mu > risk_free_rate).any():
                    ef.max_sharpe(risk_free_rate=risk_free_rate)
                else:
                    # If no assets have expected returns above risk-free rate,
                    # use a lower risk-free rate to allow optimization to proceed
                    logger.warning("No assets have expected returns above risk-free rate. Using lower risk-free rate.")
                    adjusted_rf = mu.max() * 0.9  # Use 90% of the highest expected return
                    ef.max_sharpe(risk_free_rate=adjusted_rf)
            except Exception as e:
                logger.exception("Error in max_sharpe optimization")
                raise
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
        custom = compute_custom_metrics(port_returns, benchmark_df, risk_free_rate)
        
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
            blume_adjusted_beta = custom["blume_adjusted_beta"],
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
        logger.exception("Error in %s optimization", method.value)
        return None, None
    
def run_optimization_MIN_CVAR(mu, returns, benchmark_df, risk_free_rate=0.05):
    try:
        # Check if the MOSEK license is configured
        mosek_env_var = os.environ.get('MOSEKLM_LICENSE_FILE', None)
        mosek_available = mosek_env_var is not None and os.path.exists(mosek_env_var)
        
        if not mosek_available:
            logger.warning(f"MOSEK license file not found at '{mosek_env_var}'. Using min_volatility as fallback for MIN_CVAR.")
            # Create a standard EfficientFrontier object instead
            ef_alt = EfficientFrontier(mu, returns.cov())
            # Min volatility as a proxy since we can't do min_cvar
            ef_alt.min_volatility()
            weights = ef_alt.clean_weights()
            
            # Calculate standard performance metrics
            pfolio_perf = ef_alt.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            logger.info("Used min_volatility as fallback for min_cvar (no MOSEK license)")
        else:
            # Try EfficientCVaR optimization with MOSEK solver
            try:
                # Initialize EfficientCVaR with the provided expected returns and historical returns.
                logger.info(f"Attempting MIN_CVAR optimization with MOSEK license at: {mosek_env_var}")
                ef_cvar = EfficientCVaR(expected_returns=mu, returns=returns, beta=0.95, weight_bounds=(0, 1))
                weights = ef_cvar.min_cvar()
                
                # Get portfolio metrics
                pfolio_perf = ef_cvar.portfolio_performance(verbose=False)
                logger.info("Successfully used MOSEK solver for MIN_CVAR optimization")
                
            except Exception as solver_error:
                # If any error occurs during MOSEK optimization, fall back to standard optimization
                logger.warning("Error in CVaR optimization: %s. Using min_volatility as fallback.", str(solver_error))
                
                # Create a standard EfficientFrontier object instead
                ef_alt = EfficientFrontier(mu, returns.cov())
                # Min volatility as a proxy since we can't do min_cvar
                ef_alt.min_volatility()
                weights = ef_alt.clean_weights()
                
                # Calculate standard performance metrics
                pfolio_perf = ef_alt.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                logger.info("Used min_volatility as fallback for min_cvar (MOSEK error)")

        # Calculate portfolio returns for custom metrics
        w_series = pd.Series(weights)
        port_returns = returns.dot(w_series)
        
        # Compute custom metrics
        custom = compute_custom_metrics(port_returns, benchmark_df, risk_free_rate)
        
        # Generate plots
        dist_b64, dd_b64 = generate_plots(port_returns, OptimizationMethod.MIN_CVAR.value)
        
        # Handle case where portfolio_performance doesn't return sharpe ratio
        expected_return = pfolio_perf[0] if len(pfolio_perf) > 0 else 0.0
        volatility = pfolio_perf[1] if len(pfolio_perf) > 1 else 0.0
        # If sharpe ratio is missing, calculate it manually or use a default
        if len(pfolio_perf) > 2:
            sharpe = pfolio_perf[2]
        else:
            # Calculate manually if we have expected_return and volatility
            sharpe = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0.0
            logger.info("Calculated sharpe ratio manually: %f", sharpe)
        
        # Create performance object
        performance = PortfolioPerformance(
            expected_return=expected_return,
            volatility=volatility,
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
            blume_adjusted_beta = custom["blume_adjusted_beta"],
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
        logger.exception("Error in MIN_CVAR optimization")
        return None, None

def run_optimization_MIN_CDAR(mu, returns, benchmark_df, risk_free_rate=0.05):
    try:
        # Check if the MOSEK license is configured
        mosek_env_var = os.environ.get('MOSEKLM_LICENSE_FILE', None)
        mosek_available = mosek_env_var is not None and os.path.exists(mosek_env_var)
        
        if not mosek_available:
            logger.warning(f"MOSEK license file not found at '{mosek_env_var}'. Using min_volatility as fallback for MIN_CDAR.")
            # Create a standard EfficientFrontier object instead
            ef_alt = EfficientFrontier(mu, returns.cov())
            # Min volatility as a proxy since we can't do min_cdar
            ef_alt.min_volatility()
            weights = ef_alt.clean_weights()
            
            # Calculate standard performance metrics
            pfolio_perf = ef_alt.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            logger.info("Used min_volatility as fallback for min_cdar (no MOSEK license)")
        else:
            # Try EfficientCDaR optimization with MOSEK solver
            try:
                # Initialize EfficientCDaR with the provided expected returns and historical returns.
                logger.info(f"Attempting MIN_CDAR optimization with MOSEK license at: {mosek_env_var}")
                ef_cdar = EfficientCDaR(expected_returns=mu, returns=returns, beta=0.95, weight_bounds=(0, 1))
                weights = ef_cdar.min_cdar()
                
                # Get portfolio metrics
                pfolio_perf = ef_cdar.portfolio_performance(verbose=False)
                logger.info("Successfully used MOSEK solver for MIN_CDAR optimization")
                
            except Exception as solver_error:
                # If any error occurs during MOSEK optimization, fall back to standard optimization
                logger.warning("Error in CDaR optimization: %s. Using min_volatility as fallback.", str(solver_error))
                
                # Create a standard EfficientFrontier object instead
                ef_alt = EfficientFrontier(mu, returns.cov())
                # Min volatility as a proxy since we can't do min_cdar
                ef_alt.min_volatility()
                weights = ef_alt.clean_weights()
                
                # Calculate standard performance metrics
                pfolio_perf = ef_alt.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                logger.info("Used min_volatility as fallback for min_cdar (MOSEK error)")

        # Calculate portfolio returns for custom metrics
        w_series = pd.Series(weights)
        port_returns = returns.dot(w_series)
        
        # Compute custom metrics
        custom = compute_custom_metrics(port_returns, benchmark_df, risk_free_rate)
        
        # Generate plots
        dist_b64, dd_b64 = generate_plots(port_returns, OptimizationMethod.MIN_CDAR.value)
        
        # Handle case where portfolio_performance doesn't return sharpe ratio
        expected_return = pfolio_perf[0] if len(pfolio_perf) > 0 else 0.0
        volatility = pfolio_perf[1] if len(pfolio_perf) > 1 else 0.0
        # If sharpe ratio is missing, calculate it manually or use a default
        if len(pfolio_perf) > 2:
            sharpe = pfolio_perf[2]
        else:
            # Calculate manually if we have expected_return and volatility
            sharpe = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0.0
            logger.info("Calculated sharpe ratio manually for MIN_CDAR: %f", sharpe)
        
        # Create performance object
        performance = PortfolioPerformance(
            expected_return=expected_return,
            volatility=volatility,
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
            blume_adjusted_beta = custom["blume_adjusted_beta"],
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
        logger.exception("Error in MIN_CDAR optimization")
        return None, None


def run_optimization_CLA(sub_method: str, mu, S, returns, benchmark_df, risk_free_rate=0.05):
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
        custom = compute_custom_metrics(port_returns, benchmark_df, risk_free_rate)
        
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
            blume_adjusted_beta = custom["blume_adjusted_beta"],
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
        logger.exception("Error in CLA %s optimization", sub_method)
        return None, None

def run_optimization_HRP(returns: pd.DataFrame, cov_matrix: pd.DataFrame, benchmark_df: pd.Series, risk_free_rate=0.05, linkage_method="single"):
    """Run HRP optimization using HRPOpt"""
    try:
        hrp = HRPOpt(returns=returns, cov_matrix=cov_matrix)
        weights = hrp.optimize(linkage_method=linkage_method)
        pfolio_perf = hrp.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate, frequency=252)
        
        # Calculate portfolio returns
        w_series = pd.Series(weights)
        port_returns = returns.dot(w_series)
        
        # Compute custom metrics
        custom = compute_custom_metrics(port_returns, benchmark_df, risk_free_rate)
        
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
            blume_adjusted_beta = custom["blume_adjusted_beta"],
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
        logger.exception("Error in HRP optimization")
        return None, None

def get_risk_free_rate(start_date, end_date) -> float:
    """
    Fetch risk-free rate data for the given date range.
    
    Args:
        start_date: Start date of the period
        end_date: End date of the period
        
    Returns:
        float: The average risk-free rate as a decimal (e.g., 0.05 for 5%)
        
    Raises:
        APIError: If data cannot be fetched or processed
    """
    # Convert date objects to strings in YYYYMMDD format
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")

    url = f"https://stooq.com/q/d/l/?s=10yiny.b&f={start_date_str}&t={end_date_str}&i=m"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            # Downgrade to warning instead of error for 404s
            logger.warning("Could not fetch data from Stooq API. Status code: %s. Using default risk-free rate of 0.05", response.status_code)
            return 0.05
            
        # Check if the response body is empty or too small
        if len(response.text) < 10:  # Just a basic sanity check
            logger.warning("Empty or invalid response from Stooq API. Using default risk-free rate of 0.05")
            return 0.05
            
        # Read the CSV content into a pandas DataFrame
        data = pd.read_csv(StringIO(response.text))
        
        # Ensure the 'Date' column is a datetime type and sort the DataFrame by Date
        if 'Date' not in data.columns:
            logger.warning("Expected 'Date' column not found in the retrieved data. Using default risk-free rate of 0.05")
            return 0.05
            
        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values('Date', inplace=True)
        
        # Check if the 'Close' column exists
        if 'Close' not in data.columns:
            logger.warning("Expected 'Close' column not found in the retrieved data. Using default risk-free rate of 0.05")
            return 0.05
        
        # Check if data is empty after filtering
        if data.empty:
            logger.warning("No risk-free rate data available for the specified period. Using default risk-free rate of 0.05")
            return 0.05
        
        # Compute the full average of the 'Close' column
        avg_rate = data['Close'].mean()
        
        # Log the computed full average risk-free rate (before dividing by 100)
        logger.info("Computed full average risk-free rate: %f", avg_rate)
        
        # Convert from percentage to decimal and handle negative rates
        final_rf = avg_rate / 100.0
        if final_rf < 0:
            logger.warning("Negative risk-free rate (%f) detected, using default 0.05", final_rf)
            return 0.05
            
        return final_rf
        
    except Exception as e:
        # For any exception, use default risk-free rate with warning
        logger.warning("Unexpected error getting risk-free rate: %s. Using default risk-free rate of 0.05", str(e))
        return 0.05

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
        # Using 'YE' instead of deprecated 'Y' frequency
        yearly_returns = (1 + daily_returns[ticker]).resample('YE').prod() - 1
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

def parallel_optimizations(methods, mu, S, returns, benchmark_df, risk_free_rate, cla_method):
    """Run multiple optimization methods in parallel using ThreadPoolExecutor."""
    results = {}
    with ThreadPoolExecutor() as executor:
        future_to_method = {}
        
        for method in methods:
            if method == OptimizationMethod.CRITICAL_LINE_ALGORITHM:
                # For CLA, we need to handle both sub-methods if requested
                if cla_method == CLAOptimizationMethod.BOTH:
                    future_to_method[executor.submit(run_optimization_CLA, "MVO", mu, S, returns, benchmark_df, risk_free_rate)] = f"{method.value}_MVO"
                    future_to_method[executor.submit(run_optimization_CLA, "MinVol", mu, S, returns, benchmark_df, risk_free_rate)] = f"{method.value}_MinVol"
                else:
                    future_to_method[executor.submit(run_optimization_CLA, cla_method.value, mu, S, returns, benchmark_df, risk_free_rate)] = method.value
            elif method == OptimizationMethod.MIN_CVAR:
                future_to_method[executor.submit(run_optimization_MIN_CVAR, mu, returns, benchmark_df, risk_free_rate)] = method.value
            elif method == OptimizationMethod.MIN_CDAR:
                future_to_method[executor.submit(run_optimization_MIN_CDAR, mu, returns, benchmark_df, risk_free_rate)] = method.value
            elif method == OptimizationMethod.HRP:
                future_to_method[executor.submit(run_optimization_HRP, returns, S, benchmark_df, risk_free_rate)] = method.value
            else:
                future_to_method[executor.submit(run_optimization, method, mu, S, returns, benchmark_df, risk_free_rate)] = method.value
        
        for future in as_completed(future_to_method):
            method_name = future_to_method[future]
            try:
                result = future.result()
                results[method_name] = result
            except Exception as e:
                logger.exception("Error in %s optimization", method_name)
                results[method_name] = (None, None)
    
    return results

@app.post("/optimize")
def optimize_portfolio(request: TickerRequest = Body(...)):
    try:
        # Format tickers
        formatted_tickers = format_tickers(request.stocks)
        logger.info("Stock tickers chosen: %s", formatted_tickers)
        if len(formatted_tickers) < 2:
            raise APIError(
                code=ErrorCode.INSUFFICIENT_STOCKS,
                message="Minimum of 2 stocks required for portfolio optimization",
                details={"provided_count": len(formatted_tickers)}
            )
        
        # Get benchmark ticker
        benchmark_ticker = BENCHMARK_TICKERS[request.benchmark]
        logger.info("Using benchmark: %s (ticker: %s)", request.benchmark.value, benchmark_ticker)
        
        # Fetch & align data
        try:
            df, benchmark_df = fetch_and_align_data(formatted_tickers, benchmark_ticker)
        except ValueError as e:
            if "No valid data available" in str(e):
                raise APIError(
                    code=ErrorCode.NO_DATA_FOUND,
                    message="No valid data available for the provided tickers",
                    details={"tickers": formatted_tickers}
                )
            raise  # Re-raise if it's a different ValueError
            
        if df.empty:
            raise APIError(
                code=ErrorCode.NO_DATA_FOUND,
                message="No data found for the given tickers",
                details={"tickers": formatted_tickers}
            )
        
        # Start/end date
        start_date = df.index.min().date()
        end_date = df.index.max().date()

        # Get risk-free rate using cached function
        try:
            risk_free_rate = cached_risk_free_rate(start_date, end_date)
        except Exception as e:
            logger.warning("Error fetching risk-free rate: %s. Using default 0.05", str(e))
            risk_free_rate = 0.05  # Default fallback
        
        # Prepare data for optimization
        try:
            returns = df.pct_change().dropna()
            mu = expected_returns.mean_historical_return(df, frequency=252)
            # Use cached covariance matrix
            S = cached_covariance_matrix(tuple(formatted_tickers), datetime(1990, 1, 1))
            cov_heatmap_b64 = generate_covariance_heatmap(S)
            stock_yearly_returns = compute_yearly_returns_stocks(returns)
        except Exception as e:
            logger.exception("Error preparing data for optimization")
            raise APIError(
                code=ErrorCode.DATA_FETCH_ERROR,
                message="Error preparing data for optimization",
                status_code=500,
                details={"error": str(e)}
            )
        
        # Run optimizations in parallel
        optimization_results = parallel_optimizations(
            request.methods,
            mu,
            S,
            returns,
            benchmark_df,
            risk_free_rate,
            request.cla_method
        )
        
        # Process results
        results: Dict[str, Optional[OptimizationResult]] = {}
        cum_returns_df = pd.DataFrame(index=returns.index)
        failed_methods = []
        
        for method_name, (opt_result, cum_returns) in optimization_results.items():
            if opt_result:
                results[method_name] = opt_result
                if cum_returns is not None:
                    cum_returns_df[method_name] = cum_returns
            else:
                failed_methods.append(method_name)
        
        # If all methods failed, return an error
        if len(results) == 0:
            raise APIError(
                code=ErrorCode.OPTIMIZATION_FAILED,
                message="All optimization methods failed",
                status_code=500,
                details={"failed_methods": failed_methods}
            )
        
        # Calculate benchmark returns using cached function
        benchmark_ret = cached_benchmark_returns(benchmark_ticker, datetime(1990, 1, 1))
        cum_benchmark = (1 + benchmark_ret).cumprod()
        
        # Align with benchmark
        common_dates = cum_returns_df.index.intersection(cum_benchmark.index)
        cum_returns_df = cum_returns_df.loc[common_dates]
        cum_benchmark = cum_benchmark.loc[common_dates]
        
        # Build response
        cumulative_returns = {key: cum_returns_df[key].tolist() for key in cum_returns_df.columns}
        response = PortfolioOptimizationResponse(
            results=results,
            start_date=start_date,
            end_date=end_date,
            cumulative_returns=cumulative_returns,
            dates=cum_returns_df.index.tolist(),
            benchmark_returns=[BenchmarkReturn(name=request.benchmark, returns=cum_benchmark.tolist())],
            stock_yearly_returns=stock_yearly_returns,
            covariance_heatmap=cov_heatmap_b64,
            risk_free_rate=risk_free_rate
        )
        
        # Include a warning in the response if some methods failed
        response_data = jsonable_encoder(response)
        if failed_methods:
            response_data["warnings"] = {
                "failed_methods": failed_methods,
                "message": "Some optimization methods failed to complete"
            }
        
        return response_data
    
    except APIError:
        # Let our custom exception handler deal with this
        raise
    except ValueError as e:
        # Handle other ValueError exceptions not caught earlier
        logger.exception("ValueError in optimize_portfolio: %s", str(e))
        raise APIError(
            code=ErrorCode.INVALID_TICKER,
            message=str(e),
            status_code=400
        )
    except Exception as e:
        # Let our generic exception handler deal with unexpected errors
        logger.exception("Unexpected error in optimize_portfolio")
        raise APIError(
            code=ErrorCode.UNEXPECTED_ERROR,
            message=str(e),
            status_code=503
        )