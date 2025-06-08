from typing import List, Dict, Optional
from pydantic import BaseModel
from enum import Enum, IntEnum
from datetime import datetime
from dataclasses import dataclass

########################################
# Error Codes
########################################
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

########################################
# Enums
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
    HERC = "HERC"  # Hierarchical Equal Risk Contribution
    NCO = "NCO"    # Nested Clustered Optimization
    HERC2 = "HERC2"  # Another variant of HERC
    TECHNICAL = "TECHNICAL"  # Technical indicator-based optimization

# New enum for CLA sub-methods
class CLAOptimizationMethod(str, Enum):
    MVO = "MVO"
    MIN_VOL = "MinVol"
    BOTH = "Both"

class BenchmarkName(str, Enum):
    nifty      = "nifty"
    sensex     = "sensex"
    bank_nifty = "bank_nifty"

@dataclass(frozen=True)
class Benchmark:
    name: BenchmarkName
    ticker: str

class Benchmarks:
    NIFTY = Benchmark(BenchmarkName.nifty, "^NSEI")
    SENSEX = Benchmark(BenchmarkName.sensex, "^BSESN")
    BANK_NIFTY = Benchmark(BenchmarkName.bank_nifty, "^NSEBANK")
    
    @classmethod
    def get_ticker(cls, name: BenchmarkName) -> str:
        for attr, benchmark in cls.__dict__.items():
            if isinstance(benchmark, Benchmark) and benchmark.name == name:
                return benchmark.ticker
        raise ValueError(f"Unknown benchmark: {name}")

########################################
# Pydantic Models
########################################
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
    portfolio_alpha: float = 0.0
    beta_pvalue: float = 1.0
    r_squared: float = 0.0
    blume_adjusted_beta: float = 0.0
    treynor_ratio: float = 0.0
    skewness: float
    kurtosis: float
    entropy: float
    # Additional beta and cross-moment metrics
    welch_beta: Optional[float] = None
    semi_beta: Optional[float] = None
    coskewness: Optional[float] = None
    cokurtosis: Optional[float] = None
    vasicek_beta: Optional[float] = None
    james_stein_beta: Optional[float] = None
    # garch_beta: Optional[float] = None  # Commented out temporarily - computationally intensive
    # Other metrics
    omega_ratio: float
    calmar_ratio: float
    ulcer_index: float
    evar_95: float
    gini_mean_difference: float
    dar_95: float
    cdar_95: float
    upside_potential_ratio:float
    modigliani_risk_adjusted_performance: float
    information_ratio: float
    sterling_ratio: float
    v2_ratio: float

class OptimizationResult(BaseModel):
    weights: Dict[str, float]
    performance: PortfolioPerformance
    returns_dist: Optional[str] = None
    max_drawdown_plot: Optional[str] = None
    rolling_betas: Optional[Dict[int, float]] = None

class PortfolioOptimizationResponse(BaseModel):
    results: Dict[str, Optional[OptimizationResult]]
    start_date: datetime
    end_date: datetime
    cumulative_returns: Dict[str, List[Optional[float]]]
    dates: List[datetime]
    benchmark_returns: List[BenchmarkReturn]
    stock_yearly_returns: Optional[Dict[str, Dict[str, float]]]
    covariance_heatmap: Optional[str] = None
    risk_free_rate: float
    is_technical_only: bool = False  # True if only technical methods were used
    technical_start_date: Optional[datetime] = None  # Start date for technical optimization
    technical_end_date: Optional[datetime] = None  # End date for technical optimization
    technical_risk_free_rate: Optional[float] = None  # Risk-free rate for technical optimization

class StockItem(BaseModel):
    ticker: str
    exchange: ExchangeEnum

class TechnicalIndicator(BaseModel):
    name: str
    window: int  # Required field for most indicators
    mult: Optional[float] = None  # Optional only for SUPERTREND

class TickerRequest(BaseModel):
    stocks: List[StockItem]
    methods: List[OptimizationMethod] = [OptimizationMethod.MVO]
    cla_method: Optional[CLAOptimizationMethod] = CLAOptimizationMethod.BOTH
    benchmark: BenchmarkName = BenchmarkName.nifty  # Default to Nifty
    indicators: List[TechnicalIndicator] = [] 