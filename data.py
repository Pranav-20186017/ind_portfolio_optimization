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
    INVALID_BUDGET = 40006
    BUDGET_TOO_SMALL = 40007
    ALLOCATION_INFEASIBLE = 40008
    MIN_NAMES_INFEASIBLE = 40009
    
    # Processing errors (500 range)
    OPTIMIZATION_FAILED = 50001
    DATA_FETCH_ERROR = 50002
    RISK_FREE_RATE_ERROR = 50003
    COVARIANCE_CALCULATION_ERROR = 50004
    DIVIDEND_DATA_INSUFFICIENT = 50005
    DIVIDEND_FETCH_ERROR = 50006
    RISK_CONSTRAINT_FAILED = 50007
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

class DividendOptimizationMethod(str, Enum):
    AUTO = "AUTO"          # Intelligent greedy/MILP selection
    GREEDY = "GREEDY"      # Fast round-repair 
    MILP = "MILP"          # Exact share-level optimization
    AGGRESSIVE = "AGGRESSIVE"  # Maximum deployment with relaxed constraints

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

########################################
# Dividend Optimization Models
########################################

class DividendOptimizationRequest(BaseModel):
    """Request model for dividend optimization endpoint - simplified for income focus"""
    stocks: List[StockItem]
    budget: float = 1000000  # Default ₹10L
    method: DividendOptimizationMethod = DividendOptimizationMethod.AUTO
    max_position_size: float = 0.25  # Maximum 25% per position
    min_positions: int = 3  # Minimum diversification
    min_yield: float = 0.005  # Minimum acceptable yield (0.5%)
    
    # Legacy fields kept for backward compatibility but ignored
    max_risk_variance: Optional[float] = None  # Deprecated - ignored
    individual_caps: Optional[Dict[str, float]] = None  # Deprecated - use max_position_size
    sector_caps: Optional[Dict[str, float]] = None  # Deprecated
    sector_mapping: Optional[Dict[str, str]] = None  # Deprecated
    min_names: Optional[int] = None  # Deprecated - use min_positions
    seed: Optional[int] = 42  # For reproducible results
    
    # Validation
    class Config:
        json_schema_extra = {
            "example": {
                "stocks": [
                    {"ticker": "ITC", "exchange": "NSE"},
                    {"ticker": "COALINDIA", "exchange": "NSE"},
                    {"ticker": "ONGC", "exchange": "NSE"},
                    {"ticker": "NTPC", "exchange": "NSE"}
                ],
                "budget": 1000000,
                "method": "AUTO",
                "max_position_size": 0.25,
                "min_positions": 4
            }
        }

class DividendStockData(BaseModel):
    """Individual stock dividend and price data"""
    symbol: str
    price: float
    forward_dividend: float
    forward_yield: float
    dividend_source: str  # 'info', 'trailing', 'history', 'fallback'
    confidence: Optional[str] = None  # 'very_low', 'low', 'medium', 'high'
    cadence_info: Optional[Dict] = None  # Frequency, CV, regularity data

class DividendAllocationResult(BaseModel):
    """Individual stock allocation result"""
    symbol: str
    shares: int
    price: float
    value: float
    weight: float               # weight on budget (backward compatible)
    weight_on_invested: float   # new: sums to 1 across holdings
    target_weight: float
    forward_yield: float
    annual_income: float

class DividendOptimizationResponse(BaseModel):
    """Response model for dividend optimization endpoint - income focused"""
    # Portfolio metrics
    total_budget: float
    amount_invested: float
    residual_cash: float
    deployment_rate: float  # NEW: percentage of budget deployed
    portfolio_yield: float
    yield_on_invested: float
    annual_income: float
    allocation_method: str
    
    # Individual holdings
    allocations: List[DividendAllocationResult]
    
    # Summary data
    dividend_data: List[DividendStockData]
    optimization_summary: Dict
    
    # Legacy fields for backward compatibility
    post_round_volatility: Optional[float] = None  # Deprecated
    l1_drift: Optional[float] = None  # Deprecated
    granularity_check: Optional[Dict] = None  # Deprecated
    sector_allocations: Optional[Dict[str, float]] = None  # Deprecated 