// types/index.ts

// Types for stock listings and options
export interface StockListing {
    name: string;
    isin: string;
    exchange: string;
}

export interface StockData {
    [ticker: string]: StockListing[];
}

export interface StockOption {
    ticker: string;
    name: string;
    exchange: string;
}

// Error handling type
export interface APIError {
    message: string;
    details?: string[] | Record<string, any> | string | number;
}

// Extended types for portfolio performance
export interface PortfolioPerformance {
    // From PyPortfolioOpt
    expected_return: number;
    volatility: number;
    sharpe: number;

    // Custom metrics
    sortino: number;
    max_drawdown: number;
    romad: number;
    var_95: number;
    cvar_95: number;
    var_90: number;
    cvar_90: number;
    cagr: number;
    portfolio_beta: number;
    portfolio_alpha: number;
    beta_pvalue: number;
    r_squared: number;
    blume_adjusted_beta: number;
    treynor_ratio: number;
    skewness: number;
    kurtosis: number;
    entropy: number;
    
    // Advanced beta and cross-moment metrics
    welch_beta?: number;
    semi_beta?: number;
    coskewness?: number;
    cokurtosis?: number;
    garch_beta?: number;
    
    // Other metrics
    omega_ratio: number;
    calmar_ratio: number;
    ulcer_index: number;
    evar_95: number;
    gini_mean_difference: number;
    dar_95: number;
    cdar_95: number;
    upside_potential_ratio: number;
    modigliani_risk_adjusted_performance: number;
    information_ratio: number;
    sterling_ratio: number;
    v2_ratio: number;
}

// Extended optimization result, now including drawdown plot images
export interface OptimizationResult {
    weights: { [ticker: string]: number };
    performance: PortfolioPerformance;

    // Base64-encoded images of distribution plot & drawdown plot
    returns_dist?: string;
    max_drawdown_plot?: string;
    
    // Yearly rolling betas - Note: JSON serializes the Python int keys as strings
    rolling_betas?: { [year: string]: number };
}

// Use a dictionary for "results", keyed by method name (e.g. "MVO", "MinVol", "HRP", etc.)
export interface PortfolioOptimizationResponse {
    // The server returns a "results" object with keys like "MVO", "MinVol", "HRP", etc.
    // Each key is an OptimizationResult or null if that method wasn't computed.
    results: { [methodKey: string]: OptimizationResult | null };

    start_date: string;  // e.g. "2020-01-01"
    end_date: string;    // e.g. "2025-01-01"

    // "cumulative_returns" is also a dictionary keyed by method name
    cumulative_returns: { [methodKey: string]: (number | null)[] };

    dates: string[];       // e.g. ["2020-01-02", "2020-01-03", ...]
    
    // Benchmark returns data
    benchmark_returns: BenchmarkReturn[];

    // New field: yearly returns for each stock in the portfolio.
    // Each stock maps to a dictionary where the keys are years (as string) and the values are the returns.
    stock_yearly_returns?: { [ticker: string]: { [year: string]: number } };

    // Add the covariance heatmap field (base64 encoded image)
    covariance_heatmap?: string;

    risk_free_rate?: number;
}

export enum ExchangeEnum {
  NSE = "NSE",
  BSE = "BSE"
}

export enum OptimizationMethod {
  MVO = "MVO",
  MinVol = "MinVol",
  MaxQuadraticUtility = "MaxQuadraticUtility",
  EquiWeighted = "EquiWeighted",
  CriticalLineAlgorithm = "CriticalLineAlgorithm",
  HRP = "HRP",
  MinCVaR = "MinCVaR",
  MinCDaR = "MinCDaR"
}

export enum CLAOptimizationMethod {
  MVO = "MVO",
  MinVol = "MinVol",
  Both = "Both"
}

export enum BenchmarkName {
  nifty = "nifty",
  sensex = "sensex",
  bank_nifty = "bank_nifty"
}

export interface StockItem {
  ticker: string;
  exchange: ExchangeEnum;
}

export interface TickerRequest {
  stocks: StockItem[];
  methods: OptimizationMethod[];
  cla_method?: CLAOptimizationMethod;
  benchmark: BenchmarkName;
}

export interface BenchmarkReturn {
  name: BenchmarkName;
  returns: number[];
}
