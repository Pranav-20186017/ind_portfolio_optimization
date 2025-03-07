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
}

// Extended optimization result, now including drawdown plot images
export interface OptimizationResult {
    weights: { [ticker: string]: number };
    performance: PortfolioPerformance;

    // Base64-encoded images of distribution plot & drawdown plot
    returns_dist?: string;
    max_drawdown_plot?: string;
}

// Use a dictionary for "results", keyed by method name (e.g. "MVO", "MinVol", etc.)
export interface PortfolioOptimizationResponse {
    // The server returns a "results" object with keys like "MVO", "MinVol", "HRP", etc.
    // Each key is an OptimizationResult or null if that method wasn't computed.
    results: { [methodKey: string]: OptimizationResult | null };

    start_date: string;  // e.g. "2020-01-01"
    end_date: string;    // e.g. "2025-01-01"

    // Similarly, "cumulative_returns" is also a dictionary keyed by method name
    cumulative_returns: { [methodKey: string]: (number | null)[] };

    dates: string[];       // e.g. ["2020-01-02", "2020-01-03", ...]
    nifty_returns: number[]; 
}
