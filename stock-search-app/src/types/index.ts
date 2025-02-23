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
}

// Extended optimization result, now including drawdown plot
export interface OptimizationResult {
    weights: { [ticker: string]: number };
    performance: PortfolioPerformance;

    // Base64-encoded images of distribution plot & drawdown plot
    returns_dist: string;
    max_drawdown_plot: string;
}

// Full portfolio optimization response with additional portfolios & new metrics
export interface PortfolioOptimizationResponse {
    MVO?: OptimizationResult | null;
    MinVol?: OptimizationResult | null;
    MaxQuadraticUtility?: OptimizationResult | null;
    EquiWeighted?: OptimizationResult | null;
    CriticalLineAlgorithm?: OptimizationResult | null;
    

    start_date: string;  // ISO date
    end_date: string;    // ISO date

    cumulative_returns: {
        MVO: (number | null)[];
        MinVol: (number | null)[];
        MaxQuadraticUtility: (number | null)[];
        EquiWeighted: (number | null)[];
        CriticalLineAlgorithm: (number | null)[];
    };

    dates: string[];        // ISO date strings corresponding to each data point
    nifty_returns: number[];
}
