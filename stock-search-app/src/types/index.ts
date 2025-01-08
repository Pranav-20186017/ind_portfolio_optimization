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

// Types for portfolio performance and optimization result
export interface PortfolioPerformance {
    expected_return: number;
    volatility: number;
    sharpe: number;
}

export interface OptimizationResult {
    weights: { [ticker: string]: number };  // Dictionary of stock tickers and their weights
    performance: PortfolioPerformance;      // Performance metrics (expected return, volatility, Sharpe ratio)
    returns_dist: string;                   // Base64-encoded string of the returns distribution plot
}

// Full portfolio optimization response with all portfolio types and cumulative returns
export interface PortfolioOptimizationResponse {
    MVO?: OptimizationResult | null;  // Mean-Variance Optimization (Max Sharpe Ratio)
    MinVol?: OptimizationResult | null;  // Minimum Volatility Portfolio
    MaxQuadraticUtility?: OptimizationResult | null;  // Max Quadratic Utility Portfolio
    start_date: string;  // Start date of the portfolio period (ISO format)
    end_date: string;    // End date of the portfolio period (ISO format)
    cumulative_returns: {
        MVO: (number | null)[];  // Cumulative returns for MVO
        MinVol: (number | null)[];  // Cumulative returns for Min Vol
        MaxQuadraticUtility: (number | null)[];  // Cumulative returns for Max Quadratic Utility Portfolio
    };
    dates: string[];  // Dates corresponding to the cumulative returns (ISO format)
    nifty_returns: number[];  // Nifty index returns for the same period
}
