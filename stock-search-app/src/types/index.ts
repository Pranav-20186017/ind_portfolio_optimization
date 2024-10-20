// src/types/index.ts

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

// src/types/index.ts

export interface PortfolioPerformance {
    expected_return: number;
    volatility: number;
    sharpe: number;
}

export interface OptimizationResult {
    weights: { [ticker: string]: number };
    performance: PortfolioPerformance;
}

export interface PortfolioOptimizationResponse {
    MVO?: OptimizationResult | null;
    MinVol?: OptimizationResult | null;
    start_date: string; // ISO date string
    end_date: string;   // ISO date string
    cumulative_returns: {
        MVO: (number | null)[];
        MinVol: (number | null)[];
    };
    dates: string[]; // Dates as ISO strings
    nifty_returns: number[];
}
