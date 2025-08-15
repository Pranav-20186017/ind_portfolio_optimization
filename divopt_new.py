"""
Practitioner-Focused Dividend Portfolio Optimization
====================================================

A complete rewrite focusing on what matters for dividend investors:
1. Maximum dividend income generation
2. Full capital deployment (>99%)
3. Practical diversification constraints
4. Simple, robust allocation algorithm

The volatility/risk aspect is secondary - dividend investing is about cash flow.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
import logging
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class StockData:
    """Container for stock data"""
    symbol: str
    price: float
    forward_dividend: float
    forward_yield: float
    dividend_source: str
    dividend_metadata: Dict = None

@dataclass
class DividendPortfolioResult:
    """Container for optimization results"""
    shares: np.ndarray
    weights: np.ndarray
    total_investment: float
    residual_cash: float
    annual_income: float
    portfolio_yield: float
    deployment_rate: float
    num_positions: int
    allocation_method: str
    stock_allocations: List[Dict]

class PractitionerDividendOptimizer:
    """
    Dividend optimizer that actually deploys capital like a real investor would.
    
    Core philosophy:
    - Dividend investing is about income, not volatility
    - Deploy as much capital as possible (target >99%)
    - Use simple, robust methods that work in practice
    """
    
    def __init__(self):
        self.symbols = []
        self.prices = np.array([])
        self.yields = np.array([])
        self.annual_dividends = np.array([])
        self.stocks_data = {}
        
    def fetch_dividend_data(self, symbols: List[str], period: str = "2y") -> Dict[str, StockData]:
        """
        Fetch dividend and price data for stocks.
        Simplified version focusing on getting reliable yield data.
        """
        logger.info(f"Fetching dividend data for {len(symbols)} symbols")
        
        stocks_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get current price
                try:
                    info = ticker.info
                    price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                except:
                    hist = ticker.history(period="5d")
                    price = hist['Close'].iloc[-1] if not hist.empty else 0
                
                if price <= 0:
                    logger.warning(f"Invalid price for {symbol}")
                    continue
                
                # Get dividend yield - multiple sources for robustness
                dividend_yield = 0
                dividend_source = 'none'
                
                # Try to get from info
                info_yield = info.get('dividendYield', 0) or info.get('trailingAnnualDividendYield', 0)
                if info_yield > 0 and info_yield < 0.30:  # Cap at 30% to filter errors
                    dividend_yield = info_yield
                    dividend_source = 'info'
                
                # If no yield, try dividend history
                if dividend_yield == 0:
                    try:
                        dividends = ticker.dividends
                        if not dividends.empty:
                            # Last 12 months of dividends
                            last_year = dividends.last('365D').sum()
                            if last_year > 0:
                                dividend_yield = last_year / price
                                dividend_source = 'history'
                    except:
                        pass
                
                # Conservative fallback for unknowns
                if dividend_yield == 0:
                    dividend_yield = 0.02  # 2% default
                    dividend_source = 'fallback'
                
                annual_dividend = dividend_yield * price
                
                stock_data = StockData(
                    symbol=symbol,
                    price=price,
                    forward_dividend=annual_dividend,
                    forward_yield=dividend_yield,
                    dividend_source=dividend_source,
                    dividend_metadata={}
                )
                
                stocks_data[symbol] = stock_data
                logger.info(f"{symbol}: Price={price:.2f}, Yield={dividend_yield:.2%}, Source={dividend_source}")
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                continue
        
        return stocks_data
    
    def prepare_data(self, stocks_data: Dict[str, StockData]) -> None:
        """Prepare data arrays for optimization"""
        self.stocks_data = stocks_data
        self.symbols = list(stocks_data.keys())
        n = len(self.symbols)
        
        self.prices = np.array([stocks_data[s].price for s in self.symbols])
        self.yields = np.array([stocks_data[s].forward_yield for s in self.symbols])
        self.annual_dividends = np.array([stocks_data[s].forward_dividend for s in self.symbols])
        
        logger.info(f"Prepared data for {n} stocks")
        logger.info(f"Yield range: {self.yields.min():.2%} - {self.yields.max():.2%}")
    
    def optimize_continuous_weights(self, 
                                   max_position_size: float = 0.25,
                                   min_position_size: float = 0.01,
                                   min_yield_threshold: float = 0.005) -> np.ndarray:
        """
        Find optimal continuous weights that maximize dividend income.
        
        Simple linear program:
        - Maximize: sum(w_i * yield_i)
        - Subject to: sum(w_i) = 1, 0 <= w_i <= max_position_size
        """
        n = len(self.symbols)
        
        # Filter out very low yield stocks
        valid_mask = self.yields >= min_yield_threshold
        if not np.any(valid_mask):
            # If no stocks meet threshold, use all
            valid_mask = np.ones(n, dtype=bool)
        
        # Create optimization problem
        w = cp.Variable(n, nonneg=True)
        
        # Objective: maximize dividend yield
        objective = cp.Maximize(self.yields @ w)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            w <= max_position_size,  # Position limits
        ]
        
        # Add minimum position constraint for non-zero weights
        # This helps with diversification
        for i in range(n):
            if valid_mask[i]:
                # Either 0 or at least min_position_size
                pass  # This requires binary variables, skip for continuous
            else:
                constraints.append(w[i] == 0)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            logger.warning(f"Optimization failed: {problem.status}")
            # Fallback to yield-weighted
            weights = self.yields / np.sum(self.yields)
            weights = np.clip(weights, 0, max_position_size)
            weights = weights / np.sum(weights)
            return weights
        
        weights = w.value
        
        # Clean up small weights
        weights[weights < min_position_size] = 0
        weights = weights / np.sum(weights)
        
        logger.info(f"Optimal portfolio yield: {self.yields @ weights:.2%}")
        logger.info(f"Number of positions: {np.sum(weights > 0)}")
        
        return weights
    
    def allocate_shares_milp(self,
                            budget: float,
                            target_weights: np.ndarray,
                            max_position_size: float = 0.30,
                            min_positions: int = 3,
                            min_deployment: float = 0.99) -> DividendPortfolioResult:
        """
        Exact share allocation using MILP to maximize income and deployment.
        
        This is the KEY innovation - we MUST deploy the capital!
        """
        n = len(self.symbols)
        
        # Maximum shares we could buy of each stock
        max_shares = np.floor(budget / self.prices).astype(int)
        
        # Decision variable: number of shares
        x = cp.Variable(n, integer=True)
        
        # Binary variable for position indicators
        z = cp.Variable(n, boolean=True)
        
        # Total investment
        total_investment = cp.sum(cp.multiply(self.prices, x))
        
        # Objective: Maximize dividend income + small bonus for deployment
        # The deployment bonus ensures we use all the money
        income = cp.sum(cp.multiply(self.annual_dividends, x))
        deployment_bonus = 0.0001 * total_investment  # Small incentive to deploy capital
        
        objective = cp.Maximize(income + deployment_bonus)
        
        # Constraints
        constraints = [
            x >= 0,  # Non-negative shares
            x <= cp.multiply(max_shares, z),  # Link shares to binary indicators
            total_investment <= budget,  # Budget constraint
            total_investment >= min_deployment * budget,  # MUST deploy at least 99%!
            cp.sum(z) >= min_positions,  # Minimum diversification
        ]
        
        # Position size limits (as percentage of budget)
        for i in range(n):
            constraints.append(self.prices[i] * x[i] <= max_position_size * budget)
        
        # Try to stay close to target weights (soft constraint via penalty)
        # But income maximization is primary
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        # Try different solvers
        for solver in [cp.GUROBI, cp.CBC, cp.GLPK_MI]:
            try:
                problem.solve(solver=solver, verbose=False)
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    break
            except:
                continue
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            logger.warning("MILP failed, using greedy allocation")
            return self.allocate_shares_greedy(budget, target_weights, max_position_size)
        
        # Extract solution
        shares = np.round(x.value).astype(int)
        
        return self._create_result(shares, budget, "MILP (Income Maximization)")
    
    def allocate_shares_greedy(self,
                              budget: float,
                              target_weights: np.ndarray,
                              max_position_size: float = 0.30) -> DividendPortfolioResult:
        """
        Fast greedy allocation that aggressively deploys capital.
        
        Algorithm:
        1. Sort by dividend yield
        2. Buy as many shares as possible of highest yield stocks
        3. Continue until budget exhausted
        """
        n = len(self.symbols)
        shares = np.zeros(n, dtype=int)
        remaining = budget
        
        # Sort by yield (descending)
        yield_order = np.argsort(-self.yields)
        
        # First pass: allocate based on target weights and yields
        for idx in yield_order:
            if remaining < self.prices[idx]:
                continue
            
            # Maximum we can invest in this stock
            max_invest = min(
                remaining,
                max_position_size * budget,
                target_weights[idx] * budget * 2  # Allow up to 2x target
            )
            
            # How many shares can we buy?
            shares_to_buy = int(max_invest / self.prices[idx])
            
            if shares_to_buy > 0:
                shares[idx] = shares_to_buy
                remaining -= shares_to_buy * self.prices[idx]
        
        # Second pass: use remaining cash on highest yield stocks
        for idx in yield_order:
            while remaining >= self.prices[idx]:
                # Check position limit
                current_value = shares[idx] * self.prices[idx]
                if current_value >= max_position_size * budget:
                    break
                
                shares[idx] += 1
                remaining -= self.prices[idx]
        
        # Final pass: buy ANY stock we can afford to minimize residual
        if remaining > np.min(self.prices):
            for idx in np.argsort(self.prices):  # Cheapest first
                while remaining >= self.prices[idx]:
                    shares[idx] += 1
                    remaining -= self.prices[idx]
        
        return self._create_result(shares, budget, "Greedy (Yield-First)")
    
    def optimize(self,
                budget: float,
                method: str = "AUTO",
                max_position_size: float = 0.25,
                min_positions: int = 3,
                min_yield: float = 0.005) -> DividendPortfolioResult:
        """
        Main optimization entry point.
        
        Args:
            budget: Total amount to invest
            method: "AUTO", "MILP", or "GREEDY"
            max_position_size: Maximum weight per position
            min_positions: Minimum number of positions
            min_yield: Minimum acceptable yield
        """
        # Get continuous weights first
        target_weights = self.optimize_continuous_weights(
            max_position_size=max_position_size,
            min_position_size=0.01,
            min_yield_threshold=min_yield
        )
        
        # Allocate shares
        if method == "AUTO":
            # Use MILP for smaller portfolios, greedy for larger
            if budget < 1_000_000 or len(self.symbols) < 20:
                method = "MILP"
            else:
                method = "GREEDY"
        
        if method == "MILP":
            try:
                result = self.allocate_shares_milp(
                    budget=budget,
                    target_weights=target_weights,
                    max_position_size=max_position_size,
                    min_positions=min_positions,
                    min_deployment=0.99  # Target 99% deployment!
                )
            except Exception as e:
                logger.error(f"MILP failed: {e}, falling back to greedy")
                result = self.allocate_shares_greedy(
                    budget=budget,
                    target_weights=target_weights,
                    max_position_size=max_position_size
                )
        else:
            result = self.allocate_shares_greedy(
                budget=budget,
                target_weights=target_weights,
                max_position_size=max_position_size
            )
        
        # Ensure we deployed enough capital
        if result.deployment_rate < 0.98:
            logger.warning(f"Low deployment rate: {result.deployment_rate:.1%}")
        
        return result
    
    def _create_result(self, shares: np.ndarray, budget: float, method: str) -> DividendPortfolioResult:
        """Create result object from share allocation"""
        total_investment = np.sum(shares * self.prices)
        residual_cash = budget - total_investment
        weights = (shares * self.prices) / budget
        annual_income = np.sum(shares * self.annual_dividends)
        portfolio_yield = annual_income / budget
        deployment_rate = total_investment / budget
        num_positions = np.sum(shares > 0)
        
        # Create stock allocations list
        stock_allocations = []
        for i, symbol in enumerate(self.symbols):
            if shares[i] > 0:
                stock_allocations.append({
                    'symbol': symbol,
                    'shares': int(shares[i]),
                    'price': float(self.prices[i]),
                    'value': float(shares[i] * self.prices[i]),
                    'weight': float(weights[i]),
                    'yield': float(self.yields[i]),
                    'annual_income': float(shares[i] * self.annual_dividends[i])
                })
        
        # Sort by value
        stock_allocations.sort(key=lambda x: x['value'], reverse=True)
        
        result = DividendPortfolioResult(
            shares=shares,
            weights=weights,
            total_investment=total_investment,
            residual_cash=residual_cash,
            annual_income=annual_income,
            portfolio_yield=portfolio_yield,
            deployment_rate=deployment_rate,
            num_positions=num_positions,
            allocation_method=method,
            stock_allocations=stock_allocations
        )
        
        logger.info(f"Result: {num_positions} positions, {deployment_rate:.1%} deployed, "
                   f"{portfolio_yield:.2%} yield, ₹{annual_income:,.0f} annual income")
        
        return result
