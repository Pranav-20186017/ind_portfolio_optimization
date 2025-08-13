"""
Dividend Optimization Module
============================

This module provides sophisticated forward yield portfolio optimization functionality
integrated with the existing API infrastructure. It includes:

1. Advanced dividend estimation with cadence detection
2. Risk-controlled portfolio optimization  
3. Intelligent greedy/MILP share allocation
4. Comprehensive error handling and logging

Uses Logfire for consistent logging with the rest of the application.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import data models and error handling
from data import ErrorCode, APIError, StockItem
from divopt import ForwardYieldOptimizer, StockData, PortfolioResult

# Configure logger
logger = logging.getLogger(__name__)

class DividendOptimizationService:
    """
    Service class for dividend optimization with integrated logging and error handling.
    Wraps the core ForwardYieldOptimizer with API-specific functionality.
    """
    
    def __init__(self):
        self.optimizer = ForwardYieldOptimizer()
        logger.info("Dividend optimization service initialized")
    
    async def validate_request(self, stocks: List[StockItem], budget: float, 
                             min_names: Optional[int] = None,
                             sector_caps: Optional[Dict[str, float]] = None,
                             max_risk_variance: Optional[float] = None) -> None:
        """
        Validate dividend optimization request parameters
        
        Args:
            stocks: List of stock items to optimize
            budget: Total budget for investment
            min_names: Minimum number of stocks required
            sector_caps: Sector weight caps
            max_risk_variance: Maximum portfolio variance
            
        Raises:
            APIError: If validation fails
        """
        logger.info(f"Validating dividend optimization request: {len(stocks)} stocks, budget ₹{budget:,.0f}")
        
        if not stocks or len(stocks) < 2:
            logger.error(f"Insufficient stocks provided: {len(stocks)}")
            raise APIError(
                code=ErrorCode.INSUFFICIENT_STOCKS,
                message="Please select at least 2 valid stocks for dividend optimization",
                details={"provided_stocks": len(stocks), "min_required": 2}
            )
        
        if budget <= 0:
            logger.error(f"Invalid budget provided: ₹{budget}")
            raise APIError(
                code=ErrorCode.INVALID_BUDGET,
                message="Budget must be positive",
                details={"budget": budget}
            )
        
        if min_names and min_names > len(stocks):
            logger.error(f"min_names ({min_names}) exceeds available stocks ({len(stocks)})")
            raise APIError(
                code=ErrorCode.MIN_NAMES_INFEASIBLE,
                message=f"Minimum names requirement ({min_names}) exceeds available stocks ({len(stocks)})",
                details={"min_names": min_names, "available_stocks": len(stocks)}
            )
        
        # Validate sector caps if provided
        if sector_caps:
            for sector, cap in sector_caps.items():
                if not isinstance(cap, (int, float)):
                    logger.error(f"Invalid sector cap type for {sector}: {type(cap)}")
                    raise APIError(
                        code=ErrorCode.INVALID_BUDGET,
                        message=f"Sector cap for {sector} must be a number, got {type(cap).__name__}",
                        details={"sector": sector, "cap": cap}
                    )
                
                if cap < 0 or cap > 1:
                    logger.error(f"Invalid sector cap value for {sector}: {cap}")
                    raise APIError(
                        code=ErrorCode.INVALID_BUDGET,
                        message=f"Sector cap for {sector} must be between 0 and 1, got {cap}",
                        details={"sector": sector, "cap": cap, "valid_range": "0.0 to 1.0"}
                    )
        
        # Validate max risk variance if provided
        if max_risk_variance is not None:
            if not isinstance(max_risk_variance, (int, float)):
                logger.error(f"Invalid max_risk_variance type: {type(max_risk_variance)}")
                raise APIError(
                    code=ErrorCode.INVALID_BUDGET,
                    message=f"Max risk variance must be a number, got {type(max_risk_variance).__name__}",
                    details={"max_risk_variance": max_risk_variance}
                )
            
            if max_risk_variance <= 0 or max_risk_variance > 1:
                logger.error(f"Invalid max_risk_variance value: {max_risk_variance}")
                raise APIError(
                    code=ErrorCode.INVALID_BUDGET,
                    message=f"Max risk variance must be between 0 and 1, got {max_risk_variance}",
                    details={"max_risk_variance": max_risk_variance, "valid_range": "0.0 to 1.0"}
                )
        
        logger.info("Request validation completed successfully")
    
    def fetch_and_prepare_data(self, tickers: List[str]) -> Dict[str, StockData]:
        """
        Fetch dividend and price data for given tickers
        
        Args:
            tickers: List of formatted ticker symbols
            
        Returns:
            Dictionary mapping symbols to StockData
            
        Raises:
            APIError: If data fetching fails
        """
        logger.info(f"Fetching dividend data for {len(tickers)} tickers: {tickers}")
        
        try:
            # Fetch data using the core optimizer
            stocks_data = self.optimizer.fetch_dividend_data(tickers)
            
            # NEW: robust empty check (handles MagicMock(return_value={}) cleanly)
            if not isinstance(stocks_data, dict) or len(stocks_data) == 0:
                logger.error("No dividend data could be fetched for any stocks")
                raise APIError(
                    code=ErrorCode.DIVIDEND_FETCH_ERROR,
                    message="Failed to fetch dividend data for any of the provided stocks",
                    details={"requested_tickers": tickers}
                )
            
            # Check data quality
            valid_data_count = len(stocks_data)
            fallback_count = sum(1 for stock in stocks_data.values() 
                               if stock.dividend_source == 'fallback')
            
            logger.info(f"Data fetch completed: {valid_data_count} stocks, {fallback_count} using fallback yields")
            
            # Log individual stock details for visibility in logfire
            for symbol, stock in stocks_data.items():
                logger.info(f"Stock {symbol}: price=₹{stock.price:.2f}, yield={stock.forward_yield:.2%}, "
                           f"source={stock.dividend_source}, confidence={stock.dividend_metadata.get('confidence', 'unknown') if stock.dividend_metadata else 'unknown'}")
            
            if valid_data_count < 2:
                logger.error(f"Insufficient valid data: only {valid_data_count} stocks")
                raise APIError(
                    code=ErrorCode.DIVIDEND_DATA_INSUFFICIENT,
                    message=f"Insufficient dividend data: only {valid_data_count} stocks have usable data",
                    details={"valid_stocks": valid_data_count, "min_required": 2}
                )
            
            return stocks_data
            
        except APIError:
            raise
        except Exception as e:
            logger.exception("Unexpected error during data fetching")
            raise APIError(
                code=ErrorCode.DIVIDEND_FETCH_ERROR,
                message=f"Data fetching failed: {str(e)}",
                status_code=500
            )
    
    def prepare_optimization_data(self, stocks_data: Dict[str, StockData]) -> None:
        """
        Prepare data arrays and covariance matrix for optimization
        
        Args:
            stocks_data: Dictionary of stock data
            
        Raises:
            APIError: If data preparation fails
        """
        logger.info(f"Preparing optimization data for {len(stocks_data)} stocks")
        
        try:
            # Prepare data arrays
            self.optimizer.prepare_data(stocks_data)
            
            # Estimate covariance matrix with error handling
            try:
                self.optimizer.estimate_covariance_matrix(period="2y")
                logger.info(f"Covariance matrix estimated: {self.optimizer.covariance_matrix.shape}")
            except Exception as e:
                logger.error(f"Covariance matrix estimation failed: {e}")
                raise APIError(
                    code=ErrorCode.COVARIANCE_CALCULATION_ERROR,
                    message=f"Failed to estimate risk model: {str(e)}",
                    status_code=500
                )
                
        except APIError:
            raise
        except Exception as e:
            logger.exception("Unexpected error during data preparation")
            raise APIError(
                code=ErrorCode.UNEXPECTED_ERROR,
                message=f"Data preparation failed: {str(e)}",
                status_code=500
            )
    
    def run_continuous_optimization(self, max_risk_variance: float,
                                        individual_caps: Optional[np.ndarray] = None,
                                        sector_caps: Optional[Dict[str, float]] = None,
                                        sector_mapping: Optional[Dict[str, str]] = None) -> np.ndarray:
        """
        Run continuous QP optimization to get target weights
        
        Args:
            max_risk_variance: Maximum portfolio variance (σ²)
            individual_caps: Individual stock weight caps
            sector_caps: Sector weight caps
            sector_mapping: Symbol to sector mapping
            
        Returns:
            Target weights array
            
        Raises:
            APIError: If optimization fails
        """
        logger.info(f"Running continuous optimization with risk constraint σ² ≤ {max_risk_variance:.4f}")
        
        try:
            # NEW: validate covariance matrix
            S = self.optimizer.covariance_matrix
            if S is None:
                logger.error("Covariance matrix is missing")
                raise APIError(
                    code=ErrorCode.OPTIMIZATION_FAILED,
                    message="Continuous optimization failed: risk model is missing",
                    status_code=500
                )

            # Symmetry check
            if not np.allclose(S, S.T, atol=1e-12):
                logger.error("Covariance matrix is not symmetric")
                raise APIError(
                    code=ErrorCode.OPTIMIZATION_FAILED,
                    message="Continuous optimization failed: covariance matrix is not symmetric",
                    status_code=500
                )

            # PSD check
            min_eig = float(np.linalg.eigvalsh(S).min())
            if min_eig < -1e-12:
                logger.error(f"Covariance matrix is not PSD (min eigenvalue = {min_eig:.3e})")
                raise APIError(
                    code=ErrorCode.OPTIMIZATION_FAILED,
                    message="Continuous optimization failed: covariance matrix is not positive semidefinite",
                    details={"min_eigenvalue": min_eig},
                    status_code=500
                )
            
            target_weights = self.optimizer.optimize_continuous(
                max_risk_variance=max_risk_variance,
                individual_caps=individual_caps,
                sector_caps=sector_caps,
                sector_mapping=sector_mapping
            )
            
            # Validate optimization result
            if target_weights is None or not np.isfinite(target_weights).all():
                logger.error("Optimization returned invalid weights")
                raise APIError(
                    code=ErrorCode.OPTIMIZATION_FAILED,
                    message="Continuous optimization failed to find valid solution",
                    status_code=500
                )
            
            portfolio_yield = float(self.optimizer.forward_yields @ target_weights)
            num_positions = int(np.sum(target_weights > 1e-6))
            
            logger.info(f"Continuous optimization completed: {portfolio_yield:.4%} yield, {num_positions} positions")
            
            return target_weights
            
        except APIError:
            raise
        except Exception as e:
            logger.exception("Unexpected error during continuous optimization")
            raise APIError(
                code=ErrorCode.OPTIMIZATION_FAILED,
                message=f"Continuous optimization failed: {str(e)}",
                status_code=500
            )
    
    def check_budget_feasibility(self, budget: float, target_weights: np.ndarray,
                                     min_names: Optional[int] = None) -> None:
        """
        Check if the budget is sufficient for the target allocation
        
        Args:
            budget: Total budget
            target_weights: Target portfolio weights
            min_names: Minimum number of names required
            
        Raises:
            APIError: If budget is insufficient
        """
        logger.info(f"Checking budget feasibility for ₹{budget:,.0f}")
        
        # Check if any single stock can be bought
        min_investment = float(np.min(self.optimizer.prices))
        if budget < min_investment:
            logger.error(f"Budget too small: ₹{budget:,.0f} < ₹{min_investment:,.0f} (cheapest stock)")
            raise APIError(
                code=ErrorCode.BUDGET_TOO_SMALL,
                message=f"Budget (₹{budget:,.0f}) is insufficient to buy even one share of the cheapest stock (₹{min_investment:,.0f})",
                details={"budget": budget, "min_investment": min_investment}
            )
        
        # Check granularity and feasibility
        granularity = self.optimizer.preflight_granularity(budget, target_weights, min_names)
        
        if not granularity["feasible"]:
            logger.error(f"Budget infeasible: {granularity['reason']}")
            
            if "All prices exceed budget" in granularity["reason"]:
                raise APIError(
                    code=ErrorCode.BUDGET_TOO_SMALL,
                    message=granularity["reason"],
                    details={"budget": budget, "min_price": float(np.min(self.optimizer.prices))}
                )
            elif "Budget supports only" in granularity["reason"]:
                raise APIError(
                    code=ErrorCode.MIN_NAMES_INFEASIBLE,
                    message=granularity["reason"],
                    details={"budget": budget, "min_names": min_names}
                )
            else:
                raise APIError(
                    code=ErrorCode.ALLOCATION_INFEASIBLE,
                    message=granularity["reason"],
                    details={"budget": budget}
                )
        
        logger.info(f"Budget feasibility check passed: N_target={granularity['N_target']:.1f}, g_max={granularity['g_max']:.2%}")
    
    def allocate_shares(self, target_weights: np.ndarray, budget: float,
                            method: str = "AUTO",
                            individual_caps: Optional[np.ndarray] = None,
                            sector_caps: Optional[Dict[str, float]] = None,
                            sector_mapping: Optional[Dict[str, str]] = None,
                            min_names: Optional[int] = None,
                            seed: Optional[int] = 42) -> PortfolioResult:
        """
        Allocate integer shares using the specified method
        
        Args:
            target_weights: Target portfolio weights from continuous optimization
            budget: Total budget
            method: Allocation method ("AUTO", "GREEDY", or "MILP")
            individual_caps: Individual stock weight caps
            sector_caps: Sector weight caps
            sector_mapping: Symbol to sector mapping
            min_names: Minimum number of names
            seed: Random seed for reproducibility
            
        Returns:
            PortfolioResult with share allocation
            
        Raises:
            APIError: If allocation fails
        """
        logger.info(f"Allocating shares using {method} method with budget ₹{budget:,.0f}")
        
        try:
            if method == "AUTO":
                result = self.optimizer.allocate_shares_auto(
                    target_weights=target_weights,
                    budget=budget,
                    individual_caps=individual_caps,
                    sector_caps=sector_caps,
                    sector_mapping=sector_mapping,
                    min_names=min_names,
                    thresholds=None,  # Use dynamic thresholds
                    seed=seed
                )
                
                # Safety net at service level
                deploy = 1.0 - (result.residual_cash / budget)
                if deploy < 0.95 and result.allocation_method.startswith("Greedy"):
                    logger.info(f"Under-deployed ({deploy:.2%}). Rerunning with MILP.")
                    result = self.optimizer.solve_income_milp(
                        target_weights=target_weights,
                        budget=budget,
                        individual_caps=individual_caps,
                        sector_caps=sector_caps,
                        sector_mapping=sector_mapping,
                        min_names=min_names,
                    )
            elif method == "GREEDY":
                result = self.optimizer.allocate_shares_greedy(
                    target_weights=target_weights,
                    budget=budget,
                    individual_caps=individual_caps,
                    sector_caps=sector_caps,
                    sector_mapping=sector_mapping,
                    min_names=min_names,
                    seed=seed
                )
            elif method == "MILP":
                result = self.optimizer.solve_income_milp(
                    target_weights=target_weights,
                    budget=budget,
                    individual_caps=individual_caps,
                    sector_caps=sector_caps,
                    sector_mapping=sector_mapping,
                    min_names=min_names
                )
            else:
                logger.error(f"Invalid allocation method: {method}")
                raise APIError(
                    code=ErrorCode.INVALID_OPTIMIZATION_METHOD,
                    message=f"Invalid allocation method: {method}",
                    details={"valid_methods": ["AUTO", "GREEDY", "MILP"]}
                )
            
            # Log allocation results
            total_shares = int(np.sum(result.shares))
            num_positions = int(np.sum(result.shares > 0))
            deployment_rate = (budget - result.residual_cash) / budget
            
            logger.info(f"Share allocation completed: {result.allocation_method}, "
                       f"{total_shares} shares, {num_positions} positions, "
                       f"{deployment_rate:.1%} deployed, {result.portfolio_yield:.4%} yield")
            
            return result
            
        except APIError:
            raise
        except Exception as e:
            logger.exception("Unexpected error during share allocation")
            raise APIError(
                code=ErrorCode.ALLOCATION_INFEASIBLE,
                message=f"Share allocation failed: {str(e)}",
                status_code=500
            )
    
    def convert_individual_caps(self, caps_dict: Optional[Dict[str, float]], 
                              symbols: List[str]) -> Optional[np.ndarray]:
        """
        Convert individual caps dictionary to numpy array aligned with symbols
        
        Args:
            caps_dict: Dictionary mapping symbol to cap
            symbols: List of symbols in order
            
        Returns:
            Numpy array of caps or None
            
        Raises:
            APIError: If caps are invalid
        """
        if caps_dict is None:
            return None
        elif not caps_dict:
            # Return default caps array for empty dict
            return np.array([0.15] * len(symbols))
        
        # Validate caps values
        for symbol, cap in caps_dict.items():
            if not isinstance(cap, (int, float)):
                logger.error(f"Invalid cap type for {symbol}: {type(cap)}")
                raise APIError(
                    code=ErrorCode.INVALID_BUDGET,  # Reuse existing error code
                    message=f"Individual cap for {symbol} must be a number, got {type(cap).__name__}",
                    details={"symbol": symbol, "cap": cap}
                )
            
            if cap < 0 or cap > 1:
                logger.error(f"Invalid cap value for {symbol}: {cap}")
                raise APIError(
                    code=ErrorCode.INVALID_BUDGET,  # Reuse existing error code
                    message=f"Individual cap for {symbol} must be between 0 and 1, got {cap}",
                    details={"symbol": symbol, "cap": cap, "valid_range": "0.0 to 1.0"}
                )
        
        caps_array = np.array([caps_dict.get(symbol, 0.15) for symbol in symbols])
        logger.debug(f"Converted individual caps for {len(symbols)} symbols")
        return caps_array
    
    def calculate_post_round_volatility(self, shares: np.ndarray) -> float:
        """
        Calculate post-round portfolio volatility
        
        Args:
            shares: Array of share allocations
            
        Returns:
            Portfolio volatility
        """
        invested = float(np.sum(shares * self.optimizer.prices))
        if invested <= 0:
            return 0.0
        
        w_exec = (shares * self.optimizer.prices) / invested
        return float(np.sqrt(w_exec @ self.optimizer.covariance_matrix @ w_exec))
