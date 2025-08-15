"""
New Dividend Optimization Service
=================================

Simplified, practitioner-focused dividend optimization that:
1. Actually deploys capital (>99%)
2. Focuses on income generation
3. Removes unnecessary risk/volatility constraints
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from data import ErrorCode, APIError, StockItem
from divopt_new import PractitionerDividendOptimizer, StockData, DividendPortfolioResult

logger = logging.getLogger(__name__)

class NewDividendOptimizationService:
    """
    Service wrapper for the new practitioner-focused dividend optimizer.
    """
    
    def __init__(self):
        self.optimizer = PractitionerDividendOptimizer()
        logger.info("New dividend optimization service initialized")
    
    async def validate_request(self, 
                             stocks: List[StockItem], 
                             budget: float,
                             min_positions: Optional[int] = None,
                             max_position_size: Optional[float] = None) -> None:
        """
        Validate request parameters - simplified version.
        """
        logger.info(f"Validating request: {len(stocks)} stocks, budget ₹{budget:,.0f}")
        
        if not stocks or len(stocks) < 2:
            raise APIError(
                code=ErrorCode.INSUFFICIENT_STOCKS,
                message="Please select at least 2 stocks for diversification",
                details={"provided": len(stocks), "required": 2}
            )
        
        if budget <= 10000:  # Minimum ₹10,000
            raise APIError(
                code=ErrorCode.INVALID_BUDGET,
                message="Minimum budget is ₹10,000",
                details={"budget": budget, "minimum": 10000}
            )
        
        if min_positions and min_positions > len(stocks):
            raise APIError(
                code=ErrorCode.MIN_NAMES_INFEASIBLE,
                message=f"Cannot have {min_positions} positions with only {len(stocks)} stocks",
                details={"min_positions": min_positions, "available_stocks": len(stocks)}
            )
        
        if max_position_size:
            if max_position_size <= 0 or max_position_size > 1:
                raise APIError(
                    code=ErrorCode.INVALID_BUDGET,
                    message="Max position size must be between 0 and 1",
                    details={"max_position_size": max_position_size}
                )
            
            # Check if max_position_size allows for min_positions
            if min_positions and max_position_size < (1.0 / min_positions):
                raise APIError(
                    code=ErrorCode.INVALID_BUDGET,
                    message=f"Max position size {max_position_size:.1%} too small for {min_positions} positions",
                    details={"max_position_size": max_position_size, "min_positions": min_positions}
                )
    
    async def fetch_and_prepare_data(self, tickers: List[str]) -> Dict[str, StockData]:
        """
        Fetch dividend data and prepare for optimization.
        """
        logger.info(f"Fetching data for {len(tickers)} tickers")
        
        try:
            # Fetch data
            stocks_data = self.optimizer.fetch_dividend_data(tickers)
            
            if not stocks_data or len(stocks_data) < 2:
                raise APIError(
                    code=ErrorCode.DIVIDEND_FETCH_ERROR,
                    message="Could not fetch sufficient dividend data",
                    details={"fetched": len(stocks_data), "required": 2}
                )
            
            # Prepare optimizer
            self.optimizer.prepare_data(stocks_data)
            
            # Log summary
            yields = [s.forward_yield for s in stocks_data.values()]
            logger.info(f"Data ready: {len(stocks_data)} stocks, "
                       f"yields {min(yields):.2%} - {max(yields):.2%}")
            
            return stocks_data
            
        except APIError:
            raise
        except Exception as e:
            logger.exception("Error fetching data")
            raise APIError(
                code=ErrorCode.DIVIDEND_FETCH_ERROR,
                message=f"Failed to fetch dividend data: {str(e)}",
                status_code=500
            )
    
    async def optimize_portfolio(self,
                               budget: float,
                               method: str = "AUTO",
                               max_position_size: float = 0.25,
                               min_positions: int = 3,
                               min_yield: float = 0.005) -> DividendPortfolioResult:
        """
        Run the optimization.
        """
        logger.info(f"Optimizing: budget=₹{budget:,.0f}, method={method}, "
                   f"max_position={max_position_size:.1%}, min_positions={min_positions}")
        
        try:
            result = self.optimizer.optimize(
                budget=budget,
                method=method,
                max_position_size=max_position_size,
                min_positions=min_positions,
                min_yield=min_yield
            )
            
            # Validate deployment
            if result.deployment_rate < 0.95:
                logger.warning(f"Low deployment: {result.deployment_rate:.1%}")
            
            logger.info(f"Optimization complete: {result.num_positions} positions, "
                       f"{result.deployment_rate:.1%} deployed, "
                       f"{result.portfolio_yield:.2%} yield")
            
            return result
            
        except Exception as e:
            logger.exception("Optimization failed")
            raise APIError(
                code=ErrorCode.OPTIMIZATION_FAILED,
                message=f"Optimization failed: {str(e)}",
                status_code=500
            )
    
    def format_response(self, result: DividendPortfolioResult, budget: float) -> Dict:
        """
        Format the result for API response.
        """
        # Format allocations
        allocations = []
        for alloc in result.stock_allocations:
            allocations.append({
                'symbol': alloc['symbol'],
                'shares': alloc['shares'],
                'price': alloc['price'],
                'value': alloc['value'],
                'weight': alloc['weight'],
                'weight_on_invested': alloc['value'] / result.total_investment if result.total_investment > 0 else 0,
                'forward_yield': alloc['yield'],
                'annual_income': alloc['annual_income']
            })
        
        # Format dividend data
        dividend_data = []
        for symbol in self.optimizer.symbols:
            stock = self.optimizer.stocks_data[symbol]
            dividend_data.append({
                'symbol': symbol,
                'price': stock.price,
                'forward_dividend': stock.forward_dividend,
                'forward_yield': stock.forward_yield,
                'dividend_source': stock.dividend_source
            })
        
        return {
            'total_budget': budget,
            'amount_invested': result.total_investment,
            'residual_cash': result.residual_cash,
            'portfolio_yield': result.portfolio_yield,
            'yield_on_invested': result.annual_income / result.total_investment if result.total_investment > 0 else 0,
            'annual_income': result.annual_income,
            'deployment_rate': result.deployment_rate,
            'allocation_method': result.allocation_method,
            'allocations': allocations,
            'dividend_data': dividend_data,
            'optimization_summary': {
                'num_positions': result.num_positions,
                'deployment_rate': result.deployment_rate,
                'method_used': result.allocation_method
            }
        }
