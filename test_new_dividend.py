"""
Test Suite for New Practitioner-Focused Dividend Optimizer
===========================================================

Tests the new dividend optimization approach that prioritizes:
1. Full capital deployment (>99%)
2. Income maximization
3. Simple, practical constraints
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# Import the new modules
from divopt_new import PractitionerDividendOptimizer, StockData, DividendPortfolioResult
from dividend_optimizer_new import NewDividendOptimizationService
from data import ErrorCode, APIError, StockItem, ExchangeEnum

class TestNewDividendOptimizer(unittest.TestCase):
    """Test suite for the new dividend optimizer"""
    
    def setUp(self):
        """Set up test data"""
        self.optimizer = PractitionerDividendOptimizer()
        
        # Create mock stock data
        self.mock_stocks = {
            'ITC.NS': StockData(
                symbol='ITC.NS',
                price=400.0,
                forward_dividend=20.0,
                forward_yield=0.05,  # 5% yield
                dividend_source='history',
                dividend_metadata={}
            ),
            'COALINDIA.NS': StockData(
                symbol='COALINDIA.NS',
                price=250.0,
                forward_dividend=20.0,
                forward_yield=0.08,  # 8% yield
                dividend_source='history',
                dividend_metadata={}
            ),
            'ONGC.NS': StockData(
                symbol='ONGC.NS',
                price=150.0,
                forward_dividend=10.5,
                forward_yield=0.07,  # 7% yield
                dividend_source='history',
                dividend_metadata={}
            ),
            'NTPC.NS': StockData(
                symbol='NTPC.NS',
                price=200.0,
                forward_dividend=12.0,
                forward_yield=0.06,  # 6% yield
                dividend_source='history',
                dividend_metadata={}
            )
        }
    
    def test_data_preparation(self):
        """Test data preparation"""
        self.optimizer.prepare_data(self.mock_stocks)
        
        self.assertEqual(len(self.optimizer.symbols), 4)
        self.assertEqual(self.optimizer.symbols[0], 'ITC.NS')
        self.assertAlmostEqual(self.optimizer.prices[0], 400.0)
        self.assertAlmostEqual(self.optimizer.yields[0], 0.05)
        self.assertAlmostEqual(self.optimizer.annual_dividends[0], 20.0)
    
    def test_continuous_optimization(self):
        """Test continuous weight optimization"""
        self.optimizer.prepare_data(self.mock_stocks)
        
        weights = self.optimizer.optimize_continuous_weights(
            max_position_size=0.30,
            min_position_size=0.01,
            min_yield_threshold=0.005
        )
        
        # Check constraints
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(weights <= 0.30))
        
        # Higher yield stocks should get higher weights
        # COALINDIA has highest yield (8%), should get high weight
        coalindia_idx = self.optimizer.symbols.index('COALINDIA.NS')
        self.assertGreater(weights[coalindia_idx], 0.2)
    
    def test_greedy_allocation(self):
        """Test greedy share allocation"""
        self.optimizer.prepare_data(self.mock_stocks)
        
        budget = 100000  # ₹1 lakh
        target_weights = np.array([0.20, 0.35, 0.25, 0.20])  # Favor COALINDIA
        
        result = self.optimizer.allocate_shares_greedy(
            budget=budget,
            target_weights=target_weights,
            max_position_size=0.30
        )
        
        # Check deployment rate - should be very high
        self.assertGreater(result.deployment_rate, 0.95)
        
        # Check that we bought shares
        self.assertGreater(result.num_positions, 0)
        self.assertGreater(result.total_investment, budget * 0.95)
        
        # Check portfolio yield is reasonable
        self.assertGreater(result.portfolio_yield, 0.04)  # At least 4%
        self.assertLess(result.portfolio_yield, 0.10)  # Less than 10%
    
    def test_milp_allocation(self):
        """Test MILP share allocation"""
        self.optimizer.prepare_data(self.mock_stocks)
        
        budget = 100000
        target_weights = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Mock the MILP solver to avoid dependency issues
        with patch('cvxpy.Problem.solve') as mock_solve:
            mock_solve.return_value = None
            
            # Set up mock solution
            with patch('cvxpy.Variable.value', new_callable=lambda: Mock()) as mock_value:
                # Mock shares: buy 50 ITC, 100 COALINDIA, 166 ONGC, 125 NTPC
                mock_shares = np.array([50, 100, 166, 125])
                mock_value.return_value = mock_shares
                
                # Mock problem status
                with patch('cvxpy.Problem.status', 'optimal'):
                    result = self.optimizer.allocate_shares_milp(
                        budget=budget,
                        target_weights=target_weights,
                        max_position_size=0.30,
                        min_positions=3,
                        min_deployment=0.99
                    )
                    
                    # Should fall back to greedy if MILP fails
                    self.assertIsNotNone(result)
                    self.assertGreater(result.deployment_rate, 0.90)
    
    def test_full_optimization_flow(self):
        """Test the complete optimization flow"""
        self.optimizer.prepare_data(self.mock_stocks)
        
        result = self.optimizer.optimize(
            budget=500000,  # ₹5 lakh
            method="AUTO",
            max_position_size=0.25,
            min_positions=3,
            min_yield=0.005
        )
        
        # Check key metrics
        self.assertGreater(result.deployment_rate, 0.95)
        self.assertGreaterEqual(result.num_positions, 3)
        self.assertGreater(result.portfolio_yield, 0.04)
        self.assertGreater(result.annual_income, 20000)  # At least ₹20k annual income
        
        # Check allocations
        self.assertEqual(len(result.stock_allocations), result.num_positions)
        for alloc in result.stock_allocations:
            self.assertGreater(alloc['shares'], 0)
            self.assertGreater(alloc['value'], 0)
            self.assertLessEqual(alloc['weight'], 0.25)
    
    def test_service_layer(self):
        """Test the service layer wrapper"""
        service = NewDividendOptimizationService()
        
        # Mock the optimizer
        service.optimizer = Mock()
        service.optimizer.symbols = ['ITC.NS', 'COALINDIA.NS']
        service.optimizer.stocks_data = self.mock_stocks
        
        # Test validation - valid request
        import asyncio
        async def test_valid():
            stocks = [
                StockItem(ticker='ITC', exchange=ExchangeEnum.NSE),
                StockItem(ticker='COALINDIA', exchange=ExchangeEnum.NSE)
            ]
            await service.validate_request(stocks, 100000, 2, 0.30)
        
        asyncio.run(test_valid())
        
        # Test validation - invalid budget
        async def test_invalid_budget():
            stocks = [
                StockItem(ticker='ITC', exchange=ExchangeEnum.NSE),
                StockItem(ticker='COALINDIA', exchange=ExchangeEnum.NSE)
            ]
            with self.assertRaises(APIError) as context:
                await service.validate_request(stocks, 5000, 2, 0.30)  # Too small
            self.assertEqual(context.exception.code, ErrorCode.INVALID_BUDGET)
        
        asyncio.run(test_invalid_budget())
    
    def test_deployment_rates(self):
        """Test that we achieve high deployment rates with different budgets"""
        self.optimizer.prepare_data(self.mock_stocks)
        
        budgets = [50000, 100000, 500000, 1000000]
        
        for budget in budgets:
            result = self.optimizer.optimize(
                budget=budget,
                method="GREEDY",
                max_position_size=0.30,
                min_positions=2,
                min_yield=0.005
            )
            
            # Should achieve >95% deployment for all reasonable budgets
            self.assertGreater(
                result.deployment_rate, 0.95,
                f"Low deployment rate {result.deployment_rate:.1%} for budget ₹{budget:,}"
            )
            
            print(f"Budget ₹{budget:,}: Deployment {result.deployment_rate:.1%}, "
                  f"Yield {result.portfolio_yield:.2%}, Positions {result.num_positions}")
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Test with single stock
        single_stock = {'SINGLE.NS': self.mock_stocks['ITC.NS']}
        self.optimizer.prepare_data(single_stock)
        
        result = self.optimizer.optimize(
            budget=100000,
            method="GREEDY",
            max_position_size=1.0,  # Allow 100% in single stock
            min_positions=1,
            min_yield=0.001
        )
        
        self.assertEqual(result.num_positions, 1)
        self.assertGreater(result.deployment_rate, 0.95)
        
        # Test with very expensive stocks
        expensive_stocks = {
            'MRF.NS': StockData('MRF.NS', 80000, 800, 0.01, 'info', {}),
            'PAGE.NS': StockData('PAGE.NS', 40000, 600, 0.015, 'info', {})
        }
        self.optimizer.prepare_data(expensive_stocks)
        
        result = self.optimizer.optimize(
            budget=100000,
            method="GREEDY",
            max_position_size=0.60,
            min_positions=1,
            min_yield=0.001
        )
        
        # Should still buy at least one share
        self.assertGreaterEqual(result.num_positions, 1)
        
    def test_yield_prioritization(self):
        """Test that higher yield stocks get prioritized"""
        # Create stocks with varying yields
        yield_stocks = {
            'LOW.NS': StockData('LOW.NS', 100, 2, 0.02, 'info', {}),
            'MED.NS': StockData('MED.NS', 100, 5, 0.05, 'info', {}),
            'HIGH.NS': StockData('HIGH.NS', 100, 8, 0.08, 'info', {}),
            'VHIGH.NS': StockData('VHIGH.NS', 100, 10, 0.10, 'info', {})
        }
        self.optimizer.prepare_data(yield_stocks)
        
        result = self.optimizer.optimize(
            budget=10000,
            method="GREEDY",
            max_position_size=0.30,
            min_positions=2,
            min_yield=0.01
        )
        
        # Find the stocks that were allocated
        allocated_symbols = [a['symbol'] for a in result.stock_allocations]
        
        # HIGH and VHIGH yield stocks should be prioritized
        self.assertIn('VHIGH.NS', allocated_symbols)
        if result.num_positions >= 2:
            self.assertIn('HIGH.NS', allocated_symbols)
        
        # Portfolio yield should be close to the high yields
        self.assertGreater(result.portfolio_yield, 0.07)

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
