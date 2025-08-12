"""
Forward Yield Portfolio Optimization Implementation
=================================================

This module implements the forward yield optimization strategy described:
1. Step 0: Data collection (price, forward dividends, covariance)
2. Step 1: Continuous optimization (QP) to get target weights
3. Step 2A: Round-repair algorithm for integer shares
4. Step 2B: MILP exact share picker (optional)

Uses yfinance for dividend data and cvxpy for optimization.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
import logging
from sklearn.covariance import LedoitWolf
warnings.filterwarnings('ignore')

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class StockData:
    """Container for stock data"""
    symbol: str
    price: float
    forward_dividend: float
    forward_yield: float
    dividend_source: str  # 'info', 'trailing', 'history', 'fallback'
    dividend_metadata: Dict = None  # Cadence info: frequency, cv, confidence, etc.
    
@dataclass
class PortfolioResult:
    """Container for optimization results"""
    target_weights: np.ndarray
    shares: np.ndarray
    executed_weights: np.ndarray
    annual_income: float
    portfolio_yield: float
    residual_cash: float
    drift_l1: float
    allocation_method: str = "Unknown"
    
class ForwardYieldOptimizer:
    """
    Forward yield portfolio optimizer implementing the mathematical framework
    """
    
    def __init__(self):
        self.stocks_data = []
        self.symbols = []
        self.prices = np.array([])
        self.forward_dividends = np.array([])
        self.forward_yields = np.array([])
        self.covariance_matrix = None
        
    def fetch_dividend_data(self, symbols: List[str], period: str = "2y") -> Dict[str, StockData]:
        """
        Fetch dividend and price data using yfinance
        
        Args:
            symbols: List of stock symbols
            period: Period for historical data
            
        Returns:
            Dictionary mapping symbols to StockData
        """
        logger.info(f"Fetching dividend data for {len(symbols)} symbols")
        
        stocks_data = {}
        valid_symbols = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get current price (try fast_info first, then info, then history fallback)
                current_price = None
                
                # Method 1: Try fast_info (fastest, most reliable)
                try:
                    fast_info = ticker.fast_info
                    current_price = fast_info.get('last_price')
                except:
                    pass
                
                # Method 2: Try info fields
                if current_price is None:
                    try:
                        info = ticker.info
                        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    except:
                        pass
                
                # Method 3: Fallback to latest close price
                if current_price is None:
                    try:
                        hist = ticker.history(period="5d")
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                    except:
                        pass
                
                if current_price is None:
                    logger.warning(f"Could not get price for {symbol}")
                    continue
                
                # Get info for dividend data (separate try block to avoid price failure)
                try:
                    info = ticker.info
                except:
                    info = {}
                
                # Get dividend yield - prefer history over info, be careful with quality
                dividend_yield = 0
                dividend_source = 'none'
                
                # Method 1: Calculate from dividend history using sophisticated cadence detection (MOST RELIABLE)
                dividend_metadata = {}
                try:
                    dividends = ticker.dividends
                    if not dividends.empty and len(dividends) > 0:
                        # Use sophisticated forward DPS estimation with cadence detection
                        forward_dps, metadata = self.forward_dps_from_history(
                            dividends, 
                            lookback_days=1460,  # ~4 years for cadence detection
                            irregular_span_days=730,  # ~2 years for irregular flow
                            half_life_days=540   # ~18 months decay
                        )
                        dividend_metadata = metadata
                        
                        if forward_dps > 0:
                            calculated_yield = forward_dps / current_price
                            # Tighter sanity check for history data
                            if 0 < calculated_yield < 0.15:  # Max 15% for calculated yields
                                dividend_yield = calculated_yield
                                dividend_source = 'history'
                                logger.info(f"{symbol} dividend analysis: {metadata['method']}, f={metadata['f']}, "
                                           f"events={metadata['events']}, cv={metadata['cv']:.2f}, "
                                           f"confidence={metadata['confidence']}")
                except Exception as e:
                    if "tz-naive and tz-aware" in str(e):
                        logger.warning(f"{symbol} dividend analysis: timezone handling issue (using fallback)")
                    else:
                        logger.warning(f"{symbol} dividend analysis failed: {str(e)[:50]}...")
                    pass
                
                # Method 2: Try trailingAnnualDividendYield (if no history)
                if dividend_yield == 0:
                    trailing_yield = info.get('trailingAnnualDividendYield', 0)
                    if trailing_yield and 0 < trailing_yield < 0.12:  # Tighter cap
                        dividend_yield = trailing_yield
                        dividend_source = 'trailing'
                
                # Method 3: Try dividendYield from info (LEAST RELIABLE - haircut heavily)
                if dividend_yield == 0:
                    info_yield = info.get('dividendYield', 0)
                    if info_yield and 0 < info_yield < 0.12:  # Tighter cap
                        # Haircut info yields by 50% for Indian stocks (often wrong)
                        dividend_yield = info_yield * 0.5
                        dividend_source = 'info'
                        logger.info(f"Using haircutted info yield for {symbol}: {info_yield:.2%} -> {dividend_yield:.2%}")
                
                # Fallback: Use conservative generic yield if no data found
                if dividend_yield == 0:
                    # Use a conservative generic yield for any stock with no data
                    # This is better than hardcoding specific stocks
                    fallback_yield = 0.015  # 1.5% conservative estimate
                    dividend_yield = fallback_yield
                    dividend_source = 'fallback'
                    logger.warning(f"No dividend data found for {symbol}, using conservative fallback yield: {dividend_yield:.2%}")
                    logger.warning(f"Consider providing better dividend data source for {symbol}")
                
                # Forward dividend is yield * price
                forward_dividend = dividend_yield * current_price if dividend_yield and current_price > 0 else 0
                forward_yield = dividend_yield  # This is the actual yield ratio
                
                stock_data = StockData(
                    symbol=symbol,
                    price=current_price,
                    forward_dividend=forward_dividend,
                    forward_yield=forward_yield,
                    dividend_source=dividend_source,
                    dividend_metadata=dividend_metadata
                )
                
                stocks_data[symbol] = stock_data
                valid_symbols.append(symbol)
                
                logger.debug(f"{symbol}: Price={current_price:.2f}, Div={forward_dividend:.4f}, Yield={forward_yield:.4%}")
                
            except Exception as e:
                logger.warning(f"Error fetching data for {symbol}: {e}")
                continue
        
        logger.info(f"Successfully fetched data for {len(valid_symbols)} symbols")
        return stocks_data
    
    def prepare_data(self, stocks_data: Dict[str, StockData]) -> None:
        """
        Prepare data arrays for optimization
        """
        self.stocks_data = stocks_data  # Store for confidence-based adjustments
        self.symbols = list(stocks_data.keys())
        n = len(self.symbols)
        
        self.prices = np.array([stocks_data[s].price for s in self.symbols])
        self.forward_dividends = np.array([stocks_data[s].forward_dividend for s in self.symbols])
        self.forward_yields = np.array([stocks_data[s].forward_yield for s in self.symbols])
        
        logger.info(f"Data prepared for {n} stocks")
        logger.info(f"Price range: {self.prices.min():.2f} - {self.prices.max():.2f}")
        logger.info(f"Yield range: {self.forward_yields.min():.4%} - {self.forward_yields.max():.4%}")
        
        # Report data source breakdown and confidence analysis
        sources = [stocks_data[s].dividend_source for s in self.symbols]
        source_counts = {src: sources.count(src) for src in set(sources)}
        logger.info(f"Dividend data sources: {source_counts}")
        
        # Report confidence and cadence analysis
        if any(stocks_data[s].dividend_metadata for s in self.symbols):
            confidences = []
            methods = []
            for s in self.symbols:
                meta = stocks_data[s].dividend_metadata or {}
                confidences.append(meta.get('confidence', 'unknown'))
                methods.append(meta.get('method', 'unknown'))
            
            confidence_counts = {conf: confidences.count(conf) for conf in set(confidences)}
            method_counts = {method: methods.count(method) for method in set(methods) if method != 'unknown'}
            
            logger.info(f"Dividend confidence levels: {confidence_counts}")
            if method_counts:
                logger.info(f"Cadence detection methods: {method_counts}")
        
        # Winsorize yields at 95th percentile to handle outliers
        positive_yields = self.forward_yields[self.forward_yields > 0]
        if len(positive_yields) > 0:
            p95 = np.percentile(positive_yields, 95)
            yield_cap = min(0.12, max(0.06, p95))  # Between 6-12%
            
            original_yields = self.forward_yields.copy()
            self.forward_yields = np.clip(self.forward_yields, 0, yield_cap)
            self.forward_dividends = self.forward_yields * self.prices
            
            # Report any winsorization
            winsorized = np.sum(original_yields != self.forward_yields)
            if winsorized > 0:
                logger.info(f"Winsorized {winsorized} yields above {yield_cap:.2%} (95th percentile cap)")
                for i, symbol in enumerate(self.symbols):
                    if original_yields[i] != self.forward_yields[i]:
                        logger.info(f"  {symbol}: {original_yields[i]:.2%} -> {self.forward_yields[i]:.2%}")
    
    def estimate_covariance_matrix(self, period: str = "2y", shrink: bool = False) -> np.ndarray:
        """
        Estimate covariance matrix from historical returns
        """
        logger.info("Estimating covariance matrix from historical returns")
        
        # Download historical data for all symbols (explicit auto_adjust for clarity)
        data = yf.download(self.symbols, period=period, interval="1d", auto_adjust=True)['Close']
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # CRITICAL FIX: Reindex to maintain full symbol order and alignment
        returns = returns.reindex(columns=self.symbols)  # Keep all symbols & order
        
        # Calculate annualized covariance matrix
        if shrink:
            print("Using Ledoit-Wolf shrinkage for covariance estimation...")
            try:
                
                # Remove NaN rows for shrinkage estimation
                returns_clean = returns.dropna()
                if len(returns_clean) > len(self.symbols):  # Need more observations than variables
                    lw = LedoitWolf().fit(returns_clean.values)
                    cov_matrix = pd.DataFrame(lw.covariance_, 
                                            index=returns_clean.columns, 
                                            columns=returns_clean.columns) * 252
                    # Reindex to ensure full symbol coverage
                    cov_matrix = cov_matrix.reindex(index=self.symbols, columns=self.symbols)
                else:
                    print("Insufficient data for shrinkage, falling back to sample covariance")
                    cov_matrix = returns.cov() * 252
            except ImportError:
                print("sklearn not available, falling back to sample covariance")
                cov_matrix = returns.cov() * 252
        else:
            cov_matrix = returns.cov() * 252  # Annualize
        
        # Reindex covariance matrix to ensure proper alignment
        cov_matrix = cov_matrix.reindex(index=self.symbols, columns=self.symbols)
        
        # Fill any missing values with zeros
        cov_matrix = cov_matrix.fillna(0.0)
        
        # Set diagonal floor (much lower than before - 1e-4 instead of 0.01)
        np.fill_diagonal(cov_matrix.values, np.maximum(np.diag(cov_matrix.values), 1e-4))
        
        # Ensure positive semi-definite matrix
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix.values)
        eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
        cov_matrix_psd = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        self.covariance_matrix = cov_matrix_psd
        
        # Report data availability
        available_symbols = [s for s in self.symbols if s in returns.columns and not returns[s].isna().all()]
        print(f"Return data available for {len(available_symbols)}/{len(self.symbols)} symbols")
        
        print(f"Covariance matrix shape: {self.covariance_matrix.shape}")
        print(f"Average volatility: {np.sqrt(np.diag(self.covariance_matrix)).mean():.4%}")
        
        return self.covariance_matrix
    
    def _effective_caps(self, base_cap: float = 0.15) -> np.ndarray:
        """
        Compute effective caps based on data quality, confidence, and yield validity.
        Uses sophisticated dividend metadata for precise confidence assessment.
        """
        caps = np.full(len(self.symbols), base_cap)
        
        # Reduce caps based on dividend data confidence and source quality
        for i, symbol in enumerate(self.symbols):
            if hasattr(self, 'stocks_data') and symbol in self.stocks_data:
                stock = self.stocks_data[symbol]
                source = stock.dividend_source
                metadata = stock.dividend_metadata or {}
                confidence = metadata.get('confidence', 'low')
                
                # Primary adjustment based on confidence from cadence analysis
                if confidence == 'very_low':
                    caps[i] = min(caps[i], 0.03)  # Max 3% for very low confidence
                elif confidence == 'low':
                    caps[i] = min(caps[i], 0.05)  # Max 5% for low confidence
                elif confidence == 'medium':
                    caps[i] = min(caps[i], 0.12)  # Max 12% for medium confidence (relaxed from 10%)
                # high confidence keeps base_cap (15%)
                
                # Secondary adjustment based on data source
                if source == 'fallback':
                    caps[i] = min(caps[i], 0.05)  # Max 5% for fallback data
                elif source == 'info':
                    caps[i] = min(caps[i], 0.08)  # Max 8% for info data (often unreliable)
                
                # Additional adjustment for irregular dividend patterns
                if metadata.get('cv', 0) > 0.6:  # Very irregular
                    caps[i] = min(caps[i], 0.06)  # Max 6% for very irregular payers
                elif not metadata.get('regular', True):  # Irregular but not too bad
                    caps[i] = min(caps[i], 0.08)  # Max 8% for irregular payers
        
        # Zero-cap very low yield names (< 0.5%) to reduce noise
        caps = caps * (self.forward_yields >= 0.005).astype(float)  # 0.5% minimum
        
        return caps
    
    def _post_round_risk_repair(self, shares: np.ndarray, budget: float, risk_cap_vol: float, max_iters: int = 50) -> tuple:
        """
        Post-round risk repair: if volatility exceeds cap, greedily sell shares
        that reduce risk most per unit income lost until back under cap.
        """
        shares = shares.copy().astype(int)
        
        for iteration in range(max_iters):
            invested = float(np.sum(shares * self.prices))
            if invested <= 0:
                break
                
            w_exec = (shares * self.prices) / invested
            vol = float(np.sqrt(w_exec @ self.covariance_matrix @ w_exec))
            
            if vol <= risk_cap_vol:
                break  # Under cap, we're done
            
            # Find best share to sell (reduce vol most per income lost)
            best_i, best_score = None, -np.inf
            for i in np.where(shares > 0)[0]:
                # Try selling one share
                shares_trial = shares.copy()
                shares_trial[i] -= 1
                invested_trial = float(np.sum(shares_trial * self.prices))
                
                if invested_trial <= 0:
                    continue
                    
                w_trial = (shares_trial * self.prices) / invested_trial
                vol_trial = float(np.sqrt(w_trial @ self.covariance_matrix @ w_trial))
                
                dvol = vol - vol_trial  # Volatility reduction
                dincome = self.forward_dividends[i]  # Income lost
                
                # Score: volatility reduction per unit income lost
                score = dvol / max(dincome, 1e-9)
                if score > best_score:
                    best_score, best_i = score, i
            
            if best_i is None:
                print(f"    Risk repair stuck at iteration {iteration}")
                break
                
            shares[best_i] -= 1
            
        # Final risk calculation
        invested = float(np.sum(shares * self.prices))
        w_exec = (shares * self.prices) / invested if invested > 0 else np.zeros_like(self.prices)
        final_vol = float(np.sqrt(w_exec @ self.covariance_matrix @ w_exec))
        
        return shares, final_vol

    def optimize_continuous(self, 
                          max_risk_variance: float = 0.04,
                          individual_caps: Optional[np.ndarray] = None,
                          sector_caps: Optional[Dict[str, float]] = None,
                          sector_mapping: Optional[Dict[str, str]] = None) -> np.ndarray:
        """
        Step 1: Continuous optimization using QP
        
        Args:
            max_risk_variance: Maximum portfolio variance (σ²_max)
            individual_caps: Individual stock weight caps (default 15% each)
            sector_caps: Sector weight caps
            sector_mapping: Mapping of symbols to sectors
            
        Returns:
            Optimal weights w*
        """
        print("Running continuous optimization (QP)...")
        
        n = len(self.symbols)
        
        # Use consistent cap logic
        # --- PATCH: enforce low-yield floor regardless of who provides caps ---
        LOW_YIELD_FLOOR = 0.005  # 0.5%
        
        if individual_caps is None:
            effective_caps = self._effective_caps(base_cap=0.15)
            
            # Report cap adjustments
            for i, symbol in enumerate(self.symbols):
                if hasattr(self, 'stocks_data') and symbol in self.stocks_data:
                    source = self.stocks_data[symbol].dividend_source
                    if source == 'fallback' and effective_caps[i] <= 0.05:
                        print(f"Capped {symbol} at {effective_caps[i]:.1%} due to fallback dividend data")
                    elif source == 'info' and effective_caps[i] <= 0.10:
                        print(f"Capped {symbol} at {effective_caps[i]:.1%} due to info dividend data")
        else:
            effective_caps = individual_caps.copy()
        
        # ALWAYS apply a zero-cap mask to very low yield names
        low_yield_mask = (self.forward_yields >= LOW_YIELD_FLOOR).astype(float)
        effective_caps = np.minimum(effective_caps, low_yield_mask)
        
        # Filter out stocks with very low yield (< 0.5%) to improve optimization
        valid_indices = self.forward_yields >= 0.005  # 0.5% minimum
        if not np.any(valid_indices):
            print("Warning: No stocks with sufficient yields (≥0.5%) found!")
            return np.full(n, 1.0/n)
        
        print(f"Optimizing {np.sum(valid_indices)} stocks with yields ≥0.5%")
        print(f"Yield range: {self.forward_yields[valid_indices].min():.4%} - {self.forward_yields[valid_indices].max():.4%}")
        
        # Decision variables
        w = cp.Variable(n, nonneg=True)
        
        # Objective: maximize forward yield with tiny spreader to avoid corner solutions
        objective = cp.Maximize(self.forward_yields @ w - 1e-6 * cp.sum_squares(w))
        
        # effective_caps already computed above with low-yield masking
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            w <= effective_caps,  # Individual caps (zero for zero-yield stocks)
        ]
        
        # Add risk constraint - should work now with proper alignment
        try:
            # Test the quadratic form
            test_w = np.ones(n) / n
            test_risk = test_w @ self.covariance_matrix @ test_w
            print(f"Test portfolio risk: {np.sqrt(test_risk):.4%}")
            
            if test_risk > 0 and np.isfinite(test_risk):
                constraints.append(cp.quad_form(w, self.covariance_matrix) <= max_risk_variance)
                max_vol_cap = np.sqrt(max_risk_variance)
                self._last_risk_cap = max_vol_cap  # Store for post-round repair
                print(f"Added risk constraint: max variance = {max_risk_variance:.4%} (vol = {max_vol_cap:.4%})")
            else:
                print("Warning: Skipping risk constraint due to invalid covariance matrix")
        except Exception as e:
            print(f"Warning: Could not add risk constraint: {e}")
            import traceback
            traceback.print_exc()
        
        # Add sector constraints if provided
        if sector_caps and sector_mapping:
            for sector, cap in sector_caps.items():
                sector_indices = [i for i, symbol in enumerate(self.symbols) 
                                if sector_mapping.get(symbol) == sector]
                if sector_indices:
                    constraints.append(cp.sum(w[sector_indices]) <= cap)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        # Try different solvers in order of preference for SOC problems
        # ECOS and SCS are better for quadratic constraints than OSQP
        solvers_to_try = [cp.ECOS, cp.SCS, cp.CVXOPT, cp.OSQP]
        
        solved = False
        for solver in solvers_to_try:
            try:
                problem.solve(solver=solver, verbose=False)
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    solved = True
                    break
            except Exception as e:
                print(f"Solver {solver} failed: {e}")
                continue
        
        if not solved or problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Warning: Optimization failed with status: {problem.status}")
            # Return equal weights as fallback
            return np.full(n, 1.0/n)
        
        optimal_weights = w.value
        
        # Report results
        portfolio_yield = self.forward_yields @ optimal_weights
        portfolio_risk = np.sqrt(optimal_weights @ self.covariance_matrix @ optimal_weights)
        
        print(f"Optimal portfolio yield: {portfolio_yield:.4%}")
        print(f"Portfolio risk (volatility): {portfolio_risk:.4%}")
        print(f"Number of non-zero weights: {np.sum(optimal_weights > 1e-6)}")
        
        return optimal_weights
    
    def allocate_shares_greedy(self, 
                              target_weights: np.ndarray, 
                              budget: float,
                              individual_caps: Optional[np.ndarray] = None,
                              sector_caps: Optional[Dict[str, float]] = None,
                              sector_mapping: Optional[Dict[str, str]] = None,
                              min_names: Optional[int] = None,
                              residual_threshold: float = 0.005,
                              seed: Optional[int] = 42) -> PortfolioResult:
        """
        Step 2A: Fast deterministic round-repair algorithm with optional seeding for reproducibility
        
        Args:
            target_weights: Optimal weights from continuous optimization
            budget: Total cash to deploy
            individual_caps: Individual stock weight caps
            sector_caps: Sector weight caps  
            sector_mapping: Mapping of symbols to sectors
            residual_threshold: Stop when residual cash is below this fraction of budget
            
        Returns:
            PortfolioResult with integer shares allocation
        """
        print(f"Allocating shares with budget: ₹{budget:,.0f}")
        
        n = len(self.symbols)
        
        # Use consistent cap logic
        if individual_caps is None:
            individual_caps = self._effective_caps(base_cap=0.15)
            
        # Step 1: Floor to lots (1 share = 1 lot for equity)
        max_shares_by_cap = np.floor(individual_caps * budget / self.prices).astype(int)
        initial_shares = np.floor(target_weights * budget / self.prices).astype(int)
        
        # Enforce individual caps immediately
        shares = np.minimum(initial_shares, max_shares_by_cap)
        
        # Calculate remaining cash
        invested_amount = np.sum(shares * self.prices)
        remaining_cash = budget - invested_amount
        
        print(f"After initial flooring: ₹{invested_amount:,.0f} invested, ₹{remaining_cash:,.0f} remaining")
        
        # Set seed for reproducible tie-breaking (if provided)
        if seed is not None:
            np.random.seed(seed)
        
        # Precompute sector indices for efficiency (avoid O(n²) lookup in loop)
        sector_indices_map = {}
        if sector_caps and sector_mapping:
            for sector in sector_caps.keys():
                sector_indices_map[sector] = [j for j, s in enumerate(self.symbols) 
                                            if sector_mapping.get(s) == sector]
        
        # Min names prefill: ensure we have at least min_names positions
        if min_names and min_names > 0:
            current_names = np.sum(shares > 0)
            if current_names < min_names:
                # Pick top-yield feasible names not already allocated
                yield_order = np.argsort(-self.forward_yields)
                for i in yield_order:
                    if shares[i] == 0 and self.prices[i] <= remaining_cash:
                        # Check individual cap feasibility
                        new_weight = self.prices[i] / budget
                        if new_weight <= individual_caps[i]:
                            # Check sector cap feasibility if applicable
                            feasible = True
                            if sector_caps and sector_mapping:
                                symbol = self.symbols[i]
                                sector = sector_mapping.get(symbol)
                                if sector and sector in sector_caps and sector in sector_indices_map:
                                    sector_indices = sector_indices_map[sector]
                                    current_sector_weight = np.sum([shares[j] * self.prices[j] 
                                                                  for j in sector_indices]) / budget
                                    if current_sector_weight + new_weight > sector_caps[sector]:
                                        feasible = False
                            
                            if feasible:
                                shares[i] += 1
                                remaining_cash -= self.prices[i]
                                current_names += 1
                                print(f"Min names prefill: bought 1 share of {self.symbols[i]} (yield: {self.forward_yields[i]:.2%})")
                                
                                if current_names >= min_names or remaining_cash <= 0:
                                    break
                
                invested_amount = np.sum(shares * self.prices)
                remaining_cash = budget - invested_amount
                print(f"After min names prefill: {current_names} names, ₹{remaining_cash:,.0f} remaining")
        
        # Step 2: Greedy fill with remaining cash
        iteration = 0
        max_iterations = 1000
        
        while remaining_cash > 0 and iteration < max_iterations:
            # Find feasible stocks (those we can still buy)
            feasible_mask = np.zeros(n, dtype=bool)
            
            for i in range(n):
                # Check if we can afford one more share
                if self.prices[i] <= remaining_cash:
                    # Check individual cap
                    current_weight = (shares[i] * self.prices[i]) / budget
                    if current_weight + (self.prices[i] / budget) <= individual_caps[i]:
                        # Check sector cap if applicable
                        if sector_caps and sector_mapping:
                            symbol = self.symbols[i]
                            sector = sector_mapping.get(symbol)
                            if sector and sector in sector_caps and sector in sector_indices_map:
                                # Calculate current sector weight using precomputed indices
                                sector_indices = sector_indices_map[sector]
                                current_sector_weight = np.sum([shares[j] * self.prices[j] 
                                                              for j in sector_indices]) / budget
                                if current_sector_weight + (self.prices[i] / budget) > sector_caps[sector]:
                                    continue
                        
                        feasible_mask[i] = True
            
            if not np.any(feasible_mask):
                break
                
            # Among feasible stocks, pick the one with highest yield
            # Tie-break: pick the one with largest deficit to target
            current_weights = (shares * self.prices) / budget
            deficits = np.maximum(0, target_weights - current_weights)
            
            # Calculate selection criterion: yield + small deficit bonus
            feasible_indices = np.where(feasible_mask)[0]
            yields_feasible = self.forward_yields[feasible_indices]
            deficits_feasible = deficits[feasible_indices]
            
            # --- PATCH: better scoring when lots are chunky ---
            # price_units = weight jump from buying 1 share (as a fraction of budget)
            price_units = self.prices[feasible_indices] / budget
            # deterministic, yield-dominant scorer; deficit scaled by price granularity
            score = 1000.0 * yields_feasible + deficits_feasible / np.maximum(price_units, 1e-9)
            
            # Add tiny random noise for reproducible tie-breaking (if seeded)
            if seed is not None:
                score += np.random.normal(0, 1e-10, len(score))
            
            best_idx_feasible = int(np.argmax(score))
            best_idx = feasible_indices[best_idx_feasible]
            
            # Buy one more share of the best stock
            shares[best_idx] += 1
            remaining_cash -= self.prices[best_idx]
            
            iteration += 1
            
            # Check residual threshold
            if remaining_cash / budget <= residual_threshold:
                break
        
        # Calculate final portfolio metrics
        final_invested = np.sum(shares * self.prices)
        executed_weights = (shares * self.prices) / budget
        annual_income = np.sum(self.forward_dividends * shares)
        portfolio_yield = annual_income / budget  # Yield on total budget
        yield_on_invested = annual_income / final_invested if final_invested > 0 else 0.0  # Yield on invested amount
        drift_l1 = np.sum(np.abs(executed_weights - target_weights))
        
        # Post-round risk check (normalize to invested weights for accurate calculation)
        invested = float(np.sum(shares * self.prices))
        w_exec = (shares * self.prices) / invested if invested > 0 else np.zeros_like(self.prices)
        post_vol = float(np.sqrt(w_exec @ self.covariance_matrix @ w_exec))
        
        # Optional: risk repair if post-round volatility exceeds the cap
        max_vol_cap = getattr(self, '_last_risk_cap', 0.25)  # Use last cap or 25% default
        if post_vol > max_vol_cap:
            print(f"  Warning: Post-round vol {post_vol:.4%} exceeds cap {max_vol_cap:.4%}. Running risk repair...")
            shares, post_vol = self._post_round_risk_repair(shares, budget, max_vol_cap)
            # Recalculate metrics after repair
            invested = float(np.sum(shares * self.prices))
            w_exec = (shares * self.prices) / invested if invested > 0 else np.zeros_like(self.prices)
            final_invested = invested
            remaining_cash = budget - invested
            executed_weights = (shares * self.prices) / budget
            annual_income = float(np.sum(shares * self.forward_dividends))
            portfolio_yield = annual_income / budget if budget > 0 else 0.0
            yield_on_invested = annual_income / invested if invested > 0 else 0.0
            drift_l1 = np.sum(np.abs(executed_weights - target_weights))
        
        print(f"Final allocation:")
        print(f"  Shares bought: {np.sum(shares)} total shares")
        print(f"  Amount invested: ₹{final_invested:,.0f}")
        print(f"  Residual cash: ₹{remaining_cash:,.0f} ({remaining_cash/budget:.2%})")
        print(f"  Portfolio yield (on budget): {portfolio_yield:.4%}")
        print(f"  Yield on invested amount: {yield_on_invested:.4%}")
        print(f"  Post-round volatility: {post_vol:.4%}")
        print(f"  L1 drift from target: {drift_l1:.4%}")
        
        return PortfolioResult(
            target_weights=target_weights,
            shares=shares,
            executed_weights=executed_weights,
            annual_income=annual_income,
            portfolio_yield=portfolio_yield,
            residual_cash=remaining_cash,
            drift_l1=drift_l1,
            allocation_method="Greedy (floor-repair)"
        )
    
    def preflight_granularity(self, budget: float, w_star: Optional[np.ndarray] = None, 
                             min_names: Optional[int] = None) -> Dict:
        """
        Check if budget/price granularity requires exact share-level optimization
        """
        feas_any = np.any(self.prices <= budget)
        if not feas_any:
            return {
                "feasible": False,
                "reason": "All prices exceed budget",
                "N_target": 0.0,
                "g_max": self.prices.max() / budget
            }
        
        if min_names is not None:
            feas_names = np.sum(self.prices <= budget)
            if feas_names < min_names:
                return {
                    "feasible": False,
                    "reason": f"Budget supports only {feas_names} names < {min_names}",
                    "N_target": 0.0,
                    "g_max": self.prices.max() / budget
                }
        
        N_target = 0.0
        if w_star is not None:
            N_target = float(np.sum((w_star * budget) / self.prices))
        
        g_max = float(np.max(self.prices) / budget)
        
        return {
            "feasible": True, 
            "reason": None, 
            "N_target": N_target, 
            "g_max": g_max
        }
    
    def should_use_milp(self, N_target: float, g_max: float, 
                       min_N: float = 25, max_g: float = 0.10) -> bool:
        """
        Decision rule: use MILP for small budgets or chunky prices
        """
        return (N_target < min_N) or (g_max > max_g)
    
    def dynamic_thresholds(self, budget: float) -> Dict[str, float]:
        """
        Budget-aware thresholds: stricter for small budgets to ensure discrete feasibility
        """
        if budget < 50_000:
            return {"N_min": 10, "g_max": 0.05}  # Very small budgets need MILP sooner
        elif budget < 200_000:
            return {"N_min": 15, "g_max": 0.08}  # Small budgets need more care
        else:
            return {"N_min": 25, "g_max": 0.10}  # Large budgets can use greedy more often
    
    def solve_income_milp(self,
                         target_weights: np.ndarray,
                         budget: float,
                         individual_caps: Optional[np.ndarray] = None,
                         sector_caps: Optional[Dict[str, float]] = None,
                         sector_mapping: Optional[Dict[str, str]] = None,
                         min_invest_frac: float = 0.995,
                         min_names: Optional[int] = None,
                         min_lot: int = 1,
                         epsilon_income: float = 1.0,
                         verbose: bool = False) -> PortfolioResult:
        """
        Exact share-level income MILP with 2-phase optimization:
        Phase 1: Maximize income
        Phase 2: Minimize L1 drift to target weights while preserving income
        """
        print(f"Running exact share-level MILP optimization with budget: ₹{budget:,.0f}")
        
        n = len(self.symbols)
        
        # Use consistent cap logic
        if individual_caps is None:
            individual_caps = self._effective_caps(base_cap=0.15)
        
        # Max shares from caps and budget
        max_by_cap = np.floor(individual_caps * budget / self.prices).astype(int)
        max_by_budget = np.floor(budget / self.prices).astype(int)
        U = np.minimum(max_by_cap, max_by_budget)
        U = np.maximum(U, 0)
        
        # Quick infeasible filter
        if np.all(U == 0):
            print("Budget too small after caps: falling back to greedy")
            return self.allocate_shares_greedy(target_weights, budget, individual_caps,
                                             sector_caps, sector_mapping)
        
        # Precompute sector indices
        sector_indices_map = {}
        if sector_caps and sector_mapping:
            for sector in sector_caps.keys():
                sector_indices_map[sector] = [j for j, s in enumerate(self.symbols) 
                                            if sector_mapping.get(s) == sector]
        
        # -------- Phase 1: Maximize Income --------
        x = cp.Variable(n, integer=True)
        
        constraints = [
            x >= 0,  # Nonnegativity constraint
            x <= U,
            cp.sum(cp.multiply(self.prices, x)) <= budget,
            cp.sum(cp.multiply(self.prices, x)) >= min_invest_frac * budget,
        ]
        
        # Sector constraints
        for sector, cap in (sector_caps or {}).items():
            if sector in sector_indices_map and cap < 1.0:
                sector_indices = sector_indices_map[sector]
                sector_value = cp.sum([cp.multiply(self.prices[i], x[i]) for i in sector_indices])
                constraints.append(sector_value <= cap * budget)
        
        # Min names constraint
        if min_names is not None and min_names > 0:
            z = cp.Variable(n, boolean=True)
            constraints.extend([
                cp.sum(z) >= min_names,
                x >= min_lot * z,
                x <= U * z
            ])
        
        # Phase 1 objective: maximize income
        objective1 = cp.Maximize(self.forward_dividends @ x)
        problem1 = cp.Problem(objective1, constraints)
        
        # Solve Phase 1
        solvers_to_try = [cp.GUROBI, cp.MOSEK, cp.CPLEX, cp.SCIP, cp.GLPK_MI, cp.CBC]
        solved1 = False
        
        for solver in solvers_to_try:
            try:
                problem1.solve(solver=solver, verbose=verbose)
                if problem1.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and x.value is not None:
                    solved1 = True
                    break
            except Exception as e:
                if verbose:
                    print(f"Solver {solver} failed: {e}")
                continue
        
        if not solved1:
            print("MILP Phase 1 failed: falling back to greedy")
            return self.allocate_shares_greedy(target_weights, budget, individual_caps,
                                             sector_caps, sector_mapping)
        
        # Extract Phase 1 solution
        x_phase1 = np.round(np.maximum(0, x.value)).astype(int)
        income_star = float(self.forward_dividends @ x_phase1)
        
        print(f"Phase 1 complete: optimal income = ₹{income_star:,.2f}")
        
        # -------- Phase 2: Minimize L1 drift while preserving income --------
        t = cp.Variable(n, nonneg=True)  # absolute deviation auxiliaries
        
        constraints2 = [
            x <= U,
            cp.sum(cp.multiply(self.prices, x)) <= budget,
            cp.sum(cp.multiply(self.prices, x)) >= min_invest_frac * budget,
            self.forward_dividends @ x >= income_star - epsilon_income,  # Preserve income
        ]
        
        # Sector constraints (same as Phase 1)
        for sector, cap in (sector_caps or {}).items():
            if sector in sector_indices_map and cap < 1.0:
                sector_indices = sector_indices_map[sector]
                sector_value = cp.sum([cp.multiply(self.prices[i], x[i]) for i in sector_indices])
                constraints2.append(sector_value <= cap * budget)
        
        # Min names constraint (same as Phase 1)
        if min_names is not None and min_names > 0:
            constraints2.extend([
                cp.sum(z) >= min_names,
                x >= min_lot * z,
                x <= U * z
            ])
        
        # Absolute deviation around target rupee weights
        target_rupees = target_weights * budget
        constraints2.extend([
            t >= cp.multiply(self.prices, x) - target_rupees,
            t >= -(cp.multiply(self.prices, x) - target_rupees)
        ])
        
        # Phase 2 objective: minimize L1 drift
        objective2 = cp.Minimize(cp.sum(t))
        problem2 = cp.Problem(objective2, constraints2)
        
        # Solve Phase 2
        solved2 = False
        for solver in solvers_to_try:
            try:
                problem2.solve(solver=solver, verbose=verbose)
                if problem2.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and x.value is not None:
                    solved2 = True
                    break
            except Exception as e:
                if verbose:
                    print(f"Phase 2 solver {solver} failed: {e}")
                continue
        
        # Use Phase 2 solution if successful, otherwise fallback to Phase 1
        if solved2:
            shares = np.round(np.maximum(0, x.value)).astype(int)
            print("Phase 2 complete: optimized for closeness to target weights")
        else:
            shares = x_phase1
            print("Phase 2 failed: using Phase 1 solution (income-optimal)")
        
        # Calculate final metrics
        final_invested = np.sum(shares * self.prices)
        remaining_cash = budget - final_invested
        executed_weights = (shares * self.prices) / budget
        annual_income = np.sum(self.forward_dividends * shares)
        portfolio_yield = annual_income / budget
        yield_on_invested = annual_income / final_invested if final_invested > 0 else 0.0
        drift_l1 = np.sum(np.abs(executed_weights - target_weights))
        
        # Post-round risk check (normalize to invested weights for accurate calculation)
        invested = float(np.sum(shares * self.prices))
        w_exec = (shares * self.prices) / invested if invested > 0 else np.zeros_like(self.prices)
        post_vol = float(np.sqrt(w_exec @ self.covariance_matrix @ w_exec))
        
        print(f"Exact MILP solution:")
        print(f"  Shares bought: {np.sum(shares)} total shares")
        print(f"  Amount invested: ₹{final_invested:,.0f}")
        print(f"  Residual cash: ₹{remaining_cash:,.0f} ({remaining_cash/budget:.2%})")
        print(f"  Portfolio yield (on budget): {portfolio_yield:.4%}")
        print(f"  Yield on invested amount: {yield_on_invested:.4%}")
        print(f"  Post-round volatility: {post_vol:.4%}")
        print(f"  L1 drift from target: {drift_l1:.4%}")
        
        return PortfolioResult(
            target_weights=target_weights,
            shares=shares,
            executed_weights=executed_weights,
            annual_income=annual_income,
            portfolio_yield=portfolio_yield,
            residual_cash=remaining_cash,
            drift_l1=drift_l1,
            allocation_method="MILP (exact share-level)"
        )
    
    def allocate_shares_auto(self,
                           target_weights: np.ndarray,
                           budget: float,
                           individual_caps: Optional[np.ndarray] = None,
                           sector_caps: Optional[Dict[str, float]] = None,
                           sector_mapping: Optional[Dict[str, str]] = None,
                           min_names: Optional[int] = None,
                           thresholds: Dict = None,
                           min_invest_frac: float = 0.995,
                           seed: Optional[int] = 42,
                           verbose: bool = False) -> PortfolioResult:
        """
        Auto-chooser that intelligently switches between greedy and exact MILP
        based on budget granularity and price chunkiness
        """
        if thresholds is None:
            thresholds = self.dynamic_thresholds(budget)
        
        # Check granularity
        granularity = self.preflight_granularity(budget, target_weights, min_names)
        
        if not granularity["feasible"]:
            raise ValueError(f"Budget infeasible: {granularity['reason']}")
        
        # Decision logic
        use_milp = self.should_use_milp(
            granularity["N_target"], 
            granularity["g_max"],
            min_N=thresholds.get("N_min", 25),
            max_g=thresholds.get("g_max", 0.10)
        )
        
        print(f"Granularity check:")
        print(f"  Expected shares: {granularity['N_target']:.1f}")
        print(f"  Max granularity: {granularity['g_max']:.2%}")
        print(f"  Method: {'Exact MILP' if use_milp else 'Greedy allocation'}")
        
        if use_milp:
            print("\nUsing exact share-level optimization due to budget/price granularity")
            return self.solve_income_milp(
                target_weights=target_weights,
                budget=budget,
                individual_caps=individual_caps,
                sector_caps=sector_caps,
                sector_mapping=sector_mapping,
                min_invest_frac=min_invest_frac,
                min_names=min_names,
                verbose=verbose
            )
        else:
            print("\nUsing fast greedy allocation")
            return self.allocate_shares_greedy(
                target_weights=target_weights,
                budget=budget,
                individual_caps=individual_caps,
                sector_caps=sector_caps,
                sector_mapping=sector_mapping,
                min_names=min_names,
                seed=seed
            )
    
    # Duplicate method removed - using _post_round_risk_repair instead
    
    def _guardrail_clip_specials(self, vals: np.ndarray) -> np.ndarray:
        """
        Clip special dividends at 2x median to handle one-offs
        """
        if len(vals) == 0:
            return vals
        med = np.median(vals[-6:]) if len(vals) >= 2 else np.median(vals)
        if med <= 0:
            return vals
        return np.minimum(vals, 2.0 * med)
    
    def detect_cadence(self, div_series: pd.Series, lookback_days: int = 1460) -> Tuple[int, float, bool, pd.DataFrame]:
        """
        Detect dividend payment cadence from ~4 years of history
        
        Returns:
            f: Frequency (1=annual, 2=semi, 3=quarterly, 4=quarterly+)
            cv: Coefficient of variation (regularity measure)
            regular: True if regular pattern (cv <= 0.35)
            events: DataFrame of dividend events
        """
        if div_series is None or div_series.empty:
            return 1, np.inf, False, pd.DataFrame(columns=["date", "amt"])
        
        s = div_series.sort_index()
        
        # Handle timezone issues in cutoff calculation
        try:
            cutoff = s.index.max() - pd.Timedelta(days=lookback_days)
            s = s[s.index >= cutoff]
        except Exception as e:
            # If timezone issues, convert to naive and retry
            try:
                s_naive = s.copy()
                s_naive.index = s_naive.index.tz_localize(None) if hasattr(s_naive.index, 'tz') and s_naive.index.tz else s_naive.index
                cutoff = s_naive.index.max() - pd.Timedelta(days=lookback_days)
                s = s_naive[s_naive.index >= cutoff]
            except:
                # Last resort: just use recent portion
                s = s.tail(min(20, len(s)))
        
        if s.shape[0] < 2:
            events_df = pd.DataFrame({"date": s.index, "amt": s.values})
            return 1, np.inf, False, events_df
        
        # Calculate gaps between dividend payments
        dates = s.index.to_series().sort_values()
        gaps = dates.diff().dt.days.dropna().values
        
        med_gap = np.median(gaps)
        f = int(np.clip(np.round(365.0 / max(1.0, med_gap)), 1, 4))
        cv = float(np.std(gaps) / max(1e-9, np.mean(gaps)))
        regular = cv <= 0.45  # Slightly relaxed from 0.35 for more semi/annual payers
        
        events_df = pd.DataFrame({"date": s.index, "amt": s.values})
        return f, cv, regular, events_df
    
    def forward_dps_from_history(self, div_series: pd.Series,
                                today: Optional[pd.Timestamp] = None,
                                lookback_days: int = 1460,
                                irregular_span_days: int = 730,
                                half_life_days: int = 540) -> Tuple[float, Dict]:
        """
        Robust forward 12-month dividend per share estimation from history
        
        Args:
            div_series: Historical dividend series
            today: Reference date (default: today)
            lookback_days: Window for cadence detection (~4 years)
            irregular_span_days: Window for irregular payers (~2 years)
            half_life_days: Half-life for exponential decay (~18 months)
            
        Returns:
            forward_dps: Estimated 12-month forward dividend per share
            metadata: Dict with frequency, cv, regularity, event count
        """
        # Normalize timezone handling - convert everything to naive for simplicity
        try:
            if hasattr(div_series.index, 'tz') and div_series.index.tz is not None:
                div_series = div_series.copy()
                div_series.index = div_series.index.tz_localize(None)
        except:
            pass
        
        if today is None:
            today = pd.Timestamp.today()  # Always use naive timestamp
        
        f, cv, regular, events = self.detect_cadence(div_series, lookback_days)
        
        if events.empty:
            return 0.0, {"f": f, "cv": cv, "regular": regular, "events": 0, "confidence": "none"}
        
        # Apply specials guardrail and exponentially weighted average of recent events
        events = events.sort_values("date")
        vals = events["amt"].values.astype(float)
        vals = self._guardrail_clip_specials(vals)
        
        # Use last m events where m = min(6, 2*f + 2)
        m = int(min(6, 2 * f + 2))
        tail = vals[-m:]
        tail_dates = events["date"].values[-m:]
        
        # Calculate exponential weights based on age (simplified - all timestamps are naive now)
        ages = []
        for d in tail_dates:
            try:
                age_days = (today - pd.Timestamp(d)).days
                ages.append(max(0, age_days))  # Ensure non-negative ages
            except Exception:
                # Fallback: assume 30 days old if calculation fails
                ages.append(30)
        
        ages = np.array(ages, float)
        lam = np.log(2) / max(1.0, half_life_days)  # Exponential decay parameter
        w = np.exp(-lam * np.clip(ages, 0, None))
        
        # Robust weighted average dividend per event
        d_bar = float(np.sum(w * tail) / max(1e-12, np.sum(w))) if len(tail) > 0 else 0.0
        
        # Estimate forward DPS based on regularity
        if regular and f > 0:
            # Regular payers: frequency * average amount
            D_hat = f * d_bar
            method = f"regular_f{f}"
        else:
            # Irregular payers: annualize flow over span
            cutoff = events["date"].max() - pd.Timedelta(days=irregular_span_days)
            recent_events = events.loc[events["date"] >= cutoff, "amt"]
            flow = recent_events.sum()
            span_days = (events["date"].max() - max(cutoff, events["date"].min())).days + 1
            D_hat = 365.0 * float(flow) / max(365.0, float(span_days))
            method = f"irregular_span{irregular_span_days}d"
        
        # Determine confidence level
        if events.shape[0] < 2:
            confidence = "very_low"
        elif cv > 0.6 or events.shape[0] < 4:
            confidence = "low"
        elif regular and events.shape[0] >= 6:
            confidence = "high"
        else:
            confidence = "medium"
        
        metadata = {
            "f": f,
            "cv": cv,
            "regular": regular,
            "events": int(events.shape[0]),
            "confidence": confidence,
            "method": method,
            "d_bar": d_bar
        }
        
        return D_hat, metadata
    
    def sum_recent_dividends_guardrailed(self, div_series: pd.Series, 
                                       max_divs: int = 4, lookback_days: int = 800) -> float:
        """
        Legacy method - now delegates to new sophisticated estimation
        """
        forward_dps, _ = self.forward_dps_from_history(div_series, lookback_days=lookback_days)
        return forward_dps
    
    def allocate_shares_milp(self,
                           target_weights: np.ndarray,
                           budget: float,
                           individual_caps: Optional[np.ndarray] = None,
                           sector_caps: Optional[Dict[str, float]] = None,
                           sector_mapping: Optional[Dict[str, str]] = None,
                           min_names: Optional[int] = None,
                           lambda_drift: float = 0.1) -> PortfolioResult:
        """
        Step 2B: MILP exact share picker
        
        Args:
            target_weights: Optimal weights from continuous optimization
            budget: Total cash to deploy
            individual_caps: Individual stock weight caps
            sector_caps: Sector weight caps
            sector_mapping: Mapping of symbols to sectors
            min_names: Minimum number of stocks to hold
            lambda_drift: Trade-off parameter between income and staying close to target
            
        Returns:
            PortfolioResult with optimal integer shares allocation
        """
        print(f"Running MILP optimization with budget: ₹{budget:,.0f}")
        
        n = len(self.symbols)
        
        # Use consistent cap logic
        if individual_caps is None:
            individual_caps = self._effective_caps(base_cap=0.15)
            
        # Decision variables
        x = cp.Variable(n, integer=True)  # shares
        t = cp.Variable(n, nonneg=True)  # absolute deviation auxiliaries
        
        # Binary variables for min_names constraint
        if min_names:
            z = cp.Variable(n, boolean=True)  # invested-name indicators
            
        # Objective: maximize income - lambda * drift
        median_dividend = np.median(self.forward_dividends[self.forward_dividends > 0])
        lambda_scaled = lambda_drift * median_dividend
        
        objective = cp.Maximize(self.forward_dividends @ x - lambda_scaled * cp.sum(t))
        
        # Constraints
        constraints = [
            x >= 0,  # Nonnegativity constraint
            cp.sum(cp.multiply(self.prices, x)) <= budget,  # Budget constraint (upper bound)
            cp.sum(cp.multiply(self.prices, x)) >= 0.995 * budget,  # Minimum investment (avoid under-investing)
        ]
        
        # Individual caps
        max_shares_by_cap = np.floor(individual_caps * budget / self.prices).astype(int)
        for i in range(n):
            constraints.append(x[i] <= max_shares_by_cap[i])
            
        # Sector caps
        if sector_caps and sector_mapping:
            for sector, cap in sector_caps.items():
                sector_indices = [i for i, symbol in enumerate(self.symbols) 
                                if sector_mapping.get(symbol) == sector]
                if sector_indices:
                    sector_value = cp.sum([cp.multiply(self.prices[i], x[i]) for i in sector_indices])
                    constraints.append(sector_value <= cap * budget)
        
        # Minimum names constraint
        if min_names and min_names > 0:
            L = np.ones(n)  # Minimum lot size (1 share)
            U = max_shares_by_cap  # Maximum shares per stock
            
            constraints.extend([
                cp.sum(z) >= min_names,
                cp.multiply(L, z) <= x,
                x <= cp.multiply(U, z)
            ])
        
        # Absolute deviation constraints
        target_values = target_weights * budget
        for i in range(n):
            constraints.extend([
                t[i] >= cp.multiply(self.prices[i], x[i]) - target_values[i],
                t[i] >= target_values[i] - cp.multiply(self.prices[i], x[i])
            ])
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.GUROBI, verbose=False)
        except:
            try:
                problem.solve(solver=cp.CBC, verbose=False)
            except:
                problem.solve(solver=cp.GLPK_MI, verbose=False)
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"MILP failed with status: {problem.status}. Falling back to greedy.")
            return self.allocate_shares_greedy(target_weights, budget, individual_caps, 
                                             sector_caps, sector_mapping)
        
        # Extract solution
        shares = np.round(x.value).astype(int)
        final_invested = np.sum(shares * self.prices)
        remaining_cash = budget - final_invested
        executed_weights = (shares * self.prices) / budget
        annual_income = np.sum(self.forward_dividends * shares)
        portfolio_yield = annual_income / budget  # Yield on total budget
        yield_on_invested = annual_income / final_invested if final_invested > 0 else 0.0  # Yield on invested amount
        drift_l1 = np.sum(np.abs(executed_weights - target_weights))
        
        # Post-round risk check (normalize to invested weights for accurate calculation)
        invested = float(np.sum(shares * self.prices))
        w_exec = (shares * self.prices) / invested if invested > 0 else np.zeros_like(self.prices)
        post_vol = float(np.sqrt(w_exec @ self.covariance_matrix @ w_exec))
        
        print(f"MILP solution:")
        print(f"  Shares bought: {np.sum(shares)} total shares")
        print(f"  Amount invested: ₹{final_invested:,.0f}")
        print(f"  Residual cash: ₹{remaining_cash:,.0f} ({remaining_cash/budget:.2%})")
        print(f"  Portfolio yield (on budget): {portfolio_yield:.4%}")
        print(f"  Yield on invested amount: {yield_on_invested:.4%}")
        print(f"  Post-round volatility: {post_vol:.4%}")
        print(f"  L1 drift from target: {drift_l1:.4%}")
        
        return PortfolioResult(
            target_weights=target_weights,
            shares=shares,
            executed_weights=executed_weights,
            annual_income=annual_income,
            portfolio_yield=portfolio_yield,
            residual_cash=remaining_cash,
            drift_l1=drift_l1,
            allocation_method="MILP (income-first, drift-penalized)"
        )
