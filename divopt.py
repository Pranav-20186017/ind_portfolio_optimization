# divopt.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cvxpy as cp
import yfinance as yf


@dataclass
class EntropyYieldConfig:
    price_lookback_days: int = 756          # ~3y (for covariance)
    yield_lookback_days: int = 365          # TTM dividends
    entropy_weight: float = 0.05            # λ for entropy term
    min_weight_floor: Optional[float] = None
    vol_cap: Optional[float] = None         # annualized, e.g. 0.25
    use_median_ttm: bool = False            # stabilize TTM
    drop_na_dividends_to_zero: bool = True  # tickers with no divs → 0


@dataclass
class EntropyYieldResult:
    weights: Dict[str, float]
    portfolio_yield: float
    entropy: float
    effective_n: float
    realized_variance: float
    per_ticker_yield: Dict[str, float]
    last_close: Dict[str, float]
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    
    def calculate_shares(self, budget: float) -> Dict[str, int]:
        """Calculate integer number of shares for a given budget."""
        shares = {}
        remaining_budget = budget
        
        # Sort by weight descending to allocate to largest positions first
        sorted_stocks = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        
        for ticker, weight in sorted_stocks:
            if ticker in self.last_close and self.last_close[ticker] > 0:
                price = self.last_close[ticker]
                target_value = budget * weight
                target_shares = target_value / price
                
                # Floor to get integer shares
                actual_shares = int(target_shares)
                
                # Check if we can afford these shares
                cost = actual_shares * price
                if cost <= remaining_budget:
                    shares[ticker] = actual_shares
                    remaining_budget -= cost
                else:
                    # Buy as many shares as we can afford
                    affordable_shares = int(remaining_budget / price)
                    if affordable_shares > 0:
                        shares[ticker] = affordable_shares
                        remaining_budget -= affordable_shares * price
            else:
                shares[ticker] = 0
                
        return shares
    
    def calculate_invested_amount(self, shares: Dict[str, int]) -> float:
        """Calculate total amount invested based on share allocation."""
        total = 0.0
        for ticker, num_shares in shares.items():
            if ticker in self.last_close and self.last_close[ticker] > 0:
                total += num_shares * self.last_close[ticker]
        return total


class EntropyYieldOptimizer:
    """
    Maximize:  Y^T w + λ * H(w)   where H(w) = -∑ w_i log w_i
    s.t.      ∑ w = 1, w_i ≥ ε (>0), optional wᵀΣw ≤ σ_max²
    Uses ONLY 'Close' (no Adj Close).
    """

    def __init__(self, tickers: List[str], config: Optional[EntropyYieldConfig] = None):
        if len(tickers) < 2:
            raise ValueError("Need at least 2 tickers for diversification.")
        self.tickers = tickers
        self.cfg = config or EntropyYieldConfig()

    def run(self) -> EntropyYieldResult:
        prices = self._download_close_matrix(self.tickers, self.cfg.price_lookback_days)
        prices = prices.dropna(how="any")
        if prices.empty or prices.shape[1] != len(self.tickers):
            raise ValueError("No overlapping 'Close' history across all tickers.")

        # Ensure index is timezone-naive
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        
        ttm_yield, last_close = self._compute_ttm_yields(prices.index[-1])

        rets = prices.pct_change().dropna()
        cov_ann = rets.cov() * 252.0

        w = self._solve_ye(prices.columns.tolist(), ttm_yield, cov_ann)
        weights = {t: float(w[i]) for i, t in enumerate(prices.columns)}
        weights = self._normalize(weights)

        y_vec = np.array([ttm_yield[t] for t in prices.columns])
        w_vec = np.array([weights[t] for t in prices.columns])
        port_yield = float(y_vec @ w_vec)

        eps = 1e-16
        ent = float(-np.sum(w_vec * np.log(np.clip(w_vec, eps, 1.0))))
        eff_n = float(np.exp(ent))
        realized_var = float(w_vec @ cov_ann.values @ w_vec)

        return EntropyYieldResult(
            weights=weights,
            portfolio_yield=port_yield,
            entropy=ent,
            effective_n=eff_n,
            realized_variance=realized_var,
            per_ticker_yield={t: float(ttm_yield[t]) for t in prices.columns},
            last_close={t: float(last_close[t]) for t in prices.columns},
            start_date=prices.index[0],
            end_date=prices.index[-1],
        )

    # ---------- internals ----------
    def _download_close_matrix(self, tickers: List[str], lookback_days: int) -> pd.DataFrame:
        end = pd.Timestamp.today().normalize()
        # Ensure timezone-naive
        if end.tz is not None:
            end = end.tz_localize(None)
        
        start = end - pd.Timedelta(days=int(lookback_days * 1.4))  # pad for holidays
        
        df = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=False,             # IMPORTANT: use raw 'Close'
            progress=False,
            group_by="column",
            multi_level_index=False
        )

        if isinstance(df, pd.DataFrame) and "Close" in df.columns:
            closes = df["Close"].copy()
            if isinstance(closes, pd.Series):
                closes = closes.to_frame(name=tickers[0])
            closes = closes.loc[:, [c for c in closes.columns if c in tickers]].astype(float)
            return closes
        elif isinstance(df, pd.Series):
            return df.to_frame(name=tickers[0]).astype(float)
        return pd.DataFrame()

    def _compute_ttm_yields(self, ref_date: pd.Timestamp) -> Tuple[pd.Series, pd.Series]:
        # Ensure ref_date is timezone-naive
        if ref_date.tz is not None:
            ref_date = ref_date.tz_localize(None)
        
        start_div = (ref_date - pd.Timedelta(days=365*5))
        end_div = ref_date

        last_prices = yf.download(
            self.tickers,
            start=(end_div - pd.Timedelta(days=14)),
            end=end_div + pd.Timedelta(days=1),
            auto_adjust=False,
            progress=False,
            group_by="column",
            multi_level_index=False
        )["Close"]
        if isinstance(last_prices, pd.Series):
            last_prices = last_prices.to_frame(name=self.tickers[0])
        last_prices = last_prices.dropna(how="all").ffill().bfill()

        last_close, ttm_divs = {}, {}
        for t in self.tickers:
            px = float(last_prices[t].iloc[-1]) if t in last_prices.columns else np.nan
            last_close[t] = px
            try:
                div = yf.Ticker(t).dividends
            except Exception:
                div = pd.Series(dtype=float)
            if div is None or div.empty:
                ttm = 0.0 if self.cfg.drop_na_dividends_to_zero else np.nan
            else:
                div = div.sort_index()
                # Ensure dividend index is timezone-naive for comparison
                if div.index.tz is not None:
                    div.index = div.index.tz_localize(None)
                div = div[(div.index >= start_div) & (div.index <= end_div)]
                if div.empty:
                    ttm = 0.0
                else:
                    daily = div.resample("D").sum().reindex(
                        pd.date_range(start_div, end_div, freq="D"), fill_value=0.0
                    )
                    ttm_series = daily.rolling(self.cfg.yield_lookback_days, min_periods=1).sum()
                    if self.cfg.use_median_ttm:
                        tail = ttm_series.tail(90)
                        ttm = float(tail.median() if not tail.empty else ttm_series.iloc[-1])
                    else:
                        ttm = float(ttm_series.iloc[-1])
            ttm_divs[t] = ttm

        yld = {}
        for t in self.tickers:
            px = last_close[t]
            ttm = ttm_divs[t]
            y = 0.0
            if px and px > 0 and (ttm is not None) and not math.isnan(ttm):
                y = float(ttm / px)
            yld[t] = y
        return pd.Series(yld), pd.Series(last_close)

    def _solve_ye(self, ordered_tickers: List[str], y_ttm: pd.Series, cov_ann: pd.DataFrame) -> np.ndarray:
        n = len(ordered_tickers)
        y = np.array([y_ttm[t] for t in ordered_tickers], dtype=float)
        eps = self.cfg.min_weight_floor if self.cfg.min_weight_floor is not None else 1.0 / (1000.0 * n)
        eps = float(min(eps, 0.5 / n))  # ensure feasibility

        w = cp.Variable(n)
        cons = [cp.sum(w) == 1.0, w >= eps]

        if self.cfg.vol_cap and self.cfg.vol_cap > 0:
            sigma2_max = float(self.cfg.vol_cap ** 2)
            Σ = cov_ann.values
            Σ = (Σ + Σ.T) / 2.0
            cons.append(cp.quad_form(w, Σ) <= sigma2_max)

        entropy_term = -cp.sum(cp.entr(w))  # H(w)
        obj = cp.Maximize(y @ w + self.cfg.entropy_weight * entropy_term)
        prob = cp.Problem(obj, cons)

        for solver in [cp.MOSEK, cp.CLARABEL, cp.ECOS, cp.OSQP, cp.SCS]:
            try:
                prob.solve(solver=solver, verbose=False)
                if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None:
                    wv = np.array(w.value, dtype=float)
                    wv = np.maximum(wv, eps)
                    return wv / wv.sum()
            except Exception:
                continue

        return np.full(n, 1.0 / n, dtype=float)

    @staticmethod
    def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
        s = sum(max(0.0, float(x)) for x in weights.values())
        if s <= 0:
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        return {k: float(max(0.0, v) / s) for k, v in weights.items()}
