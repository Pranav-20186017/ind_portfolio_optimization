# -*- coding: utf-8 -*-
"""
Standalone real-data unittest for MinCDaR.

- Uses your project's srv.py (must be importable from current working dir).
- Pulls live prices via yfinance through fetch_and_align_data.
- Reproduces the failing 5-ticker short-window case and a 4-ticker full-history case.
- No SystemExit tricks; standard unittest with clear prints.

Run:
  python -u test_mincdar_realdata.py -v
or
  python -m unittest -v test_mincdar_realdata.py
"""
import os
import sys
import traceback
import unittest
from datetime import datetime
from typing import List, Optional

# Ensure local project root is importable
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import project functions
try:
    from srv import (
        fetch_and_align_data,
        run_optimization_MIN_CDAR,
        get_risk_free_rate,
    )
    from pypfopt import expected_returns
except Exception as e:
    print("ERROR: Could not import from srv.py. Make sure this file is run in your project root.")
    traceback.print_exc()
    raise


def _print_header(title: str) -> None:
    print("\n" + "=" * 100, flush=True)
    print(title, flush=True)
    print("=" * 100, flush=True)


def _run_mincdar_case(
    tickers: List[str],
    label: str,
    benchmark: str = "^NSEI",
    lookback_trading_days: Optional[int] = None,
):
    """
    Helper that fetches real data and runs MinCDaR once.
    Returns (result, cum_returns, df_prices.index[0], df_prices.index[-1]).
    Prints a compact report.
    """
    _print_header(f"CASE: {label}")
    print(f"Tickers: {tickers}", flush=True)
    print(f"Benchmark: {benchmark}", flush=True)
    print(f"Lookback: {lookback_trading_days or 'FULL'} trading days", flush=True)

    print("Step 1/6: Fetching & aligning prices (real yfinance calls via your pipeline)...", flush=True)
    df_prices, bench_close = fetch_and_align_data(
        tickers=tickers,
        benchmark_ticker=benchmark,
        sanitize_bse=True,
        start_date=datetime(1990, 1, 1),
    )
    print(f"... fetched shapes -> prices={df_prices.shape}, bench={bench_close.shape}", flush=True)

    if df_prices.empty or bench_close.empty:
        raise RuntimeError("Empty price or benchmark series; cannot run MinCDaR on real data.")

    if lookback_trading_days is not None and lookback_trading_days > 0:
        df_prices = df_prices.iloc[-lookback_trading_days:]
        bench_close = bench_close.loc[bench_close.index.intersection(df_prices.index)]
        print(f"... sliced -> prices={df_prices.shape}, bench={bench_close.shape}", flush=True)

    start_dt = df_prices.index[0].to_pydatetime()
    end_dt = df_prices.index[-1].to_pydatetime()

    print("Step 2/6: Computing risk-free rate for this exact window...", flush=True)
    rf = get_risk_free_rate(start_dt, end_dt)
    print(f"... RF (annualized): {rf:.6f}", flush=True)

    print("Step 3/6: Computing mu & returns...", flush=True)
    mu = expected_returns.mean_historical_return(df_prices, frequency=252)
    returns = df_prices.pct_change().dropna()
    print(f"... mu len={len(mu)}, returns shape={returns.shape}", flush=True)

    if returns.empty:
        raise RuntimeError("Empty returns after pct_change().")

    print("Step 4/6: Running MinCDaR...", flush=True)
    result, cum = run_optimization_MIN_CDAR(mu, returns, bench_close, risk_free_rate=rf)

    if result is None or cum is None or len(cum) == 0:
        raise RuntimeError("MinCDaR returned no result/cumulative returns.")

    print("Step 5/6: Weights (non-zeros)...", flush=True)
    wts = result.weights
    nz = {k: v for k, v in wts.items() if abs(v) > 1e-8}
    for k, v in nz.items():
        print(f"  {k:>12s}: {v:7.4f}", flush=True)
    print(f"... sum(weights) = {sum(wts.values()):.6f}", flush=True)

    print("Step 6/6: Performance (annualized)...", flush=True)
    perf = result.performance
    print(f"  Window: {start_dt.date()} -> {end_dt.date()}  (N={len(df_prices)})", flush=True)
    print(f"  Expected Return: {perf.expected_return:.4%}", flush=True)
    print(f"  Volatility     : {perf.volatility:.4%}", flush=True)
    print(f"  Sharpe         : {perf.sharpe:.4f}", flush=True)
    print(f"  Max Drawdown   : {perf.max_drawdown:.2%}", flush=True)
    if hasattr(perf, "cdar_95"):
        print(f"  CDaR 95%       : {perf.cdar_95:.2%}", flush=True)

    return result, cum, start_dt, end_dt


class TestMinCDaRRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\nEnvironment:")
        print("  Python   :", sys.version.replace("\n", " "))
        print("  CWD      :", os.getcwd())
        print("  TESTING  :", os.environ.get("TESTING"))
        print("  MOSEKLM_LICENSE_FILE:", os.environ.get("MOSEKLM_LICENSE_FILE"))
        print("-" * 100, flush=True)

    def test_mincdar_short_window_five_tickers(self):
        """
        Reproduce the failing case from logs:
        ['SUPRIYA.NS', 'POLYMED.NS', 'RKFORGE.NS', 'DODLA.NS', 'ETERNAL.NS']
        with ~104-110 trading-day window.
        """
        tickers = ["SUPRIYA.NS", "POLYMED.NS", "RKFORGE.NS", "DODLA.NS", "ETERNAL.NS"]
        try:
            result, cum, start_dt, end_dt = _run_mincdar_case(
                tickers,
                label="MinCDaR — 5 tickers, ~110 trading days",
                lookback_trading_days=110,
            )
        except Exception as e:
            self.fail(f"MinCDaR (short-window 5 tickers) raised an exception:\n{traceback.format_exc()}")

        # Basic sanity checks (do NOT abort the process; just fail the test if off)
        self.assertIsNotNone(result, "No result returned by MinCDaR")
        self.assertGreater(len(cum), 0, "Empty cumulative returns")
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, places=3, msg="Weights do not sum ~1")

    def test_mincdar_full_history_four_tickers(self):
        """
        Control case that succeeded in logs: 4 tickers with full available history.
        """
        tickers = ["DODLA.NS", "RKFORGE.NS", "SUPRIYA.NS", "POLYMED.NS"]
        try:
            result, cum, start_dt, end_dt = _run_mincdar_case(
                tickers,
                label="MinCDaR — 4 tickers, full history",
                lookback_trading_days=None,
            )
        except Exception as e:
            self.fail(f"MinCDaR (full-history 4 tickers) raised an exception:\n{traceback.format_exc()}")

        self.assertIsNotNone(result, "No result returned by MinCDaR")
        self.assertGreater(len(cum), 0, "Empty cumulative returns")
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, places=3, msg="Weights do not sum ~1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
