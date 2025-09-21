import os
import sys
import time
import uuid
from typing import Any, Dict

import numpy as np
import pandas as pd

from dotenv import load_dotenv
# Load environment variables from .env at project root BEFORE importing persistence module
load_dotenv()
from persistence_supabase import persist_all, sb, BUCKET


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        print(f"[ERROR] Missing required env var: {var}")
        sys.exit(1)
    return val


def _resp_request_samples() -> tuple[Dict[str, Any], Dict[str, Any]]:
    req = {
        "stocks": [
            {"ticker": "INFY", "exchange": "NSE"},
            {"ticker": "TCS", "exchange": "NSE"},
        ],
        "methods": ["MVO"],
        "benchmark": "nifty",
        "indicators": [],
    }
    resp = {
        "results": {
            "MVO": {
                "weights": {"INFY.NS": 0.6, "TCS.NS": 0.4},
                "performance": {
                    "expected_return": 0.15,
                    "volatility": 0.20,
                    "sharpe": 0.75,
                    "sortino": 0.90,
                    "max_drawdown": -0.25,
                    "romad": 0.4,
                    "var_95": -0.03,
                    "cvar_95": -0.05,
                    "var_90": -0.02,
                    "cvar_90": -0.03,
                    "cagr": 0.12,
                    "portfolio_beta": 1.0,
                    "portfolio_alpha": 0.0,
                    "beta_pvalue": 1.0,
                    "r_squared": 0.2,
                    "blume_adjusted_beta": 1.0,
                    "treynor_ratio": 0.1,
                    "skewness": 0.0,
                    "kurtosis": 3.0,
                    "entropy": 1.0,
                    "omega_ratio": 1.1,
                    "calmar_ratio": 0.5,
                    "ulcer_index": 0.1,
                    "evar_95": 0.02,
                    "gini_mean_difference": 0.01,
                    "dar_95": 0.1,
                    "cdar_95": 0.12,
                    "upside_potential_ratio": 1.2,
                    "modigliani_risk_adjusted_performance": 0.08,
                    "information_ratio": 0.2,
                    "sterling_ratio": 0.3,
                    "v2_ratio": 0.4,
                },
                # small transparent 1x1 PNG
                "returns_dist": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=",
                "max_drawdown_plot": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=",
            }
        },
        "cumulative_returns": {"MVO": [1.0, 1.02, 1.05]},
        "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "benchmark_returns": [{"name": "nifty", "returns": [1.0, 1.01, 1.015]}],
        "stock_yearly_returns": {"INFY.NS": {"2024": 0.10}, "TCS.NS": {"2024": 0.08}},
        "covariance_heatmap": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=",
        "risk_free_rate": 0.05,
        "is_technical_only": False,
    }
    return req, resp


def main() -> int:
    # Ensure Supabase credentials are present
    _require_env("SUPABASE_URL")
    _require_env("SUPABASE_SERVICE_ROLE_KEY")
    os.environ.setdefault("SUPABASE_BUCKET", BUCKET)

    run_id = str(uuid.uuid4())
    endpoint = "/test/persist"

    req_json, resp_json = _resp_request_samples()

    # per-method maps
    method_weights = {"MVO": {"INFY.NS": 0.6, "TCS.NS": 0.4}}
    method_metrics = {"MVO": resp_json["results"]["MVO"]["performance"]}
    method_plots_b64 = {
        "MVO": {
            "returns_dist.png": resp_json["results"]["MVO"]["returns_dist"],
            "max_drawdown.png": resp_json["results"]["MVO"]["max_drawdown_plot"],
        }
    }

    # Optional parquet frames
    dt = pd.to_datetime(resp_json["dates"])  # type: ignore
    cum_rows = [
        {"dt": dt[i], "method": "MVO", "cumret": float(v)}
        for i, v in enumerate(resp_json["cumulative_returns"]["MVO"])  # type: ignore
    ]
    cumulative_returns_df = pd.DataFrame(cum_rows)

    bench_rows = [
        {"dt": dt[i], "benchmark": "nifty", "ret": float(v)}
        for i, v in enumerate(resp_json["benchmark_returns"][0]["returns"])  # type: ignore
    ]
    benchmark_returns_df = pd.DataFrame(bench_rows)

    yr_rows = []
    for tkr, yearmap in resp_json["stock_yearly_returns"].items():  # type: ignore
        for year, val in yearmap.items():
            yr_rows.append({"ticker": tkr, "year": int(year), "ret": float(val)})
    stock_yearly_returns_df = pd.DataFrame(yr_rows)

    # Simple covariance matrix
    cov_matrix = np.array([[0.01, 0.002], [0.002, 0.015]], dtype=float)

    # Call persistence
    print(f"[INFO] Persisting run_id={run_id} to Supabase bucket='{BUCKET}'")
    persist_all(
        run_id=run_id,
        endpoint=endpoint,
        request_json=req_json,
        response_json=resp_json,
        method_weights=method_weights,
        method_metrics=method_metrics,
        method_plots_b64=method_plots_b64,
        covariance_heatmap_b64=resp_json.get("covariance_heatmap"),
        cumulative_returns_df=cumulative_returns_df,
        benchmark_returns_df=benchmark_returns_df,
        stock_yearly_returns_df=stock_yearly_returns_df,
        cov_matrix=cov_matrix,
        returns_panel_df=None,
    )

    # Brief delay to allow Storage metadata to settle
    time.sleep(0.5)

    # Validate rows in relational tables
    def _get_data(resp):
        return getattr(resp, "data", None) if resp is not None else None

    runs = _get_data(sb.table("runs").select("run_id").eq("run_id", run_id).execute())
    # Select without 'method' to be compatible with older schemas
    weights = _get_data(sb.table("run_weights").select("ticker,weight").eq("run_id", run_id).execute())
    metrics = _get_data(sb.table("run_metrics").select("metric_name,metric_value").eq("run_id", run_id).execute())
    artifacts = None
    try:
        artifacts = _get_data(sb.table("run_artifacts").select("path,kind,content_type,bytes").eq("run_id", run_id).execute())
    except Exception as e:
        print("[WARN] run_artifacts query failed:", str(e))

    ok = True
    if not runs:
        print("[ERROR] No row found in 'runs' for run_id", run_id)
        ok = False
    if not weights:
        print("[ERROR] No rows found in 'run_weights'")
        ok = False
    if not metrics:
        print("[ERROR] No rows found in 'run_metrics'")
        ok = False
    if artifacts:
        # Print artifact summary
        print(f"[INFO] Stored {len(artifacts)} artifacts")
        for a in artifacts[:5]:
            print("  -", a.get("path"), a.get("content_type"), a.get("bytes"))
    else:
        print("[WARN] No rows found in 'run_artifacts' (table may be missing or insert disabled)")

    # Validate Storage listing
    prefix = f"{run_id}"
    try:
        objs = sb.storage.from_(BUCKET).list(prefix)
        if isinstance(objs, list) and len(objs) > 0:
            print(f"[INFO] Storage objects under '{prefix}': {len(objs)}")
        else:
            print(f"[ERROR] No objects listed under storage prefix '{prefix}'")
            ok = False
    except Exception as e:
        print("[ERROR] Storage list failed:", str(e))
        ok = False

    print("[RESULT]", "SUCCESS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())


