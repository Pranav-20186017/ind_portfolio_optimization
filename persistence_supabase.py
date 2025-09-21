from __future__ import annotations
import os, io, base64, logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from hashlib import sha256

import pyarrow as pa, pyarrow.parquet as pq
from supabase import create_client

# Optional Supabase binding: disable persistence if env vars are missing (CI/tests)
SB_URL  = os.getenv("SUPABASE_URL")
SB_KEY  = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET  = os.getenv("SUPABASE_BUCKET", "runs")
sb = create_client(SB_URL, SB_KEY) if (SB_URL and SB_KEY) else None
logger = logging.getLogger(__name__)

def _sha(b: bytes) -> str: return sha256(b).hexdigest()

def _upload_bytes(run_id: str, name: str, data: bytes, content_type: str, kind: str):
    if sb is None:
        return
    # Store inside the bucket under {run_id}/... (do not prefix with bucket name)
    path = f"{run_id}/{name}"
    try:
        sb.storage.from_(BUCKET).upload(
            path,
            data,
            file_options={
                "contentType": content_type,
                "upsert": "true",
            },
        )
    except Exception as e:
        logger.exception("Storage upload failed for %s: %s", path, str(e))
        raise
    try:
        sb.table("run_artifacts").insert({
            "run_id": run_id, "kind": kind, "path": path,
            "content_type": content_type, "bytes": len(data), "sha256": _sha(data)
        }).execute()
    except Exception as e:
        logger.exception("run_artifacts insert failed for %s: %s", path, str(e))

def _upload_parquet(run_id: str, name: str, table: pa.Table, kind: str):
    buf = io.BytesIO(); pq.write_table(table, buf)
    _upload_bytes(run_id, f"{name}.parquet", buf.getvalue(), "application/x-parquet", kind)

def persist_all(
    *,
    run_id: str,
    endpoint: str,
    request_json: Dict[str, Any],
    response_json: Dict[str, Any],

    # maps: method -> ...
    method_weights: Dict[str, Dict[str, float]] | None = None,
    method_metrics: Dict[str, Dict[str, float]] | None = None,
    method_plots_b64: Dict[str, Dict[str, str]] | None = None,  # {"MVO":{"returns_dist.png":"...","max_drawdown.png":"..."}}

    # response-level artifacts
    covariance_heatmap_b64: Optional[str] = None,

    # optional Parquet frames
    cumulative_returns_df=None,   # pd.DataFrame: ["dt","method","cumret"]
    benchmark_returns_df=None,    # pd.DataFrame: ["dt","benchmark","ret"]
    stock_yearly_returns_df=None, # pd.DataFrame: ["ticker","year","ret"]
    returns_panel_df=None,        # optional full panel (wide or long)
    cov_matrix=None               # optional numpy.ndarray
):
    if sb is None:
        return
    # 1) parent row (snapshot for easy history UI)
    sb.table("runs").upsert({
        "run_id": run_id,
        "endpoint": endpoint,
        "request_json": request_json,
        "response_json": response_json,
        "note": "bg persist v1"
    }).execute()

    # 2) per-method weights
    if method_weights:
        rows = []
        for method, wmap in method_weights.items():
            for t, w in (wmap or {}).items():
                rows.append({"run_id": run_id, "ticker": t, "weight": float(w), "method": method})
        if rows:
            try:
                sb.table("run_weights").insert(rows).execute()
            except Exception as e:
                # Fallback if schema lacks 'method' column
                if "method" in str(e):
                    rows_nomethod = [{k: v for k, v in r.items() if k != "method"} for r in rows]
                    sb.table("run_weights").insert(rows_nomethod).execute()
                else:
                    raise

    # 3) per-method metrics (flatten entire PortfolioPerformance)
    if method_metrics:
        rows = []
        for method, m in method_metrics.items():
            for k, v in (m or {}).items():
                rows.append({
                    "run_id": run_id,
                    "metric_name": k,
                    "metric_value": (None if v is None else float(v)),
                    "method": method
                })
        if rows:
            try:
                sb.table("run_metrics").insert(rows).execute()
            except Exception as e:
                if "method" in str(e):
                    rows_nomethod = [{k: v for k, v in r.items() if k != "method"} for r in rows]
                    sb.table("run_metrics").insert(rows_nomethod).execute()
                else:
                    raise

    # 4) per-method plots (base64 → PNG)
    if method_plots_b64:
        for method, imgs in method_plots_b64.items():
            for fname, b64 in (imgs or {}).items():
                try:
                    raw = base64.b64decode(b64)
                    _upload_bytes(run_id, f"{method}_{fname}", raw, "image/png", "image")
                except Exception:
                    pass

    # 5) response-level covariance heatmap
    if covariance_heatmap_b64:
        try:
            raw = base64.b64decode(covariance_heatmap_b64)
            _upload_bytes(run_id, "covariance_heatmap.png", raw, "image/png", "image")
        except Exception:
            pass

    # 6) optional Parquet artifacts
    try:
        import pandas as pd
        if cumulative_returns_df is not None and isinstance(cumulative_returns_df, pd.DataFrame) and not cumulative_returns_df.empty:
            _upload_parquet(run_id, "cumulative_returns", pa.Table.from_pandas(cumulative_returns_df), "returns_panel")
        if benchmark_returns_df is not None and isinstance(benchmark_returns_df, pd.DataFrame) and not benchmark_returns_df.empty:
            _upload_parquet(run_id, "benchmark_returns", pa.Table.from_pandas(benchmark_returns_df), "returns_panel")
        if stock_yearly_returns_df is not None and isinstance(stock_yearly_returns_df, pd.DataFrame) and not stock_yearly_returns_df.empty:
            _upload_parquet(run_id, "stock_yearly_returns", pa.Table.from_pandas(stock_yearly_returns_df), "returns_panel")
        if returns_panel_df is not None and isinstance(returns_panel_df, pd.DataFrame) and not returns_panel_df.empty:
            _upload_parquet(run_id, "returns_panel", pa.Table.from_pandas(returns_panel_df.reset_index()), "returns_panel")
    except Exception:
        pass

    # 7) optional dense covariance matrix
    try:
        import numpy as np
        if cov_matrix is not None and isinstance(cov_matrix, np.ndarray) and cov_matrix.size:
            n = cov_matrix.shape[0]
            tbl = pa.table({
                "i": [i for i in range(n) for _ in range(n)],
                "j": [j for _ in range(n) for j in range(n)],
                "value": cov_matrix.ravel().tolist(),
            })
            _upload_parquet(run_id, "covariance_matrix", tbl, "covariance")
    except Exception:
        pass

    # 8) finished marker (write an actual UTC timestamp)
    sb.table("runs").update({
        "finished_at": datetime.now(timezone.utc).isoformat()
    }).eq("run_id", run_id).execute()


