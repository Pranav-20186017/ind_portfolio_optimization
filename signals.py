# signals.py

from typing import List, Dict
import pandas as pd
import numpy as np
import talib

# ──────────────────────────────────────────────────────────────────────────────
# 1) MOVING AVERAGES
# ──────────────────────────────────────────────────────────────────────────────
def sma(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    return prices.rolling(window=n).mean()

def ema(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    return prices.ewm(span=n, adjust=False).mean()

def wma(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Weighted MA: weights = 1,2,3,...,n
    """
    def _wma_func(x):
        weights = np.arange(1, len(x) + 1)
        return np.dot(weights, x) / weights.sum()

    return prices.rolling(window=n).apply(_wma_func, raw=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2) OSCILLATORS / MOMENTUM INDICATORS
# ──────────────────────────────────────────────────────────────────────────────
def compute_rsi(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    df = prices.copy()
    for col in df.columns:
        df[col] = talib.RSI(df[col].values, timeperiod=n)
    return df

def compute_willr(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int
) -> pd.DataFrame:
    df = close.copy()
    for col in df.columns:
        df[col] = talib.WILLR(
            high[col].values, low[col].values, close[col].values, timeperiod=n
        )
    return df

def compute_cci(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int
) -> pd.DataFrame:
    df = close.copy()
    for col in df.columns:
        df[col] = talib.CCI(
            high[col].values, low[col].values, close[col].values, timeperiod=n
        )
    return df

def compute_roc(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    # Simple rate of change: (P_t / P_{t-n}) - 1
    return prices.pct_change(periods=n)

# ──────────────────────────────────────────────────────────────────────────────
# 3) VOLATILITY / CHANNEL INDICATORS
# ──────────────────────────────────────────────────────────────────────────────
def compute_atr(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int
) -> pd.DataFrame:
    df = close.copy()
    for col in df.columns:
        # Explicitly convert to np.float64
        high_values = high[col].values.astype(np.float64)
        low_values = low[col].values.astype(np.float64)
        close_values = close[col].values.astype(np.float64)
        df[col] = talib.ATR(high_values, low_values, close_values, timeperiod=n)
    return df

def compute_supertrend(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    atr_period: int,
    multiplier: float = 3.0
) -> pd.DataFrame:
    """
    A minimal Supertrend: +1 if price > upper_band from previous period, -1 if price < lower_band.
    We compute "true range" → ATR → upper/lower bands → final band rules → trend (±1).
    """
    # 1) True Range (TR) & ATR
    tr = pd.DataFrame(index=close.index, columns=close.columns)
    for col in close.columns:
        tr[col] = talib.TRANGE(
            high[col].values, low[col].values, close[col].shift(1).values
        )
    atr = tr.rolling(window=atr_period).mean()

    # 2) Basic Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    # 3) Final Bands (carry‐forward logic)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    for col in close.columns:
        for t in range(1, len(close)):
            # final_upper
            if close[col].iat[t - 1] <= final_upper[col].iat[t - 1]:
                final_upper[col].iat[t] = min(
                    upper_band[col].iat[t], final_upper[col].iat[t - 1]
                )
            else:
                final_upper[col].iat[t] = upper_band[col].iat[t]
            # final_lower
            if close[col].iat[t - 1] >= final_lower[col].iat[t - 1]:
                final_lower[col].iat[t] = max(
                    lower_band[col].iat[t], final_lower[col].iat[t - 1]
                )
            else:
                final_lower[col].iat[t] = lower_band[col].iat[t]

    # 4) Supertrend signal: ±1
    st = pd.DataFrame(index=close.index, columns=close.columns)
    st.iloc[0, :] = -1
    for col in close.columns:
        for t in range(1, len(close)):
            if close[col].iat[t] > final_upper[col].iat[t - 1]:
                st[col].iat[t] = 1
            elif close[col].iat[t] < final_lower[col].iat[t - 1]:
                st[col].iat[t] = -1
            else:
                st[col].iat[t] = st[col].iat[t - 1]
    return st.fillna(method="ffill")

# ──────────────────────────────────────────────────────────────────────────────
# 4) VOLUME-BASED INDICATORS (no rolling-window)
# ──────────────────────────────────────────────────────────────────────────────
def compute_obv(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    for col in df.columns:
        # Explicitly convert to np.float64 to avoid "input array type is not double" error
        price_values = prices[col].values.astype(np.float64)
        volume_values = volume[col].values.astype(np.float64)
        df[col] = talib.OBV(price_values, volume_values)
    return df

def compute_ad(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame
) -> pd.DataFrame:
    df = close.copy()
    for col in df.columns:
        # Explicitly convert to np.float64 to avoid "input array type is not double" error
        high_values = high[col].values.astype(np.float64)
        low_values = low[col].values.astype(np.float64)
        close_values = close[col].values.astype(np.float64)
        volume_values = volume[col].values.astype(np.float64)
        df[col] = talib.AD(high_values, low_values, close_values, volume_values)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 5) CROSS-SECTIONAL Z-SCORING
# ──────────────────────────────────────────────────────────────────────────────
def zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each date (row), subtract row‐mean and divide by row‐std, producing cross-sectional z-scores.
    """
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

# ──────────────────────────────────────────────────────────────────────────────
# 6) BUILD & BLEND TECHNICAL SCORES
# ──────────────────────────────────────────────────────────────────────────────
def build_technical_scores(
    prices: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    volume: pd.DataFrame,
    indicator_cfgs: List[Dict],
    blend: str = "equal",           # or accept a dict of custom weights
) -> pd.Series:
    """
    1) For each cfg in indicator_cfgs ({"name":..., "window":..., "mult":...}), compute raw indicator DataFrame.
    2) Transform raw → cross-sectional z-score (take only last row).
    3) Stack all z-Series into one DataFrame (n_assets × K) → blend equally (or with custom weights).
    4) Return a pd.Series of length n_assets: the final composite score for each ticker.
    """
    z_scores: Dict[str, pd.Series] = {}

    for cfg in indicator_cfgs:
        name = cfg["name"].upper()
        n = int(cfg.get("window", 0) or 0)

        if name == "SMA":
            raw = sma(prices, n)
            gap = (prices - raw) / prices
            z = zscore_cross_section(gap).iloc[-1]

        elif name == "EMA":
            raw = ema(prices, n)
            gap = (prices - raw) / prices
            z = zscore_cross_section(gap).iloc[-1]

        elif name == "WMA":
            raw = wma(prices, n)
            gap = (prices - raw) / prices
            z = zscore_cross_section(gap).iloc[-1]

        elif name == "RSI":
            raw = compute_rsi(prices, n)  # RSI in [0..100]
            # Convert to "higher = more bullish": 50 − (RSI − 50)/50
            transformed = 50 - (raw - 50) / 50
            z = zscore_cross_section(transformed).iloc[-1]

        elif name == "WILLR":
            raw = compute_willr(highs, lows, prices, n)  # WillR ∈ [−100..0]
            # Convert to [0..1], higher = bullish
            transformed = -raw / 100
            z = zscore_cross_section(transformed).iloc[-1]

        elif name == "CCI":
            raw = compute_cci(highs, lows, prices, n)  # CCI can be positive/negative
            z = zscore_cross_section(raw).iloc[-1]

        elif name == "ROC":
            raw = compute_roc(prices, n)  # e.g. 1-day, 12-day, 20-day percent change
            z = zscore_cross_section(raw).iloc[-1]

        elif name == "ATR":
            raw = compute_atr(highs, lows, prices, n)  # positive volatility
            # Higher ATR = more volatile = less bullish → negate
            transformed = -raw
            z = zscore_cross_section(transformed).iloc[-1]

        elif name == "SUPERTREND":
            atr_n = int(cfg.get("window", 10))
            mult = float(cfg.get("mult", 3.0))
            raw = compute_supertrend(highs, lows, prices, atr_n, mult)  # ±1
            z = zscore_cross_section(raw).iloc[-1]

        elif name == "BBANDS":
            # talib.BBANDS returns (upper, mid, lower) numpy arrays
            # Convert to float64 to avoid type errors
            prices_float64 = prices.values.astype(np.float64)
            ub, mb, lb = talib.BBANDS(
                prices_float64, timeperiod=n, nbdevup=2, nbdevdn=2
            )
            df_mb = pd.DataFrame(mb, index=prices.index, columns=prices.columns)
            gap = (prices - df_mb) / df_mb
            z = zscore_cross_section(gap).iloc[-1]

        elif name == "OBV":
            raw = compute_obv(prices, volume)
            # Use 5-day change in OBV as a momentum proxy
            obv_roc = raw.pct_change(periods=5)
            z = zscore_cross_section(obv_roc).iloc[-1]

        elif name == "AD":
            raw = compute_ad(highs, lows, prices, volume)
            ad_roc = raw.pct_change(periods=5)
            z = zscore_cross_section(ad_roc).iloc[-1]

        else:
            raise ValueError(f"Unsupported indicator: {name}")

        z_scores[name + f"_{n}"] = z  # e.g. "EMA_50" → Series(index=tickers)

    # Stack all z-Series into a DataFrame (n_assets × K_indicators)
    z_df = pd.DataFrame(z_scores)

    # Blend them
    if blend == "equal":
        alpha = pd.Series(1.0 / len(z_df.columns), index=z_df.columns)
    else:
        # assume blend is a dict e.g. {"EMA_50":0.4,"RSI_14":0.6,...}
        alpha = pd.Series(blend)

    S = (z_df * alpha).sum(axis=1)  # final composite score per ticker
    return S  # pd.Series indexed by ticker

# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC: Which indicators (and which default windows) we support. Used for validation.
# ──────────────────────────────────────────────────────────────────────────────
TECHNICAL_INDICATORS = {
    "SMA":        [200, 100, 50, 20, 10],
    "EMA":        [200, 100, 50, 20, 10],
    "WMA":        [50, 20],
    "RSI":        [14, 9, 21],
    "WILLR":      [14, 9, 21],
    "CCI":        [20, 14, 50],
    "ROC":        [12, 20, 25],
    "ATR":        [14, 7, 21],
    "SUPERTREND": [10, 7, 14],    # multiplier defaults to 3.0 unless overridden
    "BBANDS":     [20, 10, 50],
    "OBV":        [0],   # no window needed
    "AD":         [0],   # no window needed
} 