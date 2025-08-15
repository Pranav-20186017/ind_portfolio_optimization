# dividend_optimizer.py
from __future__ import annotations
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel

from data import StockItem, ExchangeEnum, APIError, ErrorCode
from divopt import EntropyYieldConfig, EntropyYieldOptimizer


class DividendOptRequest(BaseModel):
    stocks: List[StockItem]
    entropy_weight: float = 0.05
    price_lookback_days: int = 756
    yield_lookback_days: int = 365
    min_weight_floor: Optional[float] = None
    vol_cap: Optional[float] = None
    use_median_ttm: bool = False


class DividendOptResponse(BaseModel):
    weights: Dict[str, float]
    portfolio_yield: float
    entropy: float
    effective_n: float
    realized_variance: float
    per_ticker_yield: Dict[str, float]
    last_close: Dict[str, float]
    start_date: datetime
    end_date: datetime


def _format_tickers(stocks: List[StockItem]) -> List[str]:
    tickers = []
    for s in stocks:
        if s.exchange == ExchangeEnum.BSE:
            tickers.append(s.ticker + ".BO")
        elif s.exchange == ExchangeEnum.NSE:
            tickers.append(s.ticker + ".NS")
        else:
            raise APIError(
                code=ErrorCode.INVALID_TICKER,
                message=f"Unknown exchange for {s.ticker}",
                status_code=422
            )
    return tickers


def optimize_dividend_portfolio(req: DividendOptRequest) -> DividendOptResponse:
    if not req.stocks or len(req.stocks) < 2:
        raise APIError(
            code=ErrorCode.INSUFFICIENT_STOCKS,
            message="Please provide at least 2 stocks",
            status_code=422
        )

    tickers = _format_tickers(req.stocks)

    cfg = EntropyYieldConfig(
        price_lookback_days=req.price_lookback_days,
        yield_lookback_days=req.yield_lookback_days,
        entropy_weight=req.entropy_weight,
        min_weight_floor=req.min_weight_floor,
        vol_cap=req.vol_cap,
        use_median_ttm=req.use_median_ttm,
    )

    opt = EntropyYieldOptimizer(tickers, cfg)
    res = opt.run()

    return DividendOptResponse(
        weights=res.weights,
        portfolio_yield=res.portfolio_yield,
        entropy=res.entropy,
        effective_n=res.effective_n,
        realized_variance=res.realized_variance,
        per_ticker_yield=res.per_ticker_yield,
        last_close=res.last_close,
        start_date=res.start_date.to_pydatetime(),
        end_date=res.end_date.to_pydatetime(),
    )
