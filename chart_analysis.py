"""
Live chart payload: OHLCV + pattern overlays for the frontend chart.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from patterns import detect_all_patterns


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]).lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
    df.columns = [str(c).lower() for c in df.columns]
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns and col.capitalize() in df.columns:
            df[col] = df[col.capitalize()]
    return df


def _swing_points(
    high: pd.Series, low: pd.Series, order: int = 4
) -> Tuple[List[int], List[int]]:
    """Local maxima on high, local minima on low."""
    hi = high.values
    lo = low.values
    n = len(high)
    swing_hi: List[int] = []
    swing_lo: List[int] = []
    for i in range(order, n - order):
        if hi[i] >= np.nanmax(hi[i - order : i + order + 1]):
            swing_hi.append(i)
        if lo[i] <= np.nanmin(lo[i - order : i + order + 1]):
            swing_lo.append(i)
    return swing_hi, swing_lo


def build_chart_payload(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> Dict[str, Any]:
    """
    Fetch OHLCV and return JSON for Lightweight Charts + pattern analysis.
    """
    t = str(ticker).strip()
    if not t:
        return {"error": "missing ticker", "ticker": ticker}

    try:
        raw = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=True)
    except Exception as e:
        return {"error": str(e), "ticker": t}

    if raw is None or raw.empty or len(raw) < 30:
        return {"error": "not enough data", "ticker": t}

    data = _normalize_ohlc(raw)
    close = data["close"]
    high = data["high"] if "high" in data.columns else close
    low = data["low"] if "low" in data.columns else close

    support = float(close.rolling(20).min().iloc[-1]) if len(close) >= 20 else float(close.iloc[-1]) * 0.97
    resistance = float(close.rolling(20).max().iloc[-1]) if len(close) >= 20 else float(close.iloc[-1]) * 1.03

    merged, composite = detect_all_patterns(data, close, support, resistance)
    composite = float(max(-1.0, min(1.0, composite)))

    # Candles (time as YYYY-MM-DD for daily)
    candles: List[Dict[str, Any]] = []
    idx_list = list(data.index)
    for i, idx in enumerate(idx_list):
        row = data.iloc[i]
        o = float(row.get("open", row.get("close", np.nan)))
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        if hasattr(idx, "strftime"):
            ts = idx.strftime("%Y-%m-%d")
        else:
            ts = str(idx)[:10]
        candles.append({"time": ts, "open": o, "high": h, "low": l, "close": c})

    swing_hi, swing_lo = _swing_points(high, low, order=4)
    # Last several swings for markers (avoid clutter)
    markers: List[Dict[str, Any]] = []
    for j in swing_hi[-6:]:
        if 0 <= j < len(idx_list):
            idx = idx_list[j]
            ts = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            markers.append(
                {
                    "time": ts,
                    "position": "aboveBar",
                    "color": "#f87171",
                    "shape": "arrowDown",
                    "text": "SH",
                }
            )
    for j in swing_lo[-6:]:
        if 0 <= j < len(idx_list):
            idx = idx_list[j]
            ts = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            markers.append(
                {
                    "time": ts,
                    "position": "belowBar",
                    "color": "#4ade80",
                    "shape": "arrowUp",
                    "text": "SL",
                }
            )

    price_lines = [
        {"price": support, "color": "#22c55e", "title": f"Support ~{support:.2f}"},
        {"price": resistance, "color": "#ef4444", "title": f"Resistance ~{resistance:.2f}"},
    ]

    pattern_payload = [
        {
            "name": s.name,
            "direction": s.direction,
            "strength": round(s.strength, 3),
            "confidence": round(s.confidence, 3),
        }
        for s in merged
    ]

    return {
        "ticker": t,
        "period": period,
        "interval": interval,
        "candles": candles,
        "markers": markers,
        "price_lines": price_lines,
        "support": support,
        "resistance": resistance,
        "pattern_signals": pattern_payload,
        "composite_pattern_score": composite,
        "last_close": float(close.iloc[-1]),
        "bar_count": len(candles),
    }
