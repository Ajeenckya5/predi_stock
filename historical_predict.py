"""
Use past price history to estimate edge for the *current* bar's setup.

Compares how similar conditions behaved N bars forward, then blends into a prediction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _forward_return(close: pd.Series, h: int) -> pd.Series:
    return close.shift(-h) / close - 1.0


def _bb_squeeze_mask(close: pd.Series, period: int = 20, lookback: int = 40) -> pd.Series:
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = ma + 2 * sd
    lower = ma - 2 * sd
    bw = (upper - lower) / (ma.abs() + 1e-8)
    med = bw.rolling(lookback, min_periods=max(period, 10)).median()
    return (bw < med * 0.72) & med.notna()


def _uptrend_mask(close: pd.Series, fast: int = 10, slow: int = 30) -> pd.Series:
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    return (close > f) & (f > s)


def _downtrend_mask(close: pd.Series, fast: int = 10, slow: int = 30) -> pd.Series:
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    return (close < f) & (f < s)


def _near_level(close: pd.Series, level: pd.Series, pct: float = 0.012) -> pd.Series:
    return (close - level).abs() / (close.abs() + 1e-8) < pct


def _edge_for_mask(mask: pd.Series, fwd: pd.Series, fwd_days: int) -> Dict[str, Any]:
    """Historical stats where mask was True (excluding bars without full forward window)."""
    m = mask.fillna(False) & fwd.notna()
    m_hist = m.copy()
    m_hist.iloc[-fwd_days:] = False
    fr = fwd[m_hist]
    n = int(len(fr))
    if n == 0:
        return {"n": 0, "mean_fwd_pct": None, "hit_rate_up": None, "std_fwd_pct": None}
    arr = fr.values.astype(float)
    return {
        "n": n,
        "mean_fwd_pct": round(float(np.mean(arr) * 100), 3),
        "hit_rate_up": round(float(np.mean(arr > 0)), 3),
        "std_fwd_pct": round(float(np.std(arr) * 100), 3),
    }


def _k_nearest_regime_prediction(
    close: pd.Series,
    fwd_days: int = 5,
    regime_len: int = 15,
    top_k: int = 18,
) -> Optional[Dict[str, Any]]:
    """
    Find past bars whose return+volatility regime best matches the latest window,
    then average their realized forward returns.
    """
    L = len(close)
    need = regime_len + fwd_days + 5
    if L < need:
        return None
    lr = np.log(close.astype(float))
    rets = lr.diff()
    cur_slope = float(lr.iloc[-1] - lr.iloc[-regime_len - 1])
    cur_vol = float(rets.iloc[-regime_len:].std() or 0.0)
    if cur_vol < 1e-12:
        cur_vol = 1e-12

    fwd = _forward_return(close, fwd_days)
    candidates: List[Tuple[float, int]] = []
    for t in range(regime_len, L - fwd_days - 1):
        slope = float(lr.iloc[t] - lr.iloc[t - regime_len - 1])
        vol = float(rets.iloc[t - regime_len : t].std() or 0.0)
        if vol < 1e-12:
            vol = 1e-12
        d = abs(slope - cur_slope) / (abs(cur_slope) + 1e-6) + abs(vol - cur_vol) / (cur_vol + 1e-6)
        candidates.append((d, t))

    candidates.sort(key=lambda x: x[0])
    take = [t for _, t in candidates[:top_k]]
    fr = fwd.iloc[take].dropna()
    if len(fr) < 5:
        return None
    arr = fr.values.astype(float)
    return {
        "method": "similar_regime",
        "regime_len": regime_len,
        "top_k": len(fr),
        "mean_fwd_pct": round(float(np.mean(arr) * 100), 3),
        "hit_rate_up": round(float(np.mean(arr > 0)), 3),
        "std_fwd_pct": round(float(np.std(arr) * 100), 3),
    }


def compute_historical_prediction(
    data: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    support: float,
    resistance: float,
    fwd_days: int = 5,
) -> Dict[str, Any]:
    """
    Analyze *previous* history for setups like today's, then predict directional edge
    for the next ``fwd_days`` bars (proportionally to history).
    """
    close = close.astype(float)
    fwd = _forward_return(close, fwd_days)

    roll_low = close.rolling(20).min()
    roll_high = close.rolling(20).max()
    near_sup = _near_level(close, roll_low, 0.015)
    near_res = _near_level(close, roll_high, 0.015)

    masks = {
        "bb_squeeze": _bb_squeeze_mask(close),
        "uptrend": _uptrend_mask(close),
        "downtrend": _downtrend_mask(close),
        "near_support": near_sup,
        "near_resistance": near_res,
    }

    edges = {k: _edge_for_mask(v, fwd, fwd_days) for k, v in masks.items()}

    active: List[str] = []
    weighted_sum = 0.0
    weight_total = 0.0
    detail: List[Dict[str, Any]] = []

    for name, m in masks.items():
        on = bool(m.iloc[-1]) if len(m) else False
        e = edges[name]
        if not on or e["n"] < 5:
            detail.append(
                {
                    "feature": name,
                    "active_now": on,
                    "historical_samples": e["n"],
                    "mean_fwd_pct_if_active_in_past": e["mean_fwd_pct"],
                    "hit_rate_up": e["hit_rate_up"],
                }
            )
            continue
        active.append(name)
        w = float(np.sqrt(e["n"]))  # more history → slightly more weight
        mu = (e["mean_fwd_pct"] or 0) / 100.0
        weighted_sum += w * mu
        weight_total += w
        detail.append(
            {
                "feature": name,
                "active_now": True,
                "historical_samples": e["n"],
                "mean_fwd_pct_if_active_in_past": e["mean_fwd_pct"],
                "hit_rate_up": e["hit_rate_up"],
            }
        )

    regime = _k_nearest_regime_prediction(close, fwd_days=fwd_days)

    # Blend: 60% active-feature edge, 40% similar-regime (if available)
    pred_return = None
    confidence = 0.0
    if weight_total > 0:
        pred_return = weighted_sum / weight_total
        confidence = min(1.0, weight_total / (weight_total + 2.0))

    blend_mu = pred_return
    if regime and regime.get("mean_fwd_pct") is not None:
        rmu = float(regime["mean_fwd_pct"]) / 100.0
        if blend_mu is None:
            blend_mu = rmu
            confidence = min(0.85, float(regime["top_k"]) / 30.0)
        else:
            blend_mu = 0.55 * blend_mu + 0.45 * rmu
            confidence = min(1.0, confidence + 0.15)

    if blend_mu is None:
        direction = "neutral"
        score = 0.0
    else:
        score = float(np.clip(blend_mu * 25.0, -1.0, 1.0))
        if blend_mu > 0.002:
            direction = "bullish"
        elif blend_mu < -0.002:
            direction = "bearish"
        else:
            direction = "neutral"

    summary_parts = []
    if active:
        summary_parts.append(
            "Today matches: " + ", ".join(active) + f" (using past {fwd_days}-bar forward returns)."
        )
    if regime:
        hr = regime.get("hit_rate_up")
        hr_pct = f"{float(hr) * 100:.0f}" if hr is not None else "?"
        summary_parts.append(
            f"Similar past regimes ({regime['top_k']} nearest): avg fwd {regime['mean_fwd_pct']}% "
            f"({hr_pct}% up)."
        )
    if blend_mu is not None:
        summary_parts.append(
            f"Blended expectation ~{blend_mu*100:.2f}% over {fwd_days} bars (educational, not advice)."
        )

    return {
        "forward_horizon_bars": fwd_days,
        "direction": direction,
        "score": round(score, 4),
        "expected_fwd_return_pct": round(float(blend_mu * 100), 4) if blend_mu is not None else None,
        "confidence": round(float(confidence), 3),
        "active_features_now": active,
        "feature_edges": detail,
        "similar_regime": regime,
        "summary": " ".join(summary_parts) if summary_parts else "Not enough overlapping history for a strong read.",
    }
