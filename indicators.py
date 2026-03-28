"""
All Technical Indicators
========================
Stochastic, Williams %R, CCI, ADX, OBV, Pivot Points, Stoch RSI
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                      k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    stoch_k = 100 * (close - low_min) / (high_max - low_min + 1e-10)
    stoch_d = stoch_k.rolling(d_period).mean()
    return stoch_k, stoch_d


def compute_williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                       period: int = 14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll + 1e-10)


def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3
    sma = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (typical - sma) / (0.015 * mad + 1e-10)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (ADX, +DI, -DI)"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def compute_pivot_points(high: pd.Series, low: pd.Series, close: pd.Series
) -> Tuple[float, float, float, float, float]:
    """Returns (pivot, r1, r2, s1, s2) for last bar."""
    h, l, c = high.iloc[-1], low.iloc[-1], close.iloc[-1]
    pivot = (h + l + c) / 3
    r1 = 2 * pivot - l
    r2 = pivot + (h - l)
    s1 = 2 * pivot - h
    s2 = pivot - (h - l)
    return pivot, r1, r2, s1, s2


def compute_stoch_rsi(close: pd.Series, rsi_period: int = 14,
                      stoch_period: int = 14) -> Optional[float]:
    """Stochastic RSI - RSI of RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)
    if len(rsi) < stoch_period:
        return None
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    return float(stoch_rsi.iloc[-1]) if not pd.isna(stoch_rsi.iloc[-1]) else None


def compute_target_days(price: float, target: float, daily_return_pct: float,
                        atr_pct: float) -> int:
    """
    Estimate trading days to reach target.
    daily_return_pct: avg daily % move (from momentum)
    atr_pct: ATR as % of price (volatility)
    """
    if target <= 0 or price <= 0:
        return 0
    pct_to_target = abs((target - price) / price) * 100
    daily_expected = max(abs(daily_return_pct) * 0.5, atr_pct * 0.3)
    if daily_expected < 0.1:
        daily_expected = 0.5
    days = int(pct_to_target / daily_expected)
    return max(1, min(90, days))
