"""
Technical Pattern Detection & Outcome Analysis
==============================================
Detects chart patterns (double top/bottom, candlesticks, trends),
evaluates historical success rates, and contributes to buy/sell signal.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PatternSignal:
    name: str
    direction: int   # 1 = bullish, -1 = bearish, 0 = neutral
    strength: float  # 0 to 1
    confidence: float  # historical hit rate if available


def _swing_highs_lows(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 5
) -> Tuple[pd.Series, pd.Series]:
    """Detect swing highs and lows."""
    swing_high = high.rolling(window, center=True).max() == high
    swing_low = low.rolling(window, center=True).min() == low
    return swing_high, swing_low


def detect_double_bottom(close: pd.Series, high: pd.Series, low: pd.Series, lookback: int = 30
) -> Optional[PatternSignal]:
    """Double bottom: two troughs at similar levels, bullish reversal."""
    if len(close) < lookback:
        return None
    window = 5
    _, swing_low = _swing_highs_lows(high, low, close, window)
    lows = close[swing_low].dropna().tail(10)
    if len(lows) < 2:
        return None
    l1, l2 = lows.iloc[-2], lows.iloc[-1]
    # Troughs within 2% of each other
    if abs(l1 - l2) / min(l1, l2) < 0.02:
        return PatternSignal("double_bottom", 1, 0.7, 0.65)
    return None


def detect_double_top(close: pd.Series, high: pd.Series, low: pd.Series, lookback: int = 30
) -> Optional[PatternSignal]:
    """Double top: two peaks at similar levels, bearish reversal."""
    if len(close) < lookback:
        return None
    swing_high, _ = _swing_highs_lows(high, low, close, 5)
    highs = close[swing_high].dropna().tail(10)
    if len(highs) < 2:
        return None
    h1, h2 = highs.iloc[-2], highs.iloc[-1]
    if abs(h1 - h2) / max(h1, h2) < 0.02:
        return PatternSignal("double_top", -1, 0.7, 0.62)
    return None


def detect_candlestick_patterns(open_p: pd.Series, high: pd.Series, low: pd.Series,
                                close: pd.Series) -> List[PatternSignal]:
    """Detect candlestick patterns: hammer, engulfing, doji."""
    signals = []
    if len(close) < 3:
        return signals
    o, h, l, c = open_p.iloc[-1], high.iloc[-1], low.iloc[-1], close.iloc[-1]
    body = abs(c - o)
    candle_range = h - l
    if candle_range < 1e-8:
        return signals
    body_pct = body / candle_range
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)

    # Hammer (bullish): small body, long lower wick
    if body_pct < 0.35 and lower_wick > 2 * body and upper_wick < body:
        signals.append(PatternSignal("hammer", 1, 0.6, 0.58))
    # Inverted hammer at bottom
    if body_pct < 0.35 and upper_wick > 2 * body and lower_wick < body:
        signals.append(PatternSignal("inverted_hammer", 1, 0.5, 0.55))

    # Bearish engulfing (previous up, current down, body engulfs)
    if len(close) >= 2:
        po, pc = open_p.iloc[-2], close.iloc[-2]
        if pc > po and c < o and o >= pc and c <= po:
            signals.append(PatternSignal("bearish_engulfing", -1, 0.65, 0.6))
        # Bullish engulfing
        if pc < po and c > o and o <= pc and c >= po:
            signals.append(PatternSignal("bullish_engulfing", 1, 0.65, 0.6))

    # Doji: very small body
    if body_pct < 0.1:
        signals.append(PatternSignal("doji", 0, 0.3, 0.5))

    return signals


def detect_trend_pattern(close: pd.Series, window: int = 20) -> Optional[PatternSignal]:
    """Higher highs + higher lows = uptrend, opposite = downtrend."""
    if len(close) < window * 2:
        return None
    recent = close.iloc[-window:]
    first_half = close.iloc[-window:-window//2]
    second_half = close.iloc[-window//2:]
    hh = second_half.max() > first_half.max()
    hl = second_half.min() > first_half.min()
    lh = second_half.max() < first_half.max()
    ll = second_half.min() < first_half.min()
    if hh and hl:
        return PatternSignal("uptrend", 1, 0.5, 0.55)
    if lh and ll:
        return PatternSignal("downtrend", -1, 0.5, 0.55)
    return None


def detect_support_break(close: pd.Series, support: float, window: int = 5) -> Optional[PatternSignal]:
    """Price broke below support -> bearish."""
    if len(close) < window:
        return None
    recent_low = close.iloc[-window:].min()
    prev = close.iloc[:-window].tail(20)
    if len(prev) < 5:
        return None
    was_above = (prev >= support * 0.995).all()
    if was_above and recent_low < support * 0.995:
        return PatternSignal("support_break", -1, 0.6, 0.6)
    return None


def detect_resistance_break(close: pd.Series, resistance: float, window: int = 5) -> Optional[PatternSignal]:
    """Price broke above resistance -> bullish."""
    if len(close) < window:
        return None
    recent_high = close.iloc[-window:].max()
    prev = close.iloc[:-window].tail(20)
    if len(prev) < 5:
        return None
    was_below = (prev <= resistance * 1.005).all()
    if was_below and recent_high > resistance * 1.005:
        return PatternSignal("resistance_break", 1, 0.6, 0.6)
    return None


def compute_historical_pattern_outcomes(
    close: pd.Series, high: pd.Series, low: pd.Series, open_p: pd.Series,
    fwd_days: int = 5
) -> Dict[str, float]:
    """
    Backtest: for each pattern occurrence in history, what was fwd_days return?
    Returns avg hit rate per pattern type (simplified).
    """
    outcomes = {}  # pattern_name -> avg_forward_return
    if len(close) < 60:
        return outcomes
    for i in range(30, len(close) - fwd_days):
        o, h, l, c = open_p.iloc[i], high.iloc[i], low.iloc[i], close.iloc[i]
        oc = close.iloc[i - 1]
        fwd_ret = (close.iloc[i + fwd_days] / c - 1) * 100

        # Hammer-like
        body = abs(c - o)
        rng = h - l
        if rng > 1e-8:
            bw = body / rng
            lw = (min(o, c) - l) / rng
            if bw < 0.35 and lw > 0.6:
                outcomes["hammer"] = outcomes.get("hammer", []) + [fwd_ret]
        # Bullish/bearish
        if c > o and oc < open_p.iloc[i - 1]:
            outcomes["bull_candle"] = outcomes.get("bull_candle", []) + [fwd_ret]
        elif c < o and oc > open_p.iloc[i - 1]:
            outcomes["bear_candle"] = outcomes.get("bear_candle", []) + [fwd_ret]

    return {k: np.mean(v) for k, v in outcomes.items() if len(v) >= 3}


def detect_all_patterns(
    data: pd.DataFrame, close: pd.Series, support: float, resistance: float
) -> Tuple[List[PatternSignal], float]:
    """
    Run all pattern detectors. Returns (list of PatternSignals, composite_pattern_score -1 to 1).
    """
    signals = []
    high = data["high"] if "high" in data.columns else close
    low = data["low"] if "low" in data.columns else close
    open_p = data["open"] if "open" in data.columns else close

    p = detect_double_bottom(close, high, low)
    if p:
        signals.append(p)
    p = detect_double_top(close, high, low)
    if p:
        signals.append(p)
    signals.extend(detect_candlestick_patterns(open_p, high, low, close))
    p = detect_trend_pattern(close)
    if p:
        signals.append(p)
    p = detect_support_break(close, support)
    if p:
        signals.append(p)
    p = detect_resistance_break(close, resistance)
    if p:
        signals.append(p)

    if not signals:
        return [], 0.0

    # Weight by direction * strength * confidence
    score = 0.0
    total_w = 0.0
    for s in signals:
        w = s.strength * s.confidence
        score += s.direction * w
        total_w += w
    composite = score / total_w if total_w > 0 else 0.0
    composite = max(-1.0, min(1.0, composite))
    return signals, composite
