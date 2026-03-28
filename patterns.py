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


def _composite_score(signals: List[PatternSignal]) -> float:
    if not signals:
        return 0.0
    score = 0.0
    total_w = 0.0
    for s in signals:
        w = s.strength * s.confidence
        score += s.direction * w
        total_w += w
    composite = score / total_w if total_w > 0 else 0.0
    return max(-1.0, min(1.0, composite))


def detect_symmetrical_triangle(
    high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 25
) -> Optional[PatternSignal]:
    """Higher lows + lower highs in recent window → coiling (breakout pending)."""
    if len(close) < lookback + 5:
        return None
    seg_h = high.iloc[-lookback:]
    seg_l = low.iloc[-lookback:]
    # Linear regression slope of swing highs vs swing lows (simplified: first vs second half)
    mid = lookback // 2
    h1, h2 = seg_h.iloc[:mid].max(), seg_h.iloc[mid:].max()
    l1, l2 = seg_l.iloc[:mid].min(), seg_l.iloc[mid:].min()
    lower_highs = h2 < h1 * 0.998
    higher_lows = l2 > l1 * 1.002
    if lower_highs and higher_lows and (h1 - l1) > 1e-8:
        rng = (h2 - l2) / (h1 - l1)
        if rng < 0.75:  # range compressing
            direction = 1 if close.iloc[-1] > close.iloc[-lookback] else -1
            return PatternSignal("symmetrical_triangle", direction, 0.55, 0.58)
    return None


def detect_bb_squeeze(close: pd.Series, period: int = 20, lookback: int = 40) -> Optional[PatternSignal]:
    """Narrowing Bollinger bandwidth vs recent history → volatility squeeze (often precedes move)."""
    if len(close) < period + lookback:
        return None
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = ma + 2 * sd
    lower = ma - 2 * sd
    bw = (upper - lower) / (ma.abs() + 1e-8)
    cur = float(bw.iloc[-1])
    past = float(bw.iloc[-lookback:-5].median()) if lookback > 5 else float(bw.iloc[-20])
    if past > 1e-8 and cur < past * 0.72:
        direction = 1 if close.iloc[-1] > ma.iloc[-1] else -1
        return PatternSignal("bb_squeeze", direction, 0.62, 0.64)
    return None


def detect_three_line_strike(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Optional[PatternSignal]:
    """Three consecutive same-direction candles then reversal bar (bullish/bearish variant)."""
    if len(close) < 5:
        return None
    o3 = open_p.iloc[-4:-1]
    c3 = close.iloc[-4:-1]
    last_o, last_c = float(open_p.iloc[-1]), float(close.iloc[-1])
    bull = all(float(c3.iloc[i]) > float(o3.iloc[i]) for i in range(3))
    bear = all(float(c3.iloc[i]) < float(o3.iloc[i]) for i in range(3))
    if bull and last_c < last_o:
        return PatternSignal("three_line_strike_bear", -1, 0.58, 0.56)
    if bear and last_c > last_o:
        return PatternSignal("three_line_strike_bull", 1, 0.58, 0.56)
    return None


def detect_double_bottom_improved(
    close: pd.Series, high: pd.Series, low: pd.Series, lookback: int = 45
) -> Optional[PatternSignal]:
    """Double bottom using swing lows on low series (stronger than close-only)."""
    if len(low) < lookback:
        return None
    swing_high, swing_low = _swing_highs_lows(high, low, close, window=4)
    troughs = low[swing_low].dropna()
    if len(troughs) < 2:
        return None
    l1, l2 = float(troughs.iloc[-2]), float(troughs.iloc[-1])
    if min(l1, l2) < 1e-8:
        return None
    if abs(l1 - l2) / min(l1, l2) < 0.025 and l2 <= low.iloc[-lookback:].quantile(0.15):
        return PatternSignal("double_bottom", 1, 0.78, 0.7)
    return None


def detect_double_top_improved(
    close: pd.Series, high: pd.Series, low: pd.Series, lookback: int = 45
) -> Optional[PatternSignal]:
    swing_high, swing_low = _swing_highs_lows(high, low, close, window=4)
    peaks = high[swing_high].dropna()
    if len(peaks) < 2:
        return None
    h1, h2 = float(peaks.iloc[-2]), float(peaks.iloc[-1])
    if max(h1, h2) < 1e-8:
        return None
    if abs(h1 - h2) / max(h1, h2) < 0.025 and h2 >= high.iloc[-lookback:].quantile(0.85):
        return PatternSignal("double_top", -1, 0.78, 0.68)
    return None


def detect_enhanced_patterns(
    data: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    support: float,
    resistance: float,
) -> Tuple[List[PatternSignal], float]:
    """
    Extra chart-structure patterns (triangle, squeeze, three-line strike).
    Double top/bottom variants are handled in detect_all_patterns only.
    """
    signals: List[PatternSignal] = []
    open_p = data["open"] if "open" in data.columns else close

    p = detect_symmetrical_triangle(high, low, close)
    if p:
        signals.append(p)
    p = detect_bb_squeeze(close)
    if p:
        signals.append(p)
    p = detect_three_line_strike(open_p, high, low, close)
    if p:
        signals.append(p)

    return signals, _composite_score(signals)


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

    # Prefer improved double patterns when they fire (replace naive doubles)
    p_imp = detect_double_bottom_improved(close, high, low)
    p_naive = detect_double_bottom(close, high, low)
    if p_imp:
        signals.append(p_imp)
    elif p_naive:
        signals.append(p_naive)

    p_imp_t = detect_double_top_improved(close, high, low)
    p_naive_t = detect_double_top(close, high, low)
    if p_imp_t:
        signals.append(p_imp_t)
    elif p_naive_t:
        signals.append(p_naive_t)

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

    # Structure patterns (triangle, BB squeeze, three-line strike)
    extra, _ = detect_enhanced_patterns(data, close, high, low, support, resistance)
    have = {(s.name, s.direction) for s in signals}
    for s in extra:
        if (s.name, s.direction) not in have:
            signals.append(s)
            have.add((s.name, s.direction))

    return signals, _composite_score(signals)
