"""
Multi-Stock Technical Analysis Scanner
======================================
Scans Indian equities (NSE .NS) using RSI, MACD, Bollinger Bands, and momentum
to produce BUY or SELL signals only (no HOLD — weak signals resolve by score sign).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urljoin

# Yahoo Finance news base (yfinance often returns protocol-relative or site-relative links)
_YAHOO_NEWS_ORIGIN = "https://finance.yahoo.com"


def normalize_news_url(raw: str) -> str:
    """
    Turn yfinance news URLs into absolute https URLs on the real publisher/Yahoo domain.
    Fixes protocol-relative //..., site-relative /news/..., and path-only links so clicks
    never resolve to our own site's origin.
    """
    u = (raw or "").strip()
    if not u:
        return ""
    if u.startswith("//"):
        return "https:" + u
    low = u.lower()
    if low.startswith(("http://", "https://")):
        scheme = low.split(":", 1)[0]
        if scheme in ("javascript", "data", "vbscript"):
            return ""
        return u
    if u.startswith("/"):
        return _YAHOO_NEWS_ORIGIN + u
    return urljoin(_YAHOO_NEWS_ORIGIN + "/", u)

# ============================================================
# INDIA STOCK UNIVERSES (NSE)
# ============================================================

# Nifty 50 - all 50 constituents (Wikipedia Dec 2025)
NIFTY50 = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDIGO.NS", "INFY.NS", "ITC.NS",
    "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "MAXHEALTH.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS",
]
NIFTY_NEXT50 = [
    "ADANIENT.NS", "APOLLOHOSP.NS", "BEL.NS", "CIPLA.NS", "COALINDIA.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "INDIGO.NS", "JSWSTEEL.NS", "JIOFIN.NS", "MAXHEALTH.NS", "SHRIRAMFIN.NS",
    "SBILIFE.NS", "TATACONSUM.NS", "TECHM.NS", "TRENT.NS",
    "DIVISLAB.NS", "DABUR.NS", "PIDILITIND.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS",
    "BRITANNIA.NS", "BANKBARODA.NS", "BHEL.NS", "BPCL.NS", "IOC.NS",
    "VEDL.NS", "MUTHOOTFIN.NS", "AUROPHARMA.NS", "TORNTPHARM.NS", "SIEMENS.NS",
    "HAVELLS.NS", "SRF.NS", "TATAPOWER.NS", "ABB.NS", "AMBUJACEM.NS",
]

# Default scan: Nifty 50 + Nifty Next 50 (India only, de-duplicated)
DEFAULT_TICKERS = list(dict.fromkeys(NIFTY50 + NIFTY_NEXT50))


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple:
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def compute_sma_crossover(close: pd.Series, short: int = 10, long: int = 50) -> tuple:
    sma_short = close.rolling(window=short).mean()
    sma_long = close.rolling(window=long).mean()
    return sma_short, sma_long


def compute_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    return (close / close.shift(period) - 1) * 100


def compute_volume_sma_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    sma_vol = volume.rolling(window=period).mean()
    return volume / (sma_vol + 1e-10)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _get_extended_indicators(high, low, close, volume):
    """Compute Stochastic, Williams %R, CCI, ADX, OBV, Stoch RSI, Pivots."""
    from indicators import (
        compute_stochastic,
        compute_williams_r,
        compute_cci,
        compute_adx,
        compute_obv,
        compute_stoch_rsi,
        compute_pivot_points,
    )
    out = {}
    try:
        sk, sd = compute_stochastic(high, low, close)
        out["stoch_k"] = float(sk.iloc[-1]) if len(sk) and not pd.isna(sk.iloc[-1]) else None
        out["stoch_d"] = float(sd.iloc[-1]) if len(sd) and not pd.isna(sd.iloc[-1]) else None
    except Exception:
        pass
    try:
        wr = compute_williams_r(high, low, close)
        out["williams_r"] = float(wr.iloc[-1]) if len(wr) and not pd.isna(wr.iloc[-1]) else None
    except Exception:
        pass
    try:
        cci = compute_cci(high, low, close)
        out["cci"] = float(cci.iloc[-1]) if len(cci) and not pd.isna(cci.iloc[-1]) else None
    except Exception:
        pass
    try:
        adx, pdi, mdi = compute_adx(high, low, close)
        out["adx"] = float(adx.iloc[-1]) if len(adx) and not pd.isna(adx.iloc[-1]) else None
    except Exception:
        pass
    try:
        obv = compute_obv(close, volume)
        if len(obv) >= 5:
            obv_slope = (obv.iloc[-1] - obv.iloc[-5]) / (abs(obv.iloc[-1]) + 1e-10)
            out["obv_trend"] = float(np.clip(obv_slope, -1, 1))
    except Exception:
        pass
    try:
        out["stoch_rsi"] = compute_stoch_rsi(close)
    except Exception:
        pass
    try:
        pivot, r1, r2, s1, s2 = compute_pivot_points(high, low, close)
        out["pivot"], out["r1"], out["s1"] = pivot, r1, s1
    except Exception:
        pass
    return out


# ============================================================
# COMPOSITE SIGNAL LOGIC (Multi-factor scoring)
# ============================================================

@dataclass
class SignalResult:
    ticker: str
    action: str
    score: float
    price: float
    change_pct: float
    rsi: Optional[float]
    macd_hist: Optional[float]
    bb_position: Optional[float]
    momentum_10d: Optional[float]
    volume_ratio: Optional[float]
    reasons: List[str]
    buy_zone: Optional[tuple]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    support: Optional[float]
    resistance: Optional[float]
    news: List[Dict]
    news_sentiment: Optional[float] = None
    news_impact: Optional[float] = None
    pattern_signals: Optional[List[str]] = None
    pattern_score: Optional[float] = None
    factors_used: Optional[List[str]] = None  # for RLHF feedback
    # Extended indicators
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    williams_r: Optional[float] = None
    cci: Optional[float] = None
    adx: Optional[float] = None
    obv_trend: Optional[float] = None
    stoch_rsi: Optional[float] = None
    pivot: Optional[float] = None
    r1: Optional[float] = None
    s1: Optional[float] = None
    # Target timeline
    target_achieve_days: Optional[int] = None
    target_achieve_date: Optional[str] = None
    error: Optional[str] = None
    # Which indicators were used for scoring (None = all). Same order as request after normalize.
    indicators_for_score: Optional[List[str]] = None


# Indicators that participate in the composite score (manual selection applies to these).
ALL_SCORING_INDICATORS = frozenset({
    "rsi", "macd", "bollinger", "sma", "momentum", "volume",
    "stochastic", "williams_r", "cci", "adx", "obv", "stoch_rsi",
    "pattern", "news",
})

INDICATOR_ALIASES = {
    "williams": "williams_r",
    "williams_percent_r": "williams_r",
    "stoch": "stochastic",
    "bb": "bollinger",
    "vol": "volume",
    "mom": "momentum",
}

# API / UI catalog (id, human label)
INDICATOR_CATALOG: List[Dict[str, str]] = [
    {"id": "rsi", "label": "RSI"},
    {"id": "macd", "label": "MACD"},
    {"id": "bollinger", "label": "Bollinger Bands"},
    {"id": "sma", "label": "SMA crossover (10/50)"},
    {"id": "momentum", "label": "10d momentum"},
    {"id": "volume", "label": "Volume vs avg"},
    {"id": "stochastic", "label": "Stochastic"},
    {"id": "williams_r", "label": "Williams %R"},
    {"id": "cci", "label": "CCI"},
    {"id": "adx", "label": "ADX + DI"},
    {"id": "obv", "label": "OBV trend"},
    {"id": "stoch_rsi", "label": "Stochastic RSI"},
    {"id": "pattern", "label": "Chart patterns"},
    {"id": "news", "label": "News sentiment"},
]


def normalize_indicator_set(indicators: Optional[List[str]]) -> Optional[frozenset]:
    """
    None / empty / all-invalid → None (use full model).
    Otherwise return frozenset of canonical ids.
    """
    if not indicators:
        return None
    out = set()
    for x in indicators:
        k = str(x).strip().lower().replace(" ", "_").replace("-", "_")
        k = INDICATOR_ALIASES.get(k, k)
        if k in ALL_SCORING_INDICATORS:
            out.add(k)
    return frozenset(out) if out else None


def scoring_active(key: str, enabled: Optional[frozenset]) -> bool:
    return enabled is None or key in enabled


def _indicators_applied_list(enabled: Optional[frozenset]) -> List[str]:
    if enabled is None:
        return sorted(ALL_SCORING_INDICATORS)
    return sorted(enabled)


def _get_rlhf_weights() -> dict:
    """Load RLHF-adapted factor weights. Algorithm improves from feedback."""
    try:
        from rlhf import get_weights
        return get_weights()
    except Exception:
        return {}


def _get_bb_position(close: float, upper: float, mid: float, lower: float) -> float:
    """Returns -1 (at/below lower) to 1 (at/above upper). 0 = at middle."""
    if pd.isna(upper) or pd.isna(lower) or upper <= lower:
        return 0.0
    if close <= lower:
        return -1.0
    if close >= upper:
        return 1.0
    # Linear interpolate
    return 2.0 * (close - mid) / (upper - lower)


def _binary_action(score: float) -> str:
    """Always BUY or SELL: strong thresholds, else weak signals use score sign."""
    if score >= 0.35:
        return "BUY"
    if score <= -0.35:
        return "SELL"
    return "BUY" if score >= 0 else "SELL"


def analyze_ticker(
    ticker: str,
    period: str = "3mo",
    indicators: Optional[List[str]] = None,
) -> SignalResult:
    """
    Compute technical indicators and produce a composite BUY or SELL signal.
    ``indicators``: if set, only those factors contribute to the score (manual mode).
    """
    applied_labels = _indicators_applied_list(normalize_indicator_set(indicators))
    reasons = []
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data.empty or len(data) < 50:
            return SignalResult(
                ticker=ticker, action="SELL", score=0, price=0, change_pct=0,
                rsi=None, macd_hist=None, bb_position=None, momentum_10d=None,
                volume_ratio=None, reasons=["Insufficient data — defaulting to SELL until data available"],
                buy_zone=None, stop_loss=None, take_profit=None, support=None, resistance=None,
                news=[], news_sentiment=None, news_impact=None, pattern_signals=None, pattern_score=None,
                factors_used=None, error="Not enough history",
                indicators_for_score=applied_labels,
            )

        # Handle multi-index from yf
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in data.columns]
        data.columns = [c.lower() for c in data.columns]
        close = data["close"]
        volume = data["volume"] if "volume" in data.columns else pd.Series(1.0, index=data.index)

        # Compute indicators
        rsi = compute_rsi(close).iloc[-1] if len(close) >= 15 else np.nan
        macd_line, sig_line, hist = compute_macd(close)
        macd_hist = hist.iloc[-1] if len(hist) > 0 and not pd.isna(hist.iloc[-1]) else np.nan

        bb_u, bb_m, bb_l = compute_bollinger(close)
        bb_pos = _get_bb_position(close.iloc[-1], bb_u.iloc[-1], bb_m.iloc[-1], bb_l.iloc[-1])

        sma_10, sma_50 = compute_sma_crossover(close)
        mom = compute_momentum(close).iloc[-1] if len(close) >= 11 else np.nan
        vol_ratio = compute_volume_sma_ratio(volume).iloc[-1] if len(volume) >= 21 else np.nan

        price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else price
        change_pct = (price / prev_close - 1) * 100 if prev_close else 0

        high_series = data["high"] if "high" in data.columns else close
        low_series = data["low"] if "low" in data.columns else close

        # Extended indicators: Stochastic, Williams, CCI, ADX, OBV, Stoch RSI, Pivots
        ext = _get_extended_indicators(high_series, low_series, close, volume)
        stoch_k = ext.get("stoch_k")
        stoch_d = ext.get("stoch_d")
        williams_r = ext.get("williams_r")
        cci_val = ext.get("cci")
        adx_val = ext.get("adx")
        obv_trend = ext.get("obv_trend")
        stoch_rsi_val = ext.get("stoch_rsi")
        pivot_val = ext.get("pivot")
        r1_val = ext.get("r1")
        s1_val = ext.get("s1")

        # Fetch news early (needed for sentiment analysis)
        raw_news = []
        try:
            t = yf.Ticker(ticker)
            raw_news = getattr(t, "news", None) or (t.get_news() if callable(getattr(t, "get_news", None)) else []) or []
        except Exception:
            pass

        enabled = normalize_indicator_set(indicators)
        # ========== SCORING (RLHF-adaptive weights) ==========
        weights = _get_rlhf_weights()
        score = 0.0
        n_factors = 0
        factors_used = []

        # RSI: <30 bullish (+), >70 bearish (-)
        if scoring_active("rsi", enabled) and not pd.isna(rsi):
            w = weights.get("rsi", 1.0)
            if rsi < 30:
                score += 0.8 * w
                factors_used.append("rsi")
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi < 40:
                score += 0.3 * w
                factors_used.append("rsi")
                reasons.append(f"RSI low ({rsi:.0f})")
            elif rsi > 70:
                score -= 0.8 * w
                factors_used.append("rsi")
                reasons.append(f"RSI overbought ({rsi:.0f})")
            elif rsi > 60:
                score -= 0.3 * w
                factors_used.append("rsi")
                reasons.append(f"RSI high ({rsi:.0f})")

        # MACD histogram: positive = bullish, negative = bearish
        if scoring_active("macd", enabled) and not pd.isna(macd_hist) and macd_hist != 0:
            w = weights.get("macd", 1.0)
            factors_used.append("macd")
            macd_norm = np.clip(macd_hist / (price * 0.01), -1, 1)
            score += macd_norm * 0.5 * w
            if macd_hist > 0:
                reasons.append("MACD bullish")
            else:
                reasons.append("MACD bearish")

        # Bollinger position: below lower = buy, above upper = sell
        if scoring_active("bollinger", enabled) and not np.isnan(bb_pos):
            w = weights.get("bollinger", 1.0)
            factors_used.append("bollinger")
            score -= bb_pos * 0.6 * w
            if bb_pos < -0.5:
                reasons.append("Price near lower Bollinger")
            elif bb_pos > 0.5:
                reasons.append("Price near upper Bollinger")

        # SMA crossover: short > long = bullish
        if scoring_active("sma", enabled) and len(close) >= 51:
            w = weights.get("sma", 1.0)
            factors_used.append("sma")
            s10, s50 = sma_10.iloc[-1], sma_50.iloc[-1]
            if s10 > s50:
                score += 0.4 * w
                reasons.append("SMA 10 > SMA 50")
            else:
                score -= 0.4 * w
                reasons.append("SMA 10 < SMA 50")

        # Momentum (10d): positive = bullish
        if scoring_active("momentum", enabled) and not pd.isna(mom):
            w = weights.get("momentum", 1.0)
            factors_used.append("momentum")
            mom_norm = np.clip(mom / 15, -1, 1)
            score += mom_norm * 0.3 * w
            if abs(mom) > 5:
                reasons.append(f"10d momentum {mom:+.1f}%")

        # Volume confirmation: high vol on up move = stronger signal
        if scoring_active("volume", enabled) and not pd.isna(vol_ratio) and vol_ratio > 1.2:
            w = weights.get("volume", 1.0)
            factors_used.append("volume")
            if change_pct > 0:
                score += 0.2 * w
                reasons.append("Volume confirmation")
            else:
                score -= 0.2 * w
                reasons.append("High volume on decline")

        # Stochastic: <20 oversold (bullish), >80 overbought (bearish)
        if scoring_active("stochastic", enabled) and stoch_k is not None:
            w = weights.get("stochastic", 1.0)
            factors_used.append("stochastic")
            if stoch_k < 20:
                score += 0.4 * w
                reasons.append(f"Stochastic oversold ({stoch_k:.0f})")
            elif stoch_k > 80:
                score -= 0.4 * w
                reasons.append(f"Stochastic overbought ({stoch_k:.0f})")

        # Williams %R: < -80 oversold, > -20 overbought
        if scoring_active("williams_r", enabled) and williams_r is not None:
            w = weights.get("williams", 1.0)
            factors_used.append("williams_r")
            if williams_r < -80:
                score += 0.35 * w
                reasons.append(f"Williams %R oversold ({williams_r:.0f})")
            elif williams_r > -20:
                score -= 0.35 * w
                reasons.append(f"Williams %R overbought ({williams_r:.0f})")

        # CCI: < -100 oversold, > 100 overbought
        if scoring_active("cci", enabled) and cci_val is not None:
            w = weights.get("cci", 1.0)
            factors_used.append("cci")
            if cci_val < -100:
                score += 0.3 * w
                reasons.append(f"CCI oversold ({cci_val:.0f})")
            elif cci_val > 100:
                score -= 0.3 * w
                reasons.append(f"CCI overbought ({cci_val:.0f})")

        # ADX: strong trend (>25) confirms direction; +DI > -DI bullish
        if scoring_active("adx", enabled) and adx_val is not None and adx_val > 20:
            try:
                from indicators import compute_adx
                _, pdi, mdi = compute_adx(high_series, low_series, close, 14)
                pdi_val = float(pdi.iloc[-1]) if len(pdi) else None
                mdi_val = float(mdi.iloc[-1]) if len(mdi) else None
                if pdi_val is not None and mdi_val is not None:
                    w = weights.get("adx", 1.0)
                    factors_used.append("adx")
                    if pdi_val > mdi_val:
                        score += 0.25 * w
                        reasons.append(f"ADX trend bullish ({adx_val:.0f})")
                    else:
                        score -= 0.25 * w
                        reasons.append(f"ADX trend bearish ({adx_val:.0f})")
            except Exception:
                pass

        # OBV trend: positive slope = bullish
        if scoring_active("obv", enabled) and obv_trend is not None and obv_trend != 0:
            w = weights.get("obv", 1.0)
            factors_used.append("obv")
            score += obv_trend * 0.2 * w
            if obv_trend > 0.3:
                reasons.append("OBV rising")
            elif obv_trend < -0.3:
                reasons.append("OBV falling")

        # Stoch RSI: <20 oversold, >80 overbought
        if scoring_active("stoch_rsi", enabled) and stoch_rsi_val is not None:
            w = weights.get("stoch_rsi", 1.0)
            factors_used.append("stoch_rsi")
            if stoch_rsi_val < 20:
                score += 0.35 * w
                reasons.append(f"Stoch RSI oversold ({stoch_rsi_val:.0f})")
            elif stoch_rsi_val > 80:
                score -= 0.35 * w
                reasons.append(f"Stoch RSI overbought ({stoch_rsi_val:.0f})")

        # Pattern analysis (always detect for display; score only if enabled)
        pattern_signals_list = []
        pattern_score_val = 0.0
        news_sentiment_val = 0.0
        news_impact_val = 0.0
        try:
            from patterns import detect_all_patterns
            ph = data["high"] if "high" in data.columns else close
            pl = data["low"] if "low" in data.columns else close
            support_pre = float(close.rolling(20).min().iloc[-1]) if len(close) >= 20 else price * 0.97
            resistance_pre = float(close.rolling(20).max().iloc[-1]) if len(close) >= 20 else price * 1.03
            pat_sigs, pattern_score_val = detect_all_patterns(
                data, close, support_pre, resistance_pre
            )
            pattern_signals_list = [f"{s.name}({s.direction})" for s in pat_sigs]
            if scoring_active("pattern", enabled) and pat_sigs:
                w = weights.get("pattern", 1.0)
                factors_used.append("pattern")
                score += pattern_score_val * 0.5 * w
                for s in pat_sigs:
                    if s.direction != 0:
                        reasons.append(f"Pattern: {s.name} (bearish)" if s.direction < 0 else f"Pattern: {s.name} (bullish)")
        except Exception:
            pass

        # News sentiment (always compute for display; score only if enabled)
        try:
            from news_analysis import analyze_news_sentiment
            news_list_for_analysis = []
            for n in (raw_news or []):
                news_list_for_analysis.append({
                    "title": n.get("title") or n.get("link") or "",
                    "description": n.get("description", ""),
                    "content": n.get("content", ""),
                })
            news_result = analyze_news_sentiment(news_list_for_analysis)
            news_sentiment_val = news_result["sentiment_score"]
            news_impact_val = news_result["impact"]
            if scoring_active("news", enabled) and news_impact_val > 0.2 and abs(news_sentiment_val) > 0.2:
                w = weights.get("news", 1.0)
                factors_used.append("news")
                score += news_sentiment_val * news_impact_val * 0.6 * w
                reasons.append(news_result["summary"])
        except Exception:
            pass

        n_factors = len(factors_used) if factors_used else 1
        # Normalize score to [-1, 1] - balanced so BUY/SELL require clear majority
        if n_factors > 0:
            score = np.clip(score / max(n_factors * 0.45, 1), -1, 1)
        else:
            score = 0

        # BUY or SELL only (no HOLD — weak band maps by score sign)
        action = _binary_action(float(score))

        # Support / Resistance (20-day low/high)
        low_20 = float(close.rolling(20).min().iloc[-1]) if len(close) >= 20 else price * 0.97
        high_20 = float(close.rolling(20).max().iloc[-1]) if len(close) >= 20 else price * 1.03
        support = low_20
        resistance = high_20

        # ATR for stop placement
        atr = compute_atr(high_series, low_series, close, 14).iloc[-1] if len(close) >= 15 else price * 0.02
        atr = float(atr) if not (pd.isna(atr) or atr <= 0) else price * 0.02

        # Buy zone, SL, TP by action
        if action == "BUY":
            buy_zone = (support, min(price * 1.01, support * 1.03))
            stop_loss = support - atr * 1.5
            take_profit = resistance + atr * 0.5
        else:
            buy_zone = None
            stop_loss = high_20 + atr * 1.5
            take_profit = support - atr * 0.5

        # Target timeline: estimate days to reach TP (or SL for SELL)
        target_price = take_profit if action == "BUY" else stop_loss
        atr_pct = (atr / price) * 100
        daily_return_pct = abs(change_pct) if change_pct != 0 else atr_pct
        try:
            from indicators import compute_target_days
            target_days = compute_target_days(price, target_price, daily_return_pct, atr_pct)
        except Exception:
            target_days = max(5, min(30, int(20 * atr_pct)))
        from datetime import datetime, timedelta
        target_date = (datetime.now() + timedelta(days=target_days)).strftime("%Y-%m-%d")

        # Build news list from raw_news — absolute external URLs only (see normalize_news_url)
        news_list = []
        for n in (raw_news or [])[:8]:
            raw_url = n.get("link") or n.get("url") or ""
            url = normalize_news_url(str(raw_url))
            title = (n.get("title") or "").strip() or "Yahoo Finance article"
            news_list.append({
                "title": title[:120],
                "url": url,
                "publisher": n.get("publisher") or n.get("source") or "",
            })

        return SignalResult(
            ticker=ticker,
            action=action,
            score=float(score),
            price=price,
            change_pct=float(change_pct),
            rsi=float(rsi) if not pd.isna(rsi) else None,
            macd_hist=float(macd_hist) if not pd.isna(macd_hist) else None,
            bb_position=float(bb_pos),
            momentum_10d=float(mom) if not pd.isna(mom) else None,
            volume_ratio=float(vol_ratio) if not pd.isna(vol_ratio) else None,
            reasons=reasons if reasons else ["No strong signals"],
            buy_zone=buy_zone,
            stop_loss=stop_loss,
            take_profit=take_profit,
            support=support,
            resistance=resistance,
            news=news_list,
            news_sentiment=news_sentiment_val,
            news_impact=news_impact_val,
            pattern_signals=pattern_signals_list if pattern_signals_list else None,
            pattern_score=round(pattern_score_val, 3) if pattern_score_val != 0 else None,
            factors_used=factors_used if factors_used else None,
            stochastic_k=stoch_k,
            stochastic_d=stoch_d,
            williams_r=williams_r,
            cci=cci_val,
            adx=adx_val,
            obv_trend=obv_trend,
            stoch_rsi=stoch_rsi_val,
            pivot=pivot_val,
            r1=r1_val,
            s1=s1_val,
            target_achieve_days=target_days,
            target_achieve_date=target_date,
            indicators_for_score=_indicators_applied_list(enabled),
        )

    except Exception as e:
        return SignalResult(
            ticker=ticker, action="SELL", score=0, price=0, change_pct=0,
            rsi=None, macd_hist=None, bb_position=None, momentum_10d=None,
            volume_ratio=None, reasons=[f"Error: {str(e)}"],
            buy_zone=None, stop_loss=None, take_profit=None, support=None, resistance=None,
            news=[], news_sentiment=None, news_impact=None, pattern_signals=None, pattern_score=None,
            factors_used=None, error=str(e),
            indicators_for_score=applied_labels,
        )


def scan_tickers(
    tickers: Optional[List[str]] = None,
    period: str = "3mo",
    filter_action: Optional[str] = None,
    indicators: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Scan a list of tickers and return signal results.
    filter_action: "BUY" | "SELL" | None (return all)
    indicators: optional subset of factor ids for scoring (manual mode); None = all.
    """
    tickers = tickers or DEFAULT_TICKERS
    results = []
    for t in tickers:
        r = analyze_ticker(t, period=period, indicators=indicators)
        buy_zone_ser = [round(r.buy_zone[0], 2), round(r.buy_zone[1], 2)] if r.buy_zone else None
        d = {
            "ticker": r.ticker,
            "action": r.action,
            "factors_used": r.factors_used,
            "score": round(r.score, 3),
            "price": r.price,
            "change_pct": round(r.change_pct, 2),
            "rsi": round(r.rsi, 1) if r.rsi is not None else None,
            "macd_hist": round(r.macd_hist, 4) if r.macd_hist is not None else None,
            "bb_position": round(r.bb_position, 2) if r.bb_position is not None else None,
            "momentum_10d": round(r.momentum_10d, 2) if r.momentum_10d is not None else None,
            "volume_ratio": round(r.volume_ratio, 2) if r.volume_ratio is not None else None,
            "reasons": r.reasons,
            "buy_zone": buy_zone_ser,
            "stop_loss": round(r.stop_loss, 2) if r.stop_loss is not None else None,
            "take_profit": round(r.take_profit, 2) if r.take_profit is not None else None,
            "support": round(r.support, 2) if r.support is not None else None,
            "resistance": round(r.resistance, 2) if r.resistance is not None else None,
            "news": r.news,
            "news_sentiment": round(r.news_sentiment, 3) if r.news_sentiment is not None else None,
            "news_impact": round(r.news_impact, 2) if r.news_impact is not None else None,
            "pattern_signals": r.pattern_signals,
            "pattern_score": r.pattern_score,
            "stochastic_k": round(r.stochastic_k, 1) if r.stochastic_k is not None else None,
            "stochastic_d": round(r.stochastic_d, 1) if r.stochastic_d is not None else None,
            "williams_r": round(r.williams_r, 1) if r.williams_r is not None else None,
            "cci": round(r.cci, 1) if r.cci is not None else None,
            "adx": round(r.adx, 1) if r.adx is not None else None,
            "obv_trend": round(r.obv_trend, 2) if r.obv_trend is not None else None,
            "stoch_rsi": round(r.stoch_rsi, 1) if r.stoch_rsi is not None else None,
            "pivot": round(r.pivot, 2) if r.pivot is not None else None,
            "r1": round(r.r1, 2) if r.r1 is not None else None,
            "s1": round(r.s1, 2) if r.s1 is not None else None,
            "target_achieve_days": r.target_achieve_days,
            "target_achieve_date": r.target_achieve_date,
            "error": r.error,
            "indicators_for_score": r.indicators_for_score,
        }
        if filter_action and r.action != filter_action:
            continue
        results.append(d)
    return results
