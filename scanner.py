"""
Multi-Stock Technical Analysis Scanner
======================================
Scans many stocks using RSI, MACD, Bollinger Bands, and momentum
to produce BUY/SELL/HOLD signals. No per-ticker training required.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from dataclasses import dataclass

# ============================================================
# POPULAR STOCK UNIVERSES (curated for reliability & liquidity)
# ============================================================

SP100_ISHARES = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "UNH", "JNJ", "JPM",
    "V", "PG", "XOM", "MA", "HD", "CVX", "MRK", "ABBV", "KO", "PEP",
    "COST", "LLY", "WMT", "MCD", "CSCO", "ACN", "ABT", "TMO", "AVGO", "DHR",
    "NEE", "NKE", "BMY", "PM", "RTX", "HON", "INTC", "TXN", "UPS", "QCOM",
    "LOW", "AMGN", "INTU", "AMAT", "SBUX", "LMT", "ADP", "GILD", "MDLZ", "CAT",
]

NIFTY50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "WIPRO.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "BAJFINANCE.NS", "NESTLEIND.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "TATASTEEL.NS", "POWERGRID.NS", "ONGC.NS",
    "NTPC.NS", "INDUSINDBK.NS", "M&M.NS", "BAJAJFINSV.NS", "ADANIPORTS.NS",
]

# Default universe: US + India
DEFAULT_TICKERS = SP100_ISHARES + NIFTY50


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


# ============================================================
# COMPOSITE SIGNAL LOGIC (Multi-factor scoring)
# ============================================================

@dataclass
class SignalResult:
    ticker: str
    action: str       # BUY | SELL | HOLD
    score: float      # -1 to 1 (negative= bearish, positive= bullish)
    price: float
    change_pct: float
    rsi: Optional[float]
    macd_hist: Optional[float]
    bb_position: Optional[float]  # -1 to 1 (below lower = -1, above upper = 1)
    momentum_10d: Optional[float]
    volume_ratio: Optional[float]
    reasons: List[str]
    error: Optional[str] = None


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


def analyze_ticker(ticker: str, period: str = "3mo") -> SignalResult:
    """
    Compute technical indicators and produce a composite BUY/SELL/HOLD signal.
    Uses RSI, MACD, Bollinger Bands, SMA crossover, momentum, volume.
    """
    reasons = []
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data.empty or len(data) < 50:
            return SignalResult(
                ticker=ticker, action="HOLD", score=0, price=0, change_pct=0,
                rsi=None, macd_hist=None, bb_position=None, momentum_10d=None,
                volume_ratio=None, reasons=["Insufficient data"],
                error="Not enough history"
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

        # ========== SCORING (each factor contributes -1 to 1) ==========
        score = 0.0
        n_factors = 0

        # RSI: <30 bullish (+), >70 bearish (-)
        if not pd.isna(rsi):
            n_factors += 1
            if rsi < 30:
                score += 0.8
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi < 40:
                score += 0.3
                reasons.append(f"RSI low ({rsi:.0f})")
            elif rsi > 70:
                score -= 0.8
                reasons.append(f"RSI overbought ({rsi:.0f})")
            elif rsi > 60:
                score -= 0.3
                reasons.append(f"RSI high ({rsi:.0f})")

        # MACD histogram: positive = bullish, negative = bearish
        if not pd.isna(macd_hist) and macd_hist != 0:
            n_factors += 1
            # Normalize by price (roughly)
            macd_norm = np.clip(macd_hist / (price * 0.01), -1, 1)
            score += macd_norm * 0.5
            if macd_hist > 0:
                reasons.append("MACD bullish")
            else:
                reasons.append("MACD bearish")

        # Bollinger position: below lower = buy, above upper = sell
        if not np.isnan(bb_pos):
            n_factors += 1
            score -= bb_pos * 0.6
            if bb_pos < -0.5:
                reasons.append("Price near lower Bollinger")
            elif bb_pos > 0.5:
                reasons.append("Price near upper Bollinger")

        # SMA crossover: short > long = bullish
        if len(close) >= 51:
            n_factors += 1
            s10, s50 = sma_10.iloc[-1], sma_50.iloc[-1]
            if s10 > s50:
                score += 0.4
                reasons.append("SMA 10 > SMA 50")
            else:
                score -= 0.4
                reasons.append("SMA 10 < SMA 50")

        # Momentum (10d): positive = bullish
        if not pd.isna(mom):
            n_factors += 1
            mom_norm = np.clip(mom / 15, -1, 1)
            score += mom_norm * 0.3
            if abs(mom) > 5:
                reasons.append(f"10d momentum {mom:+.1f}%")

        # Volume confirmation: high vol on up move = stronger signal
        if not pd.isna(vol_ratio) and vol_ratio > 1.2 and change_pct > 0:
            score += 0.2
            reasons.append("Volume confirmation")
        elif not pd.isna(vol_ratio) and vol_ratio > 1.2 and change_pct < 0:
            score -= 0.2
            reasons.append("High volume on decline")

        # Normalize score to [-1, 1]
        if n_factors > 0:
            score = np.clip(score / max(n_factors * 0.5, 1), -1, 1)
        else:
            score = 0

        # Convert to action
        if score >= 0.4:
            action = "BUY"
        elif score <= -0.4:
            action = "SELL"
        else:
            action = "HOLD"

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
        )

    except Exception as e:
        return SignalResult(
            ticker=ticker, action="HOLD", score=0, price=0, change_pct=0,
            rsi=None, macd_hist=None, bb_position=None, momentum_10d=None,
            volume_ratio=None, reasons=[],
            error=str(e)
        )


def scan_tickers(
    tickers: Optional[List[str]] = None,
    period: str = "3mo",
    filter_action: Optional[str] = None,
) -> List[Dict]:
    """
    Scan a list of tickers and return signal results.
    filter_action: "BUY" | "SELL" | None (return all)
    """
    tickers = tickers or DEFAULT_TICKERS
    results = []
    for t in tickers:
        r = analyze_ticker(t, period=period)
        d = {
            "ticker": r.ticker,
            "action": r.action,
            "score": round(r.score, 3),
            "price": r.price,
            "change_pct": round(r.change_pct, 2),
            "rsi": round(r.rsi, 1) if r.rsi is not None else None,
            "macd_hist": round(r.macd_hist, 4) if r.macd_hist is not None else None,
            "bb_position": round(r.bb_position, 2) if r.bb_position is not None else None,
            "momentum_10d": round(r.momentum_10d, 2) if r.momentum_10d is not None else None,
            "volume_ratio": round(r.volume_ratio, 2) if r.volume_ratio is not None else None,
            "reasons": r.reasons,
            "error": r.error,
        }
        if filter_action and r.action != filter_action:
            continue
        results.append(d)
    return results
