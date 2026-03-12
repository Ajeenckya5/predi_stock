"""
Stock Ticker Database & Search
==============================
Fetches S&P 500 + NASDAQ + Nifty 50 from Wikipedia, provides search by symbol or name.
"""

import pandas as pd
from typing import List, Dict
import os

_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".ticker_cache.csv")


def _fetch_sp500() -> pd.DataFrame:
    """Fetch S&P 500 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        dfs = pd.read_html(url)
        df = dfs[0][["Symbol", "Security"]].rename(columns={"Symbol": "symbol", "Security": "name"})
        df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
        return df
    except Exception:
        return pd.DataFrame(columns=["symbol", "name"])




# Static fallback: Nifty 50 + popular US tickers (when Wikipedia fails)
FALLBACK_TICKERS = [
    ("AAPL", "Apple Inc."), ("MSFT", "Microsoft"), ("GOOGL", "Alphabet Google"), ("AMZN", "Amazon"),
    ("NVDA", "NVIDIA"), ("META", "Meta Platforms"), ("TSLA", "Tesla"), ("JPM", "JPMorgan Chase"),
    ("V", "Visa"), ("WMT", "Walmart"), ("JNJ", "Johnson & Johnson"), ("PG", "Procter & Gamble"),
    ("UNH", "UnitedHealth"), ("HD", "Home Depot"), ("DIS", "Walt Disney"), ("MA", "Mastercard"),
    ("BAC", "Bank of America"), ("XOM", "Exxon Mobil"), ("CVX", "Chevron"), ("ABBV", "AbbVie"),
    ("PEP", "PepsiCo"), ("KO", "Coca-Cola"), ("COST", "Costco"), ("MRK", "Merck"),
    ("RELIANCE.NS", "Reliance Industries"), ("TCS.NS", "Tata Consultancy"), ("HDFCBANK.NS", "HDFC Bank"),
    ("INFY.NS", "Infosys"), ("HINDUNILVR.NS", "Hindustan Unilever"), ("ICICIBANK.NS", "ICICI Bank"),
    ("SBIN.NS", "State Bank of India"), ("BHARTIARTL.NS", "Bharti Airtel"), ("ITC.NS", "ITC"),
    ("WIPRO.NS", "Wipro"), ("TATAMOTORS.NS", "Tata Motors"), ("MARUTI.NS", "Maruti Suzuki"),
]


def load_all_tickers() -> pd.DataFrame:
    """Load full ticker list: S&P 500 + NASDAQ-100 + Nifty 50, with cache."""
    if os.path.exists(_CACHE_PATH):
        try:
            return pd.read_csv(_CACHE_PATH)
        except Exception:
            pass

    dfs = []
    sp = _fetch_sp500()
    if not sp.empty:
        dfs.append(sp)

    fallback = pd.DataFrame(FALLBACK_TICKERS, columns=["symbol", "name"])
    if dfs:
        df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["symbol"], keep="first")
        df = pd.concat([df, fallback]).drop_duplicates(subset=["symbol"], keep="first")
    else:
        df = fallback

    try:
        df.to_csv(_CACHE_PATH, index=False)
    except Exception:
        pass

    return df


def search_tickers(query: str, limit: int = 20) -> List[Dict[str, str]]:
    """
    Search tickers by symbol or company name.
    Returns list of {symbol, name} matching the query.
    """
    if not query or len(query.strip()) < 1:
        return []

    q = query.strip().upper()
    df = load_all_tickers()

    # Match symbol or name (case-insensitive)
    mask_sym = df["symbol"].astype(str).str.upper().str.contains(q, regex=False, na=False)
    mask_name = df["name"].astype(str).str.upper().str.contains(q, regex=False, na=False)
    matches = df[mask_sym | mask_name].head(limit)

    return [{"symbol": r["symbol"], "name": r["name"]} for _, r in matches.iterrows()]


def get_ticker_count() -> int:
    """Return total number of tickers in database."""
    return len(load_all_tickers())
