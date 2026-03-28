"""
Stock Ticker Database & Search
==============================
Fetches S&P 500, Nifty 50, Nifty Next 50, Nifty 500 from Wikipedia.
Provides search by symbol or company name.
"""

import pandas as pd
from typing import List, Dict
import os

_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".ticker_cache.csv")


def _nse_symbol(sym: str) -> str:
    """Ensure NSE symbol has .NS suffix for yfinance."""
    s = str(sym).strip().upper()
    if not s.endswith(".NS") and not s.endswith(".BO"):
        return s + ".NS"
    return s


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


def _fetch_nifty50() -> pd.DataFrame:
    """Fetch Nifty 50 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/NIFTY_50"
        dfs = pd.read_html(url)
        for d in dfs:
            if "Symbol" in d.columns and "Company name" in d.columns:
                df = d[["Symbol", "Company name"]].rename(
                    columns={"Symbol": "symbol", "Company name": "name"}
                )
                df["symbol"] = df["symbol"].astype(str).str.strip().apply(_nse_symbol)
                return df.dropna(subset=["symbol", "name"])
        return pd.DataFrame(columns=["symbol", "name"])
    except Exception:
        return pd.DataFrame(columns=["symbol", "name"])


def _fetch_nifty_next50() -> pd.DataFrame:
    """Fetch Nifty Next 50 from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/NIFTY_Next_50"
        dfs = pd.read_html(url)
        for d in dfs:
            if "Symbol" in d.columns:
                name_col = next(
                    (c for c in d.columns if "company" in c.lower() or "name" in c.lower()),
                    d.columns[1] if len(d.columns) > 1 else d.columns[0],
                )
                df = d[["Symbol", name_col]].rename(
                    columns={"Symbol": "symbol", name_col: "name"}
                )
                df["symbol"] = df["symbol"].astype(str).str.strip().apply(_nse_symbol)
                return df.dropna(subset=["symbol", "name"])
        return pd.DataFrame(columns=["symbol", "name"])
    except Exception:
        return pd.DataFrame(columns=["symbol", "name"])


def _fetch_bse_sensex30() -> pd.DataFrame:
    """Fetch BSE Sensex 30 from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/BSE_SENSEX"
        dfs = pd.read_html(url)
        for d in dfs:
            cols = [str(c).lower() for c in d.columns]
            if any("company" in c or "name" in c for c in cols) and any("symbol" in c or "code" in c or "ticker" in c for c in cols):
                sym_col = next((c for c in d.columns if "symbol" in str(c).lower() or "code" in str(c).lower() or "ticker" in str(c).lower()), d.columns[0])
                name_col = next((c for c in d.columns if "company" in str(c).lower() or "name" in str(c).lower()), d.columns[1] if len(d.columns) > 1 else d.columns[0])
                df = d[[sym_col, name_col]].rename(columns={sym_col: "symbol", name_col: "name"})
                df["symbol"] = df["symbol"].astype(str).str.strip().apply(_nse_symbol)
                if len(df) >= 25:
                    return df.dropna(subset=["symbol", "name"])
        return pd.DataFrame(columns=["symbol", "name"])
    except Exception:
        return pd.DataFrame(columns=["symbol", "name"])


def _fetch_russell1000_sample() -> pd.DataFrame:
    """Fetch Russell 1000 sample from Wikipedia (top holdings)."""
    try:
        url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
        dfs = pd.read_html(url)
        for d in dfs:
            if len(d) > 50 and ("Symbol" in d.columns or "Ticker" in d.columns):
                sym_col = "Symbol" if "Symbol" in d.columns else "Ticker"
                name_col = next((c for c in d.columns if "company" in str(c).lower() or "name" in str(c).lower()), d.columns[1])
                df = d[[sym_col, name_col]].rename(columns={sym_col: "symbol", name_col: "name"})
                df["symbol"] = df["symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
                return df.dropna(subset=["symbol", "name"])
        return pd.DataFrame(columns=["symbol", "name"])
    except Exception:
        return pd.DataFrame(columns=["symbol", "name"])


# BSE Sensex 30 fallback (many overlap with Nifty)
BSE_SENSEX30_FALLBACK = [
    ("RELIANCE.NS", "Reliance"), ("TCS.NS", "TCS"), ("HDFCBANK.NS", "HDFC Bank"),
    ("INFY.NS", "Infosys"), ("ICICIBANK.NS", "ICICI Bank"), ("HINDUNILVR.NS", "HUL"),
    ("SBIN.NS", "SBI"), ("BHARTIARTL.NS", "Bharti Airtel"), ("ITC.NS", "ITC"),
    ("KOTAKBANK.NS", "Kotak"), ("LT.NS", "L&T"), ("AXISBANK.NS", "Axis Bank"),
    ("ASIANPAINT.NS", "Asian Paints"), ("MARUTI.NS", "Maruti"), ("HCLTECH.NS", "HCL Tech"),
    ("WIPRO.NS", "Wipro"), ("TATAMOTORS.NS", "Tata Motors"), ("SUNPHARMA.NS", "Sun Pharma"),
    ("BAJFINANCE.NS", "Bajaj Finance"), ("NESTLEIND.NS", "Nestle"), ("TITAN.NS", "Titan"),
    ("ULTRACEMCO.NS", "UltraTech"), ("TATASTEEL.NS", "Tata Steel"), ("POWERGRID.NS", "Power Grid"),
]

# Russell 1000 top US names (when Wikipedia fails)
RUSSELL1000_SAMPLE = [
    ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("GOOGL", "Alphabet"), ("AMZN", "Amazon"),
    ("NVDA", "NVIDIA"), ("META", "Meta"), ("BRK-B", "Berkshire"), ("JPM", "JPMorgan"),
    ("V", "Visa"), ("UNH", "UnitedHealth"), ("JNJ", "Johnson & Johnson"), ("PG", "P&G"),
    ("MA", "Mastercard"), ("HD", "Home Depot"), ("XOM", "ExxonMobil"), ("CVX", "Chevron"),
    ("PEP", "PepsiCo"), ("KO", "Coca-Cola"), ("COST", "Costco"), ("LLY", "Eli Lilly"),
    ("WMT", "Walmart"), ("MCD", "McDonald's"), ("ABBV", "AbbVie"), ("MRK", "Merck"),
    ("AVGO", "Broadcom"), ("TMO", "Thermo Fisher"), ("ORCL", "Oracle"), ("DIS", "Disney"),
    ("CRM", "Salesforce"), ("ADBE", "Adobe"), ("NEE", "NextEra"), ("ACN", "Accenture"),
]


def _fetch_nifty500() -> pd.DataFrame:
    """Fetch Nifty 500 (or NIFTY 200) from Wikipedia if available."""
    urls = [
        "https://en.wikipedia.org/wiki/NIFTY_500",
        "https://en.wikipedia.org/wiki/CNX_Nifty_200",
    ]
    for url in urls:
        try:
            dfs = pd.read_html(url)
            for d in dfs:
                cols = list(d.columns)
                sym_col = next((c for c in cols if "symbol" in str(c).lower() or "ticker" in str(c).lower()), None)
                name_col = next((c for c in cols if "company" in str(c).lower() or "security" in str(c).lower() or "name" in str(c).lower()), cols[1] if len(cols) > 1 else cols[0])
                if sym_col and len(d) > 20:
                    df = d[[sym_col, name_col]].rename(columns={sym_col: "symbol", name_col: "name"})
                    df["symbol"] = df["symbol"].astype(str).str.strip().apply(_nse_symbol)
                    return df.dropna(subset=["symbol", "name"])
        except Exception:
            continue
    return pd.DataFrame(columns=["symbol", "name"])


# Static fallback: full Nifty 50 + Nifty Next 50 + popular US (when Wikipedia fails)
INDIAN_FALLBACK = [
    ("RELIANCE.NS", "Reliance Industries"), ("TCS.NS", "Tata Consultancy"), ("HDFCBANK.NS", "HDFC Bank"),
    ("INFY.NS", "Infosys"), ("HINDUNILVR.NS", "Hindustan Unilever"), ("ICICIBANK.NS", "ICICI Bank"),
    ("SBIN.NS", "State Bank of India"), ("BHARTIARTL.NS", "Bharti Airtel"), ("ITC.NS", "ITC"),
    ("KOTAKBANK.NS", "Kotak Mahindra Bank"), ("LT.NS", "Larsen & Toubro"), ("AXISBANK.NS", "Axis Bank"),
    ("ASIANPAINT.NS", "Asian Paints"), ("MARUTI.NS", "Maruti Suzuki"), ("HCLTECH.NS", "HCL Technologies"),
    ("WIPRO.NS", "Wipro"), ("TATAMOTORS.NS", "Tata Motors"), ("SUNPHARMA.NS", "Sun Pharma"),
    ("BAJFINANCE.NS", "Bajaj Finance"), ("NESTLEIND.NS", "Nestle India"), ("TITAN.NS", "Titan"),
    ("ULTRACEMCO.NS", "UltraTech Cement"), ("TATASTEEL.NS", "Tata Steel"), ("POWERGRID.NS", "Power Grid"),
    ("ONGC.NS", "ONGC"), ("NTPC.NS", "NTPC"), ("INDUSINDBK.NS", "IndusInd Bank"), ("M&M.NS", "Mahindra & Mahindra"),
    ("BAJAJFINSV.NS", "Bajaj Finserv"), ("ADANIPORTS.NS", "Adani Ports"), ("ADANIENT.NS", "Adani Enterprises"),
    ("TATACONSUM.NS", "Tata Consumer"), ("CIPLA.NS", "Cipla"), ("COALINDIA.NS", "Coal India"),
    ("DRREDDY.NS", "Dr Reddy's"), ("EICHERMOT.NS", "Eicher Motors"), ("GRASIM.NS", "Grasim"),
    ("HDFCLIFE.NS", "HDFC Life"), ("HINDALCO.NS", "Hindalco"), ("JSWSTEEL.NS", "JSW Steel"),
    ("TECHM.NS", "Tech Mahindra"), ("APOLLOHOSP.NS", "Apollo Hospitals"), ("SBILIFE.NS", "SBI Life"),
    ("BEL.NS", "Bharat Electronics"), ("TRENT.NS", "Trent"), ("INDIGO.NS", "IndiGo"),
    ("DIVISLAB.NS", "Divi's Labs"), ("BRITANNIA.NS", "Britannia"), ("HEROMOTOCO.NS", "Hero MotoCorp"),
    ("PIDILITIND.NS", "Pidilite"), ("DABUR.NS", "Dabur"), ("BAJAJ-AUTO.NS", "Bajaj Auto"),
    ("HINDCOPPER.NS", "Hindustan Copper"), ("ABB.NS", "ABB India"), ("AMBUJACEM.NS", "Ambuja Cements"),
    ("BANKBARODA.NS", "Bank of Baroda"), ("BHEL.NS", "BHEL"), ("BPCL.NS", "BPCL"),
    ("HINDALCO.NS", "Hindalco"), ("IOC.NS", "Indian Oil"), ("MUTHOOTFIN.NS", "Muthoot Finance"),
    ("PEL.NS", "Piramal Enterprises"), ("TORNTPHARM.NS", "Torrent Pharma"), ("VEDL.NS", "Vedanta"),
    ("LALPATHLAB.NS", "Dr Lal PathLabs"), ("AUROPHARMA.NS", "Aurobindo Pharma"), ("BATAINDIA.NS", "Bata India"),
    ("COLPAL.NS", "Colgate Palmolive"), ("HAVELLS.NS", "Havells"), ("ICICIPRULI.NS", "ICICI Prudential"),
    ("SIEMENS.NS", "Siemens India"), ("SRF.NS", "SRF"), ("TATAPOWER.NS", "Tata Power"),
    ("UBL.NS", "United Breweries"), ("VOLTAS.NS", "Voltas"), ("BALKRISIND.NS", "Balkrishna Ind"),
]

US_FALLBACK = [
    ("AAPL", "Apple Inc."), ("MSFT", "Microsoft"), ("GOOGL", "Alphabet"), ("AMZN", "Amazon"),
    ("NVDA", "NVIDIA"), ("META", "Meta"), ("TSLA", "Tesla"), ("JPM", "JPMorgan Chase"),
    ("V", "Visa"), ("WMT", "Walmart"), ("JNJ", "Johnson & Johnson"), ("PG", "Procter & Gamble"),
]

FALLBACK_TICKERS = INDIAN_FALLBACK + US_FALLBACK


def load_all_tickers() -> pd.DataFrame:
    """Load full ticker list: S&P 500 + Nifty 50 + Nifty Next 50 + Nifty 500, with cache."""
    if os.path.exists(_CACHE_PATH):
        try:
            return pd.read_csv(_CACHE_PATH)
        except Exception:
            pass

    dfs = []
    sp = _fetch_sp500()
    if not sp.empty:
        dfs.append(sp)

    n50 = _fetch_nifty50()
    if not n50.empty:
        dfs.append(n50)

    nn50 = _fetch_nifty_next50()
    if not nn50.empty:
        dfs.append(nn50)

    bse30 = _fetch_bse_sensex30()
    if not bse30.empty:
        dfs.append(bse30)

    russell = _fetch_russell1000_sample()
    if not russell.empty:
        dfs.append(russell)

    n500 = _fetch_nifty500()
    if not n500.empty and len(n500) > 50:
        dfs.append(n500)

    fallback = pd.DataFrame(
        FALLBACK_TICKERS + BSE_SENSEX30_FALLBACK + RUSSELL1000_SAMPLE,
        columns=["symbol", "name"],
    )
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


def get_nifty50_tickers() -> List[Dict[str, str]]:
    """Return all Nifty 50 tickers for API. Uses scanner's canonical list."""
    from scanner import NIFTY50
    names = {
        "ADANIENT.NS": "Adani Enterprises", "ADANIPORTS.NS": "Adani Ports",
        "APOLLOHOSP.NS": "Apollo Hospitals", "ASIANPAINT.NS": "Asian Paints",
        "AXISBANK.NS": "Axis Bank", "BAJAJ-AUTO.NS": "Bajaj Auto",
        "BAJFINANCE.NS": "Bajaj Finance", "BAJAJFINSV.NS": "Bajaj Finserv",
        "BEL.NS": "Bharat Electronics", "BHARTIARTL.NS": "Bharti Airtel",
        "CIPLA.NS": "Cipla", "COALINDIA.NS": "Coal India",
        "DRREDDY.NS": "Dr Reddy's", "EICHERMOT.NS": "Eicher Motors",
        "ETERNAL.NS": "Eternal", "GRASIM.NS": "Grasim",
        "HCLTECH.NS": "HCL Tech", "HDFCBANK.NS": "HDFC Bank",
        "HDFCLIFE.NS": "HDFC Life", "HINDALCO.NS": "Hindalco",
        "HINDUNILVR.NS": "Hindustan Unilever", "ICICIBANK.NS": "ICICI Bank",
        "INDIGO.NS": "IndiGo", "INFY.NS": "Infosys", "ITC.NS": "ITC",
        "JIOFIN.NS": "Jio Financial", "JSWSTEEL.NS": "JSW Steel",
        "KOTAKBANK.NS": "Kotak Bank", "LT.NS": "Larsen & Toubro",
        "M&M.NS": "Mahindra & Mahindra", "MARUTI.NS": "Maruti Suzuki",
        "MAXHEALTH.NS": "Max Healthcare", "NESTLEIND.NS": "Nestle India",
        "NTPC.NS": "NTPC", "ONGC.NS": "ONGC", "POWERGRID.NS": "Power Grid",
        "RELIANCE.NS": "Reliance Industries", "SBILIFE.NS": "SBI Life",
        "SBIN.NS": "State Bank of India", "SHRIRAMFIN.NS": "Shriram Finance",
        "SUNPHARMA.NS": "Sun Pharma", "TCS.NS": "Tata Consultancy",
        "TATACONSUM.NS": "Tata Consumer", "TATAMOTORS.NS": "Tata Motors",
        "TATASTEEL.NS": "Tata Steel", "TECHM.NS": "Tech Mahindra",
        "TITAN.NS": "Titan", "TRENT.NS": "Trent",
        "ULTRACEMCO.NS": "UltraTech Cement", "WIPRO.NS": "Wipro",
    }
    return [{"symbol": s, "name": names.get(s, s.replace(".NS", ""))} for s in NIFTY50]
