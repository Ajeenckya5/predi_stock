"""
News + Stock Forecaster & Agent
===============================

Modes:
  1) longterm
     - Daily OHLCV from first trading day (IPO) -> today via yfinance
     - IPO-range company / national / global news via NewsAPI
     - Same-day news aggregation
     - Save dataframe with:
         * OHLCV
         * daily log-return
         * news embeddings and counts

  2) intraday
     - Recent intraday 1m data (~30d) -> resample to 10-minute bars
     - Fetch matching news for that intraday range
     - Encode news with FinBERT
     - Build 10-min sequences
     - Train LSTM + MDN model to predict next 10-min return distribution
     - Save model checkpoint

  3) agent
     - Load trained intraday MDN checkpoint for given ticker
     - Pull recent 10-min bars (2 days)
     - Pull last 24h news (company / national / global)
     - Auto-calibrate thresholds from historical predictions:
          sell_threshold = 30th percentile of E[r]
          buy_threshold  = 70th percentile of E[r]
     - Compute E[r_{t+1}] for latest window
     - Print BUY / SELL / HOLD decision

Dependencies:
  pip install torch pandas numpy yfinance transformers requests scikit-learn

Environment:
  export NEWSAPI_KEY="YOUR_NEWSAPI_KEY"

NOTE: This is research/educational code, NOT financial advice.
"""

import os
import time
import math
import json
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

# Slightly India-biased macro/global query
GLOBAL_NEWS_QUERY = (
    "economy OR geopolitical OR war OR sanctions OR 'interest rate' "
    "OR RBI OR 'Reserve Bank of India' OR 'Nifty 50' OR Sensex OR India"
)

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "YOUR_NEWSAPI_KEY")  # set real key in env

RESAMPLE_INTERVAL = "10min"  # 10-minute bars
SEQ_LEN = 48               # sequence length
N_MIXTURES = 3             # MDN mixtures

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
TEST_SIZE = 0.2
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# NEWSAPI HELPERS
# ============================================================

def newsapi_get_everything(q, from_dt, to_dt, language="en"):
    """
    Simple wrapper for NewsAPI 'everything' endpoint.
    """
    if NEWSAPI_KEY == "YOUR_NEWSAPI_KEY":
        print("[WARN] NEWSAPI_KEY not set, returning empty news list.")
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q,
        "from": from_dt.isoformat(),
        "to": to_dt.isoformat(),
        "language": language,
        "sortBy": "relevancy",
        "pageSize": 100,
        "apiKey": NEWSAPI_KEY,
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"[WARN] NewsAPI error: {resp.status_code} {resp.text}")
        return []
    data = resp.json()
    return data.get("articles", [])


def newsapi_get_range_chunked(q, start_dt, end_dt, chunk_days=28, language="en"):
    """
    Fetch 'everything' in chunks to cover longer ranges (IPO -> today).
    Actual history depth depends on your NewsAPI plan.
    """
    all_articles = []
    cur = start_dt
    while cur < end_dt:
        chunk_end = min(cur + timedelta(days=chunk_days), end_dt)
        print(f"[INFO] Fetching news chunk for '{q}' from {cur.date()} to {chunk_end.date()}...")
        articles = newsapi_get_everything(q, cur, chunk_end, language=language)
        all_articles.extend(articles)
        cur = chunk_end
        time.sleep(1.0)  # be nice to API
    return all_articles


# ============================================================
# DAILY DATA FROM IPO
# ============================================================

def download_daily_from_ipo(ticker: str) -> pd.DataFrame:
    """
    Download daily OHLCV from the first available trading day (usually IPO)
    up to today.
    """
    print(f"[INFO] Downloading DAILY history from IPO for {ticker}...")
    yf_ticker = yf.Ticker(ticker)
    df_daily = yf_ticker.history(period="max", interval="1d")

    if df_daily.empty:
        raise RuntimeError(
            "Daily data is empty; check ticker symbol, exchange suffix, or network access to yfinance."
        )

    df_daily = df_daily.reset_index()
    if "Date" in df_daily.columns:
        df_daily["Date"] = pd.to_datetime(df_daily["Date"])
        df_daily.set_index("Date", inplace=True)
    else:
        df_daily["Date"] = pd.to_datetime(df_daily["index"])
        df_daily.set_index("Date", inplace=True)
        df_daily.drop(columns=["index"], inplace=True)

    first_day = df_daily.index.min().date()
    last_day = df_daily.index.max().date()
    print(f"[INFO] First trading day in data: {first_day}")
    print(f"[INFO] Last trading day in data : {last_day}")
    print(f"[INFO] Total trading days       : {len(df_daily)}")

    df_daily["log_return"] = np.log(df_daily["Close"] / df_daily["Close"].shift(1))

    return df_daily


def fetch_company_news_range(ticker: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    query = f'"{ticker}"'
    articles = newsapi_get_range_chunked(query, start_dt, end_dt)
    rows = []
    for a in articles:
        rows.append({
            "source": "newsapi",
            "category": "company",
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "content": a.get("content", ""),
            "published_at": a.get("publishedAt", None),
            "url": a.get("url", "")
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def fetch_national_news_range(country_code: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    # Slightly richer query for India if country_code == "in"
    if country_code.lower() == "in":
        query = (
            "India OR RBI OR 'Reserve Bank of India' OR 'Nifty 50' OR Sensex OR 'Union Budget' "
            "OR 'Lok Sabha' OR inflation OR 'interest rate'"
        )
    else:
        query = f"(economy OR inflation OR 'interest rate' OR budget OR election) AND {country_code}"

    articles = newsapi_get_range_chunked(query, start_dt, end_dt)
    rows = []
    for a in articles:
        rows.append({
            "source": "newsapi",
            "category": "national",
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "content": a.get("content", ""),
            "published_at": a.get("publishedAt", None),
            "url": a.get("url", "")
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def fetch_global_news_range(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    query = GLOBAL_NEWS_QUERY
    articles = newsapi_get_range_chunked(query, start_dt, end_dt)
    rows = []
    for a in articles:
        rows.append({
            "source": "newsapi",
            "category": "global",
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "content": a.get("content", ""),
            "published_at": a.get("publishedAt", None),
            "url": a.get("url", "")
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def _ensure_published_at(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a tz-naive datetime 'published_at' column exists, even for empty frames."""
    df = df.copy()
    if "published_at" in df.columns:
        col = pd.to_datetime(df["published_at"], errors="coerce")
        # Normalize any tz-aware values to UTC then drop tz for safe comparisons
        if hasattr(col.dt, "tz") and col.dt.tz is not None:
            col = col.dt.tz_convert("UTC").dt.tz_localize(None)
        df["published_at"] = col
    else:
        df["published_at"] = pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))
    return df


# ============================================================
# FINBERT ENCODER
# ============================================================

class FinBertEncoder:
    def __init__(self, model_name="ProsusAI/finbert"):
        print("[INFO] Loading FinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()
        self.hidden_dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode_texts(self, texts, max_length=128):
        if len(texts) == 0:
            return np.zeros((0, self.hidden_dim), dtype=np.float32)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = self.model(**enc)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (N, hidden_dim)
        return cls_emb.detach().cpu().numpy().astype(np.float32)


def build_news_embeddings(df_news: pd.DataFrame, encoder: FinBertEncoder) -> pd.DataFrame:
    if df_news.empty:
        df_news["embedding"] = [[] for _ in range(len(df_news))]
        return df_news
    texts = (df_news["title"].fillna("") + ". " + df_news["description"].fillna("")).tolist()
    embeddings = encoder.encode_texts(texts)
    df_news["embedding"] = embeddings.tolist()
    return df_news


# ============================================================
# DAILY NEWS AGGREGATION (IPO)
# ============================================================

def aggregate_daily_news(df_daily: pd.DataFrame,
                         df_company: pd.DataFrame,
                         df_national: pd.DataFrame,
                         df_global: pd.DataFrame,
                         hidden_dim=768) -> pd.DataFrame:
    """
    For each trading day, aggregate news from that calendar day (UTC).
    """
    df = df_daily.copy()
    df_company = _ensure_published_at(df_company)
    df_national = _ensure_published_at(df_national)
    df_global = _ensure_published_at(df_global)
    df["Date"] = df.index.date

    company_embs = []
    national_embs = []
    global_embs = []
    company_counts = []
    national_counts = []
    global_counts = []

    for dt in df.index:
        day_start = datetime(dt.year, dt.month, dt.day)
        day_end   = day_start + timedelta(days=1)

        def pool(df_cat):
            mask = (df_cat["published_at"] >= day_start) & (df_cat["published_at"] < day_end)
            subset = df_cat.loc[mask]
            if subset.empty:
                return np.zeros((hidden_dim,), dtype=np.float32), 0
            embs = np.stack(subset["embedding"].values, axis=0)
            return embs.mean(axis=0), len(subset)

        c_emb, c_cnt = pool(df_company)
        n_emb, n_cnt = pool(df_national)
        g_emb, g_cnt = pool(df_global)

        company_embs.append(c_emb)
        national_embs.append(n_emb)
        global_embs.append(g_emb)
        company_counts.append(c_cnt)
        national_counts.append(n_cnt)
        global_counts.append(g_cnt)

    df["company_emb"]   = company_embs
    df["national_emb"]  = national_embs
    df["global_emb"]    = global_embs
    df["company_count"] = company_counts
    df["national_count"] = national_counts
    df["global_count"]   = global_counts

    return df


def build_daily_ipo_feature_matrices(df_daily_with_news: pd.DataFrame):
    """
    Build daily feature matrices from IPO -> today, with target = next-day return.
    """
    df = df_daily_with_news.copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col + "_z"] = (df[col] - df[col].rolling(60).mean()) / (df[col].rolling(60).std() + 1e-6)

    df = df.dropna(subset=["log_return"] + [c + "_z" for c in ["Open", "High", "Low", "Close", "Volume"]])

    price_cols = ["Open_z", "High_z", "Low_z", "Close_z", "Volume_z", "log_return"]
    price_features = df[price_cols].values.astype(np.float32)

    company_embs = np.stack(df["company_emb"].values, axis=0)
    national_embs = np.stack(df["national_emb"].values, axis=0)
    global_embs = np.stack(df["global_emb"].values, axis=0)
    counts = df[["company_count", "national_count", "global_count"]].values.astype(np.float32)

    news_features = np.concatenate([company_embs, national_embs, global_embs, counts], axis=1).astype(np.float32)

    returns = df["log_return"].values.astype(np.float32)

    y = np.roll(returns, -1)
    price_features = price_features[:-1]
    news_features  = news_features[:-1]
    y = y[:-1]

    return price_features, news_features, y, df.index[:-1]


# ============================================================
# INTRADAY DATA (10-MIN)
# ============================================================

def download_intraday_10m(ticker: str) -> pd.DataFrame:
    """
    Download recent intraday 1m data (~30 days) & resample to 10-minute bars.
    """
    # yfinance occasionally returns empty frames for 1m/30d depending on the ticker or
    # current market hours. Try a few sane fallbacks (including Ticker.history) before
    # giving up.
    attempts = [
        ("5d", "1m"),
        ("30d", "1m"),
        ("60d", "2m"),
        ("60d", "5m"),
    ]

    df_intraday, last_period, last_interval = _download_intraday_with_fallback(
        ticker, attempts, label="Training"
    )

    if df_intraday.empty:
        raise RuntimeError(
            "Intraday data is empty after trying 1m/2m/5m (download/history). "
            "Possible causes: ticker not supported for intraday, market closed for an extended period, "
            "or IP throttled. Try another ticker or run during market hours. "
            f"(last tried {last_interval} for {last_period})."
        )

    df_intraday = _standardize_ohlcv(df_intraday, ticker)
    df_intraday = df_intraday.reset_index()
    # yfinance can name the datetime column either 'Datetime' or the index name.
    dt_col = "Datetime" if "Datetime" in df_intraday.columns else df_intraday.columns[0]
    df_intraday["Datetime"] = pd.to_datetime(df_intraday[dt_col], errors="coerce")
    if hasattr(df_intraday["Datetime"].dt, "tz") and df_intraday["Datetime"].dt.tz is not None:
        df_intraday["Datetime"] = df_intraday["Datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    df_intraday.set_index("Datetime", inplace=True)

    print("[INFO] Resampling to 10-minute bars...")
    df_10m = df_intraday.resample(RESAMPLE_INTERVAL).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    df_10m["log_return"] = np.log(df_10m["Close"] / df_10m["Close"].shift(1))
    df_10m = df_10m.dropna()
    return df_10m


def fetch_company_news_intraday(ticker: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    query = f'"{ticker}"'
    articles = newsapi_get_everything(query, start_dt, end_dt)
    rows = []
    for a in articles:
        rows.append({
            "source": "newsapi",
            "category": "company",
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "content": a.get("content", ""),
            "published_at": a.get("publishedAt", None),
            "url": a.get("url", "")
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def fetch_national_news_intraday(country_code: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    if country_code.lower() == "in":
        query = (
            "India OR RBI OR 'Reserve Bank of India' OR 'Nifty 50' OR Sensex OR 'Union Budget' "
            "OR 'Lok Sabha' OR inflation OR 'interest rate'"
        )
    else:
        query = f"(economy OR inflation OR 'interest rate' OR budget OR election) AND {country_code}"

    articles = newsapi_get_everything(query, start_dt, end_dt)
    rows = []
    for a in articles:
        rows.append({
            "source": "newsapi",
            "category": "national",
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "content": a.get("content", ""),
            "published_at": a.get("publishedAt", None),
            "url": a.get("url", "")
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def fetch_global_news_intraday(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    query = GLOBAL_NEWS_QUERY
    articles = newsapi_get_everything(query, start_dt, end_dt)
    rows = []
    for a in articles:
        rows.append({
            "source": "newsapi",
            "category": "global",
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "content": a.get("content", ""),
            "published_at": a.get("publishedAt", None),
            "url": a.get("url", "")
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def aggregate_news_for_10m_windows(df_price_10m: pd.DataFrame,
                                   df_company: pd.DataFrame,
                                   df_national: pd.DataFrame,
                                   df_global: pd.DataFrame,
                                   hidden_dim=768) -> pd.DataFrame:
    df_price = df_price_10m.copy().sort_index()
    df_company = _ensure_published_at(df_company)
    df_national = _ensure_published_at(df_national)
    df_global = _ensure_published_at(df_global)

    company_embs = []
    national_embs = []
    global_embs = []
    company_counts = []
    national_counts = []
    global_counts = []

    for t in df_price.index:
        t_start = t - pd.to_timedelta(RESAMPLE_INTERVAL)

        def pool(df_cat):
            mask = (df_cat["published_at"] > t_start) & (df_cat["published_at"] <= t)
            subset = df_cat.loc[mask]
            if subset.empty:
                return np.zeros((hidden_dim,), dtype=np.float32), 0
            embs = np.stack(subset["embedding"].values, axis=0)
            return embs.mean(axis=0), len(subset)

        c_emb, c_cnt = pool(df_company)
        n_emb, n_cnt = pool(df_national)
        g_emb, g_cnt = pool(df_global)

        company_embs.append(c_emb)
        national_embs.append(n_emb)
        global_embs.append(g_emb)
        company_counts.append(c_cnt)
        national_counts.append(n_cnt)
        global_counts.append(g_cnt)

    df_price["company_emb"]   = company_embs
    df_price["national_emb"]  = national_embs
    df_price["global_emb"]    = global_embs
    df_price["company_count"] = company_counts
    df_price["national_count"] = national_counts
    df_price["global_count"]   = global_counts

    return df_price


def build_intraday_feature_matrices(df_feat: pd.DataFrame):
    df = df_feat.copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col + "_z"] = (df[col] - df[col].rolling(60).mean()) / (df[col].rolling(60).std() + 1e-6)

    df = df.dropna(subset=["log_return"] + [c + "_z" for c in ["Open", "High", "Low", "Close", "Volume"]])

    price_cols = ["Open_z", "High_z", "Low_z", "Close_z", "Volume_z", "log_return"]
    price_features = df[price_cols].values.astype(np.float32)

    company_embs = np.stack(df["company_emb"].values, axis=0)
    national_embs = np.stack(df["national_emb"].values, axis=0)
    global_embs = np.stack(df["global_emb"].values, axis=0)
    counts = df[["company_count", "national_count", "global_count"]].values.astype(np.float32)

    news_features = np.concatenate([company_embs, national_embs, global_embs, counts], axis=1).astype(np.float32)

    returns = df["log_return"].values.astype(np.float32)
    y = np.roll(returns, -1)

    df = df.iloc[:-1]
    price_features = price_features[:-1]
    news_features  = news_features[:-1]
    y = y[:-1]

    return price_features, news_features, y


# ============================================================
# INTRADAY DATA HELPERS
# ============================================================

def _download_intraday_with_fallback(ticker: str, attempts, label: str):
    """
    Try a sequence of (period, interval) combos using both yf.download and
    Ticker.history (they sometimes diverge in availability). Returns the first
    non-empty dataframe and the last attempted (period, interval).
    """
    df = pd.DataFrame()
    last_period, last_interval = None, None

    for period, interval in attempts:
        last_period, last_interval = period, interval
        print(f"[INFO] {label} download ({interval}, {period}) for {ticker}...")

        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            progress=False,
            prepost=True,
        )
        if not df.empty:
            break

        # yfinance sometimes succeeds via Ticker().history when download() fails
        print("[DEBUG] download() empty; trying Ticker.history...")
        df = yf.Ticker(ticker).history(
            period=period,
            interval=interval,
            prepost=True,
        )
        if not df.empty:
            break

    return df, last_period, last_interval


def _standardize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize yfinance output to have flat columns: Open, High, Low, Close, Volume.
    Handles MultiIndex columns and casing differences. Raises with a clear message
    if the required columns are still missing.
    """
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        # Try to slice by any level that matches the ticker (order can vary)
        for level in range(df.columns.nlevels):
            if ticker in df.columns.get_level_values(level):
                df = df.xs(ticker, level=level, axis=1)
                break

        # If still MultiIndex, flatten by picking the first string-like part
        if isinstance(df.columns, pd.MultiIndex):
            flat_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    candidate = next((c for c in col if isinstance(c, str) and c), None)
                    if candidate is None:
                        candidate = "_".join([str(c) for c in col if c is not None])
                    flat_cols.append(candidate)
                else:
                    flat_cols.append(col)
            df = df.copy()
            df.columns = flat_cols

    rename_map = {}
    for c in df.columns:
        lc = c.lower().strip() if isinstance(c, str) else c
        if lc in {"open", "high", "low", "close", "volume"}:
            rename_map[c] = lc.capitalize()
        elif lc == "adj close":
            rename_map[c] = "Close"

    df = df.rename(columns=rename_map)

    expected = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Intraday OHLCV missing columns: {missing}. Raw columns: {list(df.columns)}"
        )
    return df


# ============================================================
# DATASET & MODEL
# ============================================================

class NewsStockDataset(Dataset):
    def __init__(self, price_features, news_features, targets, seq_len=SEQ_LEN):
        self.price_features = price_features
        self.news_features  = news_features
        self.targets        = targets
        self.seq_len        = seq_len
        assert len(price_features) == len(news_features) == len(targets)
        self.n = len(targets)

    def __len__(self):
        return max(self.n - self.seq_len, 0)

    def __getitem__(self, idx):
        start = idx
        end   = idx + self.seq_len
        price_seq = self.price_features[start:end]
        news_seq  = self.news_features[start:end]
        target    = self.targets[end - 1]
        return {
            "price_seq": torch.from_numpy(price_seq).float(),
            "news_seq":  torch.from_numpy(news_seq).float(),
            "target":    torch.tensor(target, dtype=torch.float32),
        }


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return h_n[-1]


class MDNHead(nn.Module):
    def __init__(self, input_dim, n_mixtures=N_MIXTURES):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, n_mixtures * 3),
        )

    def forward(self, x):
        out = self.fc(x)
        K = self.n_mixtures
        pi, mu, log_sigma = out[:, :K], out[:, K:2*K], out[:, 2*K:3*K]
        pi = torch.softmax(pi, dim=-1)
        sigma = torch.exp(log_sigma) + 1e-6
        return pi, mu, sigma


def mdn_loss(pi, mu, sigma, y):
    y = y.unsqueeze(1)
    normal = torch.distributions.Normal(loc=mu, scale=sigma)
    log_probs = normal.log_prob(y)
    max_log_prob, _ = torch.max(log_probs, dim=-1, keepdim=True)
    log_sum_exp = max_log_prob + torch.log(torch.sum(pi * torch.exp(log_probs - max_log_prob), dim=-1, keepdim=True))
    nll = -log_sum_exp.mean()
    return nll


class NewsStockMDN(nn.Module):
    def __init__(self, price_dim, news_dim, hidden_dim=128, n_mixtures=N_MIXTURES):
        super().__init__()
        self.price_encoder = LSTMEncoder(price_dim, hidden_dim)
        self.news_encoder  = LSTMEncoder(news_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.mdn = MDNHead(2 * hidden_dim, n_mixtures=n_mixtures)

    def forward(self, price_seq, news_seq):
        h_price = self.price_encoder(price_seq)
        h_news  = self.news_encoder(news_seq)
        fused   = torch.cat([h_price, h_news], dim=-1)
        fused   = self.fusion(fused)
        pi, mu, sigma = self.mdn(fused)
        return pi, mu, sigma


def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if len(train_loader.dataset) == 0:
        raise RuntimeError("Training dataset is empty; need more intraday bars or smaller SEQ_LEN.")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            price_seq = batch["price_seq"].to(device)
            news_seq  = batch["news_seq"].to(device)
            target    = batch["target"].to(device)

            optimizer.zero_grad()
            pi, mu, sigma = model(price_seq, news_seq)
            loss = mdn_loss(pi, mu, sigma, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * price_seq.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        if val_loader is not None and len(val_loader.dataset) > 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    price_seq = batch["price_seq"].to(device)
                    news_seq  = batch["news_seq"].to(device)
                    target    = batch["target"].to(device)
                    pi, mu, sigma = model(price_seq, news_seq)
                    loss = mdn_loss(pi, mu, sigma, target)
                    total_val_loss += loss.item() * price_seq.size(0)

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            print(f"[EPOCH {epoch}/{epochs}] train_loss={avg_train_loss:.6f}  val_loss={avg_val_loss:.6f}")
        else:
            print(f"[EPOCH {epoch}/{epochs}] train_loss={avg_train_loss:.6f}  val_loss=NA (no validation data)")

    return model


# ============================================================
# LONG-TERM PIPELINE (IPO → TODAY, DAILY)
# ============================================================

def run_longterm_pipeline(ticker: str, country_code: str):
    df_daily = download_daily_from_ipo(ticker)
    ipo_start = df_daily.index.min().to_pydatetime()
    ipo_end   = df_daily.index.max().to_pydatetime()
    print(f"[INFO] LONGTERM: {ticker} daily range {ipo_start.date()} → {ipo_end.date()}")

    encoder = FinBertEncoder()

    print("[INFO] Fetching IPO-range company news...")
    df_company_daily = fetch_company_news_range(ticker, ipo_start, ipo_end)
    df_company_daily = build_news_embeddings(df_company_daily, encoder)
    print(f"[INFO] company news articles: {len(df_company_daily)}")

    print("[INFO] Fetching IPO-range national news...")
    df_national_daily = fetch_national_news_range(country_code, ipo_start, ipo_end)
    df_national_daily = build_news_embeddings(df_national_daily, encoder)
    print(f"[INFO] national news articles: {len(df_national_daily)}")

    print("[INFO] Fetching IPO-range global news...")
    df_global_daily = fetch_global_news_range(ipo_start, ipo_end)
    df_global_daily = build_news_embeddings(df_global_daily, encoder)
    print(f"[INFO] global news articles: {len(df_global_daily)}")

    df_daily_with_news = aggregate_daily_news(
        df_daily, df_company_daily, df_national_daily, df_global_daily,
        hidden_dim=encoder.hidden_dim
    )

    daily_pkl_path = f"{ticker}_daily_ipo_with_news.pkl"
    df_daily_with_news.to_pickle(daily_pkl_path)
    print(f"[INFO] Saved IPO daily+news dataframe to {daily_pkl_path}")

    daily_price_features, daily_news_features, daily_y, daily_idx = \
        build_daily_ipo_feature_matrices(df_daily_with_news)
    print("[INFO] Daily feature shapes (IPO):")
    print("    price_features:", daily_price_features.shape)
    print("    news_features :", daily_news_features.shape)
    print("    y             :", daily_y.shape)

    summary = {
        "ticker": ticker,
        "ipo_start": ipo_start.date().isoformat(),
        "ipo_end": ipo_end.date().isoformat(),
        "n_days": int(len(df_daily_with_news)),
        "company_news": int(df_daily_with_news["company_count"].sum()),
        "national_news": int(df_daily_with_news["national_count"].sum()),
        "global_news": int(df_daily_with_news["global_count"].sum()),
    }

    return daily_pkl_path, summary


# ============================================================
# INTRADAY PIPELINE (10-MIN MODEL)
# ============================================================

def run_intraday_pipeline(ticker: str, country_code: str):
    df_10m = download_intraday_10m(ticker)
    intraday_start = df_10m.index.min().to_pydatetime()
    intraday_end   = df_10m.index.max().to_pydatetime()
    print(f"[INFO] INTRADAY: {ticker} 10-min range {intraday_start} → {intraday_end}")
    print(f"[INFO] 10-min intraday shape: {df_10m.shape}")

    encoder = FinBertEncoder()

    print("[INFO] Fetching intraday company news...")
    df_company_intraday = fetch_company_news_intraday(ticker, intraday_start, intraday_end)
    df_company_intraday = build_news_embeddings(df_company_intraday, encoder)
    print(f"[INFO] company articles: {len(df_company_intraday)}")

    print("[INFO] Fetching intraday national news...")
    df_national_intraday = fetch_national_news_intraday(country_code, intraday_start, intraday_end)
    df_national_intraday = build_news_embeddings(df_national_intraday, encoder)
    print(f"[INFO] national articles: {len(df_national_intraday)}")

    print("[INFO] Fetching intraday global news...")
    df_global_intraday = fetch_global_news_intraday(intraday_start, intraday_end)
    df_global_intraday = build_news_embeddings(df_global_intraday, encoder)
    print(f"[INFO] global articles: {len(df_global_intraday)}")

    df_10m_feat = aggregate_news_for_10m_windows(
        df_10m, df_company_intraday, df_national_intraday, df_global_intraday,
        hidden_dim=encoder.hidden_dim
    )

    price_features, news_features, y = build_intraday_feature_matrices(df_10m_feat)
    print("[INFO] Intraday feature shapes:")
    print("    price_features:", price_features.shape)
    print("    news_features :", news_features.shape)
    print("    y             :", y.shape)

    n_samples = len(y)
    if n_samples <= SEQ_LEN:
        raise RuntimeError(
            f"Not enough intraday samples after preprocessing: {n_samples} <= SEQ_LEN {SEQ_LEN}. "
            "Try a longer period/interval (edit attempts list) or wait for more bars."
        )

    Xp_train, Xp_val, Xn_train, Xn_val, y_train, y_val = train_test_split(
        price_features, news_features, y,
        test_size=TEST_SIZE,
        shuffle=False
    )

    train_dataset = NewsStockDataset(Xp_train, Xn_train, y_train, seq_len=SEQ_LEN)
    val_dataset   = NewsStockDataset(Xp_val,  Xn_val,  y_val,   seq_len=SEQ_LEN)

    if len(train_dataset) <= 0:
        raise RuntimeError(
            f"Not enough data to build training sequences (train={len(train_dataset)}, seq_len={SEQ_LEN}). "
            "Increase data length or reduce SEQ_LEN."
        )

    if len(val_dataset) <= 0:
        print(
            f"[WARN] Validation set too small (len={len(val_dataset)}). Using all data for training and skipping validation."
        )
        train_dataset = NewsStockDataset(price_features, news_features, y, seq_len=SEQ_LEN)
        val_loader = None
    else:
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    price_dim = price_features.shape[1]
    news_dim  = news_features.shape[1]

    model = NewsStockMDN(price_dim=price_dim, news_dim=news_dim, hidden_dim=128, n_mixtures=N_MIXTURES)
    print("[INFO] Training 10-min MDN model...")
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE)

    checkpoint_path = f"news_stock_mdn_{ticker}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "price_dim": price_dim,
        "news_dim": news_dim,
        "hidden_dim": 128,
        "n_mixtures": N_MIXTURES,
        "seq_len": SEQ_LEN
    }, checkpoint_path)
    print(f"[INFO] Model saved to {checkpoint_path}")

    summary = {
        "ticker": ticker,
        "intraday_start": intraday_start.isoformat(),
        "intraday_end": intraday_end.isoformat(),
        "n_bars": int(len(df_10m)),
    }

    return checkpoint_path, summary


# ============================================================
# AGENT UTILITIES (AUTO-THRESHOLD)
# ============================================================

def get_recent_10m_bars(ticker: str, days: int = 2):
    # Reuse the same fallback strategy as training to make agent mode resilient.
    attempts = [
        (f"{days}d", "1m"),
        (f"{days}d", "2m"),
        ("7d", "1m"),
        ("30d", "2m"),
        ("30d", "5m"),
        ("60d", "5m"),
        ("60d", "15m"),
    ]

    df, last_period, last_interval = _download_intraday_with_fallback(
        ticker, attempts, label="Agent"
    )

    if df.empty:
        raise RuntimeError(
            "Intraday data is empty for agent after fallbacks (download/history); yfinance limit or market closed "
            f"(last tried {last_interval} for {last_period})."
        )

    df = _standardize_ohlcv(df, ticker)
    df = df.reset_index()
    dt_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df["Datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
    if hasattr(df["Datetime"].dt, "tz") and df["Datetime"].dt.tz is not None:
        df["Datetime"] = df["Datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    df.set_index("Datetime", inplace=True)

    df_10m = df.resample(RESAMPLE_INTERVAL).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    df_10m["log_return"] = np.log(df_10m["Close"] / df_10m["Close"].shift(1))
    df_10m = df_10m.dropna()
    return df_10m


def aggregate_news_for_windows_agent(df_price_10m, df_company, df_national, df_global, encoder_hidden_dim):
    return aggregate_news_for_10m_windows(
        df_price_10m,
        _ensure_published_at(df_company),
        _ensure_published_at(df_national),
        _ensure_published_at(df_global),
        hidden_dim=encoder_hidden_dim,
    )


def build_feature_matrices_for_history(df_feat: pd.DataFrame):
    df = df_feat.copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col + "_z"] = (df[col] - df[col].rolling(60).mean()) / (df[col].rolling(60).std() + 1e-6)

    df = df.dropna(subset=["log_return"] + [c + "_z" for c in ["Open", "High", "Low", "Close", "Volume"]])

    price_cols = ["Open_z", "High_z", "Low_z", "Close_z", "Volume_z", "log_return"]
    price_features = df[price_cols].values.astype(np.float32)

    company_embs = np.stack(df["company_emb"].values, axis=0)
    national_embs = np.stack(df["national_emb"].values, axis=0)
    global_embs = np.stack(df["global_emb"].values, axis=0)
    counts = df[["company_count", "national_count", "global_count"]].values.astype(np.float32)

    news_features = np.concatenate([company_embs, national_embs, global_embs, counts], axis=1).astype(np.float32)
    returns = df["log_return"].values.astype(np.float32)
    return price_features, news_features, returns, df.index


def build_features_last_window(df_feat: pd.DataFrame):
    price_features, news_features, returns, idx = build_feature_matrices_for_history(df_feat)
    n = len(price_features)
    if n == 0:
        raise RuntimeError("No 10-min bars available after preprocessing.")

    if n < SEQ_LEN:
        # Pad from the front by repeating the first observation to reach SEQ_LEN
        pad_len = SEQ_LEN - n
        print(f"[WARN] Only {n} 10-min bars available; padding with {pad_len} copies of first bar to reach SEQ_LEN={SEQ_LEN}.")
        first_price = price_features[0:1]
        first_news  = news_features[0:1]
        price_features = np.concatenate([np.repeat(first_price, pad_len, axis=0), price_features], axis=0)
        news_features  = np.concatenate([np.repeat(first_news,  pad_len, axis=0), news_features],  axis=0)

    price_seq = price_features[-SEQ_LEN:]
    news_seq  = news_features[-SEQ_LEN:]
    return price_seq, news_seq


def calibrate_thresholds_from_history(df_feat: pd.DataFrame, model: NewsStockMDN):
    """
    Calibrate sell/buy thresholds from historical expected returns.

    - If not enough history -> use small default thresholds.
    - Otherwise -> use 30th percentile as SELL, 70th percentile as BUY.
    """
    price_features, news_features, returns, idx = build_feature_matrices_for_history(df_feat)
    n = len(price_features)

    if n < SEQ_LEN + 10:
        print("[WARN] Not enough history to calibrate; using small default thresholds.")
        # ~0.02% move up/down as a tiny threshold
        sell_th = -0.0002
        buy_th  =  0.0002
        print(f"[INFO] Default thresholds: sell={sell_th:.6f}, buy={buy_th:.6f}")
        return sell_th, buy_th

    preds = []
    model.eval()
    with torch.no_grad():
        for end in range(SEQ_LEN, n - 1):
            start = end - SEQ_LEN
            price_seq = price_features[start:end]
            news_seq  = news_features[start:end]

            price_seq_t = torch.from_numpy(price_seq).unsqueeze(0).float().to(DEVICE)
            news_seq_t  = torch.from_numpy(news_seq).unsqueeze(0).float().to(DEVICE)

            pi, mu, sigma = model(price_seq_t, news_seq_t)
            exp_ret = torch.sum(pi * mu, dim=-1).cpu().numpy()[0]
            preds.append(exp_ret)

    preds = np.array(preds, dtype=np.float32)
    q_low  = np.quantile(preds, 0.3)
    q_high = np.quantile(preds, 0.7)

    sell_th = q_low
    buy_th  = q_high

    print(f"[INFO] Calibrated thresholds from history:")
    print(f"       30th percentile (sell) = {sell_th:.6f}")
    print(f"       70th percentile (buy)  = {buy_th:.6f}")

    return sell_th, buy_th


def agent_predict_once(ticker: str, country_code: str, checkpoint_path: str):
    # load model
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    price_dim = ckpt["price_dim"]
    news_dim  = ckpt["news_dim"]
    hidden_dim = ckpt["hidden_dim"]
    n_mixtures = ckpt["n_mixtures"]

    model = NewsStockMDN(price_dim, news_dim, hidden_dim, n_mixtures)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # recent 10-min bars; try a few day windows to ensure enough bars
    df_10m = None
    for days in [2, 5, 10, 15, 30, 60]:
        try:
            df_10m_candidate = get_recent_10m_bars(ticker, days=days)
            if len(df_10m_candidate) >= SEQ_LEN + 1:
                df_10m = df_10m_candidate
                break
        except RuntimeError as e:
            print(f"[WARN] Agent fetch failed for {days}d window: {e}")
            continue

    if df_10m is None:
        raise RuntimeError(
            f"Not enough 10-min bars for SEQ_LEN={SEQ_LEN} even after trying up to 15d. "
            "Wait for more market data or reduce SEQ_LEN."
        )

    encoder = FinBertEncoder()

    # fetch recent news (24h)
    now = datetime.utcnow()
    start_dt = now - timedelta(hours=24)

    df_company = fetch_company_news_intraday(ticker, start_dt, now)
    df_company = build_news_embeddings(df_company, encoder)

    df_national = fetch_national_news_intraday(country_code, start_dt, now)
    df_national = build_news_embeddings(df_national, encoder)

    df_global = fetch_global_news_intraday(start_dt, now)
    df_global = build_news_embeddings(df_global, encoder)

    df_feat = aggregate_news_for_windows_agent(
        df_10m, df_company, df_national, df_global, encoder_hidden_dim=encoder.hidden_dim
    )

    # Calibrate thresholds using historical predictions
    sell_th, buy_th = calibrate_thresholds_from_history(df_feat, model)

    # Build last window for prediction
    price_seq, news_seq = build_features_last_window(df_feat)

    with torch.no_grad():
        price_seq_t = torch.from_numpy(price_seq).unsqueeze(0).float().to(DEVICE)
        news_seq_t  = torch.from_numpy(news_seq).unsqueeze(0).float().to(DEVICE)
        pi, mu, sigma = model(price_seq_t, news_seq_t)
        exp_return = torch.sum(pi * mu, dim=-1).item()

    print(f"[INFO] Latest expected return E[r_(t+1)] = {exp_return:.6f}")
    print(f"[INFO] Decision thresholds: sell<={sell_th:.6f}, buy>={buy_th:.6f}")

    if exp_return >= buy_th:
        action = "BUY"
    elif exp_return <= sell_th:
        action = "SELL"
    else:
        action = "HOLD"

    result = {
        "ticker": ticker,
        "action": action,
        "expected_return": exp_return,
        "buy_threshold": buy_th,
        "sell_threshold": sell_th,
        "last_bar": df_10m.index[-1].isoformat(),
        "bars_used": len(df_10m),
    }
    return result


# ============================================================
# MAIN CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="News + Stock Forecaster & Agent")
    parser.add_argument("--mode", type=str,
                        choices=["longterm", "intraday", "agent"],
                        required=True,
                        help="longterm = IPO→today daily pipeline, intraday = 10-min MDN training, agent = run trading agent once")
    parser.add_argument("--ticker", type=str, required=True,
                        help="Ticker symbol (e.g., RELIANCE.NS, TCS.NS, AAPL)")
    parser.add_argument("--country", type=str, default="in",
                        help="Country code for national news (e.g., in, us, gb)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for agent mode (default: news_stock_mdn_<TICKER>.pt)")
    return parser.parse_args()


def main():
    args = parse_args()
    ticker  = args.ticker.upper()
    country = args.country

    if args.mode == "longterm":
        daily_path, summary = run_longterm_pipeline(ticker, country)
        print("\n=== LONG-TERM SUMMARY ===")
        print(json.dumps(summary, indent=2))
        print(f"Daily IPO+news data saved at: {daily_path}")

    elif args.mode == "intraday":
        checkpoint_path, summary = run_intraday_pipeline(ticker, country)
        print("\n=== INTRADAY SUMMARY ===")
        print(json.dumps(summary, indent=2))
        print(f"Checkpoint saved at: {checkpoint_path}")

    else:  # agent
        checkpoint_path = args.checkpoint or f"news_stock_mdn_{ticker}.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found. Run intraday mode first.")
        res = agent_predict_once(ticker, country, checkpoint_path)
        print("\n=== AGENT RESULT ===")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
