from datetime import datetime, timedelta
from typing import Dict, Tuple

import torch
import numpy as np
import pandas as pd

from Stock import (
    # core classes & helpers
    FinBertEncoder,
    NewsStockDataset,
    NewsStockMDN,
    train_model,

    # daily / IPO pipeline
    download_daily_from_ipo,
    fetch_company_news_range,
    fetch_national_news_range,
    fetch_global_news_range,
    aggregate_daily_news,
    build_daily_ipo_feature_matrices,

    # intraday pipeline
    download_intraday_10m,
    fetch_company_news_intraday,
    fetch_national_news_intraday,
    fetch_global_news_intraday,
    aggregate_news_for_10m_windows,
    build_intraday_feature_matrices,

    # agent utilities
    get_recent_10m_bars,
    aggregate_news_for_windows_agent,
    build_features_last_window,
    calibrate_thresholds_from_history,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 48


def train_for_ticker(ticker: str, country_code: str) -> Tuple[str, str]:
    """
    Run the FULL pipeline for a ticker:
      1) IPO->today daily + same-day news (save .pkl)
      2) Recent 10-min intraday + news
      3) Train MDN model, save checkpoint

    Returns:
      (checkpoint_path, daily_ipo_pickle_path)
    """
    ticker = ticker.upper()

    # === DAILY IPO PART ===
    df_daily = download_daily_from_ipo(ticker)
    ipo_start = df_daily.index.min().to_pydatetime()
    ipo_end = df_daily.index.max().to_pydatetime()

    encoder = FinBertEncoder()

    df_company_daily = fetch_company_news_range(ticker, ipo_start, ipo_end)
    df_company_daily = build_news_embeddings(df_company_daily, encoder)

    df_national_daily = fetch_national_news_range(country_code, ipo_start, ipo_end)
    df_national_daily = build_news_embeddings(df_national_daily, encoder)

    df_global_daily = fetch_global_news_range(ipo_start, ipo_end)
    df_global_daily = build_news_embeddings(df_global_daily, encoder)

    df_daily_with_news = aggregate_daily_news(
        df_daily, df_company_daily, df_national_daily, df_global_daily,
        hidden_dim=encoder.hidden_dim,
    )
    daily_pkl_path = f"{ticker}_daily_ipo_with_news.pkl"
    df_daily_with_news.to_pickle(daily_pkl_path)

    # === INTRADAY 10-MIN PART ===
    df_10m = download_intraday_10m(ticker)
    intraday_start = df_10m.index.min().to_pydatetime()
    intraday_end = df_10m.index.max().to_pydatetime()

    df_company_intraday = fetch_company_news_intraday(ticker, intraday_start, intraday_end)
    df_company_intraday = build_news_embeddings(df_company_intraday, encoder)

    df_national_intraday = fetch_national_news_intraday(country_code, intraday_start, intraday_end)
    df_national_intraday = build_news_embeddings(df_national_intraday, encoder)

    df_global_intraday = fetch_global_news_intraday(intraday_start, intraday_end)
    df_global_intraday = build_news_embeddings(df_global_intraday, encoder)

    df_10m_feat = aggregate_news_for_10m_windows(
        df_10m, df_company_intraday, df_national_intraday, df_global_intraday,
        hidden_dim=encoder.hidden_dim,
    )

    price_features, news_features, y = build_intraday_feature_matrices(df_10m_feat)

    from sklearn.model_selection import train_test_split
    Xp_train, Xp_val, Xn_train, Xn_val, y_train, y_val = train_test_split(
        price_features, news_features, y,
        test_size=0.2,
        shuffle=False,
    )

    from torch.utils.data import DataLoader
    train_dataset = NewsStockDataset(Xp_train, Xn_train, y_train, seq_len=SEQ_LEN)
    val_dataset   = NewsStockDataset(Xp_val,  Xn_val,  y_val,   seq_len=SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, drop_last=True)

    price_dim = price_features.shape[1]
    news_dim  = news_features.shape[1]

    model = NewsStockMDN(price_dim=price_dim, news_dim=news_dim, hidden_dim=128, n_mixtures=3)
    model = train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device=DEVICE)

    checkpoint_path = f"news_stock_mdn_{ticker}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "price_dim": price_dim,
        "news_dim": news_dim,
        "hidden_dim": 128,
        "n_mixtures": 3,
        "seq_len": SEQ_LEN,
    }, checkpoint_path)

    return checkpoint_path, daily_pkl_path


def agent_predict_once_service(ticker: str, country_code: str, checkpoint_path: str) -> Dict:
    """
    Thin wrapper around the agent logic → returns a dict ready for your UI.
    """
    ticker = ticker.upper()

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

    # recent 10-min bars
    df_10m = get_recent_10m_bars(ticker, days=2)
    encoder = FinBertEncoder()

    now = datetime.utcnow()
    start_dt = now - timedelta(hours=24)

    # reuse the intraday fetchers for recent news
    df_company = fetch_company_news_intraday(ticker, start_dt, now)
    df_company = build_news_embeddings(df_company, encoder)

    df_national = fetch_national_news_intraday(country_code, start_dt, now)
    df_national = build_news_embeddings(df_national, encoder)

    df_global = fetch_global_news_intraday(start_dt, now)
    df_global = build_news_embeddings(df_global, encoder)

    df_feat = aggregate_news_for_windows_agent(
        df_10m, df_company, df_national, df_global, encoder_hidden_dim=encoder.hidden_dim,
    )

    # NOTE: in my updated big script, calibrate_thresholds_from_history returns (sell_th, buy_th)
    sell_th, buy_th = calibrate_thresholds_from_history(df_feat, model)

    price_seq, news_seq = build_features_last_window(df_feat)

    with torch.no_grad():
        price_seq_t = torch.from_numpy(price_seq).unsqueeze(0).float().to(DEVICE)
        news_seq_t  = torch.from_numpy(news_seq).unsqueeze(0).float().to(DEVICE)
        pi, mu, sigma = model(price_seq_t, news_seq_t)
        exp_return = torch.sum(pi * mu, dim=-1).item()

    if exp_return >= buy_th:
        action = "BUY"
    elif exp_return <= sell_th:
        action = "SELL"
    else:
        action = "HOLD"

    return {
        "ticker": ticker,
        "action": action,
        "expected_return": exp_return,
        "buy_threshold": buy_th,
        "sell_threshold": sell_th,
        "last_bar": df_10m.index[-1].isoformat(),
        "bars_used": len(df_10m),
    }