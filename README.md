# Stock Signal Scanner (predi_stock)

A **buy/sell signal scanner for Indian (NSE) equities** using **technical analysis** (RSI, MACD, Bollinger Bands, momentum) with an optional **ML-based agent** for predictions. Includes a web UI and REST API. Signals are **BUY or SELL only** (no HOLD).

---

## Features

- **Technical signals** — RSI, MACD, Bollinger Bands, momentum filters
- **Web UI** — Browse and filter tickers by action (BUY/SELL)
- **REST API** — Programmatic scan, train, and agent endpoints
- **Deployable** — Runs locally or on Render with a lightweight build option

---

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open **http://localhost:8000** (only on your machine while the server is running).

---

## Production on Render (public URL)

The **public site** is whatever URL Render assigns in the dashboard (for example `https://predi-stock.onrender.com` if you named the service that way). It is **not** localhost.

1. Push this repo to GitHub (already: `Ajeenckya5/predi_stock`).
2. In [Render](https://dashboard.render.com): **New** → **Blueprint** (uses `render.yaml`) **or** **Web Service** → connect `predi_stock`.
3. If you use **Web Service** manually, set:
   - **Build:** `pip install -r requirements-render.txt`
   - **Start:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Optional: **Environment** → add `NEWSAPI_KEY` for ML `/train` / `/agent` (scanner uses Yahoo Finance without it).
5. Open the **URL** shown at the top of your service page (Render → **predi_stock** → copy link).

`requirements-render.txt` skips heavy deps (e.g. torch/transformers) for fast, low-memory deploy. Scan and news work; for ML train/agent, run locally with full `requirements.txt`.

---

## API

| Endpoint    | Method | Description                    |
|------------|--------|--------------------------------|
| `/`        | GET    | Web UI                        |
| `/scan`    | POST   | Scan stocks; body: `{"tickers": ["AAPL","MSFT"], "filter_action": "BUY"}` |
| `/train`   | POST   | Train ML model (requires `NEWSAPI_KEY`) |
| `/agent`   | POST   | Run ML agent for a ticker      |

---

## Tech stack

Python · Streamlit / FastAPI · yfinance · Technical indicators · Optional: PyTorch, Transformers, News API
