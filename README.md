# Stock Signal Scanner (predi_stock)

A **stock buy & sell signal scanner** using **technical analysis** (RSI, MACD, Bollinger Bands, momentum) with an optional **ML-based agent** for predictions. Includes a web UI and REST API.

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

Open **http://localhost:8000**

---

## Deploy on Render

1. Push the repo to GitHub and connect it to [Render](https://render.com) (New → Web Service).
2. Use:
   - **Build:** `pip install -r requirements-render.txt` (lightweight; scanner only)
   - **Start:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
3. Deploy. Example live URL: `https://predi-stock.onrender.com`

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
