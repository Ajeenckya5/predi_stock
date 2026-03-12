# predi_stock

Stock buy & sell signal scanner using technical analysis (RSI, MACD, Bollinger Bands, momentum).

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:8000

## Deploy live (Render)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Use:
   - **Build:** `pip install -r requirements-render.txt` (lightweight, scanner only)
   - **Start:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Deploy → App live at `https://predi-stock.onrender.com`

Note: `requirements-render.txt` skips torch/transformers for fast, low-memory deploy. Scan & news work. For ML train/agent, run locally with full `requirements.txt`.

## API

- `GET /` — Web UI
- `POST /scan` — Scan stocks: `{"tickers": ["AAPL","MSFT"], "filter_action": "BUY"}`
- `POST /train` — Train ML model (requires NEWSAPI_KEY)
- `POST /agent` — Run ML agent for a ticker
