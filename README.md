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
4. Render will detect the Python app. Use:
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Deploy → Your app will be live at `https://predi-stock.onrender.com` (or your chosen name)

## API

- `GET /` — Web UI
- `POST /scan` — Scan stocks: `{"tickers": ["AAPL","MSFT"], "filter_action": "BUY"}`
- `POST /train` — Train ML model (requires NEWSAPI_KEY)
- `POST /agent` — Run ML agent for a ticker
