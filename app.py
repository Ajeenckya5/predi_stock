from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List

from scanner import scan_tickers, INDICATOR_CATALOG
from ticker_data import search_tickers, get_ticker_count, get_nifty50_tickers, load_all_tickers
from signals_store import add_to_lists, get_lists, clear_lists
from rlhf import record_feedback, get_stats, reset_weights
from chart_analysis import build_chart_payload

# Run with: python app.py (uses uvicorn under the hood)
# Or manually: uvicorn app:app --reload --host 0.0.0.0 --port 8000

app = FastAPI(title="predi_stock | Stock Buy & Sell Signals")


class TrainRequest(BaseModel):
    ticker: str
    country: str = "in"


class TrainResponse(BaseModel):
    ticker: str
    checkpoint_path: str
    daily_pickle_path: str


class AgentRequest(BaseModel):
    ticker: str
    country: str = "in"
    checkpoint_path: Optional[str] = None


class AgentResponse(BaseModel):
    ticker: str
    action: str
    expected_return: float
    buy_threshold: float
    sell_threshold: float
    last_bar: str
    bars_used: int


class ScanRequest(BaseModel):
    tickers: Optional[List[str]] = None
    period: str = "3mo"
    filter_action: Optional[str] = None
    universe: Optional[str] = None  # "nifty50" | "all" = all India tickers in DB
    save_to_lists: bool = False  # add results to predefined BUY/SELL lists
    # Subset of indicator ids for composite score (see GET /indicators/catalog). None = use all.
    indicators: Optional[List[str]] = None


@app.get("/")
def serve_index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/search")
def search_endpoint(q: str = "", limit: int = 20):
    """Search stocks by symbol or company name."""
    results = search_tickers(q, limit=limit)
    return {"tickers": results}


@app.get("/tickers/count")
def ticker_count_endpoint():
    """Return total tickers in database."""
    return {"count": get_ticker_count()}


@app.get("/tickers/nifty50")
def nifty50_endpoint():
    """Return all Nifty 50 tickers for scan."""
    return {"tickers": get_nifty50_tickers(), "count": 50}


@app.get("/indicators/catalog")
def indicators_catalog_endpoint():
    """Ids and labels for manual indicator selection (scoring)."""
    return {"indicators": INDICATOR_CATALOG}


@app.get("/chart/{ticker}")
def chart_live_endpoint(ticker: str, period: str = "6mo", interval: str = "1d"):
    """
    OHLCV candles + pattern markers & S/R lines for the live chart UI.
    """
    payload = build_chart_payload(ticker, period=period, interval=interval)
    err = payload.get("error")
    if err:
        raise HTTPException(status_code=400, detail=str(err))
    return payload


@app.get("/lists")
def lists_endpoint():
    """Return predefined BUY / SELL lists from stored scan results."""
    return get_lists()


@app.delete("/lists")
def clear_lists_endpoint():
    """Clear stored BUY/SELL lists."""
    clear_lists()
    return {"status": "cleared"}


class FeedbackRequest(BaseModel):
    ticker: str
    action: str  # BUY | SELL
    was_correct: bool
    factors_used: Optional[List[str]] = None


@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):
    """RLHF: Record if prediction was correct. Algo improves from feedback."""
    record_feedback(
        ticker=req.ticker,
        action=req.action.upper(),
        was_correct=req.was_correct,
        factors_present=req.factors_used,
    )
    return get_stats()


@app.get("/feedback/stats")
def feedback_stats_endpoint():
    """Return RLHF stats: feedback count, accuracy, learned weights."""
    return get_stats()


@app.post("/feedback/reset")
def feedback_reset_endpoint():
    """Reset RLHF weights to defaults."""
    return {"weights": reset_weights()}


@app.post("/scan")
def scan_endpoint(req: ScanRequest):
    try:
        tickers = req.tickers
        if not tickers:
            if req.universe == "nifty50":
                tickers = [t["symbol"] for t in get_nifty50_tickers()]
            elif req.universe == "all":
                df = load_all_tickers()
                tickers = df["symbol"].astype(str).tolist()
        results = scan_tickers(
            tickers=tickers,
            period=req.period,
            filter_action=req.filter_action,
            indicators=req.indicators,
        )
        if req.save_to_lists and results:
            add_to_lists(results)
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/train", response_model=TrainResponse)
def train_endpoint(req: TrainRequest):
    try:
        from service import train_for_ticker
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="ML training requires torch/transformers. Use local install with full requirements.txt"
        )
    try:
        ckpt, pkl = train_for_ticker(req.ticker, req.country)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return TrainResponse(
        ticker=req.ticker.upper(),
        checkpoint_path=ckpt,
        daily_pickle_path=pkl,
    )


@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(req: AgentRequest):
    try:
        from service import agent_predict_once_service
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="ML agent requires torch/transformers. Use local install with full requirements.txt"
        )
    ckpt = req.checkpoint_path or f"news_stock_mdn_{req.ticker.upper()}.pt"
    try:
        result = agent_predict_once_service(req.ticker, req.country, ckpt)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return AgentResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
