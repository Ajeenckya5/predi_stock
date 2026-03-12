from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List

from service import train_for_ticker, agent_predict_once_service
from scanner import scan_tickers
from ticker_data import search_tickers, get_ticker_count

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


@app.post("/scan")
def scan_endpoint(req: ScanRequest):
    try:
        results = scan_tickers(
            tickers=req.tickers,
            period=req.period,
            filter_action=req.filter_action,
        )
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/train", response_model=TrainResponse)
def train_endpoint(req: TrainRequest):
    try:
        ckpt, pkl = train_for_ticker(req.ticker, req.country)
    except Exception as e:
        # Surface pipeline failures (e.g., missing ticker data) as 400 for the client
        raise HTTPException(status_code=400, detail=str(e)) from e

    return TrainResponse(
        ticker=req.ticker.upper(),
        checkpoint_path=ckpt,
        daily_pickle_path=pkl,
    )


@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(req: AgentRequest):
    # default checkpoint if not passed explicitly
    ckpt = req.checkpoint_path or f"news_stock_mdn_{req.ticker.upper()}.pt"

    try:
        result = agent_predict_once_service(req.ticker, req.country, ckpt)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return AgentResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
