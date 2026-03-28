"""
Signals Store - Predefined BUY / SELL / HOLD Lists
==================================================
Persists scan results into predefined lists. Scan all stocks and add to lists.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

_STORE_PATH = os.path.join(os.path.dirname(__file__), ".signals_store.json")


def _load_store() -> Dict[str, Dict[str, Any]]:
    """Load {ticker -> {action, score, ...}} for each list."""
    if os.path.exists(_STORE_PATH):
        try:
            with open(_STORE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"buy": {}, "sell": {}, "hold": {}}


def _save_store(store: Dict) -> None:
    try:
        with open(_STORE_PATH, "w") as f:
            json.dump(store, f, indent=2)
    except Exception:
        pass


def add_to_lists(results: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Add scan results to predefined BUY/SELL/HOLD lists.
    Each ticker is stored with latest data. Returns updated store.
    """
    store = _load_store()
    for r in results:
        ticker = str(r.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        action = (r.get("action") or "HOLD").upper()
        entry = {
            "ticker": ticker,
            "action": action,
            "score": r.get("score"),
            "price": r.get("price"),
            "change_pct": r.get("change_pct"),
            "rsi": r.get("rsi"),
            "stop_loss": r.get("stop_loss"),
            "take_profit": r.get("take_profit"),
            "reasons": r.get("reasons", []),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        list_key = "buy" if action == "BUY" else "sell" if action == "SELL" else "hold"
        store[list_key][ticker] = entry
    _save_store(store)
    return store


def get_lists() -> Dict[str, List[Dict]]:
    """Return BUY, SELL, HOLD lists as lists of entries (sorted by score desc for BUY, asc for SELL)."""
    store = _load_store()
    buy_list = sorted(store.get("buy", {}).values(), key=lambda x: (x.get("score") or 0), reverse=True)
    sell_list = sorted(store.get("sell", {}).values(), key=lambda x: (x.get("score") or 0))
    hold_list = list(store.get("hold", {}).values())
    return {"buy": buy_list, "sell": sell_list, "hold": hold_list}


def clear_lists() -> None:
    """Clear all stored lists."""
    _save_store({"buy": {}, "sell": {}, "hold": {}})
