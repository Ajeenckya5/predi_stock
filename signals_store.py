"""
Signals Store - Predefined BUY / SELL Lists
==========================================
Persists scan results into predefined lists. BUY or SELL only (no HOLD).
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

_STORE_PATH = os.path.join(os.path.dirname(__file__), ".signals_store.json")


def _load_store() -> Dict[str, Dict[str, Any]]:
    if os.path.exists(_STORE_PATH):
        try:
            with open(_STORE_PATH, "r") as f:
                data = json.load(f)
            # Migrate legacy HOLD bucket into BUY/SELL by score sign
            hold = data.pop("hold", None) or {}
            for t, entry in hold.items():
                sc = entry.get("score")
                try:
                    sc = float(sc) if sc is not None else 0.0
                except (TypeError, ValueError):
                    sc = 0.0
                lk = "buy" if sc >= 0 else "sell"
                entry["action"] = "BUY" if sc >= 0 else "SELL"
                data.setdefault(lk, {})[t] = entry
            if hold:
                _save_store_raw(data)
            return {"buy": data.get("buy", {}), "sell": data.get("sell", {})}
        except Exception:
            pass
    return {"buy": {}, "sell": {}}


def _save_store_raw(store: Dict) -> None:
    try:
        with open(_STORE_PATH, "w") as f:
            json.dump(store, f, indent=2)
    except Exception:
        pass


def _save_store(store: Dict) -> None:
    _save_store_raw(store)


def add_to_lists(results: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Add scan results to predefined BUY/SELL lists.
    Each ticker is stored with latest data. Returns updated store.
    """
    store = _load_store()
    for r in results:
        ticker = str(r.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        raw = (r.get("action") or "").upper()
        if raw == "BUY":
            action = "BUY"
        elif raw == "SELL":
            action = "SELL"
        else:
            sc = r.get("score")
            try:
                sc = float(sc) if sc is not None else 0.0
            except (TypeError, ValueError):
                sc = 0.0
            action = "BUY" if sc >= 0 else "SELL"
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
        list_key = "buy" if action == "BUY" else "sell"
        store[list_key][ticker] = entry
    _save_store(store)
    return store


def get_lists() -> Dict[str, List[Dict]]:
    """Return BUY and SELL lists as lists of entries (sorted by score)."""
    store = _load_store()
    buy_list = sorted(store.get("buy", {}).values(), key=lambda x: (x.get("score") or 0), reverse=True)
    sell_list = sorted(store.get("sell", {}).values(), key=lambda x: (x.get("score") or 0))
    return {"buy": buy_list, "sell": sell_list}


def clear_lists() -> None:
    """Clear all stored lists."""
    _save_store({"buy": {}, "sell": {}})
