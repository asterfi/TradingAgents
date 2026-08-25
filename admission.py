"""ta-shadow: deterministic live-risk admission governor (2026-08-25 hardening).

THE rule (operator-accepted policy, Codex review §"Priority 0"):
no LLM output becomes a live order until this module says `accepted`.

Enforced immediately before EVERY placement, from FRESH MEXC state:

  1. max 3 concurrent positions (open positions + pending/resting entries)
  2. <= 15% aggregate planned stop-risk (day-start capital base)
  3. one position or entry per ticker (same-ticker skip)
  4. unknown state = NO orders (an API error is never "flat")
  5. post-rounding risk re-check (min-volume/precision rounding can inflate
     the effective loss at stop past the 5% per-trade cap)
  6. post-fill protection reconciliation (order placed != success until the
     position AND its attached TP/SL are confirmed)

Every decision is persisted to admission_log.jsonl with a machine-readable
reason: accepted | max_positions | aggregate_risk | ticker_already_open |
state_unknown | rounded_risk_exceeded | protection_unconfirmed.
"""

import json
import os
from datetime import datetime, timezone

LANE = os.path.dirname(os.path.abspath(__file__))
ADMISSION_LOG = os.path.join(LANE, "admission_log.jsonl")

from symbols import TICKER_TO_MEXC, TICKER_FROM_MEXC

MAX_POSITIONS = 3                  # operator policy: 3 concurrent, not 5
MAX_AGGREGATE_RISK_PCT = 0.15      # 15% of day-start capital across positions+entries
PER_TRADE_RISK_PCT = 0.05          # 5% per trade (operator policy)
RISK_TOLERANCE = 0.001             # 0.1% relative numerical tolerance on caps
OUR_SYMBOLS = set(TICKER_TO_MEXC.values())


# --------------------------------------------------------------------------
# Typed live-state fetch — empty and error are DIFFERENT states
# --------------------------------------------------------------------------

class LiveStateError(RuntimeError):
    """MEXC state could not be confirmed. Never interpret as 'flat'."""


def _mexc(method, path, params=None, body=None):
    from mexc_orders import _mexc_request
    return _mexc_request(method, path, params, body)


def fetch_live_state():
    """Positions + resting entry orders on our universe.

    Returns {"ok": True, "positions": {sym: {...}}, "orders": {sym: {...}}}.
    Raises LiveStateError on ANY failure (fail closed — case 5: a MEXC
    timeout, malformed response, or auth error must produce zero new orders).
    """
    try:
        resp = _mexc("GET", "/api/v1/private/position/open_positions", {})
        rows = resp.get("data") or []
        if not isinstance(rows, list):
            raise LiveStateError(f"open_positions: malformed data ({type(rows).__name__})")
        if not resp.get("success", True):
            raise LiveStateError(f"open_positions: {resp.get('message')}")
    except LiveStateError:
        raise
    except Exception as e:
        raise LiveStateError(f"open_positions failed: {e}") from e

    positions = {}
    for p in rows:
        sym = p.get("symbol")
        if sym not in OUR_SYMBOLS:
            continue
        try:
            vol = float(p.get("holdVol") or 0)
        except (TypeError, ValueError):
            raise LiveStateError(f"position {sym}: unparseable holdVol {p.get('holdVol')!r}")
        if vol > 0:
            positions[sym] = {
                "side": "LONG" if p.get("positionType") == 1 else "SHORT",
                "entry": float(p.get("holdAvgPrice") or 0),
                "qty": vol,
                "leverage": p.get("leverage"),
            }

    try:
        resp = _mexc("GET", "/api/v1/private/order/list/open_orders",
                     {"page_num": 1, "page_size": 100})
        rows = resp.get("data") or []
        if not isinstance(rows, list):
            raise LiveStateError(f"open_orders: malformed data ({type(rows).__name__})")
    except LiveStateError:
        raise
    except Exception as e:
        raise LiveStateError(f"open_orders failed: {e}") from e

    orders = {}
    for o in rows:
        sym = o.get("symbol")
        if sym not in OUR_SYMBOLS:
            continue
        orders.setdefault(sym, []).append({
            "order_id": o.get("orderId"),
            "side": o.get("side"),   # 1=open long, 3=open short (MEXC futures)
            "vol": o.get("vol"),
            "price": o.get("price"),
        })
    return {"ok": True, "positions": positions, "orders": orders}


def risk_book_of_orders():
    """Planned risk ($ at stop) for our own staged/placed brackets, keyed by
    ticker, from orders.json — the deterministic record of what we intended
    to risk. Entries not ours (manual) carry no planned-risk entry and are
    conservatively charged at the full per-trade cap by the caller.
    """
    from mexc_orders import load_orders
    out = {}
    try:
        for o in load_orders().get("orders", []):
            if o.get("status") == "placed" and o.get("order_id"):
                r = o.get("risk_usd")
                if isinstance(r, (int, float)) and r > 0:
                    out.setdefault(o.get("ticker"), r)
    except Exception:
        return {}
    return out


def day_start_capital():
    """Today's fixed capital base (stock_daily._day_start_capital).

    In LIVE mode a failed equity read must NOT fall back to a default —
    the governor treats an unreadable base as unknown state (fail closed).
    """
    from stock_daily import _load_day_capital
    from datetime import datetime as _dt
    snap = _load_day_capital()
    today = _dt.now(timezone.utc).strftime("%Y-%m-%d")
    if snap and snap.get("date") == today and snap.get("equity"):
        return float(snap["equity"])
    return None


# --------------------------------------------------------------------------
# The admission gate
# --------------------------------------------------------------------------

def decide(ticker, planned_risk_usd, state=None, day_capital=None,
           quiet=False):
    """Admission decision for one NEW entry on `ticker`.

    Returns {"decision": <reason>, "accepted": bool, "detail": {...}}.
    `state`: result of fetch_live_state() (fetch fresh if omitted).
    `day_capital`: day-start capital (looked up if omitted; unknown => reject).
    """
    from stock_daily import _day_start_capital

    def rej(reason, **detail):
        return {"decision": reason, "accepted": False, "detail": detail}

    # -- 4. unknown state fails closed ------------------------------------
    if state is None:
        try:
            state = fetch_live_state()
        except LiveStateError as e:
            return rej("state_unknown", error=str(e)[:200])
    positions = state.get("positions") or {}
    orders = state.get("orders") or {}

    mexc_sym = TICKER_TO_MEXC.get(ticker)
    if not mexc_sym:
        return rej("state_unknown", error=f"unknown ticker {ticker}")

    # -- capital base (5% and 15% are relative to day-START capital) -------
    if day_capital is None or day_capital <= 0:
        try:
            day_capital = _day_start_capital()
        except Exception as e:
            day_capital = None
        if not day_capital or day_capital <= 0:
            return rej("state_unknown", error="day-start capital unavailable")

    cap_risk = day_capital * MAX_AGGREGATE_RISK_PCT
    per_trade_cap = day_capital * PER_TRADE_RISK_PCT

    if not isinstance(planned_risk_usd, (int, float)) or planned_risk_usd <= 0:
        return rej("rounded_risk_exceeded",
                   error=f"invalid planned risk {planned_risk_usd!r}",
                   per_trade_cap=round(per_trade_cap, 2))

    # -- 3. same-ticker exclusion -----------------------------------------
    if mexc_sym in positions:
        return rej("ticker_already_open", ticker=ticker,
                   open_position=positions[mexc_sym])
    if mexc_sym in orders and orders[mexc_sym]:
        return rej("ticker_already_open", ticker=ticker,
                   resting_orders=len(orders[mexc_sym]))

    # -- 1. three-position cap (positions + pending entries) --------------
    existing = len(positions) + len([s for s, lst in orders.items() if lst])
    if existing >= MAX_POSITIONS:
        return rej("max_positions", existing=existing, max=MAX_POSITIONS)

    # -- 2. 15% aggregate risk --------------------------------------------
    risk_book = risk_book_of_orders()
    open_agg = 0.0
    for sym in positions:
        tk = TICKER_FROM_MEXC[sym]
        open_agg += risk_book.get(tk, per_trade_cap)  # unknown manual position: full cap charge
    for sym, lst in orders.items():
        tk = TICKER_FROM_MEXC[sym]
        open_agg += risk_book.get(tk, per_trade_cap)
    projected = open_agg + planned_risk_usd
    if projected > cap_risk * (1 + RISK_TOLERANCE):
        return rej("aggregate_risk",
                   open_risk=round(open_agg, 2),
                   projected=round(projected, 2),
                   cap=round(cap_risk, 2))

    # -- 5. per-trade post-rounding risk ----------------------------------
    if planned_risk_usd > per_trade_cap * (1 + RISK_TOLERANCE):
        return rej("rounded_risk_exceeded",
                   planned=round(planned_risk_usd, 2),
                   cap=round(per_trade_cap, 2))

    return {"decision": "accepted", "accepted": True,
            "detail": {"existing": existing,
                       "open_risk": round(open_agg, 2),
                       "projected": round(projected, 2),
                       "cap": round(cap_risk, 2)}}


def log_admission(ticker, decision, extra=None):
    row = {"ts": datetime.now(timezone.utc).isoformat(),
           "ticker": ticker, **decision, "extra": extra or {}}
    try:
        with open(ADMISSION_LOG, "a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass
    return row


# --------------------------------------------------------------------------
# Post-fill protection reconciliation
# --------------------------------------------------------------------------

def confirm_protection(mexc_symbol, timeout_s=30.0, poll=2.0):
    """Verify a just-filled position exists AND carries attached TP/SL.

    Returns {"ok": True, "position": {...}, "tp": x, "sl": y} on success,
    or {"ok": False, "reason": ...}. Reasons: no_position (fill not
    confirmed), no_protection (position exists but TP/SL not attached),
    state_error (lookup failed — treated as protection_unconfirmed upstream).
    A submitted order is NOT success until this passes (case 10).
    """
    import time as _time
    deadline = _time.monotonic() + timeout_s
    last_err = None
    while _time.monotonic() < deadline:
        try:
            resp = _mexc("GET", "/api/v1/private/position/open_positions", {})
            if not resp.get("success", True):
                raise RuntimeError(resp.get("message"))
            for p in (resp.get("data") or []):
                if p.get("symbol") != mexc_symbol:
                    continue
                if not float(p.get("holdVol") or 0) > 0:
                    continue
                tp = p.get("takeProfitPrice")
                sl = p.get("stopLossPrice")
                if not tp or not sl:
                    return {"ok": False, "reason": "no_protection",
                            "position": {k: p.get(k) for k in
                                         ("holdVol", "holdAvgPrice", "positionType")}}
                return {"ok": True, "position": {k: p.get(k) for k in
                            ("holdVol", "holdAvgPrice", "positionType")},
                        "tp": float(tp), "sl": float(sl)}
            # limit order may still be resting (not filled) — keep polling
        except Exception as e:
            last_err = str(e)[:200]
        _time.sleep(poll)
    return {"ok": False, "reason": "state_error" if last_err else "no_position",
            "error": last_err}
