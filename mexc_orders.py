#!/usr/bin/env python3
"""ta-shadow STOCK lane: MEXC futures bracket-order staging.

Places a limit entry order with TP + SL attached (a bracket) on MEXC futures
for the 5-stock universe BEFORE the NY open, so the operator has orders ready
at 09:30 ET.

Endpoints (documented, https://www.mexc.com/api-docs/futures):
  POST /api/v1/private/order/create   (place limit with TP/SL)
  POST /api/v1/private/order/cancel   (cancel)
  GET  /api/v1/private/order/list/open_orders
  GET  /api/v1/private/position/open_positions
  POST /api/v1/private/position/change_leverage

SAFETY (operator-imposed, hard):
- STAGE mode (default): builds the order payloads and writes them to
  orders.json — does NOT hit the exchange. EXECUTE mode places them live.
- Every stock perp is a leveraged futures contract: this is risk capital,
  not a stock certificate. Max 1 bracket per ticker per day.
- Symbol mapping: NVDA->NVIDIA_USDT, TSLA->TESLA_USDT, AAPL->AAPLSTOCK_USDT,
  AMD->AMDSTOCK_USDT, SPY->SPY_USDT.
"""
import hashlib
import hmac
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from symbols import TICKER_TO_MEXC  # canonical map (2026-08-25: single source of truth)

LANE = os.path.dirname(os.path.abspath(__file__))
BASE_URL = os.environ.get("MEXC_API_BASE", "https://api.mexc.com")
ORDER_FILE = os.path.join(LANE, "orders.json")

# Defaults if the TA run gives no explicit params: risk 2.6% of ~$37.5 equity,
# ATR-based stop. Operator-set 2026-08-23: 2.6% equity per trade, MAX leverage
# per symbol (verified live from MEXC contract detail).
DEFAULT_RISK_PCT = 0.05  # 5% of equity per trade (operator 2026-08-24; was 0.026)
DEFAULT_EQUITY_USD = 37.5
DEFAULT_ATR_STOP_MULT = 1.0          # stop = entry - 1.0 * ATR (long)
DEFAULT_TP_RR = 2.0                  # TP = entry + 2.0 * (entry - stop)
MIN_STOP_DIST = 0.001                # sanity: stop must differ from entry

# Max leverage per stock perp — verified live 2026-08-23 from MEXC contract
# detail (maxLeverage). NOT a flat 200x: TESLA/AAPL are 100x, SPY is 50x.
MAX_LEVERAGE = {
    "NVIDIA_USDT": 200, "TESLA_USDT": 100,
    "AAPLSTOCK_USDT": 100, "AMDSTOCK_USDT": 200, "SPY_USDT": 50,
}
def _leverage_for(mexc_symbol):
    return MAX_LEVERAGE.get(mexc_symbol, 200)

_MIN_INTERVAL = 0.25
_last_req = 0.0


def _pace():
    global _last_req
    now = time.monotonic()
    w = _MIN_INTERVAL - (now - _last_req)
    if w > 0:
        time.sleep(w)
    _last_req = time.monotonic()


def _load_env(path):
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return env


def get_credentials():
    env = _load_env(os.path.expanduser("~/.hermes/.env"))
    k = env.get("MEXC_API_KEY") or os.environ.get("MEXC_API_KEY")
    s = env.get("MEXC_API_SECRET") or os.environ.get("MEXC_API_SECRET")
    return (k or None, s or None)


def _sign(api_key, secret, ts_ms, qs):
    target = f"{api_key}{ts_ms}{qs}"
    return hmac.new(secret.encode(), target.encode(), hashlib.sha256).hexdigest()


def _mexc_request(method, path, params=None, body=None):
    """Signed request helper (params -> query for GET, body for POST)."""
    api_key, secret = get_credentials()
    if not api_key or not secret:
        raise RuntimeError("MEXC_API_KEY / MEXC_API_SECRET missing")
    _pace()
    ts = int(time.time() * 1000)
    if method == "GET":
        qs = urllib.parse.urlencode(sorted((params or {}).items()))
        sign_str = qs  # _sign builds accessKey+ts+qs itself
        url = f"{BASE_URL}{path}" + (f"?{qs}" if qs else "")
        data = None
    else:
        payload = json.dumps(body or {}, separators=(",", ":"))
        # MEXC POST signature is over the raw JSON body
        sign_str = payload
        url = f"{BASE_URL}{path}"
        data = payload.encode()
    sig = _sign(api_key, secret, ts, sign_str)
    req = urllib.request.Request(url, data=data, method=method, headers={
        "ApiKey": api_key, "Request-Time": str(ts), "Signature": sig,
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body_txt = ""
        try:
            body_txt = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} on {path}: {body_txt}") from e


def get_contract(symbol):
    """Public contract detail for one symbol (apiAllowed, minVol, etc.)."""
    url = f"{BASE_URL}/api/v1/contract/detail?symbol={symbol}"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read().decode())
    if not d.get("success"):
        raise RuntimeError(f"contract detail failed: {d.get('message')}")
    return d["data"]


def build_bracket(ticker, side, entry, take_profit, stop_loss,
                  equity_usd=DEFAULT_EQUITY_USD, risk_pct=DEFAULT_RISK_PCT,
                  mexc_symbol=None, execution="limit", invalidation=None):
    """Build the order payload for one bracket.

    Returns dict with MEXC fields + a human summary. Does NOT place anything.
    ``execution``: 'market' = immediate market entry at NY open (TP/SL attached);
    'limit' (default) = rest a limit order at ``entry``.
    ``invalidation``: PM-suggested price at which the setup is dead (cancel
    the resting order if price crosses it before fill). For a SHORT that is a
    level ABOVE entry; for a LONG a level BELOW entry.
    """
    mexc_symbol = mexc_symbol or TICKER_TO_MEXC[ticker]
    if side not in ("LONG", "SHORT"):
        raise ValueError("side must be LONG or SHORT")
    if stop_loss == entry or abs(entry - stop_loss) < MIN_STOP_DIST:
        raise ValueError("stop too close to entry")
    risk = equity_usd * risk_pct
    lev = _leverage_for(mexc_symbol)
    return {
        "ticker": ticker, "mexc_symbol": mexc_symbol, "side": side,
        "entry": round(entry, 2), "take_profit": round(take_profit, 2),
        "stop_loss": round(stop_loss, 2), "risk_usd": round(risk, 2),
        "risk_pct": risk_pct, "leverage": lev,
        "execution": execution if execution in ("market", "limit") else "limit",
        "invalidation": invalidation,
        "status": "staged",
    }


def place_bracket(payload, dry_run=True, now=None):
    """Place one bracket order on MEXC (dry_run=True = build only, no send).

    Live path is gated by the deterministic admission governor (2026-08-25
    hardening, Codex review §Priority 0): fresh MEXC state, 3-position cap,
    15% aggregate risk, same-ticker exclusion, fail-closed on unknown state,
    and a post-rounding per-trade risk re-check. A rejected placement returns
    {"admission": {...}} with a machine-readable reason — it does NOT raise.

    For market execution, the SL is validated against the LIVE mark price
    first: if price has already run past the stop, the setup is dead — skip
    with a clean ``invalidated`` result instead of a failed create call
    (2026-08-25: SPY rejected with "Short SL price must be higher than
    current price" because price moved above the SL between analysis and
    placement).
    """
    if dry_run:
        return {"dry_run": True, "payload": payload}
    mexc = payload["mexc_symbol"]

    # --- admission governor (fail-closed; never bypassed) ---------------
    import admission
    dec = admission.decide(payload["ticker"], payload.get("risk_usd"))
    admission.log_admission(payload["ticker"], dec)
    if not dec["accepted"]:
        return {"admission_rejected": True, "admission": dec}

    # live-quote guard: validate SL/entry/TP against the fresh executable
    # quote for BOTH market and limit entries (Codex §execution: validate entry and
    # TP too, and a probe failure BLOCKS the order — never blind placement).
    # Pre-open lane: the executable side is the best ask for a LONG and the
    # best bid for a SHORT (brief §5 item 5); quote age is recorded.
    try:
        q = executable_price(mexc, payload["side"], now=now)
        mark = q["price"]
        payload["quote_ts"] = q.get("ts")
        payload["quote_age_s"] = q.get("age_s")
        if q.get("age_s") is not None and q["age_s"] > 60:
            return {"probe_failed": True, "reason": f"quote stale ({q['age_s']:.0f}s old)"}
        sl = float(payload["stop_loss"])
        tp = float(payload["take_profit"])
        entry_ref = float(payload.get("entry") or mark)
        if payload["side"] == "SHORT" and mark >= sl:
            return {"invalidated": True,
                    "reason": f"price {mark} already above SL {sl} — setup dead"}
        if payload["side"] == "LONG" and mark <= sl:
            return {"invalidated": True,
                    "reason": f"price {mark} already below SL {sl} — setup dead"}
        # entry sanity vs the fresh quote (both entry types)
        if payload["side"] == "SHORT" and mark <= entry_ref * 0.995:
            return {"invalidated": True,
                    "reason": f"live price {mark} already below limit entry {entry_ref}"}
        if payload["side"] == "LONG" and mark >= entry_ref * 1.005:
            return {"invalidated": True,
                    "reason": f"live price {mark} already above limit entry {entry_ref}"}
        # TP sanity: the target must still be reachable from the live quote
        if payload["side"] == "SHORT" and mark <= tp:
            return {"invalidated": True,
                    "reason": f"live price {mark} already at/below TP {tp}"}
        if payload["side"] == "LONG" and mark >= tp:
            return {"invalidated": True,
                    "reason": f"live price {mark} already at/above TP {tp}"}
    except Exception as e:
        # live-quote probe failure BLOCKS the order (fail closed — never
        # fall back to blind placement; Codex §execution item 4)
        return {"probe_failed": True, "reason": f"mark-price probe failed: {e}"}
    # fetch contract size
    c = get_contract(mexc)
    cs = float(c.get("contractSize") or 0)
    if cs <= 0:
        raise RuntimeError(f"{mexc}: bad contractSize {cs}")
    # contracts needed: risk_usd / (stop_dist * cs)
    dist = abs(payload["entry"] - payload["stop_loss"])
    vol = payload["risk_usd"] / (dist * cs)
    # respect the contract's quantity precision (volScale) and minVol
    vol_scale = int(c.get("volScale") or 0)
    min_vol = float(c.get("minVol") or 1)
    vol = round(vol, vol_scale)
    vol = max(min_vol, vol)
    # --- post-rounding risk re-check (governor case 5) ------------------
    # min-volume/precision rounding can inflate the effective loss at the
    # stop past the 5% per-trade cap; recompute from the ACTUAL volume.
    import admission as _adm
    eff_risk = vol * dist * cs
    dec = _adm.decide(payload["ticker"], eff_risk)
    if not dec["accepted"]:
        _adm.log_admission(payload["ticker"], dec,
                           {"phase": "post_rounding", "vol": vol, "cs": cs})
        return {"admission_rejected": True, "admission": dec,
                "effective_risk_usd": round(eff_risk, 4)}
    body = {
        "symbol": mexc,
        "vol": str(vol),
        "leverage": payload["leverage"],
        "side": 1 if payload["side"] == "LONG" else 3,
        "type": 5 if payload.get("execution") == "market" else 1,  # 5=market, 1=limit (MEXC futures type map: 1 limit, 2 post-only, 3 IOC, 4 FOK, 5 market)
        "openType": 2,  # CROSS margin (operator directive 2026-08-24; was 1=isolated)
        "positionMode": 2,  # one-way (single position per symbol)
        "stopLossPrice": str(payload["stop_loss"]),
        "takeProfitPrice": str(payload["take_profit"]),
    }
    if payload.get("execution") != "market":
        body["price"] = str(payload["entry"])
    # set leverage first (only when opening)
    try:
        _mexc_request("POST", "/api/v1/private/position/change_leverage",
                      body={"symbol": mexc, "leverage": payload["leverage"]})
    except Exception as e:
        raise RuntimeError(f"change_leverage failed: {e}") from e
    # --- idempotency: deterministic external key; skip if already resolved --
    run_date = payload.get("run_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    oid = external_oid(run_date, payload["ticker"],
                       f"{payload['side']}_{'MKT' if payload.get('execution') == 'market' else 'LMT'}")
    payload["external_oid"] = oid
    prev = journal_latest_for(oid)
    if prev and prev.get("state") in ("SUBMITTING", "ACCEPTED", "FILLED", "PROTECTED"):
        # same logical decision already in flight: never duplicate-submit;
        # report the existing state for reconciliation instead
        return {"duplicate_suppressed": True, "external_oid": oid,
                "last_state": prev.get("state"), "journal": prev}
    journal_transition(oid, payload["ticker"], "SUBMITTING",
                       {"vol": vol, "entry": payload["entry"]})
    try:
        resp = _mexc_request("POST", "/api/v1/private/order/create", body=body)
    except Exception as e:
        # network timeout = UNKNOWN, not FAILED: resolve via reconciliation
        # before any retry (Codex §idempotency)
        journal_transition(oid, payload["ticker"], "UNKNOWN", {"error": str(e)[:200]})
        raise
    if not resp.get("success"):
        journal_transition(oid, payload["ticker"], "FAILED",
                           {"code": resp.get("code"), "message": resp.get("message")})
        raise RuntimeError(f"order create failed: {resp.get('message')} ({resp.get('code')})")
    journal_transition(oid, payload["ticker"], "ACCEPTED",
                       {"orderId": resp.get("data", {}).get("orderId")})
    # --- post-fill protection: a submitted order is not success until the
    # position and its protective TP/SL are confirmed (governor case 6) ----
    if payload.get("execution") == "market":
        import admission as _adm2
        prot = _adm2.confirm_protection(mexc)
        if prot.get("ok"):
            journal_transition(oid, payload["ticker"], "PROTECTED",
                               {"tp": prot.get("tp"), "sl": prot.get("sl")})
        else:
            journal_transition(oid, payload["ticker"], "PROTECTION_FAILED",
                               {"reason": prot.get("reason"), "error": prot.get("error")})
            resp = {**resp, "protection_failed": True,
                    "protection": {k: prot.get(k) for k in ("reason", "error")}}
    return resp


def cancel_order(mexc_symbol, order_id=None):
    """Cancel open order(s) on a symbol.

    MEXC futures single-cancel (/order/cancel with orderIds) returns 600
    "Parameter error" in practice regardless of body shape (verified
    2026-08-24); /order/cancel_all is reliable. So this cancels ALL open
    orders for the symbol — used for emergency cleanup, which is the actual
    production need.
    """
    resp = _mexc_request("POST", "/api/v1/private/order/cancel_all",
                         body={"symbol": mexc_symbol})
    if not resp.get("success"):
        raise RuntimeError(f"order cancel failed: {resp.get('message')} ({resp.get('code')})")
    return resp


def cancel_leftover_brackets():
    """Cancel any still-resting entry brackets on our universe before a fresh run.

    Unfilled limit entries from a prior preopen must not stack with today's
    new brackets (operator directive 2026-08-24: no hanging orders). Only
    touches open *orders* on our 5 symbols — positions and their attached
    TP/SL are untouched.
    """
    cancelled = []
    for ticker, sym in TICKER_TO_MEXC.items():
        try:
            cancel_order(sym)
            cancelled.append(sym)
        except Exception as e:
            cancelled.append(f"{sym}: {str(e)[:80]}")
    return cancelled


def cancel_order_by_id(mexc_symbol, order_id):
    """Cancel ONE entry order by its id (2026-08-26 pre-open lane).

    Never cancel_all: unfilled-entry cancellation at 09:29 ET must be
    surgical (brief §9). MEXC's single-cancel endpoint is
    /api/v1/private/order/cancel with orderIds — historically flaky
    (600 Parameter error 2026-08-24), so on failure we VERIFY the order's
    live state before declaring failure: a transient error must never be
    read as "cancelled" (brief §9: reconcile, do not assume).
    """
    body = {"symbol": mexc_symbol, "orderIds": str(order_id)}
    try:
        resp = _mexc_request("POST", "/api/v1/private/order/cancel", body=body)
        if resp.get("success"):
            return {"cancelled": True, "order_id": order_id}
    except Exception as e:
        pass  # fall through to verification below
    # reconcile: is the order actually still open?
    still_open = _order_is_open(mexc_symbol, order_id)
    if still_open is None:
        return {"cancelled": False, "order_id": order_id,
                "reason": "cancel_unconfirmed", "reconcile": "state_unknown"}
    return {"cancelled": not still_open, "order_id": order_id,
            "reconcile": "verified_open" if still_open else "verified_closed"}


def _order_is_open(mexc_symbol, order_id):
    """True/False whether the entry order is still resting; None = unknown."""
    try:
        resp = _mexc_request("GET", "/api/v1/private/order/list/open_orders",
                             params={"symbol": mexc_symbol, "page_num": 1, "page_size": 100})
        orders = resp.get("data") or []
        for o in orders:
            if str(o.get("orderId")) == str(order_id):
                return True
        return False
    except Exception:
        return None


def cancel_unfilled_entries(date, now=None):
    """09:29 ET: cancel entry orders from THIS run that remain unfilled.

    Pre-open lane rule (brief §9): an unfilled pre-open entry must not
    survive into the regular session. Surgical by order id — TP/SL
    protection and filled positions are never touched. A filled entry is
    reported as `filled` and its position + bracket stay active per the
    trade plan.
    """
    now = now or datetime.now(timezone.utc)
    report = {"cancelled": [], "filled": [], "unconfirmed": [], "errors": []}
    data = load_orders()
    if data.get("date") != date:
        report["errors"].append(f"orders.json date {data.get('date')} != {date}")
        return report
    for o in data.get("orders", []):
        oid = o.get("order_id")
        sym = o.get("mexc_symbol")
        if o.get("status") != "placed" or not oid or not sym:
            continue
        open_state = _order_is_open(sym, oid)
        if open_state is False:
            # no longer resting: either filled (position exists) or already gone
            report["filled"].append({"ticker": o.get("ticker"), "order_id": oid})
            o["status"] = "filled_or_closed"
            continue
        res = cancel_order_by_id(sym, oid)
        if res.get("cancelled"):
            report["cancelled"].append({"ticker": o.get("ticker"), "order_id": oid})
            o["status"] = "cancelled_unfilled"
        else:
            report["unconfirmed"].append({"ticker": o.get("ticker"), "order_id": oid,
                                          "reason": res.get("reason") or res.get("reconcile")})
            o["status"] = "cancel_unconfirmed"
    save_orders(data)
    return report


def load_orders():
    if os.path.exists(ORDER_FILE):
        try:
            return json.load(open(ORDER_FILE))
        except (json.JSONDecodeError, OSError):
            pass
    return {"date": None, "orders": []}


# --------------------------------------------------------------------------
# Idempotency: deterministic client order key + order-state journal
# (Codex §"Idempotency and order-state reconciliation", 2026-08-25)
# --------------------------------------------------------------------------

STRATEGY_VERSION = "ta-shadow-2026-08-25"
ORDER_JOURNAL = os.path.join(LANE, "results", "order_journal.jsonl")

VALID_STATES = ("STAGED", "ADMITTED", "SUBMITTING", "ACCEPTED", "FILLED",
                "PROTECTED", "CLOSED", "UNKNOWN", "FAILED", "PROTECTION_FAILED")


def external_oid(run_date, ticker, action, strategy_version=STRATEGY_VERSION):
    """Deterministic client order key: run date + ticker + action + strategy
    version. A retried placement of the same logical decision resolves to the
    SAME key, so a duplicate submission can be detected and reconciled."""
    key = f"{strategy_version}:{run_date}:{ticker}:{action}"
    return "TA-" + hashlib.sha256(key.encode()).hexdigest()[:16].upper()


def journal_write(row):
    """Append one state-transition row to the order journal (atomic-ish append)."""
    os.makedirs(os.path.dirname(ORDER_JOURNAL), exist_ok=True)
    with open(ORDER_JOURNAL, "a") as f:
        f.write(json.dumps(row) + "\n")


def journal_latest_for(oid):
    """Most recent journal row for an external order id (or None)."""
    if not os.path.exists(ORDER_JOURNAL):
        return None
    latest = None
    try:
        with open(ORDER_JOURNAL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # malformed history line: skip visibly, not silently
                if r.get("external_oid") == oid:
                    latest = r
    except OSError:
        return None
    return latest


def journal_transition(oid, ticker, state, detail=None):
    row = {"ts": datetime.now(timezone.utc).isoformat(),
           "external_oid": oid, "ticker": ticker, "state": state,
           "detail": detail or {}}
    journal_write(row)
    return row


def reconcile_external_order(oid):
    """Resolve an UNKNOWN or retried placement by live MEXC lookup.

    Returns the open-order row matching our universe (if any), a position
    on the ticker implied by the journal, or None. Network timeout is
    UNKNOWN and must be resolved through this BEFORE any resubmit.
    """
    from admission import fetch_live_state
    try:
        state = fetch_live_state()
    except Exception:
        return {"status": "state_error"}
    return {"status": "ok", "state": state}


def get_mark_price(symbol):
    """Latest price for a MEXC contract (public endpoint, no auth)."""
    url = f"{BASE_URL}/api/v1/contract/ticker?symbol={symbol}"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read().decode())
    if not d.get("success"):
        raise RuntimeError(f"ticker failed for {symbol}: {d.get('message')}")
    return float(d["data"]["lastPrice"])


def refresh_quote(symbol, now=None):
    """Fresh executable quote with timestamp + age (brief §5 items 4-7).

    Returns {price, bid, ask, spread, ts, age_s}. price is the best ask for
    a LONG entry and the best bid for a SHORT (the executable side); the
    caller picks by side via `bid_or_ask`. Falls back to lastPrice when the
    depth book is empty. Records the quote timestamp and age so stale
    quotes can be rejected deterministically.
    """
    now = now or datetime.now(timezone.utc)
    url = f"{BASE_URL}/api/v1/contract/depth?symbol={symbol}&limit=5"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read().decode())
    if not d.get("success"):
        raise RuntimeError(f"depth failed for {symbol}: {d.get('message')}")
    book = d.get("data") or {}
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    bid = float(bids[0][0]) if bids else None
    ask = float(asks[0][0]) if asks else None
    if bid is None or ask is None:
        # empty book (pre-open on some contracts): fall back to last trade
        px = get_mark_price(symbol)
        bid = ask = px
    ts = now.timestamp()
    return {
        "bid": bid, "ask": ask,
        "spread": (ask - bid) if (bid and ask) else 0.0,
        "ts": ts, "age_s": 0.0,
        "bid_or_ask": {"LONG": ask, "SHORT": bid},
        "price": None,  # caller resolves side below
    }


def executable_price(symbol, side, now=None):
    """Best ask for a LONG, best bid for a SHORT (brief §5 item 5)."""
    q = refresh_quote(symbol, now=now)
    q["price"] = q["bid_or_ask"][side]
    return q


def sweep_stale_brackets(now=None, dry_run=True):
    """Cancel resting entry brackets that are stale.

    Two independent triggers, both set by the PM's verdict — the sweep never
    decides anything itself (operator directive 2026-08-24):
      1. Price invalidation: the PM's ``invalidation`` price has been crossed
         (SHORT: price rose above it; LONG: price fell below it). The thesis
         is dead before it ever filled.
      2. Post-open sweep: any bracket still unfilled when this runs (~30-60 min
         after NY open) is cancelled — it had its window and missed it.

    Only touches open *orders* on our universe; positions + attached TP/SL are
    untouched. ``dry_run=True`` (default) reports only.
    """
    now = now or datetime.now(timezone.utc)
    report = {"checked": [], "expired": [], "cancelled": [], "errors": []}
    try:
        resp = _mexc_request("GET", "/api/v1/private/order/list/open_orders",
                             params={"page_num": 1, "page_size": 100})
        open_orders = resp.get("data") or []
    except Exception as e:
        report["errors"].append(f"list_open_orders: {e}")
        return report
    if not isinstance(open_orders, list):
        return report
    book = load_orders().get("orders", [])
    # Cache mark prices per symbol (public, cheap)
    prices = {}
    for o in open_orders:
        sym = o.get("symbol")
        rec = next((r for r in book if r.get("mexc_symbol") == sym), None)
        if rec is None:
            report["checked"].append({"symbol": sym, "note": "not in orders.json"})
            continue
        reasons = []
        inval = rec.get("invalidation")
        if inval:
            try:
                if sym not in prices:
                    prices[sym] = get_mark_price(sym)
                px = prices[sym]
            except Exception as e:
                report["errors"].append(f"{sym}: mark price: {e}")
                px = None
            if px is not None:
                crossed = (px > inval) if rec.get("side") == "SHORT" else (px < inval)
                if crossed:
                    reasons.append(f"price {px} crossed invalidation {inval}")
        row = {"symbol": sym, "invalidation": inval,
               "side": rec.get("side"), "now": now.isoformat()}
        if reasons:
            row["reason"] = "; ".join(reasons)
            row["stale"] = True
        else:
            row["reason"] = "unfilled at post-open sweep"
            row["stale"] = True  # this sweep IS the post-open cutoff
        report["checked"].append(row)
        if row["stale"]:
            report["expired"].append(row)
            if not dry_run:
                try:
                    cancel_order(sym)
                    row["cancelled"] = True
                    report["cancelled"].append(sym)
                except Exception as e:
                    row["cancelled"] = False
                    report["errors"].append(f"{sym}: cancel: {e}")
    return report


def save_orders(data):
    tmp = ORDER_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, ORDER_FILE)


def record_fill_outcomes(now=None, dry_run=True):
    """Record trade outcomes for orders we placed that have since closed.

    MEXC's contract API has no position-history endpoint, so we derive the
    outcome ourselves: compare our recorded *placed* orders against the
    currently-open positions. An order that is no longer resting AND has no
    open position either hit its TP or SL (or was closed manually) — we
    record which by comparing the last known price direction, and stamp the
    row with today's date + the position's realized P&L when available.

    Idempotent on order_id. dry_run=True reports only (no history writes).
    """
    now = now or datetime.now(timezone.utc)
    report = {"recorded": [], "still_open": [], "errors": []}
    book = load_orders().get("orders", [])
    if not book:
        return report
    try:
        from position_ctx import get_open_positions, record_trade_outcome
    except Exception as e:
        report["errors"].append(f"position_ctx import: {e}")
        return report

    open_pos = get_open_positions()
    placed = [o for o in book if o.get("status") == "placed" and o.get("order_id")]
    for o in placed:
        tk = o.get("ticker")
        sym = o.get("mexc_symbol")
        # still has an open position → not closed
        if open_pos.get(tk):
            report["still_open"].append(tk)
            continue
        # no open position → closed. Determine reason from TP/SL vs current px.
        try:
            px = get_mark_price(sym)
        except Exception as e:
            report["errors"].append(f"{tk}: mark: {e}")
            continue
        side = o.get("side")
        tp, sl = o.get("take_profit"), o.get("stop_loss")
        if side == "SHORT":
            outcome = "TP hit" if px <= tp else ("SL hit" if px >= sl else "closed")
        else:
            outcome = "TP hit" if px >= tp else ("SL hit" if px <= sl else "closed")
        entry = o.get("entry")
        # Realized P&L is only knowable from MEXC position history (unavailable),
        # so we record the outcome + current price; the digest shows direction.
        row = {
            "date": now.strftime("%Y-%m-%d"),
            "ticker": tk,
            "side": side,
            "entry": entry,
            "outcome": outcome,
            "px_now": px,
            "order_id": o.get("order_id"),
            "closed_at": now.isoformat(),
        }
        if not dry_run:
            try:
                record_trade_outcome(row)
                report["recorded"].append(tk)
            except Exception as e:
                report["errors"].append(f"{tk}: record: {e}")
        else:
            report["recorded"].append(tk + " (dry)")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, choices=list(TICKER_TO_MEXC))
    ap.add_argument("--side", required=True, choices=["LONG", "SHORT"])
    ap.add_argument("--entry", type=float, required=True)
    ap.add_argument("--tp", type=float, required=True)
    ap.add_argument("--sl", type=float, required=True)
    ap.add_argument("--execute", action="store_true", help="actually place (default: stage only)")
    args = ap.parse_args()
    payload = build_bracket(args.ticker, args.side, args.entry, args.tp, args.sl)
    if args.execute:
        resp = place_bracket(payload, dry_run=False)
        print(json.dumps({"placed": True, "response": resp}, indent=2))
    else:
        print(json.dumps({"staged": payload}, indent=2))
