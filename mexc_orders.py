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

LANE = os.path.dirname(os.path.abspath(__file__))
BASE_URL = os.environ.get("MEXC_API_BASE", "https://api.mexc.com")
ORDER_FILE = os.path.join(LANE, "orders.json")

TICKER_TO_MEXC = {
    "NVDA": "NVIDIA_USDT", "TSLA": "TESLA_USDT",
    "AAPL": "AAPLSTOCK_USDT", "AMD": "AMDSTOCK_USDT", "SPY": "SPY_USDT",
}

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


def place_bracket(payload, dry_run=True):
    """Place one bracket order on MEXC (dry_run=True = build only, no send).

    Returns the MEXC response dict (with orderId when live).
    """
    if dry_run:
        return {"dry_run": True, "payload": payload}
    mexc = payload["mexc_symbol"]
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
    resp = _mexc_request("POST", "/api/v1/private/order/create", body=body)
    if not resp.get("success"):
        raise RuntimeError(f"order create failed: {resp.get('message')} ({resp.get('code')})")
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


def load_orders():
    if os.path.exists(ORDER_FILE):
        try:
            return json.load(open(ORDER_FILE))
        except (json.JSONDecodeError, OSError):
            pass
    return {"date": None, "orders": []}


def get_mark_price(symbol):
    """Latest price for a MEXC contract (public endpoint, no auth)."""
    url = f"{BASE_URL}/api/v1/contract/ticker?symbol={symbol}"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read().decode())
    if not d.get("success"):
        raise RuntimeError(f"ticker failed for {symbol}: {d.get('message')}")
    return float(d["data"]["lastPrice"])


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
