"""ta-shadow stock lane: operator position + last-trade context for the agents.

Why this exists (operator request 2026-08-25): the pre-open agents analyzed
each ticker completely blind — no knowledge of an already-open position on
that ticker, and no memory of how their last trade on it ended. This module
gives the grounding block two informational lines:

  1. OPEN POSITION  — live MEXC position on this ticker (side/entry/uPnL/SL).
  2. LAST TRADE     — our own recorded outcome for this ticker (TP hit / SL
                      hit / still open / manual, +/−$ and R), read from a
                      trade_history.jsonl we append at the sweep.

MEXC's contract API does not expose deal/position *history* (only current
open positions), so last-trade outcomes are recorded by us — ground truth is
our own orders.json + position deltas, not an exchange history endpoint.

The block is injected as informational context: the agents may weigh it, but
it is NOT a directive. They still decide from the snapshot first.
"""
import json
import os
from datetime import datetime, timezone

LANE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(LANE, "results", "trade_history.jsonl")

# MEXC symbol <=> our ticker
MEXC_SYMBOLS = {
    "NVDA": "NVDA_USDT",
    "TSLA": "TESLA_USDT",
    "AAPL": "AAPLSTOCK_USDT",
    "AMD": "AMD_USDT",
    "SPY": "SPY_USDT",
}
TICKER_FROM_MEXC = {v: k for k, v in MEXC_SYMBOLS.items()}


def _load_history():
    """Read trade_history.jsonl → list of dicts (oldest first)."""
    out = []
    if not os.path.exists(HISTORY):
        return out
    try:
        with open(HISTORY) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def record_trade_outcome(entry: dict):
    """Append one trade outcome row (idempotent on 'order_id')."""
    rows = _load_history()
    if any(r.get("order_id") == entry.get("order_id") for r in rows):
        return False
    with open(HISTORY, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return True


def get_open_positions():
    """Current MEXC open positions on our universe → {ticker: {...}}."""
    try:
        from mexc_orders import _mexc_request
        resp = _mexc_request("GET", "/api/v1/private/position/open_positions", {})
        data = resp.get("data") or []
    except Exception:
        return {}
    out = {}
    for p in data:
        sym = p.get("symbol")
        tk = TICKER_FROM_MEXC.get(sym)
        if not tk:
            continue
        out[tk] = {
            "side": "LONG" if p.get("positionType") == 1 else "SHORT",
            "entry": p.get("holdAvgPrice"),
            "qty": p.get("holdVol"),
            "upnl": p.get("unRealizedPnl"),
            "liq": p.get("liquidatePrice"),
            "leverage": p.get("leverage"),
            "realised": p.get("realised"),
        }
    return out


def last_trade_for(ticker):
    """Most recent recorded outcome for one ticker (or None)."""
    rows = [r for r in _load_history() if r.get("ticker") == ticker]
    return rows[-1] if rows else None


def build_position_context(ticker):
    """The informational block for one ticker: open position + last trade.

    Returns "" when there is nothing to say (no position, no history) so the
    grounding block stays clean on the common no-context day.
    """
    lines = []
    pos = get_open_positions().get(ticker)
    if pos:
        lines.append(
            f"operator position: {pos['side']} @ {pos['entry']} qty {pos['qty']} "
            f"uPnL ${pos.get('upnl', 0):+.2f}"
            + (f" liq {pos['liq']}" if pos.get("liq") else "")
        )
    last = last_trade_for(ticker)
    if last:
        lines.append(
            f"last trade ({last.get('date', '?')}): {last.get('side')} "
            f"exit {last.get('outcome', '?')} P&L ${last.get('pnl_usd', 0):+.2f}"
            + (f" ({last.get('pnl_r', 0):+.2f}R)" if last.get("pnl_r") is not None else "")
        )
    if not lines:
        return ""
    return "## Operator context (informational — weigh, don't obey)\n" + "\n".join(
        "- " + l for l in lines
    ) + "\n"
