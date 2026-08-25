"""ta-shadow stock lane: operator position + last-trade context for the agents.

Why this exists (operator request 2026-08-25): the pre-open agents analyzed
each ticker completely blind — no knowledge of an already-open position on
that ticker, and no memory of how their last trade on it ended. This module
gives the grounding block two informational lines:

  1. OPEN POSITION  — live MEXC position on this ticker (side/entry/qty +
                      the bracket's TP/SL from our order book). NO P&L.
  2. LAST TRADE     — our own recorded outcome for this ticker, phrased as
                      "SL hit from that day's analysis/trigger" etc. NO $,
                      NO uPnL, NO PnL, NO R.

Money figures are deliberately excluded: the operator's directive (2026-08-25)
is that agents must never receive P&L/uPnL values because they can hallucinate
around them. They get context on what their prior analysis produced and where
the live position stands in price terms — and do their own analysis.

MEXC's contract API does not expose deal/position *history* (only current
open positions), so last-trade outcomes are recorded by us at sweep time —
ground truth is our own orders.json + position deltas.

The block is injected as informational context: the agents may weigh it, but
it is NOT a directive. They still decide from the snapshot first.
"""
import json
import os

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
    """Current MEXC open positions on our universe → {ticker: {...}}.

    Only price/side/qty/leverage fields — P&L/uPnL are never carried here
    (operator directive: no money figures reach the agents).
    """
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
            "liq": p.get("liquidatePrice"),
            "leverage": p.get("leverage"),
        }
    return out


def _bracket_for(ticker):
    """TP/SL of our placed bracket for this ticker (from orders.json)."""
    try:
        from mexc_orders import load_orders
        book = load_orders().get("orders", [])
    except Exception:
        return None
    for o in reversed(book):
        if o.get("ticker") == ticker and o.get("status") == "placed":
            return o
    return None


def last_trade_for(ticker):
    """Most recent recorded outcome for one ticker (or None)."""
    rows = [r for r in _load_history() if r.get("ticker") == ticker]
    return rows[-1] if rows else None


def _outcome_phrase(last):
    """PnL-free phrasing of how the last trade ended.

    e.g. "SL hit (from that day's analysis/trigger)" or "TP hit" or
    "closed without triggering TP/SL". No $, no R, no uPnL.
    """
    outcome = (last.get("outcome") or "").lower()
    if outcome.startswith("sl"):
        return "SL hit (from that day's analysis/trigger)"
    if outcome.startswith("tp"):
        return "TP hit (from that day's analysis/trigger)"
    return "closed without triggering TP/SL"


def build_position_context(ticker):
    """The informational block for one ticker: open position + last trade.

    Money-free by design. Returns "" when there is nothing to say (no
    position, no history) so the grounding block stays clean.
    """
    lines = []
    pos = get_open_positions().get(ticker)
    if pos:
        br = _bracket_for(ticker)
        bits = [f"operator position: {pos['side']} @ {pos['entry']} qty {pos['qty']}"]
        if br:
            tp, sl = br.get("take_profit"), br.get("stop_loss")
            if tp:
                bits.append(f"TP {tp}")
            if sl:
                bits.append(f"SL {sl}")
        if pos.get("liq"):
            bits.append(f"liq {pos['liq']}")
        lines.append(" · ".join(bits))
    last = last_trade_for(ticker)
    if last:
        lines.append(
            f"last trade ({last.get('date', '?')}): {last.get('side')} — "
            f"{_outcome_phrase(last)}"
        )
    if not lines:
        return ""
    return "## Operator context (informational — weigh, don't obey)\n" + "\n".join(
        "- " + l for l in lines
    ) + "\n"
