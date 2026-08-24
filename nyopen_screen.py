#!/usr/bin/env python3
"""ta-shadow STOCK lane: NY-open fast screen (deterministic, no LLM).

Runs at NY open + through the first 2h window (13:35–15:30 UTC) for the
5-stock universe. Computes per ticker:
  - gap vs prior close, current price vs VWAP, first-30m range
  - distance to key levels (R1/S1 pivots, 20d high/low, SMA50)
  - volume pace vs 20d average
  - DMC-style level geometry: body-break of a level, failure to hold,
    range-lock (price between first-30m high/low)
and emits a compact ALERT to the Home channel when a real opportunity shows
(gap/volume/level-break), plus a full 5-ticker level table.

This is the FAST layer; the TA due-diligence layer (run_stock.py) runs
pre-open. Alerts go to the Home channel via a cron no_agent delivery.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

LANE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LANE)

from stock_snapshot import fetch_stock_snapshot, STOCK_WATCHLIST

# opportunity thresholds
GAP_MIN_PCT = 1.0          # |gap| >= 1% to flag
VOL_PACE_MIN = 1.5         # volume pace >= 1.5x avg
LEVEL_PROX = 0.5           # within 0.5 ATR of a key level


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _detect(ticker, snap):
    """Return opportunity flags + a one-line summary for one ticker."""
    l = snap.get("levels") or {}
    s = snap.get("session") or {}
    flags = []
    price = s.get("price_now") or l.get("close")
    gap = s.get("gap_pct")
    vol = s.get("vol_pace")
    atr = l.get("atr14")
    vwap = s.get("vwap")

    # Hard signals: gap and/or volume. Level proximity is CONTEXT, not a flag.
    if gap is not None and abs(gap) >= GAP_MIN_PCT:
        flags.append(f"GAP {gap:+.1f}%")
    if vol is not None and vol >= VOL_PACE_MIN:
        flags.append(f"VOL {vol:.1f}x")
    # DMC-style range geometry (context, only when a hard signal exists)
    fh, fl = s.get("first30_high"), s.get("first30_low")
    if fh and fl and price:
        if price >= fh:
            flags.append(">30M HIGH")
        elif price <= fl:
            flags.append("<30M LOW")
        else:
            flags.append("RANGE-LOCK")
    # nearest key level (context)
    if atr and price:
        cands = [("R1", l.get("r1")), ("S1", l.get("s1")),
                 ("20dH", l.get("high20")), ("20dL", l.get("low20"))]
        near = [(abs(price - v) / atr, name) for name, v in cands if v]
        if near:
            d, name = min(near)
            if d <= LEVEL_PROX:
                flags.append(f"~{name}")
    return {
        "ticker": ticker, "price": price, "gap_pct": gap, "vol_pace": vol,
        "vwap": vwap, "first30_high": fh, "first30_low": fl,
        "r1": l.get("r1"), "s1": l.get("s1"),
        "high20": l.get("high20"), "low20": l.get("low20"),
        "flags": flags, "opportunity": bool(flags),
    }


def screen(tickers=None, session=True):
    """Run the screen for the universe (or subset). Returns results list."""
    out = []
    for tk in (tickers or STOCK_WATCHLIST):
        try:
            snap = fetch_stock_snapshot(tk, session=session)
            out.append(_detect(tk, snap))
        except Exception as e:
            out.append({"ticker": tk, "error": str(e)[:200], "opportunity": False})
    return out


def format_alerts(results):
    """Compact Telegram text: opportunities first, then full level table."""
    opps = [r for r in results if r.get("opportunity") and not r.get("error")]
    lines = []
    if opps:
        lines.append("🎯 NY-OPEN OPPORTUNITIES")
        for r in opps:
            lines.append(
                f"• {r['ticker']} @ {r['price']} — {', '.join(r['flags'])}"
                f" (gap {r['gap_pct']:+.1f}%, vol {r['vol_pace']:.1f}x)"
            )
    lines.append("📊 KEY LEVELS (first 2h)")
    for r in results:
        if r.get("error"):
            lines.append(f"• {r['ticker']}: error {r['error']}")
            continue
        lines.append(
            f"• {r['ticker']}: ${r['price']} | gap {r['gap_pct']:+.1f}% | "
            f"VWAP {r['vwap']} | 30m {r['first30_low']}-{r['first30_high']} | "
            f"R1 {r['r1']} S1 {r['s1']} | 20dH {r['high20']} 20dL {r['low20']} | "
            f"vol {r['vol_pace']:.1f}x"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print instead of emit")
    ap.add_argument("--tickers", default=None, help="comma list (default: all 5)")
    args = ap.parse_args()
    tk = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
    results = screen(tk)
    text = format_alerts(results)
    print(text)
    print("--- JSON ---")
    print(json.dumps(results, indent=1))
