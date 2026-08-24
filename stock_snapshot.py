"""ta-shadow stock lane: snapshot builder + key levels for the shadow agents.

Stock-only (2026-08-23 operator directive — the crypto lane is removed).
Universe: NVDA, TSLA, AAPL, AMD, SPY. Focus: NY open + first 2 hours.

This module builds a per-symbol STOCK snapshot (a plain dict) containing the
daily-structure key levels (pivots, SMAs, ATR, 20d/52w extremes) plus, when
the market is open, intra-session levels (open, gap, VWAP, first-30m range,
volume pace). The grounding block derived from it is the ONLY data source
the shadow agents may cite.

Data: yfinance daily + 1m/5m history. No keys, no trade permissions.
"""
import json
import os
import urllib.request
from datetime import datetime, timezone

import yfinance as yf
import pandas as pd

# MEXC futures symbol mapping for the same 5 names (stock perps).
# NVDA/TSLA are NVIDIA_USDT/TESLA_USDT; AAPL/AMD use *STOCK_USDT; SPY is SPY_USDT.
TICKER_TO_MEXC = {
    "NVDA": "NVIDIA_USDT",
    "TSLA": "TESLA_USDT",
    "AAPL": "AAPLSTOCK_USDT",
    "AMD": "AMDSTOCK_USDT",
    "SPY": "SPY_USDT",
}
STOCK_WATCHLIST = list(TICKER_TO_MEXC)

_UA = {"User-Agent": "Mozilla/5.0 (research; ta-shadow stock lane)"}


def _fmt(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _sma(s, n):
    v = s.rolling(n).mean().iloc[-1]
    return float(v) if pd.notna(v) else None


def _pivots(high, low, close):
    """Classic floor-trader pivots from the PRIOR completed daily candle."""
    h, l, c = float(high), float(low), float(close)
    p = (h + l + c) / 3
    return {
        "pivot_p": round(p, 2), "r1": round(2 * p - l, 2), "r2": round(p + (h - l), 2),
        "s1": round(2 * p - h, 2), "s2": round(p - (h - l), 2),
    }


def _atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    v = tr.rolling(n).mean().iloc[-1]
    return float(v) if pd.notna(v) else None


def fetch_stock_snapshot(ticker, lookback=90, session=False):
    """Build the stock snapshot dict for one ticker.

    session=True additionally fetches intraday (1m/5m) data to fill
    open/gap/VWAP/first-30m/volume-pace fields. Safe to call pre-open
    (session fields are simply omitted then).
    """
    t = yf.Ticker(ticker)
    d = t.history(period=f"{lookback}d", interval="1d", auto_adjust=True)
    if d is None or d.empty:
        raise RuntimeError(f"{ticker}: no daily history")
    d = d.dropna(subset=["Close"])
    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) >= 2 else last

    high20 = float(d["High"].tail(20).max())
    low20 = float(d["Low"].tail(20).min())
    high52 = float(d["High"].max())
    low52 = float(d["Low"].min())
    close = float(last["Close"])
    prior_close = float(prev["Close"])
    avg_vol_20d = float(d["Volume"].tail(20).mean())
    ret_5d = (close / float(d["Close"].iloc[-6]) - 1) * 100 if len(d) >= 6 else None

    levels = {
        "close": round(close, 2),
        "prior_close": round(prior_close, 2),
        "ret_5d": round(ret_5d, 2) if ret_5d is not None else None,
        "sma20": round(_sma(d["Close"], 20), 2) if _sma(d["Close"], 20) else None,
        "sma50": round(_sma(d["Close"], 50), 2) if _sma(d["Close"], 50) else None,
        "sma200": round(_sma(d["Close"], 200), 2) if _sma(d["Close"], 200) else None,
        "high20": round(high20, 2), "low20": round(low20, 2),
        "high52w": round(high52, 2), "low52w": round(low52, 2),
        "atr14": round(_atr(d), 2) if _atr(d) else None,
        "avg_vol_20d": int(avg_vol_20d),
        **_pivots(last["High"], last["Low"], last["Close"]),
    }
    trend = "above_50" if close > (levels["sma50"] or 0) else "below_50"

    snap = {
        "kind": "stock",
        "symbol": ticker,
        "name": t.info.get("shortName") if hasattr(t, "info") else None,
        "as_of": d.index[-1].strftime("%Y-%m-%d"),
        "trend": trend,
        "earnings_date": None,
        "levels": levels,
        "session": {},
        "dmc": "",
    }
    try:
        cal = getattr(t, "calendar", None)
        if cal is not None:
            snap["earnings_date"] = str(cal.get_earnings_dates(limit=2).index[0].date())
    except Exception:
        pass  # ETFs (SPY) have no earnings calendar — leave None

    if session:
        _fill_session(snap, ticker)

    # DMC block (HTF daily + LTF intraday) built AFTER session fill
    snap["dmc"] = build_dmc_level_block(
        {"levels": levels, "session": snap["session"]},
        df=d, df_intraday=snap.pop("_intraday_df", None),
    )
    return snap


def _fill_session(snap, ticker):
    """Intraday fields (only when the market is open / has data today)."""
    t = yf.Ticker(ticker)
    try:
        m = t.history(period="1d", interval="1m", auto_adjust=True)
    except Exception:
        return
    if m is None or m.empty:
        return
    m = m.dropna(subset=["Close"])
    if m.empty:
        return
    o = float(m["Open"].iloc[0])
    c = float(m["Close"].iloc[-1])
    prior = snap["levels"]["prior_close"]
    vol = float(m["Volume"].sum())
    # elapsed minutes since first print (approx market session)
    first_ts = m.index[0]
    now_ts = m.index[-1]
    elapsed_min = max(1, (now_ts - first_ts).total_seconds() / 60)
    pace = vol / max(1, elapsed_min) / (snap["levels"]["avg_vol_20d"] / 390.0) if snap["levels"]["avg_vol_20d"] else None
    # first 30 minutes
    m30 = m[m.index <= first_ts + pd.Timedelta(minutes=30)]
    fh = float(m30["High"].max()) if not m30.empty else None
    fl = float(m30["Low"].min()) if not m30.empty else None
    # session VWAP
    tp = (m["High"] + m["Low"] + m["Close"]) / 3
    vwap = float((tp * m["Volume"]).sum() / m["Volume"].sum()) if m["Volume"].sum() else None
    snap["session"] = {
        "session_open": round(o, 2),
        "price_now": round(c, 2),
        "gap_pct": round((o / prior - 1) * 100, 2) if prior else None,
        "vwap": round(vwap, 2) if vwap else None,
        "first30_high": round(fh, 2) if fh else None,
        "first30_low": round(fl, 2) if fl else None,
        "vol_pace": round(pace, 2) if pace else None,
    }
    # stash the intraday frame for the DMC LTF block
    snap["_intraday_df"] = m
    # merge session levels into the flat levels dict for the grounding block
    snap["levels"].update({k: v for k, v in snap["session"].items() if v is not None})


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_mexc_klines(mexc_symbol, limit=120):
    """Daily klines from MEXC futures for the stock perp (public)."""
    url = (f"https://api.mexc.com/api/v1/contract/kline"
           f"?symbol={mexc_symbol}&interval=1d&limit={limit}")
    rows = _get(url)
    # rows: [open, close, high, low, volume, ts, ...] or dicts — normalize
    if rows and isinstance(rows[0], list):
        return [{"open": float(r[0]), "close": float(r[1]), "high": float(r[2]),
                 "low": float(r[3]), "volume": float(r[4])} for r in rows]
    return rows


def build_stock_grounding_block(snapshot, variant="A"):
    """The canonical stock snapshot text block (citations only from here)."""
    l = snapshot.get("levels") or {}
    s = snapshot.get("session") or {}
    lines = [
        f"- ticker: {snapshot.get('symbol')} ({snapshot.get('name', 'n/a')})",
        f"- as_of (last completed daily candle): {snapshot.get('as_of')}",
        f"- close (last daily close): {_fmt(l.get('close'))}",
        f"- prior close: {_fmt(l.get('prior_close'))}",
        f"- 5d return: {_fmt(l.get('ret_5d'))}%",
        f"- trend: {snapshot.get('trend', 'n/a')} (close vs 50d SMA)",
        f"- sma20: {_fmt(l.get('sma20'))} | sma50: {_fmt(l.get('sma50'))} | sma200: {_fmt(l.get('sma200'))}",
        f"- 20d high/low: {_fmt(l.get('high20'))} / {_fmt(l.get('low20'))}",
        f"- 52w high/low: {_fmt(l.get('high52w'))} / {_fmt(l.get('low52w'))}",
        f"- ATR14: {_fmt(l.get('atr14'))}",
        f"- pivots (prior day): P {_fmt(l.get('pivot_p'))} | R1 {_fmt(l.get('r1'))} | R2 {_fmt(l.get('r2'))} | S1 {_fmt(l.get('s1'))} | S2 {_fmt(l.get('s2'))}",
        f"- 20d avg volume: {_fmt(l.get('avg_vol_20d'), 0)}",
        f"- earnings date: {snapshot.get('earnings_date', 'n/a')}",
    ]
    if s.get("session_open") is not None:
        lines += [
            f"- session open: {_fmt(s.get('session_open'))}",
            f"- current price: {_fmt(s.get('price_now'))}",
            f"- gap vs prior close: {_fmt(s.get('gap_pct'))}%",
            f"- session VWAP: {_fmt(s.get('vwap'))}",
            f"- first-30m high/low: {_fmt(s.get('first30_high'))} / {_fmt(s.get('first30_low'))}",
            f"- session volume pace (vs 20d avg, elapsed-adjusted): {_fmt(s.get('vol_pace'))}x",
        ]
    if variant == "B":
        lines.reverse()
    block = "## Stock snapshot (source of truth for ALL figures)\n" + "\n".join(lines) + "\n"
    dmc = snapshot.get("dmc")
    if dmc:
        block += "\n" + dmc
    return block


def build_stock_mood_line(snapshot):
    """Deterministic single-line crowd mood (gap + trend bias). Labeled only."""
    s = snapshot.get("session") or {}
    gap = s.get("gap_pct")
    bits = []
    if gap is not None:
        bits.append(f"gap {gap:+.2f}% ({'buy-side open' if gap > 0 else 'sell-side open'})")
    bits.append(f"trend {snapshot.get('trend', 'n/a')}")
    mood = "risk-on" if (gap or 0) > 0 and snapshot.get("trend") == "above_50" else ("risk-off" if (gap or 0) < 0 and snapshot.get("trend") == "below_50" else "mixed")
    return f"Crowd mood: {mood} — {'; '.join(bits)}."


# ---------------------------------------------------------------------------
# DMC-style level geometry (dumb-money-concepts, reference only → deterministic
# structure for the analysts). Candle-body levels: a level is a candle BODY
# extreme (not the wick). "Gain a level" = close beyond it; "failure to gain"
# = wick beyond, close back inside; "lose a level" = close back through.
# ---------------------------------------------------------------------------

def _body_levels(df, n=10):
    """Recent swing-high/low BODY levels from the last n candles."""
    highs, lows = [], []
    for i in range(-min(n, len(df)), 0):
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        o = float(df["Open"].iloc[i])
        c = float(df["Close"].iloc[i])
        body_high = max(o, c)
        body_low = min(o, c)
        # swing: body extreme higher/lower than neighbors
        if i > -len(df) + 1 and i < -1:
            ph, nh = float(df["High"].iloc[i-1]), float(df["High"].iloc[i+1])
            pl, nl = float(df["Low"].iloc[i-1]), float(df["Low"].iloc[i+1])
            if body_high > max(ph, nh):
                highs.append(body_high)
            if body_low < min(pl, nl):
                lows.append(body_low)
    return sorted(set(round(x, 2) for x in highs), reverse=True), \
           sorted(set(round(x, 2) for x in lows))


def build_dmc_level_block(snapshot, df=None, n=10, df_intraday=None):
    """Deterministic DMC-style structure block appended to the grounding.

    df: daily OHLCV DataFrame (HTF anchor). df_intraday: 5m/1m DataFrame for
    the session (LTF). Two-timeframe (HTF+LTF) alignment is the DMC core.
    Returns a short block the analysts may cite (labeled clearly as DMC
    structure, NOT a trading rule).
    """
    l = snapshot.get("levels") or {}
    s = snapshot.get("session") or {}
    price = s.get("price_now") or l.get("close")
    lines = []
    if df is not None and not df.empty:
        highs, lows = _body_levels(df, n)
        near_h = next((h for h in highs if h >= (price or 0)), None)   # resistance above
        near_l = next((lo for lo in lows if lo <= (price or 0)), None)  # support below
        lines.append("## DMC level structure (candle-body levels, reference structure only)")
        lines.append(f"- HTF (daily) nearest body-resistance above: {near_h if near_h else 'n/a'}")
        lines.append(f"- HTF (daily) nearest body-support below: {near_l if near_l else 'n/a'}")
        lines.append(f"- HTF recent body-highs: {highs[:3] or 'n/a'}")
        lines.append(f"- HTF recent body-lows: {lows[:3] or 'n/a'}")
        fh, fl = s.get("first30_high"), s.get("first30_low")
        if fh and fl and price:
            if price >= fh:
                lines.append(f"- session state: ABOVE first-30m range ({fl}-{fh}) — body gain attempt")
            elif price <= fl:
                lines.append(f"- session state: BELOW first-30m range ({fl}-{fh}) — body loss attempt")
            else:
                lines.append(f"- session state: RANGE-LOCK inside first-30m range ({fl}-{fh})")
    if df_intraday is not None and not df_intraday.empty:
        i_highs, i_lows = _body_levels(df_intraday, n=8)
        ih = next((h for h in i_highs if h >= (price or 0)), None)
        il = next((lo for lo in i_lows if lo <= (price or 0)), None)
        lines.append(f"- LTF (session) nearest body-resistance above: {ih if ih else 'n/a'}")
        lines.append(f"- LTF (session) nearest body-support below: {il if il else 'n/a'}")
        if ih and fh and price and price >= fh and ih <= fh * 1.001:
            lines.append("- HTF/LTF ALIGNED UP: price above daily + session body-resistance — bullish DMC alignment")
        elif il and fl and price and price <= fl and il >= fl * 0.999:
            lines.append("- HTF/LTF ALIGNED DOWN: price below daily + session body-support — bearish DMC alignment")
        elif ih and fh and price and price < fh:
            lines.append("- HTF/LTF MISALIGNED: price below daily body-resistance but session pushing up")
        elif il and fl and price and price > fl:
            lines.append("- HTF/LTF MISALIGNED: price above daily body-support but session pressing down")
    return "\n".join(lines) + "\n" if lines else ""


if __name__ == "__main__":
    import sys
    tk = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    snap = fetch_stock_snapshot(tk, session=("--session" in sys.argv))
    print(json.dumps(snap, indent=2, default=str))
