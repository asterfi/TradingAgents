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


def fetch_stock_snapshot(ticker, lookback=420, session=False):
    """Build the stock snapshot dict for one ticker.

    lookback=420 calendar days ≈ 280+ trading sessions — enough completed
    history for a true 52-week high/low and SMA200 (2026-08-25 fix: the old
    90-day fetch labeled its extremes "52w" and computed a garbage SMA200).
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
    # 52-week extremes need ~252 completed SESSIONS; verify we actually have
    # them and never relabel a shorter window (Codex §snapshot-history).
    sessions_52w = d.tail(280)
    have_52w = len(sessions_52w) >= 250
    high52 = float(sessions_52w["High"].max()) if have_52w else None
    low52 = float(sessions_52w["Low"].min()) if have_52w else None
    close = float(last["Close"])
    prior_close = float(prev["Close"])
    avg_vol_20d = float(d["Volume"].tail(20).mean())
    ret_5d = (close / float(d["Close"].iloc[-6]) - 1) * 100 if len(d) >= 6 else None

    sma200 = _sma(d["Close"], 200)
    sma200 = round(sma200, 2) if sma200 and len(d.dropna()) >= 200 else None

    levels = {
        "close": round(close, 2),
        "prior_close": round(prior_close, 2),
        "ret_5d": round(ret_5d, 2) if ret_5d is not None else None,
        "sma20": round(_sma(d["Close"], 20), 2) if _sma(d["Close"], 20) else None,
        "sma50": round(_sma(d["Close"], 50), 2) if _sma(d["Close"], 50) else None,
        "sma200": sma200,
        "high20": round(high20, 2), "low20": round(low20, 2),
        "high52w": round(high52, 2) if high52 is not None else None,
        "low52w": round(low52, 2) if low52 is not None else None,
        "atr14": round(_atr(d), 2) if _atr(d) else None,
        "avg_vol_20d": int(avg_vol_20d),
        **_pivots(last["High"], last["Low"], last["Close"]),
    }
    trend = "above_50" if close > (levels["sma50"] or 0) else "below_50"

    snap = {
        "kind": "stock",
        "symbol": ticker,
        "name": None,
        "as_of": d.index[-1].strftime("%Y-%m-%d"),
        "trend": trend,
        "earnings_date": None,
        "levels": levels,
        "session": {},
        "dmc": "",
    }
    try:
        snap["name"] = t.info.get("shortName")
    except Exception:
        pass  # ETFs (SPY) have no fundamentals — name stays None
    try:
        cal = getattr(t, "calendar", None)
        if cal is not None:
            snap["earnings_date"] = str(cal.get_earnings_dates(limit=2).index[0].date())
    except Exception:
        pass  # ETFs (SPY) have no earnings calendar — leave None

    if session:
        _fill_session(snap, ticker)

    # DMC block (HTF daily + LTF intraday) built AFTER session fill.
    # Note: build_dmc_level_block writes structured levels back into the dict
    # it is given, so copy them onto the real snapshot afterwards.
    dmc_ctx = {"levels": levels, "session": snap["session"]}
    snap["dmc"] = build_dmc_level_block(
        dmc_ctx, df=d, df_intraday=snap.pop("_intraday_df", None),
    )
    snap["dmc_levels"] = dmc_ctx.get("dmc_levels") or {}
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
    """DMC body levels from the last n candles (QUALIFIED only, 2026-08-25).

    A DMC level is a candle BODY extreme that price actually interacted with.
    Old behavior promoted nearly every candle with a wick — that is noise,
    not structure (Codex §"Correct existing geometry defects first").

    A body extreme qualifies through at least ONE strong condition:
      - confirmed body pivot: body extreme pokes past BOTH neighbors' bodies
      - close-through: the candle CLOSED beyond the prior body extreme
        (a "gained" level)
      - failed close beyond: wick poked past but closed back inside (a
        "tested/failed" level) — only when the poke is beyond the PRIOR
        candle's body extreme, not any ordinary intrabar wick
    Returns (highs, lows) sorted desc/asc.
    """
    highs, lows = [], []
    arr = df.tail(n)
    for i in range(len(arr)):
        o = float(arr["Open"].iloc[i]); c = float(arr["Close"].iloc[i])
        h = float(arr["High"].iloc[i]); l = float(arr["Low"].iloc[i])
        body_high = max(o, c); body_low = min(o, c)
        prev = arr.iloc[i - 1] if i > 0 else None
        nxt = arr.iloc[i + 1] if i < len(arr) - 1 else None
        # 1) confirmed body pivot: body extreme beyond both neighbors' BODIES
        if prev is not None and nxt is not None:
            pbh = max(float(prev["Open"]), float(prev["Close"]))
            pbl = min(float(prev["Open"]), float(prev["Close"]))
            nbh = max(float(nxt["Open"]), float(nxt["Close"]))
            nbl = min(float(nxt["Open"]), float(nxt["Close"]))
            if body_high > max(pbh, nbh):
                highs.append(body_high)
            if body_low < min(pbl, nbl):
                lows.append(body_low)
        # 2) close-through: closed beyond the prior body extreme (gained)
        if prev is not None:
            pbh = max(float(prev["Open"]), float(prev["Close"]))
            pbl = min(float(prev["Open"]), float(prev["Close"]))
            if c > pbh:
                highs.append(body_high)
            if c < pbl:
                lows.append(body_low)
            # 3) failed close beyond: wick past the prior BODY extreme but
            # closed back inside it (rejection at the level)
            if h > pbh and c <= pbh:
                highs.append(pbh)
            if l < pbl and c >= pbl:
                lows.append(pbl)
    return sorted(set(round(x, 2) for x in highs), reverse=True), \
           sorted(set(round(x, 2) for x in lows))


def _cluster_levels(levels, atr, tol_mult=0.15):
    """Cluster near-duplicate levels within an ATR-relative tolerance
    (~0.10-0.20 ATR) instead of fixed cents. Returns clustered means,
    order preserved."""
    if not levels or not atr or atr <= 0:
        return levels
    tol = tol_mult * atr
    out = []
    for x in levels:  # levels arrive sorted (desc for highs, asc for lows)
        if out and abs(x - out[-1]) <= tol:
            out[-1] = round((out[-1] + x) / 2, 2)  # merge into the cluster
        else:
            out.append(x)
    return out


def _classify_level_interaction(df, level, side, n=30):
    """Classify how price interacted with a level over the last n candles.

    side: 'resistance' (level above) or 'support' (level below).
    Returns one of: gained / failed_to_gain / lost / reclaimed / tested.
    DMC semantics (video 5): gain = close beyond; failure = wick beyond but
    close back inside; lose = close back through after an earlier gain;
    reclaim = lost then re-entered (closed beyond again).

    2026-08-25 fix: 'lost' and 'reclaimed' were documented but never
    returned. We now walk the window chronologically and detect the
    transition sequence properly.
    """
    arr = df.tail(n)
    beyond = 0
    close_beyond = 0
    close_back_inside_after_gain = False
    regained_after_loss = False
    had_close_beyond = False
    for i in range(len(arr)):
        h = float(arr["High"].iloc[i])
        l = float(arr["Low"].iloc[i])
        c = float(arr["Close"].iloc[i])
        if side == "resistance":
            if h > level:
                beyond += 1
            if c > level:
                close_beyond += 1
                if close_back_inside_after_gain:
                    regained_after_loss = True
                had_close_beyond = True
            elif had_close_beyond:
                close_back_inside_after_gain = True
        else:
            if l < level:
                beyond += 1
            if c < level:
                close_beyond += 1
                if close_back_inside_after_gain:
                    regained_after_loss = True
                had_close_beyond = True
            elif had_close_beyond:
                close_back_inside_after_gain = True
    if beyond == 0 and close_beyond == 0:
        return "tested"
    if close_beyond == beyond and beyond > 0:
        return "gained"
    if regained_after_loss:
        return "reclaimed"
    if close_back_inside_after_gain:
        return "lost"
    if close_beyond == 0:
        return "failed_to_gain"
    # mixed: some closes beyond, some back inside, no full loss sequence
    return "reclaimed" if close_beyond < beyond else "gained"


def build_dmc_level_block(snapshot, df=None, n=10, df_intraday=None):
    """Deterministic DMC structure block (dual-window, 2026-08-25 redesign).

    DUAL-WINDOW LADDER (Codex §"DMC daily-candle redesign"):
      - ACTIVE  (10D): recent structure — preferred for the nearest level
      - STRUCT  (30D): structural context — confirmation / next target

    Nearest-level selection is PROXIMITY-CORRECT (the old code picked the
    farthest extreme: highs sorted desc -> first h >= price was the HIGHEST,
    not the nearest). Levels are ATR-clustered, ranked, emitted with STABLE
    IDs (DMC_ACTIVE_R1, DMC_STRUCT_S1, ...) in the text AND in
    snapshot["dmc_levels"] as machine-readable fields.

    No HTF/LTF alignment claims when intraday data is absent (the pre-open
    12:00 UTC path has only the daily block).
    """
    l = snapshot.get("levels") or {}
    s = snapshot.get("session") or {}
    price = s.get("price_now") or l.get("close")
    atr = l.get("atr14") or 0
    lines = []
    levels_out = {"active": {"resistance": [], "support": []},
                  "struct": {"resistance": [], "support": []}}
    if df is not None and not df.empty and price:
        # --- dual window ------------------------------------------------
        act_h, act_l = _body_levels(df, 10)
        str_h, str_l = _body_levels(df, 30)
        act_h, act_l = _cluster_levels(act_h, atr), _cluster_levels(act_l, atr)
        str_h, str_l = _cluster_levels(str_h, atr), _cluster_levels(str_l, atr)
        # PROXIMITY-CORRECT nearest levels (fix of the old farthest-extreme bug)
        above = [x for x in act_h + str_h if x > price]
        below = [x for x in act_l + str_l if x < price]
        near_h = min(above) if above else None          # nearest resistance ABOVE
        near_l = max(below) if below else None          # nearest support BELOW
        # ranked small sets per side (nearest first, capped)
        res_ranked = sorted(set(act_h + str_h), reverse=True)[:6]
        sup_ranked = sorted(set(act_l + str_l))[:6]
        # interaction states for the two nearest levels
        st_h = _classify_level_interaction(df, near_h, "resistance", 30) if near_h else None
        st_l = _classify_level_interaction(df, near_l, "support", 30) if near_l else None
        # bias read
        bias = "neutral"
        if st_h in ("failed_to_gain", "lost"):
            bias = "bearish"
        elif st_l in ("failed_to_gain", "lost"):
            bias = "bullish"
        elif st_h == "gained" and price > near_h:
            bias = "bullish"
        elif st_l == "gained" and price < near_l:
            bias = "bearish"

        def tier_of(x):
            """'A' when the level is in the active (10D) set, else 'S'."""
            return "A" if (x in set(act_h) or x in set(act_l)) else "S"

        lines.append("## DMC level structure (dual-window: 10D active + 30D structural; reference only)")
        # stable-ID ladder: nearest-first on each side, up to 3 each
        for i, x in enumerate(sorted([r for r in res_ranked if r > price])[:3], 1):
            tier = tier_of(x)
            st = _classify_level_interaction(df, x, "resistance", 30)
            lid = f"DMC_{'ACTIVE' if tier == 'A' else 'STRUCT'}_R{i}"
            lines.append(f"- {lid}: {x:.2f} — {'10D' if tier == 'A' else '30D'} body resistance, {st}")
            levels_out["active" if tier == "A" else "struct"]["resistance"].append(
                {"id": lid, "price": x, "state": st})
        for i, x in enumerate(sorted([v for v in sup_ranked if v < price], reverse=True)[:3], 1):
            tier = tier_of(x)
            st = _classify_level_interaction(df, x, "support", 30)
            lid = f"DMC_{'ACTIVE' if tier == 'A' else 'STRUCT'}_S{i}"
            lines.append(f"- {lid}: {x:.2f} — {'10D' if tier == 'A' else '30D'} body support, {st}")
            levels_out["active" if tier == "A" else "struct"]["support"].append(
                {"id": lid, "price": x, "state": st})
        lines.append(f"- DMC bias (level interaction): {bias}")
        fh, fl = s.get("first30_high"), s.get("first30_low")
        if fh and fl:
            if price >= fh:
                lines.append(f"- session state: ABOVE first-30m range ({fl}-{fh}) — body gain attempt")
            elif price <= fl:
                lines.append(f"- session state: BELOW first-30m range ({fl}-{fh}) — body loss attempt")
            else:
                lines.append(f"- session state: RANGE-LOCK inside first-30m range ({fl}-{fh})")
        else:
            lines.append("- intraday/session data absent — no LTF confirmation available")
    if df_intraday is not None and not df_intraday.empty:
        i_highs, i_lows = _body_levels(df_intraday, n=8)
        i_above = [x for x in i_highs if price and x > price]
        i_below = [x for x in i_lows if price and x < price]
        ih = min(i_above) if i_above else None
        il = max(i_below) if i_below else None
        st_ih = _classify_level_interaction(df_intraday, ih, "resistance", 8) if ih else None
        st_il = _classify_level_interaction(df_intraday, il, "support", 8) if il else None
        lines.append(f"- LTF (session) nearest body-resistance above: {ih if ih else 'n/a'} [{st_ih or 'n/a'}]")
        lines.append(f"- LTF (session) nearest body-support below: {il if il else 'n/a'} [{st_il or 'n/a'}]")
        fh, fl = s.get("first30_high"), s.get("first30_low")
        if ih and fh and price and price >= fh and ih <= fh * 1.001:
            lines.append("- HTF/LTF ALIGNED UP: price above daily + session body-resistance — bullish DMC alignment")
        elif il and fl and price and price <= fl and il >= fl * 0.999:
            lines.append("- HTF/LTF ALIGNED DOWN: price below daily + session body-support — bearish DMC alignment")
        elif ih and fh and price and price < fh:
            lines.append("- HTF/LTF MISALIGNED: price below daily body-resistance but session pushing up")
        elif il and fl and price and price > fl:
            lines.append("- HTF/LTF MISALIGNED: price above daily body-support but session pressing down")
    snapshot["dmc_levels"] = levels_out
    return "\n".join(lines) + "\n" if lines else ""


if __name__ == "__main__":
    import sys
    tk = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    snap = fetch_stock_snapshot(tk, session=("--session" in sys.argv))
    print(json.dumps(snap, indent=2, default=str))
