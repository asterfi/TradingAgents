"""ta-shadow: btc-swing snapshot loader + crypto-native data for the shadow lane.

Reads the latest btc-swing scan snapshot (log.jsonl rows with "json" key),
builds the grounding block every agent must cite, and fetches the crypto
fundamentals the stock fundamentals analyst cannot provide:
  - open interest + 5-day trend      (Binance futures, public)
  - spot BTC ETF flows               (from the btc-swing snapshot itself)
  - stablecoin supply change         (CoinGecko, public free tier)
  - exchange netflows                (no free public source -> honest N/A)

Everything here is read-only public data. No keys, no trade permissions.
"""
import json
import os
import urllib.request
from datetime import datetime, timezone

BTC_SWING_LOG = os.path.expanduser("~/.hermes/skills/btc-swing/log.jsonl")
BINANCE_FAPI = "https://fapi.binance.com"
COINGECKO = "https://api.coingecko.com/api/v3"

_UA = {"User-Agent": "Mozilla/5.0 (research; ta-shadow lane)"}


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def load_snapshot(path=None):
    """Return the latest btc-swing scan snapshot dict (the 'json' field of the
    last log.jsonl row that has one). Raises if none found."""
    path = path or BTC_SWING_LOG
    latest = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("json"), dict):
                latest = row["json"]
    if latest is None:
        raise RuntimeError(f"no scan snapshot found in {path}")
    return latest


def _fmt(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def extract_symbol_block(snapshot, symbol):
    """Top-level context + the per-asset block for `symbol` (e.g. BTCUSDT)."""
    assets = snapshot.get("assets", {})
    asset = assets.get(symbol)
    if asset is None:
        # fall back to base-name match (BTCUSDT vs BTC)
        base = symbol.replace("USDT", "")
        for k, v in assets.items():
            if v.get("base") == base:
                asset = v
                symbol = k
                break
    if asset is None:
        raise KeyError(f"symbol {symbol} not in snapshot assets: {sorted(assets)}")
    return symbol, asset


def build_grounding_block(snapshot, symbol, variant="A", position_blind=False):
    """The canonical btc-swing snapshot text block. Agents may cite ONLY
    figures present here (plus the fundamentals block).

    variant: "A" = canonical bullet order; "B" = reversed bullet order
    (Phase 3 addition #2 — same content, different prompt ordering, used as
    the non-determinism probe).
    position_blind: strip the operator's position/equity/risk lines so the
    run judges the signal, not the trade (Phase 3 addition #3).
    """
    symbol, a = extract_symbol_block(snapshot, symbol)
    ev = snapshot.get("events_line") or "—"
    etf = snapshot.get("etf_flow_line") or "—"
    pos = a.get("position") or {}
    extras = []
    if pos.get("leverage") is not None:
        extras.append(f"leverage {pos.get('leverage')}")
    if pos.get("liq") is not None:
        extras.append(f"liq {pos.get('liq')}")
    st = pos.get("status") or (pos.get("mexc") or {}).get("status")
    if st:
        extras.append(f"status {st}")
    pos_txt = (
        f"operator position: {pos.get('side')} entry {pos.get('entry')} stop {pos.get('stop')} "
        f"tp1 {pos.get('tp1')} size {pos.get('size')} uPnL {pos.get('u_pnl')}"
        + (" " + " ".join(extras) if extras else "")
        if pos else "position: none"
    )
    lines = [
        f"- ticker: {a.get('symbol')} ({a.get('base')})",
        f"- candle_date (UTC): {a.get('candle_date')}",
        f"- close (completed daily candle): {a.get('close')}",
        f"- price_now (live ticker, reference only): {a.get('price_now')}",
        f"- trend: {a.get('trend')} (close vs 50d SMA)",
        f"- sma50: {a.get('sma50')}",
        # high20/low20 are raw 20d extremes — deliberately no LONG/SHORT trigger
        # labels and NO btc-swing setup/signal_side line: the agents must judge
        # the data, not btc-swing's deterministic conclusion (lane integrity).
        f"- high20 (20d high): {a.get('high20')}",
        f"- low20 (20d low): {a.get('low20')}",
        f"- vol_ratio (last completed candle vol / 20d avg): {a.get('vol_ratio')}",
        f"- volume_24h_usd: {a.get('volume_24h_usd')}",
        f"- funding (8h, BTC perp): {a.get('funding')} ({'EXTREME' if a.get('funding_extreme') else 'normal'})",
        f"- macro regime: {snapshot.get('macro')}",
        f"- next events: {ev}",
        f"- ETF flows: {etf}",
    ]
    if not position_blind:
        lines.append(f"- {pos_txt}")
        lines.append(f"- equity: {snapshot.get('equity')} | risk_per_trade: {snapshot.get('risk_per_trade')}")
    if variant == "B":
        lines.reverse()
    return "## btc-swing snapshot (source of truth for ALL figures)\n" + "\n".join(lines) + "\n"


def fetch_open_interest(symbol):
    """Current OI + 5-day trend, Binance futures public endpoints."""
    try:
        hist = _get(f"{BINANCE_FAPI}/futures/data/openInterestHist?symbol={symbol}&period=1d&limit=5")
        cur = _get(f"{BINANCE_FAPI}/fapi/v1/openInterest?symbol={symbol}")
    except Exception as exc:
        return f"Open interest: unavailable (fetch failed: {exc.__class__.__name__})"
    oi_now = float(cur.get("openInterest", 0))
    days = [{"ts": datetime.fromtimestamp(d["timestamp"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
             "oi": float(d["sumOpenInterest"]), "usd": float(d["sumOpenInterestValue"])} for d in hist]
    if len(days) >= 2:
        chg = (days[-1]["oi"] / days[0]["oi"] - 1) * 100
    else:
        chg = 0.0
    trend = " | ".join(f"{d['ts']}: {d['oi']:.1f} BTC" for d in days)
    return (f"Open interest (Binance futures {symbol}): current {oi_now:.1f} {symbol.replace('USDT','')} "
            f"({days[-1]['usd']/1e9:.2f}B USD); 5-day trend {chg:+.2f}% over {len(days)} daily points [{trend}]")


def fetch_stablecoin_supply():
    """USDT+USDC market cap and 24h change (CoinGecko free tier)."""
    try:
        data = _get(f"{COINGECKO}/coins/markets?vs_currency=usd&ids=tether,usd-coin&per_page=5&page=1")
    except Exception as exc:
        return f"Stablecoin supply: unavailable (fetch failed: {exc.__class__.__name__})"
    rows = {d["symbol"].upper(): d for d in data}
    out = []
    for sym in ("USDT", "USDC"):
        d = rows.get(sym)
        if d:
            out.append(f"{sym}: ${d['market_cap']/1e9:.1f}B (24h change {d.get('market_cap_change_percentage_24h', 0):+.2f}%)")
    return "Stablecoin supply: " + "; ".join(out) if out else "Stablecoin supply: unavailable"


def exchange_netflows_text():
    return ("Exchange netflows: NOT AVAILABLE — no free public source exists "
            "(CryptoQuant/Glassnode/Nansen are paid; exchange balance history is not published freely).")


def build_crypto_fundamentals_block(snapshot, symbol):
    """Figures + short factual read for the crypto fundamentals analyst.
    NO price prediction — figures and context only."""
    symbol, a = extract_symbol_block(snapshot, symbol)
    funding = a.get("funding")
    funding_txt = (
        f"funding rate {funding} per 8h ({'EXTREME' if a.get('funding_extreme') else 'normal'})"
        if funding is not None else "funding rate unavailable"
    )
    oi = fetch_open_interest(symbol)
    stables = fetch_stablecoin_supply()
    netflows = exchange_netflows_text()
    etf = snapshot.get("etf_flow_line", "—")
    return (
        f"## Crypto fundamentals (figures only, factual read)\n"
        f"- {funding_txt}\n"
        f"- {oi}\n"
        f"- Spot BTC ETF flows: {etf}\n"
        f"- {stables}\n"
        f"- {netflows}\n"
    )


def build_crowd_mood_line(snapshot, symbol):
    """Deterministic single-line crowd mood (Phase 2.3 fence). Zero LLM cost.
    Labeled separately; downstream agents are forbidden from citing it."""
    symbol, a = extract_symbol_block(snapshot, symbol)
    bits = []
    f = a.get("funding")
    if f is not None:
        bits.append(f"perp funding {f:+.5f} ({'crowd long-biased' if f > 0 else 'crowd short-biased'})")
    etf5 = snapshot.get("etf_5d_net_m")
    if etf5 is not None:
        bits.append(f"5d ETF net {etf5:+.1f}m ({'institutional bid' if etf5 > 0 else 'institutional offer'})")
    bits.append(f"macro {snapshot.get('macro')}")
    mood = "risk-on" if snapshot.get("macro") == "risk_on" else ("risk-off" if snapshot.get("macro") == "risk_off" else "mixed")
    return f"Crowd mood: {mood} — {'; '.join(bits)}."


# ---------------------------------------------------------------------------
# Stock lane (2026-08-23): NVDA/TSLA/AAPL/AMD/SPY, NY-open focus.
# Separate from the btc-swing snapshot format above: a STOCK snapshot is a
# single-symbol dict produced by ta-shadow/stock_snapshot.py, with a "levels"
# sub-dict of key levels (pivots, SMAs, gap, VWAP, ATR, volume pace).
# The btc-swing lane is untouched; btc-swing itself stays paused.
# ---------------------------------------------------------------------------

STOCK_WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMD", "SPY"]


def _lv(l, key, nd=2):
    v = (l or {}).get(key)
    return _fmt(v, nd) if v is not None else "n/a"


def build_stock_grounding_block(snapshot, variant="A"):
    """The canonical stock snapshot text block. Agents may cite ONLY figures
    present here. `snapshot` is the per-symbol stock snapshot dict."""
    l = snapshot.get("levels") or {}
    lines = [
        f"- ticker: {snapshot.get('symbol')} ({snapshot.get('name', 'n/a')})",
        f"- as_of (last completed daily candle): {snapshot.get('as_of')}",
        f"- close (last daily close): {_lv(l, 'close')}",
        f"- prior close: {_lv(l, 'prior_close')}",
        f"- 5d return: {_lv(l, 'ret_5d', 2)}%",
        f"- trend: {snapshot.get('trend', 'n/a')} (close vs 50d SMA)",
        f"- sma20: {_lv(l, 'sma20')} | sma50: {_lv(l, 'sma50')} | sma200: {_lv(l, 'sma200')}",
        f"- 20d high/low: {_lv(l, 'high20')} / {_lv(l, 'low20')}",
        f"- 52w high/low: {_lv(l, 'high52w')} / {_lv(l, 'low52w')}",
        f"- ATR14: {_lv(l, 'atr14')}",
        f"- pivots (prior day): P {_lv(l, 'pivot_p')} | R1 {_lv(l, 'r1')} | R2 {_lv(l, 'r2')} | S1 {_lv(l, 's1')} | S2 {_lv(l, 's2')}",
        f"- 20d avg volume: {_lv(l, 'avg_vol_20d', 0)}",
        f"- earnings date: {snapshot.get('earnings_date', 'n/a')}",
    ]
    if l.get("session_open") is not None:
        lines += [
            f"- session open: {_lv(l, 'session_open')}",
            f"- current price: {_lv(l, 'price_now')}",
            f"- gap vs prior close: {_lv(l, 'gap_pct', 2)}%",
            f"- session VWAP: {_lv(l, 'vwap')}",
            f"- first-30m high/low: {_lv(l, 'first30_high')} / {_lv(l, 'first30_low')}",
            f"- session volume pace (vs 20d avg, elapsed-adjusted): {_lv(l, 'vol_pace', 2)}x",
        ]
    if variant == "B":
        lines.reverse()
    return "## Stock snapshot (source of truth for ALL figures)\n" + "\n".join(lines) + "\n"


def build_stock_mood_line(snapshot):
    """Deterministic single-line crowd mood for a stock (gap + trend bias).
    Labeled separately; downstream agents are forbidden from citing it."""
    l = snapshot.get("levels") or {}
    bits = []
    gap = l.get("gap_pct")
    if gap is not None:
        bits.append(f"gap {gap:+.2f}% ({'buy-side open' if gap > 0 else 'sell-side open'})")
    bits.append(f"trend {snapshot.get('trend', 'n/a')}")
    mood = "risk-on" if (gap or 0) > 0 and snapshot.get("trend") == "above_50" else ("risk-off" if (gap or 0) < 0 and snapshot.get("trend") == "below_50" else "mixed")
    return f"Crowd mood: {mood} — {'; '.join(bits)}."
