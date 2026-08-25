#!/usr/bin/env python3
"""ta-shadow STOCK lane: daily driver (pre-open TA + order staging).

Pre-open phase (12:00 UTC, Mon-Fri, before the 13:30 UTC NY open):
  1. Run the TradingAgents due-diligence on all 5 stocks (run_stock.py).
  2. Collect verdicts + trade params (entry/TP/SL).
  3. Stage bracket orders into orders.json via mexc_orders.build_bracket
     (STAGE mode — no exchange calls). EXECUTE mode (--execute) places them
     as resting limit orders with TP/SL on MEXC futures before the open.
  4. Emit a compact Home-channel summary.

NY-open phase (--screen, run by a separate cron at 13:35/15:30 UTC):
  Runs nyopen_screen.py and emits the level/opportunity table.

SAFETY: default is STAGE only. --execute must be passed explicitly to touch
the exchange. Never exceeds 1 bracket per ticker per day.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

LANE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(LANE, ".venv", "bin", "python")
RUN_STOCK = os.path.join(LANE, "run_stock.py")
SCREEN = os.path.join(LANE, "nyopen_screen.py")
ORDERS = os.path.join(LANE, "orders.json")
LOG = os.path.join(LANE, "stock-log.jsonl")

TICKERS = ["NVDA", "TSLA", "AAPL", "AMD", "SPY"]
EQUITY_USD = 37.5  # legacy fallback — NEVER used in live mode (fail closed; 2026-08-25)
RISK_PCT = 0.05  # operator-set 2026-08-24: 5% of day-start capital per trade
DAY_CAPITAL_FILE = os.path.join(LANE, "day-capital.json")


def _load_day_capital():
    """Return the persisted day-start capital dict, or None."""
    if os.path.exists(DAY_CAPITAL_FILE):
        try:
            return json.load(open(DAY_CAPITAL_FILE))
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _day_start_capital(date=None, live=True):
    """Fixed capital base for the day: first equity read is snapshotted and
    reused for every trade that day (operator directive 2026-08-24: 5% of
    STARTING capital per trade, not floating equity).

    2026-08-25 hardening: in LIVE mode a failed equity read returns None —
    the $37.5 default must never silently size a live order (Codex
    §orchestration item 6). Only non-live callers may use the fallback.
    """
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snap = _load_day_capital()
    if snap and snap.get("date") == date and snap.get("equity"):
        return float(snap["equity"])
    # first read of the day: snapshot live equity
    try:
        from mexc_orders import _mexc_request
        r = _mexc_request("GET", "/api/v1/private/account/assets", {})
        for a in (r.get("data") or []):
            if a.get("currency") == "USDT":
                eq = float(a.get("equity") or 0)
                if eq > 0:
                    tmp = DAY_CAPITAL_FILE + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump({"date": date, "equity": eq}, f)
                    os.replace(tmp, DAY_CAPITAL_FILE)
                    return eq
    except Exception:
        pass
    return None if live else EQUITY_USD


def _live_equity_usd():
    """Day-start capital (fixed base for all trades that day)."""
    return _day_start_capital()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def run_ta(ticker, date, test=False, timeout=3600):
    cmd = [VENV_PY, RUN_STOCK, "--ticker", ticker, "--date", date, "--variant", "A"]
    if test:
        cmd.append("--test")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode == 0:
            try:
                return json.loads(p.stdout)
            except json.JSONDecodeError:
                return {"ticker": ticker, "error": "bad stdout"}
        return {"ticker": ticker, "error": (p.stderr or p.stdout or "FAILED")[-500:]}
    except subprocess.TimeoutExpired:
        return {"ticker": ticker, "error": "TIMEOUT"}


def run_ta_parallel(date, test=False, timeout=3600):
    """Run the 5 TA evaluations in parallel (one subprocess per ticker).

    A single stock takes ~10-15 min of LLM calls; sequential 5-stock
    would blow the pre-open window. Parallel brings wall-clock down to
    roughly one stock's runtime.
    """
    procs = {}
    for tk in TICKERS:
        cmd = [VENV_PY, RUN_STOCK, "--ticker", tk, "--date", date, "--variant", "A"]
        if test:
            cmd.append("--test")
        logf = open(os.path.join(LANE, "results", f"ta-{tk}.log"), "ab")
        procs[tk] = (subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                      start_new_session=True), logf)
    results = []
    for tk, (p, logf) in procs.items():
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            results.append({"ticker": tk, "error": "TIMEOUT"})
            logf.close()
            continue
        logf.close()
        # parse the last JSON block from the log (run_stock prints JSON at end)
        path = os.path.join(LANE, "results", f"ta-{tk}.log")
        try:
            txt = open(path).read()
            # take the last {...} block that parses (handles both single-line
            # and pretty-printed multi-line JSON)
            parsed = None
            depth = 0
            start = None
            for i, ch in enumerate(txt):
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        try:
                            parsed = json.loads(txt[start:i+1])
                        except json.JSONDecodeError:
                            parsed = None
                        start = None
            results.append(parsed or {"ticker": tk, "error": "no JSON in log"})
        except Exception as e:
            results.append({"ticker": tk, "error": str(e)[:200]})
    return results


def stage_from_verdict(res, live=True):
    """Build a bracket payload from a TA result. Returns None if no trade.

    2026-08-25 hardening (Codex §"strict PM contract"): NO wrapper-generated
    fallback entry/TP/SL — if the PM gave no explicit levels, there is no
    trade (missing levels must become NO_TRADE, never an invented 1%/2R
    bracket). Capital unavailable in live mode also means no trade.
    """
    ticker = res.get("ticker")
    verdict = res.get("ta_verdict") or res.get("verdict")
    params = res.get("trade_params") or {}
    close = res.get("close")
    if verdict not in ("BUY", "SELL") or not close:
        return None
    side = "LONG" if verdict == "BUY" else "SHORT"
    entry = params.get("entry")
    tp = params.get("take_profit")
    sl = params.get("stop_loss")
    # missing any explicit level → NO_TRADE (no fallback levels, ever)
    if entry is None or tp is None or sl is None:
        return None
    equity = _day_start_capital(live=live)
    if equity is None or equity <= 0:
        return None  # fail closed: unknown capital sizes nothing in live mode
    from mexc_orders import build_bracket
    return build_bracket(ticker, side, entry, tp, sl,
                         equity_usd=equity, risk_pct=RISK_PCT,
                         execution=params.get("execution", "limit"),
                         invalidation=params.get("invalidation"))


def _portal_healthy(timeout=20):
    """Pre-flight: is the LLM provider (OpenRouter) actually responsive?

    Switched from Nous to OpenRouter on 2026-08-24 (operator directive) —
    OpenRouter is stable, no hourly key rotation, no capacity flaps. Still
    probe before a full graph run so a down provider bails early with a clear
    message instead of grinding.

    Uses the OpenAI SDK's native retry (max_retries=2 default): it respects the
    Retry-After header and backs off exponentially, so a single transient 429
    or dropped connection is retried automatically — that single-blip-skip was
    the 2026-08-24 12:00 UTC failure. The exception reason is logged so a real
    failure is diagnosable instead of a silent False.
    """
    try:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            # read from Hermes' .env at runtime (never stored in the lane)
            hermes_env = os.path.expanduser("~/.hermes/.env")
            for line in open(hermes_env):
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        if not key:
            print("⚠️ portal probe: no OPENROUTER_API_KEY (env or ~/.hermes/.env)")
            return False
        from openai import OpenAI
        c = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1",
                   timeout=timeout, max_retries=2)
        # Probe with a non-reasoning model that returns plain content — the
        # pinned 0731 is a reasoning model: with a small max_tokens it spends
        # the whole budget on reasoning and returns content=None, which would
        # falsely fail the health probe. Use a cheap content model instead.
        r = c.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "say OK"}], max_tokens=8)
        ok = bool(r.choices) and bool((r.choices[0].message.content or "").strip())
        if not ok:
            print("⚠️ portal probe: empty response from OpenRouter")
        return ok
    except Exception as e:
        print(f"⚠️ portal probe FAILED: {type(e).__name__}: {str(e)[:200]}")
        return False


def preopen(date, test=False, execute=False, dry=False):
    if not test and not dry and not _portal_healthy():
        print("⚠️ ta-shadow pre-open SKIPPED: LLM provider (OpenRouter) unresponsive. "
              "No orders staged. Will retry next scheduled run.")
        return 2
    # No hanging orders: cancel unfilled brackets from prior runs before a
    # fresh preopen (operator directive 2026-08-24). Positions + attached
    # TP/SL are untouched — this only clears resting entry orders.
    if execute:
        try:
            from mexc_orders import cancel_leftover_brackets
            cancel_leftover_brackets()
        except Exception as e:
            print(f"⚠️ leftover-bracket cleanup skipped: {e}")
    results = run_ta_parallel(date, test=test)
    staged = []
    for res in results:
        br = stage_from_verdict(res)
        if br:
            staged.append(br)
    # persist orders (only non-test)
    if not test:
        data = {"date": date, "generated": now_iso(), "orders": staged,
                "results": [{k: r.get(k) for k in ("ticker", "verdict", "ta_verdict", "confidence", "trade_params", "close")} for r in results]}
        tmp = ORDERS + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, ORDERS)
    if execute and not test:
        from mexc_orders import place_bracket
        for br in staged:
            try:
                resp = place_bracket(br, dry_run=dry)  # dry=True: build only, no send
                if isinstance(resp, dict) and resp.get("invalidated"):
                    br["status"] = "invalidated"
                    br["reason"] = resp.get("reason")
                elif isinstance(resp, dict) and resp.get("probe_failed"):
                    br["status"] = "probe_failed"
                    br["reason"] = resp.get("reason")
                elif isinstance(resp, dict) and resp.get("admission_rejected"):
                    br["status"] = f"admission_{resp['admission']['decision']}"
                    br["reason"] = resp["admission"].get("detail", {}).get("error") or \
                        resp["admission"]["decision"]
                else:
                    br["status"] = "dry_placed" if dry else "placed"
                    br["order_id"] = resp.get("data", {}).get("orderId")
            except Exception as e:
                br["status"] = f"error: {str(e)[:150]}"
        failed = [b for b in staged if b.get("status", "").startswith(("error", "admission_", "probe_failed"))]
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, ORDERS)
        # partial failure visibility (Codex §orchestration item 5): any
        # placement failure must be loud in the digest, not silently swallowed
        if failed:
            print("⚠️ PLACEMENT FAILURES: " + ", ".join(
                f"{b.get('ticker')} [{b.get('status')}]" for b in failed))
    # summary — polished per-asset due-diligence digest for Telegram
    verdict_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪", "UNKNOWN": "⚫"}
    vcount = {"BUY": 0, "SELL": 0, "HOLD": 0, "UNKNOWN": 0}
    day_cap = _live_equity_usd()
    cap_line = (f"💰 Day capital: ${day_cap:.0f} · 5% risk = **${round(day_cap*0.05, 2):.2f}** each"
                if day_cap else "⚠️ Day capital UNAVAILABLE — no live sizing possible (fail closed)")
    lines = [
        "🚀 *TA-SHADOW PRE-OPEN* — " + date,
        "🤖 TradingAgents due diligence · 5 stocks",
        cap_line,
        "",
    ]
    for r in results:
        v = r.get("ta_verdict") or r.get("verdict") or "UNKNOWN"
        vcount[v if v in vcount else "UNKNOWN"] += 1
        p = r.get("trade_params") or {}
        e = r.get("error")
        ticker = r.get("ticker", "?")
        if e:
            lines.append(f"❌ *{ticker}* — ERROR {e[:100]}")
            continue
        emoji = verdict_emoji.get(v, "⚫")
        conf = r.get("confidence")
        conf_s = ""
        if isinstance(conf, (int, float)) and conf > 0:
            conf_s = f" · conf {conf:.0f}"
        lines.append(f"{emoji} *{ticker}* — *{v}*{conf_s}")
        entry = p.get("entry")
        tp = p.get("take_profit")
        sl = p.get("stop_loss")
        exec_t = (p.get("execution") or "limit").lower()
        exec_s = "🎯 MARKET" if exec_t == "market" else ("🎯 LIMIT" if exec_t == "limit" else "")
        if v in ("BUY", "SELL") and entry:
            if tp and sl:
                lines.append(f"   entry {entry} · TP {tp} · SL {sl} {exec_s}")
            else:
                lines.append(f"   entry {entry} {exec_s}")
        thesis = r.get("thesis")
        if thesis:
            lines.append(f"   💬 {thesis}")
        lines.append("")
    lines.append(f"📊 Verdicts: 🟢{vcount['BUY']} · 🔴{vcount['SELL']} · ⚪{vcount['HOLD']}")
    if staged:
        placed = sum(1 for b in staged if b.get("status") == "placed")
        dry_placed = sum(1 for b in staged if b.get("status") == "dry_placed")
        if dry and execute:
            # dry run: brackets built but NOT sent to MEXC
            lines.append(f"🧪 *DRY RUN* — {len(staged)} bracket(s) built, {dry_placed} validated, NOT placed on MEXC")
            lines.append(f"   (no live orders sent — dry mode)")
        elif execute:
            lines.append(f"🎯 *{placed}/{len(staged)} bracket orders PLACED LIVE*")
            invalidated = [b.get("ticker") for b in staged if b.get("status") == "invalidated"]
            admitted_skip = [(b.get("ticker"), b.get("status")) for b in staged
                             if str(b.get("status", "")).startswith("admission_")]
            failed = [b.get("ticker") for b in staged
                      if b.get("status") not in ("placed", "invalidated")
                      and not str(b.get("status", "")).startswith("admission_")]
            if invalidated:
                lines.append(f"   ⚪ skipped (price already past SL): {', '.join(invalidated)}")
            for tk, st in admitted_skip:
                # admission governor decision, machine-readable reason
                lines.append(f"   ⛔ {tk}: skipped — governor [{st[len('admission_'):]}]")
            if failed:
                lines.append(f"   ⚠️ failed: {', '.join(failed)}")
        else:
            lines.append(f"🎯 {len(staged)} bracket(s) staged — review before execute")
    else:
        lines.append("🎯 No brackets (no validated BUY/SELL).")
    # total run cost
    total_cost = sum((r.get("cost_usd") or {}).get("mid", 0)
                     for r in results if isinstance(r.get("cost_usd"), dict))
    lines.append(f"💰 LLM cost ≈ ${total_cost:.3f}")
    print("\n".join(lines))
    return 0


def screen_phase(test=False):
    cmd = [VENV_PY, SCREEN, "--dry-run"] if test else [VENV_PY, SCREEN]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    out = (p.stdout or "").split("\n--- JSON ---")[0]
    print(out.strip() or (p.stderr or "SCREEN FAILED"))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preopen", action="store_true", help="pre-open TA + order staging")
    ap.add_argument("--screen", action="store_true", help="NY-open fast screen")
    ap.add_argument("--sweep", action="store_true",
                    help="cancel stale unfilled brackets (post-open cleanup + price invalidation)")
    ap.add_argument("--date", default=None)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--execute", action="store_true", help="place staged orders live")
    ap.add_argument("--dry", action="store_true",
                    help="full TA flow but build brackets only — NO live MEXC orders (dry run)")
    args = ap.parse_args()
    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if args.preopen:
        return preopen(date, test=args.test, execute=args.execute, dry=args.dry)
    if args.screen:
        return screen_phase(test=args.test)
    if args.sweep:
        from mexc_orders import sweep_stale_brackets, record_fill_outcomes
        rep = sweep_stale_brackets(dry_run=not args.execute)
        # Record TP/SL outcomes for placed orders that have since closed, so
        # tomorrow's pre-open agents see how their last trades ended.
        outcomes = record_fill_outcomes(dry_run=not args.execute)
        rep["outcomes"] = outcomes
        print(json.dumps(rep, indent=1))
        return 0 if not rep["errors"] else 2
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
