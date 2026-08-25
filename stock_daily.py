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
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

LANE = os.path.dirname(os.path.abspath(__file__))
STRATEGY_VERSION = "ta-shadow/ny-preopen-1"  # stamped on every staged decision
VENV_PY = os.path.join(LANE, ".venv", "bin", "python")
RUN_STOCK = os.path.join(LANE, "run_stock.py")
SCREEN = os.path.join(LANE, "nyopen_screen.py")
ORDERS = os.path.join(LANE, "orders.json")
LOG = os.path.join(LANE, "stock-log.jsonl")
LOCK_FILE = os.path.join(LANE, ".lane.lock")


def _acquire_singleton_lock(kind="lane"):
    """Singleton lock (Codex §orchestration item 3): overlapping pre-open or
    sweep/placement invocations are rejected, not interleaved."""
    try:
        f = open(LOCK_FILE, "w")
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        f.write(f"{kind} {os.getpid()} {datetime.now(timezone.utc).isoformat()}\n")
        f.flush()
        return f  # keep the handle open for the process lifetime
    except OSError:
        print(f"⛔ another ta-shadow run holds the lane lock — aborting")
        return None

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
    lock = _acquire_singleton_lock("preopen")
    if lock is None:
        return 3
    # --- NY pre-open lane (2026-08-26 restructure) ------------------------
    # Analysis runs 08:00-09:00 ET on a trading day; decisions stage but
    # NEVER execute during this phase (brief §4: no placement during
    # analysis/staging). Execution is the separate 09:20 ET executor phase.
    import nyse_calendar as nyc
    if not test and not dry:
        if not nyc.is_trading_day(nyc._et_today()[0]):
            print(f"⛔ preopen refused: {nyc.et_date_str()} is not an NYSE trading day")
            return 4
        if not nyc.in_analysis_window():
            print(f"⛔ preopen refused: outside the 08:00-09:00 ET analysis window "
                  f"(staging deadline {nyc.staging_deadline_et().isoformat()})")
            return 4
    # 09:15 staging deadline: a late graph result can never stage (brief §4)
    stage_deadline_passed = nyc.past_staging_deadline()
    if execute and not dry and not test:
        print("⛔ preopen: execution is not part of this phase anymore — "
              "staged decisions execute at 09:20 ET via --execute-staged")
        return 4
    if not test and not dry and not _portal_healthy():
        print("⚠️ ta-shadow pre-open SKIPPED: LLM provider (OpenRouter) unresponsive. "
              "No orders staged. Will retry next scheduled run.")
        return 2
    # No hanging orders: cancel unfilled brackets from prior runs before a
    # fresh preopen (operator directive 2026-08-24). Positions + attached
    # TP/SL are untouched — this only clears resting entry orders.
    if not test and not dry:
        try:
            from mexc_orders import cancel_leftover_brackets
            cancel_leftover_brackets()
        except Exception as e:
            print(f"⚠️ leftover-bracket cleanup skipped: {e}")
    results = run_ta_parallel(date, test=test)
    staged = []
    skipped_late = []
    for res in results:
        if res.get("error"):
            # timed-out/errored ticker: ineligible for this session (brief §4)
            skipped_late.append((res.get("ticker"), "analysis_incomplete"))
            continue
        if stage_deadline_passed and not test and not dry:
            # arrived after 09:15 ET: never eligible, not retroactively (§4)
            skipped_late.append((res.get("ticker"), "analysis_deadline_exceeded"))
            continue
        br = stage_from_verdict(res)
        if br:
            # staged-decision metadata (brief §4): ticker, trade date,
            # strategy version, analysis completion + expiry
            br["trade_date"] = date
            br["strategy_version"] = STRATEGY_VERSION
            br["analysis_completed_at"] = now_iso()
            br["expires_at"] = (nyc.staging_deadline_et()
                                + timedelta(minutes=5)).astimezone(timezone.utc).isoformat()
            staged.append(br)
    # persist orders (only non-test)
    if not test:
        data = {"date": date, "generated": now_iso(), "orders": staged,
                "results": [{k: r.get(k) for k in ("ticker", "verdict", "ta_verdict", "confidence", "trade_params", "close")} for r in results]}
        tmp = ORDERS + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, ORDERS)
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
        if skipped_late:
            for tk, why in skipped_late:
                lines.append(f"   ⛔ {tk}: not staged — {why}")
        if staged:
            lines.append(f"🎯 {len(staged)} decision(s) staged for the 09:20 ET executor")
    else:
        lines.append("🎯 No brackets (no validated BUY/SELL).")
    # total run cost
    total_cost = sum((r.get("cost_usd") or {}).get("mid", 0)
                     for r in results if isinstance(r.get("cost_usd"), dict))
    lines.append(f"💰 LLM cost ≈ ${total_cost:.3f}")
    print("\n".join(lines))
    return 0


def execute_staged(date, now=None):
    """09:20 ET pre-open executor (2026-08-26 NY pre-open restructure).

    Deterministic fresh-quote revalidation + IMMEDIATE placement — NOT a
    second LLM analysis (brief §5). Per order, before submission:
    reconcile live MEXC state (fail closed), load only completed
    current-date staged decisions created before the 09:15 deadline, take a
    fresh executable quote (best ask for longs, best bid for shorts),
    sanity-check it, re-run all governor checks, recompute quantity/risk at
    the executable price, re-check geometry + 1R + drift, then place
    immediately. Ranking (brief §7) picks the best candidates when there
    are more signals than slots. Entry cutoff 09:25 ET.
    """
    lock = _acquire_singleton_lock("execute")
    if lock is None:
        return 3
    import nyse_calendar as nyc
    now = now or datetime.now(timezone.utc)
    if not nyc.is_trading_day(nyc._et_today(now)[0]):
        print(f"⛔ executor refused: {nyc.et_date_str(now)} is not an NYSE trading day")
        return 4
    if not nyc.in_exec_window(now):
        print("⛔ executor refused: outside the 09:20-09:25 ET execution window "
              f"(staging deadline was {nyc.staging_deadline_et(now).isoformat()})")
        return 4
    from mexc_orders import load_orders, place_bracket, save_orders, refresh_quote
    data = load_orders()
    # staged decisions are valid only for the associated NYSE trading date
    # and must have been created before the 09:15 ET deadline (brief §4/§10)
    today_et = nyc.et_date_str(now)
    if data.get("date") != date or date != today_et:
        print(f"⛔ no current-date staged orders for {today_et} "
              f"(orders.json date: {data.get('date')}, arg date: {date})")
        return 5
    deadline = nyc.staging_deadline_et(now)
    candidates = []
    for o in data.get("orders", []):
        if o.get("status") != "staged":
            continue
        created = o.get("analysis_completed_at")
        if created and _parse_iso(created) and _parse_iso(created) > deadline:
            o["status"] = "rejected"; o["reason"] = "analysis_deadline_exceeded"
            continue
        if o.get("trade_date") and o.get("trade_date") != today_et:
            o["status"] = "rejected"; o["reason"] = "stage_expired"
            continue
        candidates.append(o)
    if not candidates:
        print("no eligible staged orders to execute")
        save_orders(data)
        return 0

    # --- deterministic ranking (brief §7) ------------------------------
    # Never ticker-list order / completion order / dict order. Rank by
    # validator quality, refreshed reward-to-risk, adverse entry drift,
    # spread/slippage — with ticker symbol as the FINAL tie-breaker only.
    ranked, rejections = _rank_candidates(candidates, now)
    slots, slot_reason = _available_slots()
    admitted = ranked[:slots] if slots > 0 else []
    skip_note = (f"{len(ranked)} eligible, {slots} slots, admitted "
                 f"{[a['ticker'] for a in admitted]}")
    print(skip_note)
    placed_any = False
    for br in admitted:
        # 09:25 entry cutoff: check the clock immediately before submitting
        if not nyc.in_exec_window(datetime.now(timezone.utc)):
            br["status"] = "rejected"; br["reason"] = "entry_cutoff"
            continue
        try:
            resp = place_bracket(br, dry_run=False, now=now)
            placed_any = True
            _apply_placement_result(br, resp)
        except Exception as e:
            br["status"] = f"error: {str(e)[:150]}"
    # lower-ranked candidates that did not fit the slots (brief §7)
    for br in ranked[slots:]:
        br["status"] = "rejected"; br["reason"] = "lower_ranked_than_available_slots"
    for br, why in rejections:
        br["status"] = "rejected"; br["reason"] = why
    save_orders(data)
    for br in candidates:
        print(f"  {br.get('ticker')}: {br.get('status')} "
              f"({br.get('reason') or 'ok'})")
    failed = [b for b in candidates if str(b.get("status", "")).startswith(
        ("error", "admission_", "probe_failed", "PROTECTION_FAILED"))]
    if failed:
        print("⚠️ EXECUTION FAILURES: " + ", ".join(
            f"{b.get('ticker')} [{b.get('status')}]" for b in failed))
        return 2
    return 0


def _parse_iso(s):
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _apply_placement_result(br, resp):
    if isinstance(resp, dict) and resp.get("invalidated"):
        br["status"] = "rejected"; br["reason"] = resp.get("reason", "invalidated")
    elif isinstance(resp, dict) and resp.get("probe_failed"):
        br["status"] = "rejected"; br["reason"] = "quote_unavailable"
    elif isinstance(resp, dict) and resp.get("admission_rejected"):
        br["status"] = "rejected"
        br["reason"] = (resp.get("admission") or {}).get("decision", "admission_rejected")
    elif isinstance(resp, dict) and resp.get("duplicate_suppressed"):
        br["status"] = "rejected"; br["reason"] = "duplicate_order"
    else:
        br["status"] = "placed"
        br["order_id"] = resp.get("data", {}).get("orderId")
        if isinstance(resp, dict) and resp.get("protection_failed"):
            br["status"] = "rejected"
            br["reason"] = "protection_unconfirmed"
            br["detail"] = (resp.get("protection") or {}).get("reason")


def _available_slots():
    """Position slots under the 3-cap AFTER existing positions + pending
    entries consume theirs (brief §7). Fail closed on unknown state."""
    import admission
    try:
        state = admission.fetch_live_state()
    except admission.LiveStateError as e:
        return 0, f"state_unknown: {e}"
    n_open = len(state["positions"]) + len([1 for v in state["orders"].values() if v])
    return max(0, 3 - n_open), f"open={n_open}"


def _rank_candidates(candidates, now):
    """Deterministic ranking with machine-readable pre-rejections.

    Refreshed reward-to-risk uses the live quote (already fetched for
    ranking); adverse drift compares the executable price with the PM's
    planned entry. Ticker symbol is the final stable tie-breaker.
    """
    import nyse_calendar as nyc
    from mexc_orders import refresh_quote as _refresh_quote
    ranked, rejections = [], []
    for br in candidates:
        ticker = br["ticker"]
        side = br["side"]
        try:
            q = _refresh_quote(br["mexc_symbol"])
        except Exception:
            rejections.append((br, "quote_unavailable"))
            continue
        px = q["bid_or_ask"][side] if q.get("bid_or_ask") else q.get("price")
        if px is None:
            rejections.append((br, "quote_unavailable")); continue
        if q.get("age_s") is not None and q["age_s"] > 60:
            rejections.append((br, "quote_stale")); continue
        entry = float(br["entry"]); tp = float(br["take_profit"]); sl = float(br["stop_loss"])
        # stop already breached / target already reached first — the specific
        # reason must win over generic geometry (brief §6 reason codes)
        if (side == "LONG" and px <= sl) or (side == "SHORT" and px >= sl):
            rejections.append((br, "stop_already_breached")); continue
        if (side == "LONG" and px >= tp) or (side == "SHORT" and px <= tp):
            rejections.append((br, "target_already_reached")); continue
        # geometry at the executable quote (brief §5 items 13/14)
        if side == "LONG" and not (sl < px < tp):
            rejections.append((br, "invalid_trade_geometry")); continue
        if side == "SHORT" and not (tp < px < sl):
            rejections.append((br, "invalid_trade_geometry")); continue
        # entry drift: setup invalidated if price ran past the plan (§5 item 16)
        drift = (px - entry) / entry
        if side == "LONG" and drift > 0.01:
            rejections.append((br, "entry_drift_exceeded")); continue
        if side == "SHORT" and -drift > 0.01:
            rejections.append((br, "entry_drift_exceeded")); continue
        # refreshed reward-to-risk at the executable entry (§5 item 15)
        rr = (tp - px) / (px - sl) if side == "LONG" else (px - tp) / (sl - px)
        if rr < 1.0:
            rejections.append((br, "rr_below_floor")); continue
        # spread/slippage proxy: |quote vs PM entry| is the fill risk
        spread = q.get("spread") or 0.0
        if spread and spread > 0.005 * px:
            rejections.append((br, "spread_exceeded")); continue
        # validator quality (from the staged record: validation reasons empty)
        quality = 0 if br.get("validation_reasons") else 1
        ranked.append({
            "br": br, "rr": rr, "drift": abs(drift), "spread": spread,
            "quality": quality, "ticker": ticker,
        })
    # deterministic ordering: quality desc, rr desc, drift asc, spread asc,
    # ticker asc as the final stable tie-breaker (brief §7)
    ranked.sort(key=lambda r: (-r["quality"], -r["rr"], r["drift"], r["spread"], r["ticker"]))
    return [r["br"] for r in ranked], rejections


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
    ap.add_argument("--execute-staged", action="store_true",
                    help="09:20 ET pre-open executor: deterministic fresh-quote "
                         "revalidation + immediate placement (cutoff 09:25 ET)")
    ap.add_argument("--cancel-entries", action="store_true",
                    help="09:29 ET: cancel unfilled entry orders from today's run "
                         "(surgical by order id; positions/TP/SL untouched)")
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
    if args.execute_staged:
        return execute_staged(date)
    if args.cancel_entries:
        lock = _acquire_singleton_lock("cancel")
        if lock is None:
            return 3
        import nyse_calendar as nyc
        now = datetime.now(timezone.utc)
        if not nyc.in_cancel_window(now):
            print("⛔ cancel-entries refused: outside the 09:29-09:35 ET window")
            return 4
        from mexc_orders import cancel_unfilled_entries
        rep = cancel_unfilled_entries(nyc.et_date_str(now), now=now)
        print(json.dumps(rep, default=str))
        if rep["unconfirmed"] or rep["errors"]:
            return 2
        return 0
    if args.sweep:
        lock = _acquire_singleton_lock("sweep")
        if lock is None:
            return 3
        # calendar-aware sweep timing (Codex §execution): a fixed 14:00 UTC
        # cron runs BEFORE the 14:30 UTC open during U.S. standard time —
        # the sweep must be relative to the ACTUAL session open.
        import nyse_calendar
        sweep_due = nyse_calendar.sweep_time_utc()
        if datetime.now(timezone.utc) < sweep_due:
            print(f"⏳ sweep not due yet (open+90m = {sweep_due.isoformat()}); "
                  f"nothing to do now — orders still have their window.")
            return 0
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
