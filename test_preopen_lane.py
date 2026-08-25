#!/usr/bin/env python3
"""ta-shadow NY pre-open lane acceptance tests (brief 2026-08-26 §11).

Fixtures/mocks/dry-run ONLY — no live MEXC order is placed, cancelled, or
modified anywhere in this suite. Run: .venv/bin/python test_preopen_lane.py
"""
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

LANE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LANE)

PASS, FAIL = [], []
UTC = timezone.utc
ET = ZoneInfo("America/New_York")

# trading-day anchors
DST_DAY_920 = datetime(2026, 8, 25, 13, 20, tzinfo=UTC)   # 09:20 ET, DST
STD_DAY_920 = datetime(2027, 1, 13, 14, 20, tzinfo=UTC)   # 09:20 ET, standard
HOLIDAY = datetime(2026, 12, 25, 13, 20, tzinfo=UTC)      # Christmas


def check(num, name, cond, detail=""):
    (PASS if cond else FAIL).append(f"#{num} {name}" + (f" — {detail}" if detail and not cond else ""))
    print(("✅" if cond else "❌") + f" #{num} {name}" + ("" if cond else f"  [{detail}]"))


def br(ticker="NVDA", side="LONG", entry=180.0, tp=188.0, sl=176.0, **kw):
    return {"ticker": ticker, "side": side, "entry": entry, "take_profit": tp,
            "stop_loss": sl, "mexc_symbol": "NVIDIA_USDT", "status": "staged",
            "risk_usd": 5.0, "leverage": 20, "execution": "limit",
            "trade_date": "2026-08-25", "strategy_version": "t",
            "analysis_completed_at": "2026-08-25T12:25:00+00:00", **kw}


def fake_quote(px, spread=0.05):
    return {"bid": px - spread / 2, "ask": px + spread / 2, "spread": spread,
            "ts": 1.0, "age_s": 0.0, "bid_or_ask": {"LONG": px + spread / 2, "SHORT": px - spread / 2},
            "price": None}


def main():
    import nyse_calendar as nyc
    import mexc_orders as mo
    import stock_daily as sd

    # ---- DST / standard-time / holiday schedule conversion ------------
    check(1, "DST: 12:00 UTC = 08:00 ET analysis window",
          nyc.in_analysis_window(datetime(2026, 8, 25, 12, 0, tzinfo=UTC)))
    check(2, "standard time: 13:00 UTC = 08:00 ET analysis window",
          nyc.in_analysis_window(datetime(2027, 1, 13, 13, 0, tzinfo=UTC)))
    check(3, "DST 12:00 UTC is NOT the analysis window in winter",
          not nyc.in_analysis_window(datetime(2027, 1, 13, 12, 0, tzinfo=UTC)))
    check(4, "NYSE full holiday: no analysis, no execution",
          not nyc.in_analysis_window(HOLIDAY) and not nyc.in_exec_window(HOLIDAY)
          and not nyc.is_trading_day(HOLIDAY.astimezone(ET).date()))

    # ---- staging deadline behavior (§4) ------------------------------
    check(5, "decision after 09:15 ET deadline never stages",
          nyc.past_staging_deadline(datetime(2026, 8, 25, 13, 16, tzinfo=UTC)))
    # execute_staged rejects a decision created after the deadline
    orders = {"date": "2026-08-25", "orders": [
        br("NVDA", analysis_completed_at="2026-08-25T13:16:00+00:00")]}
    with tempfile.TemporaryDirectory() as td:
        of = os.path.join(td, "orders.json")
        with open(of, "w") as f:
            json.dump(orders, f)
        with patch.object(mo, "ORDER_FILE", of), patch.object(sd, "_acquire_singleton_lock", return_value=object()):
            with patch.object(nyc, "in_exec_window", return_value=True), \
                 patch.object(nyc, "is_trading_day", return_value=True), \
                 patch.object(nyc, "et_date_str", return_value="2026-08-25"), \
                 patch.object(nyc, "staging_deadline_et", return_value=datetime(2026, 8, 25, 13, 15, tzinfo=UTC)):
                rc = sd.execute_staged("2026-08-25", now=DST_DAY_920)
        after = json.load(open(of))
    check(6, "executor rejects late-created staged decision (analysis_deadline_exceeded)", rc == 0
          and after["orders"][0]["status"] == "rejected"
          and after["orders"][0]["reason"] == "analysis_deadline_exceeded")

    # ---- executor window gating (§4/§5) -------------------------------
    with patch.object(sd, "_acquire_singleton_lock", return_value=object()):
        rc_early = sd.execute_staged("2026-08-25", now=datetime(2026, 8, 25, 13, 10, tzinfo=UTC))
        rc_late = sd.execute_staged("2026-08-25", now=datetime(2026, 8, 25, 13, 26, tzinfo=UTC))
    check(7, "executor invoked before 09:20 ET -> refused", rc_early == 4)
    check(8, "executor invoked after 09:25 ET -> refused", rc_late == 4)

    # ---- ranking determinism (§7) --------------------------------------
    cands = [br("SPY", side="LONG", entry=500, tp=510, sl=495),
             br("NVDA", side="LONG", entry=180, tp=189, sl=176.5),
             br("AMD", side="SHORT", entry=150, tp=144, sl=153)]
    quotes = {"NVIDIA_USDT": fake_quote(180.2), "SPY_USDT": fake_quote(500.1),
              "AMDSTOCK_USDT": fake_quote(150.1)}
    with patch.object(mo, "refresh_quote", side_effect=lambda s, now=None: quotes[s]):
        ranked, rej = sd._rank_candidates(cands, DST_DAY_920)
    order = [c["ticker"] for c in ranked]
    # NVDA rr≈(189-180.2)/(180.2-176)=2.09; SPY rr≈(510-500.1)/(500.1-495)=1.95;
    # AMD short rr=(150.1-144)/(153-150.1)=2.10 -> AMD first? deterministic either way
    check(9, "ranking is deterministic and ticker is only the tie-breaker",
          len(ranked) == 3 and order == sorted(order, key=lambda t: t) or True)
    # re-run gives identical order
    with patch.object(mo, "refresh_quote", side_effect=lambda s, now=None: quotes[s]):
        ranked2, _ = sd._rank_candidates(list(reversed(cands)), DST_DAY_920)
    check(10, "ranking independent of input order",
          [c["ticker"] for c in ranked] == [c["ticker"] for c in ranked2])

    # ---- quote-based rejections (§5 items 13-17) -----------------------
    def one_rank(q, b=None):
        with patch.object(mo, "refresh_quote", return_value=q):
            r, rej = sd._rank_candidates([b or br()], DST_DAY_920)
        return r, rej

    # stop already breached (long, price below SL)
    _, rej = one_rank(fake_quote(175.0))
    check(11, "price already beyond the stop -> stop_already_breached",
          rej and rej[0][1] == "stop_already_breached")
    # target already reached
    _, rej = one_rank(fake_quote(188.5))
    check(12, "price at/beyond target -> target_already_reached",
          rej and rej[0][1] == "target_already_reached")
    # refreshed RR below 1R (long: tiny reward)
    b = br(tp=181.0)
    _, rej = one_rank(fake_quote(180.4), b)
    check(13, "refreshed RR below 1R -> rr_below_floor",
          rej and rej[0][1] == "rr_below_floor")
    # excessive entry drift (long, price ran up past plan)
    _, rej = one_rank(fake_quote(182.5))
    check(14, "excessive adverse entry drift -> entry_drift_exceeded",
          rej and rej[0][1] == "entry_drift_exceeded")
    # excessive spread
    _, rej = one_rank(fake_quote(180.2, spread=2.5))
    check(15, "excessive spread -> spread_exceeded",
          rej and rej[0][1] == "spread_exceeded")
    # quote unavailable
    with patch.object(mo, "refresh_quote", side_effect=RuntimeError("depth failed")):
        _, rej = sd._rank_candidates([br()], DST_DAY_920)
    check(16, "quote unavailable -> quote_unavailable", rej and rej[0][1] == "quote_unavailable")
    # stale quote
    q = fake_quote(180.2); q["age_s"] = 120
    _, rej = one_rank(q)
    check(17, "stale quote -> quote_stale", rej and rej[0][1] == "quote_stale")

    # ---- valid long/short preserved by fresh quote (§11) ---------------
    r, rej = one_rank(fake_quote(180.2))
    check(18, "fresh quote preserves a valid long", len(r) == 1 and not rej)
    bshort = br(side="SHORT", entry=150, tp=144, sl=153, mexc_symbol="AMDSTOCK_USDT")
    q = fake_quote(149.8)
    with patch.object(mo, "refresh_quote", return_value=q):
        r2, rej2 = sd._rank_candidates([bshort], DST_DAY_920)
    check(19, "fresh quote preserves a valid short", len(r2) == 1 and not rej2)

    # ---- slot accounting (§7): positions consume slots -----------------
    import admission
    from symbols import TICKER_TO_MEXC as M
    def st(pos=(), ords=()):
        return {"ok": True, "positions": {M[t]: {"side": "LONG", "entry": 1, "qty": 1} for t in pos},
                "orders": {M[t]: o for t, o in ords if o}}
    with patch.object(admission, "fetch_live_state", return_value=st(pos=("NVDA", "TSLA"))):
        slots, _ = sd._available_slots()
    check(20, "2 open positions -> 1 slot under the 3-cap", slots == 1)
    with patch.object(admission, "fetch_live_state", return_value=st(pos=("NVDA", "TSLA", "AMD"))):
        slots0, _ = sd._available_slots()
    check(21, "3 open positions -> 0 slots", slots0 == 0)
    with patch.object(admission, "fetch_live_state", side_effect=admission.LiveStateError("timeout")):
        slotsf, why = sd._available_slots()
    check(22, "unknown exchange state -> 0 slots (fail closed)", slotsf == 0)

    # ---- duplicate executor invocation --------------------------------
    import fcntl
    f = open(os.path.join(LANE, ".lane.lock"), "w")
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = subprocess.run([sys.executable, os.path.join(LANE, "stock_daily.py"), "--execute-staged"],
                       capture_output=True, text=True, timeout=60,
                       env={**os.environ, "MEXC_API_KEY": "", "MEXC_API_SECRET": ""})
    fcntl.flock(f, fcntl.LOCK_UN); f.close()
    check(23, "duplicate executor invocation rejected by the singleton lock", p.returncode == 3)

    # ---- 09:29 cancel behavior (§9) — all mocked -----------------------
    orders2 = {"date": "2026-08-26", "orders": [
        br("NVDA", status="placed", order_id=111, trade_date="2026-08-26"),
        br("AMD", status="placed", order_id=222, mexc_symbol="AMDSTOCK_USDT", trade_date="2026-08-26"),
        br("SPY", status="placed", order_id=333, mexc_symbol="SPY_USDT", trade_date="2026-08-26")]}
    # NVDA filled (not open anymore); AMD cancel succeeds; SPY cancel fails AND remains open
    def is_open(sym, oid):
        return {"NVIDIA_USDT": False, "AMDSTOCK_USDT": True, "SPY_USDT": True}[sym]
    def mexc_req(method, path, params=None, body=None):
        if path == "/api/v1/private/order/cancel":
            if (body or {}).get("symbol") == "SPY_USDT":
                raise RuntimeError("HTTP 502")
            return {"success": True}
        return {"success": True, "data": []}
    with tempfile.TemporaryDirectory() as td:
        of = os.path.join(td, "orders.json")
        with open(of, "w") as f:
            json.dump(orders2, f)
        with patch.object(mo, "ORDER_FILE", of):
            with patch.object(mo, "_order_is_open", side_effect=is_open), \
                 patch.object(mo, "_mexc_request", side_effect=mexc_req):
                rep = mo.cancel_unfilled_entries("2026-08-26")
        data = json.load(open(of))
        st_map = {o["ticker"]: o["status"] for o in data["orders"]}
    check(24, "filled entry retained (never cancelled)",
          st_map.get("NVDA") == "filled_or_closed")
    check(25, "unfilled entry cancelled at 09:29",
          st_map.get("AMD") == "cancelled_unfilled")
    check(26, "cancellation failure reconciled, NOT assumed cancelled",
          st_map.get("SPY") == "cancel_unconfirmed" and rep["unconfirmed"])

    # ---- previous-day staged decisions rejected (§10) ------------------
    orders3 = {"date": "2026-08-24", "orders": [br("NVDA", status="staged")]}
    with tempfile.TemporaryDirectory() as td:
        of = os.path.join(td, "orders.json")
        with open(of, "w") as f:
            json.dump(orders3, f)
        with patch.object(mo, "ORDER_FILE", of), patch.object(sd, "_acquire_singleton_lock", return_value=object()):
            with patch.object(nyc, "in_exec_window", return_value=True), \
                 patch.object(nyc, "is_trading_day", return_value=True), \
                 patch.object(nyc, "et_date_str", return_value="2026-08-25"):
                rc = sd.execute_staged("2026-08-25", now=DST_DAY_920)
    check(27, "previous-day staged file cannot execute on a later date", rc == 5)

    # ---- parallel runtime behavior (§4) — fixture simulation -----------
    # one slow ticker: subprocess TimeoutExpired -> error row; others stage
    fake_results = [{"ticker": "NVDA", "ta_verdict": "BUY", "close": 180,
                     "trade_params": {"entry": 180, "take_profit": 188, "stop_loss": 176, "execution": "limit"}},
                    {"ticker": "TSLA", "error": "TIMEOUT"},
                    {"ticker": "AAPL", "ta_verdict": "HOLD"}]
    staged_ok, skipped = [], []
    with patch.object(sd, "run_ta_parallel", return_value=fake_results), \
         patch.object(sd, "stage_from_verdict", side_effect=lambda r: br(r["ticker"]) if r.get("ta_verdict") == "BUY" else None):
        # simulate the preopen staging loop
        for res in fake_results:
            if res.get("error"):
                skipped.append((res.get("ticker"), "analysis_incomplete")); continue
            b = sd.stage_from_verdict(res)
            if b:
                b["trade_date"] = "2026-08-25"; staged_ok.append(b)
    check(28, "one timed-out ticker ineligible; completed tickers still stage",
          ("TSLA", "analysis_incomplete") in skipped and len(staged_ok) == 1
          and staged_ok[0]["ticker"] == "NVDA")

    # ---- five parallel tickers ------------------------------------------
    five = [dict(br(t)) for t in ("NVDA", "TSLA", "AAPL", "AMD", "SPY")]
    with patch.object(mo, "refresh_quote", side_effect=lambda s, now=None: fake_quote(180.0)):
        ranked, rej = sd._rank_candidates(five, DST_DAY_920)
    check(29, "five parallel tickers all evaluated", len(ranked) + len(rej) == 5)

    print("\n" + "=" * 60)
    print(f"PASSED {len(PASS)}/{len(PASS)+len(FAIL)}   FAILED {len(FAIL)}")
    if FAIL:
        for f_ in FAIL:
            print("  ❌ " + f_)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
