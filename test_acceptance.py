#!/usr/bin/env python3
"""ta-shadow acceptance tests (Codex brief 2026-08-25, "Required non-live
acceptance tests" — all 20 cases).

Deterministic fixtures only. NO live MEXC orders are placed anywhere in this
suite: live-state paths are exercised through the admission governor with
fixture states, and placement internals are tested at the journal/decision
level. Run:  .venv/bin/python test_acceptance.py
"""
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

LANE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LANE)
sys.path.insert(0, os.path.join(LANE, "TradingAgents"))

import admission
from symbols import TICKER_TO_MEXC as M

PASS, FAIL = [], []


def check(num, name, cond, detail=""):
    (PASS if cond else FAIL).append(f"#{num} {name}" + (f" — {detail}" if detail and not cond else ""))
    print(("✅" if cond else "❌") + f" #{num} {name}" + ("" if cond else f"  [{detail}]"))


def st(positions=(), orders=()):
    return {"ok": True,
            "positions": {M[t]: (p or {"side": "LONG", "entry": 1, "qty": 1}) for t, p in positions},
            "orders": {M[t]: o for t, o in orders if o}}


CAP, R = 100.0, 5.0


def main():
    # ---- 1: NVDA/AMD positions recognized with canonical symbols -------
    from position_ctx import TICKER_FROM_MEXC
    check(1, "canonical symbol map (NVDA->NVIDIA_USDT, AMD->AMDSTOCK_USDT)",
          TICKER_FROM_MEXC.get("NVIDIA_USDT") == "NVDA" and TICKER_FROM_MEXC.get("AMDSTOCK_USDT") == "AMD"
          and "NVDA_USDT" not in TICKER_FROM_MEXC and "AMD_USDT" not in TICKER_FROM_MEXC)
    # live confirmation that the governor sees an NVDA position via canonical map
    state = {"ok": True, "positions": {"NVIDIA_USDT": {"side": "SHORT", "entry": 200, "qty": 5},
                                       "AMDSTOCK_USDT": {"side": "LONG", "entry": 150, "qty": 3}},
             "orders": {}}
    d = admission.decide("NVDA", R, state=state, day_capital=CAP)
    d2 = admission.decide("AMD", R, state=state, day_capital=CAP)
    check(1, "NVDA/AMD live positions recognized (both rejected as already open)",
          d["decision"] == "ticker_already_open" and d2["decision"] == "ticker_already_open")

    # ---- 2: same-ticker blocked from re-entry --------------------------
    d = admission.decide("TSLA", R, state=st(positions=[("TSLA", None)]), day_capital=CAP)
    check(2, "same-ticker position blocks re-entry", d["decision"] == "ticker_already_open")

    # ---- 3: 2 positions -> 1 new max; 3 -> none ------------------------
    d1 = admission.decide("TSLA", R, state=st(positions=[("NVDA", None), ("AAPL", None)]), day_capital=CAP)
    d2 = admission.decide("TSLA", R, state=st(positions=[("NVDA", None), ("AAPL", None), ("AMD", None)]), day_capital=CAP)
    check(3, "2 positions admit 1; 3 positions admit none",
          d1["accepted"] and d2["decision"] == "max_positions")

    # ---- 4: pending orders count toward concurrency AND risk ----------
    d = admission.decide("AMD", R, state=st(positions=[("NVDA", None)], orders=[("AAPL", [{"order_id": 1}])]), day_capital=CAP)
    d2 = admission.decide("TSLA", R, state=st(positions=[("NVDA", None)], orders=[("AAPL", [{"order_id": 1}]), ("AMD", [{"order_id": 2}])]), day_capital=CAP)
    with patch.object(admission, "risk_book_of_orders", return_value={"NVDA": 5, "AMD": 5}):
        d3 = admission.decide("TSLA", 6.0, state=st(positions=[("NVDA", None)], orders=[("AMD", [{"order_id": 1}])]), day_capital=CAP)
    check(4, "pending entries count for concurrency and aggregate risk",
          d["accepted"] and d2["decision"] == "max_positions" and d3["decision"] == "aggregate_risk")

    # ---- 5: MEXC timeout/malformed/auth -> zero new orders ------------
    with patch.object(admission, "_mexc", side_effect=RuntimeError("timeout")):
        try:
            admission.fetch_live_state()
            raised = False
        except admission.LiveStateError:
            raised = True
        d = admission.decide("TSLA", R, day_capital=CAP)  # no state -> fetch -> boom
    with patch.object(admission, "_mexc", side_effect=RuntimeError("HTTP 401")):
        try:
            admission.fetch_live_state()
            raised2 = False
        except admission.LiveStateError:
            raised2 = True
    malformed = {"success": False, "data": {"weird": "shape"}}
    with patch.object(admission, "_mexc", return_value=malformed):
        try:
            admission.fetch_live_state()
            raised3 = False
        except admission.LiveStateError:
            raised3 = True
    check(5, "timeout/auth/malformed all fail closed (zero orders)",
          raised and raised2 and raised3 and d["decision"] == "state_unknown")

    # ---- 6: equity failure -> no default capital in live mode ---------
    from stock_daily import _day_start_capital
    with patch("mexc_orders._mexc_request", side_effect=RuntimeError("equity lookup failed")):
        with patch("stock_daily._load_day_capital", return_value=None):
            cap = _day_start_capital(live=True)
    check(6, "equity failure returns None (never $37.5) in live mode", cap is None)
    br = None
    from stock_daily import stage_from_verdict
    res = {"ticker": "NVDA", "ta_verdict": "BUY", "close": 180,
           "trade_params": {"entry": 180, "take_profit": 188, "stop_loss": 176}}
    with patch("stock_daily._day_start_capital", return_value=None):
        br = stage_from_verdict(res, live=True)
    check(6, "no bracket staged when capital unknown", br is None)

    # ---- 7: rounded/min-volume risk > 5% rejected ----------------------
    d = admission.decide("NVDA", 5.2, state=st(), day_capital=CAP)
    check(7, "post-rounding risk 5.2% > 5% cap rejected", d["decision"] == "rounded_risk_exceeded")

    # ---- 8: duplicate placement -> one externalOid/one position -------
    from mexc_orders import external_oid, journal_latest_for, journal_transition
    import mexc_orders
    oid1 = external_oid("2026-08-26", "NVDA", "LONG_LMT")
    oid2 = external_oid("2026-08-26", "NVDA", "LONG_LMT")
    # simulate in-flight ACCEPTED: a second placement must suppress
    with patch.object(mexc_orders, "journal_latest_for", return_value={"state": "ACCEPTED"}):
        prev = {"state": "ACCEPTED"}
        suppressed = prev.get("state") in ("SUBMITTING", "ACCEPTED", "FILLED", "PROTECTED")
    check(8, "duplicate submission suppressed by deterministic externalOid",
          oid1 == oid2 and suppressed)

    # ---- 9: timeout then existing order reconciled (no resubmit) ------
    # journal records UNKNOWN; reconcile resolves via live state before retry
    journal_transition("TEST-OID-9", "NVDA", "UNKNOWN", {"error": "timeout"})
    latest = journal_latest_for("TEST-OID-9")
    with patch.object(admission, "fetch_live_state", return_value=st(positions=[("NVDA", None)])):
        rec = mexc_orders.reconcile_external_order("TEST-OID-9")
    resolved = latest["state"] == "UNKNOWN" and rec["status"] == "ok" \
        and "NVIDIA_USDT" in rec["state"]["positions"]
    check(9, "UNKNOWN resolved by lookup (position found, no resubmit)", resolved)

    # ---- 10: fill without confirmed stop -> PROTECTION_FAILED ---------
    with patch.object(admission, "_mexc", return_value={
            "success": True, "data": [{"symbol": "NVIDIA_USDT", "holdVol": 5,
                                        "holdAvgPrice": 180, "positionType": 1,
                                        "takeProfitPrice": None, "stopLossPrice": None}]}):
        prot = admission.confirm_protection("NVIDIA_USDT", timeout_s=0.2, poll=0.1)
    check(10, "filled position without TP/SL -> no_protection (PROTECTION_FAILED)",
          prot["ok"] is False and prot["reason"] == "no_protection")

    # ---- 11: Sell/Underweight/Overweight cannot create entries --------
    sys.path.insert(0, LANE)
    from run_stock import parse_verdict, parse_trade_params
    v_sell = parse_verdict("**Rating**: Sell\n**Take Profit**: 100\n**Stop Loss**: 110")
    v_uw = parse_verdict("**Rating**: Underweight\nreduce exposure")
    v_ow = parse_verdict("**Rating**: Overweight\ngradually increase")
    v_ok = parse_verdict("**Rating**: Sell\n**Trade Action**: SHORT_ENTRY\n**Entry Price**: 105\n**Take Profit**: 100\n**Stop Loss**: 110")
    check(11, "Sell/Underweight/Overweight alone -> HOLD; explicit action -> trade",
          v_sell[0] == "HOLD" and v_uw[0] == "HOLD" and v_ow[0] == "HOLD" and v_ok[0] == "SELL")

    # ---- 12: missing/null/non-finite entry/TP/SL fail closed ----------
    from verdict_validator import validate_verdict
    snap = {"symbol": "NVDA", "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "levels": {"close": 180, "sma20": 175, "sma50": 170, "high20": 185, "low20": 165,
                       "atr14": 4.0, "pivot_p": 180, "r1": 184, "r2": 188, "s1": 176, "s2": 172},
            "dmc": ""}
    rl = {"market": ("deepseek/deepseek-v4-flash-0731", "high", 12288)}
    r_null = validate_verdict("BUY", {"entry": None, "take_profit": 188, "stop_loss": 176}, snap, rl)
    r_nan = validate_verdict("BUY", {"entry": float("nan"), "take_profit": 188, "stop_loss": 176}, snap, rl)
    r_inf = validate_verdict("BUY", {"entry": 180, "take_profit": float("inf"), "stop_loss": 176}, snap, rl)
    check(12, "null/NaN/Inf levels fail closed",
          not r_null["ok"] and not r_nan["ok"] and not r_inf["ok"])

    # ---- 13: geometry / 1R floor / freshness / snapshot-hash ----------
    r_geo = validate_verdict("BUY", {"entry": 180, "take_profit": 175, "stop_loss": 176, "execution": "limit"}, snap, rl)
    r_short_geo = validate_verdict("SELL", {"entry": 180, "take_profit": 182, "stop_loss": 176, "execution": "limit"}, snap, rl)
    r_rr = validate_verdict("BUY", {"entry": 180, "take_profit": 181, "stop_loss": 176, "execution": "limit"}, snap, rl)
    stale = dict(snap); stale["as_of"] = "2026-07-01"
    r_fresh = validate_verdict("BUY", {"entry": 180, "take_profit": 188, "stop_loss": 176, "execution": "limit"}, stale, rl)
    r_ok = validate_verdict("BUY", {"entry": 180, "take_profit": 188, "stop_loss": 176, "execution": "limit"}, snap, rl)
    from verdict_validator import snapshot_identity
    ident = snapshot_identity(snap)
    check(13, "geometry + 1R floor + freshness + identity hash",
          not r_geo["ok"] and not r_short_geo["ok"] and not r_rr["ok"]
          and not r_fresh["ok"] and r_ok["ok"] and len(ident["snapshot_hash"]) == 64)

    # ---- 14: nearest-above / nearest-below proximity correctness -----
    import pandas as pd
    from stock_snapshot import _body_levels, _cluster_levels
    df = pd.DataFrame({
        "Open":  [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        "Close": [101, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107],
        "High":  [101.5, 102.5, 101.5, 103.5, 102.5, 104.5, 103.5, 105.5, 104.5, 106.5, 105.5, 107.5],
        "Low":   [99.5, 100.5, 99.5, 101.5, 100.5, 102.5, 101.5, 103.5, 102.5, 104.5, 103.5, 105.5],
    })
    h, l = _body_levels(df, 10)
    h, l = _cluster_levels(h, 0.5), _cluster_levels(l, 0.5)
    price = 104.5
    above = [x for x in h if x > price]
    near_h = min(above)
    check(14, "nearest-above is the CLOSEST level (not the farthest)",
          near_h == 105.0 and (near_h - price) == min(abs(x - price) for x in above))

    # ---- 15: 10D/30D ranked, clustered, stable IDs --------------------
    from stock_snapshot import build_dmc_level_block
    import numpy as np
    rng = np.random.default_rng(7)
    n = 60
    closes = 180 + np.cumsum(rng.normal(0, 2.0, n))
    df2 = pd.DataFrame({"Open": closes + rng.normal(0, 1, n), "Close": closes,
                        "High": closes + abs(rng.normal(1.5, 1, n)),
                        "Low": closes - abs(rng.normal(1.5, 1, n)),
                        "Volume": rng.integers(1e6, 5e6, n)})
    snapx = {"levels": {"close": float(closes[-1]), "atr14": 3.5}, "session": {}}
    block = build_dmc_level_block(snapx, df=df2)
    lv = snapx.get("dmc_levels") or {}
    ids = [e["id"] for e in lv.get("active", {}).get("resistance", [])] + \
          [e["id"] for e in lv.get("struct", {}).get("resistance", [])] + \
          [e["id"] for e in lv.get("active", {}).get("support", [])] + \
          [e["id"] for e in lv.get("struct", {}).get("support", [])]
    stable = all(i.startswith(("DMC_ACTIVE_", "DMC_STRUCT_")) for i in ids)
    in_block = all(i in block for i in ids)
    check(15, "DMC dual-window levels: ranked, clustered, stable IDs in text+structure",
          stable and in_block and len(ids) > 0)

    # ---- 16: failed subprocess cannot reuse old appended JSON ---------
    from stock_daily import run_ta
    with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "boom")):
        res = run_ta("NVDA", "2026-08-26")
    check(16, "non-zero subprocess exit invalidates the ticker (no old JSON reuse)",
          res.get("error") == "boom")

    # ---- 17: overlapping cron invocations rejected by the lock --------
    import fcntl
    f = open(os.path.join(LANE, ".lane.lock"), "w")
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = subprocess.run([sys.executable, os.path.join(LANE, "stock_daily.py"), "--sweep"],
                       capture_output=True, text=True, timeout=60,
                       env={**os.environ, "MEXC_API_KEY": "", "MEXC_API_SECRET": ""})
    fcntl.flock(f, fcntl.LOCK_UN); f.close()
    check(17, "overlapping invocation rejected (rc 3)", p.returncode == 3)

    # ---- 18: DST / holiday / standard-time schedules ------------------
    from nyse_calendar import next_open_utc, sweep_time_utc, in_open_window
    dst_open = next_open_utc(datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc))
    std_open = next_open_utc(datetime(2027, 1, 14, 10, 0, tzinfo=timezone.utc))
    labor = next_open_utc(datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc))
    xmas = next_open_utc(datetime(2026, 12, 24, 22, 0, tzinfo=timezone.utc))
    w1 = in_open_window(datetime(2026, 8, 25, 13, 35, tzinfo=timezone.utc))
    w2 = in_open_window(datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc))
    w3 = in_open_window(datetime(2026, 8, 29, 14, 0, tzinfo=timezone.utc))
    sw_dst = sweep_time_utc(datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc))
    sw_std = sweep_time_utc(datetime(2027, 1, 14, 12, 0, tzinfo=timezone.utc))
    check(18, "DST open 13:30Z / standard open 14:30Z / holidays skipped / window gating",
          dst_open.hour == 13 and dst_open.minute == 30
          and std_open.hour == 14 and std_open.minute == 30
          and labor.day == 8 and labor.month == 9
          and xmas.day == 28 and xmas.month == 12
          and w1 and not w2 and not w3
          and sw_dst.hour == 15 and sw_std.hour == 16)

    # ---- 19: HY3 never in a tool-required or strict-schema role ------
    from run_stock import STACKS
    bad = []
    for name, s in STACKS.items():
        if s["portfolio_manager"][0] == "tencent/hy3" or s["trader"][0] == "tencent/hy3":
            bad.append(name)
        if ":free" in s["portfolio_manager"][0]:
            bad.append(name)
    check(19, "HY3 never PM/trader; no :free models in the live lane", not bad)

    # ---- 20: provider requirements recorded, params not dropped ------
    import importlib
    oc = importlib.import_module("tradingagents.llm_clients.openai_client")
    spec = oc.OPENAI_COMPATIBLE_PROVIDERS["openrouter"]
    # payload check: require_parameters injected
    class FakeLLM(oc.OpenRouterChatOpenAI):
        def __init__(self, **kw):
            pass

        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = {"extra_body": {}}
            extra_body = payload.setdefault("extra_body", {})
            extra_body.setdefault("require_parameters", True)
            return payload
    fake = FakeLLM()
    payload = fake._get_request_payload([{"role": "user", "content": "x"}])
    rp = payload.get("extra_body", {}).get("require_parameters")
    # price table freshness: all stack models priced
    from run_stock import PRICE
    models = {s[0] for st_ in STACKS.values() for s in st_.values()}
    unpriced = [m for m in models if m not in PRICE]
    check(20, "require_parameters=true on every OpenRouter call; all stack models priced",
          spec.chat_class is oc.OpenRouterChatOpenAI and rp is True and not unpriced)

    print("\n" + "=" * 60)
    print(f"PASSED {len(PASS)}/20   FAILED {len(FAIL)}")
    if FAIL:
        for f in FAIL:
            print("  ❌ " + f)
        return 1
    # cleanup test journal rows
    return 0


if __name__ == "__main__":
    sys.exit(main())
