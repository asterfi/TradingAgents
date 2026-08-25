#!/usr/bin/env python3
"""ta-shadow STOCK lane runner: TradingAgents due diligence on one US stock.

Operator directive (2026-08-23): NVDA/TSLA/AAPL/AMD/SPY only, NY-open focus.
Runs the FULL upstream TradingAgents stock graph (market/news/sentiment/
fundamentals analysts -> bull/bear debate -> risk -> portfolio manager),
grounded in the verified stock snapshot so no agent invents figures.

Output: verdict (BUY/HOLD/SELL) + confidence + optional trade parameters
(entry, take_profit, stop_loss) parsed from the final decision text, written
to stock-log.jsonl with full provenance (provider, model, key_source, cost).

No orders placed here. Order staging lives in mexc_orders.py.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

LANE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(LANE, ".env"))
sys.path.insert(0, os.path.join(LANE, "TradingAgents"))

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from stock_snapshot import fetch_stock_snapshot, build_stock_grounding_block

# ============================================================
# LLM wiring — role-aware three-model stack (spec 2026-08-24).
# Provider: OpenRouter. Models are PINNED revisions (no ~latest
# alias) so controlled shadow testing never mixes model builds.
# ============================================================
PROVIDER = "openrouter"

# --- Pinned model revisions (spec §3.1, §4) ---
DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash-0731"   # volume: analysts, debate, risk
GEMINI_FLASH      = "google/gemini-3.7-flash"           # opposition: bear + conservative
LUNA_PRO          = "openai/gpt-5.6-luna-pro"           # adjudicator: manager/trader/PM

# --- Role -> (model, reasoning_effort, max_completion_tokens) ---
# Spec §4 table. Per-role output caps, no global 384K allowance, no global
# max reasoning, no forced temperature/top_p (provider defaults; §3.3/§3.4).
ROLE_LLMS = {
    "market":            (DEEPSEEK_V4_FLASH, "high",  12288),
    "sentiment":         (DEEPSEEK_V4_FLASH, "low",    8192),
    "news":              (DEEPSEEK_V4_FLASH, "high",  12288),
    "fundamentals":      (DEEPSEEK_V4_FLASH, "high",  12288),
    "bull_researcher":   (DEEPSEEK_V4_FLASH, "high",  12288),
    "bear_researcher":   (GEMINI_FLASH,      "high",  16384),
    "research_manager":  (LUNA_PRO,          "xhigh", 32768),
    "trader":            (LUNA_PRO,          "high",  16384),
    "aggressive":        (DEEPSEEK_V4_FLASH, "high",   8192),
    "neutral":           (DEEPSEEK_V4_FLASH, "high",   8192),
    "conservative":      (GEMINI_FLASH,      "high",  16384),
    "portfolio_manager": (LUNA_PRO,          "xhigh", 32768),
    "reflection":        (DEEPSEEK_V4_FLASH, "low",    4096),
}
# quick/deep slots kept for upstream back-compat; both resolve to the
# volume model (the graph now reads ROLE_LLMS for per-node routing).
DEEP_THINK_LLM = DEEPSEEK_V4_FLASH
QUICK_THINK_LLM = DEEPSEEK_V4_FLASH

# OpenRouter list rates (per 1M tokens), verified live 2026-08-24 from the
# /models endpoint (spec §15). Used for honest per-run cost attribution.
PRICE = {
    "deepseek/deepseek-v4-flash-0731": {"in": 0.14, "out": 0.28},
    "google/gemini-3.7-flash":         {"in": 0.375, "out": 1.875},
    "openai/gpt-5.6-luna-pro":         {"in": 0.20, "out": 1.20},
}
PEAK_WINDOWS = [(1, 4), (6, 10)]


def bridge_nous_key():
    """Expose Hermes' rotating Nous agent_key as NOUS_API_KEY (never stored)."""
    if os.environ.get("NOUS_API_KEY"):
        return "env"
    auth_path = os.path.expanduser("~/.hermes/auth.json")
    try:
        with open(auth_path) as f:
            nous = json.load(f).get("providers", {}).get("nous", {})
        key = nous.get("agent_key") or nous.get("access_token")
        if not key:
            return None
        os.environ["NOUS_API_KEY"] = key
        return "auth.json"
    except (OSError, json.JSONDecodeError):
        return None


def bridge_openrouter_key():
    """Expose Hermes' OpenRouter API key as OPENROUTER_API_KEY (never stored).

    The key lives in ~/.hermes/.env; read it at runtime so the lane never
    holds a copy in its own .env or on disk. No rotation (unlike the Nous
    agent_key), so a single read at process start is enough.
    """
    if os.environ.get("OPENROUTER_API_KEY"):
        return "env"
    hermes_env = os.path.expanduser("~/.hermes/.env")
    try:
        with open(hermes_env) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        os.environ["OPENROUTER_API_KEY"] = key
                        return "~/.hermes/.env"
        return None
    except OSError:
        return None


def is_peak(ts=None):
    ts = ts or datetime.now(timezone.utc)
    h = ts.hour
    return any(a <= h < b for a, b in PEAK_WINDOWS)


def estimate_cost(tokens_in, tokens_out, pro_share=0.15):
    """Legacy single-model estimate (back-compat). Prefer estimate_cost_calls."""
    peak = is_peak()
    mult = 2.0 if peak else 1.0
    fi, fo = PRICE["deepseek/deepseek-v4-flash-0731"]["in"] * mult, PRICE["deepseek/deepseek-v4-flash-0731"]["out"] * mult
    pi, po = PRICE["openai/gpt-5.6-luna-pro"]["in"] * mult, PRICE["openai/gpt-5.6-luna-pro"]["out"] * mult
    lo = tokens_in * fi / 1e6 + tokens_out * fo / 1e6
    hi = tokens_in * pi / 1e6 + tokens_out * po / 1e6
    mid = (tokens_in * (fi * (1 - pro_share) + pi * pro_share) / 1e6
           + tokens_out * (fo * (1 - pro_share) + po * pro_share) / 1e6)
    return {"low": round(lo, 4), "mid": round(mid, 4), "high": round(hi, 4),
            "peak": peak, "note": "estimate at OpenRouter list rates"}


def estimate_cost_calls(calls):
    """Per-model cost from per-call telemetry rows (spec §15).

    Each row: {model, latency_ms, input_tokens, output_tokens}. Rows with an
    unknown/unpriced model are counted at the volume (DeepSeek) rate and
    flagged so the total stays an honest lower bound.
    """
    by_model: dict[str, dict] = {}
    total = 0.0
    unpriced = 0
    for c in calls:
        model = c.get("model")
        rate = PRICE.get(model)
        if not rate:
            model = "<unpriced>"
            rate = PRICE["deepseek/deepseek-v4-flash-0731"]
            unpriced += 1
        ti, to = c.get("input_tokens", 0), c.get("output_tokens", 0)
        cost = ti * rate["in"] / 1e6 + to * rate["out"] / 1e6
        total += cost
        m = by_model.setdefault(model, {"calls": 0, "in": 0, "out": 0, "cost": 0.0})
        m["calls"] += 1
        m["in"] += ti
        m["out"] += to
        m["cost"] += cost
    return {"total_usd": round(total, 5), "by_model": by_model,
            "unpriced_calls": unpriced}


def parse_verdict(text):
    """BUY/HOLD/SELL (+ confidence) from decision text."""
    if not text:
        return "UNKNOWN", None
    t = text.upper()
    order = ["BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"]
    found = [(t.find(k), k) for k in order if k in t]
    if not found:
        return "UNKNOWN", None
    _, label = min(found)
    mapping = {"BUY": "BUY", "OVERWEIGHT": "BUY", "HOLD": "HOLD",
               "UNDERWEIGHT": "SELL", "SELL": "SELL"}
    m = re.search(r"confiden[ce]*[:\\s]*([0-9]+(?:\\.[0-9]+)?)", text, re.IGNORECASE)
    conf = float(m.group(1)) if m else None
    return mapping[label], conf


def thesis_from_decision(text, limit=140):
    """One-sentence "why" for the digest.

    Operator directive 2026-08-24: each asset gets ONE plain sentence the
    operator can read and understand. We prefer the Executive Summary /
    Investment Thesis body; if the PM only emits label lines, fall back to
    the first substantive sentence anywhere.
    """
    if not text:
        return None
    body = text.strip()
    same_line_label = False
    # 1) Prefer the text right after the Thesis / Executive Summary header.
    for marker in ("**Investment Thesis**:", "**Investment Thesis**: ",
                   "Investment Thesis:", "**Executive Summary**:", "**Executive Summary**: ",
                   "Executive Summary:", "**Thesis**:", "**Thesis**: ", "Thesis:"):
        i = body.find(marker)
        if i != -1:
            body = body[i + len(marker):].strip()
            break
    else:
        # 2) No such header: drop any leading **Rating**: / Action / Verdict
        #    label line so we land on the reasoning body.
        for marker in ("**Rating**", "Rating:", "**Action**", "Action:",
                       "**Decision**", "Decision:", "**Verdict**", "Verdict:"):
            if body.startswith(marker):
                nl = body.find("\n")
                if nl != -1 and body[nl + 1:].strip():
                    body = body[nl + 1:].strip()
                else:
                    # rating and reason on the same line: strip the label
                    body = body[len(marker):].lstrip(": ").strip()
                    same_line_label = True
                break
    # First sentence only: cut at a real boundary (". " + capital, or end).
    body = re.sub(r"\s+", " ", body).strip()
    body = body.replace("**", "").strip()
    body = body.lstrip(": ").strip()
    if not body:
        return None
    # If the reason shares a line with its label ("Sell. AMD rolling over…"),
    # the label's own period is NOT a sentence end — skip the first boundary.
    if same_line_label:
        body = body.split(". ", 1)[-1].strip() if ". " in body else body
    m = re.search(r"[.!?](?=\s+[A-Z0-9]|\s*$)", body)
    if m:
        body = body[:m.end()].strip()
    if not body:
        return None
    if len(body) <= limit:
        return body
    return body[:limit].rstrip() + "…"


def parse_trade_params(text, close):
    """Extract entry/TP/SL from decision text (best-effort, unit = price)."""
    params = {"entry": None, "take_profit": None, "stop_loss": None,
              "risk_reward": None, "execution": "limit", "invalidation": None}
    if not text:
        return params
    pats = {
        "entry": r"(?:entry|enter(?: at)?|limit(?: order)?)[*:\s]*\$?\s*([0-9]+(?:\.[0-9]+)?)",
        "take_profit": r"(?:take[- ]?profit|tp|target|target price)[*:\s]*\$?\s*([0-9]+(?:\.[0-9]+)?)",
        "stop_loss": r"(?:stop[- ]?loss|sl|stop)[*:\s]*\$?\s*([0-9]+(?:\.[0-9]+)?)",
    }
    for k, pat in pats.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                params[k] = float(m.group(1))
            except ValueError:
                pass
    # price invalidation: level at which the setup is dead (cancel resting order)
    im = re.search(
        r"(?:invalidation(?: price)?|invalidat(?:e|ion)?(?: above| below)?)"
        r"[*:\s]*\$?\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if im:
        try:
            v = float(im.group(1))
            if v > 0:
                params["invalidation"] = v
        except ValueError:
            pass
    # execution type from the PM decision (spec §9): market | limit | none
    em = re.search(r"(?:execution|order type|entry type)[*:\s]*\$?\s*(market|limit|none)",
                   text, re.IGNORECASE)
    if em:
        params["execution"] = em.group(1).lower()
    # fallback to snapshot close for entry when not stated
    if params["entry"] is None:
        params["entry"] = close
    if params["take_profit"] and params["stop_loss"]:
        r = (params["take_profit"] - params["entry"]) / (params["entry"] - params["stop_loss"])
        params["risk_reward"] = round(r, 2)
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, choices=["NVDA", "TSLA", "AAPL", "AMD", "SPY"])
    ap.add_argument("--date", default=None, help="trade date (default: today UTC)")
    ap.add_argument("--variant", default="A", choices=["A", "B"])
    ap.add_argument("--session", action="store_true", help="include live session fields if market open")
    ap.add_argument("--test", action="store_true", help="mark row test=true")
    ap.add_argument("--results-dir", default=os.path.join(LANE, "results"))
    args = ap.parse_args()

    trade_date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key_source = bridge_openrouter_key()
    if not key_source:
        print(json.dumps({"error": "no OpenRouter credential"}))
        return 1
    os.makedirs(args.results_dir, exist_ok=True)

    snap = fetch_stock_snapshot(args.ticker, session=args.session)
    close = snap["levels"]["close"]

    # Startup guard (spec §6.6): reject when a required role has no valid
    # model configuration — a partial graph with a different model invalidates
    # reproducibility. Fail loudly before any LLM call.
    REQUIRED_ROLES = {"market", "sentiment", "news", "fundamentals",
                      "bull_researcher", "bear_researcher", "research_manager",
                      "trader", "aggressive", "neutral", "conservative",
                      "portfolio_manager", "reflection"}
    missing_roles = REQUIRED_ROLES - set(ROLE_LLMS)
    bad_roles = {r for r, s in ROLE_LLMS.items()
                 if not s or len(s) != 3 or not s[0] or s[1] not in
                 ("low", "high", "xhigh") or int(s[2]) <= 0}
    if missing_roles or bad_roles:
        print(json.dumps({"error": "invalid role_llms config",
                          "missing_roles": sorted(missing_roles),
                          "bad_roles": sorted(bad_roles)}))
        return 1

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = PROVIDER
    config["deep_think_llm"] = DEEP_THINK_LLM
    config["quick_think_llm"] = QUICK_THINK_LLM
    # Rate limits: OpenRouter allows 20 RPM / 1000 RPD (paid). Our 5-stock
    # parallel run is ~60 calls over ~24 min (~2.5 RPM, ~5 concurrent burst) —
    # well under limits. Let the OpenAI SDK's native retry handle the rare 429:
    # it respects the Retry-After header and backs off exponentially (default
    # 2 retries). Disabling it (max_retries=0) made a single transient 429 fail
    # the call outright — the old custom capacity guard was built for the Nous
    # portal's instability and is not wired to the OpenRouter path anyway.
    config["llm_max_retries"] = 2
    config["llm_timeout"] = 120
    # Role-aware LLM routing (spec 2026-08-24 §4/§5): per-node model, reasoning
    # effort and completion cap. No global max_tokens / max reasoning /
    # temperature / top_p — provider defaults for reasoning models (§3.3/3.4).
    config["role_llms"] = ROLE_LLMS
    # Graph controls (spec §5): one debate round, one risk round, recursion 40.
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["max_recur_limit"] = 40
    config["checkpoint_enabled"] = True
    config["shadow_snapshot"] = {
        "kind": "stock", "snapshot": snap, "symbol": args.ticker,
        "variant": args.variant, "position_blind": False,
    }
    config["results_dir"] = os.path.join(args.results_dir, trade_date)

    from cli.stats_handler import StatsCallbackHandler
    stats = StatsCallbackHandler()
    ta = TradingAgentsGraph(selected_analysts=("market", "social", "news", "fundamentals"),
                            debug=False, config=config, callbacks=[stats])

    t0 = time.time()
    state, decision = ta.propagate(args.ticker, trade_date, asset_type="stock")
    elapsed = time.time() - t0

    # Thesis from the Portfolio Manager's full reasoning (the propagated
    # `decision` is only the label — e.g. "Underweight"); fall back to it.
    pm_text = state.get("final_trade_decision") or decision or ""
    thesis = thesis_from_decision(str(pm_text))

    # Parse from the FULL PM decision text, not the label-only `decision`
    # (propagate() returns process_signal(final_trade_decision) = just the
    # rating). Parsing the label means entry/TP/SL/execution/expiry are never
    # read from the agents' structured output. Fixed 2026-08-24.
    verdict, confidence = parse_verdict(str(pm_text) or decision)
    params = parse_trade_params(str(pm_text) or decision, close)
    # Deterministic post-verdict validation (spec §10): fail-closed to HOLD.
    from verdict_validator import validate_verdict
    vres = validate_verdict(verdict, params, snap, role_llms=ROLE_LLMS)
    if vres["ok"]:
        # Reattach the parsed confidence on pass; the validator nulls it on fail.
        vres["confidence"] = confidence
    verdict, confidence = vres["verdict"], vres.get("confidence")
    params = vres["params"]
    validation = {k: vres[k] for k in
                  ("ok", "reasons", "snapshot_id", "snapshot_hash",
                   "snapshot_version", "snapshot_timestamp", "dmc_version",
                   "instrument", "timeframe_set", "data_freshness_status")}
    ti = stats.get_stats().get("tokens_in", 0)
    to = stats.get_stats().get("tokens_out", 0)
    calls = stats.get_stats().get("llm_calls", 0)
    cost = estimate_cost(ti, to)
    # Per-node telemetry (spec §14/§15): per-call rows + per-model cost.
    per_call = stats.get_calls()
    cost_calls = estimate_cost_calls(per_call)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "stock_ta",
        "ticker": args.ticker,
        "trade_date": trade_date,
        "variant": args.variant,
        "llm_provider": PROVIDER, "llm_model": DEEP_THINK_LLM, "key_source": key_source,
        "role_llms": {r: {"model": s[0], "reasoning_effort": s[1], "max_tokens": s[2]}
                      for r, s in ROLE_LLMS.items()},
        "as_of": snap["as_of"], "close": close,
        "ta_verdict": verdict, "confidence": confidence,
        "trade_params": params,
        "validation": validation,
        "cost_usd": cost,
        "cost_by_model": cost_calls,
        "tokens": {"in": ti, "out": to, "calls": calls},
        "per_call": per_call,
        "wall_clock_s": round(elapsed, 1),
        "test": bool(args.test),
    }
    log_path = os.path.join(LANE, "stock-log.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(row) + "\n")

    # artifacts
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    art = os.path.join(args.results_dir, trade_date, f"{args.ticker}_{args.variant}_{stamp}")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "summary.json"), "w") as f:
        json.dump({**row, "grounding": build_stock_grounding_block(snap, args.variant)}, f, indent=2, default=str)
    with open(os.path.join(art, "decision.txt"), "w") as f:
        f.write(str(decision))

    print(json.dumps({
        "ticker": args.ticker, "variant": args.variant, "provider": PROVIDER,
        "model": DEEP_THINK_LLM, "verdict": verdict, "ta_verdict": verdict,
        "confidence": confidence, "close": close, "as_of": snap["as_of"],
        "trade_params": params, "thesis": thesis,
        "validation": validation,
        "elapsed_s": round(elapsed, 1),
        "tokens_in": ti, "tokens_out": to, "llm_calls": calls,
        "cost_usd": cost, "log_row": log_path, "artifacts": art,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
