#!/usr/bin/env python3
"""One-off diagnostic: drive the stock graph with fail-fast settings to see
the raw error the SDK is retrying on. max_retries=0, timeout=60."""
import json, os, sys, time, traceback
LANE = "/opt/hermes-projects/ta-shadow"
sys.path.insert(0, LANE)
sys.path.insert(0, os.path.join(LANE, "TradingAgents"))
from dotenv import load_dotenv
load_dotenv(os.path.join(LANE, ".env"))
from stock_snapshot import fetch_stock_snapshot
from run_stock import bridge_nous_key, PROVIDER, DEEP_THINK_LLM, QUICK_THINK_LLM
bridge_nous_key()

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
snap = fetch_stock_snapshot(ticker, session=False)
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = PROVIDER
config["deep_think_llm"] = DEEP_THINK_LLM
config["quick_think_llm"] = QUICK_THINK_LLM
config["llm_max_retries"] = 0          # fail fast
config["llm_timeout"] = 60             # 60s cap
config["max_debate_rounds"] = 1
config["temperature"] = 0.0
config["shadow_snapshot"] = {"kind": "stock", "snapshot": snap, "symbol": ticker,
                             "variant": "A", "position_blind": False}

from cli.stats_handler import StatsCallbackHandler
stats = StatsCallbackHandler()
ta = TradingAgentsGraph(selected_analysts=("market", "social", "news", "fundamentals"),
                        debug=False, config=config, callbacks=[stats])
print("graph built; class:", type(ta.quick_thinking_llm).__name__, flush=True)
t0 = time.time()
try:
    state, decision = ta.propagate(ticker, "2026-08-22", asset_type="stock")
    print(f"OK in {time.time()-t0:.1f}s", flush=True)
    print("decision:", str(decision)[:300], flush=True)
except Exception:
    print(f"FAILED after {time.time()-t0:.1f}s", flush=True)
    traceback.print_exc()
    # print last messages to see the actual tool/LLM error
    try:
        for m in state.get("messages", [])[-3:]:
            print("--- msg ---", flush=True)
            print(str(m)[:500], flush=True)
    except Exception:
        pass
