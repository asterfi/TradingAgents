"""Instrumented streaming probe: logs chunk timing to reveal the true shape
of a slow portal call. Hard wall-clock cap so it can never hang.
"""
import json, os, sys, time, threading

LANE = "/opt/hermes-projects/ta-shadow"
sys.path.insert(0, LANE)
sys.path.insert(0, os.path.join(LANE, "TradingAgents"))

from dotenv import load_dotenv
load_dotenv(os.path.join(LANE, ".env"))
from run_stock import bridge_nous_key
bridge_nous_key()

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from stock_snapshot import fetch_stock_snapshot
from tradingagents.agents.analysts.market_analyst import SHADOW_STOCK_MARKET_PROMPT
from tradingagents.dataflows.shadow_snapshot import build_stock_grounding_block
from langchain_core.prompts import ChatPromptTemplate

CAP = float(os.environ.get("PROBE_CAP_S", "300"))

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "nous"
config["deep_think_llm"] = "deepseek/deepseek-v4-flash"
config["quick_think_llm"] = "deepseek/deepseek-v4-flash"
config["llm_max_retries"] = 0
config["llm_timeout"] = 120  # idle timeout
config["max_tokens"] = 3000
config["temperature"] = 0.0
ta = TradingAgentsGraph(selected_analysts=("market",), debug=False, config=config)
llm = ta.quick_thinking_llm

snap = fetch_stock_snapshot("NVDA", session=True)
block = build_stock_grounding_block(snap, variant="A")
system = SHADOW_STOCK_MARKET_PROMPT + "\n\n" + block
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Today's date is 2026-08-22. {system_message}"),
    ("user", "{q}"),
])
chain = prompt.partial(system_message=system) | llm

# ---- instrument the underlying stream at the class level ----
import langchain_openai
_orig_stream = langchain_openai.ChatOpenAI._stream

def timed_stream(self, messages, stop=None, run_manager=None, **kwargs):
    t0 = time.time()
    first = None
    n = 0
    chars = 0
    content_parts = []
    last = t0
    finish = None
    for ch in _orig_stream(self, messages, stop=stop, run_manager=run_manager, **kwargs):
        now = time.time()
        if first is None:
            first = now - t0
            print(f"  [stream] first chunk at {first:.1f}s", flush=True)
        gap = now - last
        if gap > 5:
            print(f"  [stream] gap of {gap:.0f}s before chunk {n}", flush=True)
        last = now
        n += 1
        m = getattr(ch, "message", ch)
        c = getattr(m, "content", None)
        if isinstance(c, str):
            chars += len(c)
            if len(content_parts) < 400:
                content_parts.append(c)
        rm = getattr(m, "response_metadata", None) or {}
        fr = rm.get("finish_reason")
        if fr:
            finish = fr
        if now - t0 > CAP:
            print(f"  [stream] WALL-CLOCK CAP {CAP}s hit at chunk {n} ({chars} chars so far)", flush=True)
            raise TimeoutError(f"probe cap {CAP}s reached")
    total = time.time() - t0
    print(f"  [stream] done: {n} chunks, {chars} chars, {total:.1f}s total, finish={finish}", flush=True)
    snippet = "".join(content_parts)
    print(f"  [stream] first 300: {snippet[:300]!r}", flush=True)

langchain_openai.ChatOpenAI._stream = timed_stream  # type: ignore

t0 = time.time()
try:
    r = chain.invoke({"q": "Analyze NVDA and give a full market report."})
    print(f"PROBE OK total {time.time()-t0:.1f}s report={len(r.content)} chars")
except Exception as e:
    print(f"PROBE FAIL {time.time()-t0:.1f}s {type(e).__name__}: {str(e)[:150]}")
