"""ta-shadow: THE canonical ticker <-> MEXC symbol map (single source of truth).

2026-08-25 hardening: position_ctx.py previously used NVDA_USDT / AMD_USDT
(wrong — MEXC stock perps are NVIDIA_USDT / AMDSTOCK_USDT), which made
overnight NVDA/AMD positions invisible to the agents and to any portfolio
governor. Every component must import this map; duplicate maps are removed.

Codex review 2026-08-25 §"Fix the current symbol mismatch".
"""

TICKER_TO_MEXC = {
    "NVDA": "NVIDIA_USDT",
    "TSLA": "TESLA_USDT",
    "AAPL": "AAPLSTOCK_USDT",
    "AMD": "AMDSTOCK_USDT",
    "SPY": "SPY_USDT",
}
TICKER_FROM_MEXC = {v: k for k, v in TICKER_TO_MEXC.items()}

UNIVERSE = list(TICKER_TO_MEXC)
