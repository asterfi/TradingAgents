# Codex Brief — ta-shadow (TradingAgents stock lane) architecture review

You are acting as a **read-only second opinion** on a live automated trading
system. You will NOT edit, run, or modify anything. Your only deliverable is
a written analysis + a prioritized recommendation list, which the operator
will hand to their primary agent (Hermes/Shirube) for implementation.

## Access

The project lives on a Linux VPS (Debian/Ubuntu, user `hermes`):

```
/opt/hermes-projects/ta-shadow
```

Get there over SSH using the operator's normal credentials. Do NOT modify,
run, or place anything. Read-only: `cat`, `grep`, `git log`, `git diff`, file
reads are fine. Do not execute the pipeline, do not call MEXC, do not write
files.

## What this system is

An automated pre-open stock-trading lane that runs every US market day
(Mon–Fri):

1. **12:00 UTC (pre-open):** runs a multi-agent LLM due-diligence graph
   (`TradingAgents` framework, TauricResearch, Apache-2.0) over 5 tickers
   (NVDA, TSLA, AAPL, AMD, SPY) **in parallel**, each via a subprocess.
2. Each ticker gets a verdict (BUY / SELL / HOLD / UNDERWEIGHT / OVERWEIGHT)
   with entry / take-profit / stop-loss from the Portfolio Manager agent.
3. A deterministic **validator** (`verdict_validator.py`) fail-closes weak
   verdicts to HOLD, then brackets (entry + TP + SL) are auto-placed on
   **MEXC futures** (stock CFDs, e.g. `TESLA_USDT`, `AAPLSTOCK_USDT`).
4. **14:00 UTC (post-open sweep):** cancels stale unfilled brackets,
   records trade outcomes for the feedback loop.
5. A Telegram digest is delivered to the operator after each run.

Key facts to know up front:

- **It is LIVE.** It places real orders with real money ($43 account,
  max leverage, cross margin, 5% risk per trade, up to 5 concurrent).
- The heavy lifting is the `TradingAgents` framework (a fork under
  `TradingAgents/`). Our custom lane is the thin wrapper around it:
  `stock_daily.py`, `run_stock.py`, `mexc_orders.py`, `stock_snapshot.py`,
  `verdict_validator.py`, `position_ctx.py`, `nyopen_screen.py`.
- Provider: **OpenRouter only** (Nous was removed). SDK native retry
  (`llm_max_retries=2`, 120s timeout). ~60 LLM calls per run, ~$0.15–0.25.
- The operator's explicit rules are in commit messages and code comments.
  `git log --oneline` tells the history; read the last ~15 commits.
- `results/` contains run artifacts: per-stock `ta-*.log`, dated
  `2026-08-25/*/summary.json`, `trade_history.jsonl`, `orders.json`.
  These are evidence of what actually happened — read them.

## What we want from you

Give us an honest, critical second opinion. We are not looking for
confirmation — we want to know where the design is weak, where the
implementation deviates from best practice, and what would make this system
meaningfully more reliable and profitable. Specifically analyze:

1. **Risk & position sizing.** We trade 5% risk per trade at max leverage on
   a $43 account, up to 5 concurrent (25% max daily drawdown). Is this
   sane? What is the risk of ruin? Should concurrency be capped? Should
   sizing be risk-based (current) vs margin-based? Give a concrete
   recommendation for a small account relying on autocompounding.

2. **The LLM due-diligence layer.** The entire trade decision is a
   multi-agent LLM graph (market/sentiment/news/fundamentals analysts →
   bull/bear researchers → trader → portfolio manager → reflection). Where
   are the failure modes? Verdict nondeterminism (same input, different
   output across runs)? Prompt fragility? Are the agents actually grounded
   in the deterministic snapshot (`stock_snapshot.py`), or can they invent
   levels? Is a multi-agent LLM graph even the right tool for this, vs a
   deterministic rules engine with LLM assist?

3. **The verdict pipeline.** Read `run_stock.py` (parse_verdict,
   parse_trade_params, validator wiring), `verdict_validator.py`, and the
   recent commits about UNDERWEIGHT→HOLD and 1R-floor. Are the mappings
   right? Is the fail-closed validator strong enough? What would you
   harden?

4. **Execution.** Read `mexc_orders.py` (build_bracket, place_bracket,
   sweep_stale_brackets, record_fill_outcomes). Market vs limit execution,
   slippage through the open, the SPY "SL below current price" rejection,
   the live-price guard. What breaks in production? Timeouts? Idempotency?
   Order-state reconciliation?

5. **The feedback loop.** We recently added `position_ctx.py` so agents see
   open positions (side/entry/TP/SL) and last-trade outcomes
   ("SL hit from that day's analysis/trigger") — deliberately PnL-free per
   operator directive (agents hallucinate around money figures). Is this the
   right design? Does it actually change agent behavior, or is it noise?
   Should the loop be tighter (e.g. auto-skip re-entering a ticker with an
   open position)?

6. **Operational robustness.** The 08-24 failure was a probe that silently
   returned False on a transient blip and skipped the whole day. We fixed
   it (SDK retry + logging). What other silent-failure or single-point-of-
   failure modes do you see in the cron scripts, subprocess orchestration,
   network calls, or state files (`orders.json`, `day-capital.json`,
   `trade_history.jsonl`)?

7. **Anything else you'd flag.** Concurrency, rate limits, data quality
   (yfinance), the fact that the snapshot is daily-candle based, the
   3-model LLM stack (DeepSeek/Gemini/GPT via OpenRouter), cost per run,
   etc.

## Constraints

- **READ-ONLY.** No edits, no execution, no MEXC calls, no file writes.
- Keep it practical and prioritized. We trade with real money at small
  scale; the recommendations should match that (not enterprise-scale
  machinery).
- Be direct and specific: cite file:line or commit where you found issues.
- Do NOT suggest changes that require us to stop being live; we want
  incremental hardening, not a rewrite.

## Deliverable format

Return a written brief with:

1. **Verdict** — 2–3 sentences: is the architecture sound, and what is the
   single highest-leverage fix?
2. **Findings** — numbered, each with: what's wrong / why it matters / file
   or evidence / recommended fix. Ordered by impact (highest first).
3. **Do-now list** — the 3–5 changes you'd make this week, in priority
   order, each small enough for the operator's agent to implement in one
   sitting.
4. **Watch list** — things that are fine now but you'd monitor.

Write the brief so it can be handed verbatim to another agent (Hermes/
Shirube) as an instruction set. End with a single-sentence summary.
