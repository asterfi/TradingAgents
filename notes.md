# ta-shadow — TradingAgents Shadow Advisor Lane

**Project:** Does multi-agent LLM judgment (TradingAgents, Tauric Research) add anything on top of
btc-swing's five deterministic rules?
**Mode:** SHADOW ONLY — observes, opines, logs. Never places orders, never blocks/modifies btc-swing.
**Isolation:** separate from btc-swing and hf-scalper — no shared code/config/rules.
**Source:** https://github.com/TauricResearch/TradingAgents (Apache-2.0, arXiv 2412.20138)
**Question:** log TradingAgents verdicts alongside btc-swing signals; compare after 20–30 closed trades.

## Hard constraints
- No orders, ever. No API keys with trade permission anywhere in this lane.
- btc-swing is untouchable without explicit separate instruction.
- Report token cost at every phase.
- A clean "this doesn't work and here is why" is a successful result.

## Operator amendments (2026-08-23, Phase 2 approval)
- **Three-policy evaluation (Phase 3/4):** every shadow verdict row also records what an
  "always agree" and an "always ignore" policy would have produced → Phase 4 compares three
  policies (TradingAgents verdicts vs always-agree vs always-ignore), not two.
- **No hindsight contamination:** verdicts are logged BEFORE the trade outcome is known,
  timestamped at log time; outcome is appended later to the same row.
- **Baseline observation caveat:** Phase 1 said HOLD/don't chase BTC while the operator holds a
  BTC long entered on a late breakout. That is ONE observation, not evidence. It must not
  influence any btc-swing rule or execution until Phase 4 has 20 closed trades.

## Operator amendments (2026-08-23, Phase 3 approval)
- **#1 Verdict distribution:** count BUY/HOLD/SELL in the log and report at evaluation. If HOLD
  exceeds ~80% of verdicts, flag it as a possible default-output problem, NOT a finding.
- **#2 Non-determinism from the start:** run each shadow evaluation TWICE per signal on different
  prompt orderings (variant A = canonical bullet/constraint order; variant B = reversed). Both
  verdicts logged; disagreement = the non-determinism measure (`sibling.agree` on each row).
- **#3 position_blind variant:** same run with position/equity/risk fields stripped from the
  snapshot → judges the signal, not the operator's trade. Both verdicts logged per signal;
  evaluation compares whether knowing the position changed the answer.
- Per signal day: **4 runs** (A/B × positioned/position_blind), one evaluation per signal day
  maximum, 00:00 UTC window, signal-days only.

## Phases
- **Phase 1 — Install & baseline (COMPLETE):** clone, own venv (py3.12), DeepSeek config
  (provider deepseek, deep_think deepseek-v4-pro, quick_think deepseek-v4-flash,
  max_debate_rounds 1, temperature 0.0). One unmodified run on BTC-USD → report decision,
  full agent output, wall-clock time, token cost. STOP and report.
- **Phase 2 — Customize (COMPLETE):** crypto fundamentals agent, btc-swing snapshot
  grounding, fenced sentiment, cost control (1 run/day max, 00:00 UTC, only on signal days).
  STOP and report.
- **Phase 3 — Shadow logging (IN PROGRESS):** approved 2026-08-23 with three additions
  (verdict distribution + HOLD>=80% flag; double-run A/B prompt-order variants with sibling
  verdicts; position_blind mode). Driver `shadow_daily.py` (cron 00:10 UTC, no_agent) evaluates
  each signal day's first signal 4x, appends outcomes when trades close, reports distribution.
  Phase 4 needs >= 20 closed trades.
- **Phase 4 — Evaluation (after 20 closed trades):** ta-shadow/evaluation.md, agree/disagree
  performance, veto counterfactual equity curves (3 policies), non-determinism check,
  total cost, honest verdict.

## Lane log
- **#1 (2026-08-23):** Phase 1 COMPLETE — cloned v0.3.1, own venv, DeepSeek config (pro/flash,
  max_debate_rounds 1, temp 0.0). BTC-USD baseline: **HOLD**, 808s wall, 238,437 tokens
  (170,374 in / 68,063 out, 16 LLM calls), **≈ $0.11–0.21/run**. Full report:
  `research/phase1-baseline-2026-08-23.md`.
- **#2 (2026-08-23):** Phase 2 COMPLETE — branch `shadow-stock-baseline` (stock intact) +
  `shadow-crypto-custom` (all changes). Customizations: (2.1) crypto fundamentals agent
  (funding/OI+5d/ETF/stablecoin; exchange netflows = no free source, honest N/A);
  (2.2) btc-swing snapshot injected into state (AgentState.shadow_snapshot) + grounding
  blocks in market/news prompts + citation fence in bull/bear/trader/risk/PM; no external
  data fetches in shadow mode; verifier `scripts/verify_grounding.py` (no invented figures
  found; benign warnings = comma-split numbers, OI live values, derived ratios now labeled);
  (2.3) sentiment = single labeled crowd-mood line, deterministic, zero LLM cost, flagged
  NOT EVIDENCE; (2.4) cost control: 1 run/day, 00:00 UTC, signal-days only (Phase 3 cron),
  cost logged per run.
  **Verification run (BTCUSDT, snapshot 2026-08-22):** HOLD, 481s, 11 LLM calls, 33,280 in /
  47,991 out tokens, **≈ $0.08–0.23 (peak), ~$0.10 mid; off-peak ≈ $0.05** — 66% cheaper and
  40% faster than stock baseline. Log row written pre-outcome with 3-policy fields.
  Report: `research/phase2-customization-2026-08-23.md`. Awaiting operator go-ahead for Phase 3.
- **#3 (2026-08-23):** Phase 3 IMPLEMENTED + VERIFIED — additions #1/#2/#3 live. Driver
  `shadow_daily.py` + cron `ta-shadow-daily` (88fb627964f3, 00:10 UTC, no_agent, silent on
  no-signal days). Fixed Phase 2 row's inverted policy fields (annotated `policy_corrected`).
  Verification quad on 2026-08-22 BTC LONG signal (position open → pre-outcome): A/pos **HOLD**,
  B/pos **HOLD** (order-stable), A/blind **HOLD**, B/blind **BUY** (order-sensitive) — 4×11 LLM
  calls, 348K tokens total, **$0.33–0.98 (mid $0.43)** for the day's 4-run eval; rows flagged
  `test:true` (excluded from Phase 4). Non-determinism + position-sensitivity machinery works.
  Real evaluation of pending signal days starts with the 00:10 cron on 2026-08-24.

## 2026-08-23 — Realtime agent view + 5-token daily screen + grounding decontamination
- LIVE VIEW: TradingAgents graph now streams per-node progress (langgraph stream values-mode, signature-field diffing) into hq/live_run.json; HQ card shows current agent + steps + tokens while running, and the PORTFOLIO MANAGER's final analysis (brief) when done. 5s polling during a run, 30s idle. (User: per-agent text walls too much → slimmed to final analysis only.)
- DAILY SCREEN: 5-token watchlist (BTC/ETH/SOL/XRP/DOGE, the btc-swing universe) — one run per token per day at 00:10 UTC (kind=screen, variant A, POSITION_BLIND), ~$15/mo, ~45 min/day, own lock (.screen.lock), dedupe via SCREEN_INTERVAL_H (23 = daily; 11 = 2x, 5 = 4h). Screen rows have signal=None → excluded from verdict distribution, sibling annotation, and Phase 4 automatically. Dashboard card + cron summary line (🔭).
- GROUNDING FIX (user catch): the snapshot prompt contained btc-swing's OWN conclusion — `setup: ... (signal_side: ...)` — contaminating the LLM judgment (agents anchored to the rules' verdict; corrupts the "does LLM add edge" question). REMOVED. high20/low20 trigger labels neutralized (raw 20d extremes, no LONG/SHORT hint). Position/equity/risk remain ONLY in positioned signal-eval runs (Phase 3 amendment); screens are position_blind by design.
- Progress field mapping corrected: investment_plan = Research Manager's plan in this version; Portfolio Manager writes final_trade_decision (now captured as the final-analysis text).
- Cost guard: 3 verification runs today ≈ $0.37 total (test rows, excluded from Phase 4).

## 2026-08-23 — Model re-point: Nous portal + deepseek-v4-flash (directive #2)
- Operator directive: ta-shadow uses **deepseek/deepseek-v4-flash on the Nous portal ONLY** for BOTH think tiers. Nothing else (no ox-alpha, no qwen, no pro). Hard rule — operator restated explicitly.
- Replaces the earlier ox-alpha wiring (directive #1, same day). ox-alpha (free preview, no SLA) suffered capacity droughts that killed full graph runs (429 waves + empty bursts). deepseek-v4-flash on the same portal: plain invoke 2.0s, tool-call invoke 1.7s, stable.
- run_shadow.py: `PROVIDER=nous`, `DEEP_THINK_LLM=deepseek/deepseek-v4-flash`, `QUICK_THINK_LLM=deepseek/deepseek-v4-flash`. Credential bridge (auth.json agent_key, rotating) unchanged. Guards unchanged (still useful for portal capacity waves).
- Cost: ox-alpha-only rows stay $0; deepseek rows now estimated at DeepSeek published list rates with note "estimate at DeepSeek list rates via nous portal" (portal exposes no per-token pricing).
- Nous portal probes: /models 403 (not exposed); deepseek-v4-pro returns empty content (reasoning model, flaky on portal — avoid); qwen3-coder-next works but is OFF-LIMITS for this lane per operator.
- Verification status: see #4 below (full graph test run).

## 2026-08-23 — btc-swing PAUSED by operator (focus: ta-shadow)
- Operator decision: pause/stop btc-swing for now, focus on the TradingAgents lane. Confirmed option: pause all; the 2 open positions (BTC LONG 77,468.50/stop 72,299.00/TP1 82,638.00 0.0019 BTC; ETH LONG 2,439.18/stop 2,324.60/TP1 2,553.76 0.06 ETH) ride on their existing MEXC stops.
- Paused crons: 9ce3b7ed5b55 (btc-swing scan 4h), a72e4a9acae4 (weekly), f5617e6bb59a (#16 gate one-shot, moot while paused). Resume = unpause + fresh scan.
- ta-shadow impact: STALE_SCAN_H=26 stale-guard added to shadow_daily.py — pending_screens() returns [] when the latest btc-swing scan is older than 26h, so daily screens can NEVER evaluate a frozen snapshot. Screens resume automatically when the scanner restarts.
- Pending 08-22 BTC signal eval still fires 00:10Z 08-24 (snapshot from last scan is fine); outcomes will not append while the journal is frozen — expected under pause.

## 2026-08-23 — STOCK LANE PIVOT (operator directive, supersedes crypto lane)
- Operator: remove ALL crypto; ta-shadow focuses on **5 US stocks: NVDA/TSLA/AAPL/AMD/SPY**; NY-open first-2h focus; alerts to Home channel; auto-place limit orders with TP/SL before NY open via MEXC futures stock perps.
- **Removed:** run_shadow.py, shadow_daily.py, run_baseline.py, baseline/, research/, scripts/, results/ (recreated), shadow-log.jsonl, shadow-state.json.test, hq/ dashboard (killed :8788 + deleted), cron ta-shadow-daily (88fb627964f3). Crypto analyst + crypto shadow branches stripped from vendored TradingAgents. Backed up: /var/backups/ta-shadow/ta-shadow-crypto-lane-20260823.tar.gz.
- **Stock lane files (new):**
  - `stock_snapshot.py` — yfinance snapshot per ticker: close/prior/SMA20/50/200, 20d/52w H-L, ATR14, pivots, avg vol; session: open/gap/VWAP/first-30m range/vol pace; **DMC layer** (HTF daily body-levels + LTF session body-levels, ALIGNED/MISALIGNED states).
  - `run_stock.py` — full upstream TradingAgents stock graph grounded on snapshot (kind=stock); extract ta_verdict + confidence + trade_params (entry/TP/SL); Nous + deepseek-v4-flash both tiers.
  - `nyopen_screen.py` — deterministic screen: gap/volume/level-proximity/range-lock flags, opportunity alert + key-levels table.
  - `mexc_orders.py` — MEXC futures bracket (limit + stopLossPrice + takeProfitPrice in one create); STAGE default, `--execute` to place. Perps verified live: NVIDIA_USDT/TESLA_USDT/AAPLSTOCK_USDT/AMDSTOCK_USDT/SPY_USDT (apiAllowed, minVol 1, cs 0.01-0.001).
  - `stock_daily.py` — driver: `--preopen` (parallel 5-stock TA → stage brackets into orders.json → Home summary) and `--screen` (fast screen).
- **Crons (new, no_agent, deliver to telegram Home):** `ta-stock-preopen` (730414691df3, 0 12 * * 1-5) + `ta-stock-nyopen-screen` (9b4b5015059f, 35 13,15 * * 1-5). Scripts in ~/.hermes/scripts/ta_stock_{preopen,screen}.sh.
- **Sizing (operator, 2026-08-23):** risk **2.6% of equity per trade** (~$0.97 on $37.5), **MAX leverage per symbol** (NVIDIA/AMD 200x, TESLA/AAPL 100x, SPY 50x — verified live from contract detail, NOT a flat 200x), isolated margin. vol = risk/(stop_dist×contractSize), rounded to volScale (0 = integer), floor minVol 1.
- **MEXC base URL:** api.mexc.com (NOT futures.mexc.com — 404). Read-only key verified (37.49 USDT available). Write permission NOT yet verified — live placement pending key permission check.
- **MEXC signing bug FIXED (2026-08-24):** `_mexc_request` double-signed (passed the full `api_key+ts+qs` target into `_sign`, which prepends again). Fixed to pass raw qs/payload. Verified live: place_bracket placed a real limit order (orderId 846717520670467200, NVIDIA_USDT @100, TP/SL attached) → confirmed on book → cancelled. **Write permission CONFIRMED.** `cancel_order` uses `/order/cancel_all` (single `/order/cancel` 600s "Parameter error" regardless of body — MEXC quirk).
- **Key rotation fix:** Nous agent_key rotates hourly; `NousChatOpenAI._refresh_key()` re-reads ~/.hermes/auth.json before EVERY attempt + refreshes on 401 (fixed the mid-run Bull-Researcher 401 death).
- **DMC timeframe (operator Q):** HTF daily body-levels (anchor, matches analysts' indicators) + LTF session body-levels (5m) for alignment — DMC is multi-TF; both now in the grounding block.
- **Verification status (2026-08-24):** ✅ **FULL END-TO-END NVDA RUN COMPLETE** on OpenRouter — 12 LLM calls, 285s (4.75 min/stock), verdict SELL (PM "Underweight"), full grounding block with DMC levels. **Provider switch (directive #5):** Nous → **OpenRouter** (stable, no hourly key rotation, no capacity flaps). Model: **`~deepseek/deepseek-v4-flash-latest`** (OpenRouter follow-latest alias → resolves deepseek-v4-flash-0731) at **max effort** (`reasoning_effort=max`, wired for openrouter provider + passthrough fix). Key bridged at runtime from Hermes env file (never stored in lane). `max_tokens=3000` caps the model's unbounded hidden-thinking phase (measured: unbounded → 28K empty stream chunks/0 content/no finish; capped → 57s, finish=stop, full report). Max-effort run: 13 calls / 731s / HOLD (differs from non-max SELL — deeper reasoning + newer build). Full graph ~5-12 min/stock → 5-stock parallel pre-open fits the 90-min window. No live orders placed yet.

## 2026-08-24 — THREE-MODEL ROLE-AWARE ARCHITECTURE (spec ta-shadow-tradingagents-final-configuration.md)
- **Spec status: IMPLEMENTED + VERIFIED end-to-end** (NVDA test24: 13 calls / 667s / SELL, validation ok, $0.058).
- **Role routing (spec §4):** DeepSeek **deepseek-v4-flash-0731** (pinned, high/12288 etc — 8 calls: 4 analysts + bull + aggressive + neutral + 1 extra market loop) / Gemini **gemini-3.7-flash** (2 calls: bear + conservative, high/16384) / Luna **gpt-5.6-luna-pro** (3 calls: research manager xhigh/32768 + trader high/16384 + portfolio manager xhigh/32768). Reflection → deepseek low/4096.
- **Removed global config:** `max_tokens=393216`, `reasoning_effort=max`, `temperature=1.0`, `top_p=0.95` — replaced by per-role (model, effort, max_tokens) in `ROLE_LLMS` (run_stock.py). No forced temp/top_p (provider defaults, spec §3.4).
- **Graph controls (spec §5):** max_debate_rounds=1, max_risk_discuss_rounds=1, max_recur_limit=40, checkpoint_enabled=True. `_run_signature` now includes role→model map so checkpoints never resume across a model change (§17).
- **Client registry (spec §6):** trading_graph builds one client per unique (model,effort,mt) combo (7 unique) and passes `role_llms` dict to GraphSetup; setup.py routes per node; Reflector uses reflection role.
- **Telemetry (spec §14/§15):** StatsCallbackHandler records per-call {model, latency_ms, input/output tokens}; run_stock logs `per_call`, `cost_by_model` (per-model OpenRouter rates), `role_llms` map. Real cost verified: $0.058 for a full NVDA run (8 deepseek + 2 gemini + 3 luna).
- **Deterministic validator (spec §10):** `verdict_validator.py` — snapshot hash/id identity, level-provenance (TP/SL/entry must map to snapshot levels within 5%), direction consistency, ATR-distance (>=0.15×ATR14), min R/R (>=1.0), schema/model-identity; fail-closed to HOLD with machine-readable reasons. Wired into run_stock after parse.
- **Shadow mode restored (spec §scope):** preopen cron now STAGES only (no `--execute`) — LLM output advisory, not execution authority. Flip to live only on explicit operator go.
- **Crons unchanged:** ta-stock-preopen (12:00Z Mon-Fri) + ta-stock-nyopen-screen (13:35/15:30Z) → Home. Both no_agent script jobs.
- **Price note:** pinned deepseek-v4-flash-0731 = $0.14/$0.28 per 1M (3.5× the ~latest alias at $0.04/$0.08) — spec §15's $0.04 assumption is stale; real per-run cost measured instead.

## 2026-08-24 — RISK 5% + AGENT-CHOSEN EXECUTION (operator directives)
- **Risk per trade: 2.6% → 5%** of live equity (RISK_PCT stock_daily.py + DEFAULT_RISK_PCT mexc_orders.py). ~$1.88/trade on $37.5 equity. Verified: NVDA LONG risk $1.88 @ 5%.
- **Agents now choose market vs limit execution** (operator: "ask the agents if we should do market orders and just place TP/SL limits").
  - `PortfolioDecision.execution_type` field added (spec §9 `entry.type: market|limit|none`), rendered as `**Execution**: market|limit` in the PM markdown.
  - `parse_trade_params` extracts it (regex tolerant of `**markdown**` labels); `stage_from_verdict` threads it to `build_bracket` → `place_bracket`.
  - MEXC: `type: 2` (market, no `price` field, TP/SL attached) vs `type: 1` (limit with entry price). Verified dry-run body.
  - Validator: rejects invalid execution values + `none` with directional verdict; market skips entry-price provenance (fill unknown until open), TP/SL still validated.
  - Digest shows `🎯 MARKET` / `🎯 LIMIT` per staged asset.
- Full chain tested: PM "Execution: market" → parsed → validated (ok) → bracket risk $1.88 lev 200x → MEXC type 2 payload. Validator correctly fail-closed R/R 0.92 (<1.0) test case.
