#!/usr/bin/env bash
# ta-shadow stock lane: pre-open TA + STAGING ONLY (12:00 UTC Mon-Fri).
# 2026-08-25 hardening (Codex review §"Execution and timing"): the 12:00 UTC
# analysis runs BEFORE the 09:30 NY open — decisions are staged here, never
# placed against the prior daily close. Live placement happens in the
# NYSE-open window via --execute-staged (see ta_stock_open_exec.sh), with a
# fresh MEXC executable quote and full revalidation (admission governor,
# live-price guard on entry/TP/SL).
cd /opt/hermes-projects/ta-shadow || exit 1
exec .venv/bin/python stock_daily.py --preopen
