#!/usr/bin/env bash
# ta-shadow stock lane: pre-open TA + AUTO-PLACE brackets (12:00 UTC Mon-Fri)
# OPERATOR GO-LIVE 2026-08-24: automated placement ENABLED. Due-diligence
# verdicts pass the deterministic validator (fail-closed to HOLD); only
# validated BUY/SELL brackets reach MEXC. Max leverage, 5% equity risk,
# CROSS margin (openType 2), PM-chosen market/limit execution.
cd /opt/hermes-projects/ta-shadow || exit 1
exec .venv/bin/python stock_daily.py --preopen --execute
