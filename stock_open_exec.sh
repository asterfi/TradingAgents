#!/usr/bin/env bash
# ta-shadow 09:20 ET pre-open executor (2026-08-26 restructure):
# deterministic fresh-quote revalidation + immediate placement of today's
# staged decisions (hard cutoff 09:25 ET). NOT a second LLM analysis.
# Cron fires 13:20 UTC (= 09:20 ET during DST); script self-gates on the
# NY calendar in America/New_York.
cd /opt/hermes-projects/ta-shadow
set -a; source /opt/hermes-projects/ta-shadow/.env; set +a
exec .venv/bin/python stock_daily.py --execute-staged
