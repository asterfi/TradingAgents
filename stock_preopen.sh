#!/usr/bin/env bash
# ta-shadow NY pre-open lane (2026-08-26 restructure):
#   08:00 ET analysis + STAGING ONLY. Execution is a separate phase.
# Cron fires at 12:00 UTC (08:00 ET during DST); the script itself verifies
# America/New_York timing via nyse_calendar (works across DST changes).
cd /opt/hermes-projects/ta-shadow
set -a; source /opt/hermes-projects/ta-shadow/.env; set +a
exec .venv/bin/python stock_daily.py --preopen
