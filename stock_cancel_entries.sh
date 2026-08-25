#!/usr/bin/env bash
# ta-shadow 09:29 ET pre-open entry cancellation (2026-08-26 restructure):
# cancel unfilled entry orders from today's run (surgical by order id;
# positions and TP/SL protection untouched). Cron fires 13:29 UTC
# (= 09:29 ET during DST); script self-gates on the NY calendar.
cd /opt/hermes-projects/ta-shadow
set -a; source /opt/hermes-projects/ta-shadow/.env; set +a
exec .venv/bin/python stock_daily.py --cancel-entries
