#!/usr/bin/env bash
# ta-shadow stock lane: NY-open fast screen (13:35 & 15:30 UTC Mon-Fri)
cd /opt/hermes-projects/ta-shadow || exit 1
exec .venv/bin/python stock_daily.py --screen
