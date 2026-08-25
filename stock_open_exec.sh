#!/usr/bin/env bash
# ta-shadow stock lane: post-open EXECUTION (13:35 UTC Mon-Fri, DST-aware
# window check inside the script). Places today's staged brackets inside
# the NYSE-open window with a fresh executable quote and full revalidation:
# admission governor (3-pos / 15% aggregate / same-ticker / fail-closed),
# live-price guard (entry/TP/SL vs mark), idempotent external order IDs,
# post-fill protection confirmation. Refuses to run outside the window.
cd /opt/hermes-projects/ta-shadow || exit 1
exec .venv/bin/python stock_daily.py --execute-staged
