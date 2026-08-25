"""ta-shadow: NYSE session calendar (timezone/DST/holiday aware).

2026-08-25 hardening (Codex §"Execution and timing"):
- The 12:00 UTC pre-open analysis runs before the 09:30 New York cash open.
- Market entries must only execute inside a defined NYSE-open window.
- The sweep must run relative to the ACTUAL open (14:30 UTC during U.S.
  standard time, 13:30 UTC during DST), not a fixed 14:00 UTC cron.

No external dependency: a compact static table of NYSE holidays 2026-2028
plus weekend/09:30-16:00 ET session logic with DST handled by zoneinfo.
Shortened sessions (early closes 13:00 ET) are listed per year.
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# NYSE holidays (New Year, MLK, Presidents, Good Friday, Memorial, Juneteenth,
# Independence, Labor, Thanksgiving, Christmas) — observed dates 2026-2028.
NYSE_HOLIDAYS = {
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25),
    date(2027, 12, 24),
    # 2028
    date(2028, 1, 1), date(2028, 1, 17), date(2028, 2, 21),
    date(2028, 4, 14), date(2028, 5, 29), date(2028, 6, 19),
    date(2028, 7, 4), date(2028, 9, 4), date(2028, 11, 23),
    date(2028, 12, 25),
}

# Early-close days (13:00 ET): day after Thanksgiving, Christmas Eve (when a
# full trading day). Approximated per year by known NYSE schedule.
EARLY_CLOSES = {
    date(2026, 11, 27), date(2026, 12, 24),
    date(2027, 11, 26), date(2027, 12, 24),
    date(2028, 11, 24),
}


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def market_open_et(d: date) -> time:
    """Session open (ET). 09:30 normally; early-close days still open 09:30."""
    return time(9, 30)


def market_close_et(d: date) -> time:
    """Session close (ET): 16:00 normally, 13:00 on early-close days."""
    return time(13, 0) if d in EARLY_CLOSES else time(16, 0)


def next_open_utc(now: datetime | None = None) -> datetime:
    """The next NYSE open strictly after `now`, as an aware UTC datetime."""
    now = now or datetime.now(ZoneInfo("UTC"))
    d = now.astimezone(ET).date()
    while True:
        if is_trading_day(d):
            o = datetime.combine(d, market_open_et(d), tzinfo=ET)
            if o > now:
                return o.astimezone(ZoneInfo("UTC"))
        d += timedelta(days=1)


def now_utc() -> datetime:
    return datetime.now(ZoneInfo("UTC"))


def in_open_window(now: datetime | None = None, minutes_after_open: int = 45) -> bool:
    """True inside [open, open+minutes] on a trading day — the only window
    in which a staged market entry may execute against a fresh quote."""
    now = (now or now_utc()).astimezone(ET)
    d = now.date()
    if not is_trading_day(d):
        return False
    o = datetime.combine(d, market_open_et(d), tzinfo=ET)
    return o <= now <= o + timedelta(minutes=minutes_after_open)


# --- NY pre-open lane windows (2026-08-25/26 restructure) -------------------
# All times America/New_York on a TRADING day:
#   08:00  analysis start (5 tickers in parallel)
#   09:00  worst-case end (existing 3600s per-ticker timeout)
#   09:15  hard staging deadline — later arrivals never become eligible
#   09:20  executor: deterministic fresh-quote revalidation + placement
#   09:25  entry cutoff — no new entries initiated after this
#   09:29  cancel unfilled entry orders from this run
#   09:30  NYSE core session open
PREOPEN_ANALYSIS_START = time(8, 0)
PREOPEN_ANALYSIS_END = time(9, 0)      # worst-case timeout endpoint
STAGE_DEADLINE = time(9, 15)
EXEC_WINDOW_START = time(9, 20)
ENTRY_CUTOFF = time(9, 25)
ENTRY_CANCEL = time(9, 29)


def _et_today(now: datetime | None = None) -> tuple[date, datetime]:
    n = (now or now_utc()).astimezone(ET)
    return n.date(), n


def in_analysis_window(now: datetime | None = None) -> bool:
    """True 08:00–09:00 ET on a trading day (late starts allowed until the
    worst-case timeout endpoint; the 09:15 staging deadline still applies)."""
    d, n = _et_today(now)
    if not is_trading_day(d):
        return False
    s = datetime.combine(d, PREOPEN_ANALYSIS_START, tzinfo=ET)
    e = datetime.combine(d, PREOPEN_ANALYSIS_END, tzinfo=ET)
    return s <= n <= e


def staging_deadline_et(now: datetime | None = None) -> datetime:
    """The 09:15 ET staging deadline for the current/next trading day."""
    d, n = _et_today(now)
    for _ in range(8):
        if is_trading_day(d):
            dl = datetime.combine(d, STAGE_DEADLINE, tzinfo=ET)
            if dl > n:
                return dl
        d += timedelta(days=1)
    raise RuntimeError("no trading day found within a week")


def past_staging_deadline(now: datetime | None = None) -> bool:
    d, n = _et_today(now)
    if not is_trading_day(d):
        return True
    dl = datetime.combine(d, STAGE_DEADLINE, tzinfo=ET)
    return n > dl


def in_exec_window(now: datetime | None = None) -> bool:
    """True 09:20–09:25 ET on a trading day — the only window in which the
    pre-open executor may initiate entries (item 5 §brief: place immediately
    after revalidation; 09:25 is the hard entry cutoff)."""
    d, n = _et_today(now)
    if not is_trading_day(d):
        return False
    s = datetime.combine(d, EXEC_WINDOW_START, tzinfo=ET)
    e = datetime.combine(d, ENTRY_CUTOFF, tzinfo=ET)
    return s <= n <= e


def at_or_past_entry_cancel(now: datetime | None = None) -> bool:
    """True >= 09:29 ET on a trading day — unfilled entries get cancelled."""
    d, n = _et_today(now)
    if not is_trading_day(d):
        return False
    c = datetime.combine(d, ENTRY_CANCEL, tzinfo=ET)
    return n >= c


def in_cancel_window(now: datetime | None = None) -> bool:
    """09:29–09:35 ET: cancel unfilled entries, reconcile, never open new."""
    d, n = _et_today(now)
    if not is_trading_day(d):
        return False
    c = datetime.combine(d, ENTRY_CANCEL, tzinfo=ET)
    end = datetime.combine(d, time(9, 35), tzinfo=ET)
    return c <= n <= end


def et_date_str(now: datetime | None = None) -> str:
    """The NYSE trading date (YYYY-MM-DD) a pre-open run belongs to."""
    d, _ = _et_today(now)
    return d.isoformat()


def sweep_time_utc(now: datetime | None = None, minutes_after_open: int = 90) -> datetime:
    """The correct sweep time: open + N minutes (default 90) on the CURRENT
    or NEXT trading day, in UTC. A cron started at a fixed UTC hour can
    compare now >= sweep_time to decide whether the sweep is due yet."""
    now = now or now_utc()
    d = now.astimezone(ET).date()
    for _ in range(8):  # look up to a week ahead
        if is_trading_day(d):
            o = datetime.combine(d, market_open_et(d), tzinfo=ET)
            sweep = o + timedelta(minutes=minutes_after_open)
            if sweep.astimezone(ZoneInfo("UTC")) > now:
                return sweep.astimezone(ZoneInfo("UTC"))
        d += timedelta(days=1)
    raise RuntimeError("no trading day found within a week")
