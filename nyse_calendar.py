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
