"""Small, generic time-matching helpers shared across the library -- not
domain-specific to objects, matching, or MRMS; any caller comparing a target
datetime against a list of candidates can reuse this.
"""

from datetime import datetime


def nearest_within_tolerance(target: datetime, candidates: list[datetime], tolerance_minutes: float) -> datetime | None:
    """The candidate closest to target, or None if the closest one still
    exceeds tolerance_minutes -- callers report None as "skipped", never
    silently drop it."""
    if not candidates:
        return None
    best = min(candidates, key=lambda t: abs((t - target).total_seconds()))
    if abs((best - target).total_seconds()) > tolerance_minutes * 60:
        return None
    return best
