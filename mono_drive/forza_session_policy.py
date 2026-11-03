"""Shared explicit-stop and optional-deadline policy for local workers."""
import math


def remaining_seconds(config,now):
    if config.get('continuous') is True and config.get('started_at'):
        return None
    end=config.get('ends_at')
    if not isinstance(end,(int,float)) or not math.isfinite(end):
        return 0.
    return max(0.,end-now)


def session_active(config,now):
    remaining=remaining_seconds(config,now)
    return remaining is None or remaining>0.


def stop_background_workers(config,now,explicit_stop=False,storage_limit=False):
    """A game/recovery exception alone must not stop independent learning."""
    return explicit_stop or storage_limit or not session_active(config,now)
