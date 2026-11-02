"""Shared cruising ceiling and persisted session target, in km/h."""
import json
import math
from pathlib import Path

MAX_CRUISE_KMH = 60.
ROOT = Path(__file__).resolve().parents[1]


def cruise_limit(root=ROOT):
    """Recovery and worker restarts must retain the selected cruising target."""
    for path, keys in ((root/'out/longrun.json', ('target_kmh',)),
                       (root/'configs/forza_horizon6.json', ('forza', 'max_kmh'))):
        try:
            value = json.loads(path.read_text(encoding='utf-8-sig'))
            for key in keys:
                value = value[key]
            value = float(value)
            if math.isfinite(value) and 1 <= value <= MAX_CRUISE_KMH:
                return value
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return MAX_CRUISE_KMH
