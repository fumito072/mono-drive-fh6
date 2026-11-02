"""Recording-level validation, with an explicit vehicle-specific deployment scope."""
from copy import deepcopy
from pathlib import Path

from .forza_local_learning import read_json


def group_recordings(evaluation, root):
    """Combine temporal partitions of one recording without averaging medians.

    Old reports have one finite mean per partition. New reports also carry the
    number of finite errors. Preserve partition diagnostics and global medians.
    Unknown provenance stays independent and cannot establish a vehicle scope.
    """
    result = deepcopy(evaluation)
    segments = result.get('human_segments', result.get('human_recordings', {}))
    groups = {}
    allowed = (Path(root) / 'datasets/human').resolve()
    for name, segment in segments.items():
        path = Path(name).resolve()
        provenance = read_json(path / 'provenance.json', {}) if allowed in path.parents else {}
        recording = provenance.get('recording', name)
        group = groups.setdefault(recording, dict(frames=0, models={}, car_ordinals=[], segments=[]))
        group['frames'] += segment.get('frames', 0)
        group['segments'].append(name)
        group['car_ordinals'] = sorted(set(group['car_ordinals']) | set(provenance.get('car_ordinals', [])))
        for model, score in segment.get('models', {}).items():
            count = score.get('valid_frames', segment.get('frames', 0))
            total = group['models'].setdefault(model, dict(mean_m=0., valid_frames=0))
            if count:
                total['mean_m'] += score['mean_m'] * count
                total['valid_frames'] += count
    for group in groups.values():
        for score in group['models'].values():
            score['mean_m'] = score['mean_m'] / score['valid_frames'] if score['valid_frames'] else None
    result.update(human_segments=segments, human_recordings=groups, recording_grouping_version=1)
    return result


def vehicle_recordings(evaluation, ordinal):
    """Keep unknown-camera recordings as checks, excluding only known other cars."""
    recordings = evaluation.get('human_recordings', {})
    matched = {name: row for name, row in recordings.items() if row.get('car_ordinals') == [ordinal]}
    checks = {name: row for name, row in recordings.items()
              if not row.get('car_ordinals') or ordinal in row['car_ordinals']}
    return matched, checks
