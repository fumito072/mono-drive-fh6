"""Bounded, local-only experiment policy. Model replies are data, never code.

The video judge may nominate a preset; measured evidence must support it.
Training candidates are evaluated before they become future training seeds.
Nothing in this module installs a model into the live driving controller.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import time

PROFILES = {
    'balanced': dict(lr=1e-5, epochs=6, new_frames=100, curve_weight=1.0),
    'curves': dict(lr=1e-5, epochs=8, new_frames=75, curve_weight=2.0),
    'conservative': dict(lr=5e-6, epochs=4, new_frames=150, curve_weight=1.0),
}
FOCUSES = ('general', 'curves', 'lane', 'braking', 'observer', 'data_quality')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return default


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.learning.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for _ in range(4):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(.025)
    raise OSError(f'Could not publish {path.name}')


def json_rows(file, since=0., until=float('inf')):
    try:
        with Path(file).open(encoding='utf-8') as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    if since <= row.get('t', row.get('time', 0.)) <= until:
                        yield row
                except (ValueError, TypeError):
                    continue
    except OSError:
        return


def summarize_rows(rows):
    duration = speed_sum = stopped = recovery = braking = curve = lane = wrong_way = 0.
    samples = 0
    modes = Counter()
    brake_causes = Counter()
    previous = None
    for row in rows:
        if previous is not None:
            dt = row['t'] - previous['t']
            if 0 < dt <= 1.2:
                samples += 1
                speed = max(0., float(row.get('speed_kmh', 0.)))
                duration += dt
                speed_sum += speed * dt
                stopped += dt * (speed < 3.)
                recovery += dt * bool(row.get('recovery_phase'))
                command = row.get('sent_command') or [0., 0., 0.]
                braking += dt * (command[2] > .45 and not row.get('recovery_phase'))
                if command[2]>.03 and not row.get('recovery_phase'):
                    if row.get('applied_vehicle_brake',row.get('collision_brake_demand',0))>.03:
                        cause='vehicle'
                    elif row.get('applied_static_brake',row.get('static_brake_demand',0))>.03:
                        cause='static'
                    elif (row.get('command') or [0,0,0])[2]<=.03:
                        cause='release_tail'
                    else:
                        cause='speed_target'
                    brake_causes[cause]+=dt
                modes[str(row.get('mode'))] += dt
                path = row.get('candidate_path') or []
                curve += dt * (len(path) > 2 and abs(path[-1][1] - path[0][1]) > 3.)
                margin = row.get('lane_clearance_m')
                lane += dt * (margin is not None and margin < .15)
                wrong_way += dt * bool((row.get('carriageway') or {}).get('wrongway_confirmed'))
        previous = row
    divisor = max(.001, duration)
    return dict(samples=samples, seconds=round(duration, 2), mean_kmh=round(speed_sum/divisor, 2),
                stopped_fraction=round(stopped/divisor, 3), recovery_fraction=round(recovery/divisor, 3),
                hard_brake_fraction=round(braking/divisor, 3), curve_fraction=round(curve/divisor, 3),
                lane_edge_fraction=round(lane/divisor, 3), wrong_way_fraction=round(wrong_way/divisor, 3),
                modes_seconds={key: round(value, 1) for key, value in modes.items()},
                brake_causes_seconds={key:round(value,2) for key,value in brake_causes.items()})


def collect_metrics(out, since, until=None):
    out = Path(out)
    until = time.time() if until is None else until
    all_rows = []
    for folder in sorted(out.glob('fh6_*')):
        file = folder/'decisions.jsonl'
        try:
            if file.stat().st_mtime < since:
                continue
        except OSError:
            continue
        all_rows.extend(json_rows(file, since, until))
    all_rows.sort(key=lambda row: row['t'])
    result = summarize_rows(all_rows)
    labels = list(json_rows(out/'vlm-labels.jsonl', since, until))
    contradictions = sum(label.get('scene') in ('menu', 'loading') and label.get('telemetry_moving') is True for label in labels)
    result.update(window_start=since, window_end=until, observer_samples=len(labels),
                  observer_contradiction_fraction=round(contradictions/max(1, len(labels)), 3))
    data = read_json(out/'training-dataset.json', {})
    kept, rejected = data.get('frames', 0), data.get('rejected_frames', 0)
    result['accepted_training_frames'] = kept
    result['data_rejection_fraction'] = round(rejected/max(1, kept+rejected), 3)
    incidents=list(json_rows(out/'incident-events.jsonl',since,until))
    result['stops_analysed']=len(incidents)
    result['stop_causes']=dict(Counter(item.get('cause','unknown') for item in incidents))
    return result


def parse_review(text):
    """Exactly three integer codes; extra text cannot add executable actions."""
    start, end = text.find('['), text.find(']')
    value = json.loads(text[start:end+1])
    if not isinstance(value, list) or len(value) != 3 or any(type(x) is not int for x in value):
        raise ValueError('Review must contain three integer codes')
    profile, focus, confidence = value
    if not 0 <= profile < len(PROFILES) or not 0 <= focus < len(FOCUSES) or not 0 <= confidence <= 9:
        raise ValueError('Review code outside its allowed range')
    return dict(profile=tuple(PROFILES)[profile], focus=FOCUSES[focus], confidence=confidence/10.)


def validate_review(proposal, metrics):
    if metrics.get('seconds', 0) < 30 or metrics.get('samples', 0) < 100:
        return 'balanced', 'insufficient_observations'
    if proposal.get('confidence', 0) < .6:
        return 'balanced', 'low_review_confidence'
    profile = proposal.get('profile')
    if profile == 'curves' and metrics.get('curve_fraction', 0) >= .08:
        return profile, 'measured_curve_examples'
    if profile == 'conservative' and (metrics.get('data_rejection_fraction', 0) > .6 or
                                     metrics.get('observer_contradiction_fraction', 0) > .25 or
                                     metrics.get('wrong_way_fraction', 0) > 0. or
                                     metrics.get('recovery_fraction', 0) > .15):
        return profile, 'measured_data_or_observer_problem'
    return 'balanced', 'bounded_default'


def training_settings(out):
    config = read_json(Path(out)/'learning-policy.json', {})
    profile = config.get('profile', 'balanced')
    if profile not in PROFILES:
        profile = 'balanced'
    # Ignore arbitrary numeric values, paths, commands, and permissions in files.
    return profile, dict(PROFILES[profile])


def training_trigger(count, previous, profile, last_profile, last_trial_at, now):
    if count >= previous+PROFILES[profile]['new_frames']:
        return 'new_data'
    # Test a newly selected method once on sufficient retained data. A changing
    # model recommendation cannot create a rapid retraining loop.
    if count >= max(100,previous*.9) and profile!=last_profile and now-last_trial_at>=900.:
        return 'method_changed'
    return None


def publish_review(out, metrics, proposal, raw, model):
    out = Path(out)
    profile, reason = validate_review(proposal, metrics)
    previous = read_json(out/'learning-policy.json', {})
    if previous.get('rollback_until',0)>time.time():
        profile, reason = 'balanced', 'validation_rollback_cooldown'
    record = dict(time=time.time(), model=model, local_only=True, metrics=metrics, proposal=proposal,
                  applied_profile=profile, previous_profile=previous.get('profile', 'balanced'),
                  reason=reason, raw_reply=raw[:1000], automatic_driving_activation=False)
    # No policy changes while a candidate is being trained/evaluated.
    phase = read_json(out/'training-state.json', {}).get('phase')
    if phase not in ('training', 'evaluating'):
        write_json(out/'learning-policy.json', dict(profile=profile, time=record['time'], source='local_qwen_review',
                                                   previous_profile=record['previous_profile'],
                                                   rollback_until=previous.get('rollback_until',0), **PROFILES[profile]))
    else:
        record['deferred'] = True
    with (out/'learning-reviews.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False)+'\n')
    write_json(out/'learning-review-state.json', record)
    return record


def candidate_verdict(evaluation, baseline, candidate, vehicle_ordinal=None):
    """Compare on identical held-out examples, requiring enough finite evidence."""
    if evaluation.get('held_out_frames', 0) < 30:
        return False, 'too_few_held_out_frames'
    scores = evaluation.get('models', {})
    try:
        before, after = scores[str(baseline)], scores[str(candidate)]
        numbers = [before['mean_m'], after['mean_m'], before['median_m'], after['median_m']]
        if not all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in numbers):
            return False, 'nonfinite_scores'
        accepted = after['mean_m'] <= before['mean_m']*.98 and after['median_m'] <= before['median_m']*1.02
        if evaluation.get('human_holdout'):
            human=evaluation['human_holdout']
            if human.get('frames',0)<30:return False,'too_few_human_holdout_frames'
            h0,h1=human['models'][str(baseline)],human['models'][str(candidate)]
            if not (math.isfinite(h1['mean_m']) and h1['mean_m']<=h0['mean_m']*.98):
                return False,'human_holdout_did_not_improve'
            prior=evaluation['legacy_holdout']['models']
            if prior[str(candidate)]['mean_m']>prior[str(baseline)]['mean_m']*1.05:
                return False,'legacy_holdout_regression'
            # A large older recording must not conceal regression on a newly
            # added recording with enough independent validation examples.
            recordings=evaluation.get('human_recordings',{})
            if vehicle_ordinal is not None:
                from .forza_evaluation import vehicle_recordings
                if evaluation.get('recording_grouping_version') != 1:
                    return False,'missing_recording_provenance'
                matched,recordings=vehicle_recordings(evaluation,vehicle_ordinal)
                count=sum(row.get('frames',0) for row in matched.values())
                if len(matched)<3 or count<100:
                    return False,'insufficient_vehicle_holdout'
                old=sum(row['models'][str(baseline)]['mean_m']*row['frames'] for row in matched.values())/count
                new=sum(row['models'][str(candidate)]['mean_m']*row['frames'] for row in matched.values())/count
                if not all(math.isfinite(v) for v in (old,new)) or new>old*.98:
                    return False,'vehicle_holdout_did_not_improve'
            for recording in recordings.values():
                if recording.get('frames',0)<30:
                    continue
                old=recording['models'][str(baseline)]['mean_m']
                new=recording['models'][str(candidate)]['mean_m']
                if not all(type(x) in (int,float) and math.isfinite(x) for x in (old,new)) or new>old*1.05:
                    return False,'individual_human_recording_regression'
        if evaluation.get('evaluation_version',1)>=2:
            for name in ('blur','low_light'):
                conditions=evaluation.get('conditions',{}).get(name,{})
                old,new=conditions[str(baseline)]['mean_m'],conditions[str(candidate)]['mean_m']
                if not all(type(x) in (int,float) and math.isfinite(x) for x in (old,new)) or new>old*1.05:
                    return False, 'camera_degradation_regression'
        if vehicle_ordinal is not None and not evaluation.get('human_holdout'):
            return False,'insufficient_vehicle_holdout'
        return accepted, ('improved_vehicle_held_out_pseudo_labels' if vehicle_ordinal is not None else 'improved_held_out_pseudo_labels') if accepted else 'validation_did_not_improve'
    except (KeyError, TypeError):
        return False, 'missing_scores'


def record_candidate(out, evaluation, baseline, candidate, profile):
    accepted, reason = candidate_verdict(evaluation, baseline, candidate)
    record = dict(time=time.time(), accepted=accepted, reason=reason, profile=profile,
                  baseline=str(baseline), candidate=str(candidate), evaluation=evaluation,
                  automatic_driving_activation=False)
    with (Path(out)/'learning-experiments.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False)+'\n')
    if not accepted:
        # Roll back the experimental method, not the running controller.
        write_json(Path(out)/'learning-policy.json', dict(profile='balanced', source='validation_rollback',
                                                         time=time.time(), rollback_until=time.time()+600., **PROFILES['balanced']))
    write_json(Path(out)/'learning-experiment-state.json', record)
    return record
