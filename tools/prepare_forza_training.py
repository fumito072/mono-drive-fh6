"""Copy conservative, centred forward-driving intervals for candidate training.

These are automatically filtered trajectory pseudo-labels, not human ground truth.
Original recordings, including rejected intervals, remain unchanged.
"""
import csv
import json
import math
import os
from pathlib import Path
import shutil
import time
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mono_drive.forza_video_labels import RejectedVideoIntervals
from mono_drive.forza_training_filter import segment_geometry,training_row_valid
parser = argparse.ArgumentParser()
parser.add_argument('--scan-only',action='store_true')
args = parser.parse_args()

# Retention never removes source runs while this exporter is reading them.
from mono_drive.forza_data_worker import acquire_guard
retention_guard=acquire_guard(20076)
until=time.monotonic()+60.
while retention_guard is None and time.monotonic()<until:
    time.sleep(.5)
    retention_guard=acquire_guard(20076)
if retention_guard is None:
    raise SystemExit(75)
destination = ROOT/'datasets'/time.strftime('fh6_candidates_%Y%m%d_%H%M%S')
if not args.scan_only:
    destination.mkdir(parents=True)
config = json.loads((ROOT/'configs'/'forza_horizon6.json').read_text())
segments = []
rejected = kept = 0
video_labels=RejectedVideoIntervals(ROOT/'out'/'vlm-labels.jsonl')

def iter_log_lines(file):
    # A 20-hour recording must not be loaded into memory all at once.
    with file.open(encoding='utf-8') as lines:
        yield from lines

def export(group,source):
    global kept
    geometry=segment_geometry(group)
    if geometry is None:
        return
    if args.scan_only:
        segments.append(str(source))
        kept += len(group)
        return
    folder = destination/f'segment_{len(segments):03d}'
    (folder/'frames').mkdir(parents=True)
    with (folder/'poses.csv').open('w',newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['frame','t','x','y','z','yaw_deg','kmh'])
        for i,row in enumerate(group):
            original = source/row['frame']
            target = folder/'frames'/f'{i:07d}.jpg'
            try:
                os.link(original,target)
            except OSError:
                shutil.copy2(original,target)
            x,height,z = row['position']
            writer.writerow([i,row['t'],x,z,height,90-math.degrees(row['yaw']),row['speed_kmh']])
    camera=dict(config['camera'])
    projections=[r['projection'] for r in group if r.get('projection')]
    if projections:
        import numpy as np
        horizon=float(np.median([p['horizon'] for p in projections]))
        scale=float(np.median([p['scale'] for p in projections]))
        fy=.5*camera['width']/math.tan(math.radians(camera['hfov_deg'])/2.)
        camera.update(height_m=scale*camera['height']/fy,
                      pitch_deg=math.degrees(math.atan((.5-horizon)*camera['height']/fy)),
                      name='Recorded per-car calibrated game camera')
    (folder/'camera.json').write_text(json.dumps(camera,indent=2))
    (folder/'provenance.json').write_text(json.dumps(dict(source=str(source),
        frames=[r['frame'] for r in group],label_type='filtered executed-trajectory pseudo-label',
        human_verified=False,car_ordinal=group[0].get('car_ordinal'),
        label_basis='observed pose; paired paint or independently repeated route',**geometry),indent=2))
    segments.append(str(folder))
    kept += len(group)

for source in sorted((ROOT/'out').glob('fh6_*')):
    file = source/'decisions.jsonl'
    if not file.exists():
        continue
    group = []
    for line in iter_log_lines(file):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not row.get('frame'):
            continue
        valid = training_row_valid(row)
        valid = valid and not video_labels.rejects(row['t'])
        if valid and group:
            previous = group[-1]
            valid = (row['t']-previous['t'] < 1.2 and row.get('car_ordinal')==previous.get('car_ordinal') and
                     abs(math.atan2(math.sin(row['yaw']-previous['yaw']),math.cos(row['yaw']-previous['yaw']))) < .18)
        if not valid:
            export(group,source)
            group = []
            rejected += 1
        else:
            group.append(row)
    export(group,source)

summary = dict(segments=segments,frames=kept,rejected_frames=rejected,
    video_rejection_intervals=len(video_labels.intervals),
    automatic_filter=True,human_verified=False,activation='candidate only; never auto-install',
    policy='forward/free-roam/centred/road or paired paint or retained multi-traversal route/no blind inference/no recovery/no strong brake/continuous per-car pose; 30 frames and 22m, or 10 frames and 45m')
if not args.scan_only:
    (destination/'manifest.json').write_text(json.dumps(summary,indent=2))
    (ROOT/'out'/'training-dataset.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
