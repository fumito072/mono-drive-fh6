"""Explain training eligibility without changing recordings or labels."""
from collections import Counter
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mono_drive.forza_video_labels import RejectedVideoIntervals

start=json.loads((ROOT/'out/longrun.json').read_text())['started_at']
labels=RejectedVideoIntervals(ROOT/'out/vlm-labels.jsonl')
counts=Counter()
failures=Counter()
longest=Counter()
def finish_group(group,kind):
    if len(group)<10:
        return
    positions=np.asarray([v['position'] for v in group])
    steps=np.linalg.norm(np.diff(positions,axis=0),axis=1)
    distance=float(steps.sum())
    if distance>=45 and steps.max()<=8 and abs(positions[-1,1]-positions[0,1])/distance<=.1:
        counts[kind+'_spatial_segments']+=1
        counts[kind+'_spatial_frames']+=len(group)
for source in sorted((ROOT/'out').glob('fh6_*')):
    file=source/'decisions.jsonl'
    if not file.exists():
        continue
    groups={'with_video':[], 'without_video':[]}
    with file.open(encoding='utf-8') as handle:
        for line in handle:
            try:
                row=json.loads(line)
            except ValueError:
                continue
            if row.get('t',0)<start or not row.get('frame'):
                continue
            counts['recorded_frames']+=1
            mode=row.get('mode')
            quality=(mode=='road_candidates' and row.get('confidence',0)>=.88) or (
                mode in ('paint_corridor','paint_pair') and row.get('confidence',0)>=.80)
            path=row.get('candidate_path') or []
            checks=dict(quality=quality, armed=bool(row.get('armed')),
                speed=6<=row.get('forward_kmh',0)<=55,
                brake=row.get('actual_brake',255)<80,
                free_roam=row.get('race_position',1)==0,
                centred=bool(path and abs(path[0][1])<.65),
                no_recovery=not row.get('recovery_phase'))
            for key,value in checks.items():
                if not value:
                    failures[key]+=1
            valid=all(checks.values())
            video_bad=labels.rejects(row['t'])
            counts['video_rejected']+=video_bad
            counts['base_valid']+=valid
            counts['valid_but_video_rejected']+=valid and video_bad
            for kind,group in groups.items():
                eligible=valid and (kind=='without_video' or not video_bad)
                if eligible and group:
                    previous=group[-1]
                    gap=row['t']-previous['t']
                    angle=abs(math.atan2(math.sin(row['yaw']-previous['yaw']),math.cos(row['yaw']-previous['yaw'])))
                    eligible=gap<1.2 and angle<.18
                    if not eligible:
                        failures[kind+'_continuity']+=1
                if eligible:
                    group.append(row)
                    longest[kind]=max(longest[kind],len(group))
                else:
                    finish_group(group,kind)
                    group.clear()
    for kind,group in groups.items():
        finish_group(group,kind)
print(json.dumps(dict(counts=counts,failures=failures,longest_group=longest),indent=2))
