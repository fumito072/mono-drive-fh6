"""Summarise timed driving samples without loading long logs into memory."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--seconds',type=float,default=900.)
parser.add_argument('--since',type=float)
parser.add_argument('--out')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
since=args.since if args.since is not None else time.time()-args.seconds
modes=Counter()
states=Counter()
duration=distance=moving_time=moving_distance=braking=hard_braking=0.
throttle_gap=0.
rows=changes=0
pedal_rates=[]
steer_rates=[]
for folder in sorted((root/'out').glob('fh6_*')):
    file=folder/'decisions.jsonl'
    if not file.exists() or file.stat().st_mtime<since:
        continue
    previous=None
    for line in file.open(encoding='utf-8'):
        try:
            row=json.loads(line)
        except ValueError:
            continue
        if row.get('t',0)<since:
            continue
        if previous is None:
            previous=row
            continue
        dt=row['t']-previous['t']
        if not 0.<dt<1.:
            previous=row
            continue
        duration+=dt
        states[row['state']]+=dt
        modes[row['mode']]+=dt
        speed=max(0.,row['speed_kmh'])
        distance+=speed*dt
        if speed>3.:
            moving_time+=dt
            moving_distance+=speed*dt
        command=row.get('sent_command',[0.,0.,0.])
        if not row.get('recovery_phase'):
            braking+=dt*(command[2]>.05)
            hard_braking+=dt*(command[2]>.45)
            throttle_gap+=dt*(speed<35 and row['target_kmh']-speed>10 and command[1]<.05 and command[2]<.05)
            prior=previous.get('sent_command',[0.,0.,0.])
            pedal_rates.append(max(abs(command[1]-prior[1]),abs(command[2]-prior[2]))/dt)
            steer_rates.append(abs(command[0]-prior[0])/dt)
            if row['mode']!=previous['mode']:
                changes+=1
        rows+=1
        previous=row
result=dict(since=since,until=time.time(),samples=rows,seconds=round(duration,1),
    mean_kmh=round(distance/max(.01,duration),2),moving_mean_kmh=round(moving_distance/max(.01,moving_time),2),
    stopped_seconds=round(duration-moving_time,1),braking_seconds=round(braking,1),
    hard_braking_seconds=round(hard_braking,1),throttle_gap_seconds=round(throttle_gap,1),mode_changes=changes,
    pedal_rate_p95=round(float(np.percentile(pedal_rates,95)),3) if pedal_rates else None,
    steer_rate_p95=round(float(np.percentile(steer_rates,95)),3) if steer_rates else None,
    modes_seconds={k:round(v,1) for k,v in modes.most_common()},
    states_seconds={k:round(v,1) for k,v in states.most_common()})
if args.out:
    Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
