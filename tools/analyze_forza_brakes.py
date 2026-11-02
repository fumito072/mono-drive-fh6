"""Attribute actual and requested brake time using timestamped driving logs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

parser=argparse.ArgumentParser()
parser.add_argument('--seconds',type=float,default=300.)
parser.add_argument('--since',type=float)
parser.add_argument('--until',type=float)
parser.add_argument('--out',default='out/brake-audit.json')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
since=args.since if args.since is not None else time.time()-args.seconds
until=args.until if args.until is not None else time.time()
causes=Counter()
states=Counter()
static_ranges=Counter()
duration=actual=delivered=moving=moving_brake=0.
episodes=[]
for folder in sorted((root/'out').glob('fh6_*')):
    file=folder/'decisions.jsonl'
    if not file.exists() or file.stat().st_mtime<since:
        continue
    previous=None
    episode=None
    with file.open(encoding='utf-8') as lines:
        for line in lines:
            try:
                row=json.loads(line)
            except ValueError:
                continue
            if row.get('t',0)<since or row.get('t',0)>until:
                continue
            if previous is None:
                previous=row
                continue
            dt=row['t']-previous['t']
            previous=row
            if not 0.<dt<1.:
                episode=None
                continue
            if row.get('recovery_phase'):
                continue
            duration+=dt
            actual+=dt*(row.get('actual_brake',0)>5)
            if row.get('speed_kmh',0)>3.:
                moving+=dt
                moving_brake+=dt*(row.get('actual_brake',0)>5)
            output=(row.get('sent_command') or [0,0,0])[2]
            requested=(row.get('command') or [0,0,0])[2]
            if output<=.03 and requested<=.03:
                episode=None
                continue
            delivered+=dt*(output>.03)
            states[row.get('state','unknown')]+=dt
            if row.get('applied_static_brake',row.get('static_brake_demand',0))>.03:
                cause='static_obstacle'
                clearance=row.get('static_clearance_m',50.)
                static_ranges['near_0_10m' if clearance<10 else 'middle_10_20m' if clearance<20 else 'far_20m_plus']+=dt
            elif row.get('applied_vehicle_brake',row.get('collision_brake_demand',0))>.03:
                cause='vehicle'
            elif row.get('mode') in ('lost','blocked'):
                cause='perception_loss'
            elif requested<=.03:
                cause='release_tail'
            else:
                cause='speed_or_lane_target'
            causes[cause]+=dt
            if episode is None or episode['cause']!=cause or row['t']-episode['end']>1.:
                episode=dict(cause=cause,start=row['t'],end=row['t'],seconds=0.,peak=0.,
                    speed=row['speed_kmh'],target=row['target_kmh'],confidence=row['confidence'],
                    clearance=row.get('static_clearance_m'),run=str(folder),frame=row.get('frame'),
                    perception=row.get('perception_recording'),perception_frame=row.get('perception_frame'))
                episodes.append(episode)
            episode['end']=row['t']
            episode['seconds']+=dt
            episode['peak']=max(episode['peak'],output,requested)
            if not episode['frame'] and row.get('frame'):
                episode['frame']=row['frame']
result=dict(since=since,until=until,observed_seconds=round(duration,2),
            actual_brake_seconds=round(actual,2),delivered_brake_seconds=round(delivered,2),
            moving_seconds=round(moving,2),moving_brake_seconds=round(moving_brake,2),
            moving_brake_fraction=round(moving_brake/max(.001,moving),4),
            attributed_seconds={k:round(v,2) for k,v in causes.most_common()},
            brake_states={k:round(v,2) for k,v in states.most_common()},
            static_distance_seconds={k:round(v,2) for k,v in static_ranges.items()},
            episodes=sorted(episodes,key=lambda x:x['seconds'],reverse=True)[:15])
(root/args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))
