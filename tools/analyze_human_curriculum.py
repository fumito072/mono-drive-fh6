"""Recheck prepared demonstrations and quantify maneuver/speed coverage.

Reads completed human segments only. No images or original recordings change.
These are recorded trajectories, not independently verified lane-center labels.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import time


def analyze(manifest,data_root=None):
    data=json.loads(Path(manifest).read_text(encoding='utf-8'))
    if data.get('episodes') and 'segments' in data['episodes'][0]:
        if data_root is None:raise ValueError('Public catalog needs --data-root for restored images')
        root=Path(data_root)
        data=dict(data,frames=data['total_frames'],episodes=[dict(e,recording=e['id'],revision=data['revision'],
            train=[str(root/s['path']) for s in e['segments'] if s['partition']=='train'],
            validation=[str(root/s['path']) for s in e['segments'] if s['partition']=='validation']) for e in data['episodes']])
    total=Counter();vehicles={};episodes=[]
    for episode in data['episodes']:
        key=','.join(str(n) for n in episode.get('car_ordinals',[])) or 'unspecified'
        vehicle=vehicles.setdefault(key,Counter());counts=Counter()
        for split in ('train','validation'):
            for path in episode[split]:
                root=Path(path)
                with (root/'poses.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
                counts[split+'_frames']+=len(rows)
                images=len(list((root/'frames').glob('*.jpg')))
                counts['image_label_count_errors']+=int(images!=len(rows))
                last=None
                for row in rows:
                    try:
                        t,x,y,yaw,speed=[float(row[k]) for k in ('t','x','y','yaw_deg','kmh')]
                        if not all(math.isfinite(v) for v in (t,x,y,yaw,speed)):raise ValueError()
                    except (ValueError,KeyError):counts['invalid_numeric_rows']+=1;last=None;continue
                    bucket='under_8' if speed<8 else '8_to_30' if speed<30 else '30_to_50' if speed<50 else '50_to_70' if speed<=70 else 'over_70'
                    counts['speed_'+bucket]+=1
                    if last:
                        dt=t-last[0];distance=math.hypot(x-last[1],y-last[2])
                        if not 0<dt<=.75:counts['time_discontinuities']+=1
                        elif distance>=.1:
                            angle=math.radians((yaw-last[3]+180)%360-180)
                            curvature=angle/distance
                            maneuver='straight' if abs(curvature)<.01 else ('left' if curvature>0 else 'right')+('_sharp' if abs(curvature)>.04 else '_gentle')
                            counts['maneuver_'+maneuver]+=1
                    last=(t,x,y,yaw)
                with (root/'actions.csv').open(encoding='utf-8',newline='') as f:
                    for row in csv.DictReader(f):
                        brake=float(row.get('brake',row.get('applied_brake',0)))
                        counts['recorded_brake_rows']+=int(brake>.03)
                provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
                if provenance.get('offset_verified_for_actions') is False:counts['estimated_alignment_frames']+=len(rows)
        total.update(counts);vehicle.update(counts)
        episodes.append({'id':hashlib.sha256((episode['recording']+episode['revision']).encode()).hexdigest()[:16],
                         'vehicles':key,'counts':dict(counts)})
    return {'time':time.time(),'revision':data['revision'],'episodes':len(episodes),'frames':data['frames'],
            'counts':dict(total),'vehicles':{k:dict(v) for k,v in vehicles.items()},'recordings':episodes,
            'label_limit':'Executed trajectories; not independent road/lane annotations. Data Out controls are not raw hardware inputs.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('out/human-dataset.json'))
    parser.add_argument('--out',type=Path,default=Path('out/human-curriculum-review.json'))
    parser.add_argument('--data-root',type=Path,help='Root used when restoring the public dataset')
    a=parser.parse_args();report=analyze(a.manifest,a.data_root)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('revision','episodes','frames','counts','vehicles')}))
