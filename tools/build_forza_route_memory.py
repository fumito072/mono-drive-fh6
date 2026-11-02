"""Extract successful left-lane traversals from recorded game telemetry."""
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mono_drive.forza_video_labels import RejectedVideoIntervals


def bridge_is_verified(rows,rejected):
    """Only join measured lane sections across a continuously traversed road."""
    if len(rows)<2 or rows[-1]['t']-rows[0]['t']>20.:
        return False
    points=np.asarray([r['position'] for r in rows],dtype=float)
    if not np.isfinite(points).all():return False
    steps=np.linalg.norm(np.diff(points,axis=0),axis=1)
    if steps.sum()>100. or np.any(steps>15.):return False
    for previous,row in zip(rows,rows[1:]):
        surface=row.get('surface') or {}
        if (row['t']-previous['t']>1.5 or row.get('state')!='driving' or
            row.get('recovery_phase') or rejected.rejects(row['t']) or
            surface.get('offroad_confirmed') or surface.get('path_enters_sidewalk') or
            surface.get('road_support',0.)<.45 or
            row.get('mode') in ('known_failed_approach','wrong_way','median_boundary') or
            (row.get('static_clearance_m') or 50.)<5.):
            return False
    return True


def build(out):
    rejected=RejectedVideoIntervals(out/'vlm-labels.jsonl')
    segments=[]; group=[]; total=0
    def finish():
        nonlocal group
        if len(group)>=8:
            points=np.asarray([r['position'] for r in group],dtype=float)
            distance=float(np.linalg.norm(np.diff(points[:,[0,2]],axis=0),axis=1).sum())
            if distance>=15.:
                kept=[0]
                for i in range(1,len(points)):
                    if np.linalg.norm(points[i,[0,2]]-points[kept[-1],[0,2]])>=1.5:
                        kept.append(i)
                if len(kept)>=5:
                    segments.append(dict(id=len(segments),points=points[kept].tolist(),
                        yaws=[group[i]['yaw'] for i in kept],start=group[0]['t'],end=group[-1]['t'],
                        distance=round(distance,2)))
        group=[]
    for folder in sorted(out.glob('fh6_*')):
        file=folder/'decisions.jsonl'
        if not file.exists():continue
        finish()
        first_segment=len(segments)
        rows=[]
        for line in file.open(encoding='utf-8'):
            try:r=json.loads(line)
            except (ValueError,TypeError):continue
            total+=1
            rows.append(r)
            path=r.get('candidate_path') or []
            good=(r.get('state')=='driving' and 10.<r.get('speed_kmh',0)<65. and
                not r.get('inferred_lane') and not r.get('recovery_phase') and
                r.get('confidence',0)>=.75 and (r.get('lane_clearance_m') or 0.)>.4 and
                r.get('actual_brake',0)<40 and len(path)>=2 and not rejected.rejects(r['t']))
            if good:
                p=np.asarray(path)
                good=bool(p[0,0]<10. and abs(float(np.interp(6.,p[:,0],p[:,1])))<.65)
            if good and group:
                gap=r['t']-group[-1]['t']
                distance=np.linalg.norm(np.asarray(r['position'])-group[-1]['position'])
                turn=abs(math.atan2(math.sin(r['yaw']-group[-1]['yaw']),math.cos(r['yaw']-group[-1]['yaw'])))
                if gap>1.5 or distance>20. or turn>.35:finish()
            if good:group.append(r)
            elif group and r['t']-group[-1]['t']>1.5:finish()
        finish()
        # Missing paint inside an intersection used to truncate a successful
        # turn just before its apex. Keep the measured continuation only when
        # it reaches another measured lane section without an intervening fault.
        current=segments[first_segment:]
        joined=[]
        by_time={r['t']:i for i,r in enumerate(rows)}
        for segment in current:
            if joined:
                previous=joined[-1]
                middle=rows[by_time[previous['end']]:by_time[segment['start']]+1]
                if bridge_is_verified(middle,rejected):
                    bridge=middle[1:-1]
                    points=np.asarray(previous['points']+[r['position'] for r in bridge]+segment['points'])
                    yaws=previous['yaws']+[r['yaw'] for r in bridge]+segment['yaws']
                    keep=[0]
                    for i in range(1,len(points)):
                        if np.linalg.norm(points[i]-points[keep[-1]])>=1.5:keep.append(i)
                    previous.update(points=points[keep].tolist(),yaws=[yaws[i] for i in keep],end=segment['end'],
                        distance=round(float(np.linalg.norm(np.diff(points[:,[0,2]],axis=0),axis=1).sum()),2),
                        verified_bridges=previous.get('verified_bridges',0)+1)
                    continue
            joined.append(segment)
        segments[first_segment:]=joined
    finish()
    for index,segment in enumerate(segments):segment['id']=index
    result=dict(version=2,source='measured successful traversals with verified gap endpoints; runtime validation required',
                scanned_frames=total,segments=segments)
    temporary=out/'route-memory.tmp'
    temporary.write_text(json.dumps(result),encoding='utf-8')
    temporary.replace(out/'route-memory.json')
    print(json.dumps(dict(scanned_frames=total,segments=len(segments),points=sum(len(s['points']) for s in segments))))


if __name__=='__main__':build(ROOT/'out')
