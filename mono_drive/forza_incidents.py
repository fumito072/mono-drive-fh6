"""Local stop diagnosis and persistent, evidence-gated failure episodes."""
from collections import deque
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from .forza_local_learning import read_json,write_json


def tail_rows(path,since=0.,limit_bytes=2_000_000):
    try:
        with Path(path).open('rb') as handle:
            size=handle.seek(0,2)
            handle.seek(max(0,size-limit_bytes))
            if size>limit_bytes:
                handle.readline()
            lines=handle.read().decode('utf-8',errors='replace').splitlines()
        result=[]
        for line in lines:
            try:
                row=json.loads(line)
                if row.get('t',0)>=since:
                    result.append(row)
            except (ValueError,TypeError):
                pass
        return result
    except OSError:
        return []


class StopWatch:
    """Emit once per standstill, rearm only after actual forward progress."""
    def __init__(self,seconds=8.):
        self.seconds=seconds
        self.started=None
        self.position=None
        self.reported=False

    def update(self,now,heartbeat):
        position=heartbeat.get('position')
        if now-heartbeat.get('time',0)>3. or position is None:
            return False
        if heartbeat.get('speed_kmh',0)>4.:
            self.started=self.position=None
            self.reported=False
            return False
        point=np.asarray(position,dtype=float)
        if point.shape!=(3,) or not np.isfinite(point).all():
            return False
        if self.position is None or np.linalg.norm(point-self.position)>2.:
            self.started,self.position,self.reported=now,point,False
        if not self.reported and now-self.started>=self.seconds:
            self.reported=True
            return True
        return False


def diagnose(rows):
    if not rows:
        return dict(cause='insufficient_data',action='observe',confidence=0.)
    end=rows[-1]['t']
    recent=[r for r in rows if r['t']>=end-8.]
    stationary=[r for r in recent if r.get('speed_kmh',0)<2.]
    recent=stationary or recent
    denominator=max(1,len(recent))
    fraction=lambda predicate:sum(bool(predicate(r)) for r in recent)/denominator
    stale=fraction(lambda r:r.get('state') in ('image/decision stale','telemetry stale','game unavailable'))
    traffic=fraction(lambda r:(r.get('collision_brake_demand') or 0)>.03 or r.get('mode')=='traffic_stop')
    offroad=fraction(lambda r:(r.get('surface') or {}).get('offroad_confirmed'))
    power=fraction(lambda r:(r.get('actual_accel') or 0)>55 and (r.get('actual_brake') or 0)<5 and not r.get('actual_handbrake'))
    static=fraction(lambda r:isinstance(r.get('static_clearance_m'),(int,float)) and 0.<r['static_clearance_m']<=8.)
    inferred=fraction(lambda r:r.get('inferred_lane',False))
    low_target=fraction(lambda r:isinstance(r.get('target_kmh'),(int,float)) and r['target_kmh']<1.)
    known=fraction(lambda r:r.get('mode')=='known_failed_approach' or
                   (r.get('failure_memory') or {}).get('action')=='known_failed_approach')
    metrics=dict(stale_fraction=round(stale,3),traffic_fraction=round(traffic,3),
        offroad_fraction=round(offroad,3),powered_fraction=round(power,3),
        close_static_fraction=round(static,3),inferred_fraction=round(inferred,3),
        zero_target_fraction=round(low_target,3))
    if known>.4:
        cause,action,confidence='known_failed_approach','recover_road',1.
    elif offroad>.2 or (rows[-1].get('surface') or {}).get('offroad_confirmed'):
        cause,action,confidence='confirmed_offroad','recover_road',.95
    elif stale>.3 and power<.25:
        cause,action,confidence='input_stale','restart_driver',.9
    elif traffic>.5 and power<.25:
        cause,action,confidence='waiting_for_traffic','observe',.85
    elif power>.25:
        cause=('static_contact' if static>.25 else
               'inferred_path_contact' if inferred>.5 else 'no_progress_under_power')
        action,confidence='recover_road',.9 if static>.25 else .75
    elif low_target>.6 and static<.2 and traffic<.2:
        cause,action,confidence='geometry_stop','restart_driver',.8
    else:
        cause,action,confidence='undetermined_stop','restart_driver',.5
    return dict(cause=cause,action=action,confidence=confidence,metrics=metrics)


def world_path(row):
    path=np.asarray(row.get('candidate_path') or [],dtype=float)
    position=np.asarray(row.get('position') or [],dtype=float)
    yaw=row.get('yaw')
    if (path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or position.shape!=(3,) or
            yaw is None or not np.isfinite(path).all() or not np.isfinite(position).all() or not math.isfinite(yaw)):
        return []
    from .forza_lane_memory import rotation
    return ((rotation(yaw)@path[:,[1,0]].T).T+position[[0,2]]).tolist()


def remember_failure(out,episode,rows):
    diagnosis=episode['diagnosis']
    if not rows or rows[-1].get('speed_kmh',float('inf'))>=3.:
        return None  # An intentional transition while moving is not a contact.
    if diagnosis['cause'] not in ('confirmed_offroad','static_contact','inferred_path_contact','no_progress_under_power'):
        return None
    moving=[r for r in rows if r.get('speed_kmh',0)>4. and r.get('position') and r.get('yaw') is not None]
    if not moving:
        return None  # No powered spawn/communication stall becomes a fake wall.
    last=rows[-1]
    if not last.get('position') or last.get('yaw') is None:
        return None
    location=np.asarray(last['position'],dtype=float)[[0,2]]
    if np.linalg.norm(location-np.asarray(moving[0]['position'])[[0,2]])<3.:
        return None
    yaw=float(last['yaw'])
    # A contact point is ahead of the stopped car, not under its centre.
    from .forza_lane_memory import rotation
    obstacle=(location+rotation(yaw)@np.array([0.,2.3])).tolist()
    safe=[r for r in moving if not r.get('inferred_lane') and
          r.get('confidence',0)>=.75 and (r.get('lane_clearance_m') or 0.)>.35]
    safe_row=safe[-1] if safe else None
    file=Path(out)/'failure-memory.json'
    memory=read_json(file,{'version':1,'entries':[]})
    entries=memory.get('entries',[])
    match=next((e for e in entries if np.linalg.norm(np.asarray(e['point'])-obstacle)<4. and
                abs(math.atan2(math.sin(e['yaw']-yaw),math.cos(e['yaw']-yaw)))<.8),None)
    entry=dict(id=match['id'] if match else episode['id'],point=obstacle,yaw=yaw,height=float(last['position'][1]),
        cause=diagnosis['cause'],confidence=diagnosis['confidence'],time=episode['time'],
        count=int(match.get('count',0))+1 if match else 1,
        safe_path=world_path(safe_row) if safe_row else [],episode=episode['id'])
    entries=[e for e in entries if e['id']!=entry['id']]+[entry]
    write_json(file,dict(version=1,entries=entries[-200:]))
    return entry['id']


def capture_incident(out,heartbeat,now=None):
    out=Path(out);now=time.time() if now is None else now
    run=Path(heartbeat.get('run',''))
    rows=tail_rows(run/'decisions.jsonl',now-45.)
    episode_id=f'{int(now*1000)}'
    folder=out/'incidents'/episode_id
    if (folder/'analysis.json').is_file():
        return read_json(folder/'analysis.json')
    folder.mkdir(parents=True,exist_ok=True)
    diagnosis=diagnose(rows)
    episode=dict(id=episode_id,time=now,run=str(run),diagnosis=diagnosis,
                 heartbeat=heartbeat,frames=[],auto_verified=False)
    # Preserve the approach, not merely the last still image. All files come
    # from the game's own recorder; no desktop capture or UI input is involved.
    saved=[r for r in rows if r.get('frame')]
    if saved:
        indices=np.unique(np.linspace(0,len(saved)-1,min(8,len(saved))).round().astype(int))
        for index in indices:
            row=saved[index];source=run/row['frame']
            if source.is_file():
                name=f'{len(episode["frames"]):02d}.jpg'
                shutil.copyfile(source,folder/name)
                episode['frames'].append(dict(time=row['t'],file=name))
    with (folder/'approach.jsonl').open('w',encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row)+'\n')
    episode['memory_id']=remember_failure(out,episode,rows)
    write_json(folder/'analysis.json',episode)
    write_json(out/'incident-state.json',dict(**episode,folder=str(folder),phase='analysed'))
    with (out/'incident-events.jsonl').open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(dict(time=now,id=episode_id,**diagnosis,memory_id=episode['memory_id']))+'\n')
    # Failed manoeuvres must not be copied into the positive training set.
    if episode['memory_id']:
        with (out/'incident-rejections.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(dict(start=now-15.,end=now+2.,incident=episode_id,
                                         validated=True,reason=diagnosis['cause']))+'\n')
    return episode
