"""Read-only checks of paired human video/Data Out recordings.

The result is an eligibility report, never an instruction to install a policy.
Elapsed monotonic time is preferred over wall-clock arrival timestamps.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np

REQUIRED=('elapsed_sec','game_timestamp_ms','rpm','position_x','position_y','position_z',
          'velocity_x','velocity_y','velocity_z','speed_kmh','throttle','brake','steer','gear')


def inspect_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        names=reader.fieldnames or []
        missing=[key for key in REQUIRED if key not in names]
        if missing:raise ValueError('Missing columns: '+', '.join(missing))
        rows=list(reader)
    if len(rows)<2:raise ValueError('At least two records are required')
    columns={key:np.asarray([float(row[key]) for row in rows]) for key in names}
    t=columns['elapsed_sec'];dt=np.diff(t)
    if not np.isfinite(t).all() or np.any(dt<=0):raise ValueError('Elapsed time must be finite and strictly increasing')
    finite=np.all(np.stack([np.isfinite(columns[key]) for key in REQUIRED]),axis=0)
    position=np.column_stack([columns['position_'+axis] for axis in 'xyz'])
    active=(columns['rpm']>0)&(np.linalg.norm(position,axis=1)>1)&finite
    ranges=(columns['throttle']>=0)&(columns['throttle']<=1)&(columns['brake']>=0)&(columns['brake']<=1)&(np.abs(columns['steer'])<=1)
    valid=active&ranges
    steps=np.linalg.norm(np.diff(position,axis=0),axis=1)
    pairs=valid[:-1]&valid[1:]
    teleports=pairs&(steps>np.maximum(5.,dt*60.))
    valid[np.flatnonzero(teleports)+1]=False
    pairs=valid[:-1]&valid[1:]
    intervals=[]
    edges=np.flatnonzero(np.diff(np.r_[False,valid,False])).reshape(-1,2)
    for first,end in edges:
        intervals.append(dict(start_csv_s=float(t[first]),end_csv_s=float(t[end-1]),rows=int(end-first)))
    duration=float(dt[pairs].sum())
    weight=np.r_[dt,np.median(dt)]*valid
    speed=columns['speed_kmh']
    fraction=lambda test:float(np.sum(weight*test)/max(1e-9,np.sum(weight)))
    norm=np.sqrt(sum(columns['velocity_'+a]**2 for a in 'xyz'))*3.6
    span=lambda key:dict(min=float(np.nanmin(columns[key])),max=float(np.nanmax(columns[key])),distinct=int(len(np.unique(columns[key]))))
    stats=dict(rows=len(rows),columns=names,csv_duration_s=float(t[-1]-t[0]),
        valid_rows=int(valid.sum()),invalid_rows=int((~valid).sum()),valid_duration_s=duration,
        median_hz=float(1/np.median(dt)),max_gap_s=float(dt.max()),gaps_over_100ms=int((dt>.1).sum()),
        duplicate_game_ticks=int((np.diff(columns['game_timestamp_ms'])==0).sum()),
        decreasing_game_ticks=int((np.diff(columns['game_timestamp_ms'])<0).sum()),
        duplicate_wall_times=int((np.diff(columns['timestamp'])==0).sum()) if 'timestamp' in columns else None,
        nonfinite_rows=int((~finite).sum()),out_of_range_rows=int((~ranges).sum()),teleport_steps=int(teleports.sum()),
        distance_m=float(steps[pairs].sum()),mean_kmh=fraction(speed),max_kmh=float(speed[valid].max()) if valid.any() else 0.,
        speed_norm_max_error_kmh=float(np.max(np.abs(norm[finite]-speed[finite]))),
        controls={key:span(key) for key in ('throttle','brake','steer')},
        active_fractions={key:fraction(np.abs(columns[key])>.01) for key in ('throttle','brake','steer')},
        speed_fractions={f'{lo}-{hi}':fraction((speed>=lo)&(speed<hi)) for lo,hi in ((0,5),(5,30),(30,50),(50,70),(70,100))},
        valid_intervals=intervals)
    return stats,columns,valid


def inspect_video(path):
    process=subprocess.run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)],
                           capture_output=True,text=True,check=True,timeout=60)
    metadata=json.loads(process.stdout)
    stream=next(s for s in metadata['streams'] if s.get('codec_type')=='video')
    a,b=map(float,stream['avg_frame_rate'].split('/'))
    return dict(width=stream['width'],height=stream['height'],fps=a/b,duration_s=float(stream.get('duration',metadata['format']['duration'])),
                frames=int(stream.get('nb_frames',0)),codec=stream['codec_name'],
                creation_time=stream.get('tags',{}).get('creation_time'))


def estimate_sync(columns,review):
    anchors=review.get('speed_anchors',[])
    if len(anchors)<3:return dict(status='unknown',verified=False)
    vt=np.array([p['video_s'] for p in anchors]);vs=np.array([p['hud_kmh'] for p in anchors])
    t=columns['elapsed_sec'];s=columns['speed_kmh']
    offsets=np.arange(-5.,5.0001,.001)
    predictions=np.stack([np.interp(vt+o,t,s) for o in offsets])
    # HUD values are whole km/h. Preserve the quantization uncertainty instead
    # of reporting a spurious millisecond-precise optimum.
    matches=np.all(np.floor(predictions)==vs,axis=1)
    best=float(offsets[np.argmin(np.mean(np.abs(predictions-(vs+.5)),axis=1))])
    bounds=[float(offsets[matches].min()),float(offsets[matches].max())] if matches.any() else None
    offset=round(best,2)
    return dict(status='visually_estimated',verified=review.get('sync_verified') is True,
                equation='csv_elapsed_s = video_pts_s + offset_s',offset_s=offset,
                compatible_offset_range_s=bounds,hud_floor_assumption=True,
                anchor_source=review.get('anchor_source','unspecified'),anchors=anchors,
                uncompensated_mae_kmh=float(np.mean(abs(np.interp(vt,t,s)-vs))),
                compensated_mae_kmh=float(np.mean(abs(np.interp(vt+offset,t,s)-vs))))


def audit_pair(video,csv_path,review=None):
    review=review or {}
    statistics,columns,valid=inspect_csv(csv_path)
    media=inspect_video(video)
    sync=estimate_sync(columns,review)
    flags=[]
    if statistics['nonfinite_rows'] or statistics['out_of_range_rows']:flags.append('invalid_numeric_rows')
    if statistics['teleport_steps']:flags.append('teleport_or_reset')
    if statistics['speed_norm_max_error_kmh']>1.:flags.append('inconsistent_speed_units')
    if statistics['valid_rows']<300 or statistics['distance_m']<50:flags.append('insufficient_driving')
    if statistics['controls']['steer']['distinct']<3:flags.append('missing_steering_variation')
    if statistics['controls']['throttle']['distinct']<3:flags.append('missing_throttle_variation')
    blockers=bool(flags)
    notes=['Only one recording: independent-session validation is still needed.',
           'Filter menus, invalid leading/trailing telemetry, collisions and lane departures before imitation.',
           'Data Out controls describe applied game state, not necessarily raw controller reports.',
           'Camera calibration, car ID and scene/race flags should be saved with future recordings.']
    path_bounds=sync.get('compatible_offset_range_s')
    path_candidate=bool(not blockers and review.get('path_candidate_allowed') is True and
                        path_bounds and path_bounds[1]-path_bounds[0]<=.15)
    if not sync['verified']:notes.append('Direct action learning remains disabled. An explicitly reviewed path-only experiment may retain the measured timing uncertainty.')
    return dict(version=1,recording=Path(video).stem,
        video=str(Path(video).resolve()),csv=str(Path(csv_path).resolve()),
        csv_sha256=hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        verdict='not_ready' if blockers else 'conditionally_usable',
        training_admitted=not blockers and sync['verified'] and review.get('driving_segments_reviewed') is True,
        path_candidate_eligible=path_candidate,direct_action_training_admitted=not blockers and sync['verified'] and review.get('driving_segments_reviewed') is True,
        numeric_eligible=not blockers,input_source=review.get('input_source','unspecified'),
        segment_reviewed=review.get('driving_segments_reviewed') is True,
        statistics=statistics,media=media,synchronization=sync,flags=flags,notes=notes,
        series=[dict(csv_s=round(float(x),3),kmh=round(float(np.interp(x,columns['elapsed_sec'],columns['speed_kmh'])),2),
                     throttle=round(float(np.interp(x,columns['elapsed_sec'],columns['throttle'])),3),
                     brake=round(float(np.interp(x,columns['elapsed_sec'],columns['brake'])),3),
                     steer=round(float(np.interp(x,columns['elapsed_sec'],columns['steer'])),3))
                for x in np.arange(0,statistics['csv_duration_s'],1.)])
