"""Local, measured follow-up for a user-requested post-training driving trial."""
from pathlib import Path
import math
from .forza_local_learning import read_json,write_json
from .forza_incidents import tail_rows


def summarize(state):
    seconds=state.get('seconds',0.)
    lane_seconds=state.get('observed_lane_seconds',0.)
    return dict(driving_seconds=round(seconds,2),distance_m=round(state.get('distance_m',0.),2),
        average_kmh=round(state.get('speed_sum',0.)/seconds,2) if seconds else None,
        maximum_kmh=round(state.get('maximum_kmh',0.),2),
        fraction_55_to_65=round(state.get('near_target_seconds',0.)/seconds,4) if seconds else None,
        braking_fraction=round(state.get('brake_seconds',0.)/seconds,4) if seconds else None,
        stopped_seconds=round(state.get('stopped_seconds',0.),2),
        observed_lane_seconds=round(lane_seconds,2),
        estimated_centered_fraction=round(state.get('centered_seconds',0.)/lane_seconds,4) if lane_seconds else None,
        offroad_seconds=round(state.get('offroad_seconds',0.),2),
        mean_steering_change_per_second=round(state.get('steering_change',0.)/seconds,4) if seconds else None,
        inference_seconds=round(state.get('inferred_seconds',0.),2),
        measurement_note='Lane centering is a camera estimate, not independent ground truth.')


def update(root,heartbeat,session,now):
    root=Path(root);out=root/'out';request=read_json(out/'live-evaluation-request.json',{})
    candidate=request.get('candidate')
    if not candidate:return
    state_file=out/'live-evaluation-state.json';state=read_json(state_file,{})
    if state.get('candidate')!=candidate:state=dict(candidate=candidate,phase='waiting_for_training')
    if state.get('phase') in ('completed','failed','rejected_offline','inconclusive'):return
    if (out/'longrun.stop').exists():return
    activation=read_json(out/'model-activation.json',{})
    review=read_json(out/'vehicle-candidate-review.json',{})
    if review.get('candidate')==candidate and not review.get('accepted'):
        state.update(phase='rejected_offline',reason=review.get('reason'))
    elif activation.get('candidate')==candidate:
        if activation.get('state')=='rolled_back':
            state.update(phase='failed',reason=activation.get('reason'))
        elif activation.get('state') in ('trial','active'):
            state.setdefault('started_at',activation['started_at']);state['phase']='evaluating'
            run=Path(heartbeat.get('run','')).resolve()
            if run.parent==out.resolve():
                rows=tail_rows(run/'decisions.jsonl',max(state.get('last_t',0.),state['started_at']))
                previous=state.get('previous') if state.get('run')==str(run) else None
                state['run']=str(run)
                for row in rows:
                    if row['t']<=state.get('last_t',0.):continue
                    state['last_t']=row['t']
                    eligible=(row.get('path_model')==candidate and row.get('armed') and not row.get('recovery_phase')
                              and row.get('state')=='driving' and not row.get('race_position',0))
                    elapsed=row['t']-previous['t'] if previous else 0.
                    if eligible and previous and 0<elapsed<1.:
                        speed=float(row.get('speed_kmh',0.))
                        state['seconds']=state.get('seconds',0.)+elapsed
                        state['speed_sum']=state.get('speed_sum',0.)+speed*elapsed
                        state['maximum_kmh']=max(state.get('maximum_kmh',0.),speed)
                        for key,condition in [('near_target_seconds',55<=speed<=65),('stopped_seconds',speed<2.),
                                              ('brake_seconds',row.get('actual_brake',0)>5),
                                              ('inferred_seconds',row.get('inferred_lane',False)),
                                              ('offroad_seconds',(row.get('surface') or {}).get('offroad_confirmed',False))]:
                            state[key]=state.get(key,0.)+elapsed*bool(condition)
                        if row.get('position') and previous.get('position'):
                            distance=math.dist(row['position'],previous['position'])
                            if distance<max(5.,elapsed*35.):state['distance_m']=state.get('distance_m',0.)+distance
                        edges=row.get('lane_edges') or []
                        if edges and not row.get('inferred_lane') and row.get('confidence',0)>=.75:
                            near=min(edges,key=lambda e:abs(e[0]-6.))
                            state['observed_lane_seconds']=state.get('observed_lane_seconds',0.)+elapsed
                            if abs((near[1]+near[2])/2.)<=.5:state['centered_seconds']=state.get('centered_seconds',0.)+elapsed
                        if row.get('sent_command') and previous.get('sent_command'):
                            state['steering_change']=state.get('steering_change',0.)+abs(row['sent_command'][0]-previous['sent_command'][0])
                    previous=row if eligible else None
                state['previous']=previous
            route=read_json(out/'route-rotation-state.json',{}).get('current_id')
            if route:state['routes']=sorted(set(state.get('routes',[])+[route]))
            if state.get('seconds',0.)>=180. and activation['state']=='active':
                state.update(phase='completed',reason='measured_trial_complete')
            elif now-state['started_at']>900.:
                state.update(phase='inconclusive',reason='insufficient_active_driving_time')
        else:state['phase']='waiting_for_driving'
    elif review.get('candidate')==candidate:state['phase']='waiting_for_driving'
    state['time']=now;state['metrics']=summarize(state)
    write_json(state_file,state)
    if state['phase'] in ('completed','failed','rejected_offline','inconclusive'):
        report=out/'live-evaluations';report.mkdir(exist_ok=True)
        write_json(report/(Path(candidate).stem+'.json'),state)
