"""Promote evaluated local candidates and roll back a failed live trial."""
import hashlib
import json
from pathlib import Path
import time
import math
from .forza_local_learning import read_json,write_json,candidate_verdict


def model_path(root,value):
    path=Path(value)
    return (path if path.is_absolute() else root/path).resolve()


def offer(root,candidate,baseline,evaluation,vehicle_ordinal=None):
    root=Path(root).resolve();candidate=model_path(root,candidate)
    allowed=(root/'models/candidates').resolve()
    if allowed not in candidate.parents or candidate.suffix!='.onnx' or not candidate.is_file():
        raise ValueError('Candidate must be a saved model inside this project')
    accepted,reason=candidate_verdict(evaluation,baseline,str(candidate),vehicle_ordinal)
    if not accepted:return False
    write_json(root/'out/pending-model-activation.json',dict(candidate=str(candidate),baseline=str(baseline),
        sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),time=time.time(),reason=reason,state='pending',
        vehicle_ordinal=vehicle_ordinal))
    return True


def service(root,heartbeat,session,now):
    root=Path(root).resolve();out=root/'out';config_file=root/'configs/forza_horizon6.json'
    state=read_json(out/'model-activation.json',{})
    control=read_json(out/'control.json',{})
    def reload_driver():
        if heartbeat.get('model_hot_reload'):
            return
        run=Path(heartbeat.get('run','')).resolve()
        if run.parent==out and control.get('action')=='drive':
            (run/'stop').write_text('Evaluated model change; persistent controller retained')
    def publish():
        write_json(out/'model-activation.json',state)
    if state.get('state') in ('trial','active') and state.get('vehicle_ordinal') is not None:
        # Scope survives the trial period. Never leave a WRX-only model on a
        # different car, including after the ordinary ten-minute trial window.
        car=heartbeat.get('car_ordinal')
        if car is not None and car!=state['vehicle_ordinal']:
            config=read_json(config_file,{})
            if config.get('models',{}).get('pathnet')==state['candidate']:
                config['models']['pathnet']=state['previous_model']
                write_json(config_file,config);reload_driver()
            state.update(state='rolled_back',finished_at=now,reason='vehicle_scope_changed')
            publish();return
    evaluation=read_json(out/'live-evaluation-state.json',{})
    evaluation_active=(evaluation.get('candidate')==state.get('candidate') and evaluation.get('phase')=='evaluating')
    if state.get('state')=='trial' or (state.get('state')=='active' and (now-state.get('started_at',0)<600 or evaluation_active)):
        fresh=0<=now-heartbeat.get('time',0)<3.
        if state.get('state')=='trial' and heartbeat.get('model_hot_reload'):
            loaded=heartbeat.get('path_model')==state['candidate']
            if not loaded:
                if now-state['started_at']>60.:
                    config=read_json(config_file,{})
                    if config.get('models',{}).get('pathnet')==state['candidate']:
                        config['models']['pathnet']=state['previous_model']
                        write_json(config_file,config)
                    state.update(state='rolled_back',finished_at=now,reason='model_load_timeout')
                else:
                    state['load_state']='waiting_for_driver'
                publish()
                return
            if 'loaded_at' not in state:
                state['loaded_at']=now
                state['last_position']=heartbeat.get('position')
                state['distance_m']=0.
            state['load_state']='loaded'
        if fresh:
            point=heartbeat.get('position')
            previous=state.get('last_position')
            if point and previous:
                distance=math.dist(point,previous)
                if distance<80:state['distance_m']+=distance
            if point:state['last_position']=point
            state['offroad_observations']+=int((heartbeat.get('surface') or {}).get('offroad_confirmed',False))
        incident=session.get('last_incident') or {}
        try:incident_time=int(incident.get('id',0))/1000.
        except (TypeError,ValueError):incident_time=0.
        recent_failure=(incident_time>state['started_at'] and incident.get('cause') in ('confirmed_offroad','static_contact','inferred_path_contact'))
        recovery_delta=session.get('recoveries',0)-state['initial_recoveries']
        infrastructure_restart=(incident_time>state['started_at'] and
                                incident.get('cause') in ('input_stale','telemetry_stale'))
        state['last_incident_cause']=incident.get('cause')
        state['infrastructure_restart_excluded']=infrastructure_restart
        failed=state['offroad_observations']>=3 or (recovery_delta>=2 and not infrastructure_restart) or recent_failure
        if failed:
            config=read_json(config_file,{})
            if model_path(root,config.get('models',{}).get('pathnet','models/pathnet.onnx'))==Path(state['candidate']):
                config.setdefault('models',{})['pathnet']=state['previous_model']
                write_json(config_file,config);reload_driver()
            state.update(state='rolled_back',finished_at=now,reason='live_offroad_or_repeated_recovery')
        elif now-state.get('loaded_at',state['started_at'])>=120 and state['distance_m']>=200:
            state.update(state='active',finished_at=now,reason='offline_checks_and_live_continuation_passed')
        publish()
        return
    pending_file=out/'pending-model-activation.json'
    pending=read_json(pending_file,{})
    if pending.get('state')!='pending' or control.get('action')!='drive':return
    if pending.get('vehicle_ordinal') is not None and heartbeat.get('car_ordinal')!=pending['vehicle_ordinal']:return
    if not heartbeat.get('armed') or now-heartbeat.get('time',0)>3 or heartbeat.get('recovery_phase'):return
    # A stopped/known-failed approach is already compromised under the baseline.
    # Start an A/B trial during ordinary forward driving, not during its recovery.
    if 'speed_kmh' in heartbeat and heartbeat['speed_kmh']<8.:return
    if heartbeat.get('mode') in ('lost','known_failed_approach','offroad_surface','sidewalk_boundary','traffic_stop'):return
    surface=heartbeat.get('surface') or {}
    if surface.get('offroad_confirmed') or surface.get('path_enters_sidewalk'):return
    if (heartbeat.get('failure_memory') or {}).get('action')=='known_failed_approach':return
    candidate=model_path(root,pending['candidate'])
    if (root/'models/candidates').resolve() not in candidate.parents:return
    if hashlib.sha256(candidate.read_bytes()).hexdigest()!=pending['sha256']:return
    config=read_json(config_file,{})
    previous=config.get('models',{}).get('pathnet','models/pathnet.onnx')
    if model_path(root,previous)==candidate:
        pending['state']='already_active';write_json(pending_file,pending);return
    config.setdefault('models',{})['pathnet']=str(candidate)
    write_json(config_file,config)
    state=dict(state='trial',candidate=str(candidate),previous_model=previous,started_at=now,
               distance_m=0.,last_position=heartbeat.get('position'),offroad_observations=0,
               initial_recoveries=session.get('recoveries',0),reason='matched_offline_validation_passed',
               vehicle_ordinal=pending.get('vehicle_ordinal'))
    publish();pending.update(state='activated',activated_at=now);write_json(pending_file,pending)
    reload_driver()
