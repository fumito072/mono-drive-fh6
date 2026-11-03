"""Causal steering preview, initially shadow-only and bounded by lane geometry."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json,math
from pathlib import Path
import time
import numpy as np
from .camera import Camera,REF_CAMERA,ReferenceView
from .pathnet import net_input
from .forza_local_learning import read_json,write_json
from .forza_vehicle import steering_angle
from .temporal_steering import resample_history


def clearance(plan,speed,steer):
    try:
        edges=np.asarray(plan.lane_edges,float)
        if (edges.ndim!=2 or edges.shape[1]!=3 or len(edges)<3 or not np.isfinite(edges).all()
            or np.any(np.diff(edges[:,0])<=0) or np.any(edges[:,1]>=edges[:,2])):return -math.inf
        z=np.linspace(max(3.,edges[0,0]),min(edges[-1,0],max(6.,speed/3.6*.6)),9)
        k=math.tan(float(steer)*steering_angle(speed))/2.64
        sine=k*z
        if np.max(np.abs(sine))>=.9:return -math.inf
        angle=np.arcsin(sine)
        x=.5*k*z*z if abs(k)<1e-5 else (1-np.cos(angle))/k
        width=.925*np.cos(angle)+2.205*np.abs(np.sin(angle))
        return float(np.min(np.minimum(x-width-np.interp(z,edges[:,0],edges[:,1]),
                                       np.interp(z,edges[:,0],edges[:,2])-x-width)))
    except (ValueError,TypeError,IndexError):return -math.inf


def bounded_preview(plan,prediction,speed,road=None):
    allowed={'paint_pair','paint_corridor','road_candidates','lane_guard','road_bounds_only'}
    if (plan is None or not all(math.isfinite(v) for v in (plan.steer,plan.limit,plan.quality,prediction,speed))
        or plan.reason not in allowed or plan.inferred or plan.quality<.8 or
        not 20<=speed<=75 or plan.limit<15 or not math.isfinite(prediction)):
        return plan,'ineligible'
    if abs(prediction-plan.steer)>.12:return plan,'disagreement'
    adjustment=float(np.clip(.18*(prediction-plan.steer),-.025,.025))
    if abs(adjustment)<.002:return plan,'no_change'
    desired=float(np.clip(plan.steer+adjustment,-.65,.65))
    before=clearance(plan,speed,plan.steer);after=clearance(plan,speed,desired)
    if after<max(.45,before-.08):return plan,'lane_veto'
    if road is not None:
        from .forza_scene import footprint_quality
        z=np.linspace(3.,min(15.,max(6.,speed/3.6)),12)
        k=math.tan(desired*steering_angle(speed))/2.64
        x=.5*k*z*z
        floor,mean=footprint_quality(road,np.column_stack((z,x)))
        if floor<.5 or mean<.75:return plan,'road_veto'
    return replace(plan,steer=desired),'applied'


def load_models(folder):
    from .runtime import make_session
    report=read_json(folder/'report.json',{})
    if report.get('phase')!='complete':raise ValueError('Temporal training incomplete')
    history_frames=int(report.get('history_frames',9))
    if not 9<=history_frames<=41 or report.get('sample_hz',5)!=5:raise ValueError('Unsupported temporal history')
    encoder=make_session(str(folder/'encoder.onnx'),'cpu',threads=2,gpu_ok=False)
    head=make_session(str(folder/'steering.onnx'),'cpu',threads=2,gpu_ok=False)
    f=encoder.run(None,{'image':np.zeros((1,3,128,320),np.float32)})[0]
    y=head.run(None,{'features':np.zeros((1,history_frames,256),np.float32),'ego':np.zeros((1,history_frames,2),np.float32)})[0]
    if f.shape!=(1,256) or not np.isfinite(f).all() or not np.isfinite(y).all():raise ValueError('Invalid temporal model output')
    return encoder,head,report


class TemporalSteering:
    def __init__(self,root=None):
        self.root=Path(root) if root else Path(__file__).resolve().parents[1]
        self.config=self.root/'configs/temporal_steering.json'
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='temporal-loader')
        self.future=None;self.models=None;self.next_check=0.;self.next_frame=0.;self.rows=deque(maxlen=96)
        self.view=None;self.previous=None;self.last_position=None;self.last_yaw=None
        self.last_prediction=None
        self.next_write=0.;self.counters={'eligible':0,'agreeing':0,'applied':0}
        self.status={'phase':'waiting_for_model'};self.mode='shadow';self.folder=None;self.loading=None;self.enabled=False
        self.counted_folder=None
        previous=read_json(self.root/'out/temporal-steering-state.json',{})
        config=read_json(self.config,{})
        selected=(self.root/config.get('model_dir','models/temporal-20261103')).resolve()
        if previous.get('model_dir')==str(selected):
            self.counted_folder=selected
            self.counters={k:max(0,int(previous.get(k,0))) for k in self.counters}
    def __del__(self):
        if hasattr(self,'pool'):self.pool.shutdown(wait=False,cancel_futures=True)
    def publish_status(self,now,phase,**details):
        if now>=self.next_write:
            self.next_write=now+.5
            state=dict(phase=phase,time=time.time(),**self.counters)
            if self.folder is not None or self.counted_folder is not None:state['model_dir']=str(self.folder or self.counted_folder)
            if phase in ('load_failed','observation_failed') and 'error' in self.status:state['error']=self.status['error']
            state.update(details);self.status=state
            write_json(self.root/'out/temporal-steering-state.json',self.status)
    def update(self,frame,telemetry,plan,scene,now):
        if now>=self.next_check:
            self.next_check=now+2.;config=read_json(self.config,{})
            self.enabled=config.get('enabled') is True
            self.mode=config.get('mode','shadow')
            requested=(self.root/config.get('model_dir','models/temporal-20261103')).resolve()
            if (self.enabled and (self.root/'models').resolve() in requested.parents and
                (requested/'report.json').exists() and requested!=self.folder and self.future is None):
                self.loading=requested;self.future=self.pool.submit(load_models,requested)
        if self.future is not None and self.future.done():
            try:
                self.models=self.future.result();self.folder=self.loading;self.rows.clear()
                self.last_prediction=None
                if self.counted_folder!=self.folder:self.counters={'eligible':0,'agreeing':0,'applied':0}
                self.counted_folder=self.folder;self.status={'phase':'loaded'}
                previous=read_json(self.root/'out/temporal-steering-state.json',{})
                if previous.get('model_dir')==str(self.folder):
                    self.counters={k:max(self.counters[k],int(previous.get(k,0))) for k in self.counters}
            except Exception as e:self.status={'phase':'load_failed','error':str(e)}
            self.future=None;self.loading=None
        if not self.enabled:
            self.rows.clear();self.last_prediction=None;return plan
        if self.models is None:
            self.publish_status(now,self.status.get('phase','waiting_for_model'))
            return plan
        if self.mode not in ('shadow','trial','active'):self.rows.clear();return plan
        if telemetry.get('car_ordinal')!=3819 or telemetry.get('gear',0)<=0 or telemetry.get('race_position',0):
            self.rows.clear();self.last_prediction=None
            self.publish_status(now,'waiting_for_forward_driving',history_frames=0)
            return plan
        if now<self.next_frame:
            # Hold the preview across intermediate controller frames; applying
            # it on only the 5 Hz inference frame would create artificial pulses.
            if (self.mode in ('trial','active') and self.models[2].get('accepted_offline') is True
                and self.last_prediction is not None and 0<=now-self.last_prediction[0]<=.3):
                proposed,_=bounded_preview(plan,self.last_prediction[1],float(telemetry['speed']),scene.get('raw_road',scene.get('road')))
                return proposed
            return plan
        self.next_frame=now+.19
        try:
            observed=float(telemetry.get('frame_capture_time',now))
            if not math.isfinite(observed) or not 0<=now-observed<=.5:
                self.rows.clear();self.last_prediction=None
                self.publish_status(now,'waiting_for_fresh_frame');return plan
            if self.rows and observed<=self.rows[-1][0]:return plan
            position=np.asarray(telemetry['position'],dtype=float)
            yaw=float(telemetry['yaw']);speed=float(telemetry['speed'])
            if (not np.isfinite(position).all() or not math.isfinite(yaw) or not 8<=speed<=140):
                self.rows.clear();self.last_prediction=None;self.last_position=None
                self.publish_status(now,'waiting_for_motion',speed_kmh=speed,history_frames=0)
                return plan
            if self.last_position is not None:
                turn=math.atan2(math.sin(yaw-self.last_yaw),math.cos(yaw-self.last_yaw))
                if np.linalg.norm(position-self.last_position)>max(8.,speed/3.6*.6) or abs(turn)>.4:
                    self.rows.clear();self.last_prediction=None
            self.last_position=position;self.last_yaw=yaw
            cfg=read_json(self.root/'configs/forza_horizon6.json',{})
            if self.view is None or (self.view.cam.width,self.view.cam.height)!=(frame.shape[1],frame.shape[0]):
                self.view=ReferenceView(Camera.from_dict(cfg['camera']),REF_CAMERA,frame_size=(frame.shape[1],frame.shape[0]));self.rows.clear()
            encoder,head,report=self.models
            feature=encoder.run(None,{'image':net_input(self.view.warp(frame))[None]})[0][0]
            steer=float(np.clip(float(telemetry.get('steer',0))/127.,-1.,1.))
            if self.rows and observed-self.rows[-1][0]>.5:self.rows.clear()
            self.rows.append((observed,feature,speed,steer))
            history=resample_history(list(self.rows),history=int(report.get('history_frames',9)))
            if history is None:
                self.publish_status(now,'collecting_history',history_frames=len(self.rows),model_dir=str(self.folder))
                return plan
            features,ego,timing=history;features=features[None];ego=ego[None]
            predicted=float(head.run(None,{'features':features,'ego':ego})[0].reshape(-1)[0])
            resampling_valid=timing['max_observation_gap_s']<=.31 or report.get('history_resampling_checked') is True
            self.last_prediction=(now,predicted) if resampling_valid else None
            hypothetical,reason=bounded_preview(plan,predicted,speed,scene.get('raw_road',scene.get('road')))
            eligible=reason in ('applied','no_change','disagreement','lane_veto','road_veto')
            self.counters['eligible']+=int(eligible)
            self.counters['agreeing']+=int(eligible and abs(predicted-plan.steer)<=.12)
            applied=(self.mode in ('trial','active') and report.get('accepted_offline') is True and resampling_valid and reason=='applied')
            if applied:self.counters['applied']+=1
            self.status={'phase':self.mode,'time':time.time(),'predicted_steer':predicted,'baseline_steer':plan.steer,
                         'observation_age_s':now-observed,
                         'model_dir':str(self.folder),
                         'proposed_steer':hypothetical.steer,'applied':applied,'reason':reason,
                         'accepted_offline':bool(report.get('accepted_offline')),**timing,
                         'resampling_validated':bool(resampling_valid),
                         'speed_kmh':speed,**self.counters}
            if now>=self.next_write:
                self.next_write=now+.5;write_json(self.root/'out/temporal-steering-state.json',self.status)
                with (self.root/'out/temporal-steering-observations.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(self.status)+'\n')
            return hypothetical if applied else plan
        except Exception as e:
            self.rows.clear();self.last_prediction=None;self.status={'phase':'observation_failed','error':str(e)}
            self.publish_status(now,'observation_failed')
            return plan
