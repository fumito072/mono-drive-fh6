"""Causal video-to-steering learning. Data Out labels are not hardware inputs."""
from __future__ import annotations
import hashlib
import numpy as np

HISTORY = 9
FUTURE = 3
FEATURES = 256


def recording_split(recording):
    # Entire recordings are excluded from temporal-head training for this test.
    return 'test' if int(hashlib.sha256(recording.encode()).hexdigest()[:8],16)%7==0 else None


def window_indices(times, speed, steer, ages, history=HISTORY, future=FUTURE):
    times,speed,steer,ages=(np.asarray(v) for v in (times,speed,steer,ages))
    n=len(times)
    if not all(len(v)==n for v in (speed,steer,ages)):raise ValueError('Unaligned telemetry')
    valid=np.isfinite(times)&np.isfinite(speed)&np.isfinite(steer)&np.isfinite(ages)
    valid&=(ages>=0)&(ages<=80)&(speed>=8)&(speed<=140)&(np.abs(steer)<=1)
    result=[]
    for i in range(history,n-future):
        lo=i-history;hi=i+future+1
        dt=np.diff(times[lo:hi])
        if valid[lo:hi].all() and np.all((dt>=.12)&(dt<=.31)):
            result.append(i)
    return np.asarray(result,dtype=np.int32)


def targets(steer, index):
    """Mean Data Out steer in the next 0.2–0.6 s, and the previous target."""
    steer=np.asarray(steer)
    return np.asarray([np.mean(steer[index:index+FUTURE]),
                       np.mean(steer[index+1:index+FUTURE+1])],np.float32)


def sequence_arrays(features,speed,steer,index):
    """Inputs end at t; controls in the input end one step before t."""
    begin=index-HISTORY+1
    if begin<1 or index+FUTURE>=len(steer):raise ValueError('Incomplete temporal window')
    x=np.asarray(features[begin:index+1],np.float32)
    ego=np.stack([np.asarray(speed[begin:index+1])/100.,steer[begin-1:index]],axis=1).astype(np.float32)
    return x,ego,targets(steer,index)


def build_models(mean=None,std=None):
    import torch
    from torch import nn
    mean=np.zeros(FEATURES,np.float32) if mean is None else np.asarray(mean,np.float32)
    std=np.ones(FEATURES,np.float32) if std is None else np.asarray(std,np.float32)
    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('mean',torch.from_numpy(mean.copy()))
            self.register_buffer('scale',torch.from_numpy(np.maximum(std,.05).copy()))
        def inputs(self,features,ego):
            return torch.cat(((features-self.mean)/self.scale,ego),dim=-1)
    class Temporal(Base):
        def __init__(self):
            super().__init__()
            self.project=nn.Sequential(nn.Linear(FEATURES+2,96),nn.Tanh())
            self.memory=nn.GRU(96,64,batch_first=True)
            self.head=nn.Sequential(nn.Linear(64+96,64),nn.Tanh(),nn.Linear(64,1),nn.Tanh())
        def forward(self,features,ego):
            current=self.project(self.inputs(features,ego))
            sequence,_=self.memory(current)
            # Keep current visual cues accessible instead of forcing every
            # detail through the recurrent state. Memory adds motion context.
            return self.head(torch.cat((sequence,current),dim=-1)).squeeze(-1)
    class SingleFrame(Base):
        def __init__(self):
            super().__init__()
            # Similar parameter count to GRU; the comparison is not merely a
            # small network against a much larger network.
            self.head=nn.Sequential(nn.Linear(FEATURES+2,180),nn.Tanh(),nn.Linear(180,112),nn.Tanh(),nn.Linear(112,1),nn.Tanh())
        def forward(self,features,ego):
            return self.head(self.inputs(features,ego)).squeeze(-1)
    return Temporal(),SingleFrame()


def metrics(prediction,target,previous,segments):
    prediction,target,previous=(np.asarray(v,dtype=float) for v in (prediction,target,previous))
    active=np.abs(target)>.04
    changes=np.r_[False,(np.asarray(segments)[1:]==np.asarray(segments)[:-1])]
    delta_error=np.diff(prediction)-np.diff(target)
    turns={}
    for name,mask in [('negative',target<-.04),('positive',target>.04)]:
        turns[name]={'samples':int(mask.sum()),'mae':float(np.mean(np.abs(prediction[mask]-target[mask]))) if mask.any() else None}
    return {'samples':len(target),'mae':float(np.mean(np.abs(prediction-target))),
            'active_samples':int(active.sum()),
            'active_mae':float(np.mean(np.abs(prediction[active]-target[active]))) if active.any() else None,
            'bias':float(np.mean(prediction-target)),
            'delta_mae':float(np.mean(np.abs(delta_error[changes[1:]]))) if changes[1:].any() else None,
            'prediction_step_mae':float(np.mean(np.abs(np.diff(prediction)[changes[1:]]))) if changes[1:].any() else None,
            'hold_mae':float(np.mean(np.abs(previous-target))),'turns':turns}


def accept(temporal,baseline):
    import math
    for values in (temporal,baseline):
        if any(not isinstance(values.get(k),(float,int)) or not math.isfinite(values[k])
               for k in ('mae','active_mae','bias','delta_mae')):return False
    for direction,row in temporal.get('turns',{}).items():
        other=baseline.get('turns',{}).get(direction,{})
        if row.get('samples',0)>=30 and (other.get('mae') is None or row['mae']>other['mae']*1.1):return False
    return (temporal['samples']>=500 and temporal['active_samples']>=100 and
            temporal['mae']<=baseline['mae']*.98 and
            temporal['active_mae']<=baseline['active_mae'] and abs(temporal['bias'])<.05 and
            temporal['delta_mae']<=baseline['delta_mae']*1.05)
