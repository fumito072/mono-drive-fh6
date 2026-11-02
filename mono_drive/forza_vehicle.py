"""Effective steering response measured from FH6 yaw, speed and input logs.

This is a game response model, not a specification of the real car's axle.
"""
import math
import json
import time
from collections import deque
from pathlib import Path
import numpy as np

SPEEDS=[0.,15.,25.,35.,45.,60.]
BOOTSTRAP=[40.,40.,36.5,20.,14.,12.]
_degrees=list(BOOTSTRAP)


def steering_angle(speed_kmh):
    degrees=float(np.interp(max(0.,float(speed_kmh)),
        SPEEDS,_degrees))
    return math.radians(degrees)


class VehicleResponse:
    """Learn the game's effective steering response separately for each car ID."""
    def __init__(self,root=None):
        self.root=Path(root) if root else Path(__file__).resolve().parents[1]
        self.ordinal=None
        self.previous=None
        self.samples=[deque(maxlen=160) for _ in SPEEDS]
        self.counts=[0]*len(SPEEDS)
        self.saved=0.
        self.name='unknown'

    def update(self,ego,now):
        global _degrees
        ordinal=ego.get('car_ordinal')
        if not ordinal:return
        if ordinal!=self.ordinal:
            self.ordinal=ordinal;self.previous=None
            self.samples=[deque(maxlen=160) for _ in SPEEDS]
            self.counts=[0]*len(SPEEDS);_degrees=list(BOOTSTRAP)
            from .forza_local_learning import read_json
            active=read_json(self.root/'configs/active_vehicle.json',{})
            self.name=active.get('name','unknown') if active.get('car_ordinal')==ordinal else f'car_{ordinal}'
            saved=read_json(self.root/f'models/vehicle_profiles/car_{ordinal}.json',{})
            values=saved.get('effective_steering_degrees',[])
            if len(values)==len(SPEEDS) and all(isinstance(v,(int,float)) and 6<=v<=70 for v in values):
                _degrees=list(values)
                self.counts=saved.get('sample_counts',self.counts)
        previous=self.previous
        self.previous=(dict(ego),now)
        if previous:
            old,t=previous;dt=now-t
            speed=(ego['speed']+old['speed'])/2.
            a,b=old.get('steer',0)/127.,ego.get('steer',0)/127.
            yaw=math.atan2(math.sin(ego['yaw']-old['yaw']),math.cos(ego['yaw']-old['yaw']))
            distance=speed/3.6*dt
            # Reject standstill, abrupt wheel motion, reversals and collisions.
            if (.04<dt<.7 and 10.<speed<60. and .04<abs(a)<.55 and a*b>0 and
                abs(a-b)<.04 and abs(ego['speed']-old['speed'])<2.5 and
                distance>.12 and abs(yaw)<.2 and ego.get('forward_kmh',0)>5):
                angle=math.degrees(math.atan(2.64*yaw/distance)/((a+b)/2.))
                if 6.<angle<70.:
                    index=int(np.argmin(np.abs(np.asarray(SPEEDS)-speed)))
                    self.samples[index].append(angle);self.counts[index]+=1
                    if len(self.samples[index])>=8:
                        target=float(np.median(self.samples[index]))
                        target=float(np.clip(target,BOOTSTRAP[index]*.45,BOOTSTRAP[index]*1.8))
                        _degrees[index]+=.02*(target-_degrees[index])
        if now-self.saved>3.:
            self.saved=now
            from .forza_local_learning import write_json
            folder=self.root/'models/vehicle_profiles';folder.mkdir(parents=True,exist_ok=True)
            data=self.status()
            write_json(folder/f'car_{self.ordinal}.json',data)
            write_json(self.root/'out/vehicle-state.json',data)

    def status(self):
        return dict(time=time.time(),car_ordinal=self.ordinal,name=self.name,
                    effective_steering_degrees=list(_degrees),speed_bins_kmh=SPEEDS,
                    sample_counts=self.counts,measured_samples=sum(self.counts),
                    model='effective game response; 2.64m reference model, not measured axle geometry',
                    bootstrap_unverified=sum(self.counts)<8)
