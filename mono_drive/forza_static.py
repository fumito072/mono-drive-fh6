"""Independent swept-body checks against segmented fixed obstacles.

Research-inspired discrete safety filter; not a reproduction of MPC/CBF proofs.
"""
from dataclasses import dataclass,replace
import math
from .forza_speed import MAX_CRUISE_KMH
import cv2
import numpy as np
from .forza_vehicle import steering_angle


@dataclass
class StaticRisk:
    clearance_m: float = 50.
    limit: float = MAX_CRUISE_KMH
    emergency: bool = False
    brake: float = 0.
    immediate: bool = False


def static_risk(scene,speed,steer,path=None):
    if path is not None:
        path=np.asarray(path,dtype=float)
        if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or not np.isfinite(path).all() or np.any(np.diff(path[:,0])<=0):
            path=None
    obstacles=scene.get('obstacles')
    if obstacles is None:
        return StaticRisk()
    road=scene['road']
    occupied=cv2.dilate(obstacles,np.ones((3,3),np.uint8))
    z=np.arange(4.,31. if path is None else min(31.,float(path[-1,0])+1.),1.)
    if len(z)<2:
        return StaticRisk()
    if path is None:
        curvature=math.tan(float(steer)*steering_angle(speed))/2.64
        sine=curvature*z
        valid=np.abs(sine)<.94
        z,sine=z[valid],sine[valid]
        if not len(z):
            return StaticRisk()
        angle=np.arcsin(sine)
        x=.5*curvature*z*z if abs(curvature)<1e-5 else (1-np.cos(angle))/curvature
    else:
        from .forza_traffic import path_x
        x=path_x(path,z)
        angle=np.arctan(np.gradient(x,z))
    offsets=np.array([-.96,-.55,0.,.55,.96])
    xx=x[:,None]+np.cos(angle[:,None])*offsets
    zz=z[:,None]-np.sin(angle[:,None])*offsets
    from .forza_projection import project
    u,v=project(xx,zz)
    px=np.clip(np.rint(u*383).astype(int),0,383)
    py=np.clip(np.rint(v*215).astype(int),0,215)
    from .forza_projection import parameters
    visible=(u>0)&(u<1)&(v<.60)&(v>parameters()['horizon']+.005)
    contacts=np.any(visible&(occupied[py,px]>185)&(road[py,px]<130),axis=1)
    sustained=contacts[:-1]&contacts[1:]
    near=None
    if path is not None:
        current=static_risk(scene,speed,steer)
        if current.emergency and current.clearance_m<=max(5.,speed/3.6*.75):
            near=current
    if not np.any(sustained):
        return near or StaticRisk()
    distance=float(z[np.flatnonzero(sustained)[0]])
    limit=min(MAX_CRUISE_KMH,3.6*math.sqrt(max(0.,4.*(distance-3.))))
    velocity=max(0.,speed/3.6)
    usable=max(.4,distance-2.-.35*velocity)
    required=velocity*velocity/(2*usable)
    emergency=required>=4.5 or distance<2.+.5*velocity
    brake=min(.8,required/10.) if speed>limit+1. else 0.
    if emergency:
        brake=max(brake,.55)
    stop_distance=2.+velocity*.45+velocity*velocity/12.
    result=StaticRisk(distance,limit,emergency,brake,distance<max(5.,stop_distance))
    return near if near is not None and near.brake>result.brake else result


def avoid_static(plan,scene,speed):
    # Only lateral steering waits for confirmation; the independent immediate
    # brake check still reads the current ground-contact evidence.
    if 'confirmed_obstacles' in scene:
        confirmed=scene.get('confirmed_obstacles')
        if confirmed is None:
            return plan
        scene=dict(scene,obstacles=confirmed)
    risk=static_risk(scene,speed,plan.steer,path=plan.path)
    if risk.clearance_m>=30:
        return plan
    from .forza_scene import inside_lane,project
    choices=[]
    if plan.lane_edges:
        edges=np.asarray(plan.lane_edges)
        for steer in np.clip(plan.steer+np.array([-.18,-.12,-.06,.06,.12,.18]),-.65,.65):
            z=plan.path[:,0]
            k=math.tan(float(steer)*steering_angle(speed))/2.64
            path=np.column_stack((z,.5*k*z*z))
            left=np.interp(z,edges[:,0],edges[:,1])
            right=np.interp(z,edges[:,0],edges[:,2])
            if not inside_lane(path,left,right):
                continue
            alternative=static_risk(scene,speed,float(steer),path=path)
            if alternative.clearance_m>risk.clearance_m+3:
                choices.append((abs(steer-plan.steer),float(steer),path,alternative))
    if choices:
        _,steer,path,risk=min(choices,key=lambda x:x[0])
        plan=replace(plan,steer=steer,path=path,reason='static_avoidance',
                     image_path=tuple(project(float(x),float(z)) for z,x in path))
    return replace(plan,limit=min(plan.limit,risk.limit),
                   reason='static_stop' if risk.emergency else plan.reason)
