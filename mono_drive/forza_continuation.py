"""Continue a game drive through missing lane observations using motion priors.

These paths are explicitly inferred, never ground-truth training labels. Fresh
camera/telemetry and independent obstacle/median checks still apply downstream.
"""
from dataclasses import replace
import math
from .forza_speed import MAX_CRUISE_KMH
import numpy as np

from .forza_scene import ScenePlan, project, road_value
from .forza_lane_memory import align_plan
from .forza_vehicle import steering_angle


def extend_path(path, distance=90.):
    path=np.asarray(path,dtype=float)
    if len(path)<2 or not np.isfinite(path).all() or np.any(np.diff(path[:,0])<=0):
        raise ValueError('Motion prior needs an ordered finite path')
    z,x=path.T
    slope=float(np.clip((x[-1]-x[-2])/(z[-1]-z[-2]),-.8,.8))
    curvature=0.
    if len(path)>=3:
        slopes=np.diff(x)/np.diff(z)
        curvature=float(np.clip((slopes[-1]-slopes[-2])/max(1.,(z[-1]-z[-3])*.5),-.035,.035))
    extra=np.arange(z[-1]+2.,max(distance,z[-1]+3.),2.)
    d=extra-z[-1]
    return np.vstack((path,np.column_stack((extra,x[-1]+slope*d+.5*curvature*d*d))))


class LaneContinuation:
    def __init__(self):
        self.anchor=None
        self.origin=None
        self.observed_at=None
        self.last_steer=0.
        self.last_update=None

    def observe(self,plan,ego,now):
        if plan is None or plan.quality<.75 or plan.inferred or plan.reason in ('road_reentry','visible_road_creep'):
            return
        self.anchor=replace(plan,path=extend_path(plan.path),lane_edges=())
        self.origin=dict(position=list(ego['position']),yaw=ego['yaw'])
        self.observed_at=now
        self.last_steer=plan.steer

    def predict(self,scene,ego,now,hint=None):
        age=max(0.,now-self.observed_at) if self.observed_at is not None else None
        prior=None
        # After a long interval the map/current heading should dominate the old
        # curve. A frozen route is not allowed to wind the car into circles.
        if self.anchor is not None and age<6.:
            prior=align_plan(self.anchor,self.origin,ego)
        requested=float(np.clip(ego.get('steer',0.)/127.,-.3,.3))
        if prior is not None:
            from .forza_lateral import lookahead_target
            _,_,requested=lookahead_target(prior.path,ego['speed'])
            requested=float(np.clip(requested,-.5,.5))
        map_supported=hint is not None and hint.confidence>=.55
        if map_supported:
            mapped=float(np.clip(.5*hint.angle,-.5,.5))
            weight=.35 if prior is not None and age<1.5 else .75
            requested=(1-weight)*requested+weight*mapped
        elif prior is None:
            requested*=.8
        dt=.16 if self.last_update is None else float(np.clip(now-self.last_update,.01,.5))
        steer=float(np.clip(requested,self.last_steer-.8*dt,self.last_steer+.8*dt))
        self.last_steer,self.last_update=steer,now
        z=np.array([4.,6.,8.,10.,12.,16.,22.,28.])
        k=math.tan(steer*steering_angle(ego['speed']))/2.64
        x=.5*k*z*z
        path=np.column_stack((z,x))
        road=scene.get('raw_road',scene['road'])
        support=float(np.mean([road_value(road,float(px),float(pz)) for pz,px in path[:5]]))
        limit=MAX_CRUISE_KMH if (age is not None and age<1.5) or (map_supported and support>.35) else 40. if map_supported or (age is not None and age<4.) else 25.
        # Only a real geometric curve limits this target, not missing paint.
        curve_limit=3.6*math.sqrt(2.15/max(.001,abs(k)))
        uncertainty_limited=limit<MAX_CRUISE_KMH and curve_limit>=limit
        limit=min(limit,curve_limit)
        return ScenePlan(path,steer,limit,.60,1,len(scene.get('cars',[])),
                         'lane_inferred',support,image_path=tuple(project(float(px),float(pz)) for pz,px in path),
                         road_limit=limit,inferred=True,inference_age_s=age,
                         uncertainty_limited=uncertainty_limited)


def continue_unmarked(plan,speed):
    """Missing boundaries no longer impose a stop or a 12 km/h crawl."""
    if plan.guard_state not in ('boundary_unknown','short_boundary','invalid_boundary'):
        return plan
    reason=plan.reason if plan.reason in ('traffic_stop','static_stop') else 'lane_inferred'
    return replace(plan,inferred=True,reason=reason,quality=max(.55,plan.quality))


def retain_observed_lane(inferred, observed, speed):
    """A heading fallback may not erase boundaries from the same exposure.

    Re-centre on that observed corridor and recompute its geometric speed.
    The caller must run footprint and surface checks on the returned plan.
    Truly missing boundaries retain the existing uncertain-speed fallback.
    """
    if not observed.lane_edges:
        return inferred
    from .forza_lateral import center_lane_plan
    from .forza_scene import curve_speed
    bounded=center_lane_plan(replace(inferred,lane_edges=observed.lane_edges),speed)
    geometry=curve_speed(bounded.path,speed)[0]
    original=observed.road_limit if observed.road_limit is not None else observed.limit
    limit=min(original,geometry)
    return replace(bounded,limit=limit,road_limit=limit,uncertainty_limited=False)
