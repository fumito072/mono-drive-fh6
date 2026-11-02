"""Independent short-horizon vehicle-envelope guard for measured lane boundaries."""
from dataclasses import replace
import math
import numpy as np
from .forza_vehicle import steering_angle


class LaneDepartureGuard:
    def __init__(self, width=1.85, length=4.41, wheelbase=2.64, margin=.35):
        self.half_width = width/2
        self.half_length = length/2
        self.wheelbase = wheelbase
        self.margin = margin

    def apply(self, plan, speed):
        if not plan.lane_edges:
            return replace(plan,guard_state='boundary_unknown')
        edges = np.asarray(plan.lane_edges,dtype=float)
        if (edges.ndim != 2 or edges.shape[1] != 3 or len(edges) < 2 or
                not np.isfinite(edges).all() or np.any(np.diff(edges[:,0]) <= 0) or
                np.any(edges[:,1] >= edges[:,2])):
            return replace(plan,guard_state='invalid_boundary')
        # The predictive controller already simulates measured wheel angle,
        # actuator lag and every vehicle corner along a two-second horizon.
        # Replacing that safe trajectory with an instantaneous constant-wheel
        # arc can reverse its correction and repeatedly force a 12 km/h limit.
        measured_path=(not plan.inferred or (plan.verified_route and plan.reason=='route_memory'))
        if (plan.guard_state=='predictive' and plan.quality>=.75 and measured_path
                and plan.guard_margin_m is not None and math.isfinite(plan.guard_margin_m)
                and plan.guard_margin_m>=self.margin):
            return replace(plan,guard_state='predictive_clear')
        end = min(float(edges[-1,0]),max(float(edges[0,0])+2.,6.,4.+max(speed,0.)/3.6*.6))
        if end <= edges[0,0]:
            return replace(plan,guard_state='short_boundary')
        z = np.linspace(max(3.,float(edges[0,0])),end,9)
        left = np.interp(z,edges[:,0],edges[:,1])
        right = np.interp(z,edges[:,0],edges[:,2])
        # The path was centred on the lane before its safety envelope was
        # clipped by walls. Do not shift the target to the clipped midpoint.
        centre=np.interp(z,plan.path[:,0],plan.path[:,1])
        choices = np.unique(np.r_[np.linspace(-.65,.65,261),np.clip(plan.steer,-.65,.65)])
        best = None
        centering = None
        best_effort = None
        narrow_straight=(float(np.median(right-left))<2*(self.half_width+self.margin) and
                         float(np.max(np.abs(centre[:3])))<.35 and
                         float(np.max(np.abs(np.diff(centre)/np.diff(z))))<.04)
        aligned=(float(np.max(np.abs(centre[:3])))<.8 and
                 float(np.max(np.abs(np.diff(centre)/np.diff(z))))<.12)
        # Evaluate every candidate together. The former Python loop held the
        # GIL long enough to interrupt the independent analog output thread.
        # Candidate order, footprint geometry and selection costs are unchanged.
        curvature=np.tan(choices*steering_angle(speed))/self.wheelbase
        sine=curvature[:,None]*z
        valid=np.all(np.abs(sine)<=.95,axis=1)
        choices,curvature,sine=choices[valid],curvature[valid],sine[valid]
        if len(choices):
            angle=np.arcsin(sine)
            x=np.empty_like(angle)
            straight=np.abs(curvature)<1e-5
            x[straight]=.5*curvature[straight,None]*z*z
            x[~straight]=(1-np.cos(angle[~straight]))/curvature[~straight,None]
            extent=self.half_width*np.cos(angle)+self.half_length*np.abs(np.sin(angle))
            clearance=np.min(np.minimum(x-extent-left,right-x-extent),axis=1)
            effort=clearance-.01*np.abs(choices-plan.steer)
            i=int(np.argmax(effort))
            best_effort=float(effort[i]),float(choices[i]),float(clearance[i])
            scores=(choices-plan.steer)**2+.20*np.mean((x-centre)**2,axis=1)
            soft=(clearance>=.12)&(narrow_straight or (aligned and plan.quality>=.8))
            if np.any(soft):
                soft_scores=np.where(soft,scores+4.*np.maximum(0.,self.margin-clearance)**2,np.inf)
                i=int(np.argmin(soft_scores))
                centering=float(soft_scores[i]),float(choices[i]),float(clearance[i])
            safe=clearance>=self.margin
            if np.any(safe):
                i=int(np.argmin(np.where(safe,scores,np.inf)))
                best=float(scores[i]),float(choices[i]),float(clearance[i])
        if best is None:
            if centering is not None:
                _,steer,clearance=centering
                # A few centimetres around the preferred .35 m reserve used
                # to switch abruptly between cruise and 12 km/h. Keep the
                # swept body inside the measured lane and vary the limit with
                # its reserve while recentering. Large offsets/turns stay on
                # the original low-speed recovery branch.
                fraction=max(0.,min(1.,(clearance-.12)/.18))
                limit=plan.limit if narrow_straight else min(plan.limit,12.+48.*fraction)
                return replace(plan,steer=steer,reason='lane_guard',guard_state='centering',
                               guard_margin_m=clearance,limit=limit,
                               road_limit=min(plan.road_limit or plan.limit,limit))
            if best_effort is None:
                return replace(plan,guard_state='invalid_boundary')
            _,steer,clearance = best_effort
            return replace(plan,steer=steer,reason='lane_guard',guard_state='recentering',guard_margin_m=clearance,
                           limit=min(plan.limit,0. if clearance<-.25 else 12.),
                           road_limit=min(plan.road_limit or 50.,0. if clearance<-.25 else 12.))
        _,steer,clearance = best
        corrected = abs(steer-plan.steer) > .02
        limited = clearance < .30
        return replace(plan,steer=steer,
                       reason='lane_guard' if corrected else plan.reason,
                       guard_state='correcting' if corrected else 'near_boundary' if limited else 'clear',
                       guard_margin_m=clearance)
