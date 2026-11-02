"""Measured-steering feedback and finite-horizon lane tracking.

Uses the bicycle model and uncertainty margins; this is not a reproduction
of OF-MPC/itube-CILQR and makes no formal robust-control guarantee.
"""
from dataclasses import replace
import math
import numpy as np
from .forza_vehicle import steering_angle

WHEELBASE = 2.64
MAX_ANGLE = math.radians(35.)


class SteeringServo:
    """Close the loop around the game's reported wheel angle, not just USB output."""
    def __init__(self):
        self.trim = 0.

    def step(self, desired, measured, dt, inner=.28, outer=.98):
        desired = float(np.clip(desired,-.75,.75))
        dt = float(np.clip(dt,0.,.05))
        if abs(desired)<.008:
            self.trim *= max(0.,1.-dt*8.)
            return 0.
        error = desired-float(measured)
        if abs(error)<.015:
            error=0.
        if abs(error)<.4:
            self.trim = float(np.clip(self.trim+.18*error*dt,-.10,.10))
        base = math.copysign(inner+(outer-inner)*abs(desired),desired)
        raw = base+.35*error+self.trim
        # Never reverse a requested wheel direction because of stored bias.
        return math.copysign(float(np.clip(abs(raw),inner*.5,outer)),desired)


def extrapolate(z, values, query):
    query = np.asarray(query)
    result = np.interp(query,z,values)
    near = (values[1]-values[0])/(z[1]-z[0])
    far = (values[-1]-values[-2])/(z[-1]-z[-2])
    return np.where(query<z[0],values[0]+near*(query-z[0]),
                    np.where(query>z[-1],values[-1]+far*(query-z[-1]),result))


def lookahead_target(path,speed_kmh):
    """Aim at the measured lane centre farther ahead as speed rises."""
    z,x=np.asarray(path,dtype=float).T
    distance=float(np.clip(10.+max(0.,speed_kmh)/3.6*1.2,z[0],max(z[0],min(26.,z[-1]))))
    offset=float(np.interp(distance,z,x))
    curvature=2*offset/(distance*distance+offset*offset)
    steer=float(np.clip(math.atan(WHEELBASE*curvature)/steering_angle(speed_kmh),-.65,.65))
    return distance,offset,steer


def center_lane_plan(plan,speed_kmh):
    """Use the midpoint of the current lane, before obstacle constraints.

    Road/wall clipping is applied later and must not redefine this target.
    Explicit evasive and re-entry manoeuvres keep their own temporary path.
    """
    if not plan.lane_edges or plan.reason in ('vehicle_avoidance','static_avoidance',
            'road_reentry','visible_road_creep','intersection_turn'):
        return plan
    edges=np.asarray(plan.lane_edges,dtype=float)
    if (edges.ndim!=2 or edges.shape[1]!=3 or len(edges)<2 or
            not np.isfinite(edges).all() or np.any(np.diff(edges[:,0])<=0) or
            np.any(edges[:,1]>=edges[:,2])):
        return plan
    z,left,right=edges.T
    from .forza_scene import smooth_lane_reference,project
    middle=smooth_lane_reference(z,(left+right)/2.,left,right)
    path=np.column_stack((z,middle))
    _,_,steer=lookahead_target(path,speed_kmh)
    return replace(plan,path=path,steer=steer,
                   image_path=tuple(project(float(x),float(d)) for d,x in path))


class PredictiveLaneControl:
    """Vectorized short-horizon shooting with wheel lag and vehicle footprint."""
    def __init__(self):
        self.previous = 0.
        self.updated = None
        self.margin = None
        self.curvature = 0.
        self.target = None

    def apply(self, plan, ego, now, uncertainty=.08):
        if plan is None or len(plan.path)<3 or not plan.lane_edges:
            return plan
        edges = np.asarray(plan.lane_edges,dtype=float)
        path = np.asarray(plan.path,dtype=float)
        path = path[path[:,0]<=30.]
        if len(path)<3:
            return plan
        slopes=np.diff(path[:,1])/np.diff(path[:,0])
        jumps=np.flatnonzero(np.abs(np.diff(slopes))>.65)
        if len(jumps) and jumps[0]+2>=3:
            path=path[:jumps[0]+2]
        if not np.isfinite(edges).all() or np.any(np.diff(path[:,0])<=.05):
            return plan
        zref,xref = path.T
        distance,offset,pursuit=lookahead_target(path,ego['speed'])
        self.target=(distance,offset)
        # A quadratic keeps bend curvature; fitting an arbitrary cubic to a
        # short painted segment amplified pixel noise in the old controller.
        fit = np.polynomial.Polynomial.fit(zref,xref,2,w=1/np.sqrt(zref)).convert()
        heading0 = float(np.clip(math.atan(fit.deriv()(0.)),-.65,.65))
        lateral0 = float(np.clip(fit(0.),-3.,3.))
        velocity = max(1.2,float(ego['speed'])/3.6)
        max_angle=steering_angle(ego['speed'])
        actual = float(np.clip(ego.get('steer',0)/127.,-.8,.8))
        near_correction = (heading0+math.atan2(.65*lateral0,velocity+2.))/max_angle
        initial_ff=math.atan(WHEELBASE*fit.deriv(2)(0.)/(1+fit.deriv()(0.)**2)**1.5)/max_angle
        correction=.55*near_correction+.45*(pursuit-initial_ff)
        offsets = np.linspace(correction-.40,correction+.40,65)
        count = len(offsets)
        x,z,angle = np.zeros(count),np.zeros(count),np.zeros(count)
        wheel = np.full(count,actual)
        cost = np.zeros(count)
        minimum = np.full(count,10.)
        first = None
        dt = .10
        duration = min(2.0,max(.8,(float(zref[-1])-1.)/velocity))
        times = np.arange(dt,duration+.001,dt)
        ez,left,right = edges.T
        for t in times:
            slope = fit.deriv()(z)
            curve = fit.deriv(2)(z)/(1+slope*slope)**1.5
            ff = np.arctan(WHEELBASE*curve)/max_angle
            requested = np.clip(ff+offsets*np.exp(-t/1.25),-.65,.65)
            if first is None:
                first = requested.copy()
            # Both the actuator's slew rate and game steering lag are modeled.
            wheel += np.clip((requested-wheel)*dt/.22,-1.25*dt,1.25*dt)
            angle += velocity/WHEELBASE*np.tan(wheel*max_angle)*dt
            x += velocity*np.sin(angle)*dt
            z += velocity*np.cos(angle)*dt
            reference = fit(z)
            error = x-reference
            heading_error = angle-np.arctan(fit.deriv()(z))
            # Corners are evaluated at their own longitudinal coordinates.
            clearance = np.full(count,10.)
            for longitudinal in (-2.205,2.205):
                for lateral in (-.925,.925):
                    px = x+longitudinal*np.sin(angle)+lateral*np.cos(angle)
                    pz = z+longitudinal*np.cos(angle)-lateral*np.sin(angle)
                    local=np.minimum(px-extrapolate(ez,left,pz),extrapolate(ez,right,pz)-px)
                    # Do not extrapolate a distant, short lane fit underneath
                    # the rear of the current car and call it an observed edge.
                    local=np.where(pz>=max(0.,ez[0]-1.),local,10.)
                    clearance = np.minimum(clearance,local)
            minimum = np.minimum(minimum,clearance)
            violation = np.maximum(0.,.12+uncertainty-clearance)
            cost += (9.*error*error+5.*heading_error*heading_error+100.*violation*violation+.10*requested*requested)*dt
        cost += 3.*(x-fit(z))**2+.4*(first-self.previous)**2+.8*(first-pursuit)**2
        best = int(np.argmin(cost))
        requested = float(first[best])
        interval = .15 if self.updated is None else float(np.clip(now-self.updated,.03,.4))
        # Limit the physical requested wheel angle, not a dead-zone-shifted stick.
        requested = float(np.clip(requested,self.previous-1.2*interval,self.previous+1.2*interval))
        self.previous,self.updated = requested,now
        self.margin = float(minimum[best])
        distances = np.linspace(max(2.,zref[0]),min(28.,zref[-1]),20)
        slope = fit.deriv()(distances)
        curves = np.abs(fit.deriv(2)(distances)/(1+slope*slope)**1.5)
        self.curvature = float(np.max(curves))
        from .forza_scene import feasible_curve_speed
        preview_limit = feasible_curve_speed(curves,np.maximum(0.,distances-2.),ego['speed'])
        limit = min(plan.limit,preview_limit)
        if self.margin < -.25:
            limit = min(limit,15.)
        return replace(plan,steer=requested,limit=limit,guard_state='predictive',
                       guard_margin_m=self.margin)
