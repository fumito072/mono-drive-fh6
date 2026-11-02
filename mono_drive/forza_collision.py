"""Short-horizon traffic prediction and a final forward collision interlock."""
from dataclasses import dataclass, replace
import math
import numpy as np
from .forza_vehicle import steering_angle


@dataclass
class CollisionState:
    emergency: bool = False
    seconds: float | None = None
    reason: str = ''
    brake: float = 0.
    required_decel: float = 0.
    vehicle_id: object = None
    immediate: bool = False


def forward_collision(cars, speed_kmh, steer, path=None):
    from .forza_traffic import predict_traffic
    speed = max(0.,speed_kmh/3.6)
    if path is not None:
        path=np.asarray(path,dtype=float)
        if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or not np.isfinite(path).all() or np.any(np.diff(path[:,0])<=0):
            path=None
    curvature = math.tan(float(steer)*steering_angle(speed_kmh))/2.64
    z = np.linspace(.1,max(6.,speed*4.),40)
    current_path = np.column_stack((z,.5*curvature*z*z))
    prediction = predict_traffic(current_path if path is None else path,cars,speed_kmh,minimum_probe_kmh=0.)
    if path is not None:
        # Planned turns/avoidance govern the distant path. Preserve the
        # current-wheel check for hazards too close to wait for that turn.
        near=predict_traffic(current_path,cars,speed_kmh,minimum_probe_kmh=0.)
        if near.collision_s is not None and near.collision_s<=.75 and (
                prediction.collision_s is None or near.collision_s<prediction.collision_s):
            prediction=near
    soonest = prediction.collision_s
    if soonest is not None:
        closing=prediction.closing_mps
        usable=max(.4,prediction.gap_m-.35*closing)
        required=closing*closing/(2*usable)
        emergency=(soonest<=.75 and (not prediction.same_direction or closing>1.5 or prediction.lateral_closing)) or required>=4.5
        brake=min(.8,required/10.) if required>.6 else 0.
        if emergency:
            brake=max(brake,.8 if soonest<=.75 else .55)
        elif prediction.same_direction and not prediction.lateral_closing:
            brake=min(brake,.15)  # Coast or match a parallel lead car smoothly.
        tracked=next((car for car in cars if car.get('id')==prediction.vehicle_id),{})
        x,z=tracked.get('x',0.),tracked.get('z',float('inf'))
        stop_distance=4.4+closing*.45+closing*closing/12.
        immediate=bool((emergency and abs(x)<1.9 and 0.<z<stop_distance) or
                       (soonest<.25 and prediction.lateral_closing))
        return CollisionState(emergency,soonest,'predicted_vehicle_overlap',brake,required,
                              prediction.vehicle_id,immediate)
    return CollisionState(False,soonest,'')


def avoid_in_lane(plan, cars, speed, road):
    if not cars or not plan.lane_edges:
        return plan
    from .forza_scene import inside_lane,traffic_limit,footprint_quality
    # A single uncertain YOLO box can trigger a cautious speed check, but it
    # must not send the car sideways. Close, highly confident detections remain.
    cars=[car for car in cars if (car.get('track_frames',3)>=3 and car.get('track_age_s',.25)>=.2) or
          (car.get('confidence',0)>=.85 and 0<car.get('z',100)<8.)]
    if not cars:
        return plan
    base_limit,base_risk = traffic_limit(plan.path,cars,speed)
    if base_risk < .12:
        return plan
    edges = np.asarray(plan.lane_edges)
    left = np.interp(plan.path[:,0],edges[:,0],edges[:,1])
    right = np.interp(plan.path[:,0],edges[:,0],edges[:,2])
    best = None
    for offset in (-.65,-.5,-.35,-.2,.2,.35,.5,.65):
        path = plan.path.copy()
        path[:,1] += offset
        if any(c.get('view') == 'side_left' for c in cars) and offset < 0:
            continue
        if not inside_lane(path,left,right):
            continue
        floor,average = footprint_quality(road,path)
        if floor < .35 or average < .55:
            continue
        limit,risk = traffic_limit(path,cars,speed)
        if risk > base_risk-.08 or limit < base_limit:
            continue
        score = risk+offset*offset*.2
        if best is None or score < best[0]:
            best = score,path,limit
    if best is None:
        return plan
    _,path,limit = best
    lookahead = float(np.clip(8.+speed/7.2,path[0,0],path[-1,0]))
    x = float(np.interp(lookahead,path[:,0],path[:,1]))
    curvature = 2*x/(lookahead**2+x*x)
    steer = float(np.clip(math.atan(2.64*curvature)/steering_angle(speed),-.6,.6))
    from .forza_scene import project
    road_limit = plan.road_limit if plan.road_limit is not None else plan.limit
    return replace(plan,path=path,steer=steer,limit=min(road_limit,limit),reason='vehicle_avoidance',
                   image_path=tuple(project(float(x),float(z)) for z,x in path))
