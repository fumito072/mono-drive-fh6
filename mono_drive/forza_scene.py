"""Measured road and traffic constraints for candidate paths; no game AutoDrive.

Research-inspired engineering implementation, not reproduction of the papers:
SafeDrive (2026): footprint/road and per-agent future checks.
Expand-and-Prune (2026): lane-intent and transformed temporal candidates.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from .forza_speed import MAX_CRUISE_KMH
from pathlib import Path
import time
import cv2
import numpy as np
from .forza_vehicle import steering_angle

HORIZON = .435


def project(x, z):
    from .forza_projection import project as shared_project
    return shared_project(x,z)


def visible_road_evidence(road,frame):
    """Reject broad snow patches even when the city model calls them road."""
    image = cv2.resize(frame,(road.shape[1],road.shape[0]),interpolation=cv2.INTER_AREA).astype(np.float32)
    brightness = cv2.blur(image.max(axis=2),(9,7))
    blue_cast = cv2.blur(image[:,:,0]-image[:,:,2],(9,7))
    snow = (brightness > 175) | ((brightness > 125) & (blue_cast > 17))
    result = road.copy()
    result[snow] = 0
    return result


def road_value(road, x, z):
    if z <= 0:
        return 0.
    u,v = project(x,z)
    h,w = road.shape
    xi,yi = int(round(u*(w-1))),int(round(v*(h-1)))
    if not (0 <= xi < w and 0 <= yi < h):
        return 0.
    return float(road[yi,xi])/255


@dataclass
class ScenePlan:
    path: np.ndarray
    steer: float
    limit: float
    quality: float
    candidate_count: int
    vehicles: int
    reason: str
    road_score: float
    painted_boundaries: int = 0
    image_path: tuple = ()
    minimap_angle: float | None = None
    minimap_confidence: float = 0.
    lane_edges: tuple = ()
    guard_state: str = ''
    guard_margin_m: float | None = None
    road_limit: float | None = None
    inferred: bool = False
    inference_age_s: float | None = None
    verified_route: bool = False
    uncertainty_limited: bool = False


def road_geometry_through_traffic(road,cars):
    """Vehicles occlude the road; their silhouettes are not its outer edges."""
    result = road.copy()
    h,w = road.shape
    for car in cars:
        if car.get('view') == 'side_left' or car.get('confidence',0) < .5:
            continue
        x1,y1,x2,y2 = car['box']
        a,b = max(0,int(x1*w)),min(w,int(x2*w)+1)
        from .forza_projection import parameters
        top,bottom = max(int(parameters()['horizon']*h),int(y1*h)),min(130,int(y2*h)+1)
        if a >= b or top >= bottom:
            continue
        neighbours = np.r_[road[max(top,bottom-5):bottom,max(0,a-8):a].ravel(),
                           road[max(top,bottom-5):bottom,b:min(w,b+8)].ravel()]
        if len(neighbours) and np.mean(neighbours > 140) > .3:
            result[top:bottom,a:b] = np.maximum(result[top:bottom,a:b],220)
    return result


def paint_without_traffic(frame,cars):
    result = frame.copy()
    h,w = frame.shape[:2]
    for car in cars:
        if car.get('view') == 'side_left':
            continue
        x1,y1,x2,y2 = car['box']
        cv2.rectangle(result,(max(0,int(x1*w)-4),max(0,int(y1*h)-4)),
                      (min(w-1,int(x2*w)+4),min(h-1,int(y2*h)+4)),(40,40,40),-1)
    return result


def targets_left_half(plan,road):
    rows = bounds(road)
    if len(rows) < 2:
        return False
    for z,x in plan.path:
        if rows[0,0] <= z <= rows[-1,0]:
            middle = np.interp(z,rows[:,0],(rows[:,1]+rows[:,2])/2)
            if x > middle-.35:
                return False
    return True


def minimap_plan(frame, road, raw_road, cars, speed):
    from .forza_minimap import road_hint
    road = visible_road_evidence(road,frame)
    raw_road = visible_road_evidence(raw_road,frame)
    hint = road_hint(frame)
    if hint is None or hint.confidence < .55 or abs(hint.angle) > 1.2:
        return None
    # Map geometry is only a heading cue, never evidence that snow is drivable.
    raw_fraction = np.mean(raw_road[110:126,175:235] > 180)
    confirmed_fraction = np.mean(road[110:126,175:235] > 140)
    if raw_fraction < .8 or confirmed_fraction < .3:
        from .forza_lane import road_patch
        if not all(road_patch(frame,x,y) for x in (.48,.53,.58) for y in (.53,.56)):
            return None
    steer = float(np.clip(.5*hint.angle,-.55,.55))
    curvature = math.tan(steer*steering_angle(speed))/2.64
    z = np.array([4.,6.,8.,10.,12.])
    path = np.column_stack((z,.5*curvature*z*z))
    road_limit = min(MAX_CRUISE_KMH,math.sqrt(2.15/max(.001,abs(curvature)))*3.6)
    limit = min(road_limit,traffic_limit(path,cars,speed)[0])
    return ScenePlan(path,steer,limit,.65,1,len(cars),'minimap_supported',
                     float(confirmed_fraction),0,tuple(project(float(x),float(d)) for d,x in path),
                     hint.angle,hint.confidence,road_limit=road_limit)


def painted_corridor(frame, road):
    """Measure actual edge/divider spacing; independent of assumed road pitch."""
    im = cv2.resize(frame,(384,216),interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    value = hsv[:,:,2].astype(float)
    contrast = value-cv2.blur(value,(21,9))
    white = (hsv[:,:,1] < 105) & (value > 85) & (contrast > 9)
    yellow = (hsv[:,:,0] > 8) & (hsv[:,:,0] < 40) & (hsv[:,:,1] > 75) & (value > 50)
    allowed = cv2.dilate((road > 130).astype(np.uint8),np.ones((3,5),np.uint8)) > 0
    white &= allowed
    yellow &= allowed
    measured = []
    previous = None
    for v in (.585,.57,.555,.54,.525,.51,.495,.48,.465,.45):
        y = round(v*215)
        asphalt = road[y] > 140
        changes = np.diff(np.r_[0,asphalt,0].astype(int))
        spans = [(a,b) for a,b in zip(np.where(changes==1)[0],np.where(changes==-1)[0]) if b-a > 18]
        if not spans:
            continue
        a,b = min(spans,key=lambda pair:abs((pair[0]+pair[1])/2-260))
        def runs(mask):
            line = cv2.morphologyEx((mask[y-1:y+2].any(0)*255).astype(np.uint8)[None],cv2.MORPH_CLOSE,np.ones((1,3),np.uint8))[0] > 0
            d = np.diff(np.r_[0,line,0].astype(int))
            return [(l+r-1)/2 for l,r in zip(np.where(d==1)[0],np.where(d==-1)[0]) if 1 <= r-l <= 20 and a-6 <= (l+r)/2 <= b+4]
        whites,yellows = runs(white),runs(yellow)
        if not whites:
            continue
        edge_reference = previous[0] if previous else 383*.45
        edge_options = [x for x in whites if x < (min(yellows)-20 if yellows else 383*.58)]
        if not edge_options:
            continue
        edge = min(edge_options,key=lambda x:abs(x-edge_reference))
        dividers = [x for x in (yellows or whites) if x-edge > 25]
        if not dividers:
            continue
        divider = min(dividers) if yellows else min(dividers,key=lambda x:abs(x-(previous[1] if previous else (a+b)/2)))
        if divider-edge > 220:
            continue
        if previous and (abs(edge-previous[0]) > 65 or abs(divider-previous[1]) > 65):
            continue
        if previous and divider-edge > previous[1]-previous[0]+8:
            continue
        centre = (edge+divider)/2
        margin = .30*(divider-edge)
        if np.mean(road[y,max(0,int(centre-margin)):min(384,int(centre+margin))] > 140) < .9:
            continue
        measured.append((v,edge/383,divider/383))
        previous = edge,divider
    if len(measured) < 3 or measured[0][0] < .54:
        return None
    return measured


def paint_plan(frame, road, cars, speed):
    corridor = painted_corridor(frame,road)
    if not corridor:
        return None
    points, image_points = [],[]
    for v,left,right in corridor:
        width = right-left
        z = 1.75/width  # 3.5m lane, fx = 0.5*image width
        u = (left+right)/2
        x = (u-.5)*2*z-.35
        points.append((z,x))
        image_points.append((u,v))
    order = np.argsort([p[0] for p in points])
    points = np.array(points)[order]
    image_points = tuple(image_points[i] for i in order)
    # Ignore repeated ranges caused by line anti-aliasing.
    keep = np.r_[True,np.diff(points[:,0]) > .15]
    points = points[keep]
    image_points = tuple(p for p,k in zip(image_points,keep) if k)
    if len(points) < 2 or points[0,0] > 14 or points[-1,0]-points[0,0] < 2.:
        return None
    if not coherent_lane_shape(points):
        return None
    lookahead = float(np.clip(7.+speed/3.6*.5,points[0,0],min(18.,points[-1,0])))
    x = float(np.interp(lookahead,points[:,0],points[:,1]))
    kappa = 2*x/(lookahead**2+x*x)
    steer = float(np.clip(math.atan(2.64*kappa)/steering_angle(speed),-.65,.65))
    limit,_ = curve_speed(points,speed)
    limit = min(limit,math.sqrt(2.15/max(.001,abs(kappa)))*3.6)
    road_limit = limit
    limit = min(limit,traffic_limit(points,cars,speed)[0])
    reason = 'traffic_stop' if limit < 1 else 'paint_corridor'
    near_v,near_left,near_right = corridor[0]
    camera_centre = .5+.35/(2*points[0,0])
    if camera_centre > near_right:
        return None  # Do not steer across a dividing line into a different lane.
    if camera_centre < near_left:
        limit = min(limit,5.)
        road_limit = min(road_limit,5.)
        reason = 'road_reentry'
    return ScenePlan(points,steer,limit,.85,1,len(cars),reason,.9,len(points),image_points,
                     lane_edges=tuple((float(z),float(x-1.75),float(x+1.75)) for z,x in points),road_limit=road_limit)


def paint_pair_plan(frame, road, cars, speed):
    """Bridge dashed paint only when both fitted boundaries enclose semantic road."""
    from .forza_lane import paint_groups
    from types import SimpleNamespace
    groups = [g for g in paint_groups(frame) if g['count'] >= 2 and g['weight'] >= 70]
    pairs = []
    for left in groups:
        for right in groups:
            near_width = (right['near']-left['near'])/960
            far_width = (right['far']-left['far'])/960
            if .20 < near_width < .70 and .08 < far_width < .35 and .2 < far_width/near_width < .85:
                if left['near']/960 < .53 < right['near']/960:
                    pairs.append((left['weight']+right['weight'],left,right))
    if not pairs:
        return None
    _,left,right = max(pairs,key=lambda item:item[0])
    lane = SimpleNamespace(edge=(left['far']/960,left['near']/960),
                           center=(right['far']/960,right['near']/960))
    points, images = [], []
    for v in (.57,.555,.54,.525,.51,.495,.48,.465):
        ratio = (v*540-255)/50
        left = lane.edge[0]+ratio*(lane.edge[1]-lane.edge[0])
        right = lane.center[0]+ratio*(lane.center[1]-lane.center[0])
        if not 0 < left < right < 1 or right-left < .09:
            continue
        xs = np.linspace(left+.2*(right-left),right-.2*(right-left),9)
        quality = road[round(v*215),np.clip((xs*383).astype(int),0,383)]/255.
        if np.mean(quality > .65) < .8:
            continue
        z = 1.75/(right-left)
        u = (left+right)/2
        points.append((z,(u-.5)*2*z-.35))
        images.append((u,v))
    if len(points) < 5 or points[-1][0]-points[0][0] < 2.:
        return None
    path = np.array(points)
    if np.any(np.diff(path[:,0]) <= 0) or path[0,0] > 8:
        return None
    if not coherent_lane_shape(path):
        return None
    near_left,near_right = lane.edge[1],lane.center[1]
    camera_centre = .5+.35/(2*path[0,0])
    if not near_left < camera_centre < near_right:
        return None
    lookahead = float(np.clip(7.+speed/7.2,path[0,0],path[-1,0]))
    x = float(np.interp(lookahead,path[:,0],path[:,1]))
    kappa = 2*x/(lookahead**2+x*x)
    steer = float(np.clip(math.atan(2.64*kappa)/steering_angle(speed),-.5,.5))
    limit,_ = curve_speed(path,speed)
    road_limit = min(limit,math.sqrt(2.15/max(.001,abs(kappa)))*3.6)
    limit = min(road_limit,traffic_limit(path,cars,speed)[0])
    return ScenePlan(path,steer,limit,.80,1,len(cars),'paint_pair',.85,len(points),tuple(images),
                     lane_edges=tuple((float(z),float(x-1.75),float(x+1.75)) for z,x in path),road_limit=road_limit)


class SceneReader:
    def __init__(self, path=None):
        self.path = Path(path) if path else Path(__file__).resolve().parents[1]/'out'/'scene.npz'
        self.version = None
        self.value = None

    def read(self, now):
        try:
            version = self.path.stat().st_mtime_ns
            if version != self.version:
                with np.load(self.path,allow_pickle=False) as data:
                    road = data['road'].copy()
                    cars = json.loads(str(data['cars']))
                    self.value = dict(road=road,cars=cars,captured=float(data['captured']),
                                      image=cv2.imdecode(data['image_jpeg'],cv2.IMREAD_COLOR) if 'image_jpeg' in data else None,
                                      ego=json.loads(str(data['ego'])) if 'ego' in data else None,
                                      obstacles=data['obstacles'].copy() if 'obstacles' in data else None,
                                      confirmed_obstacles=data['confirmed_obstacles'].copy() if 'confirmed_obstacles' in data else None,
                                      raw_road=data['raw_road'].copy() if 'raw_road' in data else road,
                                      sidewalk=data['sidewalk'].copy() if 'sidewalk' in data else None,
                                      planning_roi=data['planning_roi'].copy() if 'planning_roi' in data else None,
                                      projection=json.loads(str(data['projection'])) if 'projection' in data else None,
                                      inference_ms=float(data['inference_ms']),
                                      recording=str(data.get('recording','')),frame=str(data.get('frame','')))
                self.version = version
            if self.value and 0 <= now-self.value['captured'] < .5:
                return self.value
        except (OSError,ValueError,KeyError):
            pass
        return None


def bounds(road, seed_x=.56):
    """Connected visible road spans, in approximate ground coordinates."""
    h,w = road.shape
    result = []
    previous = seed_x
    previous_edges = None
    for z in (6.,8.,10.,12.,16.,22.,30.,40.,50.,60.):
        _,v = project(0,z)
        y = round(v*(h-1))
        if not 1 <= y < h-1:
            continue
        strip = (road[y-1:y+2].mean(axis=0) >= 140).astype(np.uint8)[None]
        strip = cv2.morphologyEx(strip,cv2.MORPH_CLOSE,np.ones((1,5),np.uint8))[0]
        changes = np.diff(np.r_[0,strip,0].astype(int))
        spans = [(a,b) for a,b in zip(np.where(changes==1)[0],np.where(changes==-1)[0]) if b-a >= 5]
        if not spans:
            continue
        a,b = min(spans,key=lambda ab: abs((ab[0]+ab[1])/(2*w)-previous))
        midpoint = (a+b)/(2*w)
        # Disconnected snow patches must not create a sudden imaginary road bend.
        if result and abs(midpoint-previous) > .16:
            break
        if previous_edges is not None:
            movement=np.array([a,b])/w-previous_edges
            # A shadow/occluder that moves one edge across the road while the
            # other edge stays fixed is not a sharp bend in the lane.
            if np.max(np.abs(movement))>.13 and np.min(np.abs(movement))<.04:
                break
        previous = midpoint
        previous_edges=np.array([a,b])/w
        left = (a/(w-1)-.5)*2*z-.35
        right = ((b-1)/(w-1)-.5)*2*z-.35
        if right-left >= 2.2:
            result.append((z,left,right))
    return np.array(result,dtype=float).reshape(-1,3)


def fallback_lane_edges(road):
    """Give a map-only heading a measured, body-containing lane envelope."""
    rows=bounds(road,seed_x=.53)
    if len(rows)<3 or rows[0,0]>8.:
        return ()
    z,left,right=rows.T
    width=right-left
    if np.median(width[:3])<4.2:
        return ()  # A narrow paved strip may be a sidewalk, not a traffic lane.
    divider=left+np.minimum(3.6,width/2.)
    if left[0]>-1.0 or divider[0]<1.0:
        return ()
    return tuple(map(tuple,np.column_stack((z,left,divider))))


def footprint_quality(road, path):
    """Check center, edges and corners, not just a center pixel."""
    samples = []
    headings = np.arctan(np.gradient(path[:,1],path[:,0]))
    for index,((z,x),heading) in enumerate(zip(path,headings)):
        if index > 0 and z > path[-1,0]-2.21:
            continue  # Do not place the vehicle nose beyond the observed horizon.
        c,s = math.cos(heading),math.sin(heading)
        for length in (-2.21,0.,2.21):
            for width in (-.93,0.,.93):
                px,pz=x+s*length+c*width,z+c*length-s*width
                _,v=project(px,pz)
                if v>=.60:
                    continue  # The dashboard hides this ground; it is not a non-road observation.
                samples.append(road_value(road,px,pz))
    if len(samples)<9:
        return 0.,0.
    return float(np.quantile(samples,.12)), float(np.mean(samples))


def inside_lane(path, lane_left, lane_right):
    z = path[:,0]
    def edge_at(values, distance):
        if distance < z[0]:
            return values[0]+(values[1]-values[0])*(distance-z[0])/(z[1]-z[0])
        if distance > z[-1]:
            return values[-1]+(values[-1]-values[-2])*(distance-z[-1])/(z[-1]-z[-2])
        return float(np.interp(distance,z,values))
    headings = np.arctan(np.gradient(path[:,1],z))
    for index,((distance,x),heading) in enumerate(zip(path,headings)):
        if index > 0 and distance > z[-1]-2.21:
            continue
        c,s = math.cos(heading),math.sin(heading)
        for length in (-2.21,0.,2.21):
            for width in (-.93,.93):
                corner_x,corner_z = x+s*length+c*width,distance+c*length-s*width
                if not edge_at(lane_left,corner_z)+.05 <= corner_x <= edge_at(lane_right,corner_z)-.05:
                    return False
    return True


def coherent_lane_shape(path):
    """Crosswalk stripes must not become alternating left/right boundaries."""
    if len(path)<3:
        return True
    dz=np.diff(path[:,0])
    if np.any(dz<=.05):
        return False
    slopes=np.diff(path[:,1])/dz
    headings=np.arctan(slopes)
    return bool(np.max(np.abs(np.diff(headings)))<.55)


def smooth_lane_reference(z,reference,left,right):
    """Do not turn row-to-row segmentation noise into steering or fake bends."""
    if len(z) < 3:
        return reference
    weights = 1/np.sqrt(z)
    for degree,tolerance in ((1,.30),(2,.55)):
        fitted = np.polynomial.Polynomial.fit(z,reference,degree,w=weights)(z)
        error = float(np.sqrt(np.average((fitted-reference)**2,weights=weights)))
        if error > tolerance or np.max(np.abs(fitted-reference)) > 1.:
            continue
        if inside_lane(np.column_stack((z,fitted)),left,right):
            return fitted
    return reference


def feasible_curve_speed(curvatures, distances, current_kmh, lateral_accel=.22*9.81, brake_accel=2.0):
    """v^2*kappa <= lateral allowance; brake before each future bend.

    Comfort targets are engineering defaults, not a claim about tyre adhesion.
    Bank angle and actual grip are unknown, so no positive banking credit is used.
    """
    curvatures = np.abs(np.asarray(curvatures,dtype=float))
    distances = np.asarray(distances,dtype=float)
    curve_squared = lateral_accel/np.maximum(curvatures,.0001)
    # Keep a longitudinal reserve inside a conservative combined acceleration circle.
    now_lateral = (current_kmh/3.6)**2*curvatures[0]
    reserve = math.sqrt(max(.25,(.45*9.81)**2-min(now_lateral,.44*9.81)**2))
    braking = min(brake_accel,reserve)
    available = np.maximum(0.,distances-max(0.,current_kmh)/3.6*.45)
    allowable_squared = curve_squared+2*braking*available
    return min(MAX_CRUISE_KMH,3.6*math.sqrt(float(np.min(allowable_squared))))


def curve_speed(path, current_kmh):
    near = path[path[:,0] <= 35.]
    if len(near) < 2:
        return 0.,0.
    z,x = near.T
    line = np.polynomial.Polynomial.fit(z,x,1,w=1/np.sqrt(z))
    # A few centimetres of paint/segmentation jitter across a short visible
    # interval is not evidence of a tight bend. Pursuit steering still imposes
    # its own lateral-acceleration limit below the caller's cruising target.
    if float(np.sqrt(np.average((line(z)-x)**2,weights=1/np.sqrt(z)))) < .18:
        return MAX_CRUISE_KMH,0.
    # A low-order fit suppresses pixel-level curvature spikes from road segmentation.
    degree = 2 if z[-1]-z[0] < 14 else 3
    fit = np.polynomial.Polynomial.fit(z,x,min(degree,len(z)-1),w=1/np.sqrt(z)).convert()
    sample_z = np.linspace(z[0],z[-1],30)
    slope = fit.deriv(1)(sample_z)
    curvature = fit.deriv(2)(sample_z)/(1+slope*slope)**1.5
    arc = np.r_[0.,np.cumsum(np.diff(sample_z)*(np.sqrt(1+slope[:-1]**2)+np.sqrt(1+slope[1:]**2))/2)]
    return feasible_curve_speed(curvature,arc,current_kmh),float(np.max(np.abs(curvature)))


def traffic_limit(path, cars, speed):
    from .forza_traffic import predict_traffic
    prediction = predict_traffic(path,cars,speed)
    return prediction.limit,prediction.risk


def geometry_speed_budget(plan):
    """Discard traffic limits from a path that is about to be replanned.

    Road/turn/re-entry limits survive. Traffic is independently recalculated
    on the final path in LaneFusion; an old blocked path cannot latch zero.
    """
    if (plan is None or plan.road_limit is None or plan.reason not in
            ('paint_corridor','paint_pair','road_candidates','traffic_stop','traffic_follow',
             'minimap_supported','paint_temporal','road_reentry','visible_road_creep')):
        return plan
    from dataclasses import replace
    reason='road_candidates' if plan.reason in ('traffic_stop','traffic_follow') else plan.reason
    return replace(plan,limit=plan.road_limit,reason=reason)


class CandidatePlanner:
    def __init__(self):
        self.last_path = None
        self.last_ego = None
        self.last_seen = 0.
        self.last_steer = 0.
        self.latest = None
        self.road_seed_x = .65

    def plan(self, road, cars, ego, now, frame=None, max_rows=None, markings=None, raw_road=None, _prepared=False):
        if frame is not None and not _prepared:
            road = visible_road_evidence(road,frame)
            raw_road = visible_road_evidence(raw_road if raw_road is not None else road,frame)
        if not _prepared:
            road = road_geometry_through_traffic(road,cars)
        geometry_raw = raw_road if _prepared else road_geometry_through_traffic(raw_road if raw_road is not None else road,cars)
        if frame is not None and max_rows is None:
            clean_frame = paint_without_traffic(frame,cars)
            paint = paint_plan(clean_frame,geometry_raw,cars,ego['speed'])
            if paint is None:
                paint = paint_pair_plan(clean_frame,geometry_raw,cars,ego['speed'])
            if paint is not None and targets_left_half(paint,geometry_raw):
                self.latest = paint
                return paint
        road_cars = [car for car in cars if car.get('view','front') == 'front' and car.get('moving')
                     and car.get('confidence',0) > .6 and car.get('vz',0) > 1]
        if road_cars:
            # Moving same-direction tyres anchor which visible surface is the road.
            self.road_seed_x = float(np.median([(c['box'][0]+c['box'][2])/2 for c in road_cars]))
        rows = bounds(road,float(np.clip(self.road_seed_x,.3,.85)))
        if len(rows):
            near_z,near_left,near_right = rows[0]
            self.road_seed_x = float(np.clip(project((near_left+near_right)/2,near_z)[0],.3,.85))
        if len(rows) >= 3:
            normal_width = float(np.median(rows[:3,2]-rows[:3,1]))
            narrowed = np.flatnonzero(rows[:,2]-rows[:,1] < .68*normal_width)
            if len(narrowed):
                rows = rows[:max(3,int(narrowed[0]))]
        if max_rows is not None:
            rows = rows[:max_rows]
        if len(rows) < 2 or rows[-1,0] < 8 or rows[0,0] > 10:
            self.latest = None
            return None
        z,left,right = rows.T
        lane_left = left.copy()
        divider = left+.5*(right-left)
        painted = 0
        if frame is not None:
            from .forza_lane import paint_groups
            groups = markings if markings is not None else paint_groups(paint_without_traffic(frame,cars))
            markings = groups
            for i,distance in enumerate(z):
                _,v = project(0,distance)
                y = v*540
                candidates = []
                for group in groups:
                    if group['count'] < 2 or group['weight'] < 65 or not min(group['ys'])-12 <= y <= max(group['ys'])+12:
                        continue
                    u = (group['far']+(y-255)*(group['near']-group['far'])/50)/960
                    x = (u-.5)*2*distance-.35
                    if road_value(road,x+.12,distance) < .45:
                        continue
                    candidates.append((x,group))
                if not candidates:
                    continue
                middle = min(candidates,key=lambda pair:abs(pair[0]-divider[i]))
                width = divider[i]-left[i]
                if abs(middle[0]-divider[i]) < .35*width and min(middle[1]['left'],middle[1]['right']) > .55:
                    divider[i] = middle[0]
                    painted += 1
                edge = min(candidates,key=lambda pair:abs(pair[0]-left[i]))
                if abs(edge[0]-left[i]) < min(1.,.30*width) and edge[0] < divider[i]-2.1:
                    lane_left[i] = edge[0]
        reference = smooth_lane_reference(z,(lane_left+divider)/2,lane_left,divider)
        candidates = []
        # Different lateral positions all stay in the leftmost lane.
        for offset in (-.45,-.2,0.,.2,.45):
            candidates.append(np.column_stack((z,reference+offset)))
        # Reproject the previous accepted path through current ego translation/yaw.
        if self.last_path is not None and now-self.last_seen < .7:
            def rotation(yaw):
                c,s = math.cos(yaw),math.sin(yaw)
                return np.array([[c,s],[-s,c]])
            old_rotation = rotation(self.last_ego['yaw'])
            rotation_now = rotation(ego['yaw'])
            old_position = np.array(self.last_ego['position'])[[0,2]]
            position = np.array(ego['position'])[[0,2]]
            old_points = self.last_path[:,[1,0]]
            shifted = (rotation_now.T@((old_rotation@old_points.T)+old_position[:,None]-position[:,None])).T
            order = np.argsort(shifted[:,1])
            temporal_x = np.interp(z,shifted[order,1],shifted[order,0])
            candidates.append(np.column_stack((z,temporal_x)))
        accepted = []
        speed = ego['speed']
        for path in candidates:
            # Avoid a temporal candidate drifting into the opposing lane.
            x = path[:,1]
            if np.any(x < lane_left+1.0) or np.any(x > divider-1.0):
                continue
            if not inside_lane(path,lane_left,divider):
                continue
            floor,average = footprint_quality(road,path)
            if floor < .42 or average < .64:
                continue
            lookahead = max(8.,min(18.,8.+speed/3.6*.65))
            target = float(np.interp(lookahead,z,x))
            curvature = 2*target/(lookahead**2+target**2)
            steer = float(np.clip(math.atan(2.64*curvature)/steering_angle(speed),-.65,.65))
            preview_limit,maximum_curvature = curve_speed(path,speed)
            limit = min(preview_limit,math.sqrt(2.15/max(.001,abs(curvature)))*3.6)
            if z[-1] < 12:
                limit = min(limit,8.)  # Creep over a crest only within measured free road.
            road_limit = limit
            other_limit,risk = traffic_limit(path,cars,speed)
            limit = min(limit,other_limit)
            cost = float(np.mean((x-reference)**2))+2*(steer-self.last_steer)**2+3*risk-.3*average
            # Only supported same-direction moving trails act as weak road evidence.
            for car in cars:
                if car.get('view') == 'side_left':
                    continue
                if not car.get('moving') or car.get('vz',0) <= 1:
                    continue
                for tx,tz in car.get('trail',[]):
                    if z[0] <= tz <= z[-1] and road_value(road,tx,tz) > .6:
                        cost -= .03*math.exp(-((float(np.interp(tz,z,x))-tx)/1.5)**2)
            accepted.append((cost,path,steer,limit,average,risk,road_limit))
        if not accepted:
            if len(rows) > 2:
                # A hidden bend/crest shortens sight distance; keep only a verified
                # prefix and lower speed instead of requiring an unseen far path.
                return self.plan(road,cars,ego,now,frame,max_rows=len(rows)-1,markings=markings,
                                 raw_road=geometry_raw,_prepared=True)
            # A crest may hide the front of a car placed at a distant waypoint.
            # When the presently visible near corridor is clear, a 6 km/h crawl
            # needs less than 2m to stop, well inside the measured 6-8m corridor.
            creep_divider = left+.5*(right-left)
            creep_reference = (left+creep_divider)/2
            close_path = np.column_stack((z,creep_reference))
            clear_now = all(road_value(road,float(x+dx),float(distance)) > .80
                            for distance,x in close_path for dx in (-.93,0.,.93))
            if (clear_now and abs(creep_reference[0]) < .6 and left[0] < -1.0
                    and creep_divider[0] > 1.0 and inside_lane(close_path,left,creep_divider)):
                steer = float(np.clip(math.atan(2.64*2*creep_reference[0]/(z[0]**2+creep_reference[0]**2))
                                      /steering_angle(speed),-.25,.25))
                traffic_speed,_ = traffic_limit(close_path,cars,speed)
                plan = ScenePlan(close_path,steer,min(6.,traffic_speed),.70,1,len(cars),'visible_road_creep',.8,painted,
                                 lane_edges=tuple(map(tuple,np.column_stack((z,left,creep_divider)))),road_limit=6.)
                self.latest = plan
                return plan
            # Recover from the left road edge at walking pace. The current car
            # cannot already satisfy a lane-containment check when it is outside
            # that lane. Aim only into the adjacent left lane, never across it.
            if -1.0 <= left[0] < 2.5 and abs(creep_reference[0]) < 4.5:
                entry_clear = all(road_value(road,float(x+dx),float(distance)) > .8
                                  for distance,x in close_path for dx in (-.93,0.,.93))
                if entry_clear:
                    distance = max(6.,float(z[0]))
                    x = float(creep_reference[0])
                    steer = float(np.clip(math.atan(2.64*2*x/(distance**2+x*x))/steering_angle(speed),-.6,.6))
                    traffic_speed,_ = traffic_limit(close_path,cars,speed)
                    plan = ScenePlan(close_path,steer,min(5.,traffic_speed),.65,1,len(cars),'road_reentry',.8,0,road_limit=5.)
                    self.latest = plan
                    return plan
            self.latest = None
            return None
        _,path,steer,limit,quality,risk,road_limit = min(accepted,key=lambda row:row[0])
        reason = 'traffic_stop' if limit < 1 else ('traffic_follow' if limit < road_limit-1. and cars else 'road_candidates')
        plan = ScenePlan(path,steer,limit,min(.95,quality),len(accepted),len(cars),reason,quality,painted,
                         lane_edges=tuple(map(tuple,np.column_stack((z,lane_left,divider)))),road_limit=road_limit)
        self.last_path,self.last_ego,self.last_seen,self.last_steer = path.copy(),dict(ego),now,steer
        self.latest = plan
        return plan
