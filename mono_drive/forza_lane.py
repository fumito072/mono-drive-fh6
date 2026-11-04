"""Left-lane guidance with temporal traffic, smooth road geometry and steady cruise."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from .forza_speed import MAX_CRUISE_KMH
import cv2
import numpy as np


@dataclass
class LaneEstimate:
    confidence: float
    target_x: float
    center_x: float
    lane_width_px: float
    steer: float
    center: tuple
    edge: tuple
    path: tuple = ()
    cars: tuple = ()


@dataclass
class Guidance:
    lane: LaneEstimate
    steer: float
    confidence: float
    max_kmh: float
    source: str
    lookahead_m: float
    vehicles: int = 0
    candidates: int = 0
    uncertainty_limited: bool = False


class LaneFusion:
    """Road/paint geometry and temporally reprojected candidates guide the lane."""
    def __init__(self, semantic=False):
        if semantic:
            import importlib
            from . import forza_scene,forza_traffic,forza_collision,forza_minimap
            importlib.reload(forza_traffic)
            importlib.reload(forza_collision)
            importlib.reload(forza_scene)
            importlib.reload(forza_minimap)
        from .forza_scene import SceneReader, CandidatePlanner
        self.semantic = semantic
        self.scene_reader = SceneReader()
        self.scene_planner = CandidatePlanner()
        self.last = None
        self.seen = self.updated = 0.
        self.yaw = 0.
        self.bias = None
        self.travelled = 0.
        self.scene_cache = None
        from . import forza_lane_guard
        if semantic:
            importlib.reload(forza_lane_guard)
        self.departure_guard = forza_lane_guard.LaneDepartureGuard()
        from . import forza_turns
        if semantic:
            importlib.reload(forza_turns)
        self.turns = forza_turns.TurnCoordinator()
        from . import forza_lane_memory,forza_lateral
        if semantic:
            importlib.reload(forza_lane_memory)
            importlib.reload(forza_lateral)
        self.memory = forza_lane_memory.TemporalLaneModel()
        self.predictive = forza_lateral.PredictiveLaneControl()
        from .forza_carriageway import CarriagewayGuard
        self.carriageway=CarriagewayGuard()
        self.carriageway_status={}
        self.carriageway_publish=0.
        self.boundary_unknown_since=None
        from . import forza_continuation
        if semantic:
            importlib.reload(forza_continuation)
        self.continuation=forza_continuation.LaneContinuation()
        from . import forza_surface
        if semantic:
            importlib.reload(forza_surface)
        self.surface_guard=forza_surface.RoadSurfaceGuard()
        from . import forza_failure_memory
        if semantic:
            importlib.reload(forza_failure_memory)
        self.failure_guard=forza_failure_memory.FailedApproachGuard()
        from . import forza_route_memory
        if semantic:
            importlib.reload(forza_route_memory)
        self.route_memory=forza_route_memory.SuccessfulRouteMemory(refresh_s=2.)
        self.failure_guard.route_memory=self.route_memory
        from .forza_learned_path import LearnedPathPrior
        self.learned_prior=LearnedPathPrior()
        from . import forza_temporal
        if semantic:
            importlib.reload(forza_temporal)
        self.temporal=forza_temporal.TemporalSteering()
        self.legacy_veto_status='not_requested'

    def bridge_scene(self, scene, telemetry, now):
        if self.scene_cache is None:
            return None
        previous, origin, yaw, seen = self.scene_cache
        moved = np.linalg.norm(np.array(telemetry['position'])-origin)
        turn = math.atan2(math.sin(telemetry['yaw']-yaw),math.cos(telemetry['yaw']-yaw))
        if now-seen > .65 or moved > 1.5 or abs(turn) > .08 or not previous.image_path:
            return None
        road = scene.get('raw_road',scene['road'])
        samples = [(u-.5*turn,v) for u,v in previous.image_path]
        evidence = [road[int(v*215),int(np.clip(u*383,0,383))]/255. for u,v in samples]
        if np.mean(np.array(evidence) > .65) < .8:
            return None
        from .forza_scene import traffic_limit
        road_limit = previous.road_limit if previous.road_limit is not None else previous.limit
        limit = min(road_limit,traffic_limit(previous.path,scene['cars'],telemetry['speed'])[0])
        return replace(previous,steer=float(np.clip(previous.steer-turn,-.5,.5)),limit=limit,
                       reason='paint_temporal',image_path=tuple(samples))

    @staticmethod
    def preview(model, speed):
        lookahead = max(10., min(40., 10.+max(0., speed)/3.6*1.8))
        limit = MAX_CRUISE_KMH
        path = getattr(model, 'path', None)
        if path is not None and model.confidence >= .3:
            from .pathnet import curvature_profile
            curvature, s = curvature_profile(path['F'], path['R'], path['S'])
            relevant = np.abs(curvature[(s >= 8) & (s <= lookahead+8)])
            if len(relevant) and np.isfinite(relevant).all():
                limit = min(limit, math.sqrt(.18*9.81/max(.001, float(np.max(relevant))))*3.6)
        elif model.confidence < .3:
            limit = 20.
        return lookahead, max(2., limit)

    def update(self, frame, model, telemetry, now):
        if self.semantic:
            # Do not report a prior frame's safe margin when this frame never
            # reached the predictive controller.
            self.predictive.margin=None
            self.predictive.target=None
            self.predictive.curvature=0.
            from .forza_scene import project
            scene = self.scene_reader.read(now)
            if scene is None:
                return None
            from .forza_projection import set_projection
            set_projection(scene.get('projection'))
            source = scene.get('ego') or telemetry
            planning_frame = scene.get('image')
            if planning_frame is None:
                planning_frame = frame
            self.carriageway_status=self.carriageway.update(scene.get('raw_road',scene['road']),
                scene.get('obstacles'),scene['cars'],now)
            if now-self.carriageway_publish>.5:
                from pathlib import Path
                from .forza_local_learning import write_json
                import time
                write_json(Path(__file__).resolve().parents[1]/'out/carriageway-state.json',
                           dict(time=time.time(),**self.carriageway_status))
                self.carriageway_publish=now
            # Road masks, paint and pose must describe one exposure.
            plan = self.scene_planner.plan(scene['road'],scene['cars'],source,now,frame=planning_frame,
                                           raw_road=scene.get('raw_road'))
            plan = self.turns.update(planning_frame,scene,source,now,plan)
            if plan is None:
                plan = self.bridge_scene(scene,telemetry,now)
                if plan is None:
                    from .forza_scene import minimap_plan
                    plan = minimap_plan(planning_frame,scene['road'],scene.get('raw_road',scene['road']),
                                        scene['cars'],telemetry['speed'])
                    if plan is not None:
                        from .forza_scene import fallback_lane_edges
                        plan=replace(plan,lane_edges=fallback_lane_edges(scene['road']))
            elif plan.reason.startswith('paint_'):
                self.scene_cache = (plan,np.array(source['position']),source['yaw'],now)
            from .forza_scene import geometry_speed_budget
            plan=geometry_speed_budget(plan)
            from .forza_lane_memory import align_plan
            plan = align_plan(plan,source,telemetry)
            plan = self.memory.update(plan,telemetry,now,road=scene.get('raw_road',scene['road']),road_ego=source)
            if plan is None:
                from .forza_minimap import road_hint
                plan=self.continuation.predict(scene,telemetry,now,hint=road_hint(planning_frame))
                self.memory.state='inferred'
            from .forza_lateral import center_lane_plan
            plan=center_lane_plan(plan,telemetry['speed'])
            plan=self.surface_guard.apply(plan,scene,telemetry,now)
            # Reuse complete traversals; low-speed alignment keeps a near horizon.
            plan=self.route_memory.apply(plan,scene,telemetry)
            if (self.route_memory.status.get('action')=='successful_route' and
                    (self.surface_guard.alignment_started is not None or
                     (telemetry['speed']<5. and abs(float(np.interp(6.,plan.path[:,0],plan.path[:,1])))>1.2))):
                aligned=self.surface_guard.apply(plan,scene,telemetry,now)
                if aligned.reason=='lane_alignment':plan=aligned
            plan=self.learned_prior.apply(plan,model,scene,telemetry['speed'])
            nominal_path=plan.path.copy()
            plan=self.failure_guard.apply(plan,scene,telemetry)
            surface_stop=plan.reason if plan.reason in ('offroad_surface','sidewalk_boundary') else None
            from pathlib import Path
            from .forza_local_learning import write_json
            import time
            write_json(Path(__file__).resolve().parents[1]/'out/surface-state.json',
                       dict(time=time.time(),**self.surface_guard.status))
            if self.memory.state!='rejected_jump' and not surface_stop:
                self.continuation.observe(plan,telemetry,now)
            # The old grid alarm cannot overrule a complete independently
            # confirmed clear corridor. Close/unseen ranges remain a veto.
            from .forza_legacy_veto import obstacle_veto
            veto,self.legacy_veto_status=obstacle_veto(model,plan,scene,telemetry)
            if veto and not surface_stop:
                return None
            from .forza_collision import avoid_in_lane
            plan = avoid_in_lane(plan,scene['cars'],telemetry['speed'],scene.get('raw_road',scene['road']))
            from .forza_static import avoid_static
            plan = avoid_static(plan,scene,telemetry['speed'])
            avoiding=plan.reason in ('static_avoidance','vehicle_avoidance','remembered_avoidance')
            # Avoidance may change both path and wheel angle. Validate the final
            # requested motion, not the pre-avoidance plan.
            if plan.reason not in ('intersection_turn','offroad_surface','sidewalk_boundary'):
                boundary_reference=plan
                if len(plan.path)>=3 and plan.lane_edges:
                    plan = self.predictive.apply(plan,telemetry,now,self.memory.uncertainty)
                plan = self.departure_guard.apply(plan,telemetry['speed'])
                if (plan.guard_state=='recentering' and plan.quality<.75 and not avoiding and
                        not self.surface_guard.status.get('corrected') and
                        self.route_memory.status.get('action')!='successful_route'):
                    # Weak lane estimates are a prior, not a reason to park.
                    from .forza_minimap import road_hint
                    from .forza_continuation import retain_observed_lane
                    inferred=self.continuation.predict(scene,telemetry,now,hint=road_hint(planning_frame))
                    plan=retain_observed_lane(inferred,boundary_reference,telemetry['speed'])
                    if len(plan.path)>=3 and plan.lane_edges:
                        plan=self.predictive.apply(plan,telemetry,now,self.memory.uncertainty)
                        plan=self.departure_guard.apply(plan,telemetry['speed'])
                    self.memory.state='inferred'
                if avoiding and plan.guard_state in ('correcting','recentering'):
                    # If an evasive wheel angle is rejected, its optimistic
                    # collision-free path must not survive the rejection.
                    centre=nominal_path
                    from .forza_scene import traffic_limit
                    traffic_max=traffic_limit(centre,scene['cars'],telemetry['speed'])[0]
                    plan=replace(plan,path=centre,limit=min(plan.road_limit if plan.road_limit is not None else plan.limit,traffic_max),reason='lane_guard',
                                 image_path=tuple(project(float(x),float(z)) for z,x in centre))
                if plan.guard_state in ('boundary_unknown','invalid_boundary','short_boundary'):
                    from .forza_continuation import continue_unmarked
                    plan=continue_unmarked(plan,telemetry['speed'])
                else:
                    self.boundary_unknown_since=None
            # Re-evaluate the chosen path; an earlier rejected path must not keep
            # braking after a clear in-lane steering alternative has been found.
            from .forza_scene import traffic_limit
            road_limit = plan.road_limit if plan.road_limit is not None else plan.limit
            traffic_max=traffic_limit(plan.path,scene['cars'],telemetry['speed'])[0]
            plan = replace(plan,limit=min(plan.limit,road_limit,traffic_max),
                           reason='traffic_stop' if traffic_max<1. else plan.reason)
            if surface_stop:
                plan=replace(plan,limit=0.,road_limit=0.,reason=surface_stop,guard_state=surface_stop)
            from .forza_carriageway import path_crosses_median
            if path_crosses_median(plan.path,self.carriageway.median):
                plan=replace(plan,limit=0.,road_limit=0.,reason='median_boundary',guard_state='median_boundary')
            if self.carriageway_status.get('wrongway_confirmed'):
                plan=replace(plan,limit=0.,road_limit=0.,reason='wrong_way',guard_state='wrong_way')
            plan=self.surface_guard.apply(plan,scene,telemetry,now,correct=False)
            plan=self.failure_guard.apply(plan,scene,telemetry,final=True)
            plan=self.temporal.update(frame,telemetry,plan,scene,now)
            reference=self.route_memory.last_path
            self.route_memory.status['retained']=bool(reference is not None and
                reference.shape==plan.path.shape and np.allclose(reference,plan.path,atol=.05))
            write_json(Path(__file__).resolve().parents[1]/'out/failure-memory-state.json',
                       dict(time=time.time(),route_memory=self.route_memory.status,**self.failure_guard.status))
            self.scene_planner.latest = plan
            learned=self.learned_prior.last_path
            self.learned_prior.status['retained']=bool(learned is not None and
                learned.shape==plan.path.shape and np.allclose(learned,plan.path,atol=.01))
            path = plan.image_path or tuple(project(float(x),float(z)) for z,x in plan.path)
            near = min(range(len(plan.path)),key=lambda i:abs(plan.path[i,0]-8))
            far = min(range(len(plan.path)),key=lambda i:abs(plan.path[i,0]-16))
            target = path[far][0]
            lane = LaneEstimate(plan.quality,target,target,.1,plan.steer,
                                (target,path[near][0]),(target,path[near][0]),path,tuple(scene['cars']))
            return Guidance(lane,plan.steer,plan.quality,plan.limit,plan.reason,
                            float(plan.path[-1,0]),plan.vehicles,plan.candidate_count,
                            plan.uncertainty_limited and plan.reason=='lane_inferred' and
                            plan.limit>0 and plan.limit>= (plan.road_limit or plan.limit)-.01)
        speed, yaw = telemetry['speed'], telemetry['yaw']
        if not all(math.isfinite(v) for v in (speed, yaw, now)):
            return None
        dt = max(0., min(.5, now-self.updated)) if self.updated else 0.
        self.updated = now
        self.travelled += speed/3.6*dt
        age = now-self.seen
        prior = None
        if self.last is not None and age < 4. and self.travelled < 8.:
            delta = math.atan2(math.sin(yaw-self.yaw), math.cos(yaw-self.yaw))
            shift = -.5*delta  # 90-degree horizontal field of view
            prior = replace(self.last, center_x=self.last.center_x+shift,
                            target_x=self.last.target_x+shift,
                            center=tuple(x+shift for x in self.last.center),
                            edge=tuple(x+shift for x in self.last.edge))
        lane = estimate_left_lane(frame, prior)
        lookahead, limit = self.preview(model, speed)
        # The original depth/obstacle planner remains an independent stop veto.
        if model.reverse or model.mode == 'blocked' or (model.blocked_m is not None and model.blocked_m < 6):
            return None
        if lane is not None:
            lane = replace(lane, steer=lane.steer/(1.+max(0., speed)/35.))
            turn = abs(math.tan(lane.steer*math.radians(35))/2.7)
            limit = min(limit, math.sqrt(.2*9.81/max(.001, turn))*3.6)
            measured_bias = max(-1., min(1., lane.steer-model.steer))
            self.bias = measured_bias if self.bias is None else .8*self.bias+.2*measured_bias
            self.last, self.seen, self.yaw, self.travelled = lane, now, yaw, 0.
            return Guidance(lane, lane.steer, lane.confidence, limit, 'left_lane', lookahead)
        if prior is None:
            return None
        if not road_patch(frame, prior.target_x, 255/540):
            return None
        odometry_steer = max(-.45, min(.45, (prior.target_x-.5)*2.8))
        if age <= .75:
            steer, source = odometry_steer, 'lane_odometry'
        elif age <= 4. and model.confidence >= .35 and model.mode not in ('lost', 'blocked') and self.bias is not None:
            corrected = max(-.45, min(.45, model.steer+self.bias))
            steer, source = .7*odometry_steer+.3*corrected, 'lane_model'
        else:
            return None
        prior = replace(prior, steer=steer, confidence=.55)
        return Guidance(prior, steer, .55, min(limit, 25.), source, lookahead)


def _road_fraction(hsv, x, y):
    x, y = int(round(x)), int(round(y))
    h, w = hsv.shape[:2]
    if not 2 <= x < w-2 or not 2 <= y < h-2:
        return 0.
    p = hsv[y-2:y+3, x-2:x+3]
    return float(((p[:, :, 1] < 130) & (p[:, :, 2] > 15) & (p[:, :, 2] < 155)).mean())


def road_patch(frame, x, y=.50):
    """Reject textured bright snow/vegetation at a proposed driving point."""
    im = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    x, y = int(x*960), int(y*540)
    if not 255 <= x <= 925 or not 225 <= y <= 315:
        return False
    patch = cv2.cvtColor(im[y-9:y+10, x-14:x+15], cv2.COLOR_BGR2HSV)
    value = patch[:, :, 2]
    return bool(15 < np.median(value) < 145 and np.percentile(value, 80) < 180
                and value.std() < 48 and np.median(patch[:, :, 1]) < 135)


def road_direction(frame):
    """Choose a visible asphalt patch for a short recovery manoeuvre."""
    available = [x for x in np.linspace(.28, .88, 25) if road_patch(frame, x, .52)]
    if not available:
        return 0.
    return float(max(-.6, min(.6, (np.median(available)-.5)*3)))


def paint_groups(frame):
    """Fit paint segments in the visible road strip; reject roofs, sky and dashboard."""
    im = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    local = cv2.blur(value, (31, 31))
    white = (hsv[:, :, 1] < 120) & ((value > 163) | ((value > 60) & (value-local > 22)))
    yellow = (hsv[:, :, 0] > 12) & (hsv[:, :, 0] < 38) & (hsv[:, :, 1] > 95) & (hsv[:, :, 2] > 120)
    mask = ((white | yellow).astype(np.uint8) * 255)
    mask[:235] = 0
    mask[315:] = 0
    mask[:, :235] = 0
    mask[:, 935:] = 0
    lines = cv2.HoughLinesP(mask, 1, np.pi/360, 12, minLineLength=18, maxLineGap=14)
    groups = []
    if lines is None:
        return groups
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dy, dx = float(y2-y1), float(x2-x1)
        if abs(dy) < 8 or abs(dx/dy) > 5:
            continue
        slope, intercept = dx/dy, float(x1)-dx/dy*y1
        near, far = slope*305+intercept, slope*255+intercept
        if not -150 < near < 1200 or not 160 < far < 900:
            continue
        left, right = [], []
        for t in np.linspace(.15, .85, 7):
            x, y = x1+t*dx, y1+t*dy
            left.append(_road_fraction(hsv, x-9, y))
            right.append(_road_fraction(hsv, x+9, y))
        left, right = float(np.mean(left)), float(np.mean(right))
        if max(left, right) < .58:
            continue
        length = math.hypot(dx, dy)
        candidate = dict(near=near, far=far, left=left, right=right, weight=length,
                         ys=[int(y1), int(y2)], count=1)
        group = next((g for g in groups if abs(g['near']-near) < 28 and abs(g['far']-far) < 22), None)
        if group is None:
            groups.append(candidate)
        else:
            total = group['weight']+length
            for key in ('near', 'far', 'left', 'right'):
                group[key] = (group[key]*group['weight']+candidate[key]*length)/total
            group['weight'] = total
            group['ys'].extend(candidate['ys'])
            group['count'] += 1
    for g in groups:
        ys = np.arange(max(235, min(g['ys'])), min(315, max(g['ys']))+1)
        xs = g['far']+(ys-255)*(g['near']-g['far'])/50
        xi = np.clip(np.round(xs).astype(int), 3, 956)
        present = np.array([mask[y, x-3:x+4].any() for x, y in zip(xi, ys)])
        g['continuity'] = float(present.mean()) if len(present) else 1.
        g['yellow'] = float(yellow[ys, xi].mean()) if len(ys) else 0.
    return sorted(groups, key=lambda g: g['weight'], reverse=True)


def estimate_left_lane(frame, previous=None):
    groups = paint_groups(frame)
    # The dividing line has asphalt on both sides. A snow bank/curb does not.
    centers = [g for g in groups if min(g['left'], g['right']) >= .58 and g['count'] >= 2
               and g['weight'] >= 80 and max(g['ys'])-min(g['ys']) >= 16]
    candidates = []
    for center in centers:
        for edge in groups:
            near_width = edge['near']-center['near']
            far_width = edge['far']-center['far']
            if not 80 < abs(near_width) < 650 or not 20 < abs(far_width) < 250:
                continue
            if near_width*far_width <= 0 or abs(near_width) < abs(far_width)*1.1:
                continue
            # Road-edge paint has road on its inward side, as do solid center markings.
            inward = edge['left'] if near_width > 0 else edge['right']
            if inward < .65 or max(edge['ys'])-min(edge['ys']) < 22:
                continue
            width = abs(far_width)
            target = center['far']-.5*width
            if not 120 < target < 850:
                continue
            if not road_patch(frame, target/960, 255/540):
                continue
            score = min(1., center['weight']/95) * min(center['left'], center['right']) * inward
            if previous is not None:
                distance = abs(center['far']/960-previous.center_x)
                if distance > .10:
                    continue
                score *= max(.60, 1.-distance*4)
            if score >= .45:
                candidates.append((score, center, edge, target, width))
    if not candidates:
        return None
    # The leftmost interior marking bounds the leftmost driving lane. Track the
    # same marking once acquired; use its nearest neighbour for one lane's width.
    def order(item):
        score, center, edge, target, width = item
        identity = abs(center['far']/960-previous.center_x) if previous else center['far']/960
        return (round(identity, 4), width, -score)
    score, center, edge, target, width = min(candidates, key=order)
    # Aim at the center of the left lane; cap steering while this detector is experimental.
    error = (target-480)/480
    steer = max(-.5, min(.5, error*1.4))
    return LaneEstimate(min(1., score+.15), target/960, center['far']/960, width/960, steer,
                        (center['far']/960, center['near']/960),
                        (edge['far']/960, edge['near']/960))


def draw_lane(frame, lane, source='left_lane'):
    out = frame.copy()
    if lane is None:
        cv2.putText(out, 'LEFT LANE: waiting for road boundaries', (28, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 190, 255), 2)
        return out
    h, w = out.shape[:2]
    if lane.path:
        points = np.array([[int(u*w),int(v*h)] for u,v in lane.path],dtype=np.int32)
        cv2.polylines(out,[points],False,(40,230,70),4)
        for car in lane.cars:
            x1,y1,x2,y2 = car['box']
            cv2.rectangle(out,(int(x1*w),int(y1*h)),(int(x2*w),int(y2*h)),(0,160,255),3)
        cv2.putText(out,f'{source} | cars {len(lane.cars)} | steer {lane.steer:+.2f}',
                    (28,70),cv2.FONT_HERSHEY_SIMPLEX,.8,(40,230,70),2)
        return out
    for line, color in ((lane.center, (0, 160, 255)), (lane.edge, (255, 220, 0))):
        cv2.line(out, (int(line[0]*w), int(255/540*h)),
                 (int(line[1]*w), int(305/540*h)), color, 4)
    cv2.circle(out, (int(lane.target_x*w), int(255/540*h)), 12, (40, 230, 70), -1)
    cv2.putText(out, f'{source.upper()} {lane.confidence:.2f} steer {lane.steer:+.2f}', (28, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 230, 70), 2)
    return out
