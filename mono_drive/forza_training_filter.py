"""Spatial support for already filtered forward-driving trajectory segments."""
import numpy as np


def training_row_valid(row):
    """Measured paint or a repeatedly verified retained route; never blind motion."""
    path=row.get('candidate_path') or []
    route=row.get('route_memory') or {}
    trusted_route=(route.get('retained') is True and route.get('consensus') is True and
                   route.get('agreeing_traversals',0)>=2 and (row.get('lane_clearance_m') or 0)>.35)
    mode=row.get('mode')
    quality=(mode=='road_candidates' and row.get('confidence',0)>=.88) or (
        mode in ('paint_corridor','paint_pair') and row.get('confidence',0)>=.80) or trusted_route
    surface=row.get('surface') or {}
    return bool(quality and row.get('armed') and 6<=row.get('forward_kmh',0)<=55 and
        row.get('actual_brake',255)<80 and row.get('race_position',1)==0 and
        path and abs(path[0][1])<.65 and not row.get('recovery_phase') and
        (not row.get('inferred_lane') or trusted_route) and
        not surface.get('offroad_confirmed') and not surface.get('path_enters_sidewalk') and
        not (row.get('carriageway') or {}).get('wrongway_confirmed') and
        (row.get('static_clearance_m') or 50.)>8. and
        row.get('mode') not in ('known_failed_approach','offroad_surface','sidewalk_boundary','collision_brake'))


def segment_geometry(group):
    # Two frames per second made a fixed 30-frame minimum reject even long,
    # clean trajectories. Shorter sequences need greater spatial support.
    if len(group)<10:
        return None
    positions=np.asarray([row['position'] for row in group],dtype=float)
    if positions.shape!=(len(group),3) or not np.isfinite(positions).all():
        return None
    steps=np.linalg.norm(np.diff(positions,axis=0),axis=1)
    distance=float(steps.sum())
    required=22. if len(group)>=30 else 45.
    if distance<required or np.any(steps>8.):
        return None
    grade=float(abs(positions[-1,1]-positions[0,1])/distance)
    if grade>.10:
        return None
    return dict(distance_m=distance,grade=grade)
