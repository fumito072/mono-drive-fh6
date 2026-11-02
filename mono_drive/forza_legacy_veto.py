"""Reconcile the older obstacle grid with a separately measured road corridor."""
import math
import numpy as np


def obstacle_veto(model,plan,scene,ego):
    """Keep an uncorroborated grid alarm unless its full claimed range is clear."""
    if getattr(model,'mode',None)!='blocked' or getattr(model,'confidence',0)<.8:
        return False,'not_requested'
    distance=getattr(model,'blocked_m',None)
    if (not isinstance(distance,(int,float)) or not math.isfinite(distance) or distance<4. or
            plan is None or plan.quality<.75 or not plan.lane_edges or
            (plan.inferred and not plan.verified_route) or plan.limit<=0):
        return True,'unverified_corridor'
    from .forza_lane_memory import align_plan
    observed=align_plan(plan,ego,scene.get('ego') or ego)
    end=max(12.,distance+3.)
    if observed is None or observed.path[0,0]>6. or observed.path[-1,0]<end:
        return True,'insufficient_visible_range'
    z=np.linspace(max(4.,observed.path[0,0]),end,12)
    path=np.column_stack((z,np.interp(z,observed.path[:,0],observed.path[:,1])))
    from .forza_scene import footprint_quality
    floor,mean=footprint_quality(scene.get('raw_road',scene['road']),path)
    if floor<.65 or mean<.8:
        return True,'road_not_confirmed'
    from .forza_collision import forward_collision
    traffic=forward_collision(scene.get('cars',[]),ego['speed'],ego.get('steer',0)/127.,path=observed.path)
    from .forza_static import static_risk
    fixed=static_risk(scene,ego['speed'],ego.get('steer',0)/127.,path=observed.path)
    if (traffic.brake>.01 or traffic.emergency or fixed.brake>.01 or fixed.emergency or
            fixed.clearance_m<end):
        return True,'corroborated_obstacle'
    return False,'independent_corridor_clear'
