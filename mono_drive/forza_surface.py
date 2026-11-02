"""Keep inferred driving paths inside road, independently of lane paint."""
from dataclasses import replace
import math
import numpy as np

from .forza_scene import bounds, project, road_value, road_geometry_through_traffic
from .forza_lane_memory import align_plan
from .forza_vehicle import steering_angle


def constrain_lane_edges(plan,road):
    """Paint-derived lane limits cannot extend through a visible road edge."""
    rows=bounds(road)
    if not plan.lane_edges or len(rows)<2:
        return plan
    edges=np.asarray(plan.lane_edges,dtype=float).copy()
    for i,(z,left,right) in enumerate(edges):
        if rows[0,0]<=z<=rows[-1,0]:
            a=max(left,float(np.interp(z,rows[:,0],rows[:,1])))
            b=min(right,float(np.interp(z,rows[:,0],rows[:,2])))
            if a<b:
                edges[i]=z,a,b
    return replace(plan,lane_edges=tuple(map(tuple,edges)))


def surface_samples(mask, path, distances, offsets=(0.,),visibility=None):
    if mask is None:
        return np.array([],dtype=float)
    from .forza_traffic import path_x
    x=path_x(path,np.asarray(distances))
    samples=[]
    for z,px in zip(distances,x):
        for dx in offsets:
            if visibility is not None:
                u,v=project(float(px+dx),float(z))
                y,xp=round(v*(visibility.shape[0]-1)),round(u*(visibility.shape[1]-1))
                if not (0<=y<visibility.shape[0] and 0<=xp<visibility.shape[1] and visibility[y,xp]):continue
            samples.append(road_value(mask,float(px+dx),float(z)))
    return np.asarray(samples)


class RoadSurfaceGuard:
    def __init__(self):
        self.offroad_since=None
        self.status={}
        self.alignment_started=None

    def apply(self,plan,scene,ego,now,*,correct=True):
        road=road_geometry_through_traffic(scene.get('raw_road',scene['road']),scene.get('cars',[]))
        sidewalk=scene.get('sidewalk')
        source=scene.get('ego') or ego
        observed=align_plan(plan,ego,source)
        if observed is None:
            return plan
        if correct:
            constrained=constrain_lane_edges(observed,road)
            if constrained.lane_edges!=observed.lane_edges:
                aligned=align_plan(constrained,source,ego)
                if aligned is not None:
                    plan,observed=aligned,constrained
        near=np.array([[3.5,0.],[4.5,0.],[6.,0.]])
        visibility=scene.get('planning_roi')
        road_near=surface_samples(road,near,near[:,0],(-.6,0.,.6),visibility)
        walk_near=surface_samples(sidewalk,near,near[:,0],(-.6,0.,.6),visibility)
        offroad=bool(len(walk_near) and np.mean(walk_near>.65)>=.55 and
                     np.mean(road_near>.45)<.34)
        self.offroad_since=(self.offroad_since if self.offroad_since is not None else now) if offroad else None
        confirmed=offroad and now-self.offroad_since>=.4
        chosen_walk=surface_samples(sidewalk,observed.path,(4.,6.,8.,10.,12.),(-.8,0.,.8),visibility)
        enters_walk=bool(len(chosen_walk) and np.count_nonzero(chosen_walk>.65)>=3)
        self.status=dict(offroad_confirmed=confirmed,road_support=float(np.mean(road_near)) if len(road_near) else 0.,
                         sidewalk_support=float(np.mean(walk_near)) if len(walk_near) else None,
                         path_enters_sidewalk=enters_walk,
                         corrected=self.status.get('corrected',False) if not correct else False)
        if offroad:
            # Do not keep driving along the sidewalk while awaiting the session's
            # verified paved-road recovery. Unknown surfaces alone do not stop us.
            return replace(plan,limit=0.,road_limit=0.,reason='offroad_surface',guard_state='offroad_surface',inferred=True)
        if correct and observed.lane_edges and len(road_near) and np.mean(road_near)>.5:
            # A verified recovery can spawn in the adjacent same-direction
            # lane. Requiring the whole car to be in the left lane before it
            # is allowed to move creates a permanent geometry stop.
            edges=np.asarray(observed.lane_edges)
            middle=(edges[:,1]+edges[:,2])/2.
            outside=edges[0,1]>-.4 or edges[0,2]<.4
            aligned=abs(float(middle[0]))<.45
            if aligned or (self.alignment_started is not None and now-self.alignment_started>12.):
                self.alignment_started=None
            if (outside and ego['speed']<5.) or self.alignment_started is not None:
                rows=bounds(road)
                # A walking-speed merge does not need a guessed road envelope
                # sixty metres away. Validate its measured near corridor only.
                rows=rows[rows[:,0]<=min(22.,float(observed.path[-1,0]))]
                if len(rows)>=3 and rows[0,1]<-1.1 and rows[0,2]>1.1:
                    z=rows[:,0]
                    target=np.interp(z,edges[:,0],middle)
                    ratio=np.clip(z/18.,0.,1.)
                    path=np.column_stack((z,target*ratio*ratio*(3.-2.*ratio)))
                    from .forza_scene import inside_lane
                    from .forza_lateral import lookahead_target
                    support=surface_samples(road,path,z[:5],(-.93,0.,.93),visibility)
                    if inside_lane(path,rows[:,1],rows[:,2]) and np.mean(support>.6)>.85:
                        self.alignment_started=now if self.alignment_started is None else self.alignment_started
                        _,_,steer=lookahead_target(path,12.)
                        merged=replace(observed,path=path,steer=steer,limit=min(12.,plan.limit),
                            lane_edges=tuple(map(tuple,rows)),reason='lane_alignment',inferred=True,
                            image_path=tuple(project(float(x),float(d)) for d,x in path))
                        self.status['corrected']=True
                        self.status['lane_alignment']=True
                        return align_plan(merged,source,ego) or plan
        if not (plan.inferred or enters_walk):
            return plan
        if not correct:
            return (replace(plan,limit=0.,road_limit=0.,reason='sidewalk_boundary',guard_state='sidewalk_boundary',inferred=True)
                    if enters_walk else plan)
        rows=bounds(road)
        rows=rows[rows[:,0]<=30.] if len(rows) else rows
        if len(rows)>=3 and rows[0,0]<=8.:
            z,left,right=rows.T
            lane_right=left+np.minimum(3.6,(right-left)/2.)
            # Only the near section contains the current car. Requiring every
            # distant section to straddle x=0 would reject every genuine bend.
            usable=right-left>=3.3
            near_contains=left[0]<.25 and lane_right[0]>.98
            if (plan.inferred and not near_contains and left[0]<-1.1 and right[0]>1.1
                    and np.min(right[:3]-left[:3])>=5.8 and len(road_near)
                    and np.mean(road_near)>.65):
                # A lost-paint fallback used to keep the current heading in
                # the adjacent lane forever. Merge toward the measured left
                # lane over several seconds, inside the full road envelope.
                distance=max(24.,ego['speed']/3.6*3.)
                ratio=np.clip(z/distance,0.,1.)
                centre=(left+lane_right)/2.
                path=np.column_stack((z,centre*ratio*ratio*(3.-2.*ratio)))
                from .forza_scene import inside_lane,curve_speed
                from .forza_lateral import lookahead_target
                support=surface_samples(road,path,z[z<=18.],(-.93,0.,.93),visibility)
                if (len(support)>=9 and np.mean(support>.6)>.85 and inside_lane(path,left,right)):
                    _,_,steer=lookahead_target(path,ego['speed'])
                    merge=replace(observed,path=path,steer=steer,
                        limit=min(plan.limit,40.,curve_speed(path,ego['speed'])[0]),
                        lane_edges=tuple(map(tuple,rows)),reason='lane_alignment',guard_state='',
                        image_path=tuple(project(float(x),float(d)) for d,x in path))
                    aligned=align_plan(merge,source,ego)
                    if aligned is not None:
                        self.status.update(corrected=True,lane_alignment=True)
                        return aligned
            if near_contains and np.median(right[:3]-left[:3])>=4.2:
                end=next((i for i in range(2,len(rows)) if not usable[i]),len(rows))
                z,left,lane_right=z[:end],left[:end],lane_right[:end]
                if len(z)>=3:
                    path=np.column_stack((z,(left+lane_right)/2.))
                    lookahead=float(np.clip(8.+ego['speed']/7.2,z[0],z[-1]))
                    target=float(np.interp(lookahead,z,path[:,1]))
                    curvature=2*target/(lookahead*lookahead+target*target)
                    steer=float(np.clip(math.atan(2.64*curvature)/steering_angle(ego['speed']),-.55,.55))
                    corrected=replace(observed,path=path,steer=steer,reason='road_envelope',
                        lane_edges=tuple(map(tuple,np.column_stack((z,left,lane_right)))),
                        image_path=tuple(project(float(x),float(d)) for d,x in path),guard_state='')
                    corrected=align_plan(corrected,source,ego)
                    if corrected is not None:
                        self.status['corrected']=True
                        return corrected
            if plan.inferred and left[0]<-.98 and right[0]>.98:
                # Keep the last lane-centre intention, but never discard the
                # visible outer road boundary just because a guessed lane
                # width does not fit. Its midpoint is not a new driving goal.
                bounded=replace(observed,lane_edges=tuple(map(tuple,rows)),
                                reason='road_bounds_only',guard_state='')
                bounded=align_plan(bounded,source,ego)
                if bounded is not None:
                    self.status['corrected']=True
                    return bounded
        if enters_walk:
            return replace(plan,limit=0.,road_limit=0.,reason='sidewalk_boundary',guard_state='sidewalk_boundary',inferred=True)
        return plan
