"""Track one lane in ego coordinates and reject isolated geometric jumps."""
from dataclasses import replace
import math
import numpy as np


def rotation(yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,s],[-s,c]])


def align_plan(plan, source, target):
    if plan is None or source is None:
        return plan
    transform=rotation(target['yaw']).T@rotation(source['yaw'])
    shift=rotation(target['yaw']).T@(np.asarray(source['position'])[[0,2]]-np.asarray(target['position'])[[0,2]])
    def points(z,x):
        moved=(transform@np.stack((x,z))+shift[:,None]).T
        order=np.argsort(moved[:,1])
        return moved[order][:,[1,0]]
    path=points(plan.path[:,0],plan.path[:,1])
    valid=(path[:,0]>2.5)&(path[:,0]<50.)
    path=path[valid]
    if len(path)<2 or np.any(np.diff(path[:,0])<.05):
        return None
    edges=()
    if plan.lane_edges:
        e=np.asarray(plan.lane_edges)
        l,r=points(e[:,0],e[:,1]),points(e[:,0],e[:,2])
        z=path[:,0]
        edges=tuple(map(tuple,np.column_stack((z,np.interp(z,l[:,0],l[:,1]),np.interp(z,r[:,0],r[:,1])))))
    from .forza_scene import project
    return replace(plan,path=path,lane_edges=edges,
                   image_path=tuple(project(float(x),float(z)) for z,x in path))


class TemporalLaneModel:
    def __init__(self):
        self.plan=None
        self.ego=None
        self.seen=0.
        self.uncertainty=.08
        self.state='new'

    def update(self, candidate, ego, now, road=None,road_ego=None):
        previous=align_plan(self.plan,self.ego,ego) if self.plan is not None else None
        fresh=previous is not None and now-self.seen<.65
        if candidate is not None and fresh:
            lo=max(previous.path[0,0],candidate.path[0,0],4.)
            hi=min(previous.path[-1,0],candidate.path[-1,0],15.)
            if hi>lo+1.:
                z=np.linspace(lo,hi,8)
                a=np.interp(z,previous.path[:,0],previous.path[:,1])
                b=np.interp(z,candidate.path[:,0],candidate.path[:,1])
                innovation=float(np.median(np.abs(a-b)))
                if innovation>.65:
                    # An unverified new lane must not overwrite good memory.
                    candidate=None
                    self.uncertainty=min(.4,self.uncertainty+.06)
                    self.state='rejected_jump'
                else:
                    self.uncertainty=.8*self.uncertainty+.2*min(.3,innovation+.04)
                    self.state='tracked'
                    if candidate.lane_edges and previous.lane_edges and innovation<.45:
                        old_edges=np.asarray(previous.lane_edges,dtype=float)
                        new_edges=np.asarray(candidate.lane_edges,dtype=float)
                        nz=new_edges[:,0]
                        old_width=np.interp(nz,old_edges[:,0],old_edges[:,2]-old_edges[:,1])
                        new_width=new_edges[:,2]-new_edges[:,1]
                        if np.max(np.abs(old_width-new_width))<.25:
                            # Fuse only the overlapping, motion-aligned lane.
                            # Keep current widths; fresh road/wall clipping still
                            # follows this step and vetoes obsolete boundaries.
                            old_centre=np.interp(nz,old_edges[:,0],old_edges[:,1:3].mean(axis=1))
                            weight=.35*np.clip((hi-nz)/3.,0.,1.)
                            shift=weight*np.clip(old_centre-new_edges[:,1:3].mean(axis=1),-.3,.3)
                            shift=np.where(nz>=lo,shift,0.)
                            new_edges[:,1:]+=shift[:,None]
                            path=candidate.path.copy()
                            path[:,1]+=np.interp(path[:,0],nz,shift)
                            from .forza_scene import project
                            candidate=replace(candidate,path=path,lane_edges=tuple(map(tuple,new_edges)),
                                image_path=tuple(project(float(x),float(z)) for z,x in path))
        if candidate is None:
            if not fresh:
                self.state='unavailable'
                return None
            if road is not None:
                from .forza_scene import road_value
                observed=align_plan(previous,ego,road_ego) if road_ego is not None else previous
                if observed is None:
                    self.state='unavailable'
                    return None
                visible=[road_value(road,float(x+dx),float(z)) for z,x in observed.path if 4.<z<16.
                         for dx in (-.85,0.,.85)]
                if not visible or np.mean(np.asarray(visible)>.55)<.85:
                    self.state='unavailable'
                    return None
            self.state='bridged' if self.state!='rejected_jump' else self.state
            return replace(previous,limit=min(previous.limit,25.),quality=min(previous.quality,.7),reason='lane_temporal')
        self.plan,self.ego,self.seen=candidate,dict(position=list(ego['position']),yaw=ego['yaw']),now
        return candidate
