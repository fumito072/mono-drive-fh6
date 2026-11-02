"""Avoid evidence-backed previous contacts without learning a steering bias."""
from dataclasses import replace
import math
from pathlib import Path
import numpy as np

from .forza_local_learning import read_json
from .forza_lane_memory import rotation


class FailedApproachGuard:
    def __init__(self,path=None):
        self.path=Path(path) if path else Path(__file__).resolve().parents[1]/'out/failure-memory.json'
        self.version=None
        self.entries=[]
        self.status={}
        self.route_memory=None

    def load(self):
        try:
            version=self.path.stat().st_mtime_ns
            if version==self.version:
                return
            value=read_json(self.path,{})
            entries=[]
            for entry in value.get('entries',[])[-200:]:
                if entry.get('excluded_reason'):
                    continue
                point=np.asarray(entry.get('point'),dtype=float)
                if (point.shape==(2,) and np.isfinite(point).all() and
                        math.isfinite(float(entry.get('yaw',float('nan')))) and
                        entry.get('confidence',0)>=.7):
                    safe=np.asarray(entry.get('safe_path') or [],dtype=float)
                    if safe.ndim!=2 or safe.shape[1]!=2 or not np.isfinite(safe).all():
                        safe=np.empty((0,2))
                    entries.append(dict(entry,safe_path=safe.tolist()))
            self.entries,self.version=entries,version
        except (OSError,ValueError,TypeError):
            pass

    def local_entries(self,ego):
        position=np.asarray(ego['position'])[[0,2]]
        inverse=rotation(ego['yaw']).T
        result=[]
        for entry in self.entries:
            if abs(entry.get('height',ego['position'][1])-ego['position'][1])>3.5:
                continue
            turn=math.atan2(math.sin(entry['yaw']-ego['yaw']),math.cos(entry['yaw']-ego['yaw']))
            if abs(turn)>math.pi/2:
                continue
            x,z=inverse@(np.asarray(entry['point'])-position)
            if 1.5<z<35. and abs(x)<20.:
                result.append((entry,float(z),float(x)))
        return result

    @staticmethod
    def conflicts(path,entries):
        from .forza_traffic import path_x
        return [(entry,z,x) for entry,z,x in entries
                if abs(float(path_x(path,z))-x)<1.35]

    def apply(self,plan,scene,ego,*,final=False):
        self.load()
        entries=self.local_entries(ego)
        conflicts=self.conflicts(plan.path,entries)
        if not conflicts:
            if not final:
                self.status=dict(action='clear',known_contacts=len(self.entries))
            elif self.status.get('action')=='known_failed_approach':
                self.status['action']='cleared_by_other_guard'
            return plan
        identifiers=[str(entry.get('id')) for entry,_,_ in conflicts]
        self.status=dict(action='known_failed_approach',incidents=identifiers,
                         known_contacts=len(self.entries))
        if not final and (plan.lane_edges or self.route_memory is not None):
            from .forza_scene import inside_lane,footprint_quality,traffic_limit,project
            from .forza_static import static_risk
            from .forza_lateral import lookahead_target
            edges=np.asarray(plan.lane_edges) if plan.lane_edges else None
            proposals=[]
            if self.route_memory is not None:
                proposals.extend((0.,candidate) for _,_,candidate in self.route_memory.candidates(ego))
            for offset in ((-.25,.25,-.5,.5,-.75,.75) if edges is not None else ()):
                candidate=plan.path.copy(); candidate[:,1]+=offset
                proposals.append((abs(offset),candidate))
            for entry,_,_ in (conflicts if edges is not None else ()):
                world=np.asarray(entry.get('safe_path') or [],dtype=float)
                if world.ndim==2 and world.shape[1]==2 and len(world)>2:
                    local=(rotation(ego['yaw']).T@(world-np.asarray(ego['position'])[[0,2]]).T).T
                    local=local[(local[:,1]>2.5)&(local[:,1]<35.)][:,[1,0]]
                    local=local[np.argsort(local[:,0])]
                    if len(local)>2 and np.all(np.diff(local[:,0])>.05):
                        z=plan.path[:,0]
                        z=z[(z>=local[0,0])&(z<=local[-1,0])]
                        if len(z)>=3 and z[-1]-z[0]>=8.:
                            candidate=np.column_stack((z,np.interp(z,local[:,0],local[:,1])))
                            proposals.append((0.,candidate))
            for _,candidate in sorted(proposals,key=lambda item:item[0]):
                if self.conflicts(candidate,entries):
                    continue
                if edges is not None:
                    left=np.interp(candidate[:,0],edges[:,0],edges[:,1])
                    right=np.interp(candidate[:,0],edges[:,0],edges[:,2])
                    if not inside_lane(candidate,left,right):continue
                from .forza_lane_memory import align_plan
                observed=align_plan(replace(plan,path=candidate),ego,scene.get('ego') or ego)
                if observed is None:
                    continue
                floor,average=footprint_quality(scene.get('raw_road',scene['road']),observed.path)
                if floor<.45 or average<.7:
                    continue
                _,_,steer=lookahead_target(candidate,ego['speed'])
                if static_risk(scene,ego['speed'],steer,path=observed.path).clearance_m<10.:
                    continue
                traffic,_=traffic_limit(candidate,scene.get('cars',[]),ego['speed'])
                self.status['action']='validated_alternative'
                return replace(plan,path=candidate,steer=steer,limit=min(plan.limit,traffic),
                    reason='remembered_avoidance',inferred=True,
                    image_path=tuple(project(float(x),float(z)) for z,x in candidate))
        # Do not repeat a failed approach just because a fresh frame has no
        # lane markings. The session diagnoses this stop and uses its verified
        # road recovery; no unchecked turn or menu sequence is invented here.
        return replace(plan,limit=0.,road_limit=0.,reason='known_failed_approach',inferred=True)
