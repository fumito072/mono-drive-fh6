"""Use successful past traversals as a prior, checked against current vision."""
from dataclasses import replace
import math
from .forza_speed import MAX_CRUISE_KMH
import time
from pathlib import Path
import numpy as np
from .forza_local_learning import read_json
from .forza_lane_memory import rotation,align_plan


class SuccessfulRouteMemory:
    def __init__(self,path=None,refresh_s=0.):
        self.file=Path(path) if path else Path(__file__).resolve().parents[1]/'out/route-memory.json'
        self.version=None
        self.segments=[]
        self.grid={}
        self.times={}
        self.status={}
        self.last_path=None
        self.committed_id=None
        self.committed_at=0.
        self.refresh_s=max(0.,float(refresh_s))
        self.next_refresh=0.
        self.query_key=None
        self.query_result=None

    def load(self):
        now=time.monotonic()
        if now<self.next_refresh:return
        self.next_refresh=now+self.refresh_s
        try:
            version=self.file.stat().st_mtime_ns
            if version==self.version:return
            segments=[]
            data=read_json(self.file,{})
            for item in data.get('segments',[]):
                points=np.asarray(item['points'],dtype=float)
                yaws=np.asarray(item['yaws'],dtype=float)
                if points.ndim==2 and points.shape[1]==3 and len(points)==len(yaws) and len(points)>=5 and np.isfinite(points).all():
                    segments.append((item['id'],points,yaws))
            grid={}
            for index,(_,points,_) in enumerate(segments):
                for x,z in set(map(tuple,np.floor(points[:,[0,2]]/50.).astype(int))):
                    grid.setdefault((x,z),set()).add(index)
            self.times={item['id']:float(item.get('start',0.)) for item in data.get('segments',[])}
            self.segments,self.grid,self.version=segments,grid,version
            self.committed_id=None
            self.query_key=self.query_result=None
        except (OSError,ValueError,KeyError,TypeError):pass

    def candidates(self,ego):
        self.load()
        # The lane planner and the failed-approach check query the same pose.
        # Reuse geometry for that exact pose only; later camera safety checks
        # still run independently against the current scene.
        key=(self.version,tuple(ego['position']),float(ego['yaw']))
        if self.query_key==key:
            return list(self.query_result)
        position=np.asarray(ego['position'],dtype=float)
        found=[]
        gx,gz=np.floor(position[[0,2]]/50.).astype(int)
        nearby=set()
        for dx in (-1,0,1):
            for dz in (-1,0,1):nearby.update(self.grid.get((gx+dx,gz+dz),()))
        for index in nearby:
            identifier,points,yaws=self.segments[index]
            distances=np.linalg.norm(points[:,[0,2]]-position[[0,2]],axis=1)
            index=int(np.argmin(distances))
            if distances[index]>6. or abs(points[index,1]-position[1])>2.5:continue
            heading=abs(math.atan2(math.sin(yaws[index]-ego['yaw']),math.cos(yaws[index]-ego['yaw'])))
            if heading>.45:continue
            local=(rotation(ego['yaw']).T@(points[max(0,index-1):,[0,2]]-position[[0,2]]).T).T
            local=local[local[:,1]>3.5][:,[1,0]]
            # The part after the apex of a sharp turn can face behind today's
            # camera. It must not invalidate the usable forward arc.
            ends=np.flatnonzero((local[:,0]>=40.)|np.r_[False,np.diff(local[:,0])<.1])
            if len(ends):local=local[:int(ends[0])]
            if len(local)<4 or local[-1,0]-local[0,0]<10.:continue
            found.append((float(distances[index]+heading*4),identifier,local))
        self.query_key=key
        self.query_result=sorted(found,key=lambda item:item[0])[:8]
        return list(self.query_result)

    @staticmethod
    def validate(candidate,plan,scene,ego,*,enforce_lane=True):
        from .forza_scene import inside_lane,footprint_quality,road_geometry_through_traffic
        if plan.lane_edges and enforce_lane:
            edges=np.asarray(plan.lane_edges)
            left=np.interp(candidate[:,0],edges[:,0],edges[:,1])
            right=np.interp(candidate[:,0],edges[:,0],edges[:,2])
            if not inside_lane(candidate,left,right):return False
        observed=align_plan(replace(plan,path=candidate),ego,scene.get('ego') or ego)
        if observed is None:return False
        # Validate the visible near-term body corridor. Far pixels and traffic
        # silhouettes cannot erase a measured route through an intersection.
        road=road_geometry_through_traffic(scene.get('raw_road',scene['road']),scene.get('cars',[]))
        near=observed.path[observed.path[:,0]<=18.]
        if len(near)<3:return False
        low,average=footprint_quality(road,near)
        if low<.45 or average<.7:return False
        from .forza_static import static_risk
        return static_risk(scene,ego['speed'],0.,path=observed.path).clearance_m>8.

    def apply(self,plan,scene,ego):
        self.status=dict(action='current_lane',segments=len(self.segments))
        self.last_path=None
        if plan.reason in ('offroad_surface','sidewalk_boundary','median_boundary','wrong_way'):
            return plan
        if (not plan.inferred and plan.reason in ('paint_pair','paint_corridor') and
                plan.quality>=.8 and plan.painted_boundaries>=5 and len(plan.lane_edges)>=3):
            edges=np.asarray(plan.lane_edges,float)
            near=edges[np.argmin(np.abs(edges[:,0]-6.))]
            width=near[2]-near[1];centre=(near[1]+near[2])/2.
            if (2.6<=width<=4.8 and abs(centre)<=1. and
                    self.validate(plan.path,plan,scene,ego,enforce_lane=True)):
                self.status=dict(action='current_painted_lane',segments=len(self.segments))
                return plan
        candidates=self.candidates(ego)
        now=time.monotonic()
        candidates.sort(key=lambda item:(item[1]!=self.committed_id,item[0]))
        for _,identifier,candidate in candidates:
            # A previously adopted route is not permission to jump sideways
            # into its corridor. Reject adjacent-lane/old-pose near points even
            # when the same route was useful a few seconds ago.
            near_offset=float(np.interp(6.,candidate[:,0],candidate[:,1]))
            if abs(near_offset)>1.:
                if identifier==self.committed_id:self.committed_id=None
                continue
            supporters=[]
            for _,other_id,other in candidates:
                low=max(6.,candidate[0,0],other[0,0]);high=min(16.,candidate[-1,0],other[-1,0])
                if high-low<4.:continue
                z=np.linspace(low,high,8)
                difference=np.median(np.abs(np.interp(z,candidate[:,0],candidate[:,1])-np.interp(z,other[:,0],other[:,1])))
                if difference<.4 and all(abs(self.times.get(other_id,0)-self.times.get(prior,0))>20. for prior in supporters):
                    supporters.append(other_id)
            consensus=len(supporters)>=2 and abs(float(np.interp(6.,candidate[:,0],candidate[:,1])))<1.25
            committed=identifier==self.committed_id and now-self.committed_at<3.
            if not plan.inferred and plan.quality>=.75 and not (consensus or committed):continue
            if not self.validate(candidate,plan,scene,ego,enforce_lane=not (consensus or committed)):continue
            from .forza_lateral import lookahead_target
            from .forza_scene import traffic_limit,project,curve_speed
            _,_,steer=lookahead_target(candidate,ego['speed'])
            traffic,_=traffic_limit(candidate,scene.get('cars',[]),ego['speed'])
            road_limit=min(MAX_CRUISE_KMH,curve_speed(candidate,ego['speed'])[0])
            self.status=dict(action='successful_route',segment=identifier,segments=len(self.segments),
                             agreeing_traversals=len(supporters),consensus=consensus,committed=committed)
            self.last_path=candidate.copy()
            self.committed_id,self.committed_at=identifier,now
            result=replace(plan,path=candidate,steer=steer,reason='route_memory',inferred=True,
                limit=min(road_limit,traffic),road_limit=road_limit,
                image_path=tuple(project(float(x),float(z)) for z,x in candidate))
            if consensus or committed:
                from .forza_surface import constrain_lane_edges
                result=replace(result,lane_edges=tuple((float(z),float(x-1.75),float(x+1.75)) for z,x in candidate),
                               quality=max(.8,result.quality))
                observed=align_plan(result,ego,scene.get('ego') or ego)
                if observed is not None:
                    bounded=constrain_lane_edges(observed,scene.get('raw_road',scene['road']))
                    aligned=align_plan(bounded,scene.get('ego') or ego,ego)
                    if aligned is not None:
                        result=replace(aligned,verified_route=True)
            return result
        self.status=dict(action='no_valid_route',segments=len(self.segments))
        return plan
