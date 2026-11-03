"""Rotate between evidence-backed free-roam road destinations."""
import math
from pathlib import Path
import random

from .forza_local_learning import read_json,write_json


class RouteRotation:
    def __init__(self,root,rng=None):
        self.root=Path(root)
        self.file=self.root/'out/route-rotation-state.json'
        self.state=read_json(self.file,{})
        self.rng=rng or random.Random()

    def choose(self,now,recoveries,force=False):
        evaluation=read_json(self.root/'out/temporal-live-evaluation.json',{})
        if (not force and evaluation.get('hold_route') is True and
                now-evaluation.get('time',0)<10. and evaluation.get('phase') in ('shadow','trial')):
            return None
        config=read_json(self.root/'configs/free_roam_routes.json',{})
        if config.get('enabled') is not True:return None
        if not force and now<self.state.get('retry_at',0.):return None
        elapsed=now-self.state.get('changed_at',0.)
        repeats=recoveries-self.state.get('recoveries_at_change',recoveries)
        interval=max(300.,float(config.get('interval_seconds',900.)))
        if not force and self.state.get('current_id') and elapsed<interval and repeats<2:return None
        available=[]
        for route in config.get('destinations',[]):
            if route.get('id')==self.state.get('current_id'):continue
            if self.state.get('failed_until',{}).get(route.get('id'),0)>now:continue
            anchor=route.get('anchor',{})
            try:
                position=anchor['position'];point=anchor['map_point']
                if len(position)!=3 or len(point)!=2 or not all(math.isfinite(float(v)) for v in position+point):continue
                if not route.get('source_frame'):continue
                if not Path(route['source_frame']).is_file() or not Path(anchor['map_image']).is_file():continue
            except (KeyError,TypeError,ValueError):continue
            verified=self.state.get('verified_positions',{}).get(route['id'])
            if verified:
                anchor=dict(anchor,position=verified,arrival_radius_m=120.)
                point=self.state.get('verified_map_points',{}).get(route['id'])
                if point:anchor['map_point']=point
                route=dict(route,anchor=anchor)
            available.append(route)
        if not available:return None
        recent=self.state.get('recent',[])[-2:]
        fresh=[route for route in available if route['id'] not in recent]
        pool=fresh or available
        visits=self.state.get('visits',{})
        choice=self.rng.choices(pool,weights=[1./(1+visits.get(r['id'],0)) for r in pool],k=1)[0]
        return choice

    def started(self,route,now):
        self.state.update(phase='selecting',requested_id=route['id'],requested_name=route['name'],requested_at=now)
        write_json(self.file,self.state)

    def succeeded(self,route,position,now,recoveries,map_point=None):
        identifier=route['id']
        self.state.update(current_id=identifier,name=route['name'],changed_at=now,
                          recoveries_at_change=recoveries,phase='driving',position=list(position),retry_at=0.)
        self.state['recent']=(self.state.get('recent',[])+[identifier])[-4:]
        visits=self.state.setdefault('visits',{});visits[identifier]=visits.get(identifier,0)+1
        self.state.setdefault('verified_positions',{})[identifier]=list(position)
        if map_point is not None:
            self.state.setdefault('verified_map_points',{})[identifier]=list(map_point)
        write_json(self.file,self.state)

    def failed(self,route,now,error):
        self.state.setdefault('failed_until',{})[route['id']]=now+3600.
        self.state.update(phase='fallback',last_error=str(error),failed_route=route['id'],retry_at=now+300.)
        write_json(self.file,self.state)


def paved_arrival(scene,now):
    if not scene or not 0<=now-scene.get('captured',0)<.5:return False
    from .forza_scene import footprint_quality
    from .forza_projection import set_projection
    import numpy as np
    set_projection(scene.get('projection'))
    floor,mean=footprint_quality(scene.get('raw_road',scene['road']),np.array([[6.,0.],[8.,0.],[10.,0.],[12.,0.]]))
    return floor>=.25 and mean>=.65


def road_probe_offsets():
    """Bounded local map search; every point still needs the fast-travel dialog."""
    return [(0.,0.)]+[(x,y) for r in (12.,24.,36.) for x,y in ((0.,r),(0.,-r),(r,0.),(-r,0.))]
