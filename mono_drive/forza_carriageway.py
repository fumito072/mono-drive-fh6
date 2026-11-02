"""Raised-median boundaries and corroborated wrong-way detection.

Separate paved regions do not imply the same direction. In left-hand traffic
the median belongs on the right. A left median plus reliable approaching
traffic on our side is evidence to stop and use the verified road recovery.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .forza_scene import project


@dataclass
class Median:
    edges: np.ndarray
    side: str
    confidence: float


def detect_median(road, obstacles):
    if obstacles is None or road.shape != obstacles.shape:
        return None
    height, width = road.shape
    groups=[]
    for distance in (6.,8.,10.,12.,16.,20.,25.):
        y=int(round(project(0,distance)[1]*(height-1)))
        if not 1<=y<height-1:
            continue
        strip=np.mean(road[y-1:y+2],axis=0)>140
        d=np.diff(np.r_[False,strip,False].astype(int))
        spans=[(a,b) for a,b in zip(np.where(d==1)[0],np.where(d==-1)[0])
               if (b-a)/(width-1)*2*distance>2.0]
        for first,second in zip(spans,spans[1:]):
            a,b=first[1],second[0]
            left=(a/(width-1)-.5)*2*distance-.35
            right=(b/(width-1)-.5)*2*distance-.35
            if not .3<=right-left<=4. or min(abs(left),abs(right))>9.:
                continue
            # A paint line, shadow or car-sized occlusion is not a median.
            physical=float(np.mean(obstacles[y-1:y+2,a:b]>110)) if b>a else 0.
            if physical<.55:
                continue
            point=(distance,left,right)
            candidates=[group for group in groups if group[-1][0]<distance and
                        abs(np.mean(group[-1][1:])-np.mean(point[1:]))<.5*(distance-group[-1][0])+1.]
            if candidates:
                min(candidates,key=lambda group:abs(np.mean(group[-1][1:])-np.mean(point[1:]))).append(point)
            else:
                groups.append([point])
    groups=[group for group in groups if len(group)>=3 and group[0][0]<=10. and group[-1][0]-group[0][0]>=4.]
    if not groups:
        return None
    points=np.asarray(max(groups,key=lambda group:(len(group),-abs(np.mean(group[0][1:])))))
    side='left' if points[0,2]<-.6 else 'right' if points[0,1]>.6 else 'ahead'
    return Median(points,side,min(.95,.6+.05*len(points)))


def path_crosses_median(path, median, half_width=.925):
    if median is None:
        return False
    path=np.asarray(path)
    points=path[(path[:,0]>=median.edges[0,0]) & (path[:,0]<=median.edges[-1,0])]
    if not len(points):
        return False
    left=np.interp(points[:,0],median.edges[:,0],median.edges[:,1])
    right=np.interp(points[:,0],median.edges[:,0],median.edges[:,2])
    if median.side=='left':
        return bool(np.any(points[:,1]-half_width<right+.15))
    if median.side=='right':
        return bool(np.any(points[:,1]+half_width>left-.15))
    return bool(np.any((points[:,1]+half_width>left) & (points[:,1]-half_width<right)))


class CarriagewayGuard:
    def __init__(self):
        self.started=None
        self.last_seen=0.
        self.frames=0
        self.median=None

    def update(self,road,obstacles,cars,now):
        median=detect_median(road,obstacles)
        self.median=median
        oncoming=[]
        if median is not None and median.side=='left':
            for car in cars:
                if car.get('view','front')!='front' or not car.get('motion_reliable') or car.get('confidence',0)<.6:
                    continue
                x,z=car.get('x',100.),car.get('z',100.)
                boundary=np.interp(z,median.edges[:,0],median.edges[:,2])
                if 2.<z<45. and boundary+.8<x<4.5 and car.get('vz',0)<-2.:
                    oncoming.append(car.get('id'))
        evidence=median is not None and median.confidence>=.75 and bool(oncoming)
        if evidence:
            if self.started is None or now-self.last_seen>.6:
                self.started=now
                self.frames=0
            self.frames+=1
            self.last_seen=now
        elif now-self.last_seen>.6:
            self.started=None
            self.frames=0
        confirmed=bool(evidence and self.frames>=3 and now-self.started>=1.2)
        return dict(median_side=median.side if median else 'unknown',
                    median_confidence=median.confidence if median else 0.,
                    median_edges=median.edges.tolist() if median else [],
                    oncoming_same_carriageway=oncoming,wrongway_confirmed=confirmed,
                    reason='left_median_and_opposing_traffic' if confirmed else 'observing')
