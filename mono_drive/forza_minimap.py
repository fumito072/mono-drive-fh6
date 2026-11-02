"""Fixed-HUD minimap road direction; no game camera or menu changes."""
from dataclasses import dataclass
import heapq
import math
import cv2
import numpy as np
from functools import lru_cache


@dataclass
class MapHint:
    angle: float
    confidence: float
    path: tuple
    turn_angle: float = 0.


@lru_cache(maxsize=4)
def road_distances(pixels, shape, start):
    """Reuse an identical HUD graph when several planners inspect one frame.

    Key by immutable pixel content, so changed pixels on a reused capture
    buffer cannot accidentally reuse an old road. Cached results are immutable.
    """
    mask=np.frombuffer(pixels,dtype=np.uint8).reshape(shape)
    cost={start:0.}
    queue=[(0.,start)]
    while queue:
        d,p=heapq.heappop(queue)
        if d!=cost[p] or d>85:
            continue
        for dy,dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
            q=(p[0]+dy,p[1]+dx)
            if not (0<=q[0]<mask.shape[0] and 0<=q[1]<mask.shape[1] and mask[q]):
                continue
            nd=d+math.hypot(dx,dy)
            if nd<cost.get(q,float('inf')):
                cost[q]=nd
                heapq.heappush(queue,(nd,q))
    return tuple(cost.items())


def road_hint(frame,intent=None):
    im = cv2.resize(frame,(1920,1080),interpolation=cv2.INTER_AREA)[750:1005,50:340]
    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    y,x = np.indices(im.shape[:2])
    centre = np.array([146.,186.])
    mask = ((hsv[:,:,1] < 110) & (hsv[:,:,2] > 75) & (hsv[:,:,2] < 235) &
            ((x-146)**2+(y-138)**2 < 129**2) & (y < 192)).astype(np.uint8)
    mask[(x-centre[0])**2+(y-centre[1])**2 < 13**2] = 0
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    ys,xs = np.nonzero(mask)
    if not len(xs):
        return None
    distances = (xs-centre[0])**2+(ys-centre[1])**2
    distances = np.where(ys < 181,distances,1e6)
    index = int(np.argmin(distances))
    if distances[index] > 30**2:
        return None
    start = (int(ys[index]),int(xs[index]))
    cost = road_distances(mask.tobytes(),mask.shape,start)
    points = []
    for distance in (15,30,45,60):
        band = [(p[1],p[0]) for p,d in cost if distance-3 <= d <= distance+3]
        if len(band) < 3:
            break
        band=np.asarray(band)
        order=np.argsort(band[:,0])
        groups=np.split(band[order],np.flatnonzero(np.diff(band[order,0])>10)+1)
        if len(groups)>1:
            means=[np.mean(group,axis=0) for group in groups if len(group)>=3]
            if not means:
                break
            if intent in (-1,1):
                point=min(means,key=lambda p:intent*-p[0])
            else:
                reference=points[-1][0]-50 if points else 146.
                point=min(means,key=lambda p:abs(p[0]-reference))
        else:
            point = np.mean(band,axis=0)
        points.append((float(point[0]+50),float(point[1]+750)))
    if len(points) < 3:
        return None
    end = np.array(points[min(2,len(points)-1)])-np.array([196.,936.])
    angle = math.atan2(float(end[0]),float(-end[1]))
    last=np.array(points[-1])-np.array(points[-2])
    turn_angle=math.atan2(float(last[0]),float(-last[1]))
    return MapHint(angle,.7 if len(points)>=4 else .55,tuple(points),turn_angle)
