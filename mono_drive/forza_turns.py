"""Road-arrow cues and a bounded turn state for sharp, visually supported exits."""
from dataclasses import dataclass,replace
import math
import cv2
import numpy as np


@dataclass
class ArrowCue:
    direction: int
    confidence: float
    distance_m: float


def arrow_templates():
    straight=np.zeros((100,100),np.uint8)
    cv2.fillPoly(straight,[np.array([(43,95),(57,95),(57,38),(80,38),(50,4),(20,38),(43,38)])],255)
    left=np.zeros_like(straight)
    cv2.fillPoly(left,[np.array([(46,95),(61,95),(61,22),(28,22),(28,5),(3,30),(28,55),(28,38),(46,38)])],255)
    return {-1:left,0:straight,1:cv2.flip(left,1)}


def match_arrow(mask):
    points=cv2.findNonZero(mask)
    if points is None:
        return None
    x,y,w,h=cv2.boundingRect(points)
    if min(w,h)<4:
        return None
    crop=mask[y:y+h,x:x+w]
    sample=cv2.resize(crop,(80,80),interpolation=cv2.INTER_NEAREST)>0
    scores=[]
    for direction,template in arrow_templates().items():
        p=cv2.findNonZero(template)
        tx,ty,tw,th=cv2.boundingRect(p)
        target=cv2.resize(template[ty:ty+th,tx:tx+tw],(80,80),interpolation=cv2.INTER_NEAREST)>0
        dice=2*np.count_nonzero(sample&target)/max(1,np.count_nonzero(sample)+np.count_nonzero(target))
        scores.append((float(dice),direction))
    scores.sort(reverse=True)
    if scores[0][0]<.62 or scores[0][0]-scores[1][0]<.08:
        return None
    return scores[0][1],scores[0][0]


def road_arrow(frame,road):
    from .forza_scene import project
    im=cv2.resize(frame,(960,540))
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    value=hsv[:,:,2].astype(float)
    white=((hsv[:,:,1]<90)&(value>90)&(value-cv2.blur(value,(25,13))>14)).astype(np.uint8)*255
    yy,xx=np.indices((240,120))
    z=18.-yy/16.
    x=(xx-60)/20.
    u,v=project(x,z)
    bev=cv2.remap(white,(u*959).astype(np.float32),(v*539).astype(np.float32),cv2.INTER_NEAREST)
    support=cv2.remap(road,(u*383).astype(np.float32),(v*215).astype(np.float32),cv2.INTER_NEAREST)
    bev[support<130]=0
    count,labels,stats,centres=cv2.connectedComponentsWithStats(bev)
    candidates=[]
    for label in range(1,count):
        left,top,w,h,area=stats[label]
        lateral=(centres[label,0]-60)/20
        if not (40<area<1100 and 7<w<65 and 15<h<100 and abs(lateral)<1.6):
            continue
        match=match_arrow((labels==label).astype(np.uint8)*255)
        if match:
            direction,quality=match
            candidates.append(ArrowCue(direction,quality,float(18.-centres[label,1]/16)))
    return min(candidates,key=lambda c:c.distance_m) if candidates else None


def road_end(road):
    from .forza_scene import road_value
    last=None
    for z in (5.,6.,8.,10.,12.,16.,20.,25.):
        clear=sum(road_value(road,x,z)>.55 for x in (-.5,0.,.5))>=2
        if clear:
            last=z
        elif last is not None:
            return last
    return None


def exit_visible(frame,road,direction):
    from .forza_scene import road_value,project
    from .forza_lane import road_patch
    supported=0
    for z in (5.,7.,9.):
        x=direction*(1.1+.10*z)
        u,v=project(x,z)
        supported += road_value(road,x,z)>.65 and road_patch(frame,u,v)
    return supported>=2


class TurnCoordinator:
    def __init__(self):
        self.started=None
        self.origin_yaw=0.
        self.direction=0
        self.cue=None
        self.cue_at=0.
        self.exit_candidate=None

    def update(self,frame,scene,ego,now,plan):
        from .forza_minimap import road_hint
        from .forza_scene import ScenePlan,project,visible_road_evidence,road_value,road_geometry_through_traffic
        road=visible_road_evidence(scene['road'],frame)
        road=road_geometry_through_traffic(road,scene.get('cars',[]))
        cue=road_arrow(frame,scene.get('raw_road',road))
        if cue is not None:
            self.cue,self.cue_at=cue,now
        intent=self.cue.direction if self.cue and now-self.cue_at<1.2 else None
        hint=road_hint(frame,intent=intent)
        end=road_end(road)
        direction=intent if intent in (-1,1) else (1 if hint and hint.turn_angle>.8 else -1 if hint and hint.turn_angle<-.8 else 0)
        # A minimap branch alone often resembles a driveway. With no arrow,
        # require a persistent map direction, visible exit, and a road end.
        supported=(intent in (-1,1) and self.cue.confidence>=.80 and
                   hint and direction*hint.turn_angle>.7 and exit_visible(frame,road,direction))
        visible_exit=(direction in (-1,1) and hint and getattr(hint,'confidence',0)>=.65 and
                      end is not None and end<=10. and exit_visible(frame,road,direction))
        if visible_exit:
            if self.exit_candidate is None or self.exit_candidate[0]!=direction:
                self.exit_candidate=(direction,now,1)
            else:self.exit_candidate=(direction,self.exit_candidate[1],self.exit_candidate[2]+1)
        else:self.exit_candidate=None
        if self.exit_candidate and now-self.exit_candidate[1]>=.8 and self.exit_candidate[2]>=3:
            supported=True
        if self.started is None:
            if not supported or end is None or end>16:
                return plan
            if end>8:
                return replace(plan,limit=min(plan.limit,20.),road_limit=min(plan.road_limit or 50.,20.)) if plan else None
            self.started,self.origin_yaw,self.direction=now,ego['yaw'],direction
        progress=self.direction*math.atan2(math.sin(ego['yaw']-self.origin_yaw),math.cos(ego['yaw']-self.origin_yaw))
        if progress>1.42 or (progress>1.2 and plan is not None and plan.quality>=.8):
            self.started=None
            return plan
        if now-self.started>12:
            self.started=None
            return None
        z=np.array([4.,5.,6.])
        radius=7.
        x=self.direction*(radius-np.sqrt(radius*radius-z*z))
        path=np.column_stack((z,x))
        support=[road_value(road,float(px+side),float(pz)) for pz,px in path for side in (-.6,0.,.6)]
        if np.mean(np.array(support)>.55)<.55:
            return None
        steer=self.direction*(.62 if progress<1.05 else .38)
        return ScenePlan(path,steer,14.,.75,1,len(scene['cars']),'intersection_turn',float(np.mean(support)),
                         image_path=tuple(project(float(px),float(pz)) for pz,px in path),
                         road_limit=14.)
