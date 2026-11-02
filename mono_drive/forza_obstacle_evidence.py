"""Ground-contact and temporal evidence for fixed obstacle steering.

Roadside silhouettes above their ground contact are not occupied road cells.
Raw detections remain available for immediate braking; lateral avoidance uses
repeated, ground-supported evidence.
"""
from collections import deque
import cv2
import numpy as np


def ground_contacts(obstacles, ground, roi):
    occupied=(obstacles>140)&roi
    below=np.zeros_like(occupied)
    for shift in (1,2,3,4):
        below[:-shift] |= ground[shift:]>110
    contacts=occupied&below
    # A wall clipped by the lowest visible windshield row is a near obstacle,
    # not free space just because its hidden foot cannot be seen.
    bottom=np.zeros_like(roi)
    for x in range(roi.shape[1]):
        rows=np.flatnonzero(roi[:,x])
        if len(rows):
            bottom[rows[-1],x]=True
    contacts |= occupied&bottom
    support=cv2.dilate(contacts.astype(np.uint8),np.ones((5,3),np.uint8))>0
    return np.where(support&roi,obstacles,0).astype(np.uint8)


class StaticEvidence:
    def __init__(self):
        self.frames=deque(maxlen=5)

    def update(self,mask,now):
        if self.frames and now-self.frames[-1][0]>.6:
            self.frames.clear()
        self.frames.append((now,mask.copy()))
        recent=[value for stamp,value in self.frames if now-stamp<=.5]
        if len(recent)<2 or now-self.frames[0][0]<.12:
            return np.zeros_like(mask)
        votes=sum(cv2.dilate((value>140).astype(np.uint8),np.ones((3,3),np.uint8)) for value in recent)
        return np.where(votes>=2,mask,0).astype(np.uint8)


def remove_dynamic_footprints(obstacles,cars):
    """A tracked moving vehicle is checked by the relative-motion guard once."""
    result=obstacles.copy()
    h,w=result.shape
    for car in cars:
        if not car.get('motion_reliable') or not car.get('moving') or car.get('confidence',0)<.5:
            continue
        x1,y1,x2,y2=car['box']
        a,b=max(0,int(x1*w)),min(w,int(np.ceil(x2*w)))
        top,bottom=max(0,int(y1*h)),min(h,int(np.ceil(y2*h)))
        result[top:bottom,a:b]=0
    return result
