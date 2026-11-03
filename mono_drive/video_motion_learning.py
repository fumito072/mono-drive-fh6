"""Weak image-motion targets for footage without recorded game controls."""
import math
import cv2
import numpy as np


def motion_patch(image):
    # Remove bottom HUD/instruments. These are image coordinates, not a world pose.
    h,w=image.shape[:2]
    crop=image[:round(h*.72),round(w*.08):round(w*.92)]
    patch=cv2.resize(crop,(320,128),interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(patch,cv2.COLOR_BGR2RGB).transpose(2,0,1))


def image_motion(previous,current,seconds):
    if not .15<=seconds<=1.5:return None
    a=cv2.cvtColor(previous.transpose(1,2,0),cv2.COLOR_RGB2GRAY)
    b=cv2.cvtColor(current.transpose(1,2,0),cv2.COLOR_RGB2GRAY)
    mask=np.zeros_like(a);mask[round(a.shape[0]*.30):round(a.shape[0]*.94),30:-30]=255
    points=cv2.goodFeaturesToTrack(a,180,.015,5,mask=mask)
    if points is None or len(points)<24:return None
    forward,status,_=cv2.calcOpticalFlowPyrLK(a,b,points,None,winSize=(21,21),maxLevel=3)
    if forward is None:return None
    backward,valid,_=cv2.calcOpticalFlowPyrLK(b,a,forward,None,winSize=(21,21),maxLevel=3)
    if backward is None:return None
    keep=(status.ravel()==1)&(valid.ravel()==1)&(np.linalg.norm(backward-points,axis=2).ravel()<1.5)
    p,q=points[keep].reshape(-1,2),forward[keep].reshape(-1,2)
    if len(p)<24:return None
    matrix,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=1000)
    if matrix is None or inliers is None:return None
    ratio=float(inliers.mean());inlier_points=p[inliers.ravel().astype(bool)]
    if ratio<.65 or len(inlier_points)<20 or np.ptp(inlier_points[:,0])<100 or np.ptp(inlier_points[:,1])<30:return None
    scale=math.hypot(matrix[0,0],matrix[1,0])
    if not .75<scale<1.33:return None
    value=np.asarray([matrix[0,2]/320/seconds,matrix[1,2]/128/seconds,math.log(scale)/seconds],np.float32)
    if not np.isfinite(value).all() or np.max(np.abs(value))>.5 or np.linalg.norm(value)<.0005:return None
    return {'target':value.tolist(),'confidence':ratio,'points':len(inlier_points),
            'target_kind':'estimated_image_motion','metric_speed_available':False,'world_position_available':False}
