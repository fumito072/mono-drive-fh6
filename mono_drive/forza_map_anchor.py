"""Locate one saved paved return point by map landmarks, never the player."""
from pathlib import Path
import json
import cv2
import numpy as np


class FixedMapAnchor:
    def __init__(self, configuration):
        self.config = dict(configuration) if isinstance(configuration,dict) else json.loads(Path(configuration).read_text(encoding='utf-8'))
        self.image = cv2.imread(self.config['map_image'])
        if self.image is None:
            raise ValueError('Saved return-point map is missing')
        self.sift = cv2.SIFT_create(nfeatures=4000)
        self.keys, self.descriptors = self.features(self.image)
        self.references=[(self.keys,self.descriptors,self.config['map_point'])]
        self.last_matrix = None
        for reference in self.config.get('atlas',[]):
            image=cv2.imread(reference['image'])
            if image is not None:
                keys,descriptors=self.features(image)
                self.references.append((keys,descriptors,reference['point']))

    def features(self, image):
        gray = cv2.cvtColor(cv2.resize(image,(1920,1080)),cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        mask[50:975,40:1880] = 255
        # Exclude fixed menus, which must not count as geographic matches.
        mask[0:280,1380:] = 0
        return self.sift.detectAndCompute(gray,mask)

    def locate(self, image, offset=(0.,0.)):
        keys, descriptors = self.features(image)
        if descriptors is None or self.descriptors is None:
            return None
        for reference_keys,reference_descriptors,point in self.references:
            location=self.match(reference_keys,reference_descriptors,np.asarray(point)+offset,keys,descriptors)
            if location is not None:
                return location
        return None

    def match(self,reference_keys,reference_descriptors,point,keys,descriptors):
        pairs = cv2.BFMatcher().knnMatch(reference_descriptors,descriptors,k=2)
        good = [m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance < .67*n.distance]
        if len(good)<18:
            return None
        source = np.float32([reference_keys[m.queryIdx].pt for m in good])
        target = np.float32([keys[m.trainIdx].pt for m in good])
        matrix,inliers = cv2.estimateAffinePartial2D(source,target,method=cv2.RANSAC,ransacReprojThreshold=2.5)
        if matrix is None or inliers is None or inliers.sum()<16:
            return None
        supported = source[inliers.ravel().astype(bool)]
        if np.ptp(supported[:,0])<140 or np.ptp(supported[:,1])<100:
            return None
        scale = float(np.linalg.norm(matrix[:,0]))
        angle = abs(float(np.arctan2(matrix[1,0],matrix[0,0])))
        if not .25<scale<4 or angle>.06:
            return None
        self.last_matrix = matrix
        return matrix@np.array([*point,1.])

    def reference_point(self, cursor):
        """Save the actual selected map point, independent of later pan/zoom."""
        if self.last_matrix is None:
            return None
        return np.linalg.solve(self.last_matrix[:,:2],np.asarray(cursor)-self.last_matrix[:,2]).tolist()

    def arrived(self, position):
        goal = np.asarray(self.config['position'])
        return np.linalg.norm((np.asarray(position)-goal)[[0,2]]) < self.config.get('arrival_radius_m',100.)

    def search_direction(self, origin):
        """North-up map direction when a distant saved region is off-screen."""
        try:
            origin=np.asarray(origin,dtype=float)
            goal=np.asarray(self.config['position'],dtype=float)
            if origin.shape!=(3,) or goal.shape!=(3,) or not np.isfinite([origin,goal]).all():return None
            delta=(goal-origin)[[0,2]]
            if np.linalg.norm(delta)<800.:return None
            delta[1]*=-1.
            return delta/np.max(np.abs(delta))*.8
        except (KeyError,TypeError,ValueError):
            return None
