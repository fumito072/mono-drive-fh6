import unittest
import cv2
import numpy as np
from mono_drive.video_motion_learning import motion_patch,image_motion


class VideoMotionTests(unittest.TestCase):
    def test_translation_is_weak_image_motion_not_world_position(self):
        rng=np.random.default_rng(17)
        a=cv2.GaussianBlur(rng.integers(0,256,(128,320,3),dtype=np.uint8),(3,3),0)
        b=cv2.warpAffine(a,np.array([[1.,0.,3.],[0.,1.,-2.]],np.float32),(320,128))
        result=image_motion(a.transpose(2,0,1),b.transpose(2,0,1),.5)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result['target'][:2],[3/320/.5,-2/128/.5],atol=.004)
        self.assertFalse(result['world_position_available']);self.assertFalse(result['metric_speed_available'])
    def test_static_and_long_gaps_do_not_create_fake_movement(self):
        a=np.random.default_rng(9).integers(0,256,(3,128,320),dtype=np.uint8)
        self.assertIsNone(image_motion(a,a,.5))
        self.assertIsNone(image_motion(a,a,2.))
    def test_patch_removes_bottom_hud(self):
        image=np.zeros((360,640,3),np.uint8);image[300:]=255
        self.assertEqual(motion_patch(image).shape,(3,128,320))
        self.assertEqual(int(motion_patch(image).max()),0)
