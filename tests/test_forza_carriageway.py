import unittest
import numpy as np
from mono_drive.forza_carriageway import detect_median, path_crosses_median, CarriagewayGuard
from mono_drive.forza_scene import project
from mono_drive.forza_vlm import parse_assessment


def divided_road(median_left=-4.,median_right=-2.):
    road=np.zeros((216,384),np.uint8)
    obstacles=np.zeros_like(road)
    for y in range(95,132):
        distance=.575/(y/215.-.435)
        a=max(0,int(project(median_left-7.,distance)[0]*383))
        b=max(0,int(project(median_left,distance)[0]*383))
        c=min(384,int(project(median_right,distance)[0]*383))
        d=min(384,int(project(median_right+7.,distance)[0]*383))
        road[y,a:b]=250
        road[y,c:d]=250
        obstacles[y,b:c]=230
    return road,obstacles


class CarriagewayTests(unittest.TestCase):
    def test_median_separates_carriageways_and_paths_cannot_cross(self):
        road,obstacles=divided_road()
        median=detect_median(road,obstacles)
        self.assertIsNotNone(median)
        self.assertEqual(median.side,'left')
        self.assertFalse(path_crosses_median(np.array([[6,0],[10,0],[16,0]]),median))
        self.assertTrue(path_crosses_median(np.array([[6,0],[10,-2],[16,-5]]),median))

    def test_paint_or_shadow_is_not_a_raised_median(self):
        road,obstacles=divided_road()
        self.assertIsNone(detect_median(road,np.zeros_like(obstacles)))

    def test_opposing_traffic_and_persistent_left_median_confirm_wrong_way(self):
        road,obstacles=divided_road()
        guard=CarriagewayGuard()
        car=dict(id=4,x=0.,z=12.,vz=-9.,confidence=.9,motion_reliable=True)
        self.assertFalse(guard.update(road,obstacles,[car],0.)['wrongway_confirmed'])
        self.assertFalse(guard.update(road,obstacles,[car],.5)['wrongway_confirmed'])
        guard.update(road,obstacles,[car],1.)
        self.assertTrue(guard.update(road,obstacles,[car],1.3)['wrongway_confirmed'])

    def test_traffic_across_median_does_not_confirm_wrong_way(self):
        road,obstacles=divided_road(2.,4.)
        guard=CarriagewayGuard()
        car=dict(id=4,x=6.,z=12.,vz=-9.,confidence=.9,motion_reliable=True)
        for t in np.arange(0,3,.1):
            status=guard.update(road,obstacles,[car],float(t))
            self.assertFalse(status['wrongway_confirmed'])
        self.assertEqual(status['median_side'],'right')

    def test_video_model_reports_median_and_flow_as_data(self):
        value=parse_assessment('[0,3,0,0,1,9,1,1]')
        self.assertEqual(value['median_side'],'left')
        self.assertEqual(value['traffic_flow'],'opposing')


if __name__=='__main__':
    unittest.main()
