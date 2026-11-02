import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from mono_drive.forza_continuation import LaneContinuation,extend_path,continue_unmarked
from mono_drive.forza_scene import ScenePlan,traffic_limit
from mono_drive.forza import SpeedController


def lane():
    z=np.array([4.,6.,10.,16.])
    return ScenePlan(np.column_stack((z,np.zeros_like(z))),0.,50.,.9,1,0,'paint_pair',.9)


class ContinuationTests(unittest.TestCase):
    def test_short_uncertainty_cap_coasts_before_slowing_but_hazards_do_not_wait(self):
        d=SimpleNamespace(confidence=.6,steer=0.,target_kmh=25.,mode='lane_inferred',
                          reverse=False,uncertainty_limited=True)
        c=SpeedController();c.target=60.
        for _ in range(5):
            command=c.step(d,55.,60.,.5,.1)
            self.assertEqual(command[2],0.)
            self.assertLessEqual(c.desired_acceleration,.01)
        for _ in range(15):command=c.step(d,55.,60.,.5,.1)
        self.assertGreater(command[2],0.)
        for mode,limited,target in [('lane_inferred',False,18.),('traffic_stop',True,0.)]:
            c=SpeedController();c.target=60.
            d.mode,d.uncertainty_limited,d.target_kmh=mode,limited,target
            command=c.step(d,55.,60.,.5,.1)
            self.assertLess(c.target,60.)
            if mode=='traffic_stop':self.assertGreater(command[2],0.)

    def test_only_missing_evidence_not_a_geometric_curve_gets_uncertainty_tag(self):
        scene=dict(road=np.zeros((216,384),np.uint8),cars=[])
        ego=dict(position=[0,0,0],yaw=0.,speed=20.,steer=0.)
        self.assertTrue(LaneContinuation().predict(scene,ego,1.).uncertainty_limited)
        predictor=LaneContinuation()
        predictor.predict(scene,dict(ego,steer=38.),1.)
        curved=predictor.predict(scene,dict(ego,steer=38.),1.3)
        self.assertLess(curved.limit,25.)
        self.assertFalse(curved.uncertainty_limited)

    def test_prediction_continues_after_old_path_has_passed_under_car(self):
        predictor=LaneContinuation()
        ego=dict(position=[0,0,0],yaw=0.,speed=40.,steer=0.)
        predictor.observe(lane(),ego,1.)
        advanced=dict(ego,position=[0,0,22])
        result=predictor.predict(dict(road=np.full((216,384),255,np.uint8),cars=[]),advanced,3.)
        self.assertTrue(result.inferred)
        self.assertGreater(result.limit,30.)
        self.assertAlmostEqual(result.steer,0.)
        self.assertGreater(extend_path(lane().path)[-1,0],80)

    def test_map_and_visible_road_support_normal_cruise_without_paint(self):
        result=LaneContinuation().predict(dict(road=np.full((216,384),255,np.uint8),cars=[]),
            dict(position=[0,0,0],yaw=0.,speed=30.,steer=0.),1.,SimpleNamespace(angle=.05,confidence=.7))
        self.assertGreater(result.steer,0.)
        self.assertEqual(result.limit,60.)
        self.assertEqual(result.lane_edges,())

    def test_no_lane_or_map_uses_heading_prior_instead_of_stop(self):
        result=LaneContinuation().predict(dict(road=np.zeros((216,384),np.uint8),cars=[]),
            dict(position=[0,0,0],yaw=0.,speed=20.,steer=0.),1.)
        self.assertEqual(result.limit,25.)
        self.assertEqual(result.reason,'lane_inferred')
        d=SimpleNamespace(confidence=result.quality,steer=result.steer,target_kmh=result.limit,mode=result.reason,reverse=False)
        controller=SpeedController()
        outputs=[controller.step(d,10.,50.,.5,.1) for _ in range(20)]
        self.assertGreater(outputs[-1][1],0.)
        self.assertEqual(outputs[-1][2],0.)

    def test_inferred_frames_do_not_overwrite_the_measured_anchor(self):
        predictor=LaneContinuation()
        ego=dict(position=[0,0,0],yaw=0.,speed=30.,steer=0.)
        predictor.observe(lane(),ego,1.)
        predictor.observe(replace(lane(),inferred=True),ego,9.)
        self.assertEqual(predictor.observed_at,1.)

    def test_missing_boundary_does_not_cancel_a_real_obstacle_stop(self):
        plan=replace(lane(),guard_state='boundary_unknown',reason='static_stop',limit=0.)
        result=continue_unmarked(plan,40.)
        self.assertEqual(result.limit,0.)
        self.assertEqual(result.reason,'static_stop')
        predicted=LaneContinuation().predict(dict(road=np.full((216,384),255,np.uint8),cars=[]),
            dict(position=[0,0,0],yaw=0.,speed=30.,steer=0.),1.)
        self.assertEqual(traffic_limit(predicted.path,[dict(x=0.,z=3.,vx=0.,vz=0.)],30.)[0],0.)


if __name__=='__main__':
    unittest.main()
