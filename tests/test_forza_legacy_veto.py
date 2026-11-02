import unittest
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from mono_drive.forza_scene import ScenePlan
from mono_drive.forza_legacy_veto import obstacle_veto


class LegacyVetoTests(unittest.TestCase):
    def setUp(self):
        z=np.array([4.,8.,12.,18.,24.])
        self.plan=ScenePlan(np.column_stack((z,np.zeros_like(z))),0.,60.,.9,1,0,'paint_pair',.9,
                            lane_edges=tuple((d,-1.75,1.75) for d in z))
        self.model=SimpleNamespace(mode='blocked',confidence=1.,blocked_m=12.)
        self.ego=dict(position=[0,0,0],yaw=0.,speed=40.,steer=0.)
        self.scene=dict(road=np.full((216,384),255,np.uint8),cars=[])

    def test_clear_measured_lane_can_reject_uncorroborated_old_grid_alarm(self):
        self.assertEqual(obstacle_veto(self.model,self.plan,self.scene,self.ego),
                         (False,'independent_corridor_clear'))

    def test_close_unknown_inferred_and_short_corridors_keep_veto(self):
        for plan,distance in [(self.plan,2.),(self.plan,30.),(replace(self.plan,inferred=True),12.)]:
            self.model.blocked_m=distance
            self.assertTrue(obstacle_veto(self.model,plan,self.scene,self.ego)[0])
        self.model.blocked_m=12.
        self.scene['road'][:]=0
        self.assertTrue(obstacle_veto(self.model,self.plan,self.scene,self.ego)[0])

    def test_independent_traffic_or_static_hazard_preserves_stop(self):
        with patch('mono_drive.forza_static.static_risk',return_value=SimpleNamespace(brake=.1,emergency=False,clearance_m=8.)):
            self.assertEqual(obstacle_veto(self.model,self.plan,self.scene,self.ego),
                             (True,'corroborated_obstacle'))
        self.scene['cars']=[dict(id=1,x=0.,z=8.,vx=0.,vz=0.,confidence=.9,kind='car')]
        self.assertTrue(obstacle_veto(self.model,self.plan,self.scene,self.ego)[0])


if __name__=='__main__':unittest.main()
