import unittest
from types import SimpleNamespace
import numpy as np
from mono_drive.forza_training_filter import segment_geometry,training_row_valid
from mono_drive.teacher import Drive


class TrainingSegmentsTests(unittest.TestCase):
    def test_validated_route_can_supply_training_but_blind_and_sidewalk_motion_cannot(self):
        row=dict(armed=True,forward_kmh=30.,actual_brake=0,race_position=0,
                 candidate_path=[[5.,.1],[10.,.2]],mode='lane_guard',inferred_lane=True,
                 lane_clearance_m=.5,static_clearance_m=50.,
                 route_memory=dict(retained=True,consensus=True,agreeing_traversals=4))
        self.assertTrue(training_row_valid(row))
        row['route_memory']['consensus']=False
        self.assertFalse(training_row_valid(row))
        row['route_memory']['consensus']=True
        row['surface']=dict(path_enters_sidewalk=True)
        self.assertFalse(training_row_valid(row))

    def test_short_clean_segment_has_actual_future_trajectory_labels(self):
        positions=np.column_stack((np.arange(10)*5.,np.zeros((10,2))))
        group=[dict(position=p.tolist()) for p in positions]
        self.assertEqual(segment_geometry(group)['distance_m'],45.)
        drive=Drive.__new__(Drive)
        drive.files=list(range(10))
        drive.pos=positions
        drive.odo=positions[:,0]
        drive.jump=np.zeros(10,dtype=bool)
        drive.fwd=np.tile([1.,0.,0.],(10,1))
        drive.right=np.tile([0.,-1.,0.],(10,1))
        drive.up=np.tile([0.,0.,1.],(10,1))
        view=SimpleNamespace(k=1.,user_to_ref=lambda *point:point)
        target=drive.target(0,view)
        self.assertIsNotNone(target)
        self.assertGreaterEqual(target[1].sum(),3)

    def test_short_distance_teleport_and_steep_grade_remain_rejected(self):
        for positions in (
            [[i*2.,0.,0.] for i in range(10)],
            [[i*10.,0.,0.] for i in range(10)],
            [[i*5.,i,0.] for i in range(10)],
            [[i*5.,float('nan'),0.] for i in range(10)]):
            with self.subTest(positions=positions):
                self.assertIsNone(segment_geometry([dict(position=p) for p in positions]))

    def test_previous_long_slow_segments_are_retained(self):
        group=[dict(position=[i*.8,0.,0.]) for i in range(30)]
        self.assertAlmostEqual(segment_geometry(group)['distance_m'],23.2)


if __name__=='__main__':
    unittest.main()
