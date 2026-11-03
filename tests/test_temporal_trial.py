import unittest
from mono_drive.temporal_trial import DrivingMetrics,trial_passed,target_achieved


class TrialTests(unittest.TestCase):
    def test_unobserved_lanes_never_count_as_centered(self):
        m=DrivingMetrics();base={'armed':True,'state':'driving','car_ordinal':3819,'gear':1,'speed_kmh':60.,'sent_command':[0,.2,0]}
        m.consume([dict(base,t=1),dict(base,t=1.2,inferred_lane=True)],'a')
        self.assertIsNone(m.summary()['estimated_centered_fraction'])
        m.consume([dict(base,t=1.4,confidence=.9,lane_edges=[(6,-1.75,1.75)])],'a')
        self.assertEqual(m.summary()['estimated_centered_fraction'],1.)
    def test_success_requires_measured_speed_and_lane_coverage(self):
        good=dict(driving_seconds=150,observed_lane_seconds=120,mean_kmh=59,fraction_55_to_65=.8,
                  estimated_centered_fraction=.95,offroad_seconds=0,wrongway_seconds=0,steer_rate_p95=.2,pedal_rate_p95=.1)
        self.assertTrue(trial_passed(good,good,100,0));self.assertTrue(target_achieved(good,0))
        self.assertFalse(trial_passed(good,good,100,1))
        self.assertFalse(target_achieved(dict(good,mean_kmh=25),0))
        self.assertFalse(trial_passed(good,dict(good,observed_lane_seconds=0),100,0))
        self.assertFalse(trial_passed(good,dict(good,steer_rate_p95=float('nan')),100,0))
        self.assertFalse(target_achieved(dict(good,observed_lane_seconds=30),0))
