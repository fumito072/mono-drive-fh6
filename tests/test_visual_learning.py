import unittest
import numpy as np
from mono_drive.visual_learning import visual_targets,visual_scores,accept_visual


class VisualLearningTests(unittest.TestCase):
    def test_heading_target_wraps_at_180_degrees(self):
        steer=np.zeros(30);yaw=np.full(30,179.);yaw[13:]=-179.
        y=visual_targets(steer,yaw,10)
        self.assertAlmostEqual(y[1],2/45,places=6)
    def test_dense_targets_refer_to_same_future_duration(self):
        steer=np.arange(40)/40.;yaw=np.arange(40)*2.
        y=visual_targets(steer,yaw,12,2)
        self.assertAlmostEqual(y[0],np.mean(steer[[14,16,18]]),places=6)
        self.assertAlmostEqual(y[1],12/45,places=6)
    def test_visual_gate_requires_both_turn_accuracy_and_finite_results(self):
        base=dict(samples=1000,steer_mae=.1,turn_mae=.2,heading_mae=.05,steer_bias=0.)
        self.assertTrue(accept_visual(dict(base,steer_mae=.09),base))
        self.assertFalse(accept_visual(dict(base,steer_mae=.09,turn_mae=.3),base))
        self.assertFalse(accept_visual(dict(base,steer_mae=float('nan')),base))
