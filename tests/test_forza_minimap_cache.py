import unittest
import numpy as np
from mono_drive.forza_minimap import road_distances


class MinimapCacheTests(unittest.TestCase):
    def test_reused_capture_buffer_with_changed_road_is_not_stale(self):
        pixels=np.ones((12,12),np.uint8)
        before=dict(road_distances(pixels.tobytes(),pixels.shape,(10,6)))
        self.assertIn((0,6),before)
        pixels[5,:]=0
        after=dict(road_distances(pixels.tobytes(),pixels.shape,(10,6)))
        self.assertNotIn((0,6),after)
        self.assertEqual(after,dict(road_distances(pixels.copy().tobytes(),pixels.shape,(10,6))))
