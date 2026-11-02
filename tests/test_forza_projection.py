import unittest
import numpy as np
from mono_drive.forza_projection import project,ground_point,set_projection,fit_corridor,GroundCalibration


class ProjectionTests(unittest.TestCase):
    def tearDown(self):set_projection()

    def test_projection_and_inverse_share_one_calibration(self):
        set_projection(dict(horizon=.428,scale=.746))
        for x,z in ((0.,5.),(-1.5,12.),(3.,30.)):
            u,v=project(x,z)
            np.testing.assert_allclose(ground_point(u,v),[x,z],atol=1e-9)
        self.assertIsNone(ground_point(.5,.4))

    def test_parallel_paint_estimates_scale_without_fixed_camera_height(self):
        z=np.array([5.,6.,7.,8.,10.,12.,15.])
        width=1.75/z
        rows=[(.428+.746/d,.5-w/2,.5+w/2) for d,w in zip(z,width)]
        fit=fit_corridor(rows)
        self.assertAlmostEqual(fit['horizon'],.428,places=6)
        self.assertAlmostEqual(fit['scale'],.746,places=6)
        calibration=GroundCalibration()
        old=calibration.update(None)
        self.assertEqual(calibration.update([(0,0,0)]*7),old)

    def test_invalid_calibration_is_rejected(self):
        for scale in (-1.,float('nan'),10.):
            with self.assertRaises(ValueError):set_projection(dict(horizon=.43,scale=scale))


if __name__=='__main__':unittest.main()
