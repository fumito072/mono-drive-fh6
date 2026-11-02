import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from mono_drive import forza_vehicle as vehicle


class VehicleTests(unittest.TestCase):
    def tearDown(self):
        vehicle._degrees=list(vehicle.BOOTSTRAP)

    def test_real_motion_updates_only_its_car_and_standstill_does_not(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/'out').mkdir()
            response=vehicle.VehicleResponse(root)
            yaw=0.
            for i in range(80):
                yaw+=25/3.6/2.64*math.tan(math.radians(30)*.2)*.1
                response.update(dict(car_ordinal=3819,speed=25.,forward_kmh=25.,steer=25.4,yaw=yaw),i*.1+100)
            first=response.status()
            self.assertGreater(first['measured_samples'],50)
            self.assertLess(math.degrees(vehicle.steering_angle(25.)),36.)
            response.update(dict(car_ordinal=99,speed=0.,forward_kmh=0.,steer=25.4,yaw=yaw),110.)
            self.assertEqual(response.status()['measured_samples'],0)
            self.assertEqual(vehicle._degrees,vehicle.BOOTSTRAP)
            self.assertTrue((root/'models/vehicle_profiles/car_3819.json').exists())
