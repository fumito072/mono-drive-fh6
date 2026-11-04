import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from mono_drive.forza_longitudinal import LongitudinalResponse


class LongitudinalTests(unittest.TestCase):
    def test_only_validated_response_for_matching_vehicle_changes_feedforward(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'models/vehicle_profiles';folder.mkdir(parents=True)
            profile=dict(accepted=True,car_ordinal=3819,train_samples=1000,test_samples=300,
                held_out_runs=['a','b','c'],parameters=dict(gain=7.,drag_constant=.6,drag_speed_squared=.004),
                baseline=dict(mae=.6,rmse=.7),candidate=dict(mae=.25,rmse=.35))
            path=folder/'longitudinal_3819.json';path.write_text(json.dumps(profile))
            response=LongitudinalResponse(root)
            self.assertAlmostEqual(response.feedforward(0.,60.,3819),(.6+.004*(60/3.6)**2)/7.)
            self.assertAlmostEqual(response.feedforward(1.,30.,9999),.29)
            profile['accepted']=False;path.write_text(json.dumps(profile))
            self.assertAlmostEqual(response.feedforward(1.,30.,3819),.29)

    def test_invalid_numbers_and_weak_validation_keep_original_control(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'models/vehicle_profiles';folder.mkdir(parents=True)
            profile=dict(accepted=True,car_ordinal=3819,train_samples=1000,test_samples=300,
                held_out_runs=['a','b','c'],parameters=dict(gain=float('nan'),drag_constant=.6,drag_speed_squared=.004),
                baseline=dict(mae=.6,rmse=.7),candidate=dict(mae=.25,rmse=.35))
            path=folder/'longitudinal_3819.json';path.write_text(json.dumps(profile))
            self.assertAlmostEqual(LongitudinalResponse(root).feedforward(1.,30.,3819),.29)
            profile['parameters']['gain']=7.;profile['candidate']['mae']=.59
            path.write_text(json.dumps(profile))
            self.assertAlmostEqual(LongitudinalResponse(root).feedforward(1.,30.,3819),.29)


if __name__=='__main__':unittest.main()
