import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from mono_drive.forza import parse_control_request
from mono_drive.forza_speed import cruise_limit
from mono_drive.forza_scene import curve_speed, traffic_limit
from mono_drive.forza_static import static_risk


class CruiseSpeedTests(unittest.TestCase):
    def test_old_traffic_stop_does_not_latch_after_path_is_replanned(self):
        from dataclasses import replace
        from mono_drive.forza_scene import ScenePlan,geometry_speed_budget
        z=np.array([4.,8.,12.,18.,24.])
        plan=ScenePlan(np.column_stack((z,np.zeros_like(z))),0.,0.,.9,1,1,
                       'traffic_stop',.9,road_limit=60.)
        geometry=geometry_speed_budget(plan)
        self.assertEqual(geometry.limit,60.)
        self.assertNotEqual(geometry.reason,'traffic_stop')
        car=dict(x=0.,z=14.,vx=0.,vz=0.,confidence=.9)
        clear=plan.path.copy();clear[:,1]=-3.5
        self.assertEqual(traffic_limit(clear,[car],30.)[0],60.)
        self.assertLess(traffic_limit(plan.path,[car],30.)[0],60.)
        self.assertEqual(geometry_speed_budget(replace(plan,road_limit=12.)).limit,12.)
        explicit=replace(plan,reason='static_stop')
        self.assertIs(geometry_speed_budget(explicit),explicit)

    def test_start_uses_driving_runtime_and_saved_target(self):
        from mono_drive import forza_control
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root/'.venv/Scripts/python.exe'
            runtime.parent.mkdir(parents=True)
            runtime.touch()
            (root/'out').mkdir()
            (root/'out/longrun.json').write_text(json.dumps({'target_kmh': 60}))
            with patch.object(forza_control, 'ROOT', root), \
                 patch.object(forza_control.socket, 'socket'), \
                 patch.object(forza_control.subprocess, 'Popen') as launch:
                forza_control.request('drive')
            self.assertEqual(launch.call_args.args[0][0], str(runtime))
            self.assertEqual(json.loads((root/'out/control.json').read_text())['max_kmh'], 60.)

    def test_persisted_session_target_survives_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'out').mkdir()
            (root/'configs').mkdir()
            (root/'configs/forza_horizon6.json').write_text(json.dumps({'forza': {'max_kmh': 50}}))
            (root/'out/longrun.json').write_text(json.dumps({'target_kmh': 60, 'recoveries': 8}))
            self.assertEqual(cruise_limit(root), 60.)
            (root/'out/longrun.json').write_text(json.dumps({'target_kmh': 'nan'}))
            self.assertEqual(cruise_limit(root), 50.)

    def test_sixty_is_accepted_without_loosening_maneuver_bounds(self):
        self.assertEqual(parse_control_request({'action': 'drive', 'max_kmh': 60})[1], 60.)
        for request in ({'max_kmh': 61}, {'seconds': 7}, {'throttle': .7}):
            with self.assertRaises(ValueError):
                parse_control_request(dict(action='drive', **request))

    def test_clear_straight_allows_sixty_and_tight_bend_still_slows(self):
        z = np.linspace(6., 30., 12)
        straight = np.column_stack((z, np.zeros_like(z)))
        bend = np.column_stack((z, .015*z*z))
        self.assertEqual(curve_speed(straight, 60.)[0], 60.)
        self.assertLess(curve_speed(bend, 60.)[0], 50.)
        self.assertEqual(traffic_limit(straight, [], 60.)[0], 60.)
        self.assertEqual(static_risk({}, 60., 0.).limit, 60.)


if __name__ == '__main__':
    unittest.main()
