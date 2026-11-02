import unittest
from types import SimpleNamespace
from mono_drive.forza import SpeedController
from mono_drive.forza_relaunch import RelaunchAssist
from mono_drive.forza_analog import AnalogSmoother


class RelaunchTests(unittest.TestCase):
    def test_stalled_pedal_is_released_then_reapplied_smoothly(self):
        assist, smoother = RelaunchAssist(), AnalogSmoother()
        smoother.reset((.1, .6, 0.))
        released, applied = [], []
        for _ in range(700):
            command = assist.step((.1, .6, 0.), 0., 60., .01)
            output = smoother.step(command, .01, 0.)
            if assist.phase == 'release':
                released.append(output[1])
            if assist.phase == 'launch':
                applied.append(output[1])
            self.assertEqual(output[2], 0.)
        self.assertLess(min(released), .01)
        self.assertGreater(max(applied), .35)
        self.assertLessEqual(max(applied), .4501)

    def test_three_failed_attempts_yield_to_existing_recovery(self):
        assist = RelaunchAssist()
        for _ in range(220):
            assist.step((0., .4, 0.), 0., 60., .1)
        self.assertEqual(assist.attempts, 3)
        self.assertFalse(assist.blocks_reverse)
        self.assertEqual(assist.phase, 'idle')

    def test_moving_or_blocked_resets_attempt(self):
        assist = RelaunchAssist()
        for _ in range(25):
            assist.step((0., .4, 0.), 0., 60., .1)
        self.assertTrue(assist.blocks_reverse)
        original = (0., .3, 0.)
        self.assertEqual(assist.step(original, 3., 60., .1), original)
        self.assertEqual(assist.attempts, 0)
        for _ in range(80):
            self.assertEqual(assist.step((0., 0., .2), 0., 60., .1), (0., 0., .2))
        self.assertFalse(assist.blocks_reverse)

    def test_no_launch_into_stop_traffic_or_unobserved_road(self):
        for mode, confidence, target in [('traffic_stop', .9, 60), ('lost', 0., 60),
                                         ('lane_guard', .9, 0), ('blocked', .9, 60)]:
            with self.subTest(mode=mode):
                controller = SpeedController()
                decision = SimpleNamespace(mode=mode, confidence=confidence, target_kmh=target,
                                           steer=0., reverse=False)
                for _ in range(150):
                    self.assertEqual(controller.step(decision, 0., 60., .5, .1)[1], 0.)
                self.assertEqual(controller.relaunch.attempts, 0)


if __name__ == '__main__':
    unittest.main()
