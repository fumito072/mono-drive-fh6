import unittest
from mono_drive.forza_recovery import Recovery


class RecoveryTests(unittest.TestCase):
    def value(self,speed=0.,gear=1,z=0.):
        return dict(speed=abs(speed),forward_kmh=speed,gear=gear,accel=0,position=[0.,0.,z])

    def test_lost_road_after_moving_does_not_reverse_without_applied_throttle(self):
        recovery = Recovery()
        self.assertFalse(recovery.observe_stall(self.value(30),.3,1.))
        self.assertFalse(recovery.observe_stall(self.value(),0.,2.,guidance_ok=False))
        self.assertFalse(recovery.observe_stall(self.value(),0.,4.1,guidance_ok=False))
        self.assertFalse(recovery.observe_stall(self.value(),0.,5.,guidance_ok=False,traffic_blocked=True))

    def test_initial_park_does_not_trigger_blind_reverse(self):
        recovery = Recovery()
        self.assertFalse(recovery.observe_stall(self.value(),0.,20.,guidance_ok=False))

    def test_gear_delay_does_not_end_reverse_after_only_short_distance(self):
        recovery = Recovery()
        recovery.begin(self.value(),0.,0.,distance_m=2.,reverse_only=True)
        recovery.update(self.value(-1.,gear=0,z=-.4),None,0.,9.)
        self.assertEqual(recovery.phase,'reverse')
        recovery.update(self.value(0.,gear=0,z=-2.),None,0.,14.)
        self.assertEqual(recovery.phase,'settle')

    def test_unknown_road_is_not_a_forward_acceleration_command(self):
        recovery = Recovery()
        recovery.begin(self.value(),.5,0.)
        recovery.phase,recovery.started = 'settle',1.
        command,park,reverse = recovery.update(self.value(),None,.5,2.)
        self.assertEqual(command,(0.,0.,0.))
        self.assertTrue(park)
        self.assertEqual(recovery.phase,'assess')

    def test_larger_reverse_has_a_distance_scaled_time_budget(self):
        recovery = Recovery()
        recovery.begin(self.value(),0.,0.,distance_m=6.,reverse_only=True)
        recovery.update(self.value(-2.,gear=0,z=-3.),None,0.,21.)
        self.assertEqual(recovery.phase,'reverse')
        recovery.update(self.value(0.,gear=0,z=-6.),None,0.,25.)
        self.assertEqual(recovery.phase,'settle')


if __name__ == '__main__':
    unittest.main()
