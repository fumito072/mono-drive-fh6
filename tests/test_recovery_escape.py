import unittest
from mono_drive.forza_recovery import Recovery


class RecoveryEscapeTests(unittest.TestCase):
    def test_known_contact_allows_requested_bounded_reverse_but_still_blocks_forward(self):
        recovery=Recovery()
        self.assertTrue(recovery.surface_blocks('known_failed_approach'))
        self.assertFalse(recovery.surface_blocks('known_failed_approach',requested=True))
        value={'position':[0,0,0],'forward_kmh':0.,'speed':0.,'gear':1}
        recovery.begin(value,0,10,distance_m=2,reverse_only=True,manual=True)
        self.assertFalse(recovery.surface_blocks('known_failed_approach'))
        command,handbrake,reverse=recovery.update(value,None,0,10.1)
        self.assertEqual(command[1],0.)
        self.assertGreater(command[2],0.)
        self.assertTrue(reverse)
        value.update(position=[0,0,-2],gear=0)
        recovery.update(value,None,0,11)
        recovery.update(value,None,0,12)
        self.assertEqual(recovery.phase,'complete_reverse')
        recovery.phase=''
        self.assertTrue(recovery.surface_blocks('known_failed_approach'))


if __name__=='__main__':unittest.main()
