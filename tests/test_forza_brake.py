import unittest
from types import SimpleNamespace
from mono_drive.forza_brake import BrakeDemandFilter,hazard_pedals
from mono_drive.forza_analog import AnalogSmoother
from mono_drive.forza import SpeedController
from mono_drive.forza_collision import forward_collision


class BrakeTests(unittest.TestCase):
    def test_unconfirmed_prediction_tempers_gas_without_release_cycle(self):
        self.assertEqual(hazard_pedals((.1,.4,0.),30.,.4,0.,False),(.1,.18,0.))
        self.assertEqual(hazard_pedals((.1,.1,0.),30.,.4,0.,False),(.1,.1,0.))
        self.assertEqual(hazard_pedals((.1,.4,0.),30.,0.,0.,False),(.1,.4,0.))

    def test_confirmed_or_immediate_hazard_still_releases_gas(self):
        self.assertEqual(hazard_pedals((.1,.4,0.),30.,.6,.6,False),(.1,0.,.6))
        self.assertEqual(hazard_pedals((.1,.4,.2),30.,.8,.8,True),(.1,0.,.8))
        self.assertEqual(hazard_pedals((.1,.4,0.),0.,.8,.8,True),(.1,0.,0.))

    def test_tentative_hazard_cannot_cancel_curve_braking(self):
        self.assertEqual(hazard_pedals((.1,0.,.2),30.,.5,0.,False),(.1,0.,.2))

    def test_fleeting_predicted_hazard_coasts_without_brake(self):
        gate=BrakeDemandFilter()
        self.assertEqual(gate.step('car',.8,True,False,1.),(0.,False))
        self.assertEqual(gate.step('car',.8,True,False,1.1),(0.,False))
        self.assertEqual(gate.step('car',0.,False,False,1.15),(0.,False))
        self.assertEqual(gate.step('car',.8,True,False,1.3),(0.,False))

    def test_confirmed_hazard_and_immediate_contact_keep_braking(self):
        gate=BrakeDemandFilter()
        gate.step('static',.6,True,False,1.)
        self.assertEqual(gate.step('static',.6,True,False,1.21),(.6,True))
        self.assertEqual(gate.step('near',.8,True,True,1.22),(.8,True))

    def test_confirmation_does_not_consume_needed_stopping_distance(self):
        collision=forward_collision([dict(x=0.,z=20.)],50.,0.)
        self.assertTrue(collision.immediate)
        gate=BrakeDemandFilter()
        self.assertGreater(gate.step('car',collision.brake,collision.emergency,collision.immediate,1.)[0],0.)

    def test_cleared_hazard_does_not_brake_for_old_speed_target(self):
        controller=SpeedController()
        controller.target=10.
        controller.overspeed=2.
        controller.previous_speed=30.
        decision=SimpleNamespace(confidence=.9,mode='paint_pair',reverse=False,target_kmh=50.,steer=0.)
        command=controller.step(decision,30.,50.,.5,.1)
        self.assertEqual(command[2],0.)
        self.assertGreater(command[1],0.)

    def test_brake_releases_promptly_but_monotonically(self):
        smoother=AnalogSmoother()
        smoother.reset((0.,0.,.8))
        brakes=[]
        for _ in range(30):
            output=smoother.step((0.,.3,0.),.01,35.)
            brakes.append(output[2])
            self.assertFalse(output[1]>.01 and output[2]>.01)
        self.assertLess(brakes[-1],.01)
        self.assertTrue(all(b<=a for a,b in zip(brakes,brakes[1:])))


if __name__=='__main__':
    unittest.main()
