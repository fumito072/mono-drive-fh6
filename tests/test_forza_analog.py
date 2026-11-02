import unittest
import math
from types import SimpleNamespace
import numpy as np
from mono_drive.forza_analog import AnalogSmoother
from mono_drive.forza_collision import forward_collision
from mono_drive.forza import SpeedController,final_forward_pedals
from mono_drive.forza_traffic import predict_traffic


class AnalogTests(unittest.TestCase):
    def test_accelerator_follows_clear_road_command_with_bounded_rate(self):
        smooth=AnalogSmoother();outputs=[]
        for _ in range(160):outputs.append(smooth.step((0.,.6,0.),.01,30.)[1])
        self.assertGreater(outputs[-1],.59)
        self.assertLessEqual(max(np.diff(outputs)),.0051)
        self.assertEqual(smooth.value[2],0.)

    def test_lane_centre_and_analog_steering_at_sixty(self):
        from mono_drive.forza_lateral import PredictiveLaneControl,SteeringServo
        from mono_drive.forza_vehicle import steering_angle
        from mono_drive.forza_scene import ScenePlan
        control,servo,smooth=PredictiveLaneControl(),SteeringServo(),AnalogSmoother()
        x,yaw,actual=.25,.005,0.
        errors=[]
        for tick in range(80):
            distance=np.linspace(4.,32.,10)
            local_z=np.sin(yaw)*(-x)+np.cos(yaw)*distance
            local_x=np.cos(yaw)*(-x)-np.sin(yaw)*distance
            plan=ScenePlan(np.column_stack((local_z,local_x)),0.,60.,.9,1,0,'paint_pair',.9,
                lane_edges=tuple((z,c-1.75/np.cos(yaw),c+1.75/np.cos(yaw)) for z,c in zip(local_z,local_x)))
            desired=control.apply(plan,dict(speed=60.,steer=actual*127),tick*.1).steer
            for _ in range(10):
                raw=smooth.step((servo.step(desired,actual,.01),0.,0.),.01,60.)[0]
                wheel=math.copysign(max(0.,(abs(raw)-.28)/.70),raw)
                actual+=np.clip((wheel-actual)*.01/.22,-.0125,.0125)
                yaw+=60/3.6/2.64*math.tan(actual*steering_angle(60.))*.01
                x+=60/3.6*math.sin(yaw)*.01
                errors.append(abs(x))
        self.assertLess(max(errors),.5)
        self.assertLess(abs(x),.12)

    def test_clear_road_reaches_sixty_without_brake_hunting(self):
        controller,smoother=SpeedController(),AnalogSmoother()
        decision=SimpleNamespace(confidence=.9,mode='paint_pair',reverse=False,target_kmh=60.,steer=0.)
        speed=0.
        command=(0.,0.,0.)
        speeds=[]
        brakes=[]
        for tick in range(4000):
            if tick%15==0:
                command=controller.step(decision,speed*3.6,60.,.5,.15)
            output=smoother.step(command,.01,speed*3.6)
            speed=max(0.,speed+(4.*output[1]-10.*output[2]-.15)*.01)
            speeds.append(speed*3.6)
            brakes.append(output[2])
        self.assertGreater(min(speeds[2500:]),57.)
        self.assertLess(max(speeds),63.)
        self.assertLess(max(brakes),.02)

    def test_acceleration_feedback_reduces_throttle_before_overshoot(self):
        controller=SpeedController()
        controller.target=40.
        decision=SimpleNamespace(confidence=.9,mode='paint_pair',reverse=False,target_kmh=50.,steer=0.)
        first=controller.step(decision,20.,50.,.5,.1)[1]
        accelerating=controller.step(decision,24.,50.,.5,.1)[1]
        self.assertLess(accelerating,first)
        self.assertLessEqual(controller.desired_acceleration,1.6)

    def test_visibility_stop_does_not_start_reverse_without_forward_effort(self):
        from mono_drive.forza_recovery import Recovery
        recovery=Recovery()
        moving=dict(speed=10.,accel=0)
        recovery.observe_stall(moving,0.,1.)
        stopped=dict(speed=0.,accel=0)
        self.assertFalse(recovery.observe_stall(stopped,0.,2.,guidance_ok=False))
        self.assertFalse(recovery.observe_stall(stopped,0.,6.,guidance_ok=False))
        stopped['accel']=90
        self.assertFalse(recovery.observe_stall(stopped,.3,7.))
        self.assertTrue(recovery.observe_stall(stopped,.3,9.1))

    def test_invalid_target_releases_output(self):
        smooth=AnalogSmoother()
        smooth.reset((.2,.3,0.))
        self.assertEqual(smooth.step((float('nan'),.4,0.),.01,30.),(0.,0.,0.))

    def test_lower_speed_limit_releases_throttle_before_late_braking(self):
        controller=SpeedController()
        controller.target=50.
        decision=SimpleNamespace(confidence=.9,mode='paint_pair',reverse=False,target_kmh=15.,steer=.1)
        self.assertEqual(controller.step(decision,40.,50.,.5,.1)[1:],(0.,0.))
        for _ in range(12):
            command=controller.step(decision,35.,50.,.5,.1)
        self.assertEqual(command[1],0.)
        self.assertGreater(command[2],0.)
        self.assertLessEqual(command[2],.4)

    def test_changed_steering_target_cannot_keep_winding_up(self):
        smooth=AnalogSmoother()
        for _ in range(20):
            smooth.step((.65,0.,0.),.01,30.)
        previous=smooth.value[0]
        for _ in range(150):
            value=smooth.step((0.,0.,0.),.01,30.)[0]
            self.assertGreaterEqual(value,-1e-9)
            self.assertLessEqual(value,previous+1e-9)
            previous=value
        self.assertLess(previous,.001)

    def test_emergency_release_cannot_increase_the_brake(self):
        smooth=AnalogSmoother()
        for _ in range(15):
            smooth.step((0.,0.,.8),.01,30.,emergency=True)
        previous=smooth.value[2]
        for _ in range(160):
            brake=smooth.step((0.,.5,0.),.01,30.)[2]
            self.assertLessEqual(brake,previous+1e-9)
            previous=brake
        self.assertEqual(previous,0.)

    def test_smoothed_steering_stays_inside_a_curved_lane(self):
        from mono_drive.forza_lateral import PredictiveLaneControl,SteeringServo
        from mono_drive.forza_scene import ScenePlan
        for radius in (40.,-40.):
            with self.subTest(radius=radius):
                control,servo,smooth=PredictiveLaneControl(),SteeringServo(),AnalogSmoother()
                x=z=yaw=actual=0.
                speed=30/3.6
                errors=[]
                for tick in range(160):
                    arc=np.linspace(max(0.,tick*.1*speed-18),tick*.1*speed+45,160)
                    angle=arc/radius
                    dx=radius*(1-np.cos(angle))-x
                    dz=radius*np.sin(angle)-z
                    local_x=np.cos(yaw)*dx-np.sin(yaw)*dz
                    local_z=np.sin(yaw)*dx+np.cos(yaw)*dz
                    valid=(local_z>3)&(local_z<25)
                    path=np.column_stack((local_z[valid],local_x[valid]))
                    path=path[np.argsort(path[:,0])]
                    path=path[np.linspace(0,len(path)-1,9).astype(int)]
                    plan=ScenePlan(path,0.,50.,.9,1,0,'paint_pair',.9,
                                   lane_edges=tuple((pz,px-1.75,px+1.75) for pz,px in path))
                    desired=control.apply(plan,dict(speed=30.,steer=actual*127),tick*.1).steer
                    for _ in range(10):
                        raw=servo.step(desired,actual,.01)
                        raw=smooth.step((raw,0.,0.),.01,30.)[0]
                        wheel=math.copysign(max(0.,(abs(raw)-.28)/.70),raw)
                        actual+=np.clip((wheel-actual)*.01/.22,-.0125,.0125)
                        yaw+=speed/2.64*math.tan(actual*math.radians(35))*.01
                        x+=speed*math.sin(yaw)*.01
                        z+=speed*math.cos(yaw)*.01
                        errors.append(abs(math.hypot(x-radius,z)-abs(radius)))
                self.assertLess(max(errors),.80)

    def test_steering_reversal_has_bounded_rate_and_acceleration(self):
        smooth=AnalogSmoother()
        commands=[]
        for target in ([.65]*130+[-.65]*200+[0.]*160):
            commands.append(smooth.step((target,0.,0.),.01,50.)[0])
        velocity=np.diff(commands)/.01
        self.assertLessEqual(np.max(np.abs(velocity)),1.001)
        # Ignore the last sub-step used to settle precisely on a target.
        moving=(np.abs(velocity[:-1])>.08)&(np.abs(velocity[1:])>.08)
        self.assertLessEqual(np.max(np.abs(np.diff(velocity)[moving])/.01),5.01)
        self.assertLess(abs(commands[-1]),.001)

    def test_pedals_are_analog_and_never_pressed_together(self):
        smooth=AnalogSmoother()
        values=[]
        for target in ([(0.,.6,0.)]*250+[(0.,0.,.35)]*350+[(0.,.5,0.)]*350):
            values.append(smooth.step(target,.01,30.))
        values=np.array(values)
        self.assertTrue(np.all((values[:,1]<=.01)|(values[:,2]<=.01)))
        self.assertGreater(len(np.unique(np.round(values[:,1],3))),100)
        self.assertGreater(len(np.unique(np.round(values[:,2],3))),80)
        self.assertLessEqual(np.max(np.abs(np.diff(values[:,1]))),.0076)
        brake_steps=np.diff(values[:,2])
        self.assertLessEqual(np.max(brake_steps),.0051)
        self.assertGreaterEqual(np.min(brake_steps),-.0601)

    def test_ordinary_predicted_hazard_does_not_demand_full_braking(self):
        moderate=forward_collision([dict(x=0.,z=30.)],30.,0.)
        urgent=forward_collision([dict(x=0.,z=10.)],30.,0.)
        self.assertFalse(moderate.emergency)
        self.assertGreater(moderate.brake,0.)
        self.assertLess(moderate.brake,.4)
        self.assertTrue(urgent.emergency)
        self.assertGreaterEqual(urgent.brake,.55)

    def test_stationary_vehicle_is_stopped_for_before_contact(self):
        smooth=AnalogSmoother()
        controller=SpeedController()
        controller.target=50.
        speed=50/3.6
        gap=60.
        for _ in range(1600):
            cars=[dict(x=0.,z=gap)]
            path=np.array([[4.,0.],[20.,0.],[60.,0.]])
            limit=predict_traffic(path,cars,speed*3.6).limit
            decision=SimpleNamespace(confidence=.9,mode='paint_pair',reverse=False,target_kmh=limit,steer=0.)
            desired=controller.step(decision,speed*3.6,50.,.5,.01)
            hazard=forward_collision(cars,speed*3.6,0.)
            if hazard.brake>0:
                desired=(0.,0.,max(desired[2],hazard.brake))
            output=smooth.step(desired,.01,speed*3.6,emergency=hazard.emergency)
            output,park=final_forward_pedals(output,dict(forward_kmh=speed*3.6),emergency=hazard.emergency)
            smooth.reconcile(output)
            speed=max(0.,speed+(3.*output[1]-10.*output[2]-(10. if park else 0.)-.1)*.01)
            gap-=speed*.01
            if speed<.02:
                break
        self.assertLess(speed,.02)
        self.assertGreater(gap,4.)


if __name__=='__main__':
    unittest.main()
