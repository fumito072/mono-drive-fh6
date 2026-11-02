import math
import unittest
from dataclasses import replace
import numpy as np
from mono_drive.forza_scene import ScenePlan
from mono_drive.forza_lateral import PredictiveLaneControl, SteeringServo
from mono_drive.forza_lane_memory import TemporalLaneModel, align_plan


def lane(offset=0.):
    z=np.array([4.,6.,10.,16.,24.])
    path=np.column_stack((z,np.full_like(z,offset)))
    return ScenePlan(path,0.,50.,.9,1,0,'road_candidates',.9,
        lane_edges=tuple((v,offset-1.75,offset+1.75) for v in z))


class LateralTests(unittest.TestCase):
    def test_high_speed_steering_uses_the_measured_game_response(self):
        from mono_drive.forza_vehicle import steering_angle
        curvature=.015
        slow=math.atan(2.64*curvature)/steering_angle(25.)
        fast=math.atan(2.64*curvature)/steering_angle(45.)
        self.assertGreater(fast,slow*2.)

    def test_nominal_target_is_the_middle_of_the_selected_lane(self):
        from mono_drive.forza_lateral import center_lane_plan
        for width in (2.8,3.5,4.2):
            z=np.array([4.,6.,10.,16.,24.])
            p=replace(lane(),path=np.column_stack((z,np.full_like(z,.45))),
                      lane_edges=tuple((v,-width/2.,width/2.) for v in z))
            out=center_lane_plan(p,40.)
            np.testing.assert_allclose(out.path[:,1],0.,atol=1e-8)
            self.assertAlmostEqual(out.steer,0.,places=7)
        evasive=replace(p,reason='vehicle_avoidance')
        np.testing.assert_array_equal(center_lane_plan(evasive,40.).path,evasive.path)

    def test_combined_controller_recentres_with_realistic_wheel_lag(self):
        from mono_drive.forza_lane_guard import LaneDepartureGuard
        ctl,guard=PredictiveLaneControl(),LaneDepartureGuard()
        x,yaw,actual=-.55,-.03,0.
        lateral=[]
        for i in range(100):
            distance=np.linspace(4.,30.,12)
            dx=np.full_like(distance,-x)
            path=np.column_stack((np.sin(yaw)*dx+np.cos(yaw)*distance,
                                  np.cos(yaw)*dx-np.sin(yaw)*distance))
            edges=tuple((z,c-1.75/np.cos(yaw),c+1.75/np.cos(yaw)) for z,c in path)
            p=replace(lane(),path=path,lane_edges=edges)
            p=guard.apply(ctl.apply(p,dict(speed=30.,steer=actual*127),i*.1),30.)
            actual+=np.clip((p.steer-actual)*.1/.22,-.125,.125)
            yaw+=30/3.6/2.64*math.tan(actual*math.radians(28))*.1
            x+=30/3.6*math.sin(yaw)*.1
            lateral.append(x)
        self.assertLess(max(abs(v) for v in lateral),.65)
        self.assertLess(abs(x),.1)

    def test_forward_target_increases_with_speed_and_stays_on_observed_path(self):
        from mono_drive.forza_lateral import lookahead_target
        path=np.array([[4.,.4],[12.,.4],[20.,.4],[30.,.4]])
        slow=lookahead_target(path,10.)
        fast=lookahead_target(path,50.)
        self.assertGreater(fast[0],slow[0])
        self.assertLessEqual(fast[0],26.)
        self.assertAlmostEqual(fast[1],.4)
        short=lookahead_target(path[:2],50.)
        self.assertEqual(short[0],12.)

    def test_crosswalk_like_zigzag_is_not_a_lane_curve(self):
        from mono_drive.forza_scene import coherent_lane_shape
        self.assertFalse(coherent_lane_shape(np.array([[3.5,-.3],[4.2,.75],[6.7,.57],[10.6,-.24]])))
        z=np.array([4.,6.,10.,16.])
        self.assertTrue(coherent_lane_shape(np.column_stack((z,.01*z*z))))

    def test_servo_uses_measured_wheel_feedback(self):
        a=SteeringServo().step(.2,0.,.01)
        b=SteeringServo().step(.2,.2,.01)
        self.assertGreater(a,b)
        self.assertEqual(SteeringServo().step(0.,.2,.01),0.)

    def test_temporal_alignment_compensates_translation_and_yaw(self):
        source=dict(yaw=0.,position=[0.,0.,0.])
        target=dict(yaw=.1,position=[0.,0.,2.])
        aligned=align_plan(lane(),source,target)
        self.assertLess(aligned.path[0,1],0.)
        self.assertLess(aligned.path[0,0],4.)

    def test_bad_single_frame_does_not_replace_lane_memory(self):
        memory=TemporalLaneModel()
        ego=dict(yaw=0.,position=[0.,0.,0.])
        memory.update(lane(),ego,1.)
        output=memory.update(lane(2.5),ego,1.1)
        self.assertEqual(memory.state,'rejected_jump')
        np.testing.assert_allclose(output.path[:,1],0.)
        self.assertIsNone(memory.update(None,ego,2.))

    def test_small_aligned_lane_noise_is_filtered_without_widening(self):
        memory=TemporalLaneModel();ego=dict(yaw=0.,position=[0.,0.,0.])
        memory.update(lane(),ego,1.)
        measured=lane(.2)
        output=memory.update(measured,ego,1.1)
        index=1
        self.assertGreater(output.path[index,1],0.)
        self.assertLess(output.path[index,1],measured.path[index,1])
        np.testing.assert_allclose(np.diff(np.asarray(output.lane_edges)[:,1:]),3.5)
        # Distant fresh geometry is not smoothed from a short old observation.
        self.assertEqual(output.path[-1,1],measured.path[-1,1])


    def test_closed_loop_left_and_right_curves_with_wheel_lag(self):
        for radius,speed,physical_angle in ((40.,30.,28.),(-40.,30.,28.),
                                            (75.,45.,14.),(-75.,45.,14.)):
            ctl=PredictiveLaneControl()
            x=z=yaw=actual=0.
            errors=[]
            v=speed/3.6
            for i in range(160):
                s=np.linspace(max(0.,i*.1*v-18.),i*.1*v+45.,160)
                angle=s/radius
                dx=radius*(1-np.cos(angle))-x
                dz=radius*np.sin(angle)-z
                lx=np.cos(yaw)*dx-np.sin(yaw)*dz
                lz=np.sin(yaw)*dx+np.cos(yaw)*dz
                valid=(lz>3)&(lz<25)
                path=np.column_stack((lz[valid],lx[valid]))
                path=path[np.argsort(path[:,0])]
                path=path[np.linspace(0,len(path)-1,9).astype(int)]
                plan=replace(lane(),path=path,lane_edges=tuple((pz,px-1.75,px+1.75) for pz,px in path))
                output=ctl.apply(plan,dict(speed=speed,steer=actual*127),i*.1)
                actual+=np.clip((output.steer-actual)*.1/.22,-.125,.125)
                yaw+=v/2.64*math.tan(actual*math.radians(physical_angle))*.1
                x+=v*math.sin(yaw)*.1
                z+=v*math.cos(yaw)*.1
                errors.append(abs(math.hypot(x-radius,z)-abs(radius)))
            self.assertLess(max(errors),.30)

    def test_curved_road_reduces_speed_before_turn(self):
        p=lane()
        z=p.path[:,0]
        x=.5*.025*z*z
        p=replace(p,path=np.column_stack((z,x)),lane_edges=tuple((a,b-1.75,b+1.75) for a,b in zip(z,x)))
        out=PredictiveLaneControl().apply(p,dict(speed=50.,steer=0),0.)
        self.assertGreater(out.steer,0.)
        self.assertLess(out.limit,40.)


if __name__=='__main__':
    unittest.main()
