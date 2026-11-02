import unittest
import numpy as np
from mono_drive.forza_collision import forward_collision
from mono_drive.forza_vision import visible_detection


class CollisionTests(unittest.TestCase):
    def test_parked_kerb_bicycle_does_not_occupy_a_car_envelope(self):
        from mono_drive.forza_scene import traffic_limit
        path=np.array([[4.,0.],[12.,0.],[30.,0.]])
        bicycle=dict(kind='bicycle',x=-1.66,z=4.63,vx=0.,vz=0.,
                     motion_reliable=True,track_frames=200,track_age_s=.75)
        for speed in (0.,20.,50.):
            self.assertEqual(traffic_limit(path,[bicycle],speed)[0],60.)
            self.assertEqual(forward_collision([bicycle],speed,0.).brake,0.)

    def test_bicycle_in_path_or_crossing_still_blocks(self):
        from mono_drive.forza_scene import traffic_limit
        path=np.array([[4.,0.],[12.,0.],[30.,0.]])
        for x,vx in ((0.,0.),(1.8,-2.)):
            bicycle=dict(kind='bicycle',x=x,z=8.,vx=vx,vz=0.,motion_reliable=True)
            self.assertLess(traffic_limit(path,[bicycle],30.)[0],30.)
            self.assertTrue(forward_collision([bicycle],30.,0.).emergency)

    def test_distant_guard_follows_the_chosen_curve_but_near_guard_remains(self):
        path=np.array([[1.,0.],[6.,-1.5],[12.,-3.5],[30.,-4.5]])
        car=dict(x=0.,z=20.)
        self.assertGreater(forward_collision([car],30.,0.).brake,.1)
        self.assertEqual(forward_collision([car],30.,0.,path=path).brake,0.)
        self.assertTrue(forward_collision([dict(x=0.,z=3.)],30.,0.,path=path).emergency)

    def test_static_guard_uses_curve_and_preserves_near_body_contact(self):
        from mono_drive.forza_static import static_risk
        road=np.full((216,384),255,np.uint8)
        obstacle=np.zeros_like(road)
        obstacle[100:103,183:211]=255
        road[obstacle>0]=0
        scene=dict(road=road,obstacles=obstacle)
        path=np.array([[1.,0.],[6.,-2.5],[12.,-4.],[30.,-4.]])
        self.assertTrue(static_risk(scene,50.,0.).emergency)
        self.assertFalse(static_risk(scene,50.,0.,path=path).emergency)
        obstacle[:]=0
        obstacle[117:126,160:225]=255
        road[:]=255
        road[obstacle>0]=0
        self.assertTrue(static_risk(scene,50.,0.,path=path).emergency)

    def test_close_car_cannot_disappear_below_windscreen(self):
        visibility = np.zeros((216,384),bool)
        visibility[75:130,100:350] = True
        side = np.zeros_like(visibility)
        self.assertEqual(visible_detection((500,320,1500,850),1920,1080,visibility,side),'front')
        self.assertIsNone(visible_detection((0,180,1265,854),1920,1080,visibility,side))
        self.assertIsNone(visible_detection((2,5,1339,850),1920,1080,visibility,side))
        self.assertIsNone(visible_detection((0,0,100,80),1920,1080,visibility,side))

    def test_near_car_and_expansion_stop_forward_input(self):
        self.assertTrue(forward_collision([dict(x=0,z=3)],10,0).emergency)
        car = dict(x=.1,z=12,ttc=.8,box=[.4,.4,.7,.6])
        self.assertTrue(forward_collision([car],30,0).emergency)

    def test_same_lane_hazard_is_predicted_but_adjacent_lane_is_clear(self):
        self.assertTrue(forward_collision([dict(x=0,z=20)],50,0).emergency)
        self.assertFalse(forward_collision([dict(x=3.7,z=20)],50,0).emergency)

    def test_overtaking_car_pulling_away_does_not_brake(self):
        from mono_drive.forza_scene import traffic_limit
        path = np.array([[4.,0.],[12.,0.],[30.,0.]])
        car = dict(x=.4,z=3.,vx=0.,vz=16.,moving=True,motion_reliable=True,
                   ttc=.6,box=[.4,.4,.65,.6])
        self.assertEqual(traffic_limit(path,[car],40)[0],60.)
        self.assertFalse(forward_collision([car],40,0).emergency)

    def test_alongside_and_just_passed_receding_cars_do_not_stop_us(self):
        from mono_drive.forza_scene import traffic_limit
        path=np.array([[4.,0.],[12.,0.],[30.,0.]])
        for z in (-2.,.5,1.9,5.):
            with self.subTest(z=z):
                car=dict(x=1.5,z=z,vx=0.,vz=16.,motion_reliable=True,moving=True)
                self.assertEqual(traffic_limit(path,[car],40.)[0],60.)
                collision=forward_collision([car],40.,0.)
                self.assertFalse(collision.emergency)
                self.assertEqual(collision.brake,0.)

    def test_parallel_same_speed_vehicle_is_not_a_fixed_wall(self):
        car=dict(x=.2,z=5.,vx=0.,vz=10.,motion_reliable=True,moving=True)
        self.assertEqual(forward_collision([car],36.,0.).brake,0.)

    def test_faster_vehicle_cutting_in_is_still_checked(self):
        car=dict(x=3.,z=3.,vx=-2.,vz=12.,motion_reliable=True,moving=True)
        self.assertTrue(forward_collision([car],36.,0.).emergency)

    def test_matching_a_slow_moving_lead_does_not_demand_zero_speed(self):
        from mono_drive.forza_scene import traffic_limit
        path=np.array([[4.,0.],[12.,0.],[30.,0.]])
        car=dict(x=0.,z=3.5,vx=0.,vz=9.,motion_reliable=True,moving=True)
        self.assertGreater(traffic_limit(path,[car],36.)[0],25.)
        self.assertFalse(forward_collision([car],36.,0.).emergency)

    def test_unconfirmed_side_detection_waits_but_immediate_centre_hazard_does_not(self):
        car=dict(x=1.8,z=4.,confidence=.9,track_frames=1,track_age_s=0.,motion_reliable=False)
        self.assertFalse(forward_collision([car],36.,0.).emergency)
        self.assertTrue(forward_collision([dict(car,x=0.)],36.,0.).emergency)

    def test_box_growth_alone_does_not_stop_for_adjacent_car(self):
        car = dict(x=3.7,z=9.,vx=0.,vz=-10.,moving=True,ttc=.3,box=[.4,.4,.7,.6])
        self.assertFalse(forward_collision([car],40,0).emergency)

    def test_slower_lead_car_still_requires_speed_reduction(self):
        from mono_drive.forza_scene import traffic_limit
        path = np.array([[4.,0.],[12.,0.],[30.,0.]])
        car = dict(x=0.,z=14.,vx=0.,vz=4.,motion_reliable=True)
        self.assertLess(traffic_limit(path,[car],50)[0],30.)
        self.assertTrue(forward_collision([car],50,0).emergency)

    def test_crossing_car_is_predicted(self):
        car = dict(x=4,z=12,vx=-2,vz=0,moving=True)
        self.assertTrue(forward_collision([car],30,0).emergency)

    def test_avoidance_stays_inside_measured_lane(self):
        from mono_drive.forza_collision import avoid_in_lane
        from mono_drive.forza_scene import ScenePlan,inside_lane
        path = np.array([[6.,0.],[10.,0.],[14.,0.],[20.,0.]])
        plan = ScenePlan(path,0.,50.,.9,1,1,'paint_pair',.9,
                         lane_edges=tuple((z,-1.75,1.75) for z in path[:,0]))
        guarded = avoid_in_lane(plan,[dict(x=1.8,z=12,vx=0,vz=0)],20,np.full((216,384),255,np.uint8))
        self.assertEqual(guarded.reason,'vehicle_avoidance')
        self.assertLess(guarded.steer,0.)
        self.assertTrue(inside_lane(guarded.path,np.full(4,-1.75),np.full(4,1.75)))

    def test_clear_steering_alternative_restores_cruise_target(self):
        from mono_drive.forza_collision import avoid_in_lane
        from mono_drive.forza_scene import ScenePlan
        path = np.array([[6.,0.],[10.,0.],[14.,0.],[20.,0.]])
        plan = ScenePlan(path,0.,5.,.9,1,1,'traffic_follow',.9,road_limit=50.,
                         lane_edges=tuple((z,-1.75,1.75) for z in path[:,0]))
        car = dict(x=1.8,z=12,vx=0,vz=0)
        clear = avoid_in_lane(plan,[car],20,np.full((216,384),255,np.uint8))
        self.assertEqual(clear.reason,'vehicle_avoidance')
        self.assertEqual(clear.limit,50.)


if __name__ == '__main__':
    unittest.main()
