import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mono_drive.forza_scene import CandidatePlanner,SceneReader,project,traffic_limit,footprint_quality,feasible_curve_speed,inside_lane,curve_speed
from mono_drive.forza_vision import TrafficTracks


def straight_road(left=-1.8,right=5.2):
    image = np.zeros((216,384),dtype=np.uint8)
    for y in range(216):
        v = y/215
        if v <= .435:
            continue
        z = .575/(v-.435)
        a,_ = project(left,z)
        b,_ = project(right,z)
        image[y,max(0,int(a*383)):min(384,int(b*383)+1)] = 250
    return image


class SceneTests(unittest.TestCase):
    def test_shorter_path_retries_reuse_the_same_frame_evidence(self):
        from unittest.mock import patch
        from mono_drive import forza_scene
        frame=np.full((216,384,3),65,np.uint8)
        with patch.object(forza_scene,'visible_road_evidence',wraps=forza_scene.visible_road_evidence) as evidence:
            result=CandidatePlanner().plan(straight_road(-.8,.8),[],
                dict(position=[0,0,0],yaw=0.,speed=20.),10.,frame=frame)
        self.assertIsNone(result)
        self.assertEqual(evidence.call_count,2)

    def test_real_paint_spacing_guides_lane_without_horizon_assumption(self):
        import cv2
        from mono_drive.forza_scene import paint_plan
        frame = np.full((216,384,3),65,np.uint8)
        road = np.zeros((216,384),np.uint8)
        for y in range(92,130):
            left = 192-2*(y-90)
            right = 192+2*(y-90)
            road[y,left:min(384,2*right-left)] = 250
        cv2.line(frame,(188,92),(114,129),(245,245,245),2)
        cv2.line(frame,(196,92),(270,129),(0,200,250),2)
        plan = paint_plan(frame,road,[],20)
        self.assertIsNotNone(plan)
        self.assertLess(abs(plan.steer),.15)
        self.assertGreater(len(plan.image_path),3)
        self.assertLessEqual(plan.limit,60)

    def test_visible_traffic_selects_connected_road_side(self):
        from mono_drive.forza_scene import bounds
        road = np.zeros((216,384),np.uint8)
        road[95:130,110:175] = 250
        road[95:130,215:380] = 250
        selected = bounds(road,seed_x=.75)
        self.assertGreater(len(selected),0)
        self.assertGreater(selected[0,1],0.)
    def test_curve_speed_uses_lateral_acceleration_and_braking_distance(self):
        self.assertEqual(feasible_curve_speed([0.],[0.],60),60)
        tight = feasible_curve_speed([1/20],[0],50)
        broad = feasible_curve_speed([1/50],[0],50)
        self.assertAlmostEqual(tight,3.6*math.sqrt(.22*9.81*20))
        self.assertGreater(broad,tight)
        self.assertGreater(feasible_curve_speed([1/20],[30],50),tight)
        self.assertEqual(curve_speed(np.array([[50.,0.],[60.,0.]]),20),(0.,0.))
        self.assertEqual(curve_speed(np.array([[10.,0.]]),20),(0.,0.))

    def test_small_row_noise_is_not_a_curve_speed_restriction(self):
        from mono_drive.forza_scene import smooth_lane_reference
        z = np.array([6.,8.,10.,12.,16.,22.,30.])
        noisy = np.array([.0,.18,-.14,.22,-.2,.1,0.])
        clean = smooth_lane_reference(z,noisy,noisy-1.75,noisy+1.75)
        self.assertGreater(curve_speed(np.column_stack((z,clean)),50)[0],49.)
        self.assertGreater(curve_speed(np.column_stack((z,noisy)),50)[0],49.)
        self.assertTrue(inside_lane(np.column_stack((z,clean)),noisy-1.75,noisy+1.75))
        bend = .008*z*z
        clean_bend = smooth_lane_reference(z,bend,bend-2.5,bend+2.5)
        self.assertLess(curve_speed(np.column_stack((z,clean_bend)),50)[0],45.)

    def test_full_body_cannot_cross_divider(self):
        path = np.array([[6.,0.],[8.,0.],[10.,0.]])
        self.assertTrue(inside_lane(path,np.full(3,-1.8),np.full(3,1.8)))
        self.assertFalse(inside_lane(path,np.full(3,-1.8),np.full(3,.8)))
    def test_road_footprint_and_left_lane(self):
        planner = CandidatePlanner()
        ego = dict(position=[0,0,0],yaw=0.,speed=20.)
        plan = planner.plan(straight_road(),[],ego,10.)
        self.assertIsNotNone(plan)
        self.assertLess(abs(plan.steer),.1)
        self.assertGreater(plan.limit,45)
        self.assertGreaterEqual(plan.candidate_count,1)
        self.assertTrue(np.all(plan.path[:,1]+.93 < 1.7))
        self.assertIsNone(planner.plan(np.zeros((216,384),np.uint8),[],ego,11.))
        # Center points alone would fit, but the vehicle's corners do not.
        narrow = straight_road(-.25,.25)
        path = np.array([[z,0.] for z in (8,12,16,22)])
        self.assertLess(footprint_quality(narrow,path)[0],.42)

    def test_lane_middle_adapts_to_width(self):
        ego = dict(position=[0,0,0],yaw=0.,speed=10.)
        plan = CandidatePlanner().plan(straight_road(-2,6),[],ego,10.)
        # Eight-metre two-lane road: left lane centre is zero, not a fixed edge offset.
        self.assertIsNotNone(plan)
        self.assertLess(abs(float(np.median(plan.path[:,1]))),.2)

    def test_lead_and_opposing_vehicle(self):
        path = np.array([[6.,0.],[12.,0.],[22.,0.],[40.,0.]])
        self.assertEqual(traffic_limit(path,[dict(x=0,z=4,vx=0,vz=0)],30)[0],0)
        self.assertLess(traffic_limit(path,[dict(x=0,z=14,vx=0,vz=0)],30)[0],30)
        # Opposing traffic in the other lane does not impose a same-lane stop.
        self.assertEqual(traffic_limit(path,[dict(x=3.5,z=14,vx=0,vz=-10)],30)[0],60)
        left_path = path.copy()
        left_path[:,1] = -1
        side_car = dict(view='side_left',box=[0,.4,.1,.6])
        # A side-window car constrains lateral steering, not cruising speed.
        self.assertEqual(traffic_limit(left_path,[side_car],20)[0],60)
        self.assertEqual(traffic_limit(path,[side_car],20)[0],60)

    def test_brief_paint_gap_requires_current_road_and_limited_motion(self):
        from mono_drive.forza_lane import LaneFusion
        from mono_drive.forza_scene import ScenePlan
        fusion = LaneFusion()
        path = np.array([[6.,0.],[10.,0.],[14.,0.]])
        pixels = ((.5,.53),(.5,.50),(.5,.48))
        previous = ScenePlan(path,.1,20.,.8,1,0,'paint_pair',.9,3,pixels)
        fusion.scene_cache = (previous,np.zeros(3),0.,10.)
        scene = dict(road=np.full((216,384),255,np.uint8),cars=[])
        ego = dict(position=[0,0,0],yaw=0.,speed=8.)
        self.assertIsNotNone(fusion.bridge_scene(scene,ego,10.2))
        self.assertIsNone(fusion.bridge_scene(scene,ego,10.8))
        ego['position'] = [2,0,0]
        self.assertIsNone(fusion.bridge_scene(scene,ego,10.2))
        ego['position'] = [0,0,0]
        scene['road'][:] = 0
        self.assertIsNone(fusion.bridge_scene(scene,ego,10.2))

    def test_minimap_direction_and_front_road_gate(self):
        import cv2
        from mono_drive.forza_minimap import road_hint
        from mono_drive.forza_scene import minimap_plan
        frame = np.zeros((1080,1920,3),np.uint8)
        self.assertIsNone(road_hint(frame))
        cv2.polylines(frame,[np.array([[196,927],[160,901],[125,890],[85,890]],np.int32)],False,(150,150,150),8)
        hint = road_hint(frame)
        self.assertIsNotNone(hint)
        self.assertLess(hint.angle,-.6)
        self.assertGreaterEqual(hint.confidence,.65)
        raw = np.full((216,384),255,np.uint8)
        self.assertIsNone(minimap_plan(frame,np.zeros_like(raw),raw,[],0))
        plan = minimap_plan(frame,raw,raw,[],0)
        self.assertIsNotNone(plan)
        self.assertLess(plan.steer,0)
        self.assertLessEqual(plan.limit,20)
        self.assertEqual(minimap_plan(frame,raw,raw,[dict(x=0,z=3,vx=0,vz=0)],10).limit,0)

    def test_stale_scene_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'scene.npz'
            np.savez(file,road=straight_road(),cars=json.dumps([]),captured=10.,inference_ms=20.)
            reader = SceneReader(file)
            self.assertIsNotNone(reader.read(10.2))
            self.assertIsNone(reader.read(10.6))
            self.assertIsNone(reader.read(9.9))

    def test_static_and_moving_tracks(self):
        tracks = TrafficTracks()
        ego = dict(position=[0,0,0],yaw=0.)
        u,v = project(0,15)
        d = dict(box=[u-.03,v-.05,u+.03,v],confidence=.9,kind='car')
        for i in range(8):
            cars = tracks.update([d],ego,10+i*.15)
        self.assertFalse(cars[0]['moving'])
        self.assertEqual(cars[0]['trail'],[])
        for i in range(10):
            u,v = project(0,15+i*.6)
            d['box'] = [u-.03,v-.05,u+.03,v]
            cars = tracks.update([d],ego,11.2+i*.15)
        self.assertTrue(cars[0]['moving'])
        self.assertGreater(len(cars[0]['trail']),3)

    def test_tracks_separate_ego_motion_from_pulling_away(self):
        for car_speed in (0.,15.):
            tracks = TrafficTracks()
            for i in range(7):
                t = i*.125
                ego = dict(position=[0,0,10*t],yaw=0.,speed=36.)
                u,v = project(0,15+(car_speed-10)*t)
                detection = dict(box=[u-.03,v-.05,u+.03,v],confidence=.9,kind='car')
                car = tracks.update([detection],ego,10+t)[0]
            self.assertTrue(car['motion_reliable'])
            self.assertAlmostEqual(car['vz'],car_speed,places=5)
            self.assertAlmostEqual(car['relative_vz'],car_speed-10,places=5)
            json.dumps(car)  # Live perception records must retain plain JSON values.

    def test_car_occlusion_does_not_define_road_edge(self):
        from mono_drive.forza_scene import road_geometry_through_traffic,targets_left_half,ScenePlan
        road = straight_road()
        car = dict(box=[.45,.42,.55,.55],confidence=.9)
        road[94:119,172:211] = 0
        restored = road_geometry_through_traffic(road,[car])
        self.assertGreater(restored[110,190],140)
        left = ScenePlan(np.array([[6.,0.],[10.,0.],[16.,0.]]),0,50,.9,1,0,'test',.9)
        self.assertTrue(targets_left_half(left,restored))
        left.path[:,1] = 3.5
        self.assertFalse(targets_left_half(left,restored))

    def test_box_overlap_retains_passing_car_identity_during_depth_jitter(self):
        tracks=TrafficTracks()
        for i in range(7):
            t=i*.125
            ego=dict(position=[0,0,10*t],yaw=0.,speed=36.)
            u,v=project(1.,15+5*t)
            car=tracks.update([dict(box=[u-.03,v-.05,u+.03,v],confidence=.9,kind='car')],ego,10+t)[0]
        identity=car['id']
        t=.875
        u,v=project(1.,15+5*t+8.)
        car=tracks.update([dict(box=[u-.03,v-.05,u+.03,v],confidence=.9,kind='car')],
                          dict(position=[0,0,10*t],yaw=0.,speed=36.),10+t)[0]
        self.assertEqual(car['id'],identity)

    def test_brief_history_loss_keeps_last_measured_velocity(self):
        tracks=TrafficTracks()
        for i in range(7):
            t=i*.125
            u,v=project(0.,15+5*t)
            car=tracks.update([dict(box=[u-.03,v-.05,u+.03,v],confidence=.9,kind='car')],
                dict(position=[0,0,10*t],yaw=0.,speed=36.),10+t)[0]
        track=tracks.tracks[car['id']]
        track['history']=track['history'][-1:]
        t=.875
        u,v=project(0.,15+5*t)
        car=tracks.update([dict(box=[u-.03,v-.05,u+.03,v],confidence=.9,kind='car')],
            dict(position=[0,0,10*t],yaw=0.,speed=36.),10+t)[0]
        self.assertTrue(car['motion_reliable'])
        self.assertEqual(car['motion_source'],'recent_track')
        self.assertAlmostEqual(car['vz'],15.,places=4)

    def test_car_truck_label_flicker_does_not_reset_track(self):
        tracks=TrafficTracks()
        u,v=project(0.,15.)
        box=[u-.03,v-.05,u+.03,v]
        first=tracks.update([dict(box=box,confidence=.9,kind='car')],dict(position=[0,0,0],yaw=0.),1.)[0]
        second=tracks.update([dict(box=box,confidence=.9,kind='truck')],dict(position=[0,0,0],yaw=0.),1.1)[0]
        self.assertEqual(first['id'],second['id'])
        self.assertEqual(second['track_frames'],2)

    def test_snow_cannot_be_used_as_semantic_road(self):
        from mono_drive.forza_scene import visible_road_evidence
        road = np.full((216,384),255,np.uint8)
        asphalt = np.full((216,384,3),80,np.uint8)
        self.assertEqual(visible_road_evidence(road,asphalt)[115,200],255)
        snow = np.full((216,384,3),(210,180,160),np.uint8)
        self.assertEqual(visible_road_evidence(road,snow)[115,200],0)

    def test_side_window_is_not_projected_as_front(self):
        tracks = TrafficTracks()
        d = dict(box=[0,.4,.1,.6],confidence=.9,kind='car',view='side_left')
        car = tracks.update([d],dict(position=[0,0,0],yaw=0),1.)[0]
        self.assertNotIn('x',car)
        self.assertFalse(car['moving'])

    def test_short_bad_frame_coasts_but_sustained_loss_stops(self):
        from types import SimpleNamespace
        from mono_drive.forza import SpeedController
        c = SpeedController()
        c.target = 30
        d = SimpleNamespace(confidence=0.,mode='lost',reverse=False,target_kmh=0,steer=0)
        self.assertEqual(c.step(d,25,50,.5,.1)[1:],(0.,0.))
        for _ in range(10):
            command = c.step(d,25,50,.5,.1)
        self.assertGreater(command[2],0)
        d.confidence,d.mode = 1.,'traffic_stop'
        stop=c.step(d,25,50,.5,.1)[2]
        self.assertGreater(stop,0.)
        self.assertLessEqual(stop,.4)


if __name__ == '__main__':
    unittest.main()
