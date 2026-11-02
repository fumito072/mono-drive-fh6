import unittest
from dataclasses import replace
import numpy as np
from mono_drive.forza_scene import ScenePlan
from mono_drive.forza_surface import RoadSurfaceGuard
from mono_drive.forza_session import road_recovery_reason
from tests.test_forza_scene import straight_road


class SurfaceTests(unittest.TestCase):
    def test_distant_road_taper_does_not_block_a_near_low_speed_merge(self):
        from unittest.mock import patch
        z=np.array([6.,8.,10.,12.,16.])
        p=ScenePlan(np.column_stack((z,np.full_like(z,-2.0))),-.2,50.,.9,1,0,'road_candidates',.9,
                    lane_edges=tuple((v,-3.75,-.25) for v in z))
        road=straight_road(-4.,4.)
        rows=np.array([[6,-4,4],[8,-4,4],[10,-4,4],[12,-4,4],[16,-4,4],[40,1,2],[60,2,3]])
        ego=dict(position=[0,0,0],yaw=0.,speed=0.,steer=0.)
        with patch('mono_drive.forza_surface.bounds',return_value=rows):
            result=RoadSurfaceGuard().apply(p,dict(road=road,cars=[],ego=ego),ego,1.)
        self.assertEqual(result.reason,'lane_alignment')
        self.assertGreater(result.limit,0.)
        self.assertLessEqual(result.path[-1,0],16.)
    def test_new_dashboard_occlusion_does_not_block_adjacent_lane_alignment(self):
        z=np.array([6.,8.,10.,12.,16.,22.,30.])
        p=ScenePlan(np.column_stack((z,np.full_like(z,-2.3))),-.2,50.,.9,1,0,'route_memory',.9,
                    lane_edges=tuple((v,-4.05,-.55) for v in z),inferred=True)
        road=straight_road(-4.2,4.2)
        visibility=np.ones_like(road,dtype=bool);visibility[118:]=False
        road[~visibility]=0
        ego=dict(position=[0,0,0],yaw=0.,speed=0.,steer=0.)
        result=RoadSurfaceGuard().apply(p,dict(road=road,cars=[],ego=ego,planning_roi=visibility),ego,1.)
        self.assertEqual(result.reason,'lane_alignment')
        self.assertGreater(result.limit,0.)

    def test_verified_adjacent_lane_spawn_can_merge_without_geometry_deadlock(self):
        z=np.array([6.,8.,10.,12.,16.,22.,30.])
        p=ScenePlan(np.column_stack((z,np.full_like(z,-2.3))),-.2,50.,.9,1,0,'road_candidates',.9,
                    lane_edges=tuple((v,-4.05,-.55) for v in z))
        road=straight_road(-4.2,4.2)
        ego=dict(position=[0,0,0],yaw=0.,speed=0.,steer=0.)
        scene=dict(road=road,sidewalk=np.zeros_like(road),cars=[],ego=ego)
        result=RoadSurfaceGuard().apply(p,scene,ego,1.)
        self.assertEqual(result.reason,'lane_alignment')
        self.assertGreater(result.limit,0.)
        self.assertLessEqual(result.limit,12.)
        self.assertLess(result.steer,0.)

    def test_lost_lane_still_respects_the_visible_guardrail_boundary(self):
        from mono_drive.forza_lane_guard import LaneDepartureGuard
        road=straight_road(-3.,6.)
        guard=RoadSurfaceGuard()
        bounded=guard.apply(self.plan,dict(road=road,cars=[],ego=self.ego),self.ego,1.)
        self.assertEqual(bounded.reason,'lane_alignment')
        self.assertGreater(bounded.path[-1,1],self.plan.path[-1,1])
        self.assertTrue(bounded.lane_edges)
        output=LaneDepartureGuard().apply(bounded,30.)
        self.assertGreater(output.steer,self.plan.steer)

    def test_missing_paint_in_adjacent_lane_uses_a_gradual_left_merge(self):
        from mono_drive.forza_scene import inside_lane
        road=straight_road(-4.,4.)
        guard=RoadSurfaceGuard()
        result=guard.apply(self.plan,dict(road=road,cars=[],ego=self.ego),self.ego,1.)
        self.assertEqual(result.reason,'lane_alignment')
        self.assertLess(result.path[-1,1],-1.)
        self.assertGreater(result.path[0,1],-.5)
        edges=np.asarray(result.lane_edges)
        self.assertTrue(inside_lane(result.path,edges[:,1],edges[:,2]))
        self.assertGreater(result.limit,12.)
        self.assertLessEqual(result.limit,40.)
        self.assertTrue(guard.status['lane_alignment'])

    def test_paint_boundaries_cannot_extend_through_a_road_edge(self):
        from mono_drive.forza_surface import constrain_lane_edges
        z=np.array([6.,8.,12.,20.])
        p=ScenePlan(np.column_stack((z,z*0-.5)),0.,40.,.9,1,0,'paint_pair',.9,
                    lane_edges=tuple((v,-2.3,1.3) for v in z))
        result=constrain_lane_edges(p,straight_road(-1.25,5.75))
        self.assertGreater(min(e[1] for e in result.lane_edges),-1.35)
        self.assertTrue(all(e[2]<=1.3 for e in result.lane_edges))

    def setUp(self):
        self.ego=dict(position=[0.,0.,0.],yaw=0.,speed=25.,steer=0.)
        z=np.array([4.,6.,8.,10.,16.,22.])
        self.plan=ScenePlan(np.column_stack((z,-.025*z*z)),-.2,40.,.6,1,0,
                            'lane_inferred',.3,inferred=True)

    def test_inferred_curve_is_bounded_by_road_even_without_paint(self):
        road=straight_road(-1.8,5.2)
        scene=dict(road=road,raw_road=road,cars=[],ego=self.ego)
        guard=RoadSurfaceGuard()
        result=guard.apply(self.plan,scene,self.ego,1.)
        self.assertEqual(result.reason,'road_envelope')
        self.assertTrue(guard.status['corrected'])
        self.assertTrue(result.lane_edges)
        self.assertLess(abs(result.steer),.05)
        self.assertGreater(result.limit,25.)

    def test_known_sidewalk_is_not_a_reentry_driving_path(self):
        road=straight_road(4.,11.)
        sidewalk=straight_road(-2.,2.)
        scene=dict(road=road,raw_road=road,sidewalk=sidewalk,cars=[],ego=self.ego)
        guard=RoadSurfaceGuard()
        first=guard.apply(self.plan,scene,self.ego,1.)
        self.assertEqual(first.limit,0.)
        self.assertEqual(first.reason,'offroad_surface')
        self.assertFalse(guard.status['offroad_confirmed'])
        guard.apply(self.plan,scene,self.ego,1.5)
        self.assertTrue(guard.status['offroad_confirmed'])

    def test_road_envelope_turns_with_a_bend_and_recovers_near_kerb(self):
        from mono_drive.forza_scene import project
        road=np.zeros((216,384),np.uint8)
        for y in range(95,130):
            z=.575/(y/215-.435)
            x=.012*z*z
            a,_=project(x-1.8,z); b,_=project(x+5.2,z)
            road[y,max(0,int(a*383)):min(384,int(b*383)+1)]=250
        result=RoadSurfaceGuard().apply(self.plan,dict(road=road,cars=[],ego=self.ego),self.ego,1.)
        self.assertEqual(result.reason,'road_envelope')
        self.assertGreater(result.path[-1,1],2.)
        near_kerb=straight_road(-.5,6.5)
        result=RoadSurfaceGuard().apply(self.plan,dict(road=near_kerb,cars=[],ego=self.ego),self.ego,1.)
        self.assertEqual(result.reason,'road_envelope')
        self.assertGreater(result.steer,.05)

    def test_final_validation_does_not_rewrite_steering_after_lane_guard(self):
        road=straight_road(-1.8,5.2)
        plan=replace(self.plan,path=np.array([[4.,0.],[8.,0.],[16.,0.]]),steer=.04)
        output=RoadSurfaceGuard().apply(plan,dict(road=road,cars=[],ego=self.ego),self.ego,1.,correct=False)
        self.assertEqual(output.steer,.04)

    def test_unknown_surface_does_not_reintroduce_a_blind_stop(self):
        road=np.zeros((216,384),np.uint8)
        scene=dict(road=road,sidewalk=road,cars=[],ego=self.ego)
        result=RoadSurfaceGuard().apply(self.plan,scene,self.ego,1.)
        self.assertEqual(result.limit,self.plan.limit)
        self.assertEqual(result.reason,'lane_inferred')

    def test_measured_path_cannot_override_a_confirmed_sidewalk(self):
        road=straight_road(-1.8,5.2)
        sidewalk=straight_road(-8.,-1.9)
        guard=RoadSurfaceGuard()
        result=guard.apply(replace(self.plan,inferred=False),
            dict(road=road,raw_road=road,sidewalk=sidewalk,cars=[],ego=self.ego),self.ego,1.)
        self.assertTrue(guard.status['path_enters_sidewalk'])
        self.assertTrue(guard.status['corrected'])
        self.assertLess(abs(result.steer),.05)

    def test_recovery_requires_a_recent_confirmed_observation(self):
        self.assertIsNone(road_recovery_reason(10.,dict(time=6.,offroad_confirmed=True),{}))
        self.assertIsNone(road_recovery_reason(10.,dict(time=9.,offroad_confirmed=False),{}))
        self.assertEqual(road_recovery_reason(10.,dict(time=9.,offroad_confirmed=True),{}),'confirmed_sidewalk')
        self.assertEqual(road_recovery_reason(10.,{},dict(time=9.,requested=True)),'requested_road_recovery')


if __name__=='__main__':
    unittest.main()
