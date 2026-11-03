import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from dataclasses import replace
import numpy as np
from mono_drive.forza_route_memory import SuccessfulRouteMemory
from mono_drive.forza_failure_memory import FailedApproachGuard
from mono_drive.forza_scene import ScenePlan


class RouteMemoryTests(unittest.TestCase):
    def test_repeated_pose_reuses_geometry_but_returns_independent_list(self):
        with TemporaryDirectory() as folder:
            routes=self.write(folder)
            ego=dict(position=[0,0,20],yaw=0.,speed=40.)
            first=routes.candidates(ego)
            first.clear()
            self.assertEqual(len(routes.candidates(ego)),1)
            self.assertEqual(routes.candidates(dict(ego,position=[100,0,20])),[])
            self.assertEqual(len(routes.candidates(ego)),1)

    def test_refresh_interval_avoids_reparsing_but_observes_later_updates(self):
        from unittest.mock import patch
        with TemporaryDirectory() as folder:
            source=self.write(folder)
            routes=SuccessfulRouteMemory(source.file,refresh_s=2.)
            with patch('mono_drive.forza_route_memory.time.monotonic',return_value=10.):
                routes.load()
            data=json.loads(source.file.read_text())
            data['segments'].append(dict(data['segments'][0],id=2))
            source.file.write_text(json.dumps(data))
            with patch('mono_drive.forza_route_memory.time.monotonic',return_value=11.):
                routes.load()
                self.assertEqual(len(routes.segments),1)
            with patch('mono_drive.forza_route_memory.time.monotonic',return_value=12.1):
                routes.load()
                self.assertEqual(len(routes.segments),2)

    def test_committed_route_cannot_pull_car_into_adjacent_lane(self):
        with TemporaryDirectory() as folder:
            routes=self.write(folder)
            ego=dict(position=[0,0,20],yaw=0.,speed=40.)
            scene=dict(road=np.full((216,384),255,np.uint8),cars=[])
            self.assertEqual(routes.apply(self.plan(),scene,ego).reason,'route_memory')
            shifted=dict(ego,position=[2.5,0,20])
            result=routes.apply(self.plan(),scene,shifted)
            self.assertEqual(result.reason,'lane_inferred')
            self.assertIsNone(routes.last_path)

    def write(self,folder,x=0.):
        file=Path(folder)/'routes.json'
        points=[[x,0.,float(z)] for z in range(0,81,2)]
        file.write_text(json.dumps(dict(segments=[dict(id=1,points=points,yaws=[0.]*len(points))])))
        return SuccessfulRouteMemory(file)

    def plan(self):
        z=np.array([4.,8.,12.,20.,30.])
        return ScenePlan(np.column_stack((z,np.full_like(z,.6))),.1,25.,.6,1,0,'lane_inferred',.9,
                         lane_edges=tuple((d,-1.75,1.75) for d in z),inferred=True)

    def test_validated_successful_route_guides_current_driving(self):
        with TemporaryDirectory() as folder:
            routes=self.write(folder)
            ego=dict(position=[0,0,20],yaw=0.,speed=30.)
            scene=dict(road=np.full((216,384),255,np.uint8),cars=[])
            result=routes.apply(self.plan(),scene,ego)
            self.assertEqual(result.reason,'route_memory')
            np.testing.assert_allclose(result.path[:,1],0.)
            self.assertAlmostEqual(result.steer,0.)
            scene['road'][:]=0
            self.assertEqual(routes.apply(self.plan(),scene,ego).reason,'lane_inferred')
            self.assertEqual(routes.candidates(dict(ego,position=[0,10,20])),[])
            self.assertEqual(routes.candidates(dict(ego,yaw=3.14)),[])

    def test_route_cannot_override_a_sidewalk_stop(self):
        with TemporaryDirectory() as folder:
            routes=self.write(folder)
            plan=replace(self.plan(),reason='offroad_surface',limit=0.)
            result=routes.apply(plan,dict(road=np.full((216,384),255,np.uint8),cars=[]),
                                dict(position=[0,0,20],yaw=0.,speed=30.))
            self.assertEqual(result.limit,0.)

    def test_clear_current_painted_lane_precedes_old_route_consensus(self):
        with TemporaryDirectory() as folder:
            routes=self.write(folder)
            data=json.loads(routes.file.read_text());first=data['segments'][0];first['start']=100.
            data['segments'].append(dict(first,id=2,start=200.));routes.file.write_text(json.dumps(data))
            plan=replace(self.plan(),reason='paint_pair',quality=.8,inferred=False,painted_boundaries=6)
            scene=dict(road=np.full((216,384),255,np.uint8),cars=[])
            result=routes.apply(plan,scene,dict(position=[0,0,20],yaw=0.,speed=30.))
            self.assertIs(result,plan)
            self.assertEqual(routes.status['action'],'current_painted_lane')

    def test_turn_keeps_visible_forward_arc_when_later_route_faces_behind(self):
        with TemporaryDirectory() as folder:
            routes=self.write(folder)
            angle=np.linspace(0.,2.5,70)
            points=np.column_stack((16.*(np.cos(angle)-1.),np.zeros_like(angle),16.*np.sin(angle)))
            routes.file.write_text(json.dumps(dict(segments=[dict(id=1,points=points.tolist(),yaws=(-angle).tolist())])))
            candidates=routes.candidates(dict(position=[0,0,0],yaw=0.,speed=20.))
            self.assertEqual(len(candidates),1)
            self.assertTrue(np.all(np.diff(candidates[0][2][:,0])>0.))
            self.assertGreater(candidates[0][2][-1,0],15.)

    def test_distant_unseen_pixels_do_not_discard_a_clear_measured_route(self):
        from mono_drive.forza_projection import project
        with TemporaryDirectory() as folder:
            routes=self.write(folder)
            ego=dict(position=[0,0,20],yaw=0.,speed=30.)
            road=np.full((216,384),255,np.uint8)
            _,v=project(0.,24.)
            road[:int(v*215)]=0
            result=routes.apply(self.plan(),dict(road=road,cars=[]),ego)
            self.assertEqual(result.reason,'route_memory')

    def test_independent_successful_traversals_can_correct_a_false_paint_lane(self):
        with TemporaryDirectory() as folder:
            routes=self.write(folder)
            data=json.loads(routes.file.read_text());first=data['segments'][0];first['start']=100.
            data['segments'].append(dict(first,id=2,start=200.))
            routes.file.write_text(json.dumps(data))
            plan=replace(self.plan(),quality=.95,inferred=False,
                         lane_edges=tuple((z,.5,4.) for z in self.plan().path[:,0]))
            ego=dict(position=[0,0,20],yaw=0.,speed=30.)
            result=routes.apply(plan,dict(road=np.full((216,384),255,np.uint8),cars=[]),ego)
            self.assertEqual(result.reason,'route_memory')
            self.assertTrue(routes.status['consensus'])
            self.assertTrue(result.verified_route)
            np.testing.assert_allclose(result.path[:,1],0.)

    def test_an_adopted_route_does_not_switch_to_crosswalk_paint(self):
        with TemporaryDirectory() as folder:
            routes=self.write(folder)
            ego=dict(position=[0,0,20],yaw=0.,speed=30.)
            scene=dict(road=np.full((216,384),255,np.uint8),cars=[])
            self.assertEqual(routes.apply(self.plan(),scene,ego).reason,'route_memory')
            wrong=replace(self.plan(),quality=.95,inferred=False,
                          lane_edges=tuple((z,.5,4.) for z in self.plan().path[:,0]))
            result=routes.apply(wrong,scene,ego)
            self.assertEqual(result.reason,'route_memory')
            self.assertTrue(routes.status['committed'])
            scene['road'][:]=0
            self.assertEqual(routes.apply(wrong,scene,ego).reason,wrong.reason)

    def test_known_failure_can_choose_successful_route_inside_same_lane(self):
        with TemporaryDirectory() as folder:
            routes=self.write(folder,.5)
            file=Path(folder)/'failures.json'
            file.write_text(json.dumps(dict(entries=[dict(id=1,point=[-.95,35.],yaw=0.,height=0.,confidence=.9)])))
            guard=FailedApproachGuard(file);guard.route_memory=routes
            ego=dict(position=[0,0,20],yaw=0.,speed=30.)
            plan=self.plan();plan=replace(plan,path=np.column_stack((plan.path[:,0],plan.path[:,0]*0)))
            result=guard.apply(plan,dict(road=np.full((216,384),255,np.uint8),cars=[]),ego)
            self.assertEqual(result.reason,'remembered_avoidance')
            np.testing.assert_allclose(result.path[:,1],.5)
            result=guard.apply(replace(plan,lane_edges=()),dict(road=np.full((216,384),255,np.uint8),cars=[]),ego)
            self.assertEqual(result.reason,'remembered_avoidance')
            np.testing.assert_allclose(result.path[:,1],.5)


if __name__=='__main__':unittest.main()
