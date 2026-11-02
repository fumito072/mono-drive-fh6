import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from mono_drive.forza_incidents import StopWatch,diagnose,capture_incident
from mono_drive.forza_failure_memory import FailedApproachGuard
from mono_drive.forza_scene import ScenePlan
from mono_drive.forza_video_labels import RejectedVideoIntervals
from mono_drive.forza_vlm import parse_incident_review


class IncidentTests(unittest.TestCase):
    def rows(self,**changes):
        result=[]
        for stamp in range(90,111):
            moving=stamp<100
            row=dict(t=float(stamp),position=[0.,0.,stamp-90.],yaw=0.,speed_kmh=20. if moving else 0.,
                state='driving',mode='road_candidates',actual_accel=160,actual_brake=0,actual_handbrake=0,
                confidence=.9,lane_clearance_m=.6,inferred_lane=False,static_clearance_m=4.,
                target_kmh=30.,candidate_path=[[4.,0.],[10.,0.],[20.,0.]],surface={})
            if not moving: row.update(changes)
            result.append(row)
        return result

    def test_stop_report_emits_once_until_movement(self):
        watch=StopWatch()
        for t in (0.,2.,4.,6.):
            self.assertFalse(watch.update(t,dict(time=t,position=[0,0,0],speed_kmh=0.)))
        self.assertTrue(watch.update(8.,dict(time=8.,position=[0,0,0],speed_kmh=0.)))
        self.assertFalse(watch.update(12.,dict(time=12.,position=[0,0,0],speed_kmh=0.)))
        self.assertFalse(watch.update(14.,dict(time=14.,position=[0,0,10],speed_kmh=20.)))
        self.assertFalse(watch.update(16.,dict(time=16.,position=[0,0,10],speed_kmh=0.)))
        self.assertTrue(watch.update(24.,dict(time=24.,position=[0,0,10],speed_kmh=0.)))

    def test_diagnosis_distinguishes_input_traffic_and_contact(self):
        self.assertEqual(diagnose(self.rows())['cause'],'static_contact')
        stale=self.rows(state='image/decision stale',actual_accel=0)
        self.assertEqual(diagnose(stale)['action'],'restart_driver')
        traffic=self.rows(mode='traffic_stop',actual_accel=0,collision_brake_demand=.8)
        self.assertEqual(diagnose(traffic)['action'],'observe')
        offroad=self.rows(surface=dict(offroad_confirmed=True),actual_accel=0)
        self.assertEqual(diagnose(offroad)['cause'],'confirmed_offroad')

    def test_missing_old_log_fields_do_not_break_diagnosis(self):
        result=diagnose(self.rows(static_clearance_m=None,collision_brake_demand=None,
                                 actual_brake=None,actual_accel=None,target_kmh=None))
        self.assertEqual(result['cause'],'undetermined_stop')

    def test_incident_persists_evidence_and_excludes_failed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory); run=out/'fh6_test';run.mkdir()
            rows=self.rows()
            (run/'decisions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            report=capture_incident(out,dict(run=str(run)),110.)
            self.assertIsNotNone(report['memory_id'])
            self.assertTrue((out/'incidents'/report['id']/'approach.jsonl').is_file())
            self.assertTrue(RejectedVideoIntervals(out/'vlm-labels.jsonl').rejects(105.))
            self.assertEqual(capture_incident(out,dict(run=str(run)),110.)['id'],report['id'])

    def test_input_failure_never_becomes_a_spatial_obstacle(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);run=out/'fh6_test';run.mkdir()
            (run/'decisions.jsonl').write_text('\n'.join(json.dumps(r) for r in self.rows(
                state='image/decision stale',actual_accel=0)))
            report=capture_incident(out,dict(run=str(run)),110.)
            self.assertIsNone(report['memory_id'])
            self.assertFalse((out/'failure-memory.json').exists())

    def test_moving_course_transition_never_becomes_a_spatial_obstacle(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);run=out/'fh6_test';run.mkdir()
            rows=self.rows(speed_kmh=25.)
            (run/'decisions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            report=capture_incident(out,dict(run=str(run)),110.)
            self.assertIsNone(report['memory_id'])
            self.assertFalse((out/'failure-memory.json').exists())

    def test_excluded_course_transition_is_not_used_as_a_wall(self):
        with tempfile.TemporaryDirectory() as directory:
            guard=self.guard(directory,0.)
            data=json.loads(guard.path.read_text())
            data['entries'][0]['excluded_reason']='planned_route_change'
            guard.path.write_text(json.dumps(data))
            scene=dict(road=np.full((216,384),255,np.uint8),cars=[])
            ego=dict(position=[0,0,0],yaw=0.,speed=20.)
            output=guard.apply(self.plan(),scene,ego)
            self.assertEqual(output.limit,50.)
            self.assertEqual(guard.entries,[])

    def guard(self,folder,x):
        file=Path(folder)/'failure-memory.json'
        file.write_text(json.dumps(dict(entries=[dict(id='example',point=[x,15.],yaw=0.,height=0.,confidence=.9)])))
        return FailedApproachGuard(file)

    def plan(self):
        z=np.array([4.,8.,12.,18.,25.])
        return ScenePlan(np.column_stack((z,np.zeros_like(z))),0.,50.,.9,1,0,'paint_pair',.9,
                         lane_edges=tuple((v,-1.75,1.75) for v in z))

    def test_memory_chooses_only_an_in_lane_supported_alternative(self):
        with tempfile.TemporaryDirectory() as folder:
            guard=self.guard(folder,-.95)
            scene=dict(road=np.full((216,384),255,np.uint8),cars=[])
            ego=dict(position=[0,0,0],yaw=0.,speed=20.)
            output=guard.apply(self.plan(),scene,ego)
            self.assertEqual(output.reason,'remembered_avoidance')
            self.assertGreater(output.steer,0.)
            self.assertLessEqual(max(output.path[:,1]),.75)
            self.assertNotEqual(guard.apply(output,scene,ego,final=True).reason,'known_failed_approach')

    def test_no_valid_alternative_blocks_repeat_but_not_other_level(self):
        with tempfile.TemporaryDirectory() as folder:
            guard=self.guard(folder,0.)
            scene=dict(road=np.full((216,384),255,np.uint8),cars=[])
            ego=dict(position=[0,0,0],yaw=0.,speed=20.)
            output=guard.apply(self.plan(),scene,ego)
            self.assertEqual(output.reason,'known_failed_approach')
            self.assertEqual(output.limit,0.)
            ego['position']=[0,10,0]
            self.assertEqual(guard.apply(self.plan(),scene,ego).limit,50.)

    def test_vlm_cannot_turn_a_review_into_a_command(self):
        result=parse_incident_review('[1,9]')
        self.assertEqual(result['cause'],'fixed_obstacle')
        self.assertFalse(result['auto_verified'])
        for value in ('["press A",9]','[1,9,0]','[99,9]'):
            with self.assertRaises((ValueError,TypeError)):
                parse_incident_review(value)


if __name__=='__main__':
    unittest.main()
