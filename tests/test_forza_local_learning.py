import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mono_drive.forza_local_learning import (
    parse_review, validate_review, training_settings, publish_review,
    candidate_verdict, record_candidate, summarize_rows,
    training_trigger,
)
from mono_drive.forza_local_monitor import session_phase, health, perception_restart_due
from mono_drive.forza_vlm import parse_assessment, corroborate


class LocalLearningTests(unittest.TestCase):
    def test_closed_capture_recovers_only_with_fresh_game_telemetry(self):
        live=dict(position=[0,0,0],sample_time=499.9)
        self.assertTrue(perception_restart_due(500.,live,100.,0.))
        self.assertFalse(perception_restart_due(500.,live,499.,0.))
        self.assertFalse(perception_restart_due(500.,live,100.,450.))
        self.assertFalse(perception_restart_due(500.,dict(live,sample_time=490.),100.,0.))

    def test_live_waiting_driver_can_restore_a_closed_capture(self):
        self.assertTrue(perception_restart_due(500.,dict(position=[0,0,0],sample_time=100.,time=499.9),100.,0.))
        self.assertFalse(perception_restart_due(500.,dict(position=[0,0,0],sample_time=100.,time=400.),100.,0.))

    def test_model_cannot_set_arbitrary_parameters_or_commands(self):
        proposal=parse_review('[1,1,9]')
        self.assertEqual(proposal['profile'],'curves')
        for raw in ('[7,0,9]','[1,0,true]','[1,0,9,123]','{"command":"run"}','[1,0,NaN]'):
            with self.assertRaises((ValueError,TypeError)):
                parse_review(raw)
        with TemporaryDirectory() as temp:
            path=Path(temp)/'learning-policy.json'
            path.write_text(json.dumps(dict(profile='curves',lr=100,epochs=10000,command='shell')))
            profile,settings=training_settings(temp)
            self.assertEqual(settings['lr'],1e-5)
            self.assertEqual(settings['epochs'],8)

    def test_proposal_requires_measured_evidence(self):
        proposal=parse_review('[1,1,9]')
        self.assertEqual(validate_review(proposal,dict(seconds=40,samples=110,curve_fraction=.01))[0],'balanced')
        self.assertEqual(validate_review(proposal,dict(seconds=40,samples=110,curve_fraction=.2))[0],'curves')
        self.assertEqual(validate_review(proposal,dict(seconds=1,samples=4,curve_fraction=.8))[0],'balanced')

    def test_review_is_logged_and_applied_only_outside_training(self):
        with TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'training-state.json').write_text('{"phase":"training"}')
            result=publish_review(folder,dict(seconds=50,samples=500,curve_fraction=.5),
                                  parse_review('[1,1,9]'),'[1,1,9]','local-model')
            self.assertTrue(result['deferred'])
            self.assertFalse((folder/'learning-policy.json').exists())
            (folder/'training-state.json').write_text('{"phase":"waiting_for_new_data"}')
            publish_review(folder,dict(seconds=50,samples=500,curve_fraction=.5),
                           parse_review('[1,1,9]'),'[1,1,9]','local-model')
            self.assertEqual(training_settings(folder)[0],'curves')

    def test_candidate_acceptance_and_rollback(self):
        good=dict(held_out_frames=40,models={'before':dict(mean_m=2.,median_m=2.),'after':dict(mean_m=1.8,median_m=1.9)})
        self.assertTrue(candidate_verdict(good,'before','after')[0])
        good['models']['after']['mean_m']=float('nan')
        self.assertFalse(candidate_verdict(good,'before','after')[0])
        good['models']['after']['mean_m']=2.2
        with TemporaryDirectory() as temp:
            result=record_candidate(temp,good,'before','after','curves')
            self.assertFalse(result['accepted'])
            self.assertEqual(training_settings(temp)[0],'balanced')
            self.assertFalse(result['automatic_driving_activation'])

    def test_clean_improvement_cannot_hide_camera_degradation_failure(self):
        value=dict(held_out_frames=50,evaluation_version=2,
                   models={'old':dict(mean_m=2.,median_m=2.),'new':dict(mean_m=1.8,median_m=1.9)},
                   conditions={name:{'old':dict(mean_m=2.),'new':dict(mean_m=2.)} for name in ('blur','low_light')})
        self.assertTrue(candidate_verdict(value,'old','new')[0])
        value['conditions']['blur']['new']['mean_m']=2.3
        self.assertEqual(candidate_verdict(value,'old','new'),(False,'camera_degradation_regression'))

    def test_large_recording_cannot_conceal_new_recording_regression(self):
        good={'old':dict(mean_m=2.,median_m=2.),'new':dict(mean_m=1.8,median_m=1.9)}
        value=dict(held_out_frames=500,models=good,human_holdout=dict(frames=400,models=good),
            legacy_holdout=dict(frames=100,models=good),human_recordings={'new_session':dict(frames=40,
                models={'old':dict(mean_m=1.),'new':dict(mean_m=1.3)})})
        self.assertEqual(candidate_verdict(value,'old','new'),(False,'individual_human_recording_regression'))

    def test_measured_window_excludes_large_gaps(self):
        rows=[dict(t=t,speed_kmh=50.,sent_command=[0,0.1,0],candidate_path=[[0,0],[10,1],[20,5]]) for t in (0,.5,1,20,20.5)]
        result=summarize_rows(rows)
        self.assertEqual(result['seconds'],1.5)
        self.assertEqual(result['mean_kmh'],50.)
        self.assertEqual(result['curve_fraction'],1.)

    def test_method_change_gets_one_bounded_trial_without_new_data(self):
        self.assertEqual(training_trigger(800,800,'conservative','balanced',0,1000),'method_changed')
        self.assertIsNone(training_trigger(800,800,'conservative','conservative',1000,1100))
        self.assertIsNone(training_trigger(800,800,'balanced','conservative',1000,1100))
        self.assertEqual(training_trigger(1000,800,'balanced','balanced',1000,1100),'new_data')

    def test_watchdog_honors_explicit_stop_and_original_deadline(self):
        self.assertEqual(session_phase({'ends_at':100},False,99),'watching')
        self.assertEqual(session_phase({'ends_at':100},True,99),'stopped_by_user')
        self.assertEqual(session_phase({'ends_at':100},False,101),'completed')

    def test_continuous_mode_has_no_deadline_but_still_honors_stop(self):
        from mono_drive.forza_session_policy import remaining_seconds,session_active
        config=dict(started_at=1.,continuous=True,ends_at=None)
        self.assertIsNone(remaining_seconds(config,999999.))
        self.assertTrue(session_active(config,999999.))
        self.assertEqual(session_phase(config,False,999999.),'watching')
        self.assertEqual(session_phase(config,True,999999.),'stopped_by_user')
        self.assertFalse(session_active({},1.))
        status=health(100,{'time':99.9},{'time':5},99.8,{'time':99})
        self.assertIn('video_model_stale',status['issues'])
        self.assertNotIn('driver_stale',status['issues'])

    def test_compact_video_parser_and_independent_corroboration(self):
        assessment=parse_assessment('[0,4,1,1,0,9]')
        self.assertEqual(assessment['lane_position'],'offroad')
        self.assertFalse(corroborate(assessment,[{'mode':'paint_pair','lane_margin_m':1.}]))
        self.assertTrue(corroborate(assessment,[{'mode':'road_reentry'}]))
        assessment=parse_assessment('[1,5,3,2,0,9]')
        self.assertFalse(corroborate(assessment,[{'state':'driving','speed_kmh':0.}]))
        for raw in ('[0,8,1,1,0,9]','[0,4,1,1,true,9]','[0,4]'):
            with self.assertRaises(ValueError):
                parse_assessment(raw)


if __name__=='__main__':
    unittest.main()
