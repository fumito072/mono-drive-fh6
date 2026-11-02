import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from mono_drive.forza_model_policy import offer,service
from mono_drive.forza_local_learning import candidate_verdict


class ModelPolicyTests(unittest.TestCase):
    def test_trial_waits_until_baseline_leaves_existing_stop(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'models/candidates').mkdir(parents=True);(root/'configs').mkdir();(root/'out').mkdir()
            candidate=root/'models/candidates/new.onnx';candidate.write_bytes(b'test')
            config=root/'configs/forza_horizon6.json';config.write_text('{"models":{"pathnet":"old"}}')
            evaluation=dict(held_out_frames=40,models={'old':dict(mean_m=1,median_m=1),str(candidate):dict(mean_m=.8,median_m=.8)})
            self.assertTrue(offer(root,candidate,'old',evaluation))
            (root/'out/control.json').write_text('{"action":"drive"}')
            heart=dict(time=100.,armed=True,model_hot_reload=True,speed_kmh=0.,mode='known_failed_approach')
            service(root,heart,dict(recoveries=0),100.)
            self.assertEqual(json.loads(config.read_text())['models']['pathnet'],'old')
            service(root,dict(heart,time=101.,speed_kmh=20.,mode='paint_pair'),dict(recoveries=0),101.)
            self.assertEqual(json.loads(config.read_text())['models']['pathnet'],str(candidate))

    def test_input_pipeline_restarts_are_not_scored_as_model_collisions(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'out').mkdir();(root/'configs').mkdir()
            candidate=str(root/'new.onnx')
            (root/'configs/forza_horizon6.json').write_text(json.dumps(dict(models=dict(pathnet=candidate))))
            (root/'out/control.json').write_text('{"action":"drive"}')
            state=dict(state='trial',candidate=candidate,previous_model='old',started_at=100.,
                loaded_at=100.,distance_m=0.,offroad_observations=0,initial_recoveries=1)
            (root/'out/model-activation.json').write_text(json.dumps(state))
            heart=dict(time=113.,armed=True,model_hot_reload=True,path_model=candidate)
            service(root,heart,dict(recoveries=3,last_incident=dict(id='112000',cause='input_stale')),113.)
            result=json.loads((root/'out/model-activation.json').read_text())
            self.assertEqual(result['state'],'trial')
            self.assertTrue(result['infrastructure_restart_excluded'])
            service(root,dict(heart,time=115.),dict(recoveries=3,last_incident=dict(id='114000',cause='static_contact')),115.)
            self.assertEqual(json.loads((root/'out/model-activation.json').read_text())['state'],'rolled_back')

    def test_hot_model_waits_for_driver_ack_without_stopping_inputs(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'models/candidates').mkdir(parents=True);(root/'configs').mkdir();(root/'out/run').mkdir(parents=True)
            candidate=root/'models/candidates/new.onnx';candidate.write_bytes(b'test-model')
            config=root/'configs/forza_horizon6.json';config.write_text(json.dumps(dict(models=dict(pathnet='models/pathnet.onnx'),forza=dict(max_kmh=60))))
            evaluation=dict(held_out_frames=40,models={'old':dict(mean_m=1,median_m=1),str(candidate):dict(mean_m=.8,median_m=.8)})
            self.assertTrue(offer(root,candidate,'old',evaluation))
            (root/'out/control.json').write_text('{"action":"drive","max_kmh":60}')
            heart=dict(time=100.,armed=True,model_hot_reload=True,path_model='models/pathnet.onnx',run=str(root/'out/run'),position=[0,0,0])
            service(root,heart,dict(recoveries=0),100.)
            self.assertFalse((root/'out/run/stop').exists())
            service(root,dict(heart,time=110.,position=[40,0,0]),dict(recoveries=0),110.)
            state=json.loads((root/'out/model-activation.json').read_text())
            self.assertEqual(state['distance_m'],0)
            self.assertEqual(state['load_state'],'waiting_for_driver')
            heart.update(time=111.,path_model=str(candidate),position=[40,0,0])
            service(root,heart,dict(recoveries=0),111.)
            self.assertEqual(json.loads((root/'out/model-activation.json').read_text())['loaded_at'],111.)
            for now in (112.,114.,116.):
                service(root,dict(heart,time=now,surface=dict(offroad_confirmed=True)),dict(recoveries=0),now)
            self.assertEqual(json.loads(config.read_text())['models']['pathnet'],'models/pathnet.onnx')
            self.assertEqual(json.loads(config.read_text())['forza']['max_kmh'],60)
            self.assertFalse((root/'out/run/stop').exists())

    def test_promotion_waits_for_driving_and_bad_live_trial_rolls_back(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'models/candidates').mkdir(parents=True);(root/'configs').mkdir();(root/'out/run').mkdir(parents=True)
            candidate=root/'models/candidates/new.onnx';candidate.write_bytes(b'test-model')
            config=root/'configs/forza_horizon6.json';config.write_text(json.dumps(dict(models=dict(pathnet='models/pathnet.onnx'))))
            evaluation=dict(held_out_frames=40,models={'old':dict(mean_m=1,median_m=1),str(candidate):dict(mean_m=.8,median_m=.8)})
            self.assertTrue(offer(root,candidate,'old',evaluation))
            heart=dict(time=100.,armed=True,run=str(root/'out/run'),position=[0,0,0])
            (root/'out/control.json').write_text('{"action":"hold"}')
            service(root,heart,dict(recoveries=0),100.)
            self.assertEqual(json.loads(config.read_text())['models']['pathnet'],'models/pathnet.onnx')
            (root/'out/control.json').write_text('{"action":"drive"}')
            service(root,heart,dict(recoveries=0),100.)
            self.assertEqual(json.loads(config.read_text())['models']['pathnet'],str(candidate))
            self.assertTrue((root/'out/run/stop').exists())
            for now in (101.,102.,103.):service(root,dict(heart,time=now,surface=dict(offroad_confirmed=True)),dict(recoveries=0),now)
            self.assertEqual(json.loads(config.read_text())['models']['pathnet'],'models/pathnet.onnx')
            self.assertEqual(json.loads((root/'out/model-activation.json').read_text())['state'],'rolled_back')

    def test_average_improvement_cannot_hide_regression_on_human_holdout(self):
        evaluation=dict(held_out_frames=100,models={'old':dict(mean_m=1.,median_m=1.),'new':dict(mean_m=.8,median_m=.8)},
            human_holdout=dict(frames=50,models={'old':dict(mean_m=1.),'new':dict(mean_m=1.1)}),
            legacy_holdout=dict(models={'old':dict(mean_m=1.),'new':dict(mean_m=.8)}))
        self.assertEqual(candidate_verdict(evaluation,'old','new'),(False,'human_holdout_did_not_improve'))
