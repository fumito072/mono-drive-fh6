from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mono_drive.forza_evaluation import group_recordings
from mono_drive.forza_local_learning import candidate_verdict
from mono_drive.forza_model_policy import offer, service
from mono_drive.forza_candidate_review import review_completed


def evaluation(candidate='new'):
    scores={'old':dict(mean_m=1.,median_m=1.),candidate:dict(mean_m=.8,median_m=.8)}
    records={str(i):dict(frames=100,car_ordinals=[3819],models=deepcopy(scores)) for i in range(3)}
    records['other_car']=dict(frames=100,car_ordinals=[4222],models={
        'old':dict(mean_m=1.),candidate:dict(mean_m=1.2)})
    return dict(held_out_frames=500,models=scores,recording_grouping_version=1,
        human_holdout=dict(frames=400,models=deepcopy(scores)),
        legacy_holdout=dict(frames=100,models=deepcopy(scores)),human_recordings=records)


class EvaluationTests(unittest.TestCase):
    def test_partition_means_are_weighted_and_medians_not_invented(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);parts={}
            for name,count,new in [('a',30,1.2),('b',270,.7)]:
                part=root/'datasets/human/session'/name;part.mkdir(parents=True)
                (part/'provenance.json').write_text(json.dumps(dict(recording='recording',car_ordinals=[3819])))
                parts[str(part)]=dict(frames=count,models={'old':dict(mean_m=1.,median_m=1.),
                    'new':dict(mean_m=new,median_m=.5)})
            source=dict(human_recordings=parts)
            result=group_recordings(source,root)
            row=result['human_recordings']['recording']
            self.assertEqual(row['frames'],300)
            self.assertAlmostEqual(row['models']['new']['mean_m'],.75)
            self.assertNotIn('median_m',row['models']['new'])
            self.assertEqual(row['car_ordinals'],[3819])
            self.assertEqual(group_recordings(result,root),result)
            self.assertEqual(len(source['human_recordings']),2)

    def test_vehicle_scope_does_not_disable_same_car_or_global_checks(self):
        result=evaluation()
        self.assertEqual(candidate_verdict(result,'old','new')[1],'individual_human_recording_regression')
        self.assertTrue(candidate_verdict(result,'old','new',3819)[0])
        self.assertFalse(candidate_verdict(result,'old','new',4222)[0])
        self.assertFalse(candidate_verdict(result,'old','new',9999)[0])
        result['human_recordings']['0']['models']['new']['mean_m']=1.1
        self.assertEqual(candidate_verdict(result,'old','new',3819)[1],'individual_human_recording_regression')
        result=evaluation();result['models']['new']['mean_m']=1.1
        self.assertFalse(candidate_verdict(result,'old','new',3819)[0])
        result=evaluation();result.pop('recording_grouping_version')
        self.assertFalse(candidate_verdict(result,'old','new',3819)[0])

    def test_scoped_model_waits_for_correct_car_and_reverts_after_car_change(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'out/run').mkdir(parents=True)
            (root/'configs').mkdir();(root/'models/candidates').mkdir(parents=True)
            candidate=root/'models/candidates/new.onnx';candidate.write_bytes(b'test')
            config=root/'configs/forza_horizon6.json'
            config.write_text(json.dumps(dict(models=dict(pathnet='old'))))
            (root/'out/control.json').write_text('{"action":"drive"}')
            self.assertTrue(offer(root,candidate,'old',evaluation(str(candidate)),3819))
            heart=dict(time=100.,armed=True,car_ordinal=4222,model_hot_reload=True,run=str(root/'out/run'))
            service(root,heart,dict(recoveries=0),100.)
            self.assertEqual(json.loads(config.read_text())['models']['pathnet'],'old')
            heart['car_ordinal']=3819
            service(root,heart,dict(recoveries=0),100.)
            self.assertEqual(json.loads(config.read_text())['models']['pathnet'],str(candidate))
            state=json.loads((root/'out/model-activation.json').read_text())
            state['state']='active';(root/'out/model-activation.json').write_text(json.dumps(state))
            heart.update(car_ordinal=4222,time=900.)
            service(root,heart,dict(recoveries=0),900.)
            self.assertEqual(json.loads(config.read_text())['models']['pathnet'],'old')
            self.assertFalse((root/'out/run/stop').exists())

    def test_failed_live_candidate_is_not_reoffered(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'out').mkdir();(root/'configs').mkdir()
            candidate=str(root/'new.onnx')
            for name,value in [('learning-experiment-state',dict(candidate=candidate,baseline='old',evaluation={})),
                               ('model-activation',dict(candidate=candidate,state='rolled_back'))]:
                (root/'out'/f'{name}.json').write_text(json.dumps(value))
            (root/'configs/forza_horizon6.json').write_text('{"models":{"pathnet":"old"}}')
            (root/'configs/active_vehicle.json').write_text('{"car_ordinal":3819}')
            review_completed(root,100.)
            result=json.loads((root/'out/vehicle-candidate-review.json').read_text())
            self.assertFalse(result['offered'])
            self.assertEqual(result['reason'],'previous_live_trial_failed')
