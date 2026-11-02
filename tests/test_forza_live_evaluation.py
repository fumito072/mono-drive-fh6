import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from mono_drive.forza_live_evaluation import update


class LiveEvaluationTests(unittest.TestCase):
    def test_waiting_does_not_invent_driving_metrics(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir()
            (out/'live-evaluation-request.json').write_text(json.dumps({'candidate':'model'}))
            update(root,{}, {},100.)
            state=json.loads((out/'live-evaluation-state.json').read_text())
            self.assertEqual(state['phase'],'waiting_for_training')
            self.assertEqual(state['metrics']['driving_seconds'],0.)

    def test_counts_actual_model_driving_once_and_records_rollback(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';run=out/'run';run.mkdir(parents=True)
            (out/'live-evaluation-request.json').write_text('{"candidate":"model"}')
            (out/'model-activation.json').write_text('{"candidate":"model","state":"trial","started_at":100}')
            rows=[dict(t=100+i*.2,path_model='model',armed=True,state='driving',speed_kmh=60.,
                       position=[i*3.,0,0],sent_command=[0,0,0],actual_brake=0) for i in range(6)]
            (run/'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            update(root,{'run':str(run)},{},102.)
            update(root,{'run':str(run)},{},103.)
            state=json.loads((out/'live-evaluation-state.json').read_text())
            self.assertAlmostEqual(state['metrics']['driving_seconds'],1.)
            self.assertEqual(state['metrics']['average_kmh'],60.)
            (out/'model-activation.json').write_text('{"candidate":"model","state":"rolled_back","reason":"live_failure"}')
            update(root,{'run':str(run)},{},104.)
            self.assertEqual(json.loads((out/'live-evaluations/model.json').read_text())['phase'],'failed')


if __name__=='__main__':unittest.main()
