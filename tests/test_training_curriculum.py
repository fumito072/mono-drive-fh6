import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from mono_drive.training_curriculum import enabled,sampling_weights,BestValidationCheckpoint


class CurriculumTests(unittest.TestCase):
    def test_rare_bends_receive_more_weight_without_changing_targets(self):
        t=np.zeros((100,3,3));t[:,:,0]=[4,12,20]
        t[90:95,:,1]=-5;t[95:,:,1]=5
        valid=np.ones((100,3),bool);original=t.copy()
        w,groups=sampling_weights(t,valid,np.array([4,12,20]),balanced=True)
        self.assertEqual(groups,[90,0,0,5,5])
        self.assertGreater(w[90],w[0]);self.assertAlmostEqual(w.sum(),1.)
        np.testing.assert_array_equal(t,original)
        baseline,_=sampling_weights(t,valid,np.array([4,12,20]))
        np.testing.assert_allclose(baseline,np.full(100,.01))

    def test_requires_matching_completed_data_revision(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'out';p.mkdir()
            (p/'human-dataset.json').write_text(json.dumps({'revision':'a','frames':20000}))
            (p/'training-curriculum.json').write_text(json.dumps({'revision':'b','balanced_maneuvers':True}))
            self.assertFalse(enabled(d))
            (p/'training-curriculum.json').write_text(json.dumps({'revision':'a','balanced_maneuvers':True}))
            self.assertTrue(enabled(d))
            (p/'training-curriculum.json').write_text(json.dumps({'revision':'old','balanced_maneuvers':True,'follow_new_verified_data':True}))
            self.assertTrue(enabled(d))
            (p/'human-dataset.json').write_text(json.dumps({'revision':'new','frames':10}))
            self.assertFalse(enabled(d))

    def test_restores_the_best_epoch_instead_of_the_last(self):
        try:import torch
        except ImportError:self.skipTest('PyTorch optional')
        model=torch.nn.Linear(1,1,bias=False)
        best=BestValidationCheckpoint()
        with torch.no_grad():model.weight.fill_(1.)
        self.assertTrue(best.consider(model,.5,1))
        with torch.no_grad():model.weight.fill_(2.)
        self.assertTrue(best.consider(model,.4,2))
        with torch.no_grad():model.weight.fill_(3.)
        self.assertFalse(best.consider(model,.6,3))
        self.assertTrue(best.restore(model))
        self.assertEqual(best.epoch,2);self.assertEqual(model.weight.item(),2.)
