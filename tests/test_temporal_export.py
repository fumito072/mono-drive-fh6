import tempfile
from pathlib import Path
import unittest
import numpy as np


class TemporalExportTests(unittest.TestCase):
    def test_deep_dropout_stays_disabled_after_export(self):
        import torch
        from mono_drive.temporal_steering import build_models
        from tools.train_temporal_steering import export_steering_model
        threads=torch.get_num_threads();torch.set_num_threads(2)
        try:
            torch.manual_seed(12)
            model=build_models(variant='deep')[0]
            with torch.no_grad():model.forecast.weight.normal_(0,.1)
            features=np.random.default_rng(7).normal(size=(1,21,256)).astype('float32')
            ego=np.zeros((1,21,2),dtype='float32')
            with tempfile.TemporaryDirectory() as folder:
                difference=export_steering_model(model,features,ego,Path(folder)/'model.onnx')
                self.assertLess(difference,1e-4)
            self.assertTrue(all(not child.training for child in model.modules()))
            x=(torch.from_numpy(features),torch.from_numpy(ego))
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(*x),model(*x)))
        finally:torch.set_num_threads(threads)


if __name__=='__main__':unittest.main()
