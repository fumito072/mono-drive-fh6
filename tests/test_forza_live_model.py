import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from mono_drive.forza_live_model import LivePathModel


class LiveModelTests(unittest.TestCase):
    def fixture(self,root):
        (root/'models/candidates').mkdir(parents=True)
        (root/'configs').mkdir()
        p=root/'configs/forza_horizon6.json'
        p.write_text(json.dumps(dict(models=dict(pathnet='models/candidates/new.onnx'))))
        return p

    def test_completed_model_is_swapped_only_after_background_load(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root);replacement=object()
            live=LivePathModel(root,'models/old.onnx',loader=lambda _:replacement)
            try:
                self.assertIsNone(live.poll(0.))
                live.future.result(timeout=2)
                self.assertIs(live.poll(1.),replacement)
                self.assertEqual(live.active,root/'models/candidates/new.onnx')
                self.assertIsNone(live.poll(2.))
            finally:live.close()

    def test_invalid_update_retains_current_model(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            def fail(_):raise ValueError('bad model')
            live=LivePathModel(root,'models/old.onnx',loader=fail)
            try:
                live.poll(0.)
                try:live.future.result(timeout=2)
                except ValueError:pass
                self.assertIsNone(live.poll(1.))
                self.assertEqual(live.active,root/'models/old.onnx')
                self.assertEqual(live.status['state'],'load_failed')
                self.assertIsNone(live.future)
            finally:live.close()

    def test_does_not_load_paths_outside_project_models(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);config=self.fixture(root)
            config.write_text(json.dumps(dict(models=dict(pathnet='../external.onnx'))))
            live=LivePathModel(root,'models/old.onnx',loader=lambda _:self.fail('unexpected load'))
            try:
                self.assertIsNone(live.poll(0.))
                self.assertIsNone(live.future)
            finally:live.close()


if __name__=='__main__':unittest.main()
