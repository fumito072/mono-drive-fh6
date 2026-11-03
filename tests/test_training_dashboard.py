import http.client
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer

from mono_drive.training_dashboard import DashboardSnapshot, make_handler, read_json, finite


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'out').mkdir()
        self.model = self.root / 'models/temporal_candidates/deep_forecast_20261103'
        self.model.mkdir(parents=True)

    def write(self, relative, value):
        path = self.root / relative
        path.write_text(json.dumps(value), encoding='utf-8')

    def snapshot(self):
        value = DashboardSnapshot(self.root)
        with patch.object(value, 'gpu_status', return_value=None):
            return value.collect()

    def test_warmup_is_not_presented_as_trained_temporal_performance(self):
        self.write('out/deep-learning-state.json', {'phase': 'training', 'pid': os.getpid(), 'time': time.time()})
        self.write('out/deep-learning-plan.json', {'supervised_windows': {'train': 40, 'validation': 15, 'test': 5}, 'maximum_epochs': 48})
        records = [{'epoch': n, 'validation': [{'mae': .3/n}, {'mae': .2/n}], 'loss': [.1, .1]} for n in (8, 9)]
        (self.root / 'out/deep-forecast.log').write_text('\n'.join(json.dumps(r) for r in records) + '\n{"partial":', encoding='utf-8')
        self.write('models/temporal_candidates/deep_forecast_20261103/training-progress.json', {'epoch': 10, 'epochs': 48, 'last': records[-1]})
        value = self.snapshot()['deep']
        self.assertEqual(value['completed_epochs'], 9)
        self.assertEqual(value['current_epoch'], 10)
        self.assertIsNone(value['history'][0]['temporal'])
        self.assertAlmostEqual(value['history'][1]['temporal']['mae'], .3/9)
        self.assertEqual(value['samples']['test'], 5)

    def test_stale_driving_telemetry_cannot_look_like_current_speed(self):
        self.write('out/driver-heartbeat.json', {'time': time.time()-3600, 'speed_kmh': 60, 'armed': True})
        self.write('out/longrun-state.json', {'state': 'running', 'target_kmh': 60})
        value = self.snapshot()['drive']
        self.assertFalse(value['live'])
        self.assertIsNone(value['speed_kmh'])
        self.assertIn('停止', value['label'])
        self.assertEqual(value['last_speed_kmh'], 60)

    def test_explicit_stop_overrides_live_training_status(self):
        self.write('out/deep-learning-state.json', {'phase': 'training', 'pid': os.getpid(), 'time': time.time()})
        (self.root / 'out/training.stop').write_text('stop', encoding='utf-8')
        self.assertEqual(self.snapshot()['deep']['health']['phase'], 'stopped')

    def test_model_path_cannot_read_outside_models(self):
        value = DashboardSnapshot(self.root)
        for path in ('../private', str(self.root / 'out')):
            self.assertEqual(value.model_directory({'model_dir': path}), self.model)

    def test_partial_or_nonfinite_input_is_handled(self):
        path = self.root / 'out/broken.json'
        path.write_text('{"unfinished":', encoding='utf-8')
        self.assertEqual(read_json(path), {})
        self.assertIsNone(finite(float('nan')))
        self.assertIsNone(finite(True))
        self.assertEqual(self.snapshot()['deep']['completed_epochs'], 0)

    def test_http_is_read_only_and_does_not_expose_files(self):
        class Snapshot:
            def get(self): return {'live': True}
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(Snapshot(), 'test'))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
            for method, path, expected in [('GET', '/api/status', 200), ('POST', '/api/status', 501), ('GET', '/../out/deep-learning-state.json', 404)]:
                conn.request(method, path)
                response = conn.getresponse()
                self.assertEqual(response.status, expected)
                response.read()
            conn.request('GET', '/api/status', headers={'Host': 'untrusted.example'})
            response = conn.getresponse()
            self.assertEqual(response.status, 403)
            response.read()
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == '__main__':
    unittest.main()
