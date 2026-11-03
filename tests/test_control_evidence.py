import json,struct,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from mono_drive.control_evidence import inspect_controls


class ControlEvidenceTests(unittest.TestCase):
    def test_completion_flag_cannot_invent_samples(self):
        with TemporaryDirectory() as d:
            p=Path(d);(p/'telemetry.raw').write_bytes(b'FH6UDP1\n')
            (p/'telemetry.csv').write_text('steer,throttle,brake\n');(p/'session.json').write_text(json.dumps({'status':'complete'}))
            self.assertEqual(inspect_controls(p)['verdict'],'header_only_no_recorded_controls')
    def test_error_flag_does_not_hide_real_raw_data(self):
        with TemporaryDirectory() as d:
            p=Path(d);(p/'telemetry.raw').write_bytes(b'FH6UDP1\n'+struct.pack('<QQI',1,10,324)+bytes(324))
            (p/'session.json').write_text(json.dumps({'status':'error'}))
            self.assertTrue(inspect_controls(p)['raw_has_record'])
            self.assertEqual(inspect_controls(p)['verdict'],'control_records_present_requires_sync_validation')
    def test_truncated_raw_packet_is_not_a_usable_record(self):
        with TemporaryDirectory() as d:
            p=Path(d);(p/'telemetry.raw').write_bytes(b'FH6UDP1\n'+struct.pack('<QQI',1,10,324)+bytes(100))
            self.assertFalse(inspect_controls(p)['raw_has_record'])
    def test_csv_only_controls_are_candidates_not_proven_alignment(self):
        with TemporaryDirectory() as d:
            p=Path(d);(p/'telemetry.csv').write_text('Steer,Accel,Brake\n12,40,0\n')
            self.assertEqual(inspect_controls(p)['verdict'],'control_records_present_requires_sync_validation')
