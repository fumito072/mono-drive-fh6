import unittest
from mono_drive.human_prepare_queue import prepare_native_records


class PrepareQueueTests(unittest.TestCase):
    def test_ineligible_or_broken_recording_does_not_remove_good_results(self):
        def prepare(root,record):
            if record['value']==2:raise ValueError('bad synchronization')
            return dict(frames=record['value'])
        records=[('ok',dict(path_candidate_eligible=True,value=1)),
                 ('bad',dict(path_candidate_eligible=True,value=2)),
                 ('excluded',dict(path_candidate_eligible=False,verdict='no telemetry'))]
        result=prepare_native_records('.',records,preparer=prepare)
        self.assertEqual(list(result),['ok','bad','excluded'])
        self.assertEqual(result['ok'],(dict(frames=1),None))
        self.assertEqual(result['bad'],(None,'bad synchronization'))
        self.assertEqual(result['excluded'],(None,'no telemetry'))
