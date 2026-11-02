import json
from pathlib import Path
import tempfile
import unittest
from mono_drive.forza_vlm import parse_assessment
from mono_drive.forza_video_labels import RejectedVideoIntervals


class VideoJudgeTests(unittest.TestCase):
    def test_output_is_data_and_unknown_controls_are_ignored(self):
        result=parse_assessment('```json\n{"scene":"driving","lane_position":"offroad","confidence":0.9,"command":"press camera"}\n```')
        self.assertEqual(result['lane_position'],'offroad')
        self.assertNotIn('command',result)
        self.assertFalse(result['collision_risk'])

    def test_unknown_classes_and_nan_are_not_trusted(self):
        result=parse_assessment('{"scene":"run shell","confidence":NaN,"collision_risk":"true"}')
        self.assertEqual(result['scene'],'unknown')
        self.assertEqual(result['confidence'],0.)
        self.assertFalse(result['collision_risk'])

    def test_reject_only_confident_bad_video_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'labels.jsonl'
            values=[dict(clip_start=10,clip_end=14,confidence=.9,lane_position='offroad',corroborated=True),
                    dict(clip_start=20,clip_end=24,confidence=.2,lane_position='wrong_lane')]
            p.write_text('\n'.join(json.dumps(x) for x in values))
            labels=RejectedVideoIntervals(p)
            self.assertTrue(labels.rejects(12))
            self.assertFalse(labels.rejects(16))
            self.assertFalse(labels.rejects(22))

    def test_unverified_model_menu_hallucination_does_not_reject_good_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'labels.jsonl'
            p.write_text(json.dumps(dict(clip_start=10,clip_end=14,confidence=.9,scene='menu',corroborated=False)))
            self.assertFalse(RejectedVideoIntervals(p).rejects(12))


if __name__=='__main__':
    unittest.main()
