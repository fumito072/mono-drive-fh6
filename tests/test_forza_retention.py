import datetime as dt
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from mono_drive.forza_retention import plan_retention,protected_paths,remove_local_tree,reclaim_old_runs


class RetentionTests(unittest.TestCase):
    def test_oldest_first_keeps_recent_active_and_human_data(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'out').mkdir();folders=[]
            start=dt.datetime(2020,1,1)
            for i in range(80):
                p=root/'out'/('fh6_'+(start+dt.timedelta(hours=i)).strftime('%Y%m%d_%H%M%S'))
                p.mkdir();folders.append(p)
            with patch('mono_drive.forza_retention.protected_paths',return_value={folders[0]/'frames'}):
                candidates=plan_retention(root,dt.datetime(2020,2,1).timestamp(),{'delete_old_autonomous_runs':True})
            self.assertEqual([r['path'] for r in candidates],[str(p.resolve()) for p in folders[1:8]])
            self.assertTrue(all('human' not in r['path'] for r in candidates))
            (folders[1]/'manifest.json').write_text(json.dumps({'arguments':{'source':'human'}}))
            with patch('mono_drive.forza_retention.protected_paths',return_value={folders[0]/'frames'}):
                candidates=plan_retention(root,dt.datetime(2020,2,1).timestamp(),{'delete_old_autonomous_runs':True})
            self.assertNotIn(str(folders[1].resolve()),[r['path'] for r in candidates])

    def test_training_manifest_protects_independent_partition_snapshot(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'out').mkdir()
            p=root/'datasets/fh6_candidates_20200101_000000/segment_001'
            (root/'out/training-dataset.json').write_text(json.dumps({'segments':[str(p)]}))
            self.assertIn(p.resolve(),protected_paths(root))

    def test_runtime_calibration_and_regression_images_are_protected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'models/ui/fh6').mkdir(parents=True);(root/'tests').mkdir()
            image=root/'out/rescue_20200101_000000/map.jpg'
            (root/'models/ui/fh6/return-point.json').write_text(json.dumps({'map_image':str(image)}))
            (root/'tests/test_example.py').write_text("image=ROOT/'out/fh6_20200101_010000/latest.jpg'\n")
            protected=protected_paths(root)
            self.assertIn(image.resolve(),protected)
            self.assertIn((root/'out/fh6_20200101_010000').resolve(),protected)

    def test_deletion_is_limited_to_verified_local_recording_roots(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'out/fh6_20200101_000000'
            (target/'frames').mkdir(parents=True);(target/'frames/x.jpg').write_bytes(b'frame')
            self.assertEqual(remove_local_tree(target,root),[1,5])
            self.assertFalse(target.exists())
            protected=root/'datasets/human';protected.mkdir(parents=True)
            with self.assertRaises(ValueError):remove_local_tree(protected,root)
            self.assertTrue(protected.exists())

    def test_locked_exporter_prevents_deletion(self):
        with TemporaryDirectory() as tmp,patch('mono_drive.forza_data_worker.acquire_guard',return_value=None):
            result=reclaim_old_runs(tmp,apply=True)
            self.assertEqual(result['phase'],'preparation_busy')


if __name__=='__main__':unittest.main()
