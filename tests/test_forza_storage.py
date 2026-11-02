import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from mono_drive.forza_storage import cache_candidates,reclaim_frame_caches
from mono_drive.forza_local_monitor import storage_cleanup_mode


class StorageTests(unittest.TestCase):
    def test_requested_retention_cleans_periodically_without_blocking_watchdog(self):
        policy=dict(delete_verified_decode_caches=True)
        self.assertEqual(storage_cleanup_mode(4000.,100*1024**3,0.,policy),'all')
        self.assertIsNone(storage_cleanup_mode(4000.,100*1024**3,3000.,policy))
        self.assertEqual(storage_cleanup_mode(4000.,20*1024**3,3500.,policy),'all')
        self.assertEqual(storage_cleanup_mode(4000.,20*1024**3,0.,{}),'reserve')
        self.assertIsNone(storage_cleanup_mode(4000.,100*1024**3,0.,{}))

    def setup_cache(self,root):
        folder=root/'datasets/human/session';cache=folder/'video_frames';cache.mkdir(parents=True)
        source=root/'recording/video.mkv';source.parent.mkdir();source.write_bytes(b'original video')
        frame=cache/'0000000.jpg';frame.write_bytes(b'training image')
        partitions=[]
        for name in ('train','validation'):
            p=folder/name;(p/'frames').mkdir(parents=True)
            os.link(frame,p/'frames/0000000.jpg')
            (p/'poses.csv').write_text('frame,t,x,y,z\n0,0,0,0,0\n')
            partitions.append(str(p))
        (folder/'manifest.json').write_text(json.dumps(dict(recording='session',train=partitions[:1],validation=partitions[1:])))
        (root/'out/human-data-audit').mkdir(parents=True)
        (root/'out/human-data-audit/session.json').write_text(json.dumps(dict(path=str(source.parent))))
        return cache,source,partitions

    def test_pruning_keeps_original_and_all_training_hardlinks(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);cache,source,parts=self.setup_cache(root)
            self.assertEqual(len(list(cache_candidates(root))),1)
            # Tests must not contend with a real import/cleanup on this PC.
            with patch('mono_drive.forza_data_worker.acquire_guard'):
                result=reclaim_frame_caches(root,target_free_gib=1e9)
            self.assertEqual(result['files'],1)
            self.assertFalse(cache.exists())
            self.assertEqual(source.read_bytes(),b'original video')
            for part in parts:self.assertEqual((Path(part)/'frames/0000000.jpg').read_bytes(),b'training image')

    def test_incomplete_preparation_and_unknown_cache_files_are_preserved(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);cache,source,parts=self.setup_cache(root)
            (cache/'user.txt').write_text('keep')
            self.assertEqual(list(cache_candidates(root)),[])
            (cache/'user.txt').unlink()
            (Path(parts[0])/'frames/0000000.jpg').unlink()
            self.assertEqual(list(cache_candidates(root)),[])
            self.assertTrue(source.exists())

    def test_active_preparation_prevents_cleanup(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);cache,_,_=self.setup_cache(root)
            with patch('mono_drive.forza_data_worker.acquire_guard',return_value=None):
                result=reclaim_frame_caches(root,target_free_gib=1e9)
            self.assertEqual(result['phase'],'preparation_busy')
            self.assertTrue(cache.exists())

    def test_explicit_completed_cleanup_is_independent_of_free_space(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);cache,source,parts=self.setup_cache(root)
            with patch('mono_drive.forza_data_worker.acquire_guard'):
                result=reclaim_frame_caches(root,target_free_gib=0.)
                self.assertEqual(result['files'],0)
                result=reclaim_frame_caches(root,target_free_gib=0.,all_completed=True)
            self.assertEqual(result['files'],1)
            self.assertTrue(source.is_file())
            self.assertTrue((Path(parts[0])/'frames/0000000.jpg').is_file())
