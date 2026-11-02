import json
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest
import zipfile

from mono_drive.forza_drive_import import sync_once, safe_zip_name
from mono_drive.human_recorder import FILES


class DriveImportTests(unittest.TestCase):
    def setup_tree(self, temp):
        base = Path(temp)
        root = base/'project'
        source = base/'Drive folder'
        for p in (root/'out', root/'configs', source):
            p.mkdir(parents=True)
        config = dict(enabled=True, local_folder=str(source), expected_account='person@example.com', stable_seconds=60)
        (root/'configs/drive_data_source.json').write_text(json.dumps(config))
        return root, source

    def zip(self, path, complete=True, extra=None):
        with zipfile.ZipFile(path, 'w') as archive:
            for name in FILES:
                payload = json.dumps(dict(status='complete' if complete else 'recording', synthetic_demo=False, errors=[])) if name=='session.json' else name
                archive.writestr('recording/'+name, payload)
            if extra:
                archive.writestr(extra, 'never run this')

    def test_stable_archive_imports_once_and_never_extracts_code(self):
        with TemporaryDirectory() as temp:
            root, source = self.setup_tree(temp)
            self.zip(source/'record.zip', extra='recording/run.py')
            self.assertEqual(sync_once(root, 0)['imported_sessions'], 0)
            self.assertEqual(sync_once(root, 61)['imported_sessions'], 1)
            self.assertEqual(sync_once(root, 122)['imported_sessions'], 0)
            folders = list((root.parent/'Play-data').glob('drive_*'))
            self.assertEqual(len(folders), 1)
            self.assertFalse((folders[0]/'run.py').exists())
            self.assertEqual(set(FILES), {p.name for p in folders[0].iterdir()}-{'import-source.json'})
            self.assertTrue((source/'record.zip').exists())

    def test_unfinished_recording_is_not_published(self):
        with TemporaryDirectory() as temp:
            root, source = self.setup_tree(temp)
            self.zip(source/'record.zip', complete=False)
            sync_once(root, 0)
            result = sync_once(root, 61)
            self.assertEqual(result['imported_sessions'], 0)
            self.assertTrue(result['errors'])
            self.assertFalse(list((root.parent/'Play-data').glob('drive_*')))

    def test_changed_archive_waits_again(self):
        with TemporaryDirectory() as temp:
            root, source = self.setup_tree(temp)
            self.zip(source/'record.zip')
            sync_once(root, 0)
            self.zip(source/'record.zip', extra='other.txt')
            self.assertEqual(sync_once(root, 61)['imported_sessions'], 0)
            self.assertEqual(sync_once(root, 122)['imported_sessions'], 1)

    def test_traversal_and_links_are_rejected(self):
        for name in ('../outside', '/outside', 'C:\\outside', 'folder/../../outside'):
            with self.assertRaises(ValueError):
                safe_zip_name(zipfile.ZipInfo(name))
        info = zipfile.ZipInfo('link')
        info.external_attr = (stat.S_IFLNK|0o777)<<16
        with self.assertRaises(ValueError):
            safe_zip_name(info)

    def test_complete_folder_is_copied_without_source_mutation(self):
        with TemporaryDirectory() as temp:
            root, source = self.setup_tree(temp)
            folder = source/'session';folder.mkdir()
            for name in FILES:
                (folder/name).write_text(json.dumps(dict(status='complete', synthetic_demo=False, errors=[])) if name=='session.json' else name)
            before = {p.name:p.read_bytes() for p in folder.iterdir()}
            sync_once(root, 0)
            self.assertEqual(sync_once(root, 61)['imported_sessions'], 1)
            self.assertEqual(before, {p.name:p.read_bytes() for p in folder.iterdir()})

    def test_missing_connection_does_not_scan_or_claim_authentication(self):
        with TemporaryDirectory() as temp:
            root, source = self.setup_tree(temp)
            (root/'configs/drive_data_source.json').write_text('{"enabled":false}')
            result = sync_once(root, 1)
            self.assertEqual(result['phase'], 'waiting_for_connection')
            self.assertFalse(result['account_verified'])

    def test_existing_session_is_not_duplicated_by_an_archive(self):
        with TemporaryDirectory() as temp:
            root, source = self.setup_tree(temp)
            archive = source/'new_name.zip';self.zip(archive)
            existing = root.parent/'Play-data'/'old_name';existing.mkdir(parents=True)
            with zipfile.ZipFile(archive) as z:
                for name in FILES:
                    (existing/name).write_bytes(z.read('recording/'+name))
            sync_once(root, 0)
            self.assertEqual(sync_once(root, 61)['imported_sessions'], 0)
            self.assertEqual([p.name for p in existing.parent.iterdir()], ['old_name'])

    def test_archive_and_folder_use_the_same_content_identity(self):
        with TemporaryDirectory() as temp:
            root, source = self.setup_tree(temp)
            archive = source/'session.zip';self.zip(archive)
            folder = source/'unpacked';folder.mkdir()
            with zipfile.ZipFile(archive) as z:
                for name in FILES:
                    (folder/name).write_bytes(z.read('recording/'+name))
            sync_once(root, 0)
            self.assertEqual(sync_once(root, 61)['imported_sessions'], 1)
            self.assertEqual(len(list((root.parent/'Play-data').glob('drive_*'))), 1)
