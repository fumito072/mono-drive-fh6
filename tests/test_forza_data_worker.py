import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from mono_drive.forza_data_worker import active,acquire_guard


class DataWorkerTests(unittest.TestCase):
    def test_preparation_has_only_one_owner(self):
        first=acquire_guard(0)
        try:self.assertIsNone(acquire_guard(first.getsockname()[1]))
        finally:first.close()

    def test_obeys_user_stop_and_original_session_end(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';out.mkdir()
            (out/'longrun.json').write_text(json.dumps(dict(ends_at=time.time()+100)))
            self.assertTrue(active(root))
            (out/'longrun.stop').write_text('user')
            self.assertFalse(active(root))
            (out/'longrun.stop').unlink()
            (out/'longrun.json').write_text(json.dumps(dict(ends_at=time.time()-1)))
            self.assertFalse(active(root))


if __name__=='__main__':unittest.main()
