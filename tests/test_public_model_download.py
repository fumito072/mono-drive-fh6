import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('download_model_weights',Path(__file__).resolve().parents[1]/'tools/download_model_weights.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)


class PublicModelDownloadTests(unittest.TestCase):
    def test_reassembles_exact_bytes_and_protects_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'a';b=root/'b';out=root/'models/test.bin'
            a.write_bytes(b'abc');b.write_bytes(b'def')
            record={'bytes':6,'sha256':hashlib.sha256(b'abcdef').hexdigest()}
            model.restore_parts([a,b],out,record)
            self.assertEqual(out.read_bytes(),b'abcdef')
            model.restore_parts([a,b],out,record)
            out.write_bytes(b'user-file')
            with self.assertRaises(FileExistsError):model.restore_parts([a,b],out,record)
            model.restore_parts([a,b],out,record,True)
            b.write_bytes(b'bad')
            with self.assertRaises(ValueError):model.restore_parts([a,b],root/'other',record)

    def test_rejects_paths_outside_model_locations(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ['../x','/x','models/../../x','C:/x','models\\x','credentials.json']:
                with self.assertRaises(ValueError):model.destination(d,name)
            self.assertEqual(model.destination(d,'models/qwen/model.safetensors'),Path(d)/'models/qwen/model.safetensors')
