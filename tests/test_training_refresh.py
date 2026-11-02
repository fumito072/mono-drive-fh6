import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from mono_drive.training_refresh import HumanRevisionGuard,TrainingDataChanged


class TrainingRefreshTests(unittest.TestCase):
    def test_only_new_complete_manifest_interrupts_training(self):
        with TemporaryDirectory() as tmp:
            file=Path(tmp)/'dataset.json'
            guard=HumanRevisionGuard(file,'initial',interval=0.)
            guard()
            file.write_text('{bad')
            guard()
            file.write_text(json.dumps(dict(revision='initial')))
            guard()
            file.write_text(json.dumps(dict(revision='new')))
            with self.assertRaises(TrainingDataChanged):guard()

    def test_poll_interval_bounds_dataset_reads(self):
        with TemporaryDirectory() as tmp:
            file=Path(tmp)/'dataset.json';file.write_text('{"revision":"initial"}')
            guard=HumanRevisionGuard(file,'initial')
            with patch('mono_drive.training_refresh.time.monotonic',return_value=10.):guard()
            file.write_text('{"revision":"new"}')
            with patch('mono_drive.training_refresh.time.monotonic',return_value=11.):guard()
            with patch('mono_drive.training_refresh.time.monotonic',return_value=12.1):
                with self.assertRaises(TrainingDataChanged):guard()
