import unittest
from pathlib import Path
from tools.watch_deep_training import ROOT,training_command_matches


class DeepWatchTests(unittest.TestCase):
    def test_only_the_requested_workspace_training_job_matches(self):
        folder=ROOT/'models/example'
        command=['python.exe','-u','tools/train_temporal_steering.py','--out','models/example','--variant','deep']
        self.assertTrue(training_command_matches(command,folder))
        self.assertFalse(training_command_matches(command,ROOT/'models/other'))
        self.assertFalse(training_command_matches(command[:-1]+['base'],folder))
        self.assertFalse(training_command_matches(command,folder,ROOT.parent/'different-workspace'))
        self.assertFalse(training_command_matches(['python.exe','other.py'],folder))

