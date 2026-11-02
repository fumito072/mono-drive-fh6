import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mono_drive.training_arguments import dataset_arguments, resolve_dataset_arguments


class TrainingManifestTests(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        dataset_arguments(parser)
        return parser

    def test_large_dataset_and_space_paths_do_not_expand_command_line(self):
        with TemporaryDirectory() as temp:
            manifest = Path(temp) / 'data list.json'
            drives = [str(Path(temp) / ('long recording path ' + str(i))) for i in range(1000)]
            held = [str(Path(temp) / 'independent validation')]
            manifest.write_text(json.dumps(dict(drives=drives, extra_validation=held)), encoding='utf-8')
            parser = self.parser()
            command = ['--dataset-manifest', str(manifest)]
            args = resolve_dataset_arguments(parser.parse_args(command), parser)
            self.assertEqual(args.drives, drives)
            self.assertEqual(args.extra_validation, held)
            self.assertLess(sum(map(len, command)), 1000)

    def test_validation_leakage_is_rejected(self):
        parser = self.parser()
        with self.assertRaises(SystemExit):
            resolve_dataset_arguments(parser.parse_args(['same', '--extra-validation', 'same']), parser)

    def test_existing_positional_cli_is_preserved(self):
        parser = self.parser()
        args = resolve_dataset_arguments(parser.parse_args(['drive one', 'drive two']), parser)
        self.assertEqual(args.drives, ['drive one', 'drive two'])
