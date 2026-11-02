"""Pass growing datasets by manifest rather than Windows command-line arguments."""
from __future__ import annotations

import json
from pathlib import Path


def dataset_arguments(parser):
    parser.add_argument('drives', nargs='*')
    parser.add_argument('--extra-validation', nargs='+', default=[])
    parser.add_argument('--dataset-manifest', help='JSON with drives and extra_validation lists')


def resolve_dataset_arguments(args, parser):
    if args.dataset_manifest:
        try:
            value = json.loads(Path(args.dataset_manifest).read_text(encoding='utf-8'))
            for key, attr in (('drives', 'drives'), ('extra_validation', 'extra_validation')):
                entries = value.get(key, [])
                if not isinstance(entries, list) or any(not isinstance(p, str) or not p for p in entries):
                    raise ValueError(f'{key} must be a list of nonempty paths')
                setattr(args, attr, getattr(args, attr) + entries)
        except (OSError, ValueError, AttributeError) as error:
            parser.error(f'Invalid dataset manifest: {error}')
    if not args.drives:
        parser.error('At least one training drive is required')
    train = {Path(p).resolve() for p in args.drives}
    validation = {Path(p).resolve() for p in args.extra_validation}
    if train & validation:
        parser.error('A drive cannot be used for both training and independent validation')
    return args
