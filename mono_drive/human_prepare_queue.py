"""Prepare independent verified recorder sessions with at most two workers."""
from concurrent.futures import ThreadPoolExecutor
import subprocess

from .human_recorder import prepare


def prepare_native_records(root, records, workers=2, preparer=prepare):
    def one(item):
        name,record=item
        try:
            if not record.get('path_candidate_eligible'):
                raise ValueError(record.get('error',record.get('verdict')))
            return name,(preparer(root,record),None)
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
            return name,(None,str(error))
    with ThreadPoolExecutor(max_workers=max(1,min(2,workers))) as pool:
        return dict(pool.map(one,records))
