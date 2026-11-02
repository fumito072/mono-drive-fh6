"""Restart preprocessing at a safe boundary when new demonstrations arrive."""
import time
from pathlib import Path
from .forza_local_learning import read_json


class TrainingDataChanged(RuntimeError):
    pass


class HumanRevisionGuard:
    def __init__(self,path,expected,interval=2.):
        self.path=Path(path)
        self.expected=expected
        self.interval=interval
        self.next_check=0.

    def __call__(self):
        now=time.monotonic()
        if now<self.next_check:return
        self.next_check=now+self.interval
        revision=read_json(self.path,{}).get('revision')
        if revision and revision!=self.expected:
            raise TrainingDataChanged('New verified human recordings arrived; refresh training snapshot')
