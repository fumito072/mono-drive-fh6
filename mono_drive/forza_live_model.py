"""Load a validated path-model update without stopping the controller."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from .forza_local_learning import read_json


def load_model(path):
    from .pathnet import PathNet
    model = PathNet(str(path), 'cpu')
    result = model.predict(np.zeros((3, 128, 320), dtype=np.float32))
    if not all(np.isfinite(value).all() for value in (result.F, result.R, result.U, result.sd)):
        raise ValueError('Non-finite path-model output')
    return model


class LivePathModel:
    def __init__(self, root, active_path, loader=load_model, config_path=None):
        self.root = Path(root).resolve()
        self.active = self.resolve(active_path)
        self.loader = loader
        self.config_path=Path(config_path) if config_path else self.root/'configs/forza_horizon6.json'
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='path-model-loader')
        self.future = None
        self.loading = None
        self.next_check = 0.
        self.status = dict(state='active', model=str(self.active))

    def resolve(self, path):
        path = Path(path)
        return (path if path.is_absolute() else self.root / path).resolve()

    def poll(self, now):
        replacement = None
        if self.future is not None and self.future.done():
            try:
                replacement = self.future.result()
                self.active = self.loading
                self.status = dict(state='active', model=str(self.active), activated_at=now)
            except Exception as error:
                self.status = dict(state='load_failed', model=str(self.active),
                                   candidate=str(self.loading), error=str(error))
                self.next_check = now + 30.
            self.future = self.loading = None
        if self.future is None and now >= self.next_check:
            self.next_check = now + 2.
            value = read_json(self.config_path, {}).get('models', {}).get('pathnet')
            if value:
                candidate = self.resolve(value)
                if candidate != self.active and (self.root/'models').resolve() in candidate.parents and candidate.suffix == '.onnx':
                    self.loading = candidate
                    self.future = self.pool.submit(self.loader, candidate)
                    self.status = dict(state='loading', model=str(self.active), candidate=str(candidate))
        return replacement

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
