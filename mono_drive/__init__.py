"""mono_drive — 単眼カメラの絵だけで「どの線を・どの速さで走るか」を決める仕組み(学習済みモデル付き)。

    from mono_drive import MonoDriver
    drv = MonoDriver("configs/sample_c4.json")
    dec = drv.step(frame_bgr, speed_kmh=40.0)
    print(dec.steer, dec.target_kmh, dec.mode)

詳しくは README.md と docs/。
"""

from .camera import REF_CAMERA, Camera, ReferenceView
from .driver import MonoDriver, load_config
from .planner import Decision, Planner, PlannerConfig, VehicleConfig

__version__ = "1.0.0"
__all__ = ["MonoDriver", "Decision", "Camera", "ReferenceView", "REF_CAMERA", "Planner", "PlannerConfig",
           "VehicleConfig", "load_config", "__version__"]
