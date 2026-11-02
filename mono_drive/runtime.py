"""ONNX の推論(onnxruntime だけ)。GPU があれば使う: NVIDIA は CUDA、Mac は Core ML(MLProgram・GPU)。"""

from __future__ import annotations

import os
import tempfile

#: "auto" | "cpu" | "cuda" | "coreml"(環境変数 MONO_DRIVE_DEVICE でも選べる)
DEFAULT_DEVICE = os.environ.get("MONO_DRIVE_DEVICE", "auto")


def _static_inputs(sess_or_path) -> bool:
    import onnxruntime as ort
    s = sess_or_path if not isinstance(sess_or_path, str) else ort.InferenceSession(
        sess_or_path, providers=["CPUExecutionProvider"])
    return all(isinstance(d, int) for i in s.get_inputs() for d in i.shape)


def make_session(path: str, device: str | None = None, threads: int = 0, gpu_ok: bool = True):
    """ONNX のセッション。device="auto" なら CUDA → Core ML(入力の形が決まったモデルだけ)→ CPU。

    Core ML の既定(NeuralNetwork 形式)は深度のモデルで CPU に落ちて 1 枚 268ms だった。MLProgram + GPU なら 7ms
    (Apple M5 Max の実測)。小さいモデル(PathNet・障害物)は CPU で 1 枚 2〜5ms なので CPU のままにする(gpu_ok=False)。
    """
    import onnxruntime as ort
    device = (device or DEFAULT_DEVICE).lower()
    avail = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if device in ("auto", "cuda") and gpu_ok and "CUDAExecutionProvider" in avail:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device in ("auto", "coreml") and gpu_ok and "CoreMLExecutionProvider" in avail and _static_inputs(path):
        cache = os.path.join(tempfile.gettempdir(), "mono_drive_coreml_cache")
        os.makedirs(cache, exist_ok=True)
        providers = [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU",
                                                  "ModelCacheDirectory": cache}), "CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    threads = threads or int(os.environ.get("MONO_DRIVE_THREADS", "4"))
    if threads:
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
    try:
        return ort.InferenceSession(path, sess_options=so, providers=providers)
    except Exception:
        if providers == ["CPUExecutionProvider"]:
            raise
        return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
