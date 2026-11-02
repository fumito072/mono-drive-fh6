"""学び直し用の torch のモデル(同梱の ONNX から重みを読み込める形)。

同梱の ONNX はバッチ正規化を畳み込みに畳み込んだ形(Conv にバイアスあり)で書き出してある。ここではその形の
torch モデルを作り、ONNX の重みをそのまま入れる(続きから学べる)。--scratch で一から学ぶときもこの形で学ぶ。
"""

from __future__ import annotations

import numpy as np


def onnx_conv_weights(path: str):
    """ONNX → [(重み, バイアス, 属性 dict)](Conv の順)と、Gemm の [(重み, バイアス)]。"""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    convs, gemms = [], []
    for n in m.graph.node:
        if n.op_type == "Conv":
            attrs = {a.name: list(a.ints) if a.ints else a.i for a in n.attribute}
            w = init[n.input[1]]
            b = init[n.input[2]] if len(n.input) > 2 and n.input[2] in init else np.zeros(w.shape[0], np.float32)
            convs.append((w, b, attrs))
        elif n.op_type == "Gemm":
            attrs = {a.name: a.i if a.type == 2 else a.f for a in n.attribute}
            w = init[n.input[1]]
            if not attrs.get("transB", 0):
                w = w.T
            gemms.append((w, init[n.input[2]]))
    return convs, gemms


def build_pathnet():
    import torch.nn as nn

    def cr(cin, cout, k, s):
        return [nn.Conv2d(cin, cout, k, s, k // 2, bias=True), nn.ReLU(inplace=True)]

    class PathNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers = cr(3, 24, 5, 2)
            for cin, cout in ((24, 32), (32, 48), (48, 64), (64, 96)):
                layers += cr(cin, cout, 3, 2) + cr(cout, cout, 3, 1)
            self.body = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(96 * 4 * 10, 256), nn.ReLU(inplace=True),
                                      nn.Dropout(0.2), nn.Linear(256, 44))

        def forward(self, x):
            return self.head(self.body(x))

    return PathNet()


def build_obstacle():
    import torch
    import torch.nn as nn

    def conv(cin, cout, k=3, s=1, d=1):
        return nn.Conv2d(cin, cout, k, s, d * (k // 2), dilation=d, bias=True)

    class ObstacleNet(nn.Module):
        def __init__(self, c=(16, 24, 40, 64, 96), dd=(48, 24)):
            super().__init__()
            r = nn.ReLU(inplace=True)
            self.stem = nn.Sequential(conv(3, c[0], 3, 2), r)
            self.e1 = nn.Sequential(conv(c[0], c[1], 3, 2), nn.ReLU(inplace=True), conv(c[1], c[1]), nn.ReLU(inplace=True))
            self.e2 = nn.Sequential(conv(c[1], c[2], 3, 2), nn.ReLU(inplace=True), conv(c[2], c[2]), nn.ReLU(inplace=True))
            self.e3 = nn.Sequential(conv(c[2], c[3], 3, 2), nn.ReLU(inplace=True), conv(c[3], c[3]), nn.ReLU(inplace=True))
            self.ctx = nn.Sequential(conv(c[3], c[4], 3, 1, 2), nn.ReLU(inplace=True), conv(c[4], c[4], 3, 1, 2),
                                     nn.ReLU(inplace=True))
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.r2 = nn.Conv2d(c[4], dd[0], 1)
            self.d2 = nn.Sequential(conv(dd[0] + c[2], dd[0]), nn.ReLU(inplace=True))
            self.r1 = nn.Conv2d(dd[0], dd[1], 1)
            self.d1 = nn.Sequential(conv(dd[1] + c[1], dd[1]), nn.ReLU(inplace=True))
            self.head = nn.Conv2d(dd[1], 1, 1)

        def forward(self, x):
            f1 = self.e1(self.stem(x))
            f2 = self.e2(f1)
            f3 = self.ctx(self.e3(f2))
            y = self.d2(torch.cat([self.up(self.r2(f3)), f2], 1))
            y = self.d1(torch.cat([self.up(self.r1(y)), f1], 1))
            return self.head(y)

    return ObstacleNet()


def load_onnx_into(model, path: str) -> None:
    """ONNX の重みを、同じ並びの torch モデルへ入れる(Conv は出てくる順、Linear も順に)。"""
    import torch
    import torch.nn as nn
    convs, gemms = onnx_conv_weights(path)
    t_convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    t_lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(t_convs) != len(convs) or len(t_lins) != len(gemms):
        raise ValueError(f"{path}: 層の数が合わない(Conv {len(convs)} 対 {len(t_convs)}・Linear {len(gemms)} 対 {len(t_lins)})")
    with torch.no_grad():
        for m, (w, b, _) in zip(t_convs, convs):
            if tuple(m.weight.shape) != w.shape:
                raise ValueError(f"Conv の形が合わない {tuple(m.weight.shape)} 対 {w.shape}")
            m.weight.copy_(torch.from_numpy(np.array(w, np.float32, copy=True)))
            m.bias.copy_(torch.from_numpy(np.array(b, np.float32, copy=True)))
        for m, (w, b) in zip(t_lins, gemms):
            m.weight.copy_(torch.from_numpy(np.array(w, np.float32, copy=True)))
            m.bias.copy_(torch.from_numpy(np.array(b, np.float32, copy=True)))


def export_onnx(model, path: str, out_name: str, dynamic_batch: bool) -> float:
    """ONNX へ書き出し、onnxruntime と torch の答えの差を返す。"""
    import onnxruntime as ort
    import torch
    model = model.to("cpu").eval()
    dummy = torch.zeros(1, 3, 128, 320)
    kw = {"dynamic_axes": {"image": {0: "n"}, out_name: {0: "n"}}} if dynamic_batch else {}
    try:
        torch.onnx.export(model, dummy, path, input_names=["image"], output_names=[out_name], opset_version=17,
                          dynamo=False, **kw)
    except TypeError:                                   # 古い torch(dynamo の引数が無い)
        torch.onnx.export(model, dummy, path, input_names=["image"], output_names=[out_name], opset_version=17, **kw)
    x = np.random.default_rng(0).random((1, 3, 128, 320)).astype(np.float32)
    ref = model(torch.from_numpy(x)).detach().numpy()
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"image": x})[0]
    return float(np.abs(ref - got).max())


def pick_device(want: str = "auto") -> str:
    import torch
    if want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
