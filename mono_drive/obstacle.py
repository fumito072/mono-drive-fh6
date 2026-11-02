"""学習した障害物の検出(自前の小さな CNN。学習済み ONNX を読む)。

入力は PathNet と同じ切り出し(基準の視点の 0〜240 行 → 128x320)を ImageNet の平均・標準偏差で正規化したもの。
出力は切り出しの上の 32x80 マスの「当たる物がある」の logit。層は Conv・ReLU・Resize・Concat だけ
(Jetson の TensorRT 8.5 でもそのまま動く形)。

教師は走行の記録から自動で作った(人手のラベルなし): 実際に当たって止まった所をその手前の絵に写した箱 = 物、その後に実際に通った
帯 = 空き。学習に使っていない 2 ステージで、
誤報を約 2.8% にそろえると、当たる 1〜2 秒前に見つけた割合 約 84%(しきい値 0.84)。
雨のワイパーなど 1 枚だけの塗りを捨てるため、2 回続けて塗られたマスだけ使う(ObstacleTracker)。
"""

from __future__ import annotations

import json
import os

import numpy as np

from .camera import NET_ROWS
from .runtime import make_session

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
#: 塗りつぶしの打ち消し: 平らな地面(深度)の行の FLOOD_FRAC 以上を塗っていたら、その行の地面の上の塗りは消す
FLOOD_FRAC, FLOOD_MIN_W = 0.6, 0.02


class ObstacleNet:
    def __init__(self, onnx_path: str, thr: float | None = None, device: str | None = None):
        self.sess = make_session(onnx_path, device, threads=2, gpu_ok=False)
        self.inp = self.sess.get_inputs()[0].name
        meta = {}
        try:
            with open(os.path.splitext(onnx_path)[0] + ".json", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            pass
        self.meta = meta
        self.thr = float(thr if thr is not None else meta.get("threshold", 0.84))

    def prob(self, x_rgb01: np.ndarray) -> np.ndarray:
        """net_input の出力 (3, 128, 320) → (32, 80) の 0..1。"""
        x = (np.asarray(x_rgb01, np.float32) - MEAN) / STD
        out = self.sess.run(None, {self.inp: x[None].astype(np.float32)})[0]
        return 1.0 / (1.0 + np.exp(-np.asarray(out, np.float32).reshape(out.shape[-2], out.shape[-1])))


def cells_to_ref_mask(cells: np.ndarray, ref_shape=(360, 576)) -> np.ndarray:
    """(32, 80) bool → 基準の視点 (360, 576) bool(切り出した行の範囲へ最近傍で広げる)。"""
    import cv2
    H, W = ref_shape
    out = np.zeros((H, W), bool)
    y0, y1 = NET_ROWS
    out[y0:y1] = cv2.resize(cells.astype(np.uint8), (W, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    return out


def veto_flood(mask: np.ndarray, open_road: np.ndarray | None, frac: float = FLOOD_FRAC,
               min_w: float = FLOOD_MIN_W) -> np.ndarray:
    """物の多い絵(俵・柵・観客)で検出が地平線の帯を左右いっぱいに塗ったとき、平らな地面の上の塗りを行ごとに消す。

    open_road: 深度が「平らな地面」と言う画素。行の地面が幅の min_w 以上あり、その frac 以上を塗っている行だけ消す
    (道の上の低い物は行の地面の一部しか塗らないので残る)。"""
    if open_road is None or mask is None or open_road.shape != mask.shape:
        return mask
    n_open = open_road.sum(1)
    n_hit = (mask & open_road).sum(1)
    flood = (n_open >= max(1.0, min_w * mask.shape[1])) & (n_hit >= frac * np.maximum(n_open, 1))
    if not flood.any():
        return mask
    out = mask.copy()
    out[flood] &= ~open_road[flood]
    return out


class ObstacleTracker:
    """2 回続けて(pair_s 秒以内に)塗られたマスだけ残し、min_cells マス未満なら無しとする(persist=False なら 1 回でよい)。"""

    def __init__(self, pair_s: float = 0.5, min_cells: int = 2, persist: bool = True):
        self.pair_s, self.min_cells, self.persist = float(pair_s), int(min_cells), bool(persist)
        self.prev = None
        self.prev_t = None

    def update(self, cells: np.ndarray, t: float) -> np.ndarray:
        if not self.persist:
            both = cells
        elif self.prev is not None and self.prev_t is not None and 0 <= t - self.prev_t <= self.pair_s:
            both = cells & self.prev
        else:
            both = np.zeros_like(cells)
        self.prev, self.prev_t = cells.copy(), t
        if both.sum() < max(1, self.min_cells):
            return np.zeros_like(cells)
        return both
