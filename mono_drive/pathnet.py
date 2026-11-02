"""PathNet: 絵から「この先に車が通る線」を直接当てる小さな CNN(学習済み ONNX を読む)。

入力: 基準の視点(576x360)の 0〜240 行を 320x128 に縮めた RGB(0..1)。
出力(44 個): 道なりに ARC_S [m] 進んだ 11 か所の 前 F[11]・右 R[11]・上 U[11] と、横位置の log 分散[11]。
座標は基準の世界(学習時のカメラ・車)。利用者の車の m へは ReferenceView.ref_to_user で直す。

教師は人手のラベルではなく、走行の記録(位置・向き)から作った「実際に通った線」(tools/train_pathnet.py)。
自信(confidence)は 8〜22m 先の横の標準偏差から: 0.6m 以下で 1、2.5m 以上で 0。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .camera import NET_ROWS
from .runtime import make_session

ARC_S = np.array([4, 8, 12, 16, 20, 25, 30, 35, 40, 50, 60], np.float32)
NET_W, NET_H = 320, 128


def net_input_u8(ref_bgr: np.ndarray) -> np.ndarray:
    """Exact byte-valued training input, avoiding a float-to-byte round trip."""
    import cv2
    x=cv2.resize(ref_bgr[NET_ROWS[0]:NET_ROWS[1]],(NET_W,NET_H),interpolation=cv2.INTER_AREA)
    x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(np.transpose(x,(2,0,1)))


def net_input(ref_bgr: np.ndarray) -> np.ndarray:
    """基準の視点の絵(BGR)→ (3, 128, 320) float32 の RGB 0..1(PathNet と障害物の検出で共通)。"""
    import cv2
    x = cv2.resize(ref_bgr[NET_ROWS[0]:NET_ROWS[1]], (NET_W, NET_H), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1)))


def curvature_profile(F: np.ndarray, R: np.ndarray, S: np.ndarray):
    """点列 → 各所の曲率 [1/m](右が正)と、その道なりの距離。予測した点どうしの向きの変化から出す。"""
    F, R, S = (np.asarray(a, float) for a in (F, R, S))
    dF, dR = np.diff(F), np.diff(R)
    ds = np.maximum(0.5, np.hypot(dF, dR))
    th = np.unwrap(np.arctan2(dR, dF))
    k = np.diff(th) / (0.5 * (ds[:-1] + ds[1:]))
    return k, S[1:-1]


@dataclass
class PathPrediction:
    """当てた道(基準の世界の m)。"""
    F: np.ndarray
    R: np.ndarray
    U: np.ndarray
    sd: np.ndarray          # 横位置の標準偏差 [m](sd_scale を掛けた後)
    S: np.ndarray           # 道なりの距離 [m]
    confidence: float       # 0..1


class PathNet:
    SIGMA_GOOD, SIGMA_BAD = 0.6, 2.5

    def __init__(self, onnx_path: str, device: str | None = None):
        self.sess = make_session(onnx_path, device, gpu_ok=False)
        self.inp = self.sess.get_inputs()[0].name
        n_out = self.sess.get_outputs()[0].shape[-1]
        if n_out != 4 * len(ARC_S):
            raise ValueError(f"{onnx_path}: 出力が {n_out} 個(44 個の arc 形式のモデルが要る)")
        self.meta = {}
        try:
            with open(os.path.splitext(onnx_path)[0] + ".json", encoding="utf-8") as f:
                self.meta = json.load(f)
        except (OSError, ValueError):
            pass
        #: 自信の物差し(モデルごとに検証で合わせた値)。標準偏差にこれを掛ける
        v = float(self.meta.get("sd_scale", 1.0) or 1.0)
        self.sd_scale = v if 0.2 <= v <= 5.0 else 1.0

    def predict(self, x: np.ndarray) -> PathPrediction:
        """net_input の出力 (3, 128, 320) → PathPrediction。"""
        out = self.sess.run(None, {self.inp: x[None].astype(np.float32)})[0][0]
        n = len(ARC_S)
        F, R, U, lv = out[:n], out[n:2 * n], out[2 * n:3 * n], out[3 * n:]
        sd = np.exp(0.5 * np.clip(lv, -8, 6)) * self.sd_scale
        mid = (ARC_S >= 8) & (ARC_S <= 22)
        s = float(np.median(sd[mid]))
        conf = float(np.clip((self.SIGMA_BAD - s) / (self.SIGMA_BAD - self.SIGMA_GOOD), 0, 1))
        return PathPrediction(F=np.asarray(F, np.float64), R=np.asarray(R, np.float64),
                              U=np.asarray(U, np.float64), sd=np.asarray(sd, np.float64),
                              S=ARC_S.astype(np.float64), confidence=conf)
