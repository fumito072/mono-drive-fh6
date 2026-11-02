"""走行の記録(絵 + 位置・向き)から、PathNet の教師「実際に通った線」を作る(人手のラベルなし)。

データの形(1 走行 = 1 フォルダ):
    <フォルダ>/frames/000000.jpg …     絵(名前の順 = 時刻の順)
    <フォルダ>/poses.csv               1 枚 1 行: frame, t, x, y, z, yaw_deg[, pitch_deg, roll_deg]
                                       (または向きのベクトル fwd_x, fwd_y, fwd_z, right_x, right_y, right_z)
    <フォルダ>/camera.json(無くてもよい。無ければ --config のカメラ)

世界の座標は **右手系・z が上**(x 東・y 北など。原点はどこでもよい)、単位は m。
yaw_deg は +x 軸から反時計回り(上から見て左回り)が正。pitch_deg は機首上げが正、roll_deg は右側が下がる向きが正。
位置は車の基準点(camera.json の fwd_m・right_m を測った点)。GPS・オドメトリ・SLAM・シミュレータのどれでもよい。

教師: 各フレームの車の位置・向きから見た、その後に実際に通った点(前・右・上)を、道なりの距離 ARC_S × k の所で取り、
基準の世界(学習時のカメラ・車の大きさ)へ直したもの(ReferenceView.user_to_ref)。
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from .camera import REF_CAMERA, Camera
from .pathnet import ARC_S
from .sources import IMG_EXT, read_table

#: 教師にする先の長さ [m](ARC_S の最後 60m + 余り)・瞬間移動とみなす 1 歩の長さ [m]
FUTURE_M, JUMP_M = 65.0, 15.0
#: 基準の世界の、車の基準点からカメラまでの前の距離(教師の前の値はカメラの前から測ると比べやすい)
REF_FWD = REF_CAMERA.fwd_m


def vehicle_axes(yaw, pitch=None, roll=None):
    """角度 [rad] → 前・右・上の単位ベクトル (N, 3)。"""
    yaw = np.asarray(yaw, float)
    pitch = np.zeros_like(yaw) if pitch is None else np.asarray(pitch, float)
    roll = np.zeros_like(yaw) if roll is None else np.asarray(roll, float)
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    fwd = np.stack([cp * cy, cp * sy, sp], 1)
    right_h = np.stack([sy, -cy, np.zeros_like(sy)], 1)
    up_h = np.cross(right_h, fwd)
    right = np.cos(roll)[:, None] * right_h - np.sin(roll)[:, None] * up_h
    up = np.cross(right, fwd)
    return fwd, right, up


class Drive:
    """1 走行(絵と位置・向き)。"""

    def __init__(self, root: str, camera: Camera | None = None):
        self.root = root
        files = sorted(p for p in glob.glob(os.path.join(root, "frames", "*")) if p.lower().endswith(IMG_EXT))
        tb = read_table(os.path.join(root, "poses.csv"))
        n = len(tb.get("x", []))
        if not n:
            raise ValueError(f"{root}/poses.csv が読めない(列 x, y, z, yaw_deg が要る)")
        fr = tb["frame"].astype(int) if "frame" in tb else np.arange(n)
        if fr.max() >= len(files):
            raise ValueError(f"{root}: poses.csv の frame({fr.max()})に対して絵が {len(files)} 枚しかない")
        self.files = [files[i] for i in fr]
        self.t = tb["t"] if "t" in tb else np.arange(n) / 10.0
        self.pos = np.stack([tb["x"], tb["y"], tb.get("z", np.zeros(n))], 1)
        if all(k in tb for k in ("fwd_x", "fwd_y", "fwd_z", "right_x", "right_y", "right_z")):
            self.fwd = np.stack([tb["fwd_x"], tb["fwd_y"], tb["fwd_z"]], 1)
            self.right = np.stack([tb["right_x"], tb["right_y"], tb["right_z"]], 1)
            self.fwd /= np.linalg.norm(self.fwd, axis=1, keepdims=True)
            self.right /= np.linalg.norm(self.right, axis=1, keepdims=True)
            self.up = np.cross(self.right, self.fwd)
        else:
            if "yaw_deg" not in tb:
                raise ValueError(f"{root}/poses.csv に yaw_deg(または fwd_*・right_*)の列が要る")
            rad = np.radians
            self.fwd, self.right, self.up = vehicle_axes(rad(tb["yaw_deg"]), rad(tb.get("pitch_deg", np.zeros(n))),
                                                         rad(tb.get("roll_deg", np.zeros(n))))
        step = np.linalg.norm(np.diff(self.pos, axis=0), axis=1)
        self.jump = np.concatenate([[False], step > JUMP_M])
        step[step > JUMP_M] = 0.0
        self.odo = np.concatenate([[0.0], np.cumsum(step)])
        self.kmh = tb.get("kmh")
        cam_path = os.path.join(root, "camera.json")
        self.camera = Camera.load(cam_path) if os.path.exists(cam_path) else camera
        if self.camera is None:
            raise ValueError(f"{root}: camera.json も --config のカメラも無い")

    def __len__(self) -> int:
        return len(self.files)

    def future(self, i: int, max_m: float = FUTURE_M) -> np.ndarray:
        """i の後の、道なり max_m までの番号(瞬間移動の手前まで)。"""
        js = []
        for j in range(i + 1, len(self.files)):
            if self.jump[j] or self.odo[j] - self.odo[i] > max_m + 5.0:
                break
            js.append(j)
        return np.array(js, int)

    def to_car(self, i: int, js: np.ndarray) -> np.ndarray:
        d = self.pos[js] - self.pos[i]
        return np.stack([d @ self.fwd[i], d @ self.right[i], d @ self.up[i]], 1)

    def target(self, i: int, rv) -> tuple[np.ndarray, np.ndarray] | None:
        """フレーム i の教師 → (ARC_S の 11 点の (F, R, U) [基準の世界の m], 有り無し)。先が足りなければ None。"""
        js = self.future(i)
        if len(js) < 4:
            return None
        c = self.to_car(i, js)
        s = self.odo[js] - self.odo[i]
        su = ARC_S * rv.k                                    # 利用者の世界での道なりの距離
        out = np.zeros((len(ARC_S), 3), np.float32)
        ok = np.zeros(len(ARC_S), bool)
        for k, sv in enumerate(su):
            j = int(np.searchsorted(s, sv))
            if j == 0 or j >= len(s):
                continue
            w = 0.0 if s[j] <= s[j - 1] else float(np.clip((sv - s[j - 1]) / (s[j] - s[j - 1]), 0, 1))
            p = c[j - 1] + w * (c[j] - c[j - 1])
            fr, rr, ur = rv.user_to_ref(p[0], p[1], p[2])
            out[k] = [fr, rr, ur]
            ok[k] = True
        if ok.sum() < 3:
            return None
        # 下がっている・その場で回っている所は教師にしない(最初の点が前へ進んでいない)
        if ok[0] and out[0, 0] - REF_FWD < 0.3 * float(ARC_S[0]):
            return None
        return out, ok


def load_config_camera(config: str | None) -> Camera | None:
    if not config:
        return None
    with open(config, encoding="utf-8") as f:
        d = json.load(f)
    return Camera.from_dict(d.get("camera", d))
