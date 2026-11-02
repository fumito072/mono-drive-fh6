"""塗り(障害物・壁・落ち込み)→ **足元だけ**(地面に接している所から奥へ depth_m まで)。

絵の上で塗られた物は、背が高いほど、手前の地面の画素まで覆う(遠くの俵の上半分は、5m 先の地面と同じ行に写る)。
塗りをそのまま地面に写すと、空いている手前の道まで「塞がれた」と読む(俵のシケインで抜け道なしになった)。
そこで、縦の塗りの続き(列ごと)のいちばん下 = 地面に接した所を物の位置とし、そこから奥へ depth_m だけを塞ぎにする。
"""

from __future__ import annotations

import numpy as np

_FMAP: dict = {}


def ground_forward_map(cam) -> np.ndarray:
    """各画素の、平らな地面までの前の距離 [m](地平線より上は NaN)。"""
    key = (cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, cam.pitch_deg, cam.roll_deg, cam.yaw_deg,
           cam.height_m, cam.fwd_m, cam.right_m)
    if key not in _FMAP:
        vv, uu = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
        f, _ = cam.image_to_ground(uu, vv)
        _FMAP[key] = f
    return _FMAP[key]


def footprint(mask: np.ndarray, fmap: np.ndarray, depth_m: float = 1.5) -> np.ndarray:
    """(H, W) bool の塗り → 足元の (H, W) bool。"""
    if mask is None or not mask.any():
        return mask
    H, W = mask.shape
    foot = np.full((H, W), -1, np.int32)
    cur = np.full(W, -1, np.int32)
    for v in range(H - 1, -1, -1):
        m = mask[v]
        cur = np.where(m, np.where(cur >= 0, cur, v), -1)
        foot[v] = cur
    fv = np.clip(foot, 0, H - 1)
    f_foot = np.where(foot >= 0, fmap[fv, np.arange(W)[None, :]], np.nan)
    with np.errstate(invalid="ignore"):
        keep = mask & np.isfinite(f_foot) & np.isfinite(fmap) & (fmap - f_foot <= depth_m)
    return keep
