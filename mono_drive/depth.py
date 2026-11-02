"""深度で「平らな地面」と「地面から立ち上がった物(壁・岩・土手・柵)」「落ち込み(崖の縁・谷)」を見分ける。

モデルは Depth Anything V2 Small(Apache-2.0、ONNX・入力 266x420)。実写で学習されたモデルなので、
実カメラでもそのまま使える(このキットでいちばん実写へ持っていきやすい部分)。

**やり方** — 逆深度(近いほど大)は、平らな地面の上では画素の位置 (u, v) の一次式になる。車体のすぐ上(車体が
写らないカメラなら絵の下端)の帯で平面を当てはめ(RANSAC)、平面との比が ±15% 以内を地面、平面より 15% 以上
手前に出ている所を「立ち上がった物」、45% 以上遠い所を「落ち込み」とする。深度の尺度が分からなくても使える。
"""

from __future__ import annotations

import numpy as np

from .runtime import make_session

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class DepthGround:
    GROUND_TOL, RISE_TOL = 0.15, 0.15
    BAND_FRAC = 0.12
    NEAR_FRAC = 0.12
    DROP_TOL = 0.45

    def __init__(self, onnx_path: str, device: str | None = None):
        self.sess = make_session(onnx_path, device, gpu_ok=True)
        self.inp = self.sess.get_inputs()[0].name
        shp = self.sess.get_inputs()[0].shape
        self.H, self.W = int(shp[2]), int(shp[3])
        self.providers = self.sess.get_providers()
        self.last_disp = None
        self.ground = self.wall = self.drop = None
        self.ok = False
        self._rng = np.random.default_rng(0)

    def disparity(self, img_bgr: np.ndarray) -> np.ndarray:
        import cv2
        x = cv2.resize(img_bgr, (self.W, self.H), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        return self.sess.run(None, {self.inp: x.astype(np.float32)})[0][0].astype(np.float32)

    def predict(self, img_bgr: np.ndarray, body_mask: np.ndarray | None = None) -> bool:
        """絵(基準の視点)→ self.ground / self.wall / self.drop((h, w) bool)。平面が当てはまれば True。

        body_mask: 車体・絵の外の画素 True(当てはめに使わない)。"""
        import cv2
        h, w = img_bgr.shape[:2]
        d = self.disparity(img_bgr)
        H, W = d.shape
        if body_mask is not None and body_mask.shape[:2] == (h, w) and body_mask.any():
            body = cv2.resize(body_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        else:
            body = np.zeros((H, W), bool)
        # 各列で車体(または絵の下端)が始まる行
        col_top = np.where(body.any(0), body.argmax(0), H)
        vv, uu = np.mgrid[0:H, 0:W]
        band = (vv < col_top[None, :] - 1) & (vv >= col_top[None, :] - int(self.BAND_FRAC * H))
        band &= (uu > W * 0.08) & (uu < W * 0.92) & ~body
        ys, xs, zs = vv[band].astype(np.float32), uu[band].astype(np.float32), d[band]
        ground = np.zeros((h, w), bool)
        wall = np.zeros((h, w), bool)
        drop = np.zeros((h, w), bool)
        self.ok = False
        if len(zs) >= 50:
            if len(zs) > 1500:
                k = self._rng.choice(len(zs), 1500, replace=False)
                ys, xs, zs = ys[k], xs[k], zs[k]
            A = np.stack([ys, xs, np.ones_like(ys)], 1)
            scale = max(1e-6, float(np.percentile(zs, 90)))
            best = None
            for _ in range(40):                                  # RANSAC
                k = self._rng.choice(len(zs), 3, replace=False)
                try:
                    coef = np.linalg.solve(A[k], zs[k])
                except np.linalg.LinAlgError:
                    continue
                inl = np.abs(A @ coef - zs) < 0.05 * scale
                if best is None or inl.sum() > best[0]:
                    best = (int(inl.sum()), inl)
            if best is not None and best[0] >= 0.5 * len(zs):
                coef, *_ = np.linalg.lstsq(A[best[1]], zs[best[1]], rcond=None)
                # 奥(上の行)ほど地面の逆深度は小さいはず。当てはめた面の地平線が絵の 15〜60% に無ければ、
                # 地面ではない物(目の前の壁など)に当てはめている → この絵の深度は使わない
                v_hz = -(coef[1] * (W / 2) + coef[2]) / coef[0] if coef[0] > 1e-9 else -1.0
                if coef[0] > 0 and 0.15 * H <= v_hz <= 0.60 * H:
                    plane = coef[0] * vv + coef[1] * uu + coef[2]
                    valid = plane > 0.02 * scale
                    rel = (d - plane) / np.maximum(1e-6, np.abs(plane))
                    g_small = valid & (np.abs(rel) < self.GROUND_TOL) & ~body
                    near = plane > self.NEAR_FRAC * scale
                    w_small = near & (rel > self.RISE_TOL) & ~body & (vv < col_top[None, :])
                    d_small = near & (rel < -self.DROP_TOL) & ~body & (vv < col_top[None, :])
                    rs = lambda m: cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                    ground, wall, drop = rs(g_small), rs(w_small), rs(d_small)
                    self.ok = True
        self.last_disp = d
        self.ground, self.wall, self.drop = ground, wall, drop
        return self.ok
