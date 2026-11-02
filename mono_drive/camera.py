"""カメラのモデルと、「学習時の見え方(基準の視点)」への変換。

**車の座標**: 前 F・右 R・上 U [m]。原点は車の基準点(位置を測る点。後車軸の中心や GPS アンテナなど)の
真下の地面。カメラの取り付け位置(fwd_m・right_m・height_m)はこの基準点から測る。

**カメラ**: ピンホール(+ 必要なら OpenCV の歪み係数)。
  pitch_deg … 下向きが正 / yaw_deg … 右向きが正 / roll_deg … 絵が時計回りに回って見えるときが負(学習時 -0.25°)

**基準の視点(REF_CAMERA)**: 学習済みモデルが見てきた絵(ゲーム DiRT Rally 2.0 の Citroën C4 の車内視点、
576x360・縦の画角 58.2°・5° 下向き・高さ 1.1m・車の中心より 0.35m 左)。ほかのカメラの絵は、
カメラの中心が同じとみなして(回転だけの射影変換)この視点の絵へ作り直してからモデルに入れる。
高さの違いは「世界の大きさの違い」として扱う: 高さ h のカメラの絵は、世界を 1.1/h 倍にして高さ 1.1m の
カメラで見た絵と同じ。だからモデルの答え(m)は k = h / 1.1 倍して利用者の車の m に直す(ReferenceView)。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Camera:
    width: int = 1280
    height: int = 720
    #: 焦点距離 [画素]。0 なら hfov_deg / vfov_deg から決める
    fx: float = 0.0
    fy: float = 0.0
    #: 光軸の画素。None なら絵の真ん中(crop_top より下の真ん中)
    cx: float | None = None
    cy: float | None = None
    hfov_deg: float | None = None
    vfov_deg: float | None = None
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    #: 取り付け位置(車の基準点から)
    height_m: float = 1.2
    fwd_m: float = 0.0
    right_m: float = 0.0
    #: 上から使わない行(ウィンドウのタイトルバーなど)
    crop_top: int = 0
    #: 歪み: "pinhole"(dist = [k1, k2, p1, p2, k3])か "fisheye"(dist = [k1, k2, k3, k4])
    model: str = "pinhole"
    dist: list | None = None
    #: 車体(ボンネット・ダッシュボード)の上端の線 [[u, v], ...](元の絵の画素)。これより下は車体として使わない
    hood: list | None = None
    name: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.width, self.height = int(self.width), int(self.height)
        h_img = self.height - self.crop_top
        if not self.fx and not self.fy:
            if self.hfov_deg:
                self.fx = 0.5 * self.width / math.tan(math.radians(self.hfov_deg) / 2)
            elif self.vfov_deg:
                self.fy = 0.5 * h_img / math.tan(math.radians(self.vfov_deg) / 2)
            else:
                raise ValueError("カメラに fx/fy か hfov_deg/vfov_deg のどれかが要る")
        self.fx = float(self.fx or self.fy)
        self.fy = float(self.fy or self.fx)
        if self.cx is None:
            self.cx = self.width / 2.0
        if self.cy is None:
            self.cy = self.crop_top + h_img / 2.0
        self.cx, self.cy = float(self.cx), float(self.cy)

    # --- 読み書き ---------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        keys = set(cls.__dataclass_fields__)
        known = {k: v for k, v in d.items() if k in keys}
        extra = {k: v for k, v in d.items() if k not in keys and not k.startswith("_")}
        if extra:
            known.setdefault("extra", {}).update(extra)
        return cls(**known)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}

    @classmethod
    def load(cls, path: str) -> "Camera":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d.get("camera", d))

    def scaled(self, width: int, height: int) -> "Camera":
        """同じカメラを別の大きさの絵に合わせる(画素の値を縦横の比で直す)。"""
        sx, sy = width / float(self.width), height / float(self.height)
        d = asdict(self)
        d.update(width=int(width), height=int(height), fx=self.fx * sx, fy=self.fy * sy,
                 cx=self.cx * sx, cy=self.cy * sy, crop_top=int(round(self.crop_top * sy)),
                 hfov_deg=None, vfov_deg=None,
                 hood=[[u * sx, v * sy] for u, v in self.hood] if self.hood else None)
        return Camera(**d)

    # --- 幾何 -------------------------------------------------------------
    def rotation(self) -> np.ndarray:
        """車の向き(右・上・前)→ カメラの向き(右・上・前)の 3x3。ヨー → ピッチ → ロールの順に回す。"""
        a, p, r = (math.radians(x) for x in (self.yaw_deg, self.pitch_deg, self.roll_deg))
        Ry = np.array([[math.cos(a), 0.0, -math.sin(a)], [0.0, 1.0, 0.0], [math.sin(a), 0.0, math.cos(a)]])
        Rp = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(p), math.sin(p)], [0.0, -math.sin(p), math.cos(p)]])
        Rr = np.array([[math.cos(r), math.sin(r), 0.0], [-math.sin(r), math.cos(r), 0.0], [0.0, 0.0, 1.0]])
        return Rr @ Rp @ Ry

    def K_cv(self) -> np.ndarray:
        """OpenCV の形のカメラ行列(v は下向き)。"""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def to_camera(self, F, R, U):
        """車の座標 → カメラ座標 (x 右, y 上, z 前)。"""
        v = np.stack([np.asarray(R, float) - self.right_m,
                      np.asarray(U, float) - self.height_m,
                      np.asarray(F, float) - self.fwd_m], 0)
        c = np.tensordot(self.rotation(), v, axes=1)
        return c[0], c[1], c[2]

    def project(self, F, R, U, distort: bool = True):
        """車の座標 → 画素 (u, v) と奥行き z [m]。カメラの後ろ(z <= 0)は NaN。"""
        x, y, z = self.to_camera(F, R, U)
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = np.where(z > 1e-6, x / z, np.nan)
            yn = np.where(z > 1e-6, -y / z, np.nan)       # OpenCV の向き(下が正)
        xd, yd = self._distort(xn, yn) if distort else (xn, yn)
        return self.cx + self.fx * xd, self.cy + self.fy * yd, z

    def ground_to_image(self, F, R):
        F = np.asarray(F, float)
        u, v, _ = self.project(F, np.asarray(R, float), np.zeros_like(F))
        return u, v

    def ray(self, u, v):
        """画素 → 車の座標での視線の向き (前, 右, 上)(長さは不定)。"""
        xn, yn = self._undistort((np.asarray(u, float) - self.cx) / self.fx,
                                 (np.asarray(v, float) - self.cy) / self.fy)
        d = np.stack([xn, -yn, np.ones_like(xn)], 0)
        w = np.tensordot(self.rotation().T, d, axes=1)
        return w[2], w[0], w[1]

    def image_to_ground(self, u, v):
        """画素 → 平らな地面の上の (前 m, 右 m)。地平線より上は NaN。"""
        f, r, up = self.ray(u, v)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(up < -1e-6, self.height_m / -up, np.nan)
        return self.fwd_m + t * f, self.right_m + t * r

    @property
    def horizon_v(self) -> float:
        """真ん中の列での平らな地面の地平線の行。"""
        f, r = 1e6, 0.0
        _, v, _ = self.project(np.array([f]), np.array([r]), np.array([0.0]))
        return float(v[0])

    def hood_mask(self) -> np.ndarray | None:
        """(height, width) bool: 車体の画素(hood の線より下)。"""
        if not self.hood:
            return None
        pts = np.asarray(self.hood, float)
        pts = pts[np.argsort(pts[:, 0])]
        top = np.interp(np.arange(self.width), pts[:, 0], pts[:, 1])
        return np.arange(self.height)[:, None] >= top[None, :]

    # --- 歪み -------------------------------------------------------------
    def _has_dist(self) -> bool:
        return bool(self.dist) and any(abs(float(x)) > 0 for x in self.dist)

    def _distort(self, xn, yn):
        if not self._has_dist():
            return xn, yn
        import cv2
        shp = np.shape(xn)
        pts = np.stack([np.ravel(xn), np.ravel(yn), np.ones(np.size(xn))], 1).astype(np.float64)
        bad = ~np.isfinite(pts).all(1)
        pts[bad] = [0.0, 0.0, 1.0]
        D = np.asarray(self.dist, np.float64)
        I = np.eye(3)
        if self.model == "fisheye":
            out = cv2.fisheye.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), I, D[:4])[0]
        else:
            out = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), I, D)[0]
        out = out.reshape(-1, 2)
        out[bad] = np.nan
        return out[:, 0].reshape(shp), out[:, 1].reshape(shp)

    def _undistort(self, xd, yd):
        if not self._has_dist():
            return xd, yd
        import cv2
        shp = np.shape(xd)
        pts = np.stack([np.ravel(xd), np.ravel(yd)], 1).astype(np.float64).reshape(-1, 1, 2)
        D = np.asarray(self.dist, np.float64)
        I = np.eye(3)
        if self.model == "fisheye":
            out = cv2.fisheye.undistortPoints(pts, I, D[:4])
        else:
            out = cv2.undistortPoints(pts, I, D)
        out = out.reshape(-1, 2)
        return out[:, 0].reshape(shp), out[:, 1].reshape(shp)


def _ref_camera() -> Camera:
    # 学習時のカメラ(ゲームの絵 576x360。保存した絵 576x374 の上 14 行はウィンドウのタイトルバー)
    return Camera(width=576, height=360, vfov_deg=58.2, pitch_deg=5.0, yaw_deg=0.0, roll_deg=-0.25,
                  height_m=1.1, fwd_m=-0.2, right_m=-0.35, crop_top=0,
                  name="基準の視点(DiRT Rally 2.0・Citroën C4 Rally の車内視点)")


#: 学習済みモデルが見てきた視点
REF_CAMERA = _ref_camera()
#: モデルに入れる行(基準の視点の 0〜240 行 = 地平線のまわりと手前の道。下はダッシュボード)
NET_ROWS = (0, 240)


class ReferenceView:
    """利用者のカメラの絵 → 基準の視点の絵(576x360)。座標の換算もここ。

        rv = ReferenceView(camera)            # 絵の大きさが変わったら作り直す
        ref = rv.warp(frame_bgr)              # (360, 576, 3)
        rv.valid                              # (360, 576) bool: 元の絵から来た画素(外・車体は False)
        F, R, U = rv.ref_to_user(F, R, U)     # モデルの答え(基準の世界の m)→ 利用者の車の m
    """

    def __init__(self, cam: Camera, ref: Camera = REF_CAMERA, frame_size: tuple[int, int] | None = None):
        import cv2
        if frame_size is not None and (frame_size[0], frame_size[1]) != (cam.width, cam.height):
            cam = cam.scaled(*frame_size)
        self.cam, self.ref = cam, ref
        self.k = float(cam.height_m) / float(ref.height_m)
        # 元の絵が基準よりずっと細かいときは、先に面積平均で基準の細かさまで縮める(学習時の縮め方に合わせ、
        # ちらつきを防ぐ。ゲームの 1152x748 はちょうど半分 = 学習に使った保存画像と同じ絵になる)
        s = ref.fy / max(1e-6, min(cam.fx, cam.fy))
        s = s if s < 0.87 else 1.0
        self.pre = s
        self.cam_s = cam if s >= 0.999 else cam.scaled(max(1, int(round(cam.width * s))),
                                                       max(1, int(round(cam.height * s))))
        cs = self.cam_s
        H, W = ref.height, ref.width
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
        f, r, up = ref.ray(uu, vv)
        # 同じ向きの視線が、利用者のカメラのどの画素に写るか(カメラの中心は同じとみなす)
        big = 1e3
        u_u, v_u, z = cs.project(cs.fwd_m + big * f, cs.right_m + big * r, cs.height_m + big * up)
        ok = np.isfinite(u_u) & np.isfinite(v_u) & (z > 0)
        ok &= (u_u >= 0) & (u_u <= cs.width - 1) & (v_u >= cs.crop_top) & (v_u <= cs.height - 1)
        #: 元の絵の中から来た画素 / そのうち車体(hood)でない画素
        self.inside = ok.copy()
        hm = cs.hood_mask()
        if hm is not None:
            ui = np.clip(np.nan_to_num(u_u, nan=0).round().astype(int), 0, cs.width - 1)
            vi = np.clip(np.nan_to_num(v_u, nan=0).round().astype(int), 0, cs.height - 1)
            ok &= ~hm[vi, ui]
        self.valid = ok
        self.map_x = np.where(ok, u_u, -1).astype(np.float32)
        self.map_y = np.where(ok, v_u, -1).astype(np.float32)
        # ただ切り出すだけで済むか(学習時と同じカメラ: 上のタイトルバーを除くだけ)
        dx = np.nan_to_num(u_u - uu, nan=1e9)
        dy = np.nan_to_num(v_u - vv, nan=1e9)
        self.slice = None
        if (cs.width == W and np.allclose(dx, 0, atol=1e-3) and np.allclose(dy, dy[0, 0], atol=1e-3)
                and abs(dy[0, 0] - round(dy[0, 0])) < 1e-3 and round(dy[0, 0]) >= 0
                and round(dy[0, 0]) + H <= cs.height):
            self.slice = int(round(dy[0, 0]))
        #: モデルが見る範囲のうち、元の絵で埋まっている割合(車体は数えない。1.0 が望ましい)
        self.coverage = float(self.inside[NET_ROWS[0]:NET_ROWS[1]].mean())
        self._inv = None
        self._cv2 = cv2

    # --- 絵 ---------------------------------------------------------------
    def warp(self, frame: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        if self.pre < 0.999:
            frame = cv2.resize(frame, (self.cam_s.width, self.cam_s.height), interpolation=cv2.INTER_AREA)
        if self.slice is not None:
            return np.ascontiguousarray(frame[self.slice:self.slice + self.ref.height, :self.ref.width])
        return cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def ref_px_to_user_px(self, u, v):
        """基準の視点の画素 → 元の絵の画素(描画用)。"""
        f, r, up = self.ref.ray(np.asarray(u, float), np.asarray(v, float))
        c = self.cam
        big = 1e3
        uu, vv, z = c.project(c.fwd_m + big * f, c.right_m + big * r, c.height_m + big * up)
        bad = ~(z > 0)
        return np.where(bad, np.nan, uu), np.where(bad, np.nan, vv)

    def ref_mask_to_user(self, mask: np.ndarray) -> np.ndarray:
        """基準の視点の (360, 576) の値 → 元の絵の大きさへ(描画用。最近傍)。"""
        cv2 = self._cv2
        if self._inv is None:
            c = self.cam
            vv, uu = np.mgrid[0:c.height, 0:c.width].astype(np.float64)
            f, r, up = c.ray(uu, vv)
            ref = self.ref
            big = 1e3
            ur, vr, z = ref.project(ref.fwd_m + big * f, ref.right_m + big * r, ref.height_m + big * up)
            ok = np.isfinite(ur) & np.isfinite(vr) & (z > 0)
            self._inv = (np.where(ok, ur, -1).astype(np.float32), np.where(ok, vr, -1).astype(np.float32))
        m = mask.astype(np.uint8) if mask.dtype == bool else mask
        out = cv2.remap(m, self._inv[0], self._inv[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0)
        return out.astype(bool) if mask.dtype == bool else out

    # --- 座標 -------------------------------------------------------------
    def ref_to_user(self, F, R, U=None):
        """基準の世界(モデルの答え)の車の座標 → 利用者の車の座標 [m]。"""
        k, c, r = self.k, self.cam, self.ref
        Fu = (np.asarray(F, float) - r.fwd_m) * k + c.fwd_m
        Ru = (np.asarray(R, float) - r.right_m) * k + c.right_m
        if U is None:
            return Fu, Ru
        return Fu, Ru, np.asarray(U, float) * k

    def user_to_ref(self, F, R, U=None):
        k, c, r = self.k, self.cam, self.ref
        Fr = (np.asarray(F, float) - c.fwd_m) / k + r.fwd_m
        Rr = (np.asarray(R, float) - c.right_m) / k + r.right_m
        if U is None:
            return Fr, Rr
        return Fr, Rr, np.asarray(U, float) / k
