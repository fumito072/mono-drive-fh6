"""絵 1 枚 → 走る線と速さ(Decision)。認識(PathNet・障害物・深度・矢印板)と計画(planner.py)をつなぐ。

    from mono_drive import MonoDriver
    drv = MonoDriver("configs/sample_c4.json")
    dec = drv.step(frame_bgr, speed_kmh=40.0)          # t を渡さなければ今の時刻
    dec.steer, dec.steer_deg, dec.kappa, dec.target_kmh, dec.reverse
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from .camera import REF_CAMERA, Camera, ReferenceView
from .chevrons import ChevronTracker, find_chevrons
from .freespace import footprint, ground_forward_map
from .obstacle import ObstacleNet, ObstacleTracker, cells_to_ref_mask, veto_flood
from .pathnet import PathNet, net_input
from .planner import Decision, Planner, PlannerConfig, VehicleConfig, arc_points

KIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path: str | None, base: str) -> str | None:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    for root in (base, KIT_ROOT, os.getcwd()):
        p = os.path.join(root, path)
        if os.path.exists(p):
            return p
    return os.path.join(KIT_ROOT, path)


def load_config(config: dict | str) -> tuple[dict, str]:
    if isinstance(config, dict):
        return config, KIT_ROOT
    with open(config, encoding="utf-8") as f:
        return json.load(f), os.path.dirname(os.path.abspath(config))


class _PairMask:
    """深度の塞ぎも「2 回続けて見えた所だけ」(1 枚だけの外れ・雨粒で曲がらない)。"""

    def __init__(self, pair_s: float = 0.6):
        self.pair_s = pair_s
        self.prev, self.prev_t = None, None

    def update(self, m: np.ndarray | None, t: float) -> np.ndarray | None:
        if m is None:
            return None
        out = None
        if self.prev is not None and t - self.prev_t <= self.pair_s and self.prev.shape == m.shape:
            out = m & self.prev
        self.prev, self.prev_t = m.copy(), t
        return out


class MonoDriver:
    def __init__(self, config: dict | str, device: str | None = None, use_depth: bool | None = None,
                 use_obstacle: bool | None = None, use_chevrons: bool | None = None):
        cfg, base = load_config(config)
        self.cfg = cfg
        self.camera = Camera.from_dict(cfg["camera"])
        self.vehicle = VehicleConfig.from_dict(cfg.get("vehicle"))
        self.pcfg = PlannerConfig.from_dict(cfg.get("planner"))
        self.planner = Planner(self.vehicle, self.pcfg)
        m = cfg.get("models", {})
        self.opts = {
            "obstacle_every": int(m.get("obstacle_every", 1)),
            "depth_every": int(m.get("depth_every", 2)),
            "depth_hold_s": float(m.get("depth_hold_s", 0.6)),
            "chevron_every": int(m.get("chevron_every", 1)),
        }
        self.pathnet = PathNet(_resolve(m.get("pathnet", "models/pathnet.onnx"), base), device)
        self.obstacle = None
        p_obs = _resolve(m.get("obstacle", "models/obstacle.onnx"), base)
        if (use_obstacle if use_obstacle is not None else m.get("use_obstacle", True)) and p_obs and os.path.exists(p_obs):
            self.obstacle = ObstacleNet(p_obs, m.get("obstacle_thr"), device)
        self.depth = None
        p_dep = _resolve(m.get("depth", "models/depth_anything_v2_small_266x420.onnx"), base)
        if (use_depth if use_depth is not None else m.get("use_depth", True)) and p_dep and os.path.exists(p_dep):
            from .depth import DepthGround
            self.depth = DepthGround(p_dep, device)
        self.use_chevrons = use_chevrons if use_chevrons is not None else bool(m.get("use_chevrons", True))
        #: 足元から奥へ塞ぎにする長さ [m](基準の世界)。use_footprint=false なら塗り全体を塞ぎにする(比べる用)
        self.foot_depth_m = float(m.get("foot_depth_m", 1.5))
        self.use_footprint = bool(m.get("use_footprint", True))
        self.obs_track = ObstacleTracker(pair_s=float(m.get("obstacle_pair_s", 0.5)),
                                         min_cells=int(m.get("obstacle_min_cells", 2)))
        self.wall_pair, self.drop_pair = _PairMask(), _PairMask()
        self.chev = ChevronTracker()
        self.rv: ReferenceView | None = None
        self.n = 0
        self.last: dict = {}
        self._depth_res = None
        self._cue = None                # 矢印板の位置(車の座標 m)。読めない間は自分の動きで動かす
        self._t_prev = None
        self._cells = None
        self._cues: list = []
        self.ms: dict = {}

    # ------------------------------------------------------------------
    def view(self, shape) -> ReferenceView:
        h, w = shape[:2]
        if self.rv is None or (self.rv.cam.width, self.rv.cam.height) != (w, h):
            self.rv = ReferenceView(self.camera, REF_CAMERA, frame_size=(w, h))
        return self.rv

    def _tick(self, name: str, t0: float) -> None:
        ms = (time.perf_counter() - t0) * 1000
        self.ms[name] = ms if name not in self.ms else 0.9 * self.ms[name] + 0.1 * ms

    def perceive(self, frame: np.ndarray, t: float) -> dict:
        """認識だけ(基準の視点の絵・当てた道・障害物の塗り・矢印板)。"""
        rv = self.view(frame.shape)
        t0 = time.perf_counter()
        ref = rv.warp(frame)
        x = net_input(ref)
        self._tick("warp", t0)
        t0 = time.perf_counter()
        pred = self.pathnet.predict(x)
        self._tick("pathnet", t0)
        body = ~rv.valid
        out = {"ref": ref, "pred": pred, "body": body}
        # 学習した障害物の検出(2 回続けて塗られたマスだけ)
        if self.obstacle is not None:
            if self._cells is None or self.n % max(1, self.opts["obstacle_every"]) == 0:
                t0 = time.perf_counter()
                p = self.obstacle.prob(x)
                cells = self.obs_track.update(p > self.obstacle.thr, t)
                self._cells = (cells, p)
                self._tick("obstacle", t0)
            cells, p = self._cells
            out["obstacle_prob"] = p
            out["obstacle"] = cells_to_ref_mask(cells, ref.shape[:2]) & rv.valid
        # 深度(数枚に 1 回・少しの間は前の結果を使う)
        if self.depth is not None:
            if self.n % max(1, self.opts["depth_every"]) == 0:
                t0 = time.perf_counter()
                ok = self.depth.predict(ref, body_mask=body)
                self._tick("depth", t0)
                wall = self.wall_pair.update(self.depth.wall if ok else None, t)
                drop = self.drop_pair.update(self.depth.drop if ok else None, t)
                self._depth_res = (t, ok, self.depth.ground if ok else None, wall, drop)
            if self._depth_res is not None and t - self._depth_res[0] <= self.opts["depth_hold_s"]:
                _, ok, ground, wall, drop = self._depth_res
                if ok:
                    out["ground"], out["wall"], out["drop"] = ground, wall, drop
                    if "obstacle" in out:
                        out["obstacle"] = veto_flood(out["obstacle"], ground)
        # 塞ぎは足元だけ(背の高い物が手前の地面の画素を覆うのを、塞がれたと読まない)
        fmap = ground_forward_map(rv.ref)
        for k in ("obstacle", "wall", "drop"):
            if out.get(k) is not None:
                out[k + "_fp"] = footprint(out[k], fmap, self.foot_depth_m) if self.use_footprint else out[k]
        # 矢印板
        if self.use_chevrons:
            if self.n % max(1, self.opts["chevron_every"]) == 0:
                t0 = time.perf_counter()
                self._cues = find_chevrons(ref, body_mask=body)
                self._tick("chevrons", t0)
                out["cue"] = self.chev.update(self._cues, t)
            else:                                           # 読まない枚: 覚えている向きだけ(古くなれば忘れる)
                a = self.chev.active
                if a is not None and t - a["t"] > self.chev.hold_s:
                    self.chev.active = None
                out["cue"] = self.chev.active
            out["cues"] = self._cues
        return out

    def _block_fn(self, rv: ReferenceView, masks: dict):
        masks = {k: masks.get(k + "_fp") for k in ("obstacle", "wall", "drop")}
        names = [k for k in ("obstacle", "wall", "drop") if masks.get(k) is not None and masks[k].any()]
        if not names:
            return None
        ref = rv.ref
        H, W = ref.height, ref.width
        half = self.vehicle.width_m / 2 + self.pcfg.margin_m
        offs = np.linspace(-half, half, 7)
        valid = rv.valid
        frac_min = self.pcfg.block_frac

        def fn(kappa: float, ds: np.ndarray):
            F, R, th = arc_points(kappa, ds)
            nF, nR = -np.sin(th), np.cos(th)
            Fs = F[:, None] + offs[None, :] * nF[:, None]
            Rs = R[:, None] + offs[None, :] * nR[:, None]
            Fr, Rr = rv.user_to_ref(Fs, Rs)
            u, v, z = ref.project(Fr, Rr, np.zeros_like(Fr), distort=False)
            ok = np.isfinite(u) & np.isfinite(v) & (z > 0.5)
            ui = np.where(ok, np.round(np.nan_to_num(u, nan=-1.0)), -1).astype(int)
            vi = np.where(ok, np.round(np.nan_to_num(v, nan=-1.0)), -1).astype(int)
            inside = ok & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            uc, vc = np.clip(ui, 0, W - 1), np.clip(vi, 0, H - 1)
            inside &= valid[vc, uc]
            n_in = inside.sum(1)
            hits = {k: inside & masks[k][vc, uc] for k in names}
            anyhit = np.zeros_like(inside)
            for h in hits.values():
                anyhit |= h
            frac = anyhit.sum(1) / np.maximum(1, n_in)
            blk = (n_in >= 3) & (frac >= frac_min)
            # 2 つ続けて塞がれた所から(1 点だけの塗りでは曲がらない)
            blocked = blk & np.concatenate([blk[1:], [False]])
            if not blocked.any():
                return None, ""
            i = int(np.argmax(blocked))
            kind = max(names, key=lambda k: int(hits[k][i].sum()))
            return float(ds[i]), kind

        return fn

    def _cue_update(self, act: dict | None, t: float, kmh: float, rv: ReferenceView,
                    yaw_rate_dps: float | None = None) -> dict | None:
        """矢印板の位置。読めた枚は絵から(板の下端を地面へ)、読めない枚は自分の動きで動かす
        (ヨーレートが分かればそれ、無ければ直前に出した曲率で曲がったとみなす)。"""
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        self._t_prev = t
        if act is None:
            self._cue = None
            return None
        if act["t"] == t:                                   # この枚で読めた
            x0, y0, x1, y1 = act["box"]
            fr, rr = rv.ref.image_to_ground(np.array([(x0 + x1) / 2.0]), np.array([float(y1)]))
            fu, ru = rv.ref_to_user(fr, rr)
            if np.isfinite(fu[0]) and 0.0 < float(fu[0]) < 80.0 * rv.k:
                self._cue = {"dir": act["dir"], "F": float(fu[0]), "R": float(ru[0])}
            else:
                self._cue = {"dir": act["dir"], "F": None, "R": None}
            return dict(self._cue)
        c = self._cue
        if c is None or c.get("F") is None:
            return None if c is None else dict(c)
        # 自分の動き: 曲率 κ の弧を ds 進んで、向きが κ·ds 変わった
        ds = kmh / 3.6 * dt
        kap = float(self.planner.last_kappa)
        if yaw_rate_dps is not None and kmh > 2.0:
            kap = -np.radians(float(yaw_rate_dps)) / (kmh / 3.6)     # 左回り(上から見て)が正のヨーレート → 右が正の曲率
        fa, ra, tha = arc_points(kap, np.array([ds]))
        th = float(tha[0])
        dF, dR = c["F"] - float(fa[0]), c["R"] - float(ra[0])
        c["F"] = dF * np.cos(th) + dR * np.sin(th)
        c["R"] = -dF * np.sin(th) + dR * np.cos(th)
        if c["F"] < 0.5:                                    # 板の横を過ぎた
            self._cue = None
            self.chev.active = None
            return None
        return dict(c)

    def step(self, frame: np.ndarray, speed_kmh: float | None = None, t: float | None = None,
             yaw_rate_dps: float | None = None) -> Decision:
        """絵(BGR)と今の速さ [km/h] → Decision。speed_kmh を渡さなければ、直前の目標の速さで走っているとみなす。
        yaw_rate_dps: 今のヨーレート [deg/s](上から見て左回りが正。IMU のジャイロ)。分かれば矢印板の位置を正しく動かせる。"""
        t = time.time() if t is None else float(t)
        known = speed_kmh is not None
        if speed_kmh is None:
            speed_kmh = self.planner.last_target or 0.0
        per = self.perceive(frame, t)
        rv = self.rv
        pred = per["pred"]
        F, R, U = rv.ref_to_user(pred.F, pred.R, pred.U)
        path = {"F": F, "R": R, "U": U, "sd": pred.sd * rv.k, "S": pred.S * rv.k}
        cue = self._cue_update(per.get("cue"), t, float(speed_kmh), rv, yaw_rate_dps)
        t0 = time.perf_counter()
        dec = self.planner.step(t, float(speed_kmh), path, pred.confidence,
                                block_fn=self._block_fn(rv, per), cue=cue, speed_known=known)
        self._tick("plan", t0)
        self.n += 1
        self.last = per
        return dec
