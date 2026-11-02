"""走る線(曲率)と目標の速さを決める(学習なし・幾何と物理だけ)。単位はすべて利用者の車の m・km/h。

1. **道をたどる**: 当てた道(PathNet)の、道なりに L m 先の点を狙う Pure Pursuit。曲率 κ = 2y / (x² + y²)
2. **矢印板**: 読めた向きへ曲がる。板が cue_turn_m より近づいたら、板の矢印の側 cue_side_m の点を狙う弧と
   比べて、その向きへ強い方を取る。読んでいる間は逆へ切らない
3. **障害物は「遅くする」より「空いている方へ避ける」**: 狙いの弧に沿って車幅 + 余裕の帯を絵に写し、
   障害物(学習した検出・深度の立ち上がり・落ち込み)が無いか見る。塞がれていれば、予想した道から
   max_off_path_m 以内で、狙いに近い順に空いた弧を探す。どれも塞がれていれば、いちばん遠くまで空いた弧で
   止まれる速さへ落とし、すぐ前(back_m)が塞がれて止まりかけていたら「下がる」を出す
4. **速さ**: 車の上限・この先の曲がり(横G と減速G で間に合う速さ)・見通し(当てた道が確かな長さの中で
   sight_end_kmh まで落とせる速さ)・今の弧の横G・矢印板の手前・塞がれた所の手前、のいちばん小さい値
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .pathnet import curvature_profile

G = 9.81


@dataclass
class VehicleConfig:
    wheelbase_m: float = 2.6          # ホイールベース(切れ角 = atan(ホイールベース × 曲率))
    max_steer_deg: float = 30.0       # 前輪の最大切れ角
    width_m: float = 1.8              # 車幅(障害物の帯)
    max_kmh: float = 60.0             # 速さの上限
    a_lat_g: float = 0.45             # 曲がりで許す横G
    a_brake_g: float = 0.40           # 先読みで使う減速G
    react_s: float = 0.4              # 反応の遅れ [s]
    accel_mps2: float = 2.5           # 目標の速さを上げるときの上限 [m/s²](下げるのはすぐ)

    @classmethod
    def from_dict(cls, d: dict | None) -> "VehicleConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PlannerConfig:
    la_base_m: float = 5.0            # Pure Pursuit の先読み L = clip(base + gain × 速さ[m/s], min, max)
    la_gain_s: float = 0.9
    la_min_m: float = 7.0
    la_max_m: float = 20.0
    kappa_gain: float = 1.6           # 先読みの速さでは、予測の曲がりをこの倍で読む(予測は曲がりを小さく見積もる癖)
    sight_sd_m: float = 2.0           # 見通し: 横の標準偏差がこれ以下で続く長さ
    sight_end_kmh: float = 30.0       # 見通しの終わりで落としておく速さ
    sight_brake_g: float = 0.30
    lost_conf: float = 0.15           # 自信がこれ未満 = 道を見失った
    lost_kmh: float = 10.0            # 見失った間の速さ
    lost_hold_s: float = 1.0          # 見失ってからこの秒数は直前の曲率を保つ(その後は 0 へ)
    margin_m: float = 0.3             # 障害物の帯: 車幅の左右に足す余裕
    block_frac: float = 0.3           # 帯の横の点のこの割合以上が障害物なら「塞がれた」
    reach_min_m: float = 10.0         # 障害物を見る距離 = 止まれる距離 + 6m を [min, max] に収めたもの
    reach_max_m: float = 30.0
    stop_margin_m: float = 2.0        # 塞がれた所のこの手前で止まれる速さにする
    back_m: float = 4.0               # これより近くが塞がれ、止まりかけていたら「下がる」
    back_kmh: float = 6.0
    max_off_path_m: float = 3.0       # 避ける弧が予想した道から離れてよい横の距離
    n_arcs: int = 41                  # 探す弧の数
    cue_turn_m: float = 16.0          # 矢印板がこれより近づいたら曲がる
    cue_side_m: float = 3.0           # 板の矢印の側この距離を通る
    cue_slow_m: float = 6.0           # 板のこの手前で cue_kmh まで落とす
    cue_kmh: float = 15.0
    min_kmh: float = 0.0              # 下限(道が確かなときだけ。0 = 使わない)

    @classmethod
    def from_dict(cls, d: dict | None) -> "PlannerConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Decision:
    """1 枚の絵に対する答え。"""
    t: float
    kappa: float                      # 狙う曲率 [1/m](右が正)
    steer_deg: float                  # 前輪の切れ角 [deg](右が正)
    steer: float                      # -1..1(最大切れ角で割った値)
    target_kmh: float                 # 目標の速さ
    reverse: bool = False             # 前が塞がれた → 下がる
    confidence: float = 0.0           # 道の認識の自信 0..1
    mode: str = "path"                # path | cue | avoid | blocked | lost
    limiter: str = ""                 # 速さを決めた理由
    limiter_code: str = ""            # 同じ理由の短い英字(max / curve / sight / lost / cue / blocked / lat_g / min / reverse)
    blocked_m: float | None = None    # 狙いの弧が塞がれた距離
    blocked_by: str = ""              # 何に塞がれたか(obstacle / wall / drop)
    cue: dict | None = None           # 矢印板 {"dir", "dist_m", "side_m"}
    path: dict | None = None          # 予想した道(利用者の車の座標)
    debug: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {"t": round(self.t, 3), "kappa": round(self.kappa, 5), "steer_deg": round(self.steer_deg, 2),
                "steer": round(self.steer, 3), "target_kmh": round(self.target_kmh, 1),
                "reverse": int(self.reverse), "confidence": round(self.confidence, 3), "mode": self.mode,
                "limiter": self.limiter, "limiter_code": self.limiter_code,
                "blocked_m": "" if self.blocked_m is None else round(self.blocked_m, 1),
                "blocked_by": self.blocked_by,
                "cue_dir": "" if not self.cue else self.cue["dir"],
                "cue_m": "" if not self.cue or self.cue.get("dist_m") is None else round(self.cue["dist_m"], 1)}

    def to_json(self) -> dict:
        d = self.to_row()
        if self.path:
            d["path"] = {k: [round(float(x), 2) for x in v] for k, v in self.path.items()}
        return d


def arc_points(kappa: float, ds: np.ndarray):
    """曲率 κ の弧を道なりに ds [m] 進んだ点 (前, 右) と、その向き [rad]。"""
    ds = np.asarray(ds, float)
    th = kappa * ds
    if abs(kappa) < 1e-6:
        return ds.copy(), np.zeros_like(ds), th
    return np.sin(th) / kappa, (1.0 - np.cos(th)) / kappa, th


class Planner:
    def __init__(self, vehicle: VehicleConfig | None = None, cfg: PlannerConfig | None = None):
        self.v = vehicle or VehicleConfig()
        self.c = cfg or PlannerConfig()
        self.kappa_max = min(0.25, math.tan(math.radians(self.v.max_steer_deg)) / max(0.5, self.v.wheelbase_m))
        self.last_kappa = 0.0
        self.last_ok_t = None
        self.last_target = None
        self.last_t = None

    # --- 部品 -------------------------------------------------------------
    def lookahead(self, kmh: float) -> float:
        c = self.c
        return float(np.clip(c.la_base_m + c.la_gain_s * max(0.0, kmh) / 3.6, c.la_min_m, c.la_max_m))

    def pursuit(self, F, R, S, kmh: float) -> tuple[float, tuple]:
        L = self.lookahead(kmh)
        f = float(np.interp(L, S, F))
        y = float(np.interp(L, S, R))
        return 2.0 * y / max(1.0, f * f + y * y), (L, f, y)

    def speed_ahead(self, F, R, S, kmh: float):
        """この先の曲がりに間に合う速さ [km/h] と、それを決めた道なりの距離 [m]。"""
        k, s_mid = curvature_profile(F, R, S)
        if not len(k):
            return None, None
        v = self.v
        react = max(0.0, kmh) / 3.6 * v.react_s
        kk = np.maximum(1e-3, np.abs(k) * self.c.kappa_gain)
        v_curve = np.sqrt(v.a_lat_g * G / kk)
        v_ok = np.sqrt(v_curve ** 2 + 2 * v.a_brake_g * G * np.maximum(0.0, s_mid - react))
        i = int(np.argmin(v_ok))
        return float(v_ok[i] * 3.6), float(s_mid[i])

    def sight_speed(self, sd, S) -> tuple[float, float]:
        c = self.c
        L = 0.0
        for s, v in zip(S, sd):
            if v <= c.sight_sd_m:
                L = float(s)
            else:
                break
        v = math.sqrt((c.sight_end_kmh / 3.6) ** 2 + 2 * c.sight_brake_g * G * max(0.0, L - 4.0)) * 3.6
        return v, L

    def reach(self, kmh: float) -> float:
        v = max(0.0, kmh) / 3.6
        stop = v * self.v.react_s + v * v / (2 * self.v.a_brake_g * G)
        return float(np.clip(stop + 6.0, self.c.reach_min_m, self.c.reach_max_m))

    @staticmethod
    def _dist_to_path(F, R, pF, pR) -> np.ndarray:
        """弧の各点から、予想した道(折れ線)までの距離 [m]。"""
        pts = np.stack([np.concatenate([[0.0], pF]), np.concatenate([[0.0], pR])], 1)
        seg = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.hypot(*(b - a)) / 0.5) + 1)
            seg.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
        dense = np.concatenate(seg)
        q = np.stack([F, R], 1)
        return np.sqrt(((q[:, None, :] - dense[None, :, :]) ** 2).sum(-1)).min(1)

    # --- 1 枚分 -----------------------------------------------------------
    def step(self, t: float, kmh: float, path: dict | None, confidence: float,
             block_fn=None, cue: dict | None = None, speed_known: bool = True) -> Decision:
        """path: 予想した道 {"F","R","U","sd","S"}(利用者の車の m)。
        block_fn(kappa, ds) → (塞がれた距離 or None, 何に) : 弧が障害物に当たるか(driver.py が絵で調べる)。
        cue: 矢印板 {"dir": ±1, "F": 前 m, "R": 右 m}(読めていれば)。"""
        c, v = self.c, self.v
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        kmh = max(0.0, float(kmh))
        limits: list[tuple[float, str, str]] = [(v.max_kmh, "車の上限", "max")]
        dbg: dict = {}
        mode = "path"
        lost = path is None or confidence < c.lost_conf
        if not lost:
            F, R, S, sd = (np.asarray(path[k], float) for k in ("F", "R", "S", "sd"))
            kappa_goal, pp = self.pursuit(F, R, S, kmh)
            dbg["pursuit"] = {"L": round(pp[0], 1), "f": round(pp[1], 2), "y": round(pp[2], 2)}
            va, sa = self.speed_ahead(F, R, S, kmh)
            if va is not None:
                limits.append((va, f"この先の曲がり({sa:.0f}m 先)", "curve"))
            vs, Ls = self.sight_speed(sd, S)
            limits.append((vs, f"見通し {Ls:.0f}m", "sight"))
            self.last_ok_t = t
        else:
            # 見失った: しばらく直前の曲がりを保ち、その後はまっすぐ。ゆっくり
            held = self.last_ok_t is not None and t - self.last_ok_t <= c.lost_hold_s
            kappa_goal = self.last_kappa if held else 0.0
            limits.append((c.lost_kmh, "道を見失った", "lost"))
            mode = "lost"
        # --- 矢印板 ---
        cue_out = None
        if cue is not None and cue.get("F") is not None and np.isfinite(cue["F"]):
            d_b = float(math.hypot(cue["F"], cue["R"]))
            side = float(cue["R"]) + cue["dir"] * c.cue_side_m
            cue_out = {"dir": int(cue["dir"]), "dist_m": d_b, "side_m": side, "F": float(cue["F"]), "R": float(cue["R"])}
            if d_b <= c.cue_turn_m:
                f = max(1.0, float(cue["F"]))
                k_cue = 2.0 * side / (f * f + side * side)
                d = cue["dir"]
                kappa_goal = d * max(d * kappa_goal, d * k_cue, 0.0)
                mode = "cue" if mode != "lost" else mode
            v_cue = math.sqrt((c.cue_kmh / 3.6) ** 2 + 2 * v.a_brake_g * G * max(0.0, d_b - c.cue_slow_m)) * 3.6
            limits.append((v_cue, f"矢印板 {d_b:.0f}m", "cue"))
        elif cue is not None:
            cue_out = {"dir": int(cue["dir"]), "dist_m": None, "side_m": None}
        kappa_goal = float(np.clip(kappa_goal, -self.kappa_max, self.kappa_max))
        dbg["kappa_goal"] = round(kappa_goal, 5)
        # --- 障害物 ---
        kappa_cmd, blocked_m, blocked_by = kappa_goal, None, ""
        reverse = False
        if block_fn is not None:
            reach = self.reach(kmh)
            ds = np.arange(3.0, reach + 0.01, 1.0)
            d0, why0 = block_fn(kappa_goal, ds)
            dbg["reach_m"] = round(reach, 1)
            if d0 is not None:
                blocked_m, blocked_by = d0, why0
                cands = np.linspace(-self.kappa_max, self.kappa_max, c.n_arcs)
                pen = np.abs(cands - kappa_goal)
                # 矢印板を読んでいる間(板が cue_turn_m より近い)は、矢印と逆へ切る弧は選ばない
                cue_dir = 0
                if cue_out is not None and cue_out.get("dist_m") is not None and cue_out["dist_m"] <= c.cue_turn_m:
                    cue_dir = int(cue_out["dir"])
                found = None
                best_far = (d0, kappa_goal, why0)
                tried = []
                for kc in cands[np.argsort(pen, kind="stable")]:
                    if abs(kc - kappa_goal) < 1e-9:
                        continue
                    if cue_dir and cue_dir * kc < -0.01:
                        continue
                    if not lost:
                        Fa, Ra, _ = arc_points(float(kc), ds)
                        if self._dist_to_path(Fa, Ra, F, R).max() > c.max_off_path_m:
                            tried.append((float(kc), "道から離れる"))
                            continue
                    dk, why = block_fn(float(kc), ds)
                    if dk is None:
                        found = float(kc)
                        tried.append((float(kc), "空き"))
                        break
                    tried.append((float(kc), why))
                    if dk > best_far[0]:
                        best_far = (dk, float(kc), why)
                dbg["tried"] = tried[:12]
                if found is not None:
                    kappa_cmd, mode = found, "avoid"
                else:
                    d_far, kappa_cmd, _ = best_far
                    mode = "blocked"
                    v_stop = math.sqrt(2 * v.a_brake_g * G * max(0.0, d_far - c.stop_margin_m)) * 3.6
                    limits.append((v_stop, f"抜け道なし({d_far:.0f}m 先で塞がれる)", "blocked"))
                    if d_far <= c.back_m and kmh <= c.back_kmh:
                        reverse = True
        # --- 今の弧の横G ---
        if abs(kappa_cmd) > 1e-4:
            limits.append((math.sqrt(v.a_lat_g * G / abs(kappa_cmd)) * 3.6, "今の曲がりの横G", "lat_g"))
        target, limiter, code = min(limits, key=lambda x: x[0])
        if c.min_kmh > 0 and mode in ("path", "avoid", "cue") and confidence >= 0.45 and target < c.min_kmh:
            # 下限(道が確かで、塞がれていない時だけ)。この先の曲がり・矢印板の手前は下限より優先
            if not limiter.startswith(("この先の曲がり", "矢印板", "見通し")):
                target, limiter, code = c.min_kmh, "下限", "min"
        # 上げるのはなめらかに(下げるのはすぐ)。今の速さが分かれば、そこから accel_mps2 で 1.5 秒分まで
        if speed_known and target > kmh:
            target = min(target, kmh + v.accel_mps2 * 1.5 * 3.6)
        if self.last_target is not None and target > self.last_target and dt > 0:
            target = min(target, max(self.last_target + v.accel_mps2 * dt * 3.6, kmh if speed_known else 0.0))
        if reverse:
            target = 0.0
            limiter, code = "前が塞がれた → 下がる", "reverse"
        self.last_target = target
        self.last_kappa = kappa_cmd
        steer_rad = math.atan(v.wheelbase_m * kappa_cmd)
        steer = steer_rad / math.radians(v.max_steer_deg)
        return Decision(t=t, kappa=float(kappa_cmd), steer_deg=math.degrees(steer_rad),
                        steer=float(np.clip(steer, -1, 1)), target_kmh=float(max(0.0, target)), reverse=reverse,
                        confidence=float(confidence), mode=mode, limiter=limiter, limiter_code=code, blocked_m=blocked_m,
                        blocked_by=blocked_by, cue=cue_out, path=path, debug=dbg)

    def config(self) -> dict:
        return {"vehicle": asdict(self.v), "planner": asdict(self.c)}
