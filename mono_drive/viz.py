"""判断の重ね描き(元の絵 + 上から見た図)。文字は環境に依らないよう英字だけ。

    img = draw(frame, dec, drv)       # (h, w + 上から見た図の幅, 3) BGR
"""

from __future__ import annotations


import numpy as np

from .camera import NET_ROWS
from .planner import arc_points

MODE_COLOR = {"path": (80, 220, 80), "cue": (255, 200, 0), "avoid": (0, 220, 255), "blocked": (40, 40, 255),
              "lost": (160, 160, 160)}


def _tint(img, mask, color, alpha):
    if mask is None or not mask.any():
        return
    c = np.array(color, np.float32)
    sub = img[mask].astype(np.float32)
    img[mask] = (sub * (1 - alpha) + c * alpha).astype(np.uint8)


def _poly(img, u, v, color, thick, closed=False):
    import cv2
    ok = np.isfinite(u) & np.isfinite(v) & (np.abs(u) < 1e5) & (np.abs(v) < 1e5)
    if ok.sum() < 2:
        return
    pts = np.stack([u[ok], v[ok]], 1).round().astype(np.int32)
    cv2.polylines(img, [pts], closed, color, thick, cv2.LINE_AA)


def draw(frame: np.ndarray, dec, drv, bev: bool = True, inset: bool = True) -> np.ndarray:
    import cv2
    img = frame.copy()
    h, w = img.shape[:2]
    rv = drv.rv
    cam = rv.cam
    per = drv.last or {}
    s = max(1.0, h / 540.0)                                  # 線と文字の太さ
    th = max(1, int(round(2 * s)))
    # 障害物・深度の塗り(基準の視点 → 元の絵)
    # 薄い色 = 塗り全体、濃い色 = 塞ぎに使う足元
    for key, color in (("wall", (0, 140, 255)), ("drop", (255, 0, 200)), ("obstacle", (0, 0, 255))):
        m = per.get(key)
        if m is not None and m.any():
            _tint(img, rv.ref_mask_to_user(m), color, 0.18)
        fp = per.get(key + "_fp")
        if fp is not None and fp.any():
            _tint(img, rv.ref_mask_to_user(fp), color, 0.55)
    # 予想した道(青)と自信(点の大きさ = 標準偏差)
    if dec.path is not None and dec.mode != "lost":
        F, R, U, sd = (np.asarray(dec.path[k], float) for k in ("F", "R", "U", "sd"))
        u, v, z = cam.project(F, R, U)
        _poly(img, u, v, (255, 120, 0), th)
        for ui, vi, zi, si in zip(u, v, z, sd):
            if np.isfinite(ui) and np.isfinite(vi) and zi > 0.5 and 0 <= ui < w and 0 <= vi < h:
                r = max(2, int(cam.fx * min(si, 4.0) / max(zi, 1.0) * 0.5))
                cv2.circle(img, (int(ui), int(vi)), min(r, int(40 * s)), (255, 160, 60), 1, cv2.LINE_AA)
    # 選んだ弧(車幅の帯)
    col = MODE_COLOR.get(dec.mode, (255, 255, 255))
    reach = float(dec.debug.get("reach_m", 20.0))
    ds = np.arange(2.0, reach + 0.01, 1.0)
    Fa, Ra, tha = arc_points(dec.kappa, ds)
    half = drv.vehicle.width_m / 2
    for sgn in (-1, 1):
        u, v, _ = cam.project(Fa - sgn * half * np.sin(tha), Ra + sgn * half * np.cos(tha), np.zeros_like(Fa))
        _poly(img, u, v, col, th)
    if dec.blocked_m is not None:
        Fb, Rb, _ = arc_points(dec.debug.get("kappa_goal", dec.kappa), np.array([dec.blocked_m]))
        u, v, _ = cam.project(Fb, Rb, np.zeros(1))
        if np.isfinite(u[0]) and np.isfinite(v[0]):
            c = (int(u[0]), int(v[0]))
            cv2.drawMarker(img, c, (0, 0, 255), cv2.MARKER_TILTED_CROSS, int(24 * s), th + 1)
    # 矢印板
    act = per.get("cue")
    for cue in per.get("cues") or []:
        x0, y0, x1, y1 = cue["box"]
        uu, vv = rv.ref_px_to_user_px(np.array([x0, x1, x1, x0], float), np.array([y0, y0, y1, y1], float))
        c = (255, 140, 0) if cue["dir"] < 0 else (0, 140, 255)
        bold = act is not None and act["box"] == cue["box"]
        _poly(img, uu, vv, c, th + (2 if bold else 0), closed=True)
        if np.isfinite(uu).all():
            cv2.putText(img, "<<" if cue["dir"] < 0 else ">>", (int(uu[0]), int(vv[0] - 4 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, c, th, cv2.LINE_AA)
    # 文字
    lines = [f"{dec.mode.upper():8s} {dec.target_kmh:5.1f} km/h  ({dec.limiter_code})",
             f"steer {dec.steer:+.2f}  ({dec.steer_deg:+.1f} deg, k={dec.kappa:+.3f}/m)",
             f"path conf {dec.confidence:.2f}"]
    if dec.blocked_m is not None:
        lines.append(f"blocked {dec.blocked_m:.0f} m by {dec.blocked_by}")
    if dec.cue:
        dd = dec.cue.get("dist_m")
        lines.append(f"chevron {'LEFT' if dec.cue['dir'] < 0 else 'RIGHT'}" + (f" {dd:.0f} m" if dd else ""))
    if dec.reverse:
        lines.append("REVERSE")
    y = int(22 * s)
    for ln in lines:
        cv2.putText(img, ln, (int(10 * s), y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(img, ln, (int(10 * s), y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (255, 255, 255), th, cv2.LINE_AA)
        y += int(22 * s)
    # ハンドルの棒
    bx, by, bw = int(10 * s), y, int(160 * s)
    cv2.rectangle(img, (bx, by), (bx + bw, by + int(10 * s)), (255, 255, 255), 1)
    cx = bx + bw // 2 + int(dec.steer * bw / 2)
    cv2.rectangle(img, (min(cx, bx + bw // 2), by), (max(cx, bx + bw // 2), by + int(10 * s)), col, -1)
    # モデルが見ている絵(右上)
    if inset and per.get("ref") is not None:
        ref = per["ref"].copy()
        cv2.rectangle(ref, (0, NET_ROWS[0]), (ref.shape[1] - 1, NET_ROWS[1] - 1), (0, 255, 255), 1)
        iw = int(w * 0.28)
        ih = int(iw * ref.shape[0] / ref.shape[1])
        if ih < h // 2:
            small = cv2.resize(ref, (iw, ih), interpolation=cv2.INTER_AREA)
            img[4:4 + ih, w - iw - 4:w - 4] = small
            cv2.rectangle(img, (w - iw - 5, 3), (w - 3, 4 + ih), (0, 255, 255), 1)
            cv2.putText(img, "model view", (w - iw, 4 + ih - int(6 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * s,
                        (0, 255, 255), 1, cv2.LINE_AA)
    if bev:
        img = np.concatenate([img, draw_bev(dec, drv, h)], 1)
    return img


def draw_bev(dec, drv, height: int, fwd_m: float = 40.0) -> np.ndarray:
    """上から見た図(車は下の真ん中、上が前)。"""
    import cv2
    H = int(height)
    W = int(H * 0.45)
    img = np.full((H, W, 3), 24, np.uint8)
    ppm = (H - 20) / fwd_m
    ox, oy = W // 2, H - 10

    def to_px(F, R):
        return ox + np.asarray(R) * ppm, oy - np.asarray(F) * ppm

    for d in range(10, int(fwd_m) + 1, 10):
        y = int(oy - d * ppm)
        cv2.line(img, (0, y), (W, y), (60, 60, 60), 1)
        cv2.putText(img, f"{d}m", (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
    for r in (-10, -5, 5, 10):
        x = int(ox + r * ppm)
        cv2.line(img, (x, 0), (x, H), (45, 45, 45), 1)
    rv = drv.rv
    per = drv.last or {}
    ref = rv.ref
    # 障害物・深度の塗りを地面へ
    for key, color in (("wall", (0, 140, 255)), ("drop", (255, 0, 200)), ("obstacle", (0, 0, 255))):
        m = per.get(key + "_fp")
        if m is None or not m.any():
            continue
        vv, uu = np.nonzero(m[::3, ::3])
        fr, rr = ref.image_to_ground(uu * 3.0, vv * 3.0)
        fu, ru = rv.ref_to_user(fr, rr)
        ok = np.isfinite(fu) & (fu > 0) & (fu < fwd_m)
        x, y = to_px(fu[ok], ru[ok])
        for xi, yi in zip(x.astype(int), y.astype(int)):
            if 0 <= xi < W and 0 <= yi < H:
                img[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2] = color
    # 予想した道と幅(標準偏差)
    if dec.path is not None and dec.mode != "lost":
        F, R, sd = (np.asarray(dec.path[k], float) for k in ("F", "R", "sd"))
        for f, r, s_ in zip(F, R, sd):
            x, y = to_px(f, r)
            cv2.circle(img, (int(x), int(y)), max(1, int(min(s_, 6) * ppm)), (120, 70, 0), 1, cv2.LINE_AA)
        x, y = to_px(np.concatenate([[0.0], F]), np.concatenate([[0.0], R]))
        _poly(img, x, y, (255, 120, 0), 2)
    # 選んだ弧
    col = MODE_COLOR.get(dec.mode, (255, 255, 255))
    ds = np.arange(0.0, float(dec.debug.get("reach_m", 25.0)) + 0.01, 0.5)
    Fa, Ra, tha = arc_points(dec.kappa, ds)
    half = drv.vehicle.width_m / 2
    for sgn in (-1, 1):
        x, y = to_px(Fa - sgn * half * np.sin(tha), Ra + sgn * half * np.cos(tha))
        _poly(img, x, y, col, 1)
    pp = dec.debug.get("pursuit")
    if pp:
        x, y = to_px(pp["f"], pp["y"])
        cv2.drawMarker(img, (int(x), int(y)), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
    if dec.cue and dec.cue.get("F") is not None:
        x, y = to_px(dec.cue["F"], dec.cue["R"])
        c = (255, 140, 0) if dec.cue["dir"] < 0 else (0, 140, 255)
        cv2.drawMarker(img, (int(x), int(y)), c, cv2.MARKER_TRIANGLE_UP, 12, 2)
        xs, ys = to_px(dec.cue["F"], dec.cue["side_m"])
        cv2.circle(img, (int(xs), int(ys)), 3, c, -1)
    # 車
    vw, vl = max(2, int(half * ppm)), max(4, int(1.5 * ppm))
    cv2.rectangle(img, (ox - vw, oy - vl), (ox + vw, oy), (230, 230, 230), 1)
    cv2.putText(img, "top view", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return img
