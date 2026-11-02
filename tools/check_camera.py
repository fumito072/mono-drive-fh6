"""カメラの設定が合っているか、絵の上に地面の格子を描いて確かめる。

    python tools/check_camera.py --config configs/my_car.json --input my_video.mp4 --frame 120

書き出す絵(--out、既定 camera_check.jpg):
  左 = 元の絵に、平らな地面の格子(前 5・10・15・20・30・40m の横線、右 ±1.75・±3.5m の縦線)と地平線(水色)、
       モデルが見る範囲(黄色の枠)を描いたもの
  右 = モデルが見る絵(基準の視点)に、同じ格子を描いたもの
合わせ方: 平らでまっすぐな道で、地平線の高さが合うまで pitch_deg を、白線・道幅(普通の車線 3.5m)が格子に合うまで
height_m と hfov_deg を直す。右の絵で道がまっすぐ上へ伸び、格子の真ん中に来ていれば合っている(docs/CAMERA.md)。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mono_drive.camera import NET_ROWS, REF_CAMERA, Camera, ReferenceView  # noqa: E402
from mono_drive.driver import load_config  # noqa: E402
from mono_drive.sources import FrameSource  # noqa: E402

DIST = (5, 10, 15, 20, 30, 40)
LAT = (-3.5, -1.75, 0.0, 1.75, 3.5)


def draw_grid(img, cam):
    """元の絵に、利用者のカメラで平らな地面の格子と地平線を描く。"""
    import cv2
    s = max(1, int(round(img.shape[0] / 400)))
    for d in DIST:
        r = np.linspace(-6, 6, 60)
        u, v, z = cam.project(np.full_like(r, d), r, np.zeros_like(r))
        _line(img, u, v, z, (0, 255, 0) if d != 20 else (0, 200, 255), s)
        uu, vv, zz = cam.project(np.array([d]), np.array([6.2]), np.zeros(1))
        if np.isfinite(uu[0]) and zz[0] > 0:
            cv2.putText(img, f"{d}m", (int(uu[0]), int(vv[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * s, (0, 255, 0), s)
    for r0 in LAT:
        f = np.linspace(2, 45, 80)
        u, v, z = cam.project(f, np.full_like(f, r0), np.zeros_like(f))
        _line(img, u, v, z, (255, 255, 255) if r0 == 0 else (0, 255, 0), s)
    u, v, _ = cam.project(np.array([1e5, 1e5]), np.array([-3e4, 3e4]), np.zeros(2))
    if np.isfinite(u).all():
        cv2.line(img, (int(u[0]), int(v[0])), (int(u[1]), int(v[1])), (255, 255, 0), s, cv2.LINE_AA)


def _line(img, u, v, z, color, th):
    import cv2
    ok = np.isfinite(u) & np.isfinite(v) & (z > 0.3) & (np.abs(u) < 1e5) & (np.abs(v) < 1e5)
    if ok.sum() >= 2:
        pts = np.stack([u[ok], v[ok]], 1).round().astype(np.int32)
        cv2.polylines(img, [pts], False, color, th, cv2.LINE_AA)


def main() -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=True, help="絵・動画・絵のフォルダ・カメラ番号")
    ap.add_argument("--frame", type=int, default=0, help="何枚目の絵で見るか")
    ap.add_argument("--out", default="camera_check.jpg")
    a = ap.parse_args()
    cfg, _ = load_config(a.config)
    cam = Camera.from_dict(cfg["camera"])
    if os.path.isfile(a.input) and a.input.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        frame = cv2.imread(a.input)
    else:
        frame = None
        src = FrameSource(a.input)
        for i, _, img in src:
            frame = img
            if i >= a.frame:
                break
        src.close()
    if frame is None:
        print("絵が読めない")
        return 1
    h, w = frame.shape[:2]
    if (w, h) != (cam.width, cam.height):
        print(f"注意: 設定の大きさ {cam.width}x{cam.height} と絵の大きさ {w}x{h} が違う → 比で直して使う")
    rv = ReferenceView(cam, REF_CAMERA, frame_size=(w, h))
    c = rv.cam
    left = frame.copy()
    draw_grid(left, c)
    # モデルが見る範囲(基準の視点の 0〜240 行の枠)を元の絵へ
    H, W = REF_CAMERA.height, REF_CAMERA.width
    y0, y1 = NET_ROWS
    border_u = np.concatenate([np.linspace(0, W - 1, 50), np.full(30, W - 1.0), np.linspace(W - 1, 0, 50), np.zeros(30)])
    border_v = np.concatenate([np.full(50, float(y0)), np.linspace(y0, y1 - 1, 30), np.full(50, y1 - 1.0),
                               np.linspace(y1 - 1, y0, 30)])
    bu, bv = rv.ref_px_to_user_px(border_u, border_v)
    ok = np.isfinite(bu) & np.isfinite(bv)
    if ok.sum() > 2:
        cv2.polylines(left, [np.stack([bu[ok], bv[ok]], 1).round().astype(np.int32)], True, (0, 255, 255),
                      max(1, h // 400), cv2.LINE_AA)
    right = rv.warp(frame).copy()
    # 右: 基準のカメラで、利用者の車の m を基準の世界へ直した格子
    ref = REF_CAMERA
    s = max(1, int(round(right.shape[0] / 400)))
    for d in DIST:
        r = np.linspace(-6, 6, 60)
        fr, rr = rv.user_to_ref(np.full_like(r, d), r)
        u, v, z = ref.project(fr, rr, np.zeros_like(r))
        _line(right, u, v, z, (0, 255, 0) if d != 20 else (0, 200, 255), s)
    for r0 in LAT:
        f = np.linspace(2, 45, 80)
        fr, rr = rv.user_to_ref(f, np.full_like(f, r0))
        u, v, z = ref.project(fr, rr, np.zeros_like(f))
        _line(right, u, v, z, (255, 255, 255) if r0 == 0 else (0, 255, 0), s)
    cv2.rectangle(right, (0, y0), (W - 1, y1 - 1), (0, 255, 255), 1)
    right[~rv.valid] = (right[~rv.valid] * 0.4).astype(np.uint8)
    rh = left.shape[0]
    right = cv2.resize(right, (int(W * rh / H), rh), interpolation=cv2.INTER_LINEAR)
    out = np.concatenate([left, right], 1)
    cv2.imwrite(a.out, out)
    print(f"書いた: {a.out}")
    print(f"  距離の倍率 k = {rv.k:.2f}(カメラの高さ {c.height_m:.2f}m / 学習時 1.1m)")
    print(f"  モデルの見る範囲のうち、この絵で埋まっている割合: {rv.coverage * 100:.0f}%"
          + ("" if rv.coverage >= 0.95 else "(足りない所は端の画素を引き伸ばして埋める。画角の広いカメラが望ましい)"))
    print(f"  地平線(真ん中の列): {c.horizon_v:.0f} 行目 / 絵の高さ {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
