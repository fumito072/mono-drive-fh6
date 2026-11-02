"""キットの自己診断。インストールの後に 1 回走らせる(30 秒ほど)。

    python tests/test_kit.py

確かめること:
  1. カメラ: 地面の点を絵へ写して戻すと元に戻るか(歪みあり・なし)
  2. 基準の視点: 学習時と同じカメラは切り出すだけか・倍の解像度でも同じ絵になるか
  3. 合成カメラ: 画角・向き・解像度の違うカメラの絵を作り、基準の視点へ戻して、道の予想がほぼ同じになるか
  4. 座標の換算(利用者の車 ↔ 基準の世界)が行って戻るか
  5. 教師: まっすぐ走った位置の記録から、まっすぐの教師ができるか
  6. MonoDriver: サンプルの絵で答えが出るか(全部の数が有限・モードが正しい)
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile

import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)

from mono_drive import MonoDriver  # noqa: E402
from mono_drive.camera import NET_ROWS, REF_CAMERA, Camera, ReferenceView  # noqa: E402
from mono_drive.pathnet import ARC_S, PathNet, net_input  # noqa: E402

FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'NG'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def sample_frames(n: int = 20):
    import cv2
    files = sorted(glob.glob(os.path.join(KIT, "samples", "*", "frames", "*.jpg")))
    if not files:
        return []
    idx = np.linspace(0, len(files) - 1, n).astype(int)
    return [cv2.imread(files[i]) for i in idx]


def t_camera():
    print("1. カメラ")
    for dist, model in ((None, "pinhole"), ([-0.25, 0.08, 0.001, -0.002, 0.0], "pinhole"), ([0.05, -0.01, 0.0, 0.0], "fisheye")):
        cam = Camera(width=1280, height=720, hfov_deg=95, pitch_deg=6, roll_deg=1.5, yaw_deg=-2, height_m=1.4,
                     fwd_m=1.2, right_m=0.3, dist=dist, model=model)
        F = np.array([4.0, 8.0, 15.0, 30.0]); R = np.array([-3.0, 1.0, 2.5, -4.0])
        u, v = cam.ground_to_image(F, R)
        f2, r2 = cam.image_to_ground(u, v)
        err = float(np.max(np.hypot(f2 - F, r2 - R)))
        check(f"写して戻す({model}{'・歪み' if dist else ''})", err < 1e-3, f"最大のずれ {err:.1e} m")


def t_reference(frames):
    import cv2
    print("2. 基準の視点")
    cam = Camera.load(os.path.join(KIT, "configs", "sample_c4.json"))
    rv = ReferenceView(cam, REF_CAMERA, frame_size=(576, 374))
    check("学習時と同じカメラは切り出すだけ", rv.slice == 14 and rv.pre == 1.0, f"slice={rv.slice}")
    if frames:
        f = frames[0]
        big = cv2.resize(f, (1152, 748), interpolation=cv2.INTER_NEAREST)
        rv2 = ReferenceView(cam, REF_CAMERA, frame_size=(1152, 748))
        d = float(np.abs(rv2.warp(big).astype(int) - rv.warp(f).astype(int)).max())
        check("倍の解像度でも同じ絵", d <= 1, f"最大の差 {d:.0f}")


def t_synthetic(frames, pn):
    print("3. 合成カメラ(画角 100°・下向き 8°・ロール 2°・右向き -3°・1280x720)")
    if not frames:
        check("サンプルの絵がある", False)
        return
    V = Camera(width=1280, height=720, hfov_deg=100, pitch_deg=8, roll_deg=2, yaw_deg=-3,
               height_m=REF_CAMERA.height_m, fwd_m=REF_CAMERA.fwd_m, right_m=REF_CAMERA.right_m)
    cam = Camera.load(os.path.join(KIT, "configs", "sample_c4.json"))
    rv0 = ReferenceView(cam, REF_CAMERA, frame_size=(576, 374))
    to_v = ReferenceView(REF_CAMERA, ref=V, frame_size=(576, 360))       # 基準の視点 → 合成カメラの絵
    back = ReferenceView(V, REF_CAMERA, frame_size=(1280, 720))           # 合成カメラの絵 → 基準の視点
    errs, diffs = [], []
    for f in frames[::4]:
        ref = rv0.warp(f)
        synth = to_v.warp(ref)
        got = back.warp(synth)
        m = back.valid & to_v.valid.any() & rv0.valid
        m[:NET_ROWS[0]] = False
        m[NET_ROWS[1]:] = False
        diffs.append(float(np.abs(got.astype(int) - ref.astype(int))[m].mean()))
        p0 = pn.predict(net_input(ref))
        p1 = pn.predict(net_input(got))
        mid = (ARC_S >= 8) & (ARC_S <= 22)
        errs.append(float(np.abs(p1.R[mid] - p0.R[mid]).mean()))
    check("戻した絵が元とほぼ同じ", np.mean(diffs) < 12, f"画素の差の平均 {np.mean(diffs):.1f}/255・埋まり {back.coverage * 100:.0f}%")
    check("道の予想がほぼ同じ", np.mean(errs) < 0.5, f"8〜22m の横の差の平均 {np.mean(errs):.2f} m")


def t_convert():
    print("4. 座標の換算")
    cam = Camera(width=1280, height=720, hfov_deg=90, pitch_deg=3, height_m=2.2, fwd_m=1.5, right_m=0.4)
    rv = ReferenceView(cam, REF_CAMERA, frame_size=(1280, 720))
    F = np.array([5.0, 20.0]); R = np.array([-1.0, 2.0]); U = np.array([0.0, 0.5])
    f2, r2, u2 = rv.user_to_ref(*rv.ref_to_user(F, R, U))
    check("行って戻る", np.allclose([f2, r2, u2], [F, R, U]), f"k = {rv.k:.2f}")
    # 地面の点: 利用者のカメラで写した画素の向き = 基準の世界へ直した点を基準のカメラで写した画素の向き
    fu, ru = np.array([10.0, 25.0]), np.array([1.5, -2.0])
    uu, vu = cam.ground_to_image(fu, ru)
    fr, rr = rv.user_to_ref(fu, ru)
    ur, vr = REF_CAMERA.ground_to_image(fr, rr)
    du, dv = rv.ref_px_to_user_px(ur, vr)
    err = float(np.max(np.hypot(du - uu, dv - vu)))
    check("地面の点の写り方が合う(高さの違いを倍率で直す)", err < 0.5, f"最大 {err:.2f} 画素")


def t_teacher():
    print("5. 教師")
    import cv2
    from mono_drive.teacher import Drive
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "frames"))
        img = np.zeros((374, 576, 3), np.uint8)
        with open(os.path.join(d, "poses.csv"), "w") as f:
            f.write("frame,t,x,y,z,yaw_deg\n")
            for i in range(60):                       # 北東(45°)へ 2m ごとにまっすぐ
                cv2.imwrite(os.path.join(d, "frames", f"{i:06d}.png"), img)
                s = 2.0 * i
                f.write(f"{i},{i * 0.1:.2f},{s * 0.7071:.4f},{s * 0.7071:.4f},0,45\n")
        cam = Camera.load(os.path.join(KIT, "configs", "sample_c4.json"))
        dr = Drive(d, cam)
        rv = ReferenceView(cam, REF_CAMERA, frame_size=(576, 374))
        T, ok = dr.target(0, rv)
        e = float(np.max(np.abs(T[ok, 0] - ARC_S[ok]))) if ok.any() else 99
        r = float(np.max(np.abs(T[ok, 1]))) if ok.any() else 99
        check("まっすぐの教師", ok.sum() >= 10 and e < 0.05 and r < 0.05, f"前のずれ {e:.3f}・横 {r:.3f} m")


def t_driver(frames):
    print("6. MonoDriver(深度あり)")
    if not frames:
        check("サンプルの絵がある", False)
        return
    drv = MonoDriver(os.path.join(KIT, "configs", "sample_c4.json"))
    ok = True
    modes = set()
    for i, f in enumerate(frames):
        dec = drv.step(f, speed_kmh=40.0, t=i * 0.2)
        vals = [dec.kappa, dec.steer, dec.steer_deg, dec.target_kmh, dec.confidence]
        ok &= all(np.isfinite(v) for v in vals) and -1 <= dec.steer <= 1 and dec.target_kmh >= 0
        modes.add(dec.mode)
    check("答えが出る", ok and modes <= {"path", "cue", "avoid", "blocked", "lost"},
          f"モード {sorted(modes)}・1 枚 " + " ".join(f"{k} {v:.0f}ms" for k, v in drv.ms.items()))


def main() -> int:
    print("mono_drive 自己診断")
    frames = sample_frames()
    pn = PathNet(os.path.join(KIT, "models", "pathnet.onnx"))
    t_camera()
    t_reference(frames)
    t_synthetic(frames, pn)
    t_convert()
    t_teacher()
    t_driver(frames)
    print("結果: " + ("すべて OK" if not FAILS else f"NG {len(FAILS)} 件: {', '.join(FAILS)}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
