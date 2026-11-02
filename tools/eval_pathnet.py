"""PathNet の当たり具合を、自分の走行の記録(絵 + 位置・向き)で測る。

    python tools/eval_pathnet.py samples/sweden_unseen
    python tools/eval_pathnet.py my_drive1 my_drive2 --config configs/my_car.json --model models/pathnet.onnx

正解 = その後に実際に通った線(mono_drive/teacher.py)。測る値:
  ずれ [m]  … 道なり 8〜22m 先の点(4 か所)で、予想と実際に通った点の距離の平均(利用者の車の m)
  外れ      … ずれが 2.5m を超えた絵の割合
  自信あり  … 自信(confidence)0.5 以上の絵の割合と、その絵のずれ(自信の目盛りが合っているか)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mono_drive.camera import REF_CAMERA, ReferenceView  # noqa: E402
from mono_drive.pathnet import ARC_S, PathNet, net_input  # noqa: E402
from mono_drive.teacher import Drive, load_config_camera  # noqa: E402

MID = (ARC_S >= 8) & (ARC_S <= 22)


def evaluate(model: str, drives: list[str], config: str | None = None, stride: int = 1, out_csv: str | None = None,
             quiet: bool = False) -> dict:
    import cv2
    pn = PathNet(model)
    cam = load_config_camera(config)
    rows = []
    for root in drives:
        d = Drive(root, cam)
        rv = None
        for i in range(0, len(d), max(1, stride)):
            img = cv2.imread(d.files[i])
            if img is None:
                continue
            if rv is None or (rv.cam.width, rv.cam.height) != (img.shape[1], img.shape[0]):
                rv = ReferenceView(d.camera, REF_CAMERA, frame_size=(img.shape[1], img.shape[0]))
            tg = d.target(i, rv)
            if tg is None:
                continue
            T, ok = tg
            m = MID & ok
            if m.sum() < 2:
                continue
            p = pn.predict(net_input(rv.warp(img)))
            err = float(np.hypot(p.F[m] - T[m, 0], p.R[m] - T[m, 1]).mean() * rv.k)
            rows.append({"drive": os.path.basename(os.path.normpath(root)), "frame": i, "err_m": round(err, 3),
                         "conf": round(p.confidence, 3)})
    if not rows:
        print("測れる絵が無い(位置の記録が 65m 先まで続いているか確かめる)")
        return {}
    e = np.array([r["err_m"] for r in rows])
    c = np.array([r["conf"] for r in rows])
    res = {"frames": len(rows), "err_median_m": float(np.median(e)), "err_mean_m": float(e.mean()),
           "miss_2_5m": float((e > 2.5).mean()), "confident": float((c >= 0.5).mean()),
           "err_confident_m": float(e[c >= 0.5].mean()) if (c >= 0.5).any() else None,
           "err_unconfident_m": float(e[c < 0.5].mean()) if (c < 0.5).any() else None}
    if not quiet:
        print(f"絵 {res['frames']} 枚: ずれの中央 {res['err_median_m']:.2f}m・平均 {res['err_mean_m']:.2f}m・"
              f"2.5m 超 {res['miss_2_5m'] * 100:.0f}%")
        ec = res["err_confident_m"]
        eu = res["err_unconfident_m"]
        print(f"  自信 0.5 以上 {res['confident'] * 100:.0f}% の絵(ずれ {'-' if ec is None else f'{ec:.2f}m'})"
              f" / それ以外(ずれ {'-' if eu is None else f'{eu:.2f}m'})")
        for lo, hi in ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
            sel = (c >= lo) & (c < hi)
            if sel.any():
                print(f"  自信 {lo:.1f}〜{min(hi, 1.0):.1f}: {sel.sum():5d} 枚  ずれ平均 {e[sel].mean():.2f}m")
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drives", nargs="+", help="走行のフォルダ(frames/ と poses.csv)")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "models", "pathnet.onnx"))
    ap.add_argument("--config", default=None, help="カメラの設定(フォルダに camera.json が無いとき)")
    ap.add_argument("--stride", type=int, default=1, help="N 枚に 1 枚")
    ap.add_argument("--csv", default=None, help="1 枚ごとのずれを書く CSV")
    a = ap.parse_args()
    evaluate(a.model, a.drives, a.config, a.stride, a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
