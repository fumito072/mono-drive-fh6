"""障害物の検出(ObstacleNet)を、自分の絵と塗り(マスク)で学び直す。

    python tools/train_obstacle.py my_set1 my_set2 --config configs/my_car.json --out models/obstacle_my.onnx

データ: 1 組 = 1 フォルダ
    <フォルダ>/frames/xxx.jpg      絵
    <フォルダ>/masks/xxx.png       同じ名前の塗り(0 以外 = 当たる物。人・車・岩・柵・コーン など)。塗りの無い絵は「何も無い」
    <フォルダ>/camera.json         (無ければ --config のカメラ)
塗りは手で描いても、ほかの検出器(セグメンテーション)の出力でもよい。絵と塗りは基準の視点へ作り直してから学ぶ。
出力は 32x80 マス(1 マス = 基準の視点の 7.5 行 x 7.2 列)の「当たる物がある」。マスの 25% 以上が塗られていれば「ある」。

検証(各フォルダの終わりの --val-frac)で、「何も無い絵で 2 マス以上鳴る割合」が --max-false(既定 3%)以下になる
いちばん低いしきい値を選んで JSON に書く(MonoDriver はそれを使う)。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from mono_drive.camera import NET_ROWS, REF_CAMERA, Camera, ReferenceView  # noqa: E402
from mono_drive.obstacle import MEAN, STD  # noqa: E402
from mono_drive.pathnet import net_input  # noqa: E402
from mono_drive.sources import IMG_EXT  # noqa: E402
from mono_drive.teacher import load_config_camera  # noqa: E402

GH, GW = 32, 80
CELL_FRAC = 0.25


def cells_of(mask_ref: np.ndarray, valid_ref: np.ndarray):
    """基準の視点の塗り (360, 576) → 32x80 の (教師 0/1, 重み 0/1)。"""
    import cv2
    y0, y1 = NET_ROWS
    m = mask_ref[y0:y1].astype(np.float32)
    v = valid_ref[y0:y1].astype(np.float32)
    frac = cv2.resize(m, (GW, GH), interpolation=cv2.INTER_AREA)
    vfrac = cv2.resize(v, (GW, GH), interpolation=cv2.INTER_AREA)
    return (frac >= CELL_FRAC).astype(np.float32), (vfrac >= 0.5).astype(np.float32)


def load_items(roots, config, val_frac):
    import cv2
    cam0 = load_config_camera(config)
    train, val = [], []
    for root in roots:
        files = sorted(p for p in glob.glob(os.path.join(root, "frames", "*")) if p.lower().endswith(IMG_EXT))
        cp = os.path.join(root, "camera.json")
        cam = Camera.load(cp) if os.path.exists(cp) else cam0
        if cam is None:
            raise ValueError(f"{root}: camera.json も --config も無い")
        cut = int(len(files) * (1.0 - val_frac))
        rv = None
        n_pos = 0
        for i, f in enumerate(files):
            img = cv2.imread(f)
            if img is None:
                continue
            if rv is None or (rv.cam.width, rv.cam.height) != (img.shape[1], img.shape[0]):
                rv = ReferenceView(cam, REF_CAMERA, frame_size=(img.shape[1], img.shape[0]))
            stem = os.path.splitext(os.path.basename(f))[0]
            mp = os.path.join(root, "masks", stem + ".png")
            mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if os.path.exists(mp) else None
            if mk is None:
                mk = np.zeros(img.shape[:2], np.uint8)
            if mk.shape != img.shape[:2]:
                mk = cv2.resize(mk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mk_ref = rv.warp(((mk > 0) * 255).astype(np.uint8)[..., None].repeat(3, 2))[..., 0] > 127
            y, w = cells_of(mk_ref & rv.valid, rv.valid)
            x = (net_input(rv.warp(img)) * 255.0).round().astype(np.uint8)
            (val if i >= cut else train).append((x, y, w))
            n_pos += int(y.sum() > 0)
        print(f"  {root}: {len(files)} 枚(物のある絵 {n_pos} 枚)", flush=True)
    return train, val


def to_input(xb_u8, device):
    import torch
    x = xb_u8.astype(np.float32) / 255.0
    x = (x - MEAN[None]) / STD[None]
    return torch.from_numpy(x.astype(np.float32)).to(device)


def predict(model, X, device, bs=64):
    import torch
    out = []
    model.eval()
    with torch.no_grad():
        for a in range(0, len(X), bs):
            out.append(torch.sigmoid(model(to_input(X[a:a + bs], device))).float().cpu().numpy()[:, 0])
    return np.concatenate(out) if out else np.zeros((0, GH, GW), np.float32)


def pick_threshold(P, Y, W, max_false: float):
    """何も無い絵で 2 マス以上鳴る割合が max_false 以下になる、いちばん低いしきい値。"""
    clean = (Y * W).reshape(len(Y), -1).sum(1) == 0
    has = ~clean
    if clean.sum() < 3:                     # 何も無い絵が無いと、鳴りすぎを測れない
        return None
    best = None
    for thr in np.arange(0.2, 0.99, 0.02):
        on = (P > thr) & (W > 0)
        n_on = on.reshape(len(P), -1).sum(1)
        fa = float((n_on[clean] >= 2).mean()) if clean.any() else 0.0
        hit = float(((on & (Y > 0)).reshape(len(P), -1).sum(1)[has] >= 1).mean()) if has.any() else float("nan")
        if fa <= max_false:
            best = (float(thr), fa, hit)
            break
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="+", help="フォルダ(frames/ と masks/)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=os.path.join(os.path.dirname(HERE), "models", "obstacle.onnx"))
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-false", type=float, default=0.03, help="何も無い絵で鳴ってよい割合")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from torch_models import build_obstacle, export_onnx, load_onnx_into, pick_device

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = pick_device(a.device)
    print("データを読む…", flush=True)
    train, val = load_items(a.sets, a.config, a.val_frac)
    if len(train) < 8:
        print("絵が少なすぎる")
        return 1
    Xt = np.stack([t[0] for t in train]); Yt = np.stack([t[1] for t in train]); Wt = np.stack([t[2] for t in train])
    Xv = np.stack([t[0] for t in val]) if val else None
    Yv = np.stack([t[1] for t in val]) if val else None
    Wv = np.stack([t[2] for t in val]) if val else None
    pos = float((Yt * Wt).sum())
    neg = float(((1 - Yt) * Wt).sum())
    pw = float(np.clip(neg / max(pos, 1.0), 1.0, 20.0))
    print(f"学習 {len(Xt)} 枚・検証 {0 if Xv is None else len(Xv)} 枚・物のマス {pos:.0f}・重み {pw:.1f}・{device}")
    model = build_obstacle()
    if not a.scratch:
        load_onnx_into(model, a.init)
        print(f"続きから学ぶ: {a.init}")
    model = model.to(device)
    lr = a.lr or (1e-3 if a.scratch else 2e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pw, device=device))
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        order = rng.permutation(len(Xt))
        tot, nb = 0.0, 0
        for b in range(0, len(order), a.batch):
            idx = order[b:b + a.batch]
            if len(idx) < 2:
                continue
            xb = to_input(Xt[idx], device)
            g = torch.from_numpy(rng.uniform(0.8, 1.2, (len(idx), 1, 1, 1)).astype(np.float32)).to(device)
            xb = xb * g
            yb = torch.from_numpy(Yt[idx][:, None]).to(device)
            wb = torch.from_numpy(Wt[idx][:, None]).to(device)
            loss = (bce(model(xb), yb) * wb).sum() / wb.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        print(f"  {ep + 1}/{a.epochs}: 損失 {tot / max(1, nb):.4f}({time.time() - t0:.0f} 秒)", flush=True)
    thr_info = None
    if Xv is not None and len(Xv) >= 5:
        thr_info = pick_threshold(predict(model, Xv, device), Yv, Wv, a.max_false)
    thr = thr_info[0] if thr_info else 0.84
    if thr_info:
        print(f"しきい値 {thr:.2f}(何も無い絵で鳴る {thr_info[1] * 100:.1f}%・物のある絵で当てた {thr_info[2] * 100:.0f}%)")
    else:
        print("検証に「何も無い絵」が足りないので、しきい値は 0.84 のまま(塗りの無い絵も混ぜると選べる)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    diff = export_onnx(model, a.out, "obstacle_logit", dynamic_batch=False)
    meta = {"name": os.path.splitext(os.path.basename(a.out))[0], "created": time.strftime("%Y-%m-%d %H:%M"),
            "input": [1, 3, 128, 320], "output": [1, 1, GH, GW], "threshold": round(thr, 3),
            "init": None if a.scratch else os.path.basename(a.init),
            "sets": [os.path.basename(os.path.normpath(r)) for r in a.sets], "frames_train": int(len(Xt)),
            "val": None if not thr_info else {"false_alarm": thr_info[1], "hit": thr_info[2]}}
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"書いた: {a.out}(torch との差 {diff:.1e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
