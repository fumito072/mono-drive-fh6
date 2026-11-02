"""PathNet を自分の走行の記録で学び直す(続きから学ぶ・一から学ぶ)。人手のラベルは要らない。

    python tools/train_pathnet.py my_drive1 my_drive2 --config configs/my_car.json --out models/pathnet_my.onnx
    python tools/eval_pathnet.py my_drive_test --config configs/my_car.json --model models/pathnet_my.onnx

データ: 1 走行 = 1 フォルダ(frames/ と poses.csv、あれば camera.json)。形は docs/TRAINING.md。
教師: 各フレームから見た「その後に実際に通った線」(mono_drive/teacher.py)。絵はカメラの設定で基準の視点へ
作り直してから学ぶので、学んだモデルはそのまま MonoDriver で使える。
検証: 各走行の終わりの --val-frac(既定 15%)を学習に使わずに測る。最後に自信の物差し(sd_scale)を検証で合わせる。

要るもの: pip install -r requirements-train.txt(torch・onnx)。GPU(CUDA / Apple の MPS)があれば使う。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from mono_drive.camera import REF_CAMERA, ReferenceView  # noqa: E402
from mono_drive.pathnet import ARC_S, NET_H, NET_W, net_input, net_input_u8  # noqa: E402
from mono_drive.teacher import Drive, load_config_camera  # noqa: E402
from mono_drive.training_arguments import dataset_arguments, resolve_dataset_arguments
from mono_drive.training_refresh import HumanRevisionGuard,TrainingDataChanged
from mono_drive.training_curriculum import enabled as curriculum_enabled, sampling_weights, BestValidationCheckpoint

MID = (ARC_S >= 8) & (ARC_S <= 22)
#: 自信 0.5(横の標準偏差 1.55m)の所で、任せた絵のずれの平均をこれ以下にそろえる [m・基準の世界]
TRUST_ERR = 0.66


def load_items(roots: list[str], config: str | None, stride: int, val_frac: float, workers: int | None = None, refresh_check=None):
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    cam = load_config_camera(config)
    train, val = [], []
    workers=max(1,min(8,int(workers or os.environ.get('MONO_TRAIN_WORKERS','4'))))
    # Decode/warp independent images concurrently. map preserves chronological
    # order, and splitting still happens after validity filtering, as before.
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='training-images') as pool:
        for root in roots:
            if refresh_check is not None:refresh_check()
            d = Drive(root, cam)
            n = len(d)
            first=next((image for i in range(n) if (image:=cv2.imread(d.files[i])) is not None),None)
            rv=ReferenceView(d.camera,REF_CAMERA,frame_size=(first.shape[1],first.shape[0])) if first is not None else None
            def prepare_item(i):
                if rv is None:return None
                tg=d.target(i,rv)
                if tg is None:return None
                img=cv2.imread(d.files[i])
                if img is None:return None
                view=rv
                if (rv.cam.width,rv.cam.height)!=(img.shape[1],img.shape[0]):
                    view=ReferenceView(d.camera,REF_CAMERA,frame_size=(img.shape[1],img.shape[0]))
                    tg=d.target(i,view)
                    if tg is None:return None
                x=net_input_u8(view.warp(img))
                return x,tg[0],tg[1],view.k
            drive_items=[item for item in pool.map(prepare_item,range(0,n,max(1,stride))) if item is not None]
            cut = int(len(drive_items)*(1.0-val_frac))
            train.extend(drive_items[:cut])
            val.extend(drive_items[cut:])
            print(f"  {root}: {n} 枚 → 教師あり {len(drive_items)} 枚(距離の倍率 k = {rv.k if rv else float('nan'):.2f})", flush=True)
    return train, val


def stack(items):
    X = np.stack([it[0] for it in items])
    T = np.stack([it[1] for it in items]).astype(np.float32)
    OK = np.stack([it[2] for it in items])
    K = np.array([it[3] for it in items], np.float32)
    return X, T, OK, K


def loss_arc(out, T, ok, nll_cap: float):
    """平均は Huber、不確かさ(log 分散)は平均を止めた誤差から(元の学習と同じ)。"""
    import torch
    n = len(ARC_S)
    Fp, Rp, Up, lv = out[:, :n], out[:, n:2 * n], out[:, 2 * n:3 * n], out[:, 3 * n:]
    lv = lv.clamp(-6, 6)
    F, R, U = T[..., 0], T[..., 1], T[..., 2]
    w = ok.float()
    denom = w.sum().clamp(min=1)
    e2 = ((Fp.detach() - F) ** 2 + (Rp.detach() - R) ** 2).clamp(max=nll_cap ** 2)
    nll = 0.5 * (torch.exp(-lv) * e2 + lv)
    sl1 = torch.nn.functional.smooth_l1_loss
    hub = sl1(Fp, F, reduction="none", beta=0.5) + sl1(Rp, R, reduction="none", beta=0.5)
    hu = sl1(Up, U, reduction="none", beta=0.5)
    return ((nll + hub + 0.5 * hu) * w).sum() / denom


def augment(x, rng):
    """明るさ・コントラスト・色味を少し揺らす(形は変えない = 教師はそのまま)。x: (B, 3, H, W) 0..1。"""
    import torch
    B = x.shape[0]
    g = torch.from_numpy(rng.uniform(0.75, 1.25, (B, 1, 1, 1)).astype(np.float32)).to(x.device)
    c = torch.from_numpy(rng.uniform(0.8, 1.2, (B, 1, 1, 1)).astype(np.float32)).to(x.device)
    col = torch.from_numpy(rng.uniform(0.92, 1.08, (B, 3, 1, 1)).astype(np.float32)).to(x.device)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * c + m * g).mul(col).clamp(0, 1)


def predict(model, X, device, bs=128):
    import torch
    outs = []
    model.eval()
    with torch.no_grad():
        for a in range(0, len(X), bs):
            xb = torch.from_numpy(X[a:a + bs].astype(np.float32) / 255.0).to(device)
            outs.append(model(xb).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 44), np.float32)


def scores(out, T, OK, K):
    """→ (1 枚ごとのずれ [利用者の m], 1 枚ごとの 8〜22m の標準偏差の中央 [基準の m], 1 枚ごとのずれ [基準の m])。"""
    n = len(ARC_S)
    F, R, lv = out[:, :n], out[:, n:2 * n], out[:, 3 * n:]
    sd = np.exp(0.5 * np.clip(lv, -8, 6))
    m = MID[None, :] & OK
    d = np.hypot(F - T[..., 0], R - T[..., 1])
    e_ref = np.array([d[i][m[i]].mean() if m[i].any() else np.nan for i in range(len(d))])
    s = np.median(sd[:, MID], axis=1)
    return e_ref * K, s, e_ref


def calibrate(sd, err, thr: float = 1.55, target: float = TRUST_ERR, min_frac: float = 0.05) -> tuple[float, float]:
    """sd_scale: 自信のある順に任せて、任せた絵のずれの平均が target を超えない所まで任せる → その標準偏差を thr に合わせる。"""
    ok = np.isfinite(err)
    sd, err = sd[ok], err[ok]
    if len(sd) < 20:
        return 1.0, float("nan")
    order = np.argsort(sd, kind="stable")
    cm = np.cumsum(err[order]) / np.arange(1, len(order) + 1)
    good = np.flatnonzero(cm <= target)
    if len(good) == 0 or good[-1] < int(min_frac * len(order)):
        return 1.0, 0.0
    k = int(good[-1])
    return float(np.clip(thr / max(float(sd[order][k]), 1e-3), 0.2, 5.0)), float((k + 1) / len(order))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    dataset_arguments(ap)
    ap.add_argument("--config", default=None, help="カメラの設定(フォルダに camera.json が無いとき)")
    ap.add_argument("--out", required=True, help="書き出す ONNX(隣に .json も書く)")
    ap.add_argument("--init", default=os.path.join(os.path.dirname(HERE), "models", "pathnet.onnx"),
                    help="続きから学ぶ元のモデル(既定: 同梱の PathNet)")
    ap.add_argument("--scratch", action="store_true", help="一から学ぶ(データが数万枚以上あるとき)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None, help="学習率(既定: 続きから 2e-4・一から 1e-3)")
    ap.add_argument("--stride", type=int, default=1, help="N 枚に 1 枚")
    ap.add_argument("--val-frac", type=float, default=0.15, help="各走行の終わりのこの割合を検証に回す")
    ap.add_argument("--nll-cap", type=float, default=10.0, help="不確かさを学ぶ誤差の上限 [m]")
    ap.add_argument("--trust-err", type=float, default=TRUST_ERR,
                    help="自信 0.5 で任せた絵のずれの平均をこれにそろえる [m・基準の世界]")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curve-weight", type=float, default=1.0, help="Training-only curve sampling weight, bounded to 1..2")
    ap.add_argument('--balance-maneuvers',action='store_true',help='Bounded sampling of underrepresented bend directions')
    ap.add_argument('--human-revision',help='Refresh at a preprocessing/epoch boundary if verified human data changes')
    a = resolve_dataset_arguments(ap.parse_args(), ap)

    import torch
    from torch_models import build_pathnet, export_onnx, load_onnx_into, pick_device

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = pick_device(a.device)
    refresh=HumanRevisionGuard(Path(HERE).parent/'out/human-dataset.json',a.human_revision) if a.human_revision else None
    print(f"データを読む({len(a.drives)} 走行)…", flush=True)
    train, val = load_items(a.drives, a.config, a.stride, a.val_frac,refresh_check=refresh)
    if a.extra_validation:
        held,_=load_items(a.extra_validation,a.config,a.stride,0.,refresh_check=refresh)
        val.extend(held)
        del held
    if len(train) < 10:
        print("教師のある絵が少なすぎる(位置の記録が 65m 先まで続いているか、poses.csv の frame が合っているか)")
        return 1
    Xt, Tt, OKt, Kt = stack(train)
    del train
    if not 1. <= a.curve_weight <= 2.:
        raise ValueError('curve-weight must be within 1..2')
    # Draw only from the training partition; never oversample validation images.
    balanced=a.balance_maneuvers or curriculum_enabled(Path(HERE).parent)
    sampling,group_counts=sampling_weights(Tt,OKt,ARC_S,a.curve_weight,balanced)
    Xv, Tv, OKv, Kv = stack(val) if val else (None, None, None, None)
    del val
    print(f"Maneuver curriculum: {balanced}; training groups {group_counts}",flush=True)
    print(f"学習 {len(Xt)} 枚・検証 {0 if Xv is None else len(Xv)} 枚・{device}", flush=True)
    model = build_pathnet()
    if not a.scratch:
        load_onnx_into(model, a.init)
        print(f"続きから学ぶ: {a.init}")
    model = model.to(device)
    lr = a.lr or (1e-3 if a.scratch else 2e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = a.epochs * max(1, int(np.ceil(len(Xt) / a.batch)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    if Xv is not None and len(Xv):
        e0, _, _ = scores(predict(model, Xv, device), Tv, OKv, Kv)
        print(f"学ぶ前の検証のずれ: 平均 {np.nanmean(e0):.2f}m・中央 {np.nanmedian(e0):.2f}m", flush=True)
    else:
        e0 = None
    Tt_t = torch.from_numpy(Tt)
    OKt_t = torch.from_numpy(OKt)
    best=BestValidationCheckpoint()
    t0 = time.time()
    for ep in range(a.epochs):
        if refresh is not None:refresh()
        model.train()
        order = (rng.choice(len(Xt),len(Xt),replace=True,p=sampling) if a.curve_weight>1. or balanced
                 else rng.permutation(len(Xt)))
        tot, nb = 0.0, 0
        for b in range(0, len(order), a.batch):
            idx = order[b:b + a.batch]
            if len(idx) < 2:
                continue
            xb = torch.from_numpy(Xt[idx].astype(np.float32) / 255.0).to(device)
            xb = augment(xb, rng)
            loss = loss_arc(model(xb), Tt_t[idx].to(device), OKt_t[idx].to(device), a.nll_cap)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        msg = f"  {ep + 1}/{a.epochs}: 損失 {tot / max(1, nb):.3f}"
        if Xv is not None and len(Xv):
            e, _, _ = scores(predict(model, Xv, device), Tv, OKv, Kv)
            best.consider(model,float(np.nanmean(e)),ep+1)
            msg += f"  検証のずれ 平均 {np.nanmean(e):.2f}m・中央 {np.nanmedian(e):.2f}m"
        print(msg + f"({time.time() - t0:.0f} 秒)", flush=True)
    if best.restore(model):
        print(f"Restored best validation epoch {best.epoch}: {best.score:.4f}m",flush=True)
    # 自信の物差し(検証で。検証が少なければ学習の絵で)
    Xc, Tc, OKc, Kc = (Xv, Tv, OKv, Kv) if Xv is not None and len(Xv) >= 20 else (Xt, Tt, OKt, Kt)
    _, sd_raw, e_ref = scores(predict(model, Xc, device), Tc, OKc, Kc)
    sd_scale, usable = calibrate(sd_raw, e_ref, target=a.trust_err)
    print(f"自信の物差し sd_scale = {sd_scale:.3f}(同じ正確さで任せられる割合 {usable * 100:.0f}%)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    diff = export_onnx(model, a.out, "path", dynamic_batch=True)
    e_fin = None
    if Xv is not None and len(Xv):
        e_fin, _, _ = scores(predict(model.to(device), Xv, device), Tv, OKv, Kv)
    meta = {"name": os.path.splitext(os.path.basename(a.out))[0], "created": time.strftime("%Y-%m-%d %H:%M"),
            "format": "arc", "arc_s": ARC_S.tolist(), "input": [NET_H, NET_W], "sd_scale": sd_scale,
            "usable_frac": usable, "init": None if a.scratch else os.path.basename(a.init),
            "drives": [os.path.basename(os.path.normpath(r)) for r in a.drives], "frames_train": int(len(Xt)),
            "frames_val": 0 if Xv is None else int(len(Xv)),
            "val_err_before_m": None if e0 is None else round(float(np.nanmean(e0)), 3),
            "val_err_m": None if e_fin is None else round(float(np.nanmean(e_fin)), 3),
            "epochs": a.epochs, "lr": lr, "curve_weight": a.curve_weight,
            "best_validation_epoch":best.epoch,"balanced_maneuvers":balanced,
            "training_maneuver_counts":group_counts}
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"書いた: {a.out}(torch との差 {diff:.1e})と {os.path.splitext(a.out)[0]}.json")
    print("使うには: configs の models.pathnet をこのファイルにする")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TrainingDataChanged as error:
        print(str(error),flush=True)
        sys.exit(75)
