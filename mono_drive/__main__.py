"""動画・絵のフォルダ・カメラを入れて、走る線と速さを出す。

    python -m mono_drive --config configs/sample_c4.json --input samples/c4_sweden/frames \\
        --speed-csv samples/c4_sweden/speed.csv --out out/sample

出力(--out): decisions.csv(1 枚ごとの答え)・overlay.mp4(重ね描き)・summary.json(まとめ)。
--send udp://127.0.0.1:5005 で 1 枚ごとの答えを JSON で送る(車両・ロボットの制御プログラムへ)。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time
from collections import Counter


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mono_drive", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="設定の JSON(configs/*.json)")
    ap.add_argument("--input", required=True, help="絵のフォルダ・動画ファイル・カメラ番号(0)・ストリームの URL")
    ap.add_argument("--out", default=None, help="結果を書くフォルダ(無ければ書かない)")
    ap.add_argument("--speed", type=float, default=None, help="今の速さ [km/h](一定。無ければ目標の速さで走るとみなす)")
    ap.add_argument("--speed-csv", default=None,
                    help="今の速さの CSV(列 kmh と、frame か t。あればヨーレート yaw_rate_dps [deg/s・左回りが正] も)")
    ap.add_argument("--fps", type=float, default=None, help="絵のフォルダの 1 秒あたりの枚数(既定 10。--speed-csv に t があればそれ)")
    ap.add_argument("--start", type=int, default=0, help="この番号の絵から")
    ap.add_argument("--max-frames", type=int, default=0, help="この枚数で止める(0 = 全部)")
    ap.add_argument("--every", type=int, default=1, help="N 枚に 1 枚だけ使う")
    ap.add_argument("--no-depth", action="store_true", help="深度を使わない(速いが、壁・崖を見ない)")
    ap.add_argument("--no-obstacle", action="store_true", help="学習した障害物の検出を使わない")
    ap.add_argument("--no-video", action="store_true", help="overlay.mp4 を書かない")
    ap.add_argument("--no-bev", action="store_true", help="上から見た図を付けない")
    ap.add_argument("--video-width", type=int, default=1280, help="overlay.mp4 の横幅の上限")
    ap.add_argument("--show", action="store_true", help="窓に出す(q で止める)")
    ap.add_argument("--send", default=None, help="答えを送る先 udp://ホスト:ポート")
    ap.add_argument("--device", default=None, help="auto | cpu | cuda | coreml")
    a = ap.parse_args(argv)

    import cv2

    from .driver import MonoDriver
    from .sources import FrameSource, SpeedSource
    from .viz import draw

    drv = MonoDriver(a.config, device=a.device, use_depth=False if a.no_depth else None,
                     use_obstacle=False if a.no_obstacle else None)
    src = FrameSource(a.input, fps=a.fps, times_csv=a.speed_csv)
    spd = SpeedSource(a.speed, a.speed_csv)
    print(f"入力: {a.input}({len(src) or '?'} 枚・{src.fps:.1f} 枚/秒)")
    print(f"カメラ: {drv.camera.name or '(名前なし)'} {drv.camera.width}x{drv.camera.height}")
    print("モデル: PathNet" + (" + 障害物" if drv.obstacle else "") + (" + 深度" if drv.depth else "")
          + (" + 矢印板" if drv.use_chevrons else ""))
    sock, dest = None, None
    if a.send:
        if not a.send.startswith("udp://"):
            print("--send は udp://ホスト:ポート の形で", file=sys.stderr)
            return 2
        host, port = a.send[len("udp://"):].rsplit(":", 1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dest = (host, int(port))
    writer = None
    fcsv = wcsv = None
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fcsv = open(os.path.join(a.out, "decisions.csv"), "w", newline="", encoding="utf-8")
    modes = Counter()
    n = 0
    t_start = time.perf_counter()
    first_cov = None
    try:
        for i, t, frame in src:
            if i < a.start or (i - a.start) % max(1, a.every):
                continue
            kmh = spd.at(i, t)
            dec = drv.step(frame, speed_kmh=kmh, t=t, yaw_rate_dps=spd.yaw_rate(i, t))
            if first_cov is None:
                first_cov = drv.rv.coverage
                if first_cov < 0.95:
                    print(f"  注意: カメラの絵がモデルの見る範囲の {first_cov * 100:.0f}% しか埋めていない"
                          "(画角が狭い・向きが違う。docs/CAMERA.md)")
            modes[dec.mode] += 1
            row = dec.to_row()
            row = {"frame": i, "speed_kmh": "" if kmh is None else round(kmh, 1), **row}
            if fcsv is not None:
                if wcsv is None:
                    wcsv = csv.DictWriter(fcsv, fieldnames=list(row.keys()))
                    wcsv.writeheader()
                wcsv.writerow(row)
            if sock is not None:
                msg = dec.to_json()
                msg["frame"] = i
                sock.sendto(json.dumps(msg, ensure_ascii=False).encode("utf-8"), dest)
            if (a.out and not a.no_video) or a.show:
                img = draw(frame, dec, drv, bev=not a.no_bev)
                if img.shape[1] > a.video_width:
                    sc = a.video_width / img.shape[1]
                    img = cv2.resize(img, (a.video_width, int(round(img.shape[0] * sc / 2)) * 2),
                                     interpolation=cv2.INTER_AREA)
                if a.out and not a.no_video:
                    if writer is None:
                        fps_out = src.fps / max(1, a.every)
                        writer = cv2.VideoWriter(os.path.join(a.out, "overlay.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps_out,
                                                 (img.shape[1], img.shape[0]))
                    writer.write(img)
                if a.show:
                    cv2.imshow("mono_drive", img)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
            n += 1
            if n % 50 == 0:
                el = time.perf_counter() - t_start
                print(f"  {n} 枚({n / el:.1f} 枚/秒)  {dec.mode:7s} {dec.target_kmh:5.1f} km/h  "
                      f"ハンドル {dec.steer:+.2f}  自信 {dec.confidence:.2f}")
            if a.max_frames and n >= a.max_frames:
                break
    finally:
        src.close()
        if writer is not None:
            writer.release()
        if fcsv is not None:
            fcsv.close()
        if a.show:
            cv2.destroyAllWindows()
    el = time.perf_counter() - t_start
    summary = {"frames": n, "seconds": round(el, 2), "fps": round(n / max(el, 1e-6), 2),
               "modes": dict(modes), "ms_per_step": {k: round(v, 1) for k, v in drv.ms.items()},
               "coverage": None if first_cov is None else round(first_cov, 3),
               "providers": {"depth": drv.depth.providers if drv.depth else None}}
    print(f"終わり: {n} 枚・{summary['fps']} 枚/秒・{dict(modes)}")
    print("  1 枚あたり [ms]: " + "  ".join(f"{k} {v:.1f}" for k, v in drv.ms.items()))
    if a.out:
        with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"  書いた: {a.out}/decisions.csv" + ("" if a.no_video else f"・{a.out}/overlay.mp4")
              + f"・{a.out}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
