"""Python から使う最小の例: 動画を 1 枚ずつ MonoDriver に通して、答えを表示する。

    python examples/api_example.py samples/sweden_unseen/frames
    python examples/api_example.py my_video.mp4 --config configs/my_car.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mono_drive import MonoDriver  # noqa: E402
from mono_drive.sources import FrameSource  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="動画・絵のフォルダ・カメラ番号")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "sample_c4.json"))
    ap.add_argument("--speed", type=float, default=40.0, help="今の速さ [km/h](実車なら車から読む)")
    a = ap.parse_args()

    drv = MonoDriver(a.config)
    for i, t, frame in FrameSource(a.input, fps=5.0):
        dec = drv.step(frame, speed_kmh=a.speed, t=t)
        # ここで車両へ送る: 切れ角 dec.steer_deg(または -1..1 の dec.steer)、目標の速さ dec.target_kmh、下がる dec.reverse
        print(f"{i:4d}  {dec.mode:7s}  切れ角 {dec.steer_deg:+6.1f}°  目標 {dec.target_kmh:5.1f} km/h  "
              f"自信 {dec.confidence:.2f}  理由: {dec.limiter}")
        if i >= 30:
            break


if __name__ == "__main__":
    main()
