"""入力: 絵のフォルダ・動画ファイル・カメラ(番号)・ストリームの URL。どれも (番号, 時刻 [s], BGR の絵) を順に返す。"""

from __future__ import annotations

import csv
import glob
import os
import time

import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_table(path: str) -> dict[str, np.ndarray]:
    """CSV(見出しあり)→ 列名 → 数の配列(数でない列は捨てる)。"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = {}
    if not rows:
        return out
    for k in rows[0]:
        try:
            out[k.strip()] = np.array([float(r[k]) for r in rows])
        except (TypeError, ValueError):
            pass
    return out


class FrameSource:
    def __init__(self, spec: str, fps: float | None = None, times_csv: str | None = None):
        self.spec = str(spec)
        self.live = False
        self.cap = None
        self.files: list[str] = []
        self.fps = fps
        self.times = None
        if os.path.isdir(self.spec):
            self.files = sorted(p for p in glob.glob(os.path.join(self.spec, "*")) if p.lower().endswith(IMG_EXT))
            if not self.files:
                raise FileNotFoundError(f"{self.spec} に絵(jpg/png)が無い")
            self.fps = fps or 10.0
            if times_csv and os.path.exists(times_csv):
                tb = read_table(times_csv)
                if "t" in tb:
                    n = len(self.files)
                    if "frame" in tb:
                        order = np.argsort(tb["frame"])
                        fr, tt = tb["frame"][order], tb["t"][order]
                        self.times = np.interp(np.arange(n), fr, tt)
                    elif len(tb["t"]) >= n:
                        self.times = tb["t"][:n]
                if self.times is not None and fps is None and len(self.times) > 2:
                    dt = float(np.median(np.diff(self.times)))
                    if dt > 0:
                        self.fps = 1.0 / dt
        else:
            import cv2
            src = int(self.spec) if self.spec.isdigit() else self.spec
            self.live = isinstance(src, int) or "://" in self.spec
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise FileNotFoundError(f"開けない: {self.spec}")
            vf = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps or (vf if vf and vf > 1 else 30.0)

    def __len__(self) -> int:
        if self.files:
            return len(self.files)
        if self.cap is not None and not self.live:
            import cv2
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def __iter__(self):
        import cv2
        if self.files:
            for i, p in enumerate(self.files):
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                t = float(self.times[i]) if self.times is not None else i / self.fps
                yield i, t, img
            return
        i = 0
        t0 = time.time()
        while True:
            ok, img = self.cap.read()
            if not ok or img is None:
                break
            t = time.time() - t0 if self.live else i / self.fps
            yield i, t, img
            i += 1

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class SpeedSource:
    """今の速さ [km/h]: 一定の値か、CSV(列 kmh と、t か frame)。"""

    def __init__(self, const: float | None = None, csv_path: str | None = None):
        self.const = const
        self.tab = read_table(csv_path) if csv_path else None
        if self.tab is not None and "kmh" not in self.tab:
            raise ValueError(f"{csv_path} に kmh の列が要る")

    def yaw_rate(self, i: int, t: float) -> float | None:
        """ヨーレート [deg/s](CSV に yaw_rate_dps の列があれば)。"""
        if self.tab is None or "yaw_rate_dps" not in self.tab:
            return None
        if "frame" in self.tab:
            return float(np.interp(i, self.tab["frame"], self.tab["yaw_rate_dps"]))
        if "t" in self.tab:
            return float(np.interp(t, self.tab["t"], self.tab["yaw_rate_dps"]))
        return None

    def at(self, i: int, t: float) -> float | None:
        if self.tab is not None:
            if "frame" in self.tab:
                return float(np.interp(i, self.tab["frame"], self.tab["kmh"]))
            if "t" in self.tab:
                return float(np.interp(t, self.tab["t"], self.tab["kmh"]))
            return float(self.tab["kmh"][min(i, len(self.tab["kmh"]) - 1)])
        return self.const
