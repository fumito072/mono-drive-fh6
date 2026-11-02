"""コースに置かれた矢印板(シェブロン「<<<」「>>>」)を読んで、曲がる向きを知る(学習なし・色に依らない)。

**読み方** — 「<」は上半分の縞が「/」、下半分が「\\」。「>」はその逆。輪郭の傾きの向き(Sobel の gx·gy:
「/」で正、「\\」で負)を窓の中で平均し、傾きの揃い具合(コヒーレンス)で見る:
    C = Σ gx·gy / Σ (gx² + gy²)      … -0.5(\\ だけ)〜 +0.5(/ だけ)
上半分で C > 0 かつ下半分で C < 0 なら「<」(左)、逆なら「>」(右)。赤白・黒白・黄黒のどれでも同じ。
木の葉は向きがばらばらで C ≈ 0、テープや電線は上下で傾きが変わらないので反応しない。

基準の視点(576x360)の絵で探す(大きさ・探す範囲の定数はこの絵に合わせてある)。
矢印板は曲がり角・分かれ道の **外側** に、曲がる向きを指して置かれる → その向きへ曲がる(planner.py)。
"""

from __future__ import annotations

import numpy as np

#: 探す板の高さ(学習時の保存画像 374 行に対する割合)。半分の高さが窓の縦
SCALES = (0.035, 0.055, 0.085)
_H_EQ = 374.0               # 上の割合を測った絵の高さ(タイトルバー 14 行を含む)
_TITLE = 14.0
#: 上半分・下半分それぞれの傾きの揃い具合の下限(明るさ・赤み)と、縞の強さ(平均の勾配の大きさ)の下限
COH_MIN = (0.33, 0.25)
EDGE_MIN = 90.0
#: 見る範囲(保存画像の高さの割合): 板は地平線から道の近くに立つ。上は空・梢、下はダッシュボード
TOP_FRAC, BOTTOM_FRAC = 0.33, 0.68
#: 縞の数: 明暗が入れ替わる回数の下限(矢印 2 つ以上 = 4 回以上)。木の縁や枝は 1〜2 回
MIN_CROSS = 4


def _channels(img_bgr) -> list[np.ndarray]:
    f = img_bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    red = np.clip(r - 0.5 * (g + b), 0, None) * 2.0
    return [gray, red]


def _coherence_maps(ch: np.ndarray, hh: int, ww: int):
    import cv2
    gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
    p = cv2.boxFilter(gx * gy, -1, (ww, hh), normalize=True)
    e = np.maximum(cv2.boxFilter(gx * gx + gy * gy, -1, (ww, hh), normalize=True), 0.0)
    return p / (e + 1e-3), np.sqrt(e)


def _crossings(row: np.ndarray) -> int:
    r = row - row.mean()
    thr = 0.25 * (np.abs(r).max() + 1e-6)
    s = np.sign(np.where(np.abs(r) > thr, r, 0.0))
    s = s[s != 0]
    return int((s[1:] != s[:-1]).sum()) if len(s) > 1 else 0


def find_chevrons(ref_bgr, body_mask=None) -> list[dict]:
    """基準の視点の絵 → [{"box": (x0, y0, x1, y1), "dir": -1 左 / +1 右, "conf", "w", "cx"}]。"""
    import cv2
    H, W = ref_bgr.shape[:2]
    sy = H / (_H_EQ - _TITLE)                       # 基準の視点 360 行なら 1
    y_top = max(0, int(TOP_FRAC * _H_EQ - _TITLE) * sy)
    y_bot = min(H, int(BOTTOM_FRAC * _H_EQ - _TITLE) * sy)
    y_top, y_bot = int(y_top), int(y_bot)
    img = ref_bgr[y_top:y_bot]
    chans = _channels(img)
    cands = []
    for sc in SCALES:
        hh = max(3, int(sc * _H_EQ * sy / 2))
        ww = max(5, int(2.2 * hh))
        for ch, cmin in zip(chans, COH_MIN):
            C, E = _coherence_maps(ch, hh, ww)
            up = np.roll(C, hh // 2 + 1, axis=0)
            lo = np.roll(C, -(hh // 2 + 1), axis=0)
            eu = np.roll(E, hh // 2 + 1, axis=0)
            el = np.roll(E, -(hh // 2 + 1), axis=0)
            strong = (eu > EDGE_MIN) & (el > EDGE_MIN)
            left = np.minimum(up, -lo)               # 「<」: 上 /(正)・下 \\(負)
            right = np.minimum(-up, lo)              # 「>」
            for d, Rm in ((-1, left), (1, right)):
                Rm = np.where(strong, Rm, 0.0)
                Rm[:hh + 2] = 0
                Rm[-hh - 2:] = 0
                m = Rm > cmin
                if not m.any():
                    continue
                n, lab, stats, cent = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
                for i in range(1, n):
                    x, y, w, h, area = stats[i]
                    if area < max(4, (hh * ww) // 20):
                        continue
                    cx, cy = cent[i]
                    peak = float(Rm[lab == i].max())
                    xa, xb = int(max(0, cx - ww)), int(min(ch.shape[1], cx + ww))
                    ya = int(min(ch.shape[0] - 1, max(0, cy - hh // 2)))
                    yb = int(min(ch.shape[0] - 1, max(0, cy + hh // 2)))
                    if min(_crossings(ch[ya, xa:xb]), _crossings(ch[yb, xa:xb])) < MIN_CROSS:
                        continue
                    x0, x1 = int(max(0, cx - ww)), int(min(W, cx + ww))
                    y0, y1 = int(cy - hh), int(cy + hh)
                    cands.append({"box": (x0, y0 + y_top, x1, y1 + y_top), "dir": d, "conf": peak,
                                  "w": (x1 - x0) / W, "cx": cx / W, "scale": sc})
    cands.sort(key=lambda c: -c["conf"])
    out: list[dict] = []
    for c in cands:
        x0, y0, x1, y1 = c["box"]
        if body_mask is not None and body_mask.shape[:2] == (H, W):
            yc, xc = min(H - 1, (y0 + y1) // 2), min(W - 1, (x0 + x1) // 2)
            if body_mask[yc, xc]:
                continue
        dup = None
        for o in out:
            a0, b0, a1, b1 = o["box"]
            ix = max(0, min(x1, a1) - max(x0, a0))
            iy = max(0, min(y1, b1) - max(y0, b0))
            if ix * iy > 0.3 * min((x1 - x0) * (y1 - y0), (a1 - a0) * (b1 - b0)):
                dup = o
                break
        if dup is None:
            out.append(c)
        elif dup["dir"] != c["dir"] and c["conf"] > 0.8 * dup["conf"]:
            dup["conflict"] = True
    return [o for o in out if not o.get("conflict")]


class ChevronTracker:
    """2 回続けて同じ向きに読めたら、その向きを hold_s 秒覚える(1 枚だけの読み違いで曲がらない)。"""

    def __init__(self, need: int = 2, hold_s: float = 3.0):
        self.need, self.hold_s = int(need), float(hold_s)
        self.hist: list[int] = []
        self.active = None          # {"dir", "t", "box", "conf"}

    def update(self, cues: list[dict], t: float) -> dict | None:
        best = max(cues, key=lambda c: c["conf"]) if cues else None
        self.hist = (self.hist + [best["dir"] if best else 0])[-self.need:]
        if best and len(self.hist) == self.need and all(h == best["dir"] for h in self.hist):
            self.active = {"dir": int(best["dir"]), "t": t, "box": best["box"], "conf": best["conf"]}
        if self.active and t - self.active["t"] > self.hold_s:
            self.active = None
        return self.active
