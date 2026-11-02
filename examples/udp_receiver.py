"""車両・ロボット側の受け取りの例(`python -m mono_drive ... --send udp://127.0.0.1:5005` の相手)。

    python examples/udp_receiver.py --port 5005

1 枚ごとに JSON が 1 つ届く(中身は README の「答えの中身」と同じ名前)。この例は表示するだけ。実際の車両では
`apply()` の中で切れ角と速さの指令に直す。**安全の決まりはこちら側で持つ**:
  - 一定時間(TIMEOUT_S)届かなければ止める
  - 自信が低い・見失った(mode = lost)なら止める・遅くする
  - 速さの上限・切れ角の上限・非常停止は車両側で必ず別に持つ
"""

import argparse
import json
import socket
import time

TIMEOUT_S = 0.3          # これより長く届かなければ止める
MIN_CONF = 0.5           # 自信がこれ未満なら止める(実験の初めは高めに)


def apply(steer_deg: float, target_kmh: float, reverse: bool) -> None:
    """ここを車両の指令に置き換える(例: サーボの角度・モーターの速さ・CAN の送信)。"""
    print(f"  → 切れ角 {steer_deg:+6.1f}°  速さ {target_kmh:5.1f} km/h  {'後ろへ' if reverse else ''}")


def stop() -> None:
    print("  → 止める")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5005)
    a = ap.parse_args()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((a.host, a.port))
    s.settimeout(TIMEOUT_S)
    print(f"受け取りを待つ udp://{a.host}:{a.port}")
    last = time.time()
    while True:
        try:
            data, _ = s.recvfrom(65536)
        except socket.timeout:
            if time.time() - last > TIMEOUT_S:
                stop()
            continue
        last = time.time()
        d = json.loads(data.decode("utf-8"))
        print(f"{d.get('frame', '?')}: {d['mode']} 自信 {d['confidence']:.2f} 理由 {d['limiter']}")
        if d["mode"] == "lost" or d["confidence"] < MIN_CONF:
            stop()
            continue
        apply(d["steer_deg"], d["target_kmh"], bool(d["reverse"]))


if __name__ == "__main__":
    main()
