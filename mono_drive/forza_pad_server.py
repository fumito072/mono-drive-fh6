"""Own the virtual controller independently of AI workers and their updates."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import socket
import time
from .forza_background import DirectGamepad


def validate(data):
    command = tuple(float(x) for x in data['command'])
    if len(command) != 3 or not all(math.isfinite(x) for x in command):
        raise ValueError('Invalid axes')
    if not (-1 <= command[0] <= 1 and 0 <= command[1] <= 1 and 0 <= command[2] <= 1):
        raise ValueError('Axis outside range')
    return command, bool(data.get('handbrake', False))


def main(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        sock.bind(('127.0.0.1', 20067))
    except OSError:
        return 0
    sock.settimeout(.05)
    pad = DirectGamepad()
    received, report = 0., 0.
    print('Persistent controller connected, PID', os.getpid(), flush=True)
    try:
        while not (root/'out'/'controller-host.stop').exists():
            try:
                payload, address = sock.recvfrom(2048)
                data = json.loads(payload)
                if data.get('hello'):
                    sock.sendto(json.dumps(dict(ready=True, pid=os.getpid())).encode(), address)
                else:
                    command, brake = validate(data)
                    buttons = data.get('buttons', [])
                    allowed = {'A','B','X','Y','START','BACK','LEFT_THUMB','RIGHT_THUMB',
                               'LEFT_SHOULDER','RIGHT_SHOULDER','DPAD_UP','DPAD_DOWN','DPAD_LEFT','DPAD_RIGHT'}
                    if not isinstance(buttons,list) or any(b not in allowed for b in buttons):
                        raise ValueError('Invalid controller buttons')
                    if buttons and any(abs(x) > .001 for x in command):
                        raise ValueError('Menu buttons require neutral axes')
                    ui_stick = data.get('ui_stick')
                    if ui_stick is not None:
                        if (len(ui_stick)!=2 or any(not math.isfinite(float(v)) or abs(float(v))>1 for v in ui_stick)
                                or any(abs(v)>.001 for v in command)):
                            raise ValueError('UI stick requires neutral driving axes')
                    pad.set(command, brake, buttons, ui_stick=ui_stick)
                    received = time.monotonic()
            except (ValueError, KeyError, TypeError):
                continue
            except (socket.timeout,ConnectionResetError):
                # Windows UDP can deliver an ICMP error for an earlier reply
                # to a client that has already closed. Keep the device alive.
                pass
            now = time.monotonic()
            if now-received > .5:
                pad.set((0., 0., 0.))
            if now-report > 1:
                status = dict(pid=os.getpid(), connected=True, last_input_age_s=round(now-received, 2))
                temp = root/'out'/'controller-status.tmp'
                try:
                    temp.write_text(json.dumps(status), encoding='utf-8')
                    temp.replace(root/'out'/'controller-status.json')
                except OSError:
                    # A locked status file must never disconnect the pad.
                    pass
                report = now
    finally:
        pad.close()
        sock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
