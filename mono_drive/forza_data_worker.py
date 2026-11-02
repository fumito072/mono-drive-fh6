"""Prepare incoming recordings independently of GPU candidate training."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from .forza_local_learning import read_json, write_json
from .forza_session_policy import session_active

ROOT = Path(__file__).resolve().parents[1]


def acquire_guard(port):
    guard = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if os.name == 'nt':
        guard.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        guard.bind(('127.0.0.1', port))
        return guard
    except OSError:
        guard.close()
        return None


def active(root=ROOT):
    out = root/'out'
    return (not (out/'longrun.stop').exists() and not (out/'training.stop').exists() and
            session_active(read_json(out/'longrun.json', {}), time.time()))


def main():
    guard = acquire_guard(20074)
    if guard is None:
        return 0
    out = ROOT/'out'
    try:
        while active():
            started = time.time()
            command = [sys.executable, 'tools/prepare_human_driving.py']
            flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == 'nt' else 0
            with (out/'human-data-worker.log').open('a') as log, (out/'human-data-worker.err').open('a') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=err,
                                         env=dict(os.environ, PYTHONIOENCODING='utf-8'), creationflags=flags)
                while child.poll() is None:
                    write_json(out/'human-data-worker-state.json', dict(time=time.time(), pid=os.getpid(),
                               phase='preparing', started_at=started, child_pid=child.pid))
                    if not active():
                        from .forza_supervisor import stop_worker
                        stop_worker(child)
                        break
                    time.sleep(1.)
            data = read_json(out/'human-dataset.json', {})
            for _ in range(60):
                write_json(out/'human-data-worker-state.json', dict(time=time.time(), pid=os.getpid(),
                           phase='watching' if child.returncode == 0 else 'retry_later',
                           last_exit_code=child.returncode, dataset_revision=data.get('revision'),
                           frames=data.get('frames', 0), pending=len(data.get('pending', []))))
                if not active():
                    return 0
                time.sleep(1.)
    finally:
        guard.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
