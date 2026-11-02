"""Dedicated AI start/stop channel. Does not inject keyboard or mouse events."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from .forza_speed import MAX_CRUISE_KMH, cruise_limit

ROOT = Path(__file__).resolve().parents[1]


def request(action, limit=None):
    limit = cruise_limit(ROOT) if limit is None else float(limit)
    if action not in ('drive','hold') or not 1 <= limit <= MAX_CRUISE_KMH:
        raise ValueError('Invalid dedicated control request')
    target = ROOT/'out'/'control.json'
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix(f'.{os.getpid()}.new')
    temporary.write_text(json.dumps(dict(action=action,max_kmh=limit)),encoding='utf-8')
    temporary.replace(target)
    if action == 'hold':
        (ROOT/'out'/'longrun.stop').write_text('Dedicated stop requested',encoding='utf-8')
    if action == 'drive':
        (ROOT/'out'/'supervisor.stop').unlink(missing_ok=True)
        check = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:
            check.bind(('127.0.0.1',20070))
            check.close()
            with (ROOT/'out'/'supervisor.log').open('a') as out,(ROOT/'out'/'supervisor.err').open('a') as err:
                runtime = ROOT/'.venv/Scripts/python.exe'
                subprocess.Popen([str(runtime) if runtime.exists() else sys.executable,'-m','mono_drive.forza_supervisor'],cwd=ROOT,
                                 stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        except OSError:
            pass
        finally:
            check.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['drive','hold'])
    parser.add_argument('--limit',type=float,default=None)
    args = parser.parse_args()
    request(args.action,args.limit)
