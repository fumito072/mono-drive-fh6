"""Keep driving workers running independently of chat and physical input."""
from pathlib import Path
import json
import os
import socket
import subprocess
import sys
import time
from .forza_session_policy import session_active
from .forza_speed import cruise_limit

ROOT = Path(__file__).resolve().parents[1]

def one_shot_to_hold():
    """Never replay a timed manoeuvre after an interrupted worker."""
    target = ROOT/'out'/'control.json'
    try:
        value = json.loads(target.read_text(encoding='utf-8-sig'))
        if value.get('action') in ('test', 'recover'):
            temporary = target.with_suffix('.supervisor.tmp')
            temporary.write_text(json.dumps(dict(action='hold',max_kmh=6)),encoding='utf-8')
            temporary.replace(target)
    except (OSError, ValueError):
        pass


def stop_worker(child):
    # Stop only this owned process tree, including the Windows venv launcher.
    if os.name == 'nt':
        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],
                       capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        child.kill()
    child.wait(timeout=10)


def main():
    guard = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    guard.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
    try:
        guard.bind(('127.0.0.1',20070))
    except OSError:
        return 0
    stopping = ROOT/'out'/'supervisor.stop'
    heartbeat = ROOT/'out'/'driver-heartbeat.json'
    restarts = 0
    training_python = ROOT/'.train-venv'/'Scripts'/'python.exe'
    if training_python.exists() and not (ROOT/'out'/'training.stop').exists():
        with (ROOT/'out'/'continuous-training.log').open('a') as out,(ROOT/'out'/'continuous-training.err').open('a') as err:
            subprocess.Popen([str(training_python),'tools/train_forza_continuous.py'],cwd=ROOT,
                             stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        while not stopping.exists():
            with (ROOT/'out'/'supervised-drive.log').open('a') as out,(ROOT/'out'/'supervised-drive.err').open('a') as err:
                started = time.time()
                runtime = ROOT/'.venv/Scripts/python.exe'
                command = [str(runtime) if runtime.exists() else sys.executable,'-m','mono_drive.forza','--background','--drive',
                           '--source','local_ai','--max-kmh',str(cruise_limit(ROOT))]
                try:
                    session = json.loads((ROOT/'out'/'longrun.json').read_text())
                    if session_active(session,time.time()) and not (ROOT/'out'/'longrun.stop').exists():
                        command += ['--record-hz',str(session.get('record_hz',2.))]
                except (OSError,ValueError,KeyError):
                    pass
                child = subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                status = dict(pid=os.getpid(),driver_pid=child.pid,restarts=restarts,started=started)
                (ROOT/'out'/'supervisor-status.json').write_text(json.dumps(status))
                stalled = False
                while child.poll() is None and not stopping.exists():
                    time.sleep(.5)
                    if time.time()-started > 60:
                        updated = heartbeat.stat().st_mtime if heartbeat.exists() else started
                        if updated < started or time.time()-updated > 20:
                            stalled = True
                            print('Driving worker heartbeat stalled; restarting.',flush=True)
                            stop_worker(child)
                            break
                if stopping.exists():
                    runs = sorted((ROOT/'out').glob('fh6_*'),key=lambda p:p.stat().st_mtime)
                    if runs:
                        (runs[-1]/'stop').write_text('Supervisor stop requested')
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        stop_worker(child)
                    break
                if child.returncode == 0 and not stalled:
                    try:
                        control=json.loads((ROOT/'out/control.json').read_text())
                    except (OSError,ValueError):control={}
                    if control.get('action')!='drive' or (ROOT/'out/longrun.stop').exists():
                        break
                one_shot_to_hold()
                restarts += 1
                time.sleep(min(10.,1.+restarts))
    finally:
        guard.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
