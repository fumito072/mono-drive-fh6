"""Local session watchdog. No cloud calls, language-model commands, or UI input."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
from .forza_speed import cruise_limit
import subprocess
import time

from .forza_local_learning import read_json, write_json
from .forza_session_policy import session_active

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'out'


def session_phase(config, stop, now):
    if stop:
        return 'stopped_by_user'
    if not session_active(config,now):
        return 'completed'
    return 'watching'


def health(now, driver, vlm, scene_time, session):
    def age(value):
        return round(max(0., now-value), 2) if value else None
    ages = dict(driver=age(driver.get('time')), video_model=age(vlm.get('time')),
                perception=age(scene_time), session=age(session.get('time')))
    issues = []
    for name, limit in (('driver', 3.), ('perception', 2.), ('session', 120.), ('video_model', 90.)):
        if ages[name] is None or ages[name] > limit:
            issues.append(name+'_stale')
    return dict(ages_seconds=ages, issues=issues, driving_mode=driver.get('mode'),
                speed_kmh=driver.get('speed_kmh'), model_phase=vlm.get('phase'))


def busy(port):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.bind(('127.0.0.1', port))
            return False
        except OSError:
            return True


def perception_restart_due(now, driver, scene_time, last_attempt):
    """A newly opened game can have live UDP while its old capture is closed."""
    return (driver.get('position') is not None
              and 0 <= now-max(driver.get('sample_time',0),driver.get('time',0)) < 3.
            and now-scene_time > 4.
            and now-last_attempt > 120.)


def storage_cleanup_mode(now,free_bytes,last_attempt,policy):
    elapsed=now-last_attempt
    if policy.get('delete_verified_decode_caches') is True:
        return 'all' if elapsed>3600. or (free_bytes<35*1024**3 and elapsed>300.) else None
    return 'reserve' if free_bytes<35*1024**3 and elapsed>300. else None


def port_owner(port):
    import psutil
    try:
        for connection in psutil.net_connections(kind='udp'):
            if connection.laddr and connection.laddr.port == port:
                return connection.pid
    except psutil.Error:
        pass
    return None


def launch(command, name):
    with (OUT/(name+'.log')).open('a') as log, (OUT/(name+'.err')).open('a') as err:
        subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=err,
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


def stop_owned(pid, module):
    """Terminate only the recorded Python module inside this project's venv."""
    import psutil
    try:
        process = psutil.Process(int(pid))
        command = process.cmdline()
        in_project = str(ROOT).lower() in ' '.join(command).lower() or process.cwd().lower() == str(ROOT).lower()
        is_module = any(command[i:i+2] == ['-m', module] for i in range(len(command)-1))
        if not in_project or not is_module or 'python' not in process.name().lower():
            return False
        process.terminate()
        process.wait(timeout=8.)
        return True
    except (psutil.Error, ValueError, TypeError):
        return False


def event(kind, **details):
    with (OUT/'local-monitor-events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(dict(time=time.time(), kind=kind, **details))+'\n')


def main():
    guard = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    guard.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        guard.bind(('127.0.0.1', 20073))
    except OSError:
        return 0
    next_write = next_check = 0.
    last_restart = {'storage':read_json(OUT/'storage-maintenance-state.json',{}).get('time',0.)}
    pending_vlm = None
    try:
        while True:
            now = time.time()
            config = read_json(OUT/'longrun.json', {})
            phase = session_phase(config, (OUT/'longrun.stop').exists(), now)
            if phase != 'watching':
                # Deadline enforcement also works if the session worker died.
                write_json(OUT/'control.json', dict(action='hold', max_kmh=cruise_limit(ROOT)))
                for name in ('supervisor', 'training', 'vision', 'vlm'):
                    (OUT/(name+'.stop')).write_text('Local monitor: '+phase, encoding='utf-8')
                write_json(OUT/'local-monitor-state.json', dict(time=now, pid=os.getpid(), phase=phase,
                    local_only=True, ends_at=config.get('ends_at')))
                event(phase)
                return 0
            driver = read_json(OUT/'driver-heartbeat.json', {})
            vlm = read_json(OUT/'vlm-state.json', {})
            session = read_json(OUT/'longrun-state.json', {})
            try:
                scene_time = (OUT/'scene.npz').stat().st_mtime
            except OSError:
                scene_time = 0.
            status = health(now, driver, vlm, scene_time, session)
            if now >= next_write:
                write_json(OUT/'local-monitor-state.json', dict(time=now, pid=os.getpid(), phase=phase,
                    local_only=True, poll_hz=10, publish_hz=2, ends_at=config['ends_at'], **status))
                next_write = now+.5
            if now >= next_check:
                next_check = now+2.
                try:
                    from .forza_candidate_review import review_completed
                    from .forza_model_policy import service
                    review_completed(ROOT,now)
                    service(ROOT,driver,session,now)
                    from .forza_live_evaluation import update as update_evaluation
                    update_evaluation(ROOT,driver,session,now)
                except (OSError,ValueError,KeyError,TypeError) as error:
                    event('model_activation_error',error=str(error))
                # Resume keeps the saved deadline and never removes an explicit stop.
                import shutil
                retention=read_json(ROOT/'configs/data_retention.json',{})
                cleanup_mode=storage_cleanup_mode(now,shutil.disk_usage(ROOT).free,last_restart.get('storage',0),retention)
                if cleanup_mode:
                    command=[str(ROOT/'.train-venv/Scripts/python.exe'),'-m','mono_drive.forza_storage']
                    if cleanup_mode=='all':command.append('--all-completed')
                    launch(command,'storage-maintenance')
                    event('storage_maintenance_requested')
                    last_restart['storage']=now
                storage_ready=shutil.disk_usage(ROOT).free>=25*1024**3
                if not busy(20071) and storage_ready and now-last_restart.get('session', 0)>20.:
                    launch([str(ROOT/'.venv/Scripts/python.exe'), '-m', 'mono_drive.forza_session', 'worker'], 'longrun')
                    event('session_restarted', ends_at=config['ends_at'])
                    last_restart['session'] = now
                if (read_json(OUT/'control.json', {}).get('action') == 'drive'
                        and not (OUT/'vision.stop').exists()
                        and perception_restart_due(now, driver, scene_time, last_restart.get('vision', 0))):
                    command = [str(ROOT/'.train-venv/Scripts/python.exe'), '-m', 'mono_drive.forza_vision']
                    owner = port_owner(20068)
                    if owner:
                        command += ['--handover-pid', str(owner)]
                    if owner or not busy(20068):
                        launch(command, 'vision')
                        event('perception_capture_restart', previous_pid=owner)
                        last_restart['vision'] = now
                # Model timeouts are local. First ask it to finish, then terminate
                # only its verified process if it cannot respond within 20 seconds.
                model_unhealthy=('video_model_stale' in status['issues'] or vlm.get('consecutive_errors',0)>=5)
                if model_unhealthy and config.get('vlm_enabled') and busy(20072):
                    if pending_vlm is None and now-last_restart.get('vlm', 0)>120. and not (OUT/'vlm.stop').exists():
                        pending_vlm = (now, vlm.get('pid'))
                        (OUT/'vlm.stop').write_text('Local model watchdog restart', encoding='utf-8')
                        event('vlm_stale', model_state=vlm)
                    elif pending_vlm and now-pending_vlm[0]>20.:
                        if stop_owned(pending_vlm[1], 'mono_drive.forza_vlm'):
                            event('vlm_hung_process_stopped', pid=pending_vlm[1])
                if pending_vlm and not busy(20072):
                    marker = OUT/'vlm.stop'
                    if marker.exists() and marker.read_text(encoding='utf-8') == 'Local model watchdog restart':
                        marker.unlink()
                    last_restart['vlm'] = now
                    pending_vlm = None
                    # Session owns model startup. No duplicate model loading.
                if 'session_stale' in status['issues'] and session.get('state') not in ('recovering_to_road','changing_route') and busy(20071):
                    if now-last_restart.get('session', 0)>180.:
                        if stop_owned(session.get('pid'), 'mono_drive.forza_session'):
                            event('session_hung_process_stopped', pid=session.get('pid'))
                        last_restart['session'] = now
            time.sleep(.1)
    finally:
        guard.close()


if __name__ == '__main__':
    raise SystemExit(main())
