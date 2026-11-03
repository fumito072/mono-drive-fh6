"""Bounded local driving/learning session with observed game-menu recovery."""
from __future__ import annotations
import argparse
from collections import deque
import ctypes
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import cv2
import numpy as np
from .forza_session_policy import session_active
from .forza_speed import cruise_limit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'out'
CONFIG = OUT/'longrun.json'
STATE = OUT/'longrun-state.json'
STOP = OUT/'longrun.stop'


def read_json(path,default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):
        return default


def write_json(path,value):
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    for _ in range(4):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(.025)


def close_game_resources(*resources):
    """A failed capture shutdown must not leave the telemetry port occupied."""
    errors=[]
    for resource in resources:
        if resource is None:continue
        try:resource.close()
        except Exception as exc:errors.append(dict(resource=type(resource).__name__,error=str(exc)))
    if errors:write_json(OUT/'recovery-cleanup-error.json',dict(time=time.time(),errors=errors))


def port_busy(port):
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
        try:
            sock.bind(('127.0.0.1',port))
            return False
        except OSError:
            return True


def launch(command,log):
    with (OUT/(log+'.log')).open('a') as output,(OUT/(log+'.err')).open('a') as error:
        return subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=error,
                                creationflags=subprocess.CREATE_NO_WINDOW)


class ProgressWatch:
    def __init__(self,seconds=60.,radius=12.):
        self.seconds,self.radius = seconds,radius
        self.samples = deque()

    def update(self,now,position):
        if position is None:
            return False
        self.samples.append((now,np.asarray(position,dtype=float)))
        while len(self.samples)>1 and self.samples[1][0] < now-self.seconds:
            self.samples.popleft()
        if now-self.samples[0][0] < self.seconds:
            return False
        origin = self.samples[0][1]
        return max(float(np.linalg.norm(p-origin)) for _,p in self.samples) < self.radius


class MenuRecognizer:
    def __init__(self,root=None):
        self.root = root or ROOT/'models'/'ui'/'fh6'
        self.specs = read_json(self.root/'templates.json',{})
        self.images = {name:cv2.imread(str(self.root/spec['file']),0) for name,spec in self.specs.items()}
        self.player_card=cv2.imread(str(self.root/'player-card.png'),0)
        self.place_logo=cv2.imread(str(self.root/'place-card-logo.png'),0)
        self.driving_hud=cv2.imread(str(self.root/'driving-hud.png'),0)

    def score(self,frame,name):
        small = cv2.resize(frame,(1920,1080))
        gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        x1,y1,x2,y2 = self.specs[name].get('search_rect',self.specs[name]['rect'])
        region = gray[max(0,y1-4):min(1080,y2+4),max(0,x1-4):min(1920,x2+4)]
        template = self.images[name]
        if template is None:
            return 0.
        return float(cv2.matchTemplate(region,template,cv2.TM_CCOEFF_NORMED).max())

    def classify(self,frame):
        if frame is None or not self.specs:
            return 'unknown'
        scores = {name:self.score(frame,name) for name in self.specs}
        name = max(scores,key=scores.get)
        return self.specs[name].get('screen',name) if scores[name] >= .90 else 'unknown'

    def map_selected(self,frame):
        hsv = cv2.cvtColor(cv2.resize(frame,(1920,1080)),cv2.COLOR_BGR2HSV)
        # Campaign world-map tile's yellow selection border, not another tile.
        border = np.concatenate((hsv[250:257,550:1360].reshape(-1,3),
                                 hsv[553:560,550:1360].reshape(-1,3)))
        return float(np.mean((border[:,0]>18)&(border[:,0]<45)&(border[:,1]>150)&(border[:,2]>150))) > .25

    def player_is_selected(self,frame,player):
        if player is None or self.player_card is None:
            return False
        gray=cv2.cvtColor(cv2.resize(frame,(1920,1080)),cv2.COLOR_BGR2GRAY)
        x,y=map(int,player)
        region=gray[max(0,y-310):min(1080,y-70),max(0,x-300):min(1920,x+300)]
        # The selected account is stable; the car name/class changes when the
        # user swaps cars. Match only the account row of the captured card.
        template=self.player_card[:58]
        h,w=template.shape
        return bool(region.shape[0]>=h and region.shape[1]>=w and
                    cv2.matchTemplate(region,template,cv2.TM_CCOEFF_NORMED).max()>.95)

    def place_is_selected(self,frame):
        if self.place_logo is None:
            return False
        gray=cv2.cvtColor(cv2.resize(frame,(1920,1080)),cv2.COLOR_BGR2GRAY)
        return bool(cv2.matchTemplate(gray[:900],self.place_logo,cv2.TM_CCOEFF_NORMED).max()>.95)

    def driving_ready(self,frame):
        if frame is None or self.driving_hud is None:
            return False
        gray=cv2.cvtColor(cv2.resize(frame,(1920,1080)),cv2.COLOR_BGR2GRAY)
        return bool(cv2.matchTemplate(gray[870:960,1750:1900],self.driving_hud,
                                     cv2.TM_CCOEFF_NORMED).max()>.95)

    @staticmethod
    def ring_cursor(frame):
        """Recover the cursor even when its edge blends into white map roads."""
        hsv=cv2.cvtColor(cv2.resize(frame,(1920,1080)),cv2.COLOR_BGR2HSV)
        white=((hsv[:,:,1]<90)&(hsv[:,:,2]>170)).astype(np.float32)
        yy,xx=np.indices((97,97))
        distance=np.hypot(xx-48,yy-48)
        best=(0.,None)
        for radius in (31,33,35,37,39,41):
            ring=((distance>=radius-1)&(distance<=radius+1)).astype(np.float32)
            flanks=(((distance>=radius-5)&(distance<radius-3))|
                    ((distance>radius+3)&(distance<=radius+5))).astype(np.float32)
            fraction=cv2.filter2D(white,-1,ring/ring.sum())
            score=fraction-.8*cv2.filter2D(white,-1,flanks/flanks.sum())
            # The map retains the cursor's last screen position when reopened;
            # it can be above y=270 after panning. Mask only the actual HUD.
            score[:55]=0; score[950:]=0; score[:,:60]=0; score[:,1860:]=0
            score[:285,1370:]=0
            score[fraction<.60]=0
            _,quality,_,point=cv2.minMaxLoc(score)
            if quality>best[0]:
                best=quality,point
        return np.array(best[1],dtype=float) if best[0]>.50 else None

    @staticmethod
    def map_points(frame):
        """Find the hollow selection cursor and filled triangular player marker."""
        im = cv2.resize(frame,(1920,1080))
        hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        # A white information card also passes a simple circle/whiteness test.
        # Require the thin ring and its contrasting inner/outer flanks.
        cursor=MenuRecognizer.ring_cursor(im)
        white = ((hsv[:,:,1]<90)&(hsv[:,:,2]>210)).astype(np.uint8)*255
        contours,_ = cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        triangles=[]
        for contour in contours:
            area=cv2.contourArea(contour)
            x,y,w,h=cv2.boundingRect(contour)
            vertices=cv2.approxPolyDP(contour,.045*cv2.arcLength(contour,True),True)
            if (180<area<3000 and 25<w<100 and 25<h<100 and 200<y<950 and
                    len(vertices)==4 and not cv2.isContourConvex(vertices)):
                shape=np.array([[0,-100],[-60,70],[0,35],[60,70]],np.int32)
                score=cv2.matchShapes(vertices,shape,cv2.CONTOURS_MATCH_I1,0)
                triangles.append((score,np.array([x+w/2,y+h/2])))
        player = min(triangles,key=lambda p:p[0])[1] if triangles else None
        return cursor,player


class PadChannel:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    def send(self,buttons=(),axes=(0.,0.,0.)):
        self.sock.sendto(json.dumps(dict(command=axes,buttons=list(buttons))).encode(),('127.0.0.1',20067))

    def ready(self):
        self.sock.settimeout(.3)
        try:
            self.sock.sendto(b'{"hello":true}',('127.0.0.1',20067))
            payload,address=self.sock.recvfrom(1024)
            return address==('127.0.0.1',20067) and json.loads(payload).get('ready') is True
        except (OSError,ValueError):
            return False
        finally:
            self.sock.settimeout(None)

    def press(self,button):
        if button in ('RIGHT_SHOULDER','RIGHT_THUMB'):
            raise ValueError('Camera controls are disabled in automated sessions')
        for _ in range(6):
            self.send((button,))
            time.sleep(.025)
        self.send()

    def move_map(self,x,y,duration=.15):
        until=time.monotonic()+duration
        while time.monotonic()<until:
            self.sock.sendto(json.dumps(dict(command=[0,0,0],ui_stick=[float(x),float(y)])).encode(),('127.0.0.1',20067))
            time.sleep(.025)
        self.send()

    def close(self):
        self.send()
        self.sock.close()


def map_stick(delta,deadzone=10.):
    """Map cursor shares the game's stick dead zone; tiny reports do nothing."""
    delta=np.asarray(delta,dtype=float)
    magnitude=np.clip(np.abs(delta)/180.,.40,.90)
    return np.where(np.abs(delta)>deadzone,np.sign(delta)*magnitude,0.)


def wait_frame(capture,recognizer,wanted,seconds,folder,label):
    until = time.monotonic()+seconds
    while time.monotonic()<until and not STOP.exists():
        frame,stamp = capture.grab()
        if frame is not None and time.monotonic()-stamp < .5 and recognizer.classify(frame)==wanted:
            cv2.imwrite(str(folder/(label+'.jpg')),frame)
            return frame
        time.sleep(.2)
    raise RuntimeError(f'Expected game screen was not observed: {wanted}')


def stop_driver_for_menu(reason):
    write_json(OUT/'control.json',dict(action='hold',max_kmh=cruise_limit(ROOT)))
    (OUT/'supervisor.stop').write_text(reason)
    until = time.monotonic()+15
    while port_busy(20066) and time.monotonic()<until:
        time.sleep(.2)
    if port_busy(20066):
        raise RuntimeError('Driving worker did not release the controller/telemetry for recovery')


def recover_controller_dialog():
    """Acknowledge only the captured disconnect dialog, then the known pause menu."""
    from .forza_background import GameCapture
    from .forza import Telemetry
    capture,recognizer=GameCapture(),MenuRecognizer()
    pad=telemetry=None
    try:
        time.sleep(.5)
        frame,stamp=capture.grab()
        if (STOP.exists() or frame is None or time.monotonic()-stamp>.5 or
                recognizer.classify(frame)!='controller_disconnected' or
                recognizer.score(frame,'controller_disconnected')<.97):
            return False
        folder=OUT/time.strftime('reconnect_%Y%m%d_%H%M%S')
        folder.mkdir()
        cv2.imwrite(str(folder/'before.jpg'),frame)
        stop_driver_for_menu('Observed controller disconnect dialog')
        pad=PadChannel()
        until=time.monotonic()+8.
        launched=False
        while not pad.ready() and time.monotonic()<until and not STOP.exists():
            if not launched and not port_busy(20067):
                launch([str(ROOT/'.venv/Scripts/python.exe'),'-m','mono_drive.forza_pad_server'],'controller-host')
                launched=True
            time.sleep(.25)
        if STOP.exists() or not pad.ready():
            raise RuntimeError('Controller disconnect dialog is visible but the virtual pad is unavailable')
        telemetry=Telemetry(20066)
        # Reconfirm after stopping/reconnecting; no A input on another screen.
        frame,_=capture.grab()
        if recognizer.classify(frame)!='controller_disconnected':
            raise RuntimeError('Controller dialog changed before acknowledgement; no button sent')
        pad.press('A')
        until=time.monotonic()+8.
        closed_pause=False
        while time.monotonic()<until and not STOP.exists():
            time.sleep(.2)
            frame,stamp=capture.grab()
            if frame is None or time.monotonic()-stamp>.5:
                continue
            screen=recognizer.classify(frame)
            if screen=='pause' and not closed_pause:
                cv2.imwrite(str(folder/'pause.jpg'),frame)
                pad.press('B')
                closed_pause=True
                continue
            telemetry.poll()
            if (screen=='unknown' and recognizer.driving_ready(frame) and telemetry.fresh()
                    and telemetry.last['race'] and not telemetry.last['race_position']
                    and not telemetry.last['lap_number']):
                cv2.imwrite(str(folder/'after.jpg'),frame)
                write_json(folder/'result.json',dict(success=True,position=telemetry.last['position']))
                return True
        raise RuntimeError('Controller was reconnected but a verified free-roam screen did not return')
    finally:
        close_game_resources(telemetry,pad,capture)


def observe_resumable_free_roam():
    """Read-only probe after a failed recovery; never dismiss a menu blindly."""
    if STOP.exists() or port_busy(20066):
        return False
    from .forza import Telemetry
    from .forza_background import GameCapture
    from .forza_scene import SceneReader
    from .forza_routes import paved_arrival
    capture=telemetry=None
    stable=None
    try:
        telemetry=Telemetry(20066)
        capture=GameCapture()
        scene,recognizer=SceneReader(),MenuRecognizer()
        until=time.monotonic()+2.
        while time.monotonic()<until and not STOP.exists():
            now=time.monotonic()
            telemetry.poll()
            frame,stamp=capture.grab()
            value=telemetry.last or {}
            visible=(frame is not None and 0<=now-stamp<.5 and telemetry.fresh() and
                     value.get('race') and not value.get('race_position') and not value.get('lap_number') and
                     recognizer.classify(frame)=='unknown' and recognizer.driving_ready(frame) and
                     paved_arrival(scene.read(now),now))
            stable=(now if stable is None else stable) if visible else None
            if stable is not None and now-stable>=.5:
                return True
            time.sleep(.1)
        return False
    except (OSError,RuntimeError):
        return False
    finally:
        close_game_resources(capture,telemetry)


def restore_road(destination=None):
    """Only activate controls after each known screen is visually confirmed."""
    from .forza_background import GameCapture
    from .forza import Telemetry
    from .forza_map_anchor import FixedMapAnchor
    anchor = FixedMapAnchor(destination or ROOT/'models'/'ui'/'fh6'/'return-point.json')
    varied_route = destination is not None
    stop_driver_for_menu('Observed automatic road recovery')
    folder = OUT/time.strftime('rescue_%Y%m%d_%H%M%S')
    folder.mkdir()
    capture=pad=telemetry=None
    from .forza_routes import road_probe_offsets
    offsets=road_probe_offsets() if varied_route else [(0.,0.)]
    probe_index=0
    selected_map_point=None
    try:
        capture=GameCapture();pad=PadChannel();recognizer=MenuRecognizer();telemetry=Telemetry(20066)
        if varied_route:
            from .forza_scene import SceneReader
            from .forza_routes import paved_arrival
            scene_reader=SceneReader()
        time.sleep(.5)
        telemetry.poll()
        frame,_ = capture.grab()
        initial_screen = recognizer.classify(frame)
        observed_menu=initial_screen in ('map','fast_travel','pause')
        live=telemetry.last or {}
        if (frame is None or live.get('race_position',0) or
                (not observed_menu and (not telemetry.fresh() or not live.get('race')))):
            raise RuntimeError('Free roam is not ready for automatic road recovery')
        cv2.imwrite(str(folder/'before.jpg'),frame)
        origin = live.get('position') if telemetry.fresh() else read_json(OUT/'driver-heartbeat.json',{}).get('position')
        if initial_screen not in ('map','fast_travel'):
            if initial_screen!='pause':
                pad.press('START')
            frame = wait_frame(capture,recognizer,'pause',6,folder,'pause')
            if not recognizer.map_selected(frame):
                raise RuntimeError('World map tile is not selected; leave menus for user review')
            pad.press('A')
            wait_frame(capture,recognizer,'map',8,folder,'map')
        clear_attempts=0
        search_attempts=0
        last_cursor=None
        for attempt in range(64):
            frame,_ = capture.grab()
            if recognizer.classify(frame)=='fast_travel':
                break
            if recognizer.classify(frame)!='map':
                raise RuntimeError('Map disappeared while selecting a known location')
            cursor,player=recognizer.map_points(frame)
            # Hovering our own marker hides the thin ring. The independently
            # matched player card identifies this selected marker precisely;
            # a generic place/event card cannot provide this position.
            if cursor is None and player is not None and recognizer.player_is_selected(frame,player):
                cursor=player
            goal_pixel=anchor.locate(frame,offsets[probe_index]) if varied_route else anchor.locate(frame)
            cv2.imwrite(str(folder/f'navigation-{attempt:02d}.jpg'),frame)
            with (folder/'navigation.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps(dict(time=time.time(),cursor=None if cursor is None else cursor.tolist(),
                    destination=None if goal_pixel is None else goal_pixel.tolist(),clear_attempts=clear_attempts,
                    probe=probe_index))+'\n')
            if goal_pixel is None:
                direction=anchor.search_direction(origin)
                if direction is not None and search_attempts<8:
                    # The map is verified, but the saved region is off-screen.
                    # Move toward its world coordinates; never activate a tile
                    # until geographic landmarks locate the destination again.
                    search_attempts+=1
                    pad.move_map(direction[0],-direction[1],.4)
                    time.sleep(.2)
                    continue
                raise RuntimeError('Saved paved destination could not be located after bounded map search; no destination activated')
            if cursor is None:
                # A recognized map plus geographic landmarks permits bounded
                # navigation, even when a location/event card hides the ring.
                # Unmatched cards cannot supply a cursor or authorize A.
                if clear_attempts >= (8 if varied_route or last_cursor is not None else 3):
                    raise RuntimeError('Map cursor remained hidden after bounded navigation attempts')
                clear_attempts+=1
                if last_cursor is not None:
                    delta=goal_pixel-last_cursor
                    axes=map_stick(delta,deadzone=2.)
                    if np.linalg.norm(delta)<6:
                        axes=np.array([.4,.4])
                    pad.move_map(axes[0],-axes[1],.18)
                else:
                    pad.move_map(0.,-.65,.3)
                time.sleep(.5)
                continue
            last_cursor=cursor
            distance=float(np.linalg.norm(goal_pixel-cursor))
            if distance>4.:
                delta=goal_pixel-cursor
                axes=map_stick(delta,deadzone=2.)
                duration=min(.3,max(.05,distance/240.))
                pad.move_map(axes[0],-axes[1],duration)
                time.sleep(.2)
            else:
                selected_map_point=anchor.reference_point(cursor) if varied_route else None
                pad.press('X')
                time.sleep(.5)
                if varied_route:
                    next_frame,_=capture.grab()
                    if recognizer.classify(next_frame)=='map':
                        probe_index+=1
                        if probe_index>=len(offsets):
                            raise RuntimeError('No fast-travel road found within the bounded destination search')
        wait_frame(capture,recognizer,'fast_travel',3,folder,'fast_travel')
        # This exact observed dialog is the free fast-travel confirmation.
        pad.press('A')
        until = time.monotonic()+35
        ready_since=None
        while time.monotonic()<until and not STOP.exists():
            telemetry.poll()
            frame,stamp = capture.grab()
            if (telemetry.fresh() and telemetry.last['race'] and frame is not None and
                    not telemetry.last['race_position'] and not telemetry.last['lap_number'] and
                    time.monotonic()-stamp<.5 and recognizer.driving_ready(frame) and
                    recognizer.classify(frame)=='unknown' and
                    anchor.arrived(telemetry.last['position']) and
                    (not varied_route or paved_arrival(scene_reader.read(time.monotonic()),time.monotonic()))):
                if ready_since is None:
                    ready_since=time.monotonic()
                if time.monotonic()-ready_since>=1.:
                    cv2.imwrite(str(folder/'after.jpg'),frame)
                    result=dict(before=origin,after=telemetry.last['position'],success=True,
                        destination=anchor.config['name'],paved_verified=varied_route,map_point=selected_map_point)
                    write_json(folder/'result.json',result)
                    return result
            else:
                ready_since=None
            time.sleep(.25)
        if frame is not None:
            cv2.imwrite(str(folder/'failed.jpg'),frame)
        write_json(folder/'result.json',dict(before=origin,after=telemetry.last,success=False))
        raise RuntimeError('Fast travel did not return to fresh free-roam telemetry')
    finally:
        close_game_resources(pad,capture,telemetry)


def road_recovery_reason(now,surface,manual):
    if 0.<=now-manual.get('time',0)<60. and manual.get('requested') is True:
        return 'requested_road_recovery'
    if 0.<=now-surface.get('time',0)<3. and surface.get('offroad_confirmed') is True:
        return 'confirmed_sidewalk'
    return None


def worker():
    guard = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    guard.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
    try:
        guard.bind(('127.0.0.1',20071))
    except OSError:
        return
    config = read_json(CONFIG)
    if not config:
        raise RuntimeError('Missing long-run configuration')
    if STOP.exists() or not session_active(config,time.time()):
        guard.close()
        return
    watch = ProgressWatch()
    saved=read_json(STATE,{})
    recoveries=int(saved.get('recoveries',0)) if saved.get('started_at')==config['started_at'] else 0
    last_recovery,error = 0.,None
    recovery_failures,retry_at = 0,0.
    ui_failures,next_ui_check=0,0.
    next_free_roam_probe=0.
    state = 'starting'
    if os.name=='nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    from .forza_control import request
    from .forza_incidents import StopWatch,capture_incident
    stop_watch=StopWatch()
    last_incident=None
    local_recovery_active=False
    pending_incident_recovery=False
    resume_requested_at=0.
    from .forza_routes import RouteRotation
    rotation=RouteRotation(ROOT)
    try:
        if not port_busy(20073):
            launch([str(ROOT/'.train-venv/Scripts/python.exe'),'-m','mono_drive.forza_local_monitor'],'local-monitor')
        (OUT/'vision.stop').unlink(missing_ok=True)
        (OUT/'vlm.stop').unlink(missing_ok=True)
        request('drive',cruise_limit(ROOT))
        while not STOP.exists() and session_active(config,time.time()):
            config=read_json(CONFIG,config)
            now = time.time()
            if config.get('resume_requested_at',0)>resume_requested_at:
                resume_requested_at=config['resume_requested_at']
                watch=ProgressWatch()
                stop_watch=StopWatch()
                recovery_failures=ui_failures=0
                retry_at=next_ui_check=0.
                pending_incident_recovery=False
                state,error='starting',None
                request('drive',cruise_limit(ROOT))
            if shutil.disk_usage(ROOT).free < 25*1024**3:
                state,error = 'storage_limit','Free space is below 25 GiB; recordings retained'
                break
            heartbeat = read_json(OUT/'driver-heartbeat.json',{})
            control = read_json(OUT/'control.json',{})
            if not port_busy(20070) and state!='needs_attention':
                request('drive',cruise_limit(ROOT))
            if not port_busy(20069):
                (OUT/'training.stop').unlink(missing_ok=True)
                launch([str(ROOT/'.train-venv/Scripts/python.exe'),'tools/train_forza_continuous.py'],'continuous-training')
            if not port_busy(20074):
                launch([str(ROOT/'.train-venv/Scripts/python.exe'),'-m','mono_drive.forza_data_worker'],'human-data-worker')
            if (read_json(ROOT/'configs/temporal_steering.json',{}).get('enabled') is True
                    and not (OUT/'training.stop').exists() and not port_busy(20083)):
                launch([str(ROOT/'.train-venv/Scripts/python.exe'),'-m','mono_drive.forza_temporal_worker'],'temporal-worker')
            if config.get('vlm_enabled',False) and not (OUT/'vlm.stop').exists() and not port_busy(20072):
                launch([str(ROOT/'.train-venv/Scripts/python.exe'),'-m','mono_drive.forza_vlm'],'vlm')
            fresh = now-heartbeat.get('time',0)<8
            has_pose=heartbeat.get('position') is not None
            if heartbeat.get('speed_kmh',0)>4.:
                pending_incident_recovery=False
            incident_recovery=pending_incident_recovery
            began_local_recovery=bool(heartbeat.get('recovery_phase')) and not local_recovery_active
            local_recovery_active=bool(heartbeat.get('recovery_phase'))
            if control.get('action')=='drive' and (stop_watch.update(now,heartbeat) or (fresh and began_local_recovery)):
                try:
                    last_incident=capture_incident(OUT,heartbeat,now)
                    action=last_incident['diagnosis']['action']
                    if action=='restart_driver':
                        run=Path(heartbeat['run'])
                        if run.resolve().parent==OUT.resolve():
                            (run/'stop').write_text('Local incident diagnosis: reload driving worker')
                    incident_recovery=action=='recover_road'
                    pending_incident_recovery=incident_recovery
                except Exception as exc:
                    write_json(OUT/'incident-error.json',dict(time=now,error=str(exc)))
            if (state=='needs_attention' and control.get('action')=='hold' and now>=next_free_roam_probe
                    and config.get('auto_recovery_enabled',True)):
                next_free_roam_probe=now+30.
                if observe_resumable_free_roam():
                    watch=ProgressWatch()
                    stop_watch=StopWatch()
                    ui_failures=recovery_failures=0
                    pending_incident_recovery=False
                    request('drive',cruise_limit(ROOT))
                    state,error='waiting_for_game',None
                    control=read_json(OUT/'control.json',{})
            ui_retry=state=='needs_attention' and ui_failures>0 and control.get('action')=='hold'
            if ((not fresh or not has_pose) and (control.get('action')=='drive' or ui_retry)
                    and config.get('auto_recovery_enabled',True) and now>=next_ui_check and ui_failures<3):
                next_ui_check=now+15.
                try:
                    if recover_controller_dialog():
                        watch=ProgressWatch()
                        ui_failures=recovery_failures=0
                        request('drive',cruise_limit(ROOT))
                        state,error='waiting_for_game',None
                except Exception as exc:
                    ui_failures+=1
                    recovery_failures=3
                    state,error='needs_attention',str(exc)
            stalled = fresh and watch.update(now,heartbeat.get('position'))
            retry_ready=state=='needs_attention' and recovery_failures<3 and now>=retry_at
            carriageway=read_json(OUT/'carriageway-state.json',{})
            wrongway=now-carriageway.get('time',0)<3. and carriageway.get('wrongway_confirmed') is True
            manual=read_json(OUT/'road-recovery-request.json',{})
            surface_reason=road_recovery_reason(now,read_json(OUT/'surface-state.json',{}),manual)
            ready_recovery=((stalled or wrongway) and now-last_recovery>60) or ((surface_reason or incident_recovery) and now-last_recovery>20)
            initial_recovery=ready_recovery and control.get('action')=='drive' and state!='needs_attention'
            route_request=read_json(OUT/'route-change-request.json',{})
            force_route=route_request.get('requested') is True
            selected_route=rotation.choose(now,recoveries,force=force_route) if fresh and has_pose and control.get('action')=='drive' else None
            if config.get('auto_recovery_enabled',True) and (initial_recovery or retry_ready or selected_route is not None):
                state = 'changing_route' if selected_route is not None else 'recovering_to_road'
                write_json(STATE,dict(**config,pid=os.getpid(),time=now,state=state,recoveries=recoveries))
                try:
                    # A planned course change is not a failed approach. Its
                    # pre-menu driving history must not create a phantom wall.
                    if (initial_recovery or retry_ready) and (last_incident is None or now-last_incident['time']>15.):
                        try:
                            last_incident=capture_incident(OUT,heartbeat,now)
                        except Exception as exc:
                            write_json(OUT/'incident-error.json',dict(time=now,error=str(exc)))
                    if surface_reason=='requested_road_recovery':
                        (OUT/'road-recovery-request.json').unlink(missing_ok=True)
                    if selected_route is not None:
                        (OUT/'route-change-request.json').unlink(missing_ok=True)
                        rotation.started(selected_route,time.time())
                        try:
                            result=restore_road(selected_route['anchor'])
                            rotation.succeeded(selected_route,result['after'],time.time(),recoveries,result.get('map_point'))
                        except Exception as route_error:
                            rotation.failed(selected_route,time.time(),route_error)
                            restore_road()
                    else:
                        restore_road()
                        recoveries += 1
                    pending_incident_recovery=False
                    last_recovery = time.time()
                    watch = ProgressWatch()
                    request('drive',cruise_limit(ROOT))
                    state,error = 'running',None
                    recovery_failures=0
                except Exception as exc:
                    state,error = 'needs_attention',str(exc)
                    recovery_failures+=1
                    retry_at=time.time()+30.
            elif state not in ('needs_attention','recovering_to_road'):
                state = 'running' if fresh and has_pose else 'waiting_for_game'
            write_json(STATE,dict(**config,pid=os.getpid(),time=now,state=state,recoveries=recoveries,
                last_incident=None if last_incident is None else dict(id=last_incident['id'],**last_incident['diagnosis']),
                error=error,driver=heartbeat,training=read_json(OUT/'training-state.json',{}),
                video_monitor=read_json(OUT/'vlm-state.json',{}),
                free_gib=round(shutil.disk_usage(ROOT).free/1024**3,1)))
            time.sleep(2)
        state = 'stopped_by_user' if STOP.exists() else state if state=='storage_limit' else 'completed'
    finally:
        write_json(OUT/'control.json',dict(action='hold',max_kmh=cruise_limit(ROOT)))
        (OUT/'supervisor.stop').write_text('Bounded session finished')
        (OUT/'training.stop').write_text('Bounded session finished')
        (OUT/'vision.stop').write_text('Bounded session finished')
        (OUT/'vlm.stop').write_text('Bounded session finished')
        write_json(STATE,dict(**config,pid=os.getpid(),time=time.time(),state=state,recoveries=recoveries,error=error))
        write_json(OUT/'longrun-summary.json',read_json(STATE))
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        guard.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['start','resume','worker','stop'])
    parser.add_argument('--hours',type=float,default=20.)
    parser.add_argument('--continuous',action='store_true',help='Run until explicitly stopped or storage reserve is reached')
    parser.add_argument('--vehicle',default=None,help='User-selected vehicle label for this session')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.action=='worker':
        worker()
    elif args.action=='stop':
        STOP.write_text('Explicit session stop')
    else:
        if args.action=='start':
            if not math.isfinite(args.hours) or not 0<args.hours<=24:
                raise ValueError('Session must be bounded to at most 24 hours')
            now = time.time()
            write_json(CONFIG,dict(started_at=now,ends_at=None if args.continuous else now+args.hours*3600,
                                   hours=None if args.continuous else args.hours,continuous=args.continuous,vehicle=args.vehicle,
                                   resume_requested_at=now,
                                   objective='Free-roam driving and local candidate training',record_hz=2.,
                                   auto_recovery_enabled=True,vlm_enabled=True))
        config=read_json(CONFIG,{})
        if not session_active(config,time.time()):
            raise RuntimeError('The saved session deadline has passed')
        if args.action=='resume' and STOP.exists():
            raise RuntimeError('Explicit stop is present; resume will not clear it')
        if args.action=='start':
            STOP.unlink(missing_ok=True)
        else:
            config['resume_requested_at']=time.time()
            write_json(CONFIG,config)
        if not port_busy(20073):
            launch([str(ROOT/'.train-venv/Scripts/python.exe'),'-m','mono_drive.forza_local_monitor'],'local-monitor')
        if port_busy(20071):
            print(json.dumps(config))
            return
        launch([sys.executable,'-m','mono_drive.forza_session','worker'],'longrun')
        print(json.dumps(config))


if __name__=='__main__':
    main()
