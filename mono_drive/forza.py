"""Local Forza Horizon 6 adapter with independent virtual gamepad controls."""
from __future__ import annotations

import argparse
import ctypes as C
from ctypes import wintypes as W
import json
import math
from pathlib import Path
import socket
import struct
import threading
import time
from .forza_speed import MAX_CRUISE_KMH
from .forza_relaunch import RelaunchAssist

ROOT = Path(__file__).resolve().parents[1]


def parse_packet(data):
    # FH6 official Data Out format: 324 bytes; shared Sled header.
    if len(data) != 324:
        raise ValueError(f"Expected FH6 324-byte packet, got {len(data)}")
    race, stamp = struct.unpack_from('<iI', data)
    velocity = struct.unpack_from('<3f', data, 32)
    speed = math.sqrt(sum(v*v for v in velocity)) * 3.6
    if race not in (0, 1) or not math.isfinite(speed) or speed > 700:
        raise ValueError('Invalid telemetry')
    return dict(race=bool(race), stamp=stamp, speed=speed,
                forward_kmh=velocity[2]*3.6,
                position=list(struct.unpack_from('<3f', data, 244)),
                yaw=struct.unpack_from('<f', data, 56)[0],
                accel=data[315], brake=data[316], gear=data[319],
                handbrake=data[318], race_position=data[314],
                lap_number=struct.unpack_from('<H', data, 312)[0],
                steer=struct.unpack_from('<b', data, 320)[0],
                car_ordinal=struct.unpack_from('<i',data,212)[0],
                car_class=struct.unpack_from('<i',data,216)[0],
                performance_index=struct.unpack_from('<i',data,220)[0],
                drivetrain=struct.unpack_from('<i',data,224)[0])


class Telemetry:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', port))
        self.sock.setblocking(False)
        self.last = None
        self.received = 0.0
        self.error = ''
        self.stop_event = threading.Event()
        self.worker = None

    def start(self):
        def receive():
            while not self.stop_event.wait(.005):
                self.poll()
        self.worker = threading.Thread(target=receive, daemon=True)
        self.worker.start()

    def close(self):
        self.stop_event.set()
        if self.worker:
            self.worker.join(timeout=1)
        self.sock.close()

    def poll(self):
        # Bounded drain avoids starving control under a packet flood.
        for _ in range(256):
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                value = parse_packet(data)
            except ValueError as exc:
                self.last = None
                self.error = str(exc)
                continue
            if self.last is None or value['stamp'] != self.last['stamp']:
                self.received = time.monotonic()
            self.last = value
            self.error = ''
        return self.last

    def fresh(self):
        return self.last is not None and time.monotonic() - self.received < .35


class Windows:
    def __init__(self):
        self.u = C.WinDLL('user32', use_last_error=True)
        self.k = C.WinDLL('kernel32', use_last_error=True)
        self.u.GetForegroundWindow.restype = W.HWND
        self.u.GetWindowThreadProcessId.argtypes = [W.HWND, C.POINTER(W.DWORD)]
        self.k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
        self.k.OpenProcess.restype = W.HANDLE
        self.k.QueryFullProcessImageNameW.argtypes = [W.HANDLE, W.DWORD, W.LPWSTR, C.POINTER(W.DWORD)]
        self.k.CloseHandle.argtypes = [W.HANDLE]
        self.u.GetClientRect.argtypes = [W.HWND, C.POINTER(W.RECT)]
        self.u.ClientToScreen.argtypes = [W.HWND, C.POINTER(W.POINT)]
        self.u.IsIconic.argtypes = [W.HWND]
        self.u.SetProcessDPIAware()

    def target(self, background=False):
        hwnd = self.u.GetForegroundWindow()
        if background:
            self.u.FindWindowW.argtypes = [W.LPCWSTR, W.LPCWSTR]
            self.u.FindWindowW.restype = W.HWND
            hwnd = self.u.FindWindowW(None, 'Forza Horizon 6')
        pid = W.DWORD()
        self.u.GetWindowThreadProcessId(hwnd, C.byref(pid))
        handle = self.k.OpenProcess(0x1000, False, pid.value)
        if not handle:
            return None
        try:
            name, n = C.create_unicode_buffer(32768), W.DWORD(32768)
            if not self.k.QueryFullProcessImageNameW(handle, 0, name, C.byref(n)):
                return None
            if Path(name.value).name.lower() != 'forzahorizon6.exe':
                return None
        finally:
            self.k.CloseHandle(handle)
        if self.u.IsIconic(hwnd):
            return None
        r, p = W.RECT(), W.POINT()
        if not self.u.GetClientRect(hwnd, C.byref(r)) or not self.u.ClientToScreen(hwnd, C.byref(p)):
            return None
        if r.right < 320 or r.bottom < 200:
            return None
        return dict(left=p.x, top=p.y, width=r.right, height=r.bottom)

    def down(self, vk):
        return bool(self.u.GetAsyncKeyState(vk) & 0x8000)


class Keyboard:
    """Scan-code input compatible with DirectInput games; release only our held keys."""
    def __init__(self, win):
        class KI(C.Structure):
            _fields_ = [('vk', W.WORD), ('scan', W.WORD), ('flags', W.DWORD),
                        ('time', W.DWORD), ('extra', C.c_size_t)]
        class MI(C.Structure):
            _fields_ = [('dx', W.LONG), ('dy', W.LONG), ('data', W.DWORD),
                        ('flags', W.DWORD), ('time', W.DWORD), ('extra', C.c_size_t)]
        class U(C.Union):
            _fields_ = [('ki', KI), ('mi', MI)]
        class INPUT(C.Structure):
            _anonymous_ = ('u',)
            _fields_ = [('type', W.DWORD), ('u', U)]
        self.KI, self.INPUT, self.win = KI, INPUT, win
        win.u.SendInput.argtypes = [W.UINT, C.POINTER(INPUT), C.c_int]
        self.held = set()

    def set(self, desired):
        for scan in self.held ^ desired:
            event = self.INPUT(type=1)
            event.ki = self.KI(0, scan, 8 | (2 if scan not in desired else 0), 0, 0)
            if self.win.u.SendInput(1, C.byref(event), C.sizeof(event)) != 1:
                raise OSError('Game input failed; run game and adapter at the same privilege level')
            if scan in desired:
                self.held.add(scan)
            else:
                self.held.discard(scan)


def controls(decision, speed, limit, threshold):
    values = (decision.confidence, decision.steer, decision.target_kmh, speed)
    if not all(math.isfinite(v) for v in values):
        return 0., 0., 1.
    if decision.confidence < threshold or decision.mode in ('lost', 'blocked') or decision.reverse:
        return 0., 0., 1. if speed > 1 else 0.
    target = min(limit, max(0., decision.target_kmh))
    error = target - speed
    return (max(-.65, min(.65, decision.steer)),
            min(.35, max(0., error * .06)) if error > 1 else 0.,
            min(1., -error * .12) if error < -1 else 0.)


def forward_only(command, telemetry):
    """The brake is also reverse in FH6: release it before reaching a stop.

    Signed local velocity, not speed magnitude, distinguishes backing up.
    The forward pedal cancels reverse on the game's automatic transmission.
    """
    steer, gas, brake = command
    forward = telemetry['forward_kmh']
    if forward < -.5:
        return 0., .55, 0.
    if forward < 3.:
        brake = 0.
    return steer, gas, brake


def final_forward_pedals(command, telemetry, handbrake=False, emergency=False):
    """The final smoothed LT report must not drive reverse at a low speed."""
    if telemetry['forward_kmh'] < 3.:
        steer,gas,_=command
        # Emergency stopping uses the parking brake at this low speed; LT
        # would instead select reverse after the car reaches a standstill.
        return (steer,0. if emergency else gas,0.), bool(handbrake or emergency)
    return command,handbrake


def parse_control_request(request):
    """Validate bounded runtime commands before changing the active controller."""
    action = request['action']
    if action not in ('drive', 'test', 'hold', 'recover'):
        raise ValueError('Unknown control action')
    values = tuple(float(request.get(k, default)) for k, default in
                   (('max_kmh', 12), ('seconds', 4), ('steer', 0), ('throttle', .5)))
    limit, duration, steer, throttle = values
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Control request must be finite')
    if not (1 <= limit <= MAX_CRUISE_KMH and 1 <= duration <= 6 and
            -.8 <= steer <= .8 and 0 <= throttle <= .5):
        raise ValueError('Control request outside allowed range')
    return action, limit, duration, steer, throttle


def steering_input(steer, inner, outer):
    """Compensate the observed game dead zone without changing game settings."""
    if abs(steer) < .008:
        return 0.
    return math.copysign(inner+(outer-inner)*min(1., abs(steer)), steer)


def stop_command(forward_kmh):
    # Foot brake first; pulling the handbrake while turning can cause a spin.
    if forward_kmh < -.3:
        return (0.,0.,0.),True
    if forward_kmh > 3:
        return (0., 0., min(.65, max(.2, (forward_kmh-1)*.12))), False
    return (0., 0., 0.), True


def control_frame_fresh(age,reverse=False):
    # Bounded low-speed backing is governed by fresh distance/gear telemetry.
    # Normal forward driving retains the stricter image latency requirement.
    return 0.<=age<=(1.0 if reverse else .5)


class LinearRamp:
    """Per-second limits on actual stick/pedal travel."""
    def __init__(self):
        self.value = (0., 0., 0.)
        self.steer_rate = .65

    def step(self, target, dt):
        dt = max(0., min(.1, dt))
        up, down = (self.steer_rate, .28, .28), (self.steer_rate, .65, 1.2)
        self.value = tuple(a+max(-fall*dt, min(rise*dt, b-a))
                           for a, b, rise, fall in zip(self.value, target, up, down))
        # Do not press the accelerator and brake together during a transition.
        if self.value[2] > .01:
            self.value = (self.value[0], 0., self.value[2])
        return self.value


class SpeedController:
    """Gentle PI throttle with a coasting band and delayed overspeed braking."""
    def __init__(self):
        self.trim = 0.
        self.target = 0.
        self.overspeed = 0.
        self.unreliable_for = 0.
        self.last_steer = 0.
        self.low_evidence_for = 0.
        self.uncertainty_for = 0.
        self.previous_speed = None
        self.measured_acceleration = 0.
        self.desired_acceleration = 0.
        self.relaunch = RelaunchAssist()

    def step(self, decision, speed, limit, threshold, dt):
        dt = max(0., min(.5, dt))
        if self.previous_speed is None:
            if self.target==0.:
                self.target=max(0.,speed)
        elif dt>.01:
            measured=max(-6.,min(6.,(speed-self.previous_speed)/(3.6*dt)))
            self.measured_acceleration+=dt/(.35+dt)*(measured-self.measured_acceleration)
        self.previous_speed=speed
        trusted = (decision.confidence >= threshold and decision.mode not in ('lost', 'blocked')
                   and not decision.reverse)
        if not trusted:
            self.relaunch.reset()
            self.uncertainty_for=0.
            self.unreliable_for += dt
            if self.unreliable_for>.35:
                self.target = min(self.target,speed)
            self.overspeed = self.trim = 0.
            # A single blurred frame coasts; sustained loss brakes progressively.
            if self.unreliable_for <= .35:
                return (self.last_steer,0.,0.)
            return (0.,0.,min(.5,.15+.3*(self.unreliable_for-.35)) if speed > 3 else 0.)
        self.unreliable_for = 0.
        desired = min(limit, max(0., decision.target_kmh))
        uncertainty_drop=(getattr(decision,'uncertainty_limited',False) and
                          decision.mode=='lane_inferred' and 0.<desired<self.target)
        self.uncertainty_for=self.uncertainty_for+dt if uncertainty_drop else 0.
        if uncertainty_drop and self.uncertainty_for<=.6:
            # Brief missing paint is not a newly observed tight bend. Coast
            # at no more than the current speed while waiting for confirmation.
            # Actual curves, traffic, zero targets and prolonged loss bypass it.
            desired=max(desired,min(self.target,speed))
            self.target=min(self.target,speed)
        weak_drop=(decision.confidence<.75 and desired<self.target and
                   decision.mode in ('lane_temporal','paint_temporal','minimap_supported'))
        self.low_evidence_for=self.low_evidence_for+dt if weak_drop else 0.
        if weak_drop and self.low_evidence_for<=.2:
            desired=self.target
        if desired>self.target and speed>self.target:
            # A cleared hazard must not keep braking toward an obsolete low
            # smoothed setpoint while the permitted speed has already risen.
            self.target=min(desired,speed)
        self.target += max(-8.*dt, min(6.*dt, desired-self.target))
        error = self.target-speed
        self.desired_acceleration=max(0.,min(1.6,.65*error/3.6))
        acceleration_error=self.desired_acceleration-self.measured_acceleration
        self.trim=max(-.08,min(.12,self.trim+.025*acceleration_error*dt))
        braking_error=desired-speed
        self.overspeed = self.overspeed+dt if braking_error < -4 else 0.
        brake = min(.30, (-braking_error-4)*.045) if self.overspeed > 1. else 0.
        # Start coasting when a lower limit is observed. The smoothed setpoint
        # must not keep adding throttle before braking late for the same bend.
        throttle = min(.6,max(0.,.04+self.desired_acceleration/4.+.06*acceleration_error+self.trim)) if desired > 0. and error > -.8 and desired >= speed-.8 else 0.
        steer = max(-.65, min(.65, decision.steer))
        self.last_steer = steer
        if decision.mode == 'traffic_stop':
            self.relaunch.reset()
            return (steer,0.,min(.4,.04*speed) if speed > 1 else 0.)
        return self.relaunch.step((steer, (0. if brake else throttle), brake),
                                  speed, desired, dt)


def crop_letterbox(frame):
    """Remove near-black outer bars; retain a usable central image or reject it."""
    import numpy as np
    h, w = frame.shape[:2]
    # A native 16:9 frame needs no crop. Dark cockpit panels are not letterboxes.
    if abs(w / h - 16 / 9) < .02:
        if np.count_nonzero(np.max(frame, axis=2) > 12) < h * w * .02:
            raise ValueError('Black/loading frame')
        return frame
    light = np.max(frame, axis=2) > 12
    rows = np.flatnonzero(light.mean(axis=1) > .08)
    cols = np.flatnonzero(light.mean(axis=0) > .08)
    if not len(rows) or not len(cols):
        raise ValueError('Black/loading frame')
    cropped = frame[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    h, w = cropped.shape[:2]
    if w < 320 or h < 180 or not 1.5 < w/h < 2.5:
        raise ValueError('Unsupported viewport; use a 16:9 game window')
    return cropped


class Actuator(threading.Thread):
    """100 Hz virtual-only background output; physical hotkeys belong to legacy mode."""
    def __init__(self, win, telemetry, drive, background=False):
        super().__init__(daemon=True)
        self.win, self.telemetry, self.drive = win, telemetry, drive
        self.background = background
        self.pad = None
        if background and drive:
            from .forza_background import AnalogGamepad
            self.pad = AnalogGamepad()
        elif not background:
            self.keys = Keyboard(win)
        self.command = (0., 0., 0.)
        self.updated = 0.
        self.armed = False
        self.stop_event = threading.Event()
        self.failure = None
        self.previous = False
        self.handbrake = False
        self.requested = False
        self.stable_since = None
        self.reason = 'not started'
        self.steer_inner, self.steer_outer = 0., 1.
        self.smooth = False
        self.allow_reverse = False
        self.last_output = (0.,0.,0.)
        self.output = None
        self.gear_request_at = -10.
        self.soft_rearm = False
        self.rearm_frames = 0
        self.rearm_capture = None
        self.recently_ready = None
        self.timing = {}

    def publish(self, command, handbrake, allow_reverse, captured, emergency=False, forward_requested=True):
        # Publish one complete frame; the output thread must not see half an update.
        self.output = (tuple(command), bool(handbrake), bool(allow_reverse), bool(emergency),bool(forward_requested))
        self.updated = captured

    def set_enabled(self, enabled):
        self.requested = bool(enabled and self.drive)
        self.armed = False
        self.stable_since = None
        self.soft_rearm = False
        self.rearm_frames = 0
        self.recently_ready = None
        if not self.requested:
            self.command = (0.,0.,0.)
            self.output = ((0.,0.,0.),False,False,False,False)
            self.reason = 'dedicated control stop'

    def run(self):
        from .forza_analog import AnalogSmoother
        ramp = AnalogSmoother()
        from .forza_lateral import SteeringServo
        steering_servo = SteeringServo()
        previous_tick = time.monotonic()
        try:
            while not self.stop_event.wait(.01):
                now = time.monotonic()
                dt, previous_tick = now-previous_tick, now
                if self.background:
                    # Never query or hook the user's keyboard while they chat.
                    arm_key = stop_key = exit_key = escape_key = False
                else:
                    arm_key = self.win.down(0x77)
                    stop_key = self.win.down(0x78)
                    exit_key = self.win.down(0x79)
                    escape_key = self.win.down(0x1B)
                focus = self.win.target(background=True) is not None if self.background else self.win.target() is not None
                live_value=self.telemetry.last or {}
                live_game = focus and self.telemetry.fresh() and live_value.get('race',False)
                live_game = live_game and not live_value.get('race_position', 0) and not live_value.get('lap_number', 0)
                reversing=bool(self.output is not None and self.output[2])
                ready = live_game and control_frame_fresh(now-self.updated,reversing)
                self.timing=dict(frame_age_ms=round((now-self.updated)*1000,1),
                                 telemetry_age_ms=round((now-getattr(self.telemetry,'received',now))*1000,1),
                                 output_interval_ms=round(dt*1000,1),ready=ready)
                if exit_key:  # F10, legacy foreground mode only
                    self.requested = False
                    self.stop_event.set()
                if stop_key:
                    self.requested = False
                    self.armed = False
                    self.reason = 'F9 stop'
                    self.stable_since = None
                elif escape_key or not ready:
                    self.soft_rearm=bool(self.requested and live_game and now-self.updated<2.
                                         and (self.armed or self.soft_rearm))
                    if not self.soft_rearm:
                        self.recently_ready=None
                    self.armed = False
                    self.stable_since = None
                    self.rearm_frames=0
                    self.rearm_capture=None
                    self.reason = ('Escape/menu' if escape_key else
                                   'game unavailable' if not focus else
                                   'telemetry stale' if not self.telemetry.fresh() else 'image/decision stale')
                    if not self.background:
                        self.requested = False
                elif arm_key and not self.previous:
                    self.requested = self.drive
                    self.armed = self.drive
                elif self.background and self.requested and not self.armed:
                    if self.stable_since is None:
                        self.stable_since = now
                    if self.updated!=self.rearm_capture:
                        self.rearm_capture=self.updated
                        self.rearm_frames+=1
                    quick_resume=(self.soft_rearm and self.recently_ready is not None and
                                  now-self.recently_ready<.35)
                    if quick_resume or (now-self.stable_since >= (.25 if self.soft_rearm else 1.) and (
                            not self.soft_rearm or self.rearm_frames>=2)):
                        self.armed = True
                        self.soft_rearm=False
                if self.armed:
                    self.reason = 'driving'
                    self.recently_ready=now
                self.previous = arm_key
                desired = set()
                if self.background:
                    published, handbrake, allow_reverse, emergency, forward_requested = self.output if self.output is not None else (self.command,self.handbrake,self.allow_reverse,False,self.requested)
                    command = published if self.armed and not self.stop_event.is_set() else (0., 0., 0.)
                    reverse_speed = -live_value.get('forward_kmh',0)
                    reverse_coast = allow_reverse and reverse_speed > 5.
                    reverse_limit = allow_reverse and reverse_speed > 8.
                    if reverse_coast:
                        command = (command[0],0.,0.)
                    if reverse_limit:
                        # Bound low-speed reversing independently of vision cadence.
                        command,handbrake = (0.,0.,0.),True
                    # Last-moment brake interlock uses the newest telemetry.
                    if self.armed and not handbrake and not allow_reverse:
                        command = forward_only(command, live_value)
                    if self.pad:
                        gear_buttons = ()
                        if (self.armed and forward_requested and not allow_reverse and not emergency and
                                live_value.get('gear',1) == 0 and
                                abs(live_value.get('forward_kmh',0)) < .3):
                            command = (0.,0.,0.)
                            handbrake = abs(live_value.get('forward_kmh',0)) > .3
                            if not handbrake:
                                if now-self.gear_request_at >= 1.:
                                    self.gear_request_at = now
                                if now-self.gear_request_at < .15:
                                    gear_buttons = ('B',)
                                    handbrake = False
                        if self.armed and not allow_reverse:
                            stick = steering_servo.step(command[0],live_value.get('steer',0)/127.,dt,
                                                        self.steer_inner,self.steer_outer)
                        else:
                            steering_servo.trim = 0.
                            stick = steering_input(command[0],self.steer_inner,self.steer_outer)
                        command = (stick, command[1], command[2])
                        if self.smooth and self.armed:
                            command = ramp.step(command,dt,live_value.get('speed',0.),
                                                emergency=emergency,reverse=allow_reverse)
                            if reverse_coast:
                                command = (command[0],0.,0.)
                                if reverse_limit:
                                    command = (0.,0.,0.)
                                ramp.reconcile(command)
                        else:
                            ramp.reset(command)
                        if self.armed and not allow_reverse:
                            # Smoothing can retain the previous brake even
                            # after forward_only released it. Interlock the
                            # actual outgoing report, using the latest packet.
                            command,handbrake=final_forward_pedals(
                                command,live_value,handbrake,emergency)
                            ramp.reconcile(command)
                        parking = ready and ((handbrake and (allow_reverse or
                            live_value.get('forward_kmh',0) < -.3)) or
                            (live_value.get('speed',0) < 3 and
                             (handbrake or (not self.armed and not self.soft_rearm))))
                        if gear_buttons:
                            command = (0.,0.,0.)
                            ramp.reset(command)
                            self.pad.set(command,handbrake=parking,buttons=gear_buttons)
                        else:
                            self.pad.set(command,handbrake=parking)
                        self.last_output = command
                    continue
                if self.armed and not self.stop_event.is_set():
                    steer, throttle, brake = forward_only(self.command, live_value) if 'forward_kmh' in live_value else self.command
                    phase = (now % .12) / .12
                    if abs(steer) > .035 and phase < abs(steer):
                        desired.add(0x20 if steer > 0 else 0x1E)  # D / A
                    if brake > 0 and phase < brake:
                        desired.add(0x1F)  # S
                    elif throttle > 0 and phase < throttle:
                        desired.add(0x11)  # W
                self.keys.set(desired)
        except Exception as exc:
            self.failure = exc
            self.stop_event.set()
        finally:
            if self.pad:
                self.pad.close()
            elif not self.background:
                self.keys.set(set())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--drive', action='store_true', help='Enable dedicated drive controls (default: observe only)')
    ap.add_argument('--config', default=str(ROOT/'configs/forza_horizon6.json'))
    ap.add_argument('--seconds', type=float, default=0)
    ap.add_argument('--arm-after', type=float, help='One-time auto start after N seconds of a ready game (requires --drive)')
    ap.add_argument('--max-kmh', type=float, help='Temporarily lower the configured speed limit')
    ap.add_argument('--background', action='store_true', help='Window capture + analog gamepad; chat can remain focused')
    ap.add_argument('--record-hz', type=float, default=5, help='Save raw frames and telemetry for later learning')
    ap.add_argument('--forward-test', type=float, help='Bounded analog acceleration check (1-6 seconds, no steering)')
    ap.add_argument('--source', choices=['local_ai', 'game_teacher', 'human'], default='local_ai', help='Label who produces driving actions in the recording')
    args = ap.parse_args()
    if args.arm_after is not None and (not args.drive or args.arm_after < 1):
        ap.error('--arm-after requires --drive and at least 1 second')
    if args.forward_test is not None and (not args.drive or not args.background or not 1 <= args.forward_test <= 6):
        ap.error('--forward-test requires --drive --background and 1-6 seconds')
    import cv2
    import mss
    import numpy as np
    from .driver import MonoDriver
    from .viz import draw
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    settings = cfg['forza']
    if args.max_kmh is not None:
        if not 0 < args.max_kmh <= settings['max_kmh']:
            ap.error('--max-kmh must be positive and no greater than the configured limit')
        settings['max_kmh'] = args.max_kmh
    drv = MonoDriver(cfg, device='cpu')
    from .forza_live_model import LivePathModel
    live_model=LivePathModel(ROOT,cfg['models']['pathnet'],config_path=Path(args.config).resolve())
    win = Windows()
    telemetry = Telemetry(settings['port'])
    actuator = Actuator(win, telemetry, args.drive, args.background)
    actuator.steer_inner = float(settings.get('steering_inner', 0))
    actuator.steer_outer = float(settings.get('steering_outer', 1))
    actuator.smooth = True
    if not 0 <= actuator.steer_inner < actuator.steer_outer <= 1:
        raise ValueError('Invalid analog steering calibration')
    folder = ROOT/'out'/time.strftime('fh6_%Y%m%d_%H%M%S')
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'frames').mkdir(exist_ok=True)
    import hashlib
    model_hashes = {}
    for model_file in (ROOT/'models').glob('*.onnx'):
        with model_file.open('rb') as handle:
            model_hashes[model_file.name] = hashlib.file_digest(handle,'sha256').hexdigest()
    (folder/'manifest.json').write_text(json.dumps(dict(config=cfg,arguments=vars(args),
        models_sha256=model_hashes,created_utc=time.time(),record_hz=args.record_hz,
        training_label_status='unreviewed',game='Forza Horizon 6',mode='free roam',
        input_isolation='virtual Xbox only; no physical keyboard polling' if args.background else 'legacy keyboard',
        controls='external analog gamepad; built-in AutoDrive disabled'),indent=2),encoding='utf-8')
    print('FH6 LOCAL AI | '+('DRIVE: dedicated controls, physical keyboard ignored' if args.drive and args.background else
                               'DRIVE: F8 arm, F9 stop, F10 exit' if args.drive else 'OBSERVE ONLY'))
    print('Data Out: ON, IP 127.0.0.1, port', settings['port'])
    print('Logs:', folder, flush=True)
    actuator.start()
    telemetry.start()
    started, report = time.monotonic(), 0.
    ready_since, auto_started = None, False
    armed_at = None
    recorded, next_record = 0, 0.
    bgcapture = None
    control_path = ROOT/'out'/'control.json'
    control_version = None
    holding = False
    test_steer, test_throttle = 0., .5
    from . import forza_lane as lane_module
    from .forza_recovery import Recovery
    import importlib
    semantic = settings.get('semantic_road', False)
    if semantic:
        import subprocess
        worker_python = ROOT/'.train-venv'/'Scripts'/'python.exe'
        if not worker_python.exists():
            raise RuntimeError('Configured GPU perception environment is missing')
        def start_vision():
            with (ROOT/'out'/'vision.log').open('a') as output,(ROOT/'out'/'vision.err').open('a') as error:
                return subprocess.Popen([str(worker_python),'-m','mono_drive.forza_vision'],cwd=ROOT,
                    creationflags=subprocess.CREATE_NO_WINDOW,stdout=output,stderr=error)
        vision_process = start_vision()
        vision_restart_at = time.monotonic()+15.
    lane_fusion = lane_module.LaneFusion(semantic=semantic)
    lane_version = Path(lane_module.__file__).stat().st_mtime_ns
    recovery = Recovery()
    recovery_requested = False
    recovery_distance = None
    reverse_only = False
    reverse_steer = 0.
    lane, guidance = None, None
    left_lane = settings.get('driving_side') == 'left'
    speed_controller = SpeedController()
    from .forza_vehicle import VehicleResponse
    vehicle_response=VehicleResponse()
    control_tick = time.monotonic()
    heartbeat_at = 0.
    collision = None
    last_decision = {}
    from .forza_brake import BrakeDemandFilter
    brake_filter=BrakeDemandFilter()
    try:
        if args.background:
            from .forza_background import GameCapture
            bgcapture = GameCapture()
        with mss.MSS() as capture, (folder/'decisions.jsonl').open('w', encoding='utf-8') as log:
            while not actuator.stop_event.is_set():
                now = time.monotonic()
                if now-heartbeat_at >= .1:
                    import os
                    heartbeat = ROOT/'out'/'driver-heartbeat.json'
                    temporary = heartbeat.with_suffix('.tmp')
                    temporary.write_text(json.dumps(dict(pid=os.getpid(),time=time.time(),
                        run=str(folder),armed=actuator.armed,state=actuator.reason,**last_decision)),encoding='utf-8')
                    try:
                        temporary.replace(heartbeat)
                    except PermissionError:
                        pass  # A concurrent status reader must not stop driving.
                    heartbeat_at = now
                if semantic and now >= vision_restart_at:
                    vision_restart_at = now+15.
                    scene_file = ROOT/'out'/'scene.npz'
                    fresh_scene = scene_file.exists() and time.time()-scene_file.stat().st_mtime < 2.
                    if not fresh_scene and vision_process.poll() is not None and not (ROOT/'out'/'vision.stop').exists():
                        print('Restarting local GPU perception worker.',flush=True)
                        vision_process = start_vision()
                if args.drive and control_path.exists():
                    version = control_path.stat().st_mtime_ns
                    if version != control_version:
                        control_version = version
                        try:
                            request = json.loads(control_path.read_text(encoding='utf-8-sig'))
                            action, limit, duration, test_steer, test_throttle = parse_control_request(request)
                            if not args.drive or not args.background:
                                raise ValueError('Runtime control requires background driving mode')
                            settings['max_kmh'] = limit
                            holding = action == 'hold'
                            recovery_requested = action == 'recover'
                            recovery_distance = float(request['back_m']) if 'back_m' in request else None
                            reverse_only = bool(request.get('reverse_only',False))
                            reverse_steer = float(request.get('reverse_steer',0.))
                            if not math.isfinite(reverse_steer) or not -.8 <= reverse_steer <= .8:
                                raise ValueError('Invalid reverse steering')
                            if recovery_distance is not None and not .3 <= recovery_distance <= 10.:
                                raise ValueError('Back distance outside allowed range')
                            if holding:
                                recovery.phase = ''
                            args.forward_test = duration if action == 'test' else None
                            actuator.handbrake = holding
                            actuator.set_enabled(True)
                            armed_at = None
                            auto_started = True
                            print('Control:', action, 'limit', limit, flush=True)
                            with (folder/'control_events.jsonl').open('a',encoding='utf-8') as events:
                                events.write(json.dumps(dict(wall_time=time.time(),monotonic_s=now,request=request))+'\n')
                        except (ValueError, KeyError, TypeError) as exc:
                            print('Rejected control request:', exc, flush=True)
                if (folder/'stop').exists():
                    print('Stop requested.', flush=True)
                    break
                if args.seconds and now-started >= args.seconds:
                    break
                value = telemetry.last
                region = win.target(background=True) if args.background else win.target()
                if region is None or not telemetry.fresh() or not value['race']:
                    actuator.armed = False
                    ready_since = None
                    if now-report > 2:
                        print('Waiting: FH6 window + live Data Out.', telemetry.error, flush=True)
                        report = now
                    time.sleep(.05)
                    continue
                # Race position/lap information rejects races; visually confirm free roam as well.
                if value['race_position'] != 0 or value['lap_number'] != 0:
                    actuator.requested = False
                    actuator.armed = False
                    if now-report > 2:
                        print('Free-roam-only: race detected; stopped.', flush=True)
                        report = now
                    time.sleep(.05)
                    continue
                t0 = time.monotonic()
                frame, frame_time = bgcapture.grab() if bgcapture else (np.asarray(capture.grab(region))[:, :, :3].copy(), t0)
                if frame is None:
                    actuator.armed = False
                    ready_since = None
                    time.sleep(.03)
                    continue
                try:
                    frame = crop_letterbox(frame)
                except ValueError as exc:
                    actuator.armed = False
                    if now-report > 2:
                        print(str(exc), flush=True)
                        report = now
                    continue
                replacement=live_model.poll(time.time())
                if replacement is not None:
                    drv.pathnet=replacement
                    cfg['models']['pathnet']=str(live_model.active)
                    print('Validated path model loaded; driving input retained.',flush=True)
                decision = drv.step(frame, speed_kmh=value['speed'])
                vehicle_response.update(value,frame_time)
                if semantic:
                    ego_path = ROOT/'out'/'ego.json'
                    ego_temp = ROOT/'out'/'ego.tmp.json'
                    ego_temp.write_text(json.dumps(dict(time=frame_time,position=value['position'],
                                                        yaw=value['yaw'],speed=value['speed'],
                                                        car_ordinal=value['car_ordinal'])))
                    try:
                        ego_temp.replace(ego_path)
                    except PermissionError:
                        pass
                model_confidence, model_steer = decision.confidence, decision.steer
                new_lane_version = Path(lane_module.__file__).stat().st_mtime_ns
                if new_lane_version != lane_version:
                    lane_version = new_lane_version
                    try:
                        lane_module = importlib.reload(lane_module)
                        lane_fusion = lane_module.LaneFusion(semantic=semantic)
                        print('Lane model hot reloaded; controller stays connected.', flush=True)
                    except Exception as exc:
                        print('Lane reload rejected:', exc, flush=True)
                try:
                    guidance = lane_fusion.update(frame, decision, dict(value,frame_capture_time=frame_time), time.monotonic()) if left_lane else None
                except (ValueError,IndexError,FloatingPointError) as error:
                    guidance = None
                    if now-report > 1:
                        print('Rejected invalid geometry:',str(error),flush=True)
                lane = guidance.lane if guidance else None
                if left_lane:
                    if guidance is not None:
                        decision.steer = float(guidance.steer)
                        decision.confidence = float(guidance.confidence)
                        decision.target_kmh = min(settings['max_kmh'], guidance.max_kmh)
                        decision.mode = guidance.source
                        decision.uncertainty_limited = guidance.uncertainty_limited
                    else:
                        decision.steer, decision.confidence, decision.target_kmh = 0., 0., 0.
                        decision.mode = 'lost'
                elapsed = time.monotonic()-t0
                control_now = time.monotonic()
                actuator.command = forward_only(speed_controller.step(decision, value['speed'], settings['max_kmh'],
                    settings['min_confidence'], control_now-control_tick), value)
                control_tick = control_now
                if args.forward_test is not None:
                    actuator.command = (test_steer, test_throttle if value['forward_kmh'] < settings['max_kmh'] else 0., 0.)
                if holding:
                    actuator.command, actuator.handbrake = stop_command(value['forward_kmh'])
                    speed_controller.trim = speed_controller.overspeed = 0.
                    speed_controller.relaunch.reset()
                    speed_controller.target = value['speed']
                else:
                    actuator.handbrake = (args.forward_test is None and
                    actuator.command[1] == 0 and value['speed'] < 3 and
                    (decision.mode in ('blocked','traffic_stop') or speed_controller.unreliable_for > .35))
                # Timestamp of capture, rather than end of inference, enforces latency limit.
                actuator.updated = frame_time
                if ready_since is None:
                    ready_since = t0
                if args.arm_after is not None and not auto_started and time.monotonic()-ready_since >= args.arm_after:
                    actuator.armed = True
                    actuator.requested = True
                    auto_started = True
                    armed_at = time.monotonic()
                    print('One-time auto start; use dedicated stop control.' if args.background else
                          'One-time auto start; F9 stops, F10 exits.', flush=True)
                if args.forward_test is not None and actuator.armed:
                    if armed_at is None:
                        armed_at = time.monotonic()
                    if time.monotonic()-armed_at >= args.forward_test:
                        actuator.command, actuator.handbrake = stop_command(value['forward_kmh'])
                actuator.allow_reverse = False
                if args.drive and actuator.requested and not holding and args.forward_test is None:
                    recovery_now = time.monotonic()
                    direction = lane_module.road_direction(frame)
                    guidance_ok = guidance is not None and guidance.source not in ('road_reentry','minimap_supported')
                    traffic_blocked = decision.mode == 'traffic_stop' or bool(collision and collision.emergency)
                    surface_blocked=recovery.surface_blocks(decision.mode,recovery_requested)
                    if surface_blocked:
                        recovery.phase=''
                    if speed_controller.relaunch.blocks_reverse:
                        recovery.stall_since = None
                    if not surface_blocked and (recovery_requested or (not recovery.phase and not speed_controller.relaunch.blocks_reverse and recovery.observe_stall(value, actuator.command[1], recovery_now,
                                                            guidance_ok=guidance_ok,traffic_blocked=traffic_blocked))):
                        recovery.begin(value, direction, recovery_now, distance_m=recovery_distance,reverse_only=reverse_only,
                                       reverse_steer=reverse_steer,manual=recovery_requested)
                        recovery_requested = False
                    recovered = recovery.update(value, guidance, direction, recovery_now)
                    if recovered is not None:
                        speed_controller.relaunch.reset()
                        actuator.command, actuator.handbrake, actuator.allow_reverse = recovered
                        decision.mode = 'recover_'+recovery.phase
                    if recovery.phase == 'failed':
                        actuator.requested = False
                        actuator.armed = False
                    if recovery.phase == 'complete_reverse':
                        holding = True
                        actuator.command,actuator.handbrake = stop_command(value['forward_kmh'])
                        recovery.phase = ''
                from .forza_collision import forward_collision
                collision = None
                live_scene = lane_fusion.scene_reader.value if semantic else None
                static = None
                car_brake=wall_brake=0.
                planned_path=None
                if decision.mode in ('offroad_surface','sidewalk_boundary','known_failed_approach'):
                    actuator.command,actuator.handbrake=stop_command(value['forward_kmh'])
                    actuator.allow_reverse=False
                    speed_controller.target=0.
                nominal_brake=actuator.command[2]
                if (live_scene and time.monotonic()-live_scene['captured'] < .5 and not actuator.allow_reverse):
                    planned=lane_fusion.scene_planner.latest
                    planned_path=(planned.path if planned is not None and guidance is not None
                                  and (guidance.confidence>=.75 or (guidance.confidence>=.55 and planned.road_score>.55)) and not recovery.phase
                                  and not holding and args.forward_test is None else None)
                    measured_steer=value.get('steer',0)/127.
                    collision = forward_collision(live_scene['cars'],value['speed'],measured_steer,path=planned_path)
                    from .forza_static import static_risk
                    static = static_risk(live_scene,value['speed'],measured_steer,path=planned_path)
                    car_brake,car_urgent=brake_filter.step(('car',collision.vehicle_id),collision.brake,
                        collision.emergency,collision.immediate,control_now)
                    wall_brake,wall_urgent=brake_filter.step('static',static.brake,
                        static.emergency,static.immediate,control_now)
                    safety_brake=max(car_brake,wall_brake)
                    urgent=car_urgent or wall_urgent
                    if max(collision.brake,static.brake)>.01 or urgent:
                        from .forza_brake import hazard_pedals
                        actuator.command=hazard_pedals(actuator.command,value['speed'],
                            max(collision.brake,static.brake),safety_brake,urgent)
                        actuator.handbrake=urgent and value['speed']<=1.
                        decision.mode=('static_brake' if wall_urgent else 'collision_brake') if urgent else (
                            ('traffic_slow' if car_brake>=wall_brake else 'static_slow') if safety_brake>.01 else
                            ('traffic_caution' if collision.brake>=static.brake else 'static_caution'))
                else:
                    urgent=False
                actuator.publish(actuator.command,actuator.handbrake,actuator.allow_reverse,frame_time,
                                 emergency=urgent,
                                 forward_requested=not holding and not actuator.allow_reverse)
                last_decision = dict(sample_time=time.time(),position=value['position'],speed_kmh=value['speed'],
                                     car_ordinal=value['car_ordinal'],vehicle=vehicle_response.name,
                                     mode=decision.mode,target_kmh=decision.target_kmh,
                                     cruise_kmh=settings['max_kmh'],relaunch=speed_controller.relaunch.status(),
                                     path_model=str(live_model.active),model_hot_reload=True,model_runtime=live_model.status,
                                     learned_path=lane_fusion.learned_prior.status if semantic else None,
                                     lane_margin_m=lane_fusion.scene_planner.latest.guard_margin_m if semantic and lane_fusion.scene_planner.latest else None,
                                     recovery_phase=recovery.phase,gear=value['gear'],
                                     median_side=lane_fusion.carriageway_status.get('median_side') if semantic else None,
                                     wrongway_confirmed=lane_fusion.carriageway_status.get('wrongway_confirmed',False) if semantic else False,
                                     surface=lane_fusion.surface_guard.status if semantic else None,
                                     failure_memory=lane_fusion.failure_guard.status if semantic else None,
                                     route_memory=lane_fusion.route_memory.status if semantic else None,
                                     actual_accel=value['accel'],actual_brake=value['brake'])
                frame_name = None
                if args.record_hz > 0 and now >= next_record:
                    frame_name = f'frames/{recorded:06d}.jpg'
                    cv2.imwrite(str(folder/frame_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    recorded += 1
                    next_record = now + 1/args.record_hz
                log.write(json.dumps(dict(**decision.to_row(), speed_kmh=value['speed'],
                                          car_ordinal=value['car_ordinal'],vehicle=vehicle_response.name,
                                          projection=(lane_fusion.scene_reader.value or {}).get('projection') if semantic else None,
                                          captured_s=frame_time,decided_s=control_now,training_label_status='unreviewed',
                                          forward_kmh=value['forward_kmh'], gear=value['gear'], frame=frame_name,
                                          background=args.background, telemetry_stamp=value['stamp'],
                                          driving_source=args.source,
                                          armed=actuator.armed, inference_ms=round(elapsed*1000, 1),
                                          state=actuator.reason, frame_age_ms=round((time.monotonic()-frame_time)*1000),
                                          foreground=win.target() is not None,
                                          actual_accel=value['accel'], actual_brake=value['brake'],
                                          actual_handbrake=value['handbrake'], race_position=value['race_position'],
                                          actual_steer=value['steer'], command=actuator.command,
                                          sent_command=actuator.last_output,
                                          output_timing=actuator.timing,
                                          model_timing_ms=dict(drv.ms),
                                          cruise_kmh=settings['max_kmh'],relaunch=speed_controller.relaunch.status(),
                                          path_model=str(live_model.active),model_runtime=live_model.status,
                                          learned_path=lane_fusion.learned_prior.status if semantic else None,
                                          legacy_obstacle_veto=lane_fusion.legacy_veto_status if semantic else None,
                                          lane=None if lane is None else dict(target_x=float(lane.target_x), center_x=float(lane.center_x),
                                              width=float(lane.lane_width_px), confidence=float(lane.confidence)),
                                          model_confidence=float(model_confidence), model_steer=float(model_steer),
                                          lookahead_m=guidance.lookahead_m if guidance else None,
                                          vehicles=guidance.vehicles if guidance else None,
                                          candidates=guidance.candidates if guidance else None,
                                          candidate_path=lane_fusion.scene_planner.latest.path.tolist() if semantic and lane_fusion.scene_planner.latest else None,
                                          minimap_angle=lane_fusion.scene_planner.latest.minimap_angle if semantic and lane_fusion.scene_planner.latest else None,
                                          minimap_confidence=lane_fusion.scene_planner.latest.minimap_confidence if semantic and lane_fusion.scene_planner.latest else None,
                                          lane_guard=lane_fusion.scene_planner.latest.guard_state if semantic and lane_fusion.scene_planner.latest else None,
                                          verified_route=lane_fusion.scene_planner.latest.verified_route if semantic and lane_fusion.scene_planner.latest else False,
                                          uncertainty_limited=getattr(decision,'uncertainty_limited',False),
                                          lane_memory=lane_fusion.memory.state if semantic else None,
                                          lane_uncertainty_m=lane_fusion.memory.uncertainty if semantic else None,
                                          preview_curvature=lane_fusion.predictive.curvature if semantic else None,
                                          lane_target_m=lane_fusion.predictive.target if semantic else None,
                                          lane_clearance_m=lane_fusion.scene_planner.latest.guard_margin_m if semantic and lane_fusion.scene_planner.latest else None,
                                          lane_edges=lane_fusion.scene_planner.latest.lane_edges if semantic and lane_fusion.scene_planner.latest else None,
                                          predictive_clearance_m=lane_fusion.predictive.margin if semantic else None,
                                          collision_prediction_s=collision.seconds if collision else None,
                                          collision_guard=collision.reason if collision else None,
                                          collision_brake_demand=collision.brake if collision else 0.,
                                          collision_required_decel=collision.required_decel if collision else 0.,
                                          urgent_brake=urgent,
                                          raw_urgent_brake=bool((collision and collision.emergency) or (static and static.emergency)),
                                          static_clearance_m=static.clearance_m if static else None,
                                          static_brake_demand=static.brake if static else 0.,
                                          applied_vehicle_brake=car_brake,applied_static_brake=wall_brake,
                                          planned_collision_path=planned_path is not None,
                                          speed_brake_demand=nominal_brake,
                                          input_profile='smooth_analog_v6',
                                          inferred_lane=bool(lane_fusion.scene_planner.latest.inferred) if semantic and lane_fusion.scene_planner.latest else False,
                                          lane_inference_age_s=lane_fusion.scene_planner.latest.inference_age_s if semantic and lane_fusion.scene_planner.latest else None,
                                          carriageway=lane_fusion.carriageway_status if semantic else None,
                                          surface=lane_fusion.surface_guard.status if semantic else None,
                                          failure_memory=lane_fusion.failure_guard.status if semantic else None,
                                          route_memory=lane_fusion.route_memory.status if semantic else None,
                                          perception_recording=lane_fusion.scene_reader.value.get('recording') if semantic and lane_fusion.scene_reader.value else None,
                                          perception_frame=lane_fusion.scene_reader.value.get('frame') if semantic and lane_fusion.scene_reader.value else None,
                                          smooth_target_kmh=speed_controller.target,
                                          measured_acceleration=speed_controller.measured_acceleration,
                                          desired_acceleration=speed_controller.desired_acceleration,
                                          recovery_phase=recovery.phase, reverse_distance_m=round(recovery.backed_m, 3),
                                          position=value['position'], yaw=value['yaw']))+'\n')
                if now-report > 1:
                    log.flush()
                    cv2.imwrite(str(folder/'latest.jpg'), lane_module.draw_lane(frame, lane, decision.mode) if left_lane else draw(frame, decision, drv, bev=True))
                    print(f"{'DRIVING' if actuator.armed else 'OBSERVE'} {value['speed']:.1f}km/h "
                          f"{decision.mode} confidence={decision.confidence:.2f} {elapsed*1000:.0f}ms", flush=True)
                    report = now
                time.sleep(max(0., 1/settings['fps']-(time.monotonic()-t0)))
    finally:
        actuator.stop_event.set()
        actuator.join(timeout=2)
        telemetry.close()
        live_model.close()
        if bgcapture:
            bgcapture.close()
    if actuator.failure:
        raise actuator.failure
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
