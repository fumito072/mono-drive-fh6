"""Window-only capture and analog gamepad output for the FH6 adapter."""
from __future__ import annotations

import threading
import time


class GameCapture:
    """WGC captures the game even behind the chat; never captures the desktop."""
    def __init__(self, title='Forza Horizon 6'):
        from windows_capture import WindowsCapture
        self.lock = threading.Lock()
        self.frame = None
        self.received = 0.
        self.closed = False
        self.capture = WindowsCapture(cursor_capture=False, window_name=title,
                                      minimum_update_interval=33)

        @self.capture.event
        def on_frame_arrived(frame, capture_control):
            image = frame.frame_buffer[:, :, :3].copy()
            with self.lock:
                self.frame = image
                self.received = time.monotonic()

        @self.capture.event
        def on_closed():
            self.closed = True

        self.control = self.capture.start_free_threaded()

    def grab(self):
        with self.lock:
            if self.closed or self.frame is None or time.monotonic()-self.received > .35:
                return None, 0.
            return self.frame.copy(), self.received

    def close(self):
        self.control.stop()


class DirectGamepad:
    def __init__(self):
        try:
            import vgamepad
            self.pad = vgamepad.VX360Gamepad()
        except Exception as exc:
            raise RuntimeError('Virtual controller unavailable. Install the signed ViGEmBus driver first.') from exc
        self.pad.reset()
        self.pad.update()
        self.last_report = None
        self.last_sent = 0.

    def set(self, command, handbrake=False, buttons=(), ui_stick=None):
        import vgamepad as vg
        steer, throttle, brake = command
        report = (round(float(steer)*32767), round(float(throttle)*255),
                  round(float(brake)*255), bool(handbrake), tuple(sorted(buttons)),
                  tuple(ui_stick) if ui_stick is not None else None)
        if report == self.last_report and time.monotonic()-self.last_sent < .05:
            return
        stick = ui_stick if ui_stick is not None else (steer,0.)
        self.pad.left_joystick_float(x_value_float=float(stick[0]), y_value_float=float(stick[1]))
        self.pad.right_trigger_float(value_float=float(throttle))
        self.pad.left_trigger_float(value_float=float(brake))
        if handbrake:
            self.pad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        else:
            self.pad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        for name in ('START','BACK','B','X','Y','LEFT_THUMB','RIGHT_THUMB','LEFT_SHOULDER','RIGHT_SHOULDER',
                     'DPAD_UP','DPAD_DOWN','DPAD_LEFT','DPAD_RIGHT'):
            button = getattr(vg.XUSB_BUTTON,'XUSB_GAMEPAD_'+name)
            if name in buttons:
                self.pad.press_button(button=button)
            else:
                self.pad.release_button(button=button)
        if 'A' in buttons:
            self.pad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.pad.update()
        self.last_report = report
        self.last_sent = time.monotonic()

    def close(self):
        self.pad.reset()
        self.pad.update()


class AnalogGamepad:
    """Client of the persistent controller host; closing AI never removes the pad."""
    def __init__(self):
        import json, socket, subprocess, sys
        from pathlib import Path
        self.json = json
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(.2)
        self.address = ('127.0.0.1', 20067)
        self.tick = False
        launched = False
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            self.sock.sendto(b'{"hello":true}', self.address)
            try:
                data, source = self.sock.recvfrom(1024)
                if source == self.address and json.loads(data).get('ready'):
                    self.last_ack = time.monotonic()
                    self.last_probe = self.last_ack
                    self.sock.setblocking(False)
                    return
            except (socket.timeout, ValueError, OSError):
                if not launched:
                    root = Path(__file__).resolve().parents[1]
                    self.host = subprocess.Popen([sys.executable, '-m', 'mono_drive.forza_pad_server'],
                        cwd=root, creationflags=subprocess.CREATE_NO_WINDOW,
                        stdout=(root/'out'/'controller-host.log').open('a'),
                        stderr=(root/'out'/'controller-host.err').open('a'))
                    launched = True
        raise RuntimeError('Persistent virtual controller did not become ready')

    def set(self, command, handbrake=False, buttons=()):
        now = time.monotonic()
        if now-self.last_probe >= .5:
            self.sock.sendto(b'{"hello":true}', self.address)
            self.last_probe = now
        for _ in range(8):
            try:
                payload, source = self.sock.recvfrom(1024)
            except (BlockingIOError,ConnectionResetError):
                break
            if source == self.address and self.json.loads(payload).get('ready'):
                self.last_ack = now
        if now-self.last_ack > 3.:
            raise ConnectionError('Virtual controller host stopped responding; restart required')
        axes = [float(x) for x in command]
        if any(abs(x) > .001 for x in axes):
            self.tick = not self.tick
            axes[0] = max(-1., min(1., axes[0]+(1. if self.tick else -1.)/32767))
        data = dict(command=axes, handbrake=bool(handbrake),buttons=list(buttons))
        self.sock.sendto(self.json.dumps(data).encode(), self.address)

    def close(self):
        try:
            self.sock.sendto(b'{"command":[0,0,0],"handbrake":false}',self.address)
        finally:
            self.sock.close()
