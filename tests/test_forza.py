import math
import socket
import struct
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mono_drive.forza import Actuator, Telemetry, controls, crop_letterbox, parse_packet, forward_only, parse_control_request, steering_input, stop_command, LinearRamp, SpeedController
from unittest.mock import patch


class ForzaTests(unittest.TestCase):
    def test_forward_request_keeps_rt_when_automatic_is_stopped_in_reverse(self):
        class Window:
            def target(self,background=False): return {}
            def down(self,key): raise AssertionError('No keyboard polling')
        class Pad:
            reports=[]
            def set(self,command,handbrake=False,buttons=()):
                self.reports.append((command,buttons))
            def close(self): pass
        telemetry=SimpleNamespace(fresh=lambda:True,
            last=dict(race=True,gear=0,forward_kmh=0.,speed=0.))
        with patch('mono_drive.forza_background.AnalogGamepad',Pad):
            actuator=Actuator(Window(),telemetry,True,background=True)
        actuator.requested=actuator.armed=True
        actuator.publish((0.,.25,0.),False,False,time.monotonic())
        actuator.start()
        try:
            time.sleep(.06)
            self.assertTrue(any(command[1]>.0 for command,_ in actuator.pad.reports))
            self.assertTrue(all('B' not in buttons for _,buttons in actuator.pad.reports))
        finally:
            actuator.stop_event.set();actuator.join()

    def test_initial_rearm_survives_brief_gaps_between_fresh_captures(self):
        class Window:
            def target(self,background=False): return {}
            def down(self,key): raise AssertionError('No keyboard polling')
        class Pad:
            def set(self,*args,**kwargs): pass
            def close(self): pass
        telemetry=SimpleNamespace(fresh=lambda:True,
            last=dict(race=True,gear=1,forward_kmh=0.,speed=0.))
        with patch('mono_drive.forza_background.AnalogGamepad',Pad):
            actuator=Actuator(Window(),telemetry,True,background=True)
        actuator.set_enabled(True)
        actuator.publish((0.,.2,0.),False,False,time.monotonic())
        actuator.start()
        try:
            time.sleep(.12)
            self.assertFalse(actuator.armed)
            actuator.updated=time.monotonic()-.7
            time.sleep(.16)
            self.assertFalse(actuator.armed)  # Never drive on a stale frame.
            actuator.publish((0.,.2,0.),False,False,time.monotonic())
            time.sleep(.05)
            self.assertTrue(actuator.armed)
            actuator.updated=time.monotonic()-3.
            time.sleep(.03)
            self.assertFalse(actuator.armed)
            self.assertEqual(actuator.rearm_frames,0)
        finally:
            actuator.stop_event.set();actuator.join()

    def test_background_worker_survives_missing_initial_telemetry(self):
        class Window:
            def target(self,background=False):return {}
            def down(self,key):raise AssertionError('No keyboard polling')
        class Pad:
            def set(self,*args,**kwargs):pass
            def close(self):pass
        with patch('mono_drive.forza_background.AnalogGamepad',Pad):
            actuator=Actuator(Window(),SimpleNamespace(last=None,fresh=lambda:False),True,background=True)
        actuator.requested=True;actuator.start()
        try:
            time.sleep(.04)
            self.assertIsNone(actuator.failure)
            self.assertTrue(actuator.is_alive())
            self.assertFalse(actuator.armed)
        finally:
            actuator.stop_event.set();actuator.join()

    def test_new_fresh_frame_resumes_a_brief_active_input_gap(self):
        class Window:
            def target(self,background=False): return {}
            def down(self,key): raise AssertionError('No keyboard polling')
        class Pad:
            def set(self,*args,**kwargs): pass
            def close(self): pass
        telemetry=SimpleNamespace(fresh=lambda:True,
            last=dict(race=True,gear=1,forward_kmh=10.,speed=10.))
        with patch('mono_drive.forza_background.AnalogGamepad',Pad):
            actuator=Actuator(Window(),telemetry,True,background=True)
        actuator.requested=actuator.armed=True
        actuator.publish((0.,.2,0.),False,False,time.monotonic())
        actuator.start()
        try:
            time.sleep(.04)
            actuator.updated=time.monotonic()-.7
            time.sleep(.03)
            self.assertFalse(actuator.armed)
            actuator.publish((0.,.2,0.),False,False,time.monotonic())
            time.sleep(.05)
            self.assertTrue(actuator.armed)
        finally:
            actuator.stop_event.set()
            actuator.join()

    def test_brief_image_gap_rearms_after_two_new_frames_without_keyboard(self):
        class Window:
            def target(self,background=False): return {}
            def down(self,key): raise AssertionError('No keyboard polling')
        class Pad:
            parked=False
            def set(self,command,handbrake=False,buttons=()): self.parked=handbrake
            def close(self): pass
        telemetry=SimpleNamespace(fresh=lambda:True,
            last=dict(race=True,gear=1,forward_kmh=0.,speed=0.))
        with patch('mono_drive.forza_background.AnalogGamepad',Pad):
            actuator=Actuator(Window(),telemetry,True,background=True)
        actuator.requested=actuator.armed=True
        actuator.updated=time.monotonic()-.8
        actuator.start()
        try:
            time.sleep(.03)
            self.assertFalse(actuator.armed)
            self.assertTrue(actuator.soft_rearm)
            actuator.publish((0.,.2,0.),False,False,time.monotonic())
            time.sleep(.12)
            self.assertFalse(actuator.armed)
            self.assertFalse(actuator.pad.parked)
            actuator.publish((0.,.2,0.),False,False,time.monotonic())
            time.sleep(.18)
            self.assertTrue(actuator.armed)
            actuator.updated=time.monotonic()-3.
            time.sleep(.03)
            self.assertFalse(actuator.soft_rearm)
            actuator.publish((0.,.2,0.),False,False,time.monotonic())
            time.sleep(.1)
            actuator.publish((0.,.2,0.),False,False,time.monotonic())
            time.sleep(.18)
            self.assertFalse(actuator.armed)
        finally:
            actuator.stop_event.set()
            actuator.join()

    def test_smoothed_brake_is_released_at_low_speed_but_explicit_reverse_remains_available(self):
        class Window:
            def target(self,background=False):
                return {}
            def down(self,key):
                raise AssertionError('Background driving must not poll the keyboard')
        class Pad:
            command=None
            handbrake=False
            def set(self,command,handbrake=False,buttons=()):
                self.command,self.handbrake=command,handbrake
            def close(self):
                pass
        telemetry=SimpleNamespace(fresh=lambda:True,
            last=dict(race=True,gear=1,forward_kmh=6.,speed=6.))
        with patch('mono_drive.forza_background.AnalogGamepad',Pad):
            actuator=Actuator(Window(),telemetry,True,background=True)
        actuator.requested=actuator.armed=actuator.smooth=True
        actuator.publish((0.,0.,.8),False,False,time.monotonic(),emergency=True)
        actuator.start()
        try:
            time.sleep(.12)
            self.assertGreater(actuator.pad.command[2],0.)
            telemetry.last.update(forward_kmh=2.,speed=2.)
            actuator.publish((0.,0.,.8),False,False,time.monotonic(),emergency=True)
            time.sleep(.04)
            self.assertEqual(actuator.pad.command[2],0.)
            self.assertTrue(actuator.pad.handbrake)
            telemetry.last.update(gear=0,forward_kmh=0.,speed=0.)
            actuator.publish((0.,0.,.8),False,True,time.monotonic())
            time.sleep(.04)
            self.assertGreater(actuator.pad.command[2],0.)
            self.assertFalse(actuator.pad.handbrake)
        finally:
            actuator.stop_event.set()
            actuator.join()

    def test_reverse_gear_requests_forward_without_keyboard(self):
        class Window:
            def target(self,background=False):
                return {}
            def down(self,key):
                raise AssertionError('Physical keyboard is not part of background control')
        class Pad:
            command = None
            buttons = ()
            def set(self,command,handbrake=False,buttons=()):
                self.command,self.buttons = command,buttons
            def close(self):
                pass
        telem = SimpleNamespace(fresh=lambda:True,last={'race':True,'gear':0,'forward_kmh':0.,'speed':0.})
        with patch('mono_drive.forza_background.AnalogGamepad',Pad):
            a = Actuator(Window(),telem,True,background=True)
        a.requested,a.armed = True,True
        a.command,a.updated = (0.,.3,0.),time.monotonic()
        a.start()
        try:
            time.sleep(.04)
            self.assertEqual(a.pad.command,(0.,.3,0.))
            self.assertEqual(a.pad.buttons,())
            telem.last['gear'] = 1
            time.sleep(.04)
            self.assertEqual(a.pad.command,(0.,.3,0.))
            self.assertEqual(a.pad.buttons,())
            telem.last.update(gear=0,forward_kmh=10.,speed=10.)
            a.publish((0.,0.,.8),False,False,time.monotonic(),emergency=True)
            time.sleep(.04)
            self.assertEqual(a.pad.command,(0.,0.,.8))
        finally:
            a.stop_event.set()
            a.join()

    def test_reverse_recovery_brakes_before_distance_and_waits_for_stop(self):
        from mono_drive.forza_recovery import Recovery
        r = Recovery()
        value = dict(position=[0.,0.,0.],speed=0.,forward_kmh=0.,accel=0)
        r.begin(value, 0., 10., distance_m=2., reverse_only=True)
        value.update(position=[0.,0.,-1.7],speed=4.,forward_kmh=-4.)
        command, handbrake, reverse = r.update(value,None,0.,11.)
        self.assertEqual(r.phase,'settle')
        self.assertEqual(command[1:],(0.,0.))
        self.assertEqual(command[2],0)
        self.assertTrue(handbrake)
        r.update(value,None,0.,15.)
        self.assertEqual(r.phase,'settle')
        value.update(speed=0.,forward_kmh=0.)
        r.update(value,None,0.,15.1)
        r.update(value,None,0.,15.8)
        self.assertEqual(r.phase,'complete_reverse')
        command, handbrake = stop_command(-5.)
        self.assertEqual(command[1],0)
        self.assertEqual(command[2],0)
        self.assertTrue(handbrake)

    def test_recovery_backs_only_short_distance(self):
        from mono_drive.forza_recovery import Recovery
        r = Recovery()
        v = dict(position=[0., 0., 0.], speed=0., forward_kmh=0., accel=0)
        r.begin(v, .4, 10.)
        self.assertTrue(r.update(v, None, .4, 10.1)[2])
        v['position'] = [0., 0., -4.01]
        self.assertTrue(r.update(v, None, .4, 10.5)[1])
        self.assertEqual(r.phase, 'settle')
        command, handbrake, reverse = r.update(v, None, .4, 11.2)
        self.assertEqual(r.phase,'assess')
        self.assertEqual(command,(0.,0.,0.))
        guidance = SimpleNamespace(source='paint_pair',confidence=.8,steer=.4)
        command, handbrake, reverse = r.update(v,guidance,.4,11.3)
        self.assertGreater(command[1], 0)
        self.assertFalse(reverse)
        v['position'] = [0., 0., .6]
        self.assertIsNone(r.update(v, guidance, .4, 12.))
        self.assertEqual(r.phase, '')

    def test_road_surface_rejects_snow_and_black_frame(self):
        import numpy as np
        from mono_drive.forza_lane import road_patch
        self.assertFalse(road_patch(np.full((540, 960, 3), 220, dtype=np.uint8), .5))
        self.assertFalse(road_patch(np.zeros((540, 960, 3), dtype=np.uint8), .5))
        self.assertTrue(road_patch(np.full((540, 960, 3), 80, dtype=np.uint8), .5))

    def test_pedals_ramp_and_avoid_brake_pulses(self):
        ramp = LinearRamp()
        for _ in range(5):
            command = ramp.step((1., 1., 0.), .1)
        self.assertAlmostEqual(command[1], .14)
        self.assertAlmostEqual(command[0], .325)
        c = SpeedController()
        c.target = 8
        d = SimpleNamespace(confidence=1., mode='left_lane', reverse=False, target_kmh=8., steer=.1)
        # Brief speed error uses coasting; it does not generate a brake pulse.
        self.assertEqual(c.step(d, 13., 12., .5, .2)[2], 0)
        for _ in range(8):
            command = c.step(d, 13., 12., .5, .2)
        self.assertGreater(command[2], 0)
        self.assertEqual(command[1], 0)
        self.assertGreater(c.step(d, 2., 12., .5, .2)[1], .3)

    def test_lane_fusion_gap_prediction_and_veto(self):
        import numpy as np
        from mono_drive.forza_lane import LaneEstimate, LaneFusion
        frame = np.full((540, 960, 3), 80, dtype=np.uint8)
        lane = LaneEstimate(.8, .40, .46, .12, -.28, (.46, .44), (.58, .65))
        fusion = LaneFusion()
        model = SimpleNamespace(confidence=.7, steer=.1, reverse=False, mode='path',
                                blocked_m=None, path=None)
        telem = dict(speed=4, yaw=0.)
        with patch('mono_drive.forza_lane.estimate_left_lane', return_value=lane):
            first = fusion.update(frame, model, telem, 10.)
        self.assertEqual(first.source, 'left_lane')
        with patch('mono_drive.forza_lane.estimate_left_lane', return_value=None):
            telem['yaw'] = -.1
            short = fusion.update(frame, model, telem, 10.5)
            self.assertEqual(short.source, 'lane_odometry')
            self.assertGreater(short.steer, first.steer)
            longer = fusion.update(frame, model, telem, 11.)
            self.assertEqual(longer.source, 'lane_model')
            self.assertLessEqual(longer.max_kmh, 25)
            model.blocked_m = 2
            self.assertIsNone(fusion.update(frame, model, telem, 11.1))
            model.blocked_m = None
            self.assertIsNone(fusion.update(frame, model, telem, 15.))

    def test_lane_preview_looks_further_at_speed(self):
        from mono_drive.forza_lane import LaneFusion
        model = SimpleNamespace(confidence=.7, path=None)
        self.assertGreater(LaneFusion.preview(model, 20)[0], LaneFusion.preview(model, 0)[0])

    def test_smooth_analog_and_stop(self):
        self.assertEqual(steering_input(0, .28, .98), 0)
        self.assertAlmostEqual(steering_input(.1, .28, .98), .35)
        self.assertAlmostEqual(steering_input(-.1, .28, .98), -.35)
        command, handbrake = stop_command(8)
        self.assertGreater(command[2], 0)
        self.assertFalse(handbrake)
        self.assertEqual(stop_command(0), ((0, 0, 0), True))

    def test_left_lane_uses_width(self):
        import numpy as np
        from mono_drive.forza_lane import estimate_left_lane
        center = dict(near=310, far=400, left=1., right=1., count=3, weight=120,
                      ys=[245, 270, 291, 312], continuity=.45)
        edge = dict(near=710, far=520, left=1., right=.8, count=2, weight=180,
                    ys=[250, 312], continuity=1.)
        with patch('mono_drive.forza_lane.paint_groups', return_value=[center, edge]):
            lane = estimate_left_lane(np.full((540, 960, 3), 80, dtype=np.uint8))
        self.assertIsNotNone(lane)
        self.assertAlmostEqual(lane.target_x, (400-60)/960)
        self.assertLess(lane.steer, 0)
        edge['near'], edge['far'] = 910, 600
        with patch('mono_drive.forza_lane.paint_groups', return_value=[center, edge]):
            wider = estimate_left_lane(np.full((540, 960, 3), 80, dtype=np.uint8))
        self.assertLess(wider.target_x, lane.target_x)
        self.assertIsNone(estimate_left_lane(np.zeros((540, 960, 3), dtype=np.uint8)))

    def test_bounded_runtime_maneuver(self):
        self.assertEqual(parse_control_request({'action': 'test', 'steer': -.6, 'seconds': 3}),
                         ('test', 12., 3., -.6, .5))
        for field, value in [('seconds', 7), ('steer', 1), ('throttle', .7),
                             ('max_kmh', float('nan')), ('steer', float('inf'))]:
            with self.assertRaises(ValueError):
                parse_control_request({'action': 'test', field: value})

    def test_letterbox(self):
        import numpy as np
        f = np.zeros((1080, 1216, 3), dtype=np.uint8)
        f[198:882] = 80
        self.assertEqual(crop_letterbox(f).shape, (684, 1216, 3))
        cockpit = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cockpit[:600] = 80
        self.assertEqual(crop_letterbox(cockpit).shape, cockpit.shape)
        with self.assertRaises(ValueError):
            crop_letterbox(f*0)

    def test_deadman_focus_and_stop(self):
        class FakeWin:
            focus = True
            pressed = set()
            def target(self):
                return {} if self.focus else None
            def down(self, vk):
                return vk in self.pressed
        class FakeKeys:
            def __init__(self, win):
                self.held = set()
            def set(self, desired):
                self.held = desired
        win = FakeWin()
        t = SimpleNamespace(fresh=lambda: True, last={'race': True})
        with patch('mono_drive.forza.Keyboard', FakeKeys):
            a = Actuator(win, t, True)
        a.updated = time.monotonic()
        a.command = (0, 1, 0)
        win.pressed = {0x77}
        a.start()
        try:
            time.sleep(.04)
            self.assertTrue(a.armed)
            self.assertEqual(a.keys.held, {0x11})
            a.updated = time.monotonic()-1
            time.sleep(.04)
            self.assertFalse(a.armed)
            self.assertEqual(a.keys.held, set())
            a.armed = True
            a.updated = time.monotonic()
            win.focus = False
            time.sleep(.04)
            self.assertFalse(a.armed)
            self.assertEqual(a.keys.held, set())
        finally:
            a.stop_event.set()
            a.join()
        self.assertEqual(a.keys.held, set())

    def test_wire_format(self):
        p = bytearray(324)
        struct.pack_into('<iI', p, 0, 1, 42)
        struct.pack_into('<3f', p, 32, 3, 0, 4)
        self.assertEqual(parse_packet(p)['speed'], 18)
        self.assertEqual(parse_packet(p)['forward_kmh'], 14.4)
        p[315], p[316], p[319] = 120, 30, 2
        parsed = parse_packet(p)
        self.assertEqual((parsed['accel'], parsed['brake'], parsed['gear']), (120, 30, 2))
        struct.pack_into('<f', p, 40, -4)
        self.assertLess(parse_packet(p)['forward_kmh'], 0)
        self.assertEqual(parse_packet(p)['speed'], 18)
        for size in (0, 232, 323, 331):
            with self.assertRaises(ValueError):
                parse_packet(bytes(size))
        struct.pack_into('<f', p, 32, float('nan'))
        with self.assertRaises(ValueError):
            parse_packet(p)

    def test_forward_only_interlock(self):
        self.assertEqual(forward_only((.1, 0, 1), {'forward_kmh': 2}), (.1, 0, 0))
        self.assertEqual(forward_only((.1, 0, 1), {'forward_kmh': -5}), (0, .55, 0))
        self.assertEqual(forward_only((.1, 0, 1), {'forward_kmh': 10}), (.1, 0, 1))

    def test_background_does_not_use_keyboard(self):
        class FakeWin:
            def target(self, background=False):
                return {} if background else None
            def down(self, vk):
                raise AssertionError('Background mode must not inspect physical keys')
        class FakePad:
            closed = False
            command = None
            def set(self, command, handbrake=False):
                self.command = command
            def close(self):
                self.closed = True
        t = SimpleNamespace(fresh=lambda: True, last={'race': True, 'forward_kmh': 5})
        with patch('mono_drive.forza_background.AnalogGamepad', FakePad):
            a = Actuator(FakeWin(), t, True, background=True)
        a.armed, a.updated, a.command = True, time.monotonic(), (0, .3, 0)
        a.start()
        try:
            time.sleep(.04)
            self.assertTrue(a.armed)
            self.assertEqual(a.pad.command, (0, .3, 0))
            a.updated -= 1
            time.sleep(.04)
            self.assertFalse(a.armed)
            self.assertEqual(a.pad.command, (0, 0, 0))
        finally:
            a.stop_event.set()
            a.join()
        self.assertTrue(a.pad.closed)

    def test_transient_recovery_and_explicit_stop(self):
        class FakeWin:
            pressed = set()
            def target(self, background=False):
                return {}
            def down(self, vk):
                return vk in self.pressed
        class FakePad:
            def set(self, command, handbrake=False):
                pass
            def close(self):
                pass
        win = FakeWin()
        t = SimpleNamespace(fresh=lambda: True, last={'race': True, 'forward_kmh': 0})
        with patch('mono_drive.forza_background.AnalogGamepad', FakePad):
            a = Actuator(win, t, True, background=True)
        a.requested = True
        a.updated = time.monotonic()
        a.start()
        try:
            time.sleep(.03)
            a.stable_since = time.monotonic()-2
            time.sleep(.03)
            self.assertTrue(a.armed)
            win.pressed = {0x78}
            time.sleep(.03)
            self.assertTrue(a.armed)
            win.pressed = {0x77,0x78,0x79,0x1B}
            time.sleep(.03)
            self.assertTrue(a.armed)
            self.assertFalse(a.stop_event.is_set())
            a.set_enabled(False)
            self.assertFalse(a.armed)
            self.assertFalse(a.requested)
            win.pressed = set()
            a.stable_since = time.monotonic()-2
            time.sleep(.03)
            self.assertFalse(a.armed)
            t.last['race_position'] = 1
            a.requested, a.armed = True, True
            time.sleep(.03)
            self.assertFalse(a.armed)
        finally:
            a.stop_event.set()
            a.join()

    def test_stop_and_speed_limit(self):
        d = SimpleNamespace(confidence=1, steer=.9, target_kmh=120, mode='path', reverse=False)
        steer, gas, brake = controls(d, 45, 30, .5)
        self.assertEqual(gas, 0)
        self.assertGreater(brake, 0)
        self.assertLessEqual(steer, .65)
        for field, value in [('confidence', .1), ('mode', 'lost'), ('reverse', True), ('steer', math.nan)]:
            dd = SimpleNamespace(**vars(d))
            setattr(dd, field, value)
            self.assertEqual(controls(dd, 10, 30, .5), (0, 0, 1))

    def test_udp_stale_and_replayed(self):
        t = Telemetry(0)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            p = bytearray(324)
            struct.pack_into('<iI', p, 0, 1, 42)
            sender.sendto(p, t.sock.getsockname())
            time.sleep(.01)
            t.poll()
            self.assertTrue(t.fresh())
            t.received -= 1
            sender.sendto(p, t.sock.getsockname())
            time.sleep(.01)
            t.poll()
            self.assertFalse(t.fresh())
        finally:
            t.sock.close()
            sender.close()


if __name__ == '__main__':
    unittest.main()
