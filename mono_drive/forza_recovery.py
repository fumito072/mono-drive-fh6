"""Short, distance-bounded backing and road re-entry in free roam."""
from __future__ import annotations
import math


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


class Recovery:
    def __init__(self):
        self.phase = ''
        self.origin = None
        self.started = 0.
        self.attempt = 0
        self.stall_since = None
        self.direction = 0.
        self.backed_m = 0.
        self.back_limit = .7
        self.reverse_only = False
        self.reverse_steer = 0.
        self.last_moving_at = None
        self.manual = False

    def begin(self, value, direction, now, retry=False, distance_m=None, reverse_only=False, reverse_steer=0., manual=False):
        self.attempt = self.attempt+1 if retry else 1
        self.back_limit = float(distance_m) if distance_m is not None else min(8.,4.+2.*(self.attempt-1))
        self.origin = list(value['position'])
        self.started = now
        self.direction = direction
        self.phase = 'reverse'
        self.backed_m = 0.
        self.reverse_only = reverse_only
        self.reverse_steer = reverse_steer
        self.manual = manual
        self.stall_since = None

    def surface_blocks(self,mode,requested=False):
        return mode in ('offroad_surface','sidewalk_boundary','known_failed_approach') and not (
            requested or (self.phase and self.manual))

    def observe_stall(self, value, requested_gas, now, guidance_ok=True, traffic_blocked=False):
        if value['speed'] > 2.:
            self.last_moving_at = now
        # A visibility stop is not proof of a collision. Reverse only after
        # actual forward effort failed; the session still rescues a long stop.
        stalled = (value['speed'] < .7 and not traffic_blocked and
                   value['accel']>60 and requested_gas>.20)
        if stalled:
            self.stall_since = self.stall_since or now
            return now-self.stall_since >= 2.
        self.stall_since = None
        return False

    def update(self, value, guidance, direction, now):
        if not self.phase:
            return None
        moved = distance(value['position'], self.origin)
        elapsed = now-self.started
        if self.phase == 'reverse':
            self.backed_m = moved
            reverse_ms = max(0., -value['forward_kmh']/3.6)
            # Begin braking before the requested distance, including pedal ramp delay.
            stopping_m = reverse_ms*.35 + reverse_ms**2/(2*2.)
            # Gear engagement/ramped pedals used most of the old 8s budget.
            # Keep the manoeuvre distance bounded while allowing time to move.
            if moved+stopping_m >= self.back_limit or elapsed >= max(20.,self.back_limit*4.+6.):
                self.phase, self.started = 'settle', now
                return self.update(value, guidance, direction, now)
            # Brake pedal engages reverse only after the car has stopped.
            if value['forward_kmh'] > .8:
                return (0., 0., .25), False, True
            if value.get('gear',0) != 0:
                return (self.reverse_steer,0.,.80),False,True
            # Continuous velocity feedback instead of gas/coast/park pulses.
            # A stronger initial input selects reverse; once moving, reduce it
            # progressively toward the measured cruising pedal.
            pedal = max(0.,min(.78,.58+.30*(.85-reverse_ms)))
            return (self.reverse_steer, 0., pedal), False, True
        if self.phase == 'settle':
            self.backed_m = moved
            # RT was observed selecting forward and driving back into the tree.
            # Park briefly to settle without sending a forward pedal command.
            if value['forward_kmh'] < -3.:
                self.started = now
                return (0.,0.,0.),True,True
            if value['forward_kmh'] > 3.:
                self.started = now
                return (0., 0., .3), False, True
            if elapsed < .6:
                return (0., 0., 0.), True, True
            if self.reverse_only:
                self.phase = 'complete_reverse'
                return (0.,0.,0.),True,False
            self.phase, self.started = 'assess',now
            return (0.,0.,0.),True,False
        if self.phase == 'assess':
            # A grey pond or snow patch is not evidence for driving forward.
            supported = guidance is not None and guidance.source not in ('minimap_supported','road_reentry')
            if supported and guidance.confidence >= .7:
                self.phase,self.started,self.origin = 'forward',now,list(value['position'])
                self.direction = guidance.steer
                return (self.direction,.25,0.),False,False
            if elapsed >= 1.:
                if self.attempt < 3:
                    self.begin(value,direction,now,retry=True)
                else:
                    self.phase = 'failed'
            return (0.,0.,0.),True,False
        if self.phase == 'forward':
            if moved > 1.2 and guidance is not None:
                self.phase = ''
                return None
            if elapsed >= 6. or (elapsed > 3. and moved < .25):
                if self.attempt < 3:
                    self.begin(value, direction, now, retry=True)
                    return (0., 0., .3), False, True
                self.phase = 'failed'
            else:
                steer = guidance.steer if guidance is not None else self.direction
                return (steer, .25 if value['forward_kmh'] < 5 else 0., 0.), False, False
        return (0., 0., 0.), True, False
