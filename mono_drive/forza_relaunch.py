"""Bounded release/reapply attempts before the existing reverse recovery."""


class RelaunchAssist:
    def __init__(self):
        self.phase = 'idle'
        self.attempts = 0
        self.elapsed = self.stopped_for = 0.

    def reset(self):
        self.__init__()

    @property
    def blocks_reverse(self):
        return ((self.stopped_for > 0 and self.attempts < 3) or
                self.phase in ('release', 'launch', 'cooldown'))

    def step(self, command, speed, target, dt, *, allowed=True):
        dt = max(0., min(.5, dt))
        if not allowed or target < 5. or command[2] > .01:
            self.reset()
            return command
        if speed > .7:
            self.reset()
            return command
        self.stopped_for += dt
        self.elapsed += dt
        if self.phase == 'idle':
            if self.attempts >= 3 or self.stopped_for < 2.:
                return command
            self.attempts += 1
            self.phase, self.elapsed = 'release', 0.
        if self.phase == 'release':
            # Hold zero long enough for the analog ramp to release the trigger.
            if self.elapsed < 1.:
                return command[0], 0., 0.
            self.phase, self.elapsed = 'launch', 0.
        if self.phase == 'launch':
            if self.elapsed < 3.:
                pedal = min(.45, max(command[1], .22 + .10*self.elapsed))
                return command[0], pedal, 0.
            self.phase, self.elapsed = 'cooldown', 0.
        if self.phase == 'cooldown':
            if self.elapsed < 1.:
                return command
            self.phase, self.elapsed = 'idle', 0.
        return command

    def status(self):
        return dict(phase=self.phase, attempts=self.attempts, stopped_s=round(self.stopped_for, 2))
