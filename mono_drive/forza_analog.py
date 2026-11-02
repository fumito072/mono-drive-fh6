"""Continuous Xbox-axis output with bounded rates and smooth reversals."""
import math


class AnalogSmoother:
    def __init__(self):
        self.value=(0.,0.,0.)
        self.velocity=[0.,0.,0.]

    def reset(self,command=(0.,0.,0.)):
        self.value=tuple(command)
        self.velocity=[0.,0.,0.]
        return self.value

    def reconcile(self,command):
        for i,(before,after) in enumerate(zip(self.value,command)):
            if abs(before-after)>1e-9:
                self.velocity[i]=0.
        self.value=tuple(command)
        return self.value

    def step(self,target,dt,speed_kmh=0.,emergency=False,reverse=False):
        if len(target)!=3 or not all(math.isfinite(float(v)) for v in (*target,dt,speed_kmh)):
            return self.reset()
        dt=max(0.,min(.05,float(dt)))
        if dt==0:
            return self.value
        target=[max(-1. if i==0 else 0.,min(1.,float(v))) for i,v in enumerate(target)]
        current=list(self.value)
        speed=max(0.,min(80.,float(speed_kmh)))
        steer_rate=max(1.0,1.7-.014*speed)
        # Longitudinal acceleration is already bounded by SpeedController.
        # A second two-second pedal ramp kept clear-road acceleration behind
        # alternating perception updates. Track it with a smooth faster rise.
        limits=((steer_rate,steer_rate,5.,5.),(.50,.75,2.5,4.),(.50,6.,1.8,40.))
        if emergency:
            # A genuinely imminent collision must not wait for comfort ramping.
            current[1]=target[1]=self.velocity[1]=0.
            limits=(limits[0],limits[1],(2.5,6.,20.,40.))
        elif reverse:
            target[1]=0.
            limits=(limits[0],limits[1],(1.2,1.5,6.,6.))
        elif target[2]>.01:
            target[1]=0.
            # Release the accelerator before beginning ordinary braking.
            if current[1]>.015:
                target[2]=0.
        if target[1]>.01 and current[2]>.01:
            target[1]=0.
        for i,(rise,fall,rise_acceleration,fall_acceleration) in enumerate(limits):
            error=target[i]-current[i]
            acceleration=rise_acceleration if error>=0 else fall_acceleration
            if abs(error)<1e-9 and abs(self.velocity[i])<acceleration*dt:
                self.velocity[i]=0.
                continue
            limit=rise if error>=0 else fall
            stopping_rate=math.sqrt(2*acceleration*abs(error)+(.5*acceleration*dt)**2)-.5*acceleration*dt
            desired=math.copysign(min(limit,stopping_rate),error) if error else 0.
            rate=self.velocity[i]+max(-acceleration*dt,min(acceleration*dt,desired-self.velocity[i]))
            # A changed target may invalidate the old velocity. Never continue
            # increasing a released pedal or steering away from the new target.
            if rate*error<0:
                rate=0.
            updated=current[i]+rate*dt
            if error*(target[i]-updated)<0:
                updated,rate=target[i],0.
            current[i]=max(-1. if i==0 else 0.,min(1.,updated))
            self.velocity[i]=rate
        # Independent of target transitions, never drive both pedals together.
        if current[2]>.01:
            current[1]=self.velocity[1]=0.
        self.value=tuple(current)
        return self.value
