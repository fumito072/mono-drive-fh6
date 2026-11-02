"""One ground projection shared by road, obstacles and vehicle tracking."""
import math
import numpy as np

_parameters={'horizon':.435,'scale':.575,'version':2}


def set_projection(value=None):
    global _parameters
    value=value or {'horizon':.435,'scale':.575}
    horizon,scale=float(value['horizon']),float(value['scale'])
    if not math.isfinite(horizon+scale) or not .32<=horizon<=.52 or not .35<=scale<=1.5:
        raise ValueError('Invalid ground projection')
    _parameters=dict(horizon=horizon,scale=scale,version=2)


def parameters():
    return dict(_parameters)


def project(x,z):
    return .5+(x+.35)/(2*z),_parameters['horizon']+_parameters['scale']/z


def ground_point(u,v):
    if v<=_parameters['horizon']+.009:
        return None
    z=_parameters['scale']/(v-_parameters['horizon'])
    return np.array([(u-.5)*2*z-.35,z])


def fit_corridor(corridor):
    if corridor is None or len(corridor)<6:
        return None
    measured=np.asarray(corridor,dtype=float)
    inverse_distance=(measured[:,2]-measured[:,1])/1.75
    if np.ptp(inverse_distance)<.035:
        return None
    scale,horizon=np.polyfit(inverse_distance,measured[:,0],1)
    error=float(np.sqrt(np.mean((horizon+scale*inverse_distance-measured[:,0])**2)))
    if not (.36<horizon<.48 and .45<scale<1.25 and error<.004):
        return None
    return dict(horizon=float(horizon),scale=float(scale),error=error)


class GroundCalibration:
    def __init__(self,initial=None):
        initial=initial or {'horizon':.42846,'scale':.74574}
        self.horizon=float(initial['horizon'])
        self.scale=float(initial['scale'])
        self.samples=0

    def update(self,corridor):
        estimate=fit_corridor(corridor)
        if estimate:
            # Slow changes do not turn calibration noise into apparent traffic
            # motion. Bad paint or an unseen road keeps the last estimate.
            self.horizon+=float(np.clip(estimate['horizon']-self.horizon,-.001,.001))
            self.scale+=float(np.clip(estimate['scale']-self.scale,-.003,.003))
            self.samples+=1
        value=dict(horizon=self.horizon,scale=self.scale,version=2)
        set_projection(value)
        return dict(**value,calibration_samples=self.samples)
