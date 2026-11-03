"""Supervised visual adaptation targets; no future image is an input."""
import math
import numpy as np
from .temporal_steering import targets


def visual_targets(steer,yaw_degrees,index,stride=1):
    later=index+3*stride
    delta=math.radians(float(yaw_degrees[later])-float(yaw_degrees[index]))
    delta=math.atan2(math.sin(delta),math.cos(delta))
    return np.asarray([targets(steer,index,stride)[-1],np.clip(delta/(math.pi/4),-1.,1.)],np.float32)


def visual_scores(prediction,truth):
    prediction=np.asarray(prediction,float);truth=np.asarray(truth,float)
    active=np.abs(truth[:,0])>.04
    return {'samples':len(truth),'steer_mae':float(np.mean(np.abs(prediction[:,0]-truth[:,0]))),
            'turn_mae':float(np.mean(np.abs(prediction[active,0]-truth[active,0]))) if active.any() else None,
            'heading_mae':float(np.mean(np.abs(prediction[:,1]-truth[:,1]))),
            'steer_bias':float(np.mean(prediction[:,0]-truth[:,0]))}


def accept_visual(adapted,frozen):
    fields=('steer_mae','turn_mae','heading_mae','steer_bias')
    if any(not isinstance(v.get(k),(int,float)) or not math.isfinite(v[k]) for v in (adapted,frozen) for k in fields):return False
    return (adapted['samples']>=500 and adapted['steer_mae']<=frozen['steer_mae']*.98 and
            adapted['turn_mae']<=frozen['turn_mae'] and adapted['heading_mae']<=frozen['heading_mae']*1.05 and
            abs(adapted['steer_bias'])<=.05)
