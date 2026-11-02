"""Bounded sampling for underrepresented bends; validation stays untouched."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np


def enabled(root):
    try:
        root=Path(root)
        plan=json.loads((root/'out/training-curriculum.json').read_text(encoding='utf-8'))
        data=json.loads((root/'out/human-dataset.json').read_text(encoding='utf-8'))
        return (plan.get('balanced_maneuvers') is True and plan.get('revision')==data.get('revision')
                and int(data.get('frames',0))>=10000)
    except (OSError,ValueError,TypeError):
        return False


def sampling_weights(targets, valid, arc_s, curve_weight=1., balanced=False):
    """Use only training targets, capped weights and no duplication into holdout."""
    targets=np.asarray(targets);valid=np.asarray(valid,dtype=bool)
    if len(targets)==0:raise ValueError('No training targets')
    curved=np.max(np.where(valid,np.abs(targets[...,1]),0.),axis=1)>3.
    weight=np.where(curved,curve_weight,1.).astype(float)
    groups=np.zeros(len(targets),dtype=np.int64)
    if balanced:
        # Prefer the ~20 m target, with an earlier valid point as fallback.
        candidate=(np.asarray(arc_s)<=20)[None,:]&valid
        index=np.max(np.where(candidate,np.arange(valid.shape[1])[None,:],0),axis=1)
        f=targets[np.arange(len(targets)),index,0]
        r=targets[np.arange(len(targets)),index,1]
        lateral_20=r*20/np.maximum(4.,np.abs(f))
        gentle=(np.abs(lateral_20)>=1.)&(np.abs(lateral_20)<4.)
        sharp=np.abs(lateral_20)>=4.
        groups[gentle]=np.where(lateral_20[gentle]<0,1,2)
        groups[sharp]=np.where(lateral_20[sharp]<0,3,4)
        counts=np.bincount(groups,minlength=5)
        present=counts[counts>0]
        scale=np.ones(5)
        scale[counts>0]=np.clip(np.sqrt(float(np.median(present))/counts[counts>0]),.5,3.)
        weight*=scale[groups]
    weight=np.clip(weight,.5,4.)
    return weight/weight.sum(),np.bincount(groups,minlength=5).tolist()


class BestValidationCheckpoint:
    def __init__(self):
        self.score=math.inf
        self.epoch=None
        self.weights=None

    def consider(self,model,score,epoch):
        if not math.isfinite(float(score)) or score>=self.score:return False
        self.score=float(score);self.epoch=int(epoch)
        self.weights={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
        return True

    def restore(self,model):
        if self.weights is None:return False
        model.load_state_dict(self.weights);return True
