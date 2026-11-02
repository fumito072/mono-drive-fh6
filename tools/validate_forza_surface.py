"""Replay a saved game frame through the actual road/sidewalk model."""
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from mono_drive.forza_vision import LocalVision
from mono_drive.forza_surface import RoadSurfaceGuard
from mono_drive.forza_scene import ScenePlan


def main():
    image=Path(sys.argv[1])
    output=Path(sys.argv[2])
    frame=cv2.imread(str(image))
    if frame is None:
        raise ValueError('Cannot read the supplied game image')
    model=LocalVision()
    ego=dict(position=[0.,0.,0.],yaw=0.,speed=0.,steer=0.)
    road,cars=model.infer(frame,ego,time.monotonic())
    scene=dict(road=road,raw_road=model.raw_road,sidewalk=model.sidewalk,cars=cars,ego=ego)
    z=np.array([4.,6.,8.,10.,16.,22.])
    plan=ScenePlan(np.column_stack((z,np.zeros_like(z))),0.,25.,.6,1,0,'lane_inferred',0.,inferred=True)
    guard=RoadSurfaceGuard()
    guard.apply(plan,scene,ego,1.)
    result=guard.apply(plan,scene,ego,1.5)
    report=dict(image=str(image),reason=result.reason,limit=result.limit,**guard.status)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
