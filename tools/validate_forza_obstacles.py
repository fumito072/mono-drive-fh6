"""Replay saved game frames through the actual local detector and safety filter."""
import json
from pathlib import Path
import time
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mono_drive.forza_vision import LocalVision
from mono_drive.forza_static import static_risk

ROOT=Path(__file__).resolve().parents[1]
vision=LocalVision()
files=['out/obstacle-diagnostic-current.jpg','out/median-current.jpg',
       'out/fh6_20261027_101656/frames/000130.jpg',
       'out/fh6_20261027_101656/frames/000161.jpg']
results=[]
for index,file in enumerate(files):
    frame=cv2.imread(str(ROOT/file))
    if frame is None:
        continue
    t=time.monotonic()
    road,cars=vision.infer(frame,dict(position=[0,0,0],speed=0.,yaw=0.),t)
    old=static_risk(dict(road=road,obstacles=vision.raw_obstacles),30.,0.)
    new=static_risk(dict(road=road,obstacles=vision.obstacles),30.,0.)
    results.append(dict(file=file,cars=len(cars),raw_occupied=int((vision.raw_obstacles>185).sum()),
                        grounded_occupied=int((vision.obstacles>185).sum()),
                        previous_clearance=old.clearance_m,grounded_clearance=new.clearance_m,
                        previous_emergency=old.emergency,grounded_emergency=new.emergency))
    overlay=cv2.resize(frame,(384,216))
    mask=vision.obstacles>185
    overlay[mask]=(overlay[mask]*.5+np.array([0,0,255])*.5).astype(np.uint8)
    cv2.imwrite(str(ROOT/f'out/ground-contact-check-{index}.jpg'),overlay)
(ROOT/'out/obstacle-validation.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results))
