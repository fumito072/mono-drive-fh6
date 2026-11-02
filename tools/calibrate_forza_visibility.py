"""Estimate fixed cockpit occlusion from diverse recorded driving frames."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
folders = ['fh6_20261026_230349','fh6_20261026_235452','fh6_20261027_000903','fh6_20261027_002255']
frames,used = [],[]
for name in folders:
    files = sorted((ROOT/'out'/name/'frames').glob('*.jpg'))
    for index in np.linspace(0,max(0,len(files)-1),min(60,len(files))).astype(int):
        file = files[index]
        image = cv2.imread(str(file))
        if image is None or image.shape[:2] != (1080,1920):
            continue
        frames.append(cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(384,216)))
        used.append(str(file.relative_to(ROOT)))
if len(frames) < 80:
    raise RuntimeError('Need diverse recorded views before estimating fixed occlusion')
stack = np.stack(frames).astype(np.float32)
q10,median,q90 = np.quantile(stack,[.1,.5,.9],axis=0)
fixed_dark = ((q90 < 28) & (q90-q10 < 22)).astype(np.uint8)
fixed_dark = cv2.morphologyEx(fixed_dark,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
# Select the persistent A-pillar component, not dark scenery islands.
n,labels,stats,_ = cv2.connectedComponentsWithStats(fixed_dark)
selected = np.zeros_like(fixed_dark)
for label in range(1,n):
    x,y,w,h,area = stats[label]
    if area > 180 and x < 150 and y < 85:
        selected[labels==label] = 1
selected = cv2.dilate(selected,np.ones((3,3),np.uint8))
visible = np.ones_like(selected)
visible[selected > 0] = 0
# This camera's forward windshield ends before the dashboard. The side window
# is excluded from forward planning even though scenery is visible through it.
visible[130:] = 0
n,components = cv2.connectedComponents(visible)
visible = (components[60,270] == components).astype(np.uint8)
front = visible.copy()
side = np.zeros_like(visible)
polygon = np.array([[0,76],[46,105],[69,136],[44,153],[0,171]],np.int32)
cv2.fillPoly(side,[polygon],1)
side[selected > 0] = 0
visible = np.maximum(front,side)
planning_roi = front.copy()
planning_roi[:74] = 0
planning_roi[:,:95] = 0
out = ROOT/'models'/'fh6_driver_visibility.npz'
np.savez_compressed(out,visible=visible,front=front,side_left=side,planning_roi=planning_roi,fixed_dark=selected,median=median.astype(np.uint8),source_count=len(frames))
out.with_suffix('.json').write_text(json.dumps(dict(camera='FH6 GR Corolla driver, 1920x1080',
    method='90th-percentile persistent dark connected cockpit component; reviewed windshield bounds',
    frames=used,frame_count=len(frames),training_label_status='geometry estimate; not road ground truth'),indent=2))
preview = cv2.imread(str(ROOT/used[-1]))
mask = cv2.resize(visible,(1920,1080),interpolation=cv2.INTER_NEAREST)==0
preview[mask] = (preview[mask]*.4+np.array([20,20,180])*.6).astype(np.uint8)
side_preview = cv2.resize(side,(1920,1080),interpolation=cv2.INTER_NEAREST)>0
preview[side_preview] = (preview[side_preview]*.6+np.array([240,190,30])*.4).astype(np.uint8)
cv2.imwrite(str(ROOT/'out'/'visibility-calibration.jpg'),preview)
print(json.dumps(dict(frames=len(frames),visible_pixels=int(visible.sum()),file=str(out))))
