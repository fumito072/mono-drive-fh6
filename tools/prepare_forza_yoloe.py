"""Cache explicit traffic/roadside prompts for offline YOLOE-26 inference."""
import json
from pathlib import Path
import time

import cv2
from ultralytics import YOLOE

ROOT=Path(__file__).resolve().parents[1]
names=['car','truck','bus','motorcycle','bicycle','person','guardrail','fence','concrete barrier']
model=YOLOE(str(ROOT/'models/yoloe-26s-seg.pt'))
model.set_classes(names)
model.save_prompt_embeddings(str(ROOT/'models/fh6-yoloe-prompts.npz'))
model.save(str(ROOT/'models/fh6-yoloe-26s.pt'))
results=[]
files=[ROOT/'out/obstacle-diagnostic-current.jpg',ROOT/'out/fh6_20261027_101656/frames/000130.jpg',
       ROOT/'out/fh6_20261027_101656/frames/000161.jpg']
for i,file in enumerate(files):
    image=cv2.imread(str(file))
    if image is None:
        continue
    start=time.monotonic()
    result=model.predict(image,device=0,imgsz=960,conf=.25,verbose=False)[0]
    results.append(dict(file=str(file),seconds=time.monotonic()-start,names=result.names,
                        boxes=result.boxes.data.cpu().numpy().tolist()))
    cv2.imwrite(str(ROOT/f'out/yoloe-check-{i}.jpg'),result.plot())
(ROOT/'out/yoloe-check.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results))
