"""Inspect local GPU road/traffic segmentation on a recorded game frame."""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

MODEL = 'nvidia/segformer-b2-finetuned-cityscapes-1024-1024'

def load():
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to('cuda').eval()
    return None, model

def predict(processor, model, frame):
    # The dashboard dominates full-frame semantics. Only classify the windshield.
    horizon_crop = frame[:round(frame.shape[0]*.60)]
    rgb = cv2.resize(cv2.cvtColor(horizon_crop, cv2.COLOR_BGR2RGB), (896, 512))
    pixels = torch.from_numpy(rgb.copy()).to(device='cuda', dtype=torch.float32).permute(2,0,1)[None]/255
    mean = pixels.new_tensor([.485,.456,.406])[None,:,None,None]
    std = pixels.new_tensor([.229,.224,.225])[None,:,None,None]
    with torch.inference_mode():
        logits = model(pixel_values=(pixels-mean)/std).logits
        logits = torch.nn.functional.interpolate(logits, size=horizon_crop.shape[:2], mode='bilinear', align_corners=False)
        labels = logits.argmax(dim=1)[0].byte().cpu().numpy()
    return np.pad(labels, ((0,frame.shape[0]-len(labels)),(0,0)), constant_values=255)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--out', default='out/semantic-check.jpg')
    args = parser.parse_args()
    frame = cv2.imread(args.image)
    if frame is None:
        raise ValueError(args.image)
    processor, model = load()
    started = time.perf_counter()
    labels = predict(processor, model, frame)
    elapsed = time.perf_counter()-started
    out = frame.copy()
    out[labels == 0] = (out[labels == 0]*.45+np.array([30,220,40])*.55).astype(np.uint8)
    traffic = (labels >= 13) & (labels <= 18)
    out[traffic] = (out[traffic]*.3+np.array([0,100,255])*.7).astype(np.uint8)
    cv2.imwrite(args.out, out)
    cv2.imwrite(str(Path(args.out).with_suffix('.png')), labels)
    print({'seconds':elapsed,'labels':{model.config.id2label.get(int(k),'ignored'):int(v) for k,v in zip(*np.unique(labels,return_counts=True))}}, flush=True)
