"""Temporal held-out comparison for FH6 trajectory pseudo-labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from train_pathnet import load_items,stack,scores,predict
from torch_models import build_pathnet,load_onnx_into
from mono_drive.training_arguments import dataset_arguments,resolve_dataset_arguments
from mono_drive.forza_evaluation import group_recordings

parser = argparse.ArgumentParser()
parser.add_argument('--candidate',required=True)
parser.add_argument('--baseline',default='models/pathnet_gpu_candidate.onnx')
parser.add_argument('--out',default='out/fh6-candidate-evaluation.json')
dataset_arguments(parser)
args = resolve_dataset_arguments(parser.parse_args(),parser)
_,validation = load_items(args.drives,None,1,.15)
legacy_count=len(validation)
human_ranges=[]
if args.extra_validation:
    for drive in args.extra_validation:
        held,_=load_items([drive],None,1,0.)
        start=len(validation)
        validation.extend(held)
        human_ranges.append((drive,start,len(validation)))
if not validation:
    raise RuntimeError('No held-out FH6 trajectory labels')
X,T,OK,K = stack(validation)
result = dict(label_type='automatically filtered executed trajectory; not independently verified',
              held_out_frames=len(X),automatic_activation=False,models={},evaluation_version=2,conditions={})
# Matched camera degradation checks, inspired by DriveDegrade; these are two
# project-specific checks, not a reproduction of its full benchmark.
blur=np.stack([cv2.GaussianBlur(im.transpose(1,2,0),(5,5),1.0).transpose(2,0,1) for im in X])
dim=(X.astype(np.float32)*.65).astype(np.uint8)
for file in [args.baseline,args.candidate]:
    model = build_pathnet()
    load_onnx_into(model,file)
    model = model.to('cuda')
    errors,_,_ = scores(predict(model,X,'cuda'),T,OK,K)
    result['models'][file] = dict(mean_m=float(np.nanmean(errors)),median_m=float(np.nanmedian(errors)))
    if args.extra_validation:
        for name,part in (('legacy_holdout',errors[:legacy_count]),('human_holdout',errors[legacy_count:])):
            result.setdefault(name,dict(frames=len(part),models={}))['models'][file]=dict(mean_m=float(np.nanmean(part)),median_m=float(np.nanmedian(part)))
        for drive,start,end in human_ranges:
            part=errors[start:end]
            result.setdefault('human_recordings',{}).setdefault(drive,dict(frames=len(part),models={}))["models"][file]=dict(
                mean_m=float(np.nanmean(part)) if len(part) else None,
                median_m=float(np.nanmedian(part)) if len(part) else None,
                valid_frames=int(np.isfinite(part).sum()))
    for name,images in (('blur',blur),('low_light',dim)):
        errors,_,_=scores(predict(model,images,'cuda'),T,OK,K)
        result['conditions'].setdefault(name,{})[file]=dict(mean_m=float(np.nanmean(errors)),median_m=float(np.nanmedian(errors)))
result=group_recordings(result,Path(__file__).resolve().parents[1])
Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
