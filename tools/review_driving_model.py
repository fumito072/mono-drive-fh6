"""Reconcile a saved candidate with the model currently used by the game."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mono_drive.forza_local_learning import read_json,write_json
from mono_drive.forza_model_policy import offer,model_path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args()
    candidate=model_path(ROOT,args.candidate)
    if (ROOT/'models/candidates').resolve() not in candidate.parents:raise ValueError('Unknown candidate location')
    active=read_json(ROOT/'configs/forza_horizon6.json').get('models',{}).get('pathnet','models/pathnet.onnx')
    baseline=model_path(ROOT,active)
    automatic=read_json(ROOT/'out/training-dataset.json')['segments']
    human=read_json(ROOT/'out/human-dataset.json',{})
    evaluation=ROOT/'out/driving-model-review.json'
    manifest=ROOT/'out/driving-model-review-data.json'
    write_json(manifest,dict(drives=automatic+human.get('train',[]),extra_validation=human.get('validation',[])))
    with (ROOT/'out/driving-model-review.log').open('w',encoding='utf-8') as log:
        command=[sys.executable,'tools/evaluate_forza_candidate.py','--candidate',str(candidate),
                 '--baseline',str(baseline),'--out',str(evaluation),'--dataset-manifest',str(manifest)]
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    accepted=offer(ROOT,candidate,str(baseline),read_json(evaluation))
    write_json(ROOT/'out/driving-model-review-state.json',dict(time=time.time(),accepted=accepted,
               candidate=str(candidate),baseline=str(baseline),evaluation=str(evaluation)))


if __name__=='__main__':main()
