"""Compare temporal heads on refined visual features before the dense import finishes."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mono_drive.forza_local_learning import read_json,write_json
from mono_drive.forza_supervisor import stop_worker


def main():
    p=argparse.ArgumentParser();p.add_argument('--visual-dir',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();state=ROOT/'out/visual-refinement-state.json'
    def stopped():return (ROOT/'out/training.stop').exists() or (ROOT/'out/longrun.stop').exists()
    while not (a.visual_dir/'report.json').exists():
        if stopped():return
        write_json(state,{'phase':'waiting_for_visual_model','time':time.time()});time.sleep(3)
    report=read_json(a.visual_dir/'report.json',{})
    if not report.get('accepted_visual'):
        write_json(state,{'phase':'visual_candidate_rejected','time':time.time()});return
    cache=ROOT/'out/temporal-refined'/a.out.name
    commands=[
        [sys.executable,'tools/prepare_temporal_driving.py','--manifest',str(a.manifest),'--out',str(cache),'--visual-encoder-dir',str(a.visual_dir)],
        [sys.executable,'tools/train_temporal_steering.py','--cache',str(cache),'--out',str(a.out),'--epochs','24','--samples-per-epoch','120000'],
        [sys.executable,'tools/evaluate_temporal_live.py','--model-dir',str(a.out)],
    ]
    flags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name=='nt' else 0
    with (ROOT/'out/visual-refinement.log').open('a',encoding='utf-8') as log:
        for command in commands:
            if stopped():return
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
            while child.poll() is None:
                write_json(state,{'phase':'running','stage':Path(command[1]).stem,'time':time.time(),'child_pid':child.pid,'candidate':str(a.out)})
                if stopped():stop_worker(child);return
                time.sleep(2)
            if child.returncode:
                write_json(state,{'phase':'failed','stage':Path(command[1]).stem,'exit_code':child.returncode,'time':time.time()});return
    write_json(state,{'phase':'complete','candidate':str(a.out),'time':time.time()})

if __name__=='__main__':main()
