"""Continue local video/steering learning when verified human data changes."""
import json,os,socket,subprocess,sys,time
from pathlib import Path
from .forza_local_learning import read_json,write_json

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'out'

def main():
    guard=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    if os.name=='nt':guard.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
    try:guard.bind(('127.0.0.1',20083))
    except OSError:return 0
    first=ROOT/'models/temporal-20261103/report.json'
    previous=read_json(OUT/'temporal-worker-state.json',{}).get('last_revision')
    def stopped():return (OUT/'training.stop').exists() or (OUT/'longrun.stop').exists()
    flags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name=='nt' else 0
    while not stopped():
        if read_json(ROOT/'configs/temporal_steering.json',{}).get('enabled') is not True:break
        if not previous:
            initial=read_json(first,{})
            if initial.get('phase')=='complete':previous=initial.get('dataset_revision')
        data=read_json(OUT/'human-dataset.json',{});revision=data.get('revision')
        if previous and revision and revision!=previous:
            destination=ROOT/'models/temporal_candidates'/time.strftime('%Y%m%d_%H%M%S')
            destination.mkdir(parents=True,exist_ok=True)
            try:
                with (OUT/'temporal-worker.log').open('a',encoding='utf-8') as log:
                    for command in ([sys.executable,'tools/prepare_temporal_driving.py'],
                                    [sys.executable,'tools/train_temporal_steering.py','--out',str(destination)]):
                        write_json(OUT/'temporal-worker-state.json',{'phase':'training','pid':os.getpid(),'last_revision':previous,'time':time.time()})
                        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=flags)
                report=read_json(destination/'report.json',{});previous=report.get('dataset_revision',revision)
                if report.get('accepted_offline') and not stopped():
                    with (OUT/'temporal-live-evaluation.log').open('a',encoding='utf-8') as log:
                        subprocess.Popen([sys.executable,'tools/evaluate_temporal_live.py','--model-dir',str(destination)],
                            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
            except (subprocess.CalledProcessError,OSError) as e:
                write_json(OUT/'temporal-worker-state.json',{'phase':'retry_later','last_revision':previous,'time':time.time(),'error':str(e)})
                time.sleep(60)
        write_json(OUT/'temporal-worker-state.json',{'phase':'watching' if previous else 'waiting_for_first_training',
            'pid':os.getpid(),'last_revision':previous,'time':time.time(),'local_only':True})
        time.sleep(10)
    guard.close();return 0

if __name__=='__main__':raise SystemExit(main())
