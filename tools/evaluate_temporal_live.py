"""Bounded local shadow/trial evaluation; never changes camera or menus."""
import argparse,json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'out';CONFIG=ROOT/'configs/temporal_steering.json'
def read(p):
    try:return json.loads(Path(p).read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}
def write(p,v):
    p=Path(p);t=p.with_suffix('.live.tmp');t.write_text(json.dumps(v,indent=2),encoding='utf-8')
    for _ in range(30):
        try:t.replace(p);return
        except PermissionError:time.sleep(.1)
    raise OSError('Could not publish local evaluation state')
def mode(value):
    config=read(CONFIG);config['mode']=value;write(CONFIG,config)

parser=argparse.ArgumentParser();parser.add_argument('--model-dir',type=Path,default=ROOT/'models/temporal-20261103')
args=parser.parse_args();folder=args.model_dir.resolve()
if (ROOT/'models').resolve() not in folder.parents:raise ValueError('Model must be in models directory')
while not (folder/'report.json').exists():
    if (OUT/'training.stop').exists() or (OUT/'longrun.stop').exists():raise SystemExit('Explicit stop')
    time.sleep(3)
report=read(folder/'report.json')
config=read(CONFIG);config['model_dir']=str(folder.relative_to(ROOT));write(CONFIG,config)
mode('shadow');begin=time.time();initial=None
while True:
    if (OUT/'training.stop').exists() or (OUT/'longrun.stop').exists():mode('shadow');raise SystemExit('Explicit stop')
    state=read(OUT/'temporal-steering-state.json');now=time.time()
    if state.get('phase')=='shadow' and state.get('model_dir')==str(folder) and now-state.get('time',0)<5:
        initial=initial or state
        samples=state.get('eligible',0)-initial.get('eligible',0)
        agreeing=state.get('agreeing',0)-initial.get('agreeing',0)
        write(OUT/'temporal-live-evaluation.json',{'phase':'shadow','time':now,'samples':samples,
            'agreement_fraction':agreeing/max(1,samples),'accepted_offline':report.get('accepted_offline')})
        if samples>=500:
            if not report.get('accepted_offline') or agreeing/max(1,samples)<.9:
                write(OUT/'temporal-live-evaluation.json',{'phase':'shadow_only','time':now,'reason':'offline_or_live_agreement_gate','samples':samples})
                raise SystemExit(0)
            break
    if now-begin>1800:
        write(OUT/'temporal-live-evaluation.json',{'phase':'shadow_only','time':now,'reason':'insufficient_eligible_observations'})
        raise SystemExit(0)
    time.sleep(1)
mode('trial');start=time.time();applied_start=state.get('applied',0)
incident=read(OUT/'longrun-state.json').get('last_incident',{}).get('id')
while time.time()-start<180:
    if (OUT/'training.stop').exists() or (OUT/'longrun.stop').exists():mode('shadow');raise SystemExit('Explicit stop')
    state=read(OUT/'temporal-steering-state.json');session=read(OUT/'longrun-state.json')
    new_incident=session.get('last_incident',{}).get('id')
    if new_incident!=incident and state.get('applied',0)>applied_start:
        mode('shadow');write(OUT/'temporal-live-evaluation.json',{'phase':'rolled_back','time':time.time(),'reason':'incident_during_bounded_trial'})
        raise SystemExit(0)
    time.sleep(1)
applied=state.get('applied',0)-applied_start
mode('active' if applied>=30 else 'shadow')
write(OUT/'temporal-live-evaluation.json',{'phase':'active' if applied>=30 else 'inconclusive','time':time.time(),
      'applied_samples':applied,'duration_s':time.time()-start,'maximum_steer_adjustment':.025})
