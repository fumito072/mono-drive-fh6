"""Shadow and measured driving trial for an independently validated candidate."""
import argparse,json,os,socket,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mono_drive.forza_local_learning import read_json,write_json
from mono_drive.forza_incidents import tail_rows
from mono_drive.temporal_trial import DrivingMetrics,trial_passed,target_achieved
OUT=ROOT/'out';CONFIG=ROOT/'configs/temporal_steering.json';STATE=OUT/'temporal-live-evaluation.json'


def mode(value,folder):
    config=read_json(CONFIG,{});config.update(mode=value,model_dir=str(folder.relative_to(ROOT)))
    write_json(CONFIG,config)


def stopped():return (OUT/'training.stop').exists() or (OUT/'longrun.stop').exists()


def collect(metrics):
    heartbeat=read_json(OUT/'driver-heartbeat.json',{});run=Path(heartbeat.get('run','')).resolve()
    if run.parent==OUT.resolve():metrics.consume(tail_rows(run/'decisions.jsonl',metrics.last_t),str(run))


def save(folder,publish=True,**values):
    values['hold_route']=values.get('phase') in ('shadow','trial')
    values.update(time=time.time(),model_dir=str(folder))
    if publish:write_json(STATE,values)
    archive=OUT/'temporal-live-evaluations';archive.mkdir(exist_ok=True)
    write_json(archive/(folder.name+'.json'),values)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,default=ROOT/'models/temporal-20261103');a=p.parse_args()
    folder=a.model_dir.resolve()
    if (ROOT/'models').resolve() not in folder.parents:raise ValueError('Model must be in models directory')
    while not (folder/'report.json').exists():
        if stopped():return
        time.sleep(3)
    report=read_json(folder/'report.json',{})
    if not report.get('accepted_offline'):
        save(folder,publish=False,phase='offline_rejected',reason='independent_test_gate',controls_changed=False);return
    # Distinct training workers may finish together. Only one evaluator may
    # alter the shared runtime model/configuration at a time.
    guard=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    if os.name=='nt':guard.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
    while True:
        if stopped():return
        try:guard.bind(('127.0.0.1',20085));break
        except OSError:time.sleep(3)
    mode('shadow',folder);begin=time.time();baseline=DrivingMetrics();baseline.last_t=begin
    initial=None;samples=agreeing=0
    while time.time()-begin<1800:
        if stopped():mode('shadow',folder);return
        collect(baseline);state=read_json(OUT/'temporal-steering-state.json',{});now=time.time()
        if state.get('phase')=='shadow' and state.get('model_dir')==str(folder) and now-state.get('time',0)<5:
            if initial is None:initial=state
            samples=max(0,state.get('eligible',0)-initial.get('eligible',0))
            agreeing=max(0,state.get('agreeing',0)-initial.get('agreeing',0))
        save(folder,phase='shadow',samples=samples,agreement_fraction=agreeing/max(1,samples),baseline=baseline.summary())
        if samples>=500 and baseline.seconds>=120:
            if agreeing/max(1,samples)<.9:
                save(folder,phase='shadow_only',reason='live_prediction_disagreement',baseline=baseline.summary());return
            break
        time.sleep(1)
    else:
        save(folder,phase='shadow_only',reason='insufficient_reliable_observations',baseline=baseline.summary());return
    base=baseline.summary();trial=DrivingMetrics();start=time.time();trial.last_t=start
    applied_start=state.get('applied',0);incidents=set();session=read_json(OUT/'longrun-state.json',{})
    previous_incident=(session.get('last_incident') or {}).get('id');mode('trial',folder)
    while time.time()-start<300:
        if stopped():mode('shadow',folder);return
        collect(trial);state=read_json(OUT/'temporal-steering-state.json',{});session=read_json(OUT/'longrun-state.json',{})
        incident=(session.get('last_incident') or {}).get('id')
        applied=max(0,state.get('applied',0)-applied_start)
        if incident!=previous_incident and applied:
            incidents.add(incident);mode('shadow',folder)
            save(folder,phase='rolled_back',reason='incident_during_trial',baseline=base,trial=trial.summary(),applied_samples=applied,target_achieved=False);return
        save(folder,phase='trial',baseline=base,trial=trial.summary(),applied_samples=applied)
        time.sleep(1)
    summary=trial.summary();passed=trial_passed(base,summary,applied,len(incidents));mode('active' if passed else 'shadow',folder)
    save(folder,phase='active' if passed else 'inconclusive',baseline=base,trial=summary,applied_samples=applied,
         target_achieved=target_achieved(summary,len(incidents)),maximum_steer_adjustment=.025,
         reason='measured_trial_passed' if passed else 'driving_metrics_gate')

if __name__=='__main__':main()
