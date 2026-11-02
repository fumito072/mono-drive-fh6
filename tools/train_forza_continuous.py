"""Train filtered FH6 candidates and offer validated models for seamless trials."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mono_drive.forza_local_learning import training_settings, training_trigger, read_json, record_candidate, write_json
from mono_drive.forza_session_policy import remaining_seconds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'out'
state_path = OUT/'training-state.json'
stop_path = OUT/'training.stop'
guard = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
guard.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
try:
    guard.bind(('127.0.0.1',20069))
except OSError:
    raise SystemExit(0)
try:
    previous = json.loads(state_path.read_text()).get('last_trained_frames',40)
except (OSError,ValueError):
    previous = 40
saved=read_json(state_path,{})
last_trial_profile=saved.get('last_trial_profile','balanced')
last_trial_at=saved.get('last_trial_at',0.)
last_human_revision=saved.get('last_human_revision','')
last_refresh_at=saved.get('last_refresh_at',0.)
env = dict(os.environ,PYTHONIOENCODING='utf-8')

def status(phase,**fields):
    value = dict(phase=phase,pid=os.getpid(),time=time.time(),last_trained_frames=previous,
                 automatic_activation=True,activation_policy='validated_live_trial',
                 last_trial_profile=last_trial_profile,last_trial_at=last_trial_at,
                 last_human_revision=last_human_revision,last_refresh_at=last_refresh_at,**fields)
    temp = state_path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    try:
        temp.replace(state_path)
    except PermissionError:
        pass
    print(json.dumps(value),flush=True)

def deadline_seconds():
    try:
        config = json.loads((OUT/'longrun.json').read_text())
        return remaining_seconds(config,time.time())
    except (OSError,ValueError,KeyError):
        return None


def finished():
    remaining = deadline_seconds()
    return stop_path.exists() or (OUT/'longrun.stop').exists() or (remaining is not None and remaining <= 0)


while not finished():
    try:
        status('collecting')
        # The independent importer can prepare new human recordings while this
        # process trains. Keep the last complete, atomically published dataset.
        from mono_drive.forza_session import port_busy
        if not port_busy(20074) or not (OUT/'human-dataset.json').exists():
            subprocess.run([sys.executable,'tools/prepare_human_driving.py'],cwd=ROOT,env=env,
                           stdout=subprocess.DEVNULL,check=True,timeout=deadline_seconds())
        human=read_json(OUT/'human-dataset.json',{})
        scan = subprocess.run([sys.executable,'tools/prepare_forza_training.py','--scan-only'],cwd=ROOT,env=env,
                              capture_output=True,text=True,encoding='utf-8',check=True)
        data = json.loads(scan.stdout)
        count = data['frames']+human.get('frames',0)
        profile,settings=training_settings(OUT)
        trigger=training_trigger(count,previous,profile,last_trial_profile,last_trial_at,time.time())
        if human.get('train') and human.get('revision')!=last_human_revision:trigger='new_human_demonstrations'
        if trigger:
            last_trial_profile,last_trial_at=profile,time.time()
            subprocess.run([sys.executable,'tools/prepare_forza_training.py'],cwd=ROOT,env=env,
                           stdout=subprocess.DEVNULL,check=True)
            data = json.loads((OUT/'training-dataset.json').read_text())
            drives=data['segments']+human.get('train',[])
            held=human.get('validation',[])
            tag = time.strftime('%Y%m%d_%H%M%S')
            destination = ROOT/'models'/'candidates'
            destination.mkdir(exist_ok=True)
            candidate = destination/f'pathnet_fh6_{tag}.onnx'
            seed=Path(read_json(OUT/'learning-seed.json',{}).get('model',str(ROOT/'models/pathnet_gpu_candidate.onnx')))
            allowed=ROOT/'models'/'candidates'
            if not seed.exists() or (seed.resolve()!= (ROOT/'models/pathnet_gpu_candidate.onnx').resolve() and allowed.resolve() not in seed.resolve().parents):
                seed=ROOT/'models/pathnet_gpu_candidate.onnx'
            status('training',available_frames=count,human_frames=human.get('frames',0),candidate=str(candidate),profile=profile,settings=settings,trigger=trigger)
            training_manifest=OUT/f'training_{tag}_data.json'
            evaluation_manifest=OUT/f'training_{tag}_evaluation_data.json'
            write_json(training_manifest,dict(drives=drives+['samples/kreuzungsring_rain','samples/sweden_unseen'],extra_validation=held))
            write_json(evaluation_manifest,dict(drives=drives,extra_validation=held))
            command = [sys.executable,'tools/train_pathnet.py','--dataset-manifest',str(training_manifest),
                       '--out',str(candidate),'--init',str(seed),
                       '--epochs',str(settings['epochs']),'--batch','128','--lr',str(settings['lr']),
                       '--curve-weight',str(settings['curve_weight']),'--device','cuda']
            # One refresh per 15 minutes prevents a steady upload stream from
            # continually cancelling training before an epoch can finish.
            if human.get('revision') and time.time()-last_refresh_at>=900.:
                command+=['--human-revision',human['revision']]
            active=read_json(ROOT/'configs/forza_horizon6.json',{}).get('models',{}).get('pathnet','models/pathnet.onnx')
            evaluation_baseline=Path(active)
            if not evaluation_baseline.is_absolute():evaluation_baseline=ROOT/evaluation_baseline
            with (OUT/f'training_{tag}.log').open('w',encoding='utf-8') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,
                               timeout=deadline_seconds())
                evaluation = OUT/f'training_{tag}_evaluation.json'
                status('evaluating',available_frames=count,candidate=str(candidate),profile=profile)
                subprocess.run([sys.executable,'tools/evaluate_forza_candidate.py','--candidate',str(candidate),
                    '--baseline',str(evaluation_baseline),'--out',str(evaluation),
                    '--dataset-manifest',str(evaluation_manifest)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,
                               timeout=deadline_seconds())
            verdict=record_candidate(OUT,read_json(evaluation,{}),evaluation_baseline,candidate,profile)
            if verdict['accepted']:
                write_json(OUT/'learning-seed.json',dict(model=str(candidate),time=time.time(),
                    automatic_driving_activation=False,source='held_out_candidate_comparison'))
                from mono_drive.forza_model_policy import offer
                offer(ROOT,candidate,str(evaluation_baseline),read_json(evaluation,{}))
            previous = count
            last_human_revision=human.get('revision','')
            status('candidate_saved',candidate=str(candidate),evaluation=str(evaluation),accepted=verdict['accepted'],profile=profile,human_frames=human.get('frames',0))
        else:
            status('waiting_for_new_data',available_frames=count,next_training_at=previous+settings['new_frames'],profile=profile,settings=settings)
    except subprocess.CalledProcessError as error:
        if error.returncode==75:
            last_refresh_at=time.time()
            status('refreshing_new_human_data')
            continue
        status('retry_later',error=str(error))
    except Exception as error:
        status('retry_later',error=str(error))
    for _ in range(120):
        if finished():
            break
        time.sleep(1)
status('stopped')
guard.close()
