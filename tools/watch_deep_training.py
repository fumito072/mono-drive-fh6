"""Resume this one local training job after unexpected exits; honor stop files."""
import argparse,json,os,socket,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mono_drive.forza_local_learning import read_json,write_json


def training_command_matches(command,folder,cwd=ROOT):
    try:
        script=next((cwd/Path(x)).resolve() for x in command if x.endswith('train_temporal_steering.py'))
        target=(cwd/Path(command[command.index('--out')+1])).resolve()
        variant=command[command.index('--variant')+1]
        return script==ROOT/'tools/train_temporal_steering.py' and target==folder.resolve() and variant=='deep'
    except (StopIteration,ValueError,IndexError):return False


def matching_processes(folder):
    # Windows venv launchers have a second Python child. A launcher exiting
    # must not cause a duplicate training job while that child is still alive.
    result=[]
    for process in psutil.process_iter(['cmdline']):
        try:
            if training_command_matches(process.info['cmdline'] or [],folder,Path(process.cwd())):result.append(process)
        except (psutil.Error,OSError):continue
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--model-dir',type=Path,required=True);a=parser.parse_args();folder=a.model_dir.resolve()
    if (ROOT/'models').resolve() not in folder.parents:raise ValueError('Expected a local model directory')
    process=psutil.Process(a.pid);command=process.cmdline()
    if not training_command_matches(command,folder):raise ValueError('PID is not the requested deep training job')
    if '--resume' not in command:command.append('--resume')
    retries=0;state=ROOT/'out/deep-learning-state.json'
    guard=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    if os.name=='nt':guard.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
    try:guard.bind(('127.0.0.1',20086))
    except OSError:return
    flags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name=='nt' else 0
    with (ROOT/'out/deep-resume.log').open('a',encoding='utf-8') as log:
        while True:
            if (ROOT/'out/training.stop').exists() or (ROOT/'out/longrun.stop').exists():
                write_json(state,dict(phase='stopped',time=time.time(),retries=retries));return
            report=read_json(folder/'report.json',{})
            if report.get('phase')=='complete':
                write_json(state,dict(phase='complete',accepted_offline=report.get('accepted_offline'),time=time.time(),retries=retries));return
            live=matching_processes(folder)
            if not live:
                if retries>=3:
                    write_json(state,dict(phase='needs_attention',error='Three checkpoint-resume attempts failed',time=time.time()));return
                retries+=1
                write_json(state,dict(phase='resuming_checkpoint',time=time.time(),retries=retries))
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
                process=psutil.Process(child.pid)
            else:process=live[-1]
            write_json(state,dict(phase='training',pid=process.pid,retries=retries,time=time.time(),
                model_dir=str(folder),progress=read_json(folder/'training-progress.json',{})))
            time.sleep(5)

if __name__=='__main__':main()
