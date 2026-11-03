"""Prepare weak video observations and validate their value against real teachers."""
import os,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mono_drive.forza_local_learning import read_json,write_json
from mono_drive.forza_supervisor import stop_worker


def main():
    guard=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    if os.name=='nt':guard.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
    try:guard.bind(('127.0.0.1',20084))
    except OSError:return
    state=ROOT/'out/uncontrolled-learning-state.json'
    flags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name=='nt' else 0
    def stopped():return (ROOT/'out/training.stop').exists() or (ROOT/'out/longrun.stop').exists()
    def run(command,phase,log):
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
        while child.poll() is None:
            write_json(state,{'phase':phase,'child_pid':child.pid,'time':time.time()})
            if stopped():stop_worker(child);return False
            time.sleep(2)
        if child.returncode:write_json(state,{'phase':'failed','stage':phase,'exit_code':child.returncode,'time':time.time()})
        return child.returncode==0
    with (ROOT/'out/uncontrolled-learning.log').open('a',encoding='utf-8') as log:
        if not run([sys.executable,'tools/prepare_uncontrolled_video.py'],'preparing_image_motion',log):return
        # Keep the requested first driving evaluation ahead of this extra experiment.
        while read_json(ROOT/'out/visual-refinement-state.json',{}).get('phase') not in ('complete','failed','visual_candidate_rejected'):
            if stopped():return
            write_json(state,{'phase':'waiting_for_first_driving_evaluation','time':time.time()});time.sleep(10)
        directory=ROOT/'models/visual_encoders/weak_motion_20261103'
        if not run([sys.executable,'tools/train_visual_encoder.py','--manifest','out/visual-refinement-input.json',
                    '--out',str(directory),'--motion-manifest','out/video-motion/manifest.json'],'learning_with_weak_motion',log):return
        if not run([sys.executable,'tools/run_visual_refinement.py','--visual-dir',str(directory),
                    '--manifest','out/visual-refinement-input.json','--out','models/temporal_candidates/weak_motion_20261103'],
                    'validating_with_real_controls',log):return
    write_json(state,{'phase':'complete','time':time.time()})

if __name__=='__main__':main()
