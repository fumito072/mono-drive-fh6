"""Stage reviewed human demonstrations for path candidates, never raw action replay."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mono_drive.human_data_audit import inspect_csv
from mono_drive.forza_local_learning import read_json,write_json


def checked_ranges(review,duration):
    train=review.get('path_train_ranges',[]);validation=review.get('path_validation_ranges',[])
    if not train or not validation:raise ValueError('Disjoint training and validation time ranges are required')
    for a,b in train+validation:
        if not (0<=a<b<=duration and b-a>=10):raise ValueError('Invalid human time range')
    for a,b in train:
        if any(not (b+8<=c or d+8<=a) for c,d in validation):
            raise ValueError('Human train and validation ranges need an eight-second guard interval')
    return train,validation


def main(sync=True):
    # Drive desktop authentication remains outside the learning process. Import
    # only a completed, stable copy before the existing quality audit runs.
    from mono_drive.forza_drive_import import sync_once
    if sync:
        try:
            sync_once(ROOT)
        except (OSError, ValueError, RuntimeError) as error:
            # A disconnected Drive must not prevent training on existing local data.
            write_json(ROOT/'out/drive-import-state.json',dict(time=time.time(),phase='retry_later',error=str(error)))
    subprocess.run([sys.executable,'tools/audit_human_driving.py'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    reviews=read_json(ROOT/'configs/human_data_reviews.json',{})
    audit=read_json(ROOT/'out/human-data-audit/index.json',{})
    all_train=[];all_validation=[];pending=[];episodes=[]
    records={item['recording']:read_json(ROOT/f"out/human-data-audit/{item['recording']}.json",{})
             for item in audit.get('recordings',[])}
    from mono_drive.human_prepare_queue import prepare_native_records
    prepared=prepare_native_records(ROOT,[(name,r) for name,r in records.items()
                                          if r.get('kind')=='fh6_recorder_session'])
    for item in audit.get('recordings',[]):
        name=item['recording'];review=reviews.get(name,{})
        record=records[name]
        if record.get('kind')=='fh6_recorder_session':
            try:
                result,error=prepared[name]
                if error is not None:raise ValueError(error)
                all_train+=result['train'];all_validation+=result['validation'];episodes.append(result)
            except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
                pending.append(dict(recording=name,reason=str(error)))
            continue
        # A coarse, visually checked offset is admitted only to an explicitly
        # authorized path-candidate experiment. It never becomes a pedal label.
        if not (record.get('numeric_eligible') and review.get('path_candidate_allowed') is True):
            pending.append(name);continue
        try:
            train,validation=checked_ranges(review,record['media']['duration_s'])
            sync=record['synchronization'];bounds=sync.get('compatible_offset_range_s')
            if not bounds or bounds[1]-bounds[0]>.15:raise ValueError('Synchronization uncertainty too large')
            offset=float(sync['offset_s']);fps=float(record['media']['fps'])
            if abs(fps-30)>1e-6:raise ValueError('This importer requires 30 fps constant-frame-rate footage')
            revision=hashlib.sha256(json.dumps(dict(fingerprint=record['fingerprint'],review=review),sort_keys=True).encode()).hexdigest()[:16]
            folder=ROOT/f'datasets/human/{name}_{revision}'
            manifest=folder/'manifest.json'
            if not manifest.exists():
                frames=folder/'video_frames';frames.mkdir(parents=True,exist_ok=True)
                subprocess.run(['ffmpeg','-y','-v','error','-threads','4','-i',record['video'],
                    '-vf','select=not(mod(n\\,6)),scale=1280:720','-fps_mode','vfr','-start_number','0',
                    '-q:v','4',str(frames/'%07d.jpg')],check=True)
                _,columns,valid=inspect_csv(record['csv']);t=columns['elapsed_sec']
                camera=dict(read_json(ROOT/'configs/forza_horizon6.json')['camera'])
                camera.update(width=1280,height=720,name='WRX human recording; initial camera estimate')
                result=dict(revision=revision,recording=name,train=[],validation=[],frames=0,
                    input_source='human Forza Data Out',offset_s=offset,offset_uncertainty_s=(bounds[1]-bounds[0])/2,
                    labels='recorded future trajectory; experimental timing; not direct stick commands')
                for partition,ranges in (('train',train),('validation',validation)):
                    for number,(start,end) in enumerate(ranges):
                        target=folder/f'{partition}_{number:02d}';(target/'frames').mkdir(parents=True,exist_ok=True)
                        with (target/'poses.csv').open('w',newline='') as handle,(target/'actions.csv').open('w',newline='') as actions:
                            writer=csv.writer(handle);writer.writerow(['frame','t','x','y','z','yaw_deg','pitch_deg','kmh'])
                            aw=csv.writer(actions);aw.writerow(['frame','video_pts_s','csv_elapsed_s','applied_throttle','applied_brake','applied_steer'])
                            count=0
                            for source in sorted(frames.glob('*.jpg')):
                                video_time=int(source.stem)/5.
                                if not start<=video_time<end:continue
                                stamp=video_time+offset;j=int(np.searchsorted(t,stamp))
                                if j<1 or j>=len(t) or not(valid[j-1] and valid[j]):continue
                                dest=target/'frames'/f'{count:07d}.jpg'
                                if not dest.exists():
                                    try:dest.hardlink_to(source)
                                    except OSError:
                                        import shutil;shutil.copy2(source,dest)
                                at=lambda key:float(np.interp(stamp,t,columns[key]))
                                yaw=float(np.interp(stamp,t,np.unwrap(columns['yaw'])))
                                writer.writerow([count,stamp,at('position_x'),at('position_z'),at('position_y'),
                                    90-math.degrees(yaw),-math.degrees(at('pitch')),at('speed_kmh')])
                                aw.writerow([count,video_time,stamp,at('throttle'),at('brake'),at('steer')]);count+=1
                        (target/'camera.json').write_text(json.dumps(camera,indent=2))
                        write_json(target/'provenance.json',dict(recording=name,partition=partition,
                            source_csv=record['csv'],source_video=record['video'],time_range=[start,end],
                            offset_s=offset,offset_verified_for_actions=False,raw_controls_replayed=False))
                        result[partition].append(str(target));result['frames']+=count
                write_json(manifest,result)
            result=read_json(manifest)
            all_train+=result['train'];all_validation+=result['validation'];episodes.append(result)
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
            pending.append(dict(recording=name,reason=str(error)))
    signature=hashlib.sha256(json.dumps([e['revision'] for e in episodes],sort_keys=True).encode()).hexdigest()
    result=dict(time=time.time(),revision=signature,train=all_train,validation=all_validation,
                frames=sum(e['frames'] for e in episodes),episodes=episodes,pending=pending,
                automatic_raw_action_learning=False)
    write_json(ROOT/'out/human-dataset.json',result)
    print(json.dumps(dict(revision=signature,frames=result['frames'],pending=pending)))


if __name__=='__main__':
    from mono_drive.forza_data_worker import acquire_guard
    guard=acquire_guard(20075)
    if guard is not None:
        try:main()
        finally:guard.close()
