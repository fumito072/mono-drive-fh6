"""Extract real VFR frames and weak motion targets without inventing control labels."""
import argparse,csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mono_drive.video_motion_learning import motion_patch,image_motion
from mono_drive.temporal_steering import recording_split
from mono_drive.forza_local_learning import read_json,write_json


def main():
    cv2.setNumThreads(1)
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=0.)
    p.add_argument('--limit-videos',type=int,default=0);a=p.parse_args()
    source=read_json(ROOT/'out/teacher-control-audit.json',{});out=ROOT/'out/video-motion';out.mkdir(exist_ok=True)
    records=[x for x in source.get('records',[]) if x.get('status')=='error' and x.get('verdict')=='header_only_no_recorded_controls']
    if a.limit_videos:records=records[:a.limit_videos]
    if not records:raise ValueError('Run audit_teacher_controls.py --status error first')
    videos=[];total=0
    flags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name=='nt' else 0
    for index,record in enumerate(records):
        if (ROOT/'out/training.stop').exists() or (ROOT/'out/longrun.stop').exists():return
        video=Path(record['folder'])/'video.mkv'
        identity=hashlib.sha256((str(video)+str(video.stat().st_size)+str(video.stat().st_mtime_ns)+str(a.seconds)).encode()).hexdigest()[:20]
        folder=out/identity;frames=folder/'frames';frames.mkdir(parents=True,exist_ok=True)
        complete=folder/'complete.json'
        if not complete.exists():
            command=['ffmpeg','-y','-v','error','-hwaccel','cuda','-i',str(video)]
            if a.seconds:command+=['-t',str(a.seconds)]
            # select keeps source frames and PTS; it does not synthesize a fixed FPS stream.
            command+=['-vf','select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,0.5),scale=640:-2',
                      '-an','-fps_mode','vfr','-enc_time_base','1:1000','-frame_pts','1','-q:v','3',str(frames/'%012d.jpg')]
            with (folder/'decode.log').open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
                while child.poll() is None:
                    write_json(out/'state.json',{'phase':'extracting','video':index+1,'videos':len(records),'time':time.time()})
                    if (ROOT/'out/training.stop').exists() or (ROOT/'out/longrun.stop').exists():
                        from mono_drive.forza_supervisor import stop_worker
                        stop_worker(child);return
                    time.sleep(1)
                if child.returncode:raise RuntimeError('Video decode failed; see '+str(folder/'decode.log'))
            images=sorted(frames.glob('*.jpg'));pairs=[];previous=None
            for frame_index,image in enumerate(images):
                stamp=int(image.stem)/1000.;patch=motion_patch(cv2.imread(str(image)))
                if previous:
                    prior_path,prior_stamp,prior_patch=previous
                    motion=image_motion(prior_patch,patch,stamp-prior_stamp)
                    if motion:pairs.append({'previous':str(prior_path.resolve()),'current':str(image.resolve()),
                        'time_s':stamp,'dt_s':stamp-prior_stamp,**motion})
                previous=image,stamp,patch
                if frame_index%200==0:
                    write_json(out/'state.json',{'phase':'estimating_image_motion','video':index+1,'videos':len(records),
                        'frame':frame_index+1,'frames':len(images),'motion_pairs':total+len(pairs),'time':time.time()})
            pair_file=folder/'pairs.json';write_json(pair_file,pairs)
            write_json(complete,{'source_id':identity,'frames':len(images),'pairs':len(pairs),'pairs_file':str(pair_file.resolve()),
                'split':'validation' if recording_split(identity)=='test' else 'train',
                'supervision':'image motion only; no controls, metric velocity or world position',
                'time':time.time(),'sampled_source_pts':True})
        entry=read_json(complete,{});videos.append(entry);total+=entry['pairs']
        write_json(out/'state.json',{'phase':'preparing','complete':index+1,'videos':len(records),'motion_pairs':total,'time':time.time()})
    revision=hashlib.sha256(json.dumps([v['source_id'] for v in videos]).encode()).hexdigest()
    write_json(out/'manifest.json',{'revision':revision,'videos':videos,'motion_pairs':total,
        'contains_measured_controls':False,'contains_measured_metric_speed':False,'contains_world_position':False})
    write_json(out/'state.json',{'phase':'complete','videos':len(videos),'motion_pairs':total,'time':time.time()})

if __name__=='__main__':main()
