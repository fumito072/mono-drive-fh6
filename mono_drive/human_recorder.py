"""Validate the recorder's explicit frame-to-UDP correspondence before learning."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
from .human_data_audit import inspect_csv,inspect_video

FILES=('video.mkv','telemetry.csv','frames.csv','aligned.csv','telemetry.raw','session.json')
AUDIT_VERSION=2
PREPARATION_VERSION=2


def table(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def rebuild_alignment(frames,telemetry,original):
    """Use the common recorded clock; never invent packets across a pause."""
    frame_times=np.asarray([int(r['frame_t_ns']) for r in frames],dtype=np.int64)
    pts=np.asarray([int(r['video_pts_us']) for r in frames],dtype=np.int64)
    receive=np.asarray([int(r['receive_t_ns']) for r in telemetry],dtype=np.int64)
    if not len(receive) or np.any(np.diff(receive)<=0) or np.any(np.diff(frame_times)<=0):
        raise ValueError('Recorded clocks must be strictly increasing')
    if np.any(np.diff(pts)<=0) or np.ptp(frame_times-pts*1000)>2_000_000:
        raise ValueError('Frame clock and video presentation clock disagree')
    nearest=np.searchsorted(receive,frame_times,side='right')-1
    old={int(r['frame_index']):r for r in original}
    output=[];ages=[];excluded=Counter();corrected=0
    for frame,index in zip(frames,nearest):
        row=dict(frame,telemetry_seq='',telemetry_t_ns='',telemetry_age_ms='',valid='0')
        if index<0:
            excluded['before_first_packet']+=1
        else:
            sample=telemetry[int(index)]
            age=(int(frame['frame_t_ns'])-int(sample['receive_t_ns']))/1e6
            if age>100.:
                excluded['packet_gap_over_100ms']+=1
            elif int(sample.get('IsRaceOn',1))!=1:
                excluded['menu_or_loading']+=1
            else:
                row.update(telemetry_seq=sample['seq'],telemetry_t_ns=sample['receive_t_ns'],
                           telemetry_age_ms=f'{age:.6f}',valid='1')
                ages.append(age)
                previous=old.get(int(frame['frame_index']),{})
                try:
                    same=(previous.get('valid')=='1' and int(previous['telemetry_seq'])==int(sample['seq']) and
                          int(previous['frame_t_ns'])==int(frame['frame_t_ns']) and
                          int(previous['telemetry_t_ns'])==int(sample['receive_t_ns']) and
                          int(previous['video_pts_us'])==int(frame['video_pts_us']) and
                          abs(float(previous['telemetry_age_ms'])-age)<=.01)
                except (ValueError,KeyError,TypeError):same=False
                corrected+=int(not same)
        output.append(row)
    if len(ages)<100:raise ValueError('Too few clock-verified active frames')
    return output,ages,dict(excluded),corrected


def audit(folder,cache):
    folder=Path(folder);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    missing=[name for name in FILES if not (folder/name).is_file()]
    result=dict(recording=folder.name,kind='fh6_recorder_session',path=str(folder.resolve()),audit_version=AUDIT_VERSION,
                training_admitted=False,path_candidate_eligible=False,missing_files=missing)
    if missing:return dict(result,verdict='incomplete_recorder_session')
    metadata=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    if not isinstance(metadata,dict):raise ValueError('Invalid session metadata')
    if metadata.get('synthetic_demo') is not False or metadata.get('status')!='complete' or metadata.get('errors'):
        return dict(result,verdict='unfinished_or_error_session',session=metadata)
    telemetry=table(folder/'telemetry.csv');frames=table(folder/'frames.csv');aligned=table(folder/'aligned.csv')
    by_seq={int(r['seq']):r for r in telemetry};by_frame={int(r['frame_index']):r for r in frames}
    if len(by_seq)!=len(telemetry) or len(by_frame)!=len(frames) or sorted(by_frame)!=list(range(len(frames))):
        raise ValueError('Duplicate or missing frame/telemetry sequence')
    verified,ages,excluded,corrected=rebuild_alignment(frames,telemetry,aligned)
    packets=0;raw_sequences=set()
    with (folder/'telemetry.raw').open('rb') as raw:
        if raw.read(8)!=b'FH6UDP1\n':raise ValueError('Unknown raw packet format')
        while True:
            header=raw.read(20)
            if not header:break
            if len(header)!=20:raise ValueError('Truncated raw UDP header')
            seq,stamp,size=struct.unpack('<QQI',header)
            if size!=324:raise ValueError('Raw UDP record mismatch')
            payload=raw.read(size);row=by_seq.get(seq)
            if len(payload)!=size or row is None or stamp!=int(row['receive_t_ns']) or seq in raw_sequences:
                raise ValueError('Raw UDP record mismatch')
            raw_sequences.add(seq)
            if struct.unpack_from('<I',payload,4)[0]!=int(row['TimestampMS']):raise ValueError('Engine timestamp mismatch')
            if payload[315]!=int(row['Accel']) or struct.unpack_from('<b',payload,320)[0]!=int(row['Steer']):raise ValueError('Raw control mismatch')
            if 'IsRaceOn' in row and struct.unpack_from('<i',payload,0)[0]!=int(row['IsRaceOn']):
                raise ValueError('Raw game-state mismatch')
            position=struct.unpack_from('<3f',payload,244)
            if max(abs(v-float(row[key])) for v,key in zip(position,('PositionX','PositionY','PositionZ')))>.01:
                raise ValueError('Raw position mismatch')
            packets+=1
    if packets!=len(telemetry):raise ValueError('Raw UDP packet count mismatch')
    pts=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','packet=pts_time','-of','csv=p=0',str(folder/'video.mkv')],capture_output=True,text=True,check=True)
    times=np.array([float(line.split(',')[0]) for line in pts.stdout.splitlines() if line.strip()])
    declared=np.array([int(by_frame[i]['video_pts_us'])/1e6 for i in sorted(by_frame)])
    if len(times)!=len(frames) or np.max(abs(times-declared))>.0011:raise ValueError('Video PTS and frame table mismatch')
    verified_path=cache/(folder.name+'-aligned-verified.csv')
    temporary=verified_path.with_suffix('.new')
    with temporary.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(verified[0]));writer.writeheader();writer.writerows(verified)
    temporary.replace(verified_path)
    canonical=cache/(folder.name+'-normalized.csv')
    names=['timestamp','elapsed_sec','game_timestamp_ms','rpm','position_x','position_y','position_z','velocity_x','velocity_y','velocity_z','speed_kmh','throttle','brake','steer','gear']
    with canonical.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(names)
        for r in telemetry:
            receive=int(r['receive_t_ns'])/1e9
            writer.writerow([metadata['epoch_unix_ns_approx']/1e9+receive,receive,r['TimestampMS'],r['CurrentEngineRpm'],
                r['PositionX'],r['PositionY'],r['PositionZ'],r['VelocityX'],r['VelocityY'],r['VelocityZ'],r['speed_kmh'],r['throttle'],r['brake'],r['steer'],r['Gear']])
    statistics,_,_=inspect_csv(canonical)
    if statistics['nonfinite_rows'] or statistics['out_of_range_rows'] or statistics['valid_rows']<100:
        raise ValueError('Native telemetry quality is insufficient')
    fingerprint={name:[(folder/name).stat().st_size,(folder/name).stat().st_mtime_ns] for name in FILES}
    result.update(verdict='usable_for_path_candidates',path_candidate_eligible=True,normalized_csv=str(canonical.resolve()),
        statistics=statistics,media=inspect_video(folder/'video.mkv'),fingerprint=fingerprint,
        car_ordinals=sorted({int(r['CarOrdinal']) for r in telemetry
                             if int(r.get('IsRaceOn',1))==1 and int(r['CarOrdinal'])>0}),
        verified_alignment=str(verified_path.resolve()),
        synchronization=dict(verified=True,method='reconstructed nearest-past packet on recorded common QPC clock',
            video_pts_verified=True,raw_packets_verified=packets,matched_frames=len(ages),
            excluded_frames=excluded,corrected_alignment_rows=corrected,clock_offset_correction_s=0.,
            frame_clock_span_ms=float(np.ptp(np.asarray([int(r['frame_t_ns'])-int(r['video_pts_us'])*1000 for r in frames]))/1e6),
            median_age_ms=float(np.median(ages)),max_age_ms=float(np.max(ages)),
            input_time_is='user-space receive time, not exact engine or render latency'),
        notes=['Retain recording-specific car and camera provenance; camera parameters are initial estimates.',
               'No braking examples: do not teach a zero-brake policy.',
               'Exclude high tire slip from positive path demonstrations.'],session=metadata)
    result['media'].update(frames=len(frames),variable_frame_rate=True,
        measured_hz=(len(frames)-1)/((int(frames[-1]['frame_t_ns'])-int(frames[0]['frame_t_ns']))/1e9))
    if len(result['car_ordinals'])!=1:result.update(verdict='split_car_changes_required',path_candidate_eligible=False)
    return result


def selected_groups(align,telemetry,lo,hi,stride,minimum_dt=.18,target_hz=None,available_hz=None):
    """Never create a future-trajectory label across missing/menu/teleport data."""
    groups=[];group=[];last_time=-1.;previous=None;next_sample=None
    for r in align:
        t=int(r['video_pts_us'])/1e6
        if not lo<=t<hi or int(r['frame_index'])%stride:continue
        valid=r.get('valid')=='1'
        sample=telemetry.get(int(r['telemetry_seq'])) if valid else None
        if sample is not None:
            slip=max(abs(float(sample['TireSlipRatio'+w])) for w in ('FrontLeft','FrontRight','RearLeft','RearRight'))
            valid=(int(sample['IsRaceOn'])==1 and not int(sample['RacePosition']) and not int(sample['LapNumber'])
                   and float(sample['speed_kmh'])>=8 and slip<=.5)
        if not valid or sample is None:
            if group:groups.append(group)
            group=[];previous=None;last_time=-1.;next_sample=None
            continue
        if target_hz is None:
            if t-last_time<minimum_dt:continue
        else:
            # Use a clock grid, not a greedy minimum interval: at 15 fps a
            # 10 Hz subset must alternate one and two source-frame gaps.
            if next_sample is None:next_sample=t
            if t<next_sample-.5/available_hz-1e-8:continue
            next_sample=max(next_sample+1./target_hz,t+.5/target_hz)
        position=np.array([float(sample['Position'+a]) for a in 'XYZ'])
        if previous is not None:
            old_t,old_position,old_sample=previous
            dt=t-old_t
            jump=np.linalg.norm(position-old_position)>max(5.,dt*max(60.,float(sample['speed_kmh'])/3.6*1.5))
            reset=int(sample['TimestampMS'])<int(old_sample['TimestampMS'])
            if dt>.75 or jump or reset:
                if group:groups.append(group)
                group=[]
        group.append((r,sample));last_time=t;previous=t,position,sample
    if group:groups.append(group)
    return [g for g in groups if len(g)>=20]


def sampled_frames(source,folder,media,frame_count,stride):
    """Decode a deterministic subset; frame identity still comes from frames.csv."""
    legacy=folder/'video_frames'
    if legacy.is_dir() and len(list(legacy.glob('*.jpg')))==frame_count:
        return legacy,1,media['width'],media['height']
    width=min(1280,int(media['width']));height=round(media['height']*width/media['width']/2)*2
    cache=folder/f'sampled-v2-s{stride}-w{width}';cache.mkdir(parents=True,exist_ok=True)
    expected=(frame_count+stride-1)//stride
    if not (cache/'complete.json').exists():
        filters=f'select=not(mod(n\\,{stride})),scale={width}:{height}'
        errors=[]
        for hardware in (True,False):
            command=['ffmpeg','-y','-v','error','-threads','4']
            if hardware:command+=['-hwaccel','cuda']
            command+=['-i',str(source/'video.mkv'),'-an','-vf',filters,'-filter_threads','2',
                      '-fps_mode','vfr','-threads','4','-start_number','0','-q:v','4',str(cache/'%07d.jpg')]
            decoded=subprocess.run(command,capture_output=True,text=True)
            if decoded.returncode==0 and len(list(cache.glob('*.jpg')))==expected:break
            errors.append(decoded.stderr[-1000:])
        else:raise ValueError('Deterministic video decode failed: '+'; '.join(errors))
        from .forza_local_learning import write_json
        write_json(cache/'complete.json',dict(frame_count=frame_count,stride=stride,images=expected,width=width,height=height))
    if len(list(cache.glob('*.jpg')))!=expected:raise ValueError('Sampled video frame count mismatch')
    return cache,stride,width,height


def partition_ranges(duration,expand_training=False):
    valid_start=duration*.72
    train_end=valid_start-8. if expand_training else duration*.47
    if valid_start-train_end<8.-1e-8 or train_end<=3. or duration-1<=valid_start:
        raise ValueError('Recording too short for disjoint temporal holdout')
    return [('train',3.,train_end),('validation',valid_start,duration-1)]


def prepare(root,record,*,sample_hz=5.,expand_training=False):
    """Keep verified VFR identities and split every invalid temporal interval."""
    from .forza_local_learning import write_json
    root=Path(root);source=Path(record['path'])
    if sample_hz not in (5.,10.):raise ValueError('Supported preparation rates are 5 and 10 Hz')
    def unchanged():
        return {n:[(source/n).stat().st_size,(source/n).stat().st_mtime_ns] for n in FILES}==record['fingerprint']
    if not unchanged():raise ValueError('Source changed after synchronization audit')
    source_revision=hashlib.sha256(json.dumps(record['fingerprint'],sort_keys=True).encode()).hexdigest()[:16]
    variant='' if sample_hz==5. and not expand_training else f'-hz{sample_hz:g}-expanded{int(expand_training)}-clockgrid2'
    revision=hashlib.sha256((source_revision+f'-segments-{PREPARATION_VERSION}'+variant).encode()).hexdigest()[:16]
    folder=root/f'datasets/human/{source.name}_{source_revision}';manifest=folder/f'manifest{variant}.json'
    if manifest.exists():
        existing=json.loads(manifest.read_text(encoding='utf-8'))
        if existing.get('preparation_version')==PREPARATION_VERSION:return existing
    duration=record['media']['duration_s'];ranges=partition_ranges(duration,expand_training)
    frames=table(source/'frames.csv')
    align=table(record.get('verified_alignment',source/'aligned.csv'))
    telemetry={int(r['seq']):r for r in table(source/'telemetry.csv')}
    measured=float(record['media'].get('measured_hz',record['media']['fps']))
    stride=max(1,round(measured/sample_hz)) if sample_hz==5. else max(1,int(measured/sample_hz))
    timing={} if sample_hz==5. else {'target_hz':sample_hz,'available_hz':measured/stride}
    parts={name:selected_groups(align,telemetry,lo,hi,stride,minimum_dt=.9/sample_hz,**timing) for name,lo,hi in ranges}
    if not all(parts.values()):raise ValueError('Not enough contiguous nonslipping frames in both temporal partitions')
    rawframes,image_stride,width,height=sampled_frames(source,folder,record['media'],len(frames),stride)
    camera=dict(json.loads((root/'configs/forza_horizon6.json').read_text(encoding='utf-8'))['camera'])
    camera.update(width=width,height=height,name='Native recorder: initial camera estimate, retained car provenance')
    if record['car_ordinals']==[4222]:camera.update(right_m=.35,name='Reviewed car 4222: right-side cockpit, initial camera estimate')
    result=dict(recording=source.name,revision=revision,preparation_version=PREPARATION_VERSION,sample_hz=sample_hz,
                source_duration_s=duration,training_range=list(ranges[0][1:]),validation_range=list(ranges[1][1:]),
                train=[],validation=[],frames=0,car_ordinals=record['car_ordinals'],synchronization=record['synchronization'])
    for partition,groups in parts.items():
        for number,group in enumerate(groups):
            target=folder/f'segments-v2{variant}/{partition}_{number:03d}';(target/'frames').mkdir(parents=True,exist_ok=True)
            with (target/'poses.csv').open('w',newline='',encoding='utf-8') as f,(target/'actions.csv').open('w',newline='',encoding='utf-8') as af:
                writer=csv.writer(f);writer.writerow(['frame','t','x','y','z','yaw_deg','pitch_deg','kmh'])
                actions=csv.writer(af);actions.writerow(['frame','video_pts_s','telemetry_seq','age_ms','throttle','brake','applied_steer'])
                for count,(r,sample) in enumerate(group):
                    destination=target/'frames'/f'{count:07d}.jpg'
                    if not destination.exists():destination.hardlink_to(rawframes/f"{int(r['frame_index'])//image_stride:07d}.jpg")
                    writer.writerow([count,int(r['frame_t_ns'])/1e9,sample['PositionX'],sample['PositionZ'],sample['PositionY'],
                                     90-np.degrees(float(sample['Yaw'])),-np.degrees(float(sample['Pitch'])),sample['speed_kmh']])
                    actions.writerow([count,int(r['video_pts_us'])/1e6,sample['seq'],r['telemetry_age_ms'],sample['throttle'],sample['brake'],sample['steer']])
            write_json(target/'camera.json',camera)
            video_range=[int(group[0][0]['video_pts_us'])/1e6,int(group[-1][0]['video_pts_us'])/1e6]
            write_json(target/'provenance.json',dict(recording=source.name,partition=partition,video_range=video_range,
                camera_estimated=True,control_labels='clock-verified Data Out state; no interpolation across gaps',
                source_alignment=str(record.get('verified_alignment',source/'aligned.csv')),car_ordinals=record['car_ordinals']))
            result[partition].append(str(target.resolve()));result['frames']+=len(group)
    if not unchanged():raise ValueError('Source changed during video preparation')
    write_json(manifest,result)
    return result
