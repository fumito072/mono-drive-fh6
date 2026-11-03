"""Encode ordered, clock-verified human frames without crossing recording gaps."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv,hashlib,json,os
from pathlib import Path
import sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from mono_drive.camera import Camera,REF_CAMERA,ReferenceView
from mono_drive.pathnet import net_input_u8
from mono_drive.temporal_steering import HISTORY,FUTURE,recording_split,window_indices

def write(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    for i in range(30):
        try:tmp.replace(p);return
        except PermissionError:time.sleep(.1)
    raise OSError('Cannot publish '+str(p))

def main():
    import cv2,torch
    from torch_models import build_pathnet,load_onnx_into
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT/'out/human-dataset.json')
    p.add_argument('--out',type=Path,default=ROOT/'out/temporal-human');p.add_argument('--base',type=Path,default=ROOT/'models/pathnet.onnx')
    p.add_argument('--batch',type=int,default=256);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    base=build_pathnet();load_onnx_into(base,str(a.base))
    encoder=torch.nn.Sequential(base.body,*list(base.head.children())[:3]).eval().to(device)
    encoder_hash=hashlib.sha256(a.base.read_bytes()).hexdigest()
    export=a.out/'encoder.onnx'
    marker=a.out/'encoder-source.json'
    try:exported_hash=json.loads(marker.read_text(encoding='utf-8')).get('base_sha256')
    except (OSError,ValueError):exported_hash=None
    if not export.exists() or exported_hash!=encoder_hash:
        torch.onnx.export(encoder.cpu(),torch.zeros(1,3,128,320),str(export),input_names=['image'],output_names=['features'],
                          dynamic_axes={'image':{0:'batch'},'features':{0:'batch'}},opset_version=17,dynamo=False)
        write(marker,{'base_sha256':encoder_hash})
        encoder.to(device)
    source=json.loads(a.manifest.read_text(encoding='utf-8'));result={'revision':source['revision'],'base_sha256':encoder_hash,
        'encoder':str(export.resolve()),'source':'clock-verified WRX Data Out','segments':[],'excluded':[]}
    sample_hz=float(source.get('sample_hz',5.));stride=round(sample_hz/5.)
    if sample_hz not in (5.,10.):raise ValueError('Unsupported source sampling rate')
    result.update(source_sample_hz=sample_hz,temporal_stride=stride)
    start=time.time()
    with ThreadPoolExecutor(max_workers=max(1,min(8,a.workers))) as pool:
        for number,episode in enumerate(source['episodes']):
            if episode.get('car_ordinals')!=[3819] or not episode.get('synchronization',{}).get('verified'):
                result['excluded'].append({'recording':episode['recording'],'reason':'requires clock-verified WRX'});continue
            for partition in ('train','validation'):
                for folder in episode[partition]:
                    folder=Path(folder);identifier=hashlib.sha256(str(folder).encode()).hexdigest()[:20]
                    target=a.out/(identifier+'.npz')
                    with (folder/'actions.csv').open(encoding='utf-8',newline='') as f:actions=list(csv.DictReader(f))
                    with (folder/'poses.csv').open(encoding='utf-8',newline='') as f:poses=list(csv.DictReader(f))
                    images=sorted((folder/'frames').glob('*.jpg'))
                    if not len(images)==len(actions)==len(poses):raise ValueError('Image/action/pose counts differ')
                    n=len(images);times=np.array([float(r['video_pts_s']) for r in actions]);speed=np.array([float(r['kmh']) for r in poses],np.float32)
                    steer=np.array([float(r['applied_steer']) for r in actions],np.float32);ages=np.array([float(r['age_ms']) for r in actions])
                    indices=window_indices(times,speed,steer,ages,history=HISTORY*stride,future=FUTURE*stride,sample_hz=sample_hz)
                    if len(indices)<20:continue
                    signature=hashlib.sha256(b''.join((folder/name).read_bytes() for name in ('camera.json','poses.csv','actions.csv'))).hexdigest()
                    if stride!=1:signature=hashlib.sha256((signature+f'-stride-{stride}').encode()).hexdigest()
                    reusable=False
                    if target.exists():
                        with np.load(target) as existing:
                            reusable=(str(existing['base_sha256'])==encoder_hash and
                                      'source_signature' in existing and str(existing['source_signature'])==signature)
                    if not reusable:
                        first=cv2.imread(str(images[0]));camera=Camera.load(str(folder/'camera.json'))
                        view=ReferenceView(camera,REF_CAMERA,frame_size=(first.shape[1],first.shape[0]))
                        def decode(path):
                            image=cv2.imread(str(path))
                            if image is None:raise ValueError('Missing video frame')
                            return net_input_u8(view.warp(image))
                        features=np.empty((n,256),np.float16)
                        for begin in range(0,n,a.batch):
                            x=np.stack(list(pool.map(decode,images[begin:begin+a.batch]))).astype(np.float32)/255.
                            with torch.inference_mode():features[begin:begin+len(x)]=encoder(torch.from_numpy(x).to(device)).cpu().numpy()
                        if not np.isfinite(features).all():raise ValueError('Nonfinite visual embeddings')
                        temporary=target.with_suffix('.tmp.npz')
                        np.savez(temporary,features=features,times=times,speed=speed,steer=steer,ages=ages,indices=indices,
                                 base_sha256=np.array(encoder_hash),source_signature=np.array(signature));temporary.replace(target)
                    split=recording_split(episode['recording']) or partition
                    result['segments'].append({'file':str(target.resolve()),'source':str(folder),'recording':episode['recording'],
                                               'split':split,'frames':n,'windows':len(indices)})
            write(a.out/'state.json',{'phase':'encoding','episodes_done':number+1,'total_episodes':len(source['episodes']),
                                    'segments':len(result['segments']),'elapsed_s':time.time()-start,'time':time.time()})
            print(json.dumps({'encoded_episode':number+1,'segments':len(result['segments']),'seconds':round(time.time()-start)}),flush=True)
    result['finished_at']=time.time();write(a.out/'manifest.json',result)
    write(a.out/'state.json',{'phase':'complete','segments':len(result['segments']),'time':time.time()})
    print('Temporal feature preparation complete',flush=True)

if __name__=='__main__':main()
