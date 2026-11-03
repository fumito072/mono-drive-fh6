"""Adapt the image encoder to human driving, independently of the held-out test."""
import argparse,copy,csv,hashlib,json,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from mono_drive.camera import Camera,ReferenceView,REF_CAMERA
from mono_drive.pathnet import net_input_u8
from mono_drive.temporal_steering import HISTORY,FUTURE,recording_split,window_indices
from mono_drive.visual_learning import visual_targets,visual_scores,accept_visual
from mono_drive.forza_local_learning import write_json


def main():
    import cv2,torch
    from torch import nn
    from torch_models import build_pathnet,load_onnx_into
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT/'out/human-dataset.json')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--samples-per-epoch',type=int,default=24000);p.add_argument('--batch',type=int,default=32)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if not 1<=a.epochs<=50 or not 1000<=a.samples_per_epoch<=200000 or not 1<=a.batch<=256:raise ValueError('Invalid training budget')
    source=json.loads(a.manifest.read_text(encoding='utf-8'));source_revision=source.get('source_revision',source['revision'])
    base_path=ROOT/'models/pathnet.onnx';base_hash=hashlib.sha256(base_path.read_bytes()).hexdigest()
    if (a.out/'report.json').exists():
        previous=json.loads((a.out/'report.json').read_text(encoding='utf-8'))
        if previous.get('phase')=='complete' and previous.get('source_revision')==source_revision and previous.get('base_sha256')==base_hash:
            print('Reusing completed visual comparison',flush=True);return
    write_json(a.out/'progress.json',{'phase':'loading','source_revision':source_revision,'time':time.time()})
    cv2.setNumThreads(1);torch.set_num_threads(4);torch.manual_seed(83);rng=np.random.default_rng(83)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    hz=float(source.get('sample_hz',5.));stride=round(hz/5.)
    examples={'train':[],'validation':[]};views={};excluded_test=[]
    for episode in source['episodes']:
        if episode.get('car_ordinals')!=[3819] or not episode.get('synchronization',{}).get('verified'):continue
        if recording_split(episode['recording'])=='test':excluded_test.append(episode['recording']);continue
        for partition in examples:
            for directory in episode[partition]:
                directory=Path(directory)
                with (directory/'actions.csv').open(encoding='utf-8',newline='') as f:actions=list(csv.DictReader(f))
                with (directory/'poses.csv').open(encoding='utf-8',newline='') as f:poses=list(csv.DictReader(f))
                images=sorted((directory/'frames').glob('*.jpg'))
                if len(images)!=len(actions) or len(images)!=len(poses):raise ValueError('Frame/label mismatch')
                times=np.asarray([float(x['video_pts_s']) for x in actions]);steer=np.asarray([float(x['applied_steer']) for x in actions])
                age=np.asarray([float(x['age_ms']) for x in actions]);speed=np.asarray([float(x['kmh']) for x in poses])
                yaw=np.asarray([float(x['yaw_deg']) for x in poses])
                indices=window_indices(times,speed,steer,age,history=HISTORY*stride,future=FUTURE*stride,sample_hz=hz)
                for index in indices:
                    label=visual_targets(steer,yaw,int(index),stride)
                    if np.isfinite(label).all():examples[partition].append((images[index],directory,float(speed[index])/100.,label))
                if len(indices):
                    image=cv2.imread(str(images[indices[0]]))
                    views[directory]=ReferenceView(Camera.load(str(directory/'camera.json')),REF_CAMERA,frame_size=(image.shape[1],image.shape[0]))
    if min(map(len,examples.values()))<500:raise ValueError('Insufficient separated visual examples')
    validation=examples['validation']
    if len(validation)>6000:validation=[validation[i] for i in np.linspace(0,len(validation)-1,6000,dtype=int)]
    weights=np.asarray([3. if abs(x[3][0])>.04 else 1. for x in examples['train']]);weights/=weights.sum()
    base=build_pathnet();load_onnx_into(base,str(base_path))
    original=nn.Sequential(base.body,*list(base.head.children())[:3]).eval()
    frozen=copy.deepcopy(original).to(device).requires_grad_(False)
    adapted=copy.deepcopy(original).to(device).requires_grad_(False)
    # Train the last two visual stages and projection, retaining low-level edges.
    for layer in list(adapted[0].children())[10:]:layer.requires_grad_(True)
    adapted[2].requires_grad_(True)
    def head():return nn.Sequential(nn.LayerNorm(257),nn.Linear(257,128),nn.GELU(),nn.Linear(128,2),nn.Tanh()).to(device)
    heads=[head(),head()];heads[1].load_state_dict(heads[0].state_dict())
    opt=[torch.optim.AdamW([{'params':[x for x in adapted.parameters() if x.requires_grad],'lr':2e-5},
                           {'params':heads[0].parameters(),'lr':4e-4}],weight_decay=.01),
         torch.optim.AdamW(heads[1].parameters(),lr=4e-4,weight_decay=.01)]
    schedulers=[torch.optim.lr_scheduler.CosineAnnealingLR(o,a.epochs) for o in opt]
    best=[float('inf'),float('inf')];saved=[None,None];history=[]
    def decode(item):
        path,directory,speed,label=item;image=cv2.imread(str(path))
        if image is None:raise ValueError('Missing teacher image')
        return net_input_u8(views[directory].warp(image)),speed,label
    with ThreadPoolExecutor(max_workers=6) as pool:
        def batch(items,augment=False):
            decoded=list(pool.map(decode,items));x=np.stack([v[0] for v in decoded]).astype(np.float32)/255.
            if augment:x=np.clip(x*rng.uniform(.9,1.1,(len(items),1,1,1)).astype(np.float32),0,1)
            return (torch.from_numpy(x).to(device),torch.tensor([v[1] for v in decoded],device=device).view(-1,1),
                    torch.from_numpy(np.stack([v[2] for v in decoded])).to(device))
        def assess(encoder,h):
            encoder.eval();h.eval();pred=[];truth=[]
            with torch.inference_mode():
                for start in range(0,len(validation),a.batch):
                    x,s,y=batch(validation[start:start+a.batch]);pred.extend(h(torch.cat((encoder(x),s),1)).cpu().numpy());truth.extend(y.cpu().numpy())
            return visual_scores(pred,truth)
        for epoch in range(a.epochs):
            adapted.train();heads[0].train();heads[1].train();losses=np.zeros(2);steps=0
            selection=rng.choice(len(examples['train']),a.samples_per_epoch,replace=True,p=weights)
            for start in range(0,len(selection),a.batch):
                if (ROOT/'out/training.stop').exists() or (ROOT/'out/longrun.stop').exists():raise SystemExit('Explicit stop')
                x,s,y=batch([examples['train'][i] for i in selection[start:start+a.batch]],True)
                with torch.no_grad():fixed=frozen(x)
                for n,feature in enumerate((adapted(x),fixed)):
                    prediction=heads[n](torch.cat((feature,s),1))
                    loss=nn.functional.smooth_l1_loss(prediction[:,0],y[:,0],beta=.05)+.25*nn.functional.smooth_l1_loss(prediction[:,1],y[:,1],beta=.1)
                    opt[n].zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(heads[n].parameters())+(list(adapted.parameters()) if n==0 else []),2.);opt[n].step()
                    losses[n]+=float(loss.detach())
                steps+=1
                if steps%100==0:write_json(a.out/'progress.json',{'phase':'training','epoch':epoch+1,'epochs':a.epochs,'step':steps,'steps':int(np.ceil(len(selection)/a.batch)),'time':time.time()})
            scores=[assess(adapted,heads[0]),assess(frozen,heads[1])]
            for n,score in enumerate(scores):
                objective=score['steer_mae']+.25*score['heading_mae']
                if objective<best[n]:
                    best[n]=objective;saved[n]={'encoder':copy.deepcopy((adapted if n==0 else frozen).cpu().state_dict()),'head':copy.deepcopy(heads[n].cpu().state_dict()),'score':score}
                    (adapted if n==0 else frozen).to(device);heads[n].to(device)
                schedulers[n].step()
            record={'epoch':epoch+1,'loss':(losses/steps).tolist(),'validation':scores};history.append(record)
            write_json(a.out/'progress.json',dict(phase='validation',time=time.time(),**record));print(json.dumps(record),flush=True)
    accepted=accept_visual(saved[0]['score'],saved[1]['score'])
    selected=original.cpu().eval()
    if accepted:selected.load_state_dict(saved[0]['encoder'])
    torch.save(selected.state_dict(),a.out/'encoder-state.pt')
    torch.onnx.export(selected,torch.zeros(1,3,128,320),str(a.out/'encoder.onnx'),input_names=['image'],output_names=['features'],
                      dynamic_axes={'image':{0:'batch'},'features':{0:'batch'}},opset_version=17,dynamo=False)
    import onnxruntime as ort
    sample=np.random.default_rng(2).random((1,3,128,320),dtype=np.float32)
    reference=selected(torch.from_numpy(sample)).detach().numpy()
    got=ort.InferenceSession(str(a.out/'encoder.onnx'),providers=['CPUExecutionProvider']).run(None,{'image':sample})[0]
    difference=float(np.max(np.abs(got-reference)))
    if not np.isfinite(got).all() or difference>1e-4:raise ValueError('Visual ONNX export mismatch')
    report={'phase':'complete','accepted_visual':accepted,'source_revision':source_revision,'dataset_revision':source['revision'],
            'base_sha256':base_hash,'samples':{k:len(v) for k,v in examples.items()},'validation_used':len(validation),
            'excluded_test_recordings':len(set(excluded_test)),'comparison':['fine_tuned_visual_encoder','frozen_visual_encoder'],
            'validation':[x['score'] for x in saved],'epochs':a.epochs,'samples_per_epoch':a.samples_per_epoch,
            'targets':['future_mean_Data_Out_steer','future_heading_change'],'history':history,'onnx_max_difference':difference,
            'selected':'adapted' if accepted else 'original','time':time.time(),
            'note':'Validation selects visual features only. Separate recording-level temporal and live checks remain mandatory.'}
    write_json(a.out/'report.json',report);write_json(a.out/'progress.json',{'phase':'complete','accepted_visual':accepted,'time':time.time()})
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
