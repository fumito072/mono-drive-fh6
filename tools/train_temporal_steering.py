"""Train causal video steering and compare with an equally trained single-frame head."""
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mono_drive.temporal_steering import HISTORY,targets,sequence_arrays,build_models,metrics,accept


def write(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    for i in range(30):
        try:tmp.replace(p);return
        except PermissionError:time.sleep(.1)
    raise OSError('Could not save report')


def main():
    import torch
    from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT/'out/temporal-human')
    p.add_argument('--out',type=Path,default=ROOT/'models/temporal-20261103')
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--samples-per-epoch',type=int,default=80000)
    p.add_argument('--wait',action='store_true');a=p.parse_args()
    if not 1<=a.epochs<=100 or not 1000<=a.samples_per_epoch<=1000000:raise ValueError('Invalid training budget')
    while not (a.cache/'manifest.json').exists():
        if not a.wait:raise FileNotFoundError('Prepare temporal features first')
        time.sleep(3)
    manifest=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8'));a.out.mkdir(parents=True,exist_ok=True)
    stride=int(manifest.get('temporal_stride',1))
    write(a.out/'training-progress.json',{'phase':'loading','architecture_version':2,'time':time.time()})
    torch.manual_seed(27);np.random.seed(27);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    segments=[];split={k:[] for k in ('train','validation','test')}
    for item in manifest['segments']:
        with np.load(item['file']) as x:
            row={k:x[k].copy() for k in ('features','speed','steer','indices')}
        s=len(segments);segments.append(row)
        split[item['split']].extend((s,int(i)) for i in row['indices'])
    if min(len(v) for v in split.values())<100:raise ValueError('Not enough independent temporal sequences')
    train_features=np.concatenate([s['features'] for s,item in zip(segments,manifest['segments']) if item['split']=='train'])
    mean=train_features.astype(np.float32).mean(0);std=train_features.astype(np.float32).std(0);del train_features
    class Sequences(Dataset):
        def __init__(self,name):self.index=split[name]
        def __len__(self):return len(self.index)
        def __getitem__(self,n):
            s,i=self.index[n];row=segments[s]
            return sequence_arrays(row['features'],row['speed'],row['steer'],i,stride)
    datasets={k:Sequences(k) for k in split}
    labels=np.array([float(targets(segments[s]['steer'],i,stride)[-1]) for s,i in split['train']])
    weights=np.where(np.abs(labels)>.04,3.,1.)
    sampler=WeightedRandomSampler(torch.from_numpy(weights),min(a.samples_per_epoch,len(weights)),replacement=True)
    train=DataLoader(datasets['train'],batch_size=256,sampler=sampler,num_workers=0)
    models=list(build_models(mean,std));models=[m.to(device) for m in models]
    optimizers=[torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=.01) for m in models]
    schedulers=[torch.optim.lr_scheduler.CosineAnnealingLR(o,a.epochs) for o in optimizers]
    best=[math.inf,math.inf];states=[None,None];history=[]
    def predict(model,name,ablate=False):
        model.eval();results=[];truth=[];prior=[]
        with torch.inference_mode():
            for f,e,y in DataLoader(datasets[name],batch_size=512,num_workers=0):
                if ablate:
                    f=f[:,-1:,:].expand_as(f);e=e[:,-1:,:].expand_as(e)
                results.extend(model(f.to(device),e.to(device))[:,-1].cpu().numpy())
                truth.extend(y[:,-1].numpy());prior.extend(e[:,-1,1].numpy())
        ids=[];group=0;last=None
        for s,i in split[name]:
            if last is None or s!=last[0] or i!=last[1]+1:group+=1
            ids.append(group);last=(s,i)
        return np.array(results),np.array(truth),np.array(prior),np.array(ids)
    for epoch in range(a.epochs):
        if (ROOT/'out/training.stop').exists() or (ROOT/'out/longrun.stop').exists():raise SystemExit('Explicit stop')
        for m in models:m.train()
        total=np.zeros(2);batches=0
        for f,e,y in train:
            f,e,y=(v.to(device) for v in (f,e,y));batches+=1
            for n,(model,opt) in enumerate(zip(models,optimizers)):
                prediction=model(f,e)[:,-2:]
                loss=torch.nn.functional.smooth_l1_loss(prediction,y,beta=.05)
                loss+=.35*torch.nn.functional.smooth_l1_loss(prediction[:,1]-prediction[:,0],y[:,1]-y[:,0],beta=.025)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
                total[n]+=float(loss.detach())
        record={'epoch':epoch+1,'loss':(total/max(1,batches)).tolist(),'validation':[]}
        for n,m in enumerate(models):
            scores=metrics(*predict(m,'validation'));record['validation'].append(scores)
            criterion=scores['mae']+.5*scores['active_mae']+.25*scores['delta_mae']
            if criterion<best[n]:
                best[n]=criterion;states[n]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            schedulers[n].step()
        history.append(record);write(a.out/'training-progress.json',{'phase':'training','epoch':epoch+1,'epochs':a.epochs,'last':record})
        print(json.dumps(record),flush=True)
    test=[];predictions=[]
    for n,m in enumerate(models):
        m.load_state_dict(states[n]);values=predict(m,'test');predictions.append(values);test.append(metrics(*values))
    ablated=metrics(*predict(models[0],'test',ablate=True))
    permitted=accept(test[0],test[1]) and test[0]['active_mae']<=ablated['active_mae']*.99
    report={'phase':'complete','architecture_version':2,'accepted_offline':permitted,'created_at':time.time(),'encoder':str((a.cache/'encoder.onnx').resolve()),
            'dataset_revision':manifest['revision'],'history_frames':HISTORY,'sample_hz':5,'target':'mean applied steer over next 0.2-0.6 seconds',
            'source_sample_hz':manifest.get('source_sample_hz',5.),'temporal_stride':stride,
            'epochs':a.epochs,'samples_per_epoch':min(a.samples_per_epoch,len(weights)),
            'labels':'Clock-verified Data Out state, not raw hardware commands','samples':{k:len(v) for k,v in split.items()},
            'test_temporal':test[0],'test_single_frame':test[1],'history':history,
            'test_without_real_history':ablated,'parameter_counts':[sum(p.numel() for p in m.parameters()) for m in models],
            'deployment':'candidate only; requires runtime shadow and bounded trial'}
    class Export(torch.nn.Module):
        def __init__(self,m):super().__init__();self.m=m
        def forward(self,features,ego):return self.m(features,ego)[:,-1:]
    wrapped=Export(models[0].cpu().eval());f,e,_=datasets['test'][0]
    torch.onnx.export(wrapped,(torch.from_numpy(f[None]),torch.from_numpy(e[None])),str(a.out/'steering.onnx'),
                      input_names=['features','ego'],output_names=['steer'],opset_version=17,dynamo=False)
    import onnxruntime as ort
    session=ort.InferenceSession(str(a.out/'steering.onnx'),providers=['CPUExecutionProvider'])
    got=session.run(None,{'features':f[None],'ego':e[None]})[0]
    ref=wrapped(torch.from_numpy(f[None]),torch.from_numpy(e[None])).detach().numpy()
    report['onnx_max_difference']=float(np.max(np.abs(got-ref)))
    if report['onnx_max_difference']>1e-4:raise ValueError('ONNX output mismatch')
    torch.save(states[0],a.out/'steering-state.pt')
    import shutil
    shutil.copy2(a.cache/'encoder.onnx',a.out/'encoder.onnx')
    write(a.out/'report.json',report)
    np.savez(a.out/'test-predictions.npz',temporal=predictions[0][0],single=predictions[1][0],target=predictions[0][1],previous=predictions[0][2],segments=predictions[0][3])
    write(a.out/'training-progress.json',{'phase':'complete','accepted_offline':permitted,'test_temporal':test[0],'test_single_frame':test[1]})
    print(json.dumps(report),flush=True)
    import subprocess
    clip=ROOT/'out/temporal-report'/a.out.name
    try:
        subprocess.run([sys.executable,str(ROOT/'tools/render_temporal_report.py'),'--model-dir',str(a.out),
                        '--cache',str(a.cache),'--out',str(clip)],cwd=ROOT,check=True)
    except (OSError,subprocess.CalledProcessError) as error:
        write(a.out/'video-report-error.json',{'error':str(error),'model_still_validated':True})

if __name__=='__main__':main()
