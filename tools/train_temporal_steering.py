"""Train causal video steering with explicit paired-baseline schedules."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mono_drive.temporal_steering import HISTORY,targets,sequence_arrays,build_models,metrics,accept,jitter_inputs,extended_indices,forecast_targets,curriculum_weights


def write(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    for i in range(30):
        try:tmp.replace(p);return
        except PermissionError:time.sleep(.1)
    raise OSError('Could not save report')


def export_steering_model(model,features,ego,path):
    """Export an evaluation-mode tree; the wrapper must also be in eval mode."""
    import torch
    import onnxruntime as ort
    class Export(torch.nn.Module):
        def __init__(self,m):super().__init__();self.m=m
        def forward(self,features,ego):return self.m(features,ego)[:,-1:]
    # Export restores the wrapper's original mode recursively. An unconfigured
    # wrapper defaults to training and would re-enable dropout for comparison.
    wrapped=Export(model.cpu()).eval()
    inputs=(torch.from_numpy(features),torch.from_numpy(ego))
    torch.onnx.export(wrapped,inputs,str(path),input_names=['features','ego'],
                      output_names=['steer'],opset_version=17,dynamo=False)
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    got=session.run(None,{'features':features,'ego':ego})[0]
    with torch.inference_mode():reference=wrapped(*inputs).numpy()
    difference=float(np.max(np.abs(got-reference)))
    if not np.isfinite(got).all() or not np.isfinite(reference).all() or difference>1e-4:
        raise ValueError(f'ONNX output mismatch: {difference}')
    return difference


def main():
    import torch
    from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT/'out/temporal-human')
    p.add_argument('--out',type=Path,default=ROOT/'models/temporal-20261103')
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--samples-per-epoch',type=int,default=80000)
    p.add_argument('--variant',choices=['base','residual','deep'],default='base')
    p.add_argument('--patience',type=int,default=0,help='Validation epochs without improvement before stopping; 0 disables')
    p.add_argument('--resume',action='store_true',help='Resume a matching local training checkpoint')
    p.add_argument('--reference-model-dir',type=Path,help='Also require improvement over this previously accepted model')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--wait',action='store_true');a=p.parse_args()
    if not 1<=a.epochs<=100 or not 1000<=a.samples_per_epoch<=1000000:raise ValueError('Invalid training budget')
    staged=a.variant in ('residual','deep');history_frames=21 if a.variant=='deep' else HISTORY
    if staged and a.epochs<3:raise ValueError('Residual training needs baseline warmup and correction epochs')
    if not 0<=a.patience<=30:raise ValueError('Invalid validation patience')
    if not 16<=a.batch_size<=512:raise ValueError('Invalid batch size')
    while not (a.cache/'manifest.json').exists():
        if not a.wait:raise FileNotFoundError('Prepare temporal features first')
        time.sleep(3)
    manifest=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8'));a.out.mkdir(parents=True,exist_ok=True)
    if a.resume and (a.out/'report.json').exists():
        print('Completed model retained; choose a new output directory for another experiment',flush=True);return
    stride=int(manifest.get('temporal_stride',1))
    reference_report=None;reference_hash=None
    if a.reference_model_dir:
        reference_report=json.loads((a.reference_model_dir/'report.json').read_text(encoding='utf-8'))
        if not reference_report.get('accepted_offline') or reference_report.get('encoder_source_fingerprint')!=manifest.get('base_sha256'):
            raise ValueError('Reference must be accepted and use the exact same visual feature encoder')
        reference_hash=hashlib.sha256((a.reference_model_dir/'steering-state.pt').read_bytes()).hexdigest()
    version=4 if a.variant=='deep' else 3 if staged else 2
    write(a.out/'training-progress.json',{'phase':'loading','architecture_version':version,'time':time.time()})
    torch.manual_seed(27);np.random.seed(27);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    segments=[];split={k:[] for k in ('train','validation','test')}
    for number,item in enumerate(manifest['segments']):
        with np.load(item['file']) as x:
            row={k:x[k].copy() for k in ('features','speed','steer','indices')}
        if a.variant=='deep':row['indices']=extended_indices(row['indices'],len(row['steer']),history_frames,stride)
        s=len(segments);segments.append(row)
        split[item['split']].extend((s,int(i)) for i in row['indices'])
        if number%50==0:write(a.out/'training-progress.json',{'phase':'loading','architecture_version':version,
            'segments_loaded':number+1,'segments':len(manifest['segments']),'time':time.time()})
    if min(len(v) for v in split.values())<100:raise ValueError('Not enough independent temporal sequences')
    train_features=np.concatenate([s['features'] for s,item in zip(segments,manifest['segments']) if item['split']=='train'])
    mean=train_features.astype(np.float32).mean(0);std=train_features.astype(np.float32).std(0);del train_features
    class Sequences(Dataset):
        def __init__(self,name):self.index=split[name]
        def __len__(self):return len(self.index)
        def __getitem__(self,n):
            s,i=self.index[n];row=segments[s]
            f,e,y=sequence_arrays(row['features'],row['speed'],row['steer'],i,stride,history_frames)
            if a.variant=='deep':y=forecast_targets(row['steer'],i,stride)
            return f,e,y
    datasets={k:Sequences(k) for k in split}
    labels=np.array([targets(segments[s]['steer'],i,stride) for s,i in split['train']])
    weights=curriculum_weights(labels) if a.variant=='deep' else np.where(np.abs(labels[:,-1])>.04,3.,1.)
    sampler=WeightedRandomSampler(torch.from_numpy(weights),min(a.samples_per_epoch,len(weights)),replacement=True)
    print(json.dumps({'phase':'ready','variant':a.variant,'history_frames':history_frames,
        'samples':{k:len(v) for k,v in split.items()},'batch_size':a.batch_size,'maximum_epochs':a.epochs}),flush=True)
    train=DataLoader(datasets['train'],batch_size=a.batch_size,sampler=sampler,num_workers=0)
    models=list(build_models(mean,std,a.variant));models=[m.to(device) for m in models]
    warm_started=False
    if a.variant=='deep' and reference_report is not None:
        prior_state=torch.load(a.reference_model_dir/'steering-state.pt',map_location=device,weights_only=True)
        prefix='current.' if any(k.startswith('current.') for k in prior_state) else 'head.'
        head_state={k[len(prefix):]:v for k,v in prior_state.items() if k.startswith(prefix)}
        models[1].head.load_state_dict(head_state);models[0].current.load_state_dict(head_state)
        with torch.no_grad():
            for model in models:
                model.mean.copy_(prior_state['mean']);model.scale.copy_(prior_state['scale'])
        warm_started=True
    warmup=min(8,max(1,a.epochs//3))
    initial_temporal={k:v.detach().cpu().clone() for k,v in models[0].state_dict().items()}
    optimizers=[torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=.01) for m in models]
    if warm_started:
        for group in optimizers[1].param_groups:group['lr']=2e-4
    schedulers=[torch.optim.lr_scheduler.CosineAnnealingLR(o,a.epochs) for o in optimizers]
    best=[math.inf,math.inf];states=[None,None];history=[];best_epoch=[0,0];start_epoch=0
    checkpoint=a.out/'checkpoint.pt'
    signature={'architecture_version':version,'variant':a.variant,'dataset_revision':manifest['revision'],
               'encoder_sha256':manifest.get('base_sha256'),'history_frames':history_frames,'stride':stride,'epochs':a.epochs,
               'samples_per_epoch':a.samples_per_epoch,'patience':a.patience,'reference_sha256':reference_hash,'batch_size':a.batch_size}
    def correction_optimizer():
        optimizer=torch.optim.AdamW([v for v in models[0].parameters() if v.requires_grad],
            lr=3e-4 if a.variant=='deep' else 8e-4,weight_decay=.02 if a.variant=='deep' else .01)
        return optimizer,torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,a.epochs-warmup))
    if a.resume and checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if saved.get('signature')!=signature:raise ValueError('Checkpoint settings or data differ')
        start_epoch=saved['epoch'];best=saved['best'];states=saved['best_states'];history=saved['history'];best_epoch=saved['best_epoch']
        for model,state in zip(models,saved['models']):model.load_state_dict(state)
        if staged and start_epoch>warmup:
            models[1].requires_grad_(False);models[0].current.requires_grad_(False)
            optimizers[0],schedulers[0]=correction_optimizer()
        for optimizer,state in zip(optimizers,saved['optimizers']):optimizer.load_state_dict(state)
        for scheduler,state in zip(schedulers,saved['schedulers']):scheduler.load_state_dict(state)
        torch.set_rng_state(saved['torch_rng'].cpu());np.random.set_state(saved['numpy_rng'])
        if device=='cuda' and saved.get('cuda_rng') is not None:torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        print(json.dumps({'resumed_epoch':start_epoch,'variant':a.variant}),flush=True)
    def predict(model,name,ablate=False,jitter=False,history_limit=None):
        model.eval();results=[];truth=[];prior=[]
        with torch.inference_mode():
            for f,e,y in DataLoader(datasets[name],batch_size=512,num_workers=0):
                if history_limit is not None:f=f[:,-history_limit:];e=e[:,-history_limit:]
                if ablate:
                    f=f[:,-1:,:].expand_as(f);e=e[:,-1:,:].expand_as(e)
                if jitter:f,e=jitter_inputs(f,e,all_rows=True)
                results.extend(model(f.to(device),e.to(device))[:,-1].cpu().numpy())
                truth.extend(y[:,1].numpy());prior.extend(e[:,-1,1].numpy())
        ids=[];group=0;last=None
        for s,i in split[name]:
            if last is None or s!=last[0] or i!=last[1]+1:group+=1
            ids.append(group);last=(s,i)
        return np.array(results),np.array(truth),np.array(prior),np.array(ids)
    patience_reached=(a.patience and start_epoch>=warmup+8 and start_epoch-best_epoch[0]>=a.patience)
    stop_reason='validation_patience' if patience_reached else 'epoch_budget'
    for epoch in range(start_epoch,start_epoch if patience_reached else a.epochs):
        if (ROOT/'out/training.stop').exists() or (ROOT/'out/longrun.stop').exists():raise SystemExit('Explicit stop')
        write(a.out/'training-progress.json',{'phase':'training','epoch':epoch+1,'epochs':a.epochs,'batches':0,'time':time.time()})
        for m in models:m.train()
        if staged and epoch==warmup:
            models[1].load_state_dict(states[1]);models[1].requires_grad_(False)
            models[0].load_state_dict(initial_temporal)
            models[0].current.load_state_dict(models[1].head.state_dict());models[0].current.requires_grad_(False)
            optimizers[0],schedulers[0]=correction_optimizer()
            scores=metrics(*predict(models[0],'validation'))
            best[0]=scores['mae']+.5*scores['active_mae']+.25*scores['delta_mae']
            states[0]={k:v.detach().cpu().clone() for k,v in models[0].state_dict().items()}
            best_epoch[0]=epoch
            models[0].train()
        total=np.zeros(2);batches=0
        for f,e,y in train:
            if (ROOT/'out/training.stop').exists() or (ROOT/'out/longrun.stop').exists():raise SystemExit('Explicit stop; last complete epoch is checkpointed')
            f,e,y=(v.to(device) for v in (f,e,y));batches+=1
            f,e=jitter_inputs(f,e)
            for n,(model,opt) in enumerate(zip(models,optimizers)):
                if staged and ((epoch<warmup and n==0) or (epoch>=warmup and n==1)):continue
                auxiliary=None
                if a.variant=='deep' and n==0:
                    prediction,auxiliary=model.training_outputs(f,e);prediction=prediction[:,-2:]
                else:prediction=model(f,e)[:,-2:]
                loss=torch.nn.functional.smooth_l1_loss(prediction,y[:,:2],beta=.05)
                loss+=.35*torch.nn.functional.smooth_l1_loss(prediction[:,1]-prediction[:,0],y[:,1]-y[:,0],beta=.025)
                if auxiliary is not None:loss+=.15*torch.nn.functional.smooth_l1_loss(auxiliary[:,-1],y[:,2:],beta=.05)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
                total[n]+=float(loss.detach())
            if batches%100==0:write(a.out/'training-progress.json',{'phase':'training','epoch':epoch+1,'epochs':a.epochs,
                'batches':batches,'total_batches':len(train),'mean_loss':(total/batches).tolist(),'time':time.time()})
        record={'epoch':epoch+1,'loss':(total/max(1,batches)).tolist(),'validation':[]}
        for n,m in enumerate(models):
            scores=metrics(*predict(m,'validation'));record['validation'].append(scores)
            criterion=scores['mae']+.5*scores['active_mae']+.25*scores['delta_mae']
            selecting=not (staged and epoch<warmup and n==0)
            if selecting and criterion<best[n]:
                best[n]=criterion;states[n]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};best_epoch[n]=epoch+1
            if not (staged and ((epoch<warmup and n==0) or (epoch>=warmup and n==1))):schedulers[n].step()
        history.append(record);write(a.out/'training-progress.json',{'phase':'training','epoch':epoch+1,'epochs':a.epochs,'last':record})
        print(json.dumps(record),flush=True)
        temporary=checkpoint.with_suffix('.tmp.pt')
        torch.save({'signature':signature,'epoch':epoch+1,'models':[m.state_dict() for m in models],
            'optimizers':[o.state_dict() for o in optimizers],'schedulers':[s.state_dict() for s in schedulers],
            'best':best,'best_states':states,'best_epoch':best_epoch,'history':history,'torch_rng':torch.get_rng_state(),
            'numpy_rng':np.random.get_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None},temporary)
        temporary.replace(checkpoint)
        if a.patience and epoch+1>=warmup+8 and epoch+1-best_epoch[0]>=a.patience:
            stop_reason='validation_patience';break
    test=[];predictions=[]
    for n,m in enumerate(models):
        m.load_state_dict(states[n]);values=predict(m,'test');predictions.append(values);test.append(metrics(*values))
    ablated=metrics(*predict(models[0],'test',ablate=True))
    jittered=metrics(*predict(models[0],'test',jitter=True))
    permitted=(accept(test[0],test[1]) and test[0]['active_mae']<=ablated['active_mae']*.99 and
               jittered['mae']<=test[0]['mae']*1.15 and jittered['active_mae']<=test[0]['active_mae']*1.15)
    reference_scores=None
    if reference_report is not None:
        reference_model=build_models(variant=reference_report.get('variant','base'))[0].to(device)
        reference_model.load_state_dict(torch.load(a.reference_model_dir/'steering-state.pt',map_location=device,weights_only=True))
        reference_scores=metrics(*predict(reference_model,'test',history_limit=reference_report.get('history_frames',9)))
        permitted=permitted and accept(test[0],reference_scores)
    report={'phase':'complete','architecture_version':version,'variant':a.variant,'baseline_warmup_epochs':warmup if staged else None,
            'accepted_offline':permitted,'created_at':time.time(),'encoder':str((a.cache/'encoder.onnx').resolve()),
            'encoder_source_fingerprint':manifest.get('base_sha256'),'visual_adaptation':manifest.get('visual_adaptation'),
            'dataset_revision':manifest['revision'],'history_frames':history_frames,'sample_hz':5,'target':'mean applied steer over next 0.2-0.6 seconds',
            'source_sample_hz':manifest.get('source_sample_hz',5.),'temporal_stride':stride,
            'epochs':len(history),'max_epochs':a.epochs,'best_validation_epochs':best_epoch,'stop_reason':stop_reason,
            'samples_per_epoch':min(a.samples_per_epoch,len(weights)),
            'batch_size':a.batch_size,'warm_started_from_reference_single_frame':warm_started,
            'curriculum':'balanced turn direction and steering changes, capped at 6x' if a.variant=='deep' else 'active turns 3x',
            'auxiliary_forecasts_s':[[.8,1.2],[1.4,1.8]] if a.variant=='deep' else [],
            'labels':'Clock-verified Data Out state, not raw hardware commands','samples':{k:len(v) for k,v in split.items()},
            'test_temporal':test[0],'test_single_frame':test[1],'history':history,
            'reference_model':str(a.reference_model_dir) if a.reference_model_dir else None,'test_reference':reference_scores,
            'reference_weight_sha256':reference_hash,
            'test_without_real_history':ablated,'parameter_counts':[sum(p.numel() for p in m.parameters()) for m in models],
            'test_interpolated_history':jittered,'history_resampling_checked':True,
            'deployment':'candidate only; requires runtime shadow and bounded trial'}
    f,e,_=datasets['test'][0]
    report['onnx_max_difference']=export_steering_model(models[0],f[None],e[None],a.out/'steering.onnx')
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
