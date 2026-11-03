"""Extract 10 Hz verified video with a fixed, disjoint validation interval."""
import hashlib,json,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mono_drive.human_recorder import prepare
from mono_drive.forza_local_learning import read_json,write_json


def main():
    write_json(ROOT/'out/temporal-dense-prepare-state.json',{'phase':'starting','time':time.time()})
    dataset=read_json(ROOT/'out/human-dataset.json',{})
    admitted={e['recording'] for e in dataset.get('episodes',[]) if e.get('car_ordinals')==[3819] and e.get('synchronization',{}).get('verified')}
    records=[read_json(ROOT/'out/human-data-audit'/f'{name}.json',{}) for name in sorted(admitted)]
    records=[r for r in records if r.get('path_candidate_eligible') and r.get('car_ordinals')==[3819]]
    results=[];errors=[];start=time.time()
    write_json(ROOT/'out/temporal-dense-prepare-state.json',{'phase':'extracting','complete':0,'total':len(records),'time':start})
    def one(record):
        try:return prepare(ROOT,record,sample_hz=10.,expand_training=True),None
        except Exception as e:return None,{'recording':record['recording'],'error':str(e)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result,error in pool.map(one,records):
            if error:errors.append(error)
            else:results.append(result)
            progress={'phase':'extracting','complete':len(results)+len(errors),'total':len(records),
                      'frames':sum(x['frames'] for x in results),'errors':errors,'time':time.time(),'elapsed_s':time.time()-start}
            write_json(ROOT/'out/temporal-dense-prepare-state.json',progress)
            print(json.dumps(progress),flush=True)
    if not results:raise ValueError('No verified dense video could be prepared')
    revision=hashlib.sha256(json.dumps([e['revision'] for e in results],sort_keys=True).encode()).hexdigest()
    manifest={'revision':revision,'source_revision':dataset.get('revision'),'sample_hz':10.,'episodes':results,
              'frames':sum(x['frames'] for x in results),'pending':errors,'time':time.time(),
              'train':[p for x in results for p in x['train']],
              'validation':[p for x in results for p in x['validation']]}
    write_json(ROOT/'out/temporal-dense-dataset.json',manifest)
    write_json(ROOT/'out/temporal-dense-prepare-state.json',dict(progress,phase='complete',revision=revision))

if __name__=='__main__':main()
