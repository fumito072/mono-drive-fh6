"""Recheck source CSV and raw records; never modify original recordings."""
import argparse,csv,json,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mono_drive.control_evidence import inspect_controls
from mono_drive.forza_local_learning import read_json,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path)
    p.add_argument('--status',choices=['all','error'],default='all',help='Limit remote reads to failed recordings when requested')
    p.add_argument('--out',type=Path,default=ROOT/'out/teacher-control-audit.json');a=p.parse_args()
    source=a.source or Path(read_json(ROOT/'configs/drive_data_source.json',{})['local_folder'])
    records=[];errors=[]
    for path in sorted(source.rglob('session.json')):
        if a.status!='all' and read_json(path,{}).get('status')!=a.status:continue
        try:records.append(inspect_controls(path.parent))
        except (OSError,ValueError,csv.Error) as error:errors.append({'folder':str(path.parent),'error':str(error)})
    counts=Counter(x['verdict'] for x in records)
    result={'time':time.time(),'source':str(source),'records':records,'errors':errors,'counts':dict(counts)}
    write_json(a.out,result);print(json.dumps({'counts':dict(counts),'errors':len(errors)}))

if __name__=='__main__':
    main()
