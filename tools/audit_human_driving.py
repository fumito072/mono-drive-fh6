"""Audit each same-stem MP4/CSV pair without changing or training on originals."""
import argparse
import json
from pathlib import Path
import sys
import time
import subprocess
import csv

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mono_drive.human_data_audit import audit_pair


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT.parent/'Play-data')
    parser.add_argument('--out',type=Path,default=ROOT/'out/human-data-audit')
    parser.add_argument('--reviews',type=Path,default=ROOT/'configs/human_data_reviews.json')
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--decode-all',action='store_true',help='Decode the entire video and record any decoder errors')
    args=parser.parse_args()
    if not args.source.is_dir():parser.error('Source directory does not exist')
    args.out.mkdir(parents=True,exist_ok=True)
    reviews=json.loads(args.reviews.read_text(encoding='utf-8')) if args.reviews.exists() else {}
    items=[]
    files={p.stem.lower():p for p in args.source.glob('*') if p.suffix.lower()=='.csv'}
    for video in sorted(args.source.glob('*')):
        if video.suffix.lower()!='.mp4':continue
        csv_path=files.get(video.stem.lower())
        if csv_path is None:
            items.append(dict(recording=video.stem,verdict='missing_csv'));continue
        destination=args.out/(video.stem+'.json')
        fingerprint={str(p.resolve()):[p.stat().st_size,p.stat().st_mtime_ns] for p in (video,csv_path)}
        review=reviews.get(video.stem,{})
        previous=json.loads(destination.read_text(encoding='utf-8')) if destination.exists() else {}
        if not args.refresh and previous.get('fingerprint')==fingerprint and previous.get('review')==review:
            result=previous
        else:
            try:
                result=audit_pair(video,csv_path,review)
                if previous.get('fingerprint')==fingerprint and previous.get('full_decode'):
                    result['full_decode']=previous['full_decode']
                    if not result['full_decode'].get('verified'):
                        result.update(verdict='decode_errors',training_admitted=False)
                after={str(p.resolve()):[p.stat().st_size,p.stat().st_mtime_ns] for p in (video,csv_path)}
                if fingerprint!=after:
                    result.update(verdict='recording_in_progress',training_admitted=False)
                result.update(audited_at=time.time(),fingerprint=fingerprint,review=review)
            except (ValueError,KeyError,OSError,subprocess.SubprocessError) as error:
                result=dict(recording=video.stem,verdict='invalid',training_admitted=False,error=str(error))
            destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        if args.decode_all and not result.get('full_decode',{}).get('verified'):
            decoded=subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(video),'-map','0:v:0','-an','-f','null','-'],
                                   capture_output=True,text=True)
            result['full_decode']=dict(verified=decoded.returncode==0 and not decoded.stderr.strip(),
                                       exit_code=decoded.returncode,errors=decoded.stderr[-3000:])
            if not result['full_decode']['verified']:
                result.update(verdict='decode_errors',training_admitted=False)
            destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        items.append({k:result.get(k) for k in ('recording','verdict','numeric_eligible','training_admitted','path_candidate_eligible')})
    video_stems={p.stem.lower() for p in args.source.glob('*') if p.suffix.lower()=='.mp4'}
    for stem,path in files.items():
        if stem not in video_stems:items.append(dict(recording=path.stem,verdict='missing_video',training_admitted=False))
    # The newer recorder keeps exact synchronization in a complete session
    # directory. Inventory those artifacts; never guess an unfamiliar mapping.
    for metadata in sorted(args.source.rglob('session.json')):
        folder=metadata.parent;name=folder.name
        from mono_drive.human_recorder import audit,AUDIT_VERSION,FILES
        fingerprint={n:[(folder/n).stat().st_size,(folder/n).stat().st_mtime_ns]
                     for n in FILES if (folder/n).exists()}
        destination=args.out/(name+'.json')
        previous=json.loads(destination.read_text(encoding='utf-8')) if destination.exists() else {}
        try:
            reusable=(not args.refresh and previous.get('fingerprint')==fingerprint and
                      previous.get('audit_version')==AUDIT_VERSION and
                      (not previous.get('path_candidate_eligible') or
                       Path(previous.get('verified_alignment','')).is_file()))
            result=previous if reusable else audit(folder,args.out)
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
            result=dict(recording=name,kind='fh6_recorder_session',path=str(folder.resolve()),
                        verdict='invalid',training_admitted=False,path_candidate_eligible=False,error=str(error),
                        fingerprint=fingerprint,audit_version=AUDIT_VERSION)
        (args.out/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        items.append({k:result.get(k) for k in ('recording','kind','verdict','training_admitted','path_candidate_eligible')})
    summary=dict(time=time.time(),source=str(args.source.resolve()),recordings=items,
                 policy='Read-only audit. No automatic training admission without verified synchronization.')
    (args.out/'index.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
