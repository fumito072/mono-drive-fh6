"""Oldest-first retention for local autonomous recordings and copied datasets."""
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import time

from .forza_local_learning import read_json,write_json


GROUPS=(('out','fh6',72),('out','vlm',64),('out','rescue',64),
        ('out','perception',2),('datasets','fh6_candidates',3))


def stamp(name,prefix):
    match=re.fullmatch(re.escape(prefix)+r'_(\d{8}_\d{6})',name)
    if not match:return None
    try:return dt.datetime.strptime(match[1],'%Y%m%d_%H%M%S').timestamp()
    except ValueError:return None


def protected_paths(root):
    """Active recordings, latest selection and manifests of running trainers."""
    root=Path(root).resolve();out=root/'out';protected=set()
    def add(value):
        if isinstance(value,str):
            p=Path(value)
            protected.add((p if p.is_absolute() else root/p).resolve())
    for name in ('driver-heartbeat','vlm-state'):
        add(read_json(out/(name+'.json'),{}).get('run'))
    try:
        import numpy as np
        with np.load(out/'scene.npz',allow_pickle=False) as scene:
            if 'recording' in scene:add(str(out/str(scene['recording'].item())))
    except (OSError,ValueError,KeyError):pass
    manifests=[out/'training-dataset.json',out/'human-dataset.json']
    try:
        import psutil
        for process in psutil.process_iter(['name','cmdline']):
            if 'python' not in (process.info['name'] or '').lower():continue
            command=process.info['cmdline'] or []
            for i,arg in enumerate(command[:-1]):
                if arg=='--dataset-manifest':manifests.append(Path(command[i+1]))
    except ImportError:pass
    for manifest in manifests:
        for key in ('segments','drives','extra_validation','train','validation'):
            for value in read_json(manifest,{}).get(key,[]):add(value)
    def references(value):
        if isinstance(value,dict):
            for item in value.values():references(item)
        elif isinstance(value,list):
            for item in value:references(item)
        elif isinstance(value,str) and ('/' in value or '\\' in value):add(value)
    # Operational map calibration is sometimes stored inside an old rescue
    # recording. It is a runtime input, not an expendable historical screenshot.
    for base in (root/'configs',root/'models/ui'):
        if base.is_dir():
            for file in base.rglob('*.json'):references(read_json(file,{}))
    test_root=root/'tests'
    if test_root.is_dir():
        for file in test_root.rglob('*.py'):
            for group,name in re.findall(r'out[/\\]((?:fh6|rescue|perception|vlm)_(\d{8}_\d{6}))',file.read_text(encoding='utf-8')):
                add(str(out/group))
    return protected


def plan_retention(root,now=None,policy=None):
    root=Path(root).resolve();now=time.time() if now is None else now
    policy=policy or read_json(root/'configs/data_retention.json',{})
    if not policy.get('delete_old_autonomous_runs'):return []
    protected={a for q in protected_paths(root) for a in (q,*q.parents)}
    recent_hours=max(1.,float(policy.get('autonomous_recent_hours',6.)))
    candidates=[]
    for parent,prefix,keep in GROUPS:
        base=root/parent
        if not base.is_dir() or base.is_symlink() or base.is_junction():continue
        found=[]
        for p in base.iterdir():
            when=stamp(p.name,prefix)
            if when is None or not p.is_dir() or p.is_symlink() or p.is_junction():continue
            resolved=p.resolve()
            if resolved.parent!=base.resolve():continue
            found.append((when,p))
        found.sort(key=lambda x:x[0],reverse=True)
        for when,p in found[keep:]:
            resolved=p.resolve()
            if now-when<recent_hours*3600:continue
            if resolved in protected:continue
            if prefix=='fh6' and read_json(p/'manifest.json',{}).get('arguments',{}).get('source') in ('human','game_teacher'):
                continue
            candidates.append(dict(path=str(resolved),recorded_at=when,group=prefix))
    return sorted(candidates,key=lambda r:(r['recorded_at'],r['path']))


def remove_local_tree(path,root):
    """Never traverse a link, junction, reparse point, or unexpected root."""
    path=Path(path);root=Path(root).resolve();resolved=path.resolve()
    allowed=False
    for parent,prefix,_ in GROUPS:
        if resolved.parent==root/parent and stamp(path.name,prefix) is not None:allowed=True
    if not allowed or path.is_symlink() or path.is_junction():
        raise ValueError('Retention target is not an allowed local recording')
    counts=[0,0]
    def remove(directory):
        with os.scandir(directory) as it:entries=list(it)
        for entry in entries:
            stat=entry.stat(follow_symlinks=False)
            if entry.is_symlink() or getattr(stat,'st_file_attributes',0)&0x400:
                raise ValueError('Refusing to traverse a reparse point')
            if entry.is_dir(follow_symlinks=False):remove(entry.path)
            else:
                os.unlink(entry.path);counts[0]+=1;counts[1]+=stat.st_size
        os.rmdir(directory)
    remove(resolved)
    return counts


def reclaim_old_runs(root,apply=False):
    from .forza_data_worker import acquire_guard
    root=Path(root).resolve();out=root/'out'
    result=dict(time=time.time(),phase='preview',applied=False,removed=[],files=0,logical_bytes_removed=0)
    if apply:
        try:
            import psutil
            # Older exporters may not yet know the shared retention lock.
            for p in psutil.process_iter(['name','cmdline']):
                if 'python' in (p.info['name'] or '').lower() and any(
                        Path(a).name=='prepare_forza_training.py' for a in p.info['cmdline'] or []):
                    return dict(result,phase='preparation_busy')
        except ImportError:pass
    guard=acquire_guard(20076)
    if guard is None:return dict(result,phase='preparation_busy')
    try:
        planned=plan_retention(root);result['planned_count']=len(planned)
        if not apply:return dict(result,candidates=planned)
        result.update(applied=True,free_before_gib=shutil.disk_usage(root).free/1024**3)
        write_json(out/'run-retention-plan.json',dict(time=time.time(),candidates=planned))
        journal=out/'run-retention.jsonl'
        protected={a for q in protected_paths(root) for a in (q,*q.parents)};refresh_at=time.monotonic()+30.
        for row in planned:
            if time.monotonic()>=refresh_at:
                protected={a for q in protected_paths(root) for a in (q,*q.parents)};refresh_at=time.monotonic()+30.
            if Path(row['path']) in protected:
                continue
            with journal.open('a',encoding='utf-8') as log:
                log.write(json.dumps(dict(time=time.time(),action='deleting',**row))+'\n')
            try:
                files,size=remove_local_tree(row['path'],root)
                result['removed'].append(row);result['files']+=files;result['logical_bytes_removed']+=size
            except (OSError,ValueError) as error:
                result.setdefault('errors',[]).append(dict(path=row['path'],error=str(error)))
            if len(result['removed'])%25==0:
                write_json(out/'run-retention-state.json',dict(result,phase='cleaning',last_path=row['path']))
        result.update(phase='finished',free_after_gib=shutil.disk_usage(root).free/1024**3)
        write_json(out/'run-retention-state.json',result)
        return result
    finally:guard.close()


if __name__=='__main__':
    import argparse
    if os.name=='nt':
        import psutil
        try:
            process=psutil.Process()
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            process.ionice(psutil.IOPRIO_VERYLOW)
        except psutil.Error:pass
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();r=reclaim_old_runs(Path(__file__).resolve().parents[1],args.apply)
    print(json.dumps({k:v for k,v in r.items() if k not in ('removed','candidates')},ensure_ascii=False))
