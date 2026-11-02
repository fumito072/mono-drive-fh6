"""Reclaim decoded frame caches without removing recordings or training inputs."""
import csv
import json
from pathlib import Path
import shutil
import time

from .forza_local_learning import read_json, write_json


def cache_candidates(root):
    root=Path(root).resolve()
    base=root/'datasets/human'
    if not base.is_dir():return
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or folder.is_symlink() or folder.is_junction():continue
        caches=[p for p in folder.iterdir() if p.name=='video_frames' or p.name.startswith('sampled-v2-s')]
        if not caches:continue
        manifest=read_json(folder/'manifest.json',{})
        partitions=manifest.get('train',[])+manifest.get('validation',[])
        if not manifest.get('train') or not manifest.get('validation'):continue
        # A completed preparation leaves independent hardlinks/copies in each
        # partition. Verify every expected training image before dropping cache.
        ready=True
        for name in partitions:
            part=Path(name).resolve()
            if folder.resolve() not in part.parents or not (part/'poses.csv').is_file():
                ready=False;break
            try:
                with (part/'poses.csv').open(encoding='utf-8-sig',newline='') as handle:
                    frames=[int(row['frame']) for row in csv.DictReader(handle)]
                if not frames or any(not (part/'frames'/f'{i:07d}.jpg').is_file() for i in frames):
                    ready=False;break
            except (OSError,ValueError,KeyError):
                ready=False;break
        if not ready:continue
        audit=read_json(root/'out/human-data-audit'/f"{manifest.get('recording','')}.json",{})
        source=Path(audit['path'])/'video.mkv' if audit.get('path') else Path(audit.get('video',''))
        if not source.is_file() or source.stat().st_size==0:continue
        for cache in sorted(caches):
            target=cache.resolve()
            if (not cache.is_dir() or cache.is_symlink() or cache.is_junction() or
                    target.parent!=folder.resolve() or base.resolve() not in target.parents):continue
            entries=list(cache.iterdir())
            if any(not p.is_file() or p.is_symlink() or p.suffix.lower() not in ('.jpg','.json') for p in entries):continue
            yield cache,source,entries


def reclaim_frame_caches(root, target_free_gib=80., *, all_completed=False):
    """Serialize with preparation; decode caches are expendable, partitions aren't."""
    from .forza_data_worker import acquire_guard
    root=Path(root).resolve();out=root/'out'
    before=shutil.disk_usage(root).free
    result=dict(time=time.time(),free_before_gib=before/1024**3,removed_caches=[],files=0,
                recordings_retained=True,training_partitions_retained=True)
    guard=acquire_guard(20075)
    if guard is None:
        return dict(result,phase='preparation_busy',free_after_gib=before/1024**3)
    try:
        for cache,source,entries in cache_candidates(root):
            if not all_completed and shutil.disk_usage(root).free>=target_free_gib*1024**3:break
            # Exact absolute child paths have been checked above. No recursive
            # deletion, original media, or partition paths are permitted here.
            for path in entries:
                path.unlink()
                result['files']+=1
            cache.rmdir()
            result['removed_caches'].append(dict(path=str(cache),source_video=str(source)))
        result.update(phase='finished',free_after_gib=shutil.disk_usage(root).free/1024**3)
        if out.is_dir():
            write_json(out/'storage-maintenance-state.json',result)
            with (out/'storage-maintenance.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps(result)+'\n')
        return result
    finally:
        guard.close()


if __name__=='__main__':
    import argparse
    import os
    if os.name=='nt':
        import psutil
        try:
            process=psutil.Process()
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            process.ionice(psutil.IOPRIO_VERYLOW)
        except psutil.Error:
            pass
    parser=argparse.ArgumentParser()
    parser.add_argument('--all-completed',action='store_true',help='Prune all verified redundant decoded caches')
    args=parser.parse_args()
    result=reclaim_frame_caches(Path(__file__).resolve().parents[1],all_completed=args.all_completed)
    print(json.dumps(result),flush=True)
    from .forza_retention import reclaim_old_runs
    result=reclaim_old_runs(Path(__file__).resolve().parents[1],apply=True)
    print(json.dumps({k:v for k,v in result.items() if k!='removed'}),flush=True)
