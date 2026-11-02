"""Restore the public model snapshot, verifying every asset and output file."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import urllib.request
import zipfile


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def destination(root, name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or any(':' in p for p in path.parts):
        raise ValueError('Unsafe model path')
    if not path.parts or (path.parts[0] not in {'models','model-licenses'} and name!='yolo11s.pt'):
        raise ValueError('Unexpected model location')
    root=Path(root).resolve();result=root.joinpath(*path.parts).resolve()
    if root not in result.parents:raise ValueError('Model path escapes destination')
    return result


def fetch(asset, cache):
    name=asset['name']
    if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('Unsafe asset name')
    path=cache/name
    if path.exists() and path.stat().st_size==asset['bytes'] and sha256(path)==asset['sha256']:
        return path
    url=asset['url']
    if not url.startswith('https://github.com/fumito072/mono-drive-fh6/releases/download/model-weights-2026-11-02/'):
        raise ValueError('Unexpected model download host')
    print(f"Downloading {name} ({asset['bytes']/1024**2:.1f} MiB)",flush=True)
    temporary=path.with_suffix(path.suffix+'.downloading')
    request=urllib.request.Request(url,headers={'User-Agent':'mono-drive-model-downloader'})
    with urllib.request.urlopen(request,timeout=120) as src,temporary.open('wb') as dst:
        shutil.copyfileobj(src,dst,8*1024**2)
    if temporary.stat().st_size!=asset['bytes'] or sha256(temporary)!=asset['sha256']:
        raise ValueError('Downloaded model asset checksum mismatch')
    temporary.replace(path);return path


def restore_parts(parts, output, record, overwrite=False):
    if output.exists():
        if output.stat().st_size==record['bytes'] and sha256(output)==record['sha256']:return
        if not overwrite:raise FileExistsError(f'Different file exists: {output}. Use --overwrite explicitly.')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.downloading')
    with temporary.open('wb') as dst:
        for part in parts:
            with Path(part).open('rb') as src:shutil.copyfileobj(src,dst,8*1024**2)
    if temporary.stat().st_size!=record['bytes'] or sha256(temporary)!=record['sha256']:
        raise ValueError('Reassembled model checksum mismatch')
    temporary.replace(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,default=Path(__file__).resolve().parents[1]/'model_weights/catalog.json')
    parser.add_argument('--output',type=Path,default=Path('.'))
    parser.add_argument('--small-only',action='store_true',help='Skip the large Qwen language-model weights')
    parser.add_argument('--overwrite',action='store_true',help='Replace local files that differ from the snapshot')
    args=parser.parse_args();catalog=json.loads(args.catalog.read_text(encoding='utf-8'))
    root=args.output.resolve();cache=root/'models/.downloads';cache.mkdir(parents=True,exist_ok=True)
    assets={a['name']:a for a in catalog['assets']}
    expected={f['path']:f for f in catalog['files'] if 'archive' in f}
    archive=fetch(assets['runtime-models.zip'],cache)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            output=destination(root,member.filename)
            if member.is_dir():continue
            if stat.S_ISLNK(member.external_attr>>16):raise ValueError('ZIP symlinks are not supported')
            record=expected.get(member.filename)
            if record is None and not member.filename.startswith('model-licenses/'):
                raise ValueError('Unlisted model archive member')
            if record and output.exists() and not args.overwrite:
                if sha256(output)==record['sha256']:continue
                raise FileExistsError(f'Different file exists: {output}. Use --overwrite explicitly.')
            output.parent.mkdir(parents=True,exist_ok=True)
            temporary=output.with_suffix(output.suffix+'.downloading')
            with z.open(member) as src,temporary.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024**2)
            if record and (temporary.stat().st_size!=record['bytes'] or sha256(temporary)!=record['sha256']):
                raise ValueError('Restored file checksum mismatch')
            temporary.replace(output)
    restored={}
    if not args.small_only:
        for record in catalog['files']:
            if 'parts' not in record:continue
            output=destination(root,record['path'])
            if record['sha256'] in restored and not output.exists():
                output.parent.mkdir(parents=True,exist_ok=True)
                try:output.hardlink_to(restored[record['sha256']])
                except OSError:shutil.copyfile(restored[record['sha256']],output)
            else:
                parts=[fetch(assets[n],cache) for n in record['parts']]
                restore_parts(parts,output,record,args.overwrite)
            restored[record['sha256']]=output
    print('Model files restored. No driving model was automatically selected.')


if __name__=='__main__':main()
