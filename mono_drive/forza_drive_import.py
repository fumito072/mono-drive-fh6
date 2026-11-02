"""Import completed recorder sessions from a user-selected Drive desktop folder.

Google Drive for desktop owns authentication and downloads. This module reads
only the configured local folder, never writes to it, and never executes its
contents. Imported sessions still pass the ordinary human-data quality audit.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import uuid
import zipfile

from .forza_local_learning import read_json, write_json
from .human_recorder import FILES


def safe_zip_name(info):
    name = info.filename.replace('\\', '/')
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ('..', '') or ':' in part for part in path.parts):
        raise ValueError('Unsafe archive path')
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError('Archive links are not accepted')
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(files):
    return [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in files]


def session_hash(folder):
    digest = hashlib.sha256()
    for name in FILES:
        digest.update(name.encode())
        digest.update(sha256_file(Path(folder) / name).encode())
    return digest.hexdigest()


def index_existing_sessions(target, ledger):
    if ledger.get('existing_sessions_indexed'):
        return
    hashes = ledger.setdefault('session_hashes', {})
    for metadata in target.rglob('session.json'):
        folder = metadata.parent
        if folder.name.startswith('.incoming-') or not all((folder / name).is_file() for name in FILES):
            continue
        if complete(read_json(metadata, {})):
            hashes[session_hash(folder)] = str(folder)
    ledger['existing_sessions_indexed'] = True


def stable(ledger, key, observed, now, wait):
    old = ledger.setdefault('observations', {}).get(key)
    if not old or old['signature'] != observed:
        ledger['observations'][key] = dict(signature=observed, since=now)
        return False
    return now - old['since'] >= wait


def complete(metadata):
    return isinstance(metadata, dict) and metadata.get('status') == 'complete' and metadata.get('synthetic_demo') is False and not metadata.get('errors')


def publish(staging, destination, provenance):
    (staging / 'import-source.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    if destination.exists():
        # A receipt is necessary to recognize an already committed import.
        receipt = read_json(destination / 'import-source.json', {})
        if receipt.get('content_sha256') != provenance['content_sha256']:
            raise ValueError('Import destination already exists with different contents')
        return False
    staging.rename(destination)
    return True


def archive_sessions(archive):
    groups = {}
    names = set()
    for info in archive.infolist():
        path = safe_zip_name(info)
        if info.is_dir():
            continue
        if str(path).casefold() in names:
            raise ValueError('Duplicate archive filename')
        names.add(str(path).casefold())
        if path.name in FILES:
            groups.setdefault(path.parent, {})[path.name] = info
    return {parent: files for parent, files in groups.items() if set(files) == set(FILES)}


def import_zip(path, target, ledger, max_bytes, reserve, now):
    before = signature([path])
    digest = sha256_file(path)
    if digest in ledger.setdefault('imported', {}):
        return 0
    imported = []
    with zipfile.ZipFile(path) as archive:
        groups = archive_sessions(archive)
        if not groups:
            raise ValueError('No complete six-file recorder session in ZIP')
        total = sum(info.file_size for files in groups.values() for info in files.values())
        if total > max_bytes or shutil.disk_usage(target).free < total + reserve:
            raise ValueError('Insufficient reserved space or archive exceeds import size limit')
        for index, (parent, files) in enumerate(sorted(groups.items(), key=lambda item: str(item[0]))):
            if files['session.json'].file_size > 2 * 1024 * 1024:
                raise ValueError('Recorder metadata is too large')
            metadata = json.loads(archive.read(files['session.json']).decode('utf-8-sig'))
            if not complete(metadata):
                raise ValueError('Session is unfinished, synthetic, or contains recorder errors')
            staging = target / ('.incoming-drive-' + uuid.uuid4().hex)
            staging.mkdir()
            try:
                for name, info in files.items():
                    with archive.open(info) as source, (staging / name).open('wb') as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
                if signature([path]) != before:
                    raise ValueError('Source ZIP changed during import')
                content_hash = session_hash(staging)
                if content_hash in ledger.setdefault('session_hashes', {}):
                    continue
                destination = target / f'drive_{content_hash[:16]}'
                receipt = dict(source_archive=str(path), session_in_archive=str(parent), content_sha256=content_hash, archive_sha256=digest, imported_at=now)
                publish(staging, destination, receipt)
                ledger['session_hashes'][content_hash] = str(destination)
                imported.append(str(destination))
            finally:
                # Only our newly created, verified staging child may be removed.
                if staging.parent.resolve() == target.resolve() and staging.name.startswith('.incoming-drive-') and staging.exists():
                    shutil.rmtree(staging)
    ledger['imported'][digest] = dict(source=str(path), sessions=imported, time=now)
    return len(imported)


def import_folder(folder, target, ledger, max_bytes, reserve, now):
    files = [folder / name for name in FILES]
    before = signature(files)
    metadata = read_json(folder / 'session.json', {})
    if not complete(metadata):
        raise ValueError('Session is unfinished, synthetic, or contains recorder errors')
    total = sum(p.stat().st_size for p in files)
    if total > max_bytes or shutil.disk_usage(target).free < total + reserve:
        raise ValueError('Insufficient reserved space or session exceeds import size limit')
    staging = target / ('.incoming-drive-' + uuid.uuid4().hex)
    staging.mkdir()
    try:
        digest = hashlib.sha256()
        for source in files:
            if source.is_symlink():
                raise ValueError('Source links are not accepted')
            output = staging / source.name
            shutil.copyfile(source, output)
            digest.update(source.name.encode())
            digest.update(sha256_file(output).encode())
        key = digest.hexdigest()
        if signature(files) != before:
            raise ValueError('Source recording changed during import')
        if key in ledger.setdefault('session_hashes', {}):
            return 0
        destination = target / f'drive_{key[:16]}'
        publish(staging, destination, dict(source_directory=str(folder), content_sha256=key, imported_at=now))
        ledger.setdefault('imported', {})[key] = dict(source=str(folder), sessions=[str(destination)], time=now)
        ledger['session_hashes'][key] = str(destination)
        return 1
    finally:
        if staging.parent.resolve() == target.resolve() and staging.name.startswith('.incoming-drive-') and staging.exists():
            shutil.rmtree(staging)


def sync_once(root, now=None):
    root = Path(root).resolve()
    now = time.time() if now is None else now
    config = read_json(root / 'configs/drive_data_source.json', {})
    state_path = root / 'out/drive-import-state.json'
    if not config.get('enabled') or not config.get('local_folder'):
        result = dict(time=now, phase='waiting_for_connection', imported_sessions=0,
                      expected_account=config.get('expected_account'), account_verified=False)
        write_json(state_path, result)
        return result
    source = Path(config['local_folder']).resolve()
    target = root.parent / 'Play-data'
    if not source.is_dir() or source == target or source in target.parents or target in source.parents:
        result = dict(time=now, phase='source_unavailable_or_overlapping', imported_sessions=0)
        write_json(state_path, result)
        return result
    target.mkdir(exist_ok=True)
    ledger_path = root / 'out/drive-import-index.json'
    ledger = read_json(ledger_path, {})
    index_existing_sessions(target, ledger)
    max_bytes = min(100., max(1., float(config.get('max_session_gib', 30)))) * 1024**3
    reserve = max(25., float(config.get('reserve_gib', 25))) * 1024**3
    wait = max(60., float(config.get('stable_seconds', 120)))
    batch_limit = max(1, min(5, int(config.get('max_new_sessions_per_cycle', 2))))
    imported = waiting = 0
    errors = []
    archive_paths = sorted(source.rglob('*.zip'))
    folders = sorted({p.parent for p in source.rglob('session.json')})
    candidates = [(p, [p], import_zip) for p in archive_paths]
    for folder in folders:
        files = [folder / name for name in FILES]
        if all(p.is_file() for p in files):
            candidates.append((folder, files, import_folder))
        else:
            waiting += 1
    for path, files, importer in candidates:
        try:
            if path.is_symlink() or any(p.is_symlink() or not p.resolve().is_relative_to(source) for p in files):
                raise ValueError('Only ordinary files inside the selected folder are allowed')
            observed = signature(files)
            key = str(path)
            if not stable(ledger, key, observed, now, wait):
                waiting += 1
                continue
            # No rehash or recopy for a source whose completed import is unchanged.
            if ledger.setdefault('completed_signatures', {}).get(key) == observed:
                continue
            write_json(state_path,dict(time=time.time(),phase='importing',source=str(source),current_recording=key,
                                      imported_sessions=imported,remote_writes=False))
            added = importer(path, target, ledger, max_bytes, reserve, now)
            imported += added
            with (root/'out/drive-import-events.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps(dict(time=time.time(),source=key,imported_sessions=added),ensure_ascii=False)+'\n')
            ledger['completed_signatures'][key] = observed
            if imported >= batch_limit:
                break
        except (OSError, ValueError, zipfile.BadZipFile, RuntimeError, EOFError) as error:
            errors.append(dict(source=str(path), error=str(error)))
    write_json(ledger_path, ledger)
    result = dict(time=now, phase='watching', source=str(source), imported_sessions=imported,
                  waiting_for_complete_sync=waiting, errors=errors, expected_account=config.get('expected_account'),
                  account_verified=bool(config.get('expected_account') and config.get('desktop_account_verified')==config.get('expected_account')),
                  account_verified_at=config.get('desktop_account_verified_at'),
                  account_verification_method='Drive desktop UI at setup; no access to OAuth credentials',
                  authentication='Google Drive for desktop',
                  remote_writes=False)
    write_json(state_path, result)
    return result


if __name__ == '__main__':
    print(json.dumps(sync_once(Path(__file__).resolve().parents[1]), ensure_ascii=False))
