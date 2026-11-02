"""Download checksum-pinned human data releases and restore the training split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import urllib.request
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_member(name, root, episode_id):
    member = PurePosixPath(name)
    if "\\" in name or member.is_absolute() or ".." in member.parts or any(":" in p for p in member.parts):
        raise ValueError("Unsafe ZIP member")
    if len(member.parts) < 3 or member.parts[:2] != ("human", episode_id):
        raise ValueError("Unexpected episode path")
    root = Path(root).resolve()
    result = root.joinpath(*member.parts).resolve()
    if root not in result.parents:
        raise ValueError("ZIP member escapes output directory")
    return result


def extract_verified(archive, output, episode):
    if Path(archive).stat().st_size != episode["bytes"] or sha256(archive) != episode["sha256"]:
        raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = safe_member(member.filename, output, episode["id"])
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("ZIP symlinks are not allowed")
            if member.file_size > 512 * 1024**2:
                raise ValueError("Oversized individual ZIP member")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, 1024 * 1024)
    for segment in episode["segments"]:
        folder = safe_member(segment["path"] + "/poses.csv", output, episode["id"]).parent
        images = list((folder / "frames").iterdir())
        if len(images) != segment["frames"]:
            raise ValueError("Extracted image count differs from catalog")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parents[1] / "teacher_data/catalog.json")
    parser.add_argument("--output", type=Path, default=Path("datasets/public-human-20261102"))
    parser.add_argument("--episodes", nargs="*", help="Optional episode IDs for a small subset")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    selected = set(args.episodes or [e["id"] for e in catalog["episodes"]])
    if selected - {e["id"] for e in catalog["episodes"]}:
        parser.error("Unknown episode ID")
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / ".archives"
    cache.mkdir(exist_ok=True)
    manifest = {"drives": [], "extra_validation": []}
    for episode in catalog["episodes"]:
        if episode["id"] not in selected:
            continue
        if not re.fullmatch(r"episode-[0-9a-f]{16}\.zip", episode["asset"]):
            raise ValueError("Unexpected archive name")
        archive = cache / episode["asset"]
        if not archive.exists() or archive.stat().st_size != episode["bytes"] or sha256(archive) != episode["sha256"]:
            if not episode["url"].startswith("https://github.com/fumito072/mono-drive-fh6/releases/download/"):
                raise ValueError("Unexpected download host or repository")
            print(f"Downloading {episode['id']} ({episode['bytes'] / 1024**2:.1f} MiB)", flush=True)
            request = urllib.request.Request(episode["url"], headers={"User-Agent": "mono-drive-dataset-downloader"})
            temporary = archive.with_suffix(".part")
            with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
            if temporary.stat().st_size != episode["bytes"] or sha256(temporary) != episode["sha256"]:
                raise ValueError("Downloaded archive checksum mismatch")
            temporary.replace(archive)
        extract_verified(archive, args.output, episode)
        for segment in episode["segments"]:
            key = "drives" if segment["partition"] == "train" else "extra_validation"
            manifest[key].append(str((args.output / segment["path"]).resolve()))
        print(f"Verified and restored {episode['id']}", flush=True)
    destination = args.output / "training-manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Training manifest: {destination.resolve()}")


if __name__ == "__main__":
    main()
