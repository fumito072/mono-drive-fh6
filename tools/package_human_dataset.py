"""Package the prepared human dataset without publishing local source metadata.

This is an explicit, one-shot export. It does not alter source files or upload.
Only complete episodes from the given human-dataset.json are included.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
import zipfile


CAMERA_FIELDS = ("width", "height", "hfov_deg", "pitch_deg", "height_m", "fwd_m", "right_m")
POSE_FIELDS = {"frame", "t", "x", "y", "z", "yaw_deg", "pitch_deg", "roll_deg", "kmh"}
ACTION_FIELDS = {"frame", "video_pts_s", "csv_elapsed_s", "applied_throttle", "applied_brake",
                 "applied_steer", "telemetry_seq", "age_ms", "throttle", "brake"}
PROVENANCE_FIELDS = ("time_range", "video_range", "offset_s", "offset_verified_for_actions",
                     "raw_controls_replayed", "camera_estimated", "car_ordinals")
SYNC_FIELDS = ("verified", "video_pts_verified", "raw_packets_verified", "matched_frames",
               "corrected_alignment_rows", "clock_offset_correction_s",
               "frame_clock_span_ms", "median_age_ms", "max_age_ms")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    # Windows readers and virus scanners can briefly deny a rename.
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.1)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_value(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [numeric_value(item) for item in value]
    raise ValueError("Only finite numeric/boolean metadata may be exported")


def numeric_fields(value, allowed):
    return {key: numeric_value(value[key]) for key in allowed if key in value}


def clean_csv(path, allowed):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if not header or len(header) != len(set(header)) or not set(header) <= allowed or "frame" not in header:
            raise ValueError(f"Unexpected columns in {Path(path).name}")
        rows = list(reader)
    for number, row in enumerate(rows):
        if len(row) != len(header) or any(not math.isfinite(float(cell)) for cell in row):
            raise ValueError(f"Invalid numeric row {number} in {Path(path).name}")
        if float(row[header.index("frame")]) != number:
            raise ValueError("Frame numbers must be contiguous from zero")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue(), len(rows)


def inside(path, root):
    path, root = Path(path), Path(root).resolve()
    resolved = path.resolve()
    if path.is_symlink() or root not in resolved.parents or not resolved.exists():
        raise ValueError("Source must be a regular path inside the prepared human data root")
    return resolved


def export_episode(episode, source_root, output, release_base):
    identity = str(episode["recording"]) + ":" + str(episode["revision"])
    episode_id = "episode-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    output = Path(output)
    archive = output / (episode_id + ".zip")
    record_file = output / (episode_id + ".json")
    source_fingerprint = hashlib.sha256(json.dumps(episode, sort_keys=True).encode()).hexdigest()
    if archive.exists() and record_file.exists():
        saved = read_json(record_file)
        if saved.get("source_fingerprint") == source_fingerprint and saved["bytes"] == archive.stat().st_size and saved["sha256"] == sha256(archive):
            return {key: value for key, value in saved.items() if key != "source_fingerprint"}
        raise ValueError("Existing package differs; choose a new output directory")
    public = {"id": episode_id, "frames": 0, "train_frames": 0, "validation_frames": 0, "segments": [],
              "synchronization": numeric_fields(episode.get("synchronization", {}), SYNC_FIELDS)}
    excluded = episode.get("synchronization", {}).get("excluded_frames")
    if isinstance(excluded, dict):
        public["synchronization"]["excluded_frames_count"] = sum(numeric_value(v) for v in excluded.values())
    for key in ("car_ordinals", "offset_s", "offset_uncertainty_s", "preparation_version"):
        if key in episode:
            public[key] = numeric_value(episode[key])
    temporary = archive.with_suffix(".zip.part")
    seen = set()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for split in ("train", "validation"):
            for raw in episode.get(split, []):
                source = inside(raw, source_root)
                if not re.fullmatch(split + r"_\d+", source.name) or source.name in seen:
                    raise ValueError("Invalid or duplicate segment name")
                seen.add(source.name)
                prefix = f"human/{episode_id}/{source.name}"
                poses, count = clean_csv(inside(source / "poses.csv", source_root), POSE_FIELDS)
                actions, action_count = clean_csv(inside(source / "actions.csv", source_root), ACTION_FIELDS)
                if count != action_count:
                    raise ValueError("Pose/action frame counts differ")
                images = sorted((source / "frames").iterdir())
                if len(images) != count or not count:
                    raise ValueError("Image/label frame counts differ")
                for number, image in enumerate(images):
                    inside(image, source_root)
                    if not re.fullmatch(r"\d+\.(jpg|png)", image.name) or int(image.stem) != number or not image.is_file():
                        raise ValueError("Invalid image sequence")
                    bundle.write(image, f"{prefix}/frames/{image.name}")
                camera = numeric_fields(read_json(inside(source / "camera.json", source_root)), CAMERA_FIELDS)
                camera["name"] = "Recorded game camera; initial calibration estimate"
                provenance = numeric_fields(read_json(inside(source / "provenance.json", source_root)), PROVENANCE_FIELDS)
                provenance.update(recording=episode_id, partition=split, source="human Forza Data Out",
                                  controls_are="game-reported state, not hardware commands")
                for name, data in (("poses.csv", poses), ("actions.csv", actions),
                                   ("camera.json", json.dumps(camera, indent=2)),
                                   ("provenance.json", json.dumps(provenance, indent=2))):
                    bundle.writestr(f"{prefix}/{name}", data)
                public["segments"].append({"path": prefix, "partition": split, "frames": count})
                public["frames"] += count
                public[split + "_frames"] += count
        if public["frames"] != episode["frames"]:
            raise ValueError("Episode total differs from the completed source manifest")
        bundle.writestr(f"human/{episode_id}/episode.json", json.dumps(public, indent=2))
    if temporary.stat().st_size >= 2 * 1024**3:
        raise ValueError("Episode exceeds GitHub's per-release-asset size; split before upload")
    with zipfile.ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Archive CRC validation failed")
    temporary.replace(archive)
    public.update(asset=archive.name, bytes=archive.stat().st_size, sha256=sha256(archive),
                  url=release_base.rstrip("/") + "/" + archive.name)
    write_json(record_file, dict(public, source_fingerprint=source_fingerprint))
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release-base", required=True)
    args = parser.parse_args()
    source = read_json(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = {"schema_version": 1, "game": "Forza Horizon 6", "source": "human demonstrations",
               "revision": source["revision"], "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "pending_episodes_excluded": len(source.get("pending", [])), "episodes": []}
    for episode in source["episodes"]:
        record = export_episode(episode, args.source_root, args.output, args.release_base)
        catalog["episodes"].append(record)
        print(json.dumps({"packaged": len(catalog["episodes"]), "total": len(source["episodes"]),
                          "episode": record["id"], "frames": record["frames"], "bytes": record["bytes"]}), flush=True)
        write_json(args.output / "packaging-state.json", {"phase": "packaging", "completed": len(catalog["episodes"]),
                                                         "total": len(source["episodes"]), "time": time.time()})
    catalog["total_frames"] = sum(e["frames"] for e in catalog["episodes"])
    catalog["total_bytes"] = sum(e["bytes"] for e in catalog["episodes"])
    if catalog["total_frames"] != source["frames"]:
        raise ValueError("Dataset total differs from the completed source manifest")
    write_json(args.output / "catalog.json", catalog)
    write_json(args.output / "packaging-state.json", {"phase": "complete", "completed": len(catalog["episodes"]),
                                                     "total": len(source["episodes"]), "time": time.time()})


if __name__ == "__main__":
    main()
