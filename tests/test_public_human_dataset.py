import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


package = load("package_human_dataset")
download = load("download_human_dataset")


class PublicDatasetTests(unittest.TestCase):
    def test_metadata_write_retries_temporary_windows_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "state.json"
            original = Path.replace
            calls = []
            def transient(path, target):
                calls.append(path)
                if len(calls) == 1:
                    raise PermissionError("temporary reader lock")
                return original(path, target)
            with patch.object(Path, "replace", transient), patch.object(package.time, "sleep"):
                package.write_json(destination, {"complete": True})
            self.assertEqual(len(calls), 2)
            self.assertEqual(json.loads(destination.read_text()), {"complete": True})

    def test_roundtrip_preserves_split_and_removes_private_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "source" / "train_00"
            (segment / "frames").mkdir(parents=True)
            (segment / "frames/0000000.jpg").write_bytes(b"test-image-content")
            (segment / "poses.csv").write_text("frame,t,x,y,z,yaw_deg,pitch_deg,kmh\n0,0,1,2,3,4,5,6\n")
            (segment / "actions.csv").write_text("frame,video_pts_s,throttle,brake,applied_steer\n0,0,0.5,0,0.1\n")
            (segment / "camera.json").write_text(json.dumps({"width": 1280, "height": 720, "name": "private-name"}))
            (segment / "provenance.json").write_text(json.dumps({"recording": "private-name", "source_video": "C:/private/video.mp4", "time_range": [0, 1]}))
            episode = {"recording": "private-name", "revision": "abc", "train": [str(segment)], "validation": [], "frames": 1,
                       "synchronization": {"verified": True, "excluded_frames": {"menu": 2}}}
            output = root / "package"
            output.mkdir()
            record = package.export_episode(episode, root / "source", output, "https://example.test")
            self.assertEqual(record["train_frames"], 1)
            self.assertEqual(record["synchronization"]["excluded_frames_count"], 2)
            with zipfile.ZipFile(output / record["asset"]) as archive:
                text = "".join(archive.read(n).decode() for n in archive.namelist() if not n.endswith(".jpg"))
            self.assertNotIn("private-name", text)
            self.assertNotIn("C:/private", text)
            destination = root / "restore"
            download.extract_verified(output / record["asset"], destination, record)
            frame = destination / record["segments"][0]["path"] / "frames/0000000.jpg"
            self.assertEqual(frame.read_bytes(), b"test-image-content")
            self.assertEqual(package.export_episode(episode, root / "source", output, "https://example.test"), record)
            record["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                download.extract_verified(output / record["asset"], destination, record)

    def test_rejects_escape_paths_and_unknown_text_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("../file", "/file", "human/episode-a/../../escape", "human/episode-a/C:/file", "human\\episode-a\\file"):
                with self.assertRaises(ValueError):
                    download.safe_member(name, temporary, "episode-a")
            f = Path(temporary) / "poses.csv"
            f.write_text("frame,secret\n0,private\n")
            with self.assertRaises(ValueError):
                package.clean_csv(f, package.POSE_FIELDS)


if __name__ == "__main__":
    unittest.main()
