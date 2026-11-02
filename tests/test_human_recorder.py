import csv,json,struct
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import unittest
from mono_drive.human_recorder import audit,rebuild_alignment,selected_groups


class NativeRecorderTests(unittest.TestCase):
    def test_pause_frames_do_not_discard_verified_driving(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.fixture(root)
            with (folder/'telemetry.csv').open(newline='') as f:
                reader=csv.DictReader(f);fields=reader.fieldnames+['IsRaceOn'];rows=list(reader)
            raw=bytearray((folder/'telemetry.raw').read_bytes())
            for i,row in enumerate(rows):
                row['IsRaceOn']='0' if 40<=i<52 else '1'
                struct.pack_into('<i',raw,28+i*344,int(row['IsRaceOn']))
            (folder/'telemetry.raw').write_bytes(raw)
            with (folder/'telemetry.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            original=(folder/'aligned.csv').read_bytes()
            pts='\n'.join(str(i*.02) for i in range(120))
            with patch('mono_drive.human_recorder.subprocess.run',return_value=SimpleNamespace(stdout=pts)),patch('mono_drive.human_recorder.inspect_video',return_value={}):
                result=audit(folder,root/'out')
            self.assertTrue(result['path_candidate_eligible'])
            self.assertEqual(result['synchronization']['matched_frames'],108)
            self.assertEqual(result['synchronization']['excluded_frames']['menu_or_loading'],12)
            self.assertEqual((folder/'aligned.csv').read_bytes(),original)

    def test_clock_discontinuity_is_not_repaired_by_guessing(self):
        frames=[dict(frame_index=i,frame_t_ns=i*20_000_000,video_pts_us=i*20_000) for i in range(120)]
        telemetry=[dict(seq=i,receive_t_ns=i*20_000_000,IsRaceOn=1) for i in range(120)]
        frames[70]['video_pts_us']+=10000
        with self.assertRaisesRegex(ValueError,'presentation clock'):
            rebuild_alignment(frames,telemetry,[])

    def test_missing_data_splits_future_trajectory_segments(self):
        rows=[];telemetry={}
        for i in range(51):
            rows.append(dict(frame_index=i,frame_t_ns=i*200_000_000,video_pts_us=i*200000,
                             telemetry_seq=i,valid='0' if i==25 else '1'))
            telemetry[i]=dict(IsRaceOn=1,RacePosition=0,LapNumber=0,speed_kmh=36,TimestampMS=i*200,
                              PositionX=i*2,PositionY=0,PositionZ=0,
                              **{'TireSlipRatio'+wheel:0 for wheel in ('FrontLeft','FrontRight','RearLeft','RearRight')})
        groups=selected_groups(rows,telemetry,0,11,1)
        self.assertEqual([len(group) for group in groups],[25,25])
        self.assertLess(int(groups[0][-1][0]['frame_index']),25)
        self.assertGreater(int(groups[1][0][0]['frame_index']),25)

    def fixture(self,root):
        folder=root/'session';folder.mkdir()
        (folder/'session.json').write_text(json.dumps(dict(status='complete',synthetic_demo=False,errors=[],epoch_unix_ns_approx=1000000000)))
        (folder/'video.mkv').write_bytes(b'video')
        rows=[];raw=bytearray(b'FH6UDP1\n')
        for i in range(120):
            r=dict(seq=i,receive_t_ns=i*20_000_000,TimestampMS=i*20,CurrentEngineRpm=2000,
                PositionX=i*.2,PositionY=2.,PositionZ=5.,VelocityX=0.,VelocityY=0.,VelocityZ=10.,
                speed_kmh=36.,throttle=.2,brake=0.,steer=.1,Gear=1,Accel=51,Steer=13,CarOrdinal=4222)
            rows.append(r);payload=bytearray(324);struct.pack_into('<iI',payload,0,1,i*20)
            struct.pack_into('<3f',payload,244,r['PositionX'],2.,5.);payload[315]=51;payload[320]=13
            raw.extend(struct.pack('<QQI',i,i*20_000_000,324));raw.extend(payload)
        (folder/'telemetry.raw').write_bytes(raw)
        with (folder/'telemetry.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        frames=[dict(frame_index=i,frame_t_ns=i*20_000_000+10_000_000,video_pts_us=i*20_000) for i in range(120)]
        aligned=[dict(**r,telemetry_seq=i,telemetry_t_ns=i*20_000_000,telemetry_age_ms=10.,valid=1) for i,r in enumerate(frames)]
        for name,data in [('frames.csv',frames),('aligned.csv',aligned)]:
            with (folder/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
        return folder

    def test_exact_mapping_and_original_packets_are_required(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.fixture(root)
            pts='\n'.join(str(i*.02) for i in range(120))
            with patch('mono_drive.human_recorder.subprocess.run',return_value=SimpleNamespace(stdout=pts)),patch('mono_drive.human_recorder.inspect_video',return_value={}):
                result=audit(folder,root/'out');self.assertTrue(result['path_candidate_eligible'])
                self.assertEqual(result['synchronization']['matched_frames'],120)
                raw=bytearray((folder/'telemetry.raw').read_bytes());raw[28+315]=0;(folder/'telemetry.raw').write_bytes(raw)
                with self.assertRaisesRegex(ValueError,'Raw control mismatch'):audit(folder,root/'out')

    def test_unmatched_blank_sequence_is_excluded_before_numeric_parsing(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.fixture(root)
            with (folder/'aligned.csv').open(newline='') as f:
                reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
            rows[10].update(valid='0',telemetry_seq='',telemetry_t_ns='',telemetry_age_ms='')
            with (folder/'aligned.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            pts='\n'.join(str(i*.02) for i in range(120))
            with patch('mono_drive.human_recorder.subprocess.run',return_value=SimpleNamespace(stdout=pts)),patch('mono_drive.human_recorder.inspect_video',return_value={}):
                result=audit(folder,root/'out')
                self.assertTrue(result['path_candidate_eligible'])
                self.assertEqual(result['synchronization']['matched_frames'],120)
                self.assertEqual(result['synchronization']['corrected_alignment_rows'],1)

    def test_inactive_zero_vehicle_is_not_a_vehicle_change(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.fixture(root)
            with (folder/'telemetry.csv').open(newline='') as f:
                reader=csv.DictReader(f);fields=reader.fieldnames+['IsRaceOn'];rows=list(reader)
            for row in rows:row['IsRaceOn']='1'
            rows[0].update(IsRaceOn='0',CarOrdinal='0')
            with (folder/'telemetry.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            raw=bytearray((folder/'telemetry.raw').read_bytes());struct.pack_into('<i',raw,28,0);(folder/'telemetry.raw').write_bytes(raw)
            pts='\n'.join(str(i*.02) for i in range(120))
            with patch('mono_drive.human_recorder.subprocess.run',return_value=SimpleNamespace(stdout=pts)),patch('mono_drive.human_recorder.inspect_video',return_value={}):
                result=audit(folder,root/'out')
                self.assertTrue(result['path_candidate_eligible'])
                self.assertEqual(result['car_ordinals'],[4222])
