import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from mono_drive.human_data_audit import REQUIRED,inspect_csv,estimate_sync


class HumanDataAuditTests(unittest.TestCase):
    def test_human_partitions_cannot_overlap_or_touch(self):
        from tools.prepare_human_driving import checked_ranges
        self.assertEqual(checked_ranges(dict(path_train_ranges=[[5,40]],path_validation_ranges=[[50,70]]),100),([[5,40]],[[50,70]]))
        with self.assertRaises(ValueError):checked_ranges(dict(path_train_ranges=[[5,40]],path_validation_ranges=[[42,70]]),100)
    def write(self,path,rows):
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=REQUIRED);writer.writeheader();writer.writerows(rows)

    def rows(self):
        return [dict(zip(REQUIRED,[i*.02,i*20,2000,-i*.2,2,5,0,0,10,36,i%10/10,0,(i%5-2)/2,2])) for i in range(100)]

    def test_checks_ranges_and_position_continuity(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.csv';rows=self.rows();self.write(path,rows)
            result,_,_=inspect_csv(path)
            self.assertEqual(result['valid_rows'],100)
            self.assertAlmostEqual(result['median_hz'],50)
            self.assertAlmostEqual(result['distance_m'],19.8)
            rows[20]['position_x']=500
            rows[50]['throttle']=2
            self.write(path,rows);result,_,_=inspect_csv(path)
            self.assertGreater(result['teleport_steps'],0)
            self.assertEqual(result['out_of_range_rows'],1)

    def test_zero_data_and_reversed_clock_are_not_silently_accepted(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.csv';rows=self.rows()
            for r in rows:r['rpm']=0
            self.write(path,rows);result,_,_=inspect_csv(path)
            self.assertEqual(result['valid_rows'],0)
            rows[3]['elapsed_sec']=0
            self.write(path,rows)
            with self.assertRaises(ValueError):inspect_csv(path)

    def test_video_offset_sign_is_explicit_and_estimate_does_not_auto_verify(self):
        t=np.linspace(0,20,2001)
        columns=dict(elapsed_sec=t,speed_kmh=3*t+.2)
        review=dict(speed_anchors=[dict(video_s=x,hud_kmh=int(3*(x+1)+.2)) for x in (3,6,10)])
        result=estimate_sync(columns,review)
        self.assertLess(abs(result['offset_s']-1),.2)
        self.assertFalse(result['verified'])
        self.assertIn('video_pts_s + offset_s',result['equation'])
