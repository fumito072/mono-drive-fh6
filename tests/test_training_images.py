import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import sys
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_pathnet import load_items


class TrainingImagesTests(unittest.TestCase):
    def test_byte_preparation_is_identical_to_the_float_round_trip(self):
        from mono_drive.pathnet import net_input,net_input_u8
        rng=np.random.default_rng(18)
        for width,height in [(576,360),(800,600),(1920,1080)]:
            image=rng.integers(0,256,(height,width,3),dtype=np.uint8)
            expected=(net_input(image)*255.).round().astype(np.uint8)
            np.testing.assert_array_equal(net_input_u8(image),expected)

    def test_parallel_preparation_preserves_pixels_labels_and_temporal_split(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'frames').mkdir()
            (root/'camera.json').write_text(json.dumps(dict(name='test',width=576,height=360,
                hfov_deg=90,pitch_deg=3,height_m=1.15,fwd_m=1.,right_m=-.35)))
            with (root/'poses.csv').open('w',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['frame','t','x','y','z','fwd_x','fwd_y','fwd_z','right_x','right_y','right_z'])
                for i in range(40):
                    cv2.imwrite(str(root/'frames'/f'{i:07d}.jpg'),np.full((360,576,3),i*5,np.uint8))
                    writer.writerow([i,i*.2,0,0,i*2,0,0,1,1,0,0])
            before=load_items([str(root)],None,1,.2,workers=1)
            after=load_items([str(root)],None,1,.2,workers=4)
            self.assertTrue(before[0]);self.assertTrue(before[1])
            self.assertEqual([len(x) for x in before],[len(x) for x in after])
            for old_part,new_part in zip(before,after):
                for old,new in zip(old_part,new_part):
                    for x,y in zip(old,new):np.testing.assert_array_equal(x,y)
