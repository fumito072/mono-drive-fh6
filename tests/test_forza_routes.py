import json
from pathlib import Path
from tempfile import TemporaryDirectory
import random
import unittest
import numpy as np
from mono_drive.forza_routes import RouteRotation,paved_arrival


class RouteTests(unittest.TestCase):
    def setup_pool(self,root):
        (root/'out').mkdir();(root/'configs').mkdir()
        frame=root/'source.jpg';frame.write_bytes(b'evidence')
        config=dict(enabled=True,interval_seconds=900,destinations=[dict(id=str(i),name=str(i),
            source_frame=str(frame),anchor=dict(map_image=str(frame),map_point=[100+i*100,200],position=[i*1000,0,0])) for i in range(4)])
        (root/'configs/free_roam_routes.json').write_text(json.dumps(config))
        return RouteRotation(root,random.Random(3))

    def test_rotation_avoids_recent_roads_and_survives_restart(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);pool=self.setup_pool(root);chosen=[]
            for n in range(4):
                road=pool.choose(1000+n*901,0)
                chosen.append(road['id']);pool.succeeded(road,[1,2,3],1000+n*901,0)
            self.assertGreaterEqual(len(set(chosen)),3)
            self.assertTrue(all(a!=b for a,b in zip(chosen,chosen[1:])))
            new=RouteRotation(root,random.Random(3))
            self.assertIsNone(new.choose(3800,0))
            self.assertIsNotNone(new.choose(3800,2))
            self.assertIsNotNone(new.choose(3800,0,force=True))

    def test_failed_destination_cools_down_and_missing_evidence_is_ignored(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);pool=self.setup_pool(root)
            road=pool.choose(1000,0);pool.failed(road,1000,'map cannot locate road')
            self.assertIsNone(pool.choose(1001,0))
            self.assertNotEqual(pool.choose(1400,0)['id'],road['id'])
            (root/'source.jpg').unlink()
            self.assertIsNone(pool.choose(1500,0,force=True))

    def test_resume_grace_defers_scheduled_route_but_allows_explicit_request(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);pool=self.setup_pool(root)
            pool.resumed(1000)
            self.assertIsNone(pool.choose(1001,0))
            self.assertIsNotNone(pool.choose(1001,0,force=True))
            self.assertIsNone(RouteRotation(root).choose(1299,0))
            self.assertIsNotNone(pool.choose(1300,0))

    def test_arrival_needs_fresh_paved_footprint(self):
        self.assertFalse(paved_arrival(None,10))
        good=dict(captured=9.9,road=np.full((216,384),255,np.uint8))
        self.assertTrue(paved_arrival(good,10))
        self.assertFalse(paved_arrival(good,11))
        self.assertFalse(paved_arrival(dict(good,road=np.zeros((216,384),np.uint8)),10))

    def test_verified_arrival_replaces_approximate_destination(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);pool=self.setup_pool(root)
            road=pool.choose(1000,0)
            pool.succeeded(road,[10,20,30],1000,0,[50,60])
            pool.succeeded(pool.choose(2000,0),[40,50,60],2000,0)
            pool.state['recent']=[]
            for item in ['0','1','2','3']:
                if item!=road['id']:pool.state.setdefault('failed_until',{})[item]=9000
            selected=pool.choose(3000,0,force=True)
            self.assertEqual(selected['anchor']['position'],[10,20,30])
            self.assertEqual(selected['anchor']['map_point'],[50,60])
            self.assertEqual(selected['anchor']['arrival_radius_m'],120.)
