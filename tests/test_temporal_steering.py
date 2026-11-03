import unittest
import numpy as np
from mono_drive.temporal_steering import window_indices,targets,sequence_arrays,metrics,accept
from mono_drive.forza_temporal import bounded_preview,TemporalSteering
from mono_drive.forza_scene import ScenePlan


class TemporalTests(unittest.TestCase):
    def test_windows_never_cross_time_gaps_or_invalid_telemetry(self):
        t=np.arange(40)*.2;s=np.full(40,50.);u=np.zeros(40);age=np.zeros(40)
        good=window_indices(t,s,u,age);self.assertGreater(len(good),15)
        t[20:]+=2;bad=window_indices(t,s,u,age)
        self.assertTrue(all(not (i-9<=19 and i+3>=20) for i in bad))
        age[4]=100;self.assertNotIn(10,window_indices(t,s,u,age))
    def test_labels_are_future_average_not_future_input(self):
        u=np.arange(30)/30.
        np.testing.assert_allclose(targets(u,12),[np.mean(u[12:15]),np.mean(u[13:16])])
    def test_future_scene_and_controls_cannot_leak_into_input(self):
        f=np.zeros((30,256));speed=np.full(30,50.);u=np.arange(30)/30.
        x,e,y=sequence_arrays(f,speed,u,12)
        f[13:]=100;u[12:]=-1;speed[13:]=200
        xx,ee,yy=sequence_arrays(f,speed,u,12)
        np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(e,ee)
        self.assertFalse(np.array_equal(y,yy))
    def test_dense_source_preserves_real_history_and_forecast_times(self):
        times=np.arange(80)*.1;features=np.repeat(times[:,None],256,axis=1)
        speed=np.full(80,60.);steer=np.arange(80)/100.
        good=window_indices(times,speed,steer,np.zeros(80),history=18,future=6,sample_hz=10.)
        self.assertIn(30,good)
        x,ego,label=sequence_arrays(features,speed,steer,30,stride=2)
        np.testing.assert_allclose(x[:,0],np.arange(14,31,2)*.1)
        np.testing.assert_allclose(ego[:,1],steer[np.arange(12,29,2)])
        np.testing.assert_allclose(label,[np.mean(steer[[30,32,34]]),np.mean(steer[[32,34,36]])])
        steer[30:]=-1;features[31:]=100
        xx,ee,_=sequence_arrays(features,speed,steer,30,stride=2)
        np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(ego,ee)
    def test_expanded_training_keeps_validation_fixed_and_separate(self):
        from mono_drive.human_recorder import partition_ranges
        old=partition_ranges(1800);dense=partition_ranges(1800,True)
        self.assertEqual(old[1],dense[1])
        self.assertGreater(dense[0][2],old[0][2])
        self.assertGreaterEqual(dense[1][1]-dense[0][2],8.)
    def test_preview_never_changes_speed_or_crosses_lane(self):
        z=np.array([3.,6.,9.,12.,16.]);p=ScenePlan(np.column_stack((z,np.zeros(5))),0.,60.,.95,1,0,'paint_pair',1.,
            lane_edges=tuple((x,-2.,2.) for x in z))
        q,reason=bounded_preview(p,.02,50)
        self.assertEqual(reason,'applied');self.assertLessEqual(abs(q.steer),.025);self.assertEqual(q.limit,60.)
        q,reason=bounded_preview(p,.8,50);self.assertIs(q,p);self.assertEqual(reason,'disagreement')
        p.inferred=True;q,reason=bounded_preview(p,.02,50);self.assertIs(q,p)
    def test_gate_requires_temporal_gain_and_turn_accuracy(self):
        base=dict(samples=1000,active_samples=200,mae=.1,active_mae=.15,bias=0,delta_mae=.1)
        self.assertTrue(accept(dict(base,mae=.09),base))
        self.assertFalse(accept(dict(base,mae=.09,active_mae=.2),base))
        self.assertFalse(accept(dict(base,mae=.09,samples=20),base))
        self.assertFalse(accept(dict(base,mae=float('nan')),base))
        self.assertFalse(accept(dict(base,delta_mae=None),base))
    def test_validated_preview_is_held_between_inference_ticks(self):
        observer=TemporalSteering();observer.enabled=True;observer.mode='trial';observer.models=(None,None,{'accepted_offline':True})
        observer.next_check=100;observer.next_frame=10.2;observer.last_prediction=(10.,.02)
        z=np.array([3.,6.,9.,12.,16.]);p=ScenePlan(np.column_stack((z,np.zeros(5))),0.,60.,.95,1,0,'paint_pair',1.,
            lane_edges=tuple((x,-2.,2.) for x in z))
        ego={'car_ordinal':3819,'gear':1,'speed':50.}
        q=observer.update(None,ego,p,{},10.1)
        self.assertGreater(q.steer,0)
        ego['gear']=0;self.assertIs(observer.update(None,ego,p,{},10.11),p)
