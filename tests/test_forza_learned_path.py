from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from mono_drive.forza_learned_path import LearnedPathPrior
from mono_drive.forza_scene import ScenePlan


class LearnedPathTests(unittest.TestCase):
    def setUp(self):
        self.z=np.array([4.,6.,8.,12.,16.,20.,24.,28.])
        self.plan=ScenePlan(np.column_stack((self.z,np.zeros(8))),0.,60.,.9,1,0,'paint_pair',.9,
                            lane_edges=tuple((z,-1.75,1.75) for z in self.z))
        self.scene=dict(road=np.full((216,384),255,np.uint8))

    def prediction(self,x):
        return SimpleNamespace(confidence=.9,reverse=False,path=dict(F=self.z,R=x,sd=np.full(8,.3)))

    def test_learned_bend_contributes_only_inside_lane_and_far_from_car(self):
        prior=LearnedPathPrior()
        result=prior.apply(self.plan,self.prediction(.0009*self.z**2),self.scene,60.)
        self.assertTrue(prior.status['applied'])
        np.testing.assert_array_equal(result.path[:3],self.plan.path[:3])
        self.assertGreater(result.path[-1,1],.1)
        self.assertLessEqual(np.max(abs(result.path[:,1])),.2)
        self.assertEqual(result.limit,60.)

    def test_model_lateral_bias_does_not_move_car_off_lane_midpoint(self):
        result=LearnedPathPrior().apply(self.plan,self.prediction(np.full(8,.4)),self.scene,60.)
        np.testing.assert_allclose(result.path[:,1],0.)

    def test_other_lane_bad_predictions_and_offroad_are_rejected(self):
        cases=[(self.prediction(np.full(8,3.5)),self.scene),
               (self.prediction(np.full(8,float('nan'))),self.scene),
               (self.prediction(.0009*self.z**2),dict(road=np.zeros((216,384),np.uint8)))]
        for model,scene in cases:
            prior=LearnedPathPrior()
            self.assertIs(prior.apply(self.plan,model,scene,60.),self.plan)
            self.assertFalse(prior.status['applied'])
        for reason in ('intersection_turn','sidewalk_boundary','known_failed_approach'):
            plan=replace(self.plan,reason=reason)
            self.assertIs(LearnedPathPrior().apply(plan,self.prediction(np.zeros(8)),self.scene,60.),plan)

    def test_uncertain_model_keeps_geometric_lane_plan(self):
        model=self.prediction(np.zeros(8));model.path['sd'][:]=2.
        self.assertIs(LearnedPathPrior().apply(self.plan,model,self.scene,60.),self.plan)


if __name__=='__main__':unittest.main()
