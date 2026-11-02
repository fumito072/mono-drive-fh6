import unittest
import numpy as np
from mono_drive.forza_obstacle_evidence import ground_contacts,StaticEvidence
from mono_drive.forza_obstacle_evidence import remove_dynamic_footprints
from mono_drive.forza_collision import avoid_in_lane
from mono_drive.forza_scene import ScenePlan
from mono_drive.forza_vision import visible_detection


class ObstacleEvidenceTests(unittest.TestCase):
    def test_overhead_silhouette_is_not_projected_as_occupied_road(self):
        obstacle=np.zeros((216,384),np.uint8)
        obstacle[85:110,180:220]=255
        ground=np.zeros_like(obstacle)
        ground[110:130,:]=255
        roi=np.zeros_like(obstacle,dtype=bool)
        roi[75:130,100:350]=True
        contacts=ground_contacts(obstacle,ground,roi)
        self.assertEqual(contacts[95,200],0)
        self.assertEqual(contacts[109,200],255)

    def test_hidden_foot_of_close_wall_is_not_ignored(self):
        obstacle=np.full((216,384),255,np.uint8)
        roi=np.zeros_like(obstacle,dtype=bool)
        roi[75:130,100:350]=True
        result=ground_contacts(obstacle,np.zeros_like(obstacle),roi)
        self.assertEqual(result[129,200],255)
        self.assertEqual(result[100,200],0)

    def test_single_flicker_cannot_produce_lateral_avoidance_evidence(self):
        mask=np.zeros((216,384),np.uint8)
        mask[115:122,190:200]=255
        evidence=StaticEvidence()
        self.assertFalse(evidence.update(mask,1.).any())
        self.assertTrue(evidence.update(mask,1.2).any())
        self.assertFalse(evidence.update(mask,2.).any())

    def test_single_yolo_box_does_not_turn_the_car(self):
        path=np.array([[6.,0.],[10.,0.],[14.,0.],[20.,0.]])
        plan=ScenePlan(path,0.,50.,.9,1,1,'paint_pair',.9,
                       lane_edges=tuple((z,-1.75,1.75) for z in path[:,0]))
        car=dict(x=1.8,z=12.,vx=0.,vz=0.,track_frames=1,confidence=.5)
        self.assertEqual(avoid_in_lane(plan,[car],20.,np.full((216,384),255,np.uint8)).steer,0.)

    def test_box_mostly_on_cabin_is_not_a_road_object(self):
        visible=np.zeros((216,384),bool)
        visible[100:105,200:220]=True
        self.assertIsNone(visible_detection((500,400,1500,850),1920,1080,visible,np.zeros_like(visible)))

    def test_moving_vehicle_is_not_also_a_static_obstacle(self):
        raw=np.full((216,384),255,np.uint8)
        car=dict(box=[.4,.45,.6,.6],motion_reliable=True,moving=True,confidence=.9)
        cleaned=remove_dynamic_footprints(raw,[car])
        self.assertEqual(cleaned[115,190],0)
        self.assertEqual(cleaned[115,50],255)
        car['moving']=False
        self.assertEqual(remove_dynamic_footprints(raw,[car])[115,190],255)


if __name__=='__main__':
    unittest.main()
