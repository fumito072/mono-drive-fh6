import unittest
from tools.build_forza_route_memory import bridge_is_verified


class RejectNone:
    def rejects(self,t):return False


class RouteBridgeTests(unittest.TestCase):
    def rows(self):
        return [dict(t=i*.2,position=[0.,0.,float(i)],state='driving',
                     surface=dict(road_support=.8),mode='lane_inferred') for i in range(40)]

    def test_only_a_continuous_road_traversal_can_join_measured_sections(self):
        rows=self.rows()
        self.assertTrue(bridge_is_verified(rows,RejectNone()))
        rows[20]['surface']['path_enters_sidewalk']=True
        self.assertFalse(bridge_is_verified(rows,RejectNone()))
        rows=self.rows();rows[20]['position']=[200.,0.,200.]
        self.assertFalse(bridge_is_verified(rows,RejectNone()))
        rows=self.rows();rows[-1]['t']=25.
        self.assertFalse(bridge_is_verified(rows,RejectNone()))


if __name__=='__main__':unittest.main()
