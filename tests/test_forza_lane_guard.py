import unittest
from dataclasses import replace
import numpy as np
from mono_drive.forza_scene import ScenePlan
from mono_drive.forza_lane_guard import LaneDepartureGuard


def plan(steer=0.,left=-1.75,right=1.75):
    points = np.array([[4.,0.],[6.,0.],[8.,0.],[10.,0.]])
    points[:,1]=(left+right)/2.
    return ScenePlan(points,steer,50.,.9,1,0,'paint_pair',.9,
                     lane_edges=tuple((z,left,right) for z in points[:,0]))


class LaneGuardTests(unittest.TestCase):
    def test_safe_lag_aware_prediction_is_not_overridden_by_instant_arc(self):
        original=replace(plan(.14),guard_state='predictive',guard_margin_m=.5)
        guarded=LaneDepartureGuard().apply(original,60.)
        self.assertEqual(guarded.guard_state,'predictive_clear')
        self.assertEqual(guarded.steer,original.steer)
        self.assertEqual(guarded.limit,original.limit)
        for margin in (-.2,float('nan')):
            rejected=LaneDepartureGuard().apply(replace(original,guard_margin_m=margin),60.)
            self.assertNotEqual(rejected.guard_state,'predictive_clear')
        inferred=LaneDepartureGuard().apply(replace(original,inferred=True),60.)
        self.assertNotEqual(inferred.guard_state,'predictive_clear')

    def test_revalidated_consensus_route_uses_its_safe_predictive_steering(self):
        original=replace(plan(.14),guard_state='predictive',guard_margin_m=.5,
                         inferred=True,verified_route=True,reason='route_memory',limit=60.)
        result=LaneDepartureGuard().apply(original,40.)
        self.assertEqual(result.guard_state,'predictive_clear')
        self.assertEqual(result.limit,60.)
        self.assertEqual(result.steer,.14)
        for changed in (replace(original,verified_route=False),
                        replace(original,reason='remembered_avoidance'),
                        replace(original,guard_margin_m=-.3)):
            self.assertNotEqual(LaneDepartureGuard().apply(changed,40.).guard_state,'predictive_clear')

    def test_small_alignment_error_uses_continuous_reserve_limit(self):
        limits=[]
        for shift in (.35,.30,.25,.20):
            original=replace(plan(left=-1.4+shift,right=1.4+shift),limit=60.,road_limit=60.)
            guarded=LaneDepartureGuard().apply(original,40.)
            limits.append(guarded.limit)
            if guarded.guard_state=='centering':
                self.assertGreaterEqual(guarded.guard_margin_m,.12)
                self.assertGreaterEqual(guarded.steer,0.)
        self.assertGreater(max(limits),12.)
        self.assertLessEqual(max(limits),60.)
        self.assertEqual(limits,sorted(limits))
        self.assertLess(max(np.diff(limits)),20.)

    def test_drivable_narrow_lane_does_not_force_walking_speed(self):
        original=replace(plan(left=-1.15,right=1.15),limit=60.,road_limit=60.)
        guarded=LaneDepartureGuard().apply(original,60.)
        self.assertEqual(guarded.limit,60.)
        self.assertEqual(guarded.road_limit,60.)
        self.assertEqual(guarded.guard_state,'centering')
        self.assertGreaterEqual(guarded.guard_margin_m,.12)
        self.assertAlmostEqual(guarded.steer,0.,places=2)

    def test_low_reserve_does_not_allow_high_speed_lane_jump(self):
        original=replace(plan(left=.3,right=3.8),limit=60.,road_limit=60.)
        guarded=LaneDepartureGuard().apply(original,60.)
        self.assertLessEqual(guarded.limit,12.)
        self.assertNotEqual(guarded.guard_state,'centering')

    def test_wall_margin_does_not_move_a_clear_nominal_centre(self):
        original=plan()
        clipped=replace(original,lane_edges=tuple((z,-1.4,1.75) for z in original.path[:,0]))
        result=LaneDepartureGuard().apply(clipped,30.)
        self.assertAlmostEqual(result.steer,0.,places=2)

    def test_keeps_margin_and_aims_toward_lane_centre(self):
        guarded=LaneDepartureGuard().apply(plan(-.05,left=-1.45,right=2.05),30.)
        self.assertGreater(guarded.steer,0.)
        self.assertGreaterEqual(guarded.guard_margin_m,.35)

    def test_straight_lane_keeps_gentle_centre_input(self):
        guarded = LaneDepartureGuard().apply(plan(),30.)
        self.assertAlmostEqual(guarded.steer,0.,places=2)
        self.assertEqual(guarded.guard_state,'clear')
        self.assertGreater(guarded.guard_margin_m,.5)

    def test_hard_input_is_limited_in_both_directions(self):
        guard = LaneDepartureGuard()
        for steer in (-.6,.6):
            guarded = guard.apply(plan(steer),30.)
            self.assertLess(abs(guarded.steer),.1)
            self.assertEqual(guarded.guard_state,'correcting')
            self.assertEqual(guarded.limit,50.)

    def test_near_edge_correction_is_away_from_edge(self):
        guard = LaneDepartureGuard()
        left = guard.apply(plan(-.2,left=-1.04,right=2.46),12.)
        right = guard.apply(plan(.2,left=-2.46,right=1.04),12.)
        self.assertGreaterEqual(left.steer,0.)
        self.assertLessEqual(right.steer,0.)

    def test_no_room_corrects_and_slows_instead_of_crossing_a_boundary(self):
        guard = LaneDepartureGuard()
        self.assertEqual(guard.apply(plan(left=-.6,right=.6),20.).guard_state,'recentering')
        self.assertLessEqual(guard.apply(plan(left=-.6,right=.6),20.).limit,12.)
        self.assertEqual(guard.apply(plan(left=2,right=-2),20.).guard_state,'invalid_boundary')

    def test_unmeasured_lane_does_not_claim_protection(self):
        guarded = LaneDepartureGuard().apply(replace(plan(),lane_edges=()),20.)
        self.assertEqual(guarded.guard_state,'boundary_unknown')
        self.assertEqual(guarded.limit,50.)

    def test_sidewalk_is_not_a_map_heading_lane(self):
        from mono_drive.forza_scene import fallback_lane_edges
        from tests.test_forza_scene import straight_road
        self.assertEqual(fallback_lane_edges(straight_road(-1.1,1.1)),())
        edges=fallback_lane_edges(straight_road(-1.8,5.2))
        self.assertGreater(len(edges),2)
        self.assertLessEqual(max(right-left for _,left,right in edges),3.6)

    def test_evasive_request_is_checked_against_full_car_width(self):
        guarded=LaneDepartureGuard().apply(replace(plan(.6),reason='static_avoidance'),35.)
        self.assertLess(abs(guarded.steer),.1)
        self.assertGreaterEqual(guarded.guard_margin_m,.12)


if __name__ == '__main__':
    unittest.main()
