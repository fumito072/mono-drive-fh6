import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from tempfile import TemporaryDirectory
import cv2
import numpy as np
from mono_drive.forza_session import ProgressWatch, PadChannel, ROOT, map_stick, MenuRecognizer
from mono_drive.forza_map_anchor import FixedMapAnchor
from mono_drive.forza_turns import arrow_templates, match_arrow, TurnCoordinator
from mono_drive.forza_static import static_risk


class SessionTests(unittest.TestCase):
    def test_capture_shutdown_failure_still_releases_telemetry(self):
        from mono_drive.forza_session import close_game_resources
        capture,telemetry=MagicMock(),MagicMock()
        capture.close.side_effect=RuntimeError('capture already stopped')
        with TemporaryDirectory() as folder,patch('mono_drive.forza_session.OUT',Path(folder)):
            close_game_resources(capture,telemetry)
            self.assertTrue((Path(folder)/'recovery-cleanup-error.json').exists())
        telemetry.close.assert_called_once()

    def test_read_only_resume_requires_fresh_paved_free_roam_and_sends_no_buttons(self):
        from mono_drive.forza_session import observe_resumable_free_roam
        for screen,fresh,paved,expected in [('unknown',True,True,True),
                                           ('pause',True,True,False),
                                           ('unknown',False,True,False),
                                           ('unknown',True,False,False)]:
            with self.subTest(screen=screen,fresh=fresh,paved=paved), TemporaryDirectory() as directory:
                clock=[10.]
                capture,telemetry,recognizer=MagicMock(),MagicMock(),MagicMock()
                capture.grab.side_effect=lambda:(np.zeros((8,8,3),np.uint8),clock[0])
                telemetry.fresh.return_value=fresh
                telemetry.last=dict(race=True,race_position=0,lap_number=0)
                recognizer.classify.return_value=screen
                recognizer.driving_ready.return_value=True
                with patch('mono_drive.forza_session.STOP',Path(directory)/'stop'), \
                     patch('mono_drive.forza_session.port_busy',return_value=False), \
                     patch('mono_drive.forza_session.time.monotonic',side_effect=lambda:clock[0]), \
                     patch('mono_drive.forza_session.time.sleep',side_effect=lambda delay:clock.__setitem__(0,clock[0]+delay)), \
                     patch('mono_drive.forza_session.PadChannel') as pad, \
                     patch('mono_drive.forza_session.MenuRecognizer',return_value=recognizer), \
                     patch('mono_drive.forza_background.GameCapture',return_value=capture), \
                     patch('mono_drive.forza.Telemetry',return_value=telemetry), \
                     patch('mono_drive.forza_routes.paved_arrival',return_value=paved):
                    self.assertEqual(observe_resumable_free_roam(),expected)
                    pad.assert_not_called()
                capture.close.assert_called_once()
                telemetry.close.assert_called_once()

    def test_keyboard_map_is_recognized_without_changing_camera(self):
        self.assertEqual(MenuRecognizer().classify(cv2.imread(str(ROOT/'out/menu-current.jpg'))),'map')
        self.assertEqual(MenuRecognizer().classify(np.zeros((1080,1920,3),np.uint8)),'unknown')
    def test_controller_reconnect_only_acknowledges_observed_dialog_and_pause(self):
        from mono_drive.forza_session import recover_controller_dialog
        import time
        with TemporaryDirectory() as directory:
            capture,recognizer,pad,telemetry=MagicMock(),MagicMock(),MagicMock(),MagicMock()
            frame=np.zeros((1080,1920,3),np.uint8)
            capture.grab.side_effect=lambda:(frame,time.monotonic())
            recognizer.classify.side_effect=['controller_disconnected','controller_disconnected','pause','unknown']
            recognizer.score.return_value=.99
            recognizer.driving_ready.return_value=True
            pad.ready.return_value=True
            telemetry.fresh.return_value=True
            telemetry.last=dict(race=True,race_position=0,lap_number=0,position=[1.,2.,3.])
            with patch('mono_drive.forza_session.OUT',Path(directory)), \
                 patch('mono_drive.forza_session.time.sleep'), \
                 patch('mono_drive.forza_session.cv2.imwrite'), \
                 patch('mono_drive.forza_session.stop_driver_for_menu') as stop, \
                 patch('mono_drive.forza_session.PadChannel',return_value=pad), \
                 patch('mono_drive.forza_session.MenuRecognizer',return_value=recognizer), \
                 patch('mono_drive.forza_background.GameCapture',return_value=capture), \
                 patch('mono_drive.forza.Telemetry',return_value=telemetry):
                self.assertTrue(recover_controller_dialog())
            stop.assert_called_once()
            self.assertEqual([c.args for c in pad.press.call_args_list],[('A',),('B',)])
            telemetry.close.assert_called_once()

    def test_controller_reconnect_does_nothing_on_an_unknown_screen(self):
        from mono_drive.forza_session import recover_controller_dialog
        import time
        capture,recognizer=MagicMock(),MagicMock()
        capture.grab.side_effect=lambda:(np.zeros((1080,1920,3),np.uint8),time.monotonic())
        recognizer.classify.return_value='unknown'
        with patch('mono_drive.forza_session.time.sleep'), \
             patch('mono_drive.forza_session.stop_driver_for_menu') as stop, \
             patch('mono_drive.forza_session.PadChannel') as pad, \
             patch('mono_drive.forza_session.MenuRecognizer',return_value=recognizer), \
             patch('mono_drive.forza_background.GameCapture',return_value=capture):
            self.assertFalse(recover_controller_dialog())
        stop.assert_not_called()
        pad.assert_not_called()

    def test_controller_disconnect_template_is_specific(self):
        recognizer=MenuRecognizer()
        self.assertEqual(recognizer.classify(cv2.imread(str(ROOT/'out/monitor-0740.jpg'))),
                         'controller_disconnected')
        self.assertNotEqual(recognizer.classify(cv2.imread(str(ROOT/'out/monitor-0625.jpg'))),
                            'controller_disconnected')

    def test_hidden_map_cursor_never_activates_a_location_or_moves_without_an_anchor(self):
        from mono_drive.forza_session import restore_road
        for destination,expected_moves in ((None,0),(np.array([900.,800.]),3)):
            with self.subTest(destination=destination), TemporaryDirectory() as directory:
                frame=np.zeros((1080,1920,3),np.uint8)
                capture=MagicMock()
                capture.grab.return_value=(frame,0.)
                pad=MagicMock()
                recognizer=MagicMock()
                recognizer.classify.return_value='map'
                recognizer.map_points.return_value=(None,None)
                telemetry=MagicMock()
                telemetry.last=dict(race_position=0,position=[0.,0.,0.])
                telemetry.fresh.return_value=True
                anchor=MagicMock()
                anchor.locate.return_value=destination
                anchor.search_direction.return_value=None
                with patch('mono_drive.forza_session.OUT',Path(directory)), \
                     patch('mono_drive.forza_session.port_busy',return_value=False), \
                     patch('mono_drive.forza_session.time.sleep'), \
                     patch('mono_drive.forza_session.cv2.imwrite'), \
                     patch('mono_drive.forza_session.PadChannel',return_value=pad), \
                     patch('mono_drive.forza_session.MenuRecognizer',return_value=recognizer), \
                     patch('mono_drive.forza_background.GameCapture',return_value=capture), \
                     patch('mono_drive.forza.Telemetry',return_value=telemetry), \
                     patch('mono_drive.forza_map_anchor.FixedMapAnchor',return_value=anchor):
                    with self.assertRaises(RuntimeError):
                        restore_road()
                self.assertEqual(pad.move_map.call_count,expected_moves)
                pad.press.assert_not_called()
                pad.close.assert_called_once()

    def test_map_cursor_can_be_above_the_pause_menu_tiles(self):
        frame=cv2.imread(str(ROOT/'out/recovery-check-0510.jpg'))
        np.testing.assert_allclose(MenuRecognizer.ring_cursor(frame),[861.,122.],atol=3.)
        overlay=np.full((1080,1920,3),80,np.uint8)
        cv2.circle(overlay,(1610,130),35,(15,15,15),9)
        cv2.circle(overlay,(1610,130),33,(240,240,240),3)
        self.assertIsNone(MenuRecognizer.ring_cursor(overlay))

    def test_varied_route_keeps_direction_when_location_card_hides_cursor(self):
        from mono_drive.forza_session import restore_road
        with TemporaryDirectory() as directory:
            capture,recognizer,pad,telemetry,anchor=[MagicMock() for _ in range(5)]
            capture.grab.return_value=(np.zeros((1080,1920,3),np.uint8),0.)
            recognizer.classify.return_value='map'
            recognizer.map_points.side_effect=[(np.array([900.,800.]),None)]+[(None,None)]*9
            telemetry.last=dict(race_position=0,position=[0.,0.,0.])
            anchor.locate.return_value=np.array([900.,600.])
            with patch('mono_drive.forza_session.OUT',Path(directory)), \
                 patch('mono_drive.forza_session.time.sleep'), \
                 patch('mono_drive.forza_session.cv2.imwrite'), \
                 patch('mono_drive.forza_session.stop_driver_for_menu'), \
                 patch('mono_drive.forza_session.PadChannel',return_value=pad), \
                 patch('mono_drive.forza_session.MenuRecognizer',return_value=recognizer), \
                 patch('mono_drive.forza_background.GameCapture',return_value=capture), \
                 patch('mono_drive.forza.Telemetry',return_value=telemetry), \
                 patch('mono_drive.forza_map_anchor.FixedMapAnchor',return_value=anchor):
                with self.assertRaisesRegex(RuntimeError,'bounded navigation'):
                    restore_road({'name':'new road'})
            self.assertEqual(pad.move_map.call_count,9)
            self.assertTrue(all(call.args[1]>0 for call in pad.move_map.call_args_list))
            pad.press.assert_not_called()

    def test_place_card_does_not_invent_a_cursor(self):
        frame=cv2.imread(str(ROOT/'out/recovery-check-0440.jpg'))
        recognizer=MenuRecognizer()
        self.assertTrue(recognizer.place_is_selected(frame))
        self.assertIsNone(recognizer.map_points(frame)[0])
        card=cv2.imread(str(ROOT/'out/recovery-parking-x.jpg'))
        self.assertTrue(recognizer.place_is_selected(card))
        self.assertIsNone(recognizer.map_points(card)[0])
        self.assertFalse(recognizer.place_is_selected(np.zeros_like(frame)))

    def test_missing_hud_and_menu_are_not_completed_road_recovery(self):
        recognizer=MenuRecognizer()
        self.assertTrue(recognizer.driving_ready(cv2.imread(str(ROOT/'tests/fixtures/fh6/driving-hud.jpg'))))
        self.assertFalse(recognizer.driving_ready(np.zeros((1080,1920,3),np.uint8)))
        self.assertFalse(recognizer.driving_ready(cv2.imread(str(ROOT/'out/recovery-cursor-clear.jpg'))))

    def test_selected_player_card_is_a_known_cursor_state(self):
        frame=cv2.imread(str(ROOT/'out/recovery-check-0425.jpg'))
        recognizer=MenuRecognizer()
        self.assertTrue(recognizer.player_is_selected(frame,[966.5,539.5]))
        self.assertFalse(recognizer.player_is_selected(np.zeros_like(frame),[966.5,539.5]))

    def test_player_selection_survives_a_changed_vehicle_card(self):
        recognizer=MenuRecognizer()
        frame=np.zeros((1080,1920,3),np.uint8)
        name=recognizer.player_card[:58]
        frame[200:258,600:600+name.shape[1]]=cv2.cvtColor(name,cv2.COLOR_GRAY2BGR)
        self.assertTrue(recognizer.player_is_selected(frame,[807.,480.]))
        self.assertFalse(recognizer.player_is_selected(np.full_like(frame,255),[807.,480.]))

    def test_reverse_timing_does_not_relax_forward_frame_requirement(self):
        from mono_drive.forza import control_frame_fresh
        self.assertTrue(control_frame_fresh(.75,reverse=True))
        self.assertFalse(control_frame_fresh(.75))
        self.assertFalse(control_frame_fresh(1.1,reverse=True))

    def test_map_cursor_survives_a_white_road_behind_its_ring(self):
        frame=np.full((1080,1920,3),80,np.uint8)
        cv2.line(frame,(1020,480),(1150,610),(240,240,240),10)
        cv2.circle(frame,(1090,542),35,(15,15,15),9)
        cv2.circle(frame,(1090,542),33,(240,240,240),3)
        np.testing.assert_allclose(MenuRecognizer.ring_cursor(frame),[1090,542],atol=3.)

    def test_map_cursor_can_finish_inside_deadzone_sized_error(self):
        np.testing.assert_allclose(map_stick([-36.,35.]),[-.4,.4])
        np.testing.assert_allclose(map_stick([4.,-8.]),[0.,0.])
        np.testing.assert_allclose(map_stick([4.,-8.],deadzone=2.),[.4,-.4])

    def test_sixty_seconds_of_jitter_counts_as_stuck(self):
        watch=ProgressWatch()
        for t in range(60):
            self.assertFalse(watch.update(t,[t%3,0,0]))
        self.assertTrue(watch.update(60,[0,0,0]))

    def test_forward_progress_does_not_trigger_recovery(self):
        watch=ProgressWatch()
        for t in range(181):
            self.assertFalse(watch.update(t,[t*2,0,0]))

    def test_automated_menus_cannot_change_the_camera(self):
        pad=PadChannel()
        try:
            for button in ('RIGHT_SHOULDER','RIGHT_THUMB'):
                with self.assertRaises(ValueError):
                    pad.press(button)
        finally:
            pad.sock.close()

    def test_fixed_destination_survives_pan_and_zoom(self):
        anchor=FixedMapAnchor(ROOT/'models/ui/fh6/return-point.json')
        matrix=np.array([[.75,0.,150.],[0.,.75,60.]])
        shifted=cv2.warpAffine(anchor.image,matrix,(1920,1080))
        expected=matrix@np.array([*anchor.config['map_point'],1.])
        np.testing.assert_allclose(anchor.locate(shifted),expected,atol=2.)
        np.testing.assert_allclose(anchor.reference_point(expected),anchor.config['map_point'],atol=3.)
        self.assertIsNone(anchor.locate(np.zeros_like(anchor.image)))
        self.assertTrue(anchor.arrived(anchor.config['position']))
        self.assertFalse(anchor.arrived([0,100,0]))

    def test_distant_map_search_uses_world_direction_and_rejects_unknown_position(self):
        anchor=FixedMapAnchor.__new__(FixedMapAnchor)
        anchor.config={'position':[-200.,100.,-6600.]}
        direction=anchor.search_direction([1200.,100.,-200.])
        self.assertLess(direction[0],0.)
        self.assertAlmostEqual(direction[1],.8)
        self.assertIsNone(anchor.search_direction(None))
        self.assertIsNone(anchor.search_direction([float('nan'),0.,0.]))
        self.assertIsNone(anchor.search_direction([-200.,100.,-6600.]))

    def test_arrival_checks_use_destination_kind_not_map_pixel_variable(self):
        from mono_drive.forza_session import restore_road
        import time
        for varied in (False,True):
            with self.subTest(varied=varied), TemporaryDirectory() as directory:
                capture,recognizer,pad,telemetry,anchor=[MagicMock() for _ in range(5)]
                frame=np.zeros((1080,1920,3),np.uint8)
                capture.grab.side_effect=lambda:(frame,time.monotonic())
                recognizer.classify.side_effect=['fast_travel','fast_travel']+['unknown']*100000
                recognizer.driving_ready.return_value=True
                telemetry.last=dict(race=True,race_position=0,lap_number=0,position=[1.,2.,3.])
                telemetry.fresh.return_value=True
                anchor.config={'name':'tested road'};anchor.arrived.return_value=True
                with patch('mono_drive.forza_session.OUT',Path(directory)), \
                     patch('mono_drive.forza_session.time.sleep'), \
                     patch('mono_drive.forza_session.cv2.imwrite'), \
                     patch('mono_drive.forza_session.wait_frame'), \
                     patch('mono_drive.forza_session.stop_driver_for_menu'), \
                     patch('mono_drive.forza_session.PadChannel',return_value=pad), \
                     patch('mono_drive.forza_session.MenuRecognizer',return_value=recognizer), \
                     patch('mono_drive.forza_background.GameCapture',return_value=capture), \
                     patch('mono_drive.forza.Telemetry',return_value=telemetry), \
                     patch('mono_drive.forza_map_anchor.FixedMapAnchor',return_value=anchor), \
                     patch('mono_drive.forza_routes.paved_arrival',return_value=True) as paved:
                    result=restore_road({'name':'tested road'} if varied else None)
                self.assertTrue(result['success'])
                self.assertEqual(result['paved_verified'],varied)
                self.assertEqual(paved.called,varied)

    def test_painted_arrow_classes(self):
        for direction, mask in arrow_templates().items():
            self.assertEqual(match_arrow(mask)[0],direction)

    def test_minimap_branch_alone_cannot_commit_a_turn(self):
        from types import SimpleNamespace
        frame=np.zeros((1080,1920,3),np.uint8)
        scene=dict(road=np.full((216,384),255,np.uint8),cars=[])
        plan=object()
        with patch('mono_drive.forza_minimap.road_hint',return_value=SimpleNamespace(turn_angle=1.5)), \
             patch('mono_drive.forza_turns.road_arrow',return_value=None), \
             patch('mono_drive.forza_turns.road_end',return_value=6.), \
             patch('mono_drive.forza_turns.exit_visible',return_value=True):
            turns=TurnCoordinator()
            self.assertIs(turns.update(frame,scene,dict(yaw=0.),0.,plan),plan)
            self.assertIsNone(turns.started)

    def test_persistent_visible_exit_at_a_road_end_can_turn_without_arrow(self):
        from types import SimpleNamespace
        frame=np.zeros((1080,1920,3),np.uint8)
        scene=dict(road=np.full((216,384),255,np.uint8),cars=[])
        with patch('mono_drive.forza_minimap.road_hint',return_value=SimpleNamespace(turn_angle=-1.5,confidence=.7)), \
             patch('mono_drive.forza_turns.road_arrow',return_value=None), \
             patch('mono_drive.forza_turns.road_end',return_value=6.), \
             patch('mono_drive.forza_turns.exit_visible',return_value=True):
            turns=TurnCoordinator();plan=object()
            self.assertIs(turns.update(frame,scene,dict(yaw=0.),0.,plan),plan)
            self.assertIs(turns.update(frame,scene,dict(yaw=0.),.4,plan),plan)
            result=turns.update(frame,scene,dict(yaw=0.),.9,plan)
            self.assertEqual(result.reason,'intersection_turn');self.assertLess(result.steer,0)

    def test_static_barrier_is_checked_independently_of_traffic(self):
        road=np.full((216,384),255,np.uint8)
        obstacles=np.zeros_like(road)
        obstacles[99:111,150:225]=255
        road[99:111,150:225]=0
        risk=static_risk(dict(road=road,obstacles=obstacles),50.,0.)
        self.assertLess(risk.clearance_m,15.)
        self.assertTrue(risk.emergency)
        self.assertFalse(static_risk(dict(road=np.full_like(road,255),obstacles=np.zeros_like(road)),50.,0.).emergency)


if __name__=='__main__':
    unittest.main()
