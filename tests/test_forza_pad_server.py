import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock,patch
from mono_drive import forza_pad_server
from mono_drive.forza_background import AnalogGamepad
import time


class ControllerHostTests(unittest.TestCase):
    def test_dead_udp_client_and_locked_status_do_not_remove_the_device(self):
        for locked in (False,True):
            with self.subTest(locked=locked),TemporaryDirectory() as directory:
                root=Path(directory)
                (root/'out').mkdir()
                channel=MagicMock()
                pad=MagicMock()
                calls=0
                def receive(_):
                    nonlocal calls
                    calls+=1
                    if calls==1:
                        raise ConnectionResetError(10054,'Previous UDP client closed')
                    if calls==2:
                        return json.dumps(dict(command=[.1,.2,0.])).encode(),('127.0.0.1',12345)
                    (root/'out/controller-host.stop').write_text('Test finished')
                    raise socket.timeout()
                channel.recvfrom.side_effect=receive
                with patch('mono_drive.forza_pad_server.socket.socket',return_value=channel), \
                     patch('mono_drive.forza_pad_server.DirectGamepad',return_value=pad), \
                     patch.object(Path,'replace',side_effect=PermissionError('Reader lock') if locked else None):
                    self.assertEqual(forza_pad_server.main(root),0)
                self.assertEqual(calls,3)
                self.assertTrue(any(c.args[0]==(.1,.2,0.) for c in pad.set.call_args_list))
                pad.close.assert_called_once()
                channel.close.assert_called_once()

    def test_client_ignores_an_asynchronous_udp_reset(self):
        client=AnalogGamepad.__new__(AnalogGamepad)
        client.sock=MagicMock()
        client.sock.recvfrom.side_effect=ConnectionResetError(10054,'Old UDP error')
        client.json=json
        client.address=('127.0.0.1',20067)
        client.last_ack=client.last_probe=time.monotonic()
        client.tick=False
        client.set((0.,.2,0.))
        sent=json.loads(client.sock.sendto.call_args.args[0])
        self.assertEqual(sent['command'][1],.2)


if __name__=='__main__':
    unittest.main()
