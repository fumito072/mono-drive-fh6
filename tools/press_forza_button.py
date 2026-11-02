"""Send a bounded button press through the dedicated virtual controller."""
import argparse
import json
import socket
import time

parser = argparse.ArgumentParser()
parser.add_argument('button',choices=['A','B','X','Y','START','BACK','LEFT_THUMB','RIGHT_THUMB',
    'LEFT_SHOULDER','RIGHT_SHOULDER','DPAD_UP','DPAD_DOWN','DPAD_LEFT','DPAD_RIGHT'])
args = parser.parse_args()
with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as channel:
    address = ('127.0.0.1',20067)
    channel.settimeout(1.)
    channel.sendto(b'{"hello":true}',address)
    if not json.loads(channel.recv(1024)).get('ready'):
        raise RuntimeError('Virtual controller unavailable')
    report = json.dumps(dict(command=[0,0,0],buttons=[args.button])).encode()
    for _ in range(8):
        channel.sendto(report,address)
        time.sleep(.02)
    channel.sendto(b'{"command":[0,0,0]}',address)
