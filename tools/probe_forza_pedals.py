"""Bounded pedal measurement with the driving worker stopped; keep pad alive."""
import argparse
import json
import math
from pathlib import Path
import socket
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mono_drive.forza import Telemetry

parser = argparse.ArgumentParser()
parser.add_argument('--gas',type=float,default=0.)
parser.add_argument('--brake',type=float,default=0.)
parser.add_argument('--steer',type=float,default=0.)
parser.add_argument('--seconds',type=float,default=1.)
args = parser.parse_args()
if not (0 <= args.gas <= 1 and 0 <= args.brake <= 1 and abs(args.steer) <= .8 and 0 < args.seconds <= 2):
    raise SystemExit('Input outside bounded probe range')
telemetry = Telemetry(20066)
channel = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
address = ('127.0.0.1',20067)
def send(command,park=False):
    channel.sendto(json.dumps(dict(command=command,handbrake=park)).encode(),address)
rows=[]
start=time.monotonic()
origin=None
try:
    while time.monotonic()-start < 2 and telemetry.last is None:
        telemetry.poll()
        time.sleep(.01)
    if not telemetry.fresh():
        raise RuntimeError('No fresh game telemetry')
    origin=telemetry.last['position'][:]
    start=time.monotonic()
    while time.monotonic()-start < args.seconds:
        value=telemetry.poll()
        if not telemetry.fresh() or not value['race'] or value['race_position']:
            break
        distance=math.dist(value['position'],origin)
        rows.append(dict(t=round(time.monotonic()-start,3),distance=distance,**value))
        if distance >= 2. or value['speed'] > 10:
            break
        send([args.steer,args.gas,args.brake])
        time.sleep(.02)
finally:
    for _ in range(30):
        send([0,0,0],True)
        time.sleep(.02)
    telemetry.poll()
    if telemetry.last:
        rows.append(dict(t=round(time.monotonic()-start,3),distance=math.dist(telemetry.last['position'],origin),**telemetry.last))
    telemetry.close()
    channel.close()
    output=Path('out')/f'pedal-probe-{time.time_ns()}.json'
    output.write_text(json.dumps(dict(args=vars(args),rows=rows)))
    print(json.dumps(dict(file=str(output),samples=[{k:r.get(k) for k in
        ('t','distance','forward_kmh','gear','accel','brake','handbrake')} for r in rows[::max(1,len(rows)//6)]])))
