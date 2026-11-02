"""Conservative use of video-language labels for rejecting training examples."""
import bisect
import json
import math


class RejectedVideoIntervals:
    def __init__(self,path):
        intervals=[]
        incident_path=path.with_name('incident-rejections.jsonl')
        if incident_path.exists():
            for line in incident_path.read_text(encoding='utf-8').splitlines():
                try:
                    value=json.loads(line)
                    start,end=float(value['start']),float(value['end'])
                    if value.get('validated') is True and math.isfinite(start+end) and 0.<end-start<=30.:
                        intervals.append((start,end))
                except (ValueError,KeyError,TypeError):
                    continue
        human_path=path.with_name('human-scene-labels.jsonl')
        if human_path.exists():
            for line in human_path.read_text(encoding='utf-8').splitlines():
                try:
                    value=json.loads(line)
                    if value.get('human_verified') is True and value.get('lane_position') in ('wrong_lane','offroad'):
                        stamp=float(value['time'])
                        intervals.append((stamp-2.,stamp+2.))
                except (ValueError,KeyError,TypeError):
                    continue
        if path.exists():
            with path.open(encoding='utf-8') as handle:
                lines=handle.readlines()
            for line in lines:
                try:
                    value=json.loads(line)
                    bad=value.get('lane_position') in ('offroad','wrong_lane','left_edge','right_edge') or value.get('collision_risk') is True
                    if value.get('corroborated') is True and value.get('confidence',0)>=.85 and (bad or value.get('scene') in ('menu','loading')):
                        intervals.append((float(value['clip_start'])-.3,float(value['clip_end'])+.3))
                except (ValueError,KeyError,TypeError):
                    continue
        self.intervals=[]
        for start,end in sorted(intervals):
            if self.intervals and start<=self.intervals[-1][1]:
                self.intervals[-1]=(self.intervals[-1][0],max(end,self.intervals[-1][1]))
            else:
                self.intervals.append((start,end))
        self.starts=[a for a,b in self.intervals]

    def rejects(self,timestamp):
        index=bisect.bisect_right(self.starts,timestamp)-1
        return index>=0 and timestamp<=self.intervals[index][1]
