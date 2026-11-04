"""Vehicle-scoped feedforward from held-out, slope-corrected drive logs."""
import json
import math
from pathlib import Path


class LongitudinalResponse:
    def __init__(self,root=None):
        self.root=Path(root) if root else Path(__file__).resolve().parents[1]
        self.ordinal=None
        self.parameters=None

    def load(self,ordinal):
        self.ordinal=ordinal;self.parameters=None
        if type(ordinal) is not int or ordinal<=0:return
        try:
            report=json.loads((self.root/f'models/vehicle_profiles/longitudinal_{ordinal}.json').read_text(encoding='utf-8'))
            p=report['parameters'];values=[p['gain'],p['drag_constant'],p['drag_speed_squared']]
            valid=(report.get('accepted') is True and report.get('car_ordinal')==ordinal
                and report.get('train_samples',0)>=500 and report.get('test_samples',0)>=100
                and len(report.get('held_out_runs',[]))>=3
                and all(type(v) in (int,float) and math.isfinite(v) for v in values)
                and 2<=values[0]<=10 and 0<=values[1]<=.8 and 0<=values[2]<=.008
                and report['candidate']['mae']<report['baseline']['mae']*.9
                and report['candidate']['rmse']<report['baseline']['rmse']*.95)
            if valid:self.parameters=tuple(values)
        except (OSError,ValueError,KeyError,TypeError):pass

    def feedforward(self,acceleration,speed_kmh,ordinal=None):
        if ordinal!=self.ordinal:self.load(ordinal)
        if self.parameters is None:return .04+acceleration/4.
        gain,drag,quadratic=self.parameters
        speed=max(0.,min(65.,float(speed_kmh)))/3.6
        return min(.6,max(0.,(acceleration+drag+quadratic*speed*speed)/gain))
