"""Measured temporal-policy trials; lane estimates are not ground truth."""
import math
import numpy as np


class DrivingMetrics:
    def __init__(self):
        self.previous=None;self.last_t=0.;self.run=None
        self.seconds=self.speed_sum=self.distance=self.near=self.stopped=0.
        self.lane_time=self.centered=self.offset_square=0.
        self.offroad=self.wrongway=self.inferred=0.;self.steer_rates=[];self.pedal_rates=[]
        self.models=set()
    def consume(self,rows,run):
        if run!=self.run:self.previous=None;self.run=run
        for row in rows:
            t=float(row.get('t',0.))
            if t<=self.last_t:continue
            self.last_t=t
            eligible=(row.get('armed') and row.get('state')=='driving' and not row.get('recovery_phase')
                and not row.get('race_position',0) and row.get('car_ordinal')==3819 and row.get('gear',0)>0)
            if not eligible:self.previous=None;continue
            if row.get('path_model'):self.models.add(row['path_model'])
            old=self.previous;self.previous=row
            if not old:continue
            dt=t-float(old['t'])
            if not 0<dt<1.:continue
            speed=float(row.get('speed_kmh',0.))
            if not math.isfinite(speed):continue
            self.seconds+=dt;self.speed_sum+=dt*speed;self.distance+=dt*speed/3.6
            self.near+=dt*(55<=speed<=65);self.stopped+=dt*(speed<2.)
            self.offroad+=dt*bool((row.get('surface') or {}).get('offroad_confirmed'))
            self.wrongway+=dt*bool((row.get('carriageway') or {}).get('wrongway_confirmed'))
            self.inferred+=dt*bool(row.get('inferred_lane'))
            edges=row.get('lane_edges') or []
            if edges and not row.get('inferred_lane') and row.get('confidence',0)>=.75:
                near=min(edges,key=lambda e:abs(e[0]-6.));offset=(near[1]+near[2])/2.
                if math.isfinite(offset):
                    self.lane_time+=dt;self.centered+=dt*(abs(offset)<=.5);self.offset_square+=dt*offset*offset
            a=row.get('sent_command');b=old.get('sent_command')
            if a and b:
                self.steer_rates.append(abs(float(a[0])-float(b[0]))/dt)
                self.pedal_rates.append(max(abs(float(a[1])-float(b[1])),abs(float(a[2])-float(b[2])))/dt)
    def summary(self):
        return {'driving_seconds':self.seconds,'distance_m':self.distance,
            'mean_kmh':self.speed_sum/self.seconds if self.seconds else None,
            'fraction_55_to_65':self.near/self.seconds if self.seconds else None,'stopped_seconds':self.stopped,
            'observed_lane_seconds':self.lane_time,'estimated_centered_fraction':self.centered/self.lane_time if self.lane_time else None,
            'estimated_center_offset_rmse_m':math.sqrt(self.offset_square/self.lane_time) if self.lane_time else None,
            'offroad_seconds':self.offroad,'wrongway_seconds':self.wrongway,'inferred_seconds':self.inferred,
            'steer_rate_p95':float(np.percentile(self.steer_rates,95)) if self.steer_rates else None,
            'pedal_rate_p95':float(np.percentile(self.pedal_rates,95)) if self.pedal_rates else None,
            'path_models':sorted(self.models),
            'measurement_note':'Lane centre is a camera estimate; unknown or inferred lanes do not count as centered.'}


def trial_passed(baseline,trial,applied,incidents):
    required=('driving_seconds','observed_lane_seconds','mean_kmh','estimated_centered_fraction',
              'offroad_seconds','wrongway_seconds','steer_rate_p95','pedal_rate_p95')
    if any(not isinstance(trial.get(k),(float,int)) or not math.isfinite(trial[k]) for k in required):return False
    if any(not isinstance(baseline.get(k),(float,int)) or not math.isfinite(baseline[k]) for k in ('mean_kmh','steer_rate_p95','pedal_rate_p95')):return False
    if applied<30 or incidents or trial['driving_seconds']<120 or trial['observed_lane_seconds']<20:return False
    if baseline.get('path_models')!=trial.get('path_models'):return False
    if trial['offroad_seconds']>.1 or trial['wrongway_seconds']>.1:return False
    if trial['estimated_centered_fraction'] is None or trial['estimated_centered_fraction']<max(.8,(baseline['estimated_centered_fraction'] or 0)-.03):return False
    if trial['mean_kmh']<max(10.,(baseline['mean_kmh'] or 0)-2.):return False
    for key,margin in [('steer_rate_p95',.02),('pedal_rate_p95',.05)]:
        if trial[key] is None or baseline[key] is None or trial[key]>baseline[key]*1.1+margin:return False
    return True


def target_achieved(trial,incidents):
    return bool(not incidents and trial['driving_seconds']>=120 and trial['observed_lane_seconds']>=30 and
        50<=(trial['mean_kmh'] or 0)<=65 and (trial['fraction_55_to_65'] or 0)>=.5 and
        (trial['estimated_centered_fraction'] or 0)>=.85 and trial['offroad_seconds']<=.1 and trial['wrongway_seconds']<=.1)
