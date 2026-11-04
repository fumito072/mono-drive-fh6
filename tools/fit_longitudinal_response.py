"""Fit bounded throttle response on clean WRX logs, holding out whole runs."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def samples(path,ordinal):
    rows=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            r=json.loads(line)
        except ValueError:
            continue
        if (r.get('car_ordinal')!=ordinal or not r.get('armed') or r.get('gear',0)<1
                or r.get('recovery_phase') or not 12<r.get('speed_kmh',0)<65
                or abs(r.get('actual_steer',0))>25 or r.get('actual_brake',0)>2
                or r.get('urgent_brake') or (r.get('surface') or {}).get('road_support',0)<.7):
            rows.append(None);continue
        rows.append(r)
    result=[]
    for before,after in zip(rows,rows[1:]):
        if before is None or after is None:continue
        dt=after['t']-before['t']
        if not .04<dt<.5:continue
        speed=(before['speed_kmh']+after['speed_kmh'])/7.2
        acceleration=(after['speed_kmh']-before['speed_kmh'])/(3.6*dt)
        if not -2.5<acceleration<3.5:continue
        pos0,pos1=before.get('position'),after.get('position')
        if not pos0 or not pos1:continue
        distance=np.linalg.norm(np.array(pos1)[[0,2]]-np.array(pos0)[[0,2]])
        if distance<.15 or distance>speed*dt*1.8+.3:continue
        grade=(pos1[1]-pos0[1])/distance
        if abs(grade)>.12:continue
        throttle=(before.get('actual_accel',0)+after.get('actual_accel',0))/510.
        result.append((throttle,speed,acceleration+9.81*grade))
    return np.asarray(result,dtype=float).reshape(-1,3)


def fit(rows):
    u,v,a=rows.T
    x=np.column_stack((u,-np.ones(len(u)),-v*v/400.))
    coefficients=np.array([4.,.16,0.])
    low=np.array([2.,0.,0.]);high=np.array([10.,.8,3.2])
    for _ in range(200):
        for i in range(3):
            residual=a-x@coefficients+x[:,i]*coefficients[i]
            coefficients[i]=np.clip(x[:,i]@residual/(x[:,i]@x[:,i]+1e-6),low[i],high[i])
    return coefficients


def evaluate(rows,coefficients):
    u,v,a=rows.T
    prediction=coefficients[0]*u-coefficients[1]-coefficients[2]*v*v/400.
    return dict(mae=float(np.mean(abs(prediction-a))),rmse=float(np.sqrt(np.mean((prediction-a)**2))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--car',type=int,default=3819);p.add_argument('--max-runs',type=int,default=80)
    args=p.parse_args();runs=[]
    for folder in sorted((ROOT/'out').glob('fh6_*'),reverse=True):
        path=folder/'decisions.jsonl'
        if not path.exists() or path.stat().st_size<50000:continue
        data=samples(path,args.car)
        if len(data)>=15:runs.append((folder.name,data))
        if len(runs)>=args.max_runs:break
    runs.sort(key=lambda r:r[0]);cut=max(1,int(len(runs)*.75))
    train=np.concatenate([r[1] for r in runs[:cut]]) if runs else np.empty((0,3))
    test=np.concatenate([r[1] for r in runs[cut:]]) if len(runs)>cut else np.empty((0,3))
    report=dict(car_ordinal=args.car,accepted=False,train_samples=len(train),test_samples=len(test),
                training_runs=[r[0] for r in runs[:cut]],held_out_runs=[r[0] for r in runs[cut:]],
                method='bounded linear acceleration fit, slope corrected, whole-run holdout')
    if len(train)>=500 and len(test)>=100 and len(runs[cut:])>=3:
        coefficients=fit(train);old=evaluate(test,np.array([4.,.16,0.]));new=evaluate(test,coefficients)
        report.update(baseline=old,candidate=new,parameters=dict(gain=float(coefficients[0]),drag_constant=float(coefficients[1]),
            drag_speed_squared=float(coefficients[2]/400.)),throttle_span=float(np.ptp(train[:,0])),speed_span_kmh=float(np.ptp(train[:,1])*3.6))
        report['accepted']=bool(new['mae']<old['mae']*.9 and new['rmse']<old['rmse']*.95
                                and report['throttle_span']>.2 and report['speed_span_kmh']>25.)
    else:report['reason']='insufficient_clean_held_out_runs'
    target=ROOT/'out'/'longitudinal-calibration.json';target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_runs')},indent=2))


if __name__=='__main__':main()
