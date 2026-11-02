"""Review each completed training result locally, without interrupting training."""
from pathlib import Path
from .forza_evaluation import group_recordings
from .forza_local_learning import read_json, write_json, candidate_verdict
from .forza_model_policy import model_path, offer


def review_completed(root, now):
    root=Path(root).resolve();out=root/'out'
    experiment=read_json(out/'learning-experiment-state.json',{})
    candidate=experiment.get('candidate')
    if not candidate:return
    previous=read_json(out/'vehicle-candidate-review.json',{})
    if previous.get('candidate')==candidate:return
    activation=read_json(out/'model-activation.json',{})
    if activation.get('state')=='trial':return
    baseline=experiment.get('baseline')
    active=read_json(root/'configs/forza_horizon6.json',{}).get('models',{}).get('pathnet')
    if not baseline or not active:return
    if model_path(root,active)!=model_path(root,baseline):return
    evaluation=group_recordings(experiment.get('evaluation',{}),root)
    vehicle=read_json(root/'configs/active_vehicle.json',{}).get('car_ordinal')
    if not isinstance(vehicle,int):return
    accepted,reason=candidate_verdict(evaluation,baseline,candidate,vehicle)
    if activation.get('candidate')==candidate and activation.get('state')=='rolled_back':
        accepted,reason=False,'previous_live_trial_failed'
    saved=out/(Path(candidate).stem+'_vehicle_evaluation.json')
    write_json(saved,evaluation)
    offered=accepted and offer(root,candidate,baseline,evaluation,vehicle)
    record=dict(time=now,candidate=candidate,baseline=baseline,vehicle_ordinal=vehicle,
                accepted=accepted,offered=bool(offered),reason=reason,evaluation=str(saved),
                local_only=True)
    write_json(out/'vehicle-candidate-review.json',record)
    with (out/'vehicle-candidate-reviews.jsonl').open('a',encoding='utf-8') as handle:
        import json
        handle.write(json.dumps(record)+'\n')
