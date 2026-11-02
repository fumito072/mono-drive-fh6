"""Use the learned future path as a bounded prior inside the selected lane."""
from dataclasses import replace
import numpy as np
from .forza_lateral import lookahead_target
from .forza_scene import footprint_quality, inside_lane, project


class LearnedPathPrior:
    def __init__(self):
        self.status = dict(applied=False, reason='waiting_for_prediction')
        self.last_path = None

    def apply(self, plan, model, scene, speed):
        self.status = dict(applied=False, reason='ineligible_plan')
        self.last_path = None
        if (plan is None or not plan.lane_edges or plan.limit <= 0 or
                plan.reason in ('intersection_turn', 'lane_alignment', 'road_reentry',
                                'offroad_surface', 'sidewalk_boundary', 'known_failed_approach')):
            return plan
        predicted = getattr(model, 'path', None)
        confidence = float(getattr(model, 'confidence', 0.))
        if not predicted or not np.isfinite(confidence) or confidence < .55 or getattr(model, 'reverse', False):
            self.status['reason'] = 'weak_model'
            return plan
        try:
            forward, right, uncertainty = (np.asarray(predicted[k], dtype=float) for k in ('F', 'R', 'sd'))
            if (forward.ndim != 1 or right.shape != forward.shape or uncertainty.shape != forward.shape or
                    len(forward) < 4 or not np.isfinite([forward, right, uncertainty]).all()):
                raise ValueError('Invalid model path')
            valid = (forward >= 3.) & (forward <= 40.)
            forward, right, uncertainty = forward[valid], right[valid], uncertainty[valid]
            if len(forward) < 4 or np.any(np.diff(forward) <= .2):
                raise ValueError('Folded or short prediction')
            path = np.asarray(plan.path, dtype=float)
            edges = np.asarray(plan.lane_edges, dtype=float)
            z, nominal = path.T
            overlap = (z >= max(6., forward[0])) & (z <= min(28., forward[-1]))
            if np.count_nonzero(overlap) < 3 or np.median(np.interp(z[overlap], forward, uncertainty)) > 1.4:
                raise ValueError('Uncertain or short overlap')
            prediction = np.interp(z, forward, right)
            left = np.interp(z, edges[:, 0], edges[:, 1])
            right_edge = np.interp(z, edges[:, 0], edges[:, 2])
            # A human recording from another lane must never select that lane.
            if np.max(np.abs(prediction[overlap] - nominal[overlap])) > 1.0:
                raise ValueError('Prediction disagrees with selected lane')
            if np.any((prediction[overlap] < left[overlap] + .95) |
                      (prediction[overlap] > right_edge[overlap] - .95)):
                raise ValueError('Prediction outside lane')
            delta = prediction - nominal
            # Preserve the near lane midpoint. Learn bend anticipation, not a
            # driver's constant lateral offset; only adjust the far reference.
            delta -= np.interp(8., z, delta)
            weight = np.clip((z - 8.) / 16., 0., 1.) * min(.35, confidence * .35)
            adjustment = np.where(overlap, np.clip(weight * delta, -.20, .20), 0.)
            proposed = np.column_stack((z, nominal + adjustment))
            if not inside_lane(proposed, left, right_edge):
                raise ValueError('Full vehicle would leave lane')
            road = scene.get('raw_road', scene['road'])
            floor, mean = footprint_quality(road, proposed)
            if floor < .40 or mean < .65:
                raise ValueError('Road footprint not supported')
            _, _, steer = lookahead_target(proposed, speed)
            self.status = dict(applied=True, reason='verified_lane_prior', confidence=round(confidence, 3),
                               max_adjustment_m=round(float(np.max(np.abs(adjustment))), 4))
            self.last_path = proposed.copy()
            return replace(plan, path=proposed, steer=steer,
                           image_path=tuple(project(float(x), float(d)) for d, x in proposed))
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.status['reason'] = 'prediction_not_supported'
            self.status['detail'] = str(error)
            return plan
