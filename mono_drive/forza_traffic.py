"""Shared time-based traffic prediction for path selection and pedal control."""
from dataclasses import dataclass
import math
from .forza_speed import MAX_CRUISE_KMH
import numpy as np


@dataclass
class TrafficPrediction:
    limit: float = MAX_CRUISE_KMH
    risk: float = 0.
    collision_s: float | None = None
    vehicle_id: object = None
    closing_mps: float = 0.
    gap_m: float = 50.
    same_direction: bool = False
    lateral_closing: bool = False


def path_x(path, distance):
    """Continue the measured tangent, and connect the near path to the car."""
    z,x = np.asarray(path,dtype=float).T
    result = np.interp(distance,np.r_[0.,z],np.r_[0.,x])
    if len(z) > 1:
        result = np.where(distance > z[-1],x[-1]+(distance-z[-1])*(x[-1]-x[-2])/(z[-1]-z[-2]),result)
    return result


def vehicle_velocity(car):
    # Fresh tracks do not have enough evidence to turn detector jitter into motion.
    reliable = car.get('motion_reliable',car.get('moving', 'moving' not in car))
    if not reliable:
        return 0.,0.
    return float(car.get('vx',0.)),float(car.get('vz',0.))


def same_direction_motion(car,vx,vz):
    reliable=car.get('motion_reliable',car.get('moving',False))
    return bool(reliable and vz>1. and abs(vx)<max(1.2,.3*vz))


def collision_extents(car):
    """Combined ego/object half extents, with a small geometric reserve.

    These are conservative class priors, not measured object dimensions. A
    bicycle beside the kerb must not occupy the full width/length of a car.
    """
    half_width,half_length = {
        'bicycle': (.35,1.), 'motorcycle': (.45,1.1), 'person': (.35,.4),
        'bus': (1.3,6.), 'truck': (1.3,4.),
    }.get(car.get('kind'),(.975,2.195))
    return .925+half_width+.15, 2.205+half_length


def predict_traffic(path,cars,speed_kmh,*,minimum_probe_kmh=20.):
    result = TrafficPrediction()
    speed = max(0.,float(speed_kmh)/3.6)
    probe = max(speed,minimum_probe_kmh/3.6)
    times = np.linspace(.1,4.,40)
    for car in cars:
        # A side-window observation blocks lateral motion, not forward travel.
        if car.get('view') == 'side_left' or car.get('confidence',1.) < .3:
            continue
        if car.get('x') is None or car.get('z') is None:
            continue
        x,z = float(car['x']),float(car['z'])
        if not math.isfinite(x+z) or z < -5:
            continue
        tentative=('track_frames' in car and not car.get('motion_reliable') and
                   (car['track_frames']<3 or car.get('track_age_s',0.)<.2))
        if tentative and not (0.<z<6. and abs(x)<1.2 and car.get('confidence',0)>.75):
            continue
        vx,vz = vehicle_velocity(car)
        lateral_extent,longitudinal_extent=collision_extents(car)
        same=same_direction_motion(car,vx,vz)
        # Use actual relative motion for cars moving in our direction, including
        # alongside/just-passed cars. A faster car cutting across our path is
        # still checked; merely moving forward is not enough to ignore it.
        local_probe=speed if same else probe
        local_z=local_probe*times
        local_x=path_x(path,local_z)
        probe_dx=x+vx*times-local_x
        probe_dz=z+(vz-local_probe)*times
        lateral_closing=bool(np.any((np.abs(probe_dx)<max(0.,abs(x)-.4)) &
                                   (np.abs(probe_dx)<lateral_extent) &
                                   (np.abs(probe_dz)<longitudinal_extent+.1)))
        if same and vz>=speed-.5 and not lateral_closing:
            continue
        dx,dz=probe_dx,probe_dz
        risk = float(np.max(np.exp(-.5*(dx/(lateral_extent/1.864))**2-
                                   .5*(dz/(longitudinal_extent*.75))**2)))
        result.risk = max(result.risk,risk)
        overlaps = (np.abs(dx) < lateral_extent) & (np.abs(dz) < longitudinal_extent)
        if not np.any(overlaps):
            continue
        collision = float(times[np.flatnonzero(overlaps)[0]])
        if result.collision_s is None or collision < result.collision_s:
            result.collision_s,result.vehicle_id = collision,car.get('id')
            result.closing_mps=max(0.,speed-vz)
            result.gap_m=max(0.,z-longitudinal_extent+.4)
            result.same_direction=same
            result.lateral_closing=lateral_closing
        closing = local_probe-vz
        if z >= 0 and closing > .3:
            # Relative braking distance: a moving lead car is not a fixed wall.
            lead = max(0.,vz)
            gap = max(0.,z-longitudinal_extent-.6)
            safe = min(lead+gap/1.4,math.sqrt(lead*lead+4.*gap))
            if z < longitudinal_extent+.1 or collision <= .65:
                safe = lead if same and not lateral_closing else 0.
            result.limit = min(result.limit,max(0.,safe*3.6))
        elif collision <= 1. and (not same or lateral_closing):
            result.limit = 0.
        elif same:
            continue
        else:
            # Crossing traffic: change arrival time only when the paths overlap.
            result.limit = min(result.limit,max(0.,speed*(collision-.5)/2.)*3.6)
    return result
