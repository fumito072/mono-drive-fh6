"""Local CUDA road segmentation and traffic detection, independent of controller ownership."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import time
import hashlib
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'nvidia/segformer-b2-finetuned-cityscapes-1024-1024'
HORIZON = .435


def ground_point(u, v):
    from .forza_projection import ground_point as shared_ground_point
    return shared_ground_point(u,v)


def visible_detection(box,width,height,visibility,side_left):
    x1,y1,x2,y2 = box
    frame_wide=(x2-x1)/width>.55 and (y2-y1)/height>.5
    if y2/height < .38 or y1/height > .70 or x2 <= x1 or y2 <= y1:
        return None
    if y1/height < .15 and (x2-x1)/width > .5:
        return None  # Fixed cockpit/frame-wide detections are not traffic.
    x1,x2 = np.clip([int(x1/width*384),int(x2/width*384)+1],0,384)
    y1,y2 = np.clip([int(y1/height*216),int(y2/height*216)+1],0,216)
    visible = visibility[y1:y2,x1:x2]
    side = side_left[y1:y2,x1:x2]
    front_count = np.count_nonzero(visible & ~side)
    side_count = np.count_nonzero(visible & side)
    if max(front_count,side_count) < 12:
        return None
    if (front_count+side_count)/max(1,visible.size)<.22:
        return None
    if front_count>=side_count:
        # A box whose alleged ground contact is inside the dashboard cannot
        # supply obstacle depth. This also rejects a cockpit-wide false car.
        contact=visibility[max(y1,y2-3):y2,x1:x2] & ~side_left[max(y1,y2-3):y2,x1:x2]
        if frame_wide and (not contact.size or float(np.mean(contact))<.2):return None
    return 'side_left' if side_count > front_count else 'front'


class TrafficTracks:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def box_iou(first,second):
        a,b=np.asarray(first),np.asarray(second)
        size=np.maximum(0.,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
        intersection=float(np.prod(size))
        union=float(np.prod(np.maximum(0.,a[2:]-a[:2]))+np.prod(np.maximum(0.,b[2:]-b[:2])))-intersection
        return intersection/max(1e-8,union)

    @staticmethod
    def compatible_kind(first,second):
        vehicles={'car','truck','bus'}
        return first==second or (first in vehicles and second in vehicles)

    def update(self, detections, ego, now):
        output, used = [], set()
        yaw = float(ego.get('yaw', 0))
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, s], [-s, c]])
        origin = np.array(ego.get('position', [0, 0, 0]))[[0, 2]]
        for d in detections:
            if d.get('view') == 'side_left':
                output.append(dict(**d,id='side_left',moving=False,trail=[]))
                continue
            x1,y1,x2,y2 = d['box']
            point = ground_point((x1+x2)/2,y2)
            if point is None or point[1] > 80:
                continue
            world = rotation@point+origin
            matches = [(np.linalg.norm(t['world']+t['velocity']*(now-t['time'])-world), key) for key,t in self.tracks.items()
                       if key not in used and now-t['time'] < .8 and self.compatible_kind(t['detection'].get('kind'),d.get('kind'))]
            distance,key = min(matches, default=(1e9,None))
            if distance > max(2., point[1]*.08):
                overlaps=[(self.box_iou(t['detection']['box'],d['box']),k) for k,t in self.tracks.items()
                          if k not in used and now-t['time']<.45 and self.compatible_kind(t['detection'].get('kind'),d.get('kind'))]
                overlap,matched=max(overlaps,default=(0.,None))
                if overlap>=.3:
                    key=matched
                    old=self.tracks[key]
                else:
                    key = self.next_id
                    self.next_id += 1
                    old = None
            else:
                old = self.tracks[key]
            used.add(key)
            velocity = np.zeros(2)
            count = 1
            history = []
            widths = []
            if old:
                count = old['count']+1
                history = old['history'][:]
                widths = old.get('widths',[])[:]
            history.append((now,world.tolist()))
            history = [(t,p) for t,p in history if now-t < 2]
            widths.append((now,max(.001,x2-x1)))
            widths = [(t,w) for t,w in widths if now-t <= .8]
            recent = [(t,p) for t,p in history if now-t <= .8]
            slopes = [(np.array(b)-a)/(tb-ta) for i,(ta,a) in enumerate(recent)
                      for tb,b in recent[i+1:] if tb-ta >= .16]
            slopes = [v for v in slopes if np.linalg.norm(v) < 65]
            span = recent[-1][0]-recent[0][0]
            residual = 0.
            if slopes:
                velocity = np.median(slopes,axis=0)
                fitted = np.array([np.array(p)-velocity*(t-now) for t,p in recent])
                residual = float(np.median(np.linalg.norm(fitted-np.median(fitted,axis=0),axis=1)))
            reliable = bool(len(recent) >= 3 and span >= .25 and slopes and residual < max(.6,point[1]*.03))
            trusted_velocity=velocity.copy() if reliable else old.get('trusted_velocity') if old else None
            velocity_time=now if reliable else old.get('velocity_time',-100.) if old else -100.
            source='measured' if reliable else 'unresolved'
            if (not reliable and old and trusted_velocity is not None and now-velocity_time<.6
                    and self.box_iou(old['detection']['box'],d['box'])>=.2):
                velocity=np.asarray(trusted_velocity).copy()
                reliable=True
                source='recent_track'
            expansion = 0.
            if len(widths) >= 3 and widths[-1][0]-widths[0][0] >= .25:
                samples = np.array(widths)
                times = samples[:,0]-samples[:,0].mean()
                expansion = float(np.dot(times,np.log(samples[:,1]))/max(1e-6,np.dot(times,times)))
            ttc = min(10.,1/expansion) if reliable and expansion > .1 else 100.
            self.tracks[key] = dict(world=world,time=now,velocity=velocity,count=count,history=history,
                                    widths=widths,detection=d,trusted_velocity=trusted_velocity,velocity_time=velocity_time)
            local_velocity = rotation.T@velocity
            speed = float(np.linalg.norm(velocity))
            # A single displaced box is never evidence that a vehicle is moving.
            moving = bool(reliable and count >= 5 and (span >= .5 or source=='recent_track') and speed > 1.5)
            trail = []
            if moving and local_velocity[1] > 1.:
                trail = [(rotation.T@(np.array(p)-origin)).tolist() for _,p in history]
            output.append(dict(**d,id=key,x=float(point[0]),z=float(point[1]),
                               vx=float(local_velocity[0]),vz=float(local_velocity[1]),
                               relative_vx=float(local_velocity[0]),
                               relative_vz=float(local_velocity[1])-float(ego.get('speed',0))/3.6,
                               closing_mps=float(ego.get('speed',0))/3.6-float(local_velocity[1]),
                               motion_reliable=reliable,motion_residual_m=residual,
                               motion_source=source,
                               track_age_s=float(span),track_frames=count,expansion_rate=expansion,
                               moving=moving,trail=trail,ttc=ttc))
        self.tracks = {k:v for k,v in self.tracks.items() if now-v['time'] < 1.5}
        return output


class LocalVision:
    def __init__(self):
        import torch
        from transformers import SegformerForSemanticSegmentation
        from ultralytics import YOLO
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU is required by the configured local perception worker')
        self.torch = torch
        self.seg = SegformerForSemanticSegmentation.from_pretrained(MODEL, local_files_only=True).to('cuda').eval()
        self.detector = YOLO(str(ROOT/'yolo11s.pt'))
        self.mean = torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
        self.std = torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        self.tracks = TrafficTracks()
        from .forza_projection import GroundCalibration
        self.calibration=GroundCalibration()
        self.projection=self.calibration.update(None)
        from .forza_obstacle_evidence import StaticEvidence
        self.static_evidence=StaticEvidence()
        from .forza_local_learning import read_json
        active=read_json(ROOT/'configs/active_vehicle.json',{})
        visibility_file = ROOT/active.get('visibility_file','models/fh6_driver_visibility.npz')
        with np.load(visibility_file,allow_pickle=False) as data:
            self.visibility = data['visible'].astype(bool)
            self.side_left = data['side_left'].astype(bool)
            self.planning_roi = data['planning_roi'].astype(bool)
            self.fixed_cabin = data['fixed_dark'].astype(bool)
        from huggingface_hub import snapshot_download
        revision = Path(snapshot_download(MODEL,local_files_only=True)).name
        self.manifest = dict(segmentation=MODEL,segmentation_revision=revision,
                             vehicle=active.get('name'),car_ordinal=active.get('car_ordinal'),
                             detector='YOLO11s',torch=torch.__version__,cuda=torch.version.cuda,
                             detector_policy='class-agnostic-nms-relative-motion-v3',
                             road_surface_policy='explicit-sidewalk-mask-v1',
                             projection_policy='shared-lane-calibration-v2',
                             obstacle_evidence='ground-contact-temporal-v2',
                             device=torch.cuda.get_device_name(0),
                             visibility_sha256=hashlib.sha256(visibility_file.read_bytes()).hexdigest())

    def infer(self, frame, ego, captured):
        torch = self.torch
        height,width = frame.shape[:2]
        cut = round(height*.60)
        crop = frame[:cut]
        rgb = cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),(896,512))
        pixels = torch.from_numpy(rgb.copy()).to(device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
            logits = self.seg(pixel_values=(pixels-self.mean)/self.std).logits
            all_probability = logits.float().softmax(1)
            probability = all_probability[:,0:1]
            sidewalk_probability=torch.nn.functional.interpolate(all_probability[:,1:2],
                size=(130,384),mode='bilinear',align_corners=False)
            sidewalk=(sidewalk_probability[0,0]*255).byte().cpu().numpy()
            static_probability = all_probability[:,2:9].sum(1,keepdim=True)
            static_probability = torch.nn.functional.interpolate(static_probability,size=(130,384),mode='bilinear',align_corners=False)
            fixed = (static_probability[0,0]*255).byte().cpu().numpy()
            ground_probability=all_probability[:,0:2].sum(1,keepdim=True)+all_probability[:,9:10]
            ground_probability=torch.nn.functional.interpolate(ground_probability,size=(130,384),mode='bilinear',align_corners=False)
            ground=(ground_probability[0,0]*255).byte().cpu().numpy()
            probability = torch.nn.functional.interpolate(probability,size=(130,384),mode='bilinear',align_corners=False)
            road = (probability[0,0]*255).byte().cpu().numpy()
        full_road = np.zeros((216,384),dtype=np.uint8)
        full_road[:130] = road
        # HUD, side window and cockpit cannot provide forward driving space.
        full_road[:round(216*.34)] = 0
        full_road[:, :round(384*.25)] = 0
        full_road[~self.planning_roi] = 0
        self.sidewalk=np.zeros_like(full_road)
        self.sidewalk[:130]=sidewalk
        self.sidewalk[~self.planning_roi]=0
        self.obstacles = np.zeros_like(full_road)
        self.obstacles[:130] = fixed
        self.obstacles[~self.planning_roi] = 0
        self.raw_obstacles=self.obstacles.copy()
        full_ground=np.zeros_like(full_road)
        full_ground[:130]=ground
        from .forza_obstacle_evidence import ground_contacts
        self.obstacles=ground_contacts(self.obstacles,full_ground,self.planning_roi)
        # Preserve semantic evidence for painted boundaries: headlights and white
        # lane markings must not disappear in the separate snow-brightness veto.
        self.raw_road = full_road.copy()
        from .forza_scene import painted_corridor
        self.projection=self.calibration.update(painted_corridor(frame,self.raw_road))
        small = cv2.resize(frame,(384,216),interpolation=cv2.INTER_AREA)
        value = cv2.cvtColor(small,cv2.COLOR_BGR2HSV)[:,:,2]
        near = (full_road > 160) & (np.indices(full_road.shape)[0] > 110) & (value > 8)
        if np.count_nonzero(near) > 100:
            dark_road = float(np.percentile(value[near],25))
            night_scene = float(np.percentile(value[:74,170:],80)) < 100
            if dark_road < 65 and night_scene:
                # Night snow banks are much brighter than the adjacent asphalt.
                bright = cv2.blur(value,(5,5)) > dark_road+40
                full_road[bright] = 0
        # Include the lower side glass; avoid using the dashboard as a road cue.
        detection_crop = frame[:round(height*.80)].copy()
        glass=cv2.resize(self.visibility.astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST)
        detection_crop[glass[:len(detection_crop)]==0]=0
        results = self.detector(detection_crop,device=0,imgsz=960,classes=[0,1,2,3,5,7],conf=.30,
                                agnostic_nms=True,verbose=False)[0]
        detections = []
        for box in results.boxes.data.cpu().numpy():
            x1,y1,x2,y2,confidence,cls = [float(v) for v in box]
            view = visible_detection((x1,y1,x2,y2),width,height,self.visibility,self.side_left)
            if view is None:
                continue
            detections.append(dict(box=[x1/width,y1/height,x2/width,y2/height],
                                   confidence=confidence,kind=self.detector.names[int(cls)],
                                   view=view))
        cars = self.tracks.update(detections,ego,captured)
        from .forza_obstacle_evidence import remove_dynamic_footprints
        self.obstacles=remove_dynamic_footprints(self.obstacles,cars)
        self.confirmed_obstacles=self.static_evidence.update(self.obstacles,captured)
        return full_road,cars


def overlay(frame, road, cars):
    out = frame.copy()
    mask = cv2.resize(road,(frame.shape[1],frame.shape[0])) > 140
    out[mask] = (out[mask]*.55+np.array([30,220,40])*.45).astype(np.uint8)
    for car in cars:
        x1,y1,x2,y2 = car['box']
        p1,p2 = (int(x1*out.shape[1]),int(y1*out.shape[0])),(int(x2*out.shape[1]),int(y2*out.shape[0]))
        cv2.rectangle(out,p1,p2,(0,160,255),3)
        cv2.putText(out,f"{car['kind']} {car['confidence']:.2f} #{car['id']}",p1,cv2.FONT_HERSHEY_SIMPLEX,.6,(0,160,255),2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    parser.add_argument('--out',default='out/vision-check.jpg')
    parser.add_argument('--hz',type=float,default=8.)
    parser.add_argument('--handover-pid',type=int)
    args = parser.parse_args()
    vision=capture=None
    if args.handover_pid:
        # Warm the replacement while the old process keeps publishing frames.
        # Only the verified module PID is stopped at the final handover.
        vision=LocalVision()
        from .forza_background import GameCapture
        capture=GameCapture()
        deadline=time.monotonic()+4.
        frame=None
        while frame is None and time.monotonic()<deadline:
            frame,captured=capture.grab()
            if frame is None:
                time.sleep(.05)
        if frame is None:
            capture.close()
            raise RuntimeError('No fresh game frame; old perception left running')
        try:
            ego=json.loads((ROOT/'out/ego.json').read_text())
        except (OSError,ValueError):
            ego={}
        vision.infer(frame,ego,captured)
        from .forza_local_monitor import stop_owned
        if not stop_owned(args.handover_pid,'mono_drive.forza_vision'):
            # The old capture can exit while the replacement is warming up.
            # Proceed only when its listening port is already free; never
            # terminate an unverified/reused PID to force the handover.
            from .forza_session import port_busy
            if port_busy(20068):
                capture.close()
                raise RuntimeError('Could not verify old perception PID; handover cancelled')
    guard = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    guard.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
    try:
        guard.bind(('127.0.0.1',20068))
    except OSError:
        if capture:
            capture.close()
        return 0
    vision = vision or LocalVision()
    if args.image:
        frame = cv2.imread(args.image)
        road,cars = vision.infer(frame,{},time.monotonic())
        cv2.imwrite(args.out,overlay(frame,road,cars))
        print(json.dumps(cars),flush=True)
        return 0
    from .forza_background import GameCapture
    capture = capture or GameCapture()
    output = ROOT/'out'/'scene.npz'
    temp = ROOT/'out'/'scene.tmp.npz'
    last_preview = 0.
    run = ROOT/'out'/time.strftime('perception_%Y%m%d_%H%M%S')
    run.mkdir(parents=True,exist_ok=False)
    (run/'frames').mkdir()
    (run/'masks').mkdir()
    (run/'sidewalks').mkdir()
    (run/'manifest.json').write_text(json.dumps(dict(**vision.manifest,training_label_status='unreviewed',
        started_utc=time.time(),capture='Windows Graphics Capture, game window only',mask_frame_hz=2),indent=2))
    log = (run/'perception.jsonl').open('w',encoding='utf-8')
    sequence,next_record = 0,0.
    print('Local CUDA perception ready: SegFormer B2 + YOLO11s',flush=True)
    try:
        while not (ROOT/'out'/'vision.stop').exists():
            started = time.monotonic()
            frame,captured = capture.grab()
            if frame is None:
                time.sleep(.05)
                continue
            try:
                ego = json.loads((ROOT/'out'/'ego.json').read_text())
            except (OSError,ValueError):
                ego = {}
            road,cars = vision.infer(frame,ego,captured)
            duration = time.monotonic()-started
            # Publish the exact image used by segmentation. Mixing a newer
            # camera frame with an older road mask moved boundaries in curves.
            preview = cv2.resize(frame,(960,540),interpolation=cv2.INTER_AREA)
            _,image_jpeg = cv2.imencode('.jpg',preview,[cv2.IMWRITE_JPEG_QUALITY,88])
            frame_name = mask_name = sidewalk_name = None
            if started >= next_record:
                frame_name = f'frames/{sequence:07d}.jpg'
                mask_name = f'masks/{sequence:07d}.png'
                sidewalk_name = f'sidewalks/{sequence:07d}.png'
                cv2.imwrite(str(run/frame_name),frame,[cv2.IMWRITE_JPEG_QUALITY,90])
                cv2.imwrite(str(run/mask_name),road)
                cv2.imwrite(str(run/sidewalk_name),vision.sidewalk)
                sequence += 1
                next_record = started+.5
            log.write(json.dumps(dict(captured_s=captured,wall_time=time.time(),inference_ms=duration*1000,
                frame=frame_name,road_mask=mask_name,sidewalk_mask=sidewalk_name,cars=cars,ego=ego,
                projection=vision.projection,training_label_status='unreviewed'))+'\n')
            np.savez_compressed(temp,road=road,raw_road=vision.raw_road,sidewalk=vision.sidewalk,obstacles=vision.obstacles,
                                planning_roi=vision.planning_roi,
                                confirmed_obstacles=vision.confirmed_obstacles,raw_obstacles=vision.raw_obstacles,
                                cars=json.dumps(cars),captured=captured,
                                image_jpeg=image_jpeg,ego=json.dumps(ego),
                                projection=json.dumps(vision.projection),
                                inference_ms=duration*1000,ego_time=ego.get('time',0),recording=str(run.name),
                                frame=frame_name or '',mask=mask_name or '')
            try:
                os.replace(temp,output)
            except PermissionError:
                pass
            if started-last_preview > 1.:
                cv2.imwrite(str(ROOT/'out'/'vision.jpg'),overlay(frame,road,cars))
                print(f'GPU perception {duration*1000:.0f}ms, {len(cars)} vehicles',flush=True)
                last_preview = started
                log.flush()
            time.sleep(max(0,1/args.hz-(time.monotonic()-started)))
    finally:
        capture.close()
        guard.close()
        log.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
