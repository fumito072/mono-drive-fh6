"""Loopback-only, read-only live view of local learning and driving logs."""
from __future__ import annotations
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'dashboard'
SERVICE='mono-drive-learning-dashboard'
PHASES={
    'training':'学習中','loading':'学習データを準備中','encoding':'特徴量を計算中',
    'extracting':'動画を読み込み中','preparing':'動画を準備中','evaluating':'検証中',
    'validation':'検証中','validating_with_real_controls':'操作データで検証中',
    'learning_with_weak_motion':'映像の動きを学習中','preparing_image_motion':'映像の動きを解析中',
    'estimating_image_motion':'映像の動きを解析中','running':'実行中','watching':'監視中',
    'waiting_for_new_data':'追加データ待ち','waiting_for_visual_model':'画像モデルの完了待ち',
    'waiting_for_first_driving_evaluation':'先行する評価の完了待ち','complete':'完了',
    'completed':'完了','failed':'エラー','retry_later':'再試行待ち','retrying':'再試行中',
    'needs_attention':'確認が必要','stopped':'停止','stopped_by_user':'停止',
    'resuming_checkpoint':'途中保存から再開中','shadow':'走行映像と予測を照合中',
    'trial':'操舵補正を試験中','active':'採用済み','shadow_only':'照合のみ・採用保留',
    'inconclusive':'判定保留','offline_rejected':'検証条件を満たさず不採用',
    'visual_candidate_rejected':'画像モデルを不採用',
}


def finite(value):
    return float(value) if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) else None


def read_json(path):
    try:
        if path.stat().st_size>4*1024*1024:return {}
        value=json.loads(path.read_text(encoding='utf-8-sig'))
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):return {}


def tail_records(path,maximum_bytes=2*1024*1024):
    try:
        with path.open('rb') as stream:
            stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-maximum_bytes))
            lines=stream.read().decode('utf-8',errors='replace').splitlines()
        for line in lines:
            try:value=json.loads(line)
            except ValueError:continue
            if isinstance(value,dict):yield value
    except OSError:return


def process_alive(pid):
    if not isinstance(pid,int) or pid<=0:return None
    try:
        import psutil
        process=psutil.Process(pid)
        return process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
    except ImportError:return None
    except psutil.AccessDenied:return None
    except psutil.NoSuchProcess:return False


def phase_info(state,now,phase=None):
    phase=phase or state.get('phase',state.get('state','unknown'))
    stamp=finite(state.get('time'));age=max(0.,now-stamp) if stamp is not None else None
    alive=process_alive(state.get('pid',state.get('child_pid')))
    tone='good'
    if phase in ('needs_attention','failed'):tone='danger'
    elif phase in ('stopped','stopped_by_user','shadow_only','inconclusive','offline_rejected','visual_candidate_rejected'):
        tone='warn'
    elif phase in ('retrying','retry_later'):tone='warn'
    if alive is False and phase not in ('complete','completed','stopped','stopped_by_user'):
        label='停止（実行プロセスなし）';tone='warn'
    else:label=PHASES.get(phase,'状態を確認中')
    if age is not None and age>120 and tone=='good' and phase not in ('complete','completed'):
        tone='warn'
    return dict(phase=phase,label=label,tone=tone,age_seconds=age,process_alive=alive)


class DashboardSnapshot:
    def __init__(self,root=ROOT):
        self.root=Path(root).resolve();self.lock=threading.Lock();self.cached=None;self.updated=0.
        self.gpu=None;self.gpu_updated=0.

    def model_directory(self,state):
        default=self.root/'models/temporal_candidates/deep_forecast_20261103'
        try:
            path=Path(state.get('model_dir',default))
            if not path.is_absolute():path=self.root/path
            path=path.resolve()
            return path if path.is_relative_to(self.root/'models') else default
        except (ValueError,TypeError,OSError):return default

    def gpu_status(self,now):
        if now-self.gpu_updated<15:return self.gpu
        self.gpu_updated=now
        try:
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,creationflags=flags,check=True)
            usage,used,total,temperature=[float(x.strip()) for x in result.stdout.splitlines()[0].split(',')]
            if not all(math.isfinite(x) for x in (usage,used,total,temperature)):raise ValueError('Invalid GPU values')
            self.gpu=dict(utilization=usage,used_gib=used/1024,total_gib=total/1024,temperature=temperature,time=now)
        except (OSError,ValueError,subprocess.SubprocessError):self.gpu=None
        return self.gpu

    def collect(self):
        now=time.time();out=self.root/'out'
        watcher=read_json(out/'deep-learning-state.json');directory=self.model_directory(watcher)
        progress=read_json(directory/'training-progress.json');report=read_json(directory/'report.json')
        plan=read_json(out/'deep-learning-plan.json')
        records={}
        for path in (out/'deep-forecast.log',out/'deep-resume.log'):
            for row in tail_records(path):
                if isinstance(row.get('epoch'),int) and isinstance(row.get('validation'),list):records[row['epoch']]=row
        if report.get('phase')=='complete':
            for row in report.get('history',[]):
                if isinstance(row,dict) and isinstance(row.get('epoch'),int):records[row['epoch']]=row
        warmup=int(report.get('baseline_warmup_epochs') or 8)
        history=[]
        for epoch,row in sorted(records.items()):
            validation=row.get('validation',[])
            if not isinstance(validation,list) or len(validation)<2 or not all(isinstance(v,dict) for v in validation[:2]):continue
            history.append(dict(epoch=epoch,
                temporal={k:finite(validation[0].get(k)) for k in ('mae','active_mae','delta_mae')} if epoch>warmup else None,
                single={k:finite(validation[1].get(k)) for k in ('mae','active_mae','delta_mae')},
                loss=[finite(v) for v in row.get('loss',[])[:2]]))
        completed=max(records,default=0)
        if isinstance(progress.get('last'),dict):completed=max(completed,int(progress.get('last',{}).get('epoch',0)))
        current=int(progress.get('epoch') or completed or 0)
        maximum=int(report.get('max_epochs') or progress.get('epochs') or plan.get('maximum_epochs') or 48)
        phase='complete' if report.get('phase')=='complete' else watcher.get('phase',progress.get('phase','unknown'))
        health=phase_info(dict(watcher,pid=watcher.get('pid'),time=watcher.get('time',progress.get('time'))),now,phase)
        stop=any((out/name).exists() for name in ('training.stop','longrun.stop'))
        if stop:health.update(label='停止要求あり',tone='warn',phase='stopped')
        checkpoint=directory/'checkpoint.pt'
        try:checkpoint_time=checkpoint.stat().st_mtime
        except OSError:checkpoint_time=None
        def job(title,state_name,details=''):
            value=read_json(out/state_name)
            return dict(title=title,details=details,**phase_info(value,now),error=str(value.get('error',''))[:240])
        dense=read_json(out/'temporal-worker-state.json')
        jobs=[job('高密度データの時系列学習','temporal-worker-state.json',
                  {'prepare_dense_temporal_video':'動画を高密度で抽出','prepare_temporal_driving':'画像特徴を計算','train_temporal_steering':'操舵モデルを学習'}.get(dense.get('stage'),'')),
              job('操作データのない動画の補助学習','uncontrolled-learning-state.json'),
              job('走行経路モデル','training-state.json'),job('ローカル映像監視','vlm-state.json')]
        drive=read_json(out/'driver-heartbeat.json');session=read_json(out/'longrun-state.json')
        drive_stamp=finite(drive.get('time'));drive_fresh=drive_stamp is not None and 0<=now-drive_stamp<10
        driving=drive_fresh and drive.get('armed') is True and session.get('state') not in ('needs_attention','stopped_by_user','completed')
        drive_label='走行中' if driving and (finite(drive.get('speed_kmh')) or 0)>2 else '停止中'
        if session.get('state')=='needs_attention' or not drive_fresh:drive_label='走行停止・復帰確認が必要'
        evaluation=read_json(out/'temporal-live-evaluation.json')
        return dict(meta=dict(service=SERVICE,generated_at=now,refresh_seconds=1,read_only=True),
            deep=dict(name='4秒の映像履歴モデル',model_name=directory.name,health=health,
                stage='初期調整（単画像モデル）' if completed<warmup else '時系列・先読み学習',
                completed_epochs=completed,current_epoch=current,maximum_epochs=maximum,warmup_epochs=warmup,
                batch=progress.get('batches'),total_batches=progress.get('total_batches'),
                history=history,checkpoint_time=checkpoint_time,retries=int(watcher.get('retries',0)),
                samples=report.get('samples') or plan.get('supervised_windows',{}),
                source_frames=plan.get('source_frames'),history_seconds=plan.get('history_seconds',4),
                accepted_offline=report.get('accepted_offline'),complete=report.get('phase')=='complete'),
            jobs=jobs,gpu=self.gpu_status(now),
            drive=dict(label=drive_label,live=drive_fresh,speed_kmh=finite(drive.get('speed_kmh')) if drive_fresh else None,
                last_speed_kmh=finite(drive.get('speed_kmh')),age_seconds=max(0.,now-drive_stamp) if drive_stamp else None,
                target_kmh=finite(session.get('target_kmh')) or 60,evaluation=phase_info(evaluation,now)),
            source_note='このPCの学習ログを1秒ごとに確認。検証誤差は各周の完了後に更新します。')

    def get(self):
        with self.lock:
            if self.cached is None or time.monotonic()-self.updated>=.75:
                self.cached=self.collect();self.updated=time.monotonic()
            return self.cached


def make_handler(snapshot,workspace):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.headers.get('Host','').split(':')[0] not in ('127.0.0.1','localhost'):
                self.send_error(403);return
            path=self.path.split('?',1)[0]
            if path=='/health':body=json.dumps(dict(service=SERVICE,workspace=workspace)).encode();mime='application/json'
            elif path=='/api/status':
                try:body=json.dumps(snapshot.get(),ensure_ascii=False,allow_nan=False).encode('utf-8');mime='application/json'
                except Exception:self.send_error(503,'Status temporarily unavailable');return
            else:
                names={'/':'index.html','/index.html':'index.html','/dashboard.css':'dashboard.css','/dashboard.js':'dashboard.js'}
                name=names.get(path)
                if name is None:self.send_error(404);return
                try:body=(ASSETS/name).read_bytes()
                except OSError:self.send_error(404);return
                mime={'html':'text/html','css':'text/css','js':'text/javascript'}[name.rsplit('.',1)[-1]]
            self.send_response(200);self.send_header('Content-Type',mime+'; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'self'")
            self.end_headers();self.wfile.write(body)
    return Handler


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--background',action='store_true');parser.add_argument('--open',action='store_true');args=parser.parse_args()
    if not 1024<=args.port<=65535:raise ValueError('Invalid local port')
    url=f'http://127.0.0.1:{args.port}';workspace=hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
    if args.background:
        def ready():
            try:
                with urllib.request.urlopen(url+'/health',timeout=1) as response:value=json.load(response)
                return value.get('service')==SERVICE and value.get('workspace')==workspace
            except (OSError,ValueError):return False
        if not ready():
            (ROOT/'out').mkdir(exist_ok=True)
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else subprocess.DETACHED_PROCESS if hasattr(subprocess,'DETACHED_PROCESS') else 0
            with (ROOT/'out/dashboard-server.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-m','mono_drive.training_dashboard','--port',str(args.port)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags,start_new_session=os.name!='nt')
            for _ in range(40):
                if ready():break
                if child.poll() is not None:raise RuntimeError('Dashboard could not start; see out/dashboard-server.log')
                time.sleep(.25)
            else:raise RuntimeError('Dashboard startup timed out')
        print(json.dumps(dict(url=url,ready=True)))
        if args.open:
            import webbrowser
            webbrowser.open(url)
        return
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(DashboardSnapshot(),workspace))
    (ROOT/'out').mkdir(exist_ok=True)
    (ROOT/'out/dashboard-server.json').write_text(json.dumps(dict(pid=os.getpid(),url=url,started_at=time.time())),encoding='utf-8')
    print(json.dumps(dict(url=url,pid=os.getpid(),ready=True)),flush=True)
    if args.open:
        import webbrowser
        webbrowser.open(url)
    try:server.serve_forever(poll_interval=.5)
    finally:server.server_close()

if __name__=='__main__':main()
