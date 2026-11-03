"""Render an aligned held-out driving clip with teacher and causal forecasts."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mono_drive.temporal_steering import HISTORY,targets,sequence_arrays,extended_indices


def main():
    import cv2,onnxruntime as ort
    from PIL import Image,ImageDraw,ImageFont
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true')
    p.add_argument('--model-dir',type=Path,default=ROOT/'models/temporal-20261103')
    p.add_argument('--cache',type=Path,default=ROOT/'out/temporal-human')
    p.add_argument('--out',type=Path,default=ROOT/'out/temporal-report');a=p.parse_args()
    model=a.model_dir;cache=a.cache;out=a.out;out.mkdir(parents=True,exist_ok=True)
    while not (model/'report.json').exists():
        if not a.wait:raise FileNotFoundError('Temporal training has not completed')
        time.sleep(3)
    report=json.loads((model/'report.json').read_text(encoding='utf-8'))
    manifest=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    if report['dataset_revision']!=manifest['revision']:raise ValueError('Video cache does not match this model')
    stride=int(manifest.get('temporal_stride',1));fps=float(manifest.get('source_sample_hz',5.))
    history_frames=int(report.get('history_frames',HISTORY))
    best=None
    for item in manifest['segments']:
        if item['split']!='test':continue
        with np.load(item['file']) as f:
            steer=f['steer'];indices=f['indices']
            if history_frames>HISTORY:indices=extended_indices(indices,len(steer),history_frames,stride)
            if len(steer)<120:continue
            for start in indices[::20]:
                end=min(len(steer)-3*stride-1,start+int(20*fps))
                if end-start<80:continue
                score=float(np.std(steer[start:end]))
                if best is None or score>best[0]:best=(score,item,int(start),int(end))
    if best is None:raise ValueError('No independent video segment long enough')
    _,item,start,end=best
    with np.load(item['file']) as f:data={k:f[k].copy() for k in f.files}
    session=ort.InferenceSession(str(model/'steering.onnx'),providers=['CPUExecutionProvider'])
    image_files=sorted((Path(item['source'])/'frames').glob('*.jpg'))
    times=data['times'];steer=data['steer'];speed=data['speed'];features=data['features']
    indices=data['indices']
    if history_frames>HISTORY:indices=extended_indices(indices,len(steer),history_frames,stride)
    valid=set(indices.tolist())
    records=[]
    for i in range(start,end):
        prediction=None
        if i in valid:
            x,ego,_=sequence_arrays(features,speed,steer,i,stride,history_frames)
            prediction=float(session.run(None,{'features':x[None],'ego':ego[None]})[0].reshape(-1)[0])
        records.append({'index':i,'time':float(times[i]),'steer':float(steer[i]),'speed':float(speed[i]),
                        'prediction':prediction,'future_teacher':float(targets(steer,i,stride)[-1]),
                        'forecast_time':float(np.mean(times[i+np.arange(1,4)*stride]))})
    (out/'timeline.json').write_text(json.dumps({'split':'unseen_recording_test','records':records,'test':report['test_temporal']},indent=2),encoding='utf-8')
    font=ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',21);small=ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',16)
    filename=out/'steering-comparison.mp4'
    command=['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x960','-framerate',str(fps),
             '-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(filename)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    x0,x1,y0,y1=75,1240,786,910;t0,t1=times[start],times[end-1]+.5
    def point(t,v):return (x0+(t-t0)/(t1-t0)*(x1-x0),(y0+y1)/2-v*(y1-y0)/2)
    colors={'steer':(82,166,245),'prediction':(245,148,57),'future_teacher':(116,200,150)}
    for current in records:
        image=cv2.imread(str(image_files[current['index']]))
        canvas=Image.new('RGB',(1280,960),(20,23,29));canvas.paste(Image.fromarray(cv2.cvtColor(cv2.resize(image,(1280,720)),cv2.COLOR_BGR2RGB)),(0,0))
        draw=ImageDraw.Draw(canvas)
        draw.rectangle((0,0,1280,65),fill=(20,23,29))
        draw.text((20,5),'時間と景色から、その先の操舵を予測：学習に使っていない録画',font=font,fill='white')
        draw.text((20,36),f"動画内時刻 {current['time']:.2f} 秒  |  速度 {current['speed']:.1f} km/h  |  人の操舵 {current['steer']:+.3f}",font=small,fill='white')
        draw.text((20,726),'青：人の操舵  橙：約0.4秒先の時系列予測  緑：予測対象の平均操舵',font=font,fill='white')
        for v in (-1.,0.,1.):
            y=point(t0,v)[1];draw.line((x0,y,x1,y),fill=(70,74,80),width=1);draw.text((20,y-10),f'{v:+.0f}',font=small,fill='white')
        for key,color in colors.items():
            line=[]
            for row in records:
                if key!='steer' and row['index']>current['index']:break
                value=row[key]
                if value is not None:line.append(point(row['forecast_time'] if key!='steer' else row['time'],value))
            if len(line)>1:draw.line(line,fill=color,width=3)
        x=point(current['time'],0)[0];draw.line((x,y0,x,y1),fill='white',width=2)
        draw.text((x0,920),f'時間（秒）  予測には現在と過去約{(history_frames-1)/5:g}秒の映像のみ使用。元記録から抽出した約{fps:g}fps映像。',font=small,fill='white')
        for t in np.linspace(t0,t1,7):draw.text((point(t,0)[0]-12,755),f'{t:.1f}',font=small,fill='white')
        process.stdin.write(np.asarray(canvas).tobytes())
    process.stdin.close();error=process.stderr.read();code=process.wait()
    if code:raise RuntimeError(error.decode(errors='replace'))
    print(filename,flush=True)

if __name__=='__main__':main()
