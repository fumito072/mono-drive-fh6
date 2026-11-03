'use strict';
const $=id=>document.getElementById(id);
const integer=new Intl.NumberFormat('ja-JP');
const clock=new Intl.DateTimeFormat('ja-JP',{timeZone:'Asia/Tokyo',hour:'2-digit',minute:'2-digit',second:'2-digit'});
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(value,digits=4)=>Number.isFinite(value)?value.toFixed(digits):'—';
const time=value=>Number.isFinite(value)?clock.format(new Date(value*1000)):'未保存';
function age(value){if(!Number.isFinite(value))return '更新時刻を確認中';if(value<10)return 'たった今更新';if(value<60)return `${Math.floor(value)}秒前に更新`;if(value<3600)return `${Math.floor(value/60)}分前に更新`;return `${Math.floor(value/3600)}時間前に更新`;}
let snapshot=null,paused=false,busy=false,lastSuccess=0;
function badge(element,label,tone){element.textContent=label;element.className=`badge ${tone||'muted'}`;}
function chart(data){
  const metric=$('metric').value,history=data.deep.history||[],warmup=data.deep.warmup_epochs;
  const series=[{name:'単画像モデル',key:'single',css:'baseline',points:history.filter(r=>Number.isFinite(r.single?.[metric])).map(r=>[r.epoch,r.single[metric]])},
    {name:'時系列・先読みモデル',key:'temporal',css:'temporal',points:history.filter(r=>r.epoch>warmup&&Number.isFinite(r.temporal?.[metric])).map(r=>[r.epoch,r.temporal[metric]])}].filter(s=>s.points.length);
  $('legend').innerHTML=series.map(s=>`<span class="${s.css}"><i></i>${esc(s.name)}</span>`).join('');
  $('chart-empty').hidden=series.length>0;
  if(!series.length){$('chart').innerHTML='';$('best').textContent='検証結果を待機中';return;}
  const width=700,height=270,left=66,right=22,top=18,bottom=45;
  const all=series.flatMap(s=>s.points),maxX=Math.max(2,...all.map(p=>p[0]));
  let lo=Math.min(...all.map(p=>p[1])),hi=Math.max(...all.map(p=>p[1]));
  const padding=Math.max(.001,(hi-lo)*.2);lo=Math.max(0,lo-padding);hi+=padding;
  const x=v=>left+(v-1)/(maxX-1)*(width-left-right),y=v=>top+(hi-v)/(hi-lo)*(height-top-bottom);
  let svg='';
  for(let i=0;i<=4;i++){let v=lo+(hi-lo)*i/4,yy=y(v);svg+=`<line class="grid" x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}"/><text x="${left-10}" y="${yy+4}" text-anchor="end">${v.toFixed(3)}</text>`;}
  const step=maxX>16?Math.ceil(maxX/6):maxX>8?2:1;
  for(let i=1;i<=maxX;i+=step)svg+=`<text x="${x(i)}" y="${height-23}" text-anchor="middle">${i}</text>`;
  svg+=`<text x="${(left+width-right)/2}" y="${height-3}" text-anchor="middle">学習の周数（周）</text><text transform="translate(14,${(height-bottom)/2}) rotate(-90)" text-anchor="middle">正規化操舵誤差</text>`;
  for(const s of series){svg+=`<polyline class="${s.css}-line" points="${s.points.map(p=>`${x(p[0])},${y(p[1])}`).join(' ')}"/>`;for(const p of s.points)svg+=`<circle class="${s.css}-dot" cx="${x(p[0])}" cy="${y(p[1])}" r="3"><title>${s.name}・${p[0]}周：${p[1].toFixed(5)}</title></circle>`;}
  $('chart').innerHTML=svg;
  const parts=series.map(s=>`${s.name}の最小値 ${Math.min(...s.points.map(p=>p[1])).toFixed(4)}`);
  $('best').textContent=parts.join(' / ');
  $('warmup-note').textContent=series.some(s=>s.key==='temporal')?'初期調整中の時系列モデルの値は、比較グラフに含めていません。':'時系列モデルの線は、初期調整が終わる9周目から表示します。';
}
function render(data){
  snapshot=data;const d=data.deep;
  $('model-name').textContent=d.name;
  badge($('learning-state'),d.health.label,d.health.tone);
  $('completed').textContent=d.completed_epochs;$('maximum').textContent=d.maximum_epochs;
  const fraction=Math.min(100,100*d.completed_epochs/Math.max(1,d.maximum_epochs));
  $('progress-fill').style.width=`${fraction}%`;
  $('progressbar').setAttribute('aria-valuemax',d.maximum_epochs);$('progressbar').setAttribute('aria-valuenow',d.completed_epochs);
  $('stage').textContent=d.complete?'学習完了・検証結果を保存済み':d.stage;
  $('progress-note').textContent=d.complete?(d.accepted_offline?'オフライン検証を通過。走行中の評価は別途必要です。':'検証条件を満たしていないため、自動では採用しません。'):
    (d.batch&&d.total_batches?`第${d.current_epoch}周を計算中（${d.batch} / ${d.total_batches}区間）。最大周数より早く終了する場合があります。`:`${d.completed_epochs}周までの検証結果を保存済み。各周の完了時に更新します。最大48周、改善が止まれば早期終了。`);
  $('checkpoint').textContent=d.checkpoint_time?`${time(d.checkpoint_time)} に保存`:'最初の保存待ち';
  $('retry').textContent=d.retries?`再開 ${d.retries} 回`:'';
  const g=data.gpu;
  $('gpu-use').textContent=g?Math.round(g.utilization):'—';$('gpu-fill').style.width=g?`${Math.min(100,g.utilization)}%`:'0%';
  $('gpu-memory').textContent=g?`${g.used_gib.toFixed(1)} / ${g.total_gib.toFixed(1)} GiB`:'取得できません';
  $('gpu-temperature').textContent=g?`${Math.round(g.temperature)} °C`:'—';
  $('jobs').innerHTML=data.jobs.map(j=>`<div class="job"><div class="job-name">${esc(j.title)}</div><div class="job-state"><i class="status-dot ${esc(j.tone)}"></i><span>${esc(j.label)}</span></div><p class="job-detail">${esc(j.details||'')}${j.details?' · ':''}${esc(age(j.age_seconds))}</p></div>`).join('');
  const samples=d.samples||{},total=['train','validation','test'].reduce((sum,k)=>sum+(samples[k]||0),0);
  for(const key of ['train','validation','test']){$(`${key}-count`).textContent=Number.isFinite(samples[key])?`${integer.format(samples[key])} 例`:'—';$(`bar-${key}`).style.width=total?`${100*(samples[key]||0)/total}%`:'0%';}
  $('source-frames').textContent=d.source_frames?`元データ ${integer.format(d.source_frames)} フレーム · 過去${d.history_seconds}秒の映像履歴`:'利用中のデータを確認しています';
  $('drive-status').textContent=data.drive.label;$('drive-speed').textContent=num(data.drive.speed_kmh,1);$('drive-target').textContent=`${data.drive.target_kmh} km/h`;
  $('evaluation').textContent=data.drive.evaluation.label;
  $('source-note').textContent=data.source_note;
  $('updated').textContent=`最終取得 ${time(data.meta.generated_at)}（日本時間）`;
  chart(data);
}
async function refresh(){
  if(paused||busy)return;busy=true;
  try{const response=await fetch('/api/status',{cache:'no-store',signal:AbortSignal.timeout(7000)});if(!response.ok)throw new Error('unavailable');const data=await response.json();render(data);lastSuccess=Date.now();$('offline').hidden=true;document.body.classList.remove('disconnected');badge($('connection'),'自動更新中','good');}
  catch(error){$('offline').hidden=false;document.body.classList.add('disconnected');badge($('connection'),'接続を確認中','warn');}
  finally{busy=false;}
}
$('metric').addEventListener('change',()=>{if(snapshot)chart(snapshot);});
$('pause').addEventListener('click',()=>{paused=!paused;$('pause').textContent=paused?'表示の更新を再開':'表示の更新を一時停止';if(paused)badge($('connection'),'表示を一時停止','warn');else refresh();});
setInterval(refresh,1000);refresh();
