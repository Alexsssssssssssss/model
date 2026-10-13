"""Local-only dispatch console. Credentials stay on the user's machine."""
import argparse,asyncio,csv,io,json,os,secrets,threading,time,uuid,webbrowser,zipfile,hashlib
from collections import Counter
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
import torch
from tenhou_live import PrivateRoomClient,run_all,normalize_account,valid_account
from live_policy import load_policy,preferred_checkpoint
from mahjong_stats import rates
from ppo_core import optimize
from training_objectives import UTILITIES,OBJECTIVE_VERSION,terminal_reward

BASE=Path(__file__).resolve().parent
WEB=BASE/'dispatch-web'
RELEASE_VERSION='20261013.1'

def atomic_json(path,data,private=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp-'+uuid.uuid4().hex)
    descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600 if private else 0o644)
    with os.fdopen(descriptor,'w',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    os.replace(temporary,path)

def models():
    result=[]
    manifest=BASE/'MODEL-MANIFEST.json'
    if manifest.exists():
        release=json.loads(manifest.read_text(encoding='utf-8'))
        if release.get('version')!=RELEASE_VERSION:raise ValueError('模型包版本不匹配，请重新解压完整新版包')
        for row in release['models']:
            path=(BASE/row['path']).resolve()
            if not path.is_relative_to(BASE.resolve()) or not path.is_file():raise ValueError('模型文件缺失')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['bundle_sha256']:raise ValueError('模型文件校验失败，请重新下载')
            result.append(dict(id=row['model'],label=row['label'],path=path,profile=row['profile'],stage=row['stage']))
    else:
        report_path=BASE/'runs/rl100k-corrected-final-evaluation.json'
        report=json.loads(report_path.read_text()) if report_path.exists() else {}
        if report.get('status')=='completed' and report.get('completed_training_games')==100000:
            path=BASE/'runs/corrected-full-actions-v4-20261012/expert-best.pt'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=report['baseline_sha256']:raise ValueError('监督模型与评估绑定不匹配')
            result.append(dict(id='supervised-v4',label='监督基准 · 修复重训 v4',path=path,profile='balanced',stage='已完成独立评估'))
            for profile,label in [('first_place','争一型'),('balanced','平衡型'),('avoid_fourth','避四型')]:
                selection=report['selections'][profile];path=Path(selection['checkpoint'])
                if hashlib.sha256(path.read_bytes()).hexdigest()!=selection['sha256']:raise ValueError('强化学习模型与评估绑定不匹配')
                result.append(dict(id=profile,label=label+' · 10万半庄评估优选',path=path,profile=profile,stage='已评估，未证明显著提升'))
    for folder in (BASE/'runs/dispatch').glob('*'):
        metadata=folder/'task.json'
        if not metadata.is_file():continue
        try:task=json.loads(metadata.read_text(encoding='utf-8'))
        except (ValueError,OSError):continue
        if task.get('version')!=RELEASE_VERSION:continue
        for bot in task.get('bots',[]):
            index=bot['index']+1;paths=sorted(folder.glob(f'bot-{index}-update-*.pt'))
            if paths:
                result.append(dict(id=f'local-{folder.name}-{index}',label=f'个室训练 · {folder.name[:6]} / AI{index}',path=paths[-1],profile=bot.get('goal','balanced'),stage='本机训练，待验证'))
    return {r['id']:r for r in result}

class Console:
    def __init__(self,root=BASE):
        self.root=Path(root);self.account_path=self.root/'dispatch-accounts.local.json';self.lock=threading.RLock();self.tasks={};self.registry=models()
        self.accounts=json.loads(self.account_path.read_text(encoding='utf-8')) if self.account_path.exists() else []
        legacy=self.root/'tenhou-room-2456.local.json'
        if not self.accounts and legacy.exists():
            old=json.loads(legacy.read_text())
            self.accounts=[dict(key=uuid.uuid4().hex,platform='tenhou',label=r['profile'],login=normalize_account(r['id'])) for r in old.get('accounts',[])]
            atomic_json(self.account_path,self.accounts,True)
        for path in (self.root/'runs/dispatch').glob('*/task.json'):
            record=json.loads(path.read_text(encoding='utf-8'));record.update(folder=path.parent,rows=[])
            if record['status'] in ('starting','running','stopping'):record.update(status='stopped',note='程序已重启；已保留数据，请创建新任务。')
            self.tasks[record['id']]=record
            for bot in record.get('bots',[]):
                for log in sorted(path.parent.glob('game-*/bot-'+str(bot['index']+1)+'/public-events-and-own-actions.jsonl')):
                    with log.open(encoding='utf-8') as handle:
                        for line in handle:
                            try:event=json.loads(line).get('event',{})
                            except json.JSONDecodeError:continue
                            if event.get('type')=='start_game' and str(event.get('log','')).startswith('https://tenhou.net/'):
                                bot['replay_url']=event['log'];break
        torch.set_num_threads(1)
    def snapshot(self):
        with self.lock:
            self.registry=models()
            return dict(version=RELEASE_VERSION,objectives=UTILITIES,models=[{k:v for k,v in r.items() if k!='path'} for r in self.registry.values()],accounts=[{k:r[k] for k in ('key','platform','label')} for r in self.accounts],tasks=[self.public(t) for t in reversed(list(self.tasks.values()))],capabilities=dict(tenhou=True,majsoul=True,majsoul_reason='雀魂为实验接入：国服账号密码、标准四人半庄友人房；尚未完成真实账号对战验证，当前只支持邀请模式。遇到未支持事件会停止并保存日志。'))
    def public(self,t):return {k:v for k,v in t.items() if k not in ('loop','job','credentials','rows','folder')}
    def save_account(self,data):
        platform=data.get('platform');label=str(data.get('label','')).strip()
        if platform not in ('tenhou','majsoul') or not label or len(label)>40:raise ValueError('请输入账号别名并选择平台')
        login=str(data.get('login','')).strip();password=str(data.get('password',''))
        if platform=='tenhou':
            login=normalize_account(login)
            if not valid_account(login):raise ValueError('天凤账号须填写完整登录ID，以ID开头并包含中间的-，不能填昵称')
        elif not login or not password:raise ValueError('雀魂账号和密码不能为空')
        with self.lock:
            if any(a['platform']==platform and a['login']==login for a in self.accounts):raise ValueError('此账号已保存，请选择已有账号')
            row=dict(key=uuid.uuid4().hex,platform=platform,label=label,login=login)
            if platform=='majsoul':row['password']=password
            self.accounts.append(row);atomic_json(self.account_path,self.accounts,True)
        return dict(saved=True)
    def delete_account(self,key):
        with self.lock:
            if any(t['status'] in ('starting','running','stopping') and any(r['account']==key for r in t['rows']) for t in self.tasks.values()):raise ValueError('账号正在使用，请先停止任务')
            self.accounts=[a for a in self.accounts if a['key']!=key];atomic_json(self.account_path,self.accounts,True)
        return dict(deleted=True)
    def dispatch(self,data):
        platform=data.get('platform','tenhou');mode=data.get('mode','invite')
        if platform not in ('tenhou','majsoul'):raise ValueError('平台无效')
        if platform=='majsoul' and mode!='invite':raise ValueError('雀魂实验接入当前仅支持邀请对战；四AI训练请用天凤')
        room=data.get('room');target=data.get('games',1);rows=data.get('bots',[])
        if type(room)!=int or not (1000<=room<=9999 if platform=='tenhou' else 1<=room<=99999999):raise ValueError('房间号无效：天凤个室须为1000–9999')
        if mode not in ('invite','selfplay','rl'):raise ValueError('对战模式无效')
        if not isinstance(rows,list) or (mode=='invite' and not 1<=len(rows)<=3) or (mode!='invite' and len(rows)!=4):raise ValueError('邀请模式选择1～3位AI；四AI内战选择4位')
        if type(target)!=int or not 1<=target<=50000:raise ValueError('半庄数须为1–50000')
        if mode=='invite' and target!=1:raise ValueError('真人邀请每次派遣打一场半庄')
        if len({r.get('account') for r in rows})!=len(rows):raise ValueError('每位AI须使用不同账号')
        if mode=='rl' and not any(r.get('learn') for r in rows):raise ValueError('请选择至少一位学习AI')
        if mode=='rl' and all(r.get('learn') for r in rows):raise ValueError('保留至少一位冻结模型作为参考对手')
        with self.lock:
            credentials={r['key']:r for r in self.accounts}
            busy={r['account'] for t in self.tasks.values() if t['status'] in ('starting','running','stopping') for r in t['rows']}
            checked=[]
            for r in rows:
                if r.get('model') not in self.registry:raise ValueError('模型不存在')
                if r.get('account') not in credentials or credentials[r['account']]['platform']!=platform:raise ValueError('请为每位AI选择对应平台账号')
                if r['account'] in busy:raise ValueError('账号已在其他任务中运行')
                profile=r.get('goal',self.registry[r['model']]['profile'])
                if profile not in UTILITIES:raise ValueError('学习目标无效')
                checked.append(dict(account=r['account'],model=r['model'],goal=profile,learn=bool(r.get('learn')) if mode=='rl' else False))
            ident=uuid.uuid4().hex[:12];folder=self.root/'runs/dispatch'/ident
            task=dict(id=ident,status='starting',platform=platform,room=room,mode=mode,target=target,completed=0,updates=0,created=time.time(),bots=[dict(index=i,account_label=credentials[r['account']]['label'],model_label=self.registry[r['model']]['label'],status='准备连接',learn=r['learn'],goal=r['goal']) for i,r in enumerate(checked)],rows=checked,credentials=[dict(credentials[r['account']]) for r in checked],folder=folder,statistics=[],error=None)
            task.update(version=RELEASE_VERSION,objective_version=OBJECTIVE_VERSION,
                        exploration_temperature=.15,rank_reward_only=True)
            for bot,row in zip(task['bots'],checked):
                model_path=self.registry[row['model']]['path']
                bot.update(checkpoint_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                           selection_mode='sampled' if row['learn'] else 'argmax',
                           temperature=.15 if row['learn'] else 0,
                           placement_utility=UTILITIES[row['goal']])
            self.tasks[ident]=task
            atomic_json(folder/'task.json',self.public(task))
            threading.Thread(target=self.worker,args=(task,),daemon=True).start()
        return dict(id=ident)
    def worker(self,t):
        loop=asyncio.new_event_loop();asyncio.set_event_loop(loop)
        async def execute():
            with self.lock:t['loop']=loop;t['job']=asyncio.current_task();t['status']='running'
            current=[self.registry[r['model']]['path'] for r in t['rows']];batch=[[] for _ in current];counts=[Counter() for _ in current]
            for game in range(t['target']):
                clients=[]
                for i,r in enumerate(t['rows']):
                    def update(status,extra,index=i):
                        with self.lock:
                            t['bots'][index]['status']=status
                            if str(extra.get('replay_url','')).startswith('https://tenhou.net/'):
                                t['bots'][index]['replay_url']=extra['replay_url']
                    if t['platform']=='tenhou':
                        client=PrivateRoomClient(t['credentials'][i]['login'],r['goal'],current[i],room=t['room'],training=r['learn'],status_callback=update,output=t['folder']/f'game-{game+1:06d}'/f'bot-{i+1}',temperature=t['exploration_temperature'])
                    else:
                        from majsoul_live import SoulRoomClient
                        client=SoulRoomClient(t['credentials'][i]['login'],t['credentials'][i]['password'],r['goal'],current[i],room=t['room'],status_callback=update,output=t['folder']/f'game-{game+1:06d}'/f'bot-{i+1}')
                    clients.append(client)
                await run_all(clients,independent=t['mode']=='invite')
                results=[c.result for c in clients]
                if any(r is None for r in results):raise RuntimeError('半庄未完整结束，不更新模型')
                # Confirm all clients played the same half-game before using it for RL.
                logs={next((e.get('log') for e in c.events if e['type']=='start_game'),None) for c in clients} if t['platform']=='tenhou' else set()
                keys={str(v).split('log=',1)[-1].split('&')[0] for v in logs if v}
                if t['mode']!='invite' and (len(keys)!=1 or None in logs):raise RuntimeError('四个账号未进入同一半庄，保存日志并停止；不训练')
                for i,(c,r) in enumerate(zip(clients,t['rows'])):
                    counts[i].update(c.result['own_stats']);batch[i].append((c.policy.trajectory if t['platform']=='tenhou' else [],c.result))
                with self.lock:
                    t['completed']=game+1;t['statistics']=[dict(bot=i+1,model=t['bots'][i]['model_label'],**rates(c)) for i,c in enumerate(counts)]
                    atomic_json(t['folder']/'task.json',self.public(t))
                # Fixed behavior weights within each eight-half-game batch.
                if t['mode']=='rl' and (game+1)%8==0:
                    for i,r in enumerate(t['rows']):
                        if not r['learn']:continue
                        update_job=asyncio.create_task(asyncio.to_thread(self.train_batch,t,i,current[i],batch[i]))
                        try:path,metrics=await asyncio.shield(update_job)
                        except asyncio.CancelledError:
                            await update_job # Finish the atomic update before releasing accounts.
                            raise
                        current[i]=path
                        with self.lock:t['bots'][i]['last_update']=metrics
                    batch=[[] for _ in current]
                    with self.lock:t['updates']+=1
                if game+1<t['target']:await asyncio.sleep(5)
            with self.lock:t['status']='completed';t['note']='不足8把的尾批只保存数据，未更新模型' if t['mode']=='rl' and t['target']%8 else '任务已完成'
        try:loop.run_until_complete(execute())
        except asyncio.CancelledError:
            with self.lock:t['status']='stopped';t['note']='任务已停止，保留已完成对局和模型'
        except Exception as exc:
            with self.lock:
                t['status']='error';t['error']=str(exc)
                for path in sorted(t['folder'].glob('game-*/bot-*/diagnostics.json')):
                    detail=json.loads(path.read_text(encoding='utf-8'))
                    if detail.get('error_type')!='CancelledError':
                        index=int(path.parent.name.removeprefix('bot-'))
                        t['error_context']=dict(bot=index,phase=detail.get('phase'),
                                               last_received=detail.get('last_received'),
                                               last_sent=detail.get('last_sent'))
                        break
        finally:
            with self.lock:
                atomic_json(t['folder']/'task.json',self.public(t));t.pop('credentials',None)
            loop.close()
    def train_batch(self,t,i,checkpoint,batch):
        model=load_policy(checkpoint);anchor=load_policy(self.registry[t['rows'][i]['model']]['path']);value=torch.nn.Linear(64*34,1)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if 'value' in saved:value.load_state_dict(saved['value'])
        else:torch.nn.init.zeros_(value.weight);torch.nn.init.zeros_(value.bias)
        rows=[];returns=[];utility=np.array(UTILITIES[t['rows'][i]['goal']],dtype=float);utility/=max(abs(utility))
        for trajectory,result in batch:
            if result['checkpoint_sha256']!=model.source_sha256:raise RuntimeError('行为模型版本不一致，拒绝强化学习更新')
            if result.get('temperature')!=t['exploration_temperature']:raise RuntimeError('行为采样温度与更新不一致，拒绝训练')
            rows.extend(trajectory);reward=terminal_reward(t['rows'][i]['goal'],result['ranks'][0]);returns.extend([reward]*len(trajectory))
        if not rows:raise RuntimeError('此批没有可学习决策')
        optimizer=torch.optim.Adam([*model.parameters(),*value.parameters()],lr=1e-5)
        metrics=optimize(model,value,anchor,optimizer,rows,returns,np.random.default_rng(t['completed']+i),temperature=t['exploration_temperature'])
        path=t['folder']/f'bot-{i+1}-update-{t["updates"]+1:04d}.pt'
        temporary=path.with_suffix('.tmp');torch.save(dict(policy=model.state_dict(),value=value.state_dict(),encoding=model.encoding,selfplay=dict(profile=t['rows'][i]['goal'],source='tenhou_private_room',completed_games=t['completed'],objective_version=OBJECTIVE_VERSION,placement_utility=UTILITIES[t['rows'][i]['goal']],temperature=t['exploration_temperature'],rank_reward_only=True),parent_sha256=model.source_sha256,anchor_sha256=anchor.source_sha256),temporary);os.replace(temporary,path)
        atomic_json(path.with_suffix('.json'),metrics)
        return path,metrics
    def stop(self,ident):
        with self.lock:
            t=self.tasks.get(ident)
            if not t:raise ValueError('任务不存在')
            if t['status'] in ('starting','running'):
                if 'loop' not in t:raise ValueError('任务正在启动，请稍后停止')
                t['status']='stopping';t['loop'].call_soon_threadsafe(t['job'].cancel)
        return dict(stopping=True)

    def diagnostic_archive(self,ident):
        with self.lock:
            task=self.tasks.get(ident)
            if task is None:raise ValueError('任务不存在')
            folder=task['folder'];metadata=self.public(task)
        buffer=io.BytesIO()
        permitted={'status.json','diagnostics.json','public-events-and-own-actions.jsonl',
                   'network-events.jsonl','result.json'}
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('task.json',json.dumps(metadata,ensure_ascii=False,indent=2))
            for path in sorted(folder.glob('game-*/bot-*/*')):
                if path.is_file() and path.name in permitted and not path.is_symlink():
                    archive.writestr(str(path.relative_to(folder)),path.read_bytes())
        return buffer.getvalue()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def respond(self,data,status=200,mime='application/json; charset=utf-8'):
        raw=json.dumps(data,ensure_ascii=False).encode() if isinstance(data,(dict,list)) else data
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'self'; img-src 'self'; script-src 'self'; frame-ancestors 'none'");self.end_headers();self.wfile.write(raw)
    def allowed_host(self):return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
    def do_GET(self):
        if not self.allowed_host():return self.respond(dict(error='Invalid Host'),403)
        path=self.path.split('?',1)[0]
        if path=='/api/state':return self.respond(dict(token=self.server.token,**self.server.console.snapshot()))
        if path.startswith('/api/diagnostics/'):
            try:raw=self.server.console.diagnostic_archive(path.split('/')[-1])
            except ValueError as exc:return self.respond(dict(error=str(exc)),404)
            return self.respond(raw,mime='application/zip')
        if path.startswith('/api/statistics/'):
            ident=path.split('/')[-1]
            with self.server.console.lock:
                task=self.server.console.tasks.get(ident)
                if not task:return self.respond(dict(error='任务不存在'),404)
                rows=list(task['statistics'])
            buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]) if rows else ['half_games']);writer.writeheader();writer.writerows(rows)
            return self.respond(('\ufeff'+buffer.getvalue()).encode(),200,'text/csv; charset=utf-8')
        name='index.html' if path=='/' else path.lstrip('/')
        file=(WEB/name).resolve()
        if not file.is_relative_to(WEB.resolve()) or not file.is_file():return self.respond(dict(error='Not found'),404)
        types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.png':'image/png'}
        self.respond(file.read_bytes(),mime=types.get(file.suffix,'application/octet-stream'))
    def do_POST(self):
        origin=self.headers.get('Origin');expected=(f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}')
        if not self.allowed_host() or self.headers.get('X-Console-Token')!=self.server.token or origin not in expected:return self.respond(dict(error='请求来源无效，请从本机控制台操作'),403)
        try:
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<=16384:raise ValueError('请求大小无效')
            data=json.loads(self.rfile.read(length));console=self.server.console
            if self.path=='/api/accounts':result=console.save_account(data)
            elif self.path=='/api/accounts/delete':result=console.delete_account(data.get('key'))
            elif self.path=='/api/dispatch':result=console.dispatch(data)
            elif self.path=='/api/stop':result=console.stop(data.get('id'))
            else:return self.respond(dict(error='Not found'),404)
            self.respond(result)
        except (ValueError,KeyError,TypeError) as exc:self.respond(dict(error=str(exc)),400)
        except Exception:self.respond(dict(error='操作失败，请检查本机文件权限'),500)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8770);parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
    try:server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    except OSError:raise SystemExit('控制台端口已被占用，请先关闭旧启动窗口，再运行新版。')
    server.console=Console();server.token=secrets.token_urlsafe(32)
    if not args.no_browser:webbrowser.open(f'http://127.0.0.1:{args.port}')
    print('AI控制台已启动。请保留此窗口；关闭前先停止网页中的任务。')
    try:server.serve_forever()
    except KeyboardInterrupt:
        for t in list(server.console.tasks.values()):
            if t['status'] in ('running','starting'):server.console.stop(t['id'])
    finally:server.server_close()
if __name__=='__main__':main()
