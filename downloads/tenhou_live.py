"""Use this project's full-action models in a specified Tenhou private room."""
import argparse, asyncio, json, re, sys, time, getpass, os, unicodedata, logging
from pathlib import Path
import numpy as np
import torch
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
from riichienv import RiichiEnv, GameRule, GameType
from live_policy import load_policy, tensors, resolve, preferred_checkpoint

BASE=Path(__file__).resolve().parent
VENDOR=BASE/'vendor/mjai-gateway/src'
sys.path.insert(0,str(VENDOR))
import router
import settings
from utils.state import State

def normalize_account(value):
    if not isinstance(value,str):return ''
    value=unicodedata.normalize('NFKC',value).strip().strip('\ufeff\u200b').strip()
    value=value.strip('\"\'“”‘’').strip()
    value=value.translate(str.maketrans({'–':'-','—':'-','−':'-'}))
    if value[:2].lower()=='id':value='ID'+value[2:]
    return value

def valid_account(value):
    return re.fullmatch(r'ID[0-9A-Za-z]+-[0-9A-Za-z]+',value) is not None

def prompt_accounts():
    print('请输入三个 AI 账号的完整天凤登录 ID，形如 IDxxxxxxxx-xxxxxxxx，不是玩家昵称。')
    print('输入不回显；粘贴后按 Enter。Ctrl+C 可取消。')
    rows=[]
    for profile in ('avoid_fourth','balanced','first_place'):
        while True:
            account=normalize_account(getpass.getpass(profile+' 的登录 ID: '))
            if not valid_account(account):
                print('格式不正确：需要以 ID 开头、包含中间的 - 和后半段。请重新输入。')
                continue
            if any(r['id']==account for r in rows):
                print('该账号已用于另一位 AI，请输入不同账号。')
                continue
            rows.append({'profile':profile,'id':account});break
    return rows

def signature(action):
    return (action['type'],action.get('pai'),tuple(sorted(action.get('consumed',[]))))

class OnlinePolicy:
    def __init__(self,checkpoint,training=False,temperature=.15):
        if not 0<temperature<=1:raise ValueError('Exploration temperature must be in (0, 1]')
        self.model=load_policy(checkpoint)
        self.env=RiichiEnv(game_mode=int(GameType.YON_HANCHAN),rule=GameRule.default_tenhou())
        self.decisions=0;self.training=training;self.temperature=temperature;self.trajectory=[];self.last_decision=None
        self.value=torch.nn.Linear(64*34,1)
        torch.nn.init.zeros_(self.value.weight);torch.nn.init.zeros_(self.value.bias)
        if training:
            saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
            if "value" in saved:self.value.load_state_dict(saved["value"])

    def react(self,event):
        obs=self.env.observe_event(event,0)
        if obs is None:return {'type':'none'}
        x,mask,actions=tensors(obs,self.model.encoding)
        choices=event.get('possible_actions',[])
        signatures={signature(a) for a in choices}
        forbidden=set(event.get('cannot_dahai',[]))
        allowed={}
        for label,group in actions.items():
            for action in group:
                msg=json.loads(action.to_mjai())
                if msg['type']=='dahai':valid=msg['pai'] not in forbidden
                elif msg['type']=='none':valid=True
                elif event['type']=='reach' and msg['type']=='reach':valid=False
                else:valid=(signature(msg) in signatures or
                            msg['type'] in ('hora','reach','ryukyoku') and any(a['type']==msg['type'] for a in choices))
                if valid:allowed.setdefault(label,[]).append(action)
        if not allowed:raise RuntimeError('Tenhou and simulator legal actions do not agree; stopping')
        mask[:]=False;mask[list(allowed)]=True
        with torch.no_grad():
            from ppo_core import forward
            logits,value=forward(self.model,self.value,torch.from_numpy(x)[None],torch.from_numpy(mask)[None])
            distribution=torch.distributions.Categorical(logits=logits[0]/self.temperature)
            label=distribution.sample().item() if self.training else logits[0].argmax().item()
            if self.training and mask.sum()>1:
                self.trajectory.append((x.copy(),mask.copy(),label,distribution.log_prob(torch.tensor(label)).item(),value.item()))
        action=resolve(obs,allowed,label);msg=json.loads(action.to_mjai())
        if msg['type']=='dahai':msg['tsumogiri']=action.tile==obs.drawn_tile
        self.decisions+=1
        from riichienv.convert import tid_to_mjai
        self.last_decision=dict(
            id=self.decisions,checkpoint_sha256=self.model.source_sha256,
            selection_mode='sampled' if self.training else 'argmax',
            temperature=self.temperature if self.training else 0,
            own_hand=[tid_to_mjai(t) for t in obs.hands[0]],
            dora_indicators=[tid_to_mjai(t) for t in obs.dora_indicators],
            native_legal=[json.loads(a.to_mjai()) for group in actions.values() for a in group],
            allowed_labels=sorted(allowed),selected_label=label,selected_action=msg,
            candidates=[dict(label=int(i),probability=float(distribution.probs[i]),
                             actions=[json.loads(a.to_mjai()) for a in allowed[int(i)]])
                        for i in torch.argsort(distribution.probs,descending=True).tolist()
                        if i in allowed])
        return msg

class PrivateRoomClient:
    def __init__(self,account,profile,checkpoint,room=2456,game_type=9,training=False,status_callback=None,output=None,wait_timeout=300,temperature=.15):
        if not 1000<=room<=9999:raise ValueError('A private room between 1000 and 9999 is required')
        if game_type!=9:raise ValueError('This launcher supports four-player red-five half-games only')
        self.account=account;self.profile=profile;self.room=room;self.game_type=game_type
        self.policy=OnlinePolicy(checkpoint,training=training,temperature=temperature);self.state=State(account,f'{room}_{game_type}')
        self.ended=False;self.packet={};self.log=None;self.status_path=None
        self.events=[];self.result=None;self.first_oya=None;self.status_callback=status_callback;self.output=output;self.wait_timeout=wait_timeout;self.started=False
        self.network_log=None;self.last_sent=None;self.last_received=None;self.phase='connecting'
        self.last_sent_at=None;self.last_received_at=None;self.last_heartbeat_at=None
        self.heartbeat_count=0;self.heartbeat_error=None

    async def send_event(self,event):
        event=dict(event)
        if event['type']=='start_kyoku':
            event['kyoku']+=1 # Upstream uses zero-based kyoku; MJAI is one-based.
            event['scores']=[int(v)*100 for v in self.packet['ten'].split(',')]
            if self.first_oya is None:self.first_oya=event['oya']
        if event['type']=='hora':
            event.update(actor=int(self.packet['who']),target=int(self.packet['fromWho']))
        # Reconnection with incomplete state must not produce guessed decisions.
        if event['type']=='start_game':
            event['names']=['AI','player1','player2','player3'];event.pop('id',None)
            if self.status_callback and event.get('log'):
                self.status_callback('playing',dict(replay_url=event['log']))
        if event['type'] in ('hora','ryukyoku') and 'sc' in self.packet:
            event['deltas']=[int(float(v))*100 for v in self.packet['sc'].split(',')[1::2]]
        self.events.append(event)
        previous_decisions=self.policy.decisions
        result=self.policy.react(event)
        if self.log:
            row={'event':event,'action':result}
            if self.policy.decisions>previous_decisions:
                row['decision']=self.policy.last_decision
            self.log.write(json.dumps(row,ensure_ascii=False)+'\n');self.log.flush()
        if event['type']=='end_game':
            self.ended=True
            scores=event['scores'];order=sorted(range(4),key=lambda p:(-scores[p],(p-self.first_oya)%4))
            ranks=[order.index(p)+1 for p in range(4)]
            from mahjong_stats import analyze
            self.result=dict(scores=scores,ranks=ranks,own_stats=analyze(self.events,scores,ranks)[0],checkpoint_sha256=self.policy.model.source_sha256,profile=self.profile,temperature=self.policy.temperature if self.policy.training else 0)
            if self.policy.training and self.policy.trajectory:
                rows=self.policy.trajectory
                np.savez_compressed(self.status_path.parent/'trajectory.npz',x=np.stack([r[0] for r in rows]),mask=np.stack([r[1] for r in rows]),label=[r[2] for r in rows],log_prob=[r[3] for r in rows],value=[r[4] for r in rows])
            (self.status_path.parent/'result.json').write_text(json.dumps(self.result,ensure_ascii=False,indent=2))
        return result

    async def send_packet(self,ws,message):
        if message.get('tag')=='JOIN':
            target=f'{self.room},{self.game_type}'
            # Tenhou REJOIN appends ,r to the same room/type reservation.
            if message.get('t') not in (target,target+',r'):
                raise RuntimeError('Refusing a JOIN outside the requested private room')
        self.record_packet('sent',message)
        await ws.send(json.dumps(message))

    def record_packet(self,direction,message):
        # Never retain HELO name, auth values, or a raw account identifier.
        allowed=('tag','type','p','hai','hai0','hai1','who','fromWho','step','t','log','oya','seed','ten','sc','owari')
        packet={k:message[k] for k in allowed if k in message}
        if message.get('tag')=='HELO':packet={'tag':'HELO'}
        if direction=='sent':self.last_sent=packet;self.last_sent_at=time.time()
        else:self.last_received=packet;self.last_received_at=time.time()
        if self.network_log:
            self.network_log.write(json.dumps(dict(time=time.time(),direction=direction,packet=packet),ensure_ascii=False)+'\n');self.network_log.flush()

    async def run(self):
        folder=Path(self.output) if self.output else BASE/'runs/tenhou-live'/str(time.time_ns());folder.mkdir(parents=True,exist_ok=True)
        self.status_path=folder/'status.json'
        def status(value,**kw):
            self.phase=value
            self.status_path.write_text(json.dumps(dict(profile=self.profile,room=self.room,status=value,**kw),ensure_ascii=False,indent=2))
            if self.status_callback:self.status_callback(value,kw)
        status('connecting')
        settings.DEBUG=False # Keep the upstream human-play action delays.
        logging.getLogger('mjai-gateway').disabled=True
        logging.disable(logging.CRITICAL)
        try:
            with (folder/'public-events-and-own-actions.jsonl').open('w',encoding='utf-8') as self.log, (folder/'network-events.jsonl').open('w',encoding='utf-8') as self.network_log:
                async with connect('wss://b-ww.mjv.jp',origin='https://tenhou.net',open_timeout=20,ping_interval=None) as ws:
                    await self.send_packet(ws,{'tag':'HELO','name':self.account,'sx':'M'})
                    async def keepalive():
                        try:
                            while not self.ended:
                                await ws.send('<Z/>')
                                self.last_heartbeat_at=time.time();self.heartbeat_count+=1
                                if self.network_log:
                                    self.network_log.write(json.dumps(dict(time=self.last_heartbeat_at,direction='sent',packet={'tag':'Z'}))+'\n');self.network_log.flush()
                                await asyncio.sleep(10)
                        except asyncio.CancelledError:raise
                        except Exception as exc:
                            self.heartbeat_error=dict(type=type(exc).__name__,error=str(exc),time=time.time())
                            await ws.close()
                    heartbeat=asyncio.create_task(keepalive());connected_at=time.monotonic();game_started_at=None
                    try:
                        while not self.ended:
                            deadline=(game_started_at+7200) if self.started else connected_at+self.wait_timeout
                            remaining=deadline-time.monotonic()
                            if remaining<=0:raise RuntimeError('等待开局或对局超时，已停止')
                            try:raw=await asyncio.wait_for(ws.recv(),remaining)
                            except asyncio.TimeoutError:raise RuntimeError('等待开局或对局超时，请检查个室与四个账号的预约状态')
                            self.packet=json.loads(raw)
                            self.record_packet('received',self.packet)
                            tag=self.packet.get('tag','')
                            if tag in ('ERR','ERROR','REINIT'):
                                raise RuntimeError(f'Tenhou returned {tag}; no automatic reconnect or resubmit')
                            if tag=='HELO':status('waiting_in_private_room')
                            if tag=='INIT':
                                if game_started_at is None:game_started_at=time.monotonic()
                                self.started=True;status('playing')
                            async def send(message):await self.send_packet(ws,message)
                            handled=False
                            for process in router.processes:
                                if await process(self.state,self.packet,send,self.send_event):handled=True;break
                            if self.ended:break
                            if not handled and tag.startswith('REINIT'):raise RuntimeError('Unsupported reconnection state')
                    finally:
                        heartbeat.cancel()
                        await asyncio.gather(heartbeat,return_exceptions=True)
            if not self.ended:raise RuntimeError('Tenhou disconnected before the half-game ended')
            status('completed',decisions=self.policy.decisions,result=self.result)
            return self.result
        except BaseException as exc:
            diagnosis=dict(error_type=type(exc).__name__,error=str(exc),phase=self.phase,
                           time=time.time(),last_sent_at=self.last_sent_at,last_received_at=self.last_received_at,
                           last_heartbeat_at=self.last_heartbeat_at,heartbeat_count=self.heartbeat_count,heartbeat_error=self.heartbeat_error,
                           last_sent=self.last_sent,last_received=self.last_received,
                           checkpoint_sha256=self.policy.model.source_sha256,
                           selection_mode='sampled' if self.policy.training else 'argmax',
                           temperature=self.policy.temperature if self.policy.training else 0)
            (folder/'diagnostics.json').write_text(json.dumps(diagnosis,ensure_ascii=False,indent=2),encoding='utf-8')
            message=str(exc)
            if isinstance(exc,ConnectionClosed):
                message='天凤连接异常关闭（'+str(exc)+'）；中断半庄未训练。请下载诊断记录。'
            status('stopped_on_error',error=message,decisions=self.policy.decisions,phase_before_error=diagnosis['phase'])
            if isinstance(exc,ConnectionClosed):raise RuntimeError(message) from exc
            raise

def configurations(path):
    config=json.loads(path.read_text());room=config.get('room',2456)
    if not isinstance(room,int) or not 1000<=room<=9999:raise ValueError('个室号须为1000–9999')
    rows=config['accounts']
    if not 1<=len(rows)<=4 or any(r['profile'] not in ('avoid_fourth','balanced','first_place') for r in rows):
        raise ValueError('配置1至4个AI及有效模型类型')
    for row in rows:row['id']=normalize_account(row['id'])
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('每位AI须使用不同天凤账号')
    for row in rows:
        if not valid_account(row['id']):raise ValueError('请在配置中填写完整天凤登录 ID（IDxxxxxxxx-xxxxxxxx），不是昵称')
        name='full-action-selfplay-20261011'+('' if row['profile']=='avoid_fourth' else '-'+row['profile'])
        checkpoint=Path(row.get('checkpoint',str(preferred_checkpoint(BASE/'runs'/name/'latest.pt'))))
        if not checkpoint.is_file():raise ValueError('A local model checkpoint is missing')
        yield PrivateRoomClient(row['id'],row['profile'],checkpoint,room=room)

async def run_all(clients,independent=False):
    # One requested half-game each; no automatic matchmaking after it ends.
    jobs=[asyncio.create_task(c.run()) for c in clients]
    try:
        results=await asyncio.gather(*jobs,return_exceptions=independent)
        if independent:
            errors=[result for result in results if isinstance(result,BaseException)]
            if errors:raise errors[0]
    finally:
        for j in jobs:
            if not j.done():j.cancel()
        await asyncio.gather(*jobs,return_exceptions=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--check',action='store_true');parser.add_argument('--setup',action='store_true');args=parser.parse_args()
    if args.setup:
        if args.config.exists():raise ValueError('Configuration already exists; edit it locally instead of overwriting')
        rows=prompt_accounts()
        descriptor=os.open(args.config,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w') as f:json.dump({'room':2456,'accounts':rows},f,indent=2)
        print('已保存本机配置；账号 ID 不会写入对局日志。');return
    torch.set_num_threads(1);clients=list(configurations(args.config))
    if args.check:
        print(json.dumps(dict(status='configuration_and_models_valid',room=clients[0].room,profiles=[c.profile for c in clients],network_login_tested=False)));return
    asyncio.run(run_all(clients))

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('\n已取消。');sys.exit(1)
    except ValueError as exc:
        print('配置错误：'+str(exc),file=sys.stderr);sys.exit(1)
