"""Use this project's full-action models in a specified Tenhou private room."""
import argparse, asyncio, json, re, sys, time, getpass, os, unicodedata
from pathlib import Path
import numpy as np
import torch
from websockets.asyncio.client import connect
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
    def __init__(self,checkpoint):
        self.model=load_policy(checkpoint)
        self.env=RiichiEnv(game_mode=int(GameType.YON_HANCHAN),rule=GameRule.default_tenhou())
        self.decisions=0

    def react(self,event):
        obs=self.env.observe_event(event,0)
        if obs is None:return {'type':'none'}
        x,mask,actions=tensors(obs)
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
        with torch.no_grad():label=self.model(torch.from_numpy(x)[None],torch.from_numpy(mask)[None])[0].argmax().item()
        action=resolve(obs,allowed,label);msg=json.loads(action.to_mjai())
        if msg['type']=='dahai':msg['tsumogiri']=action.tile==obs.drawn_tile
        self.decisions+=1
        return msg

class PrivateRoomClient:
    def __init__(self,account,profile,checkpoint,room=2456,game_type=9):
        if not 1000<=room<=7999:raise ValueError('A private room between 1000 and 7999 is required')
        if game_type!=9:raise ValueError('This launcher supports four-player red-five half-games only')
        self.account=account;self.profile=profile;self.room=room;self.game_type=game_type
        self.policy=OnlinePolicy(checkpoint);self.state=State(account,f'{room}_{game_type}')
        self.ended=False;self.packet={};self.log=None;self.status_path=None

    async def send_event(self,event):
        event=dict(event)
        if event['type']=='start_kyoku':
            event['kyoku']+=1 # Upstream uses zero-based kyoku; MJAI is one-based.
            event['scores']=[int(v)*100 for v in self.packet['ten'].split(',')]
        if event['type']=='hora':
            event.update(actor=int(self.packet['who']),target=int(self.packet['fromWho']))
        # Reconnection with incomplete state must not produce guessed decisions.
        if event['type']=='start_game':event['names']=['AI','player1','player2','player3']
        result=self.policy.react(event)
        if self.log:
            self.log.write(json.dumps({'event':event,'action':result},ensure_ascii=False)+'\n');self.log.flush()
        if event['type']=='end_game':self.ended=True
        return result

    async def send_packet(self,ws,message):
        if message.get('tag')=='JOIN' and message.get('t')!=f'{self.room},{self.game_type}':
            raise RuntimeError('Refusing a JOIN outside the requested private room')
        await ws.send(json.dumps(message))

    async def run(self):
        folder=BASE/'runs/tenhou-live'/str(time.time_ns());folder.mkdir(parents=True)
        self.status_path=folder/'status.json'
        def status(value,**kw):
            self.status_path.write_text(json.dumps(dict(profile=self.profile,room=self.room,status=value,**kw),ensure_ascii=False,indent=2))
        status('connecting')
        settings.DEBUG=False # Keep the upstream human-play action delays.
        try:
            with (folder/'public-events-and-own-actions.jsonl').open('w') as self.log:
                async with connect('wss://b-ww.mjv.jp',origin='https://tenhou.net',open_timeout=20) as ws:
                    await self.send_packet(ws,{'tag':'HELO','name':self.account,'sx':'M'})
                    async def keepalive():
                        while not self.ended:
                            await ws.send('<Z/>');await asyncio.sleep(10)
                    heartbeat=asyncio.create_task(keepalive())
                    try:
                        async for raw in ws:
                            self.packet=json.loads(raw)
                            tag=self.packet.get('tag','')
                            if tag in ('ERR','ERROR','REINIT'):
                                raise RuntimeError(f'Tenhou returned {tag}; no automatic reconnect or resubmit')
                            if tag=='HELO':status('waiting_in_private_room')
                            if tag=='INIT':status('playing')
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
            status('completed',decisions=self.policy.decisions)
        except BaseException as exc:
            status('stopped_on_error',error=type(exc).__name__+': '+str(exc),decisions=self.policy.decisions)
            raise

def configurations(path):
    config=json.loads(path.read_text());room=config.get('room',2456)
    if room!=2456:raise ValueError('This match was requested for room 2456')
    rows=config['accounts']
    if len(rows)!=3 or {r['profile'] for r in rows}!={'avoid_fourth','balanced','first_place'}:
        raise ValueError('Provide one account for each of the three profiles')
    for row in rows:row['id']=normalize_account(row['id'])
    if len({r['id'] for r in rows})!=3:raise ValueError('Three distinct Tenhou accounts are required')
    for row in rows:
        if not valid_account(row['id']):raise ValueError('请在配置中填写完整天凤登录 ID（IDxxxxxxxx-xxxxxxxx），不是昵称')
        name='full-action-selfplay-20261011'+('' if row['profile']=='avoid_fourth' else '-'+row['profile'])
        checkpoint=Path(row.get('checkpoint',str(preferred_checkpoint(BASE/'runs'/name/'latest.pt'))))
        if not checkpoint.is_file():raise ValueError('A local model checkpoint is missing')
        yield PrivateRoomClient(row['id'],row['profile'],checkpoint,room=room)

async def run_all(clients):
    # One requested half-game each; no automatic matchmaking after it ends.
    jobs=[asyncio.create_task(c.run()) for c in clients]
    try:await asyncio.gather(*jobs)
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
        print(json.dumps(dict(status='configuration_and_models_valid',room=2456,profiles=[c.profile for c in clients],network_login_tested=False)));return
    asyncio.run(run_all(clients))

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('\n已取消。');sys.exit(1)
    except ValueError as exc:
        print('配置错误：'+str(exc),file=sys.stderr);sys.exit(1)
