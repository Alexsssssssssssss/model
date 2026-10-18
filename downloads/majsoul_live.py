"""Experimental CN username/password friend-room adapter, four-player half-games."""
import asyncio,base64,copy,hashlib,hmac,json,time,urllib.request,uuid
from pathlib import Path
import numpy as np
import torch
from riichienv import RiichiEnv,GameRule,GameType,ActionType
from live_policy import load_policy,tensors,resolve,retain_red_fives
from mahjong_stats import analyze
from majsoul_protocol import Codec,RPC,SoulRPCError
from majsoul_version import resolve_client_version,client_version_string,read_public,PublicFetchError
from urllib.parse import urlsplit
HONORS=['E','S','W','N','P','F','C']
ADAPTER_VERSION='20261019.2-riichi-candidates'
CODES={'dahai':1,'chi':2,'pon':3,'ankan':4,'daiminkan':5,'kakan':6,'reach':7,'ryukyoku':10}
def to_mjai(tile):
 if tile=='':return '?'
 if tile[1]=='z':return HONORS[int(tile[0])-1]
 return '5'+tile[1]+'r' if tile[0]=='0' else tile
def from_mjai(tile):
 if tile in HONORS:return str(HONORS.index(tile)+1)+'z'
 return '0'+tile[1] if tile.endswith('r') else tile

def base_tile(tile):return '5'+tile[1:] if tile.startswith('0') else tile

def fetch_json(url):
 return json.loads(read_public(url))
def gateway():
 version=resolve_client_version()
 bootstrap=fetch_json('https://game.maj-soul.com/1/version.json')['version']
 config=fetch_json('https://game.maj-soul.com/1/v'+bootstrap+'/config.json')
 origin=config['ip'][0]['gateways'][0]['url']
 routes=fetch_json(origin+'/api/clientgate/routes?platform=Web&lang=chs_t&version='+version['package'])['data']['routes']
 domain=next(r['domain'] for r in routes if r.get('ssl'))
 if not domain.split(':')[0].endswith('.maj-soul.com'):raise ValueError('Unexpected Mahjong Soul gateway domain')
 return version,'wss://'+domain

def device_info():
 return dict(platform='pc',hardware='pc',os='windows',os_version='win10',is_browser=True,software='Chrome',sale_platform='web')

def login_request(account,password,version,login_type=0):
 return dict(account=account,password=hmac.new(b'lailai',password.encode(),hashlib.sha256).hexdigest(),random_key=str(uuid.uuid4()),client_version={k:version[k] for k in ('resource','package')},client_version_string=client_version_string(version),tag='cn',type=login_type,gen_access_token=True,reconnect=False,currency_platforms=[1,2,5,6,8,10,11],device=device_info())

def operation_method(request,operation):
 operation=operation or {}
 codes={int(o['type']) for o in operation.get('operation_list',[])}
 reaction=request.get('type') in (2,3,5,9) or request.get('cancel_operation') and not codes.intersection({1,4,6,7,8,10})
 return '.lq.FastTest.inputChiPengGang' if reaction else '.lq.FastTest.inputOperation'

class SoulPolicy:
 def __init__(self,checkpoint,seat):
  self.model=load_policy(checkpoint);self.seat=seat;self.env=RiichiEnv(game_mode=int(GameType.YON_HANCHAN),rule=GameRule.default_mjsoul());self.obs=None;self.events=[];self.pending_reach=False;self.decision_count=0
 def event(self,event):
  if event['type']=='start_game':event=dict(event);event.pop('id',None)
  self.events.append(event);obs=self.env.observe_event(event,self.seat)
  if obs is not None:self.obs=obs
  return obs
 def choose(self,operation,*,discard_tiles=None):
  if operation.get('seat',self.seat)!=self.seat or not operation.get('operation_list'):return None
  if self.obs is None:raise RuntimeError('雀魂可操作事件没有对应模型局面')
  ops=operation['operation_list'];codes={int(o['type']) for o in ops};x,mask,actions=tensors(self.obs,self.model.encoding);allowed={};requests={}
  for label,group in actions.items():
   for action in group:
    msg=json.loads(action.to_mjai());kind=msg['type'];code=CODES.get(kind)
    if kind=='hora':
     # Native legal actions serialize both wins as {"type":"hora"}; no actor/target.
     if action.action_type==ActionType.TSUMO:code=8
     elif action.action_type==ActionType.RON:code=9
     else:raise RuntimeError('无法识别原生和牌动作类型')
    if kind=='none':req={'cancel_operation':True}
    elif code not in codes:continue
    elif kind=='dahai':
     tile=from_mjai(msg['pai'])
     if discard_tiles is not None and tile not in discard_tiles:continue
     req=dict(type=1,tile=tile,moqie=action.tile==self.obs.drawn_tile)
    elif kind in ('chi','pon','ankan','daiminkan','kakan'):
     options=next(o for o in ops if int(o['type'])==code).get('combination',[]);consumed=sorted(from_mjai(t) for t in msg.get('consumed',[]));index=None
     for j,combo in enumerate(options):
      tiles=sorted(combo.split('|'))
      if tiles==consumed or kind in ('ankan','kakan') and all(base_tile(t)==base_tile(from_mjai(msg.get('pai',msg.get('consumed',['?'])[0]))) for t in tiles):index=j;break
     if index is None:continue
     req=dict(type=code,index=index)
    else:req=dict(type=code)
    allowed.setdefault(label,[]).append(action);requests[id(action)]=req
  if not allowed:raise RuntimeError('雀魂服务器与模型合法动作不一致，已停止')
  allowed=retain_red_fives(self.obs,allowed)
  mask[:]=False;mask[list(allowed)]=True
  with torch.no_grad():label=self.model(torch.from_numpy(x)[None],torch.from_numpy(mask)[None])[0].argmax().item()
  action=resolve(self.obs,allowed,label);req=requests[id(action)];msg=json.loads(action.to_mjai());self.decision_count+=1
  if msg['type']=='reach':
   # Preview the local declaration without changing live state. Intersect the
   # resulting legal discards with the server's exact tiles (0m != 5m), BEFORE
   # scoring or red-five retention. Never choose freely then reject afterwards.
   choices=next(o for o in ops if int(o['type'])==7).get('combination') or []
   server_tiles={t for v in choices for t in v.split('|')}
   trial=copy.copy(self);trial.env=self.env.clone();trial.events=list(self.events)
   trial.event({'type':'reach','actor':self.seat})
   local_tiles={from_mjai(a['pai']) for action in trial.obs.legal_actions()
                if (a:=json.loads(action.to_mjai()))['type']=='dahai'}
   candidates=server_tiles & local_tiles
   if not candidates:
    # A missing/incompatible riichi list must not poison state or prevent an
    # otherwise legal normal discard. Do not manufacture a server permission.
    if 1 in codes:
     return self.choose(dict(operation,operation_list=[o for o in ops if int(o['type'])!=7]))
    raise RuntimeError('雀魂立直候选与本地合法切牌没有交集，且服务器未提供普通切牌操作')
   follow={'seat':self.seat,'operation_list':[dict(type=1)]}
   req=trial.choose(follow,discard_tiles=candidates);req['type']=7
   self.event({'type':'reach','actor':self.seat});self.pending_reach=True
   self.decision_count=trial.decision_count
  return req

class SoulRoomClient:
 def __init__(self,account,password,profile,checkpoint,room,status_callback=None,output=None,login_type=0):
  self.account=account;self.password=password;self.profile=profile;self.checkpoint=checkpoint;self.room=room;self.callback=status_callback;self.folder=Path(output);self.result=None;self.policy=None;self.codec=Codec(live=True);self.lobby=None;self.game=None;self.doras=[];self.last_discard=None;self.final_scores=None
  self.login_type=login_type;self.diagnostic={'platform':'majsoul','adapter_version':ADAPTER_VERSION,'region':'cn','room':room,'login_type':login_type,'stage':'initializing','credentials_included':False}
 def status(self,value,**extra):
  self.diagnostic['stage']=value
  if self.callback:self.callback(value,extra)
 def emit(self,event):return self.policy.event(event)
 async def connect_route(self,rpc,url):
  await rpc.open(url)
  route_id=urlsplit(url).hostname.split('.')[0]
  reply=await rpc.call('.lq.Route.requestConnection',dict(type=1,route_id=route_id,timestamp=int(time.time()),platform='Web'))
  if reply.get('result')!=1:raise RuntimeError('雀魂线路握手未被接受')
  return rpc
 def liqi(self,data):
  liqi=data.get('liqi')
  if liqi and not liqi.get('failed'):self.emit(dict(type='reach_accepted',actor=liqi['seat']));self.final_scores[liqi['seat']]-=1000
 def new_doras(self,data):
  doras=data.get('doras',[])
  for tile in doras[len(self.doras):]:self.emit(dict(type='dora',dora_marker=to_mjai(tile)))
  if doras:self.doras=list(doras)
 async def action(self,name,data):
  p=self.policy;seat=data.get('seat',0)
  if name=='ActionNewRound':
   tiles=[to_mjai(t) for t in data['tiles']];doras=data.get('doras') or [data['dora']];self.doras=doras;self.last_discard=None
   hands=[['?']*13 for _ in range(4)];hands[p.seat]=tiles[:13]
   self.final_scores=list(data['scores'])
   self.emit(dict(type='start_kyoku',bakaze=HONORS[data['chang']],kyoku=data['ju']+1,honba=data['ben'],kyotaku=data['liqibang'],oya=data['ju'],dora_marker=to_mjai(doras[0]),tehais=hands,scores=data['scores']))
   if len(tiles)==14:self.emit(dict(type='tsumo',actor=p.seat,pai=tiles[-1]))
  elif name=='ActionDealTile':
   self.liqi(data);self.new_doras(data);self.emit(dict(type='tsumo',actor=seat,pai=to_mjai(data.get('tile','')) if seat==p.seat else '?'))
  elif name=='ActionDiscardTile':
   if data.get('is_liqi') or data.get('is_wliqi'):
    if not (seat==p.seat and p.pending_reach):self.emit(dict(type='reach',actor=seat))
    if seat==p.seat:p.pending_reach=False
   self.new_doras(data);tile=to_mjai(data['tile']);self.last_discard=(seat,tile);self.emit(dict(type='dahai',actor=seat,pai=tile,tsumogiri=data['moqie']))
  elif name=='ActionChiPengGang':
   self.liqi(data);tiles=[to_mjai(t) for t in data['tiles']];sources=data['froms'];other=[i for i,source in enumerate(sources) if source!=seat]
   if len(other)!=1:raise RuntimeError('未知的雀魂副露来源')
   index=other[0];self.emit(dict(type=['chi','pon','daiminkan'][data['type']],actor=seat,target=sources[index],pai=tiles[index],consumed=[t for i,t in enumerate(tiles) if i!=index]))
  elif name=='ActionAnGangAddGang':
   # The notification supplies the base tile, use actual own copies where known.
   tile=to_mjai(data['tiles']);legal=[]
   self.last_discard=(seat,tile) # A following chankan belongs to this kan declarer.
   if seat==p.seat and p.obs is not None:
    legal=[json.loads(a.to_mjai()) for a in p.obs.legal_actions() if json.loads(a.to_mjai())['type'] in ('ankan','kakan')]
   match=next((a for a in legal if a.get('pai',a.get('consumed',[''])[0]).rstrip('r')==tile.rstrip('r')),None)
   if match:self.emit(dict(match,actor=seat))
   elif data['type']==2:self.emit(dict(type='ankan',actor=seat,consumed=([tile.rstrip('r')]*3+[tile.rstrip('r')+'r']) if tile[0]=='5' else [tile]*4))
   else:
    previous=next((e for e in reversed(p.events) if e['type']=='pon' and e['actor']==seat and e['pai'].rstrip('r')==tile.rstrip('r')),None)
    if previous is None:raise RuntimeError('找不到加杠前的公开碰牌，保存对局并停止')
    self.emit(dict(type='kakan',actor=seat,pai=tile,consumed=[previous['pai'],*previous['consumed']]))
   self.new_doras(data)
  elif name=='ActionHule':
   self.final_scores=list(data['scores'])
   for h in data['hules']:
    target=h['seat'] if h['zimo'] else self.last_discard[0] if self.last_discard else None
    if target is None:raise RuntimeError('无法定位雀魂放铳者')
    self.emit(dict(type='hora',actor=h['seat'],target=target,scores=data['scores'],**({'deltas':data['delta_scores']} if len(data['hules'])==1 else {})))
   self.emit(dict(type='end_kyoku'))
   if not data.get('gameend'):await self.game.call('.lq.FastTest.confirmNewRound')
   return
  elif name in ('ActionNoTile','ActionLiuJu'):
   delta=[0]*4
   if name=='ActionNoTile':
    for item in data.get('scores',[]):delta=[a+b for a,b in zip(delta,item['delta_scores'])]
   self.final_scores=[a+b for a,b in zip(self.final_scores,delta)]
   self.emit(dict(type='ryukyoku',deltas=delta));self.emit(dict(type='end_kyoku'))
   if not data.get('gameend'):await self.game.call('.lq.FastTest.confirmNewRound')
   return
  elif name=='ActionMJStart':
   # Empty start-animation marker; start_game is emitted after authGame.
   return
  else:raise RuntimeError('尚不支持的雀魂对局事件 '+name)
  operation=data.get('operation') or {}
  # ActionNewRound normally carries the dealer's discard operation together
  # with the 14-tile hand.  Some server versions omit that field, which used
  # to leave the dealer waiting forever on the first discard.  A 14-tile
  # initial hand is unambiguous: the only required operation is dahai.
  if name=='ActionNewRound' and len(tiles)==14 and not operation.get('operation_list'):
   operation={'seat':p.seat,'operation_list':[{'type':1}]}
  req=p.choose(operation)
  if req:
   await asyncio.sleep(.7);await self.game.call(operation_method(req,operation),req)
 async def run(self):
  self.folder.mkdir(parents=True,exist_ok=True);self.status('connecting');heartbeat=None
  try:
   version,route=await asyncio.to_thread(gateway);self.diagnostic.update(client_version=version,gateway=route,route_platform='Web');self.lobby=RPC(self.codec);await self.connect_route(self.lobby,route+'/gateway');self.status('logging_in')
   reply=await self.lobby.call('.lq.Lobby.login',login_request(self.account,self.password,version,self.login_type))
   account_id=reply['account_id']
   await self.lobby.call('.lq.Lobby.loginSuccess')
   if reply.get('game_info',{}).get('game_uuid'):raise RuntimeError('雀魂账号仍有未结束对局，先在官方客户端处理')
   self.status('joining_private_room')
   room=await self.lobby.call('.lq.Lobby.joinRoom',dict(room_id=self.room,client_version_string=client_version_string(version)))
   if room['room']['max_player_count']!=4 or room['room']['mode']['mode']!=2:raise RuntimeError('只支持四人半庄友人房')
   detail=room['room']['mode'].get('detail_rule',{})
   if detail.get('dora_count',3)!=3:raise RuntimeError('实验接入只支持标准三赤规则')
   if any(detail.get(k) for k in ('guyi_mode','muyu_mode','xuezhandaodi','chuanma','huansanzhang','field_spell_mode','begin_open_mode')):raise RuntimeError('不支持特殊玩法房间')
   await self.lobby.call('.lq.Lobby.readyPlay',{'ready':True});self.status('waiting_in_private_room')
   async def heartbeat_loop():
    while True:
     await asyncio.sleep(10);await self.lobby.call('.lq.Lobby.heatbeat',{'no_operation_counter':0})
   heartbeat=asyncio.create_task(heartbeat_loop());deadline=time.monotonic()+300;start=None
   while start is None:
    name,data=await asyncio.wait_for(self.lobby.notifications.get(),max(.1,deadline-time.monotonic()))
    if name=='lq.NotifyRoomGameStart':start=data
    elif name in ('connection_closed','lq.NotifyRoomKickOut','lq.NotifyAnotherLogin'):
     self.diagnostic['connection_failure']=data if name=='connection_closed' else {'notification':name};raise RuntimeError('雀魂入房连接已中断')
    if time.monotonic()>=deadline and start is None:raise RuntimeError('5分钟内未开局，请检查房主是否开始对局')
   suffix='/game-gateway-zone' if start.get('location')=='zone' else '/game-gateway';self.game=RPC(self.codec);await self.connect_route(self.game,route+suffix)
   auth=await self.game.call('.lq.FastTest.authGame',dict(account_id=account_id,token=start['connect_token'],game_uuid=start['game_uuid'],device=device_info()))
   seats=auth['seat_list'];seat=seats.index(account_id);self.policy=SoulPolicy(self.checkpoint,seat);self.emit(dict(type='start_game',id=seat,names=['player']*4,log=start['game_uuid']))
   entered=await self.game.call('.lq.FastTest.enterGame')
   if entered.get('game_restore'):raise RuntimeError('当前版本不支持从中途对局恢复，请开始新半庄')
   self.status('playing');deadline=time.monotonic()+7200
   while True:
    name,data=await asyncio.wait_for(self.game.notifications.get(),max(.1,deadline-time.monotonic()))
    if name=='lq.ActionPrototype':
     raw=bytearray(base64.b64decode(data['data']));keys=(132,94,78,66,57,162,31,96,28)
     decoded=bytes(v^(((23^len(raw))+5*i+keys[i%9])&255) for i,v in enumerate(raw))
     event=self.codec.decode(data['name'],decoded);await self.action(data['name'],event)
    elif name=='lq.NotifyGameEndResult':
     scores=self.final_scores
     if scores is None:raise RuntimeError('缺少雀魂终局分数')
     order=sorted(range(4),key=lambda s:(-scores[s],s));ranks=[order.index(s)+1 for s in range(4)]
     self.emit(dict(type='end_game',scores=scores));self.result=dict(scores=scores,ranks=ranks,own_seat=seat,own_stats=analyze(self.policy.events,scores,ranks)[seat],checkpoint_sha256=self.policy.model.source_sha256,profile=self.profile);break
    elif name in ('connection_closed','lq.NotifyGameTerminate'):
     self.diagnostic['connection_failure']=data if name=='connection_closed' else {'notification':name};raise RuntimeError('雀魂对局连接中断，未训练此局（'+str(data.get('error_type',name))+'，关闭码'+str(data.get('close_code','未提供'))+'）')
   (self.folder/'result.json').write_text(json.dumps(self.result,ensure_ascii=False,indent=2),encoding='utf-8');self.status('completed');return self.result
  except BaseException as exc:
   self.diagnostic['error_type']=type(exc).__name__
   if isinstance(exc,PublicFetchError):self.diagnostic['network_failure']=exc.details
   if isinstance(exc,SoulRPCError):self.diagnostic.update(method=exc.method,error_code=exc.code)
   # Never copy protocol payloads, access tokens, passwords or account IDs into diagnostics.
   message=str(exc)
   for value in (self.account,self.password,hmac.new(b'lailai',self.password.encode(),hashlib.sha256).hexdigest()):
    if value:message=message.replace(value,'[redacted]')
   self.diagnostic['error']=message[:500]
   raise
  finally:
   (self.folder/'diagnostics.json').write_text(json.dumps(self.diagnostic,ensure_ascii=False,indent=2),encoding='utf-8')
   if self.policy:(self.folder/'public-events-and-own-actions.jsonl').write_text(''.join(json.dumps({'event':e},ensure_ascii=False)+'\n' for e in self.policy.events),encoding='utf-8')
   if heartbeat:heartbeat.cancel();await asyncio.gather(heartbeat,return_exceptions=True)
   if self.game:await self.game.close()
   if self.lobby:
    try:await self.lobby.call('.lq.Lobby.leaveRoom')
    except Exception:pass
    await self.lobby.close()
