# 邀请派遣网站

用户页面只提供与我对战：邀请口令登录，选择天凤或雀魂，输入个室/友人房号，选择1–3位AI的模型后派遣。后台自动分配空闲账号，每次一把四麻赤牌半庄；用户只能查看和停止自己的任务。无账号管理、模型对照、统计、内战训练、段位战入口。

## Windows本机预览

双击start-dispatch-site.cmd。首次设置两个不同口令：管理员口令和给熟人的邀请口令，各至少12个字符，输入不回显。保存文件dispatch-site.local.json仅含口令哈希，不要上传。

用户入口：http://127.0.0.1:8780/
管理员入口：http://127.0.0.1:8780/admin

管理员登录后在后台账号池选择平台，保存天凤完整登录ID（不是昵称）或雀魂国服账号及密码，提供足够独立账号。用户不需要知道任何账号ID。网站使用本机已有dispatch-accounts.local.json；不要同时在旧控制台运行相同账号，否则两套进程无法互相检查占用。保留启动窗口。

## 云服务器部署

目前没有云服务器或域名，未上线公网。网站运行在哪台机器，AI就使用那台机器的网络。

有服务器后：Ubuntu安装Python3.12与venv，解压本包，创建.venv并安装requirements-dispatch.txt。运行 `.venv/bin/python public_dispatch_server.py --setup` 创建口令，再在管理员后台保存账号。

通过HTTPS反向代理公开网站，后端绑定127.0.0.1，启动：

```
.venv/bin/python public_dispatch_server.py --host 127.0.0.1 --port 8780 --origin https://你的域名 --no-browser
```

反向代理需保留Host和Origin，将外部HTTPS转发到本机8780；用systemd持续运行。实际域名与HTTPS证书要在购买服务器后配置。禁止直接公开旧dispatch_server.py，它是本机管理控制台，不是用户网站。

邀请口令只用于网站访问，不是天凤账号密码。管理员口令不可给用户。会话8小时过期；服务器重启后重新登录，历史中断任务不自动派遣。账号文件与口令配置都不打包上传。真实长时间天凤连接稳定性仍需测试。模型仅保留监督基准与v5测试版，旧强化学习版本已撤出；v5未被证明普遍更强。

## 雀魂友人房

在/admin账号池选择雀魂，填写账号别名、账号/邮箱/手机号及密码，选对「账号/邮箱密码」或「手机号密码」。暂不支持短信、扫码、微信/QQ等第三方登录。每位AI一个独立账号，不要同时在官方客户端登录这些AI账号。

房主创建标准四人三赤半庄友人房。用户网站选择雀魂、输入房间号、选择v5和1–3位AI，再派遣。AI入房准备后，由房主手动开始；5分钟未开局会退出。不要将三麻、特殊玩法或东风房号填入。

当前雀魂为实验接入。版本和网关连接已实测；登录、入房、v5决策与完成流程仅用模拟协议测试，没有真实账号完整对战验证。151已定位为Unity版本命名空间和平台握手缺失，已修正；若仍失败，管理员下载任务诊断，提供里面的diagnostics.json即可，无需发送账号密码。

## 20261014.3：雀魂151修复

当前官方客户端为Unity/WebGL，旧`game.ResourceVersion`和`GameMgr`控制台命令不适用。旧`version.json`的0.11.252.w不能用于Unity登录。工具现从官方Unity资源索引解析资源版本（验证时0.16.283）和首页包版本（4.0.47），使用`WebGL_2022-资源版本`；正确解压资源索引后取文件名，不从压缩字节中猜哈希。线路握手在登录和游戏鉴权之前发送平台字段（protobuf第6字段）Web。

服务器已实测：不存在的合成账号由151变为1002（账号不存在）。这验证握手修复，不等于真实账号登录或完整对战已通过。更新后仍需在本机验证。关闭旧进程，覆盖后运行启动脚本，脚本会安装新增的UnityPy依赖；仅刷新网页不会更新后端。手机号密码登录时需在账号配置选择对应登录方式。

解析Unity资源使用[UnityPy](https://github.com/K0lb3/UnityPy)（MIT）；协议核对参考[majsoul-autopilot登录实现](https://github.com/happy-shine/majsoul-autopilot/blob/81365068a4560383329d6b869f19060706fbb1b5/crates/protocol/src/login.rs)和[MajsoulPaipuConvert握手说明](https://github.com/jeff39389327/MajsoulPaipuConvert/blob/4387a3e66cc4e5792cb1df8244d6bf11b95ce718/ms_patch.py)。包内不含官方游戏资源、账号或密码。

20261014.3修正两个Windows启动脚本漏装UnityPy的问题。已有虚拟环境也会检查并补装UnityPy==1.25.4，无需重新安装模型或Torch。若已下载上一包，可在程序目录执行 `.\.venv\Scripts\python.exe -m pip install UnityPy==1.25.4` 后重启。
