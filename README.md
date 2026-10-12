# 天凤个室 2456：三个自训麻将 AI

[下载 Windows 接入包](https://raw.githubusercontent.com/Alexsssssssssssss/model/main/downloads/tenhou-room-2456.zip)

避四、平衡、争一模型合计完成 50,000 个训练半庄。接入包使用独立种子验证后选定的较早检查点，不直接使用最终训练权重。

解压后，安装 64 位 Python 3.12，双击 `start-tenhou-2456.cmd`。按提示在自己的电脑输入三个 AI 账号 ID，然后本人进入 https://tenhou.net/3/?L2456 ，预约四人半庄、食断、赤牌。完整说明在压缩包的 README.md。

包内不含账号凭据或原始牌谱。账号只保存本机，三个 AI 每次各打一场半庄。不要把本人账号作为 AI 账号，也不要同时在其他客户端登录这些 AI 账号。

已通过模型加载、511 次公开牌谱合法决策、私房 JOIN 限制和 WebSocket 握手验证；真实账号登录及完整在线对战尚未验证。重连后的 REINIT 状态暂不支持。模型强度的评估仅针对固定监督模型基准，不能换算为天凤段位。

天凤协议网关来自 [tomohxx/mjai-gateway](https://github.com/tomohxx/mjai-gateway)，MIT 许可证随包附带；没有使用其策略模型或训练牌谱。

SHA-256：`f7493a3d4aa674b61b1db7c350ed25c420c1578baa9d5cd8dad9d28e00720f2b`

## 网页对战控制台

[下载 AI 控制台（ZIP）](https://raw.githubusercontent.com/Alexsssssssssssss/model/main/downloads/ai-dispatch-console.zip) · [使用说明](downloads/AI-CONSOLE-GUIDE.md)

解压后双击 `start-ai-console.cmd`，需要 Python 3.12。账号在本机网页中填写。可选天凤个室号、模型和账号，派遣 1–3 个 AI 或使用四个账号进行内战与分批强化学习，保存统计 CSV。四账号连续在线循环尚未实际验证。

雀魂提供尚未真实账号验证的实验邀请接入；宝牌 / 攻防 v2 模型待实战强度验证。安装包不包含账号、密码或原始训练牌谱。旧2456工具仍保留。
