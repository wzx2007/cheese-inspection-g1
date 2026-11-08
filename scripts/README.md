# 运行入口

- [手臂接口](armctrl/README.md)
- [腿部接口](legctrl/README.md)
- 视觉识别参数：[vision/config.py](vision/config.py)

在本目录运行，先设置 SDK 环境：

```bash
export PYTHONPATH=/data/galbot/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/data/galbot/lib:$LD_LIBRARY_PATH
```

## 主程序：三次抓取 + 双面识别 + 笑声

```bash
python3 main.py --confirm-motion              # 完整演示
python3 main.py --confirm-motion --debug      # 每步回车，q 停止
python3 main.py --confirm-motion --no-vision  # 只跑动作
```

每一轮都是**完整的一次抓放**，阶段名和 `galbot_nav` 原管线一致，只在两处插了"查看"：

```text
gripper_open -> arm_reset -> height_ready -> [look_front]
-> arm_ready -> reach_forward -> grasp_close -> lift -> flip
-> [look_back] -> unflip -> lower -> release -> arm_reset2
```

跑完一轮**不自己开回起点**：停下来，由人把车推到下一个起点，按回车才继续。
全自动模式也卡这一道（`--debug` 只是额外多出每步回车）。推完会回读位姿、提示
离桌子多远；定位丢了会要求重推，不带着坏定位去导航。

> 2026-11-07 现场改的。展馆地面人来人往，自主返航又慢又容易被安全边距顶住
> （日志里 `CLOSE_TO_OBSTACLE` + `clearance=0.246 < required 0.35`，
> 而且卡的是**起点**，把预停位从 0.8 m 换到 2.2 m 六个点全白跑）。
> 人推两秒比机器绕半分钟靠谱。

参数在 `demo_cheese.py` 开头。

两个查看**都不额外升降高度**，直接落在标定过的位姿上：

| 查看 | 时机 | 身高 | 对应原管线窗口 |
| --- | --- | --- | --- |
| `look_front` | 抓取之前，奶酪还在桌上、双臂高举 | 1.37 m | 0~5 帧 |
| `look_back` | `flip` 之后，仍夹着 | 1.54 m | 200~205 帧 |

2026-11-07 现场：反面那次曾经先降到 1.34 m 再看，平面被压成 30 px 窄带又被顶灯
高光冲白，霉菌 Black-Hat 深度从 192 掉到 65，判据判不出来。别为了"看清楚"去降高度。

每次"查看"连拍 6 帧判霉菌（`vision/`），**判完这一面就立刻播报这一面的结果**：
没霉 → `voice/result_pass.wav`（"检测完成，表面未发现明显异常。"），
有霉 → `voice/result_fail.wav`（"检测完成，发现疑似异常，请师傅复核。"）。
三次跑完、收身之后，再放 `voice/goodbye.wav`。

> 2026-11-08 修正：这两支音频的语义之前接反了（有霉反而播"未发现异常"）。
> 现在已经按音频原意对齐，台词也用 DashScope ASR 逐个核对过了。

播报是非阻塞的（`voice/play_clip.py` 的 `VoiceBox` 用队列 + 后台线程），
说话的同时手臂继续走动作，不拖长演示；收尾的 goodbye 用 `wait=True` 等它播完，
否则后面的 `request_shutdown` 会把尾音截掉。

三支音频都是 **32 kHz**，而 SDK 的 `write_audio_stream_output` 只吃
16 kHz / 16 bit / 单声道 PCM，所以要转一次；`VoiceBox` 会转并缓存成
`*_16k_mono.pcm`（已预转好一起部署，机器人上不必现场转）。

### 霉菌判据：两条线索，对应两种视角下**霉的两种外观**

> **2026-11-08 改口径：顶面范围用用户框死的固定矩形 `config.TOP_ROI`，不再按轮廓自标定。**
> 用户原话："你不能靠侧面作为判断依据，而是看顶面，要在顶面的 ROI 范围内去找"、
> "上机的时候也要框这个 ROI，而不是你自己去图片中找，因为我给你框的一定是准的"。
> 侧壁那些白色蹭痕/反光**不是霉** —— 上一版扫整块奶酪内部，命中的 `(615,821)`
> 就在侧壁下部，被用户当场否掉。

反面（= 已翻腕后看到的顶面）的霉**不是黑的**。2026-11-07 23:36 复盘查出来：
顶面整幅里近纯黑像素为 0（`BH_V` 最大只有 134），而同一块奶酪在正面是
`vmin=0 / vmed=20 / S=255` 的近纯黑块。所以"顶面有霉"用黑霉判据测，必然漏检。

| 线索 | 视角 | 搜索范围 | 判据 | 实测 |
| --- | --- | --- | --- | --- |
| 亮度黑帽 `BH_V`（`cue=V`） | 正面（未翻腕），看侧壁 | 整块奶酪 | `BH_THR=45`、斑块内 `>=100` | 真霉 164~255；表面暗纹理只有 72~74 |
| 白/灰霉饱和度（`cue=W`） | 反面（已翻腕），看顶面 | **只在 `TOP_ROI` 框内** | 斑内中位 `S∈[50,105]`、局部凹陷 `>=18`、`V>=175`、面积 `>=120`、长宽比 `<=9` | 有霉 14/14 帧命中、没霉 0/14 误报 |

白霉线索为什么落在**饱和度**上：三种物质在 S 上分得很开，中间那条带就是霉。

```
镜面过曝   S ≈ 30~55    顶面被顶灯打白           -> 排除
白霉       S ≈ 50~80    灰白覆盖膜，V 依然很高   -> 目标
奶酪本体   S ≈ 110~135（顶面）/ 200~215（侧壁）
```

判的是"比 21×21 局部中值掉 >=18"，不是绝对饱和度，所以整幅偏明偏暗都不影响。

`TOP_ROI = (620, 520, 800, 610)`（1280×960 原图坐标）。两个附带好处：

1. **正面查看时框内奶酪面积为 0**，白霉线索天然不会触发，不用再写"视角判别"。
2. **不再需要长宽比去挡棱线**。曾经把 `WHITE_MAX_AR = 5.0` 想掐掉"顶面棱线薄带"，
   结果把真斑（`29×5`，`ar=5.8`）一起掐掉了 —— 那 29×5 就是霉本身，命中率直接 0/14。
   固定框之后棱线根本不在框里，长宽比放宽到 9.0。

**已删掉的 `BH_b`（顶面黄度黑帽）线索**：复盘 2252 + 23:36 共 24 帧反面，它的落点
**全部**在顶面带的上/下边界内 4~7 px（2252 那 6 帧的 cy 都是 550，带顶 544），
是轮廓转折棱线的边缘响应，从没命中过真霉。这条线索曾让 3 个反面判成"有霉"。

`mold.top_face()` / `cheese_fully_visible()` 都保留但**判决已不用**（标了 `[已废弃]`）。
自标定的顶面带会随相机高度和奶酪位置漂移，一漂就跑到侧壁上，这正是上次判错的根因。

回归脚本 `_tmp/regress_vision.py`：12 个已标注面，当前 0 错。
`mold.annotate()` 会把 `TOP_ROI` 画成青框，存证据时一眼能看出斑是不是在框内。

细节见上级 [README](../README.md)。

## 终点分流状态机（2026-11-08 新增）

三次跑完、道别之后，按"识别到几个"决定去哪（`mission/route.py`）：

| 个数 | 动作 |
| --- | --- |
| 0 | 保持不动，不调用导航 |
| 1 | 去终点 **A** |
| 2 | 去终点 **B** |
| ≥3 | 没有定义 -> 保持不动 + 打印警告（不自己编规则） |

```bash
python3 main.py --confirm-motion --route                 # 开启分流
python3 main.py --confirm-motion --route --count-mode spots
```

**终点坐标现在还是空的**（`mission/route.py` 开头的 `GOAL_A` / `GOAL_B`，`x/y/yaw_deg`
都是 `None`）。留空时状态机不会瞎跑：`decide()` 会打印"终点 b 坐标还没配"并降级成
保持不动。补坐标两种方式，**都优先于字典里的值**：

1. 直接填 `mission/route.py` 的 `GOAL_A = dict(name="route_a", x=…, y=…, yaw_deg=…)`。
2. 放 `nav_goals/route_a.json`（现场用 `main.py nav --record route_a` 录出来的格式），
   即 `{"pose": [x,y,z,qx,qy,qz,qw], "yaw_deg": …}`。

"个数"的口径用 `--count-mode` 选：

- `faces`（默认）：**有霉的面数**，三圈六面里几个面有霉 -> 0~6。语义最直白。
- `spots`：单次查看里**菌斑个数**的最大值。一颗霉被切成几块时口径偏碎。

只统计 `reason == "ok"` 的查看 —— 没分出奶酪那次既不算有霉也不算没霉，它没有结论。

状态：`IDLE -> COUNTED -> HOLDING / MOVING -> DONE / FAILED`。
`FAILED`（导航失败、误差超容差、没定位）**不自动重试**，等人判断。
离线回归 `_tmp/regress_route_voice.py`。

## 语音驱动全流程：`xingbao.py`（2026-11-08 新增，主入口）

`main.py` / `demo_cheese.py` 是按键驱动（跑三圈、每圈等人按回车）。`xingbao.py`
把**回车换成说话**：星宝醒着待命，人跟它讲一句，它自己决定动不动手、动几块。

```bash
bash stop_voice.sh                        # 先把别的语音进程清掉（重要！见下面"现场三个坑"）
python3 xingbao.py --self-test            # 离线自检意图/结论/记忆/搜索判定，不连机器人
python3 xingbao.py --talk-only            # 对讲模式：只连音频，只听不动（安全练嘴）
python3 xingbao_sim.py                    # 模拟：机器人一动不动跑一轮完整流程
python3 xingbao.py --confirm-motion       # 语音主控，默认一块
python3 xingbao.py --confirm-motion --route   # 测完按霉数分流到终点（默认不开）
```

日志默认写 `/tmp/xingbao.log`（同时打在终端）。启动时会打印一段横幅，写清
"检测模式/对讲模式、默认测几块、分流开关、唤醒引擎与阈值、日志和记忆路径"。

### 现场三个坑（2026-11-08 凌晨查清，都是"看起来像星宝不动"）

**坑 1：跑错脚本。** `main.py --confirm-motion` 是**键盘驱动**的老演示（3 圈、
每圈按回车），它**完全不听语音**。对着它喊"星宝星宝"当然毫无反应。现在 `main.py`
一启动就会把这句提示打在最前面。

**坑 2：`ask.py --voice` 冒充星宝。** 那是纯问答脚本，**永远不会动**，但它照样会
回答"来啦，正反面都扫一遍——嗯，干干净净，没霉"，听起来就像已经检测完了。
现在它的开场白改成"问答模式，我只聊天，不会动"，并且给模型的上下文里写明
"你没有手臂、不许假装看过奶酪"。

**坑 3：残留进程堆一堆。** 现场同时挂着 **5 个** `ask.py --voice`，一起念 TTS、
一起抢麦克风，日志全写进 `/tmp/xingbao.log` —— 用户看日志以为星宝在跑，其实
星宝本体一句都没收到。现在：

* `xingbao.py` / `ask.py --voice` 抢同一把文件锁（`/tmp/xingbao_voice.lock`）；
* 启动前扫 `/proc`（按 argv 判，不会把自己和 `timeout`/`bash` 包装当成冲突），
  发现别的语音进程就**拒绝启动**并打印 PID 和处理办法；`main.py` 这类只是提醒；
* `bash stop_voice.sh` 一键清场，`--list` 只看不动手。

`xingbao_sim.py` 是现场自测：六段全部走真东西（真音频、真麦克风、真唤醒模型、
真 ASR/TTS、真 TTS 汇报、真记忆、真联网搜索），只有 `robot/arm/leg/nav` 换成
假对象 —— **机器人全程不动**，一轮就结束。第 5 段是把识别出来的那句话送进
`VoiceCore.handle()`，走的就是现场"喊完话 → 判意图 → 干活"的同一条路（以前的
模拟直接调 `run_task`，会漏掉意图这一环）。`--no-flow` 只验语音启动，
`--fast` 跳过 TTS/ASR。

现场台词（说了就会触发对应动作）：

| 说什么 | 意图 | 星宝干什么 |
|---|---|---|
| 星宝星宝，你去检测一下面前这块奶酪 | `start_task` | 跑完整管线：导航 → 看正面 → 抓 → 翻腕 → 看反面 → 放回 → 收身 → 汇报 |
| 星宝星宝，把这三块都检测一下 | `start_task`(3) | 同上跑三圈，每圈测完立刻汇报，圈间用语音确认推车 |
| 再检测一次 / 再来一块 / 重来 / 帮我再看看 | `start_task` | 直接再跑一轮，**不会**回"刚刚已经测过了" |
| 检测结果怎么样 / 刚才那块有霉吗 | `status` | 用记忆回答上一块的结论 |
| 停 / 别做了 / 好了不要了 / 够了 | `stop` | 在**检查点**停下（不是物理急停） |
| 好了 / 推好了 / 继续 | `confirm` | 回答星宝"推好了吗"的追问，继续下一圈 |
| 今天天气怎么样 / 最近有什么新闻 | `question` | 云端**联网搜索**后回答，不往奶酪上扯 |
| 其它任何话 | `question` | 正常对话（带上下文记忆），一律不拒绝 |

汇报口径（`mission/report.py`）：两面都有霉 / 一面有霉另一面干净 / 两面干净合格 /
有面没看清不敢下结论。**没看清 ≠ 干净**，`reason != "ok"` 的查看不计入任何一边。

第二轮唤醒：一圈结束星宝问"请把我推到下一个起点，推好了跟我说一声"，人说"好了"
就继续 —— 这就是原来的那声回车。人不说话它不动，说"停"就收工。

记忆（`voice/memory.json`）：短时对话 40 条 + 检测记录 30 条，跨进程保留。
`--memory-reset` 换场时清空。让它答"刚才那块怎么样"走的是这份记忆，不是现场重测。

意图识别用**规则**（`voice/intent.py`）而不是大模型：涉及运动，必须确定性可预测。
判定顺序是"先看停 → 再看结果 → 最后看开始"，所以"停一下，检测结果怎么样"判停。
"怎么样"这类弱疑问必须配着奶酪/检测/它才算问结果，否则"今天天气怎么样"会被
误判成"问检测结果"（2026-11-08 修）。

### 联网搜索与"不拒绝"（2026-11-08）

提示词重写过：旧版"拿不准就说不知道"让模型拒绝一切角色外的问题，现在明确
"永远不要拒绝用户 / 已经有记忆的奶酪让再测就再测 / 天气新闻常识都正常答"。

搜索走 DashScope 的 `enable_search` + `search_options`，实测有两个坑：

1. **只给 `enable_search: true` 没用**，模型照样回"我不具备联网能力"；
2. 必须加 `search_options.forced_search` 才生效。默认 `false`（让模型自己判断，
   天气题 2 秒就搜），只有模型明明不知道还在推辞时才由 `Brain.answer()`
   强制搜第二次。奶酪相关问题不带搜索，省 2 秒延迟。
3. compatible-mode **不下发来源**（没有 `search_info`），所以别想着播报出处。

安全边界：语音不能代替急停；照旧要 `--confirm-motion`；移动期间不监听语音，
只在检查点开口；导航失败/看不清会说出来并停住，不自己瞎试。

### 唤醒分数的两个坑（2026-11-08 查清）

* `ask.py --wake-test` 之前用 `mic.wav_to_pcm()`，那个函数**不重采样**，只把任意
  采样率的数据当 16 kHz 读。24 kHz 的 TTS 素材会被拉成 1.5 倍慢放，分数虚高
  （0.94 对真实 0.03）。现在统一走 `play_clip.ensure_pcm()`，和上机播放同一条路。
* **TTS 语速约 5 字/秒，比真人快得多**，唤醒模型（真人录音训的）在原速下只有
  0.03；把它放慢到真人节奏（`atempo=0.67`）立刻升到 0.70。真人说话不受影响
  （现场已验证"挺准的"），所以别拿 TTS 音频调阈值。
* **ASR 会把"星宝星宝"识别成"星爆"** —— 现场日志里真的出现过
  `星，星爆，星爆。`。文本唤醒降级路径靠识别结果匹配唤醒词，所以
  `WAKE_TOKENS` 现在把 `星爆 / 星抱 / 兴宝 / 星保` 都算唤醒，否则模型一旦加载
  失败退回文本路径，真人喊十遍也唤不醒。

### 唤醒：根因是 TFLite 前处理不兼容，已换 sherpa-onnx 关键词模型（2026-11-08 第五次现场反馈）

现场原话："星宝现在唤不醒，是不是没起录音还是什么问题？" 当时已经连试
`--wake-th 0.15 / 0.08 / 0.04` 三档，全都唤不醒。**0.04 已经贴到噪声底了，
再降没有意义**——负样本实测才 0.001。

量出来的事实（`wake_probe.py` / `wake_why.py`，都不动机器人）：

| 检查 | 实测 | 结论 |
|---|---|---|
| 音频流有没有起来 | 466 块 / 37 s，块大小恒为 2560 字节 | **起来了**，不是"没起录音" |
| 数据干不干净 | 16 kHz / 1 声道 / S16_LE，DC 偏置 -13 | 干净 |
| 现场底噪 | 本底中位 **1398**，`p90` 能到 6253 | 很吵（风机 + 人群） |
| 人声进不进得来 | 喊话时 RMS 冲到 **7198**（5.1 倍信噪比） | **进得来** |
| 模型给多少分 | 全程 **0.001**，连 TTS 合成音也只有 0.001 | **模型没在"听"** |

同一个脚本里还查了两件容易误判的事：

* `hw_params` 里 `channels: 12` —— 那是 12 麦阵列的**硬件**参数；SDK 交给
  Python 的已经是降混后的单声道（443 个录音全是 `sr=16000 ch=1`），不是我们
  把多声道当单声道读错了。
* 块率 15.11 块/s 比理论 12.5 块/s 略高（隐含 19345 Hz）。这是麦克风链路自己的
  时钟，**不足以**把 0.7 的分压成 0.001，别在这上面绕。

**真正的根因：前处理模型和 TFLite 运行时版本不兼容。**

`xingbao.tflite` 的输入签名是 `(1, 16, 96)`，确认它只是 openWakeWord 三段式的
**最后一环**，前面必须配 `melspectrogram.tflite` + `embedding_model.tflite`。而：

```
melspectrogram.tflite
  默认（XNNPack）           -> XNNPack delegate failed to reshape runtimeNode 27
  experimental_delegates=[] -> 同上
  BUILTIN（纯 builtin）      -> Node number 3 (CONV_2D) failed to prepare
  BUILTIN_REF               -> Node number 3 (CONV_2D) failed to prepare
```

**这个 mel 模型在新版 TFLite 上根本 allocate 不了。** mel 出不了有效特征，
embedding 和分类器拿到的就全是垃圾，输出恒定 0.001 —— 阈值降到 0.04 也没用，
因为它压根没在听。旧方案把命门交给了"运行时恰好支持这个 mel 模型"，现场换个
TFLite 版本就崩，而且崩得很安静：不报错，只是没分。

**新方案：`voice/kws.py`（sherpa-onnx zipformer KWS）**

单一 ONNX 模型（encoder/decoder/joiner，int8 共 **4.8 MB**），特征提取在 C++ 里
做，**没有独立的 mel 模型**，所以不存在上面那类版本错配。唤醒词用拼音写在
`voice/model_kws/keywords.txt`，改唤醒词不用重训：

```
x īng b ǎo x īng b ǎo @星宝星宝
x īng b ǎo @星宝
```

实测（PC 端，同一套模型）：

| 测试 | 结果 |
|---|---|
| 官方 7 段真人音频 + 官方词表 | 命中 4/7（其余不含该词） |
| 自定义「星宝星宝」跑那 7 段无关音频 | **误唤 0/7** |
| 自定义「星宝星宝」跑 TTS 合成音频 | **命中 4/4**（纯词 / 带后接句 / 连说两遍 / 礼貌句） |
| 三个播报 wav（goodbye / pass / fail） | 命中 **0/3** |

机器人上装一次依赖：

```bash
python3 -m pip install --user sherpa-onnx
```

后端怎么选：

```bash
python3 xingbao.py --confirm-motion        # 默认 kws，只用模型
python3 xingbao.py --wake-engine model     # 旧的 openWakeWord（对比/排查）
python3 xingbao.py --wake-engine auto      # 先 kws，不行再 model
python3 xingbao.py --wake-engine text      # 不用模型，靠识别文本找"星宝"
python3 xingbao.py --wake-fallback asr     # 应急：模型不出分时才用 ASR 兜底
```

**排查顺序（别再先从阈值下手）**：

```bash
# 1. 麦克风 + 模型分，一次看清（机器人不动，要人喊）
python3 wake_probe.py 20
python3 wake_probe.py 100 kws cue   # 正式实测：带"开始收音"发令音
python3 wake_why.py 30        # 更细：块率、设备参数、频谱、多采样率假设

# 2. 离线对照：拿"确定含星宝星宝"的音频喂模型（不用麦克风、不用人喊）
python3 wake_health.py        # 打印 tflite 输入输出签名 + 现场合成一段来打分
python3 wake_ab2.py           # TTS 合成音 vs 真人录音，同一 detector 批量打分
```

**必须知道的一件事：`robot.init()` 到真正开始收音要 60 秒上下。**

2026-11-08 第一次真人实测就是这么白跑的：脚本一启动就让人喊，人喊完了
`wake_probe` 才刚开始读麦克风，40 秒窗口里 RMS 最高 1854、本底 1000 ——
从头到尾没人声，结论只能是"没人说话"。所以探针现在会**在开始收音前播三声
880 Hz 蜂鸣**（`voice/cue_start.pcm`，16k/mono/s16le 裸 PCM），响完再
`drain()` 一次把余响丢掉：

```
python3 wake_probe.py 100 kws cue      # 默认就带，听到蜂鸣再喊
python3 wake_probe.py 100 kws nocue    # 不播提示音（别人在旁边听录音时用）
```

两个配套细节：

* 蜂鸣前把音量设到 60，否则系统音量被调低就听不见发令声。
* 探针捕获 `KeyboardInterrupt`：被 `timeout` 掐掉也会照常存下 wav、打印结论
  （第一次就是被掐断在 `queue.get()` 里，音频和 A~E 结论全丢了）。
* 每 0.5 秒那行末尾出现 `<< 有声音` 才说明人声真的进来了；没有这个标记
  就别去怀疑模型，先把人喊到麦前。

`wake_health.py` 的 `[0]` 段会打出三个模型的 `IN/OUT shape`，用来确认
`xingbao.tflite` 到底是不是 openWakeWord 三段式（前处理接 melspec+embedding）。
如果它对现场合成音的分数也是 0.001，就是模型/前处理不匹配，**跟麦克风无关**。

**应急兜底：ASR 唤醒（默认关闭）**

唤醒**只信模型**（`--wake-fallback off` 是默认值）。只有在模型全挂、又要马上演示时
才开 `asr`：

```
声音够响（本底 x1.25 且峰值得够）-> 断句 -> 送 DashScope ASR -> 文本里找"星宝"
```

* 开关：`--wake-fallback asr`；`--wake-asr-cooldown 3` 限流（秒）。
* 命中词表和文本唤醒共用 `WAKE_TOKENS`（含"星爆""星抱""兴宝"等识别同音字）。
* 为什么不默认开：每个语音段一次识别，费钱，还会把现场闲聊卷进来。

### 阈值：唤醒 0.25 / VAD 1.25（2026-11-08 换 KWS 后）

两个阈值是**两回事**，别混：

| 参数 | 默认 | 管什么 | 症状 | 怎么调 |
|---|---|---|---|---|
| 唤醒阈值 `kws.DEFAULT_THRESHOLD` | **0.25** | sherpa KWS 的关键词置信度 | 喊两三遍才中 | `--wake-th 0.15` 更灵、`0.35` 更稳 |
| VAD 倍数 `mic.VAD_FLOOR_MULT` | **1.25** | 本底噪声 × 这个值 = 开口阈值 | 唤醒了但"没听清"（后半句没录到） | `--vad-mult 1.6` 更稳、`1.1` 更灵 |

唤醒阈值 0.25 是模型自带示例的默认值。实测在这个值上，"星宝星宝"稳定命中，
7 段无关真人音频零误唤 —— 比旧方案那个从 TTS 上标出来的 0.30 可信得多。

VAD 下限同时从 250 收到 **150**。现场日志证据：`[vad] 本底 1683 x1.5 -> 阈值 2525`，
而人正常说话 RMS 常在 1500~2500 —— 刚好卡在阈值下面，表现就是"喊了它像没听见"。
现在同样本底算出来是 2104。

启动横幅会把后端、阈值、兜底状态都印出来，现场一眼能对：

```
  唤醒：sherpa-onnx 关键词模型，阈值 默认 0.25，兜底 关（只信模型）
  VAD ：底噪 x1.25（下限 150）
```

**注意**：如果怎么喊都不中，别再死调阈值了。先按上一节的排查顺序量
"麦克风有没有数据 / 模型有没有出分"。阈值只能调"够不够灵敏"，
调不了"模型没在听"。

### 日志和语音彻底分家（2026-11-08 现场要求）

用户原话："不要把日志推给星宝的语音系统。"

* **`xingbao.py` 的日志只写文件**（`--log`，默认 `/tmp/xingbao.log`），用
  `_FileLog`，**不接管 `sys.stdout`**。上一版用 Tee 双写，等于把语音链路上所有
  `print`（`♪ 播报 tts`、`[vad] 本底…`、`[chat] 联网搜索`）都卷进日志文件。
  现在日志里只有 4 类关键事件：启动参数、唤醒+附带内容、意图、任务结论。
* **`Assistant.speak()` 加了日志闸门** `looks_like_log()`：任何 `[vad]`/`[chat]`/
  `[wake]`/`Traceback`/`stream_id=`/`♪` 样式，或者"纯 ASCII 且带数字等号"的字符串，
  到 `speak()` 就被拦下——只打终端、不合成、不播报。语音只服务现场观众。
* 终端照旧打印全部细节（语音模块自己打自己的），排查看终端，演示听喇叭，两条路互不干扰。


## 语音问答「星宝星宝」（2026-11-08 新增；唤醒现为 sherpa-onnx KWS）

链路：麦克风（80 ms 一块）-> **本地 KWS 模型判唤醒** -> VAD 断句录问题
-> DashScope ASR -> 回答 -> DashScope TTS -> 16k PCM -> 喇叭。

```bash
# 先在 PC 上验 key（不连机器人）
python3 ask.py --check
python3 ask.py --say "你好，我是星宝" --out hello.wav
python3 ask.py --asr heard.wav
python3 ask.py --chat "这个奶酪新鲜吗"

# 上机
python3 ask.py --voice                  # 待命，本地模型听到"星宝星宝"唤醒
python3 ask.py --voice --no-wake        # 不用唤醒词，听到就当问题
python3 ask.py --voice --rounds 3       # 答 3 轮退出
python3 ask.py --wake-watch 30          # 只监听不回答：实时打分数，现场调阈值用
python3 ask.py --wake-test a.wav b.wav  # 拿现成 wav 测模型（不连机器人）
python3 ask.py --voice --wake-engine text   # 退回老版"识别文本里找星宝"
```

### 唤醒：sherpa-onnx KWS（`voice/kws.py`，当前默认）

模型在 `voice/model_kws/`，int8 共 4.8 MB：

| 文件 | 大小 | 作用 |
| --- | --- | --- |
| `encoder-...int8.onnx` | 4 777 666 B | zipformer 编码器 |
| `decoder-...int8.onnx` | 181 069 B | 解码器 |
| `joiner-...int8.onnx` | 65 242 B | joiner |
| `tokens.txt` | 1 627 B | 词表 |
| `keywords.txt` | 63 B | 唤醒词（拼音 + `@中文`） |

机器人上装一次：`python3 -m pip install --user sherpa-onnx`

改唤醒词只改 `keywords.txt`，格式是"声母/韵母分开 + 带调"，例如
`x īng b ǎo x īng b ǎo @星宝星宝`。改完直接重启，不用重训、不用换模型。

`keywords.txt` 里同时放了「星宝」这个短词，方便只喊两个字；如果现场误唤多，
把那一行删掉，只留「星宝星宝」。

### 旧唤醒：openWakeWord 三段式（`voice/wake.py`，保留做对比）

模型来自 `github.com/wzx2007/xingbao` 的
`edge_computing/xingbao_companion_project/model_voice/`，**是 git-lfs 文件**
（仓库里显示 131 字节是指针），真权重用 LFS batch API 拉：

| 文件 | 大小 | 作用 |
| --- | --- | --- |
| `xingbao.tflite` | 210 624 B | 唤醒分类器，输入 `(1,16,96)` |
| `openwakeword/melspectrogram.tflite` | 1 092 516 B | 前处理：log-mel |
| `openwakeword/embedding_model.tflite` | 1 330 312 B | 前处理：speech embedding |

**这条路 2026-11-08 在现场彻底失效了**，原因见上面「唤醒：根因是 TFLite 前处理
不兼容」：mel 模型在新版 TFLite 上 allocate 不了，导致输出恒 0.001。保留代码只为
`--wake-engine model` 对比用，**不要**再用它当主路径。

它自己踩过的坑（对理解旧日志有用）：

1. **模型内部 1 s 音频缓冲是未初始化的**。新建实例直接喂音频，同一段"星宝星宝"
   三次跑出 0.33 / 0.41 / 0.57；先喂 13 帧静音预热后稳定。
2. **不能直接 `import openwakeword`**（会连带导入 sklearn -> scipy，和 numpy 1.24
   冲突）。`wake.py` 用桩模块只加载 `openwakeword.model`。
3. **命中判定要连续两帧**（`patience=2`），单帧会抖。
4. **别指望用机器人自己的喇叭验证唤醒**：麦克风里带 AEC，播放的声音被消掉了
   （录下来送 ASR 只返回"嗯。"）。验证必须真人开口，用 `--wake-watch` 看分数。

### 文本唤醒（降级路径）

模型没装成 / 模型文件丢了 / `--wake-engine text` 时走老路：VAD 断句 -> ASR ->
在识别文本里找"星宝"。零额外依赖，唤醒词随时可改，代价是每段候选语音一次 ASR。

**Key 放在 `voice/.env`**（照 `voice/.env.example` 抄一份）：

```bash
DASHSCOPE_API_KEY=sk-xxxxxxxx
```

或者环境变量 `DASHSCOPE_API_KEY`。优先级：`--api-key` > 环境变量 >
`voice/xingbao.json` > `voice/.env`。**别把 key 写进代码。**

| 模块 | 作用 |
| --- | --- |
| `voice/cloud.py` | DashScope TTS / ASR / Chat，纯标准库 `urllib`（机器人是 Python 3.8） |
| `voice/mic.py` | `start_microphone_stream_input` 收流 + RMS 断句 VAD |
| `voice/wake.py` | 本地 `xingbao.tflite` 唤醒（openWakeWord 三段式流水线） |
| `voice/assistant.py` | 唤醒调度、问答循环、本地兜底回答 |
| `ask.py` | 命令行入口 |
| `voice/model_voice/` | 唤醒模型三个文件（git-lfs 真权重） |

几个实现上的坑：

- **TTS 返回的是音频 URL**，不是音频本身，要再下载一次（`output.audio.url`）。
- **不是每个账号都能用 `qwen-tts`**。2026-11-08 实测这把 key 上 `qwen-tts` 直接
  404 `Model not exist`，`cosyvoice-v3-flash` / `cosyvoice-v2` 正常。所以 `tts()`
  走**候选链**，失败自动换下一个。用 `python3 ask.py --probe` 一次看清这把 key 能用哪个。
- **ASR 走 compatible-mode 的 chat 接口**，音频要包成
  `data:audio/wav;base64,…` 的 Data URI 塞进 `input_audio`。
- **TTS 输出一般是 24 kHz**，SDK 只吃 16 kHz。转换统一复用 `play_clip.ensure_pcm`
  （ffmpeg 优先，退 miniaudio），不要另写重采样。
- **回调里不能干重活**。SDK 的麦克风回调在它自己的线程上，只入队，识别/合成放主线程。
- **VAD 阈值（文本路径用）必须自适应**。现场底噪实测 RMS ≈ **700~1020**（风机+人声+散热），
  固定阈值 120 会让它"一直以为有人在说话"。现在先估 480 ms 本底，阈值 =
  `max(250, 本底×1.5)`。
- **模型唤醒后录问题不要再用自适应校准**。`record_utterance` 自己的 6 块（480 ms）
  校准会丢开头 240 ms，正好切在"星宝星宝，|这个奶酪"的接缝上。所以
  `_tail_threshold()` 用**唤醒之前**的底噪算好阈值传进去，跳过校准，一个字节不丢。
- **回答要硬压短**。`max_tokens` + 本地清洗（`Brain._speakable`）双保险：去掉
  Markdown / emoji / 分点，只留第一句、不超过 80 字。不压的话云端会甩几百字。
- **文本唤醒时唤醒词必须几乎打头**（前 3 个字内且起始位置 <= 1），否则
  "这个奶酪上有星宝吗"会被误判成唤醒。同音字"欣宝/新宝/心宝/小宝"也算。


## 不动机器人，只看相机

```bash
python3 snapshot_vision.py                    # 连拍 6 帧，打印斑块表，存图到 <仓库根>/runs
python3 snapshot_vision.py --frames 12 --interval 0.3
```

上机第一步跑这个，确认"取得到图 + 分得出奶酪"，再跑整场演示。

## 其他

```bash
python3 vision/selfcheck.py --video /path/grab.mp4   # 拿历史录像复现标定数字
```

一次运行一个脚本。失败或中途停止不自动回位。
