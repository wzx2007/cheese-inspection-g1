# CheeseBot —— 帕马森奶酪检测机器人（银河通用 G1）

银河通用 G1 双臂环抱帕马森奶酪：**抓取 → 双面查看 → 按霉菌数量分流 → 归位**，
全程由语音助手「星宝」驱动（唤醒词 **星宝星宝**）。

动作是固定顺序的函数调用，没有神经网络；识别是传统 OpenCV。

```
叫醒 → "去检查一下这块奶酪"
        ↓
导航到桌边 → 看正面 → 环抱抬起 → 翻腕 → 看反面 → 翻回来
        ↓
0 个霉 → 放回原地        1 个霉 → 抱去 route_a      2 个霉 → 抱去 route_b
        ↓
播报 pass / fail → 回桌 / 带回起点 → 收身高
```

说「检测结果怎么样」能问上一块的结论；说「笑一个」放一段奶龙的笑。

---

## 1. 一分钟上手

在机器人上（`/home/galbot/scripts`）：

```bash
export PYTHONPATH=/data/galbot/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/data/galbot/lib:$LD_LIBRARY_PATH

python3 xingbao.py --confirm-motion         # ★ 正式演示：语音驱动（推荐）
python3 xingbao.py --talk-only              # 只聊天不动：不连运动、随便试
python3 main.py --confirm-motion            # 不用语音，直接跑一遍抓放
python3 main.py --confirm-motion --debug    # 每步回车执行，q 停止
python3 snapshot_vision.py                  # 不动机器人，只拍照看有没有霉（上机第一步）
```

清场 / 后台重启：

```bash
bash stop_voice.sh                          # 杀掉所有占麦克风的语音脚本
sh start_xb.sh motion                       # 清场 + 后台启动星宝（talk / motion）
```

`--confirm-motion` 是硬门槛：确认急停已松开、`WORKING_MODE`、周围净空、有人能拍急停。
失败或中途停止**不自动回位**。Ctrl+C 不是急停。

---

## 2. 目录

```
scripts/
├── xingbao.py              ★ 主入口：星宝语音核心（唤醒 → 意图 → 动手机器人）
├── main.py                 不用语音时的入口（转发到 demo_cheese）
├── demo_cheese.py          动作流程 + 现场参数（身高 / 半间距 / 末端关节 / 圈数）
├── snapshot_vision.py      静态查看：不动关节，只拍照判霉菌
├── stop_voice.sh           清场：杀掉所有占麦克风的语音脚本
├── start_xb.sh             清场 + 后台启动星宝
│
├── armctrl/                RightArmController（高举 / Ready / 半间距 / 末端关节）
│   └── sdk_arm_planner.py  真机适配，一次 traj_plan 同时下发左右臂
├── legctrl/                LegController（只升降，不俯身）
├── navctrl/                ChassisNavigator（全局导航为主，直线补位只兜底）
├── nav_goals/              table / route_a / route_b 三个目标点
├── config/                 手臂预设（高举 / Ready）
│
├── mission/                ★ 分流状态机
│   ├── route.py            0/1/2 个霉 → 原地 / route_a / route_b
│   └── report.py           结论组织与播报文案
│
├── vision/                 ★ 霉菌识别
│   ├── config.py           相机 + 全部阈值（现场调参只改这里）
│   ├── mold.py             分割 + Black-Hat + 斑块过滤 + 判定
│   ├── inspector.py        一次"查看"：连拍 → 判定 → 播报本次结果 → 存档
│   └── selfcheck.py        迁移验收：拿历史录像复现标定数字
│
└── voice/
    ├── assistant.py        星宝的"人"：唤醒 → 听 → 回答 / 执行
    ├── intent.py           规则层意图（停 > 笑 > 开测 > 状态 > 闲聊）
    ├── intent_llm.py       大模型判定（qwen 输出 JSON：detect / status / chat）
    ├── cloud.py            DashScope TTS / ASR / 对话
    ├── kws.py              sherpa-onnx 本地唤醒词兜底
    ├── wake_asr.py         VAD 断句 + ASR 文本判唤醒（主路径）
    ├── mic.py              麦克风采集 + VAD
    ├── memory.py           对话与检测记录
    ├── play_clip.py        VoiceBox（队列 + 后台线程，16 kHz/单声道 PCM 流式写入）
    ├── model_kws/          训练好的唤醒词模型（onnx + xingbao.tflite）
    ├── result_pass.wav     停用：旧的"未发现异常"（不分正反面）
    ├── result_fail.wav     停用：旧的"发现疑似异常"（不分正反面）
    ├── front_ok.wav        正面正常
    ├── front_bad.wav       正面不正常
    ├── back_ok.wav         反面正常
    ├── back_bad.wav        反面不正常
    ├── check_done.wav      三轮跑完："检查完成"
    ├── goodbye.wav         然后道别
    ├── laugh.wav           彩蛋：奶龙大笑（19.8 s）
    └── *_16k_mono.pcm      预转 PCM，省掉机器人上的 ffmpeg
```

根目录另有 `docs/工程笔记.md`、`docs/G1-API调用.md`（队友整理的 SDK 调用手册）、
`nav_then_grab.py`（两段式导航的最初原型，逻辑已并入 `navctrl/chassis_navigator.py`）。

> `voice/.env` 存 DashScope key（照 `.env.example` 自己填），**不入库**；
> `voice/memory.json` 是对话记忆，同样不入库，首次运行会自动创建。

---

## 3. 三次抓取时序

`LOOP_COUNT = 3`。**每一圈都是一次完整抓放**，阶段名和 `galbot_nav` 原管线一致，
只在两处插了"查看"：

```text
gripper_open -> arm_reset -> height_ready -> [look_front]
-> arm_ready -> reach_forward -> grasp_close -> lift -> flip
-> [look_back] -> unflip -> lower -> release -> arm_reset2
```

| # | 阶段 | 动作 | 说明 |
| --- | --- | --- | --- |
| 1 | — | `nav.navigate_to("table")` | 全局导航为主、直线补位兜底；误差还大就先用新坐标自愈重规划，仍不行才报错不抓（见 7.1） |
| 2 | — | `switch_controller(LEG_PVT_CTRL)` → `reset_pitch()` | 导航完切回来才能升降；顺手回正俯仰 |
| 3 | `gripper_open` | 夹爪全开 0.12 m | 每圈都开，因为每圈都要重新抓 |
| 4 | `arm_reset` | `reset_both_arm()` | 双手回高举 |
| 5 | `height_ready` | `control_height(1.37)` | |
| 6 | `look_front` | **看正面** | 抓取**之前**，奶酪还在桌上、双臂高举，相机俯视它的平面 |
| 7 | `arm_ready` | `set_both_arm_ready()` | 只能从 `initial` 进入 |
| 8 | `reach_forward` | `shift_forward(0.05)` | |
| 9 | `grasp_close` | `set_half_gap(0.28)` | 环抱夹取 |
| 10 | `lift` | `control_height(1.54)` | 奶酪离桌 |
| 11 | `flip` | `set_end_joint(-1.538)` → 等停稳 | |
| 12 | `look_back` | **看反面** | `flip` 之后，**仍在 1.54 m** |
| 13 | `unflip` | `set_end_joint(+1.538)` | |
| 14 | `lower` | `control_height(1.37)` | 仍夹着 |
| 15 | `release` | `set_half_gap(0.40)` | 在抓取高度松手 = 轻放 |
| 16 | `arm_reset2` | `reset_both_arm()` | 空手回高举 |
| 17 | — | `nav.navigate_to_pose(origin)` | 回起点，跑下一圈 |

三圈跑完再 `control_height(1.32)`。

> **2026-11-07 返修**：之前把 `lower/release/arm_reset2` 挪到了三圈之外，
> 机器人会抱着 50 cm 奶酪往返起点和桌边，等于只抓了一次、看了六面。
> 现在每圈都放回桌上；这是原管线的阶段顺序。

---

## 3.1 两个"查看"为什么不能随便挪

两次查看都**不额外升降高度**，直接落在标定过的位姿上：

| 查看 | 时机 | 身高 | 对应原管线窗口 |
| --- | --- | --- | --- |
| `look_front` | 抓取之前，奶酪在桌上、双臂高举 | 1.37 m | 0–5 帧 |
| `look_back` | `flip` 之后，仍夹着 | 1.54 m | 200–205 帧 |

2026-11-07 现场：上一版把 `look_back` 放在 1.34 m（抬起后先降了 20 cm）。
结果霉菌所在的平面被压成一条约 30 px 的窄带，又被顶灯高光冲白，
同样的暗条 Black-Hat 深度从 **192 掉到 65**，低于 `SPOT_MIN_DEPTH=100`，直接判不出来；
帧里唯一超过 100 的暗块是奶酪下沿的阴影（depth 220，但圆度 0.34 / 离轮廓 5.4 px），
被几何规则正确拒掉 —— 于是输出"没有霉菌"。

**别为了"看清楚"去降高度，那样反而看不见。** 判不出来先比对 `runs/*/` 原图和
`vision/selfcheck.py` 的窗口数字，再考虑动阈值。

---

## 4. 视觉：霉菌识别

### 4.1 接在哪

第 6 步（抓取前的正面）和第 12 步（翻腕后的反面）各连拍 6 帧
（先停 0.45 s 等曝光稳定，再每 0.18 s 取 1 帧）。
一圈两面，三圈共 6 次查看、36 帧。**每一面判完就立刻播报这一面的结果**：
有霉 → `voice/result_pass.wav`（PASS），没霉 → `voice/result_fail.wav`（FAIL）。
三次跑完、收身之后再放 `voice/goodbye.wav`。

取图用 `get_rgb_data(HEAD_LEFT_CAMERA, BGR, once=True)`，**不订阅视频流**：
订阅要起回调线程，收尾时容易段错误；识别只要几张静止图，逐帧取最新图更干净。

相机必须在 `robot.init({SensorType.HEAD_LEFT_CAMERA})` 里显式打开 ——
默认传感器集合里没有相机，`init()` 空参调用取到的是空 dict（已实测）。

### 4.2 判据

只有一类斑：**菌**。两条门槛：

1. **Black-Hat 深度** —— `BH = close(V, 41×41) - V`，即"比周围暗多少"。
   `BH >= 45` 才算候选。为什么不用绝对亮度：深孔 V=8、灰霉 V=150~220、
   奶酪本体 V≈212，一个绝对阈值套不住三种情况。
2. **斑块内最深处 `>= SPOT_MIN_DEPTH=100`** —— 真斑实测 maxBH 164~228，
   奶酪表面暗纹理只有 72~74。这一条把"表面纹理"挡在外面。

几何过滤只剔除不像斑的东西（太小 / 太大 / 太细长 / 太贴轮廓），不参与"是不是霉"的语义判断：

| 参数 | 值 | 作用 |
| --- | --- | --- |
| `BH_KSIZE` | 41 | 结构元，要大于最大斑块（实测孔洞 42×36） |
| `BH_THR` | 45 | 候选低阈值（比周围暗 18%） |
| `SPOT_MIN_DEPTH` | 100 | 真斑 / 表面暗纹理的分水岭 |
| `SPOT_MAX_RATIO` | 2.5% | 超过就是整块阴影，判分割失败 |
| `SPOT_MAX_AR` | 5.5 | 菌斑常是细长弧（实测 35×7） |
| `SPOT_MIN_DIST` | 6 px | 真斑离轮廓远；轮廓假阳性只有 3~4 px |

奶酪分割：HSV 黄色（H 14–42、S≥90、V≥80）→ 最大连通域 → **填洞**（不用凸包，
避免把背景桌布包进来）→ 按包围盒 4.5% 腐蚀边缘（避开轮廓明暗过渡带）。

一次查看的结论按**过半帧命中**（`HIT_FRAME_RATIO=0.5`）算，防单帧抖动。
分不出奶酪时输出"看不到奶酪"，既不说有也不说没有 —— 不把"没对准"说成"干净"。

### 4.3 输出与证据

终端每次查看打一张小表：

```
  ── 查看：第1次 正面（未翻腕） ──
     帧  奶酪  菌斑  最大占比    中心
     00   ✅    4     1.6784%   (712, 899)
     ...
     结论：有霉菌（3/3 帧命中，最大 1.6784%）
  ♪ 播报 front_bad（2.32 s）        ← 判完立刻播"奶酪状态，正面不正常"
```

全部结束后汇总：

```
  ── 全场判定 ──
     查看                   帧数  命中  最大占比    结论
     第1次 正面（未翻腕）   6     6     1.6784%     有霉菌
     第1次 反面（已翻腕）   6     0     0.0000%     没有霉菌
     ...
     总体：有霉菌
     证据：/home/galbot/runs/20261107_210257
```

证据目录里是按 `loop1_faceA_00.jpg` 命名的原图（命中的帧另存 `_norm.jpg` 带标注框）
和一份 `verdict.json`。留原图是为了以后调阈值能离线复算。

### 4.4 迁移验收

`vision/` 的检测是从 `galbot_nav/cheese_pipeline/vision/mold.py` 原样搬过来的，
一行阈值都没改。用两段历史录像复现当时的数字（`python3 vision/selfcheck.py --video ...`）：

| 录像 | 正面 0–5 帧 | 背面 200–205 帧 | 结论 |
| --- | --- | --- | --- |
| `03_grab_indexed.mp4` | 6/6 命中、24 斑、最大 1.6791% | 0/6 | 有霉菌 |
| `04_grab_run2.mp4` | 0/6 | 6/6 命中、7 斑、最大 0.1611% | 有霉菌 |

和重构前**逐位一致**。

---

## 5. 声音

识别结果的播报按 **「哪一面 × 正常/不正常」四选一**（2026-11-08 改，对齐演示流程文档里
`demo_flow.py` 的台词）。此前只有 pass/fail 两支、台词是"检测完成，表面未发现明显异常"，
根本不提正反面 —— 一场看 6 个面，观众听不出说的是哪一面，现场反馈后换掉。

| 文件 | 什么时候播 | 台词（ASR 回读核对） | 时长 |
| --- | --- | --- | --- |
| `front_ok.wav` | 正面 + 没霉 | 奶酪状态，正面正常。 | 2.24 s |
| `front_bad.wav` | 正面 + 有霉 | 奶酪状态，正面不正常。 | 2.32 s |
| `back_ok.wav` | 反面 + 没霉 | 奶酪状态，反面正常。 | 2.24 s |
| `back_bad.wav` | 反面 + 有霉 | 奶酪状态，反面不正常。 | 2.24 s |
| `check_done.wav` | 三轮跑完 | 检查完成。 | 1.04 s |
| `goodbye.wav` | 紧接着道别 | 今天的演示到这里，谢谢大家。 | 3.40 s |
| `laugh.wav` | 彩蛋：随口让它笑一下 | 奶龙大笑（不是人声台词） | 19.78 s |

- **口径：正常 = 没霉，不正常 = 有霉。** 没分出奶酪/没取到图时**没有结论**，
  按"没看到 ≠ 干净"播**不正常**，同时终端打一行警告（`inspector.announce`）。
- 四支是同一把云端音色（`cosyvoice-v3-flash` / `longxiaochun_v3`）**离线预生成**的，
  不在现场跑 TTS —— 现场合成要 1~2 s 且依赖网络，反而拖慢动作。生成脚本的台词就是
  `demo_flow.py` 里 `speak("奶酪状态，" + ...)` 那一行，改词要两边一起改。
- 生完立刻用 DashScope ASR 回读、并且**打乱文件名再读一遍**核对（防止张冠李戴），
  四支 PCM 的 md5 也各不相同 —— 这套核对别省。
- 旧的 `result_pass.wav` / `result_fail.wav` 已停用（文件保留在 `voice/` 里，方便回退）。
- 播报用 `VoiceBox`（`voice/play_clip.py`）：队列 + 后台线程，`say()` 入队即返回，
  说话的同时手臂继续走动作。两次查看之间隔十几秒动作，队列不会堆积。
- 收尾两句都进同一条 FIFO：`say("check_done")` 再 `say("goodbye", wait=True)` ——
  只有最后一句阻塞，等它播完才返回，否则随后的 `request_shutdown()` 会把尾音截掉。
- 四支结果音是 24 kHz（TTS 输出），`goodbye` 是 32 kHz（早先录的）。SDK 的
  `write_audio_stream_output` 只吃 **16 kHz / 16 bit / 单声道 PCM**，所以必须先转：
  `ensure_pcm()` 用 ffmpeg（没有则退 miniaudio）转成 `<名字>_16k_mono.pcm` 并缓存。
- 音量 `set_volume(100)`，`VoiceBox` 初始化时设一遍（`vision/config.py: VOICE_VOLUME`）。
- 想只判不出声：`--no-voice`。想单独试听某支：
  `python3 voice/play_clip.py --file voice/front_ok.wav --volume 100`。
- `laugh.wav` 走**规则层**（`voice/intent.py: is_laugh()`），"停"字优先，命中就走，
  不等大模型往返——所以"笑一个""笑一下""给我笑"这类说法是 0 延迟的。两条音频入口
  （`--talk-only` 的 `_connect_audio` 和运动模式的 `demo_cheese.connect`）都会 `_preload_laugh()`。

---

## 6. 部署（PC → 机器人）

机器人 IP `<ROBOT_IP>`，`galbot` / `<ROBOT_PASSWORD>`，代码放 `/home/galbot/scripts`。
PC 侧用 `galbot_nav` 里的 `HPU.upload_tree()` 递归推送即可（只传 `.py/.md/.json`，
音频要单独推，见下）：

```bash
# PC 上，工作区根目录
python -c "import sys; sys.path.insert(0,'galbot_nav'); \
from cheese_pipeline.remote import HPU; \
h=HPU(); h.run('cp -a /home/galbot/scripts /home/galbot/scripts_bak_$(date +%m%d_%H%M)'); \
print(len(h.upload_tree(r'CheeseBot/scripts','/home/galbot/scripts'))); h.close()"
```

推完清一次缓存：`rm -rf /home/galbot/scripts/__pycache__ /home/galbot/scripts/vision/__pycache__`。

---

## 7. 参数速查

`demo_cheese.py` 开头：

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `LOOP_COUNT` | 3 | 抓取圈数（每圈一次完整抓放） |
| `READY_HEIGHT_M` | 1.37 | 抓取身高，也是 `look_front` 的身高 |
| `HEIGHT_HIGH_M` | 1.54 | 抬起身高，也是 `look_back` 的身高 |
| `HEIGHT_END_M` | 1.32 | 收尾身高 |
| `HUG_GAP_M` | 0.28 | 夹取半间距 |
| `RELEASE_GAP_M` | 0.40 | 撒手半间距 |
| `END_JOINT_RAD` | 1.538 | 末端关节极限（1.54 会被判越界） |
| `FORWARD_X_M` | 0.05 | Ready 后额外前伸 |
| `USE_VISION` | True | 关掉等于 `--no-vision` |

> `VERTICAL_M` / `VIEW_HEIGHT_M` 已删除：查看不再额外升降，见 3.1。

`vision/config.py`：相机名、连拍节奏（`LOOK_FRAMES/INTERVAL/SETTLE`）、
分割阈值、暗斑阈值、白霉阈值、判定阈值、播报音量 `VOICE_VOLUME`。

命令行：`--debug` `--no-vision` `--no-voice` `--vision-frames N` `--volume 0-100` `--runs-dir` `--backoff M` `--max-lastmile M` `--no-route` `--count-mode` `--tool`。

导航的其余旋钮（`direct_first` / `stall_timeout_s` / `refuse_grace_s` / `skip_global_m` /
`arrive_tol_m` / `status_grace_s` / `pose_jump_m` / `relocalize_retries` /
`navigate_budget_s`）都在 `ChassisNavigator.__init__` 的默认值里，改之前先读 7.1——它们
每一条都是为了**不要**和规划器抢方向盘。

## 7.1 导航：直接把终点交给规划器

终点 `table` 就是**建图时机器人站的位置**，机器人自身的点被写进地图，所以终点格本身常被
预检判成"不可达"（实测 `check_path_reachability(终点)` 返回 False）。而且预检在桌前**一律
不准**：车贴着桌子，自己的位姿就在规划器 0.45 m 的 `point_safe_margin` 里，从这个位姿出发
它对**任何**终点都返回 False。

所以 2026-11-08 起，候选顺序是"终点永远第一档、预检只当提示"：

1. **终点直连**——`navigate_to_goal(终点)` 派一次活就等到达，和 `demo_flow.py` 一样。
   终点**连预检都不做**：既然结论一律不信、一律照试，提前查它只是白花 100 ms IPC；
2. **预停位由近到远**（沿终点朝向往回退 0.8 → 1.2 m，最多 6 档）——只有直连真被
   `COLLISION`/`CLOSE_TO_OBSTACLE` 顶回来才往下走，预检在这里只用来排序；
3. 兜底再试一次终点（走到这一步说明规划器中间抖过：临时障碍或位姿刚被拉回）。

**为什么改成这样**：原来把预检当闸门，于是每趟都先绕预停位；而预停位在桌前必然
`COLLISION` 或原地卡死 20 s，最后"直接导航到终点"才成——三圈演示每趟白烧 30~50 s 在一条
必败的绕路上，真正成功的那一步被排在队尾。现场日志里那句"第一段重试 1/1"是同一件事的副
作用：对同一个点重派，规划器拿同一张图、同一个起点，结论必然一样。现在这层重试整个删了
（`stage1_retries` 形参只为兼容老调用方保留，值不再被读取），卡死判定也从 20 s 压到 8 s
（现场卡住时位姿是 0.000 变化，2 s 一个采样，8 s 足够确诊且不再空站）。

### 认出"规划器根本给不出路"（`REFUSED`）

2026-11-08 真机探针测出来的事实：`navigate_to_goal` 是**同步**返回的，而且它拒绝时也
照样回 `(True, 'SUCCESS')`——`0.5 s` 后状态才变成 `FAILED`，车一动没动。原来我们既不看
返回值、又要等 6 s 宽容期和 8 s 卡死看门狗，**每换一档候选就白烧 8 s**，6 档 = 49.8 s。

现在把"拒了"和"走着走着抖一下"分开判，靠的是**车动没动**：

| 情况 | 判据 | 结果 |
| --- | --- | --- |
| 派活当场返回 `False` | 返回值 | 立刻 `REFUSED`，不等 |
| 车一动没动（离派活点 < 0.12 m）+ 状态一直非 RUNNING | 持续 `refuse_grace_s`（1.5 s） | `REFUSED`，换下一档 |
| 车动过之后再报错 | 持续 `status_grace_s`（6 s）+ 确实停住 | 才认输（这才是真失败） |

再加一条省时规则：**终点和第一个预停位都被拒，而剩下的路在 `max_lastmile_m`（2.5 m）以内
→ 不再换点，直接交给直线补位**。真机实测：6 档全被拒、白花 49.8 s，而 5.7 s 的补位把车
送到位（误差 0.037 m）——那 6 档纯属仪式。

到了终点之后，如果**全局导航已判定到达、但真位姿与终点还差一点点**（`pos_err ≤
arrive_tol_m 0.22 m`），才用 `move_straight_to` 按 odom 小步（0.2 m/步）把最后那截补进去。
超过 `max_lastmile_m`（2.5 m）**不硬补**——说明全局那段根本没走成，直接交回全局自愈重规划，
而不是靠里程计一路盲走。

### 谁说了算：别抢规划器的方向盘

2026-11-08 之前的写法是"我们盯着"，结果现场老是"第一次好、第二次挂"。三个具体的抢方向盘动作：

| 我们干过的事 | 为什么会坏 | 现在 |
| --- | --- | --- |
| 轮询 `get_navigation_status()`，一看到 `FAILED` 就认输 | SDK 自己会丢任务结果（`galbot_motion_planning.cpp` 里 `has unknow task result`，一个 session 刷 363 次），状态会**凭空**报 `FAILED`，车其实还在走 | 非 `RUNNING` 必须连续 `status_grace_s`（6 s）**且车确实没动**才认输 |
| 用"剩余距离"判死刑 | 重定位一发生，`odom` 相对关系变了，剩余距离瞬间跳到 6 m 以上，看着像"走不过去" | 只在确认到达/没动时才读剩余距离 |
| `_last_mile` 补不动就 `raise` | 一 raise 整趟任务崩，本来重规划一次就能救 | 打印一行 `return False`，交给全局自愈 |

另外 `pose()` 会做**位姿突变检测**：单次位移 > 1.5 m 或转角 > 60° 记为"定位跳变"，丢弃这段
里程计并让上层用新坐标**重规划同一目标点**（最多 `relocalize_retries` 2 次），播报
"我刚才定位飘了一下，正在重新对位置"。这套逻辑就是照 `demo_flow.py` 那版来的——那版之所以稳，
是因为它**只 `navigate_to(name)` 派一次活，然后老实等 `check_goal_arrival()`**，不看中途状态、
不算剩余距离、不自己补位。

| 构造参数 | 默认 | 含义 |
| --- | --- | --- |
| `direct_first` | True | 终点是否排候选第一档（关掉才回到"预停位优先"的旧顺序） |
| `skip_global_m` | 0.5 | 离终点小于这个距离就不再走全局导航 |
| `arrive_tol_m` | 0.22 | 判定"到了"的位置容差（自愈阈值同源） |
| `status_grace_s` | 6.0 | 车**动过**之后，非 RUNNING 要持续这么久才认输 |
| `refuse_grace_s` | 1.5 | 车**一动没动**时，非 RUNNING 持续这么久就判"规划器给不出路" |
| `stall_timeout_s` | 8.0 | 车 0.10 m / 5° 都没动满这么久就判"卡住"（原 20 s） |
| `pose_jump_m` / `pose_jump_deg` | 1.5 / 60 | 位姿突变的判定门槛 |
| `relocalize_retries` | 2 | 定位跳变后重规划同一目标的最多次数 |
| `navigate_budget_s` | 100 | 一趟导航换点的总预算（不是单次派活），超时才放弃 |
| `stage1_retries` | — | **已废弃**，形参保留只为兼容老调用方，值不再被读取 |

---

## 8. 故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `No module named 'galbot_sdk'`（返回码 3） | 在 PC / WSL 上跑了，或忘了两个 `export` | 必须在**机器人 Orin** 上跑：`ssh galbot@<ROBOT_IP>` → `cd /home/galbot/scripts` → 先 `export PYTHONPATH=/data/galbot/lib:$PYTHONPATH` 和 `export LD_LIBRARY_PATH=/data/galbot/lib:$LD_LIBRARY_PATH`。SDK 是 aarch64 原生库，PC 上装不了 |
| `看不到奶酪` | 没对准 / 被指尖挡住 / 身高没到位 | 看 `runs/*/` 里的原图；先跑 `snapshot_vision.py` 确认相机视野 |
| `取不到图像，本次关闭视觉` | 相机没在 `robot.init` 里打开 | `snapshot_vision.py` 的 `open_camera()` 会显式带 `HEAD_LEFT_CAMERA` |
| 一直判"没有霉菌" | ① 查看位姿偏了（尤其别降高度看反面）② 光照变了 ③ 奶酪换了一批 | 先看 `runs/*/` 原图里那面到底在不在画面、平不平面；再用 `vision/selfcheck.py` 在录像上复算；最后才动 `BH_THR`、`SPOT_MIN_DEPTH` |
| 判"有霉菌"但肉眼没有 | 表面阴影太重 | 提高 `SPOT_MIN_DEPTH`，或看 `_norm.jpg` 的标注位置 |
| 播报不响 | 音量被外部改过 / PCM 没推上去 | `VoiceBox` 初始化就 `set_volume(100)`；确认 `voice/*_16k_mono.pcm` 三个都在；单独试听 `python3 voice/play_clip.py --file voice/result_pass.wav` |
| 播报被截断 | 收尾没等播完就 shutdown | 只有 goodbye 走 `wait=True`；若某次结果声音没播完，看日志有没有"播报 xxx"后紧跟动作 |
| 播报顺序乱 | 队列堆积（两次查看间隔太短） | 正常间隔十几秒，不会堆；若改快了就在 `say` 后加 `wait_idle()` |
| `no feasible leg pose for ee_z=... m, pitch=0.00 deg` | 开机时头高超出可解区（实测停在 1.5627 m，比 `HEIGHT_MAX=1.54` 高 2 cm），这个高度下回正俯仰无解 | 已内置兜底：`level_leg()` 失败就先落到 1.37 m 再回正；日志会先打一行"当前头高 …" |
| 导航报"到位误差超限，正在自愈" | 全局那段成了，但真位姿离终点还差一点 | 看日志里的 `位置误差 / 朝向误差`；重录 `nav_goals/table.json`。自愈会用新坐标再重规划一次，最多两次 |
| 导航日志每趟第一句就是"直接导航到终点（一次派活，和 demo_flow 一样）" | 这是**新顺序**：终点永远第一档，预检不再当闸门 | 不用管。看到它之后直接到位就是正常；只有在它后面才可能出现预停位 |
| "直接导航到终点"被拒（`COLLISION` / `CLOSE_TO_OBSTACLE`），接着"预停位后退 0.8 m" | 终点在障碍膨胀区（建图起点就贴着奶酪桌），直连确实被规划器顶回来了 | 正常兜底：会自动退预停位（最多 6 档，0.8→1.2 m），再不行兜底重试终点；全被拒才需要把车往外挪半米，或 `--backoff 1.5` |
| 日志里状态突然 `FAILED`，但车还在走 | SDK 丢了任务结果（`has unknow task result`），状态是假的 | 不用管：车动过才走 6 s 宽容期，最终看真位姿 |
| 每档都打"状态 FAILED 且车一动没动，判规划器给不出路" | 规划器从当前位姿解不出可执行的路（多半是车自己贴在障碍膨胀区里，或定位飘了） | 前几档这样然后靠"直线补位"到位属正常；**每一档都这样且位置误差好几米**，说明车不在自己以为的地方——看 `当前位姿` 对不对，重新对位（重定位）再说 |
| "定位跳变 6.87 m，正在重规划" | 重定位/里程计跳变，剩余距离会瞬间变得离谱 | 不用管：会丢弃那段 odom、用新坐标重规划同一目标（最多 2 次），期间播报"定位飘了一下" |
| 车原地来回小幅摆动不收敛 | 预停位正好压在膨胀区边缘，规划器反复解不出可行腿 | 手动把车挪开 0.5 m 重新发起；或 `--backoff` 给一个更远的起点 |
| `switch_controller` 失败 | 没切回 `LEG_PVT_CTRL` | 导航后必须切回才能升降 |

---

## 9. 变更记录

- **2026-11-08** 导航"别再绕那 20 秒"：现场日志显示每趟都是先绕预停位、卡 20 s 失败、
  重试 1/1、最后"直接导航到终点"才成。根因是 `check_path_reachability` 在车自己处于
  膨胀区时对**所有**终点返 False，旧逻辑把它当闸门。现在 ① 终点永远候选第一档、连预检
  都不做（`direct_first=True`）；② 删掉 `stage1_retries`（"第一段重试 1/1"），只留
  "卡住才原地重派一次"；③ `stall_timeout_s` 20 s → 8 s；④ 预停位扫描收窄到 0.8/1.0/1.2 m
  且默认不做横移。回归 `test_nav_fast.py` 8 组、`test_nav_field_jump.py` 9 组、
  `test_nav_second_run.py` 全过。
- **2026-11-08** 导航"认出规划器给不出路"（真机探针测出来的）：`navigate_to_goal` 拒绝时
  也回 `(True,'SUCCESS')`，0.5 s 后状态才 `FAILED`、车一动不动。于是 ⑤ 新增 `REFUSED`：
  派活返回 `False` 或"车一动没动 + 状态持续非 RUNNING 1.5 s"就立刻换档，不再等 8 s 看门狗；
  ⑥ 终点和第一个预停位都被拒、且剩余 ≤ `max_lastmile_m` 时不再换点，直接交给直线补位。
  真机两次实测：同样"全被拒"的环境下，一趟从 58.9 s 降到 ~15 s 量级。回归加到 10 组。
- **2026-11-08** 结果播报改成**按面四选一**：原来的 `result_pass/fail` 两支台词是
  "检测完成，表面未发现明显异常"，不提正反面，6 个面连续播报时听不出说的哪一面。
  换成 `front_ok/front_bad/back_ok/back_bad` 四支（台词对齐 `demo_flow.py`：
  "奶酪状态，正面正常/不正常""奶酪状态，反面正常/不正常"），云端 TTS 离线预生成 + ASR
  打乱文件名回读核对；收尾补一句 `check_done.wav`（"检查完成"）再接道别。
  `inspector.announce` 改为 `_is_back()` + `healthy` 二维选片，`xingbao_sim` 同步。
  新增回归 `_tmp/test_face_clips.py`（4 组合 + 2 种"没看清" + 文件/加载表）。
- **2026-11-08** 导航照 `demo_flow.py` 重写"谁说了算"：`ChassisNavigator` 不再是"盯着状态
  轮询的执行器"，而是"派一次活就等的客户端"。① 非 `RUNNING` 要连续 6 s 且车确实没动才认输
  （SDK 会丢任务结果凭空报 `FAILED`）；② 候选点改惰性产出（终点直连 → 预停位由近到远 →
  兜底再试终点），第一档成就不查后续；③ `pose()` 加位姿突变检测（>1.5 m / >60°），跳变后
  丢弃那段 odom、用新坐标重规划同一目标（最多 2 次）；④ `_last_mile` 不再 `raise`，超
  `max_lastmile_m` 就交回全局。现场失败回放 `test_nav_field_jump.py` 7 组全过。
- **2026-11-08** 音量统一到 100：`voice/assistant.DEFAULT_VOLUME` 改为引用
  `vision.config.VOICE_VOLUME`，两处（对讲 / 运动）不再各写一个数。
- **2026-11-08** 加奶龙大笑彩蛋 `voice/laugh.wav` + `intent.is_laugh()` 规则层（"停"优先）。
- **2026-11-08** 换音频 + 改播报时机：删 `voice/laugh.mp3`，换成
  `result_pass.wav`(有霉/PASS) / `result_fail.wav`(没霉/FAIL) / `goodbye.wav`(跑完道别)。
  **每判完一面立刻播报这一面的结果**（原来只在首次看到霉时放一次笑声）；
  三次跑完、收身之后道别。`play_laugh.py` → `play_clip.py`，`ClipPlayer`(线程重生)
  → `VoiceBox`(FIFO 队列 + `wait=True`)。新增 `--no-voice`，删掉 `--laugh-each`
  和 `LAUGH_ONCE`。
- **2026-11-08** 反面霉菌判据推翻重做：反面的霉**不是黑的**（整幅近纯黑像素为 0），
  是灰白覆盖膜。删掉只会命中顶面棱线的 `BH_b` 黄度线索，新增 `cue="W"`
  白霉线索（斑内中位 `S∈[62,105]`）。12 个已标注面回归 0 错。
- **2026-11-07** 修执行顺序：`lower/release/arm_reset2` 放回每一圈（之前在三圈之外，
  等于只抓一次、抱着奶酪往返）；`look_front` 移到抓取之前（对原管线 0–5 帧），
  `look_back` 不再降高度、就留在 1.54 m（对原管线 200–205 帧）。删掉 `VERTICAL_M`。
- **2026-11-07** `level_leg()` 兜底：回正俯仰无解时先落到 1.37 m 再回正（机器人曾停在
  1.5627 m，比 `HEIGHT_MAX` 高 2 cm，这时 pitch=0 的腿部逆解无解）。
- **2026-11-07** 修 `open_camera()` 签名 bug（多了一个 `robot` 形参，导致 `robot.init(相机集合)`
  抛异常后回退成无相机，视觉静默失效）。
- **2026-11-07** 预停位自动退档：第一段回 `CLOSE_TO_OBSTACLE`/`OCCUPIED` 时换更远的候选点
  （最多 6 个），并把 `--backoff`、`--max-lastmile` 开放到命令行。
- **2026-11-07** 加 `preflight()`：SDK/视觉依赖缺失时给明确指引，返回码 3。
- **2026-11-07** 接入视觉：`scripts/vision/`（迁移自 `galbot_nav/cheese_pipeline/vision`），
  在第 6/8 步各看一面，有菌放笑声。
- **2026-11-07** 去掉原有音频：删 `voice/拉玛刹.mp3`、`拉玛刹_16k_mono.pcm`，
  取消导航时的 `on_move` 播放。
- **2026-11-07** 主程序化：`main.py` 统一入口；`LOOP_COUNT=3` 三次抓取。
