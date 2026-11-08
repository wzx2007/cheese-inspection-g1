#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""星宝：语音作为唯一接口的主控。

和 ``main.py`` / ``demo_cheese.py`` 的区别
----------------------------------------
那两个是"按键驱动"的演示脚本：跑三圈，每圈停下来等人按回车。

这个是"**语音驱动**"的：星宝醒着待命，人跟它说话，它自己决定要不要干活。

::

    待命（本地模型听"星宝星宝"）
      -> 唤醒
      -> 听一句话，判意图
           ├─ 开始检测  -> 跑完整任务（导航 → 看正面 → 抓 → 翻 → 看反面 → 放回 → 收身）
           │                每块测完立刻汇报"两面都有霉 / 一面有霉 / 合格"
           ├─ 问结果    -> 用记忆回答"刚才那块怎么样"
           ├─ 叫停      -> 在检查点停下（不是物理急停！）
           └─ 其它      -> 正常对话（带上下文记忆）

第二轮唤醒交给星宝
------------------
原来是"抓完一次 -> 人推车 -> 按回车继续"。现在那声回车换成**对星宝说话**：
它问"请把我推到下一个起点，推好了跟我说一声"，人说"好了"就继续。
这既是第二圈的启动，也是一次确认（人不说话它就不动）。

安全边界（很重要）
------------------
* 语音**不能**代替急停。急停按钮永远优先，语音"停"只在检查点生效。
* 每个任务都要 ``--confirm-motion``，和原来一致。
* 机器人移动期间不监听语音（避免环境噪声被当成指令），只在检查点开口。
* 导航失败/看不清这类问题，星宝会说出来并停住，不会自己瞎试。

跑法::

    export PYTHONPATH=/data/galbot/lib:$PYTHONPATH
    export LD_LIBRARY_PATH=/data/galbot/lib:$LD_LIBRARY_PATH

    python3 xingbao.py --confirm-motion              # 语音主控（默认一块）
    python3 xingbao.py --confirm-motion --route      # 测完按霉数分流到终点
    python3 xingbao.py --confirm-motion --no-vision  # 只跑动作，不判霉菌
    python3 xingbao.py --talk-only                   # 对讲模式：只听不动，验语音链路
    python3 xingbao.py --self-test                   # 离线单测，不连机器人
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import demo_cheese                                    # noqa: E402
from demo_cheese import DemoStopped                   # noqa: E402
from mission import report as report_mod              # noqa: E402
from mission.route import HOLDING, DONE, FAILED       # noqa: E402
from voice import cloud, intent                       # noqa: E402
from voice import intent_llm                          # noqa: E402
from voice import kws as kws_mod                       # noqa: E402
from voice import mic as mic_mod                      # noqa: E402
from voice import wake as wake_mod                    # noqa: E402
from voice import wake_asr as wake_asr_mod            # noqa: E402
from voice.assistant import Assistant, WAKE_WORD, looks_like_log  # noqa: E402
from voice.memory import MemoryStore                  # noqa: E402

FACES_PER_ROUND = 2          # 一圈 = 正面 + 反面
# 唤醒判定默认走 ASR（原仓库 asr_wake_word.py 的机制）。2026-11-08 实测：
# 真人喊"星宝星宝"，本地 KWS 只过 2.3%，openWakeWord 分数恒为 0.001，
# 而 ASR 把 3/3 段真人录音都识别成"新宝新宝"，同音字白名单全中。
DEFAULT_WAKE_FALLBACK = "asr"

# 语音进程的互斥锁。**一把锁管所有"要开麦克风"的脚本**（xingbao.py 和
# ask.py --voice 都来抢这把）。2026-11-08 现场事故：5 个 ask.py 残留进程
# 一起开着麦克风、一起念 TTS，用户对着它们说话，星宝本体一句也没收到，
# 而 /tmp/xingbao.log 里全是 ask.py 的记录 —— 看起来就像"星宝在工作但不动"。
VOICE_LOCK = "/tmp/xingbao_voice.lock"

# 启动横幅里给人看的后端名（参数值 -> 中文说明）
_WAKE_ENGINE_LABEL = {
    "kws": "sherpa-onnx 关键词模型",
    "model": "openWakeWord 三段式",
    "auto": "自动（kws 优先）",
    "text": "文本唤醒（无模型）",
}

# 别人的语音进程长什么样：脚本名 + 这几个参数之一就说明它要占麦克风。
_VOICE_SCRIPTS = ("ask.py", "xingbao.py", "demo_cheese.py", "main.py")
_VOICE_ARGS = ("--voice", "--talk-only", "--confirm-motion", "--wake-watch")


class _FileLog(object):
    """**只写文件**的极简日志器。

    2026-11-08 用户要求："不要把日志推给星宝的语音系统。"

    上一版把 ``sys.stdout`` 整个换成 Tee（终端 + 文件双写）—— 那等于把
    语音链路上所有 ``print``（``♪ 播报 tts``、``[vad] 本底…``、``[chat] 联网搜索``）
    都卷进日志，也让日志和播报共用同一条输出通道。两条路必须彻底分开：

    * 终端 print：给现场的人看，语音模块照旧自己打；
    * 这个 ``_FileLog``：只写文件，只有本文件显式 ``log()`` 的关键事件进去。

    它不接管 stdout、不接管 stderr、不碰任何语音相关对象。写失败一律静默
    （日志丢了不能影响机器人干活）。
    """

    def __init__(self, path):
        self.path = path
        self._handle = None
        if not path:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(path, "a", encoding="utf-8", buffering=1)
        except Exception as exc:                 # noqa: BLE001
            print("  ⚠ 日志文件打不开（{}），这次就不记了".format(exc),
                  file=sys.stderr)
            self._handle = None

    def log(self, fmt, *args):
        if self._handle is None:
            return
        try:
            text = fmt.format(*args) if args else fmt
            self._handle.write("[{}] {}\n".format(
                time.strftime("%H:%M:%S"), text))
            self._handle.flush()
        except Exception:                        # noqa: BLE001
            pass

    def close(self):
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:                    # noqa: BLE001
                pass
            self._handle = None


# 全局日志器。`_setup_log()` 之后才可用；没设就是 None（no-op）。
LOG = None


def _setup_log(path):
    """打开文件日志。**不接管 stdout**，和语音系统完全隔离。"""
    global LOG
    if not path:
        LOG = None
        return None
    LOG = _FileLog(path)
    LOG.log("=" * 60)
    LOG.log("星宝启动 PID={} argv={}", os.getpid(), " ".join(sys.argv[1:]))
    return LOG


def _argv(pid):
    """读 ``/proc/<pid>/cmdline`` 成列表。读不到返回空列表。"""
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as handle:
            raw = handle.read()
    except Exception:                            # noqa: BLE001
        return []
    return [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]


def _proc_state(pid):
    """取进程状态字母（R/S/T/Z…）。Z=僵尸，T=挂起，都不占麦克风。"""
    try:
        with open("/proc/%d/stat" % pid, "r", encoding="utf-8",
                  errors="replace") as handle:
            data = handle.read()
        return data[data.rfind(")") + 2:][:1]
    except Exception:                            # noqa: BLE001
        return "?"


def _ancestors(pid):
    """自己的所有祖先 PID。排除它们，免得把启动自己的 timeout / bash 当成冲突。"""
    seen = set()
    cursor = pid
    for _ in range(16):
        try:
            with open("/proc/%d/stat" % cursor, "r", encoding="utf-8",
                      errors="replace") as handle:
                data = handle.read()
            ppid = int(data[data.rfind(")") + 2:].split()[1])
        except Exception:                        # noqa: BLE001
            break
        if ppid <= 1 or ppid in seen:
            break
        seen.add(ppid)
        cursor = ppid
    return seen


def _voice_processes(me=None):
    """扫 /proc，找出别的"语音脚本"进程。

    判定要按 **argv 数组**来，不能把整条命令行拼成字符串去搜 ——
    ``bash -c "... xingbao.py --talk-only ..."`` 这种包装命令会被误判成
    "另一个星宝在跑"，结果自己把自己拦在门外（2026-11-08 第一次实测就踩了）。
    所以：只认"某个 argv 元素的 basename 是本项目的 .py 脚本"。
    """
    me = os.getpid() if me is None else me
    skip = _ancestors(me) | {me}
    found = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return found
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in skip:
            continue
        argv = _argv(pid)
        if not argv:
            continue
        script = None
        for item in argv[:6]:                    # 脚本一般在最前面几个位置
            base = os.path.basename(item)
            if base.endswith(".py") and base in _VOICE_SCRIPTS:
                script = base
                break
        if script is None:
            continue
        if not any(arg in argv for arg in _VOICE_ARGS):
            continue
        state = _proc_state(pid)
        if state in ("Z", "T", "t"):             # 僵尸/挂起：不占麦克风
            continue
        found.append((pid, script, " ".join(argv)))
    return found


# 这两类脚本是"真的在抢麦克风"：它们都听"星宝星宝"、都会播报。
# main.py / demo_cheese.py 只是出声不出话（按键驱动），属于"提醒"级别。
_HARD_CONFLICT = ("ask.py", "xingbao.py")


def _take_lock():
    """拿语音锁。拿不到返回 None（说明已经有别的语音进程在跑）。"""
    try:
        import fcntl
    except ImportError:
        return object()                          # Windows 上跑 self-test 用，不锁
    try:
        handle = open(VOICE_LOCK, "w")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:                            # noqa: BLE001
        return None
    try:
        handle.write(str(os.getpid()))
        handle.flush()
    except Exception:                            # noqa: BLE001
        pass
    return handle


def _check_voice_env(force=False):
    """启动前的"清场检查"。返回 False 表示别启动。

    分两档：
    * **硬冲突**：``ask.py --voice`` / 另一个 ``xingbao.py``。它们同样在听
      "星宝星宝"、同样会回答，用户根本分不清谁在说话 —— 而且 ask.py 永远
      不会动机器人。必须清掉才能启动。
    * **软提醒**：``main.py`` / ``demo_cheese.py`` 这类键盘驱动的演示。它们
      会占喇叭、也会机械臂，但不抢唤醒词，只提醒一句。
    """
    others = _voice_processes()
    hard = [item for item in others if item[1] in _HARD_CONFLICT]
    soft = [item for item in others if item[1] not in _HARD_CONFLICT]

    if soft:
        print("提示：这些演示程序也在跑，喇叭/机械臂会和星宝抢：", file=sys.stderr)
        for pid, _script, cmd in soft:
            print("  PID {:>7}  {}".format(pid, cmd), file=sys.stderr)
        print("  （它们不听语音，只是会念 TTS。不冲突就继续。）\n", file=sys.stderr)

    if not hard:
        return True

    print("发现别的语音进程正在跑，它们会抢麦克风和喇叭：", file=sys.stderr)
    for pid, _script, cmd in hard:
        print("  PID {:>7}  {}".format(pid, cmd), file=sys.stderr)
    print("\n  现场最常见的坑：`ask.py --voice` 是**纯问答脚本，永远不会动**，"
          "\n  但它照样会回答'我去看看''正反面扫一遍'，看起来就像星宝在干活。"
          "\n\n  先清掉它们再启动：\n"
          "     bash stop_voice.sh\n", file=sys.stderr)
    if not force:
        print("  （确实要同时跑就加 --force，但麦克风只有一个）", file=sys.stderr)
        return False
    print("  --force：仍然启动，出问题自己负责。", file=sys.stderr)
    return True


# --------------------------------------------------------------------------
# 语音主控
# --------------------------------------------------------------------------
class VoiceCore(object):
    """把"听 -> 判 -> 做 -> 说"串起来。所有动嘴/动手都从这里走。"""

    def __init__(self, assistant, inspector=None, router=None,
                 loops=1, route_on=False, verbose=True, talk_only=False):
        self.assistant = assistant
        self.inspector = inspector
        self.router = router
        self.loops = max(1, int(loops))
        self.route_on = bool(route_on)
        self.talk_only = bool(talk_only)
        self._verbose = bool(verbose)
        self.memory = assistant.memory or MemoryStore(verbose=verbose)
        self._busy = False
        self._stops = 0

    # ------------------------------------------------------------------
    def say(self, text, wait=True):
        if text:
            print("  [星宝] {}".format(text))
        return self.assistant.speak(text, wait=wait)

    # ------------------------------------------------------------------
    def run_forever(self):
        """一直待命。唤醒 -> 处理一句话 -> 回到待命。"""
        if self.talk_only:
            self.say("对讲模式。叫我{}，我们聊天，我不会动。".format(WAKE_WORD))
        else:
            self.say("星宝已就绪。叫我{}，然后告诉我要不要检测奶酪。".format(WAKE_WORD))
        self.assistant.open_mic()
        if LOG:
            LOG.log("麦克风已开，进入待命")
        misses = 0
        while True:
            try:
                # 每轮开始都把积压倒掉：上一轮跑任务时播的 pass/fail 语音
                # 都在队列里，不清掉会被唤醒模型当成人声，自己把自己叫醒。
                self.assistant.quiet()
                rest = self.assistant.wait_for_wake(keep_open=True)
                if rest is None:
                    return
                question = rest
                if LOG:
                    LOG.log("唤醒，附带内容：{}", question or "（空）")
                if not question:
                    self.say("我在。")
                    question = self.assistant.hear()
                if not question:
                    # 唤醒了却没听到话：不吭声会让人以为机器死了（2026-11-08 现场反馈）
                    misses += 1
                    if LOG:
                        LOG.log("唤醒后没听到话（第 {} 次）", misses)
                    if misses >= 2:
                        self.say("我没听清，你再说一遍好吗？")
                        misses = 0
                    continue
                misses = 0
                self.handle(question)
            except KeyboardInterrupt:
                raise
            except DemoStopped as exc:
                self.say("好，我停下了。")
                print("  [主控] 任务被叫停：{}".format(exc))
            except Exception as exc:                     # noqa: BLE001
                print("  [主控] 这一轮出错，继续待命：{}".format(exc))
                traceback.print_exc()
                if LOG:
                    LOG.log("这一轮出错：{}", exc)
                time.sleep(1.0)

    # ------------------------------------------------------------------
    def handle(self, text):
        """一句话进来，决定干什么。

        **是否真的动手，交给大模型推理**（``intent_llm.route``）。
        2026-11-08 现场问题："无论我说什么都直接跑过去做任务" —— 根因是旧
        ``intent.classify`` 靠"看一下/帮我/检测"这种泛词匹配，几乎每句都命中。
        现在只有大模型判成 detect 才动；其它一律走正常对话 + 联网搜索。
        停止词仍是规则、且优先级最高，不交给模型。

        **入口闸门**：如果这句话其实是程序日志/技术串（识别异常、参数打印、
        报错原文），它既不能当指令去驱动机械臂，也不能被朗读。直接请求重说。
        这一层是"日志不许进流程"的落点。
        """
        text = (text or "").strip()
        if not text:
            self.say("我在，你说。")
            return
        if looks_like_log(text):
            print("  [意图] 这段像日志，不进流程也不念：{}".format(text[:70]))
            if LOG:
                LOG.log("丢弃疑似日志输入：{}", text[:120])
            self.say("我没听清，你再说一遍好吗？")
            return

        kind, why = intent_llm.route(text, brain=getattr(self.assistant, "brain", None),
                                     verbose=self._verbose)
        print("  [意图] {}（{}）：{}".format(kind, why, text))
        if LOG:
            LOG.log("意图 {}（{}）：{}", kind, why, text)

        if kind == intent.STOP:
            self.say("好，我不动。")
            return
        # 笑声：放一段音频而已，不动手臂不动底盘，所以放在规则层先判，
        # 不必等大模型那一趟往返（说"笑一个"之后干等两秒就没效果了）。
        if kind != intent.START_TASK and intent.is_laugh(text):
            self.laugh()
            return
        if kind == intent.START_TASK:
            loops = intent_llm.loops_of(text, getattr(self.assistant, "brain", None))
            if self.talk_only:
                self.say("我听懂了，是要我检测 {} 块奶酪。不过现在是对讲模式，"
                         "我只听不动；正式运行时我就直接去了。".format(loops))
                return
            self.run_task(loops)
            return
        if kind == intent.STATUS:
            self.say(self.describe_status())
            return
        if kind == intent.CONFIRM:
            # 待命状态下说"好了"没有上下文，别当成启动信号
            self.say("我现在闲着。要检测的话就说，去检测一下这块奶酪。")
            return
        # 其余：正常对话（带上下文记忆 + 按需联网搜索）
        self.say(self.assistant.reply(text))

    # ------------------------------------------------------------------
    def laugh(self):
        """放一声奶龙的笑（``voice/laugh.wav``，19.8 s）。

        为什么单独一个方法、而且**不等它播完**：笑的时候用户往往接着说话，
        卡 20 秒会让人觉得死机。丢进 VoiceBox 的队列就返回，和 pass/fail
        那几支音频走的是同一条链路（队列 + 后台线程），互不阻塞。

        任务正在跑的时候不笑 —— 那会让"该播的结论"排在笑声后面。
        """
        if self._busy:
            self.say("我正忙着呢，等这块测完再笑给你听。")
            return
        box = self.assistant.voice
        if box is None:
            self.say("我这会儿笑不出来，喇叭没连上。")
            return
        if not box.has(LAUGH_CLIP) and not load_laugh(box):
            self.say("我这会儿笑不出来，笑声素材没找到。")
            return
        print("  [笑声] 播放奶龙大笑")
        if LOG:
            LOG.log("播放奶龙大笑")
        try:
            box.say(LAUGH_CLIP)
        except Exception as exc:                          # noqa: BLE001
            print("  [笑声] 播不出来：{}".format(exc))
            self.say("我这会儿笑不出来。")

    # ------------------------------------------------------------------
    def describe_status(self):
        said = self.memory.describe_last()
        if said:
            return said
        return "我还没测过奶酪，你让我去看一块吧。"

    # ------------------------------------------------------------------
    def run_task(self, loops=None):
        """跑任务。动真格的动作都在这里。"""
        if self._busy:
            self.say("我正在测呢，等我弄完。")
            return
        loops = max(1, int(loops or self.loops))
        total_looks = len(getattr(self.inspector, "results", []) or [])

        self._busy = True
        if LOG:
            LOG.log("开始任务，块数={}", loops)
        try:
            if loops == 1:
                self.say("好，我去看看面前这块奶酪。")
            else:
                self.say("好，我连着检测 {} 块。".format(loops))

            # 收尾移动（转身谢幕 + 回巡检起点）不在 run() 里做，先接住它的
            # 状态，等我播完最终结论再动 —— 现场看的是"说完 -> 转身 -> 开走"，
            # 要是先开走再说话，观众会等一段莫名其妙的沉默。
            closing = {}
            state = demo_cheese.run(
                self.robot, self.arm, self.leg, self.nav,
                vision=self.inspector,
                debug=False,
                voice=self.assistant.voice,
                router=self.router if self.route_on else None,
                loops=loops,
                confirm=self.voice_confirm,
                on_round=self.on_round if loops > 1 else None,
                say_farewell=False,          # 汇报代替道别
                ask_after_last=False,        # 测完直接收身，不多按一次
                route_after=self.route_on,
                home_at_end=False,           # 收尾由下面播完之后自己调
                closing_hook=closing.update,
            )

            looks = list(getattr(self.inspector, "results", []) or [])[total_looks:]
            rounds = report_mod.group_by_round(looks, FACES_PER_ROUND)
            route_text = self._route_text(state)
            record = report_mod.record(rounds, route_text=route_text, loops=loops)
            self.memory.add_task(record)
            final = self._final_speech(rounds, route_text, loops)
            if LOG:
                LOG.log("任务完成：{}", final)
            self.say(final)
            self.close_show(closing)
        except DemoStopped:
            self.memory.add_task(dict(loops=loops, moldy=0, clean=0, unknown=0,
                                      spoken="这次任务被叫停，没有结论。",
                                      verdict="被叫停", route="停在原地"))
            raise
        except Exception as exc:                          # noqa: BLE001
            print("  [主控] 任务失败：{}".format(exc))
            traceback.print_exc()
            self.memory.add_task(dict(loops=loops, moldy=0, clean=0, unknown=0,
                                      spoken="这次没做成。", verdict="失败",
                                      route="停在原地", error=str(exc)[:200]))
            # 播报只用自然语言；异常原文只进终端和日志（2026-11-08 用户要求）。
            self.say(_exc_speech(exc))
        finally:
            self._busy = False

    # ------------------------------------------------------------------
    def on_round(self, index, looks):
        """多块模式：每块测完立刻汇报。单块模式由最终汇报统一说。"""
        item = report_mod.summarize(looks)
        self.say("第 {} 块：{}".format(index, item["spoken"]))

    def close_show(self, closing):
        """播完最终结论之后的收尾移动：原地转 180° + 开回巡检起点。

        搬过奶酪（送去终点 A/B 并放下）才走这一段 —— 没搬的车还在检测桌工作位上，
        转身再开走会把下一句话要用的工作位丢掉。动作本身见
        ``demo_cheese.closing_moves``；这里只负责"什么时候动"和"出错不许影响
        已经说出去的结论"。
        """
        if not closing or not closing.get("carried"):
            return
        try:
            demo_cheese.closing_moves(
                self.robot, self.arm, self.leg, self.nav,
                closing.get("origin"), closing.get("table_pose"),
                carried=True, debug=False, another_piece=False)
        except DemoStopped:
            raise
        except Exception as exc:                          # noqa: BLE001
            print("  [收尾] 移动没走成：{}".format(exc))

    def voice_confirm(self, title):
        """替代 ``input()`` 的那声回车：星宝问一句，听人答一句。

        返回 ``"go"`` / ``"stop"``。不设硬超时 —— 推车要多久取决于人；
        每 6 次没听清就再问一遍，免得现场以为机器死了。
        """
        self.say(title)
        silent = 0
        while True:
            heard = self.assistant.hear()
            if not heard:
                silent += 1
                if silent % 6 == 0:
                    self.say("推好了就跟我说一声，好了。")
                continue
            silent = 0
            print("  [确认] {}".format(heard))
            if intent.is_stop(heard):
                return "stop"
            if intent.is_confirm(heard):
                return "go"
            self.say("我是问你推好了没有。推好了说好了，不想做了就说停。")

    # ------------------------------------------------------------------
    # 终点在播报里的叫法。"route_a" 是给人看日志的，念出来像念代码，
    # 现场观众听到的应该是"一号桌/二号桌"（2026-11-08）。
    _GOAL_SPOKEN = {"route_a": "一号桌", "route_b": "二号桌"}
    # 现在这块奶酪已经随手搬走了 —— 播报要说清"它去哪儿了"，
    # 不能再邀请"还要看别的"（桌上已经空了）。
    _CARRIED_AWAY_HINT = "这块我已经搬走了，换一块再叫我。"

    def _goal_spoken(self, goal_name):
        return self._GOAL_SPOKEN.get(goal_name, "终点")

    def _route_text(self, state):
        if not self.route_on or self.router is None:
            return ""
        decision = getattr(self.router.st, "decision", None)
        goal = getattr(decision, "goal_name", None)
        if state == HOLDING:
            return "留在原地"
        if state == DONE:
            return "已经把这块奶酪送到{}了".format(self._goal_spoken(goal))
        if state == FAILED:
            return "想送走的，没走成，就放在原地了"
        return ""

    def _final_speech(self, rounds, route_text, loops):
        if not rounds:
            return "这次没看清，我没拿到结论。"
        if loops == 1:
            text = rounds[0]["spoken"]
        else:
            total_moldy = sum(r["moldy"] for r in rounds)
            total_clean = sum(r["clean"] for r in rounds)
            total_unknown = sum(r["unknown"] for r in rounds)
            parts = ["{} 块测完了".format(len(rounds))]
            parts.append("{} 面有霉".format(total_moldy) if total_moldy else "没发现霉")
            if total_unknown:
                parts.append("{} 面没看清".format(total_unknown))
            elif total_clean:
                parts.append("{} 面干净".format(total_clean))
            text = "，".join(parts) + "。"
        if route_text:
            text = text + "我" + route_text + "。"
        # 主动邀请下一轮：现场用户最常说的就是"再来一块/再测一次"，
        # 先说一句"还要就说"，省得他以为一次就结束了。
        #
        # 2026-11-08：走了"抱着奶酪送终点"以后，这张桌子上已经没有奶酪了，
        # 不能再让观众"再测一次"，得告诉他换一块。
        if not self.talk_only:
            if "送到" in (route_text or ""):
                text = text + self._CARRIED_AWAY_HINT
            else:
                text = text + "还要看别的，跟我说一声。"
        return text

    # 由 main 注入，避免构造函数参数过长
    robot = None
    arm = None
    leg = None
    nav = None


def _brief(exc):
    """异常的一句话摘要 —— **只给终端/日志看，绝不进 TTS**。

    2026-11-08 用户要求"不要让任何日志进入 TTS 播报"：以前这里是
    ``self.say("这次没做成：{}".format(_brief(exc)))``，于是
    "ConnectionError: HTTPSConnectionPool(host='dashscope.aliyuncs.com'..."
    会被当场念给观众听。播报一律走 ``_exc_speech``。
    """
    text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return text[:60]


# 异常 -> 现场能听懂的一句话。**按"观众/操作员该做什么"分类，不解释技术原因。**
# 关键词必须够具体：早期版本用了 "ik"/"arm" 这种两三个字母的词，
# "like"/"alarm" 都会被误命中，于是把网络错误说成"手不听话"。
# 顺序即优先级。导航放在网络前面：``goal unreachable`` 两边的词都像，
# 但"我找不到路"比"我连不上网"更贴近现场该做的事。
_EXC_RULES = (
    (("timeout", "timed out", "超时"),
     "我这会儿反应有点慢，没跟上。你再让我试一次就好。"),
    (("定位跳变", "位姿突变", "relocaliz"),
     "我刚才定位飘了一下，正在重新对位置。你再让我试一次就好。"),
    (("estop", "e_stop", "emergency", "急停", "collision", "safety_", "安全"),
     "我好像被拦住了，动不了。麻烦看一下我身边有没有东西挡着。"),
    (("localiz", "goal", "planning", "navigation", "navctrl", "定位", "导航"),
     "我找不到路，可能得麻烦你把我推到桌子前面。"),
    (("connection", "network", "resolution", "ssl", "proxy",
      "remote end", "连接", "网络", "断网"),
     "我这会儿连不上网，等一下再让我试试。"),
    (("armctrl", "gripper", "joint", "trajectory", "inverse kinematic",
      "手臂", "夹爪", "机械臂"),
     "我的手有点不听话，没能抓住。麻烦你帮我摆一下。"),
    (("camera", "vision", "image", "frame", "相机", "图像", "看不清"),
     "我的眼睛这会儿看不清，让我再看一次。"),
)


def _exc_speech(exc):
    """把异常翻译成一句自然语言播报（绝不带技术细节）。"""
    if isinstance(exc, DemoStopped):
        return "好，我停下了。"
    text = "{} {}".format(type(exc).__name__, _brief(exc)).lower()
    for keys, line in _EXC_RULES:
        if any(key in text for key in keys):
            return line
    return "我刚才卡了一下，没做成。你再让我试一次就好。"


# --------------------------------------------------------------------------
# 离线自检：不连机器人，验证意图/记忆/结论三块
# --------------------------------------------------------------------------
def self_test():
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print("  {} {:<34} {}".format("✓" if good else "✗", name, got if good else
                                      "{}（期望 {}）".format(got, want)))

    print("\n[1/5] 意图识别")
    cases = [
        ("你去检测一下面前这块奶酪", intent.START_TASK),
        ("开始吧", intent.START_TASK),
        ("检查一下这个奶酪", intent.START_TASK),
        ("检测结果怎么样", intent.STATUS),
        ("刚才那块奶酪有霉吗", intent.STATUS),
        ("好了", intent.CONFIRM),
        ("我推好了", intent.CONFIRM),
        ("停", intent.STOP),
        ("别做了", intent.STOP),
        ("你叫什么名字", intent.QUESTION),
        ("它为什么会长霉", intent.QUESTION),
        ("停一下，检测结果怎么样", intent.STOP),
        # 2026-11-08 用户报的 bug：测完一次说"再检测一次"被拒绝
        ("再检测一次", intent.START_TASK),
        ("再来一块", intent.START_TASK),
        ("重来", intent.START_TASK),
        ("再测一遍", intent.START_TASK),
        ("帮我再看看", intent.START_TASK),
        ("不要再检测了", intent.STOP),
        ("不用了", intent.STOP),
        ("够了", intent.STOP),
        # 问事不是命令，不能让机器伸手
        ("你刚才测过吗", intent.QUESTION),
        # 问天气必须走问答，绝不能变成动作
        ("今天天气怎么样", intent.QUESTION),
        ("这个奶酪怎么样", intent.STATUS),
        ("检测得怎么样", intent.STATUS),
        ("现在几点了", intent.QUESTION),
    ]
    for text, want in cases:
        check(text, intent.classify(text)[0], want)
    check("要几块：三块", intent.wanted_loops("把这三块都检测一下"), 3)
    check("要几块：默认", intent.wanted_loops("去检测一下面前这块奶酪"), 1)
    check("要几块：再来一块=1", intent.wanted_loops("再来一块"), 1)

    print("\n[2/4] 结论生成")
    def look(tag, mold, reason="ok"):
        return dict(tag=tag, mold=mold, reason=reason, verdict="")
    two = report_mod.summarize([look("第1次 正面（未翻腕）", True),
                                look("第1次 反面（已翻腕）", True)])
    check("两面都有霉", two["spoken"], "这块奶酪两面都有霉。")
    one = report_mod.summarize([look("正面（未翻腕）", True),
                                look("反面（已翻腕）", False)])
    check("一面有霉", one["spoken"], "这块奶酪一面有霉，另一面是干净的。")
    clean = report_mod.summarize([look("正面（未翻腕）", False),
                                  look("反面（已翻腕）", False)])
    check("合格", clean["spoken"], "这块奶酪两面都干净，是合格的。")
    blind = report_mod.summarize([look("正面", False, "no_cheese"),
                                  look("反面", False)])
    check("一面没看清", blind["spoken"],
          "我看到的那面是干净的，还有 1 面没看清，我不敢下结论。")
    check("没看清不算合格", blind["ok"], False)
    groups = report_mod.group_by_round([look("正面", True), look("反面", False),
                                        look("正面", False), look("反面", False)])
    check("按圈切分", [g["moldy"] for g in groups], [1, 0])

    print("\n[3/4] 记忆")
    tmp = Path(HERE) / "voice" / "_selftest_memory.json"
    if tmp.exists():
        tmp.unlink()
    store = MemoryStore(tmp, verbose=False)
    store.add_task(dict(spoken="这块奶酪一面有霉，另一面是干净的。", moldy=1,
                        clean=1, unknown=0, time="11-08 03:30", route="留在原地"))
    check("最近一条", store.last_task()["moldy"], 1)
    check("累计有霉次数", store.moldy_count(), 1)
    check("上下文含结论", "一面有霉" in store.context_text(), True)
    store.add_turn("user", "刚才那块怎么样")
    check("对话记忆", store.history(2)[0]["content"], "刚才那块怎么样")
    store2 = MemoryStore(tmp, verbose=False)
    check("重启后仍在", len(store2.tasks), 1)
    tmp.unlink()
    if tmp.with_suffix(".tmp").exists():
        tmp.with_suffix(".tmp").unlink()

    print("\n[4/4] 云端问答的搜索判定")
    from voice import assistant as assistant_mod
    check("天气要搜", assistant_mod._needs_search("今天天气怎么样"), True)
    check("新闻要搜", assistant_mod._needs_search("最近有什么新闻"), True)
    check("奶酪不搜", assistant_mod._needs_search("帕马森奶酪是什么"), False)
    check("推辞能认出", assistant_mod._looks_refused(
        "我无法实时获取当前天气信息，知识截止到2024年"), True)
    check("正常回答不算推辞", assistant_mod._looks_refused("成都今天小雨，12到17度。"), False)
    check("提示词禁拒绝", "永远不要拒绝" in assistant_mod.Brain.SYSTEM, True)
    check("提示词要求重测", "再测一次" in assistant_mod.Brain.SYSTEM, True)
    check("提示词允许聊天气", "天气" in assistant_mod.Brain.SYSTEM, True)

    print("\n[5/6] 阈值与日志隔离")
    check("唤醒阈值合理", 0.10 <= kws_mod.DEFAULT_THRESHOLD <= 0.40, True)
    check("VAD 倍数已放低", mic_mod.VAD_FLOOR_MULT <= 1.3, True)
    check("VAD 下限已放低", mic_mod.VAD_FLOOR_MIN <= 160, True)
    # 日志绝不能被念出去、也绝不能被当成指令（2026-11-08 用户要求）。
    # 样例取自现场真实输出，不是编的。
    for sample in ("[vad] 本底 1683 x1.5 -> 阈值 2525",
                   "[chat] 联网搜索 自动（2.0 s，prompt 2718 token）",
                   "[wake] 命中「星宝星宝」 分数 0.42",
                   "Traceback (most recent call last)",
                   "stream_id=mic_stream_179 chunk=2560",
                   "麦克风已开：stream_id=mic_stream_1794092988218973_7897",
                   "ConnectionError: HTTPSConnectionPool(host='dashscope.aliyuncs.com')",
                   "  File \"/home/galbot/scripts/voice/assistant.py\", line 606",
                   "/home/galbot/scripts/voice/cache/heard_014.wav",
                   "  [wake] ASR 唤醒已启用（VAD 断句 -> 识别 -> 同音字白名单）",
                   "意图 question（大模型判定是聊天）：帮我查一下天气",
                   "INTENT_LLM_MIN_CONFIDENCE=0.55",
                   "♪ 播报 tts（5.20 s）",
                   "md5 1564dd7386d38c206d838c2aff2de591"):
        check("日志被拦住：" + sample[:14], looks_like_log(sample), True)
    # 反过来：正常人话里出现英文词/数字，**不能**被误拦（以前 token/prompt
    # 是硬匹配，问一句"token 是什么意思"就会被当成日志）。
    for sample in ("这块奶酪一面有霉，另一面是干净的。",
                   "好，我去看看面前这块奶酪。",
                   "今天成都阴天，气温12到17度。",
                   "对对对，这个是奶酪。",
                   "你说的 token 是什么意思？",
                   "我这边网有点慢，你再说一遍好吗？",
                   "刚才那块奶酪两面都干净，是合格的。"):
        check("人话照念：" + sample[:8], looks_like_log(sample), False)
    # 拦下来之后不能闭嘴，要给一句人话
    check("拦下后改说人话",
          assistant_mod.to_natural("[vad] 本底 1683") != "", True)
    check("改说的不是原文",
          assistant_mod.to_natural("[vad] 本底 1683") == assistant_mod._LOG_REPLY, True)
    check("人话原样返回",
          assistant_mod.to_natural("我在呢。"), "我在呢。")

    # 异常 -> 人话：绝不能把异常原文念出去（2026-11-08 用户要求）
    print("\n[5b/7] 异常转人话")
    cases = [
        (RuntimeError("ConnectionError: HTTPSConnectionPool(host='dashscope.aliyuncs.com')"),
         "连不上网"),
        (TimeoutError("timed out after 5.0s"),
         "反应有点慢"),
        (RuntimeError("EstopButton pressed, safety stop"),
         "被拦住了"),
        (RuntimeError("navctrl: goal unreachable"),
         "找不到路"),
        (RuntimeError("grpcio import like error"),
         "卡了一下"),          # "like" 不能命中"手不听话"
        (RuntimeError("something odd happened"),
         "卡了一下"),
    ]
    for exc, want in cases:
        said = _exc_speech(exc)
        check("异常说人话：" + type(exc).__name__, want in said, True)
        check("不带异常原文：" + type(exc).__name__,
              "Error" not in said and "host=" not in said, True)
    for sample in ("[vad] 本底 1683", "Traceback (most recent call last)"):
        check("日志不会被经异常通道念出", looks_like_log(sample), True)

    print("\n[6/6] 唤醒模型（sherpa-onnx KWS）")
    det = kws_mod.SherpaKwsDetector(verbose=False)
    missing = det.missing_files()
    check("模型文件齐全", not missing, True)
    if missing:
        for item in missing:
            print("        缺：" + item)
    kw_text = ""
    if det.keywords_path.is_file():
        kw_text = det.keywords_path.read_text(encoding="utf-8")
    check("关键词含「星宝星宝」", "@" + WAKE_WORD in kw_text, True)
    check("关键词用拼音标注", "ng" in kw_text and "o @" in kw_text, True)
    # 装了 sherpa-onnx 就真加载一次；没装不算失败（换机器/换环境常见）
    try:
        import sherpa_onnx                              # noqa: F401
        has_sherpa = True
    except Exception:                                   # noqa: BLE001
        has_sherpa = False
    if has_sherpa:
        check("模型能加载", det.prime(), True)
        if not det.available():
            print("        " + (det.error or "原因未知"))
    else:
        print("  .. 跳过加载测试：本机没装 sherpa-onnx"
              "（机器人上需要 python3 -m pip install --user sherpa-onnx）")
    check("后端默认是 kws", kws_mod.build_detector.__defaults__[0] == "kws", True)

    # ---- ASR 唤醒（原仓库 asr_wake_word.py 的白名单机制）----
    print("\n[7/7] ASR 唤醒白名单")
    # 这几条是 2026-11-08 用真实录音跑出来的：真人喊"星宝星宝"，
    # qwen3-asr-flash 一律转成「新宝新宝」，所以同音字表必须收"新宝"。
    for text, want_woke, want_rest in (
            ("新宝新宝。", True, ""),
            ("新宝新宝，这个奶酪新鲜吗？", True, "这个奶酪新鲜吗"),
            ("星，星爆，星爆。", True, ""),
            ("这个星宝挺好玩", False, ""),
            ("今天的演示到这里，谢谢大家。", False, "")):
        woke, rest, keyword, _why = wake_asr_mod.split_wake(text)
        check("唤醒判定：{}".format(text),
              (woke, rest) == (want_woke, want_rest),
              True)
    check("默认走 ASR 唤醒", DEFAULT_WAKE_FALLBACK == "asr", True)

    print("\n{}".format("全部通过" if ok else "有失败项，看上面的 ✗"))
    return 0 if ok else 1


# --------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="星宝语音主控")
    parser.add_argument("--confirm-motion", action="store_true",
                        help="确认急停已松开、WORKING_MODE、周围净空，且有人能拍急停")
    parser.add_argument("--self-test", action="store_true",
                        help="离线自检意图/记忆/结论，不连机器人")
    parser.add_argument("--talk-only", action="store_true",
                        help="对讲模式：只连音频，验证唤醒/识别/意图/记忆，绝不动作")
    parser.add_argument("--loops", type=int, default=1,
                        help="一次任务默认测几块，默认 1（说'三块'会临时覆盖）")
    parser.add_argument("--no-route", dest="route", action="store_false",
                        help="关掉测完按霉数分流（**默认是开的**：0 不动 / 1 去 A / "
                             "2 去 B，抱着奶酪开到终点再松爪放下）")
    parser.add_argument("--count-mode", default=demo_cheese.ROUTE_COUNT_MODE,
                        choices=("faces", "spots"))
    parser.add_argument("--no-vision", action="store_true", help="只跑动作，不判霉菌")
    parser.add_argument("--volume", type=float, default=None,
                        help="喇叭音量 0-100，默认取 vision/config.py（50）")
    parser.add_argument("--wake-th", type=float, default=None,
                        help="唤醒阈值，默认 {:.2f}（嫌迟钝就 0.15，误唤多就 0.35）。"
                             "注意这是唤醒模型的分数，和 VAD 不是一个东西".format(
                                 kws_mod.DEFAULT_THRESHOLD))
    parser.add_argument("--vad-mult", type=float, default=None,
                        help="VAD 阈值倍数：阈值=本底×这个值，默认 {:.2f}。"
                             "录不到话就调小，噪声被当人声就调大".format(
                                 mic_mod.VAD_FLOOR_MULT))
    parser.add_argument("--wake-engine", choices=("kws", "model", "auto", "text"),
                        default="kws",
                        help="唤醒后端：kws=sherpa-onnx 关键词检测（默认，推荐）；"
                             "model=旧的 openWakeWord 三段式；auto=先 kws 再 model；"
                             "text=不用模型，靠识别结果找'星宝'")
    parser.add_argument("--wake-fallback", choices=("asr", "off"),
                        default=DEFAULT_WAKE_FALLBACK,
                        help="唤醒判定方式：asr=**默认**，VAD 断句后送识别，用"
                             "同音字白名单找'星宝'（原仓库 asr_wake_word.py 的机制，"
                             "现场真人喊最可靠）；off=只信本地 KWS 模型"
                             "（实测真人召回只有 2% 左右，不建议）")
    parser.add_argument("--wake-asr-cooldown", type=float, default=0.6,
                        help="两次唤醒识别之间的最小间隔秒数，默认 0.6"
                             "（只在 --wake-fallback asr 时生效）")
    parser.add_argument("--memory-reset", action="store_true",
                        help="启动前清空记忆（展会换场时用）")
    parser.add_argument("--log", default="/tmp/xingbao.log",
                        help="日志文件（只写文件，不接管终端、不进语音通道）")
    parser.add_argument("--no-log", action="store_true", help="不写日志文件")
    parser.add_argument("--force", action="store_true",
                        help="即使发现别的语音进程在跑也强行启动（不推荐）")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    # 日志最先接上：启动阶段的报错（连机器人失败、唤醒模型加载失败）最容易丢。
    _setup_log(None if args.no_log else args.log)

    if not args.confirm_motion and not args.talk_only:
        print("未加 --confirm-motion，没有连接机器人。\n"
              "  正式跑：python3 xingbao.py --confirm-motion\n"
              "  只聊天：python3 xingbao.py --talk-only", file=sys.stderr)
        return 2

    # 清场检查：别的 ask.py / xingbao.py 还在跑的话，麦克风只有一个，
    # 用户听到的"星宝"很可能是那个不会动的脚本在回话。
    if not _check_voice_env(force=args.force):
        return 4
    if _take_lock() is None:
        print("另一个星宝已经在待命了（{} 被占）。先关掉那一个。".format(
            VOICE_LOCK), file=sys.stderr)
        return 4

    print("=" * 68)
    print("星宝启动：{}，默认测 {} 块，分流 {}，音量 {}".format(
        "对讲模式（不会动）" if args.talk_only else "检测模式",
        args.loops,
        "开" if args.route else "关",
        "默认" if args.volume is None else "{:.0f}".format(args.volume)))
    print("  唤醒：{}".format(
        "ASR 判定（VAD 断句 -> 识别 -> 同音字白名单），本地 {} 兜底，阈值 {}".format(
            _WAKE_ENGINE_LABEL.get(args.wake_engine, args.wake_engine),
            "默认 {:.2f}".format(kws_mod.DEFAULT_THRESHOLD) if args.wake_th is None
            else "{:.2f}".format(args.wake_th))
        if args.wake_fallback == "asr" else
        "只信本地 {}，阈值 {}（真人召回低，不推荐）".format(
            _WAKE_ENGINE_LABEL.get(args.wake_engine, args.wake_engine),
            "默认 {:.2f}".format(kws_mod.DEFAULT_THRESHOLD) if args.wake_th is None
            else "{:.2f}".format(args.wake_th))))
    print("  VAD ：底噪 x{:.2f}（下限 {:.0f}），唤醒断句静音 {:.0f} ms".format(
        mic_mod.VAD_FLOOR_MULT, mic_mod.VAD_FLOOR_MIN,
        wake_asr_mod.current_settings()["end_silence_ms"]))
    print("  日志：{}（只写文件，不进语音）".format(
        "不写" if args.no_log else args.log))
    print("  记忆：{}".format(MemoryStore(verbose=False).path))
    print("=" * 68)
    if LOG:
        LOG.log("启动 {} 块，分流={}，唤醒={}（ASR={}）阈值={}，VAD倍数={}",
                args.loops, "开" if args.route else "关", args.wake_engine,
                args.wake_fallback,
                args.wake_th if args.wake_th is not None
                else kws_mod.DEFAULT_THRESHOLD,
                mic_mod.VAD_FLOOR_MULT)

    problem = demo_cheese.preflight()
    if problem:
        print(problem, file=sys.stderr)
        return 3

    robot = arm = leg = nav = inspector = box = router = None
    assistant = None
    try:
        if args.talk_only:
            robot, box = _connect_audio(args.volume)
            memory = MemoryStore(verbose=True)
            if args.memory_reset:
                memory.reset()
            cfg = cloud.CloudConfig.from_file()
            assistant = Assistant(
                robot,
                voice=box,
                cloud_voice=cloud.CloudVoice(cfg),
                memory=memory,
                wake_engine=args.wake_engine,
                wake_threshold=args.wake_th,
                wake_asr_fallback=(args.wake_fallback == "asr"),
                wake_asr_cooldown=args.wake_asr_cooldown,
                vad_floor_mult=args.vad_mult,
            )
            core = VoiceCore(assistant, loops=args.loops, talk_only=True)
            core.run_forever()
            return 0

        robot, arm, leg, nav, inspector, box, router = demo_cheese.connect(
            demo_cheese.TOOL_NAME,
            vision_on=not args.no_vision,
            volume=args.volume,
            voice_on=True,
            route_on=args.route,
        )
        # 现场这一路音频盒是 demo_cheese.connect 给的，**不是** _connect_audio
        # 那个（那只在 --talk-only 里用）。2026-11-08 踩过：笑声只在 talk-only
        # 里预载，正式检测模式反而没装载，只能等第一次喊"笑一个"才现场加载。
        _preload_laugh(box)
        memory = MemoryStore(verbose=True)
        if args.memory_reset:
            memory.reset()

        cfg = cloud.CloudConfig.from_file()
        if not cfg.ready():
            print("没有 API key，语音问答会退化成本地兜底。先写 voice/.env："
                  "DASHSCOPE_API_KEY=sk-xxx", file=sys.stderr)

        assistant = Assistant(
            robot,
            voice=box,                       # 和播报共用一路音频输出
            cloud_voice=cloud.CloudVoice(cfg),
            context=_inspector_context(inspector),
            memory=memory,
            wake_engine=args.wake_engine,
            wake_threshold=args.wake_th,
            wake_asr_fallback=(args.wake_fallback == "asr"),
            wake_asr_cooldown=args.wake_asr_cooldown,
            vad_floor_mult=args.vad_mult,
        )
        core = VoiceCore(assistant, inspector=inspector, router=router,
                         loops=args.loops, route_on=args.route)
        core.robot, core.arm, core.leg, core.nav = robot, arm, leg, nav
        core.run_forever()
        return 0
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as exc:                            # noqa: BLE001
        print("星宝启动失败：{}".format(exc), file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if assistant is not None:
            try:
                assistant.close_mic()
            except Exception:                           # noqa: BLE001
                pass
        if inspector is not None:
            try:
                inspector.summary()
            except Exception:                           # noqa: BLE001
                pass
        if box is not None:
            try:
                box.close()
            except Exception:                           # noqa: BLE001
                pass
        if robot is not None:
            for name in ("request_shutdown", "wait_for_shutdown", "destroy"):
                try:
                    getattr(robot, name)()
                except Exception:                       # noqa: BLE001
                    pass
        if LOG:
            LOG.log("退出")
            LOG.close()


def _connect_audio(volume=None):
    """对讲模式专用：只初始化机器人音频，不碰 motion / 导航 / 相机。

    不动就绝不需要 ``--confirm-motion``，现场随便试。返回 ``(robot, voice_box)``。
    """
    from galbot_sdk.g1 import GalbotRobot

    from vision import config as vision_config
    from voice.play_clip import VoiceBox

    robot = GalbotRobot()
    if not robot.init():
        raise RuntimeError("robot.init() 失败（音频用不了）")
    time.sleep(2.0)
    box = VoiceBox(
        robot,
        volume=vision_config.VOICE_VOLUME if volume is None else float(volume),
    )
    missing = [n for n, fn in demo_cheese.VOICE_FILES
               if not box.load(n, Path(demo_cheese.__file__).resolve().parent
                               / "voice" / fn)]
    # 彩蛋：奶龙大笑。素材是现场那段 19.8 s 的 laugh.mp3，已经转成 16 kHz
    # 单声道 PCM 并包成 wav（voice/laugh.wav），和另外三支走同一条播放路径。
    _preload_laugh(box)
    if missing:
        print("  ⚠ 这些语音没加载成功：{}".format(missing))
    return robot, box


LAUGH_CLIP = "laugh"


def laugh_path():
    return Path(demo_cheese.__file__).resolve().parent / "voice" / "laugh.wav"


def load_laugh(box):
    """把笑声装进语音盒。没有素材就返回 False（不影响其它语音）。"""
    path = laugh_path()
    if not path.is_file():
        return False
    try:
        return box.load(LAUGH_CLIP, path)
    except Exception as exc:                              # noqa: BLE001
        print("  笑声加载失败：{}".format(exc))
        return False


def _preload_laugh(box):
    """启动时就把笑声装进去。

    两条音频入口都要调：``--talk-only`` 走 ``_connect_audio``，正式检测模式走
    ``demo_cheese.connect``。只在其中一条里装，另一条模式下喊"笑一个"就要现场
    加载（还可能因为没 ffmpeg 直接失败）。
    """
    if load_laugh(box):
        return True
    print("  ⚠ 笑声没加载上（voice/laugh.wav 缺失？），喊「笑一个」只会没声音")
    return False

def _inspector_context(inspector):
    """给 Brain 的"当前检测状态"补充。没有检查器就返回 None。"""
    if inspector is None:
        return None

    def read():
        try:
            results = getattr(inspector, "results", []) or []
        except Exception:                                # noqa: BLE001
            return ""
        if not results:
            return ""
        item = report_mod.summarize(results)
        return item.get("spoken", "")

    return read


if __name__ == "__main__":
    raise SystemExit(main())
