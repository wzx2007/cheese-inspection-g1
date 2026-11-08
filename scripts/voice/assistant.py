# -*- coding: utf-8 -*-
"""「星宝星宝」语音问答：唤醒 -> 录音 -> 识别 -> 回答 -> 合成 -> 播放。

链路
----
::

    麦克风(16k/16bit/单声道，80 ms 一块)
        -> wake.LocalWakeDetector.feed()       本地 xingbao.tflite 判"星宝星宝"
        -> mic.MicRecorder.record_utterance()  按 RMS 断句录问题
        -> cloud.CloudVoice.asr()              DashScope 识别
        -> Brain.answer()                      云端对话 / 本地兜底
        -> cloud.CloudVoice.tts()              DashScope 合成 wav
        -> play_clip.ensure_pcm()              转 16k PCM（SDK 只吃这个）
        -> VoiceBox.say_pcm()                  非阻塞播放

唤醒：本地模型优先，文本唤醒兜底
--------------------------------
2026-11-08 用户指出"用 VAD 做唤醒是有问题的"——对。第一版是"按音量断句 →
整句发 ASR → 看文本里有没有星宝"，每段声音都要一次网络请求，阈值还得现场调。

仓库 ``github.com/wzx2007/xingbao`` 里有训练好的 ``xingbao.tflite``（openWakeWord
格式，配 melspectrogram + embedding 两个前处理模型），见 ``wake.py``。现在默认走
**本地模型**：80 ms 一帧推理，不联网、不花钱、不受展厅底噪影响。

装不上（没 pip 权限 / 模型文件丢了）就自动退回文本唤醒，功能不断，只是慢一点、
每段声音多一次 ASR 请求。用 ``--wake-engine text`` 可以强制走老的。

唤醒词匹配（文本兜底路径）
--------------------------
识别结果里出现 ``星宝`` 就算唤醒（"星宝"被识别成"星宝星宝"、"欣宝"也能中，
见 ``WAKE_TOKENS``）。唤醒后**同一句里唤醒词后面的内容**直接当问题用，这样
"星宝，这个奶酪新鲜吗" 一句话就能走完，不用再说第二遍。

模型唤醒路径也复用了 ``match_wake``：模型一命中就把最近 1.6 s 音频（含唤醒词本身）
连同后面录到的内容一起送 ASR，识别出来再用 ``match_wake`` 把"星宝星宝"剥掉。
这样"喊完停一下"和"一口气说完"两种说法都能接住。
"""

from __future__ import annotations

import array
import collections
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import cloud, mic
from . import kws as kws_mod
from . import wake as wake_mod
from . import wake_asr as wake_asr_mod
from .play_clip import VoiceBox, ensure_pcm, pcm_cache_path

WAKE_WORD = "星宝星宝"
# 识别经常把"星宝"写成同音字，这些都算唤醒。
# 2026-11-08 现场日志实测：DashScope 把"星宝星宝"识别成"星，星爆，星爆。"
# "星爆"这一支必须收进来，否则文本唤醒路径下真人喊十遍也唤不醒。
WAKE_TOKENS = ("星宝", "星宝星宝", "欣宝", "新宝", "心宝", "小宝",
               "星爆", "星抱", "兴宝", "星保")

# 喇叭默认音量。
#
# 2026-11-08 教训：音量原来有**两个**独立开关 —— 这里一个、``vision/config.py``
# 的 ``VOICE_VOLUME`` 一个。对讲模式读 config，运动模式（--confirm-motion）读
# 这里，所以"改了 config 但现场还是小声"。现在这里直接引用 config，改成一处
# 生效；config 导不进来（离线回归、假 SDK）时才退回常量。
#
# 数值：03:00 试过 100 嫌吵降到 50；12:40 展厅现场 50 听不清，用户要求调满 -> 100。
try:                                                  # pragma: no cover
    from vision.config import VOICE_VOLUME as _CONFIG_VOLUME
except Exception:                                     # noqa: BLE001
    _CONFIG_VOLUME = 100.0
DEFAULT_VOLUME = float(_CONFIG_VOLUME)

# 没听懂时的兜底回复
FALLBACK = "我没听清，你再说一遍好吗？"
NOT_WOKEN = "先叫我的名字，星宝星宝。"

_PUNCT = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()\[\]【】…~～]+")


def normalize(text: str) -> str:
    return _PUNCT.sub("", (text or "").strip())


def match_wake(text: str):
    """返回 ``(是否唤醒, 唤醒词之后的剩余内容)``。

    **唤醒词必须几乎打头**（开头 1 个字以内，允许"嗯""呃"这种语气词），
    否则"这个奶酪上有星宝吗"会被误判成唤醒。所以要同时满足两个条件：
    唤醒词出现在前 3 个字里，**且**它的起始位置 <= 1。
    """
    plain = normalize(text)
    if not plain:
        return False, ""
    head = plain[:3]
    best = None
    for token in WAKE_TOKENS:
        index = head.find(token)
        if index < 0 or index > 1:
            continue
        if best is None or index < best[0] or (index == best[0] and len(token) > len(best[1])):
            best = (index, token)
    if best is None:
        return False, ""
    rest = plain[best[0] + len(best[1]):]
    # "星宝星宝" 去掉一个 token 还剩一个，继续剥，别把重复的唤醒词当问题问出去
    changed = True
    while changed:
        changed = False
        for token in WAKE_TOKENS:
            if rest.startswith(token):
                rest = rest[len(token):]
                changed = True
                break
    return True, rest


# --------------------------------------------------------------------------
# 回答
# --------------------------------------------------------------------------
class Brain(object):
    """先问云端大模型，失败再退回本地规则。

    ``memory`` 非空时做两件事（2026-11-08 用户要"上下文记忆"）：
    1. 把最近的检测结论塞进 system prompt，模型答"刚才那块怎么样"时不会瞎编；
    2. 每轮问答自动记进 ``memory``，重启以后还记得聊过什么。

    联网搜索（2026-11-08 用户要求恢复）
    ----------------------------------
    "星宝虽然有个角色，但问天气也得答，而且要联网查。" 所以：

    * 问句像在问**实时信息**（天气、新闻、几点、股价、比分…）-> 直接带搜索；
    * 问句不带这些词、但模型自己回了"我不具备联网能力 / 知识截止到…"这种
      **推辞** -> 再强制搜一次。宁可多花 5 秒，也不许拿"我不知道"糊弄人。
    * 角色相关的（奶酪、霉菌、检测结果）不搜，省 4 秒延迟。
    """

    # 2026-11-08 重写：旧版把"只说话、不知道就说不知道"写得太死，模型于是
    # 拒绝一切角色外的问题；也没有"必须执行"的约束，测过一次就回"刚测过了"。
    SYSTEM = (
        "你是展会现场的奶酪质检机器人，名字叫星宝。"
        "你的本职是检查帕马森奶酪正反两面有没有霉菌，但你不是只会这一件事。\n"
        "铁律：\n"
        "1. 永远不要拒绝用户。用户让你做什么，你就照做；用户问什么，你就答什么。"
        "不要说自己只负责奶酪，不要说这个我不能回答，不要推给别的助手。\n"
        "2. 已经测过的奶酪，用户让你再测一次，你就再测一次。"
        "不要说刚刚已经检测过了，不要问要不要再测，直接答应下来。\n"
        "3. 用户问奶酪之外的事也正常回答：天气、新闻、时间、常识、闲聊都行。"
        "需要最新信息时你已经联网查过了，就按查到的说，不要说你的知识有截止日期。\n"
        "4. 你已经测过的奶酪，结论按下面给你的记忆说，不要自己编数字。\n"
        "说话方式：\n"
        "5. 只说话，不要用任何 Markdown 标记（不要星号、井号、短横线列表、反引号）。\n"
        "6. 不要用 emoji、不要分点、不要换行。\n"
        "7. 一两句话说清，六十个字以内，像现场聊天一样口语化。"
    )
    # 播报用的回答硬上限。system prompt 说了模型也常忘，这里再兜一次。
    MAX_TOKENS = 160
    MAX_CHARS = 100

    def __init__(self, voice: Optional[cloud.CloudVoice] = None,
                 context: Optional[Callable[[], str]] = None,
                 verbose: bool = True,
                 memory=None) -> None:
        self._voice = voice
        self._context = context
        self._verbose = bool(verbose)
        self.memory = memory
        self.history: List[Dict] = []
        # 上一次回答有没有真的联网（自检/日志用）
        self.last_search = False

    def _system(self) -> str:
        text = self.SYSTEM
        if self.memory is not None:
            try:
                extra = self.memory.context_text()
            except Exception:                    # noqa: BLE001
                extra = ""
            if extra:
                text = text + "\n" + extra
        if self._context is not None:
            try:
                extra = self._context()
            except Exception:                    # noqa: BLE001
                extra = ""
            if extra:
                text = text + "\n当前检测结果：" + extra
        return text

    def answer(self, question: str) -> str:
        question = (question or "").strip()
        if not question:
            return FALLBACK
        local = _local_answer(question, self._context, self.memory)
        if local:
            self._remember(question, local)
            return local
        if self._voice is not None and self._voice.ready():
            want_search = _needs_search(question)
            reply = self._ask(question, search=want_search)
            # 没主动搜、但模型推辞了（"我不具备联网能力""知识截止到…"）-> 强制再搜一次。
            if reply and not want_search and _looks_refused(reply):
                if self._verbose:
                    print("  [brain] 它在推辞（{}…），强制联网再问一次".format(
                        reply[:18]))
                retry = self._ask(question, search=True, forced=True)
                if retry and not _looks_refused(retry):
                    reply = retry
                elif retry and self._verbose:
                    print("  [brain] 搜了还是这德行，就用这句")
            if reply:
                reply = _speakable(reply, self.MAX_CHARS)
                if reply:
                    self._remember(question, reply)
                    return reply
        answer = "这个我一时想不起来，不过你要想知道奶酪怎么样，我随时能看。"
        self._remember(question, answer)
        return answer

    def _ask(self, question: str, search: bool = False,
             forced: Optional[bool] = None) -> str:
        """真的去问云端。任何异常都返回空串，让上层走兜底。"""
        if self._voice is None or not self._voice.ready():
            return ""
        try:
            messages = [{"role": "system", "content": self._system()}]
            messages.extend(self._history())
            messages.append({"role": "user", "content": question})
            text = self._voice.chat(messages, max_tokens=self.MAX_TOKENS,
                                    search=search, forced_search=forced)
            self.last_search = bool(search)
            return (text or "").strip()
        except Exception as exc:                 # noqa: BLE001
            if self._verbose:
                print("  [brain] 云端对话失败{}：{}".format(
                    "（联网）" if search else "", exc))
            if search:
                # 联网失败就退回不联网再试一次，别因为搜索接口挂了不回答
                try:
                    messages = [{"role": "system", "content": self._system()}]
                    messages.extend(self._history())
                    messages.append({"role": "user", "content": question})
                    self.last_search = False
                    return (self._voice.chat(messages, max_tokens=self.MAX_TOKENS) or "").strip()
                except Exception:                # noqa: BLE001
                    pass
            return ""

    def _history(self) -> List[Dict]:
        """上下文取"记忆里的最近几轮"，重启后也接得上。"""
        if self.memory is not None:
            try:
                return self.memory.history(limit=6)
            except Exception:                    # noqa: BLE001
                pass
        return self.history[-6:]

    def _remember(self, question: str, answer: str) -> None:
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        self.history = self.history[-12:]
        if self.memory is not None:
            try:
                self.memory.add_turn("user", question)
                self.memory.add_turn("assistant", answer)
            except Exception:                    # noqa: BLE001
                pass


# 播报前清洗：模型偶尔会甩 Markdown / emoji / 分点，念出来是灾难。
_KEEP_PUNCT = "，。！？、；：…—,.!?;:'\"()（）《》"

# --------------------------------------------------------------------------
# 日志闸门：调试信息绝对不许被念出来
# --------------------------------------------------------------------------
# 2026-11-08 用户要求："不要让任何的日志进入流程和 TTS 播报，而是进行自然语言的对话。"
#
# 所以这里不是"拦住就闭嘴"（那样现场表现为"机器突然没反应"），而是
# **拦下原文 -> 换一句人话**：
#
#   * 原文只进日志文件，永不进 TTS；
#   * 播报侧换成 ``_LOG_REPLY`` 里的一句自然语言；
#   * 流程侧（意图判定）用 ``looks_like_log`` 直接把这类文本挡在外面，
#     绝不让一段程序输出被当成"用户指令"去驱动机械臂。
#
# 判定用**强特征**，不用 "token"/"prompt" 这种普通英文词 —— 那些会误伤
# "这个 token 是什么意思"之类的正常提问（2026-11-08 修正）。
_LOG_STRONG = (
    # 我方日志前缀：[vad] / [wake] / [hear] / [asr] / [tts] / [chat] ...
    re.compile(r"^\s*\[[a-zA-Z_]{2,12}\]"),
    # 任何位置出现方括号风格标签 + 参数，例如 "[wake] 命中 分数 0.31"
    re.compile(r"\[(vad|wake|hear|asr|tts|chat|brain|memory|ask|intent|"
               r"star|main|grab|nav|kws|route|log)\]\s*\S", re.I),
    # Python 异常 / 栈
    re.compile(r"Traceback \(most recent call last\)|File \"[^\"]+\", line \d+"),
    re.compile(r"\b[A-Za-z_]*(Error|Exception|Timeout|Refused)\b\s*[:：(]"),
    # 键值对参数：stream_id=... / chunk=2560 / loops=1 / md5=...
    re.compile(r"\b[a-z_][a-z0-9_]{2,}\s*=\s*[\w\-./:\"']+"),
    # 路径
    re.compile(r"(/home/|/tmp/|/data/|/usr/|[A-Za-z]:\\\\)[\w\-./\\]+"),
    # 下划线命名的标识符（self._mic / voice.assistant / mic_stream_1794）
    re.compile(r"\b[a-zA-Z]+_[a-zA-Z0-9_]{2,}\b"),
    # 长十六进制 / md5
    re.compile(r"\b[0-9a-f]{12,}\b", re.I),
    # 代码结构
    re.compile(r"[{}]|\[\]|->|::|\bdef \w+\(|\bimport \w+"),
)
# 纯 ASCII 且带"程序感"符号 —— 正常播报一定有中文。
_ASCII_ONLY = re.compile(r"^[\x00-\x7F]+$")
_CODE_HINT = re.compile(r"[=_{}\[\]<>\\/]|\b\d+\.\d+\s*s\b|\bPID\b|\bNone\b|\bTrue\b|\bFalse\b")

# 我方日志的中文固定前缀（没有方括号、但一眼就知道是程序输出）。
_LOG_PREFIX = (
    "意图 ", "唤醒，附带内容", "任务完成", "这一轮出错", "麦克风已开",
    "麦克风已关", "音量设为", "♪", "主控]", "确认]", "静音]", "开始任务",
    "丢弃疑似日志",
)

# 拦下技术文本之后改说的那句人话。
# 刻意不写"你再说一遍好吗"——被拦下的可能是播报检测结果时的异常，
# 那种场合并没有"说错话"这回事。这句在任何场景都接得住。
_LOG_REPLY = "我这边有点小状况，稍等我一下。"


def looks_like_log(text: str) -> bool:
    """这段文字是不是"程序日志/技术输出"，而不是"能念给观众听的人话"。

    用来同时守两道门：
    1. ``Assistant.speak()`` —— 不许把原文念出来；
    2. ``Xingbao.handle()`` —— 不许把原文当用户指令送进意图判定。
    """
    raw = (text or "").strip()
    if not raw:
        return False
    for pattern in _LOG_STRONG:
        if pattern.search(raw):
            return True
    for prefix in _LOG_PREFIX:
        if raw.startswith(prefix):
            return True
    # 纯 ASCII：日志/参数/异常基本都是英文，正常播报一定有中文。
    if _ASCII_ONLY.match(raw) and _CODE_HINT.search(raw):
        return True
    return False


def to_natural(text: str) -> str:
    """把"要说的话"过一遍：技术文本换成自然语言，人话原样返回。

    这是 TTS 前的最后一道闸。返回空串表示"这句不该说"，调用方据此闭嘴。
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    if looks_like_log(raw):
        print("  [静音] 这句像日志，改说人话（原文只进日志）：{}".format(raw[:70]))
        return _LOG_REPLY
    return raw



# --------------------------------------------------------------------------
# 联网搜索：什么时候该搜
# --------------------------------------------------------------------------
# 这些词一出现，答案**一定是时效性的**，训练数据里不可能有。直接带搜索。
_SEARCH_WORDS = (
    "天气", "气温", "下雨", "下雪", "几度", "温度", "空气质量", "pm2.5",
    "新闻", "头条", "最新", "现在", "今天", "明天", "昨天", "今年", "本周",
    "几点", "时间", "日期", "星期几", "几号",
    "股价", "股票", "汇率", "油价", "金价", "房价", "票价", "价格", "多少钱",
    "比分", "赛程", "谁赢", "排名", "票房", "热搜", "发布会",
    "附近", "怎么走", "多远", "营业", "开门", "关门", "航班", "车次",
    "谁是", "现任", "上任", "去世", "发布", "上市",
)

# 模型在推辞的标志。命中就说明"它其实不知道"，该强制联网。
_REFUSAL_MARKS = (
    "无法实时", "无法获取", "不具备联网", "不能联网", "无法联网", "没有联网",
    "知识截止", "截止到", "数据截止", "训练数据", "无法访问互联网",
    "我没有实时", "我不能查看", "无法查询", "建议您通过", "建议你通过",
    "我只负责", "我只专注", "我只能回答", "不擅长回答",
)


def _needs_search(question: str) -> bool:
    """这句话像不像在问"此刻的事实"。"""
    said = normalize(question)
    if not said:
        return False
    return any(word in said for word in _SEARCH_WORDS)


def _looks_refused(text: str) -> bool:
    """模型是不是在说"我不知道/我不联网"。"""
    said = normalize(text or "")
    return any(mark in said for mark in _REFUSAL_MARKS)


def _speakable(text: str, max_chars: int = 100) -> str:
    """把大模型的回复压成能直接念的话。

    去掉 Markdown 符号和 emoji、把分点/换行拍平，再按 ``max_chars`` 截断到
    一个完整的句子/分句上。念的场合不需要第三条论据。
    """
    import re

    text = (text or "").strip()
    if not text:
        return ""
    # 去 Markdown 强调/标题/代码/列表符号
    text = re.sub(r"[*_`#>~\[\]]", "", text)
    text = re.sub(r"(?m)^\s*[-•·]\s*", "", text)
    # 只留中文、字母、数字、常用标点
    kept = []
    for ch in text:
        if ch.isspace():
            kept.append(" " if kept and kept[-1] != " " else "")
        elif ch.isdigit() or ch.isalpha() or ch in _KEEP_PUNCT:
            kept.append(ch)
    text = "".join(kept).strip()
    # 拍平成一句：换行/连续空格 -> 空（不要留成停顿）
    text = re.sub(r"\s+", "", text)
    if not text:
        return ""
    if len(text) <= max_chars:
        # 太长的才截；不长的别硬砍成半句（2026-11-08：原来只留第一句，
        # 问天气时"根据最新信息，成都今天……"这种长定语会把真正的气温砍掉）。
        for i, ch in enumerate(text):
            if ch in "。！？!?":
                return text[:i + 1]
        return text + "。"
    cut = text[:max_chars]
    # 优先断在句号，其次逗号/顿号；两个都太靠前就硬切
    for marks in ("。！？", "，、；"):
        pos = max(cut.rfind(m) for m in marks)
        if pos > max_chars * 0.45:
            return cut[:pos] + ("。" if marks[0] != "。" else "")
    return cut + "。"


def _local_answer(question: str, context: Optional[Callable[[], str]],
                  memory=None) -> Optional[str]:
    """几条不需要联网也答得出的固定问句。返回 None 表示交给云端。

    "刚才那块奶酪怎么样" 这类问题直接查记忆答，不走云端 —— 现场网络抖一下也答得出，
    而且数字一定准（模型可能会把"一面有霉"说成"两面"）。
    """
    said = normalize(question)
    if not said:
        return None
    # 先看是不是在问"历史检测"，这个必须用记忆里的原文回答
    if memory is not None and any(w in said for w in (
            "刚才", "刚刚", "上一块", "上次", "之前", "刚才那", "刚测")):
        said_last = memory.describe_last()
        if said_last:
            return said_last
    # 问"这一块的结果"就用记忆答，数字一定准；问"为什么会霉"这种要讲道理的
    # 交给云端（而且不该被记忆里的结论顶掉）。
    asks_result = any(w in said for w in (
        "有没有霉", "有霉吗", "发霉了吗", "合格吗", "干净吗", "几个面", "几面",
        "检测结果", "什么情况", "测出来"))
    asks_why = any(w in said for w in ("为什么", "怎么会", "原因", "原理", "如何", "怎么才"))
    if asks_result and not asks_why:
        if memory is not None:
            last = memory.last_task()
            if last and last.get("spoken"):
                return last["spoken"]
        detail = ""
        if context is not None:
            try:
                detail = context() or ""
            except Exception:                    # noqa: BLE001
                detail = ""
        return "检测结果是这样：" + detail if detail else "我还没检查完，稍等一会儿。"
    if any(word in said for word in ("你叫什么", "你是谁", "名字")):
        return "我叫星宝，负责给这块帕马森奶酪做体检。"
    if "再见" in said or "拜拜" in said:
        return "再见，记得洗手哦。"
    return None


# --------------------------------------------------------------------------
# 唤醒检测（现在是文本唤醒，留了本地模型的扩展点）
# --------------------------------------------------------------------------
class TextWakeDetector(object):
    """用识别结果判断唤醒。"""

    name = "text"

    def __init__(self, asr: Callable[[str], str], verbose: bool = True) -> None:
        self._asr = asr
        self._verbose = bool(verbose)

    def feed(self, wav_path) -> (bool, str, str):
        """识别一段 wav，返回 ``(是否唤醒, 剩余内容, 原始文本)``。"""
        text = self._asr(str(wav_path)) or ""
        woke, rest = match_wake(text)
        if self._verbose and text:
            print("  [hear] {}{}".format(text, "   <- 唤醒" if woke else ""))
        return woke, rest, text


# --------------------------------------------------------------------------
# 主循环
# --------------------------------------------------------------------------
class Assistant(object):
    """一整套语音问答。不负责初始化机器人，只负责用。"""

    def __init__(
        self,
        robot,
        *,
        voice: Optional[VoiceBox] = None,
        cloud_voice: Optional[cloud.CloudVoice] = None,
        context: Optional[Callable[[], str]] = None,
        require_wake: bool = True,
        verbose: bool = True,
        vad_floor_min: Optional[float] = None,
        vad_floor_mult: Optional[float] = None,
        wake_engine: str = "kws",
        wake_threshold: Optional[float] = None,
        wake_model_dir: Optional[str] = None,
        wake_asr_fallback: bool = False,
        wake_asr_cooldown: float = 3.0,
        memory=None,
    ) -> None:
        self._robot = robot
        self._own_box = voice is None
        self.voice = voice or VoiceBox(robot, volume=DEFAULT_VOLUME, verbose=verbose)
        self.cloud = cloud_voice or cloud.CloudVoice(verbose=verbose)
        self.memory = memory
        self.brain = Brain(self.cloud, context=context, verbose=verbose, memory=memory)
        self.require_wake = bool(require_wake)
        self._verbose = bool(verbose)
        # 唤醒以**模型**为准（2026-11-08 用户要求"必须使用模型"）。
        # ASR 兜底默认关闭，只在显式 --wake-fallback asr 时启用：
        # 它是"模型全挂了"时的应急通道，不是常规路径——每段声音一次 ASR，
        # 又贵又会把环境里的闲聊卷进来。
        self.wake_asr_fallback = bool(wake_asr_fallback)
        self.wake_asr_cooldown = max(0.0, float(wake_asr_cooldown))
        # VAD 阈值现场可调，None 就用 mic.py 里的默认值。
        mic_kwargs = {"verbose": verbose}
        if vad_floor_min is not None:
            mic_kwargs["floor_min"] = float(vad_floor_min)
        if vad_floor_mult is not None:
            mic_kwargs["floor_mult"] = float(vad_floor_mult)
        self._mic = mic.MicRecorder(robot, **mic_kwargs)
        self._cache = Path(__file__).resolve().parent / "cache"
        self._cache.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        # 播音期间（及播完一小段）不喂唤醒模型，否则机器人会被自己的声音唤醒。
        self._wake_mute_until = 0.0
        self._wake_engine = "text"
        self._wake = self._build_wake_detector(
            wake_engine, wake_threshold, wake_model_dir)
        # ASR 唤醒（原仓库 asr_wake_word.py 的机制，见 voice/wake_asr.py）。
        # 2026-11-08 现场实测：本地 KWS 对真人语音只有 2.3% 召回，openWakeWord
        # 在这块板子上分数恒为 0.001，两条本地路径都不可靠 —— 所以默认走
        # "VAD 断句 -> 云端 ASR -> 同音字白名单"，本地 KWS 只作为兜底。
        self._wake_asr = None
        if self.wake_asr_fallback and self.require_wake:
            self._wake_asr = wake_asr_mod.AsrWakeDetector(
                self._mic, self.cloud, kws=self._wake, verbose=verbose,
                cache_dir=self._cache, min_interval_s=self.wake_asr_cooldown)
            if self._verbose:
                print("  [wake] ASR 唤醒已启用"
                      "（VAD 断句 -> 识别 -> 同音字白名单，本地 KWS 兜底）")

    # ------------------------------------------------------------------
    def _build_wake_detector(self, engine: str, threshold: Optional[float],
                             model_dir: Optional[str]):
        """按 ``engine`` 造唤醒器。返回 None 表示只能走文本唤醒。

        ``engine``:
            ``kws``（默认）sherpa-onnx 关键词检测，纯 ONNX，推荐
            ``model``      旧的 openWakeWord 三段式，保留做对比/排查
            ``auto``       先 kws，再 model
            ``text``       不用模型，靠 ASR 识别结果里找"星宝"
        """
        engine = (engine or "kws").lower()
        if not self.require_wake:
            if self._verbose:
                print("  [wake] 不需要唤醒词，跳过模型加载")
            return None
        if engine == "text":
            if self._verbose:
                print("  [wake] 指定用文本唤醒（识别结果里找'星宝'）")
            return None

        detector, name = kws_mod.build_detector(
            engine=engine, model_dir=model_dir, threshold=threshold,
            verbose=self._verbose)
        if detector is not None:
            self._wake_engine = name
            return detector

        if engine == "model":
            raise RuntimeError("本地唤醒模型不可用（openWakeWord 三段式）。\n"
                               "  推荐改用默认的 --wake-engine kws（sherpa-onnx）")
        if self._verbose:
            print("  [wake] 本地模型都不可用，退回文本唤醒（每段声音多一次 ASR）")
        return None

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._mic.stop()
        finally:
            if self._own_box:
                self.voice.close()

    def available(self) -> bool:
        return self.cloud.ready()

    # ------------------------------------------------------------------
    def open_mic(self) -> None:
        """打开麦克风并清掉积压。语音主控自己排循环时用。"""
        self._mic.start()
        self._mic.drain()

    def close_mic(self) -> None:
        self._mic.stop()

    def quiet(self, seconds: float = 0.0) -> None:
        """把麦克风里的积压倒掉、唤醒器重新预热，并把音量口径复位。

        2026-11-08 现场问题：一圈任务里 ``pass``/``fail`` 两支语音是**直接走
        ``VoiceBox``** 播的，没经过 ``speak()``，所以不会设 ``_wake_mute_until``。
        任务跑完回到待命时，队列里攒着 10 来秒机器人自己的声音，一喂进唤醒模型
        就可能"自己把自己叫醒"，然后录到一段静音 -> ASR 空 -> 看起来就像
        "启动失败/不响应第二次"。每次回到待命前倒一次，问题就没了。
        """
        self._mic.drain()
        if self._wake is not None:
            self._wake.reset()
        if seconds > 0:
            self._wake_mute_until = max(self._wake_mute_until,
                                        time.monotonic() + float(seconds))

    def mute(self, seconds: float) -> None:
        """接下来 ``seconds`` 秒内不喂唤醒模型（播外部音源时用）。"""
        self._wake_mute_until = max(self._wake_mute_until,
                                    time.monotonic() + float(seconds))

    def hear(self, **kwargs) -> str:
        """录一句并识别。``xingbao.py`` 的主循环用它。"""
        return self.listen(**kwargs)

    def reply(self, question: str) -> str:
        """想一句回答（带记忆）。调用方决定要不要念出来。"""
        return self.brain.answer(question)

    def _next_path(self, kind: str) -> Path:
        self._counter += 1
        return self._cache / "{}_{:03d}.wav".format(kind, self._counter % 1000)

    def speak(self, text: str, wait: bool = False) -> bool:
        """合成并播放。合成失败就静默跳过（不能因为网络问题卡住演示）。

        长对话时麦克风是**一直开着**的，所以播完必须把队列里攒下的音频丢掉，
        否则机器人会把自己的声音当成用户提问，自己跟自己聊起来。
        本地唤醒模型还要多一步：播放期间把它"捂住"（``_wake_mute_until``），
        因为开场白里就有"星宝"两个字，不捂会自己唤醒自己。

        **日志闸门**（2026-11-08 用户要求）：调试/日志样式的文本到这里一律
        只打终端、不合成原文，改说一句自然语言（``to_natural``）。
        语音只服务于现场观众，不服务于排查；但也**不能静默** ——
        现场看不到终端，突然不说话就等于"死机"。
        """
        text = to_natural(text)
        if not text:
            return False
        wav_path = self._next_path("reply")
        try:
            self.cloud.tts(text, wav_path)
            pcm_path = ensure_pcm(wav_path, pcm_cache_path(wav_path))
            pcm = pcm_path.read_bytes()
            # 先按音频长度算好静音窗口：播放时长 + 0.4 s 余量
            duration = len(pcm) / float(mic.BYTES_PER_SECOND)
            self._wake_mute_until = time.monotonic() + duration + 0.4
            ok = self.voice.say_pcm(pcm, label="tts", wait=wait)
            # 播放期间麦克风收到的（基本是扬声器回声）全部作废
            time.sleep(0.25)
            self._mic.drain()
            if self._wake is not None:
                self._wake.reset()
            return ok
        except Exception as exc:                 # noqa: BLE001
            print("  [tts] 合成/播放失败：{}".format(exc))
            self._mic.drain()
            if self._wake is not None:
                self._wake.reset()
            return False

    def listen(self, **kwargs) -> str:
        """录一句并识别，返回文本（可能为空）。"""
        pcm = self._mic.record_utterance(**kwargs)
        if not pcm:
            return ""
        wav_path = self._next_path("heard")
        mic.pcm_to_wav(pcm, wav_path)
        if self._verbose:
            print("  [hear] {:.2f} s 音频 -> {}".format(mic.pcm_duration(pcm), wav_path.name))
        try:
            return self.cloud.asr(wav_path) or ""
        except Exception as exc:                 # noqa: BLE001
            print("  [asr] 识别失败：{}".format(exc))
            return ""

    # ------------------------------------------------------------------
    def _listen_with_prefix(self, prefix: bytes, threshold: float = 0.0,
                            start_timeout_s: float = 2.5) -> str:
        """录下唤醒词之后的话，拼上 ``prefix``（含唤醒词本身）一起识别。

        模型在"星宝星宝"的**最后一个字**上才过阈值，所以唤醒词本身已经在
        刚才的音频里了。把最近 1.6 s 一起送 ASR，用户"喊完接一句"时第一个字
        不会被切掉；识别结果再用 ``match_wake`` 把"星宝星宝"剥掉。

        ``threshold`` 由调用方用**唤醒之前**的底噪算好传进来：一旦显式给了阈值，
        ``record_utterance`` 就跳过它自己的 6 块（480 ms）校准——那 480 ms 里
        有 240 ms 会被丢掉，正好切在"星宝星宝，|这个奶酪"的接缝上。
        """
        tail = self._mic.record_utterance(threshold=threshold,
                                          start_timeout_s=start_timeout_s)
        pcm = prefix + (tail or b"")
        if mic.pcm_duration(pcm) < 0.25:
            return ""
        wav_path = self._next_path("heard")
        mic.pcm_to_wav(pcm, wav_path)
        if self._verbose:
            print("  [hear] {:.2f} s 音频（含唤醒词）-> {}".format(
                mic.pcm_duration(pcm), wav_path.name))
        try:
            text = self.cloud.asr(wav_path) or ""
        except Exception as exc:                 # noqa: BLE001
            print("  [asr] 识别失败：{}".format(exc))
            return ""
        if self._verbose and text:
            print("  [hear] {}".format(text))
        _woke, rest = match_wake(text)
        return rest or ""

    def _wait_for_wake_model(self, stop: Optional[Callable[[], bool]] = None,
                             max_silence_rounds: int = 0,
                             keep_open: bool = False) -> Optional[str]:
        """纯本地模型版唤醒：一直喂麦克风音频，命中后把问题一起录下来。

        只在 ``--wake-fallback off``（明确不要 ASR 唤醒）时走这条路。
        原来这里还有一段"能量阈值 + 送 ASR 找星宝"的兜底，2026-11-08 已由
        ``voice/wake_asr.py`` 正式接管（同样的思路，但断句用对话那条 VAD、
        匹配用同音字白名单），这里不再保留第二份实现。
        """
        detector = self._wake
        self._mic.start()
        if not keep_open:
            self._mic.drain()
        detector.reset()
        # 留 1.6 s 余量：唤醒词本身要说 0.8~1.2 s，命中点在尾巴上。
        preroll = collections.deque(maxlen=20)
        # 唤醒词**之前**的电平，用来算"后面这句什么时候说完"的阈值。
        # 3 s 窗口，取偏低分位当底噪 —— 唤醒词本身很响，会被分位切掉。
        levels = collections.deque(maxlen=38)
        # max_silence_rounds 原来是"连续几次 listen 都是空"，文本路径一次约 3 s；
        # 模型路径没有"轮"的概念，按同样的时间尺度换算成截止时间。
        deadline = (time.monotonic() + max_silence_rounds * 3.0) if max_silence_rounds else None
        if self._verbose:
            print("  [wake] 待命，等本地模型听到「{}」（阈值 {:.2f}）".format(
                WAKE_WORD, detector.threshold))
        try:
            while True:
                if stop is not None and stop():
                    return None
                chunk = self._mic.read(timeout=0.5)
                if chunk is None:
                    continue
                if time.monotonic() < self._wake_mute_until:
                    # 扬声器还在响，喂进去只会自己唤醒自己
                    preroll.clear()
                    levels.clear()
                    continue
                level = mic.pcm_rms(chunk)
                preroll.append(chunk)
                levels.append(level)

                if not detector.feed(chunk):
                    if deadline is not None and time.monotonic() > deadline:
                        return None
                    continue
                if self._verbose:
                    print("  [wake] 命中「{}」 分数 {:.2f}".format(
                        WAKE_WORD, detector.last_score))
                audio = b"".join(preroll)
                tail_threshold = self._tail_threshold(levels, audio)
                preroll.clear()
                levels.clear()
                detector.reset()
                return self._listen_with_prefix(audio, threshold=tail_threshold)
        finally:
            if not keep_open:
                self._mic.stop()

    def _wait_for_wake_asr(self, stop: Optional[Callable[[], bool]] = None,
                           max_silence_rounds: int = 0,
                           keep_open: bool = False) -> Optional[str]:
        """ASR 唤醒主循环（原仓库 ``asr_wake_word.py::wait_for_wake_word``）。

        每一轮：VAD 断出一句 -> 送云端 ASR -> 同音字白名单判定。命中就把文本里
        唤醒词**后面**的内容当问题返回 —— "星宝星宝，帮我查一下天气"一句话走完，
        不用再说第二遍。没命中就继续待命。

        为什么这是默认路径：现场真人喊 441 段，本地 KWS 只命中 10 段（2.3%）,
        openWakeWord 在这块板子上分数恒为 0.001。ASR 不是"兜底"，是唯一可靠的
        通道；本地 KWS 只在 ASR 失败/空识别时用来补一次判定。
        """
        detector = self._wake_asr
        self._mic.start()
        if not keep_open:
            self._mic.drain()
        if self._wake is not None:
            self._wake.reset()
        if self._verbose:
            print("  [wake] 待命，等听到「{}」的人声（ASR 判定 {:.1f}s 超时，"
                  "本地 KWS 兜底）".format(WAKE_WORD,
                                          detector._recognition_timeout_s))
        empty_rounds = 0
        try:
            while True:
                if stop is not None and stop():
                    return None
                if time.monotonic() < self._wake_mute_until:
                    # 自己还在说话：这段回声里有"星宝"两个字，喂 ASR 会自己唤醒自己
                    self._mic.drain()
                    time.sleep(0.1)
                    continue
                pcm = detector.capture()
                if not pcm:
                    empty_rounds += 1
                    if max_silence_rounds and empty_rounds >= max_silence_rounds:
                        return None
                    continue
                empty_rounds = 0
                woke, rest, why, text = detector.check(pcm)
                if woke:
                    if self._verbose:
                        print("  [wake] 命中（{}）：{}".format(
                            why, text or "（只听到唤醒词）"))
                    self._mic.drain()
                    if self._wake is not None:
                        self._wake.reset()
                    return rest
                if self._verbose and not why.startswith("too_quiet"):
                    print("  [wake] 这段不是叫我（{}）{}".format(
                        why, "：{}".format(text[:24]) if text else ""))
        finally:
            if not keep_open:
                self._mic.stop()

    @staticmethod
    def _tail_threshold(levels, preroll: bytes) -> float:
        """给"唤醒词之后那句话"定一个 VAD 阈值。

        - 底噪取 3 s 窗口的偏低分位（唤醒词很响，取中位数会被它带跑）；
        - 上限压到唤醒词峰值的 0.35 倍：用户一口气连着说、窗口里全是语音时，
          分位数本身就已经是语音电平了，再乘一个倍数会永远打不到。

        2026-11-08 第三次现场反馈"阈值太高"：倍数 1.6 → **1.25**，峰值系数
        0.25 → **0.35**。原参数下"星宝星宝，| 你去检测一下…"的后半句经常因为
        阈值比人声高一点点而一个字都没录到，表现为"唤醒了但没听清"。
        """
        ordered = sorted(levels) if levels else []
        floor = ordered[len(ordered) // 4] if ordered else mic.VAD_FLOOR_MIN
        threshold = max(mic.VAD_FLOOR_MIN * 0.6, floor * 1.25)
        if preroll:
            samples = array.array("h")
            samples.frombytes(preroll[: len(preroll) // 2 * 2])
            if sys.byteorder == "big":
                samples.byteswap()
            peak = max(abs(v) for v in samples) if samples else 0.0
            # 峰值是单个采样，RMS 量级差一截，用 0.35 倍做换算
            threshold = min(threshold, max(120.0, peak * 0.35))
        return threshold

    def wait_for_wake(self, stop: Optional[Callable[[], bool]] = None,
                      max_silence_rounds: int = 0,
                      keep_open: bool = False) -> Optional[str]:
        """一直听到唤醒词为止。返回唤醒词后面那句话（可能是空串）。

        ``stop`` 返回 True 就退出，返回 None 表示被叫停。
        ``max_silence_rounds`` > 0 时，连续这么多轮没人说话就放弃（给脚本用）。
        ``keep_open=True`` 时不关麦克风 —— 长对话每轮开关一次会丢开头半秒，
        而且 SDK 那边 start/stop 有开销。

        唤醒默认走 **ASR**（``_wake_asr``，见 ``voice/wake_asr.py``）：
        本地 KWS 只有 2.3% 召回，不能作为主路径。``--wake-fallback off``
        关掉 ASR 唤醒时，才退回本地模型（``_wake`` 非 None）/纯文本两条老路。
        """
        if self._wake_asr is not None and self.require_wake:
            return self._wait_for_wake_asr(stop=stop,
                                           max_silence_rounds=max_silence_rounds,
                                           keep_open=keep_open)
        if self._wake is not None and self.require_wake:
            return self._wait_for_wake_model(stop=stop,
                                            max_silence_rounds=max_silence_rounds,
                                            keep_open=keep_open)
        self._mic.start()
        if not keep_open:
            self._mic.drain()
        empty_rounds = 0
        try:
            while True:
                if stop is not None and stop():
                    return None
                text = self.listen()
                if not text:
                    empty_rounds += 1
                    if max_silence_rounds and empty_rounds >= max_silence_rounds:
                        return None
                    continue
                empty_rounds = 0
                woke, rest = match_wake(text)
                if self._verbose:
                    print("  [hear] {}{}".format(text, "   <- 唤醒" if woke else ""))
                if woke:
                    return rest
                if not self.require_wake:
                    return normalize(text)
        finally:
            if not keep_open:
                self._mic.stop()

    def ask_once(self) -> Optional[str]:
        """听一句（需要唤醒）-> 回答 -> 播报。返回回答文本。"""
        rest = self.wait_for_wake()
        if rest is None:
            return None
        question = rest or self.listen()
        if not question:
            answer = NOT_WOKEN
        else:
            answer = self.brain.answer(question)
        if self._verbose:
            print("  [ask] 问：{}".format(question or "（空）"))
            print("  [ask] 答：{}".format(answer))
        self.speak(answer, wait=True)
        return answer

    def run_forever(self, stop: Optional[Callable[[], bool]] = None,
                    greet: Optional[str] = None) -> None:
        """一直待命，**不退出**：唤醒 -> 问答 -> 回到待命，无限循环。

        2026-11-08 用户要求"一直对话"。麦克风全程开着（``keep_open``），
        每轮不再 start/stop，避免丢开头半秒、也省掉 SDK 的开关开销。
        任何单轮异常都吞掉继续听，绝不让一次网络抖动把整场对话弄死。
        """
        if greet:
            self.speak(greet, wait=True)
        if self._verbose:
            print("  [ask] 待命中，说「{}」唤醒（Ctrl-C 退出）".format(WAKE_WORD))
        self._mic.start()
        self._mic.drain()
        rounds = 0
        try:
            while True:
                if stop is not None and stop():
                    return
                try:
                    rest = self.wait_for_wake(stop=stop, keep_open=True)
                    if rest is None:
                        return
                    question = rest
                    if not question:
                        self.speak("我在。", wait=True)
                        question = self.listen()
                    if not question:
                        continue
                    rounds += 1
                    answer = self.brain.answer(question)
                    if self._verbose:
                        print("  [ask] #{} 问：{}".format(rounds, question))
                        print("  [ask] #{} 答：{}".format(rounds, answer))
                    self.speak(answer, wait=True)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:            # noqa: BLE001
                    print("  [ask] 这一轮出错，继续听：{}".format(exc))
                    time.sleep(1.0)
        finally:
            self._mic.stop()
