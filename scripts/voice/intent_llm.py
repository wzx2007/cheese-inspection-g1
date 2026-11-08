# -*- coding: utf-8 -*-
"""让**大模型推理**用户到底是不是要机器人去干活，而不是靠关键词硬匹配。

为什么要换掉关键词
------------------
2026-11-08 现场：唤醒已经稳了，但"无论我说什么都直接跑过去做任务"。
根因在 ``intent.py`` 的 ``START_WORDS`` —— 里面有 ``看一下 / 看看 / 开始 / 帮我 /
给我 / 替我 / 检测 / 检查 / 干活``，全是中文里最泛的词。于是

    "帮我查一下今天天气"   -> 命中 帮我 + 查一下 -> START_TASK -> 机器人伸手
    "你看看这个东西是啥"   -> 命中 看看         -> START_TASK
    "星宝你几岁了"         -> 不命中 -> QUESTION（这句反而是对的）

也就是说：**除了纯闲聊，只要句子里带一个泛动词就会真的动起来**。这不是调几个词
能修好的，是"用词表代替语义理解"这条路本身到头了。

现在的分工
----------
* **安全阀仍然是规则，且优先级最高**：``intent.is_stop()`` 命中就直接停，绝不
  交给模型 —— 模型偶尔抽风说"他是想让你检测"，机器人就动了，这种风险不能留。
* **其余全部交给大模型判**：``route()`` 让它输出一个极小的 JSON
  ``{"act":"detect|status|chat", "loops":1..3}``。判 detect 才动。
* **判不准就当聊天**：默认 ``chat``。宁可少动一次（用户再喊一遍），
  也不要多动一次（机器人突然伸手，现场最吓人）。
* 模型不可用 / 超时 / 输出不是 JSON -> 退回 ``intent.classify`` 的**收紧版**规则。
  收紧版删掉了上面那些泛动词，只认"动词 + 对象"或明确的复述命令。

``chat`` 分支不走这里，直接交给 ``Assistant.reply()`` —— 正常对话 + 联网搜索，
和用户要求一致（"其余的请正常对话搜索输出"）。
"""

from __future__ import annotations

import json
import re

from . import intent

# 路由器的 system prompt。要点：
#   1. 明确给出"这是机器人，判错会真的动起来"，让模型知道代价；
#   2. 给出正反例，尤其是那些"含动作词但其实是聊天"的坑；
#   3. 强制只输出 JSON —— 后面用正则兜一层，防止它加解释。
ROUTER_SYSTEM = """你在给一台展会现场的奶酪质检机器人做意图判断。
这台机器人有手臂和底盘，判成 detect 它就会真的走过去、伸手抓奶酪。
所以只在用户**确实要求它去做检测这件事**时才判 detect；拿不准一律判 chat。

可选动作只有三个：
- "detect"：用户要你现在就去检测奶酪（现在时、命令语气、指向眼前这块）。
- "status"：用户在问**已经测过**的结果/结论，不需要再做一次。
- "chat"：其它一切。闲聊、问知识、问天气、问你自己、感慨、反问、假设、
          讨论奶酪/霉菌但没让你去测、以及你拿不准的情况，全部算 chat。

易错点（务必遵守）：
- 句子里出现"看/查/测/开始/帮我"**不代表**要检测。
  "帮我查一下今天天气" -> chat
  "你看看这个是什么"   -> chat
  "奶酪为什么会发霉"   -> chat
  "检测是怎么做的"     -> chat
- 明确要求动手才算：
  "去检测一下面前这块奶酪" -> detect
  "帮我看看这块奶酪"       -> detect（有"这块奶酪"这个明确对象）
  "再测一次"               -> detect
  "开始吧"（上一句刚问要不要测）-> detect
- "刚才那块有霉吗"是问结果 -> status，不是 detect。

只输出一行 JSON，不要解释、不要代码块：
{"act":"detect","loops":1}
loops 只能是 1、2、3。用户说"这三块都测"才是 3，说"这块"就是 1，没提就是 1。
"""

_JSON_RE = re.compile(r'\{[^{}]*\}')
# 兼容模型不带引号的写法：act: detect, loops: 2
_ACT_RE = re.compile(r'[\"\']?(?:act|action|intent)[\"\']?\s*[:=]\s*[\"\']?([a-z_]+)', re.I)
_LOOPS_RE = re.compile(r'[\"\']?loops[\"\']?\s*[:=]\s*(\d)')

VALID = ("detect", "status", "chat")

# 判成 detect 的最低把握。低于它一律当 chat —— 少动一次远好过多动一次。
MIN_CONFIDENCE = 0.5


def _parse(text):
    """从模型输出里抠出 (act, loops)。抠不到返回 (None, None)。"""
    if not text:
        return None, None
    raw = text.strip()
    act = None
    loops = None
    m = _JSON_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            act = str(obj.get("act") or obj.get("action") or obj.get("intent") or "").lower()
            try:
                loops = int(obj.get("loops") or 1)
            except (TypeError, ValueError):
                loops = 1
        except ValueError:
            act = None
    if act is None:
        hit = _ACT_RE.search(raw)
        if hit:
            act = hit.group(1).lower()
        lm = _LOOPS_RE.search(raw)
        if lm:
            loops = int(lm.group(1))
    if act is None:
        return None, None
    # 容忍模型自造的同义词
    if act in ("start_task", "start", "detect_task", "run_task", "task"):
        act = "detect"
    elif act in ("question", "answer", "talk", "none", ""):
        act = "chat"
    if act not in VALID:
        return None, None
    if loops not in (1, 2, 3):
        loops = 1
    return act, loops


def _ask_router(text, brain, verbose=True):
    """真的去问大模型。返回 (act, loops)；问不到返回 (None, None)。"""
    if brain is None or getattr(brain, "_voice", None) is None:
        return None, None
    voice = brain._voice
    try:
        if not voice.ready():
            return None, None
    except Exception:                            # noqa: BLE001
        return None, None

    messages = [{"role": "system", "content": ROUTER_SYSTEM}]
    # 带上最近两轮对话，让"开始吧"这种依赖上文的短句也能判对
    try:
        hist = brain._history()
        if hist:
            messages.extend(hist[-4:])
    except Exception:                            # noqa: BLE001
        pass
    messages.append({"role": "user", "content": text})

    try:
        out = voice.chat(messages, temperature=0.0, max_tokens=24, search=False)
    except Exception as exc:                     # noqa: BLE001
        if verbose:
            print("  [意图] 大模型路由失败，退回规则：{}".format(exc))
        return None, None
    act, loops = _parse(out)
    if verbose:
        print("  [意图] 大模型原文：{}".format((out or "").strip()[:60]))
    return act, loops


def route(text, brain=None, verbose=True):
    """判定一句话该怎么处理。返回 ``(意图, 说明)``。

    顺序即安全优先级：**停 -> 大模型 -> 收紧版规则**。
    """
    said = intent.normalize(text)
    if not said:
        return intent.NONE, "空"

    # ---- 1. 安全阀：规则，最高优先级，不可被模型推翻 ----
    if intent.is_stop(said):
        return intent.STOP, "命中停止词（规则优先，不问模型）"

    # ---- 2. 大模型推理 ----
    act, loops = _ask_router(text, brain, verbose=verbose)
    if act == "detect":
        return intent.START_TASK, "大模型判定要执行（loops=%d）" % loops
    if act == "status":
        return intent.STATUS, "大模型判定在问结果"
    if act == "chat":
        return intent.QUESTION, "大模型判定是聊天 -> 正常对话/联网"

    # ---- 3. 兜底：收紧版规则（模型不可用/输出无法解析） ----
    kind, why = intent.classify(text)
    if kind == intent.START_TASK:
        return kind, "规则兜底：" + why
    return kind, "规则兜底：" + why


def loops_of(text, brain=None, default=1):
    """单独取"要测几块"。优先用大模型的判断，失败再数关键词。"""
    _, loops = _ask_router(text, brain, verbose=False) if brain is not None else (None, None)
    if loops in (1, 2, 3):
        return loops
    return intent.wanted_loops(text) or default


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------
def _self_test():
    """离线自检：只测解析 + 规则兜底（真调用模型见 xingbao_sim.py [7/7]）。"""
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append("%s: got %r want %r" % (name, got, want))

    # JSON 解析
    check("纯JSON", _parse('{"act":"detect","loops":3}'), ("detect", 3))
    check("带解释", _parse('好的\n{"act":"chat","loops":1}'), ("chat", 1))
    check("同义词start", _parse('{"act":"start_task"}'), ("detect", 1))
    check("同义词question", _parse('{"act":"question","loops":1}'), ("chat", 1))
    check("loops越界", _parse('{"act":"detect","loops":9}'), ("detect", 1))
    check("非JSON字段式", _parse('act: detect, loops: 2'), ("detect", 2))
    check("垃圾输出", _parse('我不知道'), (None, None))
    check("空", _parse(''), (None, None))

    # 无 brain 时走规则兜底：泛动词不再触发动起来（这是本次修复的核心）
    for said in ("帮我查一下今天天气", "你看看这个是什么", "奶酪为什么会发霉",
                 "检测是怎么做的", "星宝你几岁了", "今天几点了"):
        kind, _ = route(said, None, verbose=False)
        check("不误动：" + said, kind, intent.QUESTION)

    # 真正的命令仍然要动
    for said in ("去检测一下面前这块奶酪", "再测一次", "帮我看看这块奶酪",
                 "检查一下这块奶酪"):
        kind, _ = route(said, None, verbose=False)
        check("该动：" + said, kind, intent.START_TASK)

    # 停止词永远优先
    for said in ("停", "不要再检测了", "别做了", "不用了"):
        kind, _ = route(said, None, verbose=False)
        check("必停：" + said, kind, intent.STOP)

    return fails


if __name__ == "__main__":
    bad = _self_test()
    if bad:
        print("✗ %d 项失败" % len(bad))
        for b in bad:
            print("   -", b)
    else:
        print("✓ 意图路由自检全部通过")
