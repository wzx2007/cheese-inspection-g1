# -*- coding: utf-8 -*-
"""把一句话判成"要干什么"。

为什么用规则而不是让大模型判
----------------------------
意图决定机器人**要不要动**。这种地方不能容忍模型偶尔抽风（把"这个奶酪有霉吗"
判成"开始检测"，机器人就开始伸手）。规则是确定性的、可离线回归、出错能一眼看出
是哪条命中。真正需要理解力的部分（"它为什么会长霉"）交给大模型，但那是"回答问题"，
不涉及运动。

意图只有六种，够用且互斥::

    START_TASK  开始检测（会动！）
    STATUS      问刚才的结果
    CONFIRM     确认继续（推完车说"好了"）
    STOP        叫停（只在检查点生效）
    QUESTION    其它问题，交给大模型
    NONE        空

``classify()`` 的判定顺序就是安全意识：**先看停，再看结果，最后才看开始**。
"停一下，检测结果怎么样" 判成 STOP，不会因为里面有"检测"就去干活。
"""

from __future__ import annotations

import re

START_TASK = "start_task"
STATUS = "status"
CONFIRM = "confirm"
STOP = "stop"
QUESTION = "question"
LAUGH = "laugh"
NONE = "none"

# 出现这些词就**不许动**。宁可把"停一下"当成"这次别启动"，也不能反过来。
STOP_WORDS = (
    "停", "别动", "不要动", "取消", "不做了", "不测了", "别做了", "算了",
    "等一下", "等等", "先别", "暂停",
    # 2026-11-08 补：原来漏了这几种否定，说"不要再检测了"反而会命中"检测"去干活。
    "不用", "不要", "别再", "够了", "到此为止", "先这样", "歇会",
)

# 问结果。放在 START 之前判，"检查结果怎么样"才不会被当成要开始检测。
# 分两档：**强**（"结果""几个面"）自己就够；**弱**（"怎么样""怎样"）必须配着
# 对象/动作词才算 —— 否则"今天天气怎么样"会被判成"问检测结果"，星宝就开始
# 念上一块奶酪的结论，问天气的问不上网（2026-11-08 现场踩到）。
STATUS_WORDS = (
    "结果", "完了吗", "完成了吗", "报告", "几个面", "几面",
    "有霉吗", "合格吗", "干净吗", "什么情况", "检测完", "查完", "测出来",
)
STATUS_WEAK_WORDS = ("怎么样", "怎样", "如何", "好不好")
STATUS_CONTEXT = ("奶酪", "这块", "这个", "它", "检测", "检查", "测", "查", "验")

# --------------------------------------------------------------------------
# ⚠️ 2026-11-08 重写：这组词**不再单独决定要不要动**
# --------------------------------------------------------------------------
# 现场问题："无论我说什么都直接跑过去做任务"。根因就是旧的 START_WORDS 里有
# ``看一下 / 看看 / 开始 / 帮我 / 给我 / 替我 / 检测 / 检查 / 干活`` ——
# 全是中文最泛的动词，于是"帮我查一下今天天气"命中"帮我"+"查一下"就真的动了。
#
# 现在职责变了：
#   * 正常情况**由 ``voice/intent_llm.route()`` 请大模型判**意图（见那个模块）；
#   * 这里的规则只在"模型不可用/超时/输出无法解析"时兜底；
#   * 兜底必须**偏保守** —— 少动一次（用户再喊一遍）远好过多动一次（机器人突然伸手）。
#
# 所以 ``START_WORDS`` 只保留"检测类"动词，删掉所有泛动词；并且要求
# **动词 + 对象** 同时出现（``OBJECT_WORDS``），或命中明确的复述命令。
START_WORDS = (
    "检测", "检查", "测一下", "测一遍", "查一下", "验一下", "体检",
)

# 有"对象"才像是真的要动手；只有动词没有对象，兜底时算聊天。
OBJECT_WORDS = ("奶酪", "这块", "面前", "桌上", "它", "这边", "那块")

# "再来一次"这种复述型命令。2026-11-08 现场问题：测完一次以后用户说
# "再检测一次 / 再来一块 / 重来"，原来一个词都不命中 -> 落到 QUESTION ->
# 交给大模型 -> 模型回一句"我刚刚已经检测过了" = 拒绝执行。
# 这些词必须能自己把意图顶成 START_TASK。
RESTART_WORDS = (
    "再检测", "再测", "再看", "再查", "再验", "再试", "再来一次", "再来一遍",
    "再来一块", "再走一遍", "重来", "重新", "测一遍", "检一遍", "再来", "又一遍",
    "接着测", "继续测", "多测", "下一块",
)

# "开始"不能当泛动词用（"开始下雨了"会误触发），但**整句就是**"开始/开始吧"
# 这种短应答是明确的 go 信号（上一句刚问过要不要测）。
GO_WORDS = ("开始", "开始吧", "开始测", "那就开始", "来吧", "那就开始吧")


CONFIRM_WORDS = (
    "好了", "好", "可以", "行", "继续", "走吧", "推好了", "准备了", "准备好了",
    "嗯", "对", "没问题", "go", "ok", "okay", "接着",
)

# 说"几块"就测几块。默认一块 —— 用户说"这块奶酪"就是眼前这一块。
MANY_WORDS = {
    3: ("三块", "三个", "三遍", "三轮", "全部", "完整", "整套", "都测"),
    2: ("两块", "两个", "两遍", "两轮"),
}

_PUNCT = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()\[\]【】…~～]+")


def normalize(text):
    return _PUNCT.sub("", (text or "").strip()).lower()


def _hit(said, words):
    return any(w in said for w in words)


def wanted_loops(text):
    """从原话里数"要测几块"。默认 1。

    "再测一次"/"重来" 只算 1 块 —— 用户要的是"刚才那块再来一遍"，
    不是把三块都跑一遍。
    """
    said = normalize(text)
    for count in sorted(MANY_WORDS, reverse=True):
        if _hit(said, MANY_WORDS[count]):
            return count
    return 1


def is_command(text):
    """这句是不是**在命令机器人干活**（兜底路径用，偏保守）。

    2026-11-08 收紧：旧版对 ""帮我/给我/替我/你去/快来/上去/过去" 一律放行、
    只要后面带个"看/测/检/查"就算命令。结果"帮我查一下今天天气"也变成了
    "去检测奶酪"。现在必须**同时**满足：委托/祈使词 + 检测类动词 + 明确对象。
    正常情况本来就走大模型，这一层只是模型挂掉时的保险，宁可漏判。
    """
    said = normalize(text)
    if not said:
        return False
    if _hit(said, STOP_WORDS):
        return False
    if _hit(said, RESTART_WORDS):
        return True
    if not _hit(said, OBJECT_WORDS):
        return False
    if _hit(said, ("帮我", "给我", "替我", "麻烦你", "你去", "快来", "上去", "过去")):
        return _hit(said, ("看", "测", "检", "查")) and _hit(said, ("奶酪", "这块", "面前"))
    return False


# "既要测、又在问结果"时按**执行**算（用户说"再测一次，顺便看看结果"）。
_STRONG_IMPERATIVE = ("再测", "重来", "重新测", "再来一块", "帮我", "给我", "测一遍", "再检测")


def _is_status(said):
    """在问"检测结果"吗。"""
    if _hit(said, STATUS_WORDS):
        return True
    return _hit(said, STATUS_WEAK_WORDS) and _hit(said, STATUS_CONTEXT)


def classify(text):
    """返回 ``(意图, 说明)``。说明只用来打日志，不参与判断。

    ⚠️ 这是**兜底**路径。正常流程走 ``voice/intent_llm.route()``（大模型判）。
    这里的规则已经收紧：START_TASK 必须"检测类动词 + 明确对象"，或者命中
    复述命令（再测一次）。只有个"看/查/帮"再也不会让机器人动。
    """
    said = normalize(text)
    if not said:
        return NONE, "空"
    if _hit(said, STOP_WORDS):
        return STOP, "命中停止词"
    if _is_status(said):
        if _hit(said, _STRONG_IMPERATIVE):
            return START_TASK, "既要测又问结果 -> 执行"
        return STATUS, "命中结果词"
    # 复述命令（再测一次/重来）自己就够，不需要对象
    if _hit(said, RESTART_WORDS):
        return START_TASK, "命中复述词（再测一次/重来）"
    # 整句就是"开始吧"这类短应答 -> 明确要动手
    if said in GO_WORDS:
        return START_TASK, "短应答 go"
    # 检测类动词必须配对象，否则是"检测是怎么做的"这种问知识
    if _hit(said, START_WORDS) and _hit(said, OBJECT_WORDS):
        return START_TASK, "命中检测动词 + 对象"
    if is_command(said):
        return START_TASK, "祈使句"
    if len(said) <= 5 and _hit(said, CONFIRM_WORDS):
        return CONFIRM, "命中确认词"
    return QUESTION, "兜底"


def is_stop(text):
    return _hit(normalize(text), STOP_WORDS)


def is_confirm(text):
    """确认用：短句 + 确认词。现场推完车一般就两个字，"好了"。"""
    said = normalize(text)
    if not said:
        return False
    if _hit(said, STOP_WORDS):
        return False
    if len(said) <= 6 and _hit(said, CONFIRM_WORDS):
        return True
    # "我推好了" / "已经推到位了" 这种长一点的
    return _hit(said, ("推好", "推到位", "准备好了", "可以开始", "接着来"))

# ---------------------------------------------------------------------------
# 笑声：现场加的一个"彩蛋"（2026-11-08 用户要求）。
#
# 为什么走**规则**而不是让大模型判：这一条只是放一段音频，不动手臂不动底盘，
# 判错了最坏结果是"多笑一声"，代价极低；而走模型要等一次网络往返（0.5~2 s），
# 现场说"笑一个"之后干等两秒就没意思了。安全阀（is_stop）依然是规则优先，
# 这里也一样：句子里带"停/别/不要"就不算。
# ---------------------------------------------------------------------------
LAUGH_WORDS = (
    "大笑", "笑一个", "笑一下", "笑笑", "笑给我", "来一段笑", "来个笑",
    "笑两声", "笑一笑", "逗我笑", "奶龙笑", "奶龙大笑", "给我笑",
    "哈哈哈哈", "哈哈大笑",
)
# "笑"太泛（"你在笑什么"是聊天），必须落在这些短语里才算；单独一个"笑"
# 只在整句很短（<=4 字）时才算，比如"笑一个""笑下"。
LAUGH_SHORT = ("笑", "笑个", "笑下")


def is_laugh(text):
    """用户是不是想让机器人**放一声奶龙的笑**。

    >>> is_laugh("给我笑一个")
    True
    >>> is_laugh("你在笑什么")
    False
    >>> is_laugh("笑一个，然后停")
    False
    """
    said = normalize(text)
    if not said:
        return False
    if _hit(said, STOP_WORDS):          # "笑一个然后停" -> 停优先
        return False
    if _hit(said, LAUGH_WORDS):
        return True
    return len(said) <= 4 and _hit(said, LAUGH_SHORT)
