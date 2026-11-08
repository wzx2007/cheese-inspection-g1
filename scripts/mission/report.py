# -*- coding: utf-8 -*-
"""把一次检测的原始结果，翻译成"人话结论"。

用户 2026-11-08 明确要的三种说法::

    两面都有霉  /  一面有霉  /  合格

现场看到的是 6 条 ``CheeseInspector.results``（3 圈 × 2 面）。所以先按圈切，
每圈 2 面单独下结论，最后再给整场汇总 —— 不能把"第 1 圈的正面"和
"第 3 圈的反面"混在一起算成"一个奶酪"。

"没看清"和"干净"是两件事
------------------------
``reason != "ok"`` 表示那一面没分出奶酪（没对准 / 被手臂挡了），
它**既不是有霉也不是没霉**。这种面一多，结论就只能说"有一面我没看清"，
绝不能因为"没看到霉"就说合格 —— 这和 ``inspector.announce`` 的口径一致。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

FACE_A = "正面"
FACE_B = "反面"


def face_label(text: str) -> str:
    """从 tag 里认出是正面还是反面。认不出就原样返回。"""
    text = str(text or "")
    if "正面" in text:
        return FACE_A
    if "反面" in text:
        return FACE_B
    return text or "这一面"


def _is_usable(rec: Dict) -> bool:
    return (rec or {}).get("reason") == "ok"


def summarize(looks: Sequence[Dict]) -> Dict:
    """一组查看结果 -> 结构化结论。

    返回 ``dict(faces, moldy, clean, unknown, total, verdict, spoken, ok)``：

    * ``faces``   每一面的明细（正面/反面、是否有霉、是否看清）
    * ``moldy``   有霉的面数
    * ``clean``   看清了且干净的面数
    * ``unknown`` 没看清的面数
    * ``spoken``  可以直接念出来的一句话
    """
    faces: List[Dict] = []
    for rec in looks or []:
        rec = rec or {}
        faces.append(dict(
            face=face_label(rec.get("tag") or rec.get("slug")),
            mold=bool(rec.get("mold")),
            usable=_is_usable(rec),
            reason=rec.get("reason") or "",
            verdict=rec.get("verdict") or "",
        ))

    usable = [f for f in faces if f["usable"]]
    moldy = sum(1 for f in usable if f["mold"])
    clean = len(usable) - moldy
    unknown = len(faces) - len(usable)

    spoken, verdict, ok = _speak(moldy, clean, unknown, len(faces))
    return dict(faces=faces, moldy=moldy, clean=clean, unknown=unknown,
                total=len(faces), verdict=verdict, spoken=spoken, ok=ok)


def _speak(moldy: int, clean: int, unknown: int, total: int):
    """(念出来的话, 短结论, 是否有明确结论)。"""
    if total == 0:
        return ("我这次什么也没看到，没有结论。", "无结论", False)
    if moldy and unknown:
        return ("这块奶酪有 {} 面发现了霉，还有 {} 面我没看清。".format(moldy, unknown),
                "{} 面有霉、{} 面没看清".format(moldy, unknown), True)
    if moldy >= 2:
        return ("这块奶酪两面都有霉。", "两面都有霉", True)
    if moldy == 1:
        if clean:
            return ("这块奶酪一面有霉，另一面是干净的。", "一面有霉", True)
        return ("这块奶酪有一面有霉。", "一面有霉", True)
    if unknown:
        return ("我看到的那面是干净的，还有 {} 面没看清，我不敢下结论。".format(unknown),
                "{} 面没看清".format(unknown), False)
    return ("这块奶酪两面都干净，是合格的。", "合格", True)


def merge(looks: Sequence[Dict]) -> Dict:
    """整场（多圈）汇总。和 ``summarize`` 同一口径，只是不分圈。"""
    return summarize(looks)


def group_by_round(looks: Sequence[Dict], per_round: int = 2) -> List[Dict]:
    """按"每圈几面"切开，给多圈演示用。

    ``per_round`` 默认 2（一正一反）。返回 ``[summarize(...), ...]``。
    切不整齐（比如某圈少看了一面）也不报错，最后一段短一点而已。
    """
    per_round = max(1, int(per_round))
    looks = list(looks or [])
    groups = []
    for start in range(0, len(looks), per_round):
        groups.append(summarize(looks[start:start + per_round]))
    return groups


def record(rounds: Sequence[Dict], route_text: str = "", loops: int = 1) -> Dict:
    """拼一条写进记忆的记录。``rounds`` 是 ``group_by_round`` 的结果。"""
    faces = []
    for item in rounds:
        faces.extend(item.get("faces") or [])
    moldy = sum(int(item.get("moldy") or 0) for item in rounds)
    clean = sum(int(item.get("clean") or 0) for item in rounds)
    unknown = sum(int(item.get("unknown") or 0) for item in rounds)
    if len(rounds) == 1:
        spoken = rounds[0].get("spoken", "")
        verdict = rounds[0].get("verdict", "")
    else:
        parts = ["第 {} 块：{}".format(i + 1, r.get("verdict", ""))
                 for i, r in enumerate(rounds)]
        spoken = "；".join(parts)
        verdict = spoken
    return dict(loops=int(loops), faces=faces, moldy=moldy, clean=clean,
                unknown=unknown, spoken=spoken, verdict=verdict,
                route=route_text)
