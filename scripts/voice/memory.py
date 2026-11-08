# -*- coding: utf-8 -*-
"""星宝的记忆：短时对话 + 长期检测记录。

用户 2026-11-08 的要求："后期人通过语音提问时，需要有上下文的记忆。"

两层记忆，作用不同
------------------
**短时（对话）**
    最近 ``max_turns`` 轮原话。用来接"它""那个""刚才说的"这种指代。
    只留最近若干轮，多了就把最老的挤出去 —— 云端上下文不是无限长的。

**长期（检测）**
    每跑完一次任务记一条：什么时候、几面有霉、结论是什么、有没有被送走。
    这个必须**落盘**（``voice/memory.json``）：展会中途重启程序，
    "刚才那块奶酪怎么样" 还得答得上来。

存盘策略
--------
不记录任何用户隐私以外的东西，只有"检测过什么"和"聊过什么"。
写入用临时文件 + ``os.replace``，避免掉电留下半个 JSON。
文件读写失败一律**不抛异常** —— 记忆丢了不影响机器人干活。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

MAX_TURNS = 40          # 约 20 轮问答
MAX_TASKS = 30          # 最近 30 次检测
TURN_CHARS = 160        # 单条原话截断，防止某次 ASR 把一大段都塞进来


def _now():
    return time.strftime("%m-%d %H:%M")


class MemoryStore(object):
    """对话与任务记录。线程不安全，但只有语音主线程会碰它。"""

    def __init__(self, path=None, max_turns: int = MAX_TURNS,
                 max_tasks: int = MAX_TASKS, verbose: bool = True) -> None:
        self.path = Path(path) if path else (
            Path(__file__).resolve().parent / "memory.json")
        self.max_turns = int(max_turns)
        self.max_tasks = int(max_tasks)
        self._verbose = bool(verbose)
        self.turns: List[Dict] = []
        self.tasks: List[Dict] = []
        self.load()

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------
    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle) or {}
            self.turns = list(data.get("turns") or [])[-self.max_turns:]
            self.tasks = list(data.get("tasks") or [])[-self.max_tasks:]
            if self._verbose and (self.turns or self.tasks):
                print("  [memory] 载入 {} 轮对话、{} 条检测记录".format(
                    len(self.turns), len(self.tasks)))
        except Exception as exc:                 # noqa: BLE001
            print("  [memory] 读记忆失败（当作空的）：{}".format(exc))
            self.turns, self.tasks = [], []

    def save(self) -> None:
        payload = dict(updated=_now(), turns=self.turns[-self.max_turns:],
                       tasks=self.tasks[-self.max_tasks:])
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=1)
            os.replace(str(tmp), str(self.path))
        except Exception as exc:                 # noqa: BLE001
            print("  [memory] 存记忆失败：{}".format(exc))

    def reset(self) -> None:
        self.turns, self.tasks = [], []
        self.save()
        if self._verbose:
            print("  [memory] 已清空")

    # ------------------------------------------------------------------
    # 对话
    # ------------------------------------------------------------------
    def add_turn(self, role: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.turns.append(dict(t=_now(), role=role, text=text[:TURN_CHARS]))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
        self.save()

    def history(self, limit: int = 6) -> List[Dict]:
        """给 Brain 用的 ``[{role, content}]``。"""
        out = []
        for turn in self.turns[-limit:]:
            role = turn.get("role")
            if role in ("user", "assistant"):
                out.append({"role": role, "content": turn.get("text", "")})
        return out

    # ------------------------------------------------------------------
    # 检测记录
    # ------------------------------------------------------------------
    def add_task(self, record: Dict) -> Dict:
        record = dict(record)
        record.setdefault("time", _now())
        record.setdefault("ts", time.time())
        self.tasks.append(record)
        if len(self.tasks) > self.max_tasks:
            self.tasks = self.tasks[-self.max_tasks:]
        self.save()
        return record

    def last_task(self) -> Optional[Dict]:
        return self.tasks[-1] if self.tasks else None

    def moldy_count(self) -> int:
        return sum(1 for r in self.tasks if int(r.get("moldy") or 0) > 0)

    # ------------------------------------------------------------------
    # 给大模型的上下文
    # ------------------------------------------------------------------
    def context_text(self, max_chars: int = 700) -> str:
        """拼一段紧凑的中文上下文，塞进 system prompt。

        只放**事实**（测过几次、结论如何），不放"你应该怎么答" —— 语气由
        Brain.SYSTEM 管，这里不要跟它打架。
        """
        bits = []
        last = self.last_task()
        if last:
            bits.append("最近一次检测（{}）：{}".format(
                last.get("time", "?"), last.get("spoken") or last.get("verdict", "")))
            route = last.get("route")
            if route:
                bits.append("那次的去向：{}".format(route))
        if len(self.tasks) > 1:
            older = self.tasks[-4:-1]
            brief = "；".join(
                "{}（{}）".format(r.get("time", "?"),
                                 "有霉" if int(r.get("moldy") or 0) else "干净")
                for r in older)
            bits.append("更早的检测：{}".format(brief))
        if self.tasks:
            bits.append("累计完成 {} 次检测，其中 {} 次发现霉菌。".format(
                len(self.tasks), self.moldy_count()))
        if not bits:
            return ""
        text = "你记得这些（都是你亲自测的，可以直接引用）：\n" + "\n".join(
            "- " + b for b in bits)
        return text[:max_chars]

    def describe_last(self) -> str:
        """直接用嘴回答"刚才那块怎么样"。没有记录就返回空串。

        2026-11-08 用户要求"进行自然语言的对话"：这里原来会念成
        "刚刚那块（11-08 03:30）这块奶酪两面都干净"，时间戳夹在括号里
        是给人看的，念出来像日志。改成纯口语，时间不进播报。
        """
        last = self.last_task()
        if not last:
            return ""
        say = last.get("spoken") or last.get("verdict") or ""
        if not say:
            return ""
        # "这块奶酪两面都干净" -> "刚才那块，两面都干净"
        # 否则会念成"刚刚那块，这块奶酪两面都干净"，重复得很生硬。
        for lead in ("这块奶酪", "块奶酪", "这块", "这个"):
            if say.startswith(lead):
                say = say[len(lead):]
                break
        return "刚才那块，{}".format(say or "我记不太清了。")
