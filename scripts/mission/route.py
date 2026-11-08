# -*- coding: utf-8 -*-
"""终点状态机：按"识别到的个数"决定去哪。

用户 2026-11-08 定的规则
------------------------
::

    个数 = 0  ->  保持不动（不导航）
    个数 = 1  ->  去终点 A
    个数 = 2  ->  去终点 B

三个终点的 XY + yaw **由用户给**，现在先留空（``None``）。留空时不会瞎跑：
``decide()`` 返回 ``HOLD`` 并在日志里写明"终点没配"，绝不猜坐标。

为什么要写成状态机而不是 if/else
--------------------------------
现场顺序是「跑完三圈 -> 数有几个 -> 决定去哪 -> 开过去」。中间任何一步都可能
出岔子（没看清、定位丢了、导航被挡），把这些都显式列成状态，出问题时看日志
就知道卡在哪一步，而不是从一堆 print 里猜。

::

    IDLE --set_count--> COUNTED --decide--> HOLDING / MOVING --arrived--> DONE
                                    |                    |
                                    +--------------------+--> FAILED

* ``COUNTED``：个数已经拿到，还没决定。
* ``HOLDING``：结论是"保持不动"，已完成。
* ``MOVING``：正在开往某个终点。
* ``DONE``：到达并满足容差（``pos_tol_m`` / ``yaw_tol_deg``）。
* ``FAILED``：导航失败 / 误差超限 / 没定位。**不自动重试**，等人判断。

个数的来源
----------
"识别到的个数"有两种口径，用 ``count_mold`` 的 ``mode`` 选：

``"faces"``（默认）
    有霉的**面数**。三圈六面里几个面有霉 -> 0~6。
``"spots"``
    单次查看里**菌斑个数**的最大值（一颗霉可能被切成几块，口径偏碎）。

两种都实现了，切换只改一个参数。等用户明确口径后把另一种删掉。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------
# 终点坐标：等用户补
# --------------------------------------------------------------------------
# 单位：米 / 度，地图坐标系（和 nav_goals/table.json 同一个系）。
# 补的时候只改下面三个数，不用动逻辑。
#
# 另外也支持"放一个 JSON 文件"的方式（优先于下面的字典）：
#   scripts/nav_goals/route_a.json  ->  {"pose": [x, y, z, qx, qy, qz, qw],
#                                        "yaw_deg": ..., "note": "..."}
# 现场如果直接用 `main.py nav --record route_a` 录点，就会自动生成这种文件。
GOAL_A = dict(name="route_a", x=None, y=None, yaw_deg=None, note="个数=1 时去这里")
GOAL_B = dict(name="route_b", x=None, y=None, yaw_deg=None, note="个数=2 时去这里")

GOAL_DIR = Path(__file__).resolve().parents[1] / "nav_goals"

POS_TOL_M = 0.06                     # 和抓取段一致：夹爪压入量只有 3 cm
YAW_TOL_DEG = 4.0

# --------------------------------------------------------------------------
# 状态
# --------------------------------------------------------------------------
IDLE = "IDLE"
COUNTED = "COUNTED"
HOLDING = "HOLDING"
MOVING = "MOVING"
DONE = "DONE"
FAILED = "FAILED"

# 动作
HOLD = "hold"                        # 原地不动
GOTO = "goto"                        # 去某个终点


@dataclass
class Decision:
    """状态机的判决结果。"""
    count: int
    action: str                      # HOLD / GOTO
    goal_name: Optional[str] = None
    pose: Optional[List[float]] = None
    reason: str = ""
    ready: bool = True               # 终点配好了没（没配 -> False，且 action 会被降级成 HOLD）

    def describe(self) -> str:
        if self.action == HOLD:
            return "个数={} -> 保持不动（{}）".format(self.count, self.reason)
        return "个数={} -> 去 {}（{}）".format(self.count, self.goal_name, self.reason)


@dataclass
class MachineState:
    state: str = IDLE
    count: Optional[int] = None
    decision: Optional[Decision] = None
    history: List[Dict] = field(default_factory=list)

    def log(self, text: str) -> None:
        self.history.append(dict(t=time.strftime("%H:%M:%S"), state=self.state, msg=text))


# --------------------------------------------------------------------------
# 终点解析
# --------------------------------------------------------------------------
def _yaw_to_quat(yaw_deg: float) -> List[float]:
    half = math.radians(float(yaw_deg)) * 0.5
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def resolve_goal(spec: Dict, goal_dir: Optional[Path] = None) -> Optional[List[float]]:
    """把终点描述变成 ``[x, y, z, qx, qy, qz, qw]``。配不全就返回 None。

    优先读 ``<goal_dir>/<name>.json``（现场录的点），没有再看字典里的
    ``x / y / yaw_deg``。两处都没配 -> None，调用方据此降级成"保持不动"。
    """
    name = spec.get("name")
    directory = Path(goal_dir) if goal_dir else GOAL_DIR
    if name:
        path = directory / "{}.json".format(name)
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                pose = [float(v) for v in data["pose"]]
                if len(pose) >= 7:
                    return pose[:7]
            except Exception as exc:            # noqa: BLE001
                print("  ⚠ 终点文件 {} 读不了：{}".format(path, exc))

    x, y, yaw = spec.get("x"), spec.get("y"), spec.get("yaw_deg")
    if x is None or y is None or yaw is None:
        return None
    qx, qy, qz, qw = _yaw_to_quat(float(yaw))
    return [float(x), float(y), 0.0, qx, qy, qz, qw]


# --------------------------------------------------------------------------
# 状态机
# --------------------------------------------------------------------------
class RouteMachine:
    """按个数决定去哪，并（可选）真的开过去。

    ``navigator`` 传 ``ChassisNavigator`` 就执行导航；传 None 就只做判决，
    方便离线回归和"先看看会怎么判"。
    """

    def __init__(
        self,
        navigator=None,
        *,
        goal_dir: Optional[Path] = None,
        goal_a: Optional[Dict] = None,
        goal_b: Optional[Dict] = None,
        pos_tol_m: float = POS_TOL_M,
        yaw_tol_deg: float = YAW_TOL_DEG,
        voice=None,
        verbose: bool = True,
    ) -> None:
        self._nav = navigator
        self._goal_dir = Path(goal_dir) if goal_dir else GOAL_DIR
        self._goal_a = dict(goal_a or GOAL_A)
        self._goal_b = dict(goal_b or GOAL_B)
        self.pos_tol_m = float(pos_tol_m)
        self.yaw_tol_deg = float(yaw_tol_deg)
        self._voice = voice
        self._verbose = bool(verbose)
        self.st = MachineState()
        self._say_pending = True

    # ------------------------------------------------------------------
    # 判决（纯函数部分，可离线跑）
    # ------------------------------------------------------------------
    def decide(self, count: int) -> Decision:
        """个数 -> 该干什么。**不产生任何运动。**"""
        count = int(count)
        if count < 0:
            raise ValueError("个数不能是负数：{}".format(count))

        if count == 0:
            return Decision(count=0, action=HOLD, reason="个数 0：保持不动")

        spec = self._goal_a if count == 1 else (self._goal_b if count == 2 else None)
        if spec is None:
            # 用户只定义了 0/1/2；>=3 没有对应终点，按"不动"处理并大声提示，
            # 不要自己编一个"往第 2 个终点送"的规则。
            return Decision(count=count, action=HOLD,
                            reason="个数 {} 没有配置终点（只定义了 0/1/2），保持不动".format(count))

        pose = resolve_goal(spec, self._goal_dir)
        if pose is None:
            return Decision(count=count, action=HOLD, goal_name=spec.get("name"),
                            reason="终点 {} 坐标还没配（填 mission/route.py 的 GOAL_* 或放 "
                                   "nav_goals/{}.json），先保持不动".format(
                                       spec.get("name"), spec.get("name")),
                            ready=False)

        return Decision(count=count, action=GOTO, goal_name=spec.get("name"), pose=pose,
                        reason="个数 {} -> 去 {}".format(count, spec.get("name")))

    # ------------------------------------------------------------------
    # 状态迁移
    # ------------------------------------------------------------------
    def set_count(self, count: int) -> Decision:
        """把"数出来几个"喂进来，推进到 COUNTED 并给出判决。"""
        decision = self.decide(count)
        self.st.count = int(count)
        self.st.decision = decision
        self.st.state = COUNTED
        self.st.log(decision.describe())
        if self._verbose:
            print("  [route] {}".format(decision.describe()))
            if not decision.ready:
                print("  [route] ⚠ 终点没配，这次不会动。补坐标见 mission/route.py 开头。")
        return decision

    def execute(self, decision: Optional[Decision] = None) -> str:
        """按判决行动，返回结束状态。需要 navigator。"""
        decision = decision or self.st.decision
        if decision is None:
            raise RuntimeError("还没有判决，先调 set_count()")

        if decision.action == HOLD:
            self.st.state = HOLDING
            self.st.log("保持不动")
            if self._verbose:
                print("  [route] 保持不动，不调用导航")
            return self.st.state

        if self._nav is None:
            self.st.state = FAILED
            self.st.log("没有 navigator，无法导航")
            print("  [route] ⚠ 传了终点但没有 navigator，只做了判决")
            return self.st.state

        self.st.state = MOVING
        self.st.log("开往 {}".format(decision.goal_name))
        if self._verbose:
            print("  [route] 开往 {}：{}".format(decision.goal_name,
                                                _fmt_pose(decision.pose)))
        self._announce_move()
        try:
            # navigate_to_pose 内部已经做了到位容差检查和超差抛错
            pos_err, yaw_err = self._nav.navigate_to_pose(decision.pose)
        except Exception as exc:                    # noqa: BLE001
            self.st.state = FAILED
            self.st.log("导航失败：{}".format(exc))
            print("  [route] ✗ 导航失败：{}".format(exc))
            return self.st.state

        ok = pos_err <= self.pos_tol_m and abs(yaw_err) <= self.yaw_tol_deg
        self.st.state = DONE if ok else FAILED
        self.st.log("到位 位置 {:.3f} m 朝向 {:.2f} deg".format(pos_err, yaw_err))
        if self._verbose:
            print("  [route] {} 位置误差 {:.3f} m，朝向误差 {:.2f} deg".format(
                "✓ 到位" if ok else "✗ 超差", pos_err, yaw_err))
        return self.st.state

    def on_count(self, count: int) -> str:
        """一步到位：喂个数 + 执行。返回结束状态。"""
        return self.execute(self.set_count(count))

    # ------------------------------------------------------------------
    def _announce_move(self) -> None:
        """只有真正要走的时候才播一句（VoiceBox 的 "moving" 片段，可选）。"""
        if self._voice is None or not self._say_pending:
            return
        try:
            if hasattr(self._voice, "has") and self._voice.has("moving"):
                self._voice.say("moving")
        except Exception:                            # noqa: BLE001
            pass

    def report(self) -> List[str]:
        return ["{} {} {}".format(h["t"], h["state"], h["msg"]) for h in self.st.history]


def _fmt_pose(pose: Optional[Sequence[float]]) -> str:
    if not pose:
        return "-"
    yaw = math.degrees(math.atan2(2.0 * (pose[6] * pose[5]), 1.0 - 2.0 * pose[5] * pose[5]))
    return "x={:.3f} y={:.3f} yaw={:.1f}°".format(pose[0], pose[1], yaw)


# --------------------------------------------------------------------------
# 个数的来源
# --------------------------------------------------------------------------
def count_mold(results: Sequence[Dict], mode: str = "faces") -> int:
    """把 ``CheeseInspector.results`` 数成一个整数。

    ``mode="faces"``：有霉的面数（推荐，语义最直白）。
    ``mode="spots"``：单次查看里菌斑数的最大值。

    只统计 ``reason == "ok"`` 的查看 —— 没分出奶酪的那次既不算有霉也不算没霉，
    它压根没有结论，不能被当成 0。
    """
    usable = [r for r in results if r.get("reason") == "ok"]
    if mode == "spots":
        return max([len(r.get("spots") or []) for r in usable], default=0)
    if mode != "faces":
        raise ValueError("未知的计数口径：{}（只支持 faces / spots）".format(mode))
    return sum(1 for r in usable if r.get("mold"))


def count_from_inspector(inspector, mode: str = "faces") -> int:
    if inspector is None:
        return 0
    return count_mold(getattr(inspector, "results", []) or [], mode=mode)
