"""Two-stage chassis navigation for the cheese demo.

The table goal sits in a tight spot, so one global plan sometimes stops short of
it.  The move is therefore split:

1. ``navigate_to_goal`` **straight at the goal** (this is the only thing we do
   in the common case);
2. if the planner refuses the goal, fall back to a pre-stop + short
   ``move_straight_to`` steps in the odom frame for the last stretch.

Design rule learned the hard way (2026-11-08): **let the planner do the
driving.**  ``demo_flow.py`` navigated well because it only ever did
``navigate_to(name)`` and then waited.  This class used to steal control in
three ways, and each one turned a recoverable hiccup into a failed demo:

- it aborted the trip on a *transient* non-RUNNING status (the SDK reports
  FAILED for a moment while it replans, and its log is full of
  ``has unknow task result`` — the task result is genuinely lost);
- it judged the trip by "remaining distance", which a localization jump of
  several metres makes meaningless;
- the last-mile odom补位 raised ``NavigationError`` when it could not help,
  killing the whole task even though it is only a finishing touch.

Second lesson (same day, from the field log): **stop walking backwards first.**
``check_path_reachability`` returns False for *every* goal whenever the robot
itself is standing inside the 0.45 m inflation zone — which is exactly what
happens at the table.  We used to treat that as "the goal is unreachable", go
via a pre-stop instead, watch the pre-stop fail with COLLISION/stall, and only
then try the goal directly... which worked.  So three laps burned 30-50 s each
on a detour that always lost, and the winning move was the one at the end.
Now the goal itself is always candidate #1, and the pre-check is only a hint
used to order the fallbacks.

So: wait patiently, judge by pose, try the goal first, and never let the
fallback win the argument.

The class does not own SDK startup or shutdown.  Pass an initialised
``GalbotRobot`` and ``GalbotNavigation``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple


class NavigationError(RuntimeError):
    """Navigation did not reach the goal closely enough to start a grasp."""


# 这些状态说明"目标点/路径落在障碍膨胀区里"，原地重试没意义，要换更远的预停位。
TOO_CLOSE_STATUS = ("CLOSE_TO_OBSTACLE", "OCCUPIED", "GOAL_IN_OCCUPIED", "IN_OBSTACLE")

# 最多换几个预停位。每个失败点约 5 s（2 s 判定 + 3 s 停稳），6 个封顶半分钟。
MAX_PRESTEP_TRIES = 6

def quat_to_yaw_deg(quaternion: Sequence[float]) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny, cosy))


def wrap_deg(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def pose_from_yaw(x: float, y: float, yaw_deg: float, z: float = 0.0) -> List[float]:
    half = math.radians(float(yaw_deg)) * 0.5
    return [float(x), float(y), float(z), 0.0, 0.0, math.sin(half), math.cos(half)]


def relative_pose(current: Sequence[float], goal: Sequence[float]) -> List[float]:
    """Pose of ``goal`` expressed relative to ``current``, x forward."""
    yaw = math.radians(quat_to_yaw_deg(current[3:7]))
    dx = float(goal[0]) - float(current[0])
    dy = float(goal[1]) - float(current[1])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    dx_robot = cosine * dx + sine * dy
    dy_robot = -sine * dx + cosine * dy
    dyaw = math.radians(
        wrap_deg(quat_to_yaw_deg(goal[3:7]) - quat_to_yaw_deg(current[3:7]))
    )
    return [dx_robot, dy_robot, 0.0, 0.0, 0.0, math.sin(dyaw * 0.5), math.cos(dyaw * 0.5)]


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class ChassisNavigator:
    """Navigate to a recorded map goal, then stop within the grasp tolerance."""

    def __init__(
        self,
        robot,
        nav,
        goal_dir,
        *,
        timeout_s: float = 150.0,
        stage1_retries: int = 0,
        arrival_threshold: Sequence[float] = (0.2, 0.2, 0.2),
        backoff_m: float = 0.0,
        backoff_scan: Sequence[float] = (0.8, 1.0, 1.2),
        lateral_scan: Sequence[float] = (),
        max_lastmile_m: float = 2.5,
        skip_global_m: float = 0.5,
        lastmile_step_m: float = 0.2,
        first_step_m: float = 0.6,
        max_step_m: float = 0.4,
        lastmile_tries: int = 10,
        direct_first: bool = True,
        rot_first_deg: float = 10.0,
        max_rot_step_deg: float = 20.0,
        rot_creep_m: float = 0.05,
        pose_wait_s: float = 3.0,
        yaw_overshoot_deg: float = 25.0,
        straight_timeout_s: float = 8.0,
        stall_timeout_s: float = 8.0,
        max_distance_m: float = 15.0,
        self_heal: bool = True,
        pos_tol_m: float = 0.06,
        yaw_tol_deg: float = 4.0,
        arrive_tol_m: float = 0.22,
        status_grace_s: float = 6.0,
        refuse_grace_s: float = 1.5,
        pose_jump_m: float = 1.5,
        pose_jump_deg: float = 60.0,
        relocalize_retries: int = 2,
        navigate_budget_s: float = 100.0,
        on_move: Optional[Callable[[], None]] = None,
    ) -> None:
        self._robot = robot
        self._nav = nav
        self._goal_dir = Path(goal_dir)
        self.timeout_s = float(timeout_s)
        # 已废弃：对同一个点重派一次从来不会改变规划器的结论（同一张图、同一个
        # 起点），现场日志里每次"重试 1/1"都是同样的 COLLISION。留着这个形参
        # 只是不想让老调用方报错，值不再被读取 —— 见 _stage1 的说明。
        self.stage1_retries = int(stage1_retries)
        self.arrival_threshold = [float(value) for value in arrival_threshold]
        self.backoff_m = float(backoff_m)
        self.backoff_scan = [float(value) for value in backoff_scan]
        self.lateral_scan = [float(value) for value in lateral_scan]
        self.max_lastmile_m = float(max_lastmile_m)
        self.skip_global_m = float(skip_global_m)
        self.lastmile_step_m = float(lastmile_step_m)
        self.first_step_m = float(first_step_m)
        self.max_step_m = float(max_step_m)
        self.lastmile_tries = int(lastmile_tries)
        self.direct_first = bool(direct_first)
        self.rot_first_deg = float(rot_first_deg)
        self.max_rot_step_deg = float(max_rot_step_deg)
        self.rot_creep_m = float(rot_creep_m)
        self.pose_wait_s = float(pose_wait_s)
        self.yaw_overshoot_deg = float(yaw_overshoot_deg)
        self.straight_timeout_s = float(straight_timeout_s)
        # 卡死判定时间（车 0.10 m / 5° 都没动算"卡住"）。2026-11-08 从 20 s 压到 8 s：
        # 现场日志里卡住时位姿是 0.000 变化，2 s 一个采样，8 s 就足够确诊；
        # 而 20 s 白等一轮，三圈加起来就是一分多钟干站着。
        self.stall_timeout_s = float(stall_timeout_s)
        self.max_distance_m = float(max_distance_m)
        self.self_heal = bool(self_heal)
        self.pos_tol_m = float(pos_tol_m)
        self.yaw_tol_deg = float(yaw_tol_deg)
        # "够用了"的宽松容差：SDK 自己报 ARRIVED 时一般就在 0.2 m 内，
        # 此时再去补位是白折腾（补位走的 odom 只会把位姿弄脏）。
        # pos_tol_m 现在只决定"要不要微调"，arrive_tol_m 决定"算不算到位"。
        self.arrive_tol_m = float(arrive_tol_m)
        # SDK 在重规划/局部避障那一瞬间会瞬时报 FAILED，但车还在走
        # （2026-11-08 现场：4.0s FAILED，8.0s 又跑到 1 m 外）。
        # 所以"非 RUNNING"必须连续持续这么久、且车确实没动，才认输。
        self.status_grace_s = float(status_grace_s)
        # "派活之后车一动没动、状态却已经报错" = 规划器给不出可执行的路。
        # 2026-11-08 真机探针：navigate_to_goal 回 (True,'SUCCESS')，0.5 s 后
        # 状态就是 FAILED，车一动不动。这种情况没必要等 6 s 宽容期，更没必要
        # 等 8 s 卡死看门狗 —— 换档去试别的点，或者干脆交给直线补位。
        self.refuse_grace_s = float(refuse_grace_s)
        self.pose_jump_m = float(pose_jump_m)
        self.pose_jump_deg = float(pose_jump_deg)
        self.relocalize_retries = int(relocalize_retries)
        # 一趟导航的总预算。demo_flow 那版是一路等 SDK，从来不设上限；
        # 但它也没有"换 6 个预停位"这种逻辑，所以不会叠加。我们这边候选点
        # 多，万一赶上定位跳变，每个点都等满 150 s 就是十几分钟 —— 观众
        # 站那儿看着车发呆。超预算就收手，交给上层自愈或直接报错。
        self.navigate_budget_s = float(navigate_budget_s)
        self._on_move = on_move
        self._announced = False
        self._task_active = False
        # 位姿跳变（重定位/UWB 跳点）是**事件**：pose() 里检测到就置位，
        # 由 _wait_arrival / _last_mile 消费。不能让一次跳点直接把整趟判死
        # —— 2026-11-08 现场就是跳了 6.9 m 之后算"剩余 6.32 m"，直接报错。
        self._last_pose: Optional[List[float]] = None
        self._jump_pending = False
        self._jump_count = 0
        # 当前这一段要去的点（_wait_arrival 用它判断"是不是已经站进容差了"）
        self._current_goal: Optional[List[float]] = None
        # 每个终点上一次真正把车带到位的那档预停位（后退, 横移）。下一趟先试它，
        # 省掉整轮扫描 —— 三圈演示里同一张桌子要走三次，第二、三圈几乎零成本。
        # 按终点坐标分开存：不同终点的周边环境不一样，偏移不能互相借用。
        self._preferred: dict = {}
        # 每一段耗时，收尾时打一行，现场才知道秒数花在哪儿（见 _report_timing）
        self.last_timing: dict = {}

    def pose(self) -> List[float]:
        pose = [float(value) for value in self._nav.get_current_pose()]
        if len(pose) < 7:
            raise NavigationError("get_current_pose 返回 {}".format(pose))
        self._note_jump(pose)
        return pose

    def _note_jump(self, pose: Sequence[float]) -> None:
        """记录一次位姿读数，并检查"物理上不可能"的突变。

        一次读数从 (-1.07, 0.04) 跳到 (4.59, -3.85) 不是车真的飞了 6.9 m，
        是定位跳点（重定位到别处 / UWB 丢基站）。这种读数如果被当成真相，
        算出来的"剩余距离"就是 6.32 m，接下来所有判断都会跟着错。
        所以只标记成事件，让上层决定"重新规划"，而不是直接判失败。
        """
        previous = self._last_pose
        self._last_pose = list(pose)
        if previous is None:
            return
        moved = math.hypot(pose[0] - previous[0], pose[1] - previous[1])
        turned = abs(wrap_deg(quat_to_yaw_deg(pose[3:7]) - quat_to_yaw_deg(previous[3:7])))
        if moved > self.pose_jump_m or turned > self.pose_jump_deg:
            self._jump_pending = True
            self._jump_count += 1
            print("  ⚠ 位姿突变 {:.2f} m / {:.0f}°（{} -> {}）：判为定位跳变，"
                  "不是车真的走了这么远".format(
                      moved, turned,
                      "({:.2f}, {:.2f})".format(previous[0], previous[1]),
                      "({:.2f}, {:.2f})".format(pose[0], pose[1])))

    def _take_jump(self) -> bool:
        """消费一次"位姿跳变"事件。"""
        if self._jump_pending:
            self._jump_pending = False
            return True
        return False

    def _forget_pose(self) -> None:
        """重新派活/重新对齐之后，不要拿旧读数去比。"""
        self._last_pose = None
        self._jump_pending = False

    def load_goal(self, name: str) -> List[float]:
        path = self._goal_dir / "{}.json".format(name)
        if not path.is_file():
            raise NavigationError("找不到目标点 {}（{}）".format(name, path))
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        pose = [float(value) for value in data["pose"]]
        print(
            "  目标点 {} : x={:.4f} y={:.4f} yaw={:.2f} deg".format(
                name, pose[0], pose[1], float(data.get("yaw_deg", quat_to_yaw_deg(pose[3:7])))
            )
        )
        return pose

    def navigate_to(self, goal_name: str) -> Tuple[float, float]:
        """Drive to a recorded goal.  Raise if the final error is too large to grasp."""
        return self.navigate_to_pose(self.load_goal(goal_name))

    def turn_in_place(self, angle_deg: float, steps: int = 3) -> float:
        """原地转 ``angle_deg``，**位置不动**。返回实际转过的角度。

        为什么不直接用 ``navigate_to_pose(同位姿 + 180°)``
        ---------------------------------------------------
        ``demo_flow.py`` 是那么做的，但它会落进 ``_navigate`` 的"起点距终点
        0 m -> 跳过第一段 -> 直接补位"分支，再进 ``_last_mile`` 的"先掰朝向"：
        每步给 ``rot_creep_m``(0.05 m) 的弧线位移 + 最多 20° 转角。转 180° 要
        9 步，累计漂 0.45 m —— 转身变成了画圆。

        这里直接用 odom 相对位姿做**纯旋转**（x=y=0），并按 ``steps`` 拆开：
        一次给满 180° 会因为位姿回读滞后而超转（2026-11-07 踩过：9 条指令
        攒着一起兑现，累计超转 163°）。拆成 60° 一段，每段等位姿确认。
        """
        total = float(angle_deg)
        if abs(total) < 1.0:
            return 0.0
        steps = max(1, int(steps))
        chunk = total / steps
        before_yaw = quat_to_yaw_deg(self.pose()[3:7])
        for index in range(steps):
            half = math.radians(chunk) * 0.5
            relative = [0.0, 0.0, 0.0, 0.0, 0.0, math.sin(half), math.cos(half)]
            try:
                self._nav.move_straight_to(relative, is_blocking=True,
                                           timeout=self.straight_timeout_s)
            except Exception as exc:                      # noqa: BLE001
                print("  转身第 {}/{} 步调用失败（停在这里）：{}".format(
                    index + 1, steps, exc))
                break
            time.sleep(0.3)
        after_yaw = quat_to_yaw_deg(self.pose()[3:7])
        turned = wrap_deg(after_yaw - before_yaw)
        print("  原地转身：指令 {:+.0f}°，实际 {:+.1f}°（{} 步）".format(
            total, turned, steps))
        return turned

    def navigate_to_pose(self, pose: Sequence[float]) -> Tuple[float, float]:
        """Drive to an explicit map pose.  Same arrival check as ``navigate_to``."""
        from galbot_sdk.g1 import ControlStatus, G1ControllerName

        if not self._nav.is_localized():
            raise NavigationError("未定位，拒绝导航")
        goal = [float(value) for value in pose]
        status = self._robot.switch_controller(G1ControllerName.CHASSIS_POSE_CTRL)
        if status != ControlStatus.SUCCESS:
            raise NavigationError("switch_controller(CHASSIS_POSE_CTRL) 返回 {!r}".format(status))
        try:
            pos_err, yaw_err = self._navigate(goal)
            # 自愈：差得离谱说明多半不是"没走到"，而是**位姿已经漂了**（odom
            # 补位累积的 / 定位跳点）。这时候再补也没用，必须让全局导航重新
            # 规划一次 —— 它是唯一会拿地图/激光把位姿拉回来的环节。
            #
            # 2026-11-08 现场两次都是这个味道：
            #   a) 第二趟报"位置误差 1.9533 m、朝向 -108.34°"，车其实没动过；
            #   b) 位姿从 (-1.07, 0.04) 跳到 (4.59, -3.85)，算出"剩余 6.32 m"。
            # 有这一次强制全局，脏位姿下一趟就能自己回来，不会连着毁两趟。
            if self.self_heal and (
                pos_err > self.arrive_tol_m or abs(yaw_err) > self.yaw_tol_deg * 3
            ):
                print("  ⚠ 差了 {:.2f} m / {:+.0f}° —— 位姿可能已经漂了，"
                      "强制走一次全局导航重新对齐".format(pos_err, yaw_err))
                pos_err, yaw_err = self._navigate(goal, force_global=True)
        finally:
            if self._task_active:
                self._stop()
        # 最终判据用"够抓就行"（arrive_tol_m 0.22 m / 12°），不再强求 6 cm：
        # SDK 自己报 ARRIVED 时一般就在 0.2 m 内，非要补到 6 cm 只会让补位
        # 反复走 odom，把位姿弄脏 —— 那才是"第二趟必挂"的源头。
        if pos_err > self.arrive_tol_m or abs(yaw_err) > self.yaw_tol_deg * 3:
            # 分清两种"没到位"，因为现场该做的事完全不同：
            # - 位姿跳变/漂了：让操作员"推我到桌前"纯属误导，该说"重新对位置"；
            # - 真的走不过去（被挡、路被占）：那才需要人出手。
            if self._jump_count:
                raise NavigationError(
                    "定位跳变：位姿突然变了 {} 次，到位误差 {:.2f} m / {:.0f}°".format(
                        self._jump_count, pos_err, yaw_err))
            raise NavigationError(
                "到位误差超限：位置 {:.3f} m / 朝向 {:.2f} deg，容差 {:.2f} m / {:.1f} deg".format(
                    pos_err, yaw_err, self.arrive_tol_m, self.yaw_tol_deg * 3
                )
            )
        return pos_err, yaw_err

    def _announce_move(self) -> None:
        if self._announced:
            return
        self._announced = True
        callback = self._on_move
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            print("  语音未开始：{}".format(exc))

    def _navigate(self, goal: Sequence[float], force_global: bool = False) -> Tuple[float, float]:
        self._announced = False
        timing = {}
        t_all = time.time()
        current = self.pose()
        goal_yaw = quat_to_yaw_deg(goal[3:7])
        distance_now = math.hypot(goal[0] - current[0], goal[1] - current[1])
        # 只有"几乎已经站在终点上"才跳过全局导航（``skip_global_m``，默认 0.5 m）。
        #
        # 2026-11-08 现场事故：这里原来比的是 ``max_lastmile_m``(2.5 m)，也就是
        # "自认为 2.5 m 内就不做全局导航，全程用 odom 相对位移硬补"。
        # 但全局导航是**唯一**靠地图/激光把位姿拉回来的环节，odom 补位只会
        # 让误差继续累加。一旦某趟补位把位姿补歪（日志里那趟 0.67 m 补了 71 s），
        # 下一趟就"自认为离终点 2.05 m"→ 又跳过全局导航 → 又 odom 硬补 →
        # 又歪。这个闭环一旦进入就出不来，表现就是"第二次检测一定失败"。
        # 所以门槛压到 0.5 m：只要真要走路，就交给规划器，顺便校正位姿。
        skip_stage1 = (not force_global) and distance_now <= self.skip_global_m

        if skip_stage1:
            print("  已经在终点 {:.2f} m 内，跳过全局导航，只补最后一点".format(distance_now))
            t0 = time.time()
            self._settle(timeout_s=4.0, quiet_s=0.8)
            timing["停稳"] = time.time() - t0
        else:
            self._set_arrival_threshold()
            self._task_active = True
            t_stage = 0.0
            t_wait = 0.0
            t_scan = time.time()

            # 候选序列：**终点本身永远第一档**，预停位只是兜底。
            #
            # 为什么不再让预检决定顺序（2026-11-08 第三次现场事故后改）：
            # 现场日志摆在那里 —— 桌子终点 `check_path_reachability` 常年返回
            # False（车贴着桌子，自己的位姿就在 0.45 m 膨胀区里，于是从它出发
            # 的任何路径都判不可达）。于是每趟都先绕预停位，而预停位那一档
            # 必然 COLLISION / 卡住 20 s，最后"直接导航到终点"才成。
            # 换句话说：三圈演示里每趟白烧 30~50 s 在一个必败的绕路上，
            # 而真正成功的那一步被排在队尾。
            #
            # ``demo_flow.py`` 那版压根没有预检、没有预停位，直接
            # ``navigate_to(name)``，一直是好的。所以就照它：先直连。
            # 预检结果降级成"提示"：只用来给万一直连失败之后的预停位排序，
            # 不再当"要不要先试终点"的闸门。
            #
            # 终点本身**连预检都不做**（2026-11-08 再改一轮）：既然预检的结论
            # 我们一律不信、一律照试，那提前查它一次就只是白花 100 ms IPC；
            # 真正想知道"可不可达"，让规划器在派活时直接回答。

            def _candidates():
                # 终点的 reachable 传 None = "没预检、不知道"，打印时据此省略提示
                direct = (goal, 0.0, 0.0, None)
                if self.direct_first:
                    yield direct
                    for pre, along, lateral, reachable in self._iter_prestop(
                            goal, goal_yaw, current):
                        yield pre, along, lateral, reachable
                    # 预停位全没成：再给终点一次机会。走到这里说明规划器
                    # 中间抖过（临时障碍/位姿刚被拉回），值得再派一次活。
                    yield direct
                else:
                    for pre, along, lateral, reachable in self._iter_prestop(
                            goal, goal_yaw, current):
                        yield pre, along, lateral, reachable
                    yield direct

            arrived = False
            status = "?"
            tried = 0
            any_reachable = False
            relocalized = 0
            refused = 0
            for target, along, lateral, reachable in _candidates():
                tried += 1
                any_reachable = any_reachable or bool(reachable)
                if time.time() - t_scan > self.navigate_budget_s:
                    print("  这一趟已经花了 {:.0f}s，超过预算 {:.0f}s，不再换点".format(
                        time.time() - t_scan, self.navigate_budget_s))
                    break
                if along == 0.0 and lateral == 0.0:
                    print("  直接导航到终点（一次派活，和 demo_flow 一样）")
                else:
                    print("  预停位后退 {:.1f} m、横移 {:+.1f} m{}".format(
                        along, lateral,
                        "" if reachable else "（预检说不可达，仍然试一次：规划器说了算）"))
                while True:
                    t0 = time.time()
                    arrived, status = self._stage1(target)
                    t_stage += time.time() - t0
                    if arrived:
                        # 这一档管用，记住它：下一趟同一个终点先试预停位。
                        # 直连成功时不必记（along=0 就是终点本身，记了也没意义）。
                        if along or lateral:
                            self._preferred[self._goal_key(goal)] = (along, lateral)
                        break
                    if status == "RELOCALIZED" and relocalized < self.relocalize_retries:
                        # 位姿被重定位了：同一个目标点重新派活即可。
                        # 这不算失败 —— 车没走错，是我们刚拿到的坐标变了。
                        relocalized += 1
                        print("  第 {} 次重新对齐后重规划同一个点".format(relocalized))
                        continue
                    break
                if arrived:
                    break
                if status in TOO_CLOSE_STATUS or status == "REFUSED":
                    # 规划器在派活那一刻就拒了，车一动没动 —— 换一档再试。
                    refused += 1
                    print("  规划器不给路（{}），换下一档".format(status))
                    # 终点 + 第一个预停位都被拒，而剩下的路 odom 补位能走完：
                    # 再换多少档都是同一张图、同一个起点，结论必然一样。
                    # 2026-11-08 真机实测：6 档全被拒，白花 49.8 s，最后是
                    # 5.7 s 的直线补位把车送到的（误差 0.037 m）。
                    now_gap = math.hypot(goal[0] - self.pose()[0], goal[1] - self.pose()[1])
                    if refused >= 2 and now_gap <= self.max_lastmile_m:
                        print("  终点和预停位都拒了（剩 {:.2f} m 在补位上限 {:.2f} m 内），"
                              "不再换点，直接交给直线补位".format(now_gap, self.max_lastmile_m))
                        break
                    t0 = time.time()
                    self._settle(timeout_s=1.5, quiet_s=0.4)
                    t_wait += time.time() - t0
                    continue
                # 其它失败也换下一档：多试几个候选比原地死磕划算
                t0 = time.time()
                self._settle(timeout_s=2.0, quiet_s=0.4)
                t_wait += time.time() - t0

            timing["第一段"] = t_stage
            timing["换点停稳"] = t_wait
            timing["预检"] = max(0.0, (time.time() - t_scan) - t_stage - t_wait)
            print("  第一段试了 {} 个点，终点{}".format(
                tried, "未到" if not arrived else "已到"))
            if not arrived and not any_reachable:
                # 这是"规划不出来"最常见的一种：不是规划器算不出来，
                # 是车自己站的这一格就已经在障碍膨胀区里（安全边距 0.45 m），
                # 于是从它出发的任何一条路径都判不可达。日志里说清楚，
                # 免得现场以为是地图坏了或者算法卡住了。
                print("  ⚠ 起点附近净空不足：静态图对**所有**终点都判不可达。")
                print("    多半是车停得离墙/桌子太近（要求 0.45 m 净空），把车往外挪半米再试；")
                print("    也可能是定位飘了 —— 对一眼地图上的实际位置。")
            if arrived:
                # 到位就只做个短确认：SDK 说到达时车已经停了，
                # 原来无条件 settle() 白等 1.6 s，三圈就是 5 s。
                t0 = time.time()
                self._settle(timeout_s=2.5, quiet_s=0.4)
                timing["停稳"] = time.time() - t0
            else:
                print("  第一段未到位，剩余距离交给直线补位（能补就补）")
                t0 = time.time()
                self._settle()          # 没到位时车可能还在滑，必须等真停稳
                timing["停稳"] = time.time() - t0

        # 只有"离终点还有一点、但还在补位能力范围内"才补。
        # SDK 已经报到 0.2 m 内时不用补 —— 那点误差手臂能吸收，
        # 而补位是纯 odom 相对位移，补一次就等于给位姿加一笔脏账。
        gap = math.hypot(goal[0] - self.pose()[0], goal[1] - self.pose()[1])
        if gap > self.arrive_tol_m:
            t0 = time.time()
            self._last_mile(goal)
            timing["补位"] = time.time() - t0
        elif gap > self.pos_tol_m:
            print("  离终点还剩 {:.2f} m，在可接受范围内，不再补位（避免弄脏位姿）".format(gap))
            timing["补位"] = 0.0
        else:
            timing["补位"] = 0.0

        t0 = time.time()
        time.sleep(0.5)
        final = self.pose()
        timing["收尾"] = time.time() - t0
        pos_err = math.hypot(goal[0] - final[0], goal[1] - final[1])
        yaw_err = wrap_deg(goal_yaw - quat_to_yaw_deg(final[3:7]))
        print("  位置误差 {:.4f} m，朝向误差 {:.2f} deg".format(pos_err, yaw_err))
        self._task_active = False
        timing["合计"] = time.time() - t_all
        self.last_timing = timing
        self._report_timing(timing)
        return pos_err, yaw_err

    @staticmethod
    def _report_timing(timing: dict) -> None:
        """把各段耗时打成一行。慢的时候一眼看出是"等规划"还是"在走路"。"""
        if not timing:
            return
        order = ("预检", "第一段", "换点停稳", "停稳", "补位", "收尾", "合计")
        parts = ["{} {:.1f}s".format(k, timing[k]) for k in order if k in timing]
        print("  [nav 耗时] " + " / ".join(parts))

    def _stop_pose(self, goal, yaw_deg, along, lateral) -> List[float]:
        rad = math.radians(yaw_deg)
        x = goal[0] - along * math.cos(rad) - lateral * math.sin(rad)
        y = goal[1] - along * math.sin(rad) + lateral * math.cos(rad)
        return pose_from_yaw(x, y, yaw_deg)

    def _stop_candidates(self):
        """预停位候选（along 由近到远，先中轴后横移）。"""
        for along in self.backoff_scan:
            yield along, 0.0
        for lateral in self.lateral_scan:
            for along in self.backoff_scan:
                yield along, lateral

    @staticmethod
    def _goal_key(goal) -> Tuple[float, float]:
        return (round(float(goal[0]), 2), round(float(goal[1]), 2))

    def _candidate_order(self, goal):
        """预停位候选的**尝试顺序**：上次成功的先上，其余由近到远。"""
        pairs = []
        preferred = self._preferred.get(self._goal_key(goal))
        if preferred is not None:
            pairs.append(preferred)
        for along, lateral in self._stop_candidates():
            if along > self.max_lastmile_m:
                continue
            key = (round(float(along), 3), round(float(lateral), 3))
            if key in pairs:
                continue
            pairs.append(key)
        return pairs

    def _iter_prestop(self, goal, goal_yaw, start):
        """惰性产出 ``(预停位, 后退, 横移, 预检是否可达)``。

        三个现场教训（2026-11-08）：

        1. **不要先全扫再动手。** 以前把 18 个候选全查一遍再挑，光预检就
           1~2 s；而绝大多数时候第一档就成。改成"边查边试"，命中即停。
        2. **预检只能当提示，不能当闸门。** 实测：把车停进障碍膨胀区后，
           从车自己的位姿出发 `check_path_reachability` 对**任何**终点都返回
           False（正前方 0.8/2/4 m、身后 1/2 m 全 False），换它周围 1 m 的
           虚拟起点同一批终点就 True。这时候直接 `raise` 就是"规划不出来"，
           其实是我们在派活之前先放弃了。所以：可达的先试，全不可达就把
           最近的那一档照样交给 `navigate_to_goal` —— 真规划器有动态障碍
           和局部代价地图，它比这个静态全图预检更权威。
        3. 预检一次约 100 ms，所以给它 6 次的上限（
           ``MAX_PRESTEP_TRIES``），够找出一个能走的点。
        """
        if self.backoff_m > 0.0:
            # 现场用 --backoff 点名了预停位：只试这一个，但也不预先中止，
            # 预检说不行也交给规划器判（理由见下）。
            pre = self._stop_pose(goal, goal_yaw, self.backoff_m, 0.0)
            yield pre, self.backoff_m, 0.0, self._reachable(pre, start)
            return

        seen = set()
        fallback = None
        probes = 0
        for along, lateral in self._candidate_order(goal):
            pre = self._stop_pose(goal, goal_yaw, along, lateral)
            key = (round(pre[0], 3), round(pre[1], 3))
            if key in seen:
                continue
            seen.add(key)
            if probes >= MAX_PRESTEP_TRIES:
                break
            probes += 1
            ok = self._reachable(pre, start)
            if ok:
                yield pre, along, lateral, True
            elif fallback is None:
                fallback = (pre, along, lateral, False)
        if fallback is not None:
            # 一个可达的都没有：把最近那档递上去，让规划器自己给结论
            yield fallback

    def _prestop_candidates(self, goal, goal_yaw, start):
        """旧接口：返回全部可达预停位（离线回归/排查用，导航不再走这里）。"""
        return [(pre, along, lateral)
                for pre, along, lateral, ok in self._iter_prestop(goal, goal_yaw, start)
                if ok]

    def _reachable(self, goal, start) -> bool:
        """预检一次路径。**只是提示**：查不动就当作"不知道"，返回 True 放行。

        查不动（IPC 抖动/SDK 内部异常）时如果抛出去，整趟导航就废了；
        而真正的判定在 `navigate_to_goal` 里，让它去说是更靠谱的做法。
        """
        try:
            return bool(self._nav.check_path_reachability(goal, start))
        except Exception as exc:                       # noqa: BLE001
            print("  路径预检失败（忽略，交给规划器）：{}".format(exc))
            return True

    def _set_arrival_threshold(self) -> None:
        if not self.arrival_threshold:
            return
        try:
            self._nav.set_navigation_arrival_threshold(list(self.arrival_threshold))
        except Exception as exc:
            print("  到位判据未改：{}".format(exc))

    def _stage1(self, target) -> Tuple[bool, str]:
        """跑第一段。返回 (是否到位, 结束状态名)。

        同一个目标点**最多派两次活**（2026-11-08 改）：
        1. 正常派一次；
        2. 只有"卡住"（``STALLED``，车确实没挪窝）才原地再派一次 ——
           这相当于给 SDK 的规划器踢一脚，现场日志里两次都靠它救回来了；
        3. 其它任何失败（``COLLISION``/``OCCUPIED``/``CLOSE_TO_OBSTACLE``）
           立刻返回，交给上层换下一档候选。

        为什么删掉原来那层 ``stage1_retries``（"第一段重试 1/1"）：
        对同一个点重派，规划器拿着同一张图、同一个起点，结论当然一样。
        现场日志里每一次"重试 1/1"都是同样的 COLLISION，白花 8 s 才往下走。
        外层本来就有 5 档候选（终点直连 + 3 个预停位 + 再试终点），
        把时间留给换点，比在同一格上磕两次划算。

        ``REFUSED``（2026-11-08 真机日志后加）：``navigate_to_goal`` 是**同步**
        返回 ``(ok, msg)`` 的，规划器不给路时它当场就回 ``False``。原来我们不看
        这个返回值，于是还要等 8 s 的卡死看门狗才认输 —— 现场 6 档候选白白吃掉
        49.8 s，最后是 5.7 s 的直线补位把车送到的。现在当场拒就当场返回。
        """
        status = "?"
        stall_retried = False
        # 记下当前这一段的目标点：_wait_arrival 用它判断"是不是已经站进容差了"
        self._current_goal = [float(value) for value in target]
        while True:
            self._announce_move()
            self._forget_pose()           # 重新派活，位姿基线重取
            try:
                accepted, message = self._nav.navigate_to_goal(
                    target,
                    enable_collision_check=True,
                    is_blocking=False,
                    timeout=self.timeout_s,
                    omni_plan=False,
                )
            except Exception as exc:                      # noqa: BLE001
                print("  派活失败（{}）：{}".format(type(exc).__name__, exc))
                return False, "DISPATCH_FAILED"
            if accepted is False:
                # 规划器连任务都没接（COLLISION / 目标在膨胀区 / 车自己贴障碍）。
                # 车一动没动，等下去也没有意义。
                print("  ✗ 规划器当场拒绝：{}".format(message))
                return False, "REFUSED"
            arrived, elapsed, status = self._wait_arrival("第一段")
            if arrived:
                return True, "ARRIVED"
            self._stop()
            if status in TOO_CLOSE_STATUS:
                return False, status      # 这个点规划器不收，让上层换更远的
            if status == "RELOCALIZED":
                return False, status      # 交给上层用新坐标重规划
            if status == "STALLED" and not stall_retried:
                # 卡住多半是前面站了人/临时挡了一下，或 SDK 内部规划卡了。
                # 原地重规划一次通常就好，比干等 150 s 超时强太多。
                stall_retried = True
                print("  卡住了，原地重新派一次活（不换点）")
                self._settle(timeout_s=2.0, quiet_s=0.4)
                continue
            return False, status

    def _wait_arrival(self, label: str) -> Tuple[bool, float, str]:
        """等 SDK 把这一段走完。**只认"到了"，不认中途的瞬时状态**。

        2026-11-08 现场（送 route_b 之后第二趟）：
        ``4.0s status=FAILED`` 我们就停车重试了，可车其实还在走 ——
        ``6.0s`` 已经跑到 1 m 外、``8.0s`` 到 1.9 m 外。那一次 ``_stop()``
        不但白停，还把 SDK 的一次正常重规划打成了"两段式"。

        所以现在：
        - ``check_goal_arrival()`` 为真 -> 到；
        - 真位姿落进容差 -> 也到（SDK 判据是 0.2 m，我们有时比它先看到）；
        - 状态非 RUNNING 要**连续持续** ``status_grace_s`` 且这段时间车没挪窝，
          才认输（重规划瞬间的假 FAILED 过不了这一关）；
        - 位姿跳变 -> 返回 ``RELOCALIZED``，让上层拿新坐标重新规划，
          **不是**拿跳变后的距离判死刑。

        ``REFUSED``（2026-11-08 真机探针后加）：派活那一下 SDK 回
        ``(True, 'SUCCESS')``，可 0.5 s 后状态就已经是 ``FAILED``、车一动不动 ——
        这是"规划器给了个跑不起来的任务"，和"走着走着抖一下"完全是两回事。
        区别就在**车动没动**：动过才需要 6 s 的宽容期；一动没动又已经报错，
        ``refuse_grace_s`` 一到就认，别白等 8 s 的卡死看门狗。
        """
        started = time.time()
        last_print = 0.0
        last_status_at = 0.0
        origin = self.pose()
        anchor = origin
        anchor_at = started
        goal_pose = getattr(self, "_current_goal", None)
        bad_since = None
        moved_away = False
        name = "?"
        while time.time() - started < self.timeout_s:
            if self._nav.check_goal_arrival():
                return True, time.time() - started, "ARRIVED"
            now = time.time()
            current = self.pose()
            if self._take_jump():
                self._forget_pose()
                return False, now - started, "RELOCALIZED"
            # 已经站在容差里了：不必等 SDK 那句 ARRIVED（它按 0.2 m 算，
            # 我们只要够抓就行）。这一条让"最后 3 cm"不再反复补位。
            if goal_pose is not None:
                gap = math.hypot(current[0] - goal_pose[0], current[1] - goal_pose[1])
                yaw_gap = abs(wrap_deg(quat_to_yaw_deg(goal_pose[3:7])
                                       - quat_to_yaw_deg(current[3:7])))
                if gap <= self.pos_tol_m and yaw_gap <= self.yaw_tol_deg:
                    return True, now - started, "ARRIVED_IN_TOL"
            # 卡死看门狗：既没位移也没明显转向，就别再干等 SDK 的 150 s
            stuck = (
                math.hypot(current[0] - anchor[0], current[1] - anchor[1]) < 0.10
                and abs(wrap_deg(quat_to_yaw_deg(current[3:7])
                                 - quat_to_yaw_deg(anchor[3:7]))) < 5.0
            )
            if not stuck:
                anchor, anchor_at = current, now
            if math.hypot(current[0] - origin[0], current[1] - origin[1]) > 0.12:
                moved_away = True
            if stuck and now - anchor_at > self.stall_timeout_s:
                print("    {:5.1f}s {} 卡住 {} s 没挪窝，停车重规划".format(
                    now - started, label, int(now - anchor_at)))
                return False, now - started, "STALLED"
            # 状态单独按 0.5 s 采样：拒绝要认得早，日志仍按 2 s 打
            if now - last_status_at >= 0.5:
                last_status_at = now
                status = self._nav.get_navigation_status()
                name = getattr(status, "name", str(status))
                if name in ("RUNNING", "UNKNOWN", "?"):
                    bad_since = None
                elif bad_since is None:
                    # 不立刻认输：重规划那一瞬间也会报 FAILED。
                    bad_since = now
                elif not moved_away:
                    # 一动没动 + 一直在报错 = 这个目标规划器跑不了，早收手
                    if now - bad_since >= self.refuse_grace_s:
                        print("    {:5.1f}s {} 状态 {} 且车一动没动（{:.2f} m），"
                              "判规划器给不出路".format(
                                  now - started, label, name,
                                  math.hypot(current[0] - origin[0],
                                             current[1] - origin[1])))
                        return False, now - started, "REFUSED"
                elif now - bad_since >= self.status_grace_s and stuck:
                    # 车动过之后再报错：宽容期要长（重规划瞬间的假 FAILED）
                    print("    {} 状态 {} 已经持续 {:.0f}s 且车没动，认输".format(
                        label, name, now - bad_since))
                    return False, now - started, name
            if now - last_print > 2.0:
                last_print = now
                print(
                    "    {:5.1f}s {} x={:.3f} y={:.3f} yaw={:.1f} status={}".format(
                        now - started,
                        label,
                        current[0],
                        current[1],
                        quat_to_yaw_deg(current[3:7]),
                        name,
                    )
                )
            time.sleep(0.2)
        return False, self.timeout_s, "TIMEOUT"

    def _settle(self, timeout_s: float = 12.0, quiet_s: float = 1.2) -> List[float]:
        started = time.time()
        anchor = self.pose()
        anchored_at = started
        while True:
            time.sleep(0.4)
            now = time.time()
            current = self.pose()
            moved = (
                math.hypot(current[0] - anchor[0], current[1] - anchor[1]) > 0.02
                or abs(wrap_deg(quat_to_yaw_deg(current[3:7]) - quat_to_yaw_deg(anchor[3:7]))) > 1.0
            )
            if moved:
                anchor = current
                anchored_at = now
            elif now - anchored_at >= quiet_s:
                return current
            if now - started > timeout_s:
                return current

    def _last_mile(self, goal) -> bool:
        goal_yaw = quat_to_yaw_deg(goal[3:7])
        stalls = 0
        bad_yaw = 0
        for index in range(self.lastmile_tries):
            before = self.pose()
            if self._take_jump():
                # 位姿跳变之后，"剩余多少米"这个数已经没意义了，别拿它开车
                print("    补位中止：位姿刚跳过，剩余距离不可信")
                return False
            distance = math.hypot(goal[0] - before[0], goal[1] - before[1])
            yaw_error = wrap_deg(goal_yaw - quat_to_yaw_deg(before[3:7]))
            if distance <= self.pos_tol_m and abs(yaw_error) <= self.yaw_tol_deg:
                return True
            if distance > self.max_lastmile_m:
                # 这里**不能 raise**：一次定位跳变就能把"剩余 6.32 m"算出来，
                # 抛出去等于整趟任务直接废掉，还会跟用户说"请把我推到桌前"。
                # 补位本来就只是收尾的微调手段，不该拥有判死刑的权力 ——
                # 交给上层（navigate_to_pose 的自愈）重新走全局导航才是对的。
                print("    剩余 {:.2f} m 超过补位上限 {:.2f} m，不补位"
                      "（交给全局导航重新对齐）".format(distance, self.max_lastmile_m))
                return False
            relative = relative_pose(before, goal)
            magnitude = math.hypot(relative[0], relative[1])
            rotate_first = abs(yaw_error) > self.rot_first_deg
            if rotate_first:
                travel = self.rot_creep_m
                step_yaw = _clamp(yaw_error, self.max_rot_step_deg)
            else:
                # 自适应步长：离得远就迈大一点，近了自动收小。
                # 原来固定 0.2 m，1.5 m 要跑 8 步（每步一次阻塞调用 + 一次位姿回读），
                # 光补位就十几秒；按剩余距离减半取值，一般 3~4 步收工。
                step = max(self.lastmile_step_m, min(self.max_step_m, magnitude * 0.5))
                if abs(yaw_error) > self.yaw_tol_deg:
                    # 还要带角度修正时步子不放大 —— 边平移边甩角会把轨迹拉成螺旋
                    step = min(step, self.lastmile_step_m)
                if (
                    index == 0
                    and abs(yaw_error) <= self.yaw_tol_deg
                    and magnitude > step
                ):
                    step = max(step, min(self.first_step_m, magnitude))
                travel = min(step, magnitude)
                step_yaw = 0.0 if abs(yaw_error) <= self.yaw_tol_deg else _clamp(yaw_error, self.max_rot_step_deg)
            if magnitude > 1e-6:
                scale = travel / magnitude
                relative[0] *= scale
                relative[1] *= scale
            else:
                relative[0], relative[1] = travel, 0.0
            half = math.radians(step_yaw) * 0.5
            relative[5], relative[6] = math.sin(half), math.cos(half)
            self._announce_move()
            self._nav.move_straight_to(relative, is_blocking=True, timeout=self.straight_timeout_s)
            changed, after = self._wait_pose_change(before)
            if not changed:
                # 一条 odom 相对指令发下去，位姿却**一点没变** —— 要么车被挡死，
                # 要么位姿本身就是错的。两种情况都不可能靠再发一条指令解决，
                # 继续循环只会白等（2026-11-08：0.67 m 补了 71 s）。
                # 立刻收手，交给上层用全局导航重新对齐。
                stalls += 1
                print("    补位第 {} 步：车没动（被挡或位姿已偏），停止补位".format(index + 1))
                break
            actual_yaw = wrap_deg(quat_to_yaw_deg(after[3:7]) - quat_to_yaw_deg(before[3:7]))
            if abs(wrap_deg(actual_yaw - step_yaw)) > self.yaw_overshoot_deg:
                bad_yaw += 1
                if bad_yaw >= 2:
                    break
            else:
                bad_yaw = 0
            remaining = math.hypot(goal[0] - after[0], goal[1] - after[1])
            if not rotate_first and remaining > distance - 0.01:
                stalls += 1
                if stalls >= 3:
                    break
            else:
                stalls = 0
        final = self.pose()
        distance = math.hypot(goal[0] - final[0], goal[1] - final[1])
        yaw_error = wrap_deg(goal_yaw - quat_to_yaw_deg(final[3:7]))
        return distance <= self.pos_tol_m and abs(yaw_error) <= self.yaw_tol_deg

    def _wait_pose_change(self, before) -> Tuple[bool, List[float]]:
        started = time.time()
        while time.time() - started < self.pose_wait_s:
            current = self.pose()
            moved = (
                math.hypot(current[0] - before[0], current[1] - before[1]) > 0.005
                or abs(wrap_deg(quat_to_yaw_deg(current[3:7]) - quat_to_yaw_deg(before[3:7]))) > 0.5
            )
            if moved:
                return True, current
            time.sleep(0.25)
        return False, self.pose()

    def _stop(self) -> None:
        try:
            self._nav.stop_navigation()
        except Exception:
            pass
        self._task_active = False
