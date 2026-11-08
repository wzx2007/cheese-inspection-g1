"""帕马森奶酪演示（主程序）。流程见 scripts/目标.md。

三次抓取，每次都是**完整的一轮抓放**，阶段名和 ``galbot_nav`` 原管线一致，
只多插两个"查看"：

    gripper_open -> arm_reset -> height_ready -> [look_front]
    -> arm_ready -> reach_forward -> grasp_close -> lift -> flip
    -> [look_back] -> 分流判定 -> unflip -> [导航到终点 A/B] -> lower -> release -> arm_reset2

2026-11-08 分流改造（用户要求）
-------------------------------
以前不管结论是什么，每轮都把奶酪**放回原处**。现在两面都看完的那一刻就判一次：
要走分流（个数 1 -> 终点 A / 2 -> 终点 B）的话，**保持夹持**，翻正后直接把车
开到终点，到位后才 ``lower -> release`` 放下。等于把"松爪"这一步从检测桌搬到了
终点桌，其余动作完全复用 ``release_here()`` 那一套。

只有最后一圈这么走（多圈时前面几圈照旧放回原位）；判定保持不动（0 个霉、
终点没配）或者 ``CARRY_TO_GOAL=False`` 时，行为与以前一模一样。导航失败也会
就地放下 —— 绝不能让机器人攥着奶酪待命。

2026-11-07 现场返修：之前把 lower / release / arm_reset2 挪到了三次循环之外，
于是机器人抱着 50 cm 奶酪跑回起点再跑回桌边，等于只抓了一次、看了六面。
现在松开、回高举都放回每一轮里，和原管线逐阶段对齐。

2026-11-07 二改（现场）：**不再自己开回起点**。每抓完一次就打住，由人把车
推到下一个起点，按回车才继续 —— 全自动模式也卡这一道。展馆地面人来人往，
自主返航又慢又容易被安全边距顶住（``CLOSE_TO_OBSTACLE``），人推两秒钟
比机器绕半分钟靠谱。

2026-11-08 三改（用户要求，把 ``demo_flow.py`` 的收尾搬进来）：分完类、放下
奶酪之后，车自己走完原脚本那三段 —— ① 空手开回检测桌接下一块；② 原地转
180° 面朝观众谢幕；③ 开回巡检起点归位。见 ``closing_moves()``。

三段的触发条件跟"识别结果"绑定，这就是"结果不同、走法不同"：

===========  ======================  ==========================
判定          搬不搬                  收尾移动
===========  ======================  ==========================
0 个霉        不搬（放回检测桌原位）  一步都不动（车还在工作位）
1 个霉        抱去终点 A 再放下       转 180° → 回巡检起点
2 个霉        抱去终点 B 再放下       转 180° → 回巡检起点
===========  ======================  ==========================

没搬的那趟之所以不动，是因为车本来就在检测桌的正确工作位上；转身再开走会
把"下一句话要用的位置"丢掉。转身用 ``ChassisNavigator.turn_in_place`` 做
纯旋转 —— ``demo_flow`` 那种"同位姿 + 180°"的画圆走法会漂 0.4 m。
语音模式下这段在**播完最终结论之后**才走（``closing_hook``），先开走再说话
观众会等一段莫名其妙的沉默；键盘演示直接走。

两个查看的位置也是有讲究的，而且都**不额外升降高度**（高度正是 2026-11-07
那次"反面没识别到"的根因）：
* ``look_front`` 在**抓取之前**，身高 1.37 m、双臂高举，相机俯视桌上奶酪的平面 ——
  就是原管线 0~5 帧那个窗口，平面对相机、曝光正常，判据最稳。
* ``look_back`` 在 ``flip`` 之后，**留在抬起高度 1.54 m**，就是原管线 200~205 帧。
  之前先降到 1.34 m 再看，平面被压成一条 30 px 的窄带还被高光冲白，
  霉菌的 Black-Hat 深度从 192 掉到 65，直接判不出来。

视觉（scripts/vision）：两次"查看"各连拍 6 帧判霉菌，**每判完一面就立刻播报
这一面的结果** —— 按"哪一面 × 正常/不正常"四选一：
正面正常 / 正面不正常 / 反面正常 / 反面不正常（``front_ok.wav`` 等四支）；
三次全跑完（含收身）再放 ``goodbye.wav``。判据见 vision/config.py。

播报是非阻塞的（``voice/play_clip.py`` 的 VoiceBox 用队列 + 后台线程），
播报的同时手臂继续走动作，不会把演示拖长。音频是 24 kHz（旧两支 32 kHz），
SDK 要 16 kHz / 16 bit / 单声道，VoiceBox 会先转 PCM 并缓存。

双手一起动：左臂关节是右臂逐项取负。躯干保持直立，只升降和转腕。
改参数只改下面的常量，动作是按顺序的函数调用。

    export PYTHONPATH=/data/galbot/lib:$PYTHONPATH
    export LD_LIBRARY_PATH=/data/galbot/lib:$LD_LIBRARY_PATH
    python3 demo_cheese.py --confirm-motion
    python3 demo_cheese.py --confirm-motion --debug
    python3 demo_cheese.py --confirm-motion --no-vision     # 只做动作，不判霉菌
    python3 demo_cheese.py --confirm-motion --no-voice      # 判但不播报

--debug 时每步回车执行，q 停止。"换下一块"那一道回车在任何模式下都有
（``ask_after_last=False`` 的语音单块流程除外）。失败或停止后不自动回位。
Ctrl+C 不等于急停。
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from pathlib import Path

from armctrl.arm_controller import RightArmController
from armctrl.sdk_arm_planner import SdkArmPlanner
from legctrl.leg_controller import LegController
from mission.route import RouteMachine, count_from_inspector, count_mold
from navctrl.chassis_navigator import ChassisNavigator, quat_to_yaw_deg, wrap_deg
from voice.play_clip import VoiceBox

# ---------------------------------------------------------------------------
# 现场参数。长度 m，末端关节 rad。
# 半间距和身高都还没按 50 cm 奶酪标定。
# ---------------------------------------------------------------------------
READY_HEIGHT_M = 1.37
HEIGHT_HIGH_M = 1.54
HEIGHT_END_M = 1.32
HUG_GAP_M = 0.28
RELEASE_GAP_M = 0.40
END_JOINT_RAD = 1.538
FORWARD_X_M = 0.05
GRIPPER_OPEN_M = 0.12
TOOL_NAME = "galbot_gripper"
GOAL_NAME = "table"
LOOP_COUNT = 3
_HERE = Path(__file__).resolve().parent

# 两个"查看"都**不额外升降**，直接落在标定过的位姿上：
#   正面 = 抓取之前、身高 READY_HEIGHT_M(1.37)、双臂高举，相机俯视桌上奶酪的平面
#          对应原管线 0~5 帧那个窗口
#   反面 = 翻腕之后、身高 HEIGHT_HIGH_M(1.54)，对应原管线 200~205 帧
# 2026-11-07 现场教训：反面那次曾经先降到 1.34 m 再看，平面被压成 30 px 窄带、
# 又被顶灯高光冲白，霉菌的 Black-Hat 深度从 192 掉到 65，判据直接判不出。
# 别为了"看清楚"去降高度，那反而看不见。

# 视觉与声音
USE_VISION = True
# 2026-11-08 第二版：播报按"哪一面 × 正常/不正常"四选一（对齐演示流程文档里
# demo_flow 的台词）。旧的 pass/fail 两支台词是"检测完成，表面未发现明显异常"，
# 根本不提正反面，6 个面连续播报时听不出说的哪一面，已停用（文件留在 voice/ 里）。
# 台词已用 DashScope ASR 逐支核对（2026-11-08）：
#   front_ok.wav  = "奶酪状态，正面正常。"
#   front_bad.wav = "奶酪状态，正面不正常。"
#   back_ok.wav   = "奶酪状态，反面正常。"
#   back_bad.wav  = "奶酪状态，反面不正常。"
# 四支都是同一把云端音色（cosyvoice longxiaochun_v3）预生成的，24 kHz；
# SDK 只吃 16 kHz/16bit/单声道，VoiceBox 会转 PCM 并缓存。
# check_done.wav = "检查完成。"（流程文档第 12 步，demo_flow 也是这一句）
VOICE_FILES = (
    ("front_ok", "front_ok.wav"),      # 正面 + 没霉
    ("front_bad", "front_bad.wav"),    # 正面 + 有霉
    ("back_ok", "back_ok.wav"),        # 反面 + 没霉
    ("back_bad", "back_bad.wav"),      # 反面 + 有霉
    ("check_done", "check_done.wav"),  # 三轮跑完："检查完成。"（对齐流程文档）
    ("goodbye", "goodbye.wav"),        # 然后道别
)
VOICE_GOODBYE = "goodbye"
VOICE_CHECK_DONE = "check_done"

# 三次跑完之后按"有几个面有霉"分流到不同终点（2026-11-08 新增）。
# 0 -> 不动 / 1 -> 终点 A / 2 -> 终点 B。坐标在 mission/route.py 开头，
# 现在还是 None —— 留空时状态机只会打印"终点没配"并保持不动，不会瞎跑。
ROUTE_COUNT_MODE = "faces"       # faces = 有霉的面数；spots = 单次菌斑数最大值

# 2026-11-08 用户要求：**判定要走分流时不要放回原位**，而是保持抓取、抱着奶酪
# 直接开到终点 A/B，到位之后再松爪放到终点的桌面上。
#
# 原来的动作是"翻回去 -> 降高度 -> 松手放回原处"，也就是每次结论出来以后都把
# 奶酪放回桌子中央；现在把"松手"这一步推迟到终点：
#
#   原：unflip -> lower -> release -> arm_reset2（全程在检测桌）
#   新：unflip -> switch_leg -> 导航到 A/B -> level_leg -> lower -> release -> arm_reset2
#
# 只有"最后一圈且判定为要动"才这样走；判定保持不动（0 个霉、终点没配）时
# 仍然放回原位，跟以前一模一样。设成 False 可一键退回旧行为。
# 2026-11-08：`demo_flow.py` 里的收尾移动搬进来了。那个脚本的三段动作是
#   ① back_to_table : 送完（放下）之后空手开回检测桌，接下一块
#   ② turn_half     : 原地转 180°，面朝观众"谢幕"
#   ③ return origin : 开回巡检起点，归位
# 原脚本是固定 3 块的键盘演示，这三段写死在它的 run() 里；这里拆成独立函数，
# 由 demo_cheese.run() 按"到底搬没搬奶酪"和"还有没有下一块"来编排。
RETURN_TO_TABLE = True       # 送完空手回检测桌（只在该圈后面还有块时做）
HOME_ON_FINISH = True        # 收尾转身 + 开回巡检起点
HOME_DIST_TOL_M = 0.35       # 距起点这么近就不用再开一趟
TURN_HALF_DEG = 180.0


CARRY_TO_GOAL = True


def resolve_goal_dir(goal_name):
    filename = "{}.json".format(goal_name)
    candidates = [
        _HERE / "nav_goals",
        _HERE.parent / "nav_goals",
        Path("/userdata/galbot_nav/nav_goals"),
    ]
    for path in candidates:
        if (path / filename).is_file():
            return path
    return candidates[-1]


class DemoStopped(RuntimeError):
    """操作员在调试单步时选择停止。"""


def pause(debug, title):
    if not debug:
        return
    answer = input("下一步：{}  [回车执行 / q 停止] ".format(title))
    if answer.strip().lower() in {"q", "quit"}:
        raise DemoStopped(title)


def open_both_grippers(robot, width_m=GRIPPER_OPEN_M):
    from galbot_sdk.g1 import ControlStatus, G1JointGroup

    if not 0.0 <= width_m <= 0.12:
        raise ValueError("G1 夹爪宽度必须在 0–0.12 m 之间")
    for group in (G1JointGroup.left_gripper, G1JointGroup.right_gripper):
        status = robot.set_gripper_command(group, width_m, 0.05, 10.0, True)
        if status != ControlStatus.SUCCESS:
            raise RuntimeError("打开 {} 失败：status={!r}".format(group, status))


def switch_leg(robot):
    from galbot_sdk.g1 import ControlStatus, G1ControllerName

    status = robot.switch_controller(G1ControllerName.LEG_PVT_CTRL)
    if status != ControlStatus.SUCCESS:
        raise RuntimeError("switch_controller(LEG_PVT_CTRL) 返回 {!r}".format(status))


def stage(debug, name, title):
    """打一行阶段名再（调试时）等回车。阶段名和 galbot_nav 原管线一致。"""
    print("  [{:<13}] {}".format(name, title))
    pause(debug, title)


def release_here(arm, leg, debug, unflip=True):
    """把奶酪放回**当前**位置：翻正 -> 降高度 -> 松爪 -> 双臂回高举。

    抽成函数是为了让"放回检测桌"和"放到终点桌面"共用同一套动作 —— 用户明确
    要求终点那一下"完全复制翻回来的动作"，两处若各写一份迟早会走样。

    ``unflip=False`` 用于搬运路径：出发前已经翻正并一直夹持着，到终点再转一次
    腕是空行程（还要等一次停稳），直接进降高度更省时间。
    """
    if unflip:
        stage(debug, "unflip", "转回 {:.3f} rad".format(END_JOINT_RAD))
        set_end_joint_stable(arm, END_JOINT_RAD)

    stage(debug, "lower", "降回 {:.2f} m".format(READY_HEIGHT_M))
    leg.control_height(READY_HEIGHT_M)

    stage(debug, "release", "松手，半间距 {:.2f} m".format(RELEASE_GAP_M))
    arm.set_half_gap(RELEASE_GAP_M)

    stage(debug, "arm_reset2", "双臂回高举位（空手）")
    arm.reset_both_arm()


def carry_to_goal(robot, arm, leg, nav, router, decision, debug=False):
    """**抓着奶酪**开到终点，到位后再放下。

    返回 ``(route_state, released_ok)``。``released_ok=False`` 表示导航没走成，
    奶酪被就近放回了原地（绝不能让机器人一直攥着奶酪待命）。

    顺序上的两个讲究：

    * 翻正（``unflip``）在**开车之前**做。抱着翻过来的腕关节满场跑，腕上
      负载偏心，且到终点还要多等一次大行程转腕；先把姿态收回标定值更稳。
    * 放下的高度沿用 ``READY_HEIGHT_M``（检测桌的标定高度）。终点 A/B 是
      同一批桌子，用户给的点也只有 XY+yaw，没有桌面高度信息 —— 这里不自己
      编一个高度出来。
    """
    stage(debug, "unflip_carry", "翻正（仍夹持），准备抱走")
    set_end_joint_stable(arm, END_JOINT_RAD)

    # 导航必须切到腿部控制器，和每圈开头那次导航一致
    stage(debug, "carry_switch", "切腿部控制器，抱着奶酪去 {}".format(
        decision.goal_name))
    switch_leg(robot)

    state = router.execute(decision)
    ok = state == "DONE"
    if not ok:
        print("  [携带] ✗ 没送到 {}（{}），就近放下".format(decision.goal_name, state))

    # 到点（或没到）都要先回正俯仰再动升降，否则身子是斜的，放下去会歪
    stage(debug, "carry_level", "切回腿部升降并回正")
    switch_leg(robot)
    level_leg(leg)
    report_leg(leg)

    release_here(arm, leg, debug, unflip=False)
    return state, ok


def _safe_navigate(nav, target, label, timeout_note=""):
    """开过去，失败只告警不抛 —— 收尾移动绝不能把"这次检测的结论"打断。

    2026-11-08 的教训：导航失败（``goal unreachable`` / ``CLOSE_TO_OBSTACLE``）
    在现场是常态，尤其桌子四周净空不足的时候。收尾走不动是"演示打折"，
    但把异常抛上去会让整趟汇报丢掉，那才是真事故。
    """
    try:
        pos_err, yaw_err = nav.navigate_to_pose(target)
        print("  [收尾] {} 到位（误差 {:.3f} m / {:+.1f}°）".format(
            label, pos_err, yaw_err))
        return True
    except Exception as exc:                              # noqa: BLE001
        print("  [收尾] {} 没走成：{}{}".format(label, exc, timeout_note))
        return False


def return_to_table(robot, leg, nav, table_pose, debug=False):
    """送完奶酪、空手开回检测桌，准备接下一块（``demo_flow.back_to_table``）。

    只在"这一圈后面还有块要测"时调用。单块流程测完就结束了，桌上没有下一块，
    再开回检测桌纯属多跑一趟。
    """
    stage(debug, "back_table", "空手开回检测桌")
    switch_leg(robot)
    ok = _safe_navigate(nav, table_pose, "回检测桌")
    switch_leg(robot)
    level_leg(leg)
    report_leg(leg)
    return ok


def turn_half(robot, leg, nav, debug=False):
    """原地转 180°，面朝观众（``demo_flow.turn_half``）。

    转身本身不改位置：``ChassisNavigator.turn_in_place`` 走的是纯旋转的
    odom 相对位姿，不用全局导航 —— 后者会让"转 180°"变成画 0.4 m 的圆。
    """
    stage(debug, "turn_half", "原地转 {:.0f}°".format(TURN_HALF_DEG))
    switch_leg(robot)
    try:
        turned = nav.turn_in_place(TURN_HALF_DEG, steps=3)
    except Exception as exc:                              # noqa: BLE001
        print("  [收尾] 转身没做成：{}".format(exc))
        turned = 0.0
    switch_leg(robot)
    level_leg(leg)
    report_leg(leg)
    return turned


def home_to_origin(robot, leg, nav, origin, debug=False):
    """开回巡检起点，归位（``demo_flow.run`` 最后那句 ``navigate_to_pose(origin)``）。

    已经在起点附近就不动 —— 单块流程里起点通常就是检测桌，回完桌已经在
    原位了，再开一模一样的一趟没有意义。
    """
    if origin is None:
        return False
    try:
        here = nav.pose()
    except Exception as exc:                              # noqa: BLE001
        print("  [收尾] 读位姿失败，跳过归位：{}".format(exc))
        return False
    distance = math.hypot(origin[0] - here[0], origin[1] - here[1])
    if distance <= HOME_DIST_TOL_M:
        print("  [收尾] 已经在起点 {:.2f} m 内，不用再开一趟".format(distance))
        return True
    stage(debug, "home", "开回巡检起点（{:.2f} m）".format(distance))
    switch_leg(robot)
    ok = _safe_navigate(nav, origin, "回巡检起点")
    switch_leg(robot)
    level_leg(leg)
    report_leg(leg)
    return ok


def closing_moves(robot, arm, leg, nav, origin, table_pose, carried, debug=False,
                  another_piece=False):
    """分类之后的整套移动，照 ``demo_flow.py`` 的顺序。

    ::

        搬过奶酪? --no--> 什么都不做（车就在检测桌原位）
            |
           yes
            v
        还有下一块? --yes--> 空手开回检测桌
            |
            v
        原地转 180°（谢幕）
            |
            v
        开回巡检起点（已经在附近就跳过）

    ``carried`` 必须是"这一趟真的把奶酪抱到终点并放下了"。没搬过奶酪的话
    车一直在检测桌原位，转身/归位都会把它从工作位带走，下一句话就没法测了。
    """
    if not carried:
        return
    # 后面还有块：只开回检测桌接料，**到此为止**。转身谢幕和归位是整趟结束的
    # 动作，中间圈做了下一块就没法测了（车背对桌子还站在起点）。
    if another_piece and RETURN_TO_TABLE:
        return_to_table(robot, leg, nav, table_pose, debug=debug)
        return
    if not HOME_ON_FINISH:
        return
    turn_half(robot, leg, nav, debug=debug)
    home_to_origin(robot, leg, nav, origin, debug=debug)


def wait_still(arm, timeout=8.0, tol=2e-3, quiet=0.4):
    """等双臂真正停住再发下一条指令。

    连发关节指令时，若上一段大动作还没走完，``traj_plan`` 的起点和真实状态
    对不上会直接 FAULT。原管线就是这么处理的。
    """
    try:
        prev = arm.state_snapshot()
    except Exception:  # noqa: BLE001
        time.sleep(quiet)
        return quiet
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(quiet)
        try:
            cur = arm.state_snapshot()
        except Exception:  # noqa: BLE001
            return time.time() - t0
        still = (
            all(abs(a - b) < tol for a, b in zip(prev.joints, cur.joints))
            and all(abs(a - b) < tol for a, b in zip(prev.left_joints, cur.left_joints))
        )
        if still:
            return time.time() - t0
        prev = cur
    return timeout


def set_end_joint_stable(arm, rad, attempts=3):
    """转腕的稳版：失败就先等停稳再重试，别一失败就整场退出。"""
    last = None
    for attempt in range(attempts):
        try:
            arm.set_end_joint(rad)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            print("  第 {} 次 set_end_joint({:+.3f}) 失败：{}".format(
                attempt + 1, rad, exc))
            if attempt + 1 < attempts:
                waited = wait_still(arm)
                print("  等 {:.1f} s 停稳后重试".format(waited))
    raise last


def raise_and_stand(arm, leg):
    level_leg(leg)
    arm.reset_both_arm()


def level_leg(leg, fallback_height_m=READY_HEIGHT_M):
    """把躯干回正到 0°。

    2026-11-07 现场：机器人停在 1.5627 m、俯仰 -0.58°。这个高度已经比
    ``HEIGHT_MAX_M``(1.54) 还高 2 cm，pitch=0 的腿部逆解无解，``reset_pitch()``
    直接抛 "no feasible leg pose for ee_z=1.1856 m"。实测 0.96~1.55 m 都可解，
    差的就是这 2 cm —— 所以回正失败时先落到标定身高，再回正。
    """
    try:
        leg.reset_pitch()
        return
    except Exception as exc:  # noqa: BLE001
        print("  回正俯仰失败：{}".format(exc))
    try:
        state = leg.read_state()
        print("  当前头高 {:.3f} m、俯仰 {:+.2f}°，先到 {:.2f} m 再回正".format(
            state.height_m, state.pitch_deg, fallback_height_m))
    except Exception:  # noqa: BLE001
        pass
    leg.control_height(fallback_height_m)
    leg.reset_pitch()


def report_leg(leg):
    try:
        state = leg.read_state()
        print("  当前头高 {:.3f} m、俯仰 {:+.2f}°".format(state.height_m, state.pitch_deg))
    except Exception as exc:  # noqa: BLE001
        print("  读腿部状态失败：{}".format(exc))


def describe_pose(nav, goal_pose):
    """打印"现在离桌子多远"，推车的人好判断摆没摆到位。"""
    try:
        pose = nav.pose()
    except Exception as exc:  # noqa: BLE001
        print("  读位姿失败：{}".format(exc))
        return
    distance = math.hypot(goal_pose[0] - pose[0], goal_pose[1] - pose[1])
    heading = wrap_deg(quat_to_yaw_deg(goal_pose[3:7]) - quat_to_yaw_deg(pose[3:7]))
    print("     现在离桌子 {:.2f} m，朝向差 {:+.0f}°".format(distance, heading))


def wait_swap(tag, confirm=None):
    """送完奶酪已经自己开回桌前，请人换下一块（``demo_flow.wait_enter`` 的等价物）。

    ``demo_flow.py`` 在这一步是 ``wait_enter("桌前换好第三块，准备检查")``；
    语音模式下换成"问一句、听人答一句"，不给硬超时 —— 换奶酪要多久取决于人。
    """
    print("\n" + "-" * 64)
    print("{}完成，我已经回到桌前。换下一块，换好继续。".format(tag))
    title = "{}完成，我已经回到桌前了。换好下一块跟我说一声。".format(tag)
    while True:
        if confirm is None:
            answer = input("换好了按回车继续 > ")
            if answer.strip().lower() in {"q", "quit"}:
                raise DemoStopped("换奶酪")
            return
        if confirm(title) == "go":
            return
        raise DemoStopped("换奶酪")


def run(robot, arm, leg, nav, vision=None, debug=False, voice=None, router=None,
        count_mode=ROUTE_COUNT_MODE, loops=None, confirm=None, on_round=None,
        say_farewell=True, ask_after_last=True, route_after=True,
        home_at_end=True, closing_hook=None):
    """逐阶段执行。顺序 = 目标.md 原管线 + 两个查看，一轮一抓放。

    每面的结果由 ``vision.look`` 立刻播报（PASS/FAIL），这里只负责几次跑完之后
    的道别 —— 放在收身之后，机器人站稳不动，观众看到的是"表演结束 -> 道别"。

    ``router`` 非空时，道别之后数一下"几个面有霉"，交给终点状态机分流：
    0 保持不动 / 1 去终点 A / 2 去终点 B（见 ``mission/route.py``）。

    2026-11-08 语音化后新增的参数（都有默认值，老调用方式不受影响）：

    ``loops``
        跑几圈，默认 ``LOOP_COUNT``(3)。语音说"检测一下面前这块奶酪"时传 1 ——
        用户要的是**眼前这一块**的结论，不是三块。
    ``confirm``
        圈与圈之间"请人推车 + 确认"的注入点。``None`` 走键盘；语音模式传
        ``callable(title) -> "go"/"stop"``。
    ``on_round``
        ``callable(index, looks)``，每圈结束回调一次，``looks`` 是这一圈的查看结果。
        语音模式用它"每块测完立刻汇报"。
    ``ask_after_last``
        最后一圈做完还要不要停下来等"按回车收身"。语音单块流程传 False，
        测完直接收身汇报，不再让人多按一次。
    ``say_farewell`` / ``route_after``
        单块语音流程里道别和分流都由上层（``xingbao.py``）决定，这里可以关掉。
    ``home_at_end`` / ``closing_hook``
        收尾移动（原地转 180° + 开回巡检起点）怎么走。默认 ``home_at_end=True``
        由本函数在最后自己走；语音模式传 ``home_at_end=False`` 并给一个
        ``closing_hook(dict)``，把 ``{"origin","table_pose","carried"}`` 交出去 ——
        上层播完最终结论之后再调 ``closing_moves``，免得机器人先开走、
        结论半天没说出来。只对"搬过奶酪"的那趟生效，没搬的就是原地结束。
    """
    total = LOOP_COUNT if loops is None else max(1, int(loops))
    pause(debug, "高举双手，身体站直")
    switch_leg(robot)
    report_leg(leg)
    raise_and_stand(arm, leg)
    goal_pose = nav.load_goal(GOAL_NAME)      # 也用来算"离桌子多远"

    # 巡检起点（demo_flow.py 的 origin）。收尾要开回来，所以必须在车动之前记。
    # 读不到就记 None —— home_to_origin 会跳过归位，不会带着坏数据去导航。
    origin = None
    if HOME_ON_FINISH or RETURN_TO_TABLE:
        try:
            origin = nav.pose()
            print("  巡检起点：x={:.3f} y={:.3f}".format(origin[0], origin[1]))
        except Exception as exc:                          # noqa: BLE001
            print("  记巡检起点失败（收尾不归位）：{}".format(exc))

    carried_to_goal = False
    for index in range(total):
        tag = "第{}次".format(index + 1)
        carried = False          # 这一圈有没有抱着奶酪直接去终点
        print("\n===== {} =====".format(tag))
        look_start = len(getattr(vision, "results", []) or [])
        pause(debug, "{}，导航到 {}".format(tag, GOAL_NAME))
        nav.navigate_to_pose(goal_pose)
        pause(debug, "切回腿部升降")
        switch_leg(robot)
        level_leg(leg)          # 导航后俯仰可能带偏差，回正一次再动
        report_leg(leg)

        stage(debug, "gripper_open", "夹爪全开")
        open_both_grippers(robot)

        stage(debug, "arm_reset", "双臂回高举位")
        arm.reset_both_arm()

        stage(debug, "height_ready", "身高到 {:.2f} m".format(READY_HEIGHT_M))
        leg.control_height(READY_HEIGHT_M)

        # 抓取之前先看：奶酪在桌上、双臂高举，相机俯视平面 —— 曝光和角度最好
        stage(debug, "look_front", "查看正面（未翻腕，奶酪在桌上）")
        look(vision, "{} 正面（未翻腕）".format(tag), "loop{}_faceA".format(index + 1))

        stage(debug, "arm_ready", "双臂到 Ready")
        arm.set_both_arm_ready()

        stage(debug, "reach_forward", "双手向前伸出 {:.2f} m".format(FORWARD_X_M))
        arm.shift_forward(FORWARD_X_M)

        stage(debug, "grasp_close", "环抱，半间距 {:.2f} m".format(HUG_GAP_M))
        arm.set_half_gap(HUG_GAP_M)

        stage(debug, "lift", "抬起，身高 {:.2f} m".format(HEIGHT_HIGH_M))
        leg.control_height(HEIGHT_HIGH_M)

        stage(debug, "flip", "转腕到 {:.3f} rad".format(-END_JOINT_RAD))
        set_end_joint_stable(arm, -END_JOINT_RAD)
        print("  翻腕停稳 {:.1f} s".format(wait_still(arm)))

        # 就在抬起高度看：原管线 200~205 帧就是这个位姿，降高度会把平面压没
        stage(debug, "look_back", "查看反面（已翻腕，身高 {:.2f} m）".format(HEIGHT_HIGH_M))
        look(vision, "{} 反面（已翻腕）".format(tag), "loop{}_faceB".format(index + 1))

        # ---- 分流判定点（2026-11-08 用户要求）----
        # 两面都看完了，结论此刻就定了。以前这里是"无条件放回原处"，
        # 现在先问一句：要走分流吗？走的话**保持夹持**，抱着开到终点再松爪。
        is_last = index + 1 >= total
        decision = None
        if (CARRY_TO_GOAL and router is not None and route_after and is_last):
            looks_round = (list(getattr(vision, "results", []) or [])[look_start:]
                           if vision is not None else [])
            try:
                count_round = count_mold(looks_round, mode=count_mode)
                # 用 set_count 而不是 decide：状态机要留下"数了几个、判了什么"，
                # 上层 xingbao._route_text() 读的就是 router.st.decision，
                # 缺了它最终播报只能说"已经开到终点"而说不出是 A 还是 B。
                decision = router.set_count(count_round)
            except Exception as exc:                     # noqa: BLE001
                print("  分流判定异常（{}），照常放回原位".format(exc))
                decision = None
            if decision is not None and decision.action != "hold" and decision.ready:
                print("\n===== 抱着奶酪去 {}（{}）=====".format(
                    decision.goal_name, decision.reason))
                routed_state, ok = carry_to_goal(
                    robot, arm, leg, nav, router, decision, debug=debug)
                carried = True
                carried_to_goal = True
                self_report = "已送到 {}".format(decision.goal_name) if ok else \
                    "没送到，就地放下了"
                print("  [携带] {}（状态 {}）".format(self_report, routed_state))
            else:
                if decision is not None:
                    print("  [携带] 不带走：{}".format(decision.describe()))
                carried = False

        if not carried:
            release_here(arm, leg, debug)

        # 每圈结束回调：语音模式在这里立刻汇报"这块两面都有霉/一面有霉/合格"。
        # 放在"请人推车"之前 —— 先把结论说完，再请人动手。
        if on_round is not None:
            looks = (list(getattr(vision, "results", []) or [])[look_start:]
                     if vision is not None else [])
            try:
                on_round(index + 1, looks)
            except DemoStopped:
                raise
            except Exception as exc:  # noqa: BLE001
                print("  每圈回调异常（{}），继续".format(exc))

        # ---- 分类之后怎么动（2026-11-08 从 demo_flow.py 迁进来）----
        # demo_flow 的顺序是 go_side -> finish_piece -> back_to_table -> wait_enter
        # -> turn_half -> 回 origin。go_side / finish_piece 已经由上面的
        # carry_to_goal 覆盖（判定 0 个霉时它不动、1 个去 A、2 个去 B ——
        # "最终移动结果不一样"就是这个意思），这里接后面的回桌与换料。
        #
        # 注意：现在"抱走"只在最后一圈触发（carry_to_goal 的条件里有 is_last），
        # 所以下面这一支要等"每圈都搬"的模式才会真的走到；留着是因为它就是
        # demo_flow 的 back_to_table，将来放开 carrying 时不用再补一遍。
        if carried and not is_last:
            closing_moves(robot, arm, leg, nav, origin, goal_pose, carried,
                          debug=debug, another_piece=True)

        # 请人换下一块。三种情况要分开：
        #   · 搬走过且还有下一块 —— 车已经自己开回桌前了，请人换料；
        #   · 没搬过 —— 车本来就在桌前，照旧请人换料；
        #   · 搬走过且是最后一块 —— 人站在终点桌边，桌前没有要换的料，
        #     别问（问了也没人应，还会挡着后面的谢幕收尾）。
        if (not carried or not is_last) and not (is_last and not ask_after_last):
            wait_swap(tag, confirm=confirm)

    pause(debug, "切回腿部升降")
    switch_leg(robot)
    stage(debug, "height_end", "身高收到 {:.2f} m".format(HEIGHT_END_M))
    leg.control_height(HEIGHT_END_M)

    # 跑完：先道别，再等播报播完（不能带着没播完的队列去 shutdown）
    if say_farewell:
        farewell(voice)

    # 道别之后再分流：观众看到的是"表演结束 -> 道别 -> 车开走"
    #
    # 2026-11-08：新版已经"抱着奶酪直接送到终点"，分流在循环里就走了，
    # 这里不能再走一次 —— 否则空手又开一遍同一段路。
    if carried_to_goal:
        print("  [携带] 分流已在搬运时完成，跳过事后再跑一遍")
        state = router.st.state if router is not None else None
    elif router is not None and route_after:
        state = route_after_loops(vision, router, count_mode)
    else:
        state = None

    # ---- 收尾移动：转身谢幕 + 开回巡检起点（demo_flow.py 的收尾两段）----
    # 只在"这一趟真的把奶酪抱到终点"之后做。没搬过的车还在检测桌工作位上，
    # 转身再开走会把下一句话要用的工作位给丢了（0 个霉就是这种情况：
    # demo_flow 里 go_side 不动的分支，在这里表现为"原地结束、不谢幕"）。
    #
    # 语音模式要让结论先说完再动，所以把状态交给 ``closing_hook`` 由上层
    # 播完最终汇报后再调 ``closing_moves``；键盘演示 ``home_at_end`` 直接走。
    closing = {
        "origin": origin,
        "table_pose": goal_pose,
        "carried": carried_to_goal,
    }
    if closing_hook is not None:
        try:
            closing_hook(closing)
        except Exception as exc:                          # noqa: BLE001
            print("  收尾回调异常（{}），跳过收尾移动".format(exc))
    elif home_at_end:
        closing_moves(robot, arm, leg, nav, origin, goal_pose, carried_to_goal,
                      debug=debug, another_piece=False)
    return state


def route_after_loops(vision, router, count_mode=ROUTE_COUNT_MODE):
    """数个数 -> 交给终点状态机。任何异常都不影响"演示已经跑完"这个事实。"""
    try:
        count = count_from_inspector(vision, mode=count_mode)
    except Exception as exc:  # noqa: BLE001
        print("  统计个数失败（{}），不导航".format(exc))
        return None
    print("\n===== 终点分流 =====")
    print("  口径 {}，识别到 {} 个".format(count_mode, count))
    try:
        return router.on_count(count)
    except Exception as exc:  # noqa: BLE001
        print("  终点分流异常：{}".format(exc))
        return None


def farewell(voice):
    """三次跑完之后收尾：先"检查完成"，再道别。

    "检查完成"是流程文档第 12 步要的那句（demo_flow 用 TTS 现场合成）。
    两句都进同一条 FIFO 队列，只有最后一句 wait=True —— 等它播完再返回，
    否则后面的 request_shutdown 会把尾音截掉；入队顺序保证"检查完成"在前。
    """
    if voice is None:
        return
    try:
        voice.say(VOICE_CHECK_DONE)
        if voice.say(VOICE_GOODBYE, wait=True):
            print("  收尾播报完毕（检查完成 + 道别）")
    except Exception as exc:  # noqa: BLE001
        print("  道别播报异常：{}".format(exc))


def look(vision, tag, slug):
    """一次查看。视觉出问题也不能打断动作，异常就地吞掉。"""
    if vision is None:
        return
    try:
        vision.look(tag, slug)
    except Exception as exc:  # noqa: BLE001
        print("  视觉异常（{}）：{}".format(tag, exc))


def connect(tool, vision_on=USE_VISION, out_dir=None, volume=None,
            frames=None, backoff=None, max_lastmile=None, voice_on=True,
            route_on=False):
    from galbot_sdk.g1 import (
        GalbotMotion,
        GalbotNavigation,
        GalbotRobot,
        MotionStatus,
    )

    robot = GalbotRobot()
    inspector = None
    voice = None
    router = None
    try:
        # 相机不在默认传感器集合里，要显式打开，否则 get_rgb_data 返回空
        ok = False
        if vision_on:
            try:
                from vision import open_camera

                ok = robot.init(open_camera())
            except Exception as exc:  # noqa: BLE001
                print("  打开相机失败（{}），退回默认传感器集合".format(exc))
        if not ok:
            ok = robot.init()
        if not ok:
            raise RuntimeError("robot.init() 失败")
        time.sleep(2.0)
        motion = GalbotMotion()
        if not motion.init():
            raise RuntimeError("motion.init() 失败")
        if tool not in motion.get_supported_tool_list():
            raise RuntimeError("SDK 不支持工具 {!r}".format(tool))
        for chain in ("right_arm", "left_arm"):
            status = motion.attach_tool(chain=chain, tool=tool)
            if status != MotionStatus.SUCCESS:
                raise RuntimeError(
                    "attach_tool({}, {}) 返回 {}".format(chain, tool, status)
                )
        navigation = GalbotNavigation()
        if not navigation.init():
            raise RuntimeError("navigation.init() 失败")
        planner = SdkArmPlanner(robot, motion, settle_s=0.0)
        planner.check_tcp_frame()
        arm = RightArmController(planner)
        leg = LegController(
            robot,
            joint_speed_rad_s=0.25,
            height_duration_s=2.0,
            settle_s=0.1,
            verify=False,
        )
        if voice_on:
            from vision import config as vision_config

            voice = VoiceBox(
                robot,
                volume=vision_config.VOICE_VOLUME if volume is None else float(volume),
            )
            missing = [n for n, fn in VOICE_FILES
                       if not voice.load(n, _HERE / "voice" / fn)]
            if missing:
                print("  ⚠ 这些语音没加载成功：{}（对应的播报会静默跳过）".format(missing))
        nav_kwargs = {}
        if backoff is not None:
            nav_kwargs["backoff_m"] = float(backoff)
        if max_lastmile is not None:
            nav_kwargs["max_lastmile_m"] = float(max_lastmile)
        nav = ChassisNavigator(
            robot,
            navigation,
            resolve_goal_dir(GOAL_NAME),
            **nav_kwargs
        )
        if vision_on:
            from vision import CheeseInspector

            inspector = CheeseInspector(
                robot,
                voice=voice,
                out_dir=out_dir,
                frames=frames,
            )
        if route_on:
            # 终点坐标还没配时也要建出来 —— 它会在 decide() 里明确说"没配"并保持不动，
            # 这比"忘了传 router 所以什么也没发生"更容易排查。
            router = RouteMachine(nav, voice=voice)
            _report_route_targets(router)
        return robot, arm, leg, nav, inspector, voice, router
    except Exception:
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
        raise


def _report_route_targets(router):
    """启动时把"几个霉去哪个终点"和终点坐标都打出来。

    2026-11-08 的教训：分流逻辑写好了、终点也配了，但两个入口的 ``--route``
    默认是关的，于是现场看到的现象是"检测完就放回原处，分类根本没生效"。
    开关一开，这里就把实际会用的坐标亮出来 —— 一眼能看出走的是哪个点、
    还是压根没配。
    """
    print("  分流：0 个霉 留在原地 / 1 个 去 route_a / 2 个 去 route_b")
    print("        （抱着奶酪过去，到位后才松爪）")
    for spec in (getattr(router, "_goal_a", None), getattr(router, "_goal_b", None)):
        if not spec:
            continue
        print("        {:<8} {}".format(spec.get("name"), _fmt_goal(spec)))


def _fmt_goal(spec):
    from mission.route import resolve_goal

    pose = resolve_goal(spec)
    if pose is None:
        return "⚠ 坐标没配（补 nav_goals/{}.json 或 mission/route.py 的 GOAL_*）".format(
            spec.get("name"))
    return "x={:.3f} y={:.3f} yaw={:+.1f}°".format(
        pose[0], pose[1], quat_to_yaw_deg(pose[3:7]))


def preflight():
    """确认这支脚本能在本机跑：SDK 在不在、视觉依赖在不在。

    ``galbot_sdk`` 是装在机器人 Orin 上的 aarch64 原生库（``/data/galbot/lib``）,
    在 PC / WSL 上一定 import 不到 —— 那不是代码问题，是跑错了地方。
    """
    try:
        import galbot_sdk.g1  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return ("导入 galbot_sdk 失败：{}\n"
                "  这支脚本必须**在机器人本体（Orin）上**运行，不是在 PC / WSL：\n"
                "    ssh galbot@<ROBOT_IP>        # 密码 <ROBOT_PASSWORD>\n"
                "    cd /home/galbot/scripts\n"
                "    export PYTHONPATH=/data/galbot/lib:$PYTHONPATH\n"
                "    export LD_LIBRARY_PATH=/data/galbot/lib:$LD_LIBRARY_PATH\n"
                "    python3 main.py --confirm-motion --debug\n"
                "  两个 export 漏了也是这个报错。".format(exc))
    if USE_VISION:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return ("视觉依赖缺失：{}\n"
                    "  机器人上装：python3 -m pip install opencv-python numpy\n"
                    "  或者先用 --no-vision 只跑动作。".format(exc))
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-motion",
        action="store_true",
        help="确认急停已松开、WORKING_MODE、周围净空，且有人能拍急停",
    )
    parser.add_argument("--tool", default=TOOL_NAME)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="每步回车执行，q 停止",
    )
    parser.add_argument("--no-vision", action="store_true", help="不做霉菌识别，只跑动作")
    parser.add_argument("--no-voice", action="store_true",
                        help="不做语音播报（识别照常，只是不出声）")
    parser.add_argument("--vision-frames", type=int, default=None,
                        help="每次查看连拍帧数，默认取 vision/config.py")
    parser.add_argument("--volume", type=float, default=None,
                        help="播报音量 0-100，默认取 vision/config.py（50）")
    parser.add_argument("--runs-dir", default=None,
                        help="识别证据存档目录，默认 <仓库根>/runs")
    parser.add_argument("--backoff", type=float, default=None,
                        help="固定预停位后退距离（m）。默认自动扫描，规划器嫌近会自动再退")
    parser.add_argument("--max-lastmile", type=float, default=None,
                        help="直线补位允许的最大距离（m），默认 2.5")
    # 2026-11-08 用户要求：分流**默认就开**。以前是 --route 手动加，
    # 结果 main.py 跑完永远"放回原处"，看起来像分类功能没写进去。
    parser.add_argument("--no-route", dest="route", action="store_false",
                        help="关掉按霉数分流（默认是开的：0 不动 / 1 去 A / 2 去 B，"
                             "抱着奶酪到终点再松爪；坐标见 mission/route.py）")
    parser.add_argument("--count-mode", default=ROUTE_COUNT_MODE,
                        choices=("faces", "spots"),
                        help="个数口径：faces=有霉的面数（默认），spots=单次菌斑数最大值")
    args = parser.parse_args(argv)
    if not args.confirm_motion:
        print(
            "未加 --confirm-motion，没有连接机器人。\n"
            "  python3 demo_cheese.py --confirm-motion",
            file=sys.stderr,
        )
        return 2

    problem = preflight()
    if problem:
        print(problem, file=sys.stderr)
        return 3

    robot = None
    inspector = None
    voice = None
    try:
        robot, arm, leg, nav, inspector, voice, router = connect(
            args.tool,
            vision_on=not args.no_vision,
            out_dir=args.runs_dir,
            volume=args.volume,
            frames=args.vision_frames,
            backoff=args.backoff,
            max_lastmile=args.max_lastmile,
            voice_on=not args.no_voice,
            route_on=args.route,
        )
        state = run(robot, arm, leg, nav, vision=inspector, debug=args.debug,
                    voice=voice, router=router, count_mode=args.count_mode,
                    route_after=args.route)
        print("分流结果：{}".format(state or "未走到分流"))
    except DemoStopped as exc:
        print("已在“{}”之前停止，不自动回位。".format(exc))
        return 0
    except Exception as exc:
        print("演示失败：{}\n不自动回位。".format(exc), file=sys.stderr)
        return 1
    else:
        print("演示结束。")
        return 0
    finally:
        if inspector is not None:
            try:
                inspector.summary()
            except Exception as exc:  # noqa: BLE001
                print("汇总失败：{}".format(exc))
        if voice is not None:
            try:
                voice.close()          # 把队列播完再退出线程
            except Exception as exc:  # noqa: BLE001
                print("关闭语音失败：{}".format(exc))
        if robot is not None:
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
