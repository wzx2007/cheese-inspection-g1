#!/usr/bin/env python3
"""全流程：导航到终点（两段式）→ 检查落点误差 → 串上抓取。

为什么要两段式
--------------
终点的净空只有 0.05 m —— 建图起点就在那儿，机器人自身的点被写进了地图，
而规划器的 `point_safe_margin` 是 0.45 m。A* 走到低净空区附近就不肯再往前，
实测停在离终点 1.28 m 的地方等超时。

所以：
  第一段  navigate_to_goal(预停位)   预停位 = 终点沿朝向往回退 --backoff 米，
                                     净空足够，走完整 A* + 实时避障
  第二段  move_straight_to(相对位姿)  最后这一截用 odom 直线补上，不做全局规划

`move_straight_to` 收的是**相对当前 base_link 的位姿**，所以第二段会按
"当前实际位姿 → 终点" 现算一次，连朝向一起修正，不是盲走一米五。

    python3 nav_then_grab.py --confirm-motion              # 全流程
    python3 nav_then_grab.py --confirm-motion --dry-run    # 只查可达，不动
    python3 nav_then_grab.py --confirm-motion --no-grab    # 只导航
    python3 nav_then_grab.py --confirm-motion --backoff 0  # 不用两段式

导航段动静大、耗时长，建议用 `run_remote.py bg` 放后台跑，看日志：
    python run_remote.py bg "nav_then_grab.py --confirm-motion" navgrab
    python run_remote.py log navgrab
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GOAL_DIR = os.path.join(HERE, "nav_goals")


def load_goal(name):
    path = os.path.join(GOAL_DIR, "%s.json" % name)
    if not os.path.isfile(path):
        raise SystemExit("找不到目标点 %s（%s）。先 nav_tool.py --record %s" % (name, path, name))
    with open(path) as f:
        data = json.load(f)
    pose = [float(v) for v in data["pose"]]
    print("  目标点 %s : x=%.4f y=%.4f yaw=%.2f deg  (%s)"
          % (name, pose[0], pose[1], data.get("yaw_deg", 0.0), data.get("recorded_at", "")))
    return pose


def quat_to_yaw_deg(q):
    x, y, z, w = q
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny, cosy))


def norm_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def make_goal(x, y, yaw_deg, z=0.0):
    half = math.radians(yaw_deg) * 0.5
    return [float(x), float(y), float(z), 0.0, 0.0, math.sin(half), math.cos(half)]


def relative_pose(p, goal):
    """把"从当前位姿 p 到 goal"写成相对 p 的位姿（odom 系，x 朝车前）。"""
    pyaw = math.radians(quat_to_yaw_deg(p[3:7]))
    dx = goal[0] - p[0]
    dy = goal[1] - p[1]
    c, s = math.cos(pyaw), math.sin(pyaw)
    dx_r = c * dx + s * dy
    dy_r = -s * dx + c * dy
    dyaw = math.radians(norm_deg(quat_to_yaw_deg(goal[3:7]) - quat_to_yaw_deg(p[3:7])))
    return [dx_r, dy_r, 0.0, 0.0, 0.0, math.sin(dyaw * 0.5), math.cos(dyaw * 0.5)]


def wait_arrival(s, timeout, label):
    """非阻塞轮询到位。任务状态一旦不是 RUNNING 就立刻返回，别干等超时。"""
    t0 = time.time()
    last = 0.0
    while time.time() - t0 < timeout:
        if s.nav.check_goal_arrival():
            return True, time.time() - t0
        now = time.time()
        if now - last > 2.0:
            last = now
            p = s.pose()
            try:
                st = s.nav.get_navigation_status()
                stname = getattr(st, "name", str(st))
            except Exception:  # noqa: BLE001
                stname = "?"
            print("    %5.1fs %-8s (x=%.3f y=%.3f yaw=%.1f)  status=%s"
                  % (now - t0, label, p[0], p[1], quat_to_yaw_deg(p[3:7]), stname))
            if stname not in ("RUNNING", "UNKNOWN", "?"):
                print("    ↑ 任务已结束（status=%s），不再空等" % stname)
                return False, now - t0
        time.sleep(0.2)
    return False, timeout


def yaw_of(pose):
    return quat_to_yaw_deg(pose[3:7])


def wait_pose_change(s, p0, timeout):
    """等位姿真的变了再算下一步。

    2026-11-07 现场踩到的坑：连续 9 步"原地转正"读回来的位姿一模一样
    （1.642 m / +95.19 度），第 10 步才突然变成 1.315 m / -68.29 度 ——
    那 9 条指令是攒着一起兑现的，累计超转 163 度。
    所以每步之后必须确认"位姿真的动了"，没动就别再往上堆指令。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = s.pose()
        moved = (math.hypot(p[0] - p0[0], p[1] - p0[1]) > 0.005
                 or abs(norm_deg(yaw_of(p) - yaw_of(p0))) > 0.5)
        if moved:
            return True, p
        time.sleep(0.25)
    return False, s.pose()


def clamp(v, lim):
    return max(-lim, min(lim, v))


def settle(s, timeout=12.0, quiet=1.2, eps=0.02):
    """等车真正停稳再开始量误差。

    2026-11-07 实测：第一段导航 FAILED 时车还在 ~0.5 m/s 上跑，`stop_navigation()`
    之后还会惯性滑一两米，导致补位第 1、2 步量到的"剩余距离"从 2.8 m 跳到 1.1 m
    又跳回 1.2 m。必须在静止状态下量，否则每一步都是拿错位姿在算。
    """
    t0 = time.time()
    anchor = s.pose()
    t_anchor = t0
    while True:
        time.sleep(0.4)
        now = time.time()
        p = s.pose()
        moved = (math.hypot(p[0] - anchor[0], p[1] - anchor[1]) > eps
                 or abs(norm_deg(yaw_of(p) - yaw_of(anchor))) > 1.0)
        if moved:
            anchor = p
            t_anchor = now
        elif now - t_anchor >= quiet:
            print("    车已停稳（%.1f s 内位移 < %.0f cm）" % (now - t_anchor, eps * 100))
            return p
        if now - t0 > timeout:
            print("    ⚠️  等停稳超时 %.0f s，按当前位姿继续" % timeout)
            return p


def last_mile(s, goal, args):
    """闭环补最后一段：小步走，每步回头量误差。

    三条现场教训（2026-11-07）：
      1. 一次算 1 米多的相对位姿盲走会冲出去 1.5 m（实测 1.57 m 误差），
         必须拆成小步、每步重新量。
      2. 第一段导航可能中途 FAILED 停在转向半路，残余朝向差能到 90 度以上。
         这时先走"小弧线"把朝向掰回来（位移只给 --rot-creep），不要边走 0.15 m
         边甩 40 度，否则轨迹被拉成螺旋。
      3. 位姿回读有滞后，指令会攒着兑现。每步之后必须等到位姿真的变化，
         并且核对实际转角有没有明显超过指令值（超转就立刻收手）。
    """
    gyaw = yaw_of(goal)
    stalls = 0
    bad_yaw = 0
    for i in range(args.lastmile_tries):
        p0 = s.pose()
        d = math.hypot(goal[0] - p0[0], goal[1] - p0[1])
        dyaw = norm_deg(gyaw - yaw_of(p0))
        if d <= args.pos_tol and abs(dyaw) <= args.yaw_tol:
            print("    第 %d 步：已在容差内（%.3f m / %.2f deg），收工"
                  % (i + 1, d, dyaw))
            return True

        rel = relative_pose(p0, goal)
        mag = math.hypot(rel[0], rel[1])
        need_rot = abs(dyaw) > args.rot_first_deg
        if need_rot:
            trans = args.rot_creep
            step_yaw = clamp(dyaw, args.max_rot_step)
        else:
            # 朝向已对齐时可以放开步子：之前"冲出 1.5 m"是平移和大转角同时给的锅，
            # 纯平移大步的闭环可控性要好得多。第一步用 --first-step 走掉大头，
            # 之后回到 --lastmile-step 小步收尾（避免最后一段冲过头）。
            step = args.lastmile_step
            if (i == 0 and args.first_step > step
                    and abs(dyaw) <= args.yaw_tol and mag > step):
                step = min(args.first_step, mag)
            trans = min(step, mag)
            step_yaw = 0.0 if abs(dyaw) <= args.yaw_tol else clamp(dyaw, args.max_rot_step)
        if mag > 1e-6:
            k = trans / mag
            rel[0] *= k
            rel[1] *= k
        else:
            rel[0], rel[1] = trans, 0.0
        half = math.radians(step_yaw) * 0.5
        rel[5], rel[6] = math.sin(half), math.cos(half)

        print("    第 %d 步：剩余 %.3f m / %+.2f deg%s -> move x=%+.3f y=%+.3f dyaw=%+.1f"
              % (i + 1, d, dyaw, "（先掰朝向）" if need_rot else "",
                 rel[0], rel[1], step_yaw))

        try:
            ok, msg = s.nav.move_straight_to(rel, is_blocking=True,
                                             timeout=args.speed_timeout)
        except Exception as exc:  # noqa: BLE001
            print("      ✗ move_straight_to 抛异常：%r" % (exc,))
            return False
        print("      -> %r %r" % (ok, msg))

        changed, p1 = wait_pose_change(s, p0, args.pose_wait)
        if not changed:
            stalls += 1
            print("      ! 位姿毫无变化（%.1f s），这条指令没被执行，不再往上堆"
                  "（停滞 %d）" % (args.pose_wait, stalls))
            if stalls >= 2:
                print("      连续无响应，停止补位")
                break
            time.sleep(0.3)
            continue

        # 超转保护：实际转过的角度不该明显超过指令值
        actual_yaw = norm_deg(yaw_of(p1) - yaw_of(p0))
        if abs(norm_deg(actual_yaw - step_yaw)) > args.yaw_overshoot_deg:
            bad_yaw += 1
            print("      !! 转角异常：指令 %+.1f 度，实际 %+.1f 度（超转保护 %d/2）"
                  % (step_yaw, actual_yaw, bad_yaw))
            if bad_yaw >= 2:
                print("      连续两次转角异常，立即停止补位")
                break
        else:
            bad_yaw = 0

        d1 = math.hypot(goal[0] - p1[0], goal[1] - p1[1])
        if need_rot:
            continue
        if d1 > d - 0.01:
            stalls += 1
            print("      ! 没有靠近（%.3f -> %.3f），停滞 %d" % (d, d1, stalls))
            if stalls >= 3:
                print("      连续停滞，停止补位")
                break
        else:
            stalls = 0

    p = s.pose()
    d = math.hypot(goal[0] - p[0], goal[1] - p[1])
    dyaw = norm_deg(gyaw - yaw_of(p))
    print("    补位结束，残余 %.3f m / %.2f deg" % (d, dyaw))
    return d <= args.pos_tol and abs(dyaw) <= args.yaw_tol


def make_stop_pose(goal, gyaw, along, lateral):
    """终点沿朝向往后 along 米、横向偏 lateral 米的位姿（朝向仍朝终点方向）。"""
    rad = math.radians(gyaw)
    x = goal[0] - along * math.cos(rad) - lateral * math.sin(rad)
    y = goal[1] - along * math.sin(rad) + lateral * math.cos(rad)
    return make_goal(x, y, gyaw)


def find_stop_point(s, goal, alongs, laterals, max_lastmile, start_pose=None):
    """自动找第一个"规划器认可达"的预停位。

    为什么必须自动找：抓取要求车贴着桌子停，终点净空通常只有 0.1~0.2 m，
    而 `navigate_to_goal` / `check_path_reachability` 要求目标点净空 >=
    `point_safe_margin`（0.45 m）。所以"直接导航到终点"永远判不可达。
    但往回退 0.8~2.2 m 的位置净空够、可达，到那儿再用 `move_straight_to`
    做 odom 直线补最后这一段。

    先沿终点朝向由近到远扫（第二段补位距离 = along，越短越好）；
    都不行再试横向偏移（桌子正前方被挡时，从侧前方绕进去）。

    ⚠️ start_pose 默认取机器人当前位姿。如果机器人正停在净空不足的终点上，
    规划器会认为"起点就在障碍里"，从而把所有候选都判不可达 —— 这时用
    `--start` 指定一个正常起点来离线验证（只影响检查，不动车）。

    返回 (pre_pose, along, lateral)；全不可达返回 (None, None, None)。
    """
    gyaw = quat_to_yaw_deg(goal[3:7])
    pose = list(start_pose) if start_pose is not None else s.pose()
    cands = [(a, 0.0) for a in alongs]
    for lat in laterals:
        for a in alongs:
            cands.append((a, lat))

    seen = set()
    for along, lateral in cands:
        if along > max_lastmile:
            continue
        pre = make_stop_pose(goal, gyaw, along, lateral)
        key = (round(pre[0], 3), round(pre[1], 3))
        if key in seen:
            continue
        seen.add(key)
        d_robot = math.hypot(pre[0] - pose[0], pre[1] - pose[1])
        try:
            ok = bool(s.nav.check_path_reachability(pre, pose))
        except Exception as exc:  # noqa: BLE001
            print("    后退 %.1f 横移 %+.1f -> (%.3f, %.3f)  检查异常 %r"
                  % (along, lateral, pre[0], pre[1], exc))
            continue
        print("    后退 %.1f 横移 %+.1f -> (%.3f, %.3f)  距机器人 %.2f m  %s"
              % (along, lateral, pre[0], pre[1], d_robot, "✅ 可达" if ok else "❌"))
        if ok:
            return pre, along, lateral
    return None, None, None


def stage1_navigate(s, target, args):
    """第一段 navigate_to_goal（含瞬时失败重试）。返回 (ok, arrived, took)。

    ok=False 表示调用本身异常 / 被中断，调用方应中止整条流程。
    """
    print()
    print(">>> 第一段 navigate_to_goal（enable_collision_check=True, timeout=%.0f s）"
          % args.timeout)
    if args.arrival_threshold:
        try:
            ok_th, msg_th = s.nav.set_navigation_arrival_threshold(
                list(args.arrival_threshold))
            print("  放宽到位判据 : %r %r" % (ok_th, msg_th))
        except Exception as exc:  # noqa: BLE001
            print("  ⚠️  设置到位判据失败（忽略）：%r" % (exc,))
    s.stop_nav_on_exit = True   # 从这里开始，退出时负责停车
    arrived = False
    took = 0.0
    for attempt in range(args.stage1_retries + 1):
        if attempt:
            print()
            print(">>> 第一段重试 %d/%d（上一次只跑了 %.1f s 就 FAILED，多半是瞬时的）"
                  % (attempt, args.stage1_retries, took))
        try:
            result = s.nav.navigate_to_goal(
                target, enable_collision_check=True, is_blocking=False,
                timeout=args.timeout, omni_plan=False)
            print("    ->", result)
        except Exception as exc:  # noqa: BLE001
            print("❌ navigate_to_goal 调用失败：%r" % (exc,))
            return False, False, 0.0

        try:
            arrived, took = wait_arrival(s, args.timeout, "第一段")
        except KeyboardInterrupt:
            print("\n!! 中断，停车")
            try:
                s.nav.stop_navigation()
            except Exception:  # noqa: BLE001
                pass
            return False, False, 0.0

        if arrived:
            break
        try:
            s.nav.stop_navigation()
        except Exception:  # noqa: BLE001
            pass
        # 只有在"很快就失败、车基本没动"的情况下才值得重试；
        # 已经跑了几十秒才失败的，重试大概率还是同一处障碍
        if took > 30.0:
            print("    第一段跑了 %.1f s 才失败，不再重试" % took)
            break
    return True, arrived, took


def navigate(goal, args):
    """两段式导航。返回 (ok, 位置误差, 朝向误差)。"""
    from nav_tool import Session, require_nav_service

    pid = require_nav_service()
    print("  service_navigation_plan pid = %d" % pid)

    gyaw = quat_to_yaw_deg(goal[3:7])

    with Session() as s:
        loc, pose = s.report()
        if not loc or not pose:
            print("❌ 未定位，拒绝导航。")
            return False, None, None

        # ---- 确定第一段目标：预停位 ----------------------------------
        pre = None
        test_start = make_goal(*args.start) if args.start else None
        if args.start:
            print("  ⚠️  用 --start 指定的起点做可达性检查（仅检查，不影响真实导航）")
        # 可达性检查统一用这个起点（--start 指定时用它，否则用真实位姿）
        check_start = test_start if test_start is not None else pose
        dist_goal_now = math.hypot(goal[0] - check_start[0], goal[1] - check_start[1])
        bypass_stage1 = (dist_goal_now <= args.max_lastmile) and not args.force_stage1
        if bypass_stage1:
            print("  起点距终点仅 %.2f m（<= --max-lastmile %.2f m），"
                  "第一段目标（终点后退）反而在身后 —— 跳过第一段，直接闭环补位"
                  % (dist_goal_now, args.max_lastmile))
        elif args.backoff > 0:
            pre = make_stop_pose(goal, gyaw, args.backoff, 0.0)
            print("  两段式（固定）：预停位 x=%.4f y=%.4f yaw=%.2f deg（后退 %.2f m）"
                  % (pre[0], pre[1], gyaw, args.backoff))
        else:
            print("  自动寻找可达预停位（终点净空通常 < 0.45 m，不可能直接导航到终点）")
            pre, along, lateral = find_stop_point(
                s, goal, args.backoff_scan, args.lateral_scan, args.max_lastmile,
                start_pose=test_start)
            if pre is None:
                print("❌ 沿终点朝向后退 %s m（含横移 %s）全部不可达。"
                      % (args.backoff_scan, args.lateral_scan))
                print("   如果机器人正停在终点上（净空不足），这是预期现象 —— "
                      "换个起点、或用 --start 指定起点再验。")
                print("   否则说明终点周围被障碍包住了：需要在建图平台上清掉终点附近"
                      "的噪点，或把桌子/奶酪挪开一点。")
                return False, None, None
            print("  ✅ 选定预停位：后退 %.1f m、横移 %+.1f m -> x=%.4f y=%.4f"
                  "（第二段要补 %.2f m）" % (along, lateral, pre[0], pre[1], along))

        target = pre if pre is not None else goal
        dist = math.hypot(target[0] - check_start[0], target[1] - check_start[1])
        print("  到第 %s 段目标直线距离 %.3f m" % ("一" if pre else "唯一", dist))
        if dist > args.max_distance:
            print("❌ 距离超过 --max-distance %.1f m，拒绝。" % args.max_distance)
            return False, None, None

        if bypass_stage1:
            # 起点已在终点附近，终点净空不足本来就不适用 point_safe_margin，不做路径校验
            print("  路径可达性   : 跳过（起点已在终点 %.2f m 内，直接补位）"
                  % args.max_lastmile)
        elif pre is not None:
            # 自动扫描时 find_stop_point 已验证过，这里只是复核；
            # 固定 backoff 时这里是首次验证。都用同一个起点，避免自相矛盾。
            try:
                ok = bool(s.nav.check_path_reachability(target, check_start))
            except Exception as exc:  # noqa: BLE001
                print("❌ check_path_reachability 调用失败：%r" % (exc,))
                return False, None, None
            print("  第一段路径可达 : %s" % ("✅ 是" if ok else "❌ 否"))
            if not ok:
                print("❌ 规划器认为预停位不可达，不下发运动。")
                return False, None, None
        else:
            try:
                ok = bool(s.nav.check_path_reachability(target, check_start))
            except Exception as exc:  # noqa: BLE001
                print("❌ check_path_reachability 调用失败：%r" % (exc,))
                return False, None, None
            print("  路径可达     : %s" % ("✅ 是" if ok else "❌ 否"))
            if not ok:
                print("❌ 规划器认为不可达，不下发运动。")
                return False, None, None

        # 第二段要从预停位补到终点的距离
        if pre is not None:
            seg2 = math.hypot(goal[0] - pre[0], goal[1] - pre[1])
            print("  第二段 odom 补位距离 %.2f m（上限 --max-lastmile %.2f m）"
                  % (seg2, args.max_lastmile))

        if args.dry_run:
            print("  --dry-run：到此为止，没有下发任何运动。")
            return False, None, None

        from galbot_sdk.g1 import ControlStatus, G1ControllerName

        status = s.robot.switch_controller(G1ControllerName.CHASSIS_POSE_CTRL)
        if status != ControlStatus.SUCCESS:
            print("❌ switch_controller(CHASSIS_POSE_CTRL) 返回 %r" % (status,))
            return False, None, None
        print("  底盘位姿控制器 : 已接管")

        if args.lastmile_only:
            print()
            print(">>> --lastmile-only：跳过第一段，只做闭环补位")
            settle(s)
            p = s.pose()
            d_left = math.hypot(goal[0] - p[0], goal[1] - p[1])
            yaw_left = norm_deg(gyaw - quat_to_yaw_deg(p[3:7]))
            print("    剩余 %.3f m / %.2f deg" % (d_left, yaw_left))
            if d_left > args.max_lastmile:
                print("❌ 剩余 %.2f m 超过 --max-lastmile，不能盲走" % d_left)
                return False, d_left, yaw_left
            s.stop_nav_on_exit = True
            last_mile(s, goal, args)
            time.sleep(0.5)
            p = s.pose()
            pos_err = math.hypot(goal[0] - p[0], goal[1] - p[1])
            yaw_err = norm_deg(gyaw - quat_to_yaw_deg(p[3:7]))
            print("  终点位姿     : x=%.4f y=%.4f yaw=%.2f deg"
                  % (p[0], p[1], quat_to_yaw_deg(p[3:7])))
            print("  位置误差     : %.4f m" % pos_err)
            print("  朝向误差     : %.2f deg" % yaw_err)
            return (pos_err <= max(args.pos_tol, 0.03)), pos_err, yaw_err

        # ---- 第一段 ---------------------------------------------------
        arrived = False
        took = 0.0
        if bypass_stage1:
            print()
            print(">>> 第一段：跳过（起点已在终点 %.2f m 内，直接闭环补位）"
                  % args.max_lastmile)
            settle(s, timeout=4.0)
        else:
            ok1, arrived, took = stage1_navigate(s, target, args)
            if not ok1:
                return False, None, None

        if not arrived:
            settle(s)                      # 必须等它滑停，否则误差全是错的
            p = s.pose()
            d = math.hypot(target[0] - p[0], target[1] - p[1])
            if bypass_stage1:
                print("   已跳过第一段，直接进入闭环补位")
            else:
                print("⚠️  第一段没到位，停在离预停位 %.3f m 处" % d)
            if pre is None:
                p = s.pose()
                d_goal = math.hypot(goal[0] - p[0], goal[1] - p[1])
                if d_goal > args.max_lastmile:
                    print("   离终点还有 %.2f m，超过 --max-lastmile，不盲走" % d_goal)
                    return False, d_goal, norm_deg(gyaw - yaw_of(p))
                print("   剩下的 %.2f m 交给闭环补位" % d_goal)
            else:
                print("   两段式的意义就在这儿：剩下的用闭环补位，继续往下走")
        else:
            print("✅ 第一段到位（%.1f s）" % took)

        # ---- 第二段：最后这一截用 odom 直线闭环补 ---------------------
        # 到位（arrived）时车已停稳；FAILED 时上面 settle 过了。
        # 这里只补一个短确认，不重复 settle（settle 最少也要 1.2 s 静止才返回）。
        time.sleep(0.4)
        p = s.pose()
        d_left = math.hypot(goal[0] - p[0], goal[1] - p[1])
        yaw_left = norm_deg(gyaw - quat_to_yaw_deg(p[3:7]))
        print()
        print(">>> 第二段 闭环补位：剩余 %.3f m / %.2f deg，每步 %.2f m"
              % (d_left, yaw_left, args.lastmile_step))
        if d_left > args.max_lastmile:
            print("❌ 剩余 %.2f m 超过 --max-lastmile %.2f m（太远了不该盲走）"
                  % (d_left, args.max_lastmile))
            return False, d_left, yaw_left
        last_mile(s, goal, args)

        time.sleep(0.5)
        p = s.pose()
        pos_err = math.hypot(goal[0] - p[0], goal[1] - p[1])
        yaw_err = norm_deg(gyaw - quat_to_yaw_deg(p[3:7]))
        print()
        print("  终点位姿     : x=%.4f y=%.4f yaw=%.2f deg"
              % (p[0], p[1], quat_to_yaw_deg(p[3:7])))
        print("  位置误差     : %.4f m" % pos_err)
        print("  朝向误差     : %.2f deg" % yaw_err)
        return (pos_err <= max(args.pos_tol, 0.03)), pos_err, yaw_err


def run_arm_script(name, extra, cheesebot):
    cmd = [sys.executable, os.path.join(HERE, name),
           "--cheesebot", cheesebot, "--confirm-motion"] + extra
    print()
    print(">>> %s %s" % (name, " ".join(extra)))
    proc = subprocess.Popen(cmd, cwd=HERE)
    return proc.wait()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm-motion", action="store_true")
    ap.add_argument("--goal", default="table", help="nav_goals 里的目标点名字")
    ap.add_argument("--timeout", type=float, default=150.0, help="第一段导航超时秒数")
    ap.add_argument("--stage1-retries", type=int, default=1,
                    help="第一段很快 FAILED 时重试几次（默认 1，跑了 30 s 以上才失败就不重试）")
    ap.add_argument("--arrival-threshold", nargs=3, type=float,
                    metavar=("X", "Y", "YAW"),
                    default=[0.2, 0.2, 0.2],
                    help="导航到位判据 [x, y, yaw]，yaw 单位为弧度；传 0 0 0 表示不改（默认 0.2 0.2 0.2，放宽让第一段早点判到位）")
    ap.add_argument("--backoff", type=float, default=0.0,
                    help="固定预停位后退距离；0（默认）表示自动扫描可达预停位")
    ap.add_argument("--backoff-scan", default="0.8,1.0,1.2,1.5,1.8,2.2",
                    help="自动扫描的后退距离序列（米，逗号分隔，由近到远）")
    ap.add_argument("--lateral-scan", default="0.4,-0.4",
                    help="正前方都不可达时再试的横向偏移（米，逗号分隔）")
    ap.add_argument("--max-lastmile", type=float, default=2.5,
                    help="第二段 odom 补位允许的最大距离")
    ap.add_argument("--lastmile-step", type=float, default=0.2,
                    help="闭环补位每步最多走多少米（默认 0.2）")
    ap.add_argument("--first-step", type=float, default=0.6,
                    help="补位第一步允许的大步上限（朝向已对齐时用，默认 0.6 m；0 表示禁用）")
    ap.add_argument("--lastmile-tries", type=int, default=25,
                    help="闭环补位最多走几步（默认 25，一步 0.2 m 覆盖 5 m）")
    ap.add_argument("--rot-first-deg", type=float, default=10.0,
                    help="朝向差超过这个角度就先走小弧线掰朝向（默认 10 度）")
    ap.add_argument("--max-rot-step", type=float, default=20.0,
                    help="单步最多转多少度（默认 20 度）")
    ap.add_argument("--rot-creep", type=float, default=0.05,
                    help="掰朝向那一步的位移上限，避免纯旋转被控制器拒绝（默认 0.05 m）")
    ap.add_argument("--pose-wait", type=float, default=3.0,
                    help="每步之后等位姿变化的最长秒数（默认 3 s）")
    ap.add_argument("--yaw-overshoot-deg", type=float, default=25.0,
                    help="实际转角比指令值多出这么多度就判定超转（默认 25 度）")
    ap.add_argument("--speed-timeout", type=float, default=25.0,
                    help="move_straight_to 的单步超时秒数")
    ap.add_argument("--max-distance", type=float, default=15.0)
    ap.add_argument("--pos-tol", type=float, default=0.06,
                    help="到位后允许的位置误差，超过就不抓")
    ap.add_argument("--yaw-tol", type=float, default=4.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lastmile-only", action="store_true",
                    help="跳过第一段导航，只从当前位置闭环补到最后几厘米")
    ap.add_argument("--no-grab", action="store_true")
    ap.add_argument("--cheesebot", default="/home/galbot/scripts_test")
    ap.add_argument("--grasp-flip", action="store_true", default=True,
                    help="抓取时走一遍转腕再转回（默认开启；用 --no-grasp-flip 关掉）")
    ap.add_argument("--no-grasp-flip", dest="grasp_flip", action="store_false")
    ap.add_argument("--video-out", default="",
                    help="抓取视频输出路径（默认让 grab_video.py 用它自己的默认值）")
    ap.add_argument("--keep-height", action="store_true",
                    help="抓取结束后把身高降回 1.32（多花约 4.5 s，默认不降）")
    ap.add_argument("--start", nargs=3, metavar=("X", "Y", "YAW"), type=float,
                    help="离线验证用的起点（只影响可达性检查，不动机器人）")
    ap.add_argument("--force-stage1", action="store_true",
                    help="即使起点已在终点附近也强制执行第一段导航（默认会自动跳过）")
    args = ap.parse_args(argv)

    # "0.8,1.0,1.2" -> [0.8, 1.0, 1.2]
    args.backoff_scan = [float(v) for v in str(args.backoff_scan).split(",") if v.strip()]
    args.lateral_scan = [float(v) for v in str(args.lateral_scan).split(",") if v.strip()]
    if not args.backoff_scan:
        args.backoff_scan = [1.2]
    if args.lateral_scan is None:
        args.lateral_scan = []

    if not args.confirm_motion:
        print("未加 --confirm-motion，没有连接机器人。", file=sys.stderr)
        return 2

    def _on_signal(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except Exception:
            pass

    print("=== 1/2 导航 ===")
    goal = load_goal(args.goal)
    ok, pos_err, yaw_err = navigate(goal, args)

    if args.dry_run:
        return 0
    if not ok:
        print("\n❌ 没到位，不进入抓取。")
        return 1
    if pos_err > args.pos_tol or abs(yaw_err) > args.yaw_tol:
        print("\n❌ 到位误差超限（位置 %.3f m / 朝向 %.2f deg，容差 %.2f m / %.0f deg），"
              "**不进入抓取** —— 夹爪压入量只有 3 cm。"
              % (pos_err, yaw_err, args.pos_tol, args.yaw_tol))
        return 1
    if args.no_grab:
        print("\n--no-grab：导航完成，跳过抓取。")
        return 0

    print()
    print("=== 2/2 抓取（单进程 + 内置视频） ===")
    # 旧流程是 arm_pregrasp.py → cheese_place_aligned.py 两个进程，
    # 中间要重新 init 一次（约 8 s，手悬空不动）——那正是"合住前的长停顿"。
    # grab_video.py 一个进程跑完全部动作，并自动录视频 + 出帧-状态映射表。
    extra = []
    if not args.grasp_flip:
        extra.append("--no-flip")
    if args.video_out:
        extra += ["--out", args.video_out]
    if args.keep_height:
        extra.append("--final-height")
    rc = run_arm_script("grab_video.py", extra, args.cheesebot)
    if rc != 0:
        print("❌ 抓取失败（rc=%d）。手臂可能停在半途，用 arm_home.py 收。" % rc)
        return 1

    print()
    print("✅ 全流程完成：导航 → 夹取 → 抬起 → 放回（视频与映射表已生成）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
