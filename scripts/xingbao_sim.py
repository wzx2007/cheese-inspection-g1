#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""星宝语音链路模拟：**机器人一动不动**，跑一轮完整检测。

为什么要有这个
--------------
现场每改一次语音就得叫人来喊"星宝星宝"，成本太高；而且喇叭有 AEC，
自己放出来的声音进不了麦克风，远程根本没法验。所以这里把链路拆成六段，
每段都用真东西，只有"动"这一层换成假硬件：

::

    [1] 连音频      真的 robot.init + VoiceBox，读回音量
    [2] 麦克风      真的开流、真的读 12 块、打印 RMS（证明麦克风活着）
    [3] 唤醒        真的 xingbao.tflite，喂 TTS 合成的"星宝星宝…"看分数
    [4] 识别        真的走 DashScope ASR，把唤醒词剥掉得到命令
    [5] 一轮流程    真的 demo_cheese.run(loops=1)，但 robot/arm/leg/nav 是假的
                    -> 会走完 正/反两面查看、pass/fail 播报、结论汇报、写记忆
    [6] 记忆 + 联网 问"刚才那块怎么样"、"今天天气怎么样"

跑法::

    export PYTHONPATH=/data/galbot/lib:$PYTHONPATH
    export LD_LIBRARY_PATH=/data/galbot/lib:$LD_LIBRARY_PATH

    python3 xingbao_sim.py             # 全套
    python3 xingbao_sim.py --fast      # 跳过 TTS/ASR 那两步（只验流程逻辑）
    python3 xingbao_sim.py --no-flow   # 只验语音启动，不跑流程

**它不会动**：没有 motion.init、没有 attach_tool、没有导航，手臂腿都是空对象。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback

from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import demo_cheese                                    # noqa: E402
import xingbao                                        # noqa: E402
from mission import report as report_mod              # noqa: E402
from voice import cloud, intent                       # noqa: E402
from voice import mic as mic_mod                      # noqa: E402
from voice.assistant import Assistant, match_wake     # noqa: E402
from voice.memory import MemoryStore                  # noqa: E402
from voice.play_clip import ensure_pcm, pcm_cache_path  # noqa: E402

FRAME_BYTES = 2560                     # 80 ms @ 16 k / 16 bit / 单声道
WARM_PHRASE = "星宝星宝，你去检测一下面前这块奶酪"
REPEAT_PHRASE = "再检测一次"
WEATHER = "今天成都天气怎么样"
TOTAL = 6


def step(index, title):
    print("\n[{}/{}] {}".format(index, TOTAL, title))


# --------------------------------------------------------------------------
# 假硬件：只实现 demo_cheese.run 用到的那些方法，统统不产生运动
# --------------------------------------------------------------------------
def _success():
    from galbot_sdk.g1 import ControlStatus

    return ControlStatus.SUCCESS


class _FakeState(object):
    def __init__(self):
        self.joints = [0.0] * 7
        self.left_joints = [0.0] * 7


class FakeArm(object):
    """记录都下了哪些指令，方便核对顺序；永远"已经停稳"。"""

    def __init__(self):
        self.calls = []

    def reset_both_arm(self):
        self.calls.append("reset_both_arm")

    def set_both_arm_ready(self):
        self.calls.append("set_both_arm_ready")

    def shift_forward(self, x):
        self.calls.append("shift_forward({:.2f})".format(x))

    def set_half_gap(self, gap):
        self.calls.append("set_half_gap({:.2f})".format(gap))

    def set_end_joint(self, rad):
        self.calls.append("set_end_joint({:+.3f})".format(rad))

    def state_snapshot(self):
        return _FakeState()


class _FakeLegState(object):
    def __init__(self, height=1.37, pitch=0.0):
        self.height_m = height
        self.pitch_deg = pitch


class FakeLeg(object):
    def __init__(self):
        self.height_m = 1.37
        self.calls = []

    def reset_pitch(self):
        self.calls.append("reset_pitch")

    def control_height(self, height):
        self.height_m = float(height)
        self.calls.append("control_height({:.2f})".format(height))

    def read_state(self):
        return _FakeLegState(self.height_m, 0.0)


class FakeNav(object):
    """假装导航"到了"，其实原地不动。"""

    def __init__(self, seconds=0.4):
        self.seconds = float(seconds)
        self.calls = []

    def load_goal(self, name):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    def navigate_to_pose(self, pose):
        self.calls.append("navigate_to_pose")
        time.sleep(self.seconds)

    def is_localized(self):
        return True

    def pose(self):
        return (0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 1.0)


class FakeRobot(object):
    """夹爪/控制器的返回值照抄真 SDK 的成功码，其余不实现。"""

    def set_gripper_command(self, *args, **kwargs):
        return _success()

    def switch_controller(self, *args, **kwargs):
        return _success()


class FakeVision(object):
    """按顺序给出"第一面有霉、第二面干净"，并照真检查器那样播 pass/fail。"""

    def __init__(self, voice=None, molds=(True, False)):
        self.results = []
        self._molds = list(molds)
        self._voice = voice
        self._index = 0

    def look(self, tag, slug=None):
        mold = bool(self._molds[self._index % len(self._molds)])
        self._index += 1
        time.sleep(0.2)
        res = dict(tag=tag, slug=slug or "sim{}".format(self._index),
                   mold=mold,
                   verdict="FAIL（有霉）" if mold else "PASS（没霉）",
                   reason="ok", hits=4 if mold else 0, n=6,
                   max_ratio=0.0125 if mold else 0.0,
                   frames=[], time=time.strftime("%H:%M:%S"))
        self.results.append(res)
        # 真检查器就是在这里播音频的，模拟也照做 —— 顺带验证喇叭通路。
        # 片名与 vision/inspector.py 的 VOICE_* 一致：按"面 × 结论"四选一
        if self._voice is not None:
            back = "反面" in str(tag)
            name = ("back_bad" if back else "front_bad") if mold else \
                   ("back_ok" if back else "front_ok")
            try:
                self._voice.say(name)
            except Exception as exc:            # noqa: BLE001
                print("  （播报异常：{}）".format(exc))
        return res

    def summary(self):
        return any(r.get("mold") for r in self.results), self.results


# --------------------------------------------------------------------------
def mic_levels(mic, rounds=12):
    """读若干块，返回 [(rms, bytes)]，证明麦克风真的有数据。"""
    out = []
    for _ in range(rounds):
        chunk = mic.read(timeout=1.0)
        if chunk is None:
            break
        out.append((mic_mod.pcm_rms(chunk), len(chunk)))
    return out


def slow_to_human(pcm, tempo=0.67, work_dir=None):
    """把 PCM 放慢到真人语速（用 ffmpeg 的 atempo，不改音高）。

    为什么需要这一步：2026-11-08 实测发现 cosyvoice 的默认语速（约 5.8 字/秒）
    比真人快得多，而 ``xingbao.tflite`` 是用真人录音训的 —— 同一句
    "星宝星宝，你去检测一下面前这块奶酪"，**原速只有 0.03，放慢后 0.94**。
    所以模拟里两个都测：原速看"会不会误唤"，放慢看"模型本身认不认"。
    真人是正常语速，不受这个影响（现场已验证）。
    """
    import subprocess

    if not shutil.which("ffmpeg"):
        return b""
    work_dir = Path(work_dir or (HERE / "voice" / "cache"))
    src = work_dir / "_sim_slow_in.pcm"
    dst = work_dir / "_sim_slow_out.pcm"
    src.write_bytes(pcm)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", str(src),
        "-filter:a", "atempo={:.2f}".format(float(tempo)),
        "-f", "s16le", "-acodec", "pcm_s16le", str(dst),
    ]
    try:
        subprocess.run(command, check=True)
        return dst.read_bytes()
    except Exception:                       # noqa: BLE001
        return b""
    finally:
        for path in (src, dst):
            try:
                path.unlink()
            except OSError:
                pass


def feed_wake(detector, pcm, lead_silence_s=0.4):
    """把 PCM 按 80 ms 喂给唤醒模型，返回 (是否命中, 最高分, 第几块命中)。"""
    silence = b"\x00\x00" * int(16000 * lead_silence_s)
    data = silence + pcm
    best = 0.0
    hit_at = None
    for offset in range(0, len(data) - FRAME_BYTES + 1, FRAME_BYTES):
        chunk = data[offset:offset + FRAME_BYTES]
        hit = detector.feed(chunk)
        best = max(best, detector.last_score)
        if hit:
            hit_at = offset // FRAME_BYTES
            break
    return hit_at is not None, best, hit_at


def run(args):
    print("=" * 68)
    print("星宝语音链路模拟 —— 机器人不会动（没有 motion / 没有导航）")
    print("=" * 68)

    problem = demo_cheese.preflight()
    if problem:
        print(problem, file=sys.stderr)
        return 3

    robot = box = assistant = None
    memory = MemoryStore(HERE / "voice" / "memory_sim.json", verbose=False)
    try:
        step(1, "连音频（真的喇叭 + 音量）")
        robot, box = xingbao._connect_audio(args.volume)
        try:
            volume = robot.get_volume()
            print("  音量读回：{}".format(volume))
        except Exception as exc:                # noqa: BLE001
            print("  读音量失败（{}），跳过".format(exc))

        cfg = cloud.CloudConfig.from_file()
        cv = cloud.CloudVoice(cfg)
        print("  云端：{}".format("就绪（chat={}）".format(cfg.chat_model)
                                 if cfg.ready() else "没有 key，问答会退化"))
        assistant = Assistant(robot, voice=box, cloud_voice=cv, memory=memory,
                              wake_engine="auto", verbose=args.verbose)

        step(2, "麦克风（真的开流、真的读数据）")
        assistant.open_mic()
        levels = mic_levels(assistant._mic)
        if not levels:
            print("  ✗ 一块音频都没读到：麦克风没起来（先看 arecord -l）")
        else:
            peak = max(r for r, _ in levels)
            print("  读到 {} 块（每块 {} 字节），RMS 峰值 {:.0f}".format(
                len(levels), levels[0][1], peak))
            print("  {}".format("✓ 麦克风有声音" if peak > 1
                                else "⚠ RMS 全 0 —— 麦克风静音/被占用"))
        assistant.quiet()

        wake_ok = None
        heard = ""
        kind = None
        loops = 1
        if not args.fast:
            step(3, "唤醒模型（喂 TTS 合成的「{}」）".format(WARM_PHRASE))
            wav = HERE / "voice" / "cache" / "sim_wake.wav"
            try:
                cv.tts(WARM_PHRASE, wav)
                pcm = ensure_pcm(wav, pcm_cache_path(wav)).read_bytes()
            except Exception as exc:            # noqa: BLE001
                print("  TTS 失败（{}），这一步跳过".format(exc))
                pcm = b""
            if pcm:
                detector = assistant._wake
                if detector is None:
                    print("  ⚠ 没有本地唤醒模型，走文本唤醒")
                    wake_ok = False
                else:
                    print("  送检音频 {:.2f} s（TTS 原速）".format(len(pcm) / 32000.0))
                    detector.reset()
                    _hit, raw_best, _ = feed_wake(detector, pcm)
                    print("    原速峰值 {:.3f}  {}".format(
                        raw_best, "命中" if raw_best > detector.threshold else "未命中"))

                    slow = slow_to_human(pcm)
                    if slow:
                        print("  放慢到真人语速（{:.2f} s）".format(len(slow) / 32000.0))
                        detector.reset()
                        _hit2, best, _at = feed_wake(detector, slow)
                        cross = detector.detect_pcm(slow)
                        peak = max(best, cross)
                        wake_ok = peak > detector.threshold
                        print("    真人语速峰值 {:.3f}（阈值 {:.2f}）{}".format(
                            peak, detector.threshold,
                            "过线" if wake_ok else "没过线"))
                    else:
                        wake_ok = raw_best > detector.threshold
                    if detector.error:
                        print("  ⚠ 推理报错：{}".format(detector.error))
                    print("  {}".format(
                        "✓ 唤醒链路可用：模型认得出真人语速的唤醒词" if wake_ok else
                        "⚠ 唤醒没验成：连放慢的音频都不过阈值，"
                        "要么模型文件坏了，要么阈值设太高"))
                    if raw_best < detector.threshold:
                        chars_per_s = len("星宝星宝你去检测一下面前这块奶酪") / (
                            len(pcm) / 32000.0)
                        print("    注：TTS 原速约 {:.1f} 字/秒，比真人快得多，"
                              "原速分数低属于正常；现场真人声不受影响".format(chars_per_s))
                detector.reset()

            step(4, "识别（同一段音频走 ASR，剥掉唤醒词）")
            try:
                text = cv.asr(wav)
            except Exception as exc:            # noqa: BLE001
                text = ""
                print("  ASR 失败：{}".format(exc))
            print("  ASR：{}".format(text or "（空）"))
            woke, rest = match_wake(text)
            heard = rest or (text if not woke else "")
            if woke:
                print("  剥掉唤醒词 -> {}".format(heard or "（空）"))
            kind, why = intent.classify(heard or WARM_PHRASE)
            loops = intent.wanted_loops(heard or WARM_PHRASE)
            print("  意图：{}（{}） / 块数 {}".format(kind, why, loops))
        else:
            kind, loops = intent.START_TASK, 1
            print("\n（--fast：跳过 TTS/ASR/唤醒，直接跑流程）")

        if args.no_flow:
            print("\n（--no-flow：到这里就停了）")
            return 0

        step(5, "一轮完整流程（假硬件，不动机器人）")
        arm, leg, nav = FakeArm(), FakeLeg(), FakeNav(args.nav_s)
        vision = FakeVision(voice=box)
        core = xingbao.VoiceCore(assistant, inspector=vision, loops=1,
                                 route_on=False, verbose=args.verbose)
        core.robot, core.arm, core.leg, core.nav = FakeRobot(), arm, leg, nav
        before = len(vision.results)
        # 关键：这里**不是**直接调 run_task，而是把识别出来的那句话交给
        # ``VoiceCore.handle()`` —— 走的就是现场"喊完话 -> 判意图 -> 干活"的同一条路。
        # 2026-11-08 的事故就是这条路断了：用户以为在跟星宝说话，其实在跟
        # ask.py（纯问答）或 main.py（键盘驱动）说，永远不会有动作。
        command = heard or WARM_PHRASE
        print("  语音命令进 handle：{}".format(command))
        core.handle(command)
        ran = bool(arm.calls) and bool(nav.calls)
        print("  {}".format("✓ 语音命令真的启动了检测流程" if ran
                            else "✗ 命令没有触发任何动作（意图判错了？）"))
        looks = vision.results[before:]
        print("\n  ── 动作顺序（假硬件收到的指令）──")
        for name in arm.calls:
            print("     arm.{}".format(name))
        for name in leg.calls:
            print("     leg.{}".format(name))
        print("     导航 {} 次".format(len(nav.calls)))
        print("\n  ── 这一轮的查看 ──")
        for res in looks:
            print("     {}  mold={} {}".format(res["tag"], res["mold"], res["verdict"]))
        item = report_mod.summarize(looks)
        print("\n  结论：{}".format(item["spoken"]))
        print("  结论是否完整：{}".format("是" if item["ok"] else "否（有面没看清）"))

        step(6, "记忆 + 二次命令 + 联网问答")
        print("  记忆里最近一条：{}".format(memory.describe_last() or "（空）"))
        again = intent.classify(REPEAT_PHRASE)
        print("  说「{}」-> 意图 {}（{}）".format(REPEAT_PHRASE, again[0], again[1]))
        print("  {}".format("✓ 会再跑一轮，不会拒绝" if again[0] == intent.START_TASK
                            else "✗ 仍会被当成闲聊"))
        if cv.ready():
            t0 = time.time()
            answer = assistant.reply(WEATHER)
            print("  问「{}」".format(WEATHER))
            print("  答：{}".format(answer))
            print("  （{:.1f}s，联网={}）".format(time.time() - t0,
                                              assistant.brain.last_search))
            print("  {}".format("✓ 联网搜索生效" if assistant.brain.last_search
                                else "⚠ 这次没走搜索"))
        print("\n" + "=" * 68)
        print("模拟结束。机器人全程未动 —— 真机上仍需 --confirm-motion + 有人守急停。")
        return 0
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as exc:                    # noqa: BLE001
        print("模拟失败：{}".format(exc), file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        try:
            (HERE / "voice" / "memory_sim.json").unlink()
        except Exception:                       # noqa: BLE001
            pass
        if assistant is not None:
            try:
                assistant.close_mic()
            except Exception:                   # noqa: BLE001
                pass
        if box is not None:
            try:
                box.close()
            except Exception:                   # noqa: BLE001
                pass
        if robot is not None:
            for name in ("request_shutdown", "wait_for_shutdown", "destroy"):
                try:
                    getattr(robot, name)()
                except Exception:               # noqa: BLE001
                    pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="星宝语音链路模拟（机器人不动）")
    parser.add_argument("--fast", action="store_true", help="跳过 TTS/ASR/唤醒三步")
    parser.add_argument("--no-flow", action="store_true", help="只验语音启动，不跑流程")
    parser.add_argument("--volume", type=float, default=None, help="喇叭音量，默认 50")
    parser.add_argument("--nav-s", type=float, default=0.4,
                        help="假装导航耗时（秒），默认 0.4")
    parser.add_argument("--quiet", action="store_true", help="少打印")
    args = parser.parse_args(argv)
    args.verbose = not args.quiet
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
