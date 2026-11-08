#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""唤醒闭环自检：让机器人**自己播**「星宝星宝」，同时用自己的麦克风录，喂 KWS。

为什么需要它
------------
真人实测最难的地方是"必须有人在对的时候喊"。这个脚本把它变成不需要人的：
喇叭播一句已知的唤醒词 -> 麦克风录回来 -> KWS 判定。

* **命中** = 「麦克风 -> KWS」整条链是通的，现场唤不醒就只可能是距离/音量/朝向。
* **未命中但麦克风录到了喇叭声** = 麦克风没问题，是 KWS 在这条链上失效。
* **未命中且麦克风几乎没信号** = 自我回声消除（AEC）把喇叭声抹掉了，
  这一项测不出来，但能确认麦克风增益/链路状态。

**不动机器人任何关节，只播一段音频 + 开麦克风。**

用法::

    python3 wake_loopback.py             # 播一次，录 6 秒
    python3 wake_loopback.py 3           # 播三次
    python3 wake_loopback.py 2 --volume 70
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from voice import kws as kws_mod          # noqa: E402
from voice import mic as mic_mod          # noqa: E402

SAMPLE = HERE / "voice" / "kws_sample.pcm"
STREAM_ID = "wake_loopback"


def play_pcm(robot, data: bytes, verbose: bool = True) -> None:
    """按 SDK 要求 2560 字节一块、间隔 50 ms 写音频流。"""
    offset = 0
    while offset < len(data):
        chunk = data[offset:offset + mic_mod.CHUNK_BYTES]
        offset += len(chunk)
        if not robot.write_audio_stream_output(chunk, STREAM_ID):
            raise RuntimeError("write_audio_stream_output 失败")
        time.sleep(0.05)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="唤醒闭环自检（不动机器人）")
    parser.add_argument("times", nargs="?", type=int, default=1, help="播几次，默认 1")
    parser.add_argument("--volume", type=float, default=55.0, help="喇叭音量，默认 55")
    parser.add_argument("--record", type=float, default=6.0, help="每次录几秒，默认 6")
    args = parser.parse_args(argv)

    try:
        data = SAMPLE.read_bytes()
    except OSError:
        print("✗ 找不到 {}。先在 PC 上跑 _tmp/make_loop_pcm.py 生成并部署。".format(SAMPLE))
        return 2
    if len(data) % 2:
        data = data[:-1]
    sample_dur = len(data) / float(mic_mod.BYTES_PER_SECOND)
    print("=" * 66)
    print("唤醒闭环自检 —— 机器人自播「星宝星宝」({:.2f} s) 并自己录回来".format(sample_dur))
    print("=" * 66)

    detector, used = kws_mod.build_detector(engine="kws", verbose=True)
    if detector is None:
        print("✗ 唤醒模型加载失败")
        return 2
    print("  后端 {}，阈值 {:.2f}".format(used, detector.threshold))

    from galbot_sdk.g1 import GalbotRobot

    robot = GalbotRobot()
    if not robot.init():
        print("✗ robot.init() 失败")
        return 2
    print("  SDK 已初始化")
    time.sleep(2.0)

    recorder = mic_mod.MicRecorder(robot, verbose=False)
    results = []
    try:
        try:
            robot.set_volume(float(args.volume))
            print("  音量 -> {:.0f}".format(args.volume))
        except Exception as exc:                  # noqa: BLE001
            print("  设音量失败（忽略）：{}".format(exc))

        recorder.start()
        recorder.drain()

        for index in range(max(1, args.times)):
            print("\n--- 第 {}/{} 次 ---".format(index + 1, args.times))
            recorder.drain()
            chunks = []
            hit_at = None
            stop = threading.Event()

            def feed_loop():
                nonlocal hit_at
                started = time.monotonic()
                while not stop.is_set() and time.monotonic() - started < args.record:
                    chunk = recorder.read(timeout=0.3)
                    if chunk is None:
                        continue
                    chunks.append(chunk)
                    detector.last_score = 0.0
                    if detector.feed(chunk) and hit_at is None:
                        hit_at = time.monotonic() - started

            worker = threading.Thread(target=feed_loop, daemon=True)
            worker.start()
            time.sleep(0.4)                       # 先录一段底噪
            try:
                play_pcm(robot, data)
            except Exception as exc:              # noqa: BLE001
                print("  ✗ 播放失败：{}".format(exc))
            finally:
                try:
                    robot.stop_audio_stream_output(STREAM_ID)
                except Exception:                 # noqa: BLE001
                    pass
            worker.join(args.record + 2.0)
            stop.set()

            pcm = b"".join(chunks)
            rms = mic_mod.pcm_rms(pcm) if pcm else 0.0
            peak = 0
            if pcm:
                import array
                samples = array.array("h")
                samples.frombytes(pcm[:len(pcm) - len(pcm) % 2])
                if samples:
                    peak = max(max(samples), -min(samples))
            print("  录到 {:.1f} s  平均 RMS {:.0f}  峰值 {}".format(
                mic_mod.pcm_duration(pcm), rms, peak))
            if hit_at is not None:
                print("  ✓ KWS 命中（播放后 {:.1f} s）—— 麦克风->KWS 链路正常".format(hit_at))
            else:
                print("  ✗ 未命中")
            results.append((hit_at is not None, rms, peak))

            # 存这次录音，方便拉回来复核
            out = HERE / "voice" / "cache" / "loopback_{}.pcm".format(index + 1)
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(pcm)
                mic_mod.pcm_to_wav(pcm, out.with_suffix(".wav"))
            except Exception as exc:              # noqa: BLE001
                print("  存音频失败：{}".format(exc))
    finally:
        recorder.stop()
        for name in ("request_shutdown", "wait_for_shutdown", "destroy"):
            try:
                getattr(robot, name)()
            except Exception:                     # noqa: BLE001
                pass

    # ---------------- 结论 ----------------
    print("\n" + "=" * 66)
    print("结论")
    print("=" * 66)
    hits = sum(1 for ok, _, _ in results if ok)
    best_peak = max((p for _, _, p in results), default=0)
    print("  {} 次里命中 {} 次，麦克风录到的最大峰值 {}/32767".format(
        len(results), hits, best_peak))
    if hits:
        print("  ✓ 麦克风 -> KWS 整条链是通的。现场唤不醒就查：")
        print("    ① 人离麦克风多远（1 m 内最稳）  ② 是否对着机器人正面说话")
        print("    ③ 现场底噪是否把 VAD/人声盖住（看 wake_probe 的「<< 有声音」）")
        return 0
    if best_peak > 3000:
        print("  ⚠ 麦克风**录到了**喇叭声（峰值够），但 KWS 不认 -> 这条链上模型失效。")
        print("    把 voice/cache/loopback_*.wav 拉下来，用 wake_selftest.py 离线复核。")
        return 1
    print("  ⚠ 麦克风几乎没录到自己的喇叭声（峰值 {}）—— 大概率是自我回声消除（AEC）".format(
        best_peak))
    print("    把喇叭声当噪声消掉了，所以这一项测不出链路。改用真人实测：")
    print("    python3 wake_probe.py 100 kws cue")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
