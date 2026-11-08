#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""唤醒实测诊断：麦克风有没有数据、模型给多少分。**机器人不动。**

用法（机器人上）::

    python3 wake_probe.py 15              # 读 15 秒，kws 后端
    python3 wake_probe.py 60 kws nocue    # 不播放"开始收音"提示音

每 0.5 秒打一行：收到多少块、RMS、模型当前分、最高分。
结束时按 A/B/C/D 四种情况给结论。

**第 3 个参数 ``cue`` / ``nocue``（默认 cue）**：``robot.init()`` 到真正开始
收音要 60 s 上下，如果人在脚本刚启动时就喊，喊完麦克风还没开。所以正式收音
前先播三声 880 Hz 蜂鸣当发令声，响完再 drain 一次，让人知道"现在开始"。

它和 ``Assistant`` 走完全相同的路径（``mic.MicRecorder`` + ``kws`` / ``wake``），
所以这里的结论直接代表线上表现。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from voice import kws as kws_mod          # noqa: E402
from voice import mic as mic_mod          # noqa: E402
from voice import wake as wake_mod        # noqa: E402

CUE_PATH = HERE / "voice" / "cue_start.pcm"
CUE_VOLUME = 60.0                 # 提示音音量；系统音量若被调低就听不见了


def play_cue(robot, stream_id="wake_probe_cue"):
    """播三声蜂鸣当"开始收音"的发令声。失败不影响主流程。"""
    try:
        data = CUE_PATH.read_bytes()
    except OSError:
        print("  （提示音 {} 缺失，跳过）".format(CUE_PATH.name))
        return 0.0
    if not data or len(data) % 2:
        return 0.0
    duration = len(data) / float(mic_mod.BYTES_PER_SECOND)
    try:
        try:
            robot.set_volume(CUE_VOLUME)
            print("  音量 -> {:.0f}".format(CUE_VOLUME))
        except Exception:                          # noqa: BLE001
            pass
        offset = 0
        while offset < len(data):
            chunk = data[offset:offset + mic_mod.CHUNK_BYTES]
            offset += len(chunk)
            if not robot.write_audio_stream_output(chunk, stream_id):
                print("  （提示音写入失败，跳过）")
                return 0.0
            time.sleep(0.05)
        time.sleep(duration)                       # 等喇叭放完，别急着 drain
        try:
            robot.stop_audio_stream_output(stream_id)
        except Exception:                          # noqa: BLE001
            pass
    except Exception as exc:                       # noqa: BLE001
        print("  （提示音播放异常：{}）".format(exc))
        return 0.0
    print("  ♪ 提示音已响（3 声），现在开始收音")
    return duration


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    seconds = float(argv[0]) if argv else 15.0
    engine = argv[1] if len(argv) > 1 else "kws"
    cue = (argv[2].lower() != "nocue") if len(argv) > 2 else True

    from galbot_sdk.g1 import GalbotRobot

    print("=" * 66)
    print("唤醒实测 {} 秒 —— 现在请对着机器人喊「星宝星宝」".format(seconds))
    print("=" * 66)
    print("  后端 {}，阈值 {:.2f}".format(
        engine,
        kws_mod.DEFAULT_THRESHOLD if engine == "kws" else wake_mod.DEFAULT_THRESHOLD))

    detector, used = kws_mod.build_detector(engine=engine, verbose=True)
    if detector is None:
        print("✗ 唤醒模型加载失败（kws 与 model 都不可用）")
        print("  kws 后端需要：python3 -m pip install --user sherpa-onnx")
        return 2
    print("  实际使用后端：{}".format(used))
    # KWS 后端不会给逐帧分数，命中就是命中；这里记一个 0/1
    peak_score = 0.0

    robot = GalbotRobot()
    if not robot.init():
        print("✗ robot.init() 失败")
        return 2
    time.sleep(2.0)

    recorder = mic_mod.MicRecorder(robot, verbose=False)
    chunks = []
    levels = []
    peak_score = 0.0
    hits = 0
    interrupted = False
    try:
        recorder.start()
        recorder.drain()
        if cue:
            play_cue(robot)
            recorder.drain()                       # 把喇叭的回声/余响丢掉
        print("  开始读…（前 1 秒是底噪基线）")

        started = time.monotonic()
        last_report = 0.0
        base_ema = 0.0                             # 底噪慢速跟踪，用来标"有声音"
        while time.monotonic() - started < seconds:
            try:
                chunk = recorder.read(timeout=0.5)
            except KeyboardInterrupt:             # 被 timeout 掐了也要出结论
                interrupted = True
                break
            if chunk is None:
                continue
            chunks.append(chunk)
            rms = mic_mod.pcm_rms(chunk)
            levels.append(rms)
            loud_flag = (rms > max(400.0, base_ema * 1.6)) if base_ema else False
            base_ema = rms if base_ema <= 0 else base_ema * 0.97 + rms * 0.03
            # last_score 是"历史最高分"，不归零就看不出当前这块的分。
            # 归零后 feed 会把它设成这块的最高分 —— 命中判定用的是逐帧分，
            # 跟 last_score 无关，所以这么干不影响结论。
            detector.last_score = 0.0
            hit = detector.feed(chunk)
            peak_score = max(peak_score, detector.last_score)
            if hit:
                hits += 1
                print("  >>> 命中！{}（第 {:.1f} s） 分数 {:.3f}".format(
                    getattr(detector, "last_text", "") or "", 
                    time.monotonic() - started, detector.last_score))
                detector.reset()

            now = time.monotonic() - started
            if now - last_report >= 0.5:
                last_report = now
                print("   {:5.1f}s  块={:4d}  RMS={:6.0f}  当前分={:.3f}  "
                      "最高分={:.3f}{}".format(now, len(chunks), rms,
                                               detector.last_score, peak_score,
                                               "   << 有声音" if loud_flag else ""))
    finally:
        recorder.stop()
        for name in ("request_shutdown", "wait_for_shutdown", "destroy"):
            try:
                getattr(robot, name)()
            except Exception:                 # noqa: BLE001
                pass

    pcm = b"".join(chunks)
    out = HERE / "voice" / "cache" / "wake_probe.pcm"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(pcm)
        wav = out.with_suffix(".wav")
        mic_mod.pcm_to_wav(pcm, wav)
        print("\n  音频已存：{}（{:.1f} s）".format(wav, mic_mod.pcm_duration(pcm)))
    except Exception as exc:                  # noqa: BLE001
        print("\n  存音频失败：{}".format(exc))

    # ---------------- 结论 ----------------
    print("\n" + "=" * 66)
    print("结论")
    print("=" * 66)
    total = len(chunks)
    if interrupted:
        print("  （被中断，只拿到 {:.1f} s / {:.0f} s 的数据）".format(
            mic_mod.pcm_duration(pcm), seconds))
    ordered = sorted(levels) if levels else [0.0]
    base = ordered[len(ordered) // 2]                     # 本底中位数
    # 现场底噪就有 1300 上下（风扇+人群），固定 300 会把底噪当人声。
    loud_limit = max(400.0, base * 1.5)
    loud = [v for v in levels if v > loud_limit]
    if total == 0:
        print("A. 一块音频都没收到 —— 麦克风流没起来。")
        print("   先看 `arecord -l` 和 fuser /dev/snd/*（可能被别的进程独占）。")
        return 1
    print("  收到 {} 块（{:.1f} s），本底中位 {:.0f}，门限 {:.0f}，".format(
        total, mic_mod.pcm_duration(pcm), base, loud_limit))
    print("  RMS 最高 {:.0f}，超过门限的有 {} 块（= 有人说话的时长 {:.1f} s）".format(
        max(levels) if levels else 0, len(loud),
        len(loud) * mic_mod.CHUNK_BYTES / float(mic_mod.BYTES_PER_SECOND)))
    if not loud:
        print("B. 麦克风有数据、但录到的全是稳态底噪 —— **没人说话**。")
        print("   这不是麦克风坏了，是这段窗口里没有声音进来：录的时候要对着")
        print("   机器人喊，或者确认嘴离麦克风在 1 米以内。")
        return 1

    if used == "kws":
        # KWS 不给逐帧分数，只有"命中/未命中"，所以 C/D 两档合并判断。
        if hits == 0:
            print("C. 有人声（RMS {:.0f} / 本底 {:.0f}，信噪比 {:.1f}x）但没命中关键词。".format(
                max(levels), base, max(levels) / (base or 1)))
            print("   可能原因：① 喊的不是 keywords.txt 里的词（现在是「星宝星宝」/「星宝」）")
            print("            ② 离得太远或声音被 AEC 削掉")
            print("            ③ 阈值偏高，试 --wake-th 0.15")
            print("   把存的 wav 拉下来，用 `wake_selftest.py --wav` 离线复核能否命中。")
            return 1
        print("E. 唤醒链路正常：{} 次命中。".format(hits))
        print("   线上唤不醒就往上层找（mute 窗口、drain、单实例锁）。")
        return 0

    if peak_score < 0.02:
        print("C. 有声音（RMS {:.0f} / 本底 {:.0f}，信噪比 {:.1f}x）但模型分数只有 {:.3f}"
              " —— 模型不认这段音频。".format(max(levels), base, max(levels) / (base or 1),
                                              peak_score))
        print("   这种人声进得来、模型不给分的情况，通常是采样率/增益/重采样不对，")
        print("   或者前处理模型与 TFLite 运行时版本不兼容（2026-11-08 就是这个原因）。")
        print("   建议直接换后端：--wake-engine kws（sherpa-onnx，见 README）。")
        return 1
    if hits == 0:
        print("D. 模型分数 {:.3f} 已经明显非零，但没连续两帧过 {:.2f}。".format(
            peak_score, detector.threshold))
        print("   改进方案：回声/环境噪声让分数忽高忽低，用 --wake-th {:.2f} 再试，".format(
            max(0.01, round(peak_score * 0.7, 2))))
        print("   或者把 patience 降到 1（单帧即命中）。")
        return 1
    print("E. 唤醒链路完全正常：{} 次命中，最高分 {:.3f}。".format(hits, peak_score))
    print("   那么线上唤不醒就不是唤醒层的问题，往上层找（mute 窗口、drain）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
