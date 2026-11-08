#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""唤醒模型离线自测：拿现成 wav 看能不能命中，**不连机器人、不用真人喊**。

为什么需要它
------------
现场调唤醒最难的地方是"必须有人喊"。这个脚本把验证变成纯离线的：
拿之前录下来的 wav（或者随便一段含"星宝星宝"的音频），直接喂模型看命中。

用法::

    python3 wake_selftest.py                     # 扫 voice/cache 里最新几个 wav
    python3 wake_selftest.py a.wav b.wav         # 指定音频
    python3 wake_selftest.py --engine model      # 对比旧的 openWakeWord
    python3 wake_selftest.py --wav-dir /some/dir # 扫目录

判定：
* 含"星宝星宝"的音频 -> 应命中（返回 0）
* 不含的 -> 应不命中（也返回 0）
* 任意一条与预期不符 -> 返回 1，并列出是哪条

**它只做离线推理，不开麦克风、不动机器人。**
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from voice import kws as kws_mod              # noqa: E402


def load_detector(engine: str, threshold, verbose: bool):
    det, used = kws_mod.build_detector(engine=engine, threshold=threshold,
                                       verbose=verbose)
    if det is None:
        print("✗ 没有可用的唤醒模型。")
        print("  kws 后端需要：python3 -m pip install --user sherpa-onnx")
        print("  并确认 voice/model_kws/ 下有 3 个 onnx + tokens.txt + keywords.txt")
    return det, used


def detect(det, path: Path, used: str):
    """返回 (是否命中, 命中的词, 时长秒)。"""
    try:
        if used == "kws":
            score = det.detect_wav(path)
            return score > 0, getattr(det, "last_text", ""), None
        # 旧后端：读 16k pcm 再跑
        from voice import mic as mic_mod
        pcm = mic_mod.wav_to_pcm(str(path))
        peak = det.detect_pcm(pcm)
        return peak >= det.threshold, "", mic_mod.pcm_duration(pcm)
    except Exception as exc:                     # noqa: BLE001
        print("  {}  读取/推理失败：{}".format(path.name, exc))
        return False, "", None


def main(argv=None):
    parser = argparse.ArgumentParser(description="唤醒模型离线自测（不动机器人）")
    parser.add_argument("wavs", nargs="*", help="要测的 wav 路径")
    parser.add_argument("--engine", default="kws",
                        choices=("kws", "model", "auto"))
    parser.add_argument("--wav-dir", default="", help="扫这个目录里的 wav")
    parser.add_argument("--th", type=float, default=None, help="唤醒阈值")
    parser.add_argument("--expect", default="",
                        help="逗号分隔的期望：hit / miss。不给就只看结果不判定")
    parser.add_argument("-q", "--quiet", action="store_true", help="精简输出")
    args = parser.parse_args(argv)

    det, used = load_detector(args.engine, args.th, verbose=not args.quiet)
    if det is None:
        return 2
    print("  实际后端：{}".format(used))

    paths = [Path(p) for p in args.wavs]
    if not paths and args.wav_dir:
        paths = sorted(Path(args.wav_dir).glob("*.wav"))
    if not paths:
        cache = HERE / "voice" / "cache"
        if cache.is_dir():
            all_wavs = sorted(cache.glob("*.wav"), key=lambda p: p.stat().st_mtime,
                              reverse=True)
            paths = all_wavs[:8]
    if not paths:
        print("没有可测的 wav。先录一段：python3 wake_probe.py 20")
        return 2

    expects = [e.strip() for e in args.expect.split(",") if e.strip()]
    print()
    print("{:<28} {:>9} {:>8}  {}".format("文件", "时长s", "命中", "命中的词"))
    print("-" * 66)

    bad = 0
    for idx, p in enumerate(paths):
        if not p.is_file():
            print("{:<28} {}".format(p.name[:28], "文件不存在"))
            bad += 1
            continue
        hit, text, dur = detect(det, p, used)
        if dur is None:
            # 别用 getnframes()：DashScope 下载的 wav 头里那个字段常常是错的
            # （会显示 44739 秒）。按实际读到的字节数算。
            try:
                import wave
                with wave.open(str(p), "rb") as fh:
                    frames = fh.readframes(fh.getnframes())
                    rate = fh.getframerate() or 16000
                    channels = fh.getnchannels() or 1
                dur = len(frames) / float(2 * channels * rate)
            except Exception:                    # noqa: BLE001
                dur = 0.0
        mark = "命中" if hit else "-"
        note = ""
        if idx < len(expects):
            want_hit = expects[idx].lower() in ("hit", "1", "yes", "y")
            if want_hit != hit:
                note = "  <= 期望{}，实际不符".format("命中" if want_hit else "不命中")
                bad += 1
        print("{:<28} {:>9.2f} {:>8}  {}{}".format(
            p.name[:28], dur, mark, "「{}」".format(text) if text else "", note))

    print()
    if bad == 0:
        print("结论：{} 条全部符合预期。".format(len(paths)))
        return 0
    print("结论：{} 条不符合预期，往上翻看是哪条。".format(bad))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
