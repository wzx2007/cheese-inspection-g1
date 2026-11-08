# -*- coding: utf-8 -*-
"""A/B 对照：TTS 合成音 vs 真人录音，喂同一个唤醒模型看分数。

复用**同一个** detector（模型只加载一次），否则 400 多个文件会跑到天荒地老。
机器人不动，纯离线。
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np                                    # noqa: E402

from voice import wake as wake_mod                    # noqa: E402


def read_wav(path):
    with wave.open(str(path), "rb") as f:
        ch, sw, sr, n = f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()
        raw = f.readframes(n)
    if sw != 2 or not raw:
        return ch, sr, np.zeros(0)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if ch > 1:
        x = x[::ch]
    return ch, sr, x


def resample(x, src, dst=16000):
    if src == dst or not len(x):
        return x
    n_out = max(2, int(round(len(x) * dst / float(src))))
    return np.interp(np.linspace(0, len(x) - 1, n_out),
                     np.arange(len(x), dtype=np.float64), x)


def to_pcm(x):
    return np.clip(np.round(x), -32768, 32767).astype("<i2").tobytes()


def main():
    cache = HERE / "voice" / "cache"
    det = wake_mod.LocalWakeDetector(verbose=False)
    if not det.prime():
        print("✗ 模型加载失败：", det.error)
        return 2

    def score(path, speed=1.0):
        ch, sr, x = read_wav(path)
        if not len(x):
            return None
        y = resample(x, sr, 16000)
        if speed != 1.0:
            y = resample(y, int(16000 * speed), 16000)   # speed<1 = 放慢
        return det.detect_pcm(to_pcm(y))

    def rms_of(path):
        ch, sr, x = read_wav(path)
        return float(np.sqrt((x ** 2).mean())) if len(x) else 0.0

    replies = sorted(cache.glob("reply_*.wav"))
    heards = sorted(cache.glob("heard_*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]
    big = sorted(cache.glob("heard_*.wav"), key=lambda p: p.stat().st_size, reverse=True)[:15]
    seen, heards = set(), []
    for p in heards + big:
        if p.name not in seen:
            seen.add(p.name)
            heards.append(p)

    print("=" * 74)
    print("[A] TTS 合成音 reply_*.wav（共 {} 个，测最新的 20 个）".format(len(replies)))
    print("=" * 74)
    print("{:<18} {:>7} {:>7} {:>9} {:>9}".format("文件", "时长s", "RMS", "原速分", "放慢0.67"))
    a_best = 0.0
    a_best_name = ""
    for p in replies[-20:]:
        ch, sr, x = read_wav(p)
        if not len(x):
            continue
        dur = len(x) / float(sr)
        s1 = score(p, 1.0) or 0.0
        s2 = score(p, 0.67) or 0.0
        if max(s1, s2) > a_best:
            a_best, a_best_name = max(s1, s2), p.name
        print("{:<18} {:>7.2f} {:>7.0f} {:>9.3f} {:>9.3f}".format(
            p.name, dur, rms_of(p), s1, s2))

    print()
    print("=" * 74)
    print("[B] 真人录音 heard_*.wav（测 {} 个：最新 + 最长）".format(len(heards)))
    print("=" * 74)
    print("{:<18} {:>7} {:>7} {:>9} {:>9}".format("文件", "时长s", "RMS", "原速分", "放慢0.67"))
    b_best = 0.0
    b_best_name = ""
    for p in heards:
        ch, sr, x = read_wav(p)
        if not len(x):
            continue
        dur = len(x) / float(sr)
        s1 = score(p, 1.0) or 0.0
        s2 = score(p, 0.67) or 0.0
        if max(s1, s2) > b_best:
            b_best, b_best_name = max(s1, s2), p.name
        print("{:<18} {:>7.2f} {:>7.0f} {:>9.3f} {:>9.3f}".format(
            p.name, dur, rms_of(p), s1, s2))

    print()
    print("=" * 74)
    print("结论")
    print("=" * 74)
    print("  TTS 最高 {:.3f}（{}）".format(a_best, a_best_name or "-"))
    print("  真人最高 {:.3f}（{}）".format(b_best, b_best_name or "-"))
    if a_best >= 0.15 and b_best < 0.15:
        print("  => 模型正常，**问题在真人音频这条链**（麦克风/增益/降噪把人声搞坏了）。")
    elif a_best < 0.15 and b_best < 0.15:
        print("  => 连 TTS 都不给分：**模型或前处理模型有问题**，跟麦克风无关。")
    elif b_best >= 0.15:
        print("  => 真人录音能给分：唤醒模型没问题，")
        print("     问题在**线上循环怎么喂**（块切分 / mute 窗口 / drain / 阈值）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
