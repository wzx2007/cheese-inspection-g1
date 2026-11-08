# -*- coding: utf-8 -*-
"""离线对照：机器人上已有的 wav 喂给唤醒模型能打多少分？

完全不用人喊、不用麦克风。目的只有一个：
**分辨"模型/依赖坏了" 还是 "麦克风音频不对劲"。**
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
    if sw != 2:
        return ch, sw, sr, np.zeros(0)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if ch > 1:
        x = x[::ch]
    return ch, sw, sr, x


def resample(x, src, dst=16000):
    if src == dst or not len(x):
        return x
    n_out = max(2, int(round(len(x) * dst / float(src))))
    return np.interp(np.linspace(0, len(x) - 1, n_out),
                     np.arange(len(x), dtype=np.float64), x)


def score(pcm):
    det = wake_mod.LocalWakeDetector(verbose=False)
    if not det.prime():
        return -1.0, det.error
    return det.detect_pcm(pcm), ""


def to_pcm(x):
    return np.clip(x, -32768, 32767).astype("<i2").tobytes()


def main():
    cache = HERE / "voice" / "cache"
    print("扫描", cache)
    files = sorted(cache.glob("*.wav"))
    print("找到 {} 个 wav\n".format(len(files)))

    rows = []
    for p in files:
        try:
            ch, sw, sr, x = read_wav(p)
            if not len(x):
                continue
            y16 = resample(x, sr, 16000)
            s16, _ = score(to_pcm(y16))
            # 放慢到 0.67 倍（真人语速），看是否显著提升
            slow = resample(y16, int(16000 / 0.67), 16000)
            sslow, _ = score(to_pcm(slow))
            rows.append((max(s16, sslow), p.name, ch, sr, len(x) / float(sr), s16, sslow))
        except Exception as exc:                     # noqa: BLE001
            print("  {} 读取失败：{}".format(p.name, exc))

    rows.sort(reverse=True)
    print("{:<26} {:>3} {:>6} {:>7} {:>9} {:>9}".format(
        "文件", "声道", "采样率", "时长s", "原速分", "放慢0.67"))
    print("-" * 68)
    for best, name, ch, sr, dur, s16, sslow in rows[:25]:
        print("{:<26} {:>3} {:>6} {:>7.2f} {:>9.3f} {:>9.3f}".format(
            name[:26], ch, sr, dur, s16, sslow))

    if rows:
        best = rows[0]
        print("\n最好的一份：{}  放慢后 {:.3f}".format(best[1], best[6]))
        if best[0] >= 0.15:
            print("=> pipeline 正常，模型对**已知好音频**能过阈值。")
            print("   那么真人喊不给分 = 麦克风送进来的音频本身不对。")
        else:
            print("=> 连已知音频都拿不到分，**模型/pipeline 有问题**，")
            print("   先去查 xingbao.tflite 与前处理模型是否匹配。")
    else:
        print("\n没有可用 wav —— 这个对照做不了。")


if __name__ == "__main__":
    raise SystemExit(main())
