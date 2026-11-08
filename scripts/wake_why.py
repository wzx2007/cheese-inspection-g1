# -*- coding: utf-8 -*-
"""在机器人上跑：为什么唤醒模型对真人喊不给分？

一次查四件事：
1. SDK 回调的实际块大小分布（是不是恒定 2560 字节）
2. 录音设备/流的真实参数（arecord -l、/proc/asound）
3. 人声段按不同源采样率重采样后喂模型，看有没有哪个假设突然出高分
4. 人声段 vs 底噪段的频谱对比（判断是不是语音、有没有被压到低频）

**机器人不动**，只读麦克风 + 离线推理。
"""
from __future__ import annotations

import array
import sys
import time
import wave
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np                                    # noqa: E402

from voice import mic as mic_mod                      # noqa: E402
from voice import wake as wake_mod                    # noqa: E402

SR = 16000


def read_wav(path):
    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        raw = f.readframes(f.getnframes())
    return sr, np.frombuffer(raw, dtype="<i2").astype(np.float64)


def resample(x, src_rate, dst_rate=SR):
    if src_rate == dst_rate:
        return x
    n_out = int(round(len(x) * dst_rate / float(src_rate)))
    if n_out < 2:
        return np.zeros(2)
    return np.interp(np.linspace(0, len(x) - 1, n_out),
                     np.arange(len(x), dtype=np.float64), x)


def to_pcm(x):
    return np.clip(x, -32768, 32767).astype("<i2").tobytes()


def score(pcm):
    det = wake_mod.LocalWakeDetector(verbose=False)
    if not det.prime():
        return -1.0, det.error
    return det.detect_pcm(pcm), ""


def spectrum_profile(x, sr=SR, tag=""):
    xc = x - x.mean()
    spec = np.abs(np.fft.rfft(xc * np.hanning(len(xc))))
    freqs = np.fft.rfftfreq(len(xc), 1.0 / sr)
    total = spec.sum() or 1.0
    out = []
    for lo, hi in ((0, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 8000)):
        m = (freqs >= lo) & (freqs < hi)
        out.append("{}-{}Hz {:4.1f}%".format(lo, hi, 100.0 * spec[m].sum() / total))
    peak = freqs[int(np.argmax(spec))]
    print("  {} 峰值频点 {:.0f} Hz | {}".format(tag, peak, " | ".join(out)))


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0

    print("=" * 70)
    print("唤醒诊断（机器人不动）—— 请对着机器人喊「星宝星宝」")
    print("=" * 70)

    # ---------- 1. 实时录，同时统计块大小 ----------
    from galbot_sdk.g1 import GalbotRobot
    robot = GalbotRobot()
    if not robot.init():
        print("✗ robot.init() 失败")
        return 2
    time.sleep(2.0)

    rec = mic_mod.MicRecorder(robot, verbose=False)
    sizes = Counter()
    chunks = []
    rms_list = []
    try:
        rec.start()
        rec.drain()
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            c = rec.read(timeout=0.5)
            if c is None:
                continue
            sizes[len(c)] += 1
            chunks.append(c)
            rms_list.append(mic_mod.pcm_rms(c))
    finally:
        rec.stop()
        for name in ("request_shutdown", "wait_for_shutdown", "destroy"):
            try:
                getattr(robot, name)()
            except Exception:                          # noqa: BLE001
                pass

    pcm = b"".join(chunks)
    wall = time.monotonic() - t0
    print("\n[1] 实时采样")
    print("  墙钟 {:.1f} s，收到 {} 块，合计 {} 字节".format(wall, len(chunks), len(pcm)))
    print("  按 16k/16bit/mono 折算 = {:.2f} s".format(len(pcm) / float(SR * 2)))
    print("  实际块率 {:.2f} 块/s".format(len(chunks) / wall))
    print("  块大小分布（字节: 次数）: {}".format(
        ", ".join("{}:{}".format(k, v) for k, v in sizes.most_common(5))))
    if sizes:
        main_size = sizes.most_common(1)[0][0]
        print("  主块大小 {} 字节 = {} 采样 @16k = {:.1f} ms".format(
            main_size, main_size // 2, 1000.0 * main_size / (SR * 2)))
        implied = len(chunks) / wall * main_size / 2.0
        print("  ** 隐含采样率 = {:.0f} Hz **（若为 16000 则一致）".format(implied))

    x = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    out_dir = HERE / "voice" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "wake_why.wav"
    mic_mod.pcm_to_wav(pcm, wav_path)
    print("  已存 {}".format(wav_path))

    # ---------- 2. 设备信息 ----------
    print("\n[2] 音频设备")
    for cmd in ("arecord -l 2>&1 | head -20",
                "cat /proc/asound/card*/pcm*c/sub0/hw_params 2>/dev/null | head -20"):
        import subprocess
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=8)
            text = (r.stdout or r.stderr or "").strip()
            print("  $ {}\n{}".format(cmd, "\n".join("    " + l for l in text.splitlines()[:12])
                                      or "    （空）"))
        except Exception as exc:                       # noqa: BLE001
            print("  $ {} -> {}".format(cmd, exc))

    # ---------- 3. 找人声段 ----------
    hop = SR // 10
    frames = max(1, len(x) // hop)
    rms = np.array([np.sqrt((x[i * hop:(i + 1) * hop] ** 2).mean()) for i in range(frames)])
    med = float(np.median(rms))
    print("\n[3] 电平")
    print("  每 100ms RMS：中位 {:.0f} p10 {:.0f} p90 {:.0f} 最大 {:.0f}".format(
        med, np.percentile(rms, 10), np.percentile(rms, 90), rms.max()))
    loud = np.where(rms > max(400.0, med * 2.0))[0]
    print("  疑似人声的 100ms 格：{} 个（{:.1f} s）".format(len(loud), len(loud) / 10.0))

    segs = []
    if len(loud):
        # 把连续的 loud 合成段
        start = prev = loud[0]
        for idx in loud[1:]:
            if idx - prev > 3:
                segs.append((start, prev))
                start = idx
            prev = idx
        segs.append((start, prev))
        segs = [(a, b) for a, b in segs if (b - a) >= 2]
        print("  人声段（按 16k 时间轴）：{}".format(
            ", ".join("{:.1f}~{:.1f}s".format(a / 10.0, (b + 1) / 10.0) for a, b in segs[:6])))

    if not segs:
        print("\n✗ 这段窗口里没有明显人声，重跑一次并大声喊。")
        return 1

    # 取最长的两段，各留前后 0.4 s 余量
    segs.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
    picks = segs[:2]

    # ---------- 4. 频谱 + 多采样率假设 ----------
    print("\n[4] 频谱（判断是不是语音、有没有被压到低频）")
    spectrum_profile(x[:SR * 2], tag="底噪段  ")
    for a, b in picks:
        lo = max(0, (a - 4) * hop)
        hi = min(len(x), (b + 5) * hop)
        spectrum_profile(x[lo:hi], tag="人声段  ")

    print("\n[5] 多采样率假设（同一段人声，按不同源采样率重采样到 16k）")
    print("    若某个假设突然出高分 -> SDK 实际给的不是 16k")
    for a, b in picks:
        lo = max(0, (a - 4) * hop)
        hi = min(len(x), (b + 5) * hop)
        seg = x[lo:hi]
        print("  --- 段 {:.1f}~{}s（{} 样本）---".format(lo / SR, hi / SR, len(seg)))
        for guess in (16000, 18000, 22050, 24000, 32000, 44100, 48000):
            y = resample(seg, guess, SR)
            s, err = score(to_pcm(y))
            print("    源 {:>5} Hz -> {:>6} 样本（{:5.2f}s @16k）  最高分 {:.3f}{}".format(
                guess, len(y), len(y) / float(SR), s, "  " + err if err else ""))

    # ---------- 6. 整段 ----------
    print("\n[6] 整段按 16k 直喂（复现线上表现）")
    print("    最高分 {:.3f}".format(score(to_pcm(x))[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
