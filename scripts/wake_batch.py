#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量跑 KWS：对 cache 里所有真人录音段逐条判定，统计命中率。

用法（机器人上）::

    python3 wake_batch.py heard      # 只测 heard_*.wav（真人说话段）
    python3 wake_batch.py all 60     # 各类各取 60 个

这三类文件含义不同：
* ``heard_*.wav``  —— VAD 切出来的真人说话段，最接近线上唤醒时的输入
* ``reply_*.wav``  —— 机器人自己 TTS 播报被麦克风录回来（回声）
* ``loopback_*``/``wake_probe`` —— 诊断用的整段录音

**不连机器人、不动机器人。** 只做离线推理。
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from voice import kws as kws_mod          # noqa: E402

CACHE = HERE / "voice" / "cache"


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as fh:
            frames = fh.readframes(fh.getnframes())
            rate = fh.getframerate() or 16000
            ch = fh.getnchannels() or 1
        return len(frames) / float(2 * ch * rate)
    except Exception:                         # noqa: BLE001
        return 0.0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    group = argv[0] if argv else "heard"
    limit = int(argv[1]) if len(argv) > 1 else 0

    pattern = {"heard": "heard_*.wav", "reply": "reply_*.wav",
               "all": "*.wav"}.get(group, group)
    files = sorted(CACHE.glob(pattern))
    # 太短的（<0.4 s）没有判别价值，跳过
    files = [p for p in files if wav_duration(p) >= 0.4]
    if limit:
        files = files[:limit]
    if not files:
        print("没有匹配 {} 的文件".format(pattern))
        return 2

    det, used = kws_mod.build_detector(engine="kws", verbose=True)
    if det is None:
        print("✗ KWS 加载失败")
        return 2
    print("\n批量判定 {} 个文件（{}，阈值 {:.2f}）".format(
        len(files), used, det.threshold))
    print("=" * 62)

    hits = []
    for i, p in enumerate(files, 1):
        dur = wav_duration(p)
        try:
            ok = det.detect_wav(p) > 0
        except Exception as exc:              # noqa: BLE001
            print("  {:<22} {:.2f}s  推理异常 {}".format(p.name, dur, exc))
            continue
        if ok:
            hits.append((p.name, dur, getattr(det, "last_text", "")))
            print("  {:<22} {:.2f}s  >>> 命中 {}".format(
                p.name, dur, "「{}」".format(det.last_text) if det.last_text else ""))
        elif i % 10 == 0:
            print("  … 已测 {}/{}".format(i, len(files)))

    print("=" * 62)
    print("{} / {} 命中（{:.1f}%）".format(
        len(hits), len(files), 100.0 * len(hits) / len(files)))
    for name, dur, text in hits:
        print("  命中：{:<22} {:.2f}s  {}".format(name, dur, text))
    if not hits:
        print("\n一个都没命中 —— 说明这批真人录音里 KWS 一行都没识别出关键词。")
        print("下一步：把其中 3~5 个拉到 PC 做 ASR，确认里面到底说的是什么。")
    return 0 if hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
