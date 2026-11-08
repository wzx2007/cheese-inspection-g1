#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阈值扫描：对 cache 里的真人录音跑多个阈值，给出召回/误唤曲线。

用法（机器人上）::

    python3 wake_sweep.py heard 0.05 0.10 0.15 0.20 0.25 0.30

为什么要这个脚本
----------------
``keywords_threshold`` 是 ``KeywordSpotter`` 构造时固化的，直接改
``det.threshold`` 不生效（曾经的坑：以为降到 0.05 没用，其实是没生效）。
这里用 ``set_threshold()`` 真正重建模型，逐个阈值重跑。

**不连机器人、不动机器人。**
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from voice import kws as kws_mod          # noqa: E402

CACHE = HERE / "voice" / "cache"


def dur_of(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as fh:
            n = len(fh.readframes(fh.getnframes()))
            rate = fh.getframerate() or 16000
            ch = fh.getnchannels() or 1
        return n / float(2 * ch * rate)
    except Exception:                         # noqa: BLE001
        return 0.0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    pattern = argv[0] if argv else "heard_*.wav"
    thresholds = [float(x) for x in argv[1:]] or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    files = sorted(CACHE.glob(pattern))
    files = [p for p in files if dur_of(p) >= 0.4]
    if not files:
        print("没有匹配 {}".format(pattern))
        return 2
    print("测试集：{} 个文件（{}）".format(len(files), pattern))

    det, used = kws_mod.build_detector(engine="kws", threshold=thresholds[0],
                                       verbose=False)
    if det is None:
        print("✗ KWS 加载失败")
        return 2
    print("后端：{}\n".format(used))

    table = {}
    for th in thresholds:
        if not det.set_threshold(th):
            print("  重建 spotter 失败（阈值 {:.2f}）".format(th))
            continue
        started = time.time()
        hits = []
        for p in files:
            try:
                if det.detect_wav(p) > 0:
                    hits.append(p.name)
            except Exception:                 # noqa: BLE001
                pass
        table[th] = hits
        print("阈值 {:.2f}：命中 {:>4} / {}（{:.1f}%）  用时 {:.1f}s".format(
            th, len(hits), len(files), 100.0 * len(hits) / len(files),
            time.time() - started))

    print("\n逐阈值命中清单")
    print("=" * 70)
    base = table.get(thresholds[0], [])
    for th in thresholds:
        if th not in table:
            continue
        hits = table[th]
        extra = [h for h in hits if h not in base]
        print("{:.2f}  {}".format(th, " ".join(h[6:9] for h in hits) or "（无）"))
        if extra and th != thresholds[0]:
            print("      ↑ 比最低阈值多出来的：{}".format(" ".join(h[6:9] for h in extra)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
