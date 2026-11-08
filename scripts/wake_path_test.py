#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不需要人喊，验证"唤醒线路"本身通不通。

思路：把已知能被 KWS 命中的「星宝星宝」PCM，按麦克风真实的 80 ms 块大小
喂进 **真正的** ``Assistant.wait_for_wake()``（把 MicRecorder.read 换成一个
回放器）。这样能干净地把两种失败分开：

  * 线路坏了（比如 reset 把解码器状态清了、或者根本没调用 feed）
    -> 回放也唤不醒
  * 线路没问题，是现场收音/音量不够
    -> 回放能唤醒，那就去查麦克风和距离

机器人不动，只碰 CPU。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path("/home/galbot/scripts")
sys.path.insert(0, str(HERE))

import numpy as np                                    # noqa: E402

from voice import kws as kws_mod                      # noqa: E402
from voice import mic as mic_mod                      # noqa: E402


def load_pcm(name):
    p = HERE / "voice" / name
    if not p.is_file():
        return b""
    return p.read_bytes()


def test_direct():
    """1. 直接把这支 PCM 丢给 KWS，确认它是能命中的素材。"""
    pcm = load_pcm("kws_sample.pcm")
    if not pcm:
        print("  ✗ 找不到 voice/kws_sample.pcm")
        return None
    det = kws_mod.SherpaKwsDetector(verbose=False)
    if not det.prime():
        print("  ✗ KWS 模型加载失败：", det.error)
        return None
    hit = det.detect_pcm(pcm)
    print("  素材 %.2f s，直接检测命中 = %s（%s）"
          % (len(pcm) / 32000.0, bool(hit), det.last_text or "-"))
    return pcm


def test_through_wiring(pcm):
    """2. 按麦克风真实块大小回放，走 Assistant.wait_for_wake 全线路。"""
    from voice import assistant as asst_mod

    det = kws_mod.SherpaKwsDetector(verbose=False)
    if not det.prime():
        print("  ✗ KWS 模型加载失败")
        return False

    # 造一个"假 Assistant"，只保留 wait_for_wake 需要的那几个属性
    class FakeMic(object):
        def __init__(self, data):
            self.data = data
            self.pos = 0
            self.started = False

        def start(self):
            self.started = True
            return "fake"

        def stop(self):
            pass

        def drain(self):
            return 0

        def read(self, timeout=0.5):
            if self.pos >= len(self.data):
                time.sleep(0.05)
                return None
            chunk = self.data[self.pos:self.pos + 2560]     # 80 ms，和真机一致
            self.pos += 2560
            return chunk

    fake = asst_mod.Assistant.__new__(asst_mod.Assistant)
    fake._mic = FakeMic(pcm + b"\x00" * 32000 * 2)    # 素材后面垫 2 s 静音
    fake._wake = det
    fake.wake_asr_fallback = False
    fake.wake_asr_cooldown = 3.0
    fake._verbose = True
    fake._wake_mute_until = 0.0

    t0 = time.time()
    try:
        res = asst_mod.Assistant._wait_for_wake_model(fake, keep_open=True)
    except Exception as exc:                          # noqa: BLE001
        print("  ✗ 线路抛异常：%s: %s" % (type(exc).__name__, exc))
        return False
    dt = time.time() - t0
    print("  回放线路结果：%r（%.1f s）" % (res, dt))
    # 命中后会去录尾巴，尾巴是静音 -> 返回空串是"正常命中但没听到话"
    return res is not None


def main():
    print("=" * 66)
    print("唤醒线路自检（不需要人喊）")
    print("=" * 66)
    print("\n[1] 素材本身能不能命中")
    pcm = test_direct()

    print("\n[2] 真实 wait_for_wake 线路 + 回放素材")
    if pcm:
        ok = test_through_wiring(pcm)
        print("\n" + "=" * 66)
        if ok:
            print("结论：线路是通的。唤不醒 -> 去查麦克风收音/距离/音量。")
        else:
            print("结论：线路断了。回放都唤不醒，说明代码没把音频喂到模型。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
