# -*- coding: utf-8 -*-
"""唤醒模型体检：模型签名 + 已知样本打分 + 现场合成能拿多少分。

机器人不动，纯离线推理。
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

CACHE = HERE / "voice" / "cache"
MODEL_DIR = HERE / "voice" / "model_voice"


def read_wav(path):
    with wave.open(str(path), "rb") as f:
        ch, sw, sr, n = f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()
        raw = f.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.int16)
    if ch > 1:
        x = x.reshape(-1, ch)[:, 0]
    return sr, x


def resample(x, src, dst=16000):
    if src == dst or not len(x):
        return x.astype(np.int16)
    n_out = max(2, int(round(len(x) * dst / float(src))))
    return np.clip(np.interp(np.linspace(0, len(x) - 1, n_out),
                             np.arange(len(x), dtype=np.float64),
                             x.astype(np.float64)), -32768, 32767).astype(np.int16)


def tflite_io(path):
    """打印 tflite 模型的输入输出签名。"""
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:                                # noqa: BLE001
        try:
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        except Exception as exc:                     # noqa: BLE001
            return "  （无法导入 tflite 解释器：{}）".format(exc)
    try:
        it = Interpreter(model_path=str(path))
        it.allocate_tensors()
        lines = []
        for d in it.get_input_details():
            lines.append("    IN  {} shape={} dtype={}".format(
                d["name"], d["shape"], np.dtype(d["dtype"]).name))
        for d in it.get_output_details():
            lines.append("    OUT {} shape={} dtype={}".format(
                d["name"], d["shape"], np.dtype(d["dtype"]).name))
        return "\n".join(lines)
    except Exception as exc:                         # noqa: BLE001
        return "  （读取失败：{}）".format(exc)


def main():
    print("=" * 74)
    print("[0] 模型签名（判断是不是 openWakeWord 三段式）")
    print("=" * 74)
    for name in ("xingbao.tflite", "openwakeword/melspectrogram.tflite",
                 "openwakeword/embedding_model.tflite"):
        p = MODEL_DIR / name
        print("  {}  {} 字节".format(name, p.stat().st_size if p.is_file() else "缺失"))
        if p.is_file():
            print(tflite_io(p))

    det = wake_mod.LocalWakeDetector(verbose=False)
    if not det.prime():
        print("✗ 模型加载失败:", det.error)
        return 2

    def score_arr(x16):
        return det.detect_pcm(x16.tobytes())

    print()
    print("=" * 74)
    print("[1] 已知样本：sim_wake.wav（模拟测试里报过 0.697 的那段）")
    print("=" * 74)
    p = CACHE / "sim_wake.wav"
    if p.is_file():
        sr, x = read_wav(p)
        y = resample(x, sr)
        print("  {} Hz {} 样本 {:.2f}s  RMS {:.0f}".format(
            sr, len(x), len(x) / float(sr), float(np.sqrt((x.astype(np.float64) ** 2).mean()))))
        print("  原速最高分 {:.3f}".format(score_arr(y)))
        for sp in (0.67, 0.55, 0.8):
            yy = resample(y, int(16000 * sp), 16000)
            print("  放慢 x{:.2f} 最高分 {:.3f}".format(sp, score_arr(yy)))
    else:
        print("  没有 sim_wake.wav")

    print()
    print("=" * 74)
    print("[2] 现场合成「星宝星宝」再喂模型（排除 TTS 里没这几个字的可能）")
    print("=" * 74)
    try:
        from voice.cloud import CloudClient
        cloud = CloudClient(verbose=True)
        for text in ("星宝星宝", "星宝星宝，你去检测一下这块奶酪。"):
            out = CACHE / "selftest_tts.wav"
            cloud.tts(text, out)
            sr, x = read_wav(out)
            y = resample(x, sr)
            print("  文本「{}」  {} Hz {:.2f}s".format(text, sr, len(x) / float(sr)))
            print("    原速最高分 {:.3f}".format(score_arr(y)))
            for sp in (0.67, 0.55):
                yy = resample(y, int(16000 * sp), 16000)
                print("    放慢 x{:.2f} 最高分 {:.3f}".format(sp, score_arr(yy)))
    except Exception as exc:                         # noqa: BLE001
        import traceback
        print("  合成失败：{}".format(exc))
        traceback.print_exc()

    print()
    print("=" * 74)
    print("[3] 负样本对照：底噪段应当接近 0")
    print("=" * 74)
    p = CACHE / "wake_why.wav"
    if p.is_file():
        sr, x = read_wav(p)
        y = resample(x, sr)
        print("  wake_why.wav 最高分 {:.3f}".format(score_arr(y)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
