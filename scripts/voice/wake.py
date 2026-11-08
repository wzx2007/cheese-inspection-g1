# -*- coding: utf-8 -*-
"""本地唤醒词：仓库训练好的 ``xingbao.tflite``（openWakeWord 三段式流水线）。

为什么换掉「VAD + ASR 文本唤醒」
--------------------------------
第一版是"按音量断句 → 整句发 DashScope 识别 → 看文本里有没有星宝"。三个毛病：

1. 每段声音都要发一次 ASR —— 花钱、要联网、每次 1 s 起步；
2. 阈值靠现场调，展厅底噪一高就"一直以为在说话"；
3. 环境里**任何人**说话都会被送去识别，日志里全是无关内容。

仓库 ``github.com/wzx2007/xingbao`` 里带了训练好的唤醒模型，本地推理就能判：

===============================  ==========================================
``model_voice/xingbao.tflite``              唤醒分类器（"星宝星宝"）
``.../openwakeword/melspectrogram.tflite``  前处理：log-mel 特征
``.../openwakeword/embedding_model.tflite`` 前处理：speech embedding
===============================  ==========================================

三个模型串起来就是 openWakeWord 的流水线：每 1280 采样（80 ms）出一次分数，
CPU 上单帧 <5 ms，不需要联网。仓库自己的 ``multimodal/wake_word.py`` 也是这套。

机器人上装一次依赖::

    python3 -m pip install --user openwakeword==0.6.0 tflite-runtime

**为什么用桩模块加载**：``import openwakeword`` 会执行 ``openwakeword/__init__.py``，
那里 import 了 ``custom_verifier_model`` → scikit-learn → scipy。机器人的系统 scipy
是 1.3.3，和 numpy 1.24 不兼容（``np.typeDict`` 已被删除），整条链会抛
``AttributeError``。推理本身只用 numpy + tflite_runtime，所以这里把一个空壳包塞进
``sys.modules`` 顶掉 ``__init__``，只加载 ``openwakeword.model``。
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import List, Optional

FRAME_SAMPLES = 1280            # openWakeWord 的固定步长：16 kHz 下 80 ms
SAMPLE_RATE = 16000
BYTES_PER_FRAME = FRAME_SAMPLES * 2

# 模型内部有一个 1 s 的音频缓冲。**刚建好的实例里那块内存是未初始化的**，
# 头几帧预测会混进垃圾数据 —— 实测同一段音频三次拿到 0.33 / 0.41 / 0.57。
# 先喂 13 帧（1.04 s）静音把缓冲填满，分数立刻稳定（三次都是 0.323）。
# 所以每次 reset 之后都要重新预热，否则唤醒阈值就变成"看运气"。
PRIME_FRAMES = 13

MODEL_FILENAME = "xingbao.tflite"
MEL_FILENAME = "melspectrogram.tflite"
EMBEDDING_FILENAME = "embedding_model.tflite"

# 命中阈值。
# 2026-11-08 第三次现场反馈："阈值太高了"——0.30 在展厅里真人喊三四遍才中一次。
# 原因：这个 0.30 是从 TTS 音频上标定的，而 TTS 语速比真人快一倍（见
# README 的"唤醒分数的两个坑"），真人正常说话的分数分布和 TTS 根本不是一回事。
# 负样本实测 ≈0.001（几乎为 0），所以往下放到 0.15 很安全：离噪声底还很远，
# 但把"喊得轻/带口音/离得远"的那一批人捞进来。
# 还嫌不灵就 --wake-th 0.08；误唤多（自己把自己叫醒）就往上 0.25。
DEFAULT_THRESHOLD = 0.15
# 连续两帧过线才算命中：单帧偶发抖动会被这一条压掉，所以阈值可以放心压低。
PATIENCE_FRAMES = 2


def default_model_dir() -> Path:
    return Path(__file__).resolve().parent / "model_voice"


def _drop_cached_modules() -> None:
    """清掉 ``openwakeword*`` 的 import 缓存。

    一次失败的 ``import openwakeword`` 会把 ``openwakeword.model``/``utils``
    留在 ``sys.modules`` 里，而它们内部引用的仍是那个半成品包对象 —— 不清掉，
    后面装上的桩模块不生效。
    """
    for name in [k for k in list(sys.modules)
                 if k == "openwakeword" or k.startswith("openwakeword.")]:
        del sys.modules[name]


def load_model_class(verbose: bool = True):
    """拿到 openWakeWord 的 ``Model`` 类（绕开 sklearn 依赖）。"""
    try:
        from openwakeword.model import Model
        return Model
    except Exception as exc:                     # noqa: BLE001
        if verbose:
            print("  [wake] openwakeword 直接导入失败（改用桩模块）：{}".format(
                str(exc).splitlines()[0] if str(exc) else type(exc).__name__))

    _drop_cached_modules()
    spec = importlib.util.find_spec("openwakeword")
    if spec is None or not spec.origin:
        raise RuntimeError("没装 openwakeword。先跑：\n"
                           "  python3 -m pip install --user openwakeword==0.6.0 tflite-runtime")

    stub = types.ModuleType("openwakeword")
    stub.__path__ = [os.path.dirname(spec.origin)]
    stub.__file__ = spec.origin
    # 这几个常量/函数只在"按名字找预训练模型"时用到；我们永远显式传路径，
    # 给空实现即可（model.py 会无条件调用一次 get_pretrained_model_paths）。
    stub.MODELS = {}
    stub.FEATURE_MODELS = {}
    stub.VAD_MODELS = {}
    stub.model_class_mappings = {}
    stub.get_pretrained_model_paths = lambda inference_framework="tflite": []

    class _NoVAD(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("VAD 未启用（桩模块）")

    stub.VAD = _NoVAD
    sys.modules["openwakeword"] = stub

    module = importlib.import_module("openwakeword.model")
    if verbose:
        print("  [wake] 用桩模块加载 openwakeword.model 成功")
    return module.Model


class LocalWakeDetector(object):
    """把 PCM 流喂给 ``xingbao.tflite``，命中就返回 True。

    调用方式（必须**连续**喂，模型内部有 16 帧特征缓冲）::

        det = LocalWakeDetector()
        if det.prime():                 # 加载模型；False 表示不可用
            for chunk in mic_chunks:    # 每块 2560 字节 = 1280 采样
                if det.feed(chunk):
                    ...
                    det.reset()         # 命中后清状态，避免连续触发
    """

    name = "xingbao.tflite"

    def __init__(self, model_dir: Optional[str] = None,
                 threshold: Optional[float] = None,
                 patience: int = PATIENCE_FRAMES,
                 verbose: bool = True) -> None:
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.threshold = float(DEFAULT_THRESHOLD if threshold is None else threshold)
        self.patience = max(1, int(patience))
        self._verbose = bool(verbose)
        self._model = None
        self._np = None
        self._pending = None
        self._hits = 0
        self.last_score = 0.0
        self.error = ""

    # ------------------------------------------------------------------
    @property
    def model_path(self) -> Path:
        return self.model_dir / MODEL_FILENAME

    @property
    def melspectrogram_path(self) -> Path:
        return self.model_dir / "openwakeword" / MEL_FILENAME

    @property
    def embedding_model_path(self) -> Path:
        return self.model_dir / "openwakeword" / EMBEDDING_FILENAME

    def missing_files(self) -> List[str]:
        return [str(p) for p in (self.model_path, self.melspectrogram_path,
                                 self.embedding_model_path) if not p.is_file()]

    def available(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    def prime(self) -> bool:
        """加载模型。失败不抛异常，返回 False 并记下 ``self.error``（调用方降级）。"""
        if self._model is not None:
            return True
        missing = self.missing_files()
        if missing:
            self.error = "模型文件缺失：\n  " + "\n  ".join(missing)
            if self._verbose:
                print("  [wake] " + self.error.replace("\n", "\n  "))
            return False
        try:
            import numpy as np
            Model = load_model_class(verbose=self._verbose)
            self._model = Model(
                wakeword_models=[str(self.model_path)],
                inference_framework="tflite",
                melspec_model_path=str(self.melspectrogram_path),
                embedding_model_path=str(self.embedding_model_path),
            )
            self._np = np
            self._pending = np.empty(0, dtype="int16")
            self._warm_up()
        except Exception as exc:                 # noqa: BLE001
            self.error = "{}: {}".format(type(exc).__name__, exc)
            if self._verbose:
                print("  [wake] 模型加载失败：{}".format(self.error))
            self._model = None
            return False
        if self._verbose:
            print("  [wake] 本地模型就绪：{}  阈值 {:.2f}".format(
                self.model_path.name, self.threshold))
        return True

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.last_score = 0.0
        self._hits = 0
        if self._model is None:
            return
        try:
            reset = getattr(self._model, "reset", None)
            if callable(reset):
                reset()
        except Exception:                        # noqa: BLE001
            pass
        self._warm_up()

    def _warm_up(self) -> None:
        """喂 1 s 静音，把模型内部那块未初始化的音频缓冲填掉。

        不做这件事，同一段音频每次跑出来的分数能差 0.2 以上（见 PRIME_FRAMES
        的注释），唤醒就变成掷骰子。
        """
        if self._model is None or self._np is None:
            return
        silence = self._np.zeros(FRAME_SAMPLES, dtype="int16")
        try:
            for _ in range(PRIME_FRAMES):
                self._model.predict(silence)
        except Exception:                        # noqa: BLE001
            pass
        self._pending = self._np.empty(0, dtype="int16")

    def feed(self, chunk: bytes) -> bool:
        """喂一块 PCM（s16le / 16 k / 单声道），命中返回 True。

        判定要**连续 ``PATIENCE_FRAMES`` 帧**都超过阈值才算：单帧偶尔会抖，
        而唤醒词本身会连续拉高两三帧，用连续帧能压掉误触又不影响召回。
        """
        if self._model is None or not chunk:
            return False
        np = self._np
        usable = len(chunk) // 2 * 2
        if not usable:
            return False
        samples = np.frombuffer(chunk[:usable], dtype="<i2")
        if self._pending.size:
            samples = np.concatenate((self._pending, samples))
        hit = False
        cursor = 0
        while samples.size - cursor >= FRAME_SAMPLES:
            frame = samples[cursor:cursor + FRAME_SAMPLES]
            cursor += FRAME_SAMPLES
            try:
                scores = self._model.predict(frame)
            except Exception as exc:             # noqa: BLE001
                self.error = "推理失败：{}".format(exc)
                return False
            score = max(float(v) for v in scores.values()) if scores else 0.0
            if score > self.last_score:
                self.last_score = score
            if score > self.threshold:
                self._hits += 1
                if self._hits >= max(1, self.patience):
                    hit = True
                    self._pending = np.empty(0, dtype="int16")
                    break
            else:
                self._hits = 0
        if not hit and cursor < samples.size:
            self._pending = samples[cursor:].copy()
        return hit

    # ------------------------------------------------------------------
    def detect_pcm(self, pcm: bytes) -> float:
        """整段 PCM 跑一遍，返回最高分（自检用，不改动线上状态）。"""
        self.reset()
        peak = 0.0
        for offset in range(0, len(pcm) - BYTES_PER_FRAME + 1, BYTES_PER_FRAME):
            self.feed(pcm[offset:offset + BYTES_PER_FRAME])
            peak = max(peak, self.last_score)
        self.reset()
        return peak