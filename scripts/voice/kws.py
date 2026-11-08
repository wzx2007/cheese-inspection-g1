# -*- coding: utf-8 -*-
"""本地唤醒词：sherpa-onnx 关键词检测（zipformer KWS，纯 ONNX）。

为什么换掉 openWakeWord
-----------------------
旧方案是 ``xingbao.tflite`` + ``melspectrogram.tflite`` + ``embedding_model.tflite``
三段式。2026-11-08 现场"唤不醒"排查结论：

======================  ====================================================
观察                    含义
======================  ====================================================
麦克风流正常           466 块 / 37 s，块大小恒 2560 字节
音频干净               16 kHz 单声道 S16_LE，DC 偏置 -13
人声进得来             喊话时 RMS 7198（本底 1398，5.1 倍信噪比）
**模型分数恒 0.001**    连 TTS 合成的"星宝星宝"也只有 0.001
`melspectrogram.tflite` 用新版 TFLite 载入直接失败：
                        ``XNNPack delegate failed to reshape runtimeNode
                        number 27``；关掉默认委托又是
                        ``Node number 3 (CONV_2D) failed to prepare``
======================  ====================================================

结论：**前处理模型和 TFLite 运行时版本绑得太死**。现场只要运行时的
XNNPack / builtin kernel 组合一变，mel 就出不了有效特征，后面的 embedding
与分类器拿到的全是垃圾，输出恒为 0.001 —— 阈值降到 0.04 也没用，因为
它压根没在"听"。

现在的方案：``sherpa-onnx`` 的 zipformer KWS（关键词检测）
---------------------------------------------------------
* 单一 ONNX 模型（encoder/decoder/joiner，int8 共 4.8 MB），特征提取在 C++
  里做，**没有独立的 mel 模型**，所以没有上面那类版本错配问题；
* 自带中文词表（wenetspeech 训练），唤醒词用拼音写在 ``keywords.txt`` 里，
  改唤醒词不用重训；
* 不依赖 ``openwakeword`` / ``tflite_runtime`` / ``sklearn``；
* CPU 上实时率远低于 1，机器人 ARM 上也跑得动。

实测（2026-11-08，PC 端，同一套模型）::

    官方自带 7 段真人音频 + 官方词表   -> 命中 4/7（其余不含该词）
    自定义「星宝星宝」跑那 7 段无关音频 -> 误唤 0/7
    自定义「星宝星宝」跑 TTS 合成音频   -> 命中 2/4
                                          （纯"星宝星宝"、带后接句的都中）

安装::

    python3 -m pip install --user sherpa-onnx

模型目录（``voice/model_kws/``，随仓库一起部署）::

    encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx
    decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx
    joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx
    tokens.txt      词表
    keywords.txt    唤醒词（拼音 + 声调，@后面写中文备注）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

SAMPLE_RATE = 16000
# 命中阈值。0.25 是模型自带示例的默认值；实测"星宝星宝"在这种量级上稳定命中，
# 而 7 段无关真人音频零误唤。嫌迟钝往下 0.15，误唤多往上 0.35。
DEFAULT_THRESHOLD = 0.25
# 关键词加分（keywords_score）。1.0 = 不额外加权。
DEFAULT_BOOST = 1.0

MODEL_DIR_NAME = "model_kws"
ENCODER_GLOB = "encoder-*.int8.onnx"
DECODER_GLOB = "decoder-*.int8.onnx"
JOINER_GLOB = "joiner-*.int8.onnx"


def default_model_dir() -> Path:
    return Path(__file__).resolve().parent / MODEL_DIR_NAME


class SherpaKwsDetector(object):
    """把 PCM 流喂给 sherpa-onnx 的关键词检测器，命中返回 True。

    调用方式与旧的 ``LocalWakeDetector`` 完全一致，方便上层无感切换::

        det = SherpaKwsDetector()
        if det.prime():
            for chunk in mic_chunks:            # 任意长度，内部会攒帧
                if det.feed(chunk):
                    ...                          # 命中
                    det.reset()
    """

    name = "sherpa-kws"

    def __init__(self, model_dir: Optional[str] = None,
                 threshold: Optional[float] = None,
                 boost: float = DEFAULT_BOOST,
                 verbose: bool = True) -> None:
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.threshold = float(DEFAULT_THRESHOLD if threshold is None
                               else threshold)
        self.boost = float(boost)
        self._verbose = bool(verbose)
        self._sp = None
        self._stream = None
        self._np = None
        # 构造 KeywordSpotter 时用的那一组参数。threshold / boost 改了必须
        # 重建 spotter 才生效 —— 这是 sherpa 的 C++ 对象，属性只读。
        self._built_with = None
        self.last_score = 0.0
        self.last_text = ""
        self.error = ""

    # ------------------------------------------------------------------
    @property
    def keywords_path(self) -> Path:
        return self.model_dir / "keywords.txt"

    @property
    def tokens_path(self) -> Path:
        return self.model_dir / "tokens.txt"

    def _pick(self, pattern: str) -> Optional[Path]:
        hits = sorted(self.model_dir.glob(pattern))
        return hits[0] if hits else None

    def missing_files(self) -> List[str]:
        out = []
        for p in (self.tokens_path, self.keywords_path):
            if not p.is_file():
                out.append(str(p))
        for pattern, label in ((ENCODER_GLOB, "encoder"),
                               (DECODER_GLOB, "decoder"),
                               (JOINER_GLOB, "joiner")):
            if self._pick(pattern) is None:
                out.append("{}（{}）".format(label, pattern))
        return out

    def available(self) -> bool:
        return self._sp is not None

    # ------------------------------------------------------------------
    def prime(self) -> bool:
        """加载模型。失败不抛异常，返回 False 并记下 ``self.error``。"""
        key = (round(self.threshold, 4), round(self.boost, 4))
        if self._sp is not None and self._built_with == key:
            return True
        if self._sp is not None:
            # 阈值变了：旧的 C++ 对象参数改不了，只能重建。
            self._sp = None
            self._stream = None
        missing = self.missing_files()
        if missing:
            self.error = "模型文件缺失：\n  " + "\n  ".join(missing)
            if self._verbose:
                print("  [kws] " + self.error.replace("\n", "\n  "))
            return False
        try:
            import numpy as np
            import sherpa_onnx
        except Exception as exc:                     # noqa: BLE001
            self.error = ("没装 sherpa-onnx（{}）。先跑：\n"
                          "  python3 -m pip install --user sherpa-onnx".format(
                              type(exc).__name__))
            if self._verbose:
                print("  [kws] " + self.error.replace("\n", "\n  "))
            return False

        try:
            import inspect
            kwargs = dict(
                tokens=str(self.tokens_path),
                encoder=str(self._pick(ENCODER_GLOB)),
                decoder=str(self._pick(DECODER_GLOB)),
                joiner=str(self._pick(JOINER_GLOB)),
                keywords_file=str(self.keywords_path),
                num_threads=2,
                keywords_score=self.boost,
                keywords_threshold=self.threshold,
                provider="cpu",
            )
            # 不同版本的 sherpa-onnx 参数名有出入（2.0 没有 debug）。
            # 只传它签名里真有的，避免一升级就崩。
            try:
                allowed = set(
                    inspect.signature(
                        sherpa_onnx.KeywordSpotter.__init__).parameters)
                if "debug" in allowed:
                    kwargs["debug"] = False
            except (TypeError, ValueError):
                pass
            self._sp = sherpa_onnx.KeywordSpotter(**kwargs)
            self._np = np
            self._stream = self._sp.create_stream()
            self._built_with = key
        except Exception as exc:                     # noqa: BLE001
            self.error = "{}: {}".format(type(exc).__name__, exc)
            if self._verbose:
                print("  [kws] 模型加载失败：{}".format(self.error))
            self._sp = None
            return False

        if self._verbose:
            words = [ln.split("@")[-1].strip()
                     for ln in self.keywords_path.read_text(
                         encoding="utf-8").splitlines() if "@" in ln]
            print("  [kws] 本地模型就绪：sherpa-onnx KWS，阈值 {:.2f}，"
                  "关键词 {}".format(self.threshold, "/".join(words) or "（空）"))
        return True

    def set_threshold(self, threshold: float) -> bool:
        """改阈值并重建 spotter。**直接赋值 ``det.threshold`` 是不生效的** ——
        ``keywords_threshold`` 是 ``KeywordSpotter`` 构造时固化到 C++ 对象里的，
        改 Python 属性只改了个数，模型那边还是老值。

        返回是否重建成功（没装模型时直接返回 False，下次 prime 会用新值）。
        """
        self.threshold = float(threshold)
        if self._sp is None:
            return False
        return self.prime()

    def set_boost(self, boost: float) -> bool:
        """改关键词加分并重建。语义同 ``set_threshold``。"""
        self.boost = float(boost)
        if self._sp is None:
            return False
        return self.prime()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """清空解码状态（命中后、或跳过一段音频时调用）。"""
        self.last_score = 0.0
        self.last_text = ""
        if self._sp is None or self._stream is None:
            return
        try:
            self._sp.reset_stream(self._stream)
        except Exception:                            # noqa: BLE001
            pass

    def feed(self, chunk: bytes) -> bool:
        """喂一块 PCM（s16le / 16 k / 单声道），命中返回 True。

        块长度随意：sherpa 内部自己攒帧，所以我们不需要像旧方案那样
        手动凑 1280 样本。
        """
        if self._sp is None or self._stream is None or not chunk:
            return False
        np = self._np
        usable = len(chunk) // 2 * 2
        if not usable:
            return False
        try:
            samples = np.frombuffer(chunk[:usable], dtype="<i2")
            wave = samples.astype(np.float32) / 32768.0
            self._stream.accept_waveform(SAMPLE_RATE, wave)
        except Exception as exc:                     # noqa: BLE001
            self.error = "喂音频失败：{}".format(exc)
            return False

        # 每次喂完把能解的帧解掉。guard 防止异常情况下死循环。
        hit = False
        guard = 0
        try:
            while self._sp.is_ready(self._stream) and guard < 16:
                guard += 1
                self._sp.decode_stream(self._stream)
                text = self._sp.get_result(self._stream)
                if text:
                    self.last_text = text
                    # KWS 不给概率，命中的就是这次配置的关键词；
                    # last_score 用 1.0 表示"确认命中"，和旧接口语义对齐。
                    self.last_score = 1.0
                    self._sp.reset_stream(self._stream)
                    hit = True
                    break
        except Exception as exc:                     # noqa: BLE001
            self.error = "解码失败：{}".format(exc)
            return False
        return hit

    # ------------------------------------------------------------------
    @staticmethod
    def _resample(samples, src_rate: int, dst_rate: int = SAMPLE_RATE):
        """线性插值重采样（只用于离线验证，线上麦克风本来就是 16 k）。"""
        import numpy as np
        if src_rate == dst_rate or samples.size == 0:
            return samples
        n_out = max(2, int(round(samples.size * dst_rate / float(src_rate))))
        src_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(dst_x, src_x, samples)

    def detect_pcm(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> float:
        """整段 PCM 跑一遍，命中返回 1.0（自检用，会重置状态）。"""
        import numpy as np
        if not pcm:
            return 0.0
        samples = np.frombuffer(pcm[: len(pcm) // 2 * 2],
                                dtype="<i2").astype(np.float32)
        samples = self._resample(samples, sample_rate)
        samples = samples / 32768.0
        self.reset()
        hit = False
        hit_text = ""
        step = SAMPLE_RATE // 20                    # 50 ms
        for offset in range(0, len(samples), step):
            block = samples[offset:offset + step]
            try:
                self._stream.accept_waveform(SAMPLE_RATE, block)
            except Exception:                        # noqa: BLE001
                break
            guard = 0
            while self._sp.is_ready(self._stream) and guard < 16:
                guard += 1
                self._sp.decode_stream(self._stream)
                text = self._sp.get_result(self._stream)
                if text:
                    hit_text = text
                    self.last_score = 1.0
                    self._sp.reset_stream(self._stream)
                    hit = True
                    break
            if hit:
                break
        self.reset()
        # reset 会清 last_text，但离线验证时调用方想看到命中了哪个词
        self.last_text = hit_text
        return 1.0 if hit else 0.0

    def detect_wav(self, path) -> float:
        """读 wav（任意采样率/声道）并检测，命中返回 1.0。"""
        import wave
        with wave.open(str(path), "rb") as fh:
            channels = fh.getnchannels()
            width = fh.getsampwidth()
            rate = fh.getframerate()
            raw = fh.readframes(fh.getnframes())
        if width != 2:
            raise RuntimeError("只支持 16 bit PCM wav：{}".format(path))
        if channels > 1:
            import numpy as np
            arr = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)[:, 0]
            raw = arr.tobytes()
        return self.detect_pcm(raw, sample_rate=rate)


def build_detector(engine: str = "kws", model_dir: Optional[str] = None,
                   threshold: Optional[float] = None,
                   verbose: bool = True):
    """按名字造唤醒器。返回 ``(detector, engine_name)``，失败返回 ``(None, "")``。

    ``engine``:
        ``kws``    sherpa-onnx 关键词检测（默认，推荐）
        ``model``  旧的 openWakeWord 三段式（保留，现场排查用）
        ``auto``   先 kws，不行再 model
    """
    engine = (engine or "kws").lower()
    order = []
    if engine in ("kws", "auto", "kws-first"):
        order.append("kws")
    if engine in ("model", "auto", "kws-first"):
        order.append("model")

    for name in order:
        if name == "kws":
            det = SherpaKwsDetector(model_dir=model_dir, threshold=threshold,
                                    verbose=verbose)
        else:
            from . import wake as wake_mod
            det = wake_mod.LocalWakeDetector(model_dir=model_dir,
                                             threshold=threshold,
                                             verbose=verbose)
        if det.prime():
            return det, name
        if verbose:
            print("  [wake] {} 后端不可用，{}".format(
                name, "继续试下一个" if name != order[-1] else "只能用文本/ASR"))
    return None, ""


def python_ok() -> str:
    """部署前自检：报告解释器版本（sherpa-onnx 对 py 版本有要求）。"""
    return "{}.{}".format(sys.version_info[0], sys.version_info[1])


if __name__ == "__main__":                            # pragma: no cover
    pcm_path = sys.argv[1] if len(sys.argv) > 1 else ""
    det = SherpaKwsDetector(verbose=True)
    if not det.prime():
        raise SystemExit("模型不可用：" + det.error)
    print("  解释器 Python", python_ok())
    if pcm_path and os.path.isfile(pcm_path):
        data = Path(pcm_path).read_bytes()
        print("  {} 命中={}".format(pcm_path, det.detect_pcm(data) > 0))
