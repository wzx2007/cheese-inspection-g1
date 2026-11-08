# -*- coding: utf-8 -*-
"""机器人麦克风录音 + 简易 VAD（按音量切句）。

SDK 接口（见 ``docs/G1-API调用.md`` 第 10 节）::

    stream_id = robot.start_microphone_stream_input(callback, chunk_size=2560,
                                                    use_raw_audio=False)
    robot.stop_microphone_stream_input(stream_id)

``chunk_size=2560`` 字节 = 16 kHz / 16 bit / 单声道下的 **80 ms**，和官方
``audio_example.py`` 一致。回调在 SDK 的线程上跑，所以这里只做"塞进队列"，
任何耗时的事情（识别、合成）都不能放进回调，否则会丢帧。

**为什么要自己写 VAD**：现场没有静音检测就没法知道一句话什么时候说完。
这里不用模型，只按 RMS 阈值 + 连续静音时长断句 —— 对"星宝星宝，这个奶酪新鲜吗"
这种短句够用，而且完全可解释、可现场调。
"""

from __future__ import annotations

import array
import io
import queue
import sys
import threading
import time
import wave
from typing import Optional

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2                      # 16 bit
CHANNELS = 1
CHUNK_BYTES = 2560                    # 80 ms
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH

# VAD 默认值。阈值是"整段 80 ms 的 RMS"。
# 2026-11-08 上机发现：固定 120 在现场不够用 —— 展会环境（风机 + 人群 + 机器人自身
# 散热）底噪常常就过 120，结果"一直处于说话状态"，每轮都录满 8 s 才交出去，
# ASR 收到的是"嗯。"。所以改成**先估本底噪声再定阈值**：
#     实际阈值 = max(VAD_FLOOR_MIN, 本底中位数 × VAD_FLOOR_MULT)
# 2026-11-08 三次现场反馈都在说"阈值太高"：2.2 → 1.5 → **1.25**，下限 300 → 250 → 150。
# 现场日志证据：`[vad] 本底 1683 x1.5 -> 阈值 2525`，而人正常说话 RMS 常在
# 1500~2500，刚好卡在阈值下面 —— 于是"喊了但它像没听见"。
# 现在本底 1683 → 阈值 2104，本底 1100 → 1375。
# 太灵敏（噪声被当成人声、录到一堆废话）就 `--vad-mult 1.6` 往上调。
VAD_RMS_THRESHOLD = 120.0             # 绝对下限，估不出本底时才用它
VAD_FLOOR_MIN = 150.0                 # 自适应阈值不会低于这个
VAD_FLOOR_MULT = 1.25                 # 比本底噪声高这么多倍才算开口
VAD_CALIB_CHUNKS = 6                  # 开头这 6 块（480 ms）用来估本底
VAD_START_CHUNKS = 2                  # 连续 2 块（160 ms）超阈值算开口
VAD_SILENCE_S = 0.7                   # 说完后连续静音这么久算结束
VAD_MAX_S = 5.0                       # 单句最长。阈值调低后偶发"噪声一直超阈值"，
                                      # 8 s 太长白烧一次 ASR；5 s 足够说完一整句
VAD_MIN_S = 0.3                       # 太短的丢掉（咳嗽、碰麦克风）
VAD_START_TIMEOUT_S = 5.0             # 这么久没人开口就返回空，别死等 8 s


def pcm_rms(chunk: bytes) -> float:
    """一段 s16le PCM 的 RMS。空数据返回 0。"""
    if not chunk:
        return 0.0
    samples = array.array("h")
    samples.frombytes(chunk[: len(chunk) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return 0.0
    total = 0
    for value in samples:
        total += value * value
    return (total / float(len(samples))) ** 0.5


def pcm_duration(chunk: bytes) -> float:
    return len(chunk) / float(BYTES_PER_SECOND)


def pcm_to_wav(pcm: bytes, path, sample_rate: int = SAMPLE_RATE) -> str:
    """裸 PCM 包成 wav 文件（ASR 要 wav，不能直接喂裸流）。"""
    path = str(path)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return path


def wav_to_pcm(path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != SAMPLE_WIDTH:
            raise RuntimeError("wav 不是 16 bit：{}".format(path))
        frames = handle.readframes(handle.getnframes())
    if handle.getnchannels() > 1:            # 只取第一声道
        samples = array.array("h")
        samples.frombytes(frames)
        if sys.byteorder == "big":
            samples.byteswap()
        return samples[:: handle.getnchannels()].tobytes()
    return frames


class MicRecorder(object):
    """把 SDK 的麦克风流收进队列，供上层按句取用。"""

    def __init__(self, robot, chunk_size: int = CHUNK_BYTES,
                 use_raw_audio: bool = False, verbose: bool = True,
                 floor_min: float = VAD_FLOOR_MIN,
                 floor_mult: float = VAD_FLOOR_MULT) -> None:
        self._robot = robot
        self._chunk_size = int(chunk_size)
        self._raw = bool(use_raw_audio)
        self._verbose = bool(verbose)
        # 现场可调：底噪高就调大 floor_mult，太迟钝就调小。
        self.floor_min = float(floor_min)
        self.floor_mult = float(floor_mult)
        self._queue = queue.Queue(maxsize=64)
        self._stream_id = None
        self._dropped = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def start(self) -> str:
        if self._stream_id is not None:
            return self._stream_id
        self._stream_id = self._robot.start_microphone_stream_input(
            callback=self._on_audio, chunk_size=self._chunk_size,
            use_raw_audio=self._raw)
        if self._verbose:
            print("  麦克风已开：stream_id={} chunk={} 字节（{:.0f} ms）".format(
                self._stream_id, self._chunk_size, 1000.0 * self._chunk_size / BYTES_PER_SECOND))
        return self._stream_id

    def stop(self) -> None:
        stream_id, self._stream_id = self._stream_id, None
        if stream_id is None:
            return
        try:
            self._robot.stop_microphone_stream_input(stream_id)
        except Exception as exc:                 # noqa: BLE001
            print("  关闭麦克风异常：{}".format(exc))
        if self._verbose:
            print("  麦克风已关（丢帧 {}）".format(self._dropped))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.stop()

    # ------------------------------------------------------------------
    def _on_audio(self, data) -> None:
        """SDK 回调。**只入队**，不做任何重活。"""
        chunk = _as_bytes(data)
        if not chunk:
            return
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            # 宁可丢最旧的，也不要阻塞 SDK 线程
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(chunk)
            except queue.Empty:
                pass
            with self._lock:
                self._dropped += 1

    def drain(self) -> int:
        """清空积压，开始一次新的监听前调用，避免上一句的尾音混进来。"""
        count = 0
        while True:
            try:
                self._queue.get_nowait()
                count += 1
            except queue.Empty:
                return count

    def read(self, timeout: float = 0.5) -> Optional[bytes]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    def record_utterance(
        self,
        max_s: float = VAD_MAX_S,
        silence_s: float = VAD_SILENCE_S,
        min_s: float = VAD_MIN_S,
        threshold: float = 0.0,
        start_chunks: int = VAD_START_CHUNKS,
        calib_num: int = VAD_CALIB_CHUNKS,
        start_timeout_s: float = VAD_START_TIMEOUT_S,
        on_level=None,
        adaptive: bool = True,
    ) -> bytes:
        """录一句话，返回裸 PCM（16 kHz / 16 bit / 单声道）。

        没听到人说话就返回空 ``b""``（超时也算），调用方据此继续等。
        前面留 1 块的余量，避免把第一个字的声母切掉。

        ``threshold`` 给 0（默认）时**自适应**：先用开头 ``calib_num`` 块估本底
        噪声，实际阈值 = ``max(VAD_FLOOR_MIN, 本底 × VAD_FLOOR_MULT)``。
        显式传一个正数就完全按那个值走（调试用）。
        """
        calib_chunks_needed = max(1, int(calib_num))
        collected = []
        pre_roll = []
        voiced = 0
        silence_run = 0
        speaking = False
        started = time.monotonic()
        chunk_s = self._chunk_size / float(BYTES_PER_SECOND)
        silence_chunks_need = max(1, int(round(silence_s / chunk_s)))

        # 自适应阈值（边收边估，估完就固定，不再抖动）
        calib_levels = [] if (adaptive and threshold <= 0) else None
        calib_chunks = []
        eff = float(threshold) if threshold > 0 else VAD_RMS_THRESHOLD
        threshold_locked = calib_levels is None

        while time.monotonic() - started < max_s + silence_s:
            chunk = self.read(timeout=0.5)
            if chunk is None:
                if speaking and time.monotonic() - started > max_s:
                    break
                continue
            level = pcm_rms(chunk)
            if on_level is not None:
                on_level(level)

            # ---- 估本底 ----
            if not threshold_locked:
                calib_levels.append(level)
                calib_chunks.append(chunk)
                if len(calib_levels) >= max(1, calib_chunks_needed):
                    base = sorted(calib_levels)[len(calib_levels) // 2]   # 中位数
                    eff = max(self.floor_min, base * self.floor_mult)
                    threshold_locked = True
                    if self._verbose:
                        print("   [vad] 本底 {:.0f} x{:.1f} -> 阈值 {:.0f}".format(
                            base, self.floor_mult, eff))
                    # 估完把最后 3 块当余量留着（第一个字可能就在里面）
                    pre_roll.extend(calib_chunks[-3:])
                    calib_levels, calib_chunks = [], []
                continue

            if not speaking:
                if level >= eff:
                    voiced += 1
                    pre_roll.append(chunk)
                    if voiced >= max(1, start_chunks):
                        speaking = True
                        collected.extend(pre_roll)
                        pre_roll = []
                else:
                    voiced = 0
                    pre_roll.append(chunk)
                    pre_roll = pre_roll[-3:]          # 只留最近 240 ms 当余量
                if time.monotonic() - started > start_timeout_s:
                    return b""
                continue

            collected.append(chunk)
            if level < eff:
                silence_run += 1
            else:
                silence_run = 0
            if silence_run >= silence_chunks_need:
                break
            if pcm_duration(b"".join(collected)) >= max_s:
                break

        pcm = b"".join(collected)
        if pcm_duration(pcm) < min_s:
            return b""
        return pcm


def _as_bytes(data) -> bytes:
    """SDK 回调参数在不同版本里可能是 bytes / dict / numpy，统一成 bytes。"""
    if data is None:
        return b""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, dict):
        for key in ("data", "audio", "buffer", "pcm"):
            value = data.get(key)
            if value is not None:
                return _as_bytes(value)
        return b""
    raw = getattr(data, "data", None)
    if raw is not None:
        return _as_bytes(raw)
    try:
        return bytes(data)                       # numpy / array.array
    except Exception:                            # noqa: BLE001
        return b""


def wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()
