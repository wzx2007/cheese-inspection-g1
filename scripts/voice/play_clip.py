#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""让 Galbot G1 播放语音片段（``goodbye.wav`` / ``result_pass.wav`` / ``result_fail.wav``）。

SDK 的音频输出接口 ``write_audio_stream_output`` 只接受
**16000 Hz / 16 bit / 单声道 PCM**，不能直接播 WAV/MP3。本模块先用 ffmpeg
（没有则退 miniaudio）转成 PCM 并缓存，再每 2560 字节写一次、间隔 50 ms。

现场用的三支音频都是 **32 kHz**，所以这一层转换是必需的，不是可选项。

两种用法
--------
``VoiceBox``
    给 ``demo_cheese`` 用：预载全部片段，``say("pass")`` 入队，后台线程按
    FIFO 顺序播，不挡动作。识别完立刻播报结果靠它。
``play_pcm`` / 命令行
    独立播放/转换，用来在部署前把 PCM 转好，或现场单独试听::

        python3 play_clip.py --file result_pass.wav --volume 50
        python3 play_clip.py --convert-only --file goodbye.wav

安全说明：只初始化 SDK、设音量、写音频流，不使能关节、不控底盘、不做任何运动。
"""

from __future__ import annotations

import argparse
import array
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

# SDK 音频输出要求的固定格式：16 kHz、16 bit、单声道、小端 PCM。
TARGET_RATE = 16000
TARGET_CHANNELS = 1
BYTES_PER_SECOND = TARGET_RATE * 2  # s16le mono = 32 000 B/s
CHUNK_BYTES = 2560                  # 官方示例的块大小（80 ms 音频）
WRITE_INTERVAL_S = 0.05             # 官方示例每块写入后的间隔
DEFAULT_STREAM_ID = "demo_voice"
DEFAULT_STARTUP_DELAY_S = 5.0       # audio_example.py 初始化后等 5 s
DEFAULT_TAIL_MARGIN_S = 0.5         # 末块写入后额外等待，避免截掉尾音
DEFAULT_GAP_S = 0.30                # 两段语音之间的停顿


def pcm_cache_path(source: Path, pcm_out: Optional[str] = None) -> Path:
    """PCM 缓存路径：默认与源文件同目录的 ``<名字>_16k_mono.pcm``。"""
    if pcm_out:
        target = Path(pcm_out).expanduser()
        if target.is_dir():
            raise RuntimeError("--pcm-out 必须是文件路径，不能是目录: {}".format(target))
        return target
    if source.suffix.lower() == ".pcm":
        return source
    return source.with_name("{}_16k_mono.pcm".format(source.stem))


def validate_pcm(path: Path) -> int:
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError("PCM 文件为空: {}".format(path))
    if size % 2 != 0:
        raise RuntimeError("PCM 字节数不是 16 bit 对齐的偶数: {} ({} 字节)".format(path, size))
    return size


def pcm_peak(path: Path) -> int:
    """返回 PCM 的峰值绝对值；0 表示整段都是静音。"""
    peak = 0
    with open(path, "rb") as audio:
        while True:
            data = audio.read(65536)
            if not data:
                break
            samples = array.array("h")
            samples.frombytes(data)
            if sys.byteorder == "big":
                samples.byteswap()
            if samples:
                peak = max(peak, max(samples), abs(min(samples)))
    return peak


def _convert_with_ffmpeg(source: Path, target: Path) -> bool:
    """用 ffmpeg 转成 16 kHz / 16 bit / 单声道裸 PCM（内含抗混叠重采样）。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-vn",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_RATE),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        str(target),
    ]
    try:
        subprocess.run(command, check=True)
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        print("[警告] ffmpeg 转换失败: {}".format(exc), file=sys.stderr)
        return False


def _convert_with_miniaudio(source: Path, target: Path) -> bool:
    """用纯 Python 包 miniaudio（内含解码器）解码并重采样。"""
    try:
        import miniaudio  # type: ignore
    except ImportError:
        return False

    try:
        try:
            decoded = miniaudio.decode_file(
                str(source),
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=TARGET_CHANNELS,
                sample_rate=TARGET_RATE,
            )
        except Exception:
            # decode_file 对非 ASCII 路径会失败，字节内容是一样的
            decoded = miniaudio.decode(
                source.read_bytes(),
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=TARGET_CHANNELS,
                sample_rate=TARGET_RATE,
            )
        samples = array.array("h", decoded.samples)
        if sys.byteorder == "big":
            samples.byteswap()
        with open(target, "wb") as output:
            output.write(samples.tobytes())
        return True
    except Exception as exc:  # miniaudio 抛出的异常类型较杂，统一降级
        print("[警告] miniaudio 解码失败: {}".format(exc), file=sys.stderr)
        return False


def ensure_pcm(source: Path, target: Path, force: bool = False) -> Path:
    """确保存在可直接播放的 PCM；必要时从 WAV/MP3 转换并缓存。"""
    source = Path(source)
    if source.suffix.lower() == ".pcm":
        validate_pcm(source)
        return source

    if target == source:
        raise RuntimeError("PCM 输出路径不能与输入音频相同")

    if target.is_file() and not force:
        source_mtime = source.stat().st_mtime if source.is_file() else None
        if source_mtime is None or target.stat().st_mtime >= source_mtime:
            try:
                validate_pcm(target)
                return target
            except (OSError, RuntimeError):
                if source_mtime is None:
                    raise
    if not source.is_file():
        raise FileNotFoundError("找不到语音文件: {}".format(source))

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        converted = _convert_with_ffmpeg(source, temporary) or _convert_with_miniaudio(
            source, temporary
        )
        if not converted:
            raise RuntimeError(
                "无法把音频转成 PCM：本机没有 ffmpeg，也没有安装 miniaudio。\n"
                "请在 PC 上执行:\n"
                "  python play_clip.py --convert-only --file \"{}\"\n"
                "然后把生成的 {} 一起上传到机器人。".format(source, target.name)
            )
        validate_pcm(temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
    return target


def _load_robot_class():
    try:
        from galbot_sdk.g1 import GalbotRobot  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "未找到 galbot_sdk.g1，无法连接机器人。\n"
            "请在已部署 SDK 的机器人 Orin 上运行；如果只想先转换音频，"
            "请加 --convert-only。"
        ) from exc
    return GalbotRobot


class VoiceBox:
    """预载若干段语音，按 FIFO 顺序在后台线程里播，不挡主流程。

    现场要求是"每次识别完立刻播报当时的结果"，所以 ``say`` 必须是非阻塞的 ——
    播报的同时手臂还要继续走动作。用一个队列 + 单工作线程实现：

    * ``say("pass")`` 立即返回，片段排队；
    * 一次查看的语音通常 4~5 s，而两次查看之间隔十几秒动作，队列不会堆积；
    * ``say(..., wait=True)`` 阻塞到该片段播完（收尾的 goodbye 用它）。

    不调用 ``init``/``destroy``，机器人由调用方管理。
    """

    def __init__(self, robot, volume: Optional[float] = None,
                 stream_id: str = DEFAULT_STREAM_ID, verbose: bool = True):
        self._robot = robot
        self._stream_id = stream_id
        self._verbose = bool(verbose)
        self._clips = {}                       # name -> bytes
        self._durations = {}                   # name -> 秒
        self._keep = set()                     # load() 预载的常驻片段名
        self._pcm_lock = threading.Lock()      # TTS 动态片段和预载片段共用一个表
        self._queue = queue.Queue()
        self._quit = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._muted = False
        self._thread = threading.Thread(target=self._loop, name="voice-box", daemon=True)
        self._thread.start()
        if volume is not None:
            self.set_volume(volume)

    # ------------------------------------------------------------------
    # 装载
    # ------------------------------------------------------------------
    def load(self, name: str, source) -> bool:
        """把一个音频文件转成 PCM 并预载进内存。返回是否成功。"""
        source = Path(source)
        try:
            pcm = ensure_pcm(source, pcm_cache_path(source))
            data = pcm.read_bytes()
        except Exception as exc:  # noqa: BLE001
            print("  ⚠ 语音 {} 加载失败（{}）：{}".format(name, source.name, exc))
            return False
        validate_pcm(pcm)
        self._clips[name] = data
        self._keep.add(name)
        self._durations[name] = len(data) / float(BYTES_PER_SECOND)
        if self._verbose:
            print("  语音 {:<8} <- {}（{:.2f} s）".format(
                name, source.name, self._durations[name]))
        return True

    def has(self, name: str) -> bool:
        return name in self._clips

    def names(self):
        return list(self._clips)

    def mute(self, on=True):
        self._muted = bool(on)

    def say_pcm(self, data: bytes, label: str = "tts", wait: bool = False) -> bool:
        """直接播一段**已经是** 16 kHz/16 bit/单声道 的 PCM。

        给云端 TTS 用：合成结果每次都不一样，不能像固定片段那样预载成名字，
        所以单独开一个入口，按 ``label`` 临时塞进队列，播完就丢。
        """
        if not data or self._muted:
            return False
        if len(data) % 2:
            data = data[:-1]                       # 16 bit 对齐
        duration = len(data) / float(BYTES_PER_SECOND)
        with self._pcm_lock:
            self._clips[label] = data
            self._durations[label] = duration
        done = threading.Event()
        self._queue.put((label, done if wait else None))
        if wait:
            self._idle.clear()
            done.wait(duration + 8.0)
            self._idle.set()
        return True

    # ------------------------------------------------------------------
    # 音量
    # ------------------------------------------------------------------
    def set_volume(self, volume: float) -> bool:
        try:
            ok = self._robot.set_volume(float(volume))
        except Exception as exc:  # noqa: BLE001
            print("  设置音量失败：{}".format(exc))
            return False
        if self._verbose:
            print("  音量设为 {:.0f}".format(float(volume)))
        return bool(ok)

    # ------------------------------------------------------------------
    # 播放
    # ------------------------------------------------------------------
    def say(self, names, wait: bool = False) -> bool:
        """把若干片段排进队列。``wait=True`` 时阻塞到最后一个片段播完。"""
        if isinstance(names, str):
            names = [names]
        names = [n for n in names if n in self._clips]
        if not names or self._muted:
            if not names and self._verbose:
                print("  ⚠ 没有可播的语音片段，跳过")
            return False
        done = threading.Event()
        for i, name in enumerate(names):
            self._queue.put((name, done if (wait and i == len(names) - 1) else None))
        if wait:
            self._idle.clear()
            done.wait(sum(self._durations[n] for n in names) + 5.0 * len(names) + 3.0)
            self._idle.set()
        return True

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """等队列播空。收尾用，避免 shutdown 把尾音截掉。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._queue.empty() and self._idle.is_set():
                return True
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.05)

    def close(self, timeout: float = 8.0):
        """把队列播完再退出线程。"""
        self.wait_idle(timeout)
        self._quit.set()

    def _loop(self):
        while not self._quit.is_set() or not self._queue.empty():
            try:
                name, done = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._idle.clear()
            try:
                self._stream(self._clips[name], name, self._durations[name])
            except Exception as exc:  # noqa: BLE001
                print("  语音 {} 播放异常：{}".format(name, exc))
            finally:
                if name not in self._keep:      # TTS 临时片段，播完就丢，别攒内存
                    with self._pcm_lock:
                        self._clips.pop(name, None)
                        self._durations.pop(name, None)
                if done is not None:
                    done.set()
                self._idle.set()

    def _stream(self, data: bytes, name: str, duration: float):
        if self._verbose:
            print("  ♪ 播报 {}（{:.2f} s）".format(name, duration))
        self._stop_stream()
        started = time.monotonic()
        offset = 0
        written = 0
        while offset < len(data):
            if self._quit.is_set():
                break
            chunk = data[offset:offset + CHUNK_BYTES]
            offset += len(chunk)
            ok = self._robot.write_audio_stream_output(chunk, self._stream_id)
            written += 1
            if not ok:
                print("  语音写入失败（第 {} 块）".format(written))
                break
            time.sleep(WRITE_INTERVAL_S)
        deadline = started + duration + DEFAULT_TAIL_MARGIN_S
        while time.monotonic() < deadline:
            if self._quit.is_set():
                break
            time.sleep(0.02)
        self._stop_stream()

    def _stop_stream(self):
        try:
            self._robot.stop_audio_stream_output(self._stream_id)
        except Exception:  # noqa: BLE001
            pass


def play_pcm(
    pcm_path: Path,
    *,
    volume: Optional[float] = None,
    repeat: int = 1,
    stream_id: str = DEFAULT_STREAM_ID,
    startup_delay_s: float = DEFAULT_STARTUP_DELAY_S,
    tail_margin_s: float = DEFAULT_TAIL_MARGIN_S,
) -> None:
    """独立播放（自己 init/destroy 机器人），用于现场试听。"""
    size = validate_pcm(pcm_path)
    duration_s = size / float(BYTES_PER_SECOND)

    GalbotRobot = _load_robot_class()
    robot = GalbotRobot()
    if not robot.init():
        raise RuntimeError(
            "robot.init() 失败：请检查机器人是否上电、急停是否松开、"
            "网络/embosa 是否正常。"
        )
    try:
        if startup_delay_s > 0:
            print("等待 SDK 初始化 {:.1f} s ...".format(startup_delay_s))
            time.sleep(startup_delay_s)
        if volume is not None:
            print("系统音量 -> {}".format(volume))
            robot.set_volume(float(volume))
        for index in range(repeat):
            if not robot.is_running():
                raise RuntimeError("机器人控制已停止（is_running() 为 False）")
            print("[{}/{}] 播放 {}：{:.2f} s".format(
                index + 1, repeat, pcm_path.name, duration_s))
            started = time.monotonic()
            with open(str(pcm_path), "rb") as audio:
                while True:
                    chunk = audio.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    if not robot.write_audio_stream_output(chunk, stream_id):
                        raise RuntimeError("write_audio_stream_output 失败")
                    time.sleep(WRITE_INTERVAL_S)
            wait_s = started + duration_s + tail_margin_s - time.monotonic()
            time.sleep(max(wait_s, tail_margin_s))
        print("播放完成。")
    finally:
        try:
            robot.stop_audio_stream_output(stream_id)
        except Exception:  # noqa: BLE001
            pass
        try:
            robot.request_shutdown()
            robot.wait_for_shutdown()
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                robot.destroy()
            except Exception:  # noqa: BLE001
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 WAV/MP3 转成 16 kHz/16 bit/单声道 PCM 并在 Galbot G1 上播放。"
    )
    parser.add_argument("-f", "--file", required=True, help="WAV/MP3/PCM 路径")
    parser.add_argument("--volume", type=float, default=None, help="系统音量 0-100")
    parser.add_argument("--repeat", type=int, default=1, help="重复播放次数，默认 1")
    parser.add_argument("--convert-only", action="store_true",
                        help="只转 PCM，不连机器人")
    parser.add_argument("--force-convert", action="store_true",
                        help="忽略已有 PCM 缓存，强制重转")
    parser.add_argument("--pcm-out", default=None, help="PCM 输出路径")
    parser.add_argument("--stream-id", default=DEFAULT_STREAM_ID)
    parser.add_argument("--startup-delay", type=float, default=DEFAULT_STARTUP_DELAY_S)
    parser.add_argument("--tail-margin", type=float, default=DEFAULT_TAIL_MARGIN_S)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat 必须大于等于 1")
    if args.volume is not None and not 0.0 <= args.volume <= 100.0:
        parser.error("--volume 必须在 0-100 之间")

    try:
        source = Path(args.file).expanduser()
        if not source.is_file():
            raise FileNotFoundError("文件不存在: {}".format(source))
        target = pcm_cache_path(source, args.pcm_out)
        print("音源: {}\nPCM : {}".format(source, target))
        pcm = ensure_pcm(source, target, force=args.force_convert)
        size = validate_pcm(pcm)
        peak = pcm_peak(pcm)
        print("PCM {} 字节（{:.2f} s），峰值 {}/32767（{:.1f}%）".format(
            size, size / float(BYTES_PER_SECOND), peak, peak / 327.67))
        if peak == 0:
            print("[警告] PCM 整段为静音，检查音源本身", file=sys.stderr)
        if args.convert_only:
            print("--convert-only：不连机器人。")
            return 0
        play_pcm(pcm, volume=args.volume, repeat=args.repeat,
                 stream_id=args.stream_id, startup_delay_s=args.startup_delay,
                 tail_margin_s=args.tail_margin)
        return 0
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print("错误: {}".format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
