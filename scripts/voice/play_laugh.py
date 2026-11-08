#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""让 Galbot G1 播放 ``laugh.mp3``。

SDK 的音频输出接口只接受 **16000 Hz / 16 bit / 单声道 PCM**
(``write_audio_stream_output``)，不能直接播放 MP3。本脚本先把 MP3 转成 PCM
（优先 ffmpeg，其次 miniaudio），再按官方 ``audio_example.py`` 的方式每
2560 字节写入一次、间隔 50 ms，播放结束后等待队列播完再退出。

脚本运行位置
------------
按项目现在的部署方式，脚本放在机器人 Orin 上运行（需已部署 galbot_sdk）::

    python3 play_laugh.py
    python3 play_laugh.py --file /userdata/laugh.mp3 --volume 80
    python3 play_laugh.py --repeat 3

如果机器人上没有 ffmpeg / miniaudio，先在 PC 上只做转换，再把 PCM 和脚本
一起传到 ``/userdata``::

    uv pip install miniaudio                      # 仅首次
    python scripts/voice/play_laugh.py --convert-only --file scripts/voice/laugh.mp3
    scp scripts/voice/play_laugh.py scripts/voice/laugh_16k_mono.pcm galbot@<ROBOT_IP>:/userdata/

默认按以下顺序查找 ``laugh.mp3``：脚本所在目录（``scripts/voice``）、
``scripts`` 目录、项目根目录、当前目录、当前目录/assets、脚本目录/assets、
``scripts`` 目录/assets、项目根目录/assets。也可以用 ``--file`` 直接指定
路径（也可以是已经转好的 ``.pcm``）。

安全说明：本脚本只初始化 SDK、设置音量（可选）和播放音频，不使能关节、
不控制底盘、不做任何运动。
"""

from __future__ import annotations

import argparse
import array
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

# SDK 音频输出要求的固定格式：16 kHz、16 bit、单声道、小端 PCM。
TARGET_RATE = 16000
TARGET_CHANNELS = 1
BYTES_PER_SECOND = TARGET_RATE * 2  # s16le mono = 32 000 B/s
CHUNK_BYTES = 2560                  # 官方示例的块大小（80 ms 音频）
WRITE_INTERVAL_S = 0.05             # 官方示例每块写入后的间隔
DEFAULT_STREAM_ID = "laugh_playback"
DEFAULT_FILE_NAME = "laugh.mp3"
DEFAULT_STARTUP_DELAY_S = 5.0       # audio_example.py 初始化后等 5 s
DEFAULT_TAIL_MARGIN_S = 0.5         # 末块写入后额外等待，避免截掉尾音


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    seen = set()
    result = []
    for path in paths:
        key = os.path.normcase(os.path.abspath(str(path)))
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def resolve_source(explicit: Optional[str]) -> Path:
    """定位 MP3/PCM 文件；找不到时给出实际查找过的路径。"""
    if explicit:
        source = Path(explicit).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"文件不存在或不可读: {source}")
        return source.resolve()

    script_dir = Path(__file__).resolve().parent   # scripts/voice
    scripts_dir = script_dir.parent                # scripts
    project_root = scripts_dir.parent              # repository root
    cwd = Path.cwd()
    bases = _unique_paths(
        [
            script_dir,
            scripts_dir,
            project_root,
            cwd,
            cwd / "assets",
            script_dir / "assets",
            scripts_dir / "assets",
            project_root / "assets",
        ]
    )
    candidates = [base / DEFAULT_FILE_NAME for base in bases]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    looked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"未找到 {DEFAULT_FILE_NAME}，已查找:\n{looked}\n"
        f"请把 {DEFAULT_FILE_NAME} 放到上述任一目录，或用 --file 指定路径。"
    )


def pcm_cache_path(source: Path, pcm_out: Optional[str]) -> Path:
    if pcm_out:
        target = Path(pcm_out).expanduser()
        if target.is_dir():
            raise RuntimeError(f"--pcm-out 必须是文件路径，不能是目录: {target}")
        return target
    if source.suffix.lower() == ".pcm":
        return source
    return source.with_name(f"{source.stem}_16k_mono.pcm")


def validate_pcm(path: Path) -> int:
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"PCM 文件为空: {path}")
    if size % 2 != 0:
        raise RuntimeError(f"PCM 字节数不是 16 bit 对齐的偶数: {path} ({size} 字节)")
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
    """用 ffmpeg 转成 16 kHz / 16 bit / 单声道裸 PCM。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_RATE),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        str(target),
    ]
    try:
        subprocess.run(command, check=True)
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[警告] ffmpeg 转换失败: {exc}", file=sys.stderr)
        return False


def _convert_with_miniaudio(source: Path, target: Path) -> bool:
    """用纯 Python 包 miniaudio（内含 dr_mp3）解码并重采样。"""
    try:
        import miniaudio  # type: ignore
    except ImportError:
        return False

    try:
        decoded = miniaudio.decode_file(
            str(source),
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
    except Exception as exc:  # miniaudio 抛出的异常类型较杂，统一降级处理
        print(f"[警告] miniaudio 解码失败: {exc}", file=sys.stderr)
        return False


def ensure_pcm(source: Path, target: Path, force: bool = False) -> Path:
    """确保存在可直接播放的 PCM；必要时从 MP3 转换并缓存。"""
    if source.suffix.lower() == ".pcm":
        validate_pcm(source)
        return source

    if target == source:
        raise RuntimeError("PCM 输出路径不能与输入 MP3 相同")

    if target.is_file() and not force:
        try:
            if target.stat().st_mtime >= source.stat().st_mtime:
                validate_pcm(target)
                print(f"使用已转换的 PCM: {target}")
                return target
        except (OSError, RuntimeError):
            pass  # 缓存损坏时重新转换

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
                "  uv pip install miniaudio\n"
                f"  python scripts/voice/play_laugh.py --convert-only --file \"{source}\"\n"
                f"然后把生成的 {target.name} 和本脚本上传到机器人 /userdata，"
                "再用 --file 指定该 PCM 文件。"
            )
        validate_pcm(temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass

    print(f"已生成 PCM: {target}")
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


def play_pcm(
    pcm_path: Path,
    *,
    volume: Optional[float] = None,
    repeat: int = 1,
    stream_id: str = DEFAULT_STREAM_ID,
    startup_delay_s: float = DEFAULT_STARTUP_DELAY_S,
    tail_margin_s: float = DEFAULT_TAIL_MARGIN_S,
) -> None:
    """按 SDK 要求分块播放一个 16 kHz / 16 bit / 单声道 PCM 文件。"""
    size = validate_pcm(pcm_path)
    duration_s = size / BYTES_PER_SECOND

    print("正在导入 galbot_sdk ...", flush=True)
    GalbotRobot = _load_robot_class()
    print("正在创建 GalbotRobot 实例 ...", flush=True)
    robot = GalbotRobot()
    print("正在调用 robot.init() ...", flush=True)
    if not robot.init():
        raise RuntimeError(
            "robot.init() 失败：请检查机器人是否上电、急停是否松开、"
            "网络/embosa 是否正常。"
        )
    print("robot.init() 成功。", flush=True)

    try:
        if startup_delay_s > 0:
            print(f"等待 SDK 初始化 {startup_delay_s:.1f} s ...")
            time.sleep(startup_delay_s)

        if volume is not None:
            if robot.set_volume(volume):
                print(f"系统音量已设为 {volume:g}")
            else:
                print(f"[警告] set_volume({volume:g}) 失败，将按当前音量播放", file=sys.stderr)
        else:
            try:
                current = robot.get_volume()
                print(f"当前系统音量: {current:g}（可用 --volume 0-100 调整）")
                if current < 30.0:
                    print("[提示] 当前音量较低，建议加 --volume 80 再试。", file=sys.stderr)
            except Exception as exc:
                print(f"[警告] 读取音量失败: {exc}", file=sys.stderr)

        for index in range(repeat):
            if not robot.is_running():
                raise RuntimeError("机器人控制已停止（is_running() 为 False）")

            print(
                f"[{index + 1}/{repeat}] 播放 {pcm_path.name}: "
                f"{duration_s:.2f} s, {size} 字节, stream_id={stream_id!r}"
            )
            started = time.monotonic()
            written_chunks = 0
            with open(pcm_path, "rb") as audio:
                while True:
                    chunk = audio.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    if not robot.is_running():
                        raise RuntimeError("播放过程中机器人控制停止")
                    if not robot.write_audio_stream_output(chunk, stream_id):
                        raise RuntimeError(
                            f"write_audio_stream_output 失败（第 {written_chunks + 1} 块）"
                        )
                    if written_chunks == 0:
                        print("首个音频块写入成功。", flush=True)
                    written_chunks += 1
                    time.sleep(WRITE_INTERVAL_S)

            # SDK 没有“播放完成”回调；按音频时长等待，确保缓冲区里的尾音播完。
            # 如果写入本身比音频时长大（SDK 卡顿），至少再等一个尾音余量。
            wait_s = started + duration_s + tail_margin_s - time.monotonic()
            time.sleep(max(wait_s, tail_margin_s))
            print(f"[{index + 1}/{repeat}] 播放完成（写入 {written_chunks} 块）")

        print("全部播放完成。")
    finally:
        try:
            robot.stop_audio_stream_output(stream_id)
        except Exception as exc:
            print(f"[警告] stop_audio_stream_output 失败: {exc}", file=sys.stderr)
        try:
            robot.request_shutdown()
            robot.wait_for_shutdown()
        except Exception as exc:
            print(f"[警告] 关闭机器人控制失败: {exc}", file=sys.stderr)
        finally:
            try:
                robot.destroy()
            except Exception as exc:
                print(f"[警告] destroy() 失败: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 laugh.mp3 转成 16 kHz/16 bit/单声道 PCM 并在 Galbot G1 上播放。"
    )
    parser.add_argument(
        "-f",
        "--file",
        default=None,
        help=f"MP3 或已转换的 PCM 路径；默认自动查找 {DEFAULT_FILE_NAME}",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=None,
        help="系统音量 0-100；不传则保持机器人当前音量",
    )
    parser.add_argument("--repeat", type=int, default=1, help="重复播放次数，默认 1")
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="只把 MP3 转成 PCM，不初始化 SDK、不连接机器人",
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="忽略已有 PCM 缓存，强制重新转换",
    )
    parser.add_argument(
        "--pcm-out",
        default=None,
        help="PCM 输出/缓存路径；默认与 MP3 同目录的 <名字>_16k_mono.pcm",
    )
    parser.add_argument(
        "--stream-id",
        default=DEFAULT_STREAM_ID,
        help=f"音频输出流 ID，默认 {DEFAULT_STREAM_ID!r}",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=DEFAULT_STARTUP_DELAY_S,
        help=f"init() 后等待秒数，默认 {DEFAULT_STARTUP_DELAY_S:g}",
    )
    parser.add_argument(
        "--tail-margin",
        type=float,
        default=DEFAULT_TAIL_MARGIN_S,
        help=f"末块写入后额外等待秒数，默认 {DEFAULT_TAIL_MARGIN_S:g}",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.repeat < 1:
        parser.error("--repeat 必须大于等于 1")
    if args.volume is not None and not 0.0 <= args.volume <= 100.0:
        parser.error("--volume 必须在 0-100 之间")
    if args.startup_delay < 0:
        parser.error("--startup-delay 不能为负数")
    if args.tail_margin < 0:
        parser.error("--tail-margin 不能为负数")

    try:
        source = resolve_source(args.file)
        target = pcm_cache_path(source, args.pcm_out)
        print(f"音源: {source}")
        print(f"PCM : {target}")

        pcm = ensure_pcm(source, target, force=args.force_convert)
        size = validate_pcm(pcm)
        peak = pcm_peak(pcm)
        print(
            f"PCM 大小: {size} 字节（{size / BYTES_PER_SECOND:.2f} s），"
            f"峰值: {peak}/32767（{peak / 32767 * 100:.1f}%）",
            flush=True,
        )
        if peak == 0:
            print("[警告] PCM 整段为静音，请检查 laugh.mp3 本身是否有声音。", file=sys.stderr)

        if args.convert_only:
            print("已生成 PCM；--convert-only 模式不连接机器人。")
            return 0

        play_pcm(
            pcm,
            volume=args.volume,
            repeat=args.repeat,
            stream_id=args.stream_id,
            startup_delay_s=args.startup_delay,
            tail_margin_s=args.tail_margin,
        )
        return 0
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
