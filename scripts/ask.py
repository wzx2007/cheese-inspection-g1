#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「星宝星宝」语音问答（第一阶段：语音进、语音出）。

三种跑法
--------
::

    # 0. 只验 key / 端点，不连机器人（PC 上就能跑）
    python3 ask.py --check
    python3 ask.py --say "你好，我是星宝" --out hello.wav

    # 1. 只识别一个现成 wav（PC 上就能跑）
    python3 ask.py --asr heard.wav

    # 2. 上机：完整语音问答（要机器人麦克风和喇叭）
    python3 ask.py --voice                 # 一直待命，本地模型听到"星宝星宝"唤醒
    python3 ask.py --voice --no-wake       # 不用唤醒词，听到就当问题
    python3 ask.py --voice --rounds 3      # 答完 3 轮就退出
    python3 ask.py --voice --wake-th 0.3   # 现场漏唤就调低阈值
    python3 ask.py --voice --wake-engine text   # 退回老版"识别文本找星宝"

    # 3. 纯文字问答（不占麦克风，验证"回答"这一段）
    python3 ask.py --chat "这个奶酪新鲜吗"

    # 4. 只测唤醒模型（不连机器人，打印峰值分数）
    python3 ask.py --wake-test a.wav b.wav

Key 放在 ``voice/.env``::

    DASHSCOPE_API_KEY=sk-REPLACE_ME

或者环境变量 ``DASHSCOPE_API_KEY``。**不要把 key 写进代码或提交到 git。**

安全：只用音频接口（``start_microphone_stream_input`` / ``write_audio_stream_output``），
不使能关节、不控底盘、不动腿，和 ``play_clip.py`` 一样是"只出声"的脚本。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from voice import cloud   # noqa: E402


def _load_robot(app_name="xingbao_ask"):
    """只初始化 SDK（不碰运动），给麦克风和喇叭用。"""
    try:
        from galbot_sdk.g1 import GalbotRobot  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "未找到 galbot_sdk.g1。这个模式必须在机器人 Orin 上跑；\n"
            "  只想验云端的话用 --check / --say / --asr。\n"
            "  export PYTHONPATH=/data/galbot/lib:$PYTHONPATH\n"
            "  export LD_LIBRARY_PATH=/data/galbot/lib:$LD_LIBRARY_PATH"
        ) from exc

    robot = GalbotRobot()
    if not robot.init():
        raise RuntimeError("robot.init() 失败：检查上电、急停、网络")
    print("  等待 SDK 初始化 5 s ...")
    time.sleep(5.0)
    return robot


def _qa_context():
    """给 Brain 的上下文：这个脚本没有手臂，不会检测。

    没有这段话，模型听到"测一下"会答"来啦，正反面都扫一遍，稍等——嗯，
    干干净净，没霉"——**完全是编的**，现场观众会以为机器人已经检测完了。
    """
    return ("你现在跑在问答测试脚本里，**没有手臂，不能执行任何检测动作**。"
            "用户让你检测奶酪时，直接说明：要检测请用 xingbao 主控启动，"
            "我这边只能聊天。绝对不要假装已经看过、扫过、测过奶酪，"
            "不要给出任何检测结论。")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="星宝语音问答")
    parser.add_argument("--check", action="store_true", help="只看配置，不发请求")
    parser.add_argument("--probe", action="store_true",
                        help="真的试一遍 TTS 候选链，看这把 key 能用哪个模型")
    parser.add_argument("--say", default=None, help="合成这句话存成 wav（不连机器人）")
    parser.add_argument("--out", default=None, help="--say 的输出路径")
    parser.add_argument("--asr", default=None, help="识别这个 wav（不连机器人）")
    parser.add_argument("--chat", default=None, help="纯文字问一句（不连机器人）")
    parser.add_argument("--voice", action="store_true", help="上机：完整语音问答")
    parser.add_argument("--no-wake", action="store_true", help="不用唤醒词，听到即问")
    parser.add_argument("--rounds", type=int, default=0, help="答完几轮退出，0=一直（默认）")
    parser.add_argument("--volume", type=float, default=50.0,
                        help="喇叭音量 0-100，默认 50")
    parser.add_argument("--vad-mult", type=float, default=None,
                        help="VAD 阈值倍数：阈值=底噪×这个值，默认 1.5。太迟钝就调小，"
                             "老被噪声打断就调大")
    parser.add_argument("--vad-min", type=float, default=None,
                        help="VAD 阈值下限，默认 250")
    parser.add_argument("--wake-engine", choices=("kws", "model", "auto", "text"),
                        default="kws",
                        help="唤醒方式：kws=sherpa-onnx 关键词模型（默认，推荐）；"
                             "model=旧的 openWakeWord 三段式；auto=先 kws 再 model；"
                             "text=不用模型，靠识别文本找'星宝'")
    parser.add_argument("--wake-th", type=float, default=None,
                        help="唤醒命中阈值，默认取各后端自己的默认值（kws 0.25 / model 0.15）")
    parser.add_argument("--wake-test", nargs="+", default=None, metavar="WAV",
                        help="只测唤醒模型：给几段 wav，打印峰值分数（不连机器人）。"
                             "更推荐用 wake_selftest.py，它直接给命中结论")
    parser.add_argument("--wake-watch", type=float, default=None, metavar="SECONDS",
                        help="上机只监听不回答：实时打印唤醒情况，用来现场调阈值。"
                             "对着机器人喊'星宝星宝'，看能不能命中")
    parser.add_argument("--api-key", default=None, help="临时指定 key（默认读配置）")
    args = parser.parse_args(argv)

    cfg = cloud.CloudConfig.from_file()
    if args.api_key:
        cfg.api_key = args.api_key.strip()

    # ---- 只测唤醒模型（不需要机器人，也不需要 key）----
    if args.wake_test:
        from voice import kws as kws_mod
        from voice.play_clip import ensure_pcm, pcm_cache_path

        if args.wake_engine in ("kws", "auto"):
            detector, used = kws_mod.build_detector(
                engine=args.wake_engine, threshold=args.wake_th, verbose=True)
        else:
            detector, used = None, ""
        if detector is None:
            from voice import wake as wake_mod
            detector = wake_mod.LocalWakeDetector(threshold=args.wake_th)
            used = "model"
            if not detector.prime():
                print("唤醒模型不可用：{}".format(detector.error), file=sys.stderr)
                return 2
        print("  后端：{}".format(used))
        for path in args.wake_test:
            name = os.path.basename(path)
            try:
                if used == "kws":
                    # KWS 自己会处理采样率和声道，不用先转 pcm
                    score = detector.detect_wav(path)
                    print("  {:<36} {}  {}".format(
                        name, "命中" if score > 0 else "未命中",
                        "「{}」".format(detector.last_text) if detector.last_text else ""))
                    continue
                # 旧后端：原本用 mic.wav_to_pcm，那个函数**不重采样**，
                # 只把任意采样率的数据当 16 kHz 读。24 kHz 的 TTS 素材会被拉成
                # 1.5 倍慢放，分数直接虚高（0.94 vs 真实 0.03）。必须走
                # play_clip.ensure_pcm，和上机播放用的是同一条转换。
                pcm_path = Path(path)
                if pcm_path.suffix.lower() != ".pcm":
                    pcm_path = ensure_pcm(pcm_path, pcm_cache_path(pcm_path))
                peak = detector.detect_pcm(Path(pcm_path).read_bytes())
            except Exception as exc:                # noqa: BLE001
                print("  {:<36} 读取失败：{}".format(name, exc))
                continue
            print("  {:<36} 峰值 {:.3f}  {}".format(
                name, peak, "命中" if peak > detector.threshold else "未命中"))
        return 0

    # ---- 不需要机器人的三个模式 ----
    if args.check or args.say or args.asr or args.chat or args.probe:
        voice = cloud.CloudVoice(cfg)
        print("key      : {}".format("已配置（{}…）".format(cfg.api_key[:6])
                                     if cfg.api_key else "缺失（放 voice/.env）"))
        print("tts      : {} / {} @ {} Hz（不可用会自动退到候选链）".format(
            cfg.tts_model, cfg.tts_voice, cfg.sample_rate))
        print("asr      : {}".format(cfg.asr_model))
        print("chat     : {}".format(cfg.chat_model))
        if args.probe and cfg.ready():
            print("候选链   :")
            for model, name in voice._tts_targets():
                try:
                    cloud._post_json(cloud.TTS_URL, {
                        "model": model,
                        "input": {"text": "在", "voice": name, "format": "wav",
                                  "sample_rate": cfg.sample_rate},
                    }, cfg.api_key, cfg.timeout_s, 0)
                    print("  {:<24} {:<18} 可用".format(model, name))
                except Exception as exc:            # noqa: BLE001
                    print("  {:<24} {:<18} 不可用（{}）".format(
                        model, name, str(exc)[:60]))
        if args.check:
            return 0 if cfg.ready() else 1
        try:
            if args.say:
                out = Path(args.out) if args.out else (HERE / "voice" / "cache" / "tts.wav")
                path = voice.tts(args.say, out)
                print("wav      : {}（{} 字节）".format(path, path.stat().st_size))
            if args.asr:
                print("识别结果 : {}".format(voice.asr(args.asr) or "（空）"))
            if args.chat:
                # 走 Brain（带 system prompt + 短回答清洗），和上机时念出来的一致。
                from voice.assistant import Brain

                answer = Brain(voice=voice).answer(args.chat)
                print("回答     : {}".format(answer))
            return 0
        except Exception as exc:                    # noqa: BLE001
            print("失败：{}".format(exc), file=sys.stderr)
            return 2

    # ---- 上机：只监听唤醒词，打印命中/分数（不联网、不播音）----
    if args.wake_watch is not None:
        from voice import kws as kws_mod
        from voice.mic import MicRecorder, pcm_rms

        robot = None
        try:
            detector, used = kws_mod.build_detector(
                engine=args.wake_engine, threshold=args.wake_th, verbose=True)
            if detector is None:
                print("唤醒模型不可用。kws 后端需要 "
                      "python3 -m pip install --user sherpa-onnx", file=sys.stderr)
                return 2
            detector.reset()
            robot = _load_robot("xingbao_wake_watch")
            mic = MicRecorder(robot, verbose=True)
            mic.start()
            mic.drain()
            print("\n对着机器人喊「星宝星宝」，Ctrl-C 结束。"
                  "（后端 {}，阈值 {:.2f}，每 0.5 s 刷新）\n".format(
                      used, detector.threshold))
            t0 = time.monotonic()
            deadline = t0 + args.wake_watch if args.wake_watch > 0 else None
            window_peak = 0.0
            window_level = 0.0
            window_start = t0
            hits = 0
            while deadline is None or time.monotonic() < deadline:
                chunk = mic.read(timeout=0.5)
                if chunk is None:
                    continue
                level = pcm_rms(chunk)
                window_level = max(window_level, level)
                if detector.feed(chunk):
                    hits += 1
                    # KWS 不给逐帧概率，命中就是命中，分数记为 1.00
                    print("  {:>6.1f}s  ★ 命中  {}  分数 {:.2f}".format(
                        time.monotonic() - t0,
                        "「{}」".format(getattr(detector, "last_text", "") or ""),
                        detector.last_score))
                    detector.reset()
                    window_peak = 0.0
                    window_level = 0.0
                    window_start = time.monotonic()
                    continue
                window_peak = max(window_peak, detector.last_score)
                now = time.monotonic()
                if now - window_start >= 0.5:
                    print("  {:>6.1f}s  电平 {:>5.0f}  {}".format(
                        now - t0, window_level,
                        "#" * int(min(window_peak, 1.0) * 40)))
                    window_peak = 0.0
                    window_level = 0.0
                    window_start = now
            print("\n共命中 {} 次".format(hits))
            mic.stop()
            return 0
        except KeyboardInterrupt:
            print("\n已中断。", file=sys.stderr)
            return 130
        finally:
            if robot is not None:
                for name in ("request_shutdown", "wait_for_shutdown", "destroy"):
                    try:
                        getattr(robot, name)()
                    except Exception:               # noqa: BLE001
                        pass

    # ---- 上机：完整语音问答 ----
    if not args.voice:
        parser.print_help()
        return 2

    if not cfg.ready():
        print("没有 API key。先写 voice/.env：DASHSCOPE_API_KEY=<你的 DashScope Key>",
              file=sys.stderr)
        return 2

    from voice.assistant import Assistant, WAKE_WORD

    # 语音只有一个：和 xingbao.py 抢同一把锁，并在启动前清掉别的语音进程。
    # 2026-11-08 现场事故：5 个 ask.py 残留进程 + 星宝本体同时在跑，
    # 用户对着 ask.py 说"测一下"，它回"来啦，正反面都扫一遍"——**然后机器人没动**。
    from xingbao import _check_voice_env, _take_lock, _setup_log

    _setup_log("/tmp/ask_voice.log")
    if not _check_voice_env():
        return 4
    if _take_lock() is None:
        print("已经有一个语音进程在跑了。先关掉它。", file=sys.stderr)
        return 4
    print("注意：这是**问答测试脚本**，只会说话，不会动机器人。"
          "\n      要它真的去检测奶酪，请用 xingbao.py --confirm-motion\n")

    robot = None
    assistant = None
    try:
        robot = _load_robot()
        # 告诉模型它没有手：避免用户说"测一下"时它答"我去看看"，
        # 让人以为任务已经启动了（现场就是这么被误导的）。
        assistant = Assistant(robot, cloud_voice=cloud.CloudVoice(cfg),
                              require_wake=not args.no_wake,
                              vad_floor_min=args.vad_min,
                              vad_floor_mult=args.vad_mult,
                              wake_engine=args.wake_engine,
                              wake_threshold=args.wake_th,
                              context=_qa_context)
        assistant.voice.set_volume(args.volume)

        if args.rounds <= 0:
            # 默认就是「一直对话」：不退出，喊「星宝星宝」就问，问完回待命。
            assistant.run_forever(
                greet="问答模式，叫我{}就行。我只聊天，不会动。".format(WAKE_WORD))
            return 0

        done = 0
        while done < args.rounds:
            answer = assistant.ask_once()
            if answer is None:
                break
            done += 1
        return 0
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as exc:                        # noqa: BLE001
        print("失败：{}".format(exc), file=sys.stderr)
        return 1
    finally:
        if assistant is not None:
            try:
                assistant.close()
            except Exception:                       # noqa: BLE001
                pass
        if robot is not None:
            for name in ("request_shutdown", "wait_for_shutdown", "destroy"):
                try:
                    getattr(robot, name)()
                except Exception:                   # noqa: BLE001
                    pass


if __name__ == "__main__":
    sys.exit(main())
