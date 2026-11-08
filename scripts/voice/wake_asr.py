# -*- coding: utf-8 -*-
"""ASR 唤醒：把原仓库 ``wzx2007/xingbao`` 的 ``asr_wake_word.py`` 适配到本机。

为什么需要它
------------
本地 KWS（sherpa-onnx zipformer）在**回放合成音**上能拿到 1.00 分，
但现场真人喊的 441 段语音只命中 10 段（2.3%，见 ``_tmp/wake_batch.py`` 实测）。
openWakeWord（原仓库 ``wake_word.py``）在这块板子上更糟：TFLite 运行时版本
不匹配，分数恒为 0.001，等于彻底失效。

原仓库为"板载 USB 麦克风、本地 KWS 听不见清楚人声"这种情况专门写了
``asr_wake_word.py``：**本地 VAD 先断句，只把"有人说话"的那一小段送 ASR，
拿到文本再用同音字白名单判定是否在叫名字**。ASR 的鲁棒性远高于小模型 KWS，
代价是每段人声一次网络请求。本文件把这套机制一一对应搬过来：

======================================  ======================================
原仓库                                   本文件
======================================  ======================================
``capture_utterance_vad(WebRTC VAD)``     ``mic.MicRecorder.record_utterance()``
``SherpaOnnxASRClient.transcribe()``      ``cloud.CloudVoice.asr()``
``_wake_match_detail()`` 同音字白名单      ``wake_match_detail()``（同一张表）
``_transcribe_with_deadline()`` 超时退避   ``AsrWakeDetector.transcribe()``
``_detect_local_keyword()`` 本地复查       ``AsrWakeDetector._kws_score()``
======================================  ======================================

与原仓库的三处**必要适配**（都写在这里，避免以后忘记为什么不一样）：

1. **VAD 换成本机这套**。原仓库用 WebRTC 的语音分类器；本机麦克风只给
   ``robot.start_microphone_stream_input`` 的裸 PCM，没有 webrtcvad 依赖，
   所以复用 ``mic.record_utterance`` 的自适应 RMS 断句 —— 和对话录音**同一条
   VAD**，人在对话里能被录到，唤醒就一定也能被录到（原仓库也是这个思路：
   唤醒捕获的阈值比对话捕获更宽松，绝不让唤醒比对话更难）。
2. **先 ASR 再本地 KWS**。原仓库是 KWS 先跑、命中就省一次 ASR；本机 KWS 召回
   只有 2.3%，省不了钱还常常把"唤醒词 + 后面那句话"一起吞掉。所以先走 ASR，
   拿到整句文本后 ``split_wake`` 能直接把"星宝星宝"后面的问题剥出来；ASR
   失败/空识别时才用本地 KWS 兜底（没拿到 rest，由上层再听一句）。
3. **唤醒词必须在句首附近**（``WAKE_HEAD_CHARS``）。原仓库允许同音字出现在
   句子任意位置；展厅里观众聊天内容就和"星宝""奶酪"高度相关（"这个星宝挺
   好玩"），任意位置匹配会不停自己醒来。原仓库注释里也承认单通道麦克风
   分不清谁在说话，所以这里收紧到"句首 6 个字以内"。

调参（全部走环境变量，和原仓库同名，现场不用改代码）：

===============================  ======  ==============================
变量                              默认    含义
===============================  ======  ==============================
``XINGBAO_ASR_WAKE_END_SILENCE_MS``    450   说完后静音多久算一句结束
``XINGBAO_ASR_WAKE_MIN_UTTERANCE_MS``  420   短于这个时长不算一句话
``XINGBAO_ASR_WAKE_MAX_UTTERANCE_S``   4.5   单句最长
``XINGBAO_ASR_WAKE_LISTEN_TIMEOUT_S``  1.2   这么久没开口就返回空、重新等
``XINGBAO_ASR_WAKE_MIN_RMS``           300   整段峰值低于这个当噪声，不烧 ASR
``XINGBAO_ASR_WAKE_RECOGNITION_TIMEOUT_S`` 3.0  单次识别超时
``XINGBAO_ASR_WAKE_FAILURE_BACKOFF_S``  1.2   识别失败后歇多久再试
``XINGBAO_ASR_WAKE_MIN_INTERVAL_S``     0.6   两次识别之间的最小间隔
``XINGBAO_ASR_WAKE_LOCAL_KWS``           1    是否用本地 KWS 兜底
===============================  ======  ==============================
"""

from __future__ import annotations

import os
import re
import threading
import time
import unicodedata
from typing import Optional, Tuple

from . import mic


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    """读一个现场可调的数值，坏值不能让启动失败。"""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# 唤醒词同音字表（原仓库 asr_wake_word.py 原样搬运 + 现场实测补充）
# ---------------------------------------------------------------------------
# 顺序有意义：长词在前，``next(...)`` 才不会把"星宝星宝"截成"星宝"。
WAKE_ALIASES = (
    "星宝星宝",
    "星宝",
    "新宝新宝",
    "新宝",
    "心宝",
    "馨宝",
    "鑫宝",
    # ---- 现场实测补充（2026-11-08 06:46 日志）----
    # DashScope 把"星宝星宝"识别成"星，星爆，星爆。"，不收"星爆"的话
    # 文本唤醒路径下真人喊十遍也唤不醒。
    "星爆",
    "星抱",
    "星保",
    "兴宝",
    "欣宝",
    "小宝",
)
# 同音字兜底：声母 {星新心馨鑫欣兴} + 韵母 {宝保包堡报爆抱} 任意组合都算。
WAKE_INITIALS = "星新心馨鑫欣兴"
WAKE_FINALS = "宝保包堡报爆抱"
# 拼音/英文转写（有些 ASR 会把中文名字转成拼音）。
WAKE_PINYIN_ALIASES = ("xingbao", "xinbao")
# 唤醒词允许前面夹几个语气词（"嗯""喂""哎"），默认 1 个字。
# 见文件头"必要适配 3"：展厅里"这个星宝挺好玩"（关键词在第 3 位）不能算唤醒。
# 现场如果发现"说了但前面带了半句寒暄"，用 XINGBAO_ASR_WAKE_HEAD_CHARS 放宽。
WAKE_HEAD_CHARS = int(max(0.0, _env_float("XINGBAO_ASR_WAKE_HEAD_CHARS", 1.0)))


def current_settings() -> dict:
    """当前生效的唤醒参数（启动横幅 / 日志用，避免横幅写着默认值、实际跑别的）。"""
    return {
        "end_silence_ms": _env_float("XINGBAO_ASR_WAKE_END_SILENCE_MS", 450.0),
        "min_utterance_ms": _env_float("XINGBAO_ASR_WAKE_MIN_UTTERANCE_MS", 420.0),
        "min_rms": _env_float("XINGBAO_ASR_WAKE_MIN_RMS", 300.0),
        "head_chars": WAKE_HEAD_CHARS,
        "local_kws": _env_bool("XINGBAO_ASR_WAKE_LOCAL_KWS", True),
    }


def compact(text: str) -> str:
    """去掉标点空格、全角转半角、转小写 —— 只留可比较的字符。"""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return "".join(char.lower() for char in normalized if char.isalnum())


def wake_match_detail(text: str) -> Tuple[Optional[str], str]:
    """``(命中的唤醒词, 命中方式)``；没命中返回 ``(None, "none")``。

    命中方式只有三种，日志里只用这三类，不落识别原文（原仓库同样考虑：
    识别文本可能有别人的闲聊，不该进服务日志）。
    """
    plain = compact(text)
    if not plain:
        return None, "none"
    for alias in WAKE_ALIASES:
        if alias in plain:
            return alias, "exact_alias"
    for alias in WAKE_PINYIN_ALIASES:
        if alias in plain:
            return alias, "pinyin_alias"
    phonetic = re.search(r"[{}][{}]".format(WAKE_INITIALS, WAKE_FINALS), plain)
    if phonetic:
        return phonetic.group(0), "phonetic_name"
    return None, "none"


def wake_match(text: str) -> bool:
    return wake_match_detail(text)[0] is not None


def strip_wake(text: str) -> str:
    """把句首的唤醒词剥掉，返回剩下的问题（可能为空串）。"""
    return split_wake(text)[1]


def split_wake(text: str) -> Tuple[bool, str, Optional[str], str]:
    """``(是否唤醒, 唤醒词之后的内容, 命中的唤醒词, 命中方式)``。

    "星宝星宝，帮我查一下天气" -> ``(True, "帮我查一下天气", "星宝星宝", "exact_alias")``
    "这个星宝挺好玩"           -> ``(False, "", "星宝", "not_head")``
    """
    plain = compact(text)
    keyword, reason = wake_match_detail(plain)
    if keyword is None:
        return False, "", None, "none"
    index = plain.find(keyword)
    if index > WAKE_HEAD_CHARS:
        return False, "", keyword, "not_head"
    rest = plain[index + len(keyword):]
    # "星宝星宝" 剥掉一个还剩一个，继续剥，别把重复的唤醒词当问题问出去
    while rest:
        again, _why = wake_match_detail(rest)
        if again is None or rest.find(again) != 0:
            break
        rest = rest[len(again):]
    return True, rest, keyword, reason


class AsrWakeDetector(object):
    """VAD 断句 -> 云端 ASR -> 同音字白名单，判定是否在喊唤醒词。"""

    # 识别结果这么短就认为是环境人声（"嗯""啊""好的"），不是有人跟机器人说话
    _SHORT_TEXT_CHARS = 3
    _SHORT_SKIP_AFTER = 2        # 连续 2 次超短文本 -> 间隔 x0.6
    _QUIET_SKIP_AFTER = 4        # 连续 4 次 -> 拉到 _NOISY_INTERVAL_S
    # "比周围响"的判定：最近几段峰值的中位数当环境声，超过它的倍数就算在喊
    _LEVEL_WINDOW = 12
    _LEVEL_MIN_SAMPLES = 4
    _LOUD_FACTOR = 1.8

    def __init__(self, recorder, cloud, kws=None, verbose: bool = True,
                 cache_dir=None, recognition_timeout_s: Optional[float] = None,
                 min_interval_s: Optional[float] = None):
        self._mic = recorder
        self._cloud = cloud
        self._kws = kws
        self._verbose = bool(verbose)
        self._cache_dir = cache_dir
        self._counter = 0

        self.end_silence_s = _env_float(
            "XINGBAO_ASR_WAKE_END_SILENCE_MS", 450.0, minimum=80.0) / 1000.0
        self.min_utterance_s = _env_float(
            "XINGBAO_ASR_WAKE_MIN_UTTERANCE_MS", 420.0, minimum=120.0) / 1000.0
        self.max_utterance_s = _env_float(
            "XINGBAO_ASR_WAKE_MAX_UTTERANCE_S", 4.5, minimum=0.8)
        self.listen_timeout_s = _env_float(
            "XINGBAO_ASR_WAKE_LISTEN_TIMEOUT_S", 1.2, minimum=0.3)
        self.min_rms = _env_float("XINGBAO_ASR_WAKE_MIN_RMS", 300.0, minimum=1.0)
        self.min_interval_s = _env_float(
            "XINGBAO_ASR_WAKE_MIN_INTERVAL_S",
            0.6 if min_interval_s is None else float(min_interval_s),
            minimum=0.0)
        self.local_kws = _env_bool("XINGBAO_ASR_WAKE_LOCAL_KWS", True)
        self._recognition_timeout_s = max(
            0.5, float(recognition_timeout_s if recognition_timeout_s is not None
                       else _env_float("XINGBAO_ASR_WAKE_RECOGNITION_TIMEOUT_S",
                                       3.0, minimum=0.5)))
        self._backoff_s = _env_float(
            "XINGBAO_ASR_WAKE_FAILURE_BACKOFF_S", 1.2, minimum=0.0)

        self._retry_after = 0.0
        self._last_asr = 0.0
        self._lock = threading.Lock()
        self._inflight = False
        # 背景噪音节流（见 check() 末尾那段注释）
        self._short_miss_streak = 0
        self._levels = []
        self._NOISY_INTERVAL_S = _env_float(
            "XINGBAO_ASR_WAKE_NOISY_INTERVAL_S", 2.5, minimum=0.0)
        # 统计，收尾时打一行，方便现场判断是"没听见"还是"识别不对"
        self.stats = {"captures": 0, "asr": 0, "wakes": 0, "errors": 0}

    # ------------------------------------------------------------------
    # 采集
    # ------------------------------------------------------------------
    def capture(self) -> bytes:
        """等一段人声。没听到返回 ``b""``（超时也算）。

        VAD 参数故意和对话录音同一套（``record_utterance``），只把
        "说完后的静音" 缩短到 450 ms —— 原仓库注释：小孩说"星宝星宝"
        常常不到 1 秒，等太久会让一次短呼看起来像卡住。
        """
        pcm = self._mic.record_utterance(
            max_s=self.max_utterance_s,
            silence_s=self.end_silence_s,
            min_s=self.min_utterance_s,
            calib_num=3,                       # 240 ms 估本底（原仓库 0.18 s）
            start_timeout_s=self.listen_timeout_s,
        )
        if pcm:
            self.stats["captures"] += 1
        return pcm

    # ------------------------------------------------------------------
    # 判定
    # ------------------------------------------------------------------
    def check(self, pcm: bytes) -> Tuple[bool, str, str, str]:
        """一段人声 -> ``(是否唤醒, 唤醒词之后的内容, 原因, 识别文本)``。"""
        if not pcm:
            return False, "", "empty_audio", ""
        duration = mic.pcm_duration(pcm)
        peak = _peak(pcm)
        if peak < self.min_rms:
            return False, "", "too_quiet(peak={:.0f})".format(peak), ""

        now = time.monotonic()
        loud = self._is_loud(pcm, peak)
        if now < self._retry_after:
            skip = "backoff"
        elif now - self._last_asr < self._effective_interval() and not loud:
            # 节流只挡"平平的环境人声"。**明显比周围更响的一声照样送识别** ——
            # 现场如果有人正对着机器人喊"星宝星宝"，音量一定盖过背景聊天，
            # 不能因为还在冷却窗口里就把呼叫吞掉（那就是"又没反应了"）。
            skip = "throttled"
        else:
            skip = ""
        self._remember_level(peak)

        text = ""
        error = None
        if not skip:
            text, error = self.transcribe(pcm)
            woke, rest, keyword, reason = split_wake(text)
            if woke:
                self.stats["wakes"] += 1
                self._short_miss_streak = 0        # 命中了：节流复位
                return True, rest, "asr:{}:{}".format(reason, keyword), text

        # ---- ASR 没戏了才用本地 KWS 复查（原仓库 _detect_local_keyword）----
        score = self._kws_score(pcm)
        if score is not None:
            threshold = float(getattr(self._kws, "threshold", 0.25) or 0.25)
            if score >= threshold:
                self.stats["wakes"] += 1
                self._short_miss_streak = 0
                return True, "", "local_kws:{:.2f}".format(score), text
        if skip:
            return False, "", skip, text
        if error:
            return False, "", "asr_error:{}".format(error), text
        if not text:
            return False, "", "asr_empty", ""

        # 现场背景人声（"嗯""啊""这个"）会被 VAD 切成一句句送进来，每次都烧一次
        # 网络识别。连续识别出超短文本时说明"周围在聊天、没人叫我"，逐步降频；
        # 一旦出现正常长度的句子（说明有人在跟机器人说话）立刻恢复。
        if len(compact(text)) <= self._SHORT_TEXT_CHARS:
            self._short_miss_streak = min(self._short_miss_streak + 1, 99)
        else:
            self._short_miss_streak = 0
        return False, "", "no_match({} chars, {:.0f}ms)".format(
            len(text), duration * 1000.0), text

    def _effective_interval(self) -> float:
        """当前两次识别之间的最小间隔（随背景噪音自动拉长）。"""
        streak = self._short_miss_streak
        if streak >= self._QUIET_SKIP_AFTER:
            return max(self.min_interval_s, self._NOISY_INTERVAL_S)
        if streak >= self._SHORT_SKIP_AFTER:
            return max(self.min_interval_s, self._NOISY_INTERVAL_S * 0.6)
        return self.min_interval_s

    def _remember_level(self, peak: float) -> None:
        """记最近几段的峰值，用来判断"这一声是不是比周围响"。"""
        self._levels.append(float(peak))
        if len(self._levels) > self._LEVEL_WINDOW:
            self._levels.pop(0)

    def _is_loud(self, pcm: bytes, peak: float) -> bool:
        """这一声是不是明显盖过最近的环境声（多半是有人冲着机器人喊）。

        取最近若干段峰值的中位数当"环境声"，超过它的 ``_LOUD_FACTOR`` 倍、
        且绝对峰值不低，就算"在喊"。节流期间遇到这种声音照样送识别。
        """
        if len(self._levels) < self._LEVEL_MIN_SAMPLES:
            return False
        ordered = sorted(self._levels)
        median = ordered[len(ordered) // 2]
        if median <= 0:
            return True
        return (peak >= median * self._LOUD_FACTOR
                and peak >= self.min_rms * self._LOUD_FACTOR)

    def _kws_score(self, pcm: bytes) -> Optional[float]:
        """整段送给本地 KWS 复查。KWS 不可用就返回 None。"""
        if not self.local_kws or self._kws is None:
            return None
        detect = getattr(self._kws, "detect_pcm", None)
        if not callable(detect):
            return None
        try:
            return float(detect(pcm))
        except Exception as exc:                  # noqa: BLE001
            print("  [wake-asr] 本地 KWS 复查失败（忽略）：{}".format(exc))
            self._kws = None
            return None

    # ------------------------------------------------------------------
    # 识别（带超时 + 退避，原仓库 _transcribe_with_deadline）
    # ------------------------------------------------------------------
    def transcribe(self, pcm: bytes) -> Tuple[str, Optional[str]]:
        """把一段 PCM 送 ASR，返回 ``(文本, 错误描述)``。

        第三方 ASR 客户端不保证有网络超时，一次卡死就会让整场对话失聪。
        这里用后台线程 + 截止时间：超时就不再为后续每段声音重新起线程，
        唤醒循环本身继续活着，等那个 worker 自己跑完再恢复。
        """
        with self._lock:
            if self._inflight:
                return "", "inflight"
            self._inflight = True
        wav = self._write_wav(pcm)
        self._last_asr = time.monotonic()
        self.stats["asr"] += 1
        done = threading.Event()
        box = {"text": "", "error": None}

        def worker():
            try:
                if not self._cloud or not self._cloud.ready():
                    box["error"] = "cloud_not_ready"
                else:
                    box["text"] = str(self._cloud.asr(wav) or "").strip()
            except BaseException as exc:          # noqa: BLE001 第三方边界
                box["error"] = "{}: {}".format(type(exc).__name__, exc)
            finally:
                with self._lock:
                    self._inflight = False
                done.set()

        threading.Thread(target=worker, name="asr-wake", daemon=True).start()
        if not done.wait(timeout=self._recognition_timeout_s):
            self._defer()
            return "", "timeout"
        error = box["error"]
        if error:
            self.stats["errors"] += 1
            self._defer()
        return str(box["text"] or ""), error

    def _write_wav(self, pcm: bytes):
        self._counter += 1
        directory = self._cache_dir
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception:                         # noqa: BLE001
            directory = None
        name = "wake_{}_{}.wav".format(int(time.time() * 1000) % 100000,
                                       self._counter % 100)
        path = (directory / name) if directory is not None else name
        return mic.pcm_to_wav(pcm, path)

    def _defer(self) -> None:
        if self._backoff_s > 0:
            self._retry_after = time.monotonic() + self._backoff_s


# ---------------------------------------------------------------------------
def _peak(pcm: bytes) -> float:
    """整段 PCM 的峰值（s16le）。用来挡"噪声段"，不烧 ASR。"""
    import array
    import sys
    if not pcm:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    return float(max((abs(v) for v in samples), default=0))


# ---------------------------------------------------------------------------
def _self_test() -> int:
    """唤醒词匹配自测：``python3 -m voice.wake_asr``"""
    cases = [
        ("星宝星宝", True, ""),
        ("星宝星宝，帮我查一下天气", True, "帮我查一下天气"),
        ("星宝，这个奶酪新鲜吗", True, "这个奶酪新鲜吗"),
        ("星，星爆，星爆。", True, ""),
        ("嗯，星宝星宝，测一下", True, "测一下"),
        ("新宝新宝", True, ""),
        ("心宝", True, ""),
        ("鑫宝你好", True, "你好"),
        ("xing bao", True, ""),
        ("喂，星宝星宝", True, ""),
        ("你好呀星宝", False, ""),
        ("这个星宝挺好玩，奶酪也不错", False, ""),
        ("今天天气怎么样", False, ""),
        ("", False, ""),
    ]
    bad = 0
    for text, want, want_rest in cases:
        woke, rest, keyword, reason = split_wake(text)
        ok = (woke == want) and (rest == want_rest)
        if not ok:
            bad += 1
        print("  {} {!r:32s} woke={} rest={!r} kw={} ({})".format(
            "OK " if ok else "BAD", text, woke, rest, keyword, reason))
    print("  {} / {} 通过".format(len(cases) - bad, len(cases)))

    # ---- 背景噪音节流：连着"嗯"要降频，真人喊一声不能漏 ----
    print("\n  节流策略：")

    class _Cloud(object):
        def __init__(self, texts):
            self.texts = list(texts)

        def ready(self):
            return True

        def asr(self, wav):
            return self.texts.pop(0) if self.texts else ""

    def _pcm(amp=2000, seconds=1.2):
        import array as _array
        s = _array.array("h", [amp if i % 2 == 0 else -amp
                               for i in range(int(16000 * seconds))])
        return s.tobytes()

    det = AsrWakeDetector(None, _Cloud(["嗯。"] * 6), kws=None,
                          verbose=False, min_interval_s=0.0)
    for _ in range(4):
        det.check(_pcm())
        det._last_asr = 0.0
    quiet_ok = abs(det._effective_interval() - det._NOISY_INTERVAL_S) < 1e-6
    if not quiet_ok:
        bad += 1
    print("  {} 连续短噪声 -> 间隔拉到 {:.1f}s".format(
        "OK " if quiet_ok else "BAD", det._effective_interval()))

    loud_ok = False
    for _ in range(5):
        det._remember_level(1200.0)
    loud_ok = det._is_loud(_pcm(amp=4000), 4000.0) and \
        not det._is_loud(_pcm(amp=1300), 1300.0)
    if not loud_ok:
        bad += 1
    print("  {} 比环境响得多才算\"在喊\"（破例送识别）".format(
        "OK " if loud_ok else "BAD"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
