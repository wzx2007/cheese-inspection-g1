# -*- coding: utf-8 -*-
"""云端语音：DashScope（阿里云百炼）的 TTS / ASR / 对话。

接口照搬 ``github.com/wzx2007/Xingbao`` 的实现（``multimodal/tts_client.py``、
``multimodal/asr_client.py``），只去掉它对项目内部模块的依赖，改成纯标准库，
因为机器人那边是 Python 3.8，装不了多少东西。

三个能力
--------
``tts(text)``
    ``POST /api/v1/services/audio/tts/SpeechSynthesizer``
    body ``{model, input:{text, voice, format:"wav", sample_rate, volume, rate}}``，
    返回里给的是**音频 URL**，要再下载一次。
``asr(wav)``
    ``POST /compatible-mode/v1/chat/completions``，用多模态消息塞
    ``{"type":"input_audio","input_audio":{"data":"data:audio/wav;base64,..."}}``，
    结果在 ``choices[0].message.content``。
``chat(messages)``
    同一个 compatible-mode 端点，纯文本，用来出问答的回复。

**音频格式**：DashScope 返回的 wav 通常是 24 kHz，而机器人 SDK 只吃
16000 Hz / 16 bit / 单声道 PCM。转换交给 ``play_clip.ensure_pcm``（ffmpeg 优先，
退 miniaudio），不要在这里另写一套重采样。

Key 从哪来（按优先级）
---------------------
1. 构造参数 ``api_key``
2. 环境变量 ``DASHSCOPE_API_KEY``
3. ``voice/xingbao.json`` 里的 ``api_key``（**别提交到 git**）
4. ``voice/.env`` 里的 ``DASHSCOPE_API_KEY=...``

都没有就抛 ``CloudVoiceNotConfigured``，调用方降级成本地兜底，不要崩掉整场演示。
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

TTS_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
COMPAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# 默认模型/音色。现场要换只改这里或 voice/xingbao.json。
# 2026-11-08 实测：同一把 key 上 ``qwen-tts`` 报 404 ``Model not exist``，
# 而 ``cosyvoice-v3-flash`` / ``cosyvoice-v2`` 正常 —— 说明模型开通情况随账号/区域变。
# 所以 TTS 走**候选链**：按顺序试，第一个成功的那对记下来，后面不再重试废的那对。
# ``ask.py --probe`` 可以一次看清这把 key 到底能用哪个。
DEFAULT_TTS_MODEL = "cosyvoice-v3-flash"
DEFAULT_TTS_VOICE = "longxiaochun_v3"
TTS_CANDIDATES = (
    ("cosyvoice-v3-flash", "longxiaochun_v3"),   # 低延迟，适合对话（本账号可用）
    ("cosyvoice-v2", "longxiaochun_v2"),         # 最稳，几乎都能用
    ("qwen-tts", "Cherry"),                      # 换了 key 若开通了会优先被选中
)
DEFAULT_ASR_MODEL = "qwen3-asr-flash"
DEFAULT_CHAT_MODEL = "qwen-plus"

# 联网搜索（2026-11-08 实测，别再踩一遍）
# ---------------------------------------
# 1. **只加** ``enable_search: true`` 是**没用的** —— 模型照样回"我不具备联网能力，
#    知识截止到 2024"。实测成都天气那次就是这么被拒的。
# 2. 必须再给 ``search_options``，而且 ``forced_search`` 才是真正的开关：
#      * ``forced_search=false`` -> 让模型自己判断该不该搜（天气那题 4.4 s 就搜了）
#      * ``forced_search=true``  -> 每题都搜（5~6.7 s，奶酪常识题也白烧 2500 token）
#    所以默认用 ``false``，只有"它明明不知道还在那儿推测"时才由上层强制搜。
# 3. 返回里**没有** ``search_info`` / ``annotations``（compatible-mode 不下发来源），
#    所以别指望播报"据某某网站"。要来源得走原生 ``/api/v1/services/aigc/text-generation``。
SEARCH_OPTIONS = {"forced_search": False, "search_strategy": "standard"}

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_RETRIES = 2

_HERE = Path(__file__).resolve().parent
CONFIG_JSON = _HERE / "xingbao.json"
ENV_FILE = _HERE / ".env"


class CloudVoiceNotConfigured(RuntimeError):
    """没拿到 API key / 端点，云端语音不可用。"""


class CloudVoiceError(RuntimeError):
    """请求失败（网络、鉴权、返回格式）。"""


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
class CloudConfig(object):
    def __init__(
        self,
        api_key: Optional[str] = None,
        tts_model: str = DEFAULT_TTS_MODEL,
        tts_voice: str = DEFAULT_TTS_VOICE,
        sample_rate: int = 24000,
        volume: int = 60,
        rate: float = 1.0,
        asr_model: str = DEFAULT_ASR_MODEL,
        chat_model: str = DEFAULT_CHAT_MODEL,
        search_model: str = "",
        search_strategy: str = "standard",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        self.api_key = api_key or load_api_key()
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.sample_rate = int(sample_rate)
        self.volume = int(volume)
        self.rate = float(rate)
        self.asr_model = asr_model
        self.chat_model = chat_model
        # 空 -> 用 chat_model。联网搜索只有部分模型开，换模型时可以单独指。
        self.search_model = search_model or chat_model
        self.search_strategy = search_strategy or "standard"
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "CloudConfig":
        """从 ``voice/xingbao.json`` 读覆盖项；文件不存在就用默认值。"""
        path = Path(path) if path else CONFIG_JSON
        data = {}
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle) or {}
            except Exception as exc:            # noqa: BLE001
                print("  ⚠ {} 读不了（{}），用默认配置".format(path, exc))
        known = set(cls.__init__.__code__.co_varnames) - {"self"}
        kwargs = {k: v for k, v in data.items() if k in known and k != "api_key"}
        return cls(api_key=data.get("api_key"), **kwargs)

    def ready(self) -> bool:
        return bool(self.api_key)


def load_api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    if CONFIG_JSON.is_file():
        try:
            with CONFIG_JSON.open("r", encoding="utf-8") as handle:
                key = str((json.load(handle) or {}).get("api_key") or "").strip()
            if key:
                return key
        except Exception:                       # noqa: BLE001
            pass
    if ENV_FILE.is_file():
        try:
            for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() in ("DASHSCOPE_API_KEY", "api_key"):
                    return value.strip().strip('"').strip("'")
        except Exception:                       # noqa: BLE001
            pass
    return ""


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def _post_json(url: str, payload: Dict, api_key: str, timeout_s: float,
               retries: int = DEFAULT_RETRIES) -> Dict:
    if not api_key:
        raise CloudVoiceNotConfigured(
            "没有 DASHSCOPE_API_KEY。放到环境变量、voice/.env 或 voice/xingbao.json 里。")
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": "Bearer {}".format(api_key),
        "Content-Type": "application/json",
    }
    last = None
    for attempt in range(max(1, retries) + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read()
            return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:400]
            except Exception:                   # noqa: BLE001
                pass
            last = CloudVoiceError("HTTP {} {} {}".format(exc.code, exc.reason, detail))
            if exc.code in (400, 401, 403, 404):     # 参数/鉴权错，重试没用
                raise last
        except Exception as exc:                # noqa: BLE001  网络抖动
            last = CloudVoiceError("{}: {}".format(type(exc).__name__, exc))
        if attempt < retries:
            time.sleep(0.6 * (attempt + 1))
    raise last or CloudVoiceError("请求失败")


def _download(url: str, target: Path, timeout_s: float, retries: int = DEFAULT_RETRIES) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(max(1, retries) + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                data = response.read()
            if not data:
                raise CloudVoiceError("下载到的音频是空的")
            target.write_bytes(data)
            return target
        except Exception as exc:                # noqa: BLE001
            last = exc
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
    raise CloudVoiceError("音频下载失败：{}".format(last))


# --------------------------------------------------------------------------
# 客户端
# --------------------------------------------------------------------------
class CloudVoice(object):
    """TTS / ASR / Chat 三个动作，全部同步阻塞。"""

    def __init__(self, config: Optional[CloudConfig] = None, verbose: bool = True) -> None:
        self.cfg = config or CloudConfig.from_file()
        self._verbose = bool(verbose)
        # 第一次成功后把可用那对记在这儿，避免每次都去撞 404。
        self._tts_ok = None

    def ready(self) -> bool:
        return self.cfg.ready()

    # ---- TTS ----
    def _tts_targets(self) -> List[tuple]:
        """候选 (model, voice) 链。配置里显式指定的排最前面。"""
        prim = (self.cfg.tts_model, self.cfg.tts_voice)
        chain = [prim]
        for pair in TTS_CANDIDATES:
            if pair not in chain:
                chain.append(pair)
        return chain

    def tts(self, text: str, out_wav) -> Path:
        """合成语音到 ``out_wav``（wav，采样率见 cfg.sample_rate）。

        按候选链逐个试：某个模型在这把 key 上没开通（404 Model not exist）时
        自动退到下一个，不用改代码。成功后锁定那一对。
        """
        text = (text or "").strip()
        if not text:
            raise CloudVoiceError("要合成的文本是空的")
        targets = [self._tts_ok] if self._tts_ok else self._tts_targets()
        last = None
        for model, voice in targets:
            payload = {
                "model": model,
                "input": {
                    "text": text,
                    "voice": voice,
                    "format": "wav",
                    "sample_rate": self.cfg.sample_rate,
                    "volume": self.cfg.volume,
                    "rate": self.cfg.rate,
                },
            }
            try:
                data = _post_json(TTS_URL, payload, self.cfg.api_key,
                                  self.cfg.timeout_s, self.cfg.retries)
            except CloudVoiceError as exc:
                last = exc
                if self._verbose:
                    print("  [tts] {}/{} 不可用（{}），换下一个".format(
                        model, voice, str(exc)[:90]))
                continue
            self._tts_ok = (model, voice)
            if self._verbose and (model, voice) != tuple(targets[0]):
                print("  [tts] 改用 {}/{}".format(model, voice))
            return self._finish_tts(data, out_wav)
        raise last or CloudVoiceError("所有 TTS 候选都失败了")

    def _finish_tts(self, data: Dict, out_wav) -> Path:
        try:
            audio_url = str(data["output"]["audio"]["url"])
        except (KeyError, TypeError):
            raise CloudVoiceError("TTS 返回格式不对：{}".format(json.dumps(data, ensure_ascii=False)[:300]))
        if not audio_url:
            raise CloudVoiceError("TTS 没给音频 URL")
        if self._verbose:
            print("  [tts] 收到音频 URL，开始下载")
        return _download(audio_url, Path(out_wav), self.cfg.timeout_s, self.cfg.retries)

    # ---- ASR ----
    def asr(self, wav_path) -> str:
        """识别一个 wav 文件。空识别返回空串（不算异常）。"""
        path = Path(wav_path)
        if not path.is_file():
            raise CloudVoiceError("找不到音频：{}".format(path))
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {
            "model": self.cfg.asr_model,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"data": "data:audio/wav;base64,{}".format(encoded)},
                }],
            }],
            "stream": False,
            "asr_options": {"language": "zh", "enable_itn": True},
        }
        data = _post_json(COMPAT_URL, payload, self.cfg.api_key, self.cfg.timeout_s, self.cfg.retries)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise CloudVoiceError("ASR 返回格式不对：{}".format(json.dumps(data, ensure_ascii=False)[:300]))
        text = text.strip()
        if self._verbose:
            print("  [asr] -> {}".format(text or "（空）"))
        return text

    # ---- Chat ----
    def chat(self, messages: Sequence[Dict], temperature: float = 0.6,
             max_tokens: Optional[int] = None, search: bool = False,
             forced_search: Optional[bool] = None) -> str:
        """对话。``search=True`` 时开联网搜索（见文件头 ``SEARCH_OPTIONS`` 的实测）。

        ``forced_search`` 不传就按配置走（默认让模型自己决定搜不搜）；
        传 True 是"这题必须查网"，用于模型明明不知道还在瞎编的场合。
        """
        model = self.cfg.search_model if search else self.cfg.chat_model
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": float(temperature),
            "stream": False,
        }
        if max_tokens:
            # 播报用的回答要短。光靠 system prompt 说"两句话"模型会忘，
            # 这里再硬限一次 token 数（2026-11-08 实测云端会甩出几百字）。
            payload["max_tokens"] = int(max_tokens)
        if search:
            options = dict(SEARCH_OPTIONS)
            options["search_strategy"] = self.cfg.search_strategy
            if forced_search is not None:
                options["forced_search"] = bool(forced_search)
            payload["enable_search"] = True
            payload["search_options"] = options
        started = time.time()
        data = _post_json(COMPAT_URL, payload, self.cfg.api_key, self.cfg.timeout_s, self.cfg.retries)
        if self._verbose and search:
            print("  [chat] 联网搜索 {}（{:.1f} s，prompt {} token）".format(
                "强制" if (forced_search if forced_search is not None
                           else SEARCH_OPTIONS["forced_search"]) else "自动",
                time.time() - started,
                (data.get("usage") or {}).get("prompt_tokens", "?")))
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError):
            raise CloudVoiceError("对话返回格式不对：{}".format(json.dumps(data, ensure_ascii=False)[:300]))

    def ask(self, question: str, system: Optional[str] = None,
            max_tokens: Optional[int] = None, search: bool = False) -> str:
        messages: List[Dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})
        return self.chat(messages, max_tokens=max_tokens, search=search)


# --------------------------------------------------------------------------
# 命令行自测
# --------------------------------------------------------------------------
def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="DashScope 云端语音自测")
    parser.add_argument("--api-key", default=None, help="临时指定 key（默认读环境/配置文件）")
    parser.add_argument("--say", default=None, help="合成这句话，存成 wav")
    parser.add_argument("--out", default=None, help="输出 wav 路径，默认 voice/cache/tts.wav")
    parser.add_argument("--asr", default=None, help="识别这个 wav")
    parser.add_argument("--ask", default=None, help="问一句话，走云端对话")
    parser.add_argument("--search", action="store_true", help="问话时开联网搜索")
    parser.add_argument("--check", action="store_true", help="只检查配置")
    args = parser.parse_args(argv)

    cfg = CloudConfig.from_file()
    if args.api_key:
        cfg.api_key = args.api_key.strip()
    voice = CloudVoice(cfg)

    print("key      : {}".format("已配置（{}…）".format(cfg.api_key[:6]) if cfg.api_key else "缺失"))
    print("tts      : {} / {} @ {} Hz".format(cfg.tts_model, cfg.tts_voice, cfg.sample_rate))
    print("asr      : {}".format(cfg.asr_model))
    print("chat     : {}".format(cfg.chat_model))
    if args.check or not any((args.say, args.asr, args.ask)):
        return 0 if cfg.ready() else 1

    try:
        if args.say:
            out = Path(args.out) if args.out else (_HERE / "cache" / "tts.wav")
            path = voice.tts(args.say, out)
            print("wav      : {} ({} 字节)".format(path, path.stat().st_size))
        if args.asr:
            print("识别结果 : {}".format(voice.asr(args.asr) or "（空）"))
        if args.ask:
            print("回答     : {}".format(voice.ask(args.ask, search=args.search)))
        return 0
    except (CloudVoiceNotConfigured, CloudVoiceError) as exc:
        print("错误：{}".format(exc))
        return 2


if __name__ == "__main__":
    import sys

    sys.exit(_main())
