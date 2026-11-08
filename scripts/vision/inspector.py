# -*- coding: utf-8 -*-
"""一次"查看"的完整动作：连拍 -> 判定 -> 播报 -> 存档。

设计约束（现场踩出来的）
------------------------
1. **不要订阅视频流**。``subscribe_video_data`` 能在移动中连续录像，但解订阅时
   回调线程还在写，收尾容易段错误。识别只要几张静止图，用
   ``get_rgb_data(..., once=True)`` 逐帧取最新图，干净、也没有线程。
2. **必须显式开相机**。``robot.init()`` 的默认传感器集合里没有相机，
   要 ``robot.init({SensorType.HEAD_LEFT_CAMERA})`` 才拿得到图。
3. **看不清就不要下结论**。分不出奶酪时输出"看不到奶酪"，
   既不判有霉也不判没霉，避免把"没对准"说成"干净"。
4. **判完立刻播报当时的结果**（2026-11-08 现场要求）：没霉 -> PASS、
   有霉 -> FAIL，各一支音频。播报是非阻塞的，手臂可以同时继续走动作。
   音频台词已用 ASR 核对过（见 ``announce`` 的注释），别再搞反。
"""

from __future__ import annotations

import json
import os
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from . import config, mold

VOICE_PASS = "pass"
VOICE_FAIL = "fail"

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]   # 仓库根（scripts 的上一层）



def _width(text):
    """显示宽度：中日韩全角算 2 列，其余算 1 列（表格对齐用）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in str(text))


def _pad(text, width):
    text = str(text)
    return text + " " * max(0, width - _width(text))


class CheeseInspector:
    """握住奶酪后，把某一面转向相机拍几张，判有没有霉菌。"""

    def __init__(
        self,
        robot,
        voice=None,
        camera=None,
        out_dir=None,
        frames=None,
        interval=None,
        settle=None,
        enabled=True,
        verbose=True,
    ):
        self._robot = robot
        self._voice = voice
        self._camera = camera or config.CAMERA_NAME
        self._frames = int(config.LOOK_FRAMES if frames is None else frames)
        self._interval = float(config.LOOK_INTERVAL_S if interval is None else interval)
        self._settle = float(config.LOOK_SETTLE_S if settle is None else settle)
        self._verbose = bool(verbose)
        self.enabled = bool(enabled)
        self._cam = None
        self._broken = False
        self.results = []
        self._stamp = time.strftime("%Y%m%d_%H%M%S")
        root = Path(out_dir) if out_dir else (_DEFAULT_ROOT / "runs")
        self.out_dir = root / self._stamp
        self._saved = False

    # ------------------------------------------------------------------
    # 取帧
    # ------------------------------------------------------------------
    def _sensor(self):
        if self._cam is None:
            from galbot_sdk.g1 import SensorType

            self._cam = getattr(SensorType, self._camera)
        return self._cam

    def grab(self):
        """取一帧 BGR。失败返回 None（视觉不该拖垮整场演示）。"""
        if self._broken:
            return None
        try:
            from galbot_sdk.g1 import RgbOutputFormat

            data = self._robot.get_rgb_data(self._sensor(), RgbOutputFormat.BGR, True)
            if isinstance(data, dict) and data.get("data"):
                w = int(data.get("width") or 0)
                h = int(data.get("height") or 0)
                stride = int((data.get("stride_bytes") or [0])[0])
                buf = np.frombuffer(data["data"], np.uint8)
                if w > 0 and h > 0 and buf.size >= h * w * 3:
                    stride = stride or w * 3
                    if stride == w * 3:
                        return np.ascontiguousarray(buf[: h * w * 3].reshape(h, w, 3))
                    rows = buf[: h * stride].reshape(h, stride)
                    return np.ascontiguousarray(rows[:, : w * 3].reshape(h, w, 3))
        except Exception as exc:  # noqa: BLE001
            print("  取图（BGR）异常：{}".format(exc))

        try:                                    # 退路：JPEG 自己解
            from galbot_sdk.g1 import RgbOutputFormat

            data = self._robot.get_rgb_data(self._sensor(), RgbOutputFormat.JPEG, True)
            raw = data.get("data") if isinstance(data, dict) else None
            if raw:
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
        except Exception as exc:  # noqa: BLE001
            print("  取图（JPEG）异常：{}".format(exc))

        print("  ⚠ 取不到图像，本次演示关闭视觉（检查 camera={} 是否在 robot.init 里打开）"
              .format(self._camera))
        self._broken = True
        return None

    # ------------------------------------------------------------------
    # 查看
    # ------------------------------------------------------------------
    def look(self, tag, slug=None):
        """把当前视野当成某一面：连拍 N 帧判定一次。"""
        if not self.enabled:
            return None
        slug = slug or "look{:02d}".format(len(self.results) + 1)
        time.sleep(self._settle)

        recs = []
        for i in range(self._frames):
            bgr = self.grab()
            if bgr is None:
                break
            rec = mold.analyze_bgr(bgr, "{}_{:02d}".format(slug, i))
            recs.append(rec)
            self._save(slug, i, bgr, rec)
            if i + 1 < self._frames:
                time.sleep(self._interval)

        if not recs:
            res = dict(tag=tag, slug=slug, mold=False, verdict="取图失败", hits=0, n=0,
                       n_frames=0, hit_ratio=0.0, max_ratio=0.0, reason="no_frame", frames=[])
        else:
            res = mold.judge_frames(recs)
        res["tag"] = tag
        res["slug"] = slug
        res["time"] = time.strftime("%H:%M:%S")
        self.results.append(res)

        if self._verbose:
            self._print_look(res)
        self._write_json()

        self.announce(res)
        return res

    # ------------------------------------------------------------------
    # 播报
    # ------------------------------------------------------------------
    def announce(self, res):
        """判完立刻播报这一次的结果：没霉 -> PASS，有霉 -> FAIL。

        2026-11-08 修正（用户现场发现播反了）：两支音频的实际台词是
            result_pass.wav = "检测完成，表面未发现明显异常。"   <- 没霉
            result_fail.wav = "检测完成，发现疑似异常，请师傅复核。" <- 有霉
        之前写成"有霉 -> PASS"，语义正好颠倒。现已按音频原意对齐：
        **mold=True 播 fail，mold=False 播 pass**。

        非阻塞（``VoiceBox.say`` 入队即返回），手臂可以同时继续走动作。
        取不到图/没分出奶酪时没有结论，按"没看到 ≠ 干净"播 fail，
        同时打一行警告，免得把"没对准"悄悄当成合格。
        """
        if self._voice is None:
            return False
        if res.get("reason") == "ok":
            name = VOICE_FAIL if res.get("mold") else VOICE_PASS
        else:
            # 没取到图 / 没分出奶酪：没有结论，不能当合格播 pass。
            name = VOICE_FAIL
            if self._verbose and res.get("reason") == "no_cheese":
                print("     ⚠ 本面没分出奶酪，仍按 FAIL 播报（没看到 ≠ 干净）")
        return self._voice.say(name)

    def has_mold(self):
        """整场任意一面判为有霉。"""
        return any(r.get("mold") for r in self.results)

    def last_verdict(self):
        return self.results[-1] if self.results else None

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    def _print_look(self, res):
        print("  ── 查看：{} ──".format(res["tag"]))
        if res.get("reason") == "no_cheese":
            print("     {} 帧都没有分出奶酪（没对准 / 被手臂挡住），本面不下结论".format(
                res.get("n_frames", 0)))
            return
        if res.get("reason") != "ok":
            print("     结论：{}".format(res.get("verdict")))
            return
        print("     帧  奶酪  菌斑  最大占比    中心")
        for r in res["frames"]:
            if not r.get("cheese"):
                print("     {:<3}  -     -     -".format(r["frame"].split("_")[-1]))
                continue
            top = max(r["spots"], key=lambda s: s["ratio"]) if r["spots"] else None
            print("     {}  ✅    {}  {} {}".format(
                _pad(r["frame"].split("_")[-1], 3), _pad(len(r["spots"]), 4),
                _pad("{:.4f}%".format(r["max_ratio"] * 100), 9),
                "({}, {})".format(top["cx"], top["cy"]) if top else "-",
            ))
        print("     结论：{}（{}/{} 帧命中，最大 {:.4f}%）".format(
            res["verdict"], res["hits"], res["n"], res["max_ratio"] * 100))

    def summary(self):
        """整场演示的汇总表。返回 (有霉, 明细)。"""
        prints = [r for r in self.results if r.get("reason") == "ok"]
        hit = any(r["mold"] for r in prints)
        if self._verbose and self.results:
            print("\n  ── 全场判定 ──")
            print("     {} {} {} {} 结论".format(
                _pad("查看", 22), _pad("帧数", 5), _pad("命中", 5), _pad("最大占比", 11)))
            for r in self.results:
                print("     {} {} {} {} {}".format(
                    _pad(r["tag"], 22), _pad(r.get("n_frames", 0), 5),
                    _pad(r.get("hits", 0) if r.get("reason") == "ok" else "-", 5),
                    _pad("{:.4f}%".format(r.get("max_ratio", 0.0) * 100), 11),
                    r.get("verdict", ""),
                ))
            print("     总体：{}".format("FAIL（有霉菌）" if hit else "PASS（没有霉菌；或未看清，见各面结论）"))
            if self._saved:
                print("     证据：{}".format(self.out_dir))
        return hit, self.results

    def _save(self, slug, index, bgr, rec):
        try:
            if not self._saved:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                self._saved = True
            cv2.imwrite(str(self.out_dir / "{}_{:02d}.jpg".format(slug, index)), bgr)
            if rec.get("mold"):
                ann = mold.annotate(bgr, rec)
                cv2.imwrite(str(self.out_dir / "{}_{:02d}_norm.jpg".format(slug, index)), ann)
        except Exception as exc:  # noqa: BLE001
            print("  存档失败：{}".format(exc))

    def _write_json(self):
        if not self._saved:
            return
        try:
            payload = dict(stamp=self._stamp, camera=self._camera,
                           frames_per_look=self._frames, results=self.results)
            path = self.out_dir / "verdict.json"
            with open(str(path), "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        except Exception as exc:  # noqa: BLE001
            print("  写 JSON 失败：{}".format(exc))


def open_camera(camera=None):
    """给 ``GalbotRobot.init`` 用的传感器集合。默认集合里没有相机。"""
    from galbot_sdk.g1 import SensorType

    return {getattr(SensorType, camera or config.CAMERA_NAME)}
