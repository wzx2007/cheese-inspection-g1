# -*- coding: utf-8 -*-
"""迁移验收：拿历史抓取录像跑一遍，数字必须和 galbot_nav 时期完全一致。

这段检测在 ``galbot_nav/cheese_pipeline/vision/mold.py`` 里是 2026-11-07 标定好的，
搬进本项目时一行阈值都没改。这个脚本只做一件事：**用录像复现当时的结论**，
确认迁移没有走样。只需要 cv2 + numpy，不需要 SDK，可以在 PC 上跑::

    python3 vision/selfcheck.py --video videos/03_grab_indexed.mp4
    python3 vision/selfcheck.py --video a.mp4 --video b.mp4 --json out.json

标定基准（2026-11-07，1080p 录像，正面 0-5 帧 / 背面 200-205 帧）::

    03_grab_indexed.mp4   正面 6/6 命中 24 斑 最大 1.6791%   背面 0/6  -> 有霉菌
    04_grab_run2.mp4      正面 0/6                            背面 6/6 最大 0.1611% -> 有霉菌
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from vision import mold  # noqa: E402

FRONT_WINDOW = (0, 5)
BACK_WINDOW = (200, 205)


def read_windows(path, windows):
    """解码一遍，只留需要的帧号。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("打不开视频: %s" % path)
    want = {}
    for key, (lo, hi) in windows.items():
        for i in range(lo, hi + 1):
            want[i] = key
    recs = {key: [] for key in windows}
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx in want:
            recs[want[idx]].append(mold.analyze_bgr(bgr, "%s_f%04d" % (want[idx], idx)))
        idx += 1
    cap.release()
    return recs, idx


def main(argv=None):
    parser = argparse.ArgumentParser(description="视觉迁移验收（离线录像）")
    parser.add_argument("--video", action="append", required=True, help="可重复")
    parser.add_argument("--json", default=None, help="把明细写到这个 json")
    args = parser.parse_args(argv)

    out = {}
    for video in args.video:
        recs, total = read_windows(video, {"front": FRONT_WINDOW, "back": BACK_WINDOW})
        res = {}
        print("\n视频 {}（解码 {} 帧）".format(os.path.basename(video), total))
        print("  面    帧数  命中  斑数  最大占比     结论")
        for key in ("front", "back"):
            blk = mold.judge_frames(recs[key])
            spots = sum(r.get("n_mold", 0) for r in blk["frames"])
            print("  {:<5} {:<5} {:<5} {:<5} {:<11} {}".format(
                key, blk["n_frames"], blk["hits"], spots,
                "{:.4f}%".format(blk["max_ratio"] * 100), blk["verdict"]))
            res[key] = dict(frames=blk["n_frames"], hits=blk["hits"], spots=spots,
                            max_ratio=round(blk["max_ratio"], 6), mold=blk["mold"],
                            verdict=blk["verdict"])
        res["mold"] = res["front"]["mold"] or res["back"]["mold"]
        print("  总体：{}".format("有霉菌" if res["mold"] else "没有霉菌"))
        out[os.path.basename(video)] = res

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print("\n明细 -> {}".format(args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
