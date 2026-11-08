"""视觉：奶酪表面霉菌检测（纯传统 OpenCV，不用神经网络）。

- ``mold``      分割奶酪 + Black-Hat 找"比周围暗得够深"的斑块
- ``inspector`` 一次"查看"的完整动作：连拍 N 帧 -> 判定 -> 存档 -> 有霉放笑声
- ``config``    相机与检测阈值，现场调参只改一个文件

判据只有一条：**所有超过一定深度的黑色都算菌**，不区分孔洞。
2026-11-07 用户定调："没有'孔洞'这个事儿。"
"""

from .config import (  # noqa: F401
    CAMERA_NAME,
    FRAME_MIN_RATIO,
    HIT_FRAME_RATIO,
    LOOK_FRAMES,
    LOOK_INTERVAL_S,
    LOOK_SETTLE_S,
)
from .inspector import CheeseInspector, open_camera  # noqa: F401
from .mold import analyze_bgr, annotate, find_spots, judge_frames, segment  # noqa: F401

__all__ = [
    "CheeseInspector",
    "open_camera",
    "analyze_bgr",
    "annotate",
    "find_spots",
    "judge_frames",
    "segment",
    "CAMERA_NAME",
    "FRAME_MIN_RATIO",
    "HIT_FRAME_RATIO",
    "LOOK_FRAMES",
    "LOOK_INTERVAL_S",
    "LOOK_SETTLE_S",
]
