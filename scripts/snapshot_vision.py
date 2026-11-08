# -*- coding: utf-8 -*-
"""静态查看：不动任何关节，只用头部相机拍几张判霉菌。

上机第一步先跑这个 —— 机器停在原地看着奶酪，确认"相机拿得到图 + 分得出奶酪"，
再跑整场演示。也只初始化相机，不做任何运动。

    export PYTHONPATH=/data/galbot/lib:$PYTHONPATH
    export LD_LIBRARY_PATH=/data/galbot/lib:$LD_LIBRARY_PATH

    python3 snapshot_vision.py                       # 连拍 6 帧
    python3 snapshot_vision.py --frames 12 --interval 0.3
    python3 snapshot_vision.py --out /home/galbot/runs

图片和 verdict.json 存在 ``--out/<时间戳>/``。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from vision import CheeseInspector, open_camera  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="静态查看奶酪表面霉菌（不动机器人）")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument("--settle", type=float, default=None)
    parser.add_argument("--camera", default=None, help="默认 HEAD_LEFT_CAMERA")
    parser.add_argument("--out", default=None, help="存档目录，默认 <仓库根>/runs")
    parser.add_argument("--confirm-motion", action="store_true",
                        help="与演示脚本保持一致；本脚本不运动，加不加都行")
    args = parser.parse_args(argv)

    from galbot_sdk.g1 import GalbotRobot

    robot = GalbotRobot()
    if not robot.init(open_camera(args.camera)):
        print("robot.init(相机) 失败", file=sys.stderr)
        return 1
    time.sleep(3.0)
    try:
        insp = CheeseInspector(robot, voice=None, camera=args.camera, out_dir=args.out,
                               frames=args.frames, interval=args.interval,
                               settle=args.settle)
        frame = insp.grab()
        if frame is None:
            print("取不到图像：检查相机名与 robot.init 的传感器集合", file=sys.stderr)
            return 1
        print("相机 {}  分辨率 {}x{}".format(insp._camera, frame.shape[1], frame.shape[0]))
        res = insp.look("静态查看", "static")
        insp.summary()
        return 0 if res and res.get("reason") == "ok" else 1
    finally:
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
