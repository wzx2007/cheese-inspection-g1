#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""项目主程序入口：三次抓取 + 双面霉菌识别 + 有霉放笑声。

动作流程和现场参数在 ``demo_cheese.py``，识别在 ``vision/``，声音在 ``voice/``。
这里只做转发，方便固定一条命令：

    export PYTHONPATH=/data/galbot/lib:$PYTHONPATH
    export LD_LIBRARY_PATH=/data/galbot/lib:$LD_LIBRARY_PATH

    python3 main.py --confirm-motion                # 完整演示（3 次抓取 + 识别 + 声音）
    python3 main.py --confirm-motion --debug        # 每步回车
    python3 main.py --confirm-motion --no-vision    # 只跑动作
    python3 main.py -h                              # 全部参数
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from demo_cheese import main  # noqa: E402

if __name__ == "__main__":
    # 2026-11-08 现场事故：用户跑的其实是这个（键盘驱动、3 圈、每圈按回车），
    # 然后对着机器人喊"星宝星宝"——它压根不听语音，看起来就是"语音启动不了"。
    # 所以进来先把这件事说清楚，别让人猜。
    print("=" * 68)
    print("注意：main.py 是【键盘驱动】的老演示（默认 3 圈，每圈按回车），")
    print("      它**不听语音**。全程不会有任何唤醒/问答。")
    print()
    print("      要用语音控制星宝：")
    print("          bash stop_voice.sh                       # 先清场")
    print("          python3 xingbao.py --confirm-motion      # 默认测 1 块")
    print("=" * 68)
    raise SystemExit(main())
