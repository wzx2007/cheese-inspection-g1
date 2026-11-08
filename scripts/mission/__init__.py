# -*- coding: utf-8 -*-
"""任务层：识别结果 -> 去哪个终点。

和 ``vision`` / ``voice`` 平级，只依赖 ``navctrl.ChassisNavigator``，
不直接碰 SDK，方便在 PC 上用假对象跑回归。
"""
