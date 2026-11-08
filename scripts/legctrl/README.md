# 腿部接口

只升降，不俯身。SDK 的 `robot` 初始化后：

```python
from galbot_sdk.g1 import ControlStatus, G1ControllerName
from legctrl.leg_controller import LegController

status = robot.switch_controller(G1ControllerName.LEG_PVT_CTRL)
if status != ControlStatus.SUCCESS:
    raise RuntimeError(status)
leg = LegController(robot, joint_speed_rad_s=0.1, verify=False)
```

| 调用 | 作用 |
| --- | --- |
| `leg.read_state()` | 只读当前头高。`pitch_deg` 只是测量值 |
| `leg.control_height(1.35)` | 设置绝对头高，米，范围 `[0.96, 1.54]` |
| `leg.reset_height()` | 降到最低头高 0.96 m |

返回 `LegState`。`height_m` 是 `base_link` 下 `head_base_link` 的 z。

演示里的身高写在 `demo_cheese.py` 开头，先略低，Ready 之后升高，双手回到高举后再降回去。本机从项目根目录导入时，在 `legctrl` 前加 `scripts.`。
