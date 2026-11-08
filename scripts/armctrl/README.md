# 手臂接口

左右臂一起动。命令只写右臂，左臂关节是右臂逐项取负。位置单位米，末端关节单位弧度。躯干不俯身。

SDK 的 `robot`、`motion` 初始化并挂上左右工具后：

```python
from galbot_sdk.g1 import MotionStatus
from armctrl.arm_controller import RightArmController
from armctrl.sdk_arm_planner import SdkArmPlanner

for chain in ("right_arm", "left_arm"):
    status = motion.attach_tool(chain=chain, tool="galbot_gripper")
    if status != MotionStatus.SUCCESS:
        raise RuntimeError(status)
planner = SdkArmPlanner(robot, motion)
planner.check_tcp_frame()
arm = RightArmController(planner)
```

| 调用 | 前置状态 | 作用 |
| --- | --- | --- |
| `arm.reset_both_arm()` | 任意 | 双手回到高举，状态变为 initial |
| `arm.set_both_arm_ready()` | initial | 双手放到 Ready。右 joint7 为 1.538，左为 -1.538 |
| `arm.set_half_gap(0.32)` | ready | 右 TCP 到 `y=-0.32`，左臂使用这组关节的负值 |
| `arm.set_end_joint(0.0)` | ready | 右 joint7 到该弧度，左 joint7 取反。范围 `[-1.538, 1.538]` |
| `arm.state_snapshot()` | 任意 | 只读左右关节和左右 TCP |

通常按 `reset_both_arm → set_both_arm_ready → set_half_gap → set_end_joint → reset_both_arm` 调用。

真机演示：

```bash
python3 demo_cheese.py --confirm-motion
```

参数写在 `demo_cheese.py` 开头。本机从项目根目录导入时，在 `armctrl` 前加 `scripts.`。
