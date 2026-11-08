# G1 API 调用说明

依据本地 Galbot SDK 1.10.0。

- 签名和说明：`refs/GalbotSDK-main/docs/g1/zh/api_python_reference/index.html`
- 调用代码：`refs/GalbotSDK-main/examples/g1/python`
- 下文例程路径都相对 `examples/g1/python`

包名是 `galbot_sdk.g1`。三个控制类：

| 类 | 作用 |
| --- | --- |
| `GalbotRobot` | 初始化、关节、夹爪、相机、声音、底盘速度、身高 |
| `GalbotMotion` | 规划、逆解、正解、笛卡尔位姿、直线、障碍物 |
| `GalbotNavigation` | 建图完成之后的底盘导航 |

位姿是 7 个数：`[x, y, z, qx, qy, qz, qw]`，位置米，四元数 xyzw，必须归一化。关节角弧度。

`ControlStatus` 里例程判断成功用的是 `ControlStatus.SUCCESS`。`MotionStatus` 在 `galbot_motion/fk.py` 里列出的取值是 `SUCCESS`、`TIMEOUT`、`FAULT`、`INVALID_INPUT`、`INIT_FAILED`、`IN_PROGRESS`、`STOPPED_UNREACHED`、`DATA_FETCH_FAILED`、`PUBLISH_FAIL`、`COMM_DISCONNECTED`。

---

## 脚本放在哪里跑

可以在电脑上用 Python 直接控制，也可以把脚本放到机器人上跑。Windows 自带的 Python 不在支持范围内。

文档写的系统是 Ubuntu 20.04–24.04。电脑端架构是 x86_64，机器人端是 aarch64，Python 3.8–3.14。官方电脑部署只有 Ubuntu、WSL2、VMware。这台 Windows 上有两条路：

1. WSL2 或 Ubuntu 虚拟机里安装 SDK，网线连上后在虚拟机里 `python your_app.py`。
2. 把 `.py` 拷到 Orin 的 `/userdata`，SSH 上去运行。文档把这条叫做机器人端部署，通信开销更小。

两条路都是 `import galbot_sdk.g1`。网口不只是 SSH。SSH 用来登录和拷文件。控制走 embosa。电脑有线网口配到机器人网段后，在 SDK 目录执行 `./configure_embosa_ip.sh`，机型选 G-series。

| 设备 | IP |
| --- | --- |
| 电脑 | `<PC_IP>`，掩码 `255.255.255.0` |
| XCU | `<XCU_IP>` |
| HPU / Orin | `<ROBOT_IP>` |

三端要能互相 ping 通。电脑上出现 `get info failed` 或 `PUBLISH FAIL`，先查两端 `embosa_ip_config.json`。

机器人端登录是 `ssh galbot@<机器人IP>`，用户名 `galbot`，密码 `<ROBOT_PASSWORD>`。库在 `/data/galbot/lib`。

应用代码不处理 ROS 通信。`init()` 接的是 SDK 中间件。资料里的 ROS 2 只说明图像、点云、时间戳的数据格式与之兼容。排障工具是 HPU 上的 `/data/galbot/bin/embosa_topic_tool`。

手臂规划例程要求这三个进程已经在机器人上：`/data/galbot/bin/service_motion_plan`、`/data/galbot/bin/robot_state_publish`、`/data/galbot/bin/eyehand_calib_publish`。导航才需要先建图。只做手臂、头、夹爪、升降时，不建图也可以动。

急停按下时不能切换工作模式，平台提示 `Cannot switch mode while emergency stop is active`。

---

## 1. 启动和关闭

`GalbotRobot.init` 只第一次生效。传感器集合留空则打开默认传感器。第二个参数 `enable_sync_mode=True` 之后才能用 `get_synced_observation`。`destroy()` 之后同一进程不能再 `init`，要重新开进程。

签名：

```python
def init(enable_sensor_set: Set[SensorType] = ..., enable_sync_mode: bool = False) -> bool
def request_shutdown() -> None
def wait_for_shutdown() -> None
def destroy() -> None
def is_running() -> bool
```

`galbot_robot/set_joint_positions.py` 的生命周期：

```python
robot = GalbotRobot()
robot.init()
time.sleep(2)  # 等状态数据
# ... 控制 ...
robot.request_shutdown()
robot.wait_for_shutdown()
robot.destroy()
```

`GalbotMotion` 单独初始化，见 `galbot_motion/fk.py`：

```python
motion = GalbotMotion()
robot = GalbotRobot()
motion.init()
robot.init()
time.sleep(1)
```

同步相机初始化见 `galbot_robot/get_synced_observation.py`。注释写明第二个参数必须是 `True`，否则 `get_synced_observation` 不可用：

```python
ok = robot.init(
    {
        SensorType.HEAD_LEFT_CAMERA,
        SensorType.LEFT_ARM_CAMERA,
        SensorType.RIGHT_ARM_CAMERA,
    },
    True,
)
```

---

## 2. 关节位置

低频路点用这个。文档写明它不适合逐帧高频控制。默认速度 `0.2` rad/s，默认阻塞，超时 15 秒。`joint_names` 非空时盖过 `joint_groups`。两组都空则返回 `INVALID_INPUT`。支持的组名：`leg`、`head`、`left_arm`、`right_arm`。

```python
def set_joint_positions(
    joint_positions: list[float],
    joint_groups: Sequence[str] = [],
    joint_names: Sequence[str] = [],
    is_blocking: bool = True,
    speed_rad_s: SupportsFloat = 0.2,
    timeout_s: SupportsFloat = 15.0,
) -> ControlStatus
```

`galbot_robot/set_joint_positions.py` 先按组，再按关节名。空的 `joint_groups` 在注释里说会落到全身 `["leg", "head", "left_arm", "right_arm"]`，所以按名字控制时把组参数传空列表：

```python
status = robot.set_joint_positions(
    [0.2, 0.2], ["head"], [], True, 0.1, 10
)

status = robot.set_joint_positions(
    [0.0, 0.0], [], ["head_joint1", "head_joint2"], True, 0.1, 10
)
```

读当前角度，`example1_galbot_control_started.py`：

```python
original_pos = robot.get_joint_positions(joint_group_names, [])
```

`get_joint_names(only_active_joint=True, joint_groups=[])` 用来确认顺序，不要手写没核对过的关节名。

`G1JointGroup` 里和这次有关的组：

| 常量 | 默认关节 |
| --- | --- |
| `leg` | `leg_joint1` … `leg_joint5` |
| `head` | `head_joint1`、`head_joint2` |
| `left_arm` / `right_arm` | 各 7 个 |
| `left_gripper` / `right_gripper` | 各 `*_gripper_joint1`，开合用夹爪接口 |

抓取教程 `example6_pick_and_place.py` 的初始姿态是先腿、再头和双臂，速度 `0.2`，超时 20 秒：

```python
INITIAL_LEG_JOINT_POSITIONS = [0.4, 1.2, 0.8, 0.0, 0.0]
INITIAL_UPPER_BODY_JOINT_POSITIONS = [
    0.0, 0.0,
    1.90, -1.46, -0.54, -1.96, 0.0, -0.4, 0.0,
    -1.90, 1.46, 0.54, 1.96, 0.0, 0.4, 0.0,
]
robot.set_joint_positions(INITIAL_LEG_JOINT_POSITIONS, joint_names=INITIAL_LEG_JOINT_NAMES, is_blocking=True, speed_rad_s=0.2, timeout_s=20.0)
robot.set_joint_positions(INITIAL_UPPER_BODY_JOINT_POSITIONS, joint_names=INITIAL_UPPER_BODY_JOINT_NAMES, is_blocking=True, speed_rad_s=0.2, timeout_s=20.0)
```

预规划轨迹用 `execute_joint_trajectory(trajectory, is_blocking=True)`。标准关节当前版本只使用位置，速度、加速度、力矩被忽略。例程是 `galbot_robot/execute_joint_trajectory.py`。停掉正在走的轨迹：`stop_trajectory_execution()`，停后关节停在当前位置。非阻塞时用 `check_trajectory_execution_status(joint_groups)` 看进度。

---

## 3. 夹爪

宽度单位米。默认速度 `0.03` m/s，默认力 `5`，默认阻塞。

```python
def set_gripper_command(
    end_effector: str,
    width_m: SupportsFloat,
    velocity_mps: SupportsFloat = 0.03,
    effort: SupportsFloat = 5,
    is_blocking: bool = True,
) -> ControlStatus
```

`galbot_robot/set_gripper_command.py`：

```python
from galbot_sdk.g1 import GalbotRobot, G1JointGroup, ControlStatus

status = robot.set_gripper_command(G1JointGroup.left_gripper, 0.02, 0.05, 10, True)  # 闭合
status = robot.set_gripper_command(G1JointGroup.left_gripper, 0.1, 0.05, 10, True)   # 张开
```

`example6_pick_and_place.py` 用同一组数，但 `is_blocking=False`，然后 `sleep(0.5)`：张开 `0.1`，闭合 `0.02`，速度 `0.05`，力 `10`。右夹爪把 `G1JointGroup.left_gripper` 换成 `right_gripper`。

状态：`get_gripper_state(end_effector) -> GripperState`。`is_moving` 看一段时间里开合有没有变化。

规划抓取前要把工具挂上，否则笛卡尔目标是法兰而不是夹爪 TCP。`example6`：

```python
status = motion.attach_tool(chain="left_arm", tool="galbot_gripper")
# 结束
status = motion.detach_tool(chain="left_arm")
```

`attach_tool` 的签名是 `attach_tool(chain: str, tool: str) -> MotionStatus`。可用工具名用 `get_supported_tool_list()` 查。

---

## 4. 身高

调的是 `head_base_link` 在 `base_link` 下的 Z，不是从地面到头顶的整机高度。范围 **0.96–1.54 m**。时长 0–60 秒，0 表示尽快。默认下发后返回；`is_blocking=True` 等到结束。

```python
def set_leg_height(
    link_height: SupportsFloat,
    duration: SupportsFloat = 4.0,
    is_blocking: bool = False,
    timeout_s: SupportsFloat = -1.0,
) -> ControlStatus
```

`galbot_robot/set_leg_height_example.py` 先读再降 0.20 m，时长 4 秒，阻塞：

```python
pose, timestamp_ns = robot.get_transform(
    "base_link", "head_base_link", timestamp_ns=0, timeout_ms=500
)
height_m = pose[2]
status = robot.set_leg_height(height_m - 0.20, 4.0, True)
```

`get_transform` 的签名是 `get_transform(target_frame, source_frame, timestamp_ns=0, timeout_ms=100) -> tuple`。文档说明当前高度就是这次查询结果的 z。

---

## 5. 笛卡尔位姿

规划并执行用 `GalbotMotion.set_end_effector_pose`。内部做逆解、轨迹和碰撞检查。`end_effector_frame` 填 `"left_arm"` 或 `"right_arm"`，选的是哪条臂。目标是法兰还是夹爪 TCP，由 `Parameter.set_tool_pose` 决定。参考系默认 `base_link`。

```python
def set_end_effector_pose(
    target_pose: list[float],
    end_effector_frame: str,
    reference_frame: str = "base_link",
    reference_robot_states: RobotStates = None,
    enable_collision_check: bool = True,
    is_blocking: bool = True,
    timeout: SupportsFloat = -1.0,
    params: Parameter = ...,
) -> MotionStatus
```

还有一个重载，在 `reference_frame` 后面多一个 `assist_chains: Set[str]`，用来带躯干等辅助链。

`example6_pick_and_place.py` 的参数和预抓：

```python
motion_params = gm.Parameter()
motion_params.set_direct_execute(True)
motion_params.set_blocking(True)
motion_params.set_timeout(20.0)
motion_params.set_tool_pose(True)          # 控制夹爪 TCP，不是法兰
motion_params.set_check_collision(True)
motion_params.set_reference_frame("base_link")
motion_params.set_actuate("with_chain_only")

status = motion.set_end_effector_pose(
    target_pose=pre_grasp_pose,            # [x, y, z, qx, qy, qz, qw]
    end_effector_frame="left_arm",
    reference_frame="base_link",
    enable_collision_check=True,
    is_blocking=True,
    timeout=20.0,
    params=motion_params,
)
```

教程里的占位抓取位姿是 `[0.4, 0.3, 0.65, 0.0, 0.0, 0.0, 1.0]`。预抓把 x 减 `0.05`，抬起把 z 加 `0.05`。

`galbot_motion/get_set_end_effector_pose.py` 是不带工具、不碰撞、非阻塞的写法，超时后自己 `sleep`：

```python
status, pose = motion.get_end_effector_pose_on_chain(
    chain_name="left_arm", frame_id="EndEffector", reference_frame="base_link"
)
status = motion.set_end_effector_pose(
    target_pose=chain_pose_baselink["left_arm"],
    end_effector_frame="left_arm",
    reference_frame="base_link",
    enable_collision_check=False,
    is_blocking=False,
    timeout=10.0,
    params=custom_param,
)
```

按 URDF 连杆名读位姿用 `get_end_effector_pose(end_effector_frame, reference_frame="base_link")`。

高频跟踪不要用上面这个。用 `GalbotRobot.set_end_effector_command`。它不做规划，`reference_frames` 省略时每条位姿默认 `"world"`。`galbot_robot/set_end_effector_commands.py` 要求配置里 `using_wbc` 为 true：

```python
ee_info = robot.get_wbc_end_effector_poses()  # 键：lee_pose、ree_pose、head_pose
target_pose = list(ee_info["ree_pose"][:7])
target_pose[0] -= 0.1
robot.set_end_effector_command(
    poses=[target_pose],
    end_effector_frames=["right_arm_end_effector_mount_link"],
)
robot.clear_end_effector_command()             # 清除并回到默认参考关节位姿
# robot.clear_end_effector_command(hold=True)  # 清除并停在当前关节位姿
```

---

## 6. 直线运动

最后一段靠近和放下用 `move_line`。末端在笛卡尔空间走直线，中间点再做逆解。`motion_plan` 不保证末端是直线。

```python
def move_line(
    waypoints: Sequence[Sequence[MotionPlanChainTarget]],
    params: PlannerConfig = ...,
    start_state: RobotStates = None,
) -> tuple[MotionStatus, dict[str, list[list[float]]]]
```

`example6` 的单臂最后 5 cm：

```python
grasp_target = gm.MotionPlanChainTarget()
grasp_target.chain_name = "left_arm"
grasp_target.mode = gm.MotionPlanTargetMode.kCartesian
grasp_target.cart.chain_name = "left_arm"
grasp_target.cart.frame_id = "EndEffector"
grasp_target.cart.reference_frame = "base_link"
grasp_target.cart.pose = gm.Pose(grasp_pose)
status, _ = motion.move_line([[grasp_target]], motion_params)
```

外层列表的每个元素是一个路点，路点里可以同时放左右臂。`galbot_motion/move_line.py` 把左右臂四个路点串成一个方框，`params.set_actuate("with_torso")`，并关掉碰撞检查后直接执行：

```python
waypoints = [
    [
        cart("left_arm", [0.42666, 0.33435, 0.73569, 0.0, 0.0, 0.0, 1.0]),
        cart("right_arm", [0.422666, -0.33435, 0.73569, 0.0, 0.0, 0.0, 1.0]),
    ],
    # 后面三个路点把 z 抬到 1.03569，再把 x 收到 0.12666，再降回 0.73569
]
status, traj = motion.move_line(waypoints, params)
```

`motion_plan` 的多路点重载签名相同，走采样避障，末端路径形状由规划器决定。例程是 `galbot_motion/motion_plan.py`，调用形如 `status, traj = motion.motion_plan(waypoints, params)`。单目标重载是 `motion_plan(target: RobotStates, ...)`。只做关节插值、不搜索路径用 `traj_plan`。

障碍物不会自动从相机进来。`galbot_motion/add_obstacle.py` 写明要自己加：

```python
status = motion.add_obstacle(
    obstacle_id="box_test_1",
    obstacle_type="box",
    pose=[1.0, 0.0, 1.0, 0, 0, 0, 1],
    scale=[1.0, 1.0, 1.0],
    target_frame="world",
)
motion.clear_obstacle()
```

`obstacle_type` 用 `get_supported_obstacle_types()` 查。抓着的物体用 `attach_target_object`，松开用 `detach_target_object(obstacle_id)`。

---

## 7. 正解和逆解

逆解不直接让机器人动，返回各链关节角。`galbot_motion/ik.py`：

```python
status, joint_map = motion.inverse_kinematics(
    target_pose=chain_pose_baselink["left_arm"],
    chain_names=["left_arm"],
    target_frame="EndEffector",
    reference_frame="base_link",
    enable_collision_check=False,
)
```

同一文件里的四种用法：只规划 `left_arm`；`["left_arm", "torso"]`；`["left_arm", "torso", "head"]` 会返回 `INVALID_INPUT`；用 `initial_joint_positions=chain_joints` 做种子。`chain_names` 是链名列表，返回 `dict[str, list[float]]`。

签名：

```python
def inverse_kinematics(
    target_pose: list[float],
    chain_names: Sequence[str],
    target_frame: str = "EndEffector",
    reference_frame: str = "base_link",
    initial_joint_positions: dict = {},
    enable_collision_check: bool = True,
    params: Parameter = ...,
) -> tuple[MotionStatus, dict[str, list[float]]]
```

正解 `galbot_motion/fk.py`。不传关节就用当前状态。自定义关节是 `{链名: 角度列表}`：

```python
status, pose = motion.forward_kinematics("left_arm_end_effector_mount_link", "base_link")
status, pose = motion.forward_kinematics(
    "left_arm_end_effector_mount_link",
    "base_link",
    {"left_arm": chain_joints["left_arm"]},
    gm.Parameter(),
)
```

全身状态用 `motion.get_robot_states()`，再 `forward_kinematics_by_state`。`example2_arm_manipulation.py` 把逆解结果交回 `set_joint_positions`：

```python
status, joint_angles_ik = motion.inverse_kinematics(
    target_pose=chain_pose_baselink["left_arm"],
    chain_names=["left_arm"],
    target_frame="EndEffector",
    reference_frame="base_link",
    enable_collision_check=False,
)
status = robot.set_joint_positions(joint_angles_ik["left_arm"], ["left_arm"], [], True, 0.1, 20.0)
```

---

## 8. 高频关节命令

模型逐帧输出才用。它不从当前角度插到第一帧，控制器尽快跟上每个目标。头、腿、臂只有 `JointCommand.position` 有效。夹爪关节的 position 是宽度，速度和力有效。

```python
def set_joint_commands(
    joint_commands: Sequence[JointCommand],
    joint_groups: Sequence[str] = [],
    joint_names: Sequence[str] = [],
    time_from_start_s: SupportsFloat = 0.0,
) -> ControlStatus
```

`galbot_robot/set_joint_commands_example.py` 以 100 Hz 摆头 4 秒：

```python
joint_commands = [JointCommand(), JointCommand()]
joint_commands[0].position = target_position
joint_commands[1].position = target_position
GalbotRobot().set_joint_commands(joint_commands, ["head"], [], 0.0)
time.sleep(0.01)
```

`example5_execute_vla.py` 是 20 Hz、一次 30 帧。23 维顺序是腿 5、头 2、左臂 7、左夹爪 1、右臂 7、右夹爪 1。夹爪那一维在首帧用 `set_gripper_command(..., width_m=position)`，后续帧放进 `JointCommand.position`。没有示教模型时不要走这条。

一批轨迹点用 `set_joint_commands_batch(trajectory)`，非阻塞。

---

## 9. 相机

`init` 里没打开的传感器读不到。读之前例程会 `sleep` 1–5 秒。

```python
def get_rgb_data(camera_id: SensorType, format: RgbOutputFormat = ..., once: bool = True) -> dict
def get_depth_data(camera_id: SensorType) -> dict
def get_ir_data(camera_id: SensorType) -> dict
def subscribe_video_data(camera_id: SensorType, callback: Callable) -> SensorStatus
def unsubscribe_video_data(camera_id: SensorType) -> SensorStatus
def get_camera_intrinsic(camera_id: SensorType) -> dict
def get_sensor_extrinsic(sensor_id: SensorType, reference_frame: str = "base_link") -> tuple
def get_synced_observation(cameras: Sequence[SensorType], with_joint_state: bool = True) -> SyncedObservation
```

`get_rgb_data` 默认 JPEG。`once=True` 要一帧最新图。深度只接受 `LEFT_ARM_DEPTH_CAMERA` 和 `RIGHT_ARM_DEPTH_CAMERA`。

`galbot_robot/get_camera_data.py`：

```python
robot.init({
    SensorType.LEFT_ARM_CAMERA,
    SensorType.LEFT_ARM_DEPTH_CAMERA,
    SensorType.LEFT_ARM_INFRA_CAMERA_1,
    SensorType.LEFT_ARM_INFRA_CAMERA_2,
})
time.sleep(5)
rgb_image_data = robot.get_rgb_data(SensorType.LEFT_ARM_CAMERA, RgbOutputFormat.JPEG, True)
depth_data = robot.get_depth_data(SensorType.LEFT_ARM_DEPTH_CAMERA)
```

JPEG 的 `data` 用 `cv2.imdecode`。深度是 `uint16`，按返回的 `height`、`width` 整形，再除以 `depth_scale`。头部停留拍照把相机换成 `SensorType.HEAD_LEFT_CAMERA`。`installation_welcome_verify.py` 就是这样存头部 JPEG 的。

同步观测的 RGB 是 NV12，不是 JPEG。第一个相机是时间基准。`get_synced_observation.py`：

```python
cameras = [
    SensorType.LEFT_ARM_CAMERA,
    SensorType.RIGHT_ARM_CAMERA,
    SensorType.HEAD_LEFT_CAMERA,
]
obs = robot.get_synced_observation(cameras, True)
frame = obs.rgb_data_map.get(cameras[0])
```

连续 H.264 用 `galbot_robot/subscribe_video_data.py`：

```python
status = robot.subscribe_video_data(SensorType.HEAD_LEFT_CAMERA, video_data_callback)
```

`SensorType` 里和 G1 有关的：

| 枚举 | 内容 |
| --- | --- |
| `HEAD_LEFT_CAMERA` / `HEAD_RIGHT_CAMERA` | 头部左右彩色 |
| `LEFT_ARM_CAMERA` / `RIGHT_ARM_CAMERA` | 腕部彩色 |
| `LEFT_ARM_DEPTH_CAMERA` / `RIGHT_ARM_DEPTH_CAMERA` | 腕部深度 |
| `LEFT_ARM_INFRA_CAMERA_1/2`、`RIGHT_ARM_INFRA_CAMERA_1/2` | 腕部红外 |
| `LEFT_FRONT_SURROUND_CAMERA`、`LEFT_REAR_SURROUND_CAMERA` | 左前、左后环视 |
| `RIGHT_FRONT_SURROUND_CAMERA`、`RIGHT_REAR_SURROUND_CAMERA` | 右前、右后环视 |
| `BASE_LIDAR` | 底盘雷达，`get_lidar_data` |
| `BASE_ULTRASONIC` | 超声波，`get_ultrasonic_data` 的参数类型是 `UltrasonicType` |
| `CHASSIS_IMU`、`TORSO_IMU`、`LIDAR_IMU` | `get_imu_data` |

---

## 10. 声音

播放文件必须是 16 kHz、16 bit、单声道 PCM。`audio_example.py` 按 2560 字节一块写入。音量 0–100。

```python
def write_audio_stream_output(audio_chunk: str, stream_id: str = "") -> bool
def stop_audio_stream_output(stream_id: str = "") -> None
def set_volume(volume: SupportsFloat) -> bool
def get_volume() -> float
def start_microphone_stream_input(callback: Callable, chunk_size: SupportsInt = 2560, use_raw_audio: bool = False) -> str
def stop_microphone_stream_input(stream_id: str = "") -> None
```

`audio_example.py`：

```python
microphone_stream_id = base_instance.start_microphone_stream_input(
    callback=audio_callback, chunk_size=2560, use_raw_audio=False
)
for chunk in audio_chunks:  # 从 16K 16bit Mono PCM 文件读出
    success = base_instance.write_audio_stream_output(chunk, speaker_stream_id)
base_instance.set_volume(volume)  # 0–100
```

`installation_welcome_verify.py` 把音量设成 `100.0` 再播欢迎 PCM。

---

## 11. 急停和底盘速度

软件急停立刻打断运动和规划。`galbot_robot/emergency_stop_and_resume_example.py` 在头关节运动中调用：

```python
status = robot.set_joint_positions([0.4], [], ["head_joint2"], True, 0.05, 15.0)
status = robot.emergency_stop()
```

恢复是 `resume_from_emergency_stop()`。物理急停按钮和这个软件接口是两件事。

底盘速度是开环，单位 m/s 和 rad/s。`galbot_robot/get_set_base_velocity.py`：

```python
base_velocity_before = robot.get_base_velocity()
# linear_velocity: [vx, vy, vz]，angular_velocity: [wx, wy, wz]
status = robot.set_base_velocity([0.2, 0.0, 0.0], [0.0, 0.0, 0.0], 3.0)
robot.stop_base()
```

例程注释写：发布结束后停不停取决于看门狗，完成不等于底盘已经停住。到位导航不要用这个，用第 12 节。

`set_base_pose` 有三个重载：直接给 `Pose`；给 `x, y, yaw`，默认 `frame_id="rel(0)"`、`reference_frame_id="odom"`；再加 `time_from_start_s`。

---

## 12. 导航

用之前必须建图并定位。网页是 `http://<HPU IP>:8000`，账号 `admin`，密码 `galbot`。定位分正常约 0.9–1.0，开始导航前要大于 0.8。`GalbotNavigation.init()` 会加载地图。未定位不要下导航。

```python
def navigate_to_goal(
    goal_pose,  # [x, y, z, qx, qy, qz, qw]，地图坐标系
    enable_collision_check: bool = True,
    is_blocking: bool = False,
    timeout: SupportsFloat = 8,
    omni_plan: bool = False,
) -> tuple
def check_goal_arrival() -> bool
def stop_navigation() -> tuple
def is_localized() -> bool
def get_current_pose() -> list[float]
```

`galbot_navigation/navigate_to_goal.py` 先切底盘位姿控制器，再非阻塞导航，自己轮询：

```python
goal = np.array([0.5, 0.0, 0.0, 0, 0, 0.0, 1.0])
robot.switch_controller(G1ControllerName.CHASSIS_POSE_CTRL)
nav.navigate_to_goal(goal, enable_collision_check=True, is_blocking=False, timeout=20)
while True:
    if nav.check_goal_arrival():
        break
nav.stop_navigation()
```

`example3_robot_navigation.py` 和 `example6` 用阻塞形式：`navigate_to_goal(goal_pose, enable_collision_check=True, is_blocking=True, timeout=30)`。短距离、不避障用 `move_straight_to`，目标在里程计系，默认超时 8 秒。

---

## 13. 感知

识别不做也可以不接。例程 `galbot_perception/foundation_stereo_example.py` 要先等模型加载：

```python
perception = GalbotPerception()
perception.init({PerceptionModule.FOUNDATION_STEREO, PerceptionModule.LIGHT_STEREO})
time.sleep(12)
perception.run_once(PerceptionModule.FOUNDATION_STEREO)
perception.wait_for_new_result(PerceptionModule.FOUNDATION_STEREO, timeout_s=6.0)
ok, result = perception.get_latest_result(PerceptionModule.FOUNDATION_STEREO)
```

---

## 14. 全部接口

下面每条都来自 Python API 参考里的函数签名。有例程的在条目下写出文件。类型和枚举的字段表在同一份 HTML 的 “类型与枚举” 一节，这里不展开每个字段。

签名里的 `&gt;` 已还原成 `>`。

## 核心机器人控制（GalbotRobot）


Galbot 人形机器人的主机器人控制接口。 该类提供用于控制 Galbot 机器人的单例接口，支持关节位置与轨迹控制、末端执行器控制、移动底盘速度控制、传感器数据采集、坐标系变换和系统生命周期管理。使用 GalbotRobot::get_instance(MachineType) 获取指定平台（G1/S1）的引用。除非另有说明，角度单位为弧度，线性距离单位为米，时间戳单位为纳秒。
### 获取控制器权限（acquire_controller）

```python
def acquire_controller(controller_name: str) -> ControlStatus
```

获取硬件控制权限。 请求指定控制器接管硬件控制。该操作与 switch_controller 使用同一套 WBCS 切换请求机制。
### 检查轨迹执行状态（check_trajectory_execution_status）

```python
def check_trajectory_execution_status( joint_groups: Sequence[str] = [] ) -> list[TrajectoryControlStatus]
```

获取指定关节组的轨迹执行状态。 查询指定关节组的轨迹当前执行状态。这对于在非阻塞执行模式下监视轨迹进度很有用。
### 清除末端执行器命令（clear_end_effector_command）

```python
def clear_end_effector_command(hold: bool = False) -> ControlStatus
```

清除 WBC 末端执行器任务命令。 清除已发布到 WBC 通道的末端执行器任务轨迹命令。

例程：`galbot_robot/set_end_effector_commands.py`
### 销毁（destroy）

```python
def destroy() -> None
```

清理系统资源。 执行机器人控制系统资源的最终清理，包括中间件连接、传感器接口和通信通道。这是关闭序列的最后一步：request_shutdown() -> wait_for_shutdown() -> destroy()。 此方法应在程序结束时调用一次。调用 destroy() 后，SDK 进入终态，无法在同一进程重新初始化。如需再次使用，请退出当前进程并启动新进程。
### 紧急停止（emergency_stop）

```python
def emergency_stop() -> ControlStatus
```

机器人紧急停止。 通过软件接口立即触发机器人的安全停止机制，中断所有正在进行的运动控制命令和轨迹规划任务，强制机器人进入安全停止状态。此操作立即生效，其优先级高于所有其他控制命令。

例程：`galbot_robot/emergency_stop_and_resume_example.py`
### 执行关节轨迹（execute_joint_trajectory）

```python
def execute_joint_trajectory(trajectory: Trajectory, is_blocking: bool = True) -> ControlStatus
```

执行预先规划的关节轨迹。 执行由路径点组成的轨迹，每个路径点都关联着关节位置、速度和时间信息。轨迹控制器在路径点之间进行插值，以生成平滑的运动。 对于标准关节（头部、腿部、手臂），当前版本中只有位置有效；速度、加速度和力度目前被忽略。

例程：`galbot_robot/execute_joint_trajectory.py；galbot_robot/stop_trajectory_execution.py`
### 获取当前控制器（get_active_controller）

```python
def get_active_controller(group_name: str) -> str
```

获取指定关节组的活动控制器名称。 返回当前 SDK 实例记录的该关节组控制器。该值会以模型默认控制器初始化，并在 SDK 控制器管理接口成功调用后更新。若控制已释放或该关节组无已知控制器，则返回 "CONTROLLER_NAME_NUM"。
### 获取底盘速度（get_base_velocity）

```python
def get_base_velocity() -> dict
```

获取当前移动底盘速度。 返回包含线速度和角速度的结构化底盘速度信息。

例程：`galbot_robot/get_set_base_velocity.py`
### 获取电池管理信息（get_bms_information）

```python
def get_bms_information() -> dict
```

获取BMS（电池管理系统）信息。
### 获取相机内参（get_camera_intrinsic）

```python
def get_camera_intrinsic(camera_id: SensorType) -> dict
```

获取相机内参。 获取指定相机的内参，包括焦距、主点、畸变系数等。
### 获取配置（get_config）

```python
def get_config(service: ConfigService, keys: Sequence[str], use_default: bool = False) -> tuple
```

读取当前配置值或机器人默认配置值。 默认读取用户当前配置的值，而非运行中服务的内存状态。use_default 为 true 时，仅读取机器人内置默认配置。keys 为空时，读取适用于当前机器人的所有已注册字段。
### 获取深度图像数据（get_depth_data）

```python
def get_depth_data(camera_id: SensorType) -> dict
```

获取指定手臂深度相机的最新深度图像数据。 从手臂安装的 RGB-D 深度流中检索最新深度帧。仅 SensorType.LEFT_ARM_DEPTH_CAMERA 和 SensorType.RIGHT_ARM_DEPTH_CAMERA 可用于此方法。

例程：`galbot_robot/get_camera_data.py；tutorials/example4_sensor_data_collect.py`
### 获取设备信息（get_device_information）

```python
def get_device_information() -> dict
```

获取设备信息。 检索基本设备信息，包括设备型号、序列号、固件版本、硬件版本和制造商。此信息用于设备管理、版本控制、系统诊断和设备识别。
### 获取灵巧手状态（get_dexhand_state）

```python
def get_dexhand_state(end_effector: str, dexhand_type: DexHandType = ...) -> Any
```

获取当前灵巧手状态。 检索灵巧手反馈到 dexhand_state。对于 INSPIRE 和 BRAINCO，仅填充 dexhand_state.joint_state，force_sensor_map 为空。对于 SHARPA，填充完整关节状态以及可用的命名力传感器数据。
### 获取力传感器数据（get_force_sensor_data）

```python
def get_force_sensor_data( sensor_type: GalbotOneFoxtrotSensor, calibrated: bool = False, ref_frame: str = '' ) -> dict
```

获取力/力矩传感器数据。 从指定的力/力矩传感器获取最新的原始测量值或标定后的测量值。力和力矩分量可以按需旋转到受支持参考坐标系的坐标轴方向，同时保持传感器测量原点不变。
### 获取坐标系名称（get_frame_names）

```python
def get_frame_names() -> list[str]
```

获取 TF 树中所有可用的坐标系名称。
### 获取夹爪状态（get_gripper_state）

```python
def get_gripper_state(end_effector: str) -> GripperState
```

获取指定夹爪的当前状态。 返回夹爪的位置、速度、受力以及运动状态估计信息。 GripperState.is_moving 基于时间窗口判定：若在内部窗口内未检测到有效开合变化，则该值会变为 false。
### 获取IMU数据（get_imu_data）

```python
def get_imu_data(sensor_id: SensorType) -> dict
```

获取 IMU（惯性测量单元）传感器数据。 检索最新的 IMU 测量值，包括线性加速度、角速度和方向估计。
### 获取红外图像数据（get_ir_data）

```python
def get_ir_data(camera_id: SensorType) -> dict
```

从指定的红外相机获取最新的红外图像。 检索指定红外相机捕获的最新红外图像。仅当相机参数配置中 ir_enabled 为 true 时才可用。
### 获取关节组名称（get_joint_group_names）

```python
def get_joint_group_names() -> list[str]
```

获取机器人可用的关节组名称。 检索机器人运动学配置中定义的所有关节组名称。这对于在运行时发现可用的控制组很有用。
### 获取关节名称（get_joint_names）

```python
def get_joint_names(only_active_joint: bool = True, joint_groups: Sequence[str] = []) -> list[str]
```

通过组名获取机器人关节名称。 检索属于指定关节组的关节名称。这对于在设置关节位置时确定正确的顺序很有用。
### 获取关节位置（get_joint_positions）

```python
def get_joint_positions( joint_groups: Sequence[str] = [], joint_names: Sequence[str] = [] ) -> list[float]
```

获取关节位置
### 获取关节状态（get_joint_states）

```python
def get_joint_states( joint_groups: Sequence[str] = [], joint_names: Sequence[str] = [] ) -> list[JointState]
```

获取关节状态
### 获取激光雷达数据（get_lidar_data）

```python
def get_lidar_data(sensor_id: SensorType) -> dict
```

获取最新的激光雷达点云数据。 检索指定 LiDAR 传感器捕获的最新 3D 点云。每个点通常包含 (x, y, z) 坐标和可选的强度值。
### 获取日志信息（get_log_information）

```python
def get_log_information(timewindow_s: SupportsInt, log_level: LogLevel) -> dict
```

获取日志信息。
### 获取里程计数据（get_odom）

```python
def get_odom() -> dict
```

获取机器人里程信息。 从里程计系统检索机器人当前的姿态和速度估计。里程计通常融合车轮编码器、IMU 和其他本体感觉传感器。
### 获取RGB图像数据（get_rgb_data）

```python
def get_rgb_data(camera_id: SensorType, format: RgbOutputFormat = ..., once: bool = True) -> dict
```

从指定相机获取一张指定格式的RGB相机图像。 获取一帧 DMA FD 相机图像，并按请求的格式返回由 CPU 持有的数据。JPEG 格式使用 Jetson 硬件编码器；NV12/BGR/RGB 格式返回紧凑排列的 CPU 字节数据，布局信息保存在 RgbData 中。

例程：`galbot_robot/get_camera_data.py；tutorials/example4_sensor_data_collect.py；tutorials/example6_pick_and_place.py`
### 获取传感器外参（get_sensor_extrinsic）

```python
def get_sensor_extrinsic(sensor_id: SensorType, reference_frame: str = 'base_link') -> tuple
```

获取传感器外参。 检索指定传感器的外部参数，包括相对于机器人基础坐标系的旋转和平移向量。
### 获取吸盘状态（get_suction_cup_state）

```python
def get_suction_cup_state(end_effector: str) -> SuctionCupState
```

获取当前吸盘状态。 检索指定吸盘的当前状态，包括激活状态和真空压力测量。
### 获取同步观测数据（get_synced_observation）

```python
def get_synced_observation( cameras: Sequence[SensorType], with_joint_state: bool = True ) -> SyncedObservation
```

获取按相机时间戳对齐的同步观测数据。 使用 cameras 中的第一个传感器作为锚点。锚点相机获取最新帧，其它相机则从内部缓存中按时间戳进行最近邻匹配。如果 with_joint_state 为 true，则基于同一个锚点时间戳返回最近邻关节状态。在同步模式下，RGB 条目使用 CPU 持有的 NV12 数据，这样所有相机的采集时间戳都可以保持对齐，而不需要等待串行 JPEG 编码。如果请求的某个相机历史缓存初始为空，该调用最多等待一秒。返回的 joint_state->

例程：`galbot_robot/get_synced_observation.py；tutorials/example5_execute_vla.py`
### 获取坐标变换（get_transform）

```python
def get_transform( target_frame: str, source_frame: str, timestamp_ns: SupportsInt = 0, timeout_ms: SupportsInt = 100 ) -> tuple
```

查询坐标系变换（TF） 查询机器人TF树中两个坐标系之间的变换。 这用于在不同参考坐标系之间转换姿态和位置（例如，从相机坐标系到基坐标系，从末端执行器到世界坐标系）。

例程：`galbot_robot/get_transform.py；galbot_robot/set_leg_height_example.py`
### 获取超声波数据（get_ultrasonic_data）

```python
def get_ultrasonic_data(ultrasonic_type: UltrasonicType) -> dict
```

从指定的超声波传感器获取距离测量值。 从超声波距离传感器之一检索最新的距离测量值。机器人通常在其周边布置有多个超声波传感器。
### 获取音量（get_volume）

```python
def get_volume() -> float
```

获取当前系统全局音量值。
### 获取 WBC 末端执行器位姿（get_wbc_end_effector_poses）

```python
def get_wbc_end_effector_poses() -> dict[str, list[float]]
```

获取 WBC 末端执行器位姿。 返回 WBC 各末端执行器（左臂、右臂、头部）的当前位姿。

例程：`galbot_robot/set_end_effector_commands.py`
### 初始化（init）

```python
def init(enable_sensor_set: Set[SensorType] = ..., enable_sync_mode: bool = False) -> bool
```

初始化机器人控制系统。 初始化机器人硬件通信、中间件和传感器接口。为了优化资源使用，只有在 enable_sensor_set 中指定的传感器才会被初始化并可用于数据读取。 此方法应在程序启动时调用一次。在未调用 destroy() 的情况下多次调用不会报错，但只有第一次调用生效。
### 检查是否运行中（is_running）

```python
def is_running() -> bool
```

检查机器人控制系统是否正在运行。 查询机器人控制系统是否仍处于活动状态，或者是否已收到关闭信号（例如，SIGINT、SIGTERM）。
### 发布目标（publish_target）

```python
def publish_target(target: SingoriXTarget) -> ControlStatus
```

通过 WBC 发布通道发布原始 SingoriXTarget。
### 释放控制器权限（release_controller）

```python
def release_controller(group_name: str = 'all') -> ControlStatus
```

释放硬件权限。 控制硬件，释放关节。与 acquire_controller 相反。如果运行则隐式停止执行。
### 重新加载控制器（reload_controller）

```python
def reload_controller(group_name: str = 'all') -> ControlStatus
```

重新加载控制器。 重新初始化控制器。相当于一个完整的重启循环：停止->重置->启动。对于错误恢复或应用配置更改很有用。
### 请求关机（request_shutdown）

```python
def request_shutdown() -> None
```

请求系统关闭。 发送关闭信号以启动系统正常关闭。这会触发已注册的退出回调并开始资源清理。作为关闭序列的第一步调用：request_shutdown() -> wait_for_shutdown() -> destroy()。
### 请求目标（request_target）

```python
def request_target(target: SingoriXTarget) -> ErrorInfo
```

通过 WBC 服务通道请求执行原始 SingoriXTarget。
### 从紧急停止状态恢复（resume_from_emergency_stop）

```python
def resume_from_emergency_stop() -> ControlStatus
```

机器人紧急停止恢复。 在确认安全条件满足后，解除机器人的软件紧急停止状态，将机器人从安全停止状态恢复为可控状态，并重新启用所有控制相关功能。在恢复前，用户必须确保所有安全条件均已满足。

例程：`galbot_robot/emergency_stop_and_resume_example.py`
### 设置底盘姿态（set_base_pose）

```python
def set_base_pose( base_pose: Pose, is_blocking: bool = True, timeout_s: SupportsFloat = 15.0 ) -> ControlStatus
```

设置移动基础位姿命令。 命令机器人的移动底座移动到其参考坐标系中的指定姿态。这使用底盘姿态控制器（CHASSIS_POSE_CTRL）。当完整的 3D 姿态（位置 + 四元数方向）已经可用时，使用此重载。
### 设置底盘姿态（set_base_pose）

```python
def set_base_pose( x: SupportsFloat, y: SupportsFloat, yaw: SupportsFloat, frame_id: str = 'rel(0)', reference_frame_id: str = 'odom', is_blocking: bool = True, timeout_s: SupportsFloat = 15.0 ) -> ControlStatus
```

使用可选坐标系设置移动基础姿态（x、y、航向角）。 将此重载用于由所选坐标系中的 x/y/航向角定义的平面 2D 目标命令。
### 设置底盘姿态（set_base_pose）

```python
def set_base_pose( x: SupportsFloat, y: SupportsFloat, yaw: SupportsFloat, frame_id: str, reference_frame_id: str, time_from_start_s: SupportsFloat, is_blocking: bool = True, timeout_s: SupportsFloat = 15.0 ) -> ControlStatus
```

使用明确的插值时间设置移动基础姿态（x、y、航向角）。 当必须通过 time_from_start_s 协调到达时间时，请使用此重载。
### 设置底盘速度（set_base_velocity）

```python
def set_base_velocity( linear_velocity: list[float], angular_velocity: list[float], duration_s: SupportsFloat = 0.0 ) -> ControlStatus
```

设置移动基础速度命令。 命令机器人的移动底座以指定的线速度和角速度移动。速度在机器人的基坐标系中表示。

例程：`galbot_robot/get_set_base_velocity.py`
### 设置配置（set_config）

```python
def set_config(service: ConfigService, fields: Sequence[ConfigItem]) -> ControlStatus
```

为某个服务设置一个或多个配置字段。 若 fields 中任一字段无效，则所有字段均不生效，此调用将返回 ControlStatus::INVALID_INPUT。 此调用仅返回单一的聚合状态，不会指出具体哪个（些）字段失败或失败原因。若调用未返回 SUCCESS，请检查日志输出。
### 设置灵巧手命令（set_dexhand_command）

```python
def set_dexhand_command( end_effector: str, dexhand_command: Sequence[JointCommand], dexhand_type: DexHandType = ..., is_blocking: bool = True ) -> ControlStatus
```

通过关节命令控制灵巧手。 使用关节命令向量控制灵巧手（位置、速度、力矩等）。
### 设置末端执行器命令（set_end_effector_command）

```python
def set_end_effector_command( poses: Sequence[list[float]], end_effector_frames: Sequence[str], reference_frames: Sequence[str] = [] ) -> ControlStatus
```

设置用于高频实时控制的 WBC 末端执行器位姿命令（任务轨迹发布）。 poses 中每一行为 [x, y, z, qx, qy, qz, qw]（单位：米，四元数顺序 xyzw）。poses 与 end_effector_frames 的长度必须一致。

例程：`galbot_robot/set_end_effector_commands.py`
### 设置夹爪命令（set_gripper_command）

```python
def set_gripper_command( end_effector: str, width_m: SupportsFloat, velocity_mps: SupportsFloat = 0.03, effort: SupportsFloat = 5, is_blocking: bool = True ) -> ControlStatus
```

控制夹具张开宽度和力度。 命令夹具以受控的速度和最大夹持力移动到指定的开口宽度。

例程：`galbot_robot/set_gripper_command.py；tutorials/example6_pick_and_place.py`
### 设置关节命令（set_joint_commands）

```python
def set_joint_commands( joint_commands: Sequence[JointCommand], joint_groups: Sequence[str] = [], joint_names: Sequence[str] = [], time_from_start_s: SupportsFloat = 0.0 ) -> ControlStatus
```

设置用于高频流控制的底层关节指令。 适用于高频指令流（例如，逐帧模型推理输出）。 此 API 不会从当前/起始位置插值到第一个目标位置。控制器会尽可能快地驱动关节朝向每个指令目标位置。 对于标准关节（头部、腿部、手臂），当前版本中只有 JointCommand::position 有效；速度、加速度和力目前被忽略。 对于夹爪关节，位置字段表示夹爪宽度，速度和力字段均受支持且有效。

例程：`galbot_robot/set_joint_commands_example.py；tutorials/example5_execute_vla.py`
### 批量设置关节命令（set_joint_commands_batch）

```python
def set_joint_commands_batch(trajectory: Trajectory) -> ControlStatus
```

以批处理模式设置关节命令（非阻塞） 实时控制模式下设置多个关节命令轨迹点，支持一次性提交多个时间点的轨迹控制命令。 提供非阻塞高频轨迹执行接口。与set_joint_commands类似，但支持批量轨迹控制，适用于VLA推理批量输出等场景。

例程：`galbot_robot/set_joint_commands_example.py；tutorials/example5_execute_vla.py`
### 设置关节位置（set_joint_positions）

```python
def set_joint_positions( joint_positions: list[float], joint_groups: Sequence[str] = [], joint_names: Sequence[str] = [], is_blocking: bool = True, speed_rad_s: SupportsFloat = 0.2, timeout_s: SupportsFloat = 15.0 ) -> ControlStatus
```

按名称设置指定关节组的目标关节位置（用于低频关键帧/姿态转换） 命令机器人将指定关节移动到目标位置。 该运动以具有可配置速度限制的平滑轨迹执行。

例程：`galbot_robot/set_joint_positions.py；tutorials/example1_galbot_control_started.py；tutorials/example6_pick_and_place.py`
### 设置腿部高度（set_leg_height）

```python
def set_leg_height( link_height: SupportsFloat, duration: SupportsFloat = 4.0, is_blocking: bool = False, timeout_s: SupportsFloat = -1.0 ) -> ControlStatus
```

设置 G1 或 G3 机器人腿部高度。 将腿部关节组切换至腿部高度控制器，并异步下发 head_base_link 在 base_link 下的平滑运动目标。支持的目标范围为 [0.96, 1.54] m，支持的时长范围为 [0, 60] s。时长为 0 表示请求最快运动，使用时需谨慎。该方法默认在下发目标后即返回；在阻塞模式下，会等待对应的 leg_height 任务报告 target_finished，同时监测运行时错误和腿部传感器可用性。

例程：`galbot_robot/set_leg_height_example.py`
### 设置吸盘命令（set_suction_cup_command）

```python
def set_suction_cup_command(end_effector: str, activate: bool) -> ControlStatus
```

控制吸盘激活状态。 激活或停用指定的吸盘末端执行器。
### 设置音量（set_volume）

```python
def set_volume(volume: SupportsFloat) -> bool
```

设置系统全局音量值。

例程：`galbot_robot/audio_example.py；galbot_robot/installation_welcome_verify.py`
### 启动控制器（start_controller）

```python
def start_controller(group_name: str = 'all') -> ControlStatus
```

开始控制器执行。 激活控制器以开始发送命令。与 stop_controller 相反。需要事先获得硬件权限（获取）。
### 启动麦克风流输入（start_microphone_stream_input）

```python
def start_microphone_stream_input( callback: Callable, chunk_size: SupportsInt = 2560, use_raw_audio: bool = False ) -> str
```

启动麦克风流音频输入。

例程：`galbot_robot/audio_example.py`
### 停止音频流输出（stop_audio_stream_output）

```python
def stop_audio_stream_output(stream_id: str = '') -> None
```

停止指定音频输出流或所有活动音频输出流的播放。
### 紧急停止底盘（stop_base）

```python
def stop_base() -> ControlStatus
```

紧急停止移动底座运动。 立即命令移动基地停止一切运动。这是一项安全功能，当需要立即停止基本运动时应使用。

例程：`galbot_robot/emergency_stop_and_resume_example.py`
### 停止控制器（stop_controller）

```python
def stop_controller(group_name: str = 'all') -> ControlStatus
```

停止控制器执行。 停止命令执行但保留硬件权限。与 start_controller 相反。
### 停止麦克风流输入（stop_microphone_stream_input）

```python
def stop_microphone_stream_input(stream_id: str = '') -> None
```

停止指定的麦克风流式音频输入。
### 停止轨迹执行（stop_trajectory_execution）

```python
def stop_trajectory_execution() -> ControlStatus
```

停止所有当前正在执行的关节轨迹。 立即停止执行所有关节组中的所有活动关节轨迹。停止后关节将保持当前位置。
### 订阅视频流数据（subscribe_video_data）

```python
def subscribe_video_data(camera_id: SensorType, callback: Callable) -> SensorStatus
```

订阅指定相机的 H.264 编码视频帧。 为一个 H.264 视频流注册回调函数。用户回调函数由内部 SDK 调度线程执行，不会阻塞中间件接收回调线程。

例程：`galbot_robot/subscribe_video_data.py`
### 切换控制器（switch_controller）

```python
def switch_controller(controller_name: str) -> ControlStatus
```

切换活动控制器策略。 将硬件控制切换到新的策略。请求会发送到 WBCS 控制器管理器，由其执行安全切换，并在接受请求后启动目标控制器。
### 停止订阅视频流数据（unsubscribe_video_data）

```python
def unsubscribe_video_data(camera_id: SensorType) -> SensorStatus
```

取消订阅指定相机的 H.264 编码视频帧。 清除此前通过 subscribe_video_data 为该相机注册的所有用户回调。内部 SDK 读取器、环形缓冲区和调度线程将继续运行，以避免重复创建和销毁线程及资源。

例程：`galbot_robot/subscribe_video_data.py`
### 等待关机（wait_for_shutdown）

```python
def wait_for_shutdown() -> None
```

阻塞直到关闭完成。 阻塞调用线程直到所有模块都正常关闭完毕。作为关闭序列的第二步调用，在 request_shutdown() 之后、destroy() 之前。
### 写入音频流输出（write_audio_stream_output）

```python
def write_audio_stream_output(audio_chunk: str, stream_id: str = '') -> bool
```

将PCM格式的音频数据块写入音频输出流以进行实时播放。

例程：`galbot_robot/audio_example.py；galbot_robot/installation_welcome_verify.py`
### 全身与底盘归零（zero_whole_body_and_base）

```python
def zero_whole_body_and_base( base_zero_pose: Pose, is_blocking: bool = True, leg_head_speed_rad_s: SupportsFloat = 0.2, leg_head_timeout_s: SupportsFloat = 15.0, params: Parameter = None ) -> tuple[MotionStatus, ControlStatus]
```

一键归零：将全身关节回零，并将底盘位姿回到零位。 该接口会调用 move_whole_body_joint_zero 完成关节归零，并将底盘目标位姿设为零。若 params 为空指针，则使用 default_param。
### 全身与底盘归零（zero_whole_body_and_base）

```python
def zero_whole_body_and_base( frame_id: str = 'odom', reference_frame_id: str = 'odom', is_blocking: bool = True, leg_head_speed_rad_s: SupportsFloat = 0.2, leg_head_timeout_s: SupportsFloat = 15.0, params: Parameter = None ) -> tuple[MotionStatus, ControlStatus]
```

一键归零：通过可选坐标系将全身关节归零，并将底座（x、y、航向角）归零。

## 运动规划与执行（GalbotMotion）


Galbot 机器人的统一运动规划和控制接口。 该接口提供全面的机器人运动控制 API，包括： - 正向和逆向运动学计算 - 单链和多链轨迹规划 - 碰撞检测（自碰撞和环境） - 工具和障碍物管理 - 全身协调运动规划使用 GalbotMotion::get_instance(MachineType) 获取特定平台 (G1/G3/S1) 的实例。所有角度单位为弧度，线性单位为米（SI 标准）。四元数必须归一化：√(x² + y² + z² + w²) = 1。
### 添加障碍物（add_obstacle）

```python
def add_obstacle( obstacle_id: str, obstacle_type: str, pose: list[float], scale: list[float] = [0.0, 0.0, 0.0], key: str = '', target_frame: str = 'world', ee_frame: str = 'ee_base', reference_joint_positions: list[float] = [], reference_base_pose: list[float] = [], ignore_collision_link_names: Sequence[str] = [], safe_margin: SupportsFloat = 0.0, resolution: SupportsFloat = 0.01 ) -> MotionStatus
```

将碰撞对象加载到环境中。 将几何或基于网格的障碍物插入环境中以避免碰撞。障碍可以是静态的（世界坐标系下固定的）或与机器人相关的。支持原始形状、网格、点云和深度图像。

例程：`galbot_motion/add_obstacle.py`
### 附加目标对象（attach_target_object）

```python
def attach_target_object( obstacle_id: str, obstacle_type: str, pose: list[float], scale: list[float] = [0.0, 0.0, 0.0], key: str = '', target_frame: str = 'world', ee_frame: str = 'ee_base', reference_joint_positions: list[float] = [], reference_base_pose: list[float] = [], ignore_collision_link_names: Sequence[str] = [], safe_margin: SupportsFloat = 0.0, resolution: SupportsFloat = 0.01 ) -> MotionStatus
```

将碰撞物体附加到机器人上（例如，抓取的物体）。与 add_obstacle() 类似，但对象随机器人移动（连接到连杆/运动链），用于表示抓取的物体、传感器或有效载荷。 在运动过程中，对象相对于连接坐标系的姿态保持不变。
### 附加工具（attach_tool）

```python
def attach_tool(chain: str, tool: str) -> MotionStatus
```

将工具连接到末端执行器。 将工具（夹具、相机、定制末端执行器）加载到运动链上。更新运动学模型和碰撞几何体以包含该工具。

例程：`galbot_motion/attach_tool.py；tutorials/example6_pick_and_place.py`
### 检查碰撞（check_collision）

```python
def check_collision( start: Sequence[RobotStates], enable_collision_check: bool = True, params: Parameter = ... ) -> tuple[MotionStatus, list[bool]]
```

检查机器人状态是否存在碰撞。 检查自身碰撞和与环境障碍物的碰撞，支持批量处理。因此，如果需要 Motion 考虑环境障碍物（包括点云），必须显式加载障碍物地图/对象（例如，obstacle_type = point_cloud 且在 key 中提供文件路径）。 注意：将实时感知集成到 galbotMotion 是未来计划，目前内部验证有限。 验证给定机器人配置是否无碰撞。
### 检查碰撞详情（check_collision_detail）

```python
def check_collision_detail( robot_states: Sequence[RobotStates], is_check_once: bool = False, is_log: bool = False, params: Parameter = ... ) -> tuple[MotionStatus, list[CollisionInfo]]
```

检查机器人状态的详细碰撞对。 查询运动学碰撞检测服务，返回当前由 MPS 提供的碰撞对详情。
### 清空障碍物（clear_obstacle）

```python
def clear_obstacle() -> MotionStatus
```

从规划场景中移除所有碰撞障碍物。 清除整个障碍物集，将规划场景重置为空（机器人几何体除外）。
### 组合规划请求（combine_plan）

```python
def combine_plan( plan_reqs: Sequence[PlanRequest], params: PlannerConfig = ..., start_state: RobotStates = None ) -> tuple[MotionStatus, dict[str, list[list[float]]]]
```

将多个异构规划请求组合为单条协调轨迹。 接受一系列 PlanRequest 对象，每个对象指定各自的规划类型（MOTION_PLAN / TRAJ_PLAN / MOVE_LINE）、目标路点以及分段选项。服务器将这些请求组合为一条连续轨迹，各分段之间平滑过渡，且每个分段使用各自的规划算法。 服务器端流程：SDK 发送一个 CombinePlanReq（proto 字段：combine_plan_req），其中包含一系列 SigPlanReq 子请求，每个子请求各自携带 pl

例程：`galbot_motion/combine_plan.py`
### 分离目标对象（detach_target_object）

```python
def detach_target_object(obstacle_id: str) -> MotionStatus
```

从机器人上分离物体（例如，释放后）。从机器人上移除附着的物体。通常在释放抓取的对象后调用。 该对象被完全从规划场景中移除（不转换为静态障碍物）。
### 分离工具（detach_tool）

```python
def detach_tool(chain: str) -> MotionStatus
```

将当前工具与末端执行器分离。 相应地更新运动学模型和碰撞几何体。

例程：`galbot_motion/detach_tool.py；tutorials/example6_pick_and_place.py`
### 正向运动学（forward_kinematics）

```python
def forward_kinematics( target_frame: str, reference_frame: str = 'base_link', joint_state: dict[str, list[float]] = {}, params: Parameter = ... ) -> tuple[MotionStatus, list[float]]
```

计算目标连杆的正向运动学。 计算给定关节配置的指定连杆的笛卡尔姿态。对于确定末端执行器位置、验证配置或计算中间连杆姿态很有用。

例程：`galbot_motion/fk.py；tutorials/example2_arm_manipulation.py`
### 基于状态的正向运动学（forward_kinematics_by_state）

```python
def forward_kinematics_by_state( target_frame: str, reference_robot_states: RobotStates = None, reference_frame: str = 'base_link', params: Parameter = ... ) -> tuple[MotionStatus, list[float]]
```

使用完整的机器人状态计算正向运动学。 与 forward_kinematics() 类似，但接受 RobotStates 对象来指定完整的机器人配置（全身关节+底座姿态）。

例程：`galbot_motion/fk.py`
### 获取已构建障碍物列表（get_built_obstacles_list）

```python
def get_built_obstacles_list() -> list[str]
```

获取当前加载的障碍物 ID 列表。
### 获取运动链关节名称（get_chain_joint_names）

```python
def get_chain_joint_names(chain_name: str) -> list[str]
```

获取指定运动链的关节名称。 返回该运动链在机器人模型中配置的关节名称有序列表。
### 获取链关节状态（get_chain_joint_state）

```python
def get_chain_joint_state() -> dict[str, list[float]]
```

获取所有运动链的当前关节配置。 检索每条链的关节状态，将整体配置分解为各个链的贡献。
### 按类型获取配置（get_config_by_type）

```python
def get_config_by_type(config_type: str) -> tuple[MotionStatus, MotionPlanConfig]
```

按类型获取 MPS 配置（结果通过 MotionPlanConfig.common_str 返回）。
### 获取末端执行器姿态（get_end_effector_pose）

```python
def get_end_effector_pose( end_effector_frame: str, reference_frame: str = 'base_link' ) -> tuple[MotionStatus, list[float]]
```

从机器人状态获取当前末端执行器姿态。 使用机器人实时关节状态，通过 MPS 正向运动学计算当前笛卡尔姿态。要求该连杆已在机器人 URDF 模型中定义。
### 获取指定链末端执行器姿态（get_end_effector_pose_on_chain）

```python
def get_end_effector_pose_on_chain( chain_name: str, frame_id: str = 'EndEffector', reference_frame: str = 'base_link' ) -> tuple[MotionStatus, list[float]]
```

获取特定运动链的当前末端执行器姿态。 通过链名称和坐标系类型检索末端执行器姿态的便捷方法，无需知道 URDF 中的确切连杆名称。

例程：`galbot_motion/get_set_end_effector_pose.py；tutorials/example2_arm_manipulation.py`
### 获取雅可比矩阵（get_jacobian）

```python
def get_jacobian( chain_name: str, target_frame: str = 'EndEffector', reference_frame: str = 'base_link', joint_state: dict[str, list[float]] = {}, params: Parameter = ... ) -> tuple[MotionStatus, list[list[float]]]
```

计算指定运动链的雅可比矩阵
### 通过状态获取雅可比矩阵（get_jacobian_by_state）

```python
def get_jacobian_by_state( chain_name: str, target_frame: str = 'EndEffector', reference_frame: str = 'base_link', reference_robot_states: RobotStates = None, params: Parameter = ... ) -> tuple[MotionStatus, list[list[float]]]
```

通过给定的机器人状态计算指定运动链的雅可比矩阵
### 获取连杆名称（get_link_names）

```python
def get_link_names(only_end_effector: bool = False) -> list[str]
```

从运动学模型中获取机器人连杆名称。 检索机器人 URDF 模型中定义的连杆名称列表，可用于过滤末端执行器连杆或返回所有连杆。
### 获取运动规划配置（get_motion_plan_config）

```python
def get_motion_plan_config() -> tuple[MotionStatus, MotionPlanConfig]
```

获取当前运动规划配置。 检索活动规划器配置，包括速度/加速度限制和规划算法参数。
### 获取机器人状态（get_robot_states）

```python
def get_robot_states() -> RobotStates
```

获取当前完整的机器人状态。 检索当前全身关节配置和移动底座姿态。代表机器人的完整运动状态。
### 获取支持的链（get_supported_chains）

```python
def get_supported_chains() -> set[str]
```

获取支持的运动链名称列表（例如 left_arm、right_arm）。
### 获取支持的末端执行器坐标系（get_supported_ee_frames）

```python
def get_supported_ee_frames() -> set[str]
```

获取支持的末端执行器坐标系标识符集。
### 获取支持的坐标系（get_supported_frames）

```python
def get_supported_frames() -> set[str]
```

获取支持的参考坐标系名称集。
### 获取支持的连杆（get_supported_links）

```python
def get_supported_links() -> set[str]
```

获取支持的连杆名称列表（正向运动学/逆向运动学的 URDF 连杆名称）。
### 获取支持的障碍物类型（get_supported_obstacle_types）

```python
def get_supported_obstacle_types() -> set[str]
```

获取支持的障碍物类型集（例如盒子、球体、圆柱体、网格）。
### 获取支持的工具列表（get_supported_tool_list）

```python
def get_supported_tool_list() -> set[str]
```

获取 attach_tool 支持的工具名称列表。
### 初始化（init）

```python
def init() -> bool
```

初始化运动规划系统和通信接口。 必须在任何其他 API 函数之前调用。初始化内部通信中间件，加载机器人运动学模型，并建立与控制服务的连接。
### 逆向运动学（inverse_kinematics）

```python
def inverse_kinematics( target_pose: list[float], chain_names: Sequence[str], target_frame: str = 'EndEffector', reference_frame: str = 'base_link', initial_joint_positions: dict[str, list[float]] = {}, enable_collision_check: bool = True, params: Parameter = ... ) -> tuple[MotionStatus, dict[str, list[float]]]
```

计算目标笛卡尔姿态的逆运动学。 求解实现指定末端执行器姿态的关节配置。支持单链 IK（仅手臂）或协调多链 IK（手臂 + 躯干/腿）。

例程：`galbot_motion/ik.py；tutorials/example2_arm_manipulation.py`
### 基于状态的逆向运动学（inverse_kinematics_by_state）

```python
def inverse_kinematics_by_state( target_pose: list[float], chain_names: Sequence[str], target_frame: str = 'EndEffector', reference_frame: str = 'base_link', reference_robot_states: RobotStates = None, enable_collision_check: bool = True, params: Parameter = ... ) -> tuple[MotionStatus, dict[str, list[float]]]
```

使用完整的机器人状态作为种子计算逆运动学。 与 inverse_kinematics() 类似，但接受 RobotStates 来指定种子配置，从而允许精确控制整个机器人状态。

例程：`galbot_motion/ik.py；tutorials/example2_arm_manipulation.py`
### 通用逆向运动学（inverse_kinematics_general）

```python
def inverse_kinematics_general( target_waypoint: Sequence[MotionPlanChainTarget], reference_robot_states: RobotStates = None, params: Parameter = ... ) -> tuple[MotionStatus, dict[str, JointStates]]
```

使用通用的多链目标描述计算逆运动学。 采用较新的 MotionPlanTarget 结构（inverse_kinematic_general_req），在单次请求中支持混合链目标与辅助链设置。

例程：`galbot_motion/ik.py；tutorials/example2_arm_manipulation.py`
### 轨迹规划（motion_plan）

```python
def motion_plan( target: RobotStates, start: RobotStates = None, reference_robot_states: RobotStates = None, enable_collision_check: bool = True, params: Parameter = ... ) -> tuple[MotionStatus, dict[str, list[list[float]]]]
```

规划单个运动链的轨迹。

例程：`galbot_motion/motion_plan.py`
### 多路点运动规划（motion_plan）

```python
def motion_plan( waypoints: Sequence[Sequence[MotionPlanChainTarget]], params: PlannerConfig = ..., start_state: RobotStates = None ) -> tuple[MotionStatus, dict[str, list[list[float]]]]
```

通过运动规划服务器规划一条经过多个路点且考虑碰撞的路径。 向服务器发送 MotionPlanReq，服务器调用基于采样的规划器（默认：EITstar/RRT 变体）搜索无碰撞的几何路径，再交由轨迹规划器进行时间参数化，生成速度/加速度/加加速度曲线。 服务器端流程：将路点转换为内部的 RobotStates 与 PlannerConfig。单链时调用 singleChainMotionPlan()；多链时调用 multiChainMotionPlan()。几何路径在执行前总会

例程：`galbot_motion/motion_plan.py；galbot_motion/motion_plan_multi_waypoints.py`
### 多路点轨迹规划（motion_plan_multi_waypoints）

```python
def motion_plan_multi_waypoints( target: RobotStates, waypoint_poses: Sequence[list[float]], start: RobotStates = None, reference_robot_states: RobotStates = None, enable_collision_check: bool = True, params: Parameter = ... ) -> tuple[MotionStatus, dict[str, list[list[float]]]]
```

为单个运动链规划一条经过多个路点的轨迹。

例程：`galbot_motion/motion_plan_multi_waypoints.py`
### 多路点轨迹规划（motion_plan_multi_waypoints）

这一重载的 `targets` 类型在网页源码里被标签拆开，完整容器类型以 API 参考原文为准。例程 `galbot_motion/motion_plan_multi_waypoints.py` 用的是上面那个 `target` + `waypoint_poses` 重载。

通过多个链的路径点规划协调轨迹。 通过路径点序列实现协调的多臂或全身运动。每个链都可以有自己的路点序列，以同步方式执行。

例程：`galbot_motion/motion_plan_multi_waypoints.py`
### 直线运动规划（move_line）

```python
def move_line( waypoints: Sequence[Sequence[MotionPlanChainTarget]], params: PlannerConfig = ..., start_state: RobotStates = None ) -> tuple[MotionStatus, dict[str, list[list[float]]]]
```

生成经过多个路点的笛卡尔直线运动。 向服务器发送 MoveLineReq，服务器在笛卡尔（任务）空间中为末端执行器规划直线路径，再在中间采样点通过 IK 转换为关节空间轨迹。 服务器端流程：单链时调用 singleChainMoveLine()；多链时调用 multiChainMoveLine()。规划器按照 TrajPlanner 中配置的 move_line_intermidiate_point 数量，沿笛卡尔直线采样中间点；在每个采样点求解 IK 以得到对应的关节配置。

例程：`galbot_motion/move_line.py；galbot_motion/move_line_leg.py；tutorials/example6_pick_and_place.py`
### 移动全身关节到零位（move_whole_body_joint_zero）

```python
def move_whole_body_joint_zero( is_blocking: bool = True, leg_head_speed_rad_s: SupportsFloat = 0.2, leg_head_timeout_s: SupportsFloat = 15.0, params: Parameter = ... ) -> MotionStatus
```

将全身关节移动到预定义的零（家庭）配置。 腿部和头部关节通过 GalbotRobot（直接关节控制）进行控制，而左/右臂则通过启用碰撞检查的运动规划器进行规划。 零配置的联合顺序遵循 SDK 约定：leg(5) + head(2) + left_arm(7) + right_arm(7)。
### 移除障碍物（remove_obstacle）

```python
def remove_obstacle(obstacle_id: str) -> MotionStatus
```

从规划场景中移除碰撞障碍物。
### 按类型设置配置（set_config_by_type）

```python
def set_config_by_type(config: MotionPlanConfig) -> MotionStatus
```

按类型设置/获取 MPS 配置（GetSet 通道）。 这是对 MPS Get/Set MotionPlanConfig RPC 的一层简单封装：config.config_type 用于选择目标子系统（例如 "sampler"、"ik_solver"、"trajectory_planner"）。对于 set_config_by_type，请求负载编码在 config.common_str 中（内容因类型而异）；对于 get_config_by_type，响应负载通过 out
### 设置末端执行器姿态（set_end_effector_pose）

```python
def set_end_effector_pose( target_pose: list[float], end_effector_frame: str, reference_frame: str = 'base_link', reference_robot_states: RobotStates = None, enable_collision_check: bool = True, is_blocking: bool = True, timeout: SupportsFloat = -1.0, params: Parameter = ... ) -> MotionStatus
```

命令末端执行器移动到目标笛卡尔坐标位姿。 笛卡尔运动命令的高级接口。内部执行逆运动学 (IK)、轨迹规划并执行运动。is_blocking 标志仅控制此 API 是等待完成还是在启动后台执行任务后立即返回。

例程：`galbot_motion/get_set_end_effector_pose.py；tutorials/example2_arm_manipulation.py；tutorials/example6_pick_and_place.py`
### 设置末端执行器姿态（set_end_effector_pose）

```python
def set_end_effector_pose( target_pose: list[float], end_effector_frame: str, reference_frame: str, assist_chains: Set[str], reference_robot_states: RobotStates = None, enable_collision_check: bool = True, is_blocking: bool = True, timeout: SupportsFloat = -1.0, params: Parameter = ... ) -> MotionStatus
```

命令末端执行器在辅助链的配合下移动到目标笛卡尔姿态。 这是一个为方便使用而提供的重载成员函数，与上述函数的区别仅在于所接受的参数。 与 set_end_effector_pose(target_pose, end_effector_frame, reference_frame, ...) 相同，但 assist_chains 参数位于 reference_frame 之后，用于协调多链规划。

例程：`galbot_motion/get_set_end_effector_pose.py；tutorials/example2_arm_manipulation.py；tutorials/example6_pick_and_place.py`
### 设置运动规划配置（set_motion_plan_config）

```python
def set_motion_plan_config(config: MotionPlanConfig) -> MotionStatus
```

设置全局运动规划配置。 更新规划器设置，例如速度/加速度限制、规划算法参数和优化目标。影响后续所有的计划操作。
### 状态转字符串（status_to_string）

```python
def status_to_string(status: MotionStatus) -> str
```

将 MotionStatus 枚举转换为人类可读的字符串。 将状态代码映射到用于日志记录、错误报告或 UI 显示的描述性字符串。
### 轨迹规划（traj_plan）

```python
def traj_plan( waypoints: Sequence[Sequence[MotionPlanChainTarget]], params: PlannerConfig = ..., start_state: RobotStates = None ) -> tuple[MotionStatus, dict[str, list[list[float]]]]
```

在不进行路径搜索的情况下，生成经过多个路点的时间参数化关节轨迹。 向服务器发送 TrajPlanReq（mps_core.cpp:handle_sig_point_traj_plan_req 或 handle_mul_point_traj_plan_req），执行带时间参数化的直接关节空间插值。与 motion_plan() 不同，该方法不运行基于采样的路径搜索，而是直接在关节空间中连接各路点。 服务器端流程：单点目标调用 moveJointPlan() 进行起点到目标的直接

例程：`galbot_motion/traj_plan.py`

## 移动导航（GalbotNavigation）


移动机器人底盘导航与定位接口。 该类提供线程安全的单例接口，用于控制移动底盘导航系统。支持二维位姿估计、重定位、带动态避障的目标导航与路径规划。导航系统在全局地图坐标系下工作，支持阻塞与非阻塞两种导航模式，并兼容差分驱动和全向底盘。除非另有说明，所有位姿参数均以地图坐标系表示。
### 添加导航箱子（add_bounding_box）

```python
def add_bounding_box(box_info: dict) -> tuple
```

添加用于导航障碍物过滤的箱子。 将 SDK 定义的箱子区域发送给融合服务，使导航在处理融合障碍物点时忽略这些区域，而不是把它们当作障碍物。box_tag 是箱子的唯一标记；后续调用 remove_bounding_box() 移除该箱子时，应传入相同的 box_tag。箱子位姿相对于 parent_link_name 表示。
### 挂载箱子到连杆（attach_box_to_link）

```python
def attach_box_to_link(box_info: dict, ignore_collision_links: Sequence[str] = []) -> tuple
```

将箱子碰撞物挂载到机器人连杆。 该接口会把箱子作为附着碰撞物发送给 PNS。box_tag 是箱子的唯一标记；后续调用 detach_box_from_link() 解除该箱子挂载时，应传入相同的 box_tag。box_info.parent_link_name 作为 box_info.box_pose 的父连杆。
### 检查目标到达（check_goal_arrival）

```python
def check_goal_arrival() -> bool
```

检查机器人是否成功达到当前目标。 该方法查询导航系统以确定机器人是否已在可接受的位置和方向公差内到达目标位姿。当使用非阻塞导航模式轮询完成情况时，这特别有用。

例程：`galbot_navigation/navigate_to_goal.py`
### 检查路径可达性（check_path_reachability）

```python
def check_path_reachability(goal_pose: numpy.ArrayLike, start_pose: numpy.ArrayLike) -> bool
```

检查地图中是否存在从起点到目标的无碰撞路径。 此方法查询全局路径规划器以确定指定的起始姿态和目标姿态之间是否存在有效的无碰撞路径。这对于在尝试导航之前验证目标姿态或多目标路径规划非常有用。
### 从连杆解除箱子（detach_box_from_link）

```python
def detach_box_from_link(box_tag: SupportsInt) -> tuple
```

从机器人连杆解除箱子碰撞物。 根据 box_tag 解除对应箱子的附着碰撞物。这里的 box_tag 应与 attach_box_to_link() 挂载该箱子时使用的 box_tag 保持一致。
### 转储导航动态配置以进行调试（dump_navigation_configs）

```python
def dump_navigation_configs() -> tuple
```

转储导航动态配置以进行调试。 此方法查询常用的导航配置键，并通过 SDK 日志记录器打印响应。它用于诊断和调试。
### 获取导航箱子（get_bounding_box）

```python
def get_bounding_box() -> list
```

获取当前用于导航障碍物过滤的箱子。 查询融合服务，返回当前的箱子信息列表。box_tag 是箱子的唯一标记。
### 获取当前姿态（get_current_pose）

```python
def get_current_pose() -> list[float]
```

获取地图坐标系中机器人底盘的当前估计位姿。 该方法返回来自定位系统的最新姿态估计。位姿表示机器人的 base_link 坐标系相对于地图坐标系原点的位置和方向。
### 获取任务状态（get_navigation_status）

```python
def get_navigation_status() -> NavigationTaskStatus
```

获取当前导航任务状态。 该接口适合在非阻塞模式下轮询任务进度，并根据状态判断任务是否完成。
### 查询任务状态（get_navigation_target_status）

```python
def get_navigation_target_status(task_id: str) -> NavigationTaskSnapshot
```

查询异步导航任务的最新状态。 该接口用于获取已提交任务的当前执行结果，适合在非阻塞模式下持续监控任务进度。
### 初始化（init）

```python
def init() -> bool
```

初始化导航子系统及其依赖项。 在使用任何其他导航功能之前必须调用此方法。它初始化通信通道、加载地图、启动定位模块并准备路径规划器。
### 检查是否已定位（is_localized）

```python
def is_localized() -> bool
```

检查机器人当前是否在地图中定位。 该方法查询定位系统以确定机器人是否具有足够置信度的有效姿态估计。未定位的机器人不应执行导航任务。
### 直线移动到目标（move_straight_to）

```python
def move_straight_to( goal_pose: numpy.ArrayLike, is_blocking: bool = True, timeout: SupportsFloat = 8 ) -> tuple
```

将机器人移动到里程计坐标系中的相对目标位姿。 此方法命令机器人移动到相对于其在里程计 (odom) 坐标系中的当前位置指定的姿态。这对于不需要基于地图的规划的短距离、精确的运动非常有用。与navigate_to_goal()不同，此方法不执行动态障碍物检测或全局路径规划。它使用全向运动规划来直接移动到目标。
### 轨迹导航（navigate_along_trajectory）

```python
def navigate_along_trajectory( waypoints: Sequence[Pose], frame_id: str = 'map', speed_ratio: SupportsFloat = 1.0, enable_collision_check: bool = True ) -> TaskHandle
```

使用有序三维位姿提交轨迹导航任务。 该接口将输入位姿视为轨迹参考，导航系统可以对路径进行平滑和优化，因此中间位姿不保证被精确经过，但最终位姿会作为导航目标。
### 多路点导航（navigate_through_waypoints）

```python
def navigate_through_waypoints( waypoints: Sequence[Waypoint], frame_id: str = 'map', enable_collision_check: bool = True ) -> TaskHandle
```

提交多路点导航任务。 该接口一次请求下发多个路点，并按照输入顺序逐点执行，适合需要明确经过多个目标位置的导航场景。
### 导航到目标（navigate_to_goal）

```python
def navigate_to_goal( goal_pose: numpy.ArrayLike, enable_collision_check: bool = True, is_blocking: bool = False, timeout: SupportsFloat = 8, omni_plan: bool = False ) -> tuple
```

将机器人导航至地图坐标系中的目标位姿。 该方法命令移动底座使用全局路径规划器和局部轨迹控制器导航到指定的目标位姿。如果启用了碰撞检查，规划器将计算从当前姿态到目标的无碰撞路径，同时考虑静态地图障碍物和动态障碍物。

例程：`galbot_navigation/navigate_to_goal.py；tutorials/example3_robot_navigation.py；tutorials/example6_pick_and_place.py`
### 导航到目标V2（navigate_to_goal_v2）

```python
def navigate_to_goal_v2( goal_pose: numpy.ArrayLike, max_vel: numpy.ArrayLike, pose_frame: str = 'map', enable_collision_check: bool = True, is_blocking: bool = False, timeout: SupportsFloat = 5.0, omni_plan: bool = False ) -> tuple
```

使用导航 v2 接口引导机器人到达目标姿态。 此方法向 PNS 服务发送导航 v2 规划请求。它支持通过 pose_frame 设置全局和局部目标坐标系，并在发送目标请求之前应用导航速度和运行时超时配置。

例程：`galbot_navigation/navigate_to_goal.py；tutorials/example3_robot_navigation.py；tutorials/example6_pick_and_place.py`
### 速度指令导航机器人（navigate_with_velocity）

```python
def navigate_with_velocity( vx: SupportsFloat, vy: SupportsFloat, vyaw: SupportsFloat, duration_s: SupportsFloat = 3.0, enable_collision_check: bool = True ) -> tuple
```

使用导航 v2 接口，通过速度指令导航机器人。 此方法发送基于速度的导航指令。该指令包含平面基座速度分量和持续时间；执行持续时间由导航服务处理。
### 重定位（relocalize）

```python
def relocalize(init_pose: numpy.ArrayLike) -> tuple
```

执行重新定位以重新估计机器人在地图坐标系中的姿态。 该方法重置定位滤波器并提供初始姿态估计，以帮助机器人在已知地图中重新建立其位置。当机器人失去定位或手动将机器人放置在已知位置时，这非常有用。
### 移除导航箱子（remove_bounding_box）

```python
def remove_bounding_box(box_tag: SupportsInt) -> tuple
```

移除用于导航障碍物过滤的箱子。 根据 box_tag 移除对应箱子。这里的 box_tag 应与 add_bounding_box() 添加该箱子时使用的 box_tag 保持一致。
### 设置导航到达阈值（set_navigation_arrival_threshold）

```python
def set_navigation_arrival_threshold(threshold: numpy.ArrayLike) -> tuple
```

设置导航到达阈值。 此方法更新导航规划和控制使用的位置和偏航容差，以确定是否已到达目标。
### 设置导航运动学限制（set_navigation_kinematics_limits）

```python
def set_navigation_kinematics_limits( vel_limit: numpy.ArrayLike, acc_limit: numpy.ArrayLike, jerk_limit: numpy.ArrayLike ) -> tuple
```

设置导航运动学限制。 此方法通过 PNS 动态配置界面更新导航速度、加速度和加加速度限制。
### 设置动态目标（set_navigation_target）

```python
def set_navigation_target( target: Pose, frame: str = 'map', speed_ratio: SupportsFloat = 1.0, enable_collision_check: bool = True ) -> TaskHandle
```

异步提交动态导航目标。 该接口适用于目标可能频繁变化的场景，例如动态跟踪或遥操作。提交新目标后，导航系统会以最新目标为准重新规划； 前一个目标可能被后续请求抢占。为保证导航过程稳定，建议不要以过高频率调用。
### 设置导航超时时间（set_navigation_timeout）

```python
def set_navigation_timeout(timeout_s: SupportsFloat) -> tuple
```

设置导航超时配置。 此方法通过 PNS 动态配置接口更新导航超时。它可作为不应无限期运行的任务的安全保障。
### 设置导航速度限制（set_navigation_velocity_limit）

```python
def set_navigation_velocity_limit(vel_limit: numpy.ArrayLike) -> tuple
```

设置导航速度限制。 此方法通过 PNS 动态配置接口更新导航速度限制。
### 停止导航（stop_navigation）

```python
def stop_navigation() -> tuple
```

停止当前的导航任务并使机器人停止。 此方法立即取消任何正在进行的导航命令（来自navigate_to_goal()或move_straight_to()）并命令机器人停止。机器人将根据其运动学约束减速并安全停止。

例程：`galbot_navigation/stop_navigation.py；galbot_navigation/navigate_to_goal.py`

## 感知模块接口（GalbotPerception）


感知模块接口；通过 get_instance(MachineType) 获取单例实例。
### 获取模块的最新缓存结果（get_latest_result）

```python
def get_latest_result(module: PerceptionModule) -> tuple
```

返回模块的最新缓存结果，不阻塞。

例程：`galbot_perception/foundation_stereo_example.py`
### 初始化感知模块并加载模型（init）

```python
def init(enabled_modules: Set[PerceptionModule]) -> bool
```

初始化感知模块并加载指定模块的模型。

例程：`galbot_perception/foundation_stereo_example.py`
### 为指定模块运行一次推理（run_once）

```python
def run_once(module: PerceptionModule) -> bool
```

为指定模块运行一次推理。

例程：`galbot_perception/foundation_stereo_example.py`
### 等待模块产生新结果或超时（wait_for_new_result）

```python
def wait_for_new_result(module: PerceptionModule, timeout_s: SupportsFloat = 5.0) -> bool
```

阻塞直到模块产生新结果或超时。与run_once配合使用获取最新输出。

例程：`galbot_perception/foundation_stereo_example.py`

## 15. 类型和枚举

字段定义在 `docs/g1/zh/api_python_reference/index.html` 的“类型与枚举”。名称如下：

- 驱动类型（ActuateType）
- 音频数据结构（AudioData）
- 碰撞检查选项（CollisionCheckOption）
- 碰撞信息（CollisionInfo）
- 配置项（ConfigItem）
- 配置服务（ConfigService）
- 控制状态（ControlStatus）
- 深度数据（DepthData）
- 检测与分割结果（DetectionAndSegmentationResult）
- 检测结果（DetectionResult）
- 灵巧手状态（DexhandState）
- 灵巧手类型（DexHandType）
- 力矩信息（EffortInfo）
- 编码的视频帧数据（EncodedVideoData）
- 错误（Error）
- 错误信息（ErrorInfo）
- 力数据（ForceData）
- 坐标系（FrameTriad）
- G1控制器名称（G1ControllerName）
- G1关节组（G1JointGroup）
- Galbot One Foxtrot 传感器类型（GalbotOneFoxtrotSensor）
- 夹爪状态（GripperState）
- 关节组命令（GroupCommand）
- 消息头（Header）
- IK 求解器配置（IKSolverConfig）
- IMU数据（ImuData）
- 关节命令（JointCommand）
- 单关节状态（JointState）
- 关节状态消息（JointStateMessage）
- 关节状态（JointStates）
- 运动学边界（KinematicsBoundary）
- 激光雷达数据（LidarData）
- 直线轨迹碰撞检测基元（LineTrajCheckPrimitive）
- 日志级别（LogLevel）
- 机器类型（MachineType）
- 运动规划链目标（MotionPlanChainTarget）
- 运动规划配置（MotionPlanConfig）
- 运动规划目标模式（MotionPlanTargetMode）
- 运动规划类型（MotionPlanType）
- 运动状态（MotionStatus）
- 导航任务快照（NavigationTaskSnapshot）
- 导航任务状态（NavigationTaskStatus）
- 里程计数据（OdomData）
- 参数（Parameter）
- 感知模块（PerceptionModule）
- 规划器配置（PlannerConfig）
- 规划请求（PlanRequest）
- 点（Point）
- 二维点（Point2d）
- 点云字段（PointField）
- 点云字段数据类型（PointFieldDataType）
- 姿态（Pose）
- 二维位姿（Pose2d）
- 姿态状态（PoseState）
- 几何基元类型（PrimitiveType）
- 四元数（Quaternion）
- RGB数据（RgbData）
- RGB图像输出格式（RgbOutputFormat）
- 机器人状态（RobotStates）
- 机器人状态类型（RobotStatesType）
- 采样器配置（SamplerConfig）
- 随机种子类型（SeedType）
- 传感器状态（SensorStatus）
- 传感器类型（SensorType）
- SingoriX目标（SingoriXTarget）
- 状态检查类型（StateCheckType）
- 吸盘动作状态（SUCTION_ACTION_STATE）
- 吸盘状态（SuctionCupState）
- 同步观测数据（SyncedObservation）
- 目标配置（TargetConfig）
- 目标关节组轨迹（TargetGroupTrajectory）
- 目标采样（TargetSampling）
- 目标任务轨迹（TargetTaskTrajectory）
- 任务命令（TaskCommand）
- 导航任务句柄（TaskHandle）
- 终止条件类型（TerminationConditionType）
- 时间戳（Timestamp）
- 轨迹（Trajectory）
- 轨迹控制状态（TrajectoryControlStatus）
- 轨迹可行性检查选项（TrajectoryFeasibilityCheckOption）
- 轨迹规划配置（TrajectoryPlanConfig）
- 轨迹点（TrajectoryPoint）
- 扭曲速度（Twist）
- 超声波数据（UltrasonicData）
- 超声波类型（UltrasonicType）
- 三维向量（Vector3）
- 路点（Waypoint）
- 路点可选参数（WaypointParams）
- 全身控制异常（WBCException）
- 力与力矩（Wrench）

