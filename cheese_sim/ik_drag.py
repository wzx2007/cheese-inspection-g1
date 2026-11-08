"""交互式末端 IK：拖动红色 mocap 球，机械臂 TCP 跟随。

用法（项目根目录）：
    .venv\\Scripts\\python.exe cheese_sim\\ik_drag.py
    .venv\\Scripts\\python.exe cheese_sim\\ik_drag.py --arm left
    .venv\\Scripts\\python.exe cheese_sim\\ik_drag.py --variant fixed
    .venv\\Scripts\\python.exe cheese_sim\\ik_drag.py --test

操作：
    1. 在 viewer 中双击红色目标球（mocap body）选中它；
    2. Ctrl + 右键拖动 = 平移目标；
    3. Ctrl + 左键拖动 = 旋转目标；
    机械臂 TCP 会用 mink 逆运动学跟随目标，其余关节保持不动。
    关闭窗口退出。

说明：这是运动学 IK 演示，不做动力学步进，所以机器人不会因为重力倒下。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mink
import mujoco
import mujoco.viewer
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
MJCF_DIR = ROOT / "refs" / "galbot_one_golf_description-main" / "mjcf"
VARIANTS = {
    "wheeled": MJCF_DIR / "galbot_one_golf.xml",
    "fixed": MJCF_DIR / "galbot_one_golf_fixed_base.xml",
    "planar": MJCF_DIR / "galbot_one_golf_planar_base.xml",
}
TARGET = "ik_target"


def build_model(xml: Path) -> mujoco.MjModel:
    """在原模型上追加一个红色 mocap 目标球和 XYZ 小坐标轴。"""
    spec = mujoco.MjSpec.from_file(str(xml))
    target = spec.worldbody.add_body(name=TARGET, mocap=True, pos=[0.0, 0.0, 0.0])
    target.add_geom(
        name=f"{TARGET}_sphere",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.035, 0.0, 0.0],
        rgba=[1.0, 0.2, 0.1, 0.9],
        contype=0,
        conaffinity=0,
    )
    # 目标姿态的三根小轴：红 X、绿 Y、蓝 Z
    for vec, color in (
        ([0.09, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
        ([0.0, 0.09, 0.0], [0.0, 1.0, 0.0, 1.0]),
        ([0.0, 0.0, 0.09], [0.0, 0.0, 1.0, 1.0]),
    ):
        target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.006, 0.0, 0.0],
            fromto=[0.0, 0.0, 0.0, *vec],
            rgba=color,
            contype=0,
            conaffinity=0,
        )
    return spec.compile()


def arm_dof_mask(model: mujoco.MjModel, arm: str) -> np.ndarray:
    """只允许指定手臂的 7 个关节运动，其余 DOF 全部锁死。"""
    mask = np.zeros(model.nv, dtype=bool)
    for i in range(1, 8):
        name = f"{arm}_arm_joint{i}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit(f"模型里找不到关节 {name}")
        mask[int(model.jnt_dofadr[jid])] = True
    return mask


def make_ik(
    model: mujoco.MjModel, data: mujoco.MjData, arm: str
) -> tuple[mink.Configuration, mink.FrameTask, np.ndarray, str]:
    tcp_name = f"{arm}_gripper_tcp_link"
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tcp_name) < 0:
        raise SystemExit(f"模型里找不到 TCP body: {tcp_name}")

    mujoco.mj_forward(model, data)
    mink.move_mocap_to_frame(model, data, TARGET, tcp_name, "body")

    configuration = mink.Configuration(model, q=data.qpos.copy())
    task = mink.FrameTask(
        frame_name=tcp_name,
        frame_type="body",
        position_cost=1.0,
        orientation_cost=0.5,
        lm_damping=1.0,
    )
    task.set_target(mink.SE3.from_mocap_name(model, data, TARGET))
    return configuration, task, arm_dof_mask(model, arm), tcp_name


def ik_step(
    configuration: mink.Configuration,
    task: mink.FrameTask,
    mask: np.ndarray,
    dt: float,
) -> np.ndarray:
    velocity = mink.solve_ik(configuration, [task], dt, "daqp", damping=1e-2)
    velocity[~mask] = 0.0
    configuration.integrate_inplace(velocity, dt)
    return velocity


def run_test(model: mujoco.MjModel, data: mujoco.MjData, arm: str) -> None:
    configuration, task, mask, tcp_name = make_ik(model, data, arm)
    mocap_id = int(model.body(TARGET).mocapid[0])
    before = data.qpos.copy()
    data.mocap_pos[mocap_id] += np.array([0.02, 0.0, 0.0])

    for _ in range(400):
        task.set_target(mink.SE3.from_mocap_name(model, data, TARGET))
        ik_step(configuration, task, mask, 0.01)

    data.qpos[:] = configuration.q
    mujoco.mj_forward(model, data)
    pos_err = np.linalg.norm(data.body(tcp_name).xpos - data.mocap_pos[mocap_id])

    arm_qpos = np.array(
        [
            model.jnt_qposadr[
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_arm_joint{i}"
                )
            ]
            for i in range(1, 8)
        ]
    )
    non_arm_qpos = np.ones(model.nq, dtype=bool)
    non_arm_qpos[arm_qpos] = False
    max_locked_change = float(
        np.max(np.abs(configuration.q[non_arm_qpos] - before[non_arm_qpos]))
    )
    moved_arm_joints = int(
        np.count_nonzero(np.abs(configuration.q[arm_qpos] - before[arm_qpos]) > 1e-9)
    )

    print(f"TCP 位置误差: {pos_err * 1000:.3f} mm")
    print(f"手臂动了的关节数: {moved_arm_joints}/7")
    print(f"非手臂 qpos 最大变化: {max_locked_change:.2e}（应接近 0）")
    if pos_err > 1e-3:
        raise SystemExit("IK 测试未收敛")
    if max_locked_change > 1e-9:
        raise SystemExit("非手臂关节发生了移动")


def main() -> None:
    parser = argparse.ArgumentParser(description="拖动 mocap 球控制 Galbot One Golf 末端")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="wheeled")
    parser.add_argument("--arm", choices=("right", "left"), default="right")
    parser.add_argument("--axes", action="store_true", help="同时显示全部关节轴")
    parser.add_argument("--test", action="store_true", help="不开窗口，测试 IK 是否收敛")
    args = parser.parse_args()

    xml = VARIANTS[args.variant]
    if not xml.exists():
        raise SystemExit(f"MJCF 不存在：{xml}")

    model = build_model(xml)
    data = mujoco.MjData(model)

    if args.test:
        print(f"测试模型：{xml}")
        run_test(model, data, args.arm)
        return

    configuration, task, mask, tcp_name = make_ik(model, data, args.arm)
    viewer = mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=True
    )
    if args.axes:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
        viewer.opt.jointgroup[:] = 1

    tcp_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tcp_name)
    viewer.cam.lookat[:] = data.xpos[tcp_body_id]
    viewer.cam.distance = 1.6
    viewer.cam.elevation = -20.0
    viewer.cam.azimuth = 135.0

    print(f"模型：{xml}")
    print(f"控制 {args.arm} 臂，TCP：{tcp_name}")
    print("操作：双击红色目标球选中；Ctrl+右键拖动=平移；Ctrl+左键拖动=旋转。")
    print("右侧 Rendering 可勾 Joint 显示关节轴；关闭窗口退出。", flush=True)

    dt = 0.01
    while viewer.is_running():
        task.set_target(mink.SE3.from_mocap_name(model, data, TARGET))
        ik_step(configuration, task, mask, dt)
        data.qpos[:] = configuration.q
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(dt)


if __name__ == "__main__":
    main()
