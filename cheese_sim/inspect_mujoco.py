"""加载 Galbot One Golf 的 MJCF，打印关节/执行器定义；可选打开 MuJoCo viewer。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe cheese_sim\\inspect_mujoco.py
    .venv\\Scripts\\python.exe cheese_sim\\inspect_mujoco.py --variant fixed
    .venv\\Scripts\\python.exe cheese_sim\\inspect_mujoco.py --viewer --axes

查看器：左键拖动旋转，右键拖动平移，滚轮缩放；空格暂停/继续。
右侧 Rendering 面板勾选 Joint 可显示关节轴，Joint length / width 调整长短粗细。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
MJCF_DIR = ROOT / "refs" / "galbot_one_golf_description-main" / "mjcf"

VARIANTS = {
    "wheeled": MJCF_DIR / "galbot_one_golf.xml",
    "fixed": MJCF_DIR / "galbot_one_golf_fixed_base.xml",
    "planar": MJCF_DIR / "galbot_one_golf_planar_base.xml",
}

JOINT_TYPES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}


def name_of(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(model, obj, idx) or f"<unnamed:{idx}>"


def fmt_vec(values, precision: int = 4) -> str:
    return "[" + ", ".join(f"{v:.{precision}g}" for v in values) + "]"


def actuator_map(model: mujoco.MjModel) -> dict[int, list[str]]:
    """joint id -> 驱动它的 actuator 描述列表。"""
    mapping: dict[int, list[str]] = {}
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        jid = int(model.actuator_trnid[a, 0])
        aname = name_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        ctrl = (
            fmt_vec(model.actuator_ctrlrange[a])
            if model.actuator_ctrllimited[a]
            else "-"
        )
        mapping.setdefault(jid, []).append(f"{aname} (ctrl {ctrl})")
    return mapping


def print_joints(model: mujoco.MjModel) -> None:
    acts = actuator_map(model)
    print(f"\n{'idx':>3}  {'joint name':<34} {'type':<6} {'body':<28} "
          f"{'qpos':>4} {'dof':>4} {'axis':<14} {'range':<24} actuator")
    print("-" * 150)
    for j in range(model.njnt):
        jtype = JOINT_TYPES.get(int(model.jnt_type[j]), str(model.jnt_type[j]))
        axis = fmt_vec(model.jnt_axis[j]) if jtype != "free" else "-"
        rng = fmt_vec(model.jnt_range[j]) if model.jnt_limited[j] else "unlimited"
        act = ", ".join(acts.get(j, [])) or "-"
        print(
            f"{j:>3}  {name_of(model, mujoco.mjtObj.mjOBJ_JOINT, j):<34} "
            f"{jtype:<6} {name_of(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])):<28} "
            f"{model.jnt_qposadr[j]:>4} {model.jnt_dofadr[j]:>4} "
            f"{axis:<14} {rng:<24} {act}"
        )


def group_summary(model: mujoco.MjModel) -> None:
    prefixes = [
        ("floating_base", "floating_base"),
        ("leg_joint", "leg"),
        ("head_joint", "head"),
        ("left_arm_joint", "left_arm"),
        ("right_arm_joint", "right_arm"),
        ("left_gripper", "left_gripper"),
        ("right_gripper", "right_gripper"),
        ("wheel", "wheel"),
        ("base_x_joint", "base_x"),
        ("base_y_joint", "base_y"),
        ("base_yaw_joint", "base_yaw"),
    ]
    print("\n关节分组：")
    names = [name_of(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    for prefix, label in prefixes:
        matched = [n for n in names if n.startswith(prefix)]
        if matched:
            print(f"  {label:<14} {len(matched):>3}  {', '.join(matched)}")


def launch_viewer(model: mujoco.MjModel, show_axes: bool) -> None:
    """打开 viewer；--axes 时通过 MjvOption 默认勾上 Joint 可视化。"""
    if not show_axes:
        mujoco.viewer.launch(model)
        return

    original_option = mujoco.MjvOption

    def option_with_axes(*args, **kwargs):
        opt = original_option(*args, **kwargs)
        opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
        opt.jointgroup[:] = 1  # 显示 joint 0~5 全部组
        return opt

    mujoco.MjvOption = option_with_axes
    try:
        mujoco.viewer.launch(model)
    finally:
        mujoco.MjvOption = original_option


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 Galbot One Golf 的 MJCF 关节定义")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="wheeled",
        help="模型变体：wheeled（默认，含轮子）、fixed（固定底座）、planar（平面虚拟底座）",
    )
    parser.add_argument("--mjcf", type=Path, help="直接指定 MJCF 路径（优先于 --variant）")
    parser.add_argument("--viewer", action="store_true", help="打印后打开 MuJoCo viewer")
    parser.add_argument(
        "--axes",
        action="store_true",
        help="viewer 中默认显示关节轴（也可在右侧 Rendering 面板勾选 Joint）",
    )
    args = parser.parse_args()

    xml = args.mjcf or VARIANTS[args.variant]
    if not xml.exists():
        raise SystemExit(f"MJCF 不存在：{xml}")

    print(f"MuJoCo {mujoco.__version__}")
    print(f"模型文件：{xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # 验证模型可正常前向计算

    print(f"nbody={model.nbody}  njnt={model.njnt}  nq={model.nq}  nv={model.nv}  nu={model.nu}")
    print_joints(model)
    group_summary(model)

    print("\n提示：free 关节占 7 个 qpos / 6 个 dof；hinge 关节各占 1 个 qpos / 1 个 dof。")

    if args.viewer:
        print("打开 MuJoCo viewer（关闭窗口结束）...")
        print("操作：左键拖动旋转，右键拖动平移，滚轮缩放；空格暂停/继续。")
        print("关节轴：右侧 Rendering 面板勾选 Joint，可调 Joint length / width。")
        launch_viewer(model, args.axes)


if __name__ == "__main__":
    main()
