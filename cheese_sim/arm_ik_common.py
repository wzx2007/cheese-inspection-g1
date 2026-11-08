"""Shared MuJoCo/mink helpers for G1 arm preset generation.

This module is PC-only and read-only with respect to the robot.  It loads the
fixed-base MJCF model shipped in ``refs/`` and solves 7-DoF arm IK with mink.

All offline model helpers live in ``cheese_sim``. ``ROOT`` is the project root.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import mink
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XML = (
    ROOT
    / "refs"
    / "galbot_one_golf_description-main"
    / "mjcf"
    / "galbot_one_golf_fixed_base.xml"
)
CONFIG_DIR = ROOT / "scripts" / "config"

TCP_BODY = {
    "right": "right_gripper_tcp_link",
    "left": "left_gripper_tcp_link",
}

# URDF joint limits, in right_arm_joint1 .. right_arm_joint7 order.  The
# left arm has the same ranges because it is the mirrored chain.
ARM_JOINT_LIMITS: Tuple[Tuple[float, float], ...] = (
    (-3.00432619, 3.00432619),
    (-1.6081, 1.6081),
    (-2.916972222, 2.916972222),
    (-1.869862177, 2.567993878),
    (-2.916972222, 2.916972222),
    (-0.7353981633974483, 0.8226646259971648),
    (-1.538202778, 1.538202778),
)


def set_arm(model: mujoco.MjModel, data: mujoco.MjData, side: str, q: Sequence[float]) -> None:
    """Write seven arm joint values into ``data.qpos``."""
    if side not in TCP_BODY:
        raise ValueError("side must be 'right' or 'left'")
    if len(q) != 7:
        raise ValueError("expected 7 joint values, got {}".format(len(q)))
    for index, value in enumerate(q, start=1):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "{0}_arm_joint{1}".format(side, index)
        )
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)


def arm_qpos_indices(model: mujoco.MjModel, side: str) -> List[int]:
    """Return the seven ``qpos`` indices for one arm, in joint order."""
    indices: List[int] = []
    for index in range(1, 8):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "{0}_arm_joint{1}".format(side, index)
        )
        indices.append(int(model.jnt_qposadr[joint_id]))
    return indices


def arm_dof_mask(model: mujoco.MjModel, side: str) -> np.ndarray:
    """Boolean ``nv`` mask selecting the seven DOFs of one arm."""
    mask = np.zeros(model.nv, dtype=bool)
    for index in range(1, 8):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "{0}_arm_joint{1}".format(side, index)
        )
        mask[int(model.jnt_dofadr[joint_id])] = True
    return mask


def proper_signed_permutations() -> List[np.ndarray]:
    """All 24 axis-aligned proper rotations (signed permutation matrices)."""
    rotations: List[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            rotation = np.zeros((3, 3))
            for column, row in enumerate(permutation):
                rotation[row, column] = signs[column]
            if np.linalg.det(rotation) > 0:
                rotations.append(rotation)
    return rotations


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices, in degrees."""
    cosine = (np.trace(a.T @ b) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def quat_xyzw_from_matrix(rotation: np.ndarray) -> List[float]:
    """Convert a 3x3 rotation matrix to an ``[x, y, z, w]`` quaternion."""
    quat_wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(quat_wxyz, np.asarray(rotation, dtype=float).reshape(-1))
    return [
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
        float(quat_wxyz[0]),
    ]


def matrix_from_quat_xyzw(quat: Sequence[float]) -> np.ndarray:
    """Convert an ``[x, y, z, w]`` quaternion to a 3x3 rotation matrix."""
    if len(quat) != 4:
        raise ValueError("expected 4 quaternion components")
    x, y, z, w = (float(value) for value in quat)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def fk_pose(
    model: mujoco.MjModel, data: mujoco.MjData, side: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the TCP position and rotation of one arm after ``mj_forward``."""
    if side not in TCP_BODY:
        raise ValueError("side must be 'right' or 'left'")
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, TCP_BODY[side]
    )
    return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()


def solve_arm_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    target_position: Sequence[float],
    target_rotation: np.ndarray,
    q_seed: Sequence[float],
    *,
    posture_cost: float = 1e-4,
    iterations: int = 3000,
    dt: float = 0.005,
    damping: float = 1e-2,
) -> np.ndarray:
    """Solve one-arm IK at a fixed TCP position and orientation.

    Args:
        target_position: TCP position in ``base_link``, metres.
        target_rotation: TCP 3x3 rotation in ``base_link``.
        q_seed: Seven arm joints used as the initial state and posture target.
        posture_cost: Weight of the posture regularisation.  A small value
            keeps the solution near ``q_seed`` without preventing the frame
            task from reaching the target.
        iterations: Number of mink integration steps.
        dt: Integration step, seconds.
        damping: mink IK damping.

    Returns:
        The seven joint values as a NumPy array.  Joint limits are enforced by
        ``mink.Configuration``.
    """
    if side not in TCP_BODY:
        raise ValueError("side must be 'right' or 'left'")
    if len(q_seed) != 7:
        raise ValueError("q_seed must have 7 values")

    data.qpos[:] = model.qpos0
    set_arm(model, data, side, q_seed)
    mujoco.mj_forward(model, data)

    configuration = mink.Configuration(model, q=data.qpos.copy())
    task = mink.FrameTask(
        frame_name=TCP_BODY[side],
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture = mink.PostureTask(model, cost=posture_cost, lm_damping=1.0)
    posture.set_target(configuration.q)
    task.set_target(
        mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(np.asarray(target_rotation, dtype=float)),
            np.asarray(target_position, dtype=float),
        )
    )
    mask = arm_dof_mask(model, side)

    for _ in range(iterations):
        velocity = mink.solve_ik(
            configuration, [task, posture], dt, "daqp", damping=damping
        )
        velocity[~mask] = 0.0
        configuration.integrate_inplace(velocity, dt)

    return np.array(
        [configuration.q[index] for index in arm_qpos_indices(model, side)]
    )


def joint_limit_margins(
    model: mujoco.MjModel, side: str, q: Sequence[float]
) -> List[float]:
    """Distance from each joint to its nearest hard limit, radians."""
    margins: List[float] = []
    for index, value in enumerate(q, start=1):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "{0}_arm_joint{1}".format(side, index)
        )
        lower, upper = model.jnt_range[joint_id]
        margins.append(float(min(float(value) - lower, upper - float(value))))
    return margins


def collision_count(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    """Run ``mj_forward`` and return the number of active contacts."""
    mujoco.mj_forward(model, data)
    return int(data.ncon)


def rounded(values: Iterable[float], digits: int = 6) -> List[float]:
    """Round an iterable of floats to a fixed number of decimals."""
    return [round(float(value), digits) for value in values]
