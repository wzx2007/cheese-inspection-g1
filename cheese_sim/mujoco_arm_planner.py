"""Offline both-arm planner backed by MuJoCo and mink.

This planner implements the protocol expected by
:class:`scripts.armctrl.arm_controller.RightArmController`.  A right-arm joint
target also sets the left arm to the negated joints.  Cartesian targets are
solved for the right TCP, then mirrored the same way.

The planner is stricter than the controller:

* joint targets must be inside the URDF limits with ``min_joint_margin_rad``;
* Cartesian targets are solved with mink and checked for FK error before the
  state is accepted;
* if ``check_collisions`` is true, any active contact rejects the target.

It also records every call in ``self.calls`` so the tests can assert that only
right-arm joints were commanded and that obstacle arguments were forwarded.

Read-only on the PC: does not touch the robot.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.armctrl.arm_controller import (
    CONFIG_DIR,
    RIGHT_JOINT_NAMES,
    normalize_pose,
    quat_xyzw_to_matrix,
)
from cheese_sim.arm_ik_common import (
    XML,
    fk_pose,
    joint_limit_margins,
    quat_xyzw_from_matrix,
    set_arm,
    solve_arm_ik,
)


class MujocoArmPlanner:
    """Right-arm planner for offline tests.

    Args:
        config_dir: Directory with the arm preset JSON files.
        start_right_q: Right-arm joints at construction.  Defaults to the
            ``reset_right_arm.json`` preset.
        start_left_q: Left-arm joints at construction.  Defaults to the
            element-wise negative of ``start_right_q``.  Later plans overwrite
            it with the negation of the right-arm target.
        check_collisions: Reject joint/Cartesian targets with active contacts.
        min_joint_margin_rad: Required distance from every hard joint limit.
        ik_iterations: mink integration steps for a Cartesian target.
        ik_position_tolerance_m: Accepted FK position error after IK.
        ik_orientation_tolerance_deg: Accepted FK orientation error after IK.
    """

    def __init__(
        self,
        *,
        config_dir: Optional[Path] = None,
        start_right_q: Optional[Sequence[float]] = None,
        start_left_q: Optional[Sequence[float]] = None,
        check_collisions: bool = True,
        min_joint_margin_rad: float = 0.005,
        ik_iterations: int = 2000,
        ik_position_tolerance_m: float = 0.002,
        ik_orientation_tolerance_deg: float = 0.5,
    ) -> None:
        self._config_dir = Path(config_dir or CONFIG_DIR)
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self.check_collisions = bool(check_collisions)
        self.min_joint_margin_rad = float(min_joint_margin_rad)
        self.ik_iterations = int(ik_iterations)
        self.ik_position_tolerance_m = float(ik_position_tolerance_m)
        self.ik_orientation_tolerance_deg = float(ik_orientation_tolerance_deg)

        if start_right_q is None:
            start_right_q = self._load_preset_joints("reset_right_arm.json")
        self.right_q = self._validate_joints(start_right_q, "start_right_q")
        if start_left_q is None:
            start_left_q = [-value for value in self.right_q]
        self.left_q = [float(value) for value in start_left_q]
        if len(self.left_q) != 7:
            raise ValueError("start_left_q must have 7 values")

        self.calls: List[Dict] = []
        self.last_contacts = 0
        self.last_ik_position_error_mm = 0.0
        self.last_ik_orientation_error_deg = 0.0
        self._active_obstacles: Optional[List[Dict]] = None
        self._sync_model()

    # ------------------------------------------------------------------
    # Planner protocol
    # ------------------------------------------------------------------
    def get_joint_positions(self, joint_names: Sequence[str]) -> List[float]:
        names = [str(name) for name in joint_names]
        if names == list(RIGHT_JOINT_NAMES):
            return list(self.right_q)
        values: List[float] = []
        for name in names:
            if name.startswith("right_arm_joint"):
                index = int(name.rsplit("joint", 1)[-1]) - 1
                values.append(self.right_q[index])
            elif name.startswith("left_arm_joint"):
                index = int(name.rsplit("joint", 1)[-1]) - 1
                values.append(self.left_q[index])
            else:
                raise KeyError("unknown joint name: {}".format(name))
        return values

    def get_tcp_pose(self) -> List[float]:
        """Return ``[x, y, z, qx, qy, qz, qw]`` of the right gripper TCP."""
        return self._tcp_pose("right")

    def get_left_tcp_pose(self) -> List[float]:
        """Return ``[x, y, z, qx, qy, qz, qw]`` of the left gripper TCP."""
        return self._tcp_pose("left")

    def _tcp_pose(self, side: str) -> List[float]:
        self._sync_model()
        position, rotation = fk_pose(self.model, self.data, side)
        return [float(value) for value in position] + quat_xyzw_from_matrix(rotation)

    def plan_joint(self, joint_names, target_q, obstacles=None) -> None:
        names = [str(name) for name in joint_names]
        if names != list(RIGHT_JOINT_NAMES):
            raise ValueError(
                "plan_joint expects the seven right-arm joints; the left arm "
                "is set to their negation"
            )
        target = self._validate_joints(target_q, "plan_joint")
        if obstacles is not None:
            self._active_obstacles = copy.deepcopy(obstacles)
        self.calls.append(
            {
                "type": "joint",
                "joint_names": list(names),
                "target_q": list(target),
                "obstacles": copy.deepcopy(obstacles),
            }
        )
        old_right = list(self.right_q)
        old_left = list(self.left_q)
        self.right_q = target
        self.left_q = [-value for value in target]
        contacts = self._sync_model()
        if self.check_collisions and contacts:
            self.right_q = old_right
            self.left_q = old_left
            self._sync_model()
            raise RuntimeError(
                "joint target rejected: {} active contact(s)".format(contacts)
            )

    def plan_cartesian(self, target_pose, obstacles=None) -> None:
        target = normalize_pose(target_pose)
        if obstacles is not None:
            self._active_obstacles = copy.deepcopy(obstacles)
        self.calls.append(
            {
                "type": "cartesian",
                "target_pose": list(target),
                "obstacles": copy.deepcopy(obstacles),
            }
        )
        seed = list(self.right_q)
        target_rotation = np.array(quat_xyzw_to_matrix(target[3:]), dtype=float)
        q = solve_arm_ik(
            self.model,
            self.data,
            "right",
            target[:3],
            target_rotation,
            seed,
            posture_cost=1e-4,
            iterations=self.ik_iterations,
        )
        q = np.clip(
            np.asarray(q, dtype=float),
            self._joint_lower(),
            self._joint_upper(),
        )
        candidate = [float(value) for value in q]
        self._validate_joints(candidate, "plan_cartesian IK")

        old_right = list(self.right_q)
        old_left = list(self.left_q)
        self.right_q = candidate
        self.left_q = [-value for value in candidate]
        contacts = self._sync_model()
        position, rotation = fk_pose(self.model, self.data, "right")
        position_error_m = float(np.linalg.norm(position - np.asarray(target[:3])))
        orientation_error_deg = _rotation_error_deg(rotation, target_rotation)
        self.last_ik_position_error_mm = position_error_m * 1000.0
        self.last_ik_orientation_error_deg = orientation_error_deg

        problems = []
        if self.check_collisions and contacts:
            problems.append("{} active contact(s)".format(contacts))
        if position_error_m > self.ik_position_tolerance_m:
            problems.append(
                "position error {:.3f} mm".format(position_error_m * 1000.0)
            )
        if orientation_error_deg > self.ik_orientation_tolerance_deg:
            problems.append(
                "orientation error {:.3f} deg".format(orientation_error_deg)
            )
        if problems:
            self.right_q = old_right
            self.left_q = old_left
            self._sync_model()
            raise RuntimeError("Cartesian target rejected: " + "; ".join(problems))

    def clear_obstacles(self) -> None:
        self._active_obstacles = None

    @property
    def active_obstacles(self):
        """Obstacle list last passed to a plan call (test helper)."""
        return copy.deepcopy(self._active_obstacles)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def set_right_arm(self, q: Sequence[float]) -> None:
        """Test helper: overwrite the current right-arm state."""
        self.right_q = self._validate_joints(q, "set_right_arm")
        self._sync_model()

    def _sync_model(self) -> int:
        self.data.qpos[:] = self.model.qpos0
        set_arm(self.model, self.data, "right", self.right_q)
        set_arm(self.model, self.data, "left", self.left_q)
        mujoco.mj_forward(self.model, self.data)
        self.last_contacts = int(self.data.ncon)
        return self.last_contacts

    def _validate_joints(self, values, action: str) -> List[float]:
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != 7:
            raise ValueError("{}: expected 7 joint values".format(action))
        q = [float(value) for value in values]
        if not all(np.isfinite(value) for value in q):
            raise ValueError("{}: joint value is not finite".format(action))
        lower = self._joint_lower()
        upper = self._joint_upper()
        lower = list(lower)
        upper = list(upper)
        lower[6] = min(float(lower[6]), -1.54)
        upper[6] = max(float(upper[6]), 1.54)
        for index, (value, low, high) in enumerate(zip(q, lower, upper), start=1):
            margin = 0.0 if index == 7 else self.min_joint_margin_rad
            if value < low + margin - 1e-8:
                raise ValueError(
                    "{}: joint{}={:.6f} is within {:.4f} rad of the lower "
                    "limit {:.6f}".format(action, index, value, margin, low)
                )
            if value > high - margin + 1e-8:
                raise ValueError(
                    "{}: joint{}={:.6f} is within {:.4f} rad of the upper "
                    "limit {:.6f}".format(action, index, value, margin, high)
                )
        return q

    def _joint_lower(self) -> np.ndarray:
        return np.array(
            [
                self.model.jnt_range[
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                ][0]
                for name in RIGHT_JOINT_NAMES
            ]
        )

    def _joint_upper(self) -> np.ndarray:
        return np.array(
            [
                self.model.jnt_range[
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                ][1]
                for name in RIGHT_JOINT_NAMES
            ]
        )

    def joint_margins(self) -> List[float]:
        """Current distance from each right-arm joint to its hard limit."""
        return joint_limit_margins(self.model, "right", self.right_q)

    def _load_preset_joints(self, filename: str) -> List[float]:
        path = self._config_dir / filename
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [float(value) for value in payload["right_arm"]]


def _rotation_error_deg(
    actual: np.ndarray, target: np.ndarray
) -> float:
    cosine = (np.trace(actual.T @ target) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
