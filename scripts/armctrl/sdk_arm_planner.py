"""SDK adapter for :class:`scripts.armctrl.arm_controller.RightArmController`.

The controller owns the state machine and the preset JSON files; this class
owns the ``galbot_sdk`` calls.  A right-arm joint target is sent together with
the left arm at the negated joints.  The caller must create and initialise
``GalbotRobot`` and ``GalbotMotion`` first, enable the robot, release the
emergency stop, and confirm the workspace is clear.

Example::

    import time
    from galbot_sdk.g1 import GalbotMotion, GalbotRobot
    from scripts.armctrl.sdk_arm_planner import SdkArmPlanner
    from scripts.armctrl.arm_controller import RightArmController

    robot = GalbotRobot()
    motion = GalbotMotion()
    if not robot.init() or not motion.init():
        raise SystemExit("SDK init failed")
    try:
        time.sleep(2.0)
        planner = SdkArmPlanner(robot, motion)
        arm = RightArmController(planner)
        arm.reset_both_arm()
        arm.set_both_arm_ready()
        arm.set_half_gap(0.32)
        arm.set_end_joint(1.54)
        arm.reset_both_arm()
    finally:
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()

Obstacle dictionaries accepted by the planner methods::

    {
        "id": "table_top",
        "type": "box",                       # box/sphere/cylinder/mesh/...
        "pose": [x, y, z, qx, qy, qz, qw],   # in "frame"
        "scale": [length, width, height],    # type-dependent
        "frame": "world",                    # optional, default "world"
        "key": "",                           # optional, mesh/point cloud
        "safe_margin": 0.0,                  # optional
    }

``obstacles=None`` keeps the currently loaded set.  Passing a different list
clears and reloads the planner-owned obstacle set, so repeated calls with the
same list are idempotent.  ``clear_obstacles()`` removes the whole set.

The TCP frame is selected with ``Parameter.set_tool_pose(True)`` so the SDK
targets the gripper TCP rather than the flange.  Verify on the robot (read-only)
that this matches ``right_gripper_tcp_link`` before the first motion.
"""

from __future__ import annotations

import copy
import importlib
import time
from typing import Dict, List, Optional, Sequence

if __package__:
    from .arm_controller import normalize_pose, quat_xyzw_to_matrix, rotation_angle_deg
else:
    from arm_controller import normalize_pose, quat_xyzw_to_matrix, rotation_angle_deg

RIGHT_ARM = "right_arm"
LEFT_ARM = "left_arm"
RIGHT_JOINT_NAMES = tuple(
    "right_arm_joint{}".format(index) for index in range(1, 8)
)
LEFT_JOINT_NAMES = tuple(
    "left_arm_joint{}".format(index) for index in range(1, 8)
)


class SdkArmPlannerError(RuntimeError):
    """The SDK rejected an arm request or returned an invalid result."""


class SdkArmPlanner:
    """Right-arm planner/executor backed by ``galbot_sdk.g1``.

    Args:
        robot: Initialised ``GalbotRobot`` instance.
        motion: Initialised ``GalbotMotion`` instance.
        chain: Kept for compatibility.  Both arms are always commanded.
        tool_pose: Pass ``True`` to command the gripper TCP.  ``False``
            commands the flange.
        timeout_s: Motion timeout in seconds.
        check_collision: Enable the SDK's collision check.
        settle_s: Optional sleep after a blocking motion.
        sdk_module: Only for tests; inject a fake ``galbot_sdk.g1`` module
            instead of importing the real one.
    """

    def __init__(
        self,
        robot,
        motion,
        *,
        chain: str = RIGHT_ARM,
        tool_pose: bool = True,
        timeout_s: float = 30.0,
        check_collision: bool = True,
        settle_s: float = 0.0,
        sdk_module=None,
    ) -> None:
        for name in ("get_joint_positions",):
            if not callable(getattr(robot, name, None)):
                raise TypeError("robot is missing {}()".format(name))
        for name in (
            "get_end_effector_pose_on_chain",
            "traj_plan",
            "move_line",
            "inverse_kinematics",
            "add_obstacle",
            "clear_obstacle",
        ):
            if not callable(getattr(motion, name, None)):
                raise TypeError("motion is missing {}()".format(name))
        self._robot = robot
        self._motion = motion
        self._chain = chain
        self._tool_pose = bool(tool_pose)
        self._timeout_s = float(timeout_s)
        self._check_collision = bool(check_collision)
        self._settle_s = float(settle_s)
        self._gm = sdk_module or importlib.import_module("galbot_sdk.g1")
        self._active_obstacles: Optional[List[Dict]] = None

    # ------------------------------------------------------------------
    # Planner protocol
    # ------------------------------------------------------------------
    def get_joint_positions(self, joint_names: Sequence[str]) -> List[float]:
        names = [str(name) for name in joint_names]
        # Match the official G1 examples, which call this positionally.
        values = self._robot.get_joint_positions([], names)
        result = [float(value) for value in values]
        if len(result) != len(names):
            raise SdkArmPlannerError(
                "get_joint_positions returned {} values for {} names".format(
                    len(result), len(names)
                )
            )
        return result

    def get_tcp_pose(self) -> List[float]:
        return self._tcp_pose(RIGHT_ARM)

    def get_left_tcp_pose(self) -> List[float]:
        return self._tcp_pose(LEFT_ARM)

    def plan_joint(self, joint_names, target_q, obstacles=None) -> None:
        names = [str(name) for name in joint_names]
        if names != list(RIGHT_JOINT_NAMES):
            raise ValueError(
                "plan_joint expects the seven right-arm joints in order; "
                "the left arm is commanded as their negation"
            )
        right_q = [float(value) for value in target_q]
        if len(right_q) != 7:
            raise ValueError("plan_joint expected 7 joint values")
        left_q = [-value for value in right_q]
        self._apply_obstacles(obstacles)

        right = self._joint_target(RIGHT_ARM, RIGHT_JOINT_NAMES, right_q)
        left = self._joint_target(LEFT_ARM, LEFT_JOINT_NAMES, left_q)
        params = self._gm.Parameter()
        params.set_direct_execute(True)
        params.set_blocking(True)
        params.set_timeout(self._timeout_s)
        params.set_check_collision(self._check_collision)
        params.set_reference_frame("base_link")
        params.set_actuate("with_chain_only")

        status, _trajectory = self._motion.traj_plan([[right, left]], params)
        self._require_success(status, "traj_plan(both arms)")
        self._settle()

    def _tcp_pose(self, chain: str) -> List[float]:
        status, pose = self._motion.get_end_effector_pose_on_chain(
            chain_name=chain,
            frame_id="TCP" if self._tool_pose else "EndEffector",
            reference_frame="base_link",
        )
        self._require_success(status, "get_end_effector_pose_on_chain({})".format(chain))
        values = [float(value) for value in pose]
        if len(values) != 7:
            raise SdkArmPlannerError(
                "get_end_effector_pose_on_chain({}) returned {} values, expected 7".format(
                    chain, len(values)
                )
            )
        return values

    def _joint_target(self, chain: str, names, joints):
        target = self._gm.MotionPlanChainTarget()
        target.chain_name = chain
        target.mode = self._gm.MotionPlanTargetMode.kJoint
        target.joint.chain_name = chain
        target.joint.joint_positions = list(joints)
        target.joint.joint_names = list(names)
        return target

    def _cartesian_target(self, chain: str, pose):
        frame = "TCP" if self._tool_pose else "EndEffector"
        target = self._gm.MotionPlanChainTarget()
        target.chain_name = chain
        target.mode = self._gm.MotionPlanTargetMode.kCartesian
        target.cart.chain_name = chain
        target.cart.frame_id = frame
        target.cart.reference_frame = "base_link"
        target.cart.pose = self._gm.Pose([float(value) for value in pose])
        return target

    def plan_line(self, right_pose, left_pose, obstacles=None) -> None:
        """Move both TCPs in straight lines to the given poses."""
        self._apply_obstacles(obstacles)
        right = self._cartesian_target(RIGHT_ARM, right_pose)
        left = self._cartesian_target(LEFT_ARM, left_pose)
        params = self._gm.Parameter()
        params.set_direct_execute(True)
        params.set_blocking(True)
        params.set_timeout(self._timeout_s)
        params.set_check_collision(self._check_collision)
        params.set_tool_pose(self._tool_pose)
        params.set_reference_frame("base_link")
        params.set_actuate("with_chain_only")
        status, _trajectory = self._motion.move_line([[right, left]], params)
        self._require_success(status, "move_line(both arms)")
        self._settle()

    def check_tcp_frame(self) -> None:
        """Check frame identity once, independently of robot arrival accuracy.

        The chain's TCP and the preset's named link must be the same frame.
        EndEffector is the flange; it must not be compared to the tool TCP.
        A missing TCP/model must be fixed before Cartesian motion, not hidden
        by relaxing arrival tolerances.
        """
        pairs = (
            (RIGHT_ARM, "right_gripper_tcp_link", self.get_tcp_pose),
            (LEFT_ARM, "left_gripper_tcp_link", self.get_left_tcp_pose),
        )
        for chain, link, chain_pose in pairs:
            status, pose = self._motion.get_end_effector_pose(
                end_effector_frame=link, reference_frame="base_link"
            )
            self._require_success(status, "get_end_effector_pose({})".format(link))
            direct = normalize_pose(pose)
            measured = normalize_pose(chain_pose())
            distance = sum((a - b) ** 2 for a, b in zip(direct[:3], measured[:3])) ** 0.5
            angle = rotation_angle_deg(
                quat_xyzw_to_matrix(direct[3:]), quat_xyzw_to_matrix(measured[3:])
            )
            if distance > 0.02 or angle > 0.5:
                raise SdkArmPlannerError(
                    "{} TCP differs from {}: {:.1f} mm / {:.2f} deg".format(
                        chain, link, distance * 1000, angle
                    )
                )

    def plan_cartesian(self, target_pose, obstacles=None) -> None:
        """Solve the right TCP, then move both arms with left joints negated."""
        self._apply_obstacles(obstacles)
        pose = [float(value) for value in target_pose]
        params = self._gm.Parameter()
        params.set_timeout(self._timeout_s)
        params.set_tool_pose(self._tool_pose)
        params.set_check_collision(self._check_collision)
        params.set_reference_frame("base_link")
        frame = "TCP" if self._tool_pose else "EndEffector"
        seed = {
            RIGHT_ARM: self.get_joint_positions(list(RIGHT_JOINT_NAMES)),
        }
        status, solved = self._motion.inverse_kinematics(
            target_pose=pose,
            chain_names=[RIGHT_ARM],
            target_frame=frame,
            reference_frame="base_link",
            initial_joint_positions=seed,
            enable_collision_check=self._check_collision,
            params=params,
        )
        self._require_success(status, "inverse_kinematics(right_arm)")
        try:
            right_q = [float(value) for value in solved[RIGHT_ARM]]
        except (KeyError, TypeError) as exc:
            raise SdkArmPlannerError(
                "inverse_kinematics did not return right_arm joints: {!r}".format(solved)
            ) from exc
        if len(right_q) != 7:
            raise SdkArmPlannerError(
                "inverse_kinematics returned {} right-arm joints".format(len(right_q))
            )
        self.plan_joint(list(RIGHT_JOINT_NAMES), right_q, obstacles=None)

    def clear_obstacles(self) -> None:
        if self._active_obstacles is None:
            return
        status = self._motion.clear_obstacle()
        self._require_success(status, "clear_obstacle")
        self._active_obstacles = None

    # ------------------------------------------------------------------
    # Obstacles and status helpers
    # ------------------------------------------------------------------
    def _apply_obstacles(self, obstacles) -> None:
        if obstacles is None:
            return
        normalized = [self._normalize_obstacle(item) for item in obstacles]
        if self._active_obstacles == normalized:
            return
        if self._active_obstacles is not None:
            status = self._motion.clear_obstacle()
            self._require_success(status, "clear_obstacle")
            self._active_obstacles = None
        for spec in normalized:
            status = self._motion.add_obstacle(
                obstacle_id=spec["id"],
                obstacle_type=spec["type"],
                pose=spec["pose"],
                scale=spec["scale"],
                key=spec["key"],
                target_frame=spec["frame"],
                safe_margin=spec["safe_margin"],
            )
            self._require_success(
                status, "add_obstacle({!r})".format(spec["id"])
            )
        self._active_obstacles = normalized

    @staticmethod
    def _normalize_obstacle(spec) -> Dict:
        if not isinstance(spec, dict):
            raise TypeError("each obstacle must be a dict")
        try:
            obstacle_id = str(spec["id"])
            obstacle_type = str(spec["type"])
            pose = [float(value) for value in spec["pose"]]
        except KeyError as exc:
            raise ValueError(
                "obstacle needs 'id', 'type', and 'pose': {}".format(exc)
            ) from exc
        scale = [float(value) for value in spec.get("scale", [0.0, 0.0, 0.0])]
        if not obstacle_id:
            raise ValueError("obstacle 'id' must not be empty")
        if len(pose) != 7:
            raise ValueError("obstacle 'pose' must have 7 values")
        if len(scale) != 3:
            raise ValueError("obstacle 'scale' must have 3 values")
        return {
            "id": obstacle_id,
            "type": obstacle_type,
            "pose": pose,
            "scale": scale,
            "frame": str(spec.get("frame", "world")),
            "key": str(spec.get("key", "")),
            "safe_margin": float(spec.get("safe_margin", 0.0)),
        }

    def _settle(self) -> None:
        if self._settle_s > 0.0:
            time.sleep(self._settle_s)

    @staticmethod
    def _require_success(status, action: str) -> None:
        if SdkArmPlanner._status_is_success(status):
            return
        raise SdkArmPlannerError(
            "{} failed with status {!r}".format(action, status)
        )

    @staticmethod
    def _status_is_success(status) -> bool:
        if isinstance(status, bool):
            return status
        if status is None:
            return False
        name = getattr(status, "name", None)
        if isinstance(name, str):
            return name.upper() == "SUCCESS"
        text = str(status)
        if text.upper() == "SUCCESS":
            return True
        if text.rsplit(".", 1)[-1].upper() == "SUCCESS":
            return True
        try:
            return int(status) == 0
        except (TypeError, ValueError):
            return False

    def copy_obstacles(self):
        """Return a copy of the currently loaded obstacle list (test helper)."""
        return copy.deepcopy(self._active_obstacles)
