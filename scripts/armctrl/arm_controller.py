"""Both-arm preset and Cartesian control for the Galbot G1.

The class is independent of ``galbot_sdk``.  It talks to an injected planner,
either :class:`scripts.armctrl.sdk_arm_planner.SdkArmPlanner` or the offline
MuJoCo planner.  Every right-arm joint target is mirrored to the left arm by
negation (``left = -right``).  The planner sends both chains.

Planner protocol (duck-typed)::

    get_joint_positions(joint_names) -> Sequence[float]
    get_tcp_pose() -> Sequence[float]            # right TCP
    get_left_tcp_pose() -> Sequence[float]       # left TCP
    plan_joint(joint_names, target_q, obstacles=None) -> None
    plan_cartesian(target_pose, obstacles=None) -> None
    clear_obstacles() -> None                    # optional

``plan_joint`` receives the seven right-arm joints and commands the left arm
as their negation.  ``plan_cartesian`` receives the right TCP target; the
planner solves that arm and mirrors the joint solution.

The controller does not enable the robot, release the emergency stop, or check
that the workspace is clear.  Those are caller responsibilities.  It does not
pitch the torso.  The robot stays upright; wrist motion is the last joint.

Units: metres, radians, and degrees only where a name ends in ``_deg``.
Frames: TCP poses are in ``base_link`` and refer to ``right_gripper_tcp_link``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

RIGHT_ARM = "right_arm"
RIGHT_JOINT_NAMES: Tuple[str, ...] = tuple(
    "right_arm_joint{}".format(index) for index in range(1, 8)
)

# URDF limits for right_arm_joint1 .. right_arm_joint7.  The left arm has the
# same ranges but mirrored joint axes.
RIGHT_JOINT_LIMITS: Tuple[Tuple[float, float], ...] = (
    (-3.00432619, 3.00432619),
    (-1.6081, 1.6081),
    (-2.916972222, 2.916972222),
    (-1.869862177, 2.567993878),
    (-2.916972222, 2.916972222),
    (-0.7353981633974483, 0.8226646259971648),
    (-1.538, 1.538),
)

# Right arm joint 7.  The planner hard stop is +/-1.5382 rad.  1.538 stays on
# that stop without crossing it.  The left arm uses the negated value.
END_JOINT_LIMIT_RAD = 1.538
_END_JOINT_INDEX = 6

# Keep commanded presets away from the hard limits.  The movable preset is
# generated with a larger 0.03 rad target margin; 0.01 rad is the minimum the
# controller accepts.
_JOINT_LIMIT_MARGIN_RAD = 0.01


class ArmControllerError(RuntimeError):
    """Base class for arm controller failures."""


class ArmStateError(ArmControllerError):
    """The requested operation is not valid in the current state."""


class ArmLimitError(ArmControllerError, ValueError):
    """A requested joint angle, gap, or pitch is outside the software limits."""


class ArmVerificationError(ArmControllerError):
    """The planner returned but the robot did not reach the requested target."""


class ArmState(NamedTuple):
    """Snapshot returned by every public arm method.

    ``joints`` and ``tcp_pose`` are the right arm.  ``left_joints`` is the
    negation of the right-arm command.  ``end_joint_rad`` is right joint 7.
    """

    joints: Tuple[float, ...]
    left_joints: Tuple[float, ...]
    tcp_pose: Tuple[float, ...]
    left_tcp_pose: Tuple[float, ...]
    state: str
    is_ready: bool
    half_gap_m: Optional[float]
    end_joint_rad: float


# ---------------------------------------------------------------------------
# Pure pose helpers (no SDK, no MuJoCo; covered by testarm/test_pose_math.py)
# ---------------------------------------------------------------------------
def quat_xyzw_to_matrix(quat: Sequence[float]):
    """Return a 3x3 rotation matrix for an ``[x, y, z, w]`` quaternion."""
    if len(quat) != 4:
        raise ValueError("expected 4 quaternion components")
    x, y, z, w = (float(value) for value in quat)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def matrix_to_quat_xyzw(rotation: Sequence[Sequence[float]]) -> List[float]:
    """Return an ``[x, y, z, w]`` quaternion for a 3x3 rotation matrix."""
    m = [[float(value) for value in row] for row in rotation]
    if len(m) != 3 or any(len(row) != 3 for row in m):
        raise ValueError("rotation must be 3x3")
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("degenerate rotation matrix")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    return [x, y, z, w]


def rotation_y(angle_rad: float) -> Tuple[Tuple[float, float, float], ...]:
    """Right-handed rotation about ``base_link`` +y."""
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    return (
        (cosine, 0.0, sine),
        (0.0, 1.0, 0.0),
        (-sine, 0.0, cosine),
    )


def matrix_multiply(a, b) -> Tuple[Tuple[float, float, float], ...]:
    """Multiply two 3x3 matrices given as nested sequences."""
    return tuple(
        tuple(
            sum(float(a[row][k]) * float(b[k][column]) for k in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def rotation_angle_deg(a, b) -> float:
    """Geodesic angle between two rotation matrices, in degrees."""
    trace = sum(
        float(a[k][i]) * float(b[k][i]) for i in range(3) for k in range(3)
    )
    cosine = (trace - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def normalize_pose(pose: Sequence[float]) -> List[float]:
    """Validate a 7-vector pose and normalise its quaternion."""
    if len(pose) != 7:
        raise ValueError("expected [x, y, z, qx, qy, qz, qw], got {} values".format(len(pose)))
    values = [float(value) for value in pose]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pose contains a non-finite value")
    norm = math.sqrt(sum(value * value for value in values[3:]))
    if norm <= 1e-12:
        raise ValueError("pose quaternion is zero")
    values[3:] = [value / norm for value in values[3:]]
    return values


def pose_from_tcp_reference(reference: dict) -> List[float]:
    """Build ``[x, y, z, qx, qy, qz, qw]`` from a preset ``tcp_reference``."""
    try:
        position = [float(value) for value in reference["position_m"]]
    except (KeyError, TypeError) as exc:
        raise ValueError("tcp_reference.position_m is missing or invalid") from exc
    if len(position) != 3:
        raise ValueError("tcp_reference.position_m must have 3 values")
    if "quaternion_xyzw" in reference:
        quat = [float(value) for value in reference["quaternion_xyzw"]]
    elif "rotation_matrix" in reference:
        quat = matrix_to_quat_xyzw(reference["rotation_matrix"])
    else:
        raise ValueError("tcp_reference needs quaternion_xyzw or rotation_matrix")
    return normalize_pose(position + quat)


def half_gap_target(current_pose: Sequence[float], half_gap_m: float) -> List[float]:
    """Move the right TCP to ``y = -half_gap_m``, keeping x/z and orientation."""
    pose = normalize_pose(current_pose)
    gap = float(half_gap_m)
    if not math.isfinite(gap) or gap <= 0.0:
        raise ValueError("half_gap_m must be finite and positive")
    target = list(pose)
    target[1] = -gap
    return target


def mirrored_joints(right_q: Sequence[float]) -> List[float]:
    """Left-arm joints for a right-arm target.  Do not copy the right array."""
    return [-float(value) for value in right_q]


def _max_abs_difference(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _pose_error(actual: Sequence[float], target: Sequence[float]) -> Tuple[float, float]:
    """Return position error in metres and orientation error in degrees."""
    pos_error = math.sqrt(
        sum((float(actual[i]) - float(target[i])) ** 2 for i in range(3))
    )
    rot_error = rotation_angle_deg(
        quat_xyzw_to_matrix(actual[3:]), quat_xyzw_to_matrix(target[3:])
    )
    return pos_error, rot_error


class RightArmController:
    """Both-arm controller.  The right arm is the source; the left arm mirrors it.

    Args:
        planner: Object implementing the planner protocol in the module
            docstring.  Use ``SdkArmPlanner`` on the robot and
            ``MujocoArmPlanner`` offline.
        config_dir: Directory containing the preset JSON files.
        test: Retained flag.  Both arms are always commanded; ``test`` does
            not select a single arm.
        verify: Extra application-level arrival checks. Default False trusts
            the planner's blocking execution result and returns measured state.
        joint_tolerance_rad: Read-back tolerance for joint-space moves.
        position_tolerance_m: Read-back tolerance for Cartesian moves.
        orientation_tolerance_deg: Read-back tolerance for Cartesian moves.
        min_half_gap_m / max_half_gap_m: Software limits for ``set_half_gap``.

    Public motion methods:
    ``reset_both_arm``, ``set_both_arm_ready``, ``set_half_gap``,
    ``set_end_joint``.
    """

    def __init__(
        self,
        planner,
        *,
        config_dir: Optional[Path] = None,
        test: bool = False,
        verify: bool = False,
        joint_tolerance_rad: float = 0.02,
        position_tolerance_m: float = 0.01,
        orientation_tolerance_deg: float = 1.0,
        min_half_gap_m: float = 0.15,
        max_half_gap_m: float = 0.60,
    ) -> None:
        required = (
            "get_joint_positions",
            "get_tcp_pose",
            "get_left_tcp_pose",
            "plan_joint",
            "plan_cartesian",
        )
        missing = [name for name in required if not callable(getattr(planner, name, None))]
        if missing:
            raise TypeError(
                "planner is missing required method(s): " + ", ".join(missing)
            )

        self._planner = planner
        self._config_dir = Path(config_dir or CONFIG_DIR)
        self.test = bool(test)
        self.verify = bool(verify)
        self.joint_tolerance_rad = self._finite_positive(
            joint_tolerance_rad, "joint_tolerance_rad"
        )
        self.position_tolerance_m = self._finite_positive(
            position_tolerance_m, "position_tolerance_m"
        )
        self.orientation_tolerance_deg = self._finite_positive(
            orientation_tolerance_deg, "orientation_tolerance_deg"
        )
        self._min_half_gap_m = self._finite_positive(min_half_gap_m, "min_half_gap_m")
        self._max_half_gap_m = self._finite_positive(max_half_gap_m, "max_half_gap_m")
        if self._min_half_gap_m >= self._max_half_gap_m:
            raise ValueError("min_half_gap_m must be smaller than max_half_gap_m")

        ready = self._load_preset("set_both_arm_ready.json")
        initial = self._load_preset("reset_right_arm.json")
        self._ready_q = self._preset_joints(ready, "set_both_arm_ready.json")
        self._initial_q = self._preset_joints(initial, "reset_right_arm.json")
        self._validate_preset_symmetry(ready)
        self._validate_preset_symmetry(initial)

        self.state = "unknown"
        self.is_ready = False
        self.half_gap_m: Optional[float] = None
        self.end_joint_rad = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset_both_arm(self, obstacles=None) -> ArmState:
        """Move both arms to the raised initial pose.  Left joints are negated."""
        self._plan_joint(self._initial_q, obstacles, "reset_both_arm")
        self._mark_initial()
        return self.state_snapshot()

    def set_both_arm_ready(self, obstacles=None) -> ArmState:
        """Lower both arms from the raised pose to the level ready pose.

        This is a joint-space move.  Right joint 7 in the preset is 1.538 rad;
        the left arm receives -1.538 rad.  The torso is not pitched.
        """
        self._require_state("initial")
        joints = self._plan_joint(self._ready_q, obstacles, "set_both_arm_ready")
        pose = self._planner.get_tcp_pose()
        self.state = "ready"
        self.is_ready = True
        self.half_gap_m = abs(float(pose[1]))
        self.end_joint_rad = float(joints[_END_JOINT_INDEX])
        return self.state_snapshot()

    def set_half_gap(self, half_gap_m: float, obstacles=None) -> ArmState:
        """Move the right TCP to ``y = -half_gap_m`` and mirror that solution.

        x, z, and the current right-arm orientation stay put.  The left arm
        is not solved separately: its joints are the negation of the right-arm
        solution, so its TCP goes to ``y = +half_gap_m``.
        """
        self._require_state("ready")
        gap = self._finite(half_gap_m, "half_gap_m")
        if gap < self._min_half_gap_m or gap > self._max_half_gap_m:
            raise ArmLimitError(
                "half_gap_m {:.3f} outside [{:.3f}, {:.3f}]".format(
                    gap, self._min_half_gap_m, self._max_half_gap_m
                )
            )
        current = self._planner.get_tcp_pose()
        target = half_gap_target(current, gap)
        self._plan_cartesian(target, obstacles, "set_half_gap")
        self.half_gap_m = gap
        return self.state_snapshot()

    def shift_forward(self, distance_m: float, obstacles=None) -> ArmState:
        """Move both TCPs forward in ``base_link`` +x.  y, z and orientation stay.

        Positive ``distance_m`` reaches further toward the table.  The left arm
        follows the mirrored right-arm solution.  A later ``set_half_gap`` keeps
        this new x.
        """
        self._require_state("ready")
        distance = self._finite(distance_m, "distance_m")
        if distance < -0.05 or distance > 0.15:
            raise ArmLimitError(
                "forward distance {:.3f} m outside [-0.050, 0.150]".format(distance)
            )
        line = getattr(self._planner, "plan_line", None)
        if not callable(line):
            raise ArmControllerError("前伸需要 planner.plan_line()")
        right = normalize_pose(self._planner.get_tcp_pose())
        left = normalize_pose(self._planner.get_left_tcp_pose())
        right[0] += distance
        left[0] += distance
        line(right, left, obstacles=self._obstacles(obstacles))
        return self.state_snapshot()

    def shift_vertical(self, distance_m: float, obstacles=None) -> ArmState:
        """Move both TCPs straight along ``base_link`` z.  x, y and orientation stay.

        Positive ``distance_m`` is up.  Negative is down.  Both hands move
        together in a Cartesian line.
        """
        self._require_state("ready")
        distance = self._finite(distance_m, "distance_m")
        if abs(distance) > 0.25:
            raise ArmLimitError(
                "vertical distance {:.3f} m outside +/-0.250".format(distance)
            )
        line = getattr(self._planner, "plan_line", None)
        if not callable(line):
            raise ArmControllerError("垂直移动需要 planner.plan_line()")
        right = normalize_pose(self._planner.get_tcp_pose())
        left = normalize_pose(self._planner.get_left_tcp_pose())
        right[2] += distance
        left[2] += distance
        line(right, left, obstacles=self._obstacles(obstacles))
        return self.state_snapshot()

    def set_end_joint(self, joint_rad: float, obstacles=None) -> ArmState:
        """Set right joint 7 and the negated left joint 7.

        The other twelve joints stay at their current mirrored values.  The
        argument is the right-arm value, in radians, inside ``[-1.538, 1.538]``.
        """
        self._require_state("ready")
        value = self._finite(joint_rad, "joint_rad")
        if abs(value) > END_JOINT_LIMIT_RAD + 1e-8:
            raise ArmLimitError(
                "end joint {:.4f} outside +/-{:.2f}".format(
                    value, END_JOINT_LIMIT_RAD
                )
            )
        current = self._planner.get_joint_positions(list(RIGHT_JOINT_NAMES))
        target = [float(item) for item in current]
        target[_END_JOINT_INDEX] = value
        self._plan_joint(target, obstacles, "set_end_joint")
        self.end_joint_rad = value
        return self.state_snapshot()

    def clear_obstacles(self) -> None:
        """Clear external obstacles in the planner, if it supports it."""
        clear = getattr(self._planner, "clear_obstacles", None)
        if not callable(clear):
            raise ArmControllerError("planner does not implement clear_obstacles()")
        clear()

    def state_snapshot(self) -> ArmState:
        """Read both arms and both TCP poses from the planner."""
        joints = [
            float(value)
            for value in self._planner.get_joint_positions(list(RIGHT_JOINT_NAMES))
        ]
        left_names = [
            "left_arm_joint{}".format(index) for index in range(1, 8)
        ]
        left = [
            float(value)
            for value in self._planner.get_joint_positions(left_names)
        ]
        pose = normalize_pose(self._planner.get_tcp_pose())
        left_pose = normalize_pose(self._planner.get_left_tcp_pose())
        return ArmState(
            joints=tuple(joints),
            left_joints=tuple(left),
            tcp_pose=tuple(pose),
            left_tcp_pose=tuple(left_pose),
            state=self.state,
            is_ready=self.is_ready,
            half_gap_m=self.half_gap_m,
            end_joint_rad=float(joints[_END_JOINT_INDEX]),
        )

    # ------------------------------------------------------------------
    # Preset loading and validation
    # ------------------------------------------------------------------
    def _load_preset(self, filename: str) -> dict:
        path = self._config_dir / filename
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError as exc:
            raise ArmControllerError("preset not found: {}".format(path)) from exc
        except json.JSONDecodeError as exc:
            raise ArmControllerError("invalid preset JSON: {}".format(path)) from exc
        if not isinstance(payload, dict):
            raise ArmControllerError("preset must be a JSON object: {}".format(path))
        return payload

    def _preset_joints(self, preset: dict, filename: str) -> List[float]:
        order = preset.get("joint_order")
        if isinstance(order, dict):
            order = order.get(RIGHT_ARM)
        if not isinstance(order, list) or [str(name) for name in order] != list(
            RIGHT_JOINT_NAMES
        ):
            raise ArmControllerError(
                "{} joint_order must list {} in order".format(
                    filename, ", ".join(RIGHT_JOINT_NAMES)
                )
            )
        values = preset.get(RIGHT_ARM)
        return self._validate_q(values, filename)

    def _validate_q(self, values, action: str) -> List[float]:
        if not isinstance(values, (list, tuple)) or len(values) != 7:
            raise ArmControllerError(
                "{}: expected 7 joint values, got {!r}".format(action, values)
            )
        q = [float(value) for value in values]
        if not all(math.isfinite(value) for value in q):
            raise ArmLimitError("{}: joint value is not finite".format(action))
        for index, (value, (lower, upper)) in enumerate(
            zip(q, RIGHT_JOINT_LIMITS), start=1
        ):
            margin = 0.0 if index == _END_JOINT_INDEX + 1 else _JOINT_LIMIT_MARGIN_RAD
            if value < lower + margin - 1e-8:
                raise ArmLimitError(
                    "{}: joint{}={:.6f} is within {:.3f} rad of the lower limit "
                    "{:.6f}".format(action, index, value, margin, lower)
                )
            if value > upper - margin + 1e-8:
                raise ArmLimitError(
                    "{}: joint{}={:.6f} is within {:.3f} rad of the upper limit "
                    "{:.6f}".format(action, index, value, margin, upper)
                )
        return q

    @staticmethod
    def _validate_preset_symmetry(preset: dict) -> None:
        right = preset.get(RIGHT_ARM)
        left = preset.get("left_arm")
        if left is None or right is None:
            raise ArmControllerError(
                "preset {} needs both arms; left_arm = -right_arm".format(
                    preset.get("name", "")
                )
            )
        if not isinstance(left, (list, tuple)) or len(left) != len(right):
            raise ArmControllerError("left_arm/right_arm length mismatch")
        expected = [-float(value) for value in right]
        if _max_abs_difference(left, expected) > 1e-6:
            raise ArmControllerError(
                "left_arm must equal -right_arm; do not copy the right array"
            )

    # ------------------------------------------------------------------
    # Planning and verification
    # ------------------------------------------------------------------
    def _plan_joint(self, target_q, obstacles, action: str) -> List[float]:
        target = self._validate_q(target_q, action)
        self._planner.plan_joint(
            list(RIGHT_JOINT_NAMES), target, obstacles=self._obstacles(obstacles)
        )
        actual = self._planner.get_joint_positions(list(RIGHT_JOINT_NAMES))
        if not self.verify:
            return list(actual)
        actual = self._validate_q(actual, action + " readback")
        error = _max_abs_difference(actual, target)
        if error > self.joint_tolerance_rad:
            raise ArmVerificationError(
                "{}: joint read-back error {:.4f} rad exceeds {:.4f} rad".format(
                    action, error, self.joint_tolerance_rad
                )
            )
        return actual

    def _plan_cartesian(self, target_pose, obstacles, action: str) -> List[float]:
        target = normalize_pose(target_pose)
        self._planner.plan_cartesian(
            target, obstacles=self._obstacles(obstacles)
        )
        actual = normalize_pose(self._planner.get_tcp_pose())
        if not self.verify:
            return actual
        position_error, orientation_error = _pose_error(actual, target)
        if (
            position_error > self.position_tolerance_m
            or orientation_error > self.orientation_tolerance_deg
        ):
            raise ArmVerificationError(
                "{}: TCP read-back error {:.4f} m / {:.3f} deg exceeds "
                "{:.4f} m / {:.3f} deg".format(
                    action,
                    position_error,
                    orientation_error,
                    self.position_tolerance_m,
                    self.orientation_tolerance_deg,
                )
            )
        # A Cartesian target can be pose-correct but still leave a joint at its
        # hard limit.  Apply the same margin check used for joint-space moves.
        joints = self._planner.get_joint_positions(list(RIGHT_JOINT_NAMES))
        self._validate_q(joints, action + " joint readback")
        return actual

    def _mark_initial(self) -> None:
        self.state = "initial"
        self.is_ready = False
        self.half_gap_m = None
        self.end_joint_rad = float(self._initial_q[_END_JOINT_INDEX])

    def _require_state(self, expected: str) -> None:
        if self.state != expected:
            raise ArmStateError(
                "expected state {!r}, current state is {!r}; call "
                "reset_both_arm() first".format(expected, self.state)
            )

    @staticmethod
    def _obstacles(obstacles):
        if obstacles is None:
            return None
        return [dict(item) for item in obstacles]

    @staticmethod
    def _finite(value, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("{} must be a number".format(name)) from exc
        if not math.isfinite(result):
            raise ValueError("{} must be finite".format(name))
        return result

    @staticmethod
    def _finite_positive(value, name: str) -> float:
        result = RightArmController._finite(value, name)
        if result <= 0.0:
            raise ValueError("{} must be positive".format(name))
        return result
