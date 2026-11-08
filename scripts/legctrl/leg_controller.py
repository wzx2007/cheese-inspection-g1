"""Leg-height control for the Galbot G1.

The public interface only raises and lowers ``head_base_link``.  The torso
stays upright; this module does not command a forward bow.

The planar model in this module was derived from
``refs/galbot_one_golf_description-main/urdf/galbot_one_golf.urdf`` and
validated against ``refs/galbot_one_golf_description-main/mjcf/
galbot_one_golf_fixed_base.xml``.  All lengths are metres and all angles are
radians unless a name explicitly ends in ``_deg``.

Coordinate convention (``base_link`` frame, right-handed):

* ``+x`` points forward (the direction the robot faces).
* ``+z`` points up.
* positive pitch tilts the top of the torso forward, i.e. a forward bow.

The controller deliberately does not import ``galbot_sdk``.  It only needs an
object (normally ``GalbotRobot``) that provides::

    get_transform(target_frame, source_frame, timestamp_ns=0, timeout_ms=100)
    get_joint_positions(joint_groups=[], joint_names=[])
    set_joint_positions(joint_positions, joint_groups=[], joint_names=[],
                        is_blocking=True, speed_rad_s=0.2, timeout_s=15.0)
    set_leg_height(link_height, duration=4.0, is_blocking=False,
                   timeout_s=-1.0)
    switch_controller(controller_name)

This keeps the class unit-testable on a PC without the robot SDK.
"""

from __future__ import annotations

import logging
import math
import time
from typing import List, NamedTuple, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

__all__ = [
    "LEG_JOINT_LIMITS",
    "LEG_JOINT_NAMES",
    "LegController",
    "LegControllerError",
    "LegState",
    "UnreachableLegPoseError",
]

LEG_JOINT_NAMES: Tuple[str, ...] = (
    "leg_joint1",
    "leg_joint2",
    "leg_joint3",
    "leg_joint4",
    "leg_joint5",
)

# URDF joint limits, in the same order as LEG_JOINT_NAMES.
LEG_JOINT_LIMITS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.9374),
    (0.0, 2.5847),
    (0.0, 2.3262),
    (-1.5906, 1.5906),
    (-0.1645, 0.1645),
)

# ---------------------------------------------------------------------------
# Planar leg model (sagittal plane).
#
# The three links are 0.45 m, 0.39 m and 0.224 m long.  In the zero pose the
# link angles are alpha10, alpha20 and alpha30; a positive joint value rotates
# the corresponding link clockwise in the x-z plane, hence the minus signs in
# forward_kinematics().
# ---------------------------------------------------------------------------
# URDF: base_link -> omni_chassis_leg_mount_link = (0.134, 0.0, 0.1325).
# Do not use the MuJoCo world value 0.166866 here: the model puts base_link at
# world z=0.034366, while get_transform() returns base_link-relative poses.
_A1_X = 0.134
_A1_Z = 0.1325
_LINK1_M = 0.45
_LINK2_M = 0.39
_LINK3_M = 0.224
_ALPHA10_RAD = 2.7565098986831513  # 157.936383 deg
_ALPHA20_RAD = -9.74358971275533e-05  # -0.005583 deg
_ALPHA30_RAD = 1.570698112509498  # 89.994373 deg

# Vertical distance from the leg-chain "EndEffector" frame to head_base_link
# when the torso is upright.  Measured in MuJoCo at the zero leg pose.
_HEAD_ABOVE_EE_M = 0.377110

# Keep commanded joints away from the hard limits so the next motion has room.
_JOINT_LIMIT_MARGIN_RAD = 0.01

# Search range for the pelvis (leg EndEffector) x position when solving IK.
_EE_X_MIN_M = -0.10
_EE_X_MAX_M = 0.40
_EE_X_COARSE_STEP_M = 0.005
_EE_X_FINE_STEP_M = 0.001
_EE_X_FINE_RADIUS_M = 0.01

# Weight of the "stay near the preferred pelvis x" term relative to the
# squared joint-distance term (rad^2).
_EE_X_COST_WEIGHT = 50.0

# Controller names from G1ControllerName; kept as strings so this module does
# not need the SDK.
_LEG_PVT_CTRL = "leg_pvt_ctrl"


class LegControllerError(RuntimeError):
    """A leg command or state read failed."""


class UnreachableLegPoseError(LegControllerError):
    """The requested height/pitch combination violates the leg model."""


class LegState(NamedTuple):
    """Measured torso state.

    Attributes:
        height_m: z coordinate of ``head_base_link`` in ``base_link``, metres.
        pitch_deg: forward lean of the torso, degrees.  Positive is forward.
    """

    height_m: float
    pitch_deg: float

    @property
    def pitch_rad(self) -> float:
        return math.radians(self.pitch_deg)


def _frange(start: float, stop: float, step: float) -> List[float]:
    """Inclusive float range built from integer steps (no drift)."""
    count = int(round((stop - start) / step))
    return [start + index * step for index in range(count + 1)]


class LegController:
    """Control leg height and torso pitch on a Galbot G1.

    Args:
        robot: An initialised robot object exposing the methods listed in the
            module docstring.
        height_duration_s: Duration passed to ``set_leg_height`` when the
            torso is upright, seconds.
        height_timeout_s: Blocking timeout for ``set_leg_height``, seconds.
        joint_speed_rad_s: Speed passed to ``set_joint_positions`` for the
            leg joints.
        joint_timeout_s: Minimum blocking timeout for leg joint moves.
        settle_s: Sleep after a motion (and before switching back from the
            height controller) to let the state settle.
        max_forward_pitch_deg: Software safety cap for forward pitch.
        max_backward_pitch_deg: Software safety cap for backward pitch
            (a positive magnitude).
        verify: Read the state back after each motion and retry once with a
            corrected target when the error is outside tolerance.
        position_tolerance_m: Accepted height error, metres.
        pitch_tolerance_deg: Accepted pitch error, degrees.
        retries: Number of corrective retries after a failed verification.
        max_pitch_bias_deg: Largest corrective pitch change that
            ``control_pitch`` will command after a failed verification.  A
            larger error means the measured state is implausible, so the
            controller refuses a corrective move.

    Notes:
        ``control_height`` uses the SDK height controller and does not bow
        the torso.

        The class does not enable the robot, release the emergency stop, or
        check that the workspace is clear.  Those are caller responsibilities.
    """

    HEIGHT_MIN_M = 0.96
    HEIGHT_MAX_M = 1.54

    def __init__(
        self,
        robot,
        *,
        height_duration_s: float = 4.0,
        height_timeout_s: float = 20.0,
        joint_speed_rad_s: float = 0.2,
        joint_timeout_s: float = 20.0,
        settle_s: float = 0.5,
        max_forward_pitch_deg: float = 45.0,
        max_backward_pitch_deg: float = 15.0,
        verify: bool = True,
        position_tolerance_m: float = 0.005,
        pitch_tolerance_deg: float = 0.5,
        retries: int = 1,
        max_pitch_bias_deg: float = 3.0,
    ) -> None:
        required = (
            "get_transform",
            "get_joint_positions",
            "set_joint_positions",
            "set_leg_height",
            "switch_controller",
        )
        missing = [name for name in required if not callable(getattr(robot, name, None))]
        if missing:
            raise TypeError(
                "robot is missing required method(s): " + ", ".join(missing)
            )

        self._robot = robot
        self._height_duration_s = self._finite_non_negative(
            height_duration_s, "height_duration_s"
        )
        if self._height_duration_s > 60.0:
            raise ValueError("height_duration_s must be in [0, 60]")
        self._height_timeout_s = self._finite_positive(
            height_timeout_s, "height_timeout_s"
        )
        self._joint_speed_rad_s = self._finite_positive(
            joint_speed_rad_s, "joint_speed_rad_s"
        )
        self._joint_timeout_s = self._finite_positive(
            joint_timeout_s, "joint_timeout_s"
        )
        self._settle_s = self._finite_non_negative(settle_s, "settle_s")
        self._max_forward_pitch_deg = self._finite_non_negative(
            max_forward_pitch_deg, "max_forward_pitch_deg"
        )
        self._max_backward_pitch_deg = self._finite_non_negative(
            max_backward_pitch_deg, "max_backward_pitch_deg"
        )
        self._position_tolerance_m = self._finite_positive(
            position_tolerance_m, "position_tolerance_m"
        )
        self._pitch_tolerance_deg = self._finite_positive(
            pitch_tolerance_deg, "pitch_tolerance_deg"
        )
        self._max_pitch_bias_deg = self._finite_positive(
            max_pitch_bias_deg, "max_pitch_bias_deg"
        )
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError("retries must be a non-negative integer")
        self._retries = retries
        self._verify = bool(verify)

        # True while the leg group is known to be owned by the height
        # controller and must be switched back before joint commands.
        self._height_mode = False

        self._height_noop_m = 0.001
        self._pitch_noop_deg = 0.05
        self._pitch_zero_eps_deg = 0.05

    # ------------------------------------------------------------------
    # Pure kinematics (usable without a robot, also covered by test.py)
    # ------------------------------------------------------------------
    @staticmethod
    def forward_kinematics(
        joint_positions: Sequence[float],
    ) -> Tuple[float, float, float]:
        """Return ``(ee_x_m, ee_z_m, pitch_rad)`` for the five leg joints.

        The result is the pose of the leg-chain EndEffector frame in
        ``base_link``.  ``pitch_rad`` is the torso pitch; it is exactly
        ``q1 - q2 + q3`` in the ideal model.  The planar model ignores
        ``q4``/``q5`` (torso yaw/roll); this controller commands them as zero.
        """
        q = LegController._five_joints(joint_positions)
        q1, q2, q3 = q[0], q[1], q[2]
        alpha1 = _ALPHA10_RAD - q1
        alpha2 = _ALPHA20_RAD - q1 + q2
        alpha3 = _ALPHA30_RAD - q1 + q2 - q3

        ee_x = (
            _A1_X
            + _LINK1_M * math.cos(alpha1)
            + _LINK2_M * math.cos(alpha2)
            + _LINK3_M * math.cos(alpha3)
        )
        ee_z = (
            _A1_Z
            + _LINK1_M * math.sin(alpha1)
            + _LINK2_M * math.sin(alpha2)
            + _LINK3_M * math.sin(alpha3)
        )
        pitch = q1 - q2 + q3
        return ee_x, ee_z, pitch

    @staticmethod
    def head_pose(
        joint_positions: Sequence[float],
    ) -> Tuple[float, float, float]:
        """Return ``(head_x_m, head_z_m, pitch_rad)`` for the five leg joints.

        ``head_z_m`` is the quantity also reported by ``set_leg_height`` and
        by ``get_transform("base_link", "head_base_link")``.  ``q4``/``q5``
        are ignored as in :meth:`forward_kinematics`.
        """
        ee_x, ee_z, pitch = LegController.forward_kinematics(joint_positions)
        head_x = ee_x + _HEAD_ABOVE_EE_M * math.sin(pitch)
        head_z = ee_z + _HEAD_ABOVE_EE_M * math.cos(pitch)
        return head_x, head_z, pitch

    @staticmethod
    def inverse_kinematics(
        ee_x_m: float,
        ee_z_m: float,
        pitch_rad: float,
    ) -> Optional[List[float]]:
        """Solve the planar leg IK for the first three joints.

        Args:
            ee_x_m: Desired leg EndEffector x in ``base_link``, metres.
            ee_z_m: Desired leg EndEffector z in ``base_link``, metres.
            pitch_rad: Desired torso pitch, radians.

        Returns:
            ``[q1, q2, q3]`` or ``None`` when the target is outside the
            two-link reach.  Joint limits are *not* applied here; use
            :meth:`solve_leg_joints` for that.
        """
        values = (ee_x_m, ee_z_m, pitch_rad)
        if not all(math.isfinite(float(value)) for value in values):
            return None

        alpha3 = _ALPHA30_RAD - float(pitch_rad)
        wrist_x = float(ee_x_m) - _LINK3_M * math.cos(alpha3)
        wrist_z = float(ee_z_m) - _LINK3_M * math.sin(alpha3)

        delta_x = wrist_x - _A1_X
        delta_z = wrist_z - _A1_Z
        distance_sq = delta_x * delta_x + delta_z * delta_z
        distance = math.sqrt(distance_sq)
        if distance < 1e-9:
            return None

        cosine = (distance_sq - _LINK1_M ** 2 - _LINK2_M ** 2) / (
            2.0 * _LINK1_M * _LINK2_M
        )
        if cosine < -1.0 - 1e-6 or cosine > 1.0 + 1e-6:
            return None
        cosine = max(-1.0, min(1.0, cosine))

        # The G1 leg uses the "elbow down" branch (beta negative).
        beta = -math.acos(cosine)
        gamma = math.atan2(
            _LINK2_M * math.sin(beta),
            _LINK1_M + _LINK2_M * math.cos(beta),
        )
        alpha1 = math.atan2(delta_z, delta_x) - gamma

        q1 = _ALPHA10_RAD - alpha1
        q2 = beta - (_ALPHA20_RAD - _ALPHA10_RAD)
        q3 = _ALPHA30_RAD - q1 + q2 - alpha3
        return [q1, q2, q3]

    @staticmethod
    def joints_within_limits(joint_positions: Sequence[float]) -> bool:
        """True when all five joints are finite and inside the soft limits."""
        try:
            q = LegController._five_joints(joint_positions)
        except (TypeError, ValueError):
            return False
        for value, (lower, upper) in zip(q, LEG_JOINT_LIMITS):
            if not math.isfinite(value):
                return False
            if value < lower + _JOINT_LIMIT_MARGIN_RAD:
                return False
            if value > upper - _JOINT_LIMIT_MARGIN_RAD:
                return False
        return True

    @staticmethod
    def solve_leg_joints(
        preferred_ee_x_m: float,
        ee_z_m: float,
        pitch_rad: float,
        current_joint_positions: Sequence[float],
    ) -> List[float]:
        """Find five leg joints for an EndEffector z and torso pitch.

        The EndEffector x is a free variable: the solver searches
        ``[-0.10, 0.40]`` m and prefers poses close to
        ``preferred_ee_x_m`` and to ``current_joint_positions``.  This is what
        makes a bow rotate the torso about the pelvis instead of forcing an
        unnatural pelvis translation.

        Args:
            preferred_ee_x_m: Preferred pelvis x, metres.
            ee_z_m: Required leg EndEffector z, metres.
            pitch_rad: Required torso pitch, radians.
            current_joint_positions: Current five leg joints, used as the
                continuity reference.

        Returns:
            A five-element list; ``q4`` and ``q5`` are always zero.

        Raises:
            UnreachableLegPoseError: No pose satisfies the target and the
                soft joint limits.
        """
        current = LegController._five_joints(current_joint_positions)
        if not math.isfinite(float(ee_z_m)) or not math.isfinite(float(pitch_rad)):
            raise UnreachableLegPoseError("target height/pitch is not finite")

        def evaluate(x_m: float) -> Optional[Tuple[float, List[float]]]:
            first_three = LegController.inverse_kinematics(x_m, ee_z_m, pitch_rad)
            if first_three is None:
                return None
            candidate = [first_three[0], first_three[1], first_three[2], 0.0, 0.0]
            if not LegController.joints_within_limits(candidate):
                return None
            cost = sum(
                (candidate[index] - current[index]) ** 2 for index in range(5)
            )
            cost += _EE_X_COST_WEIGHT * (x_m - float(preferred_ee_x_m)) ** 2
            return cost, candidate

        best_cost = math.inf
        best_x = float(preferred_ee_x_m)
        best_joints: Optional[List[float]] = None
        for x_m in _frange(_EE_X_MIN_M, _EE_X_MAX_M, _EE_X_COARSE_STEP_M):
            result = evaluate(x_m)
            if result is None:
                continue
            cost, candidate = result
            if cost < best_cost:
                best_cost = cost
                best_x = x_m
                best_joints = candidate

        if best_joints is None:
            raise UnreachableLegPoseError(
                "no feasible leg pose for ee_z={:.4f} m, pitch={:.2f} deg; "
                "try a different height or pitch".format(
                    ee_z_m, math.degrees(pitch_rad)
                )
            )

        fine_min = max(_EE_X_MIN_M, best_x - _EE_X_FINE_RADIUS_M)
        fine_max = min(_EE_X_MAX_M, best_x + _EE_X_FINE_RADIUS_M)
        for x_m in _frange(fine_min, fine_max, _EE_X_FINE_STEP_M):
            result = evaluate(x_m)
            if result is None:
                continue
            cost, candidate = result
            if cost < best_cost:
                best_cost = cost
                best_joints = candidate

        return list(best_joints)

    @staticmethod
    def _five_joints(joint_positions: Sequence[float]) -> List[float]:
        if joint_positions is None:
            raise ValueError("joint_positions is None")
        values = [float(value) for value in joint_positions]
        if len(values) == 3:
            values.extend([0.0, 0.0])
        if len(values) != 5:
            raise ValueError(
                "expected 5 leg joint values, got {}".format(len(values))
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("joint_positions contains a non-finite value")
        return values

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def read_state(self) -> LegState:
        """Read the current torso height and pitch from the robot.

        Height is the z component of ``head_base_link`` in ``base_link``.
        Pitch is the rotation about ``base_link`` y, positive forward.
        """
        try:
            result = self._robot.get_transform(
                "base_link", "head_base_link", timestamp_ns=0, timeout_ms=500
            )
        except Exception as exc:  # pragma: no cover - depends on SDK types
            raise LegControllerError(
                "get_transform(base_link, head_base_link) failed: {}".format(exc)
            ) from exc

        if isinstance(result, (tuple, list)) and len(result) == 2:
            pose = result[0]
        else:
            pose = result
        try:
            pose_length = len(pose)
        except TypeError as exc:
            raise LegControllerError(
                "get_transform returned an invalid pose: {!r}".format(result)
            ) from exc
        if pose_length < 7:
            raise LegControllerError(
                "get_transform returned an invalid pose: {!r}".format(result)
            )

        try:
            height_m = float(pose[2])
            qx, qy, qz, qw = (float(value) for value in pose[3:7])
        except (TypeError, ValueError) as exc:
            raise LegControllerError(
                "get_transform returned a non-numeric pose: {!r}".format(result)
            ) from exc

        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if not math.isfinite(norm) or norm < 1e-9:
            raise LegControllerError(
                "get_transform returned an invalid quaternion: {!r}".format(pose[3:7])
            )
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

        # Rotation matrix element r02 = 2(qx qz + qw qy).
        r02 = 2.0 * (qx * qz + qw * qy)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
        pitch_deg = math.degrees(math.atan2(r02, r22))

        if not math.isfinite(height_m) or not math.isfinite(pitch_deg):
            raise LegControllerError("measured leg state is not finite")
        return LegState(height_m=height_m, pitch_deg=pitch_deg)

    def control_height(self, height_m: float) -> LegState:
        """Raise or lower the legs to ``height_m`` (head_base_link z).

        The torso stays upright.  This method only changes head height through
        the SDK height controller.

        Raises:
            ValueError: ``height_m`` is outside ``[0.96, 1.54]``.
            LegControllerError: A robot command failed or verification failed.
        """
        target = self._validate_height(height_m)
        state = self.read_state()
        return self._control_height_upright(target, state, verify=self._verify)

    def reset_height(self) -> LegState:
        """Lower the head to the minimum height.  The torso is not pitched."""
        return self.control_height(self.HEIGHT_MIN_M)

    def control_pitch(self, pitch_deg: float) -> LegState:
        """Set the torso pitch.  Positive leans forward, in degrees."""
        target = self._validate_pitch(pitch_deg)
        state = self.read_state()
        if abs(state.pitch_deg - target) <= self._pitch_noop_deg:
            return state
        return self._control_pitch_absolute(target, state, verify=self._verify)

    def reset_pitch(self) -> LegState:
        """Return the torso to upright."""
        return self.control_pitch(0.0)

    # ------------------------------------------------------------------
    # Height control
    # ------------------------------------------------------------------
    def _control_height_upright(
        self, target_m: float, state: LegState, *, verify: bool
    ) -> LegState:
        if abs(state.height_m - target_m) <= self._height_noop_m:
            return state

        self._height_mode = True
        timeout_s = max(
            self._height_timeout_s, self._height_duration_s + 5.0
        )
        try:
            status = self._robot.set_leg_height(
                target_m,
                self._height_duration_s,
                True,
                timeout_s,
            )
        except Exception as exc:  # pragma: no cover - depends on SDK types
            raise LegControllerError(
                "set_leg_height({:.3f} m) raised: {}".format(target_m, exc)
            ) from exc
        self._require_success(status, "set_leg_height({:.3f} m)".format(target_m))
        LOGGER.info("set_leg_height(%.3f m) completed", target_m)
        # The official example waits briefly before switching back to PVT.
        self._settle()
        self._restore_pvt()

        measured = self.read_state()
        if not verify:
            return measured
        return self._verify_height(target_m, measured, self._control_height_upright)

    def _control_height_pitched(
        self, target_m: float, state: LegState, *, verify: bool
    ) -> LegState:
        pitch_rad = state.pitch_rad
        current_q = self._read_leg_joints()
        measured = self._move_to_head_pose(target_m, pitch_rad, current_q)

        if not verify:
            return measured
        for attempt in range(self._retries):
            height_ok = (
                abs(measured.height_m - target_m) <= self._position_tolerance_m
            )
            pitch_ok = (
                abs(measured.pitch_deg - math.degrees(pitch_rad))
                <= self._pitch_tolerance_deg
            )
            if height_ok and pitch_ok:
                return measured
            LOGGER.warning(
                "pitched height verification failed (attempt %d/%d): "
                "height=%.4f m target=%.4f m, pitch=%.3f deg target=%.3f deg",
                attempt + 1,
                self._retries,
                measured.height_m,
                target_m,
                measured.pitch_deg,
                math.degrees(pitch_rad),
            )
            current_q = self._read_leg_joints()
            measured = self._move_to_head_pose(target_m, pitch_rad, current_q)

        if (
            abs(measured.height_m - target_m) > self._position_tolerance_m
            or abs(measured.pitch_deg - math.degrees(pitch_rad))
            > self._pitch_tolerance_deg
        ):
            raise LegControllerError(
                "pitched height target {:.4f} m / {:.3f} deg not reached: "
                "measured {:.4f} m / {:.3f} deg".format(
                    target_m,
                    math.degrees(pitch_rad),
                    measured.height_m,
                    measured.pitch_deg,
                )
            )
        return measured

    def _verify_height(self, target_m: float, measured: LegState, move) -> LegState:
        for attempt in range(self._retries):
            if abs(measured.height_m - target_m) <= self._position_tolerance_m:
                return measured
            corrected = target_m - (measured.height_m - target_m)
            if not (self.HEIGHT_MIN_M <= corrected <= self.HEIGHT_MAX_M):
                break
            LOGGER.warning(
                "height verification failed (attempt %d/%d): measured %.4f m, "
                "target %.4f m, retrying with %.4f m",
                attempt + 1,
                self._retries,
                measured.height_m,
                target_m,
                corrected,
            )
            measured = move(corrected, measured, verify=False)

        if abs(measured.height_m - target_m) > self._position_tolerance_m:
            raise LegControllerError(
                "height target {:.4f} m not reached: measured {:.4f} m".format(
                    target_m, measured.height_m
                )
            )
        return measured

    def _move_to_head_pose(
        self, target_height_m: float, pitch_rad: float, current_q: Sequence[float]
    ) -> LegState:
        ee_z = target_height_m - _HEAD_ABOVE_EE_M * math.cos(pitch_rad)
        ee_x, _, _ = self.forward_kinematics(current_q)
        return self._move_to_ee_pose(ee_x, ee_z, pitch_rad, current_q)

    # ------------------------------------------------------------------
    # Pitch control
    # ------------------------------------------------------------------
    def _control_pitch_absolute(
        self, target_deg: float, state: LegState, *, verify: bool
    ) -> LegState:
        target_rad = math.radians(target_deg)
        # Keep the pelvis at its current height while the torso rotates.
        ee_z = state.height_m - _HEAD_ABOVE_EE_M * math.cos(state.pitch_rad)
        current_q = self._read_leg_joints()
        ee_x, _, _ = self.forward_kinematics(current_q)
        model_target_rad = target_rad
        measured = self._move_to_ee_pose(ee_x, ee_z, model_target_rad, current_q)

        if not verify:
            return measured
        for attempt in range(self._retries):
            error_rad = target_rad - measured.pitch_rad
            error_deg = math.degrees(error_rad)
            if abs(error_deg) <= self._pitch_tolerance_deg:
                return measured
            if abs(error_deg) > self._max_pitch_bias_deg:
                raise LegControllerError(
                    "pitch verification error {:.3f} deg exceeds the {:.3f} deg "
                    "correction limit; refusing a corrective move (measured "
                    "{:.3f} deg, target {:.3f} deg)".format(
                        error_deg,
                        self._max_pitch_bias_deg,
                        measured.pitch_deg,
                        target_deg,
                    )
                )
            LOGGER.warning(
                "pitch verification failed (attempt %d/%d): measured %.3f deg, "
                "target %.3f deg; correcting the leg target by %+.3f deg",
                attempt + 1,
                self._retries,
                measured.pitch_deg,
                target_deg,
                error_deg,
            )
            current_q = self._read_leg_joints()
            ee_x, _, _ = self.forward_kinematics(current_q)
            model_target_rad += error_rad
            measured = self._move_to_ee_pose(
                ee_x, ee_z, model_target_rad, current_q
            )

        if abs(measured.pitch_deg - target_deg) > self._pitch_tolerance_deg:
            raise LegControllerError(
                "pitch target {:.3f} deg not reached: measured {:.3f} deg".format(
                    target_deg, measured.pitch_deg
                )
            )
        return measured

    def _move_to_ee_pose(
        self,
        preferred_ee_x_m: float,
        ee_z_m: float,
        pitch_rad: float,
        current_q: Sequence[float],
    ) -> LegState:
        joints = self.solve_leg_joints(
            preferred_ee_x_m, ee_z_m, pitch_rad, current_q
        )
        LOGGER.info(
            "leg pose target: ee_z=%.4f m pitch=%.3f deg q=%s",
            ee_z_m,
            math.degrees(pitch_rad),
            ", ".join("{:.4f}".format(value) for value in joints),
        )
        self._apply_leg_joints(joints, current_q)
        return self.read_state()

    # ------------------------------------------------------------------
    # Robot command helpers
    # ------------------------------------------------------------------
    def _read_leg_joints(self) -> List[float]:
        try:
            raw = self._robot.get_joint_positions([], list(LEG_JOINT_NAMES))
        except Exception as exc:  # pragma: no cover - depends on SDK types
            raise LegControllerError(
                "get_joint_positions(leg) raised: {}".format(exc)
            ) from exc
        try:
            q = self._five_joints(raw)
        except ValueError as exc:
            raise LegControllerError(
                "get_joint_positions(leg) returned an invalid value: {!r}".format(
                    raw
                )
            ) from exc
        return q

    def _apply_leg_joints(
        self, joints: Sequence[float], current_q: Sequence[float]
    ) -> None:
        target = self._five_joints(joints)
        current = self._five_joints(current_q)
        if not self.joints_within_limits(target):
            raise UnreachableLegPoseError(
                "refusing to command leg joints outside the soft limits: "
                + ", ".join("{:.4f}".format(value) for value in target)
            )

        if self._height_mode:
            self._restore_pvt()

        max_delta = max(abs(a - b) for a, b in zip(target, current))
        timeout = max(
            self._joint_timeout_s,
            max_delta / self._joint_speed_rad_s * 1.5 + 5.0,
        )
        timeout = min(timeout, 120.0)

        try:
            status = self._robot.set_joint_positions(
                list(target),
                [],
                list(LEG_JOINT_NAMES),
                True,
                self._joint_speed_rad_s,
                timeout,
            )
        except Exception as exc:  # pragma: no cover - depends on SDK types
            raise LegControllerError(
                "set_joint_positions(leg) raised: {}".format(exc)
            ) from exc
        self._require_success(status, "set_joint_positions(leg)")
        self._height_mode = False
        self._settle()

    def _restore_pvt(self) -> None:
        try:
            status = self._robot.switch_controller(_LEG_PVT_CTRL)
        except Exception as exc:  # pragma: no cover - depends on SDK types
            raise LegControllerError(
                "switch_controller({!r}) raised: {}; the leg group may still be "
                "under height control".format(_LEG_PVT_CTRL, exc)
            ) from exc
        self._require_success(
            status,
            "switch_controller({!r}); the leg group may still be under height "
            "control".format(_LEG_PVT_CTRL),
        )
        self._height_mode = False

    def _settle(self) -> None:
        if self._settle_s > 0.0:
            time.sleep(self._settle_s)

    @staticmethod
    def _require_success(status, action: str) -> None:
        if LegController._status_is_success(status):
            return
        raise LegControllerError(
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

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _finite_positive(value: float, name: str) -> float:
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError("{} must be a finite positive number".format(name))
        return number

    @staticmethod
    def _finite_non_negative(value: float, name: str) -> float:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(
                "{} must be a finite non-negative number".format(name)
            )
        return number

    def _validate_height(self, height_m: float) -> float:
        value = float(height_m)
        if not math.isfinite(value):
            raise ValueError("height_m must be finite")
        if value < self.HEIGHT_MIN_M or value > self.HEIGHT_MAX_M:
            raise ValueError(
                "height_m={:.4f} is outside the supported range [{:.2f}, {:.2f}]".format(
                    value, self.HEIGHT_MIN_M, self.HEIGHT_MAX_M
                )
            )
        return value

    def _validate_pitch(self, pitch_deg: float) -> float:
        value = float(pitch_deg)
        if not math.isfinite(value):
            raise ValueError("pitch_deg must be finite")
        if value > self._max_forward_pitch_deg:
            raise ValueError(
                "pitch_deg={:.3f} exceeds the forward cap of {:.3f} deg".format(
                    value, self._max_forward_pitch_deg
                )
            )
        if value < -self._max_backward_pitch_deg:
            raise ValueError(
                "pitch_deg={:.3f} exceeds the backward cap of -{:.3f} deg".format(
                    value, self._max_backward_pitch_deg
                )
            )
        return value
