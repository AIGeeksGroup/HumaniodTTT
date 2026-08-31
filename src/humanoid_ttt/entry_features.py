from __future__ import annotations

import math
from typing import Any

import numpy as np


FEATURE_NAMES = tuple(
    [f"joint_offset_{index:02d}" for index in range(29)]
    + [
        "lower_qvel_rms",
        "upper_qvel_rms",
        "root_height_offset",
        "root_roll_offset",
        "root_pitch_offset",
        "root_yaw_offset",
        "root_linear_velocity_norm",
        "root_angular_velocity_norm",
        "left_contact_stability",
        "right_contact_stability",
        "com_support_margin_m",
        "hard_limit_margin_rad",
        "slide_margin_m",
        "penetration_margin_m",
        "history_step_count",
        "history_last_action_rms",
    ]
)

CONTINUOUS_FEATURES = tuple(
    name
    for name in FEATURE_NAMES
    if name
    not in {"left_contact_stability", "right_contact_stability", "history_step_count",}
)


def quaternion_rpy(q_wxyz: np.ndarray) -> tuple[float, float, float]:
    q = np.asarray(q_wxyz, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q.tolist()
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return roll, pitch, math.atan2(siny, cosy)


def angle_delta(left: float, right: float) -> float:
    return float(math.atan2(math.sin(left - right), math.cos(left - right)))


def _support_margin(runtime: Any) -> float:
    state = runtime.sequence
    if state is None:
        return -1.0
    data = state["data"]
    body_ids = state["body_ids"]
    pelvis = np.asarray(data.subtree_com[body_ids["pelvis"], :2], dtype=np.float64)
    feet = np.stack(
        [
            np.asarray(data.xpos[body_ids["left_foot"], :2], dtype=np.float64),
            np.asarray(data.xpos[body_ids["right_foot"], :2], dtype=np.float64),
        ]
    )
    lower = np.min(feet, axis=0) - np.asarray([0.06, 0.04])
    upper = np.max(feet, axis=0) + np.asarray([0.06, 0.04])
    return float(np.min(np.concatenate([pelvis - lower, upper - pelvis])))


def capture_entry_features(
    runtime: Any,
    canonical: np.ndarray,
    *,
    slide_m: float = 0.0,
    penetration_m: float = 0.0,
) -> dict[str, Any]:
    qpos = runtime.current_qpos().astype(np.float64)
    qvel = runtime.current_qvel().astype(np.float64)
    target = np.asarray(canonical, dtype=np.float64).reshape(36)
    snapshot = runtime.recoverability_snapshot()
    audit = runtime.audit_state()
    state = runtime.sequence
    if state is None:
        raise RuntimeError("Entry features require an active sequence")
    contacts, _, _, _ = runtime._contacts(None)
    lower, upper = runtime.joint_position_limits()
    joint_margin = np.minimum(qpos[7:] - lower, upper - qpos[7:])
    roll, pitch, yaw = quaternion_rpy(qpos[3:7])
    t_roll, t_pitch, t_yaw = quaternion_rpy(target[3:7])
    last_action = np.asarray(state["last_action"], dtype=np.float64)
    values: dict[str, float] = {
        **{
            f"joint_offset_{index:02d}": float(qpos[7 + index] - target[7 + index])
            for index in range(29)
        },
        "lower_qvel_rms": float(np.sqrt(np.mean(qvel[6:21] ** 2))),
        "upper_qvel_rms": float(np.sqrt(np.mean(qvel[21:35] ** 2))),
        "root_height_offset": float(qpos[2] - target[2]),
        "root_roll_offset": angle_delta(roll, t_roll),
        "root_pitch_offset": angle_delta(pitch, t_pitch),
        "root_yaw_offset": angle_delta(yaw, t_yaw),
        "root_linear_velocity_norm": float(np.linalg.norm(qvel[:3])),
        "root_angular_velocity_norm": float(np.linalg.norm(qvel[3:6])),
        "left_contact_stability": float(contacts[0]),
        "right_contact_stability": float(contacts[1]),
        "com_support_margin_m": _support_margin(runtime),
        "hard_limit_margin_rad": float(np.min(joint_margin)),
        "slide_margin_m": float(0.150 - max(0.0, slide_m)),
        "penetration_margin_m": float(0.020 - max(0.0, penetration_m)),
        # Saturate the history summary at 32 tracker steps.
        "history_step_count": float(min(audit["tracker_step_idx"], 32)),
        "history_last_action_rms": float(np.sqrt(np.mean(last_action ** 2))),
    }
    vector = np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float64)
    return {
        "feature_names": list(FEATURE_NAMES),
        "values": values,
        "vector": vector,
        "finite": bool(np.isfinite(vector).all()),
        "quaternion_valid": bool(abs(float(np.linalg.norm(qpos[3:7])) - 1.0) <= 1e-3),
        "stable_double_support": bool(np.all(contacts > 0.5)),
        "illegal_contact": False,
        "history_ready": bool(
            audit["tracker_step_idx"] >= 32 and audit["tracker_kv_sha256"]
        ),
        "history_length_valid": bool(audit["tracker_step_idx"] >= 32),
        "filter_delay_recurrent_stable": bool(audit["tracker_step_idx"] >= 32),
        "phase": "FROZEN_STABLE_STAND",
        "body_tilt_degrees": float(snapshot["body_tilt_degrees"]),
        "audit": audit,
    }
