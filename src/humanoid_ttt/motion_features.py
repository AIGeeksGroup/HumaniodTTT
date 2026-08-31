from __future__ import annotations

import numpy as np


def descriptor_16(qpos: np.ndarray) -> np.ndarray:
    """Compute a 16D motion descriptor from a 60-frame G1 reference at 30 Hz."""
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.shape != (60, 36):
        raise ValueError(f"expected (60,36), got {qpos.shape}")
    joints = qpos[:, 7:]
    excursion = np.percentile(joints, 95, axis=0) - np.percentile(joints, 5, axis=0)
    velocity = np.diff(joints, axis=0) * 30.0
    left, right, lower, waist = (
        list(range(15, 19)),
        list(range(22, 26)),
        list(range(12)),
        list(range(12, 15)),
    )
    group_exc = [
        float(np.sum(np.abs(excursion[index]))) for index in (left, right, lower, waist)
    ]
    group_vel = [
        float(np.sqrt(np.mean(velocity[:, index] ** 2)))
        for index in (left, right, lower)
    ]
    quat = qpos[:, 3:7]
    yaw = np.unwrap(
        np.arctan2(
            2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
        )
    )
    frame_energy = np.sum(np.diff(joints, axis=0) ** 2, axis=1)
    t = np.linspace(1 / 59, 1.0, 59)
    temporal_centroid = float(
        np.sum(t * frame_energy) / max(np.sum(frame_energy), 1e-12)
    )
    balance = 1.0 - abs(group_exc[0] - group_exc[1]) / max(
        group_exc[0] + group_exc[1], 1e-8
    )
    values = [
        float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2])),
        float(abs(yaw[-1] - yaw[0])),
        float(np.mean(qpos[:, 2])),
        float(np.ptp(qpos[:, 2])),
        float(np.mean(np.abs(excursion))),
        *group_exc,
        float(np.sqrt(np.mean(velocity ** 2))),
        *group_vel,
        float(np.log1p(np.sum(frame_energy))),
        float(balance),
        temporal_centroid,
    ]
    if len(values) != 16:
        raise AssertionError(len(values))
    return np.asarray(values, dtype=np.float32)
