from __future__ import annotations

from typing import Any, Mapping

from .hashing import canonical_json_bytes, sha256_bytes


def request_signature_contract(
    package: Mapping[str, Any], *, control_fps: float = 50.0
) -> dict[str, Any]:
    return {
        **dict(package["signatures"]),
        "motion_id": package["motion_id"],
        "task_specification_sha256": sha256_bytes(
            canonical_json_bytes(package["task_specification"])
        ),
        "cache_manifest_sha256": package["manifest_sha256"],
        "cache_file_sha256": package["cache_file_sha256"],
        "contact_class": "STABLE_DOUBLE_SUPPORT",
        "history_ready_schema": "REAL_HOLOMOTION_KV32_OBS1_ACTION1",
        "reference_fps": float(package["trajectory"]["fps"]),
        "control_fps": float(control_fps),
        "control_period_s": 1.0 / float(control_fps),
    }
