from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .hashing import canonical_json_bytes, payload_sha256, sha256_array, sha256_bytes
from .entry_features import CONTINUOUS_FEATURES, FEATURE_NAMES
from .signatures import request_signature_contract


CONTINUOUS_INDICES = np.asarray(
    [FEATURE_NAMES.index(name) for name in CONTINUOUS_FEATURES], dtype=np.int64
)


def successful_endpoint_rows(motion_row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        endpoint
        for endpoint in motion_row["endpoints"]
        if len(endpoint["replays"]) == 3
        and all(bool(replay["success"]) for replay in endpoint["replays"])
    ]


def _cell_for_endpoint(
    endpoint: Mapping[str, Any], scales: np.ndarray, radius: float
) -> dict[str, Any]:
    center = np.asarray(endpoint["replays"][0]["entry"]["vector"], dtype=np.float64)
    lower = center - scales * radius
    upper = center + scales * radius
    for name in ("left_contact_stability", "right_contact_stability"):
        index = FEATURE_NAMES.index(name)
        lower[index] = upper[index] = 1.0
    history = FEATURE_NAMES.index("history_step_count")
    lower[history] = upper[history] = 32.0
    cell = {
        "cell_id": endpoint["endpoint_id"] + "::LOCAL_CELL_SCMM",
        "center": center.tolist(),
        "axis_lower": lower.tolist(),
        "axis_upper": upper.tolist(),
        "diagonal_covariance": (scales ** 2).tolist(),
        "mahalanobis_scale": scales.tolist(),
        "normalized_rms_radius": radius,
        "safety_margins": {
            "normalized_axis_margin_at_center": radius,
            "normalized_mahalanobis_margin_at_center": radius,
            "all_three_build_replays_safe": all(
                bool(row["success"]) for row in endpoint["replays"]
            ),
        },
        "endpoint_provenance": {
            "endpoint_id": endpoint["endpoint_id"],
            "endpoint_type": endpoint["endpoint_type"],
            "source_kind": endpoint["source_kind"],
        },
        "build_replay_evidence": [
            {
                "replay": int(row["replay"]),
                "success": bool(row["success"]),
                "entry_sha256": sha256_bytes(canonical_json_bytes(row["entry"])),
                "execution_sha256": sha256_bytes(
                    canonical_json_bytes(row["execution"])
                ),
            }
            for row in endpoint["replays"]
        ],
    }
    cell["sha256"] = sha256_bytes(canonical_json_bytes(cell))
    return cell


def certificate_membership(
    vector: Sequence[float], certificate: Mapping[str, Any]
) -> dict[str, Any]:
    point = np.asarray(vector, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for cell in certificate.get("local_cells", []):
        lower = np.asarray(cell["axis_lower"], dtype=np.float64)
        upper = np.asarray(cell["axis_upper"], dtype=np.float64)
        center = np.asarray(cell["center"], dtype=np.float64)
        scale = np.asarray(cell["mahalanobis_scale"], dtype=np.float64)
        axis_inside = bool(np.all((point >= lower) & (point <= upper)))
        z = (point[CONTINUOUS_INDICES] - center[CONTINUOUS_INDICES]) / scale[
            CONTINUOUS_INDICES
        ]
        distance = float(np.sqrt(np.mean(z ** 2)))
        inside = bool(axis_inside and distance <= float(cell["normalized_rms_radius"]))
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "axis_inside": axis_inside,
                "distance": distance,
                "inside": inside,
            }
        )
    nearest = min(rows, key=lambda row: row["distance"], default=None)
    return {
        "inside": any(row["inside"] for row in rows),
        "nearest": nearest,
        "cells": rows,
    }


def build_certificate(
    package: Mapping[str, Any],
    reference: np.ndarray,
    motion_build: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    successful = successful_endpoint_rows(motion_build)
    scales = np.asarray(geometry["shared_feature_scales"], dtype=np.float64)
    radius = float(geometry["normalized_rms_radius"])
    cells = [_cell_for_endpoint(endpoint, scales, radius) for endpoint in successful]
    if cells:
        global_lower = np.min(
            np.stack([np.asarray(cell["axis_lower"]) for cell in cells]), axis=0
        )
        global_upper = np.max(
            np.stack([np.asarray(cell["axis_upper"]) for cell in cells]), axis=0
        )
        widths = np.maximum(global_upper - global_lower, 0.0)
        positive = widths > 0.0
        log_volume = (
            float(np.sum(np.log(widths[positive]))) if np.any(positive) else None
        )
    else:
        global_lower = global_upper = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        positive = np.zeros(len(FEATURE_NAMES), dtype=bool)
        log_volume = None
    certificate: dict[str, Any] = {
        "schema_version": 3,
        "status": "MOTION_CERTIFICATE",
        "motion_id": package["motion_id"],
        "representation": "FROZEN_V2_CERTIFIED_LOCAL_CELL_UNION_ARCHITECTURE",
        "coordinate_origin_contract": "THIS_MOTION_ORIGINAL_UPSTREAM_CERTIFIED_ENTRY",
        "original_entry_center_qpos": np.asarray(
            reference[0], dtype=np.float32
        ).tolist(),
        "original_entry_center_qpos_sha256": sha256_array(
            np.asarray(reference[0], dtype=np.float32)
        ),
        "feature_names": list(FEATURE_NAMES),
        "continuous_feature_names": list(CONTINUOUS_FEATURES),
        "local_cells": cells,
        "local_cell_count": len(cells),
        "global_axis_lower": global_lower.tolist(),
        "global_axis_upper": global_upper.tolist(),
        "geometry_quality": {
            "nonzero_dimension_count": int(np.sum(positive)),
            "log_hypervolume_over_nonzero_dimensions": log_volume,
            "minimum_normalized_safety_margin": float(
                geometry["minimum_safety_margin_normalized"]
            ),
        },
        "signature_contract": request_signature_contract(package),
        "hard_constraints": {
            "finite": True,
            "quaternion_valid": True,
            "stable_double_support": True,
            "history_ready": True,
            "history_min_real_frames": 32,
            "body_tilt_max_deg": 20.0,
            "hard_limit_margin_min_rad": 0.01,
            "slide_margin_min_m": 0.0,
            "penetration_margin_min_m": 0.0,
            "lower_qvel_rms_max": 0.15,
            "upper_qvel_rms_max": 0.25,
            "root_linear_velocity_norm_max": 0.08,
            "root_angular_velocity_norm_max": 0.25,
        },
        "build_endpoint_ids": [row["endpoint_id"] for row in successful],
        "runtime_rollout_allowed_for_membership": False,
        "empty_certificate_rejects_all": not bool(cells),
    }
    acceptance: list[dict[str, Any]] = []
    for endpoint in motion_build["endpoints"]:
        for replay in endpoint["replays"]:
            membership = certificate_membership(replay["entry"]["vector"], certificate)
            acceptance.append(
                {
                    "endpoint_id": endpoint["endpoint_id"],
                    "endpoint_type": endpoint["endpoint_type"],
                    "replay": int(replay["replay"]),
                    "accepted": bool(membership["inside"]),
                    "execution_success": bool(replay["success"]),
                }
            )
    loo_rows: list[dict[str, Any]] = []
    for endpoint in successful:
        held = np.asarray(endpoint["replays"][0]["entry"]["vector"], dtype=np.float64)
        reduced = dict(certificate)
        reduced["local_cells"] = [
            cell
            for cell in cells
            if cell["endpoint_provenance"]["endpoint_id"] != endpoint["endpoint_id"]
        ]
        check = certificate_membership(held, reduced)
        loo_rows.append(
            {"held_endpoint_id": endpoint["endpoint_id"], "pass": bool(check["inside"])}
        )
    certificate["build_validation"] = {
        "accepted_replays": int(sum(row["accepted"] for row in acceptance)),
        "total_replays": len(acceptance),
        "all_accepted_build_replays_success": all(
            (not row["accepted"]) or row["execution_success"] for row in acceptance
        ),
        "successful_build_replays_accepted": int(
            sum(row["accepted"] and row["execution_success"] for row in acceptance)
        ),
        "successful_build_replays": int(
            sum(row["execution_success"] for row in acceptance)
        ),
        "leave_one_build_endpoint_out": loo_rows,
        "leave_one_build_endpoint_out_pass": bool(
            loo_rows and all(row["pass"] for row in loo_rows)
        ),
    }
    certificate["payload_sha256"] = payload_sha256(
        certificate, excluded=("payload_sha256",)
    )
    return certificate


def entry_domain_checks(
    target: Mapping[str, Any],
    package: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, bool]:
    metrics = target["entry_domain_metrics"]
    upstream = package["upstream_certification"]
    return {
        "joint_qvel": float(metrics["entry_joint_qvel_rms_estimate"])
        <= float(thresholds["entry_joint_qvel_rms_estimate_max"]),
        "root_velocity": float(metrics["entry_root_linear_velocity_norm_estimate"])
        <= float(thresholds["entry_root_linear_velocity_norm_estimate_max"]),
        "root_tilt": float(metrics["entry_root_tilt_deg"])
        <= float(thresholds["entry_root_tilt_deg_max"]),
        "contact_class": bool(target["upstream_stable_double_support"]),
        "root_height": float(metrics["entry_root_height_delta_to_neutral_m"])
        <= float(thresholds["entry_root_height_delta_to_neutral_m_max"]),
        "lower_body_posture": float(metrics["entry_lower_body_l2_to_neutral_rad"])
        <= float(thresholds["entry_lower_body_l2_to_neutral_rad_max"]),
        "full_joint_domain": float(metrics["entry_full_joint_l2_to_neutral_rad"])
        <= float(thresholds["entry_full_joint_l2_to_neutral_rad_max"]),
        "hard_limit_margin": bool(upstream["holo_safety_pass"]),
        "support_margin": bool(
            upstream["holo_safety_pass"]
            and float(upstream["support_slide_max_m"]) <= 0.150
            and float(upstream["sole_penetration_max_m"]) <= 0.020
        ),
    }


def apply_frozen_compatibility(
    target: Mapping[str, Any],
    motion: Mapping[str, Any],
    certificate: Mapping[str, Any],
    package: Mapping[str, Any],
    frozen_rule: Mapping[str, Any],
) -> dict[str, Any]:
    successful = successful_endpoint_rows(motion)
    semantic_checks = {
        "low_dynamic_standing": target["task_family"] == "T5_ARM_GESTURE",
        "stable_double_support": bool(target["upstream_stable_double_support"]),
        "no_flight": bool(target["upstream_no_flight"]),
        "no_dynamic_single_support": bool(target["upstream_no_dynamic_single_support"]),
        "fixed_station_scope": bool(target["upstream_fixed_station_scope"]),
        "no_momentum_dependent_entry": True,
    }
    entry_checks = entry_domain_checks(
        target, package, frozen_rule["entry_domain_thresholds"]
    )
    source_classes = {str(endpoint["source_kind"]) for endpoint in successful}
    build_checks = {
        "at_least_five_of_six_endpoints_three_of_three": len(successful) >= 5,
        "all_accepted_build_replays_success": bool(
            certificate["build_validation"]["all_accepted_build_replays_success"]
        ),
        "catastrophic_failures_zero": int(motion["catastrophic_failures"]) == 0,
        "at_least_three_endpoint_sources": len(source_classes) >= 3,
        "teacher_calls_zero": int(motion["teacher_calls"]) == 0,
    }
    geometry_checks = {
        "certificate_not_empty": int(certificate["local_cell_count"]) > 0,
        "at_least_five_local_cells": int(certificate["local_cell_count"]) >= 5,
        "nonzero_volume": int(
            certificate["geometry_quality"]["nonzero_dimension_count"]
        )
        >= 1,
        "positive_safety_margin": float(
            certificate["geometry_quality"]["minimum_normalized_safety_margin"]
        )
        > 0.0,
        "leave_one_build_endpoint_out": bool(
            certificate["build_validation"]["leave_one_build_endpoint_out_pass"]
        ),
    }
    sections = {
        "semantic": semantic_checks,
        "entry": entry_checks,
        "build": build_checks,
        "geometry": geometry_checks,
    }
    compatible = all(all(checks.values()) for checks in sections.values())
    failures = [
        f"{section}::{name}"
        for section, checks in sections.items()
        for name, passed in checks.items()
        if not passed
    ]
    return {
        "motion_id": target["motion_id"],
        "semantic_label": target["semantic_label"],
        "classification": "COMPATIBLE_WITH_SHARED_NEUTRAL"
        if compatible
        else "INCOMPATIBLE_WITH_SHARED_NEUTRAL",
        "route": "CERTIFICATE_ELIGIBLE"
        if compatible
        else "FRESH_OMG_FALLBACK_REQUIRED",
        "semantic_checks": semantic_checks,
        "entry_checks": entry_checks,
        "build_checks": build_checks,
        "geometry_checks": geometry_checks,
        "successful_build_endpoint_count": len(successful),
        "successful_source_classes": sorted(source_classes),
        "failure_reasons": failures,
    }


def request_from_entry(
    entry: Mapping[str, Any],
    package: Mapping[str, Any],
    source: Mapping[str, Any],
    stabilizer: Mapping[str, Any] | None,
) -> dict[str, Any]:
    values = entry["values"]
    stabilizer_catastrophe = bool(
        stabilizer and stabilizer.get("catastrophic_failure", False)
    )
    return {
        "vector": list(entry["vector"]),
        "signature_contract": request_signature_contract(package),
        "physical_checks": {
            "finite": bool(entry["finite"]),
            "quaternion_valid": bool(entry["quaternion_valid"]),
            "no_fall": bool(
                source.get("predecessor_safe_complete", False)
                and not stabilizer_catastrophe
                and float(entry["body_tilt_degrees"]) <= 45.0
            ),
            "no_hard_joint_limit": float(values["hard_limit_margin_rad"]) >= 0.01,
            "no_penetration": float(values["penetration_margin_m"]) >= 0.0,
            "support_slide_pass": float(values["slide_margin_m"]) >= 0.0,
            "stable_double_support": bool(entry["stable_double_support"]),
            "real_history_ready": bool(
                entry["history_ready"] and entry["history_length_valid"]
            ),
        },
        "feature_extraction_source": "CURRENT_REAL_HOLOMOTION_STATE",
    }


def evaluate_membership(
    request: Mapping[str, Any], certificate: Mapping[str, Any]
) -> dict[str, Any]:
    signature = request.get("signature_contract", {})
    signature_pass = signature == certificate["signature_contract"]
    physical_checks = {
        str(key): bool(value)
        for key, value in request.get("physical_checks", {}).items()
    }
    physical_pass = bool(physical_checks and all(physical_checks.values()))
    membership = certificate_membership(request["vector"], certificate)
    accepted = bool(signature_pass and physical_pass and membership["inside"])
    if not signature_pass:
        decision = "SIGNATURE_REJECT"
    elif not physical_pass:
        decision = "HARD_PHYSICAL_REJECT"
    elif not membership["inside"]:
        decision = "OUTSIDE_LOCAL_CELLS_REJECT"
    else:
        decision = "CERTIFICATE_ACCEPT_CACHE"
    return {
        "accepted": accepted,
        "decision": decision,
        "signature_pass": signature_pass,
        "physical_pass": physical_pass,
        "physical_checks": physical_checks,
        "membership": membership,
        "runtime_rollout_count": 0,
        "cache_execution_started": False,
    }
