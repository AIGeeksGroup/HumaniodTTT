from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def sha256_array(value: np.ndarray | torch.Tensor) -> str:
    array = (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LatentBasis:
    values: np.ndarray
    seed: int
    shape: tuple[int, int, int]
    dimension: int

    @classmethod
    def create(
        cls, shape: tuple[int, int, int], dimension: int, seed: int
    ) -> "LatentBasis":
        size = int(np.prod(shape))
        if not 1 <= int(dimension) <= 16 or dimension > size:
            raise ValueError(
                "latent dimension must be in [1,16] and no larger than flattened noise"
            )
        rng = np.random.default_rng(int(seed))
        matrix = rng.standard_normal((size, int(dimension)), dtype=np.float64)
        q, _ = np.linalg.qr(matrix, mode="reduced")
        # RMS-normalized columns make alpha interpretable as per-element noise scale.
        q *= np.sqrt(float(size))
        return cls(q.astype(np.float32), int(seed), tuple(shape), int(dimension))

    @property
    def sha256(self) -> str:
        return sha256_array(self.values)

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            values=self.values,
            seed=np.asarray([self.seed], np.int64),
            shape=np.asarray(self.shape, np.int64),
            dimension=np.asarray([self.dimension], np.int64),
        )

    @classmethod
    def load(cls, path) -> "LatentBasis":
        with np.load(path) as data:
            return cls(
                np.asarray(data["values"], np.float32),
                int(data["seed"][0]),
                tuple(map(int, data["shape"])),
                int(data["dimension"][0]),
            )


@dataclass(frozen=True)
class LatentSteering:
    basis: LatentBasis
    alpha: float
    bound: float = 1.0

    def apply(
        self, base_noise: torch.Tensor, z: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        action = np.asarray(
            z.detach().cpu() if isinstance(z, torch.Tensor) else z, dtype=np.float32
        )
        if action.shape != (self.basis.dimension,) or not np.isfinite(action).all():
            raise ValueError(f"z must be finite ({self.basis.dimension},)")
        bounded = np.clip(action, -self.bound, self.bound)
        if np.count_nonzero(bounded) == 0:
            # Preserve the input dtype and bits for zero steering.
            return base_noise.clone()
        delta = self.basis.values @ bounded
        delta = delta.reshape(self.basis.shape)
        return base_noise + torch.from_numpy(delta).to(base_noise) * float(self.alpha)

    def manifest(self) -> dict[str, Any]:
        return {
            "interface": "LOW_RANK_INITIAL_NOISE_STEERING",
            "noise_shape": list(self.basis.shape),
            "dimension": self.basis.dimension,
            "basis_seed": self.basis.seed,
            "basis_sha256": self.basis.sha256,
            "basis_column_l2_norm": float(np.sqrt(np.prod(self.basis.shape))),
            "alpha": float(self.alpha),
            "z_bounds": [-float(self.bound), float(self.bound)],
            "zero_path_arithmetic": "NONE_EXACT_CLONE",
        }


STATE_NAMES = (
    "episode_fraction",
    "remaining_budget_fraction",
    "previous_reward",
    "previous_task_score",
    "previous_tracking_score",
    "previous_safety_pass",
    "previous_motion_amplitude",
    "previous_root_stability",
    "previous_foot_slide",
    "best_reward_so_far",
    "rolling_reward_mean",
    "rolling_reward_trend",
    "previous_failure_safety",
    "previous_failure_degenerate",
    "previous_failure_task",
    "last_update_accepted",
) + tuple(f"previous_latent_{index}" for index in range(8))


class CompactState:
    """Encode recent adaptation feedback and the previous 8D action as 24 features."""

    dimension = len(STATE_NAMES)

    def __init__(self, budget: int, latent_dim: int = 8):
        if latent_dim != 8:
            raise ValueError("frozen compact schema is defined for 8D latent")
        self.budget = int(budget)
        self.rewards: list[float] = []
        self.previous_z = np.zeros(latent_dim, np.float32)
        self.previous: dict[str, Any] | None = None
        self.last_update_accepted = True

    def vector(self, episode: int) -> np.ndarray:
        p = self.previous or {}
        rewards = self.rewards[-5:]
        rolling = float(np.mean(rewards)) if rewards else 0.0
        trend = float(rewards[-1] - rewards[0]) if len(rewards) > 1 else 0.0
        failure = str(p.get("hard_gate", "pass"))
        task_pass = bool(p.get("task_pass", False))
        values = [
            episode / max(self.budget, 1),
            (self.budget - episode) / max(self.budget, 1),
            float(p.get("reward_total", 0.0)) / 100.0,
            float(p.get("continuous", {}).get("task", 0.0)),
            float(p.get("continuous", {}).get("tracking", 0.0)),
            float(bool(p.get("safety_pass", True))),
            float(p.get("diagnostics", {}).get("key_joint_excursion_rad", 0.0)) / 4.0,
            float(p.get("execution", {}).get("root_tracking_rmse_m", 0.0)) / 0.2,
            float(p.get("execution", {}).get("contact_sliding_speed_mps", 0.0)) / 1.5,
            (max(self.rewards) if self.rewards else 0.0) / 100.0,
            rolling / 100.0,
            trend / 100.0,
            float(failure == "safety_reject"),
            float(failure == "degenerate_reject"),
            float(failure == "pass" and not task_pass),
            float(self.last_update_accepted),
            *self.previous_z.tolist(),
        ]
        return np.clip(np.asarray(values, np.float32), -5.0, 5.0)

    def observe(
        self, z: np.ndarray, result: dict[str, Any], accepted: bool = True
    ) -> None:
        self.previous_z = np.asarray(z, np.float32).copy()
        self.previous = result
        self.rewards.append(float(result["reward_total"]))
        self.last_update_accepted = bool(accepted)


@dataclass(frozen=True)
class PerformanceBand:
    reward_margin: float
    pass_rate_margin: float
    catastrophe_margin: int
    source: str = "BUILD_FROZEN_VARIABILITY"

    @classmethod
    def from_frozen(
        cls,
        rewards: list[float],
        pass_blocks: list[float],
        catastrophe_blocks: list[int],
    ) -> "PerformanceBand":
        values = np.asarray(rewards, np.float64)
        mad = (
            float(np.median(np.abs(values - np.median(values)))) if len(values) else 0.0
        )
        reward_margin = max(0.25, 1.4826 * mad)
        pass_margin = max(
            0.05, float(np.std(pass_blocks, ddof=1)) if len(pass_blocks) > 1 else 0.05
        )
        catastrophe_margin = max(
            1,
            int(np.ceil(np.std(catastrophe_blocks, ddof=1)))
            if len(catastrophe_blocks) > 1
            else 1,
        )
        return cls(reward_margin, pass_margin, catastrophe_margin)

    def preserves(
        self, candidate: dict[str, float], baseline: dict[str, float]
    ) -> bool:
        return bool(
            candidate["mean_reward"] >= baseline["mean_reward"] - self.reward_margin
            and candidate["pass_rate"] >= baseline["pass_rate"] - self.pass_rate_margin
            and candidate["catastrophes"]
            <= baseline["catastrophes"] + self.catastrophe_margin
        )
