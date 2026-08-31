from __future__ import annotations

import copy
import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .motion_features import descriptor_16
from .types import CanonicalReference, StoreEntry


class Scorer(nn.Module):
    """Score each skip/replace action from its 71D store-conditioned feature."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(71, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


@dataclass(frozen=True)
class ConsolidationDecision:
    action: str
    evict_id: str | None
    scores: tuple[float, ...] = ()


class ConsolidationPolicy:
    """Update the admission scorer using reuse observed between qualified misses."""

    def __init__(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        capacity: int = 10,
        online_learning_rate: float = 3e-5,
        gamma: float = 0.95,
    ) -> None:
        if capacity != 10:
            raise ValueError("the scorer feature schema requires capacity=10")
        self.capacity = capacity
        self.online_learning_rate = float(online_learning_rate)
        self.gamma = float(gamma)
        self._torch = torch
        self._online = Scorer().float().cpu()
        self._online.load_state_dict(state_dict, strict=True)
        self._target = Scorer().float().cpu()
        self._target.load_state_dict(state_dict, strict=True)
        self._target.eval()
        self._descriptor = descriptor_16
        self._initial_state = copy.deepcopy(self._online.state_dict())
        self._recent: deque[str] = deque(maxlen=16)
        self.reset()

    def reset(self) -> None:
        self._online.load_state_dict(self._initial_state, strict=True)
        self._target.load_state_dict(self._initial_state, strict=True)
        self._optimizer = torch.optim.Adam(
            self._online.parameters(), lr=self.online_learning_rate
        )
        self._recent.clear()
        self._step = self._updates = 0
        self._replay: list[tuple[np.ndarray, int, float, np.ndarray, float]] = []
        self._pending: tuple[np.ndarray, int] | None = None
        self._current: tuple[np.ndarray, int] | None = None
        self._requests_since_pending = self._reuse_hits_since_pending = 0

    def _entry_descriptor(self, entry: StoreEntry) -> np.ndarray:
        with np.load(entry.motion_path, allow_pickle=False) as archive:
            qpos = np.asarray(archive["qpos_36"], dtype=np.float32)
        return np.asarray(self._descriptor(qpos), dtype=np.float32)

    def _features(
        self,
        *,
        entries: Sequence[StoreEntry],
        candidate: CanonicalReference,
        capability_id: str,
    ) -> np.ndarray:
        if len(entries) != self.capacity:
            raise ValueError(
                "scorer features are only defined when the primary store is full"
            )
        candidate_desc = np.asarray(
            self._descriptor(candidate.qpos_36), dtype=np.float32
        )
        store_desc = np.stack([self._entry_descriptor(entry) for entry in entries])
        mean = store_desc.mean(0)
        maximum = store_desc.max(0)
        recent = tuple(self._recent)
        candidate_recent = recent.count(capability_id) / max(len(recent), 1)
        redundancy = float(
            np.mean([np.linalg.norm(candidate_desc - row) for row in store_desc])
        )
        global_values = np.asarray(
            [
                len(entries) / 20.0,
                len(set(recent)) / 16.0,
                candidate_recent,
                redundancy / 20.0,
                min(self._step, 20) / 20.0,
            ],
            dtype=np.float32,
        )
        rows: list[np.ndarray] = []
        for action_index in range(len(entries) + 1):
            skip = action_index == 0
            target = (
                np.zeros(16, dtype=np.float32) if skip else store_desc[action_index - 1]
            )
            target_capability = (
                None if skip else entries[action_index - 1].capability_id
            )
            target_recent = (
                0.0
                if target_capability is None
                else recent.count(target_capability) / max(len(recent), 1)
            )
            rows.append(
                np.concatenate(
                    [
                        candidate_desc,
                        mean,
                        maximum,
                        target,
                        global_values,
                        np.asarray([float(skip), target_recent], dtype=np.float32),
                    ]
                )
            )
        features = np.stack(rows).astype(np.float32)
        if features.shape != (11, 71):
            raise AssertionError(
                f"unexpected frozen scorer feature shape: {features.shape}"
            )
        return features

    def observe_request(self, capability_id: str, *, reused: bool) -> None:
        self._recent.append(str(capability_id))
        if self._pending is not None:
            self._requests_since_pending += 1
            self._reuse_hits_since_pending += int(bool(reused))

    def _td_update(
        self, transition: tuple[np.ndarray, int, float, np.ndarray, float],
    ) -> float:
        torch = self._torch
        if torch is None or self._online is None or self._target is None:
            raise RuntimeError("policy runtime is unavailable")
        features, action, reward, next_features, done = transition
        states = torch.from_numpy(features)
        next_states = torch.from_numpy(next_features)
        q = self._online(states)[int(action)]
        with torch.no_grad():
            next_action = int(self._online(next_states).argmax().item())
            target = (
                float(reward)
                + self.gamma
                * (1.0 - float(done))
                * self._target(next_states)[next_action]
            )
        loss = torch.nn.functional.smooth_l1_loss(q, target)
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._online.parameters(), 5.0)
        self._optimizer.step()
        self._updates += 1
        if self._updates % 50 == 0:
            self._target.load_state_dict(self._online.state_dict())
        return float(loss.detach())

    def select(
        self,
        capability_id: str,
        entries: Sequence[StoreEntry],
        candidate: CanonicalReference,
    ) -> ConsolidationDecision:
        if self._current is not None:
            raise RuntimeError("call update() before selecting another action")
        if len(entries) < self.capacity:
            self._current = None
            return ConsolidationDecision("INSERT", None)
        features = self._features(
            entries=entries, candidate=candidate, capability_id=capability_id
        )
        with torch.no_grad():
            scores = self._online(torch.from_numpy(features))
            action = int(scores.argmax().item())
        self._current = (features, action)
        self._step += 1
        return ConsolidationDecision(
            "DROP" if action == 0 else "REPLACE",
            None if action == 0 else entries[action - 1].entry_id,
            tuple(float(value) for value in scores),
        )

    def update(self) -> list[float]:
        if self._current is None:
            return []
        losses: list[float] = []
        if self._pending is not None:
            reward = self._reuse_hits_since_pending / max(
                self._requests_since_pending, 1
            )
            transition = (
                self._pending[0],
                self._pending[1],
                float(reward),
                self._current[0],
                0.0,
            )
            self._replay.append(transition)
            losses.append(self._td_update(transition))
            if len(self._replay) >= 8:
                seed = int.from_bytes(
                    hashlib.sha256(
                        f"E2E_CAUSAL_REPLAY\n{self._step}".encode()
                    ).digest()[:4],
                    "little",
                )
                rng = np.random.default_rng(seed)
                for index in rng.integers(0, len(self._replay), size=8):
                    losses.append(self._td_update(self._replay[int(index)]))
        self._pending = self._current
        self._current = None
        self._requests_since_pending = self._reuse_hits_since_pending = 0
        return losses

    def state_dict(self) -> dict[str, torch.Tensor]:
        return copy.deepcopy(self._online.state_dict())
