from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .hashing import sha256_bytes


def sha_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def hash_order(values: Iterable[str], salt: str) -> list[str]:
    return sorted(values, key=lambda value: sha_text(f"{salt}\n{value}"))


@dataclass(frozen=True)
class StoreAction:
    kind: str
    target: str | None = None

    @property
    def label(self) -> str:
        return "SKIP" if self.kind == "SKIP" else f"REPLACE({self.target})"


def actions(store: Sequence[str]) -> list[StoreAction]:
    return [StoreAction("SKIP")] + [StoreAction("REPLACE", item) for item in store]


def apply_action(
    store: Sequence[str], candidate: str, action: StoreAction
) -> tuple[str, ...]:
    current = list(store)
    if action.kind == "SKIP":
        return tuple(current)
    index = current.index(str(action.target))
    current[index] = candidate
    if len(set(current)) != len(current):
        raise ValueError("duplicate store identity")
    return tuple(current)


def utility(store: Sequence[str], probes: Sequence[str], qualified: set[str]) -> float:
    present = set(store)
    return float(np.mean([probe in present and probe in qualified for probe in probes]))


def probe_bank(
    candidate: str,
    store: Sequence[str],
    recent: Sequence[str],
    universe: Sequence[str],
    step: int,
) -> tuple[str, ...]:
    candidate_rows = [candidate] * 4
    recent_unique = list(dict.fromkeys(reversed(list(recent))))
    recent_rows = recent_unique[:4]
    fallback = hash_order(universe, f"RECENT_FALLBACK_V1\n{step}\n{candidate}")
    for item in fallback:
        if len(recent_rows) >= 4:
            break
        if item not in recent_rows:
            recent_rows.append(item)
    coverage = hash_order(store, f"STORE_COVERAGE_V1\n{step}\n{candidate}")[:4]
    if len(coverage) < 4:
        for item in recent_rows:
            if len(coverage) >= 4:
                break
            coverage.append(item)
    result = tuple(candidate_rows + recent_rows[:4] + coverage[:4])
    if len(result) != 12:
        raise AssertionError(len(result))
    return result


def immediate_rows(
    store: Sequence[str], candidate: str, probes: Sequence[str], qualified: set[str]
) -> list[dict]:
    before = utility(store, probes, qualified)
    rows = []
    for action in actions(store):
        after_store = apply_action(store, candidate, action)
        after = utility(after_store, probes, qualified)
        rows.append(
            {
                "action": action,
                "action_label": action.label,
                "U_before": before,
                "U_after": after,
                "delta_U": after - before,
                "after_store": after_store,
            }
        )
    return rows
