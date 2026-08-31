from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CanonicalReference:
    qpos_36: np.ndarray
    fps: float = 30.0


@dataclass(frozen=True)
class StoreEntry:
    entry_id: str
    capability_id: str
    motion_path: str
    certificate: Mapping[str, Any]
