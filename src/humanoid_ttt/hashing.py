from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return sha256_bytes(
        f"{array.dtype.str}|{array.shape}".encode("ascii") + b"\0" + array.tobytes()
    )


def payload_sha256(value: Mapping[str, Any], *, excluded: Iterable[str] = ()) -> str:
    ignored = set(excluded)
    return sha256_bytes(
        canonical_json_bytes(
            {key: val for key, val in value.items() if key not in ignored}
        )
    )
