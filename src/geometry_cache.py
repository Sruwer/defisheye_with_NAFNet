from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np


CACHE_FORMAT_VERSION = 1


def cache_path(directory: str | Path | None, namespace: str, parameters: dict) -> Path | None:
    """Return a content-addressed LUT path for a geometry configuration."""
    if directory is None:
        return None
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "namespace": namespace,
        "parameters": parameters,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    return Path(directory) / f"{namespace}-{digest}.npz"


def load_arrays(path: Path | None, names: set[str]) -> dict[str, np.ndarray] | None:
    if path is None or not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            version = int(np.asarray(archive["cache_format_version"]).item())
            if version != CACHE_FORMAT_VERSION or not names.issubset(archive.files):
                return None
            return {name: np.array(archive[name], copy=True) for name in names}
    except (OSError, ValueError, KeyError, TypeError):
        # A partial/corrupt cache is simply rebuilt from the source geometry.
        return None


def save_arrays(path: Path | None, arrays: dict[str, np.ndarray]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                cache_format_version=np.asarray(CACHE_FORMAT_VERSION, dtype=np.int16),
                **arrays,
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
