"""Shared driver input checks, geometry loading, and durable submission journals."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple


def publish_submission_journal(path, document):
    # type: (Path, Dict[str, Any]) -> None
    """Atomically publish a submission journal after its bytes reach disk."""

    temporary_path = path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(document, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary_path), str(path))


    if os.name == "posix":
        try:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            parent_fd = os.open(str(path.parent), flags)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError:
            pass


def verify_local_unit_input(
    unit: 'Dict[str, Any]',
    context: 'Dict[str, Any]',
) -> 'None':
    from ghost_backend.assembly.fields import geometry_input_fingerprint
    current = geometry_input_fingerprint(
        str(unit["geometry"]), str(context["geometry_units"])
    )
    if current != unit.get("geometry_input_sha256"):
        raise RuntimeError(
            f"Geometry/material input changed during the local run: "
            f"{unit['geometry']}"
        )


def load_geometry_snapshot(geometry_path: 'str', cache: 'Dict[str, Any]') -> 'Tuple[Dict[str, Any], str]':
    """Parsed snapshot for one geometry, built at most once per process.

    ``cache`` belongs to the caller (one per driver module).  The drivers'
    pools use the spawn start method, so every worker parses its own copy of a
    geometry on its first unit; nothing is inherited from the parent.
    """

    cached = cache.get(geometry_path)
    if cached is not None:
        return cached
    from ghost_backend.geometry.io import parse_geometry, build_geometry_snapshot

    path = Path(geometry_path)
    title, segments, ibcs, dielectrics = parse_geometry(path.read_text())
    snapshot = build_geometry_snapshot(title, segments, ibcs, dielectrics)
    snapshot["source_path"] = str(path)
    entry = (snapshot, str(path.parent))
    cache[geometry_path] = entry
    return entry
