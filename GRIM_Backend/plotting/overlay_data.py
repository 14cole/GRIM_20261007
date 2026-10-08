"""Coordinate-file parsing and unit handling for editable plot overlays."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np

LENGTH_UNITS = {"m": 1.0, "cm": .01, "mm": .001, "in": .0254, "ft": .3048}
PLANES = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
MAX_POINTS = 200_000
MAX_FILE_BYTES = 32 * 1024**2
_LABEL = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")


def read_overlay_points(path: str | Path) -> np.ndarray:
    """Read ordered XY/XYZ rows; blank lines separate disconnected paths.

    Accept comma, semicolon, or whitespace delimiters, an optional x/y[/z]
    header, UTF-8 BOM, and # comments. Reject incomplete/nonfinite rows instead
    of quietly dropping points or joining across malformed data.
    """
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Overlay files must be smaller than 32 MiB.")
    rows, columns, pending_break, header_seen = [], None, False, False
    count = 0
    with path.open(encoding="utf-8-sig") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.strip():
                pending_break = bool(rows)
                continue
            text = raw.split("#", 1)[0].strip()
            if not text:
                continue
            tokens = ([value.strip() for value in re.split(r"[,;]", text)]
                      if "," in text or ";" in text else text.split())
            lowered = [value.lower() for value in tokens]
            if not rows and not header_seen and lowered in (["x", "y"], ["x", "y", "z"]):
                columns, header_seen = len(tokens), True
                continue
            if len(tokens) not in (2, 3) or (columns is not None and len(tokens) != columns):
                raise ValueError(f"Line {number}: use consistently two (XY) or three (XYZ) coordinates.")
            try:
                values = [float(value) for value in tokens]
            except ValueError as exc:
                raise ValueError(f"Line {number}: coordinates must be numbers.") from exc
            if not all(np.isfinite(values)):
                raise ValueError(f"Line {number}: coordinates must be finite numbers.")
            columns = len(values)
            count += 1
            if count > MAX_POINTS:
                raise ValueError(f"An overlay can contain at most {MAX_POINTS:,} points.")
            if pending_break:
                rows.append([np.nan] * columns)
                pending_break = False
            rows.append(values)
    if not rows:
        raise ValueError("The file does not contain any XY or XYZ points.")
    return np.asarray(rows, dtype=float)


def axis_info(ax, axis: str) -> tuple[str, str, float]:
    label = getattr(ax, f"get_{axis}label")()
    if not label:
        # ISAR labels a shared range axis on the first subplot only.
        shared = getattr(ax, f"get_shared_{axis}_axes")().get_siblings(ax)
        label = next((getattr(other, f"get_{axis}label")() for other in shared
                      if getattr(other, f"get_{axis}label")()), "")
    match = _LABEL.match(label)
    unit = match.group(2).strip() if match else ""
    name = match.group(1).strip() if match else label
    return name, unit, LENGTH_UNITS.get(unit, 1.0)


def axes_signature(ax) -> tuple[str, str]:
    result = []
    for axis in ("x", "y"):
        name, unit, _scale = axis_info(ax, axis)
        # Length values are stored in metres. Other axes stay in their shown
        # units, and must not accidentally reappear on a different quantity.
        result.append(name + (" [length]" if unit in LENGTH_UNITS else f" [{unit}]"))
    return tuple(result)


def axes_scales(ax) -> np.ndarray:
    return np.array([axis_info(ax, axis)[2] for axis in ("x", "y")])


def supports_overlays(ax) -> bool:
    return ax.name == "rectilinear" and any(
        axis_info(ax, axis)[1] in LENGTH_UNITS for axis in ("x", "y")
    )


def project_points(points, plane: str, ax, length_unit: str) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("Expected XY or XYZ points.")
    if plane not in PLANES or (points.shape[1] == 2 and plane != "XY"):
        raise ValueError("XY files support the XY plane only.")
    if length_unit not in LENGTH_UNITS:
        raise ValueError("Choose a supported length unit.")
    xy = points[:, PLANES[plane]].copy()
    for column, axis in enumerate(("x", "y")):
        if axis_info(ax, axis)[1] in LENGTH_UNITS:
            xy[:, column] *= LENGTH_UNITS[length_unit]
    return xy


@dataclass(eq=False)
class OverlayPath:
    name: str
    points: np.ndarray  # N x 2; length coordinates in metres, NaN rows separate paths.
    signature: tuple[str, str]
    panel: int
    color: str = "#ff4fa3"
    linestyle: str = "-"
    linewidth: float = 1.8
    show_points: bool = True
    visible: bool = True
    artist: object = field(default=None, repr=False)
