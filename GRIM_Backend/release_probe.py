"""Isolated, offline application acceptance probe used by release_validation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import sysconfig


def verify_bootstrap_environment(expected_environment: Path):
    """Keep the release lock sufficient to recreate its own offline environment."""
    from pip._vendor.packaging.requirements import Requirement
    from GRIM_Backend.setup_offline import MINIMUM_PIP, VIRTUALENV_REQUIREMENT

    required = ("pip>=" + ".".join(map(str, MINIMUM_PIP)), VIRTUALENV_REQUIREMENT)
    for specification in required:
        requirement = Requirement(specification)
        try:
            distribution = importlib.metadata.distribution(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            distribution = None
        if (distribution is None
                or not requirement.specifier.contains(distribution.version, prereleases=True)
                or not Path(distribution.locate_file("")).resolve().is_relative_to(expected_environment.resolve())):
            raise RuntimeError(
                f"Offline release bootstrap requires {specification} in the selected environment. "
                "Keep pip and virtualenv in offline packages/requirements.txt, supply their dependency wheels, "
                "and rerun Validate_GRIM.bat to install and validate the complete release.")
    completed = subprocess.run([sys.executable, "-I", "-B", "-m", "pip", "--isolated",
                                "--disable-pip-version-check", "check"],
                               capture_output=True, text=True, timeout=30, check=False)
    if completed.returncode:
        raise RuntimeError("Offline release dependencies are incomplete or inconsistent. "
                           "Supply the required wheels and rerun Validate_GRIM.bat.\n"
                           + completed.stdout + completed.stderr)


def run_probe(source: Path, scratch: Path, expected_environment: Path | None = None):
    """Exercise actual application code without changing user settings/data."""
    source, scratch = source.resolve(), scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    os.environ.update(QT_QPA_PLATFORM="offscreen", MPLCONFIGDIR=str(scratch / "matplotlib"),
                      XDG_CONFIG_HOME=str(scratch / "settings"),
                      XDG_CACHE_HOME=str(scratch / "cache"))
    for name in ("GRIM_CACHE_DIR", "GHOST_CACHE_DIR", "FREDDY_CACHE_DIR"):
        os.environ[name] = str(scratch / "cache")
    import numpy as np
    import scipy.linalg
    import matplotlib
    from pip._vendor.packaging.tags import sys_tags
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    import GRIM_Backend.ui.app as gui
    from GRIM_Backend.datasets.grid import RcsGrid
    from GRIM_Backend.datasets.transforms import offset_db
    from GRIM_Backend.io.loaders import load_dataset
    from GRIM_Backend.io.csv import write_flat_csv
    from GRIM_Backend.scripting.plotting import plot_datasets

    if not Path(gui.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Application imported from a different source tree")
    import site
    if not sys.flags.isolated or site.ENABLE_USER_SITE:
        raise RuntimeError("Acceptance must run in an isolated Python subprocess")
    if expected_environment is not None:
        expected_environment = expected_environment.resolve()
        if Path(sys.prefix).resolve() != expected_environment or sys.prefix == sys.base_prefix:
            raise RuntimeError("Acceptance did not run in the selected isolated environment")
        config = dict(line.split("=", 1) for line in (expected_environment / "pyvenv.cfg").read_text().splitlines() if "=" in line)
        config = {key.strip(): value.strip() for key, value in config.items()}
        base = source / "GRIM_Backend" / "python_env"
        if Path(config["home"]).resolve() != base.resolve():
            raise RuntimeError("The selected environment was created from a different embedded Python")
        import hashlib
        for name in (f"python{sys.version_info.major}{sys.version_info.minor}.dll",
                     f"python{sys.version_info.major}{sys.version_info.minor}.zip"):
            original, copied = base / name, expected_environment / "Scripts" / name
            if not original.is_file() or not copied.is_file():
                raise RuntimeError(f"Embedded runtime component is missing: {name}")
            if hashlib.sha256(original.read_bytes()).digest() != hashlib.sha256(copied.read_bytes()).digest():
                raise RuntimeError(f"Embedded Python changed after environment creation: {name}")
        import PySide6
        for module in (np, scipy.linalg, matplotlib, PySide6):
            if not Path(module.__file__).resolve().is_relative_to(expected_environment):
                raise RuntimeError(f"External dependency imported: {module.__name__}")
        verify_bootstrap_environment(expected_environment)

    # PTM reserves its first frequency record for the header and requires at
    # least 37 frequencies. This small common grid exercises all four writers.
    samples = (np.arange(120, dtype=float).reshape(3, 1, 40, 1) + 1 + 0.25j)
    grid = RcsGrid([-10., 0., 10.], [0.], 8. + np.arange(40) * .1, ["VV"], rcs=samples,
                   units={"frequency": "GHz", "azimuth": "deg", "elevation": "deg",
                          "rcs_linear_quantity": "sigma_3d", "rcs_log_unit": "dBsm"})
    input_path = Path(grid.save(scratch / "input.grim"))
    loaded = load_dataset(input_path)
    np.testing.assert_array_equal(loaded.rcs, samples)
    changed = offset_db(loaded, 6.0)
    np.testing.assert_allclose(changed.rcs, samples * 10. ** (6. / 20.), rtol=1.e-12)
    output_path = Path(changed.save(scratch / "output.grim"))
    np.testing.assert_array_equal(load_dataset(output_path).rcs, changed.rcs)
    exports = (
        (write_flat_csv(changed, scratch / "output.csv", scale="both", include_phase=True), 2.e-10),
        (changed.save_pio(scratch / "output.pio", precision="double"), 2.e-12),
        (changed.save_ptm(scratch / "output.ptm"), 2.e-6),
    )
    for path, tolerance in exports:
        restored = load_dataset(path)
        for name in ("azimuths", "elevations", "frequencies"):
            np.testing.assert_allclose(getattr(restored, name), getattr(changed, name),
                                       rtol=tolerance, atol=tolerance)
        np.testing.assert_array_equal(restored.polarizations, changed.polarizations)
        np.testing.assert_allclose(restored.rcs, changed.rcs, rtol=tolerance, atol=tolerance)
        if restored.linear_quantity() != "sigma_3d":
            raise RuntimeError(f"Export changed the physical quantity: {path}")
    figure = plot_datasets([("Acceptance", changed)], mode="waterfall",
                           azimuths=changed.azimuths, elevations=changed.elevations,
                           frequencies=changed.frequencies, polarization="VV", heatmap_cuts=True)
    figure.savefig(scratch / "waterfall.png")
    if (scratch / "waterfall.png").stat().st_size < 100:
        raise RuntimeError("Plot rendering did not produce a usable image")

    QSettings.setDefaultFormat(QSettings.IniFormat)
    for scope in (QSettings.UserScope, QSettings.SystemScope):
        QSettings.setPath(QSettings.IniFormat, scope, str(scratch / "settings"))
    app = QApplication.instance() or QApplication([])
    window = gui.GrimCutWindow(settings=QSettings(str(scratch / "grim.ini"), QSettings.IniFormat))
    try:
        window.show()
        app.processEvents()
        for name in ("ghost_integration", "freddy_integration"):
            integration = getattr(window, name)
            if integration.workspace is None:
                raise RuntimeError(f"{name} did not load: {integration.load_error}")
        if not window.isVisible():
            raise RuntimeError("Application window did not open")
        window.grab().save(str(scratch / "application.png"))
    finally:
        window.close()
        app.processEvents()
        window.deleteLater()
        app.processEvents()

    dependencies = sorted(({"name": distribution.metadata["Name"], "version": distribution.version}
                           for distribution in importlib.metadata.distributions()),
                          key=lambda value: value["name"].lower())
    return {"status": "passed", "checks": (["offline bootstrap dependencies"] if expected_environment is not None else []) + ["isolated imports", "load", "complex offset",
            "save and reload", "CSV round trip", "PIO round trip", "PTM round trip",
            "waterfall render", "GUI window", "FREDDY workspace", "GHOST workspace"],
            "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                       "architecture_bits": struct.calcsize("P") * 8, "platform": sysconfig.get_platform(),
                       "executable": sys.executable}, "dependencies": dependencies,
            "wheel_tags": [str(tag) for tag in sys_tags()]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--environment", type=Path)
    args = parser.parse_args(argv)
    result = run_probe(args.source, args.scratch, args.environment)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
