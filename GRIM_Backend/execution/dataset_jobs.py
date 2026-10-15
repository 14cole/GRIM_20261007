"""Background dataset workers and bounded parallel loading.

Publication transactions live in GRIM_Backend.io.batch; catalog/UI ownership
remains with DatasetOpsMixin."""
from __future__ import annotations

import ctypes
import csv
import math
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from PySide6.QtCore import QObject, QThread, Signal
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.io.loaders import (
    UnrecognizedTableError, is_supported_path, load_dataset as load_dataset_headless,
)
from GRIM_Backend.io.batch import _stage_and_publish_csv_batch


def _is_supported_dataset_path(path: str) -> bool:
    return is_supported_path(path)

def _available_memory_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        # Optional telemetry must not prevent loading a dataset. Use the OS
        # probe or a conservative unknown-memory budget if psutil fails.
        pass

    # GRIM is commonly copied to a clean workstation where psutil is not yet
    # installed.  Retain a real memory budget on the two primary deployment
    # families instead of falling back immediately to CPU-count concurrency.
    if os.name == "nt":
        try:
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = (
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                )

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages >= 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return None

_LOADER_MIN_WORKING_BYTES = 64 * 1024**2

_LOADER_UNKNOWN_MEMORY_BUDGET_BYTES = 512 * 1024**2

_GRIM_LOAD_PEAK_FACTOR = 3.5

_TEXT_LOADER_EXTENSIONS = (".csv", ".txt", ".dat", ".asc", ".ascii", ".tsv", ".cst_data", ".out", ".ss")

def _grim_archive_uncompressed_bytes(path: str) -> int | None:
    """Return declared uncompressed NPZ bytes without extracting the archive."""

    if not str(path).casefold().endswith(".grim"):
        return None
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            if not members:
                return 0
            return sum(max(0, int(member.file_size)) for member in members)
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile):
        # The authoritative loader will report a malformed archive.  Keep a
        # conservative file-size fallback here so planning itself remains
        # non-destructive and does not mask that parse error.
        return None

def _dataset_load_memory_estimate(path: str, *, storage="memory", selection=None) -> tuple[int, int]:
    """Return conservative ``(retained, per-load peak)`` byte estimates."""

    try:
        stored_bytes = max(0, int(os.path.getsize(path)))
    except OSError:
        stored_bytes = 0
    if str(path).casefold().endswith(".grim"):
        from GRIM_Backend.datasets.memory import estimate_native_archive_allocation
        try:
            report = estimate_native_archive_allocation(path, storage=storage, selection=selection)
            return int(report['retained_bytes']), int(report['estimated_peak_bytes'])
        except (OSError, ValueError, zipfile.BadZipFile):
            # Preserve the authoritative reader's format error. The fallback
            # below is only for malformed/unreadable archives, not valid v2 data.
            pass
    if str(path).casefold().endswith(('.pio', '.cmplx_di', '.ss')):
        from GRIM_Backend.io.binary_storage import estimate_binary_import
        try:
            report = estimate_binary_import(path, storage=storage)
            if report is not None:
                return int(report['retained_bytes']), int(report['estimated_peak_bytes'])
        except (OSError, ValueError, IndexError):
            pass  # The reader reports corrupt framing and unsupported headers.
    if str(path).casefold().endswith(_TEXT_LOADER_EXTENSIONS):
        from GRIM_Backend.io.csv import has_flat_csv_signature, load_flat_csv
        if has_flat_csv_signature(path):
            try:
                report = load_flat_csv(path, None, None, _estimate_only=True)
            except (OSError, ValueError, csv.Error):
                pass  # The per-file loader reports malformed input independently.
            else:
                return int(report['retained_bytes']), int(report['estimated_peak_bytes'])
        from GRIM_Backend.io.cst import estimate_vendor_text_import
        try:
            report = estimate_vendor_text_import(path)
        except (OSError, ValueError, KeyError, IndexError, csv.Error):
            report = None
        if report is not None:
            return int(report['retained_bytes']), int(report['estimated_peak_bytes'])
    expanded_bytes = _grim_archive_uncompressed_bytes(path)
    if expanded_bytes is not None:
        retained = max(stored_bytes, int(expanded_bytes))
        # RcsGrid validates and cleans power/phase into new arrays while the
        # NPZ members are still live.  Include those transient copies instead
        # of treating a highly compressed archive as its on-disk byte count.
        peak = max(
            _LOADER_MIN_WORKING_BYTES,
            int(math.ceil(_GRIM_LOAD_PEAK_FACTOR * retained)),
        )
        return retained, peak

    lower = str(path).casefold()
    # Delimited readers create Python strings/dicts, coordinate-key maps, dense
    # output arrays, and duplicate-validation state concurrently. Real SENTRi
    # imports have measured around 77x their file size at peak, so the old 4x
    # rule was unsafe by more than an order of magnitude.
    if lower.endswith((".csv", ".txt", ".dat", ".asc", ".ascii", ".tsv")):
        retained = max(stored_bytes, 16 * stored_bytes)
        peak = max(_LOADER_MIN_WORKING_BYTES, 96 * stored_bytes)
        return retained, peak
    if lower.endswith((".cst_data", ".out", ".ss")):
        retained = max(stored_bytes, 8 * stored_bytes)
        peak = max(_LOADER_MIN_WORKING_BYTES, 40 * stored_bytes)
        return retained, peak

    # Binary single-cut formats still materialize complex, power, and phase
    # arrays but avoid the large per-cell Python parser overhead.
    retained = max(stored_bytes, 2 * stored_bytes)
    peak = max(_LOADER_MIN_WORKING_BYTES, 6 * stored_bytes)
    return retained, peak

class LoadMemoryLimitError(MemoryError):
    """A planning estimate exceeded policy, rather than an allocation failing."""
    def __init__(self, plan):
        self.plan = plan
        super().__init__(
            f"This dataset batch needs an estimated {plan['serial_peak_bytes'] / 1024**2:.0f} MiB "
            f"but the load budget is {plan['budget_bytes'] / 1024**2:.0f} MiB. "
            "Adjust Load options, continue once, or use disk storage for native/binary files. "
            "Load fewer or smaller dataset files to reduce the estimate.")


def _load_memory_budget(max_output_bytes=None):
    from GRIM_Backend.datasets.load_policy import checked_import_budget
    explicit = checked_import_budget(max_output_bytes)
    available = _available_memory_bytes()
    budget = (explicit if explicit is not None else
              max(1, int(available * 0.5)) if available is not None else
              _LOADER_UNKNOWN_MEMORY_BUDGET_BYTES)
    return available, budget


def _dataset_load_plan(tasks, *, max_output_bytes=None, load_options=None):
    paths = [str(task[1]) for task in tasks]
    options = load_options or {}
    estimates = [_dataset_load_memory_estimate(path, storage=options.get('storage', 'memory'),
                 selection=options.get('selection')) for path in paths]
    retained = sum(value for value, _ in estimates)
    extras = sorted((max(0, peak - value) for value, peak in estimates), reverse=True)
    available, budget = _load_memory_budget(max_output_bytes)
    return dict(retained_bytes=retained, transient_bytes=extras,
                serial_peak_bytes=retained + (extras[0] if extras else 0),
                available_bytes=available, budget_bytes=budget,
                files=[dict(path=path, retained_bytes=value, peak_bytes=peak)
                       for path, (value, peak) in zip(paths, estimates)])


def _recommended_loader_workers(tasks, *, max_output_bytes=None, load_options=None,
                                allow_exceed=False) -> int:
    task_count = tasks if isinstance(tasks, int) else len(tasks)
    cpus = os.cpu_count() or 1
    target = max(1, min(int(task_count), cpus if cpus <= 2 else cpus - 1))
    if isinstance(tasks, int) or not tasks:
        return target
    plan = _dataset_load_plan(tasks, max_output_bytes=max_output_bytes, load_options=load_options)
    if allow_exceed:
        return 1  # A one-time override never multiplies concurrent workspaces.
    if plan['serial_peak_bytes'] > plan['budget_bytes']:
        raise LoadMemoryLimitError(plan)
    if any(str(task[1]).casefold().endswith(_TEXT_LOADER_EXTENSIONS) for task in tasks):
        return 1
    safe_workers = 1
    for count in range(2, target + 1):
        if plan['retained_bytes'] + sum(plan['transient_bytes'][:count]) > plan['budget_bytes']:
            break
        safe_workers = count
    return safe_workers


def _load_dataset_path_task(task: tuple[int, str], *, load_options=None, max_output_bytes=None) -> dict[str, object]:
    index, path = task
    file_name = os.path.basename(path)
    dataset_name = os.path.splitext(file_name)[0]
    lower = path.lower()
    try:
        if not _is_supported_dataset_path(path):
            return {
                "status": "ignored",
                "index": index,
                "path": path,
                "file_name": file_name,
                "error": "Unsupported file extension",
            }
        options = dict(load_options or {})
        options.pop('allow_exceed', None)
        options.pop('max_output_bytes', None)
        if not lower.endswith(('.grim', '.pio', '.cmplx_di', '.ss')):
            options.pop('storage', None)
        if not lower.endswith('.grim'):
            options.pop('selection', None)
        if max_output_bytes is not None:
            options['max_output_bytes'] = max_output_bytes
        dataset = load_dataset_headless(path, **options)
        history = str(getattr(dataset, "history", "") or path)
    except Exception as exc:
        return {
            "status": ("memory_limit" if isinstance(exc, MemoryError) else
                       "mapping_required" if isinstance(exc, UnrecognizedTableError) else "error"),
            "index": index,
            "path": path,
            "file_name": file_name,
            "error": str(exc),
            "load_options": dict(load_options or {}),
            "max_output_bytes": max_output_bytes,
        }

    return {
        "status": "ok",
        "index": index,
        "path": path,
        "file_name": file_name,
        "name": dataset_name,
        "history": history,
        "dataset": dataset,
        "load_options": dict(load_options or {}),
    }

def _join_many_with_progress(
    grids: list[RcsGrid],
    *,
    tol: float = 1e-6,
    progress_cb=None,
) -> RcsGrid:
    checked = RcsGrid._ensure_grids(grids)
    total = len(checked)
    available = _available_memory_bytes()
    from GRIM_Backend.datasets.load_policy import current_import_memory_budget
    captured_limit = current_import_memory_budget()
    limit = captured_limit if captured_limit is not None else int(available * 0.5) if available is not None else None
    result = RcsGrid.join_many(
        *checked,
        tol=tol,
        overlap="error",
        max_output_bytes=limit,
    )
    if progress_cb is not None:
        progress_cb(total, total)
    return result

class _DatasetLoadWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(self, tasks: list[tuple[int, str]], ignored_count: int = 0, parent=None, *, load_options=None) -> None:
        super().__init__(parent)
        self._tasks = list(tasks)
        self._ignored_count = int(ignored_count)
        self._load_options = dict(load_options or {})

    def run(self) -> None:
        total = len(self._tasks)
        loaded: list[dict[str, object]] = []
        mapping_required: list[dict[str, object]] = []
        failed: list[str] = []
        used_parallel = False
        memory_entries = []
        memory_review = None

        def _consume(result: dict[str, object], done_count: int) -> None:
            status = str(result.get("status", "error"))
            file_name = str(result.get("file_name", "dataset"))
            if status == "memory_limit":
                memory_entries.append(result)
                self.progress.emit(done_count, total, f"Load options needed: {file_name}")
                return
            if status == "mapping_required":
                mapping_required.append(result)
                self.progress.emit(done_count, total, f"Column labels needed: {file_name}")
                return
            if status == "ok":
                loaded.append(result)
                self.progress.emit(done_count, total, f"Loaded {file_name}")
                return
            error_text = str(result.get("error", "Unknown error"))
            failed.append(f"{file_name} ({error_text})")
            self.progress.emit(done_count, total, f"Failed {file_name}")

        try:
            options = self._load_options
            _available, budget = _load_memory_budget(options.get('max_output_bytes'))
            worker_count = _recommended_loader_workers(
                self._tasks, max_output_bytes=budget, load_options=options,
                allow_exceed=bool(options.get('allow_exceed'))) if total else 1
            cap = sys.maxsize if options.get('allow_exceed') else budget
            if total == 1:
                _consume(_load_dataset_path_task(self._tasks[0], load_options=options, max_output_bytes=cap), 1)
            elif total > 1:
                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    futures = {
                        pool.submit(_load_dataset_path_task, task, load_options=options, max_output_bytes=cap): task
                        for task in self._tasks
                    }
                    done_count = 0
                    for future in as_completed(futures):
                        result = future.result()
                        done_count += 1
                        _consume(result, done_count)
                used_parallel = worker_count > 1
        except LoadMemoryLimitError as exc:
            memory_review = {'paths': [str(task[1]) for task in self._tasks],
                             'plan': exc.plan, 'options': self._load_options, 'error': str(exc)}
        except Exception as exc:
            # Individual parse failures normally arrive as result mappings.
            # This catches pool setup/submission/future faults so the owning
            # QThread still receives exactly one terminal signal and can quit.
            failed.append(f"Dataset loader ({type(exc).__name__}: {exc})")
        finally:
            if memory_entries:
                memory_review = {'paths': [entry['path'] for entry in memory_entries],
                                 'options': self._load_options,
                                 'error': '; '.join(str(entry['error']) for entry in memory_entries)}
            self.finished.emit(
                {
                    "loaded": loaded,
                    "mapping_required": sorted(mapping_required, key=lambda item: int(item["index"])),
                    "failed": failed,
                    "ignored": self._ignored_count,
                    "used_parallel": used_parallel,
                    "total_supported": total,
                    "memory_review": memory_review,
                }
            )

class _CsvExportWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(
        self,
        entries: list[tuple[RcsGrid, str]],
        *,
        scale: str,
        include_phase: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._entries = list(entries)
        self._scale = str(scale)
        self._include_phase = bool(include_phase)

    def run(self) -> None:
        total = len(self._entries)
        try:
            self.progress.emit(0, total, "Writing staged CSV files")
            paths = _stage_and_publish_csv_batch(
                self._entries,
                scale=self._scale,
                include_phase=self._include_phase,
            )
        except Exception as exc:
            self.finished.emit({"ok": False, "error": str(exc), "total": total})
            return
        self.progress.emit(total, total, "Published CSV files")
        self.finished.emit({"ok": True, "paths": paths, "total": total})

class _BackgroundJobCancelled(RuntimeError):
    """Internal cooperative-cancellation sentinel."""


def _operation_storage_run(function):
    """Create a separate allocation context inside the actual Qt worker."""
    @wraps(function)
    def run(self):
        from GRIM_Backend.datasets.result_storage import operation_result_storage
        options = dict(getattr(self, '_result_storage_options', {}) or {})
        try:
            with operation_result_storage(**options):
                return function(self)
        except Exception as exc:
            # Context setup can fail before the worker's ordinary terminal
            # handler (for example invalid replayed options). Always quit.
            if isinstance(self, _RangeCalibrationWorker):
                self.finished.emit({'results': [], 'failed': [str(exc)], 'total': len(self._targets)})
            else:
                self.finished.emit({'ok': False, 'error': str(exc), 'error_type': type(exc).__name__})
    return run


class _BackgroundCallableWorker(QObject):
    """Run one pure-Python/NumPy callable away from Qt's GUI thread."""

    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(self, function, *, reports_progress: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._function = function
        self._reports_progress = bool(reports_progress)
        self.supports_cancellation = self._reports_progress

    @_operation_storage_run
    def run(self) -> None:
        try:
            if QThread.currentThread().isInterruptionRequested():
                raise _BackgroundJobCancelled()
            if self._reports_progress:
                def report_progress(done_count, total_count, detail=""):
                    if QThread.currentThread().isInterruptionRequested():
                        raise _BackgroundJobCancelled()
                    self.progress.emit(done_count, total_count, detail)

                result = self._function(report_progress)
            else:
                result = self._function()
            if QThread.currentThread().isInterruptionRequested():
                raise _BackgroundJobCancelled()
        except _BackgroundJobCancelled:
            self.finished.emit({"ok": False, "cancelled": True})
            return
        except Exception as exc:
            self.finished.emit(
                {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
            )
            return
        self.finished.emit({"ok": True, "result": result})

class _JoinDatasetsWorker(QObject):
    supports_cancellation = False
    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(self, grids: list[RcsGrid], tol: float = 1e-6, parent=None) -> None:
        super().__init__(parent)
        self._grids = list(grids)
        self._tol = float(tol)

    @_operation_storage_run
    def run(self) -> None:
        total = max(1, len(self._grids))
        try:
            if QThread.currentThread().isInterruptionRequested():
                raise _BackgroundJobCancelled()

            def _emit_progress(done_count: int, total_count: int) -> None:
                if QThread.currentThread().isInterruptionRequested():
                    raise _BackgroundJobCancelled()
                self.progress.emit(done_count, total_count, "Joining datasets")

            merged = _join_many_with_progress(self._grids, tol=self._tol, progress_cb=_emit_progress)
        except _BackgroundJobCancelled:
            self.finished.emit({"ok": False, "cancelled": True, "total": total})
            return
        except Exception as exc:
            self.finished.emit({"ok": False, "error": str(exc), "total": total})
            return
        self.finished.emit({"ok": True, "merged": merged, "total": total})

class _RangeCalibrationWorker(QObject):
    """Apply one calibration definition to DUT grids off the GUI thread."""

    supports_cancellation = True

    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(
        self,
        targets: list[tuple[str, RcsGrid]],
        measured_entry: tuple[str, RcsGrid],
        exact_entry: tuple[str, RcsGrid],
        params: dict[str, object],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._targets = list(targets)
        self._measured_name, self._measured = measured_entry
        self._exact_name, self._exact = exact_entry
        self._params = dict(params)

    @_operation_storage_run
    def run(self) -> None:
        total = len(self._targets)
        results: list[dict[str, object]] = []
        failed: list[str] = []
        try:
            offset_m = float(self._params["range_offset_m"])
        except (KeyError, TypeError, ValueError) as exc:
            self.finished.emit(
                {
                    "results": results,
                    "failed": [f"invalid range-calibration parameters ({exc})"],
                    "total": total,
                }
            )
            return
        exact_display = str(self._exact_name)
        if len(exact_display) > 48:
            exact_display = exact_display[:45] + "..."

        for index, (target_name, target) in enumerate(self._targets, start=1):
            if QThread.currentThread().isInterruptionRequested():
                failed.append("canceled by user")
                break
            try:
                calibrated = target.range_calibrate(
                    self._measured,
                    self._exact,
                    offset_m,
                    allow_singleton_angular_broadcast=bool(
                        self._params.get(
                            "allow_singleton_angular_broadcast", False
                        )
                    ),
                    convention_attested=False,
                    measured_label=self._measured_name,
                    exact_label=self._exact_name,
                    maximum_correction_gain_db=self._params.get(
                        "maximum_correction_gain_db", 60.0
                    ),
                )
            except Exception as exc:
                failed.append(f"{target_name} ({exc})")
                self.progress.emit(index, total, f"Skipped {target_name}")
                continue

            results.append(
                {
                    "dataset": calibrated,
                    "source_dataset": target,
                    "name": (
                        f"{target_name} [Range Cal: {exact_display}; "
                        f"DeltaR {offset_m:+.6g} m]"
                    ),
                    "history": (
                        f"Range Cal: {target_name}; measured={self._measured_name}; "
                        f"exact={self._exact_name}; Î”R={offset_m:+.12g} m "
                        "(positive away)"
                    ),
                }
            )
            self.progress.emit(index, total, f"Calibrated {target_name}")

        self.finished.emit(
            {
                "results": results,
                "failed": failed,
                "total": total,
            }
        )
