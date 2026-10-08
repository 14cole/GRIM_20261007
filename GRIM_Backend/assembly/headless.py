"""Run a saved Assembly recipe without Qt.

The Assembly tab already separates its state from its widgets: recipes,
``FeatureAssemblyValues`` and ``FeatureAssemblyFormModel`` are Qt-free, and the
panel's load path only copies recipe values into controls.  This module is the
missing entry point, so a combined platform can be built from a recipe by a
script, a batch sweep or CI with the same validation the tab performs.

The operator review gate is preserved rather than bypassed.  A plan that the
tab would make an operator acknowledge is refused here too unless
``acknowledge_warnings`` is set, and the acknowledgement is bound to that
plan's sha256 exactly as the tab binds it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from GRIM_Backend.assembly.model import (
    WORKLOAD_REVIEW_WARNING_PREFIX,
    FeatureAssemblyFormModel,
    assembly_build_confirmation_required,
    coerce_feature_workflow,
    estimate_validated_assembly_plan_workload,
    format_assembly_work_estimate,
)
from GRIM_Backend.assembly.recipe import read_feature_assembly_recipe


@dataclass(frozen=True)
class PreparedRecipe:
    """One validated recipe, ready to publish or report on."""

    recipe_path: Path
    name: str
    variant: str
    model: FeatureAssemblyFormModel
    service: Any
    plan: Any
    source_warnings: tuple[str, ...]
    validation_warnings: tuple[str, ...]
    review_required: bool

    @property
    def plan_sha256(self) -> str:
        return str(getattr(self.plan, "prepared_plan_sha256", "")).strip()

    @property
    def workload_text(self) -> str:
        estimate = estimate_validated_assembly_plan_workload(self.plan)
        return format_assembly_work_estimate(estimate)


def load_service(backend_path: str | os.PathLike[str] | None = None) -> Any:
    """Import the authoritative GHOST feature_workflow module."""

    from GRIM_Backend.integrations.ghost import load_ghost_module

    return load_ghost_module("feature_workflow", backend_path)


def prepare_recipe(
    recipe_path: str | os.PathLike[str],
    *,
    backend_path: str | os.PathLike[str] | None = None,
    service: Any = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> PreparedRecipe:
    """Load, validate and authoritatively prepare one recipe. Publishes nothing."""

    loaded = read_feature_assembly_recipe(recipe_path)
    model = FeatureAssemblyFormModel(loaded.values)
    adapter = coerce_feature_workflow(
        load_service(backend_path) if service is None else service
    )
    model.validate()
    plan = model.prepare_preview(
        adapter,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )
    warnings = tuple(
        str(value).strip()
        for value in (getattr(plan, "validation_warnings", ()) or ())
        if str(value).strip()
    )
    values = model.values
    # Mirror the tab's gate: manifest/certification policies and any workload
    # threshold crossing require an explicit acknowledgement.
    review = bool(
        values.require_feature_manifests
        or values.require_body_mesh_certification
        or any(
            message.startswith(WORKLOAD_REVIEW_WARNING_PREFIX)
            for message in warnings
        )
        or assembly_build_confirmation_required(
            estimate_validated_assembly_plan_workload(plan)
        )
    )
    return PreparedRecipe(
        recipe_path=Path(loaded.path),
        name=str(loaded.name),
        variant=str(loaded.variant),
        model=model,
        service=adapter,
        plan=plan,
        source_warnings=tuple(loaded.source_warnings or ()),
        validation_warnings=warnings,
        review_required=review and bool(warnings),
    )


def run_recipe(
    recipe_path: str | os.PathLike[str],
    *,
    backend_path: str | os.PathLike[str] | None = None,
    service: Any = None,
    acknowledge_warnings: bool = False,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Any:
    """Prepare and publish one recipe, returning the build dispatch."""

    prepared = prepare_recipe(
        recipe_path,
        backend_path=backend_path,
        service=service,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )
    if prepared.review_required and not acknowledge_warnings:
        raise RuntimeError(
            "This plan raises "
            f"{len(prepared.validation_warnings)} validation warning(s) that "
            "the Assembly tab would require an operator to review:\n  "
            + "\n  ".join(prepared.validation_warnings)
            + "\nRe-run with acknowledge_warnings=True (--acknowledge-warnings) "
            "to accept them for this output."
        )
    return prepared.model.assemble_validated(
        prepared.service,
        acknowledged_plan_sha256=(
            prepared.plan_sha256 if prepared.review_required else None
        ),
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def _emit(prepared: PreparedRecipe, *, stream=sys.stdout) -> None:
    print(f"recipe   : {prepared.recipe_path}", file=stream)
    print(f"name     : {prepared.name} / {prepared.variant}", file=stream)
    print(f"output   : {prepared.model.values.output_grim}", file=stream)
    print(f"plan     : {prepared.plan_sha256 or '(unreported)'}", file=stream)
    workload = prepared.workload_text.strip()
    if workload:
        print(f"workload : {workload}", file=stream)
    for label, messages in (
        ("source", prepared.source_warnings),
        ("plan", prepared.validation_warnings),
    ):
        for message in messages:
            print(f"warning ({label}): {message}", file=stream)
    print(
        "review   : "
        + ("acknowledgement required" if prepared.review_required else "not required"),
        file=stream,
    )


def _progress_printer(quiet: bool):
    if quiet:
        return None
    last = [-1]

    def report(done: int, total: int, message: str = "") -> None:
        total = max(1, int(total))
        percent = int(round(100.0 * int(done) / total))
        if percent == last[0]:
            return
        last[0] = percent
        print(f"\r  {percent:3d}%  {message[:60]:<60}", end="", file=sys.stderr, flush=True)

    return report


def _finish_progress(quiet: bool) -> None:
    if not quiet:
        print("", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grim-assembly",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--backend", default=None,
        help="GHOST backend folder; defaults to the bundled tools/GHOST/ghost_backend.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument("--json", action="store_true", help="Emit one JSON object instead of text.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "validate",
        help="Load, validate and prepare a recipe without publishing anything.",
    )
    check.add_argument("recipe", help="Path to a saved .json Assembly recipe.")

    run = sub.add_parser("run", help="Validate, then publish the combined platform.")
    run.add_argument("recipe", help="Path to a saved .json Assembly recipe.")
    run.add_argument(
        "--acknowledge-warnings", action="store_true",
        help="Accept plan warnings the Assembly tab would gate on an operator review.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    progress = _progress_printer(args.quiet)
    try:
        prepared = prepare_recipe(
            args.recipe, backend_path=args.backend, progress_callback=progress
        )
        _finish_progress(args.quiet)
        if args.command == "validate":
            if args.json:
                print(json.dumps({
                    "recipe": str(prepared.recipe_path),
                    "name": prepared.name,
                    "variant": prepared.variant,
                    "output_grim": str(prepared.model.values.output_grim),
                    "plan_sha256": prepared.plan_sha256,
                    "source_warnings": list(prepared.source_warnings),
                    "validation_warnings": list(prepared.validation_warnings),
                    "review_required": prepared.review_required,
                }, indent=2))
            else:
                _emit(prepared)
            return 0

        if prepared.review_required and not args.acknowledge_warnings:
            _emit(prepared)
            print(
                "\nRefusing to publish: re-run with --acknowledge-warnings to "
                "accept the warnings above for this output.",
                file=sys.stderr,
            )
            return 2
        dispatch = prepared.model.assemble_validated(
            prepared.service,
            acknowledged_plan_sha256=(
                prepared.plan_sha256 if prepared.review_required else None
            ),
            progress_callback=progress,
        )
        _finish_progress(args.quiet)
        if args.json:
            print(json.dumps({
                "recipe": str(prepared.recipe_path),
                "output_path": str(dispatch.output_path),
                "features_only_output_path": str(dispatch.features_only_output_path),
                "features_only_output_published": bool(
                    dispatch.features_only_output_published
                ),
                "reused_validated_plan": bool(dispatch.reused_validated_plan),
                "plan_sha256": prepared.plan_sha256,
            }, indent=2))
        else:
            _emit(prepared)
            print(f"published: {dispatch.output_path}")
            if dispatch.features_only_output_published:
                print(f"features : {dispatch.features_only_output_path}")
        return 0
    except InterruptedError as exc:
        print(f"cancelled: {exc}", file=sys.stderr)
        return 130
    except Exception as exc:  # actionable message, not a traceback
        _finish_progress(args.quiet)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
