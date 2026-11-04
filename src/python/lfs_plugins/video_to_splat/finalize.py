# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The download package, completed once a splat exists: splat exports, dense cloud, mesh.

Runs when training ends (or on request) because it needs the trained Gaussians:

1. splat.ply / splat.spz / splat.sog / splat.glb through the app's exporters,
2. dense.ply: the Gaussians as a colored, oriented point cloud,
3. mesh.ply: COLMAP's Poisson mesher on that cloud (CPU),
4. mesh.glb and mesh.usdz: the mesh with vertex colors, for engines and iPhone AR.

Every step is optional and reported in package.json; a missing exporter or a crashed tool
never undoes the others. All side effects are injected so this stays testable.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import CompletedProcess
from typing import Callable, Optional

from . import mesh_export
from .text import _tr

SPLAT_FORMATS = (("ply", 0), ("spz", 2), ("sog", 1), ("glb", 13))  # lf.export_scene format codes
POISSON_DEPTH = 10
POISSON_TRIM = 7
MANIFEST = "package.json"


@dataclass
class FinalizeHooks:
    """Side effects: read the splat, export it, run COLMAP."""

    splat_tensors: Callable[[], Optional[dict]]            # {means, scales, rotations, colors, opacity} or None
    export_splat: Callable[[int, Path], Optional[str]]     # (format code, path) -> error message or None
    run: Callable[[list[str], Callable[[str], None], Callable[[], bool]], CompletedProcess]
    colmap: Optional[Path] = None


@dataclass
class FinalizeResult:
    package: Path
    files: dict = field(default_factory=dict)   # name -> {"source": ..., "bytes": int}
    skipped: dict = field(default_factory=dict)  # name -> reason

    def manifest(self) -> dict:
        return {"files": self.files, "skipped": self.skipped, "written_at": time.time()}


def plan_poisson_command(colmap: Path, cloud: Path, mesh: Path) -> list[str]:
    return [str(colmap), "poisson_mesher", "--input_path", str(cloud), "--output_path", str(mesh),
            "--PoissonMeshing.depth", str(POISSON_DEPTH), "--PoissonMeshing.trim", str(POISSON_TRIM),
            "--PoissonMeshing.color", "1", "--log_target", "stderr"]


def finalize_package(package: Path, hooks: FinalizeHooks, *, on_progress: Callable[[str], None] = lambda _m: None,
                     should_cancel: Callable[[], bool] = lambda: False, formats=SPLAT_FORMATS,
                     name: str = "scene") -> FinalizeResult:
    package.mkdir(parents=True, exist_ok=True)
    result = FinalizeResult(package=package)

    def record(path: Path, source: str) -> None:
        result.files[path.name] = {"source": source, "bytes": path.stat().st_size if path.is_file() else 0}

    # 1. splat exports
    for ext, code in formats:
        if should_cancel():
            break
        target = package / f"splat.{ext}"
        on_progress(_tr("video_to_splat.progress.exporting", "Exporting {name}", name=target.name))
        error = hooks.export_splat(code, target)
        if error or not target.is_file():
            result.skipped[target.name] = error or "export produced no file"
        else:
            record(target, "lf.export_scene")

    # 2. dense cloud from the Gaussians
    tensors = hooks.splat_tensors()
    if not tensors:
        result.skipped["dense.ply"] = "no splat in the scene"
        _write_manifest(result)
        return result
    cloud = mesh_export.gaussians_to_point_cloud(tensors["means"], tensors["scales"], tensors["rotations"],
                                                 tensors["colors"], tensors["opacity"])
    dense = package / "dense.ply"
    if not dense.is_file() or result.files.get("dense.ply", {}).get("source") != "colmap_stereo":
        on_progress(_tr("video_to_splat.progress.writing_cloud", "Writing dense.ply"))
        count = mesh_export.write_point_cloud_ply(dense, cloud)
        record(dense, "gaussians")
        if count < 4:
            result.skipped["mesh.ply"] = "too few points"
            _write_manifest(result)
            return result

    # 3. Poisson mesh
    mesh = package / "mesh.ply"
    if hooks.colmap is None:
        result.skipped["mesh.ply"] = "COLMAP not resolved"
    else:
        on_progress(_tr("video_to_splat.progress.meshing_cloud", "Meshing dense.ply"))
        completed = hooks.run(plan_poisson_command(hooks.colmap, dense, mesh), lambda _line: None, should_cancel)
        if completed.returncode != 0 or not mesh.is_file():
            result.skipped["mesh.ply"] = _tr("video_to_splat.error.tool_failed", "COLMAP {tool} failed with exit code {code}",
                                             tool="poisson_mesher", code=completed.returncode)
        else:
            record(mesh, "poisson_mesher")

    # 4. GLB and USDZ
    if mesh.is_file():
        try:
            data = mesh_export.read_ply_mesh(mesh)
            on_progress(_tr("video_to_splat.progress.writing_mesh", "Writing mesh.glb and mesh.usdz"))
            mesh_export.write_glb(package / "mesh.glb", data["positions"], data["faces"], data["colors"], name=name)
            record(package / "mesh.glb", "mesh.ply")
            mesh_export.write_usdz(package / "mesh.usdz", data["positions"], data["faces"], data["colors"], name=name)
            record(package / "mesh.usdz", "mesh.ply")
        except Exception as exc:  # noqa: BLE001 - a bad mesh must not lose the rest of the package
            result.skipped["mesh.glb"] = str(exc)
    _write_manifest(result)
    return result


def _write_manifest(result: FinalizeResult) -> None:
    (result.package / MANIFEST).write_text(json.dumps(result.manifest(), indent=1), encoding="utf-8")
