# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The download package once a splat exists: exports, dense cloud, Poisson mesh, GLB and USDZ."""
import json
import struct
import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from lfs_plugins.video_to_splat import finalize, mesh_export  # noqa: E402

np = pytest.importorskip("numpy")


def tensors():
    return {"means": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0.1]],
            "scales": [[1, 1, 0.01]] * 5, "rotations": [[1, 0, 0, 0]] * 5,
            "colors": [[1, 0, 0]] * 5, "opacity": [0.9] * 5}


class Fakes:
    def __init__(self, colmap_rc=0):
        self.exports = []
        self.commands = []
        self.colmap_rc = colmap_rc
        self.export_error = None

    def export_splat(self, code, path):
        self.exports.append((code, path.name))
        if self.export_error:
            return self.export_error
        path.write_bytes(b"splat")
        return None

    def run(self, cmd, on_output, should_cancel):
        self.commands.append(cmd)
        if cmd[1] == "poisson_mesher" and self.colmap_rc == 0:
            # a quad from the cloud's first four points, like a real mesher would
            cloud = mesh_export.read_ply_mesh(Path(cmd[cmd.index("--input_path") + 1]))
            out = Path(cmd[cmd.index("--output_path") + 1])
            header = ("ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\n"
                      "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
                      "element face 2\nproperty list uchar int vertex_indices\nend_header\n")
            body = b"".join(struct.pack("<fffBBB", *cloud["positions"][i], *cloud["colors"][i]) for i in range(4))
            body += struct.pack("<Biii", 3, 0, 1, 2) + struct.pack("<Biii", 3, 1, 3, 2)
            out.write_bytes(header.encode() + body)
        return CompletedProcess(cmd, self.colmap_rc, "", "")

    def hooks(self, with_splat=True):
        return finalize.FinalizeHooks(splat_tensors=lambda: tensors() if with_splat else None,
                                      export_splat=self.export_splat, run=self.run, colmap=Path("colmap.exe"))


def test_finalize_writes_every_package_file_and_a_manifest(tmp_path):
    fakes = Fakes()
    messages = []
    result = finalize.finalize_package(tmp_path / "package", fakes.hooks(), on_progress=messages.append, name="room")
    assert [e[1] for e in fakes.exports] == ["splat.ply", "splat.spz", "splat.sog", "splat.glb"]
    assert [e[0] for e in fakes.exports] == [0, 2, 1, 13]
    package = tmp_path / "package"
    for name in ("splat.ply", "splat.spz", "splat.sog", "splat.glb", "dense.ply", "mesh.ply", "mesh.glb", "mesh.usdz"):
        assert (package / name).is_file(), name
    poisson = fakes.commands[0]
    assert poisson[1] == "poisson_mesher" and poisson[poisson.index("--PoissonMeshing.color") + 1] == "1"
    manifest = json.loads((package / "package.json").read_text())
    assert manifest["files"]["dense.ply"]["source"] == "gaussians"
    assert manifest["files"]["mesh.glb"]["source"] == "mesh.ply" and manifest["skipped"] == {}
    assert result.files["mesh.usdz"]["bytes"] > 0
    glb = mesh_export.read_glb(package / "mesh.glb")["json"]
    assert glb["meshes"][0]["name"] == "room" and glb["accessors"][glb["meshes"][0]["primitives"][0]["indices"]]["count"] == 6
    assert any("dense.ply" in m for m in messages)


def test_finalize_without_a_splat_only_records_the_reason(tmp_path):
    fakes = Fakes()
    result = finalize.finalize_package(tmp_path / "package", fakes.hooks(with_splat=False))
    assert result.skipped["dense.ply"] == "no splat in the scene"
    assert not (tmp_path / "package" / "mesh.ply").exists()
    assert (tmp_path / "package" / "package.json").is_file()


def test_finalize_survives_a_failed_mesher_and_a_failed_export(tmp_path):
    fakes = Fakes(colmap_rc=3)
    fakes.export_error = "exporter busy"
    result = finalize.finalize_package(tmp_path / "package", fakes.hooks())
    assert (tmp_path / "package" / "dense.ply").is_file()
    assert "poisson_mesher" in result.skipped["mesh.ply"]
    assert result.skipped["splat.ply"] == "exporter busy" and "splat.ply" not in result.files
    assert not (tmp_path / "package" / "mesh.glb").exists()


def test_finalize_honours_cancel_before_exports(tmp_path):
    fakes = Fakes()
    result = finalize.finalize_package(tmp_path / "package", fakes.hooks(), should_cancel=lambda: True)
    assert fakes.exports == []
    assert (tmp_path / "package" / "dense.ply").is_file()  # the cheap local steps still run
    assert "poisson_mesher" in [c[1] for c in fakes.commands]
    assert result.files
