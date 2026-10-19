# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""MCP PLY export keeps the source SH layout unless a lower degree is requested."""

import hashlib
import struct

import pytest

from runtime_test_app import isolated_app
from test_render_on_demand_idle import _call, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

_BASELINE_SH3_PLY_CONTENT_SHA256 = "797bce55539bd1683663646c9e5ef12673cb3cd93c501eaf0def9e502e6c1e0a"


def _model(path, degree):
    properties = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    properties += [f"f_rest_{index}" for index in range(3 * ((degree + 1) ** 2 - 1))]
    properties += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
    header += "".join(f"property float {name}\n" for name in properties) + "end_header\n"
    values = [0.0] * len(properties)
    values[properties.index("opacity")] = 1.0
    values[properties.index("rot_0")] = 1.0
    for index in range(3 * ((degree + 1) ** 2 - 1)):
        values[properties.index(f"f_rest_{index}")] = 0.01 * (index + 1)
    path.write_bytes(header.encode() + struct.pack(f"<{len(values)}f", *values))


def _rest_count(path):
    header = path.read_bytes().split(b"end_header\n", 1)[0]
    return sum(line.startswith(b"property float f_rest_") for line in header.splitlines())


def _without_build_provenance(data):
    marker = b"comment lichtfeld_provenance "
    start = data.index(marker)
    end = data.index(b"\n", start)
    return data[:start] + b"comment lichtfeld_provenance <build-specific>\n" + data[end + 1:]


def _lower_duplicate_sh_degree(endpoint, source_name, degree):
    duplicate = _tool(endpoint, "scene_duplicate_node", {"name": source_name})
    assert duplicate["success"], duplicate
    duplicate_name = duplicate["nodes"][0]["name"]
    result = _tool(endpoint, "editor_run", {
        "code": "import lichtfeld as lf\n"
                f"lf.get_scene().get_node({duplicate_name!r}).splat_data().set_max_sh_degree({degree})",
        "show_console": False, "wait_for_completion": True, "timeout_ms": 10000,
    })
    assert result.get("completed") and "Traceback" not in result.get("output", {}).get("text", ""), result
    return duplicate_name


def test_mcp_ply_export_preserves_available_sh_and_clamps_requested_degree(tmp_path):
    with isolated_app(tmp_path) as endpoint:
        for degree in range(4):
            source = tmp_path / f"source-{degree}.ply"
            _model(source, degree)
            assert _tool(endpoint, "scene_load_ply", {"path": str(source)})["success"]
            assert _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "timeout_ms": 60000})["success"]

            default = tmp_path / f"default-{degree}.ply"
            assert _tool(endpoint, "scene_export_ply", {"path": str(default), "include_provenance": False})["success"]
            assert _rest_count(default) == 3 * ((degree + 1) ** 2 - 1)

            explicit = tmp_path / f"explicit-{degree}.ply"
            assert _tool(endpoint, "scene_export_ply", {
                "path": str(explicit), "sh_degree": degree, "include_provenance": False})["success"]
            assert default.read_bytes() == explicit.read_bytes()

            lower = tmp_path / f"lower-{degree}.ply"
            assert _tool(endpoint, "scene_export_ply", {
                "path": str(lower), "sh_degree": 0, "include_provenance": False})["success"]
            assert _rest_count(lower) == 0

            over_high = tmp_path / f"over-high-{degree}.ply"
            clamped = _tool(endpoint, "scene_export_ply", {
                "path": str(over_high), "sh_degree": degree + 3, "include_provenance": False})
            assert clamped["success"], clamped
            assert "warning" in clamped, clamped
            assert f"clamped to {degree}" in clamped["warning"].lower(), clamped
            assert _rest_count(over_high) == 3 * ((degree + 1) ** 2 - 1)
            assert over_high.read_bytes() == default.read_bytes()

        # The default SH3 path stays byte-identical to an explicit SH3 request.
        sh3_default = (tmp_path / "default-3.ply").read_bytes()
        assert sh3_default == (tmp_path / "explicit-3.ply").read_bytes()
        # The minimal provenance comment embeds the build commit; compare all other bytes
        # against a SH3 export from the pre-fix build.
        sh3_content = _without_build_provenance(sh3_default)
        assert hashlib.sha256(sh3_content).hexdigest() == _BASELINE_SH3_PLY_CONTENT_SHA256

        models = [node["name"] for node in _tool(endpoint, "scene_list_nodes")["nodes"]
                  if node["type"] == "splat"]
        assert len(models) == 1, models
        lower_model = _lower_duplicate_sh_degree(endpoint, models[0], 1)
        mixed_nodes = [models[0], lower_model]
        mixed_path = tmp_path / "mixed-clamped.ply"
        mixed_result = _tool(endpoint, "scene_export_ply", {
            "path": str(mixed_path), "nodes": mixed_nodes, "sh_degree": 3,
            "include_provenance": False})
        assert mixed_result["success"], mixed_result
        assert "warning" in mixed_result, mixed_result
        assert "clamped to 1" in mixed_result["warning"].lower(), mixed_result
        assert _rest_count(mixed_path) == 9

        for format_name, extension in (("sog", ".sog"), ("ssog", ".ssog"),
                                       ("spz", ".spz"), ("usd", ".usd"),
                                       ("usdz_nurec", ".usdz"), ("html", ".html"),
                                       ("rad", ".rad")):
            clamped_path = tmp_path / f"clamped-{format_name}{extension}"
            result = _tool(endpoint, f"scene_export_{format_name}", {
                "path": str(clamped_path), "nodes": mixed_nodes, "sh_degree": 4,
                "include_provenance": False})
            assert result["success"], result
            assert "warning" in result, result
            assert "clamped to 1" in result["warning"].lower(), result
            assert clamped_path.exists(), clamped_path
