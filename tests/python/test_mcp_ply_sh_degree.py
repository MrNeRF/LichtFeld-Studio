# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""MCP PLY export keeps the source SH layout unless a lower degree is requested."""

import json
import struct

import pytest

from runtime_test_app import isolated_app
from test_render_on_demand_idle import _call, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


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


def test_mcp_ply_export_preserves_available_sh_and_rejects_over_high(tmp_path):
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

            if degree < 3:
                rejected = tmp_path / f"rejected-{degree}.ply"
                result = _call(endpoint, "tools/call", {"name": "scene_export_ply", "arguments": {
                    "path": str(rejected), "sh_degree": degree + 1}})
                assert result["isError"], result
                assert not rejected.exists()

        invalid = tmp_path / "invalid.ply"
        for degree in (-1, 4):
            result = _call(endpoint, "tools/call", {"name": "scene_export_ply", "arguments": {
                "path": str(invalid), "sh_degree": degree}})
            assert result["isError"], result
            assert not invalid.exists()

        for format_name, extension in (("sog", ".sog"), ("ssog", ".ssog"),
                                       ("spz", ".spz"), ("usd", ".usd"),
                                       ("usdz_nurec", ".usdz"), ("html", ".html"),
                                       ("rad", ".rad")):
            rejected = tmp_path / f"rejected-{format_name}{extension}"
            result = _call(endpoint, "tools/call", {"name": f"scene_export_{format_name}", "arguments": {
                "path": str(rejected), "sh_degree": 4}})
            assert result["isError"], result
            assert "SH degree" in json.dumps(result), result
            assert not rejected.exists()
