# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Real renderer to EXR contract, against an empty disposable MCP viewer."""
import os
import time
from pathlib import Path
import pytest
from test_render_settings_publication_runtime import endpoint
from test_render_on_demand_idle import _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_real_float_color_capture(endpoint):
    source = Path(os.environ.get("LFS_TEST_MCP_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
    script = source / "tests/python/float_image_capture_contract.py"
    result = _tool(endpoint, "editor_run", {
        "code": f"from pathlib import Path\np={str(script)!r}\nexec(Path(p).read_text(encoding='utf-8'), {{'__file__':p}})",
        "show_console": False,
        "timeout_ms": 10000,
    })
    deadline = time.monotonic() + 180
    while not result["completed"] and time.monotonic() < deadline:
        result = _tool(endpoint, "editor_wait", {
            "timeout_ms": 2000, "wait_for_completion": True,
            "output_max_chars": 20000,
        })
    assert result["completed"] and not result["timed_out"], result
    output = result["output"]["text"]
    assert "Traceback" not in output, output
    assert "PHASE5A_FLOAT_RENDER_PASS" in output, output
    assert "PHASE5A_TONE_TILING_PASS" in output, output
