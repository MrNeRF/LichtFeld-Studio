# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise live reveal controls through MCP on an isolated app instance."""
import base64
import io
import os
import time

from runtime_test_app import isolated_app
from test_render_on_demand_idle import _call

import pytest
from PIL import Image, ImageChops

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("with_project", [False, True], ids=["controls", "project"])
def test_live_reveal_operations(tmp_path, with_project):
    if not os.environ.get("LFS_EXECUTABLE"):
        pytest.skip("requires LFS_EXECUTABLE")
    project = os.environ.get("LFS_REVEAL_TEST_PROJECT") if with_project else None
    if with_project and not project:
        pytest.skip("requires LFS_REVEAL_TEST_PROJECT")
    with isolated_app(tmp_path) as endpoint:
        def call(method, params=None):
            return _call(endpoint, method, params)
        def tool(name, arguments=None):
            response = call("tools/call", {"name": name, "arguments": arguments or {}})
            assert not response.get("isError"), response
            return response["structuredContent"]
        tools = {item["name"] for item in call("tools/list")["tools"]}
        assert {"sequencer_reveal_get", "sequencer_reveal_set", "sequencer_reveal_remove"} <= tools
        clip = {"start": 1.0, "end": 4.0, "softness": 0.2, "effect": 2}
        result = tool("sequencer_reveal_set", clip)
        assert result["success"] is True
        assert tool("sequencer_reveal_get")["reveal"] == pytest.approx(clip)
        bad = call("tools/call", {"name": "sequencer_reveal_set", "arguments": {**clip, "end": 0.5}})
        assert bad.get("isError") is True, bad
        assert tool("sequencer_reveal_get")["reveal"] == pytest.approx(clip)
        assert tool("sequencer_reveal_remove")["success"] is True
        assert tool("sequencer_reveal_get")["reveal"] is None
        tool("history_undo")
        assert tool("sequencer_reveal_get")["reveal"] == pytest.approx(clip)
        tool("history_redo")
        assert tool("sequencer_reveal_get")["reveal"] is None
        result = tool("editor_run", {"code": """
import lichtfeld as lf
clip = {'start': 1.0, 'end': 4.0, 'softness': 0.2, 'effect': 2}
assert lf.ui.set_reveal_clip(clip)
assert lf.ui.get_reveal_clip()['effect'] == 2
assert not lf.ui.set_reveal_clip({**clip, 'end': 0.5})
assert lf.ui.get_reveal_clip()['end'] == 4.0
assert lf.ui.remove_reveal_clip()
assert lf.ui.get_reveal_clip() is None
""", "show_console": False, "timeout_ms": 10000})
        if result.get("running"):
            result = tool("editor_wait", {"timeout_ms": 10000})
        assert result.get("completed") and result.get("success"), result
        assert "Traceback" not in result.get("output", {}).get("text", ""), result
        tool("history_undo")
        assert tool("sequencer_reveal_get")["reveal"] == pytest.approx(clip)
        tool("history_redo")
        assert tool("sequencer_reveal_get")["reveal"] is None
        if project:
            tool("project_open", {"path": project, "discard_changes": True})
            deadline = time.monotonic() + 60
            while True:
                info = tool("project_get_info")
                if info["hydration_state"] == "complete":
                    break
                assert info["hydration_state"] != "failed" and time.monotonic() < deadline, info
                time.sleep(0.2)
            tool("camera_reset")
            assert tool("sequencer_reveal_set", clip)["success"]
            tool("sequencer_scrub", {"time": 2.0})
            capture = tool("render_capture", {"width": 640})
            assert capture["success"] and capture["data"]
            reveal_image = base64.b64decode(capture["data"])
            (tmp_path / "reveal.png").write_bytes(reveal_image)
            tool("sequencer_scrub", {"time": clip["end"]})
            normal = tool("render_capture", {"width": 640})
            assert normal["success"] and normal["data"]
            normal_image = base64.b64decode(normal["data"])
            (tmp_path / "normal.png").write_bytes(normal_image)
            with Image.open(io.BytesIO(reveal_image)) as reveal, Image.open(io.BytesIO(normal_image)) as plain:
                assert ImageChops.difference(reveal.convert("RGB"), plain.convert("RGB")).getbbox(), "reveal and normal viewport are identical"
            rectangle = {"x0": 10, "y0": 10, "x1": 100, "y1": 100}
            for effect in (1, 2, 3, 4, 5, 6, 7, 8, 10):
                assert tool("sequencer_reveal_set", {**clip, "effect": effect})["success"]
                for scene_time in (0.0, 2.0):
                    tool("sequencer_scrub", {"time": scene_time})
                    before = tool("selection_get")
                    blocked = call("tools/call", {"name": "selection_rect", "arguments": rectangle})
                    assert blocked.get("isError") is True, (effect, scene_time, blocked)
                    assert "reveal" in str(blocked).lower(), blocked
                    assert tool("selection_get") == before
                tool("sequencer_scrub", {"time": clip["end"]})
                tool("selection_rect", rectangle)
            tool("sequencer_scrub", {"time": 2.0})
            for effect in (9, 11):
                assert tool("sequencer_reveal_set", {**clip, "effect": effect})["success"]
                tool("selection_rect", rectangle)
            tool("sequencer_reveal_remove")
            tool("selection_rect", rectangle)
            for effect in range(1, 12):
                assert tool("sequencer_reveal_set", {**clip, "effect": effect})["success"]
                frame = tool("render_capture_window", {"width": 320})
                assert frame["success"] and frame["data"], effect
            tool("render_settings_set", {"raster_backend": "3dgut"})
            assert tool("sequencer_reveal_set", {**clip, "effect": 5})["success"]
            gut_reveal = tool("render_capture_window", {"width": 320})
            assert gut_reveal["success"] and gut_reveal["data"]
            tool("sequencer_scrub", {"time": clip["end"]})
            gut_normal = tool("render_capture_window", {"width": 320})
            assert gut_normal["success"] and gut_normal["data"]
            movie = tmp_path / "reveal.mp4"
            callback_log = tmp_path / "scene_times.txt"
            assert tool("sequencer_reveal_set", {"start": 0.0, "end": 1.0, "softness": 0.2, "effect": 2})["success"]
            result = tool("editor_run", {"code": f'''
import lichtfeld as lf
lf.on_scene_time(lambda t: open({str(callback_log)!r}, "a").write(str(t) + "\\n"))
lf.ui.export_video(320, 180, 8, 27, {str(movie)!r}, include_provenance=False)
''', "show_console": False, "timeout_ms": 10000})
            if result.get("running"):
                result = tool("editor_wait", {"timeout_ms": 10000})
            assert result.get("completed") and result.get("success"), result
            job = tool("runtime_job_wait", {"job_id": "export.video", "until": "inactive", "timeout_ms": 60000})
            assert job["outcome"] == "completed", job
            assert movie.stat().st_size > 0
            assert len(callback_log.read_text().splitlines()) > 1
