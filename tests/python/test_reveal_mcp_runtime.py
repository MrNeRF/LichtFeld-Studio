# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise live reveal controls through MCP on an isolated app instance."""
import base64
import io
import os
import json
import shutil
import subprocess
from uuid import uuid4
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


def test_custom_reveal_identity_ownership_and_pinned_export(tmp_path):
    if not os.environ.get("LFS_EXECUTABLE") or not shutil.which("ffmpeg"):
        pytest.skip("requires LFS_EXECUTABLE, Xvfb and ffmpeg")
    from test_shader_editor_runtime import _fixture
    from test_render_on_demand_idle import _tool
    fixture = tmp_path / "generated-reveal.ply"
    _fixture(fixture)
    with isolated_app(tmp_path) as endpoint:
        def tool(name, args=None):
            return _tool(endpoint, name, args)

        def python(code):
            result = tool("editor_run", {"code": code, "show_console": False, "timeout_ms": 30000})
            assert result["completed"] and result["success"] and "Traceback" not in result["output"]["text"], result
            return result

        def rejected(name, args):
            result = _call(endpoint, "tools/call", {"name": name, "arguments": args})
            assert result.get("isError"), result

        def capture(time):
            tool("sequencer_scrub", {"time": time})
            data = tool("render_capture", {"width": 320})["data"]
            return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

        tool("scene_load_ply", {"path": str(fixture)})
        tool("camera_reset")
        source = ("import lfs;\nLfsElement effect(LfsElement e, LfsContext c) {"
                  " e.color = lerp(float3(1, 0, 0), e.color, c.progress); return e; }\n")
        asset = tool("shader_set_source", {"source": source, "name": "Custom reveal"})["shader_id"]
        clip = {"start": .25, "end": 1.0, "softness": .25, "effect": 12, "shader_id": asset}
        rejected("shader_bind", {"shader_id": asset, "target": {"kind": "reveal"}})
        tool("sequencer_reveal_set", clip)
        assert tool("sequencer_reveal_get")["reveal"] == clip
        for invalid in ({**clip, "shader_id": str(uuid4())}, {**clip, "effect": 2},
                        {k: v for k, v in clip.items() if k != "shader_id"}):
            rejected("sequencer_reveal_set", invalid)
            assert tool("sequencer_reveal_get")["reveal"] == clip
        rejected("shader_remove", {"shader_id": asset})
        python(f"""
import lichtfeld as lf
assert lf.ui.get_reveal_clip() == {clip!r}
assert not lf.ui.set_reveal_clip({{**{clip!r}, 'effect': 2}})
assert lf.ui.get_reveal_clip() == {clip!r}
assert lf.shaders.wait({asset!r}, timeout_ms=30000)['outcome'] == 'applied'
print('CUSTOM_REVEAL_READY')
""")
        normal = capture(1.0)
        before = capture(0.0)
        start = capture(.25)
        middle = capture(.625)
        assert ImageChops.difference(before, start).getbbox() is None
        assert ImageChops.difference(before, normal).getbbox()
        assert ImageChops.difference(middle, before).getbbox()
        assert ImageChops.difference(middle, normal).getbbox()
        assert tool("shader_get", {"shader_id": asset})["bindings"] == [{"kind": "reveal"}]
        rejected("shader_remove", {"shader_id": asset})  # retained after the endpoint
        tool("shader_unbind", {"target": {"kind": "reveal"}, "shader_id": asset})
        assert tool("sequencer_reveal_get")["reveal"] is None
        assert ImageChops.difference(capture(1.0), normal).getbbox() is None
        tool("history_undo")
        assert tool("sequencer_reveal_get")["reveal"] == clip
        tool("history_redo")
        assert tool("sequencer_reveal_get")["reveal"] is None
        tool("sequencer_reveal_set", {"start": .25, "end": 1.0, "softness": .25, "effect": 2})
        document = tool("editor_new_document", {"code": source, "mode": "shader"})["document_id"]
        tool("editor_set_mode", {"shader_id": asset, "mode": "shader"})
        tool("editor_set_target", {"target": {"kind": "reveal"}})
        tool("sequencer_scrub", {"time": .625})
        run = tool("editor_run", {"timeout_ms": 30000})
        assert run["owns_binding"] and run["outcome"] == "applied", run
        assert tool("sequencer_reveal_get")["reveal"] == clip
        tool("sequencer_reveal_set", {**clip, "softness": .3})
        assert tool("editor_get_code")["owns_binding"]  # timing-only edits preserve Run ownership
        tool("shader_bind", {"shader_id": asset, "target": {"kind": "reveal"}})
        assert tool("editor_get_code")["owns_binding"]  # a rebind to the same asset stays the Run's
        tool("editor_interrupt")
        assert tool("sequencer_reveal_get")["reveal"] is None
        tool("history_undo")
        assert tool("sequencer_reveal_get")["reveal"]["shader_id"] == asset
        assert not tool("editor_get_code")["owns_binding"]
        tool("editor_run", {"timeout_ms": 30000})
        tool("editor_interrupt")
        assert tool("sequencer_reveal_get")["reveal"] is None
        tool("history_undo")
        assert tool("sequencer_reveal_get")["reveal"]["shader_id"] == asset
        assert not tool("editor_get_code", {"document_id": document})["owns_binding"]
        tool("editor_new_document")
        tool("sequencer_reveal_set", clip)
        timeline_path = tmp_path / "short-reveal.json"
        tool("sequencer_save_path", {"path": str(timeline_path)})
        timeline = json.loads(timeline_path.read_text())
        timeline["clip_duration"] = 1.25
        timeline_path.write_text(json.dumps(timeline))
        tool("sequencer_load_path", {"path": str(timeline_path)})
        assert tool("sequencer_reveal_get")["reveal"] == clip

        def export(path, mutate):
            python(f"""
lf.clear_scene_time()
export_edits = 0
def edit_live_reveal(t):
    global export_edits
    if export_edits == 0:
        assert lf.ui.set_reveal_clip({{'start': 0.0, 'end': 2.0, 'softness': .8, 'effect': 2}})
        lf.shaders.set_source('import lfs; LfsElement effect(LfsElement e, LfsContext c) {{ e.color = float3(0,1,0); return e; }}', shader_id={asset!r})
    export_edits += 1
if {mutate!r}:
    lf.on_scene_time(edit_live_reveal)
lf.ui.export_video(320, 180, 8, 27, {str(path)!r}, include_provenance=False)
""")
            job = tool("runtime_job_wait", {"job_id": "export.video", "until": "inactive", "timeout_ms": 60000})
            assert job["outcome"] == "completed", job
            assert path.stat().st_size > 0
            if mutate:
                python("lf.clear_scene_time(); assert export_edits > 1")
            return subprocess.check_output([shutil.which("ffmpeg"), "-v", "error", "-i", str(path),
                                            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])

        reference = export(tmp_path / "reference.mp4", False)
        changed = export(tmp_path / "changed-live.mp4", True)
        assert changed == reference, "live clip/source edits changed the pinned export"
        assert tool("sequencer_reveal_get")["reveal"]["effect"] == 2
