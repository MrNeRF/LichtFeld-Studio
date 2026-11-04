# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The background runner behind the MCP tools: one pipeline at a time, observable status, cancel."""
import sys
import threading
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from lfs_plugins.video_to_splat import pipeline, runner  # noqa: E402


class ScriptedRun:
    """Stands in for pipeline.run: blocks until released, then returns or raises as scripted."""

    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()
        self.outcome = None  # a State, or an exception to raise
        self.requests = []
        self.cancel_probe = None

    def __call__(self, request, hooks, *, on_progress, should_cancel):
        self.requests.append(request)
        self.cancel_probe = should_cancel
        on_progress("extract", 0.25, "Frame 1/4", "")
        self.started.set()
        assert self.release.wait(5)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def make_request(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    return pipeline.Request(video=video, project=tmp_path / "out" / "clip.licht")


def test_runner_reports_progress_then_result(tmp_path):
    run = ScriptedRun()
    r = runner.PipelineRunner(run=run)
    assert r.status()["active"] is False and r.status()["done"] is False
    request = make_request(tmp_path)
    started = r.start(request, hooks=object())
    assert started["started"] is True
    assert Path(started["work_dir"]) == tmp_path / "out" / "clip"
    assert run.started.wait(5)
    status = r.status()
    assert status["active"] is True
    assert (status["stage"], status["fraction"], status["message"]) == ("extract", 0.25, "Frame 1/4")
    run.outcome = pipeline.State(fingerprint="x", frames=40, registered=38, mode="equirectangular")
    run.release.set()
    r.wait(5)
    status = r.status()
    assert status["active"] is False and status["done"] is True and status["error"] == ""
    assert (status["frames"], status["registered"], status["mode"]) == (40, 38, "equirectangular")
    assert status["package_dir"] == ""  # the fake State carries none
    assert Path(status["project"]) == request.project


def test_runner_refuses_a_second_run_while_one_is_active(tmp_path):
    run = ScriptedRun()
    r = runner.PipelineRunner(run=run)
    request = make_request(tmp_path)
    assert r.start(request, hooks=object())["started"] is True
    assert run.started.wait(5)
    second = r.start(request, hooks=object())
    assert second["started"] is False and "running" in second["error"].lower()
    run.outcome = pipeline.State(fingerprint="x")
    run.release.set()
    r.wait(5)
    assert r.start(request, hooks=object())["started"] is True  # free again
    run.release.set()
    r.wait(5)


def test_runner_captures_pipeline_errors_and_cancel(tmp_path):
    run = ScriptedRun()
    r = runner.PipelineRunner(run=run)
    request = make_request(tmp_path)
    r.start(request, hooks=object())
    assert run.started.wait(5)
    assert r.cancel()["cancelling"] is True
    assert run.cancel_probe() is True
    run.outcome = pipeline.PipelineCancelled()
    run.release.set()
    r.wait(5)
    status = r.status()
    assert status["done"] is True and status["cancelled"] is True and status["error"] == ""

    run = ScriptedRun()
    r = runner.PipelineRunner(run=run)
    r.start(request, hooks=object())
    assert run.started.wait(5)
    run.outcome = pipeline.PipelineError("COLMAP found no features")
    run.release.set()
    r.wait(5)
    status = r.status()
    assert status["done"] is True and status["error"] == "COLMAP found no features"
    assert r.cancel()["cancelling"] is False  # nothing to cancel


@pytest.fixture
def lf_stub(monkeypatch):
    for name in list(sys.modules):
        if name == "lfs_plugins" or name.startswith("lfs_plugins."):
            sys.modules.pop(name, None)
    state = SimpleNamespace(tools={}, training=False)
    stub = ModuleType("lichtfeld")
    stub.mcp = SimpleNamespace(
        register_tool=lambda fn, name="", description="": state.tools.__setitem__(name or fn.__name__, fn),
        unregister_tool=lambda name: state.tools.pop(name, None),
    )
    stub.is_training_active = lambda: state.training
    stub.ui = SimpleNamespace(tr=lambda key: key, schedule_on_ui_thread=lambda cb: cb())
    stub.register_class = lambda cls: None
    stub.unregister_class = lambda cls: None
    monkeypatch.setitem(sys.modules, "lichtfeld", stub)
    return stub, state


def test_mcp_tools_register_and_validate(lf_stub, tmp_path, monkeypatch):
    stub, state = lf_stub
    mcp_tools = import_module("lfs_plugins.video_to_splat.mcp_tools")
    mcp_tools.register_mcp_tools()
    assert set(state.tools) == {"video_to_splat_start", "video_to_splat_status", "video_to_splat_cancel"}

    run = ScriptedRun()
    monkeypatch.setattr(mcp_tools, "_runner", runner.PipelineRunner(run=run))
    monkeypatch.setattr(mcp_tools, "build_hooks", lambda: object())
    start = state.tools["video_to_splat_start"]
    missing = start(source=str(tmp_path / "nope.mp4"), project=str(tmp_path / "p.licht"))
    assert missing["started"] is False and "nope.mp4" in missing["error"]
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    bad_preset = start(source=str(video), project=str(tmp_path / "p.licht"), preset="ultra")
    assert bad_preset["started"] is False and "ultra" in bad_preset["error"]
    state.training = True
    busy = start(source=str(video), project=str(tmp_path / "p.licht"))
    assert busy["started"] is False and busy["error"] == "video_to_splat.training_busy"
    state.training = False

    started = start(source=str(video), project=str(tmp_path / "room"), mode="on", preset="fast", use_gpu=False,
                    mask_people=False, build_mesh=True)
    assert started["started"] is True
    assert Path(started["project"]) == tmp_path / "room.licht"  # the extension is added
    assert run.started.wait(5)
    request = run.requests[0]
    assert (request.mode_override, request.preset, request.use_gpu) == ("on", "fast", False)
    assert (request.mask_people, request.build_mesh) == (False, True)
    assert state.tools["video_to_splat_status"]()["active"] is True
    assert state.tools["video_to_splat_cancel"]()["cancelling"] is True
    run.outcome = pipeline.PipelineCancelled()
    run.release.set()
    mcp_tools._runner.wait(5)
    assert state.tools["video_to_splat_status"]()["cancelled"] is True

    mcp_tools.unregister_mcp_tools()
    assert state.tools == {}


def test_plugin_register_adds_the_mcp_tools(lf_stub):
    stub, state = lf_stub
    package = import_module("lfs_plugins.video_to_splat")
    package.register()
    assert "video_to_splat_start" in state.tools
    package.unregister()
    assert state.tools == {}
