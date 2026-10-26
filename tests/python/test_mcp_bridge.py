# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The lazy bridge must initialize a freshly launched HTTP server."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def bridge():
    path = Path(__file__).resolve().parents[2] / "scripts" / "lichtfeld_mcp_bridge.py"
    spec = importlib.util.spec_from_file_location("bridge_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_first_resource_request_initializes_http_server(bridge, monkeypatch):
    request = {"jsonrpc": "2.0", "id": 7, "method": "resources/list"}
    expected = {"jsonrpc": "2.0", "id": 7, "result": {"resources": []}}
    responses = iter([
        {"error": {"code": -32600, "message": "Server not initialized. Call 'initialize' first."}},
        {"result": {"capabilities": {}}},
        expected,
    ])
    calls = []

    def post(payload, **_kwargs):
        calls.append(payload)
        return next(responses)

    monkeypatch.setattr(bridge, "post_json", post)
    assert bridge.forward_message(request) == expected
    assert [call["method"] for call in calls] == ["resources/list", "initialize", "resources/list"]
    assert calls[0] == calls[2] == request


def test_other_errors_are_not_retried(bridge, monkeypatch):
    response = {"error": {"code": -32602, "message": "Invalid arguments"}}
    calls = []

    def post(payload, **_kwargs):
        calls.append(payload)
        return response

    monkeypatch.setattr(bridge, "post_json", post)
    assert bridge.forward_message({"method": "tools/call"}) == response
    assert len(calls) == 1


def test_handshake_failure_does_not_repeat_mutation(bridge, monkeypatch):
    responses = iter([
        {"error": {"code": -32600, "message": "Server not initialized"}},
        {"error": {"code": -32603, "message": "Startup failed"}},
    ])
    monkeypatch.setattr(bridge, "post_json", lambda *a, **kw: next(responses))
    with pytest.raises(RuntimeError, match="initialization failed"):
        bridge.forward_message({"method": "tools/call"})


def test_camera_runtime_isolates_inherited_application_root(monkeypatch, tmp_path):
    import os
    from types import SimpleNamespace
    from unittest.mock import Mock
    import test_api_camera_project_runtime as runtime

    inherited = tmp_path / "inherited-profile"
    expected = tmp_path / "home" / ".lichtfeld"
    monkeypatch.setenv("LFS_HOME", str(inherited))
    monkeypatch.setenv("LFS_EXECUTABLE", str(tmp_path / "LichtFeld-Studio"))

    def start():
        assert Path(os.environ["LFS_HOME"]) == expected
        assert Path(os.environ["HOME"]) == expected.parent

    bridge = SimpleNamespace(
        endpoint_ready=lambda: False,
        pick_launch_command=lambda: [],
        ensure_server_ready=start,
        cleanup_spawned_processes=Mock(),
    )
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _module: None))
    monkeypatch.setattr(runtime, "importlib", SimpleNamespace(util=SimpleNamespace(
        spec_from_file_location=lambda *_args: spec,
        module_from_spec=lambda _spec: bridge,
    )))
    monkeypatch.setattr(runtime, "_initialize", lambda _endpoint: None)
    monkeypatch.setattr(runtime, "_call", lambda _endpoint, method, *_args: {
        "tools": [{"name": name} for name in ("editor_run", "project_save_as", "project_open")]
    } if method == "tools/list" else {})
    fixture = runtime.camera_runtime.__wrapped__(tmp_path, monkeypatch)
    try:
        assert next(fixture).startswith("http://127.0.0.1:")
    finally:
        fixture.close()
    bridge.cleanup_spawned_processes.assert_called_once()
    assert not inherited.exists()
