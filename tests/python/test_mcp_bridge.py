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
