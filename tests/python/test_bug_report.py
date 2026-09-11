# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract tests for in-app bug-report collection and submission."""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from lfs_plugins import bug_report, bug_report_panel, portal_account


class Value:
    def __init__(self, value):
        self.value = value


class FakeConfig:
    def __init__(self, values=None, *, available=True):
        self._values = values or {}
        self._available = available

    def has_params(self):
        return self._available

    def properties(self):
        if not self._available:
            raise AssertionError("properties() must be guarded by has_params()")
        return [
            {"id": key, "name": key, "group": "test", "value": value}
            for key, value in self._values.items()
        ]


class FakeResponse:
    def __init__(self, status: int, payload=None, headers=None):
        self.status = status
        self.headers = headers or {}
        self._raw = (
            b""
            if payload is None
            else payload
            if isinstance(payload, bytes)
            else json.dumps(payload).encode("utf-8")
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def getcode(self):
        return self.status

    def read(self):
        return self._raw


class StubUrlopen:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.requests = []

    def __call__(self, request, *, timeout, **kwargs):
        del timeout, kwargs
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"Unexpected request: {request.full_url}")
        response = self.responses.popleft()
        if isinstance(response, tuple):
            return FakeResponse(*response)
        return FakeResponse(200, response)


@pytest.fixture
def fake_runtime(monkeypatch):
    state = SimpleNamespace(
        training_running=Value(False),
        trainer_loaded=Value(False),
        bug_report_state=Value({}),
    )
    monkeypatch.setattr(bug_report, "RuntimeState", state)
    return state


def make_fake_lf(dataset=None, optimization=None):
    return SimpleNamespace(
        build_info=SimpleNamespace(
            version="v0.5.3-2-gabcdef",
            commit="abcdef123",
            build_type="Release",
        ),
        diagnostics=SimpleNamespace(
            collect=lambda: {
                "os": "Test OS",
                "os_build": "1.0",
                "cpu": "Test CPU",
                "ram_mb": 32768,
                "gpu": "Test GPU",
                "gpu_driver": "570.86",
                "cuda_runtime": "12.8",
                "vram_mb": 24564,
                "vram_used_mb": 2048,
            }
        ),
        dataset_params=lambda: FakeConfig(dataset),
        optimization_params=lambda: FakeConfig(optimization),
        trainer_current_iteration=lambda: 3000,
        trainer_total_iterations=lambda: 30000,
        get_num_gaussians=lambda: 1_500_000,
        get_render_mode=lambda: SimpleNamespace(name="SPLATS"),
        log=SimpleNamespace(
            buffered_text=lambda _max_bytes=bug_report.LOG_MAX_BYTES: "",
            previous_session=lambda: None,
        ),
        ui=SimpleNamespace(open_url=lambda _url: None),
    )


def recursive_strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from recursive_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from recursive_strings(child)
    elif isinstance(value, str):
        yield value


def valid_form():
    return {
        "category": "crash",
        "title": "Training freezes after a few minutes",
        "expected": "Training should continue until all iterations finish.",
        "actual": "The interface stopped responding during densification.",
        "step_1": "Load the dataset",
        "step_2": "Start training",
        "step_3": "Wait for densification",
        "frequency": "sometimes",
        "stage": "training",
    }


def write_credentials(path: Path):
    path.parent.mkdir(parents=True)
    now = time.time()
    path.write_text(
        json.dumps(
            {
                "version": portal_account.CREDENTIALS_VERSION,
                "portal_origin": portal_account.DEFAULT_PORTAL_URL,
                "access_token": "access-old",
                "access_expires_at": now + 3600,
                "refresh_token": "refresh-old",
                "refresh_expires_at": now + 86400,
                "display_name": "Ada Lovelace",
                "email": "ada@example.com",
                "customer_tier": "Professional",
                "member_since": "2025-01-02T03:04:05Z",
                "connected_since": "2026-02-03T04:05:06Z",
            }
        ),
        encoding="utf-8",
    )


def rotated_tokens():
    return {
        "access_token": "access-rotated",
        "expires_in": 3600,
        "refresh_token": "refresh-rotated",
        "refresh_expires_in": 86400,
        "token_type": "Bearer",
    }


def success_response():
    return {
        "public_id": "3c516e3c-d95b-4a2c-a8c2-e482a78ec277",
        "url": "https://portal.lichtfeld.io/bugs/3c516e3c-d95b-4a2c-a8c2-e482a78ec277/",
        "status": "submitted",
        "relayed": True,
        "completeness_problems": [],
    }


def test_payload_never_contains_dataset_or_output_paths(tmp_path, monkeypatch, fake_runtime):
    dataset = tmp_path / "private-dataset"
    output = tmp_path / "private-output"
    fake_runtime.training_running.value = True
    fake_lf = make_fake_lf(
        dataset={
            "data_path": str(dataset),
            "output_path": str(output),
            "resize_factor": -1,
            "max_width": 3840,
        },
        optimization={"strategy": "mcmc", "sh_degree": 3, "max_cap": 2_000_000},
    )
    monkeypatch.setattr(bug_report, "lf", fake_lf)
    monkeypatch.setattr(
        bug_report,
        "get_portal_account_service",
        lambda: SimpleNamespace(_redaction_tokens=lambda: ()),
    )

    form = valid_form()
    form["actual"] = f"Input {dataset} unexpectedly wrote to {output}."
    payload = bug_report.build_payload(
        form,
        consent_to_logs=True,
        current_log_text=f"dataset={dataset}\noutput={output}\n",
    )

    serialized_strings = "\n".join(recursive_strings(payload))
    assert str(dataset) not in serialized_strings
    assert str(output) not in serialized_strings
    assert "data_path" not in serialized_strings
    assert "output_path" not in serialized_strings
    assert payload["diagnostics"]["num_gaussians"] == 1_500_000
    assert "max_cap" not in payload["diagnostics"]
    assert "image_count" not in payload["diagnostics"]


def test_redact_requires_path_boundary(monkeypatch):
    monkeypatch.setattr(bug_report, "lf", make_fake_lf())
    monkeypatch.setattr(
        bug_report,
        "get_portal_account_service",
        lambda: SimpleNamespace(_redaction_tokens=lambda: ()),
    )

    text = "/data/scenes/x/file /data/scenes/xy/file /home/alice/file /home/alicefoo/file"
    assert bug_report._replace_path(text, "/data/scenes/x", "<dataset>") == (
        "<dataset>/file /data/scenes/xy/file /home/alice/file /home/alicefoo/file"
    )
    assert bug_report._replace_path(text, "/home/alice", "~") == (
        "/data/scenes/x/file /data/scenes/xy/file ~/file /home/alicefoo/file"
    )
    assert bug_report._replace_path("/data/scenes/x", "/data/scenes/x", "<dataset>") == "<dataset>"
    redacted = bug_report._replace_path(text, "/data/scenes", "<dataset>")
    redacted = bug_report._replace_path(redacted, "/home/alice", "~")
    assert redacted == "<dataset>/x/file <dataset>/xy/file ~/file /home/alicefoo/file"


def test_replace_path_redacts_mid_line_boundaries(monkeypatch):
    monkeypatch.setattr(bug_report, "lf", make_fake_lf())
    path = "/data/scenes/x"
    text = f'{path} now\n{path}"{path}:{path} '

    assert bug_report._replace_path(text, path, "<dataset>") == (
        '<dataset> now\n<dataset>"<dataset>:<dataset> '
    )


def test_redact_replaces_relative_dataset_path_mid_line(monkeypatch):
    monkeypatch.setattr(
        bug_report,
        "lf",
        make_fake_lf(dataset={"data_path": "data/bicycle"}),
    )
    monkeypatch.setattr(
        bug_report,
        "get_portal_account_service",
        lambda: SimpleNamespace(_redaction_tokens=lambda: ()),
    )

    assert bug_report.redact("Loading data/bicycle now") == "Loading <dataset> now"


def test_dataset_fingerprint_is_stable_for_trailing_slash_and_symlink(
    tmp_path,
    monkeypatch,
    fake_runtime,
):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    symlink = tmp_path / "dataset-link"
    symlink.symlink_to(dataset, target_is_directory=True)
    fake_runtime.training_running.value = True
    dataset_config = FakeConfig({"data_path": f"{dataset}/", "resize_factor": 1, "max_width": 0})
    fake_lf = make_fake_lf(optimization={"strategy": "mcmc", "sh_degree": 3})
    fake_lf.dataset_params = lambda: dataset_config
    monkeypatch.setattr(bug_report, "lf", fake_lf)

    direct = bug_report.collect_diagnostics()["dataset_fingerprint"]
    dataset_config._values["data_path"] = str(symlink)
    linked = bug_report.collect_diagnostics()["dataset_fingerprint"]

    assert direct == linked
    assert direct == __import__("hashlib").sha256(
        __import__("os").fsencode(__import__("os").path.realpath(dataset))
    ).hexdigest()[:12]


def test_redact_scrubs_paths_identity_email_host_and_tokens(tmp_path, monkeypatch):
    home = tmp_path / "home" / "alice"
    dataset = tmp_path / "mounted" / "dataset"
    output = tmp_path / "mounted" / "output"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(bug_report.getpass, "getuser", lambda: "alice")
    monkeypatch.setattr(bug_report.socket, "gethostname", lambda: "render-host")
    monkeypatch.setattr(
        bug_report,
        "lf",
        make_fake_lf(dataset={"data_path": str(dataset), "output_path": str(output)}),
    )
    monkeypatch.setattr(
        bug_report,
        "get_portal_account_service",
        lambda: SimpleNamespace(
            _redaction_tokens=lambda: ("access-secret", "refresh-secret")
        ),
    )
    source = (
        f"{dataset}/images {output}/model {home}/file alice malice "
        "alice@example.com render-host C:\\Users\\Alice\\scene "
        "\\\\nas\\share\\scene access-secret refresh-secret "
        "Authorization: Bearer arbitrary-token"
    )

    redacted = bug_report.redact(source)

    for secret in (
        str(dataset),
        str(output),
        str(home),
        "alice@example.com",
        "render-host",
        "C:\\Users\\Alice",
        "\\\\nas\\share",
        "access-secret",
        "refresh-secret",
        "arbitrary-token",
    ):
        assert secret not in redacted
    assert "<dataset>" in redacted
    assert "<output>" in redacted
    assert "<email>" in redacted
    assert "<host>" in redacted
    assert "<network>" in redacted
    assert "<user>" in redacted
    assert "malice" in redacted
    assert "Authorization: Bearer <token>" in redacted


def test_log_tail_is_utf8_byte_capped_at_a_line_boundary(monkeypatch, fake_runtime):
    fake_lf = make_fake_lf()
    monkeypatch.setattr(bug_report, "lf", fake_lf)
    monkeypatch.setattr(
        bug_report,
        "get_portal_account_service",
        lambda: SimpleNamespace(_redaction_tokens=lambda: ()),
    )
    large_log = "discarded-prefix\n" + ("é-line\n" * 180_000) + "tail-marker\n"

    payload = bug_report.build_payload(
        valid_form(),
        consent_to_logs=True,
        current_log_text=large_log,
    )
    text = payload["log"]["text"]

    assert len(text.encode("utf-8")) <= bug_report.LOG_MAX_BYTES
    assert text.endswith("tail-marker\n")
    assert not text.startswith("line")


def test_log_tail_preserves_a_single_line_larger_than_cap():
    result = bug_report.truncate_utf8_tail("prefix-" + ("é" * 100), 10)
    assert result


def test_logs_are_omitted_without_consent(monkeypatch, fake_runtime):
    monkeypatch.setattr(bug_report, "lf", make_fake_lf())
    payload = bug_report.build_payload(
        valid_form(),
        consent_to_logs=False,
        current_log_text="current log",
        include_previous_log=True,
        previous_log_text="previous log",
    )
    assert "log" not in payload
    assert "previous_log" not in payload


def test_category_detail_is_other_only_redacted_trimmed_and_capped(
    monkeypatch, fake_runtime
):
    dataset = "/private/data/bicycle"
    monkeypatch.setattr(
        bug_report,
        "lf",
        make_fake_lf(dataset={"data_path": dataset}),
    )
    monkeypatch.setattr(
        bug_report,
        "get_portal_account_service",
        lambda: SimpleNamespace(_redaction_tokens=lambda: ()),
    )

    form = valid_form()
    form["category"] = "other"
    form["category_detail"] = f"  {dataset} " + ("x" * 200)
    payload = bug_report.build_payload(form, consent_to_logs=False)

    detail = payload["category_detail"]
    assert detail.startswith("<dataset>")
    assert dataset not in detail
    assert len(detail) == bug_report.CATEGORY_DETAIL_MAX_LENGTH

    form["category_detail"] = "   "
    assert "category_detail" not in bug_report.build_payload(
        form, consent_to_logs=False
    )

    for category in bug_report.CATEGORIES:
        if category == "other":
            continue
        form["category"] = category
        form["category_detail"] = "should be omitted"
        assert "category_detail" not in bug_report.build_payload(
            form, consent_to_logs=False
        )

    form["category"] = "install"
    with pytest.raises(ValueError, match="Unsupported bug-report category"):
        bug_report.build_payload(form, consent_to_logs=False)


def test_published_bug_report_state_does_not_alias_defaults(fake_runtime):
    state = bug_report._publish_state({"detail": {"title": ["bad"]}})
    state["detail"]["actual"] = ["bad"]
    state["completeness_problems"].append("missing")

    fresh = bug_report._publish_state({})
    assert fresh["detail"] == {}
    assert fresh["completeness_problems"] == []


def test_submit_uses_authenticated_service_and_refreshes_exactly_once(
    tmp_path,
    monkeypatch,
    fake_runtime,
):
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path)
    stub = StubUrlopen(
        (401, {"error": "invalid_token"}),
        (200, rotated_tokens()),
        (201, success_response()),
    )
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = portal_account.PortalAccountService(credentials_path=credentials_path)
    monkeypatch.setattr(bug_report, "lf", make_fake_lf())
    monkeypatch.setattr(bug_report, "get_portal_account_service", lambda: service)

    payload = bug_report.build_payload(valid_form(), consent_to_logs=False)
    state = bug_report.submit_report(payload, service=service)

    assert state["success"] is True
    paths = [urllib.parse.urlsplit(request.full_url).path for request in stub.requests]
    assert paths == [
        bug_report.BUG_REPORT_PATH,
        portal_account.REFRESH_PATH,
        bug_report.BUG_REPORT_PATH,
    ]
    assert sum(path == portal_account.REFRESH_PATH for path in paths) == 1
    assert stub.requests[0].get_header("Content-type") == "application/json"
    assert stub.requests[0].get_header("Authorization") == "Bearer access-old"
    assert stub.requests[2].get_header("Authorization") == "Bearer access-rotated"
    for request in (stub.requests[0], stub.requests[2]):
        wire = json.loads(request.data.decode("utf-8"))
        assert {
            "category",
            "title",
            "expected",
            "actual",
            "app_version",
            "client_name",
            "diagnostics",
        } <= wire.keys()
        assert isinstance(wire["steps"], list)
        serialized = "\n".join(recursive_strings(wire))
        assert "data_path" not in serialized
        assert "output_path" not in serialized
        assert "log" not in wire


def test_definitive_refresh_failure_signs_out(
    tmp_path,
    monkeypatch,
    fake_runtime,
):
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path)
    stub = StubUrlopen(
        (401, {"error": "invalid_token"}),
        (401, {"error": "invalid_token"}),
    )
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = portal_account.PortalAccountService(credentials_path=credentials_path)
    monkeypatch.setattr(bug_report, "lf", make_fake_lf())
    monkeypatch.setattr(bug_report, "get_portal_account_service", lambda: service)

    state = bug_report.submit_report(
        bug_report.build_payload(valid_form(), consent_to_logs=False), service=service
    )

    assert state["error"] == "invalid_token"
    assert service.snapshot().signed_in is False
    assert not credentials_path.exists()
    assert [urllib.parse.urlsplit(request.full_url).path for request in stub.requests] == [
        bug_report.BUG_REPORT_PATH,
        portal_account.REFRESH_PATH,
    ]


def test_membership_required_switches_state_without_signing_out(
    tmp_path,
    monkeypatch,
    fake_runtime,
):
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path)
    stub = StubUrlopen((403, {"error": "membership_required"}))
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = portal_account.PortalAccountService(credentials_path=credentials_path)
    monkeypatch.setattr(bug_report, "lf", make_fake_lf())
    monkeypatch.setattr(bug_report, "get_portal_account_service", lambda: service)

    state = bug_report.submit_report(
        bug_report.build_payload(valid_form(), consent_to_logs=False), service=service
    )

    assert state["error"] == "membership_required"
    assert service.snapshot().signed_in is True
    assert service.snapshot().membership_required is True
    assert credentials_path.exists()
    assert len(stub.requests) == 1


@pytest.mark.parametrize(
    ("exception", "error", "retry_after"),
    [
        (portal_account.PortalHTTPError(404, ""), "portal_outdated", None),
        (portal_account.PortalHTTPError(429, "rate_limited", 17), "rate_limited", 17),
        (
            portal_account.PortalHTTPError(
                400,
                "report_invalid",
                detail={"title": ["Too short"]},
            ),
            "report_invalid",
            None,
        ),
    ],
)
def test_http_error_mapping(exception, error, retry_after, fake_runtime):
    service = SimpleNamespace(
        request_json_authenticated=lambda *_args, **_kwargs: (_ for _ in ()).throw(exception)
    )

    state = bug_report.submit_report(
        bug_report.build_payload(valid_form(), consent_to_logs=False), service=service
    )

    assert state["error"] == error
    assert state["retry_after"] == retry_after
    if error == "report_invalid":
        assert state["detail"] == {"title": ["Too short"]}


def test_non_json_http_error_maps_to_generic(monkeypatch, fake_runtime, tmp_path):
    stub = StubUrlopen((413, b"<html>request too large</html>"))
    monkeypatch.setattr(portal_account, "urlopen", stub)
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path)
    service = portal_account.PortalAccountService(
        credentials_path=credentials_path
    )
    monkeypatch.setattr(bug_report, "lf", make_fake_lf())
    monkeypatch.setattr(bug_report, "get_portal_account_service", lambda: service)

    state = bug_report.submit_report(
        bug_report.build_payload(valid_form(), consent_to_logs=False), service=service
    )

    assert state["error"] == "generic"


def test_report_invalid_detail_survives_real_http_mapping(monkeypatch, fake_runtime, tmp_path):
    stub = StubUrlopen(
        (400, {"error": "report_invalid", "detail": {"title": ["Too short"]}})
    )
    monkeypatch.setattr(portal_account, "urlopen", stub)
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path)
    service = portal_account.PortalAccountService(
        credentials_path=credentials_path
    )
    monkeypatch.setattr(bug_report, "lf", make_fake_lf())
    monkeypatch.setattr(bug_report, "get_portal_account_service", lambda: service)

    state = bug_report.submit_report(
        bug_report.build_payload(valid_form(), consent_to_logs=False), service=service
    )

    assert state["error"] == "report_invalid"
    assert state["detail"] == {"title": ["Too short"]}


def test_previous_log_is_offered_only_for_unclean_breadcrumb(tmp_path, monkeypatch):
    snapshot = tmp_path / "previous_session.log"
    snapshot.write_text("previous marker\n", encoding="utf-8")
    state = {"clean_exit": True, "log_path": str(snapshot)}
    fake_lf = make_fake_lf()
    fake_lf.log.previous_session = lambda: state
    monkeypatch.setattr(bug_report, "lf", fake_lf)

    assert bug_report.previous_session_log_available() is False
    state["clean_exit"] = False
    assert bug_report.previous_session_log_available() is True
    assert bug_report.previous_session_log() == "previous marker\n"
    state["log_path"] = str(tmp_path / "missing.log")
    assert bug_report.previous_session_log_available() is False


def test_no_token_material_reaches_attached_logs_or_logging(
    monkeypatch,
    fake_runtime,
    caplog,
):
    tokens = ("access-material", "refresh-material")
    monkeypatch.setattr(bug_report, "lf", make_fake_lf())
    service = SimpleNamespace(
        base_url=portal_account.DEFAULT_PORTAL_URL,
        _redaction_tokens=lambda: tokens,
        request_json_authenticated=lambda _method, _path, payload: success_response(),
    )
    monkeypatch.setattr(bug_report, "get_portal_account_service", lambda: service)
    payload = bug_report.build_payload(
        valid_form(),
        consent_to_logs=True,
        current_log_text=f"{tokens[0]}\nAuthorization: Bearer {tokens[1]}\n",
    )

    with caplog.at_level(logging.DEBUG, logger=bug_report.__name__):
        state = bug_report.submit_report(payload, service=service)

    assert state["success"] is True
    output = caplog.text + json.dumps(payload)
    assert tokens[0] not in output
    assert tokens[1] not in output


def test_portal_url_validation_rejects_suffix_host_and_shell_is_not_invoked(
    monkeypatch,
):
    opened = []
    fake_lf = make_fake_lf()
    fake_lf.ui.open_url = opened.append
    service = SimpleNamespace(base_url="https://portal.example.com")
    monkeypatch.setattr(bug_report, "lf", fake_lf)
    monkeypatch.setattr(bug_report, "get_portal_account_service", lambda: service)

    assert (
        bug_report.is_safe_portal_url(
            "https://portal.example.com.evil.com/bugs/id/",
            service.base_url,
        )
        is False
    )
    assert bug_report.open_portal_url("https://portal.example.com.evil.com/bugs/id/") is False
    shell_metacharacter_url = "https://portal.example.com/bugs/id/;touch%20/tmp/pwned"
    assert bug_report.open_portal_url(shell_metacharacter_url) is True
    assert opened == [shell_metacharacter_url]


def test_client_categories_match_in_app_panel_and_markup():
    panel_path = (
        Path(__file__).parents[2]
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "bug_report_panel.rml"
    )
    markup = panel_path.read_text(encoding="utf-8")
    assert bug_report.CATEGORIES == (
        "crash",
        "training",
        "ui",
        "performance",
        "export",
        "other",
    )
    radio_values = re.findall(
        r'<input type="radio"[^>]*value="([^"]+)"', markup
    )
    assert radio_values == list(bug_report.CATEGORIES)
    assert "data-event-change" not in markup
    assert '@tr:bugreport.category.install' not in markup
    assert '@tr:bugreport.category.download' not in markup
    assert '@tr:bugreport.category.portal' not in markup


def test_submit_async_uses_named_daemon_thread(monkeypatch, fake_runtime):
    created = {}

    class FakeThread:
        def __init__(self, *, target, daemon, name):
            created.update(target=target, daemon=daemon, name=name)

        def is_alive(self):
            return False

        def start(self):
            created["started"] = True

    monkeypatch.setattr(bug_report.threading, "Thread", FakeThread)
    monkeypatch.setattr(bug_report, "_submit_thread", None)

    assert bug_report.submit_async(valid_form()) is True
    assert created["daemon"] is True
    assert created["name"] == "lfs-portal-bugreport"
    assert created["started"] is True


def test_submit_async_publishes_submitting_and_clears_after_worker(monkeypatch, fake_runtime):
    release = __import__("threading").Event()

    def worker(_payload):
        release.wait(timeout=2)
        return bug_report._publish_state({"success": True})

    monkeypatch.setattr(bug_report, "submit_report", worker)
    monkeypatch.setattr(bug_report, "_submit_thread", None)
    payload = bug_report.build_payload(valid_form(), consent_to_logs=False)

    assert bug_report.submit_async(payload) is True
    assert fake_runtime.bug_report_state.value["submitting"] is True
    assert bug_report.submit_async(payload) is False

    release.set()
    thread = bug_report._submit_thread
    assert thread is not None
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert fake_runtime.bug_report_state.value["submitting"] is False


def test_bug_report_panel_resets_when_a_new_open_cycle_is_requested(monkeypatch):
    monkeypatch.setattr(
        bug_report_panel,
        "get_portal_account_service",
        lambda: SimpleNamespace(),
    )
    panel = bug_report_panel.BugReportPanel()
    panel._open_cycle_generation = bug_report_panel._open_cycle_generation
    prepared = []
    monkeypatch.setattr(panel, "_prepare_form", lambda: prepared.append(True))

    bug_report_panel.request_bug_report_open()

    assert panel.on_update(None) is True
    assert prepared == [True]
    assert panel.on_update(None) is False


def test_bug_report_panel_clears_category_detail_when_leaving_other(monkeypatch):
    panel = bug_report_panel.BugReportPanel.__new__(bug_report_panel.BugReportPanel)
    panel._form = {"category": "other", "category_detail": "custom problem"}
    monkeypatch.setattr(panel, "_dirty_form", lambda: None)

    panel._on_set_category(None, None, ["crash"])

    assert panel._form == {"category": "crash", "category_detail": ""}


def test_bug_report_panel_category_selection_survives_repeated_invalidations(monkeypatch):
    panel = bug_report_panel.BugReportPanel.__new__(bug_report_panel.BugReportPanel)
    panel._form = {"category": "crash", "category_detail": ""}
    monkeypatch.setattr(panel, "_dirty_form", lambda: None)

    panel._on_set_category(None, None, ["other"])
    panel._form["category_detail"] = "custom problem"
    for _ in range(3):
        panel._dirty_form()

    assert panel._form == {"category": "other", "category_detail": "custom problem"}

    panel._on_set_category(None, None, ["training"])
    assert panel._form == {"category": "training", "category_detail": ""}


def test_bug_report_panel_caps_category_detail_input(monkeypatch):
    panel = bug_report_panel.BugReportPanel.__new__(bug_report_panel.BugReportPanel)
    panel._form = {}
    monkeypatch.setattr(panel, "_dirty_form", lambda: None)

    panel._set_form_field("category_detail", "x" * 200)

    assert panel._form["category_detail"] == "x" * bug_report.CATEGORY_DETAIL_MAX_LENGTH


def test_validation_hint_is_hidden_while_submitting(monkeypatch):
    panel = bug_report_panel.BugReportPanel.__new__(bug_report_panel.BugReportPanel)
    monkeypatch.setattr(panel, "_show_form", lambda: True)
    monkeypatch.setattr(panel, "_can_submit", lambda: False)
    monkeypatch.setattr(panel, "_report_state", lambda: {"submitting": True})

    assert panel._validation_hint() == ""
