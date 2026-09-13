# SPDX-License-Identifier: GPL-3.0-or-later
"""Harness contract tests. No Studio, Django, display, build, or GPU required."""
import importlib.util
import json
from pathlib import Path
import sys
import threading

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def module(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


module("validate_gallery_sync")
common = module("gallery_sync_e2e_common")
stress = module("stress_gallery_sync")


def args(*extra):
    return stress.parse_args(["--build-dir", "/tmp/build", "--portal-source", "/tmp/portal", *extra])


def test_all_fourteen_scenarios_are_callable():
    parsed = args()
    assert len(parsed.scenarios) == 14
    assert all(callable(getattr(stress.StressRun, name)) for name in parsed.scenarios)


@pytest.mark.parametrize("extra", [
    ["--cycles", "0"], ["--cycles", "-1"], ["--scenarios", "missing"],
    ["--scenarios", ""], ["--scenarios", "listing_scale,listing_scale"],
    ["--timeout", "nan"], ["--timeout", "inf"], ["--timeout", "0"],
    ["--watchdog-seconds", "-4"], ["--display", "localhost:0"],
    ["--large-fixture", "/tmp/does-not-exist.licht"],
])
def test_rejects_invalid_cli(extra):
    with pytest.raises(SystemExit):
        args(*extra)


def test_cli_selection_and_seed():
    parsed = args("--scenarios", "listing_scale, thumbnail_cache", "--cycles", "7", "--seed", "-8", "--keep")
    assert parsed.scenarios == ["listing_scale", "thumbnail_cache"]
    assert parsed.cycles == 7 and parsed.seed == -8 and parsed.keep


def test_seed_independent_of_selected_scenarios():
    assert common.scenario_seed(7, "one") == common.scenario_seed(7, "one")
    assert common.scenario_seed(7, "one") != common.scenario_seed(7, "two")
    assert common.scenario_seed(7, "one") != common.scenario_seed(8, "one")


def test_revision_hashes_are_not_compared_lexically():
    common.assert_revision_progress([
        dict(sceneId="s", metadataRevision="z", exchangedAt=1),
        dict(sceneId="s", metadataRevision="a", exchangedAt=1),
    ])


@pytest.mark.parametrize("second", [
    dict(sceneId="s", metadataRevision="z", exchangedAt=2),
    dict(sceneId="s", metadataRevision="a", exchangedAt=0),
    dict(sceneId="other", metadataRevision="a", exchangedAt=2),
])
def test_revision_regressions_fail(second):
    with pytest.raises(AssertionError):
        common.assert_revision_progress([dict(sceneId="s", metadataRevision="z", exchangedAt=1), second])


def test_single_exchange_is_insufficient():
    with pytest.raises(AssertionError):
        common.assert_revision_progress([])


def test_retained_parts_require_reuse_and_reduced_network_bytes():
    old = [dict(number=1, etag="same", size=8)]
    new = old + [dict(number=2, etag="new", size=8)]
    common.assert_retained_parts(old, new, 16, 8)
    for before, after, sent in (([], new, 8), (old, [], 8), (old, new, 16), (old, new, 0),
                                (old, [dict(number=1, etag="changed", size=8)], 8)):
        with pytest.raises(AssertionError):
            common.assert_retained_parts(before, after, 16, sent)


def retry_events(elapsed=2):
    return [dict(method="POST", path="/upload", injected=429, retry_after=2, started=0, finished=.1),
            dict(method="GET", path="/me", injected=0, started=.2, finished=.3),
            dict(method="POST", path="/upload", injected=0, started=.1+elapsed, finished=3)]


def test_retry_matches_endpoint_and_method():
    checked = common.retry_evidence(retry_events())
    assert checked[0]["elapsed"] == 2


def test_retry_uses_response_time_not_request_start():
    with pytest.raises(AssertionError, match="Retry-After"):
        common.retry_evidence(retry_events(.5))


@pytest.mark.parametrize("events", [[], retry_events()[:1]])
def test_missing_retry_evidence_fails(events):
    with pytest.raises(AssertionError):
        common.retry_evidence(events)


def test_job_evidence_omits_checkpoint_credentials_and_signed_urls():
    data = common.safe_job(dict(id="a", status="paused", checkpoint={"uploadId": "u", "url": "secret"},
                                metadata={"token": "secret"}, stagedImport={"state": "failed", "secret": "no"}))
    assert data["uploadId"] == "u"
    assert "secret" not in json.dumps(data)


def proxy_policy(seed):
    proxy = object.__new__(common.FaultProxy)
    proxy.faults, proxy.seed, proxy.count, proxy.lock = True, seed, 0, threading.Lock()
    return proxy


def test_fault_schedule_ten_percent_with_both_statuses():
    proxy = proxy_policy(3)
    statuses = [proxy.inject("GET", "/api/gallery/v1/splats") for _ in range(100)]
    assert statuses.count(429) == statuses.count(503) == 5
    assert all(sum(bool(s) for s in statuses[i:i+10]) == 1 for i in range(0, 100, 10))
    other = proxy_policy(3)
    assert statuses == [other.inject("GET", "/api/gallery/v1/splats") for _ in range(100)]


def test_fault_schedule_does_not_touch_non_gallery_traffic():
    proxy = proxy_policy(3)
    assert proxy.inject("GET", "/health") == 0 and proxy.count == 0
    proxy.faults = False
    assert proxy.inject("GET", "/api/gallery/v1/me") == 0 and proxy.count == 0


def test_report_escapes_failures_and_keeps_evidence_links():
    rows = [dict(name="one", status="FAIL", seconds=.2, error="bad|row\nnext", json="a.json", screenshot="a.png"),
            dict(name="two", status="PASS", seconds=2, json="b.json")]
    report = stress.render_report(rows, "python script.py --seed 3")
    assert "bad&#124;row<br>next" in report
    assert "[JSON](<b.json>)" in report and "[failure screenshot](<a.png>)" in report
    assert "--seed 3" in report and "FAIL" in report


def test_proxy_rejects_nonlocal_upstream(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        common.FaultProxy("https://example.com", 1, tmp_path / "proxy.jsonl")


def test_confirmation_uses_actual_enabled_label():
    modal = {"buttons": [{"label": "Cancel", "enabled": True},
                         {"label": "Pull from portal", "enabled": True},
                         {"label": "Unavailable", "enabled": False}]}
    assert common.confirmation_label(modal) == "Pull from portal"


def test_shell_json_handles_large_result_and_diagnostics():
    value = ['scene ' + str(i) for i in range(1000)]
    assert common.shell_json('Django diagnostics\n' + common.SHELL_MARKER + json.dumps(value)) == value
    with pytest.raises(ValueError):
        common.shell_json(common.SHELL_MARKER + '{}\n' + common.SHELL_MARKER + '{}')


def test_remote_snapshot_is_evaluated_once_and_paged_below_editor_limit():
    expected = ['scene ' + str(i) for i in range(300)]
    calls = []
    class MCP:
        def __init__(self): self.scope = {'json': json, 'expected': expected}
        def rpc(self, code):
            calls.append(code)
            exec(code, self.scope)
        def value(self, expression):
            result = eval(expression, self.scope)
            # Verify against H's real framed result emitter as well.
            exec(module('validate_gallery_sync').result_code(repr(result)), {'json': json})
            return result
    assert common.remote_json(MCP(), 'expected') == expected
    assert len(calls) == 1


@pytest.mark.parametrize("labels", [[], ["Cancel"], ["Cancel", "Mine", "Portal"]])
def test_confirmation_never_guesses_between_conflict_choices(labels):
    with pytest.raises(AssertionError):
        common.confirmation_label({"buttons": [{"label": label} for label in labels]})


def test_forward_stream_and_inflated_download_without_socket(tmp_path):
    """Exercise real proxy forwarding with in-memory transport, including redaction."""
    import io
    from unittest.mock import patch
    class Response:
        status = 200
        def __init__(self):
            self.body = io.BytesIO(b'{"scene":{"contentLength":7},"url":"local"}')
        def read(self, size=-1):
            return self.body.read(size)
        def getheaders(self):
            return [("Content-Type", "application/json"), ("Content-Length", "99")]
        def getheader(self, key):
            return "99" if key == "Content-Length" else None
    class Connection:
        def __init__(self, *a, **kw):
            self.sent = b""
        def putrequest(self, *a, **kw): pass
        def putheader(self, *a): pass
        def endheaders(self): pass
        def send(self, data): self.sent += data
        def getresponse(self): return Response()
        def close(self): pass
    class Handler:
        command = "GET"
        path = "/api/gallery/v1/splats/id/download?token=SECRET"
        headers = {"Authorization": "Bearer SECRET"}
        rfile = io.BytesIO()
        def __init__(self): self.wfile, self.out_headers = io.BytesIO(), {}
        def send_response(self, status): self.status = status
        def send_header(self, key, value): self.out_headers[key] = value
        def end_headers(self): pass
    proxy = proxy_policy(1)
    proxy.faults, proxy.inflate_download, proxy.list_delay = False, 1024, 0
    proxy.upload_bps = proxy.download_bps = 0
    proxy.upstream = common.urllib.parse.urlsplit("http://127.0.0.1:12")
    proxy.origin, proxy.events, proxy.log = "http://127.0.0.1:13", [], tmp_path / "proxy.jsonl"
    handler = Handler()
    with patch.object(common.http.client, "HTTPConnection", Connection):
        proxy.forward(handler)
    assert json.loads(handler.wfile.getvalue())["scene"]["contentLength"] == 1031
    assert int(handler.out_headers["Content-Length"]) == len(handler.wfile.getvalue())
    assert "SECRET" not in proxy.log.read_text()
    assert proxy.events[0]["status"] == 200 and "finished" in proxy.events[0]


def test_main_continues_after_failure_and_cleans_each_scenario(tmp_path, monkeypatch):
    cleaned = []
    class FakeRun:
        app = None
        def __init__(self, args, command, name, directory): self.name = name
        def setup(self): pass
        def listing_scale(self): raise AssertionError("intentional invariant failure")
        def thumbnail_cache(self): pass
        def evidence(self): return {"observed": True}
        def cleanup(self): cleaned.append(self.name)
    monkeypatch.setattr(stress, "StressRun", FakeRun)
    report = tmp_path / "report.md"
    code = stress.main(["--build-dir", "/tmp/build", "--portal-source", "/tmp/portal", "--report", str(report),
                        "--scenarios", "listing_scale,thumbnail_cache"])
    assert code == 1 and cleaned == ["listing_scale", "thumbnail_cache"]
    assert "FAIL" in report.read_text() and "PASS" in report.read_text()
    first = json.loads((tmp_path / "report-artifacts/listing_scale.json").read_text())
    assert first["evidence"]["observed"] and "intentional" in first["error"]


def test_evidence_failure_still_cleans_and_reports_failure(tmp_path, monkeypatch):
    cleaned = []
    class FakeRun:
        app = None
        def __init__(self, *args): pass
        def setup(self): pass
        def listing_scale(self): pass
        def evidence(self): raise OSError("log disappeared")
        def cleanup(self): cleaned.append(True)
    monkeypatch.setattr(stress, "StressRun", FakeRun)
    report = tmp_path / "evidence.md"
    code = stress.main(["--build-dir", "/tmp/build", "--portal-source", "/tmp/portal", "--report", str(report),
                        "--scenarios", "listing_scale"])
    assert code == 1 and cleaned == [True]
    row = json.loads((tmp_path / 'evidence-artifacts/listing_scale.json').read_text())
    assert row['evidence_error'] == 'log disappeared' and row['status'] == 'FAIL'
