# SPDX-License-Identifier: GPL-3.0-or-later
"""Harness contract tests. No LichtFeld Studio, Django, display, build, or GPU required."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest
from test_asset_manager_panel import panel_module

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
    first = json.loads(next((tmp_path / 'report-artifacts').glob('*/listing_scale.json')).read_text())
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
    row = json.loads(next((tmp_path / 'evidence-artifacts').glob('*/listing_scale.json')).read_text())
    assert row['evidence_error'] == 'log disappeared' and row['status'] == 'FAIL'

@pytest.fixture
def waiting_run(monkeypatch):
    run = object.__new__(stress.StressRun)
    run.args = args('--timeout', '600')
    run.app = run.worker = None
    run.name = 'test'
    run.now = 0.
    run.started = 0.
    run.observations, run.modal_presses, run.last_jobs = [], [], []
    def advance(seconds):
        run.now += seconds
    monkeypatch.setattr(stress.time, 'monotonic', lambda: run.now)
    monkeypatch.setattr(stress.time, 'sleep', advance)
    return run

@pytest.mark.parametrize('timeout,limit', [(None, 120), (600, 120), (.4, .4)])
def test_wait_caps_deadline_and_reports_observed_facts(waiting_run, timeout, limit):
    run = waiting_run
    run.value = lambda _: {'state': 'remote', 'poll_time': run.now}
    with pytest.raises(TimeoutError) as caught:
        run.wait_value('facts', 'expected equal', timeout, accept=lambda v: v['state'] == 'equal')
    assert run.now == pytest.approx(limit)
    assert 'expected equal' in str(caught.value) and "'state': 'remote'" in str(caught.value)
    assert 'poll_time' in str(caught.value)
    assert run._wait_deadline is None

def test_wait_respects_shorter_cli_limit_and_false_is_success(waiting_run):
    run = waiting_run
    run.args.timeout = .2
    run.value = lambda _: False
    assert run.wait_value('new.busy', 'idle', accept=lambda busy: not busy) is False
    assert run.now == 0
    with pytest.raises(TimeoutError, match='limit 0.2s'):
        run.wait_value('new.busy', 'active', 100)
    assert run.now == pytest.approx(.2)

def test_wait_transport_timeout_keeps_last_observation(waiting_run):
    run = waiting_run
    def value(_):
        if run.now:
            raise TimeoutError('MCP timed out')
        return {'status': 'processing'}
    run.value = value
    with pytest.raises(TimeoutError, match="transfer.*last=.*processing.*MCP timed out"):
        run.wait_value('job', 'transfer', accept=lambda v: v['status'] == 'completed')

def test_wait_job_reports_progress_without_checkpoint_secrets(waiting_run):
    run = waiting_run
    run.args.timeout = .2
    run.jobs_now = lambda: [dict(id='j', status='running', completed=17, total=100,
                               checkpoint={'uploadId': 'u', 'url': 'SECRET'})]
    with pytest.raises(TimeoutError) as caught:
        run.wait_job('j', lambda j: j['status'] == 'completed', 'job terminal')
    assert 'running' in str(caught.value) and '17' in str(caught.value)
    assert 'SECRET' not in str(caught.value)

def panel_observation(**overrides):
    return dict(checked='Checked just now', checked_label='Checked just now',
                counts=dict(count=1, ready=0, linked=1, missing=0), selection_count=1,
                selected=dict(id='a'), available=True, catalog=['a'], cards=[], **overrides)

def test_panel_accepts_unchanged_relative_label_and_current_counts():
    assert common.panel_refresh_ready(panel_observation(), 'Checked just now',
        dict(links={'a': {'sceneId': 's'}}, scenes=[]))

def test_panel_rejects_stale_counts_and_listing_then_accepts_updated_dom():
    observed = panel_observation()
    fresh = dict(links={}, scenes=[dict(id='s', status='ready')])
    assert not common.panel_refresh_ready(observed, 'Checked just now', fresh)
    observed['cards'] = ['remote:s']
    assert not common.panel_refresh_ready(observed, 'Checked just now', fresh)
    observed['checked'] = observed['checked_label'] = 'Checked a moment ago'
    assert common.panel_refresh_ready(observed, 'Checked just now', fresh)
    observed['checked'] = 'Offline'
    assert not common.panel_refresh_ready(observed, 'Checked just now', fresh)

def test_panel_excludes_linked_and_deleted_remote_cards():
    observed = panel_observation()
    fresh = dict(links={'a': {'sceneId': 's'}}, scenes=[dict(id='s', status='ready'),
                 dict(id='deleted', status='deleted'), dict(id='other', status='ready')])
    assert not common.panel_refresh_ready(observed, observed['checked'], fresh)
    observed['cards'] = ['remote:other']
    assert common.panel_refresh_ready(observed, observed['checked'], fresh)

def test_refresh_waits_for_fresh_success_idle_and_coalesced_panel(waiting_run):
    """Execute the actual remote expressions without a panel _gallery_state."""
    run = waiting_run
    samples = iter([(99., False, True), (101., True, True), (102., False, False), (105., False, True)])
    class Service:
        busy = False
        polls = 0
        def snapshot(self):
            self.polls += 1
            checked, self.busy, ok = next(samples)
            return dict(checkedAt=checked, refresh_ok=ok, links={'a': {'sceneId': 's'}}, scenes=[])
    service = Service()
    class Panel:
        polls = 0
        def _controller(self): return self
        def refresh(self):
            assert namespace['_stress_refresh_t0'] == 100.
        def _gallery_counts(self):
            self.polls += 1
            linked = int(self.polls >= 3)
            return dict(count=1, ready=1-linked, linked=linked, missing=0)
        def _gallery_checked_label(self): return 'Checked just now'
        def get_selected_count(self): return 1
        def _get_selected_asset(self): return dict(id='a')
        def _project_available(self, asset): return True
        def _asset_index_assets(self): return {'a': dict(id='a')}
        def _gallery_remote_assets(self): return {}
    panel = Panel()
    element = SimpleNamespace(get_inner_rml=lambda: 'Checked just now')
    doc = SimpleNamespace(query_selector=lambda selector: element)
    namespace = dict(new=service, p=panel, time=SimpleNamespace(time=lambda: 100.),
                     lf=SimpleNamespace(ui=SimpleNamespace(rml=SimpleNamespace(get_document=lambda _: doc))))
    run.value = lambda expression: eval(expression, namespace)
    run.rpc = lambda code: exec(code, namespace)
    run.refresh()
    assert service.polls == 4 and panel.polls == 3
    assert run.now >= .6

def test_bounded_remote_snapshot_pages_share_remaining_transport_budget(waiting_run):
    import io
    from contextlib import redirect_stdout
    run = waiting_run
    expected = ['scene ' + str(i) for i in range(300)]
    class MCP:
        def __init__(self):
            self.scope = dict(json=json, expected=expected)
            self.timeouts = []
            self.diagnostics = lambda _: None
        def call(self, method, params, timeout):
            assert method == 'tools/call'
            assert params['arguments']['timeout_ms'] <= timeout * 1000
            self.timeouts.append(timeout)
            run.now += .1
            output = io.StringIO()
            with redirect_stdout(output):
                exec(params['arguments']['code'], self.scope)
            return {'structuredContent': {'success': True, 'output': {'text': output.getvalue()}}}
    mcp = MCP()
    assert common.remote_json(mcp, 'expected', deadline=5) == expected
    assert len(mcp.timeouts) > 3
    assert all(b < a for a, b in zip(mcp.timeouts, mcp.timeouts[1:]))
    run.now = 0
    with pytest.raises(TimeoutError, match='snapshot read exceeded wait deadline'):
        common.remote_json(mcp, 'expected', deadline=.15)
    assert run.now == pytest.approx(.2)

@pytest.mark.parametrize('flag', ['running', 'timed_out'])
def test_remote_editor_timeout_has_wait_context_and_diagnostics(waiting_run, flag):
    run = waiting_run
    diagnostics = []
    run.mcp = SimpleNamespace(diagnostics=diagnostics.append, call=lambda *a, **kw:
        {'structuredContent': {'success': True, flag: True, 'output': {'text': 'partial diagnostic'}}})
    with pytest.raises(TimeoutError, match='panel refresh.*last=None.*snapshot editor'):
        run.wait_value('True', 'panel refresh')
    assert diagnostics == ['partial diagnostic']

def test_modal_confirmation_shares_wait_deadline(waiting_run):
    run = waiting_run
    calls = []
    run.mcp = SimpleNamespace(call=lambda *a, **kw: calls.append((a, kw)))
    run._wait_deadline = .5
    run.press_modal('Keep portal', {'title': 'Resolve', 'body': 'Changes', 'buttons': [{'label': 'Keep portal'}]})
    assert calls[0][1]['timeout'] == .5
    run.now = .5
    with pytest.raises(TimeoutError, match='modal press'):
        run.press_modal('Keep portal', {'title': 'Resolve', 'body': 'Changes', 'buttons': [{'label': 'Keep portal'}]})
    assert len(calls) == 1

def test_intentional_kill_does_not_hide_a_relaunch_crash(waiting_run):
    run = waiting_run
    killed = SimpleNamespace(pid=12, exited=False)
    killed.poll = lambda: -9 if killed.exited else None
    run.app = killed
    run.stop = lambda proc, crash=False: setattr(proc, 'exited', True)
    run.stop_app(crash=True)
    assert run.until(lambda: True, 'killed socket closed')
    assert run.observations[-1]['value'] == dict(pid=12, signal='SIGKILL')
    run.app = SimpleNamespace(poll=lambda: -11)
    with pytest.raises(AssertionError, match='LichtFeld Studio exited unexpectedly'):
        run.until(lambda: True, 'relaunch')

@pytest.mark.parametrize('id_source', ['checkpoint', 'uploadId'])
def test_kill_scenario_reuses_home_and_reinjects_auth_before_resume(waiting_run, tmp_path, id_source):
    run = waiting_run
    run.home = tmp_path / 'profile'
    run.proxy = SimpleNamespace(snapshot=lambda: [])
    run.app = SimpleNamespace(pid=12, poll=lambda: -9)
    run.stop = lambda *a, **kw: None
    run.large_fixture = lambda: tmp_path / 'large.licht'
    run.queue_large = lambda _: 'job'
    middle = dict(id='job', total=16, completed=8, checkpoint=dict(uploadId='upload'))
    if id_source == 'uploadId':
        middle.update(checkpoint=None, uploadId='upload')
    run.mid_upload = lambda _: middle
    parts = [dict(number=1, etag='retained', size=8)]
    def part_rows(identifier):
        assert identifier == 'upload'
        return parts
    run.part_rows = part_rows
    calls = []
    run.worker_stop = lambda: calls.append('worker stopped')
    run.worker_start = lambda: calls.append('worker started')
    def start():
        assert run.expected_exit is run.app
        assert run.home == tmp_path / 'profile'
        calls.append('relaunch')
        run.app = SimpleNamespace(poll=lambda: None)
    run.start_app = start
    run.panel = lambda: calls.append('panel')
    run.sign_in = lambda: calls.append('sign in')
    def jobs():
        assert 'sign in' in calls
        return [dict(id='job', status='paused', interrupted=True, message='Interrupted')]
    run.jobs_now = jobs
    run.rpc = lambda code: calls.append(code)
    run.wait_job = lambda *a: None
    def finish(identifier):
        calls.append('completed')
        return dict(result=dict(id='scene'))
    run.finish_job = finish
    run.assert_large_scene = lambda job: calls.append(('scene', job['result']['id']))
    # Supply actual resumed bytes after the restart marker.
    def parts_after(_):
        run.proxy.snapshot = lambda: [dict(method='PUT', request_bytes=8, finished=1)]
        return parts
    original_panel = run.panel
    def panel():
        original_panel()
        run.part_rows = parts_after
    run.panel = panel
    run.kill_during_upload()
    assert calls.index('relaunch') < calls.index('panel') < calls.index('sign in')
    resume = next(c for c in calls if isinstance(c, str) and "command('resume'" in c)
    assert calls.index('sign in') < calls.index(resume) < calls.index('completed')
    assert ('scene', 'scene') in calls
    assert any(o['label'] == 'restored interrupted jobs' for o in run.observations)

@pytest.mark.parametrize('supplied', [False, True])
def test_large_fixture_is_saved_in_app_and_closed_before_card_publish(waiting_run, tmp_path, monkeypatch, supplied):
    run = waiting_run
    run.home = run.root = tmp_path
    (tmp_path / 'projects').mkdir()
    run.prefix = 'test-'
    run.rng = SimpleNamespace(uniform=lambda *a: .1)
    # One small batch suffices to exercise fixture generation; native Save As is
    # simulated with a sparse file. This test does not validate native bytes.
    monkeypatch.setattr(stress, 'range', lambda *a: range(1) if len(a) == 3 else range(*a), raising=False)
    source = tmp_path / 'provided.licht'
    if supplied:
        with source.open('wb') as output:
            output.truncate(25 * 1024 * 1024)
        run.args.large_fixture = source
    calls, nodes, assets, jobs = [], [], {}, []
    current = None
    def clear(**kwargs):
        nonlocal current
        current = None
        nodes.clear()
        calls.append('new project')
    def opened(path, **kwargs):
        nonlocal current
        current = path
        nodes.extend([SimpleNamespace(id=i, gaussian_count=10) for i in range(3)])
        calls.append('open supplied')
    def added(name, *arrays):
        assert len(arrays) == 6
        nodes.append(SimpleNamespace(id=0, gaussian_count=5_000_000))
        calls.append('add splat')
    def saved(path, wait):
        nonlocal current
        assert wait and nodes
        current = path
        with Path(path).open('wb') as output:
            output.truncate(200_000_000)
        calls.append('save as')
        return True
    def catalog():
        assets['large-id'] = dict(id='large-id', path=str(tmp_path / 'projects/large.licht'))
        calls.append('catalog')
    scene = SimpleNamespace(add_splat=added, get_nodes=lambda kind: nodes,
        is_node_effectively_visible=lambda node_id: node_id != 2)
    data = SimpleNamespace(**{key + '_raw': object() for key in ('means', 'sh0', 'shN', 'scaling', 'rotation', 'opacity')})
    lf = SimpleNamespace(new_project=clear, project_open=opened, project_save_as=saved,
        project_has_path=lambda: current is not None, project_poll_write=lambda: dict(path=current),
        ui=SimpleNamespace(get_import_state=lambda: dict(active=False)),
        get_scene=lambda: scene, scene=SimpleNamespace(NodeType=SimpleNamespace(SPLAT='splat')),
        io=SimpleNamespace(load=lambda path: SimpleNamespace(splat_data=data)))
    class Panel:
        _gallery_review = False
        refresh_catalog = staticmethod(catalog)
        def _asset_index_assets(self): return assets
        def _select_asset_id(self, identifier):
            assert identifier == 'large-id'
            calls.append('select large')
        def _gallery_command(self, command):
            assert command == 'publish' and current is None
            assert calls[-1] == ('select large' if not self._gallery_review else 'review')
            if not self._gallery_review:
                self._gallery_review = True
                calls.append('review')
            else:
                assert self._gallery_upload_format == 'studio'
                assert self._gallery_title == 'test-large' and self._gallery_visibility == 'private'
                calls.append('publish')
                jobs.append(dict(id='prepared-upload'))
    scope = dict(lf=lf, p=Panel())  # No service queue API: bypassing the panel fails.
    run.rpc = lambda code: exec(code, scope)
    run.value = lambda expression: eval(expression, scope)
    run.jobs_now = lambda: jobs
    path = run.large_fixture()
    assert current is None and path.parent == tmp_path / 'projects'
    assert run.large_node_count == (2 if supplied else 1)
    assert run.queue_large(path) == 'prepared-upload'
    assert run.asset_id == 'large-id'
    assert calls[-3:] == ['select large', 'review', 'publish']
    assert ('open supplied' in calls) == supplied
    assert ('save as' in calls) != supplied
    if supplied:
        assert source.exists() and source != path

@pytest.mark.parametrize('fraction,eligible', [(.29, False), (.3, True), (.5, True), (.7, True), (.71, False)])
def test_mid_upload_requires_real_large_upload_bytes(waiting_run, fraction, eligible):
    job = dict(status='running', total=32 * 1024 * 1024, completed=fraction * 32 * 1024 * 1024,
        checkpoint=dict(uploadId='upload'))
    def wait(identifier, predicate, label):
        assert bool(predicate(job)) == eligible
        assert not predicate(None)
        for patch in (dict(checkpoint={}), dict(checkpoint=None), dict(checkpoint=dict(uploadId=None)),
                      dict(serverProcessing=True), dict(status='waiting'), dict(kind='download'),
                      dict(preparation='staging', packaged=None), dict(preparation='staging', packaged=False),
                      dict(message='Preparing scene package'), dict(total=None), dict(completed=None),
                      dict(total=0), dict(total=16, completed=8)):
            assert not predicate(dict(job, **patch))
        return job
    waiting_run.wait_job = wait
    assert waiting_run.mid_upload('job') is job

@pytest.mark.parametrize('id_source', ['checkpoint', 'uploadId'])
def test_mid_upload_waits_through_native_preparation_until_45_percent(waiting_run, id_source):
    preparation_total, upload_total = 340103744, 100 * 1024 * 1024
    preparing = dict(id='job', kind='upload', status='running', preparation='staging', packaged=False,
        completed=191889408, total=preparation_total, message='Preparing scene package',
        checkpoint=None, uploadId=None, serverProcessing=None, stagedImport={})
    uploading = dict(preparing, packaged=True, completed=0, total=upload_total, message='Uploading')
    identified = dict(uploading, **({'checkpoint': dict(uploadId='upload')}
                                  if id_source == 'checkpoint' else {'uploadId': 'upload'}))
    middle = dict(identified, completed=upload_total * .45)
    sequence = [preparing,
                dict(preparing, completed=preparation_total * .45),
                dict(preparing, message='Uploading'),  # Worker startup label, still unpackaged.
                dict(preparing, checkpoint=dict(uploadId='upload')),  # Preparation still wins.
                uploading, dict(uploading, completed=upload_total * .45),  # No upload ID yet.
                identified, dict(identified, completed=upload_total * .29), middle]
    observed = []
    def jobs():
        job = sequence[len(observed)]
        observed.append(job)
        return [job]
    waiting_run.jobs_now = jobs
    assert waiting_run.mid_upload('job') is middle
    assert observed == sequence
    assert middle['total'] == upload_total
    assert waiting_run.last_wait['label'] == '30–70% upload'

@pytest.mark.parametrize('failure', [None, 'nodes', 'title', 'id', 'missing', 'extra'])
def test_large_scene_requires_expected_ready_portal_nodes(waiting_run, failure):
    run = waiting_run
    run.prefix, run.large_node_count = 'test-', 2
    row = dict(id='scene', title='test-large', nodes=2)
    if failure in {'nodes', 'title', 'id'}:
        row[failure] = 1 if failure == 'nodes' else 'wrong'
    def db(code, expression):
        if code:
            assert "pk='scene', ready=True, deleted_at__isnull=True" in code
            assert "bundle_index.get('manifest', {}).get('nodes', [])" in expression
            if failure == 'missing':
                raise LookupError('Scene does not exist')
            return row
        return [row] * (2 if failure == 'extra' else 1)
    run.db = db
    if failure:
        with pytest.raises((AssertionError, LookupError)):
            run.assert_large_scene(dict(result=dict(id='scene')))
    else:
        run.assert_large_scene(dict(result=dict(id='scene')))
        assert run.observations[-1]['value']['nodes'] == 2

@pytest.mark.parametrize('sent', [8, 16])
@pytest.mark.parametrize('id_source', ['checkpoint', 'uploadId'])
def test_outage_requires_manual_resume_and_reuses_parts(waiting_run, tmp_path, sent, id_source):
    run = waiting_run
    run.asset_id = 'large-id'
    calls, events = [], []
    run.proxy = SimpleNamespace(snapshot=lambda: list(events))
    run.portal = object()
    run.large_fixture = lambda: tmp_path / 'large.licht'
    run.queue_large = lambda path: 'upload-job'
    middle = dict(total=16, completed=8, checkpoint=dict(uploadId='upload'))
    if id_source == 'uploadId':
        middle.update(checkpoint=None, uploadId='upload')
    run.mid_upload = lambda identifier: middle
    run.worker_stop = lambda: calls.append('worker stopped')
    run.worker_start = lambda: calls.append('worker started')
    run.stop = lambda proc: calls.append('portal stopped')
    run.assert_responsive = lambda: calls.append('responsive')
    run.portal_restart = lambda: calls.append('portal restarted')
    def part_rows(identifier):
        assert identifier == 'upload'
        return [dict(number=1, etag='retained', size=8)]
    run.part_rows = part_rows
    def wait_job(identifier, predicate, label):
        if 'paused after connection loss' in label:
            assert not predicate(dict(status='running', message='Uploading'))
            job = dict(status='paused', message='Paused (connection lost)')
            assert predicate(job)
            calls.append(identifier + ' waiting')
        elif 'resumed parts' in label:
            assert calls[-1] == "p._controller().command('resume', 'upload-job')"
            events.append(dict(method='PUT', request_bytes=sent, finished=1))
            job = dict(serverProcessing=True)
            assert predicate(job)
            calls.append('parts accepted')
        else:
            assert label == 'download in progress'
            job = dict(total=16, completed=4)
            assert predicate(job)
            calls.append('download in progress')
        return job
    run.wait_job = wait_job
    def finished(identifier):
        calls.append(identifier + ' completed')
        return dict(result=dict(id='scene'))
    run.finish_job = finished
    def published(job):
        run.scene_id = job['result']['id']
        calls.append('published nodes checked')
    run.assert_large_scene = published
    run.refresh = lambda: None
    run.rpc = calls.append
    run.value = lambda expression: ['large-id']
    def pull(remote_only):
        assert remote_only and run.scene_id == 'scene'
        return 'download-job'
    run.pull = pull
    def registered(expression, label, accept):
        assert accept(['large-id', 'downloaded-id'])
        assert not accept(['large-id'])
        calls.append('registered')
    run.wait_value = registered
    if sent == 16:
        with pytest.raises(AssertionError):
            run.portal_down_mid_transfer()
        assert 'worker started' not in calls
    else:
        run.portal_down_mid_transfer()
        assert calls.index('parts accepted') < calls.index('worker started') < calls.index('published nodes checked')
        assert "new.unlink('large-id')" in calls
        assert [call for call in calls if "command('resume'" in call] == [
            "p._controller().command('resume', 'upload-job')", "p._controller().command('resume', 'download-job')"]
        assert calls.count('portal stopped') == calls.count('portal restarted') == 2
        assert calls.index('download in progress') < calls.index('download-job waiting')
        assert calls[-2:] == ['registered', 'published nodes checked']
        evidence = next(row['value'] for row in run.observations if row['label'] == 'outage resume parts')
        assert evidence['resumed_bytes'] < evidence['total']

def test_second_launcher_has_shared_oracle_and_profile_but_owns_only_its_processes(waiting_run, tmp_path, monkeypatch):
    run = waiting_run
    run.command, run.report, run.home = 'test', tmp_path / 'launcher.md', tmp_path / 'home'
    run.origin, run.backend = 'http://127.0.0.1:22', 'http://127.0.0.1:23'
    run.portal = object()
    run.mcp = SimpleNamespace(url='http://127.0.0.1:24/mcp')
    run.asset_id, run.scene_id = 'asset', 'scene'
    calls = []
    class Second:
        def __init__(self, options, command, name, directory):
            self.args = options
            self.mcp = SimpleNamespace(url='http://127.0.0.1:25/mcp')
            self.artifacts = directory / 'second'
            self.processes = []
        def start_display(self):
            assert self.args.display == ':95'
            calls.append('display')
        def start_app(self):
            assert self.home == run.home and self.origin == run.origin
            assert self.backend == run.backend
            calls.append('app')
        def panel(self): calls.append('panel')
        def sign_in(self):
            assert self.portal is run.portal and self.portal not in self.processes
            calls.append('auth')
        def rpc(self, code): pass
        def wait_value(self, *a, **kw): pass
        def refresh(self): pass
        def link(self): return dict(sceneId='scene')
    monkeypatch.setattr(stress, 'StressRun', Second)
    other = run.launch_second_studio()
    assert other is run.other
    assert calls == ['display', 'app', 'panel', 'auth']
    assert run.args.display == ':94'

@pytest.mark.parametrize('width,height', [(250, 230), (1000, 400)])
def test_grid_pages_use_real_panel_window_and_scroll_api(waiting_run, width, height):
    """Run the product's viewport math without importing the native UI module."""
    import ast
    import math
    import typing
    source = ast.parse((SCRIPTS.parent / 'src/python/lfs_plugins/asset_manager_panel.py').read_text())
    methods = {'_sync_asset_window_viewport', '_scroll_cursor_into_view', '_gallery_columns', '_window_assets'}
    panel_class = next(n for n in source.body if isinstance(n, ast.ClassDef) and any(
        isinstance(m, ast.FunctionDef) and m.name == '_window_assets' for m in n.body))
    selected = [n for n in source.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id.startswith('ASSET_') for t in n.targets)]
    selected += [n for n in panel_class.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    scope = dict(math=math, List=typing.List, Dict=typing.Dict, Any=typing.Any)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<panel viewport methods>', 'exec'), scope)
    panel = SimpleNamespace(_view_mode='gallery', _asset_window_scroll_top=0.,
        _asset_window_client_height=height, _asset_window_client_width=width)
    scroll = SimpleNamespace(scroll_top=0., client_height=height, client_width=width)
    panel._asset_scroll_container = lambda doc=None: scroll
    for name in methods:
        setattr(panel, name, scope[name].__get__(panel))
    cards = [dict(id=f'remote:{i}') for i in range(50)]
    panel._filtered_assets = lambda: cards
    panel._refresh_records = lambda **kw: None
    waiting_run.value = lambda expression: eval(expression, {'p': panel})
    visited = set()
    for index, card in enumerate(cards):
        if card['id'] in visited:
            continue
        page = waiting_run.grid_page(index)
        assert card['id'] in page['cards']
        visited.update(page['cards'])
        assert scroll.scroll_top == page['top']
        panel._sync_asset_window_viewport()  # subsequent frames keep the new page
        assert panel._asset_window_scroll_top == page['top']
    assert visited == {c['id'] for c in cards}
    if width == 250:
        assert scroll.scroll_top > 10000  # old fixed range could never reach the end

def test_poster_log_parser_counts_query_urls_and_conditional_bytes():
    lines = ['[date] "GET /api/gallery/v1/splats/a/thumbnail?size=256 HTTP/1.1" 200 858',
             '[date] "GET /api/gallery/v1/splats/a/thumbnail?size=256 HTTP/1.1" 304 0',
             '[date] "GET /api/gallery/v1/splats/other/thumbnail HTTP/1.1" 200 9']
    events = common.portal_poster_requests(lines, ['a'])
    assert len(events) == 2
    common.assert_poster_requests(events[:1], ['a'], 200)
    common.assert_poster_requests(events[1:], ['a'], 304)
    for bad in ([], events[1:] * 2, [dict(events[1], response_bytes=1)],
                [dict(events[1], path='/api/gallery/v1/splats/b/thumbnail')]):
        with pytest.raises(AssertionError):
            common.assert_poster_requests(bad, ['a'], 304)

def test_unique_artifact_directories_preserve_existing_evidence(tmp_path):
    root = tmp_path / 'artifacts'
    first = common.run_directory(root)
    (first / 'saved.json').write_text('old evidence')
    second = common.run_directory(root)
    assert first != second and first.parent == second.parent == root
    assert (first / 'saved.json').read_text() == 'old evidence'

def test_log_excerpt_keeps_last_twenty_relevant_lines_over_frame_noise(tmp_path):
    lines = [f'\x1b[31m[error] gallery upload reason {i}\x1b[0m' for i in range(30)]
    lines += ['[perf] gallery frame took 0.01ms'] * 300
    (tmp_path / 'studio-0.log').write_text('\n'.join(lines))
    (tmp_path / 'portal-server.log').write_text('portal error should not replace LichtFeld Studio lines')
    excerpt = common.studio_log_excerpt(tmp_path)
    assert len(excerpt) == 20
    assert excerpt[0].endswith('reason 10') and excerpt[-1].endswith('reason 29')
    assert '\x1b' not in ''.join(excerpt)

def test_modal_attempt_survives_failed_transport_with_complete_dialog(waiting_run):
    run = waiting_run
    modal = dict(title='Resolve camera', body='Both versions changed.', buttons=[
        dict(label='Cancel'), dict(label='Mine'), dict(label='Portal'), dict(label='Both', enabled=False)])
    def fail(*a, **kw): raise TimeoutError('socket closed')
    run.mcp = SimpleNamespace(call=fail)
    with pytest.raises(TimeoutError, match='socket closed'):
        run.press_modal('Portal', modal)
    modal['buttons'].clear()
    saved = run.modal_presses[0]
    assert saved['title'] == 'Resolve camera' and saved['body'] == 'Both versions changed.'
    assert len(saved['buttons']) == 4 and saved['label'] == 'Portal' and saved['outcome'] == 'error'
    assert run.observations[0]['value'] == saved

def test_failure_json_retains_jobs_wait_modal_and_logs_even_if_evidence_fails(tmp_path, monkeypatch):
    class FakeRun:
        app = None
        def __init__(self, args, command, name, directory):
            self.artifacts = directory / 'studio'
            self.artifacts.mkdir()
            (self.artifacts / 'studio-0.log').write_text('\n'.join(f'gallery failed {i}' for i in range(25)))
            self.last_jobs = [dict(id='j', message='Disk interrupted', reason='storage unavailable')]
            self.last_wait = dict(label='job terminal', last=dict(message='Disk interrupted'))
            self.observations = [dict(label='job', value=self.last_jobs[0])]
            self.modal_presses = [dict(title='Resolve', body='Changes', buttons=[dict(label='Mine')], label='Mine')]
            self.gallery_diagnostics = {}
            self.other = None
        def setup(self): pass
        def listing_scale(self): raise AssertionError('invariant')
        def evidence(self): raise OSError('evidence RPC failed')
        def cleanup(self): pass
        diagnostics = stress.StressRun.diagnostics
    monkeypatch.setattr(stress, 'StressRun', FakeRun)
    report = tmp_path / 'report.md'
    argv = ['--build-dir', '/tmp/build', '--portal-source', '/tmp/portal', '--report', str(report),
            '--artifacts-root', str(tmp_path / 'runs'), '--scenarios', 'listing_scale']
    for _ in range(2):
        assert stress.main(argv) == 1
    files = sorted((tmp_path / 'runs').glob('*/listing_scale.json'))
    assert len(files) == 2
    row = json.loads(files[-1].read_text())
    assert row['status'] == 'FAIL' and row['message'] == 'Disk interrupted'
    assert row['jobs'][0]['reason'] == 'storage unavailable'
    assert 'evidence RPC failed' in row['reason'] and row['last_wait']['last']['message'] == 'Disk interrupted'
    assert len(row['log_excerpt']) == 20 and row['modal_presses'][0]['title'] == 'Resolve'
    assert row['observations']
    assert f'[JSON](<{files[-1]}>)' in report.read_text()

@pytest.mark.parametrize('clobber', [False, True])
def test_shared_journal_scenario_checks_rejected_write_and_refresh(waiting_run, clobber):
    run = waiting_run
    run.scene_id = 'scene'
    run.publish = lambda: None
    run.disk = run.loaded = 'before'
    run.stale = False
    run.refreshed = 0
    def refresh():
        run.loaded = run.disk
        run.stale = False
        run.refreshed += 1
    run.refresh = refresh
    run.link = lambda: dict(sceneId='scene', metadataRevision=run.loaded, contentRevision='content', checkedAt=10)
    run.remote = lambda: dict(title='B title')
    run.assert_link = lambda: None
    def b_update(): run.disk = 'after'
    def b_value(expression):
        if expression == 'new._disk_digest': return run.disk
        if expression == "list(new.snapshot()['links'])": return ['asset']
        raise AssertionError(expression)
    other = SimpleNamespace(open_save=lambda **kw: None, update=b_update, value=b_value,
        link=lambda: dict(sceneId='scene', metadataRevision='after', contentRevision='content', checkedAt=5))
    run.launch_second_studio = lambda: other
    def value(expression):
        return {'new._disk_digest': run.loaded, 'new._journal_digest()': run.disk,
                'new._stale': run.stale, 'new.message': 'Another LichtFeld Studio changed the journal',
                "list(new.snapshot()['links'])": ['asset']}[expression]
    run.value = value
    def stale_write(code):
        assert code == "new.edit('scene', 'before', {'title': 'must not clobber'})"
        run.stale = True
        if clobber:
            run.disk = 'clobbered'
    run.rpc = stale_write
    run.wait_value = lambda expression, label, **kw: kw['accept'](dict(
        busy=False, message='Another LichtFeld Studio changed the journal'))
    if clobber:
        with pytest.raises(AssertionError, match='Rejected A write changed the journal'):
            run.two_studios_one_account()
    else:
        run.two_studios_one_account()
        assert run.refreshed == 2 and run.loaded == 'after' and not run.stale
        assert run.observations[-1]['value'] == dict(before='before', after='after', refreshed='after')

@pytest.mark.parametrize('cache_bytes,bound,passes', [(1, 1, True), (2, 1, False)])
def test_thumbnail_scenario_asserts_configured_cache_bound(waiting_run, tmp_path, cache_bytes, bound, passes):
    run = waiting_run
    run.artifacts = tmp_path
    opened = False
    ids = [str(i) for i in range(50)]
    cards = ['remote:' + key for key in ids]
    def seed_scenes(n):
        assert opened, 'Open the scope before seeding the measured posters'
        return ids if n == 50 else []
    run.seed_scenes = seed_scenes
    run.set_posters = lambda _: None
    events = []
    run.proxy = SimpleNamespace(snapshot=lambda: list(events))
    def refresh():
        status = 304 if events else 200
        with (tmp_path / 'portal-restart-1.log').open('a') as log:
            for key in ids:
                path = f'/api/gallery/v1/splats/{key}/thumbnail'
                size = 0 if status == 304 else 858
                events.append(dict(path=path, status=status, response_bytes=size))
                log.write(f'"GET {path}?size=256 HTTP/1.1" {status} {size}\n')
    run.refresh = refresh
    run.wait_value = lambda *a, **kw: None
    def rpc(code):
        nonlocal opened
        if "p._select_folder_id('__gallery__')" in code:
            opened = True
            # Real scope opening refreshes the listing. If posters were already
            # fetched this would add an unwanted conditional response batch.
            if events:
                refresh()
    run.rpc = rpc
    run.grid_page = lambda index: dict(cards=cards)
    run.assert_responsive = lambda: None
    def value(expression):
        if expression == "[a['id'] for a in p._filtered_assets()]": return cards
        assert "read_preferences(new.root)['posterCacheMiB']" in expression
        return dict(bytes=cache_bytes, bound=bound)
    run.value = value
    if passes:
        run.thumbnail_cache()
        result = run.observations[-1]['value']
        assert result['cards_visited'] == 50 and result['portal_poster_requests'] == 100
    else:
        with pytest.raises(AssertionError):
            run.thumbnail_cache()


@pytest.mark.parametrize('leaked_rows', [False, True])
def test_account_switch_asserts_public_transfer_rows(waiting_run, panel_module, tmp_path, leaked_rows):
    run = waiting_run
    run.scene_id = 'scene'
    state = dict(links={'project': {'sceneId': 'scene'}}, posters={'scene': 'poster'},
                 jobs=[dict(id='upload', status='completed')], scenes=[{'id': 'scene'}])
    tray = dict(state)
    saved = dict(state)
    (tmp_path / 'posters').mkdir()
    poster = tmp_path / 'posters' / 'scene.png'
    poster.write_bytes(b'poster')
    namespace = dict(new=SimpleNamespace(snapshot=lambda: state, root=tmp_path),
                     p=SimpleNamespace(_gallery_remote_assets=lambda: state['scenes'],
                                       _controller=lambda: SimpleNamespace(snapshot=lambda: tray)))
    run.publish = run.refresh = run.assert_link = lambda: None
    run.set_posters = lambda _: None
    run.rpc = lambda code: exec(code, namespace)
    run.value = lambda expression: eval(expression, namespace)
    def switch(account):
        state.clear()
        state.update(saved if account == 'A' else dict(links={}, posters={}, jobs=[], scenes=[]))
        tray.clear()
        tray.update(state)
        if account == 'B':
            poster.unlink()
            if leaked_rows:
                tray['jobs'] = saved['jobs']
    run.switch_account = switch
    if leaked_rows:
        with pytest.raises(AssertionError):
            run.account_switch()
    else:
        run.account_switch()

@pytest.mark.parametrize('failure', [None, 'legacy', 'wrong_reason', 'opposite_reason', 'extra_text', 'registered',
                                     'staging', 'paused', 'resume', 'download_file', 'partial'])
@pytest.mark.parametrize('failed_mode', ['over_length', 'corrupted'])
def test_bad_downloads_requires_localized_reason_and_cleanup(
        waiting_run, panel_module, tmp_path, monkeypatch, failure, failed_mode):
    run = waiting_run
    locale = json.loads((SCRIPTS.parent / 'src/visualizer/gui/resources/locales/en.json').read_text(encoding='utf-8'))
    expected = {
        'over_length': locale['asset_manager.gallery.error.download_size'],
        'corrupted': locale['asset_manager.gallery.error.download_damaged'],
    }
    run.asset_id, run.scene_id = 'asset', 'scene'
    run.proxy = SimpleNamespace(inflate_download=0)
    run.publish = run.refresh = lambda: None
    calls = []
    current_job = {}
    namespace = dict(Path=Path, new=SimpleNamespace(_job=lambda _: current_job),
                     p=SimpleNamespace(_controller=lambda: SimpleNamespace(
                         snapshot=lambda: dict(jobs=[current_job]))))
    def rpc(code):
        calls.append(code)
        if code.startswith('from lfs_plugins.gallery_transfer_panel'):
            exec(code, namespace)
    run.rpc = rpc
    run.db = lambda code: calls.append(code)
    run.wait_value = lambda *a, **kw: None
    current = None
    stage = tmp_path / 'failed.licht'
    def pull(remote_only):
        nonlocal current
        assert remote_only
        current = 'over_length' if run.proxy.inflate_download else 'corrupted'
        return current
    run.pull = pull
    def value(expression):
        assert expression == 'sorted(p._asset_index_assets())'
        return ['asset', 'unexpected'] if failure == 'registered' and current == failed_mode else ['asset']
    run.value = value
    def wait_job(identifier, accept, label):
        message = expected[identifier]
        if identifier == failed_mode:
            if failure == 'legacy': message = 'Invalid download checksum'
            if failure == 'wrong_reason': message = locale['asset_manager.gallery.state.connection_lost']
            if failure == 'opposite_reason':
                message = expected['corrupted' if identifier == 'over_length' else 'over_length']
            if failure == 'extra_text': message += ' Unexpected diagnostic'
            if failure == 'staging': stage.write_bytes(b'leftover')
        job = dict(id=identifier, status='error', message=message, retryable=False,
                   path=str(tmp_path / (identifier + '.licht')))
        if identifier == 'corrupted':
            job.update(message='Download failed', stagedImport=dict(state='failed', message=message, path=str(stage)))
        elif failure == 'staging':
            job['stagedImport'] = dict(path=str(stage))
        if failure == 'paused' and identifier == failed_mode:
            job.update(status='paused')
            job.pop('stagedImport', None)
        if identifier == failed_mode:
            if failure == 'resume':
                job['retryable'] = True
            if failure == 'download_file':
                Path(job['path']).write_bytes(b'invalid')
            if failure == 'partial':
                path = Path(job['path'])
                path.with_name('.' + path.name + '.part').write_bytes(b'partial')
        current_job.clear()
        current_job.update(job)
        assert accept(job)
        return job
    run.wait_job = wait_job
    if failure:
        with pytest.raises(AssertionError):
            run.bad_downloads()
    else:
        run.bad_downloads()
        assert [row['label'] for row in run.observations] == ['over_length', 'corrupted']
        assert calls.count("assert not list(new.root.rglob('.gallery-*'))") == 2
        assert "new.discard('over_length')" in calls and "new.discard('corrupted')" in calls
        # Execute the actual injected mutation: it must damage a native project
        # without introducing a second transport-size mismatch.
        from lfs_plugins.portal_gallery import validate_download
        data = (SCRIPTS.parent / 'tests/data/portable-ply.licht').read_bytes()
        stored = tmp_path / 'storage.licht'
        stored.write_bytes(data)
        mutation = next(code for code in calls if code.startswith('s=Scene.objects.get'))
        exec(mutation, dict(Scene=SimpleNamespace(objects=SimpleNamespace(
            get=lambda **kwargs: SimpleNamespace(object_key='key'))),
            storage=SimpleNamespace(local_path=lambda key: stored)))
        assert stored.stat().st_size == len(data) and stored.read_bytes() != data
        with pytest.raises(ValueError, match='damaged or was changed'):
            validate_download(stored, '.licht')

def test_constructor_failure_has_explicit_unavailable_diagnostics(tmp_path, monkeypatch):
    def failed(*args): raise PermissionError('loopback socket unavailable')
    monkeypatch.setattr(stress, 'StressRun', failed)
    report = tmp_path / 'setup.md'
    assert stress.main(['--build-dir', '/tmp/build', '--portal-source', '/tmp/portal',
                        '--report', str(report), '--scenarios', 'listing_scale']) == 1
    row = json.loads(next((tmp_path / 'setup-artifacts').glob('*/listing_scale.json')).read_text())
    assert row['last_wait'] is None and row['log_excerpt'] == [] and row['jobs'] == []
    assert row['observations'] == [] and row['modal_presses'] == []
    assert 'loopback socket unavailable' in row['reason']
    assert row['message'].startswith('No job message available:')
    assert 'may not have started' in row['log_excerpt_error']

def test_safe_job_keeps_reason_and_is_idempotent():
    job = dict(id='j', message='Interrupted', reason='socket closed', checkpoint=dict(uploadId='u', url='SECRET'))
    safe = common.safe_job(job)
    assert common.safe_job(safe) == safe
    assert safe['reason'] == 'socket closed' and 'SECRET' not in json.dumps(safe)
