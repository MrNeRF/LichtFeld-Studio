#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Adversarial LOCAL LichtFeld Studio/gallery integration harness. Never builds or commits."""
from __future__ import annotations

import argparse
import array
import copy
import json
import math
from pathlib import Path
import random
import shlex
import shutil
import signal
import sys
import time
import urllib.error
import urllib.request

import validate_gallery_sync as e2e
from gallery_sync_e2e_common import (FaultProxy, assert_retained_parts,
    assert_revision_progress, confirmation_label, remote_json, retry_evidence, safe_job,
    scenario_seed, shell_json, panel_refresh_ready, run_directory, studio_log_excerpt,
    portal_poster_requests, assert_poster_requests, SHELL_MARKER)

SCENARIOS = (
    "round_trip_cycles", "web_edits_during_idle", "conflict_both_sides", "replaced_elsewhere",
    "kill_during_upload", "portal_down_mid_transfer", "slow_processing_watchdog",
    "remote_delete_and_recreate", "account_switch", "two_studios_one_account",
    "listing_scale", "thumbnail_cache", "bad_downloads", "rate_limit_429",
)
TERMINAL = {"completed", "error", "paused", "conflict", "canceled"}
MAX_WAIT = 120.
PANEL_REFRESH = """dict(
    checked=(element.get_inner_rml().strip() if
        (doc := lf.ui.rml.get_document('lfs.asset_manager')) is not None and
        (element := doc.query_selector('.gallery-checked')) is not None else None),
    checked_label=p._gallery_checked_label(), counts=p._gallery_counts(),
    selection_count=p.get_selected_count(), selected=p._get_selected_asset(),
    available=p._project_available(p._get_selected_asset() or {}),
    catalog=sorted(p._asset_index_assets()), cards=sorted(p._gallery_remote_assets()))"""


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--portal-source", type=Path, required=True)
    parser.add_argument("--portal-python", type=Path)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--large-fixture", type=Path)
    parser.add_argument("--report", type=Path, default=Path("gallery-sync-stress.md"))
    parser.add_argument("--artifacts-root", type=Path,
                        help="Parent for unique run directories (default: <report stem>-artifacts)")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--display", default=":94")
    parser.add_argument("--timeout", type=positive, default=120.,
                        help="Seconds per wait (capped at 120)")
    parser.add_argument("--watchdog-seconds", type=positive, default=130.,
                        help="Hold processing for longer than the product watchdog bound")
    args = parser.parse_args(argv)
    args.scenarios = [name.strip() for name in args.scenarios.split(",")]
    if not args.scenarios or any(name not in SCENARIOS for name in args.scenarios):
        parser.error("--scenarios must contain names from: " + ", ".join(SCENARIOS))
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("duplicate scenarios are not allowed")
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    if not args.display.startswith(":") or not args.display[1:].isdigit():
        parser.error("--display must be :N")
    args.build_dir = args.build_dir.resolve()
    args.portal_source = args.portal_source.resolve()
    if args.large_fixture:
        args.large_fixture = args.large_fixture.resolve()
        if args.large_fixture.suffix != ".licht" or not args.large_fixture.is_file():
            parser.error("--large-fixture must be an existing .licht file")
    return args


def render_report(rows, command):
    lines = ["# Gallery sync stress", "", "```sh", command, "```", "",
             "Revision tokens are opaque hashes. Advancement means a new token for each distinct edit; exchange times must not decrease.",
             "", "| Scenario | Result | Seconds | Evidence / failure |", "| --- | --- | ---: | --- |"]
    for row in rows:
        detail = row.get("error", "") or row.get('reason', '')
        detail += f" [JSON](<{row['json']}>)"
        if row.get("screenshot"):
            detail += f" [failure screenshot](<{row['screenshot']}>)"
        lines.append(f"| {row['name']} | {row['status']} | {row['seconds']:.2f} | {e2e.md_cell(detail)} |")
    lines += ["", "Each JSON contains observations, job states, portal DB rows, log excerpts, artifact paths and cleanup results.",
              "A setup failure is FAIL, never a passed or silently skipped scenario. Existing binaries are used as supplied."]
    return "\n".join(lines) + "\n"


class StressRun(e2e.Run):
    def __init__(self, args, command, name, directory):
        options = copy.copy(args)
        options.timeout = min(args.timeout, MAX_WAIT)
        options.portal_origin = options.junit = None
        options.display_explicit = options.strict_display = True
        options.format, options.auth_timeout = "sog", options.timeout
        options.report = directory / f"{name}-launcher.md"
        super().__init__(options, command)
        self.name = name
        self.seed = scenario_seed(args.seed, name)
        self.rng = random.Random(self.seed)
        self.proxy = None
        self.other = None
        self.observations = []
        self.account_name = "A"
        self.expected_exit = None
        self.last_jobs = []
        self.last_wait = None
        self.modal_presses = []
        self.gallery_diagnostics = {}

    def stop_app(self, *, crash=False):
        # Match the process object, so a later unexpected relaunch exit still fails.
        self.expected_exit = self.app
        self.observe("expected LichtFeld Studio exit", dict(pid=self.app.pid, signal="SIGKILL" if crash else "SIGTERM"))
        self.stop(self.app, crash=crash)

    def observe(self, label, value):
        self.observations.append(dict(label=label, seconds=time.monotonic() - self.started, value=value))

    def rpc(self, code):
        timeout = min(self.args.timeout, MAX_WAIT)
        return e2e.editor_output(self.mcp.call("tools/call", {"name": "editor_run", "arguments": {
            "code": code, "timeout_ms": int(timeout * 1000), "show_console": False}},
            timeout=timeout))

    def value(self, expression):
        return remote_json(self.mcp, expression, deadline=getattr(self, "_wait_deadline", None))

    def until(self, callback, label, timeout=None, *, accept=bool):
        duration = min(MAX_WAIT, self.args.timeout, timeout if timeout is not None else MAX_WAIT)
        deadline = time.monotonic() + duration
        notice = time.monotonic()
        last = None
        previous_deadline = getattr(self, "_wait_deadline", None)
        self._wait_deadline = min(deadline, previous_deadline) if previous_deadline is not None else deadline
        try:
            while time.monotonic() < self._wait_deadline:
                if self.app and self.app is not getattr(self, 'expected_exit', None) and self.app.poll() is not None:
                    raise AssertionError(f"{label}: LichtFeld Studio exited unexpectedly; last={last!r}")
                if self.worker and self.worker.poll() is not None:
                    raise AssertionError(f"{label}: Processing loop exited; last={last!r}")
                last = callback()
                if time.monotonic() < self._wait_deadline and accept(last):
                    return last
                if time.monotonic() - notice > 20:
                    print(f"{self.name}: waiting for {label}; last={last!r}", flush=True)
                    notice = time.monotonic()
                time.sleep(min(.15, max(0., self._wait_deadline - time.monotonic())))
            raise TimeoutError("wait deadline exceeded")
        except TimeoutError as exc:
            raise TimeoutError(f"{label} (limit {duration:g}s); last={last!r}; {exc}") from exc
        finally:
            self.last_wait = dict(label=label, last=last)
            self._wait_deadline = previous_deadline

    def wait_value(self, expression, label, timeout=None, *, accept=bool):
        return self.until(lambda: self.value(expression), label, timeout, accept=accept)

    def db(self, code, expression="None"):
        output = self.manage("shell", "-c", "import json\nfrom gallery.models import Scene, Upload, UploadPart, Gallery\n"
            "from gallery import sync, storage\nfrom django.utils import timezone\n" + code +
            f"\nprint({SHELL_MARKER!r}+json.dumps({expression}, default=str))")
        return shell_json(output)

    def api(self, method, path, payload=None):
        # The oracle must remain available when LichtFeld Studio traffic is fault-injected.
        request = urllib.request.Request(self.backend + "/api/gallery/v1" + path, method=method,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization": "Bearer local-test-access", "Content-Type": "application/json",
                     "Host": self.origin.removeprefix("http://")})
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else {}

    def worker_start(self):
        self.worker = self.spawn([str(self.portal_python), "manage.py", "process_gallery_uploads"],
            self.portal_env, f"worker-{len(self.processes)}.log", self.args.portal_source)

    def worker_stop(self):
        self.stop(self.worker)
        self.worker = None

    def portal_restart(self):
        self.stop(self.portal)
        self.portal = self.spawn([str(self.portal_python), "manage.py", "runserver",
            self.backend.removeprefix("http://"), "--noreload"], self.portal_env,
            f"portal-restart-{len(self.processes)}.log", self.args.portal_source)
        def ready():
            if self.portal.poll() is not None:
                raise RuntimeError("Local portal exited during restart")
            try:
                with urllib.request.urlopen(self.backend, timeout=1):
                    return True
            except urllib.error.HTTPError as exc:
                return exc.code < 500
            except OSError:
                return False
        self.until(ready, "portal restart", 45)

    def setup(self):
        self.started = time.monotonic()
        self.start_portal()
        self.backend = self.origin
        self.proxy = FaultProxy(self.backend, self.seed, self.artifacts / "proxy.jsonl")
        self.origin = self.proxy.origin
        self.portal_env["DJANGO_EXTERNAL_BASE_URL"] = self.origin
        self.portal_restart()
        self.manage("shell", "-c", e2e.SEED.replace("native-sync@example.com", "native-sync-b@example.com")
                    .replace("local-test-access", "local-test-access-b").replace("local-test-refresh", "local-test-refresh-b"))
        projects = self.home / "projects"
        projects.mkdir()
        shutil.copyfile(e2e.REPO / "tests/data/portable-multi.licht", projects / "stress.licht")
        self.start_display()
        self.start_app()
        self.observe("binary", dict(path=str(self.args.build_dir / 'LichtFeld-Studio'),
                                    mtime=(self.args.build_dir / 'LichtFeld-Studio').stat().st_mtime))
        self.panel()
        self.sign_in()
        self.rpc("p.refresh_catalog()")
        self.wait_value("[a['path'] for a in p._asset_index_assets().values()]", "catalog contains stress.licht",
                        accept=lambda paths: any(Path(path).name == "stress.licht" for path in paths))
        self.asset_id = self.value("next(a['id'] for a in p._asset_index_assets().values() if Path(a['path']).name == 'stress.licht')")
        self.select()
        self.worker_start()
        self.observe("runtime", self.value("dict(plugin=__import__('lfs_plugins.gallery_sync', fromlist=['x']).__file__, domains=new.snapshot().get('revisionDomains'))"))

    def select(self):
        self.rpc(f"p._select_asset_id({self.asset_id!r})")

    def refresh(self):
        previous_label = self.begin_refresh()
        self.finish_refresh(previous_label)

    def begin_refresh(self):
        self.wait_value("new.busy", "idle before refresh", accept=lambda busy: not busy)
        previous_label = self.value(PANEL_REFRESH)["checked"]
        # Capture on LichtFeld Studio's clock immediately before starting the request.
        self.rpc("_stress_refresh_t0 = time.time()\np._controller().refresh()")
        return previous_label

    def finish_refresh(self, previous_label):
        result = self.wait_value("""(lambda s: dict(snapshot=dict(checkedAt=s['checkedAt'],
            refresh_ok=s['refresh_ok'], links={k: dict(sceneId=v['sceneId']) for k, v in s['links'].items()},
            scenes=[dict(id=v['id'], status=v.get('status')) for v in s['scenes']]),
            busy=new.busy, t0=_stress_refresh_t0))(new.snapshot())""",
            "gallery refresh: fresh checkedAt, idle and refresh_ok",
            accept=lambda value: value["snapshot"]["checkedAt"] >= value["t0"]
                and not value["busy"] and value["snapshot"]["refresh_ok"])
        self.wait_value(PANEL_REFRESH, "panel refresh: checked label or counts and remote cards",
            accept=lambda value: panel_refresh_ready(value, previous_label, result["snapshot"]))

    def state(self, state, label=None):
        self.wait_value(f"p._gallery_facts(p._asset_dict({self.asset_id!r}) or {{}})", state,
                        accept=lambda facts: facts["state"] == state)
        if label:
            self.wait_value(f"p._gallery_badge(p._asset_dict({self.asset_id!r}) or {{}})", label,
                            accept=lambda badge: badge["gallery_label"] == label)
        self.observe("badge", self.value(f"p._gallery_badge(p._asset_dict({self.asset_id!r}))"))

    def jobs_now(self):
        jobs = self.value("new.snapshot()['jobs']")
        self.last_jobs = [safe_job(job) for job in jobs]
        return jobs

    def wait_job(self, identifier, predicate, label):
        latest = None
        def observe():
            nonlocal latest
            latest = next((j for j in self.jobs_now() if j['id'] == identifier), None)
            return safe_job(latest) if latest is not None else dict(id=identifier, status='not found')
        self.until(observe, label, accept=lambda _: latest is not None and predicate(latest))
        return latest

    def finish_job(self, identifier, expected="completed"):
        job = self.wait_job(identifier, lambda j: j['status'] in TERMINAL, f"job {identifier} terminal")
        self.observe("job", safe_job(job))
        if job["status"] != expected:
            raise AssertionError(f"Expected {expected}: {safe_job(job)}")
        self.wait_value("new.busy", "worker idle", accept=lambda busy: not busy)
        return job

    def new_job(self, before):
        observed = self.until(lambda: [safe_job(j) for j in self.jobs_now()], "new transfer",
                              accept=lambda jobs: any(j['id'] not in before for j in jobs))
        jobs = [j for j in observed if j['id'] not in before]
        if len(jobs) != 1:
            raise AssertionError(f"Expected one new transfer: {[safe_job(j) for j in jobs]}")
        return jobs[0]["id"]

    def publish(self):
        before = {j["id"] for j in self.jobs_now()}
        self.select()
        self.rpc(f"""p._gallery_expanded = True
p._gallery_command('publish')
assert p._gallery_review
p._gallery_upload_format = 'sog'
p._gallery_title = {self.prefix + self.name!r}
p._gallery_description = 'Local adversarial sync fixture'
p._gallery_visibility = 'private'
p._gallery_command('publish')
""")
        job = self.finish_job(self.new_job(before))
        self.wait_value("sorted(new.snapshot()['links'])", "publication link",
                        accept=lambda links: self.asset_id in links)
        self.scene_id = self.value(f"new.snapshot()['links'][{self.asset_id!r}]['sceneId']")
        self.refresh()
        self.state("equal", "Up to date")
        self.assert_scene_count(1)
        return job

    def assert_scene_count(self, count):
        rows = self.db("", "list(Scene.objects.filter(ready=True, deleted_at__isnull=True).values('id', 'title', 'content_length'))")
        self.observe("live DB scenes", rows)
        assert len(rows) == count, rows

    def remote(self):
        code, scene = self.api("GET", f"/splats/{self.scene_id}")
        assert code == 200, (code, scene)
        return scene

    def link(self):
        return self.value(f"new.snapshot()['links'][{self.asset_id!r}]")

    def assert_link(self):
        scene, link = self.remote(), self.link()
        for key in ("contentRevision", "metadataRevision"):
            assert link.get(key) and link[key] == scene[key], (key, link, scene)
        self.observe("exchange", {key: link[key] for key in ("sceneId", "contentRevision", "metadataRevision", "exchangedAt")})
        return link

    def open_save(self, exposure=None, camera=None):
        path = self.value(f"p._asset_dict({self.asset_id!r})['path']")
        self.rpc(f"if lf.project_poll_write().get('path') != {path!r}:\n    lf.project_open({path!r}, discard_changes=True, keep_asset_manager_open=True)")
        self.wait_value("dict(write=lf.project_poll_write(), importing=lf.ui.get_import_state())", "open project",
                        accept=lambda value: value['write'].get('path') == path and not value['importing'].get('active'))
        code = f"lf.get_render_settings().color_exposure = {exposure!r}\n" if exposure is not None else ""
        if camera:
            code += f"assert lf.ui.set_camera_path({camera!r})\n"
        self.rpc(code + "assert lf.project_save(wait=True)\np.refresh_catalog()")

    def update(self):
        self.select()
        self.rpc("p._gallery_command('primary')")
        self.wait_equal_confirming("Update completed")
        self.refresh()
        self.state("equal", "Up to date")
        self.assert_link()

    def confirm_if_open(self):
        # The native accessor gives the current labels even after the controller
        # has consumed and cleared its pending confirmation tuple.
        modal = self.value("lf.ui.modal_get()")
        if modal:
            self.press_modal(confirmation_label(modal), modal)

    def press_modal(self, label, modal=None):
        modal = modal if modal is not None else self.value("lf.ui.modal_get()")
        assert modal, f"No modal open for {label!r}"
        assert any(b['label'] == label and b.get('enabled', True) for b in modal['buttons']), modal
        event = dict(label=label, title=modal.get('title', ''),
                     body=modal.get('body', modal.get('message', '')), buttons=copy.deepcopy(modal['buttons']),
                     modal=copy.deepcopy(modal), outcome='attempted')
        self.modal_presses.append(event)
        self.observe('modal press', event)
        deadline = getattr(self, '_wait_deadline', None)
        timeout = min(75., self.args.timeout, deadline - time.monotonic() if deadline is not None else MAX_WAIT)
        if timeout <= 0:
            event.update(outcome='error', error='modal press exceeded wait deadline')
            raise TimeoutError('modal press exceeded wait deadline')
        try:
            result = self.mcp.call('tools/call', {'name': 'ui_modal_press', 'arguments': {'label': label}}, timeout=timeout)
            event['outcome'] = 'error' if result and result.get('isError') else 'returned'
            return result
        except Exception as exc:
            event.update(outcome='error', error=str(exc))
            raise

    def wait_equal_confirming(self, description):
        def ready():
            self.confirm_if_open()
            return self.value(f"dict(busy=new.busy, facts=p._gallery_facts(p._asset_dict({self.asset_id!r}) or {{}}))")
        self.until(ready, description, accept=lambda value: not value['busy'] and value['facts']['state'] == 'equal')

    def edit_db(self, fields):
        self.db(f"Scene.objects.filter(pk={self.scene_id!r}).update(**{fields!r})")

    def camera(self, offset=0):
        return self.db("from gallery.tests.test_camera_path import native_path\ntrack = native_path()\n"
            f"track['keyframes'][0]['position'][0] += {offset!r}", "track")

    def pull(self, remote_only=False):
        before = {j["id"] for j in self.jobs_now()}
        selected = "remote:" + self.scene_id if remote_only else self.asset_id
        self.rpc(f"p._select_asset_id({selected!r})\np._gallery_command('pull')\nif p._gallery_pull_review:\n    p._gallery_command('pull')")
        return self.new_job(before)

    def resolve(self, choice):
        self.rpc("p._gallery_command('resolve')")
        labels = {"mine": "Keep mine", "portal": "Keep portal", "both": "Keep both"}
        pressed = []
        buttons = []
        def ready():
            nonlocal buttons
            modal = self.value("lf.ui.modal_get()")
            if modal:
                # Discover actual localized button labels from the product, not guessed tool IDs.
                buttons = self.value("[__import__('lfs_plugins.gallery_controller', fromlist=['tr']).tr('conflict.' + k) for k in ('mine', 'portal', 'both')]")
                available = [button['label'] for button in modal['buttons'] if button.get('enabled', True)]
                if not any(label in available for label in buttons):
                    self.confirm_if_open()
                    return dict(modal=modal, pressed=list(pressed))
                label = buttons[{"mine": 0, "portal": 1, "both": 2}[choice]]
                if choice == "both" and label not in available:
                    label = buttons[0]
                assert label in available, modal
                self.press_modal(label, modal)
                pressed.append(label)
            return dict(modal=modal, pressed=list(pressed), **self.value(
                f"dict(busy=new.busy, facts=p._gallery_facts(p._asset_dict({self.asset_id!r}) or {{}}))"))
        self.until(ready, f"Resolve {labels[choice]} choices applied", accept=lambda value:
            bool(value['pressed']) and value['modal'] is None and not value['busy']
            and value['facts']['state'] == 'equal')
        if choice == "both" and buttons[2] not in pressed:
            raise AssertionError("Camera conflict never offered Keep both")
        self.observe("resolve buttons", pressed)
        self.wait_equal_confirming("resolution applied and saved")
        self.state("equal", "Up to date")
        self.assert_link()

    def round_trip_cycles(self):
        self.publish()
        history = [self.assert_link()]
        for cycle in range(self.args.cycles):
            self.open_save(exposure=1.1 + cycle * .07)
            self.state("local", "Saved since last publish")
            self.update()
            history.append(self.assert_link())
            view = self.remote()["viewerSettings"]
            view.update(exposure=2.1 + cycle * .07, cameraPath=self.camera(cycle + 1))
            self.edit_db({"viewer_settings": view})
            self.refresh()
            self.state("remote", "Portal changes")
            self.finish_job(self.pull())
            self.wait_equal_confirming("portal pull applied and saved")
            self.state("equal", "Up to date")
            self.rpc(f"assert abs(lf.get_render_settings().color_exposure - {view['exposure']!r}) < 1e-5\nassert lf.ui.get_camera_path() == {view['cameraPath']!r}")
            history.append(self.assert_link())
            self.assert_scene_count(1)
            jobs = self.jobs_now()
            assert len(jobs) <= 1 + 2 * (cycle + 1), "Unbounded jobs per exchange"
            assert all(j["status"] == "completed" for j in jobs), [safe_job(j) for j in jobs]
            self.observe("cycle", dict(index=cycle, jobs=len(jobs)))
        assert_revision_progress(history)

    def web_edits_during_idle(self):
        self.publish()
        self.open_save(exposure=1.05)
        self.update()
        edits = [("title", self.prefix + "web title"), ("description", "Web description"),
                 ("visibility", "public"), ("viewer_settings", {**self.remote()["viewerSettings"], "cameraPath": self.camera(4)})]
        for field, value in edits:
            self.edit_db({field: value})
            self.refresh()
            self.state("remote", "Portal changes")
            self.select()
            self.resolve("portal")
            assert self.remote()[{"viewer_settings": "viewerSettings"}.get(field, field)] == value
            if field == "viewer_settings":
                assert self.value("lf.ui.get_camera_path()") == value["cameraPath"], "Portal camera choice was not applied locally"
        for kind in ("poster", "presentation"):
            baseline = self.remote()
            self.set_posters([self.scene_id]) if kind == "poster" else self.edit_db({"presentation": {"title": "Cover presentation"}})
            self.refresh()
            self.state("equal", "Up to date")
            assert self.remote()["metadataRevision"] == baseline["metadataRevision"]
            start = len(self.proxy.snapshot())
            self.open_save(exposure=3.1 if kind == "poster" else 3.2)
            self.update()
            assert not any(e["status"] == 409 for e in self.proxy.snapshot()[start:]), "Presentation edit caused 409"

    def conflict_both_sides(self):
        self.publish()
        for number, choice in enumerate(("mine", "portal", "both")):
            local_title, remote_title = self.prefix + f"mine {number}", self.prefix + f"portal {number}"
            mine, portal = self.camera(number + 10), self.camera(number + 20)
            self.open_save(exposure=1.5 + number * .1, camera=mine)
            self.edit_db({"title": remote_title, "viewer_settings": {**self.remote()["viewerSettings"], "cameraPath": portal}})
            self.refresh()
            self.state("diverged", "Changed here and on portal")
            self.select()
            self.rpc(f"p._gallery_title = {local_title!r}")
            self.resolve(choice)
            result = self.remote()
            assert result["title"] == (remote_title if choice == "portal" else local_title), result
            frames = result["viewerSettings"]["cameraPath"]["keyframes"]
            assert len(frames) == (4 if choice == "both" else 2), frames
            expected = portal if choice == "portal" else mine
            assert frames[0]["position"] == expected["keyframes"][0]["position"], frames
            if choice == "both":
                assert frames[2]["position"] == portal["keyframes"][0]["position"], frames

    def second_upload(self, fixture, metadata):
        # Genuine second API client in a host thread, not on LichtFeld Studio's UI thread.
        import threading
        result = {}
        def work():
            try:
                from urllib.parse import urlsplit
                status, upload = self.api("POST", "/splats/uploads", dict(metadata, sourceFormat="licht",
                    contentLength=fixture.stat().st_size, idempotencyKey=f"stress-{self.seed}-{self.rng.getrandbits(64):016x}"))
                assert status in (200, 201), (status, upload)
                parts = []
                with fixture.open("rb") as stream:
                    number = 1
                    while data := stream.read(upload["partSize"]):
                        status, urls = self.api("POST", f"/splats/uploads/{upload['id']}/part-upload-urls", {"parts": [number]})
                        assert status == 200, (status, urls)
                        url = urls["urls"][0]["url"]
                        assert urlsplit(url).netloc == urlsplit(self.origin).netloc
                        with urllib.request.urlopen(urllib.request.Request(url, data=data, method="PUT", headers={"Content-Type": "application/octet-stream"}), timeout=120) as response:
                            parts.append(dict(partNumber=number, etag=response.headers["ETag"]))
                        number += 1
                status, done = self.api("POST", f"/splats/uploads/{upload['id']}/complete", {"parts": parts})
                assert status in (200, 202), (status, done)
                deadline = time.monotonic() + min(self.args.timeout, MAX_WAIT)
                while done.get("status") == "processing" and time.monotonic() < deadline:
                    time.sleep(.2)
                    status, done = self.api("GET", f"/splats/uploads/{upload['id']}")
                assert done.get("status") == "completed", done
                result["done"] = done
            except Exception as exc:
                result["error"] = str(exc)
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self.until(lambda: not thread.is_alive(), "second API upload")
        if "error" in result:
            raise AssertionError(result["error"])
        self.observe("second client", result["done"])
        return result["done"]

    def replaced_elsewhere(self):
        self.publish()
        original = self.remote()
        self.second_upload(e2e.REPO / "tests/data/portable-ply.licht", dict(title=original["title"],
            replaceSceneId=self.scene_id, baseRevisions={d: original[d + "Revision"] for d in ("content", "metadata")}))
        replacement = self.remote()
        assert replacement["contentRevision"] != original["contentRevision"]
        self.open_save(exposure=2.3)
        code, detail = self.api("PATCH", f"/splats/{self.scene_id}", dict(viewerSettings={"exposure": 3.0},
            baseRevisions={d: original[d + "Revision"] for d in ("content", "metadata")}))
        self.observe("stale domain write", dict(status=code, body=detail))
        assert code == 409 and "content" in json.dumps(detail) and "currentRevisions" in json.dumps(detail), detail
        marker = len(self.proxy.snapshot())
        self.select()
        self.rpc("p._gallery_command('update')")
        def refused_or_reviewing():
            modal = self.value("lf.ui.modal_get()")
            if modal:
                if len([b for b in modal['buttons'] if b.get('enabled', True)]) > 2:
                    return 'review'
                self.confirm_if_open()
            if any(e['status'] == 409 for e in self.proxy.snapshot()[marker:]):
                return 'HTTP 409'
            return self.value(f"p._gallery_facts(p._asset_dict({self.asset_id!r}))['freshness'] == 'diverged'")
        rejection = self.until(refused_or_reviewing, "LichtFeld Studio conflict/review path")
        self.observe("LichtFeld Studio stale update", dict(path=rejection, requests=self.proxy.snapshot()[marker:]))
        assert self.remote()["contentRevision"] == replacement["contentRevision"], "LichtFeld Studio silently overwrote replacement"
        # Dismiss an already-open review before requesting the explicit Pull-first review.
        if self.value("lf.ui.modal_get() is not None"):
            self.press_modal("Cancel")
        self.refresh()
        self.state("diverged", "Changed here and on portal")
        self.select()
        self.resolve("portal")
        self.open_save(exposure=2.7)
        self.update()
        self.assert_scene_count(1)

    def large_fixture(self):
        path = self.home / "projects" / "large.licht"
        if self.args.large_fixture:
            if self.args.large_fixture.resolve() != path.resolve():
                shutil.copyfile(self.args.large_fixture, path)
            self.rpc(f"lf.project_open({str(path)!r}, discard_changes=True, keep_asset_manager_open=True)")
            self.wait_value("dict(write=lf.project_poll_write(), importing=lf.ui.get_import_state())", "open large fixture",
                accept=lambda value: value['write'].get('path') == str(path) and not value['importing'].get('active'))
        else:
            # Build a saved project in LichtFeld Studio; the publish command below prepares
            # its embedded data into the portal's publication format.
            ply = self.root / "large.ply"
            names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                     "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
            count = 5_000_000
            with ply.open("wb") as output:
                output.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n" +
                    "".join(f"property float {name}\n" for name in names) + "end_header\n").encode())
                for start in range(0, count, 8192):
                    values = array.array("f", (self.rng.uniform(-1, 1) for _ in range(min(8192, count-start) * len(names))))
                    if sys.byteorder != "little":
                        values.byteswap()
                    output.write(values.tobytes())
            self.rpc(f"""lf.new_project(discard_changes=True)
loaded = lf.io.load({str(ply)!r}).splat_data
assert loaded is not None
lf.get_scene().add_splat('Stress Gaussian payload', loaded.means_raw, loaded.sh0_raw, loaded.shN_raw,
    loaded.scaling_raw, loaded.rotation_raw, loaded.opacity_raw)
assert lf.project_save_as({str(path)!r}, wait=True)
""")
            assert path.stat().st_size >= 200_000_000, "Generated .licht compressed below 200 MB; pass --large-fixture"
        assert path.stat().st_size > 24 * 1024 * 1024, "Large fixture must span at least four 8 MiB parts"
        self.large_node_count = self.value("len([n for n in lf.get_scene().get_nodes(lf.scene.NodeType.SPLAT) if n.gaussian_count > 0 and lf.get_scene().is_node_effectively_visible(n.id)])")
        assert self.large_node_count > 0, "Large fixture has no publishable splat nodes"
        # Exercise closed-project preparation (lf.prepare_gallery_project).
        self.rpc("lf.new_project(discard_changes=True)\nassert not lf.project_has_path()\np.refresh_catalog()")
        self.observe("large fixture", dict(path=str(path), bytes=path.stat().st_size, nodes=self.large_node_count))
        return path

    def queue_large(self, path):
        assets = self.wait_value("list(p._asset_index_assets().values())", "large project card",
            accept=lambda assets: any(a['path'] == str(path) for a in assets))
        self.asset_id = next(a['id'] for a in assets if a['path'] == str(path))
        before = {j['id'] for j in self.jobs_now()}
        self.select()
        self.rpc(f"""assert not lf.project_has_path()
p._gallery_expanded = True
p._gallery_command('publish')
assert p._gallery_review
p._gallery_upload_format = 'studio'
p._gallery_title = {self.prefix + 'large'!r}
p._gallery_description = 'Local adversarial sync fixture'
p._gallery_visibility = 'private'
p._gallery_command('publish')
""")
        return self.new_job(before)

    def assert_large_scene(self, job):
        self.scene_id = job['result']['id']
        self.assert_scene_count(1)
        row = self.db(f"scene = Scene.objects.get(pk={self.scene_id!r}, ready=True, deleted_at__isnull=True)",
            "dict(id=str(scene.id), title=scene.title, nodes=len(scene.bundle_index.get('manifest', {}).get('nodes', [])))")
        self.observe("large published scene", row)
        assert row['id'] == self.scene_id and row['title'] == self.prefix + 'large', row
        assert row['nodes'] == self.large_node_count, (row, self.large_node_count)

    def part_rows(self, upload_id):
        return self.db("", f"list(UploadPart.objects.filter(upload_id={upload_id!r}).order_by('number').values('number', 'etag', 'size'))")

    def mid_upload(self, identifier):
        def uploading(j):
            j = j or {}
            # Preparation reports bytes through the same counters. They become
            # upload bytes only after packaging resets completed/total.
            if (j.get('status') != 'running' or j.get('kind') == 'download'
                    or j.get('serverProcessing')
                    or (j.get('preparation') and not j.get('packaged'))
                    or j.get('message') == 'Preparing scene package'):
                return False
            checkpoint_id = (j.get('checkpoint') or {}).get('uploadId')
            if not (checkpoint_id or (j.get('uploadId') and j.get('message') == 'Uploading')):
                return False
            total, completed = j.get('total') or 0, j.get('completed') or 0
            return total > 24 * 1024 * 1024 and .3 <= completed / total <= .7
        return self.wait_job(identifier, uploading, "30–70% upload")

    def kill_during_upload(self):
        path = self.large_fixture()
        self.worker_stop()
        self.proxy.upload_bps = 4 * 1024 * 1024
        identifier = self.queue_large(path)
        middle = self.mid_upload(identifier)
        upload_id = (middle.get('checkpoint') or {}).get('uploadId') or middle.get('uploadId')
        self.observe("killed at", safe_job(middle))
        original_home = self.home
        self.stop_app(crash=True)
        # Let the last proxy handler finish observing the killed socket before
        # measuring restart traffic; otherwise an old PUT could be counted twice.
        self.until(lambda: all('finished' in e for e in self.proxy.snapshot()), "killed upload socket closed", 30)
        before = self.part_rows(upload_id)
        marker = len(self.proxy.snapshot())
        self.start_app()
        assert self.home == original_home, "Relaunch changed LFS_HOME"
        self.panel()
        self.sign_in()
        jobs = self.jobs_now()
        self.observe("restored interrupted jobs", [safe_job(j) for j in jobs])
        assert len(jobs) == 1 and jobs[0]["id"] == identifier and jobs[0].get("interrupted") and jobs[0]["status"] == "paused", [safe_job(j) for j in jobs]
        self.rpc(f"p._controller().command('resume', {identifier!r})")
        self.wait_job(identifier, lambda j: j.get('serverProcessing'), "resumed parts accepted before processor cleanup")
        after = self.part_rows(upload_id)
        events = self.proxy.snapshot()[marker:]
        sent = sum(e["request_bytes"] for e in events if e["method"] == "PUT")
        assert_retained_parts(before, after, middle["total"], sent)
        self.observe("resume parts", dict(before=before, after=after, resumed_bytes=sent, total=middle["total"]))
        self.worker_start()
        self.assert_large_scene(self.finish_job(identifier))
        assert len(self.jobs_now()) == 1

    def portal_down_mid_transfer(self):
        path = self.large_fixture()
        self.worker_stop()
        self.proxy.upload_bps = 4 * 1024 * 1024
        identifier = self.queue_large(path)
        middle = self.mid_upload(identifier)
        upload_id = (middle.get('checkpoint') or {}).get('uploadId') or middle.get('uploadId')
        self.observe("portal stopped at", safe_job(middle))
        self.stop(self.portal)
        job = self.wait_job(identifier, lambda j: j['status'] == 'paused' and j['message'] == 'Paused (connection lost)', "upload paused after connection loss")
        self.observe("upload outage", safe_job(job))
        self.assert_responsive()
        self.until(lambda: all('finished' in e for e in self.proxy.snapshot()), "outage upload socket closed", 30)
        before_parts = self.part_rows(upload_id)
        marker = len(self.proxy.snapshot())
        self.portal_restart()
        self.rpc(f"p._controller().command('resume', {identifier!r})")
        self.wait_job(identifier, lambda j: j.get('serverProcessing'), "manually resumed parts accepted")
        after_parts = self.part_rows(upload_id)
        sent = sum(e['request_bytes'] for e in self.proxy.snapshot()[marker:] if e['method'] == 'PUT')
        assert_retained_parts(before_parts, after_parts, middle['total'], sent)
        self.observe("outage resume parts", dict(before=before_parts, after=after_parts, resumed_bytes=sent, total=middle['total']))
        self.worker_start()
        uploaded = self.finish_job(identifier)
        self.scene_id = uploaded['result']['id']
        self.refresh()
        # Unlink the selected project so the panel presents a remote card.
        self.rpc(f"new.unlink({self.asset_id!r})")
        self.refresh()
        before = self.value("sorted(p._asset_index_assets())")
        self.proxy.download_bps = 4 * 1024 * 1024
        download = self.pull(remote_only=True)
        self.wait_job(download, lambda j: 0 < j['completed'] < j['total'] * .7, "download in progress")
        self.stop(self.portal)
        job = self.wait_job(download, lambda j: j['status'] == 'paused' and j['message'] == 'Paused (connection lost)', "download paused after connection loss")
        self.observe("download outage", safe_job(job))
        assert self.value("sorted(p._asset_index_assets())") == before, "Partial project registered"
        self.assert_responsive()
        self.portal_restart()
        self.rpc(f"p._controller().command('resume', {download!r})")
        self.finish_job(download)
        self.wait_value("sorted(p._asset_index_assets())", "resumed download registration",
                        accept=lambda ids: set(before) < set(ids) and len(ids) == len(before) + 1)
        self.assert_large_scene(uploaded)

    def assert_responsive(self):
        started = time.monotonic()
        assert self.mcp.value("1 + 1") == 2
        elapsed = time.monotonic() - started
        self.observe("editor round trip", elapsed)
        assert elapsed < 2, f"UI unresponsive: {elapsed:.3f}s"

    def slow_processing_watchdog(self):
        self.worker_stop()
        before = {j["id"] for j in self.jobs_now()}
        self.select()
        self.rpc(f"p._gallery_command('publish')\np._gallery_upload_format = 'sog'\np._gallery_title = {self.prefix+'watchdog'!r}\np._gallery_command('publish')")
        identifier = self.new_job(before)
        self.wait_job(identifier, lambda j: j.get('serverProcessing'), "portal checking")
        deadline = time.monotonic() + self.args.watchdog_seconds
        while time.monotonic() < deadline:
            self.assert_responsive()
            job = next(j for j in self.jobs_now() if j["id"] == identifier)
            assert job["status"] in {"running", "paused", "error"}, safe_job(job)
            if job["status"] == "running":
                self.state("processing", "Portal is checking")
            self.observe("watchdog", safe_job(job))
            time.sleep(min(1., max(0., deadline - time.monotonic())))
        self.rpc("p._controller().command('pause')")
        self.wait_value("new.busy", "Cancel stops waiting", 10, accept=lambda busy: not busy)
        self.rpc(f"new.discard({identifier!r})")
        self.wait_value("new.busy", "Cancel portal upload", 10, accept=lambda busy: not busy)
        self.worker_start()
        self.assert_scene_count(0)

    def remote_delete_and_recreate(self):
        self.publish()
        old = self.scene_id
        self.db(f"Scene.objects.filter(pk={old!r}).update(deleted_at=timezone.now())")
        self.refresh()
        self.state("remote_deleted", "Removed on portal")
        before = {j["id"] for j in self.jobs_now()}
        self.select()
        self.rpc("p._gallery_command('primary')")
        self.confirm_if_open()
        self.rpc("if p._gallery_review:\n    p._gallery_command('publish')")
        self.finish_job(self.new_job(before))
        self.scene_id = self.link()["sceneId"]
        assert self.scene_id != old, "Publish again reused the deleted scene"
        self.refresh()
        self.state("equal", "Up to date")
        self.assert_scene_count(1)

    def switch_account(self, name):
        suffix = "-b" if name == "B" else ""
        email = "native-sync-b@example.com" if name == "B" else "native-sync@example.com"
        self.rpc(f"""assert not new.busy
creds = _Credentials(portal_origin={self.origin!r}, access_token={'local-test-access'+suffix!r}, access_expires_at=time.time()+3600,
    refresh_token={'local-test-refresh'+suffix!r}, refresh_expires_at=time.time()+86400,
    email={email!r}, connected_since={'stress-session-'+name!r})
account._set_current_credentials(creds)
account._apply_credentials_state(creds)
new = GallerySync(account, resolve_asset_manager_storage_path()/'gallery')
sync_module._service = new
p._controller().service = new
p._controller().refresh()
""")
        self.wait_value("(lambda s: dict(busy=new.busy, connected=s['connected'], email=s['email']))(new.snapshot())",
                        "account switched", accept=lambda value:
                        not value['busy'] and value['connected'] and value['email'] == email)
        self.account_name = name
        self.refresh()

    def account_switch(self):
        self.publish()
        self.set_posters([self.scene_id])
        self.refresh()
        old = self.value("dict(links=new.snapshot()['links'], posters=new.snapshot()['posters'], jobs=[j['id'] for j in new.snapshot()['jobs']])")
        assert old["posters"], "A poster was never cached"
        self.switch_account("B")
        self.rpc("""assert not new.snapshot()['links']
assert not new.snapshot()['scenes']
assert not new.snapshot()['posters']
assert not new.snapshot()['jobs']
assert not p._gallery_remote_assets()
from lfs_plugins.gallery_transfer_panel import transfer_rows
assert not transfer_rows(p._controller().snapshot())
assert not list((new.root/'posters').glob('*.png'))
""")
        self.observe("B isolation", self.value("dict(links=new.snapshot()['links'], posters=new.snapshot()['posters'], jobs=new.snapshot()['jobs'])"))
        self.switch_account("A")
        assert self.value("{k: v['sceneId'] for k, v in new.snapshot()['links'].items()}") == {k: v["sceneId"] for k, v in old["links"].items()}
        self.assert_link()

    def launch_second_studio(self):
        options = copy.copy(self.args)
        options.display = ":" + str(int(options.display[1:]) + 1)
        self.other = StressRun(options, self.command, self.name + "-B", self.report.parent)
        other = self.other
        other.started, other.home, other.origin = time.monotonic(), self.home, self.origin
        other.backend = self.backend
        assert other.mcp.url != self.mcp.url, "Studios must use distinct MCP ports"
        # sign_in's local mode needs the owned portal handle, but B must not own/stop it.
        other.portal = self.portal
        other.start_display()
        other.start_app()
        other.panel()
        other.sign_in()
        other.asset_id, other.scene_id = self.asset_id, self.scene_id
        other.rpc("p.refresh_catalog()")
        other.wait_value("sorted(p._asset_index_assets())", "B shared catalog", accept=lambda ids: self.asset_id in ids)
        other.refresh()
        assert other.link()["sceneId"] == self.scene_id
        self.observe("second LichtFeld Studio", dict(home=str(other.home), display=other.args.display,
                                           mcp=other.mcp.url, backend=other.backend, artifacts=str(other.artifacts)))
        return other

    def two_studios_one_account(self):
        self.publish()
        other = self.launch_second_studio()
        # Establish a shared baseline before B's edit, so the digest test proves
        # that edit persisted and is independent of journal recovery during startup.
        self.refresh()
        before = self.value("new._disk_digest")
        assert before == other.value("new._disk_digest")
        other.open_save(exposure=2.65)
        other.update()
        after = other.value("new._disk_digest")
        assert before != after, "B did not write shared journal"
        assert self.value("new._disk_digest") == before, "A unexpectedly reloaded B's edit"
        assert self.value("new._journal_digest()") == after, "B's digest does not match disk"
        self.rpc(f"new.edit({self.scene_id!r}, {self.link()['metadataRevision']!r}, {{'title': 'must not clobber'}})")
        self.wait_value("dict(busy=new.busy, message=new.snapshot()['message'])", "A stale write refused",
                        accept=lambda value: not value['busy'] and 'Another LichtFeld Studio' in value['message'])
        assert self.value("new._stale"), "A did not detect changed journal digest"
        assert "Another LichtFeld Studio" in self.value("new.message")
        assert self.value("new._journal_digest()") == after, "Rejected A write changed the journal"
        self.observe("stale journal refused", dict(before=before, after=after, message=self.value("new.message")))
        self.refresh()
        assert not self.value("new._stale")
        assert self.value("new._disk_digest") == after, "Refresh did not load B's journal digest"
        assert self.value("new._disk_digest") == self.value("new._journal_digest()")
        assert self.remote()["title"] != "must not clobber"
        assert set(self.value("list(new.snapshot()['links'])")) == set(other.value("list(new.snapshot()['links'])"))
        self.assert_link()
        # Refresh changes the in-memory checkedAt timestamp independently in A/B.
        a_link, b_link = self.link(), other.link()
        assert {k: v for k, v in a_link.items() if k != 'checkedAt'} == {
            k: v for k, v in b_link.items() if k != 'checkedAt'}, "Refresh did not load B's link"
        self.observe("shared journal", dict(before=before, after=after, refreshed=self.value("new._disk_digest")))

    def seed_scenes(self, count):
        identifiers = self.db(f"""import uuid
owner = Gallery.objects.get(owner__email='native-sync@example.com')
rows = [Scene(id=uuid.UUID(int={self.seed}*1000+i+1), gallery=owner, title='Stress seeded '+str(i),
    source_format='licht', content_length=1, object_key='stress/seeded/'+str(i), backend='local', ready=True)
    for i in range({count})]
Scene.objects.bulk_create(rows)
""", "[str(s.pk) for s in rows]")
        return identifiers

    def listing_scale(self):
        identifiers = self.seed_scenes(300)
        self.proxy.list_delay = 1
        marker = len(self.proxy.snapshot())
        previous_label = self.begin_refresh()
        self.until(lambda: self.proxy.snapshot()[marker:], "listing request in flight", 5,
            accept=lambda events: any(e['path'] == '/api/gallery/v1/splats' and 'finished' not in e for e in events))
        started = time.monotonic()
        self.assert_responsive()
        finished = time.monotonic()
        self.finish_refresh(previous_label)
        assert any(e['path'] == '/api/gallery/v1/splats' and e['started'] <= started
                   and e.get('finished', 0) >= finished for e in self.proxy.snapshot()[marker:]), \
            "Responsiveness probe missed the in-flight listing request"
        self.wait_value("sorted(p._gallery_remote_assets())", "300 remote cards",
                        accept=lambda cards: cards == sorted('remote:' + key for key in identifiers))
        cards = self.value("sorted(p._gallery_remote_assets())")
        assert cards == sorted("remote:" + key for key in identifiers), "Pagination lost/duplicated cards"
        self.observe("listing scale", dict(seconds=time.monotonic()-started, cards=len(cards)))
        self.proxy.list_delay = 0
        marker = len(self.proxy.snapshot())
        self.refresh()
        assert any(e["path"] == "/api/gallery/v1/splats" and e["status"] == 304 for e in self.proxy.snapshot()[marker:]), "Second refresh did not use ETag/304"

    def set_posters(self, identifiers):
        self.db("from PIL import Image\nfrom io import BytesIO\nb=BytesIO()\n"
            f"Image.new('RGB', (256,256), ({self.rng.randrange(256)}, 110, 210)).save(b, format='PNG')\n"
            f"Scene.objects.filter(pk__in={identifiers!r}).update(poster=b.getvalue(), poster_type='image/png')")

    def thumbnail_cache(self):
        # Opening the scope is itself a manual refresh. Finish it before
        # seeding posters so each measured window contains exactly one refresh.
        self.rpc("p._select_folder_id('__gallery__')\np.set_view_mode(None, None, ['gallery'])")
        self.wait_value("new.busy", "gallery scope refresh idle", accept=lambda busy: not busy)
        identifiers = self.seed_scenes(50)
        self.set_posters(identifiers)
        marker = len(self.proxy.snapshot())
        self.refresh()
        self.wait_value("sorted(new.snapshot()['posters'])", "50 posters cached",
                        accept=lambda posters: posters == sorted(identifiers))
        self.wait_value("(p._sync_asset_window_viewport(), p._asset_window_client_height)[1]",
                        "gallery scroll viewport", accept=lambda height: height > 0)
        initial = [e for e in self.proxy.snapshot()[marker:] if e['path'].endswith('/thumbnail')]
        assert_poster_requests(initial, identifiers, 200)
        marker = len(self.proxy.snapshot())
        cards = self.value("[a['id'] for a in p._filtered_assets()]")
        assert set(cards) == {'remote:' + key for key in identifiers}, cards
        visited, pages = set(), []
        for index, card in enumerate(cards):
            if card in visited:
                continue
            page = self.grid_page(index)
            pages.append(page)
            self.observe('grid page', page)
            assert card in page['cards'], f"Scroll window did not reach {card}: {page}"
            visited.update(page['cards'])
            self.assert_responsive()
        assert set("remote:" + key for key in identifiers) <= visited, "Grid paging did not visit all cards"
        assert not any(e['path'].endswith('/thumbnail') for e in self.proxy.snapshot()[marker:]), \
            "Scrolling fetched posters without an explicit refresh"
        marker = len(self.proxy.snapshot())
        self.refresh()
        events = [e for e in self.proxy.snapshot()[marker:] if e["path"].endswith("/thumbnail")]
        assert_poster_requests(events, identifiers, 304)
        # Confirm the requests also reached Django; proxy-only counts can hide
        # injected responses. Match IDs to exclude unrelated log entries.
        portal_lines = [line for path in self.artifacts.glob('portal-*.log')
                        if path.name != 'portal-manage.log'
                        for line in path.read_text(errors='replace').splitlines()]
        poster_requests = portal_poster_requests(portal_lines, identifiers)
        self.observe('portal poster requests', poster_requests)
        assert len(poster_requests) == len(identifiers) * 2, f"Portal poster count: {len(poster_requests)}"
        assert_poster_requests([e for e in poster_requests if e['status'] == 200], identifiers, 200)
        assert_poster_requests([e for e in poster_requests if e['status'] == 304], identifiers, 304)
        cache = self.value("dict(bytes=sum(f.stat().st_size for f in (new.root/'posters').glob('*.png')), bound=__import__('lfs_plugins.gallery_preferences', fromlist=['read_preferences']).read_preferences(new.root)['posterCacheMiB'] * 1024 * 1024)")
        assert cache["bytes"] <= cache["bound"], cache
        self.observe("thumbnail cache", dict(cache=cache, initial_requests=len(initial),
            conditional_requests=len(events), portal_poster_requests=len(poster_requests),
            cards_visited=len(visited), pages=len(pages)))

    def grid_page(self, index):
        # This API updates the actual scroll element as well as Python state;
        # a frame's viewport synchronization cannot undo the requested page.
        return self.value(f"""(p._sync_asset_window_viewport(), p._scroll_cursor_into_view({index}),
            p._refresh_records(assets=True), dict(index={index}, top=p._asset_window_scroll_top,
            height=p._asset_window_client_height, width=p._asset_window_client_width,
            cards=[a['id'] for a in p._window_assets(p._filtered_assets())]))[-1]""")

    def bad_downloads(self):
        locale = json.loads((Path(__file__).resolve().parents[1] /
            "src/visualizer/gui/resources/locales/en.json").read_text(encoding="utf-8"))
        expected_messages = {
            "over_length": locale["asset_manager.gallery.error.download_size"],
            "corrupted": locale["asset_manager.gallery.error.download_damaged"],
        }
        self.rpc("from lfs_plugins.gallery_messages import localize_message")
        self.publish()
        self.rpc(f"new.unlink({self.asset_id!r})")
        self.refresh()
        baseline = self.value("sorted(p._asset_index_assets())")
        for mode in ("over_length", "corrupted"):
            if mode == "over_length":
                self.proxy.inflate_download = 1024 * 1024
            else:
                self.proxy.inflate_download = 0
                # Preserve the transport length so this exercises container
                # validation independently of the storage-header size check.
                self.db(f"s=Scene.objects.get(pk={self.scene_id!r})\np=storage.local_path(s.object_key)\n"
                        "with p.open('r+b') as stream:\n    stream.seek(-1, 2)\n    tail=stream.read(1)\n"
                        "    stream.seek(-1, 2)\n    stream.write(bytes([tail[0] ^ 1]))")
            identifier = self.pull(remote_only=True)
            job = self.wait_job(identifier, lambda j: j['status'] in {'error', 'paused'}
                or j.get('stagedImport', {}).get('state') == 'failed', "specific bad-download failure")
            message = job.get("stagedImport", {}).get("message") or job["message"]
            reason = self.value(f"localize_message({message!r})")
            assert job['status'] == 'error' or job.get('stagedImport', {}).get('state') == 'failed', job
            assert reason == expected_messages[mode], dict(message=message, reason=reason)
            assert self.value("sorted(p._asset_index_assets())") == baseline, "Bad download registered a project"
            self.rpc("assert not list(new.root.rglob('.gallery-*'))")
            self.rpc(f"""from lfs_plugins.gallery_transfer_panel import transfer_rows
assert not next(row for row in transfer_rows(p._controller().snapshot()) if row['id'] == {identifier!r})['can_resume']
_bad_path = Path(new._job({identifier!r})['path'])
assert not _bad_path.exists()
assert not _bad_path.with_name('.' + _bad_path.name + '.part').exists()
""")
            stage = job.get("stagedImport", {}).get("path")
            if stage:
                assert not Path(stage).exists(), "Failed staging was not cleaned"
            self.observe(mode, dict(safe_job(job), localized_reason=reason))
            self.wait_value("new.busy", "bad transfer idle", accept=lambda busy: not busy)
            self.rpc(f"new.discard({identifier!r})")
            self.wait_value("new.busy", "bad transfer discarded", accept=lambda busy: not busy)

    def rate_limit_429(self):
        self.proxy.faults = True
        marker = len(self.proxy.snapshot())
        try:
            self.round_trip_cycles()
        finally:
            events = self.proxy.snapshot()[marker:]
            self.observe("fault schedule", events)
        self.observe("Retry-After", retry_evidence(events))

    def evidence(self):
        result = dict(observations=self.observations, artifacts=str(self.artifacts), temporary=str(self.root))
        try:
            if self.app and self.app.poll() is None:
                result["jobs"] = [safe_job(job) for job in self.jobs_now()]
                self.gallery_diagnostics = self.value(f"""dict(message=new.snapshot().get('message', ''),
                    facts=p._gallery_facts(p._asset_dict({self.asset_id!r}) or {{}}))""")
                result['gallery'] = self.gallery_diagnostics
                result["links"] = self.value("new.snapshot()['links']")
        except Exception as exc:
            result["studio_evidence_error"] = str(exc)
        try:
            if self.portal_env:
                result["database"] = self.db("", "dict(scenes=list(Scene.objects.values('id','title','ready','deleted_at','content_length')), uploads=list(Upload.objects.values('id','scene_id','status','processing_stage','processing_error')), parts=list(UploadPart.objects.values('upload_id','number','etag','size')))")
        except Exception as exc:
            result["db_evidence_error"] = str(exc)
        result["logs"] = {path.name: path.read_text(errors="replace").splitlines()[-30:] for path in self.artifacts.glob("*.log")}
        if self.other:
            result['second_studio'] = self.other.evidence()
        return result

    def diagnostics(self):
        """No RPC: usable after a crash, a transport timeout or evidence failure."""
        result = dict(jobs=self.last_jobs, last_wait=self.last_wait,
                      observations=self.observations, modal_presses=self.modal_presses,
                      gallery=self.gallery_diagnostics)
        try:
            result['log_excerpt'] = studio_log_excerpt(self.artifacts)
        except OSError as exc:
            result.update(log_excerpt=[], log_excerpt_error=str(exc))
        if self.other:
            result['second_studio'] = self.other.diagnostics()
        return result

    def cleanup(self):
        errors = []
        if self.other:
            try:
                self.other.close()
            except Exception as exc:
                errors.append(str(exc))
        # Stop producers before removing all scenes in this disposable database.
        for proc in reversed(self.processes):
            try:
                self.stop(proc)
            except Exception as exc:
                errors.append(str(exc))
        if self.proxy:
            self.proxy.close()
        if self.portal_env:
            try:
                self.db("Scene.objects.all().delete()", "Scene.objects.count()")
            except Exception as exc:
                errors.append("Database cleanup: " + str(exc))
        for handle in self.handles:
            handle.close()
        if not self.args.keep:
            shutil.rmtree(self.root)
        if errors:
            raise RuntimeError("; ".join(errors))


def main(argv=None):
    args = parse_args(argv)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])])
    args.report = args.report.resolve()
    directory = run_directory((args.artifacts_root or args.report.parent / (args.report.stem + "-artifacts")).resolve())
    rows = []
    stopped = False
    def interrupted(*_):
        raise KeyboardInterrupt("Termination requested")
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        for name in args.scenarios:
            started = time.monotonic()
            row = dict(name=name, seed=scenario_seed(args.seed, name), status="FAIL", seconds=0,
                       json=str(directory / (name + ".json")), artifacts=str(directory),
                       jobs=[], observations=[], log_excerpt=[], modal_presses=[], last_wait=None,
                       message='', reason='')
            run = None
            print(f"SCENARIO {name} seed={row['seed']}", flush=True)
            try:
                run = StressRun(args, command, name, directory)
                run.setup()
                getattr(run, name)()
                row["status"] = "PASS"
            except (Exception, KeyboardInterrupt) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                stopped = isinstance(exc, KeyboardInterrupt)
                print(f"FAIL {name}: {exc}", file=sys.stderr, flush=True)
                if run and run.app and run.app.poll() is None:
                    try:
                        screenshot = run.artifacts / "failure.png"
                        screenshot.write_bytes(e2e.png_data(run.mcp.tool("render_capture_window")))
                        row["screenshot"] = str(screenshot)
                    except Exception as capture:
                        row["screenshot_error"] = str(capture)
            finally:
                if run:
                    failure_wait = getattr(run, 'last_wait', None)
                    try:
                        row["evidence"] = run.evidence()
                    except (Exception, KeyboardInterrupt) as exc:
                        row.update(status="FAIL", evidence_error=str(exc))
                        stopped = stopped or isinstance(exc, KeyboardInterrupt)
                    try:
                        run.cleanup()
                        row["cleanup"] = "Scenes deleted; owned processes stopped; " + ("profile retained" if args.keep else "temporary files removed")
                    except Exception as exc:
                        row.update(status="FAIL", cleanup_error=str(exc))
                    if callable(getattr(run, 'diagnostics', None)):
                        try:
                            row.update(run.diagnostics())
                            row['last_wait'] = failure_wait
                        except Exception as exc:
                            row.update(status='FAIL', diagnostics_error=str(exc))
                if row['status'] == 'FAIL':
                    # Always explicit, even when startup failed before a job existed.
                    row['reason'] = '; '.join(str(row[key]) for key in
                        ('error', 'evidence_error', 'cleanup_error', 'diagnostics_error') if row.get(key))
                    messages = [j.get('message', '') or j.get('reason', '') for j in row['jobs']]
                    gallery = row.get('gallery', {})
                    messages += [gallery.get('message', ''), gallery.get('facts', {}).get('reason', '')]
                    last = (row.get('last_wait') or {}).get('last')
                    if isinstance(last, dict):
                        messages += [last.get('message', ''), last.get('reason', ''),
                                     last.get('facts', {}).get('reason', '')]
                    row['message'] = '\n'.join(dict.fromkeys(m for m in messages if m)) or 'No job message available: ' + row['reason']
                    if not row['log_excerpt']:
                        row.setdefault('log_excerpt_error', 'No LichtFeld Studio log lines available (LichtFeld Studio may not have started).')
                row["seconds"] = time.monotonic() - started
                Path(row["json"]).write_text(json.dumps(row, indent=2, default=str) + "\n")
                rows.append(row)
                args.report.write_text(render_report(rows, command))
            if stopped:
                break
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(f"Report: {args.report}", flush=True)
    return 1 if stopped or any(row["status"] != "PASS" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
