#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in Linux Studio/portal E2E; uses an existing build and private profile/display.

No app or Django imports at module scope: pure helpers are usable in ordinary CI.
See docs/docs/development/mcp/recipes/gallery-sync-e2e.md.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from html import escape
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
FORMATS = {"studio": ".ply", "sog": ".sog", "ssog": ".ssog", "spz": ".spz"}
MARKER = "<<LFSE2E>>"
END_MARKER = "<</LFSE2E>>"
# editor.run defaults to 20,000 bytes (mcp_gui_tools.cpp). Reserve room for
# diagnostics and terminal row breaks, and fail before printing oversized JSON.
EDITOR_OUTPUT_LIMIT = 20000
JSON_PAYLOAD_LIMIT = 8000
JOB_FIELDS = ('id', 'kind', 'status', 'total', 'completed', 'serverProcessing', 'error')
# RmlUi retains the hidden data-for template alongside the instantiated rows.
TRANSFER_ROW_SELECTOR = '.gallery-transfer-row:not([data-for])'
LOG_ERRORS = re.compile(r"Traceback|\[error\]|Syntax error parsing property|Missing localization key", re.I)
# Only this unrelated browser-launch failure is ignored, not arbitrary xdg-open lines.
LOG_ALLOWLIST = (re.compile(r"xdg-open: no method available for opening", re.I),)
RESOURCES = ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current")


def portal_origin(value):
    try:
        parsed = urllib.parse.urlsplit(value)
        parsed.port
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise argparse.ArgumentTypeError("--portal-origin requires an HTTPS origin without credentials or a path")
    return value.rstrip("/")


def owned_scenes(scenes, prefix):
    """Delete only this run's live scenes, never earlier E2E runs or tombstones."""
    if not re.fullmatch(r"E2E validate [0-9a-f]{32} ", prefix):
        raise ValueError("Cleanup requires a unique run prefix")
    return [s for s in scenes if s.get("title", "").startswith(prefix) and s.get("status") != "deleted"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--portal-source", type=Path)
    target.add_argument("--portal-origin", type=portal_origin)
    parser.add_argument("--portal-python", type=Path, help="Python with portal dependencies (default: checkout .venv, then this interpreter)")
    parser.add_argument("--display", help="Starting private X display (default: :93); skips occupied displays")
    parser.add_argument("--strict-display", action="store_true", help="Fail if an explicitly requested --display is occupied")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--format", choices=tuple(FORMATS), default="sog")
    parser.add_argument("--report", type=Path, default=Path("gallery-sync-e2e.md"))
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--timeout", type=float, default=300, help="Seconds per asynchronous operation")
    parser.add_argument("--auth-timeout", type=float, default=600)
    args = parser.parse_args(argv)
    args.display_explicit = args.display is not None
    args.display = args.display or ":93"
    if not re.fullmatch(r":\d+", args.display):
        parser.error("--display must be :N")
    if not 0 < args.timeout < float("inf") or not 0 < args.auth_timeout < float("inf"):
        parser.error("timeouts must be finite and positive")
    if not args.portal_origin and args.portal_source is None:
        args.portal_source = Path.home() / "projects" / "lichtfeld-portal"
    return args


def format_extension(value):
    return FORMATS[value]


def scan_logs(text, allowlist=LOG_ALLOWLIST):
    return [line for line in text.splitlines() if LOG_ERRORS.search(line)
            and not any(pattern.search(line) for pattern in allowlist)]


def mcp_result(payload):
    if not isinstance(payload, dict):
        raise ValueError("MCP response must be an object")
    if "error" in payload:
        raise RuntimeError(f"MCP error: {payload['error']}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("MCP response has no object result")
    if result.get("isError"):
        raise RuntimeError(f"MCP tool failed: {result}")
    return result


def editor_output(result):
    structured = result.get("structuredContent")
    if not isinstance(structured, dict) or structured.get("success") is not True:
        raise RuntimeError(f"Editor failed: {structured or result}")
    output = structured.get("output", {}).get("text", "")
    if not isinstance(output, str):
        raise ValueError("Editor output.text must be a string")
    if structured.get("running") or structured.get("timed_out"):
        raise RuntimeError("Editor did not finish before the deadline")
    if structured.get("output", {}).get("truncated"):
        raise RuntimeError("Editor output was truncated; inspect editor diagnostics")
    if re.search(r"Traceback|SyntaxError:|IndentationError:|TabError:", output):
        raise RuntimeError(output)
    return output


def split_marked_output(output):
    # TerminalWidget.getAllText inserts newlines at screen row boundaries, even
    # inside strings or sentinels. Match across those breaks while retaining the
    # original diagnostic text. The emitter escapes spaces to survive row rstrip.
    def pattern(marker):
        return r"[\r\n]*".join(re.escape(c) for c in marker)
    starts = list(re.finditer(pattern(MARKER), output))
    ends = list(re.finditer(pattern(END_MARKER), output))
    if len(starts) != 1 or len(ends) != 1 or starts[0].end() > ends[0].start():
        raise ValueError("Expected exactly one complete E2E sentinel pair; output may be truncated")
    start, end = starts[0], ends[0]
    payload = output[start.end():end.start()].replace("\r", "").replace("\n", "")
    return payload, output[:start.start()] + output[end.end():]


def marked_json(output):
    return json.loads(split_marked_output(output)[0])


def result_code(expression):
    # Escape angle brackets so values cannot impersonate framing markers.
    return f"""import json as _lfs_e2e_json
_lfs_e2e_payload = _lfs_e2e_json.dumps({expression}, ensure_ascii=True, separators=(',', ':')).replace(' ', r'\\u0020').replace('<', r'\\u003c').replace('>', r'\\u003e')
if len(_lfs_e2e_payload) > {JSON_PAYLOAD_LIMIT}:
    raise ValueError('E2E JSON payload exceeds {JSON_PAYLOAD_LIMIT} bytes; return only fields needed by the assertion')
print({MARKER!r} + _lfs_e2e_payload + {END_MARKER!r})
"""


def png_data(result):
    for item in result.get("content", []):
        if item.get("type") == "image" and item.get("mimeType") == "image/png":
            data = base64.b64decode(item["data"], validate=True)
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Capture is not PNG")
            return data
    raise ValueError("Capture has no PNG image")


def manifest_files(index, upload_format):
    nodes = index.get("manifest", {}).get("nodes", [])
    if not nodes:
        raise AssertionError("Portal bundle manifest contains no nodes")
    files = [node.get("file", "") for node in nodes]
    suffix = format_extension(upload_format)
    if not all(isinstance(name, str) and name.endswith(suffix) for name in files):
        raise AssertionError(f"Expected every manifest node to end in {suffix}: {files}")
    return files


def md_cell(value):
    return str(value).replace("|", "&#124;").replace("\n", "<br>").replace("\r", "")


def render_report(steps, command, artifacts, notes=(), logs=()):
    lines = ["# Gallery sync E2E", "", "Command:", "", "```sh", command, "```", "",
             f"Artifacts: {artifacts}", "", "| Step | Result | Seconds | Evidence / failure | Screenshot |",
             "| --- | --- | ---: | --- | --- |"]
    for step in steps:
        screenshot = step.get("screenshot")
        link = f"[PNG](<{screenshot}>)" if screenshot else "unavailable"
        lines.append("| " + " | ".join((md_cell(step["name"]), step["status"],
                     f'{step["seconds"]:.2f}', md_cell(step.get("detail", "")), link)) + " |")
    if notes:
        lines += ["", *[f"- {note}" for note in notes]]
    if logs:
        lines += ["", "Last 30 relevant log lines:", "", "<pre>", escape("\n".join(list(logs)[-30:])), "</pre>"]
    return "\n".join(lines) + "\n"


def render_junit(steps):
    suite = ET.Element("testsuite", name="gallery-sync-e2e", tests=str(len(steps)),
                       failures=str(sum(s["status"] == "FAIL" for s in steps)),
                       skipped=str(sum(s["status"] == "SKIP" for s in steps)),
                       time=f'{sum(s["seconds"] for s in steps):.3f}')
    for step in steps:
        case = ET.SubElement(suite, "testcase", name=step["name"], classname="gallery_sync", time=f'{step["seconds"]:.3f}')
        if step["status"] == "FAIL":
            ET.SubElement(case, "failure", message=step.get("detail", "")).text = step.get("detail", "")
        elif step["status"] == "SKIP":
            ET.SubElement(case, "skipped").text = step.get("detail", "")
        if step.get("screenshot"):
            ET.SubElement(case, "system-out").text = step["screenshot"]
    return ET.tostring(suite, encoding="unicode") + "\n"


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class MCP:
    def __init__(self, port, diagnostics=None):
        self.url = f"http://127.0.0.1:{port}/mcp"
        self.sequence = 0
        self.session = None
        self.diagnostics = diagnostics or (lambda text: None)

    def call(self, method, params=None, timeout=30):
        self.sequence += 1
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(self.url, data=json.dumps(dict(jsonrpc="2.0", id=self.sequence,
                                     method=method, params=params or {})).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            self.session = response.headers.get("Mcp-Session-Id", self.session)
            return mcp_result(json.load(response))

    def tool(self, name, **arguments):
        return self.call("tools/call", {"name": name, "arguments": arguments}, timeout=75)

    def execute(self, code):
        result = self.tool("editor_run", code=code, timeout_ms=60000, show_console=False,
                           output_max_chars=EDITOR_OUTPUT_LIMIT, output_tail=True)
        output = (result.get("structuredContent") or {}).get("output", {}).get("text", "")
        if isinstance(output, str):
            try:
                _, diagnostics = split_marked_output(output)
            except ValueError:
                diagnostics = output
            if diagnostics.strip():
                self.diagnostics(diagnostics)
        return marked_json(editor_output(result))

    def rpc(self, code):
        return self.execute(code + "\n" + result_code("None"))

    def value(self, expression):
        return self.execute(result_code(expression))


SEED = '''
import hashlib
from datetime import timedelta
from django.contrib.auth import get_user_model
from django.utils import timezone
from portal.models import DesktopSession
from gallery.models import Gallery
now = timezone.now()
owner = get_user_model().objects.create_user(email='native-sync@example.com', payment_status='paid',
    email_verified_at=now, access_activated_at=now-timedelta(days=1))
DesktopSession.objects.create(scope='desktop.basic gallery.sync', user=owner, device_name='E2E Studio',
    platform='Linux', client_version='test', access_token_hash=hashlib.sha256(b'local-test-access').hexdigest(),
    access_expires_at=now+timedelta(hours=12), refresh_token_hash=hashlib.sha256(b'local-test-refresh').hexdigest(),
    refresh_expires_at=now+timedelta(days=1))
Gallery.objects.create(owner=owner)
'''


class Run:
    def __init__(self, args, command):
        self.args, self.command = args, command
        self.root = Path(tempfile.mkdtemp(prefix="lfs-gallery-e2e-"))
        self.home = self.root / "home"
        self.home.mkdir()
        self.report = args.report.resolve()
        self.artifacts = self.report.parent / (self.report.stem + "-" + uuid.uuid4().hex[:10])
        self.artifacts.mkdir(parents=True)
        self.prefix = "E2E validate " + uuid.uuid4().hex + " "
        self.steps, self.notes, self.handles, self.processes = [], [], [], []
        self.app = self.portal = self.worker = None
        self.mcp = MCP(free_port(), self.record_diagnostics)
        self.origin = args.portal_origin
        self.scene_id = self.asset_id = self.pulled_id = None
        self.jobs = set()
        self.portal_env = None
        self.notes.extend([f"Temporary profile: {self.home}; keep={args.keep}", f"Run title prefix: `{self.prefix}`"])
        self.notes.append(f"Editor diagnostics: [full log](<{self.artifacts / 'editor-diagnostics.log'}>)")
        self.save()

    def record_diagnostics(self, text):
        with (self.artifacts / "editor-diagnostics.log").open("a") as log:
            log.write(text.rstrip() + "\n")

    def save(self):
        self.report.parent.mkdir(parents=True, exist_ok=True)
        relevant = []
        for path in sorted(self.artifacts.glob("*.log")):
            text = path.read_text(errors="replace")
            relevant.extend(f"{path.name}: {line}" for line in text.splitlines()
                            if path.name == "editor-diagnostics.log" or LOG_ERRORS.search(line)
                            or re.search(r"gallery|portal|failed|exception", line, re.I))
        self.report.write_text(render_report(self.steps, self.command, self.artifacts, self.notes, relevant))
        if self.args.junit:
            self.args.junit.parent.mkdir(parents=True, exist_ok=True)
            self.args.junit.write_text(render_junit(self.steps))

    @contextmanager
    def step(self, name):
        row = dict(name=name, status="RUNNING", seconds=0., detail="")
        self.steps.append(row)
        self.save()
        started = time.monotonic()
        print(f"STEP {name}", flush=True)
        try:
            yield row
            row["status"] = "PASS"
        except BaseException as exc:
            row.update(status="FAIL", detail=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if self.app and self.app.poll() is None:
                try:
                    path = self.artifacts / f"{len(self.steps):02d}-{re.sub('[^a-z0-9]+', '-', name.lower()).strip('-')}.png"
                    path.write_bytes(png_data(self.mcp.tool("render_capture_window")))
                    row["screenshot"] = str(path)
                except Exception as exc:
                    row["detail"] += f"; screenshot failed: {exc}"
                    if row["status"] == "PASS":
                        row["status"] = "FAIL"
            row["seconds"] = time.monotonic() - started
            self.save()
        if row["status"] == "FAIL":
            raise AssertionError(row["detail"])

    def spawn(self, command, env, name, cwd=None):
        handle = (self.artifacts / name).open("w")
        self.handles.append(handle)
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=handle, stderr=handle)
        self.processes.append(proc)
        return proc

    @staticmethod
    def stop(proc, crash=False):
        if proc and proc.poll() is None:
            (proc.kill if crash else proc.terminate)()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    def manage(self, *args):
        result = subprocess.run([str(self.portal_python), "manage.py", *args], cwd=self.args.portal_source,
                                env=self.portal_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=self.args.timeout)
        with (self.artifacts / "portal-manage.log").open("a") as log:
            log.write(result.stdout)
        if result.returncode:
            raise RuntimeError(f"manage.py {args[0]} failed: {result.stdout[-4000:]}")
        return result.stdout

    def start_portal(self):
        source = self.args.portal_source.resolve()
        if not (source / "manage.py").is_file():
            raise FileNotFoundError(f"Portal checkout missing: {source}; pass --portal-source")
        self.args.portal_source = source
        candidate = source / ".venv/bin/python"
        self.portal_python = self.args.portal_python or (candidate if candidate.exists() else Path(sys.executable))
        self.origin = f"http://127.0.0.1:{free_port()}"
        # Explicit overrides prevent inherited production DB/storage settings from being used.
        self.portal_env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "DJANGO_DEBUG": "1",
            "DJANGO_SECRET_KEY": uuid.uuid4().hex, "DATABASE_URL": "sqlite:///" + str(self.root / "portal.sqlite3"),
            "DJANGO_ALLOWED_HOSTS": "127.0.0.1", "DJANGO_EXTERNAL_BASE_URL": self.origin,
            "GALLERY_STAFF_ONLY": "0", "GALLERY_ALLOW_LOCAL_STORAGE": "1", "GALLERY_STUDIO_BUNDLES_ENABLED": "1",
            "GALLERY_R2_BUCKET": "", "GALLERY_STORAGE_ROOT": str(self.root / "storage"),
            "DEFAULT_SIGNUP_GRANT_ENABLED": "0", "DJANGO_SECURE_SSL_REDIRECT": "0"}
        self.manage("migrate", "--noinput")
        self.manage("shell", "-c", SEED)
        self.portal = self.spawn([str(self.portal_python), "manage.py", "runserver",
                                  self.origin.removeprefix("http://"), "--noreload"], self.portal_env,
                                 "portal-server.log", source)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if self.portal.poll() is not None:
                raise RuntimeError("Owned portal exited; see portal-server.log")
            try:
                with urllib.request.urlopen(self.origin, timeout=1):
                    return
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    return
            except OSError:
                pass
            time.sleep(.2)
        raise TimeoutError("Local portal did not become ready")

    def start_display(self):
        requested = self.args.display
        number = int(requested[1:])
        while Path(f"/tmp/.X11-unix/X{number}").exists() or Path(f"/tmp/.X{number}-lock").exists():
            if self.args.display_explicit and self.args.strict_display:
                raise RuntimeError(f"Display {requested} is occupied (--strict-display)")
            number += 1
        self.args.display = f":{number}"
        self.notes.append(f"Private display: {self.args.display} (requested {requested})")
        proc = self.spawn(["Xvfb", self.args.display, "-screen", "0", "1600x1000x24", "-nolisten", "tcp"],
                          os.environ.copy(), "xvfb.log")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("Owned Xvfb exited")
            if Path(f"/tmp/.X11-unix/X{number}").exists():
                return
            time.sleep(.1)
        raise TimeoutError("Xvfb did not become ready")

    def start_app(self):
        # Resilience's launch/rpc helpers are closures inside main and cannot be imported.
        # Reuse the bridge's public environment helper without activating its bridge/main.
        spec = importlib.util.spec_from_file_location("gallery_e2e_bridge", REPO / "scripts/lichtfeld_mcp_bridge.py")
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        binary = self.args.build_dir.resolve() / "LichtFeld-Studio"
        if not binary.is_file():
            raise FileNotFoundError(binary)
        env = bridge.launch_environment([str(binary)])
        # Do not inherit user catalog overrides into this disposable profile.
        for key in tuple(env):
            if key.startswith(("LFS_RESOLVED_", "LFS_ASSET_MANAGER_")):
                env.pop(key)
        env.update(DISPLAY=self.args.display, LFS_HOME=str(self.home), LFS_PORTAL_URL=self.origin,
                   SDL_VIDEODRIVER="x11", PYTHONDONTWRITEBYTECODE="1")
        if not env.get("VK_ICD_FILENAMES"):
            icds = sorted(Path("/usr/share/vulkan/icd.d").glob("*nvidia*.json"))
            if icds:
                env["VK_ICD_FILENAMES"] = str(icds[0])
        self.mcp.session = None
        self.app = self.spawn([str(binary), "--mcp-port", str(urllib.parse.urlsplit(self.mcp.url).port),
                               "--no-splash", "--log-level", "debug"], env,
                              f"studio-{sum(p.name.startswith('studio-') for p in self.artifacts.glob('*.log'))}.log", REPO)
        deadline = time.monotonic() + 90
        while True:
            if self.app.poll() is not None:
                raise RuntimeError("Owned Studio exited; inspect Studio log")
            try:
                self.mcp.call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "gallery-sync-e2e", "version": "1"}}, timeout=1)
                break
            except (OSError, RuntimeError, ValueError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Studio MCP did not initialize")
                time.sleep(.2)
        tools = self.mcp.call("tools/list")
        self.mcp.call("resources/list")
        for resource in RESOURCES:
            self.mcp.call("resources/read", {"uri": "lichtfeld://" + resource})
        names = {tool["name"] for tool in tools.get("tools", [])}
        required = {"editor_run", "render_capture_window", "ui_modal_get", "ui_modal_press"}
        if not required <= names:
            raise RuntimeError(f"Required MCP tools missing: {required - names}")
        (self.artifacts / "mcp-tools.json").write_text(json.dumps(tools, indent=2))
        self.mcp.rpc("import lichtfeld as lf\nimport json, time\nfrom pathlib import Path")

    def panel(self):
        self.mcp.rpc("""lf.ui.set_panel_enabled('lfs.asset_manager', True)
p = lf.ui.get_panel_object('lfs.asset_manager')
p = p._load() if hasattr(p, '_load') else p
""")

    def wait(self, expression, description, timeout=None, process_uploads=False):
        deadline = time.monotonic() + (timeout or self.args.timeout)
        last, last_notice = None, 0.
        while time.monotonic() < deadline:
            last = self.mcp.value(expression)
            if last:
                return last
            if process_uploads and self.portal:
                if self.worker and self.worker.poll() not in (None, 0):
                    raise RuntimeError("Local gallery processor failed; inspect processor log")
                if not self.worker or self.worker.poll() is not None:
                    self.worker = self.spawn([str(self.portal_python), "manage.py", "shell", "-c",
                                              "from gallery import processing; processing.run_one()"],
                                             self.portal_env, f"processor-{len(self.processes)}.log", self.args.portal_source)
            if time.monotonic() - last_notice > 20:
                print(f"Waiting: {description}", flush=True)
                last_notice = time.monotonic()
            time.sleep(.3)
        state = self.mcp.value(f"dict(notice=p._gallery_notice, phase=p._controller().phase(), jobs=[{{k: j.get(k) for k in {JOB_FIELDS!r}}} for j in p._controller().service.snapshot()['jobs']])")
        raise TimeoutError(f"{description}: {state}")

    def wait_transfer_rows(self, expected):
        deadline = time.monotonic() + min(5., self.args.timeout)
        while True:
            state = self.mcp.value(f"""(lambda elements: dict(
                count=len(elements),
                first_row_inner_rml=elements[0].get_inner_rml() if elements else None
            ))(lf.ui.rml.get_document('lfs.gallery_transfer').query_selector_all({TRANSFER_ROW_SELECTOR!r}))""")
            if state['count'] == expected:
                return state
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Rendered transfer rows: expected {expected}, actual DOM row count {state['count']}; "
                                   f"first row inner RML: {state['first_row_inner_rml']!r}")
            time.sleep(.3)

    def sign_in(self, restart=False):
        if self.portal:
            self.mcp.rpc(f"""from lfs_plugins.portal_account import PortalAccountService, _Credentials
from lfs_plugins.gallery_sync import GallerySync
import lfs_plugins.gallery_sync as sync_module
from lfs_plugins.asset_index import resolve_asset_manager_storage_path
account = PortalAccountService(base_url={self.origin!r}, credentials_path=Path({str(self.home)!r})/'credentials.json')
creds = _Credentials(portal_origin={self.origin!r}, access_token='local-test-access', access_expires_at=time.time()+3600,
    refresh_token='local-test-refresh', refresh_expires_at=time.time()+86400,
    email='native-sync@example.com', connected_since='e2e-session')
account._set_current_credentials(creds)
account._apply_credentials_state(creds)
new = GallerySync(account, resolve_asset_manager_storage_path()/'gallery')
""")
        else:
            self.mcp.rpc("""from lfs_plugins.portal_account import get_portal_account_service
from lfs_plugins.gallery_sync import GallerySync
import lfs_plugins.gallery_sync as sync_module
from lfs_plugins.asset_index import resolve_asset_manager_storage_path
account = get_portal_account_service()
""")
            if not self.mcp.value("account.snapshot().signed_in"):
                self.mcp.rpc("account.start_device_flow()")
                self.wait("bool(account.snapshot().user_code)", "Device code", timeout=60)
                approval = self.mcp.value("dict(code=account.snapshot().user_code, url=account.snapshot().verification_uri_complete)")
                print(f"Approve Studio sign-in in your browser: {approval['url']}\nCode: {approval['code']}", flush=True)
                self.wait("account.snapshot().signed_in", "Browser approval", timeout=self.args.auth_timeout)
            self.mcp.rpc("new = GallerySync(account, resolve_asset_manager_storage_path()/'gallery')")
        if restart:
            # Constructor-loaded journal, before refresh establishes account ownership/scenes.
            # snapshot()['links'] intentionally stays empty until identity is verified by refresh.
            self.mcp.rpc(f"""assert not new.busy and not new.scenes
assert any(bucket.get('links', {{}}).get({self.pulled_id!r}, {{}}).get('sceneId') == {self.scene_id!r}
           for bucket in new._data['accounts'].values()), 'Link missing from constructor-loaded journal'
assert {{j['id'] for b in new._data['accounts'].values() for j in b['jobs']}} >= {self.jobs!r}
""")
        self.mcp.rpc("sync_module._service = new\np._controller().service = new\np._controller().refresh()")
        self.wait("p._gallery_state.get('connected') and not new.busy", "Signed-in gallery refresh")

    def facts(self, identifier, state, label):
        self.wait(f"p._gallery_facts(p._asset_dict({identifier!r}) or {{}})['state'] == {state!r}", label)
        self.mcp.rpc(f"assert p._gallery_badge(p._asset_dict({identifier!r}))['gallery_label'] == {label!r}")

    def job(self, previous, kind):
        expression = f"[j for j in p._controller().service.snapshot()['jobs'] if j['id'] not in {previous!r}]"
        self.wait(f"bool({expression})", f"New {kind} job", process_uploads=True)
        deadline = time.monotonic() + self.args.timeout
        while time.monotonic() < deadline:
            jobs = self.mcp.value(f"[{{k: j.get(k) for k in {JOB_FIELDS!r}}} for j in {expression}]")
            if len(jobs) != 1:
                raise AssertionError(f"Expected exactly one new job, got {jobs}")
            job = jobs[0]
            if job["kind"] != kind:
                raise AssertionError(f"Expected {kind}: {job}")
            if job["status"] in ("error", "conflict", "canceled", "paused"):
                raise AssertionError(f"Transfer failed: {job}")
            if job["status"] == "completed":
                if job.get("total", 0) <= 0 or job.get("completed", 0) != job["total"] or job.get("serverProcessing"):
                    raise AssertionError(f"Invalid completed transfer: {job}")
                self.jobs.add(job["id"])
                return job
            if self.portal and (not self.worker or self.worker.poll() is not None):
                if self.worker and self.worker.returncode:
                    raise RuntimeError("Portal processor failed")
                self.worker = self.spawn([str(self.portal_python), "manage.py", "shell", "-c",
                                          "from gallery import processing; processing.run_one()"],
                                         self.portal_env, f"processor-{len(self.processes)}.log", self.args.portal_source)
            print(f"{kind}: {job['status']} {job.get('completed')}/{job.get('total')} processing={job.get('serverProcessing', False)}", flush=True)
            time.sleep(1)
        raise TimeoutError(f"Transfer deadline: {job}")

    def confirm(self):
        modal = self.mcp.tool("ui_modal_get").get("structuredContent", {})
        if not modal.get("open"):
            raise AssertionError(f"Expected confirmation modal: {modal}")
        self.mcp.tool("ui_modal_press", label="Continue")

    def verify_remote(self):
        self.mcp.rpc("from lfs_plugins.portal_gallery import PortalGalleryClient\nclient = PortalGalleryClient(account)")
        self.mcp.rpc(f"remote_payload = client.scene({self.scene_id!r})\nremote_scene = remote_payload.get('scene', remote_payload)")
        scene = self.mcp.value("{k: remote_scene.get(k) for k in ('id', 'status', 'title', 'revision', 'viewerSettings')}")
        if scene.get("id") != self.scene_id or scene.get("status") != "ready" or not scene.get("title", "").startswith(self.prefix):
            raise AssertionError(f"GET scene did not return this run's ready scene: {scene}")
        self.remote_scene = scene
        if self.portal:
            output = self.manage("shell", "-c", "from gallery.models import Scene; import json; "
                f"s=Scene.objects.get(pk={self.scene_id!r}, deleted_at__isnull=True, ready=True); "
                "\n" + result_code("{'manifest': {'nodes': [{'file': n.get('file')} for n in s.bundle_index.get('manifest', {}).get('nodes', [])]}}"))
            files = manifest_files(marked_json(output), self.args.format)
            return "GET scene ready; manifest files: " + ", ".join(files)
        return "Authenticated GET /api/gallery/v1/splats/{id}: ready, identity and title verified"

    def cleanup_remote(self):
        if self.portal or not self.scene_id and not self.asset_id:
            return
        with self.step("Cleanup real portal scenes") as row:
            # Stop in-flight producers before cleanup. Restart uses the same isolated credentials.
            self.stop(self.app, crash=True)
            self.start_app()
            self.mcp.rpc("""from lfs_plugins.portal_account import get_portal_account_service
from lfs_plugins.portal_gallery import PortalGalleryClient
account = get_portal_account_service()
assert account.snapshot().signed_in, 'Cleanup needs the retained profile to sign in again'
client = PortalGalleryClient(account)
from lfs_plugins.gallery_sync import GallerySync
from lfs_plugins.asset_index import resolve_asset_manager_storage_path
cleanup_journal = GallerySync(account, resolve_asset_manager_storage_path()/'gallery')
""")
            transfers = self.mcp.value(f"[{{'id': t['id'], 'title': t['title']}} for t in client._request('GET', '/transfers')['transfers'] if t.get('title', '').startswith({self.prefix!r})]")
            # The server's recent-transfers list is capped. Include our durable checkpoints too.
            upload_ids = set(self.mcp.value(f"[(j.get('checkpoint') or {{}})['uploadId'] for b in cleanup_journal._data['accounts'].values() for j in b['jobs'] if j.get('metadata', {{}}).get('title', '').startswith({self.prefix!r}) and (j.get('checkpoint') or {{}}).get('uploadId')]"))
            upload_ids.update(t['id'] for t in transfers if t.get('title', '').startswith(self.prefix))
            failures = []
            for upload_id in sorted(upload_ids):
                try:
                    status = self.mcp.value(f"client._request('GET', {'/splats/uploads/' + upload_id!r})['status']")
                    if status not in ('completed', 'canceled'):
                        self.mcp.rpc(f"client.cancel_upload({upload_id!r})")
                except Exception as exc:
                    failures.append(f"upload {upload_id}: {exc}")
            # Paginated listing catches scenes created before a failed upload returned its id.
            scene_expression = f"[{{k: s.get(k) for k in ('id', 'title', 'status', 'revision')}} for s in client.list_scenes() if s.get('title', '').startswith({self.prefix!r})]"
            scenes = self.mcp.value(scene_expression)
            owned = owned_scenes(scenes, self.prefix)
            for scene in owned:
                try:
                    self.mcp.rpc(f"client.delete({scene['id']!r}, {scene['revision']!r})")
                except Exception as exc:
                    failures.append(f"{scene['id']}: {exc}")
            remaining = [s['id'] for s in owned_scenes(self.mcp.value(scene_expression), self.prefix)]
            if failures or remaining:
                raise AssertionError(f"Could not clean scenes: {failures}; remaining={remaining}")
            row["detail"] = f"Removed {len(owned)} scenes owned by this run; none remain"

    def workflow(self):
        with self.step("Launch private portal and Studio"):
            if not self.origin:
                self.start_portal()
            projects = self.home / "projects"
            projects.mkdir()
            for name in ("ply", "sog", "ssog", "multi"):
                shutil.copyfile(REPO / f"tests/data/portable-{name}.licht", projects / f"portable-{name}.licht")
            self.start_display()
            self.start_app()
            self.panel()
        with self.step("Signed out sidebar"):
            self.wait("lf.ui.rml.get_document('lfs.asset_manager') is not None", "Asset Manager document")
            self.wait("lf.ui.rml.get_document('lfs.asset_manager').query_selector('.gallery-checked') is not None and 'Sign in to see your gallery' in lf.ui.rml.get_document('lfs.asset_manager').query_selector('.gallery-checked').get_inner_rml()", "Rendered signed-out hint")
            self.mcp.rpc("""assert not p._controller().service.account.snapshot().signed_in
assert p._gallery_checked_label() == 'Sign in to see your gallery'
doc = lf.ui.rml.get_document('lfs.asset_manager')
for scope in ('__gallery__', '__gallery_attention__', '__gallery_transfers__'):
    assert doc.query_selector('[data-folder-id="' + scope + '"]') is not None, scope
assert 'Sign in to see your gallery' in doc.query_selector('.gallery-checked').get_inner_rml()
""")
        with self.step("Sign in"):
            self.sign_in()
        with self.step("Fixture catalog: Not published"):
            self.mcp.rpc("p.refresh_catalog()")
            self.wait("{Path(a['path']).stem for a in p._asset_index_assets().values()} >= {'portable-ply', 'portable-sog', 'portable-ssog', 'portable-multi'}", "Fixture catalog")
            self.mcp.rpc("""fixtures = [a for a in p._asset_index_assets().values() if Path(a['path']).stem.startswith('portable-')]
assert len(fixtures) == 4
for a in fixtures:
    assert p._gallery_facts(a)['state'] == 'unlinked', a
    assert p._gallery_badge(a)['gallery_label'] == 'Not published', a
""")
            self.asset_id = self.mcp.value("next(a['id'] for a in fixtures if Path(a['path']).stem == 'portable-multi')")
        with self.step("Publish closed portable-multi") as row:
            self.mcp.rpc(f"""assert not lf.project_has_path(), 'Publish must start with a closed project'
p._select_asset_id({self.asset_id!r})
p._gallery_expanded = True
p._gallery_command('publish')
assert p._gallery_review
p._gallery_upload_format = {self.args.format!r}
p._gallery_title = {self.prefix + 'portable-multi'!r}
p._gallery_description = 'Automated disposable gallery sync validation'
p._gallery_visibility = 'private'
p._gallery_command('publish')
""")
            self.job(set(self.jobs), "upload")
            self.mcp.rpc("assert not lf.project_has_path(), 'Closed publication opened a document'")
            self.wait(f"{self.asset_id!r} in p._gallery_state.get('links', {{}})", "Published link")
            self.scene_id = self.mcp.value(f"p._gallery_state['links'][{self.asset_id!r}]['sceneId']")
            row["detail"] = self.verify_remote()
            self.facts(self.asset_id, "equal", "Up to date")
        with self.step("Save change: Saved since last publish"):
            path = self.mcp.value(f"p._asset_dict({self.asset_id!r})['path']")
            self.mcp.rpc(f"if lf.project_poll_write().get('path') != {path!r}:\n    lf.project_open({path!r}, discard_changes=True, keep_asset_manager_open=True)")
            self.wait(f"lf.project_poll_write().get('path') == {path!r} and not lf.ui.get_import_state().get('active')", "Published project opened for edit")
            self.mcp.rpc("lf.get_render_settings().color_exposure += 0.1\nassert lf.project_save(wait=True)\np.refresh_catalog()")
            self.facts(self.asset_id, "local", "Saved since last publish")
        with self.step("Update: Up to date") as row:
            before_revision = self.remote_scene['revision']
            exposure = self.mcp.value("lf.get_render_settings().color_exposure")
            self.mcp.rpc(f"p._select_asset_id({self.asset_id!r})\nassert p._gallery_facts(p._asset_dict({self.asset_id!r}))['action'] == 'update'\np._gallery_command('primary')")
            new_jobs = f"[j for j in p._controller().service.snapshot()['jobs'] if j['id'] not in {self.jobs!r}]"
            self.wait(f"bool({new_jobs}) or (not new.busy and p._gallery_facts(p._asset_dict({self.asset_id!r}))['state'] == 'equal')", "Update upload or metadata PATCH", process_uploads=True)
            if self.mcp.value(f"bool({new_jobs})"):
                self.job(set(self.jobs), "upload")
                mode = "Replacement upload completed. "
            else:
                mode = "Exposure-only metadata PATCH completed (no transfer job). "
            self.facts(self.asset_id, "equal", "Up to date")
            row["detail"] = mode + self.verify_remote()
            if self.remote_scene['revision'] == before_revision:
                raise AssertionError('Update did not change the remote revision')
            remote_exposure = self.remote_scene.get('viewerSettings', {}).get('exposure')
            if remote_exposure is None or abs(remote_exposure - exposure) > 1e-5:
                raise AssertionError(f'Saved exposure {exposure} did not reach the portal: {remote_exposure}')
        with self.step("Unlink: remote-only card"):
            self.mcp.rpc("p._gallery_command('unlink')")
            self.confirm()
            self.wait(f"{self.asset_id!r} not in p._gallery_state['links'] and 'remote:' + {self.scene_id!r} in p._gallery_remote_assets()", "Remote-only card")
            self.mcp.rpc(f"assert p._gallery_facts(p._gallery_remote_assets()['remote:' + {self.scene_id!r}])['relationship'] == 'remote_only'")
        with self.step("Pull: registered linked project, document preserved"):
            self.mcp.rpc(f"""before_assets = set(p._asset_index_assets())
before_document = (lf.project_poll_write().get('path'), [(n.uuid, n.name) for n in lf.get_scene().get_nodes()], lf.get_render_settings().color_exposure)
p._select_asset_id('remote:' + {self.scene_id!r})
p._gallery_command('pull')
assert p._gallery_pull_review
p._gallery_command('pull')
""")
            self.job(set(self.jobs), "download")
            self.wait("bool(set(p._asset_index_assets()) - before_assets)", "Pull registration")
            self.pulled_id = self.mcp.value("next(iter(set(p._asset_index_assets()) - before_assets))")
            self.wait(f"{self.pulled_id!r} in p._gallery_state['links']", "Pull link")
            self.mcp.rpc(f"""assert len(set(p._asset_index_assets()) - before_assets) == 1
assert Path(p._asset_dict({self.pulled_id!r})['path']).is_file()
assert p._gallery_state['links'][{self.pulled_id!r}]['sceneId'] == {self.scene_id!r}
assert lf.ui.is_panel_enabled('lfs.asset_manager')
assert before_document == (lf.project_poll_write().get('path'), [(n.uuid, n.name) for n in lf.get_scene().get_nodes()], lf.get_render_settings().color_exposure), 'Pull changed open document'
""")
        with self.step("Restart: journal links restored before refresh") as row:
            self.stop(self.app, crash=True)
            self.start_app()
            self.panel()
            self.sign_in(restart=True)
            self.mcp.rpc("p.refresh_catalog()")
            self.wait(f"p._asset_dict({self.pulled_id!r}) is not None", "Restored local registration")
            row["detail"] = "Fresh GallerySync constructor restored link and all job IDs from journal before its first refresh; authenticated refresh then restored panel links"
            self.wait(f"p._gallery_state['links'].get({self.pulled_id!r}, {{}}).get('sceneId') == {self.scene_id!r}", "Restored panel link")
        with self.step("Remove: Removed on portal, Needs attention 1"):
            self.mcp.rpc(f"p._select_asset_id({self.pulled_id!r})\np._gallery_command('remove')")
            self.confirm()
            self.facts(self.pulled_id, "remote_deleted", "Removed on portal")
            self.wait("lf.ui.rml.get_document('lfs.asset_manager').query_selector('[data-folder-id=\"__gallery_attention__\"] .asset-row-count').get_inner_rml().strip() == '1'", "Rendered Needs attention count")
            self.mcp.rpc("""assert len(p._gallery_rows(attention=True)) == 1
assert p._gallery_counts()['linked'] == 0
assert lf.ui.rml.get_document('lfs.asset_manager').query_selector('[data-folder-id="__gallery_attention__"] .asset-row-count').get_inner_rml().strip() == '1'
""")
        with self.step("Transfers tray: all jobs and human sizes"):
            self.mcp.rpc("if not lf.ui.is_panel_enabled('lfs.gallery_transfer'):\n    p._gallery_command('transfers')")
            self.wait("lf.ui.is_panel_enabled('lfs.gallery_transfer') and lf.ui.rml.get_document('lfs.gallery_transfer') is not None", "Transfers tray")
            self.wait_transfer_rows(len(self.jobs))
            self.mcp.rpc(f"""from lfs_plugins.gallery_transfer_panel import transfer_rows
import re
tray = lf.ui.get_panel_object('lfs.gallery_transfer')
tray = tray._load() if hasattr(tray, '_load') else tray
rows = transfer_rows(tray._state, tray._history_limit)
assert {{r['id'] for r in rows}} == {self.jobs!r}, rows
for row in rows:
    assert re.search(r'\\d+(?:\\.\\d+)?\\s*(?:B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)\\b', row['bytes']), row
doc = lf.ui.rml.get_document('lfs.gallery_transfer')
visible_rows = doc.query_selector_all({TRANSFER_ROW_SELECTOR!r})
assert len(visible_rows) == len(rows), f'Tray DOM: expected {{len(rows)}}, actual DOM row count {{len(visible_rows)}}; first row inner RML: {{visible_rows[0].get_inner_rml() if visible_rows else None!r}}'
for element, row in zip(visible_rows, rows):
    assert row['bytes'] in element.get_inner_rml(), row
""")

    def check_logs(self):
        with self.step("Studio and portal log scan") as row:
            failures = []
            for path in sorted(self.artifacts.glob("*.log")):
                failures.extend(f"{path.name}: {line}" for line in scan_logs(path.read_text(errors="replace")))
            if failures:
                raise AssertionError("\n".join(failures[-30:]))
            row["detail"] = "Zero non-allowlisted Traceback, [error], property syntax or missing localization lines"

    def close(self):
        for proc in reversed(self.processes):
            self.stop(proc)
        for handle in self.handles:
            handle.close()
        # Include errors written during screenshots or process shutdown after the live scan.
        late = [f"{path.name}: {line}" for path in sorted(self.artifacts.glob('*.log'))
                for line in scan_logs(path.read_text(errors='replace'))]
        log_step = next((s for s in self.steps if s['name'] == 'Studio and portal log scan'), None)
        if late and log_step:
            log_step.update(status='FAIL', detail='\n'.join(late[-30:]))
        if self.args.keep:
            self.notes.append(f"Kept temporary directory: {self.root}")
        else:
            shutil.rmtree(self.root)
        self.save()


def main(argv=None):
    args = parse_args(argv)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])])
    try:
        run = Run(args, command)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"FAILED during run setup: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('Termination requested')
    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    failed = False

    def failure(phase, exc):
        nonlocal failed
        failed = True
        detail = f"{type(exc).__name__}: {exc}"
        print(f"FAILED ({phase}): {detail}", file=sys.stderr, flush=True)
        if not run.steps or run.steps[-1]['status'] != 'FAIL':
            run.steps.append(dict(name=phase, status='FAIL', seconds=0., detail=detail))
        else:
            run.notes.append(f"{phase}: {detail}")

    try:
        run.workflow()
    except (Exception, KeyboardInterrupt) as exc:
        failure("Workflow", exc)
    finally:
        try:
            run.cleanup_remote()
        except (Exception, KeyboardInterrupt) as exc:
            failure("Remote cleanup", exc)
            # Preserve authentication needed to finish cleanup; never print bearer credentials.
            run.args.keep = True
            message = f"Could not finish remote cleanup for prefix {run.prefix!r}: {exc}. Retained profile: {run.home}"
            run.notes.append(message)
            print(message, file=sys.stderr, flush=True)
        try:
            run.check_logs()
        except (Exception, KeyboardInterrupt) as exc:
            failure("Log scan", exc)
        try:
            run.close()
        except (Exception, KeyboardInterrupt) as exc:
            failure("Shutdown", exc)
            try:
                run.save()
            except Exception as report_exc:
                print(f"Could not save failure report: {report_exc}", file=sys.stderr, flush=True)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    print(f"Report: {run.report}\nArtifacts: {run.artifacts}", flush=True)
    return int(failed or any(step["status"] == "FAIL" for step in run.steps))


if __name__ == "__main__":
    raise SystemExit(main())
