#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Standard-library stress helpers; launcher/MCP helpers live in Lane H's script."""
from __future__ import annotations

import hashlib
import http.client
import http.server
import json
import random
import threading
import time
import urllib.parse

SHELL_MARKER = "GALLERY_STRESS_DB_JSON:"


def shell_json(output):
    """Django shell output has no terminal wrapping or editor output limit."""
    rows = [line[len(SHELL_MARKER):] for line in output.splitlines() if line.startswith(SHELL_MARKER)]
    if len(rows) != 1:
        raise ValueError("Expected one Django shell result")
    return json.loads(rows[0])


def remote_json(mcp, expression):
    """Use H's framed MCP reader, paging large snapshots below its output limit.

    Evaluate once so a live job cannot change halfway through serialization.
    Chunk sizes also leave room for the frame emitter's JSON escaping.
    """
    mcp.rpc(f"_lfs_stress_json = json.dumps({expression}, ensure_ascii=True)")
    first = mcp.value("dict(size=len(_lfs_stress_json), data=_lfs_stress_json[:1024])")
    if not 0 <= first["size"] <= 64 * 1024 * 1024:
        raise ValueError("Stress snapshot exceeds 64 MiB")
    chunks = [first["data"]]
    for offset in range(1024, first["size"], 1024):
        chunks.append(mcp.value(f"_lfs_stress_json[{offset}:{offset+1024}]"))
    return json.loads("".join(chunks))


def scenario_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")


def assert_revision_progress(history):
    """Revision domains are opaque hashes, not ordered counters."""
    if len(history) < 2:
        raise AssertionError("Need at least two observed exchanges")
    tokens = [row["metadataRevision"] for row in history]
    if len(set(tokens)) != len(tokens) or not all(tokens):
        raise AssertionError(f"Distinct edits did not advance metadata revisions: {tokens}")
    times = [row["exchangedAt"] for row in history]
    if any(b < a for a, b in zip(times, times[1:])):
        raise AssertionError("Journal exchange timestamps went backwards")
    if len({row["sceneId"] for row in history}) != 1:
        raise AssertionError("Round trip created a different scene")


def assert_retained_parts(before, after, total, resumed_bytes):
    old = {row["number"]: (row["etag"], row["size"]) for row in before}
    new = {row["number"]: (row["etag"], row["size"]) for row in after}
    if not old or not 0 < sum(size for _, size in old.values()) < total:
        raise AssertionError("No partial upload was retained")
    if any(new.get(number) != value for number, value in old.items()):
        raise AssertionError("Resume replaced a retained part")
    if not 0 < resumed_bytes < total:
        raise AssertionError(f"Resume sent {resumed_bytes} bytes for a {total}-byte upload")


def retry_evidence(events):
    checked = []
    for index, event in enumerate(events):
        if event.get("injected") != 429:
            continue
        following = next((row for row in events[index + 1:]
                          if (row["method"], row["path"]) == (event["method"], event["path"])), None)
        if following is None:
            raise AssertionError(f"429 was never retried: {event['method']} {event['path']}")
        elapsed = following["started"] - event["finished"]
        if elapsed + .05 < event["retry_after"]:
            raise AssertionError(f"Retry-After violated: {elapsed:.3f}s < {event['retry_after']}s")
        checked.append(dict(path=event["path"], elapsed=elapsed, retry_after=event["retry_after"]))
    if not checked:
        raise AssertionError("No injected 429 was exercised")
    return checked


def safe_job(job):
    # Checkpoints can contain signed storage URLs. Preserve useful evidence only.
    result = {key: job[key] for key in ("id", "kind", "project", "status", "completed", "total",
              "message", "serverProcessing", "interrupted", "path", "sceneId") if key in job}
    result["uploadId"] = (job.get("checkpoint") or {}).get("uploadId")
    result["stagedImport"] = {key: value for key, value in job.get("stagedImport", {}).items()
                              if key in ("state", "path", "message")}
    return result


def confirmation_label(modal, cancel="Cancel"):
    labels = [button["label"] for button in (modal or {}).get("buttons", [])
              if button.get("enabled", True) and button["label"] != cancel]
    if len(labels) != 1:
        raise AssertionError(f"Expected one enabled confirmation choice: {labels}")
    return labels[0]


class FaultProxy:
    """Owned loopback reverse proxy. Streams bodies; logs no auth or query strings.

    A seeded schedule rejects one request per ten-request block, alternating
    429 and 503. Reproducing request ordering reproduces the fault schedule.
    """
    def __init__(self, upstream, seed, log, *, retry_after=2):
        parsed = urllib.parse.urlsplit(upstream)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise ValueError("Fault proxy only accepts a loopback HTTP upstream")
        self.upstream = parsed
        self.seed, self.log, self.retry_after = seed, log, retry_after
        self.faults = False
        self.upload_bps = self.download_bps = 0
        self.inflate_download = 0
        self.list_delay = 0
        self.events, self.count = [], 0
        self.lock = threading.Lock()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def do_GET(self):
                owner.forward(self)

            do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_GET

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return [dict(row) for row in self.events]

    def inject(self, method, path):
        if not self.faults or not path.startswith("/api/gallery/v1/"):
            return 0
        with self.lock:
            count = self.count
            self.count += 1
        block, slot = divmod(count, 10)
        # Slot zero in the first block ensures even short runs exercise a fault.
        selected = random.Random(scenario_seed(self.seed, str(block))).randrange(10)
        return (429 if block % 2 == 0 else 503) if slot == selected else 0

    def forward(self, handler):
        path = urllib.parse.urlsplit(handler.path).path
        event = dict(method=handler.command, path=path, started=time.monotonic(),
                     status=0, request_bytes=0, response_bytes=0, injected=0)
        with self.lock:
            self.events.append(event)
        connection = None
        sent_headers = False
        try:
            failure = self.inject(handler.command, path)
            if failure:
                event.update(status=failure, injected=failure, retry_after=self.retry_after)
                payload = json.dumps({"error": "stress_retry", "detail": "Injected local stress failure"}).encode()
                handler.send_response(failure)
                handler.send_header("Retry-After", str(self.retry_after))
                handler.send_header("Content-Length", str(len(payload)))
                handler.send_header("Content-Type", "application/json")
                handler.send_header("Connection", "close")
                handler.end_headers()
                sent_headers = True
                handler.wfile.write(payload)
                handler.close_connection = True
                return
            connection = http.client.HTTPConnection(self.upstream.hostname, self.upstream.port, timeout=120)
            connection.putrequest(handler.command, handler.path, skip_host=True, skip_accept_encoding=True)
            for key, value in handler.headers.items():
                if key.lower() not in ("host", "connection", "transfer-encoding", "accept-encoding"):
                    connection.putheader(key, value)
            connection.putheader("Host", urllib.parse.urlsplit(self.origin).netloc)
            connection.putheader("Connection", "close")
            connection.endheaders()
            remaining = int(handler.headers.get("Content-Length", 0))
            while remaining:
                chunk = handler.rfile.read(min(64 * 1024, remaining))
                if not chunk:
                    raise OSError("Client closed request body")
                if self.upload_bps and handler.command == "PUT":
                    time.sleep(len(chunk) / self.upload_bps)
                connection.send(chunk)
                remaining -= len(chunk)
                event["request_bytes"] += len(chunk)
            response = connection.getresponse()
            event["status"] = response.status
            if self.list_delay and path == "/api/gallery/v1/splats":
                time.sleep(self.list_delay)
            replacement = None
            if self.inflate_download and path.endswith("/download") and response.status == 200:
                data = json.loads(response.read())
                data["scene"]["contentLength"] += self.inflate_download
                replacement = json.dumps(data).encode()
            handler.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in ("connection", "transfer-encoding", "content-length"):
                    handler.send_header(key, value)
            if replacement is not None:
                handler.send_header("Content-Length", str(len(replacement)))
            elif response.getheader("Content-Length"):
                handler.send_header("Content-Length", response.getheader("Content-Length"))
            handler.send_header("Connection", "close")
            handler.end_headers()
            sent_headers = True
            if replacement is not None:
                handler.wfile.write(replacement)
                event["response_bytes"] += len(replacement)
            else:
                while chunk := response.read(64 * 1024):
                    if self.download_bps and "download-content" in path:
                        time.sleep(len(chunk) / self.download_bps)
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
                    event["response_bytes"] += len(chunk)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            event["error"] = type(exc).__name__
            if not sent_headers:
                event["status"] = 503
                try:
                    handler.send_response(503)
                    handler.send_header("Content-Length", "0")
                    handler.send_header("Connection", "close")
                    handler.end_headers()
                except OSError:
                    pass
        finally:
            event["finished"] = time.monotonic()
            handler.close_connection = True
            if connection:
                connection.close()
            with self.lock:
                with self.log.open("a") as stream:
                    stream.write(json.dumps(event, sort_keys=True) + "\n")

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
