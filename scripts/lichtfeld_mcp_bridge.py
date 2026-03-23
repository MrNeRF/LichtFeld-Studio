#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Start LichtFeld Studio on demand and bridge stdio MCP traffic to its HTTP endpoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback.
    fcntl = None


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENDPOINT = os.environ.get("LICHTFELD_MCP_ENDPOINT", "http://127.0.0.1:45677/mcp")
DEFAULT_START_TIMEOUT_S = float(os.environ.get("LICHTFELD_MCP_START_TIMEOUT_S", "90"))
START_POLL_INTERVAL_S = 0.5
LOG_PATH = Path(
    os.environ.get(
        "LICHTFELD_MCP_BRIDGE_LOG",
        str(Path.home() / ".codex" / "log" / "lichtfeld-mcp-bridge.log"),
    )
)
LOCK_PATH = Path(
    os.environ.get(
        "LICHTFELD_MCP_BRIDGE_LOCK",
        str(Path.home() / ".codex" / "log" / "lichtfeld-mcp-bridge.lock"),
    )
)


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def post_json(payload: Any, timeout_s: float = 5.0) -> Any:
    request = urllib.request.Request(
        DEFAULT_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def endpoint_ready() -> bool:
    try:
        response = post_json({"jsonrpc": "2.0", "id": 0, "method": "ping"}, timeout_s=1.0)
    except (TimeoutError, OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return False

    return isinstance(response, dict) and response.get("error") is None


def executable_candidates() -> list[Path]:
    env_executable = os.environ.get("LICHTFELD_EXECUTABLE")
    candidates: list[Path] = []
    if env_executable:
        candidates.append(Path(env_executable).expanduser())

    if os.name == "nt":
        names = ["LichtFeld-Studio.exe"]
    else:
        names = ["LichtFeld-Studio", "run_lichtfeld.sh"]

    search_roots = [
        REPO_ROOT / "build",
        REPO_ROOT / "cmake-build-release",
        REPO_ROOT / "cmake-build-debug",
        REPO_ROOT / "dist" / "bin",
    ]

    for root in search_roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                candidates.append(candidate)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique_candidates.append(resolved)
            seen.add(resolved)
    return unique_candidates


def pick_launch_command() -> list[str]:
    for candidate in executable_candidates():
        if candidate.is_file():
            return [str(candidate), "--no-splash"]
    raise RuntimeError(
        "Could not find a LichtFeld Studio executable. "
        "Set LICHTFELD_EXECUTABLE or build the app first."
    )


def launch_environment(command: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    executable = Path(command[0])
    extra_library_dirs: list[str] = []

    if executable.parent.name == "build":
        extra_library_dirs.append(str(executable.parent))
    if executable.parent == REPO_ROOT / "dist" / "bin":
        extra_library_dirs.append(str(REPO_ROOT / "dist" / "lib"))

    if extra_library_dirs and os.name != "nt":
        current = env.get("LD_LIBRARY_PATH", "")
        pieces = extra_library_dirs + ([current] if current else [])
        env["LD_LIBRARY_PATH"] = ":".join(pieces)

    return env


class StartupLock:
    def __enter__(self) -> "StartupLock":
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._handle = LOCK_PATH.open("a+", encoding="utf-8")
        if fcntl is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if fcntl is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()


def tail_log(limit: int = 20) -> str:
    if not LOG_PATH.exists():
        return ""
    lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def ensure_server_ready() -> None:
    if endpoint_ready():
        return

    with StartupLock():
        if endpoint_ready():
            return

        command = pick_launch_command()
        env = launch_environment(command)

        log(f"Starting LichtFeld Studio: {' '.join(command)}")
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        deadline = time.monotonic() + DEFAULT_START_TIMEOUT_S
        while time.monotonic() < deadline:
            if endpoint_ready():
                log("LichtFeld MCP endpoint is ready.")
                return

            exit_code = process.poll()
            if exit_code is not None:
                details = tail_log()
                raise RuntimeError(
                    f"LichtFeld Studio exited before MCP came up (exit {exit_code}).\n{details}"
                )

            time.sleep(START_POLL_INTERVAL_S)

    raise RuntimeError(
        f"Timed out after {DEFAULT_START_TIMEOUT_S:.0f}s waiting for {DEFAULT_ENDPOINT}.\n{tail_log()}"
    )


def read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break

        if b":" not in line:
            raise RuntimeError(f"Malformed MCP header: {line!r}")

        name, value = line.decode("utf-8").split(":", 1)
        headers[name.strip().lower()] = value.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        raise RuntimeError("Missing or invalid Content-Length header.")

    payload = sys.stdin.buffer.read(content_length)
    if len(payload) != content_length:
        raise RuntimeError("Unexpected EOF while reading MCP payload.")

    return json.loads(payload.decode("utf-8"))


def write_message(payload: Any) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def is_notification(message: Any) -> bool:
    return isinstance(message, dict) and "id" not in message


def error_response(message: Any, error_text: str) -> dict[str, Any]:
    request_id: Any = 0
    if isinstance(message, dict) and "id" in message:
        request_id = message["id"]
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32603,
            "message": error_text,
        },
    }


def forward_message(message: Any) -> Any:
    return post_json(message, timeout_s=30.0)


def main() -> int:
    try:
        ensure_server_ready()
    except Exception as exc:  # pragma: no cover - exercised in live startup.
        log(f"Startup failed: {exc}")
        print(f"lichtfeld-mcp-bridge: {exc}", file=sys.stderr)
        return 1

    while True:
        try:
            message = read_message()
        except Exception as exc:
            log(f"Failed to read MCP message: {exc}")
            print(f"lichtfeld-mcp-bridge: {exc}", file=sys.stderr)
            return 1

        if message is None:
            return 0

        try:
            response = forward_message(message)
        except Exception as exc:
            log(f"HTTP forward failed, retrying after startup check: {exc}")
            try:
                ensure_server_ready()
                response = forward_message(message)
            except Exception as retry_exc:
                if is_notification(message):
                    log(f"Dropping notification after transport failure: {retry_exc}")
                    continue
                write_message(error_response(message, f"LichtFeld transport error: {retry_exc}"))
                continue

        if not is_notification(message):
            write_message(response)


if __name__ == "__main__":
    raise SystemExit(main())
