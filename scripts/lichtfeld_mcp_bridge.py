#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Start LichtFeld Studio on demand and bridge stdio MCP traffic to its HTTP endpoint."""

from __future__ import annotations

import atexit
import json
import os
import signal
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
DEFAULT_ENDPOINT = os.environ.get("LFS_MCP_ENDPOINT", "http://127.0.0.1:45677/mcp")
DEFAULT_START_TIMEOUT_S = float(os.environ.get("LFS_MCP_START_TIMEOUT_S", "90"))
CHILD_SHUTDOWN_GRACE_S = float(os.environ.get("LFS_MCP_SHUTDOWN_GRACE_S", "5"))
START_POLL_INTERVAL_S = 0.5
DEFAULT_PROTOCOL_VERSION = "2024-11-05"
BRIDGE_NAME = "lichtfeld-mcp-bridge"
BRIDGE_VERSION = "1.0.0"
LOG_PATH = Path(
    os.environ.get(
        "LFS_MCP_BRIDGE_LOG",
        str(Path.home() / ".codex" / "log" / "lichtfeld-mcp-bridge.log"),
    )
)
LOCK_PATH = Path(
    os.environ.get(
        "LFS_MCP_BRIDGE_LOCK",
        str(Path.home() / ".codex" / "log" / "lichtfeld-mcp-bridge.lock"),
    )
)
TOOLS_CACHE_PATH = Path(
    os.environ.get(
        "LFS_MCP_TOOLS_CACHE",
        str(Path.home() / ".codex" / "log" / "lichtfeld-mcp-tools.json"),
    )
)

_spawned_processes: list[tuple[subprocess.Popen[Any], int | None]] = []
_attached_to_existing_instance = False
_cleanup_in_progress = False


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
    env_executable = os.environ.get("LFS_EXECUTABLE")
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
        "Set LFS_EXECUTABLE or build the app first."
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


def bridge_spawn_is_running() -> bool:
    return any(process.poll() is None for process, _ in _spawned_processes)


def record_existing_attachment() -> None:
    global _attached_to_existing_instance

    if bridge_spawn_is_running() or _attached_to_existing_instance:
        return

    _attached_to_existing_instance = True
    log(f"Attached to existing LichtFeld MCP endpoint at {DEFAULT_ENDPOINT}.")


def track_spawned_process(process: subprocess.Popen[Any]) -> None:
    process_group_id: int | None = None
    if os.name != "nt":
        try:
            process_group_id = os.getpgid(process.pid)
        except (ProcessLookupError, PermissionError):
            process_group_id = process.pid

    _spawned_processes.append((process, process_group_id))


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_posix_process_group(
    process: subprocess.Popen[Any],
    process_group_id: int | None,
) -> None:
    try:
        process_group_id = os.getpgid(process.pid)
    except ProcessLookupError:
        pass
    except PermissionError:
        log(f"Permission denied while resolving process group for child {process.pid}.")

    if process_group_id is None:
        process_group_id = process.pid

    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        process.poll()
        log(f"Bridge-spawned child {process.pid} had already exited.")
        return
    except PermissionError:
        log(f"Permission denied while terminating process group {process_group_id}.")
        return

    deadline = time.monotonic() + CHILD_SHUTDOWN_GRACE_S
    while time.monotonic() < deadline:
        process.poll()
        if not process_group_exists(process_group_id):
            log(f"Terminated bridge-spawned process group {process_group_id}.")
            return
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        process.poll()
        log(f"Terminated bridge-spawned process group {process_group_id}.")
        return
    except PermissionError:
        log(f"Permission denied while killing process group {process_group_id}.")
        return

    try:
        process.wait(timeout=1.0)
    except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
        pass
    log(f"Terminated bridge-spawned process group {process_group_id} with SIGKILL.")


def terminate_windows_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        log(f"Bridge-spawned child {process.pid} had already exited.")
        return

    try:
        process.terminate()
    except ProcessLookupError:
        log(f"Bridge-spawned child {process.pid} had already exited.")
        return
    except PermissionError:
        log(f"Permission denied while terminating child {process.pid}.")
        return

    try:
        process.wait(timeout=CHILD_SHUTDOWN_GRACE_S)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except PermissionError:
            log(f"Permission denied while killing child {process.pid}.")
            return
        try:
            process.wait(timeout=1.0)
        except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
            pass

    log(f"Terminated bridge-spawned child {process.pid}.")


def cleanup_spawned_processes() -> None:
    global _cleanup_in_progress

    if _cleanup_in_progress or not _spawned_processes:
        return

    _cleanup_in_progress = True
    processes = list(reversed(_spawned_processes))
    _spawned_processes.clear()
    try:
        for process, process_group_id in processes:
            log(f"Cleaning up bridge-spawned child {process.pid}.")
            if os.name == "nt":
                terminate_windows_process(process)
            else:
                terminate_posix_process_group(process, process_group_id)
    finally:
        _cleanup_in_progress = False


def handle_shutdown_signal(signum: int, _frame: Any) -> None:
    log(f"Bridge received signal {signum}; cleaning up.")
    cleanup_spawned_processes()
    raise SystemExit(128 + signum)


def install_signal_handlers() -> None:
    for signal_name in ("SIGTERM", "SIGINT", "SIGHUP"):
        shutdown_signal = getattr(signal, signal_name, None)
        if shutdown_signal is not None:
            signal.signal(shutdown_signal, handle_shutdown_signal)


atexit.register(cleanup_spawned_processes)


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
        record_existing_attachment()
        return

    with StartupLock():
        if endpoint_ready():
            record_existing_attachment()
            return

        command = pick_launch_command()
        env = launch_environment(command)

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
        track_spawned_process(process)
        log(f"Spawned LichtFeld Studio child {process.pid}: {' '.join(command)}")

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


def success_response(message: dict[str, Any], result: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message["id"],
        "result": result,
    }


def initialize_response(message: dict[str, Any]) -> dict[str, Any]:
    protocol_version = DEFAULT_PROTOCOL_VERSION
    params = message.get("params")
    if isinstance(params, dict) and "protocolVersion" in params:
        protocol_version = params["protocolVersion"]

    return success_response(
        message,
        {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": BRIDGE_NAME,
                "version": BRIDGE_VERSION,
            },
        },
    )


def normalize_tools_manifest(payload: Any) -> dict[str, Any]:
    manifest = payload
    if isinstance(payload, dict) and "result" in payload:
        manifest = payload["result"]
    if isinstance(manifest, list):
        manifest = {"tools": manifest}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tools"), list):
        raise ValueError("tools manifest must contain a tools list")
    return manifest


def read_tools_cache() -> dict[str, Any] | None:
    try:
        payload = json.loads(TOOLS_CACHE_PATH.read_text(encoding="utf-8"))
        return normalize_tools_manifest(payload)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        log(f"Ignoring invalid tools cache at {TOOLS_CACHE_PATH}: {exc}")
        return None


def write_tools_cache(response: Any) -> None:
    try:
        manifest = normalize_tools_manifest(response)
    except ValueError as exc:
        log(f"Not caching invalid live tools/list response: {exc}")
        return

    temporary_path = TOOLS_CACHE_PATH.with_name(
        f".{TOOLS_CACHE_PATH.name}.{os.getpid()}.tmp"
    )
    try:
        TOOLS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, TOOLS_CACHE_PATH)
    except OSError as exc:
        log(f"Failed to refresh tools cache at {TOOLS_CACHE_PATH}: {exc}")
        try:
            temporary_path.unlink()
        except (FileNotFoundError, PermissionError):
            pass
        return

    log(f"Refreshed tools cache from live app at {TOOLS_CACHE_PATH}.")


def forward_message(message: Any) -> Any:
    return post_json(message, timeout_s=30.0)


def main() -> int:
    install_signal_handlers()
    try:
        while True:
            try:
                message = read_message()
            except Exception as exc:
                log(f"Failed to read MCP message: {exc}")
                print(f"lichtfeld-mcp-bridge: {exc}", file=sys.stderr)
                return 1

            if message is None:
                return 0

            if is_notification(message):
                method = message.get("method") if isinstance(message, dict) else None
                log(f"Dropped MCP notification locally: {method or '<unknown>'}.")
                continue

            method = message.get("method") if isinstance(message, dict) else None
            if method == "initialize":
                log("Served MCP initialize locally.")
                write_message(initialize_response(message))
                continue

            if method == "ping":
                log("Served MCP ping locally.")
                write_message(success_response(message, {}))
                continue

            if method == "tools/list":
                cached_manifest = read_tools_cache()
                if cached_manifest is not None:
                    log(f"Served tools/list from cache at {TOOLS_CACHE_PATH}.")
                    write_message(success_response(message, cached_manifest))
                    continue
                log(f"tools/list cache miss at {TOOLS_CACHE_PATH}; querying live app.")

            try:
                response = forward_message(message)
                record_existing_attachment()
            except Exception as exc:
                log(f"HTTP forward failed, retrying after startup check: {exc}")
                try:
                    ensure_server_ready()
                    response = forward_message(message)
                except Exception as retry_exc:
                    write_message(
                        error_response(message, f"LichtFeld transport error: {retry_exc}")
                    )
                    continue

            if method == "tools/list":
                write_tools_cache(response)
            write_message(response)
    finally:
        cleanup_spawned_processes()


if __name__ == "__main__":
    raise SystemExit(main())
