# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cancellable subprocess execution with streaming output.

Shared by the plugin installer and the video-to-splat pipeline. The child runs
in its own process group so a cancel can take its whole tree down.
"""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
from collections import deque
from typing import Callable, Optional

from _lfs_path_io import executable_kwargs

PROCESS_POLL_SECONDS = 0.05
PROCESS_TERMINATE_GRACE_SECONDS = 0.5
OUTPUT_TAIL_LINES = 100


class ProcessCancelled(Exception):
    """Raised when should_cancel() asked to stop a running child process."""


def is_windows() -> bool:
    return sys.platform == "win32"


def cancel_requested(should_cancel: Optional[Callable[[], bool]]) -> bool:
    return bool(should_cancel and should_cancel())


def wait_for_process(proc: subprocess.Popen, timeout: float) -> bool:
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def terminate_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort termination of a process and descendants on supported hosts."""
    if proc.poll() is not None:
        return

    if is_windows():
        # CREATE_NEW_PROCESS_GROUP gives the child its own console group. taskkill /T is
        # the best portable stdlib-only tree termination available on Windows;
        # a Job Object would provide stronger guarantees but is not used here.
        try:
            tree_kill = subprocess.Popen(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not wait_for_process(tree_kill, PROCESS_TERMINATE_GRACE_SECONDS):
                tree_kill.kill()
                wait_for_process(tree_kill, PROCESS_TERMINATE_GRACE_SECONDS)
        except OSError:
            pass
        if proc.poll() is None:
            proc.terminate()
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            proc.terminate()

    if wait_for_process(proc, PROCESS_TERMINATE_GRACE_SECONDS):
        return

    if is_windows():
        proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            proc.kill()
    wait_for_process(proc, PROCESS_TERMINATE_GRACE_SECONDS)


def process_group_popen_kwargs() -> dict:
    if is_windows():
        return {
            "creationflags": getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
            )
        }
    return {"start_new_session": True}


def run_cancellable_process(
    cmd: list[str],
    *,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
    on_output: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> subprocess.CompletedProcess:
    """Run cmd in its own process group, streaming merged stdout/stderr lines.

    Does not raise on a nonzero exit code; callers check ``returncode``. The
    returned ``stdout`` holds the last OUTPUT_TAIL_LINES lines. Raises
    ProcessCancelled when should_cancel() returns True before or during the run.
    """
    if cancel_requested(should_cancel):
        raise ProcessCancelled("cancelled before starting: " + " ".join(cmd))

    popen_kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
        "env": env,
        "cwd": cwd,
    }
    popen_kwargs.update(process_group_popen_kwargs())
    popen_kwargs.update(executable_kwargs(cmd))

    proc = subprocess.Popen(cmd, **popen_kwargs)
    output_queue: "queue.Queue[Optional[str]]" = queue.Queue()
    output_tail: deque[str] = deque(maxlen=OUTPUT_TAIL_LINES)

    def read_output() -> None:
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    output_queue.put(line.rstrip())
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=read_output, name="process-runner-output", daemon=True)
    reader.start()
    output_finished = False

    try:
        while not output_finished or proc.poll() is None:
            if cancel_requested(should_cancel):
                terminate_process_tree(proc)
                raise ProcessCancelled("cancelled: " + " ".join(cmd))

            try:
                line = output_queue.get(timeout=PROCESS_POLL_SECONDS)
            except queue.Empty:
                continue

            if line is None:
                output_finished = True
            else:
                output_tail.append(line)
                if line and on_output:
                    on_output(line)

        returncode = proc.wait()
        if cancel_requested(should_cancel):
            raise ProcessCancelled("cancelled: " + " ".join(cmd))
        return subprocess.CompletedProcess(cmd, returncode, "\n".join(output_tail), "")
    except BaseException:
        if proc.poll() is None:
            terminate_process_tree(proc)
        raise
    finally:
        reader.join(timeout=PROCESS_TERMINATE_GRACE_SECONDS)
        if proc.stdout is not None:
            proc.stdout.close()
