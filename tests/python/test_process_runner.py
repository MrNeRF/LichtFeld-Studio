# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavior tests for the shared cancellable process runner."""

import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from lfs_plugins.process_runner import ProcessCancelled, run_cancellable_process  # noqa: E402


def test_streams_output_lines_and_returns_code():
    lines: list[str] = []
    result = run_cancellable_process(
        [sys.executable, "-c", "print('one'); print('two'); raise SystemExit(3)"],
        on_output=lines.append,
    )
    assert lines == ["one", "two"]
    assert result.returncode == 3
    assert "two" in result.stdout


def test_cancel_kills_child_and_raises():
    started = time.monotonic()
    cancel = threading.Event()

    def on_output(_line: str) -> None:
        cancel.set()

    with pytest.raises(ProcessCancelled):
        run_cancellable_process(
            [sys.executable, "-c", "import time,sys; print('go', flush=True); time.sleep(30)"],
            on_output=on_output,
            should_cancel=cancel.is_set,
        )
    assert time.monotonic() - started < 10


def test_cancel_before_start_raises_without_spawning():
    with pytest.raises(ProcessCancelled):
        run_cancellable_process(["definitely-not-a-real-binary"], should_cancel=lambda: True)


def test_runs_in_requested_working_directory(tmp_path):
    lines: list[str] = []
    result = run_cancellable_process(
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        cwd=str(tmp_path),
        on_output=lines.append,
    )
    assert result.returncode == 0
    assert Path(lines[0]).resolve() == tmp_path.resolve()
