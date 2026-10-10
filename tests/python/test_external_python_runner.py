# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Initialize the runner from an already-running Python interpreter."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("worker", [False, True], ids=["main-thread", "worker-thread"])
def test_external_interpreter_scripts(tmp_path, worker):
    import lichtfeld as lf
    module_dir = Path(lf.__file__).resolve().parent
    env = dict(os.environ, LFS_HOME=str(tmp_path / "home"), LFS_SAFE_MODE="0")
    code = f"""
import os, sys, pathlib, threading, time, builtins
sys.path[:] = {sys.path!r}
handles = []
if sys.platform == 'win32':
    for directory in [{str(module_dir)!r}, {str(module_dir.parents[1])!r}]:
        handles.append(os.add_dll_directory(directory))
import lichtfeld as lf
script = pathlib.Path({str(tmp_path / 'script.py')!r})
script.write_text('import builtins\\nbuiltins.external_runner_result = 42\\n', encoding='utf-8')
errors = []
def exercise():
    try:
        deadline = time.monotonic() + 20
        while True:
            result = lf.scripts.run([str(script)])
            if result['success'] or 'still loading' not in str(result).lower():
                break
            assert time.monotonic() < deadline, result
            time.sleep(0.01)
        assert result['success'], result
        assert builtins.external_runner_result == 42
        script.write_text("raise RuntimeError('expected-script-error')\\n", encoding='utf-8')
        result = lf.scripts.run([str(script)])
        assert not result['success'], result
        assert result['error'], result
    except BaseException as error:
        errors.append(repr(error))
if {worker!r}:
    thread = threading.Thread(target=exercise)
    thread.start()
    thread.join(25)
    assert not thread.is_alive(), 'runner deadlocked'
else:
    exercise()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
assert not errors, errors
# The host interpreter remains usable after runner initialization and script errors.
assert sum(range(5)) == 10
sys.__stdout__.write('PASS external interpreter\\n')
sys.__stdout__.flush()
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=35,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS external interpreter" in result.stdout
