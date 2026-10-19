# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Standalone interpreter shutdown regression for Python undo callbacks."""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BUILD_DIR = os.path.abspath(os.environ.get("LFS_TEST_BUILD_DIR", os.path.join(ROOT, "build")))


def test_committed_undo_transaction_exits_cleanly_in_standalone_python(tmp_path):
    env = dict(os.environ)
    env["HOME"] = str(tmp_path / "home")
    os.makedirs(env["HOME"])
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(BUILD_DIR, "src", "python"), os.path.join(ROOT, "src", "python")]
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, [BUILD_DIR, env.get("LD_LIBRARY_PATH", "")])
    )
    code = """
import lichtfeld.undo as undo
def commit_transaction():
    state = {"value": 0}
    with undo.transaction("shutdown regression") as tx:
        tx.add(lambda: state.update(value=0), lambda: state.update(value=1))
    assert state["value"] == 1
    assert undo.stack()["undo"]

commit_transaction()
print("TRANSACTION_COMMITTED")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"standalone interpreter exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "TRANSACTION_COMMITTED" in result.stdout
