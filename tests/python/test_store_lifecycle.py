# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Process shutdown tests for the native reactive store bridge."""

import os
from pathlib import Path
import subprocess
import sys


def test_native_store_subscriptions_release_python_callbacks_before_exit():
    project_root = Path(__file__).resolve().parents[2]
    build_dir = Path(os.environ.get("LFS_TEST_BUILD_DIR", project_root / "build"))
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        str(path)
        for path in (
            project_root / "src" / "python",
            build_dir / "src" / "python",
            child_env.get("PYTHONPATH", ""),
        )
        if str(path)
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lichtfeld; "
            "lichtfeld.ui.store.subscribe('account_state', lambda value: None)",
        ],
        cwd=project_root,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, (
        f"child exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
