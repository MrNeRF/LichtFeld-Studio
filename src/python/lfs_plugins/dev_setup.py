# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin development setup - creates plugin with venv and VS Code config."""

import json
import os
import subprocess
import sys
from pathlib import Path

from .templates import create_plugin


def create_plugin_with_venv(
    name: str,
    *,
    uv_path: str,
    python_path: str,
    typings_dir: str = "",
    site_packages_dir: str = "",
) -> str:
    """Create plugin with venv and VS Code configuration. Returns plugin path."""
    plugin_dir = create_plugin(name)

    assert uv_path, "UV path not provided"
    assert python_path, "Python path not provided"

    venv_path = plugin_dir / ".venv"
    venv_python = _venv_python_path(venv_path)

    venv_cmd = [
        uv_path,
        "venv",
        str(venv_path),
        "--python",
        python_path,
        "--no-managed-python",
        "--no-python-downloads",
    ]
    subprocess.run(venv_cmd, capture_output=True, text=True, check=True, env=_uv_env(set_pythonhome=True))

    sync_cmd = [
        uv_path,
        "sync",
        "--project",
        str(plugin_dir),
        "--python",
        str(venv_python),
        "--no-managed-python",
        "--no-python-downloads",
    ]
    subprocess.run(sync_cmd, capture_output=True, text=True, check=True, env=_uv_env(set_pythonhome=False))

    _generate_vscode_config(plugin_dir, venv_path, typings_dir, site_packages_dir)
    return str(plugin_dir)


def _uv_env(*, set_pythonhome: bool) -> dict:
    env = os.environ.copy()
    env["UV_NO_MANAGED_PYTHON"] = "1"
    env["UV_PYTHON_DOWNLOADS"] = "never"
    env.pop("UV_MANAGED_PYTHON", None)
    if set_pythonhome:
        env["PYTHONHOME"] = sys.prefix
    else:
        env.pop("PYTHONHOME", None)
    return env


def _venv_python_path(venv_path: Path) -> Path:
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _generate_vscode_config(
    plugin_dir: Path,
    venv_path: Path,
    typings_dir: str,
    site_packages_dir: str,
) -> None:
    if sys.platform == "win32":
        venv_python = venv_path / "Scripts" / "python.exe"
    else:
        venv_python = venv_path / "bin" / "python"

    vscode_dir = plugin_dir / ".vscode"
    vscode_dir.mkdir(exist_ok=True)

    extra_paths = []
    if typings_dir:
        extra_paths.append(typings_dir)
    if site_packages_dir:
        extra_paths.append(site_packages_dir)

    settings = {
        "python.defaultInterpreterPath": str(venv_python),
        "python.analysis.extraPaths": extra_paths,
        "python.analysis.typeCheckingMode": "basic",
    }
    (vscode_dir / "settings.json").write_text(
        json.dumps(settings, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    pyright = {
        "include": ["."],
        "extraPaths": extra_paths,
        "pythonVersion": "3.12",
        "typeCheckingMode": "basic",
        "venvPath": str(plugin_dir),
        "venv": ".venv",
    }
    (plugin_dir / "pyrightconfig.json").write_text(
        json.dumps(pyright, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    launch = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Attach to LichtFeld",
                "type": "debugpy",
                "request": "attach",
                "connect": {"host": "localhost", "port": 5678},
            }
        ],
    }
    (vscode_dir / "launch.json").write_text(
        json.dumps(launch, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )
