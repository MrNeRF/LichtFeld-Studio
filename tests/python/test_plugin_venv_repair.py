# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for in-place plugin venv interpreter repair."""

import logging
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

import lfs_plugins.installer as installer_module
from lfs_plugins.installer import PluginInstaller
from lfs_plugins.plugin import PluginInfo, PluginInstance


@pytest.fixture
def venv_case(tmp_path, monkeypatch):
    """Create an installer with a fake bundled CPython 3.12 runtime."""
    plugin_path = tmp_path / "plugin"
    plugin_path.mkdir()
    (plugin_path / "pyproject.toml").write_text(
        '[project]\nname = "repair-test"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )

    bundled_python = tmp_path / "current" / "bin" / "python3.12"
    bundled_python.parent.mkdir(parents=True)
    bundled_python.touch()

    plugin = PluginInstance(
        info=PluginInfo(name="repair-test", version="1.0.0", path=plugin_path)
    )
    installer = PluginInstaller(plugin)
    monkeypatch.setattr(installer, "_get_embedded_python", lambda: bundled_python)
    monkeypatch.setattr(installer, "_current_python_version", lambda: (3, 12))
    return installer, plugin_path, bundled_python


@pytest.fixture
def windows_venv_case(venv_case, monkeypatch):
    """Create an installer with an emulated Windows bundled runtime."""
    installer, plugin_path, _ = venv_case
    bundled_python = plugin_path.parent / "current-windows" / "python.exe"
    bundled_python.parent.mkdir(parents=True)
    bundled_python.write_bytes(b"bundled")
    monkeypatch.setattr(installer, "_get_embedded_python", lambda: bundled_python)
    monkeypatch.setattr(installer_module, "_is_windows", lambda: True)
    return installer, plugin_path, bundled_python


def _fabricate_venv(
    plugin_path: Path,
    bundled_python: Path,
    *,
    version: str = "3.12.7",
    version_key: str = "version_info",
    matching: bool = False,
    config_matching: bool = False,
    dangling: bool = False,
    python_names: tuple[str, ...] = ("python", "python3", "python3.12"),
) -> tuple[Path, Path]:
    venv_path = plugin_path / ".venv"
    bin_path = venv_path / "bin"
    bin_path.mkdir(parents=True)

    if matching:
        base_python = bundled_python
    else:
        base_python = plugin_path.parent / "old" / "bin" / "python3.12"
        if not dangling:
            base_python.parent.mkdir(parents=True)
            base_python.touch()

    for name in python_names:
        (bin_path / name).symlink_to(base_python)

    config_python = bundled_python if matching or config_matching else base_python
    (venv_path / "pyvenv.cfg").write_text(
        f"home = {config_python.parent}\n"
        "implementation = CPython\n"
        "uv = 0.7.12\n"
        f"{version_key} = {version}\n"
        f"executable = {config_python}\n"
        f"base-executable = {config_python}\n"
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    stamp = venv_path / PluginInstaller.DEPS_STAMP
    stamp.write_text("installed", encoding="utf-8")
    return venv_path, stamp


def _fabricate_windows_venv(
    plugin_path: Path,
    bundled_python: Path,
    *,
    matching: bool = False,
    executable_matching: bool = False,
    python_exists: bool = True,
) -> tuple[Path, Path, Path]:
    venv_path = plugin_path / ".venv"
    scripts_path = venv_path / "Scripts"
    scripts_path.mkdir(parents=True)
    venv_python = scripts_path / "python.exe"
    if python_exists:
        venv_python.write_bytes(b"launcher")

    old_python = plugin_path.parent / "old-windows" / "python.exe"
    old_python.parent.mkdir(parents=True)
    old_python.write_bytes(b"old")
    config_home = bundled_python if matching else old_python
    config_python = bundled_python if matching or executable_matching else old_python
    (venv_path / "pyvenv.cfg").write_text(
        f"home = {config_home.parent}\n"
        "implementation = CPython\n"
        "uv = 0.7.12\n"
        "version_info = 3.12.7\n"
        f"executable = {config_python}\n"
        f"base-executable = {config_python}\n"
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    stamp = venv_path / PluginInstaller.DEPS_STAMP
    stamp.write_text("installed", encoding="utf-8")
    return venv_path, venv_python, stamp


def _mock_recreation(monkeypatch, installer: PluginInstaller, plugin_path: Path):
    uv = plugin_path / "uv"
    uv.touch()
    monkeypatch.setattr(installer, "_find_uv", lambda: uv)
    run = Mock(
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
    )
    monkeypatch.setattr(installer_module, "_run_cancellable_process", run)
    real_rmtree = installer_module.shutil.rmtree
    rmtree = Mock(wraps=real_rmtree)
    monkeypatch.setattr(installer_module.shutil, "rmtree", rmtree)
    return rmtree, run


def test_stale_home_same_minor_repairs_without_removal(venv_case, monkeypatch):
    """A stale same-minor venv is repaired without losing its dependency stamp."""
    installer, plugin_path, bundled_python = venv_case
    venv_path, stamp = _fabricate_venv(plugin_path, bundled_python)
    stamp_stat = stamp.stat()
    rmtree = Mock(side_effect=AssertionError("venv must not be removed"))
    find_uv = Mock(side_effect=AssertionError("uv must not run"))
    monkeypatch.setattr(installer_module.shutil, "rmtree", rmtree)
    monkeypatch.setattr(installer, "_find_uv", find_uv)

    assert installer.ensure_venv() is True

    cfg = (venv_path / "pyvenv.cfg").read_text(encoding="utf-8")
    resolved_python = bundled_python.resolve()
    assert f"home = {resolved_python.parent}" in cfg
    assert f"executable = {resolved_python}" in cfg
    assert f"base-executable = {resolved_python}" in cfg
    assert "uv = 0.7.12" in cfg
    for name in ("python", "python3", "python3.12"):
        assert (venv_path / "bin" / name).resolve() == resolved_python
    assert stamp.read_text(encoding="utf-8") == "installed"
    assert stamp.stat().st_ino == stamp_stat.st_ino
    assert stamp.stat().st_mtime_ns == stamp_stat.st_mtime_ns
    assert installer._deps_already_installed() is True
    assert installer.install_dependencies() is True
    rmtree.assert_not_called()
    find_uv.assert_not_called()


def test_matching_config_with_stale_symlink_is_repaired(venv_case, monkeypatch):
    """A cfg-only match does not hide a stale POSIX interpreter symlink."""
    installer, plugin_path, bundled_python = venv_case
    venv_path, stamp = _fabricate_venv(
        plugin_path,
        bundled_python,
        config_matching=True,
    )
    stamp_stat = stamp.stat()
    old_python = (venv_path / "bin" / "python").resolve()
    repair_impl = installer._repair_venv_interpreter
    repair = Mock(wraps=repair_impl)
    rmtree = Mock(side_effect=AssertionError("venv must not be removed"))
    find_uv = Mock(side_effect=AssertionError("uv must not run"))
    monkeypatch.setattr(installer, "_repair_venv_interpreter", repair)
    monkeypatch.setattr(installer_module.shutil, "rmtree", rmtree)
    monkeypatch.setattr(installer, "_find_uv", find_uv)

    assert old_python != bundled_python.resolve()
    assert installer.ensure_venv() is True

    repair.assert_called_once_with(venv_path, bundled_python)
    assert (venv_path / "bin" / "python").resolve() == bundled_python.resolve()
    cfg = (venv_path / "pyvenv.cfg").read_text(encoding="utf-8")
    assert f"home = {bundled_python.resolve().parent}" in cfg
    assert stamp.stat().st_ino == stamp_stat.st_ino
    assert stamp.stat().st_mtime_ns == stamp_stat.st_mtime_ns
    rmtree.assert_not_called()
    find_uv.assert_not_called()


def test_dangling_python_symlink_same_minor_is_repaired(venv_case, monkeypatch):
    """A dangling venv Python link is redirected to the bundled interpreter."""
    installer, plugin_path, bundled_python = venv_case
    venv_path, stamp = _fabricate_venv(
        plugin_path,
        bundled_python,
        dangling=True,
        version_key="version",
        python_names=("python",),
    )
    rmtree = Mock(side_effect=AssertionError("venv must not be removed"))
    monkeypatch.setattr(installer_module.shutil, "rmtree", rmtree)

    assert not (venv_path / "bin" / "python").exists()
    assert installer.ensure_venv() is True

    assert (venv_path / "bin" / "python").resolve() == bundled_python.resolve()
    assert {path.name for path in (venv_path / "bin").iterdir()} == {"python"}
    assert stamp.exists()
    rmtree.assert_not_called()


def test_minor_mismatch_recreates_without_repair(venv_case, monkeypatch):
    """A venv from another CPython minor is recreated without repair."""
    installer, plugin_path, bundled_python = venv_case
    venv_path, _ = _fabricate_venv(
        plugin_path,
        bundled_python,
        version="3.11.9",
    )
    repair = Mock(side_effect=AssertionError("incompatible venv must not be repaired"))
    monkeypatch.setattr(installer, "_repair_venv_interpreter", repair)
    rmtree, run = _mock_recreation(monkeypatch, installer, plugin_path)

    assert installer.ensure_venv() is True

    repair.assert_not_called()
    rmtree.assert_called_once()
    assert Path(rmtree.call_args.args[0]) == venv_path
    run.assert_called_once()


def test_regular_file_interpreter_falls_back_to_recreate(venv_case, monkeypatch, caplog):
    """Repair refuses to replace a regular-file venv interpreter."""
    installer, plugin_path, bundled_python = venv_case
    venv_path, _ = _fabricate_venv(plugin_path, bundled_python)
    venv_python = venv_path / "bin" / "python"
    venv_python.unlink()
    venv_python.write_bytes(b"launcher")
    rmtree, run = _mock_recreation(monkeypatch, installer, plugin_path)

    with caplog.at_level(logging.WARNING, logger="lfs_plugins.installer"):
        assert installer.ensure_venv() is True

    assert "is not a symlink" in caplog.text
    rmtree.assert_called_once()
    assert Path(rmtree.call_args.args[0]) == venv_path
    run.assert_called_once()


@pytest.mark.parametrize("version_line", [None, "version_info = garbage"])
def test_invalid_venv_version_falls_back_to_recreate(
    venv_case, monkeypatch, caplog, version_line
):
    """Missing or malformed venv version metadata falls back to recreation."""
    installer, plugin_path, bundled_python = venv_case
    venv_path, _ = _fabricate_venv(plugin_path, bundled_python)
    cfg_path = venv_path / "pyvenv.cfg"
    lines = [
        line
        for line in cfg_path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("version_info =")
    ]
    if version_line:
        lines.append(version_line)
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rmtree, run = _mock_recreation(monkeypatch, installer, plugin_path)

    with caplog.at_level(logging.WARNING, logger="lfs_plugins.installer"):
        assert installer.ensure_venv() is True

    assert "Failed to repair plugin venv interpreter" in caplog.text
    rmtree.assert_called_once()
    assert Path(rmtree.call_args.args[0]) == venv_path
    run.assert_called_once()


def test_repair_failure_falls_back_to_recreate(venv_case, monkeypatch, caplog):
    """A filesystem failure during repair falls back to uv recreation."""
    installer, plugin_path, bundled_python = venv_case
    venv_path, _ = _fabricate_venv(plugin_path, bundled_python)
    cfg_path = venv_path / "pyvenv.cfg"
    original_write_text = Path.write_text

    def fail_cfg_write(path, *args, **kwargs):
        if path == cfg_path:
            raise PermissionError("read-only pyvenv.cfg")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_cfg_write)
    rmtree, run = _mock_recreation(monkeypatch, installer, plugin_path)

    with caplog.at_level(logging.WARNING, logger="lfs_plugins.installer"):
        assert installer.ensure_venv() is True

    assert "read-only pyvenv.cfg" in caplog.text
    rmtree.assert_called_once()
    assert Path(rmtree.call_args.args[0]) == venv_path
    run.assert_called_once()


def test_matching_home_uses_existing_fast_path(venv_case, monkeypatch):
    """A matching venv returns immediately without repair or recreation."""
    installer, plugin_path, bundled_python = venv_case
    venv_path, stamp = _fabricate_venv(
        plugin_path,
        bundled_python,
        matching=True,
    )
    repair = Mock(side_effect=AssertionError("matching venv must not be repaired"))
    rmtree = Mock(side_effect=AssertionError("matching venv must not be removed"))
    find_uv = Mock(side_effect=AssertionError("uv must not run"))
    monkeypatch.setattr(installer, "_repair_venv_interpreter", repair)
    monkeypatch.setattr(installer_module.shutil, "rmtree", rmtree)
    monkeypatch.setattr(installer, "_find_uv", find_uv)

    assert installer.ensure_venv() is True

    assert venv_path.exists()
    assert stamp.exists()
    repair.assert_not_called()
    rmtree.assert_not_called()
    find_uv.assert_not_called()


def test_windows_stale_config_repairs_without_touching_launcher(
    windows_venv_case, monkeypatch
):
    """Windows repair rewrites only pyvenv.cfg and preserves the launcher."""
    installer, plugin_path, bundled_python = windows_venv_case
    venv_path, venv_python, stamp = _fabricate_windows_venv(
        plugin_path, bundled_python, executable_matching=True
    )
    launcher_stat = venv_python.stat()
    stamp_stat = stamp.stat()
    repair_impl = installer._repair_venv_interpreter
    repair = Mock(wraps=repair_impl)
    rmtree = Mock(side_effect=AssertionError("venv must not be removed"))
    find_uv = Mock(side_effect=AssertionError("uv must not run"))
    monkeypatch.setattr(installer, "_repair_venv_interpreter", repair)
    monkeypatch.setattr(installer_module.shutil, "rmtree", rmtree)
    monkeypatch.setattr(installer, "_find_uv", find_uv)

    assert installer.ensure_venv() is True

    repair.assert_called_once_with(venv_path, bundled_python)
    cfg = (venv_path / "pyvenv.cfg").read_text(encoding="utf-8")
    resolved_python = bundled_python.resolve()
    assert f"home = {resolved_python.parent}" in cfg
    assert f"executable = {resolved_python}" in cfg
    assert f"base-executable = {resolved_python}" in cfg
    assert "uv = 0.7.12" in cfg
    assert venv_python.read_bytes() == b"launcher"
    assert venv_python.stat().st_ino == launcher_stat.st_ino
    assert venv_python.stat().st_mtime_ns == launcher_stat.st_mtime_ns
    assert not (venv_path / "bin").exists()
    assert stamp.stat().st_ino == stamp_stat.st_ino
    assert stamp.stat().st_mtime_ns == stamp_stat.st_mtime_ns
    rmtree.assert_not_called()
    find_uv.assert_not_called()


def test_windows_missing_launcher_recreates(windows_venv_case, monkeypatch):
    """A missing Windows launcher recreates the venv even if bin/python exists."""
    installer, plugin_path, bundled_python = windows_venv_case
    venv_path, venv_python, _ = _fabricate_windows_venv(
        plugin_path, bundled_python, python_exists=False
    )
    bin_path = venv_path / "bin"
    bin_path.mkdir()
    (bin_path / "python").write_bytes(b"decoy")
    repair_impl = installer._repair_venv_interpreter
    repair = Mock(wraps=repair_impl)
    monkeypatch.setattr(installer, "_repair_venv_interpreter", repair)
    rmtree, run = _mock_recreation(monkeypatch, installer, plugin_path)

    assert not venv_python.exists()
    assert installer.ensure_venv() is True

    repair.assert_called_once_with(venv_path, bundled_python)
    rmtree.assert_called_once()
    assert Path(rmtree.call_args.args[0]) == venv_path
    run.assert_called_once()


def test_windows_matching_config_uses_existing_fast_path(
    windows_venv_case, monkeypatch
):
    """A matching Windows cfg accepts its regular-file launcher unchanged."""
    installer, plugin_path, bundled_python = windows_venv_case
    venv_path, venv_python, stamp = _fabricate_windows_venv(
        plugin_path, bundled_python, matching=True
    )
    old_python = plugin_path.parent / "old-windows" / "python.exe"
    bin_path = venv_path / "bin"
    bin_path.mkdir()
    (bin_path / "python").symlink_to(old_python)
    launcher_stat = venv_python.stat()
    repair = Mock(side_effect=AssertionError("matching venv must not be repaired"))
    rmtree = Mock(side_effect=AssertionError("matching venv must not be removed"))
    find_uv = Mock(side_effect=AssertionError("uv must not run"))
    monkeypatch.setattr(installer, "_repair_venv_interpreter", repair)
    monkeypatch.setattr(installer_module.shutil, "rmtree", rmtree)
    monkeypatch.setattr(installer, "_find_uv", find_uv)

    assert installer.ensure_venv() is True

    assert venv_python.read_bytes() == b"launcher"
    assert venv_python.stat().st_ino == launcher_stat.st_ino
    assert venv_python.stat().st_mtime_ns == launcher_stat.st_mtime_ns
    assert stamp.exists()
    repair.assert_not_called()
    rmtree.assert_not_called()
    find_uv.assert_not_called()
