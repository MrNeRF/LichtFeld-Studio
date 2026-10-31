# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""COLMAP resolution tests: configured path, cache, PATH, consent-gated Windows download. No network."""

import hashlib
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from lfs_plugins.video_to_splat import toolchain  # noqa: E402

REAL_BANNER = (
    "COLMAP 4.2.1 (Commit bd1fcf6 on 2026-09-29 with CUDA, boost hash maps)\n"
    "Usage:\n  colmap [command] [options]"
)


def fake_version(text: str):
    def run(cmd: list[str]) -> str:
        assert cmd[1] == "help"
        return text
    return run


def test_parse_version():
    assert toolchain.parse_version(REAL_BANNER) == (4, 2, 1)
    assert toolchain.parse_version("COLMAP 3.11.0 -- Structure-from-Motion and Multi-View Stereo") == (3, 11, 0)
    assert toolchain.parse_version("garbage") is None
    assert toolchain.parse_version("") is None


def test_configured_path_wins(tmp_path):
    exe = tmp_path / "colmap.exe"
    exe.write_bytes(b"")
    resolved = toolchain.resolve_colmap(
        configured_path=str(exe), cache_dir=tmp_path / "cache", platform="win32",
        which=lambda name: pytest.fail("configured path must win"), run_version=fake_version(REAL_BANNER),
        request_consent=lambda: pytest.fail("no consent needed"), download=None,
    )
    assert resolved.path == exe
    assert resolved.version == (4, 2, 1)


def test_configured_path_missing_is_an_error(tmp_path):
    with pytest.raises(toolchain.ToolchainError) as err:
        toolchain.resolve_colmap(
            configured_path=str(tmp_path / "nowhere" / "colmap.exe"), cache_dir=tmp_path, platform="win32",
            which=lambda name: None, run_version=fake_version(REAL_BANNER),
            request_consent=lambda: False, download=None,
        )
    assert "nowhere" in str(err.value)


def test_old_version_rejected(tmp_path):
    exe = tmp_path / "colmap"
    exe.write_bytes(b"")
    with pytest.raises(toolchain.ToolchainError) as err:
        toolchain.resolve_colmap(
            configured_path=str(exe), cache_dir=tmp_path, platform="linux",
            which=lambda name: None, run_version=fake_version("COLMAP 3.11.0"),
            request_consent=lambda: False, download=None,
        )
    assert "3.11.0" in str(err.value)
    assert "4.1.0" in str(err.value)


def test_not_a_colmap_binary_rejected(tmp_path):
    exe = tmp_path / "colmap"
    exe.write_bytes(b"")
    with pytest.raises(toolchain.ToolchainError):
        toolchain.resolve_colmap(
            configured_path=str(exe), cache_dir=tmp_path, platform="linux",
            which=lambda name: None, run_version=fake_version("not colmap at all"),
            request_consent=lambda: False, download=None,
        )


def test_path_lookup_used_when_nothing_configured_or_cached(tmp_path):
    exe = tmp_path / "usr" / "bin" / "colmap"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    resolved = toolchain.resolve_colmap(
        configured_path="", cache_dir=tmp_path / "cache", platform="linux",
        which=lambda name: str(exe) if name == "colmap" else None, run_version=fake_version("COLMAP 4.1.0"),
        request_consent=lambda: pytest.fail("never on linux"), download=None,
    )
    assert resolved.path == exe
    assert resolved.version == (4, 1, 0)


def test_linux_without_binary_explains(tmp_path):
    with pytest.raises(toolchain.ToolchainError) as err:
        toolchain.resolve_colmap(
            configured_path="", cache_dir=tmp_path, platform="linux",
            which=lambda name: None, run_version=fake_version(""),
            request_consent=lambda: pytest.fail("never on linux"), download=None,
        )
    assert "4.1" in str(err.value)


def _make_zip(path: Path) -> str:
    """Mirror the real colmap-x64-windows-cuda.zip layout: bin/, plugins/, COLMAP.bat at the root."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("bin/colmap.exe", b"MZ fake")
        zf.writestr("bin/Qt6Core.dll", b"MZ fake dll")
        zf.writestr("plugins/platforms/qwindows.dll", b"MZ fake plugin")
        zf.writestr("COLMAP.bat", b"@echo off")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_windows_download_flow(tmp_path, monkeypatch):
    archive = tmp_path / "stage.zip"
    digest = _make_zip(archive)
    monkeypatch.setattr(toolchain, "COLMAP_WINDOWS_SHA256", digest)
    consents = []
    progress = []

    def download(url: str, dest: Path, on_progress) -> None:
        assert url == toolchain.COLMAP_WINDOWS_URL
        assert dest.parent.is_dir()
        dest.write_bytes(archive.read_bytes())
        on_progress("done")

    resolved = toolchain.resolve_colmap(
        configured_path="", cache_dir=tmp_path / "cache", platform="win32",
        which=lambda name: None, run_version=fake_version(REAL_BANNER),
        request_consent=lambda: consents.append(True) or True, download=download,
        on_progress=progress.append,
    )
    assert consents == [True]
    assert progress == ["done"]
    install_dir = tmp_path / "cache" / "tools" / "colmap-4.2.1"
    assert resolved.path == install_dir / "bin" / "colmap.exe"
    assert resolved.path.is_file()
    assert (install_dir / "plugins" / "platforms" / "qwindows.dll").is_file()
    assert (install_dir / "COLMAP.bat").is_file()
    assert not list((tmp_path / "cache" / "tools").glob("colmap-dl-*")), "staging folder must be removed"
    # second call finds the cached copy without asking again
    again = toolchain.resolve_colmap(
        configured_path="", cache_dir=tmp_path / "cache", platform="win32",
        which=lambda name: pytest.fail("cache wins over PATH"), run_version=fake_version(REAL_BANNER),
        request_consent=lambda: pytest.fail("cached"), download=None,
    )
    assert again.path == resolved.path


def test_windows_consent_refused(tmp_path):
    with pytest.raises(toolchain.ToolchainError) as err:
        toolchain.resolve_colmap(
            configured_path="", cache_dir=tmp_path, platform="win32",
            which=lambda name: None, run_version=fake_version(""),
            request_consent=lambda: False, download=None,
        )
    assert "download" in str(err.value).lower()


def test_windows_hash_mismatch(tmp_path, monkeypatch):
    archive = tmp_path / "stage.zip"
    _make_zip(archive)
    monkeypatch.setattr(toolchain, "COLMAP_WINDOWS_SHA256", "00" * 32)

    def download(url, dest, on_progress):
        dest.write_bytes(archive.read_bytes())

    with pytest.raises(toolchain.ToolchainError) as err:
        toolchain.resolve_colmap(
            configured_path="", cache_dir=tmp_path / "cache", platform="win32",
            which=lambda name: None, run_version=fake_version(REAL_BANNER),
            request_consent=lambda: True, download=download,
        )
    assert "SHA-256" in str(err.value)
    assert not (tmp_path / "cache" / "tools" / "colmap-4.2.1").exists()
    assert not list((tmp_path / "cache" / "tools").glob("colmap-dl-*"))


def test_windows_archive_without_binary_is_rejected(tmp_path, monkeypatch):
    archive = tmp_path / "stage.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("readme.txt", b"nothing here")
    monkeypatch.setattr(toolchain, "COLMAP_WINDOWS_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())

    def download(url, dest, on_progress):
        dest.write_bytes(archive.read_bytes())

    with pytest.raises(toolchain.ToolchainError) as err:
        toolchain.resolve_colmap(
            configured_path="", cache_dir=tmp_path / "cache", platform="win32",
            which=lambda name: None, run_version=fake_version(REAL_BANNER),
            request_consent=lambda: True, download=download,
        )
    assert "colmap.exe" in str(err.value)
    assert not (tmp_path / "cache" / "tools" / "colmap-4.2.1").exists()


def test_zip_slip_is_rejected(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", b"nope")
    with pytest.raises(toolchain.ToolchainError):
        toolchain.extract_archive(archive, tmp_path / "dest")
    assert not (tmp_path / "escape.txt").exists()
