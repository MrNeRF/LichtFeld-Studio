# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Locate a COLMAP 4.1+ binary, downloading the official Windows build on consent.

All I/O (which, version probe, consent, download) is injectable so tests never
touch the network or a real binary.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .text import _tr

COLMAP_VERSION = "4.2.1"
COLMAP_WINDOWS_URL = (
    f"https://github.com/colmap/colmap/releases/download/{COLMAP_VERSION}/colmap-x64-windows-cuda.zip"
)
COLMAP_WINDOWS_SHA256 = "e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881"
COLMAP_WINDOWS_SIZE_MB = 395
# The official zip has bin/, plugins/ and COLMAP.bat at its root (no wrapper folder);
# the CLI lives in bin/ next to the Qt and CUDA DLLs it loads.
COLMAP_WINDOWS_EXE_RELATIVE = Path("bin") / "colmap.exe"
MIN_VERSION = (4, 1, 0)  # first release with the EQUIRECTANGULAR camera model

Version = tuple[int, int, int]
# `colmap help` prints "COLMAP 4.2.1 (Commit ... with CUDA, boost hash maps)" on stdout.
_VERSION_RE = re.compile(r"COLMAP\s+(\d+)\.(\d+)\.(\d+)")
DownloadFn = Callable[[str, Path, Callable[[str], None]], None]


class ToolchainError(Exception):
    """COLMAP is unavailable. str(error) is a user-facing message."""


@dataclass(frozen=True)
class Resolved:
    path: Path
    version: Version


def parse_version(text: str) -> Optional[Version]:
    match = _VERSION_RE.search(text or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def default_run_version(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (completed.stdout or "") + "\n" + (completed.stderr or "")


def version_string(version: Version) -> str:
    return ".".join(str(part) for part in version)


def _check_binary(path: Path, run_version: Callable[[list[str]], str]) -> Resolved:
    text = run_version([str(path), "help"])
    version = parse_version(text)
    if version is None:
        raise ToolchainError(_tr("video_to_splat.error.not_colmap",
                                 "'{path}' did not report a COLMAP version. Is it a COLMAP executable?", path=path))
    if version < MIN_VERSION:
        raise ToolchainError(_tr(
            "video_to_splat.error.colmap_too_old",
            "COLMAP {found} at '{path}' is too old. Version {minimum} or later is needed for 360 cameras.",
            found=version_string(version), path=path, minimum=version_string(MIN_VERSION)))
    return Resolved(path=path, version=version)


def cached_install_dir(cache_dir: Path) -> Path:
    return cache_dir / "tools" / f"colmap-{COLMAP_VERSION}"


def cached_binary_path(cache_dir: Path, platform: str) -> Path:
    if platform.startswith("win"):
        return cached_install_dir(cache_dir) / COLMAP_WINDOWS_EXE_RELATIVE
    return cached_install_dir(cache_dir) / "bin" / "colmap"


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive: Path, dest: Path) -> None:
    """Extract a zip into dest, dropping a single wrapper folder if there is one, rejecting path escapes."""
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        for name in names:
            parts = name.replace("\\", "/").split("/")
            if name.startswith(("/", "\\")) or ":" in parts[0] or any(part in ("..", "") for part in parts):
                raise ToolchainError(_tr("video_to_splat.error.archive_escape",
                                         "Archive entry escapes destination: {name}", name=name))
        top_levels = {n.split("/", 1)[0] for n in names}
        prefix = ""
        if len(top_levels) == 1 and all("/" in n for n in names):
            prefix = next(iter(top_levels)) + "/"
        for name in names:
            relative = name[len(prefix):] if prefix else name
            target = (dest / relative).resolve()
            if target != dest_resolved and dest_resolved not in target.parents:
                raise ToolchainError(_tr("video_to_splat.error.archive_escape",
                                         "Archive entry escapes destination: {name}", name=name))
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def _download_windows_build(cache_dir: Path, download: DownloadFn, on_progress: Callable[[str], None]) -> Path:
    target_dir = cached_install_dir(cache_dir)
    tools_dir = target_dir.parent
    tools_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix="colmap-dl-", dir=str(tools_dir)))
    archive = staging_dir / "colmap.zip"
    try:
        download(COLMAP_WINDOWS_URL, archive, on_progress)
        actual = _sha256_of(archive)
        if actual.lower() != COLMAP_WINDOWS_SHA256.lower():
            raise ToolchainError(_tr(
                "video_to_splat.error.hash_mismatch",
                "The downloaded COLMAP archive failed its SHA-256 check and was discarded. "
                "Try again, or install COLMAP manually and set its path in preferences."))
        if target_dir.exists():
            shutil.rmtree(target_dir)
        extract_archive(archive, target_dir)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    exe = cached_binary_path(cache_dir, "win32")
    if not exe.is_file():
        shutil.rmtree(target_dir, ignore_errors=True)
        raise ToolchainError(_tr("video_to_splat.error.archive_missing_exe",
                                 "The COLMAP archive did not contain {name}.", name=COLMAP_WINDOWS_EXE_RELATIVE.name))
    return exe


def resolve_colmap(
    *,
    configured_path: str,
    cache_dir: Path,
    platform: str = sys.platform,
    which: Callable[[str], Optional[str]] = shutil.which,
    run_version: Callable[[list[str]], str] = default_run_version,
    request_consent: Callable[[], bool],
    download: Optional[DownloadFn],
    on_progress: Callable[[str], None] = lambda _msg: None,
) -> Resolved:
    """Find COLMAP 4.1+. Order: configured path, cached download, PATH, then download (Windows only)."""
    if configured_path:
        path = Path(configured_path)
        if not path.is_file():
            raise ToolchainError(_tr("video_to_splat.error.configured_path_missing",
                                     "The configured COLMAP path does not exist: {path}", path=path))
        return _check_binary(path, run_version)

    cached = cached_binary_path(cache_dir, platform)
    if cached.is_file():
        return _check_binary(cached, run_version)

    found = which("colmap")
    if found:
        return _check_binary(Path(found), run_version)

    if not platform.startswith("win"):
        raise ToolchainError(_tr(
            "video_to_splat.error.not_on_path",
            "COLMAP {minimum} or later was not found on PATH. "
            "Install COLMAP or set its path in the Video to Splat preferences.",
            minimum=version_string(MIN_VERSION)))

    if not request_consent():
        raise ToolchainError(_tr(
            "video_to_splat.error.download_declined",
            "COLMAP is needed to solve camera poses and the download was declined. "
            "Allow the download, or install COLMAP 4.1+ and set its path in preferences."))
    if download is None:
        raise ToolchainError(_tr("video_to_splat.error.no_downloader", "No downloader is available for COLMAP."))
    exe = _download_windows_build(cache_dir, download, on_progress)
    return _check_binary(exe, run_version)


__all__ = [
    "COLMAP_VERSION", "COLMAP_WINDOWS_URL", "COLMAP_WINDOWS_SHA256", "COLMAP_WINDOWS_SIZE_MB", "MIN_VERSION",
    "Resolved", "ToolchainError", "cached_binary_path", "extract_archive", "parse_version", "resolve_colmap",
    "version_string",
]
