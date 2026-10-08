# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line integration coverage for project-resume dataset selection."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import zlib

import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _png(width: int, height: int, rgba: tuple[int, int, int, int]) -> bytes:
    """Create a small, valid RGBA PNG without relying on Pillow."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    row = bytes(rgba) * width
    image = b"".join(b"\0" + row for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(image))
        + chunk(b"IEND", b"")
    )


def _make_dataset(root: Path, prefix: str) -> None:
    images = root / "images"
    images.mkdir(parents=True)
    frames = []
    for index in range(4):
        name = f"{prefix}_{index}.png"
        (images / name).write_bytes(_png(8, 8, (32 + index * 24, 90, 150, 255)))
        matrix = [
            [1, 0, 0, index * 0.03],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        frames.append({"file_path": f"images/{name}", "transform_matrix": matrix})
    (root / "transforms.json").write_text(
        json.dumps(
            {
                "w": 8,
                "h": 8,
                "fl_x": 8.0,
                "fl_y": 8.0,
                "cx": 4.0,
                "cy": 4.0,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )


def _app_path() -> Path | None:
    configured = os.environ.get("LFS_EXECUTABLE")
    candidates = [Path(configured)] if configured else []
    candidates.append(REPO_ROOT / "build" / "LichtFeld-Studio")
    candidates.append(REPO_ROOT / "build" / "LichtFeld-Studio.exe")
    return next((path for path in candidates if path.is_file()), None)


def _python_environment() -> tuple[Path, dict[str, str]]:
    build_dir = Path(os.environ.get("LFS_TEST_BUILD_DIR", REPO_ROOT / "build"))
    python = sys.executable
    cache = build_dir / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(("Python_EXECUTABLE:FILEPATH=", "Python3_EXECUTABLE:FILEPATH=")):
                candidate = Path(line.split("=", 1)[1])
                if candidate.is_file():
                    python = str(candidate)
                    break

    env = os.environ.copy()
    module_paths = [REPO_ROOT / "src" / "python", build_dir / "src" / "python"]
    existing = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    env["PYTHONPATH"] = os.pathsep.join([str(path) for path in module_paths] + existing)
    library_paths = [str(build_dir)]
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    return Path(python), env


def _run_app(
    app: Path, home: Path, deadline: float, *args: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env.pop("PYTHONPATH", None)
    env["LFS_ASSET_MANAGER_DIR"] = str(home / "assets")
    env["LFS_ASSET_MANAGER_ASSETS_DIR"] = str(home / "assets" / "assets")
    return subprocess.run(
        [str(app), "--headless", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=max(1.0, min(45.0, deadline - time.monotonic())),
        check=False,
    )


def test_resume_uses_compatible_external_dataset_masks_and_embedded_fallback(tmp_path):
    app = _app_path()
    if app is None:
        pytest.skip("LichtFeld Studio executable is not available")

    deadline = time.monotonic() + 55.0
    dataset = tmp_path / "compatible"
    incompatible = tmp_path / "incompatible"
    _make_dataset(dataset, "frame")
    _make_dataset(incompatible, "other")
    home = tmp_path / "home"
    seed_output = tmp_path / "seed"
    seed = _run_app(
        app,
        home,
        deadline,
        "--data-path",
        str(dataset),
        "--images",
        "images",
        "-r",
        "1",
        "--max-width",
        "8",
        "--iter",
        "2",
        "--max-cap",
        "1000",
        "--no-download",
        "--output-path",
        str(seed_output),
    )
    assert seed.returncode == 0, seed.stdout + seed.stderr
    project = seed_output / "project.licht"
    assert project.is_file()

    embedded_project = tmp_path / "embedded" / "project.licht"
    embedded_project.parent.mkdir()
    embedded_project.write_bytes(project.read_bytes())
    python, python_env = _python_environment()
    python_env["HOME"] = str(home)
    embed = subprocess.run(
        [
            str(python),
            "-c",
            "import sys, lichtfeld as lf; "
            "lf.io.set_dataset_reference(sys.argv[1], sys.argv[2], True); "
            "lf.io.embed_dataset_file(sys.argv[1]); "
            "assert lf.io.inspect_project_details(sys.argv[1]).parameters.embedded_dataset_complete",
            str(embedded_project),
            str(dataset),
        ],
        env=python_env,
        capture_output=True,
        text=True,
        timeout=max(1.0, min(45.0, deadline - time.monotonic())),
        check=False,
    )
    assert embed.returncode == 0, embed.stdout + embed.stderr

    moved = tmp_path / "compatible-moved"
    dataset.rename(moved)
    resumed = _run_app(
        app,
        home,
        deadline,
        "--resume",
        str(project),
        "--data-path",
        str(moved),
        "--images",
        "images",
        "-r",
        "1",
        "--max-width",
        "8",
        "--iter",
        "3",
        "--no-download",
        "--output-path",
        str(tmp_path / "moved-resume"),
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr

    conflict = _run_app(
        app,
        home,
        deadline,
        "--resume",
        str(project),
        "--data-path",
        str(incompatible),
        "--images",
        "images",
        "-r",
        "1",
        "--max-width",
        "8",
        "--iter",
        "3",
        "--no-download",
        "--output-path",
        str(tmp_path / "incompatible-resume"),
    )
    assert conflict.returncode != 0, conflict.stdout + conflict.stderr
    assert "not compatible" in conflict.stdout + conflict.stderr

    mask_dir = moved / "masks"
    mask_dir.mkdir()
    for image in sorted((moved / "images").glob("*.png")):
        (mask_dir / image.name).write_bytes(_png(8, 8, (255, 255, 255, 255)))
    masked = _run_app(
        app,
        home,
        deadline,
        "--resume",
        str(project),
        "--data-path",
        str(moved),
        "--images",
        "images",
        "--mask-mode",
        "segment",
        "--log-level",
        "debug",
        "-r",
        "1",
        "--max-width",
        "8",
        "--iter",
        "3",
        "--no-download",
        "--output-path",
        str(tmp_path / "masked-resume"),
    )
    assert masked.returncode == 0, masked.stdout + masked.stderr
    assert "Bound 4 sidecar masks to resumed project cameras" in masked.stdout

    moved.rename(dataset)
    try:
        unavailable = tmp_path / "compatible-offline"
        dataset.rename(unavailable)
        try:
            fallback = _run_app(
                app,
                home,
                deadline,
                "--resume",
                str(embedded_project),
                "--iter",
                "3",
                "--max-width",
                "8",
                "--no-download",
                "--output-path",
                str(tmp_path / "embedded-resume"),
            )
            assert fallback.returncode == 0, fallback.stdout + fallback.stderr
            assert "Embedded dataset extraction completed" in fallback.stdout
        finally:
            unavailable.rename(dataset)
    finally:
        if moved.exists() and not dataset.exists():
            moved.rename(dataset)
