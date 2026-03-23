#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sync and verify committed Python stubs against generated stubs."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import shutil
import sys

VOLATILE_STUBS = frozenset({
    # Generated from live build metadata; intentionally ignored in git.
    "lichtfeld/build_info.pyi",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "sync"))
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--committed", required=True, type=Path)
    return parser.parse_args()


def collect_files(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in VOLATILE_STUBS
    }


def _referenced_generated_files(root: Path) -> set[str]:
    referenced: set[str] = set()
    visited_packages: set[Path] = set()

    def visit_package(package_dir: Path) -> None:
        package_dir = package_dir.resolve()
        if package_dir in visited_packages:
            return
        visited_packages.add(package_dir)

        init_path = package_dir / "__init__.pyi"
        if not init_path.exists():
            return

        referenced.add(init_path.relative_to(root).as_posix())

        py_typed = package_dir / "py.typed"
        if py_typed.exists():
            referenced.add(py_typed.relative_to(root).as_posix())

        tree = ast.parse(init_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or node.level != 1 or node.module is not None:
                continue
            for alias in node.names:
                child_name = alias.name
                child_package = package_dir / child_name
                child_module = package_dir / f"{child_name}.pyi"
                if (child_package / "__init__.pyi").exists():
                    visit_package(child_package)
                elif child_module.exists():
                    referenced.add(child_module.relative_to(root).as_posix())

    for package_init in root.glob("*/__init__.pyi"):
        visit_package(package_init.parent)

    return referenced


def prune_generated_stubs(root: Path) -> None:
    if not root.exists():
        return

    referenced = _referenced_generated_files(root)
    for rel_path, path in collect_files(root).items():
        if rel_path in referenced:
            continue
        path.unlink()

    remove_empty_dirs(root)


def read_normalized(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            continue


def sync_stubs(generated: Path, committed: Path) -> int:
    prune_generated_stubs(generated)
    generated_files = collect_files(generated)
    committed_files = collect_files(committed)

    for rel_path in VOLATILE_STUBS:
        volatile_destination = committed / rel_path
        if volatile_destination.exists():
            volatile_destination.unlink()

    for rel_path in sorted(set(committed_files) - set(generated_files), reverse=True):
        (committed / rel_path).unlink()

    for rel_path, source in generated_files.items():
        destination = committed / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and read_normalized(destination) == read_normalized(source):
            continue
        shutil.copy2(source, destination)

    remove_empty_dirs(committed)
    print(f"Synced {len(generated_files)} stub files into {committed}")
    return 0


def check_stubs(generated: Path, committed: Path) -> int:
    prune_generated_stubs(generated)
    generated_files = collect_files(generated)
    committed_files = collect_files(committed)

    missing = sorted(set(generated_files) - set(committed_files))
    extra = sorted(set(committed_files) - set(generated_files))
    changed = sorted(
        rel_path
        for rel_path in set(generated_files) & set(committed_files)
        if read_normalized(generated_files[rel_path]) != read_normalized(committed_files[rel_path])
    )

    if not missing and not extra and not changed:
        print("Committed stubs match generated stubs.")
        return 0

    print("Committed stubs are out of date.", file=sys.stderr)
    if missing:
        print("Missing files:", file=sys.stderr)
        for rel_path in missing:
            print(f"  {rel_path}", file=sys.stderr)
    if extra:
        print("Extra files:", file=sys.stderr)
        for rel_path in extra:
            print(f"  {rel_path}", file=sys.stderr)
    if changed:
        print("Changed files:", file=sys.stderr)
        for rel_path in changed:
            print(f"  {rel_path}", file=sys.stderr)
    print("Run `cmake --build <build-dir> --target refresh_python_stubs`.", file=sys.stderr)
    return 1


def main() -> int:
    args = parse_args()
    generated = args.generated.resolve()
    committed = args.committed.resolve()

    if not generated.exists():
        print(f"Generated stub directory does not exist: {generated}", file=sys.stderr)
        return 1

    if args.command == "sync":
        committed.mkdir(parents=True, exist_ok=True)
        return sync_stubs(generated, committed)
    return check_stubs(generated, committed)


if __name__ == "__main__":
    raise SystemExit(main())
