#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Guard the shared native filesystem boundary against raw std::filesystem I/O.

This conservative source check complements runtime tests; it is not a C++ parser
or a guarantee about third-party libraries. Logical path construction is allowed.
"""
from pathlib import Path
import ast
import re

ROOT = Path(__file__).resolve().parent.parent
OPERATIONS = (
    "exists|is_regular_file|is_directory|is_symlink|is_empty|status|symlink_status|"
    "file_size|hard_link_count|space|remove|remove_all|create_directory|"
    "create_directories|resize_file|permissions|last_write_time|copy_file|copy|"
    "copy_symlink|rename|equivalent|create_hard_link|create_symlink|"
    "create_directory_symlink|canonical|weakly_canonical|relative|read_symlink|"
    "current_path|temp_directory_path|directory_entry|directory_iterator|recursive_directory_iterator"
)
RAW_IO = re.compile(r"\bstd::filesystem::(?:" + OPERATIONS + r")\s*\(")
RAW_ALIAS = re.compile(r"namespace\s+\w+\s*=\s*std::filesystem\s*;")
RAW_ENTRY = re.compile(r"\bstd::filesystem::directory_entry\s*(?:\w+\s*)?\(")
RAW_ITERATOR = re.compile(r"\bstd::filesystem::(?:recursive_)?directory_iterator\s*(?:\w+\s*)?\(")
RAW_JSON_PATH = re.compile(r'\bstd::filesystem::path\s+\w+\s*=\s*args\[[^\]]+\]\.get<std::string>\(\)')
RAW_STREAM = re.compile(r"\bstd::(?:ifstream|ofstream|fstream)\s*(?:[A-Za-z_]\w*\s*)?\(")


def violations(source: str) -> list[tuple[int, str]]:
    found = []
    lines = source.splitlines()
    for number, line in enumerate(lines, 1):
        code = line.split("//", 1)[0]
        if code.lstrip().startswith(("#", "*")):
            continue
        rejected = False
        for pattern in (RAW_IO, RAW_ALIAS, RAW_STREAM, RAW_ENTRY, RAW_ITERATOR, RAW_JSON_PATH):
            for match in pattern.finditer(code):
                tail = code[match.end():] + "\n" + "\n".join(lines[number:number + 2])
                # Standard streams and iterator objects are intentional when their
                # filename/root has already crossed the shared native boundary.
                adapted = re.match(r"\s*(?:lfs::core::)?native_fs::(?:native_path|traversal_path)\s*\(", tail)
                empty_iterator = "directory_iterator" in match.group() and re.match(r"\s*\)", tail)
                if not adapted and not empty_iterator:
                    rejected = True
        if rejected:
            found.append((number, line.strip()))
    return found


def main() -> int:
    failures = []
    for path in sorted((ROOT / "src").rglob("*")):
        if path.suffix == ".py" and path.is_relative_to(ROOT / "src/python"):
            if path.name == "_lfs_path_io.py" or "scripts" in path.relative_to(ROOT / "src/python").parts:
                continue
            source = path.read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and node.module == "pathlib" and any(name.name == "Path" for name in node.names):
                    failures.append((path.relative_to(ROOT).as_posix(), node.lineno, "use the internal I/O-aware Path; pure logical path types remain allowed"))
            continue
        if path.suffix not in {".cpp", ".hpp", ".h", ".cc", ".cu", ".cuh"}:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative == "src/core/include/core/native_fs.hpp":
            continue
        failures.extend((relative, line, text) for line, text in violations(path.read_text(encoding="utf-8")))
    for path, line, text in failures:
        print(f"{path}:{line}: use core/native_fs.hpp at the I/O boundary: {text}")
    if not failures:
        print("No raw standard filesystem boundaries found.")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
