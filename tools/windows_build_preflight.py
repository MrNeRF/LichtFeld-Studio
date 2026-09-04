#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fail early on Windows build-contract regressions.

The source checks are dependency-free and run on every host.  The configured
check is Windows-only: it replays affected MSVC compile-database entries with
``/Zs`` so the real Microsoft frontend performs parsing and semantic checks
without creating object files.

This is deliberately not a replacement for a full build.  CUDA compilation,
general linker resolution, code generation, and runtime behavior remain the
responsibility of the normal build and test jobs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import locale
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Sequence

try:
    from error_debt_census import mask_source
except ImportError:  # Imported as tools.windows_build_preflight in self-tests.
    from tools.error_debt_census import mask_source


REPO_ROOT = Path(__file__).resolve().parent.parent
HEADER_SUFFIXES = frozenset({".h", ".hh", ".hpp", ".hxx", ".cuh"})
CXX_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx"})
PROJECT_SOURCE_SUFFIXES = HEADER_SUFFIXES | CXX_SUFFIXES | frozenset({".cu"})
SKIPPED_COMPONENTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        "build",
        "build-debug",
        "external",
        "node_modules",
    }
)
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]', re.MULTILINE)
TYPE_DEFINITION_RE = re.compile(
    r"\b(?:class|struct)\s+(?:LFS_[A-Z_]+_API\s+)?"
    r"(?P<name>[A-Za-z_]\w*)[^;{}]*\{"
)
VISUALIZER_USING_RE = re.compile(
    r"\busing\s+(?P<qualified>lfs\s*::\s*vis"
    r"(?:\s*::\s*[A-Za-z_]\w*)+)\s*;"
)
VISUALIZER_QUALIFIED_CALL_RE = re.compile(
    r"\b(?P<qualified>lfs\s*::\s*vis"
    r"(?:\s*::\s*[A-Za-z_]\w*)+)\s*\("
)
EXPORT_EXEMPT_TOKENS = frozenset(
    {"constexpr", "friend", "inline", "static", "template", "typedef", "using"}
)


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    rule: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.message}"


@dataclass(frozen=True)
class TypeBlock:
    name: str
    open_brace: int
    close_brace: int
    body_depth: int


@dataclass(frozen=True)
class CompileCommand:
    file: Path
    directory: Path
    command: str


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _depths(text: str, opening: str, closing: str) -> list[int]:
    result = [0] * (len(text) + 1)
    depth = 0
    for index, character in enumerate(text):
        result[index] = depth
        if character == opening:
            depth += 1
        elif character == closing:
            depth = max(depth - 1, 0)
    result[len(text)] = depth
    return result


def _matching_delimiter(
    text: str, start: int, opening: str, closing: str
) -> int | None:
    depth = 0
    for index in range(start, len(text)):
        character = text[index]
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _type_blocks(masked: str) -> list[TypeBlock]:
    brace_depth = _depths(masked, "{", "}")
    blocks: list[TypeBlock] = []
    for match in TYPE_DEFINITION_RE.finditer(masked):
        open_brace = masked.find("{", match.start(), match.end())
        close_brace = _matching_delimiter(masked, open_brace, "{", "}")
        if close_brace is None:
            continue
        blocks.append(
            TypeBlock(
                name=match.group("name"),
                open_brace=open_brace,
                close_brace=close_brace,
                body_depth=brace_depth[open_brace] + 1,
            )
        )
    return blocks


def _iter_project_files(root: Path, suffixes: frozenset[str]) -> Iterable[Path]:
    tracked = _git(
        root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "src",
        "include",
        "tests",
    )
    if tracked.returncode == 0:
        for relative_name in tracked.stdout.splitlines():
            path = root / relative_name
            if path.suffix.lower() not in suffixes or not path.is_file():
                continue
            if any(component in SKIPPED_COMPONENTS for component in path.parts):
                continue
            yield path.resolve()
        return

    # Unit-test fixtures need no real Git repository. Keep a filesystem
    # fallback for those and for source archives without Git metadata.
    for base in (root / "src", root / "include", root / "tests"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if any(component in SKIPPED_COMPONENTS for component in path.parts):
                continue
            yield path.resolve()


def _inside_type(position: int, blocks: Sequence[TypeBlock]) -> bool:
    return any(block.open_brace < position < block.close_brace for block in blocks)


def _resolve_visualizer_header(root: Path, include: str) -> Path | None:
    normalized = Path(include.replace("\\", "/"))
    candidates = (
        root / "src" / "visualizer" / normalized,
        root / "src" / "visualizer" / "include" / normalized,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _function_segment(masked: str, name_offset: int) -> tuple[int, str] | None:
    open_paren = masked.find("(", name_offset)
    if open_paren < 0:
        return None
    close_paren = _matching_delimiter(masked, open_paren, "(", ")")
    if close_paren is None:
        return None
    boundaries = [
        (masked.find(character, close_paren), character)
        for character in (";", "{", "}")
    ]
    boundary, character = min(
        (item for item in boundaries if item[0] >= 0),
        default=(-1, ""),
    )
    if boundary < 0 or character == "}":
        return None
    start = max(
        masked.rfind(";", 0, name_offset),
        masked.rfind("{", 0, name_offset),
        masked.rfind("}", 0, name_offset),
    ) + 1
    return start, masked[start : boundary + 1]


def _explicit_visualizer_symbols(masked_test: str) -> set[str]:
    """Return functions that a test explicitly reaches across the DLL boundary.

    Deliberately ignore unqualified calls and broad ``using namespace`` imports:
    a lexical checker cannot reliably distinguish those from methods and local
    helpers.  The configured MSVC pass remains authoritative for C++ semantics.
    """

    qualified = {
        match.group("qualified")
        for pattern in (VISUALIZER_USING_RE, VISUALIZER_QUALIFIED_CALL_RE)
        for match in pattern.finditer(masked_test)
    }
    return {re.sub(r"\s+", "", symbol).split("::")[-1] for symbol in qualified}


def check_visualizer_test_exports(root: Path) -> list[Finding]:
    """Require exports for explicitly referenced visualizer free functions."""

    findings: set[Finding] = set()
    tests_root = root / "tests"
    if not tests_root.exists():
        return []

    for test_path in tests_root.rglob("*.cpp"):
        raw_test = test_path.read_text(encoding="utf-8", errors="replace")
        referenced_names = _explicit_visualizer_symbols(mask_source(raw_test))
        if not referenced_names:
            continue
        headers = {
            header
            for include in INCLUDE_RE.findall(raw_test)
            if (header := _resolve_visualizer_header(root, include)) is not None
        }
        for header in headers:
            raw_header = header.read_text(encoding="utf-8", errors="replace")
            masked_header = mask_source(raw_header)
            blocks = _type_blocks(masked_header)
            for name in referenced_names:
                pattern = re.compile(rf"\b{re.escape(name)}\s*\(")
                unexported: list[re.Match[str]] = []
                header_available = False
                for match in pattern.finditer(masked_header):
                    if _inside_type(match.start(), blocks):
                        continue
                    prefix = masked_header[max(0, match.start() - 2) : match.start()]
                    if prefix.endswith("::") or prefix.endswith("->"):
                        continue
                    function = _function_segment(masked_header, match.start())
                    if function is None:
                        continue
                    _start, segment = function
                    tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", segment))
                    if (
                        tokens & EXPORT_EXEMPT_TOKENS
                        or "operator" in tokens
                        or "LFS_VIS_API" in tokens
                    ):
                        header_available = True
                        break
                    unexported.append(match)

                if header_available or not unexported:
                    continue
                match = unexported[0]
                relative = header.relative_to(root).as_posix()
                findings.add(
                    Finding(
                        path=relative,
                        line=_line_number(raw_header, match.start()),
                        rule="visualizer-test-dll-export",
                        message=(
                            f"free function {name} is called by "
                            f"{test_path.relative_to(root).as_posix()} but its "
                            "declaration lacks LFS_VIS_API"
                        ),
                    )
                )
    return sorted(findings)


def run_source_checks(root: Path) -> list[Finding]:
    return check_visualizer_test_exports(root)


def load_compile_commands(path: Path) -> list[CompileCommand]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read compile database {path}: {error}") from error
    if not isinstance(payload, list):
        raise ValueError(f"compile database {path} must contain a JSON array")

    commands: list[CompileCommand] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"compile database entry {index} is not an object")
        command = entry.get("command")
        arguments = entry.get("arguments")
        file_value = entry.get("file")
        directory = entry.get("directory")
        if command is None and isinstance(arguments, list) and all(
            isinstance(value, str) for value in arguments
        ):
            command = subprocess.list2cmdline(arguments)
        if not all(
            isinstance(value, str) and value
            for value in (command, file_value, directory)
        ):
            raise ValueError(
                f"compile database entry {index} requires file and directory strings "
                "plus a command string or arguments array"
            )
        directory_path = _lexical_absolute(Path(directory))
        commands.append(
            CompileCommand(
                file=_lexical_absolute(Path(file_value), directory_path),
                directory=directory_path,
                command=command,
            )
        )
    return commands


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _valid_base(root: Path, candidate: str | None) -> str | None:
    if not candidate or set(candidate) == {"0"}:
        return None
    result = _git(root, "rev-parse", "--verify", f"{candidate}^{{commit}}")
    return candidate if result.returncode == 0 else None


def discover_changed_files(root: Path, base: str | None) -> tuple[set[Path], bool]:
    """Return changed paths and whether missing base forced full coverage."""

    requested_base = base or os.environ.get("LFS_PREFLIGHT_BASE")
    valid_base = _valid_base(root, requested_base)
    force_all = bool(requested_base and valid_base is None)
    if valid_base is not None:
        merge_base = _git(root, "merge-base", valid_base, "HEAD")
        if merge_base.returncode != 0:
            force_all = True
            valid_base = None
        else:
            diff_base = merge_base.stdout.strip()
    else:
        diff_base = "HEAD"
        if os.environ.get("GITHUB_ACTIONS") == "true" and not requested_base:
            force_all = True

    changed: set[Path] = set()
    if not force_all:
        # Treat renames as delete/add so old include spellings remain reachable.
        diff = _git(root, "diff", "--name-only", "-z", "--no-renames", diff_base, "--")
        if diff.returncode != 0:
            force_all = True
        else:
            for path in diff.stdout.split("\0"):
                if path:
                    changed.add((root / path).resolve())
            untracked = _git(root, "ls-files", "-z", "--others", "--exclude-standard")
            if untracked.returncode == 0:
                for path in untracked.stdout.split("\0"):
                    if path:
                        changed.add((root / path).resolve())
    return changed, force_all


def _lexical_absolute(path: Path, base: Path | None = None) -> Path:
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    return Path(os.path.abspath(os.path.normpath(path)))


def _iter_local_includes(
    root: Path, project_files: Iterable[Path]
) -> Iterable[tuple[Path, str, bool]]:
    grep = _git(
        root,
        "grep",
        "--untracked",
        "-n",
        "-I",
        "-E",
        r'^[[:space:]]*#[[:space:]]*include[[:space:]]*["<][^">]+[">]',
        "--",
        "src",
        "include",
        "tests",
    )
    if grep.returncode in (0, 1):
        for line in grep.stdout.splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            source = (root / parts[0]).resolve()
            match = INCLUDE_RE.search(parts[2])
            if match is not None:
                yield source, match.group(1), match.group(0).endswith('"')
        return

    # Source archives and unit-test fixtures may have no usable Git index.
    for source in project_files:
        raw = source.read_text(encoding="utf-8", errors="replace")
        for match in INCLUDE_RE.finditer(raw):
            yield source, match.group(1), match.group(0).endswith('"')


def build_reverse_include_graph(
    root: Path, extra_dependencies: Iterable[Path] = (), build_dir: Path | None = None
) -> dict[Path, set[Path]]:
    project_files = set(_iter_project_files(root, PROJECT_SOURCE_SUFFIXES))
    generated_root = (build_dir or root / "build") / "include"
    generated_files = {
        path.resolve() for path in generated_root.rglob("*")
        if path.is_file() and path.suffix.lower() in HEADER_SUFFIXES
    }
    dependency_files = project_files | generated_files | {
        path.resolve() for path in extra_dependencies
    }
    suffix_index: dict[str, list[Path]] = defaultdict(list)
    for path in dependency_files:
        try:
            relative = path.relative_to(root).as_posix().lower()
        except ValueError:
            # The selected build directory can live outside the checkout.
            relative = path.relative_to(generated_root).as_posix().lower()
        parts = relative.split("/")
        for index in range(len(parts)):
            suffix_index["/".join(parts[index:])].append(path)

    reverse: dict[Path, set[Path]] = defaultdict(set)
    for source, include, quoted in _iter_local_includes(root, project_files):
        include_path = Path(include.replace("\\", "/"))
        direct = _lexical_absolute(source.parent / include_path)
        resolved: Path | None = direct if quoted and direct in dependency_files else None
        # config.h from build/include must not alias Vulkan's private config.h.
        # Prefer the configured header after a quoted include's local directory.
        generated = _lexical_absolute(generated_root / include_path)
        if resolved is None and generated in generated_files:
            resolved = generated
        if resolved is None:
            matches = suffix_index.get(include.replace("\\", "/").lower(), [])
            if len(matches) == 1:
                resolved = matches[0]
        if resolved is not None and resolved in dependency_files:
            reverse[resolved].add(source)
    return reverse


def _is_project_compile_command(root: Path, command: CompileCommand) -> bool:
    try:
        relative = command.file.relative_to(root)
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] in {"src", "include", "tests"}


def select_compile_commands(
    root: Path,
    commands: Sequence[CompileCommand],
    changed: set[Path],
    force_all: bool,
    build_dir: Path | None = None,
) -> list[CompileCommand]:
    cxx_commands = [
        command
        for command in commands
        if command.file.suffix.lower() in CXX_SUFFIXES
        and _is_project_compile_command(root, command)
    ]
    if force_all:
        return cxx_commands

    command_files = {command.file for command in cxx_commands}
    selected_files = {path for path in changed if path in command_files}
    changed_headers = {path for path in changed if path.suffix.lower() in HEADER_SUFFIXES}
    if changed_headers:
        reverse = build_reverse_include_graph(root, changed_headers, build_dir)
        queue = deque(changed_headers)
        visited = set(changed_headers)
        while queue:
            dependency = queue.popleft()
            for consumer in reverse.get(dependency, ()):
                if consumer in visited:
                    continue
                visited.add(consumer)
                queue.append(consumer)
                if consumer in command_files:
                    selected_files.add(consumer)

    selected = [command for command in cxx_commands if command.file in selected_files]
    return selected


def _display_source_path(root: Path, source: Path) -> str:
    try:
        return source.relative_to(root).as_posix()
    except ValueError:
        return source.as_posix()


def _print_selected_compile_commands(
    root: Path, commands: Sequence[CompileCommand]
) -> None:
    source_counts = Counter(
        _display_source_path(root, command.file) for command in commands
    )
    print(
        "MSVC preflight: selected "
        f"{len(source_counts)} source file(s) for {len(commands)} compile command(s):"
    )
    for source in sorted(source_counts, key=lambda path: (path.lower(), path)):
        count = source_counts[source]
        duplicate_note = f" ({count} compile commands)" if count > 1 else ""
        print(f"  - {source}{duplicate_note}")


def _is_msvc_command(command: CompileCommand) -> bool:
    lowered = command.command.lower()
    return (
        re.search(r'(?:^|[\\/"\s])cl(?:\.exe)?(?=["\s]|$)', lowered)
        is not None
        and "nvcc" not in lowered
    )


def _windows_command_tokens(command: str) -> list[str]:
    """Split at unquoted whitespace without unescaping CMake's Windows arguments."""
    tokens = []
    start = 0
    quoted = False
    backslashes = 0
    for index, character in enumerate(command):
        if character == '"' and backslashes % 2 == 0:
            quoted = not quoted
        if character.isspace() and not quoted:
            if start < index:
                tokens.append(command[start:index])
            start = index + 1
        backslashes = backslashes + 1 if character == "\\" else 0
    if start < len(command):
        tokens.append(command[start:])
    return tokens


def _msvc_syntax_command(command: str) -> str:
    # A syntax-only invocation creates no object for a compiler cache to store.
    # Bypass a direct sccache launcher and suppress the dependency-list stream,
    # while preserving the exact compiler, defines, and include paths selected
    # by CMake.
    without_launcher = re.sub(
        r'^\s*(?:"[^"]*sccache(?:\.exe)?"|[^\s"]*sccache(?:\.exe)?)\s+',
        "",
        command,
        count=1,
        flags=re.IGNORECASE,
    )
    # Preserve command spelling/quoting. PCH switches can carry an attached or
    # separate (possibly quoted) filename; a bare switch must not eat /c or /I.
    kept = []
    skip_argument = False
    for argument in _windows_command_tokens(without_launcher):
        option = argument[1:-1] if argument.startswith('"') and argument.endswith('"') else argument
        if skip_argument:
            skip_argument = False
            if not option.startswith(("/", "-")):
                continue
        if option.lower() == "/showincludes":
            continue
        if re.match(r"^[-/](?:Yc|Yu|Fp)", option):
            skip_argument = len(option) == 3
            continue
        kept.append(argument)
    return " ".join([*kept, "/Zs"])


def _run_one_syntax_check(command: CompileCommand) -> tuple[CompileCommand, int, str]:
    encoding = locale.getpreferredencoding(False) or "utf-8"
    result = subprocess.run(
        _msvc_syntax_command(command.command),
        cwd=command.directory,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        encoding=encoding,
        errors="replace",
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    return command, result.returncode, output


def run_msvc_syntax_checks(
    commands: Sequence[CompileCommand], jobs: int
) -> list[tuple[CompileCommand, int, str]]:
    failures: list[tuple[CompileCommand, int, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        futures = {
            executor.submit(_run_one_syntax_check, command): command
            for command in commands
        }
        for future in as_completed(futures):
            command, returncode, output = future.result()
            if returncode != 0:
                failures.append((command, returncode, output))
    return sorted(failures, key=lambda item: item[0].file.as_posix().lower())


def _print_source_result(findings: Sequence[Finding]) -> None:
    if findings:
        print("Windows build source preflight failed:", file=sys.stderr)
        for finding in findings:
            print(f"  {finding.render()}", file=sys.stderr)
        return
    print("Windows build source preflight passed")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="run dependency-free source contract checks only",
    )
    parser.add_argument(
        "--skip-source-checks",
        action="store_true",
        help="skip source checks already performed by an earlier CI gate",
    )
    parser.add_argument(
        "--compile-commands",
        type=Path,
        help="compile_commands.json used for configured MSVC /Zs checks",
    )
    parser.add_argument(
        "--changed-since",
        help="Git revision used to select affected translation units",
    )
    parser.add_argument(
        "--all-commands",
        action="store_true",
        help="check every MSVC C/C++ entry in the compile database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print selected translation units without invoking MSVC",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(os.cpu_count() or 1, 4),
        help="maximum parallel MSVC syntax checks (default: up to 4)",
    )
    parser.add_argument(
        "--max-commands",
        type=int,
        default=0,
        help=(
            "skip configured replay when more commands are affected; "
            "zero means unlimited (default: 0)"
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.source_only and arguments.compile_commands is not None:
        parser.error("--source-only cannot be combined with --compile-commands")
    if arguments.source_only and arguments.skip_source_checks:
        parser.error("--source-only cannot be combined with --skip-source-checks")
    if arguments.all_commands and arguments.compile_commands is None:
        parser.error("--all-commands requires --compile-commands")
    if arguments.dry_run and arguments.compile_commands is None:
        parser.error("--dry-run requires --compile-commands")
    if arguments.jobs < 1:
        parser.error("--jobs must be at least 1")
    if arguments.max_commands < 0:
        parser.error("--max-commands cannot be negative")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    root = arguments.root.resolve()
    if not (root / ".git").exists():
        print(f"error: repository root does not contain .git: {root}", file=sys.stderr)
        return 2

    if not arguments.skip_source_checks:
        findings = run_source_checks(root)
        _print_source_result(findings)
        if findings:
            return 1
    if arguments.source_only or arguments.compile_commands is None:
        return 0

    if arguments.all_commands:
        changed, forced_all = set(), True
    else:
        changed, forced_all = discover_changed_files(root, arguments.changed_since)

    database = arguments.compile_commands.resolve()
    try:
        commands = load_compile_commands(database)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    selected = select_compile_commands(root, commands, changed, forced_all, database.parent)

    if not selected:
        print("MSVC preflight: no affected configured C/C++ translation units")
        return 0
    if arguments.dry_run:
        _print_selected_compile_commands(root, selected)
        return 0
    if arguments.max_commands and len(selected) > arguments.max_commands:
        annotation = "::warning::" if os.environ.get("GITHUB_ACTIONS") == "true" else "warning: "
        print(
            annotation + "MSVC preflight: configured replay skipped because "
            f"{len(selected)} commands exceed the {arguments.max_commands}-command "
            "budget; the full build remains authoritative"
        )
        return 0
    if os.name != "nt":
        print("error: configured MSVC checks require Windows", file=sys.stderr)
        return 2

    supported = [command for command in selected if _is_msvc_command(command)]
    if not supported:
        print("error: no selected compile command invokes MSVC cl.exe", file=sys.stderr)
        return 2

    for command in selected:
        if not _is_msvc_command(command):
            print(
                "warning: skipping non-MSVC compile command for "
                f"{_display_source_path(root, command.file)}",
                file=sys.stderr,
            )
    if forced_all:
        print("MSVC preflight: checking all configured C/C++ translation units")
    else:
        print(f"MSVC preflight: checking {len(selected)} affected compile command(s)")
    _print_selected_compile_commands(root, supported)
    failures = run_msvc_syntax_checks(supported, arguments.jobs)
    if failures:
        print("MSVC configured preflight failed:", file=sys.stderr)
        for command, returncode, output in failures:
            print(
                "\n--- "
                f"{_display_source_path(root, command.file)} (exit {returncode}) ---",
                file=sys.stderr,
            )
            if output:
                print(output, file=sys.stderr)
        return 1
    print(f"MSVC configured preflight passed ({len(supported)} command(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
