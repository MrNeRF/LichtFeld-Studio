# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Lexical Phase 0 census for error-handling migration debt.

This scanner is deliberately lexical/regex-based and is NOT an AST checker. It
masks comments and literals, then applies conservative source-text heuristics;
inactive preprocessor branches are still visible and some expression shapes are
intentionally under-counted to avoid false positives. A real AST-based
``clang-tidy`` project check is a later phase described in
``.codex_tmp/error-architecture-analysis.md`` section 5.12.

CUDA driver API calls (the lowercase ``cu*`` family) are outside Phase 0. CUDA
status checking only recognizes bare CUDA runtime statements, and the
CUDA-language Result rule is extension-based rather than CMake-aware.

The baseline gate is a two-sided count ratchet: increases list new locations,
decreases fail as ``stale-baseline`` until the baseline is regenerated in the
same change (an honest fix and a matcher-evading refactor are indistinguishable
to a lexical scanner, so both must be acknowledged explicitly).
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "src"
SOURCE_EXTENSIONS = frozenset({".cpp", ".hpp", ".h", ".cu", ".cuh"})
CUDA_SOURCE_EXTENSIONS = frozenset({".cu", ".cuh"})

STRUCTURAL_EXCLUDED_COMPONENTS = frozenset(
    {"external", "third_party", "thirdparty", "vendor"}
)
TEST_COMPONENTS = frozenset({"test", "tests"})

CUDA_STATUS_MACROS = frozenset(
    {
        "LFS_CUDA_CHECK",
        "LFS_CUDA_CHECK_MSG",
        "LFS_ENSURE_CUDA_SUCCESS",
        "LFS_ENSURE_CUDA_SUCCESS_MSG",
        "LFS_ENSURE_CUDA_SUCCESS_STATE",
    }
)
INTENTIONAL_LAST_ERROR_CALLS = frozenset(
    {"cudaGetLastError", "cudaPeekAtLastError"}
)


@dataclass(frozen=True, order=True)
class Hit:
    """One lexical policy hit at a stable source location."""

    file: str
    line: int
    rule: str
    module: str

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class ScannedFile:
    """Masked contents and stable path metadata for one source file."""

    path: Path
    relative_to_root: Path
    file: str
    module: str
    masked: str
    raw: str
    line_starts: tuple[int, ...]

    def line_number(self, offset: int) -> int:
        return bisect_right(self.line_starts, offset)

    def hit(self, offset: int, rule: str) -> Hit:
        return Hit(
            file=self.file,
            line=self.line_number(offset),
            rule=rule,
            module=self.module,
        )

    def line_window(self, first_line: int, last_line: int) -> str:
        """Return an inclusive 1-based line range from the masked source."""

        first_offset = self.line_starts[max(first_line - 1, 0)]
        if last_line < len(self.line_starts):
            last_offset = self.line_starts[last_line]
        else:
            last_offset = len(self.masked)
        return self.masked[first_offset:last_offset]


RuleChecker = Callable[[ScannedFile], list[Hit]]


@dataclass(frozen=True)
class RuleSpec:
    """Registration metadata for a census rule."""

    rule_id: str
    description: str
    checker: RuleChecker
    extensions: frozenset[str] = SOURCE_EXTENSIONS
    include_test_paths: bool = False


def mask_source(text: str) -> str:
    """Blank comments and literals without changing length or newline offsets.

    Raw strings are handled on a best-effort basis. If an apparent raw literal
    has an opener but no matching terminator, its source is left unchanged from
    that point so malformed input cannot make the rest of the file disappear.
    """

    masked = list(text)
    length = len(text)

    def blank(first: int, last: int) -> None:
        for position in range(first, last):
            if text[position] not in "\r\n":
                masked[position] = " "

    index = 0
    while index < length:
        if text.startswith("//", index):
            end = text.find("\n", index + 2)
            if end == -1:
                end = length
            blank(index, end)
            index = end
            continue

        if text.startswith("/*", index):
            terminator = text.find("*/", index + 2)
            end = length if terminator == -1 else terminator + 2
            blank(index, end)
            index = end
            continue

        raw_boundary = index == 0 or not (
            text[index - 1].isalnum() or text[index - 1] == "_"
        )
        if raw_boundary and text.startswith('R"', index):
            opening_paren = text.find("(", index + 2)
            newline = text.find("\n", index + 2)
            if opening_paren != -1 and (newline == -1 or opening_paren < newline):
                delimiter = text[index + 2 : opening_paren]
                terminator_text = ")" + delimiter + '"'
                terminator = text.find(terminator_text, opening_paren + 1)
                if terminator == -1:
                    return "".join(masked[:index]) + text[index:]
                end = terminator + len(terminator_text)
                blank(index, end)
                index = end
                continue

        if text[index] in {'"', "'"}:
            quote = text[index]
            # C++14 digit separators (e.g. 100'000, 0xAB'CD): a quote directly
            # flanked by hex/decimal digits is not a character literal — skip it
            # instead of opening a literal span. Separators are adjacent by
            # definition, so only the immediate neighbors matter.
            if (
                quote == "'"
                and index >= 1
                and index + 1 < length
                and text[index - 1] in "0123456789abcdefABCDEF"
                and text[index + 1] in "0123456789abcdefABCDEF"
            ):
                index += 1
                continue
            end = index + 1
            while end < length:
                if text[end] == "\\":
                    end += 2
                    continue
                if text[end] == quote:
                    end += 1
                    break
                end += 1
            end = min(end, length)
            blank(index, end)
            index = end
            continue

        index += 1

    return "".join(masked)


def _line_starts(text: str) -> tuple[int, ...]:
    return (0, *(index + 1 for index, char in enumerate(text) if char == "\n"))


def _find_matching(
    text: str, opening: int, opening_char: str, closing_char: str
) -> int | None:
    if opening >= len(text) or text[opening] != opening_char:
        return None

    depth = 0
    for index in range(opening, len(text)):
        char = text[index]
        if char == opening_char:
            depth += 1
        elif char == closing_char:
            depth -= 1
            if depth == 0:
                return index
    return None


def _previous_non_whitespace(text: str, before: int) -> int | None:
    index = before - 1
    while index >= 0 and text[index].isspace():
        index -= 1
    return index if index >= 0 else None


def _next_non_whitespace(text: str, after: int) -> int | None:
    index = after
    while index < len(text) and text[index].isspace():
        index += 1
    return index if index < len(text) else None


def _identifier_ending_at(text: str, end: int) -> str:
    start = end
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
        start -= 1
    return text[start:end]


def _is_own_macro_definition(source: ScannedFile, offset: int, name: str) -> bool:
    line_start = source.masked.rfind("\n", 0, offset) + 1
    line_end = source.masked.find("\n", offset)
    if line_end == -1:
        line_end = len(source.masked)
    return source.masked[line_start:line_end].lstrip().startswith(f"#define {name}")


EXPECTED_STRING_RE = re.compile(r"\bstd::expected<")


def check_expected_string(source: ScannedFile) -> list[Hit]:
    """Find std::expected whose last top-level argument is std::string."""

    hits: list[Hit] = []
    for match in EXPECTED_STRING_RE.finditer(source.masked):
        argument_start = match.end()
        last_argument_start = argument_start
        depth = 0
        closing: int | None = None

        for index in range(argument_start, len(source.masked)):
            char = source.masked[index]
            if char == "<":
                depth += 1
            elif char == ">":
                if depth == 0:
                    closing = index
                    break
                depth -= 1
            elif char == "," and depth == 0:
                last_argument_start = index + 1

        if closing is None:
            continue
        if source.masked[last_argument_start:closing].strip() == "std::string":
            hits.append(source.hit(match.start(), "expected-string"))
    return hits


CUDA_RUNTIME_CALL_RE = re.compile(r"\b(cuda[A-Z]\w*)\s*\(")


def _is_cuda_call_wrapped_or_consumed(source: ScannedFile, call_start: int) -> bool:
    previous = _previous_non_whitespace(source.masked, call_start)
    if previous is None:
        return False

    if source.masked[previous] == "(":
        macro_end = _previous_non_whitespace(source.masked, previous)
        if macro_end is not None:
            macro = _identifier_ending_at(source.masked, macro_end + 1)
            if macro in CUDA_STATUS_MACROS:
                return True

    if source.masked[previous] == "=":
        is_comparison = previous > 0 and source.masked[previous - 1] in "=!<>"
        if not is_comparison:
            return True

    preceding_word = _identifier_ending_at(source.masked, previous + 1)
    return preceding_word == "return"


def check_discarded_status_macro_free_cuda(source: ScannedFile) -> list[Hit]:
    """Find conservative bare-statement CUDA runtime status discards."""

    hits: list[Hit] = []
    for match in CUDA_RUNTIME_CALL_RE.finditer(source.masked):
        name = match.group(1)
        if name in INTENTIONAL_LAST_ERROR_CALLS:
            continue

        opening_paren = match.end() - 1
        closing_paren = _find_matching(source.masked, opening_paren, "(", ")")
        if closing_paren is None:
            continue

        if _is_cuda_call_wrapped_or_consumed(source, match.start(1)):
            continue

        after_call = _next_non_whitespace(source.masked, closing_paren + 1)
        if after_call is None:
            continue
        if any(
            source.masked.startswith(operator, after_call)
            for operator in ("==", "!=", "<=", ">=", "<", ">")
        ):
            continue
        if source.masked[after_call] != ";":
            continue

        previous = _previous_non_whitespace(source.masked, match.start(1))
        if previous is None or source.masked[previous] in ";{}":
            hits.append(source.hit(match.start(1), "discarded-status-macro-free-cuda"))
    return hits


RAW_VK_CHECK_RE = re.compile(r"\b(_THROW_ERROR|LFS_VK_CHECK_MSG)\s*\(")


def check_raw_vk_check(source: ScannedFile) -> list[Hit]:
    hits: list[Hit] = []
    for match in RAW_VK_CHECK_RE.finditer(source.masked):
        name = match.group(1)
        if not _is_own_macro_definition(source, match.start(1), name):
            hits.append(source.hit(match.start(1), "raw-vk-check"))
    return hits


EMPTY_CATCH_RE = re.compile(
    r"\bcatch\s*\(\s*(?:\.\.\.|const\s+std::exception\s*&\s*\w*)\s*\)"
)
THROW_KEYWORD_RE = re.compile(r"\bthrow\b")


def check_empty_catch(source: ScannedFile) -> list[Hit]:
    """Find empty or unreviewed catch-all/standard-exception handlers."""

    hits: list[Hit] = []
    for match in EMPTY_CATCH_RE.finditer(source.masked):
        opening_brace = _next_non_whitespace(source.masked, match.end())
        if opening_brace is None or source.masked[opening_brace] != "{":
            continue
        closing_brace = _find_matching(source.masked, opening_brace, "{", "}")
        if closing_brace is None:
            continue

        body = source.masked[opening_brace + 1 : closing_brace]
        if "LFS-CENSUS-OK(empty-catch)" in source.raw[opening_brace + 1 : closing_brace]:
            continue
        reviewed = (
            "LOG_" in body
            or THROW_KEYWORD_RE.search(body) is not None
            or "run_guarded" in body
            or "best_effort_diagnostic" in body
            or "make_error" in body
            or "std::unexpected" in body
            # Phase 9 Python-callback containment primitives: they extract the
            # error and report it (ErrorBus/log) — a contained catch, not a swallow.
            or "contain_python_callback" in body
            or "contain_cxx_callback" in body
            # Phase 10 task-boundary normalization: classifies the active
            # exception into a typed Error, reported/returned — a contained catch.
            or "normalize_current_exception" in body
        )
        if not body.strip() or not reviewed:
            hits.append(source.hit(match.start(), "empty-catch"))
    return hits


LOCAL_DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)")


def check_local_check_macro(source: ScannedFile) -> list[Hit]:
    if source.relative_to_root.parts[:3] == ("core", "include", "core"):
        return []

    hits: list[Hit] = []
    offset = 0
    for line in source.masked.splitlines(keepends=True):
        match = LOCAL_DEFINE_RE.match(line)
        if match is not None:
            name = match.group(1)
            if "ASSERT" in name or "CHECK" in name:
                hits.append(source.hit(offset + match.start(1), "local-check-macro"))
        offset += len(line)
    return hits


FATAL_INVARIANT_RE = re.compile(r"\bLFS_FATAL_INVARIANT\s*\(")


def check_fatal_invariant_site(source: ScannedFile) -> list[Hit]:
    hits: list[Hit] = []
    for match in FATAL_INVARIANT_RE.finditer(source.masked):
        if not _is_own_macro_definition(
            source, match.start(), "LFS_FATAL_INVARIANT"
        ):
            hits.append(source.hit(match.start(), "fatal-invariant-site"))
    return hits


VK_WAIT_RE = re.compile(r"\b(?:vkWait\w*|vkAcquire\w*)\s*\(")


def check_vk_infinite_wait(source: ScannedFile) -> list[Hit]:
    hits: list[Hit] = []
    for match in VK_WAIT_RE.finditer(source.masked):
        opening_paren = match.end() - 1
        closing_paren = _find_matching(source.masked, opening_paren, "(", ")")
        if closing_paren is None:
            continue
        arguments = source.masked[opening_paren + 1 : closing_paren]
        if "UINT64_MAX" in arguments or "kWaitForeverNs" in arguments:
            hits.append(source.hit(match.start(), "vk-infinite-wait"))
    return hits


KERNEL_CHECK_MARKERS = (
    "LFS_CUDA_CHECK",
    "LFS_CUDA_LAUNCH_CHECK",
    "LFS_ENSURE_CUDA_SUCCESS",
    "cudaGetLastError",
    "cudaPeekAtLastError",
)


def check_unchecked_kernel_launch(source: ScannedFile) -> list[Hit]:
    hits: list[Hit] = []
    for match in re.finditer(r"<<<", source.masked):
        configuration_end = source.masked.find(">>>", match.end())
        if configuration_end == -1:
            continue
        arguments_start = _next_non_whitespace(source.masked, configuration_end + 3)
        if arguments_start is None or source.masked[arguments_start] != "(":
            continue
        arguments_end = _find_matching(source.masked, arguments_start, "(", ")")
        if arguments_end is None:
            continue
        statement_end = _next_non_whitespace(source.masked, arguments_end + 1)
        if statement_end is None or source.masked[statement_end] != ";":
            continue

        launch_line = source.line_number(match.start())
        end_line = source.line_number(statement_end)
        check_window = source.line_window(launch_line, end_line + 3)
        if not any(marker in check_window for marker in KERNEL_CHECK_MARKERS):
            hits.append(source.hit(match.start(), "unchecked-kernel-launch"))
    return hits


RESULT_IN_CU_RE = re.compile(r"\blfs::Result<|\bstd::expected\b")


def check_result_in_cu(source: ScannedFile) -> list[Hit]:
    return [
        source.hit(match.start(), "result-in-cu")
        for match in RESULT_IN_CU_RE.finditer(source.masked)
    ]


CALL_LIKE_OPEN_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*(?:<[^;{}()]*>)?\s*\("
)
CUDA_LAUNCH_OPEN_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*<<<[^;{}]*?>>?>\s*\("
)
NON_CALL_KEYWORDS = frozenset(
    {
        "alignas",
        "alignof",
        "catch",
        "decltype",
        "dynamic_cast",
        "for",
        "if",
        "noexcept",
        "reinterpret_cast",
        "requires",
        "sizeof",
        "static_assert",
        "static_cast",
        "switch",
        "typeid",
        "while",
    }
)
POINTER_BASE_RE = re.compile(
    r"\b([A-Za-z_]\w*(?:\s*\.\s*get\s*\(\s*\))?)\s*"
    r"(?:\.\s*template\s+ptr\s*<[^;(){}]*>|\.\s*ptr\s*<[^;(){}]*>|"
    r"\.\s*data_ptr)\s*\("
)
BARE_POINTER_RE = re.compile(
    r"(?<![.\w>])(?:template\s+)?(?:ptr\s*<[^;(){}]*>|data_ptr)\s*\("
)
FACTORY_ASSIGNMENT_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*=\s*"
    r"(?:(?:[A-Za-z_]\w*)::)*"
    r"(?:empty|zeros|ones|full|full_like|empty_like|zeros_like)\s*\("
)
PIN_OPERANDS_RE = re.compile(r"\bpin_operands\s*\(")
UNPINNED_MULTI_CAPTURE_ESCAPE = "LFS-CENSUS-OK(unpinned-multi-capture)"


def _enclosing_block_stacks(
    text: str, offsets: Sequence[int]
) -> dict[int, tuple[int, ...]]:
    """Map offsets to their unmatched opening-brace ancestry."""

    result: dict[int, tuple[int, ...]] = {}
    stack: list[int] = []
    cursor = 0
    for offset in sorted(set(offsets)):
        while cursor < offset:
            if text[cursor] == "{":
                stack.append(cursor)
            elif text[cursor] == "}" and stack:
                stack.pop()
            cursor += 1
        result[offset] = tuple(stack)
    return result


def _same_block_prefix(text: str, block_start: int, before: int) -> str:
    """Return prior same-depth text with completed child blocks blanked."""

    first = block_start + 1 if text[block_start : block_start + 1] == "{" else block_start
    prefix = list(text[first:before])
    depth = 0
    for index, char in enumerate(prefix):
        if char == "{":
            depth += 1
            prefix[index] = " "
        elif char == "}":
            prefix[index] = " "
            if depth > 0:
                depth -= 1
        elif depth > 0 and char not in "\r\n":
            prefix[index] = " "
    return "".join(prefix)


def _visible_enclosing_prefix(
    text: str, block_stack: Sequence[int], before: int
) -> str:
    """Collect dominating text from each visible lexical ancestor."""

    if not block_stack:
        return _same_block_prefix(text, 0, before)
    segments = []
    for index, block_start in enumerate(block_stack):
        segment_end = (
            block_stack[index + 1]
            if index + 1 < len(block_stack)
            else before
        )
        segments.append(_same_block_prefix(text, block_start, segment_end))
    return "\n".join(segments)


def _has_unpinned_multi_capture_escape(
    source: ScannedFile, call_start: int, call_end: int
) -> bool:
    line_start = source.raw.rfind("\n", 0, call_start) + 1
    previous_line_start = source.raw.rfind("\n", 0, max(line_start - 1, 0)) + 1
    line_end = source.raw.find("\n", call_end)
    if line_end == -1:
        line_end = len(source.raw)
    return (
        UNPINNED_MULTI_CAPTURE_ESCAPE
        in source.raw[previous_line_start:line_end]
    )


def _inside_pin_operands_definition(
    source: ScannedFile, block_stack: Sequence[int]
) -> bool:
    for block_start in reversed(block_stack):
        header_start = max(
            source.masked.rfind(";", 0, block_start),
            source.masked.rfind("}", 0, block_start),
        )
        header = source.masked[header_start + 1 : block_start]
        if re.search(r"\bpin_operands\s*\([^;{}]*\)\s*$", header) is not None:
            return True
    return False


def check_unpinned_multi_capture(source: ScannedFile) -> list[Hit]:
    """Find unpinned calls capturing raw storage from multiple tensor bases."""

    relative_parts = source.relative_to_root.parts
    file_parts = Path(source.file).parts
    if relative_parts[:2] != ("core", "tensor") and file_parts[:3] != (
        "src",
        "core",
        "tensor",
    ):
        return []

    call_matches = [
        match
        for regex in (CALL_LIKE_OPEN_RE, CUDA_LAUNCH_OPEN_RE)
        for match in regex.finditer(source.masked)
        if match.group(1) not in NON_CALL_KEYWORDS
    ]
    spans: list[tuple[int, int, int]] = []
    for match in call_matches:
        opening_paren = match.end() - 1
        closing_paren = _find_matching(source.masked, opening_paren, "(", ")")
        if closing_paren is not None:
            spans.append((match.start(1), opening_paren, closing_paren))

    block_stacks = _enclosing_block_stacks(
        source.masked, [call_start for call_start, _, _ in spans]
    )
    qualifying: list[tuple[int, int]] = []
    for call_start, opening_paren, closing_paren in spans:
        call_text = source.masked[opening_paren + 1 : closing_paren]
        bases = {
            re.sub(r"\s+", "", match.group(1))
            for match in POINTER_BASE_RE.finditer(call_text)
        }
        for bare_match in BARE_POINTER_RE.finditer(call_text):
            prefix = call_text[: bare_match.start()].rstrip()
            if re.search(r"\.\s*template\s*$", prefix) is None:
                bases.add("$this")
                break
        if len(bases) < 2:
            continue

        block_stack = block_stacks[call_start]
        if _inside_pin_operands_definition(source, block_stack):
            continue
        prior_visible_scope = _visible_enclosing_prefix(
            source.masked, block_stack, call_start
        )
        if PIN_OPERANDS_RE.search(prior_visible_scope) is not None:
            continue

        safe_bases = {
            match.group(1)
            for match in FACTORY_ASSIGNMENT_RE.finditer(prior_visible_scope)
        }
        if len(bases - safe_bases) < 2:
            continue
        if _has_unpinned_multi_capture_escape(
            source, call_start, closing_paren + 1
        ):
            continue
        qualifying.append((call_start, closing_paren))

    outermost: list[tuple[int, int]] = []
    for call_start, call_end in sorted(
        qualifying, key=lambda span: (span[0], -span[1])
    ):
        if any(
            parent_start <= call_start and call_end <= parent_end
            for parent_start, parent_end in outermost
        ):
            continue
        outermost.append((call_start, call_end))

    return [
        source.hit(call_start, "unpinned-multi-capture")
        for call_start, _ in outermost
    ]


RULES = (
    RuleSpec(
        "expected-string",
        "std::expected whose final top-level error type is exactly std::string",
        check_expected_string,
    ),
    RuleSpec(
        "discarded-status-macro-free-cuda",
        "bare discarded CUDA runtime status outside approved wrappers",
        check_discarded_status_macro_free_cuda,
    ),
    RuleSpec(
        "raw-vk-check",
        "call site of _THROW_ERROR or LFS_VK_CHECK_MSG",
        check_raw_vk_check,
    ),
    RuleSpec(
        "empty-catch",
        "empty or unreviewed catch-all/std::exception handler",
        check_empty_catch,
    ),
    RuleSpec(
        "local-check-macro",
        "non-core #define whose name contains ASSERT or CHECK",
        check_local_check_macro,
    ),
    RuleSpec(
        "fatal-invariant-site",
        "call site of LFS_FATAL_INVARIANT",
        check_fatal_invariant_site,
    ),
    RuleSpec(
        "vk-infinite-wait",
        "Vulkan wait/acquire call using UINT64_MAX or kWaitForeverNs",
        check_vk_infinite_wait,
    ),
    RuleSpec(
        "unchecked-kernel-launch",
        "CUDA <<< launch lacking a recognized check through three later lines",
        check_unchecked_kernel_launch,
    ),
    RuleSpec(
        "result-in-cu",
        "lfs::Result or std::expected token in a .cu/.cuh file",
        check_result_in_cu,
        extensions=CUDA_SOURCE_EXTENSIONS,
        include_test_paths=True,
    ),
    RuleSpec(
        "unpinned-multi-capture",
        "tensor call capturing raw storage from two or more unpinned bases",
        check_unpinned_multi_capture,
    ),
)
RULE_BY_ID = {rule.rule_id: rule for rule in RULES}


@dataclass(frozen=True)
class AllowlistEntry:
    """A named, owned exemption. Every entry must carry a reason and an
    expiry milestone at which it is re-justified or deleted."""

    rule: str
    file: str
    line_pattern: str
    owner: str
    reason: str
    expiry: str


ALLOWLIST: tuple[AllowlistEntry, ...] = (
    AllowlistEntry(
        rule="expected-string",
        file="src/core/include/core/error.hpp",
        line_pattern=r"from_legacy_expected",
        owner="error-architecture",
        reason="the sanctioned legacy string bridge's own declaration",
        expiry="post-campaign (expected-string workoff)",
    ),
)


def _is_allowlisted(hit: Hit, source: "ScannedFile") -> bool:
    for entry in ALLOWLIST:
        if entry.rule != hit.rule or entry.file != hit.file:
            continue
        line_start = source.line_starts[hit.line - 1]
        line_end = source.raw.find("\n", line_start)
        raw_line = source.raw[line_start : None if line_end < 0 else line_end]
        if re.search(entry.line_pattern, raw_line):
            return True
    return False


def _iter_source_files(root: Path) -> Iterator[Path]:
    """Walk with explicitly sorted directory and file iteration."""

    for child in sorted(root.iterdir(), key=lambda entry: entry.name):
        if child.is_dir():
            yield from _iter_source_files(child)
        elif child.is_file() and child.suffix.lower() in SOURCE_EXTENSIONS:
            yield child


def _is_structurally_excluded(relative_path: Path) -> bool:
    lowered_parts = tuple(part.lower() for part in relative_path.parts)
    if any(part in STRUCTURAL_EXCLUDED_COMPONENTS for part in lowered_parts):
        return True
    if any("generated" in part for part in lowered_parts):
        return True

    basename = relative_path.name
    return basename == "html_viewer_resources.hpp" or re.fullmatch(
        r".*_generated.*", basename
    ) is not None


def _is_test_path(relative_path: Path) -> bool:
    return any(part.lower() in TEST_COMPONENTS for part in relative_path.parts)


def _display_root(root: Path) -> Path:
    try:
        root.relative_to(REPO_ROOT)
    except ValueError:
        return root.parent
    return REPO_ROOT


def scan(root: Path = DEFAULT_ROOT) -> list[Hit]:
    """Scan ``root`` once and return all registered rule hits."""

    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"scan root is not a directory: {root}")

    display_root = _display_root(root)
    hits: list[Hit] = []
    for path in _iter_source_files(root):
        relative_path = path.relative_to(root)
        if _is_structurally_excluded(relative_path):
            continue

        test_path = _is_test_path(relative_path)
        applicable_rules = tuple(
            rule
            for rule in RULES
            if path.suffix.lower() in rule.extensions
            and (rule.include_test_paths or not test_path)
        )
        if not applicable_rules:
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        masked = mask_source(text)
        source = ScannedFile(
            path=path,
            relative_to_root=relative_path,
            file=path.relative_to(display_root).as_posix(),
            module=relative_path.parts[0],
            masked=masked,
            raw=text,
            line_starts=_line_starts(masked),
        )
        for rule in applicable_rules:
            hits.extend(
                hit for hit in rule.checker(source) if not _is_allowlisted(hit, source)
            )

    return hits


def census_summary(hits: Sequence[Hit]) -> dict[str, dict[str, Any]]:
    rule_counts: dict[str, dict[str, int]] = {
        rule.rule_id: {} for rule in RULES
    }
    for hit in hits:
        module_counts = rule_counts[hit.rule]
        module_counts[hit.module] = module_counts.get(hit.module, 0) + 1
    totals = {
        rule.rule_id: sum(rule_counts[rule.rule_id].values()) for rule in RULES
    }
    return {"rules": rule_counts, "totals": totals}


def baseline_locations(hits: Sequence[Hit]) -> dict[str, dict[str, list[str]]]:
    locations: dict[str, dict[str, list[str]]] = {
        rule.rule_id: {} for rule in RULES
    }
    for hit in hits:
        locations[hit.rule].setdefault(hit.module, []).append(hit.location)
    for module_locations in locations.values():
        for values in module_locations.values():
            values.sort(key=_location_sort_key)
    return locations


def _location_sort_key(location: str) -> tuple[str, int]:
    file_name, separator, line_text = location.rpartition(":")
    if not separator or not line_text.isdigit():
        return (location, -1)
    return (file_name, int(line_text))


def _load_baseline(path: Path) -> dict[str, dict[str, set[str]]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load baseline '{path}': {error}") from error

    if not isinstance(raw, dict):
        raise ValueError(f"invalid baseline '{path}': top level must be an object")

    normalized: dict[str, dict[str, set[str]]] = {}
    for rule_id, modules in raw.items():
        if rule_id not in RULE_BY_ID:
            raise ValueError(
                f"invalid baseline '{path}': unknown rule id {rule_id!r}"
            )
        if not isinstance(modules, dict):
            raise ValueError(
                f"invalid baseline '{path}': {rule_id!r} must map to an object"
            )
        normalized_modules: dict[str, set[str]] = {}
        for module, values in modules.items():
            if not isinstance(module, str) or not isinstance(values, list):
                raise ValueError(
                    f"invalid baseline '{path}': modules must map to location lists"
                )
            if not all(
                isinstance(value, str) and _location_sort_key(value)[1] >= 1
                for value in values
            ):
                raise ValueError(
                    f"invalid baseline '{path}': locations must be FILE:LINE strings"
                )
            normalized_modules[module] = set(values)
        normalized[rule_id] = normalized_modules
    return normalized


def _new_violations(
    hits: Sequence[Hit], baseline: dict[str, dict[str, set[str]]]
) -> list[Hit]:
    """Gate on per-(rule, file) counts so edits that merely shift pre-existing
    debt to other lines cannot fail the gate; baseline locations only name the
    likely-new hits when a count actually increases."""
    baseline_locations: dict[str, set[str]] = {}
    baseline_counts: dict[tuple[str, str], int] = {}
    for rule_id, modules in baseline.items():
        merged = baseline_locations.setdefault(rule_id, set())
        for locations in modules.values():
            merged |= locations
            for location in locations:
                key = (rule_id, location.rsplit(":", 1)[0])
                baseline_counts[key] = baseline_counts.get(key, 0) + 1

    unique_current = {
        (hit.rule, hit.module, hit.file, hit.line): hit for hit in hits
    }
    current_by_rule_file: dict[tuple[str, str], list[Hit]] = {}
    for hit in unique_current.values():
        current_by_rule_file.setdefault((hit.rule, hit.file), []).append(hit)

    new_hits: list[Hit] = []
    for key, file_hits in current_by_rule_file.items():
        if len(file_hits) <= baseline_counts.get(key, 0):
            continue
        known = baseline_locations.get(key[0], set())
        unmatched = [hit for hit in file_hits if hit.location not in known]
        new_hits.extend(unmatched if unmatched else file_hits)
    return sorted(new_hits, key=lambda hit: (hit.rule, hit.module, hit.file, hit.line))


def _stale_decreases(
    hits: Sequence[Hit], baseline: dict[str, dict[str, set[str]]]
) -> list[tuple[str, str, str, int, int]]:
    """The other half of the ratchet (the 7A lesson): a per-(rule, file) count
    below the baseline is indistinguishable from a matcher-evading refactor,
    so it fails until the baseline is regenerated in the same change."""
    baseline_entries: dict[tuple[str, str], tuple[str, int]] = {}
    for rule_id, modules in baseline.items():
        for module, locations in modules.items():
            for location in locations:
                key = (rule_id, location.rsplit(":", 1)[0])
                previous = baseline_entries.get(key)
                count = (previous[1] if previous else 0) + 1
                baseline_entries[key] = (previous[0] if previous else module, count)

    unique_current = {(hit.rule, hit.module, hit.file, hit.line) for hit in hits}
    current_counts: dict[tuple[str, str], int] = {}
    for rule, _module, file, _line in unique_current:
        current_counts[(rule, file)] = current_counts.get((rule, file), 0) + 1

    stale = [
        (rule, module, file, count, current_counts.get((rule, file), 0))
        for (rule, file), (module, count) in baseline_entries.items()
        if current_counts.get((rule, file), 0) < count
    ]
    return sorted(stale)


def _rules_help() -> str:
    rule_lines = "\n".join(
        f"  {rule.rule_id:<36} {rule.description}" for rule in RULES
    )
    return f"""rule IDs:
{rule_lines}

This is a lexical/regex heuristic, NOT an AST-based checker. Comments and
literals are masked, but inactive preprocessor branches remain visible.

A reviewed catch-all may be exempted with a raw comment inside the catch body:
`// LFS-CENSUS-OK(empty-catch): <reason>`. Sanctioned permanent exceptions live
in ALLOWLIST (each with owner, reason, and expiry).

The --baseline gate is a two-sided per-(rule,file) count ratchet. Increases
list the likely-new locations. Decreases print a `stale-baseline` line and also
fail: to a lexical scanner an honest fix and a matcher-evading refactor (the
"7A lesson" — routing counted call sites through a wrapper so the token
disappears) look identical, so a reduction must be acknowledged by regenerating
the baseline (--write-baseline) in the same change.

Known over-counts: mutually exclusive macro definitions are counted separately;
inactive #if branches count; and a non-launch <<< token can resemble a kernel
launch. Known under-counts: malformed/unbalanced syntax is skipped; discarded
CUDA status detection only covers conservative bare runtime-call statements
(not lowercase CUDA driver APIs or larger expressions); an unrelated nearby
CUDA check can satisfy the launch window; and result-in-cu only recognizes
.cu/.cuh extensions, not CMake LANGUAGE CUDA overrides. The intentional
cudaGetLastError/cudaPeekAtLastError carve-out is never reported.

See .codex_tmp/error-architecture-analysis.md section 5.12 for the full
rationale and the later AST-based clang-tidy project-check plan.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Phase 0 error-debt census using conservative lexical/regex "
            "heuristics (not an AST)."
        ),
        epilog=_rules_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        metavar="DIR",
        help=f"source tree to scan (default: {DEFAULT_ROOT})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--baseline",
        type=Path,
        metavar="FILE",
        help=(
            "fail on new locations and on per-(rule,file) count decreases vs the "
            "location baseline (regenerate the baseline in the same change to "
            "acknowledge a reduction)"
        ),
    )
    mode.add_argument(
        "--write-baseline",
        type=Path,
        metavar="FILE",
        help="write a pretty location-level baseline",
    )
    mode.add_argument(
        "--list",
        dest="list_rule",
        metavar="RULE_ID",
        help="list current FILE:LINE hits for one rule",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list_rule is not None and args.list_rule not in RULE_BY_ID:
        print(f"error: unknown rule id: {args.list_rule}", file=sys.stderr)
        return 2

    try:
        hits = scan(args.root)
    except OSError as error:
        print(f"error: unable to scan '{args.root}': {error}", file=sys.stderr)
        return 2

    if args.write_baseline is not None:
        baseline = baseline_locations(hits)
        try:
            args.write_baseline.write_text(
                json.dumps(baseline, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            print(
                f"error: cannot write baseline '{args.write_baseline}': {error}",
                file=sys.stderr,
            )
            return 2

        totals = census_summary(hits)["totals"]
        print(f"wrote baseline: {args.write_baseline}", file=sys.stderr)
        for rule in RULES:
            print(f"{rule.rule_id}: {totals[rule.rule_id]}", file=sys.stderr)
        return 0

    if args.baseline is not None:
        try:
            baseline = _load_baseline(args.baseline)
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

        new_hits = _new_violations(hits, baseline)
        stale = _stale_decreases(hits, baseline)
        for hit in new_hits:
            print(f"{hit.rule}\t{hit.module}\t{hit.location}")
        for rule, module, file, baseline_count, current_count in stale:
            print(f"stale-baseline\t{rule}\t{module}\t{file}\t{baseline_count}->{current_count}")
        if stale:
            print(
                "error: per-(rule,file) counts decreased vs the baseline; an honest fix "
                "and a matcher-evading refactor look identical here. If the reduction is "
                f"intentional, regenerate in the same change: --write-baseline {args.baseline}",
                file=sys.stderr,
            )
        if new_hits or stale:
            return 1
        print("no new error-debt violations", file=sys.stderr)
        return 0

    if args.list_rule is not None:
        selected = sorted(
            (hit for hit in hits if hit.rule == args.list_rule),
            key=lambda hit: (hit.file, hit.line),
        )
        for hit in selected:
            print(hit.location)
        return 0

    print(json.dumps(census_summary(hits), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
