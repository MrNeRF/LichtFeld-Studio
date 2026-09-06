#!/ usr / bin / env python3
"""Generate the per-case tensor backend eligibility manifest."""

from __future__ import annotations

import argparse
import functools
import collections
import dataclasses
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
OUTPUTS = {
    "manifest": TESTS / "tensor_backend_manifest.json",
    "lichtfeld_tests": TESTS / "tensor_backend_suite.filter",
    "tensor_hardening_tests": TESTS / "tensor_backend_hardening.filter",
    "lichtfeld_tests_flags": TESTS / "tensor_backend_suite.flags",
    "tensor_hardening_tests_flags": TESTS / "tensor_backend_hardening.flags",
}
ALLOWLIST = TESTS / "tensor_backend_skip_allowlist.json"
# A test file may include only tensor-library headers (and helper headers under
# tests/, which are walked). Any other quoted include reaches production code
# outside core/tensor, whose GPU work is invisible to the classifier and may run
# raw CUDA kernels on Vulkan storage; such tests are consumers and run under G7.
TENSOR_LIBRARY_INCLUDES = (
    "core/tensor.hpp", "core/tensor_fwd.hpp", "core/tensor_backend.hpp",
    "core/gpu_backend_fwd.hpp", "core/tensor/", "core/alloc_counter.hpp",
    "core/logger.hpp", "core/cuda_error.hpp", "core/pinned_memory_allocator.hpp",
    "core/error.hpp", "core/export.hpp", "core/assert.hpp", "core/failure_report.hpp",
    "core/device_fault.hpp",
)
BINARIES = {
    "lichtfeld_tests": ROOT / "build/tests/lichtfeld_tests",
    "tensor_hardening_tests": ROOT / "build/tests/tensor_hardening_tests",
}
HARDENING_FILES = {
    f"test_tensor_hardening_theme_{letter}.cpp" for letter in "abcdef"
}
INACTIVE_SOURCE_REGISTRATIONS = {
    ("tests/test_gsplat_rasterizer.cpp", "GsplatRasterizerTest", "CudaAllocationFailureAbortsAndRecovers"),
    ("tests/test_rml_path_utils.cpp", "RmlSystemInterfaceTest", "JoinsWindowsDrivePathsWithoutTreatingDriveAsUriScheme"),
    ("tests/test_user_paths.cpp", "UserPathsContractTest", "WindowsDefaultUsesProfileDotLichtfeld"),
}
RAW_LITERAL_RE = re.compile(r'(?:u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\(')

#Lane B section 1 supplies the launcher rows.The values are the public Tensor
#method names from the Lane D inventory which can route to each row.
LAUNCHER_METHODS = {
    "launch_unary_op_generic": (
        "abs", "acos", "asin", "atan", "cos", "cosh", "erf", "exp", "exp2",
        "expm1", "gelu", "log", "log10", "log1p", "log2", "neg", "reciprocal",
        "relu", "round", "rsqrt", "sigmoid", "sign", "sin", "sinh", "sqrt",
        "square", "tan", "tanh", "trunc",
    ),
    "launch_ieee_round_float": ("round",),
    "launch_binary_op_generic": (
        "add", "div", "eq", "ge", "gt", "le", "logical_and", "logical_or",
        "logical_xor", "lt", "maximum", "minimum", "mul", "ne", "pow", "sub",
    ),
    "launch_ieee_maximum_float": ("maximum",),
    "launch_ieee_minimum_float": ("minimum",),
    "launch_scalar_op_generic": (
        "add", "div", "eq", "ge", "gt", "le", "lt", "mul", "ne", "pow", "sub",
    ),
    "launch_broadcast_binary": (
        "add", "div", "eq", "ge", "gt", "le", "logical_and", "logical_or",
        "logical_xor", "lt", "maximum", "minimum", "mul", "ne", "pow", "sub",
    ),
    "launch_ieee_maximum_float_broadcast": ("maximum",),
    "launch_ieee_minimum_float_broadcast": ("minimum",),
    "direct_sum_scalar": ("sum_scalar",),
    "direct_mean_scalar": ("mean_scalar",),
    "direct_max_scalar": ("max_scalar",),
    "direct_min_scalar": ("min_scalar",),
    "launch_reduce_op": (
        "all", "any", "argmax", "argmin", "max", "mean", "min", "norm", "prod", "sum",
    ),
    "launch_column_reduce": ("max", "mean", "min", "prod", "sum"),
    "launch_strided_reduce_fast": ("max", "mean", "min", "prod", "sum"),
    "launch_fused_transform_reduce": ("max", "mean", "min", "prod", "sum"),
    "launch_fused_segmented_transform_reduce": ("max", "mean", "min", "prod", "sum"),
    "launch_count_nonzero_bool": ("count_nonzero",),
    "launch_count_nonzero_float": ("count_nonzero",),
    "launch_cumsum": ("cumsum",),
    "launch_sort_1d": ("sort",),
    "launch_sort_2d": ("sort",),
    "launch_gather": ("gather", "gather_lazy"),
    "launch_gather_fused_unary": ("gather_lazy",),
    "launch_take": ("take",),
    "launch_index_select": ("index_select", "index_select_into"),
    "launch_scatter": ("scatter_", "index_fill_"),
    "launch_index_copy": ("index_copy_",),
    "launch_index_add": ("index_add_",),
    "launch_strided_scatter": ("index_put_",),
    "launch_strided_scatter_immediate": ("index_put_",),
    "launch_strided_scatter_int32_to_float32": ("index_put_",),
    "launch_masked_fill": ("masked_fill", "masked_fill_"),
    "launch_masked_select": ("masked_select", "masked_select_rows"),
    "launch_masked_scatter": ("masked_scatter_",),
    "launch_and_live": ("and_live_",),
    "launch_where": ("where",),
    "launch_nonzero": ("nonzero",),
    "launch_nonzero_bool": ("nonzero",),
    "launch_sgemm": ("linear", "linear_out", "matmul", "mm"),
    "launch_sgemm_tn": ("matmul", "mm"),
    "launch_sgemm_batched": ("bmm",),
    "launch_sgemm_bias_relu": ("linear_bias_relu", "linear_bias_relu_out"),
    "launch_dot_product": ("dot",),
    "launch_diag": ("diag",),
    "launch_max_pool2d": ("max_pool2d", "max_pool2d_out"),
    "launch_adaptive_avg_pool2d": ("adaptive_avg_pool2d", "adaptive_avg_pool2d_out"),
    "launch_bias_add": ("conv1x1", "conv1x1_bias_out", "linear"),
    "launch_bias_relu": ("conv1x1_bias_relu_out", "linear_bias_relu", "linear_bias_relu_out"),
    "launch_relu": ("relu", "relu_", "relu_out"),
    "launch_uniform": ("rand", "uniform_"),
    "launch_bernoulli": ("bernoulli", "bernoulli_"),
    "launch_randint": ("randint",),
    "launch_multinomial": ("multinomial",),
    "launch_strided_copy": ("clone", "contiguous", "copy_", "copy_from"),
    "launch_strided_copy_immediate": ("clone", "contiguous", "copy_", "copy_from"),
    "launch_strided_upload": ("from_vector",),
    "launch_convert_type": ("to",),
    "launch_cat_last_dim": ("cat", "stack"),
    "launch_cat_middle_dim": ("cat", "stack"),
    "launch_pad": ("pad",),
    "launch_fill_strided": ("fill_", "full", "ones", "zeros"),
    "launch_load_op": ("load",),
    "launch_clamp_scalar": ("clamp", "clamp_", "clamp_max", "clamp_max_", "clamp_min", "clamp_min_"),
    "launch_clamp_fused": ("clamp", "clamp_max", "clamp_min"),
    "launch_clamp_scalar_int": ("clamp", "clamp_", "clamp_max", "clamp_max_", "clamp_min", "clamp_min_"),
    "launch_fused_pointwise_chain": (
        "abs", "add", "div", "exp", "log", "mul", "neg", "reciprocal", "relu",
        "rsqrt", "sigmoid", "sqrt", "square", "sub", "tanh",
    ),
    "launch_cdist": ("cdist",),
    "launch_eye": ("eye",),
    "has_nan_gpu": ("has_nan",),
    "has_inf_gpu": ("has_inf",),
}


@dataclasses.dataclass(frozen=True)
class Fragment:
    file: Path
    line: int
    text: str


@dataclasses.dataclass(frozen=True)
class Registration:
    kind: str
    suite: str
    test: str
    file: Path
    line: int
    body: str
    conditional: bool


def mask_cpp(text: str) -> str:
    """Blank comments and literals while retaining bytes and newlines."""
    out = list(text)
    i = 0
    state = "code"
    quote = ""
    while i < len(text):
        pair = text[i : i + 2]
        if state == "code" and text[i] in "uULR":
            raw = RAW_LITERAL_RE.match(text, i)
            if raw:
                terminator = ")" + raw.group(1) + '"'
                end = text.find(terminator, raw.end())
                if end < 0:
                    end = len(text)
                else:
                    end += len(terminator)
                for pos in range(i, end):
                    if text[pos] != "\n":
                        out[pos] = " "
                i = end
                continue
        if state == "code" and pair == "//":
            out[i] = out[i + 1] = " "
            i += 2
            state = "line"
            continue
        if state == "code" and pair == "/*":
            out[i] = out[i + 1] = " "
            i += 2
            state = "block"
            continue
        if (state == "code" and text[i] == "'" and i > 0 and i + 1 < len(text)
                and text[i - 1].isdigit() and text[i + 1].isdigit()):
            i += 1
            continue
        if state == "code" and text[i] in "\"'":
            quote = text[i]
            out[i] = " "
            i += 1
            state = "literal"
            continue
        if state == "line":
            if text[i] == "\n":
                state = "code"
            else:
                out[i] = " "
            i += 1
            continue
        if state == "block":
            if pair == "*/":
                out[i] = out[i + 1] = " "
                i += 2
                state = "code"
            else:
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            continue
        if state == "literal":
            if text[i] == "\\" and i + 1 < len(text):
                out[i] = " "
                if text[i + 1] != "\n":
                    out[i + 1] = " "
                i += 2
            elif text[i] == quote:
                out[i] = " "
                i += 1
                state = "code"
            else:
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            continue
        i += 1
    return "".join(out)


def matching(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == opening:
            depth += 1
        elif text[pos] == closing:
            depth -= 1
            if depth == 0:
                return pos
    raise ValueError(f"unclosed {opening} at byte {start}")


def macro_calls(path: Path, names: tuple[str, ...]):
    original = path.read_text(encoding="utf-8")
    masked = mask_cpp(original)
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\s*\(")
    for match in pattern.finditer(masked):
        open_pos = masked.find("(", match.start())
        close_pos = matching(masked, open_pos, "(", ")")
        yield match.group(1), original, masked, match.start(), open_pos, close_pos


def split_macro_args(masked_args: str, original_args: str) -> list[str]:
    starts = [0]
    paren = brace = bracket = angle = 0
    for i, char in enumerate(masked_args):
        if char == "(":
            paren += 1
        elif char == ")":
            paren -= 1
        elif char == "{":
            brace += 1
        elif char == "}":
            brace -= 1
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket -= 1
        elif char == "<":
            angle += 1
        elif char == ">" and angle:
            angle -= 1
        elif char == "," and paren == brace == bracket == angle == 0:
            starts.append(i + 1)
    ends = [x - 1 for x in starts[1:]] + [len(original_args)]
    return [original_args[a:b].strip() for a, b in zip(starts, ends)]


def identifier(value: str, path: Path, line: int) -> str:
    value = re.sub(r"/\*.*?\*/|//[^\n]*", "", value, flags=re.S).strip()
    if not re.fullmatch(r"[A-Za-z_]\w*", value):
        raise ValueError(f"{path.relative_to(ROOT)}:{line}: unsupported gtest identifier {value!r}")
    return value


def source_registrations() -> tuple[list[Registration], list[tuple[str, str, Path, int]]]:
    registrations = []
    instantiations = []
    cmake = (TESTS / "CMakeLists.txt").read_text(encoding="utf-8")
    main_sources_text = cmake[cmake.index("set(TEST_FILES") : cmake.index("set(TEST_SOURCES")]
    main_sources = set(re.findall(r"\b[\w-]+\.cpp\b", main_sources_text))
    source_names = main_sources | HARDENING_FILES
    for path in sorted(TESTS / name for name in source_names):
        if not path.is_file():
            raise ValueError(f"test source named by CMake is missing: {path.relative_to(ROOT)}")
        for kind, original, masked, start, open_pos, close_pos in macro_calls(
            path, ("TEST", "TEST_F", "TEST_P", "INSTANTIATE_TEST_SUITE_P")
        ):
            line = original.count("\n", 0, start) + 1
            conditional_depth = 0
            for directive in re.findall(r"(?m)^\s*#\s*(if|ifdef|ifndef|endif)\b", original[:start]):
                conditional_depth += -1 if directive == "endif" else 1
            args = split_macro_args(masked[open_pos + 1 : close_pos], original[open_pos + 1 : close_pos])
            if len(args) < 2:
                raise ValueError(f"{path.relative_to(ROOT)}:{line}: malformed {kind}")
            first = identifier(args[0], path, line)
            second = identifier(args[1], path, line)
            if kind == "INSTANTIATE_TEST_SUITE_P":
                instantiations.append((first, second, path, line))
                continue
            body_start = close_pos + 1
            while body_start < len(masked) and masked[body_start].isspace():
                body_start += 1
            if body_start >= len(masked) or masked[body_start] != "{":
                raise ValueError(f"{path.relative_to(ROOT)}:{line}: {kind} has no body")
            body_end = matching(masked, body_start, "{", "}")
            registrations.append(
                Registration(
                    kind, first, second, path, line, original[body_start + 1 : body_end],
                    conditional_depth > 0,
                )
            )
    return registrations, instantiations


def run_list(executable: str, test_filter: str | None = None) -> list[str]:
    binary = BINARIES[executable]
    if not binary.is_file():
        raise ValueError(f"missing test executable: {binary.relative_to(ROOT)}")
    command = [str(binary), "--gtest_list_tests"]
    if test_filter is not None:
        command.append(f"--gtest_filter={test_filter}")
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise ValueError(f"{' '.join(command)} exited {result.returncode}\n{result.stdout}")
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    current_suite = None
    names = []
    for raw_line in ansi.sub("", result.stdout).splitlines():
        if raw_line and not raw_line[0].isspace() and raw_line.endswith(".") and not raw_line.startswith("["):
            current_suite = raw_line.strip()
        elif current_suite and raw_line.startswith("  "):
            test = raw_line.strip().split("  #", 1)[0]
            if test and not test.startswith("["):
                names.append(current_suite + test)
    if len(names) != len(set(names)):
        duplicates = sorted(name for name, count in collections.Counter(names).items() if count > 1)
        raise ValueError(f"duplicate runtime test names in {executable}: {', '.join(duplicates)}")
    return names


def extract_helpers():
    by_name: dict[tuple[Path, str], list[Fragment]] = collections.defaultdict(list)
    fixtures: dict[tuple[Path, str], list[Fragment]] = collections.defaultdict(list)
    fixture_bases: dict[tuple[Path, str], tuple[str, ...]] = {}
    paths = sorted(TESTS.glob("*.cpp")) + sorted(TESTS.glob("*.hpp"))
    control = {"if", "for", "while", "switch", "catch", "TEST", "TEST_F", "TEST_P"}
    for path in paths:
        original = path.read_text(encoding="utf-8")
        masked = mask_cpp(original)
        class_ranges = []
        class_pattern = re.compile(r"\b(?:class|struct)\s+([A-Za-z_]\w*)([^;{]*)\{")
        for match in class_pattern.finditer(masked):
            open_pos = masked.find("{", match.start())
            try:
                end = matching(masked, open_pos, "{", "}")
            except ValueError:
                continue
            bases = tuple(re.findall(r"\b([A-Za-z_]\w*)\s*(?:<[^>{}]*>)?\s*(?:,|$)",
                                     match.group(2).split(":", 1)[-1])) if ":" in match.group(2) else ()
            fixture_bases[(path, match.group(1))] = bases
            class_ranges.append((open_pos, end, match.group(1)))
        function_pattern = re.compile(
            r"(?m)(?:^|[;{}]\s*)[^;{}#\n]*?\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*"
            r"(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:->[^\n{]+)?\s*\{"
        )
        for match in function_pattern.finditer(masked):
            name = match.group(1)
            if name in control:
                continue
            open_pos = masked.find("{", match.start(), match.end())
            try:
                end = matching(masked, open_pos, "{", "}")
            except ValueError:
                continue
            fragment = Fragment(path, original.count("\n", 0, open_pos) + 1,
                                original[open_pos + 1 : end])
            by_name[(path, name)].append(fragment)
            containing_classes = [entry for entry in class_ranges if entry[0] < open_pos < entry[1]]
            if containing_classes:
                _, _, class_name = min(containing_classes, key=lambda entry: entry[1] - entry[0])
                if name in {class_name, f"~{class_name}", "SetUp", "TearDown"}:
                    fixtures[(path, class_name)].append(fragment)
    return by_name, fixtures, fixture_bases


def reachable_fragments(registration: Registration, helpers, fixtures, fixture_bases) -> list[Fragment]:
    allowed_paths = {registration.file}
    include_pending = [registration.file]
    while include_pending:
        include_path = include_pending.pop()
        include_text = include_path.read_text(encoding="utf-8")
        for include in re.findall(r'^\s*#\s*include\s*"([^"]+\.hpp)"', include_text, re.M):
            candidate = (include_path.parent / include).resolve()
            if candidate.is_file() and candidate.parent == TESTS.resolve() and candidate not in allowed_paths:
                allowed_paths.add(candidate)
                include_pending.append(candidate)
    fragments = [Fragment(registration.file, registration.line, registration.body)]
    fixture_pending = [registration.suite]
    seen_fixtures = set()
    while fixture_pending:
        fixture = fixture_pending.pop()
        if fixture in seen_fixtures:
            continue
        seen_fixtures.add(fixture)
        for path in allowed_paths:
            fragments.extend(fixtures.get((path, fixture), ()))
            fixture_pending.extend(fixture_bases.get((path, fixture), ()))
    seen_fragments = {(x.file, x.line) for x in fragments}
    seen_calls = set()
    pending = list(fragments)
    call_pattern = re.compile(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}()]*>)?\s*\(")
    while pending:
        fragment = pending.pop()
        for name in call_pattern.findall(mask_cpp(fragment.text)):
            if name in seen_calls:
                continue
            seen_calls.add(name)
            for path in allowed_paths:
                for helper in helpers.get((path, name), ()):
                    key = (helper.file, helper.line)
                    if key not in seen_fragments:
                        seen_fragments.add(key)
                        fragments.append(helper)
                        pending.append(helper)
    return fragments


def line_for(fragment: Fragment, offset: int) -> int:
    return fragment.line + fragment.text.count("\n", 0, offset)


def first_match(fragments: list[Fragment], patterns: tuple[str, ...]):
    hits = []
    for fragment in fragments:
        masked = mask_cpp(fragment.text)
        for pattern in patterns:
            match = re.search(pattern, masked, re.S)
            if match:
                hits.append((str(fragment.file.relative_to(ROOT)), line_for(fragment, match.start()), pattern))
    return min(hits, default=None, key=lambda x: (x[0], x[1]))


def launcher_rows(fragments: list[Fragment]) -> list[str]:
    text = "\n".join(mask_cpp(fragment.text) for fragment in fragments)
    called = set(re.findall(
        r"(?:\.|->|\bTensor::|\blfs::core::)([A-Za-z_]\w*)\s*\(", text
    ))
    return [launcher for launcher, methods in LAUNCHER_METHODS.items() if called.intersection(methods)]


def factory_calls_are_cpu_only(text: str) -> bool:
    pattern = re.compile(
        r"\bTensor::(?:arange|bernoulli|empty|eye|from_vector|full|linspace|ones|rand|randint|randn|zeros)\s*\("
    )
    masked = mask_cpp(text)
    found = False
    for match in pattern.finditer(masked):
        found = True
        open_pos = masked.find("(", match.start())
        try:
            close_pos = matching(masked, open_pos, "(", ")")
        except ValueError:
            return False
        if "Device::CPU" not in masked[open_pos:close_pos]:
            return False
    return found


def shared_cuda_pointer(fragments: list[Fragment]):
    for fragment in fragments:
        masked = mask_cpp(fragment.text)
        tensor_pointers = {}
        for match in re.finditer(r"\bTensor::from_blob\s*\(\s*([A-Za-z_]\w*)\b", masked):
            open_pos = masked.find("(", match.start())
            try:
                close_pos = matching(masked, open_pos, "(", ")")
            except ValueError:
                continue
            if "Device::CUDA" in masked[open_pos:close_pos]:
                tensor_pointers[match.group(1)] = match.start()
        for pointer, offset in tensor_pointers.items():
            for match in re.finditer(rf"\btorch::from_blob\s*\(\s*{re.escape(pointer)}\b", masked):
                open_pos = masked.find("(", match.start())
                try:
                    close_pos = matching(masked, open_pos, "(", ")")
                except ValueError:
                    continue
                suffix = masked[close_pos : close_pos + 100]
                if "torch::kCUDA" in masked[open_pos:close_pos] or re.search(
                    r"\.to\s*\(\s*torch::kCUDA\s*\)", suffix
                ):
                    return str(fragment.file.relative_to(ROOT)), line_for(fragment, min(offset, match.start()))
    return None


@functools.lru_cache(maxsize=None)
def includes_application_headers(path: Path) -> str | None:
    # Walks the registration file and every tests/ header it reaches; the first
    # quoted include outside the tensor-library allowlist names the consumer edge.
    pending = [path]
    visited: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in visited or not current.is_file():
            continue
        visited.add(current)
        for line in current.read_text(errors="replace").splitlines():
            match = re.match(r'\s*#\s*include\s*"([^"]+)"', line)
            if not match:
                continue
            include = match.group(1)
            local = current.parent / include
            if local.suffix in {".hpp", ".h", ".hh"} and local.is_file():
                pending.append(local)
                continue
            if include.startswith(TENSOR_LIBRARY_INCLUDES):
                continue
            return include if current == path else f"{include} (via {current.name})"
    return None


ENTRY_TO_LAUNCHERS = {
    "unary": ["launch_unary_op_generic", "launch_ieee_round_float"],
    "binary": ["launch_binary_op_generic", "launch_ieee_maximum_float", "launch_ieee_minimum_float"],
    "scalar": ["launch_scalar_op_generic"],
    "broadcast_binary": ["launch_broadcast_binary", "launch_ieee_maximum_float_broadcast",
                         "launch_ieee_minimum_float_broadcast"],
    "fused_pointwise_chain": ["launch_fused_pointwise_chain"],
    "clamp_scalar": ["launch_clamp_scalar"], "clamp_fused": ["launch_clamp_fused"],
    "clamp_scalar_int": ["launch_clamp_scalar_int"], "convert_type": ["launch_convert_type"],
    "fill_strided": ["launch_fill_strided"], "load_fill": ["launch_load_op"],
    "sum_scalar": ["direct_sum_scalar"], "mean_scalar": ["direct_mean_scalar"],
    "max_scalar": ["direct_max_scalar"], "min_scalar": ["direct_min_scalar"],
    "reduce": ["launch_reduce_op"], "column_reduce": ["launch_column_reduce"],
    "strided_reduce": ["launch_strided_reduce_fast"],
    "fused_transform_reduce": ["launch_fused_transform_reduce"],
    "fused_segmented_transform_reduce": ["launch_fused_segmented_transform_reduce"],
    "count_nonzero_bool": ["launch_count_nonzero_bool"],
    "count_nonzero_float": ["launch_count_nonzero_float"],
    "has_nan": ["has_nan_gpu"], "has_inf": ["has_inf_gpu"], "cumsum": ["launch_cumsum"],
    "sort_1d": ["launch_sort_1d"], "sort_2d": ["launch_sort_2d"], "sgemm": ["launch_sgemm"],
    "sgemm_tn": ["launch_sgemm_tn"], "sgemm_batched": ["launch_sgemm_batched"],
    "sgemm_bias_relu": ["launch_sgemm_bias_relu"], "dot_product": ["launch_dot_product"],
    "diag": ["launch_diag"], "eye": ["launch_eye"], "cdist": ["launch_cdist"],
    "max_pool2d": ["launch_max_pool2d"], "adaptive_avg_pool2d": ["launch_adaptive_avg_pool2d"],
    "bias_add": ["launch_bias_add"], "bias_relu": ["launch_bias_relu"], "relu": ["launch_relu"],
    "uniform": ["launch_uniform"], "bernoulli": ["launch_bernoulli"], "randint": ["launch_randint"],
    "multinomial": ["launch_multinomial"], "normal": ["launch_normal"], "gather": ["launch_gather"],
    "gather_fused_unary": ["launch_gather_fused_unary"], "take": ["launch_take"],
    "index_select": ["launch_index_select"], "scatter": ["launch_scatter"],
    "index_copy": ["launch_index_copy"], "index_add": ["launch_index_add"],
    "index_fill": ["launch_index_fill"], "index_put": ["launch_index_put"],
    "strided_scatter": ["launch_strided_scatter"],
    "strided_scatter_immediate": ["launch_strided_scatter_immediate"],
    "strided_scatter_int32_to_float32": ["launch_strided_scatter_int32_to_float32"],
    "masked_fill": ["launch_masked_fill"], "masked_select": ["launch_masked_select"],
    "masked_scatter": ["launch_masked_scatter"], "and_live": ["launch_and_live"],
    "where": ["launch_where"], "nonzero": ["launch_nonzero"], "nonzero_bool": ["launch_nonzero_bool"],
    "strided_copy": ["launch_strided_copy"], "strided_copy_immediate": ["launch_strided_copy_immediate"],
    "strided_upload": ["launch_strided_upload"], "cat_last_dim": ["launch_cat_last_dim"],
    "cat_middle_dim": ["launch_cat_middle_dim"], "pad": ["launch_pad"],
}


def load_trace(path: Path | None) -> dict[str, dict[str, int]]:
    """Per-test facade entry counts written by the lichtfeld_tests trace listener."""
    if path is None:
        return {}
    trace = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        trace[record["test"]] = record.get("entries", {})
    return trace


def load_allowlist() -> dict[str, dict]:
    if not ALLOWLIST.is_file():
        return {}
    entries = json.loads(ALLOWLIST.read_text())
    allowlist = {}
    for entry in entries:
        for key in ("name", "reason", "reviewed"):
            if key not in entry:
                raise ValueError(f"skip allowlist entry is missing '{key}': {entry}")
        allowlist[entry["name"]] = entry
    return allowlist


def classify(registration: Registration, fragments: list[Fragment], launchers: list[str]):
    combined = "\n".join(fragment.text for fragment in fragments)
    masked = mask_cpp(combined)
    torch_share = shared_cuda_pointer(fragments)
    if torch_share:
        file, line = torch_share
        return "torch-oracle-cuda-only", f"{file}:{line}: raw pointer shared between Torch and LichtFeld tensor storage"

    raw = first_match(
        fragments,
        (
            r"\bcuda(?:Memcpy|Memset)(?:Async|2D|3D)?\s*\([^;]*(?:\.ptr\s*<|\.data_ptr\s*\()",
            r"(?:\.ptr\s*<|\.data_ptr\s*\()[^;]*\bcuda(?:Memcpy|Memset)(?:Async|2D|3D)?\s*\(",
            r"\bGateStream\b",
            r"\b(?:thrust|cub)::[^;]*(?:\.ptr\s*<|\.data_ptr\s*\()",
            r"<<<[^>]*>>>",
            r"\bcudaSetDevice\s*\(",
            r"\bTensor::from_blob\s*\([^;]*\bDevice::CUDA\b",
            r"\bTensor::from_external_owner\s*\([^;]*\bDevice::CUDA\b",
        ),
    )
    if raw is None and re.search(r"(?:\.ptr\s*<|\.data_ptr\s*\()", masked):
        raw = first_match(fragments, (r"\bcuda(?:Memcpy|Memset)(?:Async|2D|3D)?\s*\(",))
    if raw is None and re.search(r"\b(?:thrust|cub)::", masked) and re.search(
        r"(?:\.ptr\s*<|\.data_ptr\s*\()", masked
    ):
        raw = first_match(fragments, (r"\b(?:thrust|cub)::",))
    if raw:
        file, line, _ = raw
        return "cuda-only", f"{file}:{line}: raw CUDA operation on tensor storage or CUDA-only fixture"

    cuda_launcher = first_match(fragments, (r"\btensor_ops::launch_[A-Za-z0-9_]+\s*(?:<[^;{}]*>)?\s*\(",))
    if cuda_launcher:
        file, line, _ = cuda_launcher
        return "cuda-only", f"{file}:{line}: calls the CUDA launcher layer directly"

    # Tests that assert the CUDA backend by design, or drive a CUDA launcher
    # hook, describe CUDA behaviour and cannot pass under another default.
    backend_assertion = first_match(
        fragments,
        (
            r"\b(?:EXPECT|ASSERT)_(?:EQ|NE|TRUE|FALSE)\s*\([^;]*\bGpuBackend::CUDA\b",
            r"\bset_cub_workspace_failure_for_testing\s*\(\s*true",
            r"\bset_reduce_path_override_for_testing\s*\(",
            # CUDA stream identity, CUDA pool accounting and CUDA launch counters
            # are contracts of the CUDA backend, not of the tensor interface.
            r"\b(?:EXPECT|ASSERT)_(?:EQ|NE)\s*\([^;]*\.stream\s*\(\)",
            r"\bCudaMemoryPool\b",
            r"\balloc_counter\b|\bAllocCounter\b",
            r"\b(?:reset_)?tensor_kernel_launch_count\s*\(",
            r"\bassert_device_storage_matches_tag\b|device-tag mismatch",
            r"\bfrom_blob\s*\([^;]*(?:\.data_ptr\s*\(|\.ptr\s*<)",
            r"\bErrorDomain::CUDA\b",
        ),
    )
    if backend_assertion:
        file, line, _ = backend_assertion
        return "cuda-only", f"{file}:{line}: asserts a CUDA backend contract (backend tag, stream identity, pool accounting or launch counters)"

    tensor_signal = bool(
        launchers
        or re.search(r"\b(?:Tensor|TensorShape|DataType|Device)::?\b|\bTensor\b", masked)
        or re.search(r"\.(?:cpu|cuda|device|dtype|ptr|data_ptr|numel|shape|strides)\s*\(", masked)
        or re.search(r"\binternal::(?:lazy_|lazy_executor_|pointwise_)\w*\s*\(", masked)
    )
    if not tensor_signal:
        return "not-tensor", "does not exercise the LichtFeld tensor API"

    cuda_indicator = re.search(
        r"\bDevice::CUDA\b|\bCUDAStreamGuard\b|\.cuda\s*\(|\bgetCurrentCUDAStream\s*\(", masked
    )
    if not cuda_indicator and "Device::CPU" in masked and factory_calls_are_cpu_only(combined):
        return "cpu-only", "constructs and exercises only CPU tensor storage"
    application_header = includes_application_headers(registration.file)
    if application_header is not None:
        return "consumer", (
            f"includes {application_header}: reaches production code outside core/tensor; "
            "run under G7, not G2"
        )
    return "vulkan", None


def registration_for_runtime(runtime_name: str, registrations: list[Registration]) -> Registration | None:
    suite, runtime_test = runtime_name.rsplit(".", 1)
    base_test = runtime_test.split("/", 1)[0]
    fixture = suite.rsplit("/", 1)[-1]
    candidates = [
        reg for reg in registrations
        if reg.suite == fixture and reg.test == base_test
        and ((reg.kind == "TEST_P") == ("/" in suite or "/" in runtime_test))
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def make_filter(records: list[dict], executable: str, all_names: list[str]) -> str:
    eligible = {record["name"] for record in records
                if record["executable"] == executable and record["eligibility"] == "vulkan"}
    by_suite_all: dict[str, list[str]] = collections.defaultdict(list)
    for name in all_names:
        by_suite_all[name.rsplit(".", 1)[0]].append(name)
    patterns = []
    for suite in sorted(by_suite_all):
        suite_names = by_suite_all[suite]
        selected = [name for name in suite_names if name in eligible]
        if not selected:
            continue
        if len(selected) == len(suite_names):
            patterns.append(f"{suite}.*")
        else:
            patterns.extend(sorted(selected))
    return ":".join(patterns) + "\n"


def generated_files(trace_path: Path | None = None) -> tuple[dict[Path, bytes], dict[str, int], dict[str, list[str]]]:
    trace = load_trace(trace_path)
    registrations, instantiations = source_registrations()
    runtime = {name: run_list(name) for name in BINARIES}
    runtime_to_reg = {}
    failures = []
    for executable, names in runtime.items():
        for name in names:
            reg = registration_for_runtime(name, registrations)
            if reg is None:
                failures.append(f"unmapped runtime test: {executable}:{name}")
            else:
                runtime_to_reg[(executable, name)] = reg

    registration_hits = collections.Counter(runtime_to_reg.values())
    inactive_seen = set()
    for reg in registrations:
        inactive_key = (str(reg.file.relative_to(ROOT)), reg.suite, reg.test)
        if not registration_hits[reg] and inactive_key in INACTIVE_SOURCE_REGISTRATIONS:
            if not reg.conditional:
                failures.append(
                    f"inactive registration allowlist entry is no longer conditional: "
                    f"{reg.file.relative_to(ROOT)}:{reg.line} {reg.suite}.{reg.test}"
                )
            inactive_seen.add(inactive_key)
        elif not registration_hits[reg]:
            failures.append(
                f"source registration has no runtime name: {reg.file.relative_to(ROOT)}:{reg.line} "
                f"{reg.kind}({reg.suite}, {reg.test})"
            )
    for inactive_key in sorted(INACTIVE_SOURCE_REGISTRATIONS - inactive_seen):
        failures.append(f"stale inactive source registration allowlist entry: {inactive_key}")
    runtime_names = {name for names in runtime.values() for name in names}
    for prefix, fixture, path, line in instantiations:
        if not any(name.startswith(f"{prefix}/{fixture}.") for name in runtime_names):
            failures.append(
                f"parameter instantiation has no runtime name: {path.relative_to(ROOT)}:{line} "
                f"INSTANTIATE_TEST_SUITE_P({prefix}, {fixture}, ...)"
            )
    if failures:
        raise ValueError("\n".join(failures))

    helpers, fixtures, fixture_bases = extract_helpers()
    allowlist = load_allowlist()
    stale_allowlist = set(allowlist) - runtime_names
    if stale_allowlist:
        raise ValueError(
            "skip allowlist names no longer exist: " + ", ".join(sorted(stale_allowlist))
        )
    records = []
    analysis_cache = {}
    for executable in BINARIES:
        for name in runtime[executable]:
            reg = runtime_to_reg[(executable, name)]
            if reg not in analysis_cache:
                fragments = reachable_fragments(reg, helpers, fixtures, fixture_bases)
                launchers = launcher_rows(fragments)
                eligibility, reason = classify(reg, fragments, launchers)
                analysis_cache[reg] = (launchers, eligibility, reason)
            launchers, eligibility, reason = analysis_cache[reg]
            record = {
                "name": name,
                "executable": executable,
                "file": str(reg.file.relative_to(ROOT)),
                "line": reg.line,
                "eligibility": eligibility,
            }
            if reason is not None:
                record["reason"] = reason
            record["launchers"] = launchers
            record["launchers_source"] = "inferred"
            if name in trace:
                measured = sorted({
                    row for entry_name, count in trace[name].items() if count
                    for row in ENTRY_TO_LAUNCHERS.get(entry_name, [])
                })
                record["launchers"] = measured
                record["launchers_source"] = "measured"
                # Only the set of entries is recorded: counts vary with loop
                # iterations and timing, and --check is a classification gate.
                moved_storage = any(
                    count for entry_name, count in trace[name].items()
                    if entry_name.startswith("service_"))
                if not measured and not moved_storage and record["eligibility"] == "vulkan":
                    record["eligibility"] = "no-gpu-work"
                    record["reason"] = "measured: no facade entry executed"
            record["skip_allowlisted"] = name in allowlist
            if name in allowlist:
                record["skip_reason"] = allowlist[name]["reason"]
            records.append(record)
    records.sort(key=lambda record: (record["executable"], record["name"]))

    manifest = (json.dumps(records, indent=2, ensure_ascii=True) + "\n").encode()
    filters = {
        executable: make_filter(records, executable, runtime[executable])
        for executable in BINARIES
    }
    files = {
        OUTPUTS["manifest"]: manifest,
        OUTPUTS["lichtfeld_tests"]: filters["lichtfeld_tests"].encode(),
        OUTPUTS["tensor_hardening_tests"]: filters["tensor_hardening_tests"].encode(),
#gtest reads one flag per line from-- gtest_flagfile; the joined filter
#exceeds the Windows command - line limit, the flag file does not .
        OUTPUTS["lichtfeld_tests_flags"]:
            ("--gtest_filter=" + filters["lichtfeld_tests"]).encode(),
        OUTPUTS["tensor_hardening_tests_flags"]:
            ("--gtest_filter=" + filters["tensor_hardening_tests"]).encode(),
    }
    counts = {}
    for executable in BINARIES:
        expected = sum(
            record["executable"] == executable and record["eligibility"] == "vulkan"
            for record in records
        )
        actual_names = run_list(executable, filters[executable].strip())
        if len(actual_names) != expected:
            raise ValueError(
                f"{executable} filter lists {len(actual_names)} cases, expected {expected}"
            )
        counts[executable] = expected
    return files, counts, runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify generated files are current")
    parser.add_argument("--trace", type=Path, default=None,
                        help="per-test facade entry counts (JSON lines) from LFS_TENSOR_FACADE_TRACE")
    args = parser.parse_args()
    try:
        files, filter_counts, _ = generated_files(args.trace)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.check:
        stale = False
        for path, expected in files.items():
            actual = path.read_bytes() if path.is_file() else b""
            if actual != expected:
                stale = True
                print(f"stale generated file: {path.relative_to(ROOT)}", file=sys.stderr)
                if path.suffix == ".json" and actual:
                    diff = difflib.unified_diff(
                        actual.decode().splitlines(), expected.decode().splitlines(),
                        fromfile="current", tofile="generated", n=2,
                    )
                    print("\n".join(list(diff)[:40]), file=sys.stderr)
        if stale:
            return 1
    else:
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    records = json.loads(files[OUTPUTS["manifest"]])
    histogram = collections.Counter(record["eligibility"] for record in records)
    print(f"records: {len(records)}")
    print("eligibility: " + ", ".join(f"{name}={histogram[name]}" for name in sorted(histogram)))
    print(f"lichtfeld_tests filter cases: {filter_counts['lichtfeld_tests']}")
    print(f"tensor_hardening_tests filter cases: {filter_counts['tensor_hardening_tests']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
