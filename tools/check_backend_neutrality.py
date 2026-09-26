#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reject tensor implementation details in backend-neutral source.

New-backend checklist:
* Register the backend in kGpuBackends and implement every GpuBackendOps entry.
* Implement allocation, copies, synchronization, memory accounting and services.
* Provide the expression emitter and launcher, including dtype and bounds rules.
* Supply the SPIR-V/PTX-style code-generation hook and cache identity.
* Run the tensor contract, parity and application suites with that backend selected.

Private tensor includes and symbols are forbidden throughout consumers. Native
CUDA API and backend branches are checked only in the viewer boundary below.
The header-only tensor implementation lives in src/core/include/core/detail and
may be included only from core/tensor.hpp. Headers under src/core/tensor are
private: they compile only where LFS_TENSOR_PRIVATE_ACCESS is set (lfs_core, the
trainer seam, and tensor tests). This is a lexical source gate, not C++ data-flow analysis.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cuh", ".cu"}
# Native API rules apply to the viewer and its tensor consumers.
VIEWER_ROOTS = (
    "src/visualizer/", "src/app/", "src/python/", "src/mcp/",
    "src/io/formats/", "src/io/project/", "src/rendering/",
)
TRAINER_ROOTS = ("src/training/", "src/visualizer/training/")
CUDA_RASTERIZERS = ("src/rendering/rasterizer/cuda/",)
# File-specific native interoperability and backend selection seams.
SEAMS = {
    "src/app/main.cpp": "Process startup selects and initializes the execution backend.",
    "src/app/application.cpp": "Application startup coordinates the viewer and CUDA trainer devices.",
    "src/python/lfs/py_tensor.cpp": "DLPack exposes CUDA device and stream interoperability.",
    "src/visualizer/rendering/vksplat_viewport_renderer.cpp": "Viewport interop connects Vulkan rendering to the CUDA trainer.",
    "src/visualizer/rendering/vksplat_viewport_renderer.hpp": "Viewport interop owns the CUDA trainer handshake handles.",
    "src/visualizer/gui/gpu_memory_query.cpp": "Device telemetry queries native GPU identity and memory.",
    "src/rendering/raster_rendering_engine.cpp": "Raster engine dispatches the native CUDA point-cloud renderer.",
    "src/visualizer/project/project_lifecycle.cpp": "Project loading binds the CUDA trainer thread to its device.",
    "src/visualizer/gui/gui_manager.cpp": "The GUI reports failure of the selected CUDA runtime.",
    "src/visualizer/preferences.cpp": "Preferences select a backend before tensor initialization.",
}
# Exact private-interface exceptions pending ownership changes in other lanes.
PRIVATE_SEAMS = {
    "src/core/memory_pressure.cpp": {
        ("private-header", "core/tensor/backend/cuda/runtime/memory_pool.hpp"):
            "Core-owned CUDA pressure client and allocator use the CUDA pool.",
    },
    "src/core/cuda/diagnostics_compile_check.cpp": {
        ("*", "*"): "Compile-only probe checks the private tensor/CUDA contract.",
    },
    "src/core/cuda/diagnostics_compile_check.cu": {
        ("*", "*"): "Compile-only probe checks the private tensor/CUDA contract.",
    },
    "src/core/cuda/lanczos_resize/lanczos_resize.cu": {
        ("private-header", "core/tensor/backend/cuda/runtime/cuda_memory_guard.hpp"):
            "converted by lane F2b",
        ("private-symbol", "internal::resize_image_prior_tensor"):
            "converted by lane F2b",
    },
    "src/core/cuda/undistort/undistort.cu": {
        ("private-symbol", "internal::undistort_image_tensor"):
            "converted by lane F2b",
    },
    "src/visualizer/rendering/rendering_manager_vulkan.cpp": {
        ("private-symbol", "lfs::core::internal::undistort_image_tensor"):
            "converted by lane F2b",
    },
}
INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^>"\n]+)[>"]', re.M)
CUDA_HEADER = re.compile(r'^(?:cuda(?:_runtime(?:_api)?|_fp16|_bf16)?\.h|driver_types\.h|vector_types\.h|cublas.*\.h|curand.*\.h|cub/.*)$')
NATIVE_CALL = re.compile(r'\b(cuda[A-Z]\w*|cu[A-Z]\w*)\s*\(')
CUDA_STREAM = re.compile(r'\bcudaStream_t\b')
BACKEND_BRANCH = re.compile(
    r'\bcase\s+(?:(?:lfs::)?core::)?GpuBackend::\w+|'
    r'(?:==|!=)\s*(?:(?:lfs::)?core::)?GpuBackend::\w+|'
    r'(?:(?:lfs::)?core::)?GpuBackend::\w+\s*(?:==|!=)')
INTERNAL_SYMBOL = re.compile(r'(?<![\w:])(?:(?:lfs::)?core::)?internal::\w+')
CORE_SCOPE = re.compile(r'\b(?:namespace|using\s+namespace)\s+lfs::core(?:\s*[;{]|::nn\b)')
INTERNAL_NAMESPACE = re.compile(r'\b(?:using\s+namespace|namespace\s+\w+\s*=)\s+(?:(?:lfs::)?core::)internal\b')
TOKEN = re.compile(r'/\*[\s\S]*?\*/|//[^\n]*|R"([^ ()\\\t\r\n]{0,16})\([\s\S]*?\)\1"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def mask_tokens(source: str, *, strings: bool = True) -> str:
    def replace(match: re.Match) -> str:
        text = match.group()
        if not strings and not text.startswith(("//", "/*")):
            return text
        return re.sub(r'[^\n]', ' ', text)
    return TOKEN.sub(replace, source)


def private_header(path: Path, header: str) -> bool:
    if re.search(r'(?:^|/)core/tensor/', header):
        return True
    resolved = (path.parent / header).resolve()
    tensor_root = PROJECT_ROOT / "src/core/tensor"
    return (resolved.is_relative_to(tensor_root) or
            (tensor_root / header).is_file() or
            (tensor_root / "internal" / header).is_file())


def scan(path: Path) -> list[tuple[int, str, str]]:
    relative = path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)
    if relative.startswith(("src/core/tensor/", "src/core/include/core/detail/", *TRAINER_ROOTS)):
        return []
    # Tensor tests verify the backend contract and may inspect implementation state.
    if relative.startswith("tests/") and path.name.startswith(("test_tensor_", "tensor_", "bench_tensor_", "test_allocator_",
                                                               "test_cuda_", "test_stale_stream_", "test_zero_stride_",
                                                               "test_strided_reduce_", "expression_reference",
                                                               "cuda_backend_test", "test_alloc_counter", "test_zip_gather")):
        return []
    source = path.read_text(encoding="utf-8")
    comments_removed = mask_tokens(source, strings=False)
    code = mask_tokens(source)
    core_scope = CORE_SCOPE.search(code) is not None
    native_checked = ((not path.is_relative_to(PROJECT_ROOT) or relative.startswith(VIEWER_ROOTS)) and
                      not relative.startswith(CUDA_RASTERIZERS) and relative not in SEAMS)
    findings = []

    def add(position: int, kind: str, text: str) -> None:
        findings.append((source.count('\n', 0, position) + 1, kind, text))

    for match in INCLUDE.finditer(comments_removed):
        header = match[1]
        normalized = header.replace("\\", "/")
        if normalized.startswith("core/detail/") or "/core/detail/" in normalized:
            if relative != "src/core/include/core/tensor.hpp":
                add(match.start(), "private-header", header)
        elif private_header(path, header):
            add(match.start(), "private-header", header)
        if native_checked and CUDA_HEADER.fullmatch(header):
            add(match.start(), "cuda-header", header)
    if native_checked:
        for match in NATIVE_CALL.finditer(code):
            add(match.start(), "native-api", match[1])
        for match in CUDA_STREAM.finditer(code):
            add(match.start(), "cuda-stream", match.group())
        for match in BACKEND_BRANCH.finditer(code):
            add(match.start(), "backend-branch", match.group())
    for match in INTERNAL_NAMESPACE.finditer(code):
        add(match.start(), "private-symbol", match.group())
    for match in INTERNAL_SYMBOL.finditer(code):
        if match.group().startswith("internal") and not core_scope:
            continue
        add(match.start(), "private-symbol", match.group())
    exceptions = PRIVATE_SEAMS.get(relative, {})
    return sorted({finding for finding in findings
                   if (finding[1], finding[2]) not in exceptions and
                   ("*", "*") not in exceptions})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", help="File or directory to scan; repeatable (default: src and tests).")
    args = parser.parse_args()
    files = set()
    for root in args.root or [PROJECT_ROOT / "src", PROJECT_ROOT / "tests"]:
        root = root.resolve()
        if not root.exists():
            parser.error(f"scan root does not exist: {root}")
        files.update([root] if root.is_file() else (p for p in root.rglob("*") if p.is_file() and p.suffix in SUFFIXES))
    count = 0
    for path in sorted(files):
        for line, kind, text in scan(path):
            label = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
            print(f"{label}:{line}: [{kind}] {text}")
            count += 1
    print(f"Backend neutrality: {count} violation(s).")
    return bool(count)


if __name__ == "__main__":
    raise SystemExit(main())
