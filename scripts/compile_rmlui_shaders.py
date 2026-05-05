#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VULKAN_SHADER_DIR = REPO_ROOT / "src" / "visualizer" / "gui" / "rmlui" / "vulkan"
SOURCE_DIR = VULKAN_SHADER_DIR / "shaders"
DEFAULT_OUTPUT = VULKAN_SHADER_DIR / "rmlui_shaders_spv.hpp"
SHADERS = (
    ("shader_frag_color.frag", "shader_frag_color"),
    ("shader_frag_texture.frag", "shader_frag_texture"),
    ("shader_vert.vert", "shader_vert"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile RmlUi Vulkan GLSL shaders into the embedded SPIR-V header.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Header to write.")
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build", help="CMake build directory.")
    parser.add_argument("--compiler", type=Path, help="Explicit glslc, glslangValidator, or lfs_shader_compiler path.")
    parser.add_argument("--check", action="store_true", help="Fail if the generated header differs from the current file.")
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def find_lfs_shader_compiler(build_dir: Path) -> Path | None:
    names = ("lfs_shader_compiler", "lfs_shader_compiler.exe")
    if build_dir.exists():
        for name in names:
            direct = build_dir / "src" / "visualizer" / name
            if direct.exists():
                return direct
        for name in names:
            matches = sorted(build_dir.rglob(name))
            if matches:
                return matches[0]

    if build_dir.exists():
        run(["cmake", "--build", str(build_dir), "--target", "lfs_shader_compiler"])
        for name in names:
            matches = sorted(build_dir.rglob(name))
            if matches:
                return matches[0]
    return None


def choose_compiler(args: argparse.Namespace) -> tuple[str, Path]:
    if args.compiler:
        compiler = args.compiler.resolve()
        name = compiler.name.lower()
        if "lfs_shader_compiler" in name:
            return "lfs", compiler
        if "glslangvalidator" in name:
            return "glslang", compiler
        return "glslc", compiler

    if glslc := shutil.which("glslc"):
        return "glslc", Path(glslc)
    if glslang := shutil.which("glslangValidator"):
        return "glslang", Path(glslang)
    if helper := find_lfs_shader_compiler(args.build_dir):
        return "lfs", helper

    raise SystemExit(
        "No shader compiler found. Install glslc/glslangValidator or build "
        "the in-tree helper with: cmake --build build --target lfs_shader_compiler"
    )


def compile_with_cli(kind: str, compiler: Path, shader: Path, temp_dir: Path) -> bytes:
    spv_path = temp_dir / f"{shader.stem}.spv"
    if kind == "glslc":
        run([str(compiler), str(shader), "-o", str(spv_path)])
    else:
        run([str(compiler), "-V", str(shader), "-o", str(spv_path)])
    return spv_path.read_bytes()


def compile_with_lfs(compiler: Path, shader: Path, symbol: str, temp_dir: Path) -> bytes:
    header_path = temp_dir / f"{symbol}.hpp"
    run([str(compiler), "--input", str(shader), "--output", str(header_path), "--symbol", symbol])
    words = [int(value, 16) for value in re.findall(r"0x([0-9A-Fa-f]{8})u", header_path.read_text())]
    if not words:
        raise RuntimeError(f"No SPIR-V words found in generated helper header {header_path}")
    return b"".join(struct.pack("<I", word) for word in words)


def format_byte_array(symbol: str, data: bytes) -> str:
    lines = [f"alignas(uint32_t) static const unsigned char {symbol}[] = {{"]
    for offset in range(0, len(data), 20):
        chunk = data[offset:offset + 20]
        lines.append("\t" + ",".join(f"0x{byte:02X}" for byte in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def generate_header(kind: str, compiler: Path) -> str:
    arrays: list[str] = []
    with tempfile.TemporaryDirectory(prefix="lfs-rmlui-shaders-") as temp:
        temp_dir = Path(temp)
        for filename, symbol in SHADERS:
            shader = SOURCE_DIR / filename
            print(f"Compiling {shader.relative_to(REPO_ROOT)} with {compiler}")
            if kind == "lfs":
                data = compile_with_lfs(compiler, shader, symbol, temp_dir)
            else:
                data = compile_with_cli(kind, compiler, shader, temp_dir)
            arrays.append(format_byte_array(symbol, data))

    body = "\n\n".join(arrays)
    return (
        "/* Adapted from RmlUi 6.2 Backends/RmlUi_Vulkan/ShadersCompiledSPV.h.\n"
        " *\n"
        " * SPDX-License-Identifier: MIT */\n\n"
        "#pragma once\n\n"
        "// RmlUi SPIR-V Vulkan shaders compiled using command: "
        "'python scripts/compile_rmlui_shaders.py'. Do not edit manually.\n\n"
        "#include <stdint.h>\n\n"
        f"{body}\n"
    )


def main() -> int:
    args = parse_args()
    kind, compiler = choose_compiler(args)
    generated = generate_header(kind, compiler)

    output = args.output.resolve()
    if args.check:
        current = output.read_text() if output.exists() else ""
        if current != generated:
            print(f"{output.relative_to(REPO_ROOT)} is not up to date", file=sys.stderr)
            return 1
        return 0

    output.write_text(generated)
    print(f"Wrote {output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
