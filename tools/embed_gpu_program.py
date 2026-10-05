#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Embed build-time Slang artifacts, without any runtime filesystem dependency."""
import json
import pathlib
import sys


def main():
    arguments = sys.argv[1:]
    relaxed_math = "--relaxed-math" in arguments
    directory, name, *artifacts = [a for a in arguments if a != "--relaxed-math"]
    assert len(artifacts) % 5 == 0, f"Expected groups of 5 artifact arguments, got {len(artifacts)}"
    root = pathlib.Path(directory)
    declaration = f"std::span<const lfs::core::GpuKernelModule::Entry> {name}_entries()"
    (root / f"{name}.hpp").write_text(
        '#pragma once\n#include "core/gpu_kernel_module.hpp"\n' + declaration + ';\n', encoding="utf-8")
    source = [f'#include "{name}.hpp"', 'namespace {']
    entries = []
    for i in range(0, len(artifacts), 5):
        backend, stage, entry, path, reflection_path = artifacts[i:i + 5]
        reflection = json.loads(pathlib.Path(reflection_path).read_text(encoding="utf-8"))
        parameters = reflection["parameters"]
        assert len(parameters) == 1, f"{entry}: expected one global parameter block, got {len(parameters)}"
        layout = parameters[0]["type"]["elementVarLayout"]
        parameter_bytes = layout["binding"]["size"]
        fields = layout["type"]["fields"]
        offsets = [f["binding"]["offset"] for f in fields if f["type"]["kind"] == "pointer"]
        assert all(f["type"]["kind"] in ("pointer", "scalar", "vector", "matrix") for f in fields), \
            f"{entry}: nested/array parameter blocks are not supported by the tensor binding ABI"
        group = next(e for e in reflection["entryPoints"] if e["name"] == entry).get("threadGroupSize", [1, 1, 1])
        data = pathlib.Path(path).read_bytes()
        # PTX driver loading expects a zero-terminated string.
        if backend == "CUDA":
            data += b"\0"
        symbol = f"artifact_{i // 5}"
        source.append(f"const std::array<uint32_t, {len(offsets)}> {symbol}_offsets{{" + ','.join(map(str, offsets)) + '};')
        source.append(f"alignas(4) const unsigned char {symbol}[] = {{")
        source.extend(','.join(str(b) for b in data[j:j + 32]) + ',' for j in range(0, len(data), 32))
        source.append('};')
        entries.append(f'{{"{entry}", M::Stage::{stage}, lfs::core::GpuBackend::{backend}, '
                       f'std::as_bytes(std::span({symbol})), {parameter_bytes}, {symbol}_offsets, '
                       + '{' + ','.join(map(str, group)) + '}' + (', true' if relaxed_math else '') + '}')
    source += ['}', declaration + ' {', 'using M = lfs::core::GpuKernelModule;',
               'static const M::Entry entries[] = {' + ',\n'.join(entries) + '};', 'return entries;', '}']
    (root / f"{name}.cpp").write_text('\n'.join(source) + '\n', encoding="utf-8")


if __name__ == "__main__":
    main()
