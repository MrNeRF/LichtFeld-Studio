#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract relocatable opcode instruction sequences from the build-time PTX catalog."""
import argparse
import hashlib
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ptx', type=Path)
parser.add_argument('--tensor', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--depfile', type=Path, required=True)
args = parser.parse_args()
if not args.ptx:
    paths = {Path(__file__),
             *args.tensor.glob('expression_*.cpp'), *args.tensor.glob('internal/expression_*.hpp'),
             *args.tensor.glob('backend/cuda/*expression*.cpp'),
             *args.tensor.glob('backend/vulkan/*expression*.cpp'),
             *args.tensor.glob('backend/metal/*expression*.cpp'),
             args.tensor / 'backend/vulkan/spirv_module.cpp'}
    hash_value = hashlib.sha256()
    for path in sorted(paths):
        hash_value.update(path.read_bytes())
    args.output.write_text('#include "internal/expression_emitter.hpp"\nnamespace lfs::core::internal {\nstd::string_view expression_emitter_hash() { return "' + hash_value.hexdigest() + '"; }\n}')
    escape = lambda path: str(path).replace(' ', '\\ ').replace('#', '\\#')
    args.depfile.write_text(escape(args.output) + ': ' + ' '.join(escape(p) for p in sorted(paths)) + '\n')
    raise SystemExit
text = re.sub(r'//[^\n]*', '', args.ptx.read_text())
header = '\n'.join(re.findall(r'^\.(?:version|target|address_size)[^\n]*', text, re.M)) + '\n'
# The scalar catalog uses the PTX 8.0 instruction set on SM 70 through 90.
architecture = int(re.search(r'\.target sm_(\d+)', header)[1])
if architecture <= 90:
    header = re.sub(r'\.version \S+', '.version 8.0', header)
entries = {}
for match in re.finditer(r'\.visible\s+\.func\s*\([^)]*\)\s+expr_(\w+)\s*\([^)]*\)\s*\{', text):
    name = match[1]
    start, depth, end = match.end(), 1, match.end()
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    body = text[start:end-1]
    body = re.sub(r'ld\.param\.(\w+)\s+(%\w+),\s*\[expr_\w+_param_([012])\];',
                  lambda m: f'{"cvt.u16.u32" if m[1] == "u16" else "mov.b32"} {m[2]}, @{"ABC"[int(m[3])]}@;', body)
    body = re.sub(r'st\.param\.\w+\s+\[func_retval\w*(?:\+0)?\],\s*([^;]+);', r'mov.b32 @OUT@, \1;', body)
    body = re.sub(r'\bret;', 'bra @DONE@;', body)
    # Explicit rounding prevents contraction between separate IR operations.
    body = re.sub(r'\b(add|sub|mul)(\.ftz)?\.f32\b', r'\1.rn\2.f32', body)
    if re.search(r'\b(?:call|ld\.param|st\.param)\b', body):
        raise ValueError(f'non-relocatable opcode {name}')
    entries[name] = body + '\n@DONE@:\n'
ops_header = args.tensor / 'internal/expression_program.hpp'
ops = re.search(r'enum class ExprOp[^\{]*\{(.*?)\};', ops_header.read_text(), re.S)[1]
ops = re.findall(r'\b(\w+)\s*,', ops)
required = set(ops) - {'Load', 'Gather', 'Gather2', 'Gather3', 'Iota', 'Extent', 'Immediate', 'Store', 'Fold'}
if required != entries.keys() - {'PackHalf', 'UnpackHalf'}:
    raise ValueError('PTX catalog does not match ExprOp')
paths = {args.ptx, Path(__file__), ops_header,
         *args.tensor.glob('expression_*.cpp'), *args.tensor.glob('internal/expression_*.hpp'),
         *args.tensor.glob('backend/cuda/*expression*.cpp'),
         *args.tensor.glob('backend/vulkan/*expression*.cpp'),
         *args.tensor.glob('backend/metal/*expression*.cpp'),
         args.tensor / 'backend/vulkan/spirv_module.cpp'}
hash_value = hashlib.sha256()
for path in sorted(paths):
    hash_value.update(path.read_bytes())
lines = ['#include "internal/expression_emitter.hpp"', '#include <stdexcept>',
         'namespace lfs::core::internal {',
         'std::string_view expression_ptx_header() { return R"ptx(' + header + ')ptx"; }',
         'std::string_view expression_emitter_hash() { return "' + hash_value.hexdigest() + '"; }',
         'std::string_view expression_ptx_opcode(ExprOp op) { switch(op) {']
for name in ops:
    if name in required:
        lines.append(f'case ExprOp::{name}: return R"ptx({entries[name]})ptx";')
lines.extend(['default: throw std::invalid_argument("No arithmetic PTX for expression opcode"); }}',
              'std::string_view expression_ptx_half(bool pack) { return pack ? R"ptx(' + entries['PackHalf'] + ')ptx" : R"ptx(' + entries['UnpackHalf'] + ')ptx"; }', '}'])
args.output.write_text('\n'.join(lines))
escape = lambda path: str(path).replace(' ', '\\ ').replace('#', '\\#')
args.depfile.write_text(escape(args.output) + ': ' + ' '.join(escape(p) for p in sorted(paths)) + '\n')
