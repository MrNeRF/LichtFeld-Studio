# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Adapt the project's own functors without editing CUDA or host sources."""
import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
source = (args.source / 'tensor_functors.hpp').read_text(encoding='utf-8')
source = source.replace('__CUDACC__', '__HIPCC__').replace('__CUDA_ARCH__', '__HIP_DEVICE_COMPILE__')
# HIP fmaxf chooses +0 for (-0,+0); preserve the public CPU ReLU's -0.
source = source.replace('isnan(static_cast<float>(x)) ? x : fmaxf(x, T(0))',
                        'isnan(static_cast<float>(x)) ? x : (x < T(0) ? T(0) : x)')
(args.output / 'hip_tensor_functors.hpp').write_text(source, encoding='utf-8')
rows = re.findall(r'LFS_POINTWISE_OP\((\w+),\s*([^,]+),\s*"([^"]+)"\)', (args.source / 'pointwise_ops.def').read_text(encoding='utf-8'))
scalar_ops = {'Add': 'add', 'Sub': 'sub', 'Mul': 'mul', 'Div': 'div', 'Pow': 'pow', 'Mod': 'mod', 'Maximum': 'maximum', 'Minimum': 'minimum', 'Equal': 'equal', 'NotEqual': 'not_equal', 'Less': 'less', 'LessEqual': 'less_equal', 'Greater': 'greater', 'GreaterEqual': 'greater_equal'}
output = []
half_output = []
for ident, functor, _ in rows:
    mode = 0
    if ident.endswith('Tensor'):
        mode = 2
    elif ident.endswith('Scalar') and ident[:-6] in scalar_ops:
        mode = 1
        functor = 'ops::' + scalar_ops[ident[:-6]] + '_op'
    predicate = ident.startswith(('Equal', 'NotEqual', 'Less', 'Greater', 'Logical', 'IsNan', 'IsInf', 'IsFinite'))
    if ident in ('ExpThenMulScalar', 'MulScalarThenAbs', 'MulScalarThenRelu'):
        first, second = {'ExpThenMulScalar': ('ops::exp_op', 'ops::mul_op'), 'MulScalarThenAbs': ('ops::mul_op', 'ops::abs_op'), 'MulScalarThenRelu': ('ops::mul_op', 'ops::relu_op')}[ident]
        expression = {'ExpThenMulScalar': 'ops::exp_op{}(x) * scalar', 'MulScalarThenAbs': 'ops::abs_op{}(x * scalar)', 'MulScalarThenRelu': 'ops::relu_op{}(x * scalar)'}[ident]
        output.append(f'case LFS_HIP_OP_{ident}: launch_pointwise<0, float>(context, lhs, rhs, output, count, scalar, [=] __device__ (float x) {{ return {expression}; }}); break;')
    else:
        output.append(f'case LFS_HIP_OP_{ident}: launch_pointwise<{mode}, {"uint8_t" if predicate else "float"}>(context, lhs, rhs, output, count, scalar, {functor}{{}}); break;')
    half_output.append(output[-1].replace(f'launch_pointwise<{mode}, float>', f'launch_pointwise<{mode}, __half, __half>')
                       .replace(f'launch_pointwise<{mode}, uint8_t>', f'launch_pointwise<{mode}, uint8_t, __half>'))
(args.output / 'hip_tensor_dispatch.inc').write_text('\n'.join(output) + '\n', encoding='utf-8')
(args.output / 'hip_tensor_half_dispatch.inc').write_text('\n'.join(half_output) + '\n', encoding='utf-8')

evaluation = []
flags = []
for ident, functor, _ in rows:
    mode = 0
    if ident.endswith('Tensor'):
        mode = 2
    elif ident.endswith('Scalar') and ident[:-6] in scalar_ops:
        mode = 1
        functor = 'ops::' + scalar_ops[ident[:-6]] + '_op'
    predicate = ident.startswith(('Equal', 'NotEqual', 'Less', 'Greater', 'Logical', 'IsNan', 'IsInf', 'IsFinite'))
    flags.append(str(mode | (4 if predicate else 0)))
    if predicate:
        continue
    if ident in ('ExpThenMulScalar', 'MulScalarThenAbs', 'MulScalarThenRelu'):
        expression = {'ExpThenMulScalar': 'ops::exp_op{}(x) * scalar', 'MulScalarThenAbs': 'ops::abs_op{}(x * scalar)', 'MulScalarThenRelu': 'ops::relu_op{}(x * scalar)'}[ident]
    elif mode == 2:
        expression = f'{functor}{{}}(x, y)'
    elif mode == 1:
        expression = f'{functor}{{}}(x, scalar)'
    else:
        expression = f'{functor}{{}}(x)'
    evaluation.append(f'case LFS_HIP_OP_{ident}: return {expression};')
(args.output / 'hip_tensor_eval.inc').write_text('\n'.join(evaluation) + '\n', encoding='utf-8')
(args.output / 'hip_tensor_flags.inc').write_text(','.join(flags), encoding='utf-8')

# Broadcast uses the same specialized binary functors, including predicates.
broadcast = []
for ident, functor, _ in rows:
    if not ident.endswith('Tensor'):
        continue
    predicate = ident.startswith(('Equal', 'NotEqual', 'Less', 'Greater', 'Logical'))
    broadcast.append(f'case LFS_HIP_OP_{ident}: launch_broadcast<{"uint8_t" if predicate else "float"}>(context, lhs, rhs, output, a, b, out, {functor}{{}}); break;')
(args.output / 'hip_tensor_broadcast.inc').write_text('\n'.join(broadcast) + '\n', encoding='utf-8')

half_broadcast = [line.replace("launch_broadcast<float>", "launch_broadcast<__half, __half>").replace("launch_broadcast<uint8_t>", "launch_broadcast<uint8_t, __half>") for line in broadcast]
(args.output / "hip_tensor_half_broadcast.inc").write_text("\n".join(half_broadcast) + "\n", encoding="utf-8")
