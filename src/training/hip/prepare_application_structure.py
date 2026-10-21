# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Lower our structure shader to HIP; no CUDA SDK or mutable device globals."""
import argparse
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
lines = a.source.read_text(encoding='utf-8').splitlines()
if not lines or 'slang-cuda-prelude.h' not in lines[0]:
    raise RuntimeError('Review changed structure shader prelude')
s = '\n'.join(lines[1:])
for token in ('extern "C" __constant__ GlobalParams_0 SLANG_globalParams;',
              '#define globalParams_0 (&SLANG_globalParams)'):
    if s.count(token) != 1:
        raise RuntimeError('Review changed structure global boundary')
    s = s.replace(token, '')
s, count = re.subn(r'extern "C" __global__ void main_0\(\)',
                  'template<unsigned Operation> __global__ void structure_kernel(Params_0 native_parameters)', s)
if count != 1:
    raise RuntimeError('Review changed structure entry point')
s, count = re.subn(r'ulonglong (\w+) = __ldg\(&globalParams_0->push_0->parameters_0\);',
                  r'ulonglong \1 = reinterpret_cast<ulonglong>(&native_parameters);', s)
if count != 1:
    raise RuntimeError('Review changed structure parameter indirection')
s = s.replace('(globalParams_0->operation_0)', 'Operation')
if 'globalParams_0' in s or 'SLANG_globalParams' in s:
    raise RuntimeError('Unconverted mutable structure globals')
# Read both reduction lanes before the block barrier. A read/modify/write of
# shared counts after that barrier races its XOR partner on independent waves.
pattern = r'(float3\s+other_(\d+) = \(\*&reduction_0\)\[(\w+) \^ (\w+)\];)'
s, count = re.subn(pattern, r'\1\n    float count_other_\2 = (*&count_reduction_0)[\3 ^ \4];', s)
if count != 2:
    raise RuntimeError('Review changed structure XOR reductions')
pattern = r'(\(\*&count_reduction_0\)\[(\w+)\] = \(\*&count_reduction_0\)\[\2\] \+ )\(\*&count_reduction_0\)\[\2 \^ offset_(\d+)\];'
reduction_indices = iter(range(2))
s, count = re.subn(pattern, lambda m: m[1]+'count_other_'+str(next(reduction_indices))+';', s)
if count != 2:
    raise RuntimeError('Review changed structure count reductions')
# HIP requires shared variables at block scope. Every operation owns its own
# specialization and scratch, so concurrent dispatches never share state.
shared = re.findall(r'__device__ __shared__ ([^;]+);', s)
if len(shared) != 6:
    raise RuntimeError('Review changed structure shared memory')
s = re.sub(r'__device__ __shared__ [^;]+;', '', s)
s = s.replace('void structure_kernel(Params_0 native_parameters)\n{',
              'void structure_kernel(Params_0 native_parameters)\n{\n' +
              '\n'.join('__shared__ '+v+';' for v in shared))
s = re.sub(r'((?:__device__|__global__)[^;{}]*\{)',
           r'\1\n#pragma clang fp contract(off)\n', s)
known = {'F32_asuint', 'U32_asfloat', 'F32_abs', 'F32_sqrt', 'F32_fma', 'F32_max', 'F32_min'}
unknown = set(re.findall(r'\b(?:F32|U32|I32|F64|U64|I64)_[A-Za-z0-9_]+', s)) - known
if unknown:
    raise RuntimeError('Missing HIP structure intrinsics: '+str(sorted(unknown)))
prelude = '''
using uint=unsigned; using uchar=unsigned char; using ulonglong=unsigned long long;
template<class T,int N>struct FixedArray {
    T data[N];
    __device__ T& operator[](unsigned i){return data[i];}
    __device__ const T& operator[](unsigned i)const{return data[i];}
};
__device__ inline uint F32_asuint(float x){return __float_as_uint(x);}
__device__ inline float U32_asfloat(uint x){return __uint_as_float(x);}
__device__ inline float F32_abs(float x){return fabsf(x);}
__device__ inline float F32_sqrt(float x){return sqrtf(x);}
__device__ inline float F32_fma(float x,float y,float z){return fmaf(x,y,z);}
__device__ inline float F32_max(float x,float y){return fmaxf(x,y);}
__device__ inline float F32_min(float x,float y){return fminf(x,y);}
'''
header = ('/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors\n'
          ' * SPDX-License-Identifier: GPL-3.0-or-later */\n#pragma once\n'
          '#include <hip/hip_runtime.h>\nnamespace lfs_hip_application_structure_kernels {\n' +
          prelude + s + '\n}\n')
a.output.parent.mkdir(parents=True, exist_ok=True)
if not a.output.exists() or a.output.read_text(encoding='utf-8') != header:
    a.output.write_text(header, encoding='utf-8')
