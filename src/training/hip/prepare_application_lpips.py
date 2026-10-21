# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Lower our portable LPIPS program to native HIP without a CUDA SDK/prelude.

Slang's CUDA-text target is an intermediate representation only. Arithmetic is
retained; the small, explicit intrinsic seam below uses HIP/OCML functions.
Operation selection is a template argument, avoiding mutable device globals.
"""
import argparse
from pathlib import Path
import re

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
source=a.source.read_text(encoding='utf-8')
lines=source.splitlines()
if len(lines)<2 or 'slang-cuda-prelude.h' not in lines[0]:
    raise RuntimeError('Review changed Slang prelude before generating HIP')
source='\n'.join(lines[1:])
declaration='extern "C" __constant__ GlobalParams_0 SLANG_globalParams;'
definition='#define globalParams_0 (&SLANG_globalParams)'
kernel='extern "C" __global__ void main_0(Parameters_0 p_0)'
for token in (declaration,definition,kernel):
    if source.count(token)!=1:raise RuntimeError('Review changed Slang LPIPS boundary: '+token)
source=source.replace(declaration,'').replace(definition,'')
source=source.replace(kernel,'template<unsigned Operation> __global__ void lpips_kernel(Parameters_0 p_0)')
source=source.replace('(globalParams_0->kOperation_0)','Operation')
if 'globalParams_0' in source or 'SLANG_globalParams' in source:
    raise RuntimeError('Unconverted mutable Slang globals')
source=source.replace('(blockIdx * blockDim + threadIdx).x','(blockIdx.x*blockDim.x+threadIdx.x)')
# Native grids cover the whole dispatch and need no Vulkan chunk offset.
# In pool_reduce, g is instead the optional pixel-weight pointer and must not
# be truncated into a thread offset or overwritten with a chunk base.
source,replaced=re.subn(r'uint (\w+) = uint\(p_0\.g_0\) \+ (\w+);',r'uint \1 = \2;',source)
if replaced!=1:raise RuntimeError('Review changed LPIPS dispatch index before generating HIP')
source=re.sub(r'((?:__device__|__global__)[^;{}]*\{)',r'\1\n#pragma clang fp contract(off)\n',source)
known={'F32_asuint','U32_asfloat','F32_abs','F32_exp','F32_tanh','F32_rsqrt','F32_fma','F32_max','I32_min','I32_max','U32_max'}
unknown=set(re.findall(r'\b(?:F32|U32|I32|F64|U64|I64)_[A-Za-z0-9_]+',source))-known
if unknown:raise RuntimeError('HIP intrinsic implementations required: '+', '.join(sorted(unknown)))
prelude='''
using uint=unsigned int;using ulonglong=unsigned long long;
template<class T,int N>struct FixedArray {
    T data[N];
    __device__ T& operator[](unsigned i){return data[i];}
    __device__ const T& operator[](unsigned i)const{return data[i];}
};
__device__ inline uint F32_asuint(float x){return __float_as_uint(x);}
__device__ inline float U32_asfloat(uint x){return __uint_as_float(x);}
__device__ inline float F32_abs(float x){return fabsf(x);}
__device__ inline float F32_exp(float x){return expf(x);}
__device__ inline float F32_tanh(float x){return tanhf(x);}
__device__ inline float F32_rsqrt(float x){return rsqrtf(x);}
__device__ inline float F32_fma(float x,float y,float z){return fmaf(x,y,z);}
__device__ inline float F32_max(float x,float y){return fmaxf(x,y);}
__device__ inline int I32_min(int x,int y){return x<y?x:y;}
__device__ inline int I32_max(int x,int y){return x>y?x:y;}
__device__ inline uint U32_max(uint x,uint y){return x>y?x:y;}
'''
header='''/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <hip/hip_runtime.h>
namespace lfs_hip_application_lpips_kernels {
'''+prelude+source+'\n}\n'
a.output.parent.mkdir(parents=True,exist_ok=True)
if not a.output.exists() or a.output.read_text(encoding='utf-8')!=header:
    a.output.write_text(header,encoding='utf-8')
