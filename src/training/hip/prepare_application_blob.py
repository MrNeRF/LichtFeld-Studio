# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Lower the repository blob shader to native HIP without a CUDA SDK."""
import argparse
from pathlib import Path
import re

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
lines=a.source.read_text(encoding='utf-8').splitlines()
if not lines or 'slang-cuda-prelude.h' not in lines[0]:
    raise RuntimeError('Review changed blob shader prelude')
s='\n'.join(lines[1:])
for token in ('extern "C" __constant__ GlobalParams_0 SLANG_globalParams;',
              '#define globalParams_0 (&SLANG_globalParams)'):
    if s.count(token)!=1:raise RuntimeError('Review changed blob globals')
    s=s.replace(token,'')
s,count=re.subn(r'extern "C" __global__ void main_0\(\)',
    'template<unsigned Operation> __global__ void blob_kernel(Params_0 native_parameters)',s)
if count!=1:raise RuntimeError('Review changed blob entry')
s,count=re.subn(r'ulonglong (\w+) = __ldg\(&globalParams_0->push_0->parameters_0\);',
    r'ulonglong \1 = reinterpret_cast<ulonglong>(&native_parameters);',s)
if count!=1:raise RuntimeError('Review changed blob parameter indirection')
s=s.replace('(globalParams_0->operation_0)','Operation')
if 'globalParams_0' in s or 'SLANG_globalParams' in s:
    raise RuntimeError('Unconverted blob globals')
shared=re.findall(r'__device__ __shared__ ([^;]+);',s)
if len(shared)!=3:raise RuntimeError('Review changed blob shared storage')
s=re.sub(r'__device__ __shared__ [^;]+;','',s)
s=s.replace('void blob_kernel(Params_0 native_parameters)\n{',
    'void blob_kernel(Params_0 native_parameters)\n{\n'+'\n'.join('__shared__ '+v+';' for v in shared))
s=re.sub(r'((?:__device__|__global__)[^;{}]*\{)',r'\1\n#pragma clang fp contract(off)\n',s)
known={'F32_ceil','F32_log','F32_exp','F32_round','F32_max','F32_min','I32_min','I32_max','U32_countbits'}
unknown=set(re.findall(r'\b(?:F32|U32|I32|F64|U64|I64)_[A-Za-z0-9_]+',s))-known
if unknown:raise RuntimeError('Missing HIP blob intrinsics: '+str(sorted(unknown)))
prelude='''
using uint=unsigned;using uchar=unsigned char;using ulonglong=unsigned long long;
template<class T,int N>struct FixedArray {
    T data[N];
    __device__ T& operator[](unsigned i){return data[i];}
    __device__ const T& operator[](unsigned i)const{return data[i];}
};
__device__ inline float F32_ceil(float x){return ceilf(x);}
__device__ inline float F32_log(float x){return logf(x);}
__device__ inline float F32_exp(float x){return expf(x);}
__device__ inline float F32_round(float x){return roundf(x);}
__device__ inline float F32_max(float x,float y){return fmaxf(x,y);}
__device__ inline float F32_min(float x,float y){return fminf(x,y);}
__device__ inline int I32_min(int x,int y){return x<y?x:y;}
__device__ inline int I32_max(int x,int y){return x>y?x:y;}
__device__ inline uint U32_countbits(uint x){return __popc(x);}
'''
header='/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors\n * SPDX-License-Identifier: GPL-3.0-or-later */\n#pragma once\n#include <hip/hip_runtime.h>\nnamespace lfs_hip_application_blob_kernels {\n'+prelude+s+'\n}\n'
a.output.parent.mkdir(parents=True,exist_ok=True)
if not a.output.exists() or a.output.read_text(encoding='utf-8')!=header:
    a.output.write_text(header,encoding='utf-8')
