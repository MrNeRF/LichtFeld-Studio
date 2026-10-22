# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Lower the repository's portable NN geometry kernels to isolated native HIP."""
import argparse
from pathlib import Path
import re
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
lines=a.source.read_text(encoding='utf-8').splitlines()
if not lines or 'slang-cuda-prelude.h' not in lines[0]:
    raise RuntimeError('Review changed inference prelude')
s='\n'.join(lines[1:])
for token in ('extern "C" __constant__ GlobalParams_0 SLANG_globalParams;', '#define globalParams_0 (&SLANG_globalParams)'):
    if s.count(token)!=1:raise RuntimeError('Review changed inference global boundary')
    s=s.replace(token,'')
s,n=re.subn(r'extern "C" __global__ void main_0\(Parameters_0 (\w+)\)',r'template<unsigned Operation,class Storage=float> __global__ void inference_kernel(Parameters_0 \1)',s)
if n!=1:raise RuntimeError('Review changed inference entry point')
s=s.replace('(globalParams_0->kOperation_0)','Operation')
if s.count('float *')!=5 or s.count('__device__ float sample_0(')!=1:raise RuntimeError('Review typed inference pointers')
s=s.replace('__device__ float sample_0(float * input_0,', 'template<class Pointer> __device__ float sample_0(Pointer input_0,')
s=s.replace('float * _S6', 'StoragePointer<Storage> _S6').replace('float * _S7', 'StoragePointer<Storage> _S7')
s=s.replace('(float *)p_1.inputAddress_0','(Storage *)p_1.inputAddress_0').replace('(float *)p_1.outputAddress_0','(Storage *)p_1.outputAddress_0')
if 'globalParams_0' in s or 'SLANG_globalParams' in s:raise RuntimeError('Unconverted inference globals')
s=s.replace('(blockIdx * blockDim + threadIdx).x','(blockIdx.x*blockDim.x+threadIdx.x)')
s=s.replace('SLANG_INFINITY','INFINITY')
s=re.sub(r'((?:__device__|__global__)[^;{}]*\{)',r'\1\n#pragma clang fp contract(off)\n',s)
known={'F32_abs','F32_exp','F32_floor','F32_max','F32_tanh','I32_min','I32_max'}
unknown=set(re.findall(r'\b(?:F32|U32|I32|F64|U64|I64)_[A-Za-z0-9_]+',s))-known
if unknown:raise RuntimeError('Missing inference intrinsic: '+str(sorted(unknown)))
prelude='''
using uint=unsigned;using ulonglong=unsigned long long;
// Every loaded value participates in FP32 arithmetic, including FP16 storage.
template<class T> struct FloatReference {
    T* value;
    __device__ operator float() const{return float(*value);}
    __device__ FloatReference& operator=(float x){*value=__float2half_rn(x);return *this;}
    __device__ FloatReference& operator=(const FloatReference& x){return *this=float(x);}
};
template<class T> struct FloatPointer {
    T* value;
    __device__ FloatPointer(T* p):value(p){}
    __device__ FloatPointer operator+(ptrdiff_t n)const{return value+n;}
    __device__ FloatReference<T> operator*()const{return {value};}
    __device__ FloatReference<T> operator[](ptrdiff_t n)const{return {value+n};}
};
template<class T> using StoragePointer=std::conditional_t<std::is_same_v<T,float>,float*,FloatPointer<T>>;
__device__ inline float F32_abs(float x){return fabsf(x);}
__device__ inline float F32_exp(float x){return expf(x);}
__device__ inline float F32_floor(float x){return floorf(x);}
__device__ inline float F32_max(float x,float y){return fmaxf(x,y);}
__device__ inline float F32_tanh(float x){return tanhf(x);}
__device__ inline int I32_min(int x,int y){return x<y?x:y;}
__device__ inline int I32_max(int x,int y){return x>y?x:y;}
'''
text='/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors\n * SPDX-License-Identifier: GPL-3.0-or-later */\n#pragma once\n#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n#include <type_traits>\nnamespace hip_inference {\n'+prelude+s+'\n}\n'
a.output.parent.mkdir(parents=True,exist_ok=True)
if not a.output.exists() or a.output.read_text(encoding='utf-8')!=text:a.output.write_text(text,encoding='utf-8')
