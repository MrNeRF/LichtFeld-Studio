# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Lower shared export algorithms to HIP without a CUDA prelude or toolkit."""
import argparse,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True);p.add_argument('--slang',required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
prelude="""
using uint=unsigned;using ulonglong=unsigned long long;
using ::make_float3;using ::make_float2;
__device__ inline float3 make_float3(float x){return ::make_float3(x,x,x);}
__device__ inline float2 make_float2(float x){return ::make_float2(x,x);}
template<class T,int N> struct FixedArray {
    T data[N];
    __device__ T& operator[](int i){return data[i];}
    __device__ const T& operator[](int i)const{return data[i];}
};
__device__ inline uint F32_asuint(float x){return __float_as_uint(x);}
__device__ inline float F32_fma(float x,float y,float z){return fmaf(x,y,z);}
__device__ inline float F32_min(float x,float y){return fminf(x,y);}
__device__ inline float F32_max(float x,float y){return fmaxf(x,y);}
__device__ inline float F32_abs(float x){return fabsf(x);}
__device__ inline float F32_exp(float x){return expf(x);}
__device__ inline float F32_log(float x){return logf(x);}
__device__ inline float F32_sqrt(float x){return sqrtf(x);}
__device__ inline float F32_pow(float x,float y){return powf(x,y);}
__device__ inline float F32_round(float x){return roundf(x);}
__device__ inline bool F32_isfinite(float x){return isfinite(x);}
__device__ inline bool F32_isinf(float x){return isinf(x);}
__device__ inline bool F32_isnan(float x){return isnan(x);}
__device__ inline float U32_asfloat(uint x){return __uint_as_float(x);}
__device__ inline uint U32_min(uint x,uint y){return x<y?x:y;}
__device__ inline int I32_max(int x,int y){return x>y?x:y;}
__device__ inline int I32_min(int x,int y){return x<y?x:y;}
__device__ inline float F32_ceil(float x){return ceilf(x);}
__device__ inline float F32_asin(float x){return asinf(x);}
__device__ inline float F32_atan2(float x,float y){return atan2f(x,y);}
struct bool3 {bool x,y,z;};
__device__ inline bool3 operator>=(float3 a,float3 b){return {a.x>=b.x,a.y>=b.y,a.z>=b.z};}
__device__ inline bool3 operator<=(float3 a,float3 b){return {a.x<=b.x,a.y<=b.y,a.z<=b.z};}
__device__ inline bool* _slang_vector_get_element_ptr(bool3* p,int i){return i==0?&p->x:i==1?&p->y:&p->z;}
__device__ inline bool _slang_vector_get_element(bool3 p,int i){return i==0?p.x:i==1?p.y:p.z;}
__device__ inline float* _slang_vector_get_element_ptr(float2* p,int i){return i==0?&p->x:&p->y;}
__device__ inline float _slang_vector_get_element(float2 p,int i){return i==0?p.x:p.y;}
__device__ inline uint* _slang_vector_get_element_ptr(uint3* p,int i){return i==0?&p->x:i==1?&p->y:&p->z;}
__device__ inline uint _slang_vector_get_element(uint3 p,int i){return i==0?p.x:i==1?p.y:p.z;}
__device__ inline float* _slang_vector_get_element_ptr(float3* p,int i){return i==0?&p->x:i==1?&p->y:&p->z;}
__device__ inline float _slang_vector_get_element(float3 p,int i){return i==0?p.x:i==1?p.y:p.z;}

"""
def expand(path,seen):
    path=path.resolve()
    if path in seen:return ''
    seen.add(path)
    text=path.read_text(encoding='utf-8')
    if path.name=='rounded_math.slangh':
        for name,op in [('roundedAdd','+'),('roundedSub','-'),('roundedMul','*'),('roundedDiv','/')]:
            start=text.index('float '+name+'(');body=text.index('{',start);depth=1;end=body+1
            while depth:depth+=(text[end]=='{')-(text[end]=='}');end+=1
            text=text[:body]+'{ precise float value = a '+op+' b; return value; }'+text[end:]
    return re.sub(r'^\s*#include "([^"]+)"',lambda m:expand(path.parent/m[1],seen),text,flags=re.M)
result='/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors\n * SPDX-License-Identifier: GPL-3.0-or-later */\n#pragma once\n'
result+='namespace hip_export_common {\n'+prelude+'\n}\n'
for name in ('export_kmeans','export_kmeans_accumulate','export_kmeans_group','export_radix','export_morton','export_decimate','export_decimate_cost','export_decimate_merge','point_raster'):
    source=expand(a.source/(name+'.slang'),set())
    token='spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); }'
    if source.count(token)!=(0 if name=='point_raster' else 1):raise RuntimeError('Review export grid lowering: '+name)
    source=source.replace(token,'uint3(lfsHipGroups,1,1)')
    source='[SpecializationConstant] const uint lfsHipGroups = 1;\n'+source
    shader=a.output.parent/(name+'.hip.slang');generated=a.output.parent/(name+'.hip.cu')
    shader.write_text(source,encoding='utf-8')
    subprocess.run([a.slang,str(shader),'-I',str(a.source),'-entry','main','-stage','compute','-target','cuda','-fp-mode','precise','-line-directive-mode','none','-o',str(generated)],check=True)
    lines=generated.read_text(encoding='utf-8').splitlines()
    if not lines or 'slang-cuda-prelude.h' not in lines[0]:raise RuntimeError('Review export prelude')
    text='\n'.join(lines[1:])
    text=text.replace('(blockIdx * blockDim + threadIdx).x','(blockIdx.x*blockDim.x+threadIdx.x)')
    # Slang emits its precise qualifier verbatim for this target. HIP uses
    # the per-function fp-contract pragma below to preserve these boundaries.
    text=re.sub(r'\bprecise\s+(?=float\b)', '', text)
    text=text.replace('extern "C" __constant__ GlobalParams_0 SLANG_globalParams;','').replace('#define globalParams_0 (&SLANG_globalParams)','')
    text=text.replace('globalParams_0->lfsHipGroups_0','gridDim.x').replace('(globalParams_0->kPhase_0)','phase')
    text,n=re.subn(r'extern "C" __global__ void main_0\(Parameters_0 (\w+)\)',r'__global__ void export_kernel(Parameters_0 \1,uint phase)',text)
    if n!=1 or 'globalParams_0' in text:raise RuntimeError('Review export globals/entry point: '+name)
    unknown=set(re.findall(r'\b(?:F32|I32|U32|I64|U64)_[A-Za-z0-9_]+',text))-{'F32_asuint','F32_fma','F32_min','F32_max','F32_abs','F32_exp','F32_log','F32_sqrt','F32_pow','F32_round','F32_isfinite','F32_isinf','F32_isnan','U32_asfloat','U32_min','I32_max','I32_min','F32_ceil','F32_asin','F32_atan2'}
    if unknown:raise RuntimeError('Missing export intrinsic: '+str(unknown))
    text=re.sub(r'((?:__device__|__global__)[^;{}]*\{)',r'\1\n#pragma clang fp contract(off)\n',text)
    result+='\nnamespace hip_'+name+' {\n'+'using namespace hip_export_common;\n'+text+'\n}\n'
a.output.write_text(result,encoding='utf-8')
