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
template<class T,int N> struct FixedArray {
    T data[N];
    __device__ T& operator[](int i){return data[i];}
    __device__ const T& operator[](int i)const{return data[i];}
};
__device__ inline uint F32_asuint(float x){return __float_as_uint(x);}
__device__ inline float F32_fma(float x,float y,float z){return fmaf(x,y,z);}
"""
result='/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors\n * SPDX-License-Identifier: GPL-3.0-or-later */\n#pragma once\n'
for name in ('export_kmeans','export_kmeans_accumulate','export_kmeans_group','export_radix'):
    source=(a.source/(name+'.slang')).read_text(encoding='utf-8')
    token='spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); }'
    if source.count(token)!=1:raise RuntimeError('Review export grid lowering: '+name)
    source=source.replace(token,'uint3(lfsHipGroups,1,1)')
    source='[SpecializationConstant] const uint lfsHipGroups = 1;\n'+source
    shader=a.output.parent/(name+'.hip.slang');generated=a.output.parent/(name+'.hip.cu')
    shader.write_text(source,encoding='utf-8')
    subprocess.run([a.slang,str(shader),'-I',str(a.source),'-entry','main','-stage','compute','-target','cuda','-fp-mode','precise','-line-directive-mode','none','-o',str(generated)],check=True)
    lines=generated.read_text(encoding='utf-8').splitlines()
    if not lines or 'slang-cuda-prelude.h' not in lines[0]:raise RuntimeError('Review export prelude')
    text='\n'.join(lines[1:])
    text=text.replace('extern "C" __constant__ GlobalParams_0 SLANG_globalParams;','').replace('#define globalParams_0 (&SLANG_globalParams)','')
    text=text.replace('globalParams_0->lfsHipGroups_0','gridDim.x').replace('(globalParams_0->kPhase_0)','phase')
    text,n=re.subn(r'extern "C" __global__ void main_0\(Parameters_0 (\w+)\)',r'__global__ void export_kernel(Parameters_0 \1,uint phase)',text)
    if n!=1 or 'globalParams_0' in text:raise RuntimeError('Review export globals/entry point: '+name)
    unknown=set(re.findall(r'\b(?:F32|I32|U32|I64|U64)_[A-Za-z0-9_]+',text))-{'F32_asuint','F32_fma'}
    if unknown:raise RuntimeError('Missing export intrinsic: '+str(unknown))
    text=re.sub(r'((?:__device__|__global__)[^;{}]*\{)',r'\1\n#pragma clang fp contract(off)\n',text)
    result+='\nnamespace hip_'+name+' {\n'+prelude+text+'\n}\n'
a.output.write_text(result,encoding='utf-8')
