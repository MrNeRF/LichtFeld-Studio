# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Lower shared expression source emission to HIP without changing Metal code."""
import argparse
import hashlib
import re
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
s=a.source.read_text(encoding='utf-8')
s=s[s.index('#include "../../internal/expression_emitter.hpp"'):]
s=s.replace('#include "../../internal/expression_emitter.hpp"', '#include "'+(a.source.parent/'../../internal/expression_emitter.hpp').resolve().as_posix()+'"')
s=s.replace('MslEmitter','HipExpressionEmitter').replace('expression_msl(', 'expression_hip(').replace('MSL expressions','HIP expressions').replace('MSL expression','HIP expression').replace('MSL lowering','HIP lowering')
old='''#include <metal_stdlib>
#pragma METAL fp contract(off)
using namespace metal;'''
new='''#pragma clang fp contract(off)
using uint=unsigned int;using ulong=unsigned long long;using uchar=unsigned char;using ushort=unsigned short;
template<class T,class U> __device__ T as_type(U value) { static_assert(sizeof(T)==sizeof(U));return __builtin_bit_cast(T,value); }
template<class T> __device__ T max(T a,T b){return a>b?a:b;}
template<class T> __device__ T min(T a,T b){return a<b?a:b;}
__device__ uint clz(uint v){return __clz(v);}
__device__ uint mulhi(uint a,uint b){return __umulhi(a,b);}
'''
if s.count(old)!=1:raise RuntimeError('Review shared expression prelude')
s=s.replace(old,new).replace('static float ', 'static __device__ float ').replace('static uint ', 'static __device__ uint ').replace('thread int&','int&')
s=s.replace('"device const uint"','"const uint"').replace('"device const uchar"','"const uchar"').replace('"device const ushort"','"const ushort"').replace('"device uint"','"uint"')
old='''source += "kernel void lfs_expression(constant uint* lfs_args [[buffer(0)]],\\n"
                          "                           uint lfs_thread [[thread_position_in_grid]],\\n"
                          "                           uint lfs_threads [[threads_per_grid]]) {\\n";'''
new='''source += "struct LfsExpressionArgs { uint words[256]; };\\n"
                          "extern \\\"C\\\" __global__ void lfs_expression(LfsExpressionArgs arguments) {\\n"
                          "const uint* lfs_args=arguments.words;\\n"
                          "uint lfs_thread=blockIdx.x*blockDim.x+threadIdx.x;\\n"
                          "uint lfs_threads=gridDim.x*blockDim.x;\\n";'''
if s.count(old)!=1:raise RuntimeError('Review shared expression entry point')
s=s.replace(old,new)
s=s.replace('ulong','lfs_ulong')
s=re.sub(r'\bmax\(', 'lfs_max(', s)
s=re.sub(r'\bmin\(', 'lfs_min(', s)
# Use float overloads in HIPRTC, with IEEE behavior and no implicit double math.
s=s.replace('bits("abs("','bits("fabsf("').replace('bits("rsqrt("','bits("rsqrtf("')
digest=hashlib.sha256(s.encode('utf-8')).hexdigest()
text='/* Generated HIP expression emitter; do not edit. */\n#pragma once\nnamespace lfs::core::internal { inline constexpr char hip_expression_emitter_id[]=\"'+digest+'\"; }\n'+s
if not a.output.exists() or a.output.read_text(encoding='utf-8')!=text:
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(text,encoding='utf-8')
