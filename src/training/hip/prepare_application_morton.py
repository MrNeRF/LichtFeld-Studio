# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Adapt this checkout's Morton and packed-moment device kernels for HIP."""
import argparse,re
from pathlib import Path
from prepare_application_mcmc import function
p=argparse.ArgumentParser()
for name in ('source','codec','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();text=a.source.read_text(encoding='utf-8');text=re.sub(r'(?m)^        (?=__(?:global|device)__)', '    ',text)
body='\n'.join(function(text,n) for n in ('part1by2','encode_morton3','quantize_axis','reduce_us_block','sh_moment_cell'))
for name in ('joint_permute_contiguous_bounds_cu','joint_permute_contiguous_encode_cu','joint_permute_shN_bounds_cu','joint_permute_shN_encode_cu'):
    body+='\ntemplate <int BITS>\n'+function(text,name)
body=body.replace('lfs::training::joint_adam::','lfs_hip_application_morton_codec::')
codec=a.codec.read_text(encoding='utf-8').replace('<cuda_runtime.h>','<hip/hip_runtime.h>').replace('namespace lfs::training::joint_adam {','namespace lfs_hip_application_morton_codec {')
header='/* SPDX-FileCopyrightText: 2025-2026 LichtFeld Studio Authors\n * SPDX-License-Identifier: GPL-3.0-or-later */\n#pragma once\n#include <hip/hip_runtime.h>\n#include <core/sh_layout.hpp>\n#include "application_sh_generated.hpp"\n'+codec+'\nnamespace lfs_hip_application_morton_kernels {\nconstexpr int kThreads=256;constexpr float kInf=1e30f;\n'+body+'\n__global__ void encode_from_gpu_bounds(const float* means,const float* lo,const float* hi,int32_t* keys,size_t n) {\n    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;\n    if(i>=n)return;\n    uint32_t code[3];\n    for(int c=0;c<3;++c){float length=hi[c]-lo[c];float multiplier=length==0?0:1024.f/length;code[c]=quantize_axis(means[i*3+c],lo[c],multiplier);}\n    keys[i]=int32_t(encode_morton3(code[0],code[1],code[2]));\n}\n}\n'
a.output.parent.mkdir(parents=True,exist_ok=True)
if not a.output.exists() or a.output.read_text(encoding='utf-8')!=header:a.output.write_text(header,encoding='utf-8')
