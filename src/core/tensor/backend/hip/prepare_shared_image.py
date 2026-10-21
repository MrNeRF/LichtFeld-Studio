# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Adapt the repository image-format device kernels; keep codec hosts outside HIP."""
import argparse
import re
from pathlib import Path

def function(text,name):
    match=re.search(r"(?m)^\s*__(?:global|device)__[^\n]*\b"+name+r"\(",text)
    if not match: raise ValueError("Missing image source seam: "+name)
    start=match.start(); opening=text.index('{',match.end()); depth=1; end=opening+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mask-source',type=Path,required=True);a=p.parse_args()
s=a.source.read_text(encoding='utf-8')
names=['sentinel_byte','fill_u8_sentinel_kernel','flag_u8_sentinel_unchanged_kernel','uint8_hwc_to_float32_chw_kernel','uint16_hwc_to_float32_chw_kernel','float32_hwc_to_uint16_hwc_kernel','uint16_hwc_to_float_hwc_kernel','normal_chw_to_jpeg2k_hwc_kernel','jpeg2k_hwc_to_normal_chw_kernel','srgb_encoding_to_linear_device','normal_prior_hwc_to_float32_chw_kernel','uint8_hwc_to_uint8_chw_kernel','uint16_hwc_to_uint8_chw_kernel','float_to_u8','float32_chw_to_uint8_chw_kernel','uint8_hw_to_float32_hw_kernel','uint8_rgba_split_kernel','uint8_rgba_split_uint8_rgb_kernel','mask_invert_kernel','mask_threshold_kernel']
header='/* SPDX-FileCopyrightText: 2025-2026 LichtFeld Studio Authors\n * SPDX-License-Identifier: GPL-3.0-or-later */\n#pragma once\n#include <hip/hip_runtime.h>\n#include <cstdint>\nnamespace hip_shared_image {\nstruct NormalPriorTransform { bool srgb,flip_yz,world_to_camera; float w2c[9]; };\nconstexpr float NORMALIZE_SCALE_U8=1.f/255.f,NORMALIZE_SCALE_U16=1.f/65535.f;\n__device__ inline float normalize_mask_byte(uint8_t v) { return float(v)/255.f; }\n__device__ inline float normalize_mask_u16_value(uint16_t v) { return float(v)/65535.f; }\n__device__ inline bool mask_value_at_least(float v,float threshold) { return v>=threshold; }\n'
for name in names:
    prefix='template <bool ExactAlpha=false>\n' if name in ('uint8_hwc_to_float32_chw_kernel','uint16_hwc_to_float32_chw_kernel','uint16_hwc_to_float_hwc_kernel') else 'template <typename T>\n' if name=='normal_prior_hwc_to_float32_chw_kernel' else ''
    header+=prefix+function(s,name).replace('lfs::core::cuda::','')+'\n'
mask=a.mask_source.read_text(encoding='utf-8')
header+='__device__ inline float mask_mul_rn(float a,float b) { volatile float result=a*b; return result; }\n'
header+=function(mask,'mask_grid_byte').replace('__fmul_rn(','mask_mul_rn(')+'\n'
header+='template <bool KeepBytes>\n'+function(mask,'quantize_mask_grid_kernel')+'\n'
header+='}\n'
a.output.parent.mkdir(parents=True,exist_ok=True)
if not a.output.exists() or a.output.read_text(encoding='utf-8')!=header:a.output.write_text(header,encoding='utf-8')
