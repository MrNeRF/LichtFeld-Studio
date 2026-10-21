# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract native training-image device operations from the reference sources."""
import argparse
import re
from pathlib import Path
from prepare_application_mcmc import function

def flat(text): return re.sub(r'(?m)^        (?=__(?:device|global)__)', '    ', text)
def prepare(image,roi,heatmap,background,output):
    text=image.read_text(encoding='utf-8')
    constants=text[text.index('    __constant__ float CANNY_'):text.index('} // namespace lfs::filters')]
    code=constants+'\ntemplate <typename T>\n'+function(text,'canny_input_value')+'\ntemplate <typename InputT>\n'+function(text,'fused_canny_edge_filter_chw_kernel')
    code+='\n'+function(text,'normalize_by_device_scalar_kernel')
    quant=function(flat(text),'quantize_to_grid_kernel')
    if quant.count('__fadd_rn(')!=1 or quant.count('__fmul_rn(')!=1: raise ValueError('Image quantization rounding source seam changed')
    quant=quant.replace('__fadd_rn(','image_add_rn(').replace('__fmul_rn(','image_mul_rn(')
    code+='\n__device__ inline float image_add_rn(float a,float b) { volatile float value=a+b; return value; }\n__device__ inline float image_mul_rn(float a,float b) { volatile float value=a*b; return value; }\n'+quant
    bg=background.read_text(encoding='utf-8')
    code+='\n'+'\n'.join(function(bg,n) for n in ('bilinear_resize_chw_kernel','pcg_hash','uint_to_float','random_background_kernel'))
    rt=flat(roi.read_text(encoding='utf-8'))
    for name in ('kParallelEpsilon','kMinSurfaceAlpha'):
        declarations=re.findall(r'constexpr float '+name+r'\s*=\s*[^;]+;',rt)
        if len(declarations)!=1: raise ValueError('ROI constant source seam changed: '+name)
        code+='\n'+declarations[0]+'\n'
    code+='\nstruct Affine3x4 { float4 rows[3]; };\n'
    code+='\n'.join(function(rt,n) for n in ('transform_point','transform_direction','ray_intersects_aabb','roi_weight_map_kernel'))
    code+='\n'+function(flat(heatmap.read_text(encoding='utf-8')),'update_camera_loss_heatmap_kernel')
    code=code.replace('lfs::filters::','')
    header='''/* SPDX-FileCopyrightText: 2025-2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cfloat>
#include <type_traits>
namespace lfs_hip_application_image_kernels {
'''+code+'\n}\n'
    output.parent.mkdir(parents=True,exist_ok=True)
    if not output.exists() or output.read_text(encoding='utf-8')!=header: output.write_text(header,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('image','roi','heatmap','background','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); prepare(a.image,a.roi,a.heatmap,a.background,a.output)
