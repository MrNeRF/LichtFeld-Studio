# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract complete application photometric device kernels from this checkout."""
import argparse
import re
from pathlib import Path
from prepare_application_mcmc import function

def prepare(source,reduction,l1,warp,output):
    text=source.read_text(encoding='utf-8')
    body=text[text.index('namespace {\n')+len('namespace {\n'):text.index('    template <typename Fn>\n    void dispatch_target_ptr')]
    for name in ('fusedssimCUDA','fusedssim_backwardCUDA','fusedL1SSIMForwardCUDA','fusedL1SSIMBackwardCUDA',
                 'maskedFusedL1SSIMForwardCUDA','maskedDecoupledFusedL1SSIMForwardCUDA',
                 'maskedFusedL1SSIMBackwardCUDA','decoupledFusedL1SSIMForwardCUDA','maskedSSIMMapCUDA'):
        if body.count('void '+name+'(')!=1: raise ValueError('Photometric source seam changed: '+name)
    error=function(text.replace('        __global__','    __global__'),'ssim_to_error_map_kernel')
    red=reduction.read_text(encoding='utf-8')
    red=red[red.index('        constexpr int REDUCTION_BLOCK_SIZE'):red.index('    } // namespace')]
    l1text=l1.read_text(encoding='utf-8')
    l1body=l1text[l1text.index('    template <typename TargetT>\n    __global__ void fused_l1_kernel'):l1text.index('    template <typename TargetT>\n    void launch_fused_l1_loss_impl')]
    warptext=warp.read_text(encoding='utf-8')
    if '__shfl_xor_sync(0xffffffff, val, offset)' not in warptext: raise ValueError('Warp reduction source seam changed')
    # Explicit 32-lane subgroups preserve the reference reduction order on
    # both wave32 and wave64 HIP devices.
    flattened=re.sub(r'(?m)^        (?=__device__)', '    ', warptext)
    warptext='namespace lfs_hip_application_photo_warp_ops {\n'+ '\n'.join('template <typename T>\n'+function(flattened,n) for n in ('warp_reduce_sum','block_reduce_sum'))+'\n}\n'
    warptext=warptext.replace('__shfl_xor_sync(0xffffffff, val, offset)','__shfl_xor(val, offset, 32)')
    red=red.replace('lfs::core::warp_ops::','lfs_hip_application_photo_warp_ops::')
    l1body=l1body.replace('lfs::core::warp_ops::','lfs_hip_application_photo_warp_ops::')
    header='''/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdint>
#include <type_traits>
'''+warptext+'''
namespace lfs_hip_application_photo_kernels {
namespace cg=cooperative_groups;
'''+body+error+'''
namespace reduction {
constexpr float SSIM_EPSILON=1e-8f;
'''+red+l1body+'\n}\n}\n'
    for macro in ('BLOCK_X','BLOCK_Y','HALO','SHARED_X','SHARED_Y','CONV_X','CONV_Y'):
        header+='\n#undef '+macro
    output.parent.mkdir(parents=True,exist_ok=True)
    if not output.exists() or output.read_text(encoding='utf-8')!=header: output.write_text(header,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','reduction','l1','warp','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); prepare(a.source,a.reduction,a.l1,a.warp,a.output)
