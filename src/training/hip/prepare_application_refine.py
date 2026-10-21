# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract the repository's refinement device code, keeping host ownership native."""
import argparse
import re
from pathlib import Path
from prepare_application_mcmc import function

def prepare(source, pruning, screen, output):
    text=source.read_text(encoding='utf-8')
    # The lexical extractor expects one indentation level. Nested functions are
    # extracted independently into the HIP-specific namespace.
    flattened=re.sub(r'(?m)^        (?=__(?:global|device)__)', '    ', text)
    names=('sigmoid','inverse_sigmoid','quat_to_rotmat','get_max_value_index',
           'long_axis_split_gaussians_inplace_kernel','fill_free_slots_fused_kernel',
           'packed_refine_counts_kernel','zero_nan_kernel',
           'div_by_positive_median_or_zero_kernel','clip_log_scale_by_screen_share_kernel',
           'oversize_split_scores_kernel')
    radix_start=text.index('    namespace radix_select {')
    radix_end=text.index('        template <bool kPositiveOnly>\n        void run(',radix_start)
    radix=text[radix_start:radix_end]+'    }\n'
    chunks=[function(flattened,n) for n in names]
    # Define the exact radix state before the normalization kernel uses it.
    chunks.insert(8,radix)
    prune=pruning.read_text(encoding='utf-8')
    chunks.extend(function(prune,n) for n in ('compute_dead_mask_kernel','compute_near_zero_rotation_mask_kernel'))
    screen_text=screen.read_text(encoding='utf-8')
    if screen_text.count('inline float\n        oversize_split_score(')!=1:
        raise ValueError('Screen-share source seam changed')
    score=screen_text[screen_text.index('        oversize_split_score('):screen_text.index('\n} // namespace lfs::training')]
    score=score[:score.index('#else')].replace('#ifdef __CUDA_ARCH__','')+'    }\n'
    score='__device__ inline float\n'+score
    body='\n\n'.join(chunks).replace('cub::','hipcub::')
    body=body.replace('__shfl_down_sync(0xffffffffu, counted, offset)', '__shfl_down(counted, offset, 32)')
    body=body.replace('lfs::training::oversize_split_score','oversize_split_score')
    header='''/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstdint>
namespace lfs_hip_refine_kernels {
'''+score+body+'\n}\n'
    output.parent.mkdir(parents=True,exist_ok=True)
    if not output.exists() or output.read_text(encoding='utf-8')!=header:
        output.write_text(header,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','pruning','screen','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); prepare(a.source,a.pruning,a.screen,a.output)
