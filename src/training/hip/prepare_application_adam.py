# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prepare standalone application Adam kernels without CUDA host allocation."""
import argparse
import re
from pathlib import Path

def prepare(api,kernels,sh,output):
    text=api.read_text(encoding='utf-8')
    structs=text[text.index('namespace fast_lfs::optimizer {'):text.index('    // Pure CUDA interface')]
    body=kernels.read_text(encoding='utf-8')
    body=body.replace('__grid_constant__','')
    body=re.sub(r'#include "([^"]+)"',lambda m:'#include "'+Path(m[1]).name+'"',body)
    body=body.replace('#include "adam_api.h"','')
    shtext=sh.read_text(encoding='utf-8')
    start=shtext.index('        template <int ACTIVE_SH_BASES>')
    end=shtext.index('    } // namespace',start)
    shbody=shtext[start:end]
    header='''/* SPDX-FileCopyrightText: 2025-2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
'''+structs+'\n}\n'+body+'\n#include "kernel_utils.cuh"\nnamespace lfs_hip_application_adam_kernels {\n'+shbody+'\n}\n'
    for name in ('adam_step_joint_contiguous_batched_cu','joint_encode_zero_mark_cu','joint_encode_zero_rows_cu','joint_encode_zero_shN_cu'):
        if body.count('void '+name+'(')!=1: raise ValueError('Adam source seam changed: '+name)
    if shbody.count('void __launch_bounds__(256, 3) adam_step_shN_joint_from_grad_cu(')!=1:
        raise ValueError('Adam SH source seam changed')
    output.parent.mkdir(parents=True,exist_ok=True)
    if not output.exists() or output.read_text(encoding='utf-8')!=header: output.write_text(header,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('api','kernels','sh','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); prepare(a.api,a.kernels,a.sh,a.output)
