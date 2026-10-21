# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prepare the repository's SH layout/value device kernels for native HIP."""
import argparse
from pathlib import Path

def prepare(layout,quant,codec,output):
    layout_text=layout.read_text(encoding='utf-8')
    layout_body=layout_text[layout_text.index('        // Float4 slot index'):layout_text.index('    } // namespace')]
    quant_text=quant.read_text(encoding='utf-8')
    quant_body=quant_text[quant_text.index('        constexpr int kThreads'):quant_text.index('        void exclusive_sum_i32(')]
    replacements={
        'asm("sub.rn.f32 %0, %1, %2;" : "=f"(range) : "f"(hi), "f"(lo));':'volatile float rounded_range = hi - lo; range = rounded_range;',
        'asm("mul.rn.f32 %0, %1, %2;" : "=f"(normalized) : "f"(code), "f"(inv_q));':'volatile float rounded_code = code * inv_q; normalized = rounded_code;',
        'asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(value) : "f"(range), "f"(normalized), "f"(lo));':'value = fmaf(range, normalized, lo);',
    }
    for old,new in replacements.items():
        if quant_body.count(old)!=1: raise ValueError('SH rounded serialization source seam changed')
        quant_body=quant_body.replace(old,new)
    if 'asm(' in quant_body or 'cuda' in quant_body: raise ValueError('Unexpected device dependency in SH quant source')
    codec_body=codec.read_text(encoding='utf-8').replace('<cuda_runtime.h>','<hip/hip_runtime.h>').replace('namespace lfs::core::sh_value {','namespace lfs_hip_application_sh_value {')
    quant_body=quant_body.replace('lfs::core::sh_value::','lfs_hip_application_sh_value::')
    header='''/* SPDX-FileCopyrightText: 2025-2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <core/sh_layout.hpp>
#include <hipcub/hipcub.hpp>
'''+codec_body+'''
namespace lfs_hip_application_sh_layout {
using namespace lfs::core;
'''+layout_body+'''
}
namespace lfs_hip_application_sh_quant {
'''+quant_body+'\n}\n'
    output.parent.mkdir(parents=True,exist_ok=True)
    if not output.exists() or output.read_text(encoding='utf-8')!=header: output.write_text(header,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('layout','quant','codec','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); prepare(a.layout,a.quant,a.codec,a.output)
