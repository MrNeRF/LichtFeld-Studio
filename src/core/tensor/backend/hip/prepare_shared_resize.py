# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Adapt the existing Lanczos/prior device implementation, without CUDA hosts."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
s=a.source.read_text(encoding='utf-8')
prior=s[s.index('    template <int Channels>'):s.index('    struct CoefficientLayout')]
pixels=s[s.index('    template <typename T>\n    struct PixelTraits'):s.index('} // namespace\n\nnamespace lfs::core')]
body=s[s.index('        __device__ float sinc('):s.index('        template <typename T, bool ExactAlpha = false>\n        static Tensor lanczos_resize_impl(')]
body=body.replace('cuda::normalize_mask_byte','hip_shared_image::normalize_mask_byte')
for name in ('div','mul','add','sub'):
    body=body.replace('__f'+name+'_rn(', 'resize_'+name+'_rn(')
header='/* SPDX-FileCopyrightText: 2025-2026 LichtFeld Studio Authors\n * SPDX-FileCopyrightText: 2025 Youyu Chen (original Lanczos implementation)\n * SPDX-License-Identifier: GPL-3.0-or-later\n * SPDX-License-Identifier: MIT (original Lanczos implementation) */\n#pragma once\n#include <hip/hip_cooperative_groups.h>\n#include <cassert>\nnamespace hip_shared_resize {\nnamespace cg=cooperative_groups;\nconstexpr int BLOCK_X=16,BLOCK_Y=16;\n__device__ inline float resize_div_rn(float a,float b) { volatile float v=a/b;return v; }\n__device__ inline float resize_mul_rn(float a,float b) { volatile float v=a*b;return v; }\n__device__ inline float resize_add_rn(float a,float b) { volatile float v=a+b;return v; }\n__device__ inline float resize_sub_rn(float a,float b) { volatile float v=a-b;return v; }\n#define LFS_DEBUG_ASSERT_MSG(condition,...) assert(condition)\n'+prior+pixels+body+'\n#undef LFS_DEBUG_ASSERT_MSG\n}\n'
a.output.parent.mkdir(parents=True,exist_ok=True)
if not a.output.exists() or a.output.read_text(encoding='utf-8')!=header:a.output.write_text(header,encoding='utf-8')
