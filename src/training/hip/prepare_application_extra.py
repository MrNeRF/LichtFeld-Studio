# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Adapt current repository regularization device functions for HIP."""
import argparse
from pathlib import Path
from prepare_application_mcmc import function
p=argparse.ArgumentParser()
for name in ('regularization','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();r=a.regularization.read_text(encoding='utf-8')
body='\n'.join(function(r,name) for name in ('fused_scale_regularization_kernel','final_scale_reduce_kernel','fused_opacity_regularization_kernel','final_opacity_reduce_kernel'))
body=body.replace('lfs::core::warp_ops::','lfs_hip_application_photo_warp_ops::')
header='/* SPDX-FileCopyrightText: 2025-2026 LichtFeld Studio Authors\n * SPDX-License-Identifier: GPL-3.0-or-later */\n#pragma once\n#include "application_photo_generated.hpp"\nnamespace lfs_hip_application_extra_kernels {\n'+body+'\n}\n'
a.output.parent.mkdir(parents=True,exist_ok=True)
if not a.output.exists() or a.output.read_text(encoding='utf-8')!=header:a.output.write_text(header,encoding='utf-8')
