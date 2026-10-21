# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Port repository MRNF device functions; keep host allocation in Tensor."""
import argparse,re
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();text=a.source.read_text(encoding='utf-8')
def function(name):
    matches=list(re.finditer(r'(?m)^\s*__(?:global|device)__[^\n]*\b'+re.escape(name)+r'\(',text))
    if len(matches)!=1:raise RuntimeError('Review changed MRNF function '+name)
    start=matches[0].start();opening=text.index('{',matches[0].end());depth=1
    for token in re.finditer(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|[{}]',text[opening+1:],re.S):
        if token[0]=='{':depth+=1
        elif token[0]=='}':depth-=1
        if depth==0:return text[start:opening+1+token.end()]
    raise RuntimeError('Unclosed MRNF function '+name)
names=['d_sigmoid','d_logit','replace_parent_weights_kernel','mrnf_noise_injection_kernel',
       'mrnf_decay_kernel','fold_densification_error_and_zero_kernel','gumbel_key_kernel','gumbel_key_for_indices_kernel']
code='\n'.join(('template<bool RenderedOnly>\n' if name=='mrnf_decay_kernel' else '')+function(name) for name in names)
code=code.replace('curand','hiprand')
header='''/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
namespace lfs_hip_mrnf_kernels {
'''+code+'\n}\n'
a.output.parent.mkdir(parents=True,exist_ok=True)
if not a.output.exists() or a.output.read_text(encoding='utf-8')!=header:a.output.write_text(header,encoding='utf-8')
