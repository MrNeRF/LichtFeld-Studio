# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract this repository's MCMC device functions for the HIP application seam.

Host Tensor/CUB/Thrust launchers are not copied. Scan and ownership stay in the
native Tensor runtime. CUDA sources are never modified.
"""
import argparse
import re
from pathlib import Path

NAMES = ("relocation_kernel", "raw_quat_to_rotmat", "apply_noise_to_mean", "inject_noise_kernel",
         "update_scaling_opacity_kernel", "copy_gaussian_params_kernel",
         "multinomial_sample_and_gather_kernel", "multinomial_sample_all_kernel",
         "max_error_and_zero_densification_kernel")

def function(text, name):
    matches=list(re.finditer(r"(?m)^    __(?:global|device)__[^\n]*\b"+re.escape(name)+r"\(",text))
    if len(matches)!=1: raise ValueError(f"MCMC source seam changed: {name}")
    start=matches[0].start()
    opening=text.index("{",matches[0].end())
    depth=1
    # Skip comments/strings while matching the function's complete body.
    tokens=re.finditer(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|[{}]',text[opening+1:],re.S)
    for token in tokens:
        if token.group()=="{": depth+=1
        elif token.group()=="}": depth-=1
        if depth==0: return text[start:opening+1+token.end()]
    raise ValueError(f"Unclosed MCMC function: {name}")

def prepare(source,output):
    text=source.read_text(encoding="utf-8")
    if text.count("constexpr int RELOCATION_N_MAX = 51;")!=1: raise ValueError("MCMC relocation coefficient bound changed")
    code="\n\n".join(function(text,n) for n in NAMES)
    code=code.replace("curand", "hiprand").replace("__frcp_rn(","exact_reciprocal(").replace("rsqrt(","rsqrtf(")
    header=text[:text.index('#include')]+"""#pragma once
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cstdint>
namespace lfs_hip_mcmc_kernels {
using vec3=glm::vec<3,float>; using vec4=glm::vec<4,float>; using mat3=glm::mat<3,3,float>;
constexpr int RELOCATION_N_MAX=51;
__constant__ float d_relocation_coefficients[51*51];
__device__ inline float exact_reciprocal(float value) { return 1.f/value; }
"""+code+"\n}\n"
    output.parent.mkdir(parents=True,exist_ok=True)
    if not output.exists() or output.read_text(encoding="utf-8")!=header: output.write_text(header,encoding="utf-8")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); prepare(a.source,a.output)
