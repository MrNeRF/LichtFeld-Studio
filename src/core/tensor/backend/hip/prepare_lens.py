# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Adapt existing inverse lens and RGB sampling into private HIP; never edit CUDA."""
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = args.source.read_text(encoding="utf-8")
start = source.index("        struct RadialBranch {")
end = source.index("        __global__", start)
functions = source[start:end]
name = source.index("            inverse_distortion_sample_map_kernel(")
start = source.rfind("        __global__", 0, name)
body = source.index("{", name)
depth = 1
end = body + 1
while depth:
    depth += (source[end] == "{") - (source[end] == "}")
    end += 1
kernel = source[start:end].replace("__grid_constant__ ", "")
def extract_inverse(name):
    marker = source.index('            '+name+'(')
    first = source.rfind('        __global__', 0, marker)
    body = source.index('{', marker)
    depth, last = 1, body + 1
    while depth:
        depth += (source[last] == '{') - (source[last] == '}')
        last += 1
    code = source[first:last].replace('__grid_constant__ ', '')
    old = 'if (ox >= params.src_width || oy >= params.src_height)'
    if code.count('const int output_y_offset,') != 1 or code.count(old) != 1:
        raise RuntimeError('Review changed inverse image warp row bounds')
    code = code.replace('const int output_y_offset,', 'const int output_y_offset,\n                const int output_y_end,')
    code = code.replace(old, 'if (ox >= params.src_width || oy >= params.src_height || oy >= output_y_end)')
    if name == 'distort_image_to_source_kernel':
        old = 'if (!isfinite(sx) || !isfinite(sy))'
        if code.count(old) != 1:
            raise RuntimeError('Review changed inverse RGB footprint validation')
        code = code.replace(old, 'if (!isfinite(sx) || !isfinite(sy) || fabsf(sx)>1073741824.f || fabsf(sy)>1073741824.f)')
    else:
        old = '                    if (mode == AreaFilterMode::DEPTH)'
        if code.count(old) != 1:
            raise RuntimeError('Review changed inverse prior sampling')
        code = code.replace(old, '                    if (!isfinite(sx) || !isfinite(sy) || fabsf(sx)>1073741824.f || fabsf(sy)>1073741824.f) {converged=false;break;}\n'+old)
    return code
inverse_rgb = extract_inverse('distort_image_to_source_kernel')
inverse_area = extract_inverse('distort_area_to_source_kernel')
samples_start = source.index("        template <typename T>\n        __device__ float normalized_sample")
samples_end = source.index("        struct RadialBranch {")
sampling = source[samples_start:samples_end]
old_offset = "const int output_y_offset,"
old_limit = "if (ox >= params.dst_width || oy >= params.dst_height)"
if sampling.count(old_offset) != 1 or sampling.count(old_limit) != 1:
    raise RuntimeError("Forward lens sampling source changed; review HIP tiling")
sampling = sampling.replace(old_offset, old_offset + "\n                const int output_y_end,")
sampling = sampling.replace(old_limit, "if (ox >= params.dst_width || oy >= params.dst_height || oy >= output_y_end)")
for axis in ("x", "y"):
    old = f"dn{axis} * params.src_f{axis} + params.src_c{axis} - PIXEL_CENTER_OFFSET"
    if sampling.count(old) != 1:
        raise RuntimeError("Forward lens projection changed; review HIP rounding")
    sampling = sampling.replace(old, f"fmaf(dn{axis}, params.src_f{axis}, params.src_c{axis}) - PIXEL_CENTER_OFFSET")
# Avoid float-to-int overflow for pathological projected coordinates. Such
# samples have no representable footprint; ordinary sampling is unchanged.
old_sample = "                    (void)lanczos3_sample("
if sampling.count(old_sample) != 1:
    raise RuntimeError("Forward lens sample call changed; review HIP bounds")
sampling = sampling.replace(old_sample, """                    if (!isfinite(sx) || !isfinite(sy) || fabsf(sx) > 1073741824.f || fabsf(sy) > 1073741824.f)
                        continue;
""" + old_sample)
# Area-filtered masks and priors share the same projection and bounded row tiling.
area_start=source.index("        enum class AreaFilterMode : int {")
area_end=source.index("        __global__",source.index("            undistort_area_kernel("))
# The next global declaration is the inverse area warp; retain only the forward kernel.
area=source[area_start:area_end]
if area.count(old_offset)!=1 or area.count(old_limit)!=1:
    raise RuntimeError("Area lens source seam changed; review HIP tiling")
area=area.replace(old_offset,old_offset+"\n                const int output_y_end,")
area=area.replace(old_limit,"if (ox >= params.dst_width || oy >= params.dst_height || oy >= output_y_end)")
for axis in ("x","y"):
    old=f"dn{axis} * params.src_f{axis} + params.src_c{axis} - PIXEL_CENTER_OFFSET"
    if area.count(old)!=1: raise RuntimeError("Area lens projection seam changed")
    area=area.replace(old,f"fmaf(dn{axis}, params.src_f{axis}, params.src_c{axis}) - PIXEL_CENTER_OFFSET")
area=area.replace("                    if (mode == AreaFilterMode::DEPTH)","                    if (!isfinite(sx) || !isfinite(sy) || fabsf(sx)>1073741824.f || fabsf(sy)>1073741824.f) continue;\n                    if (mode == AreaFilterMode::DEPTH)")
constants = source[source.index("        constexpr int BLOCK_DIM"):source.index("        using detail::apply_distortion;")]
# Keep RGB channels compile-time constant in the native path. Dynamic channel
# indexing otherwise prevents scalarization of the four-element accumulators.
# The default instantiation retains the general 1/2/4-channel implementation.
rgb_kernel = "template <typename T>\n        __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)\n            undistort_image_kernel("
if sampling.count(rgb_kernel) != 1:
    raise RuntimeError("Review changed forward RGB kernel template")
sampling = sampling.replace(rgb_kernel, rgb_kernel.replace("typename T>", "typename T, int Channels = 0>"))
rgb_start = sampling.index("            undistort_image_kernel(")
rgb_body = sampling.index("{", rgb_start)
signature = sampling[rgb_start:rgb_body]
if signature.count("const int channels,") != 1:
    raise RuntimeError("Review changed forward RGB channel parameter")
sampling = sampling[:rgb_start] + signature.replace("const int channels,", "const int runtime_channels,") + sampling[rgb_body:]
rgb_body = sampling.index("{", rgb_start)
sampling = sampling[:rgb_body + 1] + "\n            const int channels = Channels ? Channels : runtime_channels;" + sampling[rgb_body + 1:]
text = """// Generated from the repository's CUDA lens math. Do not edit.
#include <hip/hip_runtime.h>
#include "runtime_api.h"
#include "../../../cuda/undistort/distortion_model.cuh"
namespace hip_lens {
using lfs::core::CameraModelType;
using lfs::core::detail::apply_distortion;
using lfs::core::detail::thin_prism_fisheye_from_theta_point;
using UndistortParams = LfsHipLens;
""" + constants + functions + kernel + sampling + area + inverse_rgb + inverse_area + "\n} // namespace hip_lens\n"
text = text.replace("params.model_type", "static_cast<CameraModelType>(params.model_type)")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(text, encoding="utf-8")
