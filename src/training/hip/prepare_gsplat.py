# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate the private HIP raster kernels from this checkout's gsplat sources.

Never edits the application CUDA sources. Adapts API spellings, host diagnostics,
wave masks and reductions; retains device rasterization formulas. Instantiates
one, three, four and eight channels for RGB, depth and normal render modes.
Generated kernels do not register an application backend.
"""
import argparse
from pathlib import Path
import re


def write_if_changed(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    # Preserve timestamps of generated units unchanged by an adaptation.
    if not path.exists() or path.read_text(encoding=encoding) != text:
        path.write_text(text, encoding=encoding)


def prepare(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    names = ["Cameras.h", "Cameras.cuh", "FromWorldRay.cuh", "Rasterization.h",
             "TileBatch.h", "Utils.cuh", "RasterizeToPixelsFromWorld3DGSFwd.cu",
             "RasterizeToPixelsFromWorld3DGSBwd.cu", "Projection.h", "Intersect.h",
             "ProjectionUT3DGSFused.cu", "IntersectTile.cu",
             "SphericalHarmonics.h", "SphericalHarmonicsCUDA.cu", "GeometryFeatures.h", "GeometryFeatures.cu"]
    common = (source / "Common.h").read_text(encoding="utf-8")
    enums = common[common.index("#ifndef _GSPLAT_CAMERA_MODEL_TYPE_DEFINED"):
                   common.index("namespace gsplat_lfs {")]
    types = common[common.index("    using vec2 ="):]
    write_if_changed(output / "Common.h",
        common[:common.index("#pragma once")] + "#pragma once\n"
        '#include <hip/hip_runtime.h>\n#include <glm/gtc/type_ptr.hpp>\n'
        + enums + "namespace gsplat_lfs {\n" + types, encoding="utf-8")
    names += ["PopScores.h", "PopScores.cu"]
    for name in names:
        text = (source / name).read_text(encoding="utf-8")
        if name == "PopScores.cu":
            text = text.replace('#include "../../kernels/kernel_stream.hpp"', '#include "hip_check.hpp"')
            text = text.replace('        stream = lfs::resolve_stream(stream);', '')
        text = text.replace('#include "core/cuda_error.hpp"', '#include "hip_check.hpp"')
        text = text.replace('#include "core/cuda_safe_format.hpp"', '#include "hip_check.hpp"')
        text = text.replace("#include <cuda_runtime.h>", "#include <hip/hip_runtime.h>")
        text = text.replace("#include <cuda.h>", "#include <hip/hip_runtime.h>")
        text = text.replace("#include <cooperative_groups.h>", "#include <hip/hip_cooperative_groups.h>")
        text = text.replace("#include <cooperative_groups/reduce.h>", "#include <hip/hip_cooperative_groups.h>")
        text = re.sub(r"\bcuda(\w+)", r"hip\1", text)
        # This experiment is explicitly wave32-only. HIP sync intrinsics
        # require a 64-bit mask even when the active hardware wave has 32 lanes.
        text = text.replace("0xffffffffu", "0x00000000ffffffffull")
        text = text.replace("cg::reduce(", "lfs_hip_reduce(")
        text = re.sub(r"^#pragma nv_diag_.*$", "", text, flags=re.MULTILINE)
        if name == "RasterizeToPixelsFromWorld3DGSBwd.cu":
            # Halve LDS for the wave32 dual-pixel block. Preserve reverse
            # intersection order and adapt staging with the batch capacity.
            old_batch = "constexpr uint32_t kBwdDualBatch = 256;"
            old_stage = """            stage_splat(batch_end - static_cast<int32_t>(tr), tr);
            stage_splat(batch_end - static_cast<int32_t>(tr + kBwdDualThreads), tr + kBwdDualThreads);"""
            old_threads = "constexpr uint32_t kBwdDualThreads = 128;"
            if text.count(old_batch) != 1 or text.count(old_stage) != 1 or text.count(old_threads) != 1:
                raise RuntimeError("Dual-pixel batch source changed; review HIP staging")
            text = text.replace(old_batch, "constexpr uint32_t kBwdDualBatch = 128;")
            text = text.replace(old_stage, """            for (uint32_t slot = tr; slot < kBwdDualBatch; slot += block.size())
                stage_splat(batch_end - static_cast<int32_t>(slot), slot);""")
        if name == "Utils.cuh":
            begin = text.index("    template <typename Kernel>\n    inline void set_kernel_max_dynamic_smem")
            end = text.index("\n} // namespace gsplat_lfs", begin)
            text = text[:begin] + '''    template <typename Kernel>
    inline void set_kernel_max_dynamic_smem(Kernel kernel, int smem, const char* what) {
        static std::mutex mutex;
        static std::map<std::pair<int, uintptr_t>, int> ceilings;
        int device = 0;
        hip_check(hipGetDevice(&device), "shared memory device");
        std::lock_guard lock(mutex);
        auto& ceiling = ceilings[{device, reinterpret_cast<uintptr_t>(kernel)}];
        if (smem > ceiling) {
            hip_check(hipFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                hipFuncAttributeMaxDynamicSharedMemorySize, smem), what);
            ceiling = smem;
        }
    }
''' + text[end:]
        if name == "IntersectTile.cu":
            # The experiment supplies its own hipCUB workspace ownership.
            text = text[:text.index("    void radix_sort_double_buffer(")] + "} // namespace gsplat_lfs\n"
            text = text.replace("#include <cub/cub.cuh>", "")
            text = text.replace("#include <thrust/iterator/transform_iterator.h>", "")
            # Clamp in floating point before converting to unsigned: negative
            # image-edge coordinates must not wrap into enormous tile indices.
            for axis, extent in [("x", "width"), ("y", "height")]:
                for bound, operation, sign in [("min", "floor", "-"), ("max", "ceil", "+")]:
                    old = f"min(max(0, (uint32_t){operation}(tile_{axis} {sign} tile_radius_{axis})), tile_{extent})"
                    new = f"static_cast<uint32_t>(min(max(0.0f, {operation}(tile_{axis} {sign} tile_radius_{axis})), static_cast<float>(tile_{extent})))"
                    if old not in text:
                        raise RuntimeError("Intersection clamp source changed; review HIP adaptation")
                    text = text.replace(old, new)
        if name.endswith(".cu"):
            text = re.sub(r"^    __INS__\((?!(?:1|3|4|8)\))\d+\)\s*$", "", text, flags=re.MULTILINE)
        write_if_changed(output / name, text, encoding="utf-8")
    write_if_changed(output / "hip_check.hpp",'''#pragma once
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>
#include <map>
#include <mutex>
#include <hip/hip_cooperative_groups.h>
template <class Group, class T, class Op>
__device__ T lfs_hip_reduce(Group group, T value, Op op) {
    for (unsigned offset = group.size() / 2; offset; offset /= 2)
        value = op(value, group.shfl_xor(value, offset));
    return value;
}
struct LfsHipRasterFailure : std::runtime_error {
    hipError_t code;
    LfsHipRasterFailure(hipError_t value, const char* what)
        : std::runtime_error(std::string(what) + ": " + hipGetErrorString(value)), code(value) {}
};
inline void hip_check(hipError_t status, const char* what) {
    if (status != hipSuccess)
        throw LfsHipRasterFailure(status, what);
}
#define LFS_CUDA_CHECK_MSG(call, ...) hip_check((call), "rasterizer HIP call")
#define LFS_CUDA_LAUNCH_CHECK(stream, ...) hip_check(hipGetLastError(), "rasterizer HIP launch")
''', encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
