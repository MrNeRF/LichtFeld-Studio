# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prepare private FastGS rasterization and math sources for HIP wave32.

Original CUDA files are preserved. This isolated fixed-topology rasterizer
does not register the main application's Tensor backend or training strategies.
"""
import argparse
from pathlib import Path
import re
from fast_warp_staging import adapt as warp_staged_backward
from fast_warp_staging import adjacent_pixels as adjacent_pixel_backward
from fast_four_pixels import adapt as four_pixel_backward
from fast_forward_pixels import adapt as eight_pixel_forward
from fast_compact_sort import kernels as compact_sort_kernels, forward as compact_sort_forward


FILES = (
    "src/training/kernels/warp_reduce.cuh",
    "src/training/kernels/effective_rank.cuh",
    "src/training/kernels/joint_adam_codec.cuh",
    "src/training/include/lfs/training/mean_step_scale.cuh",
    "src/training/include/lfs/training/screen_share.cuh",
    "src/core/include/core/sh_value_codec.cuh",
    "src/training/include/lfs/training/ops/domain_types.hpp",
)

FAST_ROOT = "src/training/rasterization/fastgs/"
RASTER_FILES = tuple(FAST_ROOT + "rasterization/include/" + name for name in (
    "backward.h", "forward.h", "buffer_utils.h", "fused_adam_types.h",
    "kernel_utils.cuh", "kernels_backward.cuh", "kernels_forward.cuh",
    "rasterization_config.h", "visibility.cuh", "rasterization_api.h", "pop_scores.h")) + (
    FAST_ROOT + "utils/helper_math.h", FAST_ROOT + "utils/utils.h",
    FAST_ROOT + "rasterization/src/forward.cu",
    FAST_ROOT + "rasterization/src/backward.cu",
    FAST_ROOT + "rasterization/src/pop_scores.cu",
)

DIAGNOSTICS = r'''#pragma once
#include <hip/hip_runtime.h>
#include <type_traits>
#include <cassert>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <format>
#include <stdexcept>
#include <string>
namespace lfs::core {
inline bool hip_sync_debug_enabled() {
    return std::getenv("LFS_HIP_FAST_SYNC_DEBUG") != nullptr;
}
namespace detail {
template<class... T> std::string format_cuda_safe(const char* pattern, T&&... args) {
    return std::vformat(pattern, std::make_format_args(args...));
}
}
}
inline void lfs_fast_hip_check(hipError_t error, const char* name, const std::string& detail = {}) {
    if (error != hipSuccess)
        throw std::runtime_error(std::string(name) + ": " + hipGetErrorString(error) + " " + detail);
}
#define LFS_CUDA_CHECK_MSG(call, ...) lfs_fast_hip_check((call), #call)
#define LFS_ENSURE_CUDA_SUCCESS_MSG(error, name, detail) lfs_fast_hip_check((error), (name), (detail))
#define LFS_CUDA_LAUNCH_CHECK(stream, name) lfs_fast_hip_check(hipGetLastError(), (name))
#define LFS_ASSERT_MSG(condition, message) do { if (!(condition)) throw std::runtime_error(message); } while (false)
#define LFS_DEBUG_ASSERT_MSG(condition, message) assert(condition)
// Blocking scalar control reads must not rely on the PAL pageable-copy
// fence after an event marker. The caller already waits for the result.
inline hipError_t lfs_fast_ordered_d2h(void* destination, const void* source,
    size_t bytes, hipMemcpyKind kind, hipStream_t stream) {
    if(kind != hipMemcpyDeviceToHost) return hipErrorInvalidValue;
    const auto status=hipStreamSynchronize(stream);
    return status==hipSuccess ? hipMemcpyAsync(destination,source,bytes,kind,stream) : status;
}
struct LfsHipFastTiming {
    hipStream_t queue;
    bool active;
    std::chrono::steady_clock::time_point previous;
    explicit LfsHipFastTiming(hipStream_t queue) : queue(queue), active(false) {
        const char* value = std::getenv("LFS_HIP_TRAIN_PROFILE");
        active = value && std::string(value) == "1";
        if (active) {
            lfs_fast_hip_check(hipStreamSynchronize(queue), "HIP Fast timing start");
            previous = std::chrono::steady_clock::now();
        }
    }
    void mark(const char* phase) {
        if (!active) return;
        const auto submitted = std::chrono::steady_clock::now();
        lfs_fast_hip_check(hipStreamSynchronize(queue), "HIP Fast timing phase");
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[profile] HIP FastGS %s_ms=%.6f enqueue_wall_ms=%.6f completion_wait_ms=%.6f\n", phase,
            std::chrono::duration<double, std::milli>(now - previous).count(),
            std::chrono::duration<double, std::milli>(submitted - previous).count(),
            std::chrono::duration<double, std::milli>(now - submitted).count());
        previous = now;
    }
};
'''

WAVE = r'''#pragma once
#include <hip/hip_runtime.h>
#include <type_traits>
#ifdef LFS_HIP_FAST_WAVE_REDUCE
extern "C" __device__ __attribute__((const)) float __ockl_wfred_add_f32(float);
#endif
// Only full-mask collectives are routed here. The caller must admit wave32;
// divergent/member masks retain HIP's synchronization semantics unchanged.
template<class T> __device__ inline T lfs_fast_wave_shuffle(T value, int lane, int width = 32) {
#ifdef LFS_HIP_FAST_NATIVE_WAVE
    return __shfl(value, lane, width);
#else
    return __shfl_sync(0x00000000ffffffffull, value, lane, width);
#endif
}
template<class T> __device__ inline T lfs_fast_wave_shuffle_up(T value, unsigned lane, int width = 32) {
#ifdef LFS_HIP_FAST_NATIVE_WAVE
    return __shfl_up(value, lane, width);
#else
    return __shfl_up_sync(0x00000000ffffffffull, value, lane, width);
#endif
}
template<class T> __device__ inline T lfs_fast_wave_shuffle_xor(T value, int lane, int width = 32) {
#ifdef LFS_HIP_FAST_NATIVE_WAVE
    return __shfl_xor(value, lane, width);
#else
    return __shfl_xor_sync(0x00000000ffffffffull, value, lane, width);
#endif
}
__device__ inline unsigned long long lfs_fast_wave_ballot(int predicate) {
#ifdef LFS_HIP_FAST_NATIVE_WAVE
    return __ballot(predicate);
#else
    return __ballot_sync(0x00000000ffffffffull, predicate);
#endif
}
// Only session-owned helper gradients may use this specialization. Application
// outputs (densification, scores, camera and regularizers) retain atomicAdd:
// their bindings may refer to imported or fine-grained memory.
__device__ inline void lfs_fast_device_atomic_add(float* destination, float value) {
#ifdef LFS_HIP_FAST_DEVICE_ATOMICS
    (void)unsafeAtomicAdd(destination, value);
#else
    (void)atomicAdd(destination, value);
#endif
}
__device__ inline float lfs_fast_exp(float value) {
#ifdef LFS_HIP_FAST_INTRINSIC_MATH
    return __expf(value);
#else
    return expf(value);
#endif
}
__device__ inline float lfs_fast_transmittance_divide(float numerator, float denominator) {
#ifdef LFS_HIP_FAST_INTRINSIC_MATH
    return __fdividef(numerator, denominator);
#else
    return numerator / denominator;
#endif
}
template<class T> __device__ inline T& lfs_fast_gradient(T* pointer, size_t index) {
#ifdef LFS_HIP_FAST_PACKED_GRADIENTS
    using Byte = std::conditional_t<std::is_const_v<T>, const unsigned char, unsigned char>;
    return *reinterpret_cast<T*>(reinterpret_cast<Byte*>(pointer) + index * 16 * sizeof(float));
#else
    return pointer[index];
#endif
}
// Only use for a source lane uniform across the entire active wave32.
// Staged batches visit a common live mask, so their selected k is uniform.
template<class T> __device__ inline T lfs_fast_uniform_shuffle(T value, int lane) {
#ifdef LFS_HIP_FAST_UNIFORM_BROADCAST
    if constexpr (std::is_same_v<T, float>)
        return __uint_as_float(__builtin_amdgcn_readlane(__float_as_uint(value), lane));
    else {
        static_assert(std::is_integral_v<T> && sizeof(T) <= sizeof(unsigned));
        return static_cast<T>(__builtin_amdgcn_readlane(static_cast<unsigned>(value), lane));
    }
#else
    return lfs_fast_wave_shuffle(value, lane);
#endif
}
__device__ inline float2 lfs_fast_uniform_broadcast(float2 v, int lane) {
    return make_float2(lfs_fast_uniform_shuffle(v.x, lane), lfs_fast_uniform_shuffle(v.y, lane));
}
__device__ inline float3 lfs_fast_uniform_broadcast(float3 v, int lane) {
    return make_float3(lfs_fast_uniform_shuffle(v.x, lane), lfs_fast_uniform_shuffle(v.y, lane), lfs_fast_uniform_shuffle(v.z, lane));
}
__device__ inline float4 lfs_fast_uniform_broadcast(float4 v, int lane) {
    return make_float4(lfs_fast_uniform_shuffle(v.x, lane), lfs_fast_uniform_shuffle(v.y, lane),
                      lfs_fast_uniform_shuffle(v.z, lane), lfs_fast_uniform_shuffle(v.w, lane));
}
__device__ inline float2 lfs_fast_broadcast(float2 v, int lane) {
    return make_float2(lfs_fast_wave_shuffle(v.x, lane), lfs_fast_wave_shuffle(v.y, lane));
}
__device__ inline float3 lfs_fast_broadcast(float3 v, int lane) {
    return make_float3(lfs_fast_wave_shuffle(v.x, lane), lfs_fast_wave_shuffle(v.y, lane), lfs_fast_wave_shuffle(v.z, lane));
}
__device__ inline float4 lfs_fast_broadcast(float4 v, int lane) {
    return make_float4(lfs_fast_wave_shuffle(v.x, lane), lfs_fast_wave_shuffle(v.y, lane),
        lfs_fast_wave_shuffle(v.z, lane), lfs_fast_wave_shuffle(v.w, lane));
}
// Full-wave reduction of 16 fields, returned as field (lane & 15).
// Each stage retains only half the fields. Send the complementary half so
// that the XOR partner receives the same field as its own retained half.
__device__ inline float lfs_fast_reduce_scatter16(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8, float a9, float a10, float a11, float a12, float a13, float a14, float a15, unsigned lane) {
    const float s0_0 = (lane & 16u ? a8 : a0) + lfs_fast_wave_shuffle_xor(lane & 16u ? a0 : a8, 16);
    const float s0_1 = (lane & 16u ? a9 : a1) + lfs_fast_wave_shuffle_xor(lane & 16u ? a1 : a9, 16);
    const float s0_2 = (lane & 16u ? a10 : a2) + lfs_fast_wave_shuffle_xor(lane & 16u ? a2 : a10, 16);
    const float s0_3 = (lane & 16u ? a11 : a3) + lfs_fast_wave_shuffle_xor(lane & 16u ? a3 : a11, 16);
    const float s0_4 = (lane & 16u ? a12 : a4) + lfs_fast_wave_shuffle_xor(lane & 16u ? a4 : a12, 16);
    const float s0_5 = (lane & 16u ? a13 : a5) + lfs_fast_wave_shuffle_xor(lane & 16u ? a5 : a13, 16);
    const float s0_6 = (lane & 16u ? a14 : a6) + lfs_fast_wave_shuffle_xor(lane & 16u ? a6 : a14, 16);
    const float s0_7 = (lane & 16u ? a15 : a7) + lfs_fast_wave_shuffle_xor(lane & 16u ? a7 : a15, 16);
    const float s1_0 = (lane & 8u ? s0_4 : s0_0) + lfs_fast_wave_shuffle_xor(lane & 8u ? s0_0 : s0_4, 8);
    const float s1_1 = (lane & 8u ? s0_5 : s0_1) + lfs_fast_wave_shuffle_xor(lane & 8u ? s0_1 : s0_5, 8);
    const float s1_2 = (lane & 8u ? s0_6 : s0_2) + lfs_fast_wave_shuffle_xor(lane & 8u ? s0_2 : s0_6, 8);
    const float s1_3 = (lane & 8u ? s0_7 : s0_3) + lfs_fast_wave_shuffle_xor(lane & 8u ? s0_3 : s0_7, 8);
    const float s2_0 = (lane & 4u ? s1_2 : s1_0) + lfs_fast_wave_shuffle_xor(lane & 4u ? s1_0 : s1_2, 4);
    const float s2_1 = (lane & 4u ? s1_3 : s1_1) + lfs_fast_wave_shuffle_xor(lane & 4u ? s1_1 : s1_3, 4);
    const float s3_0 = (lane & 2u ? s2_1 : s2_0) + lfs_fast_wave_shuffle_xor(lane & 2u ? s2_0 : s2_1, 2);
    const float sum = s3_0 + lfs_fast_wave_shuffle_xor(s3_0, 1);
    return lfs_fast_wave_shuffle(sum, (lane * 2u) & 31u);
}

'''


def prepare(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    known_headers = {Path(path).name for path in FILES + RASTER_FILES}
    for relative in FILES + RASTER_FILES:
        text = (root / relative).read_text(encoding="utf-8")
        if relative.endswith("pop_scores.cu"):
            # The application adapter supplies its retained HIP queue directly.
            text = text.replace('#include "../../../../kernels/kernel_stream.hpp"', '')
            text = text.replace('        stream = lfs::resolve_stream(stream);', '')
        text = text.replace("<cuda_runtime.h>", "<hip/hip_runtime.h>")
        text = text.replace("<cuda_fp16.h>", "<hip/hip_fp16.h>")
        text = text.replace("<cooperative_groups.h>", "<hip/hip_cooperative_groups.h>")
        text = text.replace("<cub/cub.cuh>", "<hipcub/hipcub.hpp>")
        text = text.replace("cub::", "hipcub::")
        text = re.sub(r"\bcuda(\w+)", r"hip\1", text)
        text = re.sub(r'#include "([^"]+)"',
                      lambda m: '#include "' + (Path(m[1]).name if Path(m[1]).name in known_headers
                                 else "hip_fast_diagnostics.hpp" if m[1] in (
                                     "core/assert.hpp", "core/hip_error.hpp", "core/crash_handler.hpp")
                                 else m[1]) + '"', text)
        if relative.endswith("buffer_utils.h"):
            begin = text.index("            lfs::core::ensure_cuda_success(")
            end = text.index("            hipGetLastError();", begin)
            text = text[:begin] + '            lfs_fast_hip_check(err, "HIP FastGS status readback");\n' + text[end:]
        if relative.endswith("helper_math.h"):
            # HIP supplies these vector-vector operators as non-template
            # friends. Keeping CUDA helper duplicates makes overloads ambiguous.
            text = re.sub(
                r"inline __host__ __device__ (\w+) operator([*/])\(\1 a, \1 b\) \{[^{}]*\}",
                "", text)
            text = re.sub(
                r"inline __host__ __device__ void operator[+*/-]=\([^)]*\) \{[^{}]*\}",
                "", text)
        if relative.endswith("fused_adam_types.h"):
            anchor = "    struct FusedAdamParam {"
            if text.count(anchor) != 1:
                raise RuntimeError("FastGS Adam parameter contract changed")
            text = text.replace(anchor, anchor + "\n        float* hip_gradient = nullptr; // Private FP32 adapter; never present in CUDA.\n")
        if relative.endswith("kernel_utils.cuh"):
            anchor = "        // Non-joint state is unsupported (joint is the only codec)."
            if text.count(anchor) != 2:
                raise RuntimeError("FastGS gradient dispatch changed; review HIP FP32 adapter")
            row = '''        if (param.hip_gradient != nullptr && primitive_idx < static_cast<uint>(param.n_primitives)) {
            const uint base = primitive_idx * static_cast<uint>(param.n_attributes);
            for (uint i = 0; i < row_elements && base + i < static_cast<uint>(param.n_elements); ++i)
                param.hip_gradient[base + i] = grads[i];
        }
'''
            sh = '''        if (p.hip_gradient != nullptr && primitive_idx < static_cast<uint>(p.n_primitives)) {
            const ShNGradFromColor<ACTIVE_SH_BASES> source(mean3d, cam_position, grad_color, compute_sh_grads);
            const uint active_slots = shSlotsForBases(ACTIVE_SH_BASES);
            for (uint k = 0; k < sh_layout_slots; ++k) {
                const uint slot = shAt(primitive_idx, k, sh_layout_slots);
                if (slot * 4u + 3u < static_cast<uint>(p.n_elements))
                    reinterpret_cast<float4*>(p.hip_gradient)[slot] = source(primitive_idx, k, slot, k < active_slots);
            }
        }
'''
            text = text.replace(anchor, row, 1).replace(anchor, sh, 1)
        if relative.endswith("kernels_backward.cuh"):
            # These four ranges come from FastFrame::backward_helpers, which
            # creates private Buffer allocations without imported backing.
            # Never infer ownership of other outputs from the input model.
            text, replaced = re.subn(
                r"\batomicAdd(?=\(&grad_(?:mean2d|conic|compensated_opacity|color)\[)",
                "lfs_fast_device_atomic_add", text)
            if replaced != 9:
                raise RuntimeError("Review changed HIP helper-gradient atomic destinations")
        if relative.endswith(("kernel_utils.cuh", "kernels_backward.cuh", "kernels_forward.cuh")):
            text = re.sub(r"\bexpf\(", "lfs_fast_exp(", text)
        if relative.endswith("kernels_forward.cuh"):
            for old, new in (
                ('        const uint max_instances) {', '        const uint max_instances, const uint* depth_order) {'),
                ('        const uint work_idx = idx;', '        const uint work_idx = depth_order ? depth_order[idx] : idx;'),
                ('uint current_write_offset = work_idx == 0 ? 0 : static_cast<uint>(primitive_offsets[work_idx - 1]);',
                 'uint current_write_offset = idx == 0 ? 0 : static_cast<uint>(primitive_offsets[idx - 1]);'),
            ):
                if text.count(old) != 1:
                    raise RuntimeError('HIP depth-ordered instance seam changed: ' + old)
                text = text.replace(old, new)
            for old, new in (
                ("const uint st0 = warp_id;", "const uint st0 = (warp_id & 1u) + ((warp_id >> 1) * 4u);"),
                ("const uint st1 = warp_id + 4;", "const uint st1 = st0 + 2u;"),
            ):
                if text.count(old) != 1:
                    raise RuntimeError(f"HIP forward subtile seam changed: {old}")
                text = text.replace(old, "\n#ifdef LFS_HIP_FAST_FORWARD_ADJACENT\n        " + new + "\n#else\n        " + old + "\n#endif\n")
        if relative.endswith("forward.cu"):
            begin = text.index("        kernels::forward::blend_cu<RENDER_NORMAL, RENDER_DEPTH>")
            end = text.index('        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.blend");', begin)
            original_call = text[begin:end]
            eight_call = original_call.replace("kernels::forward::blend_cu<", "kernels::forward::blend_eight_cu<").replace("dim3(config::block_size_blend_forward)", "dim3(32)")
            dispatch = "        bool hip_forward_launched = false;\n#ifdef LFS_HIP_FAST_FORWARD_EIGHT_PIXELS\n"
            dispatch += "        if constexpr (!RENDER_NORMAL && !RENDER_DEPTH) {\n            if (warp_cull_mode == 0 && (blend_batch_override == 0 || blend_batch_override == 32)) {\n" + eight_call
            dispatch += "                hip_forward_launched = true;\n            }\n        }\n#endif\n"
            dispatch += "        if (!hip_forward_launched) {\n" + original_call + "        }\n"
            text = text[:begin] + dispatch + text[end:]
        if relative.endswith("kernels_forward.cuh"):
            body_begin = text.index("{", text.index(" blend_cu(")) + 1
            depth, body_end = 1, body_begin
            while depth:
                depth += (text[body_end] == "{") - (text[body_end] == "}")
                body_end += 1
            signature_start = text.rfind("    template <bool kRenderNormal", 0, body_begin)
            if signature_start < 0:
                raise RuntimeError("HIP forward eight-pixel signature seam changed")
            signature = text[signature_start:body_begin].replace(" blend_cu(", " blend_eight_cu(").replace("__launch_bounds__(config::block_size_blend_forward)", "__launch_bounds__(32)")
            clone = signature + eight_pixel_forward(text[body_begin:body_end - 1]) + "}\n"
            text = text[:body_end] + "\n#ifdef LFS_HIP_FAST_FORWARD_EIGHT_PIXELS\n" + clone + "\n#endif\n" + text[body_end:]
        if relative.endswith("kernels_backward.cuh"):
            alpha_adjoint = """                        const float dL_dalpha =
                            dot(transmittance_before * color, grad_c) -
                            grad_transmittance_after * transmittance_before +
                            transmittance_before * depth * grad_d +
                            transmittance_before * normal_dot_grad;"""
            recurrence = """                        grad_T = dot(grad_c, alpha * color) +
                                 alpha * depth * grad_d +
                                 alpha * normal_dot_grad +
                                 grad_transmittance_after * one_minus_alpha;"""
            if text.count(alpha_adjoint) != 1 or text.count(recurrence) != 1:
                raise RuntimeError("HIP Fast color adjoint seam changed")
            text = text.replace(alpha_adjoint, """#ifdef LFS_HIP_FAST_FACTOR_ADJOINT
                        const float color_adjoint = dot(color, grad_c) + depth * grad_d + normal_dot_grad;
                        const float dL_dalpha = transmittance_before * (color_adjoint - grad_transmittance_after);
#else
""" + alpha_adjoint + "\n#endif")
            text = text.replace(recurrence, """#ifdef LFS_HIP_FAST_FACTOR_ADJOINT
                        grad_T = alpha * color_adjoint + grad_transmittance_after * one_minus_alpha;
#else
""" + recurrence + "\n#endif")
            anchor = "transmittance_after / one_minus_alpha_safe"
            if text.count(anchor) != 1:
                raise RuntimeError("FastGS bounded transmittance division changed")
            text = text.replace(anchor, "lfs_fast_transmittance_divide(transmittance_after, one_minus_alpha_safe)")
            begin = text.index("                    // Warp-reduce")
            end = text.index("\n                }\n            }\n            // WARP_BWD_WALK_END", begin)
            original = text[begin:end]
            lane_begin = original.index("                        if (lane_id == 0u) {")
            brace = original.index("{", lane_begin)
            depth = 1
            lane_end = brace + 1
            while depth:
                if original[lane_end] == "{":
                    depth += 1
                elif original[lane_end] == "}":
                    depth -= 1
                lane_end += 1
            scatter = original[:lane_begin] + '''                        // Keep the reference reductions (including the sparse
                        // single-contributor path), then scatter their results
                        // to lanes for one coalesced atomic instruction.
                        const float value = lane_id == 0 ? mean_x : lane_id == 1 ? mean_y :
                            lane_id == 2 ? conic_x : lane_id == 3 ? conic_y : lane_id == 4 ? conic_z :
                            lane_id == 5 ? opac_g : lane_id == 6 ? color_x : lane_id == 7 ? color_y :
                            lane_id == 8 ? color_z : lane_id == 9 ? depth_g : lane_id == 10 ? normal_x :
                            lane_id == 11 ? normal_y : lane_id == 12 ? normal_z : lane_id == 13 ? dens_w :
                            lane_id == 14 ? dens_e : edge_score;
                        if (lane_id < 16 && value != 0.0f) {
                            float* destination = nullptr;
#ifdef LFS_HIP_FAST_PACKED_GRADIENTS
                            if (lane_id < 13)
                                destination = reinterpret_cast<float*>(grad_mean2d) + work_idx * 16 + lane_id;
#else
                            if (lane_id < 2)
                                destination = reinterpret_cast<float*>(grad_mean2d + work_idx) + lane_id;
                            else if (lane_id < 5)
                                destination = reinterpret_cast<float*>(grad_conic + work_idx) + lane_id - 2;
                            else if (lane_id == 5)
                                destination = grad_compensated_opacity + work_idx;
                            else if (lane_id < 9)
                                destination = reinterpret_cast<float*>(grad_color + work_idx) + lane_id - 6;
                            else if (lane_id == 9)
                                destination = grad_depth + work_idx;
                            else if (lane_id < 13) {
                                if constexpr (NORMAL_CHANNEL)
                                    destination = reinterpret_cast<float*>(grad_normal + work_idx) + lane_id - 10;
                            }
#endif
                            if (lane_id == 13 || lane_id == 14) {
                                if constexpr (DENSIFICATION_TYPE != DensificationType::None)
                                    destination = densification_info + (lane_id - 13) * n_primitives + primitive_idx;
                            } else if (lane_id == 15 && edge_score_out != nullptr)
                                destination = edge_score_out + primitive_idx;
                            if (destination != nullptr) {
                                if (lane_id < 13)
                                    lfs_fast_device_atomic_add(destination, clamp_grad(value));
                                else
                                    atomicAdd(destination, value);
                            }
                        }
''' + original[lane_end:]
            # The upstream removal-impact path has separate sum/max atomics.
            # Preserve it verbatim; field scattering is only for ordinary gradients.
            text = text[:begin] + "#ifdef LFS_HIP_FAST_SCATTER_REDUCE\nif constexpr (!IMPACT_SCORE) {\n" + scatter + "\n} else {\n" + original + "\n}\n#else\n" + original + "\n#endif\n" + text[end:]
        if relative.endswith(("kernel_utils.cuh", "kernels_backward.cuh")):
            names = r"grad_(?:mean2d|conic|depth|normal|opacity_helper|color_helper|color|compensated_opacity)"
            text = re.sub(r"\b(" + names + r")\[(work_idx|grad_color_idx)\]", r"lfs_fast_gradient(\1, \2)", text)
        if relative.endswith("/forward.h"):
            anchor = "float dilation_scale = 1.0f);"
            if text.count(anchor) != 1:
                raise RuntimeError("FastGS forward interface changed")
            text = text.replace(anchor, "float dilation_scale = 1.0f, const std::function<void(unsigned)>& hip_profile = {}, bool hip_presort_allowed = true, bool* hip_presort_attempted = nullptr);")
        if relative.endswith("/forward.cu"):
            for scalar in ('h_n_visible', 'h_n_instances'):
                seam='hipMemcpyAsync(&'+scalar+','
                if text.count(seam)!=1: raise RuntimeError('HIP scalar read seam changed: '+scalar)
                text=text.replace(seam,'lfs_fast_ordered_d2h(&'+scalar+',')
            sort_call = "hipcub::DeviceRadixSort::SortPairs("
            if text.count(sort_call) != 2:
                raise RuntimeError("HIP Fast radix query/execute seam changed")
            text = text.replace(sort_call, "lfs_fast_sort_pairs(")
            sort_include = "#include <hipcub/hipcub.hpp>"
            if text.count(sort_include) != 1:
                raise RuntimeError("HIP Fast radix include seam changed")
            text = text.replace(sort_include, sort_include + r'''
#ifdef LFS_HIP_FAST_RADIX_TUNING
#include <rocprim/device/device_radix_sort.hpp>
#endif
template<class Key, class Value, class Count>
hipError_t lfs_fast_full_sort_pairs(void* temporary, size_t& bytes,
    hipcub::DoubleBuffer<Key>& keys, hipcub::DoubleBuffer<Value>& values,
    Count count, int begin_bit, int end_bit, hipStream_t stream) {
#ifdef LFS_HIP_FAST_RADIX_TUNING
    using Config = rocprim::radix_sort_config<rocprim::default_config, rocprim::default_config,
        rocprim::radix_sort_onesweep_config<rocprim::kernel_config<256, 12>,
            rocprim::kernel_config<256, 16>, 8, rocprim::block_radix_rank_algorithm::match>, 0>;
    rocprim::double_buffer<Key> native_keys(keys.Current(), keys.Alternate());
    rocprim::double_buffer<Value> native_values(values.Current(), values.Alternate());
    const auto status = rocprim::radix_sort_pairs<Config>(temporary, bytes,
        native_keys, native_values, count, begin_bit, end_bit, stream);
    if (temporary && status == hipSuccess) {
        if (native_keys.current() != keys.Current()) keys.selector ^= 1;
        if (native_values.current() != values.Current()) values.selector ^= 1;
    }
    return status;
#else
    return hipcub::DeviceRadixSort::SortPairs(temporary, bytes, keys, values,
        count, begin_bit, end_bit, stream);
#endif
}
#include "fast_instance_sort.hpp"
#include "fast_depth_sort.hpp"
#include "fast_tile_workload.hpp"
''')
            for seam in ('                    key_end_bit,\n                    stream)',
                         '                    n_instances, 0, key_end_bit,\n                    stream)'):
                if text.count(seam) != 1:
                    raise RuntimeError('HIP instance sort call seam changed')
                text = text.replace(seam, seam[:-1] + ', hip_depth_order.indices != nullptr)')
            depth_anchor = '    check_cuda_with_fastgs_status(\n        hipcub::DeviceScan::InclusiveSum(\n            per_primitive_buffers.cub_workspace,'
            if text.count(depth_anchor) != 1:
                raise RuntimeError('HIP depth preparation seam changed')
            text = text.replace(depth_anchor, '''    const auto hip_depth_order = lfs_hip_prepare_depth_order(phase_buffers_func,
        per_primitive_buffers.depth_keys, per_primitive_buffers.n_touched_tiles,
        static_cast<unsigned>(n_visible), stream, hip_presort_allowed, hip_presort_attempted);
''' + depth_anchor)
            for old, new in (
                ('            per_primitive_buffers.n_touched_tiles,\n            per_primitive_buffers.offset,',
                 '            hip_depth_order.tile_counts,\n            per_primitive_buffers.offset,'),
                ('                static_cast<uint>(n_instances));\n            LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.create_instances");',
                 '                static_cast<uint>(n_instances), hip_depth_order.indices);\n            LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.create_instances");'),
            ):
                if text.count(old) != 1:
                    raise RuntimeError('HIP depth-ordered forward seam changed')
                text = text.replace(old,new)
            workload_anchor = '    check_cuda_with_fastgs_status(hipGetLastError(), "blend",'
            if text.count(workload_anchor) != 1:
                raise RuntimeError('HIP workload diagnostic seam changed')
            text = text.replace(workload_anchor, '    lfs_hip_report_tile_workload(per_tile_buffers.instance_ranges, per_tile_buffers.n_contributions,\n        width, height, grid.x, n_tiles_u32, stream);\n' + workload_anchor)
            replacements = {
                "    float dilation_scale) {": "    float dilation_scale, const std::function<void(unsigned)>& hip_profile, bool hip_presort_allowed, bool* hip_presort_attempted) {\n    if (hip_profile) hip_profile(0);",
                "    check_cuda_with_fastgs_status(\n        hipcub::DeviceScan::InclusiveSum(\n            per_primitive_buffers.cub_workspace,": "    if (hip_profile) hip_profile(1);\n    check_cuda_with_fastgs_status(\n        hipcub::DeviceScan::InclusiveSum(\n            per_primitive_buffers.cub_workspace,",
                "    FastGSSortWorkspace sort_workspace;": "    FastGSSortWorkspace sort_workspace;\n    if (hip_profile) hip_profile(2);",
                "    // Production: warp cull ON": "    if (hip_profile) hip_profile(3);\n    // Production: warp cull ON",
                "    ForwardResult result;": "    if (hip_profile) hip_profile(4);\n    ForwardResult result;",
            }
            # Additional diagnostic seams are inserted only into HIP copies.
            # Labels describe completed work; callbacks remain empty normally.
            micro_seams = {
                '    LFS_FASTGS_PHASE_CHECK("fastgs.forward.preprocess.visibility");': 5,
                '    LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.count_visible_blocks");': 6,
                '    LFS_FASTGS_PHASE_CHECK("hipcub::DeviceScan::InclusiveSum (FastGS visibility)");': 7,
                '    LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.compact_visible");': 8,
                '    const int n_visible = checked_fastgs_visible_count(h_n_visible, n_primitives);': 9,
                '            LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.create_instances");': 12,
                '            LFS_FASTGS_PHASE_CHECK("hipcub::DeviceRadixSort::SortPairs (Tile/Depth)");': 13,
                '            LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.extract_instance_ranges");': 14,
            }
            for seam, phase in micro_seams.items():
                if text.count(seam)!=1: raise RuntimeError(f'HIP forward micro-phase seam changed: {seam}')
                text=text.replace(seam,seam+f'\n    if (hip_profile) hip_profile({phase});')
            for seam, phase in [('        if (n_instances > 0) {',10),
                                ('            kernels::forward::create_instances_cu<<<',11)]:
                if text.count(seam)!=1: raise RuntimeError(f'HIP forward allocation seam changed: {seam}')
                text=text.replace(seam,f'    if (hip_profile) hip_profile({phase});\n'+seam)
            for anchor, replacement in replacements.items():
                if text.count(anchor) != 1:
                    raise RuntimeError(f"FastGS forward profiling boundary changed: {anchor}")
                text = text.replace(anchor, replacement)
        if relative.endswith("/backward.cu"):
            anchor = "                kernels::backward::blend_backward_cu<DENS_TYPE, NORMAL_CHANNEL, IMPACT_SCORE, WARP_CULL_MODE>"
            if text.count(anchor) != 1:
                raise RuntimeError("HIP optional-channel dispatch seam changed")
            text = text.replace(anchor, """                auto hip_launch_aux = [&]<bool HIP_DEPTH_OR_EDGE>() {
                kernels::backward::blend_backward_cu<DENS_TYPE, NORMAL_CHANNEL, IMPACT_SCORE, WARP_CULL_MODE, HIP_DEPTH_OR_EDGE>""")
            anchor = '                LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.blend_backward");'
            if text.count(anchor) != 1:
                raise RuntimeError("HIP optional-channel dispatch end changed")
            text = text.replace(anchor, anchor + """
                };
#ifdef LFS_HIP_FAST_RGB_SPECIALIZATION
                if (grad_depth == nullptr && edge_score_out == nullptr)
                    hip_launch_aux.template operator()<false>();
                else
#endif
                    hip_launch_aux.template operator()<true>();
""")
            anchor = "    hipStream_t stream) {"
            if text.count(anchor) != 1:
                raise RuntimeError("FastGS backward timing interface changed")
            text = text.replace(anchor, anchor + "\n    LfsHipFastTiming hip_timing(stream);")
            anchor = "    // Backward preprocess"
            if text.count(anchor) != 1:
                raise RuntimeError("FastGS backward timing boundary changed")
            text = text.replace(anchor, '    hip_timing.mark("blend_backward");\n' + anchor)
            end = text.rfind("}")
            text = text[:end] + '    hip_timing.mark("projection_backward");\n' + text[end:]
            call_start = text.index("                kernels::backward::blend_backward_cu<")
            call_end = text.index('                LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.blend_backward");', call_start)
            call = text[call_start:call_end]
            four_call = call.replace("blend_backward_cu<", "blend_backward_four_cu<").replace("config::block_size_blend_backward", "64")
            eight_call = call.replace("blend_backward_cu<", "blend_backward_eight_cu<").replace("config::block_size_blend_backward", "32")
            eligible = "DENS_TYPE == DensificationType::None && !NORMAL_CHANNEL && !IMPACT_SCORE && WARP_CULL_MODE == 0 && !HIP_DEPTH_OR_EDGE"
            dispatch = "#if defined(LFS_HIP_FAST_EIGHT_PIXELS)\n                if constexpr (" + eligible + ") {\n" + eight_call
            dispatch += "                } else\n#elif defined(LFS_HIP_FAST_FOUR_PIXELS)\n                if constexpr (" + eligible + ") {\n" + four_call
            dispatch += "                } else\n#endif\n                {\n" + call + "                }\n"
            # The real MCMC trainer also needs density/error accumulators.
            # The multi-pixel body already carries and scatters both fields;
            # Validated against the unoptimized path and a complete 2,000-step
            # trainer run. An explicit zero selects that diagnostic path.
            mcmc_eligible = "DENS_TYPE == DensificationType::MCMC && !NORMAL_CHANNEL && !IMPACT_SCORE && WARP_CULL_MODE == 0 && !HIP_DEPTH_OR_EDGE"
            fallback = "                {\n" + call + "                }\n"
            mcmc_dispatch = "#if defined(LFS_HIP_FAST_EIGHT_PIXELS)\n                if constexpr (" + mcmc_eligible + ") {\n                    if (std::getenv(\"LFS_HIP_FAST_DENSIFICATION_PIXELS\") != nullptr) {\n" + eight_call + "                    } else {\n" + call + "                    }\n                } else\n#elif defined(LFS_HIP_FAST_FOUR_PIXELS)\n                if constexpr (" + mcmc_eligible + ") {\n                    if (std::getenv(\"LFS_HIP_FAST_DENSIFICATION_PIXELS\") != nullptr) {\n" + four_call + "                    } else {\n" + call + "                    }\n                } else\n#endif\n" + fallback
            mcmc_dispatch = mcmc_dispatch.replace('std::getenv("LFS_HIP_FAST_DENSIFICATION_PIXELS") != nullptr',
                'const char* policy = std::getenv("LFS_HIP_FAST_DENSIFICATION_PIXELS"); !policy || policy[0] != \'0\' || policy[1] != \'\\0\'')
            if dispatch.count(fallback) != 1:
                raise RuntimeError("Review changed HIP multi-pixel fallback")
            dispatch = dispatch.replace(fallback, mcmc_dispatch)
            text = text[:call_start] + dispatch + text[call_end:]

        text = text.replace("__CUDACC__", "__HIPCC__")
        text = text.replace("__CUDA_ARCH__", "__HIP_DEVICE_COMPILE__")
        # Full CUDA warp masks must be explicitly 64-bit in HIP's sync API.
        text = re.sub(r"\b0xffffffff[uU]?\b", "0x00000000ffffffffull", text)
        for intrinsic, native in (("__shfl_sync", "lfs_fast_wave_shuffle"),
                                  ("__shfl_up_sync", "lfs_fast_wave_shuffle_up"),
                                  ("__shfl_xor_sync", "lfs_fast_wave_shuffle_xor"),
                                  ("__ballot_sync", "lfs_fast_wave_ballot")):
            text = re.sub(intrinsic + r"\(0x00000000ffffffffull,\s*", native + "(", text)
        if relative.endswith("warp_reduce.cuh"):
            text = text.replace("#include <cstdint>", "#include <cstdint>\n#include <limits>")
            anchor = "        __device__ inline T warp_reduce_sum(T val) {"
            if text.count(anchor) != 1:
                raise RuntimeError("FastGS reduction source changed")
            text = text.replace(anchor, anchor + '''
#ifdef LFS_HIP_FAST_WAVE_REDUCE
            if constexpr (std::is_same_v<T, float>)
                return __ockl_wfred_add_f32(val);
#endif
''')
        if relative.endswith("kernels_backward.cuh"):
            body_begin = text.index("{", text.index(" blend_backward_cu(")) + 1
            depth, body_end = 1, body_begin
            while depth:
                if text[body_end] == "{":
                    depth += 1
                elif text[body_end] == "}":
                    depth -= 1
                body_end += 1
            original = text[body_begin:body_end - 1]
            candidate = warp_staged_backward(original)
            signature_start = text.rfind("    template <DensificationType", 0, body_begin)
            if signature_start < 0:
                raise RuntimeError("HIP four-pixel signature seam changed")
            four_signature = text[signature_start:body_begin].replace("blend_backward_cu(", "blend_backward_four_cu(").replace("config::block_size_blend_backward, IMPACT_SCORE ? 6 : 8", "64, 8")
            four_kernel = four_signature + four_pixel_backward(candidate) + "}\n"
            eight_signature = text[signature_start:body_begin].replace("blend_backward_cu(", "blend_backward_eight_cu(").replace("config::block_size_blend_backward, IMPACT_SCORE ? 6 : 8", "32, 4")
            eight_kernel = eight_signature + four_pixel_backward(candidate, pixels=8) + "}\n"

            candidate = "#ifdef LFS_HIP_FAST_ADJACENT_PIXELS\n" + adjacent_pixel_backward(candidate) + "\n#else\n" + candidate + "\n#endif\n"
            text = text[:body_begin] + "\n#ifdef LFS_HIP_FAST_WARP_STAGING\n" + candidate + "\n#else\n" + original + "\n#endif\n" + text[body_end - 1:body_end] + "\n#ifdef LFS_HIP_FAST_FOUR_PIXELS\n" + four_kernel + "\n#endif\n#ifdef LFS_HIP_FAST_EIGHT_PIXELS\n" + eight_kernel + "\n#endif\n" + text[body_end:]
            anchor = "template <DensificationType DENSIFICATION_TYPE, bool NORMAL_CHANNEL, bool IMPACT_SCORE, int WARP_CULL_MODE = 0>"
            if text.count(anchor) != 3:
                raise RuntimeError("HIP optional-channel kernel template changed")
            text = text.replace(anchor, "template <DensificationType DENSIFICATION_TYPE, bool NORMAL_CHANNEL, bool IMPACT_SCORE, int WARP_CULL_MODE = 0, bool HIP_DEPTH_OR_EDGE = true>")
            for condition in ("grad_depth_map != nullptr", "edge_weight_map != nullptr", "edge_score_out != nullptr"):
                text = text.replace(condition, "(HIP_DEPTH_OR_EDGE && " + condition + ")")
        if relative.endswith("kernels_forward.cuh"):
            text = compact_sort_kernels(text)
        if relative.endswith("/forward.cu"):
            text = compact_sort_forward(text)
        guard = """
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ != 32
#error "The private FastGS HIP port currently requires wave32"
#endif
#endif
"""
        text = text.replace("#pragma once", '#pragma once\n#include "hip_fast_wave.hpp"\n' + guard, 1)
        target = output / Path(relative).name
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
    target = output / "hip_fast_diagnostics.hpp"
    if not target.exists() or target.read_text(encoding="utf-8") != DIAGNOSTICS:
        target.write_text(DIAGNOSTICS, encoding="utf-8")
    target = output / "hip_fast_wave.hpp"
    if not target.exists() or target.read_text(encoding="utf-8") != WAVE:
        target.write_text(WAVE, encoding="utf-8")
    target = output / "fast_instance_sort.hpp"
    instance_sort = (Path(__file__).parent / "fast_instance_sort.hpp").read_text(encoding="utf-8")
    if not target.exists() or target.read_text(encoding="utf-8") != instance_sort:
        target.write_text(instance_sort, encoding="utf-8")
    target = output / "fast_depth_sort.hpp"
    depth_sort = (Path(__file__).parent / "fast_depth_sort.hpp").read_text(encoding="utf-8")
    if not target.exists() or target.read_text(encoding="utf-8") != depth_sort:
        target.write_text(depth_sort, encoding="utf-8")
    target = output / "fast_tile_workload.hpp"
    workload = (Path(__file__).parent / "fast_tile_workload.hpp").read_text(encoding="utf-8")
    if not target.exists() or target.read_text(encoding="utf-8") != workload:
        target.write_text(workload, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.output)
