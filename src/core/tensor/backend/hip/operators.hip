/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lens_kernels.hpp"
#include "pointwise_runtime.hpp"
#include "tensor_ops_runtime.hpp"
#include <core/environment_math.hpp>

namespace {
    __global__ void environment_composite_kernel(const float* rgb, const float* alpha,
                                                 const float* environment, uint8_t* output,
                                                 LfsHipEnvironment p, size_t pixels) {
        namespace em = lfs::core::envmath;
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < pixels; i += size_t(gridDim.x) * blockDim.x) {
            const int x = int(i % size_t(p.band_width));
            const int y = int(i / size_t(p.band_width));
            auto direction = em::environmentWorldDirection(float(x), float(p.y_offset + y),
                                                           float(p.full_width), float(p.full_height), p.equirect_view != 0,
                                                           p.focal_x, p.focal_y, p.center_x, p.center_y, p.rotation);
            direction = em::normalized(em::rotateAroundY(direction, p.env_rotation_radians));
            const auto uv = em::equirectUvForDirection(direction);
            const auto tap = em::environmentBilinearTap(uv.u, uv.v, p.env_width, p.env_height);
            const auto fetch = [&](int px, int py) {
                const size_t offset = (size_t(py) * size_t(p.env_width) + size_t(px)) * 3;
                return em::Vec3{environment[offset], environment[offset + 1], environment[offset + 2]};
            };
            const auto top = em::mix(fetch(tap.x0, tap.y0), fetch(tap.x1, tap.y0), tap.tx);
            const auto bottom = em::mix(fetch(tap.x0, tap.y1), fetch(tap.x1, tap.y1), tap.tx);
            const auto background = em::shadeEnvironmentRadiance(em::mix(top, bottom, tap.ty), p.exposure_factor);
            const em::Vec3 color{rgb[i], rgb[pixels + i], rgb[2 * pixels + i]};
            const auto mixed = em::mix(background, color, alpha[i]);
            const auto to_byte = [](float value) {
                return static_cast<uint8_t>(fminf(fmaxf(value, 0.f), 1.f) * 255.f + 0.5f);
            };
            output[3 * i] = to_byte(mixed.x);
            output[3 * i + 1] = to_byte(mixed.y);
            output[3 * i + 2] = to_byte(mixed.z);
        }
    }
} // namespace
int lfs_hip_environment_composite(LfsHipContext* context, LfsHipView rgb, LfsHipView alpha,
                                  LfsHipView environment, LfsHipView output,
                                  const LfsHipEnvironment* params, LfsHipError* error) {
    return boundary(error, [&] {
        require(params && params->band_width > 0 && params->band_height > 0 &&
                    params->full_width == params->band_width && params->full_height >= params->band_height &&
                    params->y_offset >= 0 && params->y_offset <= params->full_height - params->band_height &&
                    params->env_width > 0 && params->env_height > 0,
                "Invalid HIP environment composite dimensions");
        const size_t pixels = size_t(params->band_width) * size_t(params->band_height);
        const size_t env_pixels = size_t(params->env_width) * size_t(params->env_height);
        require(pixels <= INT32_MAX && env_pixels <= INT32_MAX && pixels <= SIZE_MAX / 3 && env_pixels <= SIZE_MAX / 3,
                "HIP environment composite extent overflow");
        validate(context, rgb.buffer, rgb.offset, float_bytes(pixels * 3));
        validate(context, alpha.buffer, alpha.offset, float_bytes(pixels));
        validate(context, environment.buffer, environment.offset, float_bytes(env_pixels * 3));
        validate(context, output.buffer, output.offset, pixels * 3);
        require(rgb.offset % 4 == 0 && alpha.offset % 4 == 0 && environment.offset % 4 == 0,
                "Unaligned HIP environment composite input");
        require(output.buffer != rgb.buffer && output.buffer != alpha.buffer &&
                    output.buffer != environment.buffer,
                "HIP environment output overlaps an input");
        DeviceScope device(context->device);
        hipLaunchKernelGGL(environment_composite_kernel, dim3(blocks(pixels)), dim3(256), 0, context->stream,
                           reinterpret_cast<const float*>(static_cast<const char*>(rgb.buffer->data) + rgb.offset),
                           reinterpret_cast<const float*>(static_cast<const char*>(alpha.buffer->data) + alpha.offset),
                           reinterpret_cast<const float*>(static_cast<const char*>(environment.buffer->data) + environment.offset),
                           static_cast<uint8_t*>(output.buffer->data) + output.offset, *params, pixels);
        check(hipGetLastError());
    });
}
int lfs_hip_inverse_lens(LfsHipContext* context, LfsHipView output, const LfsHipLens* lens, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && lens && lens->model_type == 0, "HIP lens inverse currently requires pinhole calibration");
        require(lens->src_width > 0 && lens->src_height > 0 && uint64_t(lens->src_width) * lens->src_height <= INT32_MAX / 2,
                "HIP lens inverse extent exceeds signed indexing");
        require(std::isfinite(lens->src_fx) && std::isfinite(lens->src_fy) && lens->src_fx > 0 && lens->src_fy > 0 &&
                    std::isfinite(lens->src_cx) && std::isfinite(lens->src_cy) && lens->num_distortion >= 0 && lens->num_distortion <= 12,
                "Invalid HIP lens calibration");
        for (int i = 0; i < lens->num_distortion; ++i)
            require(std::isfinite(lens->distortion[i]), "Nonfinite HIP lens distortion");
        const size_t bytes = size_t(lens->src_width) * lens->src_height * sizeof(float2);
        validate(context, output.buffer, output.offset, bytes);
        require(output.offset % alignof(float2) == 0, "Misaligned HIP lens output");
        DeviceScope device(context->device);
        const dim3 block(16, 16), grid((lens->src_width + 15) / 16, (lens->src_height + 15) / 16);
        hipLaunchKernelGGL(hip_lens::inverse_distortion_sample_map_kernel, grid, block, 0, context->stream,
                           reinterpret_cast<float2*>(static_cast<char*>(output.buffer->data) + output.offset), *lens);
        check(hipGetLastError());
    });
}
namespace {
    template <typename T>
    __global__ void identity_rgb(const T* input, float* output, size_t count) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
             i < count; i += size_t(gridDim.x) * blockDim.x)
            output[i] = hip_lens::normalized_sample(input[i]);
    }
} // namespace
int lfs_hip_warp_rgb(LfsHipContext* context, LfsHipView input, uint32_t dtype,
                     LfsHipView output, const LfsHipLens* lens, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && lens && lens->model_type == 0, "HIP RGB warp requires pinhole calibration");
        require(dtype == LFS_HIP_F32 || dtype == LFS_HIP_U8, "HIP RGB warp input must be float32 or uint8");
        require(lens->src_width > 0 && lens->src_height > 0 && lens->dst_width > 0 && lens->dst_height > 0 &&
                    uint64_t(lens->src_width) * lens->src_height <= INT32_MAX / 3 &&
                    uint64_t(lens->dst_width) * lens->dst_height <= INT32_MAX / 3,
                "HIP RGB warp extent exceeds signed indexing");
        require(std::isfinite(lens->src_fx) && std::isfinite(lens->src_fy) && lens->src_fx > 0 && lens->src_fy > 0 &&
                    std::isfinite(lens->dst_fx) && std::isfinite(lens->dst_fy) && lens->dst_fx > 0 && lens->dst_fy > 0 &&
                    std::isfinite(lens->src_cx) && std::isfinite(lens->src_cy) && std::isfinite(lens->dst_cx) && std::isfinite(lens->dst_cy) &&
                    lens->num_distortion >= 0 && lens->num_distortion <= 12,
                "Invalid HIP RGB warp calibration");
        bool identity = lens->src_width == lens->dst_width && lens->src_height == lens->dst_height &&
                        lens->src_fx == lens->dst_fx && lens->src_fy == lens->dst_fy &&
                        lens->src_cx == lens->dst_cx && lens->src_cy == lens->dst_cy;
        for (int i = 0; i < lens->num_distortion; ++i) {
            require(std::isfinite(lens->distortion[i]), "Nonfinite HIP RGB warp distortion");
            identity = identity && lens->distortion[i] == 0;
        }
        const float footprint = std::max(lens->src_fx / lens->dst_fx, lens->src_fy / lens->dst_fy);
        require(std::isfinite(footprint) && footprint <= 64, "HIP RGB warp sampling footprint exceeds experimental limit 64");
        const int quadrature = std::max(8, int(std::ceil(footprint)));
        const uint64_t row_work = uint64_t(lens->dst_width) * quadrature * quadrature;
        require(identity || row_work <= 2 * 1024 * 1024, "HIP RGB warp row exceeds bounded sampling work");
        const size_t input_count = size_t(lens->src_width) * lens->src_height * 3;
        const size_t output_count = size_t(lens->dst_width) * lens->dst_height * 3;
        const size_t input_bytes = input_count * (dtype == LFS_HIP_U8 ? 1 : sizeof(float));
        const size_t output_bytes = output_count * sizeof(float);
        validate(context, input.buffer, input.offset, input_bytes);
        validate(context, output.buffer, output.offset, output_bytes);
        require(output.offset % 4 == 0 && (dtype == LFS_HIP_U8 || input.offset % 4 == 0), "Misaligned HIP RGB warp view");
        require(input.buffer != output.buffer || input.offset >= output.offset + output_bytes || output.offset >= input.offset + input_bytes,
                "HIP RGB warp input/output overlap");
        DeviceScope device(context->device);
        const auto* source = static_cast<const char*>(input.buffer->data) + input.offset;
        auto* destination = reinterpret_cast<float*>(static_cast<char*>(output.buffer->data) + output.offset);
        if (identity) {
            if (dtype == LFS_HIP_U8)
                hipLaunchKernelGGL((identity_rgb<uint8_t>), dim3(blocks(output_count)), dim3(256), 0, context->stream,
                                   reinterpret_cast<const uint8_t*>(source), destination, output_count);
            else
                hipLaunchKernelGGL((identity_rgb<float>), dim3(blocks(output_count)), dim3(256), 0, context->stream,
                                   reinterpret_cast<const float*>(source), destination, output_count);
            check(hipGetLastError());
            return;
        }
        const int rows = std::max(1, std::min(32, int((2 * 1024 * 1024) / row_work)));
        for (int y = 0; y < lens->dst_height; y += rows) {
            const int end = std::min(lens->dst_height, y + rows);
            const dim3 block(16, 16), grid((lens->dst_width + 15) / 16, (end - y + 15) / 16);
            if (dtype == LFS_HIP_U8)
                hipLaunchKernelGGL((hip_lens::undistort_image_kernel<uint8_t, 3>), grid, block, 0, context->stream,
                                   reinterpret_cast<const uint8_t*>(source), destination, 3, y, end, quadrature, *lens);
            else
                hipLaunchKernelGGL((hip_lens::undistort_image_kernel<float, 3>), grid, block, 0, context->stream,
                                   reinterpret_cast<const float*>(source), destination, 3, y, end, quadrature, *lens);
            check(hipGetLastError());
        }
    });
}
#include "affine_runtime.hpp"
#include "bias_runtime.hpp"
#include "device_identity_runtime.hpp"
#include "export_runtime.hpp"
#include "expression_runtime.hpp"
#include "inference_runtime.hpp"
#include "inverse_warp_runtime.hpp"
#include "morton_runtime.hpp"
#include "multinomial_runtime.hpp"
#include "nn_attention_runtime.hpp"
#include "nn_conv_runtime.hpp"
#include "nn_linear_runtime.hpp"
#include "nn_runtime.hpp"
#include "point_filter_runtime.hpp"
#include "point_tree_runtime.hpp"
#include "pool_runtime.hpp"
#include "ppisp_runtime.hpp"
#include "sh_codec_runtime.hpp"
#include "shared_image_runtime.hpp"
#include "shared_resize_runtime.hpp"
#include "shared_warp_runtime.hpp"
#include "simplify_runtime.hpp"
#include "spatial_runtime.hpp"
#include "timing_runtime.hpp"

hipError_t lfs_hip_release_expression_internal(LfsHipContext* context, LfsHipExpression* resource) {
    try {
        release_expression_native(context, resource);
        return hipSuccess;
    } catch (const HipFailure& failure) {
        return failure.code;
    }
}

hipError_t lfs_hip_release_timing_internal(LfsHipContext* context, LfsHipTiming* resource) {
    try {
        release_timing_native(context, resource);
        return hipSuccess;
    } catch (const HipFailure& failure) {
        return failure.code;
    }
}
