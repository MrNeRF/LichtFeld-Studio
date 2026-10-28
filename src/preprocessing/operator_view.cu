/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "preprocessing/operator_view.hpp"

#include <cmath>
#include <stdexcept>

namespace lfs::preprocessing {
    namespace {
        struct Coordinate {
            float x, y;
            bool valid;
        };

        __host__ __device__ float3 unproject(float x, float y, float half_angle) {
            const float radius = sqrtf(x * x + y * y);
            const float theta = radius * half_angle;
            const float scale = radius > 1e-7f ? sinf(theta) / radius : half_angle;
            return {x * scale, y * scale, cosf(theta)};
        }

        __host__ __device__ Coordinate project(float3 ray, float half_angle) {
            const float radius = sqrtf(ray.x * ray.x + ray.y * ray.y);
            const float theta = atan2f(radius, ray.z);
            const float scale = radius > 1e-7f ? theta / (radius * half_angle) : 0;
            return {ray.x * scale, ray.y * scale, theta <= half_angle};
        }

        __host__ __device__ Coordinate coordinate(const OperatorView& view, float x, float y, bool inverse) {
            const float nx = inverse ? ((x + 0.5f) / view.width - view.center_x) / view.radius_x
                                     : (x + 0.5f) * 2 / view.size - 1;
            const float ny = inverse ? ((y + 0.5f) / view.height - view.center_y) / view.radius_y
                                     : (y + 0.5f) * 2 / view.size - 1;
            if (nx * nx + ny * ny > 1)
                return {0, 0, false};
            const auto ray = unproject(nx, ny, inverse ? view.source_half_angle : view.view_half_angle);
            const float* r = view.rotation;
            const float3 rotated = inverse
                                       ? float3{r[0] * ray.x + r[3] * ray.y + r[6] * ray.z,
                                                r[1] * ray.x + r[4] * ray.y + r[7] * ray.z,
                                                r[2] * ray.x + r[5] * ray.y + r[8] * ray.z}
                                       : float3{r[0] * ray.x + r[1] * ray.y + r[2] * ray.z,
                                                r[3] * ray.x + r[4] * ray.y + r[5] * ray.z,
                                                r[6] * ray.x + r[7] * ray.y + r[8] * ray.z};
            const auto result = project(rotated, inverse ? view.view_half_angle : view.source_half_angle);
            if (inverse)
                return {(result.x + 1) * view.size * 0.5f - 0.5f,
                        (result.y + 1) * view.size * 0.5f - 0.5f, result.valid};
            return {(result.x * view.radius_x + view.center_x) * view.width - 0.5f,
                    (result.y * view.radius_y + view.center_y) * view.height - 0.5f, result.valid};
        }

        __global__ void remap(const float* input, float* output, OperatorView view, int channels, bool inverse) {
            const int out_w = inverse ? view.width : view.size, out_h = inverse ? view.height : view.size;
            const int in_w = inverse ? view.size : view.width, in_h = inverse ? view.size : view.height;
            const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= size_t(out_w) * out_h * channels)
                return;
            const int plane = int(index / (size_t(out_w) * out_h));
            const auto point = coordinate(view, float(index % out_w), float(index / out_w % out_h), inverse);
            float value = inverse ? -10000.0f : 0;
            if (point.valid && point.x >= 0 && point.y >= 0 && point.x <= in_w - 1 && point.y <= in_h - 1) {
                const int x0 = int(floorf(point.x)), y0 = int(floorf(point.y));
                const int x1 = min(x0 + 1, in_w - 1), y1 = min(y0 + 1, in_h - 1);
                const float tx = point.x - x0, ty = point.y - y0;
                const float* data = input + size_t(plane) * in_w * in_h;
                value = (1 - ty) * ((1 - tx) * data[y0 * in_w + x0] + tx * data[y0 * in_w + x1]) +
                        ty * ((1 - tx) * data[y1 * in_w + x0] + tx * data[y1 * in_w + x1]);
            }
            output[index] = value;
        }

        core::Tensor resample(const core::Tensor& input, const OperatorView& view, bool inverse) {
            using namespace core;
            if (input.device() != Device::CUDA || input.dtype() != DataType::Float32 ||
                input.ndim() != 4 || input.size(0) != 1 ||
                input.size(2) != size_t(inverse ? view.size : view.height) ||
                input.size(3) != size_t(inverse ? view.size : view.width))
                throw std::invalid_argument("Operator view input must be Float32 CUDA NCHW with matching dimensions");
            const auto contiguous = input.contiguous();
            const cudaStream_t stream = contiguous.stream();
            const CUDAStreamGuard guard(stream);
            auto output = Tensor::empty({1, input.size(1), size_t(inverse ? view.height : view.size),
                                         size_t(inverse ? view.width : view.size)},
                                        Device::CUDA, DataType::Float32);
            remap<<<(output.numel() + 255) / 256, 256, 0, stream>>>(contiguous.ptr<float>(), output.ptr<float>(),
                                                                    view, int(input.size(1)), inverse);
            LFS_CUDA_LAUNCH_CHECK(stream, "preprocessing.operator_view");
            // The input can be a temporary encoder image; retain it through completion.
            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
            return output;
        }
    } // namespace

    OperatorView make_operator_view(int width, int height, float operator_x, float operator_y,
                                    std::array<float, 4> circle, float source_fov_degrees,
                                    float view_fov_degrees, int size) {
        if (width <= 0 || height <= 0 || size <= 0 || size > 4096 ||
            !std::isfinite(operator_x) || !std::isfinite(operator_y) ||
            !(source_fov_degrees > 0 && source_fov_degrees < 360) ||
            !(view_fov_degrees > 0 && view_fov_degrees < 360) ||
            !std::isfinite(circle[0]) || !std::isfinite(circle[1]) ||
            !(circle[2] > 0 && circle[2] <= 1) || !(circle[3] > 0 && circle[3] <= 1))
            throw std::invalid_argument("Invalid operator view geometry");
        constexpr float radians = 0.008726646259971648f;
        OperatorView view{width, height, size, circle[0], circle[1], circle[2], circle[3], source_fov_degrees * radians, view_fov_degrees * radians, {}};
        const float x = ((operator_x + 0.5f) / width - circle[0]) / circle[2];
        const float y = ((operator_y + 0.5f) / height - circle[1]) / circle[3];
        if (x * x + y * y > 1)
            throw std::invalid_argument("Operator point is outside the image circle");
        const auto forward = unproject(x, y, view.source_half_angle);
        const float denominator = 1 + forward.z;
        if (denominator < 1e-5f)
            throw std::invalid_argument("Operator point is at the projection antipode");
        const float rotation[9]{1 - forward.x * forward.x / denominator, -forward.x * forward.y / denominator, forward.x,
                                -forward.x * forward.y / denominator, 1 - forward.y * forward.y / denominator, forward.y,
                                -forward.x, -forward.y, forward.z};
        for (int index = 0; index < 9; ++index)
            view.rotation[index] = rotation[index];
        return view;
    }

    std::array<float, 3> operator_view_coordinate(const OperatorView& view, float x, float y, bool inverse) {
        const auto point = coordinate(view, x, y, inverse);
        return {point.x, point.y, point.valid ? 1.0f : 0.0f};
    }

    core::Tensor render_operator_view(const core::Tensor& image, const OperatorView& view) {
        return resample(image, view, false);
    }

    core::Tensor restore_operator_mask(const core::Tensor& logits, const OperatorView& view) {
        return resample(logits, view, true);
    }
} // namespace lfs::preprocessing
