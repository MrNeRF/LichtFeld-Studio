/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_environment.hpp"
#include "core/tensor_backend.hpp"
#include "internal/environment_composite.hpp"
#include "internal/tensor_impl.hpp"
namespace lfs::core {
    Tensor environment_composite(const Tensor& rgb, const Tensor& alpha, const Tensor& environment,
                                 const EnvironmentCompositeParams& p) {
        LFS_ASSERT_MSG(rgb.is_valid() && rgb.dtype() == DataType::Float32 && rgb.ndim() == 3 && rgb.size(0) == 3 &&
                           rgb.size(1) == size_t(p.band_height) && rgb.size(2) == size_t(p.band_width) && p.band_width > 0 && p.band_height > 0 &&
                           p.band_width == p.full_width && p.full_height >= p.band_height && p.y_offset >= 0 && p.y_offset <= p.full_height - p.band_height &&
                           size_t(p.band_width) * p.band_height <= INT32_MAX,
                       "Invalid environment composite image or region");
        LFS_ASSERT_MSG(alpha.is_valid() && alpha.dtype() == DataType::Float32 && alpha.numel() == rgb.numel() / 3 &&
                           environment.is_valid() && environment.dtype() == DataType::Float32 && p.env_width > 0 && p.env_height > 0 &&
                           size_t(p.env_width) * p.env_height <= INT32_MAX && environment.shape() == TensorShape({size_t(p.env_height), size_t(p.env_width), 3}) &&
                           alpha.device() == rgb.device() && environment.device() == rgb.device(),
                       "Invalid environment composite inputs");
        internal::require_same_gpu_backend(rgb, alpha, "environment_composite");
        internal::require_same_gpu_backend(rgb, environment, "environment_composite");
        const GpuBackendScope scope(gpu_backend_of(rgb).value_or(default_gpu_backend()));
        const auto input = rgb.contiguous(), opacity = alpha.contiguous(), env = environment.contiguous();
        auto output = Tensor::empty({size_t(p.band_height), size_t(p.band_width), 3}, rgb.device(), DataType::UInt8);
        pin_operands({&input, &opacity, &env, &output});
        if (rgb.device() == Device::GPU) {
            const auto stream = prepare_inputs_for_stream({&input, &opacity, &env}, output.stream());
            output.set_stream(stream);
            internal::backend_ops_for(output).environment_composite(internal::storage_ref(input), internal::storage_ref(opacity),
                                                                    internal::storage_ref(env), internal::storage_ref(output), p, internal::ExecContext{stream});
        } else {
            for (int i = 0; i < p.band_width * p.band_height; ++i)
                internal::composite_environment_pixel(p, env.ptr<float>(), input.ptr<float>(), opacity.ptr<float>(), output.ptr<unsigned char>(), i);
        }
        return output;
    }
} // namespace lfs::core
