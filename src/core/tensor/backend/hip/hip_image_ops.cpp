/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/tensor_impl.hpp"
#include "hip_backend_ops.hpp"
#include <core/tensor_execution.hpp>
#include <core/tensor_image.hpp>
#include <cstdlib>
#include <cstring>
#include <format>
#include <new>
#include <stdexcept>
namespace lfs::core::internal {
    Tensor HipBackendOps::shared_image_resize(const Tensor& input, int height, int width, unsigned mode, int kernel_size, ExecContext execution) {
        if (execution.cuda_stream || mode > 4 || !input.is_valid() || input.device() != Device::GPU ||
            gpu_backend_of(input) != GpuBackend::HIP || !input.is_contiguous() || height <= 0 || width <= 0 ||
            (input.dtype() != DataType::Float32 && input.dtype() != DataType::UInt8) ||
            input.ndim() != (mode == 1 || mode == 3 ? 2u : 3u))
            throw std::invalid_argument("HIP shared resize requires native contiguous image storage and valid layout/extent");
        const size_t channels = mode == 0 ? input.size(2) : mode == 1 || mode == 3 ? 1
                                                                                   : input.size(0);
        const size_t sh = input.size(mode == 0 ? 0 : input.ndim() - 2), sw = input.size(mode == 0 ? 1 : input.ndim() - 1);
        if (!channels || channels > 4 || !sh || !sw || sh > INT32_MAX || sw > INT32_MAX ||
            sh * sw > INT32_MAX / channels || uint64_t(height) * width > INT32_MAX / channels ||
            ((mode == 2 || mode == 4) && channels != 3) || (mode >= 2 && input.dtype() != DataType::Float32) || (mode < 3 && kernel_size <= 0))
            throw std::invalid_argument("HIP shared resize image dimensions/dtype exceed native contract");
        LfsHipSharedResize descriptor{};
        descriptor.struct_bytes = sizeof(descriptor);
        descriptor.mode = mode;
        descriptor.dtype = static_cast<uint32_t>(input.dtype());
        descriptor.channels = uint32_t(channels);
        descriptor.kernel_size = kernel_size;
        descriptor.source_h = int(sh);
        descriptor.source_w = int(sw);
        descriptor.destination_h = height;
        descriptor.destination_w = width;
        size_t count_x = 0, count_y = 0;
        if (mode < 3) {
            const uint64_t stride_x = (2ull * kernel_size * sw + width - 1) / width + 1, stride_y = (2ull * kernel_size * sh + height - 1) / height + 1;
            if (stride_x > UINT32_MAX || stride_y > UINT32_MAX || stride_x * width > UINT32_MAX / 4 || stride_y * height > UINT32_MAX / 4)
                throw std::invalid_argument("HIP Lanczos coefficient dimensions exceed native indexing");
            descriptor.stride_x = uint32_t(stride_x);
            descriptor.stride_y = uint32_t(stride_y);
            count_x = stride_x * width;
            count_y = stride_y * height;
        }
        const TensorExecutionTarget::Scope scope(TensorExecutionTarget::default_queue(GpuBackend::HIP));
        const auto shape = mode == 1 || mode == 3 ? TensorShape({size_t(height), size_t(width)}) : TensorShape({channels, size_t(height), size_t(width)});
        auto output = Tensor::empty(shape, Device::GPU, DataType::Float32);
        auto x = count_x ? Tensor::empty({count_x}, Device::GPU, DataType::Float32) : Tensor{};
        auto y = count_y ? Tensor::empty({count_y}, Device::GPU, DataType::Float32) : Tensor{};
        pin_operands({&input});
        launch_shared_resize(descriptor, storage_ref(input), storage_ref(output), x.is_valid() ? storage_ref(x) : StorageRef{}, y.is_valid() ? storage_ref(y) : StorageRef{}, execution);
        synchronize_stream(execution); // Match the existing resize helpers' completion/lifetime contract.
        return output;
    }
    Tensor HipBackendOps::image_resize_prior(const Tensor& input, int height, int width, bool normal, ExecContext execution) {
        return shared_image_resize(input, height, width, normal ? 4 : 3, 0, execution);
    }

    Tensor HipBackendOps::image_undistort(const Tensor& input, const UndistortParams& params, bool mask, ExecContext execution) {
        return image_warp(input, params, mask ? 1 : 0, false, nullptr, execution);
    }

    // Tensor allocation belongs to lfs_core; the storage-only host adapter and
    // its isolated tests do not link Tensor factories or a second core runtime.
    Tensor HipBackendOps::image_warp(const Tensor& input, const UndistortParams& p, int mode, bool inverse, Tensor* validity, ExecContext execution) {
        if (mode >= 0 && mode <= 3 && inverse) {
            const size_t channels = input.ndim() == 2 ? 1 : input.ndim() == 3 ? input.size(0)
                                                                              : 0;
            if (execution.cuda_stream || !input.is_valid() || gpu_backend_of(input) != GpuBackend::HIP ||
                !input.is_contiguous() || input.dtype() != DataType::Float32 || !channels || channels > 4 ||
                (mode == 0 && input.ndim() != 3) || (mode == 2 && channels != 1) ||
                (mode == 3 && (input.ndim() != 3 || channels != 3)) || (mode > 0 && validity) ||
                p.src_width <= 0 || p.src_height <= 0 || p.dst_width <= 0 || p.dst_height <= 0 ||
                input.size(input.ndim() - 2) != size_t(p.dst_height) || input.size(input.ndim() - 1) != size_t(p.dst_width) ||
                uint64_t(p.src_width) * p.src_height > INT32_MAX / channels || uint64_t(p.dst_width) * p.dst_height > INT32_MAX / channels)
                throw std::invalid_argument(std::format("HIP inverse camera warp requires native float images matching the undistorted camera extent (mode={}, shape={}, dtype={}, expected={}x{}, validity={}, nondefault_stream={})",
                                                        mode, input.shape().str(), int(input.dtype()), p.dst_height, p.dst_width, validity != nullptr, execution.cuda_stream != nullptr));
            const TensorExecutionTarget::Scope scope(TensorExecutionTarget::default_queue(GpuBackend::HIP));
            const auto shape = input.ndim() == 2 ? TensorShape({size_t(p.src_height), size_t(p.src_width)}) : TensorShape({channels, size_t(p.src_height), size_t(p.src_width)});
            auto output = Tensor::empty(shape, Device::GPU, DataType::Float32);
            auto mask = mode == 0 ? Tensor::empty({size_t(p.src_height), size_t(p.src_width)}, Device::GPU, DataType::UInt8) : Tensor{};
            pin_operands({&input, &mask});
            launch_inverse_warp(storage_ref(input), storage_ref(output), mask.is_valid() ? storage_ref(mask) : StorageRef{}, p, unsigned(channels), unsigned(mode), execution);
            if (validity)
                *validity = mask;
            return output;
        }
        if (mode >= 0 && mode <= 3 && !inverse && !validity) {
            const size_t channels = input.ndim() == 2 ? 1 : input.ndim() == 3 ? input.size(0)
                                                                              : 0;
            if (execution.cuda_stream || !input.is_valid() || input.device() != Device::GPU || gpu_backend_of(input) != GpuBackend::HIP ||
                !input.is_contiguous() || !channels || channels > 4 || (mode == 0 && input.ndim() != 3) ||
                (input.dtype() != DataType::Float32 && input.dtype() != DataType::UInt8) || (mode > 0 && input.dtype() != DataType::Float32) ||
                (mode == 2 && channels != 1) || (mode == 3 && (input.ndim() != 3 || channels != 3)) ||
                static_cast<int>(p.model_type) < 0 || static_cast<int>(p.model_type) > 4 ||
                p.src_width <= 0 || p.src_height <= 0 || p.dst_width <= 0 || p.dst_height <= 0 ||
                input.size(input.ndim() - 2) != size_t(p.src_height) || input.size(input.ndim() - 1) != size_t(p.src_width) ||
                uint64_t(p.src_width) * p.src_height > INT32_MAX / channels || uint64_t(p.dst_width) * p.dst_height > INT32_MAX / channels)
                throw std::invalid_argument("HIP forward camera warp requires matching native CHW color or HW/CHW float prior storage");
            const TensorExecutionTarget::Scope scope(TensorExecutionTarget::default_queue(GpuBackend::HIP));
            const auto shape = input.ndim() == 2 ? TensorShape({size_t(p.dst_height), size_t(p.dst_width)}) : TensorShape({channels, size_t(p.dst_height), size_t(p.dst_width)});
            auto output = Tensor::empty(shape, Device::GPU, DataType::Float32);
            pin_operands({&input});
            launch_shared_warp(storage_ref(input), storage_ref(output), p, unsigned(channels), unsigned(mode), execution);
            return output;
        }
        if (mode != 4 || !inverse || validity)
            throw std::invalid_argument("HIP image warp supports only forward pinhole RGB and inverse lens maps");
        if (p.model_type != CameraModelType::PINHOLE || p.src_width <= 0 || p.src_height <= 0 ||
            uint64_t(p.src_width) * p.src_height > INT32_MAX / 2)
            throw std::invalid_argument("Unsupported HIP inverse lens model or extent");
        const GpuBackendScope scope(GpuBackend::HIP);
        auto output = Tensor::empty({size_t(p.src_height), size_t(p.src_width), 2}, Device::GPU, DataType::Float32);
        launch_inverse_lens(storage_ref(output), p, execution);
        return output;
    }
} // namespace lfs::core::internal
