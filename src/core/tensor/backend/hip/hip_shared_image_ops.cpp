/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/tensor_impl.hpp"
#include "core/shared_image_ops.hpp"
#include "hip_backend_ops.hpp"

namespace lfs::core {
    namespace {
        using namespace gpu_ops;
        using internal::storage_ref;

        internal::HipBackendOps& image_backend() {
            return static_cast<internal::HipBackendOps&>(internal::backend_ops(GpuBackend::HIP));
        }
        internal::StorageRef image_storage(const Tensor& tensor) {
            if (!tensor.is_valid() || gpu_backend_of(tensor) != GpuBackend::HIP || !tensor.is_contiguous())
                throw std::invalid_argument("HIP image operations require contiguous native storage");
            return storage_ref(tensor);
        }
        void dispatch(LfsHipSharedImage p, const Tensor* source, Tensor& destination, Tensor* auxiliary = nullptr) {
            p.struct_bytes = sizeof(p);
            pin_operands({source, &destination, auxiliary});
            image_backend().shared_image(p, source ? image_storage(*source) : internal::StorageRef{},
                                         image_storage(destination), auxiliary ? image_storage(*auxiliary) : internal::StorageRef{}, {});
        }
        void sentinel_fill(Tensor& bytes, uint32_t seed) {
            if (bytes.dtype() != DataType::UInt8)
                throw std::invalid_argument("Sentinel fill requires bytes");
            dispatch({.operation = 16, .seed = seed, .height = 1, .width = bytes.bytes(), .channels = 1}, nullptr, bytes);
        }
        void sentinel_check(const Tensor& bytes, Tensor& unchanged, uint32_t seed) {
            if (bytes.dtype() != DataType::UInt8 || unchanged.dtype() != DataType::UInt32)
                throw std::invalid_argument("Sentinel check requires bytes and a uint32 flag");
            dispatch({.operation = 17, .seed = seed, .height = 1, .width = bytes.bytes(), .channels = 1}, &bytes, unchanged);
        }
        void convert(const Tensor& source, Tensor& destination, ImageConversion conversion,
                     size_t height, size_t width, size_t channels, const NormalPriorTransform& transform) {
            LfsHipSharedImage p{.operation = static_cast<uint32_t>(conversion),
                                .srgb = transform.srgb,
                                .flip_yz = transform.flip_yz,
                                .world_to_camera = transform.world_to_camera,
                                .height = height,
                                .width = width,
                                .channels = channels};
            std::copy_n(transform.w2c, 9, p.w2c);
            dispatch(p, &source, destination);
        }
        void rgba_split(const Tensor& rgba, Tensor& rgb, Tensor& alpha) {
            if (rgba.dtype() != DataType::UInt8 || rgba.ndim() != 3 || rgba.size(2) != 4 ||
                (rgb.dtype() != DataType::UInt8 && rgb.dtype() != DataType::Float32) || alpha.dtype() != DataType::Float32)
                throw std::invalid_argument("RGBA split requires byte HWC input and byte or float RGB output");
            dispatch({.operation = rgb.dtype() == DataType::UInt8 ? 19u : 18u,
                      .height = rgba.size(0),
                      .width = rgba.size(1),
                      .channels = 4},
                     &rgba, rgb, &alpha);
        }
        void mask(Tensor& value, MaskTransform transform, float threshold) {
            if (value.dtype() != DataType::Float32)
                throw std::invalid_argument("Mask transform requires float storage");
            dispatch({.operation = 20u + static_cast<uint32_t>(transform), .height = 1, .width = value.numel(), .channels = 1, .threshold = threshold}, nullptr, value);
        }
        Tensor resize(const Tensor& source, int height, int width, Resample mode, int kernel_size) {
            return image_backend().shared_image_resize(source, height, width, static_cast<unsigned>(mode), kernel_size, {});
        }
        Tensor undistort(const Tensor& source, const UndistortParams& params, bool mask) {
            return image_backend().image_undistort(source, params, mask, {});
        }
        const SharedImageOps image_ops{sentinel_fill, sentinel_check, convert, rgba_split, mask, resize, undistort};
    } // namespace
    const gpu_ops::SharedImageOps* hip_shared_image_ops() { return &image_ops; }
} // namespace lfs::core
