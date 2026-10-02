/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "shared_image_cuda.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/tensor.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "io/cuda/image_format_kernels.cuh"

namespace lfs::core {
    namespace {
        using namespace lfs::gpu_ops;
        namespace kernels = lfs::io::cuda;

        void sentinel_fill(Tensor& bytes, uint32_t seed) {
            kernels::launch_fill_u8_sentinel(bytes.ptr<uint8_t>(), bytes.bytes(), seed, getCurrentCUDAStream());
        }
        void sentinel_check(const Tensor& bytes, Tensor& unchanged, uint32_t seed) {
            kernels::launch_flag_u8_sentinel_unchanged(bytes.ptr<uint8_t>(), bytes.bytes(), seed,
                                                       unchanged.ptr<uint32_t>(), getCurrentCUDAStream());
        }
        void convert(const Tensor& source, Tensor& destination, ImageConversion conversion,
                     size_t h, size_t w, size_t c, const NormalPriorTransform& transform) {
            const auto stream = getCurrentCUDAStream();
            const auto* u8 = static_cast<const uint8_t*>(source.data_ptr());
            const auto* u16 = static_cast<const uint16_t*>(source.data_ptr());
            const auto* f32 = static_cast<const float*>(source.data_ptr());
            auto* out_u8 = static_cast<uint8_t*>(destination.data_ptr());
            auto* out_u16 = static_cast<uint16_t*>(destination.data_ptr());
            auto* out_f32 = static_cast<float*>(destination.data_ptr());
            switch (conversion) {
            case ImageConversion::U8HWCToF32CHW: kernels::launch_uint8_hwc_to_float32_chw(u8, out_f32, h, w, c, stream); break;
            case ImageConversion::U16HWCToF32CHW: kernels::launch_uint16_hwc_to_float32_chw(u16, out_f32, h, w, c, stream); break;
            case ImageConversion::F32HWCToU16HWC: kernels::launch_float32_hwc_to_uint16_hwc(f32, out_u16, h, w, c, stream); break;
            case ImageConversion::U16HWCToF32HWC: kernels::launch_uint16_hwc_to_float32_hwc(u16, out_f32, h, w, c, stream); break;
            case ImageConversion::NormalCHWToJ2KHWC: kernels::launch_normal_chw_to_jpeg2k_hwc(f32, out_f32, h, w, stream); break;
            case ImageConversion::J2KHWCToNormalCHW: kernels::launch_jpeg2k_hwc_to_normal_chw(f32, out_f32, h, w, stream); break;
            case ImageConversion::NormalPriorU8: kernels::launch_normal_prior_u8_hwc_to_float32_chw(u8, out_f32, h, w, transform, stream); break;
            case ImageConversion::NormalPriorU16: kernels::launch_normal_prior_u16_hwc_to_float32_chw(u16, out_f32, h, w, transform, stream); break;
            case ImageConversion::U8HWCToU8CHW: kernels::launch_uint8_hwc_to_uint8_chw(u8, out_u8, h, w, c, stream); break;
            case ImageConversion::U16HWCToU8CHW: kernels::launch_uint16_hwc_to_uint8_chw(u16, out_u8, h, w, c, stream); break;
            case ImageConversion::F32CHWToU8CHW: kernels::launch_float32_chw_to_uint8_chw(f32, out_u8, h, w, c, stream); break;
            case ImageConversion::U8HWToF32HW: kernels::launch_uint8_hw_to_float32_hw(u8, out_f32, h, w, stream); break;
            }
        }
        void rgba_split(const Tensor& rgba, Tensor& rgb, Tensor& alpha) {
            const auto h = rgba.shape()[0], w = rgba.shape()[1];
            const auto stream = getCurrentCUDAStream();
            if (rgb.dtype() == DataType::UInt8) {
                kernels::launch_uint8_rgba_split_to_uint8_rgb_and_float32_alpha(rgba.ptr<uint8_t>(), rgb.ptr<uint8_t>(), alpha.ptr<float>(), h, w, stream);
            } else {
                kernels::launch_uint8_rgba_split_to_float32_rgb_and_alpha(rgba.ptr<uint8_t>(), rgb.ptr<float>(), alpha.ptr<float>(), h, w, stream);
            }
        }
        void mask(Tensor& value, MaskTransform transform, float threshold) {
            const auto h = value.shape()[0], w = value.shape()[1];
            const auto stream = getCurrentCUDAStream();
            if (transform == MaskTransform::Invert) {
                kernels::launch_mask_invert(value.ptr<float>(), h, w, stream);
            } else {
                kernels::launch_mask_threshold(value.ptr<float>(), h, w, threshold, stream);
            }
        }
        Tensor resize(const Tensor& source, int h, int w, Resample mode, int kernel_size) {
            const auto stream = getCurrentCUDAStream();
            switch (mode) {
            case Resample::LanczosRGB: return lanczos_resize(source, h, w, kernel_size, stream);
            case Resample::LanczosGray: return lanczos_resize_grayscale(source, h, w, kernel_size, stream);
            case Resample::LanczosFloatCHW: return lanczos_resize_float_chw(source, h, w, kernel_size, stream);
            case Resample::DepthPrior: return resize_depth_prior(source, h, w, stream);
            case Resample::NormalPrior: return resize_normal_prior(source, h, w, stream);
            }
            return {};
        }
        Tensor undistort(const Tensor& source, const UndistortParams& params, bool mask) {
            const auto stream = getCurrentCUDAStream();
            return mask ? undistort_mask(source, params, stream) : undistort_image(source, params, stream);
        }
        const SharedImageOps kCudaSharedImageOps{sentinel_fill, sentinel_check, convert, rgba_split, mask, resize, undistort};
    } // namespace
    const gpu_ops::SharedImageOps& cuda_shared_image_ops() { return kCudaSharedImageOps; }
} // namespace lfs::core
