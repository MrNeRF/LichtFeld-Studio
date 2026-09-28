/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal SharedImageOps; core/cuda/shared_image_cuda.cpp is the reference. The
// kernels live in image.metal.

#include "metal_context.hpp"

#include "core/shared_image_ops.hpp"
#include "core/tensor.hpp"
#include "core/tensor_image.hpp"

#include <array>
#include <cmath>
#include <format>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lfs::core::internal {
    namespace {
        using namespace lfs::gpu_ops;

        API_AVAILABLE_BEGIN(macos(26.0))

        // Operand addresses for one dispatch; invalid tensors bind address 0.
        struct Operands {
            explicit Operands(std::initializer_list<const Tensor*> tensors) : context(metal::acquire_context()) {
                for (const Tensor* const tensor : tensors) {
                    if (!tensor->is_valid() || tensor->numel() == 0) {
                        addresses.push_back(0);
                        continue;
                    }
                    uses.push_back(storage_ref(*tensor));
                    const auto at = context->locate(uses.back());
                    addresses.push_back(at.address + at.offset);
                }
            }

            template <class Params>
            void dispatch(const char* const function, const Params& params, const size_t threads) {
                if (threads == 0)
                    return;
                if (threads > std::numeric_limits<uint32_t>::max())
                    throw std::invalid_argument(std::format("{} dispatch of {} threads exceeds uint32", function, threads));
                context->dispatch(uses, {.pipeline = context->pipeline(function),
                                         .buffers = {},
                                         .params = metal::param_bytes(params),
                                         .grid = MTLSizeMake(threads, 1, 1)});
            }

            std::shared_ptr<metal::Context> context;
            std::vector<StorageRef> uses;
            std::vector<uint64_t> addresses;
        };

        uint32_t checked_u32(const size_t value, const char* const what) {
            if (value > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument(std::format("{} of {} exceeds uint32", what, value));
            return static_cast<uint32_t>(value);
        }

        struct SentinelParams {
            uint64_t bytes, unchanged;
            uint32_t count, seed;
        };

        void sentinel_fill(Tensor& bytes, const uint32_t seed) {
            Operands operands{&bytes};
            const SentinelParams params{operands.addresses[0], 0, checked_u32(bytes.bytes(), "sentinel byte count"), seed};
            operands.dispatch("shared_image_sentinel_fill", params, bytes.bytes());
        }

        void sentinel_check(const Tensor& bytes, Tensor& unchanged, const uint32_t seed) {
            if (unchanged.dtype() != DataType::UInt32 || unchanged.numel() == 0)
                throw std::invalid_argument(std::format("sentinel check needs a uint32 flag, got {} of {} elements",
                                                        dtype_name(unchanged.dtype()), unchanged.numel()));
            Operands operands{&bytes, &unchanged};
            const SentinelParams params{operands.addresses[0], operands.addresses[1],
                                        checked_u32(bytes.bytes(), "sentinel byte count"), seed};
            operands.dispatch("shared_image_sentinel_check", params, bytes.bytes());
        }

        struct ConvertParams {
            uint64_t source, destination;
            uint32_t kind, height, width, channels;
            uint32_t srgb, flip_yz, world_to_camera, padding;
            float w2c[9];
        };

        struct ConversionShape {
            size_t source_bytes, destination_bytes, threads;
        };

        ConversionShape conversion_shape(const ImageConversion kind, const size_t h, const size_t w, const size_t c) {
            const size_t pixels = h * w, total = pixels * c;
            switch (kind) {
            case ImageConversion::U8HWCToF32CHW: return {total, total * 4, total};
            case ImageConversion::U16HWCToF32CHW: return {total * 2, total * 4, total};
            case ImageConversion::F32HWCToU16HWC: return {total * 4, total * 2, total};
            case ImageConversion::U16HWCToF32HWC: return {total * 2, total * 4, total};
            case ImageConversion::NormalCHWToJ2KHWC:
            case ImageConversion::J2KHWCToNormalCHW: return {pixels * 12, pixels * 12, pixels};
            case ImageConversion::NormalPriorU8: return {pixels * 3, pixels * 12, pixels};
            case ImageConversion::NormalPriorU16: return {pixels * 6, pixels * 12, pixels};
            case ImageConversion::U8HWCToU8CHW: return {total, total, total};
            case ImageConversion::U16HWCToU8CHW: return {total * 2, total, total};
            case ImageConversion::F32CHWToU8CHW: return {total * 4, total, total};
            case ImageConversion::U8HWToF32HW: return {pixels, pixels * 4, pixels};
            }
            throw std::invalid_argument(std::format("unknown image conversion {}", static_cast<int>(kind)));
        }

        void convert(const Tensor& source, Tensor& destination, const ImageConversion kind, const size_t h,
                     const size_t w, const size_t c, const NormalPriorTransform& transform) {
            const auto shape = conversion_shape(kind, h, w, c);
            if (source.bytes() < shape.source_bytes || destination.bytes() < shape.destination_bytes)
                throw std::invalid_argument(std::format(
                    "image conversion {} of {}x{}x{} needs {} source and {} destination bytes, got {} and {}",
                    static_cast<int>(kind), h, w, c, shape.source_bytes, shape.destination_bytes, source.bytes(),
                    destination.bytes()));
            Operands operands{&source, &destination};
            ConvertParams params{.source = operands.addresses[0],
                                 .destination = operands.addresses[1],
                                 .kind = static_cast<uint32_t>(kind),
                                 .height = checked_u32(h, "image height"),
                                 .width = checked_u32(w, "image width"),
                                 .channels = checked_u32(c, "image channels"),
                                 .srgb = transform.srgb ? 1u : 0u,
                                 .flip_yz = transform.flip_yz ? 1u : 0u,
                                 .world_to_camera = transform.world_to_camera ? 1u : 0u};
            std::copy_n(transform.w2c, 9, params.w2c);
            checked_u32(shape.threads, "image element count");
            operands.dispatch("shared_image_convert", params, shape.threads);
        }

        struct RgbaParams {
            uint64_t rgba, rgb, alpha;
            uint32_t pixels, rgb_bytes;
        };

        void rgba_split(const Tensor& rgba, Tensor& rgb, Tensor& alpha) {
            const size_t pixels = rgba.ndim() == 3 ? rgba.shape()[0] * rgba.shape()[1] : 0;
            const bool bytes = rgb.dtype() == DataType::UInt8;
            if (pixels == 0 || rgba.shape()[2] != 4 || rgba.dtype() != DataType::UInt8 ||
                rgb.bytes() < pixels * 3 * (bytes ? 1 : 4) || alpha.bytes() < pixels * 4)
                throw std::invalid_argument(std::format("RGBA split needs uint8 [H, W, 4] and matching outputs, got {} -> {} and {}",
                                                        rgba.shape().str(), rgb.shape().str(), alpha.shape().str()));
            Operands operands{&rgba, &rgb, &alpha};
            const RgbaParams params{operands.addresses[0], operands.addresses[1], operands.addresses[2],
                                    checked_u32(pixels, "RGBA pixel count"), bytes ? 1u : 0u};
            operands.dispatch("shared_image_rgba_split", params, pixels);
        }

        struct MaskParams {
            uint64_t mask;
            uint32_t count, threshold_mode;
            float threshold;
        };

        void mask(Tensor& value, const MaskTransform transform, const float threshold) {
            if (value.dtype() != DataType::Float32)
                throw std::invalid_argument(std::format("mask transform needs float32, got {}", dtype_name(value.dtype())));
            Operands operands{&value};
            const MaskParams params{operands.addresses[0], checked_u32(value.numel(), "mask element count"),
                                    transform == MaskTransform::Threshold ? 1u : 0u, threshold};
            operands.dispatch("shared_image_mask", params, value.numel());
        }

        struct CoefficientParams {
            uint64_t coefficients;
            int32_t input_size, output_size, kernel_size;
            uint32_t stride;
        };

        struct LanczosParams {
            uint64_t input, output, coef_x, coef_y;
            int32_t input_h, input_w, output_h, output_w, kernel_size;
            uint32_t stride_x, stride_y, layout, bytes;
        };

        // Taps of one output along an axis: the box of 2 * kernel_size * scale plus one.
        uint32_t coefficient_stride(const int input_size, const int output_size, const int kernel_size) {
            const long double support = 2.0L * kernel_size * input_size / output_size;
            const long double stride = std::ceil(support) + 1.0L;
            if (!std::isfinite(stride) ||
                stride * output_size > static_cast<long double>(std::numeric_limits<uint32_t>::max()))
                throw std::invalid_argument(std::format("Lanczos {} -> {} with kernel {} overflows its coefficient table",
                                                        input_size, output_size, kernel_size));
            return static_cast<uint32_t>(stride);
        }

        Tensor coefficients(const int input_size, const int output_size, const int kernel_size, const uint32_t stride) {
            auto table = Tensor::empty({static_cast<size_t>(output_size) * stride}, Device::GPU, DataType::Float32);
            Operands operands{&table};
            const CoefficientParams params{operands.addresses[0], input_size, output_size, kernel_size, stride};
            operands.dispatch("shared_image_lanczos_coefficients", params, static_cast<size_t>(output_size));
            return table;
        }

        // layout: 0 HWC RGB, 1 HW gray, 2 CHW RGB (see image.metal).
        Tensor lanczos(const Tensor& input, const int input_h, const int input_w, const int h, const int w,
                       const int kernel_size, const uint32_t layout) {
            if (h <= 0 || w <= 0 || kernel_size <= 0 || input_h <= 0 || input_w <= 0)
                throw std::invalid_argument(std::format("Lanczos resize of {}x{} to {}x{} with kernel {} needs positive sizes",
                                                        input_h, input_w, h, w, kernel_size));
            const GpuBackendScope scope(GpuBackend::Metal);
            const Tensor source = input.contiguous();
            const uint32_t stride_x = coefficient_stride(input_w, w, kernel_size);
            const uint32_t stride_y = coefficient_stride(input_h, h, kernel_size);
            const Tensor coef_x = coefficients(input_w, w, kernel_size, stride_x);
            const Tensor coef_y = coefficients(input_h, h, kernel_size, stride_y);
            const auto rows = static_cast<size_t>(h), columns = static_cast<size_t>(w);
            Tensor output = Tensor::empty(layout == 1 ? TensorShape{rows, columns} : TensorShape{3, rows, columns},
                                          Device::GPU, DataType::Float32);
            Operands operands{&source, &output, &coef_x, &coef_y};
            const LanczosParams params{operands.addresses[0], operands.addresses[1], operands.addresses[2],
                                       operands.addresses[3], input_h, input_w, h, w, kernel_size, stride_x,
                                       stride_y, layout, source.dtype() == DataType::UInt8 ? 1u : 0u};
            operands.dispatch("shared_image_lanczos_resample", params, rows * columns);
            return output;
        }

        int image_dim(const Tensor& tensor, const size_t dim) {
            if (tensor.shape()[dim] > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument(std::format("image dimension {} of {} exceeds int", dim, tensor.shape().str()));
            return static_cast<int>(tensor.shape()[dim]);
        }

        Tensor resize(const Tensor& source, const int h, const int w, const Resample mode, const int kernel_size) {
            const bool bytes_or_float = source.dtype() == DataType::UInt8 || source.dtype() == DataType::Float32;
            switch (mode) {
            case Resample::LanczosRGB:
                if (source.ndim() != 3 || source.shape()[2] != 3 || !bytes_or_float)
                    throw std::invalid_argument(std::format("Lanczos RGB needs uint8 or float32 [H, W, 3], got {} {}",
                                                            dtype_name(source.dtype()), source.shape().str()));
                return lanczos(source, image_dim(source, 0), image_dim(source, 1), h, w, kernel_size, 0);
            case Resample::LanczosGray:
                if (source.ndim() != 2 || !bytes_or_float)
                    throw std::invalid_argument(std::format("Lanczos gray needs uint8 or float32 [H, W], got {} {}",
                                                            dtype_name(source.dtype()), source.shape().str()));
                return lanczos(source, image_dim(source, 0), image_dim(source, 1), h, w, kernel_size, 1);
            case Resample::LanczosFloatCHW:
                if (source.ndim() != 3 || source.shape()[0] != 3 || source.dtype() != DataType::Float32)
                    throw std::invalid_argument(std::format("Lanczos CHW needs float32 [3, H, W], got {} {}",
                                                            dtype_name(source.dtype()), source.shape().str()));
                return lanczos(source, image_dim(source, 1), image_dim(source, 2), h, w, kernel_size, 2);
            case Resample::DepthPrior: return resize_image_prior_tensor(source, h, w, false);
            case Resample::NormalPrior: return resize_image_prior_tensor(source, h, w, true);
            }
            throw std::invalid_argument(std::format("unknown resample mode {}", static_cast<int>(mode)));
        }

        Tensor undistort(const Tensor& source, const UndistortParams& params, const bool is_mask) {
            return undistort_image_tensor(source, params, is_mask);
        }

        API_AVAILABLE_END
    } // namespace
} // namespace lfs::core::internal

namespace lfs::core {
    const gpu_ops::SharedImageOps* metal_shared_image_ops() {
        using namespace internal;
        if (@available(macOS 26.0, *)) {
            static const gpu_ops::SharedImageOps ops{sentinel_fill, sentinel_check, convert, rgba_split,
                                                     mask, resize, undistort};
            return &ops;
        }
        return nullptr;
    }
} // namespace lfs::core
