/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/shared_image_ops.hpp"

#include "core/tensor.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_pipelines.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "core/tensor_image.hpp"
#include "core/vulkan_helpers.hpp"
#include "vk_shader_table.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace lfs::core {
    namespace {
        using namespace internal;
        using gpu_ops::ImageConversion;
        using gpu_ops::MaskTransform;
        using gpu_ops::NormalPriorTransform;
        using gpu_ops::Resample;

        struct Parameters {
            uint64_t source, destination, unchanged, coefficient_x, coefficient_y;
            uint32_t count, height, width, channels, kind, layout, bytes;
            uint32_t srgb, flip_yz, world_to_camera;
            uint32_t input_height, input_width, output_height, output_width;
            uint32_t kernel_size, stride_x, stride_y;
            float threshold;
            float w2c[9];
        };
        static_assert(sizeof(Parameters) == 152);
        struct Push {
            uint64_t parameters;
        };
        static_assert(sizeof(Push) == 8);

        struct ParameterBlock {
            VulkanContext* context = nullptr;
            StorageRef storage{};
            ~ParameterBlock() {
                if (context && storage.meta)
                    context->memory().deallocate(storage);
            }
        };

        uint32_t checked_u32(const size_t value, const char* const what) {
            if (value > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument(std::string(what) + " exceeds uint32 indexing");
            return static_cast<uint32_t>(value);
        }

        uint64_t address(const Tensor& tensor) {
            if (!tensor.is_valid() || tensor.numel() == 0)
                return 0;
            const auto storage = storage_ref(tensor);
            if (storage.backend != GpuBackend::Vulkan)
                throw std::invalid_argument("Vulkan SharedImage received storage from another backend");
            return vk::address(storage);
        }

        void dispatch(const uint32_t kind, Parameters parameters,
                      const std::span<const StorageRef> reads,
                      const std::span<const StorageRef> writes,
                      const size_t work_items) {
            if (work_items == 0)
                return;
            auto context = acquire_vulkan_context();
            auto parameter_block = std::make_shared<ParameterBlock>();
            parameter_block->context = context.get();
            parameter_block->storage = context->memory().allocate(sizeof(parameters), 16, {});
            context->memory().copy_host_to_device(CopyRequest{
                .src = raw_storage_ref(&parameters),
                .dst = parameter_block->storage,
                .bytes = sizeof(parameters),
                .synchronous = false});

            const Push push{vk::address(parameter_block->storage)};
            const std::array specialization{kind};
            const auto& pipeline = context->pipelines().specialized(
                "shared_image", sizeof(push), specialization);
            std::vector<StorageRef> inputs(reads.begin(), reads.end());
            inputs.push_back(parameter_block->storage);
            const uint32_t groups = vk::dispatch_groups(*context, work_items);
            context->recorders().record(
                inputs, writes,
                [&](const VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                    vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                       0, sizeof(push), &push);
                    vkCmdDispatch(command, groups, 1, 1);
                },
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, parameter_block);
        }

        void sentinel_fill(Tensor& bytes, const uint32_t seed) {
            if (!bytes.is_valid() || bytes.dtype() != DataType::UInt8 || bytes.bytes() == 0)
                throw std::invalid_argument("sentinel fill needs a non-empty uint8 tensor");
            const auto storage = storage_ref(bytes);
            Parameters p{};
            p.source = address(bytes);
            p.destination = address(bytes);
            p.count = checked_u32(bytes.bytes(), "sentinel byte count");
            p.kind = seed;
            const std::array writes{storage};
            dispatch(0, p, {}, writes, (bytes.bytes() + 3u) / 4u);
        }

        void sentinel_check(const Tensor& bytes, Tensor& unchanged, const uint32_t seed) {
            if (!bytes.is_valid() || bytes.dtype() != DataType::UInt8 || bytes.bytes() == 0 ||
                !unchanged.is_valid() || unchanged.dtype() != DataType::UInt32 || unchanged.numel() == 0)
                throw std::invalid_argument("sentinel check needs uint8 bytes and a uint32 flag");
            const auto input = storage_ref(bytes), output = storage_ref(unchanged);
            Parameters p{};
            p.source = address(bytes);
            p.unchanged = address(unchanged);
            p.count = checked_u32(bytes.bytes(), "sentinel byte count");
            p.kind = seed;
            const std::array reads{input};
            const std::array writes{output};
            dispatch(1, p, reads, writes, bytes.bytes());
        }

        size_t conversion_bytes(const ImageConversion kind, const size_t pixels, const size_t channels,
                                size_t& output_bytes, size_t& work_items) {
            const size_t total = pixels * channels;
            switch (kind) {
            case ImageConversion::U8HWCToF32CHW:
                output_bytes = total * 4;
                work_items = total;
                return total;
            case ImageConversion::U16HWCToF32CHW:
                output_bytes = total * 4;
                work_items = total;
                return total * 2;
            case ImageConversion::F32HWCToU16HWC:
                output_bytes = total * 2;
                work_items = total;
                return total * 4;
            case ImageConversion::U16HWCToF32HWC:
                output_bytes = total * 4;
                work_items = total;
                return total * 2;
            case ImageConversion::NormalCHWToJ2KHWC:
            case ImageConversion::J2KHWCToNormalCHW:
                output_bytes = pixels * 12;
                work_items = pixels;
                return pixels * 12;
            case ImageConversion::NormalPriorU8:
                output_bytes = pixels * 12;
                work_items = pixels;
                return pixels * 3;
            case ImageConversion::NormalPriorU16:
                output_bytes = pixels * 12;
                work_items = pixels;
                return pixels * 6;
            case ImageConversion::U8HWCToU8CHW:
                output_bytes = total;
                work_items = total;
                return total;
            case ImageConversion::U16HWCToU8CHW:
                output_bytes = total;
                work_items = total;
                return total * 2;
            case ImageConversion::F32CHWToU8CHW:
                output_bytes = total;
                work_items = total;
                return total * 4;
            case ImageConversion::U8HWToF32HW:
                output_bytes = pixels * 4;
                work_items = pixels;
                return pixels;
            }
            throw std::invalid_argument("unknown image conversion kind");
        }

        void convert(const Tensor& source, Tensor& destination, const ImageConversion conversion,
                     const size_t height, const size_t width, const size_t channels,
                     const NormalPriorTransform& transform) {
            if (height == 0 || width == 0 || channels == 0 || height > SIZE_MAX / width)
                throw std::invalid_argument("image conversion dimensions must be positive and fit size_t");
            const size_t pixels = height * width;
            size_t destination_bytes = 0, work_items = 0;
            const size_t source_bytes = conversion_bytes(conversion, pixels, channels,
                                                         destination_bytes, work_items);
            if (source.bytes() < source_bytes || destination.bytes() < destination_bytes)
                throw std::invalid_argument("image conversion buffers are smaller than the requested layout");
            Parameters p{};
            p.source = address(source);
            p.destination = address(destination);
            p.height = checked_u32(height, "image height");
            p.width = checked_u32(width, "image width");
            p.channels = checked_u32(channels, "image channels");
            p.kind = static_cast<uint32_t>(conversion);
            p.srgb = transform.srgb ? 1u : 0u;
            p.flip_yz = transform.flip_yz ? 1u : 0u;
            p.world_to_camera = transform.world_to_camera ? 1u : 0u;
            std::copy_n(transform.w2c, 9, p.w2c);
            const auto input = storage_ref(source), output = storage_ref(destination);
            const std::array reads{input};
            const std::array writes{output};
            dispatch(2, p, reads, writes, work_items);
        }

        void rgba_split(const Tensor& rgba, Tensor& rgb, Tensor& alpha) {
            if (rgba.dtype() != DataType::UInt8 || rgba.ndim() != 3 || rgba.size(2) != 4)
                throw std::invalid_argument("RGBA split needs uint8 [H,W,4]");
            const size_t height = rgba.size(0), width = rgba.size(1), pixels = height * width;
            const bool byte_rgb = rgb.dtype() == DataType::UInt8;
            if (rgb.bytes() < pixels * 3 * (byte_rgb ? 1 : sizeof(float)) || alpha.bytes() < pixels * sizeof(float))
                throw std::invalid_argument("RGBA split outputs have insufficient storage");
            Parameters p{};
            p.source = address(rgba);
            p.destination = address(rgb);
            p.unchanged = address(alpha);
            p.height = checked_u32(height, "image height");
            p.width = checked_u32(width, "image width");
            p.count = checked_u32(pixels, "RGBA pixel count");
            p.bytes = byte_rgb ? 1u : 0u;
            const std::array reads{storage_ref(rgba)};
            const std::array writes{storage_ref(rgb), storage_ref(alpha)};
            dispatch(3, p, reads, writes, pixels);
        }

        void mask(Tensor& value, const MaskTransform transform, const float threshold) {
            if (value.dtype() != DataType::Float32 || (value.ndim() != 2 && value.ndim() != 3))
                throw std::invalid_argument("mask transform needs float32 image data");
            Parameters p{};
            p.destination = address(value);
            p.count = checked_u32(value.numel(), "mask element count");
            p.kind = transform == MaskTransform::Threshold ? 1u : 0u;
            p.threshold = threshold;
            const std::array writes{storage_ref(value)};
            dispatch(4, p, {}, writes, value.numel());
        }

        uint32_t lanczos_stride(const int input, const int output, const int kernel) {
            const long double support = 2.0L * kernel * input / output;
            const long double stride = std::ceil(support) + 1.0L;
            if (!std::isfinite(stride) || stride > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument("Lanczos coefficient table size exceeds uint32 indexing");
            return static_cast<uint32_t>(stride);
        }

        Tensor lanczos_coefficients(const int input, const int output, const int kernel,
                                    const uint32_t stride) {
            std::vector<float> values(static_cast<size_t>(output) * stride, 0.0f);
            for (int out = 0; out < output; ++out) {
                const float scale = static_cast<float>(input) / static_cast<float>(output);
                const float center = (static_cast<float>(out) + 0.5f) * scale;
                const int lo = std::max(static_cast<int>(center - static_cast<float>(kernel) * scale + 0.5f), 0);
                const int hi = std::min(static_cast<int>(center + static_cast<float>(kernel) * scale + 0.5f), input);
                float sum = 0.0f;
                for (int at = lo; at < hi; ++at) {
                    const float x = (static_cast<float>(at) + 0.5f - center) / scale;
                    const float value = std::abs(x) < 1e-12f ? 1.0f : (x <= -kernel || x >= kernel ? 0.0f : (std::sin(3.14159265358979323846f * x) / (3.14159265358979323846f * x)) * (std::sin(3.14159265358979323846f * (x / kernel)) / (3.14159265358979323846f * (x / kernel))));
                    values[static_cast<size_t>(out) * stride + at - lo] = value;
                    sum += value;
                }
                for (int at = lo; at < hi; ++at)
                    values[static_cast<size_t>(out) * stride + at - lo] /= sum;
            }
            GpuBackendScope scope(GpuBackend::Vulkan);
            return Tensor::from_vector(values, {values.size()}, Device::GPU);
        }

        int tensor_dimension(const Tensor& source, const size_t dimension) {
            return static_cast<int>(checked_u32(source.size(dimension), "image dimension"));
        }

        Tensor resize(const Tensor& source, const int height, const int width,
                      const Resample mode, const int kernel_size) {
            if (mode == Resample::DepthPrior)
                return internal::resize_image_prior_tensor(source, height, width, false);
            if (mode == Resample::NormalPrior)
                return internal::resize_image_prior_tensor(source, height, width, true);
            const GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor input = source.contiguous();
            const bool gray = mode == Resample::LanczosGray;
            const uint32_t layout = mode == Resample::LanczosRGB ? 0u : gray ? 1u
                                                                             : 2u;
            if ((mode == Resample::LanczosRGB && (input.ndim() != 3 || (input.size(2) != 3 && input.size(2) != 4))) ||
                (gray && input.ndim() != 2) ||
                (mode == Resample::LanczosFloatCHW && (input.ndim() != 3 || (input.size(0) != 3 && input.size(0) != 4) || input.dtype() != DataType::Float32)) ||
                (input.dtype() != DataType::UInt8 && input.dtype() != DataType::Float32) ||
                height <= 0 || width <= 0 || kernel_size <= 0)
                throw std::invalid_argument("Lanczos resize received an unsupported shape, dtype, or dimension");
            const int input_h = tensor_dimension(input, gray ? 0 : layout == 2 ? 1
                                                                               : 0);
            const int input_w = tensor_dimension(input, gray ? 1 : layout == 2 ? 2
                                                                               : 1);
            const uint32_t stride_x = lanczos_stride(input_w, width, kernel_size);
            const uint32_t stride_y = lanczos_stride(input_h, height, kernel_size);
            const Tensor coef_x = lanczos_coefficients(input_w, width, kernel_size, stride_x);
            const Tensor coef_y = lanczos_coefficients(input_h, height, kernel_size, stride_y);
            const size_t channels = gray ? 1 : input.size(layout == 2 ? 0 : 2);
            Tensor output = Tensor::empty(gray ? TensorShape{size_t(height), size_t(width)}
                                               : TensorShape{channels, size_t(height), size_t(width)},
                                          Device::GPU, DataType::Float32);
            Parameters p{};
            p.source = address(input);
            p.destination = address(output);
            p.coefficient_x = address(coef_x);
            p.coefficient_y = address(coef_y);
            p.input_height = static_cast<uint32_t>(input_h);
            p.input_width = static_cast<uint32_t>(input_w);
            p.output_height = static_cast<uint32_t>(height);
            p.output_width = static_cast<uint32_t>(width);
            p.kernel_size = static_cast<uint32_t>(kernel_size);
            p.stride_x = stride_x;
            p.stride_y = stride_y;
            p.layout = layout;
            p.channels = static_cast<uint32_t>(channels);
            p.bytes = input.dtype() == DataType::UInt8 ? 1u : 0u;
            const std::array reads{storage_ref(input), storage_ref(coef_x), storage_ref(coef_y)};
            const std::array writes{storage_ref(output)};
            dispatch(5, p, reads, writes, static_cast<size_t>(height) * width);
            return output;
        }

        Tensor undistort(const Tensor& source, const UndistortParams& params, const bool mask) {
            return internal::undistort_image_tensor(source, params, mask);
        }

        const gpu_ops::SharedImageOps kVulkanSharedImageOps{
            sentinel_fill, sentinel_check, convert, rgba_split, mask, resize, undistort};
    } // namespace

    const gpu_ops::SharedImageOps* vulkan_shared_image_ops() {
        return &kVulkanSharedImageOps;
    }
} // namespace lfs::core
