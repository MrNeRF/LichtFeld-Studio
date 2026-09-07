/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "rendering/image_layout.hpp"
#include "rendering/render_constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <utility>
#include <vector>

namespace lfs::vis::gt_comparison_detail {

    inline constexpr std::uint64_t SPLIT_RIGHT_GENERATION_BIT = 1ULL << 62;

    [[nodiscard]] inline std::size_t previewBytes(const glm::ivec2 size) {
        if (size.x <= 0 || size.y <= 0) {
            return 0;
        }
        return static_cast<std::size_t>(size.x) * static_cast<std::size_t>(size.y) * 3;
    }

    [[nodiscard]] inline bool prefetchFits(const std::size_t cache_bytes,
                                           const std::size_t current_bytes,
                                           const std::size_t neighbor_bytes,
                                           const std::size_t budget_bytes) {
        return current_bytes <= budget_bytes &&
               neighbor_bytes <= budget_bytes - current_bytes &&
               cache_bytes <= budget_bytes - current_bytes - neighbor_bytes;
    }

    [[nodiscard]] inline std::pair<float, float> robustDepthDisplayRange(
        const float* src,
        const std::size_t count) {
        std::vector<float> valid;
        valid.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            const float depth = src[i];
            if (std::isfinite(depth) && depth > 0.0f && depth < 1.0e9f) {
                valid.push_back(depth);
            }
        }
        if (valid.size() < 2) {
            return {0.0f, 0.0f};
        }

        constexpr float kLoQuantile = 0.02f;
        constexpr float kHiQuantile = 0.98f;
        const auto quantile = [&](const float q) {
            const auto n = static_cast<std::size_t>(q * static_cast<float>(valid.size() - 1));
            std::nth_element(valid.begin(), valid.begin() + n, valid.end());
            return valid[n];
        };
        return {quantile(kLoQuantile), quantile(kHiQuantile)};
    }

    [[nodiscard]] inline glm::vec3 depthPaletteForDisplay(float near_t) {
        near_t = std::clamp(near_t, 0.0f, 1.0f);
        const glm::vec3 far_0(0.050f, 0.040f, 0.150f);
        const glm::vec3 far_1(0.060f, 0.195f, 0.500f);
        const glm::vec3 mid_0(0.000f, 0.500f, 0.650f);
        const glm::vec3 mid_1(0.360f, 0.735f, 0.410f);
        const glm::vec3 near_0(0.965f, 0.820f, 0.300f);
        const glm::vec3 near_1(0.985f, 0.430f, 0.125f);

        if (near_t < 0.20f) {
            return glm::mix(far_0, far_1, glm::smoothstep(0.00f, 0.20f, near_t));
        }
        if (near_t < 0.43f) {
            return glm::mix(far_1, mid_0, glm::smoothstep(0.20f, 0.43f, near_t));
        }
        if (near_t < 0.67f) {
            return glm::mix(mid_0, mid_1, glm::smoothstep(0.43f, 0.67f, near_t));
        }
        if (near_t < 0.86f) {
            return glm::mix(mid_1, near_0, glm::smoothstep(0.67f, 0.86f, near_t));
        }
        return glm::mix(near_0, near_1, glm::smoothstep(0.86f, 1.00f, near_t));
    }

    [[nodiscard]] inline std::shared_ptr<lfs::core::Tensor> makeDepthDisplayTensor(
        const lfs::core::Tensor& depth,
        const lfs::rendering::DepthVisualizationMode depth_visualization_mode,
        const glm::vec3& background_color) {
        if (!depth.is_valid() || depth.ndim() != 2) {
            return {};
        }

        auto depth_cpu = depth.cpu().contiguous();
        const int height = static_cast<int>(depth_cpu.size(0));
        const int width = static_cast<int>(depth_cpu.size(1));
        if (width <= 0 || height <= 0) {
            return {};
        }

        const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
        std::vector<float> output(3 * pixel_count, 0.0f);
        const float* const src = depth_cpu.ptr<float>();
        if (!src) {
            return {};
        }

        const auto [range_lo, range_hi] = robustDepthDisplayRange(src, pixel_count);
        const float range_span = range_hi - range_lo;
        const bool grayscale =
            depth_visualization_mode == lfs::rendering::DepthVisualizationMode::Grayscale;
        for (std::size_t idx = 0; idx < pixel_count; ++idx) {
            const float value = src[idx];
            glm::vec3 color = background_color;
            if (std::isfinite(value) && value > 0.0f && value < 1.0e9f &&
                range_span > 1.0e-6f) {
                const float depth_t = std::clamp((value - range_lo) / range_span, 0.0f, 1.0f);
                const float near_t = 1.0f - depth_t;
                color = grayscale ? glm::vec3(near_t) : depthPaletteForDisplay(near_t);
            }
            output[idx] = color.r;
            output[pixel_count + idx] = color.g;
            output[2 * pixel_count + idx] = color.b;
        }

        auto tensor = lfs::core::Tensor::from_vector(
            output,
            {std::size_t{3}, static_cast<std::size_t>(height), static_cast<std::size_t>(width)},
            lfs::core::Device::CPU);
        return std::make_shared<lfs::core::Tensor>(std::move(tensor));
    }

    [[nodiscard]] inline std::shared_ptr<lfs::core::Tensor> makeNormalDisplayTensor(
        const lfs::core::Tensor& normal) {
        if (!normal.is_valid() || normal.ndim() != 3) {
            return {};
        }
        const auto layout = lfs::rendering::detectImageLayout(normal);
        if (layout == lfs::rendering::ImageLayout::Unknown ||
            lfs::rendering::imageChannels(normal, layout) < 3) {
            return {};
        }

        auto normal_cpu = normal.cpu().contiguous();
        if (layout == lfs::rendering::ImageLayout::HWC) {
            normal_cpu = normal_cpu.permute({2, 0, 1}).contiguous();
        }

        const int height = static_cast<int>(normal_cpu.size(1));
        const int width = static_cast<int>(normal_cpu.size(2));
        if (width <= 0 || height <= 0) {
            return {};
        }

        const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
        std::vector<float> output(3 * pixel_count, 0.5f);
        const float* const src = normal_cpu.ptr<float>();
        if (!src) {
            return {};
        }

        for (std::size_t idx = 0; idx < pixel_count; ++idx) {
            glm::vec3 normal_value(src[idx], src[pixel_count + idx], src[2 * pixel_count + idx]);
            const float length = glm::length(normal_value);
            if (std::isfinite(length) && length > 1.0e-6f) {
                normal_value /= length;
                const glm::vec3 color = glm::clamp(
                    normal_value * 0.5f + glm::vec3(0.5f),
                    glm::vec3(0.0f),
                    glm::vec3(1.0f));
                output[idx] = color.r;
                output[pixel_count + idx] = color.g;
                output[2 * pixel_count + idx] = color.b;
            }
        }

        auto tensor = lfs::core::Tensor::from_vector(
            output,
            {std::size_t{3}, static_cast<std::size_t>(height), static_cast<std::size_t>(width)},
            lfs::core::Device::CPU);
        return std::make_shared<lfs::core::Tensor>(std::move(tensor));
    }

    inline void updateSplitImageGeneration(const void* const source,
                                           const glm::ivec2 size,
                                           const void* const known_static_source,
                                           const glm::ivec2 known_static_size,
                                           const std::uint64_t frame_generation,
                                           std::uint64_t& generation) {
        if (source != nullptr && source == known_static_source && size == known_static_size) {
            generation |= SPLIT_RIGHT_GENERATION_BIT;
            return;
        }
        generation = frame_generation;
    }

    [[nodiscard]] inline std::shared_ptr<lfs::core::Tensor> convertDisplayTensorToUInt8(
        const std::shared_ptr<lfs::core::Tensor>& image) {
        if (!image || !image->is_valid() || image->ndim() != 3) {
            return {};
        }
        const auto layout = lfs::rendering::detectImageLayout(*image);
        if (layout == lfs::rendering::ImageLayout::Unknown) {
            return {};
        }
        lfs::core::Tensor src = *image;
        if (src.device() == lfs::core::Device::CUDA) {
            // The conversion below deliberately uses host-side loops. Tensor
            // dtype conversion and layout materialization on CPU are not a
            // reliable synchronization boundary for an asynchronously
            // produced CUDA tensor, whereas cpu() without an explicit stream
            // completes its device-to-host copy before returning.
            src = src.cpu();
        }
        if (image->device() == lfs::core::Device::CPU &&
            src.dtype() == lfs::core::DataType::UInt8 &&
            layout == lfs::rendering::ImageLayout::CHW) {
            return image;
        }
        if (src.dtype() != lfs::core::DataType::Float32 &&
            src.dtype() != lfs::core::DataType::UInt8) {
            return {};
        }
        const int channels = lfs::rendering::imageChannels(src, layout);
        const int height = lfs::rendering::imageHeight(src, layout);
        const int width = lfs::rendering::imageWidth(src, layout);
        if (channels < 3 || height <= 0 || width <= 0) {
            return {};
        }

        const std::size_t plane = static_cast<std::size_t>(height) * width;
        auto result = lfs::core::Tensor::empty(
            {3, static_cast<std::size_t>(height), static_cast<std::size_t>(width)},
            lfs::core::Device::CPU,
            lfs::core::DataType::UInt8);
        auto* const output = result.ptr<std::uint8_t>();

        const auto source_index = [&](const int channel,
                                      const int row,
                                      const int column) {
            if (layout == lfs::rendering::ImageLayout::CHW) {
                return static_cast<std::size_t>(channel) * src.stride(0) +
                       static_cast<std::size_t>(row) * src.stride(1) +
                       static_cast<std::size_t>(column) * src.stride(2);
            }
            return static_cast<std::size_t>(row) * src.stride(0) +
                   static_cast<std::size_t>(column) * src.stride(1) +
                   static_cast<std::size_t>(channel) * src.stride(2);
        };

        if (src.dtype() == lfs::core::DataType::Float32) {
            const float* const input = src.ptr<float>();
            for (int channel = 0; channel < 3; ++channel) {
                for (int row = 0; row < height; ++row) {
                    for (int column = 0; column < width; ++column) {
                        const float value = std::clamp(
                            input[source_index(channel, row, column)], 0.0f, 1.0f);
                        output[static_cast<std::size_t>(channel) * plane +
                               static_cast<std::size_t>(row) * width +
                               static_cast<std::size_t>(column)] =
                            static_cast<std::uint8_t>(value * 255.0f + 0.5f);
                    }
                }
            }
        } else {
            const std::uint8_t* const input = src.ptr<std::uint8_t>();
            for (int channel = 0; channel < 3; ++channel) {
                for (int row = 0; row < height; ++row) {
                    for (int column = 0; column < width; ++column) {
                        output[static_cast<std::size_t>(channel) * plane +
                               static_cast<std::size_t>(row) * width +
                               static_cast<std::size_t>(column)] =
                            input[source_index(channel, row, column)];
                    }
                }
            }
        }
        return std::make_shared<lfs::core::Tensor>(std::move(result));
    }

} // namespace lfs::vis::gt_comparison_detail
