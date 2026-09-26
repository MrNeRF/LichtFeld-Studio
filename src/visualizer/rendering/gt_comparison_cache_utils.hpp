/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "rendering/image_layout.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>

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
        if (src.device() == lfs::core::Device::GPU) {
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
