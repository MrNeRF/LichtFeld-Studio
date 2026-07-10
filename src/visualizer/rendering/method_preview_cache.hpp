/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "rendering/image_layout.hpp"
#include "training/method_session.hpp"

#include <expected>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace lfs::vis {

    class MethodPreviewCache {
    public:
        enum class StoreResult {
            Fresh,
            CacheHit,
        };

        [[nodiscard]] std::expected<StoreResult, std::string> store(CameraOutputs outputs) {
            if (!outputs.rgb || !outputs.rgb->is_valid()) {
                return std::unexpected("method camera render returned no RGB tensor");
            }
            if (outputs.width <= 0 || outputs.height <= 0) {
                return std::unexpected(std::format(
                    "method camera render returned invalid size {}x{}", outputs.width, outputs.height));
            }
            if (outputs.rgb->device() != lfs::core::Device::CUDA) {
                return std::unexpected("method camera RGB tensor must be on CUDA");
            }
            if (outputs.rgb->dtype() != lfs::core::DataType::UInt8 &&
                outputs.rgb->dtype() != lfs::core::DataType::Float32) {
                return std::unexpected("method camera RGB tensor must be UInt8 or Float32");
            }
            if (outputs.rgb->ndim() != 3) {
                return std::unexpected("method camera RGB tensor must be three-dimensional");
            }

            const auto layout = lfs::rendering::detectImageLayout(*outputs.rgb);
            if (layout == lfs::rendering::ImageLayout::Unknown) {
                return std::unexpected("method camera RGB tensor has an unsupported channel layout");
            }
            const int tensor_width = lfs::rendering::imageWidth(*outputs.rgb, layout);
            const int tensor_height = lfs::rendering::imageHeight(*outputs.rgb, layout);
            if (tensor_width != outputs.width || tensor_height != outputs.height) {
                return std::unexpected(std::format(
                    "method camera RGB tensor is {}x{}, but CameraOutputs reports {}x{}",
                    tensor_width,
                    tensor_height,
                    outputs.width,
                    outputs.height));
            }

            const auto validate_optional_output =
                [&outputs](const std::shared_ptr<const lfs::core::Tensor>& tensor,
                           const std::string_view label) -> std::optional<std::string> {
                if (!tensor) {
                    return std::nullopt;
                }
                if (!tensor->is_valid() || tensor->device() != lfs::core::Device::CUDA) {
                    return std::format("method camera {} tensor must be valid and on CUDA", label);
                }
                int width = 0;
                int height = 0;
                if (tensor->ndim() == 2) {
                    height = static_cast<int>(tensor->size(0));
                    width = static_cast<int>(tensor->size(1));
                } else if (tensor->ndim() == 3) {
                    const auto tensor_layout = lfs::rendering::detectImageLayout(*tensor);
                    if (tensor_layout == lfs::rendering::ImageLayout::Unknown) {
                        return std::format("method camera {} tensor has an unsupported layout", label);
                    }
                    width = lfs::rendering::imageWidth(*tensor, tensor_layout);
                    height = lfs::rendering::imageHeight(*tensor, tensor_layout);
                } else {
                    return std::format("method camera {} tensor must be two- or three-dimensional", label);
                }
                if (width != outputs.width || height != outputs.height) {
                    return std::format(
                        "method camera {} tensor is {}x{}, but CameraOutputs reports {}x{}",
                        label,
                        width,
                        height,
                        outputs.width,
                        outputs.height);
                }
                return std::nullopt;
            };
            if (const auto error = validate_optional_output(outputs.depth, "depth")) {
                return std::unexpected(*error);
            }
            if (const auto error = validate_optional_output(outputs.alpha, "alpha")) {
                return std::unexpected(*error);
            }

            if (cached_ && cached_->content_generation == outputs.content_generation) {
                if (cached_->width != outputs.width || cached_->height != outputs.height ||
                    cached_->flip_y != outputs.flip_y) {
                    return std::unexpected(
                        "method camera render reused a content generation with different presentation metadata");
                }
                return StoreResult::CacheHit;
            }

            cached_ = std::move(outputs);
            return StoreResult::Fresh;
        }

        [[nodiscard]] const std::optional<CameraOutputs>& output() const noexcept {
            return cached_;
        }

        [[nodiscard]] bool hasOutput() const noexcept {
            return cached_.has_value();
        }

        void clear() {
            cached_.reset();
        }

    private:
        std::optional<CameraOutputs> cached_;
    };

} // namespace lfs::vis
