/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tensor_frosted_glass.hpp"

#include "core/gpu_kernel_module.hpp"
#include "core/tensor_backend.hpp"
#include "rmlui_frosted_glass_program.hpp"

#include <algorithm>
#include <array>
#include <span>
#include <string>
#include <utility>

namespace lfs::vis::gui {
    namespace {
        using Module = lfs::core::GpuKernelModule;
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        struct alignas(8) ResampleParameters {
            std::uint64_t source = 0;
            std::uint64_t destination = 0;
            std::uint32_t source_width = 0;
            std::uint32_t source_height = 0;
            std::uint32_t destination_width = 0;
            std::uint32_t destination_height = 0;
        };

        [[nodiscard]] lfs::Status invalidBackdrop(std::string detail) {
            return lfs::Status::failure(lfs::make_error({
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::Rendering,
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
    } // namespace

    std::vector<TensorFrostedGlassRect> frostedGlassClipRects(
        const std::span<const TensorFrostedGlassRegion> regions,
        const float target_width, const float target_height) {
        std::vector<TensorFrostedGlassRect> rectangles;
        rectangles.reserve(regions.size() * 2);
        const auto append = [&](const float left, const float top,
                                const float right, const float bottom) {
            if (right > left && bottom > top)
                rectangles.push_back({left, top, right, bottom});
        };
        for (const auto& region : regions) {
            const float left = std::clamp(region.x, 0.0f, target_width);
            const float top = std::clamp(region.y, 0.0f, target_height);
            const float right = std::clamp(region.x + region.width, 0.0f, target_width);
            const float bottom = std::clamp(region.y + region.height, 0.0f, target_height);
            const float radius = std::clamp(
                region.radius, 0.0f, 0.5f * std::min(right - left, bottom - top));
            if (radius < 1.0f) {
                append(left, top, right, bottom);
            } else {
                append(left + radius, top, right - radius, bottom);
                append(left, top + radius, right, bottom - radius);
            }
        }
        return rectangles;
    }

    struct TensorFrostedGlassBackdrop::Impl {
        std::unique_ptr<Module> program;
        Tensor primary;
        Tensor secondary;

        [[nodiscard]] lfs::Status resample(const Tensor& source, Tensor& destination) {
            ResampleParameters parameters{
                .source_width = static_cast<std::uint32_t>(source.size(1)),
                .source_height = static_cast<std::uint32_t>(source.size(0)),
                .destination_width = static_cast<std::uint32_t>(destination.size(1)),
                .destination_height = static_cast<std::uint32_t>(destination.size(0)),
            };
            const std::array bindings{
                Module::Binding{0, &source},
                Module::Binding{8, &destination, Module::Access::ReadWrite},
            };
            auto result = program->dispatch({
                .function = "resampleRgba8",
                .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                .groups = {Module::groups_for(destination.size(1), 8),
                           Module::groups_for(destination.size(0), 8), 1},
                .group = {8, 8, 1},
            });
            if (!result)
                return lfs::Status::failure(std::move(result).error());
            return {};
        }

        [[nodiscard]] lfs::Status update(const Tensor& source) {
            if (!source.is_valid() || source.device() != Device::GPU ||
                source.dtype() != DataType::UInt8 || source.ndim() != 3 ||
                source.size(2) != 4 || source.size(0) == 0 || source.size(1) == 0) {
                return invalidBackdrop("Frosted glass source must be a non-empty GPU UInt8 [H,W,4] tensor");
            }
            const auto backend = lfs::core::gpu_backend_of(source);
            if (!backend)
                return invalidBackdrop("Frosted glass source has no GPU backend identity");
            if (!program) {
                auto loaded = Module::load(rmlui_frosted_glass_program_entries(), *backend);
                if (!loaded)
                    return lfs::Status::failure(std::move(loaded).error());
                program = std::move(*loaded);
            }

            const std::size_t primary_height = std::max<std::size_t>(1, source.size(0) / 4);
            const std::size_t primary_width = std::max<std::size_t>(1, source.size(1) / 4);
            const std::size_t secondary_height = std::max<std::size_t>(1, source.size(0) / 8);
            const std::size_t secondary_width = std::max<std::size_t>(1, source.size(1) / 8);
            if (!primary.is_valid() || primary.size(0) != primary_height ||
                primary.size(1) != primary_width ||
                lfs::core::gpu_backend_of(primary) != backend) {
                primary = Tensor::empty({primary_height, primary_width, 4},
                                        Device::GPU, DataType::UInt8);
                secondary = Tensor::empty({secondary_height, secondary_width, 4},
                                          Device::GPU, DataType::UInt8);
            }
            if (auto status = resample(source, primary); !status)
                return status;
            if (auto status = resample(primary, secondary); !status)
                return status;
            return resample(secondary, primary);
        }
    };

    TensorFrostedGlassBackdrop::TensorFrostedGlassBackdrop()
        : impl_(std::make_unique<Impl>()) {}
    TensorFrostedGlassBackdrop::~TensorFrostedGlassBackdrop() = default;

    lfs::Status TensorFrostedGlassBackdrop::update(const Tensor& source) {
        return impl_->update(source);
    }

    const Tensor& TensorFrostedGlassBackdrop::image() const noexcept {
        return impl_->primary;
    }

    std::size_t TensorFrostedGlassBackdrop::bytes() const noexcept {
        return (impl_->primary.is_valid() ? impl_->primary.bytes() : 0) +
               (impl_->secondary.is_valid() ? impl_->secondary.bytes() : 0);
    }

    void TensorFrostedGlassBackdrop::reset() {
        impl_ = std::make_unique<Impl>();
    }

} // namespace lfs::vis::gui
