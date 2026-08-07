/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_exportable_storage.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/logger.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>

namespace lfs::core {

    namespace {

        constexpr std::size_t kFloatBytes = sizeof(float);
        constexpr std::size_t kRegionAlignment = 256;

        std::size_t align_up(std::size_t v, std::size_t a) {
            return ((v + a - 1) / a) * a;
        }

        std::size_t region_bytes_for(std::size_t capacity, std::size_t per_primitive_floats) {
            return capacity * per_primitive_floats * kFloatBytes;
        }

        struct Layout {
            std::array<std::size_t, SplatExportableStorage::Count> offsets{};
            std::array<std::size_t, SplatExportableStorage::Count> bytes{};
            std::size_t total = 0;
        };

        Layout compute_layout(std::size_t capacity, int sh_degree) {
            using R = SplatExportableStorage;
            const auto rest_coeffs =
                static_cast<std::uint32_t>(sh_rest_coefficients_for_degree(sh_degree));
            const std::size_t shN_capacity_floats = sh_swizzled_float_count(capacity, rest_coeffs);

            const std::array<std::size_t, R::Count> raw_bytes{
                region_bytes_for(capacity, 3),     // Means {N,3}
                region_bytes_for(capacity, 3),     // Scaling {N,3}
                region_bytes_for(capacity, 4),     // Rotation {N,4}
                region_bytes_for(capacity, 1),     // Opacity {N,1}
                region_bytes_for(capacity, 3),     // Sh0 {N,1,3}
                shN_capacity_floats * kFloatBytes, // ShN
            };

            Layout layout{};
            std::size_t cursor = 0;
            for (std::size_t i = 0; i < R::Count; ++i) {
                cursor = align_up(cursor, kRegionAlignment);
                layout.offsets[i] = cursor;
                layout.bytes[i] = raw_bytes[i];
                cursor += raw_bytes[i];
            }
            layout.total = cursor;
            return layout;
        }

        SplatExportableStorage::Region region_from_name(std::string_view name) {
            if (name == "SplatData.means")
                return SplatExportableStorage::Means;
            if (name == "SplatData.scaling")
                return SplatExportableStorage::Scaling;
            if (name == "SplatData.rotation")
                return SplatExportableStorage::Rotation;
            if (name == "SplatData.opacity")
                return SplatExportableStorage::Opacity;
            if (name == "SplatData.sh0")
                return SplatExportableStorage::Sh0;
            if (name == "SplatData.shN")
                return SplatExportableStorage::ShN;
            throw std::runtime_error(
                std::format("SplatExportableStorage: unknown allocator name '{}'", name));
        }

    } // namespace

    std::size_t SplatExportableStorage::layoutBytes(std::size_t capacity, int sh_degree) {
        if (capacity == 0) {
            return 0;
        }
        return compute_layout(capacity, sh_degree).total;
    }

    std::size_t SplatExportableStorage::growthCapacity(std::size_t live_or_needed,
                                                       std::size_t max_capacity) {
        if (live_or_needed == 0) {
            return max_capacity > 0 ? std::min<std::size_t>(1, max_capacity) : 1;
        }
        // 1.5× headroom (same growth factor as AdamOptimizer).
        std::size_t grown = live_or_needed;
        if (live_or_needed <= std::numeric_limits<std::size_t>::max() / 3 * 2) {
            grown = live_or_needed + live_or_needed / 2;
        }
        grown = std::max(grown, live_or_needed);
        if (max_capacity > 0) {
            grown = std::min(grown, max_capacity);
        }
        return grown;
    }

    void SplatExportableStorage::syncControl() const {
        if (!control_) {
            return;
        }
        control_->block = block;
        control_->region_offsets = region_offsets;
        control_->region_bytes = region_bytes;
        control_->capacity = capacity_;
        control_->sh_degree = sh_degree_;
        control_->generation = generation_;
    }

    std::expected<SplatExportableStorage, std::string>
    SplatExportableStorage::create(std::size_t capacity, int sh_degree, int device,
                                   std::size_t reserve_capacity) {
        if (capacity == 0) {
            return std::unexpected("SplatExportableStorage::create: capacity must be > 0");
        }

        const std::size_t reserve_gaussians =
            reserve_capacity > 0 ? std::max(reserve_capacity, capacity) : capacity;
        const Layout layout = compute_layout(capacity, sh_degree);
        const std::size_t reserve_bytes = compute_layout(reserve_gaussians, sh_degree).total;

        auto block_result =
            allocateExportableDeviceBlock(layout.total, device, /*track_splat_bytes=*/true, reserve_bytes);
        if (!block_result) {
            return std::unexpected(std::format(
                "SplatExportableStorage::create: backing-block allocation failed: {}",
                block_result.error()));
        }

        SplatExportableStorage out{};
        out.block = std::move(*block_result);
        out.region_offsets = layout.offsets;
        out.region_bytes = layout.bytes;
        out.capacity_ = capacity;
        out.reserved_capacity_ = reserve_gaussians;
        out.sh_degree_ = sh_degree;
        out.generation_ = 1;
        out.control_ = std::make_shared<Control>();
        out.syncControl();

        LOG_INFO("SplatExportableStorage: total={} MiB capacity={} reserve_capacity={} "
                 "sh_degree={} (means={}, scaling={}, rotation={}, opacity={}, sh0={}, shN={} MiB)",
                 out.block->size >> 20,
                 capacity,
                 reserve_gaussians,
                 sh_degree,
                 layout.bytes[Means] >> 20,
                 layout.bytes[Scaling] >> 20,
                 layout.bytes[Rotation] >> 20,
                 layout.bytes[Opacity] >> 20,
                 layout.bytes[Sh0] >> 20,
                 layout.bytes[ShN] >> 20);

        return out;
    }

    std::expected<bool, std::string> SplatExportableStorage::grow(std::size_t new_capacity) {
        if (!valid() || !control_) {
            return std::unexpected("SplatExportableStorage::grow: storage is not valid");
        }
        if (new_capacity == 0) {
            return std::unexpected("SplatExportableStorage::grow: new_capacity must be > 0");
        }
        if (new_capacity <= capacity_) {
            return false;
        }
        if (reserved_capacity_ > 0 && new_capacity > reserved_capacity_) {
            return std::unexpected(std::format(
                "SplatExportableStorage::grow: requested capacity {} exceeds reserved {}",
                new_capacity,
                reserved_capacity_));
        }

        const Layout prev_layout = compute_layout(capacity_, sh_degree_);
        const Layout grown_layout = compute_layout(new_capacity, sh_degree_);

        // Grow physical under the stable VA when the packed layout needs more bytes.
        if (grown_layout.total > block->size) {
            auto grew = growExportableDeviceBlock(block, grown_layout.total);
            if (!grew) {
                return std::unexpected(std::format(
                    "SplatExportableStorage::grow: block grow failed: {}", grew.error()));
            }
        }

        // Staging copy of the old packed SoA so region expansion can rewrite
        // offsets without overlapping in-place memmoves.
        void* staging = nullptr;
        const std::size_t old_total = prev_layout.total;
        if (old_total > 0) {
            if (const auto err = cudaMalloc(&staging, old_total); err != cudaSuccess) {
                return std::unexpected(std::format(
                    "SplatExportableStorage::grow: staging cudaMalloc failed: {}",
                    cudaGetErrorString(err)));
            }
            if (const auto err = cudaMemcpy(staging, block->device_ptr, old_total, cudaMemcpyDeviceToDevice);
                err != cudaSuccess) {
                cudaFree(staging);
                return std::unexpected(std::format(
                    "SplatExportableStorage::grow: staging cudaMemcpy failed: {}",
                    cudaGetErrorString(err)));
            }
        }

        // Zero the full committed range so expanded slack rows read as zeros.
        if (const auto err = cudaMemset(block->device_ptr, 0, grown_layout.total); err != cudaSuccess) {
            if (staging) {
                cudaFree(staging);
            }
            return std::unexpected(std::format(
                "SplatExportableStorage::grow: zero-fill failed: {}", cudaGetErrorString(err)));
        }

        if (staging) {
            for (std::size_t i = 0; i < Count; ++i) {
                const std::size_t copy_bytes = std::min(prev_layout.bytes[i], grown_layout.bytes[i]);
                if (copy_bytes == 0) {
                    continue;
                }
                void* dst = static_cast<char*>(block->device_ptr) + grown_layout.offsets[i];
                const void* src = static_cast<const char*>(staging) + prev_layout.offsets[i];
                if (const auto err = cudaMemcpy(dst, src, copy_bytes, cudaMemcpyDeviceToDevice);
                    err != cudaSuccess) {
                    cudaFree(staging);
                    return std::unexpected(std::format(
                        "SplatExportableStorage::grow: region {} relocate failed: {}",
                        i,
                        cudaGetErrorString(err)));
                }
            }
            cudaFree(staging);
        }

        if (const auto err = cudaDeviceSynchronize(); err != cudaSuccess) {
            return std::unexpected(std::format(
                "SplatExportableStorage::grow: synchronize failed: {}", cudaGetErrorString(err)));
        }

        region_offsets = grown_layout.offsets;
        region_bytes = grown_layout.bytes;
        capacity_ = new_capacity;
        ++generation_;
        syncControl();

        diagnostics::VramProfiler::instance().setExportableSplatBytes(block->size);

        LOG_INFO("SplatExportableStorage grew: capacity={} generation={} block={} MiB",
                 capacity_,
                 generation_,
                 block->size >> 20);
        return true;
    }

    SplatTensorAllocator SplatExportableStorage::make_allocator() const {
        auto ctrl = control_;
        if (!ctrl) {
            // Fallback for partially-constructed storage: capture by value.
            auto block_copy = block;
            auto offsets = region_offsets;
            const std::size_t cap = capacity_;
            return [block = std::move(block_copy), offsets, cap](TensorShape shape,
                                                                 std::size_t capacity,
                                                                 DataType dtype,
                                                                 std::string_view name) -> Tensor {
                const Region region = region_from_name(name);
                void* const data = static_cast<char*>(block->device_ptr) + offsets[region];
                std::shared_ptr<void> owner = block;
                const std::size_t clamped =
                    (region == ShN) ? capacity : std::min(capacity, cap > 0 ? cap : capacity);
                return Tensor::from_external_owner(data,
                                                   std::move(shape),
                                                   Device::CUDA,
                                                   dtype,
                                                   std::move(owner),
                                                   clamped,
                                                   /*stream=*/getCurrentCUDAStream(),
                                                   "splat.exportable");
            };
        }

        return [ctrl](TensorShape shape,
                      std::size_t capacity,
                      DataType dtype,
                      std::string_view name) -> Tensor {
            const Region region = region_from_name(name);
            void* const data =
                static_cast<char*>(ctrl->block->device_ptr) + ctrl->region_offsets[region];
            std::shared_ptr<void> owner = ctrl->block;
            // Clamp requested capacity to committed storage so max_cap callers
            // cannot claim more rows than the packed regions hold.
            std::size_t clamped = capacity;
            if (region == ShN) {
                const std::size_t max_floats =
                    ctrl->region_bytes[ShN] / kFloatBytes;
                clamped = std::min(capacity, max_floats);
            } else if (ctrl->capacity > 0) {
                clamped = std::min(capacity, ctrl->capacity);
            }
            return Tensor::from_external_owner(data,
                                               std::move(shape),
                                               Device::CUDA,
                                               dtype,
                                               std::move(owner),
                                               clamped,
                                               /*stream=*/getCurrentCUDAStream(),
                                               "splat.exportable");
        };
    }

    std::expected<void, std::string>
    SplatExportableStorage::rebindSplatData(SplatData& model,
                                            SplatTensorAllocator allocator) const {
        if (!valid()) {
            return std::unexpected("SplatExportableStorage::rebindSplatData: storage invalid");
        }
        if (capacity_ == 0) {
            return std::unexpected("SplatExportableStorage::rebindSplatData: capacity is 0");
        }

        try {
            if (!allocator) {
                allocator = make_allocator();
            }
            const size_t n = static_cast<size_t>(model.size());
            if (n > capacity_) {
                return std::unexpected(std::format(
                    "SplatExportableStorage::rebindSplatData: model size {} exceeds capacity {}",
                    n,
                    capacity_));
            }

            const auto copy_param =
                [&](const Tensor& source, const TensorShape& shape, size_t cap,
                    std::string_view name) -> Tensor {
                Tensor source_cuda =
                    source.device() == Device::CUDA ? source : source.cuda();
                if (!source_cuda.is_contiguous()) {
                    source_cuda = source_cuda.contiguous();
                }
                Tensor dst = allocator(shape, cap, source_cuda.dtype(), name);
                dst.set_name(std::string{name});
                if (source_cuda.numel() > 0) {
                    dst.copy_from(source_cuda);
                }
                return dst;
            };

            const int max_sh = model.get_max_sh_degree();
            const int active_sh = model.get_active_sh_degree();
            const float scene_scale = model.get_scene_scale();
            auto frozen_ranges = model.frozen_ranges();
            Tensor deleted = model.has_deleted_mask() ? model.deleted() : Tensor{};
            Tensor densification_info = model._densification_info;

            Tensor means =
                copy_param(model.means_raw(), model.means_raw().shape(), capacity_, "SplatData.means");
            Tensor sh0 =
                copy_param(model.sh0_raw(), model.sh0_raw().shape(), capacity_, "SplatData.sh0");
            Tensor scaling = copy_param(
                model.scaling_raw(), model.scaling_raw().shape(), capacity_, "SplatData.scaling");
            Tensor rotation = copy_param(
                model.rotation_raw(), model.rotation_raw().shape(), capacity_, "SplatData.rotation");
            Tensor opacity = copy_param(
                model.opacity_raw(), model.opacity_raw().shape(), capacity_, "SplatData.opacity");

            Tensor shN;
            if (model.shN_raw().is_valid() && model.shN_raw().numel() > 0) {
                const auto layout_rest =
                    static_cast<std::uint32_t>(model.max_sh_coeffs_rest());
                const size_t shN_cap = sh_swizzled_float_count(capacity_, layout_rest);
                shN = copy_param(
                    model.shN_raw(), model.shN_raw().shape(), shN_cap, "SplatData.shN");
            }

            SplatData rebound(max_sh,
                              std::move(means),
                              std::move(sh0),
                              std::move(shN),
                              std::move(scaling),
                              std::move(rotation),
                              std::move(opacity),
                              scene_scale,
                              SplatData::ShNLayout::Swizzled);
            rebound.set_active_sh_degree(active_sh);
            if (deleted.is_valid()) {
                rebound.deleted() = std::move(deleted);
                // Preserve soft-delete content across exportable rebind; force a
                // version bump and reconcile if densify grew N under the old mask.
                rebound.reconcile_deleted_mask();
                if (rebound.has_deleted_mask()) {
                    rebound.notify_deleted_mask_changed();
                }
            }
            if (densification_info.is_valid()) {
                rebound._densification_info = std::move(densification_info);
            }
            rebound.set_frozen_ranges(std::move(frozen_ranges));
            rebound.set_tensor_allocator(allocator);
            // Note: capacity_ensure is not preserved across rebind (private);
            // TrainerManager re-installs it after grow+rebind.
            model = std::move(rebound);
        } catch (const std::exception& e) {
            return std::unexpected(std::format(
                "SplatExportableStorage::rebindSplatData failed: {}", e.what()));
        }
        return {};
    }

} // namespace lfs::core
