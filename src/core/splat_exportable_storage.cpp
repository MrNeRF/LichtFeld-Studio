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
#include <vector>

namespace lfs::core {

    namespace {

        constexpr std::size_t kFloatBytes = sizeof(float);
        // Exportable/viewer SH rest is IEEE f16 float4-swizzle (same topology as
        // fp32, half the bytes). Training FastGS loads half→float in registers;
        // the Vulkan projection shader does the same via f16tof32 on uint2 slots.
        // Headless stays on pad-dropped q16 outside this block (quant skip removed
        // only for non-exportable tensors).
        constexpr std::size_t kShNElementBytes = sizeof(std::uint16_t);
        constexpr std::size_t kRegionAlignment = 256;

        // True when `source` is a view into `block`'s VA range (CUDA-only or
        // Vulkan-interop alias of the same ExportableBlock). In that case
        // rebind must install views only — never copy_from the (possibly stale
        // offset) source into the new layout (ISS-025).
        [[nodiscard]] bool tensor_aliases_exportable_block(const Tensor& source,
                                                           const ExportableBlock& block) {
            if (!source.is_valid() || !source.is_external_storage()) {
                return false;
            }
            if (!block.device_ptr || block.size == 0 || source.numel() == 0) {
                return false;
            }
            const auto* base = static_cast<const char*>(block.device_ptr);
            const auto* end = base + block.size;
            // storage_ptr is the allocation base (non-materializing); for
            // external views it equals the region start baked into the tensor.
            const auto* ptr = static_cast<const char*>(source.storage_ptr());
            if (!ptr) {
                return false;
            }
            return ptr >= base && ptr < end;
        }

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
            const std::size_t shN_capacity_elems = sh_swizzled_float_count(capacity, rest_coeffs);

            const std::array<std::size_t, R::Count> raw_bytes{
                region_bytes_for(capacity, 3),         // Means {N,3}
                region_bytes_for(capacity, 3),         // Scaling {N,3}
                region_bytes_for(capacity, 4),         // Rotation {N,4}
                region_bytes_for(capacity, 1),         // Opacity {N,1}
                region_bytes_for(capacity, 3),         // Sh0 {N,1,3}
                shN_capacity_elems * kShNElementBytes, // ShN (IEEE f16)
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

        const std::size_t old_capacity = capacity_;
        const Layout prev_layout = compute_layout(capacity_, sh_degree_);
        const Layout grown_layout = compute_layout(new_capacity, sh_degree_);

        // ISS-025 stream fence: relocation memcpys use the default stream; drain
        // trainer/render work that may still be reading the block first.
        if (const auto err = cudaDeviceSynchronize(); err != cudaSuccess) {
            return std::unexpected(std::format(
                "SplatExportableStorage::grow: pre-relocation synchronize failed: {}",
                cudaGetErrorString(err)));
        }

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

        // Zero the full committed range so expanded slack starts clean, then
        // relocate live rows and mark slack non-renderable (opacity/rotation).
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

        // ISS-025 hardening (Analyst A): slack rows [old_capacity, new_capacity)
        // must not render if ever exposed — opacity raw → −∞ (sigmoid≈0),
        // identity quaternion (1,0,0,0). Zero-fill alone yields opacity=0.5 and
        // zero quat → NaN extents (half-screen splat blast radius).
        if (new_capacity > old_capacity) {
            const std::size_t n_slack = new_capacity - old_capacity;
            std::vector<float> opacity_host(n_slack, -std::numeric_limits<float>::infinity());
            std::vector<float> rotation_host(n_slack * 4, 0.0f);
            for (std::size_t i = 0; i < n_slack; ++i) {
                rotation_host[i * 4 + 0] = 1.0f; // w
            }
            void* opacity_dst = static_cast<char*>(block->device_ptr) + grown_layout.offsets[Opacity] +
                                old_capacity * kFloatBytes;
            void* rotation_dst = static_cast<char*>(block->device_ptr) + grown_layout.offsets[Rotation] +
                                 old_capacity * 4 * kFloatBytes;
            if (const auto err = cudaMemcpy(opacity_dst,
                                            opacity_host.data(),
                                            opacity_host.size() * kFloatBytes,
                                            cudaMemcpyHostToDevice);
                err != cudaSuccess) {
                return std::unexpected(std::format(
                    "SplatExportableStorage::grow: slack opacity init failed: {}",
                    cudaGetErrorString(err)));
            }
            if (const auto err = cudaMemcpy(rotation_dst,
                                            rotation_host.data(),
                                            rotation_host.size() * kFloatBytes,
                                            cudaMemcpyHostToDevice);
                err != cudaSuccess) {
                return std::unexpected(std::format(
                    "SplatExportableStorage::grow: slack rotation init failed: {}",
                    cudaGetErrorString(err)));
            }
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
                // ShN region is IEEE f16 regardless of the caller's requested dtype.
                if (region == ShN) {
                    dtype = DataType::Float16;
                }
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
                // ShN is IEEE f16 float4-swizzle: capacity is in half-elements
                // (same count as the historical "float count" topology).
                dtype = DataType::Float16;
                const std::size_t max_elems =
                    ctrl->region_bytes[ShN] / kShNElementBytes;
                clamped = std::min(capacity, max_elems);
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

            // ISS-025: same-block rebind (post-grow, pre-grow Vulkan drop) only
            // installs views at current region offsets. grow() already relocated
            // live rows; copying from stale pre-grow views destroys them.
            // Cross-allocator migrations (cuda.direct → exportable) still copy.
            const ExportableBlock& block_ref = *block;
            const auto install_param =
                [&](const Tensor& source, const TensorShape& shape, size_t cap,
                    std::string_view name) -> Tensor {
                const bool aliases = tensor_aliases_exportable_block(source, block_ref);
                Tensor source_cuda;
                DataType dtype = DataType::Float32;
                if (source.is_valid()) {
                    dtype = source.dtype();
                    if (!aliases) {
                        source_cuda =
                            source.device() == Device::CUDA ? source : source.cuda();
                        if (!source_cuda.is_contiguous()) {
                            source_cuda = source_cuda.contiguous();
                        }
                        dtype = source_cuda.dtype();
                    }
                }
                Tensor dst = allocator(shape, cap, dtype, name);
                dst.set_name(std::string{name});
                if (!aliases && source_cuda.is_valid() && source_cuda.numel() > 0) {
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
            // Preserve layout generation across the SplatData rebuild so
            // ensure_param_capacity's layout_changed signal stays monotonic.
            const std::uint64_t layout_gen = model.param_layout_generation();

            Tensor means = install_param(
                model.means_raw(), model.means_raw().shape(), capacity_, "SplatData.means");
            Tensor sh0 = install_param(
                model.sh0_raw(), model.sh0_raw().shape(), capacity_, "SplatData.sh0");
            Tensor scaling = install_param(
                model.scaling_raw(), model.scaling_raw().shape(), capacity_, "SplatData.scaling");
            Tensor rotation = install_param(
                model.rotation_raw(), model.rotation_raw().shape(), capacity_, "SplatData.rotation");
            Tensor opacity = install_param(
                model.opacity_raw(), model.opacity_raw().shape(), capacity_, "SplatData.opacity");

            Tensor shN;
            if (model.shN_raw().is_valid() && model.shN_raw().numel() > 0) {
                const auto layout_rest =
                    static_cast<std::uint32_t>(model.max_sh_coeffs_rest());
                const size_t shN_cap = sh_swizzled_float_count(capacity_, layout_rest);
                const size_t shN_logical = sh_swizzled_float_count(
                    static_cast<size_t>(model.size()), layout_rest);
                // Exportable ShN is always IEEE f16 float4-swizzle (half bytes of
                // the historical fp32 layout). Same-block rebind only re-views
                // (ISS-025). Cross-allocator install converts fp32 → f16.
                // q16 sources must be expanded to fp32 before rebind (training
                // ensure_shN_fp32_for_mutation); we refuse silent bitcast of codes.
                const Tensor& shN_src = model.shN_raw();
                const bool aliases = tensor_aliases_exportable_block(shN_src, block_ref);
                if (!aliases && model.shN_value_quantized()) {
                    return std::unexpected(
                        "SplatExportableStorage::rebindSplatData: shN is q16-quantized; "
                        "expand to fp32 before exportable install (IEEE f16 path)");
                }
                Tensor dst = allocator(
                    TensorShape({shN_logical}),
                    shN_cap,
                    DataType::Float16,
                    "SplatData.shN");
                dst.set_name("SplatData.shN");
                if (!aliases && shN_src.is_valid() && shN_src.numel() > 0) {
                    Tensor half_src = shN_src;
                    if (half_src.dtype() != DataType::Float16) {
                        half_src = half_src.to(DataType::Float16);
                    }
                    if (half_src.device() != Device::CUDA) {
                        half_src = half_src.cuda();
                    }
                    if (!half_src.is_contiguous()) {
                        half_src = half_src.contiguous();
                    }
                    dst.copy_from(half_src);
                }
                shN = std::move(dst);
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
            // capacity_ensure cannot be transferred here: rebind is often called
            // FROM inside the hook (growExportableForDensify), and moving the
            // active std::function would destroy the running frame. Callers
            // reinstall after rebind returns (TrainerManager, tests).
            model = std::move(rebound);
            // Restore + bump generation so densify re-fetch discipline sees the
            // layout change (ISS-025 post-grow re-fetch signal).
            while (model.param_layout_generation() <= layout_gen) {
                model.note_param_layout_changed();
            }
        } catch (const std::exception& e) {
            return std::unexpected(std::format(
                "SplatExportableStorage::rebindSplatData failed: {}", e.what()));
        }
        return {};
    }

} // namespace lfs::core
