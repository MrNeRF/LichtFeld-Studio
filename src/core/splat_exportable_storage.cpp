/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_exportable_storage.hpp"

#include "core/checked_arithmetic.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/logger.hpp"
#include "core/sh_value_quant.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <format>
#include <limits>
#include <new>
#include <vector>

namespace lfs::core {

    namespace {

        constexpr std::size_t kFloatBytes = sizeof(float);
        // Exportable training SH rest is pad-dropped q16 (uint16 cells) with
        // per-256-splat float2 bounds. Same format as headless; the Vulkan
        // projection shader dequants in registers (LFS_SHN_Q16). Standalone
        // PLY/SOG viewing keeps the separate IEEE f16 resident path.
        constexpr std::size_t kShNElementBytes = sizeof(std::uint16_t);
        constexpr std::size_t kRegionAlignment = 256;
        constexpr std::size_t kNoInjectedRelocateFailure =
            std::numeric_limits<std::size_t>::max();
        std::atomic<std::size_t> g_relocate_failure_for_testing{
            kNoInjectedRelocateFailure};

        class CudaStagingBuffer {
        public:
            CudaStagingBuffer() = default;
            CudaStagingBuffer(const CudaStagingBuffer&) = delete;
            CudaStagingBuffer& operator=(const CudaStagingBuffer&) = delete;

            ~CudaStagingBuffer() {
                if (ptr_) {
                    if (const auto err = cudaFree(ptr_); err != cudaSuccess) {
                        LOG_ERROR("SplatExportableStorage: staging cudaFree failed: {}",
                                  cudaGetErrorString(err));
                    }
                }
            }

            void reset(void* ptr) noexcept { ptr_ = ptr; }
            [[nodiscard]] void* get() const noexcept { return ptr_; }

        private:
            void* ptr_ = nullptr;
        };

        [[nodiscard]] bool consume_relocate_failure_for_testing(const std::size_t region) {
            std::size_t expected = region;
            return g_relocate_failure_for_testing.compare_exchange_strong(
                expected,
                kNoInjectedRelocateFailure,
                std::memory_order_acq_rel,
                std::memory_order_acquire);
        }

        // True when `source` is a view into `block`'s VA range (CUDA-only or
        // Vulkan-interop alias of the same ExportableBlock). In that case
        // rebind must install views only — never copy_from the (possibly stale
        // offset) source into the new layout.
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
            const std::size_t elements = checked_product(
                capacity, per_primitive_floats, "exportable splat region element count");
            return checked_product(
                elements, kFloatBytes, "exportable splat region byte count");
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
            const std::size_t shN_u16_cells =
                sh_value_quant::sh_value_u16_count(capacity, rest_coeffs);
            const std::size_t bounds_float2s = sh_value_quant::n_bounds_for_prims(capacity);
            // bounds region: float2 per 256-splat block → 2 floats each
            const std::size_t bounds_bytes = bounds_float2s * 2u * kFloatBytes;

            const std::array<std::size_t, R::Count> raw_bytes{
                region_bytes_for(capacity, 3),    // Means {N,3}
                region_bytes_for(capacity, 3),    // Scaling {N,3}
                region_bytes_for(capacity, 4),    // Rotation {N,4}
                region_bytes_for(capacity, 1),    // Opacity {N,1}
                region_bytes_for(capacity, 3),    // Sh0 {N,1,3}
                shN_u16_cells * kShNElementBytes, // ShN (pad-dropped q16)
                bounds_bytes,                     // ShNBounds (float2 / 256)
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
            if (name == "SplatData.shN_value_bounds")
                return SplatExportableStorage::ShNBounds;
            throw std::runtime_error(
                std::format("SplatExportableStorage: unknown allocator name '{}'", name));
        }

    } // namespace

    void set_splat_exportable_relocate_failure_for_testing(
        const std::optional<std::size_t> region) noexcept {
        g_relocate_failure_for_testing.store(
            region.value_or(kNoInjectedRelocateFailure), std::memory_order_release);
    }

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
        control_->generation = generation_;
        control_->poisoned = poisoned_;
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
                 "sh_degree={} (means={}, scaling={}, rotation={}, opacity={}, sh0={}, "
                 "shN(q16)={}, shN_bounds={} MiB)",
                 out.block->size >> 20,
                 capacity,
                 reserve_gaussians,
                 sh_degree,
                 layout.bytes[Means] >> 20,
                 layout.bytes[Scaling] >> 20,
                 layout.bytes[Rotation] >> 20,
                 layout.bytes[Opacity] >> 20,
                 layout.bytes[Sh0] >> 20,
                 layout.bytes[ShN] >> 20,
                 layout.bytes[ShNBounds] >> 20);

        return out;
    }

    std::expected<bool, std::string> SplatExportableStorage::grow(std::size_t new_capacity) {
        if (poisoned_) {
            return std::unexpected(
                "SplatExportableStorage::grow: storage is poisoned by a failed rollback");
        }
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

        // stream fence: relocation memcpys use the default stream; drain
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
        CudaStagingBuffer staging;
        const std::size_t old_total = prev_layout.total;
        if (old_total > 0) {
            void* staging_ptr = nullptr;
            if (const auto err = cudaMalloc(&staging_ptr, old_total); err != cudaSuccess) {
                return std::unexpected(std::format(
                    "SplatExportableStorage::grow: staging cudaMalloc failed: {}",
                    cudaGetErrorString(err)));
            }
            staging.reset(staging_ptr);
            if (const auto err = cudaMemcpy(staging.get(), block->device_ptr, old_total,
                                            cudaMemcpyDeviceToDevice);
                err != cudaSuccess) {
                return std::unexpected(std::format(
                    "SplatExportableStorage::grow: staging cudaMemcpy failed: {}",
                    cudaGetErrorString(err)));
            }
        }

        // Prepare all fallible host allocations before the first destructive
        // device write. The device staging snapshot remains live through commit.
        const std::size_t n_slack = new_capacity - old_capacity;
        std::vector<float> opacity_host;
        std::vector<float> rotation_host;
        try {
            opacity_host.assign(n_slack, -std::numeric_limits<float>::infinity());
            rotation_host.assign(n_slack * 4, 0.0f);
            for (std::size_t i = 0; i < n_slack; ++i) {
                rotation_host[i * 4] = 1.0f; // identity quaternion w
            }
        } catch (const std::bad_alloc&) {
            return std::unexpected(
                "SplatExportableStorage::grow: slack host staging allocation failed");
        }

        const auto fail_after_destructive_write =
            [&](std::string failure) -> std::expected<bool, std::string> {
            const auto restore_err =
                staging.get() && old_total > 0
                    ? cudaMemcpy(block->device_ptr, staging.get(), old_total,
                                 cudaMemcpyDeviceToDevice)
                    : cudaErrorInvalidValue;
            const auto restore_sync_err =
                restore_err == cudaSuccess ? cudaDeviceSynchronize() : restore_err;
            if (restore_err == cudaSuccess && restore_sync_err == cudaSuccess) {
                return std::unexpected(std::format("{}; previous layout restored", failure));
            }

            poisoned_ = true;
            syncControl();
            return std::unexpected(std::format(
                "{}; rollback failed: copy={} sync={}; storage poisoned",
                failure,
                cudaGetErrorString(restore_err),
                cudaGetErrorString(restore_sync_err)));
        };

        // Zero the full committed range so expanded slack starts clean, then
        // relocate live rows and mark slack non-renderable (opacity/rotation).
        if (const auto err = cudaMemset(block->device_ptr, 0, grown_layout.total); err != cudaSuccess) {
            return fail_after_destructive_write(std::format(
                "SplatExportableStorage::grow: zero-fill failed: {}",
                cudaGetErrorString(err)));
        }

        if (staging.get()) {
            for (std::size_t i = 0; i < Count; ++i) {
                const std::size_t copy_bytes = std::min(prev_layout.bytes[i], grown_layout.bytes[i]);
                if (copy_bytes == 0) {
                    continue;
                }
                void* dst = static_cast<char*>(block->device_ptr) + grown_layout.offsets[i];
                const void* src =
                    static_cast<const char*>(staging.get()) + prev_layout.offsets[i];
                if (const auto err = cudaMemcpy(dst, src, copy_bytes, cudaMemcpyDeviceToDevice);
                    err != cudaSuccess) {
                    return fail_after_destructive_write(std::format(
                        "SplatExportableStorage::grow: region {} relocate failed: {}",
                        i,
                        cudaGetErrorString(err)));
                }
                if (consume_relocate_failure_for_testing(i)) {
                    return fail_after_destructive_write(std::format(
                        "SplatExportableStorage::grow: injected region {} relocate failure",
                        i));
                }
            }
        }

        // Slack rows [old_capacity, new_capacity) must not render if exposed:
        // opacity raw → −∞ (sigmoid≈0),
        // identity quaternion (1,0,0,0). Zero-fill alone yields opacity=0.5 and
        // zero quat → NaN extents (half-screen splat blast radius).
        if (new_capacity > old_capacity) {
            void* opacity_dst = static_cast<char*>(block->device_ptr) + grown_layout.offsets[Opacity] +
                                old_capacity * kFloatBytes;
            void* rotation_dst = static_cast<char*>(block->device_ptr) + grown_layout.offsets[Rotation] +
                                 old_capacity * 4 * kFloatBytes;
            if (const auto err = cudaMemcpy(opacity_dst,
                                            opacity_host.data(),
                                            opacity_host.size() * kFloatBytes,
                                            cudaMemcpyHostToDevice);
                err != cudaSuccess) {
                return fail_after_destructive_write(std::format(
                    "SplatExportableStorage::grow: slack opacity init failed: {}",
                    cudaGetErrorString(err)));
            }
            if (const auto err = cudaMemcpy(rotation_dst,
                                            rotation_host.data(),
                                            rotation_host.size() * kFloatBytes,
                                            cudaMemcpyHostToDevice);
                err != cudaSuccess) {
                return fail_after_destructive_write(std::format(
                    "SplatExportableStorage::grow: slack rotation init failed: {}",
                    cudaGetErrorString(err)));
            }
        }

        if (const auto err = cudaDeviceSynchronize(); err != cudaSuccess) {
            return fail_after_destructive_write(std::format(
                "SplatExportableStorage::grow: synchronize failed: {}",
                cudaGetErrorString(err)));
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

    void* SplatExportableStorage::live_region_ptr(Region region) const {
        if (!control_) {
            return nullptr;
        }
        return control_->region_ptr(region);
    }

    namespace {

        [[nodiscard]] std::size_t element_bytes_for_dtype(DataType dtype) {
            const std::size_t n = dtype_size(dtype);
            if (n == 0) {
                throw std::runtime_error(
                    "SplatExportableStorage allocator: invalid dtype");
            }
            return n;
        }

        // requested logical shape must fit inside the packed region. Capacity
        // is clamped separately; shape overrun is a silent-OOB-by-construction
        // footgun (Tensor::from_external_owner will otherwise happily view past
        // the region / committed frontier).
        void validate_exportable_shape_fits_region(std::string_view name,
                                                   const TensorShape& shape,
                                                   DataType dtype,
                                                   std::size_t region_bytes) {
            const std::size_t elems = shape.elements();
            const std::size_t elem_bytes = element_bytes_for_dtype(dtype);
            if (elems > 0 && elem_bytes > 0 &&
                elems > (std::numeric_limits<std::size_t>::max() / elem_bytes)) {
                throw std::runtime_error(std::format(
                    "SplatExportableStorage allocator: shape byte overflow for '{}' "
                    "(elems={}, elem_bytes={})",
                    name,
                    elems,
                    elem_bytes));
            }
            const std::size_t shape_bytes = elems * elem_bytes;
            if (shape_bytes > region_bytes) {
                throw std::runtime_error(std::format(
                    "SplatExportableStorage allocator: shape for '{}' needs {} bytes "
                    "but region only holds {}",
                    name,
                    shape_bytes,
                    region_bytes));
            }
        }

        [[nodiscard]] Tensor make_exportable_view(
            const std::shared_ptr<SplatExportableStorage::Control>& ctrl,
            SplatExportableStorage::Region region,
            TensorShape shape,
            std::size_t capacity,
            DataType dtype,
            std::string_view name,
            std::string external_kind,
            std::shared_ptr<void> owner,
            cudaStream_t stream) {
            if (!ctrl || !ctrl->block || !ctrl->block->device_ptr) {
                throw std::runtime_error(
                    "SplatExportableStorage allocator: control block missing "
                    "(partially-constructed storage must fail loudly)");
            }
            if (ctrl->poisoned) {
                throw std::runtime_error(
                    "SplatExportableStorage allocator: storage is poisoned");
            }
            if (region >= SplatExportableStorage::Count) {
                throw std::runtime_error(
                    "SplatExportableStorage allocator: region out of range");
            }

            // Live offsets must come from control, never a by-value snapshot.
            void* const data = ctrl->region_ptr(region);
            const std::size_t region_bytes = ctrl->region_bytes[region];

            std::size_t clamped = capacity;
            if (region == SplatExportableStorage::ShN) {
                dtype = DataType::Float16;
                const std::size_t max_cells = region_bytes / kShNElementBytes;
                clamped = std::min(capacity, max_cells);
            } else if (region == SplatExportableStorage::ShNBounds) {
                dtype = DataType::Float32;
                const std::size_t max_floats = region_bytes / kFloatBytes;
                clamped = std::min(capacity, max_floats);
            } else if (ctrl->capacity > 0) {
                clamped = std::min(capacity, ctrl->capacity);
            }

            validate_exportable_shape_fits_region(name, shape, dtype, region_bytes);

            // Capacity headroom (dim0 rows) must also fit the packed region.
            // Mirrors Tensor::storage_allocation_bytes without calling the private helper.
            std::size_t row_elems = 1;
            if (shape.rank() > 1) {
                for (std::size_t i = 1; i < shape.rank(); ++i) {
                    if (shape[i] > 0 &&
                        row_elems > std::numeric_limits<std::size_t>::max() / shape[i]) {
                        throw std::runtime_error(std::format(
                            "SplatExportableStorage allocator: row overflow for '{}'", name));
                    }
                    row_elems *= shape[i];
                }
            }
            const std::size_t rows =
                shape.rank() == 0 ? 0 : (clamped == 0 ? shape[0] : clamped);
            const std::size_t elem_bytes = element_bytes_for_dtype(dtype);
            if (rows > 0 && row_elems > 0 &&
                rows > std::numeric_limits<std::size_t>::max() / row_elems) {
                throw std::runtime_error(std::format(
                    "SplatExportableStorage allocator: capacity element overflow for '{}'",
                    name));
            }
            const std::size_t total_elems = rows * row_elems;
            if (total_elems > 0 &&
                total_elems > std::numeric_limits<std::size_t>::max() / elem_bytes) {
                throw std::runtime_error(std::format(
                    "SplatExportableStorage allocator: capacity byte overflow for '{}'",
                    name));
            }
            const std::size_t alloc_bytes = total_elems * elem_bytes;
            if (alloc_bytes > region_bytes) {
                throw std::runtime_error(std::format(
                    "SplatExportableStorage allocator: capacity for '{}' needs {} bytes "
                    "but region only holds {}",
                    name,
                    alloc_bytes,
                    region_bytes));
            }

            if (!owner) {
                owner = ctrl->block;
            }
            Tensor t = Tensor::from_external_owner(data,
                                                   std::move(shape),
                                                   Device::CUDA,
                                                   dtype,
                                                   std::move(owner),
                                                   clamped,
                                                   stream,
                                                   std::move(external_kind));
            stamp_exportable_provenance(t, ctrl, region);
            return t;
        }

    } // namespace

    SplatTensorAllocator SplatExportableStorage::make_allocator() const {
        auto ctrl = control_;
        if (!ctrl) {
            // no by-value snapshot flavor. Partially-constructed storage must
            // fail loud rather than hand out offsets that go stale on grow.
            throw std::runtime_error(
                "SplatExportableStorage::make_allocator: control block missing "
                "(storage not fully constructed — refuse by-value snapshot views)");
        }

        return [ctrl](TensorShape shape,
                      std::size_t capacity,
                      DataType dtype,
                      std::string_view name) -> Tensor {
            const Region region = region_from_name(name);
            return make_exportable_view(ctrl,
                                        region,
                                        std::move(shape),
                                        capacity,
                                        dtype,
                                        name,
                                        "splat.exportable",
                                        /*owner=*/{},
                                        getCurrentCUDAStream());
        };
    }

    void stamp_exportable_provenance(
        Tensor& tensor,
        std::shared_ptr<SplatExportableStorage::Control> control,
        SplatExportableStorage::Region region) {
        if (!control) {
            return;
        }
        // Store Control as shared_ptr<void> for Tensor metadata.
        std::shared_ptr<void> ctrl_void = control;
        tensor.set_exportable_provenance(
            std::move(ctrl_void),
            static_cast<std::uint32_t>(region),
            control->generation);
    }

    void* resolve_exportable_device_ptr(const Tensor& tensor) {
        if (!tensor.is_valid() || !tensor.has_exportable_provenance()) {
            // Non-exportable or unstamped: fall through to baked pointer.
            return const_cast<void*>(
                tensor.is_valid() ? tensor.data_ptr() : nullptr);
        }
        auto ctrl_void = tensor.exportable_control();
        if (!ctrl_void) {
            return const_cast<void*>(tensor.data_ptr());
        }
        auto* ctrl =
            static_cast<SplatExportableStorage::Control*>(ctrl_void.get());
        if (!ctrl || !ctrl->block || !ctrl->block->device_ptr) {
            throw std::runtime_error(
                "resolve_exportable_device_ptr: exportable control block invalid");
        }
        if (ctrl->poisoned) {
            throw std::runtime_error(
                "resolve_exportable_device_ptr: exportable storage is poisoned");
        }
        const auto region =
            static_cast<SplatExportableStorage::Region>(tensor.exportable_region());
        if (region >= SplatExportableStorage::Count) {
            throw std::runtime_error(
                "resolve_exportable_device_ptr: exportable region out of range");
        }
        // Always re-resolve through the live control block (by construction —
        // no baked pointer can survive a grow). Log generation mismatch so
        // hold-across-grow without rebind is visible in receipts.
        void* const live = ctrl->region_ptr(region);
        const std::uint64_t bound_gen = tensor.exportable_bound_generation();
        if (bound_gen != ctrl->generation) {
            // Fail-loud observability + by-construction safety: always return the
            // live region base so FastGS/Adam cannot illegal-address. Shape/capacity
            // still require rebindSplatData; pointer re-resolve alone is enough for
            // stale q16 readers after the region base moves.
            LOG_ERROR(
                "exportable generation mismatch at resolve: bound_gen={} live_gen={} "
                "region={} baked_ptr={:#x} live_ptr={:#x} (stale q16 pointer — "
                "returning live pointer; caller should rebind for shape/capacity)",
                bound_gen,
                ctrl->generation,
                static_cast<std::size_t>(region),
                reinterpret_cast<std::uintptr_t>(tensor.storage_ptr()),
                reinterpret_cast<std::uintptr_t>(live));
        }
        return live;
    }

    Q16BindPtrs resolve_q16_bind_ptrs(const SplatData& model) {
        Q16BindPtrs out{};
        if (!model.shN_value_quantized()) {
            return out;
        }
        const Tensor& codes = model.shN_raw();
        const Tensor& bounds = model.shN_value_bounds();
        if (!codes.is_valid() || codes.numel() == 0) {
            return out;
        }
        out.codes = static_cast<const float*>(resolve_exportable_device_ptr(codes));
        if (bounds.is_valid() && bounds.numel() > 0) {
            out.bounds = static_cast<const float*>(resolve_exportable_device_ptr(bounds));
        }
        out.n_cells_per_prim = static_cast<unsigned>(
            sh_value_quant::n_value_cells_per_prim(
                static_cast<std::uint32_t>(model.max_sh_coeffs_rest())));
        if (codes.has_exportable_provenance()) {
            out.generation = codes.exportable_bound_generation();
            out.generation_checked = true;
            // Codes + bounds must share the same live generation (0.15 alt).
            if (bounds.is_valid() && bounds.has_exportable_provenance() &&
                bounds.exportable_bound_generation() != codes.exportable_bound_generation()) {
                LOG_ERROR(
                    "q16 codes/bounds generation pair mismatch: codes_gen={} bounds_gen={}",
                    codes.exportable_bound_generation(),
                    bounds.exportable_bound_generation());
            }
        }
        return out;
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

            // same-block rebind (post-grow, pre-grow Vulkan drop) only
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
            Tensor shN_bounds;
            const auto layout_rest =
                static_cast<std::uint32_t>(model.max_sh_coeffs_rest());
            const size_t n_live = static_cast<size_t>(model.size());
            if (layout_rest > 0 && model.shN_raw().is_valid() && model.shN_raw().numel() > 0) {
                // Pad-dropped q16 codes + per-256 float2 bounds live in the block.
                // Same-block rebind only re-views. Cross-allocator install
                // copies q16 codes when the source is already quantized.
                // Float densify temps (ensure_shN_fp32) are kept outside the block —
                // commit_shN_after_mutation re-encodes into exportable after mutation.
                const size_t shN_cap = sh_value_quant::sh_value_u16_count(capacity_, layout_rest);
                const size_t shN_logical = sh_value_quant::sh_value_u16_count(n_live, layout_rest);
                const size_t bounds_cap = sh_value_quant::n_bounds_for_prims(capacity_) * 2u;
                const size_t bounds_logical = sh_value_quant::n_bounds_for_prims(n_live) * 2u;

                const Tensor& shN_src = model.shN_raw();
                const bool aliases = tensor_aliases_exportable_block(shN_src, block_ref);
                const bool src_is_q16 = model.shN_value_quantized();

                if (!aliases && !src_is_q16) {
                    // Densify expand path: preserve float/f16 temp; do not zero-fill.
                    shN = shN_src;
                    if (shN.device() != Device::CUDA) {
                        shN = shN.cuda();
                    }
                    if (!shN.is_contiguous()) {
                        shN = shN.contiguous();
                    }
                    // Bounds stay empty until commit re-encodes.
                    shN_bounds = Tensor{};
                } else {
                    Tensor dst = allocator(
                        TensorShape({shN_logical}),
                        shN_cap,
                        DataType::Float16,
                        "SplatData.shN");
                    dst.set_name("SplatData.shN");
                    if (!aliases && src_is_q16 && shN_src.is_valid() && shN_src.numel() > 0) {
                        Tensor src = shN_src;
                        if (src.device() != Device::CUDA) {
                            src = src.cuda();
                        }
                        if (!src.is_contiguous()) {
                            src = src.contiguous();
                        }
                        dst.copy_from(src);
                    }
                    shN = std::move(dst);

                    Tensor bounds_dst = allocator(
                        TensorShape({bounds_logical}),
                        bounds_cap,
                        DataType::Float32,
                        "SplatData.shN_value_bounds");
                    bounds_dst.set_name("SplatData.shN_value_bounds");
                    const Tensor& bounds_src = model.shN_value_bounds();
                    const bool bounds_alias =
                        bounds_src.is_valid() &&
                        tensor_aliases_exportable_block(bounds_src, block_ref);
                    if (!bounds_alias && bounds_src.is_valid() && bounds_src.numel() > 0) {
                        Tensor bsrc = bounds_src;
                        if (bsrc.device() != Device::CUDA) {
                            bsrc = bsrc.cuda();
                        }
                        if (!bsrc.is_contiguous()) {
                            bsrc = bsrc.contiguous();
                        }
                        if (bsrc.dtype() != DataType::Float32) {
                            bsrc = bsrc.to(DataType::Float32);
                        }
                        bounds_dst.copy_from(bsrc);
                    }
                    shN_bounds = std::move(bounds_dst);
                }
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
            rebound.set_active_sh_degree(active_sh, std::move(shN_bounds));
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
            // layout change (post-grow re-fetch signal).
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
