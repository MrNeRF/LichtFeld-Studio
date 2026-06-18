/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "io/formats/rad_packed_page.hpp"
#include "lod_page_cache.hpp"
#include "rendering/cuda_vulkan_interop.hpp"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <list>
#include <mutex>
#include <vector>

namespace lfs::vis {

    // Streams packed RAD pages into the CUDA-imported pool buffers off the
    // render thread. Decode workers acquire a pinned staging slot, fill it
    // via decode_rad_chunk_packed (inflated-but-quantized planes + sidecar
    // metadata), then submit; the engine copies the slot to a paired device
    // scratch buffer and runs the page-dequant kernel on a dedicated CUDA
    // stream. The render thread polls collectPublished() and publishes
    // residency only for pages whose kernel completed, so the selector never
    // observes a partially written page. Staging slots are the backpressure.
    class LFS_VIS_API LodUploadEngine {
    public:
        struct DeviceLayout {
            // Payload pool (page-input regions, InputRegion order): xyz_ws,
            // sh0, shN, rotations, scaling_raw, opacity_raw, page_frames —
            // canonical quantized layout per lod_pool_quant.hpp.
            void* device_base = nullptr;
            std::array<std::size_t, 7> region_offset{};
            std::size_t splat_capacity = 0;
            // Destination SH-rest coefficient count and float4 slot count.
            std::uint32_t dst_rest = 0;
            std::uint32_t dst_slots = 0;
            // Tree-metadata pool (expanded NodeBounds/NodeLinks records).
            // Its capacity can lag the payload pool's across reconfigure
            // frames; submits bounds-check against it independently.
            void* meta_base = nullptr;
            std::size_t meta_bounds_offset = 0;
            std::size_t meta_links_offset = 0;
            std::size_t meta_capacity_nodes = 0;

            [[nodiscard]] bool valid() const { return device_base != nullptr && splat_capacity > 0; }
            [[nodiscard]] friend bool operator==(const DeviceLayout&, const DeviceLayout&) = default;
        };

        // Staging slots are opaque byte arenas laid out by
        // decode_rad_chunk_packed; only the capacity is fixed.
        struct StagingLayout {
            std::size_t total_bytes = 0;
        };

        struct StagingSlot {
            std::uint8_t* data = nullptr;        // pinned host arena
            std::uint8_t* device_data = nullptr; // paired device scratch
            cudaEvent_t last_use = nullptr;
            bool used = false;
            bool acquired = false;
        };

        LodUploadEngine();
        ~LodUploadEngine();
        LodUploadEngine(const LodUploadEngine&) = delete;
        LodUploadEngine& operator=(const LodUploadEngine&) = delete;

        // Layout changes drain outstanding work first (returns those results,
        // see drainAndSync). The timeline may be null (tests: no Vulkan
        // handoff signal). Safe to call every frame with an unchanged layout.
        std::vector<LodPageCache::PendingUpload>
        configure(const DeviceLayout& layout, const lfs::rendering::CudaTimelineSemaphore* timeline);

        [[nodiscard]] bool configured() const;
        [[nodiscard]] StagingLayout stagingLayout() const;
        [[nodiscard]] bool idle() const;

        // Decode-worker API: blocks until a staging slot's previous copies
        // completed (ring depth >= worker count keeps waits rare); returns
        // nullptr when unconfigured or shutting down. Every acquired slot must
        // be passed to submitPackedPage or releaseSlot.
        [[nodiscard]] StagingSlot* acquireStagingSlot();
        void releaseSlot(StagingSlot* slot);
        // Issues the slot copy + dequant kernel for a packed slot. Thread-safe
        // across decode workers; timeline values stay monotone in stream order.
        void submitPackedPage(StagingSlot* slot,
                              const lfs::io::RadPagePackedDesc& desc,
                              std::uint32_t page,
                              std::uint64_t generation);

        // Completed (event-signaled) uploads in submission order.
        [[nodiscard]] std::vector<LodPageCache::PendingUpload> collectPublished();
        // Timeline value covering every upload returned by collectPublished()
        // so far — the value a Vulkan submit must wait on before sampling the
        // published pages. Never includes still-running copies.
        [[nodiscard]] std::uint64_t lastPublishedSignalValue() const;

        // Blocks until all submitted copies are stream-complete; returns the
        // final results so the caller can release page reservations. Called
        // before the pool buffers are freed.
        std::vector<LodPageCache::PendingUpload> drainAndSync();

    private:
        struct Job {
            LodPageCache::PendingUpload upload;
            cudaEvent_t event = nullptr;
            std::uint64_t signal_value = 0;
        };

        [[nodiscard]] cudaEvent_t acquireEventLocked();
        [[nodiscard]] std::vector<LodPageCache::PendingUpload> takeCompletedLocked(bool wait_for_all);
        void releaseStagingRingLocked();

        mutable std::mutex mutex_;
        std::condition_variable slot_cv_;
        bool shutdown_ = false;

        DeviceLayout layout_{};
        StagingLayout staging_layout_{};
        const lfs::rendering::CudaTimelineSemaphore* timeline_ = nullptr;
        cudaStream_t stream_ = nullptr;
        std::uint64_t signal_counter_ = 0;
        std::uint64_t last_published_signal_ = 0;

        std::list<Job> in_flight_;
        std::vector<cudaEvent_t> event_pool_;
        std::vector<StagingSlot> staging_ring_;
        std::size_t staging_cursor_ = 0;
    };

} // namespace lfs::vis
