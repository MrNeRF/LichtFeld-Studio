/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "io/formats/rad.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace lfs::vis {

    class LFS_VIS_API LodPageCache {
    public:
        static constexpr std::uint32_t kInvalidPage = lfs::core::SplatLodTree::kInvalidPage;
        static constexpr std::size_t kChunkSplats = lfs::core::SplatLodTree::kChunkSplats;

        // Chunks rendered from (or actively requested) within this many frames
        // are exempt from eviction; covers the one-frame-stale GPU readback.
        static constexpr std::uint64_t kDefaultProtectWindowFrames = 3;

        struct ChunkRequest {
            std::uint32_t chunk = kInvalidPage;
            // Larger is more urgent. The GPU selector supplies asuint(pixel_scale).
            std::uint32_t priority = 0;
        };

        struct Snapshot {
            std::vector<std::uint32_t> page_to_chunk;
            std::vector<std::uint32_t> chunk_to_page;
            // Frame stamp of each page's last publish; drives selector fade-in.
            std::vector<std::uint32_t> page_resident_frame;
            std::size_t logical_chunks = 0;
            std::size_t physical_pages = 0;
            std::size_t resident_chunks = 0;
            std::uint64_t generation = 0;
        };

        struct PendingUpload {
            std::uint32_t page = kInvalidPage;
            std::uint32_t chunk = kInvalidPage;
            std::uint64_t generation = 0;
            std::string error;
        };

        // Installed by the renderer: called on decode workers with a chunk's
        // raw file bytes; decodes into upload-engine staging and submits the
        // async copies. Returns an error string (empty = accepted); failures
        // release the page reservation and the chunk re-requests later.
        using PageSink = std::function<std::string(std::uint32_t chunk,
                                                   std::uint32_t page,
                                                   std::uint64_t generation,
                                                   std::span<const std::uint8_t> chunk_bytes)>;

        LodPageCache();
        ~LodPageCache();
        LodPageCache(const LodPageCache&) = delete;
        LodPageCache& operator=(const LodPageCache&) = delete;

        void reset();
        // disk_backed: pages stream from the RAD file; never pre-publish,
        // even at full capacity (only a preview prefix is loaded).
        void configure(std::size_t logical_chunk_count,
                       std::size_t physical_page_capacity,
                       std::size_t root_chunk_count = 1,
                       std::size_t page_payload_bytes = 0,
                       bool disk_backed = false);
        void setRadSource(const lfs::core::SplatLodTree::RadSource* source,
                          int max_sh_degree,
                          bool lod_opacity_encoded);
        void setPageSink(PageSink sink);

        // Advances the frame clock used by the eviction-protection window,
        // residency fade stamps, and stale-request expiry. Call once per
        // rendered frame before submitting traversal priorities.
        void beginFrame();

        // Re-requests any pinned root chunk that is not resident or in
        // flight; runs every beginFrame so failed bootstrap streams
        // (sink installed late, transient decode error) self-heal.
        void ensureRootResidency();

        // Legacy overload: chunks in caller-priority order, per-call protection
        // only (CPU SparkLodController path).
        void submitTraversalPriority(std::span<const std::uint32_t> chunks,
                                     std::span<const std::uint32_t> protected_chunks = {});
        // GPU-selector overload: explicit per-chunk priorities. Protected and
        // resident-requested chunks receive the multi-frame eviction-protection
        // window; queued decode requests are reprioritized and stale ones drop.
        void submitTraversalPriority(std::span<const ChunkRequest> requests,
                                     std::span<const std::uint32_t> protected_chunks);

        // max_bytes 0 = unlimited. Otherwise drains whole pages up to the
        // budget (at least one), leaving the rest for subsequent frames.
        [[nodiscard]] std::vector<PendingUpload> drainPendingUploads(std::size_t max_bytes = 0);
        void completeUploads(std::span<const PendingUpload> uploads);

        [[nodiscard]] const Snapshot& snapshot() const { return snapshot_; }
        [[nodiscard]] bool configured() const { return snapshot_.logical_chunks > 0; }
        [[nodiscard]] bool fullyResident() const {
            return snapshot_.resident_chunks == snapshot_.logical_chunks;
        }
        [[nodiscard]] bool hasOutstandingWork() const { return outstandingWorkCount() > 0; }
        [[nodiscard]] std::size_t outstandingWorkCount() const;
        [[nodiscard]] std::size_t deferredRequestCount() const { return deferred_requests_; }
        // Requests admitted (reservation created) since the last beginFrame;
        // deferred>0 with zero admissions and nothing in flight is a frozen
        // pool — every resident page is in the live cut and unevictable.
        [[nodiscard]] std::size_t admittedRequestCount() const { return admitted_requests_; }
        [[nodiscard]] std::uint64_t frameIndex() const { return frame_; }

    private:
        struct PageSlot {
            std::uint32_t chunk = kInvalidPage;
            std::uint32_t loading_chunk = kInvalidPage;
            // Latest traversal interest (float bits of pixel scale); decays
            // with idle frames in effectivePriority for eviction admission.
            std::uint32_t last_priority = 0;
            std::uint64_t last_touch_frame = 0;
            std::uint64_t last_used = 0;
            std::uint64_t protected_until_frame = 0;
            bool pinned = false;
        };
        struct RadSourceSnapshot {
            std::filesystem::path path;
            std::vector<lfs::core::SplatLodTree::ChunkFileRange> chunks;
            int max_sh_degree = 0;
            bool lod_opacity_encoded = false;

            [[nodiscard]] bool valid() const { return !path.empty() && !chunks.empty(); }
        };
        struct DecodeScheduler;

        // Admission outcome, kept distinct per refusal so the perf log can
        // tell a frozen pool (margin/no-slot) from normal flow control.
        enum class AdmitResult : std::uint8_t {
            Admitted,
            AlreadyActive,
            NoSlot,
            MarginRefused,
            Invalid,
        };

        void submitInternal(std::span<const ChunkRequest> requests,
                            std::span<const std::uint32_t> protected_chunks,
                            bool stamp_resident_requests);
        AdmitResult requestResident(std::uint32_t chunk,
                                    bool pin,
                                    std::uint32_t priority,
                                    std::span<const std::uint8_t> protected_pages = {},
                                    bool screened_inactive = false);
        [[nodiscard]] std::size_t chooseEvictionSlot(
            std::span<const std::uint8_t> protected_pages = {}) const;
        void buildEvictionScratch(std::span<const std::uint8_t> protected_pages) const;
        void reserveUpload(std::size_t page, std::uint32_t chunk, bool pin, std::uint32_t priority);
        void publishPage(std::size_t page, std::uint32_t chunk, bool pin);
        void invalidateResidentPage(std::size_t page);
        void collectFinishedDecodes();
        [[nodiscard]] bool decodeInFlight(std::uint32_t chunk) const;
        void stampProtection(std::uint32_t chunk, std::uint32_t priority = 0);
        [[nodiscard]] std::uint32_t effectivePriority(const PageSlot& slot) const;
        [[nodiscard]] std::size_t requestBudgetPages() const;

        // Per-submit eviction scratch: one O(pages) candidate pass amortized
        // over all of a frame's admissions, replacing an O(pages) scan per
        // admission (measured 50 ms/frame at 20K pages x 330 admits). Built
        // lazily on the first chooseEvictionSlot inside submitInternal;
        // outside a submit the legacy linear scan runs.
        struct EvictionCandidate {
            std::uint32_t priority = 0;
            std::uint64_t last_used = 0;
            std::uint32_t page = 0;
        };
        mutable std::vector<std::uint32_t> eviction_free_scratch_;
        mutable std::vector<EvictionCandidate> eviction_heap_;
        mutable bool eviction_scratch_enabled_ = false;
        mutable bool eviction_scratch_dirty_ = false;

        std::vector<PageSlot> pages_;
        std::vector<PendingUpload> pending_uploads_;
        std::shared_ptr<const PageSink> page_sink_;
        std::unique_ptr<DecodeScheduler> scheduler_;
        RadSourceSnapshot rad_source_;
        Snapshot snapshot_;
        std::uint64_t clock_ = 0;
        std::uint64_t frame_ = 0;
        std::uint64_t protect_window_frames_ = kDefaultProtectWindowFrames;
        std::size_t root_chunk_count_ = 0;
        bool disk_backed_ = false;
        std::size_t page_payload_bytes_ = 0;
        std::size_t deferred_requests_ = 0;
        std::size_t admitted_requests_ = 0;
        std::uint64_t last_admission_log_frame_ = 0;
    };

} // namespace lfs::vis
