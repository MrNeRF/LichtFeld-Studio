/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "io/formats/rad.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace lfs::vis {

    namespace detail {
        // A refreshed budget may request a new capacity without requiring a
        // cache reset when the model and effective page count are unchanged.
        [[nodiscard]] constexpr bool lodPageCacheNeedsReconfigure(
            const bool same_model,
            const bool sizing_dirty,
            const std::size_t current_pages,
            const std::size_t requested_pages) {
            return !same_model || (sizing_dirty && current_pages != requested_pages);
        }

        // What the page-pool sizing policy sees of a model and its device.
        struct LodPoolSizing {
            std::size_t logical_chunks = 0;
            // Deleted mask, or all tensors already in VRAM: page every chunk.
            bool full_pool = false;
            // More tree nodes than resident splats: the pool is the working set.
            bool out_of_core = false;
            std::size_t budget_splats = 0; // explicit host-streaming budget, 0 = auto
            float vram_fraction = 0.15f;
            std::size_t free_bytes = 0; // device headroom, 0 = unknown
            std::size_t current_pages = 0, page_bytes = 1;
        };

        // Physical pages of the LOD page pool, shared by every GPU LOD viewer.
        [[nodiscard]] inline std::size_t lodPoolPages(const LodPoolSizing& sizing) {
            const std::size_t chunks = sizing.logical_chunks;
            if (chunks == 0 || sizing.full_pool)
                return chunks;
            constexpr std::size_t kMinPoolPages = 2048;
            if (!sizing.out_of_core && sizing.budget_splats > 0) {
                // Explicit splat budget for host-resident streaming.
                const std::size_t budget_pages =
                    (sizing.budget_splats + lfs::core::SplatLodTree::kChunkSplats - 1) /
                    lfs::core::SplatLodTree::kChunkSplats;
                return std::clamp(budget_pages, std::min(kMinPoolPages, chunks), chunks);
            }
            // Out-of-core, or host-resident leaves on auto budget: the pool is
            // real VRAM the renderer must share — size it from headroom. A
            // host-resident model previously fell through to a FULL pool here,
            // which allocates pages x page_bytes regardless of free VRAM
            // (issue #1295: 94K pages = 16.6 GB on a card with 11.8 free).
            // Only a coarse LOD prefix is host-resident, so the pool must
            // hold the cut's whole working set or streaming thrashes. Size
            // it from VRAM headroom; the splat-count budget setting was
            // tuned for host-resident streaming and is far too small here.
            if (sizing.free_bytes == 0)
                return std::min(kMinPoolPages, chunks);
            // The current pool's own allocation counts as available to
            // its replacement, otherwise resizing feeds back on itself.
            const double page_bytes = static_cast<double>(std::max<std::size_t>(sizing.page_bytes, 1));
            const double budget_bytes =
                (static_cast<double>(sizing.free_bytes) + static_cast<double>(sizing.current_pages) * page_bytes) *
                std::clamp(sizing.vram_fraction, 0.05f, 0.9f);
            auto budget_pages = static_cast<std::size_t>(budget_bytes / page_bytes);
            // Scratch/ring reallocations wobble the free-VRAM reading
            // by a few pages between frames; an unquantized re-derive
            // reconfigure-loops (each pool reset moves free VRAM,
            // landing on a new count, resetting again). Page-quantum
            // steps absorb the wobble so the re-derive is idempotent.
            constexpr std::size_t kPoolPageQuantum = 512;
            budget_pages -= budget_pages % kPoolPageQuantum;
            return std::clamp(budget_pages, std::min(kMinPoolPages, chunks), chunks);
        }
    } // namespace detail

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
        // flight; runs every beginFrame so transient decode errors recover.
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
