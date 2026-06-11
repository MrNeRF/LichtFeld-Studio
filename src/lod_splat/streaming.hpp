/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Streaming layer: storage is chunked by (octree node -> splat payload), so a
// hierarchy cut directly defines the fetch set.
//
// Behaviors: refines blocked on residency keep rendering the coarser parent
// (never holes out); LRU eviction protects the current cut, its ancestors
// (fallback chain) and recently-touched chunks; prefetch extrapolates the
// camera along its velocity and requests the children the predicted cut would
// need. Payloads are quantized — coarse interior chunks store quantized
// position/covariance and 8-bit color (SH degree 0 only); leaf chunks keep
// full-precision positions. Transfer is simulated against a configurable
// latency + bandwidth model so the scheduling behavior can be profiled
// deterministically without real storage in the loop.
#pragma once

#include "cut.hpp"
#include "hierarchy.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace lfs::lod {

    struct StreamParams {
        double base_latency_ms = 2.0;  // per-request fixed cost (NVMe-ish; raise for network)
        double bandwidth_mb_s = 800.0; // shared fetch bandwidth
        int max_in_flight = 16;
        size_t resident_budget_bytes = 256ull << 20; // decoded-pool cap
        float prefetch_lookahead_s = 0.6f;
        int prefetch_max_chunks_per_frame = 128; // keeps prefetch from starving demand
        int protect_frames = 4;                  // touched within N frames is not evictable
    };

    struct StreamStats {
        uint64_t requests_demand = 0, requests_prefetch = 0;
        uint64_t completed = 0, bytes_fetched = 0;
        uint64_t evictions = 0, evicted_bytes = 0;
        uint64_t prefetch_useful = 0; // prefetched chunks later touched by the cut
        double decode_ms = 0;
        size_t resident_bytes = 0, resident_chunks = 0;
        double avg_wait_ms = 0; // request issue -> completion (completed requests)
    };

    class StreamingManager : public ResidencyProvider {
    public:
        // Encodes every node payload into the simulated backing store.
        StreamingManager(const Hierarchy& h, const StreamParams& p);

        // -- ResidencyProvider --
        bool isResident(uint32_t node) const override { return resident_.count(node) != 0; }
        void request(uint32_t node, float priority) override { enqueue(node, priority, false); }
        void touch(uint32_t node) override;

        // Advance simulated time; complete arrivals (decode), issue queued fetches,
        // evict over budget. Call once per frame with the simulated clock.
        void tick(double sim_now_ms, uint32_t frame_idx);

        // Request what a camera extrapolated along `velocity` would refine to.
        void prefetch(const Camera& cam, const Vec3& velocity_per_s, const CutParams& cut_params,
                      const std::vector<uint32_t>& cut_nodes);

        const SplatCloud* chunk(uint32_t node) const {
            auto it = resident_.find(node);
            return it == resident_.end() ? nullptr : &it->second.data;
        }

        size_t encodedStoreBytes() const { return store_total_bytes_; }
        const StreamStats& stats() const { return stats_; }
        StreamStats& stats() { return stats_; }

    private:
        struct ResidentChunk {
            SplatCloud data;
            size_t bytes = 0; // decoded footprint
            uint32_t last_touch_frame = 0;
            bool from_prefetch = false;
            bool counted_useful = false;
        };
        struct Pending {
            uint32_t node;
            float priority;
            bool prefetch;
            double enqueue_ms;
        };
        struct InFlight {
            uint32_t node;
            double issue_ms, done_ms;
            bool prefetch;
        };

        void enqueue(uint32_t node, float priority, bool prefetch);
        void decodeChunk(uint32_t node, ResidentChunk& out) const;
        void evictIfNeeded();

        const Hierarchy& h_;
        StreamParams params_;
        // simulated backing store: encoded payload per node
        std::vector<std::vector<uint8_t>> store_;
        std::vector<uint8_t> coarse_format_; // per node
        size_t store_total_bytes_ = 0;

        std::unordered_map<uint32_t, ResidentChunk> resident_;
        std::vector<uint8_t> queued_or_flying_; // per node
        std::vector<Pending> pending_;
        std::vector<InFlight> in_flight_;
        double bandwidth_free_at_ms_ = 0;
        bool lod_opacity_ = false; // merged chunks carry linear alpha in [0,4]
        uint32_t frame_idx_ = 0;
        StreamStats stats_;
        double wait_accum_ms_ = 0;
    };

} // namespace lfs::lod
