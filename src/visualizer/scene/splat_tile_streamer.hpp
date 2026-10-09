/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/splat_data.hpp"
#include "io/splat_tile_source.hpp"
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::vis {

    struct SplatTileStreamSettings {
        float cache_fraction = 0.5f; // share of total GPU memory the tile cache may use
        float max_sse = 16.0f;       // pixels of error a tile may show before refining
        bool cull = true;            // false: select by distance only, keeping off-view tiles
        bool freeze = false;         // keep the current selection while the camera moves
        int num_load_workers = 0;    // tile decode/upload threads; 0 picks an automatic count
        // Distance-selected sources (streamed SOG), as PlayCanvas lodBaseDistance / lodMultiplier.
        float lod_base_distance = 5.0f;
        float lod_multiplier = 3.0f;
    };

    // Automatic decode-worker count: enough to overlap read/decode/upload without
    // starving the render thread or a concurrent training job. The parallel work is
    // CPU-side SPZ decode; host->device uploads share one bus and the render-set merge
    // stays serial, so returns flatten past a few workers. clamp(cores / 4, 2, 4).
    [[nodiscard]] std::size_t auto_tile_load_workers();

    struct SplatTileBudgetUsage {
        std::uint64_t cached_bytes = 0;
        std::uint64_t in_flight_bytes = 0;
        std::uint64_t drawn_bytes = 0;
        std::uint64_t replacement_bytes = 0;
        std::uint64_t limit_bytes = 0;
        bool replacing_drawn = false;
        bool minimum_cut = false;
    };

    // A replacement is charged for the model that will remain after the swap, not the
    // old model that it releases. The old and new models may overlap temporarily to keep
    // REPLACE refinement hole-free. The minimum cut is always admitted as a budget floor.
    [[nodiscard]] LFS_VIS_API bool splat_tile_budget_allows_load(const SplatTileBudgetUsage& usage,
                                                                 std::uint64_t incoming_bytes);

    struct SplatTileStreamStats {
        std::size_t tiles = 0;
        std::size_t drawn_tiles = 0;
        std::size_t cached_tiles = 0;
        std::size_t loading_tiles = 0;
        std::size_t failed_tiles = 0;
        std::size_t skipped_contents = 0; // contents without supported splat data
        std::uint64_t drawn_splats = 0;
        std::uint64_t full_detail_splats = 0;
        std::uint64_t cache_bytes = 0;
        std::uint64_t drawn_bytes = 0; // merged models (drawn and pending); count toward the limit
        std::uint64_t cache_limit_bytes = 0;
        std::uint64_t gpu_total_bytes = 0;
        double build_ms = 0.0;        // last merge of the drawn tiles
        float max_sse = 0.0f;         // in use; above the setting while the view exceeds the cache
        std::size_t load_workers = 0; // tile decode/upload threads currently running
        bool distance_lod = false;    // the source refines at the LOD distances
    };

    // Blocks until every retired streamer's workers have exited. Workers can be in the
    // middle of a tile decode or a merge when their streamer is released; call this
    // before the GPU backend goes away (application shutdown).
    void wait_for_retired_tile_workers();

    // View-dependent streaming of a SplatTileSource into one scene node. The main
    // thread selects tiles; a worker loads them into a GPU LRU cache and merges
    // the render set into a renderer-ready model for the caller to swap in.
    class SplatTileStreamer {
    public:
        // Releasing a Handle retires the streamer without waiting: its workers stop after
        // their current job and the last one to exit frees the streamer.
        struct Retire {
            void operator()(SplatTileStreamer* streamer) const;
        };
        using Handle = std::unique_ptr<SplatTileStreamer, Retire>;

        // The node initially shows the source's coarsest cut (see Tiles3dLoader).
        [[nodiscard]] static Handle create(std::shared_ptr<const io::SplatTileSource> source,
                                           core::SplatTensorAllocator allocator);
        ~SplatTileStreamer();
        SplatTileStreamer(const SplatTileStreamer&) = delete;
        SplatTileStreamer& operator=(const SplatTileStreamer&) = delete;

        // view: camera and frustum in the source-local frame. Returns the next
        // model when a new render set finished building. `wake` is called from the
        // worker when loaded tiles call for another update.
        [[nodiscard]] std::unique_ptr<core::SplatData> update(const io::SplatTileView& view,
                                                              const SplatTileStreamSettings& settings,
                                                              std::function<void()> wake);
        [[nodiscard]] SplatTileStreamStats stats();

    private:
        struct CachedTile {
            std::shared_ptr<const core::SplatData> data;
            std::uint64_t bytes = 0;
            std::uint64_t last_wanted = 0;
        };

        SplatTileStreamer(std::shared_ptr<const io::SplatTileSource> source,
                          core::SplatTensorAllocator allocator);

        void work(const std::stop_token& stop);
        // Grows or shrinks the decode-worker pool to `count` (at least one). Main thread;
        // retired workers are stopped and detached, never joined.
        void resizeWorkers(std::size_t count);
        // Releases least recently wanted tiles until `incoming` more bytes fit the limit.
        void evictLocked(std::uint64_t incoming = 0);
        // GPU bytes held by the stream: cached and in-flight tiles plus the drawn,
        // finished and in-progress merged models, which copy their tiles.
        [[nodiscard]] std::uint64_t usedBytesLocked() const;
        [[nodiscard]] SplatTileBudgetUsage budgetUsageLocked(bool minimum_cut) const;

        std::shared_ptr<const io::SplatTileSource> source_;
        core::SplatTensorAllocator allocator_;
        std::uint64_t gpu_total_bytes_ = 0;
        std::uint64_t full_detail_splats_ = 0;
        std::uint64_t cache_limit_bytes_ = 0;
        double build_ms_ = 0.0;

        std::mutex mutex_;
        std::condition_variable_any cv_;
        std::unordered_map<std::uint32_t, CachedTile> cache_;
        std::unordered_set<std::uint32_t> failed_;
        std::unordered_set<std::uint32_t> in_flight_; // tiles a worker is currently loading
        std::uint64_t cache_bytes_ = 0;
        std::uint64_t in_flight_bytes_ = 0;        // cache space reserved for in-flight tiles
        std::uint64_t drawn_bytes_ = 0;            // model the node currently shows
        std::uint64_t built_bytes_ = 0;            // finished model awaiting its swap (built_)
        std::uint64_t building_bytes_ = 0;         // model a worker is merging
        std::uint64_t replacement_bytes_ = 0;      // merged model reserved by the ideal cut
        std::uint64_t release_gen_ = 0;            // bumped when a merged model's memory frees up
        std::vector<std::uint32_t> wanted_;        // load queue, most urgent first
        std::vector<std::uint32_t> build_request_; // latest render set awaiting a merge
        // Generations order merges so the newest finished cut wins regardless of the order
        // workers finish in, instead of discarding a merge whenever a finer cut was asked
        // for meanwhile. A merge installs only when its generation is newer than the drawn
        // one; a single in-flight merge (building_) keeps the other workers loading tiles.
        std::uint64_t build_gen_ = 0;         // monotonic request counter
        std::uint64_t build_request_gen_ = 0; // generation of build_request_
        std::uint64_t installed_gen_ = 0;     // generation of the drawn model
        bool building_ = false;               // a worker is merging a render set
        int merge_failures_ = 0;              // consecutive failed merges; new cuts pause at the limit
        std::unique_ptr<core::SplatData> built_;
        std::vector<std::uint32_t> built_set_;
        std::uint64_t frame_ = 0;
        bool cache_changed_ = true;
        bool over_budget_ = false; // the view's tiles do not fit the cache
        bool replacing_drawn_ = false;
        bool minimum_cut_ = false; // the coarsest cut is the non-negotiable budget floor
        float sse_factor_ = 1.0f;  // memory-adjusted multiplier of the max SSE
        std::function<void()> wake_;

        // Written by update() on the main thread; workers read them under mutex_.
        std::vector<std::uint32_t> shown_set_;
        std::vector<std::uint32_t> requested_set_;
        io::SplatTileView last_view_{};

        // Workers each hold a reference, so the streamer outlives its released Handle until
        // they exit; this one is the Handle's, dropped on retirement.
        std::shared_ptr<SplatTileStreamer> self_;
        std::vector<std::jthread> workers_;
    };

} // namespace lfs::vis
