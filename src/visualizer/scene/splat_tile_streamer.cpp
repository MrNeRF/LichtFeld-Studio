/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "splat_tile_streamer.hpp"
#include "core/logger.hpp"
#include "core/tensor_backend.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <thread>

namespace lfs::vis {

    namespace {
        constexpr float kSseFactorStep = 1.02f;
        constexpr float kMaxSseFactor = 64.0f;

        std::uint64_t tile_bytes(const io::SplatTile& tile) {
            const int rest = (tile.sh_degree + 1) * (tile.sh_degree + 1) - 1;
            return tile.splat_count * (14 + 3 * rest) * sizeof(float);
        }

        bool same_view(const io::SplatTileView& a, const io::SplatTileView& b) {
            return a.camera == b.camera && a.planes == b.planes && a.sse_per_error == b.sse_per_error &&
                   a.max_sse == b.max_sse;
        }
    } // namespace

    std::size_t auto_tile_load_workers() {
        const unsigned cores = std::thread::hardware_concurrency();
        return std::clamp<std::size_t>(cores / 4, 2, 4);
    }

    SplatTileStreamer::SplatTileStreamer(std::shared_ptr<const io::SplatTileSource> source,
                                         core::SplatTensorAllocator allocator)
        : source_(std::move(source)),
          allocator_(std::move(allocator)) {
        gpu_total_bytes_ = core::gpu_backend_memory_info(core::default_gpu_backend()).total_bytes;
        cache_limit_bytes_ = static_cast<std::uint64_t>(SplatTileStreamSettings{}.cache_fraction *
                                                        static_cast<double>(gpu_total_bytes_));
        const auto everything = [](std::uint32_t) { return true; };
        full_detail_splats_ = io::select_splat_tiles(*source_, {.sse_per_error = 1.0f, .max_sse = 0.0f}, everything)
                                  .render_splats;
        shown_set_ = io::select_splat_tiles(*source_, {.max_sse = std::numeric_limits<float>::infinity()}, everything)
                         .render;
        std::ranges::sort(shown_set_);
        requested_set_ = shown_set_;
        for (const auto tile : shown_set_)
            drawn_bytes_ += tile_bytes(source_->tiles()[tile]);
        resizeWorkers(auto_tile_load_workers());
    }

    SplatTileStreamer::~SplatTileStreamer() {
        for (auto& worker : workers_)
            worker.request_stop();
        cv_.notify_all();
    }

    void SplatTileStreamer::resizeWorkers(std::size_t count) {
        count = std::max<std::size_t>(count, 1);
        if (count == workers_.size())
            return;
        if (count < workers_.size()) {
            for (std::size_t i = count; i < workers_.size(); ++i)
                workers_[i].request_stop();
            cv_.notify_all();
            workers_.resize(count); // jthread destructors join the retired workers
        } else {
            while (workers_.size() < count)
                workers_.emplace_back([this](const std::stop_token& stop) { work(stop); });
        }
    }

    std::unique_ptr<core::SplatData> SplatTileStreamer::update(const io::SplatTileView& view,
                                                               const SplatTileStreamSettings& settings,
                                                               std::function<void()> wake) {
        resizeWorkers(settings.num_load_workers > 0 ? static_cast<std::size_t>(settings.num_load_workers)
                                                    : auto_tile_load_workers());
        std::lock_guard lock(mutex_);
        if (!wake_)
            wake_ = std::move(wake);
        const auto cache_limit = static_cast<std::uint64_t>(
            std::clamp(settings.cache_fraction, 0.0f, 1.0f) * static_cast<double>(gpu_total_bytes_));
        if (cache_limit != cache_limit_bytes_) {
            // A new size starts from an empty cache; the drawn model is a separate copy
            // and stays until the reloaded tiles cover the view again.
            cache_limit_bytes_ = cache_limit;
            cache_.clear();
            cache_bytes_ = 0;
            over_budget_ = false;
            sse_factor_ = 1.0f;
            requested_set_.clear();
            build_request_.clear();
            cache_changed_ = true;
            cv_.notify_all();
        }
        // Memory-adjusted error: when the view's tiles do not fit the cache, accept a
        // slightly larger error each frame; relax it again once memory frees up.
        if (over_budget_ && sse_factor_ < kMaxSseFactor) {
            sse_factor_ = std::min(sse_factor_ * kSseFactorStep, kMaxSseFactor);
            cache_changed_ = true;
        } else if (!over_budget_ && sse_factor_ > 1.0f && usedBytesLocked() < cache_limit_bytes_ * 4 / 5) {
            sse_factor_ = std::max(sse_factor_ / kSseFactorStep, 1.0f);
            cache_changed_ = true;
        }
        if (!settings.freeze && (cache_changed_ || !same_view(view, last_view_))) {
            cache_changed_ = false;
            last_view_ = view;
            ++frame_;
            auto adjusted = view;
            adjusted.max_sse *= sse_factor_;
            auto selection = io::select_splat_tiles(*source_, adjusted, [this](const std::uint32_t tile) {
                return cache_.contains(tile);
            });
            if (sse_factor_ > 1.0f && wake_)
                wake_(); // keep adjusting while the cache is over or under budget
            wanted_ = std::move(selection.wanted);
            for (const auto tile : wanted_)
                if (const auto it = cache_.find(tile); it != cache_.end())
                    it->second.last_wanted = frame_;
            std::ranges::sort(selection.render);
            // An incomplete cut would open holes; keep showing the current one until it fills in.
            if (selection.complete && !selection.render.empty() && selection.render != requested_set_) {
                requested_set_ = selection.render;
                build_request_ = std::move(selection.render);
                build_request_gen_ = ++build_gen_;
            }
            cv_.notify_all();
        }
        if (!built_)
            return nullptr;
        shown_set_ = std::move(built_set_);
        drawn_bytes_ = built_bytes_; // the caller releases the previous model on swap
        built_bytes_ = 0;
        ++release_gen_;
        cv_.notify_all();
        return std::move(built_);
    }

    SplatTileStreamStats SplatTileStreamer::stats() {
        std::lock_guard lock(mutex_);
        const auto tiles = source_->tiles();
        SplatTileStreamStats out{.tiles = tiles.size(),
                                 .drawn_tiles = shown_set_.size(),
                                 .cached_tiles = cache_.size(),
                                 .failed_tiles = failed_.size(),
                                 .full_detail_splats = full_detail_splats_,
                                 .cache_bytes = cache_bytes_,
                                 .drawn_bytes = drawn_bytes_ + building_bytes_ + (built_ ? built_bytes_ : 0),
                                 .cache_limit_bytes = cache_limit_bytes_,
                                 .gpu_total_bytes = gpu_total_bytes_,
                                 .build_ms = build_ms_,
                                 .max_sse = last_view_.max_sse * sse_factor_,
                                 .load_workers = workers_.size()};
        for (const auto tile : shown_set_)
            out.drawn_splats += tiles[tile].splat_count;
        for (const auto tile : wanted_)
            out.loading_tiles += !cache_.contains(tile) && !failed_.contains(tile);
        return out;
    }

    std::uint64_t SplatTileStreamer::usedBytesLocked() const {
        return cache_bytes_ + in_flight_bytes_ + drawn_bytes_ + building_bytes_ + (built_ ? built_bytes_ : 0);
    }

    void SplatTileStreamer::evictLocked(const std::uint64_t incoming) {
        while (usedBytesLocked() + incoming > cache_limit_bytes_) {
            auto victim = cache_.end();
            for (auto it = cache_.begin(); it != cache_.end(); ++it) {
                if (it->second.last_wanted == frame_ || std::ranges::binary_search(requested_set_, it->first))
                    continue;
                if (victim == cache_.end() || it->second.last_wanted < victim->second.last_wanted)
                    victim = it;
            }
            if (victim == cache_.end())
                return;
            cache_bytes_ -= victim->second.bytes;
            cache_.erase(victim);
        }
    }

    void SplatTileStreamer::work(const std::stop_token& stop) {
        const auto tiles = source_->tiles();
        std::unique_lock lock(mutex_);
        while (!stop.stop_requested()) {
            // A completed render set is merged by one worker at a time (building_), so the
            // others keep loading tiles. The merge installs only when its generation is
            // still newer than the drawn model; a newer request started meanwhile wins, but
            // an otherwise-valid merge is never thrown away just because a finer cut was
            // requested while it ran. Pieces hold shared ownership, so the merge stays valid
            // even if a tile is evicted before it finishes.
            if (!build_request_.empty() && !building_) {
                auto set = std::move(build_request_);
                build_request_.clear();
                const auto gen = build_request_gen_;
                std::unordered_map<std::uint32_t, std::shared_ptr<const core::SplatData>> pieces;
                for (const auto tile : set)
                    if (const auto it = cache_.find(tile); it != cache_.end())
                        pieces.emplace(tile, it->second.data);
                if (pieces.size() != set.size()) {
                    requested_set_.clear(); // evicted meanwhile; ask the next selection again
                    cache_changed_ = true;
                    continue;
                }
                // The merged model is a second copy of its tiles and the drawn one stays
                // until it is replaced, so both count against the memory limit too.
                std::uint64_t merge_bytes = 0;
                for (const auto tile : set)
                    merge_bytes += tile_bytes(tiles[tile]);
                building_bytes_ = merge_bytes;
                evictLocked();
                building_ = true;
                lock.unlock();
                const auto build_start = std::chrono::steady_clock::now();
                std::unique_ptr<core::SplatData> merged;
                try {
                    // Merged straight into renderer storage: no separate migration copy.
                    merged = io::merge_splat_tiles(
                        *source_, set, [&](const std::uint32_t tile) { return pieces.at(tile).get(); }, allocator_);
                } catch (const std::exception& e) {
                    LOG_ERROR("3D Tiles: cannot prepare streamed model: {}", e.what());
                }
                pieces.clear();
                const auto build_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_start).count();
                lock.lock();
                building_ = false;
                building_bytes_ = 0;
                ++release_gen_;
                cv_.notify_all();
                build_ms_ = build_ms;
                if (merged && gen > installed_gen_) {
                    built_bytes_ = merge_bytes;
                    built_ = std::move(merged);
                    built_set_ = std::move(set);
                    installed_gen_ = gen;
                    if (wake_)
                        wake_();
                }
                continue;
            }

            // Next tile to load: the most urgent one no worker already holds or is loading,
            // releasing tiles the view no longer needs to make room. If it still does not
            // fit, the view is over budget. In-flight tiles reserve their cache bytes so
            // parallel workers do not collectively overrun the cache.
            std::uint32_t next = io::SplatTile::kNone;
            std::uint64_t next_bytes = 0;
            evictLocked();
            over_budget_ = false;
            for (const auto tile : wanted_) {
                if (cache_.contains(tile) || failed_.contains(tile) || in_flight_.contains(tile))
                    continue;
                const auto bytes = tile_bytes(tiles[tile]);
                evictLocked(bytes);
                const auto used = usedBytesLocked();
                const auto transient = building_bytes_ + (built_ ? built_bytes_ : 0);
                if (used + bytes <= cache_limit_bytes_) {
                    next = tile;
                    next_bytes = bytes;
                } else if (used - transient + bytes <= cache_limit_bytes_) {
                    // Fits once the pending merge replaces the drawn model: wait for it
                    // instead of coarsening the view over a transient peak.
                } else if (!over_budget_) {
                    over_budget_ = true;
                    if (wake_)
                        wake_(); // the next update coarsens the selection
                }
                break;
            }
            if (next == io::SplatTile::kNone) {
                const auto selection = frame_;
                const auto limit = cache_limit_bytes_;
                const auto released = release_gen_;
                cv_.wait(lock, stop, [&] {
                    return (!build_request_.empty() && !building_) || frame_ != selection ||
                           cache_limit_bytes_ != limit || release_gen_ != released;
                });
                continue;
            }

            in_flight_.insert(next);
            in_flight_bytes_ += next_bytes;
            lock.unlock();
            auto loaded = io::load_splat_tile_gpu(*source_, next);
            lock.lock();
            in_flight_.erase(next);
            in_flight_bytes_ -= next_bytes;
            if (!loaded) {
                LOG_ERROR("3D Tiles: tile {}: {}", next, loaded.error());
                failed_.insert(next);
                continue;
            }
            cache_.emplace(next,
                           CachedTile{std::make_shared<const core::SplatData>(std::move(*loaded)), next_bytes, frame_});
            cache_bytes_ += next_bytes;
            cache_changed_ = true;
            if (wake_)
                wake_();
        }
    }

} // namespace lfs::vis
