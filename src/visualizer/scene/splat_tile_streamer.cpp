/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "splat_tile_streamer.hpp"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor_backend.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <thread>

namespace lfs::vis {

    namespace {
        constexpr float kSseFactorStep = 1.02f; // coarsen per update under memory pressure
        constexpr float kSseRelaxStep = 1.01f;  // relax per update, slower than coarsening
        constexpr double kRelaxHeadroom = 0.9;  // the finer cut must fit in this share of the limit
        constexpr float kMaxSseFactor = 64.0f;
        // A failed merge (almost always out of GPU memory) retries a coarser cut a few
        // times, then waits for the view or the cache size to change.
        constexpr int kMaxMergeAttempts = 3;
        constexpr float kMergeFailureSseStep = 1.5f;

        std::uint64_t tile_bytes(const io::SplatTile& tile) {
            const int rest = (tile.sh_degree + 1) * (tile.sh_degree + 1) - 1;
            return tile.splat_count * (14 + 3 * rest) * sizeof(float);
        }

        // Memory the cut at `sse_factor` needs once settled: every tile its traversal wants
        // (cached and not evictable while wanted) plus the merged model drawn from its render
        // tiles. The same terms the workers weigh when they report the view over budget.
        std::uint64_t cut_bytes(const io::SplatTileSource& source, io::SplatTileView view, const float sse_factor) {
            view.max_sse *= sse_factor;
            const auto selection = io::select_splat_tiles(source, view, [](std::uint32_t) { return true; });
            const auto tiles = source.tiles();
            std::uint64_t bytes = 0;
            for (const auto tile : selection.wanted)
                bytes += tile_bytes(tiles[tile]);
            for (const auto tile : selection.render)
                bytes += tile_bytes(tiles[tile]);
            return bytes;
        }

        // Whether the GPU can hold the merged model of `set` right now. The drawn model and
        // the cache are already allocated, so only the new model and the float SH workspace
        // a quantized merge encodes from are planned; the coordinator adds reclaimable
        // memory and the application-wide safety reserve.
        bool merge_fits(const std::span<const io::SplatTile> tiles, const std::vector<std::uint32_t>& set) {
            std::size_t splats = 0;
            int sh_degree = 0;
            for (const auto tile : set) {
                splats += tiles[tile].splat_count;
                sh_degree = std::max(sh_degree, tiles[tile].sh_degree);
            }
            std::size_t workspace = 0;
            if (sh_degree > 0 && core::sh_value_quant::enabled())
                workspace = core::sh_swizzled_float_count(splats, core::sh_rest_coefficients_for_degree(sh_degree)) *
                            sizeof(float);
            const core::OperationMemoryPlan plan{
                .operation = "3D Tiles merge",
                .persistent_device_bytes = core::SplatExportableStorage::layoutBytes(splats, sh_degree),
                .temporary_device_bytes = workspace};
            return core::MemoryPressureCoordinator::instance()
                .preflight(plan, core::device_memory_domain(core::default_gpu_backend()))
                .ok;
        }

        bool same_view(const io::SplatTileView& a, const io::SplatTileView& b) {
            return a.camera == b.camera && a.planes == b.planes && a.sse_per_error == b.sse_per_error &&
                   a.max_sse == b.max_sse && a.orthographic == b.orthographic;
        }
    } // namespace

    std::size_t auto_tile_load_workers() {
        const unsigned cores = std::thread::hardware_concurrency();
        return std::clamp<std::size_t>(cores / 4, 2, 4);
    }

    bool splat_tile_budget_allows_load(const SplatTileBudgetUsage& usage,
                                       const std::uint64_t incoming_bytes) {
        if (usage.minimum_cut)
            return true;
        const auto model_bytes = usage.replacing_drawn ? usage.replacement_bytes : usage.drawn_bytes;
        std::uint64_t remaining = usage.limit_bytes;
        for (const auto bytes : {usage.cached_bytes, usage.in_flight_bytes, model_bytes, incoming_bytes}) {
            if (bytes > remaining)
                return false;
            remaining -= bytes;
        }
        return true;
    }

    namespace {
        // Workers of every streamer, counted from launch until their last reference to the
        // streamer is gone, so shutdown can wait for retired ones.
        std::mutex live_workers_mutex;
        std::condition_variable live_workers_cv;
        std::size_t live_workers = 0;
    } // namespace

    void wait_for_retired_tile_workers() {
        std::unique_lock lock(live_workers_mutex);
        live_workers_cv.wait(lock, [] { return live_workers == 0; });
    }

    SplatTileStreamer::Handle SplatTileStreamer::create(std::shared_ptr<const io::SplatTileSource> source,
                                                        core::SplatTensorAllocator allocator) {
        std::shared_ptr<SplatTileStreamer> streamer(new SplatTileStreamer(std::move(source), std::move(allocator)));
        streamer->self_ = streamer;
        return Handle(streamer.get());
    }

    void SplatTileStreamer::Retire::operator()(SplatTileStreamer* const streamer) const {
        {
            // Workers must not call back into the viewer once the owner let go.
            std::lock_guard lock(streamer->mutex_);
            streamer->wake_ = nullptr;
        }
        for (auto& worker : streamer->workers_) {
            worker.request_stop();
            worker.detach();
        }
        streamer->workers_.clear();
        streamer->cv_.notify_all();
        // Frees the streamer now when no worker still holds it.
        const auto self = std::move(streamer->self_);
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
        // Workers start on the first update(), once self_ exists for them to hold.
    }

    // Runs on the last thread holding the streamer: the retiring owner, or the last
    // worker to exit. Workers are already detached by then.
    SplatTileStreamer::~SplatTileStreamer() = default;

    void SplatTileStreamer::resizeWorkers(std::size_t count) {
        count = std::max<std::size_t>(count, 1);
        if (count == workers_.size() || !self_)
            return;
        if (count < workers_.size()) {
            // Retired workers finish their current job on their own; nobody waits for them.
            for (std::size_t i = count; i < workers_.size(); ++i) {
                workers_[i].request_stop();
                workers_[i].detach();
            }
            workers_.resize(count);
            cv_.notify_all();
        } else {
            while (workers_.size() < count) {
                {
                    std::lock_guard lock(live_workers_mutex);
                    ++live_workers;
                }
                workers_.emplace_back([self = self_](const std::stop_token& stop) mutable {
                    self->work(stop);
                    self.reset(); // may free the streamer; counted as live until then
                    std::lock_guard lock(live_workers_mutex);
                    --live_workers;
                    live_workers_cv.notify_all();
                });
            }
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
            minimum_cut_ = false;
            requested_set_.clear();
            build_request_.clear();
            built_.reset();
            built_set_.clear();
            built_bytes_ = 0;
            installed_gen_ = ++build_gen_; // invalidate a merge that started under the old budget
            merge_failures_ = 0;
            cache_changed_ = true;
            cv_.notify_all();
        }
        if (!same_view(view, last_view_))
            merge_failures_ = 0;
        // Memory-adjusted error: when the view's tiles do not fit the cache, accept a
        // slightly larger error each frame. Relax it, more slowly, only when the finer cut
        // is estimated to fit with headroom: coarsening at the limit and relaxing below it
        // keeps the cut from flipping between a level that fits and one that does not.
        bool factor_changed = false;
        if (over_budget_ && sse_factor_ < kMaxSseFactor) {
            sse_factor_ = std::min(sse_factor_ * kSseFactorStep, kMaxSseFactor);
            factor_changed = true;
        } else if (over_budget_ && !minimum_cut_) {
            // A finite SSE cap can still refine very high-error tiles. Infinity is the
            // actual coarsest cut and, as the last usable representation, is allowed even
            // when its settled footprint exceeds the configured cache limit.
            minimum_cut_ = true;
            factor_changed = true;
        } else if (!over_budget_ && minimum_cut_) {
            auto finest_bounded = view;
            finest_bounded.max_sse *= kMaxSseFactor;
            if (static_cast<double>(cut_bytes(*source_, finest_bounded, 1.0f)) <=
                kRelaxHeadroom * static_cast<double>(cache_limit_bytes_)) {
                minimum_cut_ = false;
                factor_changed = true;
            }
        } else if (!over_budget_ && sse_factor_ > 1.0f) {
            const float finer = std::max(sse_factor_ / kSseRelaxStep, 1.0f);
            if (static_cast<double>(cut_bytes(*source_, view, finer)) <=
                kRelaxHeadroom * static_cast<double>(cache_limit_bytes_)) {
                sse_factor_ = finer;
                factor_changed = true;
            }
        }
        cache_changed_ = cache_changed_ || factor_changed;
        if (!settings.freeze && (cache_changed_ || !same_view(view, last_view_))) {
            cache_changed_ = false;
            last_view_ = view;
            ++frame_;
            auto adjusted = view;
            adjusted.max_sse = minimum_cut_ ? std::numeric_limits<float>::infinity()
                                            : adjusted.max_sse * sse_factor_;
            auto ideal = io::select_splat_tiles(*source_, adjusted, [](std::uint32_t) { return true; });
            std::ranges::sort(ideal.render);
            replacement_bytes_ = 0;
            for (const auto tile : ideal.render)
                replacement_bytes_ += tile_bytes(source_->tiles()[tile]);
            replacing_drawn_ = ideal.render != shown_set_;
            auto selection = io::select_splat_tiles(*source_, adjusted, [this](const std::uint32_t tile) {
                return cache_.contains(tile);
            });
            if (factor_changed && wake_)
                wake_(); // keep adjusting until the factor settles
            wanted_ = std::move(selection.wanted);
            for (const auto tile : wanted_)
                if (const auto it = cache_.find(tile); it != cache_.end())
                    it->second.last_wanted = frame_;
            std::ranges::sort(selection.render);
            // An incomplete cut would open holes; keep showing the current one until it fills in.
            if (selection.complete && !selection.render.empty() && selection.render != requested_set_ &&
                merge_failures_ < kMaxMergeAttempts) {
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
                                 .skipped_contents = source_->skipped_contents,
                                 .full_detail_splats = full_detail_splats_,
                                 .cache_bytes = cache_bytes_,
                                 .drawn_bytes = drawn_bytes_ + building_bytes_ + (built_ ? built_bytes_ : 0),
                                 .cache_limit_bytes = cache_limit_bytes_,
                                 .gpu_total_bytes = gpu_total_bytes_,
                                 .build_ms = build_ms_,
                                 .max_sse = minimum_cut_ ? std::numeric_limits<float>::infinity()
                                                         : last_view_.max_sse * sse_factor_,
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

    SplatTileBudgetUsage SplatTileStreamer::budgetUsageLocked(const bool minimum_cut) const {
        return SplatTileBudgetUsage{
            .cached_bytes = cache_bytes_,
            .in_flight_bytes = in_flight_bytes_,
            .drawn_bytes = drawn_bytes_,
            .replacement_bytes = std::max({replacement_bytes_, building_bytes_, built_ ? built_bytes_ : 0}),
            .limit_bytes = cache_limit_bytes_,
            .replacing_drawn = replacing_drawn_,
            .minimum_cut = minimum_cut,
        };
    }

    void SplatTileStreamer::evictLocked(const std::uint64_t incoming) {
        // The requested cut pins its tiles only until it is merged; once drawn (or built and
        // awaiting the swap) the model is its own copy and the tiles are ordinary cache.
        // Pinning a drawn cut would block the next cut's tiles from ever loading.
        const bool request_pending =
            requested_set_ != shown_set_ && !(built_ && requested_set_ == built_set_);
        while (!splat_tile_budget_allows_load(budgetUsageLocked(false), incoming)) {
            auto victim = cache_.end();
            for (auto it = cache_.begin(); it != cache_.end(); ++it) {
                if (it->second.last_wanted == frame_ ||
                    (request_pending && std::ranges::binary_search(requested_set_, it->first)))
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
        // A merge that cannot run (no GPU room) or failed (almost always out of memory):
        // forget the cut so the next update asks again, a little coarser.
        const auto back_off_merge = [&] {
            if (++merge_failures_ >= kMaxMergeAttempts)
                LOG_WARN("3D Tiles: {} merges failed in a row; waiting for the view or cache size to change",
                         merge_failures_);
            requested_set_.clear();
            sse_factor_ = std::min(sse_factor_ * kMergeFailureSseStep, kMaxSseFactor);
            cache_changed_ = true;
            if (wake_)
                wake_();
        };
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
                if (stop.stop_requested())
                    break; // retired: a merge would only allocate a model nobody shows
                building_bytes_ = merge_bytes;
                evictLocked();
                // The budget may be briefly exceeded while the old model is still drawn,
                // but never beyond what the GPU can actually hold.
                if (!merge_fits(tiles, set)) {
                    LOG_DEBUG("3D Tiles: not enough free GPU memory to merge {} tiles", set.size());
                    building_bytes_ = 0;
                    back_off_merge();
                    continue;
                }
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
                } catch (...) {
                    LOG_ERROR("3D Tiles: cannot prepare streamed model: unknown error");
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
                if (!merged) {
                    back_off_merge();
                } else if (gen > installed_gen_) {
                    merge_failures_ = 0;
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
                if (splat_tile_budget_allows_load(budgetUsageLocked(minimum_cut_), bytes)) {
                    next = tile;
                    next_bytes = bytes;
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
            // GPU allocation, upload and the placement transform can throw; an exception
            // leaving the worker would end the app, so it fails just this tile.
            auto loaded = [&]() -> lfs::Result<core::SplatData> {
                try {
                    return io::load_splat_tile_gpu(*source_, next);
                } catch (const lfs::Exception& e) {
                    return e.error();
                } catch (const std::exception& e) {
                    return lfs::make_error(lfs::ErrorInit{
                        .code = lfs::ErrorCode::Internal,
                        .domain = lfs::ErrorDomain::Rendering,
                        .detail = e.what(),
                        .detection = LFS_SOURCE_SITE_CURRENT(),
                        .fields = lfs::SmallFields{}.add("tile", static_cast<std::uint64_t>(next)),
                    });
                } catch (...) {
                    return lfs::make_error(lfs::ErrorInit{
                        .code = lfs::ErrorCode::Internal,
                        .domain = lfs::ErrorDomain::Rendering,
                        .detail = "Unknown failure while loading a 3D Tiles tile",
                        .detection = LFS_SOURCE_SITE_CURRENT(),
                        .fields = lfs::SmallFields{}.add("tile", static_cast<std::uint64_t>(next)),
                    });
                }
            }();
            lock.lock();
            in_flight_.erase(next);
            in_flight_bytes_ -= next_bytes;
            if (stop.stop_requested())
                break; // retired while loading: drop the tile instead of caching it
            if (!loaded) {
                LOG_ERROR("3D Tiles: tile {}: {}", next, lfs::format_for_developer(loaded.error()));
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
