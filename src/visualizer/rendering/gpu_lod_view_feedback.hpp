/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lod_page_cache.hpp"
#include "view_output_key.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lfs::vis::detail {

    // One GPU selector sample after the tagged readback copy has completed.
    // Independent of Vulkan so host tests can feed out-of-order completions.
    struct GpuLodSelectorSample {
        float threshold_scale = 1.0f;
        std::size_t candidate_count = 0;
        std::size_t rendered_capacity = 0;
        std::size_t overflow_count = 0;
        std::vector<std::uint32_t> protected_chunks;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> miss_candidates;
    };

    struct GpuLodViewDemand {
        std::vector<LodPageCache::ChunkRequest> prefetch_requests;
        std::vector<std::uint32_t> protected_chunks;
        bool valid = false;
    };

    struct GpuLodViewFeedback {
        float pixel_scale_feedback = 1.0f;
        std::uint32_t frozen_frames = 0;
        std::size_t last_candidate_count = 0;
        std::size_t last_overflow_count = 0;
        std::size_t last_miss_count = 0;
        std::size_t render_capacity = 0;
        bool selection_active = false;
        std::uint64_t model_generation = 0;
        std::uint64_t tree_generation = 0;
        GpuLodViewDemand demand;
    };

    // Per-view LOD feedback, tagged by key and scene/tree generation.
    // Live demand is combined for shared page-cache scheduling.
    class GpuLodViewFeedbackTable {
    public:
        void reset() {
            views_.clear();
            visible_keys_.clear();
            visible_filter_active_ = false;
            last_rendered_key_ = kInvalidViewOutputKey;
            model_generation_ = 0;
            tree_generation_ = 0;
        }

        void setSceneGeneration(const std::uint64_t model_generation,
                                const std::uint64_t tree_generation) {
            if (model_generation_ == model_generation && tree_generation_ == tree_generation) {
                return;
            }
            views_.clear();
            visible_keys_.clear();
            visible_filter_active_ = false;
            last_rendered_key_ = kInvalidViewOutputKey;
            model_generation_ = model_generation;
            tree_generation_ = tree_generation;
        }

        [[nodiscard]] std::uint64_t modelGeneration() const noexcept { return model_generation_; }
        [[nodiscard]] std::uint64_t treeGeneration() const noexcept { return tree_generation_; }

        void retire(const ViewOutputKey key) {
            views_.erase(key);
            if (last_rendered_key_ == key) {
                last_rendered_key_ = kInvalidViewOutputKey;
            }
        }

        // Hiding a workspace pane must preserve its output/camera state while
        // excluding its stale demand from shared page-cache scheduling.
        void setVisibleKeys(const std::span<const ViewOutputKey> keys) {
            visible_keys_.clear();
            visible_keys_.insert(keys.begin(), keys.end());
            visible_filter_active_ = true;
        }

        void clearVisibleKeys() {
            visible_keys_.clear();
            visible_filter_active_ = false;
        }

        void noteLiveRender(const ViewOutputKey key,
                            const std::size_t render_capacity,
                            const bool selection_active) {
            if (!key.valid()) {
                return;
            }
            auto& view = views_[key];
            view.render_capacity = render_capacity;
            view.selection_active = selection_active;
            view.model_generation = model_generation_;
            view.tree_generation = tree_generation_;
            last_rendered_key_ = key;
        }

        // Route a tagged completion to its originating key. Late results for a
        // retired key or a previous scene/tree generation are discarded.
        [[nodiscard]] bool applyTaggedSample(const ViewOutputKey key,
                                             const std::uint64_t model_generation,
                                             const std::uint64_t tree_generation,
                                             const GpuLodSelectorSample& sample) {
            if (!key.valid()) {
                return false;
            }
            if (model_generation != model_generation_ || tree_generation != tree_generation_) {
                return false;
            }
            const auto found = views_.find(key);
            if (found == views_.end()) {
                return false;
            }
            auto& view = found->second;
            if (std::isfinite(sample.threshold_scale) && sample.threshold_scale > 1.0f) {
                view.pixel_scale_feedback = std::min(
                    64.0f, view.pixel_scale_feedback * sample.threshold_scale);
            }
            view.last_candidate_count = sample.candidate_count;
            view.last_overflow_count = sample.overflow_count;
            view.model_generation = model_generation;
            view.tree_generation = tree_generation;

            GpuLodViewDemand demand;
            demand.protected_chunks = sample.protected_chunks;
            auto miss_candidates = sample.miss_candidates;
            std::sort(miss_candidates.begin(),
                      miss_candidates.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            demand.prefetch_requests.reserve(miss_candidates.size());
            for (const auto& [chunk, priority] : miss_candidates) {
                demand.prefetch_requests.push_back({.chunk = chunk, .priority = priority});
            }
            demand.valid = true;
            view.last_miss_count = demand.prefetch_requests.size();
            view.demand = std::move(demand);
            return true;
        }

        [[nodiscard]] const GpuLodViewFeedback* find(const ViewOutputKey key) const {
            const auto found = views_.find(key);
            return found == views_.end() ? nullptr : &found->second;
        }

        [[nodiscard]] GpuLodViewFeedback* find(const ViewOutputKey key) {
            const auto found = views_.find(key);
            return found == views_.end() ? nullptr : &found->second;
        }

        [[nodiscard]] ViewOutputKey lastRenderedKey() const noexcept { return last_rendered_key_; }

        [[nodiscard]] bool anyBelowFrozenWindow(const std::uint32_t frozen_window) const {
            if (views_.empty()) {
                return true;
            }
            for (const auto& [unused_key, view] : views_) {
                (void)unused_key;
                if (view.frozen_frames < frozen_window) {
                    return true;
                }
            }
            return false;
        }

        // Union of every live view's most-recent demand. Same chunk keeps the
        // higher miss priority. The frontier is never truncated here.
        [[nodiscard]] GpuLodViewDemand unionLiveDemand() const {
            GpuLodViewDemand out;
            std::unordered_map<std::uint32_t, std::uint32_t> miss_priority;
            std::vector<std::uint32_t> protected_all;
            for (const auto& [key, view] : views_) {
                if (visible_filter_active_ && !visible_keys_.contains(key)) {
                    continue;
                }
                if (!view.demand.valid) {
                    continue;
                }
                out.valid = true;
                protected_all.insert(protected_all.end(),
                                     view.demand.protected_chunks.begin(),
                                     view.demand.protected_chunks.end());
                for (const auto& request : view.demand.prefetch_requests) {
                    auto [it, inserted] = miss_priority.try_emplace(request.chunk, request.priority);
                    if (!inserted) {
                        it->second = std::max(it->second, request.priority);
                    }
                }
            }
            std::sort(protected_all.begin(), protected_all.end());
            protected_all.erase(std::unique(protected_all.begin(), protected_all.end()),
                                protected_all.end());
            out.protected_chunks = std::move(protected_all);
            out.prefetch_requests.reserve(miss_priority.size());
            for (const auto& [chunk, priority] : miss_priority) {
                out.prefetch_requests.push_back({.chunk = chunk, .priority = priority});
            }
            std::sort(out.prefetch_requests.begin(),
                      out.prefetch_requests.end(),
                      [](const LodPageCache::ChunkRequest& a,
                         const LodPageCache::ChunkRequest& b) { return a.priority > b.priority; });
            return out;
        }

    private:
        std::unordered_map<ViewOutputKey, GpuLodViewFeedback, ViewOutputKeyHash> views_;
        std::unordered_set<ViewOutputKey, ViewOutputKeyHash> visible_keys_;
        bool visible_filter_active_ = false;
        ViewOutputKey last_rendered_key_{};
        std::uint64_t model_generation_ = 0;
        std::uint64_t tree_generation_ = 0;
    };

} // namespace lfs::vis::detail
