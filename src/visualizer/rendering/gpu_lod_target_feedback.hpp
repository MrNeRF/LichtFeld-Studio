/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "lod_page_cache.hpp"
#include "render_target_id.hpp"
#include <algorithm>
#include <cstdint>
#include <unordered_map>

namespace lfs::vis {
    struct GpuLodTargetFeedback {
        float pixel_scale_feedback = 1.0f;
        std::uint32_t frozen_frames = 0;
        std::size_t last_candidate_count = 0;
        std::size_t last_overflow_count = 0;
        std::size_t last_miss_count = 0;
        std::vector<LodPageCache::ChunkRequest> prefetch_requests;
        std::vector<std::uint32_t> protected_chunks;
        bool prefetch_valid = false;
        bool selection_active = false;
        std::size_t render_capacity_last = 0;
        std::uint64_t last_frame = 0;
    };
    class GpuLodTargetFeedbackTable {
    public:
        GpuLodTargetFeedback& touch(RenderTargetId target, std::uint64_t frame) {
            auto& state = states_[target];
            state.last_frame = frame;
            return state;
        }
        GpuLodTargetFeedback* find(RenderTargetId target) {
            auto it = states_.find(target);
            return it == states_.end() ? nullptr : &it->second;
        }
        const GpuLodTargetFeedback* find(RenderTargetId target) const {
            auto it = states_.find(target);
            return it == states_.end() ? nullptr : &it->second;
        }
        void release(RenderTargetId target) { states_.erase(target); }
        void clear() { states_.clear(); }
        [[nodiscard]] GpuLodTargetFeedback demand(std::uint64_t frame) const {
            GpuLodTargetFeedback result;
            std::unordered_map<std::uint32_t, std::uint32_t> priorities;
            for (const auto& [target, state] : states_) {
                if (frame > state.last_frame && frame - state.last_frame > 3)
                    continue;
                result.prefetch_valid |= state.prefetch_valid;
                result.protected_chunks.insert(result.protected_chunks.end(),
                                               state.protected_chunks.begin(), state.protected_chunks.end());
                for (const auto& request : state.prefetch_requests)
                    priorities[request.chunk] = std::max(priorities[request.chunk], request.priority);
            }
            std::sort(result.protected_chunks.begin(), result.protected_chunks.end());
            result.protected_chunks.erase(std::unique(result.protected_chunks.begin(),
                                                      result.protected_chunks.end()),
                                          result.protected_chunks.end());
            for (const auto& [chunk, priority] : priorities)
                result.prefetch_requests.push_back({chunk, priority});
            std::sort(result.prefetch_requests.begin(), result.prefetch_requests.end(),
                      [](const auto& a, const auto& b) { return a.priority != b.priority ? a.priority > b.priority : a.chunk < b.chunk; });
            return result;
        }

    private:
        std::unordered_map<RenderTargetId, GpuLodTargetFeedback, RenderTargetIdHash> states_;
    };
} // namespace lfs::vis
