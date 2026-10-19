/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "lod_page_cache.hpp"
#include "render_target_id.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace lfs::vis {
    // Readback frames of deferred-wants-with-zero-admissions before the
    // pool counts as frozen: long enough to outlast the publish
    // protection window (3) and the eviction freshness window (12), so
    // only a genuinely unevictable pool qualifies.
    constexpr std::uint32_t kLodAdmissionFrozenFrames = 30u;

    // One completed GPU LOD selection, as the budget-fill controller sees it.
    struct GpuLodFillSample {
        std::size_t candidates = 0, capacity = 0, overflow = 0;
        // Same-frame repair multiplier the selector applied to fit the budget.
        float threshold_scale = 1.0f;
        // Misses or streaming in flight; admission frozen for kLodAdmissionFrozenFrames.
        bool converging = false, admission_frozen = false;
    };

    // Advances the pixel-scale feedback (the multiplier on the next frame's
    // limit) by one selection, shared by every GPU LOD viewer. Returns the
    // feedback after same-frame repair, before the gradual step.
    inline float advanceGpuLodPixelScaleFeedback(float& feedback, const GpuLodFillSample& sample) {
        // Start the next frame at the threshold that actually fit. The
        // gradual controller can refine it, but cannot undo same-frame repair.
        if (std::isfinite(sample.threshold_scale) && sample.threshold_scale > 1.0f)
            feedback = std::min(64.0f, feedback * sample.threshold_scale);
        const float repaired = feedback;
        if (sample.capacity == 0)
            return repaired;
        // Damped threshold controller with a wide deadband; the per-frame
        // rate clamp keeps the limit (and with it the global transition
        // band and the touch set) stable at rest.
        const bool overflowed = sample.overflow > 0 || sample.candidates > sample.capacity;
        const double fill_ratio = static_cast<double>(sample.candidates) / static_cast<double>(sample.capacity);
        float target = feedback;
        if (overflowed) {
            target *= static_cast<float>(
                std::clamp(std::pow(std::max(fill_ratio / 0.97, 1.0), 0.85), 1.02, 1.15));
        } else if (fill_ratio < 0.85 && !sample.admission_frozen) {
            // While streaming converges, parents stand in for missing
            // children and the candidate count runs low — recover
            // GENTLY rather than freezing: a coarse cut always has
            // misses in flight, so a freeze would let one transient
            // overflow ratchet the limit up with no path back down.
            // Do NOT coarsen toward "the touch set fits the pool":
            // the cut uses only a sliver of each touched chunk, so
            // that equilibrium is dozens of times coarser than the
            // splat budget. Overcommit is the designed operating
            // point — parents stand in for missing chunks and
            // priority admission allocates pages toward the gaze.
            // The one exception is a sustained admission freeze:
            // descending further only grows a want set that can
            // never be served (budget ≈ pool pathologies dig to the
            // 0.05 floor and freeze whole level bands). Halt the
            // descent there — one-sided, never force-coarsen.
            const double floor_rate = sample.converging ? 0.98 : 0.90;
            target *= static_cast<float>(
                std::clamp(std::pow(std::max(fill_ratio / 0.85, 0.05), 0.5), floor_rate, 1.0));
        }
        // The ceiling must accommodate tight budgets on km-scale
        // scenes: close-up views legitimately need the limit dozens of
        // times coarser to fit the cut, or capacity-clamp truncation
        // punches holes in the image. Recovery handles coming back.
        constexpr float kMaxRatePerFrame = 1.03f;
        feedback = std::clamp(std::clamp(target, feedback / kMaxRatePerFrame, feedback * kMaxRatePerFrame),
                              0.05f, 64.0f);
        return repaired;
    }

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
