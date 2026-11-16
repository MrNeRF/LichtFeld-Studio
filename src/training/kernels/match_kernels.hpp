/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>

// Kernels of the match strategy (strategies/match.hpp). All rows are [N] float unless noted.
namespace lfs::training::match {

    // Per-step bookkeeping after the backward, one launch per step:
    //   rendered = step_peak > 0 (the splat reached at least one pixel with a non-zero blend weight);
    //   streak = rendered ? 0 : streak + 1;  renders += rendered (when the row is given);
    //   while scoring:
    //     radius > limit:          oversize += log2(radius / limit)
    //     visible and error != 0:  score = (score n + |error|^power) / (n + 1), n += 1
    //     peak = max(peak x peak_decay, step_peak)
    // error and step_peak are cleared for the next step (visible by launch_walk).
    struct StepRows {
        float* error = nullptr;     // summed error map x blend weight of this step (backward)
        float* step_peak = nullptr; // largest blend weight of this step (backward)
        float* visible = nullptr;   // > 0 when the splat was in the view (backward)
        const float* radius = nullptr;
        float* score = nullptr;
        float* score_count = nullptr;
        float* oversize = nullptr;
        float* peak = nullptr;
        float* streak = nullptr;
        float* renders = nullptr; // optional: steps rendered since birth
    };
    struct StepSettings {
        bool scoring = true;
        float screen_limit = 0.3f;
        float score_power = 0.2f;
        float peak_decay = 1.0f;
    };
    void launch_step_update(const StepRows& rows, const StepSettings& settings, size_t n, void* stream = nullptr);

    // The peak rows of launch_step_update on their own, for any strategy that tracks the peak blend weight:
    //   renders += 1 when step_peak > 0 (when renders is given);  peak = max(peak x peak_decay, step_peak);
    //   step_peak = 0.
    void launch_peak_update(float* step_peak, float* peak, float* renders, float peak_decay, size_t n,
                            void* stream = nullptr);

    // Size clip: f = min(radius / limit, max_factor); when f > 1 every log scale drops by ln f.
    void launch_size_clip(float* log_scales, const float* radius, float limit, float max_factor, size_t n,
                          void* stream = nullptr);

    // dead[i] = 1 when opacity <= min_opacity, or peak <= dead_peak (dead_peak > 0, and renders >= peak_grace
    // when renders is given), or streak >= dead_streak (dead_streak > 0), or any parameter is not finite, or every
    // log scale is <= -40.
    // eligible[i] = 1 when the splat is not dead and its score is positive.
    struct DeadSettings {
        float min_opacity = 0.005f;
        float dead_peak = 0.0f;
        float dead_streak = 0.0f;
        float peak_grace = 0.0f;
    };
    void launch_dead_mask(const float* means, const float* log_scales, const float* raw_rotations,
                          const float* raw_opacities, const float* peak, const float* streak, const float* score,
                          const DeadSettings& settings, uint8_t* dead, uint8_t* eligible, size_t n,
                          const float* renders = nullptr, void* stream = nullptr);

    // Marks dead (dead = 1, eligible = 0) the `count` live splats (dead == 0) with renders >= grace that have the
    // lowest key (ties by row). Returns the number marked: at most count, at most the candidates.
    size_t mark_lowest_dead(const float* key, const float* renders, float grace, uint8_t* dead, uint8_t* eligible,
                            size_t n, size_t count, void* stream = nullptr);

    // Draw `count` rows without replacement, each with probability proportional to its weight
    // (exponential race: the rows with the smallest Exp(1) / weight win). The weight of row i is
    // max(score, 0), times (oversize + 0.001) when oversize is given; rows with mask[i] == 0 (when a mask is
    // given) and rows of weight 0 are never drawn. Returns the number drawn (at most count, at most the
    // rows of positive weight), written to out (int64).
    size_t draw_without_replacement(const float* score, const float* oversize, const uint8_t* mask, size_t n,
                                    size_t count, uint64_t seed, int64_t* out, void* stream = nullptr);

    // Long-axis split of src[j] into src[j] and dst[j] (src rows distinct, dst rows distinct, disjoint):
    // with a the axis of the largest scale and u its world direction, delta = 0.5 exp(s_a) u; the source
    // moves to mean - delta and dst sits at mean + delta. Both: s_a += ln 0.5, the other two += ln 0.85,
    // opacity = k x opacity, peak = k x source peak, streak = 0; dst renders = 0. dst copies rotation, sh0 (3 floats),
    // score, score count and oversize of the source. SH rest and optimizer state are the caller's.
    struct SplitRows {
        float* means = nullptr;
        float* log_scales = nullptr;
        float* raw_rotations = nullptr;
        float* raw_opacities = nullptr;
        float* sh0 = nullptr;
        float* score = nullptr;
        float* score_count = nullptr;
        float* oversize = nullptr;
        float* peak = nullptr;
        float* streak = nullptr;
        float* renders = nullptr; // optional: dst renders = 0 (a new splat)
    };
    void launch_long_axis_split(const SplitRows& rows, const int64_t* src, const int64_t* dst, size_t count,
                                float opacity_share, void* stream = nullptr);

    // Position walk on every splat visible in the step: mean += k R diag(exp(s)) z, z ~ N(0, I),
    // k = min(amplitude x E(o) x (1 - g)^150, 1), E(o) = sqrt(2 ln(255 o)) (0 at o <= 1/255),
    // g = min(o, peak) when peak is given, else o. Clears visible; amplitude <= 0 moves nothing.
    void launch_walk(float* means, const float* log_scales, const float* raw_rotations, const float* raw_opacities,
                     float* visible, const float* peak, float amplitude, size_t n, uint64_t seed,
                     void* stream = nullptr);

    // Zero the first n entries of up to four rows (null rows are skipped).
    void launch_zero_rows(float* a, float* b, float* c, float* d, size_t n, void* stream = nullptr);

} // namespace lfs::training::match
