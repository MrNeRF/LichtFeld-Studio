/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Per-frame LOD selection: a cut through the hierarchy maintained
// incrementally under a hard splat budget.
//
// Screen-space error of a node:  eps = rep_extent * focal_px / dist / w_target
// with w_target = tau_px / sqrt(quality) pixels — i.e. refine while the node's
// representative splats project larger than the pixel footprint divided by the
// overdraw quality knob. The metric uses the representatives' actual world
// extent (not node size), so a flat wall seen edge-on keeps the density its
// splat extents demand instead of coarsening into mush.
//
// Budget enforcement is greedy: a max-heap ordered by eps weighted by screen
// coverage refines the worst node first and stops at the budget, degrading
// gracefully instead of blowing past the cap.
//
// Level transitions use a continuous dissolve ported from LichtFeld's
// lod_select_threshold shader: a refined parent keeps rendering with weight
// 1-t while its children render with weight t, where t ramps over the error
// band [1, 1+dissolve_band] of the PARENT. Weights are a continuous function
// of camera pose, so switching cannot pop and needs no temporal state;
// hysteresis remains only to avoid refine/coarsen churn (the switches it
// gates happen at weight ~0 and are visually free).
#pragma once

#include "camera.hpp"
#include "hierarchy.hpp"

#include <cstdint>
#include <vector>

namespace lfs::lod {

    struct CutParams {
        float quality = 3.f;     // target overdraw factor q (contributions/pixel)
        float tau_px = 1.f;      // base screen-space error tolerance in pixels
        float hysteresis = 1.5f; // coarsen only when eps < tau/hysteresis (1 = off)
        uint64_t splat_budget = 2'000'000;
        float dissolve_band = 0.18f; // parent<->children cross-dissolve width in eps; 0 = hard switch
    };

    // Residency interface implemented by the streaming layer. A refine is applied
    // only when all children payloads are resident; otherwise the parent keeps
    // rendering (never holes out) and fetches are requested.
    struct ResidencyProvider {
        virtual ~ResidencyProvider() = default;
        virtual bool isResident(uint32_t node) const = 0;
        virtual void request(uint32_t node, float priority) = 0;
        virtual void touch(uint32_t node) = 0; // mark used this frame (LRU)
    };

    struct DrawEntry {
        uint32_t node;
        uint32_t offset, count; // into leaves or merged cloud
        float weight;           // dissolve weight in (0,1]
        bool from_leaves;
    };

    struct CutStats {
        double select_ms = 0;
        uint32_t cut_nodes = 0, visible_nodes = 0;
        uint32_t refines = 0, coarsens = 0, forced_coarsens = 0;
        uint32_t coarsens_threshold = 0;   // coarsens due to error band (vs frustum exit)
        uint32_t blocked_by_residency = 0; // refines deferred, parent kept (holes prevented)
        uint32_t band_parents = 0;         // parents co-rendered by the dissolve this frame
        uint64_t drawn_splats = 0;         // splats in draw list (incl. dissolving parents)
        bool budget_saturated = false;
        float max_unresolved_error = 0.f; // worst eps among visible non-leaf cut nodes
    };

    class CutSelector {
    public:
        CutSelector(const Hierarchy& h, const CutParams& p);

        void setParams(const CutParams& p) { params_ = p; }
        const CutParams& params() const { return params_; }

        // Advances the cut for this camera; fills draw list + stats.
        void update(const Camera& cam, ResidencyProvider* residency,
                    std::vector<DrawEntry>& draw, CutStats& stats);

        const std::vector<uint32_t>& cutNodes() const { return cut_; }

    private:
        float nodeError(const Node& n, const Camera& cam) const;
        float nodePriority(const Node& n, float eps, const Camera& cam) const;
        // Walks the emission rule (children at t, band parents at 1-t) over the
        // current cut; returns total emitted splats. With `draw`/`stats` non-null
        // also fills the draw list and touches residency.
        uint64_t collectEmission(const Camera& cam, ResidencyProvider* residency,
                                 std::vector<DrawEntry>* draw, CutStats* stats);

        const Hierarchy& h_;
        CutParams params_;
        std::vector<uint32_t> cut_;
        std::vector<uint8_t> in_cut_; // per node
        // per-frame scratch, persistent to avoid reallocation
        std::vector<uint8_t> kids_in_cut_;
        std::vector<uint8_t> parent_done_;
        std::vector<uint32_t> scratch_nodes_;
    };

} // namespace lfs::lod
