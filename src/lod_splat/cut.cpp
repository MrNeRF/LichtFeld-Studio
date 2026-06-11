/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "cut.hpp"

#include <algorithm>
#include <queue>

namespace lfs::lod {

    CutSelector::CutSelector(const Hierarchy& h, const CutParams& p) : h_(h), params_(p) {
        const size_t n = h.nodes.size();
        in_cut_.assign(n, 0);
        kids_in_cut_.assign(n, 0);
        parent_done_.assign(n, 0);
        cut_.push_back(0);
        in_cut_[0] = 1;
    }

    float CutSelector::nodeError(const Node& n, const Camera& cam) const {
        const float d = std::max(n.bounds.distance(cam.position), cam.znear);
        const float projected_px = n.rep_extent * cam.focalPx() / d;
        // target spacing = pixel footprint / sqrt(overdraw quality)
        return projected_px * std::sqrt(params_.quality) / params_.tau_px;
    }

    float CutSelector::nodePriority(const Node& n, float eps, const Camera& cam) const {
        const float d = std::max(n.bounds.distance(cam.position), cam.znear);
        const float diag_px = n.bounds.extent().norm() * cam.focalPx() / d;
        const float screen_px = (float)cam.width * (float)cam.height;
        const float coverage_px = std::min(0.785f * diag_px * diag_px, screen_px);
        return eps * std::sqrt(coverage_px + 1.f);
    }

    uint64_t CutSelector::collectEmission(const Camera& cam, ResidencyProvider* residency,
                                          std::vector<DrawEntry>* draw, CutStats* stats) {
        const Frustum fr = Frustum::fromCamera(cam);
        const float band = params_.dissolve_band;
        uint64_t emitted = 0;
        scratch_nodes_.clear(); // parents marked done, for reset

        auto emit = [&](uint32_t node, float w) {
            const Node& n = h_.nodes[node];
            emitted += n.rep_count;
            if (draw)
                draw->push_back({node, n.rep_offset, n.rep_count, w, n.is_leaf});
        };

        for (uint32_t u : cut_) {
            if (residency) {
                residency->touch(u);
                // keep the immediate fallback chain warm: coarsening or dissolving
                // must never wait on a fetch
                if (h_.nodes[u].parent != kInvalidNode)
                    residency->touch(h_.nodes[u].parent);
            }
            if (!fr.intersects(h_.nodes[u].bounds))
                continue;
            const Node& n = h_.nodes[u];
            if (stats) {
                stats->visible_nodes++;
                if (!n.is_leaf) {
                    const float e = nodeError(n, cam);
                    if (e > 1.f)
                        stats->max_unresolved_error = std::max(stats->max_unresolved_error, e);
                }
            }
            float w = 1.f;
            if (band > 0.f && n.parent != kInvalidNode) {
                const uint32_t p = n.parent;
                const float t = std::clamp((nodeError(h_.nodes[p], cam) - 1.f) / band, 0.f, 1.f);
                const bool p_resident = !residency || residency->isResident(p);
                if (p_resident) {
                    w = t;
                    if (t < 1.f && !parent_done_[p]) {
                        parent_done_[p] = 1;
                        scratch_nodes_.push_back(p);
                        emit(p, 1.f - t);
                        if (residency)
                            residency->touch(p);
                        if (stats)
                            stats->band_parents++;
                    }
                } else if (t < 1.f && residency) {
                    residency->request(p, 1e6f); // dissolve wants the parent back
                }
            }
            if (w > 0.f)
                emit(u, w);
        }
        for (uint32_t p : scratch_nodes_)
            parent_done_[p] = 0;
        return emitted;
    }

    void CutSelector::update(const Camera& cam, ResidencyProvider* residency,
                             std::vector<DrawEntry>& draw, CutStats& stats) {
        const double t0 = nowMs();
        stats = CutStats{};
        draw.clear();

        const Frustum fr = Frustum::fromCamera(cam);
        auto vis = [&](uint32_t i) { return fr.intersects(h_.nodes[i].bounds); };
        auto resident = [&](uint32_t node) { return !residency || residency->isResident(node); };

        // ---- 1. coarsen pass: merge sibling groups whose parent is already fine
        // enough (hysteresis band) or fully outside the frustum. With the dissolve
        // these switches happen at child weight ~0, so they are visually free. ----
        const float coarsen_eps = 1.f / std::max(params_.hysteresis, 1.f);
        std::vector<uint32_t> touched, added;
        for (int guard = 0; guard < 16; ++guard) {
            touched.clear();
            added.clear();
            for (uint32_t u : cut_) {
                const uint32_t p = h_.nodes[u].parent;
                if (p == kInvalidNode)
                    continue;
                if (kids_in_cut_[p]++ == 0)
                    touched.push_back(p);
            }
            bool changed = false;
            for (uint32_t p : touched) {
                const Node& pn = h_.nodes[p];
                const bool complete = (kids_in_cut_[p] == pn.child_count);
                kids_in_cut_[p] = 0; // reset scratch as we go
                if (!complete)
                    continue;
                const bool pvis = vis(p);
                if (pvis && nodeError(pn, cam) >= coarsen_eps)
                    continue;
                if (pvis)
                    stats.coarsens_threshold++;
                if (!resident(p)) {
                    if (residency)
                        residency->request(p, 1e6f); // must come back before children evict
                    continue;
                }
                for (int c = 0; c < pn.child_count; ++c)
                    in_cut_[pn.children[c]] = 0;
                in_cut_[p] = 1;
                added.push_back(p);
                stats.coarsens++;
                changed = true;
            }
            if (changed) {
                std::vector<uint32_t> nc;
                nc.reserve(cut_.size());
                for (uint32_t u : cut_)
                    if (in_cut_[u])
                        nc.push_back(u);
                for (uint32_t u : added)
                    if (in_cut_[u])
                        nc.push_back(u);
                cut_.swap(nc);
            } else {
                break;
            }
        }

        // ---- 2. current budget usage = exactly what the dissolve will emit ----
        uint64_t used = collectEmission(cam, nullptr, nullptr, nullptr);

        // ---- 3. greedy refine under budget: worst screen-space error first ----
        struct HeapItem {
            float prio, eps;
            uint32_t node;
            bool operator<(const HeapItem& o) const { return prio < o.prio; }
        };
        std::priority_queue<HeapItem> heap;
        for (uint32_t u : cut_) {
            const Node& n = h_.nodes[u];
            if (n.is_leaf || !vis(u))
                continue;
            const float e = nodeError(n, cam);
            if (e > 1.f)
                heap.push({nodePriority(n, e, cam), e, u});
        }
        added.clear();
        while (!heap.empty()) {
            const HeapItem it = heap.top();
            heap.pop();
            if (!in_cut_[it.node])
                continue;
            const Node& n = h_.nodes[it.node];

            uint64_t visible_child_splats = 0;
            for (int c = 0; c < n.child_count; ++c)
                if (vis(n.children[c]))
                    visible_child_splats += h_.nodes[n.children[c]].rep_count;
            // parent keeps rendering while inside the dissolve band
            const bool parent_in_band = params_.dissolve_band > 0.f &&
                                        it.eps < 1.f + params_.dissolve_band;
            const int64_t delta =
                (int64_t)visible_child_splats - (parent_in_band ? 0 : (int64_t)n.rep_count);
            if ((int64_t)used + delta > (int64_t)params_.splat_budget) {
                stats.budget_saturated = true;
                stats.max_unresolved_error = std::max(stats.max_unresolved_error, it.eps);
                continue; // a cheaper refine may still fit
            }
            if (residency) {
                bool all = true;
                for (int c = 0; c < n.child_count; ++c)
                    if (!residency->isResident(n.children[c])) {
                        residency->request(n.children[c], it.prio);
                        all = false;
                    }
                if (!all) {
                    stats.blocked_by_residency++;
                    stats.max_unresolved_error = std::max(stats.max_unresolved_error, it.eps);
                    continue; // parent keeps rendering — no holes
                }
            }
            // apply
            in_cut_[it.node] = 0;
            stats.refines++;
            used += delta;
            for (int c = 0; c < n.child_count; ++c) {
                const uint32_t ci = n.children[c];
                in_cut_[ci] = 1;
                added.push_back(ci);
                const Node& cn = h_.nodes[ci];
                if (vis(ci) && !cn.is_leaf) {
                    const float e = nodeError(cn, cam);
                    if (e > 1.f)
                        heap.push({nodePriority(cn, e, cam), e, ci});
                }
            }
        }
        {
            std::vector<uint32_t> nc;
            nc.reserve(cut_.size() + added.size());
            for (uint32_t u : cut_)
                if (in_cut_[u])
                    nc.push_back(u);
            for (uint32_t u : added)
                if (in_cut_[u])
                    nc.push_back(u);
            cut_.swap(nc);
        }

        // ---- 4. force-coarsen if still over budget (budget drop / sudden
        // visibility increase): merge sibling groups cheapest-first, cascading
        // upward through a min-heap, until the working set fits ----
        if (used > params_.splat_budget) {
            struct CoarsenItem {
                float prio;
                uint32_t node;
                bool operator<(const CoarsenItem& o) const { return prio > o.prio; } // min-heap
            };
            std::priority_queue<CoarsenItem> cheap;
            touched.clear();
            for (uint32_t u : cut_) {
                const uint32_t p = h_.nodes[u].parent;
                if (p == kInvalidNode)
                    continue;
                if (kids_in_cut_[p]++ == 0)
                    touched.push_back(p);
            }
            for (uint32_t p : touched) {
                if (kids_in_cut_[p] != h_.nodes[p].child_count)
                    continue;
                if (!resident(p)) {
                    // over budget and the fallback parent was evicted: bring it
                    // back at top priority or the cut can never shrink
                    if (residency)
                        residency->request(p, 1e7f);
                    continue;
                }
                cheap.push({nodePriority(h_.nodes[p], nodeError(h_.nodes[p], cam), cam), p});
            }
            std::vector<uint32_t> new_parents;
            while (used > params_.splat_budget && !cheap.empty()) {
                const CoarsenItem it = cheap.top();
                cheap.pop();
                const Node& pn = h_.nodes[it.node];
                bool ok = pn.child_count > 0;
                for (int c = 0; ok && c < pn.child_count; ++c)
                    ok = in_cut_[pn.children[c]] != 0;
                if (!ok || in_cut_[it.node])
                    continue;
                for (int c = 0; c < pn.child_count; ++c) {
                    const uint32_t ci = pn.children[c];
                    in_cut_[ci] = 0;
                    if (vis(ci))
                        used -= h_.nodes[ci].rep_count;
                }
                in_cut_[it.node] = 1;
                if (vis(it.node))
                    used += pn.rep_count;
                new_parents.push_back(it.node);
                stats.forced_coarsens++;
                const uint32_t gp = pn.parent;
                if (gp != kInvalidNode && resident(gp)) {
                    bool complete = true;
                    for (int c = 0; complete && c < h_.nodes[gp].child_count; ++c)
                        complete = in_cut_[h_.nodes[gp].children[c]] != 0;
                    if (complete)
                        cheap.push({nodePriority(h_.nodes[gp], nodeError(h_.nodes[gp], cam), cam), gp});
                }
            }
            for (uint32_t p : touched)
                kids_in_cut_[p] = 0;
            if (!new_parents.empty()) {
                std::vector<uint32_t> nc;
                nc.reserve(cut_.size());
                for (uint32_t u : cut_)
                    if (in_cut_[u])
                        nc.push_back(u);
                for (uint32_t u : new_parents)
                    if (in_cut_[u])
                        nc.push_back(u);
                cut_.swap(nc);
            }
        }

        // ---- 5. emit draw list via the dissolve rule (stateless weights) ----
        stats.cut_nodes = (uint32_t)cut_.size();
        stats.drawn_splats = collectEmission(cam, residency, &draw, &stats);
        stats.select_ms = nowMs() - t0;
    }

} // namespace lfs::lod
