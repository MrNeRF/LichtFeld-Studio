/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "bhatt_ref.hpp"
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_map>

namespace lfs::lod {

    namespace {

        // ---- constants from src/core/bhatt_lod.cpp ----
        constexpr float kMinScale = 1e-12f;
        constexpr float kEllipsoidAreaP = 1.6075f;
        constexpr float kMinEval = 1e-18f;
        constexpr float kEpsCov = 1e-8f;
        constexpr float MERGE_BASE = 2.00f;

        constexpr int8_t kNeighborOffsets[26][3] = {
            {-1, -1, -1},
            {-1, -1, 0},
            {-1, -1, 1},
            {-1, 0, -1},
            {-1, 0, 0},
            {-1, 0, 1},
            {-1, 1, -1},
            {-1, 1, 0},
            {-1, 1, 1},
            {0, -1, -1},
            {0, -1, 0},
            {0, -1, 1},
            {0, 0, -1},
            {0, 0, 1},
            {0, 1, -1},
            {0, 1, 0},
            {0, 1, 1},
            {1, -1, -1},
            {1, -1, 0},
            {1, -1, 1},
            {1, 0, -1},
            {1, 0, 0},
            {1, 0, 1},
            {1, 1, -1},
            {1, 1, 0},
            {1, 1, 1}};

        float ellipsoidArea(float sx, float sy, float sz) {
            const float t1 = std::pow(sx * sy, kEllipsoidAreaP);
            const float t2 = std::pow(sx * sz, kEllipsoidAreaP);
            const float t3 = std::pow(sy * sz, kEllipsoidAreaP);
            return 4.f * 3.14159265358979f * std::pow((t1 + t2 + t3) / 3.f, 1.f / kEllipsoidAreaP);
        }

        float lodOpacityFactor(float opacity) {
            if (opacity > 1.f) {
                constexpr float kE = 2.718281828459045f;
                return std::sqrt(1.f + kE * std::log(opacity));
            }
            return 1.f;
        }

        // Plain-vector port of BhattLodWorkset (SH bands omitted: this module is
        // SH0-only end to end, matching the comparison's rendering path).
        struct Workset {
            std::vector<float> cx, cy, cz;
            std::vector<float> sx, sy, sz;
            std::vector<Quat> quat;
            std::vector<float> alpha; // display-space, may exceed 1 after merges
            std::vector<Vec3> color;
            std::vector<float> feature_size, area;
            std::vector<Sym3> cov;
            std::vector<float> cov_det;
            std::vector<int32_t> child_a, child_b;
            std::vector<uint8_t> is_active;
            size_t initial_count = 0, count = 0;

            void reserve(size_t n) {
                initial_count = n;
                const size_t cap = n * 2 + 8; // binary merge tree: <= 2n-1 nodes
                cx.reserve(cap);
                cy.reserve(cap);
                cz.reserve(cap);
                sx.reserve(cap);
                sy.reserve(cap);
                sz.reserve(cap);
                quat.reserve(cap);
                alpha.reserve(cap);
                color.reserve(cap);
                feature_size.reserve(cap);
                area.reserve(cap);
                cov.reserve(cap);
                cov_det.reserve(cap);
                child_a.reserve(cap);
                child_b.reserve(cap);
                is_active.reserve(cap);
            }

            size_t addRaw() {
                cx.push_back(0);
                cy.push_back(0);
                cz.push_back(0);
                sx.push_back(0);
                sy.push_back(0);
                sz.push_back(0);
                quat.push_back({});
                alpha.push_back(0);
                color.push_back({});
                feature_size.push_back(0);
                area.push_back(0);
                cov.push_back({});
                cov_det.push_back(0);
                child_a.push_back(-1);
                child_b.push_back(-1);
                is_active.push_back(1);
                return count++;
            }

            float computeFeatureSize(size_t i) const {
                return 2.f * std::max({sx[i], sy[i], sz[i]}) * lodOpacityFactor(alpha[i]);
            }

            // exp(-Bhattacharyya) * exp(-|dcolor|^2), exactly as bhatt_lod.cpp
            float similarity(size_t a, size_t b) const {
                Sym3 m;
                for (int i = 0; i < 6; ++i)
                    m.v[i] = 0.5f * (cov[a].v[i] + cov[b].v[i]);
                const float det_a = cov_det[a], det_b = cov_det[b];
                const float C00 = m.v[3] * m.v[5] - m.v[4] * m.v[4];
                const float C01 = m.v[2] * m.v[4] - m.v[1] * m.v[5];
                const float C02 = m.v[1] * m.v[4] - m.v[2] * m.v[3];
                const float C11 = m.v[0] * m.v[5] - m.v[2] * m.v[2];
                const float C12 = m.v[1] * m.v[2] - m.v[0] * m.v[4];
                const float C22 = m.v[0] * m.v[3] - m.v[1] * m.v[1];
                const float det = m.v[0] * C00 + m.v[1] * C01 + m.v[2] * C02;
                if (det <= kEpsCov || det_a <= kEpsCov || det_b <= kEpsCov || !std::isfinite(det))
                    return 0.f;
                const float inv_det = 1.f / det;
                const float dx = cx[b] - cx[a], dy = cy[b] - cy[a], dz = cz[b] - cz[a];
                const float quad = (C00 * dx * dx + C11 * dy * dy + C22 * dz * dz +
                                    2.f * (C01 * dx * dy + C02 * dx * dz + C12 * dy * dz)) *
                                   inv_det;
                const float distance = 0.125f * quad + 0.5f * std::log(det / std::sqrt(det_a * det_b));
                const float spatial = std::exp(-distance);
                const Vec3 dc = color[a] - color[b];
                const float metric = spatial * std::exp(-dc.dot(dc));
                return std::isfinite(metric) ? metric : 0.f;
            }

            size_t mergeNodes(size_t a, size_t b) {
                const size_t ni = addRaw();
                float wa = area[a] * alpha[a];
                float wb = area[b] * alpha[b];
                float total_weight = wa + wb;
                if (total_weight < 1e-30f)
                    total_weight = 1e-30f;
                wa /= total_weight;
                wb /= total_weight;

                cx[ni] = wa * cx[a] + wb * cx[b];
                cy[ni] = wa * cy[a] + wb * cy[b];
                cz[ni] = wa * cz[a] + wb * cz[b];
                color[ni] = color[a] * wa + color[b] * wb;

                Sym3 total;
                auto addCov = [&](size_t i, float w) {
                    const Vec3 d{cx[i] - cx[ni], cy[i] - cy[ni], cz[i] - cz[ni]};
                    total.addScaled(cov[i], w);
                    total.addOuterScaled(d, w);
                };
                addCov(a, wa);
                addCov(b, wb);
                total.v[0] += kEpsCov;
                total.v[3] += kEpsCov;
                total.v[5] += kEpsCov;

                Vec3 eval;
                Mat3 evec;
                eigenSym3(total, eval, evec);
                const float e0 = std::max(eval.x, kMinEval), e1 = std::max(eval.y, kMinEval),
                            e2 = std::max(eval.z, kMinEval);
                sx[ni] = std::sqrt(e0);
                sy[ni] = std::sqrt(e1);
                sz[ni] = std::sqrt(e2);
                quat[ni] = Quat::fromMat3(evec);
                cov[ni] = Sym3::fromCovariance(evec, {sx[ni], sy[ni], sz[ni]});
                cov_det[ni] = e0 * e1 * e2;

                float new_area = ellipsoidArea(sx[ni], sy[ni], sz[ni]);
                if (new_area < 1e-30f)
                    new_area = 1e-30f;
                alpha[ni] = std::clamp(total_weight / new_area, 0.000001f, 1000.f);

                child_a[ni] = (int32_t)a;
                child_b[ni] = (int32_t)b;
                feature_size[ni] = computeFeatureSize(ni);
                area[ni] = new_area;
                return ni;
            }
        };

        struct CellKey {
            int64_t x, y, z;
            bool operator==(const CellKey& o) const { return x == o.x && y == o.y && z == o.z; }
        };
        struct CellKeyHash {
            size_t operator()(const CellKey& k) const {
                uint64_t h = 0x9e3779b97f4a7c15ULL;
                h ^= (uint64_t)k.x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                h ^= (uint64_t)k.y + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                h ^= (uint64_t)k.z + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                return (size_t)h;
            }
        };
        using CellMap = std::unordered_map<CellKey, std::vector<uint32_t>, CellKeyHash>;

        CellKey cellFor(const Workset& ws, size_t i, float step) {
            return {(int64_t)std::floor(ws.cx[i] / step), (int64_t)std::floor(ws.cy[i] / step),
                    (int64_t)std::floor(ws.cz[i] / step)};
        }

        void eraseFromCell(CellMap& cells, const CellKey& key, uint32_t idx) {
            auto it = cells.find(key);
            if (it == cells.end())
                return;
            auto& v = it->second;
            v.erase(std::remove(v.begin(), v.end(), idx), v.end());
            if (v.empty())
                cells.erase(it);
        }

        struct ActiveEntry {
            float neg_feature_size;
            uint32_t index;
            bool operator<(const ActiveEntry& o) const {
                if (neg_feature_size != o.neg_feature_size)
                    return neg_feature_size < o.neg_feature_size;
                return index < o.index;
            }
        };

    } // namespace

    void buildBhattRef(const SplatCloud& input, float lod_base, BhattRefTree& out) {
        const double t0 = nowMs();
        const size_t n = input.size();
        Workset ws;
        ws.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const size_t k = ws.addRaw();
            ws.cx[k] = input.means[i].x;
            ws.cy[k] = input.means[i].y;
            ws.cz[k] = input.means[i].z;
            const Vec3 s = input.linearScale(i);
            ws.sx[k] = std::max(s.x, kMinScale);
            ws.sy[k] = std::max(s.y, kMinScale);
            ws.sz[k] = std::max(s.z, kMinScale);
            ws.quat[k] = input.rotations[i].normalized();
            ws.alpha[k] = input.opacity(i);
            ws.color[k] = input.color(i);
            ws.cov[k] = Sym3::fromCovariance(ws.quat[k].toMat3(), {ws.sx[k], ws.sy[k], ws.sz[k]});
            ws.cov_det[k] = detSym3(ws.cov[k]);
            ws.feature_size[k] = ws.computeFeatureSize(k);
            ws.area[k] = ellipsoidArea(ws.sx[k], ws.sy[k], ws.sz[k]);
        }

        // ---- level-doubling greedy merge loop (verbatim control flow) ----
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), (size_t)0);
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return ws.feature_size[a] < ws.feature_size[b]; });
        const float min_fs = ws.feature_size[order[0]];
        int level = (int)std::ceil(std::log(std::max(min_fs, 1e-12f)) / std::log(MERGE_BASE));

        std::vector<ActiveEntry> active_entries;
        size_t frontier = 0;
        size_t total_merges = 0;
        for (;; ++level) {
            const float step = std::pow(MERGE_BASE, (float)level);
            while (frontier < order.size() && ws.feature_size[order[frontier]] <= step) {
                const auto idx = (uint32_t)order[frontier];
                active_entries.push_back({-ws.feature_size[idx], idx});
                ++frontier;
            }

            std::priority_queue<ActiveEntry> active(active_entries.begin(), active_entries.end());
            CellMap cells;
            cells.reserve(active_entries.size() * 2 + 1);
            for (const auto& e : active_entries)
                if (ws.is_active[e.index])
                    cells[cellFor(ws, e.index, step)].push_back(e.index);

            std::vector<ActiveEntry> next_active;
            next_active.reserve(active_entries.size());

            int neighbor_indices[26];
            std::iota(neighbor_indices, neighbor_indices + 26, 0);
            std::mt19937 rng((uint32_t)(level + 12345));
            std::shuffle(neighbor_indices, neighbor_indices + 26, rng);

            while (!active.empty()) {
                const ActiveEntry entry = active.top();
                active.pop();
                const uint32_t idx = entry.index;
                if (!ws.is_active[idx])
                    continue;

                const CellKey grid = cellFor(ws, idx, step);
                uint32_t best = UINT32_MAX;
                CellKey best_cell{};
                float best_sim = -1e30f;
                auto scanCell = [&](const CellKey& key) {
                    auto it = cells.find(key);
                    if (it == cells.end())
                        return;
                    for (uint32_t nb : it->second) {
                        if (nb == idx || !ws.is_active[nb])
                            continue;
                        const float sim = ws.similarity(idx, nb);
                        if (sim > best_sim) {
                            best_sim = sim;
                            best = nb;
                            best_cell = key;
                        }
                    }
                };
                scanCell(grid);
                for (int i = 0; i < 8; ++i) {
                    const auto& off = kNeighborOffsets[neighbor_indices[i]];
                    scanCell({grid.x + off[0], grid.y + off[1], grid.z + off[2]});
                }
                if (best == UINT32_MAX) {
                    next_active.push_back(entry);
                    continue;
                }
                const auto merged = (uint32_t)ws.mergeNodes(idx, best);
                ws.is_active[idx] = 0;
                eraseFromCell(cells, grid, idx);
                ws.is_active[best] = 0;
                eraseFromCell(cells, best_cell, best);
                ++total_merges;

                const float fs = ws.feature_size[merged];
                if (fs > step) {
                    next_active.push_back({-fs, merged});
                } else {
                    cells[cellFor(ws, merged, step)].push_back(merged);
                    active.push({-fs, merged});
                }
            }
            active_entries = std::move(next_active);
            if (frontier >= order.size() && active_entries.size() <= 1)
                break;
        }

        // ---- lod_base post-order pruning (iterative; result identical to their
        // recursive Pruner) ----
        const size_t node_count = ws.count;
        const size_t root = node_count - 1;
        std::vector<uint8_t> to_output(node_count, 0);
        for (size_t i = 0; i < ws.initial_count; ++i)
            to_output[i] = 1;
        std::vector<std::vector<uint32_t>> output_children(node_count);
        std::vector<float> subtree_size(node_count, 0.f);
        to_output[root] = 1; // forced, matching Spark

        {
            std::vector<std::pair<uint32_t, uint8_t>> stack{{(uint32_t)root, 0}};
            while (!stack.empty()) {
                auto& [idx, phase] = stack.back();
                const int32_t ca = ws.child_a[idx], cb = ws.child_b[idx];
                if (ca < 0 && cb < 0) { // raw leaf
                    subtree_size[idx] = ws.area[idx] * ws.alpha[idx];
                    to_output[idx] = 1;
                    stack.pop_back();
                    continue;
                }
                if (phase == 0) {
                    phase = 1;
                    if (ca >= 0)
                        stack.push_back({(uint32_t)ca, 0});
                    if (cb >= 0)
                        stack.push_back({(uint32_t)cb, 0});
                    continue;
                }
                const float my_size = ws.area[idx] * ws.alpha[idx];
                float max_child = -1e30f;
                std::vector<uint32_t> all;
                for (int32_t c : {ca, cb}) {
                    if (c < 0)
                        continue;
                    max_child = std::max(max_child, subtree_size[c]);
                    if (to_output[c])
                        all.push_back((uint32_t)c);
                    else
                        all.insert(all.end(), output_children[c].begin(), output_children[c].end());
                }
                // exact Pruner semantics: pre-set flags (root) survive the else
                if (my_size >= max_child * lod_base) {
                    to_output[idx] = 1;
                    subtree_size[idx] = my_size;
                } else {
                    subtree_size[idx] = max_child;
                }
                output_children[idx] = std::move(all);
                stack.pop_back();
            }
        }

        // ---- flatten to variable-arity tree (children contiguous) ----
        std::vector<uint32_t> old_indices;
        out.child_start.clear();
        out.child_count.clear();
        out.level.clear();
        {
            struct Item {
                uint32_t old_idx, new_idx;
                uint8_t lvl;
            };
            auto append = [&](uint32_t old_idx, uint8_t lvl) {
                const uint32_t ni = (uint32_t)old_indices.size();
                old_indices.push_back(old_idx);
                out.child_start.push_back(0);
                out.child_count.push_back(0);
                out.level.push_back(lvl);
                return ni;
            };
            std::vector<Item> queue{{(uint32_t)root, append((uint32_t)root, 0), 0}};
            size_t qi = 0;
            while (qi < queue.size()) {
                const Item it = queue[qi++];
                const auto& children = output_children[it.old_idx];
                if (children.empty())
                    continue;
                out.child_start[it.new_idx] = (uint32_t)old_indices.size();
                out.child_count[it.new_idx] = (uint16_t)children.size();
                for (uint32_t c : children)
                    queue.push_back({c, append(c, (uint8_t)(it.lvl + 1)), (uint8_t)(it.lvl + 1)});
            }
        }

        // parents + payload
        const size_t out_n = old_indices.size();
        out.parent.assign(out_n, kInvalidNode);
        for (size_t i = 0; i < out_n; ++i)
            for (uint32_t c = out.child_start[i]; c < out.child_start[i] + out.child_count[i]; ++c)
                out.parent[c] = (uint32_t)i;

        out.splats.resize(out_n);
        out.splats.lod_opacity_encoded = true;
        out.centers.resize(out_n);
        out.sizes.resize(out_n);
        for (size_t i = 0; i < out_n; ++i) {
            const size_t o = old_indices[i];
            out.splats.means[i] = {ws.cx[o], ws.cy[o], ws.cz[o]};
            out.splats.log_scales[i] = {std::log(std::max(ws.sx[o], kMinScale)),
                                        std::log(std::max(ws.sy[o], kMinScale)),
                                        std::log(std::max(ws.sz[o], kMinScale))};
            out.splats.rotations[i] = ws.quat[o];
            out.splats.opacity_raw[i] = ws.alpha[o];
            const Vec3 c = ws.color[o];
            out.splats.sh0[i] = {(c.x - 0.5f) / SH_C0, (c.y - 0.5f) / SH_C0, (c.z - 0.5f) / SH_C0};
            out.centers[i] = out.splats.means[i];

            // node size with Spark lodOpacity expansion (bhatt_lod.cpp output stage)
            const float max_scale = std::max({ws.sx[o], ws.sy[o], ws.sz[o]});
            float expansion = 1.f;
            const float la = std::max(ws.alpha[o], 0.f);
            if (la > 1.f) {
                const float spark = std::min(la * 4.f - 3.f, 5.f);
                expansion = 1.f + 0.7f * (spark - 1.f);
            }
            out.sizes[i] = 2.f * expansion * max_scale;
        }

        out.input_count = n;
        out.total_merges = total_merges;
        out.output_count = out_n;
        out.build_ms = nowMs() - t0;
    }

    namespace {

        // node_pixel_scale() from lod_select_threshold.slang with the gaze cone
        // disabled (foveate=1 in front of the camera) and viewport-edge foveation
        // kept (outside-view content coarsens to the 0.05 floor instead of being
        // culled — their substitute for frustum culling and prefetch).
        float nodePixelScale(const BhattRefTree& t, uint32_t i, const Camera& cam) {
            const Vec3 v = cam.worldToView(t.centers[i]);
            const float radius = std::max(t.sizes[i], 1e-6f);
            const float distance = std::max(v.norm(), 1e-6f);
            const float view_depth = -v.z;
            float distance_for_scale = distance;
            float foveate = 1.f;
            constexpr float kOutsideFloor = 0.05f;
            constexpr float kBehindPenalty = 0.05f; // min(behind=0.2, outside=0.05) per shader
            if (view_depth + radius <= 1e-6f) {
                foveate = kBehindPenalty;
            } else {
                const float depth = std::max(view_depth, 1e-6f);
                const float half_w = depth * std::tan(0.5f * cam.fovy_rad) * cam.aspect();
                const float half_h = depth * std::tan(0.5f * cam.fovy_rad);
                const float overflow =
                    std::max(std::fabs(v.x) - (half_w + radius), std::fabs(v.y) - (half_h + radius));
                if (overflow > 0.f) {
                    const float blend = std::max(std::max(half_w, half_h) * 0.08f, radius * 0.5f);
                    const float u = 1.f - std::clamp(overflow / blend, 0.f, 1.f);
                    foveate = kOutsideFloor + (1.f - kOutsideFloor) * u;
                }
                if (view_depth > 1e-6f)
                    distance_for_scale = view_depth;
            }
            return radius / std::max(distance_for_scale, 1e-6f) * foveate;
        }

    } // namespace

    void bhattSelect(const BhattRefTree& t, const Camera& cam, float limit, BhattSelection& out) {
        out.nodes.clear();
        out.weights.clear();
        constexpr float kTransitionEnd = 1.18f; // LOD_TRANSITION_END
        const size_t n = t.centers.size();
        std::vector<float> scale_cache(n);
        parallelFor(0, n, std::thread::hardware_concurrency(), [&](size_t lo, size_t hi) {
            for (size_t i = lo; i < hi; ++i)
                scale_cache[i] = nodePixelScale(t, (uint32_t)i, cam);
        });
        auto transition = [&](float parent_scale) {
            return std::clamp((parent_scale - limit) / std::max(limit * (kTransitionEnd - 1.f), 1e-12f),
                              0.f, 1.f);
        };
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t p = t.parent[i];
            const bool parent_above = (p == kInvalidNode) || scale_cache[p] > limit;
            if (!parent_above)
                continue;
            const bool leaf = t.child_count[i] == 0;
            const float ps = scale_cache[i];
            if (!leaf && ps > limit) {
                // expanded interior node: renders only inside the transition band
                if (ps <= limit * kTransitionEnd) {
                    const float w = 1.f - transition(ps);
                    if (w > 0.f) {
                        out.nodes.push_back(i);
                        out.weights.push_back(w);
                    }
                }
                continue;
            }
            const float w = (p == kInvalidNode) ? 1.f : transition(scale_cache[p]);
            if (w > 0.f) {
                out.nodes.push_back(i);
                out.weights.push_back(w);
            }
        }
    }

    float bhattLimitForBudget(const BhattRefTree& t, const Camera& cam, uint64_t budget) {
        float lo = 1e-7f, hi = 1.f;
        BhattSelection sel;
        for (int it = 0; it < 40; ++it) {
            const float mid = std::sqrt(lo * hi); // log-space bisection
            bhattSelect(t, cam, mid, sel);
            if (sel.nodes.size() > budget)
                lo = mid;
            else
                hi = mid;
            if (hi / lo < 1.001f)
                break;
        }
        return hi;
    }

    void bhattGather(const BhattRefTree& t, const BhattSelection& sel, SplatCloud& out) {
        const size_t n = sel.nodes.size();
        out.resize(n);
        out.lod_opacity_encoded = true;
        for (size_t i = 0; i < n; ++i) {
            const uint32_t s = sel.nodes[i];
            out.means[i] = t.splats.means[s];
            out.log_scales[i] = t.splats.log_scales[s];
            out.rotations[i] = t.splats.rotations[s];
            out.opacity_raw[i] = t.splats.opacity_raw[s] * sel.weights[i];
            out.sh0[i] = t.splats.sh0[s];
        }
    }

} // namespace lfs::lod
