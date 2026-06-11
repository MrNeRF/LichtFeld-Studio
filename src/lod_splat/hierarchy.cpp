/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "hierarchy.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>

namespace lfs::lod {

    namespace {

        struct SourceRef {
            const SplatCloud* cloud;
            uint32_t offset, count;
        };

        struct MergeAccum {
            // Moment-matched merge of an index list of source splats.
            static void mergeGroup(const SplatCloud& src, const uint32_t* idx, uint32_t n,
                                   SplatCloud& dst, uint32_t dst_idx, const HierarchyParams& params) {
                double W = 0;
                Vec3 mu{0, 0, 0};
                double alpha_area = 0;
                Vec3 col{0, 0, 0};
                // first pass: weights + mean
                static thread_local std::vector<float> wstore;
                if (wstore.size() < n)
                    wstore.resize(n);
                float* wbuf = wstore.data();
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t s = idx[i];
                    const float a = src.opacity(s);
                    const float area = src.meanArea(s);
                    float w = a * area;
                    if (w < 1e-20f)
                        w = 1e-20f;
                    wbuf[i] = w;
                    W += w;
                    mu += src.means[s] * w;
                    alpha_area += (double)a * area;
                    col += src.sh0[s] * w;
                }
                const float invW = (float)(1.0 / W);
                mu = mu * invW;
                col = col * invW;

                // second pass: mixture covariance = E[cov] + spread of means
                Sym3 cov;
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t s = idx[i];
                    const Sym3 ci = Sym3::fromCovariance(src.rotations[s].toMat3(), src.linearScale(s));
                    cov.addScaled(ci, wbuf[i]);
                    cov.addOuterScaled(src.means[s] - mu, wbuf[i]);
                }
                cov.scale(invW);

                Vec3 eval;
                Mat3 evec;
                eigenSym3(cov, eval, evec);
                const Vec3 s{std::sqrt(std::max(eval.x, 1e-12f)),
                             std::sqrt(std::max(eval.y, 1e-12f)),
                             std::sqrt(std::max(eval.z, 1e-12f))};
                const float merged_area = (s.x * s.y + s.y * s.z + s.z * s.x) * (3.14159265f / 3.f);
                const float alpha_exact = (float)(alpha_area / std::max(merged_area, 1e-20f));

                dst.means[dst_idx] = mu;
                dst.log_scales[dst_idx] = {std::log(s.x), std::log(s.y), std::log(s.z)};
                dst.rotations[dst_idx] = Quat::fromMat3(evec);
                if (params.lod_opacity) {
                    // lodOpacity: linear alpha, may exceed 1 — conserves the
                    // cluster's integrated alpha exactly (up to the safety cap)
                    dst.opacity_raw[dst_idx] = std::clamp(alpha_exact, 0.005f, params.max_lod_opacity);
                } else {
                    dst.opacity_raw[dst_idx] = logit(std::clamp(alpha_exact, 0.02f, 0.995f));
                }
                dst.sh0[dst_idx] = col;
            }
        };

        // Greedy similarity pairing over a Morton-coherent source list: each
        // unpaired splat picks its Bhattacharyya-nearest partner among the next
        // few unpaired candidates. Produces ceil(n/2) groups in `group_idx`
        // (pairs flattened, optional trailing singleton).
        void similarityPairs(const SplatCloud& src, uint32_t n, std::vector<uint32_t>& group_idx) {
            constexpr int kWindow = 16;
            static thread_local std::vector<Sym3> covs;
            static thread_local std::vector<float> dets;
            static thread_local std::vector<uint8_t> paired;
            covs.resize(n);
            dets.resize(n);
            paired.assign(n, 0);
            for (uint32_t i = 0; i < n; ++i) {
                covs[i] = Sym3::fromCovariance(src.rotations[i].toMat3(), src.linearScale(i));
                dets[i] = detSym3(covs[i]);
            }
            group_idx.clear();
            group_idx.reserve(n);
            for (uint32_t i = 0; i < n; ++i) {
                if (paired[i])
                    continue;
                int seen = 0;
                uint32_t best = UINT32_MAX;
                float best_d = 1e30f;
                for (uint32_t j = i + 1; j < n && seen < kWindow; ++j) {
                    if (paired[j])
                        continue;
                    ++seen;
                    const float d = bhattDistance(src.means[i] - src.means[j], covs[i], dets[i],
                                                  covs[j], dets[j]);
                    if (d < best_d) {
                        best_d = d;
                        best = j;
                    }
                }
                paired[i] = 1;
                group_idx.push_back(i);
                if (best != UINT32_MAX) {
                    paired[best] = 1;
                    group_idx.push_back(best);
                } else {
                    group_idx.push_back(UINT32_MAX); // trailing singleton sentinel
                }
            }
        }

        struct Builder {
            const HierarchyParams& params;
            Hierarchy& h;
            std::vector<uint64_t> codes; // Morton codes aligned with h.leaves

            Builder(const HierarchyParams& p, Hierarchy& out) : params(p), h(out) {}

            uint32_t buildTopology(uint32_t begin, uint32_t end, int depth) {
                const uint32_t idx = (uint32_t)h.nodes.size();
                h.nodes.emplace_back();
                {
                    Node& n = h.nodes[idx];
                    n.depth = (uint8_t)depth;
                    n.subtree_splats = end - begin;
                }
                const uint32_t count = end - begin;
                if (count <= params.max_leaf_splats || depth >= params.max_depth || depth >= 20) {
                    Node& n = h.nodes[idx];
                    n.is_leaf = true;
                    n.rep_offset = begin;
                    n.rep_count = count;
                    return idx;
                }
                const int shift = 3 * (20 - depth);
                uint32_t child_begin = begin;
                uint32_t kids[8];
                uint8_t kid_count = 0;
                for (int oct = 0; oct < 8 && child_begin < end; ++oct) {
                    // first index whose octant digit at this level exceeds `oct`
                    const uint64_t* lo = codes.data() + child_begin;
                    const uint64_t* hi = codes.data() + end;
                    const uint64_t limit = (uint64_t)oct;
                    const uint64_t* it = std::upper_bound(lo, hi, limit, [shift](uint64_t v, uint64_t c) {
                        return v < ((c >> shift) & 7);
                    });
                    const uint32_t child_end = (uint32_t)(it - codes.data());
                    if (child_end > child_begin) {
                        const uint32_t ci = buildTopology(child_begin, child_end, depth + 1);
                        kids[kid_count++] = ci;
                        h.nodes[ci].parent = idx;
                    }
                    child_begin = child_end;
                }
                Node& n = h.nodes[idx]; // re-fetch: recursion may have reallocated
                n.child_count = kid_count;
                for (int i = 0; i < kid_count; ++i)
                    n.children[i] = kids[i];
                if (kid_count == 0) { // degenerate (identical positions)
                    n.is_leaf = true;
                    n.rep_offset = begin;
                    n.rep_count = count;
                }
                return idx;
            }
        };

    } // namespace

    void buildHierarchy(SplatCloud&& input, const HierarchyParams& params, Hierarchy& out) {
        const double t_start = nowMs();
        BuildStats& st = out.stats;
        const size_t n = input.size();

        Aabb bb;
        for (const Vec3& p : input.means)
            bb.grow(p);
        out.scene_bounds = bb;
        const Vec3 ext = bb.extent();
        const Vec3 inv{ext.x > 1e-12f ? 1.f / ext.x : 0.f,
                       ext.y > 1e-12f ? 1.f / ext.y : 0.f,
                       ext.z > 1e-12f ? 1.f / ext.z : 0.f};

        Builder b(params, out);

        std::vector<uint32_t> order(n);
        {
            ScopedTimer t(st.morton_ms);
            b.codes.resize(n);
            parallelFor(0, n, params.threads, [&](size_t lo, size_t hi) {
                for (size_t i = lo; i < hi; ++i) {
                    b.codes[i] = morton3D(input.means[i], bb.mn, inv);
                    order[i] = (uint32_t)i;
                }
            });
        }
        {
            ScopedTimer t(st.sort_ms);
            radixSortPairs(b.codes, order);
        }
        {
            ScopedTimer t(st.reorder_ms);
            out.leaves.resize(n);
            parallelFor(0, n, params.threads, [&](size_t lo, size_t hi) {
                for (size_t i = lo; i < hi; ++i) {
                    const uint32_t s = order[i];
                    out.leaves.means[i] = input.means[s];
                    out.leaves.log_scales[i] = input.log_scales[s];
                    out.leaves.rotations[i] = input.rotations[s];
                    out.leaves.opacity_raw[i] = input.opacity_raw[s];
                    out.leaves.sh0[i] = input.sh0[s];
                }
            });
            input = SplatCloud{}; // release
        }
        {
            ScopedTimer t(st.topology_ms);
            out.nodes.reserve(2 * n / std::max(1u, params.max_leaf_splats) + 64);
            b.buildTopology(0, (uint32_t)n, 0);
        }

        // --- representative counts + merged storage layout (children have higher
        // pre-order indices than parents, so reverse order is bottom-up) ---
        const uint32_t node_count = (uint32_t)out.nodes.size();
        uint64_t merged_total = 0;
        for (uint32_t i = node_count; i-- > 0;) {
            Node& nd = out.nodes[i];
            if (nd.is_leaf)
                continue;
            uint64_t source = 0;
            for (int c = 0; c < nd.child_count; ++c)
                source += out.nodes[nd.children[c]].rep_count;
            nd.rep_count = (uint32_t)std::clamp<uint64_t>(
                (source + params.reduction - 1) / params.reduction, 1, params.max_rep_count);
        }
        for (Node& nd : out.nodes)
            if (!nd.is_leaf) {
                nd.rep_offset = (uint32_t)merged_total;
                merged_total += nd.rep_count;
            }
        out.merged.resize(merged_total);
        out.merged.lod_opacity_encoded = params.lod_opacity;

        // --- depth buckets for parallel bottom-up fill ---
        int max_depth = 0;
        for (const Node& nd : out.nodes)
            max_depth = std::max(max_depth, (int)nd.depth);
        std::vector<std::vector<uint32_t>> by_depth(max_depth + 1);
        for (uint32_t i = 0; i < node_count; ++i)
            by_depth[out.nodes[i].depth].push_back(i);

        st.nodes_per_depth.assign(max_depth + 1, 0);
        st.reps_per_depth.assign(max_depth + 1, 0);

        {
            ScopedTimer t(st.merge_ms);
            for (int d = max_depth; d >= 0; --d) {
                auto& bucket = by_depth[d];
                parallelFor(0, bucket.size(), params.threads, [&](size_t lo, size_t hi) {
                    SplatCloud scratch; // contiguous gather buffer reused per chunk
                    for (size_t bi = lo; bi < hi; ++bi) {
                        Node& nd = out.nodes[bucket[bi]];
                        if (nd.is_leaf) {
                            Aabb nb;
                            float max_s = 0.f;
                            double ext_sum = 0;
                            for (uint32_t i = nd.rep_offset; i < nd.rep_offset + nd.rep_count; ++i) {
                                nb.grow(out.leaves.means[i]);
                                const Vec3 s = out.leaves.linearScale(i);
                                const float smax = s.maxComp();
                                max_s = std::max(max_s, smax);
                                ext_sum += 2.0 * smax;
                            }
                            nb.mn = nb.mn - Vec3{3 * max_s, 3 * max_s, 3 * max_s};
                            nb.mx = nb.mx + Vec3{3 * max_s, 3 * max_s, 3 * max_s};
                            nd.bounds = nb;
                            nd.rep_extent = nd.rep_count ? (float)(ext_sum / nd.rep_count) : 0.f;
                            continue;
                        }
                        // gather child representatives into a contiguous scratch
                        // cloud, normalizing opacity to linear (leaf children are
                        // logit-encoded, interior children may be lodOpacity)
                        uint32_t total = 0;
                        for (int c = 0; c < nd.child_count; ++c)
                            total += out.nodes[nd.children[c]].rep_count;
                        scratch.resize(total);
                        scratch.lod_opacity_encoded = true;
                        uint32_t w = 0;
                        Aabb nb;
                        for (int c = 0; c < nd.child_count; ++c) {
                            const Node& ch = out.nodes[nd.children[c]];
                            nb.grow(ch.bounds);
                            const SplatCloud& src = out.cloudFor(ch);
                            for (uint32_t i = 0; i < ch.rep_count; ++i, ++w) {
                                const uint32_t s = ch.rep_offset + i;
                                scratch.means[w] = src.means[s];
                                scratch.log_scales[w] = src.log_scales[s];
                                scratch.rotations[w] = src.rotations[s];
                                scratch.opacity_raw[w] = src.opacity(s);
                                scratch.sh0[w] = src.sh0[s];
                            }
                        }
                        double ext_sum = 0;
                        float max_s = 0.f;
                        static thread_local std::vector<uint32_t> group_idx;
                        const bool pair_mode = params.similarity_pairing && params.reduction == 2 &&
                                               (uint64_t)nd.rep_count * 2 >= total;
                        if (pair_mode)
                            similarityPairs(scratch, total, group_idx);
                        for (uint32_t k = 0; k < nd.rep_count; ++k) {
                            if (pair_mode) {
                                const uint32_t pair[2] = {group_idx[2 * k], group_idx[2 * k + 1]};
                                const uint32_t cnt = (pair[1] == UINT32_MAX) ? 1 : 2;
                                MergeAccum::mergeGroup(scratch, pair, cnt, out.merged,
                                                       nd.rep_offset + k, params);
                            } else {
                                const uint32_t gb = (uint32_t)((uint64_t)k * total / nd.rep_count);
                                const uint32_t ge = std::max(
                                    (uint32_t)((uint64_t)(k + 1) * total / nd.rep_count), gb + 1);
                                static thread_local std::vector<uint32_t> run;
                                run.resize(ge - gb);
                                for (uint32_t i = gb; i < ge; ++i)
                                    run[i - gb] = i;
                                MergeAccum::mergeGroup(scratch, run.data(), ge - gb, out.merged,
                                                       nd.rep_offset + k, params);
                            }
                            const Vec3 s = out.merged.linearScale(nd.rep_offset + k);
                            const float smax = s.maxComp();
                            max_s = std::max(max_s, smax);
                            ext_sum += 2.0 * smax;
                        }
                        nb.mn = nb.mn - Vec3{3 * max_s, 3 * max_s, 3 * max_s};
                        nb.mx = nb.mx + Vec3{3 * max_s, 3 * max_s, 3 * max_s};
                        nd.bounds = nb;
                        nd.rep_extent = nd.rep_count ? (float)(ext_sum / nd.rep_count) : 0.f;
                    }
                });
            }
        }

        for (const Node& nd : out.nodes) {
            st.nodes_per_depth[nd.depth]++;
            st.reps_per_depth[nd.depth] += nd.rep_count;
        }
        st.total_ms = nowMs() - t_start;
    }

} // namespace lfs::lod
