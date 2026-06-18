/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/bhatt_lod.hpp"
#include "core/octree_lod.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    constexpr int kShDegree = 2;
    constexpr int kRestCoeffs = 8;

    SplatData make_synthetic_input(const std::size_t count) {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> cluster_dist(-100.0f, 100.0f);
        std::normal_distribution<float> offset_dist(0.0f, 2.0f);
        std::uniform_real_distribution<float> color_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> logit_dist(-3.0f, 3.0f);
        std::uniform_real_distribution<float> log_scale_dist(-5.0f, -2.0f);
        std::normal_distribution<float> quat_dist(0.0f, 1.0f);
        std::uniform_real_distribution<float> sh_dist(-0.5f, 0.5f);

        constexpr std::size_t kClusters = 64;
        std::vector<std::array<float, 3>> centers(kClusters);
        for (auto& c : centers) {
            c = {cluster_dist(rng), cluster_dist(rng), cluster_dist(rng) * 0.1f};
        }

        std::vector<float> means(count * 3);
        std::vector<float> sh0(count * 3);
        std::vector<float> shN(count * kRestCoeffs * 3);
        std::vector<float> scaling(count * 3);
        std::vector<float> rotation(count * 4);
        std::vector<float> opacity(count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto& c = centers[i % kClusters];
            means[i * 3 + 0] = c[0] + offset_dist(rng);
            means[i * 3 + 1] = c[1] + offset_dist(rng);
            means[i * 3 + 2] = c[2] + offset_dist(rng);
            for (int d = 0; d < 3; ++d) {
                sh0[i * 3 + d] = color_dist(rng);
                scaling[i * 3 + d] = log_scale_dist(rng);
            }
            for (int k = 0; k < kRestCoeffs * 3; ++k) {
                shN[i * kRestCoeffs * 3 + k] = sh_dist(rng);
            }
            float q[4] = {quat_dist(rng), quat_dist(rng), quat_dist(rng), quat_dist(rng)};
            const float norm = std::max(
                std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]), 1e-6f);
            for (int d = 0; d < 4; ++d) {
                rotation[i * 4 + d] = q[d] / norm;
            }
            opacity[i] = logit_dist(rng);
        }

        return SplatData(
            kShDegree,
            Tensor::from_vector(means, {count, 3}, Device::CPU),
            Tensor::from_vector(sh0, {count, 1, 3}, Device::CPU),
            Tensor::from_vector(shN, {count, kRestCoeffs, 3}, Device::CPU),
            Tensor::from_vector(scaling, {count, 3}, Device::CPU),
            Tensor::from_vector(rotation, {count, 4}, Device::CPU),
            Tensor::from_vector(opacity, {count, 1}, Device::CPU),
            1.0f);
    }

    float ellipsoid_area(const float sx, const float sy, const float sz) {
        constexpr float p = 1.6075f;
        const float t1 = std::pow(sx * sy, p);
        const float t2 = std::pow(sx * sz, p);
        const float t3 = std::pow(sy * sz, p);
        return 4.0f * 3.14159265f * std::pow((t1 + t2 + t3) / 3.0f, 1.0f / p);
    }

    struct TreeView {
        std::size_t n = 0;
        const float* means = nullptr;
        const float* opacity = nullptr;
        const float* scaling = nullptr;
        const float* shN = nullptr;
        const lfs::core::SplatLodTree* tree = nullptr;
        Tensor means_cpu, opacity_cpu, scaling_cpu, shN_cpu;

        explicit TreeView(const SplatData& data) {
            n = static_cast<std::size_t>(data.size());
            means_cpu = data.means_raw().cpu().contiguous();
            opacity_cpu = data.opacity_raw().cpu().contiguous();
            scaling_cpu = data.scaling_raw().cpu().contiguous();
            means = means_cpu.ptr<float>();
            opacity = opacity_cpu.ptr<float>();
            scaling = scaling_cpu.ptr<float>();
            if (data.shN_raw().is_valid() && data.shN_raw().numel() > 0) {
                shN_cpu = data.shN_canonical_cpu();
                shN = shN_cpu.ptr<float>();
            }
            tree = data.lod_tree.get();
        }

        float integrated_alpha(const std::size_t i) const {
            const float sx = std::exp(scaling[i * 3 + 0]);
            const float sy = std::exp(scaling[i * 3 + 1]);
            const float sz = std::exp(scaling[i * 3 + 2]);
            return opacity[i] * ellipsoid_area(sx, sy, sz);
        }
    };

    // Shared structural contract of both builders: root at 0, children
    // contiguous and after their parent, exactly one parent per non-root node.
    void check_structure(const TreeView& v, std::size_t& leaf_count) {
        ASSERT_NE(v.tree, nullptr);
        ASSERT_EQ(v.tree->total_nodes(), v.n);
        leaf_count = 0;
        std::vector<std::uint32_t> parent_count(v.n, 0);
        for (std::size_t i = 0; i < v.n; ++i) {
            const std::uint32_t cc = v.tree->child_count[i];
            const std::uint32_t cs = v.tree->child_start[i];
            if (cc == 0) {
                ++leaf_count;
                continue;
            }
            ASSERT_GT(cs, i) << "children must follow their parent (node " << i << ")";
            ASSERT_LE(static_cast<std::size_t>(cs) + cc, v.n);
            for (std::uint32_t c = 0; c < cc; ++c) {
                ++parent_count[cs + c];
            }
        }
        ASSERT_EQ(parent_count[0], 0u) << "root must have no parent";
        for (std::size_t i = 1; i < v.n; ++i) {
            ASSERT_EQ(parent_count[i], 1u) << "node " << i << " must have exactly one parent";
        }
    }

    // Interior alpha is lodOpacity: alpha * area of a parent matches the sum
    // over its children up to per-level float rounding and the [1e-6, 1000]
    // clamp.
    void check_alpha_conservation(const TreeView& v) {
        std::size_t checked = 0;
        for (std::size_t i = 0; i < v.n; ++i) {
            const std::uint32_t cc = v.tree->child_count[i];
            if (cc == 0 || v.opacity[i] >= 999.0f) {
                continue;
            }
            const std::uint32_t cs = v.tree->child_start[i];
            float child_sum = 0.0f;
            for (std::uint32_t c = 0; c < cc; ++c) {
                child_sum += v.integrated_alpha(cs + c);
            }
            const float own = v.integrated_alpha(i);
            ASSERT_NEAR(own, child_sum, std::max(child_sum * 5e-3f, 1e-6f))
                << "integrated alpha not conserved at node " << i;
            ++checked;
        }
        ASSERT_GT(checked, 0u);
    }

    std::vector<std::array<float, 3>> sorted_leaf_positions(const TreeView& v) {
        std::vector<std::array<float, 3>> positions;
        for (std::size_t i = 0; i < v.n; ++i) {
            if (v.tree->child_count[i] == 0) {
                positions.push_back({v.means[i * 3 + 0], v.means[i * 3 + 1], v.means[i * 3 + 2]});
            }
        }
        std::sort(positions.begin(), positions.end());
        return positions;
    }

    std::vector<std::array<float, 3>> sorted_input_positions(const SplatData& input) {
        const auto means_cpu = input.means_raw().cpu().contiguous();
        const float* const ptr = means_cpu.ptr<float>();
        std::vector<std::array<float, 3>> positions(static_cast<std::size_t>(input.size()));
        for (std::size_t i = 0; i < positions.size(); ++i) {
            positions[i] = {ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]};
        }
        std::sort(positions.begin(), positions.end());
        return positions;
    }

    void check_level_order(const TreeView& v) {
        for (std::size_t i = 1; i < v.n; ++i) {
            ASSERT_GE(v.tree->lod_level[i], v.tree->lod_level[i - 1])
                << "output not level-ordered at node " << i;
        }
    }

    // Interior nodes carry blended SH1-3 at every emitted level: every
    // coefficient is a convex combination of its children's, and the blend is
    // non-trivial.
    void check_sh_convexity(const TreeView& v) {
        constexpr std::size_t kShFloats = kRestCoeffs * 3;
        float max_abs_interior_sh = 0.0f;
        for (std::size_t i = 0; i < v.n; ++i) {
            const std::uint32_t cc = v.tree->child_count[i];
            if (cc == 0) {
                continue;
            }
            const std::uint32_t cs = v.tree->child_start[i];
            for (std::size_t k = 0; k < kShFloats; ++k) {
                float lo = std::numeric_limits<float>::max();
                float hi = std::numeric_limits<float>::lowest();
                for (std::uint32_t c = 0; c < cc; ++c) {
                    const float val = v.shN[(cs + c) * kShFloats + k];
                    lo = std::min(lo, val);
                    hi = std::max(hi, val);
                }
                const float own = v.shN[i * kShFloats + k];
                ASSERT_GE(own, lo - 1e-4f) << "SH blend out of range at node " << i << " coeff " << k;
                ASSERT_LE(own, hi + 1e-4f) << "SH blend out of range at node " << i << " coeff " << k;
                max_abs_interior_sh = std::max(max_abs_interior_sh, std::abs(own));
            }
        }
        EXPECT_GT(max_abs_interior_sh, 1e-3f) << "interior SH must not collapse to zero";
    }

} // namespace

TEST(OctreeLod, MatchesBhattContractOnSyntheticInput) {
    constexpr std::size_t kSplats = 50'000;
    const SplatData input = make_synthetic_input(kSplats);

    lfs::core::OctreeLodBuildOptions pure_octree;
    pure_octree.bhatt_top_nodes = 0;
    auto octree = lfs::core::build_octree_lod(input, pure_octree);
    ASSERT_TRUE(octree.has_value()) << octree.error();
    auto bhatt = lfs::core::build_bhatt_lod(input);
    ASSERT_TRUE(bhatt.has_value()) << bhatt.error();

    const TreeView ov(**octree);
    const TreeView bv(**bhatt);

    std::size_t octree_leaves = 0;
    std::size_t bhatt_leaves = 0;
    check_structure(ov, octree_leaves);
    check_structure(bv, bhatt_leaves);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }
    EXPECT_EQ(octree_leaves, kSplats);
    EXPECT_EQ(bhatt_leaves, kSplats);

    // Exact leaf preservation: identical position multisets, identical to the
    // input.
    const auto expected = sorted_input_positions(input);
    EXPECT_EQ(sorted_leaf_positions(ov), expected);
    EXPECT_EQ(sorted_leaf_positions(bv), expected);

    // BFS level order is part of the octree builder's contract (bhatt emits
    // DFS pre-order and relies on the converter's relabel pass).
    check_level_order(ov);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }

    // Binary refinement contract: every interior node merges exactly two
    // children, so the output is a strict binary tree of 2n - 1 nodes whose
    // level populations grow by at most 2x -- the ~2x granularity steps the
    // pixel-threshold LOD selector needs (the unrefined octree jumped ~8x).
    EXPECT_EQ(ov.n, 2 * kSplats - 1);
    std::vector<std::size_t> per_level;
    for (std::size_t i = 0; i < ov.n; ++i) {
        ASSERT_TRUE(ov.tree->child_count[i] == 0 || ov.tree->child_count[i] == 2)
            << "interior node " << i << " must have exactly two children";
        const std::size_t level = ov.tree->lod_level[i];
        if (level >= per_level.size()) {
            per_level.resize(level + 1, 0);
        }
        ++per_level[level];
    }
    for (std::size_t level = 0; level + 1 < per_level.size(); ++level) {
        ASSERT_LE(per_level[level + 1], 2 * per_level[level])
            << "level population jumped more than 2x at level " << level;
    }
    const std::size_t depth = per_level.size() - 1;
    EXPECT_GE(depth, 16u) << "binary tree over " << kSplats << " leaves must be at least log2 deep";

    check_alpha_conservation(ov);
    check_alpha_conservation(bv);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }

    // Conservation is transitive, so both roots integrate to the same total
    // leaf alpha regardless of tree shape.
    const float octree_root = ov.integrated_alpha(0);
    const float bhatt_root = bv.integrated_alpha(0);
    EXPECT_NEAR(octree_root, bhatt_root, bhatt_root * 0.02f);

    // Depth is bhatt-like: within 2x of the binary merge tree's.
    std::uint8_t bhatt_depth = 0;
    for (std::size_t i = 0; i < bv.n; ++i) {
        bhatt_depth = std::max(bhatt_depth, bv.tree->lod_level[i]);
    }
    EXPECT_LE(depth, 2u * bhatt_depth);

    check_sh_convexity(ov);
}

// The hybrid default stitches a similarity-ordered bhatt top onto the
// octree-built binary bottom. The seam where bhatt leaves hand over to the
// octree representatives is the risky boundary: child ranges are rebased
// across two builders, so contiguity, parent-before-child level order, and
// per-node integrated-alpha conservation are asserted over the whole tree --
// a wrong rebase at the seam breaks the child sums of the bhatt parents
// directly above it.
TEST(OctreeLod, HybridStitchedTreeInvariants) {
    struct Config {
        std::size_t splats;
        std::uint32_t top_nodes;
    };
    for (const Config cfg : {Config{50'000, 4'096}, Config{50'000, 32'768},
                             Config{20'000, 64}, Config{10'000, 1u << 20}}) {
        SCOPED_TRACE(::testing::Message()
                     << "splats=" << cfg.splats << " top_nodes=" << cfg.top_nodes);
        const SplatData input = make_synthetic_input(cfg.splats);
        lfs::core::OctreeLodBuildOptions options;
        options.bhatt_top_nodes = cfg.top_nodes;
        auto hybrid = lfs::core::build_octree_lod(input, options);
        ASSERT_TRUE(hybrid.has_value()) << hybrid.error();

        const TreeView v(**hybrid);
        std::size_t leaves = 0;
        check_structure(v, leaves);
        check_level_order(v);
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
        EXPECT_EQ(leaves, cfg.splats);
        EXPECT_EQ(sorted_leaf_positions(v), sorted_input_positions(input));
        check_alpha_conservation(v);
        check_sh_convexity(v);
        if (::testing::Test::HasFatalFailure()) {
            return;
        }

        // The bhatt top prunes, so the stitched tree is smaller than the
        // strict binary 2n - 1 the pure octree emits.
        EXPECT_LT(v.n, 2 * cfg.splats - 1);

        // Seam decomposition: the leaves must hang under at most top_nodes
        // maximal all-binary subtrees (the octree bottom), and at least one
        // interior node above them must be non-binary (the pruned bhatt top).
        std::vector<std::uint32_t> parent(v.n, std::numeric_limits<std::uint32_t>::max());
        for (std::size_t i = 0; i < v.n; ++i) {
            const std::uint32_t cs = v.tree->child_start[i];
            for (std::uint32_t c = 0; c < v.tree->child_count[i]; ++c) {
                parent[cs + c] = static_cast<std::uint32_t>(i);
            }
        }
        std::vector<std::uint8_t> pure_binary(v.n, 0);
        for (std::size_t i = v.n; i-- > 0;) {
            const std::uint32_t cc = v.tree->child_count[i];
            const std::uint32_t cs = v.tree->child_start[i];
            if (cc == 0) {
                pure_binary[i] = 1;
            } else if (cc == 2) {
                pure_binary[i] = pure_binary[cs] && pure_binary[cs + 1];
            }
        }
        std::size_t maximal_binary_roots = 0;
        std::size_t top_interior = 0;
        for (std::size_t i = 0; i < v.n; ++i) {
            if (pure_binary[i] && (i == 0 || !pure_binary[parent[i]])) {
                ++maximal_binary_roots;
            }
            if (v.tree->child_count[i] > 0 && !pure_binary[i]) {
                ++top_interior;
            }
        }
        EXPECT_LE(maximal_binary_roots, cfg.top_nodes);
        EXPECT_GE(top_interior, 1u);
    }
}

TEST(OctreeLod, SmallInputs) {
    lfs::core::OctreeLodBuildOptions pure_octree;
    pure_octree.bhatt_top_nodes = 0;
    for (const auto& options : {lfs::core::OctreeLodBuildOptions{}, pure_octree}) {
        for (const std::size_t count : {std::size_t{1}, std::size_t{2}, std::size_t{7},
                                        std::size_t{9}, std::size_t{100}}) {
            SCOPED_TRACE(::testing::Message() << "count=" << count
                                              << " top_nodes=" << options.bhatt_top_nodes);
            const SplatData input = make_synthetic_input(count);
            auto octree = lfs::core::build_octree_lod(input, options);
            ASSERT_TRUE(octree.has_value()) << octree.error();

            const TreeView v(**octree);
            std::size_t leaves = 0;
            check_structure(v, leaves);
            if (::testing::Test::HasFatalFailure()) {
                return;
            }
            EXPECT_EQ(leaves, count);
            if (count == 1) {
                EXPECT_EQ(v.n, 1u);
            } else {
                check_alpha_conservation(v);
            }
        }
    }
}

TEST(OctreeLod, IdenticalPositionsStayBounded) {
    constexpr std::size_t kSplats = 1'000;
    std::vector<float> means(kSplats * 3, 1.5f);
    std::vector<float> sh0(kSplats * 3, 0.2f);
    std::vector<float> scaling(kSplats * 3, -3.0f);
    std::vector<float> rotation(kSplats * 4, 0.0f);
    std::vector<float> opacity(kSplats, 0.5f);
    for (std::size_t i = 0; i < kSplats; ++i) {
        rotation[i * 4] = 1.0f;
    }
    const SplatData input(
        0,
        Tensor::from_vector(means, {kSplats, 3}, Device::CPU),
        Tensor::from_vector(sh0, {kSplats, 1, 3}, Device::CPU),
        Tensor{},
        Tensor::from_vector(scaling, {kSplats, 3}, Device::CPU),
        Tensor::from_vector(rotation, {kSplats, 4}, Device::CPU),
        Tensor::from_vector(opacity, {kSplats, 1}, Device::CPU),
        1.0f);

    lfs::core::OctreeLodBuildOptions pure_octree;
    pure_octree.bhatt_top_nodes = 0;
    auto octree = lfs::core::build_octree_lod(input, pure_octree);
    ASSERT_TRUE(octree.has_value()) << octree.error();
    const TreeView v(**octree);
    std::size_t leaves = 0;
    check_structure(v, leaves);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }
    EXPECT_EQ(leaves, kSplats);
    EXPECT_EQ(v.n, 2 * kSplats - 1);
    std::uint16_t max_children = 0;
    for (std::size_t i = 0; i < v.n; ++i) {
        max_children = std::max(max_children, v.tree->child_count[i]);
    }
    EXPECT_EQ(max_children, 2);

    // The hybrid default routes identical positions through the bhatt top
    // (n below the budget); the degenerate input must stay bounded there too.
    auto hybrid = lfs::core::build_octree_lod(input);
    ASSERT_TRUE(hybrid.has_value()) << hybrid.error();
    const TreeView hv(**hybrid);
    std::size_t hybrid_leaves = 0;
    check_structure(hv, hybrid_leaves);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }
    EXPECT_EQ(hybrid_leaves, kSplats);
    check_alpha_conservation(hv);
}
