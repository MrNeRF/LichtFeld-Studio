/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// LOD builder benchmark + quality-gate harness. Env-gated:
//   LFS_LOD_BUILDER_BENCH=synthetic[:count]  one synthetic bucket (default 2M)
//   LFS_LOD_BUILDER_BENCH=/path/to/file.ply  Morton-bucketed real input
// Builds every bucket with bhatt, the pure refined octree, and the hybrid
// default (octree bottom + bhatt top) and reports wall time and tree stats
// (node counts, level distribution, per-level integrated alpha).

#include "core/bhatt_lod.hpp"
#include "core/environment.hpp"
#include "core/octree_lod.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/loader.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    constexpr std::size_t kBucketSplats = 2'000'000;

    struct RawSplats {
        std::size_t count = 0;
        int sh_degree = 0;
        int rest_coeffs = 0;
        std::vector<float> means, sh0, shN, scaling, rotation, opacity;
    };

    RawSplats make_synthetic(const std::size_t count) {
        RawSplats raw;
        raw.count = count;
        raw.sh_degree = 3;
        raw.rest_coeffs = 15;
        raw.means.resize(count * 3);
        raw.sh0.resize(count * 3);
        raw.shN.resize(count * static_cast<std::size_t>(raw.rest_coeffs) * 3);
        raw.scaling.resize(count * 3);
        raw.rotation.resize(count * 4);
        raw.opacity.resize(count);

        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> cluster_dist(-100.0f, 100.0f);
        std::normal_distribution<float> offset_dist(0.0f, 2.0f);
        std::uniform_real_distribution<float> color_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> logit_dist(-3.0f, 3.0f);
        std::uniform_real_distribution<float> log_scale_dist(-5.0f, -2.0f);
        std::normal_distribution<float> quat_dist(0.0f, 1.0f);
        std::uniform_real_distribution<float> sh_dist(-0.5f, 0.5f);

        constexpr std::size_t kClusters = 4096;
        std::vector<std::array<float, 3>> centers(kClusters);
        for (auto& c : centers) {
            c = {cluster_dist(rng), cluster_dist(rng), cluster_dist(rng) * 0.1f};
        }
        for (std::size_t i = 0; i < count; ++i) {
            const auto& c = centers[i % kClusters];
            raw.means[i * 3 + 0] = c[0] + offset_dist(rng);
            raw.means[i * 3 + 1] = c[1] + offset_dist(rng);
            raw.means[i * 3 + 2] = c[2] + offset_dist(rng);
            for (int d = 0; d < 3; ++d) {
                raw.sh0[i * 3 + d] = color_dist(rng);
                raw.scaling[i * 3 + d] = log_scale_dist(rng);
            }
            for (int k = 0; k < raw.rest_coeffs * 3; ++k) {
                raw.shN[i * static_cast<std::size_t>(raw.rest_coeffs) * 3 + k] = sh_dist(rng);
            }
            float q[4] = {quat_dist(rng), quat_dist(rng), quat_dist(rng), quat_dist(rng)};
            const float norm = std::max(
                std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]), 1e-6f);
            for (int d = 0; d < 4; ++d) {
                raw.rotation[i * 4 + d] = q[d] / norm;
            }
            raw.opacity[i] = logit_dist(rng);
        }
        return raw;
    }

    RawSplats load_ply(const std::string& path) {
        const auto loader = lfs::io::Loader::create();
        auto result = loader->load(path);
        if (!result) {
            ADD_FAILURE() << "failed to load " << path << ": " << result.error().message;
            return {};
        }
        auto* splat_ptr = std::get_if<std::shared_ptr<SplatData>>(&result->data);
        if (splat_ptr == nullptr || !*splat_ptr) {
            ADD_FAILURE() << "not a splat file: " << path;
            return {};
        }
        const SplatData& splat = **splat_ptr;

        RawSplats raw;
        raw.count = static_cast<std::size_t>(splat.size());
        raw.sh_degree = splat.get_max_sh_degree();
        raw.rest_coeffs = static_cast<int>(splat.max_sh_coeffs_rest());

        const auto copy_out = [](const Tensor& t, std::vector<float>& dst) {
            const auto cpu = t.cpu().contiguous();
            dst.assign(cpu.ptr<float>(), cpu.ptr<float>() + cpu.numel());
        };
        copy_out(splat.means_raw(), raw.means);
        copy_out(splat.sh0_raw(), raw.sh0);
        copy_out(splat.scaling_raw(), raw.scaling);
        copy_out(splat.rotation_raw(), raw.rotation);
        copy_out(splat.opacity_raw(), raw.opacity);
        if (raw.rest_coeffs > 0) {
            const auto shN = splat.shN_canonical_cpu();
            raw.shN.assign(shN.ptr<float>(), shN.ptr<float>() + shN.numel());
        }
        return raw;
    }

    inline std::uint32_t spread_bits_10(std::uint32_t v) {
        v &= 0x3FFu;
        v = (v | (v << 16)) & 0x030000FFu;
        v = (v | (v << 8)) & 0x0300F00Fu;
        v = (v | (v << 4)) & 0x030C30C3u;
        v = (v | (v << 2)) & 0x09249249u;
        return v;
    }

    // Morton-sorted contiguous buckets, mirroring the converter's spatial
    // bucketing closely enough for like-for-like builder timings.
    std::vector<std::uint32_t> morton_order(const RawSplats& raw) {
        float mn[3], mx[3];
        std::fill_n(mn, 3, std::numeric_limits<float>::max());
        std::fill_n(mx, 3, std::numeric_limits<float>::lowest());
        for (std::size_t i = 0; i < raw.count; ++i) {
            for (int a = 0; a < 3; ++a) {
                const float v = raw.means[i * 3 + a];
                if (std::isfinite(v)) {
                    mn[a] = std::min(mn[a], v);
                    mx[a] = std::max(mx[a], v);
                }
            }
        }
        float inv[3];
        for (int a = 0; a < 3; ++a) {
            inv[a] = mx[a] > mn[a] ? 1.0f / (mx[a] - mn[a]) : 0.0f;
        }
        std::vector<std::uint64_t> keys(raw.count);
        for (std::size_t i = 0; i < raw.count; ++i) {
            std::uint32_t cell[3];
            for (int a = 0; a < 3; ++a) {
                const float norm = (raw.means[i * 3 + a] - mn[a]) * inv[a];
                cell[a] = static_cast<std::uint32_t>(
                    std::clamp(std::isfinite(norm) ? norm : 0.0f, 0.0f, 1.0f) * 1023.0f);
            }
            const std::uint64_t code = spread_bits_10(cell[0]) |
                                       (spread_bits_10(cell[1]) << 1) |
                                       (spread_bits_10(cell[2]) << 2);
            keys[i] = (code << 24) | i;
        }
        std::sort(keys.begin(), keys.end());
        std::vector<std::uint32_t> order(raw.count);
        for (std::size_t i = 0; i < raw.count; ++i) {
            order[i] = static_cast<std::uint32_t>(keys[i] & ((1u << 24) - 1));
        }
        return order;
    }

    SplatData make_bucket(const RawSplats& raw, const std::uint32_t* order, const std::size_t count) {
        std::vector<float> means(count * 3);
        std::vector<float> sh0(count * 3);
        std::vector<float> shN(count * static_cast<std::size_t>(raw.rest_coeffs) * 3);
        std::vector<float> scaling(count * 3);
        std::vector<float> rotation(count * 4);
        std::vector<float> opacity(count);
        const std::size_t sh_floats = static_cast<std::size_t>(raw.rest_coeffs) * 3;
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t s = order[i];
            std::memcpy(means.data() + i * 3, raw.means.data() + s * 3, 3 * sizeof(float));
            std::memcpy(sh0.data() + i * 3, raw.sh0.data() + s * 3, 3 * sizeof(float));
            std::memcpy(scaling.data() + i * 3, raw.scaling.data() + s * 3, 3 * sizeof(float));
            std::memcpy(rotation.data() + i * 4, raw.rotation.data() + s * 4, 4 * sizeof(float));
            opacity[i] = raw.opacity[s];
            if (sh_floats > 0) {
                std::memcpy(shN.data() + i * sh_floats, raw.shN.data() + s * sh_floats,
                            sh_floats * sizeof(float));
            }
        }
        Tensor shN_tensor;
        if (raw.rest_coeffs > 0) {
            shN_tensor = Tensor::from_vector(
                shN, {count, static_cast<std::size_t>(raw.rest_coeffs), 3}, Device::CPU);
        }
        return SplatData(
            raw.sh_degree,
            Tensor::from_vector(means, {count, 3}, Device::CPU),
            Tensor::from_vector(sh0, {count, 1, 3}, Device::CPU),
            std::move(shN_tensor),
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

    struct BuilderStats {
        double total_ms = 0.0;
        std::uint64_t nodes = 0;
        std::uint64_t interior = 0;
        std::uint64_t leaves = 0;
        double interior_alpha_sum = 0.0;
        double interior_integrated_alpha = 0.0;
        double root_integrated_alpha = 0.0;
        std::map<int, std::uint64_t> nodes_per_level;
        std::map<int, double> integrated_alpha_per_level;
    };

    void accumulate(const SplatData& lod, BuilderStats& st) {
        const std::size_t n = static_cast<std::size_t>(lod.size());
        ASSERT_TRUE(lod.lod_tree && lod.lod_tree->has_tree());
        const auto& tree = *lod.lod_tree;
        const auto opacity_cpu = lod.opacity_raw().cpu().contiguous();
        const auto scaling_cpu = lod.scaling_raw().cpu().contiguous();
        const float* const opacity = opacity_cpu.ptr<float>();
        const float* const scaling = scaling_cpu.ptr<float>();

        st.nodes += n;
        for (std::size_t i = 0; i < n; ++i) {
            ++st.nodes_per_level[tree.lod_level[i]];
            const float area = ellipsoid_area(std::exp(scaling[i * 3 + 0]),
                                              std::exp(scaling[i * 3 + 1]),
                                              std::exp(scaling[i * 3 + 2]));
            st.integrated_alpha_per_level[tree.lod_level[i]] +=
                static_cast<double>(opacity[i]) * area;
            if (i == 0) {
                st.root_integrated_alpha += static_cast<double>(opacity[i]) * area;
            }
            if (tree.child_count[i] == 0) {
                ++st.leaves;
                continue;
            }
            ++st.interior;
            st.interior_alpha_sum += opacity[i];
            st.interior_integrated_alpha += static_cast<double>(opacity[i]) * area;
        }
    }

    void print_stats(const char* name, const BuilderStats& st) {
        std::printf("%s: %.0f ms total, %llu nodes (%llu leaves, %llu interior)\n",
                    name, st.total_ms,
                    static_cast<unsigned long long>(st.nodes),
                    static_cast<unsigned long long>(st.leaves),
                    static_cast<unsigned long long>(st.interior));
        std::printf("%s: interior alpha sum %.1f, interior integrated alpha %.1f, "
                    "root integrated alpha %.4f\n",
                    name, st.interior_alpha_sum, st.interior_integrated_alpha,
                    st.root_integrated_alpha);
        std::printf("%s: nodes per level:", name);
        for (const auto& [level, count] : st.nodes_per_level) {
            std::printf(" %d:%llu", level, static_cast<unsigned long long>(count));
        }
        std::printf("\n%s: integrated alpha per level:", name);
        for (const auto& [level, alpha] : st.integrated_alpha_per_level) {
            std::printf(" %d:%.0f", level, alpha);
        }
        std::printf("\n");
    }

} // namespace

TEST(LodBuilderBench, CompareBuilders) {
    const auto env = lfs::core::environment::value("LFS_LOD_BUILDER_BENCH");
    if (!env) {
        GTEST_SKIP() << "set LFS_LOD_BUILDER_BENCH=synthetic[:count] or =<file.ply> to run";
    }

    RawSplats raw;
    const std::string spec(*env);
    if (spec.starts_with("synthetic")) {
        std::size_t count = kBucketSplats;
        if (const auto sep = spec.find(':'); sep != std::string::npos) {
            const auto* const first = spec.data() + sep + 1;
            const auto [ptr, ec] = std::from_chars(first, spec.data() + spec.size(), count);
            ASSERT_EQ(ec, std::errc{}) << "bad count in " << spec;
        }
        raw = make_synthetic(count);
    } else {
        raw = load_ply(spec);
    }
    ASSERT_GT(raw.count, 0u);
    ASSERT_LT(raw.count, std::size_t{1} << 24)
        << "bench loader is in-memory; use a bucket-sized input";

    const auto order = morton_order(raw);
    const std::size_t bucket_count = (raw.count + kBucketSplats - 1) / kBucketSplats;
    std::printf("input: %zu splats, SH degree %d, %zu bucket(s)\n",
                raw.count, raw.sh_degree, bucket_count);

    BuilderStats bhatt_stats, octree_stats, hybrid_stats;
    lfs::core::OctreeLodBuildOptions pure_octree;
    pure_octree.bhatt_top_nodes = 0;
    for (std::size_t b = 0; b < bucket_count; ++b) {
        const std::size_t first = b * kBucketSplats;
        const std::size_t count = std::min(kBucketSplats, raw.count - first);
        const SplatData bucket = make_bucket(raw, order.data() + first, count);

        const auto t0 = std::chrono::steady_clock::now();
        auto bhatt = lfs::core::build_bhatt_lod(bucket);
        const auto t1 = std::chrono::steady_clock::now();
        auto octree = lfs::core::build_octree_lod(bucket, pure_octree);
        const auto t2 = std::chrono::steady_clock::now();
        auto hybrid = lfs::core::build_octree_lod(bucket);
        const auto t3 = std::chrono::steady_clock::now();
        ASSERT_TRUE(bhatt.has_value()) << bhatt.error();
        ASSERT_TRUE(octree.has_value()) << octree.error();
        ASSERT_TRUE(hybrid.has_value()) << hybrid.error();

        const double bhatt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double octree_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        const double hybrid_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        bhatt_stats.total_ms += bhatt_ms;
        octree_stats.total_ms += octree_ms;
        hybrid_stats.total_ms += hybrid_ms;
        std::printf("bucket %zu (%zu splats): bhatt %.0f ms, octree %.0f ms, hybrid %.0f ms\n",
                    b, count, bhatt_ms, octree_ms, hybrid_ms);

        accumulate(**bhatt, bhatt_stats);
        accumulate(**octree, octree_stats);
        accumulate(**hybrid, hybrid_stats);
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
    }

    print_stats("bhatt ", bhatt_stats);
    print_stats("octree", octree_stats);
    print_stats("hybrid", hybrid_stats);
    std::printf("bhatt/octree %.1fx, bhatt/hybrid %.1fx, hybrid/octree %.2fx\n",
                bhatt_stats.total_ms / std::max(octree_stats.total_ms, 1e-3),
                bhatt_stats.total_ms / std::max(hybrid_stats.total_ms, 1e-3),
                hybrid_stats.total_ms / std::max(octree_stats.total_ms, 1e-3));

    EXPECT_EQ(bhatt_stats.leaves, raw.count);
    EXPECT_EQ(octree_stats.leaves, raw.count);
    EXPECT_EQ(hybrid_stats.leaves, raw.count);
}
