/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal and Vulkan Mrnf, Refine and Mcmc ops against CPU transliterations of the CUDA
// kernels (mrnf_kernels.cu, densification_kernels.cu, pruning_kernels.cu,
// mcmc_kernels.cu) on fixture-sized inputs. Integer results, selections and
// RNG-driven choices are exact; float math compares within the fast-math
// budget, since both backends use approximate transcendentals.

#include "core/gpu_backend_fwd.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/refine_scratch.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

namespace {
    // The backend of the running parameterized test.
    lfs::core::GpuBackend backend_under_test() { return testing::TestWithParam<lfs::core::GpuBackend>::GetParam(); }

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    namespace ops = lfs::gpu_ops;

    constexpr float kAbs = 1e-5f;
    constexpr float kRel = 1e-4f;

    // ---- Tensor helpers --------------------------------------------------

    Tensor gpu(const std::vector<float>& values) { return Tensor::from_vector(values, {values.size()}, Device::GPU); }

    Tensor gpu_rows(const std::vector<float>& values, const size_t cols) {
        return Tensor::from_vector(values, {values.size() / cols, cols}, Device::GPU);
    }

    Tensor gpu_bool(const std::vector<bool>& values) { return Tensor::from_vector(values, {values.size()}, Device::GPU); }

    Tensor gpu_i64(const std::vector<int64_t>& values) {
        auto cpu = Tensor::empty({values.size()}, Device::CPU, DataType::Int64);
        std::copy(values.begin(), values.end(), cpu.ptr<int64_t>());
        return cpu.gpu();
    }

    Tensor gpu_i32(const std::vector<int32_t>& values) {
        auto cpu = Tensor::empty({values.size()}, Device::CPU, DataType::Int32);
        std::copy(values.begin(), values.end(), cpu.ptr<int32_t>());
        return cpu.gpu();
    }

    std::vector<float> host_f(const Tensor& t) {
        const auto cpu = t.cpu().contiguous();
        return {cpu.ptr<float>(), cpu.ptr<float>() + cpu.numel()};
    }

    std::vector<int64_t> host_i64(const Tensor& t) {
        const auto cpu = t.cpu().contiguous();
        return {cpu.ptr<int64_t>(), cpu.ptr<int64_t>() + cpu.numel()};
    }

    std::vector<uint8_t> host_u8(const Tensor& t) {
        const auto cpu = t.cpu().contiguous();
        const auto* data = static_cast<const uint8_t*>(cpu.data_ptr());
        return {data, data + cpu.numel()};
    }

    std::vector<float> values(const size_t count, const float scale, const int seed, const float offset = 0.f) {
        std::vector<float> out(count);
        for (size_t i = 0; i < count; ++i) {
            const auto k = static_cast<int>((i * 17u + static_cast<size_t>(seed) * 13u) % 97u);
            out[i] = offset + scale * (static_cast<float>(k) / 48.f - 1.f);
        }
        return out;
    }

    std::vector<float> unit_values(const size_t count, const float offset = 0.f) {
        std::vector<float> out(count);
        for (size_t i = 0; i < count; ++i)
            out[i] = offset + static_cast<float>((i * 17 + 3) % 101) / 101.f;
        return out;
    }

    std::vector<bool> every(const size_t count, const size_t period) {
        std::vector<bool> out(count);
        for (size_t i = 0; i < count; ++i)
            out[i] = i % period == 0;
        return out;
    }

    void expect_close(const std::vector<float>& actual, const std::vector<float>& expected, const char* what,
                      const float abs = kAbs, const float rel = kRel) {
        ASSERT_EQ(actual.size(), expected.size()) << what;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (std::isnan(expected[i])) {
                EXPECT_TRUE(std::isnan(actual[i])) << what << "[" << i << "]";
                continue;
            }
            EXPECT_NEAR(actual[i], expected[i], abs + rel * std::abs(expected[i])) << what << "[" << i << "]";
        }
    }

    // Exact values; fast math does not keep the sign of a zero product.
    void expect_same(const std::vector<float>& actual, const std::vector<float>& expected, const char* what) {
        ASSERT_EQ(actual.size(), expected.size()) << what;
        for (size_t i = 0; i < actual.size(); ++i)
            EXPECT_EQ(actual[i], expected[i]) << what << "[" << i << "]";
    }

    // ---- curand Philox4_32_10 and its distributions ----------------------

    std::array<uint32_t, 4> philox(std::array<uint32_t, 4> c, std::array<uint32_t, 2> k) {
        for (int round = 0; round < 10; ++round) {
            if (round > 0) {
                k[0] += 0x9E3779B9u;
                k[1] += 0xBB67AE85u;
            }
            const uint64_t p0 = uint64_t{0xD2511F53u} * c[0];
            const uint64_t p1 = uint64_t{0xCD9E8D57u} * c[2];
            c = {static_cast<uint32_t>(p1 >> 32) ^ c[1] ^ k[0], static_cast<uint32_t>(p1),
                 static_cast<uint32_t>(p0 >> 32) ^ c[3] ^ k[1], static_cast<uint32_t>(p0)};
        }
        return c;
    }

    // curand_init(seed, subsequence, 0, &state): the state's first output block.
    std::array<uint32_t, 4> curand_block(const uint64_t seed, const uint64_t subsequence) {
        return philox({0u, 0u, static_cast<uint32_t>(subsequence), static_cast<uint32_t>(subsequence >> 32)},
                      {static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)});
    }

    float curand_uniform(const uint32_t x) { return std::fma(static_cast<float>(x), 2.3283064e-10f, 2.3283064e-10f / 2.0f); }

    std::array<float, 4> curand_normal4(const std::array<uint32_t, 4>& x) {
        constexpr float kInv2Pi = 2.3283064e-10f * 6.2831855f;
        std::array<float, 4> out{};
        for (int pair = 0; pair < 2; ++pair) {
            const float u = curand_uniform(x[2 * pair]);
            const float v = std::fma(static_cast<float>(x[2 * pair + 1]), kInv2Pi, kInv2Pi / 2.0f);
            const float s = std::sqrt(-2.0f * std::log(u));
            out[2 * pair] = std::sin(v) * s;
            out[2 * pair + 1] = std::cos(v) * s;
        }
        return out;
    }

    // cub::DeviceRadixSort's float order.
    uint32_t radix_key(const float v) {
        const uint32_t bits = std::bit_cast<uint32_t>(v);
        return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    }

    std::vector<float> radix_sorted(std::vector<float> v) {
        std::stable_sort(v.begin(), v.end(), [](float a, float b) { return radix_key(a) < radix_key(b); });
        return v;
    }

    float sigmoid(const float x) { return 1.0f / (1.0f + std::exp(-x)); }

    class PortableDensifyOps : public ::testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(backend_under_test()))
                GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " device unavailable";
            scope_.emplace(backend_under_test());
            table_ = &lfs::training::training_ops(backend_under_test());
            if (table_->mrnf == nullptr || table_->refine == nullptr || table_->mcmc == nullptr)
                GTEST_SKIP() << "Mrnf/Refine/Mcmc slots are empty";
        }

        const ops::MrnfOps& mrnf() const { return *table_->mrnf; }
        const ops::RefineOps& refine() const { return *table_->refine; }
        const ops::McmcOps& mcmc() const { return *table_->mcmc; }

    private:
        std::optional<lfs::core::GpuBackendScope> scope_;
        const lfs::training::TrainingOps* table_ = nullptr;
    };

    TEST(MetalDensifyRng, PhiloxMatchesRandom123KnownAnswers) {
        EXPECT_EQ(philox({0, 0, 0, 0}, {0, 0}), (std::array<uint32_t, 4>{0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}));
        EXPECT_EQ(philox({~0u, ~0u, ~0u, ~0u}, {~0u, ~0u}),
                  (std::array<uint32_t, 4>{0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}));
        EXPECT_EQ(philox({0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u}),
                  (std::array<uint32_t, 4>{0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}));
    }

    // ---- MRNF ------------------------------------------------------------

    TEST_P(PortableDensifyOps, MrnfNoiseFollowsTheCurandStream) {
        constexpr size_t n = 257;
        constexpr uint64_t seed = 0x4d524e46ull;
        auto opacity = values(n, 4.f, 3, -2.f);
        const auto visibility = values(n, 1.f, 7);
        const auto means = values(3 * n, 0.5f, 1);
        const auto frozen = every(n - 5, 4);
        const ops::MrnfNoiseParams params{.seed = seed, .lr_mean = 1.f, .noise_weight = 0.02f, .median_scale = 0.03f};

        auto means_gpu = gpu_rows(means, 3);
        mrnf().noise(means_gpu, gpu(opacity), gpu(visibility), gpu_bool(frozen), params);

        auto expected = means;
        for (size_t i = 0; i < n; ++i) {
            if ((i < frozen.size() && frozen[i]) || visibility[i] <= 0.f)
                continue;
            float weight = std::pow(std::max(1.0f - sigmoid(opacity[i]), 0.0f), 150.0f);
            weight *= params.lr_mean * params.noise_weight;
            if (weight < 1e-12f)
                continue;
            const auto noise = curand_normal4(curand_block(seed, i));
            for (int d = 0; d < 3; ++d)
                expected[i * 3 + d] += std::clamp(noise[d] * weight, -params.median_scale, params.median_scale);
        }
        expect_close(host_f(means_gpu), expected, "mrnf.noise");
    }

    TEST_P(PortableDensifyOps, MrnfDecay) {
        constexpr size_t n = 257;
        auto raw = values(n, 1.5f, 9);
        raw[1] = std::numeric_limits<float>::infinity();
        raw[2] = -std::numeric_limits<float>::infinity();
        const auto log_scales = values(3 * n, 0.4f, 11);
        const auto frozen = every(n, 4);
        const auto far = every(n, 3);
        const ops::DecayParams params{.opacity_decay = 0.02f, .scale_decay = 0.01f, .far_decay_scale = 0.25f, .train_t = 0.4f};
        auto raw_gpu = gpu(raw);
        auto scales_gpu = gpu_rows(log_scales, 3);
        mrnf().decay(raw_gpu, scales_gpu, gpu_bool(frozen), gpu_bool(far), params);

        auto expected_raw = raw;
        auto expected_scales = log_scales;
        for (size_t i = 0; i < n; ++i) {
            if (frozen[i])
                continue;
            const float scale = far[i] ? params.far_decay_scale : 1.f;
            const float t = 1.0f - params.train_t;
            float p = sigmoid(raw[i]) - params.opacity_decay * scale * t;
            p = std::clamp(p, 1e-12f, std::nextafter(1.0f, 0.0f));
            expected_raw[i] = std::log(p / (1.0f - p));
            for (int d = 0; d < 3; ++d)
                expected_scales[i * 3 + d] =
                    std::log(std::max(std::exp(log_scales[i * 3 + d]) * (1.0f - params.scale_decay * scale * t), 1e-12f));
        }
        expect_close(host_f(raw_gpu), expected_raw, "mrnf.decay.opacity");
        expect_close(host_f(scales_gpu), expected_scales, "mrnf.decay.scales");
    }

    TEST_P(PortableDensifyOps, MrnfPercentileBoundsSelectCubOrder) {
        constexpr size_t n = 1029;
        auto means = values(3 * n, 3.f, 2);
        for (size_t i = 0; i < 3 * n; i += 7)
            means[i] = -0.0f;
        const float percentile = 0.8f;
        const auto bounds = mrnf().percentile_bounds(gpu_rows(means, 3), percentile);

        const float low_pct = (1.0f - percentile) / 2.0f;
        const auto low = static_cast<size_t>(low_pct * static_cast<float>(n - 1));
        const auto high = static_cast<size_t>((1.0f - low_pct) * static_cast<float>(n - 1));
        float extents[3];
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<float> column(n);
            for (size_t i = 0; i < n; ++i)
                column[i] = means[i * 3 + axis];
            const auto sorted = radix_sorted(column);
            EXPECT_EQ(std::bit_cast<uint32_t>(bounds.center[axis]),
                      std::bit_cast<uint32_t>((sorted[low] + sorted[high]) * 0.5f));
            extents[axis] = (sorted[high] - sorted[low]) * 0.5f;
            EXPECT_EQ(bounds.extent[axis], extents[axis]);
        }
        std::sort(extents, extents + 3);
        EXPECT_EQ(bounds.median_size, extents[1] * 2.0f);
        EXPECT_EQ(bounds.max_extent, extents[2]);
    }

    TEST_P(PortableDensifyOps, MrnfMedianExtentAndSortedMedian) {
        constexpr size_t n = 1537;
        auto scales = values(3 * n, 0.8f, 4);
        for (size_t i = 0; i < 3 * n; i += 11)
            scales[i] = -200.f; // exp underflows to 0: not positive
        const auto extent = mrnf().median_extent(gpu_rows(scales, 3));
        std::vector<float> positive;
        for (size_t i = 0; i < n; ++i) {
            const float g = std::exp((scales[i * 3] + scales[i * 3 + 1] + scales[i * 3 + 2]) * (1.0f / 3.0f));
            if (std::isfinite(g) && g > 0.f)
                positive.push_back(g);
        }
        const auto sorted = radix_sorted(positive);
        ASSERT_FALSE(sorted.empty());
        EXPECT_TRUE(extent.valid);
        EXPECT_NEAR(extent.value, sorted[sorted.size() / 2], 1e-6f * sorted[sorted.size() / 2] + 1e-7f);

        const auto empty = mrnf().median_extent(gpu_rows(std::vector<float>(9, -200.f), 3));
        EXPECT_FALSE(empty.valid);
        EXPECT_EQ(empty.value, 0.f);

        auto median_values = values(4099, 3.f, 9);
        median_values[3] = -0.0f;
        median_values[4] = 0.0f;
        EXPECT_EQ(mrnf().sorted_median(gpu(median_values)), radix_sorted(median_values)[median_values.size() / 2]);
        // CUB orders -0 before +0.
        EXPECT_EQ(std::bit_cast<uint32_t>(mrnf().sorted_median(gpu({0.0f, -0.0f, -1.f}))), 0x80000000u);
        EXPECT_EQ(std::bit_cast<uint32_t>(mrnf().sorted_median(gpu({0.0f, -0.0f, -1.f, 1.f}))), 0u);
    }

    // Stable descending order of CUDA's Gumbel keys.
    std::vector<int64_t> gumbel_reference(const std::vector<float>& weights, const size_t k, const uint64_t seed,
                                          const bool compact_sparse) {
        const size_t n = weights.size();
        if (k == n) {
            std::vector<int64_t> all(n);
            std::iota(all.begin(), all.end(), 0);
            return all;
        }
        std::vector<int64_t> sources;
        for (size_t i = 0; i < n; ++i)
            if (weights[i] > 0.f)
                sources.push_back(static_cast<int64_t>(i));
        const bool compact = compact_sparse && sources.size() >= k && sources.size() < n;
        if (!compact) {
            sources.resize(n);
            std::iota(sources.begin(), sources.end(), 0);
        }
        std::vector<float> keys(sources.size());
        for (size_t i = 0; i < sources.size(); ++i) {
            const float w = weights[static_cast<size_t>(sources[i])];
            if (!compact && w <= 0.f) {
                keys[i] = -1e30f;
                continue;
            }
            const float u = std::min(std::max(curand_uniform(curand_block(seed, i)[0]), 1e-10f), 1.0f - 1e-7f);
            keys[i] = -std::log(-std::log(u)) + std::log(w);
        }
        std::vector<size_t> order(keys.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return keys[a] > keys[b]; });
        std::vector<int64_t> out(k);
        for (size_t i = 0; i < k; ++i)
            out[i] = sources[order[i]];
        return out;
    }

    TEST_P(PortableDensifyOps, MrnfGumbelTopKMatchesCurandKeys) {
        constexpr size_t n = 257;
        constexpr uint64_t seed = 0x4d524e46ull;
        auto weights = values(n, 1.f, 8);
        for (auto& w : weights)
            w = std::abs(w);
        for (size_t i = 0; i < n; i += 5)
            weights[i] = 0.f;
        const size_t nnz = static_cast<size_t>(std::count_if(weights.begin(), weights.end(), [](float w) { return w > 0.f; }));

        const auto run = [&](const size_t k, const ops::GumbelParams& params) {
            auto top = Tensor::empty({k}, Device::GPU, DataType::Int64);
            lfs::training::GumbelTopKScratch scratch;
            mrnf().gumbel(&scratch, gpu(weights), top, params);
            return host_i64(top);
        };
        EXPECT_EQ(run(17, {.seed = seed}), gumbel_reference(weights, 17, seed, true));
        EXPECT_EQ(run(17, {.seed = seed, .known_nnz = nnz}), gumbel_reference(weights, 17, seed, true));
        EXPECT_EQ(run(17, {.seed = seed, .compact_sparse = false}), gumbel_reference(weights, 17, seed, false));
        // More picks than positive weights: zero weights follow in index order.
        EXPECT_EQ(run(nnz + 9, {.seed = 7}), gumbel_reference(weights, nnz + 9, 7, true));
        EXPECT_EQ(run(n, {.seed = seed}), gumbel_reference(weights, n, seed, true));

        // A sort longer than the tensor library's one-threadgroup network.
        std::vector<float> large(5000);
        for (size_t i = 0; i < large.size(); ++i)
            large[i] = static_cast<float>((i * 7919) % 1009) / 1009.f;
        auto top = Tensor::empty({64}, Device::GPU, DataType::Int64);
        mrnf().gumbel(nullptr, gpu(large), top, {.seed = 99});
        EXPECT_EQ(host_i64(top), gumbel_reference(large, 64, 99, true));
    }

    TEST_P(PortableDensifyOps, MrnfFoldsAndMasks) {
        constexpr size_t n = 64;
        const auto vis = values(n, 1.f, 1, 1.f);
        const auto weight = values(n, 0.2f, 2);
        const auto dens = values(2 * n, 0.5f, 3, 0.5f);
        const auto ratio = values(n, 0.1f, 4);
        auto vis_gpu = gpu(vis);
        auto weight_gpu = gpu(weight);
        auto dens_gpu = gpu_rows(dens, n);
        auto ratio_gpu = gpu(ratio);
        mrnf().fold(vis_gpu, weight_gpu, dens_gpu, ratio_gpu, 0.75f);
        std::vector<float> expected_vis(n), expected_weight(n), expected_ratio(n);
        for (size_t i = 0; i < n; ++i) {
            const float v = dens[i], e = dens[n + i];
            expected_vis[i] = vis[i] + v;
            expected_weight[i] = std::max(weight[i], e);
            expected_ratio[i] = std::max(ratio[i], v >= 0.05f ? e / std::pow(v, 0.75f) : 0.f);
        }
        expect_same(host_f(vis_gpu), expected_vis, "mrnf.fold.visibility");
        expect_same(host_f(weight_gpu), expected_weight, "mrnf.fold.weight");
        expect_close(host_f(ratio_gpu), expected_ratio, "mrnf.fold.ratio");
        expect_same(host_f(dens_gpu), std::vector<float>(2 * n, 0.f), "mrnf.fold.rows");

        auto err_gpu = gpu_rows(dens, n);
        auto max_gpu = gpu(weight);
        mrnf().fold_error(max_gpu, err_gpu);
        std::vector<float> expected_rows = dens;
        std::fill(expected_rows.begin() + n, expected_rows.end(), 0.f);
        expect_same(host_f(err_gpu), expected_rows, "mrnf.fold_error.rows");
        expect_same(host_f(max_gpu), expected_weight, "mrnf.fold_error.max");

        // fold without a ratio tensor.
        auto vis2 = gpu(vis);
        auto weight2 = gpu(weight);
        auto dens2 = gpu_rows(dens, n);
        Tensor absent;
        mrnf().fold(vis2, weight2, dens2, absent, 0.f);
        expect_same(host_f(vis2), expected_vis, "mrnf.fold.no_ratio");

        const auto means = values(3 * n, 1.f, 12);
        const std::array<float, 3> center{0.1f, -0.2f, 0.3f};
        auto far_gpu = Tensor::zeros({n}, Device::GPU, DataType::Bool);
        mrnf().far_mask(gpu_rows(means, 3), far_gpu, center, 1.5f);
        const auto scale_max = values(n, 0.5f, 4);
        const auto seed_mask = every(n, 3);
        auto prune_gpu = gpu_bool(seed_mask);
        auto nan_means = means;
        nan_means[3] = std::numeric_limits<float>::quiet_NaN();
        mrnf().prune_bounds(gpu_rows(nan_means, 3), gpu(scale_max), prune_gpu, {0.f, 0.f, 0.f}, 0.9f, 0.3f);
        const auto far = host_u8(far_gpu);
        const auto prune = host_u8(prune_gpu);
        for (size_t i = 0; i < n; ++i) {
            const float dx = means[i * 3] - center[0], dy = means[i * 3 + 1] - center[1], dz = means[i * 3 + 2] - center[2];
            EXPECT_EQ(far[i], (dx * dx + dy * dy + dz * dz) > 1.5f * 1.5f ? 1 : 0) << i;
            const float ax = std::abs(nan_means[i * 3]), ay = std::abs(nan_means[i * 3 + 1]),
                        az = std::abs(nan_means[i * 3 + 2]);
            const bool distance = !std::isnan(ax) && !std::isnan(ay) && !std::isnan(az) &&
                                  std::max(ax, std::max(ay, az)) > 0.9f;
            EXPECT_EQ(prune[i], seed_mask[i] || scale_max[i] > 0.3f || distance ? 1 : 0) << i;
        }
    }

    TEST_P(PortableDensifyOps, MrnfProjectionAndSeeds) {
        constexpr size_t n = 64;
        const auto means = values(3 * n, 1.f, 12);
        const std::vector<float> view{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1};
        auto means2d_gpu = Tensor::zeros({n, 2}, Device::GPU);
        auto radii_gpu = Tensor::zeros({n}, Device::GPU);
        const ops::ProjectParams project{.image = {.h = 8, .w = 12}, .intrinsics = {.fx = 20.f, .fy = 18.f, .cx = 6.f, .cy = 4.f}};
        mrnf().project_centers(gpu_rows(means, 3), Tensor::from_vector(view, {4, 4}, Device::GPU), means2d_gpu,
                               radii_gpu, project);
        std::vector<float> expected_means2d(2 * n), expected_radii(n);
        for (size_t i = 0; i < n; ++i) {
            const float x = means[i * 3], y = means[i * 3 + 1], z = means[i * 3 + 2] + 2.f;
            if (!(z > 0.01f))
                continue;
            const float px = 20.f * (x / z) + 6.f, py = 18.f * (y / z) + 4.f;
            expected_means2d[i * 2] = px;
            expected_means2d[i * 2 + 1] = py;
            expected_radii[i] = px >= 0.f && py >= 0.f && px < 12.f && py < 8.f ? 1.f : 0.f;
        }
        const auto means2d = host_f(means2d_gpu);
        expect_close(means2d, expected_means2d, "mrnf.project.means2d");
        const auto radii = host_f(radii_gpu);
        expect_same(radii, expected_radii, "mrnf.project.radii");

        const auto error = values(8 * 12, 1.f, 15, 1.f);
        auto scores_gpu = Tensor::zeros({n}, Device::GPU);
        mrnf().gather_center_error(means2d_gpu, radii_gpu, gpu_rows(error, 12), scores_gpu);
        std::vector<float> expected_scores(n);
        for (size_t i = 0; i < n; ++i) {
            if (radii[i] <= 0.f)
                continue;
            const int x = std::clamp(static_cast<int>(std::floor(means2d[i * 2])), 0, 11);
            const int y = std::clamp(static_cast<int>(std::floor(means2d[i * 2 + 1])), 0, 7);
            expected_scores[i] = error[static_cast<size_t>(y) * 12 + static_cast<size_t>(x)];
        }
        expect_same(host_f(scores_gpu), expected_scores, "mrnf.center_error");

        constexpr size_t h = 6, w = 8, hw = h * w;
        const auto predicted = values(3 * hw, 1.f, 1);
        const auto target = values(3 * hw, 0.7f, 2);
        auto error_gpu = Tensor::zeros({h, w}, Device::GPU);
        mrnf().mean_abs_error(Tensor::from_vector(predicted, {3, h, w}, Device::GPU),
                              Tensor::from_vector(target, {3, h, w}, Device::GPU), error_gpu);
        std::vector<float> expected_error(hw);
        for (size_t i = 0; i < hw; ++i) {
            float sum = 0.f;
            for (size_t c = 0; c < 3; ++c)
                sum += std::abs(predicted[c * hw + i] - target[c * hw + i]);
            expected_error[i] = sum / 3.f;
        }
        expect_close(host_f(error_gpu), expected_error, "mrnf.mae");

        const auto alpha = values(hw, 0.5f, 3, 0.5f);
        auto weights_gpu = error_gpu.clone().reshape({static_cast<int>(hw)});
        mrnf().seed_weights(weights_gpu, gpu(alpha), weights_gpu);
        std::vector<float> expected_weights(hw);
        for (size_t i = 0; i < hw; ++i)
            expected_weights[i] = expected_error[i] * (1.f - alpha[i]);
        expect_close(host_f(weights_gpu), expected_weights, "mrnf.seed_weights");

        const std::vector<int64_t> pixels{1, 4, 7, 20, 47, -3, 900};
        const auto depth = values(hw, 2.f, 6, 2.f);
        auto rgb_gpu = Tensor::zeros({pixels.size(), 3}, Device::GPU);
        auto alpha_gpu = Tensor::zeros({pixels.size()}, Device::GPU);
        auto depth_gpu = Tensor::zeros({pixels.size()}, Device::GPU);
        const auto target_chw = Tensor::from_vector(target, {3, h, w}, Device::GPU);
        mrnf().gather_seeds(gpu_i64(pixels), target_chw, gpu(alpha), gpu(depth), rgb_gpu, alpha_gpu, depth_gpu);
        std::vector<float> expected_rgb, expected_alpha, expected_depth;
        for (const int64_t p : pixels) {
            const size_t pix = p >= 0 && static_cast<size_t>(p) < hw ? static_cast<size_t>(p) : 0;
            for (size_t c = 0; c < 3; ++c)
                expected_rgb.push_back(target[c * hw + pix]);
            expected_alpha.push_back(alpha[pix]);
            expected_depth.push_back(depth[pix]);
        }
        expect_same(host_f(rgb_gpu), expected_rgb, "mrnf.gather.rgb");
        expect_same(host_f(alpha_gpu), expected_alpha, "mrnf.gather.alpha");
        expect_same(host_f(depth_gpu), expected_depth, "mrnf.gather.depth");
        mrnf().gather_seeds(gpu_i64(pixels), target_chw, gpu(alpha), Tensor(), rgb_gpu, alpha_gpu, depth_gpu);
        expect_same(host_f(depth_gpu), std::vector<float>(pixels.size(), 0.f), "mrnf.gather.no_depth");
    }

    TEST_P(PortableDensifyOps, MrnfWeightsAndCompaction) {
        constexpr size_t n = 33;
        auto weights = values(n, 1.f, 10, 1.f);
        auto vis = values(n, 2.f, 11, 1.f);
        vis[5] = 0.f;
        const float median = 1.3f;
        auto weights_gpu = gpu(weights);
        mrnf().starvation_weights(weights_gpu, gpu(vis), median);
        for (size_t i = 0; i < n; ++i) {
            if (vis[i] == 0.f) {
                weights[i] = 0.f;
                continue;
            }
            const float starved = std::clamp(1.0f - vis[i] / std::max(median, 1.19209290e-07f), 0.0f, 1.0f);
            weights[i] *= lfs::training::mrnf_strategy::kStarvEps + std::pow(starved, lfs::training::mrnf_strategy::kStarvGamma);
        }
        expect_close(host_f(weights_gpu), weights, "mrnf.starvation");

        constexpr size_t m = 64;
        const auto opacity = values(m, 1.f, 3, 0.5f);
        auto visibility = values(m, 1.f, 7);
        const auto active = every(m, 2);
        const auto trainable = every(m, 5);
        const auto edge = values(m, 0.2f, 6, 0.3f);
        auto parent_gpu = Tensor::zeros({m}, Device::GPU);
        mrnf().replace_parent_weights(gpu(opacity), gpu(visibility), gpu_bool(active), gpu_bool(trainable), gpu(edge),
                                      parent_gpu);
        std::vector<float> expected(m);
        for (size_t i = 0; i < m; ++i)
            expected[i] = opacity[i] * static_cast<float>(visibility[i] > 0.f) * static_cast<float>(active[i]) *
                          static_cast<float>(trainable[i]) * edge[i];
        expect_same(host_f(parent_gpu), expected, "mrnf.replace");
        mrnf().replace_parent_weights(gpu(opacity), gpu(visibility), Tensor(), Tensor(), Tensor(), parent_gpu);
        for (size_t i = 0; i < m; ++i)
            expected[i] = opacity[i] * static_cast<float>(visibility[i] > 0.f);
        expect_same(host_f(parent_gpu), expected, "mrnf.replace.plain");

        constexpr size_t mask_n = 10000;
        std::vector<bool> mask(mask_n);
        std::vector<int64_t> set;
        for (size_t i = 0; i < mask_n; ++i) {
            mask[i] = (i * 2654435761u) % 7 == 0;
            if (mask[i])
                set.push_back(static_cast<int64_t>(i));
        }
        auto compact = Tensor::empty({set.size()}, Device::GPU, DataType::Int64);
        EXPECT_EQ(mrnf().compact_bool_indices(gpu_bool(mask), compact, set.size()), set.size());
        EXPECT_EQ(host_i64(compact), set);
    }

    // ---- Refine ----------------------------------------------------------

    TEST_P(PortableDensifyOps, RefineSplitAndFill) {
        constexpr size_t n = 257;
        auto means = unit_values(3 * n);
        auto rotations = unit_values(4 * n, 0.1f);
        auto scales = unit_values(3 * n, -2.f);
        scales[128 * 3 + 1] = scales[128 * 3 + 2]; // tie between two axes
        const auto sh0 = unit_values(3 * n);
        auto opacity = unit_values(n, -0.5f);
        const std::vector<int64_t> ids{0, 128, 256};
        auto means_gpu = gpu_rows(means, 3), rotations_gpu = gpu_rows(rotations, 4), scales_gpu = gpu_rows(scales, 3);
        auto sh0_gpu = Tensor::from_vector(sh0, {n, 1, 3}, Device::GPU);
        auto opacity_gpu = Tensor::from_vector(opacity, {n, 1}, Device::GPU);
        auto child_means = Tensor::zeros({3, 3}, Device::GPU), child_rotations = Tensor::zeros({3, 4}, Device::GPU);
        auto child_scales = Tensor::zeros({3, 3}, Device::GPU), child_sh0 = Tensor::zeros({3, 1, 3}, Device::GPU);
        auto child_opacity = Tensor::zeros({3, 1}, Device::GPU);
        refine().split({means_gpu, rotations_gpu, scales_gpu, sh0_gpu, opacity_gpu},
                       {child_means, child_rotations, child_scales, child_sh0, child_opacity}, gpu_i64(ids));

        std::vector<float> expected_child_means, expected_child_rotations, expected_child_scales, expected_child_sh0,
            expected_child_opacity;
        for (const int64_t id : ids) {
            const size_t s = static_cast<size_t>(id);
            const float w = rotations[s * 4], x = rotations[s * 4 + 1], y = rotations[s * 4 + 2], z = rotations[s * 4 + 3];
            const float r[9] = {1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - w * z), 2.0f * (x * z + w * y),
                                2.0f * (x * y + w * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - w * x),
                                2.0f * (x * z - w * y), 2.0f * (y * z + w * x), 1.0f - 2.0f * (x * x + y * y)};
            const float* sc = &scales[s * 3];
            const float largest = std::max(sc[0], std::max(sc[1], sc[2]));
            const unsigned axes[3][3] = {{0, 1, 2}, {1, 0, 2}, {2, 0, 1}};
            const unsigned* order = largest == sc[0] ? axes[0] : (largest == sc[1] ? axes[1] : axes[2]);
            const float magnitude = std::exp(sc[order[0]]) * 0.5f;
            float new_scale[3];
            new_scale[order[0]] = sc[order[0]] + std::log(0.5f);
            new_scale[order[1]] = sc[order[1]] + std::log(0.85f);
            new_scale[order[2]] = sc[order[2]] + std::log(0.85f);
            const float raw = std::clamp(sigmoid(opacity[s]) * 0.6f, 1e-7f, 1.0f - 1e-7f);
            const float new_opacity = std::log(raw / (1.0f - raw));
            for (int d = 0; d < 3; ++d) {
                const float offset = r[order[0] + 3 * d] * magnitude;
                expected_child_means.push_back(means[s * 3 + d] - offset);
                means[s * 3 + d] += offset;
                expected_child_scales.push_back(new_scale[d]);
                scales[s * 3 + d] = new_scale[d];
                expected_child_sh0.push_back(sh0[s * 3 + d]);
            }
            for (int d = 0; d < 4; ++d)
                expected_child_rotations.push_back(rotations[s * 4 + d]);
            opacity[s] = new_opacity;
            expected_child_opacity.push_back(new_opacity);
        }
        expect_close(host_f(means_gpu), means, "refine.split.means");
        expect_close(host_f(scales_gpu), scales, "refine.split.scales");
        expect_close(host_f(opacity_gpu), opacity, "refine.split.opacity");
        expect_close(host_f(child_means), expected_child_means, "refine.split.child_means");
        expect_close(host_f(child_scales), expected_child_scales, "refine.split.child_scales");
        expect_close(host_f(child_opacity), expected_child_opacity, "refine.split.child_opacity");
        expect_same(host_f(child_rotations), expected_child_rotations, "refine.split.child_rotations");
        expect_same(host_f(child_sh0), expected_child_sh0, "refine.split.child_sh0");

        const auto src_means = host_f(child_means), src_rot = host_f(child_rotations), src_scales = host_f(child_scales);
        const auto src_sh0 = host_f(child_sh0), src_opacity = host_f(child_opacity);
        auto dst_means = host_f(means_gpu), dst_rot = host_f(rotations_gpu), dst_scales = host_f(scales_gpu);
        auto dst_sh0 = host_f(sh0_gpu), dst_opacity = host_f(opacity_gpu);
        const std::vector<int64_t> destinations{2, -1, 254};
        auto free_gpu = Tensor::ones({n}, Device::GPU, DataType::Bool);
        refine().fill_slots(gpu_i64(destinations), {child_means, child_rotations, child_scales, child_sh0, child_opacity},
                            {means_gpu, rotations_gpu, scales_gpu, sh0_gpu, opacity_gpu}, free_gpu);
        std::vector<uint8_t> expected_free(n, 1);
        for (size_t i = 0; i < destinations.size(); ++i) {
            if (destinations[i] < 0)
                continue;
            const auto d = static_cast<size_t>(destinations[i]);
            for (int c = 0; c < 3; ++c) {
                dst_means[d * 3 + c] = src_means[i * 3 + c];
                dst_scales[d * 3 + c] = src_scales[i * 3 + c];
                dst_sh0[d * 3 + c] = src_sh0[i * 3 + c];
            }
            for (int c = 0; c < 4; ++c)
                dst_rot[d * 4 + c] = src_rot[i * 4 + c];
            dst_opacity[d] = src_opacity[i];
            expected_free[d] = 0;
        }
        expect_same(host_f(means_gpu), dst_means, "refine.fill.means");
        expect_same(host_f(rotations_gpu), dst_rot, "refine.fill.rotations");
        expect_same(host_f(scales_gpu), dst_scales, "refine.fill.scales");
        expect_same(host_f(sh0_gpu), dst_sh0, "refine.fill.sh0");
        expect_same(host_f(opacity_gpu), dst_opacity, "refine.fill.opacity");
        EXPECT_EQ(host_u8(free_gpu), expected_free);
    }

    TEST_P(PortableDensifyOps, RefineCountsAndMedian) {
        constexpr size_t n = 5003;
        std::vector<bool> b0(n), b1(n / 2);
        for (size_t i = 0; i < n; ++i)
            b0[i] = i % 3 == 0;
        for (size_t i = 0; i < b1.size(); ++i)
            b1[i] = i % 7 == 1;
        const auto f0 = unit_values(n, -0.5f);
        const auto f1 = values(777, 1.f, 5);
        auto counts = Tensor::full({4}, -1.f, Device::GPU, DataType::Int64);
        refine().counts(gpu_bool(b0), gpu_bool(b1), gpu(f0), gpu(f1), counts);
        const auto positive = [](const std::vector<float>& v) {
            return static_cast<int64_t>(std::count_if(v.begin(), v.end(), [](float x) { return x > 0.f; }));
        };
        EXPECT_EQ(host_i64(counts),
                  (std::vector<int64_t>{static_cast<int64_t>(std::count(b0.begin(), b0.end(), true)),
                                        static_cast<int64_t>(std::count(b1.begin(), b1.end(), true)), positive(f0),
                                        positive(f1)}));
        Tensor absent;
        refine().counts(gpu_bool(b0), absent, absent, gpu(f1), counts);
        EXPECT_EQ(host_i64(counts), (std::vector<int64_t>{static_cast<int64_t>(std::count(b0.begin(), b0.end(), true)),
                                                          0, 0, positive(f1)}));

        const float nan = std::numeric_limits<float>::quiet_NaN();
        const std::vector<float> fixture{nan, 0.f, -2.f, 9.f, 1.f, 4.f, 4.f, 100.f};
        auto fixture_gpu = gpu(fixture);
        refine().normalize_positive_median(fixture_gpu);
        expect_same(host_f(fixture_gpu), {0.f, 0.f, -0.5f, 2.25f, 0.25f, 1.f, 1.f, 25.f}, "refine.median.fixture");

        auto large = values(4097, 2.f, 3);
        large[10] = nan;
        std::vector<float> clean = large;
        std::vector<float> positive_values;
        for (auto& v : clean) {
            if (std::isnan(v))
                v = 0.f;
            if (v > 0.f)
                positive_values.push_back(v);
        }
        const float median = radix_sorted(positive_values)[positive_values.size() / 2];
        for (auto& v : clean)
            v /= std::max(median, 1e-9f);
        auto large_gpu = gpu(large);
        refine().normalize_positive_median(large_gpu);
        expect_close(host_f(large_gpu), clean, "refine.median.large", 0.f, 1e-6f);

        auto none_gpu = gpu({-1.f, 0.f, nan, -3.f});
        refine().normalize_positive_median(none_gpu);
        expect_same(host_f(none_gpu), {0.f, 0.f, 0.f, 0.f}, "refine.median.none");
    }

    TEST_P(PortableDensifyOps, RefineScreenShareAndMasks) {
        constexpr size_t n = 257;
        const auto shares = unit_values(n);
        const auto log_scales = unit_values(3 * n, -2.f);
        const auto frozen_values = unit_values(129);
        std::vector<bool> frozen(129);
        for (size_t i = 0; i < frozen.size(); ++i)
            frozen[i] = frozen_values[i] > 0.5f;
        constexpr float limit = 0.3f;
        auto scales_gpu = gpu_rows(log_scales, 3);
        refine().clip_scales(scales_gpu, gpu(shares), gpu_bool(frozen), limit);
        auto expected_scales = log_scales;
        for (size_t i = 0; i < n; ++i) {
            if ((i < frozen.size() && frozen[i]) || !(shares[i] > limit))
                continue;
            const float delta = std::min(std::log(shares[i] / limit), std::log(1.5f));
            const float* s = &log_scales[i * 3];
            const float largest = std::max(s[0], std::max(s[1], s[2]));
            const size_t axis = largest == s[0] ? 0 : (largest == s[1] ? 1 : 2);
            expected_scales[i * 3 + axis] -= delta;
        }
        expect_close(host_f(scales_gpu), expected_scales, "refine.clip");

        const auto error = unit_values(n, -0.2f);
        auto scores_gpu = Tensor::full({n}, -1.f, Device::GPU);
        refine().oversize_scores(gpu(error), gpu(shares), gpu_bool(frozen), scores_gpu, limit);
        std::vector<float> expected_scores(n);
        for (size_t i = 0; i < n; ++i) {
            if ((i < frozen.size() && frozen[i]) || !(shares[i] > limit) || !(error[i] > 0.f))
                continue;
            expected_scores[i] = std::sqrt(error[i]) * (shares[i] / limit);
        }
        expect_close(host_f(scores_gpu), expected_scores, "refine.oversize");

        const auto opacity = unit_values(n);
        auto rotations = unit_values(4 * n);
        std::fill_n(rotations.begin(), 4, 0.f);
        std::fill_n(rotations.begin() + 4 * 7, 4, 0.f);
        rotations[4 * 7 + 2] = 5e-5f;
        auto dead_gpu = Tensor::zeros({n}, Device::GPU, DataType::Bool);
        auto rotation_gpu = Tensor::zeros({n}, Device::GPU, DataType::Bool);
        refine().dead_mask(gpu(opacity), gpu_rows(rotations, 4), dead_gpu, 0.2f);
        refine().rotation_mask(gpu_rows(rotations, 4), rotation_gpu);
        const auto dead = host_u8(dead_gpu), rotation = host_u8(rotation_gpu);
        for (size_t i = 0; i < n; ++i) {
            const float* q = &rotations[i * 4];
            const bool zero = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] < 1e-8f;
            EXPECT_EQ(rotation[i], zero ? 1 : 0) << i;
            EXPECT_EQ(dead[i], opacity[i] <= 0.2f || zero ? 1 : 0) << i;
        }
    }

    // ---- MCMC ------------------------------------------------------------

    TEST_P(PortableDensifyOps, McmcRelocate) {
        constexpr size_t n = 257;
        mcmc().initialize(51);
        auto opacity = unit_values(n, 0.01f);
        for (auto& o : opacity)
            o = std::clamp(o, 0.01f, 0.99f);
        const auto scales = unit_values(3 * n, 0.1f);
        std::vector<int32_t> ratios(n);
        for (size_t i = 0; i < n; ++i)
            ratios[i] = static_cast<int32_t>(1 + i % 50);
        auto new_opacity = Tensor::zeros({n}, Device::GPU);
        auto new_scales = Tensor::zeros({n, 3}, Device::GPU);
        mcmc().relocate(gpu(opacity), gpu_rows(scales, 3), gpu_i32(ratios), new_opacity, new_scales, 0.005f);

        std::vector<float> coefficients(51 * 51, 0.f);
        for (int row = 0; row < 51; ++row) {
            float binom = 1.f;
            for (int k = 0; k <= row; ++k) {
                coefficients[row * 51 + k] = binom * (k % 2 == 0 ? 1.f : -1.f) *
                                             static_cast<float>(1.0 / std::sqrt(static_cast<double>(k + 1)));
                if (k < row)
                    binom *= static_cast<float>(row - k) / static_cast<float>(k + 1);
            }
        }
        std::vector<float> expected_opacity(n), expected_scales(3 * n);
        for (size_t i = 0; i < n; ++i) {
            const int count = ratios[i];
            const float o = std::clamp(opacity[i], 1e-6f, 1.0f - 1e-6f);
            const float updated = std::min(std::max(1.0f - std::pow(1.0f - o, 1.0f / static_cast<float>(count)),
                                                    std::max(1e-6f, 0.005f)),
                                           1.0f - 1e-6f);
            expected_opacity[i] = updated;
            float denom = 0.f;
            for (int row = 1; row <= count; ++row)
                for (int k = 0; k <= row - 1; ++k)
                    denom += coefficients[(row - 1) * 51 + k] * std::pow(updated, static_cast<float>(k + 1));
            float safe = std::max(std::abs(denom), 1e-8f);
            if (denom < 0.f)
                safe = -safe;
            const float coeff = std::clamp(o / safe, -1e6f, 1e6f);
            for (int d = 0; d < 3; ++d)
                expected_scales[i * 3 + d] = std::max(std::abs(coeff * scales[i * 3 + d]), 1e-10f);
        }
        // Up to 1275 pow terms of alternating sign per row: a looser budget.
        expect_close(host_f(new_opacity), expected_opacity, "mcmc.relocate.opacity");
        expect_close(host_f(new_scales), expected_scales, "mcmc.relocate.scales", 1e-4f, 1e-3f);
    }

    TEST_P(PortableDensifyOps, McmcNoiseFollowsTheCurandStream) {
        constexpr size_t n = 257;
        constexpr uint64_t seed = 12345;
        const auto raw_opacity = unit_values(n, -4.f);
        const auto raw_scales = unit_values(3 * n, -2.f);
        const auto quats = unit_values(4 * n);
        const auto means = unit_values(3 * n);
        const auto frozen = every(n, 3);
        auto means_gpu = gpu_rows(means, 3);
        mcmc().noise(gpu(raw_opacity), gpu_rows(raw_scales, 3), gpu_rows(quats, 4), gpu_bool(frozen), means_gpu, seed, 0.2f);
        auto expected = means;
        for (size_t i = 0; i < n; ++i) {
            if (frozen[i])
                continue;
            const auto noise = curand_normal4(curand_block(seed, i));
            float w = quats[i * 4], x = quats[i * 4 + 1], y = quats[i * 4 + 2], z = quats[i * 4 + 3];
            const float inv = std::min(1.0f / std::sqrt(x * x + y * y + z * z + w * w), 1e12f);
            x *= inv, y *= inv, z *= inv, w *= inv;
            const float r[3][3] = {{1.f - 2.f * (y * y + z * z), 2.f * (x * y - w * z), 2.f * (x * z + w * y)},
                                   {2.f * (x * y + w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - w * x)},
                                   {2.f * (x * z - w * y), 2.f * (y * z + w * x), 1.f - 2.f * (x * x + y * y)}};
            float s2[3];
            for (int d = 0; d < 3; ++d)
                s2[d] = std::exp(2.f * raw_scales[i * 3 + d]);
            const float opacity = 1.f / (1.f + std::exp(-raw_opacity[i]));
            const float factor = 0.2f / (1.f + std::exp(100.f * opacity - 0.5f));
            for (int row = 0; row < 3; ++row) {
                float t = 0.f;
                for (int col = 0; col < 3; ++col) {
                    float cov = 0.f;
                    for (int k = 0; k < 3; ++k)
                        cov += r[row][k] * s2[k] * r[col][k];
                    t += cov * noise[col];
                }
                expected[i * 3 + row] += factor * t;
            }
        }
        expect_close(host_f(means_gpu), expected, "mcmc.noise", 1e-6f, 1e-4f);
    }

    TEST_P(PortableDensifyOps, McmcRowsAndFold) {
        constexpr size_t rows = 17;
        const auto means = unit_values(3 * rows), sh0 = unit_values(3 * rows, 0.3f), scales = unit_values(3 * rows, -2.f);
        const auto quats = unit_values(4 * rows, 0.2f), opacity = unit_values(rows, -1.f);
        auto means_gpu = gpu_rows(means, 3), sh0_gpu = Tensor::from_vector(sh0, {rows, 1, 3}, Device::GPU);
        auto scales_gpu = gpu_rows(scales, 3), quats_gpu = gpu_rows(quats, 4), opacity_gpu = gpu(opacity);
        const std::vector<int64_t> src{0, 3, 5, 40}, dst{9, 11, 16, 2};
        mcmc().copy_rows(gpu_i64(src), gpu_i64(dst), {means_gpu, sh0_gpu, scales_gpu, quats_gpu, opacity_gpu});
        auto em = means, es = sh0, esc = scales, eq = quats, eo = opacity;
        for (size_t i = 0; i < 3; ++i) {
            const auto s = static_cast<size_t>(src[i]), d = static_cast<size_t>(dst[i]);
            for (int c = 0; c < 3; ++c) {
                em[d * 3 + c] = em[s * 3 + c];
                es[d * 3 + c] = es[s * 3 + c];
                esc[d * 3 + c] = esc[s * 3 + c];
            }
            for (int c = 0; c < 4; ++c)
                eq[d * 4 + c] = eq[s * 4 + c];
            eo[d] = eo[s];
        }
        expect_same(host_f(means_gpu), em, "mcmc.copy.means");
        expect_same(host_f(sh0_gpu), es, "mcmc.copy.sh0");
        expect_same(host_f(scales_gpu), esc, "mcmc.copy.scales");
        expect_same(host_f(quats_gpu), eq, "mcmc.copy.quats");
        expect_same(host_f(opacity_gpu), eo, "mcmc.copy.opacity");

        const auto new_scales = unit_values(12, -5.f), new_opacity = unit_values(4, -3.f);
        mcmc().update_rows(gpu_i64({9, 11, 16, 17}), gpu_rows(new_scales, 3), gpu(new_opacity), scales_gpu, opacity_gpu);
        const int64_t targets[3] = {9, 11, 16};
        for (size_t i = 0; i < 3; ++i) {
            const auto t = static_cast<size_t>(targets[i]);
            for (int c = 0; c < 3; ++c)
                esc[t * 3 + c] = new_scales[i * 3 + c];
            eo[t] = new_opacity[i];
        }
        expect_same(host_f(scales_gpu), esc, "mcmc.update.scales");
        expect_same(host_f(opacity_gpu), eo, "mcmc.update.opacity");

        constexpr size_t n = 257;
        const auto error_max = unit_values(n), dens = unit_values(2 * n, 0.5f);
        auto max_gpu = gpu(error_max), dens_gpu = gpu_rows(dens, n);
        mcmc().fold_error(max_gpu, dens_gpu);
        std::vector<float> expected_max(n);
        for (size_t i = 0; i < n; ++i)
            expected_max[i] = std::max(error_max[i], dens[n + i]);
        expect_same(host_f(max_gpu), expected_max, "mcmc.fold_error.max");
        expect_same(host_f(dens_gpu), std::vector<float>(2 * n, 0.f), "mcmc.fold_error.rows");
    }

    TEST_P(PortableDensifyOps, McmcSampleFollowsTheCurandStream) {
        constexpr size_t n = 4099;
        constexpr size_t samples = 1025;
        constexpr uint64_t seed = 98765;
        // Integer weights: every partial sum is exact, so the draws depend only
        // on the uniform stream and the search.
        std::vector<float> weights(n);
        for (size_t i = 0; i < n; ++i)
            weights[i] = static_cast<float>((i * 37) % 5);
        const auto opacity = unit_values(n);
        const auto raw_scales = unit_values(3 * n, -2.f);
        std::vector<int64_t> alive;
        for (size_t i = 0; i < n; i += 3)
            alive.push_back(static_cast<int64_t>(i));

        for (const auto domain : {ops::SampleDomain::AliveIndices, ops::SampleDomain::All}) {
            const bool by_alive = domain == ops::SampleDomain::AliveIndices;
            auto indices = Tensor::empty({samples}, Device::GPU, DataType::Int64);
            auto sampled_opacity = Tensor::zeros({samples}, Device::GPU);
            auto sampled_scales = Tensor::zeros({samples, 3}, Device::GPU);
            mcmc().sample(gpu(weights), gpu(opacity), gpu_rows(raw_scales, 3), by_alive ? gpu_i64(alive) : Tensor(),
                          indices, sampled_opacity, sampled_scales, domain, seed);

            std::vector<float> cumsum;
            float running = 0.f;
            const size_t categories = by_alive ? alive.size() : n;
            for (size_t c = 0; c < categories; ++c) {
                running += weights[by_alive ? static_cast<size_t>(alive[c]) : c];
                cumsum.push_back(running);
            }
            std::vector<int64_t> expected_indices(samples);
            std::vector<float> expected_opacity(samples), expected_scales(3 * samples);
            for (size_t i = 0; i < samples; ++i) {
                const float u = curand_uniform(curand_block(seed, i)[0]) * cumsum.back();
                const size_t selected = static_cast<size_t>(
                    std::min<ptrdiff_t>(std::lower_bound(cumsum.begin(), cumsum.end(), u) - cumsum.begin(),
                                        static_cast<ptrdiff_t>(categories) - 1));
                const auto row = by_alive ? static_cast<size_t>(alive[selected]) : selected;
                expected_indices[i] = static_cast<int64_t>(row);
                expected_opacity[i] = opacity[row];
                for (int d = 0; d < 3; ++d)
                    expected_scales[i * 3 + d] = std::exp(raw_scales[row * 3 + d]);
            }
            EXPECT_EQ(host_i64(indices), expected_indices) << (by_alive ? "alive" : "all");
            expect_same(host_f(sampled_opacity), expected_opacity, "mcmc.sample.opacity");
            expect_close(host_f(sampled_scales), expected_scales, "mcmc.sample.scales");
        }

        auto indices = Tensor::full({4}, 7.f, Device::GPU, DataType::Int64);
        auto sampled_opacity = Tensor::full({4}, 1.f, Device::GPU);
        auto sampled_scales = Tensor::full({4, 3}, 1.f, Device::GPU);
        mcmc().sample(gpu({0.f, 0.f}), gpu({0.5f, 0.5f}), gpu_rows(std::vector<float>(6, 0.f), 3), Tensor(), indices,
                      sampled_opacity, sampled_scales, ops::SampleDomain::All, seed);
        EXPECT_EQ(host_i64(indices), std::vector<int64_t>(4, 0));
        expect_same(host_f(sampled_scales), std::vector<float>(12, 0.f), "mcmc.sample.empty");
    }

    INSTANTIATE_TEST_SUITE_P(Backends, PortableDensifyOps, testing::Values(GpuBackend::Metal, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

} // namespace
