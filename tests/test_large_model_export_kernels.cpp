/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The export kernels past the 32-bit wraps of SH degree 3. The SH palette of the SOG, SSOG and HTML exports reads a
// point's SH from the float4 store at about point x 48 floats, which passes 2^32 at row 89,478,464.
//
// The SH3 assignment (exact and screened, the palette's last k-means step) runs over a store of 95,444,224 points
// whose SH is W in the first 256 rows, V in the blocks around the float4 and Q16 wraps and zero elsewhere, against
// three centroids 0, V and W: every point's label is known. A 32-bit offset reads another point's SH for the far
// points, and their label changes.
//
// Decimation reads a point's canonical SH at i x 45 floats, which passes 2^32 at row 95,443,717. A grid of 95,444,224
// points, one unit apart, holds four pairs of points a hundredth of a unit apart: two below every wrap and two past
// the canonical wrap. Each pair member's best merge candidate is its partner, at a cost that depends only on the pair
// (its SH differs, so the SH term counts); the same pairs alone in a small model give the reference costs, bit for
// bit. A 32-bit offset reads another row's SH for the far pairs and changes their cost.
//
// Tiers by device memory: see large_model_test_support.hpp. Each test's memory is in its skip message.

#include "large_model_test_support.hpp"

#include "core/tensor_backend.hpp"
#include "core/tensor_export.hpp"

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <string>
#include <array>
#include <tuple>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::ShFormat;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;

    constexpr float kV = 0.5f, kW = -0.5f;

    // W in H, V in F and Q, zero in Z.
    float palette_value(const size_t row, uint32_t) { return row < 256 ? kW : kV; }

    // The label each block's points must get against the centroids {0, V, W}.
    int32_t expected_label(const lm::Block& block) {
        return !block.holds_values ? 0 : block.begin == 0 ? 2 : 1;
    }

    class LargeModelExportKmeans : public ::testing::TestWithParam<std::tuple<GpuBackend, bool>> {
    protected:
        void SetUp() override {
            const auto [backend, fast] = GetParam();
            if (const auto reason = lm::skip_reason(lm::store_bytes(ShFormat::Float32, lm::kRows) + lm::kRows * 16))
                GTEST_SKIP() << *reason;
            if (!lfs::core::gpu_backend_available(backend))
                GTEST_SKIP() << lfs::core::gpu_backend_name(backend) << " device unavailable";
            session_.emplace(backend);
            ASSERT_TRUE(session_->switched());
        }

        void TearDown() override { session_.reset(); }

        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
    };

    TEST_P(LargeModelExportKmeans, AssignmentReadsPointsPastTheWrap) {
        const bool fast = std::get<1>(GetParam());
        auto store = lm::zero_store(ShFormat::Float32, lm::kRows, Device::GPU);
        if (!store)
            GTEST_SKIP() << "this device can't allocate or clear " << lm::kRows << " rows of fp32 SH";
        lm::place_blocks(*store, lm::compact_blocks(ShFormat::Float32, lm::kRows, palette_value));

        std::vector<float> centroids(3 * lm::kCells, 0.0f);
        std::fill(centroids.begin() + lm::kCells, centroids.begin() + 2 * lm::kCells, kV);
        std::fill(centroids.begin() + 2 * lm::kCells, centroids.end(), kW);
        const std::vector<float> norms{0.0f, kV * kV * lm::kCells, kW * kW * lm::kCells};
        const Tensor centroid_tensor = Tensor::from_vector(centroids, {size_t{3}, size_t{lm::kCells}}, Device::GPU);
        const Tensor norm_tensor = Tensor::from_vector(norms, {size_t{3}}, Device::GPU);
        Tensor labels = Tensor::full({lm::kRows}, -1.0f, Device::GPU, DataType::Int32);

        lfs::core::assign_sh3(store->values, centroid_tensor, norm_tensor, labels, fast);

        for (const auto& block : lm::kBlocks) {
            const auto got = lm::host_copy<int32_t>(labels.slice(0, block.begin, block.end));
            const int32_t want = expected_label(block);
            size_t wrong = 0, first = 0;
            for (size_t i = 0; i < got.size(); ++i)
                if (got[i] != want && wrong++ == 0)
                    first = i;
            EXPECT_EQ(wrong, 0u) << (fast ? "screened" : "exact") << " assignment, block " << block.name << ": "
                                 << wrong << " of " << got.size() << " labels wrong; row " << block.begin + first
                                 << " got " << (wrong ? got[first] : want) << " instead of " << want;
        }
    }

    // ---- Decimation ----------------------------------------------------------------------------------------------

    // The first member of each pair; the second is the next row.
    constexpr std::array<size_t, 4> kPairs{5, 89'478'470, 95'443'720, 95'444'000};
    static_assert(kPairs[2] > lm::kCanonicalWrapRow && kPairs[3] + 1 < lm::kRows);

    struct Splats {
        Tensor position, rotation, scale, opacity, dc, sh;
    };

    // `rows` points of a grid one unit apart, the pairs at `pair_rows` moved together, every SH zero except the first
    // member's of each pair. Returns nullopt when the device can't hold it.
    std::optional<Splats> decimation_splats(const size_t rows, const std::vector<size_t>& pair_rows) {
        try {
            Tensor host = Tensor::empty({rows, size_t{3}}, Device::CPU, DataType::Float32);
            auto* const p = host.ptr<float>();
            for (size_t i = 0; i < rows; ++i) {
                p[i * 3 + 0] = static_cast<float>(i % 1024);
                p[i * 3 + 1] = static_cast<float>(i / 1024 % 1024);
                p[i * 3 + 2] = static_cast<float>(i / (1024 * 1024));
            }
            for (const size_t a : pair_rows) {
                p[(a + 1) * 3 + 0] = p[a * 3 + 0] + 0.01f;
                p[(a + 1) * 3 + 1] = p[a * 3 + 1];
                p[(a + 1) * 3 + 2] = p[a * 3 + 2];
            }
            Splats s;
            s.position = host.gpu();
            host = Tensor{};
            Tensor rotation = Tensor::zeros({rows, size_t{4}}, Device::GPU);
            rotation.slice(1, 0, 1).fill_(1.0f);
            s.rotation = rotation;
            s.scale = Tensor::full({rows, size_t{3}}, -3.0f, Device::GPU);
            s.opacity = Tensor::zeros({rows}, Device::GPU);
            s.dc = Tensor::zeros({rows, size_t{3}}, Device::GPU);
            s.sh = Tensor::zeros({rows, size_t{lm::kRest}, size_t{3}}, Device::GPU);
            // Small enough that a pair's cost stays below a grid neighbour's (one unit away), large enough that a
            // wrong SH read changes the cost.
            std::vector<float> values(lm::kCells);
            for (uint32_t c = 0; c < lm::kCells; ++c)
                values[c] = 0.001f * static_cast<float>(c + 1);
            const Tensor row = Tensor::from_vector(values, {size_t{1}, size_t{lm::kRest}, size_t{3}}, Device::GPU);
            for (const size_t a : pair_rows)
                s.sh.slice(0, a, a + 1).copy_(row);
            return s;
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }

    class LargeModelDecimate : public ::testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            // Splats and SH (75 floats a point), the cost cache (32), neighbours (16), candidates (8) and the Morton
            // keys and order (3).
            if (const auto reason = lm::skip_reason(lm::kRows * 134 * sizeof(float), lm::kRows * 8 * sizeof(float)))
                GTEST_SKIP() << *reason;
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " device unavailable";
            session_.emplace(GetParam());
            ASSERT_TRUE(session_->switched());
        }

        void TearDown() override { session_.reset(); }

        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
    };

    TEST_P(LargeModelDecimate, CandidateCostsReadShPastTheWrap) {
        // The reference: the four pairs alone, in the same places.
        std::vector<size_t> small_pairs;
        std::vector<size_t> big_rows;
        for (size_t k = 0; k < kPairs.size(); ++k) {
            small_pairs.push_back(2 * k);
            big_rows.push_back(kPairs[k]);
            big_rows.push_back(kPairs[k] + 1);
        }
        auto small = decimation_splats(2 * kPairs.size(), small_pairs);
        ASSERT_TRUE(small);
        {
            // Put the small model's points where the big model's pairs are.
            Tensor host = Tensor::empty({2 * kPairs.size(), size_t{3}}, Device::CPU, DataType::Float32);
            auto* const p = host.ptr<float>();
            for (size_t r = 0; r < big_rows.size(); ++r) {
                const size_t i = kPairs[r / 2];
                p[r * 3 + 0] = static_cast<float>(i % 1024) + (r % 2 ? 0.01f : 0.0f);
                p[r * 3 + 1] = static_cast<float>(i / 1024 % 1024);
                p[r * 3 + 2] = static_cast<float>(i / (1024 * 1024));
            }
            small->position = host.gpu();
        }
        std::vector<uint32_t> small_idx, big_idx;
        std::vector<float> small_cost, big_cost;
        lfs::core::decimate_candidates(small->position, small->rotation, small->scale, small->opacity, small->dc,
                                       small->sh, lm::kRest, small_idx, small_cost);

        auto big = decimation_splats(lm::kRows, std::vector<size_t>(kPairs.begin(), kPairs.end()));
        if (!big)
            GTEST_SKIP() << "this device can't hold " << lm::kRows << " points of decimation input";
        lfs::core::decimate_candidates(big->position, big->rotation, big->scale, big->opacity, big->dc, big->sh,
                                       lm::kRest, big_idx, big_cost);
        ASSERT_EQ(big_idx.size(), lm::kRows * 4);

        for (size_t r = 0; r < big_rows.size(); ++r) {
            const size_t row = big_rows[r], partner = r % 2 ? row - 1 : row + 1;
            EXPECT_EQ(small_idx[r * 4], uint32_t(r % 2 ? r - 1 : r + 1)) << "the reference pair of row " << row;
            EXPECT_EQ(big_idx[row * 4], partner) << "row " << row << "'s best merge candidate";
            uint32_t got, want;
            std::memcpy(&got, &big_cost[row * 4], 4);
            std::memcpy(&want, &small_cost[r * 4], 4);
            EXPECT_EQ(got, want) << "row " << row << "'s merge cost: " << big_cost[row * 4] << " instead of "
                                 << small_cost[r * 4];
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, LargeModelDecimate,
                             testing::Values(GpuBackend::CUDA, GpuBackend::Metal, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

    INSTANTIATE_TEST_SUITE_P(
        Backends, LargeModelExportKmeans,
        testing::Combine(testing::Values(GpuBackend::CUDA, GpuBackend::Metal, GpuBackend::Vulkan), testing::Bool()),
        [](const auto& info) {
            return std::string(lfs::core::gpu_backend_name(std::get<0>(info.param))) +
                   (std::get<1>(info.param) ? "_Screened" : "_Exact");
        });

} // namespace
