/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The RAD page quantizer's resident path past the SH wraps: when the viewer holds a whole model, rad_page_quantize
// fills a LOD pool page from the resident rows, and it read a row's SH at a Q16 cell index and a float4 slot index
// computed in 32 bits, which wrap at SH degree 3 at row 95,443,712 (Q16) and 89,478,464 (f32 and f16). A resident
// store of 95,444,224 rows holds the large-model blocks of large_model_test_support.hpp; one page over the block at
// the wrap (Q16: rows 95,443,456 to the end; f32 and f16: 89,478,400 to 89,478,911) is quantized from it and must
// equal, byte for byte, the same page quantized from those blocks alone: a store of 2,304 rows, where nothing wraps,
// with the same attributes in the same rows. A 32-bit index reads other rows' SH, and the page's SH maxima and its
// quantized SH change.
//
// Runs with LFS_TEST_LARGE_SH=1: the store (8.6 GB in Q16, 9.2 GB in f16, 18.3 GB in f32) and 5.3 GB of attributes.

#include "large_model_test_support.hpp"

#include "core/tensor_backend.hpp"
#include "core/tensor_rad.hpp"

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::RadPagePool;
    using lfs::core::RadPageSources;
    using lfs::core::ShFormat;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;
    namespace radq = lfs::core::radq;

    constexpr uint32_t kPageSplats = 1024; // one page, whole 32-splat tiles
    constexpr uint32_t kSlots = 12;        // SH degree 3
    // Floats a row: means 3, sh0 3, rotation 4, log scales 3, opacity 1.
    constexpr std::array<size_t, 5> kDims{3, 3, 4, 3, 1};

    // An attribute value of compact row `row`, column `column` of attribute `a`: small and exact in fp32.
    float attribute(const size_t row, const size_t a, const size_t column) {
        const int v = static_cast<int>((row * 7 + a * 31 + column * 13) % 251) - 125;
        return a == 2 ? (column == 0 ? 1.0f : 0.0f) + v / 1024.0f : v / 128.0f;
    }

    // The attributes of `rows` rows: zero, except the rows of every block, which take their compact row's values
    // (at their own rows in the big store, at their compact rows in the small one).
    std::array<Tensor, 5> attributes(const size_t rows, const bool big) {
        std::array<Tensor, 5> result;
        for (size_t a = 0; a < kDims.size(); ++a) {
            result[a] = Tensor::zeros({rows * kDims[a]}, Device::GPU);
            for (const auto& block : lm::kBlocks) {
                Tensor host = Tensor::empty({block.rows() * kDims[a]}, Device::CPU, DataType::Float32);
                auto* const data = host.ptr<float>();
                for (size_t i = 0; i < block.rows(); ++i)
                    for (size_t c = 0; c < kDims[a]; ++c)
                        data[i * kDims[a] + c] = attribute(block.compact + i, a, c);
                const size_t first = big ? block.begin : block.compact;
                result[a].slice(0, first * kDims[a], (first + block.rows()) * kDims[a]).copy_(host.gpu());
            }
        }
        return result;
    }

    // Quantizes one page of `count` resident rows from row `offset` and returns the pool's regions.
    std::vector<std::vector<uint8_t>> quantize_page(const lm::Store& store, const std::array<Tensor, 5>& a,
                                                    const size_t offset, const uint32_t count) {
        RadPagePool pool{.page_splats = kPageSplats, .sh_slots = kSlots};
        const size_t n = kPageSplats;
        const std::array<size_t, 7> sizes{n * 12, n * 8, n * kSlots * 4, n * 8, n * 8, n * 2, radq::kPageFrameBytes};
        for (size_t i = 0; i < sizes.size(); ++i)
            pool.regions[i] = Tensor::zeros({sizes[i]}, Device::GPU, DataType::UInt8);
        const RadPageSources source{.means = a[0],
                                    .sh0 = a[1],
                                    .shN = store.values,
                                    .rotation = a[2],
                                    .scaling = a[3],
                                    .opacity = a[4],
                                    .shN_bounds = store.format == ShFormat::Q16 ? store.bounds : Tensor{},
                                    .offset = static_cast<uint32_t>(offset),
                                    .count = count,
                                    .sh_rest = lm::kRest,
                                    .sh_q16 = store.format == ShFormat::Q16};
        lfs::core::rad_page_quantize(source, pool, 0);
        std::vector<std::vector<uint8_t>> regions;
        for (size_t i = 0; i < sizes.size(); ++i)
            regions.push_back(lm::host_copy<uint8_t>(pool.regions[i]));
        return regions;
    }

    class LargeModelRadPages : public ::testing::TestWithParam<std::tuple<GpuBackend, ShFormat>> {
    protected:
        void SetUp() override {
            const auto [backend, format] = GetParam();
            const size_t attribute_bytes = lm::kRows * 14 * sizeof(float);
            if (const auto reason = lm::skip_reason(lm::store_bytes(format, lm::kRows) + attribute_bytes))
                GTEST_SKIP() << *reason;
            if (!lfs::core::gpu_backend_available(backend))
                GTEST_SKIP() << lfs::core::gpu_backend_name(backend) << " device unavailable";
            session_.emplace(backend);
            ASSERT_TRUE(session_->switched());
        }

        void TearDown() override { session_.reset(); }

        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
    };

    TEST_P(LargeModelRadPages, ResidentPagePastTheWrapMatchesTheSameRowsAlone) {
        const auto [backend, format] = GetParam();
        const auto& block = format == ShFormat::Q16 ? lm::kBlocks[3] : lm::kBlocks[2];
        const auto count = static_cast<uint32_t>(block.rows());

        // The page from the blocks alone.
        lm::Store small = lm::compact_blocks(format);
        small.values = small.values.gpu();
        if (format == ShFormat::Q16)
            small.bounds = small.bounds.gpu();
        const auto expected = quantize_page(small, attributes(lm::kCompactRows, false), block.compact, count);

        // The same page from the resident store of the large model.
        auto big = lm::zero_store(format, lm::kRows, Device::GPU);
        if (!big)
            GTEST_SKIP() << std::format("this device can't allocate or clear {} rows of {} ({:.1f} GB)", lm::kRows,
                                        lm::format_name(format),
                                        static_cast<double>(lm::store_bytes(format, lm::kRows)) / lm::kGB);
        lm::place_blocks(*big, small);
        const auto got = quantize_page(*big, attributes(lm::kRows, true), block.begin, count);

        static constexpr std::array<const char*, 7> kRegions{"positions", "colours", "SH", "rotations",
                                                              "scales",    "opacities", "page frame"};
        ASSERT_EQ(got.size(), expected.size());
        for (size_t r = 0; r < got.size(); ++r) {
            ASSERT_EQ(got[r].size(), expected[r].size()) << kRegions[r];
            size_t wrong = 0, first = 0;
            for (size_t i = 0; i < got[r].size(); ++i)
                if (got[r][i] != expected[r][i] && wrong++ == 0)
                    first = i;
            EXPECT_EQ(wrong, 0u) << lm::format_name(format) << " page of rows " << block.begin << "-"
                                 << block.end - 1 << ", " << kRegions[r] << ": " << wrong << " of " << got[r].size()
                                 << " bytes differ from the same rows alone; first at byte " << first;
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        Backends, LargeModelRadPages,
        testing::Combine(testing::Values(GpuBackend::CUDA, GpuBackend::Metal, GpuBackend::Vulkan),
                         testing::Values(ShFormat::Q16, ShFormat::Float16, ShFormat::Float32)),
        [](const auto& info) {
            return std::string(lfs::core::gpu_backend_name(std::get<0>(info.param))) + "_" +
                   lm::format_name(std::get<1>(info.param));
        });

} // namespace
