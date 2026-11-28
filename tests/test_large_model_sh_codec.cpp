/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The core SH codec (sh_codec, and sh_encode for Q16 outputs) past the 32-bit wraps of SH degree 3, on stores of
// 95,444,224 rows: past the float4 wrap (row 89,478,464), the Q16 wrap (row 95,443,712) and the canonical wrap (row
// 95,443,717), and, as a row count, past 2^32 elements of float4 and canonical rows. These are the conversions a
// PLY or SPZ export, a crop, MRNF's mutation commit and gather, the fp32 expansion before a densification, the
// post-training compaction and a project save run over a whole model.
//
// Every format pair runs in each mode its codec accepts:
// - Whole: every row from a big source into a big destination (the destination enumeration passes 2^32);
// - GatherInt32, GatherInt64: indexed rows of H, F and Q from a big source into a small destination;
// - SourceOffset: a row range from a far source offset into a small destination;
// - DestinationOffset: a small source into a far destination offset (the blocks keep their other rows);
// - ScatterInt32, ScatterInt64: canonical rows scattered into far rows of a big destination.
// The big stores hold values only in the blocks of large_model_test_support.hpp; the reference is the CPU codec on
// the compact store of those blocks. A 32-bit index reads zeros or other rows' values and writes into the first rows
// of the destination, which the blocks H and Z check.
//
// Tiers by device memory: see large_model_test_support.hpp. Each case's memory is in its skip message.

#include "large_model_test_support.hpp"

#include "core/tensor_backend.hpp"

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::ShCodec;
    using lfs::core::ShFormat;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;

    enum class Mode { Whole,
                      GatherInt32,
                      GatherInt64,
                      SourceOffset,
                      DestinationOffset,
                      ScatterInt32,
                      ScatterInt64 };

    const char* mode_name(const Mode mode) {
        switch (mode) {
        case Mode::Whole: return "Whole";
        case Mode::GatherInt32: return "GatherInt32";
        case Mode::GatherInt64: return "GatherInt64";
        case Mode::SourceOffset: return "SourceOffset";
        case Mode::DestinationOffset: return "DestinationOffset";
        case Mode::ScatterInt32: return "ScatterInt32";
        case Mode::ScatterInt64: return "ScatterInt64";
        }
        return "?";
    }

    struct Case {
        GpuBackend backend = GpuBackend::Metal;
        Mode mode = Mode::Whole;
        ShFormat source = ShFormat::Float32;
        ShFormat destination = ShFormat::Float32;
    };

    constexpr std::array kFormats{ShFormat::Canonical, ShFormat::Float32, ShFormat::Float16, ShFormat::Q16};

    // The codec's own rules (tensor_sh.cpp): canonical to canonical is no conversion; a canonical source takes
    // indices only to scatter; a scatter goes from canonical rows into resident storage; a Q16 output covers its
    // whole destination, so it takes no destination offset.
    std::vector<Case> all_cases() {
        std::vector<Case> cases;
        for (const GpuBackend backend : {GpuBackend::CUDA, GpuBackend::Metal, GpuBackend::Vulkan})
            for (const Mode mode : {Mode::Whole, Mode::GatherInt32, Mode::GatherInt64, Mode::SourceOffset,
                                    Mode::DestinationOffset, Mode::ScatterInt32, Mode::ScatterInt64})
                for (const ShFormat source : kFormats)
                    for (const ShFormat destination : kFormats) {
                        if (source == ShFormat::Canonical && destination == ShFormat::Canonical)
                            continue;
                        const bool gather = mode == Mode::GatherInt32 || mode == Mode::GatherInt64;
                        const bool scatter = mode == Mode::ScatterInt32 || mode == Mode::ScatterInt64;
                        if (gather && source == ShFormat::Canonical)
                            continue;
                        if (scatter && (source != ShFormat::Canonical || destination == ShFormat::Q16))
                            continue;
                        if (mode == Mode::DestinationOffset && destination == ShFormat::Q16)
                            continue;
                        cases.push_back({backend, mode, source, destination});
                    }
        return cases;
    }

    // Small sides: a gather reads all rows of H, F and Q; a scatter writes all rows of F and Q; the offset cases run
    // two windows, one from the float4 wrap block and one across the Q16 and canonical wrap rows.
    constexpr std::array<std::pair<size_t, size_t>, 2> kWindows{{{lm::kFloat4WrapRow, 300}, {95'443'700, 500}}};
    static_assert(95'443'700 < lm::kQ16WrapRow && 95'443'700 + 500 > lm::kCanonicalWrapRow + 1);

    std::vector<size_t> rows_of(const std::initializer_list<size_t> block_numbers, const unsigned seed) {
        std::vector<size_t> rows;
        for (const size_t b : block_numbers)
            for (size_t r = lm::kBlocks[b].begin; r < lm::kBlocks[b].end; ++r)
                rows.push_back(r);
        std::mt19937 rng(seed);
        std::shuffle(rows.begin(), rows.end(), rng);
        return rows;
    }

    size_t small_rows(const Mode mode) {
        switch (mode) {
        case Mode::GatherInt32:
        case Mode::GatherInt64: return lm::kBlocks[0].rows() + lm::kBlocks[2].rows() + lm::kBlocks[3].rows();
        case Mode::ScatterInt32:
        case Mode::ScatterInt64: return lm::kBlocks[2].rows() + lm::kBlocks[3].rows();
        default: return 500;
        }
    }

    // Device memory of a case: the big stores, the small side and the indices.
    size_t device_bytes(const Case& c) {
        const size_t small = small_rows(c.mode);
        switch (c.mode) {
        case Mode::Whole: return lm::store_bytes(c.source, lm::kRows) + lm::store_bytes(c.destination, lm::kRows);
        case Mode::GatherInt32:
        case Mode::GatherInt64:
        case Mode::SourceOffset:
            return lm::store_bytes(c.source, lm::kRows) + lm::store_bytes(c.destination, small) + small * 8;
        default: return lm::store_bytes(c.destination, lm::kRows) + lm::store_bytes(c.source, small) + small * 8;
        }
    }

    class LargeModelShCodec : public ::testing::TestWithParam<Case> {
    protected:
        void SetUp() override {
            const Case& c = GetParam();
            if (const auto reason = lm::skip_reason(device_bytes(c)))
                GTEST_SKIP() << *reason;
            if (!lfs::core::gpu_backend_available(c.backend))
                GTEST_SKIP() << lfs::core::gpu_backend_name(c.backend) << " device unavailable";
            session_.emplace(c.backend);
            ASSERT_TRUE(session_->switched());
        }

        void TearDown() override { session_.reset(); }

        // A zeroed device store of `rows`, or nullopt with the reason in unavailable_ when the device can't hold or
        // clear it (the caller skips).
        std::optional<lm::Store> device_store(const ShFormat format, const size_t rows) {
            auto store = lm::zero_store(format, rows, Device::GPU);
            if (!store)
                unavailable_ = std::format("this device can't allocate or clear {} rows of {} ({:.1f} GB)", rows,
                                           lm::format_name(format), lm::store_bytes(format, rows) / lm::kGB);
            return store;
        }

        // The small side of the writing cases: rows of second_value() for the given destination rows, in `format`.
        static lm::Store small_source(const ShFormat format, const std::vector<size_t>& destination_rows) {
            lm::Store canonical = lm::canonical_rows(
                destination_rows.size(), [&](const size_t r) { return std::optional(destination_rows[r]); },
                lm::second_value);
            if (format == ShFormat::Canonical)
                return canonical;
            lm::Store converted = lm::cpu_store(format, destination_rows.size());
            lm::convert(canonical, converted, {.count = destination_rows.size()});
            return converted;
        }

        static lm::Store to_device(const lm::Store& host) {
            return {.format = host.format, .rows = host.rows, .values = host.values.gpu(),
                    .bounds = host.format == ShFormat::Q16 ? host.bounds.gpu() : Tensor{}};
        }

        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
        std::string unavailable_;
    };

    TEST_P(LargeModelShCodec, MatchesTheCpuCodecPastTheWrap) {
        const Case& c = GetParam();
        const std::string pair = std::string(lm::format_name(c.source)) + " to " + lm::format_name(c.destination);

        if (c.mode == Mode::Whole || c.mode == Mode::GatherInt32 || c.mode == Mode::GatherInt64 ||
            c.mode == Mode::SourceOffset) {
            // A big source holding the blocks.
            const lm::Store compact_source = lm::compact_blocks(c.source);
            auto source = device_store(c.source, lm::kRows);
            if (!source)
                GTEST_SKIP() << unavailable_;
            lm::place_blocks(*source, compact_source);

            if (c.mode == Mode::Whole) {
                auto destination = device_store(c.destination, lm::kRows);
                if (!destination)
                    GTEST_SKIP() << unavailable_;
                lm::convert(*source, *destination, {.count = lm::kRows});
                lm::Store reference = lm::cpu_store(c.destination, lm::kCompactRows);
                lm::convert(compact_source, reference, {.count = lm::kCompactRows});
                lm::expect_blocks_match(*destination, reference, pair + ", every row");
                return;
            }

            if (c.mode == Mode::SourceOffset) {
                for (const auto& [first, count] : kWindows) {
                    auto destination = device_store(c.destination, count);
                    if (!destination)
                        GTEST_SKIP() << unavailable_;
                    lm::convert(*source, *destination, {.count = count, .source_offset = first});
                    lm::Store reference = lm::cpu_store(c.destination, count);
                    lm::convert(compact_source, reference, {.count = count, .source_offset = lm::compact_row(first)});
                    lm::expect_store_matches(*destination, reference,
                                             pair + ", " + std::to_string(count) + " rows from row " +
                                                 std::to_string(first));
                }
                return;
            }

            // Gathers: H, F and Q in a shuffled order.
            const std::vector<size_t> rows = rows_of({0, 2, 3}, 4202);
            std::vector<size_t> compact(rows.size());
            std::transform(rows.begin(), rows.end(), compact.begin(), lm::compact_row);
            const bool wide = c.mode == Mode::GatherInt64;
            const Tensor indices = wide ? lm::index_tensor<int64_t>(rows, Device::GPU)
                                        : lm::index_tensor<int32_t>(rows, Device::GPU);
            const Tensor compact_indices = wide ? lm::index_tensor<int64_t>(compact, Device::CPU)
                                                : lm::index_tensor<int32_t>(compact, Device::CPU);
            auto destination = device_store(c.destination, rows.size());
            if (!destination)
                GTEST_SKIP() << unavailable_;
            lm::convert(*source, *destination, {.count = rows.size()}, &indices);
            lm::Store reference = lm::cpu_store(c.destination, rows.size());
            lm::convert(compact_source, reference, {.count = rows.size()}, &compact_indices);
            lm::expect_store_matches(*destination, reference, pair + ", gathered rows");
            return;
        }

        // The writing cases: a big destination holding the blocks, written in F and Q.
        lm::Store reference = lm::compact_blocks(c.destination);
        auto destination = device_store(c.destination, lm::kRows);
        if (!destination)
            GTEST_SKIP() << unavailable_;
        lm::place_blocks(*destination, reference);

        if (c.mode == Mode::DestinationOffset) {
            for (const auto& [first, count] : kWindows) {
                std::vector<size_t> rows(count);
                for (size_t i = 0; i < count; ++i)
                    rows[i] = first + i;
                const lm::Store host = small_source(c.source, rows);
                const lm::Store device = to_device(host);
                lm::convert(device, *destination, {.count = count, .destination_offset = first});
                lm::convert(host, reference, {.count = count, .destination_offset = lm::compact_row(first)});
            }
            lm::expect_blocks_match(*destination, reference, pair + ", written at two far offsets");
            return;
        }

        // Scatters into F and Q in a shuffled order; H and Z must keep their contents.
        const std::vector<size_t> rows = rows_of({2, 3}, 4203);
        std::vector<size_t> compact(rows.size());
        std::transform(rows.begin(), rows.end(), compact.begin(), lm::compact_row);
        const bool wide = c.mode == Mode::ScatterInt64;
        const Tensor indices =
            wide ? lm::index_tensor<int64_t>(rows, Device::GPU) : lm::index_tensor<int32_t>(rows, Device::GPU);
        const Tensor compact_indices = wide ? lm::index_tensor<int64_t>(compact, Device::CPU)
                                            : lm::index_tensor<int32_t>(compact, Device::CPU);
        const lm::Store host = small_source(ShFormat::Canonical, rows);
        const lm::Store device = to_device(host);
        lm::convert(device, *destination, {.count = rows.size(), .scatter = true}, &indices);
        lm::convert(host, reference, {.count = rows.size(), .scatter = true}, &compact_indices);
        lm::expect_blocks_match(*destination, reference, pair + ", scattered rows");
    }

    INSTANTIATE_TEST_SUITE_P(Backends, LargeModelShCodec, testing::ValuesIn(all_cases()), [](const auto& info) {
        const Case& c = info.param;
        return std::string(lfs::core::gpu_backend_name(c.backend)) + "_" + mode_name(c.mode) + "_" +
               lm::format_name(c.source) + "To" + lm::format_name(c.destination);
    });

} // namespace
