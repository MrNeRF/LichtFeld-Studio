/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The training SH kernels past the 32-bit wraps that the base does not fix yet:
// - the float4 slot index, (row / 32) * 384 + slot * 32 + row % 32 at SH degree 3, which passes 2^32 in the block
//   of row 357,913,920: the byte-moment gather of MRNF's and MCMC's compaction, the range decode of IEEE f16
//   storage and the row zeroing of SH gradients all address slots through it;
// Each store holds values only in the first 256 rows and in the four swizzle blocks around the wrap, rows
// 357,913,856 to 357,913,983, and the results are checked against the values those rows hold.
//
// Tiers by device memory: see large_model_test_support.hpp. Each test's memory is in its skip message.

#include "large_model_test_support.hpp"

#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/ops/sh.hpp"

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;
    namespace ops = lfs::gpu_ops;

    // Rows of the slot tests: the wrap block and one more.
    constexpr size_t kSlotRows = lm::kSlotWrapRow + 2 * lm::kReorder;
    constexpr size_t kFarBegin = lm::kSlotWrapRow - 2 * lm::kReorder; // four swizzle blocks: two before the wrap
    constexpr size_t kHeadRows = 256;
    static_assert(kFarBegin % lm::kReorder == 0 && kSlotRows % lm::kReorder == 0);

    // A byte and a value for each row and cell (48 cells a row in float4 storage, the padding included).
    uint8_t byte_of(const size_t row, const uint32_t cell) {
        return static_cast<uint8_t>(1 + (row * lm::kFloatWidth + cell) % 251);
    }
    float value_of(const size_t row, const uint32_t cell) { return cell < lm::kCells ? lm::value(row, cell) : 0.0f; }

    // Index of a row's cell in float4 storage, relative to the first row of a 32-aligned range.
    size_t cell_index(const size_t row, const uint32_t cell, const size_t first_row) {
        return ((row / lm::kReorder - first_row / lm::kReorder) * lm::kSlots + cell / 4) * lm::kReorder * 4 +
               row % lm::kReorder * 4 + cell % 4;
    }

    // Host bytes of rows [first, first + rows) of a float4 store of T, from fill(row, cell).
    template <class T, class Fill>
    Tensor float4_rows(const size_t first, const size_t rows, const DataType dtype, Fill&& fill) {
        Tensor host = Tensor::zeros({rows * lm::kFloatWidth}, Device::CPU, dtype);
        auto* const data = static_cast<T*>(host.data_ptr());
        for (size_t r = first; r < first + rows; ++r)
            for (uint32_t c = 0; c < lm::kFloatWidth; ++c)
                data[cell_index(r, c, first)] = fill(r, c);
        return host;
    }

    // Writes the head rows and the far rows of a float4 store of `dtype` (fp32, f16 or bytes).
    void write_rows(Tensor& store, const DataType dtype) {
        for (const auto& [first, rows] : {std::pair{size_t{0}, kHeadRows}, std::pair{kFarBegin, kSlotRows - kFarBegin}}) {
            Tensor host = dtype == DataType::UInt8 ? float4_rows<uint8_t>(first, rows, dtype, byte_of)
                                                   : float4_rows<float>(first, rows, DataType::Float32, value_of);
            if (dtype == DataType::Float16)
                host = host.to(DataType::Float16);
            store.slice(0, first * lm::kFloatWidth, (first + rows) * lm::kFloatWidth).copy_(host.gpu());
        }
    }

    std::vector<size_t> far_and_head_rows(const unsigned seed) {
        std::vector<size_t> rows;
        for (size_t r = kFarBegin; r < kSlotRows; ++r)
            rows.push_back(r);
        for (size_t r = 0; r < kHeadRows; ++r)
            rows.push_back(r);
        std::mt19937 rng(seed);
        std::shuffle(rows.begin(), rows.end(), rng);
        return rows;
    }

    class LargeModelShSlots : public ::testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " device unavailable";
            session_.emplace(GetParam());
            ASSERT_TRUE(session_->switched());
            sh_ = lfs::training::training_ops(GetParam()).sh;
        }

        void TearDown() override { session_.reset(); }

        // An uninitialized device tensor (its unused bytes are never read), or nullopt with the reason. A store
        // past 2^32 16-byte elements is not zero-filled here: the generic fill wraps there on its own.
        std::optional<Tensor> device_tensor(const size_t elements, const DataType dtype) {
            try {
                Tensor tensor = Tensor::empty({elements}, Device::GPU, dtype);
                (void)tensor.slice(0, elements - 1, elements).cpu();
                return tensor;
            } catch (const std::exception& e) {
                unavailable_ = std::format("this device can't allocate {} elements: {}", elements, e.what());
                return std::nullopt;
            }
        }

        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
        const ops::ShOps* sh_ = nullptr;
        std::string unavailable_;
    };

#define LFS_LARGE_MODEL_NEEDS(bytes)                    \
    do {                                                \
        if (const auto reason = lm::skip_reason(bytes)) \
            GTEST_SKIP() << *reason;                    \
    } while (false)

    // ShOps::gather_swizzled moves whole float4 slots of 4-byte cells (legacy 8-bit SH moments, one byte a cell)
    // from far rows into a small store.
    TEST_P(LargeModelShSlots, GatherBytesPastTheSlotWrap) {
        const size_t source_bytes = kSlotRows * lm::kFloatWidth;
        LFS_LARGE_MODEL_NEEDS(source_bytes);
        if (sh_ == nullptr)
            GTEST_SKIP() << "Sh slot is empty";
        auto source = device_tensor(source_bytes, DataType::UInt8);
        if (!source)
            GTEST_SKIP() << unavailable_;
        write_rows(*source, DataType::UInt8);
        const std::vector<size_t> rows = far_and_head_rows(4208);
        Tensor destination = Tensor::zeros({lfs::core::sh_swizzled_float_count(rows.size(), lm::kRest)}, Device::GPU,
                                           DataType::UInt8);
        sh_->gather_swizzled(*source, lm::index_tensor<int32_t>(rows, Device::GPU), destination,
                             {.source_rows = kSlotRows, .count = rows.size(), .destination_offset = 0,
                              .source_rest = lm::kRest, .destination_rest = lm::kRest});
        const auto bytes = lm::host_copy<uint8_t>(destination);
        size_t wrong = 0;
        std::string first;
        for (size_t i = 0; i < rows.size(); ++i)
            for (uint32_t c = 0; c < lm::kFloatWidth; ++c) {
                const uint8_t got = bytes[cell_index(i, c, 0)], want = byte_of(rows[i], c);
                if (got != want && wrong++ == 0)
                    first = std::format("destination row {} (source row {}), cell {}: {} instead of {}", i, rows[i], c,
                                        got, want);
            }
        EXPECT_EQ(wrong, 0u) << wrong << " of " << rows.size() * lm::kFloatWidth << " bytes wrong; first: " << first;
    }

    // ShOps::decode_range of IEEE f16 float4 storage (the viewer's and the exportable block's SH) from the far rows.
    TEST_P(LargeModelShSlots, DecodeRangeHalfPastTheSlotWrap) {
        const size_t store_elements = lfs::core::sh_swizzled_float_count(kSlotRows, lm::kRest);
        LFS_LARGE_MODEL_NEEDS(store_elements * 2);
        if (sh_ == nullptr)
            GTEST_SKIP() << "Sh slot is empty";
        auto store = device_tensor(store_elements, DataType::Float16);
        if (!store)
            GTEST_SKIP() << unavailable_;
        write_rows(*store, DataType::Float16);
        const size_t rows = kSlotRows - kFarBegin;
        Tensor decoded = Tensor::zeros({rows * lm::kCells}, Device::GPU);
        sh_->decode_range(*store, Tensor{}, decoded,
                          {.canonical_float_offset = kFarBegin * lm::kCells,
                           .float_count = rows * lm::kCells,
                           .primitives = kSlotRows,
                           .destination_rest = lm::kRest,
                           .layout_rest = lm::kRest,
                           .storage = ops::ShStorage::IeeeFloat16});
        const auto values = lm::host_copy<float>(decoded);
        size_t wrong = 0;
        std::string first;
        for (size_t r = 0; r < rows; ++r)
            for (uint32_t c = 0; c < lm::kCells; ++c) {
                const float got = values[r * lm::kCells + c], want = lm::value(kFarBegin + r, c);
                if (got != want && wrong++ == 0)
                    first = std::format("row {}, cell {}: {} instead of {}", kFarBegin + r, c, got, want);
            }
        EXPECT_EQ(wrong, 0u) << wrong << " of " << rows * lm::kCells << " values wrong; first: " << first;
    }

    // ShOps::zero_rows (MRNF resets the SH gradients of new rows with it) on far rows of an fp32 store: those rows
    // become zero and the head rows keep their values.
    TEST_P(LargeModelShSlots, ZeroRowsPastTheSlotWrap) {
        const size_t store_elements = lfs::core::sh_swizzled_float_count(kSlotRows, lm::kRest);
        LFS_LARGE_MODEL_NEEDS(store_elements * sizeof(float));
        if (sh_ == nullptr)
            GTEST_SKIP() << "Sh slot is empty";
        auto store = device_tensor(store_elements, DataType::Float32);
        if (!store)
            GTEST_SKIP() << unavailable_;
        write_rows(*store, DataType::Float32);
        std::vector<size_t> rows;
        for (size_t r = kFarBegin; r < kSlotRows; ++r)
            rows.push_back(r);
        sh_->zero_rows(*store, lm::index_tensor<int32_t>(rows, Device::GPU), lm::kRest);
        const auto far = lm::host_copy<float>(store->slice(0, kFarBegin * lm::kFloatWidth, kSlotRows * lm::kFloatWidth));
        const auto head = lm::host_copy<float>(store->slice(0, 0, kHeadRows * lm::kFloatWidth));
        const auto expected_head = lm::host_copy<float>(float4_rows<float>(0, kHeadRows, DataType::Float32, value_of));
        size_t far_wrong = 0, head_wrong = 0;
        for (const float v : far)
            far_wrong += v != 0.0f ? 1 : 0;
        for (size_t i = 0; i < head.size(); ++i)
            head_wrong += std::memcmp(&head[i], &expected_head[i], sizeof(float)) != 0 ? 1 : 0;
        EXPECT_EQ(far_wrong, 0u) << far_wrong << " of " << far.size() << " floats of the zeroed rows are not zero";
        EXPECT_EQ(head_wrong, 0u) << head_wrong << " of " << head.size() << " floats of rows 0-255 changed";
    }

    INSTANTIATE_TEST_SUITE_P(Backends, LargeModelShSlots,
                             testing::Values(GpuBackend::CUDA, GpuBackend::Metal, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

#undef LFS_LARGE_MODEL_NEEDS

} // namespace
