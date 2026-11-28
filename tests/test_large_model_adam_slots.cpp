/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The SH Adam step past the float4 slot wrap of SH degree 3 (row 357,913,920): the fp32 gradient's and the 8-bit
// moments' slots, and the Q16 values around them.
//
// Runs with LFS_TEST_HUGE_SH=1 and, on a shared machine, alone in an exclusive hold of the GPU lock: about 135 GB of
// device memory.

#include "large_model_test_support.hpp"

#include "core/tensor_backend.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/ops/adam.hpp"
#include "lfs/training/ops/registry.hpp"

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
#include <string>
#include <tuple>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;

    class LargeModelAdamSlots : public ::testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " device unavailable";
            session_.emplace(GetParam());
            ASSERT_TRUE(session_->switched());
        }

        void TearDown() override { session_.reset(); }

        // An uninitialized device tensor, or nullopt with the reason.
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
        std::string unavailable_;
    };

#define LFS_LARGE_MODEL_NEEDS(bytes)                    \
    do {                                                \
        if (const auto reason = lm::skip_reason(bytes)) \
            GTEST_SKIP() << *reason;                    \
    } while (false)

    // One SH Adam step over a Q16 model of 357,914,112 rows (1,398,102 Q16 blocks): the fp32 gradient's and the
    // moments' float4 slots pass 2^32 in the block of row 357,913,920. Gradients are nonzero only in the first 256
    // rows and in the two Q16 blocks from row 357,913,600; the reference is the same kernel on a model of those
    // blocks alone, byte for byte (as LargeModelShStorage.AdamStepQ16PastTheWrap at the Q16 wrap). The gradient store
    // (68.7 GB) is not zero-filled: rows outside the blocks are never compared.
    TEST_P(LargeModelAdamSlots, StepPastTheSlotWrap) {
        namespace joint = lfs::training::joint_adam;
        namespace ops = lfs::gpu_ops;
        constexpr size_t kAdamRows = 357'914'112;
        constexpr size_t kAdamFar = 357'913'600;
        static_assert(kAdamRows % lm::kQuantBlock == 0 && kAdamFar % lm::kQuantBlock == 0 &&
                      kAdamFar < lm::kSlotWrapRow && lm::kSlotWrapRow < kAdamRows);
        const auto moment_bytes = [](const size_t rows) { return lfs::core::sh_swizzled_float_count(rows, lm::kRest) * 2; };
        const size_t gradient_bytes = lfs::core::sh_swizzled_float_count(kAdamRows, lm::kRest) * sizeof(float);
        LFS_LARGE_MODEL_NEEDS(lm::store_bytes(lm::ShFormat::Q16, kAdamRows) + gradient_bytes + moment_bytes(kAdamRows));
        const ops::AdamOps* const adam = lfs::training::training_ops(GetParam()).adam;
        if (adam == nullptr)
            GTEST_SKIP() << "Adam slot is empty";

        // The compact model: rows [0, 256) and the far 512 rows, values and gradients a function of the row.
        constexpr size_t kCompact = 256 + (kAdamRows - kAdamFar);
        const auto big_row = [](const size_t compact) { return compact < 256 ? compact : kAdamFar + (compact - 256); };
        const auto fill_canonical = [&](float (*const fill)(size_t, uint32_t)) {
            return lm::canonical_rows(kCompact, [&](const size_t c) { return std::optional(big_row(c)); }, fill);
        };
        lm::Store compact_values = lm::cpu_store(lm::ShFormat::Q16, kCompact);
        lm::convert(fill_canonical(lm::value), compact_values, {.count = kCompact});
        lm::Store compact_gradient = lm::cpu_store(lm::ShFormat::Float32, kCompact);
        lm::convert(fill_canonical([](const size_t r, const uint32_t c) { return (lm::value(r, c) - 1024.0f) / 8192.0f; }),
                    compact_gradient, {.count = kCompact});

        const Tensor none;
        const ops::AdamMasks masks{none, none, none};
        const auto step = [&](lm::Store& values, const Tensor& gradient, const size_t rows, Tensor& packed,
                              Tensor& bounds) {
            packed = Tensor::zeros({moment_bytes(rows)}, Device::GPU, DataType::UInt8);
            bounds = Tensor::zeros({joint::n_bounds_for_prims(rows), 4}, Device::GPU);
            const ops::ShStepParams params{.primitives = static_cast<int>(rows),
                                           .layout_slots = static_cast<int>(lm::kSlots),
                                           .active_bases = 16,
                                           .value_bits = 16,
                                           .value_cells = static_cast<int>(lm::kCells),
                                           .step_size = 0.05f,
                                           .bc2_sqrt_rcp = static_cast<float>(1.0 / std::sqrt(0.001))};
            adam->step_sh(values.values, packed, bounds, values.bounds, gradient, masks, ops::AdamHyper{},
                          ops::AdamModifiers{}, params);
        };
        lm::Store small{.format = lm::ShFormat::Q16, .rows = kCompact, .values = compact_values.values.gpu(),
                        .bounds = compact_values.bounds.gpu()};
        Tensor small_packed, small_bounds;
        step(small, compact_gradient.values.gpu(), kCompact, small_packed, small_bounds);

        auto values = lm::zero_store(lm::ShFormat::Q16, kAdamRows, Device::GPU);
        if (!values)
            GTEST_SKIP() << "this device can't allocate or clear " << kAdamRows << " rows of Q16 SH";
        auto gradient = device_tensor(lfs::core::sh_swizzled_float_count(kAdamRows, lm::kRest), DataType::Float32);
        if (!gradient)
            GTEST_SKIP() << unavailable_;
        // The two ranges of rows, in compact order: [0, 256) and [256, kCompact).
        const auto place = [&](lm::Store& big, Tensor* big_gradient) {
            for (const auto& [first, count, compact_first] :
                 {std::tuple{size_t{0}, size_t{256}, size_t{0}}, std::tuple{kAdamFar, kAdamRows - kAdamFar, size_t{256}}}) {
                if (big_gradient)
                    lm::values_of_rows(*big_gradient, lm::ShFormat::Float32, first, count)
                        .copy_(lm::values_of_rows(compact_gradient.values, lm::ShFormat::Float32, compact_first, count).gpu());
                else {
                    lm::values_of_rows(big.values, lm::ShFormat::Q16, first, count)
                        .copy_(lm::values_of_rows(compact_values.values, lm::ShFormat::Q16, compact_first, count).gpu());
                    lm::bounds_of_rows(big.bounds, first, count)
                        .copy_(lm::bounds_of_rows(compact_values.bounds, compact_first, count).gpu());
                }
            }
        };
        place(*values, nullptr);
        place(*values, &*gradient);
        Tensor packed, bounds;
        step(*values, *gradient, kAdamRows, packed, bounds);

        for (const auto& [first, count, compact_first] :
             {std::tuple{size_t{0}, size_t{256}, size_t{0}}, std::tuple{kAdamFar, kAdamRows - kAdamFar, size_t{256}}}) {
            const std::string where = std::format("adam_step_sh, rows {}-{}", first, first + count - 1);
            lm::expect_rows_match(*values, first, small, compact_first, count, where, 0);
            const auto a = lm::host_copy<uint8_t>(packed.slice(0, first * lm::kFloatWidth * 2, (first + count) * lm::kFloatWidth * 2));
            const auto e = lm::host_copy<uint8_t>(
                small_packed.slice(0, compact_first * lm::kFloatWidth * 2, (compact_first + count) * lm::kFloatWidth * 2));
            ASSERT_EQ(a.size(), e.size()) << where << ", moments";
            EXPECT_TRUE(std::equal(a.begin(), a.end(), e.begin()))
                << where << ", moments: "
                << std::inner_product(a.begin(), a.end(), e.begin(), size_t{0}, std::plus<>(),
                                      [](const uint8_t x, const uint8_t y) { return size_t{x != y}; })
                << " of " << a.size() << " bytes differ";
            const auto ab = lm::host_copy<float>(bounds.slice(0, first / lm::kQuantBlock, (first + count) / lm::kQuantBlock));
            const auto eb = lm::host_copy<float>(
                small_bounds.slice(0, compact_first / lm::kQuantBlock, (compact_first + count) / lm::kQuantBlock));
            ASSERT_EQ(ab.size(), eb.size()) << where << ", moment bounds";
            EXPECT_EQ(std::memcmp(ab.data(), eb.data(), ab.size() * sizeof(float)), 0) << where << ", moment bounds";
        }
    }

#undef LFS_LARGE_MODEL_NEEDS

    INSTANTIATE_TEST_SUITE_P(Backends, LargeModelAdamSlots,
                             testing::Values(GpuBackend::CUDA, GpuBackend::Metal, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

} // namespace
