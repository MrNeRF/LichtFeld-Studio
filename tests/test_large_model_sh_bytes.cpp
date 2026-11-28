/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The SH byte fill and the Metal row copy past 4 GiB. MRNF's compaction fills legacy 8-bit SH moments through the
// fill (89.5M rows at SH degree 3), Morton's grouped moment copy and the arena concatenation go through the row copy.
// One launch over 2^32 + 1 MiB bytes must write every byte; the probes sit at the start, around 1 MiB (where a launch
// cut to its thread count modulo 2^32 stops), at 2 GiB, around 4 GiB and at the end.
//
// Tiers by device memory: see large_model_test_support.hpp. Each test's memory is in its skip message.

#include "large_model_test_support.hpp"

#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/ops/sh.hpp"
#ifdef LFS_TENSOR_METAL
#include "ops/metal_kernels.hpp"
#include "ops/sh_metal.hpp"
#endif

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <optional>
#include <string>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;
    namespace ops = lfs::gpu_ops;

    class LargeModelShBytesBase : public ::testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " device unavailable";
            session_.emplace(GetParam());
            ASSERT_TRUE(session_->switched());
            sh_ = lfs::training::training_ops(GetParam()).sh;
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
        const ops::ShOps* sh_ = nullptr;
        std::string unavailable_;
    };

#define LFS_LARGE_MODEL_NEEDS(bytes)                    \
    do {                                                \
        if (const auto reason = lm::skip_reason(bytes)) \
            GTEST_SKIP() << *reason;                    \
    } while (false)

    // The SH byte fill: 2^32 + 1 MiB bytes of 128 (the zero code of 8-bit moments). Vulkan refuses such a fill loudly
    // (its size is checked to 32 bits), so it runs on CUDA and Metal.
    class LargeModelShBytes : public LargeModelShBytesBase {};

    constexpr size_t kBytes = (size_t{1} << 32) + (size_t{1} << 20);

    // Every byte of [0, 64 KiB), of the 64 KiB around 1 MiB (where a launch cut to its thread count modulo 2^32 would
    // stop), of the 64 KiB from 2 GiB, of the 64 KiB around 4 GiB and of the last 64 KiB must be `value`.
    void expect_filled(const Tensor& bytes, const uint8_t value, const std::string& what) {
        constexpr size_t kProbe = size_t{1} << 16;
        for (const size_t first : {size_t{0}, (size_t{1} << 20) - kProbe / 2, size_t{1} << 31,
                                   (size_t{1} << 32) - kProbe / 2, kBytes - kProbe}) {
            const auto probe = lm::host_copy<uint8_t>(bytes.slice(0, first, first + kProbe));
            const size_t wrong = static_cast<size_t>(std::count_if(probe.begin(), probe.end(),
                                                                   [value](const uint8_t b) { return b != value; }));
            EXPECT_EQ(wrong, 0u) << what << ": " << wrong << " of the " << kProbe << " bytes from byte " << first
                                 << " are not " << int(value);
        }
    }

    TEST_P(LargeModelShBytes, FillBytesPast4GiB) {
        LFS_LARGE_MODEL_NEEDS(kBytes);
        if (sh_ == nullptr)
            GTEST_SKIP() << "Sh slot is empty";
        auto bytes = device_tensor(kBytes, DataType::UInt8);
        if (!bytes)
            GTEST_SKIP() << unavailable_;
        sh_->fill_bytes(*bytes, kBytes, 128);
        expect_filled(*bytes, 128, "ShOps::fill_bytes");
    }

#ifdef LFS_TENSOR_METAL
    // The Metal row copy behind Morton's grouped moment permute and the arena concatenation: one row of 2^32 + 1 MiB
    // bytes.
    TEST_P(LargeModelShBytes, CopyRowsPast4GiB) {
        if (GetParam() != GpuBackend::Metal)
            GTEST_SKIP() << "the row copy is Metal's";
        LFS_LARGE_MODEL_NEEDS(2 * kBytes);
        std::optional<Tensor> source;
        try {
            // 2^32 + 1 MiB bytes fill as 16-byte elements, below the generic fill's own wrap.
            source = Tensor::full({kBytes}, 7.0f, Device::GPU, DataType::UInt8);
        } catch (const std::exception& e) {
            GTEST_SKIP() << "this device can't allocate " << kBytes << " bytes: " << e.what();
        }
        auto destination = device_tensor(kBytes, DataType::UInt8);
        if (!destination)
            GTEST_SKIP() << unavailable_;
        expect_filled(*source, 7, "the source");
        namespace mk = lfs::training::metal;
        mk::copy_rows(mk::address(*source), mk::address(*destination), kBytes, 1, kBytes, kBytes,
                      {&*source, &*destination});
        expect_filled(*destination, 7, "metal::copy_rows");
    }
#endif

    INSTANTIATE_TEST_SUITE_P(Backends, LargeModelShBytes, testing::Values(GpuBackend::CUDA, GpuBackend::Metal),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

#undef LFS_LARGE_MODEL_NEEDS

} // namespace
