/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// SH rest storage past 2^32 elements. At SH degree 3 a row of the swizzled store holds 45 Q16
// cells or 12 float4 slots (48 floats), so the store's element index passes 2^32 in the block of
// row 95,443,712 (Q16) and of row 89,478,464 (floats). Each backend decodes that block and the
// next one from a store whose only nonzero values are in those two blocks, and must read every
// value back; an index computed in 32 bits reads the zeros at the start of the store instead.
//
// The stores take about 8.6 GB of device memory (Q16, f16) and 17.2 GB (fp32). The tests run only
// with LFS_TEST_LARGE_SH=1 and skip a store that can't be allocated or cleared.

#include "core/sh_value_quant.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    namespace ops = lfs::gpu_ops;
    namespace quant = lfs::core::sh_value_quant;

    constexpr uint32_t kRest = 15;         // SH degree 3
    constexpr uint32_t kCells = kRest * 3; // canonical values, and Q16 cells, per row
    constexpr uint32_t kSlots = 12;        // float4 slots per row
    constexpr size_t kReorder = 32;        // rows per swizzle block
    constexpr size_t kTailRows = 2 * kReorder;

    // First row of the block that holds element 2^32: 2^32 / (45 * 32) and 2^32 / (48 * 32) blocks,
    // rounded down, times 32 rows.
    constexpr size_t kQ16WrapRow = 95'443'712;
    constexpr size_t kFloatWrapRow = 89'478'464;
    static_assert((kQ16WrapRow / kReorder) * kCells * kReorder < (size_t{1} << 32));
    static_assert((kQ16WrapRow / kReorder + 1) * kCells * kReorder > (size_t{1} << 32));
    static_assert((kFloatWrapRow / kReorder) * kSlots * kReorder * 4 < (size_t{1} << 32));
    static_assert((kFloatWrapRow / kReorder + 1) * kSlots * kReorder * 4 > (size_t{1} << 32));

    // Value of tail row r, canonical cell c: an integer in [1, 2047], exact in half and in Q16 with
    // bounds [0, 65535].
    float value(const size_t r, const uint32_t c) { return static_cast<float>(1 + (r * kCells + c) % 2047); }

    template <class T>
    Tensor upload(const std::vector<T>& values, const DataType dtype) {
        Tensor cpu = Tensor::empty({values.size()}, Device::CPU, dtype);
        EXPECT_EQ(cpu.bytes(), values.size() * sizeof(T));
        std::memcpy(cpu.data_ptr(), values.data(), cpu.bytes());
        return cpu.gpu();
    }

    Tensor upload_f32(const std::vector<float>& values) { return upload(values, DataType::Float32); }

    // Q16 codes travel as Float16-typed bits, as the training store holds them.
    Tensor upload_codes(const std::vector<uint16_t>& codes) { return upload(codes, DataType::Float16); }

    std::vector<float> download_f32(const Tensor& tensor) {
        const Tensor cpu = tensor.cpu().contiguous();
        std::vector<float> values(cpu.numel());
        std::memcpy(values.data(), cpu.data_ptr(), values.size() * sizeof(float));
        return values;
    }

    void expect_tail(const std::vector<float>& decoded, const float tolerance) {
        ASSERT_EQ(decoded.size(), kTailRows * kCells);
        size_t wrong = 0;
        std::string first;
        for (size_t r = 0; r < kTailRows; ++r)
            for (uint32_t c = 0; c < kCells; ++c) {
                const float want = value(r, c);
                const float got = decoded[r * kCells + c];
                if (std::fabs(got - want) > tolerance && wrong++ == 0)
                    first = "tail row " + std::to_string(r) + ", cell " + std::to_string(c) + ": " +
                            std::to_string(got) + " instead of " + std::to_string(want);
            }
        EXPECT_EQ(wrong, 0u) << wrong << " of " << kTailRows * kCells << " values wrong; first: " << first;
    }

    class ShRestPast32BitWrap : public ::testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            const char* flag = std::getenv("LFS_TEST_LARGE_SH");
            if (flag == nullptr || std::string(flag) != "1")
                GTEST_SKIP() << "set LFS_TEST_LARGE_SH=1 to run (8.6 GB of device memory, 17.2 GB for fp32)";
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " device unavailable";
            session_.emplace(GetParam());
            ASSERT_TRUE(session_->switched());
            sh_ = lfs::training::training_ops(GetParam()).sh;
            if (sh_ == nullptr)
                GTEST_SKIP() << "Sh slot is empty";
        }

        void TearDown() override { session_.reset(); }

        // A zeroed store of `elements`, or nullopt when the device can't hold or clear it. A device reports a
        // failed clear only when its commands complete (MoltenVK can't clear the 17.2 GB fp32 store), so this
        // waits for the clear by reading back the store's last element.
        static std::optional<Tensor> store(const size_t elements, const DataType dtype) {
            try {
                Tensor values = Tensor::zeros({elements}, Device::GPU, dtype);
                (void)values.slice(0, elements - 1, elements).cpu();
                return values;
            } catch (const std::exception&) {
                return std::nullopt;
            }
        }

        void float_case(const ops::ShStorage storage) {
            const size_t rows = kFloatWrapRow + kTailRows;
            const DataType dtype = storage == ops::ShStorage::IeeeFloat16 ? DataType::Float16 : DataType::Float32;
            auto values = store(rows / kReorder * kSlots * kReorder * 4, dtype);
            if (!values)
                GTEST_SKIP() << "can't allocate or clear " << rows << " rows of " << (dtype == DataType::Float16 ? "f16" : "fp32");
            // The two tail blocks in storage order: [block][slot][lane][4].
            std::vector<float> tail(kTailRows * kSlots * 4, 0.0f);
            for (size_t r = 0; r < kTailRows; ++r)
                for (uint32_t c = 0; c < kCells; ++c)
                    tail[((r / kReorder) * kSlots * kReorder + (c / 4) * kReorder + r % kReorder) * 4 + c % 4] = value(r, c);
            const size_t first = kFloatWrapRow / kReorder * kSlots * kReorder * 4;
            Tensor tail_gpu = upload_f32(tail);
            if (dtype == DataType::Float16)
                tail_gpu = tail_gpu.to(DataType::Float16);
            values->slice(0, first, first + tail.size()).copy_(tail_gpu);

            Tensor decoded = Tensor::zeros({kTailRows * kCells}, Device::GPU);
            sh_->decode_range(*values, Tensor{}, decoded,
                              {.canonical_float_offset = kFloatWrapRow * kCells,
                               .float_count = kTailRows * kCells,
                               .primitives = rows,
                               .destination_rest = kRest,
                               .layout_rest = kRest,
                               .storage = storage});
            expect_tail(download_f32(decoded), 0.0f);
        }

        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
        const ops::ShOps* sh_ = nullptr;
    };

    TEST_P(ShRestPast32BitWrap, Q16DecodesTheBlocksPastTheWrap) {
        const size_t rows = kQ16WrapRow + kTailRows;
        auto codes = store(quant::sh_value_u16_count(rows, kRest), DataType::Float16);
        if (!codes)
            GTEST_SKIP() << "can't allocate or clear " << rows << " rows of Q16 codes";
        Tensor bounds = Tensor::zeros({quant::n_bounds_for_prims(rows) * 2}, Device::GPU);
        // The two tail blocks in storage order: [block][cell][lane].
        std::vector<uint16_t> tail(kTailRows * kCells);
        for (size_t r = 0; r < kTailRows; ++r)
            for (uint32_t c = 0; c < kCells; ++c)
                tail[(r / kReorder) * kCells * kReorder + c * kReorder + r % kReorder] = static_cast<uint16_t>(value(r, c));
        const size_t first = kQ16WrapRow / kReorder * kCells * kReorder;
        codes->slice(0, first, first + tail.size()).copy_(upload_codes(tail));
        // Bounds [0, 65535] decode every code to its own value; the 64 rows share one bounds block.
        const size_t block = kQ16WrapRow / 256;
        ASSERT_EQ((rows - 1) / 256, block);
        bounds.slice(0, block * 2, block * 2 + 2).copy_(upload_f32({0.0f, 65535.0f}));

        Tensor decoded = Tensor::zeros({kTailRows * kCells}, Device::GPU);
        sh_->decode_range(*codes, bounds, decoded,
                          {.canonical_float_offset = kQ16WrapRow * kCells,
                           .float_count = kTailRows * kCells,
                           .primitives = rows,
                           .destination_rest = kRest,
                           .layout_rest = kRest,
                           .storage = ops::ShStorage::Q16});
        expect_tail(download_f32(decoded), 0.02f);
    }

    TEST_P(ShRestPast32BitWrap, HalfDecodesTheBlocksPastTheWrap) { float_case(ops::ShStorage::IeeeFloat16); }

    TEST_P(ShRestPast32BitWrap, FloatDecodesTheBlocksPastTheWrap) { float_case(ops::ShStorage::Float32); }

    INSTANTIATE_TEST_SUITE_P(Backends, ShRestPast32BitWrap,
                             testing::Values(GpuBackend::CUDA, GpuBackend::Metal, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

} // namespace
