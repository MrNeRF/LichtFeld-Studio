/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Generic device fills and copies past 2^32 elements. A fill or copy moves elements of 16 bytes when its bytes and
// offsets are 16-byte aligned, of 4 bytes when they are 4-byte aligned and single bytes otherwise; with one thread
// an element and a 32-bit thread index, a call past 2^32 elements wraps: 4 GiB of unaligned bytes, 16 GiB of 4-byte
// elements, 64 GiB aligned. Every whole-tensor clone, contiguous copy, capacity growth, zeros, full or fill_ goes
// through them: an fp32 canonical SH tensor [N, 15, 3] is 4-byte aligned only when N is not a multiple of 4 and
// passes 2^32 elements at 95,443,718 rows; the fp32 SH gradient the optimizer zeroes passes 2^32 16-byte elements
// at 357.9M rows.
//
// Each test fills or copies past the wrap and checks the bytes at the start, around 2^32 elements and at the end.
// Sources are written so that they do not need a fill past the wrap themselves.
//
// Tiers by device memory: see large_model_test_support.hpp. Each test's memory is in its skip message.

#include "large_model_test_support.hpp"

#include "core/tensor_backend.hpp"

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <string>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;

    constexpr size_t kTwoTo32 = size_t{1} << 32;
    constexpr size_t kProbe = 4096; // elements checked at each place

    class LargeModelTensorCopy : public ::testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " device unavailable";
            session_.emplace(GetParam());
            ASSERT_TRUE(session_->switched());
        }

        void TearDown() override { session_.reset(); }

        // An uninitialized device tensor, or nullopt with the reason in unavailable_.
        std::optional<Tensor> device_tensor(const lfs::core::TensorShape& shape, const DataType dtype) {
            try {
                Tensor tensor = Tensor::empty(shape, Device::GPU, dtype);
                const size_t n = tensor.numel();
                (void)tensor.flatten().slice(0, n - 1, n).cpu();
                return tensor;
            } catch (const std::exception& e) {
                unavailable_ = std::format("this device can't allocate {} elements: {}", shape.elements(), e.what());
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

#define LFS_LARGE_MODEL_OR_SKIP(optional) \
    do {                                  \
        if (!(optional))                  \
            GTEST_SKIP() << unavailable_; \
    } while (false)

    // Elements at the start, from 2^32 - kProbe / 2 and at the end of [0, count) must equal expected(index).
    template <class T, class Expected>
    void expect_elements(const Tensor& flat, const size_t count, Expected&& expected, const std::string& what) {
        for (const size_t first : {size_t{0}, kTwoTo32 - kProbe / 2, count - kProbe}) {
            const auto got = lm::host_copy<T>(flat.slice(0, first, std::min(first + kProbe, count)));
            size_t wrong = 0, first_wrong = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                const T want = expected(first + i);
                if (std::memcmp(&got[i], &want, sizeof(T)) != 0 && wrong++ == 0)
                    first_wrong = first + i;
            }
            EXPECT_EQ(wrong, 0u) << what << ": " << wrong << " of the " << got.size() << " elements from element "
                                 << first << " are wrong; first at element " << first_wrong;
        }
    }

    // Writes `value` into the elements at the three probe places of a flat fp32 tensor of `count` elements.
    void write_probes(Tensor& flat, const size_t count, float (*const value)(size_t)) {
        for (const size_t first : {size_t{0}, kTwoTo32 - kProbe / 2, count - kProbe}) {
            const size_t n = std::min(first + kProbe, count) - first;
            Tensor host = Tensor::empty({n}, Device::CPU, DataType::Float32);
            for (size_t i = 0; i < n; ++i)
                host.ptr<float>()[i] = value(first + i);
            flat.slice(0, first, first + n).copy_(host.gpu());
        }
    }

    float element_value(const size_t index) { return static_cast<float>(1 + index % 65'521); }

    // Tensor::full over 2^32 + 1 bytes moves single bytes (the size is not a multiple of 4).
    TEST_P(LargeModelTensorCopy, FillBytesPastTheGrid) {
        constexpr size_t kCount = kTwoTo32 + 1;
        LFS_LARGE_MODEL_NEEDS(kCount);
        Tensor bytes;
        try {
            bytes = Tensor::full({kCount}, 7.0f, Device::GPU, DataType::UInt8);
        } catch (const std::exception& e) {
            GTEST_SKIP() << "this device can't allocate " << kCount << " bytes: " << e.what();
        }
        expect_elements<uint8_t>(bytes, kCount, [](size_t) { return uint8_t{7}; }, "Tensor::full of bytes");
    }

    // A copy of 2^32 + 1 bytes from an odd offset moves single bytes. The source is 2^32 + 16 bytes, so its own fill
    // moves 16-byte elements below the wrap.
    TEST_P(LargeModelTensorCopy, CopyUnalignedBytesPastTheGrid) {
        constexpr size_t kCount = kTwoTo32 + 1;
        LFS_LARGE_MODEL_NEEDS(2 * (kTwoTo32 + 16));
        Tensor source;
        try {
            source = Tensor::full({kTwoTo32 + 16}, 9.0f, Device::GPU, DataType::UInt8);
        } catch (const std::exception& e) {
            GTEST_SKIP() << "this device can't allocate " << kTwoTo32 + 16 << " bytes: " << e.what();
        }
        expect_elements<uint8_t>(source, kTwoTo32 + 16, [](size_t) { return uint8_t{9}; }, "the source");
        auto destination = device_tensor({kTwoTo32 + 16}, DataType::UInt8);
        LFS_LARGE_MODEL_OR_SKIP(destination);
        destination->slice(0, 0, kCount).copy_from(source.slice(0, 1, kCount + 1));
        expect_elements<uint8_t>(*destination, kCount, [](size_t) { return uint8_t{9}; }, "unaligned byte copy");
    }

    // A copy of 2^32 + 1 floats from a 4-byte offset moves 4-byte elements (16 GiB of them).
    TEST_P(LargeModelTensorCopy, CopyFloatsPastTheGrid) {
        constexpr size_t kCount = kTwoTo32 + 1;
        LFS_LARGE_MODEL_NEEDS(2 * (kTwoTo32 + 4) * sizeof(float));
        auto source = device_tensor({kTwoTo32 + 4}, DataType::Float32);
        LFS_LARGE_MODEL_OR_SKIP(source);
        auto destination = device_tensor({kTwoTo32 + 4}, DataType::Float32);
        LFS_LARGE_MODEL_OR_SKIP(destination);
        Tensor shifted = source->slice(0, 1, kCount + 1);
        write_probes(shifted, kCount, element_value);
        destination->slice(0, 0, kCount).copy_from(shifted);
        expect_elements<float>(*destination, kCount, element_value, "4-byte float copy");
    }

    // A clone of an fp32 canonical SH tensor [95,443,718, 15, 3]: its bytes are 4-byte aligned only (the row count
    // is not a multiple of 4), and it holds 2^32 + 14 floats.
    TEST_P(LargeModelTensorCopy, CloneCanonicalShWithRowsNotAMultipleOfFour) {
        constexpr size_t kShRows = 95'443'718;
        constexpr size_t kCount = kShRows * lm::kCells;
        static_assert(kShRows % 4 != 0 && kCount > kTwoTo32);
        LFS_LARGE_MODEL_NEEDS(2 * kCount * sizeof(float));
        auto source = device_tensor({kShRows, size_t{lm::kRest}, size_t{3}}, DataType::Float32);
        LFS_LARGE_MODEL_OR_SKIP(source);
        Tensor flat = source->flatten();
        write_probes(flat, kCount, element_value);
        Tensor copy;
        try {
            copy = source->clone();
        } catch (const std::exception& e) {
            GTEST_SKIP() << "this device can't allocate the clone: " << e.what();
        }
        expect_elements<float>(copy.flatten(), kCount, element_value, "clone of a canonical SH tensor");
    }

    // zero_ of 68.7 GB, the fp32 SH gradient of 357,913,984 rows at SH degree 3: 16-byte elements past 2^32.
    TEST_P(LargeModelTensorCopy, ZeroFillPast64GiB) {
        const size_t count = lfs::core::sh_swizzled_float_count(lm::kSlotWrapRow + 2 * lm::kReorder, lm::kRest);
        ASSERT_GT(count / 4, kTwoTo32);
        LFS_LARGE_MODEL_NEEDS(count * sizeof(float));
        auto gradient = device_tensor({count}, DataType::Float32);
        LFS_LARGE_MODEL_OR_SKIP(gradient);
        write_probes(*gradient, count, element_value);
        // The probe around 2^32 elements sits below the wrap of 16-byte elements; the one at the end is past it.
        gradient->zero_();
        expect_elements<float>(*gradient, count, [](size_t) { return 0.0f; }, "zero_ of an fp32 SH gradient");
    }

#undef LFS_LARGE_MODEL_NEEDS
#undef LFS_LARGE_MODEL_OR_SKIP

    INSTANTIATE_TEST_SUITE_P(Backends, LargeModelTensorCopy,
                             testing::Values(GpuBackend::CUDA, GpuBackend::Metal, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

} // namespace
