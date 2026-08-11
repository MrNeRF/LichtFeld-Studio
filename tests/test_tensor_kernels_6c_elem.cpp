/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * SM-capped vectorized elementwise, device-side reductions, multi-block
 * count_nonzero, float4 comparisons, and Float16 operations.
 */

#include "core/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;

namespace {

    bool has_cuda_device() {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    Tensor f32_cuda(const std::vector<float>& host, std::initializer_list<size_t> shape) {
        return Tensor::from_vector(host, TensorShape(shape), Device::CUDA);
    }

    Tensor f32_cuda(const std::vector<float>& host, size_t n) {
        return Tensor::from_vector(host, TensorShape({n}), Device::CUDA);
    }

} // namespace

// ---------------------------------------------------------------------------
// vectorized elementwise correctness across cutoff boundary
// ---------------------------------------------------------------------------

TEST(TensorElementwiseKernels, UnaryBinaryAcrossVectorizedCutoff) {
    // Below old 1024 cutoff but above new 256 → must hit vectorized path.
    for (size_t n : {64u, 257u, 512u, 1025u, 4096u, 100003u}) {
        std::vector<float> a(n), b(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i % 97) * 0.01f - 0.3f;
            b[i] = static_cast<float>((i * 3) % 53) * 0.02f + 0.1f;
        }
        auto A = f32_cuda(a, n);
        auto B = f32_cuda(b, n);

        auto sum = (A + B).cpu().to_vector();
        auto prod = (A * B).cpu().to_vector();
        auto neg = (-A).cpu().to_vector();
        auto absv = A.abs().cpu().to_vector();

        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(sum[i], a[i] + b[i], 1e-5f) << "n=" << n << " i=" << i;
            EXPECT_NEAR(prod[i], a[i] * b[i], 1e-5f) << "n=" << n << " i=" << i;
            EXPECT_NEAR(neg[i], -a[i], 1e-5f) << "n=" << n << " i=" << i;
            EXPECT_NEAR(absv[i], std::fabs(a[i]), 1e-5f) << "n=" << n << " i=" << i;
        }
    }
}

// ---------------------------------------------------------------------------
// mean / prod fully device-side; count_nonzero; compares
// ---------------------------------------------------------------------------

TEST(TensorElementwiseKernels, MeanAndProdMatchReference) {
    std::vector<float> vals = {1.f, 2.f, 3.f, 4.f, 0.5f, -1.f, 8.f, 0.25f};
    auto t = f32_cuda(vals, vals.size());

    float sum = 0.f, prod = 1.f;
    for (float v : vals) {
        sum += v;
        prod *= v;
    }
    const float mean = sum / static_cast<float>(vals.size());

    EXPECT_NEAR(t.mean().item(), mean, 1e-5f);
    EXPECT_NEAR(t.sum().item(), sum, 1e-5f);
    EXPECT_NEAR(t.prod().item(), prod, 1e-4f); // prod accumulates more error
}

TEST(TensorElementwiseKernels, MeanSegmentedScaleDeviceSide) {
    // [2, 4] mean over last dim → two means
    std::vector<float> vals = {1, 2, 3, 4, 10, 20, 30, 40};
    auto t = f32_cuda(vals, {2, 4});
    auto m = t.mean({1}).cpu().to_vector();
    ASSERT_EQ(m.size(), 2u);
    EXPECT_NEAR(m[0], 2.5f, 1e-5f);
    EXPECT_NEAR(m[1], 25.f, 1e-5f);
}

TEST(TensorElementwiseKernels, CountNonzeroMatchesReference) {
    // Large enough to exercise multi-block path (>=100k)
    const size_t n = 200000;
    std::vector<float> vals(n, 0.f);
    size_t expected = 0;
    for (size_t i = 0; i < n; i += 7) {
        vals[i] = 1.5f;
        ++expected;
    }
    auto t = f32_cuda(vals, n);
    EXPECT_EQ(t.count_nonzero(), expected);

    // Bool path
    auto mask = (t > 0.0f);
    EXPECT_EQ(mask.count_nonzero(), expected);
}

TEST(TensorElementwiseKernels, CompareFloat4SameShape) {
    const size_t n = 5000; // > kVectorizedMinElems
    std::vector<float> a(n), b(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i % 10);
        b[i] = static_cast<float>((i + 3) % 10);
    }
    auto A = f32_cuda(a, n);
    auto B = f32_cuda(b, n);

    auto eq = (A == B).cpu().to_vector_bool();
    auto lt = (A < B).cpu().to_vector_bool();
    auto gt = (A > B).cpu().to_vector_bool();
    ASSERT_EQ(eq.size(), n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(eq[i], a[i] == b[i]) << i;
        EXPECT_EQ(lt[i], a[i] < b[i]) << i;
        EXPECT_EQ(gt[i], a[i] > b[i]) << i;
    }
}

TEST(TensorElementwiseKernels, Float16BinaryVectorized) {
    // Host already wires Float16 binary arithmetic (tensor_exports).
    // Unary/reduce host gates remain fail-loud; kernel launches exist.
    if (!has_cuda_device())
        GTEST_SKIP() << "CUDA required";

    const size_t n = 4096;
    std::vector<float> ha(n), hb(n);
    for (size_t i = 0; i < n; ++i) {
        ha[i] = (static_cast<float>(i % 50) - 25.f) * 0.1f;
        hb[i] = (static_cast<float>((i * 2) % 40) - 10.f) * 0.05f + 0.5f;
    }
    auto A = f32_cuda(ha, n).to(DataType::Float16);
    auto B = f32_cuda(hb, n).to(DataType::Float16);

    auto sum_f16 = (A + B).to(DataType::Float32).cpu().to_vector();
    auto mul_f16 = (A * B).to(DataType::Float32).cpu().to_vector();

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(sum_f16[i], ha[i] + hb[i], 2e-2f) << i;
        EXPECT_NEAR(mul_f16[i], ha[i] * hb[i], 2e-2f) << i;
    }
}

TEST(TensorElementwiseKernels, Float16HostReduceIsCorrect) {
    if (!has_cuda_device())
        GTEST_SKIP() << "CUDA required";

    auto t = f32_cuda({1.f, 2.f, 3.f, 4.f}, 4).to(DataType::Float16);
    auto s = t.sum();
    auto s_f32 = s.to(DataType::Float32).cpu();
    ASSERT_EQ(s_f32.numel(), 1u);
    EXPECT_NEAR(s_f32.ptr<float>()[0], 10.f, 1e-2f);
}
