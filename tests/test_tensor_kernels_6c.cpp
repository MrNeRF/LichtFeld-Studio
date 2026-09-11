/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor/internal/lazy_ir.hpp"
#include "core/tensor/internal/tensor_ops.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;

namespace {

    class Kernels6CGuard {
    public:
        Kernels6CGuard() {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
            internal::lazy_executor_set_size_heuristic_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
            internal::lazy_ir_set_active_for_testing(false);
            Tensor::reset_lazy_telemetry();
            tensor_ops::reset_tensor_kernel_launch_count();
        }

        ~Kernels6CGuard() {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_heuristic_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
            internal::lazy_ir_set_active_for_testing(std::nullopt);
            Tensor::reset_lazy_telemetry();
            tensor_ops::reset_tensor_kernel_launch_count();
        }
    };

    bool has_cuda_device() {
        int device_count = 0;
        const auto status = cudaGetDeviceCount(&device_count);
        return status == cudaSuccess && device_count > 0;
    }

    Tensor fill_linear(const std::vector<size_t>& shape, float scale, Device dev) {
        auto t = Tensor::empty(TensorShape(shape), Device::CPU, DataType::Float32);
        auto* p = t.ptr<float>();
        const size_t n = t.numel();
        for (size_t i = 0; i < n; ++i) {
            p[i] = scale * static_cast<float>(i % 97) + 0.25f;
        }
        return dev == Device::CUDA ? t.to(Device::CUDA) : t;
    }

    void expect_close(const Tensor& got, const Tensor& expected, float atol = 1e-5f,
                      const char* label = "") {
        auto g = got.cpu().contiguous();
        auto e = expected.cpu().contiguous();
        ASSERT_EQ(g.numel(), e.numel()) << label;
        const float* gp = g.ptr<float>();
        const float* ep = e.ptr<float>();
        for (size_t i = 0; i < g.numel(); ++i) {
            EXPECT_NEAR(gp[i], ep[i], atol) << label << " i=" << i;
            if (std::abs(gp[i] - ep[i]) > atol) {
                break;
            }
        }
    }

} // namespace

// ---------------------------------------------------------------------------
// Binary(+reduce) fusion
// ---------------------------------------------------------------------------

TEST(TensorKernelFusion, BinaryMulAddFusesToOneLaunch) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA required";
    }
    Kernels6CGuard guard;

    // Large enough to cross the size-heuristic defer threshold (4 KiB default).
    constexpr size_t N = 8192;
    auto a = fill_linear({N}, 0.01f, Device::CUDA);
    auto b = fill_linear({N}, 0.02f, Device::CUDA);
    auto c = fill_linear({N}, 0.03f, Device::CUDA);

    // Reference: force unfused via fusion off
    Tensor ref;
    {
        Kernels6CGuard g2;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        ref = a.mul(b).add(c);
        (void)ref.data_ptr(); // public force-materialize
        cudaDeviceSynchronize();
    }

    tensor_ops::reset_tensor_kernel_launch_count();
    auto fused = a.mul(b).add(c);
    EXPECT_TRUE(fused.is_deferred()) << "large mul+add should form a deferred fusion chain";
    (void)fused.data_ptr(); // public force-materialize
    cudaDeviceSynchronize();

    const auto launches = tensor_ops::tensor_kernel_launch_count();
    EXPECT_EQ(launches, 1u) << "mul+add fused chain must be a single kernel launch, got "
                            << launches;

    expect_close(fused, ref, 1e-5f, "mul+add");
}

TEST(TensorKernelFusion, BinaryMulSumFusesToOneOrTwoLaunches) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA required";
    }
    Kernels6CGuard guard;

    constexpr size_t N = 8192;
    auto a = fill_linear({N}, 0.01f, Device::CUDA);
    auto b = fill_linear({N}, 0.02f, Device::CUDA);

    Tensor ref;
    {
        Kernels6CGuard g2;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        ref = a.mul(b).sum();
        (void)ref.data_ptr();
        cudaDeviceSynchronize();
    }

    tensor_ops::reset_tensor_kernel_launch_count();
    auto prod = a.mul(b);
    EXPECT_TRUE(prod.is_deferred());
    auto s = prod.sum();
    (void)s.data_ptr();
    cudaDeviceSynchronize();

    const auto launches = tensor_ops::tensor_kernel_launch_count();
    EXPECT_GE(launches, 1u);
    EXPECT_LE(launches, 2u) << "mul→sum fused transform-reduce should be 1-2 launches, got "
                            << launches;

    expect_close(s, ref, 1e-3f, "mul+sum"); // reduce accum may use double path vs fused
}

TEST(TensorKernelFusion, SingleBinaryKeepsFastPathWhenSmall) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA required";
    }
    Kernels6CGuard guard;

    // Below 4 KiB threshold → eager fast path, not deferred fusion seed.
    auto a = fill_linear({64}, 0.01f, Device::CUDA);
    auto b = fill_linear({64}, 0.02f, Device::CUDA);
    auto c = a.mul(b);
    EXPECT_FALSE(c.is_deferred()) << "small single binary must stay on the eager path";
    expect_close(c, a.cpu().mul(b.cpu()).to(Device::CUDA), 1e-5f, "small mul");
}

TEST(TensorKernelFusion, BinaryFusionNumericalSuite) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA required";
    }
    Kernels6CGuard guard;

    constexpr size_t N = 4096;
    auto a = fill_linear({N}, 0.011f, Device::CUDA);
    auto b = fill_linear({N}, 0.017f, Device::CUDA);
    auto c = fill_linear({N}, 0.023f, Device::CUDA);
    auto d = fill_linear({N}, 0.029f, Device::CUDA);

    // (a*b + c) * d
    auto chain = a.mul(b).add(c).mul(d);
    auto got = chain.cpu().contiguous();

    // CPU reference
    auto ar = a.cpu();
    auto br = b.cpu();
    auto cr = c.cpu();
    auto dr = d.cpu();
    auto expected = ar.mul(br).add(cr).mul(dr);
    // Force materialize with fusion off for a clean ref
    {
        Kernels6CGuard g2;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        expected = ar.mul(br).add(cr).mul(dr);
    }
    expect_close(got, expected, 1e-4f, "mul+add+mul");
}

// ---------------------------------------------------------------------------
// where host clones + same-shape broadcast
// ---------------------------------------------------------------------------

TEST(TensorKernelFusion, WhereSameShapeZeroExtraAllocs) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA required";
    }
    Kernels6CGuard guard;

    constexpr size_t N = 4096;
    auto cond_f = fill_linear({N}, 1.0f, Device::CUDA);
    // Build bool condition: > 0.5
    auto cond = cond_f.gt(0.5f);
    auto x = fill_linear({N}, 0.1f, Device::CUDA);
    auto y = fill_linear({N}, 0.2f, Device::CUDA);

    // Warm pools so where itself should not drive new device allocs for clones.
    {
        auto warm = Tensor::where(cond, x, y);
        (void)warm;
        cudaDeviceSynchronize();
    }

    const auto before = alloc_counter::snapshot();
    auto out = Tensor::where(cond, x, y);
    cudaDeviceSynchronize();
    const auto delta = alloc_counter::delta_since(before);

    EXPECT_EQ(delta, 0u) << "where on same-shape operands must not clone (0 extra driver allocs), got "
                         << delta;

    // Numerical: where cond ? x : y
    auto out_cpu = out.cpu();
    auto cond_cpu = cond.cpu();
    auto x_cpu = x.cpu();
    auto y_cpu = y.cpu();
    for (size_t i = 0; i < N; ++i) {
        const float expect = cond_cpu.ptr<uint8_t>()[i] ? x_cpu.ptr<float>()[i]
                                                        : y_cpu.ptr<float>()[i];
        EXPECT_NEAR(out_cpu.ptr<float>()[i], expect, 1e-6f);
    }
}

TEST(TensorKernelFusion, WhereSameShapePeakMemoryNoCloneBuffers) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA required";
    }
    Kernels6CGuard guard;

    constexpr size_t N = 1 << 18; // 256k floats
    auto cond = fill_linear({N}, 1.0f, Device::CUDA).gt(0.5f);
    auto x = fill_linear({N}, 0.1f, Device::CUDA);
    auto y = fill_linear({N}, 0.2f, Device::CUDA);
    // Warm
    (void)Tensor::where(cond, x, y);
    cudaDeviceSynchronize();

    size_t free_b = 0, total_b = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_b, &total_b), cudaSuccess);
    const size_t free_before = free_b;

    auto out = Tensor::where(cond, x, y);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaMemGetInfo(&free_b, &total_b), cudaSuccess);
    const size_t free_after = free_b;

    // Output is one N-float buffer (~1 MiB). Clones of cond/x/y would be ~3 extra.
    // Allow one output + small pool slack; reject multi-MiB clone spikes.
    const size_t used = free_before > free_after ? free_before - free_after : 0;
    const size_t out_bytes = N * sizeof(float);
    EXPECT_LT(used, out_bytes * 3) << "where peak should be ~1 output buffer, used=" << used
                                   << " out=" << out_bytes;
    EXPECT_EQ(out.numel(), N);
}

// ---------------------------------------------------------------------------
// Channel3D kernel selection
// ---------------------------------------------------------------------------

TEST(TensorKernelFusion, Channel3DEquivalenceAcrossC) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA required";
    }
    Kernels6CGuard guard;

    const std::vector<size_t> Cs = {1, 3, 4, 16, 64};
    constexpr size_t H = 32;
    constexpr size_t W = 32;

    for (size_t C : Cs) {
        auto img = fill_linear({H, W, C}, 0.01f, Device::CUDA);
        auto ch = fill_linear({1, 1, C}, 0.05f, Device::CUDA);
        auto got = img.add(ch); // Channel3D broadcast pattern
        (void)got.data_ptr();

        // CPU reference (generic broadcast)
        auto img_c = img.cpu();
        auto ch_c = ch.cpu();
        auto exp = Tensor::empty(TensorShape({H, W, C}), Device::CPU, DataType::Float32);
        for (size_t i = 0; i < H * W; ++i) {
            for (size_t c = 0; c < C; ++c) {
                exp.ptr<float>()[i * C + c] =
                    img_c.ptr<float>()[i * C + c] + ch_c.ptr<float>()[c];
            }
        }
        expect_close(got, exp, 1e-5f, ("C=" + std::to_string(C)).c_str());
    }
}
