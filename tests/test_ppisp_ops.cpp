/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/kernels/ppisp.cuh"
#include "lfs/kernels/ppisp_controller.cuh"
#include "lfs/training/ops/registry.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {
    using lfs::core::Device;
    using lfs::core::Tensor;
    namespace ops = lfs::gpu_ops;
    namespace kernels = lfs::training::kernels;
    using PPISPOpsTest = lfs::test::CudaBackendTest;

    Tensor pattern(const lfs::core::TensorShape& shape, int seed = 1) {
        std::vector<float> v(shape.elements());
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = float(int((i * 17 + seed * 13) % 31) - 15) / 64.f;
        return Tensor::from_vector(v, shape, Device::GPU);
    }
    void same(const Tensor& a, const Tensor& b) {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto x = a.cpu().contiguous(), y = b.cpu().contiguous();
        ASSERT_EQ(x.bytes(), y.bytes());
        EXPECT_EQ(std::memcmp(x.data_ptr(), y.data_ptr(), x.bytes()), 0);
    }
    const ops::PPISPOps& ppisp() {
        return *lfs::training::training_ops(lfs::core::GpuBackend::CUDA).ppisp;
    }
    const ops::ControllerOps& controller() {
        return *lfs::training::training_ops(lfs::core::GpuBackend::CUDA).controller;
    }
    struct Params {
        Tensor e = pattern({3}), v = pattern({30}, 2), c = pattern({24}, 3), r = pattern({24}, 4);
        ops::PPISPInputs inputs() const { return {e, v, c, r}; }
        ops::PPISPOutputs outputs() { return {e, v, c, r}; }
    };

    TEST_F(PPISPOpsTest, IdentityForwardAndRegionMatchLaunchers) {
        lfs::core::TensorCudaStream current;
        lfs::core::CUDAStreamGuard guard(current.get());
        Params actual, expected;
        ppisp().initialize(actual.outputs());
        kernels::launch_ppisp_init_identity(expected.e.ptr<float>(), expected.v.ptr<float>(),
                                            expected.c.ptr<float>(), expected.r.ptr<float>(), 2, 3, nullptr);
        same(actual.e, expected.e);
        same(actual.v, expected.v);
        same(actual.c, expected.c);
        same(actual.r, expected.r);
        Params p;
        const auto rgb = pattern({3, 2, 7}, 7).add(0.5f);
        auto a = Tensor::empty_like(rgb), b = Tensor::empty_like(rgb);
        for (int offset : {0, 3}) {
            ppisp().forward(p.inputs(), rgb, a, {offset, 8, 2, 3, 1, 2});
            kernels::launch_ppisp_forward_chw_region(p.e.ptr<float>(), p.v.ptr<float>(), p.c.ptr<float>(),
                                                     p.r.ptr<float>(), rgb.ptr<float>(), b.ptr<float>(), 2, 7, offset, 8, 2, 3, 1, 2, nullptr);
            same(a, b);
        }
    }

    TEST_F(PPISPOpsTest, BackwardMatchesLauncher) {
        Params p, a, b;
        // A single pixel makes the parameter atomics deterministic.
        auto rgb = pattern({3, 1, 1}, 7).add(0.5f), grad = pattern({3, 1, 1}, 9);
        auto ga = Tensor::empty_like(rgb), gb = Tensor::empty_like(rgb);
        ppisp().backward(p.inputs(), rgb, grad, a.outputs(), ga, 2, 3, 1, 2);
        kernels::launch_ppisp_backward_chw(p.e.ptr<float>(), p.v.ptr<float>(), p.c.ptr<float>(),
                                           p.r.ptr<float>(), rgb.ptr<float>(), grad.ptr<float>(), b.e.ptr<float>(),
                                           b.v.ptr<float>(), b.c.ptr<float>(), b.r.ptr<float>(), gb.ptr<float>(), 1, 1, 2, 3, 1, 2, nullptr);
        same(ga, gb);
        same(a.e, b.e);
        same(a.v, b.v);
        same(a.c, b.c);
        same(a.r, b.r);
    }

    struct Group {
        Tensor p, m, v, g;
        explicit Group(size_t n) : p(pattern({n})), m(pattern({n}, 2)), v(pattern({n}, 3).abs().add(0.1f)), g(pattern({n}, 4)) {}
        ops::PPISPAdamGroup bindings() { return {p, m, v, g}; }
        kernels::PPISPAdamGroup native() {
            return {p.ptr<float>(), m.ptr<float>(), v.ptr<float>(), g.ptr<float>(), int(p.numel())};
        }
    };
    TEST_F(PPISPOpsTest, SingleAndBatchedAdamMatchLaunchers) {
        const ops::PPISPAdamUpdateParams h{0.003f, 0.9f, 0.999f, 10.f, 31.622776f, 1e-8f};
        Group a(257), b(257);
        ppisp().adam(a.bindings(), h);
        kernels::launch_ppisp_adam_update(b.p.ptr<float>(), b.m.ptr<float>(), b.v.ptr<float>(),
                                          b.g.ptr<float>(), 257, h.lr, h.beta1, h.beta2, h.bc1_rcp, h.bc2_sqrt_rcp, h.eps, nullptr);
        same(a.p, b.p);
        same(a.m, b.m);
        same(a.v, b.v);
        for (bool crf : {false, true}) {
            std::array<Group, 4> x{Group(3), Group(30), Group(24), Group(24)};
            std::array<Group, 4> y{Group(3), Group(30), Group(24), Group(24)};
            Tensor absent;
            const ops::PPISPAdamGroup last = crf ? x[3].bindings() : ops::PPISPAdamGroup{absent, absent, absent, absent};
            ppisp().adam_batch({x[0].bindings(), x[1].bindings(), x[2].bindings(), last}, h);
            kernels::launch_ppisp_adam_update_batched(y[0].native(), y[1].native(), y[2].native(),
                                                      crf ? y[3].native() : kernels::PPISPAdamGroup{}, h.lr, h.beta1, h.beta2, h.bc1_rcp, h.bc2_sqrt_rcp, h.eps, nullptr);
            for (size_t i = 0; i < x.size(); ++i) {
                same(x[i].p, y[i].p);
                same(x[i].m, y[i].m);
                same(x[i].v, y[i].v);
            }
        }
    }

    TEST_F(PPISPOpsTest, RegularizationAndProjectionMatchLaunchers) {
        Params a, b;
        ppisp().project_mean(a.e, a.c);
        kernels::launch_ppisp_project_mean(b.e.ptr<float>(), b.c.ptr<float>(), 3, nullptr);
        same(a.e, b.e);
        same(a.c, b.c);
        auto p = pattern({15});
        for (int mode : {0, 1, 2}) {
            auto ga = pattern({15}, 3), gb = ga.clone();
            auto la = Tensor::zeros({1}, Device::GPU), lb = la.clone();
            Tensor absent;
            ppisp().vignetting_regularization(p, mode == 0 ? absent : ga, mode == 1 ? absent : la, .01f, .02f, .03f);
            kernels::launch_ppisp_vignetting_reg(p.ptr<float>(), mode == 0 ? nullptr : gb.ptr<float>(),
                                                 mode == 1 ? nullptr : lb.ptr<float>(), 1, .01f, .02f, .03f, nullptr);
            same(ga, gb);
            same(la, lb);
        }
    }

    TEST_F(PPISPOpsTest, ControllerInputAndBackwardMatchDirectSequence) {
        lfs::core::TensorCudaStream current;
        lfs::core::CUDAStreamGuard guard(current.get());
        auto a = Tensor::zeros({1, 1601}, Device::GPU), b = a.clone();
        auto features = pattern({1, 1600});
        const auto stream = lfs::core::getCurrentCUDAStream();
        float prior = 1.f;
        controller().prepare_input({}, a, prior);
        ASSERT_EQ(cudaMemcpyAsync(b.ptr<float>() + 1600, &prior, sizeof(float), cudaMemcpyHostToDevice, stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        same(a, b);
        for (float value : {1.f, .7f, 1.f}) {
            controller().prepare_input(features, a, value);
            cudaMemcpyAsync(b.ptr<float>(), features.ptr<float>(), 1600 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            if (value != 1.f)
                cudaMemcpyAsync(b.ptr<float>() + 1600, &value, sizeof(float), cudaMemcpyHostToDevice, stream);
            same(a, b);
        }
        for (bool input_grad : {false, true}) {
            constexpr int m = 9, n = 128;
            auto grad = pattern({1, m}), activation = pattern({1, n}, 2), weight = pattern({m, n}, 3);
            auto wa = pattern({m, n}, 4), wb = wa.clone(), ba = pattern({m}, 5), bb = ba.clone();
            auto ia = Tensor::empty({1, n}, Device::GPU), ib = Tensor::empty_like(ia);
            Tensor absent;
            controller().backward_layer(grad, activation, weight, wa, ba, input_grad ? ia : absent);
            kernels::launch_outer_product_accumulate(grad.ptr<float>(), activation.ptr<float>(), wb.ptr<float>(), m, n, 1.f, stream);
            kernels::launch_bias_grad_accumulate(grad.ptr<float>(), bb.ptr<float>(), m, stream);
            if (input_grad) {
                ib = grad.mm(weight);
                kernels::launch_relu_backward(ib.ptr<float>(), activation.ptr<float>(), ib.ptr<float>(), n, stream);
                same(ia, ib);
            }
            same(wa, wb);
            same(ba, bb);
        }
    }

    TEST(TrainingOpsPPISP, RequiresEnabledFamiliesAndLeavesOtherBackendsUnavailable) {
        lfs::core::param::TrainingParameters p;
        p.optimization.use_ppisp = true;
        p.optimization.ppisp_use_controller = true;
        auto required = lfs::training::required_training_families(p, {});
        EXPECT_TRUE(required.test(size_t(lfs::training::Family::PPISP)));
        EXPECT_TRUE(required.test(size_t(lfs::training::Family::Controller)));
        for (auto backend : {lfs::core::GpuBackend::Vulkan, lfs::core::GpuBackend::Metal}) {
            const auto& table = lfs::training::training_ops(backend);
            EXPECT_EQ(table.ppisp, nullptr);
            EXPECT_EQ(table.controller, nullptr);
            auto missing = lfs::training::missing_training_families(table, required);
            EXPECT_NE(std::find(missing.begin(), missing.end(), "PPISP"), missing.end());
            EXPECT_NE(std::find(missing.begin(), missing.end(), "Controller"), missing.end());
        }
    }
} // namespace
