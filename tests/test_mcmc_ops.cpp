/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "kernels/mcmc_kernels.hpp"
#include "lfs/training/ops/mcmc_cuda.hpp"
#include "lfs/training/ops/registry.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <optional>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    namespace kernels = lfs::training::mcmc;
    namespace ops = lfs::gpu_ops;

    Tensor pattern(const lfs::core::TensorShape& shape, float offset = 0.f) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = offset + static_cast<float>((i * 17 + 3) % 101) / 101.f;
        }
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    Tensor indices(const std::vector<int64_t>& values) {
        auto cpu = Tensor::empty({values.size()}, Device::CPU, DataType::Int64);
        std::copy(values.begin(), values.end(), cpu.ptr<int64_t>());
        return cpu.gpu();
    }

    std::vector<uint8_t> bytes(const Tensor& tensor) {
        const auto cpu = tensor.cpu().contiguous();
        const auto* data = static_cast<const uint8_t*>(cpu.data_ptr());
        return {data, data + cpu.bytes()};
    }

    void same(const Tensor& actual, const Tensor& expected) {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(bytes(actual), bytes(expected));
    }

    const auto& cuda_ops() {
        return *lfs::training::training_ops(lfs::core::GpuBackend::CUDA).mcmc;
    }
} // namespace

TEST(McmcOpsCapability, OnlyCudaProvidesTheFamily) {
    using namespace lfs::training;
    EXPECT_EQ(training_ops(lfs::core::GpuBackend::CUDA).mcmc, &cuda_mcmc_ops());
    FamilySet required;
    required.set(static_cast<size_t>(Family::Mcmc));
    EXPECT_EQ(missing_training_families(TrainingOps{}, required), std::vector<std::string_view>{"Mcmc"});
    for (auto backend : {lfs::core::GpuBackend::Vulkan, lfs::core::GpuBackend::Metal}) {
        EXPECT_EQ(training_ops(backend).mcmc, nullptr);
        const auto reason = unavailable_training_family(backend, Family::Mcmc);
        ASSERT_TRUE(reason.has_value());
        EXPECT_NE(reason->find("Missing families: Mcmc"), std::string::npos);
        for (const auto* strategy : {"mcmc", "igs+"}) {
            lfs::core::param::TrainingParameters params;
            params.optimization.strategy = strategy;
            const auto training_reason = unavailable_training_reason(params, backend, {});
            ASSERT_TRUE(training_reason.has_value());
            EXPECT_NE(training_reason->find("Mcmc"), std::string::npos);
        }
    }
}

class McmcOpsBytes : public lfs::test::CudaBackendTest {
protected:
    void SetUp() override {
        LFS_CUDA_BACKEND_OR_RETURN();
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
        guard_.emplace(stream_);
    }
    void TearDown() override {
        if (stream_) {
            EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
            guard_.reset();
            EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
        }
    }

private:
    cudaStream_t stream_ = nullptr;
    std::optional<lfs::core::CUDAStreamGuard> guard_;
};

TEST_F(McmcOpsBytes, InitializeAndRelocate) {
    constexpr size_t n = 257;
    auto opacity = pattern({n}, 0.01f).clamp(0.01f, 0.99f);
    auto scales = pattern({n, 3}, 0.1f);
    auto ratios = Tensor::empty({n}, Device::CPU, DataType::Int32);
    for (size_t i = 0; i < n; ++i)
        ratios.ptr<int32_t>()[i] = 1 + i % 50;
    ratios = ratios.gpu();
    auto a = Tensor::zeros({n}, Device::GPU), b = a.clone();
    auto sa = Tensor::zeros({n, 3}, Device::GPU), sb = sa.clone();
    kernels::init_relocation_coefficients(51);
    kernels::launch_relocation_kernel(opacity.ptr<float>(), scales.ptr<float>(), ratios.ptr<int32_t>(),
                                      0.005f, b.ptr<float>(), sb.ptr<float>(), n);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    kernels::init_relocation_coefficients(1);
    cuda_ops().initialize(51);
    cuda_ops().relocate(opacity, scales, ratios, a, sa, 0.005f);
    same(a, b);
    same(sa, sb);
    EXPECT_NE(bytes(a), bytes(Tensor::zeros({n}, Device::GPU)));
}

TEST_F(McmcOpsBytes, NoiseWithAndWithoutFrozenRows) {
    constexpr size_t n = 257;
    auto opacity = pattern({n}, -4.f), scales = pattern({n, 3}, -2.f), quats = pattern({n, 4});
    for (bool masked : {false, true}) {
        auto a = pattern({n, 3}), b = a.clone();
        const auto before = bytes(a);
        Tensor frozen;
        if (masked) {
            std::vector<bool> mask(n);
            for (size_t i = 0; i < n; ++i)
                mask[i] = i % 3 == 0;
            frozen = Tensor::from_vector(mask, {n}, Device::GPU);
        }
        kernels::launch_inject_noise_kernel(opacity.ptr<float>(), scales.ptr<float>(), quats.ptr<float>(),
                                            b.ptr<float>(), masked ? frozen.ptr<bool>() : nullptr, masked ? n : 0, 0.2f, n, 12345);
        cuda_ops().noise(opacity, scales, quats, frozen, a, 12345, 0.2f);
        same(a, b);
        EXPECT_NE(bytes(a), before);
    }
}

TEST_F(McmcOpsBytes, CopyAndUpdateRowsPreserveOtherRows) {
    constexpr size_t n = 17;
    const auto src = indices({0, 3, 5}), dst = indices({9, 11, 16});
    for (bool column : {false, true}) {
        auto means = pattern({n, 3}), sh0 = pattern({n, 1, 3}), scales = pattern({n, 3}, -2.f);
        auto quats = pattern({n, 4}), opacity = pattern(column ? lfs::core::TensorShape{n, 1} : lfs::core::TensorShape{n});
        auto mb = means.clone(), sb = sh0.clone(), scb = scales.clone(), qb = quats.clone(), ob = opacity.clone();
        const auto before = bytes(means);
        kernels::launch_copy_gaussian_params(src.ptr<int64_t>(), dst.ptr<int64_t>(), mb.ptr<float>(), sb.ptr<float>(),
                                             nullptr, scb.ptr<float>(), qb.ptr<float>(), ob.ptr<float>(), 3, 0, column ? 1 : 0, n);
        cuda_ops().copy_rows(src, dst, {means, sh0, scales, quats, opacity});
        same(means, mb);
        same(sh0, sb);
        same(scales, scb);
        same(quats, qb);
        same(opacity, ob);
        EXPECT_NE(bytes(means), before);
        auto new_scales = pattern({3, 3}, -5.f), new_opacity = pattern({3}, -3.f);
        kernels::launch_update_scaling_opacity(dst.ptr<int64_t>(), new_scales.ptr<float>(), new_opacity.ptr<float>(),
                                               scb.ptr<float>(), ob.ptr<float>(), 3, column ? 1 : 0, n);
        cuda_ops().update_rows(dst, new_scales, new_opacity, scales, opacity);
        same(scales, scb);
        same(opacity, ob);
    }
}

TEST_F(McmcOpsBytes, SampleBothDomains) {
    constexpr size_t n = 257, count = 513;
    auto weights = pattern({n}, 0.01f), opacity = pattern({n}), scales = pattern({n, 3}, -2.f);
    const auto alive = indices({0, 4, 7, 31, 42, 128, 256});
    for (auto domain : {ops::SampleDomain::All, ops::SampleDomain::AliveIndices}) {
        auto a = Tensor::empty({count}, Device::GPU, DataType::Int64), b = a.clone();
        auto oa = Tensor::zeros({count}, Device::GPU), ob = oa.clone();
        auto sa = Tensor::zeros({count, 3}, Device::GPU), sb = sa.clone();
        if (domain == ops::SampleDomain::All) {
            kernels::launch_multinomial_sample_all(weights.ptr<float>(), opacity.ptr<float>(), scales.ptr<float>(), n,
                                                   count, 98765, b.ptr<int64_t>(), ob.ptr<float>(), sb.ptr<float>());
        } else {
            kernels::launch_multinomial_sample_and_gather(weights.ptr<float>(), opacity.ptr<float>(), scales.ptr<float>(),
                                                          alive.ptr<int64_t>(), alive.numel(), count, 98765, b.ptr<int64_t>(), ob.ptr<float>(), sb.ptr<float>(), n);
        }
        cuda_ops().sample(weights, opacity, scales, domain == ops::SampleDomain::All ? Tensor{} : alive,
                          a, oa, sa, domain, 98765);
        same(a, b);
        same(oa, ob);
        same(sa, sb);
        EXPECT_NE(bytes(oa), bytes(Tensor::zeros({count}, Device::GPU)));
    }
}

TEST_F(McmcOpsBytes, FoldErrorZerosBothRows) {
    constexpr size_t n = 257;
    auto a = pattern({n}), b = a.clone(), da = pattern({2, n}, 0.5f), db = da.clone();
    kernels::launch_max_error_and_zero_densification(b.ptr<float>(), db.ptr<float>(), n);
    cuda_ops().fold_error(a, da);
    same(a, b);
    same(da, db);
    same(da, Tensor::zeros({2, n}, Device::GPU));
}
