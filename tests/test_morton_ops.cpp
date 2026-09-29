/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/sh_layout.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "kernels/morton_reorder_kernels.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/ops/morton_cuda.hpp"
#include "lfs/training/ops/registry.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    namespace ops = lfs::gpu_ops;
    namespace kernels = lfs::training::kernels;
    namespace joint = lfs::training::joint_adam;

    const auto& cuda_ops() {
        return *lfs::training::training_ops(lfs::core::GpuBackend::CUDA).morton;
    }

    std::vector<uint8_t> bytes(const Tensor& tensor) {
        const auto cpu = tensor.cpu().contiguous();
        const auto* data = static_cast<const uint8_t*>(cpu.data_ptr());
        return {data, data + cpu.bytes()};
    }

    void same(const Tensor& actual, const Tensor& expected) {
        ASSERT_EQ(cudaStreamSynchronize(lfs::core::getCurrentCUDAStream()), cudaSuccess);
        EXPECT_EQ(bytes(actual), bytes(expected));
    }

    Tensor shuffled_indices(size_t n) {
        auto cpu = Tensor::empty({n}, Device::CPU, DataType::Int64);
        std::iota(cpu.ptr<int64_t>(), cpu.ptr<int64_t>() + n, int64_t{0});
        std::mt19937 rng(731);
        std::shuffle(cpu.ptr<int64_t>(), cpu.ptr<int64_t>() + n, rng);
        return cpu.gpu();
    }

    Tensor packed_pattern(size_t count) {
        auto cpu = Tensor::empty({count}, Device::CPU, DataType::UInt8);
        std::mt19937 rng(991);
        for (size_t i = 0; i < count; ++i)
            cpu.ptr<uint8_t>()[i] = static_cast<uint8_t>(rng());
        return cpu.gpu();
    }

    Tensor block_bounds(size_t n) {
        std::vector<float> values(joint::n_bounds_for_prims(n) * 4);
        for (size_t b = 0; b < values.size() / 4; ++b) {
            values[b * 4] = -0.3f - b * 0.1f;
            values[b * 4 + 1] = 0.5f + b * 0.1f;
            values[b * 4 + 2] = 0.01f;
            values[b * 4 + 3] = 0.1f + b * 0.01f;
        }
        return Tensor::from_vector(values, {values.size()}, Device::GPU);
    }
} // namespace

TEST(MortonOpsCapability, OnlyCudaProvidesScheduledFamily) {
    using namespace lfs::training;
    EXPECT_EQ(training_ops(lfs::core::GpuBackend::CUDA).morton, &cuda_morton_ops());
    FamilySet required;
    required.set(static_cast<size_t>(Family::Morton));
    EXPECT_EQ(missing_training_families(TrainingOps{}, required), std::vector<std::string_view>{"Morton"});
    EXPECT_NE(training_ops(lfs::core::GpuBackend::Vulkan).morton, nullptr);
    EXPECT_FALSE(unavailable_training_family(lfs::core::GpuBackend::Vulkan, Family::Morton));
    for (auto backend : {lfs::core::GpuBackend::Metal}) {
        EXPECT_EQ(training_ops(backend).morton, nullptr);
        const auto reason = unavailable_training_family(backend, Family::Morton);
        ASSERT_TRUE(reason.has_value());
        EXPECT_NE(reason->find("Missing families: Morton"), std::string::npos);
    }
    lfs::core::param::TrainingParameters params;
    params.optimization.morton_reorder_interval = 100;
    EXPECT_TRUE(required_training_families(params, {}).test(static_cast<size_t>(Family::Morton)));
    params.optimization.morton_reorder_interval = 0;
    EXPECT_FALSE(required_training_families(params, {}).test(static_cast<size_t>(Family::Morton)));
}

class MortonOpsBytes : public lfs::test::CudaBackendTest {
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

TEST_F(MortonOpsBytes, PermutationPreservesBoundsAndTies) {
    for (size_t n : {size_t{1}, size_t{513}}) {
        std::mt19937 rng(12345);
        std::vector<float> xyz(n * 3);
        for (size_t i = 0; i < n; ++i) {
            xyz[i * 3] = static_cast<float>(rng() % 17) - 8.f;
            xyz[i * 3 + 1] = static_cast<float>(rng() % 17) - 8.f;
            xyz[i * 3 + 2] = 2.f;
        }
        const auto means = Tensor::from_vector(xyz, {n, 3}, Device::GPU);
        const auto expected = kernels::launch_morton_permutation(means, lfs::core::getCurrentCUDAStream());
        const auto actual = cuda_ops().permutation(means);
        ASSERT_EQ(actual.numel(), n);
        same(actual, expected);
    }
}

TEST_F(MortonOpsBytes, JointLayoutsAndGroupedScratchMatchLaunchers) {
    constexpr size_t n = 513;
    const auto perm = shuffled_indices(n), bounds = block_bounds(n);
    for (auto layout : {ops::JointLayout::Rows, ops::JointLayout::SwizzledSH}) {
        for (int bits : {8, 16}) {
            SCOPED_TRACE(bits);
            SCOPED_TRACE(static_cast<int>(layout));
            constexpr int width = 5;
            const size_t cells = layout == ops::JointLayout::Rows ? n * width : lfs::core::sh_swizzled_padded_n(n) * width * 4;
            const auto packed = packed_pattern(cells * joint::bytes_per_cell(bits));
            auto a = Tensor::zeros(packed.shape(), Device::GPU, DataType::UInt8), b = a.clone();
            auto ab = Tensor::zeros(bounds.shape(), Device::GPU), bb = ab.clone();
            const ops::JointCodecParams p{layout, static_cast<int>(n), width, bits};
            const auto launch = layout == ops::JointLayout::Rows ? kernels::launch_joint_permute_contiguous : kernels::launch_joint_permute_shN;
            launch(packed.ptr<uint8_t>(), bounds.ptr<float>(), b.ptr<uint8_t>(), bb.ptr<float>(),
                   perm.ptr<int64_t>(), n, width, bits, lfs::core::getCurrentCUDAStream());
            cuda_ops().permute_joint(packed, bounds, perm, a, ab, p);
            same(a, b);
            same(ab, bb);
            EXPECT_NE(bytes(a), bytes(Tensor::zeros(packed.shape(), Device::GPU, DataType::UInt8)));
            if (layout == ops::JointLayout::SwizzledSH) {
                const size_t slot_bytes = cells / width * joint::bytes_per_cell(bits);
                for (size_t group_width : {size_t{1}, size_t{3}, size_t{5}}) {
                    auto ga = packed.clone(), gb = packed.clone();
                    auto gab = Tensor::zeros(bounds.shape(), Device::GPU), gbb = gab.clone();
                    auto sa = Tensor::empty({slot_bytes * group_width}, Device::GPU, DataType::UInt8), sb = sa.clone();
                    kernels::launch_joint_permute_shN_grouped(gb.ptr<uint8_t>(), bounds.ptr<float>(), gbb.ptr<float>(),
                                                              perm.ptr<int64_t>(), n, width, bits, sb.ptr<uint8_t>(), sb.bytes(), lfs::core::getCurrentCUDAStream());
                    cuda_ops().permute_joint_grouped(ga, bounds, perm, gab, sa, p);
                    same(ga, gb);
                    same(gab, gbb);
                    same(ga, b);
                    same(gab, bb);
                }
            }
        }
    }
}
