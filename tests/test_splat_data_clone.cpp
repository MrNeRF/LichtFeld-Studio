/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/splat_exportable_storage.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "training/training_setup.hpp"

#include <cstring>
#include <gtest/gtest.h>
#include <optional>
#include <random>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    constexpr size_t kN = 600; // ≥ 2 q16 blocks
    constexpr int kShDegree = 3;

    SplatData make_random_sh3(const size_t n, const uint32_t seed = 42) {
        auto means = Tensor::zeros({n, size_t{3}}, Device::CUDA, DataType::Float32);
        auto sh0 = Tensor::zeros({n, size_t{1}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto shN_can = Tensor::zeros({n, size_t{15}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto scaling = Tensor::zeros({n, size_t{3}}, Device::CUDA, DataType::Float32);
        auto rotation = Tensor::zeros({n, size_t{4}}, Device::CUDA, DataType::Float32);
        auto opacity = Tensor::zeros({n, size_t{1}}, Device::CUDA, DataType::Float32);

        {
            std::mt19937 rng(seed);
            std::normal_distribution<float> nd(0.0f, 0.15f);
            auto cpu = shN_can.cpu();
            auto* p = cpu.ptr<float>();
            for (size_t i = 0; i < n * 15 * 3; ++i)
                p[i] = nd(rng);
            shN_can = cpu.to(Device::CUDA);

            auto rcpu = rotation.cpu();
            auto* r = rcpu.ptr<float>();
            for (size_t i = 0; i < n; ++i)
                r[i * 4] = 1.0f;
            rotation = rcpu.to(Device::CUDA);
        }

        return SplatData(kShDegree, means, sh0, shN_can, scaling, rotation, opacity, 1.0f);
    }

    [[nodiscard]] bool tensors_equal(const Tensor& a, const Tensor& b) {
        if (!a.is_valid() || !b.is_valid()) {
            return a.is_valid() == b.is_valid();
        }
        if (a.dtype() != b.dtype() || a.shape() != b.shape()) {
            return false;
        }
        auto ac = a.cpu().contiguous();
        auto bc = b.cpu().contiguous();
        return std::memcmp(ac.data_ptr(), bc.data_ptr(), ac.bytes()) == 0;
    }

} // namespace

TEST(SplatDataCloneTest, Q16CloneCarriesBounds) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto model = make_random_sh3(kN);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(model));
    ASSERT_TRUE(model.shN_value_quantized());

    const auto canonical_before = model.shN_canonical();
    const void* const src_means_ptr = model.means_raw().data_ptr();
    const float src_mean0 = model.means_raw().cpu().ptr<float>()[0];

    auto copy = model.clone();

    EXPECT_TRUE(copy.shN_value_quantized());
    ASSERT_TRUE(copy.shN_value_bounds().is_valid());
    EXPECT_EQ(copy.shN_value_bounds().numel(), model.shN_value_bounds().numel());

    EXPECT_NO_THROW(copy.set_active_sh_degree(1));
    EXPECT_NO_THROW(copy.set_active_sh_degree(3));
    EXPECT_NO_THROW(copy.set_max_sh_degree(3));

    EXPECT_TRUE(tensors_equal(copy.shN_canonical(), canonical_before));

    EXPECT_NE(copy.means_raw().data_ptr(), src_means_ptr);
    copy.means_raw().fill_(123.0f);
    EXPECT_FLOAT_EQ(model.means_raw().cpu().ptr<float>()[0], src_mean0);

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(SplatDataCloneTest, Fp32CloneUnchangedBehavior) {
    sh_value::set_sh_value_quant_enabled_for_testing(false);
    auto model = make_random_sh3(kN);
    ASSERT_FALSE(model.shN_value_quantized());

    const auto canonical_before = model.shN_canonical();
    auto copy = model.clone();

    EXPECT_FALSE(model.shN_value_quantized());
    EXPECT_FALSE(copy.shN_value_quantized());
    EXPECT_NO_THROW(copy.set_active_sh_degree(1));
    EXPECT_NO_THROW(copy.set_active_sh_degree(3));
    EXPECT_TRUE(tensors_equal(copy.shN_canonical(), canonical_before));

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(SplatDataCloneTest, CloneCarriesDeletedMask) {
    auto model = make_random_sh3(64);
    std::vector<bool> deleted(64, false);
    deleted[1] = true;
    deleted[17] = true;
    deleted[63] = true;
    model.deleted() = Tensor::from_vector(deleted, {deleted.size()}, Device::CPU).to(Device::CUDA);
    model.refresh_deleted_count();
    ASSERT_TRUE(model.has_deleted_mask());
    ASSERT_EQ(model.deleted_count(), 3u);

    auto copy = model.clone();

    ASSERT_TRUE(copy.has_deleted_mask());
    EXPECT_EQ(copy.deleted_count(), model.deleted_count());
    EXPECT_EQ(copy.deleted().cpu().to_vector_bool(), model.deleted().cpu().to_vector_bool());

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(SplatDataCloneTest, Q16CloneMigratesToExportableAllocator) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto model = make_random_sh3(kN);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(model));
    ASSERT_TRUE(model.shN_value_quantized());

    auto copy = model.clone();
    ASSERT_TRUE(copy.shN_value_quantized());
    const auto canonical_before = copy.shN_canonical();

    constexpr size_t kCap = 1024;
    auto storage_result = SplatExportableStorage::create(kCap, kShDegree, /*device=*/0, kCap * 4);
    if (!storage_result) {
        GTEST_SKIP() << "exportable create failed: " << storage_result.error();
    }
    auto storage = std::make_shared<SplatExportableStorage>(std::move(*storage_result));

    param::TrainingParameters params;
    params.optimization.max_cap = static_cast<int>(kCap);
    const auto result = migrateTrainingModelToAllocator(params, copy, storage->make_allocator());
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_TRUE(copy.shN_value_quantized());
    EXPECT_TRUE(tensors_equal(copy.shN_canonical(), canonical_before));

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}
