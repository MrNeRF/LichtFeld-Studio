/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/splat_exportable_storage.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "training/training_setup.hpp"

#include <algorithm>
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

    [[nodiscard]] size_t rest_coeffs_for_degree(const int sh_degree) {
        if (sh_degree <= 0)
            return 0;
        if (sh_degree == 1)
            return 3;
        if (sh_degree == 2)
            return 8;
        return 15;
    }

    SplatData make_random_model(const size_t n, const int sh_degree, const uint32_t seed = 42) {
        const size_t rest = rest_coeffs_for_degree(sh_degree);
        auto means = Tensor::zeros({n, size_t{3}}, Device::CUDA, DataType::Float32);
        auto sh0 = Tensor::zeros({n, size_t{1}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto shN_can = Tensor::zeros({n, rest, size_t{3}}, Device::CUDA, DataType::Float32);
        auto scaling = Tensor::zeros({n, size_t{3}}, Device::CUDA, DataType::Float32);
        auto rotation = Tensor::zeros({n, size_t{4}}, Device::CUDA, DataType::Float32);
        auto opacity = Tensor::zeros({n, size_t{1}}, Device::CUDA, DataType::Float32);

        {
            std::mt19937 rng(seed);
            std::normal_distribution<float> nd(0.0f, 0.15f);
            if (rest > 0) {
                auto cpu = shN_can.cpu();
                auto* p = cpu.ptr<float>();
                for (size_t i = 0; i < n * rest * 3; ++i)
                    p[i] = nd(rng);
                shN_can = cpu.to(Device::CUDA);
            }

            auto rcpu = rotation.cpu();
            auto* r = rcpu.ptr<float>();
            for (size_t i = 0; i < n; ++i)
                r[i * 4] = 1.0f;
            rotation = rcpu.to(Device::CUDA);
        }

        return SplatData(sh_degree, means, sh0, shN_can, scaling, rotation, opacity, 1.0f);
    }

    SplatData make_random_sh3(const size_t n, const uint32_t seed = 42) {
        return make_random_model(n, kShDegree, seed);
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

    struct ShValueQuantGuard {
        explicit ShValueQuantGuard(const bool enabled) {
            sh_value::set_sh_value_quant_enabled_for_testing(enabled);
        }
        ~ShValueQuantGuard() {
            sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
        }
    };
} // namespace

TEST(SplatDataCloneTest, Q16CloneCarriesBounds) {
    const ShValueQuantGuard quant_guard{true};
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
}

TEST(SplatDataCloneTest, Fp32CloneUnchangedBehavior) {
    const ShValueQuantGuard quant_guard{false};
    auto model = make_random_sh3(kN);
    ASSERT_FALSE(model.shN_value_quantized());

    const auto canonical_before = model.shN_canonical();
    auto copy = model.clone();

    EXPECT_FALSE(model.shN_value_quantized());
    EXPECT_FALSE(copy.shN_value_quantized());
    EXPECT_NO_THROW(copy.set_active_sh_degree(1));
    EXPECT_NO_THROW(copy.set_active_sh_degree(3));
    EXPECT_TRUE(tensors_equal(copy.shN_canonical(), canonical_before));
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
}

TEST(SplatDataCloneTest, Q16CloneMigratesToExportableAllocator) {
    const ShValueQuantGuard quant_guard{true};
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
}

TEST(SplatDataCloneTest, WritebackClearsQ16Pair) {
    const ShValueQuantGuard quant_guard{true};

    // Degree 2: q16 cell count equals ieee-f16 swizzle count. Degree 3 does not.
    for (const int sh_degree : {2, 3}) {
        auto model = make_random_model(kN, sh_degree);
        ASSERT_TRUE(sh_value::apply_shN_value_quant(model)) << "degree=" << sh_degree;
        ASSERT_TRUE(model.shN_value_quantized()) << "degree=" << sh_degree;

        const auto captured = model.shN_canonical();
        const size_t cap = model.means().is_valid()
                               ? std::max(model.means().capacity(), static_cast<size_t>(model.size()))
                               : static_cast<size_t>(model.size());
        model.shN_set_from_canonical(captured, cap);

        EXPECT_FALSE(model.shN_value_quantized()) << "degree=" << sh_degree;
        EXPECT_TRUE(!model.shN_value_bounds().is_valid() ||
                    model.shN_value_bounds().numel() == 0)
            << "degree=" << sh_degree;
        ASSERT_TRUE(model.shN().is_valid()) << "degree=" << sh_degree;
        EXPECT_EQ(model.shN().dtype(), DataType::Float32) << "degree=" << sh_degree;
        EXPECT_TRUE(tensors_equal(model.shN_canonical(), captured)) << "degree=" << sh_degree;
    }
}

TEST(SplatDataCloneTest, ReserveCapacityRebuildsCudaDirect) {
    const size_t n = 64;
    auto means = Tensor::zeros_direct({n, size_t{3}}, n, Device::CUDA, DataType::Float32);
    auto sh0 = Tensor::zeros_direct({n, size_t{1}, size_t{3}}, n, Device::CUDA, DataType::Float32);
    auto shN_can = Tensor::zeros_direct({n, size_t{15}, size_t{3}}, n, Device::CUDA, DataType::Float32);
    auto scaling = Tensor::zeros_direct({n, size_t{3}}, n, Device::CUDA, DataType::Float32);
    auto rotation = Tensor::zeros_direct({n, size_t{4}}, n, Device::CUDA, DataType::Float32);
    auto opacity = Tensor::zeros_direct({n, size_t{1}}, n, Device::CUDA, DataType::Float32);
    means.fill_(1.25f);
    shN_can.fill_(0.05f);

    auto model = SplatData(kShDegree, std::move(means), std::move(sh0), std::move(shN_can),
                           std::move(scaling), std::move(rotation), std::move(opacity), 1.0f);
    ASSERT_EQ(model.means().external_storage_kind(), "cuda.direct");
    ASSERT_EQ(model.shN().external_storage_kind(), "cuda.direct");

    const auto means_before = model.means().clone();
    const auto shN_before = model.shN_canonical();
    const size_t old_means_cap = model.means().capacity();
    const size_t old_shN_cap = model.shN().capacity();
    const size_t new_cap = old_means_cap + 256;
    model.reserve_capacity(new_cap);

    EXPECT_GE(model.means().capacity(), new_cap);
    EXPECT_GT(model.shN().capacity(), old_shN_cap);
    EXPECT_EQ(model.means().external_storage_kind(), "cuda.direct");
    EXPECT_TRUE(tensors_equal(model.means(), means_before));
    EXPECT_TRUE(tensors_equal(model.shN_canonical(), shN_before));
}

TEST(SplatDataCloneTest, ReserveCapacitySkipsRendererStorage) {
    const ShValueQuantGuard quant_guard{true};
    auto model = make_random_sh3(kN);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(model));
    ASSERT_TRUE(model.shN_value_quantized());

    constexpr size_t kCap = 1024;
    auto storage_result = SplatExportableStorage::create(kCap, kShDegree, /*device=*/0, kCap * 4);
    if (!storage_result) {
        GTEST_SKIP() << "exportable create failed: " << storage_result.error();
    }
    auto storage = std::make_shared<SplatExportableStorage>(std::move(*storage_result));

    param::TrainingParameters params;
    params.optimization.max_cap = static_cast<int>(kCap);
    const auto result = migrateTrainingModelToAllocator(params, model, storage->make_allocator());
    ASSERT_TRUE(result.has_value()) << result.error();

    const auto means_kind = model.means().external_storage_kind();
    const auto shN_kind = model.shN_raw().external_storage_kind();
    ASSERT_TRUE(means_kind == "splat.exportable" || means_kind == "vulkan_external_buffer")
        << "means kind=" << means_kind;
    ASSERT_TRUE(shN_kind == "splat.exportable" || shN_kind == "vulkan_external_buffer")
        << "shN kind=" << shN_kind;

    const void* const means_ptr = model.means().data_ptr();
    const void* const shN_ptr = model.shN_raw().data_ptr();
    const size_t means_cap = model.means().capacity();
    const size_t shN_cap = model.shN_raw().capacity();
    const auto means_before = model.means().clone();
    const auto shN_before = model.shN_canonical();

    const size_t new_cap = std::max(means_cap, shN_cap) + 256;
    model.reserve_capacity(new_cap);

    EXPECT_EQ(model.means().data_ptr(), means_ptr);
    EXPECT_EQ(model.shN_raw().data_ptr(), shN_ptr);
    EXPECT_EQ(model.means().capacity(), means_cap);
    EXPECT_EQ(model.shN_raw().capacity(), shN_cap);
    EXPECT_EQ(model.means().external_storage_kind(), means_kind);
    EXPECT_EQ(model.shN_raw().external_storage_kind(), shN_kind);
    EXPECT_TRUE(model.shN_value_quantized());
    EXPECT_TRUE(tensors_equal(model.means(), means_before));
    EXPECT_TRUE(tensors_equal(model.shN_canonical(), shN_before));
}
