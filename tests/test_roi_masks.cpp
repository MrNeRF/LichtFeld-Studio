/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "training/ops/roi_masks.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <optional>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;

    // Splat centres: inside the box, behind it, in front of it, beside it, behind but off-axis.
    const std::vector<float> kCentres{0, 0, 10, 0, 0, 20, 0, 0, 5, 5, 0, 10, 0, 5, 20};
    const lfs::training::RoiBox kBox{glm::mat4(1.0f), {-1.0f, -1.0f, 9.0f}, {1.0f, 1.0f, 11.0f}, false};

    class RoiMasks : public testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << "Backend unavailable";
            scope_.emplace(GetParam());
        }

        static std::vector<bool> bools(const Tensor& mask) {
            const auto values = mask.to(DataType::Float32).cpu().to_vector();
            return {values.begin(), values.end()};
        }

        // Rejected rows of the five centres for the given box-space cameras.
        static std::vector<bool> reject(const std::vector<float>& cameras, const bool inverse,
                                        const Tensor& scales = {}, const Tensor& deleted = {},
                                        Tensor* in_roi = nullptr) {
            const auto means = Tensor::from_vector(kCentres, {size_t{5}, size_t{3}}, Device::GPU);
            const auto centres = Tensor::from_vector(cameras, {cameras.size() / 3, size_t{3}}, Device::GPU);
            auto rejected = Tensor::empty({size_t{5}}, Device::GPU, DataType::Bool);
            Tensor no_roi;
            auto box = kBox;
            box.inverse = inverse;
            lfs::training::roi_reject_mask(means, scales, deleted, centres, box, rejected, in_roi ? *in_roi : no_roi);
            return bools(rejected);
        }

        std::optional<lfs::core::GpuBackendScope> scope_;
    };

    // Catches crop damping that slows the background behind (or the occluders in front of) the crop
    // box: those splats shade full-weight pixels, so they must train at full rate.
    TEST_P(RoiMasks, ViewMaskKeepsSplatsInsideInFrontAndBehindTheBox) {
        EXPECT_EQ(reject({0, 0, 0}, false), (std::vector<bool>{false, false, false, true, true}));
        EXPECT_EQ(reject({0, 0, 0}, true), (std::vector<bool>{true, true, true, false, false}));
        // From the side the box now hides splat 3 instead of splats 1 and 2.
        EXPECT_EQ(reject({20, 0, 10}, false), (std::vector<bool>{false, true, true, false, true}));
        // Deleted rows are never damped.
        const auto deleted = Tensor::from_vector({0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, {size_t{5}}, Device::GPU).to(DataType::Bool);
        EXPECT_EQ(reject({0, 0, 0}, false, {}, deleted), (std::vector<bool>{false, false, false, false, true}));
    }

    // Catches the halo at the box rim: a splat whose footprint reaches the box trains at full rate.
    // Inverse boxes shrink instead.
    TEST_P(RoiMasks, FootprintMarginReachesSplatsBesideTheBox) {
        const auto scales = [](const float sigma) {
            return Tensor::from_vector(std::vector<float>(15, std::log(sigma)), {size_t{5}, size_t{3}}, Device::GPU);
        };
        EXPECT_EQ(reject({0, 0, 0}, false, scales(0.01f)), (std::vector<bool>{false, false, false, true, true}));
        EXPECT_EQ(reject({0, 0, 0}, false, scales(2.0f)), (std::vector<bool>{false, false, false, false, false}));
        EXPECT_EQ(reject({0, 0, 0}, true, scales(2.0f)), (std::vector<bool>{false, false, false, false, false}));
        EXPECT_NEAR(lfs::training::box_units_per_model_unit(glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, 3.0f, 2.0f))),
                    3.0f, 1e-4f);
    }

    // Catches refine scores damped for splats that some training camera sees through the box.
    TEST_P(RoiMasks, RelevanceMaskKeepsSplatsAnyCameraSeesThroughTheBox) {
        auto in_roi = Tensor::empty({size_t{5}}, Device::GPU, DataType::Bool);
        EXPECT_EQ(reject({0, 0, 0}, false, {}, {}, &in_roi), (std::vector<bool>{false, false, false, true, true}));
        EXPECT_EQ(bools(in_roi), (std::vector<bool>{true, false, false, false, false}));
        // A second camera behind splat 4 sees it through the box: only splat 3 stays irrelevant.
        EXPECT_EQ(reject({0, 0, 0, 0, -5, 0}, false), (std::vector<bool>{false, false, false, true, false}));
        EXPECT_EQ(reject({0, 0, 0}, true, {}, {}, &in_roi), (std::vector<bool>{true, true, true, false, false}));
        EXPECT_EQ(bools(in_roi), (std::vector<bool>{false, true, true, true, true}));
    }

    // An L outline: the square [-1, 1]^2 without its (0, 1) x (0, 1) quarter, the notch, over |z| <= 0.5.
    const std::vector<float> kLOutline{-1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f};
    // Splat centres in polygon space: in the bottom arm, in the notch, below the bottom arm, beside the outline.
    const std::vector<float> kPolygonCentres{0.5f, -0.5f, 0.0f, 0.5f, 0.5f, 0.0f, 0.5f, -0.5f, -5.0f, 5.0f, 0.0f, 0.0f};

    lfs::training::RoiBox l_prism(const bool inverse) {
        return {glm::mat4(1.0f), {-1.0f, -1.0f, -0.5f}, {1.0f, 1.0f, 0.5f}, inverse, Tensor::from_vector(kLOutline, {kLOutline.size() / 2, size_t{2}}, Device::GPU)};
    }

    // Catches polygon damping that tests the outline's bounds: seen from above, the notch splat is neither in nor
    // in front of or behind the L, so it trains damped, while the splat under the bottom arm trains at full rate.
    TEST_P(RoiMasks, PolygonViewMaskFollowsTheConcaveOutline) {
        const auto means = Tensor::from_vector(kPolygonCentres, {size_t{4}, size_t{3}}, Device::GPU);
        const auto camera = Tensor::from_vector({0.0f, 0.0f, 10.0f}, {size_t{1}, size_t{3}}, Device::GPU);
        auto rejected = Tensor::empty({size_t{4}}, Device::GPU, DataType::Bool);
        const auto run = [&](const float sigma, const bool inverse) {
            const auto scaling = Tensor::from_vector(std::vector<float>(12, std::log(sigma)), {size_t{4}, size_t{3}},
                                                     Device::GPU);
            Tensor no_roi;
            lfs::training::roi_reject_mask(means, scaling, {}, camera, l_prism(inverse), rejected, no_roi);
            return bools(rejected);
        };
        EXPECT_EQ(run(0.01f, false), (std::vector<bool>{false, true, false, true}));
        EXPECT_EQ(run(0.01f, true), (std::vector<bool>{true, false, true, false}));
        // A footprint reaching over the notch edges keeps the notch splat at full rate.
        EXPECT_EQ(run(0.2f, false), (std::vector<bool>{false, false, false, true}));
    }

    // Catches polygon refine scores or ROI rows that use the outline's bounds instead of the outline.
    TEST_P(RoiMasks, PolygonRelevanceMaskAndRoiRowsFollowTheConcaveOutline) {
        const auto means = Tensor::from_vector(kPolygonCentres, {size_t{4}, size_t{3}}, Device::GPU);
        const auto cameras = Tensor::from_vector({0.0f, 0.0f, 10.0f}, {size_t{1}, size_t{3}}, Device::GPU);
        auto rejected = Tensor::empty({size_t{4}}, Device::GPU, DataType::Bool);
        auto in_roi = Tensor::empty({size_t{4}}, Device::GPU, DataType::Bool);
        lfs::training::roi_reject_mask(means, {}, {}, cameras, l_prism(false), rejected, in_roi);
        EXPECT_EQ(bools(rejected), (std::vector<bool>{false, true, false, true}));
        EXPECT_EQ(bools(in_roi), (std::vector<bool>{true, false, false, false}));
    }

    INSTANTIATE_TEST_SUITE_P(Backends, RoiMasks, testing::Values(GpuBackend::CUDA, GpuBackend::Vulkan, GpuBackend::Metal),
                             [](const auto& info) {
                                 return info.param == GpuBackend::Metal    ? "Metal"
                                        : info.param == GpuBackend::Vulkan ? "Vulkan"
                                                                           : "CUDA";
                             });
} // namespace
