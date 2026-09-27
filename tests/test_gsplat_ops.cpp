/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/training/ops/registry.hpp"
#include "training/rasterization/gsplat_rasterizer.hpp"

#include <array>
#include <cstring>
#include <gtest/gtest.h>

namespace {
    using namespace lfs;
    using core::Device;
    using core::Tensor;

    void expect_bytes(const Tensor& actual, const Tensor& expected) {
        auto a = actual.cpu().contiguous();
        auto b = expected.cpu().contiguous();
        ASSERT_EQ(a.shape(), b.shape());
        ASSERT_EQ(a.dtype(), b.dtype());
        ASSERT_EQ(a.bytes(), b.bytes());
        EXPECT_EQ(std::memcmp(a.data_ptr(), b.data_ptr(), a.bytes()), 0);
    }

    class GsplatOpsTest : public test::CudaBackendTest {};
} // namespace

TEST_F(GsplatOpsTest, ForwardBackwardAndReleaseMatchDirectLaunchers) {
    Tensor::manual_seed(42);
    const auto& ops = *training::training_ops(core::GpuBackend::CUDA).gsplat;
    auto means = Tensor::from_vector({0.f, 0.f, 3.f}, {1, 3}, Device::GPU);
    auto scales = Tensor::full({1, 3}, -1.f, Device::GPU);
    auto rotations = Tensor::from_vector({1.f, 0.f, 0.f, 0.f}, {1, 4}, Device::GPU);
    auto opacity = Tensor::full({1, 1}, 1.f, Device::GPU);
    auto sh0 = Tensor::full({1, 1, 3}, .25f, Device::GPU);
    auto shN = Tensor::zeros({384}, Device::GPU);
    auto view = Tensor::eye(4, Device::GPU);
    auto bg = Tensor::full({3}, .1f, Device::GPU);
    Tensor empty;
    const gpu_ops::SplatInputs inputs{means, scales, rotations, opacity, sh0, shN, empty};
    gpu_ops::GsplatParams params{
        .full_image = {1, 1},
        .intrinsics = {1.f, 1.f, .5f, .5f},
        .sh = {.active_bases = 4, .layout_bases = 4}};
    gpu_ops::GsplatSaved direct{training::gsplat_create()}, table{ops.create()};
    ASSERT_TRUE(direct.backend);
    ASSERT_TRUE(table.backend);
    EXPECT_NE(direct.backend.get(), table.backend.get());

    struct GradientBuffers {
        std::array<Tensor*, 6> tensors;
        std::vector<gpu_ops::AdamSlot> requested;
    };
    // One pixel avoids unordered atomic additions in the direct comparison.
    for (const auto bases : {1u, 4u}) {
        params.sh.active_bases = bases;
        for (const auto mode : {gpu_ops::GsplatRenderMode::RGB, gpu_ops::GsplatRenderMode::RGB_D}) {
            params.render_mode = mode;
            std::vector<Tensor> reference;
            for (bool through_table : {false, true}) {
                auto& saved = through_table ? table : direct;
                Tensor image, alpha, depth, normal;
                const auto result = (through_table ? ops.forward : training::gsplat_forward)(
                    saved, inputs, view, empty, empty, bg, empty, params, {image, alpha, depth, normal});
                ASSERT_EQ(result.code, gpu_ops::RasterResult::Code::Success);
                ASSERT_TRUE(result.has_work);
                auto image_grad = Tensor::full({mode == gpu_ops::GsplatRenderMode::RGB ? 3ul : 4ul, 1, 1}, .125f, Device::GPU);
                auto alpha_grad = Tensor::full({1, 1, 1}, .25f, Device::GPU);
                auto gm = Tensor::zeros_like(means), gs = Tensor::zeros_like(scales);
                auto gr = Tensor::zeros_like(rotations), go = Tensor::zeros_like(opacity);
                auto g0 = Tensor::zeros_like(sh0), gn = Tensor::zeros_like(shN);
                auto densification = Tensor::zeros({2, 1}, Device::GPU);
                auto errors = Tensor::full({1, 1}, .3f, Device::GPU);
                auto edges = Tensor::full({1, 1}, .7f, Device::GPU);
                auto scores = Tensor::zeros({1}, Device::GPU), share = Tensor::zeros({1}, Device::GPU);
                GradientBuffers gradient_tensors{{&gm, &gs, &gr, &go, &g0, &gn}, {}};
                const gpu_ops::GsplatGradients gradients{&gradient_tensors,
                                                         [](void* owner, gpu_ops::AdamSlot slot) -> Tensor& {
                                                             auto& buffers = *static_cast<GradientBuffers*>(owner);
                                                             buffers.requested.push_back(slot);
                                                             return *buffers.tensors[static_cast<size_t>(slot)];
                                                         }};
                (through_table ? ops.backward : training::gsplat_backward)(saved, image_grad, alpha_grad,
                                                                           gradients, densification, errors, edges, scores, share);
                std::vector<gpu_ops::AdamSlot> expected{gpu_ops::AdamSlot::Means, gpu_ops::AdamSlot::Scaling,
                                                        gpu_ops::AdamSlot::Rotation, gpu_ops::AdamSlot::Opacity};
                if (bases > 1)
                    expected.push_back(gpu_ops::AdamSlot::ShN);
                expected.push_back(gpu_ops::AdamSlot::Sh0);
                EXPECT_EQ(gradient_tensors.requested, expected);
                std::vector<Tensor> actual{image, alpha, gm, gs, gr, go, g0, gn, densification, scores, share};
                if (depth.is_valid())
                    actual.push_back(depth);
                if (!through_table) {
                    for (auto& value : actual)
                        reference.push_back(value.clone());
                } else {
                    ASSERT_EQ(actual.size(), reference.size());
                    for (size_t i = 0; i < actual.size(); ++i)
                        expect_bytes(actual[i], reference[i]);
                }
                (through_table ? ops.release : training::gsplat_release)(saved);
            }
        }
    }
    Tensor image, alpha, depth, normal;
    ASSERT_EQ(ops.forward(table, inputs, view, empty, empty, bg, empty, params,
                          {image, alpha, depth, normal})
                  .code,
              gpu_ops::RasterResult::Code::Success);
    const auto before = image.clone();
    const auto* first_intersections = training::cuda_gsplat_frame(table).isect_ids_ptr;
    ops.release(table);
    Tensor second_image, second_alpha, second_depth, second_normal;
    const auto other_background = Tensor::full({3}, .8f, Device::GPU);
    ASSERT_EQ(training::gsplat_forward(direct, inputs, view, empty, empty, other_background, empty, params,
                                       {second_image, second_alpha, second_depth, second_normal})
                  .code,
              gpu_ops::RasterResult::Code::Success);
    EXPECT_NE(image.data_ptr(), second_image.data_ptr());
    EXPECT_NE(first_intersections,
              training::cuda_gsplat_frame(direct).isect_ids_ptr);
    expect_bytes(image, before);
    training::gsplat_release(direct);
    ops.release(table);
    ops.release(table);
    ASSERT_EQ(ops.forward(table, inputs, view, empty, empty, bg, empty, params,
                          {image, alpha, depth, normal})
                  .code,
              gpu_ops::RasterResult::Code::Success);
    expect_bytes(image, before);
    ops.release(table);
}

TEST(TrainingOpsCapability, GsplatFamilyIsCudaOnly) {
    using namespace lfs;
    EXPECT_NE(training::training_ops(core::GpuBackend::CUDA).gsplat, nullptr);
    for (auto backend : {core::GpuBackend::Vulkan, core::GpuBackend::Metal}) {
        EXPECT_EQ(training::training_ops(backend).gsplat, nullptr);
        EXPECT_NE(training::unavailable_training_family(backend, training::Family::Gsplat)->find("Gsplat"), std::string::npos);
    }
    core::param::TrainingParameters params;
    params.optimization.gut = true;
    EXPECT_TRUE(training::required_training_families(params, {}).test(static_cast<size_t>(training::Family::Gsplat)));
    auto cuda = training::training_ops(core::GpuBackend::CUDA);
    cuda.gsplat = nullptr;
    training::FamilySet required;
    required.set(static_cast<size_t>(training::Family::Gsplat));
    EXPECT_EQ(training::missing_training_families(cuda, required), (std::vector<std::string_view>{"Gsplat"}));
}
