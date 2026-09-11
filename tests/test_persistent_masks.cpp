/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *  - frozen GPU mask rebuilt once across N unchanged calls; rebuilt on range change
 *  - cropbox damping mask rebuilt once across unchanged steps; rebuilt on topology
 */

#include "core/camera.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/strategies/strategy_utils.hpp"
#include "training/trainer.hpp"
#include "training_cropbox_mask.hpp"

#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    std::unique_ptr<SplatData> make_splat(size_t n) {
        std::vector<float> means(n * 3, 0.f);
        for (size_t i = 0; i < n; ++i) {
            means[i * 3] = static_cast<float>(i);
        }
        std::vector<float> rot(n * 4, 0.f);
        for (size_t i = 0; i < n; ++i)
            rot[i * 4] = 1.f;
        return std::make_unique<SplatData>(
            0,
            Tensor::from_vector(means, {n, size_t{3}}, Device::CUDA),
            Tensor::zeros({n, size_t{1}, size_t{3}}, Device::CUDA),
            Tensor::zeros({n, size_t{0}, size_t{3}}, Device::CUDA),
            Tensor::zeros({n, size_t{3}}, Device::CUDA),
            Tensor::from_vector(rot, {n, size_t{4}}, Device::CUDA),
            Tensor::zeros({n, size_t{1}}, Device::CUDA),
            1.0f);
    }

    std::shared_ptr<Camera> make_test_camera() {
        return std::make_shared<Camera>(
            Tensor::eye(3, Device::CPU),
            Tensor::zeros({3}, Device::CPU),
            100.0f, 100.0f, 32.0f, 32.0f,
            Tensor(), Tensor(), CameraModelType::PINHOLE,
            "camera.png", std::filesystem::path{}, std::filesystem::path{},
            64, 64, 0);
    }

} // namespace

TEST(PersistentMasksTest, FrozenMaskRebuiltOnceAcrossUnchangedCalls) {
    auto splat = make_splat(32);
    splat->set_frozen_ranges({{.start = 0, .count = 8}});

    reset_frozen_mask_rebuild_count();
    auto m1 = make_frozen_mask(*splat, 32, Device::CUDA);
    ASSERT_TRUE(m1.is_valid());
    EXPECT_EQ(frozen_mask_rebuild_count(), 1u);

    for (int i = 0; i < 10; ++i) {
        auto m = make_frozen_mask(*splat, 32, Device::CUDA);
        ASSERT_TRUE(m.is_valid());
        EXPECT_EQ(m.ptr<bool>(), m1.ptr<bool>()) << "must reuse same device buffer";
    }
    EXPECT_EQ(frozen_mask_rebuild_count(), 1u);

    // Topology change (N) forces rebuild.
    auto m2 = make_frozen_mask(*splat, 16, Device::CUDA);
    ASSERT_TRUE(m2.is_valid());
    EXPECT_EQ(frozen_mask_rebuild_count(), 2u);

    // Range change forces rebuild.
    splat->set_frozen_ranges({{.start = 4, .count = 4}});
    auto m3 = make_frozen_mask(*splat, 16, Device::CUDA);
    ASSERT_TRUE(m3.is_valid());
    EXPECT_EQ(frozen_mask_rebuild_count(), 3u);
}

// Cropbox damping: rebuild exactly once across N unchanged installs; again on
// topology (N) change. Geometry fingerprint also invalidates (cropbox move).
TEST(PersistentMasksTest, CropboxDampingRebuiltOnceAcrossUnchangedSteps) {
    Scene scene;
    // Trainer(scene) requires cameras (hasTrainingData).
    const auto cameras = scene.addGroup("Cameras");
    const auto train_group = scene.addCameraGroup("Training", cameras, 1);
    scene.addCamera("camera.png", train_group, make_test_camera());

    auto model = make_splat(5);
    {
        auto cpu = model->means().cpu();
        float* p = cpu.ptr<float>();
        // Place two means near cropbox center (10,0,0) so remove-mask is non-empty.
        p[0] = 10.0f;
        p[1] = 0;
        p[2] = 0;
        p[3] = 10.5f;
        p[4] = 0;
        p[5] = 0;
        p[6] = 0;
        p[7] = 0;
        p[8] = 0;
        model->means() = cpu.to(Device::CUDA);
    }
    auto* model_ptr = model.get();
    const auto model_id = scene.addSplat("Model", std::move(model));
    scene.setTrainingModelNode("Model");
    const auto cropbox_id = scene.addCropBox("Model_cropbox", model_id);
    CropBoxData cropbox;
    cropbox.min = {-1, -1, -1};
    cropbox.max = {1, 1, 1};
    cropbox.enabled = true;
    cropbox.inverse = false;
    scene.setCropBoxData(cropbox_id, cropbox);
    scene.setNodeTransform("Model_cropbox", glm::translate(glm::mat4(1.f), glm::vec3(10.f, 0.f, 0.f)));

    Trainer trainer(scene);
    TrainerCropboxMaskTestAccess::set_cropbox_lr_scale(trainer, 0.1f);
    AdamConfig cfg{.lr = 0.001f};
    AdamOptimizer opt(*model_ptr, cfg);

    TrainerCropboxMaskTestAccess::reset_rebuild_count(trainer);
    TrainerCropboxMaskTestAccess::install(trainer, *model_ptr, opt);
    EXPECT_EQ(TrainerCropboxMaskTestAccess::rebuild_count(trainer), 1u);
    ASSERT_TRUE(opt.crop_damping_mask().is_valid());
    EXPECT_EQ(opt.crop_damping_mask().numel(), 5u);

    for (int i = 0; i < 8; ++i) {
        TrainerCropboxMaskTestAccess::install(trainer, *model_ptr, opt);
    }
    EXPECT_EQ(TrainerCropboxMaskTestAccess::rebuild_count(trainer), 1u)
        << "unchanged cropbox/topology must not rebuild the damping mask";

    // Scale change forces rebuild (cache key includes scale).
    TrainerCropboxMaskTestAccess::set_cropbox_lr_scale(trainer, 0.2f);
    TrainerCropboxMaskTestAccess::install(trainer, *model_ptr, opt);
    EXPECT_EQ(TrainerCropboxMaskTestAccess::rebuild_count(trainer), 2u);

    // Cropbox transform change → geometry fingerprint miss → rebuild.
    scene.setNodeTransform("Model_cropbox", glm::translate(glm::mat4(1.f), glm::vec3(11.f, 0.f, 0.f)));
    TrainerCropboxMaskTestAccess::install(trainer, *model_ptr, opt);
    EXPECT_EQ(TrainerCropboxMaskTestAccess::rebuild_count(trainer), 3u);
}
