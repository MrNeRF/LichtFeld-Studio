/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *  - frozen GPU mask rebuilt once across N unchanged calls; rebuilt on range change
 *  - crop box damping follows each view and box move; a disabled box installs nothing
 */

#include "core/camera.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/strategies/strategy_utils.hpp"
#include "training/trainer.hpp"
#include "training_cropbox_mask.hpp"

#include <cmath>
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

// Catches crop damping that lags the view or the box: splats inside, in front of or behind the box
// in this view train at full rate, the rest at cropbox_lr_scale, and a moved box applies at once.
TEST(PersistentMasksTest, CropboxViewDampingFollowsTheViewAndTheBox) {
    Scene scene;
    // Trainer(scene) requires cameras (hasTrainingData).
    const auto cameras = scene.addGroup("Cameras");
    const auto train_group = scene.addCameraGroup("Training", cameras, 1);
    scene.addCamera("camera.png", train_group, make_test_camera());

    auto model = make_splat(5);
    {
        auto cpu = model->means().cpu();
        float* p = cpu.ptr<float>();
        // Inside the box at z = 10, behind it, in front of it, beside it, behind but off-axis.
        const float centres[] = {0, 0, 10, 0, 0, 20, 0, 0, 5, 5, 0, 10, 0, 5, 20};
        std::copy(std::begin(centres), std::end(centres), p);
        model->means() = cpu.to(Device::CUDA);
        // Small splats, so their footprint margin stays well inside the gaps between them.
        model->scaling_raw() = Tensor::from_vector(std::vector<float>(15, std::log(0.01f)), {size_t{5}, size_t{3}}, Device::CUDA);
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
    scene.setNodeTransform("Model_cropbox", glm::translate(glm::mat4(1.f), glm::vec3(0.f, 0.f, 10.f)));

    Trainer trainer(scene);
    TrainerCropboxMaskTestAccess::set_cropbox_lr_scale(trainer, 0.1f);
    AdamConfig cfg{.lr = 0.001f};
    AdamOptimizer opt(*model_ptr, cfg);
    const auto camera = make_test_camera();
    const auto damped = [&] {
        TrainerCropboxMaskTestAccess::refresh(trainer);
        TrainerCropboxMaskTestAccess::install_view(trainer, *model_ptr, opt, *camera);
        if (!opt.crop_damping_mask().is_valid())
            return std::vector<bool>{};
        const auto values = opt.crop_damping_mask().to(DataType::Float32).cpu().to_vector();
        return std::vector<bool>(values.begin(), values.end());
    };

    EXPECT_EQ(damped(), (std::vector<bool>{false, false, false, true, true}));
    EXPECT_FALSE(TrainerCropboxMaskTestAccess::contains_camera(trainer, *camera));

    // A box around the camera: its pixels are weighted by the surface they show.
    scene.setNodeTransform("Model_cropbox", glm::mat4(1.f));
    TrainerCropboxMaskTestAccess::refresh(trainer);
    EXPECT_TRUE(TrainerCropboxMaskTestAccess::contains_camera(trainer, *camera));

    scene.setNodeTransform("Model_cropbox", glm::translate(glm::mat4(1.f), glm::vec3(5.f, 0.f, 10.f)));
    EXPECT_EQ(damped(), (std::vector<bool>{true, true, true, false, true}))
        << "a moved box must apply on the next step";

    TrainerCropboxMaskTestAccess::set_cropbox_lr_scale(trainer, 1.0f);
    EXPECT_TRUE(damped().empty()) << "scale 1 installs no damping";
    TrainerCropboxMaskTestAccess::set_cropbox_lr_scale(trainer, 0.1f);

    // A published snapshot replaces the scene read: the viewer says there is no box.
    trainer.setCropBoxSnapshot(std::nullopt);
    EXPECT_TRUE(damped().empty());
    EXPECT_FALSE(TrainerCropboxMaskTestAccess::snapshot(trainer).has_value());
}

// Catches training changes when the crop box is off: no box, no masks, no ROI geometry.
TEST(PersistentMasksTest, DisabledCropboxInstallsNothing) {
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
    cropbox.enabled = false;
    cropbox.inverse = false;
    scene.setCropBoxData(cropbox_id, cropbox);
    scene.setNodeTransform("Model_cropbox", glm::translate(glm::mat4(1.f), glm::vec3(10.f, 0.f, 0.f)));

    Trainer trainer(scene);
    TrainerCropboxMaskTestAccess::set_cropbox_lr_scale(trainer, 0.1f);
    AdamConfig cfg{.lr = 0.001f};
    AdamOptimizer opt(*model_ptr, cfg);
    TrainerCropboxMaskTestAccess::refresh(trainer);
    TrainerCropboxMaskTestAccess::install_view(trainer, *model_ptr, opt, *make_test_camera());
    TrainerCropboxMaskTestAccess::install_score(trainer, *model_ptr, opt);
    EXPECT_FALSE(TrainerCropboxMaskTestAccess::snapshot(trainer).has_value());
    EXPECT_FALSE(opt.crop_damping_mask().is_valid());
    EXPECT_FALSE(opt.crop_score_mask().is_valid());
}

// Catches crop polygon damping or camera tests that use the outline's bounds: the splat in the notch of an L
// trains damped, and a camera standing in the notch is not inside the region.
TEST(PersistentMasksTest, CropPolygonViewDampingAndCameraTestFollowTheOutline) {
    Scene scene;
    // Trainer(scene) requires cameras (hasTrainingData).
    const auto cameras = scene.addGroup("Cameras");
    const auto train_group = scene.addCameraGroup("Training", cameras, 1);
    scene.addCamera("camera.png", train_group, make_test_camera());

    auto model = make_splat(4);
    // In the bottom arm of the L at z = 10, in its notch, behind the bottom arm, beside the L.
    model->means() = Tensor::from_vector({0.5f, -0.5f, 10.0f, 0.5f, 0.5f, 10.0f, 0.5f, -0.5f, 20.0f, 5.0f, 0.0f, 10.0f},
                                         {size_t{4}, size_t{3}}, Device::CUDA);
    model->scaling_raw() = Tensor::from_vector(std::vector<float>(12, std::log(0.01f)), {size_t{4}, size_t{3}}, Device::CUDA);
    auto* model_ptr = model.get();
    const auto model_id = scene.addSplat("Model", std::move(model));
    scene.setTrainingModelNode("Model");
    const std::vector<glm::vec2> outline{{-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 1.0f}};
    ASSERT_NE(scene.addCropPolygon("Model_crop_polygon", model_id, CropPolygonData{.vertices = outline, .half_depth = 0.5f},
                                   glm::translate(glm::mat4(1.f), glm::vec3(0.f, 0.f, 10.f))),
              NULL_NODE);

    Trainer trainer(scene);
    TrainerCropboxMaskTestAccess::set_cropbox_lr_scale(trainer, 0.1f);
    AdamConfig cfg{.lr = 0.001f};
    AdamOptimizer opt(*model_ptr, cfg);
    const auto camera = make_test_camera();
    TrainerCropboxMaskTestAccess::refresh(trainer);
    TrainerCropboxMaskTestAccess::install_view(trainer, *model_ptr, opt, *camera);
    ASSERT_TRUE(opt.crop_damping_mask().is_valid());
    const auto values = opt.crop_damping_mask().to(DataType::Float32).cpu().to_vector();
    EXPECT_EQ(std::vector<bool>(values.begin(), values.end()), (std::vector<bool>{false, true, false, true}));
    EXPECT_FALSE(TrainerCropboxMaskTestAccess::contains_camera(trainer, *camera));

    // The camera in the bottom arm, then in the notch.
    scene.setNodeTransform("Model_crop_polygon", glm::translate(glm::mat4(1.f), glm::vec3(-0.5f, 0.5f, 0.f)));
    TrainerCropboxMaskTestAccess::refresh(trainer);
    EXPECT_TRUE(TrainerCropboxMaskTestAccess::contains_camera(trainer, *camera));
    scene.setNodeTransform("Model_crop_polygon", glm::translate(glm::mat4(1.f), glm::vec3(-0.5f, -0.5f, 0.f)));
    TrainerCropboxMaskTestAccess::refresh(trainer);
    EXPECT_FALSE(TrainerCropboxMaskTestAccess::contains_camera(trainer, *camera));
}
