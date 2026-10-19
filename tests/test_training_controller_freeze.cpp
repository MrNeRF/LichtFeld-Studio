/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/event_bridge/control_boundary.hpp"
#include "core/image_io.hpp"
#include "core/scene.hpp"
#include "licht_test_support.hpp"
#include "training/trainer.hpp"
#include "training_snapshot_test_helpers.hpp"

#include <array>
#include <gtest/gtest.h>
#include <sstream>
#include <tuple>

namespace {
    using namespace lfs::core;
    using namespace lfs::training;
    using namespace lfs::training::test;

    using ModelSnapshot = std::array<TensorByteSnapshot, 6>;

    std::array<const Tensor*, 6> fields(const SplatData& model) {
        return {&model.means(), &model.scaling_raw(), &model.rotation_raw(),
                &model.opacity_raw(), &model.sh0(), &model.shN()};
    }

    ModelSnapshot capture(const SplatData& model) {
        ModelSnapshot result;
        const auto tensors = fields(model);
        for (size_t i = 0; i < tensors.size(); ++i)
            result[i] = capture_tensor_bytes(*tensors[i]);
        return result;
    }

    class TrainingControllerFreeze : public testing::TestWithParam<std::tuple<std::string, bool>> {
    protected:
        void run(const bool freeze, const bool controller = true, const int activation = 10) {
            const auto [strategy, gut] = GetParam();
            const lfs::test::licht::TemporaryDirectory directory("controller-freeze");
            const auto image_path = directory.path / "image.png";
            save_image(image_path, Tensor::full({3, 32, 32}, 0.7f, Device::CPU));
            Scene scene;
            auto camera = std::make_shared<Camera>(
                Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU),
                32.0f, 32.0f, 16.0f, 16.0f, Tensor{}, Tensor{}, CameraModelType::PINHOLE,
                "image.png", image_path, std::filesystem::path{}, 32, 32, 0);
            scene.addCamera("image.png", scene.addGroup("Cameras"), camera);
            constexpr size_t count = 100;
            std::vector<float> means(count * 3), rotations(count * 4, 0.0f);
            for (size_t i = 0; i < count; ++i) {
                means[i * 3] = (static_cast<float>(i % 10) - 4.5f) * 0.1f;
                means[i * 3 + 1] = (static_cast<float>(i / 10) - 4.5f) * 0.1f;
                means[i * 3 + 2] = 2.0f;
                rotations[i * 4] = 1.0f;
            }
            auto model = std::make_unique<SplatData>(
                1, Tensor::from_vector(means, {count, 3}, Device::CUDA),
                Tensor::full({count, 1, 3}, 0.2f, Device::CUDA),
                Tensor::zeros({count, 3, 3}, Device::CUDA),
                Tensor::full({count, 3}, -2.5f, Device::CUDA),
                Tensor::from_vector(rotations, {count, 4}, Device::CUDA),
                Tensor::full({count, 1}, 0.5f, Device::CUDA), 1.0f);
            scene.setTrainingModelNode(scene.addSplat("Model", std::move(model)));

            param::TrainingParameters params;
            params.dataset.data_path = directory.path;
            params.dataset.output_path = directory.path / "output";
            params.no_download = true;
            auto& opt = params.optimization;
            opt = strategy == "mrnf"   ? param::OptimizationParameters::mrnf_defaults()
                  : strategy == "mcmc" ? param::OptimizationParameters::mcmc_defaults()
                                       : param::OptimizationParameters::igs_plus_defaults();
            opt.gut = gut;
            opt.iterations = 30;
            opt.start_refine = 0;
            opt.stop_refine = 30;
            opt.refine_every = 5;
            opt.morton_reorder_interval = 5;
            opt.max_cap = 200;
            opt.sh_degree = 1;
            opt.sh_degree_interval = 10;
            opt.headless = true;
            opt.enable_eval = false;
            opt.save_steps.clear();
            opt.use_exposure_correction = false;
            opt.use_ppisp = true;
            opt.ppisp_use_controller = controller;
            opt.ppisp_controller_activation_step = activation;
            opt.ppisp_freeze_gaussians_on_distill = freeze;

            Trainer trainer(scene);
            const auto initialized = trainer.initialize(params);
            ASSERT_TRUE(initialized) << initialized.error();
            const auto initial = capture(trainer.get_strategy().get_model());
            ModelSnapshot boundary;
            int boundary_degree = -1;
            std::string controller_before, controller_after;
            int samples = 0;
            auto& hooks = ControlBoundary::instance();
            const auto handle = hooks.register_callback(ControlHook::PostStep, [&](const HookContext& ctx) {
                if (ctx.trainer != &trainer)
                    return;
                const auto& current = trainer.get_strategy().get_model();
                if (ctx.iteration == 9) {
                    boundary = capture(current);
                    boundary_degree = current.get_active_sh_degree();
                    EXPECT_NE(initial[0].bytes, boundary[0].bytes) << "Pre-activation Gaussian learning must continue";
                    if (const auto* pool = trainer.getPPISPControllerPool()) {
                        std::ostringstream stream;
                        pool->serialize_inference(stream);
                        controller_before = stream.str();
                    }
                }
                if (ctx.iteration == 10 || ctx.iteration == 29 || ctx.iteration == 30) {
                    ++samples;
                    if (freeze && controller && activation == 10) {
                        EXPECT_EQ(current.get_active_sh_degree(), boundary_degree);
                        const auto tensors = fields(current);
                        for (size_t i = 0; i < tensors.size(); ++i)
                            expect_tensor_bytes_equal(boundary[i], *tensors[i], "Frozen Gaussian field");
                    } else {
                        EXPECT_NE(boundary[0].bytes, capture(current)[0].bytes)
                            << "Gaussian learning must continue when freeze is inactive";
                    }
                }
            });
            const auto trained = trainer.train();
            hooks.unregister_callback(ControlHook::PostStep, handle);
            ASSERT_TRUE(trained) << lfs::format_for_developer(trained.error());
            EXPECT_EQ(samples, 3);
            if (freeze && controller && activation == 10) {
                std::ostringstream stream;
                trainer.getPPISPControllerPool()->serialize_inference(stream);
                controller_after = stream.str();
                EXPECT_NE(controller_before, controller_after) << "Controller must keep learning";
            }
            trainer.shutdown();
        }
    };

    TEST_P(TrainingControllerFreeze, FrozenModelIsUnchangedAcrossRefinementAndStopBoundary) {
        run(true);
    }
    TEST_P(TrainingControllerFreeze, DisabledFreezeKeepsGaussianLearning) {
        run(false);
    }
    TEST_P(TrainingControllerFreeze, BeforeActivationKeepsGaussianLearning) {
        run(true, true, 30);
    }
    TEST_P(TrainingControllerFreeze, WithoutControllerKeepsGaussianLearning) {
        run(true, false);
    }

    INSTANTIATE_TEST_SUITE_P(Strategies, TrainingControllerFreeze,
                             testing::Values(std::tuple{"mrnf", false}, std::tuple{"mcmc", false},
                                             std::tuple{"igs+", false}, std::tuple{"mrnf", true},
                                             std::tuple{"mcmc", true}));
} // namespace
