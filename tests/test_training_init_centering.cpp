/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/uuid.hpp"
#include "io/loader.hpp"
#include "training/training_setup.hpp"

#include <array>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <tuple>
#include <vector>

namespace {
    namespace fs = std::filesystem;
    using namespace lfs::core;

    // The init file deliberately has a different center from points3D. Both
    // must receive the DATASET's shift, not be centered independently.
    const std::array<glm::vec3, 4> dataset_points = {{{10, 20, 0}, {14, 20, 0}, {10, 24, 0}, {14, 24, 0}}};
    const std::array<glm::vec3, 4> init_points = {{{11, 21, 3}, {13, 21, 3}, {11, 23, 5}, {13, 23, 5}}};

    class TrainingInitCentering : public ::testing::TestWithParam<std::tuple<int, bool, int>> {
    protected:
        fs::path root;

        void SetUp() override {
            int devices = 0;
            if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
                GTEST_SKIP() << "CUDA device required for training initialization";
            root = fs::temp_directory_path() / ("lfs_init_centering_" + generate_uuid_v4().to_string());
            fs::create_directories(root / "images");
            fs::create_directories(root / "sparse" / "0");
            std::ofstream(root / "sparse/0/cameras.txt") << "1 PINHOLE 1 1 1 1 0.5 0.5\n";
            // Camera centers (10,20,-40), (14,24,-40); second camera has a
            // 90-degree rotation, so checking only a common T offset is wrong.
            std::ofstream(root / "sparse/0/images.txt")
                << "1 1 0 0 0 -10 -20 40 1 a.png\n\n"
                << "2 0.7071067811865476 0 0 0.7071067811865476 24 -14 40 1 b.png\n\n";
            std::ofstream points(root / "sparse/0/points3D.txt");
            for (size_t i = 0; i < dataset_points.size(); ++i) {
                const auto p = dataset_points[i];
                points << i + 1 << ' ' << p.x << ' ' << p.y << ' ' << p.z << " 128 128 128 0 1 0 2 0\n";
            }
            static constexpr unsigned char png[] = {
                0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
                0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00,
                0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x00, 0x01, 0x00, 0x00,
                0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
                0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
            for (const auto* name : {"a.png", "b.png"}) {
                std::ofstream image(root / "images" / name, std::ios::binary);
                image.write(reinterpret_cast<const char*>(png), sizeof(png));
            }
            for (const bool gaussian : {false, true}) {
                std::ofstream ply(root / (gaussian ? "splat.ply" : "points.ply"), std::ios::binary);
                ply << "ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n";
                if (gaussian) {
                    for (const auto* field : {"f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"})
                        ply << "property float " << field << '\n';
                } else {
                    ply << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
                }
                ply << "end_header\n";
                for (const auto p : init_points) {
                    const std::array<float, 3> xyz{p.x, p.y, p.z};
                    ply.write(reinterpret_cast<const char*>(xyz.data()), sizeof(xyz));
                    if (gaussian) {
                        const std::array<float, 11> attributes{0.1f, 0.2f, 0.3f, 0.7f, -2, -3, -4, 1, 0, 0, 0};
                        ply.write(reinterpret_cast<const char*>(attributes.data()), sizeof(attributes));
                    } else {
                        const std::array<unsigned char, 3> rgb{128, 128, 128};
                        ply.write(reinterpret_cast<const char*>(rgb.data()), sizeof(rgb));
                    }
                }
            }
        }

        void TearDown() override {
            if (!root.empty()) {
                std::error_code ec;
                fs::remove_all(root, ec);
            }
        }

        void check_gaussian_attributes(const SplatData& model, const size_t offset) {
            const auto check_rows = [offset](const Tensor& tensor, const std::vector<float>& expected) {
                const auto cpu = tensor.cpu().contiguous();
                const auto* values = cpu.ptr<float>();
                for (size_t i = offset; i < offset + init_points.size(); ++i)
                    for (size_t c = 0; c < expected.size(); ++c)
                        EXPECT_NEAR(values[i * expected.size() + c], expected[c], 1e-6f);
            };
            check_rows(model.rotation_raw(), {1, 0, 0, 0});
            check_rows(model.scaling_raw(), {-2, -3, -4});
            check_rows(model.opacity_raw(), {0.7f});
            check_rows(model.sh0(), {0.1f, 0.2f, 0.3f});
        }

        void check_positions(const Tensor& means, const std::array<glm::vec3, 4>& source,
                             const glm::vec3 origin, const Scene& scene, const size_t offset = 0) {
            auto cpu = means.cpu().contiguous();
            ASSERT_GE(cpu.numel(), (offset + source.size()) * 3);
            const auto* values = cpu.ptr<float>() + offset * 3;
            for (size_t i = 0; i < source.size(); ++i) {
                const glm::vec3 actual{values[i * 3], values[i * 3 + 1], values[i * 3 + 2]};
                for (int axis = 0; axis < 3; ++axis)
                    EXPECT_NEAR(actual[axis], (source[i] - origin)[axis], 1e-4f);
                // Point-to-camera vectors must stay invariant. A rotated
                // camera also catches mistakes in translating camera T.
                for (const auto& camera : scene.getAllCameras()) {
                    const auto position = camera->cam_position().cpu();
                    const auto* c = position.ptr<float>();
                    const glm::vec3 original_center = camera->image_name() == "a.png"
                                                          ? glm::vec3{10, 20, -40}
                                                          : glm::vec3{14, 24, -40};
                    for (int axis = 0; axis < 3; ++axis)
                        EXPECT_NEAR(actual[axis] - c[axis], (source[i] - original_center)[axis], 1e-4f);
                }
            }
        }
    };

    TEST_P(TrainingInitCentering, PreviewAndTrainingKeepCameraAlignment) {
        const auto [mode, async, init_kind] = GetParam();
        param::TrainingParameters params;
        params.dataset.data_path = root;
        params.dataset.centralize_dataset = mode == 0 ? "off" : mode == 1 ? "by_cameras"
                                                                          : "by_pointcloud";
        params.optimization.sh_degree = 0;
        params.optimization.max_cap = 100;
        if (init_kind == 1 || init_kind == 2 || init_kind == 4)
            params.init_path = (root / (init_kind == 1 ? "points.ply" : "splat.ply")).string();

        const bool added = init_kind >= 3;
        if (added) {
            params.add_splat_paths.push_back(root / "splat.ply");
            params.add_splat_freeze.push_back(true);
        }
        Scene scene;
        if (async) {
            auto loader = lfs::io::Loader::create();
            lfs::io::LoadOptions options;
            options.centralize = mode == 0 ? lfs::io::CentralizeDataset::Off : mode == 1 ? lfs::io::CentralizeDataset::ByCameras
                                                                                         : lfs::io::CentralizeDataset::ByPointCloud;
            auto loaded = loader->load(root, options);
            ASSERT_TRUE(loaded) << loaded.error().format();
            const auto applied = lfs::training::applyLoadResultToScene(params, scene, std::move(*loaded));
            ASSERT_TRUE(applied) << applied.error();
        } else {
            const auto loaded = lfs::training::loadTrainingDataIntoScene(params, scene);
            ASSERT_TRUE(loaded) << loaded.error();
        }
        const glm::vec3 origin = mode == 0 ? glm::vec3{0} : mode == 1 ? glm::vec3{12, 22, -40}
                                                                      : glm::vec3{12, 22, 0};
        const auto& source = (init_kind == 0 || init_kind == 3) ? dataset_points : init_points;
        ASSERT_TRUE(scene.getInitialPointCloud());
        ASSERT_EQ(scene.getAllCameras().size(), 2u);
        check_positions(scene.getInitialPointCloud()->means, source, origin, scene);

        // Exercise deferred initialization, before any optimizer can move a
        // point; the viewer uses this owner-thread capture / worker split.
        auto capture = lfs::training::captureTrainingModelGraph(scene);
        auto prepared = lfs::training::prepareTrainingModel(params, scene, {}, async ? &capture : nullptr);
        ASSERT_TRUE(prepared) << prepared.error();
        ASSERT_TRUE(prepared->has_value());
        ASSERT_EQ((**prepared).model->size(), added ? 8u : 4u);
        if (init_kind == 2 || init_kind == 4)
            check_gaussian_attributes(*(**prepared).model, 0);
        if (added)
            check_gaussian_attributes(*(**prepared).model, 4);
        check_positions((**prepared).model->means(), source, origin, scene);
        if (added)
            check_positions((**prepared).model->means(), init_points, origin, scene, 4);
        const auto installed = lfs::training::installTrainingModel(scene, std::move(**prepared));
        ASSERT_TRUE(installed) << installed.error();
        ASSERT_TRUE(scene.getTrainingModel());
        check_positions(scene.getTrainingModel()->means(), source, origin, scene);
        if (added)
            check_positions(scene.getTrainingModel()->means(), init_points, origin, scene, 4);
        // Reusing an already initialized model must never shift it again.
        params.add_splat_paths.clear();
        params.add_splat_freeze.clear();
        auto reused = lfs::training::prepareTrainingModel(params, scene);
        ASSERT_TRUE(reused) << reused.error();
        EXPECT_FALSE(reused->has_value());
        check_positions(scene.getTrainingModel()->means(), source, origin, scene);

        // Loading another dataset with centering off must not reuse the old
        // origin. Exercise the real clear/load path, not just the setter.
        scene.clear();
        params.dataset.centralize_dataset = "off";
        const auto reloaded = lfs::training::loadTrainingDataIntoScene(params, scene);
        ASSERT_TRUE(reloaded) << reloaded.error();
        const auto initialized = lfs::training::initializeTrainingModel(params, scene);
        ASSERT_TRUE(initialized) << initialized.error();
        ASSERT_TRUE(scene.getTrainingModel());
        check_positions(scene.getTrainingModel()->means(), source, glm::vec3{0}, scene);
    }

    TEST_F(TrainingInitCentering, DeferredInitializationUsesCapturedOrigin) {
        param::TrainingParameters params;
        params.dataset.data_path = root;
        params.dataset.centralize_dataset = "by_cameras";
        params.init_path = (root / "splat.ply").string();
        params.optimization.sh_degree = 0;
        params.optimization.max_cap = 100;
        Scene scene;
        const auto loaded = lfs::training::loadTrainingDataIntoScene(params, scene);
        ASSERT_TRUE(loaded) << loaded.error();
        const auto capture = lfs::training::captureTrainingModelGraph(scene);

        // Worker preparation must use the owner-thread snapshot even when
        // the live scene has moved on to another dataset origin.
        scene.setTrainingDataOrigin(glm::vec3{100, 200, 300});
        auto prepared = lfs::training::prepareTrainingModel(params, scene, {}, &capture);
        ASSERT_TRUE(prepared) << prepared.error();
        ASSERT_TRUE(prepared->has_value());
        check_positions((**prepared).model->means(), init_points, glm::vec3{12, 22, -40}, scene);
    }

    INSTANTIATE_TEST_SUITE_P(LoadPaths, TrainingInitCentering,
                             ::testing::Combine(::testing::Values(0, 1, 2), ::testing::Bool(), ::testing::Values(0, 1, 2, 3, 4)));
} // namespace
