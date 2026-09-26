/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/memory_pressure.hpp"
#include "core/scene.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_completion.hpp"
#include "gui/gallery_scene_publication.hpp"
#include "io/loader.hpp"
#include "licht_test_support.hpp"
#include "project/session_state.hpp"
#include "rendering/rendering_types.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <stdexcept>

namespace {
    using namespace lfs::core;
    using namespace lfs::vis::gui;

    std::unique_ptr<SplatData> model() {
        constexpr size_t count = 8;
        std::vector<float> means(count * 3), rotations(count * 4);
        for (size_t row = 0; row < count; ++row) {
            means[row * 3] = static_cast<float>(row);
            rotations[row * 4] = 1.0f;
        }
        auto result = std::make_unique<SplatData>(
            1, Tensor::from_vector(means, {count, 3}, Device::GPU), Tensor::zeros({count, 1, 3}, Device::GPU),
            Tensor::full({count, 3, 3}, 0.25f, Device::GPU),
            Tensor::zeros({count, 3}, Device::GPU), Tensor::from_vector(rotations, {count, 4}, Device::GPU),
            Tensor::zeros({count, 1}, Device::GPU), 1.0f);
        result->set_active_sh_degree(1);
        result->soft_delete(Tensor::from_vector(
                                std::vector<int>{0, 0, 1, 0, 0, 0, 0, 0}, {count}, Device::GPU)
                                .to(DataType::Bool));
        return result;
    }

    class SceneSnapshotBackend : public ::testing::TestWithParam<GpuBackend> {
    protected:
        std::optional<GpuBackendScope> backend;
        void SetUp() override {
            if (!gpu_backend_available(GetParam()))
                GTEST_SKIP() << "Requested backend unavailable";
            backend.emplace(GetParam());
            MemoryPressureCoordinator::instance().reset_for_testing();
        }
        void TearDown() override {
            MemoryPressureCoordinator::instance().reset_for_testing();
        }
        GpuBackend other() const {
            return GetParam() == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA;
        }
        MemoryDomain domain() const {
            return GetParam() == GpuBackend::CUDA ? MemoryDomain::CudaDevice : MemoryDomain::VulkanDevice;
        }
    };

    TEST_P(SceneSnapshotBackend, CopiesOnlyVisibleDataOnItsStorageBackend) {
        Scene scene;
        const auto visible = scene.addSplat("visible", model());
        const auto hidden = scene.addSplat("hidden", model());
        scene.setNodeVisibility(hidden, false);
        const auto source = scene.getNodeById(visible)->model->means_raw().to_vector();
        std::vector<MemoryDomain> queried;
        MemoryPressureCoordinator::instance().set_free_memory_probe([&](const MemoryDomain requested) {
            queried.push_back(requested);
            return requested == domain() ? std::numeric_limits<size_t>::max() / 2 : 0;
        });
        std::vector<Scene::SplatSnapshot> snapshots;
        {
            // A worker's default must not redirect an existing model's storage.
            GpuBackendScope different_default(other());
            snapshots = scene.snapshotVisibleSplats();
        }
        ASSERT_EQ(snapshots.size(), 1u);
        EXPECT_EQ(queried, std::vector<MemoryDomain>{domain()});
        EXPECT_EQ(gpu_backend_of(snapshots[0].data->means_raw()), GetParam());
        EXPECT_EQ(snapshots[0].active_sh_degree, 1);
        EXPECT_NE(snapshots[0].data->means_raw().data_ptr(), scene.getNodeById(visible)->model->means_raw().data_ptr());
        scene.getNodeById(visible)->model->means_raw().fill_(99.0f);
        scene.clear();
        EXPECT_EQ(snapshots[0].data->means_raw().to_vector(), source);
        EXPECT_EQ(snapshots[0].data->visible_count(), 7u);
    }

    TEST_P(SceneSnapshotBackend, MaterializesSliceAndDeletionWithAnotherDefault) {
        auto source = std::shared_ptr<SplatData>(model());
        const Scene::SplatSnapshot snapshot{source, glm::mat4{1.0f}, 1, 5, 0};
        std::shared_ptr<SplatData> extracted;
        {
            GpuBackendScope different_default(other());
            extracted = snapshot.materialize();
        }
        ASSERT_NE(extracted, nullptr);
        EXPECT_EQ(gpu_backend_of(extracted->means_raw()), GetParam());
        EXPECT_EQ(extracted->size(), 4u);
        EXPECT_EQ(extracted->get_active_sh_degree(), 0);
        EXPECT_EQ(extracted->get_max_sh_degree(), 1);
        source->means_raw().fill_(99.0f);
        EXPECT_EQ(extracted->means_raw().to_vector(), (std::vector<float>{1, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0}));
        EXPECT_EQ(extracted->shN_canonical().to_vector(), std::vector<float>(4 * 9, 0.25f));
    }

    TEST_P(SceneSnapshotBackend, LowMemoryRefusalLeavesLiveDataUsable) {
        Scene scene;
        const auto id = scene.addSplat("source", model());
        const auto before = scene.getNodeById(id)->model->means_raw().to_vector();
        MemoryPressureCoordinator::instance().set_free_memory_probe([](MemoryDomain) { return 0; });
        EXPECT_THROW((void)scene.snapshotVisibleSplats(), std::runtime_error);
        EXPECT_EQ(scene.getNodeById(id)->model->means_raw().to_vector(), before);
        MemoryPressureCoordinator::instance().set_free_memory_probe(nullptr);
        EXPECT_NO_THROW((void)scene.snapshotVisibleSplats());
    }

    TEST_P(SceneSnapshotBackend, PublishesLazyPayloadOnTheWorkerBackend) {
        lfs::test::licht::TemporaryDirectory temporary;
        GalleryScenePublishRequest request;
        request.path = temporary.path / "prepared";
        request.published_render = lfs::vis::project::renderSettingsToProjectJson(lfs::vis::RenderSettings{});
        request.published_camera = lfs::vis::project::panelCameraProjectStateToJson(
            "primary", lfs::vis::project::PanelCameraProjectState{});
        request.nodes.push_back({.name = "lazy payload",
                                 .load_payload = [](TensorCompletion& completion) {
                                     completion.include_current_gpu();
                                     return std::shared_ptr<SplatData>(model());
                                 },
                                 .metadata_known = false});
        writeGalleryScenePublication(request, {}, {});
        EXPECT_TRUE(request.materialized_payload);
        EXPECT_TRUE(std::filesystem::is_regular_file(request.path / "manifest.json"));
        auto loader = lfs::io::Loader::create();
        const auto loaded = loader->load(request.path / "0.ply", {});
        ASSERT_TRUE(loaded) << loaded.error().message;
        const auto* data = std::get_if<std::shared_ptr<SplatData>>(&loaded->data);
        ASSERT_NE(data, nullptr);
        ASSERT_NE(*data, nullptr);
        EXPECT_EQ(gpu_backend_of((*data)->means_raw()), GetParam());
        EXPECT_EQ((*data)->means_raw().to_vector(),
                  (std::vector<float>{0, 0, 0, 1, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0}));
    }

    INSTANTIATE_TEST_SUITE_P(Backends, SceneSnapshotBackend, ::testing::ValuesIn(kGpuBackends),
                             [](const auto& info) { return gpu_backend_name(info.param); });

} // namespace
