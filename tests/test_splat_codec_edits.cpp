/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "core/sh_value_quant.hpp"
#include "core/tensor_backend.hpp"
#include "operation/undo_history.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/gui_capabilities.hpp"
#include <gtest/gtest.h>
#include <optional>
namespace {
    using namespace lfs::core;
    class SplatCodecEdits : public testing::TestWithParam<GpuBackend> {
    protected:
        std::optional<GpuBackendScope> scope;
        void SetUp() override {
            if (!gpu_backend_available(GetParam()))
                GTEST_SKIP();
            scope.emplace(GetParam());
            sh_value_quant::set_enabled_for_testing(false);
            lfs::vis::op::undoHistory().clear();
        }
        void TearDown() override {
            lfs::vis::op::undoHistory().clear();
            lfs::vis::services().clear();
            lfs::core::event::bus().clear_all();
            lfs::event::EventBridge::instance().clear_all();
            sh_value_quant::set_enabled_for_testing(std::nullopt);
        }
        static std::unique_ptr<SplatData> model(size_t n, float offset = 0) {
            std::vector<float> values(n * 9), q(n * 4);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = offset + float(i) / 16;
            for (size_t i = 0; i < n; ++i)
                q[i * 4] = 1;
            return std::make_unique<SplatData>(1, Tensor::zeros({n, 3}, Device::GPU), Tensor::zeros({n, 1, 3}, Device::GPU),
                                               Tensor::from_vector(values, {n, 3, 3}, Device::GPU), Tensor::zeros({n, 3}, Device::GPU),
                                               Tensor::from_vector(q, {n, 4}, Device::GPU), Tensor::zeros({n, 1}, Device::GPU), 1.0f);
        }
    };
    TEST_P(SplatCodecEdits, MergeTwoObjects) {
        auto a = model(33), b = model(257, 100);
        auto expected = Tensor::cat({a->shN_canonical(), b->shN_canonical()}, 0).to_vector();
        auto merged = Scene::mergeSplatsWithTransforms({{a.get(), glm::mat4(1)}, {b.get(), glm::mat4(1)}});
        ASSERT_NE(merged, nullptr);
        EXPECT_EQ(merged->size(), 290u);
        EXPECT_EQ(gpu_backend_of(merged->shN_raw()), GetParam());
        EXPECT_EQ(merged->shN_canonical().to_vector(), expected);
    }
    TEST_P(SplatCodecEdits, DuplicateAndUndo) {
        lfs::vis::SceneManager manager;
        lfs::vis::services().set(&manager);
        auto& scene = manager.getScene();
        auto id = scene.addSplat("original", model(33));
        auto expected = scene.getNodeById(id)->model->shN_canonical().to_vector();
        const auto name = manager.duplicateNodeTree(id);
        ASSERT_FALSE(name.empty());
        ASSERT_NE(scene.getNode(name), nullptr);
        EXPECT_EQ(scene.getNode(name)->model->shN_canonical().to_vector(), expected);
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(scene.getNode(name), nullptr);
        ASSERT_TRUE(lfs::vis::op::undoHistory().redo().success);
        EXPECT_EQ(scene.getNode(name)->model->shN_canonical().to_vector(), expected);
        lfs::vis::op::undoHistory().clear();
        lfs::vis::services().clear();
    }
    TEST_P(SplatCodecEdits, CompactAfterNodeDelete) {
        Scene scene;
        auto first = scene.addSplat("first", model(33));
        auto second = scene.addSplat("second", model(257, 100));
        auto expected = scene.getNodeById(second)->model->shN_canonical().to_vector();
        ASSERT_EQ(scene.consolidateNodeModels(), 2u);
        scene.removeNodeById(first);
        const auto snapshot = scene.captureConsolidatedCompaction();
        ASSERT_TRUE(snapshot);
        std::vector<Scene::ConsolidatedNodeSlot> slots;
        auto compact = Scene::compactConsolidatedSnapshot(*snapshot, slots);
        ASSERT_NE(compact, nullptr);
        EXPECT_EQ(compact->size(), 257u);
        EXPECT_EQ(compact->shN_canonical().to_vector(), expected);
    }
    TEST_P(SplatCodecEdits, ApplyDeletion) {
        auto data = model(257);
        auto expected = data->shN_canonical().slice(0, 1, 257).to_vector();
        std::vector<int> mask(257);
        mask[0] = 1;
        data->soft_delete(Tensor::from_vector(mask, {257}, Device::GPU).to(DataType::Bool));
        EXPECT_EQ(data->apply_deleted(), 1u);
        EXPECT_EQ(data->size(), 256u);
        EXPECT_EQ(data->shN_canonical().to_vector(), expected);
    }
    TEST_P(SplatCodecEdits, ShFieldWriteAndUndo) {
        lfs::vis::SceneManager manager;
        lfs::vis::services().set(&manager);
        manager.getScene().addSplat("model", model(257));
        auto* node = manager.getScene().getMutableNode("model");
        const auto before = node->model->shN_canonical().to_vector();
        std::vector<float> replacement(18, -7);
        auto result = lfs::vis::cap::writeGaussianField(manager, nullptr, "model", "shN", {1, 256}, replacement);
        ASSERT_TRUE(result) << result.error();
        auto expected = before;
        std::fill(expected.begin() + 9, expected.begin() + 18, -7);
        std::fill(expected.begin() + 256 * 9, expected.end(), -7);
        EXPECT_EQ(node->model->shN_canonical().to_vector(), expected);
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(node->model->shN_canonical().to_vector(), before);
        ASSERT_TRUE(lfs::vis::op::undoHistory().redo().success);
        EXPECT_EQ(node->model->shN_canonical().to_vector(), expected);
        lfs::vis::op::undoHistory().clear();
        lfs::vis::services().clear();
    }
    INSTANTIATE_TEST_SUITE_P(CudaVulkan, SplatCodecEdits, testing::ValuesIn(kGpuBackends),
                             [](const testing::TestParamInfo<GpuBackend>& p) { return p.param == GpuBackend::CUDA ? "CUDA" : "Vulkan"; });
} // namespace
