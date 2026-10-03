/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nodes/nodes.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/nodes/modifier_manager.hpp"
#include "visualizer/operation/undo_history.hpp"

#include <future>
#include <gtest/gtest.h>
#include <thread>

namespace {
    std::unique_ptr<lfs::core::SplatData> model() {
        using lfs::core::Device;
        using lfs::core::Tensor;
        const auto device = Device::GPU;
        return std::make_unique<lfs::core::SplatData>(
            1,
            Tensor::from_vector({0.0f, 0.0f, 0.0f}, {1, 3}, Device::CPU).to(device),
            Tensor::from_vector({0.1f, 0.2f, 0.3f}, {1, 1, 3}, Device::CPU).to(device),
            Tensor::zeros({1, 3, 3}, device),
            Tensor::zeros({1, 3}, device),
            Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f}, {1, 4}, Device::CPU).to(device),
            Tensor::zeros({1, 1}, device), 1.0f);
    }

    lfs::nodes::NodeTree& colour_tree(lfs::vis::ModifierManager& manager) {
        auto& tree = manager.newTree("Colour");
        const auto input_name = tree.nodes[0].name;
        const auto output_name = tree.nodes[1].name;
        auto& correct = tree.add_node("lfs.colour_correct", "Correct");
        correct.input_values["Exposure"] = 1.0f;
        EXPECT_TRUE(tree.remove_link({input_name, "Geometry", output_name, "Geometry"}));
        EXPECT_TRUE(tree.add_link({input_name, "Geometry", correct.name, "Geometry"}));
        EXPECT_TRUE(tree.add_link({correct.name, "Geometry", output_name, "Geometry"}));
        return tree;
    }
} // namespace

class NodesModifierManager : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::vis::op::undoHistory().clear();
    }

    void TearDown() override {
        lfs::vis::op::undoHistory().clear();
    }
};

TEST_F(NodesModifierManager, StackOrderEvaluationAndJsonRoundTrip) {
    lfs::vis::SceneManager scene_manager;
    scene_manager.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
    const auto id = scene_manager.getScene().addSplat("Host", model());
    ASSERT_NE(id, lfs::core::NULL_NODE);
    auto& manager = scene_manager.modifierManager();
    auto& tree = colour_tree(manager);
    const auto uuid = scene_manager.getScene().getNodeUuid(id);
    manager.addModifier(uuid, tree.uuid, "First");
    manager.addModifier(uuid, tree.uuid, "Second");

    const auto result = manager.evaluate(uuid);
    ASSERT_TRUE(result.ok);
    ASSERT_TRUE(result.geometry.splats);
    EXPECT_NE(result.geometry.splats->sh0.to_vector(),
              scene_manager.getScene().getNodeById(id)->model->sh0_raw().to_vector());
    ASSERT_NE(manager.stack(uuid), nullptr);
    ASSERT_EQ(manager.stack(uuid)->modifiers.size(), 2u);
    EXPECT_EQ(manager.stack(uuid)->modifiers[0].name, "First");
    EXPECT_EQ(manager.stack(uuid)->modifiers[1].name, "Second");

    const auto saved = manager.toJson(false);
    ASSERT_TRUE(manager.restoreJson(saved));
    EXPECT_EQ(manager.toJson(false), saved);
}

TEST_F(NodesModifierManager, ObjectInfoUploadsCpuMeshBeforeTransformAndJoin) {
    using namespace lfs::nodes;
    using lfs::core::Device;
    using lfs::core::Tensor;
    lfs::vis::SceneManager scene;
    scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
    const auto host_id = scene.getScene().addSplat("Host", model());
    const auto host = scene.getScene().getNodeUuid(host_id);
    auto mesh = std::make_shared<lfs::core::MeshData>();
    mesh->vertices = Tensor::from_vector({0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, {3, 3}, Device::CPU);
    mesh->indices = Tensor::from_vector({0, 1, 2}, {1, 3}, Device::CPU);
    scene.getScene().addMesh("Reference", mesh);
    auto& manager = scene.modifierManager();
    NodeTypeInfo check;
    check.id = "test.gpu_mesh";
    check.inputs = {{"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)}};
    check.outputs = check.inputs;
    check.evaluate = [](NodeContext& context) {
        const auto* geometry = context.input("Geometry").get_if<Geometry>();
        ASSERT_NE(geometry, nullptr);
        ASSERT_TRUE(geometry->mesh);
        EXPECT_EQ(geometry->mesh->mesh->vertices.device(), Device::GPU);
        EXPECT_EQ(geometry->mesh->mesh->indices.device(), Device::GPU);
        context.set_output("Geometry", *geometry);
    };
    manager.registry().register_type(std::move(check));
    auto& tree = manager.newTree("CPU mesh join");
    tree.add_node("lfs.object_info", "Reference").properties["object"] = "Reference";
    tree.add_node("lfs.transform_geometry", "Transform");
    tree.add_node("test.gpu_mesh", "Check");
    tree.add_node("lfs.mesh_to_splats", "Sample").input_values["Max Count"] = int64_t(12);
    tree.add_node("lfs.join_geometry", "Join");
    ASSERT_TRUE(tree.add_link({"Reference", "Geometry", "Transform", "Geometry"}));
    ASSERT_TRUE(tree.add_link({"Transform", "Geometry", "Check", "Geometry"}));
    ASSERT_TRUE(tree.add_link({"Check", "Geometry", "Sample", "Geometry"}));
    ASSERT_TRUE(tree.add_link({"Sample", "Geometry", "Join", "Geometry"}));
    ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Join", "Geometry"}));
    ASSERT_TRUE(tree.add_link({"Join", "Geometry", tree.output_node().name, "Geometry"}));
    manager.addModifier(host, tree.uuid);
    const auto result = manager.evaluate(host);
    ASSERT_TRUE(result.ok) << result.errors.size();
    ASSERT_TRUE(result.geometry.splats);
    EXPECT_EQ(result.geometry.splats->means.device(), Device::GPU);
    EXPECT_EQ(result.geometry.splats->means.shape()[0], 13);
    EXPECT_EQ(mesh->vertices.device(), Device::CPU);
}

TEST_F(NodesModifierManager, ScriptedInputBurstHasOneUndoAndOneQueuedEvaluation) {
    lfs::vis::SceneManager scene;
    scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
    const auto id = scene.getScene().addSplat("Host", model());
    const auto host = scene.getScene().getNodeUuid(id);
    auto& manager = scene.modifierManager();
    auto& tree = colour_tree(manager);
    const auto uuid = tree.uuid;
    manager.addModifier(host, uuid);
    ASSERT_TRUE(manager.evaluate(host).ok);
    (void)manager.performance(true);
    lfs::vis::op::undoHistory().clear();
    for (int i = 0; i < 120; ++i)
        ASSERT_TRUE(manager.setNodeInput(uuid, "Correct", "Exposure", float(i) / 120.0f));
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
    EXPECT_EQ(manager.performance()["requests"], 0);
    ASSERT_TRUE(manager.evaluate(host).ok);
    EXPECT_EQ(manager.performance()["requests"], 1);
    ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
    EXPECT_EQ(*manager.tree(uuid)->find_node("Correct")->input_values.at("Exposure").get_if<float>(), 1.0f);
}

TEST_F(NodesModifierManager, ReplacingPublishedPayloadReleasesStorageOffViewer) {
    auto released = std::make_shared<std::promise<std::thread::id>>();
    auto released_on = released->get_future();
    lfs::vis::SceneManager scene;
    scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
    const auto id = scene.getScene().addSplat("Host", model());
    auto payload = std::shared_ptr<lfs::core::SplatData>(model().release(), [released](auto* value) {
        delete value;
        released->set_value(std::this_thread::get_id());
    });
    scene.getScene().setNodeEvaluatedPayload(id, std::move(payload), {}, {});
    auto& manager = scene.modifierManager();
    auto& tree = colour_tree(manager);
    const auto host = scene.getScene().getNodeUuid(id);
    manager.addModifier(host, tree.uuid);
    ASSERT_TRUE(manager.evaluate(host).ok);
    ASSERT_EQ(released_on.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_NE(released_on.get(), std::this_thread::get_id());
}

TEST_F(NodesModifierManager, DeletingGraphRemovesItsInstancesAndUndoRestoresBoth) {
    lfs::vis::SceneManager scene;
    scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
    const auto id = scene.getScene().addSplat("Host", model());
    const auto host = scene.getScene().getNodeUuid(id);
    auto& manager = scene.modifierManager();
    const auto deleted = manager.newTree("Deleted").uuid;
    const auto retained = manager.newTree("Retained").uuid;
    manager.addModifier(host, deleted);
    manager.addModifier(host, retained);
    lfs::vis::op::undoHistory().clear();
    ASSERT_TRUE(manager.removeTree(deleted));
    ASSERT_EQ(manager.stack(host)->modifiers.size(), 1u);
    EXPECT_EQ(manager.stack(host)->modifiers.front().tree_uuid, retained);
    ASSERT_TRUE(manager.evaluate(host).ok);
    ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
    EXPECT_NE(manager.tree(deleted), nullptr);
    EXPECT_EQ(manager.stack(host)->modifiers.size(), 2u);
}

TEST_F(NodesModifierManager, ApplyBakesOnceAndUndoRestoresPayloadAndStack) {
    lfs::vis::SceneManager scene_manager;
    scene_manager.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
    const auto id = scene_manager.getScene().addSplat("Host", model());
    auto& manager = scene_manager.modifierManager();
    auto& tree = colour_tree(manager);
    const auto uuid = scene_manager.getScene().getNodeUuid(id);
    manager.addModifier(uuid, tree.uuid, "Correct");
    const auto before = scene_manager.getScene().getNodeById(id)->model->sh0_raw().to_vector();
    lfs::vis::op::undoHistory().clear();

    ASSERT_TRUE(manager.applyModifier(uuid, "Correct"));
    EXPECT_TRUE(manager.stack(uuid)->modifiers.empty());
    EXPECT_NE(scene_manager.getScene().getNodeById(id)->model->sh0_raw().to_vector(), before);
    ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
    ASSERT_NE(manager.stack(uuid), nullptr);
    EXPECT_EQ(manager.stack(uuid)->modifiers.size(), 1u);
    EXPECT_EQ(scene_manager.getScene().getNodeById(id)->model->sh0_raw().to_vector(), before);
}

TEST_F(NodesModifierManager, LayoutUndoAndRepeatedReadsNeverEvaluate) {
    lfs::vis::SceneManager scene;
    scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
    const auto id = scene.getScene().addSplat("Host", model());
    const auto uuid = scene.getScene().getNodeUuid(id);
    auto& manager = scene.modifierManager();
    auto& tree = colour_tree(manager);
    manager.addModifier(uuid, tree.uuid);
    ASSERT_TRUE(manager.evaluate(uuid).ok);
    (void)manager.performance(true);
    lfs::vis::op::undoHistory().clear();
    const auto before = tree.to_json();
    tree.find_node("Correct")->location = {25.0f, 70.0f};
    manager.recordTreeEdit(tree.uuid, before);
    manager.tick();
    EXPECT_EQ(manager.performance()["requests"], 0);
    EXPECT_EQ(manager.performance()["evaluations"], 0);
    ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
    manager.tick();
    ASSERT_TRUE(lfs::vis::op::undoHistory().redo().success);
    manager.tick();
    (void)manager.evaluate(uuid);
    EXPECT_EQ(manager.performance()["evaluations"], 0);
}

TEST_F(NodesModifierManager, ValueChangeRequestsOnceAndKeepsUpstreamCached) {
    lfs::vis::SceneManager scene;
    scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
    const auto id = scene.getScene().addSplat("Host", model());
    const auto uuid = scene.getScene().getNodeUuid(id);
    auto& manager = scene.modifierManager();
    auto& tree = colour_tree(manager);
    tree.add_node("lfs.hsv_range", "HSV");
    ASSERT_TRUE(tree.add_link({"HSV", "Selection", "Correct", "Selection"}));
    manager.addModifier(uuid, tree.uuid);
    ASSERT_TRUE(manager.evaluate(uuid).ok);
    (void)manager.performance(true);
    const auto before = tree.to_json();
    tree.find_node("Correct")->input_values["Exposure"] = 0.5f;
    manager.recordTreeEdit(tree.uuid, before);
    ASSERT_TRUE(manager.evaluate(uuid).ok);
    const auto performance = manager.performance();
    EXPECT_EQ(performance["requests"], 1);
    EXPECT_EQ(performance["evaluations"], 1);
    EXPECT_EQ(performance["installed"], 1);
    EXPECT_EQ(performance["node_runs"]["Correct"], 1);
    EXPECT_FALSE(performance["node_runs"].contains("HSV"));
    EXPECT_FALSE(performance["node_runs"].contains(tree.input_node().name));
}

TEST_F(NodesModifierManager, CachedRequestReusesPublishedPayloadWithoutSharingWorkerStorage) {
    lfs::vis::SceneManager scene;
    scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
    const auto id = scene.getScene().addSplat("Host", model());
    const auto uuid = scene.getScene().getNodeUuid(id);
    auto& manager = scene.modifierManager();
    auto& tree = colour_tree(manager);
    manager.addModifier(uuid, tree.uuid);
    const auto result = manager.evaluate(uuid);
    ASSERT_TRUE(result.ok);
    const auto* node = scene.getScene().getNodeById(id);
    const auto payload = node->evaluated_model;
    ASSERT_NE(payload, nullptr);
    EXPECT_NE(payload->means_raw().debug_id(), result.geometry.splats->means.debug_id());
    EXPECT_NE(node->model->means_raw().debug_id(), result.geometry.splats->means.debug_id());
    (void)manager.performance(true);
    manager.markDirty(uuid);
    const auto cached = manager.evaluate(uuid);
    EXPECT_TRUE(cached.ok);
    EXPECT_TRUE(cached.unchanged);
    EXPECT_EQ(node->evaluated_model, payload);
    EXPECT_TRUE(manager.performance()["node_runs"].empty());
}

TEST_F(NodesModifierManager, WorkerDiscardsSupersededResultsAndInstallsOnViewer) {
    lfs::vis::SceneManager scene;
    scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
    const auto id = scene.getScene().addSplat("Host", model());
    const auto uuid = scene.getScene().getNodeUuid(id);
    auto& manager = scene.modifierManager();
    std::promise<void> started;
    std::promise<void> release;
    auto released = release.get_future().share();
    std::thread::id evaluation_thread;
    int calls = 0;
    lfs::nodes::NodeTypeInfo type;
    type.id = "test.wait";
    type.label = "Wait";
    type.category = "Test";
    type.inputs = {{"Geometry", "Geometry", std::string(lfs::nodes::GEOMETRY_SOCKET)}};
    type.outputs = type.inputs;
    type.evaluate = [&](lfs::nodes::NodeContext& context) {
        evaluation_thread = std::this_thread::get_id();
        if (++calls == 1) {
            started.set_value();
            released.wait();
        }
        context.set_output("Geometry", context.input("Geometry"));
    };
    manager.registry().register_type(std::move(type));
    auto& tree = manager.newTree("Worker");
    const auto input = tree.input_node().name;
    const auto output = tree.output_node().name;
    tree.add_node("test.wait", "Wait");
    tree.remove_link({input, "Geometry", output, "Geometry"});
    ASSERT_TRUE(tree.add_link({input, "Geometry", "Wait", "Geometry"}));
    ASSERT_TRUE(tree.add_link({"Wait", "Geometry", output, "Geometry"}));
    manager.addModifier(uuid, tree.uuid);
    manager.tick();
    auto began = started.get_future();
    const auto status = began.wait_for(std::chrono::seconds(10));
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_FALSE(scene.getScene().hasEvaluatedPayload(id));
    auto before = tree.to_json();
    tree.find_node("Wait")->muted = true;
    manager.recordTreeEdit(tree.uuid, before);
    auto mesh = std::make_shared<lfs::core::MeshData>();
    mesh->vertices = lfs::core::Tensor::zeros({3, 3}, lfs::core::Device::CPU);
    mesh->indices = lfs::core::Tensor::zeros({1, 3}, lfs::core::Device::CPU, lfs::core::DataType::Int32);
    const auto mesh_id = scene.getScene().addMesh("Mesh", mesh);
    auto& mesh_tree = manager.newTree("Mesh Graph");
    manager.addModifier(scene.getScene().getNodeUuid(mesh_id), mesh_tree.uuid);
    manager.tick();
    // Replacing live mesh metadata after capture must not change the queued snapshot.
    mesh->vertices = lfs::core::Tensor::zeros({6, 3}, lfs::core::Device::CPU);
    release.set_value();
    const auto result = manager.evaluate(uuid);
    ASSERT_TRUE(result.ok);
    EXPECT_NE(evaluation_thread, std::this_thread::get_id());
    EXPECT_TRUE(scene.getScene().hasEvaluatedPayload(id));
    const auto* mesh_node = scene.getScene().getNodeById(mesh_id);
    ASSERT_NE(mesh_node->evaluated_mesh, nullptr);
    EXPECT_EQ(mesh_node->evaluated_mesh->vertex_count(), 3);
    EXPECT_EQ(manager.performance()["discarded"], 1);
    EXPECT_EQ(manager.performance()["installed"], 1);
}
