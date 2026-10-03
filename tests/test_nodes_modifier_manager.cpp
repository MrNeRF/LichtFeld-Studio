/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nodes/nodes.hpp"
#include "core/tensor_backend.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/nodes/modifier_manager.hpp"
#include "visualizer/operation/undo_history.hpp"

#include <future>
#include <gtest/gtest.h>
#include <thread>

namespace {
    std::unique_ptr<lfs::core::SplatData> model(const std::size_t count = 1,
                                                const lfs::core::Device device = lfs::core::Device::GPU) {
        using lfs::core::Device;
        using lfs::core::Tensor;
        const auto on_device = [device](Tensor value) {
            return device == Device::CPU ? value : value.to(device);
        };
        return std::make_unique<lfs::core::SplatData>(
            1,
            Tensor::zeros({count, 3}, device),
            on_device(Tensor::zeros({count, 1, 3}, Device::CPU)),
            Tensor::zeros({count, 3, 3}, device),
            Tensor::zeros({count, 3}, device),
            on_device(Tensor::cat({Tensor::ones({count, 1}, Device::CPU),
                                   Tensor::zeros({count, 3}, Device::CPU)},
                                  1)),
            Tensor::zeros({count, 1}, device), 1.0f);
    }

    std::vector<std::pair<const char*, std::optional<lfs::core::GpuBackend>>> worker_targets() {
        using lfs::core::GpuBackend;
        return {{"Metal", GpuBackend::Metal},
                {"Vulkan", GpuBackend::Vulkan}};
    }

    template <typename Run>
    void for_each_worker_target(Run run) {
        for (const auto& [name, backend] : worker_targets()) {
            SCOPED_TRACE(name);
            if (backend && !lfs::core::gpu_backend_available(*backend))
                continue;
            std::optional<lfs::core::GpuBackendScope> scope;
            if (backend)
                scope.emplace(*backend);
            run(lfs::core::Device::GPU);
        }
    }

    lfs::core::Tensor selection(std::initializer_list<bool> values) {
        return lfs::core::Tensor::from_vector(std::vector<bool>(values), {values.size()},
                                              lfs::core::Device::CPU);
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

TEST_F(NodesModifierManager, ObjectInfoReportsNamedDependencyCycleAtTheNode) {
    using namespace lfs::nodes;
    if (!lfs::core::gpu_backend_available(lfs::core::GpuBackend::Metal))
        GTEST_SKIP() << "Metal backend unavailable";
    lfs::core::GpuBackendScope scope(lfs::core::GpuBackend::Metal);
    lfs::vis::SceneManager scene;
    scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
    const auto garden_id = scene.getScene().addSplat("garden", model());
    const auto torus_id = scene.getScene().addSplat("torus", model());
    const auto garden = scene.getScene().getNodeUuid(garden_id);
    const auto torus = scene.getScene().getNodeUuid(torus_id);
    auto& manager = scene.modifierManager();
    const auto make_reference = [&](std::string name, std::string target) -> NodeTree& {
        auto& tree = manager.newTree(std::move(name));
        tree.add_node("lfs.object_info", "Object Info").properties["object"] = std::move(target);
        EXPECT_TRUE(tree.remove_link(
            {tree.input_node().name, "Geometry", tree.output_node().name, "Geometry"}));
        EXPECT_TRUE(tree.add_link({"Object Info", "Geometry", tree.output_node().name, "Geometry"}));
        return tree;
    };
    auto& garden_tree = make_reference("Garden reference", "torus");
    auto& torus_tree = make_reference("Torus reference", "garden");
    manager.addModifier(garden, garden_tree.uuid, "Garden modifier");
    manager.addModifier(torus, torus_tree.uuid, "Torus modifier");

    const auto result = manager.evaluate(garden);
    EXPECT_FALSE(result.ok);
    ASSERT_TRUE(result.errors.contains("Garden modifier/Object Info"));
    EXPECT_EQ(result.errors.at("Garden modifier/Object Info"),
              "Dependency cycle: garden → torus → garden");
}

TEST_F(NodesModifierManager, HidingModifierImmediatelyAllowsStoredSelectionCapture) {
    using namespace lfs::nodes;
    using lfs::core::Device;
    using lfs::core::Tensor;
    if (!lfs::core::gpu_backend_available(lfs::core::GpuBackend::Metal))
        GTEST_SKIP() << "Metal backend unavailable";
    lfs::core::GpuBackendScope scope(lfs::core::GpuBackend::Metal);
    lfs::vis::SceneManager scene;
    scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
    const auto id = scene.getScene().addSplat("Host", model());
    const auto host = scene.getScene().getNodeUuid(id);
    auto& manager = scene.modifierManager();
    auto& tree = manager.newTree("Stored selection");
    tree.add_node("lfs.stored_selection", "Stored");
    tree.add_node("lfs.delete_geometry", "Delete");
    ASSERT_TRUE(tree.remove_link(
        {tree.input_node().name, "Geometry", tree.output_node().name, "Geometry"}));
    ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Delete", "Geometry"}));
    ASSERT_TRUE(tree.add_link({"Delete", "Geometry", tree.output_node().name, "Geometry"}));
    auto& modifier = manager.addModifier(host, tree.uuid, "Delete");
    ASSERT_TRUE(manager.evaluate(host).ok);
    ASSERT_NE(scene.getScene().getNodeById(id)->evaluated_model, nullptr);
    EXPECT_EQ(scene.getScene().getNodeById(id)->evaluated_model->size(), 0);
    EXPECT_FALSE(manager.captureSelection(host, "Delete", "Stored"));

    const nlohmann::json before = manager.stack(host)->modifiers;
    modifier.show_viewport = false;
    manager.recordStackEdit(host, before);
    EXPECT_EQ(scene.getScene().getNodeById(id)->evaluated_model, nullptr);
    scene.getScene().setSelectionMask(std::make_shared<Tensor>(
        Tensor::from_vector(std::vector<bool>{true}, {1}, Device::CPU).to(Device::GPU)));
    const auto captured = manager.captureSelection(host, "Delete", "Stored");
    ASSERT_TRUE(captured) << captured.error().message;
    ASSERT_TRUE(modifier.stored_selections.contains("Stored"));
    EXPECT_EQ(modifier.stored_selections.at("Stored").at("size"), 1);
}

TEST_F(NodesModifierManager, StoredSelectionDeleteUsesConsumerInputGeometry) {
    using namespace lfs::nodes;
    for_each_worker_target([](const lfs::core::Device device) {
        lfs::vis::SceneManager scene;
        scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        const auto id = scene.getScene().addSplat("Host", model(6, device));
        const auto host = scene.getScene().getNodeUuid(id);
        auto& manager = scene.modifierManager();
        auto& tree = manager.newTree("Stored delete");
        auto& stored = tree.add_node("lfs.stored_selection", "Stored");
        set_stored_selection(stored, selection({true, false, true, false, false, false}));
        const auto stored_properties = stored.properties;
        tree.add_node("lfs.delete_geometry", "Delete");
        ASSERT_TRUE(tree.remove_link(
            {tree.input_node().name, "Geometry", tree.output_node().name, "Geometry"}));
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Delete", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Stored", "Selection", "Delete", "Selection"}));
        ASSERT_TRUE(tree.add_link({"Delete", "Geometry", tree.output_node().name, "Geometry"}));
        auto& modifier = manager.addModifier(host, tree.uuid, "Delete");
        modifier.stored_selections["Stored"] = stored_properties;

        const auto result = manager.evaluate(host);
        ASSERT_TRUE(result.ok) << (result.errors.empty() ? "no error text"
                                                         : result.errors.begin()->second);
        ASSERT_TRUE(result.geometry.splats);
        EXPECT_EQ(result.geometry.splats->means.shape()[0], 4u);
        const auto status = result.nodes.find(modifier.uuid + "/Delete");
        ASSERT_NE(status, result.nodes.end());
        ASSERT_TRUE(status->second.selected_share);
        EXPECT_NEAR(*status->second.selected_share, 2.0 / 6.0, 1e-6);
        EXPECT_FALSE(manager.selectionPreview(host, modifier.uuid, "Stored"));
        EXPECT_FALSE(manager.selectionPreview(host, modifier.uuid, "Delete"));
    });
}

TEST_F(NodesModifierManager, StoredSelectionSeparateUsesConsumerInputGeometry) {
    using namespace lfs::nodes;
    for_each_worker_target([](const lfs::core::Device device) {
        lfs::vis::SceneManager scene;
        scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        const auto id = scene.getScene().addSplat("Host", model(6, device));
        const auto host = scene.getScene().getNodeUuid(id);
        auto& manager = scene.modifierManager();
        auto& tree = manager.newTree("Stored separate");
        auto& stored = tree.add_node("lfs.stored_selection", "Stored");
        set_stored_selection(stored, selection({true, false, true, false, false, false}));
        const auto stored_properties = stored.properties;
        tree.add_node("lfs.separate_geometry", "Separate");
        ASSERT_TRUE(tree.remove_link(
            {tree.input_node().name, "Geometry", tree.output_node().name, "Geometry"}));
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Separate", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Stored", "Selection", "Separate", "Selection"}));
        ASSERT_TRUE(tree.add_link({"Separate", "Selection", tree.output_node().name, "Geometry"}));
        auto& modifier = manager.addModifier(host, tree.uuid, "Separate");
        modifier.stored_selections["Stored"] = stored_properties;

        const auto result = manager.evaluate(host);
        ASSERT_TRUE(result.ok) << (result.errors.empty() ? "no error text"
                                                         : result.errors.begin()->second);
        ASSERT_TRUE(result.geometry.splats);
        EXPECT_EQ(result.geometry.splats->means.shape()[0], 2u);
        const auto status = result.nodes.find(modifier.uuid + "/Separate");
        ASSERT_NE(status, result.nodes.end());
        ASSERT_TRUE(status->second.selected_share);
        EXPECT_NEAR(*status->second.selected_share, 2.0 / 6.0, 1e-6);
        EXPECT_FALSE(manager.selectionPreview(host, modifier.uuid, "Stored"));
    });
}

TEST_F(NodesModifierManager, IndexAndRandomDeleteStatsUseConsumerInputGeometry) {
    using namespace lfs::nodes;
    for_each_worker_target([](const lfs::core::Device device) {
        for (const std::string source_type : {"lfs.index", "lfs.random_value"}) {
            SCOPED_TRACE(source_type);
            lfs::vis::SceneManager scene;
            scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
            constexpr std::size_t input_count = 32;
            const auto id = scene.getScene().addSplat("Host", model(input_count, device));
            const auto host = scene.getScene().getNodeUuid(id);
            auto& manager = scene.modifierManager();
            auto& tree = manager.newTree("Field delete");
            tree.add_node(source_type, "Source");
            auto& compare = tree.add_node("lfs.compare", "Compare");
            compare.properties["operation"] = "less_than";
            compare.input_values["B"] = source_type == "lfs.index" ? 11.0f : 0.5f;
            tree.add_node("lfs.delete_geometry", "Delete");
            ASSERT_TRUE(tree.remove_link(
                {tree.input_node().name, "Geometry", tree.output_node().name, "Geometry"}));
            ASSERT_TRUE(tree.add_link({"Source",
                                       source_type == "lfs.index" ? "Index" : "Value",
                                       "Compare", "A"}));
            ASSERT_TRUE(tree.add_link({"Compare", "Result", "Delete", "Selection"}));
            ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Delete", "Geometry"}));
            ASSERT_TRUE(tree.add_link({"Delete", "Geometry", tree.output_node().name, "Geometry"}));
            const auto& modifier = manager.addModifier(host, tree.uuid, "Delete");

            const auto result = manager.evaluate(host);
            ASSERT_TRUE(result.ok) << (result.errors.empty() ? "no error text"
                                                             : result.errors.begin()->second);
            ASSERT_TRUE(result.geometry.splats);
            const auto output_count = result.geometry.splats->means.shape()[0];
            const auto status = result.nodes.find(modifier.uuid + "/Delete");
            ASSERT_NE(status, result.nodes.end());
            ASSERT_TRUE(status->second.selected_share);
            EXPECT_NEAR(*status->second.selected_share,
                        static_cast<double>(input_count - output_count) / input_count, 1e-6);
        }
    });
}

TEST_F(NodesModifierManager, PreviewFieldErrorIsIgnoredButEarlierModifierMismatchStillFails) {
    using namespace lfs::nodes;
    for_each_worker_target([](const lfs::core::Device device) {
        {
            lfs::vis::SceneManager scene;
            scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
            const auto id = scene.getScene().addSplat("Host", model(6, device));
            const auto host = scene.getScene().getNodeUuid(id);
            auto& manager = scene.modifierManager();
            auto& tree = manager.newTree("Muted preview error");
            auto& stored = tree.add_node("lfs.stored_selection", "Stored");
            set_stored_selection(stored, selection({true, false, true, false, false}));
            const auto stored_properties = stored.properties;
            auto& writer = tree.add_node("lfs.set_position", "Writer");
            writer.muted = true;
            ASSERT_TRUE(tree.remove_link(
                {tree.input_node().name, "Geometry", tree.output_node().name, "Geometry"}));
            ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Writer", "Geometry"}));
            ASSERT_TRUE(tree.add_link({"Stored", "Selection", "Writer", "Selection"}));
            ASSERT_TRUE(tree.add_link({"Writer", "Geometry", tree.output_node().name, "Geometry"}));
            auto& modifier = manager.addModifier(host, tree.uuid, "Muted");
            modifier.stored_selections["Stored"] = stored_properties;

            const auto result = manager.evaluate(host);
            ASSERT_TRUE(result.ok) << (result.errors.empty() ? "no error text"
                                                             : result.errors.begin()->second);
            EXPECT_FALSE(manager.selectionPreview(host, modifier.uuid, "Stored"));
            const auto status = result.nodes.find(modifier.uuid + "/Writer");
            ASSERT_NE(status, result.nodes.end());
            EXPECT_FALSE(status->second.selected_share);
        }

        {
            lfs::vis::SceneManager scene;
            scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
            const auto id = scene.getScene().addSplat("Host", model(6, device));
            const auto host = scene.getScene().getNodeUuid(id);
            auto& manager = scene.modifierManager();
            auto& first = manager.newTree("First delete");
            auto& first_stored = first.add_node("lfs.stored_selection", "First Stored");
            set_stored_selection(first_stored,
                                 selection({true, false, true, false, false, false}));
            const auto first_stored_properties = first_stored.properties;
            first.add_node("lfs.delete_geometry", "First Delete");
            ASSERT_TRUE(first.remove_link(
                {first.input_node().name, "Geometry", first.output_node().name, "Geometry"}));
            ASSERT_TRUE(first.add_link({first.input_node().name, "Geometry", "First Delete", "Geometry"}));
            ASSERT_TRUE(first.add_link({"First Stored", "Selection", "First Delete", "Selection"}));
            ASSERT_TRUE(first.add_link({"First Delete", "Geometry", first.output_node().name, "Geometry"}));

            auto& second = manager.newTree("Second delete");
            auto& second_stored = second.add_node("lfs.stored_selection", "Stored");
            set_stored_selection(second_stored,
                                 selection({true, false, true, false, false, false}));
            const auto second_stored_properties = second_stored.properties;
            second.add_node("lfs.delete_geometry", "Delete");
            ASSERT_TRUE(second.remove_link(
                {second.input_node().name, "Geometry", second.output_node().name, "Geometry"}));
            ASSERT_TRUE(second.add_link({second.input_node().name, "Geometry", "Delete", "Geometry"}));
            ASSERT_TRUE(second.add_link({"Stored", "Selection", "Delete", "Selection"}));
            ASSERT_TRUE(second.add_link({"Delete", "Geometry", second.output_node().name, "Geometry"}));
            auto& first_modifier = manager.addModifier(host, first.uuid, "First");
            first_modifier.stored_selections["First Stored"] = first_stored_properties;
            auto& second_modifier = manager.addModifier(host, second.uuid, "Second");
            second_modifier.stored_selections["Stored"] = second_stored_properties;

            const auto result = manager.evaluate(host);
            EXPECT_FALSE(result.ok);
            ASSERT_TRUE(result.errors.contains("Second/Stored"));
            EXPECT_NE(result.errors.at("Second/Stored").find("captured on 6 splats but receives 4"),
                      std::string::npos);
        }
    });
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
