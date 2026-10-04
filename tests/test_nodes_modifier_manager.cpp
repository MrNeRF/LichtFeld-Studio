/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/nodes/nodes.hpp"
#include "core/services.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_completion.hpp"
#include "core/tensor_vulkan_interop.hpp"
#include "scene/scene_manager.hpp"
#include "sequencer/interpolation.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/project_snapshot_chapters.hpp"
#include "training/trainer.hpp"
#include "visualizer/core/training_manager.hpp"
#include "visualizer/nodes/modifier_manager.hpp"
#include "visualizer/nodes/node_animation.hpp"
#include "visualizer/nodes/viewport_coordinates.hpp"
#include "visualizer/operation/undo_history.hpp"

#include <cmath>
#include <cstring>
#include <future>
#include <glm/gtc/matrix_transform.hpp>
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
                {"Vulkan", GpuBackend::Vulkan},
                {"CUDA", GpuBackend::CUDA}};
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

    struct ServicesScope {
        ServicesScope() { lfs::vis::services().clear(); }
        ~ServicesScope() { lfs::vis::services().clear(); }
    };

    bool transition(lfs::vis::TrainerManager& manager, const lfs::vis::TrainingState state) {
        return const_cast<lfs::vis::TrainingStateMachine&>(manager.getStateMachine())
            .transitionTo(state);
    }

    void add_training_camera(lfs::core::Scene& scene) {
        const auto group = scene.addGroup("Training cameras");
        auto camera = std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
            100.0f, 100.0f, 32.0f, 32.0f,
            lfs::core::Tensor{}, lfs::core::Tensor{},
            lfs::core::CameraModelType::PINHOLE,
            "camera.png", std::filesystem::path{}, std::filesystem::path{},
            64, 64, 0);
        ASSERT_NE(scene.addCamera("camera.png", group, std::move(camera)),
                  lfs::core::NULL_NODE);
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

TEST_F(NodesModifierManager, AnimatedInputsEasingPersistenceAndNonAnimatedPlayheadCounts) {
    for_each_worker_target([&](const auto device) {
        lfs::vis::SequencerController controller;
        lfs::vis::SceneManager scene;
        scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        auto& manager = scene.modifierManager();
        manager.setSequencer(&controller);
        const auto id = scene.getScene().addSplat("Host", model(3, device));
        const auto host = scene.getScene().getNodeUuid(id);
        auto& tree = colour_tree(manager);
        const auto tree_id = tree.uuid;
        manager.addModifier(host, tree_id);
        ASSERT_TRUE(manager.evaluate(host).ok);
        (void)manager.performance(true);
        for (float time : {0.25f, 0.5f, 1.0f}) {
            controller.seek(time);
            manager.tick();
        }
        EXPECT_EQ(manager.performance()["requests"], 0);
        EXPECT_EQ(manager.performance()["evaluations"], 0);
        ASSERT_TRUE(manager.keyframeSet(tree_id, "Correct", "Exposure", 0, 0.0f, 1));
        ASSERT_TRUE(manager.keyframeSet(tree_id, "Correct", "Exposure", 2, 2.0f));
        const auto path = lfs::vis::nodeInputTrackPath(tree_id, "Correct", "Exposure");
        const auto* track = controller.timeline().animationClip()->getTrackByPath(path);
        ASSERT_NE(track, nullptr);
        EXPECT_EQ(track->keyframeCount(), 2);
        for (float time : {0.0f, 0.5f, 1.0f, 1.75f, 2.0f, 3.0f}) {
            controller.seek(time);
            const auto result = manager.evaluate(host);
            ASSERT_TRUE(result.ok);
            const float t = std::clamp(time / 2, 0.0f, 1.0f);
            const float exposure = 2 * t * t * t;
            const float expected = (0.5f * std::exp2(exposure) - 0.5f) / 0.28209479177387814f;
            EXPECT_NEAR(result.geometry.splats->sh0.cpu().to_vector()[0], expected, 2e-5f) << time;
        }
        // Project NODE contains graphs; the existing sequencer chapter contains tracks.
        const auto graphs = manager.toJson();
        const auto sequence = controller.saveToJson();
        EXPECT_FALSE(graphs.contains("animation"));
        controller.clear();
        ASSERT_TRUE(controller.loadFromJson(nlohmann::json::parse(sequence.dump())));
        ASSERT_TRUE(manager.restoreJson(nlohmann::json::parse(graphs.dump())));
        const auto* restored = controller.timeline().animationClip()->getTrackByPath(path);
        ASSERT_NE(restored, nullptr);
        EXPECT_FLOAT_EQ(std::get<float>(*restored->evaluate(1)), 0.25f);
        EXPECT_EQ(restored->keyframe(0).easing, lfs::sequencer::EasingType::EASE_IN);
        ASSERT_TRUE(manager.keyframeRemove(tree_id, "Correct", "Exposure", 2));
        EXPECT_EQ(controller.timeline().animationClip()->getTrackByPath(path)->keyframeCount(), 1);
        lfs::vis::op::undoHistory().undo();
        EXPECT_EQ(controller.timeline().animationClip()->getTrackByPath(path)->keyframeCount(), 2);
    });
}

TEST_F(NodesModifierManager, AnimationSkipsStaticHostsAndFollowsObjectInfoDependencies) {
    for_each_worker_target([&](const auto device) {
        using namespace lfs;
        std::atomic<int> static_calls = 0;
        vis::SequencerController controller;
        vis::SceneManager scene;
        scene.changeContentType(vis::SceneManager::ContentType::SplatFiles);
        auto& manager = scene.modifierManager();
        manager.setSequencer(&controller);
        nodes::NodeTypeInfo counter;
        counter.id = "test.static_counter";
        counter.label = "Static counter";
        counter.uses_host = true; // Defeat host-generation caching: count actual submissions.
        counter.inputs.push_back({"Geometry", "Geometry", std::string(nodes::GEOMETRY_SOCKET)});
        counter.outputs = counter.inputs;
        counter.evaluate = [&](nodes::NodeContext& context) {
            ++static_calls;
            context.set_output("Geometry", context.input("Geometry"));
        };
        ASSERT_TRUE(manager.registry().register_type(std::move(counter)));
        const auto fixed = scene.getScene().getNodeUuid(scene.getScene().addSplat("Static", model(1, device)));
        const auto animated = scene.getScene().getNodeUuid(scene.getScene().addSplat("Animated", model(1, device)));
        const auto consumer = scene.getScene().getNodeUuid(scene.getScene().addSplat("Consumer", model(1, device)));
        auto& fixed_tree = manager.newTree("Static graph");
        fixed_tree.add_node("test.static_counter", "Counter");
        fixed_tree.add_node("lfs.scene_time", "Disconnected clock");
        ASSERT_TRUE(fixed_tree.add_link({fixed_tree.input_node().name, "Geometry", "Counter", "Geometry"}));
        ASSERT_TRUE(fixed_tree.add_link({"Counter", "Geometry", fixed_tree.output_node().name, "Geometry"}));
        manager.addModifier(fixed, fixed_tree.uuid);
        auto& moving_tree = colour_tree(manager);
        manager.addModifier(animated, moving_tree.uuid);
        ASSERT_TRUE(manager.keyframeSet(moving_tree.uuid, "Correct", "Exposure", 0, 0.0f));
        ASSERT_TRUE(manager.keyframeSet(moving_tree.uuid, "Correct", "Exposure", 2, 2.0f));
        auto& consumer_tree = manager.newTree("Object dependency");
        consumer_tree.add_node("lfs.object_info", "Source").properties["object"] = "Animated";
        ASSERT_TRUE(consumer_tree.add_link({"Source", "Geometry", consumer_tree.output_node().name, "Geometry"}));
        manager.addModifier(consumer, consumer_tree.uuid);
        ASSERT_TRUE(manager.evaluate(fixed).ok);
        const int initial = static_calls;
        EXPECT_GT(initial, 0);
        EXPECT_FALSE(manager.timeDependent(fixed));
        EXPECT_TRUE(manager.timeDependent(consumer));
        for (float time : {0.25f, 0.5f, 1.0f, 1.75f}) {
            controller.seek(time);
            ASSERT_TRUE(manager.evaluate(animated).ok);
            EXPECT_EQ(static_calls.load(), initial);
            EXPECT_EQ(manager.evaluated(animated)->splats->sh0.cpu().to_vector(),
                      manager.evaluated(consumer)->splats->sh0.cpu().to_vector());
        }
        controller.clear();
        ASSERT_TRUE(manager.evaluate(animated).ok);
        EXPECT_FALSE(manager.timeDependent());
        EXPECT_NEAR(manager.evaluated(animated)->splats->sh0.cpu().to_vector()[0], 0.5f / 0.28209479177387814f, 2e-5f);
    });
}

TEST_F(NodesModifierManager, AnimationScrubsCoalesceToLatestWorkerSnapshot) {
    for_each_worker_target([&](const auto device) {
        using namespace lfs;
        vis::SequencerController controller;
        vis::SceneManager scene;
        scene.changeContentType(vis::SceneManager::ContentType::SplatFiles);
        auto& manager = scene.modifierManager();
        manager.setSequencer(&controller);
        const auto id = scene.getScene().addSplat("Host", model(1, device));
        const auto host = scene.getScene().getNodeUuid(id);
        std::promise<void> started, release;
        auto began = started.get_future();
        auto released = release.get_future().share();
        bool first = true;
        nodes::NodeTypeInfo wait;
        wait.id = "test.animation_wait";
        wait.label = "Wait";
        wait.inputs = {{"Geometry", "Geometry", std::string(nodes::GEOMETRY_SOCKET)}};
        wait.outputs = wait.inputs;
        wait.evaluate = [&](nodes::NodeContext& context) {
            if (first) {
                first = false;
                started.set_value();
                released.wait();
            }
            context.set_output("Geometry", context.input("Geometry"));
        };
        ASSERT_TRUE(manager.registry().register_type(std::move(wait)));
        auto& tree = colour_tree(manager);
        tree.add_node("test.animation_wait", "Wait");
        ASSERT_TRUE(tree.add_link({"Correct", "Geometry", "Wait", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Wait", "Geometry", tree.output_node().name, "Geometry"}));
        manager.addModifier(host, tree.uuid);
        ASSERT_TRUE(manager.keyframeSet(tree.uuid, "Correct", "Exposure", 0, 0.0f));
        ASSERT_TRUE(manager.keyframeSet(tree.uuid, "Correct", "Exposure", 2, 2.0f));
        manager.tick();
        EXPECT_EQ(began.wait_for(std::chrono::seconds(10)), std::future_status::ready);
        controller.seek(0.5f);
        manager.tick();
        controller.seek(1.5f);
        manager.tick();
        release.set_value();
        const auto result = manager.evaluate(host);
        ASSERT_TRUE(result.ok);
        EXPECT_NEAR(result.geometry.splats->sh0.cpu().to_vector()[0],
                    (0.5f * std::exp2(1.5f) - 0.5f) / 0.28209479177387814f, 2e-5f);
        EXPECT_GE(manager.performance()["discarded"].get<int>(), 1);
        EXPECT_EQ(manager.performance()["installed"], 1);
    });
}

TEST_F(NodesModifierManager, AnimationPathsRenamePasteGroupUngroupCopyAndUndo) {
    lfs::vis::SequencerController controller;
    lfs::vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    manager.setSequencer(&controller);
    auto& graph = colour_tree(manager);
    const auto uuid = graph.uuid;
    ASSERT_TRUE(manager.keyframeSet(uuid, "Correct", "Exposure", 0, 0.0f));
    ASSERT_TRUE(manager.keyframeSet(uuid, "Correct", "Exposure", 2, 2.0f));
    ASSERT_TRUE(manager.renameNode(uuid, "Correct", "Autumn Grade"));
    auto has = [&](const std::string& tree, const std::string& node) {
        const auto* track = controller.timeline().animationClip()->getTrackByPath(lfs::vis::nodeInputTrackPath(tree, node, "Exposure"));
        return track && track->keyframeCount() == 2 && std::get<float>(*track->evaluate(1)) == 1;
    };
    EXPECT_TRUE(has(uuid, "Autumn Grade"));
    EXPECT_FALSE(has(uuid, "Correct"));
    const auto clipboard = manager.copyNodes(uuid, {"Autumn Grade"});
    ASSERT_TRUE(clipboard);
    const auto paste = manager.pasteNodes(uuid, *clipboard);
    ASSERT_TRUE(paste);
    ASSERT_EQ(paste->nodes.size(), 1);
    EXPECT_TRUE(has(uuid, paste->nodes[0]));
    const auto group = manager.makeGroup(uuid, {"Autumn Grade"});
    ASSERT_TRUE(group);
    EXPECT_TRUE(has(group->graph, "Autumn Grade"));
    EXPECT_FALSE(has(uuid, "Autumn Grade"));
    const auto copied = manager.makeGroupSingleUser(uuid, group->group_node);
    ASSERT_TRUE(copied);
    const auto duplicate = manager.tree(uuid)->find_node(group->group_node)->properties.at("tree").get<std::string>();
    EXPECT_NE(duplicate, group->graph);
    EXPECT_TRUE(has(duplicate, "Autumn Grade"));
    const auto ungroup = manager.ungroup(uuid, group->group_node);
    ASSERT_TRUE(ungroup);
    ASSERT_EQ(ungroup->size(), 1);
    EXPECT_TRUE(has(uuid, ungroup->front()));
    lfs::vis::op::undoHistory().undo();
    EXPECT_NE(manager.tree(uuid)->find_node(group->group_node), nullptr);
    EXPECT_FALSE(has(uuid, "Autumn Grade"));
}

TEST_F(NodesModifierManager, SequencerExportControllerWaitsForEachAnimatedFrame) {
    for_each_worker_target([&](const auto device) {
        lfs::vis::SequencerController controller;
        lfs::vis::SceneManager scene;
        scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        auto& manager = scene.modifierManager();
        manager.setSequencer(&controller);
        const auto id = scene.getScene().addSplat("Host", model(2, device));
        const auto host = scene.getScene().getNodeUuid(id);
        auto& tree = manager.newTree("Export animation");
        const auto input = tree.input_node().name, output = tree.output_node().name;
        tree.add_node("lfs.set_position", "Move");
        tree.add_node("lfs.scene_time", "Clock");
        tree.add_node("lfs.combine_xyz", "Vector");
        ASSERT_TRUE(tree.add_link({input, "Geometry", "Move", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Clock", "Seconds", "Vector", "X"}));
        ASSERT_TRUE(tree.add_link({"Clock", "Frame", "Vector", "Y"}));
        ASSERT_TRUE(tree.add_link({"Vector", "Vector", "Move", "Offset"}));
        ASSERT_TRUE(tree.add_link({"Move", "Geometry", output, "Geometry"}));
        ASSERT_TRUE(manager.keyframeSet(tree.uuid, "Vector", "Z", 0, 1.0f));
        ASSERT_TRUE(manager.keyframeSet(tree.uuid, "Vector", "Z", 1, 11.0f));
        manager.addModifier(host, tree.uuid);
        controller.seek(7); // Export must use the frame time, not this playhead.
        ASSERT_TRUE(manager.evaluate(host).ok);
        (void)manager.performance(true);
        for (int frame = 0; frame < 7; ++frame) {
            const float time = float(frame) / 30;
            ASSERT_TRUE(controller.prepareExportFrame(manager, time, 30));
            const auto* node = scene.getScene().getNodeById(id);
            ASSERT_NE(node->evaluated_model, nullptr);
            const auto positions = node->evaluated_model->means_raw().cpu().to_vector();
            EXPECT_NEAR(positions[0], time, 1e-6f);
            EXPECT_NEAR(positions[1], float(frame), 1e-6f);
            EXPECT_NEAR(positions[2], 1 + 10 * time, 1e-5f);
            EXPECT_FALSE(manager.progress().busy);
        }
        EXPECT_EQ(manager.performance()["evaluations"], 7);
        EXPECT_FLOAT_EQ(controller.playhead(), 7);
        ASSERT_TRUE(manager.evaluate(host).ok); // Restore normal viewport time.
        EXPECT_NEAR(manager.evaluated(host)->splats->means.cpu().to_vector()[0], 7, 1e-6f);
    });
}

TEST_F(NodesModifierManager, AnimatedInputTypesEditingDeletionAndAllEasings) {
    using namespace lfs;
    vis::SequencerController controller;
    vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    manager.setSequencer(&controller);
    auto& graph = manager.newTree("Types");
    const auto uuid = graph.uuid;
    graph.add_node("lfs.set_position", "Position");
    graph.add_node("lfs.set_colour", "Colour");
    graph.add_node("lfs.value", "Value");
    ASSERT_TRUE(manager.keyframeSet(uuid, "Position", "Offset", 0, glm::vec3(0, 2, 4)));
    ASSERT_TRUE(manager.keyframeSet(uuid, "Position", "Offset", 2, glm::vec3(2, 4, 8)));
    ASSERT_TRUE(manager.keyframeSet(uuid, "Colour", "Colour", 0, glm::vec3(0, 0, 0)));
    ASSERT_TRUE(manager.keyframeSet(uuid, "Colour", "Colour", 2, glm::vec4(1, 0.5f, 0.25f, 0.5f)));
    for (int easing = 0; easing < 4; ++easing) {
        ASSERT_TRUE(manager.keyframeSet(uuid, "Value", "Value", 0, 0.0f, easing));
        ASSERT_TRUE(manager.keyframeSet(uuid, "Value", "Value", 2, 8.0f));
        for (float time : {0.0f, 0.25f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f}) {
            auto snapshot = graph;
            vis::applyNodeAnimation(snapshot, controller.timeline().animationClip(), time);
            const float t = std::clamp(time / 2, 0.0f, 1.0f);
            const float eased = easing == 0 ? t : easing == 1 ? t * t * t
                                              : easing == 2   ? 1 - std::pow(1 - t, 3)
                                              : t < 0.5f      ? 4 * t * t * t
                                                              : 1 - std::pow(-2 * t + 2, 3) / 2;
            EXPECT_NEAR(*snapshot.find_node("Value")->input_values.at("Value").get_if<float>(), 8 * eased, 1e-6f);
            const auto offset = *snapshot.find_node("Position")->input_values.at("Offset").get_if<glm::vec3>();
            EXPECT_EQ(offset, glm::vec3(2 * t, 2 + 2 * t, 4 + 4 * t));
            const auto colour = *snapshot.find_node("Colour")->input_values.at("Colour").get_if<glm::vec4>();
            EXPECT_EQ(colour, glm::vec4(t, 0.5f * t, 0.25f * t, 1 - 0.5f * t));
        }
    }
    controller.seek(1);
    ASSERT_TRUE(manager.setNodeInput(uuid, "Value", "Value", 6.0f));
    const auto path = vis::nodeInputTrackPath(uuid, "Value", "Value");
    EXPECT_FLOAT_EQ(std::get<float>(*controller.timeline().animationClip()->getTrackByPath(path)->evaluate(1)), 6);
    vis::op::undoHistory().undo();
    EXPECT_FLOAT_EQ(std::get<float>(*controller.timeline().animationClip()->getTrackByPath(path)->evaluate(1)), 4);
    auto* restored = manager.tree(uuid);
    const auto before = restored->to_json();
    ASSERT_TRUE(restored->remove_node("Value"));
    manager.recordTreeEdit(uuid, before);
    EXPECT_EQ(controller.timeline().animationClip()->getTrackByPath(path), nullptr);
    vis::op::undoHistory().undo();
    EXPECT_NE(controller.timeline().animationClip()->getTrackByPath(path), nullptr);
    ASSERT_TRUE(manager.removeTree(uuid));
    EXPECT_EQ(controller.timeline().animationClip()->trackCount(), 0);
    vis::op::undoHistory().undo();
    EXPECT_NE(controller.timeline().animationClip()->getTrackByPath(path), nullptr);
}

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

TEST_F(NodesModifierManager, ViewportCoordinatesRoundTripWithHostTransformAndBasisFlip) {
    using lfs::vis::nodes::NodeViewportTransform;
    using lfs::vis::nodes::ViewportCoordinates;
    lfs::vis::SceneManager scene_manager;
    scene_manager.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
    const auto id = scene_manager.getScene().addSplat("Host", model(1, lfs::core::Device::CPU));
    const glm::mat4 host = glm::translate(glm::mat4(1.0f), {3.0f, -2.0f, 5.0f}) *
                           glm::rotate(glm::mat4(1.0f), glm::radians(37.0f), {0.0f, 1.0f, 0.0f}) *
                           glm::scale(glm::mat4(1.0f), {2.0f, 0.75f, 1.5f});
    scene_manager.getScene().setNodeTransform(id, host);
    const ViewportCoordinates coordinates(scene_manager.getScene(), id);
    ASSERT_TRUE(coordinates.valid());

    const glm::vec3 local_point(0.25f, -1.5f, 2.0f);
    const glm::vec3 data_world = glm::vec3(host * glm::vec4(local_point, 1.0f));
    const glm::vec3 expected_visualizer_world(data_world.x, -data_world.y, -data_world.z);
    const glm::vec3 visualizer_world = coordinates.pointToWorld(local_point);
    EXPECT_NEAR(visualizer_world.x, expected_visualizer_world.x, 1e-5f);
    EXPECT_NEAR(visualizer_world.y, expected_visualizer_world.y, 1e-5f);
    EXPECT_NEAR(visualizer_world.z, expected_visualizer_world.z, 1e-5f);
    const glm::vec3 round_trip = coordinates.pointToLocal(visualizer_world);
    EXPECT_NEAR(round_trip.x, local_point.x, 1e-5f);
    EXPECT_NEAR(round_trip.y, local_point.y, 1e-5f);
    EXPECT_NEAR(round_trip.z, local_point.z, 1e-5f);

    const NodeViewportTransform local{
        .translation = {0.4f, -0.7f, 1.2f},
        .rotation_degrees = {13.0f, -21.0f, 32.0f},
        .scale = {1.25f, 0.8f, 1.6f},
    };
    const auto recovered = coordinates.transformToLocal(coordinates.transformToWorld(local));
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_NEAR(recovered.translation[axis], local.translation[axis], 1e-4f);
        EXPECT_NEAR(recovered.rotation_degrees[axis], local.rotation_degrees[axis], 1e-3f);
        EXPECT_NEAR(recovered.scale[axis], local.scale[axis], 1e-4f);
    }
}

// Fails when the viewport gizmo composes rotations in another order than node evaluation,
// which draws a rotated box or ellipsoid where it does not select.
TEST(NodesViewportCoordinates, ComposeUsesTheEvaluatorRotationOrder) {
    using lfs::vis::nodes::NodeViewportTransform;
    using lfs::vis::nodes::ViewportCoordinates;
    const NodeViewportTransform transform{
        .translation = {1.0f, -2.0f, 0.5f},
        .rotation_degrees = {30.0f, 45.0f, 60.0f},
        .scale = {2.0f, 1.0f, 0.5f}};
    const glm::mat4 expected = glm::translate(glm::mat4(1.0f), transform.translation) *
                               lfs::nodes::rotation_matrix(transform.rotation_degrees) *
                               glm::scale(glm::mat4(1.0f), transform.scale);
    const glm::mat4 composed = ViewportCoordinates::composeLocal(transform);
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            EXPECT_NEAR(composed[column][row], expected[column][row], 1e-5f) << column << "," << row;
    const auto decomposed = ViewportCoordinates::decomposeLocal(composed);
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_NEAR(decomposed.rotation_degrees[axis], transform.rotation_degrees[axis], 1e-3f) << axis;
        EXPECT_NEAR(decomposed.scale[axis], transform.scale[axis], 1e-5f) << axis;
    }
}

TEST_F(NodesModifierManager, HsvEyedropperCentresBandsAndKeepsTheirWidthsAtBounds) {
    const auto bands = lfs::vis::centreHsvPickBands({0.2f, 0.8f, 0.4f}, 0.2f, 0.4f);
    EXPECT_NEAR(bands.hue, 1.0f / 3.0f + 1.0f / 18.0f, 1e-5f);
    EXPECT_NEAR(bands.saturation_max - bands.saturation_min, 0.2f, 1e-5f);
    EXPECT_NEAR(bands.value_max - bands.value_min, 0.4f, 1e-5f);
    EXPECT_NEAR((bands.saturation_min + bands.saturation_max) * 0.5f, 0.75f, 1e-5f);
    EXPECT_NEAR((bands.value_min + bands.value_max) * 0.5f, 0.8f, 1e-5f);

    const auto edge = lfs::vis::centreHsvPickBands({1.0f, 0.0f, 0.0f}, 0.4f, 0.6f);
    EXPECT_NEAR(edge.saturation_max - edge.saturation_min, 0.4f, 1e-5f);
    EXPECT_NEAR(edge.value_max - edge.value_min, 0.6f, 1e-5f);
    EXPECT_FLOAT_EQ(edge.saturation_max, 1.0f);
    EXPECT_FLOAT_EQ(edge.value_max, 1.0f);
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

// Relative space moves the target into the host's frame and rotates its SH as transform() does.
TEST_F(NodesModifierManager, ObjectInfoRelativeSpaceMatchesSplatTransform) {
    using namespace lfs::nodes;
    using lfs::core::Device;
    using lfs::core::Tensor;
    for_each_worker_target([](const Device device) {
        const auto values = [&](const std::vector<std::size_t>& shape, const float scale, const int seed) {
            std::size_t count = 1;
            for (const auto extent : shape)
                count *= extent;
            std::vector<float> data(count);
            for (std::size_t i = 0; i < count; ++i)
                data[i] = scale * std::sin(0.7f * static_cast<float>(i) + static_cast<float>(seed));
            return Tensor::from_vector(data, lfs::core::TensorShape(shape), Device::CPU).to(device);
        };
        const auto reference = [&] {
            return std::make_unique<lfs::core::SplatData>(
                1, values({5, 3}, 2.0f, 1), values({5, 1, 3}, 0.5f, 2), values({5, 3, 3}, 0.3f, 3),
                values({5, 3}, 0.2f, 4), values({5, 4}, 1.0f, 5), values({5, 1}, 1.0f, 6), 1.0f);
        };
        const glm::mat4 world = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)) *
                                glm::rotate(glm::mat4(1.0f), 0.7f, glm::normalize(glm::vec3(0.3f, 0.5f, 0.8f))) *
                                glm::scale(glm::mat4(1.0f), glm::vec3(1.5f));
        lfs::vis::SceneManager scene;
        scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        const auto host = scene.getScene().getNodeUuid(scene.getScene().addSplat("Host", model(6, device)));
        scene.getScene().addSplat("Reference", reference());
        scene.getScene().setNodeTransform("Reference", world);
        auto& manager = scene.modifierManager();
        auto& tree = manager.newTree("Relative reference");
        auto& info = tree.add_node("lfs.object_info", "Object Info");
        info.properties["object"] = "Reference";
        info.properties["transform_space"] = "relative";
        ASSERT_TRUE(tree.remove_link(
            {tree.input_node().name, "Geometry", tree.output_node().name, "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Object Info", "Geometry", tree.output_node().name, "Geometry"}));
        manager.addModifier(host, tree.uuid);

        const auto result = manager.evaluate(host);
        ASSERT_TRUE(result.ok) << (result.errors.empty() ? "no error text"
                                                         : result.errors.begin()->second);
        ASSERT_TRUE(result.geometry.splats);
        auto expected = reference();
        lfs::core::transform(*expected, world);
        const auto& actual = *result.geometry.splats;
        const auto expect_near = [](const Tensor& a, const Tensor& b, const char* name) {
            const auto x = a.cpu().contiguous().to_vector();
            const auto y = b.cpu().contiguous().to_vector();
            ASSERT_EQ(x.size(), y.size()) << name;
            for (std::size_t i = 0; i < x.size(); ++i)
                ASSERT_NEAR(x[i], y[i], 1e-4f) << name << " element " << i;
        };
        expect_near(actual.means, expected->means_raw(), "means");
        expect_near(actual.rotation, expected->rotation_raw(), "rotation");
        expect_near(actual.scaling, expected->scaling_raw(), "scaling");
        expect_near(actual.shN, expected->shN_canonical(), "shN");
    });
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

TEST_F(NodesModifierManager, SelectionPreviewSurvivesAttributeNodes) {
    using namespace lfs::nodes;
    for_each_worker_target([](const lfs::core::Device device) {
        lfs::vis::SceneManager scene;
        scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        const auto id = scene.getScene().addSplat("Host", model(6, device));
        const auto host = scene.getScene().getNodeUuid(id);
        auto& manager = scene.modifierManager();
        auto& tree = manager.newTree("Preview opacity");
        auto& stored = tree.add_node("lfs.stored_selection", "Stored");
        set_stored_selection(stored, selection({true, false, true, false, false, false}));
        const auto stored_properties = stored.properties;
        tree.add_node("lfs.set_opacity", "Opacity").input_values["Opacity"] = 0.25f;
        ASSERT_TRUE(tree.remove_link(
            {tree.input_node().name, "Geometry", tree.output_node().name, "Geometry"}));
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Opacity", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Stored", "Selection", "Opacity", "Selection"}));
        ASSERT_TRUE(tree.add_link({"Opacity", "Geometry", tree.output_node().name, "Geometry"}));
        auto& modifier = manager.addModifier(host, tree.uuid, "Opacity");
        modifier.stored_selections["Stored"] = stored_properties;

        const auto result = manager.evaluate(host);
        ASSERT_TRUE(result.ok) << (result.errors.empty() ? "no error text"
                                                         : result.errors.begin()->second);
        for (const auto* name : {"Stored", "Opacity"}) {
            const auto preview = manager.selectionPreview(host, modifier.uuid, name);
            ASSERT_TRUE(preview) << name;
            EXPECT_EQ(preview->cpu().to_vector_bool(),
                      (std::vector<bool>{true, false, true, false, false, false}))
                << name;
        }
    });
}

// Nodes inside groups report as "Group/Inner"; attribute-only groups and reroutes keep rows.
TEST_F(NodesModifierManager, SelectionPreviewSurvivesGroupsAndReroutes) {
    using namespace lfs::nodes;
    for_each_worker_target([](const lfs::core::Device device) {
        lfs::vis::SceneManager scene;
        scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        const auto id = scene.getScene().addSplat("Host", model(6, device));
        const auto host = scene.getScene().getNodeUuid(id);
        auto& manager = scene.modifierManager();
        const auto resolver = [&](const std::string_view uuid) { return manager.tree(uuid); };
        auto& inner = manager.newTree("Tint group");
        inner.add_node("lfs.set_colour", "Tint");
        ASSERT_TRUE(inner.remove_link(
            {inner.input_node().name, "Geometry", inner.output_node().name, "Geometry"}));
        ASSERT_TRUE(inner.add_link({inner.input_node().name, "Geometry", "Tint", "Geometry"}));
        ASSERT_TRUE(inner.add_link({"Tint", "Geometry", inner.output_node().name, "Geometry"}));

        auto& tree = manager.newTree("Grouped");
        auto& stored = tree.add_node("lfs.stored_selection", "Stored");
        set_stored_selection(stored, selection({true, false, true, false, false, false}));
        const auto stored_properties = stored.properties;
        tree.add_node("lfs.reroute", "Route");
        tree.add_node("lfs.group", "Group");
        tree.add_node("lfs.set_opacity", "Opacity").input_values["Opacity"] = 0.25f;
        ASSERT_TRUE(manager.setGroupGraph(tree.uuid, "Group", inner.uuid));
        ASSERT_TRUE(tree.remove_link(
            {tree.input_node().name, "Geometry", tree.output_node().name, "Geometry"}));
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Route", "Input"}, nullptr, resolver));
        ASSERT_TRUE(tree.add_link({"Route", "Output", "Group", "Geometry"}, nullptr, resolver));
        ASSERT_TRUE(tree.add_link({"Group", "Geometry", "Opacity", "Geometry"}, nullptr, resolver));
        ASSERT_TRUE(tree.add_link({"Stored", "Selection", "Opacity", "Selection"}));
        ASSERT_TRUE(tree.add_link({"Opacity", "Geometry", tree.output_node().name, "Geometry"}));
        auto& modifier = manager.addModifier(host, tree.uuid, "Grouped");
        modifier.stored_selections["Stored"] = stored_properties;

        const auto result = manager.evaluate(host);
        ASSERT_TRUE(result.ok) << (result.errors.empty() ? "no error text"
                                                         : result.errors.begin()->second);
        EXPECT_NE(result.nodes.find(modifier.uuid + "/Group/Tint"), result.nodes.end());
        EXPECT_TRUE(result.rows_follow_source);
        const auto preview = manager.selectionPreview(host, modifier.uuid, "Opacity");
        ASSERT_TRUE(preview);
        EXPECT_EQ(preview->cpu().to_vector_bool(), (std::vector<bool>{true, false, true, false, false, false}));
    });
}

TEST_F(NodesModifierManager, SelectionPreviewHidesWhenAJoinReordersElements) {
    using namespace lfs::nodes;
    for_each_worker_target([](const lfs::core::Device device) {
        lfs::vis::SceneManager scene;
        scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        const auto id = scene.getScene().addSplat("Host", model(6, device));
        const auto host = scene.getScene().getNodeUuid(id);
        auto& manager = scene.modifierManager();
        auto& tree = manager.newTree("Reorder");
        auto& stored = tree.add_node("lfs.stored_selection", "Stored");
        set_stored_selection(stored, selection({true, false, false, false, false, false}));
        const auto stored_properties = stored.properties;
        tree.add_node("lfs.separate_geometry", "Separate");
        tree.add_node("lfs.join_geometry", "Join");
        ASSERT_TRUE(tree.remove_link(
            {tree.input_node().name, "Geometry", tree.output_node().name, "Geometry"}));
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Separate", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Stored", "Selection", "Separate", "Selection"}));
        ASSERT_TRUE(tree.add_link({"Separate", "Inverted", "Join", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Separate", "Selection", "Join", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Join", "Geometry", tree.output_node().name, "Geometry"}));
        auto& modifier = manager.addModifier(host, tree.uuid, "Reorder");
        modifier.stored_selections["Stored"] = stored_properties;

        const auto result = manager.evaluate(host);
        ASSERT_TRUE(result.ok) << (result.errors.empty() ? "no error text"
                                                         : result.errors.begin()->second);
        ASSERT_TRUE(result.geometry.splats);
        EXPECT_EQ(result.geometry.splats->means.shape()[0], 6);
        EXPECT_FALSE(result.rows_follow_source);
        // The count matches, but displayed row 0 is stored row 1.
        EXPECT_FALSE(manager.selectionPreview(host, modifier.uuid, "Stored"));
        EXPECT_FALSE(manager.selectionPreview(host, modifier.uuid, "Separate"));
        const auto status = result.nodes.find(modifier.uuid + "/Separate");
        ASSERT_NE(status, result.nodes.end());
        ASSERT_TRUE(status->second.selected_share);
        EXPECT_NEAR(*status->second.selected_share, 1.0 / 6.0, 1e-9);
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

TEST_F(NodesModifierManager, TrainingSuspendsEvaluationAndResumesLatestGraphOnce) {
    ServicesScope services_scope;
    lfs::vis::SceneManager scene_manager;
    auto& scene = scene_manager.getScene();
    const auto id = scene.addSplat("Training model", model(4));
    scene.setTrainingModelNode(id);
    add_training_camera(scene);
    scene_manager.changeContentType(lfs::vis::SceneManager::ContentType::Dataset);

    lfs::vis::TrainerManager trainer_manager;
    trainer_manager.setScene(&scene);
    lfs::vis::services().set(&trainer_manager);

    auto& manager = scene_manager.modifierManager();
    std::atomic<int> evaluations = 0;
    lfs::nodes::NodeTypeInfo counter;
    counter.id = "test.training_counter";
    counter.label = "Training counter";
    counter.inputs = {{"Geometry", "Geometry", std::string(lfs::nodes::GEOMETRY_SOCKET)}};
    counter.outputs = counter.inputs;
    counter.evaluate = [&](lfs::nodes::NodeContext& context) {
        ++evaluations;
        context.set_output("Geometry", context.input("Geometry"));
    };
    ASSERT_TRUE(manager.registry().register_type(std::move(counter)));
    auto& tree = manager.newTree("Training graph");
    tree.add_node("test.training_counter", "Counter");
    ASSERT_TRUE(tree.remove_link({tree.input_node().name, "Geometry",
                                  tree.output_node().name, "Geometry"}));
    ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Counter", "Geometry"}));
    ASSERT_TRUE(tree.add_link({"Counter", "Geometry", tree.output_node().name, "Geometry"}));
    const auto host = scene.getNodeUuid(id);
    manager.addModifier(host, tree.uuid);
    ASSERT_TRUE(manager.evaluate(host).ok);
    ASSERT_EQ(evaluations.load(), 1);
    ASSERT_NE(scene.getNodeById(id)->evaluated_model, nullptr);
    EXPECT_EQ(scene_manager.buildRenderState().combined_model,
              scene.getNodeById(id)->evaluated_model.get());

    ASSERT_TRUE(transition(trainer_manager, lfs::vis::TrainingState::Ready));
    ASSERT_TRUE(transition(trainer_manager, lfs::vis::TrainingState::Starting));
    ASSERT_TRUE(transition(trainer_manager, lfs::vis::TrainingState::Running));
    manager.tick();
    EXPECT_TRUE(manager.trainingSuspended());
    EXPECT_EQ(scene.getNodeById(id)->evaluated_model, nullptr);
    EXPECT_EQ(scene_manager.buildRenderState().combined_model,
              scene.getNodeById(id)->model.get());
    for (int iteration = 0; iteration < 8; ++iteration) {
        scene.notifyMutation(lfs::core::Scene::MutationType::MODEL_CHANGED);
        manager.markDirty(host);
        manager.tick();
    }
    EXPECT_EQ(evaluations.load(), 1);

    ASSERT_TRUE(transition(trainer_manager, lfs::vis::TrainingState::Paused));
    ASSERT_TRUE(manager.evaluate(host).ok);
    EXPECT_FALSE(manager.trainingSuspended());
    EXPECT_EQ(evaluations.load(), 2);
    manager.tick();
    EXPECT_EQ(evaluations.load(), 2);
}

TEST_F(NodesModifierManager, TrainingApplyResetsAllAdamMomentsAndRejectsStructuralResult) {
    ServicesScope services_scope;
    lfs::vis::SceneManager scene_manager;
    auto& scene = scene_manager.getScene();
    const auto id = scene.addSplat("Training model", model(4));
    scene.setTrainingModelNode(id);
    add_training_camera(scene);
    scene_manager.changeContentType(lfs::vis::SceneManager::ContentType::Dataset);

    auto optimizer_model = model(4);
    lfs::training::AdamOptimizer optimizer(*optimizer_model, {});
    for (const auto parameter : lfs::training::AdamOptimizer::all_param_types()) {
        optimizer.get_grad(parameter);
        auto* state = optimizer.get_state_mutable(parameter);
        ASSERT_NE(state, nullptr);
        ASSERT_TRUE(state->exp_avg.is_valid());
        state->exp_avg.copy_from(
            lfs::core::Tensor::ones(state->exp_avg.shape(), state->exp_avg.device())
                .to(state->exp_avg.dtype()));
        if (state->joint_bounds.is_valid())
            state->joint_bounds.fill_(3);
        state->step_count = 9;
    }
    optimizer.reset_all_states();
    for (const auto parameter : lfs::training::AdamOptimizer::all_param_types()) {
        const auto* state = optimizer.get_state(parameter);
        ASSERT_NE(state, nullptr);
        EXPECT_EQ(state->step_count, 0);
        EXPECT_EQ(state->exp_avg.count_nonzero(), 0u);
        if (state->joint_bounds.is_valid())
            EXPECT_EQ(state->joint_bounds.count_nonzero(), 0u);
    }

    auto& manager = scene_manager.modifierManager();
    auto& colour = colour_tree(manager);
    const auto host = scene.getNodeUuid(id);
    manager.addModifier(host, colour.uuid, "Colour");
    const auto before_count = scene.getNodeById(id)->model->size();
    ASSERT_TRUE(manager.applyModifier(host, "Colour"));
    EXPECT_EQ(scene.getNodeById(id)->model->size(), before_count);

    auto& structural = manager.newTree("Structural");
    auto& remove = structural.add_node("lfs.delete_geometry", "Delete");
    ASSERT_TRUE(structural.remove_link({structural.input_node().name, "Geometry",
                                        structural.output_node().name, "Geometry"}));
    ASSERT_TRUE(structural.add_link({structural.input_node().name, "Geometry",
                                     remove.name, "Geometry"}));
    ASSERT_TRUE(structural.add_link({remove.name, "Geometry",
                                     structural.output_node().name, "Geometry"}));
    manager.addModifier(host, structural.uuid, "Structural");
    const auto applied = manager.applyModifier(host, "Structural");
    ASSERT_FALSE(applied);
    EXPECT_EQ(applied.error().message, "Export the result or stop training first");
    EXPECT_EQ(scene.getNodeById(id)->model->size(), before_count);
}

TEST_F(NodesModifierManager, CheckpointDocumentContextRoundTripsGraphsAndStacks) {
    lfs::vis::SceneManager source_scene;
    source_scene.changeContentType(lfs::vis::SceneManager::ContentType::Dataset);
    const auto id = source_scene.getScene().addSplat("Training model", model(3));
    source_scene.getScene().setTrainingModelNode(id);
    auto& source = source_scene.modifierManager();
    auto& tree = colour_tree(source);
    const auto host = source_scene.getScene().getNodeUuid(id);
    source.addModifier(host, tree.uuid, "Checkpoint colour");

    auto parsed = lfs::io::JsonChapterDom::parse(source.toJson().dump());
    ASSERT_TRUE(parsed) << lfs::format_for_developer(parsed.error());
    lfs::training::ProjectSnapshotDocumentContext context;
    context.nodes = lfs::io::project::NodesSessionChapter(std::move(*parsed));
    auto restored_chapter = lfs::io::project::NodesSessionChapter::from_bytes(
        context.nodes.to_bytes());
    ASSERT_TRUE(restored_chapter) << lfs::format_for_developer(restored_chapter.error());

    lfs::vis::SceneManager restored_scene;
    auto& restored = restored_scene.modifierManager();
    ASSERT_TRUE(restored.restoreJson(nlohmann::json::parse(restored_chapter->dom().dump())));
    ASSERT_NE(restored.tree(tree.uuid), nullptr);
    ASSERT_NE(restored.stack(host), nullptr);
    ASSERT_EQ(restored.stack(host)->modifiers.size(), 1u);
    EXPECT_EQ(restored.stack(host)->modifiers.front().name, "Checkpoint colour");
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

TEST_F(NodesModifierManager, ClipboardGroupsInterfaceAndLayoutRoundTrip) {
    lfs::vis::SceneManager scene;
    scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
    const auto id = scene.getScene().addSplat("Host", model(4));
    const auto host = scene.getScene().getNodeUuid(id);
    auto& manager = scene.modifierManager();
    auto& tree = colour_tree(manager);
    auto& hsv = tree.add_node("lfs.hsv_range", "HSV");
    ASSERT_TRUE(tree.add_link({hsv.name, "Selection", "Correct", "Selection"}));
    auto& modifier = manager.addModifier(host, tree.uuid);

    const auto baseline = manager.evaluate(host);
    ASSERT_TRUE(baseline.ok);
    ASSERT_TRUE(baseline.geometry.splats);
    const auto baseline_colours = baseline.geometry.splats->sh0.to_vector();

    const auto clipboard = manager.copyNodes(tree.uuid, {"HSV", "Correct"});
    ASSERT_TRUE(clipboard.has_value());
    const auto encoded = nlohmann::json::parse(*clipboard);
    EXPECT_EQ(encoded["format"], "lfs.node-clipboard");
    EXPECT_EQ(encoded["links"].size(), 1u);
    const auto pasted = manager.pasteNodes(tree.uuid, *clipboard, std::array<float, 2>{500, 200});
    ASSERT_TRUE(pasted.has_value());
    EXPECT_EQ(pasted->nodes.size(), 2u);
    EXPECT_EQ(pasted->dropped_links, 0u);
    EXPECT_TRUE(std::ranges::any_of(tree.links, [&](const auto& link) {
        return std::ranges::find(pasted->nodes, link.from_node) != pasted->nodes.end() &&
               std::ranges::find(pasted->nodes, link.to_node) != pasted->nodes.end();
    }));

    const auto grouped = manager.makeGroup(tree.uuid, {"Correct"}, "Grade");
    ASSERT_TRUE(grouped.has_value());
    EXPECT_NE(manager.tree(grouped->graph), nullptr);
    const auto grouped_result = manager.evaluate(host);
    ASSERT_TRUE(grouped_result.ok);
    ASSERT_TRUE(grouped_result.geometry.splats);
    EXPECT_EQ(grouped_result.geometry.splats->sh0.to_vector(), baseline_colours);
    auto* grade_graph = manager.tree(grouped->graph);
    ASSERT_NE(grade_graph, nullptr);
    const auto nested_group = manager.makeGroup(grade_graph->uuid, {"Correct"}, "Nested Grade");
    ASSERT_TRUE(nested_group.has_value());
    const auto inlined = manager.ungroup(tree.uuid, grouped->group_node);
    ASSERT_TRUE(inlined.has_value());
    const auto inlined_result = manager.evaluate(host);
    ASSERT_TRUE(inlined_result.ok);
    EXPECT_EQ(inlined_result.geometry.splats->sh0.to_vector(), baseline_colours);

    const auto amount = manager.interfaceAdd(tree.uuid, false, "lfs.float", "Amount", 0.5f,
                                             0.0, 1.0, 0.1);
    ASSERT_TRUE(amount.has_value());
    modifier.input_overrides[*amount] = 0.75f;
    ASSERT_TRUE(manager.interfaceUpdate(tree.uuid, false, *amount, {{"label", "Strength"}}));
    EXPECT_TRUE(modifier.input_overrides.contains(*amount));
    ASSERT_TRUE(manager.interfaceMove(tree.uuid, false, *amount, 0));
    EXPECT_EQ(tree.group_interface.inputs.front().identifier, *amount);
    ASSERT_TRUE(manager.interfaceRemove(tree.uuid, false, *amount));
    EXPECT_FALSE(modifier.input_overrides.contains(*amount));

    const auto frame = manager.frameWrap(tree.uuid, {"HSV"}, "Mask");
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(tree.find_node("HSV")->ui.value("frame", ""), *frame);
    ASSERT_TRUE(manager.frameSetMembers(tree.uuid, *frame, pasted->nodes));
    EXPECT_FALSE(tree.find_node("HSV")->ui.contains("frame"));
    for (const auto& name : pasted->nodes)
        EXPECT_EQ(tree.find_node(name)->ui.value("frame", ""), *frame);

    const auto geometry_link = std::ranges::find_if(tree.links, [&](const auto& link) {
        return link.from_node == tree.input_node().name && link.from_socket == "Geometry";
    });
    ASSERT_NE(geometry_link, tree.links.end());
    const auto route = manager.rerouteInsert(tree.uuid, *geometry_link);
    ASSERT_TRUE(route.has_value());
    EXPECT_EQ(lfs::nodes::effective_outputs(tree, *tree.find_node(*route),
                                            [&](std::string_view uuid) { return manager.tree(uuid); })
                  .front()
                  .type,
              lfs::nodes::GEOMETRY_SOCKET);
    EXPECT_TRUE(manager.evaluate(host).ok);
    const auto saved_nested = manager.toJson(false);
    ASSERT_TRUE(manager.restoreJson(saved_nested));
    const auto restored_nested = manager.evaluate(host);
    ASSERT_TRUE(restored_nested.ok);
    ASSERT_TRUE(restored_nested.geometry.splats);
    EXPECT_EQ(restored_nested.geometry.splats->sh0.to_vector(), baseline_colours);
}

TEST_F(NodesModifierManager, ClipboardPreservesInternalLinksDropsExternalAndRenamesCollisions) {
    lfs::vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    auto& tree = colour_tree(manager);
    tree.add_node("lfs.hsv_range", "Mask");
    ASSERT_TRUE(tree.add_link({"Mask", "Selection", "Correct", "Selection"}));
    const auto copy = manager.copyNodes(tree.uuid, {"Mask", "Correct", tree.input_node().name});
    ASSERT_TRUE(copy);
    const auto payload = nlohmann::json::parse(*copy);
    EXPECT_EQ(payload["nodes"].size(), 2u);
    EXPECT_EQ(payload["links"].size(), 1u);
    const auto paste = manager.pasteNodes(tree.uuid, *copy);
    ASSERT_TRUE(paste);
    ASSERT_EQ(paste->nodes.size(), 2u);
    EXPECT_NE(paste->nodes[0], "Mask");
    EXPECT_NE(paste->nodes[1], "Correct");
    const std::unordered_set<std::string> pasted(paste->nodes.begin(), paste->nodes.end());
    EXPECT_EQ(std::ranges::count_if(tree.links, [&](const auto& link) {
                  return pasted.contains(link.from_node) && pasted.contains(link.to_node);
              }),
              1);
    EXPECT_EQ(std::ranges::count_if(tree.links, [&](const auto& link) {
                  return pasted.contains(link.from_node) != pasted.contains(link.to_node);
              }),
              0);
}

TEST_F(NodesModifierManager, ClipboardReusesIdenticalGroupTreeAndImportsConflictingUuid) {
    lfs::vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    auto& tree = colour_tree(manager);
    const auto grouped = manager.makeGroup(tree.uuid, {"Correct"}, "Grade");
    ASSERT_TRUE(grouped);
    const auto copied = manager.copyNodes(tree.uuid, {grouped->group_node});
    ASSERT_TRUE(copied);
    const auto count = manager.trees().size();
    const auto same = manager.pasteNodes(tree.uuid, *copied);
    ASSERT_TRUE(same);
    EXPECT_EQ(manager.trees().size(), count);
    EXPECT_EQ(tree.find_node(same->nodes.front())->properties["tree"], grouped->graph);
    manager.tree(grouped->graph)->find_node("Correct")->input_values["Exposure"] = 3.0f;
    const auto different = manager.pasteNodes(tree.uuid, *copied);
    ASSERT_TRUE(different);
    const auto imported = tree.find_node(different->nodes.front())->properties["tree"].get<std::string>();
    EXPECT_NE(imported, grouped->graph);
    ASSERT_NE(manager.tree(imported), nullptr);
    EXPECT_EQ(*manager.tree(imported)->find_node("Correct")->input_values.at("Exposure").get_if<float>(), 1.0f);
    EXPECT_EQ(manager.trees().size(), count + 1);
}

TEST_F(NodesModifierManager, ClipboardRemapsReusedParentWhenNestedTreeConflicts) {
    lfs::vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    auto& tree = colour_tree(manager);
    const auto inner = manager.makeGroup(tree.uuid, {"Correct"}, "Inner");
    ASSERT_TRUE(inner);
    const auto outer = manager.makeGroup(tree.uuid, {inner->group_node}, "Outer");
    ASSERT_TRUE(outer);
    const auto copied = manager.copyNodes(tree.uuid, {outer->group_node});
    ASSERT_TRUE(copied);
    manager.tree(inner->graph)->find_node("Correct")->input_values["Exposure"] = 4.0f;
    const auto pasted = manager.pasteNodes(tree.uuid, *copied);
    ASSERT_TRUE(pasted);
    const auto parent = tree.find_node(pasted->nodes.front())->properties["tree"].get<std::string>();
    EXPECT_NE(parent, outer->graph);
    const auto child = manager.tree(parent)->find_node(inner->group_node)->properties["tree"].get<std::string>();
    EXPECT_NE(child, inner->graph);
    EXPECT_EQ(*manager.tree(child)->find_node("Correct")->input_values.at("Exposure").get_if<float>(), 1.0f);
}

TEST_F(NodesModifierManager, NonClipboardTextDoesNotMutateLibraryOrUndoHistory) {
    lfs::vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    const auto id = manager.newTree("Clipboard").uuid;
    const auto before = manager.toJson(false);
    lfs::vis::op::undoHistory().clear();
    for (const auto* text : {"ordinary text", "{}", "[]", "null", R"({"format":17})",
                             R"({"format":"lfs.node-clipboard","version":"one"})"}) {
        const auto result = manager.pasteNodes(id, text);
        ASSERT_FALSE(result);
        EXPECT_NE(result.error().message.find("Clipboard"), std::string::npos);
        EXPECT_EQ(manager.toJson(false), before);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    }
}

TEST_F(NodesModifierManager, MakeGroupAndUngroupPreservePositionDerivedFieldBitwise) {
    using namespace lfs::nodes;
    using lfs::core::Device;
    using lfs::core::Tensor;
    lfs::vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    auto& tree = colour_tree(manager);
    tree.add_node("lfs.position", "Position");
    tree.add_node("lfs.separate_xyz", "Axes");
    tree.add_node("lfs.compare", "Mask").properties["operation"] = "greater_than";
    ASSERT_TRUE(tree.add_link({"Position", "Position", "Axes", "Vector"}));
    ASSERT_TRUE(tree.add_link({"Axes", "X", "Mask", "A"}));
    ASSERT_TRUE(tree.add_link({"Mask", "Result", "Correct", "Selection"}));
    auto geometry = geometry_from_splat_data(*model(3));
    geometry.splats->means = Tensor::from_vector({-1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f}, {3, 3}, Device::GPU);
    const TreeResolver resolver = [&](std::string_view id) { return manager.tree(id); };
    const auto baseline = evaluate(tree, {.geometry = geometry, .tree_resolver = resolver});
    ASSERT_TRUE(baseline.ok);
    ASSERT_NE(baseline.geometry.splats->sh0.to_vector(), geometry.splats->sh0.to_vector());
    const auto check = [&](const EvalResult& result) {
        ASSERT_TRUE(result.ok);
        const auto& a = *baseline.geometry.splats;
        const auto& b = *result.geometry.splats;
        for (const auto pair : {std::pair{a.means, b.means}, {a.sh0, b.sh0}, {a.shN, b.shN}, {a.scaling, b.scaling}, {a.rotation, b.rotation}, {a.opacity, b.opacity}}) {
            const auto expected = pair.first.to_vector();
            const auto actual = pair.second.to_vector();
            ASSERT_EQ(actual.size(), expected.size());
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
        }
    };
    const auto grouped = manager.makeGroup(tree.uuid, {"Correct"}, "Field Grade");
    ASSERT_TRUE(grouped);
    check(evaluate(tree, {.geometry = geometry, .tree_resolver = resolver}));
    ASSERT_TRUE(manager.ungroup(tree.uuid, grouped->group_node));
    check(evaluate(tree, {.geometry = geometry, .tree_resolver = resolver}));
}

TEST_F(NodesModifierManager, MakeGroupAndUngroupPreserveMultiInputLinkOrder) {
    lfs::vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    auto& tree = manager.newTree("Test Graph");
    for (const auto* name : {"A", "B", "C"})
        tree.add_node("lfs.transform_geometry", name);
    tree.add_node("lfs.join_geometry", "Join");
    for (const auto* name : {"B", "A", "C"})
        ASSERT_TRUE(tree.add_link({name, "Geometry", "Join", "Geometry"}));
    const auto links = tree.links;
    const auto grouped = manager.makeGroup(tree.uuid, {"Join"});
    ASSERT_TRUE(grouped);
    ASSERT_TRUE(manager.ungroup(tree.uuid, grouped->group_node));
    EXPECT_EQ(tree.links, links);
}

TEST_F(NodesModifierManager, NonConvexGroupSelectionIsRejectedTransactionally) {
    lfs::vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    auto& tree = manager.newTree("Test Graph");
    for (const auto* name : {"A", "B", "C"})
        tree.add_node("lfs.math", name);
    ASSERT_TRUE(tree.add_link({"A", "Value", "B", "A"}));
    ASSERT_TRUE(tree.add_link({"B", "Value", "C", "A"}));
    const auto before = manager.toJson(false);
    lfs::vis::op::undoHistory().clear();
    const auto result = manager.makeGroup(tree.uuid, {"A", "C"});
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().message.find("leave and re-enter"), std::string::npos);
    EXPECT_EQ(manager.toJson(false), before);
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
}

TEST_F(NodesModifierManager, MakeGroupAndUngroupEachCreateOneUndoStep) {
    lfs::vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    const auto id = colour_tree(manager).uuid;
    auto& history = lfs::vis::op::undoHistory();
    history.clear();
    const auto before = manager.toJson(false);
    const auto grouped = manager.makeGroup(id, {"Correct"});
    ASSERT_TRUE(grouped);
    EXPECT_EQ(history.undoCount(), 1u);
    const auto after = manager.toJson(false);
    ASSERT_TRUE(history.undo().success);
    EXPECT_EQ(manager.toJson(false), before);
    ASSERT_TRUE(history.redo().success);
    EXPECT_EQ(manager.toJson(false), after);
    history.clear();
    ASSERT_TRUE(manager.ungroup(id, grouped->group_node));
    EXPECT_EQ(history.undoCount(), 1u);
    ASSERT_TRUE(history.undo().success);
    EXPECT_EQ(manager.toJson(false), after);
}

TEST_F(NodesModifierManager, SetGroupGraphRejectsDirectAndTransitiveCycles) {
    lfs::vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    auto& a = manager.newTree("A");
    auto& b = manager.newTree("B");
    a.add_node("lfs.group", "Group");
    b.add_node("lfs.group", "Group");
    EXPECT_FALSE(manager.setGroupGraph(a.uuid, "Group", a.uuid));
    ASSERT_TRUE(manager.setGroupGraph(a.uuid, "Group", b.uuid));
    const auto before = manager.toJson(false);
    const auto result = manager.setGroupGraph(b.uuid, "Group", a.uuid);
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().message.find("Group cycle"), std::string::npos);
    EXPECT_EQ(manager.toJson(false), before);
}

TEST_F(NodesModifierManager, InterfaceRenameAndMoveRetainOverrideByStableIdentifier) {
    lfs::vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    const auto host = scene.getScene().getNodeUuid(scene.getScene().addSplat("Host", model()));
    const auto tree = manager.newTree("Interface").uuid;
    const auto id = manager.interfaceAdd(tree, false, "lfs.float", "Strength", 0.5f, 0.0, 1.0, 0.01);
    ASSERT_TRUE(id);
    auto& modifier = manager.addModifier(host, tree);
    modifier.input_overrides[*id] = 0.8f;
    ASSERT_TRUE(manager.interfaceUpdate(tree, false, *id, {{"label", "Amount"}}));
    ASSERT_TRUE(manager.interfaceMove(tree, false, *id, 0));
    EXPECT_EQ(manager.tree(tree)->group_interface.inputs.front().identifier, *id);
    EXPECT_EQ(manager.tree(tree)->group_interface.inputs.front().label, "Amount");
    EXPECT_EQ(*modifier.input_overrides.at(*id).get_if<float>(), 0.8f);
}

TEST_F(NodesModifierManager, InterfaceRemovalPrunesLinksAndOverridesInSameUndoStep) {
    lfs::vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    const auto host = scene.getScene().getNodeUuid(scene.getScene().addSplat("Host", model()));
    auto& inner = manager.newTree("Inner");
    const auto id = manager.interfaceAdd(inner.uuid, false, "lfs.float", "Strength", 0.5f);
    ASSERT_TRUE(id);
    inner.add_node("lfs.math", "Math");
    ASSERT_TRUE(inner.add_link({inner.input_node().name, *id, "Math", "A"}));
    auto& outer = manager.newTree("Outer");
    outer.add_node("lfs.group", "Group");
    outer.add_node("lfs.value", "Value");
    ASSERT_TRUE(manager.setGroupGraph(outer.uuid, "Group", inner.uuid));
    ASSERT_TRUE(outer.add_link({"Value", "Value", "Group", *id}, nullptr,
                               [&](std::string_view uuid) { return manager.tree(uuid); }));
    auto& modifier = manager.addModifier(host, inner.uuid);
    modifier.input_overrides[*id] = 0.8f;
    auto& history = lfs::vis::op::undoHistory();
    history.clear();
    const auto before = manager.toJson(false);
    ASSERT_TRUE(manager.interfaceRemove(inner.uuid, false, *id));
    EXPECT_EQ(history.undoCount(), 1u);
    EXPECT_EQ(inner.links.size(), 1u);
    EXPECT_EQ(outer.links.size(), 1u);
    EXPECT_TRUE(modifier.input_overrides.empty());
    ASSERT_TRUE(history.undo().success);
    EXPECT_EQ(manager.toJson(false), before);
}

TEST_F(NodesModifierManager, ClipboardIncludesLayoutAndRemapsFrameMembership) {
    lfs::vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    auto& tree = manager.newTree("Test Graph");
    tree.add_node("lfs.note", "Note").properties["text"] = "Two lines\nStay readable";
    tree.add_node("lfs.reroute", "Route");
    const auto frame = manager.frameWrap(tree.uuid, {"Note", "Route"}, "Notes");
    ASSERT_TRUE(frame);
    const auto copy = manager.copyNodes(tree.uuid, {"Note", "Route", *frame});
    ASSERT_TRUE(copy);
    const auto paste = manager.pasteNodes(tree.uuid, *copy);
    ASSERT_TRUE(paste);
    EXPECT_EQ(paste->nodes.size(), 3u);
    const auto* note = tree.find_node("Note 2");
    ASSERT_NE(note, nullptr);
    EXPECT_EQ(note->properties["text"], "Two lines\nStay readable");
    EXPECT_EQ(note->ui["frame"], "Notes 2");
}

TEST_F(NodesModifierManager, DeletingFrameKeepsMembersAndClearsMembership) {
    lfs::vis::SceneManager scene;
    auto& manager = scene.modifierManager();
    auto& tree = colour_tree(manager);
    const auto frame = manager.frameWrap(tree.uuid, {"Correct"});
    ASSERT_TRUE(frame);
    ASSERT_TRUE(tree.remove_node(*frame));
    ASSERT_NE(tree.find_node("Correct"), nullptr);
    EXPECT_FALSE(tree.find_node("Correct")->ui.contains("frame"));
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

// An edit that changes only opacity republishes the other attributes from the previous payload's
// storage where the payload is shared with the renderer, and every published value stays exact.
TEST_F(NodesModifierManager, OpacityEditRepublishesUntouchedAttributes) {
    using lfs::core::Device;
    using lfs::core::Tensor;
    for_each_worker_target([](const Device device) {
        const auto values = [&](const std::vector<std::size_t>& shape, const float scale, const int seed) {
            std::size_t count = 1;
            for (const auto extent : shape)
                count *= extent;
            std::vector<float> data(count);
            for (std::size_t i = 0; i < count; ++i)
                data[i] = scale * std::sin(0.37f * static_cast<float>(i) + static_cast<float>(seed));
            return Tensor::from_vector(data, lfs::core::TensorShape(shape), Device::CPU).to(device);
        };
        lfs::vis::SceneManager scene;
        scene.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        const auto id = scene.getScene().addSplat(
            "Host", std::make_unique<lfs::core::SplatData>(
                        1, values({7, 3}, 2.0f, 1), values({7, 1, 3}, 0.5f, 2), values({7, 3, 3}, 0.3f, 3),
                        values({7, 3}, 0.2f, 4), values({7, 4}, 1.0f, 5), values({7, 1}, 1.0f, 6), 1.0f));
        const auto uuid = scene.getScene().getNodeUuid(id);
        auto& manager = scene.modifierManager();
        auto& tree = manager.newTree("Opacity");
        tree.add_node("lfs.set_opacity", "Opacity").input_values["Opacity"] = 0.25f;
        ASSERT_TRUE(tree.remove_link(
            {tree.input_node().name, "Geometry", tree.output_node().name, "Geometry"}));
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Opacity", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Opacity", "Geometry", tree.output_node().name, "Geometry"}));
        manager.addModifier(uuid, tree.uuid);
        ASSERT_TRUE(manager.evaluate(uuid).ok);
        const auto* node = scene.getScene().getNodeById(id);
        const auto first = node->evaluated_model;
        ASSERT_NE(first, nullptr);

        const auto before = tree.to_json();
        tree.find_node("Opacity")->input_values["Opacity"] = 0.75f;
        manager.recordTreeEdit(tree.uuid, before);
        const auto result = manager.evaluate(uuid);
        ASSERT_TRUE(result.ok);
        const auto second = node->evaluated_model;
        ASSERT_NE(second, nullptr);
        ASSERT_NE(second, first);
        const auto backend = lfs::core::gpu_backend_of(second->means_raw());
        if (backend && lfs::core::splat_publication(*backend) == lfs::core::SplatPublication::Shared) {
            EXPECT_EQ(second->means_raw().data_ptr(), first->means_raw().data_ptr());
            EXPECT_EQ(second->shN_raw().data_ptr(), first->shN_raw().data_ptr());
            EXPECT_NE(second->opacity_raw().data_ptr(), first->opacity_raw().data_ptr());
        }
        const auto expected = lfs::nodes::splat_data_from_geometry(result.geometry);
        const auto expect_equal = [](const Tensor& actual, const Tensor& wanted, const char* name) {
            EXPECT_EQ(actual.cpu().contiguous().to_vector(), wanted.cpu().contiguous().to_vector()) << name;
        };
        expect_equal(second->means_raw(), expected->means_raw(), "means");
        expect_equal(second->sh0_raw(), expected->sh0_raw(), "sh0");
        expect_equal(second->shN_canonical(), expected->shN_canonical(), "shN");
        expect_equal(second->scaling_raw(), expected->scaling_raw(), "scaling");
        expect_equal(second->rotation_raw(), expected->rotation_raw(), "rotation");
        expect_equal(second->opacity_raw(), expected->opacity_raw(), "opacity");
    });
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
