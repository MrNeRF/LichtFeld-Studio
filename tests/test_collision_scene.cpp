/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/events.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "input/input_bindings.hpp"
#include "input/input_controller.hpp"
#include "input/key_codes.hpp"
#include "io/project_document.hpp"
#include "io/scene_chapter_adapter.hpp"
#include "operation/undo_entry.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/viewport_request_builder.hpp"
#include "scene/scene_manager.hpp"
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

namespace {
    using namespace lfs::core;
    using namespace lfs::io::project;
    std::shared_ptr<const CollisionVolume> volume() {
        auto g = std::make_shared<CollisionVolume>();
        g->settings.applyPreset(CollisionPreset::Object);
        g->settings.voxel_size = 0.1f;
        g->dimensions = {30, 30, 30};
        g->solid.resize(27'000);
        for (int z = 0; z < 30; ++z)
            for (int x = 0; x < 30; ++x)
                g->solid[g->index({x, 0, z})] = 1;
        g->tree = g->octreeOccupancy();
        g->tree_size = 32;
        g->solid.clear();
        return g;
    }
    std::shared_ptr<MeshData> mesh(const CollisionVolume& v) {
        auto surface = v.mesh(CollisionShape::Faces);
        auto result = std::make_shared<MeshData>();
        result->vertices = Tensor::from_vector(surface.vertices, {surface.vertices.size() / 3, 3}, Device::CPU);
        result->indices = Tensor::from_vector(surface.indices, {surface.indices.size() / 3, 3}, Device::CPU);
        result->compute_normals();
        return result;
    }
} // namespace
TEST(CollisionScene, ArchiveRestoresGeneratedMeshAndVolumeWithDeferredHydration) {
    Scene scene;
    auto boxes = std::make_shared<CollisionVolume>(*volume());
    boxes->show_voxels = true;
    std::shared_ptr<const CollisionVolume> g = boxes;
    auto geometry = mesh(*g);
    const auto group = scene.addGroup("collision group");
    auto id = scene.addMesh("Collision", geometry, group);
    auto* node = scene.getNodeById(id);
    const auto uuid = node->uuid;
    node->collision = g;
    node->payload_diverged = true;
    node->visible = false;
    scene.setNodeTransform(group, glm::translate(glm::mat4(1), glm::vec3(4, 5, 6)));
    scene.setNodeTransform(id, glm::rotate(glm::mat4(1), glm::radians(45.0f), glm::vec3(0, 1, 0)));
    auto world = scene.getWorldTransform(id);
    ScenePayloadBindings bindings{{uuid, {.fourcc = "MESH", .instance_uuid = uuid, .source_kind = "mesh"}}};
    auto captured = capture_scene_graph(scene, bindings);
    ASSERT_TRUE(captured);
    auto document = ProjectDocument::create(generate_uuid_v4());
    ASSERT_TRUE(document);
    document->edit_scene_graph() = std::move(*captured);
    ASSERT_TRUE(document->set_mesh(uuid, MeshPayload(geometry)));
    ASSERT_TRUE(document->edit_project().upsert_embedded_payload_provenance({
        .uuid = uuid,
        .node_uuid = uuid,
        .fourcc = "MESH",
        .import_locator = {.preferred = "generated:" + uuid.to_string(), .base = LocatorBase::Absolute},
        .import_fingerprint = {},
        .content_xxh3_128 = {},
    }));
    ASSERT_TRUE(document->edit_project().upsert_embed_decision({
        .uuid = uuid,
        .node_uuid = uuid,
        .payload_fourcc = "MESH",
        .decision = "embedded",
        .reference_uuid = std::nullopt,
        .reason = "project-owned scene payload",
    }));
    const auto path = std::filesystem::temp_directory_path() /
                      ("lichtfeld-collision-" + generate_uuid_v4().to_string() + ".licht");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } cleanup{path};
    ProjectDocumentSaveOptions options;
    options.file_uuid = generate_uuid_v4();
    options.commit.commit_uuid = generate_uuid_v4();
    options.commit.snapshot_uuid = generate_uuid_v4();
    auto saved = document->save(path, options);
    ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
    auto opened = ProjectDocument::open(path, {.defer_geometry_payloads = true});
    ASSERT_TRUE(opened) << lfs::format_for_developer(opened.error());
    Scene restored;
    auto shell = stage_scene_shell(opened->scene_graph(), restored);
    ASSERT_TRUE(shell);
    auto old = restored.commitRestoreStage(std::move(*shell));
    auto* shell_node = restored.getNodeByUuid(uuid);
    ASSERT_NE(shell_node, nullptr);
    ASSERT_TRUE(shell_node->collision);
    EXPECT_EQ(*shell_node->collision, *g);
    EXPECT_FALSE(shell_node->mesh);
    auto plan = opened->stage_hydration(restored);
    ASSERT_TRUE(plan) << lfs::format_for_developer(plan.error());
    auto report = ProjectDocument::commit_partial_hydration(restored, std::move(*plan), true);
    auto* loaded = restored.getNodeByUuid(uuid);
    ASSERT_NE(loaded, nullptr);
    ASSERT_TRUE(loaded->mesh);
    ASSERT_TRUE(loaded->collision);
    EXPECT_EQ(*loaded->collision, *g);
    EXPECT_EQ(loaded->mesh->vertices.to_vector(), geometry->vertices.to_vector());
    EXPECT_EQ(loaded->mesh->indices.to_vector_int(), geometry->indices.to_vector_int());
    EXPECT_FALSE(loaded->visible.get());
    EXPECT_TRUE(loaded->payload_diverged);
    EXPECT_EQ(restored.getWorldTransform(loaded->id), world);
    EXPECT_GT(report.hydrated_payload_units, 0u);
    EXPECT_TRUE(restored.getVisibleMeshes().empty());
    loaded->visible = true;
    lfs::vis::SceneRenderState state;
    state.meshes = restored.getVisibleMeshes();
    ASSERT_EQ(state.meshes.size(), 1u);
    auto draws = lfs::vis::buildViewportMeshDrawItems(state, {}, {0, 0, 1});
    ASSERT_EQ(draws.size(), 1u);
    EXPECT_TRUE(draws[0].collision_voxel_boxes);
    EXPECT_EQ(draws[0].collision_volume, loaded->collision.get());
    EXPECT_EQ(draws[0].collision_volume->gpuOccupancy(), g->gpuOccupancy());
    EXPECT_FALSE(draws[0].shadow_enabled);
    EXPECT_EQ(draws[0].model, world);
}
TEST(CollisionScene, DuplicateAndDeleteUndoPreserveCollisionPayload) {
    lfs::vis::SceneManager manager;
    auto& scene = manager.getScene();
    auto g = volume();
    auto id = scene.addMesh("Collision", mesh(*g));
    scene.getNodeById(id)->collision = g;
    auto uuid = scene.getNodeUuid(id);
    auto copy_name = manager.duplicateNodeTree(id);
    auto* copy = scene.getNode(copy_name);
    ASSERT_NE(copy, nullptr);
    ASSERT_TRUE(copy->collision);
    EXPECT_EQ(*copy->collision, *g);
    auto before = lfs::vis::op::SceneGraphPatchEntry::captureStateByIds(manager, {id});
    scene.removeNodeById(id);
    auto after = lfs::vis::op::SceneGraphPatchEntry::captureStateByIds(manager, {});
    lfs::vis::op::SceneGraphPatchEntry entry(manager, "Delete collision", std::move(before), std::move(after));
    entry.undo();
    auto* restored = scene.getNodeByUuid(uuid);
    ASSERT_NE(restored, nullptr);
    ASSERT_TRUE(restored->mesh);
    ASSERT_TRUE(restored->collision);
    EXPECT_EQ(*restored->collision, *g);
    entry.redo();
    EXPECT_EQ(scene.getNodeByUuid(uuid), nullptr);
}
TEST(CollisionScene, GraphEpochChangesAcrossReloadWithSameUuids) {
    Scene scene;
    auto id = scene.addGroup("group");
    auto uuid = scene.getNodeUuid(id);
    auto captured = capture_scene_graph(scene, {});
    ASSERT_TRUE(captured);
    auto epoch = scene.graphEpoch();
    ASSERT_TRUE(hydrate_scene_graph(*captured, scene, {}));
    EXPECT_GT(scene.graphEpoch(), epoch);
    EXPECT_NE(scene.getNodeByUuid(uuid), nullptr);
}
TEST(CollisionWalkController, RejectsMissingCollisionAndExitsOnTeleportResetOrEscape) {
    lfs::vis::input::InputBindings::setPersistenceEnabled(false);
    lfs::vis::SceneManager manager;
    lfs::vis::RenderingManager rendering;
    lfs::vis::services().set(&manager);
    lfs::vis::services().set(&rendering);
    struct Cleanup {
        ~Cleanup() {
            lfs::vis::services().clear();
            lfs::vis::input::InputBindings::setPersistenceEnabled(true);
        }
    } cleanup;
    Viewport viewport;
    viewport.camera.t = {1, 1.9f, -1};
    lfs::vis::InputController input(nullptr, viewport);
    using Mode = lfs::vis::InputController::CameraNavigationMode;
    input.setCameraNavigationMode(Mode::Walk);
    EXPECT_NE(input.cameraNavigationMode(), Mode::Walk);
    EXPECT_FALSE(input.walkError().empty());
    auto g = volume();
    auto id = manager.getScene().addMesh("Collision", mesh(*g));
    manager.getScene().getNodeById(id)->collision = g;
    // Map grid +Y to visualizer +Y for the test's native scene transform.
    manager.getScene().setNodeTransform(id, glm::scale(glm::mat4(1), glm::vec3(1, -1, -1)));
    input.setCameraNavigationMode(Mode::Walk);
    ASSERT_EQ(input.cameraNavigationMode(), Mode::Walk);
    viewport.camera.startGlide({100, 100, 100});
    input.update(0.01f);
    EXPECT_LT(glm::length(viewport.camera.t), 5);
    viewport.camera.t += glm::vec3(1, 0, 0);
    input.update(0.01f);
    EXPECT_EQ(input.cameraNavigationMode(), Mode::FPV);
    input.setCameraNavigationMode(Mode::Walk);
    ASSERT_EQ(input.cameraNavigationMode(), Mode::Walk);
    input.goHome();
    EXPECT_EQ(input.cameraNavigationMode(), Mode::FPV);
    viewport.camera.t = {1, 1.9f, 1};
    input.setCameraNavigationMode(Mode::Walk);
    ASSERT_EQ(input.cameraNavigationMode(), Mode::Walk);
    input.handleKey(lfs::vis::input::KEY_ESCAPE, lfs::vis::input::ACTION_PRESS, 0);
    EXPECT_EQ(input.cameraNavigationMode(), Mode::FPV);
}
