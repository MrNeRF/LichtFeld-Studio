/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/events.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "input/input_bindings.hpp"
#include "input/input_controller.hpp"
#include "input/input_router.hpp"
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
    node->collision_display = {.preview = true, .sharp = false, .surface_ready = false, .opacity = 0.45f, .slice = 0.8f, .surface_stride = 8};
    const auto display = node->collision_display;
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
    EXPECT_EQ(loaded->collision_display, display);
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

TEST(CollisionWalkController, AvailabilityTracksVisibilityHydrationAndTransforms) {
    lfs::vis::SceneManager manager;
    lfs::vis::services().set(&manager);
    struct Cleanup {
        ~Cleanup() { lfs::vis::services().clear(); }
    } cleanup;
    Viewport viewport;
    lfs::vis::InputController input(nullptr, viewport);
    EXPECT_FALSE(input.hasWalkCollision());
    auto& scene = manager.getScene();
    const auto g = volume();
    const auto geometry = mesh(*g);
    const auto group = scene.addGroup("parent");
    const auto id = scene.addMesh("collision", geometry, group);
    auto* node = scene.getNodeById(id);
    EXPECT_FALSE(input.hasWalkCollision()); // Ordinary surfaces are not colliders.
    node->collision = g;
    EXPECT_TRUE(input.hasWalkCollision());
    node->visible = false;
    EXPECT_FALSE(input.hasWalkCollision());
    node->visible = true;
    scene.getNodeById(group)->visible = false;
    EXPECT_FALSE(input.hasWalkCollision());
    scene.getNodeById(group)->visible = true;
    node->mesh.reset();
    EXPECT_FALSE(input.hasWalkCollision()); // Deferred hydration is not ready yet.
    node->mesh = geometry;
    EXPECT_TRUE(input.hasWalkCollision());
    scene.setNodeTransform(group, glm::scale(glm::mat4(1), glm::vec3(0, 1, 1)));
    EXPECT_FALSE(input.hasWalkCollision());
    scene.setNodeTransform(group, glm::mat4(1));
    node->locked = true;
    EXPECT_TRUE(input.hasWalkCollision()); // Edit locks do not disable physics.
    scene.removeNode("collision", false);
    EXPECT_FALSE(input.hasWalkCollision());
}

TEST(CollisionWalkController, LiveCapsuleChangesKeepFeetGroundedRejectCeilingsAndIgnoreZoom) {
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
    auto g = std::make_shared<CollisionVolume>(*volume());
    g->solid.assign(27'000, 0);
    g->tree.clear();
    g->tree_size = 0;
    for (int z = 0; z < 30; ++z)
        for (int x = 0; x < 30; ++x) {
            g->solid[g->index({x, 0, z})] = 1;
            g->solid[g->index({x, 25, z})] = 1;
        }
    const auto id = manager.getScene().addMesh("floor and ceiling", mesh(*g));
    manager.getScene().getNodeById(id)->collision = g;
    manager.getScene().setNodeTransform(id, glm::scale(glm::mat4(1), glm::vec3(1, -1, -1)));
    Viewport viewport;
    viewport.camera.t = {1, 1.9f, 1};
    lfs::vis::InputController input(nullptr, viewport);
    input.initialize();
    using Mode = lfs::vis::InputController::CameraNavigationMode;
    input.setCameraNavigationMode(Mode::Walk);
    ASSERT_EQ(input.cameraNavigationMode(), Mode::Walk);
    auto settings = input.walkSettings();
    settings.height = 2.1f;
    settings.radius = .15f;
    settings.speed = 4;
    settings.step_height = .35f;
    input.setWalkSettings(settings);
    EXPECT_NEAR(viewport.camera.t.y, 2.2f, .0002f); // includes the solver's contact skin
    EXPECT_FLOAT_EQ(input.walkSettings().radius, .15f);
    EXPECT_FLOAT_EQ(input.walkSettings().speed, 4);
    EXPECT_FLOAT_EQ(input.walkSettings().step_height, .35f);
    const auto resized = viewport.camera.t;
    input.handleScroll(0, 5);
    input.handleScroll(0, -5);
    EXPECT_EQ(viewport.camera.t, resized);
    input.update(.01f);
    EXPECT_EQ(input.cameraNavigationMode(), Mode::Walk);
    EXPECT_NEAR(viewport.camera.t.y - settings.height, .1f, .001f);
    const auto safe_eye = viewport.camera.t;
    settings.height = 2.7f;
    EXPECT_THROW(input.setWalkSettings(settings), std::runtime_error);
    EXPECT_FLOAT_EQ(input.walkSettings().height, 2.1f);
    EXPECT_EQ(viewport.camera.t, safe_eye);
    EXPECT_EQ(input.cameraNavigationMode(), Mode::Walk);
    settings.height = 1.4f;
    settings.radius = .1f;
    input.setWalkSettings(settings);
    input.update(.01f);
    EXPECT_NEAR(viewport.camera.t.y - settings.height, .1f, .001f);
    EXPECT_EQ(input.cameraNavigationMode(), Mode::Walk);
}

TEST(CollisionScene, PreviewControlsShareOccupancyAndSurviveSceneLifecycle) {
    Scene scene;
    const auto g = volume();
    const auto id = scene.addMesh("Voxel preview", mesh(*g));
    auto* node = scene.getNodeById(id);
    node->collision = g;
    node->collision_display = {.preview = true, .surface_ready = false, .opacity = 0.4f, .slice = 0.35f, .surface_stride = 8};
    const auto uuid = node->uuid;
    const auto* data = g->tree.data();
    ScenePayloadBindings bindings{{uuid, {.fourcc = "MESH", .instance_uuid = uuid, .source_kind = "mesh"}}};
    auto graph = capture_scene_graph(scene, bindings);
    ASSERT_TRUE(graph);
    const auto found = graph->find(uuid);
    ASSERT_TRUE(found);
    ASSERT_TRUE(*found);
    const auto& record = **found;
    ASSERT_TRUE(record.collision);
    const auto copy = record;
    EXPECT_EQ(copy.collision.get(), record.collision.get());
    EXPECT_EQ(copy.collision->tree.data(), record.collision->tree.data());
    EXPECT_EQ(record.collision_display, node->collision_display);
    const auto encoded = graph->to_bytes();
    auto parsed = SceneGraphChapter::from_bytes(encoded);
    ASSERT_TRUE(parsed);
    const auto restored = parsed->find(uuid);
    ASSERT_TRUE(restored);
    ASSERT_TRUE(*restored);
    EXPECT_EQ((**restored).collision_display, node->collision_display);
    EXPECT_EQ(*(**restored).collision, *g);
    EXPECT_EQ(node->collision->tree.data(), data);
}

TEST(CollisionScene, GeometrySnapshotsOwnOnlyVisibleGeometryAndSurviveEdits) {
    Scene scene;
    auto make = [] {
        return std::make_unique<SplatData>(0,
                                           Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, Device::CPU),
                                           Tensor::zeros({2, 1, 3}, Device::CPU), Tensor::zeros({2, 0, 3}, Device::CPU),
                                           Tensor::zeros({2, 3}, Device::CPU), Tensor::zeros({2, 4}, Device::CPU),
                                           Tensor::zeros({2, 1}, Device::CPU), 1.0f);
    };
    const auto visible = scene.addSplat("Visible", make());
    const auto hidden = scene.addSplat("Hidden", make());
    scene.getNodeById(hidden)->visible = false;
    const auto transform = glm::translate(glm::mat4(1), glm::vec3(2, 3, 4));
    scene.setNodeTransform(visible, transform);
    auto snapshot = scene.snapshotVisibleSplatGeometry();
    ASSERT_EQ(snapshot.size(), 1);
    EXPECT_EQ(snapshot[0].size, 2);
    EXPECT_EQ(snapshot[0].world_transform, transform);
    EXPECT_EQ(snapshot[0].means.device(), Device::CPU);
    EXPECT_EQ(snapshot[0].means.ptr<float>()[0], 1);
    scene.getNodeById(visible)->model->means_raw().ptr<float>()[0] = 99;
    EXPECT_EQ(snapshot[0].means.ptr<float>()[0], 1);
    scene.removeNodeById(visible);
    EXPECT_EQ(snapshot[0].means.ptr<float>()[5], 6);
}

TEST(CollisionScene, InFlightDrawKeepsReplacedAndDeletedCollisionPayloadsAlive) {
    Scene scene;
    auto old_volume = volume();
    auto old_mesh = mesh(*old_volume);
    std::weak_ptr<const CollisionVolume> weak_volume = old_volume;
    std::weak_ptr<const MeshData> weak_mesh = old_mesh;
    const auto id = scene.addMesh("Collision", old_mesh);
    auto* node = scene.getNodeById(id);
    node->collision = old_volume;
    lfs::vis::SceneRenderState snapshot;
    snapshot.meshes = scene.getVisibleMeshes();
    auto draws = lfs::vis::buildViewportMeshDrawItems(snapshot, {}, {0, 0, 1});
    ASSERT_EQ(draws.size(), 1);
    EXPECT_EQ(draws[0].collision_owner.get(), old_volume.get());
    EXPECT_EQ(draws[0].mesh_owner.get(), old_mesh.get());
    node->collision = volume();
    node->mesh = mesh(*node->collision);
    snapshot.meshes.clear();
    old_volume.reset();
    old_mesh.reset();
    scene.removeNodeById(id);
    ASSERT_FALSE(weak_volume.expired());
    ASSERT_FALSE(weak_mesh.expired());
    EXPECT_NO_THROW(draws[0].collision_volume->validate());
    EXPECT_GT(draws[0].mesh->face_count(), 0);
    draws.clear();
    EXPECT_TRUE(weak_volume.expired());
    EXPECT_TRUE(weak_mesh.expired());
}

TEST(CollisionScene, DisplayScrubsAvoidGaussianCacheInvalidation) {
    static int full, composite;
    full = composite = 0;
    Scene scene;
    scene.setRenderInvalidationCallback(+[] { ++full; });
    scene.setTransformInvalidationCallback(+[] { ++composite; });
    const auto id = scene.addMesh("Collision", mesh(*volume()));
    auto* node = scene.getNodeById(id);
    node->collision = volume();
    const auto initial_full = full, initial_composite = composite;
    const auto generation = scene.renderGeneration();
    const auto* occupancy = node->collision->tree.data();
    node->collision_display.opacity = 0.4f;
    node->collision_display.slice = 0.6f;
    scene.notifyMutation(Scene::MutationType::DISPLAY_CHANGED);
    EXPECT_EQ(full, initial_full);
    EXPECT_EQ(composite, initial_composite + 1);
    EXPECT_GT(scene.renderGeneration(), generation);
    EXPECT_EQ(node->collision->tree.data(), occupancy);
}

TEST(CollisionWalkController, SmoothedEyeDoesNotTeleportOrReplaceBodyAndKeepsAnimationAlive) {
    using namespace lfs::core;
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
    auto g = std::make_shared<CollisionVolume>();
    g->settings.voxel_size = .05f;
    g->dimensions = {60, 60, 60};
    g->solid.resize(60 * 60 * 60);
    for (int z = 0; z < 60; ++z)
        for (int x = 0; x < 60; ++x) {
            g->solid[g->index({x, 0, z})] = 1;
            g->solid[g->index({x, 1, z})] = 1;
            if (z >= 20 && z < 30)
                g->solid[g->index({x, 2, z})] = 1;
        }
    const auto id = manager.getScene().addMesh("Walk bump", mesh(*g));
    manager.getScene().getNodeById(id)->collision = g;
    manager.getScene().setNodeTransform(id, glm::scale(glm::mat4(1), glm::vec3(1, -1, -1)));
    Viewport viewport;
    viewport.camera.t = {1, 1.9f, 2};
    viewport.camera.R = glm::mat3(1);
    lfs::vis::InputController input(nullptr, viewport);
    input.initialize();
    lfs::vis::input::InputRouter router;
    router.setInputController(&input);
    input.setInputRouter(&router);
    router.focusViewportKeyboard();
    using Mode = lfs::vis::InputController::CameraNavigationMode;
    input.setCameraNavigationMode(Mode::Walk);
    ASSERT_EQ(input.cameraNavigationMode(), Mode::Walk);
    // Physical events work without polling SDL's keyboard in this windowless fixture.
    input.handleKey(lfs::vis::input::KEY_W, lfs::vis::input::KEY_UNKNOWN, 0, lfs::vis::input::ACTION_PRESS, 0);
    bool climbed = false;
    for (int i = 0; i < 50; ++i) {
        const auto previous = viewport.camera.t;
        input.update(1.f / 60);
        ASSERT_EQ(input.cameraNavigationMode(), Mode::Walk) << input.walkError();
        EXPECT_LT(viewport.camera.t.y - previous.y, .02f);
        if (viewport.camera.t.y > 1.9005f) {
            climbed = true;
            break;
        }
    }
    ASSERT_TRUE(climbed) << "eye=" << viewport.camera.t.x << "," << viewport.camera.t.y << "," << viewport.camera.t.z
                         << " up=" << viewport.camera.navigationUp().y << " error=" << input.walkError();
    input.handleKey(lfs::vis::input::KEY_W, lfs::vis::input::KEY_UNKNOWN, 0, lfs::vis::input::ACTION_RELEASE, 0);
    EXPECT_TRUE(input.needsCameraAnimationFrame());
    const auto released = viewport.camera.t;
    for (int i = 0; i < 120; ++i) {
        input.update(1.f / 60);
        ASSERT_EQ(input.cameraNavigationMode(), Mode::Walk) << input.walkError();
    }
    EXPECT_GT(viewport.camera.t.y, released.y);
    EXPECT_NEAR(viewport.camera.t.z, released.z, .0001f);
    EXPECT_FALSE(input.needsCameraAnimationFrame());
    viewport.camera.t.x += .5f;
    input.update(1.f / 60);
    EXPECT_EQ(input.cameraNavigationMode(), Mode::FPV);
}
