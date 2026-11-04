/* SPDX-License-Identifier: GPL-3.0-or-later */
// Small standalone contract test linked against the actual lfs_core DLL.
#include "core/scene.hpp"
#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <stdexcept>

int main() try {
    using namespace lfs::core;
    Scene scene;
    const auto parent = scene.addGroup("container");
    const auto remote = scene.addSplatPlaceholder("external", parent);
    auto* node = scene.getNodeById(remote);
    node->transient = true;
    node->external_training.emplace();
    node->locked.setQuiet(true);
    auto require = [](bool result) {
        if (!result)
            throw std::runtime_error("Read-only scene contract failed");
    };
    auto rejects = [&](auto&& action) {
        bool rejected = false;
        try {
            action();
        } catch (const std::runtime_error&) { rejected = true; }
        require(rejected);
    };
    require(scene.hasExternalTraining() && scene.hasExternalTraining(parent));
    require(scene.isNodeEffectivelyLocked(remote));
    rejects([&] { scene.setNodeLocked("external", false); });
    rejects([&] { scene.removeNodeById(remote); });
    rejects([&] { scene.removeNodeById(parent); });
    rejects([&] { (void)scene.detachSplatModelsForRemoval(parent); });
    rejects([&] { scene.setTrainingModelNode(remote); });
    rejects([&] { (void)scene.swapNodeModel("external", nullptr); });
    require(!scene.renameNode(remote, "renamed"));
    require(scene.duplicateNode("external").empty());
    const auto transform = glm::translate(glm::mat4{1}, glm::vec3{7, 8, 9});
    scene.setNodeTransform(remote, transform);
    require(scene.getNodeTransform(remote) == glm::mat4{1});
    node->local_transform.set(transform, false); // Generic property path.
    require(scene.getNodeTransform(remote) == glm::mat4{1});
    node->locked.set(false, false);
    require(static_cast<bool>(node->locked));
    require(scene.addCropBox("crop", remote) == NULL_NODE);
    require(scene.addEllipsoid("ellipsoid", remote) == NULL_NODE);
    scene.setNodeVisibility(remote, false);
    require(!static_cast<bool>(node->visible));
    const auto local = scene.addGroup("local");
    scene.setNodeTransform(local, transform);
    require(scene.getNodeTransform(local) == transform);
    require(scene.renameNode(local, "editable"));
    scene.removeNodeById(local);
    require(!scene.getNodeById(local));
    require(scene.getNodeById(remote) != nullptr);
    const auto owner_uuid = node->uuid.to_string();
    const auto preview_root = scene.addGroup("preview owner");
    auto* root = scene.getNodeById(preview_root);
    root->transient = true;
    root->external_training.emplace();
    root->external_training->owner_uuid = owner_uuid;
    root->locked.setQuiet(true);
    const auto unrelated = scene.addGroup("unrelated local group");
    require(scene.addGroup("forbidden child", preview_root) == NULL_NODE);
    require(scene.addSplatPlaceholder("forbidden model", preview_root) == NULL_NODE);
    require(scene.addCameraGroup("forbidden cameras", preview_root, 1) == NULL_NODE);
    require(!scene.reparent(unrelated, preview_root));
    require(!scene.moveNode(unrelated, preview_root, 0));
    require(scene.getNodeById(unrelated)->parent_id == NULL_NODE);
    const auto camera_group = scene.addCameraGroup("owned cameras", preview_root, 1, owner_uuid);
    require(camera_group != NULL_NODE);
    const auto remote_camera = scene.addSplatPlaceholder("external camera");
    auto* camera_node = scene.getNodeById(remote_camera);
    // No camera payload is needed to test selection/ownership state itself.
    camera_node->type = NodeType::CAMERA;
    camera_node->transient = true;
    camera_node->training_enabled = false;
    camera_node->external_training.emplace();
    camera_node->external_training->owner_uuid = owner_uuid;
    require(scene.reparent(remote_camera, camera_group));
    camera_node->locked.setQuiet(true);
    scene.setCameraTrainingEnabled(remote_camera, true);
    require(!camera_node->training_enabled);
    require(!scene.setCameraTrainingSelection({remote_camera}));
    require(!scene.setCameraGroupTrainingSelection(camera_group, {remote_camera}));
    scene.setAllCameraTrainingEnabled(true);
    require(!camera_node->training_enabled);
    scene.invertAllCameraTrainingEnabled();
    require(!camera_node->training_enabled);
    scene.setAllCameraTrainingEnabled(false);
    require(!camera_node->training_enabled);
    scene.clear(); // Project replacement is allowed; per-node edits are not.
    require(!scene.hasExternalTraining());
    require(scene.allocateCameraUid() == 0);
    require(scene.allocateCameraUid() == 1);
    auto staged = Scene::createRestoreStage(scene);
    auto outgoing = scene.commitRestoreStage(std::move(staged));
    require(scene.allocateCameraUid() == 2); // A restore cannot revive stale UID-keyed caches.
    std::cout << "READ_ONLY_SCENE_CORE_OK\n";
    return 0;
} catch (const std::exception& error) {
    return std::fprintf(stderr, "%s\n", error.what()) < 0 ? 2 : 1;
} catch (...) {
    return 1;
}
