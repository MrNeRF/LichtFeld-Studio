/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/camera.hpp"
#include "training/camera_pose/pose_refinement_session.hpp"
#include "training/trainer.hpp"
#include "visualizer/core/services.hpp"
#include "visualizer/training/training_manager.hpp"
#include <algorithm>
#include <format>
#include <glm/glm.hpp>
#include <optional>

namespace lfs::vis {
    enum class CameraPoseMarker { Diamond, Cross, Square, Pause };

    struct CameraPoseIndicator {
        unsigned int rgb;
        std::string_view symbol;
        CameraPoseMarker marker;
    };

    // A separate palette for optimizer activity, not reconstruction quality.
    inline CameraPoseIndicator cameraPoseIndicator(std::string_view state) {
        if (state == "updated") return {0x45D6B0, "+", CameraPoseMarker::Diamond};
        if (state == "rejected") return {0xFFB454, "!", CameraPoseMarker::Cross};
        if (state == "ready") return {0x65C9F2, "o", CameraPoseMarker::Diamond};
        if (state == "anchor") return {0x83A6FF, "A", CameraPoseMarker::Square};
        if (state == "evaluation") return {0xC5C9D3, "E", CameraPoseMarker::Square};
        if (state == "frozen") return {0xC7A0F2, "=", CameraPoseMarker::Square};
        if (state == "paused") return {0xFFCE73, "=", CameraPoseMarker::Pause};
        return {0x939BAA, ".", CameraPoseMarker::Diamond};
    }

    inline std::string_view cameraPoseVisualState(
        const training::camera_pose::PoseCameraDisplay& pose,
        const training::camera_pose::PoseSessionSnapshot& snapshot) {
        using training::camera_pose::PoseDisplayState;
        if (snapshot.paused && pose.state != PoseDisplayState::Anchor &&
            pose.state != PoseDisplayState::Evaluation && pose.state != PoseDisplayState::Frozen)
            return "paused";
        return training::camera_pose::pose_display_state_name(pose.state);
    }

    inline glm::vec4 cameraPoseIndicatorColor(const CameraPoseIndicator& indicator, float alpha) {
        return {((indicator.rgb >> 16) & 255) / 255.0f,
                ((indicator.rgb >> 8) & 255) / 255.0f,
                (indicator.rgb & 255) / 255.0f, alpha};
    }

    inline std::shared_ptr<const training::camera_pose::PoseSessionSnapshot> activeCameraPoses() {
        const auto* manager = services().trainerOrNull();
        const auto* trainer = manager ? manager->getTrainer() : nullptr;
        return trainer ? trainer->cameraPoseSnapshot() : nullptr;
    }
    // Borrowed only while the caller retains the immutable session snapshot.
    inline const training::camera_pose::PoseCameraDisplay* findCameraPose(
        const training::camera_pose::PoseSessionSnapshot* snapshot, int uid) {
        if (!snapshot)
            return nullptr;
        const auto found = std::lower_bound(snapshot->cameras.begin(), snapshot->cameras.end(), uid,
                                            [](const auto& entry, int id) { return entry.pose.uid < id; });
        return found != snapshot->cameras.end() && found->pose.uid == uid ? &*found : nullptr;
    }

    inline glm::mat4 cameraPoseMatrix(const training::camera_pose::Matrix4& pose) {
        glm::mat4 result(1.0f);
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                result[col][row] = pose[row * 4 + col];
        return result;
    }

    inline std::string cameraPoseDisplacementLabel(const training::camera_pose::PoseCameraDisplay* pose) {
        if (!pose)
            return {};
        return std::format("\u0394 {:.3g} / {:.2f}\u00b0", pose->pose.center_displacement,
                           pose->pose.rotation_displacement * 57.29577951308232);
    }

    // Current poses already live on the host. Unrefined cameras retain the
    // source path; this never rewrites Camera tensors or checkpoint sources.
    inline std::optional<glm::mat4> cameraWorldToCamera(
        const core::Camera& camera, const training::camera_pose::PoseSessionSnapshot* snapshot) {
        if (const auto* pose = findCameraPose(snapshot, camera.uid()))
            return cameraPoseMatrix(pose->pose.current);
        if (!camera.R().is_valid() || !camera.T().is_valid())
            return std::nullopt;
        const auto rotation = camera.R().cpu().contiguous();
        const auto translation = camera.T().cpu().contiguous();
        if (!rotation.is_valid() || !translation.is_valid() ||
            rotation.dtype() != core::DataType::Float32 || translation.dtype() != core::DataType::Float32 ||
            rotation.numel() != 9 || translation.numel() != 3)
            return std::nullopt;
        glm::mat4 result(1.0f);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col)
                result[col][row] = rotation.ptr<float>()[row * 3 + col];
            result[3][row] = translation.ptr<float>()[row];
        }
        return result;
    }
} // namespace lfs::vis
