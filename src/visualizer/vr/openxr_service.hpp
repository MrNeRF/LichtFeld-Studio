/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "rendering/frame_contract.hpp"
#include "rendering/rendering_manager.hpp"
#include "window/vulkan_context.hpp"
#include <array>
#include <functional>
#include <memory>
#include <string>

namespace lfs::vis::vr {

    // Headset viewing through OpenXR. The instance is created once at startup
    // when VR support is on; a session exists only while viewing in VR.
    class OpenXrService {
    public:
        using EyeFrames = std::array<RenderingManager::VulkanFrameResult, 2>;
        using RenderEyes = std::function<lfs::Result<EyeFrames>(const std::array<lfs::rendering::FrameView, 2>&)>;

        OpenXrService();
        ~OpenXrService();
        OpenXrService(const OpenXrService&) = delete;
        OpenXrService& operator=(const OpenXrService&) = delete;

        // Finds the runtime and headset; the status explains a failure.
        bool prepare();
        // Vulkan instance and device creation through the runtime; valid while available().
        [[nodiscard]] VulkanDeviceHooks deviceHooks();
        void setUnavailable(std::string reason);
        [[nodiscard]] bool available() const;
        [[nodiscard]] bool active() const;
        // The runtime has begun the session and paces frames.
        [[nodiscard]] bool running() const;
        // Localization key of the status line and its argument.
        [[nodiscard]] const char* statusKey() const;
        [[nodiscard]] const std::string& statusDetail() const;

        // Reads the VR preferences; the view starts at the desktop camera.
        lfs::Status start(VulkanContext& context, const glm::mat3& rotation, const glm::vec3& position);
        void stop();
        void recenter();
        void snapTurn(int direction);
        void setMoveSpeed(float meters_per_second);
        // move: keyboard axes (+x right, +y up, +z back). Returns false once the
        // runtime ends the session; a failure stops it.
        lfs::Result<bool> frame(glm::vec3 move, const RenderEyes& render);
        // Left eye of the last submitted frame, for the desktop window.
        [[nodiscard]] const RenderingManager::VulkanFrameResult& mirror() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis::vr
