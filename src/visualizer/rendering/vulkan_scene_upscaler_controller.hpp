/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "rendering/vulkan_scene_upscaler_adapter.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace lfs::vis {
    class VulkanContext;

    enum class VulkanSceneUpscalerFailure : std::uint8_t {
        None = 0,
        Unavailable,
        WrongAdapterType,
        Initialization,
        InvalidDispatch,
        Record,
        InvalidOutput,
    };

    struct VulkanSceneUpscalerControllerStatus {
        std::string requested_id = "native";
        std::string active_id;
        SceneUpscalerAvailabilityReason availability =
            SceneUpscalerAvailabilityReason::NotCompiled;
        VulkanSceneUpscalerFailure failure = VulkanSceneUpscalerFailure::None;
        std::uint64_t generation = 0;
        std::uint64_t evaluation_count = 0;
        std::uint64_t fallback_count = 0;

        [[nodiscard]] bool active() const {
            return !active_id.empty() && failure == VulkanSceneUpscalerFailure::None;
        }
    };

    class LFS_VIS_API VulkanSceneUpscalerController {
    public:
        explicit VulkanSceneUpscalerController(OptionalSceneUpscalerRegistry& registry);
        ~VulkanSceneUpscalerController();

        VulkanSceneUpscalerController(const VulkanSceneUpscalerController&) = delete;
        VulkanSceneUpscalerController& operator=(const VulkanSceneUpscalerController&) = delete;
        VulkanSceneUpscalerController(VulkanSceneUpscalerController&&) noexcept = default;
        VulkanSceneUpscalerController& operator=(VulkanSceneUpscalerController&&) noexcept = default;

        [[nodiscard]] bool select(std::string_view id,
                                  const SceneUpscalerProbeContext& probe_context,
                                  VulkanContext& context) noexcept;
        [[nodiscard]] bool record(VkCommandBuffer command_buffer,
                                  const VulkanSceneUpscalerDispatch& dispatch) noexcept;
        [[nodiscard]] VulkanSceneUpscalerOutput output(TemporalViewId view) const noexcept;
        void reset(TemporalViewId view, TemporalResetReason reasons) noexcept;
        void deactivate() noexcept;

        [[nodiscard]] const VulkanSceneUpscalerControllerStatus& status() const noexcept {
            return status_;
        }
        [[nodiscard]] const SceneUpscalerRequirements& requirements() const noexcept {
            return requirements_;
        }

    private:
        void fail(VulkanSceneUpscalerFailure failure,
                  SceneUpscalerAvailabilityReason availability) noexcept;

        OptionalSceneUpscalerRegistry* registry_ = nullptr;
        std::unique_ptr<VulkanSceneUpscalerAdapter> adapter_;
        SceneUpscalerRequirements requirements_{};
        VulkanSceneUpscalerControllerStatus status_{};
    };
} // namespace lfs::vis
