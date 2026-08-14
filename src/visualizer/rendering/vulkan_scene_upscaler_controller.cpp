/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_upscaler_controller.hpp"

#include "window/vulkan_context.hpp"

namespace lfs::vis {
    VulkanSceneUpscalerController::VulkanSceneUpscalerController(
        OptionalSceneUpscalerRegistry& registry)
        : registry_(&registry) {}

    VulkanSceneUpscalerController::~VulkanSceneUpscalerController() {
        deactivate();
    }

    bool VulkanSceneUpscalerController::select(
        const std::string_view id,
        const SceneUpscalerProbeContext& probe_context,
        VulkanContext& context) noexcept {
        if (sceneUpscalerBackendFromId(id)) {
            status_.failure = VulkanSceneUpscalerFailure::None;
            deactivate();
            status_.requested_id = id;
            status_.availability = SceneUpscalerAvailabilityReason::Ready;
            return false;
        }
        if (adapter_ && status_.active_id == id) {
            return true;
        }
        if (!adapter_ && status_.requested_id == id &&
            status_.failure != VulkanSceneUpscalerFailure::None) {
            return false;
        }

        deactivate();
        status_.requested_id = id;
        status_.failure = VulkanSceneUpscalerFailure::None;
        const auto descriptor = registry_->descriptor(id);
        if (!descriptor) {
            fail(VulkanSceneUpscalerFailure::Unavailable,
                 SceneUpscalerAvailabilityReason::NotCompiled);
            return false;
        }
        requirements_ = descriptor->requirements;

        auto candidate_result = registry_->createAvailableResult(id, probe_context);
        if (!candidate_result) {
            fail(VulkanSceneUpscalerFailure::Unavailable, candidate_result.error());
            return false;
        }
        auto candidate = std::move(*candidate_result);
        auto* const vulkan_adapter = dynamic_cast<VulkanSceneUpscalerAdapter*>(candidate.get());
        if (!vulkan_adapter) {
            fail(VulkanSceneUpscalerFailure::WrongAdapterType,
                 SceneUpscalerAvailabilityReason::ProbeFailed);
            return false;
        }

        candidate.release();
        adapter_.reset(vulkan_adapter);
        const auto availability = adapter_->initialize(context);
        if (!availability.available()) {
            fail(VulkanSceneUpscalerFailure::Initialization, availability.reason);
            return false;
        }

        status_.active_id = id;
        status_.availability = SceneUpscalerAvailabilityReason::Ready;
        ++status_.generation;
        return true;
    }

    bool VulkanSceneUpscalerController::record(
        const VkCommandBuffer command_buffer,
        const VulkanSceneUpscalerDispatch& dispatch) noexcept {
        if (!adapter_ || command_buffer == VK_NULL_HANDLE || !dispatch.valid(requirements_)) {
            fail(VulkanSceneUpscalerFailure::InvalidDispatch,
                 SceneUpscalerAvailabilityReason::ProbeFailed);
            return false;
        }
        if (dispatch.reset_reasons != TemporalResetReason::None) {
            adapter_->reset(dispatch.view, dispatch.reset_reasons);
        }
        if (!adapter_->record(command_buffer, dispatch)) {
            fail(VulkanSceneUpscalerFailure::Record,
                 SceneUpscalerAvailabilityReason::ProbeFailed);
            return false;
        }
        if (!adapter_->output(dispatch.view).valid(dispatch.output_extent)) {
            fail(VulkanSceneUpscalerFailure::InvalidOutput,
                 SceneUpscalerAvailabilityReason::ProbeFailed);
            return false;
        }
        ++status_.evaluation_count;
        return true;
    }

    VulkanSceneUpscalerOutput VulkanSceneUpscalerController::output(
        const TemporalViewId view) const noexcept {
        return adapter_ ? adapter_->output(view) : VulkanSceneUpscalerOutput{};
    }

    void VulkanSceneUpscalerController::reset(
        const TemporalViewId view,
        const TemporalResetReason reasons) noexcept {
        if (adapter_ && reasons != TemporalResetReason::None) {
            adapter_->reset(view, reasons);
        }
    }

    void VulkanSceneUpscalerController::deactivate() noexcept {
        if (adapter_) {
            adapter_->shutdown();
            adapter_.reset();
        }
        requirements_ = {};
        status_.active_id.clear();
    }

    void VulkanSceneUpscalerController::fail(
        const VulkanSceneUpscalerFailure failure,
        const SceneUpscalerAvailabilityReason availability) noexcept {
        if (adapter_) {
            adapter_->shutdown();
            adapter_.reset();
        }
        requirements_ = {};
        status_.active_id.clear();
        status_.failure = failure;
        status_.availability = availability;
        ++status_.generation;
        ++status_.fallback_count;
    }
} // namespace lfs::vis
