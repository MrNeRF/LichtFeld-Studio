/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/scene_upscaler_registry.hpp"

#include <array>
#include <ranges>

namespace lfs::vis {
    namespace {
        constexpr std::array DESCRIPTORS{
            SceneUpscalerDescriptor{
                .backend = SceneUpscalerBackend::Native,
                .id = "native",
                .requirements = {},
                .requires_adapter = false,
                .available = true,
            },
            SceneUpscalerDescriptor{
                .backend = SceneUpscalerBackend::Spatial,
                .id = "spatial",
                .requirements = {},
                .requires_adapter = true,
                .available = true,
            },
        };
    } // namespace

    std::span<const SceneUpscalerDescriptor> sceneUpscalerDescriptors() {
        return DESCRIPTORS;
    }

    const SceneUpscalerDescriptor& nativeSceneUpscalerDescriptor() {
        return DESCRIPTORS.front();
    }

    const SceneUpscalerDescriptor& spatialSceneUpscalerDescriptor() {
        return DESCRIPTORS[1];
    }

    const SceneUpscalerDescriptor& sceneUpscalerDescriptor(const SceneUpscalerBackend backend) {
        const auto descriptor = std::ranges::find(DESCRIPTORS, backend,
                                                  &SceneUpscalerDescriptor::backend);
        return descriptor != DESCRIPTORS.end() ? *descriptor : nativeSceneUpscalerDescriptor();
    }

    std::optional<SceneUpscalerBackend> sceneUpscalerBackendFromId(const std::string_view id) {
        const auto descriptor = std::ranges::find(DESCRIPTORS, id, &SceneUpscalerDescriptor::id);
        if (descriptor == DESCRIPTORS.end())
            return std::nullopt;
        return descriptor->backend;
    }

    SceneUpscalerSelection resolveSceneUpscalerSelection(
        const SceneUpscalerBackend requested, const bool adapter_available) {
        if (requested == SceneUpscalerBackend::Native) {
            return {};
        }
        if (requested == SceneUpscalerBackend::Spatial && adapter_available) {
            return {.requested = requested, .effective = requested, .fallback = false};
        }
        return {
            .requested = requested,
            .effective = SceneUpscalerBackend::Native,
            .fallback = true,
        };
    }
} // namespace lfs::vis
