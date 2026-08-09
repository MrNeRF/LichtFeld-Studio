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

    std::optional<SceneUpscalerBackend> sceneUpscalerBackendFromId(const std::string_view id) {
        const auto descriptor = std::ranges::find(DESCRIPTORS, id, &SceneUpscalerDescriptor::id);
        if (descriptor == DESCRIPTORS.end())
            return std::nullopt;
        return descriptor->backend;
    }
} // namespace lfs::vis
