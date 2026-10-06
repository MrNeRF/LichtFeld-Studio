/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene_upscaler_plugin.hpp"

#include <array>

namespace lfs::vis {
    std::span<SceneUpscalerPlugin* const> sceneUpscalerPlugins() {
        static constexpr std::array<SceneUpscalerPlugin*, 0> plugins{};
        return plugins;
    }

    SceneUpscalerPlugin* sceneUpscalerPlugin(SceneUpscalerBackend) {
        return nullptr;
    }

    void configureSceneUpscalerPluginLoading(bool) {}
} // namespace lfs::vis
