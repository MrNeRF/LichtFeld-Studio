/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace lfs::vis {

    enum class SceneUpscalerBackend : std::uint8_t {
        Native = 0,
        Spatial,
    };

    struct SceneUpscalerRequirements {
        bool depth = false;
        bool motion_vectors = false;
        bool jitter = false;
        bool history = false;
        bool reactive_mask = false;
        bool exposure = false;

        [[nodiscard]] constexpr bool any() const {
            return depth || motion_vectors || jitter || history || reactive_mask || exposure;
        }

        [[nodiscard]] constexpr bool temporal() const {
            return motion_vectors || jitter || history;
        }

        constexpr bool operator==(const SceneUpscalerRequirements&) const = default;
    };

    struct SceneUpscalerDescriptor {
        SceneUpscalerBackend backend = SceneUpscalerBackend::Native;
        std::string_view id;
        SceneUpscalerRequirements requirements;
        bool requires_adapter = false;
        bool available = false;
    };

    struct SceneUpscalerSelection {
        SceneUpscalerBackend requested = SceneUpscalerBackend::Native;
        SceneUpscalerBackend effective = SceneUpscalerBackend::Native;
        bool fallback = false;

        constexpr bool operator==(const SceneUpscalerSelection&) const = default;
    };

    [[nodiscard]] LFS_VIS_API std::span<const SceneUpscalerDescriptor> sceneUpscalerDescriptors();
    [[nodiscard]] LFS_VIS_API const SceneUpscalerDescriptor& nativeSceneUpscalerDescriptor();
    [[nodiscard]] LFS_VIS_API const SceneUpscalerDescriptor& spatialSceneUpscalerDescriptor();
    [[nodiscard]] LFS_VIS_API const SceneUpscalerDescriptor& sceneUpscalerDescriptor(
        SceneUpscalerBackend backend);
    [[nodiscard]] LFS_VIS_API std::optional<SceneUpscalerBackend> sceneUpscalerBackendFromId(
        std::string_view id);
    [[nodiscard]] LFS_VIS_API SceneUpscalerSelection resolveSceneUpscalerSelection(
        SceneUpscalerBackend requested, bool adapter_available);

} // namespace lfs::vis
