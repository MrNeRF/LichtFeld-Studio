/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdint>
#include <optional>
#include <string_view>

namespace lfs::rendering {
    // GPU API for the viewer, independent of tensor execution and 3DGS/3DGUT.
    enum class ViewerBackend { Automatic,
                               Vulkan,
                               Metal };
    [[nodiscard]] constexpr std::string_view viewerBackendName(ViewerBackend backend) {
        switch (backend) {
        case ViewerBackend::Automatic: return "auto";
        case ViewerBackend::Vulkan: return "vulkan";
        case ViewerBackend::Metal: return "metal";
        }
        return {};
    }
    [[nodiscard]] constexpr std::optional<ViewerBackend> parseViewerBackend(std::string_view name) {
        if (name == "auto")
            return ViewerBackend::Automatic;
        if (name == "vulkan")
            return ViewerBackend::Vulkan;
        if (name == "metal")
            return ViewerBackend::Metal;
        return std::nullopt;
    }
    // Zero means no scene backend was published. Metadata retains explicit
    // backend identity for diagnostic captures and reference tests.
    [[nodiscard]] constexpr uint32_t viewerBackendBit(ViewerBackend backend) {
        return backend == ViewerBackend::Vulkan ? 1u : backend == ViewerBackend::Metal ? 2u
                                                                                       : 0u;
    }
    // Software point-cloud panels can participate in a mixed split frame.
    inline constexpr uint32_t softwareViewerBackendBit = 4u;

    // Scene rendering is fixed by the platform; the compositor remains Vulkan.
    [[nodiscard]] constexpr ViewerBackend desktopViewerBackend() {
#ifdef __APPLE__
        return ViewerBackend::Metal;
#else
        return ViewerBackend::Vulkan;
#endif
    }
} // namespace lfs::rendering
