/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
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
    struct ViewerBackendCapabilities {
        bool vulkan = false;
        bool metal = false;
        // Enabled by the host only after its native presentation and frame contracts
        // are supported. Merely compiling a Metal shader does not satisfy this gate.
        bool prefer_metal = false;
    };
    enum class ViewerBackendReason { None,
                                     Unavailable,
                                     UnsupportedFrame,
                                     NoAvailableBackend };
    struct ViewerBackendSelection {
        ViewerBackend requested;
        std::optional<ViewerBackend> effective;
        ViewerBackendReason reason;
        [[nodiscard]] constexpr bool fallback() const {
            return effective && requested != ViewerBackend::Automatic && requested != *effective;
        }
    };
    [[nodiscard]] constexpr ViewerBackendSelection selectViewerBackend(
        ViewerBackend requested, ViewerBackendCapabilities capabilities, bool metal_supports_frame = true) {
        const bool metal = capabilities.metal && metal_supports_frame;
        if (requested == ViewerBackend::Automatic) {
            if (metal && (capabilities.prefer_metal || !capabilities.vulkan))
                return {requested, ViewerBackend::Metal, ViewerBackendReason::None};
            if (capabilities.vulkan)
                return {requested, ViewerBackend::Vulkan, ViewerBackendReason::None};
        } else if (requested == ViewerBackend::Metal) {
            if (metal)
                return {requested, ViewerBackend::Metal, ViewerBackendReason::None};
            if (capabilities.vulkan)
                return {requested, ViewerBackend::Vulkan,
                        capabilities.metal ? ViewerBackendReason::UnsupportedFrame : ViewerBackendReason::Unavailable};
        } else if (requested == ViewerBackend::Vulkan) {
            if (capabilities.vulkan)
                return {requested, ViewerBackend::Vulkan, ViewerBackendReason::None};
            if (metal)
                return {requested, ViewerBackend::Metal, ViewerBackendReason::Unavailable};
        }
        return {requested, std::nullopt, ViewerBackendReason::NoAvailableBackend};
    }
} // namespace lfs::rendering
