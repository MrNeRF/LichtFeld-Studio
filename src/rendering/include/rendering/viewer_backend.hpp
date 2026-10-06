/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <string_view>
namespace lfs::rendering {
    // API that produced a scene output, independent of 3DGS/3DGUT.
    // CUDA identifies utility point rendering; desktop renderers use Vulkan/Metal.
    enum class ViewerBackend { Vulkan,
                               Metal,
                               Cuda };
    [[nodiscard]] constexpr std::string_view viewerBackendName(ViewerBackend backend) {
        switch (backend) {
        case ViewerBackend::Vulkan: return "vulkan";
        case ViewerBackend::Metal: return "metal";
        case ViewerBackend::Cuda: return "cuda";
        }
        return {};
    }
    [[nodiscard]] constexpr std::string_view viewerBackendDisplayName(ViewerBackend backend) {
        switch (backend) {
        case ViewerBackend::Vulkan: return "Vulkan";
        case ViewerBackend::Metal: return "Metal";
        case ViewerBackend::Cuda: return "CUDA";
        }
        return {};
    }
    // Scene rendering is fixed by the build: the native Metal viewer when it is
    // compiled, otherwise the Vulkan rasterizer.
    [[nodiscard]] constexpr ViewerBackend desktopViewerBackend() {
#ifdef LFS_TENSOR_METAL
        return ViewerBackend::Metal;
#else
        return ViewerBackend::Vulkan;
#endif
    }
} // namespace lfs::rendering
