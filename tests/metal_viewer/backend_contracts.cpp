/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "rendering/viewer_backend.hpp"
using namespace lfs::rendering;
// Platform policy and actual diagnostic names need no GPU initialization.
int main() {
    if (viewerBackendName(ViewerBackend::Vulkan) != "vulkan" ||
        viewerBackendName(ViewerBackend::Metal) != "metal" ||
        viewerBackendName(ViewerBackend::Cuda) != "cuda")
        return 1;
    if (viewerBackendDisplayName(ViewerBackend::Cuda) != "CUDA")
        return 2;
#ifdef __APPLE__
    if (desktopViewerBackend() != ViewerBackend::Metal)
#else
    if (desktopViewerBackend() != ViewerBackend::Vulkan)
#endif
        return 3;
    return 0;
}
