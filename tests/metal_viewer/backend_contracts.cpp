/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "rendering/viewer_backend.hpp"
using namespace lfs::rendering;
// The desktop scene API is fixed by the platform, independently of tensor
// execution. Legacy backend names remain readable for metadata.
int main() {
    for (auto requested : {ViewerBackend::Automatic, ViewerBackend::Vulkan, ViewerBackend::Metal}) {
        if (parseViewerBackend(viewerBackendName(requested)) != requested)
            return 1;
        if (viewerBackendBit(requested) != (requested == ViewerBackend::Metal ? 2u : requested == ViewerBackend::Vulkan ? 1u
                                                                                                                        : 0u))
            return 2;
    }

    if (parseViewerBackend("cuda") || parseViewerBackend("unknown"))
        return 10;
    // Exercise the production desktop policy without requiring a GPU in CI.
#ifdef __APPLE__
    if (desktopViewerBackend() != ViewerBackend::Metal)
#else
    if (desktopViewerBackend() != ViewerBackend::Vulkan)
#endif
        return 11;
    return 0;
}
