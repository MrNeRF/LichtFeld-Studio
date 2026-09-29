// SPDX-License-Identifier: GPL-3.0-or-later
#include "rendering/viewer_backend.hpp"
using namespace lfs::rendering;
// Host/platform availability must not rewrite a saved preference or couple it
// to the tensor backend. Exercise all availability/default/frame combinations.
int main() {
    for (auto requested : {ViewerBackend::Automatic, ViewerBackend::Vulkan, ViewerBackend::Metal}) {
        if (parseViewerBackend(viewerBackendName(requested)) != requested) return 1;
        for (int bits=0; bits<16; ++bits) {
            const ViewerBackendCapabilities caps{bool(bits&1),bool(bits&2),bool(bits&4)};
            const bool frame=bool(bits&8);
            const auto result=selectViewerBackend(requested,caps,frame);
            if (result.requested!=requested) return 2;
            if (result.effective==ViewerBackend::Metal && (!caps.metal || !frame)) return 3;
            if (result.effective==ViewerBackend::Vulkan && !caps.vulkan) return 4;
            if (!result.effective && (caps.vulkan || (caps.metal && frame))) return 5;
            if (requested==ViewerBackend::Vulkan && caps.vulkan && result.effective!=requested) return 6;
            if (requested==ViewerBackend::Metal && caps.metal && frame && result.effective!=requested) return 7;
            if (requested==ViewerBackend::Automatic && caps.vulkan && (!caps.prefer_metal || !caps.metal || !frame)
                && result.effective!=ViewerBackend::Vulkan) return 8;
            if (result.fallback() && result.reason==ViewerBackendReason::None) return 9;
        }
    }
    if (parseViewerBackend("cuda") || parseViewerBackend("unknown")) return 10;
    return 0;
}
