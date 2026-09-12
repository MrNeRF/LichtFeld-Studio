// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: MIT
//
// The upscaler-only Linux archive omits the SDK frame-interpolation sources,
// but the public Vulkan interface still exposes this callback.

#include <FidelityFX/host/backends/vk/ffx_vk.h>

extern "C" FfxErrorCode ffxSetFrameGenerationConfigToSwapchainVK(const FfxFrameGenerationConfig*) {
    return FFX_ERROR_INVALID_ARGUMENT;
}
