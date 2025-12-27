/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "framebuffer.hpp"
#include <memory>

namespace lfs::rendering {
    enum class FrameBufferMode {
        CPU,
        CUDA_INTEROP
    };

    // Template-based feature detection instead of preprocessor
    template <bool EnableInterop = false>
    struct InteropCapability {
        static constexpr bool available = EnableInterop;
    };

    // Runtime interop disable flag (set before RenderingEngine creation)
    inline bool& isInteropDisabled() {
        static bool disabled = false;
        return disabled;
    }

    inline void disableInterop() { isInteropDisabled() = true; }

    inline FrameBufferMode getPreferredFrameBufferMode() {
        if (isInteropDisabled()) {
            return FrameBufferMode::CPU;
        }

        constexpr bool CUDA_INTEROP_AVAILABLE =
#ifdef CUDA_GL_INTEROP_ENABLED
            true;
#else
            false;
#endif

        if constexpr (CUDA_INTEROP_AVAILABLE) {
            return FrameBufferMode::CUDA_INTEROP;
        }
        return FrameBufferMode::CPU;
    }

    inline bool isInteropAvailable() {
        return getPreferredFrameBufferMode() == FrameBufferMode::CUDA_INTEROP;
    }

    // Create a framebuffer with the specified mode
    std::shared_ptr<FrameBuffer> createFrameBuffer(FrameBufferMode preferred = FrameBufferMode::CPU);
} // namespace lfs::rendering