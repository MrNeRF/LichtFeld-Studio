/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/error.hpp"
#include "render_target_id.hpp"
#include "rendering/rendering.hpp"
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace lfs::core {
    class SplatData;
}
namespace lfs::vis::shaders {
    struct ShaderPrograms;
}

namespace lfs::vis {
    // Optional SceneRenderer capability: draws splats with linked user shader
    // programs (custom effects). Only a renderer that runs the programs' host
    // stages provides it; without it user shaders are unavailable.
    class SceneCustomEffects {
    public:
        struct DrawPlan {
            std::string host;      // the host the draw runs; empty when none takes a custom effect
            std::string warm_host; // a warmup draw's host once warmup ends, else empty
        };
        virtual ~SceneCustomEffects() = default;
        // Changes whenever the device dropped every program set.
        [[nodiscard]] virtual std::uint64_t deviceGeneration() const = 0;
        [[nodiscard]] virtual lfs::Result<void> initializePrograms() = 0;
        // Prepares the inputs of drawing `model` for `request` into `target`, as
        // SceneRenderer::render with the same arguments does, and plans its hosts.
        [[nodiscard]] virtual lfs::Result<DrawPlan> plan(
            const core::SplatData& model, const rendering::ViewportRenderRequest& request, RenderTargetId target,
            bool synchronize_input_upload = false, bool deterministic_export = false) = 0;
        // The host of a draw whose inputs plan() prepared; `capture` is a depth capture.
        [[nodiscard]] virtual std::string planHost(const rendering::ViewportRenderRequest& request, RenderTargetId target,
                                                   bool synchronize_input_upload, bool capture, bool deterministic_export) = 0;
        // Builds the program set of `programs` into `handle`; true once it can draw.
        [[nodiscard]] virtual lfs::Result<bool> preparePrograms(const std::shared_ptr<const shaders::ShaderPrograms>& programs,
                                                                rendering::ProgramSetHandle& handle,
                                                                std::function<void()> finished, bool live = false) = 0;
        virtual void usePrograms(const rendering::ProgramSetHandle& handle) = 0;
        // Runs before every splat draw with the host it runs; selects its programs
        // with usePrograms(). An error fails the draw.
        std::function<Status(std::string_view host, const rendering::ViewportRenderRequest& request)> select_programs;
    };
} // namespace lfs::vis
