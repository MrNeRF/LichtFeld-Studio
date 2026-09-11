/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Internal to lfs_visualizer — consumed only inside VulkanViewportPass::Impl (PIMPL).
// Not part of the public DLL surface, so no LFS_VIS_API.

#include "vulkan_viewport_pass.hpp"
#include "window/vulkan_result.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <nvtx3/nvToolsExt.h>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    // Ordering buckets. Passes within a phase run in registration order.
    enum class ViewportPhase : std::uint32_t {
        Background = 0,   // environment
        Scene = 1,        // scene-blit XOR split-view
        DepthBlit = 2,    // splat depth -> framebuffer depth
        Geometry = 3,     // mesh
        WorldOverlay = 4, // textured, base, world shape overlays, pivot, grid
        Effect = 5,       // vignette
        UiOverlay = 6,    // UI shape overlays, post-UI overlays
    };

    // ARGB range colors so each phase reads as a distinct band on the nsys/Nsight timeline.
    [[nodiscard]] inline std::uint32_t nvtxColorForPhase(ViewportPhase phase) {
        switch (phase) {
        case ViewportPhase::Background: return 0xFF607D8B;   // blue-grey
        case ViewportPhase::Scene: return 0xFF2196F3;        // blue
        case ViewportPhase::DepthBlit: return 0xFF00BCD4;    // cyan
        case ViewportPhase::Geometry: return 0xFF4CAF50;     // green
        case ViewportPhase::WorldOverlay: return 0xFFFF9800; // orange
        case ViewportPhase::Effect: return 0xFF9C27B0;       // purple
        case ViewportPhase::UiOverlay: return 0xFFFFEB3B;    // yellow
        }
        return 0xFFFFFFFF;
    }

    // RAII NVTX range. The codebase profiles with nsys (--trace=nvtx); naming each sub-pass turns the
    // viewport into a self-labelling timeline. Near-zero cost with no collector attached.
    class ScopedNvtxRange {
    public:
        ScopedNvtxRange(const char* name, std::uint32_t argb) {
            nvtxEventAttributes_t attribs{};
            attribs.version = NVTX_VERSION;
            attribs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attribs.colorType = NVTX_COLOR_ARGB;
            attribs.color = argb;
            attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attribs.message.ascii = name;
            nvtxRangePushEx(&attribs);
        }
        ~ScopedNvtxRange() { nvtxRangePop(); }
        ScopedNvtxRange(const ScopedNvtxRange&) = delete;
        ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;
        ScopedNvtxRange(ScopedNvtxRange&&) = delete;
        ScopedNvtxRange& operator=(ScopedNvtxRange&&) = delete;
    };

    // Shared per-frame state computed once by the driver and handed to every pass. The rect is
    // carried as plain fields to keep this header free of the .cpp-local FramebufferRect type.
    struct ViewportRecordContext {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkExtent2D extent{};
        std::int32_t rect_x = 0;
        std::int32_t rect_y = 0;
        std::uint32_t rect_w = 0;
        std::uint32_t rect_h = 0;
        bool depth_available = false;
        std::size_t frame_slot = 0;
        glm::vec4 viewport_rect_push{0.0f};
        glm::vec4 world_depth_params_push{0.0f};
    };

    class ViewportSubPass {
    public:
        virtual ~ViewportSubPass() = default;

        [[nodiscard]] virtual const char* name() const = 0;
        [[nodiscard]] virtual ViewportPhase phase() const = 0;

        // Authoritative draw condition. The graph skips the pass entirely when this returns false,
        // so record() may assume it holds.
        [[nodiscard]] virtual bool active(const VulkanViewportPassParams& params) const = 0;

        // Records draws into the already-open shared dynamic-rendering scope.
        virtual void record(const ViewportRecordContext& ctx,
                            const VulkanViewportPassParams& params) = 0;
    };

    // Concrete pass wrapping existing Impl-coupled code via lambdas (capture `this`). Avoids one
    // near-identical subclass per inline draw step; the gate + draw stay declarative and ordered.
    class LambdaSubPass final : public ViewportSubPass {
    public:
        using ActiveFn = std::function<bool(const VulkanViewportPassParams&)>;
        using RecordFn = std::function<void(const ViewportRecordContext&, const VulkanViewportPassParams&)>;

        LambdaSubPass(const char* name, ViewportPhase phase, ActiveFn active, RecordFn record)
            : name_(name),
              phase_(phase),
              active_(std::move(active)),
              record_(std::move(record)) {}

        [[nodiscard]] const char* name() const override { return name_; }
        [[nodiscard]] ViewportPhase phase() const override { return phase_; }
        [[nodiscard]] bool active(const VulkanViewportPassParams& params) const override {
            return active_(params);
        }
        void record(const ViewportRecordContext& ctx, const VulkanViewportPassParams& params) override {
            record_(ctx, params);
        }

    private:
        const char* name_;
        ViewportPhase phase_;
        ActiveFn active_;
        RecordFn record_;
    };

    // Ordered registry. Non-owning: passes are members of VulkanViewportPass::Impl and outlive it.
    class ViewportPassGraph {
    public:
        void add(ViewportSubPass& pass) {
            const std::uint32_t order = static_cast<std::uint32_t>(entries_.size());
            entries_.push_back({&pass, order});
        }

        void finalize() {
            std::stable_sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) {
                const auto pa = static_cast<std::uint32_t>(a.pass->phase());
                const auto pb = static_cast<std::uint32_t>(b.pass->phase());
                return pa != pb ? pa < pb : a.order < b.order;
            });
            finalized_ = true;
        }

        void record(const ViewportRecordContext& ctx,
                    const VulkanViewportPassParams& params,
                    const std::function<void()>& restore_viewport_quad_state) {
            // Recording before finalize() would draw passes in registration order, not phase order —
            // a silent z-ordering bug. The driver finalizes once in init().
            LFS_VK_DEBUG_ASSERT(
                finalized_,
                "Viewport pass graph must be finalized before command recording (finalized={}, registered_passes={}, command_buffer={:#x}, frame_slot={}, extent={}x{})",
                finalized_,
                entries_.size(),
                vkHandleValue(ctx.cmd),
                ctx.frame_slot,
                ctx.extent.width,
                ctx.extent.height);
            for (auto& e : entries_) {
                if (!e.pass->active(params)) {
                    continue;
                }
                const ScopedNvtxRange pass_range{e.pass->name(), nvtxColorForPhase(e.pass->phase())};
                e.pass->record(ctx, params);
                if (restore_viewport_quad_state) {
                    restore_viewport_quad_state();
                }
            }
        }

    private:
        struct Entry {
            ViewportSubPass* pass;
            std::uint32_t order;
        };
        std::vector<Entry> entries_;
        bool finalized_ = false;
    };

} // namespace lfs::vis
