/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "gui/error_surface_types.hpp"
#include "gui/rmlui/rmlui_manager.hpp"

#include <chrono>
#include <core/export.hpp>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace Rml {
    class Context;
    class Element;
    class ElementDocument;
} // namespace Rml

// Native toast surface for the ErrorBus (Phase 8, packet P2). Mirrors
// RmlModalOverlay's threading exactly: workers enqueue a ToastRequest under a
// mutex; ALL RmlUi work happens in render() on the UI frame. Toasts are
// non-interactive (no processInput, never in input-block logic) and render
// beneath the modal overlay.
namespace lfs::vis::gui {

    class RmlUIManager;

    // Headless-testable stack policy: newest-last ordering, max kMaxVisible with
    // evict-oldest overflow, same-fingerprint collapse into a counter with a
    // restarted timer, kDuration auto-dismiss and a kFadeMs linear fade.
    struct LFS_VIS_API ToastStack {
        struct Entry {
            ToastRequest request;
            std::uint32_t count = 1;
            std::chrono::steady_clock::time_point shown_at{};
        };

        static constexpr std::size_t kMaxVisible = 4;
        static constexpr std::chrono::milliseconds kDuration{6000};
        static constexpr float kFadeMs = 500.0f;

        // Same-fingerprint (non-zero) match against a visible entry: ++count,
        // shown_at = now (timer restarts), position kept. Otherwise append; if
        // size() would exceed kMaxVisible, evict entries.front() (oldest).
        // Returns true when membership or a count changed.
        bool push(ToastRequest request, std::chrono::steady_clock::time_point now);

        // Remove entries with now - shown_at >= kDuration. True when any removed.
        bool expire(std::chrono::steady_clock::time_point now);

        // 1.0 until the last kFadeMs of kDuration, then linear to 0.0.
        [[nodiscard]] static float alpha(const Entry& entry,
                                         std::chrono::steady_clock::time_point now);

        std::vector<Entry> entries; // oldest first; index 0 renders topmost
    };

    class LFS_VIS_API RmlToastOverlay {
    public:
        explicit RmlToastOverlay(RmlUIManager* rml_manager);
        ~RmlToastOverlay();

        RmlToastOverlay(const RmlToastOverlay&) = delete;
        RmlToastOverlay& operator=(const RmlToastOverlay&) = delete;

        void enqueue(ToastRequest request);
        void render(int screen_w, int screen_h, float screen_x, float screen_y,
                    float vp_x, float vp_y, float vp_w, float vp_h);
        void releaseRendererResources();
        void reloadResources();

        [[nodiscard]] bool hasPendingRenderWork() const;
        [[nodiscard]] bool needsAnimationFrame() const;

    private:
        void initContext();
        bool syncTheme();
        void rebuildStackRml();
        void updateOpacities(std::chrono::steady_clock::time_point now);

        RmlUIManager* rml_manager_;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* el_stack_ = nullptr;

        mutable std::mutex queue_mutex_;
        std::deque<ToastRequest> queue_;
        ToastStack stack_;

        std::string base_rcss_;
        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        int width_ = 0;
        int height_ = 0;
        float last_right_px_ = -1.0f;
        float last_bottom_px_ = -1.0f;
        std::vector<std::uint8_t> last_alpha_;
        CachedVulkanContextRender direct_cache_;
        bool render_needed_ = true;
    };

} // namespace lfs::vis::gui
