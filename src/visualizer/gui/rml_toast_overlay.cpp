/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_toast_overlay.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "internal/resource_paths.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>

namespace lfs::vis::gui {

    bool ToastStack::push(ToastRequest request, const std::chrono::steady_clock::time_point now) {
        if (request.fingerprint != 0) {
            for (Entry& entry : entries) {
                if (entry.request.fingerprint == request.fingerprint) {
                    ++entry.count;
                    entry.shown_at = now;
                    return true;
                }
            }
        }
        entries.push_back(Entry{.request = std::move(request), .count = 1, .shown_at = now});
        while (entries.size() > kMaxVisible) {
            entries.erase(entries.begin());
        }
        return true;
    }

    bool ToastStack::expire(const std::chrono::steady_clock::time_point now) {
        const std::size_t before = entries.size();
        std::erase_if(entries, [&](const Entry& entry) {
            return now - entry.shown_at >= kDuration;
        });
        return entries.size() != before;
    }

    float ToastStack::alpha(const Entry& entry, const std::chrono::steady_clock::time_point now) {
        const auto elapsed = now - entry.shown_at;
        if (elapsed >= kDuration) {
            return 0.0f;
        }
        const auto remaining =
            kDuration - std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
        const float remaining_ms = static_cast<float>(remaining.count());
        return remaining_ms < kFadeMs ? remaining_ms / kFadeMs : 1.0f;
    }

    RmlToastOverlay::RmlToastOverlay(RmlUIManager* rml_manager)
        : rml_manager_(rml_manager) {
        assert(rml_manager_);
    }

    RmlToastOverlay::~RmlToastOverlay() {
        if (rml_manager_ && rml_manager_->isInitialized())
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        if (rml_context_ && rml_manager_ && rml_manager_->isInitialized())
            rml_manager_->destroyContext("toast_overlay");
    }

    void RmlToastOverlay::enqueue(ToastRequest request) {
        std::lock_guard lock(queue_mutex_);
        queue_.push_back(std::move(request));
    }

    bool RmlToastOverlay::hasPendingRenderWork() const {
        std::lock_guard lock(queue_mutex_);
        return !queue_.empty() || !stack_.entries.empty();
    }

    bool RmlToastOverlay::needsAnimationFrame() const {
        return hasPendingRenderWork();
    }

    void RmlToastOverlay::initContext() {
        if (rml_context_)
            return;

        rml_context_ = rml_manager_->createContext("toast_overlay", 800, 600);
        if (!rml_context_) {
            LOG_ERROR("RmlToastOverlay: failed to create context");
            return;
        }

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/toast_overlay.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlToastOverlay: failed to load toast_overlay.rml");
                return;
            }
            document_->Show();
            el_stack_ = document_->GetElementById("toast-stack");
            if (!el_stack_)
                LOG_ERROR("RmlToastOverlay: missing toast-stack element");
        } catch (const std::exception& e) {
            LOG_ERROR("RmlToastOverlay: resource not found: {}", e.what());
        }
    }

    void RmlToastOverlay::reloadResources() {
        if (!rml_context_)
            return;

        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }

        document_ = nullptr;
        el_stack_ = nullptr;
        base_rcss_.clear();
        has_theme_signature_ = false;
        width_ = 0;
        height_ = 0;
        last_right_px_ = -1.0f;
        last_bottom_px_ = -1.0f;
        last_alpha_.clear();
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        render_needed_ = true;

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/toast_overlay.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlToastOverlay: failed to reload toast_overlay.rml");
                return;
            }
            document_->Show();
            el_stack_ = document_->GetElementById("toast-stack");
        } catch (const std::exception& e) {
            LOG_ERROR("RmlToastOverlay: resource not found during reload: {}", e.what());
            return;
        }

        syncTheme();
        // The reloaded stack element starts empty; repopulate it so surviving
        // toasts render immediately instead of waiting for the next mutation.
        rebuildStackRml();
    }

    bool RmlToastOverlay::syncTheme() {
        if (!document_)
            return false;

        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && theme_signature == last_theme_signature_)
            return false;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/toast_overlay.rcss");

        rml_theme::applyTheme(document_, base_rcss_,
                              rml_theme::loadBaseRCSS("rmlui/toast_overlay.theme.rcss"));
        return true;
    }

    void RmlToastOverlay::rebuildStackRml() {
        if (!el_stack_)
            return;

        std::string html;
        html.reserve(stack_.entries.size() * 160);
        for (std::size_t i = 0; i < stack_.entries.size(); ++i) {
            const ToastStack::Entry& entry = stack_.entries[i];
            const char* level = entry.request.level == ErrorNoticeLevel::Info      ? "info"
                                : entry.request.level == ErrorNoticeLevel::Warning ? "warning"
                                                                                   : "error";
            std::string count_span;
            if (entry.count > 1)
                count_span = std::format("<span class=\"toast-count\"> x{}</span>", entry.count);

            html += std::format(
                "<div id=\"toast-{}\" class=\"toast level-{}\">"
                "<div class=\"toast-title\">{}{}</div>"
                "<div class=\"toast-message\">{}</div>"
                "</div>",
                i, level, escapeRmlText(entry.request.title), count_span,
                escapeRmlText(entry.request.message));
        }
        el_stack_->SetInnerRML(html);
        last_alpha_.clear();
    }

    void RmlToastOverlay::updateOpacities(const std::chrono::steady_clock::time_point now) {
        if (!document_)
            return;

        std::vector<std::uint8_t> current;
        current.reserve(stack_.entries.size());
        bool any_changed = false;
        for (std::size_t i = 0; i < stack_.entries.size(); ++i) {
            const float a = std::clamp(ToastStack::alpha(stack_.entries[i], now), 0.0f, 1.0f);
            const auto quantized = static_cast<std::uint8_t>(std::lround(a * 255.0f));
            current.push_back(quantized);
            const bool changed = i >= last_alpha_.size() || last_alpha_[i] != quantized;
            if (changed) {
                if (auto* el = document_->GetElementById(std::format("toast-{}", i)))
                    el->SetProperty("opacity", std::format("{:.3f}", a));
                any_changed = true;
            }
        }
        last_alpha_ = std::move(current);
        if (any_changed)
            render_needed_ = true;
    }

    void RmlToastOverlay::render(int screen_w, int screen_h, float screen_x, float screen_y,
                                 float vp_x, float vp_y, float vp_w, float vp_h) {
        const auto now = std::chrono::steady_clock::now();

        bool changed = false;
        std::deque<ToastRequest> drained;
        {
            std::lock_guard lock(queue_mutex_);
            drained.swap(queue_);
        }
        for (ToastRequest& request : drained)
            changed |= stack_.push(std::move(request), now);
        changed |= stack_.expire(now);

        if (stack_.entries.empty() && !changed)
            return;

        if (!rml_context_) {
            initContext();
            if (!rml_context_)
                return;
        }
        if (!document_ || !el_stack_)
            return;

        if (!rml_manager_ || !rml_manager_->getVulkanRenderInterface())
            return;

        rml_manager_->trackContextFrame(rml_context_, 0, 0);
        const bool theme_changed = syncTheme();

        const int w = screen_w;
        const int h = screen_h;
        if (w <= 0 || h <= 0)
            return;

        bool needs_update = render_needed_ || theme_changed || changed;
        if (w != width_ || h != height_) {
            width_ = w;
            height_ = h;
            rml_context_->SetDimensions(Rml::Vector2i(w, h));
            needs_update = true;
        }

        const float right_px = static_cast<float>(w) - ((vp_x - screen_x) + vp_w) + 16.0f;
        const float bottom_px = static_cast<float>(h) - ((vp_y - screen_y) + vp_h) + 16.0f;
        if (last_right_px_ < 0.0f || std::abs(right_px - last_right_px_) > 0.5f ||
            std::abs(bottom_px - last_bottom_px_) > 0.5f) {
            el_stack_->SetProperty("right", std::format("{}px", right_px));
            el_stack_->SetProperty("bottom", std::format("{}px", bottom_px));
            last_right_px_ = right_px;
            last_bottom_px_ = bottom_px;
            needs_update = true;
        }

        if (changed || theme_changed)
            rebuildStackRml();

        updateOpacities(now);

        needs_update = needs_update || render_needed_;
        if (needs_update)
            rml_context_->Update();

        const bool refresh_cache = needs_update || direct_cache_.texture == 0;
        render_needed_ = false;
        rml_manager_->queueCachedVulkanContext({
            .context = rml_context_,
            .cache = &direct_cache_,
            .cache_width = w,
            .cache_height = h,
            .offset_x = 0.0f,
            .offset_y = 0.0f,
            .draw_width = static_cast<float>(w),
            .draw_height = static_cast<float>(h),
            .refresh = refresh_cache,
            .foreground = true,
            .clip = {},
        });
    }

    void RmlToastOverlay::releaseRendererResources() {
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
    }

} // namespace lfs::vis::gui
