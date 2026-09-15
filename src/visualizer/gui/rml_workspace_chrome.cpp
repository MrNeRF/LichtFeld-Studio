/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_workspace_chrome.hpp"

#include "core/logger.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "internal/resource_paths.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <utility>

namespace lfs::vis::gui {

    namespace {
        [[nodiscard]] int px(const int value) { return value; }

        [[nodiscard]] std::string rectStyle(const ViewRect& rect, const ViewRect& outer) {
            return std::format("left:{}px;top:{}px;width:{}px;height:{}px;",
                               px(rect.x - outer.x), px(rect.y - outer.y),
                               px(rect.width), px(rect.height));
        }

        [[nodiscard]] std::string gripMarkup(const AreaSnapshot& area,
                                             const ViewRect& outer) {
            if (area.rect.width < 28 || area.rect.height < 28)
                return {};
            const ViewRect grip{area.rect.right() - 24, area.rect.bottom() - 24, 24, 24};
            return std::format(
                R"(<div class="workspace-grip" style="{}"><span class="workspace-grip__line workspace-grip__line--1"></span><span class="workspace-grip__line workspace-grip__line--2"></span><span class="workspace-grip__line workspace-grip__line--3"></span></div>)",
                rectStyle(grip, outer));
        }

        [[nodiscard]] std::string splitterMarkup(const ViewRect& rect,
                                                 const ViewRect& outer,
                                                 const char* class_name) {
            if (rect.empty())
                return {};
            return std::format(R"(<div class="{}" style="{}"></div>)", class_name,
                               rectStyle(rect, outer));
        }
    } // namespace

    void RmlWorkspaceChrome::init(RmlUIManager* const manager) {
        assert(manager);
        shutdown();
        rml_manager_ = manager;
        rml_context_ = rml_manager_->createContext("workspace_chrome", 800, 600);
        if (!rml_context_) {
            LOG_ERROR("RmlWorkspaceChrome: failed to create RML context");
            return;
        }

        try {
            document_ = rml_documents::loadDocument(
                rml_context_, lfs::vis::getAssetPath("rmlui/workspace_chrome.rml"));
            if (!document_) {
                LOG_ERROR("RmlWorkspaceChrome: failed to load workspace_chrome.rml");
                return;
            }
            document_->Show();
            updateTheme();
        } catch (const std::exception& error) {
            LOG_ERROR("RmlWorkspaceChrome: resource load failed: {}", error.what());
        }
    }

    void RmlWorkspaceChrome::shutdown() {
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        if (rml_manager_ && rml_context_)
            rml_manager_->destroyContext("workspace_chrome");
        rml_context_ = nullptr;
        document_ = nullptr;
        rml_manager_ = nullptr;
        has_theme_signature_ = false;
        last_markup_.clear();
    }

    void RmlWorkspaceChrome::reloadResources() {
        if (!rml_context_ || !rml_manager_)
            return;
        rml_manager_->releaseCachedVulkanContext(direct_cache_);
        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }
        document_ = nullptr;
        has_theme_signature_ = false;
        last_markup_.clear();
        try {
            document_ = rml_documents::loadDocument(
                rml_context_, lfs::vis::getAssetPath("rmlui/workspace_chrome.rml"));
            if (document_)
                document_->Show();
            else
                LOG_ERROR("RmlWorkspaceChrome: failed to reload workspace_chrome.rml");
            updateTheme();
        } catch (const std::exception& error) {
            LOG_ERROR("RmlWorkspaceChrome: resource reload failed: {}", error.what());
        }
    }

    bool RmlWorkspaceChrome::updateTheme() {
        if (!document_)
            return false;
        const auto signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && signature == last_theme_signature_)
            return false;
        last_theme_signature_ = signature;
        has_theme_signature_ = true;
        const auto base = rml_theme::loadBaseRCSS("rmlui/workspace_chrome.rcss");
        const auto theme = rml_theme::loadBaseRCSS("rmlui/workspace_chrome.theme.rcss");
        rml_theme::applyTheme(document_, base, theme);
        return true;
    }

    std::string RmlWorkspaceChrome::makeMarkup(
        const WorkspaceFrameSnapshot& snapshot,
        const std::optional<CornerSplitPreview>& preview) const {
        std::string grips;
        std::string splitters;
        grips.reserve(snapshot.areas.size() * 260);
        splitters.reserve(snapshot.splitters.size() * 100);
        for (const auto& area : snapshot.areas)
            grips += gripMarkup(area, snapshot.outer);
        for (const auto& splitter : snapshot.splitters)
            splitters += splitterMarkup(splitter.rect, snapshot.outer, "workspace-splitter");

        std::string preview_markup;
        if (preview) {
            preview_markup += splitterMarkup(preview->first, snapshot.outer,
                                             "workspace-preview-first");
            preview_markup += splitterMarkup(preview->second, snapshot.outer,
                                             "workspace-preview-second");
            preview_markup += splitterMarkup(preview->splitter, snapshot.outer,
                                             "workspace-preview-splitter");
        }
        return std::format(R"(<div id="workspace-grips">{}</div><div id="workspace-splitters">{}</div><div id="workspace-preview">{}</div>)",
                           grips, splitters, preview_markup);
    }

    void RmlWorkspaceChrome::render(const WorkspaceFrameSnapshot& snapshot,
                                    const std::optional<CornerSplitPreview> preview,
                                    const bool ui_hidden) {
        if (ui_hidden || !rml_manager_ || !rml_context_ || !document_ ||
            !rml_manager_->getVulkanRenderInterface())
            return;
        const auto& outer = snapshot.outer;
        if (outer.empty())
            return;
        const bool theme_changed = updateTheme();
        const std::string markup = makeMarkup(snapshot, preview);
        const bool markup_changed = markup != last_markup_;
        const int width = std::max(1, outer.width);
        const int height = std::max(1, outer.height);
        if (markup_changed) {
            if (auto* const root = document_->GetElementById("workspace-chrome"))
                root->SetInnerRML(markup);
            last_markup_ = markup;
        }
        rml_context_->SetDimensions(Rml::Vector2i(width, height));
        const bool refresh = theme_changed || markup_changed || direct_cache_.texture == 0 ||
                             direct_cache_.width != width || direct_cache_.height != height;
        if (refresh)
            rml_context_->Update();
        rml_manager_->queueCachedVulkanContext({
            .context = rml_context_,
            .cache = &direct_cache_,
            .cache_width = width,
            .cache_height = height,
            .offset_x = static_cast<float>(outer.x),
            .offset_y = static_cast<float>(outer.y),
            .draw_width = static_cast<float>(width),
            .draw_height = static_cast<float>(height),
            .refresh = refresh,
            .foreground = true,
            .clip_enabled = false,
            .clip = {},
        });
    }

} // namespace lfs::vis::gui
